"""AE13K — Clean Forward Market Feed (provider-verified pairs only).

Builds a NEW forward feed from DexScreener. Does not read or mutate old
historical / training data. Paper/demo/research display only — not Live Market.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from app.ae13b_product.dexscreener_pair_verify import (
    SOURCE_PROVIDER,
    get_pair_verify_limiter,
    normalize_chain_id,
    validate_dexscreener_pair,
)
from app.dexscreener import get_trending_pairs_sync

DEFAULT_MAX_ROWS_PER_BASE_TOKEN = 1
DEFAULT_MAX_ROWS_PER_SYMBOL = 1
MAX_ROWS_PER_PAIR_ADDRESS = 1  # hard limit

#: Extra search terms to surface multi-pool bases (e.g. WIF) for diversity proof.
CLEAN_FEED_QUERIES = [
    "bonk",
    "pepe",
    "doge",
    "shib",
    "wif",
    "pump",
    "meme",
    "sol",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_addr(v: Any) -> str:
    return str(v or "").strip()


def _norm_addr_key(v: Any) -> str:
    return _norm_addr(v).lower()


def _as_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _volume_24h(volume: Any) -> float | None:
    if isinstance(volume, dict):
        return _as_float(volume.get("h24"))
    return _as_float(volume)


def _rank_score(pair: dict[str, Any]) -> float:
    liq = _as_float(
        (pair.get("liquidity") or {}).get("usd")
        if isinstance(pair.get("liquidity"), dict)
        else pair.get("liquidity")
    )
    vol = _volume_24h(pair.get("volume"))
    return float(liq or 0) + float(vol or 0) * 0.01


def candidate_from_search_pair(raw: dict[str, Any]) -> dict[str, Any]:
    """Lightweight candidate before provider pair lookup verification.

    Intentionally does NOT construct DexScreener chart URLs here.
    """
    chain = normalize_chain_id(raw.get("chainId"))
    pair = _norm_addr(raw.get("pairAddress"))
    base = raw.get("baseToken") if isinstance(raw.get("baseToken"), dict) else {}
    quote = raw.get("quoteToken") if isinstance(raw.get("quoteToken"), dict) else {}
    return {
        "source_provider": SOURCE_PROVIDER,
        "chain_id": chain or None,
        "normalized_chain_id": chain or None,
        "pair_address": pair or None,
        "provider_pair_id": pair or None,
        # No provider_pair_url until verified
        "provider_pair_url": None,
        "base_symbol": base.get("symbol"),
        "quote_symbol": quote.get("symbol"),
        "search_price_usd": raw.get("priceUsd"),
        "search_liquidity_usd": (raw.get("liquidity") or {}).get("usd")
        if isinstance(raw.get("liquidity"), dict)
        else None,
        "search_volume_24h": _volume_24h(raw.get("volume")),
        "_rank": _rank_score(raw),
    }


def verify_provider_pair(
    *,
    chain_id: str | None,
    pair_address: str | None,
    expected_url: str | None = None,  # noqa: ARG001 — kept for API compat; unused
    use_cache: bool = True,
) -> dict[str, Any]:
    """Verify via GET /latest/dex/pairs/{chainId}/{pairId} with rate-limit safety.

    Returns a verification record. Clean-feed admission requires clean_feed_eligible.
    """
    result = validate_dexscreener_pair(
        str(chain_id or ""),
        str(pair_address or ""),
        use_cache=use_cache,
    )
    d = result.to_dict(include_raw=True)
    # Backward-compatible aliases used by proof script / UI
    d["status"] = d.get("verification_status")
    d["payload_hash"] = d.get("provider_payload_hash")
    d["chain_id"] = d.get("normalized_chain_id")
    d["reject_reason"] = d.get("exclusion_reason")
    d["observed_at"] = d.get("fetched_at")
    # price_change aliases for UI
    d["price_change_m5"] = d.get("price_change_5m")
    d["price_change_h1"] = d.get("price_change_1h")
    d["price_change_h6"] = d.get("price_change_6h")
    d["price_change_h24"] = d.get("price_change_24h")
    d["txns_24h"] = {
        "buys": d.get("txns_24h_buys"),
        "sells": d.get("txns_24h_sells"),
        "total": (
            int(d.get("txns_24h_buys") or 0) + int(d.get("txns_24h_sells") or 0)
            if d.get("txns_24h_buys") is not None or d.get("txns_24h_sells") is not None
            else None
        ),
    }
    if d.get("clean_feed_eligible"):
        d["address_role_note"] = (
            "Displayed address is the DexScreener pair/pool id, not the token mint/contract. "
            "Base and quote token addresses are shown separately."
        )
        d["label"] = f"Fresh: fetched recently" if d.get("freshness_status") == "fresh" else d.get(
            "freshness_status"
        )
    return d


def _txns_total(row: dict[str, Any]) -> int:
    buys = row.get("txns_24h_buys")
    sells = row.get("txns_24h_sells")
    if buys is None and sells is None:
        tx = row.get("txns_24h")
        if isinstance(tx, dict) and tx.get("total") is not None:
            return int(tx["total"])
        return 0
    return int(buys or 0) + int(sells or 0)


def _apply_diversity(
    verified: list[dict[str, Any]],
    *,
    max_rows_per_base_token: int,
    max_rows_per_symbol: int,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (main_rows, alternative_pools, suppression_events)."""
    ordered = sorted(
        verified,
        key=lambda r: (
            float(_as_float(r.get("liquidity_usd")) or 0),
            float(_as_float(r.get("volume_24h")) or 0),
            str(r.get("fetched_at") or ""),
            _txns_total(r),
        ),
        reverse=True,
    )

    main: list[dict[str, Any]] = []
    alts: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()
    base_counts: dict[str, int] = {}
    symbol_counts: dict[str, int] = {}

    for row in ordered:
        pair_key = _norm_addr_key(row.get("pair_address"))
        base_key = _norm_addr_key(row.get("base_token_address"))
        sym_key = str(row.get("base_token_symbol") or "").strip().upper()
        dup_group = base_key or sym_key or pair_key

        if pair_key in seen_pairs:
            events.append(
                {
                    "pair_address": row.get("pair_address"),
                    "base_token_address": row.get("base_token_address"),
                    "base_token_symbol": row.get("base_token_symbol"),
                    "reason": "duplicate_pair_address_hard_limit",
                    "action": "suppressed",
                    "duplicate_group_id": dup_group,
                }
            )
            continue

        seen_pairs.add(pair_key)

        suppress_main = False
        reason = None
        if base_key and base_counts.get(base_key, 0) >= max_rows_per_base_token:
            suppress_main = True
            reason = "max_rows_per_base_token"
        elif sym_key and symbol_counts.get(sym_key, 0) >= max_rows_per_symbol:
            suppress_main = True
            reason = "max_rows_per_symbol"

        if suppress_main:
            alt = dict(row)
            alt["feed_section"] = "alternative_pools"
            alt["suppressed_from_main_reason"] = reason
            alt["duplicate_suppressed"] = True
            alt["duplicate_group_id"] = dup_group
            alt["clean_feed_eligible"] = True  # verified, but not main-displayed
            primary = next(
                (
                    m
                    for m in main
                    if _norm_addr_key(m.get("base_token_address")) == base_key
                    or str(m.get("base_token_symbol") or "").strip().upper() == sym_key
                ),
                None,
            )
            alt["alternative_for_pair_address"] = (primary or {}).get("pair_address")
            alt["alternative_for_base_token"] = base_key or sym_key
            alts.append(alt)
            events.append(
                {
                    "pair_address": row.get("pair_address"),
                    "base_token_address": row.get("base_token_address"),
                    "base_token_symbol": row.get("base_token_symbol"),
                    "liquidity_usd": row.get("liquidity_usd"),
                    "volume_24h": row.get("volume_24h"),
                    "reason": reason,
                    "action": "moved_to_alternative_pools",
                    "kept_main_pair_address": (primary or {}).get("pair_address"),
                    "duplicate_group_id": dup_group,
                }
            )
            continue

        if len(main) >= limit:
            events.append(
                {
                    "pair_address": row.get("pair_address"),
                    "base_token_address": row.get("base_token_address"),
                    "base_token_symbol": row.get("base_token_symbol"),
                    "reason": "main_feed_limit",
                    "action": "omitted",
                    "duplicate_group_id": dup_group,
                }
            )
            continue

        display = dict(row)
        display["feed_section"] = "main"
        display["suppressed_from_main_reason"] = None
        display["duplicate_suppressed"] = False
        display["duplicate_group_id"] = dup_group
        main.append(display)
        if base_key:
            base_counts[base_key] = base_counts.get(base_key, 0) + 1
        if sym_key:
            symbol_counts[sym_key] = symbol_counts.get(sym_key, 0) + 1

    # Annotate alternative_pool_count on main rows
    alt_counts: dict[str, int] = {}
    for a in alts:
        gk = a.get("duplicate_group_id") or ""
        alt_counts[gk] = alt_counts.get(gk, 0) + 1
    for m in main:
        m["alternative_pool_count"] = alt_counts.get(m.get("duplicate_group_id") or "", 0)
    for a in alts:
        a["alternative_pool_count"] = alt_counts.get(a.get("duplicate_group_id") or "", 0)

    return main, alts, events


def _row_from_verification(v: dict[str, Any]) -> dict[str, Any]:
    """Build CleanForwardMarketRow display dict from verification result."""
    chain = v.get("normalized_chain_id") or v.get("chain_id")
    pair = v.get("pair_address")
    row_id = f"{str(chain).lower()}|pair|{pair}"
    return {
        "row_id": row_id,
        "row_key": row_id,
        "source_provider": SOURCE_PROVIDER,
        "normalized_chain_id": chain,
        "chain": chain,
        "chain_id": chain,
        "dex_id": v.get("dex_id"),
        "dex": v.get("dex_id"),
        "provider_pair_id": v.get("provider_pair_id") or pair,
        "pair_address": pair,
        "provider_pair_url": v.get("provider_pair_url"),
        "provider_pair_url_source": v.get("provider_pair_url_source"),
        "dexscreener_url": v.get("provider_pair_url"),
        "address_role": v.get("address_role"),
        "address_role_note": v.get("address_role_note"),
        "address_role_label": (
            "Pair / Pool address"
            if v.get("address_role") in ("pool_address", "pair_contract", "provider_pair_id")
            else str(v.get("address_role") or "Pair / Pool address")
        ),
        "base_token_address": v.get("base_token_address"),
        "base_token_symbol": v.get("base_token_symbol"),
        "base_token_name": v.get("base_token_name"),
        "base_token_address_label": "Base token address",
        "quote_token_address": v.get("quote_token_address"),
        "quote_token_symbol": v.get("quote_token_symbol"),
        "quote_token_name": v.get("quote_token_name"),
        "quote_token_address_label": "Quote token address",
        "pair": v.get("pair_label"),
        "pair_label": v.get("pair_label"),
        "price": v.get("price_usd"),
        "price_usd": v.get("price_usd"),
        "liquidity": v.get("liquidity_usd"),
        "liquidity_usd": v.get("liquidity_usd"),
        "volume_5m": v.get("volume_5m"),
        "volume_1h": v.get("volume_1h"),
        "volume_6h": v.get("volume_6h"),
        "volume_24h": v.get("volume_24h"),
        "volume": v.get("volume"),
        "txns_24h_buys": v.get("txns_24h_buys"),
        "txns_24h_sells": v.get("txns_24h_sells"),
        "txns_24h": v.get("txns_24h"),
        "txns": v.get("txns"),
        "price_change_5m": v.get("price_change_5m"),
        "price_change_1h": v.get("price_change_1h"),
        "price_change_6h": v.get("price_change_6h"),
        "price_change_24h": v.get("price_change_24h"),
        "price_change_m5": v.get("price_change_5m"),
        "price_change_h1": v.get("price_change_1h"),
        "price_change_h6": v.get("price_change_6h"),
        "price_change_h24": v.get("price_change_24h"),
        "price_change": v.get("price_change"),
        "pair_created_at": v.get("pair_created_at"),
        "fetched_at": v.get("fetched_at"),
        "observed_at": v.get("observed_at") or v.get("fetched_at"),
        "ingested_at": v.get("ingested_at") or v.get("fetched_at"),
        "last_fetched": v.get("fetched_at"),
        "provider_payload_hash": v.get("provider_payload_hash") or v.get("payload_hash"),
        "payload_hash": v.get("provider_payload_hash") or v.get("payload_hash"),
        "verification_status": v.get("verification_status") or v.get("status"),
        "freshness_status": v.get("freshness_status"),
        "freshness_label": v.get("label") or v.get("freshness_status"),
        "tradability_status": v.get("tradability_status"),
        "identity_status": v.get("identity_status"),
        "status": v.get("verification_status") or v.get("status"),
        "clean_feed_eligible": bool(v.get("clean_feed_eligible")),
        "exclusion_reason": v.get("exclusion_reason") or v.get("reject_reason"),
        "duplicate_group_id": None,
        "duplicate_suppressed": False,
        "alternative_pool_count": 0,
        "shown_as_token_contract": False,
        "paper_demo_only": True,
        "live_trading_ready": False,
        "verification_attempted_at": v.get("verification_attempted_at"),
        "verification_http_status": v.get("verification_http_status"),
        "verification_error": v.get("verification_error"),
        "verification_cache_hit": v.get("verification_cache_hit"),
        "verification_retry_after_seconds": v.get("verification_retry_after_seconds"),
        "verification_attempt_count": v.get("verification_attempt_count"),
    }


# In-process baseline for consecutive refresh comparisons when the client
# does not supply previous_rows (first poll still reports no baseline).
_LAST_MAIN_FEED_ROWS: list[dict[str, Any]] | None = None

_PROVIDER_VALUE_COMPARE_FIELDS = (
    "price_usd",
    "liquidity_usd",
    "volume_24h",
    "txns_24h_buys",
    "txns_24h_sells",
    "price_change_5m",
    "price_change_1h",
    "price_change_6h",
    "price_change_24h",
)


def _row_key_for_compare(row: dict[str, Any]) -> str:
    chain = normalize_chain_id(
        row.get("normalized_chain_id") or row.get("chain_id") or row.get("chain")
    )
    pair = _norm_addr_key(row.get("pair_address") or row.get("provider_pair_id"))
    if chain and pair:
        return f"{chain.lower()}|{pair}"
    return pair or str(row.get("row_id") or row.get("row_key") or "").lower()


def _compare_field_value(field: str, left: Any, right: Any) -> bool:
    """Return True when normalized field values differ."""
    if field in _PROVIDER_VALUE_COMPARE_FIELDS:
        left_num = _as_float(left)
        right_num = _as_float(right)
        if left_num is not None or right_num is not None:
            if left_num is None or right_num is None:
                return left_num != right_num
            return left_num != right_num
    return str(left or "") != str(right or "")


def _provider_values_changed(prev: dict[str, Any], row: dict[str, Any]) -> bool:
    return any(
        _compare_field_value(field, prev.get(field), row.get(field))
        for field in _PROVIDER_VALUE_COMPARE_FIELDS
    )


def _payload_hash_for_row(row: dict[str, Any]) -> str:
    return str(row.get("provider_payload_hash") or row.get("payload_hash") or "")


def _compute_refresh_change_counters(
    current_rows: list[dict[str, Any]],
    *,
    previous_rows: list[dict[str, Any]] | None,
    provider_refetch_completed: bool,
) -> dict[str, Any]:
    baseline_rows = previous_rows or []
    comparison_baseline_available = bool(baseline_rows)

    if not comparison_baseline_available:
        return {
            "comparison_baseline_available": False,
            "provider_values_changed_count": 0,
            "payload_hash_changed_count": 0,
            "provider_unchanged_but_refetched_count": 0,
            "rows_entered_main_feed": len(current_rows),
            "rows_exited_main_feed": 0,
        }

    prev_map = {
        _row_key_for_compare(row): row
        for row in baseline_rows
        if _row_key_for_compare(row)
    }
    cur_map = {
        _row_key_for_compare(row): row
        for row in current_rows
        if _row_key_for_compare(row)
    }

    provider_values_changed = 0
    payload_hash_changed = 0
    provider_unchanged_refetched = 0

    for key, row in cur_map.items():
        prev = prev_map.get(key)
        if not prev:
            continue
        cur_hash = _payload_hash_for_row(row)
        prev_hash = _payload_hash_for_row(prev)
        values_changed = _provider_values_changed(prev, row)
        hash_changed = cur_hash != prev_hash
        if values_changed:
            provider_values_changed += 1
        if hash_changed:
            payload_hash_changed += 1
        if (
            provider_refetch_completed
            and not values_changed
            and not hash_changed
        ):
            provider_unchanged_refetched += 1

    return {
        "comparison_baseline_available": True,
        "provider_values_changed_count": provider_values_changed,
        "payload_hash_changed_count": payload_hash_changed,
        "provider_unchanged_but_refetched_count": provider_unchanged_refetched,
        "rows_entered_main_feed": len(set(cur_map) - set(prev_map)),
        "rows_exited_main_feed": len(set(prev_map) - set(cur_map)),
    }


def _compute_refresh_metadata(
    feed: dict[str, Any],
    *,
    refresh_id: str,
    refresh_started_at: str,
    refresh_completed_at: str,
    force: bool,
    clear_cache: bool,
    previous_rows: list[dict[str, Any]] | None,
    limiter_stats_before: dict[str, Any],
    limiter_stats_after: dict[str, Any],
) -> dict[str, Any]:
    """Build refresh metadata block for UI / audit."""
    limiter = get_pair_verify_limiter()
    verifications = feed.get("verifications") or []
    cache_hits = sum(1 for v in verifications if v.get("verification_cache_hit"))
    cache_misses = sum(1 for v in verifications if not v.get("verification_cache_hit"))
    total_verify = len(verifications)

    ages: list[float] = []
    for v in verifications:
        if v.get("verification_cache_hit"):
            chain = v.get("normalized_chain_id") or v.get("chain_id")
            pair = v.get("pair_address") or v.get("requested_pair_address")
            if chain and pair:
                age = limiter.cache_age_seconds(str(chain), str(pair))
                if age is not None:
                    ages.append(age)

    stats = feed.get("stats") or {}
    rate_limited = int(stats.get("provider_rate_limited_count") or 0)
    deferred = int(stats.get("verification_deferred_count") or 0)
    ttl = float(
        (limiter.settings_snapshot() or {}).get("DEXSCREENER_PAIR_VERIFY_CACHE_TTL_SECONDS", 20)
    )

    if rate_limited > 0:
        refresh_mode = "provider_rate_limited"
    elif deferred > 0 and not (feed.get("rows") or []):
        refresh_mode = "verification_deferred"
    elif total_verify == 0:
        refresh_mode = "provider_refetch"
    elif cache_hits == total_verify and cache_hits > 0:
        refresh_mode = "cache_hit"
    elif cache_hits > 0 and cache_misses > 0:
        refresh_mode = "mixed_cache_and_provider_refetch"
    elif cache_misses > 0:
        refresh_mode = "provider_refetch"
    else:
        refresh_mode = "provider_refetch"

    http_delta = int(limiter_stats_after.get("http_calls", 0)) - int(
        limiter_stats_before.get("http_calls", 0)
    )
    provider_refetch_completed = cache_misses > 0 or http_delta > 0
    change_counters = _compute_refresh_change_counters(
        feed.get("rows") or [],
        previous_rows=previous_rows,
        provider_refetch_completed=provider_refetch_completed,
    )
    provider_values_changed = int(change_counters["provider_values_changed_count"])
    payload_hash_changed = int(change_counters["payload_hash_changed_count"])
    provider_unchanged_refetched = int(
        change_counters["provider_unchanged_but_refetched_count"]
    )

    latest_fetch = None
    for row in feed.get("rows") or []:
        fa = row.get("fetched_at")
        if fa and (latest_fetch is None or str(fa) > str(latest_fetch)):
            latest_fetch = fa

    return {
        "ok": True,
        "refresh_id": refresh_id,
        "refresh_started_at": refresh_started_at,
        "refresh_completed_at": refresh_completed_at,
        "refresh_mode": refresh_mode,
        "provider_refetch_attempted": bool(force or clear_cache or cache_misses > 0 or http_delta > 0),
        "provider_refetch_completed": provider_refetch_completed,
        "comparison_baseline_available": change_counters["comparison_baseline_available"],
        "cache_hit_count": cache_hits,
        "cache_miss_count": cache_misses,
        "cache_age_seconds_min": min(ages) if ages else None,
        "cache_age_seconds_max": max(ages) if ages else None,
        "cache_ttl_seconds": ttl,
        "force_refresh_supported": True,
        "force_refresh_used": bool(force),
        "clear_cache_used": bool(clear_cache),
        "provider_values_changed_count": provider_values_changed,
        "payload_hash_changed_count": payload_hash_changed,
        "provider_unchanged_but_refetched_count": provider_unchanged_refetched,
        "rows_entered_main_feed": change_counters["rows_entered_main_feed"],
        "rows_exited_main_feed": change_counters["rows_exited_main_feed"],
        "verification_deferred_count": deferred,
        "provider_rate_limited_count": rate_limited,
        "clean_rows_displayed": stats.get("clean_rows_displayed"),
        "duplicate_pools_suppressed": stats.get("duplicate_pools_suppressed"),
        "invalid_or_unresolved_excluded": stats.get("invalid_or_unresolved_addresses"),
        "latest_provider_fetch_at": latest_fetch,
        "rendered_at": refresh_completed_at,
        "verifier_used": SOURCE_PROVIDER,
        "rate_limit_protection_used": True,
        "chain_aware_url_validation_used": True,
        "http_calls_this_refresh": http_delta,
        "ui_message": _refresh_ui_message(
            refresh_mode=refresh_mode,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            provider_values_changed=provider_values_changed,
            provider_unchanged_refetched=provider_unchanged_refetched,
            rate_limited=rate_limited,
            deferred=deferred,
            force=force,
            comparison_baseline_available=change_counters["comparison_baseline_available"],
        ),
    }


def _refresh_ui_message(
    *,
    refresh_mode: str,
    cache_hits: int,
    cache_misses: int,
    provider_values_changed: int,
    provider_unchanged_refetched: int,
    rate_limited: int,
    deferred: int,
    force: bool,
    comparison_baseline_available: bool = True,
) -> str:
    if rate_limited > 0 or refresh_mode == "provider_rate_limited":
        return (
            "Provider verification deferred due to rate limit. Not tradable until verified."
        )
    if deferred > 0 and refresh_mode == "verification_deferred":
        return "Provider verification deferred. Row is not tradable until verified."
    if refresh_mode == "cache_hit" and cache_misses == 0:
        return "Using cached provider verification. No provider refetch performed."
    if not comparison_baseline_available:
        if cache_misses > 0:
            return "Provider refetch completed. No prior main-feed baseline for change comparison."
        return "Provider refetch completed."
    if provider_values_changed > 0:
        return f"Provider refetched; market values changed for {provider_values_changed} rows."
    if provider_unchanged_refetched > 0 or cache_misses > 0:
        return "Provider refetched; no market value changes observed."
    if force and cache_hits > 0 and cache_misses == 0:
        return "Using cached provider verification. No provider refetch performed."
    return "Provider refetch completed."


def reset_clean_forward_refresh_baseline() -> None:
    """Clear in-process main-feed baseline (tests / isolated refresh runs)."""
    global _LAST_MAIN_FEED_ROWS
    _LAST_MAIN_FEED_ROWS = None


def get_cached_clean_forward_rows() -> list[dict[str, Any]]:
    """Return the last in-process Clean Forward main-feed rows (no network).

    Used by AE14 demo_queue / demo_bot bridge lookup. Empty when the feed has
    not been built/refreshed in this process yet.
    """
    return [dict(row) for row in (_LAST_MAIN_FEED_ROWS or [])]


def set_cached_clean_forward_rows(rows: list[dict[str, Any]] | None) -> None:
    """Test/smoke helper: seed the in-process Clean Forward cache without HTTP."""
    global _LAST_MAIN_FEED_ROWS
    if rows is None:
        _LAST_MAIN_FEED_ROWS = None
        return
    _LAST_MAIN_FEED_ROWS = [dict(row) for row in rows if isinstance(row, dict)]


def refresh_clean_forward_market_feed(
    *,
    force: bool = False,
    clear_cache: bool = False,
    previous_rows: list[dict[str, Any]] | None = None,
    limit: int = 25,
    max_candidates: int = 80,
    max_rows_per_base_token: int = DEFAULT_MAX_ROWS_PER_BASE_TOKEN,
    max_rows_per_symbol: int = DEFAULT_MAX_ROWS_PER_SYMBOL,
    max_verify: int = 40,
    queries: list[str] | None = None,
) -> dict[str, Any]:
    """Explicit refresh path for Clean Forward Market Feed UI.

    force=True bypasses pair-verify TTL cache but still respects rate limits.
    clear_cache=True clears in-process verify cache before building (Force Provider Refresh).
    """
    refresh_id = uuid.uuid4().hex[:12]
    refresh_started_at = _utc_now()
    limiter = get_pair_verify_limiter()
    stats_before = limiter.stats_snapshot()

    if clear_cache:
        limiter.clear_cache()

    use_cache = not (force or clear_cache)

    feed = build_clean_forward_market_feed(
        limit=limit,
        max_candidates=max_candidates,
        max_rows_per_base_token=max_rows_per_base_token,
        max_rows_per_symbol=max_rows_per_symbol,
        max_verify=max_verify,
        queries=queries,
        use_cache=use_cache,
    )

    refresh_completed_at = _utc_now()
    stats_after = limiter.stats_snapshot()

    global _LAST_MAIN_FEED_ROWS
    baseline_rows = (
        previous_rows
        if previous_rows is not None
        else _LAST_MAIN_FEED_ROWS
    )

    refresh_meta = _compute_refresh_metadata(
        feed,
        refresh_id=refresh_id,
        refresh_started_at=refresh_started_at,
        refresh_completed_at=refresh_completed_at,
        force=force,
        clear_cache=clear_cache,
        previous_rows=baseline_rows,
        limiter_stats_before=stats_before,
        limiter_stats_after=stats_after,
    )

    _LAST_MAIN_FEED_ROWS = [dict(row) for row in (feed.get("rows") or [])]

    feed["refresh"] = refresh_meta
    feed["refresh_metadata"] = refresh_meta
    feed["rendered_at"] = refresh_completed_at
    feed["refresh_hint"] = refresh_meta.get("ui_message")
    return feed


def build_clean_forward_market_feed(
    *,
    limit: int = 25,
    max_candidates: int = 80,
    max_rows_per_base_token: int = DEFAULT_MAX_ROWS_PER_BASE_TOKEN,
    max_rows_per_symbol: int = DEFAULT_MAX_ROWS_PER_SYMBOL,
    max_verify: int = 40,
    queries: list[str] | None = None,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Collect + verify a clean forward feed from DexScreener only.

    Does not touch trader.db historical snapshots or training datasets.

    AE16D: when CLEAN_FORWARD_USE_CURATED_TARGETS is enabled and the curated
    ready file is loadable, use the curated exact-pair overlay instead of
    trending/search discovery. Flag default is OFF (existing behavior).
    Lazy-imported so module import never fetches or reads curated files.
    """
    # --- AE16D curated overlay (feature-flagged, default OFF) ---
    try:
        from app.clean_forward.curated_overlay import try_curated_overlay_or_none

        curated_feed = try_curated_overlay_or_none(limit=limit, use_cache=use_cache)
        if curated_feed is not None:
            return curated_feed
    except Exception:
        # Never break the existing collector if overlay import/path fails.
        pass
    # --- existing search-based Clean Forward path (unchanged) ---
    built_at = _utc_now()
    limiter = get_pair_verify_limiter()
    raw_pairs = get_trending_pairs_sync(
        max_pairs=max_candidates,
        queries=queries or CLEAN_FEED_QUERIES,
    )
    candidates = [candidate_from_search_pair(p) for p in raw_pairs]

    verifications: list[dict[str, Any]] = []
    invalid: list[dict[str, Any]] = []
    verified_rows: list[dict[str, Any]] = []
    seen_candidate_pairs: set[str] = set()
    rate_limited_count = 0
    deferred_count = 0

    for cand in candidates:
        pair = _norm_addr(cand.get("pair_address"))
        chain = _norm_addr(cand.get("normalized_chain_id") or cand.get("chain_id"))
        pkey = f"{chain.lower()}|{pair.lower()}"
        if not pair or pkey in seen_candidate_pairs:
            if pair and pkey in seen_candidate_pairs:
                invalid.append(
                    {
                        "pair_address": pair,
                        "chain_id": chain,
                        "reason": "duplicate_candidate_pair",
                        "status": "duplicate_candidate",
                        "verification_status": "duplicate_candidate",
                    }
                )
            continue
        seen_candidate_pairs.add(pkey)
        if len(verifications) >= max_verify:
            invalid.append(
                {
                    "pair_address": pair,
                    "chain_id": chain,
                    "reason": "verify_budget_exceeded",
                    "status": "not_verified",
                    "verification_status": "not_verified",
                }
            )
            continue

        v = verify_provider_pair(
            chain_id=chain,
            pair_address=pair,
            use_cache=use_cache,
        )
        raw = v.pop("raw_pair", None)
        verifications.append({**{k: val for k, val in v.items() if k != "raw_pair"}, "had_raw_pair": bool(raw)})

        status = str(v.get("verification_status") or v.get("status") or "")
        if status in ("provider_rate_limited",):
            rate_limited_count += 1
            deferred_count += 1
        elif status in ("provider_unavailable",) or v.get("tradability_status") == "verification_deferred":
            deferred_count += 1

        if not v.get("clean_feed_eligible") or not v.get("lookup_ok"):
            invalid.append(
                {
                    "pair_address": pair,
                    "chain_id": chain,
                    "provider_pair_url": v.get("provider_pair_url"),
                    "reason": v.get("exclusion_reason") or v.get("reject_reason") or status,
                    "status": status,
                    "verification_status": status,
                    "tradability_status": v.get("tradability_status"),
                    "verification_http_status": v.get("verification_http_status"),
                    "verification_cache_hit": v.get("verification_cache_hit"),
                }
            )
            continue

        # Hard gate: never admit without verified URL
        if not v.get("provider_pair_url") or not v.get("provider_pair_url_source"):
            invalid.append(
                {
                    "pair_address": pair,
                    "chain_id": chain,
                    "reason": "provider_pair_url_missing_after_verify",
                    "status": "provider_pair_incomplete",
                    "verification_status": "provider_pair_incomplete",
                    "tradability_status": "not_tradable_without_provider_pair",
                }
            )
            continue

        verified_rows.append(_row_from_verification(v))

    main, alts, suppress_events = _apply_diversity(
        verified_rows,
        max_rows_per_base_token=max_rows_per_base_token,
        max_rows_per_symbol=max_rows_per_symbol,
        limit=limit,
    )

    unique_bases = {
        _norm_addr_key(r.get("base_token_address"))
        for r in main
        if r.get("base_token_address")
    }
    unique_pairs = {
        _norm_addr_key(r.get("pair_address")) for r in main if r.get("pair_address")
    }

    stats = {
        "total_candidates_seen": len(candidates),
        "valid_provider_pairs": len(verified_rows),
        "unique_base_tokens": len(unique_bases),
        "unique_pair_addresses": len(unique_pairs),
        "duplicate_pools_suppressed": len(
            [e for e in suppress_events if e.get("action") == "moved_to_alternative_pools"]
        ),
        "invalid_or_unresolved_addresses": len(invalid),
        "clean_rows_displayed": len(main),
        "alternative_pools_count": len(alts),
        "verifications_attempted": len(verifications),
        "provider_rate_limited_count": rate_limited_count,
        "verification_deferred_count": deferred_count,
    }

    limiter_stats = limiter.stats_snapshot()

    user_message = ""
    if not main:
        user_message = "No clean provider-verified market rows available yet."
    if deferred_count and not main:
        user_message = (
            "Provider verification deferred due to rate limit. Not tradable until verified."
            if rate_limited_count
            else "Provider verification deferred. Row is not tradable until verified."
        )

    return {
        "ok": True,
        "status": "ready" if main else "empty",
        "panel_title": "Clean Forward Market Feed",
        "not_live_market": True,
        "ui_label": "Clean Forward Market Feed",
        "legacy_panel_label": "Market Snapshot Feed",
        "demo_mode_badge": "LIVE DISABLED / DEMO ONLY",
        "paper_demo_only": True,
        "wallet_configured": False,
        "live_trading_ready": False,
        "training_run": False,
        "backtest_run": False,
        "ae14_run": False,
        "paper_positions_opened_from_clean_feed": 0,
        "live_trading_enabled": False,
        "source_provider": SOURCE_PROVIDER,
        "built_at_utc": built_at,
        "warning": (
            "Rows represent verified provider pairs. Pair/pool addresses are not token contracts. "
            "Trading eligibility requires fresh price, fresh liquidity, and validated tradability."
        ),
        "diversity_controls": {
            "max_rows_per_base_token": max_rows_per_base_token,
            "max_rows_per_symbol": max_rows_per_symbol,
            "max_rows_per_pair_address": MAX_ROWS_PER_PAIR_ADDRESS,
            "prefer": "highest_liquidity_then_volume_24h_then_freshest_then_txns",
        },
        "rate_limit_controls": limiter.settings_snapshot() if hasattr(limiter, "settings_snapshot") else {},
        "rate_limit_stats": limiter_stats,
        "stats": stats,
        "rows": main,
        "alternative_pools": alts,
        "suppression_events": suppress_events,
        "invalid_or_unresolved": invalid,
        "verifications": verifications,
        "refresh_hint": None,
        "user_message": user_message,
        "count": len(main),
    }


def classify_refresh(
    prev: dict[str, Any] | None,
    curr: dict[str, Any] | None,
    *,
    lookup_ok: bool,
    verification_status: str | None = None,
) -> str:
    status = str(verification_status or (curr or {}).get("verification_status") or "")
    if status == "provider_rate_limited":
        return "provider_rate_limited"
    if status in ("provider_unavailable",) or (
        (curr or {}).get("tradability_status") == "verification_deferred"
    ):
        return "verification_deferred"
    if not lookup_ok or not curr:
        return "provider_pair_not_found"
    if not prev:
        return "provider_updated"  # first observation
    changed = any(
        str(prev.get(k)) != str(curr.get(k))
        for k in (
            "price_usd",
            "liquidity_usd",
            "volume",
            "txns",
            "payload_hash",
            "provider_payload_hash",
        )
    )
    fetched_changed = str(prev.get("fetched_at")) != str(curr.get("fetched_at"))
    if changed:
        return "provider_updated"
    if fetched_changed:
        return "provider_unchanged_but_refetched"
    return "locally_refreshed_only"
