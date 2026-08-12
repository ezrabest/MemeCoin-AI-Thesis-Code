"""AE18 hot-path market feed builders — read runtime identity index only."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.clean_forward.display_identity import (
    SYMBOL_PAIR_UNAVAILABLE,
    UNAVAILABLE_STATUSES,
    derive_symbol_pair_display,
    is_symbol_pair_available,
)
from app.clean_forward.display_resilience import (
    RESILIENCE_FIELDS,
    apply_display_resilience,
)
from app.clean_forward.provider_resilience_statuses import is_proper_symbol_pair_display
from app.clean_forward.provider_url_key import try_normalize_provider_pair_url_key
from app.clean_forward.runtime_identity_index import (
    INDEX_MISSING_CODE,
    load_runtime_identity_index,
)


ACTIVITY_FIELDS = (
    "provider_fetch_at",
    "market_data_refreshed_at",
    "last_market_update_at",
    "price_updated_at",
    "display_status",
    "market_activity_status",
    "activity_trade_readiness_status",
    "activity_trade_block_reason",
    "market_activity_blocks_demo_entry",
    "provider_txns_observed_field_count",
    "provider_txns_recent_total",
    "provider_volume_observed_field_count",
    "provider_volume_recent_total",
    "provider_price_delta_observed_field_count",
    "provider_price_delta_any_nonzero",
    "market_activity_provenance",
    "activity_uses_symbol_display",
    "activity_uses_liquidity_or_market_cap_as_activity_proxy",
)

def _activity_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {f: row.get(f) for f in ACTIVITY_FIELDS if row.get(f) not in (None, "")}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v or "").strip().lower() in {"true", "1", "yes"}


def resolve_display_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Single central SYMBOL/PAIR resolver shared by every UI surface.

    Clean Forward, Market Snapshot, Market Opportunities and Portfolio all go
    through this function, so one canonical identity can never render two
    different symbol pairs. It re-derives from the runtime index row's provider
    symbol/address fields, which also self-heals rows written by older builds.
    """
    derived = derive_symbol_pair_display(row)
    stored = str(row.get("symbol_pair_display") or "").strip()
    if (
        not is_symbol_pair_available(derived["symbol_pair_display"])
        and is_symbol_pair_available(stored)
        and "/" in stored
    ):
        # Index carries a provider-verified pair the raw fields no longer show.
        derived = {
            **derived,
            "symbol_pair_display": stored,
            "symbol_pair_display_status": str(row.get("symbol_pair_display_status") or "FULL_PAIR"),
            "symbol_pair_display_reason": "",
        }
    derived.setdefault("symbol_pair_known_side_symbol", "")
    # Read-only resilience overlay (last-good / manual override). No cache writes.
    probe = {**row, **derived}
    apply_display_resilience(probe, allow_cache_lookup=True)
    for fld in RESILIENCE_FIELDS:
        if probe.get(fld) not in (None, ""):
            derived[fld] = probe[fld]
    return derived


# Backwards-compatible private alias used by audits/tests.
_display_fields = resolve_display_fields


def _social_fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "social_classification": row.get("social_classification") or "NON_SOCIAL_OR_UNCLASSIFIED",
        "is_social_candidate": _as_bool(row.get("is_social_candidate")),
        "is_social_confirmed": _as_bool(row.get("is_social_confirmed")),
        "social_source": row.get("social_source") or "",
        "social_reason": row.get("social_reason") or "",
        "linked_sources": row.get("linked_sources") or "",
        "seed_collection": row.get("seed_collection") or "",
        "manual_curation_status": row.get("manual_curation_status") or "",
    }


def _as_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _index_row_to_clean_forward(row: dict[str, Any]) -> dict[str, Any]:
    final_seg = row.get("provider_pair_url_final_segment_exact") or ""
    url = row.get("provider_pair_url_exact") or row.get("open_chart_url") or ""
    buys = row.get("txns_h24_buys")
    sells = row.get("txns_h24_sells")
    buys_f = _as_float(buys)
    sells_f = _as_float(sells)
    tx_total = None
    if buys_f is not None or sells_f is not None:
        tx_total = int((buys_f or 0) + (sells_f or 0))
    dex = row.get("provider_dex_id") or row.get("dex_id") or ""
    disp = _display_fields(row)
    sym_display = disp["symbol_pair_display"]
    return {
        **_social_fields(row),
        "canonical_market_identity": row.get("canonical_market_identity"),
        "canonical_market_identity_type": row.get("canonical_market_identity_type"),
        "provider_pair_url": url,
        "provider_pair_url_exact": url,
        "provider_pair_url_final_segment_exact": final_seg,
        "open_chart_url": url,
        "dexscreener_url": url,
        "market_url_id": final_seg,
        "market_url_id_label": "Market URL",
        "pair_address": row.get("pair_address_derived"),
        "pair_address_derived": row.get("pair_address_derived"),
        "pair_address_label": "DERIVED HELPER ID",
        "pair_address_for_rpc": row.get("pair_address_for_rpc"),
        "chain_id": row.get("chain"),
        "chain": row.get("chain"),
        "normalized_chain_id": row.get("chain"),
        "dex": dex or "",
        "dex_id": dex or "",
        "provider_dex_id": dex or "",
        "dex_display": dex or "unavailable",
        "base_symbol": disp["base_token_symbol"],
        "quote_symbol": disp["quote_token_symbol"],
        "base_token_symbol": disp["base_token_symbol"],
        "quote_token_symbol": disp["quote_token_symbol"],
        "symbol_pair_display": sym_display,
        "symbol_pair_display_status": disp["symbol_pair_display_status"],
        "symbol_pair_display_reason": disp["symbol_pair_display_reason"],
        "symbol_pair_address_fallback": disp["symbol_pair_address_fallback"],
        "symbol_pair_known_side_symbol": disp.get("symbol_pair_known_side_symbol", ""),
        "symbol_pair_available": is_symbol_pair_available(sym_display),
        "pair": sym_display,
        "pair_label": sym_display,
        "base_token_address": row.get("provider_base_token_address")
        or row.get("base_token_address_derived"),
        "quote_token_address": row.get("provider_quote_token_address")
        or row.get("quote_token_address_derived"),
        "price_usd": row.get("price_usd"),
        "price": row.get("price_usd"),
        "liquidity_usd": row.get("liquidity_usd"),
        "liquidity": row.get("liquidity_usd"),
        "volume_24h": row.get("volume_h24"),
        "volume_h24": row.get("volume_h24"),
        "txns_24h": {"buys": buys, "sells": sells, "total": tx_total},
        "txns_24h_buys": buys,
        "txns_24h_sells": sells,
        "price_change_m5": row.get("price_change_m5"),
        "price_change_h1": row.get("price_change_h1"),
        "price_change_h6": row.get("price_change_h6"),
        "price_change_h24": row.get("price_change_h24"),
        "price_change_5m": row.get("price_change_m5"),
        "price_change_1h": row.get("price_change_h1"),
        "price_change_6h": row.get("price_change_h6"),
        "price_change_24h": row.get("price_change_h24"),
        "whale_score": row.get("whale_score"),
        "semantic_status": row.get("semantic_status"),
        "feed_status": row.get("feed_status"),
        "freshness_status": row.get("freshness_status"),
        "tradability_status": row.get("tradability_status"),
        "verification_status": row.get("verification_status"),
        "mark_price_lookup_key": row.get("mark_price_lookup_key"),
        "mark_price_lookup_status": row.get("mark_price_lookup_status"),
        "identity_status": row.get("identity_status"),
        "safe_for_price_lookup": row.get("safe_for_price_lookup"),
        "address_display_label": "Market URL",
        "address_role_label": "Canonical Market ID (URL)",
        "data_row_key": row.get("canonical_market_identity") or final_seg,
        "row_key": row.get("canonical_market_identity") or final_seg,
        "source_provider": row.get("provider") or "dexscreener",
        "last_fetched": row.get("last_market_update_at"),
        "fetched_at": row.get("last_market_update_at"),
        "runtime_index_sourced": True,
        "external_network_on_load": False,
        **_activity_fields(row),
        **{f: disp[f] for f in RESILIENCE_FIELDS if disp.get(f) not in (None, "")},
    }


def _index_row_to_live_market(row: dict[str, Any]) -> dict[str, Any]:
    disp = _display_fields(row)
    social = _social_fields(row)
    # Market Snapshot SYMBOL column must show the full pair, never base-only.
    sym_display = disp["symbol_pair_display"]
    final_seg = row.get("provider_pair_url_final_segment_exact") or ""
    url = row.get("provider_pair_url_exact") or row.get("open_chart_url") or ""
    dex = row.get("provider_dex_id") or row.get("dex_id") or ""
    whale = row.get("whale_score")
    semantic = row.get("semantic_status") or ""
    if social["is_social_confirmed"]:
        semantic = "SOCIAL_CONFIRMED"
    elif social["is_social_candidate"] and not semantic:
        semantic = "SOCIAL_CANDIDATE_UNCONFIRMED"
    return {
        **social,
        "symbol": sym_display,
        "symbol_pair_display": sym_display,
        "symbol_pair_display_status": disp["symbol_pair_display_status"],
        "symbol_pair_display_reason": disp["symbol_pair_display_reason"],
        "symbol_pair_address_fallback": disp["symbol_pair_address_fallback"],
        "symbol_pair_known_side_symbol": disp.get("symbol_pair_known_side_symbol", ""),
        "symbol_pair_available": is_symbol_pair_available(sym_display),
        "base_token_symbol": disp["base_token_symbol"],
        "quote_token_symbol": disp["quote_token_symbol"],
        "chain": row.get("chain"),
        "dex": dex,
        "dex_id": dex,
        "canonical_market_identity": row.get("canonical_market_identity"),
        "provider_pair_url_exact": url,
        "provider_pair_url": url,
        "open_chart_url": url,
        "provider_pair_url_final_segment_exact": final_seg,
        "market_url_id": final_seg,
        "pair_address": row.get("pair_address_derived"),
        "pair_address_derived": row.get("pair_address_derived"),
        "pair": sym_display or final_seg,
        "price": row.get("price_usd"),
        "liquidity": row.get("liquidity_usd"),
        "volume_24h": row.get("volume_h24"),
        "price_change_5m": row.get("price_change_m5"),
        "price_change_1h": row.get("price_change_h1"),
        "price_change_6h": row.get("price_change_h6"),
        "price_change_24h": row.get("price_change_h24"),
        "whale_score": whale,
        "semantic_label": semantic or ("unavailable" if whale is None else "cached"),
        "semantic_signal_family": semantic or "UNKNOWN_UNRESOLVED",
        "semantic_status": semantic or row.get("tradability_status") or "indexed",
        "time": row.get("last_market_update_at") or row.get("last_identity_rebuild_at"),
        "last_seen_at": row.get("last_market_update_at"),
        "status": row.get("feed_status") or "indexed",
        "opportunity_state": "INDEXED",
        "reason": "Runtime canonical identity index (hot path)",
        "address_display_label": "Market URL ID",
        "contract_address": final_seg,
        "contract_address_deprecated": True,
        "contract_address_role": "canonical_url_final_segment",
        "contract_address_warning": None,
        "token_contract_address": row.get("base_token_address_derived")
        or row.get("provider_base_token_address"),
        "token_mint_address": row.get("base_token_address_derived")
        or row.get("provider_base_token_address"),
        "mark_price_lookup_key": row.get("mark_price_lookup_key"),
        "mark_price_lookup_status": row.get("mark_price_lookup_status"),
        "runtime_index_sourced": True,
        "_stale": row.get("freshness_status") not in (None, "", "fresh"),
        **_activity_fields(row),
        **{f: disp[f] for f in RESILIENCE_FIELDS if disp.get(f) not in (None, "")},
    }


def _compute_clean_feed_stats(raw_rows: list[dict[str, Any]], displayed: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in raw_rows if r.get("canonical_market_identity")]
    invalid = [r for r in raw_rows if not r.get("canonical_market_identity")]
    bases = {
        str(r.get("provider_base_token_address") or r.get("base_token_address_derived") or "").strip()
        for r in valid
        if str(r.get("provider_base_token_address") or r.get("base_token_address_derived") or "").strip()
    }
    markets = {
        str(r.get("canonical_market_identity") or "").strip()
        for r in valid
        if str(r.get("canonical_market_identity") or "").strip()
    }
    return {
        "total_candidates_seen": len(raw_rows),
        "runtime_index_rows": len(raw_rows),
        "valid_provider_pairs": len(valid),
        "unique_base_tokens": len(bases),
        "unique_canonical_markets": len(markets),
        "unique_pair_addresses": len(markets),  # legacy alias for UI card id
        "duplicate_pools_suppressed": max(0, len(raw_rows) - len(markets)),
        "invalid_or_unresolved_addresses": len(invalid),
        "clean_rows_displayed": len(displayed),
        "total_rows": len(displayed),
    }


def build_clean_forward_from_index(*, limit: int = 100) -> dict[str, Any]:
    loaded = load_runtime_identity_index()
    if not loaded.get("ok"):
        return {
            "ok": False,
            "status": "index_missing",
            "error_code": loaded.get("error_code", INDEX_MISSING_CODE),
            "user_message": loaded.get("user_message", INDEX_MISSING_CODE),
            "rebuild_instruction": loaded.get("rebuild_instruction"),
            "rows": [],
            "stats": {},
            "panel_title": "Clean Forward Market Feed",
            "not_live_market": True,
            "runtime_index_sourced": True,
            "external_network_on_load": False,
            "demo_mode_badge": "LIVE DISABLED / DEMO ONLY",
            **{k: loaded[k] for k in (
                "measured_load_time_ms",
                "recursive_audit_scan_used",
                "external_network_calls_on_load",
                "helius_calls_on_load",
                "dexscreener_calls_on_load",
                "pair_address_required_for_load",
            ) if k in loaded},
        }

    raw_rows = loaded.get("rows") or []
    rows = [_index_row_to_clean_forward(r) for r in raw_rows[:limit]]
    stats = _compute_clean_feed_stats(raw_rows, rows)
    stats["source_file"] = loaded.get("source_file")
    return {
        "ok": True,
        "status": "ready",
        "user_message": loaded.get("stale_message") or "",
        "stale_warning": loaded.get("stale_warning", False),
        "rows": rows,
        "alternative_pools": [],
        "stats": stats,
        "panel_title": "Clean Forward Market Feed",
        "not_live_market": True,
        "runtime_index_sourced": True,
        "external_network_on_load": False,
        "demo_mode_badge": "LIVE DISABLED / DEMO ONLY",
        "source_provider": "runtime_index",
        "built_at_utc": _utc_now(),
        "measured_load_time_ms": loaded.get("measured_load_time_ms"),
        "recursive_audit_scan_used": False,
        "external_network_calls_on_load": False,
        "helius_calls_on_load": False,
        "dexscreener_calls_on_load": False,
        "pair_address_required_for_load": False,
    }


_SEMANTIC_FILTER_FAMILIES = {
    "social": ("SOCIAL_CONFIRMED", "SOCIAL_CANDIDATE_UNCONFIRMED", "SOCIAL_COMMUNITY_ADJACENT"),
    "opportunistic": ("NON_SOCIAL_OPPORTUNISTIC_CONFIRMED", "OPPORTUNISTIC_SUSPECTED"),
    "unknown": ("UNKNOWN_INSUFFICIENT_EVIDENCE", "NEEDS_REVIEW", "UNKNOWN_UNRESOLVED"),
    "unresolved": ("UNKNOWN_UNRESOLVED", "UNKNOWN_INSUFFICIENT_EVIDENCE", "NEEDS_REVIEW"),
    "infrastructure": ("NON_SOCIAL_INFRASTRUCTURE_CONFIRMED",),
}


def _row_matches_filter(row: dict[str, Any], status_filter: str) -> bool:
    key = (status_filter or "all").strip().lower()
    if key in ("", "all"):
        return True
    if key == "social":
        if row.get("is_social_candidate") or row.get("is_social_confirmed"):
            return True
    families = _SEMANTIC_FILTER_FAMILIES.get(key)
    if families:
        return str(row.get("semantic_signal_family") or "").upper() in families
    if key == "demo":
        return str(row.get("opportunity_state") or "") == "DEMO_CANDIDATE"
    status = str(row.get("status") or "").lower()
    return key in status


def build_live_market_from_index(
    *,
    limit: int = 50,
    status_filter: str | None = None,
    filter_mode: str | None = None,
) -> dict[str, Any]:
    loaded = load_runtime_identity_index()
    if not loaded.get("ok"):
        return {
            "ok": False,
            "status": "index_missing",
            "error_code": loaded.get("error_code", INDEX_MISSING_CODE),
            "user_message": loaded.get("user_message", INDEX_MISSING_CODE),
            "rebuild_instruction": loaded.get("rebuild_instruction"),
            "rows": [],
            "live_pairs_count": 0,
            "runtime_index_sourced": True,
            "external_network_on_load": False,
            "demo_mode_badge": "LIVE DISABLED / DEMO ONLY",
            "measured_load_time_ms": loaded.get("measured_load_time_ms"),
        }

    raw_rows = loaded.get("rows") or []
    all_rows = [_index_row_to_live_market(r) for r in raw_rows]
    key = (status_filter or "all").strip().lower()
    mode = (filter_mode or "hide").strip().lower()
    matched = [r for r in all_rows if _row_matches_filter(r, key)]
    if key in ("", "all") or mode != "hide":
        for r in all_rows:
            r["filter_matched"] = _row_matches_filter(r, key)
        rows = all_rows[:limit]
    else:
        for r in matched:
            r["filter_matched"] = True
        rows = matched[:limit]
    return {
        "ok": True,
        "status": "ready",
        "user_message": loaded.get("stale_message") or "",
        "stale_warning": loaded.get("stale_warning", False),
        "rows": rows,
        "live_pairs_count": len(rows),
        "status_filter": key,
        "passed_filter": len(matched),
        "social_rows_count": sum(1 for r in all_rows if r.get("is_social_candidate")),
        "social_confirmed_rows_count": sum(1 for r in all_rows if r.get("is_social_confirmed")),
        "dropped_blocked": max(0, len(all_rows) - len(matched)) if key not in ("", "all") else 0,
        "count": len(rows),
        "filter_mode": mode,
        "runtime_index_sourced": True,
        "external_network_on_load": False,
        "demo_mode_badge": "LIVE DISABLED / DEMO ONLY",
        "latest_market_update": _utc_now(),
        "measured_load_time_ms": loaded.get("measured_load_time_ms"),
        "recursive_audit_scan_used": False,
        "external_network_calls_on_load": False,
        "helius_calls_on_load": False,
        "dexscreener_calls_on_load": False,
        "pair_address_required_for_load": False,
    }


def apply_index_mark_prices_to_trader(trader: Any) -> dict[str, Any]:
    """Inject mark prices from runtime index into PaperTrader (hot path, no network)."""
    loaded = load_runtime_identity_index()
    if not loaded.get("ok"):
        return {"applied": False, **loaded}

    from app.clean_forward.runtime_identity_index import index_rows_to_market_price_entries

    entries = index_rows_to_market_price_entries(loaded.get("rows") or [])
    trader.set_market_prices(entries, price_timestamp=loaded.get("loaded_at"))
    return {
        "applied": True,
        "entries_count": len(entries),
        "measured_load_time_ms": loaded.get("measured_load_time_ms"),
        "loaded_at": loaded.get("loaded_at"),
        "external_network_calls_on_load": False,
        "rows": loaded.get("rows") or [],
    }


def repair_legacy_position_identity(
    position: dict[str, Any],
    index_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Transitional repair: map legacy positions onto URL-first canonical identity."""
    from app.clean_forward.runtime_identity_index import resolve_position_canonical_key

    out = dict(position)
    existing = str(
        out.get("canonical_market_identity")
        or out.get("provider_pair_url_exact")
        or out.get("provider_pair_url")
        or ""
    ).strip()
    if existing:
        out["canonical_market_identity"] = existing
        out["provider_pair_url_exact"] = existing
        out["mark_price_lookup_key"] = existing
        out["open_chart_url"] = existing
        _attach_index_display(out, existing, index_rows)
        return out

    ckey = resolve_position_canonical_key(out, index_rows)
    if not ckey:
        out["mark_price_lookup_status"] = "LEGACY_POSITION_IDENTITY_REPAIR_NEEDED"
        out["mark_price_unavailable_reason"] = "LEGACY_POSITION_IDENTITY_REPAIR_NEEDED"
        out["price_resolution_failure_reason"] = "legacy_position_missing_canonical_url"
        return out

    out["canonical_market_identity"] = ckey
    out["provider_pair_url_exact"] = ckey
    out["mark_price_lookup_key"] = ckey
    out["open_chart_url"] = ckey
    _attach_index_display(out, ckey, index_rows)
    return out


def find_index_row_by_canonical(
    canonical: str, index_rows: list[dict[str, Any]]
) -> dict[str, Any] | None:
    key = str(canonical or "").strip()
    if not key:
        return None
    for ir in index_rows:
        if str(ir.get("canonical_market_identity") or "").strip() == key:
            return ir
        if str(ir.get("provider_pair_url_exact") or "").strip() == key:
            return ir
    lowered = key.lower()
    for ir in index_rows:
        if str(ir.get("canonical_market_identity") or "").strip().lower() == lowered:
            return ir
    return None


def _attach_index_display(
    out: dict[str, Any], canonical: str, index_rows: list[dict[str, Any]]
) -> None:
    """Attach display fields (never identity) from the runtime index."""
    ir = find_index_row_by_canonical(canonical, index_rows)
    if not ir:
        if not is_symbol_pair_available(out.get("symbol_pair_display")):
            out["symbol_pair_display"] = SYMBOL_PAIR_UNAVAILABLE
            out["symbol_pair_display_status"] = SYMBOL_PAIR_UNAVAILABLE
            out["symbol_pair_display_reason"] = "canonical_url_not_present_in_runtime_index"
        return
    disp = _display_fields(ir)
    out["symbol_pair_display"] = disp["symbol_pair_display"]
    out["symbol_pair_display_status"] = disp["symbol_pair_display_status"]
    out["symbol_pair_display_reason"] = disp["symbol_pair_display_reason"]
    out["symbol_pair_address_fallback"] = disp["symbol_pair_address_fallback"]
    out["symbol_pair_available"] = is_symbol_pair_available(disp["symbol_pair_display"])
    out["provider_pair_url_final_segment_exact"] = (
        out.get("provider_pair_url_final_segment_exact")
        or ir.get("provider_pair_url_final_segment_exact")
    )
    out["provider_base_token_symbol"] = ir.get("provider_base_token_symbol") or ""
    out["provider_quote_token_symbol"] = ir.get("provider_quote_token_symbol") or ""
    out["pair_address_derived"] = out.get("pair_address_derived") or ir.get("pair_address_derived")
    out["dex_id"] = out.get("dex_id") or ir.get("dex_id") or ir.get("provider_dex_id") or ""
    for k, v in _social_fields(ir).items():
        out.setdefault(k, v)


def build_index_join_map(index_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Map normalized_provider_pair_url_key → index row (exact-case preserving).

    The only permitted join key for cross-surface enrichment. Symbol,
    pair_address and lowercased final segments are never used.
    """
    by_key: dict[str, dict[str, Any]] = {}
    for ir in index_rows:
        exact = str(
            ir.get("provider_pair_url_exact") or ir.get("canonical_market_identity") or ""
        ).strip()
        key, _ = try_normalize_provider_pair_url_key(exact, require_dexscreener=True)
        if key:
            by_key.setdefault(key, ir)
    return by_key


def _opportunity_join_key(row: dict[str, Any]) -> tuple[str | None, str | None]:
    """Resolve a row's normalized URL join key. Never falls back to pair_address."""
    exact = str(
        row.get("provider_pair_url_exact")
        or row.get("canonical_market_identity")
        or row.get("provider_pair_url")
        or row.get("open_chart_url")
        or ""
    ).strip()
    if not exact:
        return None, "no_provider_pair_url_on_opportunity_row"
    key, reason = try_normalize_provider_pair_url_key(exact, require_dexscreener=True)
    if not key:
        return None, reason or "provider_pair_url_not_normalizable"
    return key, None


def enrich_opportunity_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Attach URL-first canonical identity + central pair display to opportunity rows.

    Joins runtime index rows strictly by normalized_provider_pair_url_key.
    pair_address, symbol and lowercased URL segments are never join keys, and a
    valid symbol_pair_display is never overwritten with an unavailable status.
    """
    loaded = load_runtime_identity_index()
    index_rows = loaded.get("rows") or []
    by_key = build_index_join_map(index_rows)

    base_only = 0
    joined = 0
    missing_join = 0
    join_failure_reasons: dict[str, int] = {}
    overwritten = 0
    rows_with_url = 0
    rows_with_key = 0

    for row in rows:
        key, fail_reason = _opportunity_join_key(row)
        if str(row.get("provider_pair_url_exact") or row.get("canonical_market_identity") or "").strip():
            rows_with_url += 1
        ir = by_key.get(key) if key else None
        if key:
            row["normalized_provider_pair_url_key"] = key
            rows_with_key += 1

        prior_display = str(row.get("symbol_pair_display") or "").strip()
        prior_was_proper = is_proper_symbol_pair_display(prior_display)

        if ir is not None:
            joined += 1
            canonical = str(ir.get("canonical_market_identity") or "")
            disp = _display_fields(ir)
            row["canonical_market_identity"] = canonical
            row["canonical_market_identity_type"] = (
                ir.get("canonical_market_identity_type") or "PROVIDER_URL"
            )
            row["provider_pair_url_exact"] = ir.get("provider_pair_url_exact") or canonical
            row["open_chart_url"] = ir.get("open_chart_url") or canonical
            row["provider_pair_url_final_segment_exact"] = ir.get(
                "provider_pair_url_final_segment_exact"
            )
            row["pair_address_derived"] = ir.get("pair_address_derived")
            row["dex_id"] = row.get("dex_id") or ir.get("dex_id") or ir.get("provider_dex_id") or ""
            row["symbol_pair_display"] = disp["symbol_pair_display"]
            row["symbol_pair_display_status"] = disp["symbol_pair_display_status"]
            row["symbol_pair_display_reason"] = disp["symbol_pair_display_reason"]
            row["symbol_pair_address_fallback"] = disp["symbol_pair_address_fallback"]
            row["symbol_pair_known_side_symbol"] = disp.get("symbol_pair_known_side_symbol", "")
            for fld in RESILIENCE_FIELDS:
                if disp.get(fld) not in (None, ""):
                    row[fld] = disp[fld]
            for fld in ACTIVITY_FIELDS:
                if ir.get(fld) not in (None, ""):
                    row[fld] = ir.get(fld)
            row.update(_social_fields(ir))
        else:
            missing_join += 1
            reason = fail_reason or "canonical_url_not_present_in_runtime_index"
            join_failure_reasons[reason] = join_failure_reasons.get(reason, 0) + 1
            derived = resolve_display_fields(row)
            row["symbol_pair_display"] = derived["symbol_pair_display"]
            row["symbol_pair_display_status"] = derived["symbol_pair_display_status"]
            row["symbol_pair_display_reason"] = (
                derived["symbol_pair_display_reason"] or reason
            )
            row["symbol_pair_address_fallback"] = derived["symbol_pair_address_fallback"]
            for fld in RESILIENCE_FIELDS:
                if derived.get(fld) not in (None, ""):
                    row[fld] = derived[fld]
            row.setdefault("canonical_market_identity", "")
            row.setdefault("join_failure_reason", reason)

        display = str(row.get("symbol_pair_display") or "").strip()
        # A previously valid pair display must never degrade to a status token.
        if prior_was_proper and not is_proper_symbol_pair_display(display):
            row["symbol_pair_display"] = prior_display
            display = prior_display
            overwritten += 1

        row["symbol_pair_available"] = is_symbol_pair_available(display)
        row["symbol_display"] = display
        if is_symbol_pair_available(display) and "/" not in display:
            base_only += 1
        row["pair_address_is_canonical"] = False

    return {
        "rows": rows,
        "index_rows": index_rows,
        "base_only_display_count": base_only,
        "opportunities_joined_runtime_index_count": joined,
        "opportunities_missing_runtime_index_join_count": missing_join,
        "opportunities_join_failure_reasons": join_failure_reasons,
        "opportunities_rows_with_provider_pair_url_exact": rows_with_url,
        "opportunities_rows_with_normalized_provider_pair_url_key": rows_with_key,
        "opportunities_rows_where_valid_symbol_was_overwritten": overwritten,
        "joined_by": "normalized_provider_pair_url_key",
        "runtime_index_sourced": True,
        "external_network_on_load": False,
        "measured_load_time_ms": loaded.get("measured_load_time_ms"),
    }


def build_opportunities_from_index(
    *,
    limit: int = 40,
    supplemental_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build Market Opportunities from the runtime index (same source as
    Clean Forward / Market Snapshot), guaranteeing cross-surface parity.

    Supplemental candidate metadata (semantic family, whale score, action) is
    attached only when it joins by normalized_provider_pair_url_key.
    """
    loaded = load_runtime_identity_index()
    if not loaded.get("ok"):
        return {
            "ok": False,
            "status": "index_missing",
            "error_code": loaded.get("error_code", INDEX_MISSING_CODE),
            "user_message": loaded.get("user_message", INDEX_MISSING_CODE),
            "rows": [],
            "runtime_index_sourced": True,
            "external_network_on_load": False,
        }

    index_rows = loaded.get("rows") or []
    supplemental_by_key: dict[str, dict[str, Any]] = {}
    for extra in supplemental_rows or []:
        key, _ = _opportunity_join_key(extra)
        if key:
            supplemental_by_key.setdefault(key, extra)

    rows: list[dict[str, Any]] = []
    supplemental_joined = 0
    for ir in index_rows[:limit]:
        row = _index_row_to_clean_forward(ir)
        key, _ = try_normalize_provider_pair_url_key(
            str(row.get("provider_pair_url_exact") or ""), require_dexscreener=True
        )
        row["normalized_provider_pair_url_key"] = key or ""
        extra = supplemental_by_key.get(key) if key else None
        if extra:
            supplemental_joined += 1
            for fld in (
                "whale_score",
                "semantic_signal_family",
                "semantic_label_human",
                "classification_source",
                "coin_id",
            ):
                if extra.get(fld) not in (None, ""):
                    row.setdefault(fld, extra[fld])
        rows.append(row)

    return {
        "ok": True,
        "status": "ready" if rows else "empty",
        "rows": rows,
        "index_rows": index_rows,
        "opportunities_joined_runtime_index_count": len(rows),
        "opportunities_missing_runtime_index_join_count": 0,
        "opportunities_join_failure_reasons": {},
        "supplemental_joined_count": supplemental_joined,
        "joined_by": "normalized_provider_pair_url_key",
        "runtime_index_sourced": True,
        "external_network_on_load": False,
        "measured_load_time_ms": loaded.get("measured_load_time_ms"),
    }
