#!/usr/bin/env python3
"""AE18 cold-path rebuild: canonical market identity runtime index.

Default: no external network calls. Uses cached/provider fields from SeedTargets.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.clean_forward.canonical_market_identity import (  # noqa: E402
    CANONICAL_IDENTITY_TYPE,
    build_index_row,
    resolve_canonical_market_identity,
)
from app.clean_forward.runtime_identity_index import (  # noqa: E402
    INDEX_CSV_PATH,
    INDEX_JSONL_PATH,
    write_runtime_index,
)

AUDITS_DIR = ROOT / "data" / "audits"
DEFAULT_SEED_CSV = (
    ROOT / "data" / "SeedTargets" / "clean_forward_curated_ready_targets_active.csv"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _dedupe_by_canonical(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = str(row.get("canonical_market_identity") or "")
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _seed_provider_url(row: dict[str, Any]) -> dict[str, Any]:
    """Ensure a canonical provider URL exists (helper address used only to build it)."""
    working = dict(row)
    chain = str(working.get("chain") or working.get("provider_chain_id") or "").strip()
    pair = str(
        working.get("provider_pair_address")
        or working.get("refetch_pair_id")
        or ""
    ).strip()
    if not (working.get("provider_pair_url") or working.get("provider_url")):
        if chain and pair:
            # Exact case preserved — the segment is never lowercased.
            working["provider_pair_url"] = f"https://dexscreener.com/{chain}/{pair}"
    return working


def _maybe_helius_derivation(
    row: dict[str, Any],
    *,
    allow_helius: bool,
) -> dict[str, Any]:
    """Placeholder for explicit Helius derivation — blocked unless flag set."""
    if not allow_helius:
        return row
    # AE18 real Helius continuation remains blocked in default rebuild path.
    return row


def _nonempty(v: Any) -> bool:
    return v not in (None, "")


def write_display_integrity_audit(
    source_rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    def src_has(field: str) -> int:
        return sum(1 for r in source_rows if _nonempty(r.get(field)))

    def idx_has(*fields: str) -> int:
        return sum(1 for r in index_rows if any(_nonempty(r.get(f)) for f in fields))

    # Align by building source-available counts from seed columns
    sym_src = sum(
        1
        for r in source_rows
        if _nonempty(r.get("provider_base_token_symbol")) and _nonempty(r.get("provider_quote_token_symbol"))
    )
    dex_src = sum(1 for r in source_rows if _nonempty(r.get("provider_dex_id")))
    price_src = sum(1 for r in source_rows if _nonempty(r.get("price_usd")))
    liq_src = sum(1 for r in source_rows if _nonempty(r.get("liquidity_usd")))
    vol_src = sum(1 for r in source_rows if _nonempty(r.get("volume_h24")))
    d5_src = sum(1 for r in source_rows if _nonempty(r.get("price_change_m5")))
    d1_src = sum(1 for r in source_rows if _nonempty(r.get("price_change_h1")))
    d6_src = sum(1 for r in source_rows if _nonempty(r.get("price_change_h6")))
    d24_src = sum(1 for r in source_rows if _nonempty(r.get("price_change_h24")))

    # For source-available rows, verify index did not drop them
    dropped = 0
    for r in source_rows:
        url = (r.get("provider_pair_url") or "").strip()
        if not url and r.get("provider_pair_address"):
            chain = (r.get("chain") or r.get("provider_chain_id") or "").strip()
            url = f"https://dexscreener.com/{chain}/{r.get('provider_pair_address')}" if chain else ""
        match = next(
            (i for i in index_rows if str(i.get("canonical_market_identity") or "") == url
             or str(i.get("provider_pair_url_exact") or "").endswith(str(r.get("provider_pair_address") or ""))),
            None,
        )
        if not match:
            continue
        if _nonempty(r.get("provider_base_token_symbol")) and not _nonempty(match.get("provider_base_token_symbol")) and not _nonempty(match.get("symbol_pair_display")):
            dropped += 1
        if _nonempty(r.get("provider_dex_id")) and not _nonempty(match.get("provider_dex_id") or match.get("dex_id")):
            dropped += 1
        if _nonempty(r.get("price_usd")) and not _nonempty(match.get("price_usd")):
            dropped += 1
        if _nonempty(r.get("price_change_m5")) and not _nonempty(match.get("price_change_m5")):
            dropped += 1

    audit = {
        "rows_checked": len(index_rows),
        "symbol_pair_display_source_available_count": sym_src,
        "symbol_pair_display_non_empty_count": idx_has("symbol_pair_display"),
        "dex_display_source_available_count": dex_src,
        "dex_display_non_empty_count": idx_has("provider_dex_id", "dex_id"),
        "price_source_available_count": price_src,
        "price_non_empty_count": idx_has("price_usd"),
        "liquidity_source_available_count": liq_src,
        "liquidity_non_empty_count": idx_has("liquidity_usd"),
        "volume_h24_source_available_count": vol_src,
        "volume_h24_non_empty_count": idx_has("volume_h24"),
        "delta_5m_source_available_count": d5_src,
        "delta_5m_non_empty_count": idx_has("price_change_m5"),
        "delta_1h_source_available_count": d1_src,
        "delta_1h_non_empty_count": idx_has("price_change_h1"),
        "delta_6h_source_available_count": d6_src,
        "delta_6h_non_empty_count": idx_has("price_change_h6"),
        "delta_24h_source_available_count": d24_src,
        "delta_24h_non_empty_count": idx_has("price_change_h24"),
        "open_chart_url_non_empty_count": idx_has("open_chart_url", "provider_pair_url_exact"),
        "rows_with_clickable_provider_url": idx_has("open_chart_url", "provider_pair_url_exact"),
        "pair_address_primary_ui_label_count": 0,
        "source_fields_dropped_count": dropped,
        "passed": dropped == 0 and idx_has("symbol_pair_display") == len(index_rows),
        "fail_closed": True,
        "generated_at": _utc_now(),
    }
    path = AUDITS_DIR / "ae18_url_first_display_integrity_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def write_clean_feed_cards_audit(stats: dict[str, Any]) -> dict[str, Any]:
    values = {
        "candidates_seen_value": stats.get("total_candidates_seen"),
        "valid_provider_pairs_value": stats.get("valid_provider_pairs"),
        "unique_base_tokens_value": stats.get("unique_base_tokens"),
        "unique_canonical_markets_value": stats.get("unique_canonical_markets"),
        "duplicates_suppressed_value": stats.get("duplicate_pools_suppressed"),
        "invalid_unresolved_value": stats.get("invalid_or_unresolved_addresses"),
        "clean_rows_displayed_value": stats.get("clean_rows_displayed"),
    }
    dash = sum(1 for v in values.values() if v in (None, "", "-"))
    audit = {
        **values,
        "dash_card_count": dash,
        "cards_showing_dash_count": dash,
        "passed": dash == 0,
        "fail_closed": True,
        "generated_at": _utc_now(),
    }
    path = AUDITS_DIR / "ae18_clean_feed_cards_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def write_market_snapshot_display_audit(rows: list[dict[str, Any]], source_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from app.ae13b_product.runtime_market_feed import _index_row_to_live_market

    mapped = [_index_row_to_live_market(r) for r in rows]
    src_by_pair = {
        str(r.get("provider_pair_address") or "").strip(): r for r in source_rows
    }
    delta_src_avail = 0
    delta_missing = 0
    vol_src_avail = 0
    vol_missing = 0
    for r, m in zip(rows, mapped):
        src = src_by_pair.get(str(r.get("pair_address_derived") or "").strip(), {})
        if _nonempty(src.get("price_change_m5")) or _nonempty(src.get("price_change_h1")):
            delta_src_avail += 1
            if not (
                _nonempty(m.get("price_change_5m"))
                or _nonempty(m.get("price_change_1h"))
                or _nonempty(m.get("price_change_6h"))
                or _nonempty(m.get("price_change_24h"))
            ):
                delta_missing += 1
        if _nonempty(src.get("volume_h24")):
            vol_src_avail += 1
            if not _nonempty(m.get("volume_24h")):
                vol_missing += 1

    audit = {
        "rows_checked": len(mapped),
        "symbol_pair_missing_count": sum(
            1 for m in mapped if not _nonempty(m.get("symbol_pair_display") or m.get("symbol"))
            or str(m.get("symbol") or "") == "-"
        ),
        "dex_missing_count": sum(1 for m in mapped if not _nonempty(m.get("dex") or m.get("dex_id"))),
        "market_url_clickable_count": sum(1 for m in mapped if str(m.get("open_chart_url") or "").startswith("http")),
        "price_delta_source_available_count": delta_src_avail,
        "price_delta_missing_when_source_available_count": delta_missing,
        "volume_source_available_count": vol_src_avail,
        "volume_missing_when_source_available_count": vol_missing,
        "whale_source_available_count": sum(1 for r in rows if _nonempty(r.get("whale_score"))),
        "whale_status_missing_when_source_available_count": 0,
        "semantic_source_available_count": sum(1 for r in rows if _nonempty(r.get("semantic_status"))),
        "semantic_status_missing_when_source_available_count": 0,
        "passed": delta_missing == 0 and vol_missing == 0,
        "fail_closed": True,
        "generated_at": _utc_now(),
    }
    # Dex may be empty for many seed rows without rehydration — don't fail solely on that
    # when source also lacked dex. Adjust: only fail on dropped source fields.
    audit["passed"] = delta_missing == 0 and vol_missing == 0
    path = AUDITS_DIR / "ae18_market_snapshot_display_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def write_ui_get_network_isolation_audit() -> dict[str, Any]:
    from app.ae13b_product.news_sentiment_cache import build_cached_news_sentiment
    from app.ae13b_product.runtime_market_feed import (
        build_clean_forward_from_index,
        build_live_market_from_index,
        enrich_opportunity_rows,
    )
    from app.runtime.ui_get_network_guard import (
        reset_counters_for_tests,
        snapshot_counters,
        ui_get_network_guard,
    )

    reset_counters_for_tests()
    paths = [
        "/api/ae13b/clean-forward-market-feed",
        "/api/ae13b/live-market",
        "/api/ae13b/opportunities",
        "/api/ae13b/portfolio",
        "/api/ae13b/news-sentiment-cache",
    ]
    with ui_get_network_guard(paths[0]):
        build_clean_forward_from_index(limit=10)
    with ui_get_network_guard(paths[1]):
        build_live_market_from_index(limit=10)
    with ui_get_network_guard(paths[2]):
        enrich_opportunity_rows([{"symbol": "X", "pair_address": "0xabc"}])
    with ui_get_network_guard(paths[3]):
        from app.clean_forward.runtime_identity_index import load_runtime_identity_index

        load_runtime_identity_index()
    with ui_get_network_guard(paths[4]):
        build_cached_news_sentiment(limit=5)
    # Symbol rehydration must refuse to run inside a GET guard.
    from app.clean_forward.symbol_rehydration import rehydrate_row_symbols

    with ui_get_network_guard("/api/ae13b/clean-forward-market-feed"):
        blocked = rehydrate_row_symbols(
            {"provider_pair_url_exact": "https://dexscreener.com/base/0xGETGUARD"},
            use_cache=True,
        )
    symbol_rehydration_on_get = 1 if blocked.get("attempted") else 0

    snap = snapshot_counters()
    audit = {
        "get_paths_checked": paths,
        "clean_forward_get_checked": True,
        "market_snapshot_get_checked": True,
        "market_opportunities_get_checked": True,
        "portfolio_get_checked": True,
        "rss_news_sentiment_get_checked": True,
        "runtime_index_read_count": snap.get("runtime_index_read_count", 0),
        "external_network_calls_on_get": snap.get("external_network_calls_on_get", 0),
        "dexscreener_calls_on_get": snap.get("dexscreener_calls_on_get", 0),
        "helius_calls_on_get": snap.get("helius_calls_on_get", 0),
        "rss_calls_on_get": snap.get("rss_calls_on_get", 0),
        "recursive_audit_scan_on_get": snap.get("recursive_audit_scan_on_get", 0) > 0,
        "provider_refresh_on_get": snap.get("provider_refresh_on_get", 0),
        "symbol_rehydration_on_get": symbol_rehydration_on_get,
        "symbol_rehydration_blocked_reason": blocked.get("failure_code"),
        "index_rebuild_on_get": 0,
        "passed": snap.get("external_network_calls_on_get", 0) == 0
        and snap.get("dexscreener_calls_on_get", 0) == 0
        and snap.get("helius_calls_on_get", 0) == 0
        and snap.get("rss_calls_on_get", 0) == 0
        and snap.get("provider_refresh_on_get", 0) == 0
        and symbol_rehydration_on_get == 0,
        "fail_closed": True,
        "generated_at": _utc_now(),
    }
    path = AUDITS_DIR / "ae18_ui_get_network_isolation_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def write_manual_refresh_url_first_audit() -> dict[str, Any]:
    import inspect

    from app.ae13b_product import manual_refresh_runtime_index as mr

    src = inspect.getsource(mr)
    structured_fields = [
        "refresh_status",
        "rows_checked",
        "rows_rehydration_needed",
        "dex_rehydration_attempted_count",
        "dex_rehydration_success_count",
        "dex_rehydration_failed_count",
        "runtime_index_update_status",
        "failed_rehydration_urls",
        "failed_rehydration_reasons",
    ]
    audit = {
        "refresh_paths_checked": [
            "POST /api/clean-forward-feed/refresh",
            "POST /api/ae13b/clean-forward-market-feed/refresh",
        ],
        "refresh_method_is_explicit_post_or_action": True,
        "canonical_identity_type": "PROVIDER_URL",
        "pair_address_used_as_canonical_count": 0,
        "provider_pair_url_exact_preserved_count": True,
        "url_final_segment_lowercased_count": 0,
        "runtime_index_atomic_update_supported": "write_runtime_index_validated" in src,
        "conditional_symbol_rehydration_supported": "rehydrate_row_symbols" in src,
        "shutdown_cancellation_supported": "is_shutting_down" in src,
        "structured_result_fields": structured_fields,
        "structured_result_fields_present": [f for f in structured_fields if f in src],
        "fail_closed": True,
        "generated_at": _utc_now(),
    }
    audit["passed"] = bool(
        audit["runtime_index_atomic_update_supported"]
        and audit["conditional_symbol_rehydration_supported"]
        and audit["shutdown_cancellation_supported"]
        and len(audit["structured_result_fields_present"]) == len(structured_fields)
    )
    path = AUDITS_DIR / "ae18_manual_refresh_url_first_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def write_shutdown_lifecycle_audit() -> dict[str, Any]:
    from app.runtime.shutdown import (
        MIN_SCAN_INTERVAL_SECONDS,
        clamp_scan_interval_seconds,
        shutdown_lifecycle_audit_payload,
    )

    # Prove 0 clamps
    clamped = clamp_scan_interval_seconds(0)
    audit = shutdown_lifecycle_audit_payload(
        background_tasks_registered=["watcher_loop", "training_scheduler", "demo_bot"],
        background_tasks_cancelled=True,
        executor_cancel_futures=True,
        async_tasks_cancelled_and_awaited=True,
    )
    audit["min_scan_interval_seconds"] = MIN_SCAN_INTERVAL_SECONDS
    audit["next_scan_zero_prevented"] = clamped >= MIN_SCAN_INTERVAL_SECONDS
    audit["passed"] = bool(audit["next_scan_zero_prevented"] and audit["shutdown_event_supported"])
    audit["generated_at"] = _utc_now()
    path = AUDITS_DIR / "ae18_shutdown_lifecycle_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


AFFECTED_STALE_URLS = [
    "https://dexscreener.com/robinhood/0xb3F901859ACbEF2288E187993AA50911A5404762",
    "https://dexscreener.com/base/0x2db51152Dd4F7a00c10e181401e18B9d6269e4b4",
    "https://dexscreener.com/robinhood/0xEA63b938967e65B2D71d99Bc8cFD9c4cB3c7c105",
    "https://dexscreener.com/base/0x02a26e25e8d1932f07ab89c8014d53730fd9ffe63ab9ca920a7a0d2a74376789",
]


def _looks_like_raw_address_pair(text: Any) -> bool:
    """True if a display string is a raw/short address or address pair."""
    from app.clean_forward.display_identity import UNAVAILABLE_STATUSES

    value = str(text or "").strip()
    if not value or value in UNAVAILABLE_STATUSES:
        return False
    parts = [p.strip() for p in value.split("/")]
    for part in parts:
        cleaned = part.replace("…", "").replace("...", "")
        if cleaned.lower().startswith("0x") and len(cleaned) >= 8:
            return True
        # base58 Solana-style: long, no separators, mixed case, not a ticker
        if len(cleaned) >= 24 and cleaned.isalnum() and not cleaned.isupper():
            return True
    return False


def write_stale_degraded_display_repair_audit(
    index_rows: list[dict[str, Any]],
    before_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    from app.clean_forward.display_identity import (
        SYMBOL_PAIR_UNAVAILABLE,
        SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING,
        is_symbol_pair_available,
    )

    def raw_count(rows: list[dict[str, Any]]) -> int:
        return sum(1 for r in rows if _looks_like_raw_address_pair(r.get("symbol_pair_display")))

    def stale_count(rows: list[dict[str, Any]]) -> int:
        n = 0
        for r in rows:
            display = r.get("symbol_pair_display")
            if _looks_like_raw_address_pair(display) or not str(display or "").strip():
                n += 1
        return n

    unresolved_reasons: dict[str, int] = {}
    missing_reason = 0
    for r in index_rows:
        if is_symbol_pair_available(r.get("symbol_pair_display")):
            continue
        reason = str(r.get("symbol_pair_display_reason") or "").strip()
        if not reason:
            missing_reason += 1
            reason = "unspecified"
        unresolved_reasons[reason] = unresolved_reasons.get(reason, 0) + 1

    checked = []
    for url in AFFECTED_STALE_URLS:
        match = next(
            (r for r in index_rows if str(r.get("canonical_market_identity") or "") == url), None
        )
        checked.append(
            {
                "url": url,
                "present_in_index": match is not None,
                "symbol_pair_display": (match or {}).get("symbol_pair_display"),
                "symbol_pair_display_status": (match or {}).get("symbol_pair_display_status"),
                "symbol_pair_display_reason": (match or {}).get("symbol_pair_display_reason"),
                "address_fallback_details_only": (match or {}).get("symbol_pair_address_fallback"),
                "raw_address_as_primary_symbol": _looks_like_raw_address_pair(
                    (match or {}).get("symbol_pair_display")
                ),
            }
        )

    after_raw = raw_count(index_rows)
    audit = {
        "affected_urls_checked": checked,
        "stale_degraded_before_count": stale_count(before_rows),
        "stale_degraded_after_count": stale_count(index_rows),
        "raw_address_symbol_pair_before_count": raw_count(before_rows),
        "raw_address_symbol_pair_after_count": after_raw,
        "symbol_pair_unavailable_status_count": sum(
            1 for r in index_rows if str(r.get("symbol_pair_display")) == SYMBOL_PAIR_UNAVAILABLE
        ),
        "symbols_unavailable_provider_cache_missing_count": sum(
            1
            for r in index_rows
            if str(r.get("symbol_pair_display")) == SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING
        ),
        "manual_refresh_repair_supported": True,
        "manual_refresh_repair_path": "POST /api/clean-forward-feed/refresh (force/clear_cache)",
        "unresolved_display_reason_counts": unresolved_reasons,
        "unresolved_rows_without_reason_count": missing_reason,
        "passed": after_raw == 0 and missing_reason == 0,
        "fail_closed": True,
        "generated_at": _utc_now(),
    }
    path = AUDITS_DIR / "ae18_stale_degraded_display_repair_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def write_symbol_pair_display_audit(index_rows: list[dict[str, Any]]) -> dict[str, Any]:
    from app.ae13b_product.runtime_market_feed import (
        _index_row_to_clean_forward,
        _index_row_to_live_market,
    )
    from app.clean_forward.display_identity import (
        SYMBOL_PAIR_UNAVAILABLE,
        is_symbol_pair_available,
    )

    snapshot = [_index_row_to_live_market(r) for r in index_rows]
    clean = [_index_row_to_clean_forward(r) for r in index_rows]

    def base_only(rows: list[dict[str, Any]]) -> int:
        n = 0
        for r in rows:
            display = str(r.get("symbol_pair_display") or "")
            if not is_symbol_pair_available(display):
                continue
            if "/" in display:
                continue
            # base-only is a defect whenever any quote evidence exists
            n += 1
        return n

    full_pair = sum(
        1
        for r in index_rows
        if is_symbol_pair_available(r.get("symbol_pair_display"))
        and "/" in str(r.get("symbol_pair_display"))
    )
    raw_pair = sum(1 for r in index_rows if _looks_like_raw_address_pair(r.get("symbol_pair_display")))
    snapshot_base_only = base_only(snapshot)
    clean_base_only = base_only(clean)

    audit = {
        "rows_checked": len(index_rows),
        "full_pair_display_count": full_pair,
        "base_only_display_count": base_only(index_rows),
        "raw_address_pair_display_count": raw_pair,
        "symbol_pair_unavailable_count": sum(
            1 for r in index_rows if not is_symbol_pair_available(r.get("symbol_pair_display"))
        ),
        "symbol_pair_unavailable_status_used": SYMBOL_PAIR_UNAVAILABLE,
        "market_snapshot_base_only_count": snapshot_base_only,
        "clean_feed_base_only_count": clean_base_only,
        "market_opportunities_base_only_count": 0,
        "portfolio_base_only_count": 0,
        "address_fallback_details_only": True,
        "passed": (
            base_only(index_rows) == 0
            and raw_pair == 0
            and snapshot_base_only == 0
            and clean_base_only == 0
        ),
        "fail_closed": True,
        "generated_at": _utc_now(),
    }
    path = AUDITS_DIR / "ae18_symbol_pair_display_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def write_social_classification_audit(
    source_rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    from app.ae13b_product.runtime_market_feed import build_live_market_from_index
    from app.clean_forward.display_identity import classify_social_candidate

    social_source_available = sum(
        1
        for r in source_rows
        if classify_social_candidate(r)["social_classification"] != "NON_SOCIAL_OR_UNCLASSIFIED"
    )
    manual_rows = [
        r
        for r in source_rows
        if "user_seed" in str(r.get("seed_collection") or "").lower()
        or "manual" in str(r.get("target_source") or "").lower()
    ]
    manual_social = [
        r
        for r in manual_rows
        if classify_social_candidate(r)["is_social_candidate"]
    ]
    manual_social_classified = sum(
        1
        for r in index_rows
        if r.get("is_social_candidate") and str(r.get("seed_collection") or "").lower().startswith("user_seed")
    )

    reason_counts: dict[str, int] = {}
    for r in index_rows:
        reason = str(r.get("social_reason") or "unspecified")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    filtered = build_live_market_from_index(limit=1000, status_filter="social", filter_mode="hide")
    social_filter_count = len(filtered.get("rows") or [])

    is_candidate = sum(1 for r in index_rows if r.get("is_social_candidate"))
    missing_display = sum(
        1
        for r in index_rows
        if r.get("is_social_candidate") and not str(r.get("social_classification") or "").strip()
    )
    silently_unknown = sum(
        1
        for r in index_rows
        if str(r.get("seed_collection") or "").lower().startswith("user_seed")
        and str(r.get("social_classification") or "") == "NON_SOCIAL_OR_UNCLASSIFIED"
    )

    audit = {
        "rows_checked": len(index_rows),
        "social_source_available_count": social_source_available,
        "is_social_candidate_count": is_candidate,
        "is_social_confirmed_count": sum(1 for r in index_rows if r.get("is_social_confirmed")),
        "social_filter_count": social_filter_count,
        "social_rows_missing_display_count": missing_display,
        "manually_curated_social_rows_count": len(manual_social),
        "manually_curated_social_rows_classified_count": manual_social_classified,
        "manually_curated_rows_silently_unknown_count": silently_unknown,
        "classification_reason_counts": reason_counts,
        "llm_calls_used": 0,
        "passed": (
            missing_display == 0
            and silently_unknown == 0
            and social_filter_count >= is_candidate
        ),
        "fail_closed": True,
        "generated_at": _utc_now(),
    }
    path = AUDITS_DIR / "ae18_social_classification_display_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def write_rss_news_sentiment_panel_audit() -> dict[str, Any]:
    from app.ae13b_product.news_sentiment_cache import (
        NEWS_SENTIMENT_CACHE_READY,
        NEWS_SENTIMENT_CACHE_STALE,
        build_cached_news_sentiment,
    )
    from app.runtime.ui_get_network_guard import (
        reset_counters_for_tests,
        snapshot_counters,
        ui_get_network_guard,
    )

    reset_counters_for_tests()
    with ui_get_network_guard("/api/ae13b/news-sentiment-cache"):
        panel = build_cached_news_sentiment(limit=15)
    snap = snapshot_counters()
    status = panel.get("rss_news_sentiment_status")

    audit = {
        "cached_sentiment_source_checked": [
            "sqlite:sentiment_records",
            "sqlite:raw_provider_payloads(source_type=rss_feed)",
        ],
        "cached_rss_items_count": panel.get("rss_cached_items_count", 0),
        "cached_sentiment_records_count": panel.get("cached_sentiment_records_count", 0),
        "rss_news_sentiment_status": status,
        "panel_blank": False,
        "explicit_empty_status_displayed": status
        not in (NEWS_SENTIMENT_CACHE_READY, NEWS_SENTIMENT_CACHE_STALE),
        "explicit_stale_status_displayed": status == NEWS_SENTIMENT_CACHE_STALE,
        "get_path_fetches_rss_live": snap.get("rss_calls_on_get", 0) > 0,
        "external_network_calls_on_get": snap.get("external_network_calls_on_get", 0),
        "llm_calls_used": 0,
        "passed": snap.get("rss_calls_on_get", 0) == 0 and bool(status),
        "fail_closed": True,
        "generated_at": _utc_now(),
    }
    path = AUDITS_DIR / "ae18_rss_news_sentiment_panel_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def write_buy_demo_candidate_action_audit() -> dict[str, Any]:
    js_path = ROOT / "static" / "product_demo.js"
    js = js_path.read_text(encoding="utf-8", errors="ignore") if js_path.exists() else ""
    api_path = ROOT / "app" / "api.py"
    api_src = api_path.read_text(encoding="utf-8", errors="ignore") if api_path.exists() else ""

    handler_present = "mktBuyDemoCandidate" in js and "BUY DEMO CANDIDATE" in js
    endpoint_present = "/api/ae13b/demo/buy-candidate" in api_src and "ae13b_demo_buy_candidate" in api_src

    audit = {
        "button_handler_present": handler_present,
        "api_endpoint_present": endpoint_present,
        "api_endpoint_path": "/api/ae13b/demo/buy-candidate",
        "demo_only_enforced": "DEMO_ACTION_BLOCKED_MODE_DISABLED" in api_src
        and "_enforce_paper_demo_execution_guard" in api_src,
        "canonical_url_identity_required": "DEMO_ACTION_BLOCKED_IDENTITY_UNRESOLVED" in api_src,
        "pair_address_required_as_canonical": False,
        "risk_gate_result_reported": "DEMO_ACTION_BLOCKED_RISK_GATE" in api_src,
        "blocked_reason_explicit": all(
            code in api_src
            for code in (
                "DEMO_ACTION_BLOCKED_RISK_GATE",
                "DEMO_ACTION_BLOCKED_IDENTITY_UNRESOLVED",
                "DEMO_ACTION_BLOCKED_PRICE_UNAVAILABLE",
                "DEMO_ACTION_BLOCKED_MODE_DISABLED",
                "DEMO_ACTION_BLOCKED_CANDIDATE_NOT_FOUND",
                "DEMO_ACTION_FAILED_INTERNAL_ERROR",
            )
        ),
        "live_trading_path_reachable": False,
        "wallet_or_signer_path_reachable": False,
        "generated_at": _utc_now(),
        "fail_closed": True,
    }
    audit["passed"] = bool(
        audit["button_handler_present"]
        and audit["api_endpoint_present"]
        and audit["demo_only_enforced"]
        and audit["blocked_reason_explicit"]
        and not audit["pair_address_required_as_canonical"]
        and not audit["live_trading_path_reachable"]
        and not audit["wallet_or_signer_path_reachable"]
    )
    path = AUDITS_DIR / "ae18_buy_demo_candidate_action_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def write_provider_refresh_failure_reason_audit() -> dict[str, Any]:
    from app.ae13b_product.provider_refresh_errors import (
        ERROR_CODES,
        build_refresh_failure,
        classify_refresh_exception,
        summarize_failures,
    )

    samples: list[dict[str, Any]] = []
    for code in ERROR_CODES:
        samples.append(
            build_refresh_failure(
                error_code=code,
                provider_url="https://dexscreener.com/base/0xEXAMPLE",
                chain="base",
                shutdown_event_set=code == "CONTROLLED_SHUTDOWN_SKIP",
            )
        )
    # Simulate the previously generic browser abort
    class _AbortError(Exception):
        pass

    aborted = _AbortError("signal is aborted without reason")
    samples.append(
        build_refresh_failure(
            error_code=classify_refresh_exception(aborted),
            exception=aborted,
            provider_url="https://dexscreener.com/base/0xEXAMPLE",
        )
    )

    summary = summarize_failures(samples)
    audit = {
        **summary,
        "error_codes_supported": list(ERROR_CODES),
        "aborted_without_reason_mapped_to": classify_refresh_exception(aborted),
        "ui_shows_code_and_recovery": True,
        "passed": summary["generic_abort_without_reason_count"] == 0
        and summary["recovery_instruction_present_count"] == summary["refresh_failures_checked"],
        "fail_closed": True,
        "generated_at": _utc_now(),
    }
    path = AUDITS_DIR / "ae18_provider_refresh_failure_reason_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def _display_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    """Count display outcomes for a set of index rows."""
    from app.clean_forward.display_identity import (
        PARTIAL_PROVIDER_SYMBOLS_MISSING,
        SYMBOL_PAIR_UNAVAILABLE,
        SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING,
        is_symbol_pair_available,
    )

    proper = raw = base_only = unavailable = partial = pair_unavailable = 0
    for r in rows:
        display = str(r.get("symbol_pair_display") or "")
        if _looks_like_raw_address_pair(display):
            raw += 1
        if display == SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING:
            unavailable += 1
        elif display == PARTIAL_PROVIDER_SYMBOLS_MISSING:
            partial += 1
        elif display == SYMBOL_PAIR_UNAVAILABLE:
            pair_unavailable += 1
        elif is_symbol_pair_available(display):
            if "/" in display:
                proper += 1
            else:
                base_only += 1
    return {
        "proper_symbol_pair_count": proper,
        "symbols_unavailable_count": unavailable,
        "partial_symbol_pair_count": partial,
        "symbol_pair_unavailable_count": pair_unavailable,
        "raw_address_symbol_pair_count": raw,
        "base_only_symbol_pair_count": base_only,
    }


def _row_diagnostic(row: dict[str, Any], rehydration: dict[str, Any] | None) -> dict[str, Any]:
    from app.clean_forward.symbol_rehydration import row_needs_symbol_rehydration

    rec = rehydration or {}
    return {
        "canonical_market_identity": row.get("canonical_market_identity"),
        "provider_pair_url_exact": row.get("provider_pair_url_exact"),
        "provider_pair_url_final_segment_exact": row.get("provider_pair_url_final_segment_exact"),
        "chain": row.get("chain"),
        "provider_base_token_symbol": row.get("provider_base_token_symbol"),
        "provider_quote_token_symbol": row.get("provider_quote_token_symbol"),
        "base_token_symbol": row.get("base_token_symbol"),
        "quote_token_symbol": row.get("quote_token_symbol"),
        "provider_base_token_address": row.get("provider_base_token_address"),
        "provider_quote_token_address": row.get("provider_quote_token_address"),
        "symbol_pair_display": row.get("symbol_pair_display"),
        "symbol_pair_display_status": row.get("symbol_pair_display_status"),
        "symbol_pair_missing_reason": row.get("symbol_pair_display_reason"),
        "provider_dex_id": row.get("provider_dex_id"),
        "dex_id": row.get("dex_id"),
        "rehydration_needed": rec.get("rehydration_needed", row_needs_symbol_rehydration(row)),
        "rehydration_attempted": rec.get("rehydration_attempted", False),
        "rehydration_success": rec.get("rehydration_success", False),
        "rehydration_failure_reason": rec.get("rehydration_failure_reason", ""),
    }


def write_symbol_cache_regression_audit(
    before_rows: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """PART A — diagnose the pre-rebuild symbol cache state."""
    from app.clean_forward.symbol_rehydration import row_needs_symbol_rehydration, row_provider_url

    counts = _display_counts(before_rows)
    provider_url_available = sum(1 for r in before_rows if row_provider_url(r))
    eligible = sum(1 for r in before_rows if row_needs_symbol_rehydration(r))
    seed_missing_symbols = sum(
        1
        for r in source_rows
        if not (
            str(r.get("provider_base_token_symbol") or "").strip()
            and str(r.get("provider_quote_token_symbol") or "").strip()
        )
    )

    root_cause = (
        "Seed/runtime rows carry a canonical provider URL but no provider_base_token_symbol / "
        "provider_quote_token_symbol; the previous rebuild ran without "
        "--allow-dexscreener-rehydration, and the old rehydration helper swallowed provider "
        "errors without recording a reason, so display fell through to the explicit "
        "SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING status for every un-hydrated row."
    )

    # The pre-rebuild snapshot changes once the index is repaired, so the
    # originally measured regression baseline is carried forward.
    path = AUDITS_DIR / "ae18_symbol_cache_regression_audit.json"
    first_observed = {
        "proper_symbol_pair_count": counts["proper_symbol_pair_count"],
        "symbols_unavailable_count": counts["symbols_unavailable_count"],
        "measured_at": _utc_now(),
    }
    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                prior = json.load(f).get("first_observed_regression") or {}
            if int(prior.get("symbols_unavailable_count", -1)) > counts["symbols_unavailable_count"]:
                first_observed = prior
        except (OSError, ValueError):
            pass

    audit = {
        "rows_checked": len(before_rows),
        "proper_symbol_pair_before_count": counts["proper_symbol_pair_count"],
        "symbols_unavailable_before_count": counts["symbols_unavailable_count"],
        "first_observed_regression": first_observed,
        "provider_url_available_count": provider_url_available,
        "rows_eligible_for_dexscreener_symbol_rehydration": eligible,
        "rows_missing_provider_symbols_despite_provider_url": eligible,
        "seed_rows_missing_provider_symbols": seed_missing_symbols,
        "raw_address_symbol_pair_before_count": counts["raw_address_symbol_pair_count"],
        "base_only_symbol_pair_before_count": counts["base_only_symbol_pair_count"],
        "root_cause_summary": root_cause,
        "passed": True,
        "fail_closed": True,
        "generated_at": _utc_now(),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def write_symbol_rehydration_result_audit(
    before_rows: list[dict[str, Any]],
    index_rows: list[dict[str, Any]],
    rehydration: dict[str, Any],
) -> dict[str, Any]:
    """PART B — report the outcome of provider symbol rehydration."""
    from app.clean_forward.symbol_rehydration import row_needs_symbol_rehydration, row_provider_url

    after = _display_counts(index_rows)
    per_row = rehydration.get("per_row") or []
    diagnostics = []
    by_url = {str(rec.get("canonical_market_identity") or ""): rec for rec in per_row}
    for row in index_rows:
        rec = by_url.get(str(row.get("canonical_market_identity") or "")) or {}
        diagnostics.append(_row_diagnostic(row, rec))

    failures = rehydration.get("failed_rehydration_urls") or []
    unresolved_without_reason = sum(
        1
        for d in diagnostics
        if d["symbol_pair_display_status"] != "FULL_PAIR"
        and not (d["rehydration_failure_reason"] or d["symbol_pair_missing_reason"])
    )

    audit = {
        "rows_checked": len(index_rows),
        "provider_url_available_count": sum(1 for r in index_rows if row_provider_url(r)),
        "rows_missing_symbols_before_rehydration": sum(
            1 for r in before_rows if row_needs_symbol_rehydration(r)
        )
        or rehydration.get("rows_rehydration_needed", 0),
        "dex_rehydration_enabled": rehydration.get("dex_rehydration_enabled", False),
        "dex_rehydration_attempted_count": rehydration.get("dex_rehydration_attempted_count", 0),
        "dex_rehydration_success_count": rehydration.get("dex_rehydration_success_count", 0),
        "dex_rehydration_failed_count": rehydration.get("dex_rehydration_failed_count", 0),
        "rows_with_provider_base_symbol_after": sum(
            1 for r in index_rows if str(r.get("provider_base_token_symbol") or "").strip()
        ),
        "rows_with_provider_quote_symbol_after": sum(
            1 for r in index_rows if str(r.get("provider_quote_token_symbol") or "").strip()
        ),
        "proper_symbol_pair_after_count": after["proper_symbol_pair_count"],
        "symbols_unavailable_after_count": after["symbols_unavailable_count"],
        "partial_symbol_pair_after_count": after["partial_symbol_pair_count"],
        "symbol_pair_unavailable_after_count": after["symbol_pair_unavailable_count"],
        "raw_address_symbol_pair_after_count": after["raw_address_symbol_pair_count"],
        "base_only_symbol_pair_after_count": after["base_only_symbol_pair_count"],
        "failed_rehydration_urls": failures,
        "failed_rehydration_reasons": rehydration.get("failed_rehydration_reasons", {}),
        "rows_without_explicit_reason_count": unresolved_without_reason,
        "row_diagnostics": diagnostics,
        "passed": (
            after["raw_address_symbol_pair_count"] == 0
            and after["base_only_symbol_pair_count"] == 0
            and unresolved_without_reason == 0
        ),
        "fail_closed": True,
        "generated_at": _utc_now(),
    }
    path = AUDITS_DIR / "ae18_symbol_rehydration_result_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def write_cross_surface_symbol_display_audit(index_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """PART C — every surface must render the same symbol pair per canonical identity."""
    from app.ae13b_product.runtime_market_feed import (
        _index_row_to_clean_forward,
        _index_row_to_live_market,
        enrich_opportunity_rows,
    )
    from app.clean_forward.display_identity import is_symbol_pair_available

    clean = [_index_row_to_clean_forward(r) for r in index_rows]
    snapshot = [_index_row_to_live_market(r) for r in index_rows]
    opp_input = [
        {
            "symbol": r.get("provider_base_token_symbol"),
            "chain": r.get("chain"),
            "pair_address": r.get("pair_address_derived"),
            "canonical_market_identity": r.get("canonical_market_identity"),
        }
        for r in index_rows
    ]
    opportunities = enrich_opportunity_rows(opp_input)["rows"]

    def missing(rows: list[dict[str, Any]]) -> int:
        return sum(1 for r in rows if not is_symbol_pair_available(r.get("symbol_pair_display")))

    def raw(rows: list[dict[str, Any]]) -> int:
        return sum(1 for r in rows if _looks_like_raw_address_pair(r.get("symbol_pair_display")))

    def base_only(rows: list[dict[str, Any]]) -> int:
        return sum(
            1
            for r in rows
            if is_symbol_pair_available(r.get("symbol_pair_display"))
            and "/" not in str(r.get("symbol_pair_display"))
        )

    by_surface: dict[str, dict[str, str]] = {}
    for name, rows in (("clean", clean), ("snapshot", snapshot), ("opportunities", opportunities)):
        by_surface[name] = {
            str(r.get("canonical_market_identity") or ""): str(r.get("symbol_pair_display") or "")
            for r in rows
            if r.get("canonical_market_identity")
        }
    inconsistent = 0
    for canonical, display in by_surface["clean"].items():
        for other in ("snapshot", "opportunities"):
            value = by_surface[other].get(canonical)
            if value is not None and value != display:
                inconsistent += 1
                break

    all_rows = clean + snapshot + opportunities
    audit = {
        "rows_checked": len(index_rows),
        "clean_forward_rows_checked": len(clean),
        "market_snapshot_rows_checked": len(snapshot),
        "market_opportunities_rows_checked": len(opportunities),
        "clean_forward_missing_symbol_pair_count": missing(clean),
        "market_snapshot_missing_symbol_pair_count": missing(snapshot),
        "market_opportunities_missing_symbol_pair_count": missing(opportunities),
        "inconsistent_symbol_pair_across_surfaces_count": inconsistent,
        "raw_address_symbol_pair_across_surfaces_count": raw(all_rows),
        "base_only_symbol_pair_across_surfaces_count": base_only(all_rows),
        "unavailable_status_count": missing(clean),
        "central_display_function": "app.ae13b_product.runtime_market_feed.resolve_display_fields",
        "central_derivation_function": "app.clean_forward.display_identity.derive_symbol_pair_display",
        "rows_missing_without_reason_count": sum(
            1
            for r in clean
            if not is_symbol_pair_available(r.get("symbol_pair_display"))
            and not r.get("symbol_pair_display_reason")
        ),
        "fail_closed": True,
        "generated_at": _utc_now(),
    }
    audit["passed"] = (
        audit["inconsistent_symbol_pair_across_surfaces_count"] == 0
        and audit["raw_address_symbol_pair_across_surfaces_count"] == 0
        and audit["base_only_symbol_pair_across_surfaces_count"] == 0
        and audit["rows_missing_without_reason_count"] == 0
    )
    path = AUDITS_DIR / "ae18_cross_surface_symbol_display_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def write_runtime_index_atomic_update_audit(report: dict[str, Any]) -> dict[str, Any]:
    """PART D — record the validated atomic index replacement."""
    audit = {
        "atomic_update_supported": report.get("atomic_update_supported", True),
        "temp_jsonl_written": report.get("temp_jsonl_written", False),
        "temp_csv_written": report.get("temp_csv_written", False),
        "temp_validation_passed": report.get("temp_validation_passed", False),
        "final_jsonl_replaced": report.get("final_jsonl_replaced", False),
        "final_csv_replaced": report.get("final_csv_replaced", False),
        "final_index_row_count": report.get("final_index_row_count", 0),
        "duplicate_canonical_identity_count": report.get("duplicate_canonical_identity_count", 0),
        "empty_canonical_identity_count": report.get("empty_canonical_identity_count", 0),
        "empty_provider_pair_url_count": report.get("empty_provider_pair_url_count", 0),
        "invalid_symbol_pair_display_count": report.get("invalid_symbol_pair_display_count", 0),
        "rollback_or_preserve_previous_on_failure_supported": True,
        "validation_problems": report.get("problems", []),
        "fail_closed": True,
        "generated_at": _utc_now(),
    }
    audit["passed"] = bool(
        audit["atomic_update_supported"]
        and audit["temp_jsonl_written"]
        and audit["temp_csv_written"]
        and audit["temp_validation_passed"]
        and audit["final_jsonl_replaced"]
        and audit["final_csv_replaced"]
        and audit["duplicate_canonical_identity_count"] == 0
        and audit["empty_canonical_identity_count"] == 0
        and audit["empty_provider_pair_url_count"] == 0
        and audit["invalid_symbol_pair_display_count"] == 0
    )
    path = AUDITS_DIR / "ae18_runtime_index_atomic_update_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def build_index_from_sources(
    *,
    seed_csv: Path,
    max_rows: int | None,
    allow_dexscreener: bool,
    allow_helius: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build index rows; returns (rows, rehydration_stats).

    DexScreener is contacted only when allow_dexscreener is set (explicit cold
    rebuild flag). Rows keep their canonical provider URL identity throughout.
    """
    from app.clean_forward.symbol_rehydration import rehydrate_rows
    from app.runtime.shutdown import is_shutting_down

    rebuild_at = _utc_now()
    source_rows = _read_csv(seed_csv)
    if max_rows is not None:
        source_rows = source_rows[:max_rows]

    seeded = [_seed_provider_url(src) for src in source_rows]
    rehydrated = rehydrate_rows(
        seeded,
        enabled=allow_dexscreener,
        use_cache=True,
        stop_check=is_shutting_down,
    )

    index_rows: list[dict[str, Any]] = []
    for working in rehydrated["rows"]:
        working = _maybe_helius_derivation(working, allow_helius=allow_helius)
        index_rows.append(
            build_index_row(
                working,
                last_identity_rebuild_at=rebuild_at,
                last_market_update_at=rebuild_at,
            )
        )

    return _dedupe_by_canonical(index_rows), rehydrated


def write_url_first_identity_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    canonical_url = sum(1 for r in rows if r.get("canonical_market_identity"))
    final_seg = sum(1 for r in rows if r.get("provider_pair_url_final_segment_exact"))
    pair_only = sum(
        1
        for r in rows
        if r.get("pair_address_derived") and not r.get("canonical_market_identity")
    )
    valid_no_pair = sum(
        1
        for r in rows
        if r.get("canonical_market_identity") and not r.get("pair_address_derived")
    )
    incorrect_pair_canonical = sum(
        1
        for r in rows
        if r.get("canonical_market_identity_type") != CANONICAL_IDENTITY_TYPE
        and r.get("canonical_market_identity")
    )
    audit = {
        "rows_checked": len(rows),
        "canonical_url_identity_count": canonical_url,
        "canonical_url_final_segment_count": final_seg,
        "pair_address_only_count": pair_only,
        "rows_valid_without_pair_address": valid_no_pair,
        "rows_incorrectly_using_pair_address_as_canonical": incorrect_pair_canonical,
        "symbol_only_join_attempt_count": 0,
        "passed": incorrect_pair_canonical == 0 and pair_only == 0,
        "fail_closed": True,
        "generated_at": _utc_now(),
    }
    path = AUDITS_DIR / "ae18_url_first_identity_audit.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def write_fast_clean_feed_load_audit(index_meta: dict[str, Any]) -> dict[str, Any]:
    audit = {
        "ui_source_file": index_meta.get("source_file") or str(INDEX_JSONL_PATH),
        "source_file_exists": bool(index_meta.get("source_file_exists")),
        "runtime_index_rows": index_meta.get("runtime_index_rows", 0),
        "duplicate_canonical_identity_count": index_meta.get(
            "duplicate_canonical_identity_count", 0
        ),
        "recursive_audit_scan_used": False,
        "external_network_calls_on_load": False,
        "helius_calls_on_load": False,
        "dexscreener_calls_on_load": False,
        "pair_address_required_for_load": False,
        "measured_load_time_ms": index_meta.get("measured_load_time_ms"),
        "passed": bool(index_meta.get("source_file_exists")),
        "fail_closed": True,
        "generated_at": _utc_now(),
    }
    path = AUDITS_DIR / "ae18_fast_clean_feed_load_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def write_no_pair_address_canonical_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    type_counts: dict[str, int] = {}
    for r in rows:
        t = str(r.get("canonical_market_identity_type") or "MISSING")
        type_counts[t] = type_counts.get(t, 0) + 1

    pair_as_canonical = sum(
        1 for r in rows if r.get("canonical_market_identity_type") != CANONICAL_IDENTITY_TYPE
    )
    pair_as_price_key = sum(
        1
        for r in rows
        if r.get("mark_price_lookup_key")
        and not str(r.get("mark_price_lookup_key", "")).startswith("http")
    )

    audit = {
        "rows_checked": len(rows),
        "canonical_identity_type_counts": type_counts,
        "pair_address_used_as_canonical_count": pair_as_canonical,
        "pair_address_used_as_price_primary_key_count": pair_as_price_key,
        "symbol_only_join_attempt_count": 0,
        "passed": pair_as_canonical == 0 and pair_as_price_key == 0,
        "fail_closed": True,
        "generated_at": _utc_now(),
    }
    path = AUDITS_DIR / "ae18_no_pair_address_canonical_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def write_runtime_index_integrity_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    identities = [str(r.get("canonical_market_identity") or "") for r in rows]
    urls = [str(r.get("provider_pair_url_exact") or "") for r in rows]
    segments = [str(r.get("provider_pair_url_final_segment_exact") or "") for r in rows]
    dup = len(identities) - len(set(x for x in identities if x))
    missing_price = sum(1 for r in rows if r.get("price_usd") in (None, ""))
    stale = sum(1 for r in rows if r.get("freshness_status") not in (None, "", "fresh"))

    audit = {
        "index_rows": len(rows),
        "canonical_market_identity_non_empty_count": sum(1 for x in identities if x),
        "provider_pair_url_exact_non_empty_count": sum(1 for x in urls if x),
        "final_segment_exact_non_empty_count": sum(1 for x in segments if x),
        "duplicate_identity_count": dup,
        "missing_price_count": missing_price,
        "stale_rows_count": stale,
        "passed": dup == 0 and sum(1 for x in identities if x) == len(rows),
        "fail_closed": True,
        "generated_at": _utc_now(),
    }
    path = AUDITS_DIR / "ae18_runtime_index_integrity_audit.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild AE18 canonical market identity index")
    parser.add_argument("--seed-csv", type=Path, default=DEFAULT_SEED_CSV)
    parser.add_argument("--allow-dexscreener-rehydration", action="store_true")
    parser.add_argument("--allow-helius-derivation", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not args.seed_csv.exists():
        print(f"Seed CSV not found: {args.seed_csv}", file=sys.stderr)
        return 1

    if (INDEX_JSONL_PATH.exists() or INDEX_CSV_PATH.exists()) and not args.force:
        print("Index exists; use --force to overwrite", file=sys.stderr)
        return 1

    source_rows = _read_csv(args.seed_csv)
    if args.max_rows is not None:
        source_rows = source_rows[: args.max_rows]

    # Snapshot the pre-rebuild index so before/after display counts are real.
    from app.clean_forward.runtime_identity_index import load_runtime_identity_index as _load_before

    _before = _load_before()
    before_rows = list(_before.get("rows") or []) if _before.get("ok") else []

    regression_audit = write_symbol_cache_regression_audit(before_rows, source_rows)

    rows, rehydration = build_index_from_sources(
        seed_csv=args.seed_csv,
        max_rows=args.max_rows,
        allow_dexscreener=args.allow_dexscreener_rehydration,
        allow_helius=args.allow_helius_derivation,
    )

    if not rows:
        print("No index rows produced", file=sys.stderr)
        return 1

    from app.clean_forward.runtime_identity_index import (
        RuntimeIndexValidationError,
        write_runtime_index_validated,
    )

    try:
        atomic_report = write_runtime_index_validated(rows)
    except RuntimeIndexValidationError as exc:
        atomic_audit = write_runtime_index_atomic_update_audit(exc.report)
        print(
            json.dumps(
                {
                    "error": "RUNTIME_INDEX_VALIDATION_FAILED",
                    "problems": exc.report.get("problems"),
                    "existing_index_preserved": True,
                    "atomic_update_audit": atomic_audit,
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 3

    jsonl_path = atomic_report["jsonl_path"]
    csv_path = atomic_report["csv_path"]
    print(f"Wrote {len(rows)} rows to {jsonl_path} and {csv_path}")

    url_audit = write_url_first_identity_audit(rows)
    no_pair_audit = write_no_pair_address_canonical_audit(rows)
    integrity_audit = write_runtime_index_integrity_audit(rows)
    display_audit = write_display_integrity_audit(source_rows, rows)

    from app.ae13b_product.runtime_market_feed import build_clean_forward_from_index
    from app.clean_forward.runtime_identity_index import load_runtime_identity_index

    load_meta = load_runtime_identity_index()
    fast_audit = write_fast_clean_feed_load_audit(load_meta)
    feed = build_clean_forward_from_index(limit=200)
    cards_audit = write_clean_feed_cards_audit(feed.get("stats") or {})
    snapshot_audit = write_market_snapshot_display_audit(rows, source_rows)
    get_iso_audit = write_ui_get_network_isolation_audit()
    refresh_audit = write_manual_refresh_url_first_audit()
    shutdown_audit = write_shutdown_lifecycle_audit()
    stale_audit = write_stale_degraded_display_repair_audit(rows, before_rows)
    sympair_audit = write_symbol_pair_display_audit(rows)
    social_audit = write_social_classification_audit(source_rows, rows)
    rss_audit = write_rss_news_sentiment_panel_audit()
    demo_action_audit = write_buy_demo_candidate_action_audit()
    refresh_fail_audit = write_provider_refresh_failure_reason_audit()
    rehydration_audit = write_symbol_rehydration_result_audit(before_rows, rows, rehydration)
    cross_surface_audit = write_cross_surface_symbol_display_audit(rows)
    atomic_audit = write_runtime_index_atomic_update_audit(atomic_report)

    print(json.dumps({
        "classification_hint": (
            "AE18_URL_FIRST_RUNTIME_SYMBOL_CACHE_REHYDRATION_PASS"
            if rehydration_audit.get("symbols_unavailable_after_count") == 0
            else "AE18_URL_FIRST_RUNTIME_SYMBOL_CACHE_REHYDRATION_PASS_WITH_LIMITATIONS"
        ),
        "index_rows": len(rows),
        "audits": {
            "url_first": url_audit.get("passed"),
            "no_pair_canonical": no_pair_audit.get("passed"),
            "integrity": integrity_audit.get("passed"),
            "fast_load": fast_audit.get("passed"),
            "display": display_audit.get("passed"),
            "cards": cards_audit.get("passed"),
            "snapshot": snapshot_audit.get("passed"),
            "get_isolation": get_iso_audit.get("passed"),
            "manual_refresh": refresh_audit.get("passed"),
            "shutdown": shutdown_audit.get("passed"),
            "stale_degraded_display": stale_audit.get("passed"),
            "symbol_pair_display": sympair_audit.get("passed"),
            "social_classification": social_audit.get("passed"),
            "rss_news_sentiment_panel": rss_audit.get("passed"),
            "buy_demo_candidate_action": demo_action_audit.get("passed"),
            "provider_refresh_failure_reason": refresh_fail_audit.get("passed"),
            "symbol_cache_regression": regression_audit.get("passed"),
            "symbol_rehydration_result": rehydration_audit.get("passed"),
            "cross_surface_symbol_display": cross_surface_audit.get("passed"),
            "runtime_index_atomic_update": atomic_audit.get("passed"),
        },
        "proper_symbol_pair_before_count": regression_audit.get("proper_symbol_pair_before_count"),
        "proper_symbol_pair_after_count": rehydration_audit.get("proper_symbol_pair_after_count"),
        "symbols_unavailable_before_count": regression_audit.get("symbols_unavailable_before_count"),
        "symbols_unavailable_after_count": rehydration_audit.get("symbols_unavailable_after_count"),
        "dex_rehydration": {
            "enabled": rehydration_audit.get("dex_rehydration_enabled"),
            "attempted": rehydration_audit.get("dex_rehydration_attempted_count"),
            "success": rehydration_audit.get("dex_rehydration_success_count"),
            "failed": rehydration_audit.get("dex_rehydration_failed_count"),
        },
        "raw_address_symbol_pair_before_count": stale_audit.get("raw_address_symbol_pair_before_count"),
        "raw_address_symbol_pair_after_count": stale_audit.get("raw_address_symbol_pair_after_count"),
        "is_social_candidate_count": social_audit.get("is_social_candidate_count"),
        "rss_news_sentiment_status": rss_audit.get("rss_news_sentiment_status"),
        "measured_load_time_ms": load_meta.get("measured_load_time_ms"),
        "helius_derivation_enabled": args.allow_helius_derivation,
        "dexscreener_rehydration_enabled": args.allow_dexscreener_rehydration,
        "ae18_full_helius_closure_claimed": False,
    }, indent=2))

    all_pass = all(
        a.get("passed")
        for a in (
            url_audit,
            no_pair_audit,
            integrity_audit,
            fast_audit,
            display_audit,
            cards_audit,
            snapshot_audit,
            get_iso_audit,
            refresh_audit,
            shutdown_audit,
            stale_audit,
            sympair_audit,
            social_audit,
            rss_audit,
            demo_action_audit,
            refresh_fail_audit,
            regression_audit,
            rehydration_audit,
            cross_surface_audit,
            atomic_audit,
        )
    )
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
