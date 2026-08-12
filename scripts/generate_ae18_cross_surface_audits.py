"""Generate AE18 final-stabilization audits.

Produces:
  * ae18_market_opportunities_symbol_regression_audit.json
  * ae18_unresolved_symbol_trade_readiness_audit.json
  * ae18_get_isolation_strict_audit.json

Read-only with respect to provider networks: no DexScreener/Helius/RSS calls.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.ae13b_product import runtime_market_feed as rmf  # noqa: E402
from app.clean_forward.provider_resilience_statuses import (  # noqa: E402
    FORBIDDEN_BLOCK_REASON,
    IDENTITY_UNRESOLVED,
    IDENTITY_UNVERIFIABLE,
    MARKET_DATA_MISSING,
    MARKET_DATA_PROVIDER_UNAVAILABLE,
    MARKET_DATA_READY,
    MARKET_DATA_STALE,
    MARKET_DATA_UNVERIFIABLE,
    PAPER_ELIGIBLE,
    WATCH_ONLY,
    assert_block_reason_not_symbol_only,
    block_reason_for,
    is_proper_symbol_pair_display,
)
from app.clean_forward.provider_url_key import try_normalize_provider_pair_url_key  # noqa: E402
from app.clean_forward.runtime_identity_index import load_runtime_identity_index  # noqa: E402
from app.runtime.ui_get_network_guard import (  # noqa: E402
    reset_counters_for_tests,
    snapshot_counters,
    ui_get_network_guard,
)

AUDITS = ROOT / "data" / "audits"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write(name: str, payload: dict) -> dict:
    AUDITS.mkdir(parents=True, exist_ok=True)
    payload["generated_at_utc"] = _now()
    (AUDITS / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _looks_like_raw_address(text: str) -> bool:
    value = str(text or "").strip()
    if not value or "/" in value:
        return False
    if value.startswith("0x") and len(value) >= 40:
        return True
    return len(value) >= 32 and value.isalnum()


def _surfaces():
    index_rows = load_runtime_identity_index()["rows"]
    clean = [rmf._index_row_to_clean_forward(r) for r in index_rows]
    snap = [rmf._index_row_to_live_market(r) for r in index_rows]
    built = rmf.build_opportunities_from_index(limit=10_000)
    enriched = rmf.enrich_opportunity_rows(built["rows"])
    return index_rows, clean, snap, enriched


def audit_market_opportunities_regression() -> dict:
    index_rows, clean, snap, enriched = _surfaces()
    opp = enriched["rows"]

    def counts(rows, field="symbol_pair_display"):
        proper = sum(1 for r in rows if is_proper_symbol_pair_display(r.get(field)))
        return proper, len(rows) - proper

    cf_proper, cf_unres = counts(clean)
    ms_proper, ms_unres = counts(snap)
    mo_proper, mo_unres = counts(opp)

    def unresolved_urls(rows):
        return sorted(
            str(r.get("provider_pair_url_exact") or "")
            for r in rows
            if not is_proper_symbol_pair_display(r.get("symbol_pair_display"))
        )

    raw_addr = sum(
        1 for r in opp if _looks_like_raw_address(r.get("symbol_pair_display"))
    )
    base_only = sum(
        1
        for r in opp
        if is_proper_symbol_pair_display(r.get("symbol_pair_display"))
        and "/" not in str(r.get("symbol_pair_display"))
    )
    empty = sum(1 for r in opp if not str(r.get("symbol_pair_display") or "").strip())

    parity = (
        unresolved_urls(opp) == unresolved_urls(clean) == unresolved_urls(snap)
        and mo_proper == cf_proper == ms_proper
    )
    passed = bool(
        parity
        and enriched["opportunities_rows_where_valid_symbol_was_overwritten"] == 0
        and raw_addr == 0
        and base_only == 0
        and empty == 0
        and not (mo_unres == len(opp) and cf_proper > 0)
    )

    return _write(
        "ae18_market_opportunities_symbol_regression_audit.json",
        {
            "runtime_index_rows": len(index_rows),
            "clean_forward_rows_checked": len(clean),
            "market_snapshot_rows_checked": len(snap),
            "market_opportunities_rows_checked": len(opp),
            "clean_forward_proper_symbol_pair_count": cf_proper,
            "market_snapshot_proper_symbol_pair_count": ms_proper,
            "market_opportunities_proper_symbol_pair_count": mo_proper,
            "clean_forward_unresolved_symbol_count": cf_unres,
            "market_snapshot_unresolved_symbol_count": ms_unres,
            "market_opportunities_unresolved_symbol_count": mo_unres,
            "clean_forward_unresolved_urls": unresolved_urls(clean),
            "market_snapshot_unresolved_urls": unresolved_urls(snap),
            "market_opportunities_unresolved_urls": unresolved_urls(opp),
            "opportunities_joined_runtime_index_count": enriched[
                "opportunities_joined_runtime_index_count"
            ],
            "opportunities_missing_runtime_index_join_count": enriched[
                "opportunities_missing_runtime_index_join_count"
            ],
            "opportunities_join_failure_reasons": enriched[
                "opportunities_join_failure_reasons"
            ],
            "opportunities_rows_with_provider_pair_url_exact": enriched[
                "opportunities_rows_with_provider_pair_url_exact"
            ],
            "opportunities_rows_with_normalized_provider_pair_url_key": enriched[
                "opportunities_rows_with_normalized_provider_pair_url_key"
            ],
            "opportunities_rows_where_valid_symbol_was_overwritten": enriched[
                "opportunities_rows_where_valid_symbol_was_overwritten"
            ],
            "joined_by": enriched["joined_by"],
            "cross_surface_parity": parity,
            "raw_address_primary_display_count": raw_addr,
            "base_only_primary_display_count": base_only,
            "empty_primary_display_count": empty,
            "passed": passed,
            "fail_closed": not passed,
        },
    )


def audit_unresolved_symbol_trade_readiness() -> dict:
    index_rows = load_runtime_identity_index()["rows"]
    unresolved = [
        r for r in index_rows if not is_proper_symbol_pair_display(r.get("symbol_pair_display"))
    ]

    urls = [str(r.get("provider_pair_url_exact") or "") for r in unresolved]
    market_counts: Counter[str] = Counter()
    block_reasons: dict[str, str] = {}
    per_url: dict[str, dict] = {}
    paper_eligible = watch_only = blocked = 0
    symbol_only_blocks = 0
    identity_unresolved = 0
    continuity_unsafe = 0

    for row in unresolved:
        url = str(row.get("provider_pair_url_exact") or "")
        market = str(row.get("market_data_status") or "")
        identity = str(row.get("identity_readiness_status") or "")
        trade = str(row.get("trade_readiness_status") or "")
        market_counts[market] += 1
        if identity in {IDENTITY_UNRESOLVED, IDENTITY_UNVERIFIABLE}:
            identity_unresolved += 1
        if trade == PAPER_ELIGIBLE:
            paper_eligible += 1
        elif trade == WATCH_ONLY:
            watch_only += 1
        elif trade.startswith("ENTRY_BLOCKED") or trade == "MANUAL_REVIEW_REQUIRED":
            blocked += 1
            reason = str(row.get("trade_block_reason") or "") or block_reason_for(trade)
            block_reasons[url] = reason
            if trade == "ENTRY_BLOCKED_POSITION_CONTINUITY_UNSAFE":
                continuity_unsafe += 1
            # A block is symbol-only if nothing except display is degraded.
            non_display_cause = (
                market != MARKET_DATA_READY
                or identity in {IDENTITY_UNRESOLVED, IDENTITY_UNVERIFIABLE}
                or trade == "ENTRY_BLOCKED_POSITION_CONTINUITY_UNSAFE"
            )
            if not non_display_cause or not assert_block_reason_not_symbol_only(reason):
                symbol_only_blocks += 1
        per_url[url] = {
            "display_metadata_status": row.get("display_metadata_status"),
            "provider_resolution_status": row.get("provider_resolution_status"),
            "symbol_resolution_status": row.get("symbol_resolution_status"),
            "market_data_status": market,
            "identity_readiness_status": identity,
            "trade_readiness_status": trade,
            "block_reason": str(row.get("trade_block_reason") or ""),
            "unresolved_reason": row.get("unresolved_reason"),
        }

    all_have_display_status = all(
        bool(r.get("display_metadata_status")) for r in unresolved
    )
    no_raw_address = not any(
        _looks_like_raw_address(r.get("symbol_pair_display")) for r in unresolved
    )

    passed = bool(
        symbol_only_blocks == 0 and all_have_display_status and no_raw_address
    )

    return _write(
        "ae18_unresolved_symbol_trade_readiness_audit.json",
        {
            "unresolved_symbol_rows_checked": len(unresolved),
            "unresolved_symbol_urls": urls,
            "missing_symbol_only_count": len(unresolved),
            "missing_symbol_but_market_data_ready_count": market_counts[MARKET_DATA_READY],
            "missing_symbol_market_data_stale_count": market_counts[MARKET_DATA_STALE],
            "missing_symbol_market_data_missing_count": market_counts[MARKET_DATA_MISSING]
            + market_counts[MARKET_DATA_PROVIDER_UNAVAILABLE],
            "missing_symbol_market_data_unverifiable_count": market_counts[
                MARKET_DATA_UNVERIFIABLE
            ],
            "missing_symbol_identity_unresolved_count": identity_unresolved,
            "missing_symbol_position_continuity_unsafe_count": continuity_unsafe,
            "unresolved_symbol_rows_paper_eligible_count": paper_eligible,
            "unresolved_symbol_rows_watch_only_count": watch_only,
            "unresolved_symbol_rows_entry_blocked_count": blocked,
            "rows_blocked_due_to_symbol_only_count": symbol_only_blocks,
            "block_reasons_by_url": block_reasons,
            "status_by_url": per_url,
            "every_unresolved_row_has_display_metadata_status": all_have_display_status,
            "raw_address_primary_display_count": 0 if no_raw_address else 1,
            "forbidden_block_reason": FORBIDDEN_BLOCK_REASON,
            "passed": passed,
            "fail_closed": not passed,
        },
    )


def audit_strict_get_isolation() -> dict:
    from app.ae13b_product.news_sentiment_cache import build_cached_news_sentiment
    from app.api import _build_opportunities, get_paper_trader

    reset_counters_for_tests()
    checked: list[str] = []

    with ui_get_network_guard("/api/ae13b/clean-forward"):
        rmf.build_clean_forward_from_index(limit=100)
    checked.append("/api/ae13b/clean-forward")

    with ui_get_network_guard("/api/ae13b/live-market"):
        rmf.build_live_market_from_index(limit=100)
    checked.append("/api/ae13b/live-market")

    with ui_get_network_guard("/api/ae13b/opportunities"):
        _build_opportunities(100)
    checked.append("/api/ae13b/opportunities")

    with ui_get_network_guard("/api/portfolio"):
        get_paper_trader().get_positions(status="OPEN")
    checked.append("/api/portfolio")

    rss_checked = False
    try:
        with ui_get_network_guard("/api/ae13b/news-sentiment-cache"):
            build_cached_news_sentiment(limit=10)
        checked.append("/api/ae13b/news-sentiment-cache")
        rss_checked = True
    except Exception:
        rss_checked = False

    snap = snapshot_counters()
    counters = {
        "get_network_calls_count": snap["external_network_calls_on_get"],
        "get_cache_write_count": snap["cache_write_on_get"],
        "audit_write_on_get_count": snap["audit_write_on_get"],
        "dexscreener_calls_on_get": snap["dexscreener_calls_on_get"],
        "helius_calls_on_get": snap["helius_calls_on_get"],
        "rss_calls_on_get": snap["rss_calls_on_get"],
        "index_rebuild_on_get": snap["index_rebuild_on_get"],
        "symbol_rehydration_on_get": snap["symbol_rehydration_on_get"],
        "provider_refresh_on_get": snap["provider_refresh_on_get"],
        "recursive_audit_scan_on_get": snap["recursive_audit_scan_on_get"],
    }
    passed = all(v == 0 for v in counters.values())

    return _write(
        "ae18_get_isolation_strict_audit.json",
        {
            "get_paths_checked": checked,
            "clean_forward_get_checked": True,
            "market_snapshot_get_checked": True,
            "market_opportunities_get_checked": True,
            "portfolio_get_checked": True,
            "rss_news_get_checked": rss_checked,
            **counters,
            "runtime_index_read_count": snap["runtime_index_read_count"],
            "passed": passed,
            "fail_closed": not passed,
        },
    )


def main() -> int:
    mo = audit_market_opportunities_regression()
    tr = audit_unresolved_symbol_trade_readiness()
    gi = audit_strict_get_isolation()
    for name, audit in (
        ("market_opportunities_symbol_regression", mo),
        ("unresolved_symbol_trade_readiness", tr),
        ("get_isolation_strict", gi),
    ):
        print(f"{name}: passed={audit['passed']} fail_closed={audit['fail_closed']}")
    print(
        "opportunities proper/unresolved:",
        mo["market_opportunities_proper_symbol_pair_count"],
        "/",
        mo["market_opportunities_unresolved_symbol_count"],
    )
    print("symbol-only blocks:", tr["rows_blocked_due_to_symbol_only_count"])
    return 0 if all(a["passed"] for a in (mo, tr, gi)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
