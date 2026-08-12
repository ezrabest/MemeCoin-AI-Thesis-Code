"""Print the AE18 final stabilization status report from generated audits."""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.clean_forward.runtime_identity_index import load_runtime_identity_index  # noqa: E402

AUDITS = ROOT / "data" / "audits"


def load(name: str) -> dict:
    path = AUDITS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    mo = load("ae18_market_opportunities_symbol_regression_audit.json")
    tr = load("ae18_unresolved_symbol_trade_readiness_audit.json")
    gi = load("ae18_get_isolation_strict_audit.json")
    cache = load("ae18_cache_atomicity_and_locking_audit.json")
    ovr = load("ae18_manual_display_overrides_validation_audit.json")
    pc = load("ae18_position_continuity_contract_audit.json")
    dto = load("ae18_position_financial_dto_safety_audit.json")
    prov = load("ae18_provider_resilience_status_audit.json")

    rows = load_runtime_identity_index()["rows"]

    print("== SURFACE COUNTS ==")
    print("runtime_index_rows:", mo.get("runtime_index_rows"))
    for surface in ("clean_forward", "market_snapshot", "market_opportunities"):
        print(
            f"  {surface}: proper={mo.get(f'{surface}_proper_symbol_pair_count')} "
            f"unresolved={mo.get(f'{surface}_unresolved_symbol_count')}"
        )
    for surface in ("clean_forward", "market_snapshot", "market_opportunities"):
        print(f"  {surface}_unresolved_urls:")
        for u in mo.get(f"{surface}_unresolved_urls", []):
            print("     ", u)

    print("\n== OPPORTUNITIES JOIN ==")
    for k in (
        "opportunities_joined_runtime_index_count",
        "opportunities_missing_runtime_index_join_count",
        "opportunities_join_failure_reasons",
        "opportunities_rows_where_valid_symbol_was_overwritten",
        "raw_address_primary_display_count",
        "base_only_primary_display_count",
        "empty_primary_display_count",
        "joined_by",
    ):
        print(f"  {k}: {mo.get(k)}")

    print("\n== UNRESOLVED SYMBOL TRADE READINESS ==")
    print("  rows_checked:", tr.get("unresolved_symbol_rows_checked"))
    print("  rows_blocked_due_to_symbol_only_count:", tr.get("rows_blocked_due_to_symbol_only_count"))
    print("  paper_eligible:", tr.get("unresolved_symbol_rows_paper_eligible_count"))
    print("  watch_only:", tr.get("unresolved_symbol_rows_watch_only_count"))
    print("  entry_blocked:", tr.get("unresolved_symbol_rows_entry_blocked_count"))
    for url, st in (tr.get("status_by_url") or {}).items():
        print(f"  {url}")
        for k, v in st.items():
            print(f"      {k}: {v}")

    print("\n== STATUS COUNTS (runtime index) ==")
    for field in (
        "display_metadata_status",
        "provider_resolution_status",
        "symbol_resolution_status",
        "market_data_status",
        "identity_readiness_status",
        "trade_readiness_status",
    ):
        print(f"  {field}: {dict(Counter(str(r.get(field) or '') for r in rows))}")

    print("\n== DISPLAY SOURCES ==")
    print("  last_good_display_count:", prov.get("last_good_display_count"))
    print("  manual_override_display_count:", prov.get("manual_override_display_count"))
    print("  applied_override_count:", ovr.get("applied_override_count"))
    print("  override rejected_rows:", ovr.get("rejected_rows"))

    print("\n== GET ISOLATION ==")
    for k in (
        "get_network_calls_count",
        "get_cache_write_count",
        "audit_write_on_get_count",
        "dexscreener_calls_on_get",
        "helius_calls_on_get",
        "rss_calls_on_get",
        "index_rebuild_on_get",
        "symbol_rehydration_on_get",
        "provider_refresh_on_get",
        "recursive_audit_scan_on_get",
    ):
        print(f"  {k}: {gi.get(k)}")
    print("  paths:", gi.get("get_paths_checked"))

    print("\n== POSITIONS ==")
    try:
        from app.api import get_paper_trader

        positions = get_paper_trader().get_positions(status="OPEN")
        print("  open_positions:", len(positions))
        print(
            "  position_market_data_state:",
            dict(Counter(str(p.get("position_market_data_state") or "") for p in positions)),
        )
        print(
            "  financial_data_status:",
            dict(Counter(str(p.get("financial_data_status") or "") for p in positions)),
        )
    except Exception as exc:  # noqa: BLE001
        print("  positions unavailable:", exc)

    print("\n== AUDIT RESULTS ==")
    for name, audit in (
        ("market_opportunities_symbol_regression", mo),
        ("unresolved_symbol_trade_readiness", tr),
        ("get_isolation_strict", gi),
        ("cache_atomicity_and_locking", cache),
        ("manual_display_overrides_validation", ovr),
        ("position_continuity_contract", pc),
        ("position_financial_dto_safety", dto),
        ("provider_resilience_status", prov),
    ):
        print(f"  {name}: passed={audit.get('passed')} fail_closed={audit.get('fail_closed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
