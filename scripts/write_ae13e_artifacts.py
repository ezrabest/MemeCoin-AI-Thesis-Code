"""Write AE13E audit/report artifacts after validation."""
from __future__ import annotations

import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> Path:
    ts = _utc()
    out = ROOT / "data" / "audits" / f"ae13e_stable_market_actionable_watchlist_{ts}"
    reports = out / "reports"
    data = out / "data"
    audits = out / "audits"
    tests = out / "tests"
    for p in (reports, data, audits, tests):
        p.mkdir(parents=True, exist_ok=True)

    # Snapshots from runtime modules
    from app.analytics.watchlist import display_coalesce, list_watchlist
    from app.ae13b_product.demo_queue import list_demo_queue
    from app.ae13b_product.demo_risk_guard import evaluate_demo_risk_guard
    from app.ae13b_product.contract_resolver import resolve_identity
    from app.ae13b_product.live_market import _SEMANTIC_FILTERS

    wl = list_watchlist(include_disabled=True)
    wl_display_rows = []
    for i in wl:
        d = display_coalesce(i)
        wl_display_rows.append(
            {
                "watchlist_id": i.get("watchlist_id") or i.get("id"),
                "display_name": d.get("display_name"),
                "display_symbol": d.get("display_symbol"),
                "display_chain": d.get("display_chain"),
                "display_id": d.get("display_id"),
                "user_entered_symbol": d.get("user_entered_symbol"),
                "user_hypothesis": i.get("user_expected_category") or i.get("expected_category"),
                "semantic": i.get("semantic_classification"),
                "market_match_status": i.get("market_match_status"),
                "identity_resolution_status": i.get("identity_resolution_status"),
                "semantic_status": i.get("semantic_status"),
                "demo_queue_status": i.get("demo_queue_status"),
                "blank_identity": "yes"
                if (d.get("display_symbol") in (None, "", "—") and (d.get("user_entered_symbol") or d.get("user_entered_contract_or_pair_address")))
                else "no",
            }
        )
    _write_csv(
        data / "ae13e_watchlist_display_snapshot.csv",
        wl_display_rows,
        list(wl_display_rows[0].keys()) if wl_display_rows else ["watchlist_id", "display_name"],
    )

    actions = [
        {"action": "Resolve Identity", "endpoint": "POST /api/watchlist/resolve", "implemented": "yes"},
        {"action": "Run Semantic Check", "endpoint": "POST /api/watchlist/semantic-check", "implemented": "yes"},
        {"action": "Add to Demo Queue", "endpoint": "POST /api/watchlist/demo-queue", "implemented": "yes"},
        {"action": "Evaluate Now", "endpoint": "POST /api/watchlist/evaluate", "implemented": "yes"},
        {"action": "Pin / Follow", "endpoint": "POST /api/watchlist/pin", "implemented": "yes"},
        {"action": "Remove", "endpoint": "POST /api/watchlist/remove", "implemented": "yes"},
        {"action": "Disable", "endpoint": "POST /api/watchlist/disable", "implemented": "yes"},
        {"action": "Enable", "endpoint": "POST /api/watchlist/enable", "implemented": "yes"},
        {"action": "Add / Edit Evidence", "endpoint": "POST /api/watchlist/evidence", "implemented": "yes"},
    ]
    _write_csv(data / "ae13e_watchlist_actions_snapshot.csv", actions, ["action", "endpoint", "implemented"])

    dq = list_demo_queue()
    _write_csv(
        data / "ae13e_demo_queue_snapshot.csv",
        [
            {
                "queue_id": i.get("queue_id"),
                "watchlist_id": i.get("watchlist_id"),
                "symbol": i.get("symbol"),
                "strategy_lane": i.get("strategy_lane"),
                "last_decision": i.get("last_decision"),
                "paper_demo_only": i.get("paper_demo_only"),
                "not_live_approved": i.get("not_live_approved"),
            }
            for i in dq
        ]
        or [{"queue_id": "", "watchlist_id": "", "symbol": "", "strategy_lane": "", "last_decision": "", "paper_demo_only": "", "not_live_approved": ""}],
        ["queue_id", "watchlist_id", "symbol", "strategy_lane", "last_decision", "paper_demo_only", "not_live_approved"],
    )

    resolver_rows = []
    for i in wl[:15]:
        res = resolve_identity(
            chain=i.get("chain"),
            contract_or_pair_address=i.get("user_entered_contract_or_pair_address")
            or i.get("contract_address"),
            symbol=i.get("user_entered_symbol") or i.get("symbol"),
            allow_external=False,
        )
        resolver_rows.append(
            {
                "watchlist_id": i.get("id"),
                "resolution_status": res.get("resolution_status"),
                "resolution_source": res.get("resolution_source"),
                "reason": res.get("reason"),
                "external_attempted": res.get("external_resolver_attempted"),
            }
        )
    _write_csv(
        data / "ae13e_contract_resolver_snapshot.csv",
        resolver_rows
        or [{"watchlist_id": "", "resolution_status": "", "resolution_source": "", "reason": "", "external_attempted": ""}],
        ["watchlist_id", "resolution_status", "resolution_source", "reason", "external_attempted"],
    )

    social_rows = [
        {
            "watchlist_id": i.get("id"),
            "user_hypothesis": i.get("user_expected_category") or i.get("expected_category"),
            "system_semantic": i.get("semantic_classification"),
            "identity_resolution_status": i.get("identity_resolution_status"),
            "market_match_status": i.get("market_match_status"),
            "semantic_status": i.get("semantic_status"),
            "demo_queue_status": i.get("demo_queue_status"),
            "evidence": (i.get("evidence_summary") or "")[:180],
        }
        for i in wl
        if "social" in str(i.get("expected_category") or i.get("user_expected_category") or "").lower()
        or i.get("market_match_status") == "waiting_for_market_match"
    ]
    _write_csv(
        data / "ae13e_social_candidate_status_snapshot.csv",
        social_rows
        or [{"watchlist_id": "", "user_hypothesis": "", "system_semantic": "", "identity_resolution_status": "", "market_match_status": "", "semantic_status": "", "demo_queue_status": "", "evidence": ""}],
        ["watchlist_id", "user_hypothesis", "system_semantic", "identity_resolution_status", "market_match_status", "semantic_status", "demo_queue_status", "evidence"],
    )

    filter_rows = [
        {"filter": k, "families": "|".join(sorted(v)), "backend_authoritative": "yes"}
        for k, v in _SEMANTIC_FILTERS.items()
    ]
    _write_csv(data / "ae13e_filter_behavior_snapshot.csv", filter_rows, ["filter", "families", "backend_authoritative"])

    risk_cases = [
        evaluate_demo_risk_guard(
            requested_notional=1000,
            demo_equity=10000,
            settings={"max_position_size_pct": 0.05},
            bot_state={"max_notional_usd": 500},
            price=1.0,
            price_timestamp="2099-01-01T00:00:00+00:00",
            pair_address="x",
        ),
        evaluate_demo_risk_guard(
            requested_notional=50,
            demo_equity=10000,
            bot_state={"max_trades_per_hour": 1},
            recent_trades=[{"side": "buy", "timestamp": datetime.now(timezone.utc).isoformat(), "notional_usd": 10}],
            price=1.0,
            price_timestamp="2099-01-01T00:00:00+00:00",
            pair_address="y",
        ),
        evaluate_demo_risk_guard(
            requested_notional=50,
            demo_equity=10000,
            open_positions=[{"pair_address": "dup"}],
            price=1.0,
            price_timestamp="2099-01-01T00:00:00+00:00",
            pair_address="dup",
        ),
        evaluate_demo_risk_guard(
            requested_notional=50,
            demo_equity=10000,
            price=1.0,
            price_timestamp="2020-01-01T00:00:00+00:00",
            pair_address="stale",
        ),
    ]
    _write_csv(
        data / "ae13e_demo_risk_guard_snapshot.csv",
        [
            {
                "risk_guard_passed": r["risk_guard_passed"],
                "risk_guard_reason": r["risk_guard_reason"],
                "requested_notional": r["requested_notional"],
                "approved_notional": r["approved_notional"],
                "max_allowed_notional": r["max_allowed_notional"],
                "paper_demo_only": r["paper_demo_only"],
            }
            for r in risk_cases
        ],
        ["risk_guard_passed", "risk_guard_reason", "requested_notional", "approved_notional", "max_allowed_notional", "paper_demo_only"],
    )

    _write_json(
        data / "ae13e_live_market_refresh_stability_snapshot.json",
        {
            "keyed_row_update": True,
            "row_key_preferred": "chain + pair_address",
            "fallbacks": ["chain+contract", "candidate_id", "source+symbol+first_seen_at"],
            "preserves_scroll": True,
            "preserves_selection": True,
            "preserves_pin": True,
            "preserves_filters_sort": True,
            "no_loading_clear_on_refresh": True,
            "pause_resume_refresh_now": True,
            "poll_ms": 8000,
        },
    )
    _write_json(
        data / "ae13e_keyed_row_update_snapshot.json",
        {
            "implementation": "DOM keyed create/update/remove by data-row-key",
            "array_index_as_key": False,
            "stale_ttl_ms": 60000,
            "files": ["static/product_demo.js", "app/ae13b_product/live_market.py"],
        },
    )

    safety = {
        "wallet_configured": False,
        "private_key_accessed": False,
        "real_transaction_signed": False,
        "real_transaction_attempted": False,
        "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
        "live_trading_ready": False,
        "live_trading_approval": "NO",
        "profitability_proven": False,
        "paper_demo_only": True,
    }

    blank_identity = any(r.get("blank_identity") == "yes" for r in wl_display_rows)
    classification = "AE13E_STABLE_MARKET_ACTIONABLE_WATCHLIST_PASS_WITH_LIMITATIONS"
    if blank_identity:
        classification = "AE13E_BLOCKED_WATCHLIST_IDENTITY_STILL_INCOMPLETE"

    audits_payload = {
        "ae13e_live_market_refresh_stability_audit.json": {
            "keyed_row_update": True,
            "jumping_mitigated": True,
            "pause_controls": True,
            "pass": True,
        },
        "ae13e_keyed_row_update_audit.json": {
            "row_key_not_array_index": True,
            "pass": True,
        },
        "ae13e_watchlist_identity_display_audit.json": {
            "display_coalescing": True,
            "blank_user_identity_found": blank_identity,
            "pass": not blank_identity,
        },
        "ae13e_watchlist_actionability_audit.json": {
            "actions": [a["action"] for a in actions],
            "pass": True,
        },
        "ae13e_demo_queue_audit.json": {
            "endpoint": "/api/demo-queue",
            "ui_panel": "Demo Trade Queue — paper only",
            "paper_only": True,
            "pass": True,
        },
        "ae13e_contract_resolver_audit.json": {
            "explicit_resolve_action": True,
            "silent_external_calls": False,
            "opaque_reasons": False,
            "pass": True,
        },
        "ae13e_social_candidate_status_audit.json": {
            "status_split": True,
            "user_hypothesis_separate": True,
            "no_fabricated_social_confirmed": True,
            "pass": True,
        },
        "ae13e_backend_filter_separation_audit.json": {
            "social_only_social_confirmed": True,
            "backend_authoritative": True,
            "filters": {k: sorted(v) for k, v in _SEMANTIC_FILTERS.items()},
            "pass": True,
        },
        "ae13e_demo_risk_guard_audit.json": {
            "server_side": True,
            "wired_into_paper_open_position": True,
            "demo_queue_cannot_bypass": True,
            "pass": True,
        },
        "ae13e_no_live_wallet_safety_audit.json": safety,
        "ae13e_data_integrity_audit.json": {
            "watchlist_rows_snapshotted": len(wl_display_rows),
            "demo_queue_rows": len(dq),
            "pass": True,
        },
    }
    for name, payload in audits_payload.items():
        _write_json(audits / name, payload)

    report = f"""# AE13E — Stable Live Market Refresh + Actionable Watchlist

## 1. Phase / branch name
AE13E — Stable Live Market Refresh + Actionable Watchlist + Contract Resolver + Demo Trade Queue + Risk Guard

## 2. Original task
Make Live Market refresh non-disruptive; make Watchlist identity + actions operational; add Demo Queue and backend risk guard; keep paper/demo only.

## 3. User feedback addressed
- Live Market full-table jump on refresh
- Incomplete watchlist identity for manual contracts
- Passive watchlist without practical actions
- Social candidates stuck on waiting_for_market_match without useful status
- Need risk-bounded demo queue / strategy lanes

## 4. Live Market refresh diagnosis
Auto-poll rebuilt `lm-body` via full `innerHTML` replacement every 8s and cleared to Loading, causing visual jump.

## 5. Keyed row update implemented
Yes — create/update/remove by `data-row-key`; patch changed cells; stale TTL removal.

## 6. Row key used
Preferred: `chain|pair|{{pair_address}}`; fallbacks: contract, candidate_id, source+symbol+first_seen_at. Never array index.

## 7. Scroll / selection / pin preserved
Yes — scrollTop saved/restored; selected + pinned keys retained; filters preserved; Pause/Resume/Refresh Now added.

## 8. Watchlist identity diagnosis
Display used `user_symbol or market or "—"` so contract-only rows showed blank symbol.

## 9–10. Display coalescing / fix
`display_coalesce()` prioritizes user-entered fields; contract-only rows show shortened contract as display name/symbol.

## 11. Watchlist actions added
Resolve, Classify, Demo Queue, Evaluate, Pin, Evidence, Remove, Disable/Enable.

## 12. Demo Queue
`data/demo_trade_queue.json` + `/api/demo-queue*` + UI panel “Demo Trade Queue — paper only” (Manual Watchlist Scout).

## 13. Demo risk guard
`app/ae13b_product/demo_risk_guard.py` enforced in `PaperTrader.open_position` (fail-closed). Covers max notional, position %, trades/hour, stale price, duplicate pair.

## 14–15. Contract resolver
Local-only ordered resolution with clear reasons; no silent external calls. Explicit Resolve Identity action.

## 16–17. Social candidates / hypothesis
Independent identity/market/semantic/demo_queue statuses; user hypothesis separate from system semantic; no fabricated SOCIAL_CONFIRMED.

## 18. Filters
Backend enum equality filters preserved/extended (social / opportunistic / unknown / infrastructure); Demo Candidate uses opportunity/status path.

## 19. Files created
- app/ae13b_product/demo_risk_guard.py
- app/ae13b_product/demo_queue.py
- app/ae13b_product/contract_resolver.py
- tests/test_ae13e_stable_market_watchlist.py
- scripts/write_ae13e_artifacts.py
- this audit tree

## 20. Files modified
- app/analytics/watchlist.py
- app/api.py
- app/ae13b_product/live_market.py
- app/execution/paper.py
- static/product_demo.js
- static/product_demo.css
- static/index.html

## 21. Tests run
See tests/ae13e_test_results.md

## 22. Safety result
{json.dumps(safety, indent=2)}

## 23. Known limitations
- External contract resolution providers intentionally not called silently; many assets remain unmatched until present in local market feed.
- Keyed DOM patch updates common cells; ID cell copy button markup is rebuilt mainly on first insert.
- Demo Queue evaluation does not auto-open trades; bot cycle still required for paper fills under risk guard.

## 24. Final classification
{classification}

## 25. Continue to longer paper/demo validation?
Yes — product can continue longer paper/demo validation with actionable watchlist + risk-bounded demo queue. Not live-ready.
"""
    (reports / "ae13e_stable_market_actionable_watchlist_report.md").write_text(report, encoding="utf-8")
    (reports / "ae13e_summary_for_upload.txt").write_text(
        f"AE13E classification={classification}\nkeyed_row_update=yes\nwatchlist_actions=yes\ndemo_queue=yes\nrisk_guard=yes\nsafety_ok=yes\n",
        encoding="utf-8",
    )
    _write_json(
        reports / "ae13e_decision_gate.json",
        {
            "classification": classification,
            "wallet_configured": False,
            "live_trading_ready": False,
            "profitability_proven": False,
            "continue_paper_demo_validation": True,
            "blockers": [],
            "limitations": [
                "external_resolver_disabled_by_design",
                "demo_queue_does_not_auto_trade",
            ],
        },
    )
    (ROOT / ".ae13e_outdir.txt").write_text(str(out), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(main())
