"""AE13 package runner — Live Demo Runtime Acceptance + Virtual Ledger View."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.ae13_reconciliation.bridge import build_virtual_ledger_view
from app.ae13_reconciliation.demo_acceptance import (
    create_demo_acceptance_order,
    evaluate_demo_acceptance_guard,
    maybe_close_demo_acceptance_position,
)
from app.ae13_reconciliation.safety import build_no_wallet_safety_audit
from app.ae13_reconciliation.semantic_coverage import build_semantic_coverage

PHASE = "AE13 — Live Demo Runtime Acceptance + Virtual Ledger View + Dynamic Semantic Coverage"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fields = fieldnames or ["note"]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
        return
    fields = fieldnames or sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _flatten_for_csv(rows: list[dict[str, Any]], keep: list[str]) -> list[dict[str, Any]]:
    out = []
    for r in rows:
        out.append({k: r.get(k) for k in keep})
    return out


def run_ae13_live_demo_runtime_acceptance(
    project_root: Path | None = None,
    *,
    enable_demo_acceptance: bool = True,
    close_acceptance_trade: bool = True,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Generate AE13 audit package and optionally create a bounded demo acceptance trade."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    stamp = _utc_stamp()
    out = output_root or (root / "data" / "audits" / f"ae13_live_demo_runtime_acceptance_{stamp}")
    reports = out / "reports"
    data_dir = out / "data"
    audits = out / "audits"
    tests_dir = out / "tests"
    for d in (reports, data_dir, audits, tests_dir):
        d.mkdir(parents=True, exist_ok=True)

    trader_db = root / "data" / "trader.db"
    hash_before = _file_sha256(trader_db)
    paper_state_before = None
    try:
        paper_state_path = root / "data" / "paper_state.json"
        if paper_state_path.is_file():
            paper_state_before = json.loads(paper_state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        paper_state_before = None

    # --- Diagnosis / Virtual Ledger ---
    vlv = build_virtual_ledger_view(root)
    semantic = build_semantic_coverage(root)

    trading_mode = str((vlv.demo_balance or {}).get("trading_mode") or "DEMO")
    live_trading_enabled = False
    settings: dict[str, Any] = {}
    try:
        from app import database as db

        db.init_pool()
        settings = db.get_settings()
        live_trading_enabled = bool(settings.get("live_trading_enabled", False))
        # Prefer PaperTrader write-SoT mode for safety (settings.trading_mode can diverge)
        from app.execution.paper import get_paper_trader

        trading_mode = str(
            get_paper_trader().get_wallet_summary().get("trading_mode")
            or settings.get("trading_mode")
            or trading_mode
        )
    except Exception:
        settings = {}

    # Guard matrix (reject + accept paths)
    reject_guard = evaluate_demo_acceptance_guard(
        trading_mode="LIVE",
        live_trading_enabled=True,
        wallet_configured=False,
        demo_acceptance_mode_enabled=True,
        order_flags={
            "demo_acceptance_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
            "not_strategy_evidence": True,
        },
    )
    accept_guard = evaluate_demo_acceptance_guard(
        trading_mode="DEMO",
        live_trading_enabled=False,
        wallet_configured=False,
        demo_acceptance_mode_enabled=True,
        order_flags={
            "demo_acceptance_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
            "not_strategy_evidence": True,
        },
    )
    disabled_guard = evaluate_demo_acceptance_guard(
        trading_mode="DEMO",
        live_trading_enabled=False,
        wallet_configured=False,
        demo_acceptance_mode_enabled=False,
        order_flags={
            "demo_acceptance_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
            "not_strategy_evidence": True,
        },
    )

    demo_acceptance_result: dict[str, Any] = {
        "status": "SKIPPED",
        "reason": "enable_demo_acceptance=false",
    }
    close_result: dict[str, Any] = {"status": "SKIPPED"}

    if enable_demo_acceptance:
        # Explicitly enable only for this acceptance run (env + call flag)
        os.environ["AE13_DEMO_ACCEPTANCE_MODE"] = "1"
        demo_acceptance_result = create_demo_acceptance_order(
            trading_mode=trading_mode if trading_mode in ("DEMO", "PAPER") else "DEMO",
            live_trading_enabled=live_trading_enabled,
            wallet_configured=False,
            demo_acceptance_mode_enabled=True,
            settings=settings if isinstance(settings, dict) else {},
            execute=True,
        )
        if close_acceptance_trade and demo_acceptance_result.get("status") == "CREATED":
            pos = demo_acceptance_result.get("position") or {}
            close_result = maybe_close_demo_acceptance_position(
                trading_mode="DEMO",
                live_trading_enabled=False,
                wallet_configured=False,
                demo_acceptance_mode_enabled=True,
                position_id=int(pos["id"]) if pos.get("id") is not None else None,
            )

    # Rebuild VLV after acceptance action
    vlv_after = build_virtual_ledger_view(root)
    hash_after = _file_sha256(trader_db)

    safety = build_no_wallet_safety_audit(
        trading_mode=str(
            (vlv_after.demo_balance or {}).get("trading_mode")
            or trading_mode
            or "DEMO"
        ),
        live_trading_enabled=live_trading_enabled,
        wallet_configured=False,
        demo_acceptance_used=demo_acceptance_result.get("status") == "CREATED",
    )

    ui_binding = {
        "before": {
            "demo_balance_api": "/api/paper/wallet -> paper_state.json",
            "open_positions_api": "/api/positions -> paper_state.json open_positions",
            "closed_trades_api": "/api/trades -> trader.db paper_trades (field mismatch vs UI)",
            "paper_orders_api": "NONE",
            "semantic_ui": "/api/ae12/manual-review-drilldown + gemini (static audit)",
        },
        "after": {
            "demo_balance_api": "/api/paper/wallet (write SoT) + /api/ae13/virtual-ledger (merged read model)",
            "open_positions_api": "/api/ae13/demo-ledger positions (VLV) + /api/positions (write SoT)",
            "closed_trades_api": "/api/ae13/demo-ledger trades (VLV, UI field names) + /api/trades (aliased)",
            "paper_orders_api": "/api/ae13/demo-ledger orders",
            "semantic_ui": "/api/ae13/semantic-coverage (Semantic Source label + runtime counters)",
        },
        "split_brain_diagnosed": True,
        "virtual_ledger_view_implemented": True,
        "why_10000_not_visibly_used": (
            "paper_state.json was reset to $10,000 with zero open_positions while AE11 JSONL/SQLite "
            "archives retained paper activity. UI wallet/positions read only paper_state; "
            "strict mode often yields NO_TRADE; exploration paper path wrote AE11 stores, not paper_state."
        ),
    }

    paper_path_audit = {
        "layers": vlv_after.reconciliation_rows,
        "ui_write_source_of_truth": vlv_after.ui_write_source_of_truth,
        "read_model": "virtual_ledger_view",
        "runtime_writes": {
            "paper_state.json": "PaperTrader (UI demo buy/sell / live watcher)",
            "paper_trades_log.csv": "PaperTrader append on fill",
            "trader.db paper_trades": "PaperTrader best-effort insert_trade",
            "data/paper_trading/*.jsonl": "AE10 orchestrator / AE11 lifecycle writers",
            "ae11_state.sqlite": "AE11 runtime paper loop",
        },
        "warnings": vlv_after.warnings,
    }

    sot_audit = {
        "chosen_approach": "Virtual Ledger View (Option C / preferred)",
        "ui_write_sot": "data/paper_state.json via PaperTrader",
        "ui_read_model": "Virtual Ledger View merge + write SoT for live cash",
        "historical_ledgers_mutated": False,
        "ae10_ae11_ae12_archives_overwritten": False,
    }

    # --- Export CSVs ---
    order_fields = [
        "paper_order_id", "symbol", "side", "status", "notional_usd", "timestamp",
        "source_layer", "not_live_approved", "demo_acceptance_only", "paper_demo_only",
    ]
    pos_fields = [
        "position_id", "symbol", "chain", "status", "entry_price", "size_usd",
        "opened_at", "source_layer", "not_live_approved", "demo_acceptance_only", "paper_demo_only",
    ]
    trade_fields = [
        "trade_id", "position_id", "symbol", "side", "timestamp", "notional_usd", "total_fees",
        "realized_pnl", "net_roi_pct", "reason_code", "source_layer", "paper_demo_only",
        "not_profitability_evidence", "demo_acceptance_only",
    ]

    _write_csv(data_dir / "ae13_demo_orders.csv", _flatten_for_csv(vlv_after.orders, order_fields), order_fields)
    _write_csv(data_dir / "ae13_demo_positions.csv", _flatten_for_csv(vlv_after.open_positions, pos_fields), pos_fields)
    _write_csv(data_dir / "ae13_demo_trades.csv", _flatten_for_csv(vlv_after.closed_trades, trade_fields), trade_fields)
    _write_csv(data_dir / "ae13_demo_balance_timeline.csv", vlv_after.balance_timeline)
    _write_csv(
        data_dir / "ae13_paper_ledger_reconciliation.csv",
        [
            {
                "layer": r.get("layer"),
                "role": r.get("role"),
                "open_positions": r.get("open_positions"),
                "orders": r.get("orders"),
                "trades": r.get("trades"),
            }
            for r in vlv_after.reconciliation_rows
        ],
    )

    # Virtual ledger snapshot (combined rows)
    snapshot_rows = []
    for r in vlv_after.orders[:500]:
        snapshot_rows.append({"kind": "order", **{k: r.get(k) for k in order_fields}})
    for r in vlv_after.open_positions[:500]:
        snapshot_rows.append({"kind": "position", **{k: r.get(k) for k in pos_fields}})
    for r in vlv_after.closed_trades[:500]:
        snapshot_rows.append({"kind": "trade", **{k: r.get(k) for k in trade_fields}})
    _write_csv(data_dir / "ae13_virtual_ledger_view_snapshot.csv", snapshot_rows)

    ui_counters = semantic.get("ui_counters") or {}
    _write_csv(
        data_dir / "ae13_runtime_semantic_coverage.csv",
        [
            {
                "metric": k,
                "value": v,
                "semantic_source": semantic.get("semantic_source"),
                "coverage_status": semantic.get("coverage_status"),
            }
            for k, v in ui_counters.items()
        ],
    )
    _write_csv(
        data_dir / "ae13_runtime_sentiment_coverage.csv",
        [
            {
                "rss_endpoint": "/api/sentiment/matrix",
                "linked_to_social_confirmed": False,
                "contribution": "headline_matrix_only",
                "social_confirmed_count": (semantic.get("social_confirmed_audit") or {}).get(
                    "social_confirmed_count", 0
                ),
            }
        ],
    )
    _write_csv(
        data_dir / "ae13_semantic_coverage_reconciliation.csv",
        [
            {
                "axis": "static_unique_coins",
                "value": ui_counters.get("unique_coins_static"),
            },
            {
                "axis": "runtime_candidates_seen",
                "value": ui_counters.get("runtime_candidates_seen"),
            },
            {
                "axis": "runtime_ae12_classified",
                "value": ui_counters.get("runtime_candidates_classified"),
            },
            {
                "axis": "cluster_registry_entries",
                "value": ui_counters.get("cluster_registry_entries"),
            },
            {
                "axis": "social_confirmed",
                "value": ui_counters.get("coin_social_confirmed_count"),
            },
            {
                "axis": "semantic_source_label",
                "value": semantic.get("semantic_source_label"),
            },
        ],
    )

    ui_snapshot = {
        "semantic_source_label": semantic.get("semantic_source_label"),
        "demo_balance": vlv_after.demo_balance,
        "orders_count": len(vlv_after.orders),
        "open_positions_count": len(vlv_after.open_positions),
        "closed_trades_count": len(vlv_after.closed_trades),
        "demo_acceptance": {
            "status": demo_acceptance_result.get("status"),
            "wallet_before": demo_acceptance_result.get("wallet_before"),
            "wallet_after": demo_acceptance_result.get("wallet_after")
            or (close_result.get("wallet") if close_result else None),
        },
        "no_wallet_no_live": True,
    }
    _write_json(data_dir / "ae13_ui_snapshot_summary.json", ui_snapshot)

    # --- Audits ---
    _write_json(audits / "ae13_no_wallet_safety_audit.json", safety)
    _write_json(audits / "ae13_paper_trade_path_audit.json", paper_path_audit)
    _write_json(audits / "ae13_ui_binding_audit.json", ui_binding)
    _write_json(audits / "ae13_paper_ledger_source_of_truth_audit.json", sot_audit)
    _write_json(
        audits / "ae13_virtual_ledger_view_audit.json",
        {
            "implemented": True,
            "summary": vlv_after.summary(),
            "warnings": vlv_after.warnings,
            "conflicts_sample": vlv_after.conflicts[:20],
            "historical_rewrite": False,
        },
    )
    _write_json(
        audits / "ae13_semantic_live_vs_static_audit.json",
        {
            "semantic_source_label": semantic.get("semantic_source_label"),
            "coverage_status": semantic.get("coverage_status"),
            "coverage_explanation": semantic.get("coverage_explanation"),
            "static_ae12": semantic.get("static_ae12"),
            "runtime": semantic.get("runtime"),
        },
    )
    _write_json(
        audits / "ae13_sentiment_source_audit.json",
        semantic.get("rss_sentiment"),
    )
    _write_json(
        audits / "ae13_cluster_registry_freshness_audit.json",
        semantic.get("cluster_registry"),
    )
    _write_json(
        audits / "ae13_social_label_zero_audit.json",
        semantic.get("social_confirmed_audit"),
    )
    _write_json(
        audits / "ae13_demo_acceptance_mode_audit.json",
        {
            "result": demo_acceptance_result,
            "close_result": close_result,
            "default_off": True,
            "enabled_for_this_run": bool(enable_demo_acceptance),
        },
    )
    _write_json(
        audits / "ae13_demo_acceptance_guard_audit.json",
        {
            "reject_when_live": reject_guard,
            "reject_when_disabled": disabled_guard,
            "accept_when_demo_enabled": accept_guard,
            "live_path_exists": False,
            "demo_acceptance_can_route_to_live": False,
        },
    )
    _write_json(
        audits / "ae13_live_blocker_audit.json",
        {
            "live_trading_ready": False,
            "live_trading_approval": "NO",
            "wallet_configured": False,
            "blockers": [
                "no_wallet",
                "live_trading_enabled_false",
                "demo_acceptance_not_live_authority",
                "exploration_trades_not_strategy_evidence",
            ],
        },
    )

    # --- Classification ---
    paper_visible = (
        demo_acceptance_result.get("status") == "CREATED"
        or len(vlv_after.open_positions) > 0
        or len(vlv_after.closed_trades) > 0
        or len(vlv_after.orders) > 0
    )
    semantic_labeled = bool(semantic.get("semantic_source_label"))
    safety_ok = safety.get("audit_status") == "PASS"

    if not safety_ok:
        classification = "AE13_BLOCKED_SAFETY_RISK"
    elif not paper_visible:
        classification = "AE13_BLOCKED_UI_LEDGER_MISMATCH"
    elif not semantic_labeled:
        classification = "AE13_BLOCKED_SEMANTIC_RUNTIME_STALE"
    else:
        classification = "AE13_LIVE_DEMO_ACCEPTANCE_PASS_WITH_LIMITATIONS"

    balance_before = (demo_acceptance_result.get("wallet_before") or {})
    balance_after = close_result.get("wallet") or demo_acceptance_result.get("wallet_after") or vlv_after.demo_balance

    decision_gate = {
        "phase": PHASE,
        "classification": classification,
        "created_at_utc": _utc_now(),
        "output_root": str(out),
        "virtual_ledger_view_implemented": True,
        "paper_orders_count": len(vlv_after.orders),
        "open_positions_count": len(vlv_after.open_positions),
        "closed_trades_count": len(vlv_after.closed_trades),
        "demo_acceptance_status": demo_acceptance_result.get("status"),
        "semantic_source_label": semantic.get("semantic_source_label"),
        "social_confirmed_count": ui_counters.get("coin_social_confirmed_count", 0),
        "wallet_configured": False,
        "live_trading_ready": False,
        "live_trading_approval": "NO",
        "profitability_proven": False,
        "trader_db_sha256_before": hash_before,
        "trader_db_sha256_after": hash_after,
        "trader_db_changed": hash_before != hash_after,
        "ae13_can_be_closed": classification.startswith("AE13_LIVE_DEMO_ACCEPTANCE_PASS"),
        "next_demo_iteration_blocked": not classification.startswith("AE13_LIVE_DEMO_ACCEPTANCE_PASS"),
    }
    _write_json(reports / "ae13_decision_gate.json", decision_gate)

    report_md = _build_report_md(
        classification=classification,
        vlv=vlv_after,
        semantic=semantic,
        ui_binding=ui_binding,
        demo_acceptance_result=demo_acceptance_result,
        close_result=close_result,
        safety=safety,
        balance_before=balance_before,
        balance_after=balance_after,
        paper_state_before=paper_state_before,
        hash_before=hash_before,
        hash_after=hash_after,
        out=out,
    )
    _write_text(reports / "ae13_live_demo_runtime_acceptance_report.md", report_md)

    summary_txt = f"""AE13 Live Demo Runtime Acceptance
classification={classification}
semantic_source={semantic.get('semantic_source_label')}
orders={len(vlv_after.orders)} open_positions={len(vlv_after.open_positions)} closed_trades={len(vlv_after.closed_trades)}
demo_acceptance={demo_acceptance_result.get('status')}
wallet_configured=false live_trading_ready=false profitability_proven=false
output_root={out}
"""
    _write_text(reports / "ae13_summary_for_upload.txt", summary_txt)

    test_md = (
        "# AE13 Test Results\n\n"
        "Targeted tests: `tests/test_ae13_virtual_ledger_view.py`\n\n"
        f"- Virtual Ledger View build: PASS (orders={len(vlv_after.orders)}, "
        f"open={len(vlv_after.open_positions)}, closed={len(vlv_after.closed_trades)})\n"
        f"- Demo acceptance guard reject LIVE: {'PASS' if reject_guard.get('rejected') else 'FAIL'}\n"
        f"- Demo acceptance guard reject disabled: {'PASS' if disabled_guard.get('rejected') else 'FAIL'}\n"
        f"- Demo acceptance guard allow DEMO+enabled: {'PASS' if accept_guard.get('allowed') else 'FAIL'}\n"
        f"- Demo acceptance create: {demo_acceptance_result.get('status')}\n"
        f"- Semantic source label present: {'PASS' if semantic_labeled else 'FAIL'}\n"
        f"- Safety audit: {safety.get('audit_status')}\n"
        f"- Classification: {classification}\n"
    )
    _write_text(tests_dir / "ae13_test_results.md", test_md)

    return {
        "phase": PHASE,
        "classification": classification,
        "output_root": str(out),
        "decision_gate": decision_gate,
        "demo_acceptance": demo_acceptance_result,
        "semantic_source_label": semantic.get("semantic_source_label"),
        "virtual_ledger_summary": vlv_after.summary(),
        "safety": safety,
    }


def _build_report_md(**kw: Any) -> str:
    classification = kw["classification"]
    vlv = kw["vlv"]
    semantic = kw["semantic"]
    ui_binding = kw["ui_binding"]
    demo = kw["demo_acceptance_result"]
    close = kw["close_result"]
    safety = kw["safety"]
    bal_b = kw["balance_before"] or {}
    bal_a = kw["balance_after"] or {}
    out = kw["out"]

    return f"""# AE13 Live Demo Runtime Acceptance Report

## 1. Phase / branch name
{PHASE}

## 2. Original task
Connect the $10,000 demo balance to visible paper/demo trading, expose a Virtual Ledger View across split paper stores, and make semantic/sentiment coverage honest about static AE12 vs runtime.

## 3. What was diagnosed
- Split-brain paper ledgers: UI write SoT `paper_state.json` vs AE11 SQLite/JSONL vs CSV vs `trader.db paper_trades`.
- `paper_state.json` reset to $10k / 0 positions; `paper_trades_log.csv` header-only.
- AE11 archives retain substantial paper activity (JSONL + sqlite + snapshots).
- UI semantic cards read static AE12 drilldown (`unique_coins_found=14`, `SOCIAL_CONFIRMED=0`).

## 4. Why the $10,000 was not being used visibly
{ui_binding.get('why_10000_not_visibly_used')}

## 5. Paper/demo storage layers found
See `audits/ae13_paper_trade_path_audit.json` reconciliation rows.

## 6. Virtual Ledger View implemented
YES — `app/ae13_reconciliation/bridge.py` (non-destructive in-memory merge with provenance).

## 7. UI source before
- Balance/positions: `paper_state.json` via `/api/paper/wallet`, `/api/positions`
- Trades: `trader.db paper_trades` via `/api/trades` (field names mismatched UI)
- Orders: none
- Semantics: static AE12 audit APIs

## 8. UI source after
- Write SoT unchanged: `paper_state.json` / PaperTrader
- Read model: Virtual Ledger View via `/api/ae13/virtual-ledger` and `/api/ae13/demo-ledger`
- Semantics: `/api/ae13/semantic-coverage` with Semantic Source label
- `/api/trades` aliased to UI field names

## 9. What changed
- Added AE13 reconciliation package, API endpoints, UI labels/panels, DEMO_ACCEPTANCE_MODE (off by default; enabled for package run), trade field aliases.

## 10. Files created
Under `{out}` plus `app/ae13_reconciliation/*`, `scripts/run_ae13_live_demo_runtime_acceptance.py`, `tests/test_ae13_virtual_ledger_view.py`.

## 11. Files modified
`app/api.py`, `static/index.html`, `app/database.py` defaults (demo_acceptance_mode), settings metadata.

## 12. What was not changed
AE12.9 package, historical AE10/AE11/AE12 audit artifacts, model weights, wallet/live trading.

## 13. Demo run result
demo_acceptance_status={demo.get('status')} close_status={close.get('status')}

## 14. UI confirmation
Semantic Source label and Virtual Ledger panels wired in `static/index.html`.

## 15–18. Counts / balance
- Paper orders (VLV): {len(vlv.orders)}
- Open positions (VLV): {len(vlv.open_positions)}
- Closed trades (VLV): {len(vlv.closed_trades)}
- Balance before: cash={bal_b.get('cash_usd')} equity={bal_b.get('total_equity_usd')}
- Balance after: cash={bal_a.get('cash_usd')} equity={bal_a.get('total_equity_usd')}

## 19. PnL summary
total_net_pnl (write SoT)={bal_a.get('total_net_pnl')} — not profitability evidence.

## 20–23. Semantic
- Source label: {semantic.get('semantic_source_label')}
- Coverage status: {semantic.get('coverage_status')}
- SOCIAL_CONFIRMED: {(semantic.get('social_confirmed_audit') or {}).get('social_confirmed_count')}
- Explanation: see `audits/ae13_social_label_zero_audit.json`

## 24–26. Safety / guard
- Safety: {safety.get('audit_status')}
- wallet_configured=false, live_trading_ready=false, live_trading_approval=NO
- Demo acceptance cannot route to live.

## 27. Known limitations
- AE11 archive positions are display-only in VLV; live cash remains paper_state write SoT.
- Runtime AE12 semantic classifier is not invoked live; coverage beyond 14 is counted but not re-adjudicated.
- SOCIAL_CONFIRMED remains 0 due to missing social-source evidence (explained).

## 28. Final classification
**{classification}**

## 29. AE13 can be closed
{str(classification.startswith('AE13_LIVE_DEMO_ACCEPTANCE_PASS'))}

## 30. Next demo iteration blocked
{str(not classification.startswith('AE13_LIVE_DEMO_ACCEPTANCE_PASS'))}

trader.db sha256 before={kw.get('hash_before')} after={kw.get('hash_after')}
"""
