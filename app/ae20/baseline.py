"""Pre-AE20 paper/demo baseline partition (hard separation from AE20-created rows)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.consensus.serialization import write_csv, write_json


def _safe_row(obj: dict[str, Any], *, baseline_class: str) -> dict[str, Any]:
    """Carry baseline context without requiring AE20 identity/decision fields."""
    row = dict(obj or {})
    # Never KeyError on missing AE20 fields — fill blanks.
    for key in (
        "coin_id",
        "decision_ref_id",
        "order_id",
        "candidate_id",
        "position_id",
        "pair_address",
        "provider_pair_url_exact",
        "canonical_market_identity",
        "price_source_key",
        "symbol",
        "chain",
    ):
        row.setdefault(key, row.get(key) or "")
    row["preexisting_baseline"] = True
    row["created_during_ae20"] = False
    row["baseline_class"] = baseline_class
    row["excluded_from_ae20_created_pnl"] = True
    row["excluded_from_ae20_orphan_checks"] = True
    return row


def load_paper_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def load_paper_trades_log(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        from app.consensus.serialization import read_csv_dicts

        return [dict(r) for r in read_csv_dicts(path)]
    except OSError:
        return []


def snapshot_preexisting_baseline(
    project_root: Path,
    data_dir: Path,
    audits_dir: Path,
    *,
    paper_state_path: Path | None = None,
    paper_trades_path: Path | None = None,
    paper_orders_path: Path | None = None,
) -> dict[str, Any]:
    """Snapshot pre-run positions/trades/orders before AE20 cycle processing."""
    project_root = project_root.resolve()
    state_path = paper_state_path or (project_root / "data" / "paper_state.json")
    trades_path = paper_trades_path or (project_root / "data" / "paper_trades_log.csv")
    orders_path = paper_orders_path  # optional; may not exist

    state = load_paper_state(state_path)
    open_positions = list(state.get("open_positions") or [])
    position_rows = [
        _safe_row(dict(p), baseline_class="PREEXISTING_POSITION_BASELINE")
        for p in open_positions
    ]

    trade_rows_raw = load_paper_trades_log(trades_path)
    trade_rows = [
        _safe_row(dict(t), baseline_class="PREEXISTING_TRADE_BASELINE")
        for t in trade_rows_raw
    ]

    order_rows: list[dict[str, Any]] = []
    orders_available = False
    if orders_path and Path(orders_path).is_file():
        orders_available = True
        try:
            from app.consensus.serialization import read_csv_dicts

            order_rows = [
                _safe_row(dict(o), baseline_class="PREEXISTING_ORDER_BASELINE")
                for o in read_csv_dicts(Path(orders_path))
            ]
        except OSError:
            order_rows = []

    # Also capture closed_trades count / cash from paper_state as context.
    state_snapshot = {
        "paper_state_path": str(state_path.resolve()),
        "starting_capital": state.get("starting_capital"),
        "cash_usd": state.get("cash_usd"),
        "next_position_id": state.get("next_position_id"),
        "closed_trades": state.get("closed_trades"),
        "total_net_pnl": state.get("total_net_pnl"),
        "cumulative_total_fees": state.get("cumulative_total_fees"),
        "trading_mode": state.get("trading_mode"),
        "open_positions_count": len(open_positions),
        "snapshot_note": (
            "Pre-AE20 baseline only. Never join into AE20-created "
            "candidate/decision/PnL/orphan checks."
        ),
    }

    pos_csv = data_dir / "ae20_preexisting_positions_baseline.csv"
    trades_csv = data_dir / "ae20_preexisting_trades_baseline.csv"
    orders_csv = data_dir / "ae20_preexisting_orders_baseline.csv"
    write_csv(pos_csv, position_rows)
    write_csv(trades_csv, trade_rows)
    write_csv(orders_csv, order_rows)

    audit = {
        "preexisting_baseline_partition": True,
        "paper_state_path": str(state_path.resolve()),
        "paper_trades_path": str(trades_path.resolve()) if trades_path.is_file() else None,
        "paper_orders_path": str(Path(orders_path).resolve()) if orders_available else None,
        "preexisting_positions_count": len(position_rows),
        "preexisting_trades_count": len(trade_rows),
        "preexisting_orders_count": len(order_rows),
        "orders_baseline_available": orders_available,
        "paper_state_snapshot": state_snapshot,
        "excluded_from_ae20_created_pnl": True,
        "excluded_from_ae20_orphan_checks": True,
        "excluded_from_ae20_candidate_decision_joins": True,
        "artifacts": {
            "positions": str(pos_csv),
            "trades": str(trades_csv),
            "orders": str(orders_csv),
        },
    }
    write_json(audits_dir / "ae20_preexisting_position_baseline_audit.json", audit)
    return {
        "positions": position_rows,
        "trades": trade_rows,
        "orders": order_rows,
        "paper_state_snapshot": state_snapshot,
        "audit": audit,
    }
