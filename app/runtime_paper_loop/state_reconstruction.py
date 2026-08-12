"""Reconstruct paper ledger state from AE11-owned logs and state index."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.decision.persistence import read_jsonl_records_safe
from app.runtime_paper_loop.types import ReconstructedAccountState


def _read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records, _ = read_jsonl_records_safe(path)
    return records


def ledger_reconstruction(
    *,
    paper_orders_paths: list[Path],
    paper_positions_paths: list[Path],
    paper_trades_paths: list[Path],
    state_db: Any | None = None,
    starting_balance_usd: float = 10_000.0,
    allow_negative_cash: bool = False,
    allow_duplicate_pair: bool = False,
) -> ReconstructedAccountState:
    """Reconstruct account state from append-only logs + optional SQLite index."""
    orders: list[dict[str, Any]] = []
    positions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []

    for p in paper_orders_paths:
        orders.extend(_read_jsonl_if_exists(p))
    for p in paper_positions_paths:
        positions.extend(_read_jsonl_if_exists(p))
    for p in paper_trades_paths:
        trades.extend(_read_jsonl_if_exists(p))

    state = ReconstructedAccountState()
    state.cash_balance_usd = starting_balance_usd
    mismatches: list[str] = []

    order_by_id = {o["paper_order_id"]: o for o in orders if o.get("paper_order_id")}
    open_positions: dict[str, dict[str, Any]] = {}
    closed_positions: list[dict[str, Any]] = []
    processed_ids: set[str] = set()
    pair_locks: dict[str, str] = {}

    for order in orders:
        sid = order.get("source_decision_id")
        if sid:
            processed_ids.add(str(sid))
        status = order.get("status", "")
        if status == "PAPER_FILLED":
            notional = float(order.get("notional_usd") or 0)
            state.cash_balance_usd -= notional
            state.reserved_cash_usd += notional

    for pos in positions:
        pid = pos.get("position_id", "")
        pair = pos.get("pair_address", "")
        status = pos.get("status", "OPEN")
        if status == "OPEN":
            if pair in open_positions and not allow_duplicate_pair:
                mismatches.append(f"duplicate_open_position:{pair}")
            open_positions[pid] = pos
            if pair:
                pair_locks[pair] = pid
        else:
            closed_positions.append(pos)

    for trade in trades:
        notional = float(trade.get("notional_usd") or 0)
        pnl = float(trade.get("realized_pnl_usd") or 0)
        state.cash_balance_usd += notional + pnl
        state.reserved_cash_usd -= notional
        state.realized_pnl_usd += pnl
        state.gross_pnl_usd += pnl
        state.net_pnl_usd += pnl
        pid = trade.get("position_id")
        if pid in open_positions:
            del open_positions[pid]
            pair = trade.get("pair_address") or open_positions.get(pid, {}).get("pair_address", "")
            if pair in pair_locks and pair_locks[pair] == pid:
                del pair_locks[pair]

    for order_id, order in order_by_id.items():
        if order.get("status") == "PAPER_FILLED":
            has_pos = any(p.get("paper_order_id") == order_id for p in positions)
            if not has_pos and order_id not in {t.get("paper_order_id") for t in trades}:
                mismatches.append(f"fill_without_position:{order_id}")

    for trade in trades:
        oid = trade.get("paper_order_id")
        if oid and oid not in order_by_id:
            mismatches.append(f"trade_without_order:{trade.get('trade_id')}")

    if state_db is not None:
        state.processed_decision_count_from_db = state_db.processed_count()
        db_open = {p["position_id"]: p for p in state_db.load_active_positions()}
        for pid, pos in db_open.items():
            if pid not in open_positions:
                open_positions[pid] = pos
        state.cooldowns = state_db.load_cooldowns()

    if state.cash_balance_usd < 0 and not allow_negative_cash:
        mismatches.append("negative_cash_balance")

    state.open_positions = list(open_positions.values())
    state.closed_positions = closed_positions
    state.active_pair_locks = pair_locks
    state.processed_decision_ids = processed_ids
    state.mismatches = mismatches
    state.reconstruction_status = "MISMATCH" if mismatches else "OK"
    state.no_wallet_path = True
    return state


def verify_checkpoint_vs_reconstruction(
    checkpoint: dict[str, Any] | None,
    reconstructed: ReconstructedAccountState,
) -> list[str]:
    """Compare checkpoint convenience state against reconstructed truth."""
    if not checkpoint:
        return []
    mismatches: list[str] = []
    ck_cash = float(checkpoint.get("cash_balance", 0))
    if abs(ck_cash - reconstructed.cash_balance_usd) > 0.01:
        mismatches.append(f"cash_mismatch:checkpoint={ck_cash},reconstructed={reconstructed.cash_balance_usd}")
    ck_positions = set(checkpoint.get("active_position_ids") or [])
    recon_positions = {p.get("position_id") for p in reconstructed.open_positions}
    if ck_positions != recon_positions:
        mismatches.append("active_position_ids_mismatch")
    ck_pairs = set(checkpoint.get("active_pair_keys") or [])
    recon_pairs = set(reconstructed.active_pair_locks.keys())
    if ck_pairs != recon_pairs:
        mismatches.append("active_pair_keys_mismatch")
    return mismatches


def write_state_reconstruction_audit(
    path: Path,
    reconstructed: ReconstructedAccountState,
    checkpoint_mismatches: list[str],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "field,value",
        f"reconstruction_status,{reconstructed.reconstruction_status}",
        f"cash_balance_usd,{reconstructed.cash_balance_usd}",
        f"reserved_cash_usd,{reconstructed.reserved_cash_usd}",
        f"open_position_count,{len(reconstructed.open_positions)}",
        f"closed_position_count,{len(reconstructed.closed_positions)}",
        f"processed_decision_count,{len(reconstructed.processed_decision_ids)}",
        f"realized_pnl_usd,{reconstructed.realized_pnl_usd}",
        f"no_wallet_path,{reconstructed.no_wallet_path}",
    ]
    for m in reconstructed.mismatches + checkpoint_mismatches:
        rows.append(f"mismatch,{m}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path
