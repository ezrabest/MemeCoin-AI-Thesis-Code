"""Backfill AE11E position economics from paper_orders JSONL — pre-loop only."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.runtime_paper_loop.decimal_money import (
    bps_cost,
    decimal_to_str,
    quantize_price,
    quantize_quantity,
    quantize_usd,
    to_decimal,
)
from app.runtime_paper_loop.types import utc_now_iso

REQUIRED_ECONOMICS = ("entry_price", "quantity", "notional_usd", "cash_debited_usd")

BACKFILL_AUDIT_FIELDS = [
    "audit_timestamp_utc",
    "loop_run_id",
    "invocation_id",
    "position_id",
    "paper_order_id",
    "source_decision_id",
    "pair_address",
    "backfill_status",
    "missing_fields",
    "jsonl_order_found",
    "jsonl_source_file",
    "entry_price",
    "quantity",
    "notional_usd",
    "cost_basis_usd",
    "entry_fee_usd",
    "time_stop_at_utc",
    "severity",
    "notes",
]


def _load_orders_index(paper_trading_dir: Path) -> dict[str, tuple[dict[str, Any], str]]:
    """Index paper orders by paper_order_id across all paper_orders_*.jsonl files."""
    index: dict[str, tuple[dict[str, Any], str]] = {}
    if not paper_trading_dir.is_dir():
        return index
    for path in sorted(paper_trading_dir.glob("paper_orders_*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                oid = rec.get("paper_order_id")
                if oid:
                    index[str(oid)] = (rec, str(path))
    return index


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _compute_tp_sl(
    entry: Decimal,
    take_profit_pct: float,
    stop_loss_pct: float,
) -> tuple[Decimal, Decimal]:
    tp = quantize_price(entry * (Decimal("1") + to_decimal(take_profit_pct) / Decimal("100")))
    sl = quantize_price(entry * (Decimal("1") - to_decimal(stop_loss_pct) / Decimal("100")))
    return tp, sl


def _already_enriched(row: dict[str, Any]) -> bool:
    status = (row.get("economic_enrichment_status") or "").upper()
    if status == "FULL":
        return True
    return all(row.get(f) not in (None, "") for f in ("entry_price", "quantity", "notional_usd"))


def backfill_position_economics(
    state_db: Any,
    *,
    project_root: Path,
    loop_run_id: str,
    invocation_id: str,
    take_profit_pct: float = 20.0,
    stop_loss_pct: float = 10.0,
    time_stop_minutes: float = 240.0,
    entry_fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
    historical_cash_debit_notional_only: bool = True,
) -> dict[str, Any]:
    """
    Backfill missing economics for OPEN positions from paper_orders JSONL.

    Must run after migrate_db_schema and before the runtime loop.
    Historical positions: cash_debited = notional (matching pre-AE11E ledger).
    """
    started = utc_now_iso()
    paper_dir = project_root / "data" / "paper_trading"
    orders_index = _load_orders_index(paper_dir)
    positions = state_db.load_active_positions()
    audit_rows: list[dict[str, Any]] = []
    mismatch_events: list[dict[str, Any]] = []

    success = partial = missing = skipped = 0

    for pos in positions:
        pid = pos.get("position_id")
        oid = pos.get("paper_order_id")
        if _already_enriched(pos):
            skipped += 1
            audit_rows.append(
                {
                    "audit_timestamp_utc": utc_now_iso(),
                    "loop_run_id": loop_run_id,
                    "invocation_id": invocation_id,
                    "position_id": pid,
                    "paper_order_id": oid,
                    "source_decision_id": pos.get("source_decision_id"),
                    "pair_address": pos.get("pair_address"),
                    "backfill_status": "SKIPPED_ALREADY_ENRICHED",
                    "missing_fields": "",
                    "jsonl_order_found": False,
                    "jsonl_source_file": None,
                    "entry_price": pos.get("entry_price"),
                    "quantity": pos.get("quantity"),
                    "notional_usd": pos.get("notional_usd"),
                    "cost_basis_usd": pos.get("cost_basis_usd"),
                    "entry_fee_usd": pos.get("entry_fee_usd"),
                    "time_stop_at_utc": pos.get("time_stop_at_utc"),
                    "severity": "INFO",
                    "notes": "Already enriched",
                }
            )
            continue

        order_tuple = orders_index.get(str(oid)) if oid else None
        if not order_tuple:
            missing += 1
            miss_fields = ",".join(REQUIRED_ECONOMICS)
            state_db.update_position_economics(
                pid,
                {
                    "economic_enrichment_status": "MISSING",
                    "economic_enrichment_missing_fields": miss_fields,
                },
            )
            audit_rows.append(
                {
                    "audit_timestamp_utc": utc_now_iso(),
                    "loop_run_id": loop_run_id,
                    "invocation_id": invocation_id,
                    "position_id": pid,
                    "paper_order_id": oid,
                    "source_decision_id": pos.get("source_decision_id"),
                    "pair_address": pos.get("pair_address"),
                    "backfill_status": "MISSING_JSONL_ORDER",
                    "missing_fields": miss_fields,
                    "jsonl_order_found": False,
                    "jsonl_source_file": None,
                    "entry_price": None,
                    "quantity": None,
                    "notional_usd": None,
                    "cost_basis_usd": None,
                    "entry_fee_usd": None,
                    "time_stop_at_utc": None,
                    "severity": "CRITICAL",
                    "notes": "No matching paper_order_id in paper_orders JSONL",
                }
            )
            mismatch_events.append(
                {
                    "mismatch_type": "POSITION_ECONOMICS_MISSING",
                    "component": "positions",
                    "field_path": f"positions.{pid}.economics",
                    "field": "economic_enrichment_status",
                    "sqlite_value": "MISSING",
                    "source_of_truth": "sqlite_jsonl_reconciliation",
                    "repair_action": "BACKFILL_FAILED_WARNING_ONLY",
                    "severity": "CRITICAL",
                    "notes": f"paper_order_id={oid} missing_fields={miss_fields}",
                    "position_id": pid,
                }
            )
            continue

        order, source_file = order_tuple
        entry = quantize_price(order.get("filled_price_usd") or order.get("requested_price_usd"))
        qty = quantize_quantity(order.get("quantity"))
        notional = quantize_usd(order.get("notional_usd") or 100)
        # Historical opens debited notional only; new fee model recorded separately for audit.
        entry_fee = quantize_usd(0) if historical_cash_debit_notional_only else bps_cost(notional, entry_fee_bps)
        entry_slip = quantize_usd(0) if historical_cash_debit_notional_only else bps_cost(notional, slippage_bps)
        cash_debited = quantize_usd(notional + entry_fee + entry_slip)
        cost_basis = quantize_usd(notional + entry_fee)  # cost basis includes fee when charged
        if historical_cash_debit_notional_only:
            cost_basis = notional
            entry_fee = quantize_usd(0)
            entry_slip = quantize_usd(0)
            cash_debited = notional

        opened = order.get("filled_at_utc") or order.get("created_at_utc") or pos.get("opened_at_utc")
        opened_dt = _parse_dt(opened) or datetime.now(timezone.utc)
        time_stop_at = (opened_dt + timedelta(minutes=float(time_stop_minutes))).isoformat()
        tp, sl = _compute_tp_sl(entry, take_profit_pct, stop_loss_pct) if entry > 0 else (None, None)

        patch = {
            "candidate_id": order.get("candidate_id"),
            "symbol": order.get("symbol"),
            "chain": order.get("chain"),
            "entry_price": decimal_to_str(entry),
            "entry_price_timestamp_utc": order.get("price_timestamp")
            or order.get("price_timestamp_used"),
            "entry_price_source": order.get("price_source") or "paper_order_jsonl",
            "entry_snapshot_id": str(order.get("price_snapshot_id"))
            if order.get("price_snapshot_id") is not None
            else None,
            "notional_usd": decimal_to_str(notional),
            "quantity": decimal_to_str(qty),
            "cost_basis_usd": decimal_to_str(cost_basis),
            "entry_fee_usd": decimal_to_str(entry_fee),
            "entry_slippage_usd": decimal_to_str(entry_slip),
            "cash_debited_usd": decimal_to_str(cash_debited),
            "tp_price": decimal_to_str(tp) if tp is not None else None,
            "sl_price": decimal_to_str(sl) if sl is not None else None,
            "time_stop_at_utc": time_stop_at,
            "take_profit_pct": str(take_profit_pct),
            "stop_loss_pct": str(stop_loss_pct),
            "time_stop_minutes": str(time_stop_minutes),
            "trade_authority": order.get("trade_authority"),
            "not_model_approved": str(order.get("not_model_approved", True)),
            "not_live_approved": str(order.get("not_live_approved", True)),
            "override_type": order.get("override_type"),
            "opened_at_utc": opened or pos.get("opened_at_utc"),
        }
        missing_fields = [f for f in REQUIRED_ECONOMICS if not patch.get(f)]
        if entry <= 0:
            missing_fields.append("entry_price")
        if qty <= 0:
            missing_fields.append("quantity")

        if missing_fields:
            partial += 1
            status = "BACKFILLED_PARTIAL"
            patch["economic_enrichment_status"] = "PARTIAL"
            patch["economic_enrichment_missing_fields"] = ",".join(sorted(set(missing_fields)))
            severity = "WARNING"
            mismatch_events.append(
                {
                    "mismatch_type": "POSITION_ECONOMICS_MISSING",
                    "component": "positions",
                    "field_path": f"positions.{pid}.economics",
                    "field": "economic_enrichment_status",
                    "sqlite_value": "PARTIAL",
                    "source_of_truth": "sqlite_jsonl_reconciliation",
                    "repair_action": "BACKFILLED_FROM_JSONL",
                    "severity": "WARNING",
                    "notes": f"paper_order_id={oid} missing_fields={patch['economic_enrichment_missing_fields']}",
                    "position_id": pid,
                }
            )
        else:
            success += 1
            status = "BACKFILLED_FULL"
            patch["economic_enrichment_status"] = "FULL"
            patch["economic_enrichment_missing_fields"] = ""
            severity = "INFO"
            mismatch_events.append(
                {
                    "mismatch_type": "POSITION_ECONOMICS_MISSING",
                    "component": "positions",
                    "field_path": f"positions.{pid}.economics",
                    "field": "economic_enrichment_status",
                    "sqlite_value": "FULL",
                    "source_of_truth": "sqlite_jsonl_reconciliation",
                    "repair_action": "BACKFILLED_FROM_JSONL",
                    "severity": "INFO",
                    "notes": f"paper_order_id={oid} backfilled from {source_file}",
                    "position_id": pid,
                }
            )

        state_db.update_position_economics(pid, patch)
        audit_rows.append(
            {
                "audit_timestamp_utc": utc_now_iso(),
                "loop_run_id": loop_run_id,
                "invocation_id": invocation_id,
                "position_id": pid,
                "paper_order_id": oid,
                "source_decision_id": pos.get("source_decision_id"),
                "pair_address": pos.get("pair_address"),
                "backfill_status": status
                if status != "BACKFILLED_PARTIAL"
                else (
                    "MISSING_REQUIRED_ECONOMICS"
                    if missing_fields
                    else "BACKFILLED_PARTIAL"
                ),
                "missing_fields": patch.get("economic_enrichment_missing_fields") or "",
                "jsonl_order_found": True,
                "jsonl_source_file": source_file,
                "entry_price": patch.get("entry_price"),
                "quantity": patch.get("quantity"),
                "notional_usd": patch.get("notional_usd"),
                "cost_basis_usd": patch.get("cost_basis_usd"),
                "entry_fee_usd": patch.get("entry_fee_usd"),
                "time_stop_at_utc": patch.get("time_stop_at_utc"),
                "severity": severity,
                "notes": "Historical cash_debited=notional_only"
                if historical_cash_debit_notional_only
                else "Fees included in cash_debited",
            }
        )

    completed = utc_now_iso()
    audit_path = project_root / "audits" / "ae11_position_backfill_audit.csv"
    _write_backfill_audit(audit_path, audit_rows)

    return {
        "backfill_started_at_utc": started,
        "backfill_completed_at_utc": completed,
        "backfill_position_count": len(positions),
        "backfill_success_count": success,
        "backfill_partial_count": partial,
        "backfill_missing_count": missing,
        "backfill_skipped_count": skipped,
        "audit_path": str(audit_path),
        "mismatch_events": mismatch_events,
        "missing_position_ids": [
            r["position_id"] for r in audit_rows if r["backfill_status"] in (
                "MISSING_JSONL_ORDER",
                "MISSING_REQUIRED_ECONOMICS",
            )
        ],
    }


def _write_backfill_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=BACKFILL_AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
