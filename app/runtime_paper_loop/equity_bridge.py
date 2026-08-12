"""
AE11H equity bridge — explain account_equity from starting balance via
realized PnL and open unrealized (price / cost-drag / after-cost) decomposition.

All money math uses Decimal. Comparisons use quantize + LEDGER_TOLERANCE.
No Python assert for financial invariants.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.runtime_paper_loop.decimal_money import (
    LEDGER_TOLERANCE,
    decimal_almost_equal,
    decimal_to_str,
    quantize_usd,
    to_decimal,
)
from app.runtime_paper_loop.types import utc_now_iso

LEDGER_SCHEMA_VERSION_AE11H = "AE11H_LEDGER_V1"
ACCOUNTING_MODEL_VERSION_AE11H = "AE11H_EQUITY_BRIDGE_V1"

BRIDGE_FORMULA = (
    "cash_balance + open_market_value = account_equity; "
    "starting_balance + realized_net_pnl + (open_market_value - open_cash_debited) "
    "= account_equity; "
    "equivalently starting + realized_net + total_unrealized_after_cost_pnl "
    "- open_entry_slippage = account_equity "
    "(entry_slippage is cash-debited but excluded from cost_basis); "
    "total_unrealized_after_cost_pnl = open_market_value - open_cost_basis; "
    "price_unrealized_pnl = open_market_value - open_notional; "
    "open_entry_cost_drag = open_notional - open_cost_basis"
)

REQUIRED_OPEN_ECONOMICS = (
    "position_id",
    "pair_address",
    "entry_price",
    "quantity",
    "notional_usd",
    "cost_basis_usd",
    "cash_debited_usd",
    "entry_fee_usd",
    "entry_slippage_usd",
    "opened_at_utc",
    "economic_enrichment_status",
)

EQUITY_BRIDGE_AUDIT_FIELDS = [
    "audit_timestamp_utc",
    "reconciliation_started_at_utc",
    "reconciliation_completed_at_utc",
    "reconciliation_duration_ms",
    "loop_run_id",
    "invocation_id",
    "starting_balance_usd",
    "cash_balance_usd",
    "open_positions_count",
    "closed_positions_count",
    "open_notional_usd",
    "open_cost_basis_usd",
    "open_entry_fee_usd",
    "open_entry_slippage_usd",
    "open_cash_debited_usd",
    "open_market_value_usd",
    "open_price_unrealized_pnl_usd",
    "open_entry_cost_drag_usd",
    "open_total_unrealized_after_cost_pnl_usd",
    "realized_gross_pnl_usd",
    "realized_net_pnl_usd",
    "realized_exit_fee_usd",
    "realized_total_fees_usd",
    "total_entry_fees_usd",
    "total_exit_fees_usd",
    "total_fees_usd",
    "total_slippage_usd",
    "account_equity_usd",
    "expected_account_equity_usd",
    "equity_bridge_diff_usd",
    "pnl_bridge_diff_usd",
    "bridge_status",
    "accounting_model_version",
    "ledger_schema_version",
    "bridge_formula",
    "valuation_source",
    "missing_open_economics_count",
    "blocked_open_economics_count",
    "notes",
]

ACCOUNT_EQUITY_SUMMARY_FIELDS = [
    "audit_timestamp_utc",
    "loop_run_id",
    "invocation_id",
    "starting_balance_usd",
    "cash_balance_usd",
    "open_notional_usd",
    "open_cost_basis_usd",
    "open_market_value_usd",
    "open_entry_fee_usd",
    "open_entry_slippage_usd",
    "open_entry_cost_drag_usd",
    "realized_net_pnl_usd",
    "realized_gross_pnl_usd",
    "total_fees_usd",
    "total_slippage_usd",
    "price_unrealized_pnl_usd",
    "total_unrealized_after_cost_pnl_usd",
    "account_equity_usd",
    "bridge_status",
    "bridge_diff_usd",
    "pnl_bridge_diff_usd",
    "valuation_source",
    "missing_open_economics_count",
    "open_position_economic_completeness_status",
]


class LedgerInvariantViolation(Exception):
    """Raised when a Decimal ledger invariant fails after an economic mutation."""

    def __init__(
        self,
        reason: str,
        *,
        stage: str = "",
        details: dict[str, Any] | None = None,
    ) -> None:
        self.reason = reason
        self.stage = stage
        self.details = details or {}
        super().__init__(f"LedgerInvariantViolation[{stage}]: {reason}")


@dataclass
class EquityBridgeResult:
    starting_balance_usd: Decimal = Decimal("0")
    cash_balance_usd: Decimal = Decimal("0")
    open_positions_count: int = 0
    closed_positions_count: int = 0
    open_notional_usd: Decimal = Decimal("0")
    open_cost_basis_usd: Decimal = Decimal("0")
    open_entry_fee_usd: Decimal = Decimal("0")
    open_entry_slippage_usd: Decimal = Decimal("0")
    open_cash_debited_usd: Decimal = Decimal("0")
    open_market_value_usd: Decimal = Decimal("0")
    open_price_unrealized_pnl_usd: Decimal = Decimal("0")
    open_entry_cost_drag_usd: Decimal = Decimal("0")
    open_total_unrealized_after_cost_pnl_usd: Decimal = Decimal("0")
    realized_gross_pnl_usd: Decimal = Decimal("0")
    realized_net_pnl_usd: Decimal = Decimal("0")
    realized_exit_fee_usd: Decimal = Decimal("0")
    realized_total_fees_usd: Decimal = Decimal("0")
    total_entry_fees_usd: Decimal = Decimal("0")
    total_exit_fees_usd: Decimal = Decimal("0")
    total_fees_usd: Decimal = Decimal("0")
    total_slippage_usd: Decimal = Decimal("0")
    account_equity_usd: Decimal = Decimal("0")
    expected_account_equity_usd: Decimal = Decimal("0")
    equity_bridge_diff_usd: Decimal = Decimal("0")
    pnl_bridge_diff_usd: Decimal = Decimal("0")
    bridge_status: str = "PASS"
    accounting_model_version: str = ACCOUNTING_MODEL_VERSION_AE11H
    ledger_schema_version: str = LEDGER_SCHEMA_VERSION_AE11H
    bridge_formula: str = BRIDGE_FORMULA
    valuation_source: str = "sqlite_open_market_value_or_cost_basis_fallback"
    missing_open_economics_count: int = 0
    blocked_open_economics_count: int = 0
    open_position_economic_completeness_status: str = "PASS"
    notes: str = ""
    reconciliation_started_at_utc: str = ""
    reconciliation_completed_at_utc: str = ""
    reconciliation_duration_ms: float = 0.0
    legacy_unrealized_pnl_usd: Decimal = Decimal("0")
    # legacy unrealized_pnl_usd semantics: price-only (mv - notional) for open book

    def to_meta(self) -> dict[str, Any]:
        return {
            "equity_bridge_status": self.bridge_status,
            "bridge_status": self.bridge_status,
            "equity_bridge_diff_usd": float(self.equity_bridge_diff_usd),
            "pnl_bridge_diff_usd": float(self.pnl_bridge_diff_usd),
            "starting_balance_usd": float(self.starting_balance_usd),
            "cash_balance_usd": float(self.cash_balance_usd),
            "open_notional_usd": float(self.open_notional_usd),
            "open_cost_basis_usd": float(self.open_cost_basis_usd),
            "open_market_value_usd": float(self.open_market_value_usd),
            "open_entry_fee_usd": float(self.open_entry_fee_usd),
            "open_entry_slippage_usd": float(self.open_entry_slippage_usd),
            "open_entry_cost_drag_usd": float(self.open_entry_cost_drag_usd),
            "open_cash_debited_usd": float(self.open_cash_debited_usd),
            "price_unrealized_pnl_usd": float(self.open_price_unrealized_pnl_usd),
            "total_unrealized_after_cost_pnl_usd": float(
                self.open_total_unrealized_after_cost_pnl_usd
            ),
            "realized_net_pnl_usd": float(self.realized_net_pnl_usd),
            "realized_gross_pnl_usd": float(self.realized_gross_pnl_usd),
            "account_equity_usd": float(self.account_equity_usd),
            "expected_account_equity_usd": float(self.expected_account_equity_usd),
            "missing_open_economics_count": self.missing_open_economics_count,
            "blocked_open_economics_count": self.blocked_open_economics_count,
            "open_position_economic_completeness_status": (
                self.open_position_economic_completeness_status
            ),
            "bridge_formula": self.bridge_formula,
            "valuation_source": self.valuation_source,
            "accounting_model_version": self.accounting_model_version,
            "ledger_schema_version": self.ledger_schema_version,
            "unrealized_pnl_semantics": (
                "legacy unrealized_pnl_usd = price_unrealized_pnl_usd "
                "(open_market_value - open_notional); "
                "total_unrealized_after_cost_pnl_usd = open_market_value - open_cost_basis"
            ),
            "total_fees_usd": float(self.total_fees_usd),
            "total_slippage_usd": float(self.total_slippage_usd),
        }


def _missing_open_fields(pos: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    for f in REQUIRED_OPEN_ECONOMICS:
        val = pos.get(f)
        if val is None or str(val).strip() == "":
            missing.append(f)
            continue
        if f in ("entry_price", "quantity", "notional_usd", "cost_basis_usd", "cash_debited_usd"):
            if to_decimal(val) < 0:
                missing.append(f)
            elif f != "cost_basis_usd" and to_decimal(val) <= 0:
                missing.append(f)
        if f in ("entry_fee_usd", "entry_slippage_usd") and to_decimal(val) < 0:
            missing.append(f)
    status = (pos.get("status") or "OPEN").upper()
    if status != "OPEN":
        missing.append("status_not_OPEN")
    enrichment = (pos.get("economic_enrichment_status") or "").upper()
    if enrichment in ("MISSING",):
        if "economic_enrichment_status" not in missing:
            missing.append("economic_enrichment_status")
    return missing


def build_equity_bridge(
    state_db: Any,
    *,
    starting_balance_usd: float | Decimal,
    loop_run_id: str = "",
    invocation_id: str = "",
    cash_balance_override: Decimal | None = None,
    account_equity_override: Decimal | None = None,
) -> EquityBridgeResult:
    """Build full equity / PnL bridge from SQLite open + closed economics."""
    started = utc_now_iso()
    t0 = time.perf_counter()
    starting = quantize_usd(starting_balance_usd)

    open_rows = (
        state_db.get_open_positions()
        if hasattr(state_db, "get_open_positions")
        else state_db.load_active_positions()
    )
    closed_rows: list[dict[str, Any]] = []
    if hasattr(state_db, "load_closed_positions"):
        closed_rows = state_db.load_closed_positions()
    if not closed_rows:
        try:
            closed_rows = [
                dict(r)
                for r in state_db._conn.execute("SELECT * FROM closed_positions").fetchall()
            ]
        except Exception:
            closed_rows = []

    open_notional = Decimal("0")
    open_cost = Decimal("0")
    open_fee = Decimal("0")
    open_slip = Decimal("0")
    open_debit = Decimal("0")
    open_mv = Decimal("0")
    missing = 0
    blocked = 0
    valuation_sources: set[str] = set()

    for pos in open_rows:
        miss = _missing_open_fields(pos)
        enrichment = (pos.get("economic_enrichment_status") or "").upper()
        if miss or enrichment in ("MISSING", "PARTIAL"):
            missing += 1
            if enrichment == "MISSING" or any(
                f in miss for f in ("entry_price", "quantity", "notional_usd", "cash_debited_usd")
            ):
                blocked += 1
            # Do not silently value incomplete economics as zero in bridge aggregates
            continue

        notional = to_decimal(pos.get("notional_usd"))
        cost = to_decimal(pos.get("cost_basis_usd") or notional)
        fee = to_decimal(pos.get("entry_fee_usd") or 0)
        slip = to_decimal(pos.get("entry_slippage_usd") or 0)
        debit = to_decimal(pos.get("cash_debited_usd") or (notional + fee + slip))
        if slip <= 0 and debit > cost:
            slip = quantize_usd(debit - cost)
        raw_mv = pos.get("open_market_value_usd")
        if raw_mv is not None and str(raw_mv).strip() != "":
            mv = to_decimal(raw_mv)
            valuation_sources.add("sqlite_open_market_value")
        else:
            mv = cost
            valuation_sources.add("cost_basis_fallback")

        open_notional += notional
        open_cost += cost
        open_fee += fee
        open_slip += slip
        open_debit += debit
        open_mv += mv

    open_notional = quantize_usd(open_notional)
    open_cost = quantize_usd(open_cost)
    open_fee = quantize_usd(open_fee)
    open_slip = quantize_usd(open_slip)
    open_debit = quantize_usd(open_debit)
    open_mv = quantize_usd(open_mv)

    price_upnl = quantize_usd(open_mv - open_notional)
    cost_drag = quantize_usd(open_notional - open_cost)
    total_after_cost = quantize_usd(open_mv - open_cost)

    realized_gross = Decimal("0")
    realized_net = Decimal("0")
    realized_exit_fee = Decimal("0")
    realized_total_fees = Decimal("0")
    closed_entry_fees = Decimal("0")
    closed_slip = Decimal("0")
    seen: set[str] = set()
    for pos in closed_rows:
        pid = str(pos.get("position_id") or "")
        if pid and pid in seen:
            continue
        if pid:
            seen.add(pid)
        realized_gross += to_decimal(pos.get("gross_pnl_usd") or 0)
        realized_net += to_decimal(pos.get("net_pnl_usd") or 0)
        realized_exit_fee += to_decimal(pos.get("exit_fee_usd") or 0)
        realized_total_fees += to_decimal(pos.get("total_fees_usd") or 0)
        closed_entry_fees += to_decimal(pos.get("entry_fee_usd") or 0)
        closed_slip += to_decimal(pos.get("exit_slippage_usd") or 0)

    realized_gross = quantize_usd(realized_gross)
    realized_net = quantize_usd(realized_net)
    realized_exit_fee = quantize_usd(realized_exit_fee)
    realized_total_fees = quantize_usd(realized_total_fees)

    # Reconstruct cash if not overridden
    cash_debited_all = open_debit
    cash_credited = Decimal("0")
    seen2: set[str] = set()
    for pos in closed_rows:
        pid = str(pos.get("position_id") or "")
        if pid and pid in seen2:
            continue
        if pid:
            seen2.add(pid)
        cash_debited_all += to_decimal(pos.get("cash_debited_usd") or pos.get("notional_usd") or 0)
        cash_credited += to_decimal(pos.get("cash_credited_usd") or 0)
    expected_cash = quantize_usd(starting - cash_debited_all + cash_credited)
    cash = cash_balance_override if cash_balance_override is not None else expected_cash
    cash = quantize_usd(cash)
    equity = (
        account_equity_override
        if account_equity_override is not None
        else quantize_usd(cash + open_mv)
    )
    equity = quantize_usd(equity)

    # Closed cash-flow PnL may differ from stored net_pnl when legacy rows omitted entry slip.
    closed_debit_only = quantize_usd(cash_debited_all - open_debit)
    realized_from_cash = quantize_usd(cash_credited - closed_debit_only)
    realized_for_bridge = realized_net
    if not decimal_almost_equal(realized_net, realized_from_cash):
        realized_for_bridge = realized_from_cash
        notes_legacy = (
            f"realized_net_stored={decimal_to_str(realized_net)} "
            f"realized_from_cash_flows={decimal_to_str(realized_from_cash)}; "
            "using cash-flow realized for PnL bridge"
        )
    else:
        notes_legacy = ""

    # Full PnL bridge: starting + realized_cash_flow + (mv - open_cash_debited)
    expected_equity = quantize_usd(starting + realized_for_bridge + open_mv - open_debit)
    equity_diff = quantize_usd(abs(equity - quantize_usd(cash + open_mv)))
    pnl_diff = quantize_usd(abs(equity - expected_equity))

    notes_parts = [
        "legacy unrealized_pnl_usd := price_unrealized_pnl_usd (mv - notional)",
        "total_unrealized_after_cost_pnl_usd := mv - cost_basis (entry fee drag vs cost_basis)",
        "open_entry_cost_drag_usd := notional - cost_basis",
        "PnL bridge uses starting + realized_cash_flow + (mv - open_cash_debited)",
    ]
    if notes_legacy:
        notes_parts.append(notes_legacy)
    status = "PASS"
    completeness = "PASS"
    if missing > 0 or blocked > 0:
        completeness = "FAIL" if blocked > 0 else "WARNING"
        status = "FAIL" if blocked > 0 else "WARNING"
        notes_parts.append(
            f"missing_open_economics={missing}; blocked_open_economics={blocked}"
        )
    if not decimal_almost_equal(equity, cash + open_mv):
        status = "FAIL"
        notes_parts.append("cash_plus_mv_equity_mismatch")
    if not decimal_almost_equal(equity, expected_equity):
        status = "FAIL"
        notes_parts.append(
            f"pnl_bridge_mismatch expected={decimal_to_str(expected_equity)} "
            f"actual={decimal_to_str(equity)}"
        )

    completed = utc_now_iso()
    duration_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    valuation = (
        ",".join(sorted(valuation_sources))
        if valuation_sources
        else "no_valued_open_positions"
    )

    return EquityBridgeResult(
        starting_balance_usd=starting,
        cash_balance_usd=cash,
        open_positions_count=len(open_rows),
        closed_positions_count=len({str(r.get("position_id") or "") for r in closed_rows if r.get("position_id")}),
        open_notional_usd=open_notional,
        open_cost_basis_usd=open_cost,
        open_entry_fee_usd=open_fee,
        open_entry_slippage_usd=open_slip,
        open_cash_debited_usd=open_debit,
        open_market_value_usd=open_mv,
        open_price_unrealized_pnl_usd=price_upnl,
        open_entry_cost_drag_usd=cost_drag,
        open_total_unrealized_after_cost_pnl_usd=total_after_cost,
        realized_gross_pnl_usd=realized_gross,
        realized_net_pnl_usd=realized_net,
        realized_exit_fee_usd=realized_exit_fee,
        realized_total_fees_usd=realized_total_fees,
        total_entry_fees_usd=quantize_usd(open_fee + closed_entry_fees),
        total_exit_fees_usd=realized_exit_fee,
        total_fees_usd=quantize_usd(open_fee + closed_entry_fees + realized_exit_fee),
        total_slippage_usd=quantize_usd(open_slip + closed_slip),
        account_equity_usd=equity,
        expected_account_equity_usd=expected_equity,
        equity_bridge_diff_usd=equity_diff,
        pnl_bridge_diff_usd=pnl_diff,
        bridge_status=status,
        valuation_source=valuation,
        missing_open_economics_count=missing,
        blocked_open_economics_count=blocked,
        open_position_economic_completeness_status=completeness,
        notes="; ".join(notes_parts),
        reconciliation_started_at_utc=started,
        reconciliation_completed_at_utc=completed,
        reconciliation_duration_ms=duration_ms,
        legacy_unrealized_pnl_usd=price_upnl,
    )


def write_equity_bridge_audit(
    path: Path,
    *,
    loop_run_id: str,
    invocation_id: str,
    bridge: EquityBridgeResult,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "audit_timestamp_utc": utc_now_iso(),
        "reconciliation_started_at_utc": bridge.reconciliation_started_at_utc,
        "reconciliation_completed_at_utc": bridge.reconciliation_completed_at_utc,
        "reconciliation_duration_ms": bridge.reconciliation_duration_ms,
        "loop_run_id": loop_run_id,
        "invocation_id": invocation_id,
        "starting_balance_usd": decimal_to_str(bridge.starting_balance_usd),
        "cash_balance_usd": decimal_to_str(bridge.cash_balance_usd),
        "open_positions_count": bridge.open_positions_count,
        "closed_positions_count": bridge.closed_positions_count,
        "open_notional_usd": decimal_to_str(bridge.open_notional_usd),
        "open_cost_basis_usd": decimal_to_str(bridge.open_cost_basis_usd),
        "open_entry_fee_usd": decimal_to_str(bridge.open_entry_fee_usd),
        "open_entry_slippage_usd": decimal_to_str(bridge.open_entry_slippage_usd),
        "open_cash_debited_usd": decimal_to_str(bridge.open_cash_debited_usd),
        "open_market_value_usd": decimal_to_str(bridge.open_market_value_usd),
        "open_price_unrealized_pnl_usd": decimal_to_str(bridge.open_price_unrealized_pnl_usd),
        "open_entry_cost_drag_usd": decimal_to_str(bridge.open_entry_cost_drag_usd),
        "open_total_unrealized_after_cost_pnl_usd": decimal_to_str(
            bridge.open_total_unrealized_after_cost_pnl_usd
        ),
        "realized_gross_pnl_usd": decimal_to_str(bridge.realized_gross_pnl_usd),
        "realized_net_pnl_usd": decimal_to_str(bridge.realized_net_pnl_usd),
        "realized_exit_fee_usd": decimal_to_str(bridge.realized_exit_fee_usd),
        "realized_total_fees_usd": decimal_to_str(bridge.realized_total_fees_usd),
        "total_entry_fees_usd": decimal_to_str(bridge.total_entry_fees_usd),
        "total_exit_fees_usd": decimal_to_str(bridge.total_exit_fees_usd),
        "total_fees_usd": decimal_to_str(bridge.total_fees_usd),
        "total_slippage_usd": decimal_to_str(bridge.total_slippage_usd),
        "account_equity_usd": decimal_to_str(bridge.account_equity_usd),
        "expected_account_equity_usd": decimal_to_str(bridge.expected_account_equity_usd),
        "equity_bridge_diff_usd": decimal_to_str(bridge.equity_bridge_diff_usd),
        "pnl_bridge_diff_usd": decimal_to_str(bridge.pnl_bridge_diff_usd),
        "bridge_status": bridge.bridge_status,
        "accounting_model_version": bridge.accounting_model_version,
        "ledger_schema_version": bridge.ledger_schema_version,
        "bridge_formula": bridge.bridge_formula,
        "valuation_source": bridge.valuation_source,
        "missing_open_economics_count": bridge.missing_open_economics_count,
        "blocked_open_economics_count": bridge.blocked_open_economics_count,
        "notes": bridge.notes,
    }
    write_header = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=EQUITY_BRIDGE_AUDIT_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
    return path


def write_account_equity_bridge_summary(
    path: Path,
    *,
    loop_run_id: str,
    invocation_id: str,
    bridge: EquityBridgeResult,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "audit_timestamp_utc": utc_now_iso(),
        "loop_run_id": loop_run_id,
        "invocation_id": invocation_id,
        "starting_balance_usd": decimal_to_str(bridge.starting_balance_usd),
        "cash_balance_usd": decimal_to_str(bridge.cash_balance_usd),
        "open_notional_usd": decimal_to_str(bridge.open_notional_usd),
        "open_cost_basis_usd": decimal_to_str(bridge.open_cost_basis_usd),
        "open_market_value_usd": decimal_to_str(bridge.open_market_value_usd),
        "open_entry_fee_usd": decimal_to_str(bridge.open_entry_fee_usd),
        "open_entry_slippage_usd": decimal_to_str(bridge.open_entry_slippage_usd),
        "open_entry_cost_drag_usd": decimal_to_str(bridge.open_entry_cost_drag_usd),
        "realized_net_pnl_usd": decimal_to_str(bridge.realized_net_pnl_usd),
        "realized_gross_pnl_usd": decimal_to_str(bridge.realized_gross_pnl_usd),
        "total_fees_usd": decimal_to_str(bridge.total_fees_usd),
        "total_slippage_usd": decimal_to_str(bridge.total_slippage_usd),
        "price_unrealized_pnl_usd": decimal_to_str(bridge.open_price_unrealized_pnl_usd),
        "total_unrealized_after_cost_pnl_usd": decimal_to_str(
            bridge.open_total_unrealized_after_cost_pnl_usd
        ),
        "account_equity_usd": decimal_to_str(bridge.account_equity_usd),
        "bridge_status": bridge.bridge_status,
        "bridge_diff_usd": decimal_to_str(bridge.equity_bridge_diff_usd),
        "pnl_bridge_diff_usd": decimal_to_str(bridge.pnl_bridge_diff_usd),
        "valuation_source": bridge.valuation_source,
        "missing_open_economics_count": bridge.missing_open_economics_count,
        "open_position_economic_completeness_status": (
            bridge.open_position_economic_completeness_status
        ),
    }
    write_header = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=ACCOUNT_EQUITY_SUMMARY_FIELDS, extrasaction="ignore"
        )
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
    return path


def check_ledger_invariants(
    *,
    cash_balance: Any,
    open_market_value_usd: Any,
    account_equity_usd: Any,
    starting_balance_usd: Any,
    realized_net_pnl_usd: Any,
    total_unrealized_after_cost_pnl_usd: Any,
    open_entry_slippage_usd: Any = 0,
    open_cash_debited_usd: Any | None = None,
    expected_cash_balance: Any | None = None,
    realized_from_cash_flows: Any | None = None,
    stage: str = "unspecified",
    raise_on_failure: bool = False,
) -> dict[str, Any]:
    """
    Explicit Decimal invariant checks — never use Python assert.
    Returns status dict; optionally raises LedgerInvariantViolation.
    """
    cash = quantize_usd(cash_balance)
    mv = quantize_usd(open_market_value_usd)
    equity = quantize_usd(account_equity_usd)
    starting = quantize_usd(starting_balance_usd)
    realized = quantize_usd(
        realized_from_cash_flows
        if realized_from_cash_flows is not None
        else realized_net_pnl_usd
    )
    after_cost = quantize_usd(total_unrealized_after_cost_pnl_usd)
    slip = quantize_usd(open_entry_slippage_usd)

    failures: list[str] = []
    if not decimal_almost_equal(equity, cash + mv):
        failures.append(
            f"CASH_PLUS_MV_NE_EQUITY cash={cash} mv={mv} equity={equity}"
        )
    if open_cash_debited_usd is not None:
        expected_equity = quantize_usd(
            starting + realized + mv - quantize_usd(open_cash_debited_usd)
        )
    else:
        expected_equity = quantize_usd(starting + realized + after_cost - slip)
    if not decimal_almost_equal(equity, expected_equity):
        failures.append(
            f"PNL_BRIDGE_NE_EQUITY expected={expected_equity} equity={equity}"
        )
    if expected_cash_balance is not None:
        if not decimal_almost_equal(cash, expected_cash_balance):
            failures.append(
                f"CASH_NE_EXPECTED cash={cash} expected={quantize_usd(expected_cash_balance)}"
            )
    if cash < 0:
        failures.append(f"NEGATIVE_CASH cash={cash}")
    if equity < 0:
        failures.append(f"NEGATIVE_EQUITY equity={equity}")

    status = "PASS" if not failures else "FAIL"
    result = {
        "ledger_invariant_status": status,
        "invariant_check_stage": stage,
        "ledger_invariant_failure_count": len(failures),
        "ledger_invariant_last_failure_reason": failures[-1] if failures else None,
        "failures": failures,
        "tolerance_usd": str(LEDGER_TOLERANCE),
    }
    if failures and raise_on_failure:
        raise LedgerInvariantViolation(
            failures[-1],
            stage=stage,
            details=result,
        )
    return result
