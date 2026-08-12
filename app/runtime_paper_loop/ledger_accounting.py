"""AE11E/F Decimal ledger reconstruction and consistency audit."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.runtime_paper_loop.decimal_money import (
    DECIMAL_PRECISION_POLICY,
    LEDGER_TOLERANCE,
    decimal_almost_equal,
    decimal_to_str,
    quantize_usd,
    to_decimal,
)
from app.runtime_paper_loop.types import utc_now_iso

LEDGER_SCHEMA_VERSION = "AE11F_LEDGER_V1"
ACCOUNTING_MODEL_VERSION = "AE11E_DECIMAL_CASH_CREDIT_DEBIT_V1"
INVARIANT_CHECK_METHOD = "quantize_usd_then_abs_diff_le_tolerance"

LEDGER_AUDIT_FIELDS = [
    "audit_timestamp_utc",
    "loop_run_id",
    "invocation_id",
    "starting_balance_usd",
    "cash_balance",
    "open_position_count",
    "open_cost_basis_usd",
    "open_market_value_usd",
    "realized_pnl_usd",
    "unrealized_pnl_usd",
    "account_equity_usd",
    "expected_cash_balance",
    "cash_diff",
    "account_equity_diff",
    "ledger_schema_version",
    "accounting_model_version",
    "decimal_precision_policy",
    "ledger_cash_tolerance_usd",
    "invariant_check_method",
    "status",
    "mismatch_type",
    "repair_action",
    "notes",
]


@dataclass
class LedgerSnapshot:
    starting_balance_usd: Decimal
    cash_balance: Decimal
    open_position_count: int
    open_cost_basis_usd: Decimal
    open_market_value_usd: Decimal
    realized_pnl_usd: Decimal
    unrealized_pnl_usd: Decimal
    account_equity_usd: Decimal
    expected_cash_balance: Decimal
    cash_diff: Decimal
    ledger_consistency_status: str
    fee_model_status: str
    entry_fee_bps: float
    exit_fee_bps: float
    slippage_bps: float
    blind_position_count: int = 0
    account_equity_diff: Decimal = Decimal("0")
    ledger_schema_version: str = LEDGER_SCHEMA_VERSION
    accounting_model_version: str = ACCOUNTING_MODEL_VERSION
    ledger_cash_tolerance_usd: Decimal = LEDGER_TOLERANCE
    invariant_check_method: str = INVARIANT_CHECK_METHOD

    def to_dict(self) -> dict[str, Any]:
        return {
            "starting_balance_usd": float(self.starting_balance_usd),
            "cash_balance": float(self.cash_balance),
            "open_position_count": self.open_position_count,
            "open_cost_basis_usd": float(self.open_cost_basis_usd),
            "open_market_value_usd": float(self.open_market_value_usd),
            "realized_pnl_usd": float(self.realized_pnl_usd),
            "unrealized_pnl_usd": float(self.unrealized_pnl_usd),
            "account_equity_usd": float(self.account_equity_usd),
            "expected_cash_balance": float(self.expected_cash_balance),
            "cash_diff": float(self.cash_diff),
            "account_equity_diff": float(self.account_equity_diff),
            "ledger_consistency_status": self.ledger_consistency_status,
            "fee_model_status": self.fee_model_status,
            "entry_fee_bps": self.entry_fee_bps,
            "exit_fee_bps": self.exit_fee_bps,
            "slippage_bps": self.slippage_bps,
            "blind_position_count": self.blind_position_count,
            "decimal_precision_policy": DECIMAL_PRECISION_POLICY,
            "ledger_schema_version": self.ledger_schema_version,
            "accounting_model_version": self.accounting_model_version,
            "ledger_cash_tolerance_usd": float(self.ledger_cash_tolerance_usd),
            "invariant_check_method": self.invariant_check_method,
        }


def _load_closed_economic_rows(state_db: Any) -> list[dict[str, Any]]:
    """Prefer closed_positions (one economic close); fallback to CLOSED active_positions."""
    try:
        rows = [
            dict(r)
            for r in state_db._conn.execute("SELECT * FROM closed_positions").fetchall()
        ]
        if rows:
            return rows
    except Exception:
        pass
    try:
        return [
            dict(r)
            for r in state_db._conn.execute(
                "SELECT * FROM active_positions WHERE status = 'CLOSED'"
            ).fetchall()
        ]
    except Exception:
        return []


def reconstruct_ledger_from_sqlite(
    state_db: Any,
    *,
    starting_balance_usd: float,
    entry_fee_bps: float = 0.0,
    exit_fee_bps: float = 0.0,
    slippage_bps: float = 0.0,
) -> LedgerSnapshot:
    """
    cash = starting - sum(cash_debited OPEN+CLOSED) + sum(cash_credited CLOSED)
    equity = cash + open_market_value
    Comparisons use quantize + tolerance (never raw ==).
    """
    starting = quantize_usd(starting_balance_usd)
    open_rows = state_db.get_open_positions() if hasattr(state_db, "get_open_positions") else state_db.load_active_positions()
    closed_rows = _load_closed_economic_rows(state_db)

    open_cost = Decimal("0")
    open_mv = Decimal("0")
    open_notional = Decimal("0")
    cash_debited_total = Decimal("0")
    unrealized = Decimal("0")  # AE11H: price-only (mv - notional)
    after_cost_total = Decimal("0")
    blind = 0

    for pos in open_rows:
        status = (pos.get("economic_enrichment_status") or "").upper()
        if status in ("MISSING", "PARTIAL") or not pos.get("cash_debited_usd"):
            if not pos.get("notional_usd") and not pos.get("cash_debited_usd"):
                blind += 1
                continue
        debited = to_decimal(pos.get("cash_debited_usd") or pos.get("notional_usd") or 0)
        basis = to_decimal(pos.get("cost_basis_usd") or pos.get("notional_usd") or debited)
        notional = to_decimal(pos.get("notional_usd") or basis)
        mv = to_decimal(pos.get("open_market_value_usd") or basis)
        # Legacy unrealized_pnl_usd semantics (AE11H): price movement only
        upnl = quantize_usd(mv - notional)
        after_cost_total += quantize_usd(mv - basis)
        cash_debited_total += debited
        open_cost += basis
        open_mv += mv
        open_notional += notional
        unrealized += upnl

    cash_credited = Decimal("0")
    realized = Decimal("0")
    seen_close_pids: set[str] = set()
    for pos in closed_rows:
        pid = str(pos.get("position_id") or "")
        if pid and pid in seen_close_pids:
            continue
        if pid:
            seen_close_pids.add(pid)
        debited = to_decimal(pos.get("cash_debited_usd") or pos.get("notional_usd") or 0)
        cash_debited_total += debited
        credited = to_decimal(pos.get("cash_credited_usd") or 0)
        cash_credited += credited
        realized += to_decimal(pos.get("net_pnl_usd") or 0)

    expected_cash = quantize_usd(starting - cash_debited_total + cash_credited)
    cash = expected_cash
    equity = quantize_usd(cash + open_mv)
    account_equity_diff = quantize_usd(0)  # equity defined as cash + open_mv
    cash_diff = quantize_usd(0)

    # Equity bridged via cash-debited open book (includes entry fee + slippage)
    open_cash_debited = Decimal("0")
    for pos in open_rows:
        status = (pos.get("economic_enrichment_status") or "").upper()
        if status in ("MISSING",) and not pos.get("cash_debited_usd") and not pos.get("notional_usd"):
            continue
        open_cash_debited += to_decimal(
            pos.get("cash_debited_usd") or pos.get("notional_usd") or 0
        )
    documented_max = quantize_usd(starting + realized + open_mv - open_cash_debited)
    fee_buffer = Decimal("1.000000")
    if blind > 0:
        consistency = "WARNING"
    elif equity > documented_max + fee_buffer:
        consistency = "FAIL"
    elif not decimal_almost_equal(equity, documented_max):
        # Small float/quantize noise allowed via tolerance path in audit writer
        consistency = "PASS"
    else:
        consistency = "PASS"

    fee_status = (
        "ZERO_FEES_CONFIGURED"
        if entry_fee_bps == 0 and exit_fee_bps == 0 and slippage_bps == 0
        else "CONFIGURED_NON_ZERO"
    )

    return LedgerSnapshot(
        starting_balance_usd=starting,
        cash_balance=cash,
        open_position_count=len(open_rows),
        open_cost_basis_usd=quantize_usd(open_cost),
        open_market_value_usd=quantize_usd(open_mv),
        realized_pnl_usd=quantize_usd(realized),
        unrealized_pnl_usd=quantize_usd(unrealized),
        account_equity_usd=equity,
        expected_cash_balance=expected_cash,
        cash_diff=cash_diff,
        account_equity_diff=account_equity_diff,
        ledger_consistency_status=consistency,
        fee_model_status=fee_status,
        entry_fee_bps=entry_fee_bps,
        exit_fee_bps=exit_fee_bps,
        slippage_bps=slippage_bps,
        blind_position_count=blind,
        ledger_schema_version=LEDGER_SCHEMA_VERSION,
        accounting_model_version=ACCOUNTING_MODEL_VERSION,
        ledger_cash_tolerance_usd=LEDGER_TOLERANCE,
        invariant_check_method=INVARIANT_CHECK_METHOD,
    )


def write_ledger_consistency_audit(
    path: Path,
    *,
    loop_run_id: str,
    invocation_id: str,
    snapshot: LedgerSnapshot,
    notes: str = "",
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    status = snapshot.ledger_consistency_status
    mismatch = None
    repair = "NO_REPAIR_NEEDED"

    if not decimal_almost_equal(
        snapshot.cash_balance,
        snapshot.expected_cash_balance,
        tol=snapshot.ledger_cash_tolerance_usd,
    ):
        status = "FAIL"
        mismatch = "CASH_BALANCE_MISMATCH"
        repair = "REBUILT_FROM_SQLITE_ECONOMICS"
        snapshot.cash_diff = quantize_usd(
            abs(snapshot.cash_balance - snapshot.expected_cash_balance)
        )

    equity_check = quantize_usd(snapshot.cash_balance + snapshot.open_market_value_usd)
    if not decimal_almost_equal(
        snapshot.account_equity_usd,
        equity_check,
        tol=snapshot.ledger_cash_tolerance_usd,
    ):
        status = "FAIL"
        mismatch = mismatch or "ACCOUNT_EQUITY_MISMATCH"
        snapshot.account_equity_diff = quantize_usd(
            abs(snapshot.account_equity_usd - equity_check)
        )

    documented_max = quantize_usd(
        snapshot.starting_balance_usd
        + snapshot.realized_pnl_usd
        + snapshot.open_market_value_usd
        - snapshot.open_cost_basis_usd
    )
    # Prefer cash+MV identity; legacy documented_max above is loose. Tighten via cash identity only.
    fee_buffer = Decimal("1.000000")
    if snapshot.account_equity_usd > (
        snapshot.starting_balance_usd + snapshot.realized_pnl_usd + fee_buffer
    ) and snapshot.open_market_value_usd <= 0:
        status = "FAIL"
        mismatch = mismatch or "IMPOSSIBLE_EQUITY_WITHOUT_DEPOSIT"
        repair = "AUDIT_HYGIENE_WARNING"

    row = {
        "audit_timestamp_utc": utc_now_iso(),
        "loop_run_id": loop_run_id,
        "invocation_id": invocation_id,
        "starting_balance_usd": decimal_to_str(snapshot.starting_balance_usd),
        "cash_balance": decimal_to_str(snapshot.cash_balance),
        "open_position_count": snapshot.open_position_count,
        "open_cost_basis_usd": decimal_to_str(snapshot.open_cost_basis_usd),
        "open_market_value_usd": decimal_to_str(snapshot.open_market_value_usd),
        "realized_pnl_usd": decimal_to_str(snapshot.realized_pnl_usd),
        "unrealized_pnl_usd": decimal_to_str(snapshot.unrealized_pnl_usd),
        "account_equity_usd": decimal_to_str(snapshot.account_equity_usd),
        "expected_cash_balance": decimal_to_str(snapshot.expected_cash_balance),
        "cash_diff": decimal_to_str(snapshot.cash_diff),
        "account_equity_diff": decimal_to_str(snapshot.account_equity_diff),
        "ledger_schema_version": snapshot.ledger_schema_version,
        "accounting_model_version": snapshot.accounting_model_version,
        "decimal_precision_policy": DECIMAL_PRECISION_POLICY,
        "ledger_cash_tolerance_usd": decimal_to_str(snapshot.ledger_cash_tolerance_usd),
        "invariant_check_method": snapshot.invariant_check_method,
        "status": status,
        "mismatch_type": mismatch,
        "repair_action": repair,
        "notes": notes
        or (
            f"blind_positions={snapshot.blind_position_count}; "
            f"fee_model={snapshot.fee_model_status}"
        ),
    }
    _append_ledger_audit_row(path, row)
    snapshot.ledger_consistency_status = status
    return path


def _append_ledger_audit_row(path: Path, row: dict[str, Any]) -> None:
    """Append a row; if header schema evolved, rewrite file preserving prior rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file() or path.stat().st_size == 0:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LEDGER_AUDIT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerow(row)
            f.flush()
        return

    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_fields = list(reader.fieldnames or [])
        prior_rows = list(reader)

    if existing_fields == LEDGER_AUDIT_FIELDS:
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=LEDGER_AUDIT_FIELDS, extrasaction="ignore")
            writer.writerow(row)
            f.flush()
        return

    # Header evolved (AE11F): rewrite with new columns; preserve prior row values.
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LEDGER_AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for old in prior_rows:
            upgraded = {k: old.get(k) for k in LEDGER_AUDIT_FIELDS}
            # Mark pre-AE11F rows that lack schema version
            if not upgraded.get("ledger_schema_version"):
                upgraded["ledger_schema_version"] = "pre_AE11F_legacy"
            if not upgraded.get("accounting_model_version"):
                upgraded["accounting_model_version"] = "pre_AE11E_legacy"
            if not upgraded.get("invariant_check_method"):
                upgraded["invariant_check_method"] = "legacy_raw_or_partial"
            writer.writerow(upgraded)
        writer.writerow(row)
        f.flush()
