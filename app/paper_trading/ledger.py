"""Demo account ledger and reset — atomic fill registration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.paper_trading.types import (
    DemoAccount,
    PaperLedgerSnapshot,
    PaperOrder,
    PaperOrderStatus,
    PaperPosition,
    PaperTradeRecord,
    utc_now_iso,
)


@dataclass
class DemoLedger:
    account: DemoAccount = field(default_factory=DemoAccount)
    orders: list[PaperOrder] = field(default_factory=list)
    positions: list[PaperPosition] = field(default_factory=list)
    trades: list[PaperTradeRecord] = field(default_factory=list)
    snapshots: list[PaperLedgerSnapshot] = field(default_factory=list)
    reset_audit_log: list[dict[str, Any]] = field(default_factory=list)
    _filled_order_ids: set[str] = field(default_factory=set)

    def register_order_intent(self, order: PaperOrder) -> None:
        """Register accepted intent as PAPER_PENDING before fill attempt."""
        if any(o.paper_order_id == order.paper_order_id for o in self.orders):
            raise ValueError(f"duplicate_order_registration:{order.paper_order_id}")
        order.status = PaperOrderStatus.PAPER_PENDING.value
        self.orders.append(order)
        self.account.updated_at_utc = utc_now_iso()

    def apply_fill(self, order: PaperOrder, position: PaperPosition) -> bool:
        """Debit cash only after order is registered and fill transition succeeded."""
        if order.paper_order_id in self._filled_order_ids:
            return False
        if order.status != PaperOrderStatus.PAPER_FILLED.value:
            return False
        if not any(o.paper_order_id == order.paper_order_id for o in self.orders):
            return False

        for idx, existing in enumerate(self.orders):
            if existing.paper_order_id == order.paper_order_id:
                self.orders[idx] = order
                break

        self.positions.append(position)
        self.account.cash_balance_usd -= order.notional_usd
        self._filled_order_ids.add(order.paper_order_id)
        self.account.open_position_count = len([p for p in self.positions if p.status == "OPEN"])
        self._refresh_equity()
        self.account.updated_at_utc = utc_now_iso()
        self.take_snapshot()
        return True

    def finalize_rejected(self, order: PaperOrder) -> None:
        """Record rejection without cash mutation."""
        found = False
        for idx, existing in enumerate(self.orders):
            if existing.paper_order_id == order.paper_order_id:
                self.orders[idx] = order
                found = True
                break
        if not found:
            self.orders.append(order)
        self.account.updated_at_utc = utc_now_iso()

    def open_position(self, order: PaperOrder, position: PaperPosition) -> bool:
        """Backward-compatible alias for apply_fill."""
        return self.apply_fill(order, position)

    def record_rejected_order(self, order: PaperOrder) -> None:
        self.finalize_rejected(order)

    def close_position(
        self,
        position: PaperPosition,
        trade: PaperTradeRecord,
        order: PaperOrder,
    ) -> None:
        position.status = "CLOSED"
        position.closed_at_utc = trade.closed_at_utc
        position.exit_price_usd = trade.exit_price_usd
        position.realized_pnl_usd = trade.realized_pnl_usd
        self.trades.append(trade)
        self.account.cash_balance_usd += trade.notional_usd + trade.realized_pnl_usd
        self.account.realized_pnl_usd += trade.realized_pnl_usd
        self.account.closed_trade_count += 1
        self.account.open_position_count = len([p for p in self.positions if p.status == "OPEN"])
        self._refresh_equity()
        self.account.updated_at_utc = utc_now_iso()
        if order.paper_order_id:
            for o in self.orders:
                if o.paper_order_id == order.paper_order_id:
                    o.status = order.status
                    o.closed_at_utc = order.closed_at_utc
                    break

    def take_snapshot(self) -> PaperLedgerSnapshot:
        snap = PaperLedgerSnapshot(
            account_id=self.account.account_id,
            cash_balance_usd=self.account.cash_balance_usd,
            equity_usd=self.account.equity_usd,
            open_position_count=self.account.open_position_count,
            closed_trade_count=self.account.closed_trade_count,
            realized_pnl_usd=self.account.realized_pnl_usd,
            unrealized_pnl_usd=self.account.unrealized_pnl_usd,
        )
        self.snapshots.append(snap)
        return snap

    def _refresh_equity(self) -> None:
        open_notional = sum(p.notional_usd for p in self.positions if p.status == "OPEN")
        self.account.equity_usd = self.account.cash_balance_usd + open_notional
        self.account.unrealized_pnl_usd = 0.0


def reset_demo_account(
    ledger: DemoLedger,
    *,
    starting_balance_usd: float = 10_000.0,
    clear_history: bool = False,
    preserve_account_id: bool = True,
) -> dict[str, Any]:
    """Reset demo account state; preserve history unless clear_history."""
    old_account_id = ledger.account.account_id
    old_reset_count = ledger.account.reset_count

    ledger.account.starting_balance_usd = starting_balance_usd
    ledger.account.cash_balance_usd = starting_balance_usd
    ledger.account.equity_usd = starting_balance_usd
    ledger.account.open_position_count = 0
    ledger.account.closed_trade_count = 0
    ledger.account.realized_pnl_usd = 0.0
    ledger.account.unrealized_pnl_usd = 0.0
    ledger.account.reset_count = old_reset_count + 1
    ledger.account.updated_at_utc = utc_now_iso()
    if preserve_account_id:
        ledger.account.account_id = old_account_id

    if clear_history:
        ledger.orders.clear()
        ledger.positions.clear()
        ledger.trades.clear()
        ledger.snapshots.clear()
        ledger._filled_order_ids.clear()

    audit = {
        "reset_executed": True,
        "starting_balance_usd": starting_balance_usd,
        "clear_history": clear_history,
        "reset_count": ledger.account.reset_count,
        "account_id": ledger.account.account_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "live_wallet_settings_affected": False,
    }
    ledger.reset_audit_log.append(audit)
    return audit
