"""Execution adapter interfaces for AE10."""

from __future__ import annotations

from typing import Any

from abc import ABC, abstractmethod

from app.execution.types import ExecutionResult, OrderIntent
from app.paper_trading.ledger import DemoLedger
from app.paper_trading.order_simulator import PaperOrderSimulator
from app.paper_trading.position_manager import open_position_from_order
from app.paper_trading.types import PaperOrder, PaperOrderStatus


class ExecutionAdapter(ABC):
    @abstractmethod
    def execute(self, intent: OrderIntent, **kwargs: Any) -> ExecutionResult:
        ...


class PaperExecutionAdapter(ExecutionAdapter):
    """Paper/demo execution adapter with atomic ledger registration."""

    def __init__(
        self,
        ledger: DemoLedger,
        simulator: PaperOrderSimulator,
    ) -> None:
        self.ledger = ledger
        self.simulator = simulator

    def execute(
        self,
        intent: OrderIntent,
        *,
        traceability: dict[str, Any],
        price_result: dict[str, Any],
        allow_audit_blockers: bool = False,
    ) -> ExecutionResult:
        cash_before = self.ledger.account.cash_balance_usd
        pending = self.simulator.build_pending_order(
            traceability,
            price_result=price_result,
            coin_id=intent.coin_id,
            symbol=intent.symbol,
            pair_address=intent.pair_address,
            notional_usd=intent.notional_usd,
            allow_audit_blockers=allow_audit_blockers,
            decision_created_at_utc=intent.decision_created_at_utc,
            order_created_at_utc=intent.order_created_at_utc,
        )

        if pending.status == PaperOrderStatus.PAPER_REJECTED.value:
            self.ledger.finalize_rejected(pending)
            return ExecutionResult(
                success=False,
                execution_mode="PAPER_DEMO",
                no_live_submission=True,
                order_id=pending.paper_order_id,
                message=pending.paper_trade_reason or pending.status,
                record=pending.to_dict(),
            )

        try:
            self.ledger.register_order_intent(pending)
        except ValueError:
            pending.status = PaperOrderStatus.PAPER_REJECTED.value
            pending.paper_trade_reason = "duplicate_order_registration"
            self.ledger.finalize_rejected(pending)
            return ExecutionResult(
                success=False,
                execution_mode="PAPER_DEMO",
                no_live_submission=True,
                order_id=pending.paper_order_id,
                message=pending.paper_trade_reason,
                record=pending.to_dict(),
            )

        order = self.simulator.complete_fill(
            pending,
            price_result=price_result,
            notional_usd=intent.notional_usd,
            traceability=traceability,
            allow_audit_blockers=allow_audit_blockers,
            decision_created_at_utc=intent.decision_created_at_utc,
        )

        if order.status == PaperOrderStatus.PAPER_FILLED.value:
            position = open_position_from_order(order)
            if position and self.ledger.apply_fill(order, position):
                return ExecutionResult(
                    success=True,
                    execution_mode="PAPER_DEMO",
                    no_live_submission=True,
                    order_id=order.paper_order_id,
                    message=order.paper_trade_reason or order.status,
                    record=order.to_dict(),
                )
            order.status = PaperOrderStatus.PAPER_REJECTED.value
            order.paper_trade_reason = "ledger_apply_fill_failed"
            self.ledger.finalize_rejected(order)
        else:
            self.ledger.finalize_rejected(order)

        if self.ledger.account.cash_balance_usd != cash_before:
            order.paper_trade_reason = "unexpected_cash_mutation_on_rejected_order"

        return ExecutionResult(
            success=False,
            execution_mode="PAPER_DEMO",
            no_live_submission=True,
            order_id=order.paper_order_id,
            message=order.paper_trade_reason or order.status,
            record=order.to_dict(),
        )


class LiveWalletExecutionAdapter(ExecutionAdapter):
    """Placeholder for future live wallet execution — not active by default."""

    def execute(self, intent: OrderIntent, **kwargs: Any) -> ExecutionResult:
        return ExecutionResult(
            success=False,
            execution_mode="LIVE_WALLET_NOT_CONFIGURED",
            wallet_required=True,
            wallet_configured=False,
            real_transaction_attempted=False,
            no_live_submission=True,
            message="LiveWalletExecutionAdapter is a placeholder; not active in AE10",
        )
