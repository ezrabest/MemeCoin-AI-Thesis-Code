"""Live no-wallet dry-run adapter — validates intent, never signs or submits."""

from __future__ import annotations

from typing import Any

from app.execution.adapters import ExecutionAdapter
from app.execution.types import ExecutionResult, LiveDryRunOrder, OrderIntent
from app.paper_trading.order_simulator import compute_execution_latency
from app.paper_trading.types import ExecutionLatencyStatus


class LiveNoWalletDryRunAdapter(ExecutionAdapter):
    """Accepts order intent, validates fields, does not sign or submit."""

    def __init__(self) -> None:
        self.orders: list[LiveDryRunOrder] = []
        self.real_transaction_attempted = False
        self.private_key_accessed = False

    def is_wallet_configured(self) -> bool:
        return False

    def execute(self, intent: OrderIntent, **kwargs: Any) -> ExecutionResult:
        if not intent.candidate_id:
            return ExecutionResult(
                success=False,
                execution_mode="LIVE_NO_WALLET_DRY_RUN",
                live_submission_status="NOT_SUBMITTED_NO_WALLET",
                wallet_required=True,
                wallet_configured=False,
                real_transaction_attempted=False,
                no_wallet_dry_run=True,
                no_live_submission=True,
                message="missing_candidate_id",
            )

        order = LiveDryRunOrder(
            candidate_id=intent.candidate_id,
            symbol=intent.symbol,
            pair_address=intent.pair_address,
            side=intent.side,
            order_type=intent.order_type,
            notional_usd=intent.notional_usd,
            requested_price_usd=intent.requested_price_usd,
            source_decision_id=intent.source_decision_id,
            source_context_record_id=intent.source_context_record_id,
            source_llm_audit_record_id=intent.source_llm_audit_record_id,
            decision_status=intent.decision_status,
            consensus_family=intent.consensus_family,
            context_schema_id=intent.context_schema_id,
            audit_verdict=intent.audit_verdict,
            audit_blockers=list(intent.audit_blockers),
            audit_warnings=list(intent.audit_warnings),
            scoring_policy_id=intent.scoring_policy_id,
            decision_created_at_utc=intent.decision_created_at_utc,
        )

        latency_ms, latency_status = compute_execution_latency(
            intent.decision_created_at_utc, order.order_created_at_utc
        )
        order.execution_latency_ms = latency_ms
        order.execution_latency_status = (
            latency_status
            if latency_status != ExecutionLatencyStatus.NOT_FILLED.value
            else ExecutionLatencyStatus.OK.value
        )

        self.orders.append(order)

        return ExecutionResult(
            success=True,
            execution_mode="LIVE_NO_WALLET_DRY_RUN",
            live_submission_status="NOT_SUBMITTED_NO_WALLET",
            wallet_required=True,
            wallet_configured=False,
            real_transaction_attempted=False,
            no_wallet_dry_run=True,
            no_live_submission=True,
            order_id=order.live_order_id,
            message="LIVE_ORDER_NOT_SUBMITTED_NO_WALLET",
            record=order.to_dict(),
        )

    def audit_summary(self) -> dict[str, Any]:
        return {
            "adapter": "LiveNoWalletDryRunAdapter",
            "wallet_configured": False,
            "real_transaction_attempted": self.real_transaction_attempted,
            "private_key_accessed": self.private_key_accessed,
            "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
            "execution_mode": "LIVE_NO_WALLET_DRY_RUN",
            "orders_created": len(self.orders),
        }
