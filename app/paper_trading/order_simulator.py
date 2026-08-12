"""Paper order simulation and execution latency."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.paper_trading.order_state_machine import OrderStateMachine
from app.paper_trading.price_oracle import DemoPriceOracle, parse_ts
from app.paper_trading.types import (
    ExecutionLatencyStatus,
    PaperOrder,
    PaperOrderStatus,
    PaperTradeDecisionStatus,
    PriceStatus,
    utc_now_iso,
)

DEFAULT_NOTIONAL_USD = 100.0


def compute_execution_latency(
    decision_created_at_utc: str | None,
    filled_at_utc: str | None,
) -> tuple[float | None, str]:
    if filled_at_utc is None:
        return None, ExecutionLatencyStatus.NOT_FILLED.value
    if decision_created_at_utc is None:
        return None, ExecutionLatencyStatus.MISSING_DECISION_TIMESTAMP.value
    decision_dt = parse_ts(decision_created_at_utc)
    filled_dt = parse_ts(filled_at_utc)
    if decision_dt is None or filled_dt is None:
        return None, ExecutionLatencyStatus.MISSING_DECISION_TIMESTAMP.value
    latency_ms = (filled_dt - decision_dt).total_seconds() * 1000.0
    return latency_ms, ExecutionLatencyStatus.OK.value


@dataclass
class PaperOrderSimulator:
    state_machine: OrderStateMachine = field(default_factory=OrderStateMachine)
    price_oracle: DemoPriceOracle = field(default_factory=DemoPriceOracle)
    latency_audit: list[dict[str, Any]] = field(default_factory=list)

    def evaluate_paper_decision(
        self,
        traceability: dict[str, Any],
        *,
        allow_audit_blockers: bool = False,
    ) -> tuple[str, list[str]]:
        """Deterministic demo policy — LLM cannot approve alone."""
        candidate_id = traceability.get("candidate_id")
        if not candidate_id:
            return PaperTradeDecisionStatus.DEMO_PAPER_REJECTED_MISSING_IDENTITY.value, [
                "missing_candidate_id"
            ]

        audit_blockers = traceability.get("audit_blockers") or []
        audit_verdict = traceability.get("audit_verdict") or ""

        if audit_blockers and not allow_audit_blockers:
            return PaperTradeDecisionStatus.DEMO_PAPER_REJECTED_AUDIT_BLOCKED.value, [
                "audit_blockers_present",
                *audit_blockers,
            ]

        if audit_verdict in ("BUY", "SELL", "EXECUTE", "PAPER_BUY", "APPROVE_TRADE"):
            return PaperTradeDecisionStatus.DEMO_PAPER_REJECTED_AUDIT_BLOCKED.value, [
                "llm_verdict_cannot_approve_trade_alone"
            ]

        return PaperTradeDecisionStatus.DEMO_PAPER_BUY_CANDIDATE.value, []

    def _base_order(
        self,
        traceability: dict[str, Any],
        *,
        price_result: dict[str, Any],
        symbol: str,
        pair_address: str,
        decision_created_at_utc: str | None,
        order_created_at_utc: str | None,
    ) -> PaperOrder:
        created = order_created_at_utc or utc_now_iso()
        return PaperOrder(
            candidate_id=traceability.get("candidate_id") or "",
            symbol=symbol,
            pair_address=pair_address,
            side="BUY",
            order_type="MARKET",
            source_decision_id=traceability.get("source_decision_id"),
            source_context_record_id=traceability.get("source_context_record_id"),
            source_llm_audit_record_id=traceability.get("source_llm_audit_record_id"),
            decision_created_at_utc=decision_created_at_utc,
            decision_status=traceability.get("decision_status"),
            consensus_family=traceability.get("consensus_family"),
            context_schema_id=traceability.get("context_schema_id"),
            audit_verdict=traceability.get("audit_verdict"),
            audit_blockers=list(traceability.get("audit_blockers") or []),
            audit_warnings=list(traceability.get("audit_warnings") or []),
            scoring_policy_id=traceability.get("scoring_policy_id"),
            execution_mode="PAPER_DEMO",
            no_wallet_dry_run=False,
            no_live_submission=True,
            max_price_age_seconds=price_result.get("max_price_age_seconds", 30.0),
            price_source=price_result.get("price_source"),
            price_snapshot_id=price_result.get("price_snapshot_id"),
            price_timestamp=price_result.get("price_timestamp_used") or price_result.get("price_timestamp"),
            price_age_seconds=price_result.get("price_age_seconds"),
            price_status=price_result.get("price_status"),
            created_at_utc=created,
            status=PaperOrderStatus.PAPER_PENDING.value,
        )

    def build_pending_order(
        self,
        traceability: dict[str, Any],
        *,
        price_result: dict[str, Any],
        coin_id: int | None = None,
        symbol: str = "",
        pair_address: str = "",
        notional_usd: float = DEFAULT_NOTIONAL_USD,
        allow_audit_blockers: bool = False,
        decision_created_at_utc: str | None = None,
        order_created_at_utc: str | None = None,
    ) -> PaperOrder:
        del coin_id, notional_usd
        order = self._base_order(
            traceability,
            price_result=price_result,
            symbol=symbol,
            pair_address=pair_address,
            decision_created_at_utc=decision_created_at_utc,
            order_created_at_utc=order_created_at_utc,
        )

        decision_status, _ = self.evaluate_paper_decision(
            traceability, allow_audit_blockers=allow_audit_blockers
        )

        if not order.candidate_id:
            order.status = PaperOrderStatus.PAPER_REJECTED.value
            order.paper_trade_reason = PaperTradeDecisionStatus.DEMO_PAPER_REJECTED_MISSING_IDENTITY.value
            return order

        if decision_status == PaperTradeDecisionStatus.DEMO_PAPER_REJECTED_AUDIT_BLOCKED.value:
            order.status = PaperOrderStatus.PAPER_REJECTED.value
            order.paper_trade_reason = PaperTradeDecisionStatus.DEMO_PAPER_REJECTED_AUDIT_BLOCKED.value
            return order

        price_status = price_result.get("price_status")
        if price_status != PriceStatus.PRICE_OK.value:
            order.status = PaperOrderStatus.PAPER_REJECTED.value
            order.paper_trade_reason = PaperTradeDecisionStatus.DEMO_PAPER_REJECTED_MISSING_PRICE.value
            return order

        price = price_result.get("price")
        if price is None or float(price) <= 0:
            order.status = PaperOrderStatus.PAPER_REJECTED.value
            order.paper_trade_reason = PaperTradeDecisionStatus.DEMO_PAPER_REJECTED_MISSING_PRICE.value
            return order

        return order

    def complete_fill(
        self,
        order: PaperOrder,
        *,
        price_result: dict[str, Any],
        notional_usd: float = DEFAULT_NOTIONAL_USD,
        traceability: dict[str, Any],
        allow_audit_blockers: bool = False,
        decision_created_at_utc: str | None = None,
    ) -> PaperOrder:
        price = price_result.get("price")
        if price is None or float(price) <= 0:
            order.status = PaperOrderStatus.PAPER_REJECTED.value
            order.paper_trade_reason = PaperTradeDecisionStatus.DEMO_PAPER_REJECTED_MISSING_PRICE.value
            self.state_machine.transition(
                order.paper_order_id,
                PaperOrderStatus.PAPER_PENDING.value,
                PaperOrderStatus.PAPER_REJECTED.value,
            )
            return order

        order.requested_price_usd = float(price)
        order.filled_price_usd = float(price)
        order.notional_usd = notional_usd
        order.quantity = notional_usd / float(price)

        if traceability.get("audit_blockers") and allow_audit_blockers:
            order.not_model_approved = True
            order.not_live_approved = True
            order.paper_trade_reason = "DEMO_SANDBOX_EXPLORATION"
            order.override_type = "DEMO_ONLY_USER_APPROVED_EXPLORATION"

        transition = self.state_machine.transition(
            order.paper_order_id,
            PaperOrderStatus.PAPER_PENDING.value,
            PaperOrderStatus.PAPER_FILLED.value,
        )
        if not transition.allowed:
            order.status = PaperOrderStatus.PAPER_REJECTED.value
            order.paper_trade_reason = PaperTradeDecisionStatus.DEMO_PAPER_REJECTED_STATE_MACHINE.value
            return order

        order.status = PaperOrderStatus.PAPER_FILLED.value
        order.filled_at_utc = utc_now_iso()
        order.paper_trade_reason = order.paper_trade_reason or PaperTradeDecisionStatus.DEMO_PAPER_FILLED.value

        latency_ms, latency_status = compute_execution_latency(
            decision_created_at_utc, order.filled_at_utc
        )
        order.execution_latency_ms = latency_ms
        order.execution_latency_status = latency_status
        self._record_latency_audit(order)
        return order

    def create_and_fill_order(
        self,
        traceability: dict[str, Any],
        *,
        price_result: dict[str, Any],
        coin_id: int | None = None,
        symbol: str = "",
        pair_address: str = "",
        notional_usd: float = DEFAULT_NOTIONAL_USD,
        allow_audit_blockers: bool = False,
        decision_created_at_utc: str | None = None,
        order_created_at_utc: str | None = None,
    ) -> PaperOrder:
        """Backward-compatible single-shot helper (used in unit tests)."""
        pending = self.build_pending_order(
            traceability,
            price_result=price_result,
            coin_id=coin_id,
            symbol=symbol,
            pair_address=pair_address,
            notional_usd=notional_usd,
            allow_audit_blockers=allow_audit_blockers,
            decision_created_at_utc=decision_created_at_utc,
            order_created_at_utc=order_created_at_utc,
        )
        if pending.status == PaperOrderStatus.PAPER_REJECTED.value:
            if pending.paper_trade_reason == PaperTradeDecisionStatus.DEMO_PAPER_REJECTED_MISSING_PRICE.value:
                self.state_machine.transition(
                    pending.paper_order_id,
                    PaperOrderStatus.PAPER_PENDING.value,
                    PaperOrderStatus.PAPER_REJECTED.value,
                )
            return pending
        return self.complete_fill(
            pending,
            price_result=price_result,
            notional_usd=notional_usd,
            traceability=traceability,
            allow_audit_blockers=allow_audit_blockers,
            decision_created_at_utc=decision_created_at_utc,
        )

    def _record_latency_audit(self, order: PaperOrder) -> None:
        self.latency_audit.append(
            {
                "paper_order_id": order.paper_order_id,
                "candidate_id": order.candidate_id,
                "decision_created_at_utc": order.decision_created_at_utc,
                "order_created_at_utc": order.created_at_utc,
                "filled_at_utc": order.filled_at_utc,
                "execution_latency_ms": order.execution_latency_ms,
                "execution_latency_status": order.execution_latency_status,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            }
        )
