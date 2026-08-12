"""Paper order state machine with transition audit."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.paper_trading.types import PaperOrderStatus

VALID_TRANSITIONS: dict[str, frozenset[str]] = {
    PaperOrderStatus.PAPER_PENDING.value: frozenset(
        {
            PaperOrderStatus.PAPER_FILLED.value,
            PaperOrderStatus.PAPER_REJECTED.value,
            PaperOrderStatus.PAPER_CANCELED.value,
        }
    ),
    PaperOrderStatus.PAPER_FILLED.value: frozenset(
        {
            PaperOrderStatus.PAPER_CLOSED_TP.value,
            PaperOrderStatus.PAPER_CLOSED_SL.value,
            PaperOrderStatus.PAPER_CLOSED_TIME_STOP.value,
            PaperOrderStatus.PAPER_CLOSED_MANUAL.value,
            PaperOrderStatus.PAPER_EXPIRED.value,
        }
    ),
}

CLOSED_STATES = frozenset(
    {
        PaperOrderStatus.PAPER_CLOSED_TP.value,
        PaperOrderStatus.PAPER_CLOSED_SL.value,
        PaperOrderStatus.PAPER_CLOSED_TIME_STOP.value,
        PaperOrderStatus.PAPER_CLOSED_MANUAL.value,
        PaperOrderStatus.PAPER_EXPIRED.value,
        PaperOrderStatus.PAPER_REJECTED.value,
        PaperOrderStatus.PAPER_CANCELED.value,
    }
)

TERMINAL_STATES = CLOSED_STATES


@dataclass
class StateTransitionResult:
    allowed: bool
    from_state: str
    to_state: str
    reason: str
    status: str = "OK"

    def to_audit_row(self, paper_order_id: str) -> dict[str, Any]:
        return {
            "paper_order_id": paper_order_id,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "transition_allowed": self.allowed,
            "reason": self.reason,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }


@dataclass
class OrderStateMachine:
    """Minimal state machine for PaperOrder lifecycle."""

    audit_log: list[dict[str, Any]] = field(default_factory=list)

    def can_transition(self, from_state: str, to_state: str) -> StateTransitionResult:
        if from_state == to_state:
            return StateTransitionResult(
                allowed=False,
                from_state=from_state,
                to_state=to_state,
                reason="no_op_same_state",
                status="PAPER_STATE_TRANSITION_REJECTED",
            )

        if from_state in TERMINAL_STATES:
            return StateTransitionResult(
                allowed=False,
                from_state=from_state,
                to_state=to_state,
                reason=f"terminal_state_{from_state}_cannot_transition",
                status="PAPER_STATE_TRANSITION_REJECTED",
            )

        if from_state in CLOSED_STATES and to_state == PaperOrderStatus.PAPER_FILLED.value:
            return StateTransitionResult(
                allowed=False,
                from_state=from_state,
                to_state=to_state,
                reason="closed_to_filled_invalid",
                status="PAPER_STATE_TRANSITION_REJECTED",
            )

        if to_state in CLOSED_STATES and from_state != PaperOrderStatus.PAPER_FILLED.value:
            if from_state == PaperOrderStatus.PAPER_PENDING.value:
                return StateTransitionResult(
                    allowed=False,
                    from_state=from_state,
                    to_state=to_state,
                    reason="pending_to_closed_invalid",
                    status="PAPER_STATE_TRANSITION_REJECTED",
                )

        allowed_targets = VALID_TRANSITIONS.get(from_state, frozenset())
        if to_state not in allowed_targets:
            return StateTransitionResult(
                allowed=False,
                from_state=from_state,
                to_state=to_state,
                reason=f"invalid_transition_{from_state}_to_{to_state}",
                status="PAPER_STATE_TRANSITION_REJECTED",
            )

        return StateTransitionResult(
            allowed=True,
            from_state=from_state,
            to_state=to_state,
            reason="valid_transition",
            status="OK",
        )

    def transition(
        self,
        paper_order_id: str,
        from_state: str,
        to_state: str,
    ) -> StateTransitionResult:
        result = self.can_transition(from_state, to_state)
        self.audit_log.append(result.to_audit_row(paper_order_id))
        return result
