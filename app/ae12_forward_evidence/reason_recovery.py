"""Rejection / reason-not-traded recovery without inventing reasons."""

from __future__ import annotations

from typing import Any

from app.ae12_forward_evidence.types import ReasonRecoveryStatus, TRADED_ACTIONS


def _first_nonempty(*vals: Any) -> str | None:
    for v in vals:
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            cleaned = [str(x) for x in v if x is not None and str(x).strip()]
            if cleaned:
                return "|".join(cleaned)
            continue
        s = str(v).strip()
        if s and s.lower() not in {"none", "null", "nan"}:
            return s
    return None


def extract_reason_from_opportunity(row: dict[str, Any]) -> str | None:
    return _first_nonempty(
        row.get("reason_for_no_trade"),
        row.get("reason_not_traded"),
        row.get("rejection_reason"),
        row.get("duplicate_reason"),
    )


def extract_reason_from_trade_decision(td: dict[str, Any] | None) -> str | None:
    if not td:
        return None
    strict = td.get("strict_mode") if isinstance(td.get("strict_mode"), dict) else {}
    exploration = td.get("exploration_mode") if isinstance(td.get("exploration_mode"), dict) else {}
    return _first_nonempty(
        td.get("reason_for_no_trade"),
        td.get("reason_not_traded"),
        td.get("rejection_reason"),
        td.get("duplicate_reason"),
        strict.get("reason"),
        exploration.get("reason"),
        "|".join(str(x) for x in (strict.get("blockers") or []) if x) or None,
    )


def extract_reason_from_ae6(ae6: dict[str, Any] | None) -> str | None:
    if not ae6:
        return None
    reasons = ae6.get("reasons")
    missing = ae6.get("missingness")
    return _first_nonempty(
        ae6.get("reason_for_no_trade"),
        ae6.get("rejection_reason"),
        "|".join(str(x) for x in (reasons or []) if x) if reasons else None,
        (ae6.get("signal_context") or {}).get("reason")
        if isinstance(ae6.get("signal_context"), dict)
        else None,
        "|".join(str(x) for x in (missing or []) if x) if missing else None,
    )


def extract_reason_from_runtime_events(
    events_by_decision: dict[str, dict[str, Any]],
    decision_id: str | None,
    candidate_id: str | None,
) -> str | None:
    for key in (decision_id, candidate_id):
        if not key:
            continue
        ev = events_by_decision.get(str(key))
        if not ev:
            continue
        return _first_nonempty(
            ev.get("reason_for_no_trade"),
            ev.get("rejection_reason"),
            ev.get("blocker"),
            ev.get("blockers"),
        )
    return None


def recover_reason(
    *,
    opportunity: dict[str, Any],
    trade_decision: dict[str, Any] | None,
    ae6: dict[str, Any] | None,
    runtime_events_by_key: dict[str, dict[str, Any]],
    was_traded: bool,
    paper_action_taken: str | None,
) -> dict[str, Any]:
    """Recover rejection/no-trade reason in required priority order. Never invent."""

    action = (paper_action_taken or opportunity.get("paper_action_taken") or "").upper()
    traded = was_traded or action in TRADED_ACTIONS

    direct = extract_reason_from_opportunity(opportunity)
    if direct:
        return {
            "reason_not_traded": direct if not traded else None,
            "rejection_reason": direct,
            "reason_source": "opportunity_capture",
            "reason_recovery_status": ReasonRecoveryStatus.RECOVERED_FROM_OPPORTUNITY.value,
        }

    td_reason = extract_reason_from_trade_decision(trade_decision)
    if td_reason:
        return {
            "reason_not_traded": td_reason if not traded else None,
            "rejection_reason": td_reason,
            "reason_source": "ae11_trade_decisions",
            "reason_recovery_status": ReasonRecoveryStatus.RECOVERED_FROM_TRADE_DECISION.value,
        }

    decision_id = opportunity.get("source_decision_id") or opportunity.get("decision_id")
    candidate_id = opportunity.get("candidate_id")
    ae6_reason = extract_reason_from_ae6(ae6)
    if ae6_reason:
        return {
            "reason_not_traded": ae6_reason if not traded else None,
            "rejection_reason": ae6_reason,
            "reason_source": "ae6_decisions",
            "reason_recovery_status": ReasonRecoveryStatus.RECOVERED_FROM_AE6.value,
        }

    ev_reason = extract_reason_from_runtime_events(
        runtime_events_by_key, decision_id, candidate_id
    )
    if ev_reason:
        return {
            "reason_not_traded": ev_reason if not traded else None,
            "rejection_reason": ev_reason,
            "reason_source": "ae11_runtime_events",
            "reason_recovery_status": ReasonRecoveryStatus.RECOVERED_FROM_RUNTIME_EVENT.value,
        }

    if traded:
        return {
            "reason_not_traded": None,
            "rejection_reason": None,
            "reason_source": "paper_linkage",
            "reason_recovery_status": ReasonRecoveryStatus.TRADED_NO_REJECTION.value,
        }

    return {
        "reason_not_traded": "UNKNOWN_NOT_RECORDED",
        "rejection_reason": "UNKNOWN_NOT_RECORDED",
        "reason_source": "NONE",
        "reason_recovery_status": ReasonRecoveryStatus.MISSING_IN_SOURCE.value,
    }
