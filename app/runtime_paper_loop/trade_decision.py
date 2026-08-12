"""Hierarchical AE11B trade decision record builder."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.runtime_paper_loop.decision_policy import PolicyDecision
from app.runtime_paper_loop.price_freshness import PriceFreshnessResult
from app.runtime_paper_loop.types import AE11B_SCHEMA_VERSION, EXPLORATION_OVERRIDE_TYPE, EXPLORATION_TRADE_AUTHORITY, utc_now_iso


def build_hierarchical_trade_decision(
    *,
    source_decision_id: str | None,
    source_event_key: str | None,
    candidate_id: str | None,
    pair_address: str | None,
    loop_run_id: str,
    loop_iteration: int,
    policy: PolicyDecision,
    price_freshness: PriceFreshnessResult,
    paper_action: str,
    paper_order_id: str | None = None,
    decision: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build AE11B hierarchical trade decision JSONL record."""
    no_trade_authority = bool((decision or {}).get("no_trade_authority", True))
    strict_decision = policy.strict_mode_decision or "BLOCKED"
    exploration_decision = policy.exploration_mode_decision or "NO_TRADE"

    return {
        "record_type": "TRADE_DECISION",
        "schema_version": AE11B_SCHEMA_VERSION,
        "decision_id": str(uuid4()),
        "source_decision_id": source_decision_id,
        "source_event_key": source_event_key,
        "candidate_id": candidate_id,
        "pair_address": pair_address,
        "created_at_utc": utc_now_iso(),
        "loop_run_id": loop_run_id,
        "loop_iteration": loop_iteration,
        "strict_mode": {
            "decision": strict_decision,
            "reason": policy.strict_reason,
            "blockers": list(policy.strict_blockers),
            "price_age_seconds": price_freshness.price_age_seconds,
            "max_price_age_seconds": price_freshness.strict_max_price_age_seconds,
            "no_trade_authority_respected": True,
            "not_live_approved": True,
        },
        "exploration_mode": {
            "decision": exploration_decision,
            "reason": policy.exploration_reason,
            "override_type": policy.override_type if policy.should_trade_exploration else None,
            "trade_authority": policy.trade_authority if policy.should_trade_exploration else None,
            "not_model_approved": True,
            "not_live_approved": True,
            "original_no_trade_authority": no_trade_authority,
            "paper_order_id": paper_order_id,
            "hard_safety_gates_passed": policy.hard_safety_gates_passed,
        },
        "hard_safety": {
            "wallet_configured": False,
            "private_key_accessed": False,
            "real_transaction_attempted": False,
            "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
            "valid_pair_address": bool(pair_address),
            "price_available": price_freshness.price is not None and float(price_freshness.price or 0) > 0,
            "price_not_stale_for_exploration": price_freshness.exploration_price_fresh,
            "duplicate_source_decision": policy.duplicate_reason == "DUPLICATE_DECISION_ID",
            "active_pair_lock": policy.duplicate_active_pair,
            "cooldown_active": policy.cooldown_active,
            "max_open_positions_exceeded": policy.max_open_positions_hit,
            "essential_identity_missing": policy.missing_identity,
        },
        "paper_action_taken": paper_action,
        "strict_shadow_decision": policy.strict_shadow_decision,
        "exploration_decision": policy.exploration_decision,
        "reason_for_no_trade": policy.reason_for_no_trade,
        "duplicate_reason": policy.duplicate_reason,
    }
