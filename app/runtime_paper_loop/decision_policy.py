"""Exploration vs strict shadow decision policy for AE11B."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.paper_trading.types import PaperTradeDecisionStatus
from app.runtime_paper_loop.price_freshness import PriceFreshnessResult
from app.runtime_paper_loop.types import (
    EXPLORATION_OVERRIDE_TYPE,
    EXPLORATION_TRADE_AUTHORITY,
    DuplicateReason,
)


@dataclass
class PolicyDecision:
    exploration_decision: str = "NO_TRADE"
    strict_shadow_decision: str = "NO_TRADE"
    strict_mode_decision: str = "BLOCKED"
    exploration_mode_decision: str = "NO_TRADE"
    strict_reason: str | None = None
    exploration_reason: str | None = None
    strict_blockers: list[str] = field(default_factory=list)
    should_trade_exploration: bool = False
    should_trade_strict: bool = False
    reason_for_no_trade: str | None = None
    duplicate_reason: str | None = None
    blocked_by_ae9: bool = False
    blocked_by_no_trade_authority: bool = False
    blocked_by_missing_consensus: bool = False
    stale_price: bool = False
    missing_context: bool = False
    max_open_positions_hit: bool = False
    cooldown_active: bool = False
    duplicate_active_pair: bool = False
    missing_identity: bool = False
    paper_policy_prevented: bool = False
    hard_safety_gates_passed: bool = False
    override_type: str | None = None
    not_model_approved: bool = True
    not_live_approved: bool = True
    trade_authority: str = EXPLORATION_TRADE_AUTHORITY
    exploration_flags: list[str] = field(default_factory=list)


def _hard_safety_blockers(
    *,
    pair_address: str,
    price_freshness: PriceFreshnessResult,
    has_active_pair_lock: bool,
    cooldown_active: bool,
    max_open_positions_hit: bool,
    missing_identity: bool,
    already_processed: bool,
) -> list[str]:
    blockers: list[str] = []
    if already_processed:
        blockers.append(DuplicateReason.DUPLICATE_DECISION_ID.value)
    if missing_identity:
        blockers.append("essential_identity_missing")
    if not pair_address:
        blockers.append("invalid_pair_address")
    if price_freshness.price_missing:
        blockers.append("price_unavailable")
    if price_freshness.price_timestamp_missing:
        blockers.append("PRICE_TIMESTAMP_MISSING")
    if not price_freshness.exploration_price_fresh:
        blockers.append("price_stale_exploration")
    if has_active_pair_lock:
        blockers.append(DuplicateReason.ACTIVE_PAIR_LOCK.value)
    if cooldown_active:
        blockers.append(DuplicateReason.COOLDOWN_ACTIVE.value)
    if max_open_positions_hit:
        blockers.append("max_open_positions")
    return blockers


def evaluate_strict_shadow_decision(
    *,
    decision: dict[str, Any] | None,
    traceability: dict[str, Any],
    price_freshness: PriceFreshnessResult,
    open_position_count: int,
    max_open_positions: int,
    has_active_pair_lock: bool,
    cooldown_active: bool,
    already_processed: bool,
    missing_identity: bool,
) -> PolicyDecision:
    """Strict mode — no_trade_authority, missing consensus/context, strict price age."""
    result = PolicyDecision()
    result.strict_mode_decision = "BLOCKED"
    result.strict_shadow_decision = "NO_TRADE"
    blockers: list[str] = []

    if already_processed:
        result.duplicate_reason = DuplicateReason.DUPLICATE_DECISION_ID.value
        result.strict_reason = result.duplicate_reason
        result.strict_blockers = [result.duplicate_reason]
        return result

    if missing_identity:
        result.missing_identity = True
        blockers.append("missing_identity")
    if not traceability.get("source_context_record_id"):
        result.missing_context = True
        blockers.append("missing_context")
    if has_active_pair_lock:
        result.duplicate_active_pair = True
        blockers.append(DuplicateReason.ACTIVE_PAIR_LOCK.value)
    if cooldown_active:
        result.cooldown_active = True
        blockers.append(DuplicateReason.COOLDOWN_ACTIVE.value)
    if open_position_count >= max_open_positions:
        result.max_open_positions_hit = True
        blockers.append("max_open_positions")

    if price_freshness.price_missing:
        blockers.append("price_missing")
    elif price_freshness.price_timestamp_missing:
        blockers.append("PRICE_TIMESTAMP_MISSING")
    elif not price_freshness.strict_price_fresh:
        result.stale_price = True
        blockers.append("price_stale_strict")

    if decision and decision.get("no_trade_authority"):
        result.blocked_by_no_trade_authority = True
        blockers.append("no_trade_authority")

    consensus = (decision or {}).get("consensus") or {}
    if consensus.get("consensus_family") == "NO_MODEL_CONSENSUS_AVAILABLE":
        result.blocked_by_missing_consensus = True
        blockers.append("no_model_consensus")

    decision_status = (decision or {}).get("decision_status")
    if decision_status in ("BLOCK", "NO_DECISION"):
        blockers.append(f"decision_status_{decision_status}")

    audit_blockers = traceability.get("audit_blockers") or []
    if audit_blockers:
        result.blocked_by_ae9 = True
        blockers.extend(audit_blockers)

    audit_verdict = traceability.get("audit_verdict") or ""
    if audit_verdict in ("BUY", "SELL", "EXECUTE", "PAPER_BUY", "APPROVE_TRADE"):
        result.blocked_by_ae9 = True
        blockers.append("llm_verdict_cannot_approve_trade_alone")

    result.strict_blockers = blockers
    if blockers:
        result.strict_reason = blockers[0]
        result.reason_for_no_trade = result.strict_reason
        return result

    result.strict_mode_decision = "TRADE"
    result.strict_shadow_decision = "TRADE"
    result.should_trade_strict = True
    return result


def evaluate_exploration_decision(
    strict: PolicyDecision,
    *,
    decision: dict[str, Any] | None,
    exploration_mode: bool,
    enable_paper_demo_orders: bool,
    allow_paper_trades_with_audit_blockers: bool,
    no_real_wallet: bool,
    price_freshness: PriceFreshnessResult,
    traceability: dict[str, Any],
    pair_address: str,
    has_active_pair_lock: bool,
    cooldown_active: bool,
    max_open_positions_hit: bool,
    missing_identity: bool,
    already_processed: bool,
) -> PolicyDecision:
    """Exploration paper mode — no_trade_authority alone does not block."""
    result = PolicyDecision(
        strict_shadow_decision=strict.strict_shadow_decision,
        strict_mode_decision=strict.strict_mode_decision,
        strict_reason=strict.strict_reason,
        strict_blockers=list(strict.strict_blockers),
        should_trade_strict=strict.should_trade_strict,
        blocked_by_ae9=strict.blocked_by_ae9,
        blocked_by_no_trade_authority=strict.blocked_by_no_trade_authority,
        blocked_by_missing_consensus=strict.blocked_by_missing_consensus,
        missing_context=strict.missing_context,
        duplicate_reason=strict.duplicate_reason,
    )
    result.exploration_mode_decision = "NO_TRADE"

    exploration_enabled = (
        exploration_mode
        and enable_paper_demo_orders
        and allow_paper_trades_with_audit_blockers
        and no_real_wallet
    )

    if not exploration_enabled:
        result.exploration_decision = "NO_TRADE"
        result.paper_policy_prevented = True
        result.exploration_reason = "exploration_flags_not_enabled"
        result.reason_for_no_trade = result.exploration_reason
        return result

    hard_blockers = _hard_safety_blockers(
        pair_address=pair_address,
        price_freshness=price_freshness,
        has_active_pair_lock=has_active_pair_lock,
        cooldown_active=cooldown_active,
        max_open_positions_hit=max_open_positions_hit,
        missing_identity=missing_identity,
        already_processed=already_processed,
    )

    if hard_blockers:
        result.exploration_decision = "NO_TRADE"
        result.exploration_reason = hard_blockers[0]
        result.reason_for_no_trade = result.exploration_reason
        if "price_stale" in hard_blockers[0]:
            result.stale_price = True
        if DuplicateReason.ACTIVE_PAIR_LOCK.value in hard_blockers:
            result.duplicate_active_pair = True
        if DuplicateReason.COOLDOWN_ACTIVE.value in hard_blockers:
            result.cooldown_active = True
        if "max_open_positions" in hard_blockers:
            result.max_open_positions_hit = True
        if "essential_identity_missing" in hard_blockers:
            result.missing_identity = True
        return result

    result.hard_safety_gates_passed = True
    result.exploration_mode_decision = "PAPER_BUY"
    result.exploration_decision = "TRADE_EXPLORATION_OVERRIDE"
    result.should_trade_exploration = True
    result.override_type = EXPLORATION_OVERRIDE_TYPE
    result.not_model_approved = True
    result.not_live_approved = True
    result.trade_authority = EXPLORATION_TRADE_AUTHORITY
    result.exploration_reason = "research_candidate_exploration_override"
    result.exploration_flags = [
        "no_trade_authority_preserved_for_strict",
        "audit_blocker_override",
        "missing_consensus_override",
        "exploration_mode",
    ]
    if decision and decision.get("no_trade_authority"):
        result.exploration_flags.append("original_no_trade_authority_true")
    return result
