"""LLM short-circuit gate structure — Phase 1 preserves current call behavior."""
from __future__ import annotations

from typing import Any

from ..llm_config import (
    get_llm_provider,
    is_headless_data_collection,
    is_ollama_provider_active,
    ollama_budget_remaining,
    try_consume_ollama_call,
)
from .audit_reasons import AuditReason

BEARISH_ALERT_TYPES = frozenset({"LARGE_SELL", "DISTRIBUTION"})


def evaluate_llm_short_circuit(
    *,
    alert: dict[str, Any] | None,
    coin_id: int | None,
    whale_score: float,
    llm_threshold: float,
    signal_action: str | None = None,
    alert_type: str | None = None,
    has_open_position: bool = False,
    price_usd: float | None = None,
    pair_address: str | None = None,
    trading_mode: str = "DEMO",
    auto_execution_enabled: bool = True,
    enforce_risk_gate: bool = False,
    risk_score: int | None = None,
    max_risk_score: int = 70,
    economic_model_approved: bool | None = None,
    dry_run_budget_check: bool = True,
) -> tuple[bool, list[str]]:
    """
    Evaluate whether LLM should be invoked.

    Phase 1: only enforces gates that currently block LLM calls in live.py.
    Additional gates are audited but do not block (deferred to Phase 2).
    Returns (should_call_llm, audit_reasons).
    """
    reasons: list[str] = []

    # Gate 1 — hard settings (currently enforced via live.py pre-check)
    if not alert:
        reasons.append(AuditReason.ALERT_REQUIRED_BUT_MISSING.value)
        return False, reasons

    if coin_id is None:
        reasons.append(AuditReason.MISSING_PRICE_OR_PAIR.value)
        return False, reasons

    if whale_score < llm_threshold:
        reasons.append(AuditReason.LLM_THRESHOLD_NOT_MET.value)
        return False, reasons

    if not pair_address:
        reasons.append(AuditReason.MISSING_PRICE_OR_PAIR.value)
        return False, reasons

    if price_usd is None or price_usd <= 0:
        reasons.append(AuditReason.MISSING_PRICE_OR_PAIR.value)
        # Phase 1: live.py already has price from pair; do not add new block
        reasons.append(AuditReason.MISSING_MODEL_SNAPSHOT_PRICE.value)

    if str(trading_mode).upper() == "LIVE":
        reasons.append(AuditReason.SETTINGS_BLOCKED.value)

    if not auto_execution_enabled:
        reasons.append(AuditReason.SETTINGS_BLOCKED.value)

    # Gate 2 — bearish veto (audit only in Phase 1; does NOT block LLM)
    effective_alert = alert_type or (alert.get("alert_type") if alert else None)
    effective_signal = signal_action or ""
    if effective_alert in BEARISH_ALERT_TYPES:
        reasons.append(AuditReason.BLOCKED_BY_BEARISH_ALERT_DETAIL.value)
    if effective_signal == "BUY" and effective_alert in BEARISH_ALERT_TYPES:
        reasons.append(AuditReason.CONFLICT_ENGINE_BUY_BEARISH_ALERT.value)

    # Gate 3 — economic model (audit only Phase 1)
    if economic_model_approved is False:
        reasons.append(AuditReason.BLOCKED_BY_ECONOMIC_MODEL_DETAIL.value)
    elif economic_model_approved is None:
        reasons.append(AuditReason.MISSING_MODEL_PREDICTION.value)

    # Gate 4 — duplicate pair (audit only Phase 1)
    if has_open_position:
        reasons.append(AuditReason.DUPLICATE_PAIR.value)

    # Gate 5 — risk gate (audit only unless enforce_risk_gate set)
    if enforce_risk_gate and risk_score is not None and risk_score > max_risk_score:
        reasons.append(AuditReason.BLOCKED_BY_RISK_GATE.value)

    # In-call gates (audited; blocking deferred to analyze_market_state in Phase 1)
    if is_headless_data_collection() or get_llm_provider() == "none":
        reasons.append(AuditReason.LLM_SHORT_CIRCUITED.value)

    if is_ollama_provider_active() and dry_run_budget_check and ollama_budget_remaining() <= 0:
        reasons.append(AuditReason.LLM_BUDGET_BLOCKED.value)
        reasons.append(AuditReason.LLM_BUDGET_BLOCKED_DETAIL.value)

    return True, reasons


def peek_ollama_budget() -> int:
    return ollama_budget_remaining()


BLOCKING_LLM_REASONS = frozenset({
    AuditReason.BLOCKED_BY_SLIPPAGE_LIMIT.value,
    AuditReason.BLOCKED_BY_PRICE_DRIFT.value,
    AuditReason.MODEL_SNAPSHOT_TOO_OLD.value,
    AuditReason.MARKET_SNAPSHOT_TOO_OLD.value,
    AuditReason.MODEL_PREDICTION_TOO_OLD.value,
    AuditReason.MODEL_ARTIFACT_TOO_OLD.value,
    AuditReason.MISSING_MODEL_SNAPSHOT_PRICE.value,
    AuditReason.BLOCKED_BY_MISSING_SLIPPAGE_INPUTS.value,
    AuditReason.BLOCKED_BY_ECONOMIC_MODEL.value,
    AuditReason.BLOCKED_BY_BEARISH_ALERT.value,
    AuditReason.BLOCKED_BY_RISK_GATE.value,
    AuditReason.BLOCKED_BY_DUPLICATE_PAIR.value,
    AuditReason.PROBABILITY_BELOW_THRESHOLD.value,
    AuditReason.EXPECTED_RETURN_BELOW_MARGIN.value,
    AuditReason.MODEL_RUNTIME_INFERENCE_NOT_AVAILABLE.value,
    AuditReason.MODEL_ARTIFACT_LOAD_FAILED.value,
    AuditReason.MODEL_SCHEMA_MISMATCH.value,
    AuditReason.MODEL_FEATURE_MISSING.value,
    AuditReason.MODEL_FEATURE_EXTRA.value,
    AuditReason.MODEL_SCHEMA_METADATA_MISSING.value,
    AuditReason.MODEL_PREPROCESSOR_MISSING.value,
    AuditReason.MODEL_TRAINED_WITH_TARGET_LEAKAGE.value,
    AuditReason.PRICE_FILL_RESOLUTION_FAILED.value,
})


def evaluate_llm_short_circuit_phase2(
    *,
    economic_approved: bool,
    candidate: Any,
    gate_result: Any,
    settings: dict[str, Any],
    alert: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """
    Phase 2 LLM gate — only after economic model approval.
    Blocks LLM for slippage/price-drift/stale-model/bearish/risk/duplicate failures.
    """
    reasons: list[str] = []

    if not economic_approved:
        reasons.append(AuditReason.BLOCKED_BY_ECONOMIC_MODEL.value)
        return False, reasons

    for r in (gate_result.reasons if gate_result else []):
        if r in BLOCKING_LLM_REASONS:
            reasons.append(r)
            return False, reasons

    if candidate.bearish_alert_active:
        reasons.append(AuditReason.BLOCKED_BY_BEARISH_ALERT.value)
        return False, reasons

    if candidate.existing_open_position_for_pair:
        reasons.append(AuditReason.BLOCKED_BY_DUPLICATE_PAIR.value)
        return False, reasons

    if not candidate.pair_address or candidate.coin_id is None:
        reasons.append(AuditReason.MISSING_PRICE_OR_PAIR.value)
        return False, reasons

    if candidate.current_execution_price is None or candidate.current_execution_price <= 0:
        reasons.append(AuditReason.PRICE_FILL_RESOLUTION_FAILED.value)
        return False, reasons

    if is_headless_data_collection() or get_llm_provider() == "none":
        reasons.append(AuditReason.LLM_SHORT_CIRCUITED.value)
        return False, reasons

    if is_ollama_provider_active() and ollama_budget_remaining() <= 0:
        reasons.append(AuditReason.LLM_BUDGET_BLOCKED.value)
        reasons.append(AuditReason.LLM_SKIPPED_BUDGET.value)
        return False, reasons

    return True, reasons
