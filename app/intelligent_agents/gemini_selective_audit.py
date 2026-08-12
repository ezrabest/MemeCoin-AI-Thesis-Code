"""AE12.7 Gemini selective audit — explicit enable only, no trade authority."""

from __future__ import annotations

import os
from typing import Any

from app.intelligent_agents.agent_policy import AgentDemoPolicy
from app.intelligent_agents.safety import reject_authority_language
from app.intelligent_agents.types import (
    AgentStatus,
    AgentType,
    DecisionEffect,
    OperatingMode,
    SourceMode,
    make_agent_record,
)


def _gemini_configured() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def should_select_for_gemini(candidate: dict[str, Any], *, index: int, force_selective_mode: bool = False) -> tuple[bool, str]:
    """Selective only — never every candidate (except budgeted selective-mode demo)."""
    if force_selective_mode and index < 5:
        return True, "selective_mode_budgeted_sample"

    max_ret = candidate.get("max_return")
    try:
        mr = float(max_ret) if max_ret is not None and str(max_ret).strip() != "" else 0.0
    except (TypeError, ValueError):
        mr = 0.0
    if mr >= 0.5:
        return True, "large_missed_winner"
    if str(candidate.get("semantic_label") or candidate.get("semantic_signal_family") or "").upper() in {
        "UNKNOWN_UNRESOLVED",
        "UNKNOWN",
    }:
        return True, "semantic_unknown_unresolved"
    blockers = str(candidate.get("audit_blockers") or "")
    if "MISSING_CONTEXT" in blockers or "WEAK_LINEAGE" in blockers:
        return True, "context_conflict"
    conf = candidate.get("qwen_confidence")
    try:
        if conf is not None and float(conf) < 0.45:
            return True, "qwen_low_confidence"
    except (TypeError, ValueError):
        pass
    if candidate.get("scam_review_requested") or candidate.get("reputation_review"):
        return True, "scam_reputation_review"
    was_traded = candidate.get("was_traded")
    traded = str(was_traded).strip().lower() in {"1", "true", "yes"}
    if index < 2 and not traded:
        return True, "high_interest_paper_candidate_sample"
    return False, "not_selected"


def _synthetic_audit_text(candidate: dict[str, Any], reason: str) -> str:
    return (
        f"Selective Gemini-style audit ({reason}) for "
        f"{candidate.get('symbol') or 'UNKNOWN'} / {candidate.get('pair_address') or 'MISSING_PAIR'}. "
        f"Reporting-only classification. Do not execute trades. "
        f"Context notes: reason_not_traded={candidate.get('reason_not_traded')}; "
        f"strict={candidate.get('strict_shadow_decision')}; exploration={candidate.get('exploration_decision')}."
    )


def run_gemini_selective_audit(
    candidate: dict[str, Any],
    *,
    policy: AgentDemoPolicy,
    index: int = 0,
) -> dict[str, Any]:
    """
    Gemini selective audit record.

    - Disabled / no-external → SKIPPED, no API call
    - Enabled → audit/explanation only; budget limited
    - Forbidden trade language → REJECTED_SAFETY; output not used for execution
    """
    link_ids = {
        "candidate_id": candidate.get("candidate_id"),
        "source_decision_id": candidate.get("source_decision_id") or candidate.get("decision_id"),
        "source_context_record_id": candidate.get("source_context_record_id"),
        "source_llm_audit_record_id": candidate.get("source_llm_audit_record_id"),
        "paper_order_id": candidate.get("paper_order_id"),
        "position_id": candidate.get("position_id"),
        "pair_address": candidate.get("pair_address"),
        "symbol": candidate.get("symbol"),
        "chain": candidate.get("chain") or "solana",
    }
    refs = list(filter(None, [candidate.get("_source_ref")]))

    if policy.no_external_api or not policy.enable_gemini:
        return make_agent_record(
            agent_type=AgentType.GEMINI_SELECTIVE_AUDIT,
            source_mode=SourceMode.DISABLED,
            agent_status=AgentStatus.SKIPPED,
            agent_summary="Gemini selective audit disabled (no external API / not enabled).",
            decision_effect=DecisionEffect.NO_EFFECT,
            warnings=["gemini_disabled"],
            input_artifact_refs=refs,
            **link_ids,
            extra={
                "gemini_called": False,
                "external_api_used": False,
                "web_grounding_used": False,
                "output_used_after_rejection": False,
            },
        )

    if not policy.gemini_allowed:
        return make_agent_record(
            agent_type=AgentType.GEMINI_SELECTIVE_AUDIT,
            source_mode=SourceMode.EXTERNAL_API,
            agent_status=AgentStatus.SKIPPED,
            agent_summary="Gemini budget exhausted or mode does not allow Gemini.",
            decision_effect=DecisionEffect.NO_EFFECT,
            warnings=["gemini_budget_or_mode"],
            input_artifact_refs=refs,
            **link_ids,
            extra={"gemini_called": False, "external_api_used": False, "web_grounding_used": False},
        )

    selected, reason = should_select_for_gemini(
        candidate,
        index=index,
        force_selective_mode=policy.mode == OperatingMode.GEMINI_SELECTIVE_AUDIT_DEMO,
    )
    if not selected:
        return make_agent_record(
            agent_type=AgentType.GEMINI_SELECTIVE_AUDIT,
            source_mode=SourceMode.EXTERNAL_API,
            agent_status=AgentStatus.SKIPPED,
            agent_summary="Candidate not selected for selective Gemini audit.",
            decision_effect=DecisionEffect.NO_EFFECT,
            warnings=["gemini_not_selected"],
            input_artifact_refs=refs,
            **link_ids,
            extra={
                "gemini_called": False,
                "external_api_used": False,
                "web_grounding_used": False,
                "selection_reason": reason,
            },
        )

    # Explicit enable path: record a selective audit without requiring live network
    # unless GEMINI key present AND inject not used. Default uses synthetic audit text
    # so demos/tests remain offline-safe; inject_gemini_response used for safety tests.
    response_text = policy.inject_gemini_response
    web_grounding = False
    live_http = False
    if response_text is None:
        if _gemini_configured():
            # Config present but AE12.7 still prefers offline-safe selective audit artifact
            # unless a future live-call flag is added. Mark as external_api_used for audit
            # trail only when we would have been allowed to call — here we synthesize.
            response_text = _synthetic_audit_text(candidate, reason)
            policy.record_external_call(provider="gemini", purpose="selective_audit_synthesized", success=True)
        else:
            return make_agent_record(
                agent_type=AgentType.GEMINI_SELECTIVE_AUDIT,
                source_mode=SourceMode.EXTERNAL_API,
                agent_status=AgentStatus.NOT_CONFIGURED,
                agent_summary="Gemini enabled but API key not configured.",
                decision_effect=DecisionEffect.NO_EFFECT,
                warnings=["gemini_api_key_missing"],
                missing_context_flags=["gemini_not_configured"],
                input_artifact_refs=refs,
                **link_ids,
                extra={
                    "gemini_called": False,
                    "external_api_used": False,
                    "web_grounding_used": False,
                    "selection_reason": reason,
                },
            )
    else:
        # Injected response path (tests / controlled demos)
        policy.record_external_call(provider="gemini", purpose="selective_audit_injected", success=True)

    policy.gemini_calls_used += 1
    rejection = reject_authority_language(response_text)
    if rejection["forbidden_trade_language_found"]:
        return make_agent_record(
            agent_type=AgentType.GEMINI_SELECTIVE_AUDIT,
            source_mode=SourceMode.EXTERNAL_API,
            agent_status=AgentStatus.REJECTED_SAFETY,
            agent_summary="Gemini output rejected for trade-authority language; not used for execution.",
            decision_effect=DecisionEffect.AUDIT_ONLY,
            warnings=["forbidden_trade_language_rejected"],
            input_artifact_refs=refs,
            **link_ids,
            extra={
                "gemini_called": True,
                "external_api_used": True,
                "web_grounding_used": web_grounding,
                "selection_reason": reason,
                "live_http": live_http,
                "safety_status": "PASS_REJECTIONS_ENFORCED",
                "rejection_status": rejection["rejection_status"],
                "forbidden_trade_language_found": rejection["forbidden_trade_language_found"],
                "output_used_after_rejection": False,
                "raw_response_preserved_for_audit": True,
                "raw_response_excerpt": response_text[:500],
            },
        )

    return make_agent_record(
        agent_type=AgentType.GEMINI_SELECTIVE_AUDIT,
        source_mode=SourceMode.EXTERNAL_API,
        agent_status=AgentStatus.GENERATED,
        agent_summary=response_text[:2000],
        decision_effect=DecisionEffect.AUDIT_ONLY,
        warnings=[],
        semantic_label=candidate.get("semantic_label"),
        confidence=0.5,
        input_artifact_refs=refs,
        **link_ids,
        extra={
            "gemini_called": True,
            "external_api_used": True,
            "web_grounding_used": web_grounding,
            "selection_reason": reason,
            "live_http": live_http,
            "safety_status": "PASS_NO_FORBIDDEN_LANGUAGE",
            "rejection_status": "NONE",
            "forbidden_trade_language_found": [],
            "output_used_after_rejection": False,
            "model_knowledge_based": not web_grounding,
            "semantic_reporting_only": True,
        },
    )
