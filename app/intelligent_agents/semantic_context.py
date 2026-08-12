"""AE12.7 semantic / SentimentFix dual-axis context (reporting only)."""

from __future__ import annotations

from typing import Any

from app.intelligent_agents.types import (
    AgentStatus,
    AgentType,
    DecisionEffect,
    SourceMode,
    make_agent_record,
)


def _normalize_unknown(label: str | None) -> str:
    s = (label or "").strip().upper()
    if s in {"", "NONE", "NULL", "N/A", "UNKNOWN"}:
        return "UNKNOWN_UNRESOLVED"
    return s


def link_semantic_context(
    candidate: dict[str, Any],
    *,
    existing_semantic: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Expose dual-axis taxonomy fields without converting UNKNOWN_UNRESOLVED
    into social/opportunistic, and without treating legacy cluster as final.
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
    sem = dict(existing_semantic or {})

    # Prefer explicit dual-axis fields
    family = sem.get("semantic_signal_family") or candidate.get("semantic_signal_family")
    trade_state = sem.get("trading_opportunity_state") or candidate.get("trading_opportunity_state")
    legacy = sem.get("legacy_cluster_label") or candidate.get("legacy_cluster_label") or candidate.get("cluster_label")

    # Never promote legacy OPPORTUNISTIC into semantic family
    family_norm = _normalize_unknown(str(family) if family is not None else None)
    if "OPPORTUNISTIC" in family_norm or "SPECULATIVE" in family_norm:
        family_norm = "UNKNOWN_UNRESOLVED"

    # UNKNOWN_UNRESOLVED means: not social, not opportunistic, insufficient evidence
    if family_norm == "UNKNOWN_UNRESOLVED":
        assert_not_social = True
        assert_not_opp = True
    else:
        assert_not_social = family_norm != "SOCIAL"
        assert_not_opp = "OPPORTUNISTIC" not in family_norm

    label = family_norm
    if not family and not sem and not candidate.get("semantic_label"):
        status = AgentStatus.UNKNOWN_UNRESOLVED
        summary = "Semantic classification unresolved — insufficient evidence (not social, not opportunistic)."
    elif family_norm == "UNKNOWN_UNRESOLVED":
        status = AgentStatus.UNKNOWN_UNRESOLVED
        summary = "UNKNOWN_UNRESOLVED: not social; not opportunistic; insufficient evidence."
    else:
        status = AgentStatus.READ_FROM_EXISTING_ARTIFACT if sem.get("from_artifact") else AgentStatus.GENERATED
        summary = (
            f"semantic_signal_family={family_norm}; "
            f"trading_opportunity_state={trade_state or 'UNKNOWN'}; "
            f"legacy_cluster_label={legacy or 'n/a'} (diagnostic only)."
        )

    return make_agent_record(
        agent_type=AgentType.SEMANTIC_CLASSIFICATION,
        source_mode=SourceMode.ARTIFACT_ONLY,
        agent_status=status,
        agent_summary=summary,
        decision_effect=DecisionEffect.CONTEXT_ONLY,
        semantic_label=label,
        confidence=float(sem.get("semantic_signal_confidence") or candidate.get("semantic_confidence") or 0.0) or None,
        warnings=["legacy_cluster_not_final_semantic"] if legacy else [],
        input_artifact_refs=refs,
        **link_ids,
        extra={
            "semantic_signal_family": family_norm,
            "trading_opportunity_state": trade_state or "UNKNOWN",
            "legacy_cluster_label": legacy,
            "legacy_is_final_semantic": False,
            "unknown_unresolved_means": {
                "not_social": True if family_norm == "UNKNOWN_UNRESOLVED" else assert_not_social,
                "not_opportunistic": True if family_norm == "UNKNOWN_UNRESOLVED" else assert_not_opp,
                "insufficient_evidence": family_norm == "UNKNOWN_UNRESOLVED",
            },
            "pair_level_audit_counts": sem.get("pair_level_audit_counts"),
            "coin_level_final_counts": sem.get("coin_level_final_counts"),
            "semantic_trade_authority": False,
            "reporting_only": True,
        },
    )
