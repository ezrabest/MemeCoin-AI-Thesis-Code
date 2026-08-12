"""AE12.7 RSS / news sentiment context linker (reporting only)."""

from __future__ import annotations

from typing import Any

from app.intelligent_agents.types import (
    AgentStatus,
    AgentType,
    DecisionEffect,
    SourceMode,
    make_agent_record,
)

HYPE_TERMS = ("moon", "100x", "pump", "gem", "ape")
SCAM_TERMS = ("rug", "honeypot", "scam", "drain", "phishing")


def link_rss_context(
    candidate: dict[str, Any],
    *,
    existing_rss: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Link RSS/news sentiment context if available from artifacts/local DB summary.
    Never creates trades. Never fetches externally in AE12.7 default path.
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
    rss = dict(existing_rss or candidate.get("rss_context") or {})

    if not rss:
        return make_agent_record(
            agent_type=AgentType.RSS_CONTEXT,
            source_mode=SourceMode.ARTIFACT_ONLY,
            agent_status=AgentStatus.NOT_CONFIGURED,
            agent_summary="RSS/news sentiment context not available for this candidate.",
            decision_effect=DecisionEffect.NO_EFFECT,
            warnings=["rss_context_unavailable"],
            missing_context_flags=["rss_missing"],
            input_artifact_refs=refs,
            **link_ids,
            extra={
                "linkage_method": "unresolved",
                "source_count": 0,
                "rss_trade_authority": False,
            },
        )

    headlines = rss.get("headlines") or rss.get("titles") or []
    blob = " ".join(str(h) for h in headlines).lower()
    pos_terms = [t for t in ("bullish", "surge", "rally", "partnership") if t in blob]
    neg_terms = [t for t in ("bearish", "dump", "hack", "lawsuit") if t in blob]
    hype = [t for t in HYPE_TERMS if t in blob]
    scam = [t for t in SCAM_TERMS if t in blob]

    linkage = rss.get("linkage_method") or "symbol-level" if candidate.get("symbol") else "unresolved"
    if rss.get("global"):
        linkage = "global"
    if rss.get("coin_level"):
        linkage = "coin-level"

    soft = []
    if scam:
        soft.append("rss_scam_rug_warning_terms")
    if hype:
        soft.append("rss_hype_warning_terms")

    return make_agent_record(
        agent_type=AgentType.RSS_CONTEXT,
        source_mode=SourceMode.ARTIFACT_ONLY,
        agent_status=AgentStatus.READ_FROM_EXISTING_ARTIFACT if rss.get("from_artifact") else AgentStatus.GENERATED,
        agent_summary=rss.get("summary")
        or f"RSS sentiment linked ({linkage}); sources={rss.get('source_count', 0)}.",
        decision_effect=DecisionEffect.CONTEXT_ONLY if not soft else DecisionEffect.SOFT_WARNING_ONLY,
        warnings=soft,
        soft_veto_flags=soft[:1],
        input_artifact_refs=refs,
        **link_ids,
        extra={
            "latest_rss_sentiment_summary": rss.get("summary") or rss.get("sentiment_mean"),
            "source_count": rss.get("source_count") or rss.get("rss_source_count_24h") or 0,
            "source_reliability": rss.get("source_reliability"),
            "positive_headline_terms": pos_terms,
            "negative_headline_terms": neg_terms,
            "hype_scam_rug_warning_terms": hype + scam,
            "sentiment_source_timestamp": rss.get("timestamp") or rss.get("source_timestamp"),
            "linkage_method": linkage,
            "rss_trade_authority": False,
        },
    )
