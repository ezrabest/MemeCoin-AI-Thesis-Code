"""Serialization helpers for unified candidates (Phase E2)."""

from __future__ import annotations

import json
import math
from typing import Any

from app.candidates.schema import (
    ArtifactLineage,
    CandidateDecisionState,
    CandidateIdentity,
    ConsensusTier,
    DecisionStatus,
    EnrichmentState,
    EnrichmentStatus,
    ExitPolicyContext,
    LLMReviewState,
    LLMReviewStatus,
    MarketContext,
    ModelScores,
    SCHEMA_VERSION,
    UnifiedCandidate,
)

FLAT_COLUMN_ORDER: tuple[str, ...] = (
    "candidate_id",
    "pair_address",
    "chain",
    "event_timestamp",
    "event_timestamp_normalized",
    "timestamp_precision",
    "source",
    "source_artifact_id",
    "source_row_id",
    "coin_id",
    "symbol",
    "price",
    "liquidity_usd",
    "volume_24h",
    "fdv",
    "score_xgb",
    "score_tab",
    "score_rf",
    "rank_pct_xgb",
    "rank_pct_tab",
    "rank_pct_rf",
    "in_xgb",
    "in_tab",
    "in_rf",
    "vote_count",
    "consensus_tier",
    "exit_policy_id",
    "horizon",
    "top_pct",
    "pair_cap",
    "tp_ratio",
    "sl_ratio",
    "time_stop_minutes",
    "round_trip_fee_pct",
    "selected_by_policy",
    "sim_exit_status",
    "sim_net_return",
    "target_net_profitable_after_exit",
    "source_artifact_id_lineage",
    "exit_policy_artifact_id",
    "registry_version",
    "content_hash",
    "schema_hash",
    "solana_enrichment_status",
    "helius_validation_status",
    "wallet_identity_status",
    "wallet_behavior_status",
    "rss_sentiment_status",
    "reputation_status",
    "whale_intelligence_status",
    "sentiment_score",
    "reputation_risk_score",
    "whale_behavior_score",
    "qwen_review_status",
    "gemini_review_status",
    "llm_veto",
    "llm_review_artifact_id",
    "decision",
    "decision_trace_id",
    "paper_trade_id",
    "paper_position_id",
    "schema_version",
    "warnings",
)


def _is_missing_scalar(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def normalize_flat_value_for_export(value: Any, *, target_format: str = "parquet") -> Any:
    """Normalize values for deterministic flat export."""
    fmt = target_format.lower()
    if _is_missing_scalar(value):
        if fmt == "csv":
            return ""
        return None
    if isinstance(value, bool):
        if fmt == "csv":
            return "true" if value else "false"
        return value
    if isinstance(value, (ConsensusTier, EnrichmentStatus, LLMReviewStatus, DecisionStatus)):
        return value.value
    if isinstance(value, list):
        if fmt == "csv":
            return "|".join(str(item) for item in value)
        return list(value)
    return value


def normalize_flat_value_for_import(
    value: Any,
    *,
    source_format: str = "parquet",
    allow_nan_for_research: bool = False,
) -> Any:
    """Normalize flat import values back to internal Python representation."""
    fmt = source_format.lower()
    if _is_missing_scalar(value):
        return None
    if isinstance(value, float):
        if math.isnan(value):
            if allow_nan_for_research:
                return value
            return None
        return value
    if fmt == "csv" and isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        lowered = text.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if lowered == "none":
            return None
        if "|" in text and text.count("|") >= 1:
            return [part for part in text.split("|") if part != ""]
        return text
    return value


def candidate_to_dict(candidate: UnifiedCandidate) -> dict[str, Any]:
    return candidate.model_dump(mode="json")


def candidate_from_dict(
    payload: dict[str, Any],
    *,
    strict_consensus: bool = True,
) -> UnifiedCandidate:
    return UnifiedCandidate.model_validate(
        payload,
        context={"strict_consensus": strict_consensus},
    )


def candidate_to_json(candidate: UnifiedCandidate, *, indent: int | None = None) -> str:
    return json.dumps(candidate_to_dict(candidate), indent=indent, sort_keys=True)


def candidate_from_json(
    payload: str,
    *,
    strict_consensus: bool = True,
) -> UnifiedCandidate:
    return candidate_from_dict(json.loads(payload), strict_consensus=strict_consensus)


def candidate_to_flat_dict(
    candidate: UnifiedCandidate,
    *,
    target_format: str = "parquet",
) -> dict[str, Any]:
    identity = candidate.identity
    market = candidate.market
    scores = candidate.model_scores
    exit_policy = candidate.exit_policy
    lineage = candidate.lineage
    enrichment = candidate.enrichment
    llm_review = candidate.llm_review
    decision = candidate.decision

    raw: dict[str, Any] = {
        "candidate_id": identity.candidate_id,
        "pair_address": identity.pair_address,
        "chain": identity.chain,
        "event_timestamp": identity.event_timestamp,
        "event_timestamp_normalized": identity.event_timestamp_normalized,
        "timestamp_precision": identity.timestamp_precision,
        "source": identity.source,
        "source_artifact_id": identity.source_artifact_id,
        "source_row_id": identity.source_row_id,
        "coin_id": identity.coin_id,
        "symbol": identity.symbol,
        "price": market.price,
        "liquidity_usd": market.liquidity_usd,
        "volume_24h": market.volume_24h,
        "fdv": market.fdv,
        "score_xgb": scores.score_xgb,
        "score_tab": scores.score_tab,
        "score_rf": scores.score_rf,
        "rank_pct_xgb": scores.rank_pct_xgb,
        "rank_pct_tab": scores.rank_pct_tab,
        "rank_pct_rf": scores.rank_pct_rf,
        "in_xgb": scores.in_xgb,
        "in_tab": scores.in_tab,
        "in_rf": scores.in_rf,
        "vote_count": scores.vote_count,
        "consensus_tier": candidate.consensus_tier,
        "exit_policy_id": exit_policy.exit_policy_id,
        "horizon": exit_policy.horizon,
        "top_pct": exit_policy.top_pct,
        "pair_cap": exit_policy.pair_cap,
        "tp_ratio": exit_policy.tp_ratio,
        "sl_ratio": exit_policy.sl_ratio,
        "time_stop_minutes": exit_policy.time_stop_minutes,
        "round_trip_fee_pct": exit_policy.round_trip_fee_pct,
        "selected_by_policy": exit_policy.selected_by_policy,
        "sim_exit_status": exit_policy.sim_exit_status,
        "sim_net_return": exit_policy.sim_net_return,
        "target_net_profitable_after_exit": exit_policy.target_net_profitable_after_exit,
        "source_artifact_id_lineage": lineage.source_artifact_id,
        "exit_policy_artifact_id": lineage.exit_policy_artifact_id,
        "registry_version": lineage.registry_version,
        "content_hash": lineage.content_hash,
        "schema_hash": lineage.schema_hash,
        "solana_enrichment_status": enrichment.solana_enrichment_status,
        "helius_validation_status": enrichment.helius_validation_status,
        "wallet_identity_status": enrichment.wallet_identity_status,
        "wallet_behavior_status": enrichment.wallet_behavior_status,
        "rss_sentiment_status": enrichment.rss_sentiment_status,
        "reputation_status": enrichment.reputation_status,
        "whale_intelligence_status": enrichment.whale_intelligence_status,
        "sentiment_score": enrichment.sentiment_score,
        "reputation_risk_score": enrichment.reputation_risk_score,
        "whale_behavior_score": enrichment.whale_behavior_score,
        "qwen_review_status": llm_review.qwen_review_status,
        "gemini_review_status": llm_review.gemini_review_status,
        "llm_veto": llm_review.llm_veto,
        "llm_review_artifact_id": llm_review.llm_review_artifact_id,
        "decision": decision.decision,
        "decision_trace_id": decision.decision_trace_id,
        "paper_trade_id": decision.paper_trade_id,
        "paper_position_id": decision.paper_position_id,
        "schema_version": candidate.schema_version,
        "warnings": candidate.warnings,
    }

    return {
        column: normalize_flat_value_for_export(raw[column], target_format=target_format)
        for column in FLAT_COLUMN_ORDER
    }


def candidate_from_flat_dict(
    row: dict[str, Any],
    *,
    source_format: str = "parquet",
    strict_consensus: bool = True,
    allow_nan_for_research: bool = False,
) -> UnifiedCandidate:
    def pick(key: str, default: Any = None) -> Any:
        if key not in row:
            return default
        return normalize_flat_value_for_import(
            row[key],
            source_format=source_format,
            allow_nan_for_research=allow_nan_for_research,
        )

    identity_kwargs: dict[str, Any] = {
        "pair_address": pick("pair_address"),
        "chain": pick("chain"),
        "event_timestamp": pick("event_timestamp"),
        "timestamp_precision": pick("timestamp_precision", "seconds"),
        "source": pick("source"),
        "source_artifact_id": pick("source_artifact_id"),
        "source_row_id": pick("source_row_id"),
        "coin_id": pick("coin_id"),
        "symbol": pick("symbol"),
    }
    normalized_ts = pick("event_timestamp_normalized")
    if normalized_ts is not None:
        identity_kwargs["event_timestamp_normalized"] = normalized_ts
    candidate_id = pick("candidate_id")
    if candidate_id is not None:
        identity_kwargs["candidate_id"] = candidate_id
    identity = CandidateIdentity(**identity_kwargs)

    market = MarketContext(
        price=pick("price"),
        liquidity_usd=pick("liquidity_usd"),
        volume_24h=pick("volume_24h"),
        fdv=pick("fdv"),
    )

    model_scores_kwargs: dict[str, Any] = {
        "score_xgb": pick("score_xgb"),
        "score_tab": pick("score_tab"),
        "score_rf": pick("score_rf"),
        "rank_pct_xgb": pick("rank_pct_xgb"),
        "rank_pct_tab": pick("rank_pct_tab"),
        "rank_pct_rf": pick("rank_pct_rf"),
        "in_xgb": pick("in_xgb"),
        "in_tab": pick("in_tab"),
        "in_rf": pick("in_rf"),
    }
    vote_count = pick("vote_count")
    if vote_count is not None:
        model_scores_kwargs["vote_count"] = vote_count
    model_scores = ModelScores(**model_scores_kwargs)

    consensus_raw = pick("consensus_tier")
    consensus_tier = ConsensusTier(consensus_raw) if consensus_raw is not None else None

    exit_policy = ExitPolicyContext(
        exit_policy_id=pick("exit_policy_id"),
        horizon=pick("horizon"),
        top_pct=pick("top_pct"),
        pair_cap=pick("pair_cap"),
        tp_ratio=pick("tp_ratio"),
        sl_ratio=pick("sl_ratio"),
        time_stop_minutes=pick("time_stop_minutes"),
        round_trip_fee_pct=pick("round_trip_fee_pct"),
        selected_by_policy=pick("selected_by_policy"),
        sim_exit_status=pick("sim_exit_status"),
        sim_net_return=pick("sim_net_return"),
        target_net_profitable_after_exit=pick("target_net_profitable_after_exit"),
    )

    lineage = ArtifactLineage(
        source_artifact_id=pick("source_artifact_id_lineage", pick("source_artifact_id")),
        exit_policy_artifact_id=pick("exit_policy_artifact_id"),
        registry_version=pick("registry_version"),
        content_hash=pick("content_hash"),
        schema_hash=pick("schema_hash"),
    )

    enrichment = EnrichmentState(
        solana_enrichment_status=EnrichmentStatus(
            pick("solana_enrichment_status", EnrichmentStatus.NOT_REQUESTED.value)
        ),
        helius_validation_status=EnrichmentStatus(
            pick("helius_validation_status", EnrichmentStatus.NOT_REQUESTED.value)
        ),
        wallet_identity_status=EnrichmentStatus(
            pick("wallet_identity_status", EnrichmentStatus.NOT_REQUESTED.value)
        ),
        wallet_behavior_status=EnrichmentStatus(
            pick("wallet_behavior_status", EnrichmentStatus.NOT_REQUESTED.value)
        ),
        rss_sentiment_status=EnrichmentStatus(
            pick("rss_sentiment_status", EnrichmentStatus.NOT_REQUESTED.value)
        ),
        reputation_status=EnrichmentStatus(
            pick("reputation_status", EnrichmentStatus.NOT_REQUESTED.value)
        ),
        whale_intelligence_status=EnrichmentStatus(
            pick("whale_intelligence_status", EnrichmentStatus.NOT_REQUESTED.value)
        ),
        sentiment_score=pick("sentiment_score"),
        reputation_risk_score=pick("reputation_risk_score"),
        whale_behavior_score=pick("whale_behavior_score"),
    )

    llm_review = LLMReviewState(
        qwen_review_status=LLMReviewStatus(
            pick("qwen_review_status", LLMReviewStatus.NOT_REQUESTED.value)
        ),
        gemini_review_status=LLMReviewStatus(
            pick("gemini_review_status", LLMReviewStatus.NOT_REQUESTED.value)
        ),
        llm_veto=pick("llm_veto"),
        llm_review_artifact_id=pick("llm_review_artifact_id"),
    )

    decision = CandidateDecisionState(
        decision=DecisionStatus(pick("decision", DecisionStatus.UNDECIDED.value)),
        decision_trace_id=pick("decision_trace_id"),
        paper_trade_id=pick("paper_trade_id"),
        paper_position_id=pick("paper_position_id"),
    )

    warnings = pick("warnings", [])
    if warnings is None:
        warnings = []

    payload = {
        "identity": identity.model_dump(mode="python"),
        "market": market.model_dump(mode="python"),
        "model_scores": model_scores.model_dump(mode="python"),
        "exit_policy": exit_policy.model_dump(mode="python"),
        "lineage": lineage.model_dump(mode="python"),
        "enrichment": enrichment.model_dump(mode="python"),
        "llm_review": llm_review.model_dump(mode="python"),
        "decision": decision.model_dump(mode="python"),
        "warnings": warnings,
        "schema_version": pick("schema_version", SCHEMA_VERSION),
    }
    if consensus_tier is not None:
        payload["consensus_tier"] = consensus_tier.value
    return candidate_from_dict(payload, strict_consensus=strict_consensus)
