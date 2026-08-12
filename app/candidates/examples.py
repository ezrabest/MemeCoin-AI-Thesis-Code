"""Example unified candidates for tests and documentation (Phase E2)."""

from __future__ import annotations

from app.candidates.schema import (
    ArtifactLineage,
    CandidateDecisionState,
    CandidateIdentity,
    ConsensusTier,
    DecisionStatus,
    EnrichmentState,
    ExitPolicyContext,
    LLMReviewState,
    MarketContext,
    ModelScores,
    UnifiedCandidate,
)


def _base_identity(**overrides: object) -> CandidateIdentity:
    payload = {
        "pair_address": "So11111111111111111111111111111111111111112",
        "chain": "solana",
        "event_timestamp": "2024-06-15T18:30:00Z",
        "source": "dexscreener_snapshot",
        "source_row_id": "row-1",
    }
    payload.update(overrides)
    return CandidateIdentity(**payload)


def make_minimal_candidate_example() -> UnifiedCandidate:
    return UnifiedCandidate(
        identity=_base_identity(),
        market=MarketContext(price=0.0012, liquidity_usd=12500.0),
        model_scores=ModelScores(in_xgb=False, in_tab=False, in_rf=False),
    )


def make_all3_candidate_example() -> UnifiedCandidate:
    return UnifiedCandidate(
        identity=_base_identity(source_row_id="row-all3"),
        market=MarketContext(price=0.0045, liquidity_usd=88000.0, volume_24h=210000.0),
        model_scores=ModelScores(
            score_xgb=0.91,
            score_tab=0.88,
            score_rf=0.79,
            rank_pct_xgb=0.01,
            rank_pct_tab=0.02,
            rank_pct_rf=0.03,
            in_xgb=True,
            in_tab=True,
            in_rf=True,
        ),
        consensus_tier=ConsensusTier.TAB_XGB_RF_ALL3,
        exit_policy=ExitPolicyContext(
            horizon="4h",
            top_pct=0.02,
            pair_cap=50,
            tp_ratio=2.0308,
            sl_ratio=0.80,
            round_trip_fee_pct=0.0308,
            selected_by_policy=True,
        ),
        lineage=ArtifactLineage(
            source_artifact_id="a" * 64,
            model_prediction_artifact_ids={
                "xgb": "b" * 64,
                "tab": "c" * 64,
                "rf": "d" * 64,
            },
            lineage_warnings=["model_prediction_artifact_ids not fully verified"],
        ),
        decision=CandidateDecisionState(decision=DecisionStatus.WATCH),
    )


def make_tab_rf_candidate_example() -> UnifiedCandidate:
    return UnifiedCandidate(
        identity=_base_identity(source_row_id="row-tab-rf"),
        market=MarketContext(price=0.0021, liquidity_usd=42000.0),
        model_scores=ModelScores(
            score_tab=0.86,
            score_rf=0.74,
            in_tab=True,
            in_xgb=False,
            in_rf=True,
        ),
        consensus_tier=ConsensusTier.TAB_RF_ONLY,
        exit_policy=ExitPolicyContext(
            horizon="4h",
            tp_ratio=2.0308,
            sl_ratio=0.80,
            round_trip_fee_pct=0.0308,
        ),
        enrichment=EnrichmentState(),
        llm_review=LLMReviewState(),
    )


def make_research_rejected_candidate_example() -> UnifiedCandidate:
    return UnifiedCandidate(
        identity=_base_identity(source_row_id="row-research"),
        market=MarketContext(price=0.0008, liquidity_usd=9000.0),
        model_scores=ModelScores(
            score_xgb=0.82,
            score_tab=0.77,
            in_tab=True,
            in_xgb=True,
            in_rf=False,
        ),
        consensus_tier=ConsensusTier.TAB_XGB_ONLY,
        decision=CandidateDecisionState(
            decision=DecisionStatus.REJECTED_RESEARCH_ONLY,
            decision_reason_codes=["consensus_tier_research_only"],
        ),
        warnings=["TAB_XGB_ONLY is research-only under Anchor Plan"],
    )
