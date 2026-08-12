"""AE17 constants, schemas, and forbidden-field contracts."""

from __future__ import annotations

from app.meta import AUTHORITY_STATUS, META_MODE

# Known AE16 audit roots (preferred discovery order after --ae16-root).
KNOWN_AE16_ROOTS: tuple[str, ...] = (
    "data/audits/ae16_tab16_direct_target_serving_safe_20260724T205012Z",
    "data/audits/manual_tab16_direct_consensus_tier_check_20260724T212036Z",
    "data/audits/manual_ae16_original_closure_audit_20260724T212124Z",
    "data/audits/manual_ae16_tier_semantic_coverage_corrected_20260724T230109Z",
)

# Filename / relative path patterns used during discovery.
AE16_CONSENSUS_PATTERNS: tuple[str, ...] = (
    "**/rf_xgb_tab16_consensus_preview.csv",
    "**/ae16_clean_forward_consensus_decisions.csv",
    "**/ae16_tiered_consensus_rows.csv",
    "**/ae16f_tiered_consensus_rows.csv",
)

AE16_EVIDENCE_PATTERNS: tuple[str, ...] = (
    "**/rf_xgb_tab16_current_model_evidence.csv",
    "**/ae16_model_evidence_attachment.csv",
    "**/ae16f_model_evidence.csv",
)

REQUIRED_AE16_ARTIFACT_KINDS: tuple[str, ...] = (
    "consensus_rows",
)

META_FEATURE_FIELDS: tuple[str, ...] = (
    "clean_forward_candidate_id",
    "clean_forward_decision_input_id",
    "price_source_key",
    "provider",
    "chain",
    "pair_address",
    "base_token_address",
    "quote_token_address",
    "provider_pair_url",
    "provider_payload_hash",
    "rf_evidence_status",
    "xgb_evidence_status",
    "tab_evidence_status",
    "rf_score",
    "xgb_score",
    "tab_score",
    "rf_vote",
    "xgb_vote",
    "tab_vote",
    "attached_model_count",
    "model_vote_count",
    "consensus_tier",
    "consensus_reason",
    "context_status",
    "context_feature_available",
    "context_missingness_reason",
    "context_score_weight",
    "observed_at",
    "fetched_at",
    "ingested_at",
    "source_ae16_artifact",
    "source_schema_hash",
    "lineage_status",
)

# Numeric / score columns in the feature matrix (null-safe; never zero-fill missing).
META_SCORE_FIELDS: tuple[str, ...] = (
    "rf_score",
    "xgb_score",
    "tab_score",
    "context_score_weight",
)

SHADOW_OUTPUT_FIELDS: tuple[str, ...] = (
    "clean_forward_candidate_id",
    "clean_forward_decision_input_id",
    "price_source_key",
    "pair_address",
    "consensus_tier",
    "pre_clamp_meta_score",
    "meta_score",
    "score_clamped",
    "score_clamp_reason",
    "meta_decision",
    "meta_reason",
    "meta_mode",
    "context_score_weight",
    "pair_concentration_status",
    "authority_status",
    "trade_authority",
    "live_trading_ready",
    "paper_demo_only",
    "risk_override_authority",
)

FORBIDDEN_FEATURE_FIELDS: frozenset[str] = frozenset(
    {
        "future_return",
        "max_upside",
        "max_drawdown",
        "hit_tp",
        "hit_sl",
        "time_stop",
        "realized_pnl",
        "net_return",
        "profitability",
        "profitability_label",
        "target",
        "target_label",
        "paper_result",
        "position_result",
        "outcome",
        "outcome_label",
        "outcome_status",
        "matured_at",
        "closed_at",
        "exit_status",
        "pnl",
        "profit",
        "label_available",
        "paper_order_id",
        "paper_position_id",
    }
)

FORBIDDEN_FEATURE_SUBSTRINGS: tuple[str, ...] = (
    "future_return",
    "max_upside",
    "max_drawdown",
    "hit_tp",
    "hit_sl",
    "realized_pnl",
    "net_return",
    "profitability",
    "target_label",
    "paper_result",
    "position_result",
    "outcome_label",
    "matured_at",
    "closed_at",
    "exit_status",
)

KNOWN_CONSENSUS_TIERS: frozenset[str] = frozenset(
    {
        "TAB_XGB_RF_ALL3",
        "TAB_RF_ONLY",
        "TAB_XGB_ONLY",
        "RF_XGB_ONLY",
        "XGB_RF_ONLY",  # AE16 tiered_engine alias
        "SINGLE_MODEL_ONLY",
        "REJECT",
        "MODEL_EVIDENCE_UNAVAILABLE",
        "CONSENSUS_NOT_COMPUTABLE",
        "PARTIAL_MODEL_EVIDENCE",
    }
)

CONTEXT_STATUS_PENDING = "AE17_CONTEXT_NOT_AVAILABLE_PENDING_AE18"
CONTEXT_MISSINGNESS_REASON = "AE18_NOT_IMPLEMENTED_OR_NO_CONTEXT_ATTACHED"
LINEAGE_COMPLETE = "AE17_LINEAGE_COMPLETE"
LINEAGE_INCOMPLETE = "AE17_LINEAGE_INCOMPLETE"
LINEAGE_REQUIRED_FIELDS: tuple[str, ...] = (
    "clean_forward_candidate_id",
    "clean_forward_decision_input_id",
    "price_source_key",
    "provider_payload_hash",
    "provider_pair_url",
    "pair_address",
    "base_token_address",
    "quote_token_address",
    "source_ae16_artifact",
    "source_schema_hash",
    "lineage_status",
)

# Pair concentration thresholds.
TOP_PAIR_SHARE_OK = 0.30
TOP_PAIR_SHARE_WARNING = 0.50
HHI_LOW = 0.15
HHI_MODERATE = 0.25
SMALL_SAMPLE_N = 20
LOW_PAIR_DIVERSITY_N = 5

# Bounded pair-concentration penalty applied to meta score (shadow only).
PAIR_CONCENTRATION_PENALTY_WARNING = 0.03
PAIR_CONCENTRATION_PENALTY_HIGH = 0.08

DEFAULT_SHADOW_CONSTANTS: dict[str, object] = {
    "meta_mode": META_MODE,
    "trade_authority": False,
    "live_trading_ready": False,
    "paper_demo_only": True,
    "risk_override_authority": False,
    "authority_status": AUTHORITY_STATUS,
}

TIER_BASE_SCORES: dict[str, tuple[float | None, str, str]] = {
    # tier -> (base_meta_score, meta_decision, reason_fragment)
    "TAB_XGB_RF_ALL3": (
        0.90,
        "META_STRONG_WATCH",
        "all three model slots agree",
    ),
    "TAB_RF_ONLY": (
        0.75,
        "META_SECONDARY_WATCH",
        "TAB and RF agreement",
    ),
    "TAB_XGB_ONLY": (
        0.45,
        "META_RESEARCH_ONLY",
        "historically weaker/research-only tier",
    ),
    "RF_XGB_ONLY": (
        0.40,
        "META_RESEARCH_ONLY",
        "historically weaker/research-only tier",
    ),
    "XGB_RF_ONLY": (
        0.40,
        "META_RESEARCH_ONLY",
        "historically weaker/research-only tier",
    ),
    "SINGLE_MODEL_ONLY": (
        0.25,
        "META_LOW_CONFIDENCE",
        "single model vote only",
    ),
    "REJECT": (
        0.0,
        "META_REJECT",
        "consensus reject (attached evidence, no positive votes)",
    ),
    "MODEL_EVIDENCE_UNAVAILABLE": (
        None,
        "META_UNAVAILABLE",
        "model evidence unavailable",
    ),
    "CONSENSUS_NOT_COMPUTABLE": (
        None,
        "META_UNAVAILABLE",
        "consensus not computable",
    ),
}
