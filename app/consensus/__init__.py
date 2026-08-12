"""AE16 Tiered Consensus Engine on Direct Target / Clean Forward Bridge.

Research/shadow/paper-demo only. Does not grant live trading authority.
"""

from __future__ import annotations

CONSENSUS_ENGINE_VERSION = "ae16_tiered_consensus_v1"
PHASE = "AE16"

MODEL_FAMILIES = ("RF", "XGB", "TAB")

ATTACHMENT_STATUS_ATTACHED = "MODEL_EVIDENCE_ATTACHED"
ATTACHMENT_STATUS_UNAVAILABLE = "MODEL_EVIDENCE_UNAVAILABLE"

ALLOWED_ATTACHMENT_STATUSES = frozenset(
    {
        "MODEL_EVIDENCE_ATTACHED",
        "MODEL_EVIDENCE_UNAVAILABLE",
        "SCORE_NOT_ATTACHED",
        "ARTIFACT_NOT_FOUND",
        "ARTIFACT_READ_ERROR",
        "ARTIFACT_SCHEMA_UNSUPPORTED",
        "FEATURE_PARITY_NOT_APPROVED",
        "CANDIDATE_ID_NOT_MATCHED",
        "PAIR_TIMESTAMP_NOT_MATCHED",
        "PAIR_TIMESTAMP_JOIN_REJECTED",
        "POLICY_ID_NOT_AVAILABLE",
        "TARGET_ROW_ID_NOT_AVAILABLE",
        "PREDICTION_FILE_NOT_FOUND",
        "MODEL_ARTIFACT_NOT_FOUND",
        "SCHEMA_ARTIFACT_NOT_FOUND",
        "EXACT_ID_JOIN_NOT_AVAILABLE",
        "EXACT_ID_JOIN_AMBIGUOUS",
        "LEGACY_SOURCE_REJECTED",
        "RETRAINING_REQUIRED",
        "MODEL_RUNTIME_AUTHORITY_NOT_APPROVED",
        "ATTACHMENT_EXCEPTION_CAUGHT",
    }
)

CONSENSUS_ENGINE_VERSION_V2 = "ae16_tiered_consensus_v2"

CANONICAL_DIRECT_TARGET_ROOTS = (
    "data/training/manual_verified_results/phase_e4_direct_target_xgb_rf_full_20260630_195312",
    "data/training/manual_verified_results/phase_e5_direct_target_tabicl_20260703_203824",
    "data/training/manual_verified_results/phase_e5_direct_target_tabicl_20260703_223609",
    "data/audits/ae16_tiered_consensus_engine_20260722_202855",
)

# Clean Forward fields that can map to direct-target model features (name or alias).
CF_TO_MODEL_FEATURE_MAP = {
    "price_usd": ("price_usd", "price"),
    "liquidity_usd": ("liquidity_usd", "liquidity"),
    "volume_24h": ("volume_24h",),
    "txns_buys_24h": ("txns_buys",),
    "txns_sells_24h": ("txns_sells",),
    "price_change_m5": ("price_change_m5",),
    "price_change_h1": ("price_change_h1",),
    "price_change_h6": ("price_change_h6",),
    "price_change_h24": ("price_change_h24",),
}

CONSENSUS_TIERS = frozenset(
    {
        "TAB_XGB_RF_ALL3",
        "TAB_RF_ONLY",
        "TAB_XGB_ONLY",
        "XGB_RF_ONLY",
        "SINGLE_MODEL_ONLY",
        "MODEL_DISAGREEMENT",
        "MODEL_EVIDENCE_UNAVAILABLE",
        "CONSENSUS_NOT_COMPUTABLE",
        "RESEARCH_ONLY_WATCH",
        "REJECT_OR_SKIP",
    }
)

RESEARCH_ONLY_TIERS = frozenset({"TAB_XGB_ONLY", "XGB_RF_ONLY", "SINGLE_MODEL_ONLY", "RESEARCH_ONLY_WATCH"})

DEFAULT_CLEANED_INPUT_ROOT = (
    "data/audits/ae15_cleaned_for_ae16_20260722_194200/data"
)

REQUIRED_INPUT_FILES = (
    "ae16_clean_forward_candidates.csv",
    "ae16_clean_forward_decision_inputs.csv",
    "ae16_clean_forward_outcome_label_contract.csv",
    "ae16_clean_forward_paper_execution_links.csv",
)

EXPECTED_INPUT_COUNTS = {
    "candidates": 961,
    "decision_inputs": 961,
    "outcome_contracts": 961,
    "execution_links": 1,
}

__all__ = [
    "CONSENSUS_ENGINE_VERSION",
    "CONSENSUS_ENGINE_VERSION_V2",
    "PHASE",
    "MODEL_FAMILIES",
    "ALLOWED_ATTACHMENT_STATUSES",
    "CONSENSUS_TIERS",
    "RESEARCH_ONLY_TIERS",
    "DEFAULT_CLEANED_INPUT_ROOT",
    "REQUIRED_INPUT_FILES",
    "EXPECTED_INPUT_COUNTS",
    "CANONICAL_DIRECT_TARGET_ROOTS",
    "CF_TO_MODEL_FEATURE_MAP",
]
