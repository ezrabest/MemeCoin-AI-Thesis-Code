"""AE15 Clean Forward schema bridge — data contracts, identity, and lineage.

Does not train models, backtest, enable live trading, or call external APIs.
"""
from __future__ import annotations

from app.clean_forward.identity import (
    build_instrument_identity,
    normalize_address_for_chain,
    pair_address_for_id,
)
from app.clean_forward.lineage import (
    reconcile_ae14_order_position_lineage,
    summarize_order_position_lineage,
)
from app.clean_forward.schema import (
    DECISION_INPUT_VERSION,
    CleanForwardCandidate,
    CleanForwardDecisionInput,
    CleanForwardInstrumentIdentity,
    CleanForwardOutcomeLabel,
    CleanForwardPaperExecutionLink,
    CleanForwardSkipReason,
    make_clean_forward_candidate_id,
    make_clean_forward_decision_input_id,
    make_execution_link_id,
    make_outcome_label_id,
    make_skip_record_id,
)
from app.clean_forward.serialization import (
    record_to_dict,
    stable_json_dumps,
    stable_payload_hash,
)
from app.clean_forward.validation import (
    CLEAN_FEED_ELIGIBILITY_RULES,
    evaluate_clean_feed_eligibility,
    validate_identity_separation,
)
from app.clean_forward.curated_overlay import (
    FLAG_CURATED_PATH,
    FLAG_USE_CURATED,
    curated_targets_enabled,
    curated_targets_path,
)

__all__ = [
    "CLEAN_FEED_ELIGIBILITY_RULES",
    "DECISION_INPUT_VERSION",
    "FLAG_CURATED_PATH",
    "FLAG_USE_CURATED",
    "CleanForwardCandidate",
    "CleanForwardDecisionInput",
    "CleanForwardInstrumentIdentity",
    "CleanForwardOutcomeLabel",
    "CleanForwardPaperExecutionLink",
    "CleanForwardSkipReason",
    "build_instrument_identity",
    "curated_targets_enabled",
    "curated_targets_path",
    "evaluate_clean_feed_eligibility",
    "make_clean_forward_candidate_id",
    "make_clean_forward_decision_input_id",
    "make_execution_link_id",
    "make_outcome_label_id",
    "make_skip_record_id",
    "normalize_address_for_chain",
    "pair_address_for_id",
    "reconcile_ae14_order_position_lineage",
    "record_to_dict",
    "stable_json_dumps",
    "stable_payload_hash",
    "summarize_order_position_lineage",
    "validate_identity_separation",
]
