"""AE15 canonical Clean Forward schema records and deterministic ID builders."""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field, fields
from typing import Any

DECISION_INPUT_VERSION = "AE15_CLEAN_FORWARD_DECISION_INPUT_V1"
CANDIDATE_ID_NAMESPACE = "AE15_CLEAN_FORWARD_CANDIDATE_V1"
EXECUTION_LINK_NAMESPACE = "AE15_CLEAN_FORWARD_EXECUTION_LINK_V1"
OUTCOME_LABEL_NAMESPACE = "AE15_CLEAN_FORWARD_OUTCOME_LABEL_V1"
SKIP_RECORD_NAMESPACE = "AE15_CLEAN_FORWARD_SKIP_V1"

# Fields that must NEVER participate in clean_forward_candidate_id.
CANDIDATE_ID_FORBIDDEN_FIELDS = frozenset(
    {
        "xgb_score",
        "tab_score",
        "rf_score",
        "model_score",
        "consensus_tier",
        "consensus_tier_shadow",
        "paper_order_id",
        "paper_position_id",
        "future_return",
        "max_upside",
        "max_drawdown",
        "hit_tp",
        "hit_sl",
        "target_label",
        "outcome_status",
        "label_available",
    }
)


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _pipe_join(*parts: Any) -> str:
    return "|".join("" if p is None else str(p) for p in parts)


def make_clean_forward_candidate_id(
    *,
    chain: str,
    provider: str,
    pair_address_for_id: str,
    base_token_address: str,
    quote_token_address: str,
    observed_at_or_fetched_at: str,
    provider_payload_hash: str,
) -> str:
    """Deterministic event-level candidate id (no model/order/future fields)."""
    material = _pipe_join(
        CANDIDATE_ID_NAMESPACE,
        chain,
        provider,
        pair_address_for_id,
        base_token_address,
        quote_token_address,
        observed_at_or_fetched_at,
        provider_payload_hash,
    )
    return _sha256_hex(material)


def make_clean_forward_decision_input_id(
    *,
    clean_forward_candidate_id: str,
    candidate_snapshot_timestamp: str,
    active_preset_id: str,
    risk_mode: str,
    strict_mode: bool,
    exploration_mode: bool,
    decision_input_version: str = DECISION_INPUT_VERSION,
) -> str:
    material = _pipe_join(
        clean_forward_candidate_id,
        candidate_snapshot_timestamp,
        active_preset_id,
        risk_mode,
        "strict" if strict_mode else "nonstrict",
        "exploration" if exploration_mode else "no_exploration",
        decision_input_version,
    )
    return _sha256_hex(material)


def make_execution_link_id(
    *,
    clean_forward_candidate_id: str,
    paper_order_id: str | None,
    paper_position_id: str | None,
    position_created_at: str | None,
) -> str:
    material = _pipe_join(
        EXECUTION_LINK_NAMESPACE,
        clean_forward_candidate_id,
        paper_order_id,
        paper_position_id,
        position_created_at,
    )
    return _sha256_hex(material)


def make_outcome_label_id(
    *,
    clean_forward_candidate_id: str,
    clean_forward_decision_input_id: str | None,
    horizon: str | None,
) -> str:
    material = _pipe_join(
        OUTCOME_LABEL_NAMESPACE,
        clean_forward_candidate_id,
        clean_forward_decision_input_id,
        horizon,
    )
    return _sha256_hex(material)


def make_skip_record_id(
    *,
    clean_forward_candidate_id: str,
    skipped_at: str,
    skip_stage: str,
    skip_reason_code: str,
) -> str:
    material = _pipe_join(
        SKIP_RECORD_NAMESPACE,
        clean_forward_candidate_id,
        skipped_at,
        skip_stage,
        skip_reason_code,
    )
    return _sha256_hex(material)


@dataclass
class CleanForwardInstrumentIdentity:
    chain: str
    provider: str
    provider_pair_url: str | None
    pair_address: str
    pair_address_normalized: str
    base_token_address: str
    base_token_symbol: str | None
    base_token_name: str | None
    quote_token_address: str
    quote_token_symbol: str | None
    quote_token_name: str | None
    dex_id: str | None
    pair_created_at: Any | None
    shown_as_token_contract: bool
    identity_status: str
    verification_status: str
    freshness_status: str
    paper_demo_only: bool
    live_trading_ready: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CleanForwardCandidate:
    clean_forward_candidate_id: str
    source_clean_forward_row_key: str
    source_poll_file: str | None
    source_poll_index: int | None
    provider_payload_hash: str
    provider_pair_url: str | None
    chain: str
    pair_address: str
    base_token_address: str
    quote_token_address: str
    symbol_pair: str | None
    price_usd: float | None
    liquidity_usd: float | None
    volume_24h: float | None
    txns_buys_24h: int | None
    txns_sells_24h: int | None
    price_change_m5: float | None
    price_change_h1: float | None
    price_change_h6: float | None
    price_change_h24: float | None
    observed_at: str | None
    fetched_at: str | None
    ingested_at: str | None
    rendered_at: str | None
    latest_provider_fetch_at: str | None
    verification_status: str
    freshness_status: str
    identity_status: str
    clean_feed_eligible: bool
    paper_demo_only: bool
    live_trading_ready: bool
    provider: str = "dexscreener"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


@dataclass
class CleanForwardDecisionInput:
    clean_forward_decision_input_id: str
    clean_forward_candidate_id: str
    candidate_snapshot_timestamp: str
    gatekeeper_input_payload_hash: str | None
    riskguard_input_payload_hash: str | None
    active_preset_id: str
    risk_mode: str
    strict_mode: bool
    exploration_mode: bool
    max_price_age_seconds: int | None
    strict_shadow_max_price_age_seconds: int | None
    model_scores_available: bool
    xgb_score: float | None
    tab_score: float | None
    rf_score: float | None
    model_score_source_status: str
    consensus_tier_shadow: str | None
    context_status: str
    llm_status: str
    decision_input_version: str = DECISION_INPUT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CleanForwardPaperExecutionLink:
    execution_link_id: str
    clean_forward_candidate_id: str | None
    clean_forward_decision_input_id: str | None
    source_clean_forward_row_key: str | None
    paper_order_id: str | None
    paper_position_id: str | None
    order_created_at: str | None
    position_created_at: str | None
    order_side: str | None
    order_status: str | None
    fill_status: str | None
    position_status: str | None
    order_notional_usd: float | None
    fill_price_usd: float | None
    position_quantity: float | None
    source_provider_pair_url: str | None
    pair_address: str | None
    base_token_address: str | None
    quote_token_address: str | None
    provider_payload_hash: str | None
    gatekeeper_decision: str | None
    riskguard_decision: str | None
    entry_reason: str | None
    skip_reason: str | None
    position_created_by: str | None
    position_creation_reason: str | None
    preexisting_position_detected: bool
    reconstructed_position_detected: bool
    duplicate_position_detected: bool
    one_order_to_one_position_expected: bool
    one_order_to_one_position_passed: bool | None
    counter_consistency_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CleanForwardOutcomeLabel:
    clean_forward_outcome_label_id: str
    clean_forward_candidate_id: str
    clean_forward_decision_input_id: str | None
    paper_order_id: str | None
    paper_position_id: str | None
    outcome_source: str
    outcome_status: str
    horizon: str | None
    future_return: float | None
    max_upside: float | None
    max_drawdown: float | None
    hit_tp: bool | None
    hit_sl: bool | None
    time_stop: bool | None
    stale_price_exit: bool | None
    matured_at: str | None
    no_lookahead_passed: bool | None
    label_available: bool
    label_unavailable_reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CleanForwardSkipReason:
    skip_record_id: str
    clean_forward_candidate_id: str | None
    clean_forward_decision_input_id: str | None
    skipped_at: str
    skip_stage: str
    skip_reason_code: str
    skip_reason_detail: str | None
    stale_price: bool = False
    missing_price: bool = False
    missing_identity: bool = False
    missing_provider_verification: bool = False
    blocked_by_gatekeeper: bool = False
    blocked_by_riskguard: bool = False
    blocked_by_max_positions: bool = False
    blocked_by_cooldown: bool = False
    blocked_by_live_disabled: bool = False
    blocked_by_wallet_absent: bool = False
    blocked_by_clean_feed_eligibility: bool = False
    strict_shadow_decision: bool = False
    exploration_decision: bool = False
    extra_flags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        extra = d.pop("extra_flags", {}) or {}
        d.update(extra)
        return d
