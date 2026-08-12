"""Unified candidate schema models (Phase E2)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from app.artifacts.hash_utils import sha256_hex
from app.candidates.validation import (
    compute_vote_count,
    normalize_event_timestamp,
    normalize_pair_address,
    validate_finite_numeric,
)

SCHEMA_VERSION = "candidate_schema_v1"


class ConsensusTier(StrEnum):
    TAB_XGB_RF_ALL3 = "TAB_XGB_RF_ALL3"
    TAB_RF_ONLY = "TAB_RF_ONLY"
    TAB_XGB_ONLY = "TAB_XGB_ONLY"
    XGB_RF_ONLY = "XGB_RF_ONLY"
    XGB_ONLY = "XGB_ONLY"
    TAB_ONLY = "TAB_ONLY"
    RF_ONLY = "RF_ONLY"
    NONE = "NONE"
    UNKNOWN = "UNKNOWN"


class EnrichmentStatus(StrEnum):
    NOT_REQUESTED = "NOT_REQUESTED"
    PENDING = "PENDING"
    AVAILABLE = "AVAILABLE"
    UNAVAILABLE = "UNAVAILABLE"
    FAILED = "FAILED"
    STALE = "STALE"
    BLOCKING_RISK = "BLOCKING_RISK"
    UNKNOWN = "UNKNOWN"


class LLMReviewStatus(StrEnum):
    NOT_REQUESTED = "NOT_REQUESTED"
    PENDING = "PENDING"
    AVAILABLE = "AVAILABLE"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"
    STALE = "STALE"
    UNKNOWN = "UNKNOWN"


class DecisionStatus(StrEnum):
    UNDECIDED = "UNDECIDED"
    WATCH = "WATCH"
    PAPER_BUY_CANDIDATE = "PAPER_BUY_CANDIDATE"
    PAPER_BUY_APPROVED = "PAPER_BUY_APPROVED"
    PAPER_BUY_EXECUTED = "PAPER_BUY_EXECUTED"
    BLOCKED = "BLOCKED"
    REJECTED_RESEARCH_ONLY = "REJECTED_RESEARCH_ONLY"
    ERROR = "ERROR"


def infer_consensus_tier(
    in_tab: bool | None,
    in_xgb: bool | None,
    in_rf: bool | None,
    *,
    strict: bool = True,
) -> ConsensusTier:
    """Infer consensus tier from model inclusion flags.

    When strict=True, any None flag yields UNKNOWN.
    When strict=False, None is treated as False.
    """
    if strict and any(flag is None for flag in (in_tab, in_xgb, in_rf)):
        return ConsensusTier.UNKNOWN

    tab = bool(in_tab)
    xgb = bool(in_xgb)
    rf = bool(in_rf)

    if tab and xgb and rf:
        return ConsensusTier.TAB_XGB_RF_ALL3
    if tab and not xgb and rf:
        return ConsensusTier.TAB_RF_ONLY
    if tab and xgb and not rf:
        return ConsensusTier.TAB_XGB_ONLY
    if not tab and xgb and rf:
        return ConsensusTier.XGB_RF_ONLY
    if xgb and not tab and not rf:
        return ConsensusTier.XGB_ONLY
    if tab and not xgb and not rf:
        return ConsensusTier.TAB_ONLY
    if rf and not tab and not xgb:
        return ConsensusTier.RF_ONLY
    if not tab and not xgb and not rf:
        return ConsensusTier.NONE
    return ConsensusTier.UNKNOWN


def compute_candidate_id(
    *,
    chain: str,
    pair_address: str,
    event_timestamp_normalized: str,
    source: str,
    source_row_id: str | None = None,
) -> str:
    """Deterministic SHA-256 candidate id from normalized identity fields."""
    normalized_pair = normalize_pair_address(pair_address, chain)
    parts = [
        chain.strip(),
        normalized_pair,
        event_timestamp_normalized,
        source.strip(),
    ]
    if source_row_id is not None and str(source_row_id).strip() != "":
        parts.append(str(source_row_id).strip())
    payload = "|".join(parts)
    return sha256_hex(f"candidate:v1|{payload}")


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


class CandidateIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    pair_address: str
    chain: str
    event_timestamp: datetime | str | int | float
    event_timestamp_normalized: str
    timestamp_precision: Literal["seconds", "milliseconds"] = "seconds"
    source: str
    source_artifact_id: str | None = None
    source_row_id: str | None = None
    created_at: str
    coin_id: str | None = None
    symbol: str | None = None
    base_symbol: str | None = None
    quote_symbol: str | None = None

    @model_validator(mode="before")
    @classmethod
    def prepare_identity(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        precision = payload.get("timestamp_precision", "seconds")
        normalized_ts = payload.get("event_timestamp_normalized")
        if normalized_ts is None:
            normalized_ts = normalize_event_timestamp(
                payload["event_timestamp"],
                precision=precision,
            )
        payload["event_timestamp_normalized"] = normalized_ts
        payload["created_at"] = payload.get("created_at") or utc_now_iso()
        if not payload.get("candidate_id"):
            payload["candidate_id"] = compute_candidate_id(
                chain=payload["chain"],
                pair_address=payload["pair_address"],
                event_timestamp_normalized=normalized_ts,
                source=payload["source"],
                source_row_id=payload.get("source_row_id"),
            )
        return payload


class MarketContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    price: float | None = None
    liquidity_usd: float | None = None
    volume_24h: float | None = None
    fdv: float | None = None
    txns_buys: float | None = None
    txns_sells: float | None = None
    buy_sell_ratio: float | None = None
    price_change_5m: float | None = None
    price_change_1h: float | None = None
    price_change_4h: float | None = None
    price_change_24h: float | None = None

    @field_validator(
        "price",
        "liquidity_usd",
        "volume_24h",
        "fdv",
        "txns_buys",
        "txns_sells",
        "buy_sell_ratio",
        "price_change_5m",
        "price_change_1h",
        "price_change_4h",
        "price_change_24h",
        mode="before",
    )
    @classmethod
    def validate_market_numeric(cls, value: Any) -> float | None:
        return validate_finite_numeric(value)


class ModelScores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    score_xgb: float | None = Field(default=None, ge=0, le=1)
    score_tab: float | None = Field(default=None, ge=0, le=1)
    score_rf: float | None = Field(default=None, ge=0, le=1)
    rank_pct_xgb: float | None = Field(default=None, ge=0, le=1)
    rank_pct_tab: float | None = Field(default=None, ge=0, le=1)
    rank_pct_rf: float | None = Field(default=None, ge=0, le=1)
    in_xgb: bool | None = None
    in_tab: bool | None = None
    in_rf: bool | None = None
    vote_count: int = Field(default=0, ge=0, le=3)

    @model_validator(mode="before")
    @classmethod
    def prepare_vote_count(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        expected = compute_vote_count(
            payload.get("in_xgb"),
            payload.get("in_tab"),
            payload.get("in_rf"),
        )
        if payload.get("vote_count") is None:
            payload["vote_count"] = expected
        elif payload["vote_count"] != expected:
            raise ValueError(
                f"vote_count {payload['vote_count']} inconsistent with inclusion flags "
                f"(expected {expected})"
            )
        return payload


class ExitPolicyContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    exit_policy_id: str | None = None
    horizon: str | None = None
    top_pct: float | None = None
    pair_cap: int | Literal["none"] | None = None
    tp_ratio: float | None = None
    sl_ratio: float | None = None
    time_stop_minutes: float | None = None
    round_trip_fee_pct: float | None = None
    selected_by_policy: bool | None = None
    sim_exit_status: str | None = None
    sim_net_return: float | None = None
    target_net_profitable_after_exit: bool | None = None

    @field_validator("top_pct", "tp_ratio", "sl_ratio", "time_stop_minutes", mode="before")
    @classmethod
    def validate_policy_numeric(cls, value: Any) -> float | None:
        return validate_finite_numeric(value)

    @field_validator("round_trip_fee_pct", "sim_net_return", mode="before")
    @classmethod
    def validate_decimal_fraction(cls, value: Any) -> float | None:
        return validate_finite_numeric(value)

    @field_validator("pair_cap", mode="before")
    @classmethod
    def validate_pair_cap(cls, value: Any) -> int | Literal["none"] | None:
        if value is None:
            return None
        if isinstance(value, str) and value.strip().lower() == "none":
            return "none"
        numeric = validate_finite_numeric(value)
        if numeric is None:
            return None
        return int(numeric)


class ArtifactLineage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_artifact_id: str | None = None
    model_prediction_artifact_ids: dict[str, str] = Field(default_factory=dict)
    exit_policy_artifact_id: str | None = None
    registry_version: str | None = None
    content_hash: str | None = None
    schema_hash: str | None = None
    lineage_warnings: list[str] = Field(default_factory=list)


class EnrichmentState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    solana_enrichment_status: EnrichmentStatus = EnrichmentStatus.NOT_REQUESTED
    helius_validation_status: EnrichmentStatus = EnrichmentStatus.NOT_REQUESTED
    wallet_identity_status: EnrichmentStatus = EnrichmentStatus.NOT_REQUESTED
    wallet_behavior_status: EnrichmentStatus = EnrichmentStatus.NOT_REQUESTED
    rss_sentiment_status: EnrichmentStatus = EnrichmentStatus.NOT_REQUESTED
    reputation_status: EnrichmentStatus = EnrichmentStatus.NOT_REQUESTED
    whale_intelligence_status: EnrichmentStatus = EnrichmentStatus.NOT_REQUESTED
    solana_summary: str | None = None
    helius_summary: str | None = None
    wallet_summary: str | None = None
    sentiment_score: float | None = Field(default=None, ge=-1, le=1)
    sentiment_summary: str | None = None
    reputation_risk_score: float | None = Field(default=None, ge=0, le=1)
    reputation_summary: str | None = None
    whale_behavior_score: float | None = Field(default=None, ge=0, le=1)
    whale_summary: str | None = None


class LLMReviewState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qwen_review_status: LLMReviewStatus = LLMReviewStatus.NOT_REQUESTED
    gemini_review_status: LLMReviewStatus = LLMReviewStatus.NOT_REQUESTED
    qwen_summary: str | None = None
    gemini_summary: str | None = None
    llm_veto: bool | None = None
    llm_risk_flags: list[str] = Field(default_factory=list)
    llm_review_artifact_id: str | None = None


class CandidateDecisionState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: DecisionStatus = DecisionStatus.UNDECIDED
    decision_reason_codes: list[str] = Field(default_factory=list)
    decision_trace_id: str | None = None
    decision_created_at: str | None = None
    paper_trade_id: str | None = None
    paper_position_id: str | None = None


class UnifiedCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    identity: CandidateIdentity
    market: MarketContext = Field(default_factory=MarketContext)
    model_scores: ModelScores = Field(default_factory=ModelScores)
    consensus_tier: ConsensusTier
    exit_policy: ExitPolicyContext = Field(default_factory=ExitPolicyContext)
    lineage: ArtifactLineage = Field(default_factory=ArtifactLineage)
    enrichment: EnrichmentState = Field(default_factory=EnrichmentState)
    llm_review: LLMReviewState = Field(default_factory=LLMReviewState)
    decision: CandidateDecisionState = Field(default_factory=CandidateDecisionState)
    warnings: list[str] = Field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    @model_validator(mode="before")
    @classmethod
    def prepare_candidate(cls, data: Any, info: ValidationInfo) -> Any:
        if not isinstance(data, dict):
            return data
        payload = dict(data)
        strict_consensus = True
        if info.context is not None:
            strict_consensus = bool(info.context.get("strict_consensus", True))

        model_scores = payload.get("model_scores") or {}
        if isinstance(model_scores, ModelScores):
            scores_data = model_scores.model_dump()
        elif isinstance(model_scores, dict):
            scores_data = model_scores
        else:
            scores_data = {}

        inferred = infer_consensus_tier(
            scores_data.get("in_tab"),
            scores_data.get("in_xgb"),
            scores_data.get("in_rf"),
            strict=strict_consensus,
        )

        warnings = list(payload.get("warnings") or [])
        consensus_tier = payload.get("consensus_tier")
        if consensus_tier is None:
            payload["consensus_tier"] = inferred.value
        else:
            existing = (
                consensus_tier
                if isinstance(consensus_tier, ConsensusTier)
                else ConsensusTier(str(consensus_tier))
            )
            if existing != inferred:
                message = (
                    f"consensus_tier {existing.value} conflicts with inferred "
                    f"{inferred.value} from inclusion flags"
                )
                if strict_consensus:
                    raise ValueError(message)
                warnings.append(message)
            payload["consensus_tier"] = existing.value

        identity = payload.get("identity")
        if isinstance(identity, CandidateIdentity):
            if identity.event_timestamp_normalized is None:
                raise ValueError("event_timestamp_normalized must be present after construction")
        elif isinstance(identity, dict) and not identity.get("event_timestamp_normalized"):
            precision = identity.get("timestamp_precision", "seconds")
            identity = dict(identity)
            identity["event_timestamp_normalized"] = normalize_event_timestamp(
                identity["event_timestamp"],
                precision=precision,
            )
            payload["identity"] = identity

        payload["warnings"] = warnings
        return payload

    def to_dict(self) -> dict[str, Any]:
        from app.candidates.serialization import candidate_to_dict

        return candidate_to_dict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any], *, strict_consensus: bool = True) -> UnifiedCandidate:
        from app.candidates.serialization import candidate_from_dict

        return candidate_from_dict(payload, strict_consensus=strict_consensus)

    def to_json(self, *, indent: int | None = None) -> str:
        from app.candidates.serialization import candidate_to_json

        return candidate_to_json(self, indent=indent)

    @classmethod
    def from_json(cls, payload: str, *, strict_consensus: bool = True) -> UnifiedCandidate:
        from app.candidates.serialization import candidate_from_json

        return candidate_from_json(payload, strict_consensus=strict_consensus)

    def to_flat_dict(self, *, target_format: str = "parquet") -> dict[str, Any]:
        from app.candidates.serialization import candidate_to_flat_dict

        return candidate_to_flat_dict(self, target_format=target_format)

    @classmethod
    def from_flat_dict(
        cls,
        row: dict[str, Any],
        *,
        source_format: str = "parquet",
        strict_consensus: bool = True,
    ) -> UnifiedCandidate:
        from app.candidates.serialization import candidate_from_flat_dict

        return candidate_from_flat_dict(
            row,
            source_format=source_format,
            strict_consensus=strict_consensus,
        )
