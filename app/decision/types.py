"""AE6 decision record types (Pydantic v2)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

AE6_PHASE = "AE6_CONSENSUS_DECISION_LAYER"

LINEAGE_FALLBACK_REASON = (
    "Lineage fallback: best-effort match using provider, pair_address, timestamps, and scan/run context."
)
LINEAGE_IMPLICIT_CAVEAT = (
    "RAW/derived lineage is best-effort implicit, based on provider/pair/timestamp context."
)

MODEL_MISSING_REASON = "NOT_AVAILABLE_IN_CURRENT_RUNTIME_CONTEXT"
AE8_MISSING_REASON = "AE8_NOT_IMPLEMENTED_YET"
AE9_MISSING_REASON = "AE9_NOT_IMPLEMENTED_YET"


class LineageValidationError(ValueError):
    """Raised when lineage metadata is missing or fails AE6 fail-closed rules."""


class DecisionStatusAE6(StrEnum):
    WATCH = "WATCH"
    RESEARCH_CANDIDATE = "RESEARCH_CANDIDATE"
    PAPER_CANDIDATE_REVIEW = "PAPER_CANDIDATE_REVIEW"
    BLOCK = "BLOCK"
    NO_DECISION = "NO_DECISION"


class LineageMode(StrEnum):
    EXPLICIT_LINKAGE = "EXPLICIT_LINKAGE"
    BEST_EFFORT_IMPLICIT_LINKAGE = "BEST_EFFORT_IMPLICIT_LINKAGE"


class LineageStrength(StrEnum):
    STRONG_EXPLICIT_LINKS = "STRONG_EXPLICIT_LINKS"
    WEAK_IMPLICIT_TIME_PAIR_LINKS = "WEAK_IMPLICIT_TIME_PAIR_LINKS"


class LineageResolutionMethod(StrEnum):
    EXPLICIT_COLUMN = "EXPLICIT_COLUMN"
    FOREIGN_KEY = "FOREIGN_KEY"
    DIRECT_SOURCE_REFERENCE = "DIRECT_SOURCE_REFERENCE"
    BEST_EFFORT_PAIR_TIME_MATCH = "BEST_EFFORT_PAIR_TIME_MATCH"
    BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH = "BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH"
    MISSING = "MISSING"


class ConsensusFamily(StrEnum):
    TAB_XGB_RF_ALL3 = "TAB_XGB_RF_ALL3"
    TAB_RF_ONLY = "TAB_RF_ONLY"
    TAB_XGB_ONLY = "TAB_XGB_ONLY"
    XGB_RF_ONLY = "XGB_RF_ONLY"
    SINGLE_MODEL_ONLY = "SINGLE_MODEL_ONLY"
    NO_MODEL_CONSENSUS_AVAILABLE = "NO_MODEL_CONSENSUS_AVAILABLE"


class LineageMetadata(BaseModel):
    """Mandatory lineage object for every AE6 decision record."""

    model_config = ConfigDict(extra="forbid")

    lineage_mode: LineageMode
    lineage_strength: LineageStrength
    provider: str | None = None
    source: str | None = None
    endpoint: str | None = None
    pair_address: str | None = None
    symbol: str | None = None
    snapshot_timestamp: str | None = None
    signal_timestamp: str | None = None
    raw_payload_id: int | str | None = None
    snapshot_id: int | str | None = None
    signal_id: int | str | None = None
    raw_payload_id_resolution_method: LineageResolutionMethod = (
        LineageResolutionMethod.MISSING
    )
    snapshot_id_resolution_method: LineageResolutionMethod = (
        LineageResolutionMethod.MISSING
    )
    signal_id_resolution_method: LineageResolutionMethod = (
        LineageResolutionMethod.MISSING
    )
    raw_payload_timestamp_window: str | None = None
    fallback_reason: str | None = None
    lineage_warning: str | None = None

    @model_validator(mode="after")
    def validate_lineage_contract(self) -> LineageMetadata:
        if self.lineage_mode is None:
            raise LineageValidationError("lineage_mode is required")
        if self.lineage_strength is None:
            raise LineageValidationError("lineage_strength is required")

        explicit_methods = {
            LineageResolutionMethod.EXPLICIT_COLUMN,
            LineageResolutionMethod.FOREIGN_KEY,
            LineageResolutionMethod.DIRECT_SOURCE_REFERENCE,
        }
        best_effort_methods = {
            LineageResolutionMethod.BEST_EFFORT_PAIR_TIME_MATCH,
            LineageResolutionMethod.BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH,
        }
        required_links = (
            ("raw_payload_id", self.raw_payload_id, self.raw_payload_id_resolution_method),
            ("snapshot_id", self.snapshot_id, self.snapshot_id_resolution_method),
            ("signal_id", self.signal_id, self.signal_id_resolution_method),
        )

        for field_name, value, method in required_links:
            if value is None:
                if method != LineageResolutionMethod.MISSING:
                    raise LineageValidationError(
                        f"{field_name}_resolution_method must be MISSING when {field_name} is absent"
                    )
            elif method == LineageResolutionMethod.MISSING:
                raise LineageValidationError(
                    f"{field_name}_resolution_method cannot be MISSING when {field_name} is present"
                )

        has_missing_required = any(
            value is None for _, value, _ in required_links
        )
        has_best_effort = any(
            value is not None and method in best_effort_methods
            for _, value, method in required_links
        )
        all_explicit = all(
            value is not None and method in explicit_methods
            for _, value, method in required_links
        )

        if has_missing_required:
            raise LineageValidationError(
                "required lineage links (raw_payload_id, snapshot_id, signal_id) must be present"
            )

        if all_explicit:
            if self.lineage_mode != LineageMode.EXPLICIT_LINKAGE:
                raise LineageValidationError(
                    "lineage_mode must be EXPLICIT_LINKAGE when all required links are explicitly resolved"
                )
            if self.lineage_strength != LineageStrength.STRONG_EXPLICIT_LINKS:
                raise LineageValidationError(
                    "lineage_strength must be STRONG_EXPLICIT_LINKS when all required links are explicitly resolved"
                )
        elif has_best_effort:
            if self.lineage_mode != LineageMode.BEST_EFFORT_IMPLICIT_LINKAGE:
                raise LineageValidationError(
                    "lineage_mode must be BEST_EFFORT_IMPLICIT_LINKAGE when any required link is best-effort"
                )
            if self.lineage_strength != LineageStrength.WEAK_IMPLICIT_TIME_PAIR_LINKS:
                raise LineageValidationError(
                    "lineage_strength must be WEAK_IMPLICIT_TIME_PAIR_LINKS when any required link is best-effort"
                )
            if not (self.fallback_reason or "").strip():
                raise LineageValidationError(
                    "fallback_reason is required for BEST_EFFORT_IMPLICIT_LINKAGE"
                )
            if not (self.lineage_warning or "").strip():
                raise LineageValidationError(
                    "lineage_warning is required for BEST_EFFORT_IMPLICIT_LINKAGE"
                )
        else:
            raise LineageValidationError(
                "explicit linkage requires explicit structural resolution methods"
            )
        return self


class CandidateIdentityBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pair_address: str | None = None
    chain: str | None = None
    symbol: str | None = None
    base_token_address: str | None = None
    quote_token_address: str | None = None
    coin_id: int | str | None = None
    candidate_id: str | None = None
    candidate_policy_id: str | None = None
    target_row_id: str | None = None
    event_timestamp: str | None = None
    source_signal_id: int | str | None = None
    source_snapshot_id: int | str | None = None
    source_raw_payload_id: int | str | None = None


class ModelScoreSlot(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available: bool = False
    score: float | None = None
    rank: float | None = None
    model_artifact_id: str | None = None
    prediction_artifact_id: str | None = None
    horizon: str | None = None
    filter: str | None = None
    exit_policy: str | None = None
    missing_reason: str | None = MODEL_MISSING_REASON


class ModelScoresBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    RF: ModelScoreSlot = Field(default_factory=ModelScoreSlot)
    XGB: ModelScoreSlot = Field(default_factory=ModelScoreSlot)
    TAB: ModelScoreSlot = Field(default_factory=ModelScoreSlot)
    META: ModelScoreSlot = Field(default_factory=ModelScoreSlot)


class ConsensusBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    available_model_count: int = 0
    vote_count: int = 0
    consensus_family: ConsensusFamily = ConsensusFamily.NO_MODEL_CONSENSUS_AVAILABLE
    consensus_strength: str = "UNAVAILABLE"
    consensus_caveat: str | None = None


class ResearchContextBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    historical_signal_family: str = "NOT_EVALUATED"
    direct_target_signal_status: str = "NOT_EVALUATED"
    pair_concentration_status: str = "NOT_EVALUATED"
    robustness_status: str = "NOT_EVALUATED"
    whale_score_asof_status: str = "RESEARCH_ONLY_PLAUSIBLE_FEATURE_CANDIDATE"
    whale_score_asof_not_rule: bool = True
    whale_score_asof_not_runtime_approved: bool = True
    context_evidence_status: str = "NOT_EVALUATED"


class ContextPlaceholdersBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rss_sentiment_available: bool = False
    rss_sentiment_score: float | None = None
    rss_source_count: int | None = None
    rss_caveat: str | None = None
    helius_available: bool = False
    helius_missing_reason: str = AE8_MISSING_REASON
    solana_available: bool = False
    solana_missing_reason: str = AE8_MISSING_REASON
    wallet_intelligence_available: bool = False
    wallet_intelligence_missing_reason: str = AE8_MISSING_REASON
    reputation_available: bool = False
    reputation_missing_reason: str = AE8_MISSING_REASON
    scam_flags_available: bool = False
    scam_flags_missing_reason: str = AE8_MISSING_REASON
    context_support_score: float | None = None
    context_risk_score: float | None = None
    context_missingness: list[str] = Field(default_factory=list)
    context_caveats: list[str] = Field(default_factory=list)


class LLMContextBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    qwen_memo_available: bool = False
    gemini_audit_available: bool = False
    llm_execution_allowed: bool = False
    llm_decision_authority: bool = False
    llm_missing_reason: str = AE9_MISSING_REASON


class RiskContextBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    risk_gate_evaluated: bool = False
    risk_notes: list[str] = Field(default_factory=list)


class DecisionRecord(BaseModel):
    """Full AE6 consensus decision record — no trade authority."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at_utc: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    phase: Literal["AE6_CONSENSUS_DECISION_LAYER"] = AE6_PHASE
    mode: str = "AUDIT"
    decision_status: DecisionStatusAE6 = DecisionStatusAE6.NO_DECISION
    decision_confidence: float | None = None
    confidence_components: dict[str, float | None] = Field(default_factory=dict)
    candidate_identity: CandidateIdentityBlock = Field(default_factory=CandidateIdentityBlock)
    lineage: LineageMetadata
    market_context: dict[str, Any] = Field(default_factory=dict)
    signal_context: dict[str, Any] = Field(default_factory=dict)
    model_scores: ModelScoresBlock = Field(default_factory=ModelScoresBlock)
    consensus: ConsensusBlock = Field(default_factory=ConsensusBlock)
    research_context: ResearchContextBlock = Field(default_factory=ResearchContextBlock)
    llm_context: LLMContextBlock = Field(default_factory=LLMContextBlock)
    risk_context: RiskContextBlock = Field(default_factory=RiskContextBlock)
    context_placeholders: ContextPlaceholdersBlock = Field(
        default_factory=ContextPlaceholdersBlock
    )
    missingness: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    no_trade_authority: Literal[True] = True

    @model_validator(mode="after")
    def require_lineage(self) -> DecisionRecord:
        if self.lineage is None:
            raise LineageValidationError("lineage is mandatory for DecisionRecord")
        return self
