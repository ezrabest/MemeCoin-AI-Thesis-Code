"""AE7 model score slot types, constants, and helpers."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.decision.types import ModelScoreSlot, ModelScoresBlock

AE7_PHASE = "AE7_MODEL_SCORE_SLOT_POPULATION"
AE6_SOURCE_PHASE = "AE6_CONSENSUS_DECISION_LAYER"

# Exact identity alignment keys — priority order for lookup.
EXACT_ID_COLUMNS: tuple[str, ...] = (
    "target_row_id",
    "candidate_policy_id",
    "candidate_id",
)

MODEL_FAMILIES: tuple[str, ...] = ("RF", "XGB", "TAB")

# Columns that must never be used as score sources.
LEAKAGE_COLUMN_FRAGMENTS: tuple[str, ...] = (
    "target",
    "label",
    "future",
    "return",
    "net_return",
    "exit",
    "tp",
    "sl",
    "profit",
    "profitable",
    "realized",
    "outcome",
    "simulation",
)

# Safe score column name fragments (case-insensitive substring match).
SAFE_SCORE_FRAGMENTS: tuple[str, ...] = (
    "score",
    "probability",
    "prediction",
    "prob",
    "rank",
)

# Disallowed alignment — never used for model-score population.
FORBIDDEN_ALIGNMENT_KEYS: frozenset[str] = frozenset(
    {
        "pair_address",
        "event_timestamp",
        "symbol",
        "provider",
        "liquidity",
        "volume",
    }
)

MAX_SAFE_INSPECTION_BYTES = 500 * 1024 * 1024
MAX_INDEX_BUILD_BYTES = 200 * 1024 * 1024

# Score population modes (AE7 implements historical/offline only).
SCORE_POPULATION_MODE_HISTORICAL_OFFLINE = "HISTORICAL_OFFLINE_EXACT_ID_LOOKUP"
SCORE_POPULATION_MODE_RUNTIME_LIVE = "RUNTIME_LIVE_FEATURE_INFERENCE"

BRIDGE_PHASE_NAME = "Runtime Candidate Identity + Feature Matrix Bridge"
BRIDGE_PHASE_GOAL = (
    "Enable runtime model inference by preserving candidate identity and an as-of "
    "feature matrix keyed for registered model artifacts — not by looking up "
    "historical labeled target_row_id values at live collection time."
)
RUNTIME_BRIDGE_FIELDS: tuple[str, ...] = (
    "candidate_id",
    "candidate_policy_id",
    "scoring_policy_id",
    "as_of_feature_row_id",
    "feature_schema_id",
    "model_artifact_id",
    "runtime_inference_id",
)


class ScorePopulationDecision(StrEnum):
    MODEL_SCORE_SLOTS_POPULATED = "MODEL_SCORE_SLOTS_POPULATED"
    PARTIAL_MODEL_SCORE_SLOT_POPULATION = "PARTIAL_MODEL_SCORE_SLOT_POPULATION"
    NO_SAFE_RUNTIME_ALIGNMENT_YET = "NO_SAFE_RUNTIME_ALIGNMENT_YET"
    RUNTIME_IDENTITY_BRIDGE_REQUIRED = "RUNTIME_IDENTITY_BRIDGE_REQUIRED"
    BLOCKED_UNSAFE_ALIGNMENT_ONLY = "BLOCKED_UNSAFE_ALIGNMENT_ONLY"


class ArtifactStatus(StrEnum):
    CURRENT = "CURRENT"
    STALE = "STALE"
    DEPRECATED = "DEPRECATED"
    UNREGISTERED = "UNREGISTERED"
    UNKNOWN = "UNKNOWN"
    TOO_LARGE_SKIPPED = "TOO_LARGE_SKIPPED"
    REJECTED_LEAKAGE_RISK = "REJECTED_LEAKAGE_RISK"
    REJECTED_NO_SAFE_ID = "REJECTED_NO_SAFE_ID"
    REJECTED_NO_SAFE_SCORE = "REJECTED_NO_SAFE_SCORE"


class ModelFamily(StrEnum):
    RF = "RF"
    XGB = "XGB"
    TAB = "TAB"
    UNKNOWN = "UNKNOWN"


class ArtifactRole(StrEnum):
    PREDICTION_TABLE = "prediction_table"
    SELECTED_TRADES = "selected_trades"
    METRICS = "metrics"
    MANIFEST = "manifest"
    MODEL_ARTIFACT = "model_artifact"
    UNKNOWN = "unknown"


class MissingReason(StrEnum):
    NO_SAFE_EXACT_ID_ALIGNMENT = "NO_SAFE_EXACT_ID_ALIGNMENT"
    RUNTIME_RECORD_MISSING_MODEL_COMPATIBLE_ID = "RUNTIME_RECORD_MISSING_MODEL_COMPATIBLE_ID"
    NO_SAFE_SCORE_COLUMN = "NO_SAFE_SCORE_COLUMN"
    NO_SAFE_MODEL_ARTIFACT = "NO_SAFE_MODEL_ARTIFACT"
    ARTIFACT_NOT_REPRODUCIBLE_OR_STALE = "ARTIFACT_NOT_REPRODUCIBLE_OR_STALE"
    NOT_AVAILABLE_IN_CURRENT_RUNTIME_CONTEXT = "NOT_AVAILABLE_IN_CURRENT_RUNTIME_CONTEXT"


class PopulatedModelScoreSlot(BaseModel):
    """AE7-enriched model score slot — superset of AE6 ``ModelScoreSlot``."""

    model_config = ConfigDict(extra="forbid")

    available: bool = False
    score: float | None = None
    rank: float | None = None
    model_artifact_id: str | None = None
    prediction_artifact_id: str | None = None
    horizon: str | None = None
    filter: str | None = None
    exit_policy: str | None = None
    missing_reason: str | None = None
    # AE7 enrichment fields
    model_family: str | None = None
    id_key_used: str | None = None
    id_value: str | None = None
    split: str | None = None
    score_column_used: str | None = None
    rank_column_used: str | None = None
    population_method: str | None = None
    artifact_status: str | None = None
    is_reproducible: bool | None = None
    artifact_path: str | None = None


class PopulatedModelScoresBlock(BaseModel):
    model_config = ConfigDict(extra="forbid")

    RF: PopulatedModelScoreSlot = Field(default_factory=PopulatedModelScoreSlot)
    XGB: PopulatedModelScoreSlot = Field(default_factory=PopulatedModelScoreSlot)
    TAB: PopulatedModelScoreSlot = Field(default_factory=PopulatedModelScoreSlot)
    META: PopulatedModelScoreSlot = Field(default_factory=PopulatedModelScoreSlot)


def unavailable_ae7_slot(
    *,
    missing_reason: str,
    model_family: str | None = None,
) -> PopulatedModelScoreSlot:
    return PopulatedModelScoreSlot(
        available=False,
        missing_reason=missing_reason,
        model_family=model_family,
    )


def unavailable_ae7_scores() -> PopulatedModelScoresBlock:
    reason = MissingReason.NOT_AVAILABLE_IN_CURRENT_RUNTIME_CONTEXT.value
    slot = unavailable_ae7_slot(missing_reason=reason)
    return PopulatedModelScoresBlock(RF=slot, XGB=slot, TAB=slot, META=slot)


def ae6_slot_to_populated(slot: ModelScoreSlot | dict[str, Any]) -> PopulatedModelScoreSlot:
    if isinstance(slot, ModelScoreSlot):
        data = slot.model_dump()
    else:
        data = dict(slot)
    return PopulatedModelScoreSlot(**{k: v for k, v in data.items() if k in PopulatedModelScoreSlot.model_fields})


def populated_slot_to_ae6_dict(slot: PopulatedModelScoreSlot) -> dict[str, Any]:
    """Serialize populated slot for JSONL — includes AE7 extension fields."""
    return slot.model_dump(mode="json")


def can_attempt_offline_exact_id_lookup(identity: dict[str, Any]) -> bool:
    """True when the record carries any exact-ID key usable for offline artifact lookup."""
    return any(_non_empty(identity.get(key)) for key in EXACT_ID_COLUMNS)


def has_model_compatible_runtime_id(identity: dict[str, Any]) -> bool:
    """True when the record has runtime bridge fields for live inference.

    ``target_row_id`` is a historical labeled-row key and is intentionally excluded —
    it is not normally knowable at live collection time.
    """
    return bool(
        _non_empty(identity.get("candidate_policy_id"))
        or _non_empty(identity.get("scoring_policy_id"))
        or (
            _non_empty(identity.get("as_of_feature_row_id"))
            and _non_empty(identity.get("feature_schema_id"))
        )
    )


def runtime_bridge_fields_present(identity: dict[str, Any]) -> dict[str, bool]:
    """Presence map for next-phase runtime bridge fields."""
    return {field: _non_empty(identity.get(field)) for field in RUNTIME_BRIDGE_FIELDS}


def runtime_id_keys_present(identity: dict[str, Any]) -> dict[str, bool]:
    """Presence map for exact-ID keys (including historical-only target_row_id)."""
    return {
        "target_row_id": _non_empty(identity.get("target_row_id")),
        "candidate_policy_id": _non_empty(identity.get("candidate_policy_id")),
        "candidate_id": _non_empty(identity.get("candidate_id")),
    }


def _non_empty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    return True


def column_is_leakage_risk(column_name: str) -> bool:
    lower = column_name.lower()
    return any(frag in lower for frag in LEAKAGE_COLUMN_FRAGMENTS)


def column_is_safe_score(column_name: str) -> bool:
    lower = column_name.lower()
    if column_is_leakage_risk(column_name):
        return False
    return any(frag in lower for frag in SAFE_SCORE_FRAGMENTS)


def infer_model_family_from_path(path: str) -> ModelFamily:
    lower = path.lower().replace("\\", "/")
    if "/tabicl/" in lower or "tabicl" in lower or "_tab_" in lower or "tab_score" in lower:
        return ModelFamily.TAB
    if "tab_" in lower and "table" not in lower:
        return ModelFamily.TAB
    if "_rf" in lower or "/rf_" in lower or "random_forest" in lower or "_rf." in lower:
        return ModelFamily.RF
    if "_xgb" in lower or "/xgb" in lower or "xgboost" in lower:
        return ModelFamily.XGB
    if "direct_target_tabicl" in lower:
        return ModelFamily.TAB
    if "predictions" in lower and "rf" in lower:
        return ModelFamily.RF
    if "predictions" in lower and "xgb" in lower:
        return ModelFamily.XGB
    return ModelFamily.UNKNOWN


def scores_block_from_ae6(model_scores: dict[str, Any] | ModelScoresBlock) -> ModelScoresBlock:
    if isinstance(model_scores, ModelScoresBlock):
        return model_scores
    return ModelScoresBlock.model_validate(model_scores)
