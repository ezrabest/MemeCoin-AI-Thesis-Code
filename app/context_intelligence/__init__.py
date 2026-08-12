"""AE8 context intelligence collection / integration layer."""

from app.context_intelligence.context_audits import decide_ae8_status, write_ae8_audits
from app.context_intelligence.context_feature_builder import (
    build_context_feature_record,
    build_context_lineage,
    run_ae8_context_intelligence,
)
from app.context_intelligence.context_persistence import (
    ContextJsonlWriter,
    context_jsonl_path_for_date,
    read_context_jsonl_safe,
)
from app.context_intelligence.context_schema import (
    ContextSchema,
    build_context_schema,
    compute_context_schema_id,
    compute_schema_hash,
    validate_feature_names,
)
from app.context_intelligence.types import is_forbidden_context_feature
from app.context_intelligence.freshness import (
    apply_stale_nulling,
    compute_freshness,
    default_threshold_for_family,
)
from app.context_intelligence.types import (
    AE8_PHASE,
    ContextFeatureRecord,
    FreshnessMode,
    FreshnessStatus,
    SourceStatus,
)

__all__ = [
    "AE8_PHASE",
    "ContextFeatureRecord",
    "ContextSchema",
    "ContextJsonlWriter",
    "FreshnessMode",
    "FreshnessStatus",
    "SourceStatus",
    "apply_stale_nulling",
    "build_context_feature_record",
    "build_context_lineage",
    "build_context_schema",
    "compute_context_schema_id",
    "compute_freshness",
    "compute_schema_hash",
    "context_jsonl_path_for_date",
    "decide_ae8_status",
    "default_threshold_for_family",
    "is_forbidden_context_feature",
    "read_context_jsonl_safe",
    "run_ae8_context_intelligence",
    "validate_feature_names",
    "write_ae8_audits",
]
