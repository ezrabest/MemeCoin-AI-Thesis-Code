"""AE6/AE7 decision layer — audit-ready records without trade authority."""

from app.decision.builder import build_decision_record, build_lineage_metadata
from app.decision.consensus import compute_consensus
from app.decision.model_scores import AE7_PHASE
from app.decision.persistence import (
    DecisionJsonlWriter,
    read_jsonl_records_safe,
    write_decision_record_jsonl,
)
from app.decision.ae7c0_feature_enrichment import AE7C0_PHASE, run_ae7c0_enrichment
from app.decision.ae7c1_binding_parity_gate import AE7C1_PHASE, run_ae7c1_binding_parity_gate
from app.decision.meta_layer_decision import AE7_FINAL_PHASE
from app.decision.runtime_feature_bridge import AE7B_PHASE, run_ae7b_bridge
from app.decision.score_population import enrich_decision_record, run_ae7_score_population
from app.decision.types import (
    AE6_PHASE,
    DecisionRecord,
    LineageMetadata,
    LineageValidationError,
)

__all__ = [
    "AE6_PHASE",
    "AE7_PHASE",
    "DecisionJsonlWriter",
    "DecisionRecord",
    "LineageMetadata",
    "LineageValidationError",
    "build_decision_record",
    "build_lineage_metadata",
    "compute_consensus",
    "enrich_decision_record",
    "read_jsonl_records_safe",
    "run_ae7_score_population",
    "AE7B_PHASE",
    "run_ae7b_bridge",
    "AE7C0_PHASE",
    "run_ae7c0_enrichment",
    "AE7C1_PHASE",
    "run_ae7c1_binding_parity_gate",
    "AE7_FINAL_PHASE",
    "write_decision_record_jsonl",
]
