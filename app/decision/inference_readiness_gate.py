"""AE7C-1 inference readiness gate."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.decision.feature_parity_harness import HarnessParityStatus
from app.decision.scoring_policy_binding import ScoringPolicyBindingStatus


class InferenceGateStatus(StrEnum):
    PASS_READY_FOR_AE7C2_DRY_RUN = "PASS_READY_FOR_AE7C2_DRY_RUN"
    BLOCKED_POLICY_PLACEHOLDER = "BLOCKED_POLICY_PLACEHOLDER"
    BLOCKED_PARITY = "BLOCKED_PARITY"
    BLOCKED_SCHEMA = "BLOCKED_SCHEMA"
    BLOCKED_MISSING_FEATURES = "BLOCKED_MISSING_FEATURES"
    BLOCKED_WEAK_LINEAGE = "BLOCKED_WEAK_LINEAGE"
    BLOCKED_ARTIFACT_REPRODUCIBILITY = "BLOCKED_ARTIFACT_REPRODUCIBILITY"
    BLOCKED_SAFETY_CONSTRAINT = "BLOCKED_SAFETY_CONSTRAINT"
    BLOCKED_UNKNOWN = "BLOCKED_UNKNOWN"


PASS_BINDING_STATUSES = {
    ScoringPolicyBindingStatus.PASS_CONFIG_BOUND.value,
    ScoringPolicyBindingStatus.PASS_SIGNAL_CONTEXT_BOUND.value,
}

EXACT_PARITY_PASS = HarnessParityStatus.PASS_EXACT_OVERLAP.value


@dataclass
class InferenceReadinessGateResult:
    inference_gate_status: str
    inference_allowed: bool
    blocking_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    gate_inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "inference_gate_status": self.inference_gate_status,
            "inference_allowed": self.inference_allowed,
            "blocking_reasons": self.blocking_reasons,
            "warnings": self.warnings,
            "gate_inputs": self.gate_inputs,
        }


def evaluate_inference_readiness_gate(
    *,
    schema_compatibility_status: str,
    feature_parity_status: str,
    scoring_policy_binding_status: str,
    missing_required_features_count: int,
    forbidden_feature_check: str,
    lineage_confidence_score: float | None,
    exact_id_match: bool,
    model_artifacts_reproducible: bool,
    target_row_id_required: bool = False,
    external_calls_required: bool = False,
    weak_lineage_blocks_live: bool = True,
) -> InferenceReadinessGateResult:
    """Evaluate whether AE7C-2 local inference dry-run may proceed."""
    blocking: list[str] = []
    warnings: list[str] = []

    gate_inputs = {
        "schema_compatibility_status": schema_compatibility_status,
        "feature_parity_status": feature_parity_status,
        "scoring_policy_binding_status": scoring_policy_binding_status,
        "missing_required_features_count": missing_required_features_count,
        "forbidden_feature_check": forbidden_feature_check,
        "lineage_confidence_score": lineage_confidence_score,
        "exact_id_match": exact_id_match,
        "model_artifacts_reproducible": model_artifacts_reproducible,
        "target_row_id_required": target_row_id_required,
        "external_calls_required": external_calls_required,
    }

    if target_row_id_required:
        blocking.append("target_row_id_required_at_runtime")
    if external_calls_required:
        blocking.append("external_calls_required")
    if forbidden_feature_check != "PASS":
        blocking.append(f"forbidden_feature_check:{forbidden_feature_check}")
    if missing_required_features_count > 0:
        blocking.append(f"missing_required_features:{missing_required_features_count}")
    if schema_compatibility_status != "PASS":
        blocking.append(f"schema_compatibility_status:{schema_compatibility_status}")
    if feature_parity_status != EXACT_PARITY_PASS:
        blocking.append(f"feature_parity_status:{feature_parity_status}")
    if scoring_policy_binding_status not in PASS_BINDING_STATUSES:
        if scoring_policy_binding_status == ScoringPolicyBindingStatus.PLACEHOLDER_BOUND.value:
            blocking.append("scoring_policy_placeholder_not_inference_ready")
        else:
            blocking.append(f"scoring_policy_binding_status:{scoring_policy_binding_status}")
    if not model_artifacts_reproducible:
        blocking.append("model_artifacts_not_reproducible")
    if weak_lineage_blocks_live and not exact_id_match:
        warnings.append("weak_lineage_blocks_full_runtime_live_inference")
        blocking.append("weak_lineage_exact_id_match_false")

    if feature_parity_status == HarnessParityStatus.PASS_SYNTHETIC_FIXTURE_ONLY.value:
        warnings.append("synthetic_fixture_parity_does_not_fully_unblock_inference")

    if blocking:
        status = _primary_block_status(blocking, scoring_policy_binding_status, feature_parity_status)
        return InferenceReadinessGateResult(
            inference_gate_status=status,
            inference_allowed=False,
            blocking_reasons=blocking,
            warnings=warnings,
            gate_inputs=gate_inputs,
        )

    return InferenceReadinessGateResult(
        inference_gate_status=InferenceGateStatus.PASS_READY_FOR_AE7C2_DRY_RUN.value,
        inference_allowed=True,
        blocking_reasons=[],
        warnings=warnings,
        gate_inputs=gate_inputs,
    )


def _primary_block_status(
    blocking: list[str],
    binding_status: str,
    parity_status: str,
) -> str:
    if binding_status == ScoringPolicyBindingStatus.PLACEHOLDER_BOUND.value:
        return InferenceGateStatus.BLOCKED_POLICY_PLACEHOLDER.value
    if parity_status != EXACT_PARITY_PASS:
        return InferenceGateStatus.BLOCKED_PARITY.value
    if any("schema_compatibility" in b for b in blocking):
        return InferenceGateStatus.BLOCKED_SCHEMA.value
    if any("missing_required_features" in b for b in blocking):
        return InferenceGateStatus.BLOCKED_MISSING_FEATURES.value
    if any("weak_lineage" in b for b in blocking):
        return InferenceGateStatus.BLOCKED_WEAK_LINEAGE.value
    if any("model_artifacts_not_reproducible" in b for b in blocking):
        return InferenceGateStatus.BLOCKED_ARTIFACT_REPRODUCIBILITY.value
    if any("target_row_id" in b or "external_calls" in b or "forbidden" in b for b in blocking):
        return InferenceGateStatus.BLOCKED_SAFETY_CONSTRAINT.value
    return InferenceGateStatus.BLOCKED_UNKNOWN.value


def attempt_local_inference_if_allowed(
    *,
    gate_result: InferenceReadinessGateResult,
    allow_local_inference_if_gates_pass: bool,
) -> dict[str, Any]:
    """AE7C-1 infrastructure only — never calls model.predict/predict_proba."""
    if not allow_local_inference_if_gates_pass:
        return {
            "inference_executed": False,
            "reason": "allow_local_inference_if_gates_pass_not_set",
        }
    if not gate_result.inference_allowed:
        return {
            "inference_executed": False,
            "reason": "inference_gates_not_pass",
            "blocking_reasons": gate_result.blocking_reasons,
        }
    return {
        "inference_executed": False,
        "reason": "AE7C1_GATE_INFRASTRUCTURE_ONLY_DEFER_TO_AE7C2",
        "inference_would_be_allowed": True,
    }


def write_inference_gate_json(result: InferenceReadinessGateResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
