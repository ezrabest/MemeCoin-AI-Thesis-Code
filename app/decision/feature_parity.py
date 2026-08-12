"""AE7B feature parity audit between runtime and offline/training schemas."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

FLOAT_TOLERANCE = 1e-6
MAX_PARITY_ROWS = 100


class FeatureParityStatus(StrEnum):
    PASS = "PASS"
    FAIL_MISMATCH = "FAIL_MISMATCH"
    BLOCKED_NO_OVERLAP = "BLOCKED_NO_OVERLAP"
    BLOCKED_MISSING_SCHEMA = "BLOCKED_MISSING_SCHEMA"
    BLOCKED_UNSAFE_ID_ALIGNMENT = "BLOCKED_UNSAFE_ID_ALIGNMENT"


@dataclass
class FeatureParityResult:
    feature_parity_status: str
    overlap_rows: int = 0
    compared_features: list[str] = field(default_factory=list)
    mismatch_count: int = 0
    max_abs_diff: float | None = None
    mean_abs_diff: float | None = None
    future_inference_readiness: str = "BLOCKED_PENDING_PARITY_SET"
    details: list[dict[str, Any]] = field(default_factory=list)
    reason: str = ""

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "feature_parity_status": self.feature_parity_status,
            "overlap_rows": self.overlap_rows,
            "compared_features": self.compared_features,
            "mismatch_count": self.mismatch_count,
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "future_inference_readiness": self.future_inference_readiness,
            "reason": self.reason,
        }

    def to_csv_rows(self) -> list[dict[str, Any]]:
        if self.details:
            return self.details
        return [
            {
                "feature_parity_status": self.feature_parity_status,
                "overlap_rows": self.overlap_rows,
                "mismatch_count": self.mismatch_count,
                "max_abs_diff": self.max_abs_diff,
                "reason": self.reason,
            }
        ]


def _values_match(runtime_val: Any, offline_val: Any, tol: float = FLOAT_TOLERANCE) -> bool:
    if runtime_val is None and offline_val is None:
        return True
    if runtime_val is None or offline_val is None:
        return False
    try:
        return abs(float(runtime_val) - float(offline_val)) <= tol
    except (TypeError, ValueError):
        return str(runtime_val) == str(offline_val)


def run_feature_parity_check(
    *,
    runtime_bridge_records: list[dict[str, Any]],
    offline_rows_by_exact_id: dict[str, dict[str, Any]] | None = None,
    offline_feature_names: list[str] | None = None,
    id_key: str = "candidate_id",
) -> FeatureParityResult:
    """Compare runtime feature rows to offline rows via exact identity only.

  If no safe overlap exists, returns BLOCKED_NO_OVERLAP without failing AE7B.
    """
    if not runtime_bridge_records:
        return FeatureParityResult(
            feature_parity_status=FeatureParityStatus.BLOCKED_MISSING_SCHEMA.value,
            reason="no_runtime_bridge_records",
            future_inference_readiness="BLOCKED_PENDING_PARITY_SET",
        )

    if not offline_rows_by_exact_id:
        return FeatureParityResult(
            feature_parity_status=FeatureParityStatus.BLOCKED_NO_OVERLAP.value,
            reason="no_safe_exact_id_overlap_between_runtime_and_offline",
            future_inference_readiness="BLOCKED_PENDING_PARITY_SET",
        )

    runtime_by_id = {}
    for rec in runtime_bridge_records:
        cid = rec.get("candidate_id")
        if cid:
            runtime_by_id[str(cid)] = rec

    overlap_ids = [
        rid
        for rid in runtime_by_id
        if rid in offline_rows_by_exact_id
    ][:MAX_PARITY_ROWS]

    if not overlap_ids:
        return FeatureParityResult(
            feature_parity_status=FeatureParityStatus.BLOCKED_NO_OVERLAP.value,
            reason="runtime_candidate_ids_do_not_match_offline_exact_ids",
            future_inference_readiness="BLOCKED_PENDING_PARITY_SET",
        )

    compared_features = offline_feature_names or []
    if not compared_features:
        sample = runtime_bridge_records[0]
        fv = sample.get("feature_values") or {}
        compared_features = sorted(fv.keys())

    details: list[dict[str, Any]] = []
    mismatch_count = 0
    diffs: list[float] = []

    for rid in overlap_ids:
        runtime_rec = runtime_by_id[rid]
        offline_row = offline_rows_by_exact_id[rid]
        runtime_fv = runtime_rec.get("feature_values") or {}
        for feat in compared_features:
            rv = runtime_fv.get(feat)
            ov = offline_row.get(feat)
            if rv is None and ov is None:
                continue
            if not _values_match(rv, ov):
                mismatch_count += 1
                try:
                    diff = abs(float(rv) - float(ov)) if rv is not None and ov is not None else float("inf")
                except (TypeError, ValueError):
                    diff = float("inf")
                if diff != float("inf"):
                    diffs.append(diff)
                details.append(
                    {
                        "candidate_id": rid,
                        "feature": feat,
                        "runtime_value": rv,
                        "offline_value": ov,
                        "match": False,
                    }
                )

    max_diff = max(diffs) if diffs else None
    mean_diff = sum(diffs) / len(diffs) if diffs else None

    if mismatch_count > 0:
        return FeatureParityResult(
            feature_parity_status=FeatureParityStatus.FAIL_MISMATCH.value,
            overlap_rows=len(overlap_ids),
            compared_features=compared_features,
            mismatch_count=mismatch_count,
            max_abs_diff=max_diff,
            mean_abs_diff=mean_diff,
            future_inference_readiness="BLOCKED_PARITY_MISMATCH",
            details=details,
            reason="exact_aligned_feature_values_differ_beyond_tolerance",
        )

    return FeatureParityResult(
        feature_parity_status=FeatureParityStatus.PASS.value,
        overlap_rows=len(overlap_ids),
        compared_features=compared_features,
        mismatch_count=0,
        max_abs_diff=max_diff,
        mean_abs_diff=mean_diff,
        future_inference_readiness="READY_PENDING_AE7C_INFERENCE",
        details=details,
        reason="exact_aligned_features_match",
    )


def write_feature_parity_csv(result: FeatureParityResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = result.to_csv_rows()
    if not rows:
        rows = [{"feature_parity_status": result.feature_parity_status, "reason": result.reason}]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
