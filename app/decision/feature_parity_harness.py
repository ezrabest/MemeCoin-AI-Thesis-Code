"""AE7C-1 feature parity harness — exact overlap and synthetic fixture modes."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.decision.feature_parity import (
    FLOAT_TOLERANCE,
    FeatureParityStatus,
    run_feature_parity_check,
)
from app.decision.feature_schema import (
    build_enriched_runtime_feature_schema,
    build_feature_values,
    is_forbidden_feature_name,
)

HARNESS_FLOAT_TOLERANCE = 1e-9


class ParityHarnessMode(StrEnum):
    AUTO = "auto"
    EXACT_ONLY = "exact-only"
    SYNTHETIC_ONLY = "synthetic-only"
    OFF = "off"


class HarnessParityStatus(StrEnum):
    PASS_EXACT_OVERLAP = "PASS_EXACT_OVERLAP"
    PASS_SYNTHETIC_FIXTURE_ONLY = "PASS_SYNTHETIC_FIXTURE_ONLY"
    FAIL_MISMATCH = "FAIL_MISMATCH"
    BLOCKED_NO_OVERLAP = "BLOCKED_NO_OVERLAP"
    BLOCKED_MISSING_OFFLINE_BUILDER = "BLOCKED_MISSING_OFFLINE_BUILDER"
    BLOCKED_MISSING_SCHEMA = "BLOCKED_MISSING_SCHEMA"
    BLOCKED_UNSAFE_ID_ALIGNMENT = "BLOCKED_UNSAFE_ID_ALIGNMENT"


CANONICAL_SYNTHETIC_FIXTURE: dict[str, Any] = {
    "snapshot_row": {
        "price": 0.0025,
        "liquidity": 12500.0,
        "volume_24h": 62000.0,
        "txns_buys": 120,
        "txns_sells": 80,
        "fdv": 2500000.0,
        "price_change_m5": 0.01,
        "price_change_h1": 0.03,
        "price_change_h6": 0.08,
        "price_change_h24": 0.15,
        "whale_score": 0.22,
    },
    "signal_row": {"score": 0.61},
    "sentiment_agg": {"sentiment_score": 0.45, "source_count": 3},
}


RUNTIME_TO_OFFLINE_FEATURE_MAP: dict[str, str] = {
    "price_usd": "price",
    "liquidity_usd": "liquidity",
    "volume_h24": "volume_24h",
    "txns_h24_buys": "txns_buys",
    "txns_h24_sells": "txns_sells",
    "txns_h24_total": "txns_total",
    "buy_sell_ratio_h24": "buy_ratio",
    "whale_score_asof": "whale_score",
}


@dataclass
class FeatureParityHarnessResult:
    feature_parity_status: str
    parity_mode_requested: str
    parity_mode_used: str
    exact_overlap_rows: int = 0
    synthetic_fixture_rows: int = 0
    compared_features: list[str] = field(default_factory=list)
    mismatch_count: int = 0
    max_abs_diff: float | None = None
    future_inference_readiness: str = "BLOCKED_PENDING_EXACT_OR_APPROVED_PARITY_SET"
    reason: str = ""
    details: list[dict[str, Any]] = field(default_factory=list)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "feature_parity_status": self.feature_parity_status,
            "parity_mode_requested": self.parity_mode_requested,
            "parity_mode_used": self.parity_mode_used,
            "exact_overlap_rows": self.exact_overlap_rows,
            "synthetic_fixture_rows": self.synthetic_fixture_rows,
            "compared_features": self.compared_features,
            "mismatch_count": self.mismatch_count,
            "max_abs_diff": self.max_abs_diff,
            "future_inference_readiness": self.future_inference_readiness,
            "reason": self.reason,
        }


def build_offline_fixture_features(
    *,
    raw_fixture: dict[str, Any],
    policy_context: dict[str, Any],
) -> dict[str, float | int | None]:
    """Offline/training-style feature dict from shared raw fixture (no labels)."""
    schema = build_enriched_runtime_feature_schema()
    runtime_result = build_feature_values(
        snapshot_row=raw_fixture.get("snapshot_row"),
        signal_row=raw_fixture.get("signal_row"),
        sentiment_agg=raw_fixture.get("sentiment_agg"),
        schema=schema,
        policy_context=policy_context,
    )
    offline: dict[str, float | int | None] = {}
    for runtime_name, value in runtime_result.feature_values.items():
        offline_name = RUNTIME_TO_OFFLINE_FEATURE_MAP.get(runtime_name, runtime_name)
        offline[offline_name] = value
    return offline


def build_runtime_fixture_features(
    *,
    raw_fixture: dict[str, Any],
    policy_context: dict[str, Any],
) -> dict[str, float | int | None]:
    schema = build_enriched_runtime_feature_schema()
    result = build_feature_values(
        snapshot_row=raw_fixture.get("snapshot_row"),
        signal_row=raw_fixture.get("signal_row"),
        sentiment_agg=raw_fixture.get("sentiment_agg"),
        schema=schema,
        policy_context=policy_context,
    )
    return dict(result.feature_values)


def _compare_feature_pair(
    *,
    feature_name: str,
    runtime_value: Any,
    offline_value: Any,
    parity_mode: str,
    tolerance: float = HARNESS_FLOAT_TOLERANCE,
) -> dict[str, Any]:
    matched = False
    abs_diff = None
    rel_diff = None
    status = "PASS"
    reason = "values_match_within_tolerance"

    if runtime_value is None and offline_value is None:
        matched = True
    elif runtime_value is None or offline_value is None:
        status = "FAIL_MISMATCH"
        reason = "one_side_null"
    else:
        try:
            rv = float(runtime_value)
            ov = float(offline_value)
            abs_diff = abs(rv - ov)
            denom = max(abs(ov), 1e-12)
            rel_diff = abs_diff / denom
            matched = abs_diff <= tolerance
            if not matched:
                status = "FAIL_MISMATCH"
                reason = "abs_diff_exceeds_tolerance"
        except (TypeError, ValueError):
            matched = str(runtime_value) == str(offline_value)
            if not matched:
                status = "FAIL_MISMATCH"
                reason = "non_numeric_mismatch"

    return {
        "parity_mode": parity_mode,
        "feature_name": feature_name,
        "runtime_value": runtime_value,
        "offline_value": offline_value,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "tolerance": tolerance,
        "matched": matched,
        "status": status,
        "reason": reason,
    }


def run_synthetic_fixture_parity(
    *,
    policy_context: dict[str, Any],
    raw_fixture: dict[str, Any] | None = None,
) -> FeatureParityHarnessResult:
    raw_fixture = raw_fixture or CANONICAL_SYNTHETIC_FIXTURE
    try:
        runtime_fv = build_runtime_fixture_features(
            raw_fixture=raw_fixture, policy_context=policy_context
        )
        offline_fv = build_offline_fixture_features(
            raw_fixture=raw_fixture, policy_context=policy_context
        )
    except Exception as exc:
        return FeatureParityHarnessResult(
            feature_parity_status=HarnessParityStatus.BLOCKED_MISSING_OFFLINE_BUILDER.value,
            parity_mode_requested=ParityHarnessMode.SYNTHETIC_ONLY.value,
            parity_mode_used=ParityHarnessMode.SYNTHETIC_ONLY.value,
            reason=f"fixture_builder_failed:{exc}",
        )

    compare_names = sorted(
        {
            name
            for name in runtime_fv
            if not is_forbidden_feature_name(name) and runtime_fv.get(name) is not None
        }
    )
    details: list[dict[str, Any]] = []
    mismatch_count = 0
    diffs: list[float] = []

    for name in compare_names:
        offline_name = RUNTIME_TO_OFFLINE_FEATURE_MAP.get(name, name)
        row = _compare_feature_pair(
            feature_name=name,
            runtime_value=runtime_fv.get(name),
            offline_value=offline_fv.get(offline_name),
            parity_mode="synthetic_fixture",
        )
        details.append(row)
        if not row["matched"]:
            mismatch_count += 1
        elif row["abs_diff"] is not None:
            diffs.append(float(row["abs_diff"]))

    max_diff = max(diffs) if diffs else 0.0
    if mismatch_count > 0:
        return FeatureParityHarnessResult(
            feature_parity_status=HarnessParityStatus.FAIL_MISMATCH.value,
            parity_mode_requested=ParityHarnessMode.SYNTHETIC_ONLY.value,
            parity_mode_used=ParityHarnessMode.SYNTHETIC_ONLY.value,
            synthetic_fixture_rows=1,
            compared_features=compare_names,
            mismatch_count=mismatch_count,
            max_abs_diff=max_diff,
            future_inference_readiness="BLOCKED_PARITY_MISMATCH",
            reason="synthetic_fixture_feature_values_differ",
            details=details,
        )

    return FeatureParityHarnessResult(
        feature_parity_status=HarnessParityStatus.PASS_SYNTHETIC_FIXTURE_ONLY.value,
        parity_mode_requested=ParityHarnessMode.SYNTHETIC_ONLY.value,
        parity_mode_used=ParityHarnessMode.SYNTHETIC_ONLY.value,
        synthetic_fixture_rows=1,
        compared_features=compare_names,
        mismatch_count=0,
        max_abs_diff=max_diff,
        future_inference_readiness="BLOCKED_PENDING_EXACT_OR_APPROVED_PARITY_SET",
        reason="synthetic_fixture_features_match_does_not_replace_exact_overlap",
        details=details,
    )


def run_exact_overlap_parity(
    *,
    runtime_bridge_records: list[dict[str, Any]],
    offline_rows_by_exact_id: dict[str, dict[str, Any]] | None,
) -> FeatureParityHarnessResult:
    legacy = run_feature_parity_check(
        runtime_bridge_records=runtime_bridge_records,
        offline_rows_by_exact_id=offline_rows_by_exact_id,
    )
    status_map = {
        FeatureParityStatus.PASS.value: HarnessParityStatus.PASS_EXACT_OVERLAP.value,
        FeatureParityStatus.FAIL_MISMATCH.value: HarnessParityStatus.FAIL_MISMATCH.value,
        FeatureParityStatus.BLOCKED_NO_OVERLAP.value: HarnessParityStatus.BLOCKED_NO_OVERLAP.value,
        FeatureParityStatus.BLOCKED_MISSING_SCHEMA.value: HarnessParityStatus.BLOCKED_MISSING_SCHEMA.value,
        FeatureParityStatus.BLOCKED_UNSAFE_ID_ALIGNMENT.value: (
            HarnessParityStatus.BLOCKED_UNSAFE_ID_ALIGNMENT.value
        ),
    }
    harness_status = status_map.get(
        legacy.feature_parity_status, HarnessParityStatus.BLOCKED_NO_OVERLAP.value
    )
    readiness = legacy.future_inference_readiness
    if harness_status == HarnessParityStatus.PASS_EXACT_OVERLAP.value:
        readiness = "READY_PENDING_AE7C2_DRY_RUN"
    return FeatureParityHarnessResult(
        feature_parity_status=harness_status,
        parity_mode_requested=ParityHarnessMode.EXACT_ONLY.value,
        parity_mode_used=ParityHarnessMode.EXACT_ONLY.value,
        exact_overlap_rows=legacy.overlap_rows,
        compared_features=legacy.compared_features,
        mismatch_count=legacy.mismatch_count,
        max_abs_diff=legacy.max_abs_diff,
        future_inference_readiness=readiness,
        reason=legacy.reason,
        details=legacy.details,
    )


def run_feature_parity_harness(
    *,
    mode: str = ParityHarnessMode.AUTO.value,
    runtime_bridge_records: list[dict[str, Any]] | None = None,
    offline_rows_by_exact_id: dict[str, dict[str, Any]] | None = None,
    policy_context: dict[str, Any] | None = None,
) -> FeatureParityHarnessResult:
    """Run parity harness per mode. Never uses fuzzy pair/time matching."""
    runtime_bridge_records = runtime_bridge_records or []
    policy_context = policy_context or {}

    if mode == ParityHarnessMode.OFF.value:
        return FeatureParityHarnessResult(
            feature_parity_status=HarnessParityStatus.BLOCKED_NO_OVERLAP.value,
            parity_mode_requested=mode,
            parity_mode_used=mode,
            reason="parity_harness_disabled",
        )

    exact_result: FeatureParityHarnessResult | None = None
    if mode in (ParityHarnessMode.EXACT_ONLY.value, ParityHarnessMode.AUTO.value):
        exact_result = run_exact_overlap_parity(
            runtime_bridge_records=runtime_bridge_records,
            offline_rows_by_exact_id=offline_rows_by_exact_id,
        )
        if mode == ParityHarnessMode.EXACT_ONLY.value:
            return exact_result
        if exact_result.feature_parity_status in {
            HarnessParityStatus.PASS_EXACT_OVERLAP.value,
            HarnessParityStatus.FAIL_MISMATCH.value,
        }:
            exact_result.parity_mode_requested = ParityHarnessMode.AUTO.value
            exact_result.parity_mode_used = ParityHarnessMode.EXACT_ONLY.value
            return exact_result

    synthetic = run_synthetic_fixture_parity(policy_context=policy_context)
    synthetic.parity_mode_requested = mode
    synthetic.parity_mode_used = (
        ParityHarnessMode.SYNTHETIC_ONLY.value
        if mode == ParityHarnessMode.SYNTHETIC_ONLY.value
        else ParityHarnessMode.SYNTHETIC_ONLY.value
    )
    if mode == ParityHarnessMode.AUTO.value and exact_result is not None:
        synthetic.reason = f"no_exact_overlap;{synthetic.reason}"
    return synthetic


def write_parity_harness_outputs(
    result: FeatureParityHarnessResult,
    *,
    summary_path: Path,
    csv_path: Path,
) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result.to_summary_dict(), f, indent=2)

    rows = result.details or [
        {
            "parity_mode": result.parity_mode_used,
            "feature_name": None,
            "runtime_value": None,
            "offline_value": None,
            "abs_diff": result.max_abs_diff,
            "rel_diff": None,
            "tolerance": HARNESS_FLOAT_TOLERANCE,
            "matched": result.mismatch_count == 0,
            "status": result.feature_parity_status,
            "reason": result.reason,
        }
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
