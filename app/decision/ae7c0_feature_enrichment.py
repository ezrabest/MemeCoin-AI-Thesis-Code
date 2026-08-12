"""AE7C-0 scoring policy feature enrichment + schema compatibility repair."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.decision.bridge_persistence import (
    RuntimeBridgeJsonlWriter,
    ae7c0_runtime_bridge_jsonl_path_for_date,
)
from app.decision.feature_parity import FeatureParityStatus, run_feature_parity_check
from app.decision.feature_schema import (
    AE7B_FEATURE_SCHEMA_VERSION,
    AE7C0_FEATURE_SCHEMA_VERSION,
    build_enriched_runtime_feature_schema,
    build_runtime_feature_schema,
)
from app.decision.runtime_feature_bridge import (
    DEFAULT_PREFLIGHT_SCHEMA_CSV,
    build_model_schema_compatibility_matrix,
    fetch_recent_signal_bundles,
    write_compatibility_matrix,
)
from app.decision.runtime_identity import default_scoring_policy
from app.decision.scoring_policy_features import resolve_scoring_policy_context

AE7C0_PHASE = "AE7C0_SCORING_POLICY_FEATURE_ENRICHMENT"

AE7B_TYPICAL_MISSING_FEATURES = (
    "round_trip_fee_pct",
    "sl_ratio",
    "time_stop_minutes",
    "tp_ratio",
    "volume_to_liquidity_ratio",
)


def build_enriched_runtime_bridge_record(
    bundle: dict[str, Any],
    *,
    schema: Any,
    policy_context: dict[str, Any],
    scoring_policy: Any | None = None,
) -> dict[str, Any]:
    """Build AE7C-0 enriched bridge record."""
    from app.decision.runtime_feature_bridge import build_runtime_bridge_record

    record = build_runtime_bridge_record(
        bundle,
        schema=schema,
        scoring_policy=scoring_policy,
        policy_context=policy_context,
        phase=AE7C0_PHASE,
    )
    record["scoring_policy_context"] = {
        "policy_feature_source": policy_context.get("policy_feature_source"),
        "policy_feature_status": policy_context.get("policy_feature_status"),
        "exit_policy_id": policy_context.get("exit_policy_id"),
        "horizon": policy_context.get("horizon"),
    }
    return record


def _load_schema_candidate_paths(project_root: Path, schema_csv: Path | None, limit: int = 30) -> list[Path]:
    schema_csv = schema_csv or (project_root / DEFAULT_PREFLIGHT_SCHEMA_CSV)
    candidate_paths: list[Path] = []
    if not schema_csv.is_file():
        return candidate_paths
    with open(schema_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get("path", "")
            if p and p.endswith(".json") and ("schema" in p.lower() or "feature" in p.lower()):
                candidate_paths.append(Path(p))
            if len(candidate_paths) >= limit:
                break
    return candidate_paths


def _compatibility_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(r.get("compatibility_status") for r in rows))


def _aggregate_missing_features(rows: list[dict[str, Any]]) -> set[str]:
    missing: set[str] = set()
    for row in rows:
        sample = row.get("missing_features_sample") or ""
        for part in sample.split("|"):
            if part:
                missing.add(part)
    return missing


def summarize_compatibility_delta(
    *,
    before_rows: list[dict[str, Any]],
    after_rows: list[dict[str, Any]],
    parity_status: str,
) -> dict[str, Any]:
    before_counts = _compatibility_counts(before_rows)
    after_counts = _compatibility_counts(after_rows)
    before_missing = _aggregate_missing_features(before_rows)
    after_missing = _aggregate_missing_features(after_rows)
    resolved = sorted(before_missing - after_missing)
    remaining = sorted(after_missing)

    after_safe = sum(1 for r in after_rows if r.get("safe_for_future_inference"))

    return {
        "compatibility_before_enrichment": before_counts,
        "compatibility_after_enrichment": after_counts,
        "compatible_before": before_counts.get("COMPATIBLE", 0),
        "compatible_after": after_counts.get("COMPATIBLE", 0),
        "partial_missing_before": before_counts.get("PARTIAL_MISSING_FEATURES", 0),
        "partial_missing_after": after_counts.get("PARTIAL_MISSING_FEATURES", 0),
        "missing_features_resolved_count": len(resolved),
        "missing_features_remaining_count": len(remaining),
        "resolved_features": resolved,
        "remaining_missing_features_sample": remaining[:15],
        "safe_for_future_inference": after_safe > 0 and parity_status == FeatureParityStatus.PASS.value,
        "safe_for_future_inference_note": (
            "false_when_parity_blocked_or_weak_lineage_or_missing_model_features"
        ),
    }


def write_feature_missingness_csv(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for rec in records:
        cid = rec.get("candidate_id")
        reasons = rec.get("feature_missingness_reasons") or {}
        for feat in rec.get("feature_missingness") or []:
            rows.append(
                {
                    "candidate_id": cid,
                    "feature": feat,
                    "missing_reason": reasons.get(feat, "MISSING_SOURCE_FEATURE"),
                }
            )
    if not rows:
        rows = [{"candidate_id": None, "feature": None, "missing_reason": "no_missing_features"}]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_ae7c0_enrichment(
    *,
    project_root: Path,
    conn: sqlite3.Connection,
    max_records: int = 50,
    lookback_hours: float = 24.0,
    output_root: Path | None = None,
    audit_only: bool = True,
    schema_candidates_csv: Path | None = None,
    timestamp_slug: str | None = None,
    settings_path: Path | None = None,
) -> dict[str, Any]:
    from scripts.diagnostics._common import timestamp_slug as default_slug

    slug = timestamp_slug or default_slug()
    output_root = output_root or (project_root / "data" / "audits")
    audit_dir = output_root / f"ae7c0_scoring_policy_feature_enrichment_{slug}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    schema_before = build_runtime_feature_schema(enriched=False)
    schema_after = build_enriched_runtime_feature_schema()
    settings_path = settings_path or (project_root / "data" / "settings.json")
    policy_context = resolve_scoring_policy_context(settings_path=settings_path)
    scoring_policy = default_scoring_policy()

    bundles = fetch_recent_signal_bundles(
        conn, limit=max_records, lookback_hours=lookback_hours
    )

    records: list[dict[str, Any]] = []
    jsonl_path: str | None = None

    if audit_only:
        for bundle in bundles:
            records.append(
                build_enriched_runtime_bridge_record(
                    bundle,
                    schema=schema_after,
                    policy_context=policy_context,
                    scoring_policy=scoring_policy,
                )
            )
    else:
        path = ae7c0_runtime_bridge_jsonl_path_for_date()
        jsonl_path = str(path)
        with RuntimeBridgeJsonlWriter(path=path) as writer:
            for bundle in bundles:
                rec = build_enriched_runtime_bridge_record(
                    bundle,
                    schema=schema_after,
                    policy_context=policy_context,
                    scoring_policy=scoring_policy,
                )
                writer.append_record(rec)
                records.append(rec)

    parity_result = run_feature_parity_check(runtime_bridge_records=records)
    candidate_paths = _load_schema_candidate_paths(project_root, schema_candidates_csv)

    before_rows = build_model_schema_compatibility_matrix(
        runtime_schema=schema_before,
        schema_candidate_paths=candidate_paths,
        project_root=project_root,
        max_schemas=30,
        parity_status=parity_result.feature_parity_status,
        weak_lineage=True,
    )
    after_rows = build_model_schema_compatibility_matrix(
        runtime_schema=schema_after,
        schema_candidate_paths=candidate_paths,
        project_root=project_root,
        max_schemas=30,
        parity_status=parity_result.feature_parity_status,
        weak_lineage=True,
    )

    compat_delta = summarize_compatibility_delta(
        before_rows=before_rows,
        after_rows=after_rows,
        parity_status=parity_result.feature_parity_status,
    )

    after_compat_path = audit_dir / "ae7c0_model_schema_compatibility_after_enrichment.csv"
    write_compatibility_matrix(after_rows, after_compat_path)

    schema_json_path = audit_dir / "ae7c0_runtime_feature_schema_after_enrichment.json"
    with open(schema_json_path, "w", encoding="utf-8") as f:
        json.dump(schema_after.to_dict(), f, indent=2)

    missingness_path = audit_dir / "ae7c0_feature_missingness_after_enrichment.csv"
    write_feature_missingness_csv(records, missingness_path)

    lineage_modes = Counter((r.get("lineage") or {}).get("lineage_mode") for r in records)
    lineage_strengths = Counter((r.get("lineage") or {}).get("lineage_strength") for r in records)
    confidence_scores = [
        (r.get("lineage") or {}).get("lineage_confidence_score") for r in records
    ]

    policy_values = policy_context.get("values") or {}
    summary = {
        "phase": AE7C0_PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bridge_records_created": len(records),
        "feature_schema_id_before": schema_before.feature_schema_id,
        "feature_schema_id_after": schema_after.feature_schema_id,
        "feature_schema_version_before": AE7B_FEATURE_SCHEMA_VERSION,
        "feature_schema_version_after": AE7C0_FEATURE_SCHEMA_VERSION,
        "feature_count_before": len(schema_before.feature_names),
        "feature_count_after": len(schema_after.feature_names),
        "policy_features_added": list(policy_values.keys()),
        "derived_features_added": [
            "volume_to_liquidity_ratio",
            "txns_h24_total",
            "buy_sell_ratio_h24",
        ],
        "required_feature_count": len(schema_after.required_features),
        "optional_feature_count": len(schema_after.optional_features),
        "rejected_feature_count": len(schema_after.rejected_features),
        "missing_required_feature_count": sum(
            len(r.get("missing_required_features") or []) for r in records
        ),
        "missing_optional_feature_count": sum(
            len(r.get("feature_missingness") or []) for r in records
        ),
        "policy_feature_source": {
            "tp_ratio": policy_context.get("policy_feature_source"),
            "sl_ratio": policy_context.get("policy_feature_source"),
            "time_stop_minutes": policy_context.get("policy_feature_source"),
            "round_trip_fee_pct": policy_context.get("policy_feature_source"),
            "policy_feature_status": policy_context.get("policy_feature_status"),
            "exit_policy_id": policy_context.get("exit_policy_id"),
            "horizon": policy_context.get("horizon"),
            "placeholder_values": policy_values,
        },
        "feature_parity": parity_result.to_summary_dict(),
        "lineage_mode_distribution": dict(lineage_modes),
        "lineage_strength_distribution": dict(lineage_strengths),
        "lineage_confidence_score_min": min(confidence_scores) if confidence_scores else None,
        "lineage_confidence_score_max": max(confidence_scores) if confidence_scores else None,
        "lineage_confidence_score_mean": (
            round(sum(confidence_scores) / len(confidence_scores), 4) if confidence_scores else None
        ),
        **compat_delta,
        "audit_dir": str(audit_dir),
        "jsonl_path": jsonl_path,
        "audit_only": audit_only,
        "sqlite_runtime_feature_rows_written": False,
        "target_row_id_not_required_at_runtime": True,
        "used_for_inference": False,
        "safety": {
            "no_model_training": True,
            "no_model_inference": True,
            "no_llm_calls": True,
            "no_external_api_calls": True,
            "no_live_trading": True,
            "no_paper_trade_execution": True,
            "no_sqlite_runtime_feature_row_writes": True,
        },
        "output_paths": {
            "summary": str(audit_dir / "ae7c0_feature_enrichment_summary.json"),
            "compatibility_after": str(after_compat_path),
            "schema_after": str(schema_json_path),
            "missingness_after": str(missingness_path),
        },
    }

    summary_path = audit_dir / "ae7c0_feature_enrichment_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary
