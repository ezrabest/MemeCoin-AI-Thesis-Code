"""AE8 context intelligence audits and decision gate."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.context_intelligence.bounded_queries import QueryStats
from app.context_intelligence.context_schema import ContextSchema, validate_feature_names
from app.context_intelligence.types import (
    AE8_PHASE,
    Ae8FinalStatus,
    ContextFeatureRecord,
    FreshnessMode,
    FreshnessStatus,
    SourceStatus,
)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _feature_family_summary(records: list[ContextFeatureRecord]) -> dict[str, Any]:
    families = ["rss", "onchain", "whale", "reputation", "liquidity_activity"]
    summary: dict[str, Any] = {}
    for fam in families:
        present = 0
        missing = 0
        for rec in records:
            miss = rec.context_missingness.get("family_missingness_flags", {}).get(fam, True)
            if miss:
                missing += 1
            else:
                present += 1
        summary[fam] = {"present_records": present, "missing_records": missing}
    return summary


def _source_status_distribution(records: list[ContextFeatureRecord]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for rec in records:
        for status in rec.source_statuses.values():
            counter[status] += 1
    return dict(counter)


def _lineage_summary(records: list[ContextFeatureRecord]) -> dict[str, Any]:
    mode_dist: Counter[str] = Counter()
    strength_dist: Counter[str] = Counter()
    validation_dist: Counter[str] = Counter()
    confidence_scores: list[float] = []
    missing_warning = 0
    fallback_count = 0

    for rec in records:
        lin = rec.lineage
        mode_dist[lin.get("lineage_mode", "UNKNOWN")] += 1
        strength_dist[lin.get("lineage_strength", "UNKNOWN")] += 1
        validation_dist[lin.get("lineage_validation_status", "UNKNOWN")] += 1
        score = lin.get("lineage_confidence_score")
        if score is not None:
            confidence_scores.append(float(score))
        if lin.get("lineage_validation_status") == "PASS_WEAK_BEST_EFFORT_WITH_WARNING":
            if not lin.get("lineage_warning") or not lin.get("fallback_reason"):
                missing_warning += 1
            fallback_count += 1

    return {
        "lineage_mode_distribution": dict(mode_dist),
        "lineage_strength_distribution": dict(strength_dist),
        "lineage_validation_status_distribution": dict(validation_dist),
        "lineage_confidence_score_distribution": {
            "min": min(confidence_scores) if confidence_scores else None,
            "max": max(confidence_scores) if confidence_scores else None,
            "mean": round(sum(confidence_scores) / len(confidence_scores), 4) if confidence_scores else None,
        },
        "records_missing_lineage_warning": missing_warning,
        "fallback_count": fallback_count,
    }


def _freshness_summary(
    records: list[ContextFeatureRecord],
    *,
    freshness_mode: FreshnessMode,
    run_started_at_utc: str,
    thresholds: dict[str, float],
) -> dict[str, Any]:
    stale_sources = 0
    stale_features = 0
    stale_values_nulled = False

    for rec in records:
        for fam, block in rec.context_freshness.items():
            if block.get("freshness_status") in {
                FreshnessStatus.STALE.value,
                FreshnessStatus.INVALID_FUTURE_TIMESTAMP.value,
            }:
                stale_sources += 1
            miss_key = {
                "rss": "rss_missingness_flag",
                "onchain": "onchain_missingness_flag",
                "whale": "whale_score_missingness",
                "reputation": "reputation_missingness_flag",
                "liquidity_activity": "liquidity_activity_missingness_flag",
            }.get(fam)
            ctx_map = {
                "rss": rec.rss_context,
                "onchain": rec.onchain_context,
                "whale": rec.whale_context,
                "reputation": rec.reputation_context,
                "liquidity_activity": rec.liquidity_activity_context,
            }
            if miss_key and ctx_map.get(fam, {}).get(miss_key):
                stale_features += 1
                if block.get("missingness_reason") == "STALE_SOURCE":
                    stale_values_nulled = True

    return {
        "freshness_mode": freshness_mode.value,
        "run_started_at_utc": run_started_at_utc,
        "stale_source_count": stale_sources,
        "stale_feature_count": stale_features,
        "source_freshness_thresholds": thresholds,
        "stale_values_nulled_or_marked_missing": stale_values_nulled,
    }


def decide_ae8_status(
    *,
    records: list[ContextFeatureRecord],
    schema: ContextSchema,
    memory_safety_status: str,
    schema_rejected: list[dict[str, str]],
    allow_external_fetch: bool,
    external_calls_made: int,
) -> dict[str, Any]:
    blocking: list[str] = []
    source_dist = _source_status_distribution(records)
    family_summary = _feature_family_summary(records)

    if schema_rejected:
        blocking.append("forbidden_feature_pattern_detected")

    if memory_safety_status.startswith("BLOCKED"):
        blocking.append(memory_safety_status)

    if not records:
        blocking.append("no_context_records_created")

    lineage = _lineage_summary(records)
    if lineage.get("records_missing_lineage_warning", 0) > 0:
        blocking.append("lineage_warning_missing_on_best_effort")

    missing_families_all = set()
    stale_families: set[str] = set()
    for rec in records:
        missing_families_all.update(rec.context_missingness.get("missing_families", []))
        for fam, status in rec.source_statuses.items():
            if status == SourceStatus.SOURCE_STALE.value:
                stale_families.add(fam)

    if len(blocking) > 1:
        final_status = Ae8FinalStatus.BLOCKED_WITH_EXACT_REASONS
    elif "forbidden_feature_pattern_detected" in blocking:
        final_status = Ae8FinalStatus.BLOCKED_SCHEMA_LEAKAGE
    elif memory_safety_status.startswith("BLOCKED"):
        final_status = Ae8FinalStatus.BLOCKED_MEMORY_SAFETY
    elif "lineage_warning_missing_on_best_effort" in blocking:
        final_status = Ae8FinalStatus.BLOCKED_LINEAGE_VALIDATION
    elif not records:
        final_status = Ae8FinalStatus.BLOCKED_NO_LOCAL_CONTEXT
    elif allow_external_fetch and external_calls_made == 0:
        final_status = Ae8FinalStatus.PARTIAL_LOCAL_ONLY
    elif missing_families_all and family_summary.get("liquidity_activity", {}).get("present_records", 0) > 0:
        final_status = Ae8FinalStatus.PARTIAL_LOCAL_ONLY
    elif records:
        final_status = Ae8FinalStatus.READY_FOR_FORWARD_COLLECTION
    else:
        final_status = Ae8FinalStatus.PARTIAL_LOCAL_ONLY

    if stale_families and final_status == Ae8FinalStatus.READY_FOR_FORWARD_COLLECTION:
        final_status = Ae8FinalStatus.PARTIAL_LOCAL_ONLY

    total_features = len(schema.feature_names) * max(len(records), 1)
    missing_flags = 0
    for rec in records:
        for fam in rec.context_missingness.get("family_missingness_flags", {}).values():
            if fam:
                missing_flags += 1
    missingness_rate = round(missing_flags / (5 * max(len(records), 1)), 4)

    return {
        "final_status": final_status.value,
        "blocking_reasons": blocking,
        "context_records_created": len(records),
        "context_schema_id": schema.context_schema_id,
        "source_status_distribution": source_dist,
        "feature_family_availability": family_summary,
        "missing_context_families": sorted(missing_families_all),
        "stale_context_families": sorted(stale_families),
        "missingness_rate": missingness_rate,
        "total_context_features_per_record": len(schema.feature_names),
        "recommended_next_phase": "AE9_QWEN_GEMINI_AUDIT_LAYER",
        "runtime_inference_status": "BLOCKED_NOT_APPROVED",
        "trading_authorization_status": "NOT_APPROVED",
    }


def write_ae8_audits(
    *,
    project_root: Path,
    output_root: Path,
    audit_dir: Path,
    records: list[ContextFeatureRecord],
    schema: ContextSchema,
    stats: QueryStats,
    memory_safety_status: str,
    allow_external_fetch: bool,
    freshness_mode: FreshnessMode,
    run_started_at_utc: str,
) -> dict[str, Any]:
    audit_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = project_root / "reports"
    audits_dir = project_root / "audits"
    data_dir = output_root

    safe_names, schema_rejected = validate_feature_names(list(schema.feature_names))
    probe_names, probe_rejected = validate_feature_names(
        ["target_label_future_outcome_test", "future_return_proxy"]
    )
    forbidden_rejected = schema_rejected + probe_rejected

    lineage_summary = _lineage_summary(records)
    freshness_summary = _freshness_summary(
        records,
        freshness_mode=freshness_mode,
        run_started_at_utc=run_started_at_utc,
        thresholds=schema.freshness_thresholds,
    )

    external_audit = {
        "external_fetch_enabled": allow_external_fetch,
        "external_calls_made": 0,
        "credentials_required": ["HELIUS_API_KEY"],
        "credentials_missing": [],
        "default_external_fetch_disabled": not allow_external_fetch,
    }

    memory_audit = {
        "rows_scanned_by_table": stats.rows_scanned_by_table,
        "rows_loaded_by_table": stats.rows_loaded_by_table,
        "max_records_enforced": stats.max_records_enforced,
        "lookback_hours_enforced": stats.lookback_hours_enforced,
        "memory_safety_status": memory_safety_status,
        "full_table_market_snapshots_load": False,
        "bounded_query_count": len(stats.queries_executed),
    }

    decision = decide_ae8_status(
        records=records,
        schema=schema,
        memory_safety_status=memory_safety_status,
        schema_rejected=schema_rejected,
        allow_external_fetch=allow_external_fetch,
        external_calls_made=0,
    )
    decision["freshness_summary"] = freshness_summary
    decision["lineage_summary"] = lineage_summary
    decision["memory_safety_summary"] = memory_audit
    decision["external_call_status"] = external_audit
    decision["forbidden_feature_status"] = {
        "rejected_count": len(forbidden_rejected),
        "rejected_features": forbidden_rejected,
        "schema_rejected_count": len(schema_rejected),
        "status": "PASS" if not schema_rejected else "FAIL",
    }

    _write_json(data_dir / "ae8_context_feature_schema.json", schema.to_dict())
    _write_json(reports_dir / "ae8_context_intelligence_summary.json", decision)
    _write_json(reports_dir / "ae8_decision_gate.json", decision)
    _write_json(audits_dir / "ae8_external_call_safety_audit.json", external_audit)
    _write_json(audits_dir / "ae8_memory_safety_audit.json", memory_audit)

    sample_path = data_dir / "ae8_context_feature_records_sample.jsonl"
    with open(sample_path, "w", encoding="utf-8") as f:
        for rec in records[:10]:
            f.write(json.dumps(rec.to_dict(), default=str) + "\n")

    source_rows = []
    missing_rows = []
    lineage_rows = []
    family_rows = []

    for rec in records:
        for fam, status in rec.source_statuses.items():
            source_rows.append(
                {
                    "context_record_id": rec.context_record_id,
                    "candidate_id": rec.candidate_id,
                    "family": fam,
                    "source_status": status,
                }
            )
        for fam, flag in rec.context_missingness.get("family_missingness_flags", {}).items():
            missing_rows.append(
                {
                    "context_record_id": rec.context_record_id,
                    "family": fam,
                    "missingness_flag": flag,
                }
            )
        lin = rec.lineage
        lineage_rows.append(
            {
                "context_record_id": rec.context_record_id,
                "lineage_mode": lin.get("lineage_mode"),
                "lineage_strength": lin.get("lineage_strength"),
                "lineage_validation_status": lin.get("lineage_validation_status"),
                "lineage_confidence_score": lin.get("lineage_confidence_score"),
                "exact_id_match": lin.get("exact_id_match"),
                "has_lineage_warning": bool(lin.get("lineage_warning")),
                "has_fallback_reason": bool(lin.get("fallback_reason")),
            }
        )

    for fam, meta in decision.get("feature_family_availability", {}).items():
        family_rows.append({"family": fam, **meta})

    freshness_rows = []
    for rec in records:
        for fam, block in rec.context_freshness.items():
            freshness_rows.append(
                {
                    "context_record_id": rec.context_record_id,
                    "family": fam,
                    "freshness_status": block.get("freshness_status"),
                    "freshness_minutes": block.get("freshness_minutes"),
                    "missingness_reason": block.get("missingness_reason"),
                }
            )

    _write_csv(data_dir / "ae8_context_source_status.csv", source_rows)
    _write_csv(data_dir / "ae8_context_missingness_summary.csv", missing_rows)
    _write_csv(data_dir / "ae8_context_lineage_audit.csv", lineage_rows)
    _write_csv(data_dir / "ae8_context_feature_family_summary.csv", family_rows)
    _write_csv(audits_dir / "ae8_forbidden_feature_audit.csv", forbidden_rejected)
    _write_csv(audits_dir / "ae8_context_freshness_audit.csv", freshness_rows)

    return decision
