"""AE7C-1 scoring policy binding + parity harness + inference gate orchestration."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.decision.bridge_persistence import RuntimeBridgeJsonlWriter
from app.decision.feature_parity_harness import (
    HarnessParityStatus,
    ParityHarnessMode,
    run_feature_parity_harness,
    write_parity_harness_outputs,
)
from app.decision.feature_schema import build_enriched_runtime_feature_schema, is_forbidden_feature_name
from app.decision.inference_readiness_gate import (
    attempt_local_inference_if_allowed,
    evaluate_inference_readiness_gate,
    write_inference_gate_json,
)
from app.decision.model_artifact_readiness import (
    build_model_artifact_readiness_rows,
    summarize_artifact_readiness,
    write_model_artifact_readiness_csv,
)
from app.decision.runtime_feature_bridge import (
    DEFAULT_PREFLIGHT_SCHEMA_CSV,
    build_model_schema_compatibility_matrix,
    build_runtime_bridge_record,
    fetch_recent_signal_bundles,
)
from app.decision.runtime_identity import default_scoring_policy
from app.decision.scoring_policy_binding import (
    ScoringPolicyBindingStatus,
    bind_scoring_policy,
    load_settings_dict,
)

AE7C1_PHASE = "AE7C1_SCORING_POLICY_BINDING_PARITY_GATE"


def ae7c1_runtime_bridge_jsonl_path_for_date(dt: datetime | None = None) -> Path:
    dt = dt or datetime.now(timezone.utc)
    day = dt.strftime("%Y%m%d")
    return Path(__file__).parent.parent.parent / "data" / "runtime_bridge" / f"ae7c1_runtime_feature_rows_{day}.jsonl"


def _load_schema_candidate_paths(project_root: Path, limit: int = 30) -> list[Path]:
    schema_csv = project_root / DEFAULT_PREFLIGHT_SCHEMA_CSV
    paths: list[Path] = []
    if not schema_csv.is_file():
        return paths
    with open(schema_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get("path", "")
            if p and p.endswith(".json") and ("schema" in p.lower() or "feature" in p.lower()):
                paths.append(Path(p))
            if len(paths) >= limit:
                break
    return paths


def _binding_to_policy_context(binding: Any) -> dict[str, Any]:
    return binding.policy_context or {
        "values": binding.policy_features,
        "policy_feature_source": binding.scoring_policy_source,
        "exit_policy_id": binding.exit_policy,
        "horizon": binding.horizon,
    }


def build_ae7c1_bridge_record(
    bundle: dict[str, Any],
    *,
    schema: Any,
    binding: Any,
    scoring_policy: Any | None = None,
) -> dict[str, Any]:
    policy_context = _binding_to_policy_context(binding)
    record = build_runtime_bridge_record(
        bundle,
        schema=schema,
        scoring_policy=scoring_policy,
        policy_context=policy_context,
        phase=AE7C1_PHASE,
    )
    record["scoring_policy_binding"] = binding.to_dict()
    record["scoring_policy_id"] = binding.scoring_policy_id
    return record


def write_binding_audit_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"scoring_policy_binding_status": "NONE"}]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_ae7c1_binding_parity_gate(
    *,
    project_root: Path,
    conn: sqlite3.Connection,
    max_records: int = 50,
    lookback_hours: float = 24.0,
    output_root: Path | None = None,
    audit_only: bool = True,
    parity_mode: str = ParityHarnessMode.AUTO.value,
    allow_local_inference_if_gates_pass: bool = False,
    timestamp_slug: str | None = None,
) -> dict[str, Any]:
    from scripts.diagnostics._common import timestamp_slug as default_slug

    slug = timestamp_slug or default_slug()
    output_root = output_root or (project_root / "data" / "audits")
    audit_dir = output_root / f"ae7c1_scoring_policy_binding_parity_gate_{slug}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    settings_path = project_root / "data" / "settings.json"
    settings = load_settings_dict(settings_path)
    schema = build_enriched_runtime_feature_schema()
    scoring_policy = default_scoring_policy()
    bundles = fetch_recent_signal_bundles(
        conn, limit=max_records, lookback_hours=lookback_hours
    )

    records: list[dict[str, Any]] = []
    binding_rows: list[dict[str, Any]] = []
    jsonl_path: str | None = None

    global_binding = bind_scoring_policy(settings=settings, signal_row=None)

    for bundle in bundles:
        signal_row = bundle.get("signal_row") or {}
        binding = bind_scoring_policy(settings=settings, signal_row=signal_row)
        rec = build_ae7c1_bridge_record(
            bundle,
            schema=schema,
            binding=binding,
            scoring_policy=scoring_policy,
        )
        records.append(rec)
        binding_rows.append(
            binding.to_audit_row(
                candidate_id=rec.get("candidate_id"),
                signal_id=signal_row.get("id"),
            )
        )

    if not audit_only:
        path = ae7c1_runtime_bridge_jsonl_path_for_date()
        jsonl_path = str(path)
        with RuntimeBridgeJsonlWriter(path=path) as writer:
            for rec in records:
                writer.append_record(rec)

    policy_context = _binding_to_policy_context(global_binding)
    parity_result = run_feature_parity_harness(
        mode=parity_mode,
        runtime_bridge_records=records,
        offline_rows_by_exact_id=None,
        policy_context=policy_context,
    )

    parity_summary_path = audit_dir / "ae7c1_feature_parity_harness_summary.json"
    parity_csv_path = audit_dir / "ae7c1_feature_parity_harness_results.csv"
    write_parity_harness_outputs(
        parity_result,
        summary_path=parity_summary_path,
        csv_path=parity_csv_path,
    )

    candidate_paths = _load_schema_candidate_paths(project_root)
    compat_rows = build_model_schema_compatibility_matrix(
        runtime_schema=schema,
        schema_candidate_paths=candidate_paths,
        project_root=project_root,
        max_schemas=30,
        parity_status=parity_result.feature_parity_status,
        weak_lineage=True,
    )
    compatible_count = sum(1 for r in compat_rows if r.get("compatibility_status") == "COMPATIBLE")
    schema_status = "PASS" if compatible_count > 0 else "BLOCKED"

    binding_statuses = Counter(r.get("scoring_policy_binding_status") for r in binding_rows)
    missing_required = sum(len(r.get("missing_required_features") or []) for r in records)
    rejected_forbidden = [
        f for f in schema.feature_names if is_forbidden_feature_name(f)
    ]
    forbidden_check = "PASS" if not rejected_forbidden else "FAIL"

    lineage_scores = [
        (r.get("lineage") or {}).get("lineage_confidence_score") for r in records
    ]
    exact_matches = [(r.get("lineage") or {}).get("exact_id_match") for r in records]
    weak_lineage = not all(exact_matches) if exact_matches else True

    policy_binding_pass = any(
        binding_statuses.get(s, 0) > 0
        for s in (
            ScoringPolicyBindingStatus.PASS_CONFIG_BOUND.value,
            ScoringPolicyBindingStatus.PASS_SIGNAL_CONTEXT_BOUND.value,
        )
    )
    parity_exact_pass = (
        parity_result.feature_parity_status == HarnessParityStatus.PASS_EXACT_OVERLAP.value
    )

    artifact_rows = build_model_artifact_readiness_rows(
        runtime_schema=schema,
        schema_candidate_paths=candidate_paths,
        project_root=project_root,
        max_schemas=30,
        parity_exact_pass=parity_exact_pass,
        policy_binding_pass=policy_binding_pass,
    )
    artifact_summary = summarize_artifact_readiness(artifact_rows)
    artifact_path = audit_dir / "ae7c1_model_artifact_readiness.csv"
    write_model_artifact_readiness_csv(artifact_rows, artifact_path)

    reproducible_count = sum(1 for r in artifact_rows if r.get("is_reproducible"))
    model_artifacts_reproducible = reproducible_count > 0 and compatible_count > 0

    dominant_binding = (
        binding_statuses.most_common(1)[0][0] if binding_statuses else global_binding.scoring_policy_binding_status
    )

    gate_result = evaluate_inference_readiness_gate(
        schema_compatibility_status=schema_status,
        feature_parity_status=parity_result.feature_parity_status,
        scoring_policy_binding_status=dominant_binding,
        missing_required_features_count=missing_required,
        forbidden_feature_check=forbidden_check,
        lineage_confidence_score=min(lineage_scores) if lineage_scores else None,
        exact_id_match=not weak_lineage,
        model_artifacts_reproducible=model_artifacts_reproducible,
        target_row_id_required=False,
        external_calls_required=False,
    )

    gate_path = audit_dir / "ae7c1_inference_readiness_gate.json"
    write_inference_gate_json(gate_result, gate_path)

    inference_attempt = attempt_local_inference_if_allowed(
        gate_result=gate_result,
        allow_local_inference_if_gates_pass=allow_local_inference_if_gates_pass,
    )

    binding_audit_path = audit_dir / "ae7c1_scoring_policy_binding_audit.csv"
    write_binding_audit_csv(binding_rows, binding_audit_path)

    summary = {
        "phase": AE7C1_PHASE,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rows_inspected": len(records),
        "scoring_policy_binding_distribution": dict(binding_statuses),
        "dominant_scoring_policy_binding_status": dominant_binding,
        "global_policy_source": global_binding.scoring_policy_source,
        "global_policy_features": global_binding.policy_features,
        "feature_parity": parity_result.to_summary_dict(),
        "model_artifact_readiness": artifact_summary,
        "schema_compatible_artifact_count": compatible_count,
        "inference_readiness_gate": gate_result.to_dict(),
        "inference_attempt": inference_attempt,
        "lineage_mode_distribution": dict(
            Counter((r.get("lineage") or {}).get("lineage_mode") for r in records)
        ),
        "lineage_strength_distribution": dict(
            Counter((r.get("lineage") or {}).get("lineage_strength") for r in records)
        ),
        "lineage_confidence_score_min": min(lineage_scores) if lineage_scores else None,
        "lineage_confidence_score_max": max(lineage_scores) if lineage_scores else None,
        "lineage_confidence_score_mean": (
            round(sum(lineage_scores) / len(lineage_scores), 4) if lineage_scores else None
        ),
        "audit_dir": str(audit_dir),
        "jsonl_path": jsonl_path,
        "audit_only": audit_only,
        "sqlite_runtime_feature_rows_written": False,
        "target_row_id_not_required_at_runtime": True,
        "safety": {
            "no_model_training": True,
            "no_model_inference_by_default": True,
            "no_llm_calls": True,
            "no_external_api_calls": True,
            "no_live_trading": True,
            "no_paper_trade_execution": True,
            "no_sqlite_runtime_feature_row_writes": True,
        },
        "output_paths": {
            "summary": str(audit_dir / "ae7c1_summary.json"),
            "binding_audit": str(binding_audit_path),
            "parity_summary": str(parity_summary_path),
            "parity_results": str(parity_csv_path),
            "inference_gate": str(gate_path),
            "artifact_readiness": str(artifact_path),
        },
    }

    summary_path = audit_dir / "ae7c1_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    return summary
