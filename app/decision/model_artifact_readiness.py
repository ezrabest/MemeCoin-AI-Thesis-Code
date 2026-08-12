"""AE7C-1 model artifact readiness (schema/metadata only, no inference)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from app.decision.feature_schema import RuntimeFeatureSchema, infer_model_family_from_schema_path
from app.decision.runtime_feature_bridge import (
    MAX_SCHEMA_FILE_BYTES,
    load_model_schema_from_json,
)


def build_model_artifact_readiness_rows(
    *,
    runtime_schema: RuntimeFeatureSchema,
    schema_candidate_paths: list[Path],
    project_root: Path,
    max_schemas: int = 30,
    parity_exact_pass: bool = False,
    policy_binding_pass: bool = False,
) -> list[dict[str, Any]]:
    """Inspect model schema artifacts for future dry-run readiness."""
    runtime_features = set(runtime_schema.feature_names)
    runtime_alias = {
        "liquidity_usd": "liquidity",
        "price_usd": "price",
        "volume_h24": "volume_24h",
        "buy_sell_ratio_h24": "buy_ratio",
        "whale_score_asof": "whale_score",
        "txns_h24_buys": "txns_buys",
        "txns_h24_sells": "txns_sells",
        "txns_h24_total": "txns_total",
    }
    rows: list[dict[str, Any]] = []
    inspected = 0

    for rel_path in schema_candidate_paths:
        if inspected >= max_schemas:
            break
        path = project_root / str(rel_path).replace("\\", "/")
        family = infer_model_family_from_schema_path(str(path))
        if family not in {"RF", "XGB", "TAB"} and "schema" not in str(rel_path).lower():
            continue
        if not path.is_file() or path.suffix.lower() != ".json":
            continue
        if "schema" not in path.name.lower() and "feature" not in path.name.lower():
            continue

        model_artifact_path = ""
        if path.parent.name == "models":
            model_artifact_path = str(path).replace("_schema.json", ".pkl")

        if path.stat().st_size > MAX_SCHEMA_FILE_BYTES:
            rows.append(
                _row(
                    family=family,
                    model_artifact_path=model_artifact_path,
                    schema_source_path=str(rel_path),
                    artifact_status="TOO_LARGE",
                    is_reproducible=False,
                    required_feature_count=0,
                    runtime_available_feature_count=len(runtime_features),
                    compatibility_status="BLOCKED_UNSUPPORTED_ARTIFACT",
                    ready_for_dry_run=False,
                    reason="schema_file_too_large",
                )
            )
            inspected += 1
            continue

        try:
            model_features, schema_kind = load_model_schema_from_json(path)
        except (OSError, ValueError):
            rows.append(
                _row(
                    family=family,
                    model_artifact_path=model_artifact_path,
                    schema_source_path=str(rel_path),
                    artifact_status="UNREADABLE",
                    is_reproducible=False,
                    required_feature_count=0,
                    runtime_available_feature_count=0,
                    compatibility_status="UNKNOWN",
                    ready_for_dry_run=False,
                    reason="schema_parse_failed",
                )
            )
            inspected += 1
            continue

        if not model_features:
            rows.append(
                _row(
                    family=family,
                    model_artifact_path=model_artifact_path,
                    schema_source_path=str(rel_path),
                    artifact_status=schema_kind,
                    is_reproducible=False,
                    required_feature_count=0,
                    runtime_available_feature_count=0,
                    compatibility_status="BLOCKED_MISSING_SCHEMA",
                    ready_for_dry_run=False,
                    reason="no_feature_columns",
                )
            )
            inspected += 1
            continue

        model_set = set(model_features)
        runtime_mapped: set[str] = set()
        for rf in runtime_features:
            runtime_mapped.add(rf)
            if rf in runtime_alias:
                runtime_mapped.add(runtime_alias[rf])
        missing = sorted(model_set - runtime_mapped)
        overlap = model_set & runtime_mapped
        compat = "COMPATIBLE" if not missing else "PARTIAL_MISSING_FEATURES"
        is_reproducible = bool(model_features) and path.is_file()

        ready = (
            compat == "COMPATIBLE"
            and is_reproducible
            and parity_exact_pass
            and policy_binding_pass
        )

        rows.append(
            _row(
                family=family,
                model_artifact_path=model_artifact_path,
                schema_source_path=str(rel_path),
                artifact_status=schema_kind,
                is_reproducible=is_reproducible,
                required_feature_count=len(model_set),
                runtime_available_feature_count=len(overlap),
                compatibility_status=compat,
                ready_for_dry_run=ready,
                reason="ready" if ready else "gates_or_compatibility_not_sufficient",
            )
        )
        inspected += 1

    return rows


def _row(**kwargs: Any) -> dict[str, Any]:
    return {
        "model_family": kwargs.get("family", "UNKNOWN"),
        "model_artifact_path": kwargs.get("model_artifact_path", ""),
        "schema_source_path": kwargs.get("schema_source_path", ""),
        "artifact_status": kwargs.get("artifact_status", ""),
        "is_reproducible": kwargs.get("is_reproducible", False),
        "required_feature_count": kwargs.get("required_feature_count", 0),
        "runtime_available_feature_count": kwargs.get("runtime_available_feature_count", 0),
        "compatibility_status": kwargs.get("compatibility_status", "UNKNOWN"),
        "ready_for_dry_run": kwargs.get("ready_for_dry_run", False),
        "reason": kwargs.get("reason", ""),
    }


def write_model_artifact_readiness_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"model_family": "NONE", "reason": "no_artifacts_inspected"}]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize_artifact_readiness(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_family = {"RF": 0, "XGB": 0, "TAB": 0}
    ready_by_family = {"RF": 0, "XGB": 0, "TAB": 0}
    unreproducible = 0
    schema_compatible = 0
    dry_run_eligible = 0
    for row in rows:
        fam = row.get("model_family", "UNKNOWN")
        if fam in by_family:
            by_family[fam] += 1
            if row.get("is_reproducible"):
                ready_by_family[fam] += 1
        if not row.get("is_reproducible"):
            unreproducible += 1
        if row.get("compatibility_status") == "COMPATIBLE":
            schema_compatible += 1
        if row.get("ready_for_dry_run"):
            dry_run_eligible += 1
    return {
        "rf_artifacts_inspected": by_family["RF"],
        "xgb_artifacts_inspected": by_family["XGB"],
        "tab_artifacts_inspected": by_family["TAB"],
        "rf_reproducible_schema_count": ready_by_family["RF"],
        "xgb_reproducible_schema_count": ready_by_family["XGB"],
        "tab_reproducible_schema_count": ready_by_family["TAB"],
        "unreproducible_artifacts": unreproducible,
        "schema_compatible_artifacts": schema_compatible,
        "dry_run_eligible_artifacts": dry_run_eligible,
    }
