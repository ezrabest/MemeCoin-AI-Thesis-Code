"""AE7 score artifact inventory, classification, and reproducibility."""

from __future__ import annotations

import csv
import gc
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

from app.artifacts.hash_utils import normalize_project_relative_path
from app.artifacts.registry import load_registry
from app.decision.model_scores import (
    EXACT_ID_COLUMNS,
    MAX_INDEX_BUILD_BYTES,
    MAX_SAFE_INSPECTION_BYTES,
    ArtifactRole,
    ArtifactStatus,
    ModelFamily,
    column_is_leakage_risk,
    column_is_safe_score,
    infer_model_family_from_path,
)

INVENTORY_KIND_TO_ROLE: dict[str, ArtifactRole] = {
    "PREDICTION_OR_SCORE_TABLE": ArtifactRole.PREDICTION_TABLE,
    "METRICS_OR_SUMMARY": ArtifactRole.METRICS,
    "MANIFEST": ArtifactRole.MANIFEST,
    "MODEL_OR_MANIFEST": ArtifactRole.MODEL_ARTIFACT,
    "OTHER_RELEVANT": ArtifactRole.UNKNOWN,
}

DEPRECATED_PATH_FRAGMENTS: tuple[str, ...] = (
    "diagnostic",
    "scratch",
    "deprecated",
    "superseded",
    "recheck_failed",
)

STALE_AGE_DAYS = 14


@dataclass
class ArtifactInspection:
    path: str
    model_family: str
    artifact_role: str
    safe_for_score_population: bool
    reason: str
    id_columns: list[str] = field(default_factory=list)
    score_columns: list[str] = field(default_factory=list)
    rank_columns: list[str] = field(default_factory=list)
    policy_columns: list[str] = field(default_factory=list)
    split_columns: list[str] = field(default_factory=list)
    size_bytes: int = 0
    modified_utc: str | None = None
    row_count: int | None = None
    manifest_path_if_found: str | None = None
    registry_entry_if_found: str | None = None
    content_hash_if_available: str | None = None
    schema_hash_if_available: str | None = None
    is_reproducible: bool = False
    artifact_status: str = ArtifactStatus.UNKNOWN.value
    reproducibility_reason: str = ""
    safe_population_reason: str = ""

    def to_matrix_row(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "model_family": self.model_family,
            "artifact_role": self.artifact_role,
            "size_bytes": self.size_bytes,
            "modified_utc": self.modified_utc,
            "id_columns": "|".join(self.id_columns),
            "score_columns": "|".join(self.score_columns),
            "rank_columns": "|".join(self.rank_columns),
            "policy_columns": "|".join(self.policy_columns),
            "manifest_path_if_found": self.manifest_path_if_found or "",
            "registry_entry_if_found": self.registry_entry_if_found or "",
            "content_hash_if_available": self.content_hash_if_available or "",
            "schema_hash_if_available": self.schema_hash_if_available or "",
            "is_reproducible": self.is_reproducible,
            "artifact_status": self.artifact_status,
            "reproducibility_reason": self.reproducibility_reason,
            "safe_for_score_population": self.safe_for_score_population,
            "safe_population_reason": self.safe_population_reason or self.reason,
        }

    def to_audit_row(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "model_family": self.model_family,
            "row_count": self.row_count,
            "id_columns": "|".join(self.id_columns),
            "score_columns": "|".join(self.score_columns),
            "rank_columns": "|".join(self.rank_columns),
            "policy_columns": "|".join(self.policy_columns),
            "safe_for_score_population": self.safe_for_score_population,
            "rejection_reason": "" if self.safe_for_score_population else self.reason,
            "artifact_status": self.artifact_status,
            "is_reproducible": self.is_reproducible,
        }


def load_inventory_csv(inventory_csv: Path) -> list[dict[str, str]]:
    """Load AE7-0 artifact inventory CSV."""
    rows: list[dict[str, str]] = []
    with open(inventory_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(dict(row))
    return rows


def _parse_pipe_columns(value: str | None) -> list[str]:
    if not value:
        return []
    return [c.strip() for c in value.split("|") if c.strip()]


def _classify_columns(columns: list[str]) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    id_cols = [c for c in columns if c in EXACT_ID_COLUMNS]
    score_cols = [c for c in columns if column_is_safe_score(c)]
    rank_cols = [c for c in columns if "rank" in c.lower() and not column_is_leakage_risk(c)]
    policy_cols = [
        c
        for c in columns
        if c in {"exit_policy", "exit_policy_id", "policy_name", "candidate_policy_id"}
        or "policy" in c.lower()
    ]
    split_cols = [c for c in columns if c == "split" or c.endswith("_split")]
    return id_cols, score_cols, rank_cols, policy_cols, split_cols


def _read_schema_columns(path: Path) -> tuple[list[str], int | None]:
    """Schema-only column inspection — memory safe for parquet and CSV."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        columns = list(pf.schema_arrow.names)
        row_count = pf.metadata.num_rows if pf.metadata is not None else None
        return columns, row_count
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        return [c.strip() for c in header if c.strip()], None
    return [], None


def _find_manifest_near(path: Path, project_root: Path) -> str | None:
    for parent in [path.parent, path.parent.parent]:
        if not parent.is_dir():
            continue
        for pattern in ("*manifest*.json", "*_manifest.json", "manifest.json"):
            matches = sorted(parent.glob(pattern))
            if matches:
                try:
                    return normalize_project_relative_path(matches[0], project_root)
                except ValueError:
                    return str(matches[0])
    return None


def _is_deprecated_path(path: str) -> bool:
    lower = path.lower().replace("\\", "/")
    return any(frag in lower for frag in DEPRECATED_PATH_FRAGMENTS)


def _parse_modified_utc(modified: str | None, path: Path) -> datetime | None:
    if modified:
        try:
            dt = datetime.fromisoformat(modified.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass
    if path.is_file():
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return None


def assess_reproducibility(
    *,
    path: Path,
    project_root: Path,
    registry_by_path: dict[str, Any],
    manifest_path: str | None,
    modified_utc: datetime | None,
) -> tuple[bool, str, str, str | None, str | None, str | None]:
    """Return (is_reproducible, artifact_status, reason, registry_id, content_hash, schema_hash)."""
    rel_path = None
    registry_entry = None
    try:
        rel_path = normalize_project_relative_path(path, project_root)
        registry_entry = registry_by_path.get(rel_path.replace("\\", "/"))
        if registry_entry is None:
            registry_entry = registry_by_path.get(rel_path)
    except ValueError:
        rel_path = str(path)

    content_hash = None
    schema_hash = None
    registry_id = None
    if registry_entry is not None:
        registry_id = getattr(registry_entry, "artifact_id", None) or registry_entry.get("artifact_id")
        content_hash = getattr(registry_entry, "content_hash", None) or registry_entry.get("content_hash")
        schema_hash = getattr(registry_entry, "schema_hash", None) or registry_entry.get("schema_hash")

    path_str = rel_path or str(path)
    if _is_deprecated_path(path_str):
        return False, ArtifactStatus.DEPRECATED.value, "path_in_deprecated_or_diagnostic_branch", registry_id, content_hash, schema_hash

    if registry_entry is None and manifest_path is None:
        return False, ArtifactStatus.UNREGISTERED.value, "no_registry_entry_or_manifest", registry_id, content_hash, schema_hash

    if registry_entry is None:
        return False, ArtifactStatus.UNKNOWN.value, "manifest_found_but_not_in_registry", registry_id, content_hash, schema_hash

    # Registered artifact — assess freshness
    is_stale = False
    if modified_utc is not None:
        age = datetime.now(timezone.utc) - modified_utc
        if age > timedelta(days=STALE_AGE_DAYS):
            is_stale = True

    warnings = getattr(registry_entry, "warnings", None) or registry_entry.get("warnings") or []
    if warnings and any("DEPRECATED" in str(w).upper() for w in warnings):
        return False, ArtifactStatus.DEPRECATED.value, "registry_warnings_indicate_deprecated", registry_id, content_hash, schema_hash

    if is_stale:
        return True, ArtifactStatus.STALE.value, "registered_but_modified_over_stale_threshold", registry_id, content_hash, schema_hash

    return True, ArtifactStatus.CURRENT.value, "registered_with_manifest_or_hash_metadata", registry_id, content_hash, schema_hash


def classify_inventory_row(
    row: dict[str, str],
    *,
    project_root: Path,
    registry_by_path: dict[str, Any] | None = None,
) -> ArtifactInspection:
    """Classify a single inventory row without loading full artifact."""
    registry_by_path = registry_by_path or {}
    raw_path = row.get("path", "")
    path = project_root / raw_path.replace("\\", "/")
    kind = row.get("kind", "")
    size_bytes = int(row.get("size_bytes") or 0)
    modified_utc_str = row.get("modified_utc")

    artifact_role = INVENTORY_KIND_TO_ROLE.get(kind, ArtifactRole.UNKNOWN).value
    model_family = infer_model_family_from_path(raw_path).value

    if kind not in {"PREDICTION_OR_SCORE_TABLE"}:
        return ArtifactInspection(
            path=raw_path,
            model_family=model_family,
            artifact_role=artifact_role,
            safe_for_score_population=False,
            reason=f"artifact_role_not_prediction_table:{kind}",
            size_bytes=size_bytes,
            modified_utc=modified_utc_str,
            artifact_status=ArtifactStatus.UNKNOWN.value,
            reproducibility_reason="not_a_prediction_table",
            safe_population_reason="not_a_prediction_table",
        )

    if size_bytes > MAX_SAFE_INSPECTION_BYTES and path.suffix.lower() != ".parquet":
        return ArtifactInspection(
            path=raw_path,
            model_family=model_family,
            artifact_role=artifact_role,
            safe_for_score_population=False,
            reason="TOO_LARGE_FOR_AE7_SAFE_INSPECTION",
            size_bytes=size_bytes,
            modified_utc=modified_utc_str,
            artifact_status=ArtifactStatus.TOO_LARGE_SKIPPED.value,
            is_reproducible=False,
            reproducibility_reason="too_large_for_safe_inspection",
            safe_population_reason="TOO_LARGE_FOR_AE7_SAFE_INSPECTION",
        )

    # Column hints from inventory
    inv_id_cols = _parse_pipe_columns(row.get("id_column_hits"))
    inv_score_cols = _parse_pipe_columns(row.get("score_column_hits"))

    columns: list[str] = []
    row_count: int | None = None
    if path.is_file():
        try:
            columns, row_count = _read_schema_columns(path)
        except Exception as exc:
            return ArtifactInspection(
                path=raw_path,
                model_family=model_family,
                artifact_role=artifact_role,
                safe_for_score_population=False,
                reason=f"schema_inspection_failed:{exc}",
                size_bytes=size_bytes,
                modified_utc=modified_utc_str,
                artifact_status=ArtifactStatus.UNKNOWN.value,
            )
    else:
        columns = list({*inv_id_cols, *inv_score_cols})
        if not columns:
            return ArtifactInspection(
                path=raw_path,
                model_family=model_family,
                artifact_role=artifact_role,
                safe_for_score_population=False,
                reason="artifact_file_not_found",
                size_bytes=size_bytes,
                modified_utc=modified_utc_str,
                artifact_status=ArtifactStatus.UNKNOWN.value,
            )

    id_cols, score_cols, rank_cols, policy_cols, split_cols = _classify_columns(columns)

    # Re-infer model family from columns when path is ambiguous
    if model_family == ModelFamily.UNKNOWN.value:
        for col in columns:
            lower = col.lower()
            if lower.startswith("tab_") or lower == "tab_score":
                model_family = ModelFamily.TAB.value
                break
            if "xgb" in lower:
                model_family = ModelFamily.XGB.value
                break
            if lower in {"predicted_probability"} and "rf" in raw_path.lower():
                model_family = ModelFamily.RF.value

    # Reject pair/time-only ID alignment
    unsafe_id_hits = {"pair_address", "event_timestamp", "symbol", "provider"}
    has_exact_id = bool(id_cols)
    has_only_unsafe_ids = bool(inv_id_cols) and not has_exact_id and all(
        c in unsafe_id_hits or c in {"filter", "horizon", "exit_policy_id"} for c in inv_id_cols
    )

    if has_only_unsafe_ids or (inv_id_cols and not has_exact_id):
        manifest_path = _find_manifest_near(path, project_root) if path.is_file() else None
        modified_dt = _parse_modified_utc(modified_utc_str, path)
        is_rep, status, rep_reason, reg_id, ch, sh = assess_reproducibility(
            path=path,
            project_root=project_root,
            registry_by_path=registry_by_path,
            manifest_path=manifest_path,
            modified_utc=modified_dt,
        )
        return ArtifactInspection(
            path=raw_path,
            model_family=model_family,
            artifact_role=artifact_role,
            safe_for_score_population=False,
            reason="REJECTED_NO_SAFE_ID:only_pair_time_or_fuzzy_id_columns",
            id_columns=id_cols,
            score_columns=score_cols,
            rank_columns=rank_cols,
            policy_columns=policy_cols,
            split_columns=split_cols,
            size_bytes=size_bytes,
            modified_utc=modified_utc_str,
            row_count=row_count,
            manifest_path_if_found=manifest_path,
            registry_entry_if_found=reg_id,
            content_hash_if_available=ch,
            schema_hash_if_available=sh,
            is_reproducible=is_rep,
            artifact_status=ArtifactStatus.REJECTED_NO_SAFE_ID.value,
            reproducibility_reason=rep_reason,
            safe_population_reason="no_exact_id_columns",
        )

    if not has_exact_id:
        return ArtifactInspection(
            path=raw_path,
            model_family=model_family,
            artifact_role=artifact_role,
            safe_for_score_population=False,
            reason="REJECTED_NO_SAFE_ID",
            id_columns=id_cols,
            score_columns=score_cols,
            rank_columns=rank_cols,
            policy_columns=policy_cols,
            split_columns=split_cols,
            size_bytes=size_bytes,
            modified_utc=modified_utc_str,
            row_count=row_count,
            artifact_status=ArtifactStatus.REJECTED_NO_SAFE_ID.value,
            reproducibility_reason="missing_exact_id_column",
            safe_population_reason="REJECTED_NO_SAFE_ID",
        )

    if not score_cols:
        leakage_only = [c for c in columns if column_is_leakage_risk(c)]
        reason = "REJECTED_NO_SAFE_SCORE"
        if leakage_only and not any(column_is_safe_score(c) for c in columns):
            reason = "REJECTED_LEAKAGE_RISK:only_label_outcome_columns"
        return ArtifactInspection(
            path=raw_path,
            model_family=model_family,
            artifact_role=artifact_role,
            safe_for_score_population=False,
            reason=reason,
            id_columns=id_cols,
            score_columns=score_cols,
            rank_columns=rank_cols,
            policy_columns=policy_cols,
            split_columns=split_cols,
            size_bytes=size_bytes,
            modified_utc=modified_utc_str,
            row_count=row_count,
            artifact_status=(
                ArtifactStatus.REJECTED_LEAKAGE_RISK.value
                if "LEAKAGE" in reason
                else ArtifactStatus.REJECTED_NO_SAFE_SCORE.value
            ),
            reproducibility_reason=reason,
            safe_population_reason=reason,
        )

    if model_family == ModelFamily.UNKNOWN.value:
        return ArtifactInspection(
            path=raw_path,
            model_family=model_family,
            artifact_role=artifact_role,
            safe_for_score_population=False,
            reason="unknown_model_family",
            id_columns=id_cols,
            score_columns=score_cols,
            rank_columns=rank_cols,
            policy_columns=policy_cols,
            split_columns=split_cols,
            size_bytes=size_bytes,
            modified_utc=modified_utc_str,
            row_count=row_count,
            artifact_status=ArtifactStatus.UNKNOWN.value,
            safe_population_reason="unknown_model_family",
        )

    manifest_path = _find_manifest_near(path, project_root) if path.is_file() else None
    modified_dt = _parse_modified_utc(modified_utc_str, path)
    is_rep, status, rep_reason, reg_id, ch, sh = assess_reproducibility(
        path=path,
        project_root=project_root,
        registry_by_path=registry_by_path,
        manifest_path=manifest_path,
        modified_utc=modified_dt,
    )

    safe = is_rep and status in {
        ArtifactStatus.CURRENT.value,
        ArtifactStatus.STALE.value,
    }
    # Default reject stale per spec unless explicit exception — spec says do not populate from STALE
    if status == ArtifactStatus.STALE.value:
        safe = False

    safe_reason = rep_reason if safe else f"artifact_status_{status}"
    if not safe:
        reason = (
            "ARTIFACT_NOT_REPRODUCIBLE_OR_STALE"
            if not is_rep
            else f"artifact_status_{status}"
        )
    else:
        reason = "safe_exact_id_and_score_columns_reproducible"

    return ArtifactInspection(
        path=raw_path,
        model_family=model_family,
        artifact_role=artifact_role,
        safe_for_score_population=safe,
        reason=reason,
        id_columns=id_cols,
        score_columns=score_cols,
        rank_columns=rank_cols,
        policy_columns=policy_cols,
        split_columns=split_cols,
        size_bytes=size_bytes,
        modified_utc=modified_utc_str,
        row_count=row_count,
        manifest_path_if_found=manifest_path,
        registry_entry_if_found=reg_id,
        content_hash_if_available=ch,
        schema_hash_if_available=sh,
        is_reproducible=is_rep,
        artifact_status=status,
        reproducibility_reason=rep_reason,
        safe_population_reason=safe_reason,
    )


def classify_inventory(
    inventory_rows: list[dict[str, str]],
    *,
    project_root: Path,
    max_artifacts: int | None = None,
    registry_path: Path | None = None,
) -> list[ArtifactInspection]:
    """Classify inventory rows up to max_artifacts prediction tables."""
    registry_by_path: dict[str, Any] = {}
    reg_path = registry_path or (project_root / "data/training/artifact_registry/artifact_registry.jsonl")
    if reg_path.is_file():
        try:
            registry_by_path = load_registry(reg_path)
        except Exception:
            registry_by_path = {}

    # Prioritize prediction tables with exact ID columns
    candidates = [
        r
        for r in inventory_rows
        if r.get("kind") == "PREDICTION_OR_SCORE_TABLE"
    ]

    def _priority(row: dict[str, str]) -> tuple[int, int]:
        id_hits = row.get("id_column_hits") or ""
        has_exact = any(k in id_hits for k in EXACT_ID_COLUMNS)
        size = int(row.get("size_bytes") or 0)
        return (0 if has_exact else 1, size)

    candidates.sort(key=_priority)
    if max_artifacts is not None:
        candidates = candidates[:max_artifacts]

    inspections: list[ArtifactInspection] = []
    for row in candidates:
        inspections.append(
            classify_inventory_row(row, project_root=project_root, registry_by_path=registry_by_path)
        )
    return inspections


def write_reproducibility_matrix(
    inspections: list[ArtifactInspection],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(inspections[0].to_matrix_row().keys()) if inspections else [
        "path", "model_family", "artifact_role", "size_bytes", "modified_utc",
        "id_columns", "score_columns", "rank_columns", "policy_columns",
        "manifest_path_if_found", "registry_entry_if_found",
        "content_hash_if_available", "schema_hash_if_available",
        "is_reproducible", "artifact_status", "reproducibility_reason",
        "safe_for_score_population", "safe_population_reason",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for insp in inspections:
            writer.writerow(insp.to_matrix_row())


def load_registry_safe(project_root: Path) -> dict[str, Any]:
    reg_path = project_root / "data/training/artifact_registry/artifact_registry.jsonl"
    if not reg_path.is_file():
        return {}
    try:
        return load_registry(reg_path)
    except Exception:
        return {}


@dataclass
class IndexedScoreRow:
    score: float
    rank: float | None
    model_family: str
    artifact_path: str
    id_key_used: str
    id_value: str
    horizon: str | None
    filter: str | None
    exit_policy: str | None
    split: str | None
    score_column_used: str
    rank_column_used: str | None
    artifact_status: str
    is_reproducible: bool
    model_artifact_id: str | None = None


class PredictionIndex:
    """Lightweight exact-ID lookup index per model family."""

    def __init__(self) -> None:
        self._by_family: dict[str, dict[str, IndexedScoreRow]] = {
            "RF": {},
            "XGB": {},
            "TAB": {},
        }

    def add_row(self, row: IndexedScoreRow) -> None:
        family = row.model_family
        if family not in self._by_family:
            return
        key = f"{row.id_key_used}::{row.id_value}"
        # Prefer first indexed artifact (stable ordering from classify_inventory)
        if key not in self._by_family[family]:
            self._by_family[family][key] = row

    def lookup(
        self,
        model_family: str,
        id_key: str,
        id_value: str,
    ) -> IndexedScoreRow | None:
        key = f"{id_key}::{id_value}"
        return self._by_family.get(model_family, {}).get(key)

    def lookup_identity(
        self,
        model_family: str,
        identity: dict[str, Any],
    ) -> IndexedScoreRow | None:
        for id_key in EXACT_ID_COLUMNS:
            value = identity.get(id_key)
            if value is None or (isinstance(value, str) and not value.strip()):
                continue
            hit = self.lookup(model_family, id_key, str(value))
            if hit is not None:
                return hit
        return None

    @property
    def families_indexed(self) -> list[str]:
        return [f for f, m in self._by_family.items() if m]


def _read_index_columns_parquet(
    path: Path,
    columns: list[str],
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    available = [c for c in columns if c in pf.schema_arrow.names]
    if not available:
        return []
    table = pf.read(columns=available)
    try:
        df = table.to_pandas()
        return df.to_dict(orient="records")
    finally:
        del table
        gc.collect()


def _read_index_columns_csv(
    path: Path,
    columns: list[str],
) -> list[dict[str, Any]]:
    import pandas as pd

    header_cols, _ = _read_schema_columns(path)
    usecols = [c for c in columns if c in header_cols]
    if not usecols:
        return []
    chunks: list[dict[str, Any]] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=50_000):
        chunks.extend(chunk.to_dict(orient="records"))
        del chunk
    gc.collect()
    return chunks


def build_prediction_index(
    inspections: list[ArtifactInspection],
    *,
    project_root: Path,
) -> PredictionIndex:
    """Build index from safe, reproducible, CURRENT artifacts only."""
    index = PredictionIndex()
    safe_current = [
        i
        for i in inspections
        if i.safe_for_score_population
        and i.is_reproducible
        and i.artifact_status == ArtifactStatus.CURRENT.value
    ]

    for insp in safe_current:
        path = project_root / insp.path.replace("\\", "/")
        if not path.is_file():
            continue
        if insp.size_bytes > MAX_INDEX_BUILD_BYTES:
            continue

        read_cols = list(
            dict.fromkeys(
                [
                    *insp.id_columns,
                    *insp.score_columns,
                    *insp.rank_columns,
                    *insp.policy_columns,
                    *insp.split_columns,
                    "horizon",
                    "filter",
                    "exit_policy",
                    "exit_policy_id",
                ]
            )
        )

        try:
            if path.suffix.lower() == ".parquet":
                rows = _read_index_columns_parquet(path, read_cols)
            elif path.suffix.lower() == ".csv":
                rows = _read_index_columns_csv(path, read_cols)
            else:
                continue
        except Exception:
            continue

        score_col = insp.score_columns[0] if insp.score_columns else None
        rank_col = insp.rank_columns[0] if insp.rank_columns else None
        if not score_col:
            continue

        for raw in rows:
            for id_key in insp.id_columns:
                id_val = raw.get(id_key)
                if id_val is None or (isinstance(id_val, float) and str(id_val) == "nan"):
                    continue
                id_str = str(id_val).strip()
                if not id_str:
                    continue
                score_val = raw.get(score_col)
                if score_val is None:
                    continue
                try:
                    score_f = float(score_val)
                except (TypeError, ValueError):
                    continue

                rank_f = None
                if rank_col and raw.get(rank_col) is not None:
                    try:
                        rank_f = float(raw[rank_col])
                    except (TypeError, ValueError):
                        rank_f = None

                exit_policy = raw.get("exit_policy") or raw.get("exit_policy_id")
                index.add_row(
                    IndexedScoreRow(
                        score=score_f,
                        rank=rank_f,
                        model_family=insp.model_family,
                        artifact_path=insp.path,
                        id_key_used=id_key,
                        id_value=id_str,
                        horizon=str(raw["horizon"]) if raw.get("horizon") is not None else None,
                        filter=str(raw["filter"]) if raw.get("filter") is not None else None,
                        exit_policy=str(exit_policy) if exit_policy is not None else None,
                        split=str(raw["split"]) if raw.get("split") is not None else None,
                        score_column_used=score_col,
                        rank_column_used=rank_col,
                        artifact_status=insp.artifact_status,
                        is_reproducible=insp.is_reproducible,
                        model_artifact_id=insp.registry_entry_if_found,
                    )
                )
        del rows
        gc.collect()

    return index
