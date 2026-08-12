"""AE16 model-evidence adapter for Clean Forward candidates.

Discovers RF / XGB / TAB historical prediction artifacts and attempts safe
exact-ID attachment. Never invents scores. Never defaults missing scores to 0.
Never joins on forbidden keys alone (pair_address / event_timestamp).
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.consensus import MODEL_FAMILIES
from app.decision.model_scores import EXACT_ID_COLUMNS, FORBIDDEN_ALIGNMENT_KEYS, SAFE_SCORE_FRAGMENTS

# Known historical artifact roots (project-relative). Discovery only — not Clean Forward SoT.
DEFAULT_ARTIFACT_SEARCH_ROOTS = (
    "data/training/manual_verified_results",
    "data/training/artifact_registry",
    "data/training/models",
    "data/audits/e5_focused_review_pack_20260704_000002",
)

PREDICTION_NAME_HINTS = ("prediction", "predictions", "scores", "scored")
FAMILY_PATH_HINTS = {
    "RF": ("rf", "random_forest", "random-forest"),
    "XGB": ("xgb", "xgboost"),
    "TAB": ("tab", "tabicl", "tabiclv2"),
}


@dataclass
class DiscoveredArtifact:
    path: Path
    model_family: str
    role: str  # prediction_table | model_artifact | registry | unknown
    has_exact_id_columns: bool = False
    has_safe_score_column: bool = False
    columns: list[str] = field(default_factory=list)
    read_error: str | None = None


@dataclass
class AttachmentResult:
    clean_forward_candidate_id: str
    clean_forward_decision_input_id: str
    pair_address: str
    base_token_address: str
    quote_token_address: str
    model_family: str
    evidence_attached: bool
    score: float | None
    rank: float | None
    percentile_rank: float | None
    source_artifact_path: str
    source_run_id: str
    source_prediction_file: str
    source_model_artifact: str
    candidate_policy_id: str
    target_row_id: str
    target_name: str
    target_version: str
    horizon: str
    filter_name: str
    exit_policy_id: str
    evidence_type: str
    attachment_status: str
    attachment_failure_reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "clean_forward_candidate_id": self.clean_forward_candidate_id,
            "clean_forward_decision_input_id": self.clean_forward_decision_input_id,
            "pair_address": self.pair_address,
            "base_token_address": self.base_token_address,
            "quote_token_address": self.quote_token_address,
            "model_family": self.model_family,
            "evidence_attached": self.evidence_attached,
            "score": self.score,
            "rank": self.rank,
            "percentile_rank": self.percentile_rank,
            "source_artifact_path": self.source_artifact_path,
            "source_run_id": self.source_run_id,
            "source_prediction_file": self.source_prediction_file,
            "source_model_artifact": self.source_model_artifact,
            "candidate_policy_id": self.candidate_policy_id,
            "target_row_id": self.target_row_id,
            "target_name": self.target_name,
            "target_version": self.target_version,
            "horizon": self.horizon,
            "filter_name": self.filter_name,
            "exit_policy_id": self.exit_policy_id,
            "evidence_type": self.evidence_type,
            "attachment_status": self.attachment_status,
            "attachment_failure_reason": self.attachment_failure_reason,
        }


def _empty_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    s = str(value).strip().lower()
    return s in {"", "none", "null", "nan", "n/a", "na"}


def parse_numeric_score(value: Any) -> float | None:
    """Parse a numeric score. Blank/placeholder -> None. Never coerces missing to 0."""
    if _is_blank(value):
        return None
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def infer_model_family_from_path(path: Path) -> str | None:
    text = str(path).replace("\\", "/").lower()
    # Prefer more specific TAB / XGB tokens before generic "rf" fragment collisions.
    for family in ("TAB", "XGB", "RF"):
        for hint in FAMILY_PATH_HINTS[family]:
            if f"/{hint}" in f"/{text}" or f"_{hint}" in text or f"-{hint}" in text or f"{hint}_" in text:
                return family
    return None


def _looks_like_prediction_csv(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() != ".csv":
        return False
    return any(h in name for h in PREDICTION_NAME_HINTS) or "prediction" in str(path.parent).lower()


def _safe_score_column(columns: list[str]) -> str | None:
    for col in columns:
        low = col.lower()
        if any(frag in low for frag in ("target", "label", "future", "return", "profit")):
            continue
        if any(frag in low for frag in SAFE_SCORE_FRAGMENTS):
            return col
    return None


def inspect_prediction_csv(path: Path, *, max_bytes: int = 2_000_000) -> DiscoveredArtifact:
    family = infer_model_family_from_path(path) or "UNKNOWN"
    try:
        size = path.stat().st_size
        if size > max_bytes:
            # Header-only read still OK for schema probe.
            pass
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
        columns = [c.strip() for c in header if c is not None]
        has_exact = any(c in columns for c in EXACT_ID_COLUMNS)
        has_score = _safe_score_column(columns) is not None
        return DiscoveredArtifact(
            path=path,
            model_family=family if family in MODEL_FAMILIES else "UNKNOWN",
            role="prediction_table",
            has_exact_id_columns=has_exact,
            has_safe_score_column=has_score,
            columns=columns,
        )
    except Exception as exc:  # noqa: BLE001 — fail closed per artifact
        return DiscoveredArtifact(
            path=path,
            model_family=family if family in MODEL_FAMILIES else "UNKNOWN",
            role="prediction_table",
            read_error=f"{type(exc).__name__}: {exc}",
        )


def discover_model_artifacts(
    project_root: Path,
    *,
    search_roots: tuple[str, ...] | None = None,
    max_files_per_family: int = 40,
) -> dict[str, list[DiscoveredArtifact]]:
    """Discover historical RF/XGB/TAB prediction/model artifacts under known roots."""
    roots = search_roots or DEFAULT_ARTIFACT_SEARCH_ROOTS
    by_family: dict[str, list[DiscoveredArtifact]] = {f: [] for f in MODEL_FAMILIES}

    for rel in roots:
        root = project_root / rel
        if not root.exists():
            continue
        # Registry jsonl (metadata only)
        if root.is_file() and root.name.endswith(".jsonl"):
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            # Skip enormous trees of unrelated files
            if path.suffix.lower() not in {".csv", ".json", ".jsonl", ".joblib", ".pkl", ".pt", ".bin"}:
                continue
            family = infer_model_family_from_path(path)
            if family not in MODEL_FAMILIES:
                continue
            if len(by_family[family]) >= max_files_per_family:
                continue

            if _looks_like_prediction_csv(path):
                art = inspect_prediction_csv(path)
                if art.model_family == "UNKNOWN":
                    art.model_family = family
                by_family[family].append(art)
            elif path.suffix.lower() in {".joblib", ".pkl", ".pt", ".bin"} or "model" in path.name.lower():
                by_family[family].append(
                    DiscoveredArtifact(path=path, model_family=family, role="model_artifact")
                )
            elif path.name == "artifact_registry.jsonl":
                by_family[family].append(
                    DiscoveredArtifact(path=path, model_family=family, role="registry")
                )

    return by_family


def _pick_best_prediction(artifacts: list[DiscoveredArtifact]) -> DiscoveredArtifact | None:
    preds = [a for a in artifacts if a.role == "prediction_table" and not a.read_error]
    if not preds:
        return None
    # Prefer exact-ID + safe score schema
    scored = [a for a in preds if a.has_exact_id_columns and a.has_safe_score_column]
    if scored:
        return scored[0]
    scored_only = [a for a in preds if a.has_safe_score_column]
    if scored_only:
        return scored_only[0]
    return preds[0]


def _candidate_join_keys(candidate: dict[str, Any], decision: dict[str, Any]) -> dict[str, str]:
    return {
        "clean_forward_candidate_id": _empty_str(candidate.get("clean_forward_candidate_id")),
        "candidate_id": _empty_str(candidate.get("clean_forward_candidate_id")),
        "candidate_policy_id": _empty_str(
            candidate.get("candidate_policy_id") or decision.get("candidate_policy_id")
        ),
        "target_row_id": _empty_str(candidate.get("target_row_id") or decision.get("target_row_id")),
    }


def attach_model_evidence_for_candidate(
    *,
    candidate: dict[str, Any],
    decision: dict[str, Any],
    model_family: str,
    discovered: dict[str, list[DiscoveredArtifact]],
    project_root: Path,
    allow_pair_timestamp_join: bool = False,
) -> AttachmentResult:
    """Attach one model family to one Clean Forward candidate. Never raises for expected failures."""
    base = AttachmentResult(
        clean_forward_candidate_id=_empty_str(candidate.get("clean_forward_candidate_id")),
        clean_forward_decision_input_id=_empty_str(decision.get("clean_forward_decision_input_id")),
        pair_address=_empty_str(candidate.get("pair_address")),
        base_token_address=_empty_str(candidate.get("base_token_address")),
        quote_token_address=_empty_str(candidate.get("quote_token_address")),
        model_family=model_family,
        evidence_attached=False,
        score=None,
        rank=None,
        percentile_rank=None,
        source_artifact_path="",
        source_run_id="",
        source_prediction_file="",
        source_model_artifact="",
        candidate_policy_id="",
        target_row_id="",
        target_name="",
        target_version="",
        horizon="",
        filter_name="",
        exit_policy_id="",
        evidence_type="historical_prediction_artifact",
        attachment_status="MODEL_EVIDENCE_UNAVAILABLE",
        attachment_failure_reason="",
    )

    try:
        # AE15 placeholder columns must never be treated as attached evidence.
        placeholder_key = {"RF": "rf_score", "XGB": "xgb_score", "TAB": "tab_score"}[model_family]
        placeholder = decision.get(placeholder_key)
        scores_available_raw = str(decision.get("model_scores_available", "")).strip().lower()
        if scores_available_raw in {"true", "1", "yes"} and not _is_blank(placeholder):
            # Still require explicit artifact-backed attachment for AE16 authority.
            # Placeholder alone is not MODEL_EVIDENCE_ATTACHED.
            pass

        arts = discovered.get(model_family) or []
        if not arts:
            base.attachment_status = "ARTIFACT_NOT_FOUND"
            base.attachment_failure_reason = (
                f"No {model_family} prediction/model artifacts discovered under known roots"
            )
            return base

        pred = _pick_best_prediction(arts)
        model_art = next((a for a in arts if a.role == "model_artifact"), None)

        if pred is None:
            if model_art is not None:
                base.source_model_artifact = str(model_art.path)
                base.source_artifact_path = str(model_art.path)
                base.attachment_status = "PREDICTION_FILE_NOT_FOUND"
                base.attachment_failure_reason = (
                    f"{model_family} model artifact present but no readable prediction table found"
                )
            else:
                base.attachment_status = "PREDICTION_FILE_NOT_FOUND"
                base.attachment_failure_reason = f"No {model_family} prediction CSV discovered"
            return base

        if pred.read_error:
            base.source_artifact_path = str(pred.path)
            base.source_prediction_file = str(pred.path)
            base.attachment_status = "ARTIFACT_READ_ERROR"
            base.attachment_failure_reason = pred.read_error
            return base

        try:
            base.source_artifact_path = str(
                pred.path.resolve().relative_to(project_root.resolve())
            ).replace("\\", "/")
        except ValueError:
            base.source_artifact_path = str(pred.path).replace("\\", "/")
        base.source_prediction_file = base.source_artifact_path
        if model_art is not None:
            try:
                base.source_model_artifact = str(
                    model_art.path.resolve().relative_to(project_root.resolve())
                ).replace("\\", "/")
            except ValueError:
                base.source_model_artifact = str(model_art.path).replace("\\", "/")

        if not pred.has_safe_score_column:
            base.attachment_status = "ARTIFACT_SCHEMA_UNSUPPORTED"
            base.attachment_failure_reason = "Prediction artifact lacks a safe score/probability column"
            return base

        join_keys = _candidate_join_keys(candidate, decision)
        base.candidate_policy_id = join_keys["candidate_policy_id"]
        base.target_row_id = join_keys["target_row_id"]

        # Exact-ID join required for safe attachment. Clean Forward AE15 package
        # does not carry candidate_policy_id / target_row_id.
        if not join_keys["candidate_policy_id"] and not join_keys["target_row_id"]:
            if not pred.has_exact_id_columns:
                # Without exact IDs on either side, pair/timestamp join would be required —
                # that is explicitly rejected for score alignment.
                if not allow_pair_timestamp_join:
                    base.attachment_status = "LEGACY_SOURCE_REJECTED"
                    base.attachment_failure_reason = (
                        "Cannot safely join Clean Forward candidate to historical predictions: "
                        "missing candidate_policy_id/target_row_id and pair_address/event_timestamp "
                        "alignment is forbidden"
                    )
                    return base
                base.attachment_status = "PAIR_TIMESTAMP_NOT_MATCHED"
                base.attachment_failure_reason = "pair/timestamp join not approved for AE16"
                return base

            if not join_keys["candidate_policy_id"]:
                base.attachment_status = "POLICY_ID_NOT_AVAILABLE"
                base.attachment_failure_reason = (
                    "Clean Forward candidate/decision lacks candidate_policy_id required for exact-ID join"
                )
                return base
            if not join_keys["target_row_id"]:
                base.attachment_status = "TARGET_ROW_ID_NOT_AVAILABLE"
                base.attachment_failure_reason = (
                    "Clean Forward candidate/decision lacks target_row_id required for exact-ID join"
                )
                return base

        # If policy/target IDs are present, attempt exact match in prediction file.
        matched = _lookup_score_by_exact_id(pred.path, join_keys)
        if matched.get("error"):
            base.attachment_status = "ARTIFACT_READ_ERROR"
            base.attachment_failure_reason = str(matched["error"])
            return base
        if not matched.get("found"):
            base.attachment_status = "CANDIDATE_ID_NOT_MATCHED"
            base.attachment_failure_reason = (
                "Exact-ID present on candidate but no matching row in prediction artifact"
            )
            return base

        score = parse_numeric_score(matched.get("score"))
        if score is None:
            base.attachment_status = "SCORE_NOT_ATTACHED"
            base.attachment_failure_reason = "Matched prediction row but score was null/non-numeric"
            return base

        # Runtime model authority is not approved in AE16 — still attach as research evidence.
        base.evidence_attached = True
        base.score = score
        base.rank = parse_numeric_score(matched.get("rank"))
        base.percentile_rank = parse_numeric_score(matched.get("percentile_rank"))
        base.horizon = _empty_str(matched.get("horizon"))
        base.filter_name = _empty_str(matched.get("filter") or matched.get("filter_name"))
        base.exit_policy_id = _empty_str(matched.get("exit_policy_id"))
        base.target_name = _empty_str(matched.get("target_name"))
        base.target_version = _empty_str(matched.get("target_version"))
        base.source_run_id = _empty_str(matched.get("source_run_id") or pred.path.parent.name)
        base.attachment_status = "MODEL_EVIDENCE_ATTACHED"
        base.attachment_failure_reason = ""
        base.evidence_type = "historical_exact_id_prediction"
        return base

    except Exception as exc:  # noqa: BLE001 — capture per-candidate; do not crash AE16
        base.attachment_status = "ATTACHMENT_EXCEPTION_CAUGHT"
        base.attachment_failure_reason = f"{type(exc).__name__}: {exc}"
        base.evidence_attached = False
        base.score = None
        return base


def _lookup_score_by_exact_id(path: Path, join_keys: dict[str, str]) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            score_col = _safe_score_column(list(columns))
            if not score_col:
                return {"found": False, "error": "no safe score column"}
            id_cols = [c for c in EXACT_ID_COLUMNS if c in columns]
            if not id_cols:
                return {"found": False}
            # Reject if only forbidden keys would be usable
            if set(columns).issubset(FORBIDDEN_ALIGNMENT_KEYS):
                return {"found": False, "error": "forbidden alignment keys only"}

            wanted = {
                "target_row_id": join_keys.get("target_row_id") or "",
                "candidate_policy_id": join_keys.get("candidate_policy_id") or "",
                "candidate_id": join_keys.get("candidate_id") or "",
            }
            for row in reader:
                for col in id_cols:
                    val = _empty_str(row.get(col))
                    if not val:
                        continue
                    if wanted.get(col) and val == wanted[col]:
                        return {
                            "found": True,
                            "score": row.get(score_col),
                            "rank": row.get("rank") or row.get("predicted_rank"),
                            "percentile_rank": row.get("percentile_rank"),
                            "horizon": row.get("horizon"),
                            "filter": row.get("filter") or row.get("filter_name"),
                            "exit_policy_id": row.get("exit_policy_id"),
                            "target_name": row.get("target_name"),
                            "target_version": row.get("target_version"),
                            "source_run_id": row.get("run_id") or row.get("source_run_id"),
                            "id_key_used": col,
                        }
            return {"found": False}
    except Exception as exc:  # noqa: BLE001
        return {"found": False, "error": f"{type(exc).__name__}: {exc}"}


def attach_all_model_evidence(
    *,
    candidates: list[dict[str, Any]],
    decision_by_candidate: dict[str, dict[str, Any]],
    project_root: Path,
    discovered: dict[str, list[DiscoveredArtifact]] | None = None,
) -> list[AttachmentResult]:
    if discovered is None:
        discovered = discover_model_artifacts(project_root)
    rows: list[AttachmentResult] = []
    for cand in candidates:
        cid = _empty_str(cand.get("clean_forward_candidate_id"))
        decision = decision_by_candidate.get(cid) or {}
        for family in MODEL_FAMILIES:
            rows.append(
                attach_model_evidence_for_candidate(
                    candidate=cand,
                    decision=decision,
                    model_family=family,
                    discovered=discovered,
                    project_root=project_root,
                )
            )
    return rows


def summarize_model_availability(attachments: list[AttachmentResult]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for family in MODEL_FAMILIES:
        fam_rows = [a for a in attachments if a.model_family == family]
        attached = [a for a in fam_rows if a.evidence_attached and a.attachment_status == "MODEL_EVIDENCE_ATTACHED"]
        status_counts: dict[str, int] = {}
        for a in fam_rows:
            status_counts[a.attachment_status] = status_counts.get(a.attachment_status, 0) + 1
        out.append(
            {
                "model_family": family,
                "attachment_rows": len(fam_rows),
                "evidence_attached_count": len(attached),
                "evidence_missing_count": len(fam_rows) - len(attached),
                "scores_non_null_count": sum(1 for a in fam_rows if a.score is not None),
                "scores_null_count": sum(1 for a in fam_rows if a.score is None),
                "primary_attachment_status": max(status_counts.items(), key=lambda kv: kv[1])[0]
                if status_counts
                else "MODEL_EVIDENCE_UNAVAILABLE",
                "attachment_status_counts_json": json.dumps(status_counts, sort_keys=True),
            }
        )
    return out
