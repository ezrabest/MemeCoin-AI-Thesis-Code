"""AE16 continuation: direct-target model-evidence bridge discovery + parity.

Read-only discovery of RF/XGB/TAB artifacts, exact-ID join audit, and
feature-parity / inference compatibility. Never invents scores.
Never joins on pair_address alone or approximate timestamps.
"""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.consensus import (
    CANONICAL_DIRECT_TARGET_ROOTS,
    CF_TO_MODEL_FEATURE_MAP,
    MODEL_FAMILIES,
)

EXACT_JOIN_KEYS = (
    "target_row_id",
    "candidate_policy_id",
    "candidate_id",
    "clean_forward_candidate_id",
)

FORBIDDEN_STANDALONE_JOIN_KEYS = frozenset(
    {
        "pair_address",
        "event_timestamp",
        "observed_at",
        "fetched_at",
        "ingested_at",
    }
)

RELEVANT_NAME_TOKENS = (
    "direct_target",
    "xgb",
    "rf",
    "random_forest",
    "tab",
    "tabicl",
    "prediction",
    "predictions",
    "schema",
    "model",
    "manifest",
    "policy",
    "consensus",
    "preprocessing",
)

SCORE_COL_HINTS = ("predicted_probability", "tab_score", "score", "prob", "prediction")
RANK_COL_HINTS = ("rank", "percentile", "rank_pct")
ID_COL_HINTS = EXACT_JOIN_KEYS + ("pair_address", "event_timestamp")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _mtime_iso(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return ""


def _infer_family(path: Path) -> str:
    """Infer RF/XGB/TAB from filename first, then path — avoid parent-dir false positives.

    Important: E4 root ``phase_e4_direct_target_xgb_rf_full_*`` contains ``_xgb_`` in the
    directory name; classifying from the full path alone would mis-label every RF artifact
    as XGB.
    """
    name = path.name.lower()
    parent = path.parent.name.lower()
    text = str(path).replace("\\", "/").lower()

    # Filename-first (highest priority)
    if "tabicl" in name or name.startswith("direct_target_tab"):
        return "TAB"
    if name.startswith("direct_target_xgb") or "_xgb_" in name or name.startswith("xgb_"):
        return "XGB"
    if name.startswith("direct_target_rf") or "_rf_" in name or name.startswith("rf_"):
        return "RF"
    if "xgboost" in name:
        return "XGB"
    if "random_forest" in name:
        return "RF"

    # Parent folder hints (models/predictions file stems often include family)
    if "tabicl" in parent or parent in {"tab", "tabicl"}:
        return "TAB"

    # Consensus / comparison artifacts spanning families
    if "tab_xgb_rf" in name or "tab_xgb_rf" in text:
        return "UNKNOWN"

    # Path-level hints only when filename did not decide — require path segment boundaries
    parts = text.split("/")
    for part in parts:
        if part in {"xgb", "xgboost"} or part.startswith("xgb_"):
            return "XGB"
        if part in {"rf", "random_forest"} or part.startswith("rf_"):
            return "RF"
        if part in {"tab", "tabicl"} or part.startswith("tabicl"):
            return "TAB"

    if "tabicl" in text:
        return "TAB"
    return "UNKNOWN"


def _infer_artifact_type(path: Path) -> str:
    name = path.name.lower()
    parent = path.parent.name.lower()
    if "preprocessing" in name or name.endswith("_schema.json") or "feature" in name and name.endswith(".json"):
        return "schema"
    if path.suffix.lower() in {".joblib", ".pkl", ".pt", ".bin", ".ckpt"}:
        return "model"
    if "prediction" in name or parent == "predictions":
        return "prediction"
    if "policy_grid" in name:
        return "policy_grid"
    if "selected_polic" in name or "validation_selected" in name:
        return "selected_policy"
    if "consensus" in parent or "consensus" in name:
        return "consensus"
    if "manifest" in name:
        return "manifest"
    if "audit" in parent or "audit" in name:
        return "audit"
    if parent == "metrics" or "metrics" in name:
        return "metrics"
    if parent == "reports":
        return "report"
    return "unknown"


def _name_relevant(path: Path) -> bool:
    text = str(path).replace("\\", "/").lower()
    return any(tok in text for tok in RELEVANT_NAME_TOKENS)


def _probe_tabular_columns(path: Path) -> dict[str, Any]:
    """Header/schema probe without loading huge files fully."""
    out: dict[str, Any] = {
        "columns": [],
        "readable": False,
        "load_attempted": True,
        "load_success": False,
        "failure_reason": "",
        "row_count_estimate": "",
    }
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            # Only read header (+ avoid loading 11GB selected_trades)
            with path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, [])
            out["columns"] = [c.strip() for c in header if c is not None]
            out["readable"] = True
            out["load_success"] = True
            return out
        if suffix == ".parquet":
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            out["columns"] = list(pf.schema_arrow.names)
            out["row_count_estimate"] = pf.metadata.num_rows if pf.metadata else ""
            out["readable"] = True
            out["load_success"] = True
            return out
        if suffix == ".json":
            obj = json.loads(path.read_text(encoding="utf-8"))
            out["readable"] = True
            out["load_success"] = True
            if isinstance(obj, dict):
                feats = (
                    obj.get("feature_columns_in_order")
                    or obj.get("feature_columns")
                    or obj.get("feature_names_in_")
                    or []
                )
                if isinstance(feats, list):
                    out["columns"] = [str(x) for x in feats]
            return out
        if suffix == ".jsonl":
            out["readable"] = True
            out["load_success"] = True
            return out
        # binary model — existence-only probe
        out["readable"] = path.is_file() and path.stat().st_size > 0
        out["load_success"] = out["readable"]
        out["load_attempted"] = False  # do not deserialize weights here
        return out
    except Exception as exc:  # noqa: BLE001
        out["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return out


def _classify_columns(columns: list[str]) -> dict[str, list[str]]:
    low_map = {c: c.lower() for c in columns}
    def pick(hints: tuple[str, ...]) -> list[str]:
        found = []
        for c, low in low_map.items():
            if any(h in low for h in hints):
                found.append(c)
        return found

    return {
        "key_columns_detected": [c for c in columns if c in EXACT_JOIN_KEYS or c in ("pair_address", "event_timestamp")],
        "score_columns_detected": pick(SCORE_COL_HINTS),
        "rank_columns_detected": pick(RANK_COL_HINTS),
        "target_columns_detected": [c for c in columns if "target" in c.lower()],
        "policy_columns_detected": [
            c for c in columns if any(x in c.lower() for x in ("policy", "filter", "horizon", "exit_policy", "top_pct"))
        ],
        "timestamp_columns_detected": [
            c for c in columns if any(x in c.lower() for x in ("timestamp", "observed_at", "event_time", "fetched_at"))
        ],
        "pair_address_columns_detected": [c for c in columns if "pair" in c.lower() and "address" in c.lower()],
    }


def discover_direct_target_artifacts(
    project_root: Path,
    *,
    exclude_roots: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Discover RF/XGB/TAB direct-target artifacts with Path.exists() discipline."""
    rows: list[dict[str, Any]] = []
    canonical_status: list[dict[str, Any]] = []
    seen: set[str] = set()
    exclude = {e.replace("\\", "/").rstrip("/") for e in (exclude_roots or ())}

    def add_path(path: Path, *, artifact_root_source: str, expected_path: str = "") -> None:
        if not path.is_file():
            return
        if not _name_relevant(path) and path.suffix.lower() not in {".joblib", ".parquet", ".pkl"}:
            return
        rel = _rel(path, project_root)
        if rel in seen:
            return
        # Skip enormous full-load of selected_trades content; still record metadata + header
        size = path.stat().st_size
        probe = _probe_tabular_columns(path)
        cols = probe.get("columns") or []
        col_info = _classify_columns(cols)
        family = _infer_family(path)
        atype = _infer_artifact_type(path)
        usable = False
        unusable_reason = ""
        if atype == "prediction" and col_info["score_columns_detected"] and (
            any(k in cols for k in EXACT_JOIN_KEYS)
        ):
            usable = True  # usable as historical evidence IF exact join succeeds later
            unusable_reason = ""
        elif atype == "model" and path.suffix.lower() in {".joblib", ".pkl"}:
            usable = True  # usable for inference IF feature parity passes later
        elif atype == "schema" and cols:
            usable = True
        elif atype in {"selected_policy", "policy_grid"}:
            usable = True
        else:
            unusable_reason = "Not a primary prediction/model/schema/policy artifact for attachment"

        # Huge consensus membership tables are not loaded for CF join (wrong universe anyway)
        if size > 500_000_000 and atype == "consensus":
            usable = False
            unusable_reason = "Consensus membership table too large and not Clean-Forward-keyed; metadata-only"

        row = {
            "expected_path": expected_path,
            "discovered_path": rel,
            "path": rel,
            "path_exists": True,
            "file_size_bytes": size,
            "last_modified_time": _mtime_iso(path),
            "artifact_family": family,
            "artifact_type": atype,
            "readable": probe.get("readable"),
            "load_attempted": probe.get("load_attempted"),
            "load_success": probe.get("load_success"),
            "failure_reason": probe.get("failure_reason") or "",
            "columns": "|".join(cols[:80]),
            "column_count": len(cols),
            "row_count_estimate": probe.get("row_count_estimate"),
            **{k: "|".join(v) for k, v in col_info.items()},
            "schema_path": "",
            "direct_target_relevance": "high" if "direct_target" in rel.lower() else "medium",
            "usable_for_attachment": usable,
            "unusable_reason": unusable_reason,
            "artifact_root_source": artifact_root_source,
        }
        rows.append(row)
        seen.add(rel)

    # Canonical roots first
    for rel_root in CANONICAL_DIRECT_TARGET_ROOTS:
        root = project_root / rel_root
        exists = root.exists()
        canonical_status.append(
            {
                "expected_path": rel_root.replace("\\", "/"),
                "path_exists": exists,
                "is_dir": root.is_dir() if exists else False,
                "file_size_bytes": root.stat().st_size if exists and root.is_file() else "",
            }
        )
        if not exists:
            rows.append(
                {
                    "expected_path": rel_root.replace("\\", "/"),
                    "discovered_path": rel_root.replace("\\", "/"),
                    "path": rel_root.replace("\\", "/"),
                    "path_exists": False,
                    "file_size_bytes": "",
                    "last_modified_time": "",
                    "artifact_family": "UNKNOWN",
                    "artifact_type": "root",
                    "readable": False,
                    "load_attempted": False,
                    "load_success": False,
                    "failure_reason": "ARTIFACT_NOT_FOUND",
                    "columns": "",
                    "column_count": 0,
                    "usable_for_attachment": False,
                    "unusable_reason": "Canonical expected root missing",
                    "artifact_root_source": "CANONICAL_EXPECTED",
                    "direct_target_relevance": "high",
                }
            )
            continue
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    add_path(path, artifact_root_source="CANONICAL_EXPECTED", expected_path=rel_root)

    # Fallback parent scans for any additional direct-target roots not already covered
    fallback_parents = (
        "data/training/manual_verified_results",
        "data/training",
        "data/audits",
    )
    fallback_roots_used: list[str] = []
    for parent_rel in fallback_parents:
        parent = project_root / parent_rel
        if not parent.exists():
            continue
        # Find sibling direct_target phase dirs
        try:
            candidates = [p for p in parent.iterdir() if p.is_dir()]
        except OSError:
            continue
        for cand in candidates:
            rel = _rel(cand, project_root)
            if rel.replace("\\", "/") in {r.replace("\\", "/") for r in CANONICAL_DIRECT_TARGET_ROOTS}:
                continue
            name_l = cand.name.lower()
            if not any(tok in name_l for tok in ("direct_target", "tabicl", "xgb_rf", "ae16")):
                continue
            # Do not scan the live output root or prior completion roots mid-run
            if "ae16_model_evidence_bridge_completion_" in name_l:
                continue
            if rel.replace("\\", "/").rstrip("/") in exclude:
                continue
            fallback_roots_used.append(rel)
            for path in sorted(cand.rglob("*")):
                if path.is_file() and _name_relevant(path):
                    add_path(path, artifact_root_source="FALLBACK_DISCOVERY")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_expected_roots": canonical_status,
        "fallback_discovered_roots": fallback_roots_used,
        "artifact_row_count": len(rows),
        "counts_by_family": {},
        "counts_by_type": {},
        "usable_for_attachment_count": sum(1 for r in rows if r.get("usable_for_attachment")),
    }
    for fam in list(MODEL_FAMILIES) + ["UNKNOWN"]:
        manifest["counts_by_family"][fam] = sum(1 for r in rows if r.get("artifact_family") == fam)
    for r in rows:
        t = str(r.get("artifact_type") or "unknown")
        manifest["counts_by_type"][t] = manifest["counts_by_type"].get(t, 0) + 1
    return rows, manifest


def _load_id_set_from_artifact(path: Path, key: str, project_root: Path) -> tuple[set[str], str]:
    abs_path = path if path.is_absolute() else project_root / path
    if not abs_path.exists():
        return set(), "ARTIFACT_NOT_FOUND"
    try:
        if abs_path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq

            table = pq.read_table(abs_path, columns=[key])
            return {str(v) for v in table.column(key).to_pylist() if v is not None and str(v).strip()}, ""
        if abs_path.suffix.lower() == ".csv":
            # Cap rows for safety on huge CSVs
            ids: set[str] = set()
            with abs_path.open("r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                if key not in (reader.fieldnames or []):
                    return set(), f"key_absent:{key}"
                for i, row in enumerate(reader):
                    if i >= 2_000_000:
                        break
                    v = row.get(key)
                    if v is not None and str(v).strip():
                        ids.add(str(v).strip())
            return ids, ""
        return set(), "unsupported_format"
    except Exception as exc:  # noqa: BLE001
        return set(), f"{type(exc).__name__}: {exc}"


def audit_exact_id_joins(
    *,
    project_root: Path,
    candidates: list[dict[str, Any]],
    discovery_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Test whether any exact deterministic join exists to CF candidates."""
    cf_key_sets: dict[str, set[str]] = {
        "clean_forward_candidate_id": {str(r.get("clean_forward_candidate_id") or "") for r in candidates},
        "candidate_id": {str(r.get("clean_forward_candidate_id") or "") for r in candidates},
        "pair_address": {str(r.get("pair_address") or "") for r in candidates},
        "provider_payload_hash": {str(r.get("provider_payload_hash") or "") for r in candidates},
        "source_clean_forward_row_key": {str(r.get("source_clean_forward_row_key") or "") for r in candidates},
    }
    # CF does not carry these:
    for missing in ("target_row_id", "candidate_policy_id"):
        cf_key_sets[missing] = set()

    pred_rows = [
        r
        for r in discovery_rows
        if r.get("path_exists")
        and r.get("artifact_type") == "prediction"
        and r.get("load_success")
        and str(r.get("file_size_bytes") or 0).isdigit()
        and int(r.get("file_size_bytes") or 0) < 200_000_000  # skip huge tables for join sampling
    ]
    # Prefer one representative prediction per family from canonical E4/E5
    by_family: dict[str, list[dict[str, Any]]] = {f: [] for f in MODEL_FAMILIES}
    for r in pred_rows:
        fam = r.get("artifact_family")
        if fam in by_family:
            by_family[fam].append(r)

    audit_rows: list[dict[str, Any]] = []
    family_summary: dict[str, Any] = {}

    tested_keys = (
        "target_row_id",
        "candidate_policy_id",
        "candidate_id",
        "clean_forward_candidate_id",
        "pair_address",  # tested only to document rejection
    )

    for family in MODEL_FAMILIES:
        arts = by_family.get(family) or []
        if not arts:
            for key in tested_keys:
                audit_rows.append(
                    {
                        "model_family": family,
                        "artifact_path": "",
                        "tested_join_key": key,
                        "key_present_in_clean_forward": key in ("clean_forward_candidate_id", "candidate_id", "pair_address")
                        or bool(cf_key_sets.get(key)),
                        "key_present_in_artifact": False,
                        "clean_rows_matched": 0,
                        "artifact_rows_matched": 0,
                        "duplicate_key_count_clean": 0,
                        "duplicate_key_count_artifact": 0,
                        "ambiguous_match_count": 0,
                        "exact_join_safe": False,
                        "rejection_reason": "PREDICTION_FILE_NOT_FOUND",
                    }
                )
            family_summary[family] = {
                "exact_join_safe": False,
                "best_key": None,
                "matched_rows": 0,
                "rejection_reason": "PREDICTION_FILE_NOT_FOUND",
            }
            continue

        # Use up to 3 representative prediction files per family
        safe_any = False
        best = None
        for art in arts[:3]:
            path = project_root / str(art["path"])
            cols = str(art.get("columns") or "").split("|") if art.get("columns") else []
            for key in tested_keys:
                key_in_cf = key in ("clean_forward_candidate_id", "candidate_id", "pair_address", "provider_payload_hash", "source_clean_forward_row_key")
                # candidate_id on CF side only via alias to clean_forward_candidate_id
                if key == "candidate_id":
                    key_in_cf = True
                if key in ("target_row_id", "candidate_policy_id"):
                    key_in_cf = False
                key_in_art = key in cols or (
                    key == "clean_forward_candidate_id" and "candidate_id" in cols
                )
                art_key = "candidate_id" if key == "clean_forward_candidate_id" and "candidate_id" in cols else key

                if key in FORBIDDEN_STANDALONE_JOIN_KEYS or key == "pair_address":
                    # Document rejection even if overlap exists
                    matched = 0
                    if key_in_art and key == "pair_address":
                        art_ids, err = _load_id_set_from_artifact(path, "pair_address", project_root)
                        matched = len(art_ids & cf_key_sets["pair_address"]) if not err else 0
                    audit_rows.append(
                        {
                            "model_family": family,
                            "artifact_path": art["path"],
                            "tested_join_key": key,
                            "key_present_in_clean_forward": key_in_cf,
                            "key_present_in_artifact": key_in_art,
                            "clean_rows_matched": matched,
                            "artifact_rows_matched": matched,
                            "duplicate_key_count_clean": 0,
                            "duplicate_key_count_artifact": 0,
                            "ambiguous_match_count": 0,
                            "exact_join_safe": False,
                            "rejection_reason": "PAIR_TIMESTAMP_JOIN_REJECTED",
                        }
                    )
                    continue

                if not key_in_cf:
                    audit_rows.append(
                        {
                            "model_family": family,
                            "artifact_path": art["path"],
                            "tested_join_key": key,
                            "key_present_in_clean_forward": False,
                            "key_present_in_artifact": key_in_art,
                            "clean_rows_matched": 0,
                            "artifact_rows_matched": 0,
                            "duplicate_key_count_clean": 0,
                            "duplicate_key_count_artifact": 0,
                            "ambiguous_match_count": 0,
                            "exact_join_safe": False,
                            "rejection_reason": "POLICY_ID_NOT_AVAILABLE"
                            if key == "candidate_policy_id"
                            else ("TARGET_ROW_ID_NOT_AVAILABLE" if key == "target_row_id" else "EXACT_ID_JOIN_NOT_AVAILABLE"),
                        }
                    )
                    continue

                if not key_in_art:
                    audit_rows.append(
                        {
                            "model_family": family,
                            "artifact_path": art["path"],
                            "tested_join_key": key,
                            "key_present_in_clean_forward": True,
                            "key_present_in_artifact": False,
                            "clean_rows_matched": 0,
                            "artifact_rows_matched": 0,
                            "duplicate_key_count_clean": 0,
                            "duplicate_key_count_artifact": 0,
                            "ambiguous_match_count": 0,
                            "exact_join_safe": False,
                            "rejection_reason": "EXACT_ID_JOIN_NOT_AVAILABLE",
                        }
                    )
                    continue

                art_ids, err = _load_id_set_from_artifact(path, art_key, project_root)
                if err:
                    audit_rows.append(
                        {
                            "model_family": family,
                            "artifact_path": art["path"],
                            "tested_join_key": key,
                            "key_present_in_clean_forward": True,
                            "key_present_in_artifact": True,
                            "clean_rows_matched": 0,
                            "artifact_rows_matched": 0,
                            "duplicate_key_count_clean": 0,
                            "duplicate_key_count_artifact": 0,
                            "ambiguous_match_count": 0,
                            "exact_join_safe": False,
                            "rejection_reason": f"ARTIFACT_READ_ERROR:{err}",
                        }
                    )
                    continue

                cf_ids = cf_key_sets.get("candidate_id" if key in ("candidate_id", "clean_forward_candidate_id") else key) or set()
                overlap = art_ids & cf_ids
                safe = len(overlap) > 0 and key not in FORBIDDEN_STANDALONE_JOIN_KEYS and key != "pair_address"
                # Historical candidate_id hashes are a different namespace than clean_forward_candidate_id
                # even when both are named candidate_id — overlap must be non-zero AND key must be true shared lineage.
                if key in ("candidate_id", "clean_forward_candidate_id") and len(overlap) == 0:
                    safe = False
                audit_rows.append(
                    {
                        "model_family": family,
                        "artifact_path": art["path"],
                        "tested_join_key": key,
                        "key_present_in_clean_forward": True,
                        "key_present_in_artifact": True,
                        "clean_rows_matched": len(overlap),
                        "artifact_rows_matched": len(overlap),
                        "duplicate_key_count_clean": 0,
                        "duplicate_key_count_artifact": 0,
                        "ambiguous_match_count": 0,
                        "exact_join_safe": safe,
                        "rejection_reason": ""
                        if safe
                        else "EXACT_ID_JOIN_NOT_AVAILABLE:zero_overlap_disjoint_id_namespace",
                    }
                )
                if safe:
                    safe_any = True
                    best = {"key": key, "matched": len(overlap), "path": art["path"]}

        family_summary[family] = {
            "exact_join_safe": safe_any,
            "best_key": None if not best else best["key"],
            "matched_rows": 0 if not best else best["matched"],
            "rejection_reason": None
            if safe_any
            else "EXACT_ID_JOIN_NOT_AVAILABLE: Clean Forward IDs do not overlap historical direct-target IDs; pair/timestamp joins rejected",
        }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "any_exact_join_safe": any(v.get("exact_join_safe") for v in family_summary.values()),
        "by_family": family_summary,
        "notes": [
            "pair_address-only and pair_address+timestamp joins are rejected by AE16 policy",
            "clean_forward_candidate_id is a different hash namespace than training candidate_id",
            "Clean Forward package lacks candidate_policy_id and target_row_id",
        ],
    }
    return audit_rows, summary


def load_required_features_from_schema(schema_path: Path) -> list[str]:
    if not schema_path.exists():
        return []
    obj = json.loads(schema_path.read_text(encoding="utf-8"))
    feats = (
        obj.get("feature_columns_in_order")
        or obj.get("feature_columns")
        or obj.get("feature_names_in_")
        or []
    )
    return [str(x) for x in feats] if isinstance(feats, list) else []


def audit_feature_parity(
    *,
    project_root: Path,
    candidates: list[dict[str, Any]],
    discovery_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Compare Clean Forward available fields to model-required features."""
    cf_cols = set(candidates[0].keys()) if candidates else set()
    # Build sparse feature matrix from mappable CF fields only (no invented values).
    feature_matrix_rows: list[dict[str, Any]] = []
    for cand in candidates:
        row: dict[str, Any] = {
            "clean_forward_candidate_id": cand.get("clean_forward_candidate_id"),
            "pair_address": cand.get("pair_address"),
        }
        # Direct / aliased mappings — leave unmapped required features absent (null).
        if "price_usd" in cand:
            row["price_usd"] = cand.get("price_usd")
            row["price"] = cand.get("price_usd")
        if "liquidity_usd" in cand:
            row["liquidity_usd"] = cand.get("liquidity_usd")
            row["liquidity"] = cand.get("liquidity_usd")
        if "volume_24h" in cand:
            row["volume_24h"] = cand.get("volume_24h")
        if "txns_buys_24h" in cand:
            row["txns_buys"] = cand.get("txns_buys_24h")
        if "txns_sells_24h" in cand:
            row["txns_sells"] = cand.get("txns_sells_24h")
        for pc in ("price_change_m5", "price_change_h1", "price_change_h6", "price_change_h24"):
            if pc in cand:
                row[pc] = cand.get(pc)
        # Derived only when both inputs present (not invented)
        try:
            buys = float(cand.get("txns_buys_24h") or "")
            sells = float(cand.get("txns_sells_24h") or "")
            row["txns_total"] = buys + sells
            row["buy_ratio"] = buys / (buys + sells) if (buys + sells) > 0 else None
        except (TypeError, ValueError):
            row["txns_total"] = None
            row["buy_ratio"] = None
        try:
            vol = float(cand.get("volume_24h") or "")
            liq = float(cand.get("liquidity_usd") or "")
            row["volume_to_liquidity_ratio"] = (vol / liq) if liq else None
        except (TypeError, ValueError):
            row["volume_to_liquidity_ratio"] = None
        feature_matrix_rows.append(row)

    available_feature_names = set()
    for r in feature_matrix_rows[:1]:
        available_feature_names = {k for k, v in r.items() if k not in {"clean_forward_candidate_id", "pair_address"} and v is not None and str(v).strip() != ""}
    # Also count columns that exist even if some rows null
    if feature_matrix_rows:
        available_feature_names = {
            k
            for k in feature_matrix_rows[0].keys()
            if k not in {"clean_forward_candidate_id", "pair_address"}
        }

    parity_rows: list[dict[str, Any]] = []
    compat_rows: list[dict[str, Any]] = []
    family_result: dict[str, Any] = {}

    for family in MODEL_FAMILIES:
        schema_rows = [
            r
            for r in discovery_rows
            if r.get("artifact_family") == family
            and r.get("artifact_type") == "schema"
            and r.get("path_exists")
        ]
        model_rows = [
            r
            for r in discovery_rows
            if r.get("artifact_family") == family
            and r.get("artifact_type") == "model"
            and r.get("path_exists")
            and str(r.get("path", "")).endswith((".joblib", ".pkl", ".pt"))
        ]

        schema_path = schema_rows[0]["path"] if schema_rows else ""
        model_path = model_rows[0]["path"] if model_rows else ""
        schema_exists = bool(schema_path) and (project_root / schema_path).exists()
        model_exists = bool(model_path) and (project_root / model_path).exists()

        required: list[str] = []
        if schema_exists:
            required = load_required_features_from_schema(project_root / schema_path)

        # TAB often has preprocessing JSON under metrics, not models/
        if not required and family == "TAB":
            tab_schemas = [
                r
                for r in discovery_rows
                if r.get("artifact_family") == "TAB"
                and "preprocessing" in str(r.get("path", "")).lower()
                and r.get("path_exists")
            ]
            if tab_schemas:
                schema_path = tab_schemas[0]["path"]
                schema_exists = (project_root / schema_path).exists()
                if schema_exists:
                    required = load_required_features_from_schema(project_root / schema_path)

        missing = [f for f in required if f not in available_feature_names]
        extra = sorted(available_feature_names - set(required)) if required else sorted(available_feature_names)
        parity_passed = bool(required) and len(missing) == 0
        inference_allowed = parity_passed and model_exists and family in ("RF", "XGB")

        blocker = ""
        if not schema_exists and family != "TAB":
            blocker = "SCHEMA_ARTIFACT_NOT_FOUND"
        elif not required:
            blocker = "SCHEMA_ARTIFACT_NOT_FOUND" if not schema_exists else "ARTIFACT_SCHEMA_UNSUPPORTED"
        elif missing:
            blocker = "FEATURE_PARITY_NOT_APPROVED"
        elif family == "TAB" and not model_exists:
            # TabICL has no per-dataset joblib; needs foundation ckpt + context construction features
            blocker = "MODEL_ARTIFACT_NOT_FOUND:TabICL requires context/runtime not available for CF inference"
            inference_allowed = False
        elif not model_exists:
            blocker = "MODEL_ARTIFACT_NOT_FOUND"
            inference_allowed = False

        parity_rows.append(
            {
                "model_family": family,
                "model_artifact_path": model_path,
                "model_artifact_exists": model_exists,
                "schema_artifact_path": schema_path,
                "schema_artifact_exists": schema_exists,
                "required_feature_count": len(required),
                "available_feature_count": len([f for f in required if f in available_feature_names]),
                "missing_required_features": "|".join(missing),
                "extra_clean_forward_features": "|".join(extra[:40]),
                "dtype_compatibility_status": "NOT_EVALUATED_PARITY_FAILED" if missing else "PENDING_INFERENCE",
                "categorical_compatibility_status": "N/A_NUMERIC_SCHEMA",
                "preprocessing_compatibility_status": "NOT_APPROVED" if missing else "SCHEMA_COMPATIBLE",
                "leakage_status": "NO_LEAKAGE_FIELDS_INJECTED",
                "feature_parity_passed": parity_passed,
                "inference_allowed": inference_allowed,
                "blocker_reason": blocker,
            }
        )
        compat_rows.append(
            {
                "model_family": family,
                "model_artifact_path": model_path,
                "model_artifact_exists": model_exists,
                "schema_artifact_path": schema_path,
                "schema_artifact_exists": schema_exists,
                "compatible_for_cf_inference": inference_allowed,
                "feature_parity_passed": parity_passed,
                "missing_required_feature_count": len(missing),
                "blocker_reason": blocker,
            }
        )
        family_result[family] = {
            "feature_parity_passed": parity_passed,
            "inference_allowed": inference_allowed,
            "inference_run": False,
            "missing_required_feature_count": len(missing),
            "required_feature_count": len(required),
            "available_feature_count": len([f for f in required if f in available_feature_names]),
            "model_artifact_exists": model_exists,
            "schema_artifact_exists": schema_exists,
            "blocker_reason": blocker,
        }

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "any_feature_parity_passed": any(v["feature_parity_passed"] for v in family_result.values()),
        "any_inference_allowed": any(v["inference_allowed"] for v in family_result.values()),
        "by_family": family_result,
        "clean_forward_mappable_fields": sorted(available_feature_names),
        "cf_source_columns": sorted(cf_cols),
        "cf_to_model_map": CF_TO_MODEL_FEATURE_MAP,
    }
    return parity_rows, compat_rows, feature_matrix_rows, summary


def build_attachment_v2(
    *,
    candidates: list[dict[str, Any]],
    decision_by_candidate: dict[str, dict[str, Any]],
    join_summary: dict[str, Any],
    parity_summary: dict[str, Any],
    discovery_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Emit per-candidate × family attachment rows (fail-closed)."""
    attachments: list[dict[str, Any]] = []
    vote_policy_rows: list[dict[str, Any]] = []

    # Vote policies exist in artifacts but cannot be applied without attached scores/ranks.
    policy_arts = [
        r
        for r in discovery_rows
        if r.get("artifact_type") in {"selected_policy", "policy_grid"} and r.get("path_exists")
    ]
    policy_available = bool(policy_arts)

    for family in MODEL_FAMILIES:
        join_info = (join_summary.get("by_family") or {}).get(family) or {}
        parity_info = (parity_summary.get("by_family") or {}).get(family) or {}

        if join_info.get("exact_join_safe"):
            status = "MODEL_EVIDENCE_ATTACHED"  # would attach — not reached in current data
            source_type = "EXACT_ID_PREDICTION_JOIN"
            reason = ""
        elif parity_info.get("inference_allowed"):
            status = "MODEL_EVIDENCE_ATTACHED"
            source_type = "EXISTING_MODEL_INFERENCE"
            reason = ""
        elif parity_info.get("model_artifact_exists") and not parity_info.get("feature_parity_passed"):
            status = "FEATURE_PARITY_NOT_APPROVED"
            source_type = "UNAVAILABLE"
            reason = parity_info.get("blocker_reason") or "FEATURE_PARITY_NOT_APPROVED"
        elif not join_info.get("exact_join_safe"):
            status = "EXACT_ID_JOIN_NOT_AVAILABLE"
            source_type = "UNAVAILABLE"
            # Prefer more specific if model missing entirely
            if not parity_info.get("model_artifact_exists") and family == "TAB":
                status = "EXACT_ID_JOIN_NOT_AVAILABLE"
                reason = (
                    "No exact-ID overlap with historical TAB predictions; "
                    "TabICL CF inference blocked (no compatible local CF feature/context path)"
                )
            else:
                reason = join_info.get("rejection_reason") or "EXACT_ID_JOIN_NOT_AVAILABLE"
            if parity_info.get("model_artifact_exists") and not parity_info.get("feature_parity_passed"):
                status = "FEATURE_PARITY_NOT_APPROVED"
                reason = (
                    f"Exact-ID join unavailable; inference blocked: "
                    f"{parity_info.get('blocker_reason')}; "
                    f"missing_features={parity_info.get('missing_required_feature_count')}/"
                    f"{parity_info.get('required_feature_count')}"
                )
        else:
            status = "MODEL_EVIDENCE_UNAVAILABLE"
            source_type = "UNAVAILABLE"
            reason = "MODEL_EVIDENCE_UNAVAILABLE"

        vote_policy_rows.append(
            {
                "model_family": family,
                "policy_artifact_available": policy_available,
                "policy_artifact_path": policy_arts[0]["path"] if policy_arts else "",
                "vote_policy_source": "POLICY_UNAVAILABLE"
                if not policy_available
                else "VALIDATION_SELECTED_POLICY_PRESENT_BUT_NOT_APPLICABLE_WITHOUT_SCORES",
                "vote_threshold": "",
                "scores_attached": False,
                "votes_allowed": False,
                "note": "Historical top_pct/pair_cap policies exist but cannot create CF votes without attached scores",
            }
        )

        # Representative source paths for auditability
        pred = next(
            (
                r
                for r in discovery_rows
                if r.get("artifact_family") == family and r.get("artifact_type") == "prediction" and r.get("path_exists")
            ),
            None,
        )
        model = next(
            (
                r
                for r in discovery_rows
                if r.get("artifact_family") == family
                and r.get("artifact_type") == "model"
                and str(r.get("path", "")).endswith((".joblib", ".pkl"))
                and r.get("path_exists")
            ),
            None,
        )
        schema = next(
            (
                r
                for r in discovery_rows
                if r.get("artifact_family") == family and r.get("artifact_type") == "schema" and r.get("path_exists")
            ),
            None,
        )

        for cand in candidates:
            cid = str(cand.get("clean_forward_candidate_id") or "")
            dec = decision_by_candidate.get(cid) or {}
            attachments.append(
                {
                    "clean_forward_candidate_id": cid,
                    "clean_forward_decision_input_id": dec.get("clean_forward_decision_input_id") or "",
                    "pair_address": cand.get("pair_address") or "",
                    "base_token_address": cand.get("base_token_address") or "",
                    "quote_token_address": cand.get("quote_token_address") or "",
                    "provider_pair_url": cand.get("provider_pair_url") or "",
                    "provider_payload_hash": cand.get("provider_payload_hash") or "",
                    "model_family": family,
                    "evidence_attached": False,
                    "evidence_source_type": source_type,
                    "score": None,
                    "rank": None,
                    "percentile_rank": None,
                    "model_vote": None,
                    "vote_policy_source": "POLICY_UNAVAILABLE",
                    "vote_threshold": "",
                    "source_artifact_path": (pred or model or {}).get("path", "") if (pred or model) else "",
                    "source_prediction_file": pred["path"] if pred else "",
                    "source_model_artifact": model["path"] if model else "",
                    "source_schema_artifact": schema["path"] if schema else "",
                    "source_run_id": "",
                    "candidate_policy_id": "",
                    "target_row_id": "",
                    "target_name": "",
                    "target_version": "",
                    "horizon": "",
                    "filter_name": "",
                    "exit_policy_id": "",
                    "attachment_status": status,
                    "attachment_failure_reason": reason,
                }
            )

    return attachments, vote_policy_rows


def decide_completion_classification(
    *,
    join_summary: dict[str, Any],
    parity_summary: dict[str, Any],
    attachments: list[dict[str, Any]],
    invented_ok: bool,
    legacy_ok: bool,
    authority_ok: bool,
) -> str:
    if not invented_ok:
        return "AE16_BLOCKED_INVENTED_OR_DEFAULTED_SCORES"
    if not legacy_ok:
        return "AE16_BLOCKED_LEGACY_CONTAMINATION"
    if not authority_ok:
        return "AE16_BLOCKED_AUTHORITY_ESCALATION"

    attached = [a for a in attachments if a.get("evidence_attached") and a.get("attachment_status") == "MODEL_EVIDENCE_ATTACHED"]
    if attached:
        fams = {a["model_family"] for a in attached}
        # Without valid votes, cannot claim full pass with tiers — handled by caller
        if fams >= {"RF", "XGB", "TAB"}:
            return "AE16_TIERED_CONSENSUS_ENGINE_PASS_WITH_MODEL_EVIDENCE"
        return "AE16_TIERED_CONSENSUS_ENGINE_PARTIAL_PASS_WITH_MODEL_EVIDENCE"

    # No attachments — choose most specific blocker
    parity_by = parity_summary.get("by_family") or {}
    models_exist = any(v.get("model_artifact_exists") for v in parity_by.values())
    parity_failed = any(
        v.get("model_artifact_exists") and not v.get("feature_parity_passed") for v in parity_by.values()
    )
    if parity_failed and models_exist:
        return "AE16_BLOCKED_FEATURE_PARITY_GAP"
    if not join_summary.get("any_exact_join_safe") and not parity_summary.get("any_inference_allowed"):
        return "AE16_BLOCKED_NO_COMPATIBLE_MODEL_EVIDENCE"
    return "AE16_BLOCKED_NO_COMPATIBLE_MODEL_EVIDENCE"
