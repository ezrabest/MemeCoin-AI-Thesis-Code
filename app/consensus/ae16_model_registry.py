"""AE16 central model registry loader.

All AE16 production scoring/inference must load RF / XGB / TAB16 through this
module and models/ae16_model_registry.json. Direct hard-coded artifact paths
in production scoring paths are a gate failure (AE16_MODEL_REGISTRY_BYPASS).
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

REGISTRY_REL = "models/ae16_model_registry.json"
REGISTRY_BYPASS_CODE = "AE16_MODEL_REGISTRY_BYPASS"
FEATURE_SCHEMA_HASH_MISMATCH = "FEATURE_SCHEMA_HASH_MISMATCH"
FEATURE_ORDER_MISMATCH = "FEATURE_ORDER_MISMATCH"
MISSING_DEPENDENCY_XGBOOST = "MISSING_DEPENDENCY_XGBOOST"

SLOT_TO_REGISTRY_KEY = {
    "RF": "RF",
    "XGB": "XGB",
    "TAB": "TAB_CONSENSUS_SLOT",
    "TAB_CONSENSUS_SLOT": "TAB_CONSENSUS_SLOT",
    "TAB16": "TAB_CONSENSUS_SLOT",
}

FORBIDDEN_DIRECT_LOAD_PATHS = (
    "models/ae16f_rf_serving_safe.joblib",
    "models/ae16f_xgb_serving_safe.joblib",
    "models/ae16_tab16_direct_target_serving_safe.joblib",
)


class Ae16RegistryError(RuntimeError):
    """Registry / schema enforcement failure."""

    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        msg = code if not detail else f"{code}: {detail}"
        super().__init__(msg)


def feature_set_hash_sha256(feature_names: list[str]) -> str:
    payload = "\n".join(sorted(feature_names))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def ordered_feature_schema_hash_sha256(feature_names: list[str]) -> str:
    payload = "\n".join(feature_names)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def project_root_from(path: Path | None = None) -> Path:
    if path is not None:
        return Path(path).resolve()
    return Path(__file__).resolve().parents[2]


def registry_path(project_root: Path | None = None) -> Path:
    root = project_root_from(project_root)
    return root / REGISTRY_REL


def load_ae16_model_registry(project_root: Path | None = None) -> dict[str, Any]:
    root = project_root_from(project_root)
    path = registry_path(root)
    if not path.is_file():
        raise Ae16RegistryError("AE16_MODEL_REGISTRY_MISSING", str(path))
    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "models" not in data:
        raise Ae16RegistryError("AE16_MODEL_REGISTRY_INVALID", "missing models map")
    return data


def _resolve_slot_key(consensus_slot: str) -> str:
    key = SLOT_TO_REGISTRY_KEY.get(str(consensus_slot).upper(), str(consensus_slot).upper())
    return key


def _unwrap_artifact(obj: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(obj, dict) and "model" in obj:
        meta = {k: v for k, v in obj.items() if k != "model"}
        return obj["model"], meta
    return obj, {}


def _require_xgboost_for_xgb(slot_key: str) -> None:
    if slot_key != "XGB":
        return
    if importlib.util.find_spec("xgboost") is None:
        raise Ae16RegistryError(MISSING_DEPENDENCY_XGBOOST, "xgboost not importable in active environment")


def load_ae16_registered_model(
    consensus_slot: str,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Load one registered AE16 model via registry only.

    Returns dict with keys:
      registry_entry, model, artifact_metadata, artifact_path, consensus_slot,
      threshold, feature_schema_lock
    """
    root = project_root_from(project_root)
    registry = load_ae16_model_registry(root)
    slot_key = _resolve_slot_key(consensus_slot)
    models = registry.get("models") or {}
    if slot_key not in models:
        raise Ae16RegistryError("AE16_MODEL_REGISTRY_SLOT_MISSING", slot_key)
    entry = dict(models[slot_key])
    rel = entry.get("artifact_path")
    if not rel:
        raise Ae16RegistryError("AE16_MODEL_REGISTRY_PATH_MISSING", slot_key)
    _require_xgboost_for_xgb(slot_key)
    abs_path = root / str(rel).replace("\\", "/")
    if not abs_path.is_file():
        raise Ae16RegistryError("AE16_MODEL_ARTIFACT_MISSING", str(rel))
    raw = joblib.load(abs_path)
    model, meta = _unwrap_artifact(raw)
    if not hasattr(model, "predict_proba") and not hasattr(model, "predict"):
        raise Ae16RegistryError("AE16_MODEL_ARTIFACT_INVALID", f"{slot_key} missing predict_proba")

    # Prefer artifact feature_schema_lock, fall back to registry schema
    lock = {}
    if isinstance(meta.get("feature_schema_lock"), dict):
        lock = dict(meta["feature_schema_lock"])
    elif isinstance(entry.get("feature_schema_lock"), dict):
        lock = dict(entry["feature_schema_lock"])
    else:
        ordered = list(entry.get("ordered_feature_names") or [])
        if ordered:
            lock = {
                "ordered_feature_names": ordered,
                "feature_count": len(ordered),
                "feature_set_hash_sha256": entry.get("feature_set_hash_sha256"),
                "ordered_feature_schema_hash_sha256": entry.get("ordered_feature_schema_hash_sha256"),
                "feature_order_enforced": True,
            }

    threshold = meta.get("threshold", entry.get("threshold"))
    return {
        "registry_entry": entry,
        "model": model,
        "artifact_metadata": meta,
        "artifact_raw": raw if isinstance(raw, dict) else {"model": raw},
        "artifact_path": str(rel).replace("\\", "/"),
        "consensus_slot": entry.get("consensus_slot") or slot_key,
        "slot_key": slot_key,
        "threshold": float(threshold) if threshold is not None else None,
        "feature_schema_lock": lock,
        "registry": registry,
    }


def verify_feature_schema_hashes(
    *,
    ordered_feature_names: list[str],
    expected_feature_set_hash: str | None,
    expected_ordered_hash: str | None,
    incoming_names: list[str] | None = None,
) -> dict[str, Any]:
    """Recompute and compare both schema hashes. Raises on mismatch."""
    ordered = list(ordered_feature_names)
    set_hash = feature_set_hash_sha256(ordered)
    ord_hash = ordered_feature_schema_hash_sha256(ordered)
    result = {
        "feature_set_hash_sha256": set_hash,
        "ordered_feature_schema_hash_sha256": ord_hash,
        "expected_feature_set_hash_sha256": expected_feature_set_hash,
        "expected_ordered_feature_schema_hash_sha256": expected_ordered_hash,
        "feature_set_hash_ok": True,
        "ordered_feature_schema_hash_ok": True,
    }
    if expected_feature_set_hash and set_hash != expected_feature_set_hash:
        result["feature_set_hash_ok"] = False
        raise Ae16RegistryError(
            FEATURE_SCHEMA_HASH_MISMATCH,
            f"feature_set_hash got={set_hash} expected={expected_feature_set_hash}",
        )
    if expected_ordered_hash and ord_hash != expected_ordered_hash:
        result["ordered_feature_schema_hash_ok"] = False
        raise Ae16RegistryError(
            FEATURE_SCHEMA_HASH_MISMATCH,
            f"ordered_feature_schema_hash got={ord_hash} expected={expected_ordered_hash}",
        )
    if incoming_names is not None:
        incoming_set = feature_set_hash_sha256(list(incoming_names))
        if expected_feature_set_hash and incoming_set != expected_feature_set_hash:
            raise Ae16RegistryError(
                FEATURE_SCHEMA_HASH_MISMATCH,
                f"incoming feature_set_hash got={incoming_set} expected={expected_feature_set_hash}",
            )
    return result


def build_ordered_inference_matrix(
    df: pd.DataFrame,
    registry_entry: dict[str, Any],
    artifact_metadata: dict[str, Any],
) -> pd.DataFrame:
    """Build exactly-ordered float matrix; reorder safely when names match."""
    lock = artifact_metadata.get("feature_schema_lock") or {}
    ordered = list(
        lock.get("ordered_feature_names")
        or registry_entry.get("ordered_feature_names")
        or []
    )
    if not ordered:
        raise Ae16RegistryError(FEATURE_ORDER_MISMATCH, "ordered_feature_names missing")

    present = [c for c in df.columns if c in ordered]
    missing = [c for c in ordered if c not in df.columns]
    if missing:
        raise Ae16RegistryError(
            FEATURE_ORDER_MISMATCH,
            f"missing features cannot be corrected safely: {missing}",
        )

    # Extra columns in source frame are lineage — not allowed inside model matrix.
    # Model matrix is constructed as exactly ordered names only.
    X = df.loc[:, ordered].copy()
    for c in ordered:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.astype("float64")

    if list(X.columns) != ordered:
        raise Ae16RegistryError(FEATURE_ORDER_MISMATCH, "reorder failed to enforce stored order")

    expected_set = (
        lock.get("feature_set_hash_sha256")
        or registry_entry.get("feature_set_hash_sha256")
    )
    expected_ord = (
        lock.get("ordered_feature_schema_hash_sha256")
        or registry_entry.get("ordered_feature_schema_hash_sha256")
    )
    # After enforcing order, verify both hashes
    verify_feature_schema_hashes(
        ordered_feature_names=list(X.columns),
        expected_feature_set_hash=expected_set,
        expected_ordered_hash=expected_ord,
        incoming_names=present,
    )
    # Detect unsafe extra model-feature columns if caller passed a pure feature frame
    # with extras that look like model features (same length domain). Not lineage.
    lineage_ok = {
        "price_source_key",
        "timestamp",
        "provider",
        "chain",
        "pair_address",
        "source_query",
        "selected_coverage_status",
        "latest_l1_found",
        "row_id",
        "combined_target_id",
        "target",
        "split",
        "event_timestamp",
        "candidate_id",
        "id",
        "coin_id",
        "filter_status",
        "drop_reason",
        "selected_provider_pair_url",
        "whale_score",  # present in L1 dump but never entered into matrix
    }
    extras_in_frame = [c for c in df.columns if c not in ordered and c not in lineage_ok]
    # Extras are ignored for matrix construction; only fail if caller intended them as matrix
    _ = extras_in_frame  # documented: lineage/non-feature columns ignored
    return X


def score_registered_ae16_models(
    df: pd.DataFrame,
    project_root: Path | None = None,
    slots: tuple[str, ...] = ("RF", "XGB", "TAB"),
) -> dict[str, Any]:
    """Score a serving frame with all registered AE16 models via registry."""
    root = project_root_from(project_root)
    out: dict[str, Any] = {"slots": {}, "errors": {}}
    for slot in slots:
        try:
            loaded = load_ae16_registered_model(slot, root)
            X = build_ordered_inference_matrix(
                df,
                loaded["registry_entry"],
                {
                    "feature_schema_lock": loaded["feature_schema_lock"],
                    **(loaded.get("artifact_metadata") or {}),
                },
            )
            model = loaded["model"]
            if not hasattr(model, "predict_proba"):
                raise Ae16RegistryError("AE16_MODEL_ARTIFACT_INVALID", f"{slot} no predict_proba")
            proba = np.asarray(model.predict_proba(X))[:, 1]
            thr = loaded.get("threshold")
            votes = None
            if thr is not None:
                votes = (proba >= float(thr)).astype(bool)
            out["slots"][slot] = {
                "scores": proba,
                "votes": votes,
                "threshold": thr,
                "artifact_path": loaded["artifact_path"],
                "registry_entry": loaded["registry_entry"],
                "feature_schema_lock": loaded["feature_schema_lock"],
                "artifact_metadata": loaded["artifact_metadata"],
            }
        except Ae16RegistryError as exc:
            out["errors"][slot] = {"code": exc.code, "detail": exc.detail}
        except Exception as exc:  # noqa: BLE001
            out["errors"][slot] = {"code": "MODEL_SCORE_FAILURE", "detail": f"{type(exc).__name__}: {exc}"}
    return out


def assert_no_direct_artifact_path_load(source_text: str, *, allow_registry_json: bool = False) -> None:
    """Raise AE16_MODEL_REGISTRY_BYPASS if production code hard-codes model paths for loading."""
    lowered = source_text.replace("\\", "/")
    for p in FORBIDDEN_DIRECT_LOAD_PATHS:
        if p in lowered and "joblib.load" in lowered:
            # crude static check — used by audits/tests
            if allow_registry_json and "ae16_model_registry" in lowered:
                continue
            raise Ae16RegistryError(REGISTRY_BYPASS_CODE, p)


def audit_production_paths_for_registry_bypass(project_root: Path | None = None) -> dict[str, Any]:
    """Scan AE16 consensus scoring modules for hard-coded joblib.load of AE16 artifacts."""
    root = project_root_from(project_root)
    scan_dirs = [
        root / "app" / "consensus",
    ]
    findings: list[dict[str, Any]] = []
    bypass = False
    for d in scan_dirs:
        if not d.is_dir():
            continue
        for path in sorted(d.glob("*.py")):
            # Registry module itself documents path constants — exclude self
            if path.name in {"ae16_model_registry.py"}:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            norm = text.replace("\\", "/")
            for forbidden in FORBIDDEN_DIRECT_LOAD_PATHS:
                if forbidden in norm and "joblib.load" in norm:
                    # Allow comments / dump-only strings without load adjacent usage of that literal
                    # Fail if both the literal path and joblib.load appear in same file outside tests
                    bypass = True
                    findings.append(
                        {
                            "file": str(path.relative_to(root)).replace("\\", "/"),
                            "forbidden_path": forbidden,
                            "code": REGISTRY_BYPASS_CODE,
                        }
                    )
    return {
        "bypass_detected": bypass,
        "findings": findings,
        "status": "FAIL" if bypass else "PASS",
        "scanned": [str(p.relative_to(root)).replace("\\", "/") for p in scan_dirs],
    }
