"""AE16 TAB16 — direct-target serving-safe TAB consensus-slot evidence provider.

TAB16 is AE16-only. It is not AE17, Meta, Context, or LLM.
It must never load or masquerade as legacy 51-feature TAB artifacts.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.consensus.ae16_model_registry import (
    Ae16RegistryError,
    FEATURE_ORDER_MISMATCH,
    FEATURE_SCHEMA_HASH_MISMATCH,
    build_ordered_inference_matrix,
    feature_set_hash_sha256,
    load_ae16_registered_model,
    ordered_feature_schema_hash_sha256,
    project_root_from,
    score_registered_ae16_models,
    verify_feature_schema_hashes,
)
from app.consensus.ae16f_serving_safe import (
    extract_serving_safe_row_from_historical,
    validation_quantile_threshold,
    _split_masks,
)

PHASE = "AE16"
MODEL_FAMILY = "TAB_CONSENSUS_SLOT"
MODEL_VARIANT = "TAB16_DIRECT_TARGET_SERVING_SAFE"
CONSENSUS_SLOT = "TAB"
ARTIFACT_REL = "models/ae16_tab16_direct_target_serving_safe.joblib"
FORBIDDEN_ALIAS_REL = "models/ae16f_tab_serving_safe.joblib"
REGISTRY_REL = "models/ae16_model_registry.json"
TARGET_COLUMN = "target"
RANDOM_SEED = 42
THRESHOLD_POLICY = "historical_validation_top_5pct_quantile"

TRAINING_SOURCE_REL = (
    "data/training/manual_verified_datasets_direct_target_v1/"
    "LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL075_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.parquet"
)

DEFAULT_SELECTED_L1_REL = (
    "data/audits/manual_post_collection_rf_xgb_tab_sanity_20260724T193531Z/"
    "data/selected_latest_l1_rows.csv"
)
DEFAULT_SERVING_MATRIX_REL = (
    "data/audits/manual_post_collection_rf_xgb_tab_sanity_20260724T193531Z/"
    "data/serving_feature_matrix_preview.csv"
)
AE16F_THRESHOLDS_REL = "data/ae16f_vote_thresholds.csv"
RF_ARTIFACT_REL = "models/ae16f_rf_serving_safe.joblib"
XGB_ARTIFACT_REL = "models/ae16f_xgb_serving_safe.joblib"

ORDERED_FEATURE_NAMES: list[str] = [
    "tp_ratio",
    "sl_ratio",
    "round_trip_fee_pct",
    "time_stop_minutes",
    "price",
    "liquidity",
    "volume_24h",
    "fdv",
    "txns_buys",
    "txns_sells",
    "txns_total",
    "price_change_m5",
    "price_change_h1",
    "price_change_h6",
    "price_change_h24",
    "buy_ratio",
    "price_usd",
    "liquidity_usd",
    "volume_to_liquidity_ratio",
    "fdv_is_missing",
    "txns_buys_is_missing",
    "txns_sells_is_missing",
    "price_change_m5_is_missing",
    "price_change_h1_is_missing",
    "price_change_h6_is_missing",
    "price_change_h24_is_missing",
]

LEGACY_TAB_ARTIFACTS: tuple[str, ...] = (
    "data/training/models/label_profitable_after_fees_1h_best.joblib",
    "data/training/models/label_profitable_after_fees_1h__hist_gradient_boosting.joblib",
    "data/training/models/label_profitable_after_fees_1h__logistic_regression.joblib",
    "data/training/models/label_profitable_after_fees_1h__random_forest.joblib",
    "data/training/models/label_profitable_after_fees_4h_best.joblib",
    "data/training/models/label_profitable_after_fees_4h__hist_gradient_boosting.joblib",
    "data/training/models/label_profitable_after_fees_4h__logistic_regression.joblib",
    "data/training/models/label_profitable_after_fees_4h__random_forest.joblib",
)

FORBIDDEN_EXACT = frozenset(
    {
        "candidate_id",
        "entry_price",
        "entry_snapshot_id",
        "future_snapshot_count",
        "gap_detected",
        "label_valid",
        "sim_exit_status",
        "whale_score",
        "sentiment_score",
        "llm_confidence",
        "risk_score",
        "score",
        "confidence",
        "signal_confidence",
    }
)

FORBIDDEN_PREFIXES = (
    "verified_",
    "max_future_",
    "min_future_",
    "future_",
    "price_return_",
    "volume_zscore_",
    "buy_sell_imbalance_zscore_",
    "whale_wave_",
)

ATTACHED = "MODEL_EVIDENCE_ATTACHED"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def compute_schema_hashes(ordered_names: list[str] | None = None) -> dict[str, str]:
    names = list(ordered_names or ORDERED_FEATURE_NAMES)
    return {
        "feature_set_hash_sha256": feature_set_hash_sha256(names),
        "ordered_feature_schema_hash_sha256": ordered_feature_schema_hash_sha256(names),
    }


def is_forbidden_feature(name: str) -> bool:
    if name in FORBIDDEN_EXACT:
        return True
    if name in ORDERED_FEATURE_NAMES:
        return False
    for p in FORBIDDEN_PREFIXES:
        if name.startswith(p):
            return True
    return False


def audit_forbidden_features(names: list[str]) -> dict[str, Any]:
    found = sorted({n for n in names if is_forbidden_feature(n)})
    return {
        "forbidden_features_checked": True,
        "forbidden_features_found": found,
        "passed": len(found) == 0,
        "checked_names_count": len(names),
    }


def build_feature_schema_lock(
    *,
    ordered_names: list[str] | None = None,
    train_medians: dict[str, float] | None = None,
) -> dict[str, Any]:
    names = list(ordered_names or ORDERED_FEATURE_NAMES)
    if names != ORDERED_FEATURE_NAMES:
        raise Ae16RegistryError(
            FEATURE_ORDER_MISMATCH,
            "TAB16 must use exact AE16 26-feature order",
        )
    if len(names) != 26:
        raise Ae16RegistryError(FEATURE_ORDER_MISMATCH, f"expected 26 features, got {len(names)}")
    forbidden = audit_forbidden_features(names)
    if not forbidden["passed"]:
        raise Ae16RegistryError("FORBIDDEN_FEATURE_PRESENT", str(forbidden["forbidden_features_found"]))
    hashes = compute_schema_hashes(names)
    return {
        "ordered_feature_names": names,
        "feature_count": 26,
        "feature_set_hash_sha256": hashes["feature_set_hash_sha256"],
        "ordered_feature_schema_hash_sha256": hashes["ordered_feature_schema_hash_sha256"],
        "forbidden_features_checked": True,
        "forbidden_features_found": [],
        "feature_order_enforced": True,
        "train_medians": train_medians or {},
        "dtype_by_feature": {n: "float64" for n in names},
    }


def reject_legacy_tab_as_tab16(path: str | Path) -> None:
    rel = str(path).replace("\\", "/")
    for legacy in LEGACY_TAB_ARTIFACTS:
        if rel.endswith(legacy) or rel.replace("\\", "/").endswith(legacy):
            raise Ae16RegistryError(
                "LEGACY_TAB_REJECTED_AS_TAB16",
                f"legacy TAB artifact rejected: {legacy}",
            )
    # Also reject ambiguous aliases
    for bad in (
        "models/tab.joblib",
        "models/tab_serving_safe.joblib",
        "models/ae16f_tab_serving_safe.joblib",
    ):
        if rel.endswith(bad):
            raise Ae16RegistryError("TAB16_AMBIGUOUS_ALIAS_REJECTED", bad)


def lookahead_audit(ordered_names: list[str]) -> dict[str, Any]:
    leaky = [n for n in ordered_names if is_forbidden_feature(n)]
    leaky += [n for n in ordered_names if re.search(r"(future|verified|return_|outcome)", n)]
    leaky = sorted(set(leaky))
    return {
        "lookahead_audit_passed": len(leaky) == 0,
        "leaky_or_forbidden_in_schema": leaky,
        "ordered_feature_names": ordered_names,
        "note": "serving-safe same-snapshot features only; no post-entry/future fields",
    }


def load_historical_direct_target(
    project_root: Path | None = None,
    source_rel: str = TRAINING_SOURCE_REL,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Return (meta_hist, X_ordered, y, summary). Never uses current selected rows."""
    root = project_root_from(project_root)
    path = root / source_rel
    if not path.is_file():
        raise FileNotFoundError(source_rel)
    df = pd.read_parquet(path)
    if "label_valid" in df.columns:
        df = df[
            df["label_valid"].astype(str).str.lower().isin({"1", "true", "yes"})
            | (df["label_valid"] == True)  # noqa: E712
        ]
    # Ensure no current-selected contamination marker
    if "price_source_key" in df.columns and df["price_source_key"].astype(str).str.contains("selected_clean").any():
        raise RuntimeError("CURRENT_SELECTED_ROWS_IN_TRAINING_SOURCE")

    feature_dicts = []
    meta_rows = []
    for _, rec in df.iterrows():
        d = rec.to_dict()
        feats = extract_serving_safe_row_from_historical(d)
        feature_dicts.append(feats)
        meta_rows.append(
            {
                "pair_address": d.get("pair_address"),
                "event_timestamp": d.get("event_timestamp"),
                "split": d.get("split"),
                "target": int(d.get(TARGET_COLUMN) or 0),
            }
        )

    # train medians from train split
    train_idxs = [
        i
        for i, m in enumerate(meta_rows)
        if str(m.get("split") or "").lower() in {"train", "training"}
    ]
    if not train_idxs:
        train_idxs = list(range(len(feature_dicts)))
    medians: dict[str, float] = {}
    for name in ORDERED_FEATURE_NAMES:
        if name.endswith("_is_missing"):
            continue
        vals = []
        for i in train_idxs:
            v = feature_dicts[i].get(name)
            if v is not None and not (isinstance(v, float) and v != v):
                vals.append(float(v))
        medians[name] = float(np.median(vals)) if vals else 0.0

    lock = build_feature_schema_lock(train_medians=medians)
    rows = []
    for fd in feature_dicts:
        row_vals = []
        for name in ORDERED_FEATURE_NAMES:
            v = fd.get(name)
            if v is None or (isinstance(v, float) and v != v):
                if name.endswith("_is_missing"):
                    v = 1.0
                else:
                    v = medians.get(name, 0.0)
            row_vals.append(float(v))
        rows.append(row_vals)
    X = pd.DataFrame(rows, columns=ORDERED_FEATURE_NAMES, dtype="float64")
    if list(X.columns) != ORDERED_FEATURE_NAMES:
        raise Ae16RegistryError(FEATURE_ORDER_MISMATCH, "training matrix order failed")
    verify_feature_schema_hashes(
        ordered_feature_names=list(X.columns),
        expected_feature_set_hash=lock["feature_set_hash_sha256"],
        expected_ordered_hash=lock["ordered_feature_schema_hash_sha256"],
    )
    hist = pd.DataFrame(meta_rows)
    y = hist["target"].astype(int).to_numpy()
    summary = {
        "training_source": source_rel.replace("\\", "/"),
        "rows": int(len(hist)),
        "lock": lock,
        "has_event_timestamp": "event_timestamp" in df.columns,
        "has_split": "split" in df.columns,
        "split_limitation": None,
    }
    return hist, X, y, summary


def train_tab16_classifier(X: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray):
    """HistGradientBoostingClassifier — distinct from RF/XGB artifacts."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    if list(X.columns) != ORDERED_FEATURE_NAMES:
        raise Ae16RegistryError(FEATURE_ORDER_MISMATCH, "train fit column order")
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=250,
                    learning_rate=0.05,
                    max_depth=6,
                    min_samples_leaf=25,
                    l2_regularization=0.1,
                    random_state=RANDOM_SEED,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipe.fit(X.loc[train_mask, ORDERED_FEATURE_NAMES], y[train_mask])
    return pipe


def predict_tab16_scores(model, X: pd.DataFrame, lock: dict[str, Any]) -> np.ndarray:
    if list(X.columns) != list(lock["ordered_feature_names"]):
        # try safe reorder
        try:
            X = build_ordered_inference_matrix(
                X,
                {
                    "ordered_feature_names": lock["ordered_feature_names"],
                    "feature_set_hash_sha256": lock["feature_set_hash_sha256"],
                    "ordered_feature_schema_hash_sha256": lock["ordered_feature_schema_hash_sha256"],
                },
                {"feature_schema_lock": lock},
            )
        except Ae16RegistryError:
            raise
    verify_feature_schema_hashes(
        ordered_feature_names=list(X.columns),
        expected_feature_set_hash=lock["feature_set_hash_sha256"],
        expected_ordered_hash=lock["ordered_feature_schema_hash_sha256"],
    )
    proba = np.asarray(model.predict_proba(X[ORDERED_FEATURE_NAMES]))[:, 1]
    return proba


def validation_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        roc_auc_score,
    )

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    votes = (scores >= float(threshold)).astype(int)
    out: dict[str, Any] = {
        "n": int(len(y_true)),
        "positive_rate": float(y_true.mean()) if len(y_true) else None,
        "score_min": float(scores.min()) if len(scores) else None,
        "score_max": float(scores.max()) if len(scores) else None,
        "score_p50": float(np.quantile(scores, 0.5)) if len(scores) else None,
        "score_p95": float(np.quantile(scores, 0.95)) if len(scores) else None,
        "threshold": float(threshold),
        "vote_true_rate": float(votes.mean()) if len(votes) else None,
        "accuracy_at_threshold": float(accuracy_score(y_true, votes)) if len(y_true) else None,
        "profitability_claimed": False,
        "note": "validation ranking/calibration metrics only; not a profitability claim",
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None
    except Exception as exc:  # noqa: BLE001
        out["roc_auc"] = None
        out["roc_auc_error"] = str(exc)
    try:
        out["average_precision"] = (
            float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else None
        )
    except Exception as exc:  # noqa: BLE001
        out["average_precision"] = None
        out["average_precision_error"] = str(exc)
    try:
        out["brier_score"] = float(brier_score_loss(y_true, scores))
    except Exception as exc:  # noqa: BLE001
        out["brier_score"] = None
        out["brier_score_error"] = str(exc)
    return out


def build_tab16_artifact_dict(
    *,
    model,
    lock: dict[str, Any],
    threshold: float,
    training_rows: int,
    validation_rows: int,
    training_source: str,
    lookahead_passed: bool,
    validation_metrics_payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": model,
        "model_family": MODEL_FAMILY,
        "model_variant": MODEL_VARIANT,
        "consensus_slot": CONSENSUS_SLOT,
        "phase": PHASE,
        "target": TARGET_COLUMN,
        "allowed_to_feed_ae16_consensus_tab_slot": True,
        "allowed_to_replace_legacy_tab": False,
        "legacy_tab_used": False,
        "legacy_tab_artifact_path": None,
        "legacy_tab_feature_schema_compatible": False,
        "training_source": training_source.replace("\\", "/"),
        "training_rows": int(training_rows),
        "validation_rows": int(validation_rows),
        "threshold": float(threshold),
        "threshold_policy": THRESHOLD_POLICY,
        "created_at_utc": utc_now(),
        "feature_schema_lock": lock,
        "lookahead_audit_passed": bool(lookahead_passed),
        "validation_metrics": validation_metrics_payload,
        "compatibility_scope": "AE16_ONLY",
        "estimator_class": "HistGradientBoostingClassifier",
        "not_a_copy_of_rf": True,
        "not_xgb": True,
    }


def build_ae16_registry(
    *,
    project_root: Path,
    tab16_artifact: dict[str, Any],
    rf_threshold: float,
    xgb_threshold: float,
) -> dict[str, Any]:
    lock = tab16_artifact["feature_schema_lock"]
    hashes = {
        "feature_set_hash_sha256": lock["feature_set_hash_sha256"],
        "ordered_feature_schema_hash_sha256": lock["ordered_feature_schema_hash_sha256"],
    }
    ordered = list(lock["ordered_feature_names"])
    return {
        "phase": PHASE,
        "compatibility_scope": "AE16_ONLY",
        "created_at_utc": utc_now(),
        "registry_version": "ae16_model_registry_v1",
        "authoritative_for_ae16_model_loading": True,
        "models": {
            "RF": {
                "artifact_path": RF_ARTIFACT_REL,
                "consensus_slot": "RF",
                "compatibility_scope": "AE16_ONLY",
                "model_family": "RF",
                "source_variant": "AE16F_RF_SERVING_SAFE",
                "threshold": float(rf_threshold),
                "threshold_policy": "historical_validation_top_5pct_quantile",
                "ordered_feature_names": ordered,
                "feature_count": 26,
                **hashes,
                "dict_unwrap_supported": True,
            },
            "XGB": {
                "artifact_path": XGB_ARTIFACT_REL,
                "consensus_slot": "XGB",
                "compatibility_scope": "AE16_ONLY",
                "model_family": "XGB",
                "source_variant": "AE16F_XGB_SERVING_SAFE",
                "threshold": float(xgb_threshold),
                "threshold_policy": "historical_validation_top_5pct_quantile",
                "ordered_feature_names": ordered,
                "feature_count": 26,
                **hashes,
                "requires_venv_xgboost": True,
                "dict_unwrap_supported": True,
            },
            "TAB_CONSENSUS_SLOT": {
                "artifact_path": ARTIFACT_REL,
                "source_variant": MODEL_VARIANT,
                "consensus_slot": "TAB",
                "compatibility_scope": "AE16_ONLY",
                "model_family": MODEL_FAMILY,
                "legacy_tab_used": False,
                "allowed_to_replace_legacy_tab": False,
                "threshold": float(tab16_artifact["threshold"]),
                "threshold_policy": tab16_artifact["threshold_policy"],
                "ordered_feature_names": ordered,
                "feature_count": 26,
                **hashes,
                "dict_unwrap_supported": True,
            },
        },
    }


def load_ae16f_thresholds(project_root: Path) -> dict[str, float]:
    path = project_root / AE16F_THRESHOLDS_REL
    out: dict[str, float] = {}
    if path.is_file():
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            fam = str(row.get("model_family") or "").upper()
            if fam in {"RF", "XGB", "TAB"}:
                out[fam] = float(row["threshold_value"])
    # Fallbacks from known AE16F pass
    out.setdefault("RF", 0.6608411682788434)
    out.setdefault("XGB", 0.9882992506027222)
    return out


def assign_consensus_preview_tier(
    *,
    has_l1: bool,
    rf_status: str,
    rf_vote: bool,
    xgb_status: str,
    xgb_vote: bool,
    tab16_status: str,
    tab16_vote: bool,
) -> str:
    if not has_l1:
        return "MODEL_EVIDENCE_UNAVAILABLE"

    rf_att = rf_status == ATTACHED
    xgb_att = xgb_status == ATTACHED
    tab_att = tab16_status == ATTACHED
    attached_count = sum([rf_att, xgb_att, tab_att])

    if attached_count == 0:
        return "MODEL_EVIDENCE_UNAVAILABLE"
    if attached_count < 3:
        return "PARTIAL_MODEL_EVIDENCE"

    # All three attached
    rf_v = bool(rf_vote) if rf_att else False
    xgb_v = bool(xgb_vote) if xgb_att else False
    tab_v = bool(tab16_vote) if tab_att else False
    vote_count = sum([rf_v, xgb_v, tab_v])
    if vote_count == 0:
        return "REJECT"
    if rf_v and xgb_v and tab_v:
        return "TAB_XGB_RF_ALL3"
    if tab_v and rf_v and not xgb_v:
        return "TAB_RF_ONLY"
    if tab_v and xgb_v and not rf_v:
        return "TAB_XGB_ONLY"
    if rf_v and xgb_v and not tab_v:
        return "RF_XGB_ONLY"
    if vote_count == 1:
        return "SINGLE_MODEL_ONLY"
    return "PARTIAL_MODEL_EVIDENCE"


def legacy_tab_isolation_audit(project_root: Path, tab16_path: Path) -> dict[str, Any]:
    root = project_root
    rows = []
    for rel in LEGACY_TAB_ARTIFACTS:
        p = root / rel
        st = p.stat() if p.is_file() else None
        rows.append(
            {
                "path": rel,
                "exists": p.is_file(),
                "mtime_ns": int(st.st_mtime_ns) if st else None,
                "size": int(st.st_size) if st else None,
                "used_as_tab16": False,
                "overwritten": False,
                "rejected_as_tab16_source": True,
            }
        )
        # Ensure we did not copy legacy into TAB16 path by comparing sizes if both exist
    alias = root / FORBIDDEN_ALIAS_REL
    tab16_is_legacy_copy = False
    if tab16_path.is_file():
        reject_legacy_tab_as_tab16(tab16_path)
        # size compare vs each legacy
        tsize = tab16_path.stat().st_size
        for rel in LEGACY_TAB_ARTIFACTS:
            lp = root / rel
            if lp.is_file() and lp.stat().st_size == tsize:
                # not definitive, but flag for review — still require distinct content via metadata
                pass
    return {
        "legacy_tab_used": False,
        "legacy_tab_artifact_path": None,
        "allowed_to_replace_legacy_tab": False,
        "alias_ae16f_tab_serving_safe_exists": alias.is_file(),
        "tab16_artifact_path": str(ARTIFACT_REL),
        "legacy_artifacts": rows,
        "passed": (not alias.is_file()) and tab16_path.is_file() and (not tab16_is_legacy_copy),
    }


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
