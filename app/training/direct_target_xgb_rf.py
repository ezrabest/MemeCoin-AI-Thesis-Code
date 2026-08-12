"""Phase E4A — direct-target XGB/RF offline training and evaluation infrastructure."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from app.artifacts.hash_utils import sha256_hex
from app.artifacts.registry import detect_project_root
from app.training.direct_target_ids import (
    DEFAULT_EXIT_POLICIES,
    DEFAULT_FILTERS,
    DEFAULT_HORIZONS,
    output_dataset_basename,
)
from app.training.tabicl_v2_eval import precision_at_top_k_with_count, return_metrics_for_top_k

PHASE = "E4A"
BRANCH_NAME = "phase_e4_direct_target_xgb_rf"
CANONICAL_TARGET = "target_net_profitable"
ACCEPTABLE_TARGET_ALIASES = (
    "target_net_profitable_after_exit_policy",
    "target_net_profitable_after_exit",
)
VALIDITY_COLUMNS = (
    "valid_label",
    "label_valid",
    "target_valid",
    "direct_target_valid",
    "valid_sim",
)
EVAL_NET_RETURN_COLUMNS = ("sim_net_return", "net_return")
SPLIT_VALUES = frozenset({"train", "validation", "test"})
TOP_PCTS = (0.5, 1.0, 2.0, 5.0)
PAIR_CAPS: tuple[int | None, ...] = (1, 2, 3, 5, 10, 20, 50, None)
IDENTITY_COLUMNS = frozenset(
    {
        "candidate_id",
        "candidate_policy_id",
        "target_row_id",
        "source_row_id",
        "artifact_id",
        "label_source_artifact_id",
        "source_artifact_id",
        "event_timestamp",
        "timestamp",
        "pair_address",
        "symbol",
        "chain",
        "split",
        "filter",
        "horizon",
        "exit_policy_id",
        "target_name",
        "target_version",
    }
)
LEAKAGE_SUBSTRINGS = (
    "target",
    "label",
    "future",
    "return",
    "net_return",
    "sim_net_return",
    "realized",
    "exit_status",
    "exit_ratio",
    "minutes_to_exit",
    "max_future",
    "min_future",
    "max_ratio",
    "min_ratio",
    "tp_count",
    "sl_count",
    "time_count",
    "valid_label",
    "valid_sim",
    "policy_result",
    "selected",
    "rank",
    "outcome",
    "profit",
    "pnl",
    "drawdown",
)
PREDICTION_ID_COLUMNS = (
    "candidate_id",
    "candidate_policy_id",
    "target_row_id",
    "pair_address",
    "event_timestamp",
    "filter",
    "horizon",
    "exit_policy_id",
    "split",
)
SMOKE_DEFAULT_MAX_ROWS = 5000
DATASET_PATTERN = re.compile(
    r"^(?P<filter>.+)_(?P<horizon>30m|1h|4h|8h|24h)_(?P<exit_policy>TP20308_SL0\d+_FEE0308_TIME_BY_HORIZON)_DIRECT_TARGET_v(?P<version>\d+)\.(?:parquet|csv)$",
    re.I,
)


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _atomic_tmp_path(path: Path) -> Path:
    return path.with_name(f".tmp_{uuid.uuid4().hex}{path.suffix}")


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_tmp_path(path)
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_bytes(data: bytes, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_tmp_path(path)
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_json(payload: Any, path: Path) -> None:
    atomic_write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), path)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_tmp_path(path)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _atomic_tmp_path(path)
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def feature_columns_hash(columns: list[str]) -> str:
    payload = "|".join(columns)
    return sha256_hex(payload)


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


@dataclass
class DatasetDescriptor:
    dataset_name: str
    dataset_path: Path
    filter_name: str
    horizon: str
    exit_policy_id: str
    target_name: str
    target_version: str
    row_count: int | None = None


class IncrementalAuditLogger:
    """Append-only JSONL audit logger with flush after each event."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, **fields: Any) -> None:
        payload = {
            "created_at_utc": utc_now_iso(),
            "event_type": event_type,
            "phase": PHASE,
            "branch_name": BRANCH_NAME,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class TargetNormalizationAudit:
    """Separate compact target-normalization audit CSV."""

    FIELDNAMES = [
        "dataset_name",
        "dataset_path",
        "filter",
        "horizon",
        "exit_policy_id",
        "target_column_original",
        "target_column_canonical",
        "target_column_dtype_before",
        "target_column_dtype_after",
        "target_unique_values_before",
        "target_unique_values_after",
        "target_null_count_before",
        "target_null_count_after",
        "positive_count_after_normalization",
        "negative_count_after_normalization",
        "normalization_status",
        "normalization_error",
        "created_at_utc",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
                writer.writeheader()
                handle.flush()

    def append_row(self, row: dict[str, Any]) -> None:
        out = {key: row.get(key) for key in self.FIELDNAMES}
        out["created_at_utc"] = out.get("created_at_utc") or utc_now_iso()
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            writer.writerow(out)
            handle.flush()
            os.fsync(handle.fileno())


def _unique_values_json(series: pd.Series) -> str:
    values = [str(v) for v in series.dropna().unique().tolist()]
    return json.dumps(sorted(values))


def _coerce_binary_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("Int64").astype(float)
    as_str = series.astype(str).str.strip().str.lower()
    mapped = series.copy()
    bool_map = {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0}
    for raw, val in bool_map.items():
        mapped = mapped.where(as_str != raw, val)
    return pd.to_numeric(mapped, errors="coerce")


def load_e3_manifest(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.exists():
        return None
    with manifest_path.open(encoding="utf-8") as handle:
        return json.load(handle)


def parse_dataset_filename(filename: str) -> dict[str, str] | None:
    match = DATASET_PATTERN.match(filename)
    if not match:
        return None
    return {
        "filter": match.group("filter"),
        "horizon": match.group("horizon"),
        "exit_policy_id": match.group("exit_policy"),
        "target_version": f"v{match.group('version')}",
        "target_name": "net_profitable_after_exit_policy",
    }


def discover_direct_target_datasets(
    input_dir: Path,
    *,
    manifest: dict[str, Any] | None = None,
) -> list[DatasetDescriptor]:
    if not input_dir.is_dir():
        return []

    manifest_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    if manifest:
        for entry in manifest.get("dataset_hashes", []):
            key = (entry["filter"], entry["horizon"], entry["exit_policy_id"])
            manifest_index[key] = entry

    by_stem: dict[str, Path] = {}
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        meta = parse_dataset_filename(path.name)
        if meta is None:
            continue
        stem = path.stem
        existing = by_stem.get(stem)
        if existing is None:
            by_stem[stem] = path
            continue
        if existing.suffix.lower() == ".csv" and path.suffix.lower() == ".parquet":
            by_stem[stem] = path

    descriptors: list[DatasetDescriptor] = []
    for stem, path in sorted(by_stem.items()):
        meta = parse_dataset_filename(path.name)
        if meta is None:
            continue
        key = (meta["filter"], meta["horizon"], meta["exit_policy_id"])
        manifest_row = manifest_index.get(key, {})
        descriptors.append(
            DatasetDescriptor(
                dataset_name=stem,
                dataset_path=path,
                filter_name=meta["filter"],
                horizon=meta["horizon"],
                exit_policy_id=meta["exit_policy_id"],
                target_name=meta["target_name"],
                target_version=meta["target_version"],
                row_count=manifest_row.get("row_count"),
            )
        )
    return descriptors


def filter_descriptors(
    descriptors: list[DatasetDescriptor],
    *,
    filter_name: str | None,
    horizon: str | None,
    exit_policy: str | None,
    smoke: bool,
) -> list[DatasetDescriptor]:
    selected = descriptors
    if filter_name:
        selected = [d for d in selected if d.filter_name == filter_name]
    if horizon:
        selected = [d for d in selected if d.horizon == horizon]
    if exit_policy:
        selected = [d for d in selected if d.exit_policy_id == exit_policy]
    if smoke and not filter_name and not horizon and not exit_policy:
        if selected:
            selected = [selected[0]]
    return selected


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def normalize_target_column(
    df: pd.DataFrame,
    descriptor: DatasetDescriptor,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    present = [col for col in ACCEPTABLE_TARGET_ALIASES if col in df.columns]
    audit: dict[str, Any] = {
        "dataset_name": descriptor.dataset_name,
        "dataset_path": str(descriptor.dataset_path),
        "filter": descriptor.filter_name,
        "horizon": descriptor.horizon,
        "exit_policy_id": descriptor.exit_policy_id,
        "target_column_canonical": CANONICAL_TARGET,
    }
    if not present:
        audit.update(
            {
                "target_column_original": None,
                "normalization_status": "DIRECT_TARGET_COLUMN_MISSING",
                "normalization_error": "No acceptable direct target column found",
            }
        )
        return df, audit
    if len(present) > 1:
        audit.update(
            {
                "target_column_original": "|".join(present),
                "normalization_status": "AMBIGUOUS_TARGET_ALIAS",
                "normalization_error": "Multiple acceptable target aliases present",
            }
        )
        return df, audit

    source_col = present[0]
    series = df[source_col]
    audit["target_column_original"] = source_col
    audit["target_column_dtype_before"] = str(series.dtype)
    audit["target_unique_values_before"] = _unique_values_json(series)
    audit["target_null_count_before"] = int(series.isna().sum())

    out = df.copy()
    out[CANONICAL_TARGET] = _coerce_binary_target(series)
    normalized = out[CANONICAL_TARGET]
    audit["target_column_dtype_after"] = str(normalized.dtype)
    audit["target_unique_values_after"] = _unique_values_json(normalized)
    audit["target_null_count_after"] = int(normalized.isna().sum())
    positives = int((normalized == 1).sum())
    negatives = int((normalized == 0).sum())
    audit["positive_count_after_normalization"] = positives
    audit["negative_count_after_normalization"] = negatives
    audit["normalization_status"] = "ok"
    audit["normalization_error"] = None
    return out, audit


def _is_leakage_column(name: str) -> bool:
    lower = name.lower()
    if name in IDENTITY_COLUMNS:
        return True
    if name == CANONICAL_TARGET:
        return True
    return any(token in lower for token in LEAKAGE_SUBSTRINGS)


def derive_valid_label_mask(df: pd.DataFrame) -> pd.Series:
    for col in VALIDITY_COLUMNS:
        if col in df.columns:
            return df[col].fillna(False).astype(bool)
    target = df[CANONICAL_TARGET]
    return target.notna() & target.isin([0, 1])


def apply_deterministic_row_limit(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if len(df) <= max_rows:
        return df
    if "split" not in df.columns:
        step = max(1, len(df) // max_rows)
        return df.iloc[::step].head(max_rows).copy()
    parts: list[pd.DataFrame] = []
    total = len(df)
    remaining = max_rows
    for split_name in ("train", "validation", "test"):
        split_df = df[df["split"] == split_name]
        if split_df.empty:
            continue
        alloc = max(1, int(round(len(split_df) / total * max_rows)))
        alloc = min(alloc, len(split_df), remaining)
        if len(split_df) <= alloc:
            parts.append(split_df)
        else:
            step = max(1, len(split_df) // alloc)
            parts.append(split_df.iloc[::step].head(alloc))
        remaining -= len(parts[-1])
        if remaining <= 0:
            break
    if not parts:
        step = max(1, len(df) // max_rows)
        return df.iloc[::step].head(max_rows).copy()
    return pd.concat(parts, ignore_index=True)


def validate_split_column(df: pd.DataFrame) -> tuple[bool, str | None]:
    if "split" not in df.columns:
        return False, "SPLIT_COLUMN_MISSING"
    values = set(df["split"].dropna().astype(str).str.lower())
    if not values.issubset(SPLIT_VALUES):
        return False, "SPLIT_COLUMN_INVALID"
    if not values:
        return False, "SPLIT_COLUMN_MISSING"
    return True, None


def build_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    excluded_leakage: list[str] = []
    excluded_identity: list[str] = []
    dropped_all_null: list[str] = []
    numeric_features: list[str] = []
    boolean_features: list[str] = []

    train_df = df[df["split"] == "train"] if "split" in df.columns else df

    for col in df.columns:
        if col in IDENTITY_COLUMNS:
            excluded_identity.append(col)
            continue
        if _is_leakage_column(col):
            excluded_leakage.append(col)
            continue
        dtype = df[col].dtype
        if pd.api.types.is_bool_dtype(dtype):
            boolean_features.append(col)
            continue
        if pd.api.types.is_numeric_dtype(dtype):
            if train_df[col].notna().any():
                numeric_features.append(col)
            else:
                dropped_all_null.append(col)
            continue
        excluded_leakage.append(col)

    feature_columns = numeric_features + boolean_features
    return feature_columns, excluded_leakage, excluded_identity, dropped_all_null


def assert_row_count_invariants(counts: dict[str, int]) -> None:
    required_equal = [
        (
            counts["post_valid_filter_row_count"],
            counts["train_row_count"] + counts["validation_row_count"] + counts["test_row_count"],
            "post_valid_filter_row_count == train+validation+test",
        ),
        (counts["feature_matrix_train_row_count"], counts["train_row_count"], "feature_matrix_train == train"),
        (
            counts["feature_matrix_validation_row_count"],
            counts["validation_row_count"],
            "feature_matrix_validation == validation",
        ),
        (counts["feature_matrix_test_row_count"], counts["test_row_count"], "feature_matrix_test == test"),
        (
            counts["prediction_validation_row_count"],
            counts["validation_row_count"],
            "prediction_validation == validation",
        ),
        (counts["prediction_test_row_count"], counts["test_row_count"], "prediction_test == test"),
    ]
    for left, right, label in required_equal:
        if left != right:
            raise RuntimeError(
                f"ROW_COUNT_INVARIANT_FAILED: {label} ({left} != {right})"
            )


def build_feature_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    split_name: str,
) -> pd.DataFrame:
    split_df = df[df["split"] == split_name].copy()
    missing = [col for col in feature_columns if col not in split_df.columns]
    if missing:
        raise RuntimeError(f"MODEL_SCHEMA_MISMATCH: missing columns {missing}")
    matrix = split_df[feature_columns].copy()
    for col in feature_columns:
        if pd.api.types.is_bool_dtype(matrix[col].dtype):
            matrix[col] = matrix[col].astype(float)
    if len(matrix) != len(split_df):
        raise RuntimeError("ROW_COUNT_INVARIANT_FAILED: feature matrix row loss")
    return matrix


def build_xgb_classifier(
    *,
    random_state: int,
    scale_pos_weight: float,
    device: str,
) -> XGBClassifier:
    params: dict[str, Any] = {
        "n_estimators": 700,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "tree_method": "hist",
        "random_state": random_state,
        "scale_pos_weight": scale_pos_weight,
    }
    if device == "cuda":
        params["device"] = "cuda"
    return XGBClassifier(**params)


def build_rf_classifier(*, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=random_state,
    )


def build_training_pipeline(model_name: str, estimator: Any) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def resolve_xgb_device(
    *,
    requested: str,
    allow_cpu_fallback: bool,
) -> dict[str, Any]:
    import xgboost

    info: dict[str, Any] = {
        "xgb_device_requested": requested,
        "xgb_device_used": requested,
        "cuda_available_if_detected": None,
        "cpu_fallback_used": False,
        "cuda_error": None,
        "xgboost_version": xgboost.__version__,
    }
    build_info = getattr(xgboost, "build_info", lambda: {})()
    cuda_detected = bool(build_info.get("USE_CUDA", False)) if isinstance(build_info, dict) else False
    info["cuda_available_if_detected"] = cuda_detected
    if requested != "cuda":
        info["xgb_device_used"] = requested
        return info
    if not cuda_detected and not allow_cpu_fallback:
        info["cuda_error"] = "CUDA not available in XGBoost build"
        return info
    if not cuda_detected and allow_cpu_fallback:
        info["xgb_device_used"] = "cpu_fallback"
        info["cpu_fallback_used"] = True
        info["cuda_error"] = "CUDA not available; using CPU fallback"
        return info
    return info


def fit_xgb_pipeline(
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    *,
    device_info: dict[str, Any],
    allow_cpu_fallback: bool,
) -> dict[str, Any]:
    used = dict(device_info)
    model: XGBClassifier = pipeline.named_steps["model"]
    try:
        pipeline.fit(x_train, y_train)
        return used
    except Exception as exc:
        if device_info.get("xgb_device_requested") == "cuda" and not allow_cpu_fallback:
            used["cuda_error"] = str(exc)
            raise RuntimeError(f"CUDA_XGB_TRAINING_FAILED: {exc}") from exc
        if allow_cpu_fallback:
            model.set_params(device="cpu")
            used["xgb_device_used"] = "cpu_fallback"
            used["cpu_fallback_used"] = True
            used["cuda_error"] = str(exc)
            pipeline.fit(x_train, y_train)
            return used
        raise


def extract_positive_probability(pipeline: Pipeline, x_matrix: pd.DataFrame) -> np.ndarray:
    estimator = pipeline.named_steps["model"]
    proba = pipeline.predict_proba(x_matrix)
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise RuntimeError("POSITIVE_CLASS_MISSING")
    pos_idx = classes.index(1)
    return proba[:, pos_idx]


def resolve_eval_return_column(df: pd.DataFrame) -> str | None:
    for col in EVAL_NET_RETURN_COLUMNS:
        if col in df.columns:
            return col
    return None


def compute_split_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    frame: pd.DataFrame,
    *,
    split_name: str,
    return_col: str | None,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "split": split_name,
        "row_count": int(len(y_true)),
        "positive_count": int((y_true == 1).sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else None,
        "pr_auc": None,
        "roc_auc": None,
        "evaluation_net_return_status": "ok" if return_col else "EVALUATION_NET_RETURN_UNAVAILABLE",
    }
    if len(np.unique(y_true)) > 1:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))

    eval_frame = frame.copy()
    eval_frame["y_true"] = y_true
    for pct in TOP_PCTS:
        key = str(pct).replace(".", "_")
        prec = precision_at_top_k_with_count(y_true, y_score, pct)
        metrics[f"precision_at_top_{key}_percent"] = prec["precision"]
        metrics[f"selected_trade_count_top_{key}_percent"] = prec["trade_count"]
        ret = return_metrics_for_top_k(eval_frame, y_score, pct, return_col)
        metrics[f"mean_sim_net_return_top_{key}_percent"] = ret["mean_return"]
        metrics[f"total_sim_net_return_top_{key}_percent"] = ret["total_return"]
    return metrics


def select_top_with_pair_cap(
    df: pd.DataFrame,
    *,
    score_col: str,
    k: int,
    pair_cap: int | None,
) -> pd.DataFrame:
    ranked = df.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)
    if pair_cap is None:
        return ranked.head(k).copy()
    selected_idx: list[int] = []
    counts: dict[str, int] = {}
    for idx, row in ranked.iterrows():
        pair = str(row.get("pair_address", ""))
        n = counts.get(pair, 0)
        if n >= pair_cap:
            continue
        selected_idx.append(idx)
        counts[pair] = n + 1
        if len(selected_idx) >= k:
            break
    return ranked.loc[selected_idx].copy()


def summarize_policy_selection(
    selected: pd.DataFrame,
    *,
    target_col: str,
    return_col: str | None,
) -> dict[str, Any]:
    y_true = selected[target_col].fillna(0).astype(int)
    valid_count = int(len(selected))
    positive_count = int((y_true == 1).sum())
    result: dict[str, Any] = {
        "selected_count": valid_count,
        "valid_count": valid_count,
        "positive_count": positive_count,
        "target_precision": float(y_true.mean()) if valid_count else None,
        "net_win_rate": float(y_true.mean()) if valid_count else None,
        "avg_net_return": None,
        "median_net_return": None,
        "total_net_return": None,
        "unique_pairs": int(selected["pair_address"].nunique()) if "pair_address" in selected.columns else None,
        "top_pair_share": None,
    }
    if return_col and return_col in selected.columns:
        returns = pd.to_numeric(selected[return_col], errors="coerce").dropna()
        if not returns.empty:
            result["avg_net_return"] = float(returns.mean())
            result["median_net_return"] = float(returns.median())
            result["total_net_return"] = float(returns.sum())
            result["net_win_rate"] = float((returns > 0).mean())
    if "pair_address" in selected.columns and valid_count:
        pair_counts = selected["pair_address"].value_counts()
        result["top_pair_share"] = float(pair_counts.iloc[0] / valid_count)
    return result


def evaluate_policy_grid(
    pred_df: pd.DataFrame,
    *,
    model_name: str,
    filter_name: str,
    horizon: str,
    exit_policy_id: str,
    split_name: str,
    target_col: str = CANONICAL_TARGET,
    return_col: str | None,
    score_col: str = "predicted_probability",
) -> list[dict[str, Any]]:
    if "pair_address" not in pred_df.columns:
        return []
    rows: list[dict[str, Any]] = []
    n = len(pred_df)
    for top_pct in TOP_PCTS:
        k = max(1, int(n * top_pct / 100.0))
        for pair_cap in PAIR_CAPS:
            selected = select_top_with_pair_cap(
                pred_df,
                score_col=score_col,
                k=k,
                pair_cap=pair_cap,
            )
            summary = summarize_policy_selection(
                selected,
                target_col=target_col,
                return_col=return_col,
            )
            rows.append(
                {
                    "model": model_name,
                    "filter": filter_name,
                    "horizon": horizon,
                    "exit_policy_id": exit_policy_id,
                    "split": split_name,
                    "top_pct": top_pct,
                    "pair_cap": "none" if pair_cap is None else pair_cap,
                    **summary,
                }
            )
    return rows


def rank_validation_policies(grid_df: pd.DataFrame) -> pd.DataFrame:
    if grid_df.empty:
        return pd.DataFrame()
    val = grid_df[grid_df["split"] == "validation"].copy()
    if val.empty:
        return pd.DataFrame()
    val["pair_cap_str"] = val["pair_cap"].astype(str)
    eligible = val.copy()
    if "valid_sim_rate" in eligible.columns:
        eligible = eligible[eligible["valid_sim_rate"].fillna(1.0) >= 0.95]
    eligible = eligible[
        (eligible["selected_count"].fillna(0) >= 50)
        & (eligible["total_net_return"].fillna(-1) > 0)
        & (eligible["avg_net_return"].fillna(-1) > 0)
        & (eligible["pair_cap_str"] != "none")
        & (eligible["unique_pairs"].fillna(0) >= 7)
        & (eligible["top_pair_share"].fillna(1.0) <= 0.25)
    ]
    if eligible.empty:
        return pd.DataFrame()
    eligible = eligible.sort_values(
        ["total_net_return", "avg_net_return", "target_precision"],
        ascending=[False, False, False],
        kind="mergesort",
    )
    keys = ["model", "filter", "horizon", "exit_policy_id"]
    return eligible.drop_duplicates(keys, keep="first")


def apply_validation_policies_to_test(
    grid_df: pd.DataFrame,
    selected_val: pd.DataFrame,
) -> pd.DataFrame:
    if selected_val.empty or grid_df.empty:
        return pd.DataFrame()
    test = grid_df[grid_df["split"] == "test"].copy()
    test["pair_cap_str"] = test["pair_cap"].astype(str)
    rows: list[dict[str, Any]] = []
    for _, policy in selected_val.iterrows():
        mask = (
            (test["model"] == policy["model"])
            & (test["filter"] == policy["filter"])
            & (test["horizon"] == policy["horizon"])
            & (test["exit_policy_id"] == policy["exit_policy_id"])
            & (test["top_pct"] == policy["top_pct"])
            & (test["pair_cap_str"] == policy["pair_cap_str"])
        )
        matched = test.loc[mask]
        if matched.empty:
            continue
        row = matched.iloc[0].to_dict()
        row["validation_total_net_return"] = policy.get("total_net_return")
        row["validation_avg_net_return"] = policy.get("avg_net_return")
        row["validation_target_precision"] = policy.get("target_precision")
        rows.append(row)
    return pd.DataFrame(rows)


def build_xgb_rf_agreement_diagnostic(
    xgb_preds: pd.DataFrame,
    rf_preds: pd.DataFrame,
    *,
    filter_name: str,
    horizon: str,
    exit_policy_id: str,
    split_name: str,
    target_col: str = CANONICAL_TARGET,
    return_col: str | None,
    pair_cap: int = 50,
) -> list[dict[str, Any]]:
    join_cols = ["candidate_policy_id", "target_row_id", "split"]
    merged = xgb_preds.merge(
        rf_preds,
        on=join_cols,
        suffixes=("_xgb", "_rf"),
        how="inner",
    )
    if merged.empty:
        return []
    score_xgb = merged["predicted_probability_xgb"].to_numpy()
    score_rf = merged["predicted_probability_rf"].to_numpy()
    rows: list[dict[str, Any]] = []
    for top_pct in TOP_PCTS:
        k = max(1, int(len(merged) * top_pct / 100.0))
        xgb_top_idx = set(np.argsort(-score_xgb)[:k])
        rf_top_idx = set(np.argsort(-score_rf)[:k])
        both_idx = xgb_top_idx & rf_top_idx
        xgb_only_idx = xgb_top_idx - rf_top_idx
        rf_only_idx = rf_top_idx - xgb_top_idx
        for label, idx_set in (
            ("XGB_top_k", xgb_top_idx),
            ("RF_top_k", rf_top_idx),
            ("XGB_AND_RF", both_idx),
            ("XGB_ONLY", xgb_only_idx),
            ("RF_ONLY", rf_only_idx),
        ):
            if not idx_set:
                continue
            selected = merged.iloc[sorted(idx_set)].copy()
            if pair_cap is not None and "pair_address_xgb" in selected.columns:
                selected = select_top_with_pair_cap(
                    selected.rename(columns={"pair_address_xgb": "pair_address"}),
                    score_col="predicted_probability_xgb",
                    k=len(selected),
                    pair_cap=pair_cap,
                )
            eval_df = selected.copy()
            if f"{target_col}_xgb" in eval_df.columns:
                eval_df[target_col] = eval_df[f"{target_col}_xgb"]
            if return_col and f"{return_col}_xgb" in eval_df.columns:
                eval_df[return_col] = eval_df[f"{return_col}_xgb"]
            if "pair_address_xgb" in eval_df.columns:
                eval_df["pair_address"] = eval_df["pair_address_xgb"]
            summary = summarize_policy_selection(
                eval_df,
                target_col=target_col,
                return_col=return_col,
            )
            rows.append(
                {
                    "model": "XGB_RF_DIAGNOSTIC",
                    "filter": filter_name,
                    "horizon": horizon,
                    "exit_policy_id": exit_policy_id,
                    "split": split_name,
                    "top_pct": top_pct,
                    "pair_cap": pair_cap,
                    "agreement_slice": label,
                    **summary,
                }
            )
    return rows


def extract_imputer_metadata(
    pipeline: Pipeline,
    feature_columns: list[str],
    train_matrix: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, int], dict[str, float]]:
    imputer: SimpleImputer = pipeline.named_steps["imputer"]
    stats = dict(zip(feature_columns, imputer.statistics_.tolist()))
    for name, value in stats.items():
        if value is None or (isinstance(value, float) and not np.isfinite(value)):
            raise RuntimeError(f"INVALID_IMPUTER_STATISTIC: {name}={value}")
    missing_counts = {col: int(train_matrix[col].isna().sum()) for col in feature_columns}
    medians = {col: float(stats[col]) for col in feature_columns}
    return stats, missing_counts, medians


def build_preprocessing_sidecar(
    *,
    model_name: str,
    descriptor: DatasetDescriptor,
    target_column_source: str,
    feature_columns: list[str],
    dropped_all_null_columns: list[str],
    excluded_leakage_columns: list[str],
    excluded_identity_columns: list[str],
    imputer_statistics_by_feature: dict[str, float],
    train_missing_count_by_feature: dict[str, int],
    train_median_by_feature: dict[str, float],
    pipeline: Pipeline,
    random_state: int,
) -> dict[str, Any]:
    boolean_count = sum(1 for c in feature_columns if c.startswith("is_") or c.endswith("_flag"))
    try:
        joblib_version = joblib.__version__
    except Exception:
        joblib_version = None
    return {
        "model": model_name,
        "filter": descriptor.filter_name,
        "horizon": descriptor.horizon,
        "exit_policy_id": descriptor.exit_policy_id,
        "target_column_source": target_column_source,
        "target_column_canonical": CANONICAL_TARGET,
        "feature_columns_in_order": feature_columns,
        "dropped_all_null_columns": dropped_all_null_columns,
        "excluded_leakage_columns": excluded_leakage_columns,
        "excluded_identity_columns": excluded_identity_columns,
        "numeric_feature_count": len(feature_columns) - boolean_count,
        "boolean_feature_count": boolean_count,
        "imputer_strategy": "median",
        "imputer_statistics_by_feature": imputer_statistics_by_feature,
        "train_missing_count_by_feature": train_missing_count_by_feature,
        "train_median_by_feature": train_median_by_feature,
        "pipeline_class": type(pipeline).__name__,
        "pipeline_steps": [name for name, _ in pipeline.steps],
        "sklearn_version": sklearn.__version__,
        "joblib_version_if_available": joblib_version,
        "python_version": sys.version,
        "created_at_utc": utc_now_iso(),
        "random_state": random_state,
    }


def artifact_combo_key(model_name: str, descriptor: DatasetDescriptor) -> str:
    return f"{model_name}_{descriptor.filter_name}_{descriptor.horizon}_{descriptor.exit_policy_id}"


def model_artifact_stem(model_name: str, descriptor: DatasetDescriptor) -> str:
    return f"direct_target_{artifact_combo_key(model_name, descriptor)}"


def register_e4_artifacts(project_root: Path, output_dir: Path) -> dict[str, Any]:
    status: dict[str, Any] = {
        "attempted": True,
        "success": False,
        "error": None,
        "artifacts_registered": 0,
        "repair_command": (
            "python scripts/register_existing_artifacts.py "
            f"--include-root {output_dir.relative_to(project_root).as_posix()}"
        ),
    }
    try:
        from app.artifacts.registry import get_git_commit_hash, load_registry, scan_artifacts, write_registry_jsonl

        rel_output = output_dir.relative_to(project_root).as_posix()
        scan_roots = [rel_output]
        registry_path = project_root / "data/training/artifact_registry/artifact_registry.jsonl"
        git_commit_hash, git_warnings = get_git_commit_hash(project_root)
        previous = load_registry(registry_path)
        records, scan_warnings = scan_artifacts(
            project_root=project_root,
            scan_roots=scan_roots,
            branch_name=BRANCH_NAME,
            generated_by_script="scripts/train_direct_target_xgb_rf.py",
            previous_registry=previous,
            git_commit_hash=git_commit_hash,
            git_warnings=git_warnings,
        )
        merged = {r.project_relative_path: r for r in previous.values()}
        for record in records:
            merged[record.project_relative_path] = record
        write_registry_jsonl(list(merged.values()), registry_path)
        status["success"] = True
        status["artifacts_registered"] = len(records)
        status["scan_warnings"] = scan_warnings
    except Exception as exc:
        status["error"] = str(exc)
    return status


@dataclass
class TrainConfig:
    input_dir: Path
    output_dir: Path
    models: tuple[str, ...]
    smoke: bool = False
    overwrite: bool = False
    register_artifacts: bool = True
    xgb_device: str = "cuda"
    allow_cpu_fallback: bool = False
    min_train_positives: int = 10
    min_validation_positives: int = 3
    min_test_positives: int = 3
    max_rows: int | None = None
    random_state: int = 42
    selected_descriptors: list[DatasetDescriptor] | None = None


@dataclass
class RunState:
    policy_grid_rows: list[dict[str, Any]] = field(default_factory=list)
    agreement_rows: list[dict[str, Any]] = field(default_factory=list)
    dataset_summaries: list[dict[str, Any]] = field(default_factory=list)
    skipped_datasets: list[dict[str, Any]] = field(default_factory=list)
    feature_columns_rows: list[dict[str, Any]] = field(default_factory=list)
    xgb_preds_by_key: dict[tuple[str, str, str, str], pd.DataFrame] = field(default_factory=dict)
    rf_preds_by_key: dict[tuple[str, str, str, str], pd.DataFrame] = field(default_factory=dict)


def prepare_output_dirs(output_dir: Path, overwrite: bool) -> None:
    subdirs = ("models", "predictions", "metrics", "policy_evaluation", "reports", "audit")
    if overwrite and output_dir.exists():
        for sub in subdirs:
            sub_path = output_dir / sub
            if sub_path.exists():
                for child in sub_path.iterdir():
                    if child.is_file():
                        child.unlink()
    for sub in subdirs:
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def train_single_model(
    *,
    model_name: str,
    descriptor: DatasetDescriptor,
    df: pd.DataFrame,
    feature_columns: list[str],
    excluded_leakage: list[str],
    excluded_identity: list[str],
    dropped_all_null: list[str],
    target_column_source: str,
    config: TrainConfig,
    output_dir: Path,
    audit: IncrementalAuditLogger,
    return_col: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "model": model_name,
        "status": "started",
        "error_code": None,
        "error_message": None,
    }
    audit.log(
        "model_started",
        dataset_path=str(descriptor.dataset_path),
        model=model_name,
        filter=descriptor.filter_name,
        horizon=descriptor.horizon,
        exit_policy_id=descriptor.exit_policy_id,
        status="started",
        random_state=config.random_state,
    )
    try:
        train_df = df[df["split"] == "train"]
        val_df = df[df["split"] == "validation"]
        test_df = df[df["split"] == "test"]
        y_train = train_df[CANONICAL_TARGET].astype(int).to_numpy()
        y_val = val_df[CANONICAL_TARGET].astype(int).to_numpy()
        y_test = test_df[CANONICAL_TARGET].astype(int).to_numpy()

        pos_counts = {
            "train": int((y_train == 1).sum()),
            "validation": int((y_val == 1).sum()),
            "test": int((y_test == 1).sum()),
        }
        if pos_counts["train"] < config.min_train_positives:
            raise RuntimeError("MIN_TRAIN_POSITIVES_NOT_MET")
        if pos_counts["validation"] < config.min_validation_positives:
            raise RuntimeError("MIN_VALIDATION_POSITIVES_NOT_MET")
        if pos_counts["test"] < config.min_test_positives:
            raise RuntimeError("MIN_TEST_POSITIVES_NOT_MET")

        x_train = build_feature_matrix(df, feature_columns, split_name="train")
        x_val = build_feature_matrix(df, feature_columns, split_name="validation")
        x_test = build_feature_matrix(df, feature_columns, split_name="test")

        counts = {
            "post_valid_filter_row_count": len(df),
            "train_row_count": len(train_df),
            "validation_row_count": len(val_df),
            "test_row_count": len(test_df),
            "feature_matrix_train_row_count": len(x_train),
            "feature_matrix_validation_row_count": len(x_val),
            "feature_matrix_test_row_count": len(x_test),
            "prediction_validation_row_count": len(val_df),
            "prediction_test_row_count": len(test_df),
        }
        assert_row_count_invariants(counts)
        audit.log("row_count_invariants_checked", model=model_name, status="ok", **counts)

        scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
        if model_name == "XGB":
            device_info = resolve_xgb_device(
                requested=config.xgb_device,
                allow_cpu_fallback=config.allow_cpu_fallback,
            )
            if config.xgb_device == "cuda" and device_info.get("cuda_error") and not config.allow_cpu_fallback:
                raise RuntimeError(device_info["cuda_error"])
            estimator = build_xgb_classifier(
                random_state=config.random_state,
                scale_pos_weight=scale_pos_weight,
                device="cpu" if device_info.get("cpu_fallback_used") else config.xgb_device,
            )
        else:
            device_info = {}
            estimator = build_rf_classifier(random_state=config.random_state)

        pipeline = build_training_pipeline(model_name, estimator)
        if model_name == "XGB":
            device_info = fit_xgb_pipeline(
                pipeline,
                x_train,
                y_train,
                device_info=device_info,
                allow_cpu_fallback=config.allow_cpu_fallback,
            )
        else:
            pipeline.fit(x_train, y_train)

        imputer_stats, missing_counts, medians = extract_imputer_metadata(pipeline, feature_columns, x_train)
        sidecar = build_preprocessing_sidecar(
            model_name=model_name,
            descriptor=descriptor,
            target_column_source=target_column_source,
            feature_columns=feature_columns,
            dropped_all_null_columns=dropped_all_null,
            excluded_leakage_columns=excluded_leakage,
            excluded_identity_columns=excluded_identity,
            imputer_statistics_by_feature=imputer_stats,
            train_missing_count_by_feature=missing_counts,
            train_median_by_feature=medians,
            pipeline=pipeline,
            random_state=config.random_state,
        )

        base = model_artifact_stem(model_name, descriptor)
        combo = artifact_combo_key(model_name, descriptor)
        model_path = output_dir / "models" / f"{base}.joblib"
        sidecar_path = output_dir / "models" / f"{base}_preprocessing.json"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, model_path)
        atomic_write_json(sidecar, sidecar_path)
        audit.log("model_trained", model=model_name, status="ok")
        audit.log(
            "pipeline_artifact_written",
            model=model_name,
            status="ok",
            output_paths={"model": str(model_path), "preprocessing": str(sidecar_path)},
        )

        val_proba = extract_positive_probability(pipeline, x_val)
        test_proba = extract_positive_probability(pipeline, x_test)

        pred_paths: dict[str, str] = {}
        split_frames: dict[str, pd.DataFrame] = {}
        for split_name, split_df, proba in (
            ("validation", val_df, val_proba),
            ("test", test_df, test_proba),
        ):
            pred = split_df[list(PREDICTION_ID_COLUMNS)].copy()
            pred[CANONICAL_TARGET] = split_df[CANONICAL_TARGET].astype(int).values
            pred["predicted_probability"] = proba
            pred["model"] = model_name
            if return_col and return_col in split_df.columns:
                pred["sim_net_return"] = split_df[return_col].values
            path = output_dir / "predictions" / f"direct_target_predictions_{split_name}_{combo}.parquet"
            atomic_write_parquet(pred, path)
            pred_paths[split_name] = str(path)
            split_frames[split_name] = pred
        audit.log("predictions_written", model=model_name, status="ok", output_paths=pred_paths)

        split_metrics = []
        for split_name, split_df, proba in (
            ("validation", val_df, val_proba),
            ("test", test_df, test_proba),
        ):
            y_true = split_df[CANONICAL_TARGET].astype(int).to_numpy()
            split_metrics.append(
                compute_split_metrics(
                    y_true,
                    proba,
                    split_df,
                    split_name=split_name,
                    return_col=return_col,
                )
            )

        metrics_payload = {
            "model": model_name,
            "filter": descriptor.filter_name,
            "horizon": descriptor.horizon,
            "exit_policy_id": descriptor.exit_policy_id,
            "model_artifact_path": str(model_path),
            "preprocessing_sidecar_path": str(sidecar_path),
            "sklearn_version": sklearn.__version__,
            "feature_columns_hash": feature_columns_hash(feature_columns),
            "training_row_count": len(train_df),
            "validation_row_count": len(val_df),
            "test_row_count": len(test_df),
            "random_state": config.random_state,
            "smoke_only": config.smoke,
            "split_metrics": split_metrics,
            **device_info,
        }
        metrics_path = output_dir / "metrics" / f"direct_target_metrics_{combo}.json"
        atomic_write_json(metrics_payload, metrics_path)
        audit.log("metrics_written", model=model_name, status="ok", output_paths={"metrics": str(metrics_path)})

        policy_rows: list[dict[str, Any]] = []
        for split_name, pred in split_frames.items():
            policy_rows.extend(
                evaluate_policy_grid(
                    pred,
                    model_name=model_name,
                    filter_name=descriptor.filter_name,
                    horizon=descriptor.horizon,
                    exit_policy_id=descriptor.exit_policy_id,
                    split_name=split_name,
                    return_col=return_col,
                )
            )
        result["policy_rows"] = policy_rows
        result["pred_frames"] = split_frames
        result["metrics_path"] = str(metrics_path)
        result["status"] = "completed"
        audit.log("model_completed", model=model_name, status="ok")
    except Exception as exc:
        result["status"] = "failed"
        result["error_code"] = getattr(exc, "args", [str(exc)])[0]
        result["error_message"] = str(exc)
        audit.log(
            "model_failed",
            model=model_name,
            status="failed",
            error_code=result["error_code"],
            error_message=result["error_message"],
        )
    return result


def process_dataset(
    descriptor: DatasetDescriptor,
    *,
    config: TrainConfig,
    output_dir: Path,
    audit: IncrementalAuditLogger,
    target_norm_audit: TargetNormalizationAudit,
    state: RunState,
) -> None:
    summary: dict[str, Any] = {
        "dataset_name": descriptor.dataset_name,
        "dataset_path": str(descriptor.dataset_path),
        "filter": descriptor.filter_name,
        "horizon": descriptor.horizon,
        "exit_policy_id": descriptor.exit_policy_id,
        "status": "started",
    }
    audit.log(
        "dataset_started",
        dataset_path=str(descriptor.dataset_path),
        filter=descriptor.filter_name,
        horizon=descriptor.horizon,
        exit_policy_id=descriptor.exit_policy_id,
        status="started",
        random_state=config.random_state,
    )
    try:
        df = load_dataset(descriptor.dataset_path)
        input_row_count = len(df)
        audit.log(
            "dataset_loaded",
            dataset_path=str(descriptor.dataset_path),
            input_row_count=input_row_count,
            status="ok",
        )

        df, norm_audit = normalize_target_column(df, descriptor)
        target_norm_audit.append_row(norm_audit)
        audit.log(
            "target_column_normalized",
            dataset_path=str(descriptor.dataset_path),
            status=norm_audit.get("normalization_status"),
            error_code=norm_audit.get("normalization_status") if norm_audit.get("normalization_status") != "ok" else None,
        )
        if norm_audit.get("normalization_status") != "ok":
            raise RuntimeError(norm_audit.get("normalization_status"))

        valid_mask = derive_valid_label_mask(df)
        valid_df = df.loc[valid_mask].copy()
        invalid_rows = int((~valid_mask).sum())
        audit.log(
            "valid_label_filter_applied",
            dataset_path=str(descriptor.dataset_path),
            valid_row_count=len(valid_df),
            invalid_row_count=invalid_rows,
            status="ok",
        )

        ok_split, split_error = validate_split_column(valid_df)
        if not ok_split:
            raise RuntimeError(split_error or "SPLIT_COLUMN_MISSING")
        audit.log("split_validated", dataset_path=str(descriptor.dataset_path), status="ok")

        max_rows = config.max_rows
        if config.smoke and max_rows is None:
            max_rows = SMOKE_DEFAULT_MAX_ROWS
        if max_rows is not None:
            valid_df = apply_deterministic_row_limit(valid_df, max_rows)
            audit.log(
                "smoke_row_limit_applied",
                dataset_path=str(descriptor.dataset_path),
                max_rows=max_rows,
                post_limit_row_count=len(valid_df),
                smoke_only=True,
                status="ok",
            )

        feature_columns, excluded_leakage, excluded_identity, dropped_all_null = build_feature_columns(valid_df)
        audit.log(
            "feature_matrix_built",
            dataset_path=str(descriptor.dataset_path),
            numeric_feature_count=len(feature_columns),
            status="ok",
        )
        state.feature_columns_rows.append(
            {
                "dataset_name": descriptor.dataset_name,
                "filter": descriptor.filter_name,
                "horizon": descriptor.horizon,
                "exit_policy_id": descriptor.exit_policy_id,
                "feature_columns": json.dumps(feature_columns),
                "feature_columns_hash": feature_columns_hash(feature_columns),
            }
        )

        return_col = resolve_eval_return_column(valid_df)
        target_column_source = norm_audit["target_column_original"]
        model_results: dict[str, Any] = {}
        for model_name in config.models:
            model_results[model_name] = train_single_model(
                model_name=model_name,
                descriptor=descriptor,
                df=valid_df,
                feature_columns=feature_columns,
                excluded_leakage=excluded_leakage,
                excluded_identity=excluded_identity,
                dropped_all_null=dropped_all_null,
                target_column_source=target_column_source,
                config=config,
                output_dir=output_dir,
                audit=audit,
                return_col=return_col,
            )
            if model_results[model_name].get("policy_rows"):
                state.policy_grid_rows.extend(model_results[model_name]["policy_rows"])
            key = (descriptor.filter_name, descriptor.horizon, descriptor.exit_policy_id, "validation")
            if model_name == "XGB" and "pred_frames" in model_results[model_name]:
                state.xgb_preds_by_key[(descriptor.filter_name, descriptor.horizon, descriptor.exit_policy_id)] = (
                    pd.concat(
                        [
                            model_results[model_name]["pred_frames"]["validation"],
                            model_results[model_name]["pred_frames"]["test"],
                        ],
                        ignore_index=True,
                    )
                )
            if model_name == "RF" and "pred_frames" in model_results[model_name]:
                state.rf_preds_by_key[(descriptor.filter_name, descriptor.horizon, descriptor.exit_policy_id)] = (
                    pd.concat(
                        [
                            model_results[model_name]["pred_frames"]["validation"],
                            model_results[model_name]["pred_frames"]["test"],
                        ],
                        ignore_index=True,
                    )
                )

        if (
            (descriptor.filter_name, descriptor.horizon, descriptor.exit_policy_id) in state.xgb_preds_by_key
            and (descriptor.filter_name, descriptor.horizon, descriptor.exit_policy_id) in state.rf_preds_by_key
        ):
            xgb_all = state.xgb_preds_by_key[(descriptor.filter_name, descriptor.horizon, descriptor.exit_policy_id)]
            rf_all = state.rf_preds_by_key[(descriptor.filter_name, descriptor.horizon, descriptor.exit_policy_id)]
            for split_name in ("validation", "test"):
                state.agreement_rows.extend(
                    build_xgb_rf_agreement_diagnostic(
                        xgb_all[xgb_all["split"] == split_name],
                        rf_all[rf_all["split"] == split_name],
                        filter_name=descriptor.filter_name,
                        horizon=descriptor.horizon,
                        exit_policy_id=descriptor.exit_policy_id,
                        split_name=split_name,
                        return_col=return_col,
                    )
                )

        summary.update(
            {
                "status": "completed"
                if any(v.get("status") == "completed" for v in model_results.values())
                else "failed",
                "input_row_count": input_row_count,
                "valid_rows": len(valid_df),
                "invalid_rows": invalid_rows,
                "models": {k: v.get("status") for k, v in model_results.items()},
            }
        )
        if summary["status"] == "completed":
            audit.log("dataset_completed", dataset_path=str(descriptor.dataset_path), status="completed")
        else:
            audit.log("dataset_failed", dataset_path=str(descriptor.dataset_path), status="failed")
    except Exception as exc:
        summary["status"] = "failed"
        summary["error_code"] = str(exc)
        if "ROW_COUNT_INVARIANT_FAILED" in str(exc):
            audit.log(
                "row_count_invariant_failed",
                dataset_path=str(descriptor.dataset_path),
                status="failed",
                error_code="ROW_COUNT_INVARIANT_FAILED",
                error_message=str(exc),
            )
        audit.log(
            "dataset_skipped" if "MIN_" in str(exc) or "MISSING" in str(exc) else "dataset_failed",
            dataset_path=str(descriptor.dataset_path),
            status="failed",
            error_code=str(exc),
            error_message=str(exc),
        )
        state.skipped_datasets.append(summary)
    state.dataset_summaries.append(summary)


def finalize_run_outputs(
    *,
    config: TrainConfig,
    output_dir: Path,
    state: RunState,
    audit: IncrementalAuditLogger,
    descriptors: list[DatasetDescriptor],
    registration_status: dict[str, Any] | None,
) -> None:
    policy_grid_path = output_dir / "policy_evaluation" / "direct_target_policy_grid_xgb_rf.csv"
    grid_df = pd.DataFrame(state.policy_grid_rows)
    atomic_write_csv(grid_df, policy_grid_path)
    audit.log("policy_evaluation_written", status="ok", output_paths={"policy_grid": str(policy_grid_path)})

    selected_val = rank_validation_policies(grid_df)
    applied_test = apply_validation_policies_to_test(grid_df, selected_val)
    applied_path = (
        output_dir
        / "policy_evaluation"
        / "validation_selected_policies_direct_target_xgb_rf_applied_to_test.csv"
    )
    atomic_write_csv(applied_test, applied_path)

    agreement_path = output_dir / "policy_evaluation" / "direct_target_xgb_rf_agreement_diagnostic.csv"
    atomic_write_csv(pd.DataFrame(state.agreement_rows), agreement_path)

    audit_summary_path = output_dir / "audit" / "phase_e4_dataset_audit_summary.csv"
    atomic_write_csv(pd.DataFrame(state.dataset_summaries), audit_summary_path)

    skipped_path = output_dir / "reports" / "phase_e4_skipped_datasets.csv"
    atomic_write_csv(pd.DataFrame(state.skipped_datasets), skipped_path)

    feature_cols_path = output_dir / "reports" / "phase_e4_feature_columns.csv"
    atomic_write_csv(pd.DataFrame(state.feature_columns_rows), feature_cols_path)

    manifest = {
        "phase": PHASE,
        "branch_name": BRANCH_NAME,
        "created_at_utc": utc_now_iso(),
        "input_dir": str(config.input_dir),
        "output_dir": str(output_dir),
        "datasets_requested": len(descriptors),
        "datasets_completed": sum(1 for s in state.dataset_summaries if s.get("status") == "completed"),
        "models": list(config.models),
        "random_state": config.random_state,
        "smoke": config.smoke,
        "sklearn_version": sklearn.__version__,
        "registration": registration_status,
    }
    manifest_path = output_dir / "reports" / "phase_e4_manifest.json"
    atomic_write_json(manifest, manifest_path)

    summary_text = (
        f"Phase E4A direct-target XGB/RF training\n"
        f"Datasets completed: {manifest['datasets_completed']}/{manifest['datasets_requested']}\n"
        f"Smoke mode: {config.smoke}\n"
        f"Output: {output_dir}\n"
    )
    if registration_status and not registration_status.get("success"):
        summary_text += (
            f"Registry registration failed: {registration_status.get('error')}\n"
            f"Repair: {registration_status.get('repair_command')}\n"
        )
    atomic_write_text(summary_text, output_dir / "reports" / "phase_e4_summary_for_upload.txt")
    audit.log("run_completed", status="ok")


def run_training(config: TrainConfig, *, e3_manifest_path: Path | None = None) -> dict[str, Any]:
    project_root = detect_project_root(config.input_dir)
    output_dir = config.output_dir
    prepare_output_dirs(output_dir, config.overwrite)

    audit_path = output_dir / "audit" / "phase_e4_run_audit.jsonl"
    audit = IncrementalAuditLogger(audit_path)
    target_norm_audit = TargetNormalizationAudit(output_dir / "audit" / "phase_e4_target_normalization_audit.csv")
    state = RunState()

    manifest = load_e3_manifest(e3_manifest_path) if e3_manifest_path else None
    descriptors = config.selected_descriptors
    if descriptors is None:
        descriptors = discover_direct_target_datasets(config.input_dir, manifest=manifest)
    audit.log("run_started", status="started", random_state=config.random_state, dataset_count=len(descriptors))

    for descriptor in descriptors:
        audit.log(
            "dataset_discovered",
            dataset_path=str(descriptor.dataset_path),
            filter=descriptor.filter_name,
            horizon=descriptor.horizon,
            exit_policy_id=descriptor.exit_policy_id,
            status="discovered",
        )

    for descriptor in descriptors:
        process_dataset(
            descriptor,
            config=config,
            output_dir=output_dir,
            audit=audit,
            target_norm_audit=target_norm_audit,
            state=state,
        )

    registration_status = None
    if config.register_artifacts:
        audit.log("artifact_registration_started", status="started")
        registration_status = register_e4_artifacts(project_root, output_dir)
        event = "artifact_registration_completed" if registration_status.get("success") else "artifact_registration_failed"
        audit.log(event, status=registration_status.get("success"), error_message=registration_status.get("error"))

    finalize_run_outputs(
        config=config,
        output_dir=output_dir,
        state=state,
        audit=audit,
        descriptors=descriptors,
        registration_status=registration_status,
    )
    return {
        "output_dir": str(output_dir),
        "datasets_completed": sum(1 for s in state.dataset_summaries if s.get("status") == "completed"),
        "registration": registration_status,
    }
