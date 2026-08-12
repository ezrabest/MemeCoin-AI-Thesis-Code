#!/usr/bin/env python
"""
Evaluate XGBoost on manual-verified CLEAN model inputs with safe artifact handling.

Purpose
-------
Run XGB on the same CLEAN inputs used for TAB/RF comparison, without creating stale
or duplicate artifacts.

Safety
------
- Does not touch TAB/RF outputs.
- Does not modify SQLite.
- Does not modify live/demo/paper trading.
- Does not retrain RF/TAB.
- Cleans only XGB artifacts for the selected run IDs.
- Writes all parquet/json outputs atomically using temporary files + os.replace().
- Removes partial temp files from interrupted prior runs.

Expected output directory
-------------------------
data/training/manual_verified_results/xgb_clean/

Typical outputs
---------------
xgb_predictions_validation_<RUN_ID>_XGB.parquet
xgb_predictions_test_<RUN_ID>_XGB.parquet
xgb_metrics_<RUN_ID>_XGB.json
xgb_clean_summary.json

Example
-------
python scripts/evaluate_xgb_manual_verified_clean.py ^
  --filter LIQ_5K_HIGH_ACTIVITY ^
  --horizons 30m 1h 4h 8h 24h ^
  --clean-existing

For a complete refresh of all discovered CLEAN inputs:

python scripts/evaluate_xgb_manual_verified_clean.py --all --clean-existing
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


try:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (
        average_precision_score,
        precision_score,
        roc_auc_score,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing sklearn dependency. Activate the project venv and install scikit-learn."
    ) from exc


try:
    from xgboost import XGBClassifier
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "Missing xgboost dependency. Install it in the active venv before running this script."
    ) from exc


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOTS = [
    ROOT / "data" / "training" / "manual_verified_results" / "clean_model_inputs",
    ROOT / "data" / "training" / "manual_verified_results",
]
DEFAULT_OUTPUT_DIR = ROOT / "data" / "training" / "manual_verified_results" / "xgb_clean_full"

HORIZONS = ("30m", "1h", "4h", "8h", "24h")

LEAKAGE_SUBSTRINGS = (
    "target",
    "future",
    "label",
    "outcome",
    "minutes_to",
    "max_future",
    "min_future",
    "return_after",
    "realized",
    "pnl",
    "profit",
    "profitable",
)

NON_FEATURE_EXACT = {
    "id",
    "row_id",
    "index",
    "__index_level_0__",
    "split",
    "dataset_split",
    "fold",
    "filter_status",
    "drop_reason",
    "legacy",
    "source_file",
}

METADATA_COLUMNS = (
    "pair_address",
    "pool_address",
    "coin_id",
    "symbol",
    "base_symbol",
    "quote_symbol",
    "event_timestamp",
    "timestamp",
    "provider",
    "chain",
    "price",
    "liquidity",
    "volume_24h",
    "fdv",
)

TARGET_CANDIDATE_TEMPLATES = (
    "target_x2_{h}",
    "label_x2_{h}",
    "x2_{h}",
    "target_2x_{h}",
    "label_2x_{h}",
    "target_hit_x2_{h}",
    "hit_x2_{h}",
    "manual_verified_x2_{h}",
    "target_profitable_x2_{h}",
    "target",
    "label",
    "y",
)


@dataclass
class RunSpec:
    input_path: str
    run_id: str
    filter_name: str | None
    horizon: str
    target_col: str


@dataclass
class RunResult:
    run_id: str
    input_path: str
    horizon: str
    target_col: str
    row_count: int
    train_rows: int
    validation_rows: int
    test_rows: int
    feature_count: int
    numeric_feature_count: int
    categorical_feature_count: int
    positive_train: int
    positive_validation: int
    positive_test: int
    validation_pr_auc: float | None
    validation_roc_auc: float | None
    validation_precision_top_1pct: float | None
    validation_precision_top_2pct: float | None
    validation_precision_top_5pct: float | None
    test_pr_auc: float | None
    test_roc_auc: float | None
    test_precision_top_1pct: float | None
    test_precision_top_2pct: float | None
    test_precision_top_5pct: float | None
    validation_output: str
    test_output: str
    metrics_output: str
    status: str
    error: str | None = None


def log(message: str) -> None:
    print(message, flush=True)


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def infer_filter_from_name(name: str) -> str | None:
    upper = name.upper()
    known = [
        "LIQ_5K_HIGH_ACTIVITY",
        "LOW_LIQ_MOMENTUM",
        "RAW_ALL_VERIFIED",
        "NO_WHALE_FILTER",
    ]
    for item in known:
        if item in upper:
            return item
    return None


def infer_horizon_from_name(name: str) -> str | None:
    lower = name.lower()
    for horizon in HORIZONS:
        if re.search(rf"(^|[_\-.]){re.escape(horizon)}($|[_\-.])", lower):
            return horizon
        if f"x2_{horizon}" in lower:
            return horizon
    return None


def make_run_id(path: Path, filter_name: str | None, horizon: str) -> str:
    stem = path.stem

    prefixes = (
        "clean_model_input_",
        "model_input_",
        "manual_verified_",
        "dataset_",
    )
    clean_stem = stem
    for prefix in prefixes:
        if clean_stem.lower().startswith(prefix):
            clean_stem = clean_stem[len(prefix) :]

    clean_stem = clean_stem.replace("__", "_").strip("_")

    if "clean" not in clean_stem.lower():
        base = f"CLEAN_{filter_name or 'UNKNOWN'}_x2_{horizon}"
    else:
        base = clean_stem

    base = re.sub(r"[^A-Za-z0-9_]+", "_", base).strip("_")
    return base


def discover_input_files(
    input_roots: list[Path],
    filters: list[str] | None,
    horizons: list[str],
    all_files: bool,
) -> list[Path]:
    candidates: list[Path] = []

    for root in input_roots:
        if not root.exists():
            continue

        for path in sorted(root.rglob("*.parquet")):
            lower = path.name.lower()
            full_lower = str(path).lower()

            if "xgb_clean" in full_lower:
                continue
            if "predictions" in lower:
                continue
            if "_xgb" in lower:
                continue
            if "clean" not in lower and not all_files:
                continue

            path_horizon = infer_horizon_from_name(path.name)
            if path_horizon is None:
                continue
            if path_horizon not in horizons:
                continue

            path_filter = infer_filter_from_name(path.name)
            if filters and path_filter not in filters:
                continue

            candidates.append(path)

    unique: dict[str, Path] = {}
    for path in candidates:
        unique[str(path.resolve())] = path

    return list(unique.values())


def infer_target_column(df: pd.DataFrame, horizon: str) -> str:
    columns = list(df.columns)
    lower_to_original = {c.lower(): c for c in columns}

    for template in TARGET_CANDIDATE_TEMPLATES:
        candidate = template.format(h=horizon).lower()
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    horizon_lower = horizon.lower()
    scored: list[tuple[int, str]] = []

    for col in columns:
        lower = col.lower()
        if horizon_lower not in lower:
            continue
        if "x2" not in lower and "2x" not in lower:
            continue
        if not any(token in lower for token in ("target", "label", "hit", "y")):
            continue

        series = df[col].dropna()
        if series.empty:
            continue

        unique_values = set(pd.Series(series).astype(str).unique())
        binary_like = unique_values.issubset({"0", "1", "False", "True", "false", "true"})
        if not binary_like:
            continue

        score = 0
        if "target" in lower:
            score += 5
        if "label" in lower:
            score += 4
        if "x2" in lower or "2x" in lower:
            score += 3
        scored.append((score, col))

    if scored:
        scored.sort(reverse=True)
        return scored[0][1]

    raise ValueError(
        f"Could not infer target column for horizon={horizon}. "
        f"Available columns include: {columns[:40]}"
    )


def to_binary_y(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)

    mapped = series.map(
        {
            True: 1,
            False: 0,
            "True": 1,
            "False": 0,
            "true": 1,
            "false": 0,
            "YES": 1,
            "NO": 0,
            "yes": 1,
            "no": 0,
        }
    )

    numeric = pd.to_numeric(series, errors="coerce")
    combined = mapped.where(mapped.notna(), numeric)
    combined = combined.fillna(0).astype(float)

    return (combined > 0).astype(int)


def split_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_col = None
    for candidate in ("split", "dataset_split", "fold"):
        if candidate in df.columns:
            split_col = candidate
            break

    if split_col:
        values = df[split_col].astype(str).str.lower()
        train = df[values.isin(["train", "training", "0"])]
        validation = df[values.isin(["validation", "valid", "val", "1"])]
        test = df[values.isin(["test", "2"])]

        if len(train) and len(validation) and len(test):
            return train.copy(), validation.copy(), test.copy()

    timestamp_col = None
    for candidate in ("event_timestamp", "timestamp", "created_at"):
        if candidate in df.columns:
            timestamp_col = candidate
            break

    if timestamp_col:
        ordered = df.copy()
        ordered[timestamp_col] = pd.to_datetime(
            ordered[timestamp_col],
            errors="coerce",
            utc=True,
        )
        ordered = ordered.sort_values(timestamp_col, kind="mergesort")
    else:
        ordered = df.reset_index(drop=True).copy()

    n = len(ordered)
    train_end = int(n * 0.70)
    validation_end = int(n * 0.85)

    train = ordered.iloc[:train_end].copy()
    validation = ordered.iloc[train_end:validation_end].copy()
    test = ordered.iloc[validation_end:].copy()

    return train, validation, test


def is_leakage_column(col: str, target_col: str) -> bool:
    lower = col.lower()

    if col == target_col:
        return True
    if lower in NON_FEATURE_EXACT:
        return True
    if lower in {c.lower() for c in METADATA_COLUMNS}:
        return True

    return any(token in lower for token in LEAKAGE_SUBSTRINGS)


def choose_feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    feature_cols: list[str] = []

    for col in df.columns:
        if is_leakage_column(col, target_col):
            continue

        series = df[col]

        if series.dtype == object:
            nunique = series.dropna().astype(str).nunique()
            if nunique > 200:
                continue

        feature_cols.append(col)

    if not feature_cols:
        raise ValueError("No usable feature columns after leakage/metadata exclusion.")

    return feature_cols


def build_output_frame(
    source: pd.DataFrame,
    y_true: pd.Series,
    score: np.ndarray,
    run_id: str,
    horizon: str,
    target_col: str,
) -> pd.DataFrame:
    output = pd.DataFrame(index=source.index)

    for col in METADATA_COLUMNS:
        if col in source.columns:
            output[col] = source[col]

    output["run_id"] = run_id
    output["model"] = "XGB"
    output["horizon"] = horizon
    output["target_col"] = target_col
    output["y_true"] = y_true.astype(int).values
    output["xgb_probability"] = score.astype(float)
    output["score"] = score.astype(float)

    return output.reset_index(drop=False).rename(columns={"index": "source_index"})


def safe_metric(metric_fn, y_true: pd.Series, score: np.ndarray) -> float | None:
    try:
        if len(set(pd.Series(y_true).dropna().astype(int).tolist())) < 2:
            return None
        value = metric_fn(y_true, score)
        if value is None or np.isnan(value):
            return None
        return float(value)
    except Exception:
        return None


def precision_at_fraction(y_true: pd.Series, score: np.ndarray, fraction: float) -> float | None:
    if len(y_true) == 0:
        return None

    n = max(1, int(np.ceil(len(y_true) * fraction)))
    order = np.argsort(-score)[:n]
    selected = pd.Series(y_true).iloc[order].astype(int)

    if len(selected) == 0:
        return None

    return float(selected.mean())


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_json(payload: dict[str, Any] | list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def clean_temp_files(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0

    count = 0
    for path in output_dir.glob(".*.tmp.*"):
        path.unlink(missing_ok=True)
        count += 1
    return count


def clean_existing_xgb_outputs_for_run(output_dir: Path, run_id: str) -> int:
    if not output_dir.exists():
        return 0

    patterns = [
        f"xgb_predictions_validation_{run_id}_XGB.parquet",
        f"xgb_predictions_test_{run_id}_XGB.parquet",
        f"xgb_metrics_{run_id}_XGB.json",
    ]

    removed = 0
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok=True)
            removed += 1

    return removed


def clean_all_xgb_outputs(output_dir: Path) -> int:
    if not output_dir.exists():
        return 0

    patterns = [
        "xgb_predictions_*_XGB.parquet",
        "xgb_metrics_*_XGB.json",
        "xgb_clean_summary.json",
    ]

    removed = 0
    for pattern in patterns:
        for path in output_dir.glob(pattern):
            path.unlink(missing_ok=True)
            removed += 1

    return removed


def make_preprocessor(train_x: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    numeric_cols: list[str] = []
    categorical_cols: list[str] = []

    for col in train_x.columns:
        if pd.api.types.is_numeric_dtype(train_x[col]) or pd.api.types.is_bool_dtype(train_x[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    transformers = []

    if numeric_cols:
        transformers.append(
            (
                "num",
                SimpleImputer(strategy="median"),
                numeric_cols,
            )
        )

    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        ),
                    ]
                ),
                categorical_cols,
            )
        )

    if not transformers:
        raise ValueError("No numeric or categorical columns available for preprocessing.")

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor, numeric_cols, categorical_cols


def get_positive_probability(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(x)

    estimator = model.named_steps["model"]
    classes = list(getattr(estimator, "classes_", []))

    positive_idx = None
    for idx, cls in enumerate(classes):
        if cls in (1, True, "1", "true", "True", "profit", "positive"):
            positive_idx = idx
            break

    if positive_idx is None:
        if len(classes) == 2:
            positive_idx = 1
        else:
            raise ValueError(f"Could not identify positive class from classes_={classes}")

    return np.asarray(proba[:, positive_idx], dtype=float)


def train_and_predict(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    random_state: int,
    n_jobs: int,
) -> tuple[Pipeline, dict[str, Any]]:
    y_train = to_binary_y(train[target_col])
    y_validation = to_binary_y(validation[target_col])
    y_test = to_binary_y(test[target_col])

    if y_train.nunique() < 2:
        raise ValueError(
            f"Training target has only one class. positives={int(y_train.sum())}, rows={len(y_train)}"
        )

    train_x = train[feature_cols].copy()
    validation_x = validation[feature_cols].copy()
    test_x = test[feature_cols].copy()

    preprocessor, numeric_cols, categorical_cols = make_preprocessor(train_x)

    classifier = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        device="cuda",
        random_state=random_state,
        n_jobs=n_jobs,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", classifier),
        ]
    )

    model.fit(train_x, y_train)

    validation_score = get_positive_probability(model, validation_x)
    test_score = get_positive_probability(model, test_x)

    payload = {
        "model": model,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "y_train": y_train,
        "y_validation": y_validation,
        "y_test": y_test,
        "validation_score": validation_score,
        "test_score": test_score,
    }

    return model, payload


def evaluate_one_file(
    input_path: Path,
    output_dir: Path,
    clean_existing: bool,
    random_state: int,
    n_jobs: int,
) -> RunResult:
    filter_name = infer_filter_from_name(input_path.name)
    horizon = infer_horizon_from_name(input_path.name)

    if horizon is None:
        raise ValueError(f"Could not infer horizon from input filename: {input_path.name}")

    run_id = make_run_id(input_path, filter_name, horizon)
    target_col = ""

    validation_output = output_dir / f"xgb_predictions_validation_{run_id}_XGB.parquet"
    test_output = output_dir / f"xgb_predictions_test_{run_id}_XGB.parquet"
    metrics_output = output_dir / f"xgb_metrics_{run_id}_XGB.json"

    try:
        log(f"\n=== XGB CLEAN RUN: {run_id} ===")
        log(f"input: {input_path}")

        if clean_existing:
            removed = clean_existing_xgb_outputs_for_run(output_dir, run_id)
            if removed:
                log(f"removed existing XGB outputs for {run_id}: {removed}")

        df = pd.read_parquet(input_path)

        if df.empty:
            raise ValueError(f"Input parquet is empty: {input_path}")

        target_col = infer_target_column(df, horizon)
        train, validation, test = split_dataframe(df)

        if len(train) == 0 or len(validation) == 0 or len(test) == 0:
            raise ValueError(
                f"Invalid split sizes: train={len(train)}, validation={len(validation)}, test={len(test)}"
            )

        feature_cols = choose_feature_columns(df, target_col)

        model, payload = train_and_predict(
            train=train,
            validation=validation,
            test=test,
            feature_cols=feature_cols,
            target_col=target_col,
            random_state=random_state,
            n_jobs=n_jobs,
        )

        y_train = payload["y_train"]
        y_validation = payload["y_validation"]
        y_test = payload["y_test"]
        validation_score = payload["validation_score"]
        test_score = payload["test_score"]

        validation_predictions = build_output_frame(
            source=validation,
            y_true=y_validation,
            score=validation_score,
            run_id=run_id,
            horizon=horizon,
            target_col=target_col,
        )
        test_predictions = build_output_frame(
            source=test,
            y_true=y_test,
            score=test_score,
            run_id=run_id,
            horizon=horizon,
            target_col=target_col,
        )

        metrics = {
            "run_id": run_id,
            "model": "XGB",
            "input_path": str(input_path),
            "horizon": horizon,
            "filter_name": filter_name,
            "target_col": target_col,
            "row_count": int(len(df)),
            "train_rows": int(len(train)),
            "validation_rows": int(len(validation)),
            "test_rows": int(len(test)),
            "feature_count": int(len(feature_cols)),
            "features": feature_cols,
            "numeric_feature_count": int(len(payload["numeric_cols"])),
            "numeric_features": payload["numeric_cols"],
            "categorical_feature_count": int(len(payload["categorical_cols"])),
            "categorical_features": payload["categorical_cols"],
            "positive_train": int(y_train.sum()),
            "positive_validation": int(y_validation.sum()),
            "positive_test": int(y_test.sum()),
            "validation_pr_auc": safe_metric(average_precision_score, y_validation, validation_score),
            "validation_roc_auc": safe_metric(roc_auc_score, y_validation, validation_score),
            "validation_precision_top_1pct": precision_at_fraction(y_validation, validation_score, 0.01),
            "validation_precision_top_2pct": precision_at_fraction(y_validation, validation_score, 0.02),
            "validation_precision_top_5pct": precision_at_fraction(y_validation, validation_score, 0.05),
            "test_pr_auc": safe_metric(average_precision_score, y_test, test_score),
            "test_roc_auc": safe_metric(roc_auc_score, y_test, test_score),
            "test_precision_top_1pct": precision_at_fraction(y_test, test_score, 0.01),
            "test_precision_top_2pct": precision_at_fraction(y_test, test_score, 0.02),
            "test_precision_top_5pct": precision_at_fraction(y_test, test_score, 0.05),
            "validation_output": str(validation_output),
            "test_output": str(test_output),
            "metrics_output": str(metrics_output),
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
            "artifact_write_mode": "atomic_replace",
            "clean_existing_xgb_outputs_for_run": bool(clean_existing),
        }

        atomic_write_parquet(validation_predictions, validation_output)
        atomic_write_parquet(test_predictions, test_output)
        atomic_write_json(metrics, metrics_output)

        log(f"target: {target_col}")
        log(f"features: {len(feature_cols)}")
        log(f"validation rows: {len(validation)}, positives: {int(y_validation.sum())}")
        log(f"test rows: {len(test)}, positives: {int(y_test.sum())}")
        log(f"validation PR-AUC: {metrics['validation_pr_auc']}")
        log(f"test PR-AUC: {metrics['test_pr_auc']}")
        log(f"wrote: {validation_output}")
        log(f"wrote: {test_output}")
        log(f"wrote: {metrics_output}")

        return RunResult(
            run_id=run_id,
            input_path=str(input_path),
            horizon=horizon,
            target_col=target_col,
            row_count=int(len(df)),
            train_rows=int(len(train)),
            validation_rows=int(len(validation)),
            test_rows=int(len(test)),
            feature_count=int(len(feature_cols)),
            numeric_feature_count=int(len(payload["numeric_cols"])),
            categorical_feature_count=int(len(payload["categorical_cols"])),
            positive_train=int(y_train.sum()),
            positive_validation=int(y_validation.sum()),
            positive_test=int(y_test.sum()),
            validation_pr_auc=metrics["validation_pr_auc"],
            validation_roc_auc=metrics["validation_roc_auc"],
            validation_precision_top_1pct=metrics["validation_precision_top_1pct"],
            validation_precision_top_2pct=metrics["validation_precision_top_2pct"],
            validation_precision_top_5pct=metrics["validation_precision_top_5pct"],
            test_pr_auc=metrics["test_pr_auc"],
            test_roc_auc=metrics["test_roc_auc"],
            test_precision_top_1pct=metrics["test_precision_top_1pct"],
            test_precision_top_2pct=metrics["test_precision_top_2pct"],
            test_precision_top_5pct=metrics["test_precision_top_5pct"],
            validation_output=str(validation_output),
            test_output=str(test_output),
            metrics_output=str(metrics_output),
            status="ok",
            error=None,
        )

    except Exception as exc:
        log(f"ERROR in {input_path}: {exc}")

        error_payload = {
            "run_id": run_id,
            "input_path": str(input_path),
            "horizon": horizon,
            "target_col": target_col,
            "status": "error",
            "error": str(exc),
            "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        }
        atomic_write_json(error_payload, metrics_output)

        return RunResult(
            run_id=run_id,
            input_path=str(input_path),
            horizon=horizon,
            target_col=target_col,
            row_count=0,
            train_rows=0,
            validation_rows=0,
            test_rows=0,
            feature_count=0,
            numeric_feature_count=0,
            categorical_feature_count=0,
            positive_train=0,
            positive_validation=0,
            positive_test=0,
            validation_pr_auc=None,
            validation_roc_auc=None,
            validation_precision_top_1pct=None,
            validation_precision_top_2pct=None,
            validation_precision_top_5pct=None,
            test_pr_auc=None,
            test_roc_auc=None,
            test_precision_top_1pct=None,
            test_precision_top_2pct=None,
            test_precision_top_5pct=None,
            validation_output=str(validation_output),
            test_output=str(test_output),
            metrics_output=str(metrics_output),
            status="error",
            error=str(exc),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate XGB on manual-verified CLEAN inputs with safe atomic artifact overwrite."
    )

    parser.add_argument(
        "--input-root",
        action="append",
        default=None,
        help="Input root containing CLEAN parquet files. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for XGB artifacts.",
    )
    parser.add_argument(
        "--filter",
        action="append",
        default=None,
        help="Filter name to include, e.g. LIQ_5K_HIGH_ACTIVITY. Can be passed multiple times.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=list(HORIZONS),
        choices=list(HORIZONS),
        help="Horizons to evaluate.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Allow all matching parquet files under the input root, even if file name does not include CLEAN.",
    )
    parser.add_argument(
        "--clean-existing",
        action="store_true",
        help="Remove existing XGB outputs for each selected run before writing fresh outputs.",
    )
    parser.add_argument(
        "--clean-all-xgb",
        action="store_true",
        help="Remove all XGB outputs in output-dir before running. Does not touch RF/TAB outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print discovered input files and planned run IDs. Do not train/write.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state for XGB.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of CPU threads for XGB.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    input_roots = (
        [Path(p).resolve() for p in args.input_root]
        if args.input_root
        else DEFAULT_INPUT_ROOTS
    )
    output_dir = Path(args.output_dir).resolve()
    filters = args.filter

    output_dir.mkdir(parents=True, exist_ok=True)

    removed_tmp = clean_temp_files(output_dir)
    if removed_tmp:
        log(f"removed stale temp files: {removed_tmp}")

    if args.clean_all_xgb and not args.dry_run:
        removed = clean_all_xgb_outputs(output_dir)
        log(f"removed all existing XGB artifacts in output dir: {removed}")

    input_files = discover_input_files(
        input_roots=input_roots,
        filters=filters,
        horizons=args.horizons,
        all_files=args.all,
    )

    if not input_files:
        log("No matching CLEAN parquet inputs found.")
        log("Searched roots:")
        for root in input_roots:
            log(f"  - {root}")
        log("Use --input-root to point to the folder that contains CLEAN model input parquet files.")
        return 2

    planned: list[dict[str, Any]] = []
    for path in input_files:
        horizon = infer_horizon_from_name(path.name)
        filter_name = infer_filter_from_name(path.name)
        if horizon is None:
            continue
        run_id = make_run_id(path, filter_name, horizon)
        planned.append(
            {
                "input_path": str(path),
                "run_id": run_id,
                "filter_name": filter_name,
                "horizon": horizon,
                "validation_output": str(output_dir / f"xgb_predictions_validation_{run_id}_XGB.parquet"),
                "test_output": str(output_dir / f"xgb_predictions_test_{run_id}_XGB.parquet"),
                "metrics_output": str(output_dir / f"xgb_metrics_{run_id}_XGB.json"),
            }
        )

    log("\n=== Planned XGB runs ===")
    for item in planned:
        log(f"- {item['run_id']} :: {item['input_path']}")

    if args.dry_run:
        return 0

    results: list[RunResult] = []

    started = time.time()

    for path in input_files:
        result = evaluate_one_file(
            input_path=path,
            output_dir=output_dir,
            clean_existing=args.clean_existing,
            random_state=args.random_state,
            n_jobs=args.n_jobs,
        )
        results.append(result)

    summary = {
        "status": "ok" if all(r.status == "ok" for r in results) else "partial_or_error",
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "duration_seconds": round(time.time() - started, 3),
        "output_dir": str(output_dir),
        "run_count": len(results),
        "ok_count": sum(1 for r in results if r.status == "ok"),
        "error_count": sum(1 for r in results if r.status != "ok"),
        "clean_existing": bool(args.clean_existing),
        "clean_all_xgb": bool(args.clean_all_xgb),
        "artifact_write_mode": "atomic_replace",
        "results": [asdict(r) for r in results],
    }

    summary_path = output_dir / "xgb_clean_summary.json"
    atomic_write_json(summary, summary_path)

    log("\n=== XGB summary ===")
    log(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    log(f"\nwrote summary: {summary_path}")

    return 0 if summary["error_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())