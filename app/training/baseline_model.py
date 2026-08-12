"""Baseline sklearn training for whale-wave model-ready dataset."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "data" / "training"
MODELS_DIR = TRAINING_DIR / "models"
DEFAULT_DATASET_PATH = TRAINING_DIR / "model_ready_dataset.parquet"

LEAKAGE_SUBSTRINGS = (
    "future_return",
    "max_future_return",
    "min_future_return",
    "label_",
    "big_pump",
    "big_dump",
    "pump_then_dump",
    "optimal_trade_class",
    "position_size_multiplier",
    "profitable_after_fees",
    "positive_return",
    "outcome",
    "realized",
    "oracle",
)

IDENTIFIER_EXACT = frozenset({
    "event_timestamp",
    "timestamp",
    "symbol",
    "pair_address",
    "coin_id",
    "raw_json",
    "reasoning",
    "prompt",
    "response",
    "decision",
    "action",
})

ADDITIONAL_LEAKAGE_EXACT = frozenset({
    "future_price_15m",
    "future_price_1h",
    "future_price_4h",
    "target_return_15m",
    "target_return_1h",
    "target_return_4h",
    "target_profitable_15m",
    "target_profitable_1h",
    "target_profitable_4h",
    "max_upside_15m",
    "max_upside_1h",
    "max_upside_4h",
    "max_drawdown_15m",
    "max_drawdown_1h",
    "max_drawdown_4h",
    "label_up_15m",
    "label_up_1h",
    "label_up_4h",
    "pending_outcome",
    "outcome_pnl",
    "outcome_status",
    "ts",
    "id",
    "signal_id",
    "decision_id",
    "source_id",
    "linked_trade_id",
    "coin_chain",
    "coin_pair_address",
    "coin_symbol",
    "pair_address_snap",
    "features_json",
    "input_context_json",
    "gemini_response_json",
    "response_json",
    "prompt_summary",
    "rationale",
    "reason",
    "reasoning_summary",
    "provider",
    "model_source",
    "source_kind",
    "source_type",
    "llm_action",
    "signal_action",
    "signal_type",
    "strategy_type",
    "trigger_type",
    "cluster_label",
})

SAFE_CATEGORICAL = frozenset({"whale_wave_direction"})

TARGET_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "label_profitable_after_fees_1h",
        "aliases": ["profitable_after_fees_1h", "target_profitable_1h"],
    },
    {
        "name": "label_profitable_after_fees_4h",
        "aliases": ["profitable_after_fees_4h", "target_profitable_4h"],
    },
    {"name": "big_pump_1h", "aliases": []},
    {"name": "big_pump_4h", "aliases": []},
    {
        "name": "is_aggressive_whale_trade_1h",
        "aliases": [],
        "derive_from": "optimal_trade_class_1h",
        "derive_value": "AGGRESSIVE_WHALE_TRADE",
    },
    {
        "name": "is_aggressive_whale_trade_4h",
        "aliases": [],
        "derive_from": "optimal_trade_class_4h",
        "derive_value": "AGGRESSIVE_WHALE_TRADE",
    },
]

PREDICTION_META_COLUMNS = [
    "event_timestamp",
    "symbol",
    "pair_address",
    "target_return_1h",
    "target_return_4h",
    "future_return_1h",
    "future_return_4h",
    "whale_wave_score",
    "optimal_trade_class_1h",
    "optimal_trade_class_4h",
]


def _contains_leakage(name: str) -> bool:
    lower = name.lower()
    return any(sub in lower for sub in LEAKAGE_SUBSTRINGS)


def select_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return (numeric_features, categorical_features, excluded_columns)."""
    excluded: list[str] = []
    numeric: list[str] = []
    categorical: list[str] = []

    for col in frame.columns:
        if col in IDENTIFIER_EXACT or col in ADDITIONAL_LEAKAGE_EXACT or _contains_leakage(col):
            excluded.append(col)
            continue
        if col in SAFE_CATEGORICAL:
            non_null = frame[col].dropna()
            if non_null.empty or non_null.nunique() > 32:
                excluded.append(col)
            else:
                categorical.append(col)
            continue
        dtype = frame[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            numeric.append(col)
            continue
        if pd.api.types.is_bool_dtype(dtype):
            numeric.append(col)
            continue
        excluded.append(col)

    return numeric, categorical, excluded


def resolve_target_column(frame: pd.DataFrame, spec: dict[str, Any]) -> tuple[str | None, pd.Series | None]:
    for candidate in [spec["name"], *spec.get("aliases", [])]:
        if candidate in frame.columns:
            return spec["name"], frame[candidate]
    derive_from = spec.get("derive_from")
    if derive_from and derive_from in frame.columns:
        derive_value = spec.get("derive_value", "AGGRESSIVE_WHALE_TRADE")
        series = (frame[derive_from] == derive_value).astype(int)
        return spec["name"], series
    return None, None


def chronological_split(
    frame: pd.DataFrame,
    *,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("event_timestamp").reset_index(drop=True)
    n = len(ordered)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = ordered.iloc[:train_end].copy()
    val = ordered.iloc[train_end:val_end].copy()
    test = ordered.iloc[val_end:].copy()
    return train, val, test


def assert_chronological_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    timestamp_col: str = "event_timestamp",
) -> None:
    """Fail loudly if chronological split boundaries are violated."""
    for name, frame in (("train", train_df), ("validation", val_df), ("test", test_df)):
        if timestamp_col not in frame.columns:
            raise AssertionError(f"{name} split missing {timestamp_col}")
        if not frame[timestamp_col].is_monotonic_increasing:
            raise AssertionError(f"{name} split is not sorted by {timestamp_col}")

    train_max = pd.to_datetime(train_df[timestamp_col], utc=True).max()
    val_min = pd.to_datetime(val_df[timestamp_col], utc=True).min()
    val_max = pd.to_datetime(val_df[timestamp_col], utc=True).max()
    test_min = pd.to_datetime(test_df[timestamp_col], utc=True).min()
    if train_max >= val_min:
        raise AssertionError(
            f"train/validation overlap: max(train)={train_max} >= min(val)={val_min}"
        )
    if val_max >= test_min:
        raise AssertionError(
            f"validation/test overlap: max(val)={val_max} >= min(test)={test_min}"
        )


def sanitize_feature_frame(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    subset = frame[feature_cols].copy()
    for col in feature_cols:
        if pd.api.types.is_numeric_dtype(subset[col]):
            subset[col] = subset[col].replace([np.inf, -np.inf], np.nan)
    return subset


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric_features:
        transformers.append(
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]),
                categorical_features,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def get_model_specs() -> list[tuple[str, Any]]:
    models: list[tuple[str, Any]] = [
        (
            "logistic_regression",
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        ),
        (
            "random_forest",
            RandomForestClassifier(
                class_weight="balanced",
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=20,
                n_jobs=-1,
                random_state=42,
            ),
        ),
    ]
    models.append(
        (
            "hist_gradient_boosting",
            HistGradientBoostingClassifier(
                class_weight="balanced",
                max_iter=100,
                max_depth=8,
                random_state=42,
            ),
        )
    )
    return models


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def _safe_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(average_precision_score(y_true, y_score))
    except ValueError:
        return None


def precision_at_top_k(y_true: np.ndarray, y_score: np.ndarray, k_pct: float) -> float | None:
    if len(y_true) == 0:
        return None
    k = max(1, int(len(y_true) * k_pct / 100.0))
    order = np.argsort(-y_score)
    return float(y_true[order[:k]].mean())


def recall_at_top_k(y_true: np.ndarray, y_score: np.ndarray, k_pct: float) -> float | None:
    positives = float(y_true.sum())
    if positives == 0:
        return None
    k = max(1, int(len(y_true) * k_pct / 100.0))
    order = np.argsort(-y_score)
    return float(y_true[order[:k]].sum() / positives)


def confusion_at_threshold(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, int]:
    pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def confusion_at_top_percent(y_true: np.ndarray, y_score: np.ndarray, top_pct: float) -> dict[str, int]:
    if len(y_true) == 0:
        return {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    k = max(1, int(len(y_true) * top_pct / 100.0))
    order = np.argsort(-y_score)
    pred = np.zeros_like(y_true)
    pred[order[:k]] = 1
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def best_threshold_by_f1(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    best_th = 0.5
    best_f1 = -1.0
    for th in np.linspace(0.05, 0.95, 19):
        pred = (y_score >= th).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_th = float(th)
    return best_th


def metrics_at_split(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    best_threshold: float | None,
) -> dict[str, Any]:
    best_th = best_threshold if best_threshold is not None else 0.5
    test_at_best: dict[str, Any] = {}
    if best_threshold is not None and len(np.unique(y_true)) >= 2:
        pred = (y_score >= best_th).astype(int)
        test_at_best = {
            "precision": round(float(precision_score(y_true, pred, zero_division=0)), 6),
            "recall": round(float(recall_score(y_true, pred, zero_division=0)), 6),
            "f1": round(float(f1_score(y_true, pred, zero_division=0)), 6),
            "confusion_matrix": confusion_at_threshold(y_true, y_score, best_th),
        }

    return {
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "pr_auc": _safe_pr_auc(y_true, y_score),
        "precision_at_top_1_percent": precision_at_top_k(y_true, y_score, 1.0),
        "precision_at_top_5_percent": precision_at_top_k(y_true, y_score, 5.0),
        "recall_at_top_1_percent": recall_at_top_k(y_true, y_score, 1.0),
        "recall_at_top_5_percent": recall_at_top_k(y_true, y_score, 5.0),
        "confusion_matrix_threshold_0_5": confusion_at_threshold(y_true, y_score, 0.5),
        "confusion_matrix_top_5_percent": confusion_at_top_percent(y_true, y_score, 5.0),
        "best_threshold_by_validation_f1": best_threshold,
        "test_metrics_at_best_validation_threshold": test_at_best,
    }


def _class_counts(series: pd.Series) -> dict[str, Any]:
    y = series.fillna(0).astype(int)
    pos = int(y.sum())
    total = int(len(y))
    return {
        "positive_count": pos,
        "positive_rate": round(pos / total, 6) if total else 0.0,
    }


def extract_feature_importance(
    model_name: str,
    fitted_pipeline: Pipeline,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, float] | None:
    classifier = fitted_pipeline.named_steps["classifier"]
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    if model_name == "random_forest" and hasattr(classifier, "feature_importances_"):
        try:
            names = preprocessor.get_feature_names_out()
            values = classifier.feature_importances_
            pairs = sorted(zip(names, values), key=lambda x: -x[1])[:40]
            return {str(n): round(float(v), 6) for n, v in pairs}
        except Exception:
            return None
    if model_name == "logistic_regression" and hasattr(classifier, "coef_"):
        try:
            names = preprocessor.get_feature_names_out()
            coef = np.abs(classifier.coef_[0])
            pairs = sorted(zip(names, coef), key=lambda x: -x[1])[:40]
            return {str(n): round(float(v), 6) for n, v in pairs}
        except Exception:
            return None
    return None


def build_prediction_rows(
    frame: pd.DataFrame,
    *,
    target_name: str,
    y_true: pd.Series,
    y_score: np.ndarray,
    model_name: str,
    split: str,
) -> pd.DataFrame:
    rows = pd.DataFrame({
        "event_timestamp": frame.get("event_timestamp"),
        "symbol": frame.get("symbol"),
        "pair_address": frame.get("pair_address"),
        "target_name": target_name,
        "y_true": y_true.fillna(0).astype(int).values,
        "predicted_probability": y_score,
        "model_name": model_name,
        "split": split,
    })
    for col in PREDICTION_META_COLUMNS:
        if col in frame.columns and col not in rows.columns:
            rows[col] = frame[col].values
    return rows


def train_baseline_models(
    *,
    dataset_path: Path | None = None,
    models_dir: Path | None = None,
    max_rows: int | None = None,
) -> dict[str, Any]:
    dataset_path = dataset_path or DEFAULT_DATASET_PATH
    models_dir = models_dir or MODELS_DIR
    models_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    frame = pd.read_parquet(dataset_path)
    if "event_timestamp" not in frame.columns:
        raise ValueError("model_ready_dataset must include event_timestamp")

    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"], utc=True, errors="coerce")
    frame = frame[frame["event_timestamp"].notna()].copy()
    frame = frame.sort_values("event_timestamp").reset_index(drop=True)
    if max_rows is not None and len(frame) > max_rows:
        frame = frame.iloc[-max_rows:].reset_index(drop=True)

    numeric_features, categorical_features, excluded = select_feature_columns(frame)
    feature_cols = numeric_features + categorical_features
    train_df, val_df, test_df = chronological_split(frame)

    assert_chronological_splits(train_df, val_df, test_df)

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "rows_total": int(len(frame)),
        "rows_train": int(len(train_df)),
        "rows_validation": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "features_used_count": len(feature_cols),
        "features_excluded_count": len(excluded),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "excluded_features": excluded,
        "targets_trained": [],
        "targets_skipped": [],
        "models_by_target": {},
        "best_model_by_target": {},
        "feature_importance": {},
    }

    val_predictions: list[pd.DataFrame] = []
    test_predictions: list[pd.DataFrame] = []

    for spec in TARGET_DEFINITIONS:
        target_name = spec["name"]
        _, y_series = resolve_target_column(frame, spec)
        if y_series is None:
            report["targets_skipped"].append({
                "target": target_name,
                "reason": "column_missing",
            })
            continue

        y_all = y_series.fillna(0).astype(int)
        if y_all.nunique() < 2:
            report["targets_skipped"].append({
                "target": target_name,
                "reason": "fewer_than_two_classes",
                "positive_count": int(y_all.sum()),
            })
            continue

        y_train = y_all.loc[train_df.index].astype(int)
        y_val = y_all.loc[val_df.index].astype(int)
        y_test = y_all.loc[test_df.index].astype(int)

        X_train = sanitize_feature_frame(train_df, feature_cols)
        X_val = sanitize_feature_frame(val_df, feature_cols)
        X_test = sanitize_feature_frame(test_df, feature_cols)

        target_entry: dict[str, Any] = {
            "rows_train": int(len(train_df)),
            "rows_validation": int(len(val_df)),
            "rows_test": int(len(test_df)),
            "train": _class_counts(y_train),
            "validation": _class_counts(y_val),
            "test": _class_counts(y_test),
            "models": {},
        }

        best_model_name: str | None = None
        best_val_pr: float = -1.0
        best_val_threshold: float | None = None

        for model_name, estimator in get_model_specs():
            preprocessor = build_preprocessor(numeric_features, categorical_features)
            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("classifier", estimator),
            ])
            pipeline.fit(X_train, y_train)

            val_scores = pipeline.predict_proba(X_val)[:, 1]
            test_scores = pipeline.predict_proba(X_test)[:, 1]

            y_val_arr = y_val.to_numpy()
            y_test_arr = y_test.to_numpy()

            val_th = best_threshold_by_f1(y_val_arr, val_scores)
            val_metrics = metrics_at_split(y_val_arr, val_scores, best_threshold=val_th)
            val_metrics.update({
                "rows_train": int(len(train_df)),
                "rows_validation": int(len(val_df)),
                "rows_test": int(len(test_df)),
                "positive_count_train": target_entry["train"]["positive_count"],
                "positive_count_validation": target_entry["validation"]["positive_count"],
                "positive_count_test": target_entry["test"]["positive_count"],
                "positive_rate_train": target_entry["train"]["positive_rate"],
                "positive_rate_validation": target_entry["validation"]["positive_rate"],
                "positive_rate_test": target_entry["test"]["positive_rate"],
            })

            test_metrics = metrics_at_split(y_test_arr, test_scores, best_threshold=val_th)
            test_metrics.update({
                "rows_train": int(len(train_df)),
                "rows_validation": int(len(val_df)),
                "rows_test": int(len(test_df)),
                "positive_count_train": target_entry["train"]["positive_count"],
                "positive_count_validation": target_entry["validation"]["positive_count"],
                "positive_count_test": target_entry["test"]["positive_count"],
                "positive_rate_train": target_entry["train"]["positive_rate"],
                "positive_rate_validation": target_entry["validation"]["positive_rate"],
                "positive_rate_test": target_entry["test"]["positive_rate"],
            })

            model_key = f"{target_name}__{model_name}"
            joblib.dump(pipeline, models_dir / f"{model_key}.joblib")

            importance = extract_feature_importance(model_name, pipeline, numeric_features, categorical_features)
            if importance:
                report["feature_importance"][model_key] = importance

            target_entry["models"][model_name] = {
                "validation": val_metrics,
                "test": test_metrics,
                "model_path": str(models_dir / f"{model_key}.joblib"),
            }

            val_pr = val_metrics.get("pr_auc")
            if val_pr is not None and val_pr > best_val_pr:
                best_val_pr = val_pr
                best_model_name = model_name
                best_val_threshold = val_th

            val_predictions.append(
                build_prediction_rows(
                    val_df, target_name=target_name, y_true=y_val,
                    y_score=val_scores, model_name=model_name, split="validation",
                )
            )
            if model_name == best_model_name or best_model_name is None:
                pass  # defer best-only test preds until loop ends

        if best_model_name is None:
            report["targets_skipped"].append({
                "target": target_name,
                "reason": "no_valid_model_metrics",
            })
            continue

        report["targets_trained"].append(target_name)
        report["models_by_target"][target_name] = target_entry

        best_key = f"{target_name}__{best_model_name}"
        joblib.dump(
            joblib.load(models_dir / f"{best_key}.joblib"),
            models_dir / f"{target_name}_best.joblib",
        )

        best_test = target_entry["models"][best_model_name]["test"]
        report["best_model_by_target"][target_name] = {
            "model_name": best_model_name,
            "best_validation_pr_auc": round(best_val_pr, 6),
            "best_validation_threshold": best_val_threshold,
            "test_pr_auc": best_test.get("pr_auc"),
            "test_roc_auc": best_test.get("roc_auc"),
            "test_precision_at_top_1_percent": best_test.get("precision_at_top_1_percent"),
            "test_precision_at_top_5_percent": best_test.get("precision_at_top_5_percent"),
        }

        # Best-model-only test predictions
        best_pipeline = joblib.load(models_dir / f"{best_key}.joblib")
        best_test_scores = best_pipeline.predict_proba(sanitize_feature_frame(test_df, feature_cols))[:, 1]
        test_predictions.append(
            build_prediction_rows(
                test_df, target_name=target_name, y_true=y_test,
                y_score=best_test_scores, model_name=best_model_name, split="test",
            )
        )

    if val_predictions:
        pd.concat(val_predictions, ignore_index=True).to_parquet(
            models_dir / "predictions_validation.parquet", index=False
        )
    if test_predictions:
        pd.concat(test_predictions, ignore_index=True).to_parquet(
            models_dir / "predictions_test.parquet", index=False
        )

    metrics_path = models_dir / "baseline_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    importance_path = models_dir / "feature_importance.json"
    with open(importance_path, "w", encoding="utf-8") as handle:
        json.dump(report["feature_importance"], handle, indent=2)

    report["output_files"] = [
        str(metrics_path),
        str(models_dir / "predictions_validation.parquet"),
        str(models_dir / "predictions_test.parquet"),
        str(importance_path),
    ]
    return report


def load_baseline_metrics(models_dir: Path | None = None) -> dict[str, Any] | None:
    models_dir = models_dir or MODELS_DIR
    path = models_dir / "baseline_metrics.json"
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
