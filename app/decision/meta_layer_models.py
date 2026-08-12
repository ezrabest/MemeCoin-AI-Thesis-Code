"""AE7 FINAL meta-layer model comparators and evaluations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class ModelEvalResult:
    approach: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "approach": self.approach,
            "status": self.status,
            "metrics": self.metrics,
            "reason": self.reason,
        }


def _prepare_xy(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    split_name: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    data = frame
    if split_name and "split" in frame.columns:
        data = frame[frame["split"] == split_name]
    X = data[feature_columns].copy()
    y = data[target_column].astype(int)
    return X, y


def _numeric_and_categorical_columns(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical = [c for c in X.columns if c not in numeric]
    return numeric, categorical


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric, categorical = _numeric_and_categorical_columns(X)
    transformers = []
    if numeric:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers=transformers)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return None


def evaluate_rule_baseline(
    frame: pd.DataFrame,
    target_column: str,
    *,
    split_eval: str = "validation",
) -> ModelEvalResult:
    """Rule comparator using vote_count / consensus tier."""
    data = frame
    if "split" in frame.columns:
        data = frame[frame["split"] == split_eval]
    if data.empty:
        return ModelEvalResult("rule_based_tier_comparator", "BLOCKED_NO_SPLIT_DATA")

    y_true = data[target_column].astype(int).to_numpy()
    if "vote_count" in data.columns:
        y_score = (data["vote_count"].fillna(0) >= 2).astype(float).to_numpy()
    elif "consensus_tier" in data.columns:
        y_score = (~data["consensus_tier"].astype(str).str.contains("ONLY", na=False)).astype(float).to_numpy()
    else:
        return ModelEvalResult("rule_based_tier_comparator", "BLOCKED_MISSING_CONSENSUS_FIELDS")

    selected = int(y_score.sum())
    precision = float(y_true[y_score >= 0.5].mean()) if selected else 0.0
    auc = _safe_auc(y_true, y_score)
    return ModelEvalResult(
        approach="rule_based_tier_comparator",
        status="PASS",
        metrics={
            "selected_count": selected,
            "precision_at_rule": precision,
            "auc": auc,
            "positive_rate": float(y_true.mean()),
        },
    )


def train_logistic_baseline(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> ModelEvalResult:
    if "split" not in frame.columns:
        return ModelEvalResult("logistic_regression_baseline", "BLOCKED_NO_SPLIT")
    train_splits = {"train", "validation"}
    train_df = frame[frame["split"].isin(train_splits)]
    test_df = frame[frame["split"] == "test"]
    if train_df.empty or test_df.empty:
        return ModelEvalResult("logistic_regression_baseline", "BLOCKED_INSUFFICIENT_SPLIT_DATA")

    X_train, y_train = _prepare_xy(train_df, feature_columns, target_column)
    X_test, y_test = _prepare_xy(test_df, feature_columns, target_column)
    if X_train.empty:
        return ModelEvalResult("logistic_regression_baseline", "BLOCKED_NO_FEATURES")

    preprocessor = _build_preprocessor(X_train)
    model = Pipeline(
        [
            ("prep", preprocessor),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    y_pred = (proba >= 0.5).astype(int)
    metrics = {
        "auc": _safe_auc(y_test.to_numpy(), proba),
        "pr_auc": float(average_precision_score(y_test, proba)) if len(np.unique(y_test)) > 1 else None,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "selected_count": int(y_pred.sum()),
        "precision_at_0_5": float(y_test[y_pred == 1].mean()) if y_pred.sum() else 0.0,
    }
    return ModelEvalResult("logistic_regression_baseline", "PASS", metrics=metrics)


def train_calibrated_logistic(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
) -> ModelEvalResult:
    if "split" not in frame.columns:
        return ModelEvalResult("calibrated_logistic", "BLOCKED_NO_SPLIT")
    cal_df = frame[frame["split"] == "validation"]
    test_df = frame[frame["split"] == "test"]
    train_df = frame[frame["split"] == "train"]
    if train_df.empty:
        train_df = frame[frame["split"].isin({"train", "validation"})]
    if cal_df.empty or test_df.empty or len(cal_df) < 50:
        return ModelEvalResult(
            "calibrated_logistic",
            "BLOCKED_INSUFFICIENT_CALIBRATION_DATA",
            reason="validation_fold_too_small_for_calibration",
        )

    X_train, y_train = _prepare_xy(train_df, feature_columns, target_column)
    X_cal, y_cal = _prepare_xy(cal_df, feature_columns, target_column)
    X_test, y_test = _prepare_xy(test_df, feature_columns, target_column)

    base = Pipeline(
        [
            ("prep", _build_preprocessor(X_train)),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )
    base.fit(X_train, y_train)
    try:
        calibrator = CalibratedClassifierCV(base, method="sigmoid", cv="prefit")
        calibrator.fit(X_cal, y_cal)
    except (TypeError, ValueError):
        calibrator = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid")
        calibrator.fit(X_cal, y_cal)
    proba = calibrator.predict_proba(X_test)[:, 1]
    metrics = {
        "auc": _safe_auc(y_test.to_numpy(), proba),
        "brier_score": float(brier_score_loss(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)) if len(np.unique(y_test)) > 1 else None,
    }
    return ModelEvalResult("calibrated_logistic", "PASS", metrics=metrics)


def train_xgb_meta_model(
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    *,
    include_xgb_meta: bool = True,
) -> ModelEvalResult:
    if not include_xgb_meta:
        return ModelEvalResult("xgb_meta_model", "BLOCKED_DISABLED_BY_FLAG")
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return ModelEvalResult(
            "xgb_meta_model",
            "XGB_META_BLOCKED_DEPENDENCY_MISSING",
            reason="xgboost_not_installed",
        )

    if "split" not in frame.columns:
        return ModelEvalResult("xgb_meta_model", "BLOCKED_NO_SPLIT")
    train_df = frame[frame["split"].isin({"train", "validation"})]
    test_df = frame[frame["split"] == "test"]
    if train_df.empty or test_df.empty:
        return ModelEvalResult("xgb_meta_model", "BLOCKED_INSUFFICIENT_SPLIT_DATA")

    X_train, y_train = _prepare_xy(train_df, feature_columns, target_column)
    X_test, y_test = _prepare_xy(test_df, feature_columns, target_column)
    numeric, _ = _numeric_and_categorical_columns(X_train)
    if not numeric:
        return ModelEvalResult("xgb_meta_model", "BLOCKED_NO_NUMERIC_FEATURES")

    X_train_num = X_train[numeric].apply(pd.to_numeric, errors="coerce")
    X_test_num = X_test[numeric].apply(pd.to_numeric, errors="coerce")
    X_train_num = X_train_num.fillna(X_train_num.median())
    X_test_num = X_test_num.fillna(X_train_num.median())

    model = XGBClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X_train_num, y_train)
    proba = model.predict_proba(X_test_num)[:, 1]
    metrics = {
        "auc": _safe_auc(y_test.to_numpy(), proba),
        "pr_auc": float(average_precision_score(y_test, proba)) if len(np.unique(y_test)) > 1 else None,
    }
    return ModelEvalResult("xgb_meta_model", "PASS", metrics=metrics)


def run_signal_family_ablations(
    frame: pd.DataFrame,
    target_column: str,
    families: dict[str, list[str]],
) -> dict[str, Any]:
    findings: dict[str, Any] = {}
    base_cols = families.get("model_score_family", [])
    layers = [
        ("model_scores_only", base_cols),
        ("model_scores_plus_consensus", base_cols + families.get("consensus_family", [])),
        (
            "plus_policy_context",
            base_cols + families.get("consensus_family", []) + families.get("policy_family", []),
        ),
        (
            "plus_liquidity_activity",
            base_cols
            + families.get("consensus_family", [])
            + families.get("policy_family", [])
            + families.get("liquidity_activity_family", []),
        ),
        (
            "plus_concentration_robustness",
            base_cols
            + families.get("consensus_family", [])
            + families.get("policy_family", [])
            + families.get("liquidity_activity_family", [])
            + families.get("concentration_robustness_family", []),
        ),
        (
            "plus_whale_family",
            base_cols
            + families.get("consensus_family", [])
            + families.get("policy_family", [])
            + families.get("liquidity_activity_family", [])
            + families.get("concentration_robustness_family", [])
            + families.get("whale_family", []),
        ),
    ]
    prev_auc = None
    for name, cols in layers:
        cols = [c for c in cols if c in frame.columns]
        if not cols or "split" not in frame.columns:
            findings[name] = {"status": "SKIPPED_NO_FEATURES"}
            continue
        result = train_logistic_baseline(frame, cols, target_column)
        auc = (result.metrics or {}).get("auc")
        delta = None if prev_auc is None or auc is None else auc - prev_auc
        findings[name] = {
            "status": result.status,
            "feature_count": len(cols),
            "auc": auc,
            "delta_auc_vs_previous_layer": delta,
        }
        if auc is not None:
            prev_auc = auc
    return findings


def run_robustness_audits(
    frame: pd.DataFrame,
    target_column: str,
    *,
    rule_result: ModelEvalResult,
) -> dict[str, Any]:
    if "pair_address" not in frame.columns:
        return {"status": "BLOCKED_NO_PAIR_ADDRESS_FOR_GROUPING"}

    eval_df = frame[frame["split"] == "validation"] if "split" in frame.columns else frame
    if eval_df.empty:
        eval_df = frame

    counts = eval_df["pair_address"].value_counts(normalize=True)
    top_pair = counts.index[0] if len(counts) else None
    top_share = float(counts.iloc[0]) if len(counts) else 0.0

    baseline_precision = (rule_result.metrics or {}).get("precision_at_rule", 0.0)
    removed_precision = None
    if top_pair is not None:
        subset = eval_df[eval_df["pair_address"] != top_pair]
        if "vote_count" in subset.columns and len(subset):
            y_true = subset[target_column].astype(int)
            y_score = (subset["vote_count"].fillna(0) >= 2).astype(int)
            if y_score.sum():
                removed_precision = float(y_true[y_score == 1].mean())

    lopo_scores = []
    pairs = counts.head(5).index.tolist()
    for pair in pairs:
        subset = eval_df[eval_df["pair_address"] != pair]
        if "vote_count" not in subset.columns or subset.empty:
            continue
        y_true = subset[target_column].astype(int)
        y_score = (subset["vote_count"].fillna(0) >= 2).astype(int)
        if y_score.sum():
            lopo_scores.append(float(y_true[y_score == 1].mean()))

    outlier_dependency = top_share > 0.35 and (
        removed_precision is not None and removed_precision < baseline_precision * 0.7
    )
    return {
        "top_pair_share": top_share,
        "unique_pairs": int(counts.shape[0]),
        "baseline_rule_precision": baseline_precision,
        "top_pair_removed_precision": removed_precision,
        "leave_one_pair_out_precision_sample": lopo_scores,
        "outlier_dependency_flag": bool(outlier_dependency),
        "robustness_pass_flag": not outlier_dependency and top_share <= 0.5,
    }
