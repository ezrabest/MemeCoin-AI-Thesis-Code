"""Phase E8B — clean historical Random Forest training with temporal split and leakage guards."""

from __future__ import annotations

import csv
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline

from app.artifacts.registry import detect_project_root
from app.training.direct_target_xgb_rf import (
    DatasetDescriptor,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    discover_direct_target_datasets,
    utc_now_iso,
)
from app.training.tabicl_v2_eval import precision_at_top_k_with_count

PHASE = "E8B"
SCRIPT_PATH = "scripts/train_clean_historical_rf.py"
TARGET_COL = "target_net_profitable_after_exit"
LABEL_VALID_COL = "label_valid"
EVAL_RETURN_COL = "sim_net_return"
SPLIT_ONLY_COLS = ("event_timestamp",)
DIAGNOSTICS_ONLY_COLS = ("pair_address",)
TOP_PCTS: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0)
TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
SMOKE_DEFAULT_MAX_ROWS = 2000
DEFAULT_OUTPUT_ROOT = "data/training/manual_verified_results"
DEFAULT_DATASET_ROOT = "data/training/manual_verified_datasets_direct_target_v1"

SAFE_CORE_FEATURES: tuple[str, ...] = (
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
    "whale_score",
    "buy_ratio",
    "price_usd",
    "liquidity_usd",
    "volume_to_liquidity_ratio",
)

FORBIDDEN_EXACT_FEATURES: frozenset[str] = frozenset(
    {
        "target_net_profitable_after_exit",
        "target_net_profitable",
        "target",
        "sim_net_return",
        "sim_exit_status",
        "exit_ratio",
        "max_future_ratio",
        "min_future_ratio",
        "label_valid",
        "label_error_code",
        "label_error_detail",
        "future_window_start_timestamp",
        "future_window_end_timestamp",
        "first_future_snapshot_timestamp",
        "last_future_snapshot_timestamp",
        "future_snapshot_count",
        "max_future_gap_minutes",
        "gap_detected",
        "gap_start_timestamp",
        "gap_end_timestamp",
        "gap_minutes",
        "exit_timestamp",
        "target_name",
        "target_version",
        "label_source_artifact_id",
        "candidate_id",
        "candidate_policy_id",
        "target_row_id",
        "entry_snapshot_id",
        "source_row_id",
        "pair_address",
        "event_timestamp",
        "snapshot_timestamp",
        "chain",
        "symbol",
        "address",
        "url",
        "entry_price",
        "entry_price_raw",
        "entry_price_verified_30m",
        "entry_price_verified_1h",
        "entry_price_verified_4h",
        "entry_price_verified_8h",
        "entry_price_verified_24h",
        "price_step_ratio_prev",
        "is_extreme_step_ratio_100x",
    }
)

FORBIDDEN_TOKEN_PATTERNS: tuple[str, ...] = (
    "future",
    "target_",
    "label_",
    "sim_",
    "exit_",
    "gap_",
)

_FORBIDDEN_AUDIT_PATH: Path | None = None


class E8BAuditLogger:
    """Append-only JSONL audit logger for E8B runs."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, **fields: Any) -> None:
        payload = {
            "created_at_utc": utc_now_iso(),
            "event_type": event_type,
            "phase": PHASE,
            **fields,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class CleanRFForbiddenFeatureError(RuntimeError):
    """Raised when feature schema validation fails closed."""


def set_forbidden_audit_path(path: Path | None) -> None:
    global _FORBIDDEN_AUDIT_PATH
    _FORBIDDEN_AUDIT_PATH = path


def validate_feature_schema(feature_cols: list[str]) -> dict[str, Any]:
    """Fail-closed validator for X feature columns."""
    exact_hits: list[str] = []
    pattern_hits: list[dict[str, str]] = []
    accepted: list[str] = []

    for col in feature_cols:
        col_lower = col.lower()
        if col_lower in {f.lower() for f in FORBIDDEN_EXACT_FEATURES} or col in FORBIDDEN_EXACT_FEATURES:
            exact_hits.append(col)
            continue
        matched_pattern = None
        for pattern in FORBIDDEN_TOKEN_PATTERNS:
            if pattern in col_lower:
                matched_pattern = pattern
                break
        if matched_pattern:
            pattern_hits.append({"feature": col, "pattern": matched_pattern})
            continue
        accepted.append(col)

    result: dict[str, Any] = {
        "attempted_feature_list": list(feature_cols),
        "accepted_feature_list": accepted,
        "exact_blacklist_hits": exact_hits,
        "negative_pattern_hits": pattern_hits,
        "valid": not exact_hits and not pattern_hits,
    }

    if not result["valid"]:
        audit_rows = []
        for col in exact_hits:
            audit_rows.append(
                {
                    "feature": col,
                    "violation_type": "exact_blacklist",
                    "pattern": "",
                    "created_at_utc": utc_now_iso(),
                }
            )
        for hit in pattern_hits:
            audit_rows.append(
                {
                    "feature": hit["feature"],
                    "violation_type": "forbidden_token_pattern",
                    "pattern": hit["pattern"],
                    "created_at_utc": utc_now_iso(),
                }
            )
        if _FORBIDDEN_AUDIT_PATH is not None:
            df = pd.DataFrame(audit_rows)
            atomic_write_csv(df, _FORBIDDEN_AUDIT_PATH)
        msg = (
            f"FORBIDDEN_FEATURE_SCHEMA: exact={exact_hits}, "
            f"patterns={[h['feature'] for h in pattern_hits]}"
        )
        raise CleanRFForbiddenFeatureError(msg)
    return result


def resolve_safe_features(available_columns: list[str]) -> list[str]:
    """Return safe core features present in the dataset."""
    available = set(available_columns)
    return [col for col in SAFE_CORE_FEATURES if col in available]


@dataclass
class TrainConfig:
    dataset_root: Path
    output_dir: Path
    filters: tuple[str, ...] = ("RAW_ALL_VERIFIED",)
    horizons: tuple[str, ...] = ("30m", "1h", "4h", "8h", "24h")
    exit_policies: tuple[str, ...] = (
        "TP20308_SL075_FEE0308_TIME_BY_HORIZON",
        "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
    )
    smoke: bool = False
    full: bool = False
    max_rows: int | None = None
    n_estimators: int = 100
    max_depth: int | None = None
    min_samples_leaf: int = 5
    class_weight: str = "balanced_subsample"
    random_state: int = 42
    n_jobs: int = 1
    selected_descriptors: list[DatasetDescriptor] | None = None


@dataclass
class RunState:
    datasets_completed: int = 0
    datasets_failed: int = 0
    models_trained: int = 0
    dataset_summaries: list[dict[str, Any]] = field(default_factory=list)
    split_summaries: list[dict[str, Any]] = field(default_factory=list)
    metrics_rows: list[dict[str, Any]] = field(default_factory=list)
    validation_policy_rows: list[dict[str, Any]] = field(default_factory=list)
    test_applied_rows: list[dict[str, Any]] = field(default_factory=list)
    pair_overlap_rows: list[dict[str, Any]] = field(default_factory=list)
    pair_concentration_rows: list[dict[str, Any]] = field(default_factory=list)
    seen_unseen_rows: list[dict[str, Any]] = field(default_factory=list)
    robustness_rows: list[dict[str, Any]] = field(default_factory=list)
    leakage_audits: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)


def prepare_output_dirs(output_dir: Path) -> None:
    for sub in ("reports", "models", "predictions", "audit"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def make_output_dir(output_root: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = output_root / f"phase_e8b_clean_historical_rf_{ts}"
    prepare_output_dirs(out)
    return out


def filter_descriptors(
    descriptors: list[DatasetDescriptor],
    *,
    filters: tuple[str, ...],
    horizons: tuple[str, ...],
    exit_policies: tuple[str, ...],
    smoke: bool,
) -> list[DatasetDescriptor]:
    out: list[DatasetDescriptor] = []
    for desc in descriptors:
        if desc.filter_name not in filters:
            continue
        if desc.horizon not in horizons:
            continue
        if desc.exit_policy_id not in exit_policies:
            continue
        out.append(desc)
    if smoke and out:
        preferred = [
            d
            for d in out
            if d.filter_name == "RAW_ALL_VERIFIED"
            and d.horizon == "1h"
            and d.exit_policy_id == "TP20308_SL080_FEE0308_TIME_BY_HORIZON"
        ]
        return [preferred[0]] if preferred else [out[0]]
    return out


def load_dataset_columns(path: Path) -> pd.DataFrame:
    load_cols = list(
        dict.fromkeys(
            [
                *SAFE_CORE_FEATURES,
                TARGET_COL,
                LABEL_VALID_COL,
                EVAL_RETURN_COL,
                *SPLIT_ONLY_COLS,
                *DIAGNOSTICS_ONLY_COLS,
            ]
        )
    )
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path, columns=load_cols)
        except Exception:
            return pd.read_parquet(path)
    try:
        return pd.read_csv(path, usecols=lambda c: c in load_cols)
    except Exception:
        return pd.read_csv(path)


def _parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False)
    as_str = series.astype(str).str.strip().str.lower()
    return as_str.isin({"1", "true", "yes", "y", "t"})


def _coerce_binary_target(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype(int)
    as_str = series.astype(str).str.strip().str.lower()
    mapped = series.copy()
    for raw, val in {"true": 1, "false": 0, "1": 1, "0": 0}.items():
        mapped = mapped.where(as_str != raw, val)
    return pd.to_numeric(mapped, errors="coerce").fillna(0).astype(int)


def prepare_dataset(
    df: pd.DataFrame,
    *,
    max_rows: int | None,
) -> pd.DataFrame:
    if LABEL_VALID_COL not in df.columns:
        raise RuntimeError(f"MISSING_{LABEL_VALID_COL.upper()}")
    if TARGET_COL not in df.columns:
        raise RuntimeError(f"MISSING_{TARGET_COL.upper()}")
    if "event_timestamp" not in df.columns:
        raise RuntimeError("MISSING_EVENT_TIMESTAMP")

    valid_mask = _parse_bool_series(df[LABEL_VALID_COL])
    out = df.loc[valid_mask].copy()
    out[TARGET_COL] = _coerce_binary_target(out[TARGET_COL])
    out["event_timestamp"] = pd.to_datetime(out["event_timestamp"], utc=True, errors="coerce")
    out = out.dropna(subset=["event_timestamp"])
    out = out.sort_values("event_timestamp", ascending=True, kind="mergesort").reset_index(drop=True)

    if max_rows is not None and len(out) > max_rows:
        out = out.iloc[:max_rows].copy().reset_index(drop=True)
    return out


def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    n = len(df)
    if n < 3:
        raise RuntimeError("INSUFFICIENT_ROWS_FOR_TEMPORAL_SPLIT")
    train_end = max(1, int(n * TRAIN_FRAC))
    val_end = max(train_end + 1, int(n * (TRAIN_FRAC + VAL_FRAC)))
    if val_end >= n:
        val_end = n - 1
    train = df.iloc[:train_end].copy()
    val = df.iloc[train_end:val_end].copy()
    test = df.iloc[val_end:].copy()
    meta = {
        "total_rows": n,
        "train_rows": len(train),
        "validation_rows": len(val),
        "test_rows": len(test),
        "train_min_timestamp": train["event_timestamp"].min().isoformat() if len(train) else None,
        "train_max_timestamp": train["event_timestamp"].max().isoformat() if len(train) else None,
        "validation_min_timestamp": val["event_timestamp"].min().isoformat() if len(val) else None,
        "validation_max_timestamp": val["event_timestamp"].max().isoformat() if len(val) else None,
        "test_min_timestamp": test["event_timestamp"].min().isoformat() if len(test) else None,
        "test_max_timestamp": test["event_timestamp"].max().isoformat() if len(test) else None,
    }
    return train, val, test, meta


def build_rf_pipeline(config: TrainConfig) -> Pipeline:
    cw: str | None = config.class_weight
    if cw == "none":
        cw = None
    estimator = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_leaf=config.min_samples_leaf,
        class_weight=cw,
        n_jobs=config.n_jobs,
        random_state=config.random_state,
    )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", estimator),
        ]
    )


def extract_positive_probability(pipeline: Pipeline, x_matrix: pd.DataFrame) -> np.ndarray:
    proba = pipeline.predict_proba(x_matrix)
    classes = list(pipeline.named_steps["model"].classes_)
    if 1 not in classes:
        raise RuntimeError("POSITIVE_CLASS_MISSING")
    return proba[:, classes.index(1)]


def recall_at_top_pct(y_true: np.ndarray, y_score: np.ndarray, top_pct: float) -> float | None:
    positives = float(y_true.sum())
    if positives == 0 or len(y_true) == 0:
        return None
    k = max(1, int(len(y_true) * top_pct / 100.0))
    order = np.argsort(-y_score)
    return float(y_true[order[:k]].sum() / positives)


def select_top_rows(frame: pd.DataFrame, y_score: np.ndarray, top_pct: float) -> pd.DataFrame:
    n = len(frame)
    if n == 0:
        return frame.iloc[0:0].copy()
    k = max(1, int(n * top_pct / 100.0))
    order = np.argsort(-y_score)
    return frame.iloc[order[:k]].copy()


def compute_top_pct_metrics(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    top_pct: float,
    *,
    return_col: str | None,
) -> dict[str, Any]:
    selected = select_top_rows(frame, y_score, top_pct)
    sel_y = selected[TARGET_COL].to_numpy() if len(selected) else np.array([], dtype=int)
    prec_info = precision_at_top_k_with_count(y_true, y_score, top_pct)
    selected_count = int(len(selected))
    selected_positive_count = int((sel_y == 1).sum()) if selected_count else 0
    false_positive_count = int((sel_y == 0).sum()) if selected_count else 0
    metrics: dict[str, Any] = {
        "top_pct": top_pct,
        "precision_at_top_pct": prec_info["precision"],
        "recall_at_top_pct": recall_at_top_pct(y_true, y_score, top_pct),
        "selected_count": selected_count,
        "selected_positive_count": selected_positive_count,
        "selected_positive_rate": float(sel_y.mean()) if selected_count else None,
        "false_positive_count": false_positive_count,
        "selected_win_rate_by_y": float(sel_y.mean()) if selected_count else None,
        "selected_average_sim_net_return": None,
        "selected_total_sim_net_return": None,
        "selected_unique_pairs": int(selected["pair_address"].nunique()) if "pair_address" in selected.columns and selected_count else 0,
        "selected_top_pair_share": None,
    }
    if return_col and return_col in selected.columns and selected_count:
        returns = pd.to_numeric(selected[return_col], errors="coerce").dropna()
        if not returns.empty:
            metrics["selected_average_sim_net_return"] = float(returns.mean())
            metrics["selected_total_sim_net_return"] = float(returns.sum())
    if "pair_address" in selected.columns and selected_count:
        pair_counts = selected["pair_address"].value_counts()
        metrics["selected_top_pair_share"] = float(pair_counts.iloc[0] / selected_count)
    return metrics


def compute_split_metrics(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    split_name: str,
    return_col: str | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base: dict[str, Any] = {
        "split": split_name,
        "rows": int(len(y_true)),
        "positives": int((y_true == 1).sum()),
        "positive_rate": float(y_true.mean()) if len(y_true) else None,
        "roc_auc": None,
        "pr_auc": None,
    }
    if len(np.unique(y_true)) > 1:
        try:
            base["roc_auc"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            base["roc_auc"] = None
        try:
            base["pr_auc"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            base["pr_auc"] = None

    for top_pct in TOP_PCTS:
        top_metrics = compute_top_pct_metrics(
            frame,
            y_true,
            y_score,
            top_pct,
            return_col=return_col,
        )
        rows.append({**base, **top_metrics})
    return rows


def select_validation_policy(val_metrics: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [m for m in val_metrics if m.get("split") == "validation"]
    if not candidates:
        return None

    def score_row(row: dict[str, Any]) -> tuple[float, float, float, float]:
        prec = row.get("precision_at_top_pct") or 0.0
        avg_ret = row.get("selected_average_sim_net_return") or -1e9
        tot_ret = row.get("selected_total_sim_net_return") or -1e9
        sel_count = float(row.get("selected_count") or 0)
        return (prec, avg_ret, tot_ret, sel_count)

    ranked = sorted(candidates, key=score_row, reverse=True)
    min_selected = 3
    min_pairs = 2
    for row in ranked:
        if (row.get("selected_count") or 0) < min_selected:
            continue
        if (row.get("selected_unique_pairs") or 0) < min_pairs:
            continue
        return row
    return ranked[0] if ranked else None


def pair_overlap_diagnostics(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    *,
    descriptor: DatasetDescriptor,
) -> dict[str, Any]:
    train_pairs = set(train["pair_address"].dropna().astype(str)) if "pair_address" in train.columns else set()
    val_pairs = set(val["pair_address"].dropna().astype(str)) if "pair_address" in val.columns else set()
    test_pairs = set(test["pair_address"].dropna().astype(str)) if "pair_address" in test.columns else set()
    tv_overlap = train_pairs & val_pairs
    tt_overlap = train_pairs & test_pairs
    vt_overlap = val_pairs & test_pairs
    return {
        "dataset_name": descriptor.dataset_name,
        "filter": descriptor.filter_name,
        "horizon": descriptor.horizon,
        "exit_policy_id": descriptor.exit_policy_id,
        "unique_pairs_train": len(train_pairs),
        "unique_pairs_validation": len(val_pairs),
        "unique_pairs_test": len(test_pairs),
        "train_val_pair_overlap_count": len(tv_overlap),
        "train_test_pair_overlap_count": len(tt_overlap),
        "val_test_pair_overlap_count": len(vt_overlap),
        "train_val_pair_overlap_ratio": len(tv_overlap) / len(train_pairs) if train_pairs else None,
        "train_test_pair_overlap_ratio": len(tt_overlap) / len(train_pairs) if train_pairs else None,
        "val_test_pair_overlap_ratio": len(vt_overlap) / len(val_pairs) if val_pairs else None,
    }


def pair_concentration_diagnostics(
    selected: pd.DataFrame,
    positives: pd.DataFrame,
    *,
    descriptor: DatasetDescriptor,
    split_name: str,
    top_pct: float,
) -> dict[str, Any]:
    selected_count = len(selected)
    unique_pairs = int(selected["pair_address"].nunique()) if "pair_address" in selected.columns and selected_count else 0
    top_pair_share = None
    positive_top_pair_share = None
    selected_positive_top_pair_share = None
    if selected_count and "pair_address" in selected.columns:
        pair_counts = selected["pair_address"].value_counts()
        top_pair_share = float(pair_counts.iloc[0] / selected_count)
        if TARGET_COL in selected.columns:
            pos_selected = selected[selected[TARGET_COL] == 1]
            if len(pos_selected):
                pos_counts = pos_selected["pair_address"].value_counts()
                selected_positive_top_pair_share = float(pos_counts.iloc[0] / len(pos_selected))
    if len(positives) and "pair_address" in positives.columns:
        pos_pair_counts = positives["pair_address"].value_counts()
        positive_top_pair_share = float(pos_pair_counts.iloc[0] / len(positives))
    return {
        "dataset_name": descriptor.dataset_name,
        "filter": descriptor.filter_name,
        "horizon": descriptor.horizon,
        "exit_policy_id": descriptor.exit_policy_id,
        "split": split_name,
        "top_pct": top_pct,
        "selected_unique_pairs": unique_pairs,
        "selected_top_pair_share": top_pair_share,
        "positive_top_pair_share": positive_top_pair_share,
        "selected_positive_top_pair_share": selected_positive_top_pair_share,
    }


def seen_unseen_pair_diagnostics(
    test: pd.DataFrame,
    train: pd.DataFrame,
    y_score: np.ndarray,
    top_pct: float,
    *,
    descriptor: DatasetDescriptor,
    return_col: str | None,
) -> list[dict[str, Any]]:
    train_pairs = set(train["pair_address"].dropna().astype(str)) if "pair_address" in train.columns else set()
    rows: list[dict[str, Any]] = []
    for label, mask_fn in (
        ("seen_pair", lambda df: df["pair_address"].astype(str).isin(train_pairs)),
        ("unseen_pair", lambda df: ~df["pair_address"].astype(str).isin(train_pairs)),
    ):
        if "pair_address" not in test.columns:
            continue
        subset = test.loc[mask_fn(test)].copy()
        if subset.empty:
            rows.append(
                {
                    "dataset_name": descriptor.dataset_name,
                    "filter": descriptor.filter_name,
                    "horizon": descriptor.horizon,
                    "exit_policy_id": descriptor.exit_policy_id,
                    "group": label,
                    "rows": 0,
                    "top_pct": top_pct,
                    "precision_at_top_pct": None,
                    "selected_count": 0,
                    "selected_positive_rate": None,
                    "selected_average_sim_net_return": None,
                }
            )
            continue
        idx = subset.index
        pos_map = {i: j for j, i in enumerate(test.index)}
        sub_scores = np.array([y_score[pos_map[i]] for i in idx])
        sub_y = subset[TARGET_COL].to_numpy()
        top_m = compute_top_pct_metrics(subset, sub_y, sub_scores, top_pct, return_col=return_col)
        rows.append(
            {
                "dataset_name": descriptor.dataset_name,
                "filter": descriptor.filter_name,
                "horizon": descriptor.horizon,
                "exit_policy_id": descriptor.exit_policy_id,
                "group": label,
                "rows": len(subset),
                "top_pct": top_pct,
                **{k: top_m.get(k) for k in ("precision_at_top_pct", "selected_count", "selected_positive_rate", "selected_average_sim_net_return")},
            }
        )
    return rows


def classify_robustness_tier(
    baseline: dict[str, Any],
    stressed: dict[str, Any],
) -> str:
    base_prec = baseline.get("precision_at_top_pct") or 0.0
    stress_prec = stressed.get("precision_at_top_pct") or 0.0
    base_total = baseline.get("selected_total_sim_net_return") or 0.0
    stress_total = stressed.get("selected_total_sim_net_return") or 0.0
    unique_pairs = baseline.get("selected_unique_pairs") or 0
    top_share = baseline.get("selected_top_pair_share") or 1.0

    prec_retained = stress_prec >= max(0.05, base_prec * 0.5) if base_prec else stress_prec > 0
    return_retained = stress_total >= max(0.0, base_total * 0.25) if base_total > 0 else stress_total >= 0

    if unique_pairs >= 5 and top_share <= 0.35 and prec_retained and return_retained:
        return "Robust Strategy Candidate"
    if (stress_prec > 0 or stress_total > 0) and (base_prec > 0 or base_total > 0):
        return "Rare Winner Detector"
    return "Lottery Artifact"


def robustness_diagnostics(
    frame: pd.DataFrame,
    y_score: np.ndarray,
    top_pct: float,
    *,
    descriptor: DatasetDescriptor,
    split_name: str,
    return_col: str | None,
) -> list[dict[str, Any]]:
    y_true = frame[TARGET_COL].to_numpy()
    baseline = compute_top_pct_metrics(frame, y_true, y_score, top_pct, return_col=return_col)
    selected = select_top_rows(frame, y_score, top_pct)
    rows: list[dict[str, Any]] = []

    def _append(diagnostic: str, stressed_frame: pd.DataFrame) -> None:
        if stressed_frame.empty:
            stressed = {**baseline, "selected_count": 0, "precision_at_top_pct": None, "selected_total_sim_net_return": None}
        else:
            idx_map = {i: j for j, i in enumerate(frame.index)}
            sub_scores = np.array([y_score[idx_map[i]] for i in stressed_frame.index])
            sub_y = stressed_frame[TARGET_COL].to_numpy()
            stressed = compute_top_pct_metrics(stressed_frame, sub_y, sub_scores, 100.0, return_col=return_col)
            stressed["top_pct"] = top_pct
        tier = classify_robustness_tier(baseline, stressed)
        rows.append(
            {
                "dataset_name": descriptor.dataset_name,
                "filter": descriptor.filter_name,
                "horizon": descriptor.horizon,
                "exit_policy_id": descriptor.exit_policy_id,
                "split": split_name,
                "top_pct": top_pct,
                "diagnostic": diagnostic,
                "baseline_precision_at_top_pct": baseline.get("precision_at_top_pct"),
                "stressed_precision_at_top_pct": stressed.get("precision_at_top_pct"),
                "baseline_selected_total_sim_net_return": baseline.get("selected_total_sim_net_return"),
                "stressed_selected_total_sim_net_return": stressed.get("selected_total_sim_net_return"),
                "robustness_tier": tier,
            }
        )

    if selected.empty:
        for diag in (
            "remove_best_selected_trade",
            "remove_best_selected_pair",
            "remove_top_selected_pair_entirely",
            "leave_one_top_pair_out",
        ):
            _append(diag, selected)
        return rows

    if return_col and return_col in selected.columns:
        best_idx = selected[return_col].astype(float).idxmax()
        _append("remove_best_selected_trade", selected.drop(index=best_idx))

    if "pair_address" in selected.columns:
        pair_totals = selected.groupby("pair_address")[return_col or TARGET_COL].sum()
        best_pair = str(pair_totals.idxmax())
        _append("remove_best_selected_pair", selected[selected["pair_address"].astype(str) != best_pair])
        top_pair = str(selected["pair_address"].value_counts().index[0])
        _append("remove_top_selected_pair_entirely", selected[selected["pair_address"].astype(str) != top_pair])

        top_pair_val = selected["pair_address"].value_counts().index[0]
        loo = selected[selected["pair_address"] != top_pair_val]
        _append("leave_one_top_pair_out", loo)
    else:
        for diag in ("remove_best_selected_pair", "remove_top_selected_pair_entirely", "leave_one_top_pair_out"):
            _append(diag, selected)

    return rows


def build_predictions_frame(
    frame: pd.DataFrame,
    y_score: np.ndarray,
    split_name: str,
    *,
    descriptor: DatasetDescriptor,
) -> pd.DataFrame:
    out = pd.DataFrame(
        {
            "pair_address": frame["pair_address"] if "pair_address" in frame.columns else None,
            "event_timestamp": frame["event_timestamp"],
            TARGET_COL: frame[TARGET_COL],
            EVAL_RETURN_COL: frame[EVAL_RETURN_COL] if EVAL_RETURN_COL in frame.columns else np.nan,
            "predicted_probability": y_score,
            "split": split_name,
            "filter": descriptor.filter_name,
            "horizon": descriptor.horizon,
            "exit_policy_id": descriptor.exit_policy_id,
        }
    )
    return out


def build_leakage_audit(
    *,
    loaded_columns: list[str],
    feature_cols: list[str],
    validation_result: dict[str, Any],
) -> dict[str, Any]:
    feature_set = set(feature_cols)
    return {
        "attempted_feature_list": validation_result.get("attempted_feature_list", feature_cols),
        "final_accepted_feature_list": validation_result.get("accepted_feature_list", feature_cols),
        "exact_blacklist_hits": validation_result.get("exact_blacklist_hits", []),
        "negative_pattern_hits": validation_result.get("negative_pattern_hits", []),
        "columns_loaded_but_excluded": [c for c in loaded_columns if c not in feature_set and c not in {TARGET_COL, LABEL_VALID_COL, EVAL_RETURN_COL, *SPLIT_ONLY_COLS, *DIAGNOSTICS_ONLY_COLS}],
        "y_column": TARGET_COL,
        "evaluation_only_columns": [c for c in loaded_columns if c == EVAL_RETURN_COL],
        "split_only_columns": [c for c in loaded_columns if c in SPLIT_ONLY_COLS],
        "diagnostics_only_columns": [c for c in loaded_columns if c in DIAGNOSTICS_ONLY_COLS],
    }


def process_dataset(
    descriptor: DatasetDescriptor,
    *,
    config: TrainConfig,
    output_dir: Path,
    audit: E8BAuditLogger,
    state: RunState,
) -> None:
    audit.log("dataset_started", dataset_path=str(descriptor.dataset_path), status="started")
    try:
        raw_df = load_dataset_columns(descriptor.dataset_path)
        loaded_columns = list(raw_df.columns)
        max_rows = config.max_rows
        if config.smoke and max_rows is None:
            max_rows = SMOKE_DEFAULT_MAX_ROWS

        df = prepare_dataset(raw_df, max_rows=max_rows)
        feature_cols = resolve_safe_features(loaded_columns)
        if not feature_cols:
            raise RuntimeError("NO_SAFE_FEATURES_AVAILABLE")

        validation_result = validate_feature_schema(feature_cols)
        x_all = df[feature_cols].apply(pd.to_numeric, errors="coerce")
        return_col = EVAL_RETURN_COL if EVAL_RETURN_COL in df.columns else None

        train, val, test, split_meta = temporal_split(df)
        for part in (train, val, test):
            if part.empty:
                raise RuntimeError("EMPTY_SPLIT_AFTER_TEMPORAL_SPLIT")

        np.random.seed(config.random_state)
        pipeline = build_rf_pipeline(config)
        x_train = x_all.loc[train.index]
        y_train = train[TARGET_COL].to_numpy()
        validate_feature_schema(list(x_train.columns))
        pipeline.fit(x_train, y_train)
        state.models_trained += 1

        model_name = descriptor.dataset_name
        model_path = output_dir / "models" / f"{model_name}_clean_rf.joblib"
        schema_path = output_dir / "models" / f"{model_name}_clean_rf_schema.json"
        validate_feature_schema(feature_cols)
        joblib.dump(pipeline, model_path)
        atomic_write_json(
            {
                "feature_columns": feature_cols,
                "target_column": TARGET_COL,
                "imputer_strategy": "median",
                "model_type": "RandomForestClassifier",
                "class_weight": config.class_weight,
                "random_state": config.random_state,
            },
            schema_path,
        )

        split_scores: dict[str, np.ndarray] = {}
        split_frames: dict[str, pd.DataFrame] = {}
        for split_name, part in (("train", train), ("validation", val), ("test", test)):
            x_part = x_all.loc[part.index]
            validate_feature_schema(list(x_part.columns))
            scores = extract_positive_probability(pipeline, x_part)
            split_scores[split_name] = scores
            split_frames[split_name] = part

            for metric_row in compute_split_metrics(
                part,
                part[TARGET_COL].to_numpy(),
                scores,
                split_name=split_name,
                return_col=return_col,
            ):
                state.metrics_rows.append(
                    {
                        "dataset_name": descriptor.dataset_name,
                        "filter": descriptor.filter_name,
                        "horizon": descriptor.horizon,
                        "exit_policy_id": descriptor.exit_policy_id,
                        **metric_row,
                    }
                )

        val_preds = build_predictions_frame(val, split_scores["validation"], "validation", descriptor=descriptor)
        test_preds = build_predictions_frame(test, split_scores["test"], "test", descriptor=descriptor)
        pred_base = output_dir / "predictions" / model_name
        atomic_write_parquet(val_preds, pred_base.with_name(f"{model_name}_validation_predictions.parquet"))
        atomic_write_csv(val_preds, pred_base.with_name(f"{model_name}_validation_predictions.csv"))
        atomic_write_parquet(test_preds, pred_base.with_name(f"{model_name}_test_predictions.parquet"))
        atomic_write_csv(test_preds, pred_base.with_name(f"{model_name}_test_predictions.csv"))

        val_metrics = [r for r in state.metrics_rows if r["dataset_name"] == descriptor.dataset_name and r["split"] == "validation"]
        selected_policy = select_validation_policy(val_metrics)
        if selected_policy:
            state.validation_policy_rows.append(
                {
                    "dataset_name": descriptor.dataset_name,
                    "filter": descriptor.filter_name,
                    "horizon": descriptor.horizon,
                    "exit_policy_id": descriptor.exit_policy_id,
                    **{k: selected_policy.get(k) for k in selected_policy},
                    "selection_source": "validation_only",
                }
            )
            top_pct = float(selected_policy["top_pct"])
            test_part = split_frames["test"]
            test_scores = split_scores["test"]
            test_selected_metrics = compute_top_pct_metrics(
                test_part,
                test_part[TARGET_COL].to_numpy(),
                test_scores,
                top_pct,
                return_col=return_col,
            )
            state.test_applied_rows.append(
                {
                    "dataset_name": descriptor.dataset_name,
                    "filter": descriptor.filter_name,
                    "horizon": descriptor.horizon,
                    "exit_policy_id": descriptor.exit_policy_id,
                    "validation_selected_top_pct": top_pct,
                    "selection_source": "validation_only",
                    **test_selected_metrics,
                }
            )

            val_selected = select_top_rows(val, split_scores["validation"], top_pct)
            test_selected = select_top_rows(test, test_scores, top_pct)
            state.pair_concentration_rows.append(
                pair_concentration_diagnostics(
                    val_selected,
                    val[val[TARGET_COL] == 1],
                    descriptor=descriptor,
                    split_name="validation",
                    top_pct=top_pct,
                )
            )
            state.pair_concentration_rows.append(
                pair_concentration_diagnostics(
                    test_selected,
                    test[test[TARGET_COL] == 1],
                    descriptor=descriptor,
                    split_name="test",
                    top_pct=top_pct,
                )
            )
            state.seen_unseen_rows.extend(
                seen_unseen_pair_diagnostics(
                    test,
                    train,
                    test_scores,
                    top_pct,
                    descriptor=descriptor,
                    return_col=return_col,
                )
            )
            state.robustness_rows.extend(
                robustness_diagnostics(
                    val,
                    split_scores["validation"],
                    top_pct,
                    descriptor=descriptor,
                    split_name="validation",
                    return_col=return_col,
                )
            )
            state.robustness_rows.extend(
                robustness_diagnostics(
                    test,
                    test_scores,
                    top_pct,
                    descriptor=descriptor,
                    split_name="test",
                    return_col=return_col,
                )
            )

        overlap = pair_overlap_diagnostics(train, val, test, descriptor=descriptor)
        state.pair_overlap_rows.append(overlap)
        state.split_summaries.append(
            {
                "dataset_name": descriptor.dataset_name,
                "filter": descriptor.filter_name,
                "horizon": descriptor.horizon,
                "exit_policy_id": descriptor.exit_policy_id,
                **split_meta,
            }
        )
        state.dataset_summaries.append(
            {
                "dataset_name": descriptor.dataset_name,
                "dataset_path": str(descriptor.dataset_path),
                "filter": descriptor.filter_name,
                "horizon": descriptor.horizon,
                "exit_policy_id": descriptor.exit_policy_id,
                "loaded_rows": len(raw_df),
                "valid_label_rows": len(df),
                "feature_count": len(feature_cols),
                "features": "|".join(feature_cols),
                "status": "ok",
            }
        )
        state.leakage_audits.append(
            {
                "dataset_name": descriptor.dataset_name,
                **build_leakage_audit(
                    loaded_columns=loaded_columns,
                    feature_cols=feature_cols,
                    validation_result=validation_result,
                ),
            }
        )
        atomic_write_json(
            {
                "dataset_name": descriptor.dataset_name,
                "feature_columns": feature_cols,
                "forbidden_exact_features": sorted(FORBIDDEN_EXACT_FEATURES),
                "forbidden_token_patterns": list(FORBIDDEN_TOKEN_PATTERNS),
            },
            output_dir / "reports" / f"clean_rf_feature_schema_{descriptor.dataset_name}.json",
        )
        state.datasets_completed += 1
        audit.log("dataset_completed", dataset_path=str(descriptor.dataset_path), status="ok")
    except CleanRFForbiddenFeatureError as exc:
        state.datasets_failed += 1
        err = {"dataset_name": descriptor.dataset_name, "error": str(exc), "error_type": "forbidden_feature"}
        state.errors.append(err)
        audit.log("dataset_failed", dataset_path=str(descriptor.dataset_path), status="forbidden_feature", error=str(exc))
        if config.smoke:
            raise
    except Exception as exc:
        state.datasets_failed += 1
        err = {"dataset_name": descriptor.dataset_name, "error": str(exc), "error_type": type(exc).__name__}
        state.errors.append(err)
        err_path = output_dir / "audit" / "clean_rf_errors.jsonl"
        with err_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({**err, "created_at_utc": utc_now_iso()}, default=str) + "\n")
        audit.log("dataset_failed", dataset_path=str(descriptor.dataset_path), status="error", error=str(exc))
        if config.smoke:
            raise


def finalize_outputs(config: TrainConfig, output_dir: Path, state: RunState, descriptors: list[DatasetDescriptor]) -> dict[str, Any]:
    reports = output_dir / "reports"
    atomic_write_csv(pd.DataFrame(state.dataset_summaries), reports / "clean_rf_dataset_summary.csv")
    atomic_write_csv(pd.DataFrame(state.split_summaries), reports / "clean_rf_split_summary.csv")
    atomic_write_csv(pd.DataFrame(state.metrics_rows), reports / "clean_rf_metrics_by_dataset.csv")
    atomic_write_csv(pd.DataFrame(state.validation_policy_rows), reports / "clean_rf_validation_policy_selection.csv")
    atomic_write_csv(pd.DataFrame(state.test_applied_rows), reports / "clean_rf_test_applied_selected_policies.csv")
    atomic_write_csv(pd.DataFrame(state.pair_overlap_rows), reports / "clean_rf_pair_overlap_diagnostic.csv")
    atomic_write_csv(pd.DataFrame(state.pair_concentration_rows), reports / "clean_rf_pair_concentration_diagnostic.csv")
    atomic_write_csv(pd.DataFrame(state.seen_unseen_rows), reports / "clean_rf_seen_vs_unseen_pair_diagnostic.csv")
    atomic_write_csv(pd.DataFrame(state.robustness_rows), reports / "clean_rf_robustness_diagnostic.csv")

    leakage_flat: list[dict[str, Any]] = []
    for audit_row in state.leakage_audits:
        leakage_flat.append(
            {
                "dataset_name": audit_row.get("dataset_name"),
                "attempted_feature_list": json.dumps(audit_row.get("attempted_feature_list", [])),
                "final_accepted_feature_list": json.dumps(audit_row.get("final_accepted_feature_list", [])),
                "exact_blacklist_hits": json.dumps(audit_row.get("exact_blacklist_hits", [])),
                "negative_pattern_hits": json.dumps(audit_row.get("negative_pattern_hits", [])),
                "columns_loaded_but_excluded": json.dumps(audit_row.get("columns_loaded_but_excluded", [])),
                "y_column": audit_row.get("y_column"),
                "evaluation_only_columns": json.dumps(audit_row.get("evaluation_only_columns", [])),
                "split_only_columns": json.dumps(audit_row.get("split_only_columns", [])),
                "diagnostics_only_columns": json.dumps(audit_row.get("diagnostics_only_columns", [])),
            }
        )
    atomic_write_csv(pd.DataFrame(leakage_flat), reports / "clean_rf_leakage_audit.csv")

    if not (_FORBIDDEN_AUDIT_PATH and _FORBIDDEN_AUDIT_PATH.exists()):
        atomic_write_csv(
            pd.DataFrame(columns=["feature", "violation_type", "pattern", "created_at_utc"]),
            reports / "clean_rf_forbidden_feature_audit.csv",
        )

    feature_schema = {
        "safe_core_features": list(SAFE_CORE_FEATURES),
        "forbidden_exact_features": sorted(FORBIDDEN_EXACT_FEATURES),
        "forbidden_token_patterns": list(FORBIDDEN_TOKEN_PATTERNS),
        "datasets": {row["dataset_name"]: row.get("features", "").split("|") for row in state.dataset_summaries},
    }
    atomic_write_json(feature_schema, reports / "clean_rf_feature_schema.json")

    manifest: dict[str, Any] = {
        "phase": PHASE,
        "created_at": utc_now_iso(),
        "script_path": SCRIPT_PATH,
        "input_dataset_root": str(config.dataset_root),
        "selected_filters": list(config.filters),
        "selected_horizons": list(config.horizons),
        "selected_exit_policies": list(config.exit_policies),
        "model_hyperparameters": {
            "n_estimators": config.n_estimators,
            "max_depth": config.max_depth,
            "min_samples_leaf": config.min_samples_leaf,
            "class_weight": config.class_weight,
        },
        "random_state": config.random_state,
        "n_jobs": config.n_jobs,
        "class_weight": config.class_weight,
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version,
        "safe_feature_list": list(SAFE_CORE_FEATURES),
        "forbidden_exact_feature_list": sorted(FORBIDDEN_EXACT_FEATURES),
        "forbidden_token_patterns": list(FORBIDDEN_TOKEN_PATTERNS),
        "output_directory": str(output_dir),
        "smoke": config.smoke,
        "full": config.full,
        "dataset_count": len(descriptors),
        "trained_model_count": state.models_trained,
        "failed_dataset_count": state.datasets_failed,
        "no_runtime_changes": True,
        "no_db_writes": True,
        "old_rf_sidecars_used": False,
        "reservoir_scoring_performed": False,
        "reproducibility_note": (
            "n_jobs > 1 may affect bit-for-bit reproducibility across environments"
            if config.n_jobs > 1
            else "fixed random_state and n_jobs=1 for deterministic smoke/debug"
        ),
    }
    atomic_write_json(manifest, reports / "clean_rf_run_manifest.json")

    summary_lines = [
        "Phase E8B Clean Historical RF Summary",
        f"output_directory: {output_dir}",
        f"datasets_completed: {state.datasets_completed}",
        f"datasets_failed: {state.datasets_failed}",
        f"models_trained: {state.models_trained}",
        f"smoke: {config.smoke}",
        f"full: {config.full}",
        "",
        "Safe features:",
        ", ".join(SAFE_CORE_FEATURES),
    ]
    atomic_write_text("\n".join(summary_lines) + "\n", reports / "clean_rf_summary_for_upload.txt")
    return manifest


def run_training(config: TrainConfig) -> dict[str, Any]:
    project_root = detect_project_root(config.dataset_root)
    output_dir = config.output_dir
    prepare_output_dirs(output_dir)

    forbidden_audit = output_dir / "reports" / "clean_rf_forbidden_feature_audit.csv"
    set_forbidden_audit_path(forbidden_audit)

    audit_path = output_dir / "audit" / "clean_rf_run_audit.jsonl"
    audit = E8BAuditLogger(audit_path)
    state = RunState()

    descriptors = config.selected_descriptors
    if descriptors is None:
        descriptors = discover_direct_target_datasets(config.dataset_root)
        descriptors = filter_descriptors(
            descriptors,
            filters=config.filters,
            horizons=config.horizons,
            exit_policies=config.exit_policies,
            smoke=config.smoke,
        )

    audit.log(
        "run_started",
        status="started",
        smoke=config.smoke,
        full=config.full,
        dataset_count=len(descriptors),
        random_state=config.random_state,
        n_jobs=config.n_jobs,
    )

    for descriptor in descriptors:
        process_dataset(descriptor, config=config, output_dir=output_dir, audit=audit, state=state)

    manifest = finalize_outputs(config, output_dir, state, descriptors)
    audit.log(
        "run_completed",
        status="completed",
        datasets_completed=state.datasets_completed,
        datasets_failed=state.datasets_failed,
    )
    return {
        "output_dir": str(output_dir),
        "manifest": manifest,
        "datasets_completed": state.datasets_completed,
        "datasets_failed": state.datasets_failed,
        "models_trained": state.models_trained,
    }
