"""Offline RF + TabICLv2 policy matrix evaluation (no live trading)."""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.training.baseline_model import MODELS_DIR, TRAINING_DIR, load_baseline_metrics
from app.training.tabicl_v2_eval import (
    DEFAULT_TARGET,
    precision_at_top_k_with_count,
    resolve_return_column,
    return_metrics_for_top_k,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BACKTEST_DIR = TRAINING_DIR / "policy_backtests"
DEFAULT_REPORT_PATH = DEFAULT_BACKTEST_DIR / "rf_tab_matrix_report.json"
DEFAULT_GRID_PATH = DEFAULT_BACKTEST_DIR / "rf_tab_matrix_grid.parquet"

RF_MODEL_NAME = "random_forest"
TAB_MODEL_NAME = "tabicl_v2"
PRIMARY_RF_TARGET = "label_profitable_after_fees_4h"
PRIMARY_TAB_TARGET = DEFAULT_TARGET

RF_TARGET_ALIASES = (
    PRIMARY_RF_TARGET,
    "target_profitable_4h",
    PRIMARY_TAB_TARGET,
)
TAB_TARGET_ALIASES = (
    PRIMARY_TAB_TARGET,
    PRIMARY_RF_TARGET,
    "label_profitable_after_fees_4h",
)

TAB_METADATA_KEYS = (
    "context_strategy",
    "context_size_used",
    "batch_size_used",
    "max_train_context_rows",
    "max_features",
    "feature_count",
    "knn_rolling_days_used",
    "knn_time_decay_alpha",
    "positive_context_ratio",
    "full_evaluation",
    "output_label",
)

DEFAULT_JOIN_MISMATCH_THRESHOLD = 0.0
MIN_STABILITY_TRADE_COUNT = 5

RANK_POLICIES: list[tuple[str, float]] = [
    ("top_0_5_percent", 0.5),
    ("top_1_percent", 1.0),
    ("top_2_percent", 2.0),
    ("top_5_percent", 5.0),
    ("top_10_percent", 10.0),
]

FIXED_THRESHOLD_POLICIES: list[tuple[str, float]] = [
    ("probability_threshold_0_50", 0.50),
    ("probability_threshold_0_60", 0.60),
    ("probability_threshold_0_70", 0.70),
    ("probability_threshold_0_80", 0.80),
    ("probability_threshold_0_90", 0.90),
]

ALL_POLICY_NAMES = [name for name, _ in RANK_POLICIES] + [name for name, _ in FIXED_THRESHOLD_POLICIES]

COMBINATION_METHODS = (
    "rf_only",
    "tab_only",
    "average",
    "minimum",
    "maximum",
    "product",
)

TAB_VAL_PATTERN = re.compile(r"^tabicl_v2_predictions_validation(?:_(.+))?\.parquet$")
TAB_TEST_PATTERN = re.compile(r"^tabicl_v2_predictions_test(?:_(.+))?\.parquet$")


def base_report_flags() -> dict[str, Any]:
    return {
        "is_oracle_backtest": False,
        "offline_only": True,
        "uses_tabicl_v2": True,
        "uses_new_llm_calls": False,
        "calls_gemini": False,
        "calls_ollama": False,
        "modifies_sqlite": False,
        "changes_live_behavior": False,
    }


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_timestamps(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def discover_tab_suffixes(models_dir: Path) -> list[str | None]:
    """Return sorted unique TabICL prediction suffixes (None = unsuffixed default)."""
    suffixes: set[str | None] = set()
    if not models_dir.is_dir():
        return []

    for path in models_dir.iterdir():
        if not path.is_file():
            continue
        match = TAB_VAL_PATTERN.match(path.name)
        if match:
            suffixes.add(match.group(1))

    ordered: list[str | None] = []
    if None in suffixes:
        ordered.append(None)
    ordered.extend(sorted(s for s in suffixes if s is not None))
    return ordered


def tab_prediction_paths(models_dir: Path, suffix: str | None) -> tuple[Path, Path]:
    if suffix:
        return (
            models_dir / f"tabicl_v2_predictions_validation_{suffix}.parquet",
            models_dir / f"tabicl_v2_predictions_test_{suffix}.parquet",
        )
    return (
        models_dir / "tabicl_v2_predictions_validation.parquet",
        models_dir / "tabicl_v2_predictions_test.parquet",
    )


def tab_report_path(backtest_dir: Path, suffix: str | None) -> Path:
    if suffix:
        return backtest_dir / f"tabicl_v2_report_{suffix}.json"
    return backtest_dir / "tabicl_v2_report.json"


def load_tab_metadata(report_path: Path) -> dict[str, Any]:
    """Load optional TabICLv2 report metadata for experiment disambiguation."""
    if not report_path.is_file():
        return {"report_available": False, "report_path": str(report_path)}
    with open(report_path, encoding="utf-8") as handle:
        report = json.load(handle)
    metadata: dict[str, Any] = {
        "report_available": True,
        "report_path": str(report_path),
    }
    for key in TAB_METADATA_KEYS:
        metadata[key] = report.get(key)
    return metadata


def _filter_predictions(
    preds: pd.DataFrame,
    *,
    model_name: str,
    target_aliases: tuple[str, ...],
    split: str,
) -> pd.DataFrame:
    if preds.empty:
        return preds.copy()
    split_mask = preds["split"] == split if "split" in preds.columns else pd.Series(True, index=preds.index)
    model_mask = preds["model_name"] == model_name if "model_name" in preds.columns else pd.Series(True, index=preds.index)
    subset = preds.loc[split_mask & model_mask].copy()
    for target_name in target_aliases:
        target_mask = subset["target_name"] == target_name if "target_name" in subset.columns else pd.Series(True, index=subset.index)
        filtered = subset.loc[target_mask]
        if not filtered.empty:
            return filtered.copy()
    return pd.DataFrame()


def prepare_prediction_frame(
    preds: pd.DataFrame,
    *,
    model_name: str,
    target_aliases: tuple[str, ...],
    split: str,
    score_col: str,
) -> pd.DataFrame:
    subset = _filter_predictions(
        preds,
        model_name=model_name,
        target_aliases=target_aliases,
        split=split,
    )
    if subset.empty:
        raise ValueError(
            f"No {model_name} predictions for split={split!r} "
            f"and targets={list(target_aliases)}."
        )

    frame = subset.copy()
    frame["event_timestamp"] = _normalize_timestamps(frame["event_timestamp"])
    frame = frame[frame["event_timestamp"].notna()].copy()
    frame[score_col] = frame["predicted_probability"].astype(float)
    frame["y_true"] = frame["y_true"].fillna(0).astype(int)
    frame = frame.sort_values("event_timestamp").reset_index(drop=True)
    return frame


def _alignment_mismatch_rate(left: pd.Series, right: pd.Series) -> float:
    if len(left) == 0:
        return 0.0
    if left.dtype == "datetime64[ns, UTC]" or pd.api.types.is_datetime64_any_dtype(left):
        left_norm = _normalize_timestamps(left)
        right_norm = _normalize_timestamps(right)
        mismatches = int((left_norm != right_norm).sum())
        return mismatches / len(left)
    left_norm = left.fillna("__MISSING__").astype(str)
    right_norm = right.fillna("__MISSING__").astype(str)
    mismatches = int((left_norm != right_norm).sum())
    return mismatches / len(left)


def _merge_on_keys(
    rf_frame: pd.DataFrame,
    tab_frame: pd.DataFrame,
    join_keys: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    merged = pd.merge(
        rf_frame,
        tab_frame,
        on=join_keys,
        how="inner",
        suffixes=("_rf", "_tab"),
    )
    meta = {
        "join_strategy_used": "key_merge",
        "join_keys": join_keys,
        "rf_row_count": int(len(rf_frame)),
        "tab_row_count": int(len(tab_frame)),
        "joined_row_count": int(len(merged)),
    }
    if len(merged) != len(rf_frame) or len(merged) != len(tab_frame):
        meta["join_warning"] = "key_merge_row_loss"
    return merged, meta


def _column_values(
    primary: pd.DataFrame,
    col: str,
    length: int,
    *,
    secondary: pd.DataFrame | None = None,
) -> np.ndarray:
    if col in primary.columns:
        return primary[col].values
    if secondary is not None and col in secondary.columns:
        return secondary[col].values
    return np.full(length, np.nan, dtype=object)


def _try_row_order_join(
    rf_frame: pd.DataFrame,
    tab_frame: pd.DataFrame,
    *,
    mismatch_threshold: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rf_count = len(rf_frame)
    tab_count = len(tab_frame)
    if rf_count != tab_count:
        raise ValueError(
            "RF/Tab row-order fallback rejected: row_count mismatch "
            f"(rf={rf_count}, tab={tab_count})."
        )

    timestamp_mismatch_rate = _alignment_mismatch_rate(
        rf_frame["event_timestamp"],
        tab_frame["event_timestamp"],
    )
    pair_mismatch_rate: float | None = None
    if "pair_address" in rf_frame.columns and "pair_address" in tab_frame.columns:
        pair_mismatch_rate = _alignment_mismatch_rate(
            rf_frame["pair_address"],
            tab_frame["pair_address"],
        )

    mismatch_rates = [timestamp_mismatch_rate]
    if pair_mismatch_rate is not None:
        mismatch_rates.append(pair_mismatch_rate)
    max_mismatch = max(mismatch_rates)
    if max_mismatch > mismatch_threshold:
        details = {
            "timestamp_mismatch_rate": round(timestamp_mismatch_rate, 6),
            "pair_mismatch_rate": (
                round(pair_mismatch_rate, 6) if pair_mismatch_rate is not None else None
            ),
            "mismatch_threshold": mismatch_threshold,
        }
        raise ValueError(
            "RF/Tab row-order fallback rejected: alignment mismatch rate exceeds threshold. "
            f"{details}"
        )

    joined = pd.DataFrame({
        "event_timestamp": rf_frame["event_timestamp"].values,
        "pair_address": _column_values(rf_frame, "pair_address", rf_count, secondary=tab_frame),
        "symbol": _column_values(rf_frame, "symbol", rf_count, secondary=tab_frame),
        "y_true": rf_frame["y_true"].values,
        "rf_score": rf_frame["rf_score"].astype(float).values,
        "tab_score": tab_frame["tab_score"].astype(float).values,
        "target_return_4h": _column_values(
            rf_frame, "target_return_4h", rf_count, secondary=tab_frame
        ),
        "future_return_4h": _column_values(
            rf_frame, "future_return_4h", rf_count, secondary=tab_frame
        ),
    })
    meta = {
        "join_strategy_used": "row_order_fallback",
        "join_keys": ["row_order"],
        "rf_row_count": rf_count,
        "tab_row_count": tab_count,
        "joined_row_count": int(len(joined)),
        "timestamp_mismatch_rate": round(timestamp_mismatch_rate, 6),
        "pair_mismatch_rate": (
            round(pair_mismatch_rate, 6) if pair_mismatch_rate is not None else None
        ),
        "max_mismatch_rate": round(max_mismatch, 6),
        "mismatch_threshold": mismatch_threshold,
    }
    return joined, meta



def join_rf_tab_predictions(
    rf_frame: pd.DataFrame,
    tab_frame: pd.DataFrame,
    *,
    mismatch_threshold: float = DEFAULT_JOIN_MISMATCH_THRESHOLD,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join RF and Tab prediction rows.

    Preferred behavior:
    1. Try deterministic key joins.
    2. Accept high-overlap inner joins instead of rejecting an entire Tab run
       because a tiny number of rows are missing.
    3. Report dropped RF-only/TAB-only rows clearly in join metadata.
    4. Fall back to row-order join only when row counts match and alignment is verified.

    Default relaxed key-join tolerance:
    - allow up to 1% row loss on key joins
    - reject obvious many-to-many explosions
    """
    rf = rf_frame.copy()
    tab = tab_frame.copy()

    if "event_timestamp" not in rf.columns or "event_timestamp" not in tab.columns:
        raise ValueError("Both RF and Tab frames must include 'event_timestamp'.")

    rf["event_timestamp"] = _normalize_timestamps(rf["event_timestamp"])
    tab["event_timestamp"] = _normalize_timestamps(tab["event_timestamp"])
    rf = rf[rf["event_timestamp"].notna()].copy()
    tab = tab[tab["event_timestamp"].notna()].copy()

    key_candidates: list[list[str]] = []
    if "pair_address" in rf.columns and "pair_address" in tab.columns:
        key_candidates.append(["event_timestamp", "pair_address"])
    if "symbol" in rf.columns and "symbol" in tab.columns:
        key_candidates.append(["event_timestamp", "symbol"])

    # Timestamp-only joins can create accidental many-to-many matches.
    # Use timestamp-only only as a strict/perfect final key-merge attempt.
    key_candidates.append(["event_timestamp"])

    rf_count = int(len(rf))
    tab_count = int(len(tab))
    max_count = max(rf_count, tab_count)
    min_count = min(rf_count, tab_count)

    # For key-based inner joins, allow small row loss by default.
    # mismatch_threshold can loosen this, but never below the practical 1% default.
    allowed_loss_rate = max(float(mismatch_threshold or 0.0), 0.01)
    min_overlap_ratio = max(0.0, 1.0 - allowed_loss_rate)

    best_partial_error: str | None = None
    best_partial_meta: dict[str, Any] | None = None

    for join_keys in key_candidates:
        merged, meta = _merge_on_keys(rf, tab, join_keys)
        joined_count = int(len(merged))

        meta["mismatch_threshold"] = mismatch_threshold
        meta["allowed_key_join_loss_rate"] = round(allowed_loss_rate, 6)
        meta["rf_only_rows_dropped"] = max(0, rf_count - joined_count)
        meta["tab_only_rows_dropped"] = max(0, tab_count - joined_count)
        meta["join_overlap_ratio_vs_smaller"] = (
            round(joined_count / min_count, 6) if min_count else 0.0
        )
        meta["join_overlap_ratio_vs_larger"] = (
            round(joined_count / max_count, 6) if max_count else 0.0
        )

        # Perfect join remains the cleanest case.
        if joined_count == rf_count == tab_count:
            meta["join_warning"] = None
            return _standardize_joined_frame(merged), meta

        # Reject suspicious many-to-many explosions.
        if joined_count > max_count:
            best_partial_error = (
                f"key_merge on {join_keys} produced join explosion "
                f"(joined={joined_count}, rf={rf_count}, tab={tab_count})"
            )
            best_partial_meta = meta
            continue

        # Accept high-overlap key joins.
        # This is the important fix: do not skip a whole experiment because
        # 26/51461 or 38/77191 rows are missing.
        overlap = meta["join_overlap_ratio_vs_larger"]
        if joined_count > 0 and overlap >= min_overlap_ratio:
            meta["join_warning"] = "accepted_high_overlap_inner_join"
            meta["join_loss_rate_vs_larger"] = round(1.0 - overlap, 6)
            return _standardize_joined_frame(merged), meta

        best_partial_error = (
            f"key_merge on {join_keys} produced insufficient overlap "
            f"(joined={joined_count}, rf={rf_count}, tab={tab_count}, "
            f"overlap_vs_larger={overlap})"
        )
        best_partial_meta = meta

    try:
        joined, meta = _try_row_order_join(rf, tab, mismatch_threshold=mismatch_threshold)
        if best_partial_error:
            meta["key_merge_failure"] = best_partial_error
            meta["best_partial_key_merge"] = best_partial_meta
        return _standardize_joined_frame(joined), meta
    except ValueError as exc:
        details = str(exc)
        if best_partial_error:
            details += f" Last key-merge attempt: {best_partial_error}"
        raise ValueError(details) from exc


def _pick_column(frame: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series(np.full(len(frame), np.nan, dtype=object), index=frame.index)


def _standardize_joined_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "rf_score" not in frame.columns:
        if "predicted_probability_rf" in frame.columns:
            frame = frame.assign(rf_score=frame["predicted_probability_rf"])
        else:
            raise ValueError("Joined frame missing RF score column.")

    if "tab_score" not in frame.columns:
        if "predicted_probability_tab" in frame.columns:
            frame = frame.assign(tab_score=frame["predicted_probability_tab"])
        else:
            raise ValueError("Joined frame missing Tab score column.")

    if "y_true" not in frame.columns:
        if "y_true_rf" in frame.columns:
            y_true = frame["y_true_rf"]
        elif "y_true_tab" in frame.columns:
            y_true = frame["y_true_tab"]
        else:
            raise ValueError("Joined frame missing y_true column.")
        frame = frame.assign(y_true=y_true.fillna(0).astype(int))

    standardized = pd.DataFrame({
        "event_timestamp": _pick_column(
            frame,
            "event_timestamp",
            "event_timestamp_rf",
            "event_timestamp_tab",
        ),
        "pair_address": _pick_column(
            frame,
            "pair_address",
            "pair_address_rf",
            "pair_address_tab",
        ),
        "symbol": _pick_column(
            frame,
            "symbol",
            "symbol_rf",
            "symbol_tab",
        ),
        "y_true": frame["y_true"].fillna(0).astype(int),
        "rf_score": frame["rf_score"].astype(float),
        "tab_score": frame["tab_score"].astype(float),
    })

    for col in ("target_return_4h", "future_return_4h"):
        if col in frame.columns:
            standardized[col] = frame[col]
            continue
        rf_col = f"{col}_rf"
        tab_col = f"{col}_tab"
        if rf_col in frame.columns:
            standardized[col] = frame[rf_col]
        elif tab_col in frame.columns:
            standardized[col] = frame[tab_col]

    return standardized

def compute_score_diagnostics(rf_scores: np.ndarray, tab_scores: np.ndarray) -> dict[str, Any]:
    rf = np.asarray(rf_scores, dtype=float)
    tab = np.asarray(tab_scores, dtype=float)
    tab_unique = int(len(np.unique(tab))) if tab.size else 0
    tab_std = float(np.std(tab)) if tab.size else 0.0
    return {
        "rf_score_min": round(float(np.min(rf)), 6) if rf.size else None,
        "rf_score_max": round(float(np.max(rf)), 6) if rf.size else None,
        "rf_score_mean": round(float(np.mean(rf)), 6) if rf.size else None,
        "rf_score_std": round(float(np.std(rf)), 6) if rf.size else None,
        "tab_score_min": round(float(np.min(tab)), 6) if tab.size else None,
        "tab_score_max": round(float(np.max(tab)), 6) if tab.size else None,
        "tab_score_mean": round(float(np.mean(tab)), 6) if tab.size else None,
        "tab_score_std": round(tab_std, 6) if tab.size else None,
        "tab_unique_score_count": tab_unique,
        "tab_constant_score_flag": bool(tab.size > 0 and (tab_unique <= 1 or tab_std == 0.0)),
    }


def combine_scores(
    rf_scores: np.ndarray,
    tab_scores: np.ndarray,
    method: str,
) -> np.ndarray:
    if method not in COMBINATION_METHODS:
        raise ValueError(f"Unknown combination method: {method!r}")
    rf = np.asarray(rf_scores, dtype=float)
    tab = np.asarray(tab_scores, dtype=float)
    if method == "rf_only":
        return rf.copy()
    if method == "tab_only":
        return tab.copy()
    if method == "average":
        return (rf + tab) / 2.0
    if method == "minimum":
        return np.minimum(rf, tab)
    if method == "maximum":
        return np.maximum(rf, tab)
    if method == "product":
        return rf * tab
    raise ValueError(f"Unhandled combination method: {method!r}")


def top_percent_probability_cutoff(probabilities: np.ndarray, top_pct: float) -> float:
    probs = np.asarray(probabilities, dtype=float)
    probs = probs[np.isfinite(probs)]
    if probs.size == 0:
        return 1.0
    k = max(1, int(probs.size * top_pct / 100.0))
    sorted_desc = np.sort(probs)[::-1]
    return float(sorted_desc[min(k - 1, sorted_desc.size - 1)])


def policy_probability_cutoff(
    policy_name: str,
    validation_probabilities: np.ndarray,
    *,
    rank_cutoffs: dict[str, float] | None = None,
) -> tuple[float | None, float | None]:
    for name, top_pct in RANK_POLICIES:
        if policy_name == name:
            if rank_cutoffs is not None and policy_name in rank_cutoffs:
                return rank_cutoffs[policy_name], top_pct
            return top_percent_probability_cutoff(validation_probabilities, top_pct), top_pct
    for name, cutoff in FIXED_THRESHOLD_POLICIES:
        if policy_name == name:
            return cutoff, None
    raise ValueError(f"Unknown policy: {policy_name}")


def derive_rank_cutoffs(validation_probabilities: np.ndarray) -> dict[str, float]:
    return {
        policy_name: top_percent_probability_cutoff(validation_probabilities, top_pct)
        for policy_name, top_pct in RANK_POLICIES
    }


def evaluate_policy_metrics(
    frame: pd.DataFrame,
    y_score: np.ndarray,
    *,
    policy_name: str,
    probability_cutoff: float,
    top_percent: float | None,
    return_col: str | None,
) -> dict[str, Any]:
    y_true = frame["y_true"].fillna(0).astype(int).to_numpy()
    scores = np.asarray(y_score, dtype=float)
    mask = scores >= probability_cutoff
    selected = frame.loc[mask]
    trade_count = int(len(selected))

    metrics: dict[str, Any] = {
        "policy_name": policy_name,
        "probability_cutoff": round(float(probability_cutoff), 6),
        "top_percent": top_percent,
        "selected_trade_count": trade_count,
        "precision": round(float(selected["y_true"].mean()), 6) if trade_count else None,
    }

    if top_percent is not None:
        prec = precision_at_top_k_with_count(y_true, scores, top_percent)
        metrics["precision_at_top_pct"] = prec["precision"]
        metrics["trade_count_at_top_pct"] = prec["trade_count"]
        if return_col:
            policy = return_metrics_for_top_k(frame, scores, top_percent, return_col)
            metrics["total_return_4h"] = policy["total_return"]
            metrics["mean_return_4h"] = policy["mean_return"]
            metrics["win_rate"] = policy["win_rate"]
    else:
        if return_col and trade_count:
            returns = selected[return_col].astype(float)
            metrics["total_return_4h"] = round(float(returns.sum()), 6)
            metrics["mean_return_4h"] = round(float(returns.mean()), 6)
            metrics["win_rate"] = round(float((selected["y_true"] == 1).mean()), 6)
        else:
            metrics["total_return_4h"] = None
            metrics["mean_return_4h"] = None
            metrics["win_rate"] = None

    return metrics


def compute_stability_flag(
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any],
    *,
    min_trade_count: int = MIN_STABILITY_TRADE_COUNT,
) -> str:
    val_return = validation_metrics.get("total_return_4h")
    test_return = test_metrics.get("total_return_4h")
    val_trades = int(validation_metrics.get("selected_trade_count") or 0)
    test_trades = int(test_metrics.get("selected_trade_count") or 0)

    if val_trades < min_trade_count or test_trades < min_trade_count:
        return "unstable"
    if val_return is None or test_return is None:
        return "unstable"

    val_positive = float(val_return) > 0.0
    test_positive = float(test_return) > 0.0
    if val_positive and test_positive:
        return "stable_positive"
    if val_positive and not test_positive:
        return "validation_only"
    return "unstable"


def select_best_validation_policy(policy_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [row for row in policy_rows if int(row.get("selected_trade_count") or 0) > 0]
    if not candidates:
        candidates = list(policy_rows)
    if not candidates:
        return None

    def sort_key(row: dict[str, Any]) -> tuple:
        total_return = row.get("total_return_4h")
        if total_return is None:
            total_return = float("-inf")
        precision = row.get("precision")
        if precision is None:
            precision = 0.0
        trade_count = int(row.get("selected_trade_count") or 0)
        return (float(total_return), float(precision), trade_count)

    return max(candidates, key=sort_key)


def evaluate_rf_tab_candidate(
    *,
    tab_suffix: str | None,
    tab_metadata: dict[str, Any],
    combination_method: str,
    val_joined: pd.DataFrame,
    test_joined: pd.DataFrame,
    join_meta: dict[str, Any],
    return_col: str | None,
    min_stability_trade_count: int = MIN_STABILITY_TRADE_COUNT,
) -> dict[str, Any]:
    val_scores = combine_scores(
        val_joined["rf_score"].to_numpy(),
        val_joined["tab_score"].to_numpy(),
        combination_method,
    )
    test_scores = combine_scores(
        test_joined["rf_score"].to_numpy(),
        test_joined["tab_score"].to_numpy(),
        combination_method,
    )
    rank_cutoffs = derive_rank_cutoffs(val_scores)

    diagnostics = compute_score_diagnostics(
        val_joined["rf_score"].to_numpy(),
        val_joined["tab_score"].to_numpy(),
    )
    diagnostics.update({
        "validation_row_count": int(len(val_joined)),
        "test_row_count": int(len(test_joined)),
    })

    policy_results: list[dict[str, Any]] = []
    for policy_name in ALL_POLICY_NAMES:
        cutoff, top_percent = policy_probability_cutoff(
            policy_name,
            val_scores,
            rank_cutoffs=rank_cutoffs,
        )
        assert cutoff is not None
        val_metrics = evaluate_policy_metrics(
            val_joined,
            val_scores,
            policy_name=policy_name,
            probability_cutoff=cutoff,
            top_percent=top_percent,
            return_col=return_col,
        )
        test_metrics = evaluate_policy_metrics(
            test_joined,
            test_scores,
            policy_name=policy_name,
            probability_cutoff=cutoff,
            top_percent=top_percent,
            return_col=return_col,
        )
        policy_results.append({
            "policy_name": policy_name,
            "validation": val_metrics,
            "test": test_metrics,
        })

    best_val = select_best_validation_policy([row["validation"] for row in policy_results])
    best_policy_name = best_val["policy_name"] if best_val else None
    best_test = next(
        (row["test"] for row in policy_results if row["policy_name"] == best_policy_name),
        None,
    ) if best_policy_name else None

    stability: dict[str, Any] = {}
    if best_val and best_test:
        val_return = best_val.get("total_return_4h")
        test_return = best_test.get("total_return_4h")
        val_precision = best_val.get("precision")
        test_precision = best_test.get("precision")
        stability = {
            "best_policy_name": best_policy_name,
            "validation_total_return_4h": val_return,
            "test_total_return_4h": test_return,
            "validation_precision": val_precision,
            "test_precision": test_precision,
            "validation_to_test_return_delta": (
                round(float(test_return) - float(val_return), 6)
                if val_return is not None and test_return is not None
                else None
            ),
            "validation_to_test_precision_delta": (
                round(float(test_precision) - float(val_precision), 6)
                if val_precision is not None and test_precision is not None
                else None
            ),
            "stability_flag": compute_stability_flag(
                best_val,
                best_test,
                min_trade_count=min_stability_trade_count,
            ),
            "selected_trade_count_validation": best_val.get("selected_trade_count"),
            "selected_trade_count_test": best_test.get("selected_trade_count"),
        }

    return {
        "tab_suffix": tab_suffix,
        "tab_metadata": tab_metadata,
        "combination_method": combination_method,
        "join": join_meta,
        "diagnostics": diagnostics,
        "policies": policy_results,
        "best_policy": stability,
    }


def _flatten_grid_rows(candidate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    base = {
        "tab_suffix": candidate.get("tab_suffix"),
        "combination_method": candidate.get("combination_method"),
        **{
            f"tab_{key}": value
            for key, value in (candidate.get("tab_metadata") or {}).items()
        },
        **(candidate.get("diagnostics") or {}),
        **{
            f"join_{key}": value
            for key, value in (candidate.get("join") or {}).items()
        },
    }
    best = candidate.get("best_policy") or {}
    for policy in candidate.get("policies") or []:
        row = {
            **base,
            "policy_name": policy.get("policy_name"),
            "validation_selected_trade_count": (policy.get("validation") or {}).get("selected_trade_count"),
            "test_selected_trade_count": (policy.get("test") or {}).get("selected_trade_count"),
            "validation_total_return_4h": (policy.get("validation") or {}).get("total_return_4h"),
            "test_total_return_4h": (policy.get("test") or {}).get("total_return_4h"),
            "validation_precision": (policy.get("validation") or {}).get("precision"),
            "test_precision": (policy.get("test") or {}).get("precision"),
            "is_best_validation_policy": policy.get("policy_name") == best.get("best_policy_name"),
            "best_policy_stability_flag": best.get("stability_flag"),
            "validation_to_test_return_delta": best.get("validation_to_test_return_delta"),
            "validation_to_test_precision_delta": best.get("validation_to_test_precision_delta"),
        }
        rows.append(row)
    return rows


def run_rf_tab_matrix(
    *,
    models_dir: Path | None = None,
    backtest_dir: Path | None = None,
    tab_suffixes: list[str | None] | None = None,
    combination_methods: list[str] | None = None,
    mismatch_threshold: float = DEFAULT_JOIN_MISMATCH_THRESHOLD,
    min_stability_trade_count: int = MIN_STABILITY_TRADE_COUNT,
    rf_target_aliases: tuple[str, ...] = RF_TARGET_ALIASES,
    tab_target_aliases: tuple[str, ...] = TAB_TARGET_ALIASES,
) -> dict[str, Any]:
    models_dir = models_dir or MODELS_DIR
    backtest_dir = backtest_dir or DEFAULT_BACKTEST_DIR
    backtest_dir.mkdir(parents=True, exist_ok=True)

    rf_val_path = models_dir / "predictions_validation.parquet"
    rf_test_path = models_dir / "predictions_test.parquet"
    if not rf_val_path.is_file() or not rf_test_path.is_file():
        raise FileNotFoundError(
            "RF prediction parquet files missing. Run train_baseline_model.py first."
        )

    rf_val_preds = pd.read_parquet(rf_val_path)
    rf_test_preds = pd.read_parquet(rf_test_path)

    suffixes = tab_suffixes if tab_suffixes is not None else discover_tab_suffixes(models_dir)
    methods = combination_methods or list(COMBINATION_METHODS)

    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for suffix in suffixes:
        val_path, test_path = tab_prediction_paths(models_dir, suffix)
        if not val_path.is_file() or not test_path.is_file():
            skipped.append({
                "tab_suffix": suffix,
                "reason": "tab_prediction_files_missing",
                "validation_path": str(val_path),
                "test_path": str(test_path),
            })
            continue

        tab_metadata = load_tab_metadata(tab_report_path(backtest_dir, suffix))
        tab_val_preds = pd.read_parquet(val_path)
        tab_test_preds = pd.read_parquet(test_path)

        try:
            rf_val = prepare_prediction_frame(
                rf_val_preds,
                model_name=RF_MODEL_NAME,
                target_aliases=rf_target_aliases,
                split="validation",
                score_col="rf_score",
            )
            rf_test = prepare_prediction_frame(
                rf_test_preds,
                model_name=RF_MODEL_NAME,
                target_aliases=rf_target_aliases,
                split="test",
                score_col="rf_score",
            )
            tab_val = prepare_prediction_frame(
                tab_val_preds,
                model_name=TAB_MODEL_NAME,
                target_aliases=tab_target_aliases,
                split="validation",
                score_col="tab_score",
            )
            tab_test = prepare_prediction_frame(
                tab_test_preds,
                model_name=TAB_MODEL_NAME,
                target_aliases=tab_target_aliases,
                split="test",
                score_col="tab_score",
            )
            val_joined, val_join_meta = join_rf_tab_predictions(
                rf_val,
                tab_val,
                mismatch_threshold=mismatch_threshold,
            )
            test_joined, test_join_meta = join_rf_tab_predictions(
                rf_test,
                tab_test,
                mismatch_threshold=mismatch_threshold,
            )
        except ValueError as exc:
            skipped.append({
                "tab_suffix": suffix,
                "reason": "join_failed",
                "error": str(exc),
            })
            continue

        return_col = resolve_return_column(val_joined)
        join_meta = {
            "validation": val_join_meta,
            "test": test_join_meta,
        }

        for method in methods:
            candidate = evaluate_rf_tab_candidate(
                tab_suffix=suffix,
                tab_metadata=tab_metadata,
                combination_method=method,
                val_joined=val_joined,
                test_joined=test_joined,
                join_meta=join_meta,
                return_col=return_col,
                min_stability_trade_count=min_stability_trade_count,
            )
            if candidate.get("best_policy"):
                candidate["best_policy"]["min_stability_trade_count"] = min_stability_trade_count
            candidates.append(candidate)

    ranked = sorted(
        [c for c in candidates if c.get("best_policy")],
        key=lambda item: (
            float((item.get("best_policy") or {}).get("validation_total_return_4h") or float("-inf")),
            float((item.get("best_policy") or {}).get("validation_precision") or 0.0),
        ),
        reverse=True,
    )
    for rank, candidate in enumerate(ranked, start=1):
        if candidate.get("best_policy"):
            candidate["best_policy"]["validation_rank"] = rank

    grid_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        grid_rows.extend(_flatten_grid_rows(candidate))

    baseline_metrics = load_baseline_metrics(models_dir)
    report: dict[str, Any] = {
        **base_report_flags(),
        "generated_at": _utcnow_iso(),
        "models_dir": str(models_dir),
        "backtest_dir": str(backtest_dir),
        "rf_model_name": RF_MODEL_NAME,
        "tab_model_name": TAB_MODEL_NAME,
        "rf_target_aliases": list(rf_target_aliases),
        "tab_target_aliases": list(tab_target_aliases),
        "combination_methods": methods,
        "join_mismatch_threshold": mismatch_threshold,
        "min_stability_trade_count": min_stability_trade_count,
        "tab_suffixes_requested": [suffix or "" for suffix in suffixes],
        "candidates_evaluated": len(candidates),
        "candidates_skipped": skipped,
        "ranking_basis": [
            "validation_total_return_4h",
            "validation_precision",
        ],
        "candidates": candidates,
        "best_candidate": ranked[0] if ranked else None,
        "baseline_metrics_available": baseline_metrics is not None,
    }
    return report


def write_rf_tab_matrix_outputs(
    report: dict[str, Any],
    *,
    report_path: Path | None = None,
    grid_path: Path | None = None,
) -> dict[str, str]:
    report_path = report_path or DEFAULT_REPORT_PATH
    grid_path = grid_path or DEFAULT_GRID_PATH
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    grid_rows: list[dict[str, Any]] = []
    for candidate in report.get("candidates") or []:
        grid_rows.extend(_flatten_grid_rows(candidate))
    if grid_rows:
        pd.DataFrame(grid_rows).to_parquet(grid_path, index=False)

    output_files = {"report_json": str(report_path)}
    if grid_rows:
        output_files["grid_parquet"] = str(grid_path)
    report["output_files"] = output_files
    return output_files
