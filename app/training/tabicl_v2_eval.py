"""Offline TabICLv2 evaluation helpers (no live trading, no LLM calls)."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler

from app.training.baseline_model import (
    MODELS_DIR,
    PREDICTION_META_COLUMNS,
    TRAINING_DIR,
    DEFAULT_DATASET_PATH,
    precision_at_top_k,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = TRAINING_DIR
DEFAULT_TARGET = "target_profitable_4h"
DEFAULT_TRAIN_FRAC = 0.70
DEFAULT_VAL_FRAC = 0.15

CONTEXT_STRATEGIES = frozenset({
    "stratified_recent",
    "positive_enriched",
    "nearest_neighbors_context",
    "whale_wave_context",
    "ensemble_small_contexts",
})

WHALE_WAVE_COLUMN_KEYWORDS = (
    "whale",
    "wave",
    "volume",
    "liquidity",
    "txns",
    "buy",
    "sell",
    "ratio",
    "price_change",
    "momentum",
)

DEFAULT_POSITIVE_CONTEXT_RATIO = {
    "stratified_recent": 0.25,
    "positive_enriched": 0.50,
}

ENSEMBLE_MEMBER_STRATEGIES = (
    "stratified_recent",
    "positive_enriched",
    "nearest_neighbors_context",
    "whale_wave_context",
)

DEFAULT_KNN_ROLLING_DAYS = 14
DEFAULT_KNN_MIN_CONTEXT_ROWS = 512
DEFAULT_KNN_EXPAND_WINDOW = True
DEFAULT_KNN_MAX_ROLLING_DAYS = 90
DEFAULT_KNN_TIME_DECAY_ALPHA = 0.0
DEFAULT_KNN_CACHE_TIME_BUCKET = "day"

TABICL_PREDICTIONS_VAL = MODELS_DIR / "tabicl_v2_predictions_validation.parquet"
TABICL_PREDICTIONS_TEST = MODELS_DIR / "tabicl_v2_predictions_test.parquet"
TABICL_REPORT_PATH = TRAINING_DIR / "policy_backtests" / "tabicl_v2_report.json"

RF_VALIDATION_PREDICTIONS = MODELS_DIR / "predictions_validation.parquet"
RF_TEST_PREDICTIONS = MODELS_DIR / "predictions_test.parquet"
RF_METRICS_PATH = MODELS_DIR / "baseline_metrics.json"

LEAKAGE_SUBSTRINGS = (
    "future",
    "outcome",
    "return_15m",
    "return_1h",
    "return_4h",
    "target_profitable",
    "target_return",
    "label_up",
    "big_pump",
    "big_dump",
    "pump_then_dump",
    "optimal_trade_class",
    "position_size_multiplier",
    "labeled",
    "pending",
    "profitable_after_fees",
    "positive_return",
    "realized",
    "oracle",
    "max_future_return",
    "min_future_return",
    "max_upside",
    "max_drawdown",
)

IDENTIFIER_EXACT = frozenset({
    "event_timestamp",
    "timestamp",
    "created_at",
    "symbol",
    "pair_address",
    "coin_id",
    "raw_json",
    "reasoning",
    "prompt",
    "response",
    "decision",
    "action",
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
    "ts",
})


def tabicl_available() -> bool:
    try:
        import tabicl  # noqa: F401

        return True
    except ImportError:
        return False


def _contains_leakage(name: str) -> bool:
    lower = name.lower()
    return any(sub in lower for sub in LEAKAGE_SUBSTRINGS)


def select_tabicl_feature_columns(
    frame: pd.DataFrame,
    *,
    target: str | None = None,
) -> tuple[list[str], list[str]]:
    """Return (numeric_features, excluded_columns). Numeric/bool only.

    Safety rule: the selected target and any target_* columns are never features.
    """
    excluded: list[str] = []
    numeric: list[str] = []

    for col in frame.columns:
        lower_col = col.lower()
        if (target is not None and col == target) or lower_col.startswith("target_"):
            excluded.append(col)
            continue
        if col in IDENTIFIER_EXACT or _contains_leakage(col):
            excluded.append(col)
            continue
        dtype = frame[col].dtype
        if pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype):
            numeric.append(col)
            continue
        excluded.append(col)

    return numeric, excluded


def chronological_split(
    frame: pd.DataFrame,
    *,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    val_frac: float = DEFAULT_VAL_FRAC,
    timestamp_col: str = "event_timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if timestamp_col in frame.columns:
        ordered = frame.sort_values(timestamp_col).reset_index(drop=True)
    else:
        ordered = frame.reset_index(drop=True)
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
    from app.training.baseline_model import assert_chronological_splits as _assert

    _assert(train_df, val_df, test_df, timestamp_col=timestamp_col)


def cap_context_size(context_size: int, max_train_context_rows: int, train_row_count: int) -> int:
    """Cap context size safely when train rows are fewer."""
    if train_row_count <= 0:
        return 0
    return max(1, min(context_size, max_train_context_rows, train_row_count))


def sample_context_indices(
    y_train: np.ndarray,
    *,
    context_size: int,
    positive_fraction_target: float = 0.25,
    random_state: int = 42,
) -> np.ndarray:
    """Deterministic train-only context sampling with positive/negative balance."""
    n = len(y_train)
    if n == 0:
        return np.array([], dtype=int)
    effective_size = min(context_size, n)
    y_train = np.asarray(y_train, dtype=int)

    pos_idx = np.flatnonzero(y_train == 1)
    neg_idx = np.flatnonzero(y_train == 0)

    rng = np.random.RandomState(random_state)
    n_pos_target = min(len(pos_idx), max(0, int(round(effective_size * positive_fraction_target))))
    if len(pos_idx) > 0 and n_pos_target == 0:
        n_pos_target = 1

    if len(pos_idx) <= n_pos_target:
        selected_pos = pos_idx
    else:
        selected_pos = rng.choice(pos_idx, size=n_pos_target, replace=False)

    n_neg = effective_size - len(selected_pos)
    if len(neg_idx) <= n_neg:
        selected_neg = neg_idx
    else:
        selected_neg = neg_idx[-n_neg:]

    indices = np.concatenate([selected_pos, selected_neg])
    if len(indices) < effective_size:
        used = set(indices.tolist())
        remaining = np.array([i for i in range(n) if i not in used], dtype=int)
        need = effective_size - len(indices)
        if len(remaining) > 0:
            extra = remaining[-need:] if len(remaining) >= need else remaining
            indices = np.concatenate([indices, extra])

    indices = np.unique(indices)
    if len(indices) > effective_size:
        pos_keep = indices[np.isin(indices, selected_pos)]
        neg_keep = indices[np.isin(indices, selected_neg)]
        if len(pos_keep) + len(neg_keep) > effective_size:
            trim_neg = max(0, len(neg_keep) - (effective_size - len(pos_keep)))
            if trim_neg > 0:
                neg_keep = neg_keep[trim_neg:]
            indices = np.concatenate([pos_keep, neg_keep])
        else:
            indices = indices[:effective_size]

    return np.sort(indices.astype(int))


def validate_context_strategy(context_strategy: str | None) -> None:
    if context_strategy is None:
        return
    if context_strategy not in CONTEXT_STRATEGIES:
        raise ValueError(
            f"Invalid context_strategy={context_strategy!r}. "
            f"Choose one of: {', '.join(sorted(CONTEXT_STRATEGIES))}"
        )


def resolve_output_label(
    context_strategy: str | None,
    output_suffix: str | None,
) -> str | None:
    if output_suffix is not None and str(output_suffix).strip():
        label = str(output_suffix).strip().strip("_")
        if not label:
            raise ValueError("--output-suffix must not be empty or only underscores.")
        return label
    if context_strategy:
        return context_strategy
    return None


def prediction_output_paths(
    output_dir: Path,
    output_label: str | None,
) -> tuple[Path, Path, Path, Path | None]:
    models_dir = output_dir / "models"
    backtest_dir = output_dir / "policy_backtests"
    if output_label:
        val_path = models_dir / f"tabicl_v2_predictions_validation_{output_label}.parquet"
        test_path = models_dir / f"tabicl_v2_predictions_test_{output_label}.parquet"
        report_path = backtest_dir / f"tabicl_v2_report_{output_label}.json"
        features_path = models_dir / f"tabicl_v2_features_{output_label}.json"
    else:
        val_path = models_dir / "tabicl_v2_predictions_validation.parquet"
        test_path = models_dir / "tabicl_v2_predictions_test.parquet"
        report_path = backtest_dir / "tabicl_v2_report.json"
        features_path = None
    return val_path, test_path, report_path, features_path


def resolve_full_evaluation(
    *,
    max_rows: int | None,
    partial_evaluation_reason: str | None,
) -> tuple[bool, str | None]:
    actual_cap = max_rows is not None
    full_evaluation = not actual_cap
    return full_evaluation, partial_evaluation_reason


def select_whale_wave_feature_columns(feature_cols: list[str]) -> list[str]:
    matched = [
        col for col in feature_cols
        if any(keyword in col.lower() for keyword in WHALE_WAVE_COLUMN_KEYWORDS)
    ]
    if len(matched) >= 5:
        return matched
    return list(feature_cols)


def whale_wave_feature_indices(feature_cols: list[str]) -> np.ndarray:
    whale_cols = select_whale_wave_feature_columns(feature_cols)
    return np.array([feature_cols.index(col) for col in whale_cols], dtype=int)


def _deterministic_sample_from_pool(
    pool_indices: np.ndarray,
    y_train: np.ndarray,
    *,
    context_size: int,
    positive_context_ratio: float,
    random_state: int,
    prefer_recent: bool,
) -> np.ndarray:
    n = len(y_train)
    if n == 0:
        return np.array([], dtype=int)
    effective_size = min(context_size, n)
    if len(pool_indices) == 0:
        pool_indices = np.arange(n, dtype=int)
    if prefer_recent and len(pool_indices) > effective_size:
        pool_indices = pool_indices[-max(effective_size * 2, effective_size):]

    y_pool = y_train[pool_indices]
    pos_mask = y_pool == 1
    pos_pool = pool_indices[pos_mask]
    neg_pool = pool_indices[~pos_mask]

    rng = np.random.RandomState(random_state)
    ratio = positive_context_ratio if len(pos_pool) > 0 else 0.0
    n_pos_target = min(len(pos_pool), max(0, int(round(effective_size * ratio))))
    if len(pos_pool) > 0 and n_pos_target == 0:
        n_pos_target = 1

    if len(pos_pool) <= n_pos_target:
        selected_pos = pos_pool
    else:
        selected_pos = rng.choice(pos_pool, size=n_pos_target, replace=False)

    n_neg = effective_size - len(selected_pos)
    if prefer_recent:
        neg_source = neg_pool[-n_neg:] if len(neg_pool) > n_neg else neg_pool
    elif len(neg_pool) <= n_neg:
        neg_source = neg_pool
    else:
        neg_source = rng.choice(neg_pool, size=n_neg, replace=False)

    indices = np.concatenate([selected_pos, neg_source])
    if len(indices) < effective_size:
        used = set(indices.tolist())
        remaining = np.array([i for i in pool_indices if i not in used], dtype=int)
        if prefer_recent:
            remaining = remaining[-(effective_size - len(indices)):]
        need = effective_size - len(indices)
        if len(remaining) > 0:
            extra = remaining[-need:] if len(remaining) >= need else remaining
            indices = np.concatenate([indices, extra])

    indices = np.unique(indices)
    if len(indices) > effective_size:
        indices = indices[-effective_size:] if prefer_recent else indices[:effective_size]
    return np.sort(indices.astype(int))


def sample_stratified_recent_indices(
    y_train: np.ndarray,
    *,
    context_size: int,
    positive_context_ratio: float = 0.25,
    random_state: int = 42,
) -> np.ndarray:
    """Train-only stratified context preferring recent rows."""
    pool = np.arange(len(y_train), dtype=int)
    return _deterministic_sample_from_pool(
        pool,
        y_train,
        context_size=context_size,
        positive_context_ratio=positive_context_ratio,
        random_state=random_state,
        prefer_recent=True,
    )


def sample_positive_enriched_indices(
    y_train: np.ndarray,
    *,
    context_size: int,
    positive_context_ratio: float = 0.50,
    random_state: int = 42,
) -> np.ndarray:
    """Train-only context enriched with positive examples."""
    pool = np.arange(len(y_train), dtype=int)
    return _deterministic_sample_from_pool(
        pool,
        y_train,
        context_size=context_size,
        positive_context_ratio=positive_context_ratio,
        random_state=random_state,
        prefer_recent=False,
    )


class NearestNeighborContextIndex:
    """Train-fitted neighbor index; fit once per slice, query per batch."""

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        metric: str = "euclidean",
        feature_column_indices: np.ndarray | None = None,
        global_index_map: np.ndarray | None = None,
        neighbor_timestamps: np.ndarray | None = None,
    ) -> None:
        self.x_train = x_train
        self.y_train = np.asarray(y_train, dtype=int)
        self.metric = metric
        self.feature_column_indices = feature_column_indices
        self.global_index_map = (
            np.asarray(global_index_map, dtype=int)
            if global_index_map is not None
            else np.arange(len(x_train), dtype=int)
        )
        self.neighbor_timestamps = neighbor_timestamps
        if feature_column_indices is not None and len(feature_column_indices) > 0:
            self._search_matrix = x_train[:, feature_column_indices]
        else:
            self._search_matrix = x_train
        self._fit_count = 0
        n_train = len(self._search_matrix)
        n_neighbors = min(n_train, max(2, n_train)) if n_train else 1
        self._nn = NearestNeighbors(
            n_neighbors=n_neighbors,
            metric=metric,
            algorithm="auto",
        )
        if n_train > 0:
            self._nn.fit(self._search_matrix)
            self._fit_count = 1

    @property
    def fit_count(self) -> int:
        return self._fit_count

    def _query_matrix(self, x_query: np.ndarray) -> np.ndarray:
        if self.feature_column_indices is not None and len(self.feature_column_indices) > 0:
            return x_query[:, self.feature_column_indices]
        return x_query

    def build_context_indices(
        self,
        x_query: np.ndarray,
        *,
        context_size: int,
        exclude_zero_distance: bool = False,
        batch_min_time: pd.Timestamp | None = None,
        time_decay_alpha: float = 0.0,
    ) -> np.ndarray:
        local_indices = self._build_local_context_indices(
            x_query,
            context_size=context_size,
            exclude_zero_distance=exclude_zero_distance,
            batch_min_time=batch_min_time,
            time_decay_alpha=time_decay_alpha,
        )
        return self.global_index_map[local_indices]

    def _build_local_context_indices(
        self,
        x_query: np.ndarray,
        *,
        context_size: int,
        exclude_zero_distance: bool = False,
        batch_min_time: pd.Timestamp | None = None,
        time_decay_alpha: float = 0.0,
    ) -> np.ndarray:
        n_train = len(self.x_train)
        if n_train == 0:
            return np.array([], dtype=int)
        effective_size = min(context_size, n_train)
        if time_decay_alpha > 0:
            retrieve_k = min(n_train, max(effective_size * 2, effective_size + 10))
        else:
            retrieve_k = min(n_train, max(effective_size * 3, effective_size + 10))
        query_matrix = self._query_matrix(x_query)
        distances, indices = self._nn.kneighbors(query_matrix, n_neighbors=retrieve_k)

        min_dist: dict[int, float] = {}
        for row_dists, row_idxs in zip(distances, indices):
            for dist, idx in zip(row_dists, row_idxs):
                idx_int = int(idx)
                if exclude_zero_distance and dist == 0.0:
                    continue
                prev = min_dist.get(idx_int)
                if prev is None or dist < prev:
                    min_dist[idx_int] = float(dist)

        if not min_dist:
            return np.arange(max(0, n_train - effective_size), n_train, dtype=int)

        if (
            time_decay_alpha > 0
            and batch_min_time is not None
            and self.neighbor_timestamps is not None
        ):
            batch_min = _as_utc_timestamp(batch_min_time)
            neighbor_ts = pd.to_datetime(self.neighbor_timestamps, utc=True, errors="coerce")
            scored: list[tuple[int, float]] = []
            ages: list[float] = []
            for idx_int, dist in min_dist.items():
                age_days = max(
                    0.0,
                    (batch_min - _as_utc_timestamp(neighbor_ts[idx_int])).total_seconds() / 86400.0,
                )
                ages.append(age_days)
                scored.append((idx_int, dist))
            max_age = max(ages) if ages else 1.0
            if max_age <= 0:
                max_age = 1.0
            age_by_idx = dict(zip([item[0] for item in scored], ages))
            ranked = sorted(
                scored,
                key=lambda item: item[1] + time_decay_alpha * (age_by_idx[item[0]] / max_age),
            )
        else:
            ranked = sorted(min_dist.items(), key=lambda item: item[1])

        selected: list[int] = []
        for idx, _ in ranked:
            if self.y_train[idx] == 1:
                selected.append(idx)
                if len(selected) >= effective_size:
                    break
        for idx, _ in ranked:
            if len(selected) >= effective_size:
                break
            if idx not in selected:
                selected.append(idx)

        if len(selected) < effective_size:
            recent = np.arange(max(0, n_train - effective_size), n_train, dtype=int)
            for idx in recent:
                if len(selected) >= effective_size:
                    break
                if idx not in selected:
                    selected.append(int(idx))

        return np.array(selected[:effective_size], dtype=int)


@dataclass(frozen=True)
class RollingKnnConfig:
    rolling_days: int = DEFAULT_KNN_ROLLING_DAYS
    min_context_rows: int = DEFAULT_KNN_MIN_CONTEXT_ROWS
    expand_window: bool = DEFAULT_KNN_EXPAND_WINDOW
    max_rolling_days: int = DEFAULT_KNN_MAX_ROLLING_DAYS
    time_decay_alpha: float = DEFAULT_KNN_TIME_DECAY_ALPHA
    cache_time_bucket: str = DEFAULT_KNN_CACHE_TIME_BUCKET


def parse_bool_flag(value: str | bool | None, *, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def rolling_knn_enabled(knn_rolling_days: int | None) -> bool:
    return knn_rolling_days is not None and knn_rolling_days > 0


def _as_utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def knn_time_bucket_key(batch_min_time: pd.Timestamp, bucket: str) -> str:
    ts = _as_utc_timestamp(batch_min_time)
    if bucket == "day":
        return ts.floor("D").isoformat()
    if bucket == "hour":
        return ts.floor("h").isoformat()
    raise ValueError(f"Unsupported knn_cache_time_bucket: {bucket!r}")


def select_rolling_temporal_indices(
    train_timestamps: np.ndarray,
    batch_min_time: pd.Timestamp,
    *,
    rolling_days: int,
    min_context_rows: int,
    expand_window: bool,
    max_rolling_days: int,
) -> tuple[np.ndarray, int, bool]:
    """Return train-only indices strictly before batch_min_time within a rolling window."""
    if len(train_timestamps) == 0:
        return np.array([], dtype=int), rolling_days, True

    batch_min = _as_utc_timestamp(batch_min_time)
    ts_values = pd.to_datetime(train_timestamps, utc=True, errors="coerce").to_numpy(dtype="datetime64[ns]")
    eligible = np.flatnonzero(ts_values < np.datetime64(batch_min.to_datetime64()))
    if len(eligible) == 0:
        return np.array([], dtype=int), rolling_days, True

    effective_window = rolling_days
    selected = np.array([], dtype=int)
    while True:
        window_start = batch_min - pd.Timedelta(days=effective_window)
        window_start64 = np.datetime64(window_start.to_datetime64())
        slice_indices = eligible[ts_values[eligible] >= window_start64]
        selected = np.asarray(slice_indices, dtype=int)
        if len(selected) >= min_context_rows:
            return selected, effective_window, False
        if not expand_window or effective_window >= max_rolling_days:
            break
        next_window = min(max_rolling_days, max(effective_window + rolling_days, effective_window * 2))
        if next_window == effective_window:
            break
        effective_window = next_window

    if len(selected) > 0:
        return selected, effective_window, False

    return eligible, effective_window, True


class RollingKnnContextSelector:
    """Time-aware rolling KNN context builder with per-batch fitting and day-bucket cache."""

    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        train_timestamps: np.ndarray,
        *,
        config: RollingKnnConfig,
        metric: str = "euclidean",
    ) -> None:
        self.x_train = x_train
        self.y_train = np.asarray(y_train, dtype=int)
        self.train_timestamps = train_timestamps
        self.config = config
        self.metric = metric
        self._cache: dict[tuple[str, int], NearestNeighborContextIndex] = {}
        self.knn_index_fit_count = 0
        self.knn_cache_hit_count = 0
        self.knn_cache_miss_count = 0
        self.temporal_slice_sizes: list[int] = []
        self.context_sizes: list[int] = []
        self.fallback_recent_context_count = 0

    def build_global_context_indices(
        self,
        x_query: np.ndarray,
        batch_timestamps: np.ndarray,
        *,
        context_size: int,
        exclude_zero_distance: bool = False,
    ) -> np.ndarray:
        if len(batch_timestamps) == 0:
            raise ValueError("batch_timestamps required for rolling KNN context selection.")
        batch_min_time = min(_as_utc_timestamp(ts) for ts in batch_timestamps)
        slice_indices, effective_window, used_fallback = select_rolling_temporal_indices(
            self.train_timestamps,
            batch_min_time,
            rolling_days=self.config.rolling_days,
            min_context_rows=self.config.min_context_rows,
            expand_window=self.config.expand_window,
            max_rolling_days=self.config.max_rolling_days,
        )
        self.temporal_slice_sizes.append(int(len(slice_indices)))
        if used_fallback:
            self.fallback_recent_context_count += 1

        if len(slice_indices) == 0:
            slice_indices = np.arange(max(0, len(self.x_train) - context_size), len(self.x_train), dtype=int)
            self.fallback_recent_context_count += 1

        cache_key = (
            knn_time_bucket_key(batch_min_time, self.config.cache_time_bucket),
            int(effective_window),
        )
        cached = self._cache.get(cache_key)
        if cached is not None:
            self.knn_cache_hit_count += 1
            nn_index = cached
        else:
            self.knn_cache_miss_count += 1
            x_slice = self.x_train[slice_indices]
            y_slice = self.y_train[slice_indices]
            ts_slice = self.train_timestamps[slice_indices]
            nn_index = NearestNeighborContextIndex(
                x_slice,
                y_slice,
                metric=self.metric,
                global_index_map=slice_indices,
                neighbor_timestamps=ts_slice,
            )
            self._cache[cache_key] = nn_index
            self.knn_index_fit_count += nn_index.fit_count

        global_indices = nn_index.build_context_indices(
            x_query,
            context_size=context_size,
            exclude_zero_distance=exclude_zero_distance,
            batch_min_time=batch_min_time,
            time_decay_alpha=self.config.time_decay_alpha,
        )
        self.context_sizes.append(int(len(global_indices)))
        return global_indices

    def diagnostics(self) -> dict[str, Any]:
        def _stats(values: list[int]) -> dict[str, float | int | None]:
            if not values:
                return {"average": None, "min": None, "max": None}
            return {
                "average": round(float(np.mean(values)), 3),
                "min": int(min(values)),
                "max": int(max(values)),
            }

        temporal = _stats(self.temporal_slice_sizes)
        context = _stats(self.context_sizes)
        return {
            "knn_rolling_days_used": self.config.rolling_days,
            "knn_min_context_rows": self.config.min_context_rows,
            "knn_expand_window": self.config.expand_window,
            "knn_max_rolling_days": self.config.max_rolling_days,
            "knn_time_decay_alpha": self.config.time_decay_alpha,
            "knn_cache_time_bucket": self.config.cache_time_bucket,
            "knn_index_fit_count": self.knn_index_fit_count,
            "knn_cache_hit_count": self.knn_cache_hit_count,
            "knn_cache_miss_count": self.knn_cache_miss_count,
            "average_temporal_slice_rows": temporal["average"],
            "min_temporal_slice_rows": temporal["min"],
            "max_temporal_slice_rows": temporal["max"],
            "average_context_rows_used": context["average"],
            "min_context_rows_used": context["min"],
            "max_context_rows_used": context["max"],
            "fallback_recent_context_count": self.fallback_recent_context_count,
            "rolling_context_mode": "train_only",
            "event_timestamp_used_for_slicing": True,
            "event_timestamp_used_as_feature": False,
        }


def build_rolling_knn_config(
    *,
    knn_rolling_days: int | None,
    knn_min_context_rows: int | None = None,
    knn_expand_window: bool | str | None = None,
    knn_max_rolling_days: int | None = None,
    knn_time_decay_alpha: float | None = None,
    knn_cache_time_bucket: str | None = None,
) -> RollingKnnConfig:
    return RollingKnnConfig(
        rolling_days=knn_rolling_days or DEFAULT_KNN_ROLLING_DAYS,
        min_context_rows=knn_min_context_rows or DEFAULT_KNN_MIN_CONTEXT_ROWS,
        expand_window=parse_bool_flag(knn_expand_window, default=DEFAULT_KNN_EXPAND_WINDOW),
        max_rolling_days=knn_max_rolling_days or DEFAULT_KNN_MAX_ROLLING_DAYS,
        time_decay_alpha=(
            DEFAULT_KNN_TIME_DECAY_ALPHA if knn_time_decay_alpha is None else knn_time_decay_alpha
        ),
        cache_time_bucket=knn_cache_time_bucket or DEFAULT_KNN_CACHE_TIME_BUCKET,
    )


def build_static_context_indices(
    strategy: str,
    y_train: np.ndarray,
    *,
    context_size: int,
    max_train_context_rows: int,
    positive_context_ratio: float | None = None,
    random_state: int = 42,
) -> np.ndarray:
    effective = cap_context_size(context_size, max_train_context_rows, len(y_train))
    if effective <= 0:
        return np.array([], dtype=int)

    if strategy == "stratified_recent":
        ratio = positive_context_ratio if positive_context_ratio is not None else 0.25
        indices = sample_stratified_recent_indices(
            y_train,
            context_size=effective,
            positive_context_ratio=ratio,
            random_state=random_state,
        )
    elif strategy == "positive_enriched":
        ratio = positive_context_ratio if positive_context_ratio is not None else 0.50
        indices = sample_positive_enriched_indices(
            y_train,
            context_size=effective,
            positive_context_ratio=ratio,
            random_state=random_state,
        )
    else:
        indices = sample_context_indices(
            y_train,
            context_size=effective,
            positive_fraction_target=positive_context_ratio or 0.25,
            random_state=random_state,
        )
    return indices[: cap_context_size(effective, max_train_context_rows, len(y_train))]


class TrainOnlyPreprocessor:
    """Median imputer + optional scaler fit on train only."""

    def __init__(self, scaler: str = "standard") -> None:
        self.scaler_name = scaler
        self.imputer = SimpleImputer(strategy="median")
        self.scaler: StandardScaler | RobustScaler | None
        if scaler == "standard":
            self.scaler = StandardScaler()
        elif scaler == "robust":
            self.scaler = RobustScaler()
        elif scaler == "none":
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaler: {scaler}")
        self.feature_names_: list[str] = []
        self._fitted = False

    def fit(self, frame: pd.DataFrame, feature_cols: list[str]) -> TrainOnlyPreprocessor:
        self.feature_names_ = list(feature_cols)
        x = self._sanitize(frame, feature_cols)
        x_imp = self.imputer.fit_transform(x)
        if self.scaler is not None:
            self.scaler.fit(x_imp)
        self._fitted = True
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Preprocessor must be fit on train before transform.")
        x = self._sanitize(frame, self.feature_names_)
        x_imp = self.imputer.transform(x)
        if self.scaler is not None:
            return self.scaler.transform(x_imp)
        return x_imp

    @staticmethod
    def _sanitize(frame: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
        subset = frame[feature_cols].copy()
        for col in feature_cols:
            if pd.api.types.is_numeric_dtype(subset[col]) or pd.api.types.is_bool_dtype(subset[col]):
                subset[col] = subset[col].replace([np.inf, -np.inf], np.nan)
                if pd.api.types.is_bool_dtype(subset[col]):
                    subset[col] = subset[col].astype(float)
        return subset.to_numpy(dtype=float)


def limit_features_by_variance(
    train_frame: pd.DataFrame,
    feature_cols: list[str],
    max_features: int | None,
) -> list[str]:
    if max_features is None or len(feature_cols) <= max_features:
        return feature_cols
    variances = train_frame[feature_cols].astype(float).var(numeric_only=True)
    ranked = variances.sort_values(ascending=False)
    return ranked.head(max_features).index.tolist()


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


def precision_at_top_k_with_count(
    y_true: np.ndarray,
    y_score: np.ndarray,
    k_pct: float,
) -> dict[str, Any]:
    n = len(y_true)
    if n == 0:
        return {"precision": None, "trade_count": 0, "top_pct": k_pct}
    k = max(1, int(n * k_pct / 100.0))
    order = np.argsort(-y_score)
    selected = y_true[order[:k]]
    return {
        "precision": float(selected.mean()),
        "trade_count": int(k),
        "top_pct": k_pct,
    }


def return_metrics_for_top_k(
    frame: pd.DataFrame,
    y_score: np.ndarray,
    k_pct: float,
    return_col: str | None,
) -> dict[str, Any]:
    n = len(frame)
    if n == 0:
        return {
            "mean_return": None,
            "total_return": None,
            "win_rate": None,
            "return_kind": None,
        }
    k = max(1, int(n * k_pct / 100.0))
    order = np.argsort(-y_score)
    selected_idx = frame.index[order[:k]]
    selected = frame.loc[selected_idx]
    y_true = selected.get("y_true", pd.Series(0, index=selected.index)).fillna(0).astype(int)
    result: dict[str, Any] = {
        "win_rate": round(float(y_true.mean()), 6),
        "return_kind": None,
        "mean_return": None,
        "total_return": None,
    }
    if return_col and return_col in selected.columns:
        returns = selected[return_col].astype(float)
        result["return_kind"] = "raw" if return_col == "target_return_4h" else "unknown"
        if return_col == "target_return_4h":
            result["return_kind"] = "raw_not_fee_adjusted"
        result["mean_return"] = round(float(returns.mean()), 6)
        result["total_return"] = round(float(returns.sum()), 6)
    return result


def resolve_return_column(frame: pd.DataFrame) -> str | None:
    for candidate in ("target_return_4h", "future_return_4h"):
        if candidate in frame.columns and frame[candidate].notna().any():
            return candidate
    return None


def compute_split_metrics(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    top_pcts: list[float],
    return_col: str | None,
) -> dict[str, Any]:
    pos = int(y_true.sum())
    n = int(len(y_true))
    metrics: dict[str, Any] = {
        "row_count": n,
        "positive_count": pos,
        "positive_rate": round(pos / n, 6) if n else 0.0,
        "pr_auc": _safe_pr_auc(y_true, y_score),
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "return_column": return_col,
        "return_column_kind": "raw_not_fee_adjusted" if return_col == "target_return_4h" else None,
    }
    pct_labels = {
        0.01: "top_1_percent",
        0.02: "top_2_percent",
        0.05: "top_5_percent",
        1.0: "top_1_percent",
        2.0: "top_2_percent",
        5.0: "top_5_percent",
    }
    for pct in top_pcts:
        k_pct = pct * 100.0 if pct <= 1.0 else pct
        label = pct_labels.get(pct) or pct_labels.get(k_pct) or f"top_{int(k_pct)}_percent"
        prec = precision_at_top_k_with_count(y_true, y_score, k_pct)
        policy = return_metrics_for_top_k(frame, y_score, k_pct, return_col)
        metrics[f"precision_at_{label}"] = prec["precision"]
        metrics[f"trade_count_{label}"] = prec["trade_count"]
        metrics[f"win_rate_{label}"] = policy["win_rate"]
        if return_col:
            metrics[f"mean_{return_col}_{label}"] = policy["mean_return"]
            metrics[f"total_{return_col}_{label}"] = policy["total_return"]
    return metrics


def build_prediction_rows(
    frame: pd.DataFrame,
    *,
    target_name: str,
    y_true: pd.Series,
    y_score: np.ndarray,
    split: str,
) -> pd.DataFrame:
    rows = pd.DataFrame({
        "event_timestamp": frame.get("event_timestamp"),
        "symbol": frame.get("symbol"),
        "pair_address": frame.get("pair_address"),
        "target_name": target_name,
        "y_true": y_true.fillna(0).astype(int).values,
        "predicted_probability": y_score,
        "model_name": "tabicl_v2",
        "split": split,
    })
    for col in PREDICTION_META_COLUMNS:
        if col in frame.columns and col not in rows.columns:
            rows[col] = frame[col].values
    return rows


def get_cuda_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "cuda_available": False,
        "torch_version": None,
        "torch_cuda_version": None,
        "device_name": None,
    }
    try:
        import torch

        info["torch_version"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["torch_cuda_version"] = torch.version.cuda
        if info["cuda_available"]:
            info["device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def resolve_device(requested: str) -> str:
    info = get_cuda_info()
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not info["cuda_available"]:
            raise RuntimeError("CUDA requested via --device cuda but torch.cuda.is_available() is False.")
        return "cuda"
    if info["cuda_available"]:
        return "cuda"
    return "cpu"


def clear_cuda_cache() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except ImportError:
        pass


def is_cuda_oom(exc: BaseException) -> bool:
    name = type(exc).__name__
    if name == "OutOfMemoryError":
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda" in msg and "memory" in msg


def predict_proba_batched(
    classifier: Any,
    x: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    if len(x) == 0:
        return np.empty((0, 2), dtype=float)
    chunks: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        end = min(start + batch_size, len(x))
        chunks.append(classifier.predict_proba(x[start:end]))
    return np.vstack(chunks)


def _positive_class_probability(
    classifier: Any,
    x_infer: np.ndarray,
    *,
    batch_size: int,
    default_class: int | None = None,
) -> np.ndarray:
    """Return positive-class probabilities robustly for binary or single-column outputs."""
    if len(x_infer) == 0:
        return np.array([], dtype=float)

    proba = predict_proba_batched(classifier, x_infer, batch_size=batch_size)
    proba = np.asarray(proba, dtype=float)

    if proba.ndim == 1:
        return proba.astype(float)

    classes = np.asarray(getattr(classifier, "classes_", []))

    if proba.shape[1] == 1:
        if len(classes) == 1:
            try:
                return proba[:, 0].astype(float) if int(classes[0]) == 1 else np.zeros(len(x_infer), dtype=float)
            except (TypeError, ValueError):
                pass
        if default_class == 1:
            return proba[:, 0].astype(float)
        return np.zeros(len(x_infer), dtype=float)

    if len(classes) == proba.shape[1]:
        try:
            positive_cols = np.flatnonzero(classes.astype(int) == 1)
            if len(positive_cols) > 0:
                return proba[:, int(positive_cols[0])].astype(float)
        except (TypeError, ValueError):
            pass

    # Conventional binary-class fallback: column 1 is the positive probability.
    return proba[:, 1].astype(float)


def _constant_scores_for_single_class(
    y_context: np.ndarray,
    length: int,
) -> np.ndarray | None:
    """Return constant positive probability when context contains one class, else None."""
    y_context = np.asarray(y_context).astype(int).reshape(-1)
    if len(y_context) == 0:
        raise ValueError("TabICL context is empty; cannot fit classifier.")

    unique_classes = np.unique(y_context)
    if len(unique_classes) != 1:
        return None

    constant_score = 1.0 if int(unique_classes[0]) == 1 else 0.0
    return np.full(length, constant_score, dtype=float)


def _build_tabicl_classifier(
    *,
    device: str,
    batch_size: int,
    model_path: Path | None = None,
) -> Any:
    from tabicl import TabICLClassifier

    kwargs: dict[str, Any] = {
        "device": device,
        "batch_size": batch_size,
        "checkpoint_version": "tabicl-classifier-v2-20260212.ckpt",
        "allow_auto_download": True,
        "verbose": False,
        "n_estimators": 4,
    }
    if model_path is not None and model_path.is_file():
        kwargs["model_path"] = str(model_path)

    return TabICLClassifier(**kwargs)


def fit_and_predict_tabicl(
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    *,
    device: str,
    batch_size: int,
    model_path: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """
    Fit TabICLv2 on a static context and return positive-class probabilities.

    Dynamic/sampled contexts can occasionally contain only one class.
    In that case, do not fit TabICL; return deterministic constant scores:
    - 1.0 when the context contains only positives
    - 0.0 when the context contains only negatives
    """
    y_context = np.asarray(y_context).astype(int).reshape(-1)

    val_constant = _constant_scores_for_single_class(y_context, len(x_val))
    if val_constant is not None:
        test_constant = _constant_scores_for_single_class(y_context, len(x_test))
        return val_constant, test_constant, None

    classifier = _build_tabicl_classifier(
        device=device,
        batch_size=batch_size,
        model_path=model_path,
    )
    classifier.fit(x_context, y_context)

    val_scores = _positive_class_probability(
        classifier,
        x_val,
        batch_size=batch_size,
    )
    test_scores = _positive_class_probability(
        classifier,
        x_test,
        batch_size=batch_size,
    )
    return val_scores, test_scores, classifier


def fit_and_predict_query_batch(
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_query: np.ndarray,
    *,
    device: str,
    batch_size: int,
    model_path: Path | None = None,
) -> np.ndarray:
    """
    Fit TabICLv2 on a dynamic context and return positive-class probabilities
    for one query batch.

    This function is called repeatedly by dynamic/rolling context strategies,
    so it must be robust to single-class contexts and single-column proba output.
    """
    if len(x_query) == 0:
        return np.array([], dtype=float)

    y_context = np.asarray(y_context).astype(int).reshape(-1)

    constant_scores = _constant_scores_for_single_class(y_context, len(x_query))
    if constant_scores is not None:
        return constant_scores

    classifier = _build_tabicl_classifier(
        device=device,
        batch_size=batch_size,
        model_path=model_path,
    )
    classifier.fit(x_context, y_context)

    return _positive_class_probability(
        classifier,
        x_query,
        batch_size=batch_size,
    )


def predict_with_dynamic_context(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_split: np.ndarray,
    *,
    context_builder: Callable[..., np.ndarray],
    device: str,
    batch_size: int,
    model_path: Path | None = None,
    split_timestamps: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    if len(x_split) == 0:
        return np.array([], dtype=float), {"batches": 0}

    scores = np.zeros(len(x_split), dtype=float)
    batches = 0
    for start in range(0, len(x_split), batch_size):
        end = min(start + batch_size, len(x_split))
        batch_x = x_split[start:end]
        batch_ts = split_timestamps[start:end] if split_timestamps is not None else None
        if batch_ts is not None:
            ctx_indices = context_builder(batch_x, batch_ts)
        else:
            ctx_indices = context_builder(batch_x)
        if len(ctx_indices) == 0:
            raise RuntimeError("Context builder returned no train indices.")
        batch_scores = fit_and_predict_query_batch(
            x_train[ctx_indices],
            y_train[ctx_indices],
            batch_x,
            device=device,
            batch_size=batch_size,
            model_path=model_path,
        )
        scores[start:end] = batch_scores
        batches += 1
    return scores, {"batches": batches}


def run_tabicl_with_oom_retry(
    x_context: np.ndarray,
    y_context: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    *,
    device: str,
    batch_size: int,
    context_size: int,
    y_train_full: np.ndarray,
    max_train_context_rows: int,
    random_state: int = 42,
    model_path: Path | None = None,
    context_strategy: str | None = None,
    positive_context_ratio: float | None = None,
    nearest_neighbor_metric: str = "euclidean",
    feature_cols: list[str] | None = None,
    ensemble_count: int = 4,
    ensemble_context_size: int = 2048,
    train_timestamps: np.ndarray | None = None,
    val_timestamps: np.ndarray | None = None,
    test_timestamps: np.ndarray | None = None,
    knn_rolling_days: int | None = None,
    knn_min_context_rows: int | None = None,
    knn_expand_window: bool | str | None = None,
    knn_max_rolling_days: int | None = None,
    knn_time_decay_alpha: float | None = None,
    knn_cache_time_bucket: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not tabicl_available():
        raise ImportError(
            "tabicl is not installed. Run evaluation from .venv-tabicl after "
            "pip install -r requirements-tabicl.txt"
        )

    validate_context_strategy(context_strategy)
    effective_context_size = cap_context_size(context_size, max_train_context_rows, len(y_train_full))

    oom_retry_count = 0
    current_batch = batch_size
    current_context_size = effective_context_size
    last_error: Exception | None = None
    member_metrics: dict[str, Any] = {}

    while True:
        meta: dict[str, Any] = {
            "context_strategy": context_strategy,
            "context_size_used": int(current_context_size),
            "batch_size_used": int(current_batch),
            "max_train_context_rows": int(max_train_context_rows),
            "ensemble_count": int(ensemble_count),
            "ensemble_context_size": int(ensemble_context_size),
            "nearest_neighbor_metric": nearest_neighbor_metric,
            "positive_context_ratio": positive_context_ratio,
            "oom_retry_count": oom_retry_count,
            "ensemble_member_metrics": member_metrics,
        }

        try:
            if context_strategy == "ensemble_small_contexts":
                val_scores, test_scores, ensemble_meta = _run_ensemble_small_contexts(
                    x_context,
                    y_train_full,
                    x_val,
                    x_test,
                    device=device,
                    batch_size=current_batch,
                    ensemble_count=ensemble_count,
                    ensemble_context_size=min(
                        ensemble_context_size,
                        cap_context_size(ensemble_context_size, max_train_context_rows, len(y_train_full)),
                    ),
                    max_train_context_rows=max_train_context_rows,
                    model_path=model_path,
                    positive_context_ratio=positive_context_ratio,
                    nearest_neighbor_metric=nearest_neighbor_metric,
                    feature_cols=feature_cols or [],
                    random_state=random_state,
                )
                meta.update(ensemble_meta)
                meta["context_positive_count"] = None
                meta["context_negative_count"] = None
                meta["context_positive_rate"] = None
                return val_scores, test_scores, meta

            if context_strategy in {"nearest_neighbors_context", "whale_wave_context"}:
                use_rolling = (
                    context_strategy == "nearest_neighbors_context"
                    and rolling_knn_enabled(knn_rolling_days)
                )
                if use_rolling:
                    if train_timestamps is None or val_timestamps is None or test_timestamps is None:
                        raise ValueError(
                            "Rolling KNN requires event_timestamp on train/validation/test splits."
                        )
                    rolling_config = build_rolling_knn_config(
                        knn_rolling_days=knn_rolling_days,
                        knn_min_context_rows=knn_min_context_rows,
                        knn_expand_window=knn_expand_window,
                        knn_max_rolling_days=knn_max_rolling_days,
                        knn_time_decay_alpha=knn_time_decay_alpha,
                        knn_cache_time_bucket=knn_cache_time_bucket,
                    )
                    rolling_selector = RollingKnnContextSelector(
                        x_context,
                        y_train_full,
                        train_timestamps,
                        config=rolling_config,
                        metric=nearest_neighbor_metric,
                    )

                    def _rolling_builder(batch_x: np.ndarray, batch_ts: np.ndarray) -> np.ndarray:
                        return rolling_selector.build_global_context_indices(
                            batch_x,
                            batch_ts,
                            context_size=current_context_size,
                            exclude_zero_distance=False,
                        )

                    val_scores, val_batch_meta = predict_with_dynamic_context(
                        x_context,
                        y_train_full,
                        x_val,
                        context_builder=_rolling_builder,
                        device=device,
                        batch_size=current_batch,
                        model_path=model_path,
                        split_timestamps=val_timestamps,
                    )
                    test_scores, test_batch_meta = predict_with_dynamic_context(
                        x_context,
                        y_train_full,
                        x_test,
                        context_builder=_rolling_builder,
                        device=device,
                        batch_size=current_batch,
                        model_path=model_path,
                        split_timestamps=test_timestamps,
                    )
                    meta.update(rolling_selector.diagnostics())
                    meta["nearest_neighbor_fit_count"] = rolling_selector.knn_index_fit_count
                    meta["dynamic_context_batches"] = {
                        "validation": val_batch_meta["batches"],
                        "test": test_batch_meta["batches"],
                    }
                elif context_strategy == "whale_wave_context" and feature_cols:
                    col_idx = whale_wave_feature_indices(feature_cols)
                    nn_index = NearestNeighborContextIndex(
                        x_context,
                        y_train_full,
                        metric=nearest_neighbor_metric,
                        feature_column_indices=col_idx if len(col_idx) > 0 else None,
                    )
                    meta["nearest_neighbor_fit_count"] = nn_index.fit_count

                    def _builder(batch_x: np.ndarray) -> np.ndarray:
                        return nn_index.build_context_indices(
                            batch_x,
                            context_size=current_context_size,
                            exclude_zero_distance=False,
                        )

                    val_scores, val_batch_meta = predict_with_dynamic_context(
                        x_context,
                        y_train_full,
                        x_val,
                        context_builder=_builder,
                        device=device,
                        batch_size=current_batch,
                        model_path=model_path,
                    )
                    test_scores, test_batch_meta = predict_with_dynamic_context(
                        x_context,
                        y_train_full,
                        x_test,
                        context_builder=_builder,
                        device=device,
                        batch_size=current_batch,
                        model_path=model_path,
                    )
                    meta["dynamic_context_batches"] = {
                        "validation": val_batch_meta["batches"],
                        "test": test_batch_meta["batches"],
                    }
                else:
                    nn_index = NearestNeighborContextIndex(
                        x_context,
                        y_train_full,
                        metric=nearest_neighbor_metric,
                    )
                    meta["nearest_neighbor_fit_count"] = nn_index.fit_count

                    def _builder(batch_x: np.ndarray) -> np.ndarray:
                        return nn_index.build_context_indices(
                            batch_x,
                            context_size=current_context_size,
                            exclude_zero_distance=False,
                        )

                    val_scores, val_batch_meta = predict_with_dynamic_context(
                        x_context,
                        y_train_full,
                        x_val,
                        context_builder=_builder,
                        device=device,
                        batch_size=current_batch,
                        model_path=model_path,
                    )
                    test_scores, test_batch_meta = predict_with_dynamic_context(
                        x_context,
                        y_train_full,
                        x_test,
                        context_builder=_builder,
                        device=device,
                        batch_size=current_batch,
                        model_path=model_path,
                    )
                    meta["dynamic_context_batches"] = {
                        "validation": val_batch_meta["batches"],
                        "test": test_batch_meta["batches"],
                    }
                meta["context_positive_count"] = None
                meta["context_negative_count"] = None
                meta["context_positive_rate"] = None
                return val_scores, test_scores, meta

            if context_strategy is None:
                ctx_indices = sample_context_indices(
                    y_train_full,
                    context_size=current_context_size,
                    random_state=random_state,
                )
            else:
                ctx_indices = build_static_context_indices(
                    context_strategy,
                    y_train_full,
                    context_size=current_context_size,
                    max_train_context_rows=max_train_context_rows,
                    positive_context_ratio=positive_context_ratio,
                    random_state=random_state,
                )
            ctx_indices = ctx_indices[
                : cap_context_size(current_context_size, max_train_context_rows, len(y_train_full))
            ]
            if len(ctx_indices) == 0:
                raise RuntimeError("No train rows available for TabICL context sampling.")

            x_ctx = x_context[ctx_indices]
            y_ctx = y_train_full[ctx_indices]
            pos_count = int(y_ctx.sum())
            neg_count = int(len(y_ctx) - pos_count)
            meta.update({
                "context_size_used": int(len(ctx_indices)),
                "context_positive_count": pos_count,
                "context_negative_count": neg_count,
                "context_positive_rate": round(pos_count / len(y_ctx), 6) if len(y_ctx) else 0.0,
            })

            val_scores, test_scores, _ = fit_and_predict_tabicl(
                x_ctx,
                y_ctx,
                x_val,
                x_test,
                device=device,
                batch_size=current_batch,
                model_path=model_path,
            )
            return val_scores, test_scores, meta
        except Exception as exc:
            last_error = exc
            if not is_cuda_oom(exc):
                raise
            clear_cuda_cache()
            oom_retry_count += 1
            meta["oom_retry_count"] = oom_retry_count
            if current_batch > 32:
                current_batch = max(32, current_batch // 2)
                continue
            if current_context_size > 128:
                current_context_size = max(128, current_context_size // 2)
                continue
            raise RuntimeError(
                "CUDA out of memory persisted after reducing batch_size and context_size. "
                f"Last batch_size={current_batch}, context_size={current_context_size}. "
                f"Original error: {last_error}"
            ) from exc


def _run_ensemble_small_contexts(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    *,
    device: str,
    batch_size: int,
    ensemble_count: int,
    ensemble_context_size: int,
    max_train_context_rows: int,
    model_path: Path | None,
    positive_context_ratio: float | None,
    nearest_neighbor_metric: str,
    feature_cols: list[str],
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    members = list(ENSEMBLE_MEMBER_STRATEGIES[: max(1, ensemble_count)])
    while len(members) < ensemble_count:
        members.append(members[-1])

    val_member_scores: list[np.ndarray] = []
    test_member_scores: list[np.ndarray] = []
    member_details: dict[str, Any] = {}

    for member in members[:ensemble_count]:
        member_val, member_test, member_meta = run_tabicl_with_oom_retry(
            x_train,
            y_train,
            x_val,
            x_test,
            device=device,
            batch_size=batch_size,
            context_size=ensemble_context_size,
            y_train_full=y_train,
            max_train_context_rows=max_train_context_rows,
            random_state=random_state,
            model_path=model_path,
            context_strategy=member,
            positive_context_ratio=positive_context_ratio,
            nearest_neighbor_metric=nearest_neighbor_metric,
            feature_cols=feature_cols,
            ensemble_count=ensemble_count,
            ensemble_context_size=ensemble_context_size,
        )
        val_member_scores.append(member_val)
        test_member_scores.append(member_test)
        member_details[member] = {
            "context_size_used": member_meta.get("context_size_used"),
            "batch_size_used": member_meta.get("batch_size_used"),
            "context_positive_rate": member_meta.get("context_positive_rate"),
        }

    val_avg = np.mean(np.vstack(val_member_scores), axis=0)
    test_avg = np.mean(np.vstack(test_member_scores), axis=0)
    return val_avg, test_avg, {
        "ensemble_members": members[:ensemble_count],
        "ensemble_member_details": member_details,
        "context_size_used": int(ensemble_context_size),
        "batch_size_used": int(batch_size),
    }


def compute_train_self_neighbor_exclusion(
    nn_index: NearestNeighborContextIndex,
    x_train: np.ndarray,
    *,
    context_size: int,
) -> dict[str, Any]:
    """Diagnostics helper: verify zero-distance neighbors are excluded."""
    ctx = nn_index.build_context_indices(
        x_train[: min(32, len(x_train))],
        context_size=context_size,
        exclude_zero_distance=True,
    )
    return {
        "sample_context_size": int(len(ctx)),
        "exclude_zero_distance": True,
    }


def extract_sweep_metrics(report: dict[str, Any]) -> dict[str, Any]:
    val_m = (report.get("tabicl_metrics") or {}).get("validation") or {}
    test_m = (report.get("tabicl_metrics") or {}).get("test") or {}
    row: dict[str, Any] = {
        "strategy": report.get("context_strategy") or "default",
        "context_size_used": report.get("context_size_used"),
        "batch_size_used": report.get("batch_size_used"),
        "ensemble_count": report.get("ensemble_count"),
        "ensemble_context_size": report.get("ensemble_context_size"),
        "max_features": report.get("max_features"),
        "feature_count": report.get("feature_count"),
        "feature_names_path": report.get("feature_names_path"),
        "total_rows_used": report.get("total_rows_used"),
        "train_row_count": report.get("train_row_count"),
        "validation_row_count": report.get("validation_row_count"),
        "test_row_count": report.get("test_row_count"),
        "validation_positive_rate": val_m.get("positive_rate"),
        "test_positive_rate": test_m.get("positive_rate"),
        "validation_pr_auc": val_m.get("pr_auc"),
        "test_pr_auc": test_m.get("pr_auc"),
        "validation_roc_auc": val_m.get("roc_auc"),
        "test_roc_auc": test_m.get("roc_auc"),
        "validation_precision_at_top_1_percent": val_m.get("precision_at_top_1_percent"),
        "test_precision_at_top_1_percent": test_m.get("precision_at_top_1_percent"),
        "validation_precision_at_top_2_percent": val_m.get("precision_at_top_2_percent"),
        "test_precision_at_top_2_percent": test_m.get("precision_at_top_2_percent"),
        "validation_precision_at_top_5_percent": val_m.get("precision_at_top_5_percent"),
        "test_precision_at_top_5_percent": test_m.get("precision_at_top_5_percent"),
        "validation_total_target_return_4h_top_1_percent": val_m.get("total_target_return_4h_top_1_percent"),
        "test_total_target_return_4h_top_1_percent": test_m.get("total_target_return_4h_top_1_percent"),
        "validation_total_target_return_4h_top_2_percent": val_m.get("total_target_return_4h_top_2_percent"),
        "test_total_target_return_4h_top_2_percent": test_m.get("total_target_return_4h_top_2_percent"),
        "validation_total_target_return_4h_top_5_percent": val_m.get("total_target_return_4h_top_5_percent"),
        "test_total_target_return_4h_top_5_percent": test_m.get("total_target_return_4h_top_5_percent"),
        "return_column_kind": report.get("return_column_kind"),
        "oom_retry_count": report.get("oom_retry_count"),
        "runtime_seconds": report.get("runtime_seconds"),
        "device_used": report.get("device_used"),
        "cuda_available": report.get("cuda_available"),
    }
    return row


def rank_strategy_sweep_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    def sort_key(row: dict[str, Any]) -> tuple:
        def metric(name: str) -> float:
            value = row.get(name)
            if value is None:
                return float("-inf")
            return float(value)

        return (
            metric("validation_precision_at_top_1_percent"),
            metric("validation_total_target_return_4h_top_1_percent"),
            metric("validation_pr_auc"),
        )

    ranked = sorted(rows, key=sort_key, reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["ranking_basis"] = "validation_precision_at_top_1_percent"
    return ranked


def load_rf_baseline_comparison(
    target: str,
    top_pcts: list[float],
) -> dict[str, Any]:
    comparison: dict[str, Any] = {
        "available": False,
        "reason": None,
        "target_queried": target,
        "rf_target_names_tried": [],
        "metrics": {},
    }
    if not RF_VALIDATION_PREDICTIONS.is_file() or not RF_TEST_PREDICTIONS.is_file():
        comparison["reason"] = "RF prediction parquet files missing; run train_baseline_model.py first."
        return comparison

    val_preds = pd.read_parquet(RF_VALIDATION_PREDICTIONS)
    test_preds = pd.read_parquet(RF_TEST_PREDICTIONS)

    target_aliases = [target, "label_profitable_after_fees_4h", "target_profitable_4h"]
    seen: set[str] = set()
    target_names: list[str] = []
    for name in target_aliases:
        if name not in seen:
            seen.add(name)
            target_names.append(name)
    comparison["rf_target_names_tried"] = target_names

    def _filter(preds: pd.DataFrame) -> pd.DataFrame:
        for name in target_names:
            subset = preds[(preds["target_name"] == name) & (preds["model_name"] == "random_forest")]
            if not subset.empty:
                return subset.copy()
        return pd.DataFrame()

    val_rf = _filter(val_preds)
    test_rf = _filter(test_preds)
    if val_rf.empty and test_rf.empty:
        comparison["reason"] = f"No random_forest predictions found for targets {target_names}."
        return comparison

    comparison["available"] = True
    for split_name, subset in (("validation", val_rf), ("test", test_rf)):
        if subset.empty:
            comparison["metrics"][split_name] = {"available": False}
            continue
        y_true = subset["y_true"].fillna(0).astype(int).to_numpy()
        y_score = subset["predicted_probability"].astype(float).to_numpy()
        return_col = resolve_return_column(subset)
        comparison["metrics"][split_name] = {
            "available": True,
            "model_name": "random_forest",
            "target_name": subset["target_name"].iloc[0],
            **compute_split_metrics(subset, y_true, y_score, top_pcts=top_pcts, return_col=return_col),
        }

    if RF_METRICS_PATH.is_file():
        with open(RF_METRICS_PATH, encoding="utf-8") as handle:
            baseline = json.load(handle)
        for canonical in ("label_profitable_after_fees_4h", target):
            entry = (baseline.get("best_model_by_target") or {}).get(canonical)
            if entry:
                comparison["baseline_metrics_best_model"] = entry
                break

    return comparison


def compare_tabicl_vs_rf(
    tabicl_metrics: dict[str, Any],
    rf_comparison: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rf_available": rf_comparison.get("available", False),
        "verdict": "inconclusive",
        "notes": [],
    }
    if not rf_comparison.get("available"):
        result["notes"].append(rf_comparison.get("reason") or "RF comparison unavailable.")
        return result

    wins = 0
    losses = 0
    ties = 0
    for split in ("validation", "test"):
        rf_split = (rf_comparison.get("metrics") or {}).get(split) or {}
        tab_split = tabicl_metrics.get(split) or {}
        if not rf_split.get("available"):
            continue
        for metric in (
            "pr_auc",
            "precision_at_top_1_percent",
            "precision_at_top_2_percent",
            "precision_at_top_5_percent",
        ):
            t_val = tab_split.get(metric)
            r_val = rf_split.get(metric)
            if t_val is None or r_val is None:
                continue
            if t_val > r_val:
                wins += 1
            elif t_val < r_val:
                losses += 1
            else:
                ties += 1

    if wins > losses:
        result["verdict"] = "tabicl_beats_rf"
    elif losses > wins:
        result["verdict"] = "tabicl_hurts_vs_rf"
    else:
        result["verdict"] = "inconclusive"
    result["metric_wins"] = wins
    result["metric_losses"] = losses
    result["metric_ties"] = ties
    return result


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
        "venv_expected": ".venv-tabicl",
    }


def evaluate_tabicl_v2(
    *,
    dataset_path: Path | None = None,
    target: str = DEFAULT_TARGET,
    max_rows: int | None = None,
    max_features: int | None = None,
    validation_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_VAL_FRAC,
    top_pcts: list[float] | None = None,
    device: str = "auto",
    output_dir: Path | None = None,
    context_size: int = 1024,
    batch_size: int = 256,
    max_train_context_rows: int = 1024,
    scaler: str = "standard",
    model_path: Path | None = None,
    partial_evaluation_reason: str | None = None,
    context_strategy: str | None = None,
    output_suffix: str | None = None,
    ensemble_count: int = 4,
    ensemble_context_size: int = 2048,
    positive_context_ratio: float | None = None,
    nearest_neighbor_metric: str = "euclidean",
    overwrite_outputs: bool = False,
    knn_rolling_days: int | None = None,
    knn_min_context_rows: int | None = None,
    knn_expand_window: bool | str | None = None,
    knn_max_rolling_days: int | None = None,
    knn_time_decay_alpha: float | None = None,
    knn_cache_time_bucket: str | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    validate_context_strategy(context_strategy)
    output_label = resolve_output_label(context_strategy, output_suffix)

    dataset_path = dataset_path or DEFAULT_DATASET_PATH
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    top_pcts = top_pcts or [0.01, 0.02, 0.05]
    train_frac = 1.0 - validation_frac - test_frac
    if train_frac <= 0:
        raise ValueError("validation_frac + test_frac must be less than 1.0")

    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    frame = pd.read_parquet(dataset_path)
    if target not in frame.columns:
        raise ValueError(f"Target column missing: {target}")

    timestamp_col = "event_timestamp" if "event_timestamp" in frame.columns else None
    if timestamp_col:
        frame[timestamp_col] = pd.to_datetime(frame[timestamp_col], utc=True, errors="coerce")
        frame = frame[frame[timestamp_col].notna()].copy()
        frame = frame.sort_values(timestamp_col).reset_index(drop=True)

    if max_rows is not None and len(frame) > max_rows:
        frame = frame.iloc[-max_rows:].reset_index(drop=True)

    numeric_features, excluded = select_tabicl_feature_columns(frame, target=target)
    numeric_features = limit_features_by_variance(frame, numeric_features, max_features)

    forbidden_features = [
        col for col in numeric_features
        if col == target or col.lower().startswith("target_") or _contains_leakage(col)
    ]
    if forbidden_features:
        raise RuntimeError(
            "Leakage features selected for TabICL: "
            + ", ".join(forbidden_features)
        )

    train_df, val_df, test_df = chronological_split(
        frame,
        train_frac=train_frac,
        val_frac=validation_frac,
        timestamp_col=timestamp_col or "event_timestamp",
    )
    assert_chronological_splits(train_df, val_df, test_df, timestamp_col=timestamp_col or "event_timestamp")

    y_train = train_df[target].fillna(0).astype(int)
    y_val = val_df[target].fillna(0).astype(int)
    y_test = test_df[target].fillna(0).astype(int)

    if y_train.nunique() < 2:
        raise ValueError(f"Target {target} has fewer than two classes in train split.")

    preprocessor = TrainOnlyPreprocessor(scaler=scaler)
    preprocessor.fit(train_df, numeric_features)
    x_train = preprocessor.transform(train_df)
    x_val = preprocessor.transform(val_df)
    x_test = preprocessor.transform(test_df)

    train_timestamps = (
        train_df[timestamp_col].to_numpy(dtype="datetime64[ns]")
        if timestamp_col
        else None
    )
    val_timestamps = (
        val_df[timestamp_col].to_numpy(dtype="datetime64[ns]")
        if timestamp_col
        else None
    )
    test_timestamps = (
        test_df[timestamp_col].to_numpy(dtype="datetime64[ns]")
        if timestamp_col
        else None
    )

    resolved_device = resolve_device(device)
    cuda_info = get_cuda_info()

    effective_context_size = cap_context_size(context_size, max_train_context_rows, len(train_df))

    val_scores, test_scores, run_meta = run_tabicl_with_oom_retry(
        x_train,
        y_train.to_numpy(),
        x_val,
        x_test,
        device=resolved_device,
        batch_size=batch_size,
        context_size=effective_context_size,
        y_train_full=y_train.to_numpy(),
        max_train_context_rows=max_train_context_rows,
        model_path=model_path,
        context_strategy=context_strategy,
        positive_context_ratio=positive_context_ratio,
        nearest_neighbor_metric=nearest_neighbor_metric,
        feature_cols=numeric_features,
        ensemble_count=ensemble_count,
        ensemble_context_size=ensemble_context_size,
        train_timestamps=train_timestamps,
        val_timestamps=val_timestamps,
        test_timestamps=test_timestamps,
        knn_rolling_days=knn_rolling_days,
        knn_min_context_rows=knn_min_context_rows,
        knn_expand_window=knn_expand_window,
        knn_max_rolling_days=knn_max_rolling_days,
        knn_time_decay_alpha=knn_time_decay_alpha,
        knn_cache_time_bucket=knn_cache_time_bucket,
    )

    return_col = resolve_return_column(frame)
    val_metrics = compute_split_metrics(
        val_df.assign(y_true=y_val),
        y_val.to_numpy(),
        val_scores,
        top_pcts=top_pcts,
        return_col=return_col,
    )
    test_metrics = compute_split_metrics(
        test_df.assign(y_true=y_test),
        y_test.to_numpy(),
        test_scores,
        top_pcts=top_pcts,
        return_col=return_col,
    )

    rf_comparison = load_rf_baseline_comparison(target, top_pcts)
    tabicl_metrics = {"validation": val_metrics, "test": test_metrics}
    vs_rf = compare_tabicl_vs_rf(tabicl_metrics, rf_comparison)

    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    backtest_dir = output_dir / "policy_backtests"
    backtest_dir.mkdir(parents=True, exist_ok=True)

    val_pred_path, test_pred_path, report_path, features_path = prediction_output_paths(
        output_dir,
        output_label,
    )
    for path in (val_pred_path, test_pred_path, report_path):
        if path.exists() and not overwrite_outputs:
            raise FileExistsError(
                f"Refusing to overwrite existing output: {path}. "
                "Use a different --output-suffix or --context-strategy."
            )

    val_preds = build_prediction_rows(
        val_df, target_name=target, y_true=y_val, y_score=val_scores, split="validation",
    )
    test_preds = build_prediction_rows(
        test_df, target_name=target, y_true=y_test, y_score=test_scores, split="test",
    )
    val_preds.to_parquet(val_pred_path, index=False)
    test_preds.to_parquet(test_pred_path, index=False)

    feature_names_payload = {
        "feature_names_used": numeric_features,
        "feature_count": len(numeric_features),
        "max_features": max_features,
        "context_strategy": context_strategy,
        "output_label": output_label,
    }
    if features_path is not None:
        with open(features_path, "w", encoding="utf-8") as handle:
            json.dump(feature_names_payload, handle, indent=2)
        feature_names_path = str(features_path)
    else:
        feature_names_path = None

    full_evaluation, partial_reason = resolve_full_evaluation(
        max_rows=max_rows,
        partial_evaluation_reason=partial_evaluation_reason,
    )
    runtime_seconds = round(time.perf_counter() - started, 3)

    report: dict[str, Any] = {
        **base_report_flags(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_path),
        "target": target,
        "context_strategy": context_strategy,
        "output_label": output_label,
        "rows_total": int(len(frame)),
        "total_rows_used": int(len(frame)),
        "train_row_count": int(len(train_df)),
        "validation_row_count": int(len(val_df)),
        "test_row_count": int(len(test_df)),
        "rows_train": int(len(train_df)),
        "rows_validation": int(len(val_df)),
        "rows_test": int(len(test_df)),
        "features_used_count": len(numeric_features),
        "feature_count": len(numeric_features),
        "max_features": max_features,
        "feature_names_used": numeric_features,
        "feature_names_path": feature_names_path,
        "features_excluded_count": len(excluded),
        "numeric_features": numeric_features,
        "excluded_features": excluded,
        "validation_frac": validation_frac,
        "test_frac": test_frac,
        "top_pcts": top_pcts,
        "device_requested": device,
        "device_used": resolved_device,
        "cuda_available": cuda_info["cuda_available"],
        "torch_version": cuda_info["torch_version"],
        "torch_cuda_version": cuda_info["torch_cuda_version"],
        "device_name": cuda_info["device_name"],
        "scaler_used": scaler,
        "max_train_context_rows": max_train_context_rows,
        "ensemble_count": ensemble_count,
        "ensemble_context_size": ensemble_context_size,
        "positive_context_ratio": positive_context_ratio,
        "nearest_neighbor_metric": nearest_neighbor_metric,
        "return_column": return_col,
        "return_column_kind": "raw_not_fee_adjusted" if return_col == "target_return_4h" else None,
        "full_evaluation": full_evaluation,
        "partial_evaluation_reason": partial_reason,
        "runtime_seconds": runtime_seconds,
        "tabicl_metrics": tabicl_metrics,
        "rf_baseline_comparison": rf_comparison,
        "tabicl_vs_rf": vs_rf,
        "output_files": {
            "validation_predictions": str(val_pred_path),
            "test_predictions": str(test_pred_path),
            "report_json": str(report_path),
            "feature_names_json": feature_names_path,
        },
        **run_meta,
    }

    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)

    return report


SWEEP_STRATEGIES = tuple(sorted(CONTEXT_STRATEGIES))


def run_context_strategy_sweep(
    *,
    strategies: list[str] | None = None,
    dataset_path: Path | None = None,
    target: str = DEFAULT_TARGET,
    max_rows: int | None = None,
    max_features: int | None = None,
    validation_frac: float = DEFAULT_VAL_FRAC,
    test_frac: float = DEFAULT_VAL_FRAC,
    top_pcts: list[float] | None = None,
    device: str = "auto",
    output_dir: Path | None = None,
    context_size: int = 1024,
    batch_size: int = 256,
    max_train_context_rows: int = 1024,
    scaler: str = "standard",
    model_path: Path | None = None,
    partial_evaluation_reason: str | None = None,
    ensemble_count: int = 4,
    ensemble_context_size: int = 2048,
    positive_context_ratio: float | None = None,
    nearest_neighbor_metric: str = "euclidean",
    overwrite_outputs: bool = False,
    knn_rolling_days: int | None = None,
    knn_min_context_rows: int | None = None,
    knn_expand_window: bool | str | None = None,
    knn_max_rolling_days: int | None = None,
    knn_time_decay_alpha: float | None = None,
    knn_cache_time_bucket: str | None = None,
) -> dict[str, Any]:
    output_dir = output_dir or DEFAULT_OUTPUT_DIR
    strategies = strategies or list(SWEEP_STRATEGIES)
    for strategy in strategies:
        validate_context_strategy(strategy)

    rows: list[dict[str, Any]] = []
    strategy_reports: dict[str, Any] = {}
    for strategy in strategies:
        report = evaluate_tabicl_v2(
            dataset_path=dataset_path,
            target=target,
            max_rows=max_rows,
            max_features=max_features,
            validation_frac=validation_frac,
            test_frac=test_frac,
            top_pcts=top_pcts,
            device=device,
            output_dir=output_dir,
            context_size=context_size,
            batch_size=batch_size,
            max_train_context_rows=max_train_context_rows,
            scaler=scaler,
            model_path=model_path,
            partial_evaluation_reason=partial_evaluation_reason,
            context_strategy=strategy,
            output_suffix=strategy,
            ensemble_count=ensemble_count,
            ensemble_context_size=ensemble_context_size,
            positive_context_ratio=positive_context_ratio,
            nearest_neighbor_metric=nearest_neighbor_metric,
            overwrite_outputs=overwrite_outputs,
            knn_rolling_days=knn_rolling_days,
            knn_min_context_rows=knn_min_context_rows,
            knn_expand_window=knn_expand_window,
            knn_max_rolling_days=knn_max_rolling_days,
            knn_time_decay_alpha=knn_time_decay_alpha,
            knn_cache_time_bucket=knn_cache_time_bucket,
        )
        row = extract_sweep_metrics(report)
        rows.append(row)
        strategy_reports[strategy] = {
            "report_path": report["output_files"]["report_json"],
            "tabicl_vs_rf": report.get("tabicl_vs_rf"),
        }

    ranked_rows = rank_strategy_sweep_rows(rows)
    sweep_payload = {
        **base_report_flags(),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "strategies_evaluated": strategies,
        "ranking_basis": "validation_precision_at_top_1_percent",
        "secondary_ranking": [
            "validation_total_target_return_4h_top_1_percent",
            "validation_pr_auc",
        ],
        "results": ranked_rows,
        "strategy_reports": strategy_reports,
        "best_strategy": ranked_rows[0]["strategy"] if ranked_rows else None,
    }

    backtest_dir = output_dir / "policy_backtests"
    backtest_dir.mkdir(parents=True, exist_ok=True)
    json_path = backtest_dir / "tabicl_v2_context_strategy_sweep.json"
    csv_path = backtest_dir / "tabicl_v2_context_strategy_sweep.csv"

    if json_path.exists() and not overwrite_outputs:
        raise FileExistsError(
            f"Refusing to overwrite existing sweep output: {json_path}"
        )

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(sweep_payload, handle, indent=2, default=str)

    pd.DataFrame(ranked_rows).to_csv(csv_path, index=False)
    sweep_payload["output_files"] = {
        "sweep_json": str(json_path),
        "sweep_csv": str(csv_path),
    }
    return sweep_payload
