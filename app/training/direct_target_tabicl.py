"""Phase E5 — direct-target TabICL / TabICLv2 offline evaluation infrastructure."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import random
import sys
import uuid
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from app.artifacts.hash_utils import sha256_hex
from app.artifacts.registry import detect_project_root, load_registry
from app.training.direct_target_ids import (
    DEFAULT_EXIT_POLICIES,
    DEFAULT_FILTERS,
    DEFAULT_HORIZONS,
)
from app.training.direct_target_xgb_rf import (
    CANONICAL_TARGET,
    DatasetDescriptor,
    apply_deterministic_row_limit,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    build_feature_columns,
    build_feature_matrix,
    derive_valid_label_mask,
    discover_direct_target_datasets,
    evaluate_policy_grid as _evaluate_policy_grid_e4,
    feature_columns_hash,
    filter_descriptors,
    load_dataset,
    load_e3_manifest,
    precision_at_top_k_with_count,
    rank_validation_policies,
    apply_validation_policies_to_test,
    resolve_eval_return_column,
    select_top_with_pair_cap,
    summarize_policy_selection,
    validate_split_column,
)
from app.training.tabicl_v2_eval import (
    CONTEXT_STRATEGIES,
    TrainOnlyPreprocessor,
    cap_context_size,
    clear_cuda_cache,
    get_cuda_info,
    is_cuda_oom,
    resolve_device,
    run_tabicl_with_oom_retry,
    tabicl_available,
    validate_context_strategy,
)

PHASE = "E5"
BRANCH_NAME = "phase_e5_direct_target_tabicl"
TAB_MODEL_NAME = "TABICLv2"
TAB_MODEL_VERSION = "tabicl-classifier-v2-20260212.ckpt"

STRICT_TARGET_MAP: dict[Any, int] = {
    "True": 1,
    "False": 0,
    "true": 1,
    "false": 0,
    "TRUE": 1,
    "FALSE": 0,
    1: 1,
    0: 0,
    True: 1,
    False: 0,
    1.0: 1,
    0.0: 0,
}

ACCEPTABLE_TARGET_ALIASES = (
    "target_net_profitable_after_exit_policy",
    "target_net_profitable_after_exit",
)

DEFAULT_CONTEXT_STRATEGIES = ("stratified_recent",)
FOCUSED_FILTERS = ("LIQ_5K_HIGH_ACTIVITY", "NO_WHALE_FILTER")
FOCUSED_HORIZONS = ("1h", "4h", "8h", "24h")
FOCUSED_EXIT_POLICIES = (
    "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
    "TP20308_SL075_FEE0308_TIME_BY_HORIZON",
)

E5_TOP_PCTS = (0.5, 1.0, 2.0, 5.0)
E5_PAIR_CAPS: tuple[int, ...] = (10, 25, 50)
CONSENSUS_TOP_PCTS = (0.5, 1.0, 2.0, 5.0)

SMOKE_DEFAULT_MAX_ROWS = 200
SMOKE_DEFAULT_MAX_CONTEXT_SIZE = 75
SMOKE_DEFAULT_QUERY_BATCH_SIZE = 32
DEFAULT_MAX_WORKERS = 1

IDENTITY_COLUMNS_PRESERVE = (
    "candidate_id",
    "candidate_policy_id",
    "target_row_id",
    "pair_address",
    "symbol",
    "event_timestamp",
    "split",
    "filter",
    "horizon",
    "exit_policy_id",
    "tp_ratio",
    "sl_ratio",
    "time_stop_minutes",
    "round_trip_fee_pct",
)

JOIN_KEY_COLUMNS = ("target_row_id", "candidate_policy_id", "candidate_id")
PRIMARY_JOIN_KEY = "target_row_id"

CONSENSUS_TIER_LABELS = (
    "TAB_XGB_RF_ALL3",
    "TAB_RF_ONLY",
    "TAB_XGB_ONLY",
    "XGB_RF_ONLY",
    "TAB_ONLY",
    "XGB_ONLY",
    "RF_ONLY",
    "NONE",
    "UNKNOWN",
)

E4A_REQUIRED_PATTERNS = {
    "xgb_val_predictions": "predictions/direct_target_predictions_validation_XGB_*.parquet",
    "xgb_test_predictions": "predictions/direct_target_predictions_test_XGB_*.parquet",
    "rf_val_predictions": "predictions/direct_target_predictions_validation_RF_*.parquet",
    "rf_test_predictions": "predictions/direct_target_predictions_test_RF_*.parquet",
    "xgb_metrics": "metrics/direct_target_metrics_XGB_*.json",
    "rf_metrics": "metrics/direct_target_metrics_RF_*.json",
    "policy_grid": "policy_evaluation/direct_target_policy_grid_xgb_rf.csv",
    "validation_selected": (
        "policy_evaluation/validation_selected_policies_direct_target_xgb_rf_applied_to_test.csv"
    ),
    "manifest": "reports/phase_e4_manifest.json",
}

CONTEXT_STRATEGY_VERSION = "tabicl_v2_eval.py:2026-06"


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def strict_coerce_binary_target(series: pd.Series) -> tuple[pd.Series, list[str]]:
    """Deterministic strict target coercion; returns coerced series and error tokens."""
    errors: list[str] = []
    out = pd.Series(index=series.index, dtype="float64")
    for idx, raw in series.items():
        if pd.isna(raw):
            out.loc[idx] = np.nan
            continue
        if isinstance(raw, str):
            key = raw.strip()
        else:
            key = raw
        if key in STRICT_TARGET_MAP:
            out.loc[idx] = float(STRICT_TARGET_MAP[key])
        else:
            errors.append(f"INVALID_TARGET_COERCION_ERROR:{raw!r}")
            out.loc[idx] = np.nan
    return out, errors


def target_sanity_check(series: pd.Series) -> tuple[bool, dict[str, Any]]:
    non_null = series.dropna()
    invalid = non_null[~non_null.isin([0, 1, 0.0, 1.0])]
    value_counts = non_null.value_counts().to_dict()
    detail = {
        "valid": invalid.empty,
        "invalid_count": int(len(invalid)),
        "invalid_values": sorted({str(v) for v in invalid.unique()}),
        "value_counts": {str(k): int(v) for k, v in value_counts.items()},
    }
    if not invalid.empty:
        detail["error"] = "INVALID_TARGET_VALUES"
    return invalid.empty, detail


class E5IncrementalAuditLogger:
    """Append-only JSONL audit logger with line buffering and fsync."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, *, fsync: bool = True, **fields: Any) -> None:
        payload = {
            "created_at_utc": utc_now_iso(),
            "event_type": event_type,
            "phase": PHASE,
            "branch_name": BRANCH_NAME,
            **fields,
        }
        with self.path.open("a", encoding="utf-8", buffering=1) as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            if fsync:
                os.fsync(handle.fileno())


class E5TargetNormalizationAudit:
    FIELDNAMES = [
        "dataset_name",
        "filter",
        "horizon",
        "exit_policy_id",
        "total_rows",
        "valid_rows",
        "invalid_rows",
        "positive_rows",
        "negative_rows",
        "positive_rate",
        "train_rows",
        "validation_rows",
        "test_rows",
        "train_positive_rate",
        "validation_positive_rate",
        "test_positive_rate",
        "normalization_status",
        "normalization_error",
        "skipped_reason",
        "created_at_utc",
    ]

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with self.path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
                writer.writeheader()

    def append_row(self, row: dict[str, Any]) -> None:
        out = {key: row.get(key) for key in self.FIELDNAMES}
        out["created_at_utc"] = out.get("created_at_utc") or utc_now_iso()
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.FIELDNAMES)
            writer.writerow(out)
            handle.flush()
            os.fsync(handle.fileno())


def propagate_random_state(random_state: int) -> dict[str, Any]:
    random.seed(random_state)
    np.random.seed(random_state)
    status: dict[str, Any] = {"random_state": random_state, "python": True, "numpy": True}
    try:
        import torch

        torch.manual_seed(random_state)
        status["torch"] = True
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_state)
            status["torch_cuda"] = True
    except ImportError:
        status["torch"] = False
        status["torch_cuda"] = False
    return status


def torch_availability() -> dict[str, Any]:
    try:
        import torch

        return {
            "torch_available": True,
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_version": getattr(torch, "__version__", None),
        }
    except ImportError:
        return {"torch_available": False, "cuda_available": False, "torch_version": None}


def run_memory_cleanup(*, audit: E5IncrementalAuditLogger | None = None) -> dict[str, Any]:
    before: dict[str, Any] = {}
    after: dict[str, Any] = {}
    try:
        import torch

        if torch.cuda.is_available():
            before["cuda_allocated"] = int(torch.cuda.memory_allocated())
            before["cuda_reserved"] = int(torch.cuda.memory_reserved())
    except ImportError:
        pass
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            after["cuda_allocated"] = int(torch.cuda.memory_allocated())
            after["cuda_reserved"] = int(torch.cuda.memory_reserved())
    except ImportError:
        pass
    summary = {"gc_collect": True, "before": before, "after": after}
    if audit is not None:
        audit.log("cleanup_event", status="ok", memory_cleanup=summary)
    return summary


def normalize_target_column_strict(
    df: pd.DataFrame,
    descriptor: DatasetDescriptor,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    present = [col for col in ACCEPTABLE_TARGET_ALIASES if col in df.columns]
    audit: dict[str, Any] = {
        "dataset_name": descriptor.dataset_name,
        "filter": descriptor.filter_name,
        "horizon": descriptor.horizon,
        "exit_policy_id": descriptor.exit_policy_id,
        "total_rows": len(df),
    }
    if not present:
        audit.update(
            {
                "normalization_status": "DIRECT_TARGET_COLUMN_MISSING",
                "normalization_error": "No acceptable direct target column found",
            }
        )
        return df, audit
    if len(present) > 1:
        audit.update(
            {
                "normalization_status": "AMBIGUOUS_TARGET_ALIAS",
                "normalization_error": "Multiple acceptable target aliases present",
            }
        )
        return df, audit

    source_col = present[0]
    out = df.copy()
    coerced, coercion_errors = strict_coerce_binary_target(out[source_col])
    out[CANONICAL_TARGET] = coerced
    if coercion_errors:
        audit.update(
            {
                "normalization_status": "INVALID_TARGET_COERCION_ERROR",
                "normalization_error": "; ".join(coercion_errors[:5]),
                "invalid_rows": int(coerced.isna().sum()),
            }
        )
        return out, audit

    ok, sanity = target_sanity_check(out[CANONICAL_TARGET])
    if not ok:
        audit.update(
            {
                "normalization_status": sanity.get("error", "INVALID_TARGET_VALUES"),
                "normalization_error": json.dumps(sanity),
                "invalid_rows": sanity.get("invalid_count", 0),
            }
        )
        return out, audit

    positives = int((out[CANONICAL_TARGET] == 1).sum())
    negatives = int((out[CANONICAL_TARGET] == 0).sum())
    valid_rows = int(out[CANONICAL_TARGET].notna().sum())
    audit.update(
        {
            "valid_rows": valid_rows,
            "invalid_rows": int(len(out) - valid_rows),
            "positive_rows": positives,
            "negative_rows": negatives,
            "positive_rate": round(positives / valid_rows, 6) if valid_rows else 0.0,
            "normalization_status": "ok",
            "normalization_error": None,
        }
    )
    for split_name in ("train", "validation", "test"):
        split_df = out[out["split"] == split_name] if "split" in out.columns else pd.DataFrame()
        audit[f"{split_name}_rows"] = len(split_df)
        if len(split_df):
            audit[f"{split_name}_positive_rate"] = round(
                float(split_df[CANONICAL_TARGET].mean()), 6
            )
        else:
            audit[f"{split_name}_positive_rate"] = None
    return out, audit


def count_split_positives(df: pd.DataFrame, split_name: str) -> int:
    split_df = df[df["split"] == split_name]
    if split_df.empty:
        return 0
    return int((split_df[CANONICAL_TARGET] == 1).sum())


def assert_full_split_positive_minimums(
    full_df: pd.DataFrame,
    *,
    min_train_positives: int,
    min_validation_positives: int,
    min_test_positives: int,
) -> None:
    """Raise MIN_* only when the full valid split lacks enough positives."""
    checks = (
        ("train", min_train_positives, "MIN_TRAIN_POSITIVES_NOT_MET"),
        ("validation", min_validation_positives, "MIN_VALIDATION_POSITIVES_NOT_MET"),
        ("test", min_test_positives, "MIN_TEST_POSITIVES_NOT_MET"),
    )
    for split_name, minimum, error_code in checks:
        if count_split_positives(full_df, split_name) < minimum:
            raise RuntimeError(error_code)


def _stratified_sample_one_split(
    split_df: pd.DataFrame,
    *,
    target_rows: int,
    min_positives: int,
    split_name: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    if len(split_df) <= target_rows:
        return split_df.copy()

    pos_df = split_df[split_df[CANONICAL_TARGET] == 1]
    neg_df = split_df[split_df[CANONICAL_TARGET] == 0]
    full_pos = len(pos_df)

    if full_pos >= min_positives:
        n_pos = min(max(min_positives, 1), full_pos, target_rows)
    else:
        n_pos = min(full_pos, target_rows)

    if full_pos >= min_positives and n_pos < min_positives:
        raise RuntimeError(
            f"SMOKE_SAMPLING_POSITIVES_NOT_PRESERVED: split={split_name} "
            f"full_pos={full_pos} sample_pos={n_pos} required={min_positives}"
        )

    n_neg = min(len(neg_df), target_rows - n_pos)
    if split_name == "train" and len(neg_df) > 0 and n_neg == 0 and target_rows > n_pos:
        n_neg = 1

    pos_indices = pos_df.index.to_numpy()
    neg_indices = neg_df.index.to_numpy()
    selected: list[Any] = []
    if n_pos > 0 and len(pos_indices):
        selected.extend(rng.choice(pos_indices, size=n_pos, replace=False).tolist())
    if n_neg > 0 and len(neg_indices):
        selected.extend(rng.choice(neg_indices, size=n_neg, replace=False).tolist())

    remaining = target_rows - len(selected)
    if remaining > 0:
        leftover = [idx for idx in split_df.index if idx not in selected]
        if leftover:
            extra_n = min(remaining, len(leftover))
            selected.extend(rng.choice(leftover, size=extra_n, replace=False).tolist())

    sampled = split_df.loc[selected].copy()
    shuffle_seed = int(rng.integers(0, 2**31 - 1))
    return sampled.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)


def apply_smoke_stratified_split_sampling(
    df: pd.DataFrame,
    *,
    max_rows: int,
    min_train_positives: int,
    min_validation_positives: int,
    min_test_positives: int,
    random_state: int,
) -> pd.DataFrame:
    """
    Split-aware smoke sampling: within each E3 split, sample positives and negatives
    separately and preserve minimum positive counts when available in the full split.
    """
    if len(df) <= max_rows:
        return df.copy()

    split_mins = {
        "train": min_train_positives,
        "validation": min_validation_positives,
        "test": min_test_positives,
    }
    rng = np.random.default_rng(random_state)
    split_frames = {name: df[df["split"] == name] for name in ("train", "validation", "test")}
    total_len = len(df)

    floor: dict[str, int] = {}
    for name, split_df in split_frames.items():
        if split_df.empty:
            floor[name] = 0
            continue
        full_pos = int((split_df[CANONICAL_TARGET] == 1).sum())
        full_neg = int((split_df[CANONICAL_TARGET] == 0).sum())
        need = min(full_pos, split_mins[name]) if full_pos >= split_mins[name] else full_pos
        if name == "train" and full_neg > 0:
            need += 1
        floor[name] = min(len(split_df), need)

    floor_total = sum(floor.values())
    if floor_total > max_rows:
        raise RuntimeError(
            f"SMOKE_SAMPLING_BUDGET_EXCEEDED: minimum rows {floor_total} exceed max_rows={max_rows}"
        )

    alloc = dict(floor)
    remaining = max_rows - floor_total
    if remaining > 0:
        for name, split_df in split_frames.items():
            if split_df.empty:
                continue
            extra_cap = len(split_df) - floor[name]
            if extra_cap <= 0:
                continue
            proportional = int(round(len(split_df) / total_len * max_rows))
            extra = min(proportional - floor[name], extra_cap, remaining)
            extra = max(0, extra)
            alloc[name] += extra
            remaining -= extra
        if remaining > 0:
            for name in ("train", "validation", "test"):
                split_df = split_frames[name]
                if split_df.empty or remaining <= 0:
                    continue
                extra_cap = len(split_df) - alloc[name]
                if extra_cap <= 0:
                    continue
                bump = min(extra_cap, remaining)
                alloc[name] += bump
                remaining -= bump

    parts: list[pd.DataFrame] = []
    for name in ("train", "validation", "test"):
        split_df = split_frames[name]
        if split_df.empty:
            continue
        target_n = min(alloc[name], len(split_df))
        parts.append(
            _stratified_sample_one_split(
                split_df,
                target_rows=target_n,
                min_positives=split_mins[name],
                split_name=name,
                rng=np.random.default_rng(random_state + hash(name) % 10_000),
            )
        )

    if not parts:
        raise RuntimeError("SMOKE_SAMPLING_EMPTY: no split rows available after sampling")

    out = pd.concat(parts, ignore_index=True)
    for name in ("train", "validation", "test"):
        full_split = split_frames[name]
        if full_split.empty:
            continue
        full_pos = count_split_positives(full_split, name)
        sample_pos = count_split_positives(out, name)
        if full_pos >= split_mins[name] and sample_pos < split_mins[name]:
            raise RuntimeError(
                f"SMOKE_SAMPLING_POSITIVES_NOT_PRESERVED: split={name} "
                f"full_pos={full_pos} sample_pos={sample_pos} required={split_mins[name]}"
            )
    return out


def prediction_parquet_path(output_dir: Path, split_name: str, combo: str) -> Path:
    """Short on-disk prediction name; full combo key is stored in metrics/manifest sidecars."""
    digest = sha256_hex(combo)[:16]
    return output_dir / "predictions" / f"direct_target_tabicl_predictions_{split_name}_{digest}.parquet"


def _write_prediction_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write prediction parquet with explicit parent mkdir (Windows-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.writing{path.suffix}")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def build_preprocessing_sidecar_e5(
    *,
    descriptor: DatasetDescriptor,
    feature_columns: list[str],
    excluded_leakage: list[str],
    excluded_identity: list[str],
    context_strategy: str,
    context_size: int,
    random_state: int,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> dict[str, Any]:
    numeric = numeric_features or feature_columns
    categorical = categorical_features or []
    fhash = feature_columns_hash(feature_columns)
    return {
        "model": "TAB",
        "filter": descriptor.filter_name,
        "horizon": descriptor.horizon,
        "exit_policy_id": descriptor.exit_policy_id,
        "target_column_canonical": CANONICAL_TARGET,
        "feature_columns": feature_columns,
        "feature_names_in_": feature_columns,
        "feature_count": len(feature_columns),
        "numeric_features": numeric,
        "categorical_features": categorical,
        "excluded_leakage_columns": excluded_leakage,
        "excluded_metadata_columns": excluded_identity,
        "preprocessing_steps": ["median_imputer", "standard_scaler", "train_only_fit"],
        "scaler": "standard",
        "imputer_strategy": "median",
        "context_strategy": context_strategy,
        "context_size": context_size,
        "feature_order_hash": fhash,
        "random_state": random_state,
        "created_at_utc": utc_now_iso(),
    }


def reindex_features(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    strict: bool = True,
) -> pd.DataFrame:
    missing = [c for c in feature_columns if c not in frame.columns]
    unexpected = [c for c in frame.columns if c not in feature_columns and c not in IDENTITY_COLUMNS_PRESERVE]
    if missing and strict:
        raise RuntimeError(f"FEATURE_SCHEMA_MISSING_COLUMNS: {missing}")
    if unexpected:
        pass  # logged by caller
    ordered = frame.reindex(columns=feature_columns)
    if list(ordered.columns) != feature_columns:
        raise RuntimeError("FEATURE_ORDER_MISMATCH_ERROR")
    return ordered


def compute_rank_percentile(scores: np.ndarray) -> np.ndarray:
    n = len(scores)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(-scores, kind="mergesort")
    ranks = np.empty(n, dtype=float)
    ranks[order] = np.arange(1, n + 1, dtype=float)
    return (1.0 - (ranks - 1.0) / max(n - 1, 1)) * 100.0


def build_context_construction_strategy_hash(config: dict[str, Any]) -> str:
    canonical = json.dumps(config, sort_keys=True, default=str)
    return sha256_hex(canonical)


def build_train_identity_hash(train_df: pd.DataFrame) -> str:
    parts: list[str] = []
    if "target_row_id" in train_df.columns:
        ids = train_df["target_row_id"].astype(str).tolist()
        parts.append(sha256_hex("|".join(sorted(ids))))
    if "candidate_policy_id" in train_df.columns:
        cpids = train_df["candidate_policy_id"].astype(str).tolist()
        parts.append(sha256_hex("|".join(sorted(cpids))))
    return sha256_hex("|".join(parts)) if parts else sha256_hex(str(len(train_df)))


def _project_relative(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_context_cache_key(
    *,
    strategy_hash: str,
    dataset_rel_path: str,
    dataset_content_hash: str | None,
    train_row_count: int,
    train_identity_hash: str,
    feature_order_hash: str,
    target_name: str,
    target_version: str | None,
    filter_name: str,
    horizon: str,
    exit_policy_id: str,
    random_state: int,
) -> str:
    payload = {
        "context_construction_strategy_hash": strategy_hash,
        "dataset_rel_path": dataset_rel_path,
        "dataset_content_hash": dataset_content_hash or "",
        "train_row_count": train_row_count,
        "train_identity_hash": train_identity_hash,
        "feature_order_hash": feature_order_hash,
        "target_name": target_name,
        "target_version": target_version or "",
        "filter": filter_name,
        "horizon": horizon,
        "exit_policy_id": exit_policy_id,
        "split_definition": "train_only_context",
        "random_state": random_state,
    }
    return sha256_hex(json.dumps(payload, sort_keys=True))


class BoundedContextCache:
    """CPU-only bounded context cache; never stores GPU tensors."""

    def __init__(
        self,
        mode: str = "off",
        max_entries: int = 1,
        *,
        audit: E5IncrementalAuditLogger | None = None,
    ) -> None:
        self.mode = mode
        self.max_entries = max(1, max_entries)
        self.audit = audit
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, key: str, metadata: dict[str, Any]) -> dict[str, Any] | None:
        if self.mode == "off":
            return None
        entry = self._cache.get(key)
        if entry is None:
            self.misses += 1
            if self.audit:
                self.audit.log("context_cache_miss", context_cache_key=key, context_cache_mode=self.mode)
            return None
        if entry.get("metadata") != metadata:
            self.misses += 1
            if self.audit:
                self.audit.log(
                    "context_cache_miss",
                    context_cache_key=key,
                    reason="metadata_mismatch",
                    context_cache_mode=self.mode,
                )
            return None
        self.hits += 1
        self._cache.move_to_end(key)
        if self.audit:
            self.audit.log("context_cache_hit", context_cache_key=key, context_cache_mode=self.mode)
        return entry

    def put(self, key: str, value: dict[str, Any], metadata: dict[str, Any]) -> None:
        if self.mode == "off":
            return
        if self.mode not in {"cpu_only", "disk"}:
            return
        while len(self._cache) >= self.max_entries:
            evicted_key, _ = self._cache.popitem(last=False)
            self.evictions += 1
            if self.audit:
                self.audit.log("context_cache_eviction", context_cache_key=evicted_key)
            run_memory_cleanup()
        self._cache[key] = {"value": value, "metadata": metadata, "storage": "cpu_only"}

    def summary(self) -> dict[str, Any]:
        return {
            "context_cache_mode": self.mode,
            "max_context_cache_entries": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
        }


def sanitize_join_keys(frame: pd.DataFrame, key: str) -> pd.Series:
    if key not in frame.columns:
        raise RuntimeError(f"JOIN_KEY_NULL_ERROR: missing column {key}")
    series = frame[key].astype(str).str.strip()
    null_mask = (
        frame[key].isna()
        | series.eq("")
        | series.str.lower().eq("nan")
        | series.str.lower().eq("none")
    )
    if null_mask.any():
        raise RuntimeError(f"JOIN_KEY_NULL_ERROR: {key} has {int(null_mask.sum())} null/empty values")
    return series


def validate_join_key_uniqueness(frame: pd.DataFrame, key: str, side: str) -> None:
    dup = frame[key].duplicated(keep=False)
    if dup.any():
        raise RuntimeError(
            f"JOIN_KEY_DUPLICATE_ERROR: {side} has {int(dup.sum())} duplicate {key} values"
        )


def strict_merge_one_to_one(
    left: pd.DataFrame,
    right: pd.DataFrame,
    on: str,
    *,
    left_name: str = "left",
    right_name: str = "right",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    left_keys = sanitize_join_keys(left, on)
    right_keys = sanitize_join_keys(right, on)
    left_norm = left.copy()
    right_norm = right.copy()
    left_norm[on] = left_keys.astype(str)
    right_norm[on] = right_keys.astype(str)

    validate_join_key_uniqueness(left_norm, on, left_name)
    validate_join_key_uniqueness(right_norm, on, right_name)

    expected = len(set(left_norm[on]) & set(right_norm[on]))
    diagnostic: dict[str, Any] = {
        "join_key": on,
        f"{left_name}_row_count": len(left),
        f"{right_name}_row_count": len(right),
        f"{left_name}_dtype_before": str(left[on].dtype),
        f"{right_name}_dtype_before": str(right[on].dtype),
        f"{left_name}_dtype_after": "string",
        f"{right_name}_dtype_after": "string",
        "expected_matched_rows": expected,
        "validate_one_to_one": True,
    }

    merged = left_norm.merge(right_norm, on=on, how="inner", validate="one_to_one")
    diagnostic["actual_matched_rows"] = len(merged)
    diagnostic["unmatched_left"] = len(left_norm) - len(merged)
    diagnostic["unmatched_right"] = len(right_norm) - len(merged)

    if len(merged) != expected:
        raise RuntimeError(
            f"JOIN_ROW_COUNT_ASSERTION_ERROR: expected {expected}, got {len(merged)}"
        )
    if len(merged) > expected:
        raise RuntimeError(f"JOIN_CARDINALITY_ERROR: merged rows {len(merged)} > expected {expected}")
    diagnostic["post_merge_assertion_status"] = "pass"
    return merged, diagnostic


def merge_tab_xgb_rf_predictions(
    tab_preds: pd.DataFrame,
    xgb_preds: pd.DataFrame,
    rf_preds: pd.DataFrame,
    *,
    join_key: str = PRIMARY_JOIN_KEY,
    fallback_used: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    tab_xgb, diag1 = strict_merge_one_to_one(
        tab_preds, xgb_preds, join_key, left_name="tab", right_name="xgb"
    )
    diag1["merge_stage"] = "tab_xgb"
    diag1["fallback_join_used"] = fallback_used
    diagnostics.append(diag1)

    expected_tab = len(tab_xgb)
    expected_xgb = len(tab_xgb)
    if tab_xgb.shape[0] != expected_tab or tab_xgb.shape[0] != expected_xgb:
        raise RuntimeError("JOIN_ROW_COUNT_ASSERTION_ERROR: tab_xgb row count mismatch")

    tab_xgb_rf, diag2 = strict_merge_one_to_one(
        tab_xgb, rf_preds, join_key, left_name="tab_xgb", right_name="rf"
    )
    diag2["merge_stage"] = "tab_xgb_rf"
    diag2["fallback_join_used"] = fallback_used
    diagnostics.append(diag2)

    if (
        tab_xgb_rf.shape[0] != expected_tab
        or tab_xgb_rf.shape[0] != expected_xgb
        or tab_xgb_rf.shape[0] != len(tab_xgb_rf)
    ):
        raise RuntimeError("JOIN_ROW_COUNT_ASSERTION_ERROR: tab_xgb_rf row count mismatch")
    return tab_xgb_rf, diagnostics


def assign_consensus_tier(
    in_tab: bool,
    in_xgb: bool,
    in_rf: bool,
) -> str:
    vote_count = int(in_tab) + int(in_xgb) + int(in_rf)
    if in_tab and in_xgb and in_rf:
        return "TAB_XGB_RF_ALL3"
    if in_tab and in_rf and not in_xgb:
        return "TAB_RF_ONLY"
    if in_tab and in_xgb and not in_rf:
        return "TAB_XGB_ONLY"
    if in_xgb and in_rf and not in_tab:
        return "XGB_RF_ONLY"
    if in_tab and vote_count == 1:
        return "TAB_ONLY"
    if in_xgb and vote_count == 1:
        return "XGB_ONLY"
    if in_rf and vote_count == 1:
        return "RF_ONLY"
    if vote_count == 0:
        return "NONE"
    return "UNKNOWN"


def build_consensus_frame(
    merged: pd.DataFrame,
    *,
    top_pct: float,
    tab_score_col: str = "tab_score",
    xgb_score_col: str = "predicted_probability_xgb",
    rf_score_col: str = "predicted_probability_rf",
) -> pd.DataFrame:
    n = len(merged)
    k = max(1, int(n * top_pct / 100.0))
    tab_top = set(np.argsort(-merged[tab_score_col].to_numpy(), kind="mergesort")[:k])
    xgb_top = set(np.argsort(-merged[xgb_score_col].to_numpy(), kind="mergesort")[:k])
    rf_top = set(np.argsort(-merged[rf_score_col].to_numpy(), kind="mergesort")[:k])

    rows: list[dict[str, Any]] = []
    for idx in range(n):
        in_tab = idx in tab_top
        in_xgb = idx in xgb_top
        in_rf = idx in rf_top
        tier = assign_consensus_tier(in_tab, in_xgb, in_rf)
        row = merged.iloc[idx].to_dict()
        row.update(
            {
                "top_pct": top_pct,
                "in_tab": in_tab,
                "in_xgb": in_xgb,
                "in_rf": in_rf,
                "vote_count": int(in_tab) + int(in_xgb) + int(in_rf),
                "consensus_tier": tier,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


# --- E5C consensus-tier validation-selected reporting (validation-driven, not test-driven) ---

E5C_MIN_UNIQUE_PAIRS = 7
E5C_MAX_TOP_PAIR_SHARE = 0.25
E5C_RESEARCH_ONLY_TIERS = frozenset({"TAB_XGB_ONLY", "XGB_RF_ONLY"})
E5C_ANCHOR_TIER_1 = "TAB_XGB_RF_ALL3"
E5C_ANCHOR_TIER_2 = "TAB_RF_ONLY"
E5C_VALIDATION_SELECTED_TIERS = frozenset(
    {
        "TAB_XGB_RF_ALL3",
        "TAB_RF_ONLY",
        "TAB_XGB_ONLY",
        "XGB_RF_ONLY",
        "TAB_ONLY",
    }
)


def _first_present_column(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def normalize_consensus_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize merged consensus columns to canonical names for reporting."""
    out = df.copy()
    mappings: dict[str, tuple[str, ...]] = {
        "filter": ("filter", "filter_x", "filter_y"),
        "horizon": ("horizon", "horizon_x", "horizon_y"),
        "exit_policy_id": ("exit_policy_id", "exit_policy_id_x", "exit_policy_id_y"),
        "split": ("split", "split_x", "split_y"),
        "pair_address": ("pair_address", "pair_address_x", "pair_address_y"),
        CANONICAL_TARGET: (
            CANONICAL_TARGET,
            "target_net_profitable",
            "target_net_profitable_x",
            "target_net_profitable_y",
        ),
        "sim_net_return": ("sim_net_return", "sim_net_return_x", "sim_net_return_y"),
    }
    for canonical, candidates in mappings.items():
        source = _first_present_column(out, candidates)
        if source is not None:
            out[canonical] = out[source]
    if "tab_score" not in out.columns:
        raise ValueError("consensus dataframe missing tab_score column")
    if "consensus_tier" not in out.columns or "top_pct" not in out.columns:
        raise ValueError("consensus dataframe missing consensus_tier or top_pct")
    return out


def concentration_status(metrics: dict[str, Any]) -> str:
    unique_pairs = metrics.get("unique_pairs")
    top_share = metrics.get("top_pair_share")
    if unique_pairs is None or int(unique_pairs) < E5C_MIN_UNIQUE_PAIRS:
        return "blocked"
    if top_share is not None and float(top_share) > E5C_MAX_TOP_PAIR_SHARE:
        return "blocked"
    return "ok"


def aggregate_consensus_tier_policy_metrics(
    df: pd.DataFrame,
    *,
    consensus_tier: str,
    filter_name: str,
    horizon: str,
    exit_policy_id: str,
    top_pct: float,
    split_name: str,
    pair_cap: int,
) -> dict[str, Any] | None:
    sub = df[
        (df["consensus_tier"] == consensus_tier)
        & (df["filter"] == filter_name)
        & (df["horizon"] == horizon)
        & (df["exit_policy_id"] == exit_policy_id)
        & (df["top_pct"] == top_pct)
        & (df["split"] == split_name)
    ]
    if sub.empty:
        return None
    selected = select_top_with_pair_cap(
        sub,
        score_col="tab_score",
        k=len(sub),
        pair_cap=pair_cap,
    )
    summary = summarize_policy_selection(
        selected,
        target_col=CANONICAL_TARGET,
        return_col="sim_net_return",
    )
    return {
        "row_count": summary["selected_count"],
        "positive_count": summary["positive_count"],
        "precision": summary["target_precision"],
        "total_net_return": summary["total_net_return"],
        "avg_net_return": summary["avg_net_return"],
        "unique_pairs": summary["unique_pairs"],
        "top_pair_share": summary["top_pair_share"],
    }


def _classify_selection_status(
    *,
    consensus_tier: str,
    validation_metrics: dict[str, Any],
    test_metrics: dict[str, Any] | None,
) -> tuple[str, str, str]:
    if consensus_tier in E5C_RESEARCH_ONLY_TIERS:
        return "RESEARCH_ONLY", "research_only", "Anchor Plan research-only tier"

    val_conc = concentration_status(validation_metrics)
    val_positive = (
        validation_metrics.get("total_net_return") is not None
        and float(validation_metrics["total_net_return"]) > 0
        and validation_metrics.get("avg_net_return") is not None
        and float(validation_metrics["avg_net_return"]) > 0
    )

    if test_metrics is None or int(test_metrics.get("row_count") or 0) == 0:
        return "NO_TEST_MATCH", val_conc, "Validation policy has no matching test rows"

    test_conc = concentration_status(test_metrics)
    test_positive = (
        test_metrics.get("total_net_return") is not None
        and float(test_metrics["total_net_return"]) > 0
        and test_metrics.get("avg_net_return") is not None
        and float(test_metrics["avg_net_return"]) > 0
    )

    if val_positive and test_positive and val_conc == "ok" and test_conc == "ok":
        return (
            "VALIDATION_TO_TEST_PASS",
            "ok",
            "Validation-selected policy passed on test with concentration controls",
        )

    if val_positive and (val_conc == "blocked" or test_conc == "blocked"):
        return (
            "POSITIVE_BUT_CONCENTRATION_BLOCKED",
            "blocked",
            "Validation economics positive but concentration limits exceeded",
        )

    if val_positive and not test_positive:
        return "TEST_NEGATIVE", test_conc, "Validation positive but test economics negative"

    if not val_positive:
        return "TEST_NEGATIVE", val_conc, "Validation economics not positive"

    return "TEST_NEGATIVE", test_conc, "Test economics not positive under validation-selected policy"


def build_validation_selected_consensus_applied_to_test(
    consensus_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validation-driven consensus tier policy selection applied to test.

    Select policies using validation rows only; apply identical keys to test rows.
    """
    if consensus_df.empty:
        return pd.DataFrame()

    df = normalize_consensus_dataframe(consensus_df)
    rows: list[dict[str, Any]] = []

    val_df = df[df["split"] == "validation"]
    test_df = df[df["split"] == "test"]

    policy_keys: set[tuple[Any, ...]] = set()
    for _, combo in val_df[["consensus_tier", "filter", "horizon", "exit_policy_id", "top_pct"]].drop_duplicates().iterrows():
        tier = str(combo["consensus_tier"])
        if tier not in E5C_VALIDATION_SELECTED_TIERS:
            continue
        for pair_cap in E5_PAIR_CAPS:
            policy_keys.add(
                (
                    tier,
                    str(combo["filter"]),
                    str(combo["horizon"]),
                    str(combo["exit_policy_id"]),
                    float(combo["top_pct"]),
                    int(pair_cap),
                )
            )

    for tier, filter_name, horizon, exit_policy_id, top_pct, pair_cap in sorted(policy_keys):
        val_metrics = aggregate_consensus_tier_policy_metrics(
            val_df,
            consensus_tier=tier,
            filter_name=filter_name,
            horizon=horizon,
            exit_policy_id=exit_policy_id,
            top_pct=top_pct,
            split_name="validation",
            pair_cap=pair_cap,
        )
        if val_metrics is None or int(val_metrics.get("row_count") or 0) == 0:
            continue

        val_positive = (
            val_metrics.get("total_net_return") is not None
            and float(val_metrics["total_net_return"]) > 0
            and val_metrics.get("avg_net_return") is not None
            and float(val_metrics["avg_net_return"]) > 0
        )
        # Validation-selected candidates: validation economics must be positive
        # (research-only tiers are still reported with RESEARCH_ONLY status).
        if tier not in E5C_RESEARCH_ONLY_TIERS and not val_positive:
            continue

        test_metrics = aggregate_consensus_tier_policy_metrics(
            test_df,
            consensus_tier=tier,
            filter_name=filter_name,
            horizon=horizon,
            exit_policy_id=exit_policy_id,
            top_pct=top_pct,
            split_name="test",
            pair_cap=pair_cap,
        )

        selection_status, conc_status, recommendation = _classify_selection_status(
            consensus_tier=tier,
            validation_metrics=val_metrics,
            test_metrics=test_metrics,
        )

        rows.append(
            {
                "consensus_tier": tier,
                "filter": filter_name,
                "horizon": horizon,
                "exit_policy_id": exit_policy_id,
                "top_pct": top_pct,
                "pair_cap": pair_cap,
                "validation_row_count": val_metrics["row_count"],
                "validation_positive_count": val_metrics["positive_count"],
                "validation_precision": val_metrics["precision"],
                "validation_total_net_return": val_metrics["total_net_return"],
                "validation_avg_net_return": val_metrics["avg_net_return"],
                "validation_unique_pairs": val_metrics["unique_pairs"],
                "validation_top_pair_share": val_metrics["top_pair_share"],
                "test_row_count": test_metrics["row_count"] if test_metrics else 0,
                "test_positive_count": test_metrics["positive_count"] if test_metrics else None,
                "test_precision": test_metrics["precision"] if test_metrics else None,
                "test_total_net_return": test_metrics["total_net_return"] if test_metrics else None,
                "test_avg_net_return": test_metrics["avg_net_return"] if test_metrics else None,
                "test_unique_pairs": test_metrics["unique_pairs"] if test_metrics else None,
                "test_top_pair_share": test_metrics["top_pair_share"] if test_metrics else None,
                "concentration_status": conc_status,
                "selection_status": selection_status,
                "recommendation": recommendation,
            }
        )

    return pd.DataFrame(rows)


def build_e5c_decision_summary(applied_df: pd.DataFrame) -> str:
    """Compact E5C decision summary for Anchor Plan tiers and E6 gate."""
    lines = [
        "Phase E5C — Direct-Target Consensus Validation-to-Test Decision Summary",
        "",
    ]

    if applied_df.empty:
        lines.extend(
            [
                "No validation-selected consensus policies were generated.",
                "E6: NOT ALLOWED — missing validation-to-test reporting.",
            ]
        )
        return "\n".join(lines)

    def tier_verdict(tier: str) -> str:
        tier_rows = applied_df[applied_df["consensus_tier"] == tier]
        if tier_rows.empty:
            return "no validation-selected evidence"
        passes = int((tier_rows["selection_status"] == "VALIDATION_TO_TEST_PASS").sum())
        blocked = int((tier_rows["selection_status"] == "POSITIVE_BUT_CONCENTRATION_BLOCKED").sum())
        if passes > 0:
            return "confirmed (validation-to-test pass with concentration controls)"
        if blocked > 0:
            return "concentration-blocked (positive signal but pair concentration too high)"
        return "not confirmed in focused E5 validation-to-test reporting"

    lines.append(f"TAB_XGB_RF_ALL3 (Tier 1 candidate): {tier_verdict(E5C_ANCHOR_TIER_1)}")
    lines.append(f"TAB_RF_ONLY (Tier 2 candidate): {tier_verdict(E5C_ANCHOR_TIER_2)}")
    lines.append("TAB_XGB_ONLY: remains research-only (Anchor Plan)")
    lines.append("XGB_RF_ONLY: remains rejected/research-only (Anchor Plan)")

    tab_only = applied_df[applied_df["consensus_tier"] == "TAB_ONLY"]
    if not tab_only.empty and (tab_only["validation_total_net_return"].fillna(-1) > 0).any():
        lines.append("TAB_ONLY: TAB signal exists outside current Anchor Plan consensus tiers")
    else:
        lines.append("TAB_ONLY: no strong standalone TAB signal reported outside Anchor tiers")

    pass_count = int((applied_df["selection_status"] == "VALIDATION_TO_TEST_PASS").sum())
    tier1_pass = int(
        (
            (applied_df["consensus_tier"] == E5C_ANCHOR_TIER_1)
            & (applied_df["selection_status"] == "VALIDATION_TO_TEST_PASS")
        ).sum()
    )
    non_research = applied_df[~applied_df["consensus_tier"].isin(E5C_RESEARCH_ONLY_TIERS)]
    status_counts = applied_df["selection_status"].value_counts().to_dict()
    lines.extend(
        [
            "",
            f"Validation-selected policy rows: {len(applied_df)}",
            f"Non-research validation-selected rows: {len(non_research)}",
            f"Selection status counts: {status_counts}",
            f"VALIDATION_TO_TEST_PASS count: {pass_count}",
            f"TAB_XGB_RF_ALL3 VALIDATION_TO_TEST_PASS count: {tier1_pass}",
            "",
        ]
    )

    if tier1_pass > 0:
        lines.append("E6: ALLOWED — concentration-controlled Tier 1 validation-to-test evidence exists.")
    else:
        lines.append(
            "E6: NOT ALLOWED — no concentration-controlled TAB_XGB_RF_ALL3 validation-to-test pass. "
            "TAB signal may exist, but Anchor Tier 1/2 are not confirmed in focused E5."
        )

    return "\n".join(lines)


def regenerate_e5c_reporting_from_artifacts(output_dir: Path) -> dict[str, Any]:
    """Lightweight E5C reporting regeneration from existing E5 focused outputs."""
    output_dir = Path(output_dir)
    trades_path = output_dir / "consensus" / "direct_target_selected_trades_by_tier.csv"
    if not trades_path.is_file():
        raise FileNotFoundError(f"Missing consensus trades file: {trades_path}")

    consensus_df = pd.read_csv(trades_path)
    applied_df = build_validation_selected_consensus_applied_to_test(consensus_df)

    applied_path = (
        output_dir
        / "policy_evaluation"
        / "validation_selected_policies_direct_target_tabicl_applied_to_test.csv"
    )
    atomic_write_csv(applied_df, applied_path)

    decision_text = build_e5c_decision_summary(applied_df)
    decision_path = output_dir / "reports" / "direct_target_tabicl_e5c_decision_summary.txt"
    atomic_write_text(decision_text, decision_path)

    status_counts = (
        applied_df["selection_status"].value_counts().to_dict() if not applied_df.empty else {}
    )
    return {
        "output_dir": str(output_dir),
        "validation_selected_rows": len(applied_df),
        "validation_selected_path": str(applied_path),
        "decision_summary_path": str(decision_path),
        "selection_status_counts": status_counts,
    }


def evaluate_e5_policy_grid(
    pred_df: pd.DataFrame,
    *,
    model_name: str,
    filter_name: str,
    horizon: str,
    exit_policy_id: str,
    split_name: str,
    return_col: str | None,
    score_col: str = "tab_score",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "pair_address" not in pred_df.columns:
        return rows
    n = len(pred_df)
    for top_pct in E5_TOP_PCTS:
        k = max(1, int(n * top_pct / 100.0))
        for pair_cap in E5_PAIR_CAPS:
            selected = select_top_with_pair_cap(
                pred_df,
                score_col=score_col,
                k=k,
                pair_cap=pair_cap,
            )
            summary = summarize_policy_selection(
                selected,
                target_col=CANONICAL_TARGET,
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
                    "pair_cap": pair_cap,
                    **summary,
                }
            )
    return rows


def resolve_e4a_prediction_path(
    e4a_root: Path,
    model: str,
    split: str,
    descriptor: DatasetDescriptor,
) -> Path:
    combo = f"{model}_{descriptor.filter_name}_{descriptor.horizon}_{descriptor.exit_policy_id}"
    return e4a_root / "predictions" / f"direct_target_predictions_{split}_{combo}.parquet"


def run_dependency_audit(
    *,
    project_root: Path,
    e3_root: Path,
    e4a_root: Path,
    fail_on_missing_registry: bool = True,
    allow_registry_warnings: bool = False,
) -> dict[str, Any]:
    audit: dict[str, Any] = {
        "checked_at_utc": utc_now_iso(),
        "e3_dataset_root": str(e3_root),
        "e4a_output_root": str(e4a_root),
        "checked_paths": [],
        "required_artifact_patterns": E4A_REQUIRED_PATTERNS,
        "found_artifact_counts": {},
        "missing_artifact_counts": {},
        "registry_hit_counts": 0,
        "missing_registry_hits": [],
        "readable_status": {},
        "status": "pass",
        "failure_reason": None,
    }

    for label, path in (("e3_dataset_root", e3_root), ("e4a_output_root", e4a_root)):
        audit["checked_paths"].append(str(path))
        exists = path.is_dir()
        audit["readable_status"][label] = "readable" if exists else "missing"
        if not exists:
            audit["status"] = "fail"
            audit["failure_reason"] = f"{label} missing or unreadable"

    registry_path = project_root / "data/training/artifact_registry/artifact_registry.jsonl"
    registry_records: dict[str, Any] = {}
    if registry_path.exists():
        registry_records = load_registry(registry_path)

    e4a_rel_prefix = _project_relative(e4a_root, project_root)

    for pattern_key, pattern in E4A_REQUIRED_PATTERNS.items():
        matches = list(e4a_root.glob(pattern)) if e4a_root.is_dir() else []
        audit["found_artifact_counts"][pattern_key] = len(matches)
        audit["missing_artifact_counts"][pattern_key] = 0 if matches else 1
        if not matches:
            audit["status"] = "fail"
            audit["failure_reason"] = f"Missing E4A artifacts for pattern {pattern_key}: {pattern}"
        for match in matches:
            rel = _project_relative(match, project_root) if match.is_file() else match.name
            readable = match.is_file() and os.access(match, os.R_OK)
            audit["readable_status"][rel] = "readable" if readable else "unreadable"
            if not readable:
                audit["status"] = "fail"
                audit["failure_reason"] = f"Unreadable artifact: {rel}"
            reg_hit = any(
                r.project_relative_path.replace("\\", "/") == rel.replace("\\", "/")
                for r in registry_records.values()
            )
            if reg_hit:
                audit["registry_hit_counts"] += 1
            else:
                audit["missing_registry_hits"].append(rel)

    manifest_path = e4a_root / E4A_REQUIRED_PATTERNS["manifest"]
    if manifest_path.is_file():
        try:
            with manifest_path.open(encoding="utf-8") as handle:
                json.load(handle)
            audit["readable_status"][str(manifest_path)] = "readable"
        except (json.JSONDecodeError, OSError) as exc:
            audit["status"] = "fail"
            audit["failure_reason"] = f"E4A manifest unreadable: {exc}"

    e4a_registered = [
        r.project_relative_path
        for r in registry_records.values()
        if r.project_relative_path.replace("\\", "/").startswith(e4a_rel_prefix)
    ]
    if not e4a_registered:
        msg = f"E4A root not registered in artifact registry: {e4a_rel_prefix}"
        if fail_on_missing_registry and not allow_registry_warnings:
            audit["status"] = "fail"
            audit["failure_reason"] = msg
        else:
            audit.setdefault("warnings", []).append(msg)

    if audit["missing_registry_hits"] and fail_on_missing_registry and not allow_registry_warnings:
        if audit["status"] != "fail":
            audit["status"] = "warn"
            audit["failure_reason"] = "Some E4A artifacts missing from registry"

    return audit


def check_context_drift(
    strategy_hash: str,
    *,
    prior_reports_glob: list[Path],
) -> dict[str, Any]:
    for report_path in prior_reports_glob:
        if not report_path.is_file():
            continue
        try:
            with report_path.open(encoding="utf-8") as handle:
                prior = json.load(handle)
            prior_hash = prior.get("context_construction_strategy_hash") or prior.get(
                "context_strategy_hash"
            )
            if prior_hash and prior_hash != strategy_hash:
                return {
                    "status": "TAB_CONTEXT_DRIFT_WARNING",
                    "prior_hash": prior_hash,
                    "current_hash": strategy_hash,
                    "prior_path": str(report_path),
                }
            if prior_hash:
                return {"status": "ok", "prior_hash": prior_hash}
        except (json.JSONDecodeError, OSError):
            continue
    return {"status": "TAB_CONTEXT_BASELINE_NOT_FOUND"}


def artifact_combo_key_tab(descriptor: DatasetDescriptor, context_strategy: str) -> str:
    return f"TAB_{descriptor.filter_name}_{descriptor.horizon}_{descriptor.exit_policy_id}_{context_strategy}"


def load_e4a_predictions_for_descriptor(
    e4a_root: Path,
    descriptor: DatasetDescriptor,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None, pd.DataFrame | None]:
    paths = {
        "xgb_val": resolve_e4a_prediction_path(e4a_root, "XGB", "validation", descriptor),
        "xgb_test": resolve_e4a_prediction_path(e4a_root, "XGB", "test", descriptor),
        "rf_val": resolve_e4a_prediction_path(e4a_root, "RF", "validation", descriptor),
        "rf_test": resolve_e4a_prediction_path(e4a_root, "RF", "test", descriptor),
    }
    loaded: dict[str, pd.DataFrame | None] = {}
    for key, path in paths.items():
        loaded[key] = pd.read_parquet(path) if path.is_file() else None
    return loaded["xgb_val"], loaded["xgb_test"], loaded["rf_val"], loaded["rf_test"]


@dataclass
class EvalConfig:
    input_dir: Path
    output_dir: Path
    e4a_root: Path
    smoke: bool = True
    focused: bool = False
    full: bool = False
    context_strategies: tuple[str, ...] = DEFAULT_CONTEXT_STRATEGIES
    context_sizes: tuple[int, ...] = (SMOKE_DEFAULT_MAX_CONTEXT_SIZE,)
    max_context_size: int = SMOKE_DEFAULT_MAX_CONTEXT_SIZE
    query_batch_size: int = SMOKE_DEFAULT_QUERY_BATCH_SIZE
    max_query_batch_size: int = SMOKE_DEFAULT_QUERY_BATCH_SIZE
    max_workers: int = DEFAULT_MAX_WORKERS
    context_cache_mode: str = "off"
    max_context_cache_entries: int = 1
    register_artifacts: bool = True
    random_state: int = 42
    max_rows: int | None = None
    device: str = "auto"
    min_train_positives: int = 10
    min_validation_positives: int = 3
    min_test_positives: int = 3
    fail_on_missing_e4a_registry: bool = True
    allow_registry_warnings: bool = False
    selected_descriptors: list[DatasetDescriptor] | None = None
    skip_tab_inference: bool = False


@dataclass
class RunState:
    policy_grid_rows: list[dict[str, Any]] = field(default_factory=list)
    context_audit_rows: list[dict[str, Any]] = field(default_factory=list)
    feature_audit_rows: list[dict[str, Any]] = field(default_factory=list)
    join_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    consensus_rows: list[dict[str, Any]] = field(default_factory=list)
    comparison_rows: list[dict[str, Any]] = field(default_factory=list)
    skipped_jobs: list[dict[str, Any]] = field(default_factory=list)
    failed_jobs: list[dict[str, Any]] = field(default_factory=list)
    submitted_jobs: int = 0
    processed_jobs: int = 0
    successful_jobs: int = 0
    oom_events: list[dict[str, Any]] = field(default_factory=list)
    memory_cleanup_events: list[dict[str, Any]] = field(default_factory=list)
    context_drift_warnings: list[dict[str, Any]] = field(default_factory=list)


def prepare_output_dirs(output_dir: Path) -> None:
    for sub in ("predictions", "metrics", "policy_evaluation", "consensus", "audit", "reports"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)


def register_e5_artifacts(project_root: Path, output_dir: Path) -> dict[str, Any]:
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
        from app.artifacts.registry import get_git_commit_hash, scan_artifacts, write_registry_jsonl

        rel_output = output_dir.relative_to(project_root).as_posix()
        registry_path = project_root / "data/training/artifact_registry/artifact_registry.jsonl"
        git_commit_hash, git_warnings = get_git_commit_hash(project_root)
        previous = load_registry(registry_path)
        records, scan_warnings = scan_artifacts(
            project_root=project_root,
            scan_roots=[rel_output],
            branch_name=BRANCH_NAME,
            generated_by_script="scripts/evaluate_direct_target_tabicl.py",
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


def _fallback_tab_scores(n: int, y_train: np.ndarray, random_state: int) -> np.ndarray:
    """Deterministic fallback when TabICL unavailable (smoke/tests only)."""
    rng = np.random.default_rng(random_state)
    base_rate = float(y_train.mean()) if len(y_train) else 0.5
    return rng.random(n) * 0.3 + base_rate * 0.7


def evaluate_single_job(
    job: dict[str, Any],
    *,
    config: EvalConfig,
    output_dir: Path,
    audit: E5IncrementalAuditLogger,
    context_cache: BoundedContextCache,
    project_root: Path,
    run_id: str,
    target_norm_audit: E5TargetNormalizationAudit | None = None,
) -> dict[str, Any]:
    descriptor: DatasetDescriptor = job["descriptor"]
    context_strategy: str = job["context_strategy"]
    context_size: int = min(job["context_size"], config.max_context_size)
    result: dict[str, Any] = {
        "descriptor": descriptor.dataset_name,
        "context_strategy": context_strategy,
        "status": "started",
    }

    propagate_random_state(config.random_state)
    try:
        validate_context_strategy(context_strategy)
        df = load_dataset(descriptor.dataset_path)
        df, norm_audit = normalize_target_column_strict(df, descriptor)
        if target_norm_audit is not None:
            target_norm_audit.append_row(norm_audit)
        if norm_audit.get("normalization_status") != "ok":
            raise RuntimeError(norm_audit.get("normalization_status"))

        valid_mask = derive_valid_label_mask(df)
        valid_df = df.loc[valid_mask].copy()
        ok_split, split_err = validate_split_column(valid_df)
        if not ok_split:
            raise RuntimeError(split_err or "SPLIT_COLUMN_MISSING")

        assert_full_split_positive_minimums(
            valid_df,
            min_train_positives=config.min_train_positives,
            min_validation_positives=config.min_validation_positives,
            min_test_positives=config.min_test_positives,
        )

        max_rows = config.max_rows
        if config.smoke and max_rows is None:
            max_rows = SMOKE_DEFAULT_MAX_ROWS
        if config.smoke and max_rows is not None:
            valid_df = apply_smoke_stratified_split_sampling(
                valid_df,
                max_rows=max_rows,
                min_train_positives=config.min_train_positives,
                min_validation_positives=config.min_validation_positives,
                min_test_positives=config.min_test_positives,
                random_state=config.random_state,
            )
            audit.log(
                "smoke_stratified_sampling_applied",
                dataset=descriptor.dataset_name,
                max_rows=max_rows,
                post_sample_row_count=len(valid_df),
                train_positives=count_split_positives(valid_df, "train"),
                validation_positives=count_split_positives(valid_df, "validation"),
                test_positives=count_split_positives(valid_df, "test"),
            )
        elif max_rows is not None:
            valid_df = apply_deterministic_row_limit(valid_df, max_rows)

        feature_columns, excluded_leakage, excluded_identity, dropped_all_null = build_feature_columns(
            valid_df
        )
        fhash = feature_columns_hash(feature_columns)
        train_df = valid_df[valid_df["split"] == "train"]
        val_df = valid_df[valid_df["split"] == "validation"]
        test_df = valid_df[valid_df["split"] == "test"]
        pos_train = count_split_positives(valid_df, "train")

        if len(np.unique(train_df[CANONICAL_TARGET])) < 2:
            raise RuntimeError("TRAIN_SINGLE_CLASS")

        strategy_config = {
            "strategy": context_strategy,
            "strategy_version": CONTEXT_STRATEGY_VERSION,
            "context_size": context_size,
            "max_context_size": config.max_context_size,
            "random_state": config.random_state,
            "filter": descriptor.filter_name,
            "horizon": descriptor.horizon,
            "exit_policy_id": descriptor.exit_policy_id,
            "train_split": "train",
            "scaler": "standard",
        }
        strategy_hash = build_context_construction_strategy_hash(strategy_config)
        train_identity_hash = build_train_identity_hash(train_df)
        dataset_rel = _project_relative(descriptor.dataset_path, project_root)
        cache_key = build_context_cache_key(
            strategy_hash=strategy_hash,
            dataset_rel_path=dataset_rel,
            dataset_content_hash=None,
            train_row_count=len(train_df),
            train_identity_hash=train_identity_hash,
            feature_order_hash=fhash,
            target_name=descriptor.target_name,
            target_version=descriptor.target_version,
            filter_name=descriptor.filter_name,
            horizon=descriptor.horizon,
            exit_policy_id=descriptor.exit_policy_id,
            random_state=config.random_state,
        )
        cache_meta = {
            "train_row_count": len(train_df),
            "train_identity_hash": train_identity_hash,
            "feature_order_hash": fhash,
        }

        audit.log(
            "context_construction_start",
            dataset=descriptor.dataset_name,
            context_strategy=context_strategy,
            context_construction_strategy_hash=strategy_hash,
            context_cache_key=cache_key,
        )

        preprocessor = TrainOnlyPreprocessor(scaler="standard")
        x_train_raw = reindex_features(build_feature_matrix(valid_df, feature_columns, split_name="train"), feature_columns)
        x_val_raw = reindex_features(build_feature_matrix(valid_df, feature_columns, split_name="validation"), feature_columns)
        x_test_raw = reindex_features(build_feature_matrix(valid_df, feature_columns, split_name="test"), feature_columns)
        preprocessor.fit(x_train_raw, feature_columns)
        x_train = preprocessor.transform(x_train_raw)
        x_val = preprocessor.transform(x_val_raw)
        x_test = preprocessor.transform(x_test_raw)
        y_train = train_df[CANONICAL_TARGET].astype(int).to_numpy()

        sidecar = build_preprocessing_sidecar_e5(
            descriptor=descriptor,
            feature_columns=feature_columns,
            excluded_leakage=excluded_leakage,
            excluded_identity=excluded_identity,
            context_strategy=context_strategy,
            context_size=context_size,
            random_state=config.random_state,
        )
        combo = artifact_combo_key_tab(descriptor, context_strategy)
        combo_digest = sha256_hex(combo)[:16]
        sidecar_path = output_dir / "metrics" / f"direct_target_tabicl_preprocessing_{combo_digest}.json"
        atomic_write_json(sidecar, sidecar_path)

        device = resolve_device(config.device)
        query_batch = min(config.query_batch_size, config.max_query_batch_size)
        effective_context = cap_context_size(context_size, config.max_context_size, len(y_train))

        cached = context_cache.get(cache_key, cache_meta)
        if cached is not None:
            val_scores = cached["value"]["val_scores"]
            test_scores = cached["value"]["test_scores"]
        elif config.skip_tab_inference or not tabicl_available():
            val_scores = _fallback_tab_scores(len(val_df), y_train, config.random_state)
            test_scores = _fallback_tab_scores(len(test_df), y_train, config.random_state + 1)
        else:
            try:
                val_scores, test_scores, tab_meta = run_tabicl_with_oom_retry(
                    x_train,
                    y_train,
                    x_val,
                    x_test,
                    device=device,
                    batch_size=query_batch,
                    context_size=effective_context,
                    y_train_full=y_train,
                    max_train_context_rows=config.max_context_size,
                    random_state=config.random_state,
                    context_strategy=context_strategy,
                )
                if tab_meta.get("oom_retry_count"):
                    audit.log("oom_retry", dataset=descriptor.dataset_name, **tab_meta)
            except Exception as exc:
                if is_cuda_oom(exc):
                    audit.log("oom_event", dataset=descriptor.dataset_name, error=str(exc))
                    run_memory_cleanup(audit=audit)
                    raise RuntimeError(f"TAB_OOM_UNRECOVERABLE: {exc}") from exc
                raise
            context_cache.put(
                cache_key,
                {"val_scores": val_scores, "test_scores": test_scores},
                cache_meta,
            )

        audit.log(
            "context_construction_end",
            dataset=descriptor.dataset_name,
            context_construction_strategy_hash=strategy_hash,
            context_cache_key=cache_key,
        )

        return_col = resolve_eval_return_column(valid_df)
        pred_frames: dict[str, pd.DataFrame] = {}
        required_identity = ("candidate_id", "candidate_policy_id", "target_row_id")
        (output_dir / "predictions").mkdir(parents=True, exist_ok=True)
        pred_paths: dict[str, str] = {}
        for split_name, split_df, scores in (
            ("validation", val_df, val_scores),
            ("test", test_df, test_scores),
        ):
            missing_required = [c for c in required_identity if c not in split_df.columns]
            if missing_required:
                raise RuntimeError(f"IDENTITY_COLUMNS_MISSING: {missing_required}")
            present_identity = [c for c in IDENTITY_COLUMNS_PRESERVE if c in split_df.columns]
            pred = split_df[present_identity].copy()
            score_arr = np.asarray(scores, dtype=float)
            if len(score_arr) != len(pred):
                raise RuntimeError(
                    f"PREDICTION_LENGTH_MISMATCH: split={split_name} "
                    f"scores={len(score_arr)} rows={len(pred)}"
                )
            pred[CANONICAL_TARGET] = split_df[CANONICAL_TARGET].astype(int).values
            pred["tab_score"] = score_arr
            pred["tab_rank_pct"] = compute_rank_percentile(score_arr)
            pred["tab_context_strategy"] = context_strategy
            pred["tab_context_size"] = int(effective_context)
            pred["tab_query_batch_size"] = int(query_batch)
            pred["tab_model_name_or_version"] = TAB_MODEL_VERSION
            pred["context_construction_strategy_hash"] = strategy_hash
            pred["context_cache_key"] = cache_key
            pred["run_id"] = run_id
            if return_col and return_col in split_df.columns:
                pred["sim_net_return"] = pd.to_numeric(split_df[return_col], errors="coerce").values
            for col in pred.columns:
                if pred[col].dtype == object:
                    pred[col] = pred[col].astype(str)
            path = prediction_parquet_path(output_dir, split_name, combo)
            _write_prediction_parquet(pred.reset_index(drop=True), path)
            pred_paths[split_name] = str(path)
            pred_frames[split_name] = pred

        y_val = val_df[CANONICAL_TARGET].astype(int).to_numpy()
        y_test = test_df[CANONICAL_TARGET].astype(int).to_numpy()
        from sklearn.metrics import average_precision_score, roc_auc_score

        split_metrics = []
        for split_name, y_true, scores in (
            ("validation", y_val, val_scores),
            ("test", y_test, test_scores),
        ):
            sm: dict[str, Any] = {
                "split": split_name,
                "row_count": len(y_true),
                "positive_count": int((y_true == 1).sum()),
                "context_construction_strategy_hash": strategy_hash,
                "context_cache_key": cache_key,
            }
            if len(np.unique(y_true)) > 1:
                sm["pr_auc"] = float(average_precision_score(y_true, scores))
                sm["roc_auc"] = float(roc_auc_score(y_true, scores))
            for pct in E5_TOP_PCTS:
                key = str(pct).replace(".", "_")
                prec = precision_at_top_k_with_count(y_true, scores, pct)
                sm[f"precision_at_top_{key}_percent"] = prec["precision"]
                sm[f"selected_count_top_{key}_percent"] = prec["trade_count"]
            split_metrics.append(sm)

        metrics_payload = {
            "model": "TAB",
            "filter": descriptor.filter_name,
            "horizon": descriptor.horizon,
            "exit_policy_id": descriptor.exit_policy_id,
            "context_strategy": context_strategy,
            "artifact_combo_key": combo,
            "context_construction_strategy_hash": strategy_hash,
            "context_cache_key": cache_key,
            "prediction_paths": pred_paths,
            "split_metrics": split_metrics,
            "random_state": config.random_state,
            "smoke_only": config.smoke,
        }
        metrics_path = output_dir / "metrics" / f"direct_target_tabicl_metrics_{combo_digest}.json"
        atomic_write_json(metrics_payload, metrics_path)

        for split_name, pred in pred_frames.items():
            result["policy_rows"] = evaluate_e5_policy_grid(
                pred,
                model_name="TAB",
                filter_name=descriptor.filter_name,
                horizon=descriptor.horizon,
                exit_policy_id=descriptor.exit_policy_id,
                split_name=split_name,
                return_col=return_col,
            )

        xgb_val, xgb_test, rf_val, rf_test = load_e4a_predictions_for_descriptor(config.e4a_root, descriptor)
        join_status = "skipped"
        if xgb_val is not None and rf_val is not None:
            tab_val = pred_frames["validation"].rename(columns={"tab_score": "tab_score"})
            tab_test = pred_frames["test"].rename(columns={"tab_score": "tab_score"})
            xgb_val = xgb_val.rename(columns={"predicted_probability": "predicted_probability_xgb"})
            xgb_test = xgb_test.rename(columns={"predicted_probability": "predicted_probability_xgb"})
            rf_val = rf_val.rename(columns={"predicted_probability": "predicted_probability_rf"})
            rf_test = rf_test.rename(columns={"predicted_probability": "predicted_probability_rf"})
            for split_name, tab_p, xgb_p, rf_p in (
                ("validation", tab_val, xgb_val, rf_val),
                ("test", tab_test, xgb_test, rf_test),
            ):
                try:
                    merged, diags = merge_tab_xgb_rf_predictions(tab_p, xgb_p, rf_p)
                    for d in diags:
                        d.update(
                            {
                                "filter": descriptor.filter_name,
                                "horizon": descriptor.horizon,
                                "exit_policy_id": descriptor.exit_policy_id,
                                "split": split_name,
                                "final_join_status": "pass",
                            }
                        )
                    result.setdefault("join_diagnostics", []).extend(diags)
                    for top_pct in CONSENSUS_TOP_PCTS:
                        consensus = build_consensus_frame(merged, top_pct=top_pct)
                        result.setdefault("consensus_frames", []).append(consensus)
                        tier_counts = consensus["consensus_tier"].value_counts().to_dict()
                        result.setdefault("comparison_rows", []).append(
                            {
                                "filter": descriptor.filter_name,
                                "horizon": descriptor.horizon,
                                "exit_policy_id": descriptor.exit_policy_id,
                                "split": split_name,
                                "top_pct": top_pct,
                                "matched_rows": len(merged),
                                **{f"tier_{k}": v for k, v in tier_counts.items()},
                            }
                        )
                    join_status = "pass"
                except RuntimeError as join_exc:
                    result.setdefault("join_diagnostics", []).append(
                        {
                            "filter": descriptor.filter_name,
                            "horizon": descriptor.horizon,
                            "exit_policy_id": descriptor.exit_policy_id,
                            "split": split_name,
                            "final_join_status": "fail",
                            "error": str(join_exc),
                        }
                    )

        result.update(
            {
                "status": "completed",
                "context_audit": {
                    "dataset_name": descriptor.dataset_name,
                    "filter": descriptor.filter_name,
                    "horizon": descriptor.horizon,
                    "exit_policy_id": descriptor.exit_policy_id,
                    "context_strategy": context_strategy,
                    "context_size": effective_context,
                    "max_context_size": config.max_context_size,
                    "query_batch_size": query_batch,
                    "random_state": config.random_state,
                    "context_construction_strategy_hash": strategy_hash,
                    "context_cache_key": cache_key,
                    "context_cache_mode": config.context_cache_mode,
                    "context_cache_status": "hit" if cached else "miss",
                    "train_row_count": len(train_df),
                    "train_positive_count": pos_train,
                    "train_positive_rate": round(pos_train / len(train_df), 6) if len(train_df) else 0,
                    "context_row_count": effective_context,
                },
                "feature_audit": {
                    "dataset_name": descriptor.dataset_name,
                    "feature_columns": json.dumps(feature_columns),
                    "feature_order_hash": fhash,
                    "excluded_leakage_columns": json.dumps(excluded_leakage),
                    "excluded_metadata_columns": json.dumps(excluded_identity),
                    "feature_count": len(feature_columns),
                },
                "join_status": join_status,
                "pred_frames": pred_frames,
            }
        )
        audit.log("dataset_completed", dataset=descriptor.dataset_name, status="completed")
    except Exception as exc:
        exc_str = str(exc)
        if exc_str.startswith("MIN_") and exc_str.endswith("_NOT_MET"):
            result["status"] = "skipped"
        elif "SINGLE_CLASS" in exc_str:
            result["status"] = "skipped"
        elif "SMOKE_SAMPLING" in exc_str:
            result["status"] = "failed"
        else:
            result["status"] = "failed"
        result["error"] = exc_str
        audit.log(
            "dataset_skip" if result["status"] == "skipped" else "dataset_failed",
            dataset=descriptor.dataset_name,
            error=exc_str,
        )
    finally:
        run_memory_cleanup(audit=audit)
    return result


def build_job_list(
    descriptors: list[DatasetDescriptor],
    config: EvalConfig,
) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for descriptor in descriptors:
        for strategy in config.context_strategies:
            for ctx_size in config.context_sizes:
                effective = min(ctx_size, config.max_context_size)
                jobs.append(
                    {
                        "descriptor": descriptor,
                        "context_strategy": strategy,
                        "context_size": effective,
                    }
                )
    return jobs


def run_bounded_executor(
    jobs: list[dict[str, Any]],
    worker_fn: Callable[[dict[str, Any]], dict[str, Any]],
    *,
    max_workers: int,
    audit: E5IncrementalAuditLogger,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    max_workers = max(1, int(max_workers))
    audit.log(
        "executor_start",
        max_workers=max_workers,
        submitted_jobs=len(jobs),
        executor_type="ThreadPoolExecutor" if max_workers > 1 else "sequential",
    )
    if max_workers == 1:
        for job in jobs:
            results.append(worker_fn(job))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(worker_fn, job): job for job in jobs}
            for future in as_completed(futures):
                results.append(future.result())
    audit.log("executor_complete", processed_jobs=len(results))
    return results


def filter_descriptors_for_mode(
    descriptors: list[DatasetDescriptor],
    config: EvalConfig,
    *,
    filters: list[str] | None = None,
    horizons: list[str] | None = None,
    exit_policies: list[str] | None = None,
) -> list[DatasetDescriptor]:
    selected = descriptors
    if config.smoke and not config.focused and not config.full:
        return filter_descriptors(selected, filter_name="LIQ_5K_HIGH_ACTIVITY", horizon="1h", exit_policy=FOCUSED_EXIT_POLICIES[0], smoke=True)
    if config.focused:
        filt = filters or list(FOCUSED_FILTERS)
        hor = horizons or list(FOCUSED_HORIZONS)
        pol = exit_policies or list(FOCUSED_EXIT_POLICIES)
        selected = [d for d in selected if d.filter_name in filt and d.horizon in hor and d.exit_policy_id in pol]
        return selected
    if filters:
        selected = [d for d in selected if d.filter_name in filters]
    if horizons:
        selected = [d for d in selected if d.horizon in horizons]
    if exit_policies:
        selected = [d for d in selected if d.exit_policy_id in exit_policies]
    return selected


def finalize_run_outputs(
    *,
    config: EvalConfig,
    output_dir: Path,
    state: RunState,
    audit: E5IncrementalAuditLogger,
    descriptors: list[DatasetDescriptor],
    dependency_audit: dict[str, Any],
    registration_status: dict[str, Any] | None,
    run_id: str,
    seed_status: dict[str, Any],
    context_cache: BoundedContextCache,
) -> None:
    grid_df = pd.DataFrame(state.policy_grid_rows)
    if not grid_df.empty:
        atomic_write_csv(
            grid_df,
            output_dir / "policy_evaluation" / "direct_target_tabicl_policy_grid.csv",
        )

    if state.consensus_rows:
        consensus_df = pd.DataFrame(state.consensus_rows)
        applied_df = build_validation_selected_consensus_applied_to_test(consensus_df)
        atomic_write_csv(
            applied_df,
            output_dir
            / "policy_evaluation"
            / "validation_selected_policies_direct_target_tabicl_applied_to_test.csv",
        )
        atomic_write_text(
            build_e5c_decision_summary(applied_df),
            output_dir / "reports" / "direct_target_tabicl_e5c_decision_summary.txt",
        )
    elif not grid_df.empty:
        # Fallback: TAB-only policy grid (legacy path) when consensus rows unavailable
        selected_val = rank_validation_policies(grid_df)
        applied = apply_validation_policies_to_test(grid_df, selected_val)
        atomic_write_csv(
            applied,
            output_dir
            / "policy_evaluation"
            / "validation_selected_policies_direct_target_tabicl_applied_to_test.csv",
        )

    if state.context_audit_rows:
        atomic_write_csv(
            pd.DataFrame(state.context_audit_rows),
            output_dir / "audit" / "direct_target_tabicl_context_audit.csv",
        )
    if state.feature_audit_rows:
        atomic_write_csv(
            pd.DataFrame(state.feature_audit_rows),
            output_dir / "audit" / "direct_target_tabicl_feature_audit.csv",
        )
    if state.join_diagnostics:
        atomic_write_csv(
            pd.DataFrame(state.join_diagnostics),
            output_dir / "consensus" / "direct_target_tab_xgb_rf_join_diagnostic.csv",
        )
    if state.consensus_rows:
        consensus_df = pd.DataFrame(state.consensus_rows)
        atomic_write_csv(
            consensus_df,
            output_dir / "consensus" / "direct_target_selected_trades_by_tier.csv",
        )
        tier_summary = (
            consensus_df.groupby(["consensus_tier", "top_pct", "split"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        atomic_write_csv(
            tier_summary,
            output_dir / "consensus" / "direct_target_consensus_tier_summary.csv",
        )
        policy_tier = (
            consensus_df.groupby(["consensus_tier", "top_pct"], dropna=False)
            .agg(
                row_count=("consensus_tier", "size"),
                tab_inclusions=("in_tab", "sum"),
                xgb_inclusions=("in_xgb", "sum"),
                rf_inclusions=("in_rf", "sum"),
            )
            .reset_index()
        )
        atomic_write_csv(
            policy_tier,
            output_dir / "consensus" / "direct_target_policy_tier_summary.csv",
        )
        if state.comparison_rows:
            atomic_write_csv(
                pd.DataFrame(state.comparison_rows),
                output_dir / "consensus" / "direct_target_tab_xgb_rf_comparison.csv",
            )

    torch_info = torch_availability()
    manifest = {
        "phase": PHASE,
        "branch_name": BRANCH_NAME,
        "run_id": run_id,
        "created_at_utc": utc_now_iso(),
        "input_dataset_root": str(config.input_dir),
        "e4a_comparison_root": str(config.e4a_root),
        "output_root": str(output_dir),
        "tab_model_name": TAB_MODEL_NAME,
        "tab_model_version": TAB_MODEL_VERSION,
        "context_strategies_used": list(config.context_strategies),
        "context_sizes_used": list(config.context_sizes),
        "max_context_size": config.max_context_size,
        "query_batch_sizes_used": [config.query_batch_size],
        "max_query_batch_size": config.max_query_batch_size,
        "max_workers": config.max_workers,
        "concurrency_mode": "bounded",
        "executor_type": "ThreadPoolExecutor" if config.max_workers > 1 else "sequential",
        "context_cache": context_cache.summary(),
        "device": config.device,
        **torch_info,
        "random_state": config.random_state,
        "random_state_propagation": seed_status,
        "dependency_audit_status": dependency_audit.get("status"),
        "datasets_requested": len(descriptors),
        "datasets_completed": state.successful_jobs,
        "successful_jobs": state.successful_jobs,
        "datasets_skipped": len(state.skipped_jobs),
        "submitted_jobs": state.submitted_jobs,
        "processed_jobs": state.processed_jobs,
        "skipped_jobs": len(state.skipped_jobs),
        "failed_jobs": len(state.failed_jobs),
        "skip_reasons": state.skipped_jobs,
        "failed_job_details": state.failed_jobs,
        "oom_retry_summary": state.oom_events,
        "memory_cleanup_summary": state.memory_cleanup_events,
        "context_drift_warnings": state.context_drift_warnings,
        "join_validation_summary": {"diagnostics_count": len(state.join_diagnostics)},
        "registration": registration_status,
        "smoke": config.smoke,
        "focused": config.focused,
        "full": config.full,
    }
    atomic_write_json(manifest, output_dir / "reports" / "direct_target_tabicl_manifest.json")

    summary = (
        f"Phase E5 direct-target TabICL evaluation\n"
        f"Run ID: {run_id}\n"
        f"Datasets completed: {state.successful_jobs}/{len(descriptors)}\n"
        f"Jobs: submitted={state.submitted_jobs} processed={state.processed_jobs} "
        f"successful={state.successful_jobs} skipped={len(state.skipped_jobs)} "
        f"failed={len(state.failed_jobs)}\n"
        f"Dependency audit: {dependency_audit.get('status')}\n"
        f"Smoke mode: {config.smoke}\n"
        f"Output: {output_dir}\n"
        f"\nE6 readiness: {'conditional' if dependency_audit.get('status') in ('pass', 'warn') and state.successful_jobs > 0 else 'blocked'}\n"
        f"- TAB signal under direct target requires full focused/full runs with TabICL venv.\n"
        f"- Consensus tier reconstruction depends on E4A prediction parquets on disk.\n"
    )
    if dependency_audit.get("status") != "pass":
        summary += f"\nBLOCKER: {dependency_audit.get('failure_reason')}\n"
    atomic_write_text(summary, output_dir / "reports" / "direct_target_tabicl_summary_for_upload.txt")
    audit.log("run_completed", status="ok", run_id=run_id)


def run_evaluation(
    config: EvalConfig,
    *,
    e3_manifest_path: Path | None = None,
    filters: list[str] | None = None,
    horizons: list[str] | None = None,
    exit_policies: list[str] | None = None,
) -> dict[str, Any]:
    project_root = detect_project_root(config.input_dir)
    output_dir = config.output_dir
    prepare_output_dirs(output_dir)
    run_id = uuid.uuid4().hex[:12]

    audit_path = output_dir / "audit" / "direct_target_tabicl_run_audit.jsonl"
    audit = E5IncrementalAuditLogger(audit_path)
    target_audit = E5TargetNormalizationAudit(
        output_dir / "audit" / "direct_target_tabicl_target_normalization_audit.csv"
    )
    state = RunState()
    context_cache = BoundedContextCache(
        mode=config.context_cache_mode,
        max_entries=config.max_context_cache_entries,
        audit=audit,
    )

    seed_status = propagate_random_state(config.random_state)
    audit.log("run_started", run_id=run_id, random_state=config.random_state, seed_status=seed_status)

    dependency_audit = run_dependency_audit(
        project_root=project_root,
        e3_root=config.input_dir,
        e4a_root=config.e4a_root,
        fail_on_missing_registry=config.fail_on_missing_e4a_registry,
        allow_registry_warnings=config.allow_registry_warnings,
    )
    dep_path = output_dir / "audit" / "direct_target_tabicl_dependency_audit.json"
    atomic_write_json(dependency_audit, dep_path)
    audit.log(
        "dependency_audit",
        status=dependency_audit.get("status"),
        failure_reason=dependency_audit.get("failure_reason"),
    )

    if dependency_audit.get("status") == "fail":
        finalize_run_outputs(
            config=config,
            output_dir=output_dir,
            state=state,
            audit=audit,
            descriptors=[],
            dependency_audit=dependency_audit,
            registration_status=None,
            run_id=run_id,
            seed_status=seed_status,
            context_cache=context_cache,
        )
        return {
            "status": "dependency_failed",
            "dependency_audit": dependency_audit,
            "run_id": run_id,
        }

    manifest = load_e3_manifest(e3_manifest_path) if e3_manifest_path else None
    descriptors = config.selected_descriptors
    if descriptors is None:
        all_desc = discover_direct_target_datasets(config.input_dir, manifest=manifest)
        descriptors = filter_descriptors_for_mode(
            all_desc, config, filters=filters, horizons=horizons, exit_policies=exit_policies
        )

    jobs = build_job_list(descriptors, config)
    state.submitted_jobs = len(jobs)

    def worker(job: dict[str, Any]) -> dict[str, Any]:
        result = evaluate_single_job(
            job,
            config=config,
            output_dir=output_dir,
            audit=audit,
            context_cache=context_cache,
            project_root=project_root,
            run_id=run_id,
            target_norm_audit=target_audit,
        )
        state.processed_jobs += 1
        if result.get("status") == "completed":
            state.successful_jobs += 1
            if result.get("policy_rows"):
                state.policy_grid_rows.extend(result["policy_rows"])
            if result.get("context_audit"):
                state.context_audit_rows.append(result["context_audit"])
            if result.get("feature_audit"):
                state.feature_audit_rows.append(result["feature_audit"])
            if result.get("join_diagnostics"):
                state.join_diagnostics.extend(result["join_diagnostics"])
            for row in result.get("comparison_rows") or []:
                state.comparison_rows.append(row)
            for frame in result.get("consensus_frames") or []:
                state.consensus_rows.extend(frame.to_dict("records"))
        elif result.get("status") == "skipped":
            state.skipped_jobs.append(result)
        else:
            state.failed_jobs.append(result)
        return result

    run_bounded_executor(jobs, worker, max_workers=config.max_workers, audit=audit)

    registration_status = None
    if config.register_artifacts:
        registration_status = register_e5_artifacts(project_root, output_dir)
        audit.log("artifact_registration", **(registration_status or {}))

    finalize_run_outputs(
        config=config,
        output_dir=output_dir,
        state=state,
        audit=audit,
        descriptors=descriptors,
        dependency_audit=dependency_audit,
        registration_status=registration_status,
        run_id=run_id,
        seed_status=seed_status,
        context_cache=context_cache,
    )
    return {
        "status": "completed",
        "run_id": run_id,
        "dependency_audit": dependency_audit,
        "datasets_completed": state.successful_jobs,
        "successful_jobs": state.successful_jobs,
        "processed_jobs": state.processed_jobs,
        "skipped_jobs": len(state.skipped_jobs),
        "failed_jobs": len(state.failed_jobs),
        "registration": registration_status,
    }
