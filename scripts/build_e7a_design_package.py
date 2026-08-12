#!/usr/bin/env python3
"""E7A offline target redesign and pair-generalization robustness audit (design only)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

IDENTITY_COLUMNS = ["candidate_id", "candidate_policy_id", "target_row_id"]
DRAWDOWN_COLUMNS = [
    "max_drawdown_during_hold",
    "max_drawdown",
    "drawdown",
    "max_drawdown_pct",
    "drawdown_during_hold",
]
AUDIT_COLUMNS = [
    "split",
    "pair_address",
    "target",
    "target_net_profitable_after_exit",
    "sim_net_return",
    "label_valid",
    "filter",
    "horizon",
    "exit_policy_id",
    "tp_ratio",
    "sl_ratio",
    "round_trip_fee_pct",
    *IDENTITY_COLUMNS,
]

INPUT_PATHS: list[tuple[str, bool, str]] = [
    ("data/training/manual_verified_datasets_direct_target_v1", True, "E3 direct-target datasets"),
    (
        "data/training/manual_verified_results/phase_e4_direct_target_xgb_rf_full_20260630_195312",
        False,
        "E4A XGB/RF outputs",
    ),
    (
        "data/training/manual_verified_results/phase_e5_direct_target_tabicl_20260703_203824",
        False,
        "E5 focused TAB outputs",
    ),
    (
        "data/training/manual_verified_results/phase_e5_direct_target_tabicl_20260703_223609",
        False,
        "E5 RAW_ALL targeted TAB outputs",
    ),
    ("data/audits/e5d_outlier_attribution_summary.txt", False, "E5D outlier attribution"),
    ("data/audits/e5d_return_robustness_gate_matrix.csv", False, "E5D return robustness gates"),
    ("data/audits/e5d_wide_e3_direct_target_split_distribution.csv", False, "E5D E3 split distribution"),
    ("data/audits/e5d_manual_liq_no_whale_duplicate_audit.csv", False, "E5D LIQ/NO_WHALE duplicate audit"),
    (
        "data/training/manual_verified_results/phase_e6r_tabicl_full_recheck_20260704_124645",
        False,
        "E6R full recheck results",
    ),
    ("data/audits/phase_e6r_full_recheck_20260704_124645", False, "E6R robustness audits"),
    ("data/training/artifact_registry/artifact_registry.csv", False, "Artifact registry CSV"),
    ("data/training/artifact_registry/artifact_registry.jsonl", False, "Artifact registry JSONL"),
    ("data/training/artifact_registry/artifact_registry_summary.json", False, "Artifact registry summary"),
]

E6R_ROBUSTNESS_FILES = [
    "data/audits/phase_e6r_full_recheck_20260704_124645/e6r7_candidate_robustness_matrix.csv",
    "data/audits/phase_e6r_full_recheck_20260704_124645/e6r7_split_robustness_stats.csv",
    "data/audits/phase_e6r_full_recheck_20260704_124645/e6r7_pair_concentration_stats.csv",
]

E6R_SIGNAL_FILES = [
    "data/audits/phase_e6r_full_recheck_20260704_124645/e6r6_by_tier.csv",
    "data/audits/phase_e6r_full_recheck_20260704_124645/e6r6_tab_rf_only_ranked.csv",
    "data/audits/phase_e6r_full_recheck_20260704_124645/e6r6_positive_both_ranked.csv",
]

SIGNAL_FAMILIES = [
    "TAB_XGB_RF_ALL3",
    "TAB_RF_ONLY",
    "TAB_XGB_ONLY",
    "XGB_RF_ONLY",
    "TAB_ONLY",
    "XGB_ONLY",
    "RF_ONLY",
    "XGB standalone",
    "RF standalone",
    "TAB standalone",
]

POLICY_STATUSES = [
    "USABLE_OFFLINE",
    "RESEARCH_ONLY",
    "UNUSABLE_OFFLINE",
    "INSUFFICIENT_EVIDENCE",
    "DIAGNOSTIC_ONLY",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ts_slug(dt: datetime | None = None) -> str:
    dt = dt or utc_now()
    return dt.strftime("%Y%m%d_%H%M%S")


def rel_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def log_event(stream_path: Path, event: str, **payload: Any) -> None:
    record = {"ts": utc_now().isoformat(), "event": event, **payload}
    with stream_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def load_registry_index() -> dict[str, dict[str, Any]]:
    registry_path = ROOT / "data/training/artifact_registry/artifact_registry.jsonl"
    index: dict[str, dict[str, Any]] = {}
    if not registry_path.exists():
        return index
    with registry_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("project_relative_path") or row.get("path", "")
            key = str(key).replace("\\", "/")
            index[key] = row
    return index


def discover_e3_datasets(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Required E3 root missing: {root}")
    by_stem: dict[str, Path] = {}
    for path in sorted(root.iterdir()):
        if path.suffix not in {".parquet", ".csv"}:
            continue
        stem = path.stem
        existing = by_stem.get(stem)
        if existing is None or path.suffix == ".parquet":
            by_stem[stem] = path
    datasets = sorted(by_stem.values(), key=lambda p: p.name)
    if len(datasets) != 40:
        raise RuntimeError(f"Expected 40 E3 datasets, found {len(datasets)} under {root}")
    return datasets


def available_columns(path: Path) -> list[str]:
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        return pq.ParquetFile(path).schema.names
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def iter_column_batches(path: Path, columns: list[str], chunksize: int = 100_000) -> Iterator[pd.DataFrame]:
    present = [c for c in columns if c in available_columns(path)]
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=chunksize, columns=present):
            yield batch.to_pandas()
        return
    for chunk in pd.read_csv(path, usecols=present, chunksize=chunksize, low_memory=False):
        yield chunk


def parse_dataset_metadata(path: Path, sample: pd.DataFrame) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "file": path.name,
        "filter": None,
        "horizon": None,
        "exit_policy_id": None,
        "tp_ratio": None,
        "sl_ratio": None,
        "round_trip_fee_pct": None,
    }
    if not sample.empty:
        for col in ["filter", "horizon", "exit_policy_id", "tp_ratio", "sl_ratio", "round_trip_fee_pct"]:
            if col in sample.columns:
                values = sample[col].dropna()
                if not values.empty:
                    meta[col] = values.iloc[0]
    if meta["filter"] is None:
        match = re.match(
            r"^(RAW_ALL_VERIFIED|LIQ_5K_HIGH_ACTIVITY|NO_WHALE_FILTER|LOW_LIQ_MOMENTUM)_"
            r"(\d+m|\d+h)_",
            path.stem,
        )
        if match:
            meta["filter"] = match.group(1)
            meta["horizon"] = match.group(2)
    if meta["exit_policy_id"] is None:
        match = re.search(r"(TP\d+_SL\d+_FEE\d+_TIME_BY_HORIZON)", path.stem)
        if match:
            meta["exit_policy_id"] = match.group(1)
    return meta


def positive_mask(frame: pd.DataFrame) -> pd.Series:
    if "target_net_profitable_after_exit" in frame.columns:
        col = frame["target_net_profitable_after_exit"]
        if col.dtype == object:
            return col.astype(str).str.lower().isin({"1", "true", "yes"})
        return col.fillna(0).astype(float) > 0
    if "target" in frame.columns:
        return frame["target"].fillna(0).astype(float) > 0
    return pd.Series([False] * len(frame), index=frame.index)


def valid_target_mask(frame: pd.DataFrame) -> pd.Series:
    if "label_valid" in frame.columns:
        col = frame["label_valid"]
        if col.dtype == object:
            return col.astype(str).str.lower().isin({"1", "true", "yes"})
        return col.fillna(False).astype(bool)
    return pd.Series([True] * len(frame), index=frame.index)


def top_pair_share(pairs: pd.Series) -> float:
    if pairs.empty:
        return 0.0
    counts = pairs.value_counts()
    return float(counts.iloc[0] / counts.sum())


def audit_e3_dataset(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    cols = available_columns(path)
    schema_row = {
        "file": path.name,
        "path": rel_path(path),
        "format": path.suffix.lstrip("."),
        "row_count": 0,
        "column_count": len(cols),
        "identity_columns_present": "|".join(c for c in IDENTITY_COLUMNS if c in cols),
        "identity_columns_missing": "|".join(c for c in IDENTITY_COLUMNS if c not in cols),
        "sim_net_return_present": "sim_net_return" in cols,
        "target_net_profitable_after_exit_present": "target_net_profitable_after_exit" in cols,
        "drawdown_columns_present": "|".join(c for c in DRAWDOWN_COLUMNS if c in cols) or "none",
        "drawdown_columns_missing": "all" if not any(c in cols for c in DRAWDOWN_COLUMNS) else "",
    }

    split_stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "valid_target_rows": 0,
            "positive_rows": 0,
            "pairs": set(),
        }
    )
    sample_meta = pd.DataFrame()
    read_cols = [c for c in AUDIT_COLUMNS if c in cols]

    for chunk in iter_column_batches(path, read_cols):
        if sample_meta.empty:
            sample_meta = chunk.head(1)
        schema_row["row_count"] += len(chunk)
        if "split" not in chunk.columns:
            split_stats["all"]["rows"] += len(chunk)
            continue
        valid = valid_target_mask(chunk)
        pos = positive_mask(chunk) & valid
        for split_value, group in chunk.groupby("split", dropna=False):
            key = str(split_value)
            split_stats[key]["rows"] += len(group)
            split_stats[key]["valid_target_rows"] += int(valid.loc[group.index].sum())
            split_stats[key]["positive_rows"] += int(pos.loc[group.index].sum())
            if "pair_address" in group.columns:
                split_stats[key]["pairs"].update(group["pair_address"].dropna().astype(str).tolist())

    meta = parse_dataset_metadata(path, sample_meta)
    schema_row.update(
        {
            "filter": meta["filter"],
            "horizon": meta["horizon"],
            "exit_policy_id": meta["exit_policy_id"],
            "tp_ratio": meta["tp_ratio"],
            "sl_ratio": meta["sl_ratio"],
            "round_trip_fee_pct": meta["round_trip_fee_pct"],
        }
    )

    distribution_rows: list[dict[str, Any]] = []
    pair_sets = {split: stats["pairs"] for split, stats in split_stats.items()}
    train_pairs = pair_sets.get("train", set())
    val_pairs = pair_sets.get("validation", set())
    test_pairs = pair_sets.get("test", set())

    overlap_row = {
        "file": path.name,
        "filter": meta["filter"],
        "horizon": meta["horizon"],
        "exit_policy_id": meta["exit_policy_id"],
        "train_unique_pairs": len(train_pairs),
        "validation_unique_pairs": len(val_pairs),
        "test_unique_pairs": len(test_pairs),
        "train_validation_overlap_pairs": len(train_pairs & val_pairs),
        "train_test_overlap_pairs": len(train_pairs & test_pairs),
        "validation_test_overlap_pairs": len(val_pairs & test_pairs),
        "all_splits_overlap_pairs": len(train_pairs & val_pairs & test_pairs),
        "train_validation_overlap_share_of_validation": (
            len(train_pairs & val_pairs) / len(val_pairs) if val_pairs else 0.0
        ),
        "train_test_overlap_share_of_test": (
            len(train_pairs & test_pairs) / len(test_pairs) if test_pairs else 0.0
        ),
    }

    for split, stats in sorted(split_stats.items()):
        pairs_series = pd.Series(list(stats["pairs"]))
        distribution_rows.append(
            {
                "file": path.name,
                "filter": meta["filter"],
                "horizon": meta["horizon"],
                "exit_policy_id": meta["exit_policy_id"],
                "split": split,
                "rows": stats["rows"],
                "valid_target_rows": stats["valid_target_rows"],
                "positive_rows": stats["positive_rows"],
                "positive_rate": (
                    stats["positive_rows"] / stats["valid_target_rows"]
                    if stats["valid_target_rows"]
                    else 0.0
                ),
                "unique_pairs": len(stats["pairs"]),
                "top_pair_share": top_pair_share(pairs_series),
            }
        )

    return schema_row, distribution_rows, overlap_row


def inventory_inputs(registry: dict[str, dict[str, Any]], used_paths: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rel, required, notes in INPUT_PATHS:
        path = ROOT / rel
        exists = path.exists()
        entry: dict[str, Any] = {
            "relative_path": rel,
            "exists": exists,
            "type": "missing",
            "size_bytes": "",
            "file_count": "",
            "required": required,
            "used": rel in used_paths or any(rel in u for u in used_paths),
            "registry_hit": rel in registry,
            "notes": notes,
        }
        if exists:
            if path.is_file():
                entry["type"] = "file"
                entry["size_bytes"] = path.stat().st_size
            else:
                entry["type"] = "directory"
                entry["file_count"] = sum(1 for _ in path.rglob("*") if _.is_file())
        rows.append(entry)

    for rel in E6R_ROBUSTNESS_FILES + E6R_SIGNAL_FILES:
        path = ROOT / rel
        rows.append(
            {
                "relative_path": rel,
                "exists": path.exists(),
                "type": "file" if path.exists() else "missing",
                "size_bytes": path.stat().st_size if path.exists() else "",
                "file_count": "",
                "required": False,
                "used": rel in used_paths,
                "registry_hit": rel in registry,
                "notes": "E6R small robustness/signal summary",
            }
        )
    return rows


def slug_or_hash(*parts: Any) -> str:
    text = "|".join("" if p is None else str(p) for p in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:48]
    return f"{slug}_{digest}" if slug else digest


def target_family_matrix() -> list[dict[str, Any]]:
    return [
        {
            "target_family_id": "A",
            "target_name": "binary_direct_exit_target",
            "target_type": "binary",
            "required_columns": "target_net_profitable_after_exit|target|label_valid",
            "feasible_from_current_E3": True,
            "deferred_reason": "",
            "leakage_risk": "low",
            "outlier_sensitivity": "low",
            "winsorization_strategy": "none",
            "identity_preservation_requirement": "preserve target_row_id and candidate_policy_id on every labeled row",
            "expected_modeling_method": "XGB/RF classifier baseline; TAB optional later",
            "expected_evaluation_metric": "precision_at_k, avg_net_return, robustness gates",
            "allowed_in_E7B": True,
            "diagnostic_only": False,
            "notes": "Baseline comparator; already used in E3-E6R.",
        },
        {
            "target_family_id": "B",
            "target_name": "continuous_sim_net_return",
            "target_type": "continuous",
            "required_columns": "sim_net_return|label_valid",
            "feasible_from_current_E3": True,
            "deferred_reason": "",
            "leakage_risk": "medium",
            "outlier_sensitivity": "high",
            "winsorization_strategy": "train-only percentile clip recommended before modeling",
            "identity_preservation_requirement": "preserve target_row_id; join predictions by target_row_id",
            "expected_modeling_method": "XGB/RF regressor",
            "expected_evaluation_metric": "spearman_rank_corr, selected-trade net return, robustness gates",
            "allowed_in_E7B": True,
            "diagnostic_only": False,
            "notes": "Primary continuous target candidate.",
        },
        {
            "target_family_id": "C",
            "target_name": "clipped_sim_net_return",
            "target_type": "continuous",
            "required_columns": "sim_net_return|label_valid|split",
            "feasible_from_current_E3": True,
            "deferred_reason": "",
            "leakage_risk": "medium",
            "outlier_sensitivity": "medium",
            "winsorization_strategy": "fit train-only lower/upper quantile thresholds (default p01/p99); apply same thresholds to validation/test",
            "identity_preservation_requirement": "store clip thresholds keyed by dataset_id + target_family_id; preserve target_row_id",
            "expected_modeling_method": "XGB/RF regressor on clipped label",
            "expected_evaluation_metric": "selected-trade economics + concentration gates",
            "allowed_in_E7B": True,
            "diagnostic_only": False,
            "notes": "Preferred robust continuous variant for E7B.",
        },
        {
            "target_family_id": "D",
            "target_name": "ranked_sim_net_return",
            "target_type": "rank/percentile",
            "required_columns": "sim_net_return|label_valid|split",
            "feasible_from_current_E3": True,
            "deferred_reason": "",
            "leakage_risk": "medium",
            "outlier_sensitivity": "low",
            "winsorization_strategy": "optional train-only preclip before rank; rank computed within split only for labels, never using val/test rows to fit rank mapping",
            "identity_preservation_requirement": "preserve target_row_id; store rank definition metadata separately",
            "expected_modeling_method": "XGB/RF regressor or rank-aware ranker",
            "expected_evaluation_metric": "ndcg_at_k proxy, selected-trade economics",
            "allowed_in_E7B": True,
            "diagnostic_only": False,
            "notes": "Use split-local ranks for labels; do not pool val/test into rank fit.",
        },
        {
            "target_family_id": "E",
            "target_name": "exit_status_aware_return",
            "target_type": "diagnostic_hybrid",
            "required_columns": "sim_net_return|sim_exit_status|label_valid",
            "feasible_from_current_E3": True,
            "deferred_reason": "",
            "leakage_risk": "medium",
            "outlier_sensitivity": "high",
            "winsorization_strategy": "analyze by exit_status bucket; clip within bucket on train only",
            "identity_preservation_requirement": "preserve target_row_id; keep sim_exit_status as metadata not join key",
            "expected_modeling_method": "diagnostic only in E7B",
            "expected_evaluation_metric": "bucketed return stability",
            "allowed_in_E7B": False,
            "diagnostic_only": True,
            "notes": "Design/diagnostic target only.",
        },
        {
            "target_family_id": "F",
            "target_name": "drawdown_adjusted_return",
            "target_type": "continuous",
            "required_columns": "sim_net_return|max_drawdown_during_hold",
            "feasible_from_current_E3": False,
            "deferred_reason": "Drawdown columns absent from current E3 artifacts; requires new controlled drawdown artifact",
            "leakage_risk": "high_if_reconstructed",
            "outlier_sensitivity": "high",
            "winsorization_strategy": "deferred",
            "identity_preservation_requirement": "n/a until drawdown artifact exists",
            "expected_modeling_method": "deferred",
            "expected_evaluation_metric": "deferred",
            "allowed_in_E7B": False,
            "diagnostic_only": True,
            "notes": "Do not reconstruct drawdown in E7A/E7B.",
        },
        {
            "target_family_id": "G",
            "target_name": "old_x2_or_momentum_style_comparator",
            "target_type": "diagnostic_comparator",
            "required_columns": "legacy momentum/x2 features",
            "feasible_from_current_E3": False,
            "deferred_reason": "Not part of direct-target E3 label contract",
            "leakage_risk": "unknown",
            "outlier_sensitivity": "unknown",
            "winsorization_strategy": "n/a",
            "identity_preservation_requirement": "if used, still join via target_row_id",
            "expected_modeling_method": "diagnostic comparator only",
            "expected_evaluation_metric": "comparison against direct targets",
            "allowed_in_E7B": False,
            "diagnostic_only": True,
            "notes": "Do not promote as production target.",
        },
    ]


def pair_generalization_matrix() -> list[dict[str, Any]]:
    return [
        {
            "strategy_id": "existing_temporal_split_baseline",
            "what_it_tests": "Standard chronological train/validation/test economics under current E3 split",
            "what_it_does_not_test": "Unseen pair generalization; pair memory effects",
            "temporal_leakage_risk": "low",
            "pair_memory_test_strength": "none",
            "compute_cost": "low",
            "required_columns": "split|pair_address|sim_net_return|target",
            "allowed_in_E7B": True,
            "implementation_priority": 1,
            "notes": "Mandatory baseline for all target families.",
        },
        {
            "strategy_id": "temporal_split_plus_pair_overlap_diagnostic",
            "what_it_tests": "Quantifies pair overlap across splits as a robustness risk flag",
            "what_it_does_not_test": "Causal pair generalization",
            "temporal_leakage_risk": "low",
            "pair_memory_test_strength": "diagnostic",
            "compute_cost": "low",
            "required_columns": "split|pair_address",
            "allowed_in_E7B": True,
            "implementation_priority": 2,
            "notes": "Already evidenced: overlap exists and must be reported.",
        },
        {
            "strategy_id": "temporal_split_plus_leave_one_best_pair_out",
            "what_it_tests": "Whether economics survive removing highest-return pair",
            "what_it_does_not_test": "Future unseen pairs",
            "temporal_leakage_risk": "low",
            "pair_memory_test_strength": "medium",
            "compute_cost": "low",
            "required_columns": "pair_address|sim_net_return|selected trade outputs",
            "allowed_in_E7B": True,
            "implementation_priority": 3,
            "notes": "Maps to validation/test_remove_best_pair_pass gates.",
        },
        {
            "strategy_id": "temporal_split_plus_leave_one_top_pair_out",
            "what_it_tests": "Whether economics survive removing most frequent pair by row count",
            "what_it_does_not_test": "Unseen pair universe",
            "temporal_leakage_risk": "low",
            "pair_memory_test_strength": "medium",
            "compute_cost": "low",
            "required_columns": "pair_address|row counts",
            "allowed_in_E7B": True,
            "implementation_priority": 4,
            "notes": "Complements best-return pair removal.",
        },
        {
            "strategy_id": "temporal_split_plus_leave_one_best_trade_out",
            "what_it_tests": "Whether economics survive removing single best trade",
            "what_it_does_not_test": "Pair-level generalization",
            "temporal_leakage_risk": "low",
            "pair_memory_test_strength": "low",
            "compute_cost": "low",
            "required_columns": "sim_net_return|target_row_id",
            "allowed_in_E7B": True,
            "implementation_priority": 3,
            "notes": "Maps to remove_best_trade_pass gates.",
        },
        {
            "strategy_id": "pair_disjoint_validation_diagnostic",
            "what_it_tests": "Performance when validation pairs are absent from train",
            "what_it_does_not_test": "Test-set deployment economics under current split",
            "temporal_leakage_risk": "medium_if_mis_split",
            "pair_memory_test_strength": "high",
            "compute_cost": "medium",
            "required_columns": "split|pair_address|event_timestamp",
            "allowed_in_E7B": True,
            "implementation_priority": 5,
            "notes": "Diagnostic only; do not replace primary temporal test.",
        },
        {
            "strategy_id": "purged_grouped_temporal_split_design",
            "what_it_tests": "Temporal split with pair purge window to reduce leakage",
            "what_it_does_not_test": "Live market regime shift",
            "temporal_leakage_risk": "low_when_purged",
            "pair_memory_test_strength": "high",
            "compute_cost": "medium",
            "required_columns": "split|pair_address|event_timestamp",
            "allowed_in_E7B": True,
            "implementation_priority": 6,
            "notes": "Preferred advanced design if E7B expands robustness.",
        },
        {
            "strategy_id": "GroupKFold_diagnostic_only",
            "what_it_tests": "Pair-held-out folds for memorization screening",
            "what_it_does_not_test": "Trading-time temporal ordering unless combined with purge",
            "temporal_leakage_risk": "high_if_unpurged",
            "pair_memory_test_strength": "high",
            "compute_cost": "medium",
            "required_columns": "pair_address|event_timestamp",
            "allowed_in_E7B": False,
            "implementation_priority": 99,
            "notes": "Diagnostic only; not primary trading validation.",
        },
    ]


def robustness_gate_spec() -> list[dict[str, Any]]:
    gates = [
        ("validation_positive_return", "validation_total_net_return", "> 0", True),
        ("test_positive_return", "test_total_net_return", "> 0", True),
        ("positive_both_after_pair_cap", "positive_both_after_pair_cap", "== True", True),
        ("validation_remove_best_trade_pass", "validation_remove_best_trade_pass", "== True", True),
        ("test_remove_best_trade_pass", "test_remove_best_trade_pass", "== True", True),
        ("validation_remove_best_pair_pass", "validation_remove_best_pair_pass", "== True", True),
        ("test_remove_best_pair_pass", "test_remove_best_pair_pass", "== True", True),
        ("min_validation_unique_pairs", "validation_unique_pairs", ">= 3", True),
        ("min_test_unique_pairs", "test_unique_pairs", ">= 3", True),
        ("max_validation_top_pair_share", "validation_top_pair_share", "<= 0.75", True),
        ("max_test_top_pair_share", "test_top_pair_share", "<= 0.75", True),
        ("max_single_trade_return_share", "best_trade_return_share", "<= 0.40", True),
        ("max_single_pair_return_share", "best_pair_return_share", "<= 0.40", True),
        ("validation_to_test_return_delta_limit", "abs(validation_avg_net_return - test_avg_net_return)", "<= 0.50", False),
        ("validation_to_test_precision_delta_limit", "abs(validation_precision - test_precision)", "<= 0.25", False),
        ("min_selected_trades", "validation_rows + test_rows", ">= 20", True),
        ("train_validation_test_pair_overlap_diagnostic", "pair_overlap_flags", "report only", False),
        ("no_single_pair_production_pass", "single_pair_dominance_flag", "== False", True),
    ]
    rows = []
    for gate_name, input_columns, default_threshold, required in gates:
        rows.append(
            {
                "gate_name": gate_name,
                "input_columns": input_columns,
                "default_threshold": default_threshold,
                "required_for_USABLE_OFFLINE": required,
                "fail_status": "UNUSABLE_OFFLINE" if required else "RESEARCH_ONLY",
                "fail_behavior": "block policy promotion; emit machine-readable policy_status",
                "offline_only": True,
                "allowed_at_runtime": False,
                "notes": "Runtime may consume precomputed policy_status only.",
            }
        )
    return rows


def identity_preservation_spec() -> dict[str, Any]:
    return {
        "phase": "E7A",
        "hierarchy": {
            "candidate_id": "Event-level identity for a candidate observation.",
            "candidate_policy_id": "Candidate plus policy-context identity (filter/horizon/exit/top_pct/pair_cap context).",
            "target_row_id": "Target-row / label identity; required on every labeled row and prediction row.",
        },
        "e7b_rules": [
            "Every new target-family dataset must preserve target_row_id.",
            "Every E7B prediction artifact must carry target_row_id.",
            "Join TAB/XGB/RF outputs using target_row_id or a documented stable derivative keyed from target_row_id.",
            "Never use malformed E6R candidate_spec_id strings as canonical identity.",
        ],
        "candidate_spec_id_definition": {
            "purpose": "Policy-summary slug only; not a replacement for target_row_id.",
            "formula": "slug_or_hash(consensus_tier, filter, horizon, exit_policy_id, top_pct, pair_cap)",
            "example_fields": [
                "consensus_tier",
                "filter",
                "horizon",
                "exit_policy_id",
                "top_pct",
                "pair_cap",
            ],
        },
        "malformed_identity_warning": "E6R candidate_spec_id values containing Python object repr text are audit defects and must be regenerated in E7B.",
    }


def winsorization_preprocessing_spec() -> dict[str, Any]:
    return {
        "phase": "E7A",
        "applies_to": ["continuous_sim_net_return", "clipped_sim_net_return", "ranked_sim_net_return"],
        "fit_scope": "train split only",
        "default_method": "quantile_clip",
        "default_lower_quantile": 0.01,
        "default_upper_quantile": 0.99,
        "application": "Use train-fitted thresholds unchanged on validation and test",
        "storage": "Persist thresholds in E7B manifest keyed by dataset_id + target_family_id + exit_policy_id",
        "leakage_rule": "Never compute clip or rank thresholds using validation/test rows",
        "identity_rule": "Winsorization must not alter target_row_id or candidate_policy_id",
    }


def map_policy_status(row: pd.Series) -> str:
    val_status = str(row.get("validation_robustness_status", ""))
    test_status = str(row.get("test_robustness_status", ""))
    gate_soft = bool(row.get("e6r7_gate_soft", False))

    if not gate_soft:
        return "UNUSABLE_OFFLINE"
    if val_status == "BEST_PAIR_DOMINATED_FAIL" or test_status == "BEST_PAIR_DOMINATED_FAIL":
        return "UNUSABLE_OFFLINE"
    if val_status == "BEST_TRADE_DOMINATED_FAIL" or test_status == "BEST_TRADE_DOMINATED_FAIL":
        return "UNUSABLE_OFFLINE"
    if val_status == "PAIR_COUNT_TOO_LOW" or test_status == "PAIR_COUNT_TOO_LOW":
        return "UNUSABLE_OFFLINE"

    required_pass = all(
        [
            bool(row.get("validation_remove_best_trade_pass", False)),
            bool(row.get("test_remove_best_trade_pass", False)),
            bool(row.get("validation_remove_best_pair_pass", False)),
            bool(row.get("test_remove_best_pair_pass", False)),
            bool(row.get("validation_positive_return", False)),
            bool(row.get("test_positive_return", False)),
            bool(row.get("positive_both_after_pair_cap", False)),
        ]
    )
    if gate_soft and required_pass:
        return "USABLE_OFFLINE"

    if bool(row.get("validation_positive_return", False)) or bool(row.get("test_positive_return", False)):
        if "CONCENTRATION" in val_status or "CONCENTRATION" in test_status or "DOMINATED" in val_status:
            return "RESEARCH_ONLY"
        return "INSUFFICIENT_EVIDENCE"
    return "UNUSABLE_OFFLINE"


def summarize_e6r_robustness(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    df = pd.read_csv(path, low_memory=False)
    malformed = df["candidate_spec_id"].astype(str).str.contains("<bound method", na=False).sum()
    df["mapped_policy_status"] = df.apply(map_policy_status, axis=1)

    summary: dict[str, Any] = {
        "total_candidates": len(df),
        "malformed_candidate_spec_id_count": int(malformed),
        "mapped_policy_status_counts": df["mapped_policy_status"].value_counts().to_dict(),
    }

    rows: list[dict[str, Any]] = []
    for metric, series in [
        ("consensus_tier", df["consensus_tier"].value_counts()),
        ("e6r7_gate_soft", df["e6r7_gate_soft"].value_counts()),
        ("validation_robustness_status", df["validation_robustness_status"].value_counts()),
        ("test_robustness_status", df["test_robustness_status"].value_counts()),
        ("mapped_policy_status", df["mapped_policy_status"].value_counts()),
    ]:
        for key, count in series.items():
            rows.append({"metric_group": metric, "metric_value": key, "count": int(count), "notes": ""})

    failure_reasons = Counter()
    for col in ["validation_robustness_status", "test_robustness_status"]:
        for status, count in df[col].value_counts().items():
            if status not in {"PASS", "RETURN_ROBUST_PASS"}:
                failure_reasons[str(status)] += int(count)
    for reason, count in failure_reasons.most_common(15):
        rows.append(
            {
                "metric_group": "top_failure_reason",
                "metric_value": reason,
                "count": count,
                "notes": "aggregated validation/test status failures",
            }
        )

    rows.append(
        {
            "metric_group": "summary",
            "metric_value": "total_candidates",
            "count": len(df),
            "notes": json.dumps(summary),
        }
    )
    return rows, summary


def summarize_signal_families() -> list[dict[str, Any]]:
    by_tier_path = ROOT / E6R_SIGNAL_FILES[0]
    ranked_path = ROOT / E6R_SIGNAL_FILES[2]
    by_tier = pd.read_csv(by_tier_path)
    ranked = pd.read_csv(ranked_path)
    robustness_path = ROOT / E6R_ROBUSTNESS_FILES[0]
    robust = pd.read_csv(robustness_path, usecols=["consensus_tier", "e6r7_gate_soft"], low_memory=False)

    tier_lookup = {row["consensus_tier"]: row for _, row in by_tier.iterrows()}
    robust_pass = robust.groupby("consensus_tier")["e6r7_gate_soft"].sum().to_dict()
    robust_total = robust.groupby("consensus_tier")["e6r7_gate_soft"].count().to_dict()

    rows: list[dict[str, Any]] = []
    standalone_map = {
        "XGB standalone": "XGB_RF_ONLY",
        "RF standalone": "XGB_RF_ONLY",
        "TAB standalone": "TAB_ONLY",
    }

    for family in SIGNAL_FAMILIES:
        tier_key = standalone_map.get(family, family)
        tier_row = tier_lookup.get(tier_key)
        candidate_count = int(tier_row["rows"]) if tier_row is not None else ""
        positive_both = int(tier_row["positive_both"]) if tier_row is not None else ""
        pass_count = int(robust_pass.get(tier_key, 0))
        total_count = int(robust_total.get(tier_key, 0))
        fail_count = total_count - pass_count

        if family == "TAB_RF_ONLY":
            include = "diagnostic"
            status = "RESEARCH_ONLY"
            notes = "Positive-both count is tiny (4); concentration blocked in E6R ranked outputs."
        elif family in {"TAB_XGB_RF_ALL3", "XGB_RF_ONLY", "TAB_ONLY", "TAB_XGB_ONLY"}:
            include = "yes" if family in {"XGB_RF_ONLY", "TAB_ONLY"} else "diagnostic"
            status = "RESEARCH_ONLY"
            notes = "Use as E7B control/comparator; robustness gating still required."
        elif family in {"XGB_ONLY", "RF_ONLY"}:
            include = "diagnostic"
            status = "DIAGNOSTIC_ONLY"
            notes = "Not primary E6R consensus tiers; keep as standalone controls if reproduced."
        else:
            include = "diagnostic"
            status = "DIAGNOSTIC_ONLY"
            notes = "Standalone alias for tier summary."

        rows.append(
            {
                "signal_family": family,
                "candidate_count": candidate_count,
                "positive_both_count": positive_both,
                "robustness_pass_count": pass_count,
                "robustness_fail_count": fail_count,
                "typical_failure_reason": "concentration / pair overlap / negative test return",
                "pair_concentration_concerns": "validation top_pair_share often 1.0 on focused filters",
                "include_in_E7B": include,
                "status": status,
                "notes": notes,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def render_target_family_summary(matrix: list[dict[str, Any]], schema_rows: list[dict[str, Any]]) -> str:
    feasible = [r for r in matrix if r["feasible_from_current_E3"]]
    deferred = [r for r in matrix if not r["feasible_from_current_E3"]]
    drawdown_absent = all(
        r.get("drawdown_columns_present") in {"none", "", None} for r in schema_rows
    )
    lines = [
        "# E7A Target Family Summary",
        "",
        "## Feasible from current E3",
    ]
    for row in feasible:
        lines.append(
            f"- **{row['target_name']}** ({row['target_family_id']}): {row['notes']} "
            f"Winsorization: {row['winsorization_strategy']}."
        )
    lines.extend(["", "## Deferred"])
    for row in deferred:
        lines.append(f"- **{row['target_name']}**: {row['deferred_reason']}")
    lines.extend(
        [
            "",
            "## Schema confirmation",
            f"- Audited E3 dataset files: {len(schema_rows)}",
            f"- Drawdown columns absent across audit: {drawdown_absent}",
            "- Identity hierarchy required for E7B: candidate_id, candidate_policy_id, target_row_id",
        ]
    )
    return "\n".join(lines) + "\n"


def render_pair_generalization_summary(matrix: list[dict[str, Any]], overlap_rows: list[dict[str, Any]]) -> str:
    avg_val_overlap = pd.Series([r["train_validation_overlap_share_of_validation"] for r in overlap_rows]).mean()
    avg_test_overlap = pd.Series([r["train_test_overlap_share_of_test"] for r in overlap_rows]).mean()
    lines = [
        "# E7A Pair Generalization Summary",
        "",
        "## Key finding",
        f"- Mean train/validation pair overlap share: {avg_val_overlap:.3f}",
        f"- Mean train/test pair overlap share: {avg_test_overlap:.3f}",
        "- Pair overlap is material; temporal baseline alone is insufficient for pair-generalization claims.",
        "",
        "## Recommended E7B order",
    ]
    for row in sorted(matrix, key=lambda r: r["implementation_priority"]):
        if row["allowed_in_E7B"]:
            lines.append(
                f"- `{row['strategy_id']}` (priority {row['implementation_priority']}): {row['what_it_tests']}"
            )
    lines.extend(
        [
            "",
            "## Not recommended as primary",
            "- `GroupKFold_diagnostic_only` unless paired with temporal purge controls.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_robustness_gate_summary(gates: list[dict[str, Any]], e6r_summary: dict[str, Any]) -> str:
    required = [g for g in gates if g["required_for_USABLE_OFFLINE"]]
    lines = [
        "# E7A Robustness Gate Summary",
        "",
        "## Final policy statuses",
        "- USABLE_OFFLINE",
        "- RESEARCH_ONLY",
        "- UNUSABLE_OFFLINE",
        "- INSUFFICIENT_EVIDENCE",
        "- DIAGNOSTIC_ONLY",
        "",
        "## Rule",
        "If any required robustness gate fails, final policy_status must be UNUSABLE_OFFLINE.",
        "",
        f"## E6R mapping snapshot",
        f"- Total candidates: {e6r_summary.get('total_candidates')}",
        f"- Malformed candidate_spec_id count: {e6r_summary.get('malformed_candidate_spec_id_count')}",
        f"- Mapped policy_status counts: {e6r_summary.get('mapped_policy_status_counts')}",
        "",
        "## Required gates",
    ]
    for gate in required:
        lines.append(
            f"- `{gate['gate_name']}` threshold `{gate['default_threshold']}` -> fail `{gate['fail_status']}`"
        )
    return "\n".join(lines) + "\n"


def render_signal_family_summary(rows: list[dict[str, Any]]) -> str:
    lines = ["# E7A Existing Signal Family Summary", ""]
    for row in rows:
        lines.append(
            f"- **{row['signal_family']}**: candidates={row['candidate_count']}, "
            f"positive_both={row['positive_both_count']}, pass={row['robustness_pass_count']}, "
            f"include_in_E7B={row['include_in_E7B']}, status={row['status']}"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- Do not assume ALL3 must win.",
            "- TAB_RF_ONLY remains diagnostic unless new evidence clears concentration gates.",
            "- Keep TAB_ONLY and XGB_RF_ONLY as useful controls.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_e7b_recommendation(
    matrix: list[dict[str, Any]],
    e6r_summary: dict[str, Any],
    overlap_rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], str]:
    recommendation = {
        "recommends_E7B": True,
        "runtime_demo_ui_trading_blocked": True,
        "anchor_plan_challenged": False,
        "first_target_families": [
            "binary_direct_exit_target",
            "clipped_sim_net_return",
            "continuous_sim_net_return",
            "ranked_sim_net_return",
        ],
        "deferred_target_families": ["drawdown_adjusted_return", "old_x2_or_momentum_style_comparator"],
        "mandatory_robustness_gates": [
            "validation_positive_return",
            "test_positive_return",
            "positive_both_after_pair_cap",
            "validation_remove_best_trade_pass",
            "test_remove_best_trade_pass",
            "validation_remove_best_pair_pass",
            "test_remove_best_pair_pass",
            "min_validation_unique_pairs",
            "min_test_unique_pairs",
            "max_validation_top_pair_share",
            "max_test_top_pair_share",
            "no_single_pair_production_pass",
        ],
        "model_scope": {
            "start_with": "compact XGB/RF target-family comparison",
            "tab": "delay until target methodology narrowed; optional second wave",
        },
        "filter_horizon_priority": [
            "LIQ_5K_HIGH_ACTIVITY and NO_WHALE_FILTER treated as one duplicated family",
            "prioritize 4h/8h horizons first",
            "LOW_LIQ_MOMENTUM diagnostic only due to concentration",
            "RAW_ALL diagnostic subset only in E7B, not full 40-grid expansion",
        ],
        "identity_preservation": {
            "required_columns": IDENTITY_COLUMNS,
            "prediction_join_key": "target_row_id",
            "candidate_spec_id": "policy summary slug only",
        },
        "artifact_explosion_controls": [
            "one manifest per E7B run",
            "aggregate CSV/JSON summaries only",
            "no row-level selected-trade dumps unless explicitly gated",
        ],
        "minimum_reopen_criteria": [
            "E7B target-family comparison complete for prioritized filters/horizons",
            "robustness gate spec applied with machine-readable policy_status outputs",
            "no promotion without USABLE_OFFLINE offline status",
        ],
        "before_runtime_integration": [
            "Independent replay of E7B selected policies",
            "Stable non-malformed candidate_spec_id generation",
            "Explicit offline-to-runtime separation review",
        ],
        "e6r_mapping_snapshot": e6r_summary,
        "pair_overlap_mean_train_test_share": float(
            pd.Series([r["train_test_overlap_share_of_test"] for r in overlap_rows]).mean()
        ),
    }

    md = "\n".join(
        [
            "# E7A E7B Recommendation",
            "",
            "## Proceed to E7B",
            "Yes. E7A recommends a compact offline E7B implementation pass.",
            "",
            "## Target families first",
            "- Binary direct exit target (comparator)",
            "- Clipped sim_net_return (primary continuous)",
            "- Continuous sim_net_return",
            "- Ranked sim_net_return",
            "",
            "## Deferred",
            "- drawdown_adjusted_return (missing drawdown artifact)",
            "- old_x2_or_momentum_style_comparator (diagnostic only)",
            "",
            "## Models",
            "- Start with XGB/RF only for target-family comparison.",
            "- Delay TAB until cheaper models narrow target methodology.",
            "",
            "## Filters / horizons",
            "- Treat NO_WHALE_FILTER / LIQ_5K_HIGH_ACTIVITY duplication as one family in E7B.",
            "- Prioritize 4h and 8h horizons.",
            "- LOW_LIQ_MOMENTUM: diagnostic only.",
            "- RAW_ALL: small diagnostic subset only; do not expand to full grid in first E7B wave.",
            "",
            "## Robustness",
            "- Apply mandatory gates before any selected policy.",
            "- Failed required gates map to UNUSABLE_OFFLINE.",
            "",
            "## Identity",
            "- Preserve target_row_id and candidate_policy_id in all E7B datasets and predictions.",
            "- Regenerate clean candidate_spec_id for policy summaries only.",
            "",
            "## Still blocked",
            "- Runtime, demo, UI, trading integration remain closed after E7A.",
            "",
            "## Anchor plan",
            "Not challenged; E7A narrows E7B scope and strengthens robustness gating.",
        ]
    ) + "\n"
    return recommendation, md


def build_summary_text(
    timestamp: str,
    output_root: Path,
    audit_root: Path,
    files_created: list[str],
    commands_run: list[str],
    tests_run: list[str],
    key_results: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "E7A Offline Target Redesign & Pair-Generalization Robustness Audit",
            f"Phase: E7A",
            f"Branch: phase_e7a_target_redesign_pair_generalization",
            "",
            "Original task:",
            "Compact offline audit/design pass for target families, pair-generalization, and robustness gates.",
            "",
            "What changed:",
            "- Added scripts/build_e7a_design_package.py",
            "- Generated new E7A audit/design outputs only",
            "",
            "Files created:",
            *[f"- {path}" for path in files_created],
            "",
            "What was not changed:",
            "- No runtime/trading/UI/API/DB/Solana/Helius/Gemini/Qwen changes",
            "- No mutation of existing E3/E4/E5/E6R artifacts",
            "- No model training or retraining",
            "",
            "Key results:",
            f"- E3 datasets audited: {key_results.get('dataset_count')}",
            f"- Drawdown-adjusted target deferred: {key_results.get('drawdown_deferred')}",
            f"- E6R candidates mapped: {key_results.get('e6r_candidates')}",
            f"- E7B recommended: {key_results.get('recommends_E7B')}",
            "",
            "Unexpected findings:",
            f"- {key_results.get('unexpected')}",
            "",
            "Anchor Plan challenged:",
            str(key_results.get("anchor_plan_challenged", False)),
            "",
            "Branch recommendation:",
            "phase_e7a_target_redesign_pair_generalization",
            "",
            "Exact output paths:",
            f"- {rel_path(output_root)}",
            f"- {rel_path(audit_root)}",
            "",
            "Exact commands run:",
            *[f"- {cmd}" for cmd in commands_run],
            "",
            "Tests run and results:",
            *[f"- {test}" for test in tests_run],
            "",
            "E7A recommends E7B:",
            str(key_results.get("recommends_E7B", True)),
            "",
            "Runtime/demo/UI/trading remain blocked:",
            "True",
        ]
    )


def run_e7a(timestamp: str | None = None) -> tuple[Path, Path]:
    stamp = timestamp or ts_slug()
    output_root = ROOT / f"data/training/manual_verified_results/phase_e7a_target_redesign_pair_generalization_{stamp}"
    audit_root = ROOT / f"data/audits/phase_e7a_target_redesign_pair_generalization_{stamp}"
    for sub in ["reports", "design", "audits", "logs", "manifests"]:
        (output_root / sub).mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)

    stream_path = output_root / "logs/e7a_audit_stream.jsonl"
    registry = load_registry_index()
    used_paths = {
        INPUT_PATHS[0][0],
        *E6R_ROBUSTNESS_FILES,
        *E6R_SIGNAL_FILES,
        "data/audits/e5d_wide_e3_direct_target_split_distribution.csv",
        "data/audits/e5d_manual_liq_no_whale_duplicate_audit.csv",
    }

    log_event(stream_path, "start", phase="E7A", timestamp=stamp)

    inventory = inventory_inputs(registry, used_paths)
    write_csv(output_root / "audits/e7a_input_artifact_inventory.csv", inventory)
    log_event(stream_path, "inventory_complete", rows=len(inventory))

    e3_root = ROOT / INPUT_PATHS[0][0]
    datasets = discover_e3_datasets(e3_root)
    schema_rows: list[dict[str, Any]] = []
    distribution_rows: list[dict[str, Any]] = []
    overlap_rows: list[dict[str, Any]] = []

    for index, dataset in enumerate(datasets, start=1):
        schema_row, dist_rows, overlap_row = audit_e3_dataset(dataset)
        schema_rows.append(schema_row)
        distribution_rows.extend(dist_rows)
        overlap_rows.append(overlap_row)
        log_event(stream_path, "dataset_audited", file=dataset.name, index=index, total=len(datasets))

    write_csv(output_root / "audits/e7a_direct_target_schema_audit.csv", schema_rows)
    write_csv(output_root / "audits/e7a_direct_target_distribution_by_dataset.csv", distribution_rows)
    write_csv(output_root / "audits/e7a_pair_split_overlap_summary.csv", overlap_rows)

    target_matrix = target_family_matrix()
    write_csv(output_root / "design/e7a_target_family_design_matrix.csv", target_matrix)
    write_json(output_root / "design/e7a_target_family_design_matrix.json", target_matrix)
    write_json(output_root / "design/e7a_identity_preservation_spec.json", identity_preservation_spec())
    write_json(
        output_root / "design/e7a_winsorization_preprocessing_spec.json",
        winsorization_preprocessing_spec(),
    )

    pair_matrix = pair_generalization_matrix()
    write_csv(output_root / "design/e7a_pair_generalization_design_matrix.csv", pair_matrix)

    gate_rows = robustness_gate_spec()
    write_csv(output_root / "design/e7a_robustness_gate_spec.csv", gate_rows)
    write_json(output_root / "design/e7a_robustness_gate_spec.json", gate_rows)

    e6r_rows, e6r_summary = summarize_e6r_robustness(ROOT / E6R_ROBUSTNESS_FILES[0])
    write_csv(output_root / "audits/e7a_existing_robustness_status_mapping.csv", e6r_rows)

    signal_rows = summarize_signal_families()
    write_csv(output_root / "audits/e7a_existing_signal_family_audit.csv", signal_rows)

    recommendation, recommendation_md = build_e7b_recommendation(target_matrix, e6r_summary, overlap_rows)
    (output_root / "reports/e7a_target_family_summary.md").write_text(
        render_target_family_summary(target_matrix, schema_rows),
        encoding="utf-8",
    )
    (output_root / "reports/e7a_pair_generalization_summary.md").write_text(
        render_pair_generalization_summary(pair_matrix, overlap_rows),
        encoding="utf-8",
    )
    (output_root / "reports/e7a_robustness_gate_summary.md").write_text(
        render_robustness_gate_summary(gate_rows, e6r_summary),
        encoding="utf-8",
    )
    (output_root / "reports/e7a_existing_signal_family_summary.md").write_text(
        render_signal_family_summary(signal_rows),
        encoding="utf-8",
    )
    (output_root / "reports/e7a_e7b_recommendation.md").write_text(recommendation_md, encoding="utf-8")
    write_json(output_root / "reports/e7a_e7b_recommendation.json", recommendation)

    files_created = sorted(
        p.relative_to(ROOT).as_posix() for p in output_root.rglob("*") if p.is_file()
    )
    commands_run = [f"python scripts/build_e7a_design_package.py --timestamp {stamp}"]
    tests_run: list[str] = []

    key_results = {
        "dataset_count": len(datasets),
        "drawdown_deferred": True,
        "e6r_candidates": e6r_summary.get("total_candidates"),
        "recommends_E7B": True,
        "anchor_plan_challenged": False,
        "unexpected": "E3 directory serves CSV and/or parquet; audit prefers parquet when both exist. Pair overlap is nonzero across most focused datasets.",
    }
    summary_text = build_summary_text(
        stamp,
        output_root,
        audit_root,
        files_created,
        commands_run,
        tests_run,
        key_results,
    )
    (output_root / "reports/e7a_summary_for_upload.txt").write_text(summary_text, encoding="utf-8")

    manifest = {
        "phase": "E7A",
        "branch_name": "phase_e7a_target_redesign_pair_generalization",
        "created_at": utc_now().isoformat(),
        "output_root": rel_path(output_root),
        "audit_root": rel_path(audit_root),
        "input_roots": [p[0] for p in INPUT_PATHS[:1]],
        "files_created": [],
        "commands_run": commands_run,
        "tests_run": tests_run,
        "status": "complete",
        "recommendation": recommendation,
    }
    files_created = sorted(
        p.relative_to(ROOT).as_posix() for p in output_root.rglob("*") if p.is_file()
    )
    manifest["files_created"] = files_created
    write_json(output_root / "manifests/e7a_manifest.json", manifest)

    # Mirror a lightweight pointer in audit root without exploding artifacts
    (audit_root / "E7A_README.txt").write_text(
        f"E7A outputs live at {rel_path(output_root)}\n",
        encoding="utf-8",
    )

    log_event(stream_path, "complete", files_created=len(files_created))
    return output_root, audit_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build E7A offline design/audit package.")
    parser.add_argument("--timestamp", default=None, help="Override output timestamp slug")
    args = parser.parse_args(argv)
    output_root, audit_root = run_e7a(args.timestamp)
    print(f"E7A complete: {output_root}")
    print(f"Audit root: {audit_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
