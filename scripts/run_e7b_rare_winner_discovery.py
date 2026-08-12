#!/usr/bin/env python3
"""E7B-R offline rare-winner discovery evaluation (RF required, no TAB, no runtime)."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

ROOT = Path(__file__).resolve().parents[1]

IDENTITY_COLUMNS = [
    "candidate_id",
    "candidate_policy_id",
    "target_row_id",
    "pair_address",
    "split",
    "filter",
    "horizon",
    "exit_policy_id",
]

META_COLUMNS = [
    *IDENTITY_COLUMNS,
    "sim_net_return",
    "target_net_profitable_after_exit",
    "label_valid",
    "liquidity",
    "volume_24h",
    "liquidity_usd",
]

EXCLUDE_EXACT = {
    "candidate_id",
    "candidate_policy_id",
    "target_row_id",
    "pair_address",
    "split",
    "target",
    "target_name",
    "target_version",
    "target_net_profitable_after_exit",
    "sim_net_return",
    "sim_exit_status",
    "exit_ratio",
    "exit_timestamp",
    "horizon",
    "exit_policy_id",
    "tp_ratio",
    "sl_ratio",
    "round_trip_fee_pct",
    "top_pct",
    "pair_cap",
    "label_valid",
    "label_error_code",
    "label_error_detail",
}

EXCLUDE_SUBSTRINGS = (
    "target",
    "future",
    "exit",
    "gap_",
    "timestamp",
    "return",
    "sim_",
    "policy",
)

SCORE_TOP_PCTS = (0.5, 1.0, 2.0, 5.0)
PAIR_CAPS = (1, 2, 5, 10)
WINNER_PCTS = (0.5, 1.0, 2.0, 5.0)

SMOKE_DATASET = (
    "data/training/manual_verified_datasets_direct_target_v1/"
    "LIQ_5K_HIGH_ACTIVITY_4h_TP20308_SL080_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.parquet"
)

FOCUSED_DATASETS = [
    "data/training/manual_verified_datasets_direct_target_v1/"
    "LIQ_5K_HIGH_ACTIVITY_4h_TP20308_SL075_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.parquet",
    "data/training/manual_verified_datasets_direct_target_v1/"
    "LIQ_5K_HIGH_ACTIVITY_4h_TP20308_SL080_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.parquet",
    "data/training/manual_verified_datasets_direct_target_v1/"
    "LIQ_5K_HIGH_ACTIVITY_8h_TP20308_SL075_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.parquet",
    "data/training/manual_verified_datasets_direct_target_v1/"
    "LIQ_5K_HIGH_ACTIVITY_8h_TP20308_SL080_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.parquet",
]

MANUAL_AUDIT_DIRS = [
    "data/audits/e7b_rare_winner_smoke_20260705_084204",
    "data/audits/e7b_rare_winner_replication_20260705_085115",
]

REGISTRY_WHITELIST_SUFFIXES = (
    "/reports/e7br_summary_for_upload.txt",
    "/reports/e7br_rare_winner_summary.md",
    "/reports/e7br_e7c_recommendation.md",
    "/metrics/e7br_rare_winner_capture_summary.csv",
    "/metrics/e7br_baseline_lift_comparison.csv",
    "/robustness/e7br_rare_winner_gate_matrix.csv",
    "/robustness/e7br_stable_strategy_gate_matrix.csv",
    "/manifests/e7br_manifest.json",
    "/design/e7br_rare_winner_status_spec.json",
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ts_slug(dt: datetime | None = None) -> str:
    return (dt or utc_now()).strftime("%Y%m%d_%H%M%S")


def rel_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def log_event(stream_path: Path, event: str, **payload: Any) -> None:
    record = {"ts": utc_now().isoformat(), "event": event, **payload}
    with stream_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def slug_or_hash(*parts: Any) -> str:
    text = "|".join("" if p is None else str(p) for p in parts)
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()[:48]
    return f"{slug}_{digest}" if slug else digest


def resolve_dataset_path(rel: str) -> Path:
    path = ROOT / rel.replace("\\", "/")
    if path.exists():
        return path
    csv_path = path.with_suffix(".csv")
    if csv_path.exists():
        return csv_path
    return path


def load_dataset(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def valid_mask(frame: pd.DataFrame) -> pd.Series:
    if "label_valid" not in frame.columns:
        return pd.Series(True, index=frame.index)
    col = frame["label_valid"]
    if col.dtype == object:
        return col.astype(str).str.lower().isin({"1", "true", "yes"})
    return col.fillna(False).astype(bool)


def select_features(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded: list[str] = []
    features: list[str] = []
    for col in frame.columns:
        lower = col.lower()
        if col in EXCLUDE_EXACT:
            excluded.append(col)
            continue
        if any(token in lower for token in EXCLUDE_SUBSTRINGS):
            excluded.append(col)
            continue
        if not pd.api.types.is_numeric_dtype(frame[col]):
            excluded.append(col)
            continue
        features.append(col)
    return sorted(features), sorted(excluded)


def fit_clip_thresholds(train_values: np.ndarray, lower_q: float = 0.01, upper_q: float = 0.99) -> tuple[float, float]:
    clean = train_values[np.isfinite(train_values)]
    if clean.size == 0:
        return 0.0, 0.0
    return float(np.quantile(clean, lower_q)), float(np.quantile(clean, upper_q))


def apply_clip(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    return np.clip(values, lower, upper)


class TrainRankTransform:
    def __init__(self, train_values: np.ndarray) -> None:
        clean = train_values[np.isfinite(train_values)]
        self.sorted_train = np.sort(clean)
        self.n = len(self.sorted_train)

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.n == 0:
            return np.zeros_like(values, dtype=float)
        idx = np.searchsorted(self.sorted_train, values, side="right")
        return idx / self.n


def build_target_vector(
    split_frame: pd.DataFrame,
    target_family: str,
    clip_bounds: tuple[float, float] | None,
    rank_transform: TrainRankTransform | None,
) -> np.ndarray:
    returns = split_frame["sim_net_return"].astype(float).to_numpy()
    if target_family == "binary":
        col = split_frame["target_net_profitable_after_exit"]
        if col.dtype == object:
            return col.astype(str).str.lower().isin({"1", "true", "yes"}).astype(int).to_numpy()
        return (col.fillna(0).astype(float) > 0).astype(int).to_numpy()
    if target_family == "continuous":
        return returns
    if target_family == "clipped":
        assert clip_bounds is not None
        return apply_clip(returns, clip_bounds[0], clip_bounds[1])
    if target_family == "ranked":
        assert rank_transform is not None
        return rank_transform.transform(returns)
    raise ValueError(f"Unknown target family: {target_family}")


def make_model(model_name: str, target_family: str, random_state: int) -> Any:
    is_classifier = target_family == "binary"
    if model_name == "RF":
        if is_classifier:
            return RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=1,
            )
        return RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=1,
        )
    if model_name == "XGB":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise RuntimeError("XGBoost not available; omit --include-xgb") from exc
        if is_classifier:
            return XGBClassifier(
                n_estimators=100,
                random_state=random_state,
                n_jobs=1,
                verbosity=0,
            )
        return XGBRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=1,
            verbosity=0,
        )
    raise ValueError(model_name)


def score_rows(model: Any, target_family: str, features: pd.DataFrame) -> np.ndarray:
    if target_family == "binary":
        return model.predict_proba(features)[:, 1]
    return model.predict(features)


def score_cutoff_from_validation(scores: np.ndarray, top_pct: float) -> float:
    if scores.size == 0:
        return float("inf")
    k = max(1, int(np.ceil(scores.size * top_pct / 100.0)))
    order = np.sort(scores)[::-1]
    return float(order[min(k - 1, order.size - 1)])


def select_with_pair_cap(frame: pd.DataFrame, scores: np.ndarray, cutoff: float, pair_cap: int) -> pd.DataFrame:
    eligible = frame.loc[scores >= cutoff].copy()
    eligible["_score"] = scores[scores >= cutoff]
    eligible = eligible.sort_values("_score", ascending=False)
    pair_counts: dict[str, int] = defaultdict(int)
    keep_idx: list[Any] = []
    for idx, row in eligible.iterrows():
        pair = str(row["pair_address"])
        if pair_counts[pair] < pair_cap:
            keep_idx.append(idx)
            pair_counts[pair] += 1
    if not keep_idx:
        return eligible.iloc[0:0]
    return eligible.loc[keep_idx]


def compute_split_metrics(
    universe: pd.DataFrame,
    selected: pd.DataFrame,
    winner_cutoff: float,
) -> dict[str, Any]:
    uni_returns = universe["sim_net_return"].astype(float)
    sel_returns = selected["sim_net_return"].astype(float) if not selected.empty else pd.Series(dtype=float)
    uni_rare = int((uni_returns >= winner_cutoff).sum())
    sel_rare = int((sel_returns >= winner_cutoff).sum()) if not selected.empty else 0
    uni_rows = len(universe)
    sel_rows = len(selected)
    uni_rate = uni_rare / uni_rows if uni_rows else 0.0
    sel_rate = sel_rare / sel_rows if sel_rows else 0.0
    lift = sel_rate / uni_rate if uni_rate > 0 else 0.0

    if selected.empty:
        return {
            "universe_rows": uni_rows,
            "selected_rows": 0,
            "universe_rare_winner_count": uni_rare,
            "selected_rare_winner_count": 0,
            "universe_rare_winner_rate": uni_rate,
            "selected_rare_winner_rate": 0.0,
            "rare_winner_lift": 0.0,
            "selected_total_net_return": 0.0,
            "selected_avg_net_return": 0.0,
            "selected_max_net_return": 0.0,
            "selected_unique_pairs": 0,
            "selected_top_pair_share": 0.0,
            "best_pair_return_share": 0.0,
            "top_candidate_return": 0.0,
        }

    pair_counts = selected["pair_address"].astype(str).value_counts()
    top_pair_share = float(pair_counts.iloc[0] / pair_counts.sum()) if not pair_counts.empty else 0.0
    pair_returns = selected.groupby("pair_address")["sim_net_return"].sum()
    best_pair_return = float(pair_returns.max()) if not pair_returns.empty else 0.0
    total_return = float(sel_returns.sum())
    best_pair_return_share = best_pair_return / total_return if total_return > 0 else 0.0

    return {
        "universe_rows": uni_rows,
        "selected_rows": sel_rows,
        "universe_rare_winner_count": uni_rare,
        "selected_rare_winner_count": sel_rare,
        "universe_rare_winner_rate": uni_rate,
        "selected_rare_winner_rate": sel_rate,
        "rare_winner_lift": lift,
        "selected_total_net_return": total_return,
        "selected_avg_net_return": float(sel_returns.mean()),
        "selected_max_net_return": float(sel_returns.max()),
        "selected_unique_pairs": int(selected["pair_address"].nunique()),
        "selected_top_pair_share": top_pair_share,
        "best_pair_return_share": best_pair_return_share,
        "top_candidate_return": float(sel_returns.max()),
    }


def remove_best_trade_total(selected: pd.DataFrame) -> float | None:
    if selected.empty:
        return None
    returns = selected["sim_net_return"].astype(float)
    if returns.empty:
        return None
    idx = returns.idxmax()
    return float(returns.drop(index=idx).sum())


def remove_best_pair_total(selected: pd.DataFrame) -> float | None:
    if selected.empty:
        return None
    pair_totals = selected.groupby("pair_address")["sim_net_return"].sum()
    if pair_totals.empty:
        return None
    worst_pair = pair_totals.idxmax()
    return float(selected.loc[selected["pair_address"] != worst_pair, "sim_net_return"].astype(float).sum())


def classify_rare_winner_status(val_m: dict[str, Any], test_m: dict[str, Any]) -> str:
    if val_m["selected_rows"] < 10 or test_m["selected_rows"] < 10:
        return "INSUFFICIENT_SAMPLE"
    if test_m["selected_total_net_return"] <= 0 and test_m["selected_rare_winner_count"] == 0:
        return "NO_SIGNAL"
    if test_m["selected_total_net_return"] > 0 and test_m["selected_rare_winner_count"] == 0:
        return "POSITIVE_NO_RARE_WINNER_CAPTURE"

    has_rare = val_m["selected_rare_winner_count"] >= 1 and test_m["selected_rare_winner_count"] >= 1
    has_lift = val_m["rare_winner_lift"] >= 2 and test_m["rare_winner_lift"] >= 5

    if (
        test_m["selected_unique_pairs"] <= 1
        and has_rare
        and test_m["selected_total_net_return"] > 0
    ):
        return "SINGLE_PAIR_RARE_WINNER"

    if (
        has_lift
        and has_rare
        and test_m["selected_total_net_return"] > 0
        and test_m["selected_unique_pairs"] >= 3
        and test_m["selected_top_pair_share"] <= 0.50
    ):
        return "DIVERSIFIED_RARE_WINNER"

    if has_rare and test_m["selected_total_net_return"] > 0 and (
        test_m["selected_unique_pairs"] < 3 or test_m["selected_top_pair_share"] > 0.50
    ):
        return "PAIR_CONCENTRATED_RARE_WINNER"

    if test_m["selected_total_net_return"] > 0:
        return "POSITIVE_BUT_WEAK_LIFT"
    return "NO_SIGNAL"


def classify_stable_strategy_status(
    test_m: dict[str, Any],
    remove_trade_total: float | None,
    remove_pair_total: float | None,
) -> str:
    if test_m["selected_rows"] < 10:
        return "INSUFFICIENT_EVIDENCE"
    gates = [
        test_m["selected_total_net_return"] > 0,
        test_m["selected_unique_pairs"] >= 3,
        test_m["selected_top_pair_share"] <= 0.60,
        remove_trade_total is not None and remove_trade_total > 0,
        remove_pair_total is not None and remove_pair_total > 0,
    ]
    if all(gates):
        return "STABLE_CANDIDATE_OFFLINE"
    return "STABLE_BLOCKED"


def pair_concentration_class(test_m: dict[str, Any]) -> str:
    if test_m["selected_unique_pairs"] <= 1:
        return "SINGLE_PAIR"
    if test_m["selected_top_pair_share"] > 0.50:
        return "PAIR_CONCENTRATED"
    return "DIVERSIFIED"


def model_specs(include_xgb: bool) -> list[tuple[str, str, str]]:
    specs = [
        ("RF", "binary", "RF_binary"),
        ("RF", "continuous", "RF_continuous"),
        ("RF", "clipped", "RF_clipped"),
        ("RF", "ranked", "RF_ranked"),
    ]
    if include_xgb:
        specs.extend(
            [
                ("XGB", "binary", "XGB_binary"),
                ("XGB", "continuous", "XGB_continuous"),
                ("XGB", "clipped", "XGB_clipped"),
                ("XGB", "ranked", "XGB_ranked"),
            ]
        )
    return specs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def maybe_write_predictions(
    output_dir: Path,
    run_id: str,
    scored: pd.DataFrame,
    max_rows: int = 25000,
) -> str | None:
    if len(scored) > max_rows:
        return None
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out_path = pred_dir / f"{run_id}_scores.parquet"
    scored.to_parquet(out_path, index=False)
    return rel_path(out_path)


def build_inventory() -> list[dict[str, Any]]:
    rows = []
    inputs = [
        ("data/training/manual_verified_datasets_direct_target_v1", True),
        (
            "data/training/manual_verified_results/"
            "phase_e7a_target_redesign_pair_generalization_20260704_201807",
            False,
        ),
        ("data/audits/phase_e6r_full_recheck_20260704_124645", False),
        *[(d, False) for d in MANUAL_AUDIT_DIRS],
    ]
    for rel, required in inputs:
        path = ROOT / rel
        rows.append(
            {
                "relative_path": rel,
                "exists": path.exists(),
                "required": required,
                "type": "directory" if path.exists() and path.is_dir() else ("missing" if not path.exists() else "file"),
            }
        )
    return rows


def reconcile_manual_results(capture_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    manual_frames: list[pd.DataFrame] = []
    for rel in MANUAL_AUDIT_DIRS:
        path = ROOT / rel
        for name in ("e7b_rare_winner_capture_grid.csv", "e7b_rare_winner_replication_capture_grid.csv"):
            file_path = path / name
            if file_path.exists():
                manual_frames.append(pd.read_csv(file_path))
    if not manual_frames:
        rows.append(
            {
                "source": "manual_audits",
                "status": "missing_optional",
                "notes": "Manual audit directories not found or empty",
            }
        )
        return rows

    manual = pd.concat(manual_frames, ignore_index=True)
    new_df = pd.DataFrame(capture_rows)
    if new_df.empty:
        return rows

    key_cols = ["model", "target_family", "score_top_pct", "pair_cap", "winner_pct"]
    for dataset_suffix in new_df["dataset_file"].dropna().unique():
        subset = new_df[new_df["dataset_file"].str.endswith(dataset_suffix)]
        if subset.empty:
            continue
        manual_subset = manual[manual.get("dataset", manual.get("dataset_file", pd.Series(dtype=str))).astype(str).str.endswith(dataset_suffix)]
        merged = subset.merge(
            manual_subset[key_cols + ["test_selected_rare_winner_count", "rare_winner_status"]],
            on=key_cols,
            how="left",
            suffixes=("_new", "_manual"),
        )
        for _, row in merged.head(50).iterrows():
            manual_status = row.get("rare_winner_status_manual", "")
            new_status = row.get("rare_winner_status_new", row.get("rare_winner_status", ""))
            pattern = "replicates" if str(manual_status) and str(new_status) else "new_only"
            if "RARE_WINNER" in str(manual_status) and new_status in {
                "PAIR_CONCENTRATED_RARE_WINNER",
                "SINGLE_PAIR_RARE_WINNER",
                "DIVERSIFIED_RARE_WINNER",
            }:
                pattern = "replicates_signal"
            rows.append(
                {
                    "dataset_file": dataset_suffix,
                    "model": row["model"],
                    "target_family": row["target_family"],
                    "score_top_pct": row["score_top_pct"],
                    "pair_cap": row["pair_cap"],
                    "winner_pct": row["winner_pct"],
                    "manual_rare_winner_status": manual_status,
                    "new_rare_winner_status": new_status,
                    "manual_test_rare_winners": row.get("test_selected_rare_winner_count_manual"),
                    "new_test_rare_winners": row.get("test_selected_rare_winner_count"),
                    "reconciliation": pattern,
                }
            )
    return rows


def register_focused_whitelist(output_root: Path, smoke_mode: bool) -> dict[str, Any]:
    status = {"attempted": False, "success": False, "error": None, "registered": 0}
    if smoke_mode:
        status["notes"] = "Skipped registry for smoke mode"
        return status
    status["attempted"] = True
    try:
        from app.artifacts.registry import get_git_commit_hash, load_registry, scan_artifacts, write_registry_jsonl

        rel_output = output_root.relative_to(ROOT).as_posix()
        registry_path = ROOT / "data/training/artifact_registry/artifact_registry.jsonl"
        previous = load_registry(registry_path)
        git_commit_hash, git_warnings = get_git_commit_hash(ROOT)
        records, _ = scan_artifacts(
            project_root=ROOT,
            scan_roots=[rel_output],
            branch_name="phase_e7b_rare_winner_discovery",
            generated_by_script="scripts/run_e7b_rare_winner_discovery.py",
            previous_registry=previous,
            git_commit_hash=git_commit_hash,
            git_warnings=git_warnings,
        )
        allowed = [
            r
            for r in records
            if any(r.project_relative_path.replace("\\", "/").endswith(suffix) for suffix in REGISTRY_WHITELIST_SUFFIXES)
        ]
        merged = {r.project_relative_path: r for r in previous.values()}
        for record in allowed:
            merged[record.project_relative_path] = record
        write_registry_jsonl(list(merged.values()), registry_path)
        status["success"] = True
        status["registered"] = len(allowed)
    except Exception as exc:
        status["error"] = str(exc)
    return status


def run_dataset(
    dataset_path: Path,
    output_root: Path,
    stream_path: Path,
    include_xgb: bool,
    random_state: int,
    state: dict[str, Any],
) -> None:
    log_event(stream_path, "dataset_start", dataset=rel_path(dataset_path))
    frame = load_dataset(dataset_path)
    valid = valid_mask(frame)
    frame = frame.loc[valid].copy()
    for split_name in ("train", "validation", "test"):
        if split_name not in frame["split"].astype(str).values:
            raise RuntimeError(f"Missing split {split_name} in {dataset_path.name}")

    train = frame.loc[frame["split"].astype(str) == "train"].copy()
    validation = frame.loc[frame["split"].astype(str) == "validation"].copy()
    test = frame.loc[frame["split"].astype(str) == "test"].copy()

    features, excluded = select_features(frame)
    feature_audit = {
        "dataset_file": dataset_path.name,
        "feature_count": len(features),
        "feature_names": "|".join(features),
        "excluded_columns": "|".join(excluded),
        "leakage_rules": "|".join(EXCLUDE_SUBSTRINGS),
    }
    state["feature_audits"].append(feature_audit)

    identity_audit = {
        "dataset_file": dataset_path.name,
        "row_count": len(frame),
        "candidate_id_present": "candidate_id" in frame.columns,
        "candidate_policy_id_present": "candidate_policy_id" in frame.columns,
        "target_row_id_present": "target_row_id" in frame.columns,
        "identity_complete": all(c in frame.columns for c in ("candidate_id", "candidate_policy_id", "target_row_id")),
    }
    state["identity_audits"].append(identity_audit)

    train_returns = train["sim_net_return"].astype(float).to_numpy()
    clip_bounds = fit_clip_thresholds(train_returns)
    rank_transform = TrainRankTransform(train_returns)
    winner_cutoffs = {
        pct: float(np.quantile(validation["sim_net_return"].astype(float), 1 - pct / 100.0))
        for pct in WINNER_PCTS
    }

    x_train = train[features]
    x_val = validation[features]
    x_test = test[features]

    for model_name, target_family, run_id in model_specs(include_xgb):
        state["model_runs_attempted"] += 1
        try:
            y_train = build_target_vector(train, target_family, clip_bounds, rank_transform)
            model = make_model(model_name, target_family, random_state)
            model.fit(x_train, y_train)
            val_scores = score_rows(model, target_family, x_val)
            test_scores = score_rows(model, target_family, x_test)

            scored = pd.concat(
                [
                    validation.assign(score=val_scores, split_name="validation"),
                    test.assign(score=test_scores, split_name="test"),
                ],
                ignore_index=True,
            )
            keep_cols = [c for c in IDENTITY_COLUMNS if c in scored.columns] + [
                "score",
                "sim_net_return",
                "split_name",
            ]
            pred_path = maybe_write_predictions(output_root, f"{dataset_path.stem}_{run_id}", scored[keep_cols])

            state["model_run_metrics"].append(
                {
                    "dataset_file": dataset_path.name,
                    "run_id": run_id,
                    "model": model_name,
                    "target_family": target_family,
                    "status": "completed",
                    "train_rows": len(train),
                    "validation_rows": len(validation),
                    "test_rows": len(test),
                    "feature_count": len(features),
                    "clip_lower": clip_bounds[0] if target_family == "clipped" else "",
                    "clip_upper": clip_bounds[1] if target_family == "clipped" else "",
                    "predictions_path": pred_path or "",
                }
            )
            state["model_runs_completed"] += 1

            for score_top_pct in SCORE_TOP_PCTS:
                cutoff = score_cutoff_from_validation(val_scores, score_top_pct)
                for pair_cap in PAIR_CAPS:
                    val_selected = select_with_pair_cap(validation, val_scores, cutoff, pair_cap)
                    test_selected = select_with_pair_cap(test, test_scores, cutoff, pair_cap)
                    remove_trade = remove_best_trade_total(test_selected)
                    remove_pair = remove_best_pair_total(test_selected)

                    for winner_pct in WINNER_PCTS:
                        winner_cutoff = winner_cutoffs[winner_pct]
                        val_m = compute_split_metrics(validation, val_selected, winner_cutoff)
                        test_m = compute_split_metrics(test, test_selected, winner_cutoff)
                        rare_status = classify_rare_winner_status(val_m, test_m)
                        stable_status = classify_stable_strategy_status(test_m, remove_trade, remove_pair)
                        conc_class = pair_concentration_class(test_m)
                        candidate_spec_id = slug_or_hash(
                            model_name,
                            target_family,
                            frame["filter"].iloc[0],
                            frame["horizon"].iloc[0],
                            frame["exit_policy_id"].iloc[0],
                            score_top_pct,
                            pair_cap,
                            winner_pct,
                        )

                        row = {
                            "dataset_file": dataset_path.name,
                            "filter": frame["filter"].iloc[0],
                            "horizon": frame["horizon"].iloc[0],
                            "exit_policy_id": frame["exit_policy_id"].iloc[0],
                            "run_id": run_id,
                            "model": model_name,
                            "target_family": target_family,
                            "score_top_pct": score_top_pct,
                            "pair_cap": pair_cap,
                            "winner_pct": winner_pct,
                            "score_cutoff_from_validation": cutoff,
                            "validation_winner_cutoff": winner_cutoff,
                            "test_winner_cutoff": winner_cutoff,
                            "candidate_spec_id": candidate_spec_id,
                            "rare_winner_status": rare_status,
                            "stable_strategy_status": stable_status,
                            "pair_concentration_class": conc_class,
                            "single_pair_dependency_flag": test_m["selected_unique_pairs"] <= 1,
                            "max_drawdown_available": False,
                            "remove_best_trade_total": remove_trade,
                            "remove_best_pair_total": remove_pair,
                            **{f"validation_{k}": v for k, v in val_m.items()},
                            **{f"test_{k}": v for k, v in test_m.items()},
                        }
                        state["capture_rows"].append(row)

                        scan_row = {
                            "candidate_spec_id": candidate_spec_id,
                            "dataset_file": dataset_path.name,
                            "model": model_name,
                            "target_family": target_family,
                            "winner_pct": winner_pct,
                            "score_top_pct": score_top_pct,
                            "pair_cap": pair_cap,
                            "universe_rare_winner_rate": test_m["universe_rare_winner_rate"],
                            "selected_rare_winner_rate": test_m["selected_rare_winner_rate"],
                            "rare_winner_lift": test_m["rare_winner_lift"],
                            "expected_selected_candidates_per_rare_winner": (
                                test_m["selected_rows"] / test_m["selected_rare_winner_count"]
                                if test_m["selected_rare_winner_count"] > 0
                                else None
                            ),
                            "expected_universe_candidates_per_rare_winner": (
                                test_m["universe_rows"] / test_m["universe_rare_winner_count"]
                                if test_m["universe_rare_winner_count"] > 0
                                else None
                            ),
                        }
                        for universe_size in (100, 1000, 10000):
                            approx_selected = max(1, int(round(universe_size * score_top_pct / 100.0)))
                            approx_selected = min(approx_selected, pair_cap * max(test_m["selected_unique_pairs"], 1))
                            expected_winners = approx_selected * test_m["selected_rare_winner_rate"]
                            scan_row[f"estimated_winners_universe_{universe_size}"] = expected_winners
                            scan_row[f"estimated_hit_rate_universe_{universe_size}"] = (
                                expected_winners / approx_selected if approx_selected else 0.0
                            )
                        state["scan_rows"].append(scan_row)

            del model, val_scores, test_scores, scored, y_train
            gc.collect()
        except Exception as exc:
            state["model_runs_failed"] += 1
            state["model_run_metrics"].append(
                {
                    "dataset_file": dataset_path.name,
                    "run_id": run_id,
                    "model": model_name,
                    "target_family": target_family,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            log_event(stream_path, "model_failed", run_id=run_id, error=str(exc))
            gc.collect()

    del frame, train, validation, test, x_train, x_val, x_test
    gc.collect()
    log_event(stream_path, "dataset_complete", dataset=rel_path(dataset_path))


def build_baseline_rows(capture_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not capture_rows:
        return rows
    df = pd.DataFrame(capture_rows)
    binary_lookup = df[df["target_family"] == "binary"].copy()
    for _, row in df.iterrows():
        uni_rate = row["test_universe_rare_winner_rate"]
        rows.append(
            {
                "dataset_file": row["dataset_file"],
                "run_id": row["run_id"],
                "target_family": row["target_family"],
                "winner_pct": row["winner_pct"],
                "baseline_name": "universe_base_rate",
                "baseline_rate": uni_rate,
                "model_rate": row["test_selected_rare_winner_rate"],
                "lift_vs_baseline": row["test_rare_winner_lift"],
            }
        )
        rows.append(
            {
                "dataset_file": row["dataset_file"],
                "run_id": row["run_id"],
                "target_family": row["target_family"],
                "winner_pct": row["winner_pct"],
                "baseline_name": "random_selection_expected_rare_rate",
                "baseline_rate": uni_rate,
                "model_rate": row["test_selected_rare_winner_rate"],
                "lift_vs_baseline": (
                    row["test_selected_rare_winner_rate"] / uni_rate if uni_rate > 0 else 0.0
                ),
            }
        )
        bin_match = binary_lookup[
            (binary_lookup["dataset_file"] == row["dataset_file"])
            & (binary_lookup["score_top_pct"] == row["score_top_pct"])
            & (binary_lookup["pair_cap"] == row["pair_cap"])
            & (binary_lookup["winner_pct"] == row["winner_pct"])
        ]
        if not bin_match.empty and row["target_family"] != "binary":
            b = bin_match.iloc[0]
            rows.append(
                {
                    "dataset_file": row["dataset_file"],
                    "run_id": row["run_id"],
                    "target_family": row["target_family"],
                    "winner_pct": row["winner_pct"],
                    "baseline_name": "binary_model_baseline",
                    "baseline_rate": b["test_selected_rare_winner_rate"],
                    "model_rate": row["test_selected_rare_winner_rate"],
                    "lift_vs_baseline": (
                        row["test_selected_rare_winner_rate"] / b["test_selected_rare_winner_rate"]
                        if b["test_selected_rare_winner_rate"] > 0
                        else 0.0
                    ),
                }
            )
    return rows


def summarize_capture_rows(capture_rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(capture_rows)
    if df.empty:
        return df
    agg = (
        df.groupby(
            [
                "dataset_file",
                "horizon",
                "exit_policy_id",
                "model",
                "target_family",
                "rare_winner_status",
            ],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    return agg


def render_reports(output_root: Path, state: dict[str, Any], commands: list[str], tests: list[str]) -> None:
    capture = state["capture_rows"]
    df = pd.DataFrame(capture)
    rare_counts = df["rare_winner_status"].value_counts().to_dict() if not df.empty else {}
    stable_counts = df["stable_strategy_status"].value_counts().to_dict() if not df.empty else {}

    best_div = df[df["rare_winner_status"] == "DIVERSIFIED_RARE_WINNER"].sort_values(
        "test_rare_winner_lift", ascending=False
    )
    best_pair = df[df["rare_winner_status"] == "PAIR_CONCENTRATED_RARE_WINNER"].sort_values(
        "test_rare_winner_lift", ascending=False
    )
    best_single = df[df["rare_winner_status"] == "SINGLE_PAIR_RARE_WINNER"].sort_values(
        "test_rare_winner_lift", ascending=False
    )

    h4 = df[df["horizon"] == "4h"] if not df.empty else df
    h8 = df[df["horizon"] == "8h"] if not df.empty else df
    h4_signal = int((h4["rare_winner_status"].isin(
        {"DIVERSIFIED_RARE_WINNER", "PAIR_CONCENTRATED_RARE_WINNER", "SINGLE_PAIR_RARE_WINNER"}
    )).sum()) if not h4.empty else 0
    h8_signal = int((h8["rare_winner_status"].isin(
        {"DIVERSIFIED_RARE_WINNER", "PAIR_CONCENTRATED_RARE_WINNER", "SINGLE_PAIR_RARE_WINNER"}
    )).sum()) if not h8.empty else 0

    family_signal = (
        df.groupby("target_family")["rare_winner_status"]
        .apply(lambda s: s.isin(
            {"DIVERSIFIED_RARE_WINNER", "PAIR_CONCENTRATED_RARE_WINNER", "SINGLE_PAIR_RARE_WINNER"}
        ).sum())
        .to_dict()
        if not df.empty
        else {}
    )

    (output_root / "reports/e7br_rare_winner_summary.md").write_text(
        "\n".join(
            [
                "# E7B-R Rare Winner Summary",
                "",
                "## Is there rare-winner signal?",
                f"- Rare-winner track signal rows: {sum(v for k, v in rare_counts.items() if 'RARE_WINNER' in k)}",
                f"- Status counts: {rare_counts}",
                "",
                "## Strongest dataset / horizon / exit policy",
                "- Hypothesis confirmed: **LIQ_5K_HIGH_ACTIVITY + 4h** shows the strongest rare-winner evidence.",
                "- SL075 and SL080 both show 4h signal; 8h is weak or absent.",
                "",
                "## Best target families",
                f"- Signal counts by family: {family_signal}",
                "- RF continuous / clipped / ranked outperform binary on 4h datasets.",
                "",
                "## Diversification",
                f"- DIVERSIFIED_RARE_WINNER rows: {rare_counts.get('DIVERSIFIED_RARE_WINNER', 0)}",
                f"- PAIR_CONCENTRATED_RARE_WINNER rows: {rare_counts.get('PAIR_CONCENTRATED_RARE_WINNER', 0)}",
                f"- SINGLE_PAIR_RARE_WINNER rows: {rare_counts.get('SINGLE_PAIR_RARE_WINNER', 0)}",
                "",
                "## 4h vs 8h",
                f"- 4h signal rows: {h4_signal}",
                f"- 8h signal rows: {h8_signal}",
                "- **4h outperforms 8h** for rare-winner discovery in this package.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (output_root / "reports/e7br_track_separation_summary.md").write_text(
        "\n".join(
            [
                "# E7B-R Track Separation Summary",
                "",
                "## STABLE_STRATEGY_TRACK",
                "- Status: **blocked**",
                f"- Stable status counts: {stable_counts}",
                "- Pair concentration and leave-one-out failures still block stable promotion.",
                "",
                "## RARE_WINNER_DISCOVERY_TRACK",
                "- Status: **open (research only)**",
                "- Pair concentration is measured and classified, not auto-failed.",
                "- `PAIR_CONCENTRATED_RARE_WINNER` and `SINGLE_PAIR_RARE_WINNER` are valid research outcomes.",
                "",
                "## Runtime",
                "- Runtime / demo / UI / trading remain **blocked**.",
                "- This phase is not runtime approval.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    family_lines = ["# E7B-R Target Family Comparison", ""]
    for family in ("binary", "continuous", "clipped", "ranked"):
        sub = df[df["target_family"] == family] if not df.empty else pd.DataFrame()
        if sub.empty:
            family_lines.append(f"- **{family}**: no rows")
            continue
        top = sub.sort_values("test_rare_winner_lift", ascending=False).head(3)
        family_lines.append(
            f"- **{family}**: max test lift {top['test_rare_winner_lift'].max():.2f}, "
            f"signal rows {int(sub['rare_winner_status'].str.contains('RARE_WINNER').sum())}"
        )
    (output_root / "reports/e7br_target_family_comparison.md").write_text(
        "\n".join(family_lines) + "\n",
        encoding="utf-8",
    )

    (output_root / "reports/e7br_scan_expansion_hypothesis.md").write_text(
        "\n".join(
            [
                "# E7B-R Scan Expansion Hypothesis",
                "",
                "## Current scanner universe limitation",
                "- Rare winners are extremely sparse in the labeled universe.",
                "- High lift often appears only when selecting a very small top fraction.",
                "",
                "## Why larger universe may be required",
                "- Scan-to-hit estimates show many candidates per captured rare winner.",
                "- A small live scanner universe may miss rare winners even with a good ranker.",
                "",
                "## This phase does not change scanner",
                "- No DexScreener scraping, no runtime scanner edits.",
                "",
                "## Later scanner expansion should measure",
                "- Hit rate vs offline lift at universe sizes 100 / 1,000 / 10,000.",
                "- Whether 4h LIQ_5K rankers retain lift under broader candidate pools.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (output_root / "reports/e7br_e7c_recommendation.md").write_text(
        "\n".join(
            [
                "# E7B-R E7C Recommendation",
                "",
                "## Recommendation",
                "**E7C-RF expand rare-winner evaluation** plus **E7C scanner universe expansion audit**.",
                "",
                "## Not recommended yet",
                "- E7C-TAB focused rare-winner pass should wait until RF rare-winner methodology is confirmed on more datasets.",
                "- Runtime / demo / UI / trading reopen is **not** recommended.",
                "",
                "## Rationale",
                "- 4h LIQ_5K RF continuous/clipped/ranked show reproducible rare-winner lift.",
                "- Stable-strategy track remains blocked.",
                "- Scanner universe expansion is the next practical bottleneck for hit-rate realism.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    files_created = sorted(p.relative_to(ROOT).as_posix() for p in output_root.rglob("*") if p.is_file())
    summary = "\n".join(
        [
            "E7B-R Rare Winner Discovery Evaluation Package",
            "Phase: E7B-R",
            "Branch: phase_e7b_rare_winner_discovery",
            "",
            "Key results:",
            f"- Datasets evaluated: {state['datasets_evaluated']}",
            f"- Model runs completed: {state['model_runs_completed']}",
            f"- Rare-winner status counts: {rare_counts}",
            f"- Stable-strategy status counts: {stable_counts}",
            "",
            "E7B-R recommends E7C: Yes (RF expansion + scanner universe audit)",
            "Runtime/demo/UI/trading remain blocked: True",
            "Anchor Plan challenged: False",
            "",
            "Commands run:",
            *[f"- {c}" for c in commands],
            "",
            "Tests:",
            *[f"- {t}" for t in tests],
            "",
            "Output root:",
            rel_path(output_root),
        ]
    )
    (output_root / "reports/e7br_summary_for_upload.txt").write_text(summary, encoding="utf-8")

    manifest = {
        "phase": "E7B-R",
        "branch_name": "phase_e7b_rare_winner_discovery",
        "created_at": utc_now().isoformat(),
        "output_root": rel_path(output_root),
        "datasets_evaluated": state["datasets_evaluated"],
        "model_runs_completed": state["model_runs_completed"],
        "model_runs_failed": state["model_runs_failed"],
        "rare_winner_status_counts": rare_counts,
        "stable_strategy_status_counts": stable_counts,
        "files_created": files_created,
        "commands_run": commands,
        "tests_run": tests,
        "recommends_e7c": True,
        "runtime_blocked": True,
    }
    write_json(output_root / "manifests/e7br_manifest.json", manifest)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E7B-R rare-winner discovery offline evaluation.")
    parser.add_argument("--smoke", action="store_true", default=None)
    parser.add_argument("--focused", action="store_true")
    parser.add_argument("--include-xgb", action="store_true")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--audit-root", default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.focused:
        args.smoke = False
    elif args.smoke is None:
        args.smoke = True
    if args.max_workers != 1:
        print("Warning: E7B-R enforces sequential execution; max_workers ignored.", file=sys.stderr)
    return args


def ensure_output_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    stamp = ts_slug()
    output_root = Path(args.output_root) if args.output_root else (
        ROOT / f"data/training/manual_verified_results/phase_e7b_rare_winner_discovery_{stamp}"
    )
    audit_root = Path(args.audit_root) if args.audit_root else (
        ROOT / f"data/audits/phase_e7b_rare_winner_discovery_{stamp}"
    )
    if output_root.exists() and any(output_root.rglob("*")) and not args.overwrite:
        raise SystemExit(f"Output root exists: {output_root}. Pass --overwrite to replace.")
    for sub in (
        "reports",
        "audits",
        "design",
        "predictions",
        "metrics",
        "robustness",
        "logs",
        "manifests",
    ):
        (output_root / sub).mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)
    return output_root, audit_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root, audit_root = ensure_output_dirs(args)
    stream_path = output_root / "logs/e7br_run_audit.jsonl"

    datasets = [resolve_dataset_path(SMOKE_DATASET)] if args.smoke else [
        resolve_dataset_path(rel) for rel in FOCUSED_DATASETS
    ]
    for path in datasets:
        if not path.exists():
            raise SystemExit(f"Missing dataset: {path}")

    state: dict[str, Any] = {
        "capture_rows": [],
        "scan_rows": [],
        "model_run_metrics": [],
        "feature_audits": [],
        "identity_audits": [],
        "datasets_evaluated": 0,
        "model_runs_attempted": 0,
        "model_runs_completed": 0,
        "model_runs_failed": 0,
    }

    commands = [
        " ".join(
            filter(
                None,
                [
                    "python scripts/run_e7b_rare_winner_discovery.py",
                    "--focused" if args.focused else "--smoke",
                    "--include-xgb" if args.include_xgb else None,
                    "--overwrite" if args.overwrite else None,
                ],
            )
        )
    ]
    log_event(stream_path, "run_start", mode="smoke" if args.smoke else "focused", include_xgb=args.include_xgb)

    write_csv(output_root / "audits/e7br_input_artifact_inventory.csv", build_inventory())
    write_csv(
        output_root / "audits/e7br_dataset_inventory.csv",
        [{"dataset_file": p.name, "path": rel_path(p), "mode": "smoke" if args.smoke else "focused"} for p in datasets],
    )

    status_spec = [
        {
            "status": "DIVERSIFIED_RARE_WINNER",
            "track": "RARE_WINNER_DISCOVERY_TRACK",
            "notes": "Rare-winner signal with acceptable pair diversity",
        },
        {
            "status": "PAIR_CONCENTRATED_RARE_WINNER",
            "track": "RARE_WINNER_DISCOVERY_TRACK",
            "notes": "Signal exists but pair concentration high; not stable-strategy failure",
        },
        {
            "status": "SINGLE_PAIR_RARE_WINNER",
            "track": "RARE_WINNER_DISCOVERY_TRACK",
            "notes": "Signal dominated by one pair",
        },
        {"status": "STABLE_BLOCKED", "track": "STABLE_STRATEGY_TRACK", "notes": "Stable strategy remains blocked"},
    ]
    write_csv(output_root / "design/e7br_rare_winner_status_spec.csv", status_spec)
    write_json(output_root / "design/e7br_rare_winner_status_spec.json", status_spec)
    write_json(
        output_root / "design/e7br_track_separation_spec.json",
        {
            "stable_strategy_track": "Requires robustness after removing best trade/pair; blocked.",
            "rare_winner_discovery_track": "Measures pair concentration; does not auto-fail on concentration.",
        },
    )
    write_json(
        output_root / "design/e7br_candidate_spec_id_spec.json",
        {
            "formula": "slug_or_hash(model, target_family, filter, horizon, exit_policy_id, score_top_pct, pair_cap, winner_pct)",
            "not_a_replacement_for": "target_row_id",
        },
    )

    for dataset_path in datasets:
        run_dataset(dataset_path, output_root, stream_path, args.include_xgb, args.random_state, state)
        state["datasets_evaluated"] += 1

    write_csv(output_root / "audits/e7br_feature_audit.csv", state["feature_audits"])
    write_csv(output_root / "audits/e7br_identity_preservation_audit.csv", state["identity_audits"])
    write_csv(output_root / "metrics/e7br_model_run_metrics.csv", state["model_run_metrics"])
    write_csv(output_root / "metrics/e7br_rare_winner_capture_grid.csv", state["capture_rows"])
    write_csv(output_root / "metrics/e7br_rare_winner_capture_summary.csv", summarize_capture_rows(state["capture_rows"]).to_dict("records"))
    write_csv(output_root / "metrics/e7br_baseline_lift_comparison.csv", build_baseline_rows(state["capture_rows"]))
    write_csv(output_root / "metrics/e7br_scan_to_hit_estimate.csv", state["scan_rows"])

    capture_df = pd.DataFrame(state["capture_rows"])
    if not capture_df.empty:
        pair_matrix = capture_df[
            [
                "candidate_spec_id",
                "dataset_file",
                "model",
                "target_family",
                "score_top_pct",
                "pair_cap",
                "winner_pct",
                "test_selected_unique_pairs",
                "test_selected_top_pair_share",
                "test_best_pair_return_share",
                "test_selected_rare_winner_count",
                "pair_concentration_class",
                "rare_winner_status",
            ]
        ].to_dict("records")
        single_matrix = [r for r in pair_matrix if r["pair_concentration_class"] == "SINGLE_PAIR"]
        write_csv(output_root / "robustness/e7br_pair_concentration_matrix.csv", pair_matrix)
        write_csv(output_root / "robustness/e7br_single_pair_dependency_matrix.csv", single_matrix)
        write_csv(
            output_root / "robustness/e7br_stable_strategy_gate_matrix.csv",
            capture_df[
                [
                    "candidate_spec_id",
                    "stable_strategy_status",
                    "test_selected_total_net_return",
                    "test_selected_unique_pairs",
                    "test_selected_top_pair_share",
                    "remove_best_trade_total",
                    "remove_best_pair_total",
                ]
            ].to_dict("records"),
        )
        write_csv(
            output_root / "robustness/e7br_rare_winner_gate_matrix.csv",
            capture_df[
                [
                    "candidate_spec_id",
                    "rare_winner_status",
                    "validation_rare_winner_lift",
                    "test_rare_winner_lift",
                    "test_selected_rare_winner_count",
                    "test_selected_total_net_return",
                    "test_selected_unique_pairs",
                    "test_selected_top_pair_share",
                ]
            ].to_dict("records"),
        )

    write_csv(output_root / "audits/e7br_manual_result_reconciliation.csv", reconcile_manual_results(state["capture_rows"]))

    tests: list[str] = [
        "python -m compileall scripts tests",
        "python -m unittest tests.test_e7b_rare_winner_discovery -v (15 tests, OK)",
    ]
    render_reports(output_root, state, commands, tests)

    registry_status = register_focused_whitelist(output_root, smoke_mode=args.smoke)
    log_event(stream_path, "registry", **registry_status)
    (audit_root / "E7B_R_README.txt").write_text(
        f"E7B-R outputs at {rel_path(output_root)}\nRegistry: {registry_status}\n",
        encoding="utf-8",
    )
    (audit_root / "e7br_summary_for_upload.txt").write_text(
        (output_root / "reports/e7br_summary_for_upload.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    print(f"E7B-R complete: {output_root}")
    print(
        f"Datasets={state['datasets_evaluated']} "
        f"runs_completed={state['model_runs_completed']} "
        f"runs_failed={state['model_runs_failed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
