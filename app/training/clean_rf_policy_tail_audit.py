"""Phase E8C — prediction tail and economic policy audit for E8B clean RF runs."""

from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.training.direct_target_xgb_rf import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    utc_now_iso,
)

PHASE = "E8C"
SCRIPT_PATH = "scripts/analyze_clean_rf_policy_tail.py"
TARGET_COL = "target_net_profitable_after_exit"
RETURN_COL = "sim_net_return"
PAIR_COL = "pair_address"
SCORE_COL = "predicted_probability"

TOP_PCT_PERCENT_VALUES: list[float] = [
    0.02,
    0.05,
    0.10,
    0.20,
    0.25,
    0.50,
    1.00,
    2.00,
    5.00,
    10.00,
]

CLASSIFICATION_LABELS = (
    "ROBUST_STRATEGY_CANDIDATE",
    "RARE_WINNER_DETECTOR",
    "VALIDATION_ONLY_ARTIFACT",
    "TEST_ONLY_ARTIFACT",
    "LOTTERY_ARTIFACT",
    "NO_USABLE_SIGNAL",
    "UNUSABLE_OFFLINE",
)

PREDICTION_FILE_PATTERN = re.compile(
    r"^(?P<dataset>.+)_(?P<split>validation|test)_predictions\.(?:csv|parquet)$",
    re.I,
)


def validate_top_pct_percent(top_pct_percent: float) -> float:
    value = float(top_pct_percent)
    if value <= 0 or value > 100:
        raise ValueError(f"top_pct_percent must be > 0 and <= 100, got {value}")
    return value


def top_fraction_from_percent(top_pct_percent: float) -> float:
    validated = validate_top_pct_percent(top_pct_percent)
    return validated / 100.0


def selected_count_from_rows(rows: int, top_pct_percent: float) -> int:
    if rows <= 0:
        return 0
    top_fraction = top_fraction_from_percent(top_pct_percent)
    return max(1, int(math.ceil(rows * top_fraction)))


def normalize_id_column(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip()


@dataclass
class PredictionDataset:
    dataset_name: str
    validation_path: Path
    test_path: Path
    filter_name: str | None = None
    horizon: str | None = None
    exit_policy_id: str | None = None


@dataclass
class AuditConfig:
    run_dir: Path
    output_dir: Path | None = None


@dataclass
class AuditState:
    validation_metrics: list[dict[str, Any]] = field(default_factory=list)
    test_metrics: list[dict[str, Any]] = field(default_factory=list)
    validation_candidates: list[dict[str, Any]] = field(default_factory=list)
    validation_selected: list[dict[str, Any]] = field(default_factory=list)
    test_applied: list[dict[str, Any]] = field(default_factory=list)
    pair_concentration: list[dict[str, Any]] = field(default_factory=list)
    robustness: list[dict[str, Any]] = field(default_factory=list)
    final_classification: list[dict[str, Any]] = field(default_factory=list)
    identity_sanity: list[dict[str, Any]] = field(default_factory=list)
    prediction_files: list[str] = field(default_factory=list)
    validation_row_count: int = 0
    test_row_count: int = 0
    join_performed: bool = False
    join_keys: list[str] = field(default_factory=list)


def make_output_dir(run_dir: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = run_dir / f"e8c_policy_tail_audit_{ts}"
    (out / "reports").mkdir(parents=True, exist_ok=True)
    return out


def discover_prediction_datasets(predictions_dir: Path) -> list[PredictionDataset]:
    by_dataset: dict[str, dict[str, Path]] = {}
    for path in sorted(predictions_dir.iterdir()):
        if not path.is_file():
            continue
        match = PREDICTION_FILE_PATTERN.match(path.name)
        if match is None:
            continue
        dataset_name = match.group("dataset")
        split = match.group("split").lower()
        by_dataset.setdefault(dataset_name, {})[split] = path

    datasets: list[PredictionDataset] = []
    for dataset_name, splits in sorted(by_dataset.items()):
        if "validation" not in splits or "test" not in splits:
            continue
        datasets.append(
            PredictionDataset(
                dataset_name=dataset_name,
                validation_path=splits["validation"],
                test_path=splits["test"],
            )
        )
    return datasets


def load_predictions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    df["_row_order"] = np.arange(len(df))
    for col in ("candidate_id", "target_row_id", "candidate_policy_id"):
        if col in df.columns:
            df[col] = normalize_id_column(df[col])
    if TARGET_COL in df.columns:
        df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype(int)
    if SCORE_COL in df.columns:
        df[SCORE_COL] = pd.to_numeric(df[SCORE_COL], errors="coerce").fillna(0.0)
    return df


def infer_metadata(df: pd.DataFrame, dataset: PredictionDataset) -> None:
    for col, attr in (("filter", "filter_name"), ("horizon", "horizon"), ("exit_policy_id", "exit_policy_id")):
        if col in df.columns and df[col].notna().any():
            setattr(dataset, attr, str(df[col].dropna().iloc[0]))


def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "_row_order" not in out.columns:
        out = out.reset_index(drop=True)
        out["_row_order"] = np.arange(len(out))
    if SCORE_COL not in out.columns:
        out[SCORE_COL] = 0.0
    else:
        out[SCORE_COL] = pd.to_numeric(out[SCORE_COL], errors="coerce").fillna(0.0)
    if TARGET_COL not in out.columns:
        out[TARGET_COL] = 0
    return out


def rank_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = prepare_frame(df)
    sort_cols = [SCORE_COL]
    ascending = [False]
    for col in ("target_row_id", "candidate_id", "_row_order"):
        if col in out.columns:
            sort_cols.append(col)
            ascending.append(True)
    return out.sort_values(sort_cols, ascending=ascending, kind="mergesort").reset_index(drop=True)


def select_top_rows(df: pd.DataFrame, top_pct_percent: float) -> tuple[pd.DataFrame, int, bool]:
    ranked = rank_frame(df)
    rows = len(ranked)
    k = selected_count_from_rows(rows, top_pct_percent)
    selected = ranked.iloc[:k].copy()
    tie_at_boundary = False
    if k < rows and k > 0:
        boundary_score = ranked.iloc[k - 1][SCORE_COL]
        next_score = ranked.iloc[k][SCORE_COL]
        tie_at_boundary = bool(boundary_score == next_score)
    return selected, k, tie_at_boundary


def recall_at_top_pct(y_true: np.ndarray, y_score: np.ndarray, top_pct_percent: float) -> float | None:
    positives = float(y_true.sum())
    if positives == 0 or len(y_true) == 0:
        return None
    k = selected_count_from_rows(len(y_true), top_pct_percent)
    order = np.argsort(-y_score)
    return float(y_true[order[:k]].sum() / positives)


def economic_available(df: pd.DataFrame) -> bool:
    return RETURN_COL in df.columns and df[RETURN_COL].notna().any()


def return_stats(selected: pd.DataFrame) -> dict[str, float | None]:
    if RETURN_COL not in selected.columns or selected.empty:
        return {
            "selected_average_sim_net_return": None,
            "selected_total_sim_net_return": None,
            "selected_median_sim_net_return": None,
            "selected_min_sim_net_return": None,
            "selected_max_sim_net_return": None,
        }
    returns = pd.to_numeric(selected[RETURN_COL], errors="coerce").dropna()
    if returns.empty:
        return {
            "selected_average_sim_net_return": None,
            "selected_total_sim_net_return": None,
            "selected_median_sim_net_return": None,
            "selected_min_sim_net_return": None,
            "selected_max_sim_net_return": None,
        }
    return {
        "selected_average_sim_net_return": float(returns.mean()),
        "selected_total_sim_net_return": float(returns.sum()),
        "selected_median_sim_net_return": float(returns.median()),
        "selected_min_sim_net_return": float(returns.min()),
        "selected_max_sim_net_return": float(returns.max()),
    }


def pair_stats(selected: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {
        "selected_unique_pairs": 0,
        "selected_top_pair_share": None,
        "selected_positive_top_pair_share": None,
        "selected_pair_count": 0,
        "top_pair_address": None,
        "top_pair_selected_count": None,
        "top_pair_selected_total_sim_net_return": None,
        "top_pair_selected_positive_count": None,
        "top_pair_selected_share": None,
        "top_pair_positive_share": None,
    }
    if PAIR_COL not in selected.columns or selected.empty:
        return out

    pair_counts = selected[PAIR_COL].astype(str).value_counts()
    out["selected_unique_pairs"] = int(pair_counts.shape[0])
    out["selected_pair_count"] = int(pair_counts.shape[0])
    out["selected_top_pair_share"] = float(pair_counts.iloc[0] / len(selected))

    top_pair = str(pair_counts.index[0])
    top_rows = selected[selected[PAIR_COL].astype(str) == top_pair]
    out["top_pair_address"] = top_pair
    out["top_pair_selected_count"] = int(len(top_rows))
    out["top_pair_selected_share"] = float(len(top_rows) / len(selected))
    if RETURN_COL in top_rows.columns:
        out["top_pair_selected_total_sim_net_return"] = float(
            pd.to_numeric(top_rows[RETURN_COL], errors="coerce").fillna(0).sum()
        )
    if TARGET_COL in top_rows.columns:
        pos = int((top_rows[TARGET_COL] == 1).sum())
        out["top_pair_selected_positive_count"] = pos
        positives = selected[selected[TARGET_COL] == 1]
        if len(positives):
            pos_pair_counts = positives[PAIR_COL].astype(str).value_counts()
            out["selected_positive_top_pair_share"] = float(pos_pair_counts.iloc[0] / len(positives))
            out["top_pair_positive_share"] = float(pos / len(positives)) if len(positives) else None
    return out


def precision_on_frame(frame: pd.DataFrame) -> float | None:
    if frame.empty or TARGET_COL not in frame.columns:
        return None
    return float(frame[TARGET_COL].mean())


def total_return_on_frame(frame: pd.DataFrame) -> float | None:
    if frame.empty or RETURN_COL not in frame.columns:
        return None
    returns = pd.to_numeric(frame[RETURN_COL], errors="coerce").dropna()
    if returns.empty:
        return None
    return float(returns.sum())


def compute_robustness_on_selected(
    selected: pd.DataFrame,
    *,
    baseline_precision: float | None,
    baseline_total: float | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "baseline_selected_total_sim_net_return": baseline_total,
        "remove_best_selected_trade_total_sim_net_return": None,
        "remove_worst_selected_trade_total_sim_net_return": None,
        "remove_top_selected_pair_total_sim_net_return": None,
        "remove_best_selected_pair_total_sim_net_return": None,
        "leave_one_top_pair_out_total_sim_net_return": None,
        "baseline_precision_at_top_pct": baseline_precision,
        "remove_top_pair_precision_at_top_pct": None,
        "selected_unique_pairs_after_remove_top_pair": None,
        "selected_return_without_top_pair": None,
        "hard_robustness_gate_status": "UNUSABLE_OFFLINE",
    }
    if selected.empty:
        result["hard_robustness_gate_status"] = (
            "UNUSABLE_OFFLINE" if baseline_total is None else "NO_USABLE_SIGNAL"
        )
        return result

    if RETURN_COL in selected.columns:
        returns = pd.to_numeric(selected[RETURN_COL], errors="coerce")
        best_idx = returns.idxmax()
        worst_idx = returns.idxmin()
        result["remove_best_selected_trade_total_sim_net_return"] = total_return_on_frame(
            selected.drop(index=best_idx)
        )
        result["remove_worst_selected_trade_total_sim_net_return"] = total_return_on_frame(
            selected.drop(index=worst_idx)
        )

    if PAIR_COL in selected.columns:
        pair_counts = selected[PAIR_COL].astype(str).value_counts()
        top_pair = str(pair_counts.index[0])
        wo_top = selected[selected[PAIR_COL].astype(str) != top_pair]
        result["remove_top_selected_pair_total_sim_net_return"] = total_return_on_frame(wo_top)
        result["leave_one_top_pair_out_total_sim_net_return"] = total_return_on_frame(wo_top)
        result["remove_top_pair_precision_at_top_pct"] = precision_on_frame(wo_top)
        result["selected_unique_pairs_after_remove_top_pair"] = (
            int(wo_top[PAIR_COL].astype(str).nunique()) if not wo_top.empty else 0
        )
        result["selected_return_without_top_pair"] = total_return_on_frame(wo_top)

        if RETURN_COL in selected.columns:
            pair_totals = selected.groupby(selected[PAIR_COL].astype(str))[RETURN_COL].apply(
                lambda s: pd.to_numeric(s, errors="coerce").fillna(0).sum()
            )
            best_pair = str(pair_totals.idxmax())
            wo_best_pair = selected[selected[PAIR_COL].astype(str) != best_pair]
            result["remove_best_selected_pair_total_sim_net_return"] = total_return_on_frame(wo_best_pair)

    result["hard_robustness_gate_status"] = apply_hard_robustness_gate(
        baseline_total=baseline_total,
        remove_best_total=result["remove_best_selected_trade_total_sim_net_return"],
        remove_top_pair_total=result["remove_top_selected_pair_total_sim_net_return"],
    )
    return result


def apply_hard_robustness_gate(
    *,
    baseline_total: float | None,
    remove_best_total: float | None,
    remove_top_pair_total: float | None,
) -> str:
    if baseline_total is None:
        return "UNUSABLE_OFFLINE"
    if baseline_total <= 0:
        return "UNUSABLE_OFFLINE"
    if remove_best_total is None or remove_best_total <= 0:
        return "LOTTERY_ARTIFACT"
    if remove_top_pair_total is None or remove_top_pair_total <= 0:
        return "RARE_WINNER_DETECTOR"
    return "ROBUST_STRATEGY_CANDIDATE"


def compute_tail_metrics_row(
    df: pd.DataFrame,
    *,
    dataset: PredictionDataset,
    split: str,
    top_pct_percent: float,
) -> dict[str, Any]:
    frame = prepare_frame(df)
    top_fraction = top_fraction_from_percent(top_pct_percent)
    y_true = frame[TARGET_COL].to_numpy()
    y_score = frame[SCORE_COL].to_numpy()
    selected, selected_count, tie_at_boundary = select_top_rows(frame, top_pct_percent)
    sel_y = selected[TARGET_COL].to_numpy() if TARGET_COL in selected.columns else np.array([], dtype=int)
    positives = int((y_true == 1).sum())
    row: dict[str, Any] = {
        "dataset_name": dataset.dataset_name,
        "filter": dataset.filter_name,
        "horizon": dataset.horizon,
        "exit_policy_id": dataset.exit_policy_id,
        "split": split,
        "top_pct_percent": top_pct_percent,
        "top_fraction": top_fraction,
        "rows": int(len(frame)),
        "positives": positives,
        "positive_rate": float(y_true.mean()) if len(y_true) else None,
        "selected_count": selected_count,
        "selected_positive_count": int((sel_y == 1).sum()) if len(sel_y) else 0,
        "precision_at_top_pct": float(sel_y.mean()) if len(sel_y) else None,
        "recall_at_top_pct": recall_at_top_pct(y_true, y_score, top_pct_percent),
        "false_positive_count": int((sel_y == 0).sum()) if len(sel_y) else 0,
        "tie_at_boundary": tie_at_boundary,
        "economic_metrics_available": economic_available(frame),
        **return_stats(selected),
        **pair_stats(selected),
    }
    return row


def is_robust_strategy_eligible(row: dict[str, Any], robust: dict[str, Any]) -> bool:
    if not row.get("economic_metrics_available"):
        return False
    total = row.get("selected_total_sim_net_return")
    avg = row.get("selected_average_sim_net_return")
    if total is None or avg is None or total <= 0 or avg <= 0:
        return False
    if (row.get("selected_count") or 0) < 50:
        return False
    if (row.get("selected_unique_pairs") or 0) < 5:
        return False
    baseline_rate = row.get("positive_rate") or 0.0
    prec = row.get("precision_at_top_pct") or 0.0
    if prec <= baseline_rate:
        return False
    if (row.get("selected_top_pair_share") or 1.0) > 0.50:
        return False
    if (robust.get("remove_best_selected_trade_total_sim_net_return") or 0) <= 0:
        return False
    if (robust.get("remove_top_selected_pair_total_sim_net_return") or 0) <= 0:
        return False
    return True


def is_rare_winner_eligible(row: dict[str, Any], robust: dict[str, Any]) -> bool:
    if not row.get("economic_metrics_available"):
        return False
    total = row.get("selected_total_sim_net_return")
    if total is None or total <= 0:
        return False
    if (row.get("selected_count") or 0) < 5:
        return False
    concentrated = (row.get("selected_top_pair_share") or 0) > 0.50
    pair_fragile = (robust.get("remove_top_selected_pair_total_sim_net_return") or 0) <= 0
    return concentrated or pair_fragile


def select_validation_policy(
    val_metrics: list[dict[str, Any]],
    robustness_by_key: dict[tuple[str, float], dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    candidates: list[dict[str, Any]] = []
    for row in val_metrics:
        key = (row["dataset_name"], float(row["top_pct_percent"]))
        robust = robustness_by_key.get(key, {})
        robust_eligible = is_robust_strategy_eligible(row, robust)
        rare_eligible = is_rare_winner_eligible(row, robust)
        if not robust_eligible and not rare_eligible:
            continue
        candidate = {
            **row,
            "candidate_type": (
                "ROBUST_STRATEGY_CANDIDATE" if robust_eligible else "RARE_WINNER_DETECTOR"
            ),
            "robust_strategy_eligible": robust_eligible,
            "rare_winner_eligible": rare_eligible,
            **{f"robust_{k}": v for k, v in robust.items()},
        }
        candidates.append(candidate)

    if not candidates:
        return [], None

    def rank_key(c: dict[str, Any]) -> tuple[int, float, float]:
        type_rank = 0 if c["candidate_type"] == "ROBUST_STRATEGY_CANDIDATE" else 1
        return (
            type_rank,
            -(c.get("selected_total_sim_net_return") or -1e18),
            -(c.get("precision_at_top_pct") or 0),
        )

    candidates.sort(key=rank_key)
    selected = dict(candidates[0])
    selected["selection_source"] = "validation_only"
    return candidates, selected


def test_evaluation_outcome(
    val_row: dict[str, Any] | None,
    test_row: dict[str, Any] | None,
    test_robust: dict[str, Any],
) -> str:
    if val_row is None or test_row is None:
        return "no_validation_policy"
    val_total = val_row.get("selected_total_sim_net_return")
    test_total = test_row.get("selected_total_sim_net_return")
    if test_total is None or test_total <= 0:
        return "collapses_to_loss"
    if val_total is not None and test_total < val_total * 0.25:
        return "weakens_validation"
    if (test_row.get("selected_top_pair_share") or 0) > 0.50:
        return "depends_on_one_pair"
    if test_robust.get("hard_robustness_gate_status") == "LOTTERY_ARTIFACT":
        return "weakens_validation"
    val_prec = val_row.get("precision_at_top_pct") or 0
    test_prec = test_row.get("precision_at_top_pct") or 0
    if test_prec >= val_prec * 0.8 and (test_total or 0) > 0:
        return "confirms_validation"
    return "weakens_validation"


def classify_dataset(
    *,
    dataset: PredictionDataset,
    val_selected: dict[str, Any] | None,
    test_row: dict[str, Any] | None,
    val_robust: dict[str, Any],
    test_robust: dict[str, Any],
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, Any]:
    econ_val = economic_available(val_df)
    econ_test = economic_available(test_df)
    if not econ_val and not econ_test:
        classification = "UNUSABLE_OFFLINE"
        reason = "sim_net_return missing in validation and test predictions"
    elif val_selected is None:
        test_positive_any = False
        if econ_test:
            for pct in TOP_PCT_PERCENT_VALUES:
                row = compute_tail_metrics_row(test_df, dataset=dataset, split="test", top_pct_percent=pct)
                if (row.get("selected_total_sim_net_return") or 0) > 0:
                    test_positive_any = True
                    break
        if test_positive_any:
            classification = "TEST_ONLY_ARTIFACT"
            reason = "test shows positive tail return but no validation policy passed economic rules"
        else:
            classification = "NO_USABLE_SIGNAL"
            reason = "no validation policy passed economic and robustness eligibility rules"
    else:
        val_total = val_selected.get("selected_total_sim_net_return")
        test_total = test_row.get("selected_total_sim_net_return") if test_row else None
        val_gate = val_robust.get("hard_robustness_gate_status", "UNUSABLE_OFFLINE")
        test_gate = test_robust.get("hard_robustness_gate_status", "UNUSABLE_OFFLINE")

        if val_total is None or val_total <= 0:
            classification = "UNUSABLE_OFFLINE" if not econ_val else "NO_USABLE_SIGNAL"
            reason = "validation selected policy has non-positive total sim_net_return"
        elif val_gate == "LOTTERY_ARTIFACT" or test_gate == "LOTTERY_ARTIFACT":
            classification = "LOTTERY_ARTIFACT"
            reason = "hard robustness gate: performance collapses after removing best trade"
        elif test_total is None or test_total <= 0:
            classification = "VALIDATION_ONLY_ARTIFACT"
            reason = "validation positive but test selected_total_sim_net_return <= 0"
        elif val_selected.get("candidate_type") == "ROBUST_STRATEGY_CANDIDATE" and test_gate == "ROBUST_STRATEGY_CANDIDATE":
            if is_robust_strategy_eligible(test_row or {}, test_robust):
                classification = "ROBUST_STRATEGY_CANDIDATE"
                reason = "validation and test pass robust economic and pair-diversification gates"
            else:
                classification = "RARE_WINNER_DETECTOR"
                reason = "validation robust but test fails full robust gate; rare-winner evidence remains"
        elif (val_total or 0) > 0 and (test_total or 0) > 0:
            classification = "RARE_WINNER_DETECTOR"
            reason = "positive validation/test tail economics with pair concentration or fragile robustness"
        else:
            classification = "NO_USABLE_SIGNAL"
            reason = "validation policy did not produce usable offline economic signal on test"

    return {
        "dataset_name": dataset.dataset_name,
        "filter": dataset.filter_name,
        "horizon": dataset.horizon,
        "exit_policy_id": dataset.exit_policy_id,
        "final_classification": classification,
        "classification_reason": reason,
        "validation_selected_top_pct_percent": (
            val_selected.get("top_pct_percent") if val_selected else None
        ),
        "validation_selected_total_sim_net_return": (
            val_selected.get("selected_total_sim_net_return") if val_selected else None
        ),
        "test_applied_total_sim_net_return": (
            test_row.get("selected_total_sim_net_return") if test_row else None
        ),
        "validation_hard_robustness_gate_status": val_robust.get("hard_robustness_gate_status"),
        "test_hard_robustness_gate_status": test_robust.get("hard_robustness_gate_status"),
        "test_evaluation_outcome": test_evaluation_outcome(val_selected, test_row, test_robust),
    }


def analyze_dataset(
    dataset: PredictionDataset,
    state: AuditState,
) -> None:
    val_df = load_predictions(dataset.validation_path)
    test_df = load_predictions(dataset.test_path)
    infer_metadata(val_df, dataset)

    state.prediction_files.extend([str(dataset.validation_path), str(dataset.test_path)])
    state.validation_row_count += len(val_df)
    state.test_row_count += len(test_df)

    val_metrics_for_dataset: list[dict[str, Any]] = []
    test_metrics_for_dataset: list[dict[str, Any]] = []
    robustness_by_key: dict[tuple[str, float], dict[str, Any]] = {}

    for top_pct_percent in TOP_PCT_PERCENT_VALUES:
        val_row = compute_tail_metrics_row(
            val_df, dataset=dataset, split="validation", top_pct_percent=top_pct_percent
        )
        test_row = compute_tail_metrics_row(
            test_df, dataset=dataset, split="test", top_pct_percent=top_pct_percent
        )
        val_metrics_for_dataset.append(val_row)
        test_metrics_for_dataset.append(test_row)
        state.validation_metrics.append(val_row)
        state.test_metrics.append(test_row)

        for split_name, row, frame in (
            ("validation", val_row, val_df),
            ("test", test_row, test_df),
        ):
            selected, _, _ = select_top_rows(frame, top_pct_percent)
            robust = compute_robustness_on_selected(
                selected,
                baseline_precision=row.get("precision_at_top_pct"),
                baseline_total=row.get("selected_total_sim_net_return"),
            )
            key = (dataset.dataset_name, float(top_pct_percent))
            if split_name == "validation":
                robustness_by_key[key] = robust
            state.robustness.append(
                {
                    "dataset_name": dataset.dataset_name,
                    "filter": dataset.filter_name,
                    "horizon": dataset.horizon,
                    "exit_policy_id": dataset.exit_policy_id,
                    "split": split_name,
                    "top_pct_percent": top_pct_percent,
                    **robust,
                }
            )
            state.pair_concentration.append(
                {
                    "dataset_name": dataset.dataset_name,
                    "filter": dataset.filter_name,
                    "horizon": dataset.horizon,
                    "exit_policy_id": dataset.exit_policy_id,
                    "split": split_name,
                    "top_pct_percent": top_pct_percent,
                    "selected_unique_pairs": row.get("selected_unique_pairs"),
                    "selected_top_pair_share": row.get("selected_top_pair_share"),
                    "selected_positive_top_pair_share": row.get("selected_positive_top_pair_share"),
                    "top_pair_address": row.get("top_pair_address"),
                    "top_pair_selected_share": row.get("top_pair_selected_share"),
                }
            )

    candidates, val_selected = select_validation_policy(val_metrics_for_dataset, robustness_by_key)
    for c in candidates:
        state.validation_candidates.append(c)

    val_robust: dict[str, Any] = {}
    test_robust: dict[str, Any] = {}
    test_row: dict[str, Any] | None = None

    if val_selected:
        state.validation_selected.append(val_selected)
        top_pct = float(val_selected["top_pct_percent"])
        test_row = next(r for r in test_metrics_for_dataset if float(r["top_pct_percent"]) == top_pct)
        val_robust = robustness_by_key[(dataset.dataset_name, top_pct)]
        test_robust = next(
            r
            for r in state.robustness
            if r["dataset_name"] == dataset.dataset_name
            and r["split"] == "test"
            and float(r["top_pct_percent"]) == top_pct
        )
        state.test_applied.append(
            {
                **test_row,
                "validation_selected_top_pct_percent": top_pct,
                "validation_selected_total_sim_net_return": val_selected.get(
                    "selected_total_sim_net_return"
                ),
                "validation_candidate_type": val_selected.get("candidate_type"),
                "selection_source": "validation_only",
                "test_evaluation_outcome": test_evaluation_outcome(val_selected, test_row, test_robust),
            }
        )

    state.final_classification.append(
        classify_dataset(
            dataset=dataset,
            val_selected=val_selected,
            test_row=test_row,
            val_robust=val_robust,
            test_robust=test_robust,
            val_df=val_df,
            test_df=test_df,
        )
    )


def write_outputs(config: AuditConfig, state: AuditState) -> Path:
    output_dir = config.output_dir or make_output_dir(config.run_dir)
    reports = output_dir / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    val_df = pd.DataFrame(state.validation_metrics)
    test_df = pd.DataFrame(state.test_metrics)
    atomic_write_parquet(val_df, reports / "e8c_tail_metrics_validation.parquet")
    atomic_write_parquet(test_df, reports / "e8c_tail_metrics_test.parquet")
    atomic_write_csv(val_df, reports / "e8c_tail_metrics_validation.csv")
    atomic_write_csv(test_df, reports / "e8c_tail_metrics_test.csv")

    atomic_write_csv(pd.DataFrame(state.validation_candidates), reports / "e8c_validation_policy_candidates.csv")
    atomic_write_csv(pd.DataFrame(state.validation_selected), reports / "e8c_validation_selected_policies.csv")
    atomic_write_csv(pd.DataFrame(state.test_applied), reports / "e8c_test_applied_selected_policies.csv")
    atomic_write_csv(pd.DataFrame(state.pair_concentration), reports / "e8c_pair_tail_concentration.csv")
    atomic_write_csv(pd.DataFrame(state.robustness), reports / "e8c_tail_robustness_diagnostic.csv")
    atomic_write_csv(pd.DataFrame(state.final_classification), reports / "e8c_final_classification.csv")

    if state.identity_sanity:
        atomic_write_csv(pd.DataFrame(state.identity_sanity), reports / "e8c_identity_join_sanity_check.csv")

    counts = state.final_classification
    classification_counts = {label: 0 for label in CLASSIFICATION_LABELS}
    for row in counts:
        label = row.get("final_classification", "NO_USABLE_SIGNAL")
        classification_counts[label] = classification_counts.get(label, 0) + 1

    summary_lines = [
        "Phase E8C Prediction Tail & Economic Policy Audit",
        f"input_run_dir: {config.run_dir}",
        f"output_dir: {output_dir}",
        f"prediction_files_analyzed: {len(state.prediction_files)}",
        f"validation_rows: {state.validation_row_count}",
        f"test_rows: {state.test_row_count}",
        f"validation_policies_selected: {len(state.validation_selected)}",
        "",
        "Final classification counts:",
    ]
    for label in CLASSIFICATION_LABELS:
        summary_lines.append(f"  {label}: {classification_counts.get(label, 0)}")
    summary_lines.extend(
        [
            "",
            f"join_performed: {state.join_performed}",
            "",
            "Leakage / safety:",
            "  no_training_performed = true",
            "  no_features_fit = true",
            "  sim_net_return_used_for_evaluation_only = true",
            "  pair_address_used_for_diagnostics_only = true",
            "  no_runtime_changes = true",
            "  no_db_writes = true",
            "  reservoir_scoring_performed = false",
        ]
    )

    robust_n = classification_counts.get("ROBUST_STRATEGY_CANDIDATE", 0)
    rare_n = classification_counts.get("RARE_WINNER_DETECTOR", 0)
    if robust_n > 0:
        recommendation = "One or more datasets show robust strategy candidate signal; review e8c_validation_selected_policies.csv before any offline follow-up."
    elif rare_n > 0:
        recommendation = "No robust strategy candidates; rare-winner detector signal may exist for research watchlists only."
    else:
        recommendation = "No usable economic signal or robust/rare-winner approval from E8B clean RF predictions; do not connect to runtime."

    summary_lines.extend(["", f"Recommendation: {recommendation}"])
    atomic_write_text("\n".join(summary_lines) + "\n", reports / "e8c_decision_summary.txt")

    manifest = {
        "phase": PHASE,
        "created_at": utc_now_iso(),
        "script_path": SCRIPT_PATH,
        "input_e8b_run_directory": str(config.run_dir),
        "output_directory": str(output_dir),
        "prediction_files_analyzed": state.prediction_files,
        "top_pct_percent_values": TOP_PCT_PERCENT_VALUES,
        "selection_rules": {
            "robust_strategy_candidate": {
                "selected_count_min": 50,
                "selected_unique_pairs_min": 5,
                "selected_average_sim_net_return_min_exclusive": 0,
                "selected_total_sim_net_return_min_exclusive": 0,
                "precision_at_top_pct_gt_baseline_positive_rate": True,
                "selected_top_pair_share_max_inclusive": 0.50,
                "remove_best_selected_trade_total_sim_net_return_min_exclusive": 0,
                "remove_top_selected_pair_total_sim_net_return_min_exclusive": 0,
            },
            "rare_winner_detector": {
                "selected_total_sim_net_return_min_exclusive": 0,
                "selected_count_min": 5,
                "pair_concentration_or_remove_top_pair_non_positive": True,
            },
            "validation_only_selection": True,
            "test_applied_without_reselection": True,
        },
        "hard_robustness_gate_rules": {
            "baseline_total_le_zero": "NO_USABLE_SIGNAL or UNUSABLE_OFFLINE",
            "remove_best_trade_le_zero": "LOTTERY_ARTIFACT unless baseline <= 0 then UNUSABLE_OFFLINE",
            "remove_top_pair_le_zero_with_positive_best_trade": "RARE_WINNER_DETECTOR not ROBUST_STRATEGY_CANDIDATE",
            "robust_strategy_requires_all_positive_removals_and_pair_limits": True,
            "test_only_artifact_when_test_positive_no_validation_policy": True,
            "validation_only_artifact_when_validation_positive_test_non_positive": True,
        },
        "join_performed": state.join_performed,
        "join_keys": state.join_keys,
        "validation_row_count": state.validation_row_count,
        "test_row_count": state.test_row_count,
        "datasets_analyzed": len({r["dataset_name"] for r in state.final_classification}),
        "validation_policies_selected": len(state.validation_selected),
        "classification_counts": classification_counts,
        "recommendation": recommendation,
        "no_training_performed": True,
        "no_features_fit": True,
        "sim_net_return_used_for_evaluation_only": True,
        "pair_address_used_for_diagnostics_only": True,
        "no_runtime_changes": True,
        "no_db_writes": True,
        "reservoir_scoring_performed": False,
        "python_version": sys.version,
    }
    atomic_write_json(manifest, reports / "e8c_run_manifest.json")
    return output_dir


def run_audit(config: AuditConfig) -> dict[str, Any]:
    predictions_dir = config.run_dir / "predictions"
    if not predictions_dir.is_dir():
        raise FileNotFoundError(f"Missing predictions directory: {predictions_dir}")

    output_dir = config.output_dir or make_output_dir(config.run_dir)
    config.output_dir = output_dir
    state = AuditState()

    datasets = discover_prediction_datasets(predictions_dir)
    if not datasets:
        raise RuntimeError("No validation/test prediction pairs discovered")

    for dataset in datasets:
        analyze_dataset(dataset, state)

    write_outputs(config, state)

    classification_counts = {label: 0 for label in CLASSIFICATION_LABELS}
    for row in state.final_classification:
        classification_counts[row.get("final_classification", "NO_USABLE_SIGNAL")] = (
            classification_counts.get(row.get("final_classification", "NO_USABLE_SIGNAL"), 0) + 1
        )

    return {
        "output_dir": str(output_dir),
        "prediction_files_analyzed": len(state.prediction_files),
        "validation_rows": state.validation_row_count,
        "test_rows": state.test_row_count,
        "validation_policies_selected": len(state.validation_selected),
        "classification_counts": classification_counts,
        "join_performed": state.join_performed,
        "datasets_analyzed": len(datasets),
    }
