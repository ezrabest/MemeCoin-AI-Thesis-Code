#!/usr/bin/env python3
"""Predictive policy backtest with validation-only threshold selection."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.training.baseline_model import MODELS_DIR, load_baseline_metrics
from app.training.config import (
    TRADE_CLASS_MULTIPLIERS,
    get_whale_wave_aggressive_threshold,
    get_whale_wave_normal_threshold,
    get_round_trip_fee_pct,
)

VALIDATION_PREDICTIONS_PATH = MODELS_DIR / "predictions_validation.parquet"
TEST_PREDICTIONS_PATH = MODELS_DIR / "predictions_test.parquet"
BACKTEST_DIR = ROOT / "data" / "training" / "policy_backtests"
REPORT_PATH = BACKTEST_DIR / "predicted_policy_report.json"
GRID_PATH = BACKTEST_DIR / "predicted_policy_grid.parquet"

PRIMARY_TARGET = "label_profitable_after_fees_4h"
SECONDARY_TARGET = "label_profitable_after_fees_1h"
PUMP_TARGET = "big_pump_1h"
LEGACY_THRESHOLD = 0.20

TARGET_HORIZON = {
    PRIMARY_TARGET: "4h",
    SECONDARY_TARGET: "1h",
}

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


def top_percent_probability_cutoff(probabilities: "np.ndarray", top_pct: float) -> float:
    """Return the minimum probability among the top *top_pct* validation rows."""
    import numpy as np

    probs = np.asarray(probabilities, dtype=float)
    probs = probs[np.isfinite(probs)]
    if probs.size == 0:
        return 1.0
    k = max(1, int(probs.size * top_pct / 100.0))
    sorted_desc = np.sort(probs)[::-1]
    return float(sorted_desc[min(k - 1, sorted_desc.size - 1)])


def policy_probability_cutoff(
    policy_name: str,
    validation_probabilities: "np.ndarray",
    *,
    rank_cutoffs: dict[str, float] | None = None,
) -> tuple[float | None, float | None]:
    """Return (probability_cutoff, top_percent) for a policy."""
    for name, top_pct in RANK_POLICIES:
        if policy_name == name:
            if rank_cutoffs is not None and policy_name in rank_cutoffs:
                return rank_cutoffs[policy_name], top_pct
            return top_percent_probability_cutoff(validation_probabilities, top_pct), top_pct
    for name, cutoff in FIXED_THRESHOLD_POLICIES:
        if policy_name == name:
            return cutoff, None
    raise ValueError(f"Unknown policy: {policy_name}")


def build_policy_mask(probabilities: "pd.Series", probability_cutoff: float) -> "pd.Series":
    return probabilities.astype(float).fillna(0.0) >= probability_cutoff


def profit_factor_from_net(net_returns: "np.ndarray") -> float | None:
    import numpy as np

    net = np.asarray(net_returns, dtype=float)
    if net.size == 0:
        return None
    wins = net[net > 0]
    losses = net[net < 0]
    if losses.size == 0 or losses.sum() == 0:
        return None
    if wins.size == 0:
        return 0.0
    return float(wins.sum() / abs(losses.sum()))


def max_drawdown_from_net(net_returns: "np.ndarray") -> float:
    import numpy as np

    net = np.asarray(net_returns, dtype=float)
    if net.size == 0:
        return 0.0
    cumulative = np.cumsum(net)
    if cumulative.size == 1:
        return float(min(0.0, cumulative[0]))
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    return float(drawdown.min())


def evaluate_policy_trades(
    frame: "pd.DataFrame",
    mask: "pd.Series",
    *,
    return_col: str,
    label_col: str,
    prob_col: str,
    fee_pct: float,
    size_series: "pd.Series | None" = None,
) -> dict[str, Any]:
    import numpy as np
    import pandas as pd

    if size_series is None:
        size_series = pd.Series(1.0, index=frame.index)

    traded = frame.loc[mask].copy()
    trade_count = int(len(traded))
    if trade_count == 0:
        return {
            "trade_count": 0,
            "zero_trade_policy": True,
            "win_rate": None,
            "total_return_after_fees": 0.0,
            "average_return": None,
            "median_return": None,
            "max_drawdown": 0.0,
            "profit_factor": None,
            "precision": None,
            "average_predicted_probability": None,
            "average_realized_return": None,
            "best_trades": [],
            "worst_trades": [],
        }

    gross = traded[return_col].astype(float)
    sizes = size_series.loc[traded.index].astype(float)
    net = (gross - fee_pct) * sizes
    net_values = net.to_numpy()

    y_true = traded[label_col].fillna(0).astype(int) if label_col in traded.columns else pd.Series(0, index=traded.index)
    probs = traded[prob_col].astype(float) if prob_col in traded.columns else pd.Series(0.0, index=traded.index)

    traded = traded.assign(_net=net, _prob=probs, _gross=gross)
    traded = traded.sort_values("event_timestamp")

    best_cols = ["symbol", "event_timestamp", return_col, prob_col, "_net"]
    if "whale_wave_score" in traded.columns:
        best_cols.append("whale_wave_score")

    return {
        "trade_count": trade_count,
        "zero_trade_policy": False,
        "win_rate": round(float((net > 0).mean()), 6),
        "total_return_after_fees": round(float(net.sum()), 6),
        "average_return": round(float(net.mean()), 6),
        "median_return": round(float(net.median()), 6),
        "max_drawdown": round(max_drawdown_from_net(traded["_net"].to_numpy()), 6),
        "profit_factor": (
            round(pf, 6) if (pf := profit_factor_from_net(net_values)) is not None else None
        ),
        "precision": round(float(y_true.mean()), 6),
        "average_predicted_probability": round(float(probs.mean()), 6),
        "average_realized_return": round(float(gross.mean()), 6),
        "best_trades": traded.nlargest(5, "_net")[best_cols].rename(columns={"_net": "net"}).to_dict(orient="records"),
        "worst_trades": traded.nsmallest(5, "_net")[best_cols].rename(columns={"_net": "net"}).to_dict(orient="records"),
    }


def select_best_validation_policy(
    validation_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Pick the policy that maximizes validation total return after fees."""
    candidates = [row for row in validation_rows if row.get("trade_count", 0) > 0]
    if not candidates:
        candidates = list(validation_rows)
    if not candidates:
        return None

    def sort_key(row: dict[str, Any]) -> tuple:
        total_return = row.get("total_return_after_fees")
        if total_return is None:
            total_return = float("-inf")
        max_dd = row.get("max_drawdown")
        if max_dd is None:
            max_dd = 0.0
        profit_factor = row.get("profit_factor")
        if profit_factor is None:
            profit_factor = float("-inf")
        win_rate = row.get("win_rate")
        if win_rate is None:
            win_rate = 0.0
        trade_count = row.get("trade_count", 0)
        return (
            total_return,
            max_dd,
            profit_factor,
            1 if trade_count > 0 else 0,
            win_rate,
            -trade_count,
        )

    return max(candidates, key=sort_key)


def _resolve_return_column(frame: "pd.DataFrame", horizon: str) -> str:
    for candidate in (f"target_return_{horizon}", f"future_return_{horizon}"):
        if candidate in frame.columns and frame[candidate].notna().any():
            return candidate
    raise RuntimeError(f"No realized {horizon} return column found for backtest.")


def _filter_best_model_predictions(preds: "pd.DataFrame", target_name: str, model_name: str) -> "pd.DataFrame":
    mask = (preds["target_name"] == target_name) & (preds["model_name"] == model_name)
    return preds.loc[mask].copy()


def _prepare_target_frame(preds: "pd.DataFrame", target_name: str, model_name: str) -> "pd.DataFrame":
    import pandas as pd

    subset = _filter_best_model_predictions(preds, target_name, model_name)
    if subset.empty:
        raise RuntimeError(f"No predictions for {target_name} / {model_name}")

    horizon = TARGET_HORIZON[target_name]
    return_col = _resolve_return_column(subset, horizon)

    frame = subset.copy()
    frame[target_name] = frame["predicted_probability"].astype(float)
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"], utc=True, errors="coerce")
    frame = frame[frame["event_timestamp"].notna() & frame[return_col].notna()].copy()
    frame = frame.sort_values("event_timestamp").reset_index(drop=True)
    frame["_return_col"] = return_col
    frame["_prob_col"] = target_name
    frame["_label_col"] = "y_true"
    return frame


def _derive_rank_cutoffs(validation_probabilities: "np.ndarray") -> dict[str, float]:
    return {
        policy_name: top_percent_probability_cutoff(validation_probabilities, top_pct)
        for policy_name, top_pct in RANK_POLICIES
    }


def _evaluate_all_policies_for_target(
    validation_frame: "pd.DataFrame",
    test_frame: "pd.DataFrame",
    *,
    target_name: str,
    model_name: str,
    fee_pct: float,
    rank_cutoffs: dict[str, float],
) -> list[dict[str, Any]]:
    import pandas as pd

    return_col = validation_frame["_return_col"].iloc[0]
    prob_col = target_name
    label_col = "y_true"
    val_probs = validation_frame[prob_col].astype(float).fillna(0.0)
    test_probs = test_frame[prob_col].astype(float).fillna(0.0)

    rows: list[dict[str, Any]] = []
    for policy_name in ALL_POLICY_NAMES:
        cutoff, top_percent = policy_probability_cutoff(
            policy_name,
            val_probs.to_numpy(),
            rank_cutoffs=rank_cutoffs,
        )
        assert cutoff is not None

        for split_name, frame, probs in (
            ("validation", validation_frame, val_probs),
            ("test", test_frame, test_probs),
        ):
            mask = build_policy_mask(probs, cutoff)
            metrics = evaluate_policy_trades(
                frame,
                mask,
                return_col=return_col,
                label_col=label_col,
                prob_col=prob_col,
                fee_pct=fee_pct,
            )
            rows.append({
                "split": split_name,
                "target_name": target_name,
                "model_name": model_name,
                "policy_name": policy_name,
                "probability_cutoff": round(cutoff, 6),
                "top_percent": top_percent,
                "is_oracle_backtest": False,
                **metrics,
            })
    return rows


def _assign_whale_sizes(
    frame: "pd.DataFrame",
    *,
    profitability_mask: "pd.Series",
    prof_prob_col: str,
    top_1_cutoff: float,
    top_2_cutoff: float,
) -> "pd.Series":
    import pandas as pd

    aggressive_th = get_whale_wave_aggressive_threshold()
    normal_th = get_whale_wave_normal_threshold()
    prof_prob = frame[prof_prob_col].astype(float).fillna(0.0)
    score = frame.get("whale_wave_score", pd.Series(0.0, index=frame.index)).astype(float).fillna(0.0)

    trade = profitability_mask
    cond_agg = trade & (score >= aggressive_th) & (prof_prob >= top_1_cutoff)
    cond_norm = trade & (~cond_agg) & (score >= normal_th) & (prof_prob >= top_2_cutoff)
    cond_small = trade & (~cond_agg) & (~cond_norm)

    predicted_class = pd.Series("NO_TRADE", index=frame.index)
    predicted_class = predicted_class.mask(cond_small, "SMALL_PROBE")
    predicted_class = predicted_class.mask(cond_norm, "NORMAL_TRADE")
    predicted_class = predicted_class.mask(cond_agg, "AGGRESSIVE_WHALE_TRADE")

    frame["predicted_trade_class"] = predicted_class
    sizes = predicted_class.map(TRADE_CLASS_MULTIPLIERS).fillna(0.0)
    frame["predicted_size"] = sizes
    return sizes


def _legacy_threshold_metrics(
    test_frame: "pd.DataFrame",
    *,
    target_name: str,
    model_name: str,
    fee_pct: float,
    threshold: float = LEGACY_THRESHOLD,
) -> dict[str, Any]:
    prob_col = target_name
    return_col = test_frame["_return_col"].iloc[0]
    mask = build_policy_mask(test_frame[prob_col], threshold)
    metrics = evaluate_policy_trades(
        test_frame,
        mask,
        return_col=return_col,
        label_col="y_true",
        prob_col=prob_col,
        fee_pct=fee_pct,
    )
    return {
        "policy_name": f"legacy_f1_threshold_{threshold:.2f}",
        "target_name": target_name,
        "model_name": model_name,
        "probability_cutoff": threshold,
        "split": "test",
        "description": "Previous policy: F1-selected validation threshold applied to test only.",
        **metrics,
    }


def run_backtest() -> dict[str, Any]:
    import pandas as pd

    metrics = load_baseline_metrics()
    if metrics is None:
        raise FileNotFoundError("baseline_metrics.json missing — run train_baseline_model.py first.")
    if not VALIDATION_PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            "Run python scripts/train_baseline_model.py first to create predictions_validation.parquet."
        )
    if not TEST_PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(
            "Run python scripts/train_baseline_model.py first to create predictions_test.parquet."
        )

    fee_pct = get_round_trip_fee_pct()
    val_preds = pd.read_parquet(VALIDATION_PREDICTIONS_PATH)
    test_preds = pd.read_parquet(TEST_PREDICTIONS_PATH)

    best_by_target = metrics.get("best_model_by_target", {})
    primary_model = (best_by_target.get(PRIMARY_TARGET) or {}).get("model_name")
    secondary_model = (best_by_target.get(SECONDARY_TARGET) or {}).get("model_name")
    if not primary_model or not secondary_model:
        raise RuntimeError("Missing best model metadata for profitability targets.")

    val_primary = _prepare_target_frame(val_preds, PRIMARY_TARGET, primary_model)
    test_primary = _prepare_target_frame(test_preds, PRIMARY_TARGET, primary_model)
    val_secondary = _prepare_target_frame(val_preds, SECONDARY_TARGET, secondary_model)
    test_secondary = _prepare_target_frame(test_preds, SECONDARY_TARGET, secondary_model)

    primary_rank_cutoffs = _derive_rank_cutoffs(
        val_primary[PRIMARY_TARGET].astype(float).fillna(0.0).to_numpy()
    )

    grid_rows: list[dict[str, Any]] = []
    grid_rows.extend(
        _evaluate_all_policies_for_target(
            val_primary,
            test_primary,
            target_name=PRIMARY_TARGET,
            model_name=primary_model,
            fee_pct=fee_pct,
            rank_cutoffs=primary_rank_cutoffs,
        )
    )
    grid_rows.extend(
        _evaluate_all_policies_for_target(
            val_secondary,
            test_secondary,
            target_name=SECONDARY_TARGET,
            model_name=secondary_model,
            fee_pct=fee_pct,
            rank_cutoffs=_derive_rank_cutoffs(
                val_secondary[SECONDARY_TARGET].astype(float).fillna(0.0).to_numpy()
            ),
        )
    )

    primary_validation = [row for row in grid_rows if row["split"] == "validation" and row["target_name"] == PRIMARY_TARGET]
    selected = select_best_validation_policy(primary_validation)
    if selected is None:
        raise RuntimeError("No policies evaluated for primary target.")

    selected_policy_name = selected["policy_name"]
    selected_cutoff = selected["probability_cutoff"]

    selected_val_result = selected
    selected_test_result = next(
        row for row in grid_rows
        if row["split"] == "test"
        and row["target_name"] == PRIMARY_TARGET
        and row["policy_name"] == selected_policy_name
    )

    whale_results: dict[str, Any] = {}
    for split_name, frame in (("validation", val_primary), ("test", test_primary)):
        profitability_mask = build_policy_mask(frame[PRIMARY_TARGET], selected_cutoff)
        sizes = _assign_whale_sizes(
            frame,
            profitability_mask=profitability_mask,
            prof_prob_col=PRIMARY_TARGET,
            top_1_cutoff=primary_rank_cutoffs["top_1_percent"],
            top_2_cutoff=primary_rank_cutoffs["top_2_percent"],
        )
        whale_mask = sizes > 0
        whale_metrics = evaluate_policy_trades(
            frame,
            whale_mask,
            return_col=frame["_return_col"].iloc[0],
            label_col="y_true",
            prob_col=PRIMARY_TARGET,
            fee_pct=fee_pct,
            size_series=sizes,
        )
        whale_results[split_name] = {
            "target_name": PRIMARY_TARGET,
            "model_name": primary_model,
            "policy_name": "predicted_whale_size_policy",
            "selected_profitability_policy": selected_policy_name,
            "probability_cutoff": selected_cutoff,
            "top_1_percent_cutoff": primary_rank_cutoffs["top_1_percent"],
            "top_2_percent_cutoff": primary_rank_cutoffs["top_2_percent"],
            "is_oracle_backtest": False,
            **whale_metrics,
        }

    legacy_comparison = {
        PRIMARY_TARGET: _legacy_threshold_metrics(
            test_primary,
            target_name=PRIMARY_TARGET,
            model_name=primary_model,
            fee_pct=fee_pct,
        ),
        SECONDARY_TARGET: _legacy_threshold_metrics(
            test_secondary,
            target_name=SECONDARY_TARGET,
            model_name=secondary_model,
            fee_pct=fee_pct,
        ),
    }

    policies_by_name = {
        row["policy_name"]: {
            "validation": row,
            "test": next(
                r for r in grid_rows
                if r["split"] == "test"
                and r["target_name"] == PRIMARY_TARGET
                and r["policy_name"] == row["policy_name"]
            ),
        }
        for row in primary_validation
    }

    return {
        "_grid_rows": grid_rows,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validation_predictions_path": str(VALIDATION_PREDICTIONS_PATH),
        "test_predictions_path": str(TEST_PREDICTIONS_PATH),
        "rows_validation_primary": int(len(val_primary)),
        "rows_test_primary": int(len(test_primary)),
        "is_oracle_backtest": False,
        "selection_method": "validation_only",
        "test_set_never_used_for_threshold_selection": True,
        "rank_cutoffs_derived_from_validation": True,
        "primary_target": PRIMARY_TARGET,
        "secondary_target": SECONDARY_TARGET,
        "policies_evaluated": ALL_POLICY_NAMES,
        "validation_derived_cutoffs": {
            PRIMARY_TARGET: {
                "rank_policies": primary_rank_cutoffs,
                "fixed_threshold_policies": {
                    name: cutoff for name, cutoff in FIXED_THRESHOLD_POLICIES
                },
            },
        },
        "selected_policy": {
            "policy_name": selected_policy_name,
            "probability_cutoff": selected_cutoff,
            "top_percent": selected.get("top_percent"),
            "target_name": PRIMARY_TARGET,
            "model_name": primary_model,
            "selection_objective": "maximize_total_return_after_fees_on_validation",
        },
        "selected_policy_validation_result": selected_val_result,
        "selected_policy_test_result": selected_test_result,
        "best_4h_policy": {
            "validation": selected_val_result,
            "test": selected_test_result,
        },
        "policies": policies_by_name,
        "secondary_target_results": {
            row["policy_name"]: {
                "validation": next(
                    r for r in grid_rows
                    if r["split"] == "validation"
                    and r["target_name"] == SECONDARY_TARGET
                    and r["policy_name"] == row["policy_name"]
                ),
                "test": next(
                    r for r in grid_rows
                    if r["split"] == "test"
                    and r["target_name"] == SECONDARY_TARGET
                    and r["policy_name"] == row["policy_name"]
                ),
            }
            for row in primary_validation
        },
        "comparison_to_legacy_threshold_0_20": legacy_comparison,
        "predicted_whale_size_policy": whale_results,
        "grid_path": str(GRID_PATH),
    }


def main() -> int:
    import pandas as pd

    report = run_backtest()
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)

    grid_rows = report.pop("_grid_rows", [])
    pd.DataFrame(grid_rows).to_parquet(GRID_PATH, index=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {REPORT_PATH}")
    print(f"Wrote {GRID_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
