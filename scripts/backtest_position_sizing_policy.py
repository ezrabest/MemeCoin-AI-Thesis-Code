#!/usr/bin/env python3
"""Oracle sanity-check backtest for position sizing policies (dataset only, not predictive)."""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.training.config import get_round_trip_fee_pct

TRAINING_DIR = ROOT / "data" / "training"
BACKTEST_DIR = TRAINING_DIR / "policy_backtests"
DATASET_PATH = TRAINING_DIR / "model_ready_dataset.parquet"

ORACLE_NOTE = (
    "This backtest uses realized labels and is a sanity-check upper bound, "
    "not a predictive trading backtest."
)
NEXT_STEP = (
    "Train baseline model and run policy using predicted probabilities on chronological test split."
)


def _load_dataset() -> "pd.DataFrame":
    import pandas as pd

    if DATASET_PATH.is_file():
        return pd.read_parquet(DATASET_PATH)
    csv_path = TRAINING_DIR / "model_ready_dataset.csv"
    if csv_path.is_file():
        return pd.read_csv(csv_path)
    raise FileNotFoundError("Run python scripts/build_training_dataset.py first.")


def _simulate(frame, mask, size_series, fee_pct: float) -> dict:
    traded = frame.loc[mask].copy()
    if traded.empty:
        return {
            "total_return_after_fees": 0.0,
            "max_drawdown": 0.0,
            "trade_count": 0,
            "win_rate": 0.0,
            "average_return": 0.0,
            "profit_factor": 0.0,
        }

    gross = traded["target_return_1h"].astype(float)
    net = (gross - fee_pct) * size_series.loc[traded.index].astype(float)
    cumulative = net.cumsum()
    running_max = cumulative.cummax()
    drawdown = cumulative - running_max

    wins = net[net > 0]
    losses = net[net < 0]
    profit_factor = float(wins.sum() / abs(losses.sum())) if not losses.empty and losses.sum() != 0 else float("inf")

    return {
        "total_return_after_fees": round(float(net.sum()), 6),
        "max_drawdown": round(float(drawdown.min()), 6),
        "trade_count": int(len(traded)),
        "win_rate": round(float((net > 0).mean()), 6),
        "average_return": round(float(net.mean()), 6),
        "profit_factor": round(profit_factor, 6) if profit_factor != float("inf") else None,
    }


def _class_contribution(frame, mask, fee_pct: float) -> dict:
    import pandas as pd

    traded = frame.loc[mask].copy()
    if traded.empty or "optimal_trade_class_1h" not in traded.columns:
        return {}
    traded["net"] = (traded["target_return_1h"].astype(float) - fee_pct) * traded.get(
        "position_size_multiplier_1h", pd.Series(1.0, index=traded.index)
    ).astype(float)
    out: dict[str, float] = {}
    for cls in ("AGGRESSIVE_WHALE_TRADE", "NORMAL_TRADE", "SMALL_PROBE"):
        subset = traded[traded["optimal_trade_class_1h"] == cls]
        out[f"return_contribution_{cls}"] = round(float(subset["net"].sum()), 6) if not subset.empty else 0.0
    return out


def run_backtest() -> dict:
    import pandas as pd

    fee_pct = get_round_trip_fee_pct()
    df = _load_dataset()
    ready = df[df["target_return_1h"].notna()].copy()
    if ready.empty:
        raise RuntimeError("No rows with target_return_1h — rebuild dataset first.")

    flat_mask = ready["target_return_1h"].notna()
    flat_sizes = pd.Series(1.0, index=ready.index)

    prof_mask = ready["target_profitable_1h"].fillna(False).astype(bool)
    prof_sizes = pd.Series(1.0, index=ready.index)

    whale_mask = ready.get("position_size_multiplier_1h", pd.Series(0.0, index=ready.index)).astype(float) > 0
    whale_sizes = ready.get("position_size_multiplier_1h", pd.Series(0.0, index=ready.index)).astype(float)

    policies = {
        "flat_position": {
            "is_oracle_backtest": False,
            "description": "Baseline: unit size on every row with a 1h outcome label.",
            **_simulate(ready, flat_mask, flat_sizes, fee_pct),
        },
        "oracle_profitability_only": {
            "is_oracle_backtest": True,
            "description": "Oracle upper bound: trade only rows where realized target_profitable_1h is true.",
            **_simulate(ready, prof_mask, prof_sizes, fee_pct),
        },
        "oracle_whale_size_policy": {
            "is_oracle_backtest": True,
            "description": "Oracle upper bound: trade using realized optimal_trade_class_1h size multipliers.",
            **_simulate(ready, whale_mask, whale_sizes, fee_pct),
        },
    }
    policies["oracle_whale_size_policy"].update(_class_contribution(ready, whale_mask, fee_pct))

    aggressive = ready[
        (ready.get("optimal_trade_class_1h") == "AGGRESSIVE_WHALE_TRADE")
        & whale_mask
    ]
    if not aggressive.empty:
        aggressive = aggressive.assign(
            net=lambda f: (f["target_return_1h"].astype(float) - fee_pct)
            * f.get("position_size_multiplier_1h", 1.0).astype(float)
        )
        policies["oracle_whale_size_policy"]["best_aggressive_trades"] = (
            aggressive.nlargest(5, "net")[["symbol", "event_timestamp", "target_return_1h", "whale_wave_score", "net"]]
            .to_dict(orient="records")
        )
        policies["oracle_whale_size_policy"]["worst_aggressive_trades"] = (
            aggressive.nsmallest(5, "net")[["symbol", "event_timestamp", "target_return_1h", "whale_wave_score", "net"]]
            .to_dict(orient="records")
        )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(DATASET_PATH),
        "fee_pct": fee_pct,
        "rows_evaluated": int(len(ready)),
        "is_oracle_backtest": True,
        "note": ORACLE_NOTE,
        "next_required_step": NEXT_STEP,
        "policies": policies,
    }
    return report


def main() -> int:
    report = run_backtest()
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BACKTEST_DIR / "position_sizing_policy_report.json"
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(json.dumps(report, indent=2))
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
