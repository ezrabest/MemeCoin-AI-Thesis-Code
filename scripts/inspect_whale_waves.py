#!/usr/bin/env python3
"""Print top whale-wave events from the latest training datasets."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRAINING_DIR = ROOT / "data" / "training"
DISPLAY_COLS = [
    "event_timestamp",
    "symbol",
    "pair_address",
    "chain",
    "price_usd",
    "volume_24h",
    "liquidity_usd",
    "whale_wave_score",
    "whale_wave_direction",
    "future_return_1h",
    "future_return_4h",
    "big_pump_1h",
    "big_pump_4h",
    "pump_then_dump_4h",
    "optimal_trade_class_1h",
    "optimal_trade_class_4h",
]


def _load(path_stem: str):
    import pandas as pd

    parquet = TRAINING_DIR / f"{path_stem}.parquet"
    if parquet.is_file():
        return pd.read_parquet(parquet)
    csv = TRAINING_DIR / f"{path_stem}.csv"
    if csv.is_file():
        return pd.read_csv(csv)
    return None


def _print_top(title: str, frame, sort_col: str, n: int = 20) -> None:
    import pandas as pd

    if frame is None or frame.empty or sort_col not in frame.columns:
        print(f"\n=== {title} ===\n(no data)")
        return
    cols = [c for c in DISPLAY_COLS if c in frame.columns]
    view = frame.sort_values(sort_col, ascending=False).head(n)
    print(f"\n=== {title} (top {n}) ===")
    print(view[cols].to_string(index=False))


def main() -> int:
    signals = _load("signal_outcomes")
    llm = _load("llm_decision_outcomes")
    import pandas as pd

    combined = pd.concat([f for f in (signals, llm) if f is not None], ignore_index=True)
    if combined.empty:
        print("No training datasets found. Run: python scripts/build_training_dataset.py")
        return 1

    _print_top("whale_wave_score", combined, "whale_wave_score")
    _print_top("big_pump_1h", combined, "future_return_1h")
    _print_top("big_pump_4h", combined, "future_return_4h")
    if "pump_then_dump_4h" in combined.columns:
        ptd = combined[combined["pump_then_dump_4h"].fillna(False).astype(bool)]
        _print_top("pump_then_dump_4h", ptd, "whale_wave_score")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
