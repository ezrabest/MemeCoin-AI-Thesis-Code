"""Exit path simulation for Phase E3 direct target labels."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import numpy as np
import pandas as pd

EXIT_COMPARE_EPSILON = 1e-9

NS_PER_MINUTE = 60_000_000_000


@dataclass
class GapDiagnostic:
    gap_detected: bool = False
    gap_start_timestamp: str | None = None
    gap_end_timestamp: str | None = None
    gap_minutes: float | None = None
    max_allowed_gap_minutes: float | None = None


@dataclass
class ExitSimulationResult:
    label_valid: bool = False
    sim_exit_status: str = "UNKNOWN_ERROR"
    exit_ratio: float | None = None
    sim_net_return: float | None = None
    target_net_profitable_after_exit: bool | None = None
    label_error_code: str | None = None
    label_error_detail: str | None = None
    entry_snapshot_timestamp: str | None = None
    entry_price_raw: float | None = None
    entry_price: float | None = None
    entry_snapshot_id: int | None = None
    future_window_start_timestamp: str | None = None
    future_window_end_timestamp: str | None = None
    first_future_snapshot_timestamp: str | None = None
    last_future_snapshot_timestamp: str | None = None
    future_snapshot_count: int = 0
    max_future_ratio: float | None = None
    min_future_ratio: float | None = None
    exit_timestamp: str | None = None
    gap: GapDiagnostic = field(default_factory=GapDiagnostic)
    extra: dict[str, Any] = field(default_factory=dict)


def _ts_iso(ts_ns: int) -> str:
    return pd.Timestamp(ts_ns, unit="ns", tz="UTC").isoformat().replace("+00:00", "Z")


def _is_finite_positive(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def _ratio_from_prices(entry_price: float, future_price: float) -> float:
    entry_dec = Decimal(str(entry_price))
    future_dec = Decimal(str(future_price))
    return float(future_dec / entry_dec)


def tp_hit(ratio: float, tp_ratio: float) -> bool:
    return ratio >= tp_ratio - EXIT_COMPARE_EPSILON


def sl_hit(ratio: float, sl_ratio: float) -> bool:
    return ratio <= sl_ratio + EXIT_COMPARE_EPSILON


def effective_max_gap_minutes(time_stop_minutes: int, max_future_gap_minutes: float) -> float:
    return float(min(time_stop_minutes, max_future_gap_minutes))


def _check_gap(
    gap_start_ns: int,
    gap_end_ns: int,
    max_allowed_gap_minutes: float,
) -> GapDiagnostic | None:
    gap_minutes = (gap_end_ns - gap_start_ns) / NS_PER_MINUTE
    if gap_minutes <= max_allowed_gap_minutes + EXIT_COMPARE_EPSILON:
        return None
    return GapDiagnostic(
        gap_detected=True,
        gap_start_timestamp=_ts_iso(gap_start_ns),
        gap_end_timestamp=_ts_iso(gap_end_ns),
        gap_minutes=float(gap_minutes),
        max_allowed_gap_minutes=float(max_allowed_gap_minutes),
    )


def simulate_exit_path(
    *,
    pair: str,
    event_ns: int,
    time_stop_minutes: int,
    tp_ratio: float,
    sl_ratio: float,
    round_trip_fee_pct: float,
    max_future_gap_minutes: float,
    ts_ns: np.ndarray,
    prices: np.ndarray,
    snapshot_ids: np.ndarray | None = None,
) -> ExitSimulationResult:
    """Simulate TP/SL/TIME exit on a sorted snapshot series for one pair."""
    result = ExitSimulationResult()
    max_gap = effective_max_gap_minutes(time_stop_minutes, max_future_gap_minutes)

    if not pair or not str(pair).strip():
        result.sim_exit_status = "BAD_PAIR_ADDRESS"
        result.label_error_code = "BAD_PAIR_ADDRESS"
        result.label_error_detail = "pair_address is empty"
        return result

    if not np.isfinite(event_ns):
        result.sim_exit_status = "BAD_TIMESTAMP"
        result.label_error_code = "BAD_TIMESTAMP"
        result.label_error_detail = "event_timestamp is not finite"
        return result

    if len(ts_ns) == 0:
        result.sim_exit_status = "NO_ENTRY_SNAPSHOT"
        result.label_error_code = "NO_ENTRY_SNAPSHOT"
        result.label_error_detail = "no snapshots for pair"
        return result

    entry_idx = int(np.searchsorted(ts_ns, event_ns, side="right") - 1)
    if entry_idx < 0:
        result.sim_exit_status = "NO_ENTRY_SNAPSHOT"
        result.label_error_code = "NO_ENTRY_SNAPSHOT"
        result.label_error_detail = "no snapshot at or before event_timestamp"
        return result

    entry_ts = int(ts_ns[entry_idx])
    entry_price_raw = float(prices[entry_idx])
    result.entry_snapshot_timestamp = _ts_iso(entry_ts)
    result.entry_price_raw = entry_price_raw
    if snapshot_ids is not None and entry_idx < len(snapshot_ids):
        sid = snapshot_ids[entry_idx]
        if sid is not None and not (isinstance(sid, float) and math.isnan(sid)):
            result.entry_snapshot_id = int(sid)

    if not _is_finite_positive(entry_price_raw):
        result.sim_exit_status = "BAD_ENTRY_PRICE"
        result.label_error_code = "BAD_ENTRY_PRICE"
        result.label_error_detail = f"entry price invalid: {entry_price_raw!r}"
        return result

    entry_price = entry_price_raw
    result.entry_price = entry_price

    end_ns = entry_ts + int(time_stop_minutes * NS_PER_MINUTE)
    result.future_window_start_timestamp = _ts_iso(entry_ts)
    result.future_window_end_timestamp = _ts_iso(end_ns)

    start = entry_idx + 1
    end = int(np.searchsorted(ts_ns, end_ns, side="right"))

    if end <= start:
        result.sim_exit_status = "NO_FUTURE_WINDOW"
        result.label_error_code = "NO_FUTURE_WINDOW"
        result.label_error_detail = "no future snapshots strictly after entry within time-stop"
        return result

    fut_ts = ts_ns[start:end]
    fut_price = prices[start:end]

    ok = np.isfinite(fut_price) & (fut_price > 0)
    if not ok.any():
        result.sim_exit_status = "NO_VALID_FUTURE_PRICE"
        result.label_error_code = "NO_VALID_FUTURE_PRICE"
        result.label_error_detail = "no valid future prices in bounded window"
        return result

    fut_ts = fut_ts[ok]
    fut_price = fut_price[ok]
    result.future_snapshot_count = int(len(fut_ts))
    result.first_future_snapshot_timestamp = _ts_iso(int(fut_ts[0]))
    result.last_future_snapshot_timestamp = _ts_iso(int(fut_ts[-1]))

    gap = _check_gap(entry_ts, int(fut_ts[0]), max_gap)
    if gap is not None:
        result.gap = gap
        result.sim_exit_status = "GAP_IN_FUTURE_DATA"
        result.label_error_code = "GAP_IN_FUTURE_DATA"
        result.label_error_detail = "entry to first future snapshot gap exceeds threshold"
        return result

    ratios = np.array([_ratio_from_prices(entry_price, float(p)) for p in fut_price], dtype=float)
    result.max_future_ratio = float(np.max(ratios))
    result.min_future_ratio = float(np.min(ratios))

    exit_status = "TIME"
    exit_ratio = float(ratios[-1])
    exit_ts = int(fut_ts[-1])

    for idx in range(len(fut_ts)):
        if idx > 0:
            gap = _check_gap(int(fut_ts[idx - 1]), int(fut_ts[idx]), max_gap)
            if gap is not None:
                result.gap = gap
                result.sim_exit_status = "GAP_IN_FUTURE_DATA"
                result.label_error_code = "GAP_IN_FUTURE_DATA"
                result.label_error_detail = "consecutive future snapshot gap exceeds threshold"
                return result

        ratio = float(ratios[idx])
        if tp_hit(ratio, tp_ratio):
            exit_status = "TP"
            exit_ratio = float(tp_ratio)
            exit_ts = int(fut_ts[idx])
            break
        if sl_hit(ratio, sl_ratio):
            exit_status = "SL"
            exit_ratio = float(sl_ratio)
            exit_ts = int(fut_ts[idx])
            break

    if exit_status == "TIME":
        gap = _check_gap(int(fut_ts[-1]), end_ns, max_gap)
        if gap is not None:
            result.gap = gap
            result.sim_exit_status = "GAP_IN_FUTURE_DATA"
            result.label_error_code = "GAP_IN_FUTURE_DATA"
            result.label_error_detail = "last valid snapshot too far before time-stop boundary"
            return result

    if not _is_finite_positive(exit_ratio):
        result.sim_exit_status = "BAD_EXIT_RATIO"
        result.label_error_code = "BAD_EXIT_RATIO"
        result.label_error_detail = f"exit_ratio invalid: {exit_ratio!r}"
        return result

    net_return = float((exit_ratio - 1.0) - round_trip_fee_pct)
    if not math.isfinite(net_return):
        result.sim_exit_status = "UNKNOWN_ERROR"
        result.label_error_code = "UNKNOWN_ERROR"
        result.label_error_detail = "net_return is not finite"
        return result

    result.label_valid = True
    result.sim_exit_status = exit_status
    result.exit_ratio = exit_ratio
    result.sim_net_return = net_return
    result.target_net_profitable_after_exit = net_return > 0
    result.exit_timestamp = _ts_iso(exit_ts)
    return result
