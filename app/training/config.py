"""Training dataset thresholds from environment variables."""
from __future__ import annotations

import os


def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError:
        return default


def get_round_trip_fee_pct() -> float:
    return _float_env("TRAINING_ROUND_TRIP_FEE_PCT", 0.0308)


def get_big_pump_threshold(horizon: str) -> float:
    defaults = {"15m": 0.25, "1h": 0.50, "4h": 1.00}
    return _float_env(f"BIG_PUMP_{horizon.upper()}_THRESHOLD", defaults[horizon])


def get_big_dump_threshold(horizon: str) -> float:
    defaults = {"15m": -0.20, "1h": -0.35, "4h": -0.60}
    return _float_env(f"BIG_DUMP_{horizon.upper()}_THRESHOLD", defaults[horizon])


def get_pump_then_dump_pump_threshold() -> float:
    return _float_env("PUMP_THEN_DUMP_PUMP_THRESHOLD", 0.50)


def get_pump_then_dump_drop_threshold() -> float:
    return _float_env("PUMP_THEN_DUMP_DROP_FROM_PEAK_THRESHOLD", -0.30)


def get_whale_wave_aggressive_threshold() -> float:
    return _float_env("WHALE_WAVE_AGGRESSIVE_THRESHOLD", 0.60)


def get_whale_wave_normal_threshold() -> float:
    return _float_env("WHALE_WAVE_NORMAL_THRESHOLD", 0.40)


TRADE_CLASS_MULTIPLIERS = {
    "NO_TRADE": 0.0,
    "AVOID_DUMP": 0.0,
    "SMALL_PROBE": 0.25,
    "NORMAL_TRADE": 1.0,
    "AGGRESSIVE_WHALE_TRADE": 2.0,
}
