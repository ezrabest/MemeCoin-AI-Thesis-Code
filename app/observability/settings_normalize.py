"""Normalize raw settings to numeric canonical values with decimal-fraction *_pct units."""
from __future__ import annotations

from typing import Any

# Internal economic unit: decimal fraction (0.05 = 5%, 0.015 = 1.5%).
DECIMAL_FRACTION_PCT_KEYS = frozenset({
    "stop_loss_pct",
    "take_profit_pct",
    "max_position_size_pct",
    "max_slippage_pct",
    "baseline_slippage_pct",
    "round_trip_fee_pct",
    "required_margin_after_costs_pct",
    "max_price_drift_from_model_pct",
    "gas_or_priority_cost_pct",
    "max_daily_loss_pct",
    "max_drawdown_pct",
    "trailing_stop_pct",
})

# Probability / score fields — never divide by 100.
UNIT_INTERVAL_KEYS = frozenset({
    "rf_probability_threshold",
    "tab_confidence_percentile_threshold",
    "min_whale_score",
    "min_signal_score",
    "min_buy_ratio",
    "llm_score_threshold",
    "probability_profitable_threshold",
})

FLOAT_KEYS = frozenset({
    "starting_capital",
    "min_liquidity_usd",
    "tab_position_size_multiplier",
    "slippage_liquidity_impact_multiplier",
    "slippage_volume_liquidity_multiplier",
    "effective_liquidity_conservative_factor",
    "max_model_artifact_age_hours",
    "required_margin_after_costs",
})

INT_KEYS = frozenset({
    "max_open_positions",
    "cooldown_minutes",
    "time_stop_minutes",
    "max_llm_calls_per_hour",
    "max_llm_calls_per_scan",
    "llm_cache_window_minutes",
    "max_risk_score",
    "min_market_read_interval_seconds",
    "max_market_snapshot_age_seconds",
    "max_model_prediction_age_seconds",
    "max_model_snapshot_age_seconds",
    "paper_fee_bps",
})


def _coerce_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    f = _coerce_float(value)
    if f is None:
        return None
    return int(round(f))


def normalize_decimal_fraction_pct(value: Any) -> float | None:
    """
    Convert user-facing percent values to internal decimal fractions.
    Examples: 7 -> 0.07, 1.5 -> 0.015, 30 -> 0.30, 0.01 -> 0.01 (unchanged).
    """
    raw = _coerce_float(value)
    if raw is None:
        return None
    if abs(raw) > 1.0:
        return round(raw / 100.0, 8)
    return round(raw, 8)


def normalize_required_margin_after_costs_pct(value: Any) -> float | None:
    """
    required_margin_after_costs_pct uses decimal fractions (0.005 = 0.5%).
    Legacy percent-point inputs like 0.5 (meaning 0.5%) are converted to 0.005.
    Values already in decimal form (0.02 = 2%) are preserved.
    """
    raw = _coerce_float(value)
    if raw is None:
        return None
    if abs(raw) > 1.0:
        return round(raw / 100.0, 8)
    if abs(raw) <= 0.01:
        return round(raw, 8)
    # Legacy percent-point style: 0.5, 0.1 (single decimal place, >= 0.1)
    if abs(raw) >= 0.1 and abs(raw * 10 - round(raw * 10)) < 1e-6:
        return round(raw / 100.0, 8)
    return round(raw, 8)


def normalize_paper_fee_bps(value: Any) -> float | None:
    """
    paper_fee_bps is stored as basis points (150 = 1.5%).
    UI tradingFee alias may send percent like 1.5 -> 150 bps.
    """
    raw = _coerce_float(value)
    if raw is None:
        return None
    if raw <= 10.0:
        return round(raw * 100.0, 4)
    return round(raw, 4)


def normalize_canonical_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """Return settings with numeric types and consistent decimal-fraction *_pct values."""
    out = dict(settings)

    for key in DECIMAL_FRACTION_PCT_KEYS:
        if key in out:
            norm = normalize_decimal_fraction_pct(out[key])
            if norm is not None:
                out[key] = norm

    if "required_margin_after_costs" in out and "required_margin_after_costs_pct" not in out:
        legacy_raw = _coerce_float(out["required_margin_after_costs"])
        if legacy_raw is not None:
            out["required_margin_after_costs_pct"] = round(legacy_raw / 100.0, 8)

    if "required_margin_after_costs_pct" in out:
        norm = normalize_required_margin_after_costs_pct(out["required_margin_after_costs_pct"])
        if norm is not None:
            out["required_margin_after_costs_pct"] = norm

    for key in UNIT_INTERVAL_KEYS:
        if key in out:
            val = _coerce_float(out[key])
            if val is not None:
                out[key] = round(val, 8)

    for key in FLOAT_KEYS:
        if key in out:
            val = _coerce_float(out[key])
            if val is not None:
                out[key] = val

    if "paper_fee_bps" in out:
        norm = normalize_paper_fee_bps(out["paper_fee_bps"])
        if norm is not None:
            out["paper_fee_bps"] = norm

    for key in INT_KEYS:
        if key in out and key != "paper_fee_bps":
            val = _coerce_int(out[key])
            if val is not None:
                out[key] = val

    pct = out.get("max_position_size_pct", 0.05)
    if isinstance(pct, (int, float)):
        pct_f = float(pct)
        if pct_f > 1.0:
            pct_f = pct_f / 100.0
        out["max_position_size_pct"] = round(max(0.0001, min(1.0, pct_f)), 8)

    return out
