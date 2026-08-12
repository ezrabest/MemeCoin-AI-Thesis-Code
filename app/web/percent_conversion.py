"""Display/save conversion between UI human values and backend canonical units."""
from __future__ import annotations

from typing import Any

from app.observability.settings_normalize import (
    DECIMAL_FRACTION_PCT_KEYS,
    UNIT_INTERVAL_KEYS,
    normalize_decimal_fraction_pct,
    normalize_required_margin_after_costs_pct,
)

DISPLAY_AS_PERCENT_POINTS = frozenset({
    "rf_probability_threshold",
    "tab_confidence_percentile_threshold",
})


def is_decimal_fraction_pct_key(key: str) -> bool:
    return key in DECIMAL_FRACTION_PCT_KEYS


def internal_to_display_number(key: str, internal: Any) -> float | None:
    if internal is None:
        return None
    try:
        val = float(internal)
    except (TypeError, ValueError):
        return None
    if key in DECIMAL_FRACTION_PCT_KEYS:
        return round(val * 100.0, 6)
    if key in DISPLAY_AS_PERCENT_POINTS:
        return round(val * 100.0, 4)
    return val


def display_to_internal_number(key: str, display: Any) -> float | None:
    if display is None or display == "":
        return None
    try:
        val = float(display)
    except (TypeError, ValueError):
        return None
    if key in DECIMAL_FRACTION_PCT_KEYS:
        return normalize_decimal_fraction_pct(val)
    if key in DISPLAY_AS_PERCENT_POINTS:
        if val > 1.0:
            return round(val / 100.0, 8)
        return round(val, 8)
    if key == "required_margin_after_costs_pct":
        return normalize_required_margin_after_costs_pct(val)
    return val


def format_display_value(key: str, internal: Any) -> str:
    if internal is None:
        return "—"
    if isinstance(internal, bool):
        return "ON" if internal else "OFF"
    if key == "min_liquidity_usd":
        return f"${float(internal):,.2f}"
    if key in DECIMAL_FRACTION_PCT_KEYS:
        return f"{float(internal) * 100:.4g}%"
    if key in DISPLAY_AS_PERCENT_POINTS:
        return f"{float(internal) * 100:.4g}%"
    if key in UNIT_INTERVAL_KEYS and key not in DISPLAY_AS_PERCENT_POINTS:
        return f"{float(internal):.4g}"
    if isinstance(internal, (list, dict)):
        return str(internal)
    return str(internal)


def format_unit_label(key: str) -> str:
    if key in DECIMAL_FRACTION_PCT_KEYS:
        return "decimal fraction percent (UI: %)"
    if key in DISPLAY_AS_PERCENT_POINTS:
        return "probability / percentile (UI: %)"
    if key == "min_liquidity_usd":
        return "USD"
    if key.endswith("_enabled"):
        return "boolean"
    return ""
