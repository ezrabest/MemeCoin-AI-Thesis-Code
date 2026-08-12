"""Minimal frontend validation — backend remains authority for business rules."""
from __future__ import annotations

from typing import Any

from app.web.percent_conversion import DISPLAY_AS_PERCENT_POINTS, is_decimal_fraction_pct_key


def validate_field(key: str, value: Any, *, widget_kind: str = "number") -> str | None:
    if widget_kind == "bool":
        return None if isinstance(value, bool) else "Expected boolean"
    if widget_kind == "text":
        return "Required" if value is None or str(value).strip() == "" else None
    if widget_kind == "int":
        if value is None or value == "":
            return "Required integer"
        try:
            ival = int(round(float(value)))
            if ival < 0:
                return "Must be non-negative"
        except (TypeError, ValueError):
            return "Invalid integer"
        return None
    if value is None or value == "":
        return "Required numeric value"
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return "Invalid number"
    if key == "min_liquidity_usd" and fval < 0:
        return "Must be non-negative"
    if is_decimal_fraction_pct_key(key) and (fval < 0 or fval > 100):
        return "Percent should be between 0 and 100"
    if key in DISPLAY_AS_PERCENT_POINTS and (fval < 0 or fval > 100):
        return "Should be between 0 and 100"
    return None
