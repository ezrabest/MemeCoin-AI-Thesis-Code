"""Exact case-preserved identity lookup keys for AE20.

Hard rule: never lowercase, uppercase, casefold, or URL-normalize identity fields.
Only stringify + strip leading/trailing whitespace; reject empty/invalid literals.
"""

from __future__ import annotations

from typing import Any

# Explicit invalid literals — membership only. Do NOT use .lower()/.casefold().
INVALID_IDENTITY_LITERALS = frozenset(
    {
        "",
        "nan",
        "NaN",
        "NAN",
        "none",
        "None",
        "NONE",
        "null",
        "Null",
        "NULL",
        "<NA>",
        "NA",
        "N/A",
    }
)


def make_exact_identity_lookup_key(value: Any) -> str | None:
    """Build an exact case-preserved identity lookup key.

    - None / NA / NaN → None
    - stringify, strip leading/trailing whitespace only
    - empty or explicit invalid literals → None
    - otherwise return stripped value exactly (casing preserved)
    """
    if value is None:
        return None
    # pandas NA / NaN without importing pandas as a hard dependency
    try:
        if value is getattr(type(value), "NA", object()):
            return None
    except Exception:
        pass
    try:
        # float NaN
        if isinstance(value, float) and value != value:  # noqa: PLR0124
            return None
    except Exception:
        pass
    try:
        import math

        if isinstance(value, float) and math.isnan(value):
            return None
    except Exception:
        pass
    # pandas.NA equality quirks
    try:
        import pandas as pd  # type: ignore

        if value is pd.NA:
            return None
        if isinstance(value, float) and pd.isna(value):
            return None
        # pd.NA via scalar check for non-float
        if not isinstance(value, (str, bytes)) and pd.isna(value):
            return None
    except Exception:
        pass

    text = str(value)
    stripped = text.strip()  # leading/trailing whitespace only
    if stripped in INVALID_IDENTITY_LITERALS:
        return None
    return stripped
