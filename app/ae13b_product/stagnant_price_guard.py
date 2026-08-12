"""Block stagnant candidates without recent momentum (AE13I).

Production `coins` rows frequently do not carry 1h/4h activity deltas at
all (only `market_snapshots` does). Missing delta fields are evidence of
*unknown* momentum, not proof that the price has not moved -- the AE13I
spec requires evidence of no change before blocking, so this guard never
auto-blocks purely because deltas are absent. It only blocks when it has
a concrete, present delta value that is below the activity threshold.
"""
from __future__ import annotations

from typing import Any

REJECTION_CODE = "PRICE_STAGNANT_NO_RECENT_MOMENTUM"

DEFAULT_MIN_ACTIVITY_DELTA_1H_PCT = 0.5
DEFAULT_MIN_ACTIVITY_DELTA_4H_PCT = 1.0
DEFAULT_STAGNANT_WINDOW_MINUTES = 240

#: `momentum_evidence` value when no delta field exists anywhere on the row.
MOMENTUM_EVIDENCE_UNKNOWN = "unknown_insufficient_delta_fields"


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


#: window -> ordered list of candidate field names checked for that window.
_WINDOW_KEYS: dict[str, tuple[str, ...]] = {
    "5m": (
        "activity_delta_5m_pct", "latest_delta_5m", "delta_5m_pct",
        "delta_5m", "price_change_5m", "priceChange5m",
    ),
    "15m": (
        "activity_delta_15m_pct", "latest_delta_15m", "delta_15m_pct",
        "delta_15m", "price_change_15m", "priceChange15m",
    ),
    "1h": (
        "activity_delta_1h_pct", "latest_delta_1h", "delta_1h_pct",
        "price_change_1h", "priceChange1h",
    ),
    "4h": (
        "activity_delta_4h_pct", "latest_delta_4h", "delta_4h_pct",
        "price_change_4h", "priceChange4h",
    ),
    "6h": (
        "activity_delta_6h_pct", "latest_delta_6h", "delta_6h_pct",
        "delta_6h", "price_change_6h", "priceChange6h",
    ),
    "24h": (
        "activity_delta_24h_pct", "latest_delta_24h", "delta_24h_pct",
        "delta_24h", "change_24h", "price_change_24h", "priceChange24h",
    ),
}

#: Raw field names named explicitly in the AE13I spec, kept here (in
#: addition to `_WINDOW_KEYS`) so field-presence detection is defensive
#: even if a caller uses a spelling not already covered above.
_EXTRA_RAW_FIELDS = (
    "price_change_5m", "price_change_15m", "price_change_6h",
    "delta_5m", "delta_15m", "delta_6h", "change_24h",
)


def _window_delta(row: dict[str, Any], window: str) -> float | None:
    for key in _WINDOW_KEYS.get(window, ()):
        val = _f(row.get(key))
        if val is not None:
            return val
    return None


def _any_momentum_field_present(row: dict[str, Any]) -> bool:
    for keys in _WINDOW_KEYS.values():
        for key in keys:
            if row.get(key) not in (None, ""):
                return True
    for key in _EXTRA_RAW_FIELDS:
        if row.get(key) not in (None, ""):
            return True
    return False


def _has_fresh_catalyst(row: dict[str, Any]) -> bool:
    """Fresh-catalyst / momentum-proxy flags that justify bypassing the guard.

    Beyond the explicit `fresh_catalyst` flag, a whale-signal refresh,
    volume spike, liquidity increase, improving buy ratio, or fresh RSS
    signal are all treated as evidence the coin is not actually stagnant
    even if 1h/4h price deltas look flat or are missing.
    """
    if row.get("fresh_catalyst") or row.get("has_fresh_catalyst") or row.get("catalyst_fresh"):
        return True
    if row.get("fresh_whale_signal") or row.get("fresh_rss_signal"):
        return True
    if row.get("volume_spike"):
        return True
    if row.get("buy_ratio_improved"):
        return True
    liq_increase = _f(row.get("liquidity_increase_pct"))
    if liq_increase is not None and liq_increase > 0:
        return True
    return False


def evaluate_stagnant_price(
    candidate: dict[str, Any],
    *,
    min_activity_delta_1h_pct: float = DEFAULT_MIN_ACTIVITY_DELTA_1H_PCT,
    min_activity_delta_4h_pct: float = DEFAULT_MIN_ACTIVITY_DELTA_4H_PCT,
    stagnant_window_minutes: int = DEFAULT_STAGNANT_WINDOW_MINUTES,
    allow_stagnant_buy_only_with_fresh_catalyst: bool = True,
) -> dict[str, Any]:
    row = dict(candidate or {})
    fresh_catalyst = _has_fresh_catalyst(row)

    delta_1h = _window_delta(row, "1h")
    delta_4h = _window_delta(row, "4h")
    delta_5m = _window_delta(row, "5m")
    delta_15m = _window_delta(row, "15m")
    delta_6h = _window_delta(row, "6h")
    delta_24h = _window_delta(row, "24h")

    base_result: dict[str, Any] = {
        "activity_delta_1h_pct": delta_1h,
        "activity_delta_4h_pct": delta_4h,
        "activity_delta_5m_pct": delta_5m,
        "activity_delta_15m_pct": delta_15m,
        "activity_delta_6h_pct": delta_6h,
        "activity_delta_24h_pct": delta_24h,
        "stagnant_window_minutes": stagnant_window_minutes,
        "allow_stagnant_buy_only_with_fresh_catalyst": allow_stagnant_buy_only_with_fresh_catalyst,
    }

    if fresh_catalyst and allow_stagnant_buy_only_with_fresh_catalyst:
        return {
            **base_result,
            "passed": True,
            "rejection_code": None,
            "rejection_reason": None,
            "blocking_guards": [],
            "fresh_catalyst_bypass": True,
            "momentum_evidence": "fresh_catalyst_bypass",
        }

    if not _any_momentum_field_present(row):
        # No evidence either way. AE13I requires proof of *no* change before
        # blocking as stagnant -- unknown is not the same as proven stagnant.
        return {
            **base_result,
            "passed": True,
            "rejection_code": None,
            "rejection_reason": None,
            "blocking_guards": [],
            "fresh_catalyst_bypass": False,
            "momentum_evidence": MOMENTUM_EVIDENCE_UNKNOWN,
            "note": (
                "No 1h/4h/5m/15m/6h/24h activity delta fields present on this "
                "row; unknown momentum is not treated as proven stagnation."
            ),
        }

    # 4h evidence, falling back to 6h/24h proxies only when 4h itself is
    # absent (an explicit 4h delta always takes precedence when present).
    effective_4h = delta_4h if delta_4h is not None else (
        delta_6h if delta_6h is not None else delta_24h
    )
    low_4h = effective_4h is not None and abs(effective_4h) < min_activity_delta_4h_pct

    # 1h evidence, falling back to 15m/5m proxies only when 1h itself is
    # absent.
    effective_1h = delta_1h if delta_1h is not None else (
        delta_15m if delta_15m is not None else delta_5m
    )
    low_1h = effective_1h is not None and abs(effective_1h) < min_activity_delta_1h_pct

    blocking_guards: list[str] = []
    reasons: list[str] = []
    stagnant = False

    if effective_4h is not None and low_4h:
        stagnant = True
        blocking_guards.append("stagnant_price_guard_4h")
        reasons.append(
            f"4h activity delta {abs(effective_4h):.4f}% below {min_activity_delta_4h_pct}%"
        )

    if effective_1h is not None and low_1h and (effective_4h is None or low_4h):
        stagnant = True
        blocking_guards.append("stagnant_price_guard_1h")
        reasons.append(
            f"1h activity delta {abs(effective_1h):.4f}% below {min_activity_delta_1h_pct}%"
        )

    if not stagnant:
        return {
            **base_result,
            "passed": True,
            "rejection_code": None,
            "rejection_reason": None,
            "blocking_guards": [],
            "fresh_catalyst_bypass": False,
            "momentum_evidence": "recent_momentum_present",
        }

    blocking_guards = ["stagnant_price_guard", "no_recent_momentum", *blocking_guards]
    return {
        **base_result,
        "passed": False,
        "rejection_code": REJECTION_CODE,
        "rejection_reason": "; ".join(reasons) if reasons else "No recent price momentum detected.",
        "blocking_guards": blocking_guards,
        "fresh_catalyst_bypass": False,
        "momentum_evidence": "stagnant_confirmed",
        "thresholds": {
            "min_activity_delta_1h_pct": min_activity_delta_1h_pct,
            "min_activity_delta_4h_pct": min_activity_delta_4h_pct,
        },
    }
