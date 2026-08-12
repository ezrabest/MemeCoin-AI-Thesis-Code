"""Require meaningful new signal after system-close cooldown."""
from __future__ import annotations

from typing import Any

REENTRY_BLOCK_NO_NEW_SIGNAL = "REENTRY_BLOCK_NO_NEW_SIGNAL"

DEFAULT_MIN_PRICE_DELTA_PCT = 0.75
DEFAULT_MIN_VOLUME_DELTA_PCT = 5.0
DEFAULT_MIN_LIQUIDITY_DELTA_PCT = 2.0


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pick(row: dict[str, Any], *keys: str) -> float | None:
    for key in keys:
        val = _f(row.get(key))
        if val is not None:
            return val
    return None


def _pct_delta(new: float | None, old: float | None) -> float | None:
    if new is None or old is None or old == 0:
        return None
    return abs((new - old) / old) * 100.0


def check_system_reentry_signal(
    candidate: dict[str, Any],
    close_snapshot: dict[str, Any] | None,
    *,
    min_price_delta_pct: float = DEFAULT_MIN_PRICE_DELTA_PCT,
    min_volume_delta_pct: float = DEFAULT_MIN_VOLUME_DELTA_PCT,
    min_liquidity_delta_pct: float = DEFAULT_MIN_LIQUIDITY_DELTA_PCT,
) -> dict[str, Any]:
    cand = dict(candidate or {})
    snap = dict(close_snapshot or {})

    price = _pick(
        cand,
        "latest_price",
        "price",
        "matched_price",
        "mark_price",
    )
    volume = _pick(cand, "latest_volume_24h", "volume_24h", "volume24h")
    liquidity = _pick(cand, "latest_liquidity", "liquidity", "liquidity_usd")

    close_price = _pick(snap, "price", "exit_price", "marked_price", "latest_price")
    close_volume = _pick(snap, "volume_24h", "latest_volume_24h")
    close_liquidity = _pick(snap, "liquidity", "latest_liquidity")

    price_delta_pct = _pct_delta(price, close_price)
    volume_delta_pct = _pct_delta(volume, close_volume)
    liquidity_delta_pct = _pct_delta(liquidity, close_liquidity)

    meaningful = False
    signals: list[str] = []
    if price_delta_pct is not None and price_delta_pct >= min_price_delta_pct:
        meaningful = True
        signals.append(f"price_delta_pct={price_delta_pct:.2f}")
    if volume_delta_pct is not None and volume_delta_pct >= min_volume_delta_pct:
        meaningful = True
        signals.append(f"volume_delta_pct={volume_delta_pct:.2f}")
    if liquidity_delta_pct is not None and liquidity_delta_pct >= min_liquidity_delta_pct:
        meaningful = True
        signals.append(f"liquidity_delta_pct={liquidity_delta_pct:.2f}")

    if not snap:
        return {
            "passed": True,
            "rejection_code": None,
            "new_signal_detected": True,
            "note": "No close snapshot supplied; reentry signal check skipped.",
            "price_delta_pct": price_delta_pct,
            "volume_delta_pct": volume_delta_pct,
            "liquidity_delta_pct": liquidity_delta_pct,
            "signals": signals,
        }

    passed = meaningful
    return {
        "passed": passed,
        "rejection_code": None if passed else REENTRY_BLOCK_NO_NEW_SIGNAL,
        "new_signal_detected": meaningful,
        "price_delta_pct": price_delta_pct,
        "volume_delta_pct": volume_delta_pct,
        "liquidity_delta_pct": liquidity_delta_pct,
        "signals": signals,
        "thresholds": {
            "min_price_delta_pct": min_price_delta_pct,
            "min_volume_delta_pct": min_volume_delta_pct,
            "min_liquidity_delta_pct": min_liquidity_delta_pct,
        },
    }