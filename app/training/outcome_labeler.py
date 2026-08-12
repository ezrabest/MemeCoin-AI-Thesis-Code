"""
Future outcome labeling from market snapshot time series.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

HORIZON_MINUTES = {"15m": 15, "1h": 60, "4h": 240}
EVENT_SNAPSHOT_MAX_MINUTES = 30
HORIZON_SLACK_MINUTES = 10


def get_round_trip_fee_pct() -> float:
    try:
        return float(os.getenv("TRAINING_ROUND_TRIP_FEE_PCT", "0.0308"))
    except ValueError:
        return 0.0308


def parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def snapshot_times(snapshots: list[dict[str, Any]]) -> list[datetime | None]:
    return [parse_timestamp(s.get("timestamp")) for s in snapshots]


def _snapshot_price(snapshot: dict[str, Any] | None) -> float | None:
    if not snapshot:
        return None
    price = snapshot.get("price")
    if price is None:
        return None
    try:
        p = float(price)
    except (TypeError, ValueError):
        return None
    return p if p > 0 else None


def closest_snapshot(
    snapshots: list[dict[str, Any]],
    event_ts: datetime,
    *,
    times: list[datetime | None] | None = None,
    max_delta_minutes: float = EVENT_SNAPSHOT_MAX_MINUTES,
) -> dict[str, Any] | None:
    """Nearest snapshot to event time within tolerance."""
    if times is None:
        times = snapshot_times(snapshots)
    best: dict[str, Any] | None = None
    best_delta: float | None = None
    for snap, ts in zip(snapshots, times):
        if ts is None:
            continue
        delta = abs((ts - event_ts).total_seconds()) / 60.0
        if delta <= max_delta_minutes and (best_delta is None or delta < best_delta):
            best = snap
            best_delta = delta
    return best


def price_near_horizon(
    snapshots: list[dict[str, Any]],
    event_ts: datetime,
    horizon_minutes: int,
    *,
    times: list[datetime | None] | None = None,
    slack_minutes: float = HORIZON_SLACK_MINUTES,
) -> tuple[float | None, datetime | None]:
    """Price at event + horizon using the nearest later snapshot."""
    if times is None:
        times = snapshot_times(snapshots)
    target = event_ts + timedelta(minutes=horizon_minutes)
    best_snap: dict[str, Any] | None = None
    best_ts: datetime | None = None
    best_delta: float | None = None
    for snap, ts in zip(snapshots, times):
        if ts is None or ts <= event_ts:
            continue
        delta = abs((ts - target).total_seconds()) / 60.0
        if delta <= slack_minutes and (best_delta is None or delta < best_delta):
            best_snap = snap
            best_ts = ts
            best_delta = delta
    if not best_snap:
        return None, None
    return _snapshot_price(best_snap), best_ts


def window_return_extremes(
    snapshots: list[dict[str, Any]],
    event_ts: datetime,
    horizon_minutes: int,
    base_price: float,
    *,
    times: list[datetime | None] | None = None,
) -> tuple[float | None, float | None]:
    """Max upside and max drawdown within (event, event + horizon]."""
    if base_price <= 0:
        return None, None
    if times is None:
        times = snapshot_times(snapshots)
    end = event_ts + timedelta(minutes=horizon_minutes)
    returns: list[float] = []
    for snap, ts in zip(snapshots, times):
        if ts is None or ts <= event_ts or ts > end:
            continue
        price = _snapshot_price(snap)
        if price is None:
            continue
        returns.append((price - base_price) / base_price)
    if not returns:
        return None, None
    return max(returns), min(returns)


def compute_future_return(base_price: float | None, future_price: float | None) -> float | None:
    if base_price is None or future_price is None or base_price <= 0:
        return None
    return (future_price - base_price) / base_price


def label_outcomes(
    snapshots: list[dict[str, Any]],
    event_ts: datetime,
    *,
    fee_pct: float | None = None,
    times: list[datetime | None] | None = None,
) -> dict[str, Any]:
    """
    Compute future prices, returns, labels, and pending flag for one event.
    snapshots must be sorted ascending by timestamp for the same coin_id.
    """
    fee = get_round_trip_fee_pct() if fee_pct is None else fee_pct
    if times is None:
        times = snapshot_times(snapshots)
    event_snap = closest_snapshot(snapshots, event_ts, times=times)
    base_price = _snapshot_price(event_snap)

    result: dict[str, Any] = {
        "event_snapshot_timestamp": event_snap.get("timestamp") if event_snap else None,
        "price_usd": base_price,
        "pending_outcome": False,
    }

    if event_snap:
        for key in (
            "liquidity", "volume_24h", "txns_buys", "txns_sells",
            "buy_ratio", "whale_score", "price_change_h1", "price_change_h24",
            "chain", "pair_address",
        ):
            if key in event_snap and event_snap.get(key) is not None:
                mapped = key
                if key == "liquidity":
                    mapped = "liquidity_usd"
                if key == "price_change_h1":
                    mapped = "price_change_1h"
                if key == "price_change_h24":
                    mapped = "price_change_24h"
                result[mapped] = event_snap.get(key)

    labeled_timestamps: list[datetime] = []
    missing_horizons = 0

    for label, minutes in HORIZON_MINUTES.items():
        future_price, labeled_ts = price_near_horizon(
            snapshots, event_ts, minutes, times=times,
        )
        future_return = compute_future_return(base_price, future_price)
        result[f"future_price_{label}"] = future_price
        result[f"future_return_{label}"] = future_return
        if future_return is None:
            missing_horizons += 1
        else:
            result[f"label_up_{label}"] = future_return > 0
            result[f"label_profitable_after_fees_{label}"] = future_return > fee
            if labeled_ts:
                labeled_timestamps.append(labeled_ts)

    for label, minutes in (("1h", 60), ("4h", 240)):
        upside, drawdown = window_return_extremes(
            snapshots, event_ts, minutes, base_price or 0.0, times=times,
        )
        result[f"max_upside_{label}"] = upside
        result[f"max_drawdown_{label}"] = drawdown

    result["pending_outcome"] = missing_horizons > 0 or base_price is None
    result["latest_labeled_timestamp"] = (
        max(labeled_timestamps).isoformat() if labeled_timestamps else None
    )
    return result
