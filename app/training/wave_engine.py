"""
Future wave labels, whale_wave_score, and position-sizing labels (vectorized).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    TRADE_CLASS_MULTIPLIERS,
    get_big_dump_threshold,
    get_big_pump_threshold,
    get_pump_then_dump_drop_threshold,
    get_pump_then_dump_pump_threshold,
    get_round_trip_fee_pct,
    get_whale_wave_aggressive_threshold,
    get_whale_wave_normal_threshold,
)
from .snapshot_features import HISTORICAL_FEATURE_COLUMNS, sanitize

log = logging.getLogger("training.wave_engine")

HORIZON_MINUTES = {"15m": 15, "1h": 60, "4h": 240}


def detect_pump_then_dump(
    prices: np.ndarray,
    base_price: float,
    *,
    pump_threshold: float,
    drop_threshold: float,
) -> bool | None:
    """
    True when price pumps >= pump_threshold from base, then falls >= |drop_threshold|
    from the peak price before window end. False when window exists but pattern absent.
    None when insufficient data.
    """
    if base_price <= 0 or len(prices) < 2:
        return None

    rets = (prices - base_price) / base_price
    peak_idx = int(np.argmax(rets))
    peak_ret = float(rets[peak_idx])
    if peak_ret < pump_threshold:
        return False

    if peak_idx >= len(prices) - 1:
        return False

    peak_price = float(prices[peak_idx])
    if peak_price <= 0:
        return False

    after_prices = prices[peak_idx + 1 :]
    if after_prices.size == 0:
        return False

    trough_price = float(np.min(after_prices))
    drop_from_peak = (trough_price - peak_price) / peak_price
    if drop_from_peak > drop_threshold:
        return False

    trough_rel = int(np.argmin(after_prices))
    if trough_rel < 0 or after_prices[trough_rel] >= peak_price:
        return False

    return True


def _norm01(series: pd.Series | float) -> pd.Series:
    if not isinstance(series, pd.Series):
        series = pd.Series([series])
    s = sanitize(pd.to_numeric(series, errors="coerce").fillna(0.0))
    return (np.tanh(s / 2.0) + 1.0) / 2.0


def compute_future_labels_for_coin(
    events: pd.DataFrame,
    snaps: pd.DataFrame,
    fee_pct: float,
) -> pd.DataFrame:
    """
    Per-coin_id future outcome labels. Iterates events within one coin group only.
    """
    if events.empty:
        return events

    out = events.copy()
    snaps = snaps.sort_values("ts")
    snap_ts = snaps["ts"].to_numpy(dtype="datetime64[ns]")
    snap_prices = snaps["price"].astype(float).to_numpy()
    event_ts = out["ts"].to_numpy(dtype="datetime64[ns]")
    base = out["price_usd"].astype(float).to_numpy()

    pump_peak_thresh = get_pump_then_dump_pump_threshold()
    pump_drop_thresh = get_pump_then_dump_drop_threshold()

    for horizon, minutes in HORIZON_MINUTES.items():
        delta = np.timedelta64(int(minutes), "m")
        max_ret = np.full(len(out), np.nan)
        min_ret = np.full(len(out), np.nan)
        end_ret = np.full(len(out), np.nan)
        end_price = np.full(len(out), np.nan)

        for i in range(len(out)):
            b = base[i]
            if np.isnan(b) or b <= 0:
                continue
            t0 = event_ts[i]
            t1 = t0 + delta
            mask = (snap_ts > t0) & (snap_ts <= t1)
            if not mask.any():
                continue
            prices = snap_prices[mask]
            rets = (prices - b) / b
            max_ret[i] = float(np.max(rets))
            min_ret[i] = float(np.min(rets))
            end_ret[i] = float(rets[-1])
            end_price[i] = float(prices[-1])

        out[f"max_future_return_{horizon}"] = max_ret
        out[f"min_future_return_{horizon}"] = min_ret
        out[f"future_return_{horizon}"] = end_ret
        out[f"future_price_{horizon}"] = end_price
        out[f"max_upside_{horizon}"] = max_ret
        out[f"max_drawdown_{horizon}"] = min_ret

        valid = ~np.isnan(end_ret)
        out[f"label_up_{horizon}"] = np.where(valid, end_ret > 0, np.nan)
        out[f"label_profitable_after_fees_{horizon}"] = np.where(valid, end_ret > fee_pct, np.nan)

        pump_th = get_big_pump_threshold(horizon)
        dump_th = get_big_dump_threshold(horizon)
        out[f"big_pump_{horizon}"] = np.where(~np.isnan(max_ret), max_ret >= pump_th, np.nan)
        out[f"big_dump_{horizon}"] = np.where(~np.isnan(min_ret), min_ret <= dump_th, np.nan)

    ptd_1h = np.full(len(out), np.nan)
    ptd_4h = np.full(len(out), np.nan)
    for i in range(len(out)):
        b = base[i]
        if np.isnan(b) or b <= 0:
            continue
        for _horizon, minutes, arr in (("1h", 60, ptd_1h), ("4h", 240, ptd_4h)):
            t0 = event_ts[i]
            t1 = t0 + np.timedelta64(int(minutes), "m")
            mask = (snap_ts > t0) & (snap_ts <= t1)
            if not mask.any():
                continue
            prices = snap_prices[mask]
            arr[i] = detect_pump_then_dump(
                prices,
                float(b),
                pump_threshold=pump_peak_thresh,
                drop_threshold=pump_drop_thresh,
            )

    out["pump_then_dump_1h"] = ptd_1h
    out["pump_then_dump_4h"] = ptd_4h

    missing = sum(
        1
        for h in HORIZON_MINUTES
        if out[f"future_return_{h}"].isna().all()
    )
    out["pending_outcome"] = out[[f"future_return_{h}" for h in HORIZON_MINUTES]].isna().any(axis=1)
    return out


def compute_future_labels(events: pd.DataFrame, snaps: pd.DataFrame, fee_pct: float) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for coin_id, evt_group in events.groupby("coin_id", sort=False):
        snap_group = snaps[snaps["coin_id"] == coin_id]
        parts.append(compute_future_labels_for_coin(evt_group, snap_group, fee_pct))
    if not parts:
        return events
    return pd.concat(parts, ignore_index=True)


def add_whale_wave_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    whale = out["whale_score"].astype(float) if "whale_score" in out else pd.Series(0.0, index=out.index)
    score = (
        _norm01(out.get("price_velocity_15m", pd.Series(0.0, index=out.index))) * 0.18
        + _norm01(out.get("price_acceleration_5m_to_15m", pd.Series(0.0, index=out.index))) * 0.12
        + _norm01(out.get("log_volume_spike", pd.Series(0.0, index=out.index))) * 0.20
        + _norm01(out.get("buy_sell_imbalance", pd.Series(0.0, index=out.index)).abs()) * 0.15
        + _norm01(out.get("txn_velocity_15m", pd.Series(0.0, index=out.index))) * 0.10
        + _norm01(out.get("liquidity_shock_score", pd.Series(0.0, index=out.index))) * 0.10
        + whale.fillna(0.0).clip(0.0, 1.0) * 0.15
    )
    out["whale_wave_score"] = sanitize(score).clip(0.0, 1.0)

    buy_p = out.get("buy_pressure", pd.Series(np.nan, index=out.index)).astype(float)
    sell_p = out.get("sell_pressure", pd.Series(np.nan, index=out.index)).astype(float)
    imb = out.get("buy_sell_imbalance", pd.Series(np.nan, index=out.index)).astype(float)

    direction = np.select(
        [
            buy_p.gt(sell_p + 0.15) & imb.gt(0.1),
            sell_p.gt(buy_p + 0.15) & imb.lt(-0.1),
            imb.abs().lt(0.05),
        ],
        ["UP", "DOWN", "UNKNOWN"],
        default="MIXED",
    )
    out["whale_wave_direction"] = direction
    return out


def add_position_sizing_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    aggressive_th = get_whale_wave_aggressive_threshold()
    normal_th = get_whale_wave_normal_threshold()
    wws = out.get("whale_wave_score", pd.Series(0.0, index=out.index)).astype(float).fillna(0.0)

    for horizon in ("1h", "4h"):
        prof = out.get(f"label_profitable_after_fees_{horizon}", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        pump = out.get(f"big_pump_{horizon}", pd.Series(False, index=out.index)).fillna(False).astype(bool)
        min_ret = out.get(f"min_future_return_{horizon}", pd.Series(np.nan, index=out.index)).astype(float)
        dump_th = get_big_dump_threshold(horizon)

        cond_avoid = min_ret.le(dump_th * 1.2)
        cond_agg = (~cond_avoid) & (((prof) & (wws >= aggressive_th)) | ((pump) & (wws >= aggressive_th)))
        cond_norm = (~cond_avoid) & (~cond_agg) & prof & (wws >= normal_th) & (wws < aggressive_th)
        cond_small = (~cond_avoid) & (~cond_agg) & (~cond_norm) & prof

        col = f"optimal_trade_class_{horizon}"
        out[col] = np.select(
            [cond_avoid, cond_agg, cond_norm, cond_small],
            ["AVOID_DUMP", "AGGRESSIVE_WHALE_TRADE", "NORMAL_TRADE", "SMALL_PROBE"],
            default="NO_TRADE",
        )
        out[f"position_size_multiplier_{horizon}"] = out[col].map(TRADE_CLASS_MULTIPLIERS).fillna(0.0)

    return out


def _snapshot_attach_columns(featured_snaps: pd.DataFrame) -> list[str]:
    candidates = [
        "ts",
        "price",
        "liquidity",
        "volume_24h",
        "txns_buys",
        "txns_sells",
        "buy_ratio",
        "whale_score",
        "price_change_h1",
        "price_change_h24",
        "chain",
        "pair_address",
        "has_whale_wave_history",
        *HISTORICAL_FEATURE_COLUMNS,
    ]
    return [c for c in candidates if c in featured_snaps.columns]


def _merge_group_key(frame: pd.DataFrame) -> pd.Series:
    pair = frame["pair_address"] if "pair_address" in frame.columns else pd.Series(np.nan, index=frame.index)
    coin = frame["coin_id"] if "coin_id" in frame.columns else pd.Series(np.nan, index=frame.index)
    pair_str = pair.astype("string")
    coin_str = coin.astype("string")
    grouped = pair_str.where(pair_str.notna() & (pair_str != ""), other=pd.NA)
    grouped = grouped.fillna("coin:" + coin_str)
    return grouped


def _ensure_snapshot_ts(snaps: pd.DataFrame) -> pd.Series:
    if "ts" in snaps.columns:
        return pd.to_datetime(snaps["ts"], utc=True, errors="coerce")
    if "timestamp" in snaps.columns:
        return pd.to_datetime(snaps["timestamp"], utc=True, errors="coerce")
    return pd.Series(pd.NaT, index=snaps.index)


def _finalize_attached_columns(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    if "price" in out.columns:
        out["price_usd"] = out["price"].astype(float)
    if "liquidity" in out.columns:
        out["liquidity_usd"] = out["liquidity"].astype(float)
    if "price_change_h1" in out.columns:
        out["price_change_1h"] = out["price_change_h1"].astype(float)
    if "price_change_h24" in out.columns:
        out["price_change_24h"] = out["price_change_h24"].astype(float)
    return out


def attach_historical_features(
    events: pd.DataFrame,
    featured_snaps: pd.DataFrame,
    warnings: list[str] | None = None,
) -> pd.DataFrame:
    """
    Attach historical snapshot features via per-pair merge_asof (backward only).
    Group-level processing — not row-by-row.
    """
    warn = warnings if warnings is not None else []

    if events.empty:
        return events

    input_count = len(events)
    evt = events.copy()
    evt["_event_row_id"] = np.arange(len(evt), dtype=np.int64)
    evt["ts"] = pd.to_datetime(evt["event_timestamp"], utc=True, errors="coerce")
    evt["_merge_group"] = _merge_group_key(evt)

    snap_attach_cols = _snapshot_attach_columns(featured_snaps)
    if featured_snaps.empty or not snap_attach_cols:
        warn.append("No featured snapshots available — whale-wave features set to NaN")
        for col in (*HISTORICAL_FEATURE_COLUMNS, "has_whale_wave_history", "price_usd"):
            if col not in evt.columns:
                evt[col] = np.nan
        evt = evt.sort_values("_event_row_id").drop(columns=["_event_row_id", "_merge_group"], errors="ignore")
        return _finalize_attached_columns(evt)

    snaps = featured_snaps.copy()
    snaps["ts"] = _ensure_snapshot_ts(snaps)
    snaps["_merge_group"] = _merge_group_key(snaps)
    snaps = snaps[snaps["ts"].notna() & snaps["_merge_group"].notna()]

    mergeable_mask = evt["ts"].notna() & evt["_merge_group"].notna()
    mergeable = evt.loc[mergeable_mask].copy()
    skipped = evt.loc[~mergeable_mask].copy()

    merged_parts: list[pd.DataFrame] = []
    snap_cols_for_merge = [c for c in snap_attach_cols if c != "ts"]

    for group_key, evt_group in mergeable.groupby("_merge_group", sort=False):
        snap_group = snaps.loc[snaps["_merge_group"] == group_key, ["ts", *snap_cols_for_merge]]
        evt_sorted = evt_group.sort_values("ts")
        if snap_group.empty:
            merged_parts.append(evt_sorted)
            continue

        snap_sorted = snap_group.sort_values("ts").drop_duplicates(subset=["ts"], keep="last")
        if not evt_sorted["ts"].is_monotonic_increasing:
            raise ValueError(f"Event timestamps not sorted for merge group {group_key}")
        if not snap_sorted["ts"].is_monotonic_increasing:
            raise ValueError(f"Snapshot timestamps not sorted for merge group {group_key}")

        merged_group = pd.merge_asof(
            evt_sorted,
            snap_sorted,
            on="ts",
            direction="backward",
            suffixes=("", "_snap"),
        )
        merged_parts.append(merged_group)

    if merged_parts:
        merged = pd.concat([*merged_parts, skipped], ignore_index=True)
    else:
        merged = skipped

    merged = merged.sort_values("_event_row_id")
    merged = merged.drop(columns=["_event_row_id", "_merge_group"], errors="ignore")

    if len(merged) != input_count:
        raise ValueError(
            f"attach_historical_features row count mismatch: in={input_count} out={len(merged)}"
        )

    return _finalize_attached_columns(merged)
