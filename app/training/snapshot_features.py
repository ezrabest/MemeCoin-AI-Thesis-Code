"""
Vectorized historical whale-wave features on market snapshot time series.

Features use only data at or before each snapshot timestamp (no future leakage).
Per-coin_id group iteration is used (~hundreds of coins, not per event row).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger("training.snapshot_features")

EPS = 1e-9
CLIP = 1e6
# ~5 min poll cadence proxies
WIN_15M = 3
WIN_1H = 12
WIN_4H = 48
WIN_24H = 288

HISTORICAL_FEATURE_COLUMNS = [
    "volume_spike_ratio_15m_vs_1h",
    "volume_spike_ratio_15m_vs_4h",
    "volume_zscore_1h",
    "volume_zscore_4h",
    "volume_relative_to_pair_median_4h",
    "volume_relative_to_pair_median_24h",
    "log_volume_spike",
    "buy_sell_ratio",
    "buy_pressure",
    "sell_pressure",
    "buy_sell_imbalance",
    "buy_sell_imbalance_change_15m",
    "buy_sell_imbalance_zscore_4h",
    "txn_count_change_15m",
    "txn_velocity_15m",
    "price_return_5m",
    "price_return_15m",
    "price_return_1h",
    "price_velocity_5m",
    "price_velocity_15m",
    "price_acceleration_5m_to_15m",
    "price_distance_from_recent_low",
    "price_distance_from_recent_high",
    "liquidity_change_15m",
    "liquidity_change_1h",
    "liquidity_to_volume_ratio",
    "fdv_liquidity_ratio",
    "liquidity_shock_score",
]


def sanitize(series: pd.Series) -> pd.Series:
    out = series.replace([np.inf, -np.inf], np.nan)
    return out.clip(lower=-CLIP, upper=CLIP)


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    denom = b.where(b.abs() > EPS, np.nan)
    return sanitize(a / denom)


def _compute_coin_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.sort_values("ts").copy()
    n = len(g)
    if n == 0:
        return g

    price = g["price"].astype(float)
    volume = g["volume_24h"].astype(float) if "volume_24h" in g else pd.Series(np.nan, index=g.index)
    liquidity = g["liquidity"].astype(float) if "liquidity" in g else pd.Series(np.nan, index=g.index)
    fdv = g["fdv"].astype(float) if "fdv" in g else pd.Series(np.nan, index=g.index)
    buys = g["txns_buys"].astype(float) if "txns_buys" in g else pd.Series(0.0, index=g.index)
    sells = g["txns_sells"].astype(float) if "txns_sells" in g else pd.Series(0.0, index=g.index)
    total_txn = buys + sells

    vol_15m = volume.rolling(WIN_15M, min_periods=1).mean()
    vol_1h = volume.rolling(WIN_1H, min_periods=1).mean()
    vol_4h = volume.rolling(WIN_4H, min_periods=1).mean()
    vol_24h_med = volume.rolling(WIN_24H, min_periods=1).median()

    g["volume_spike_ratio_15m_vs_1h"] = _safe_div(vol_15m, vol_1h)
    g["volume_spike_ratio_15m_vs_4h"] = _safe_div(vol_15m, vol_4h)
    vol_std_1h = volume.rolling(WIN_1H, min_periods=2).std()
    vol_std_4h = volume.rolling(WIN_4H, min_periods=2).std()
    g["volume_zscore_1h"] = sanitize((volume - vol_1h) / vol_std_1h.replace(0, np.nan))
    g["volume_zscore_4h"] = sanitize((volume - vol_4h) / vol_std_4h.replace(0, np.nan))
    g["volume_relative_to_pair_median_4h"] = _safe_div(volume, vol_4h)
    g["volume_relative_to_pair_median_24h"] = _safe_div(volume, vol_24h_med)
    g["log_volume_spike"] = sanitize(np.log1p(g["volume_spike_ratio_15m_vs_1h"].clip(lower=0)))

    g["buy_sell_ratio"] = _safe_div(buys, sells)
    g["buy_pressure"] = _safe_div(buys, total_txn)
    g["sell_pressure"] = _safe_div(sells, total_txn)
    g["buy_sell_imbalance"] = sanitize(g["buy_pressure"] - g["sell_pressure"])

    imb = g["buy_sell_imbalance"]
    imb_lag = imb.shift(WIN_15M)
    g["buy_sell_imbalance_change_15m"] = sanitize(imb - imb_lag)
    imb_mean_4h = imb.rolling(WIN_4H, min_periods=1).mean()
    imb_std_4h = imb.rolling(WIN_4H, min_periods=2).std()
    g["buy_sell_imbalance_zscore_4h"] = sanitize((imb - imb_mean_4h) / imb_std_4h.replace(0, np.nan))

    txn_lag = total_txn.shift(WIN_15M)
    g["txn_count_change_15m"] = sanitize(total_txn - txn_lag)
    g["txn_velocity_15m"] = sanitize(g["txn_count_change_15m"] / 15.0)

    g["price_return_5m"] = sanitize(price.pct_change(1))
    g["price_return_15m"] = sanitize(price.pct_change(WIN_15M))
    if "price_change_h1" in g.columns:
        g["price_return_1h"] = sanitize(g["price_change_h1"].astype(float) / 100.0)
    else:
        g["price_return_1h"] = sanitize(price.pct_change(WIN_1H))

    g["price_velocity_5m"] = g["price_return_5m"]
    g["price_velocity_15m"] = g["price_return_15m"]
    g["price_acceleration_5m_to_15m"] = sanitize(g["price_velocity_15m"] - g["price_velocity_5m"])

    roll_low = price.rolling(WIN_1H, min_periods=1).min()
    roll_high = price.rolling(WIN_1H, min_periods=1).max()
    g["price_distance_from_recent_low"] = _safe_div(price - roll_low, roll_low)
    g["price_distance_from_recent_high"] = _safe_div(roll_high - price, roll_high)

    liq_lag_15m = liquidity.shift(WIN_15M)
    liq_lag_1h = liquidity.shift(WIN_1H)
    g["liquidity_change_15m"] = _safe_div(liquidity - liq_lag_15m, liq_lag_15m)
    g["liquidity_change_1h"] = _safe_div(liquidity - liq_lag_1h, liq_lag_1h)
    g["liquidity_to_volume_ratio"] = _safe_div(liquidity, volume)
    g["fdv_liquidity_ratio"] = _safe_div(fdv, liquidity)
    g["liquidity_shock_score"] = sanitize(g["liquidity_change_15m"].abs() + g["liquidity_change_1h"].abs())

    return g


def compute_snapshot_historical_features(
    snap_df: pd.DataFrame,
    warnings: list[str],
) -> pd.DataFrame:
    if snap_df.empty:
        return snap_df

    required = {"coin_id", "timestamp"}
    missing = required - set(snap_df.columns)
    if missing:
        warnings.append(f"Snapshot table missing columns {sorted(missing)} — historical features skipped")
        return snap_df

    df = snap_df.copy()
    df["ts"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts", "coin_id"])

    if "liquidity" not in df.columns and "liquidity_usd" in df.columns:
        df["liquidity"] = df["liquidity_usd"]

    parts = [_compute_coin_features(group) for _, group in df.groupby("coin_id", sort=False)]
    featured = pd.concat(parts, ignore_index=True)
    featured["has_whale_wave_history"] = featured["volume_spike_ratio_15m_vs_1h"].notna()
    return featured.sort_values(["coin_id", "ts"])
