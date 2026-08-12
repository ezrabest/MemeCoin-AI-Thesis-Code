"""Rolling whale-wave features for candidate payload (audit only)."""
from __future__ import annotations

from typing import Any


def compute_rolling_whale_wave_features(
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute rolling features from ordered snapshots (oldest first).
    No future leakage — uses only provided history.
    """
    if not snapshots:
        return {}

    latest = snapshots[-1]
    prev = snapshots[-2] if len(snapshots) >= 2 else None
    prev3 = snapshots[-4] if len(snapshots) >= 4 else None

    features: dict[str, Any] = {
        "snapshot_count": len(snapshots),
    }

    def _delta(cur: dict, old: dict | None, key: str) -> float | None:
        if old is None:
            return None
        return float((cur.get(key) or 0) - (old.get(key) or 0))

    if prev:
        features["volume_delta_5m"] = _delta(latest, prev, "volume_24h")
        features["buy_txn_delta_5m"] = _delta(latest, prev, "txns_buys")
        features["sell_txn_delta_5m"] = _delta(latest, prev, "txns_sells")
        prev_br = prev.get("buy_ratio") or 0.5
        cur_br = latest.get("buy_ratio") or 0.5
        features["buy_ratio_slope"] = round(cur_br - prev_br, 6)
        prev_ws = prev.get("whale_score") or 0
        cur_ws = latest.get("whale_score") or 0
        features["whale_score_slope"] = round(cur_ws - prev_ws, 6)
        prev_pc = prev.get("price_change_h1") or 0
        cur_pc = latest.get("price_change_h1") or 0
        features["price_acceleration"] = round(cur_pc - prev_pc, 6)

    if prev3:
        features["volume_delta_15m"] = _delta(latest, prev3, "volume_24h")

    liq = float(latest.get("liquidity") or latest.get("liquidity_usd") or 0)
    vol = float(latest.get("volume_24h") or 0)
    if liq > 0:
        features["volume_liquidity_ratio"] = round(vol / liq, 6)
        if prev and float(prev.get("liquidity") or prev.get("liquidity_usd") or 0) > 0:
            prev_ratio = float(prev.get("volume_24h") or 0) / float(prev.get("liquidity") or prev.get("liquidity_usd") or 1)
            features["volume_liquidity_ratio_change"] = round(features["volume_liquidity_ratio"] - prev_ratio, 6)

    burst_count = 0
    for i in range(1, len(snapshots)):
        ws_delta = float(snapshots[i].get("whale_score") or 0) - float(snapshots[i - 1].get("whale_score") or 0)
        if ws_delta > 0.05:
            burst_count += 1
    features["burst_count_recent"] = burst_count
    features["repeated_accumulation_burst"] = burst_count >= 2 and (latest.get("buy_ratio") or 0) >= 0.55
    features["repeated_distribution_burst"] = burst_count >= 2 and (latest.get("buy_ratio") or 0) <= 0.45

    return features


def load_snapshots_for_pair(pair_address: str, *, limit: int = 20) -> list[dict[str, Any]]:
    """Load recent market snapshots from SQLite for whale-wave features."""
    from .. import database as db

    try:
        with db.get_db() as conn:
            rows = conn.execute(
                """
                SELECT timestamp, price, liquidity, volume_24h, txns_buys, txns_sells,
                       buy_ratio, whale_score, price_change_m5, price_change_h1, price_change_h24
                FROM market_snapshots
                WHERE pair_address = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (pair_address, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]
    except Exception:
        return []
