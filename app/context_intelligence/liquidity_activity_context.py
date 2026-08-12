"""AE8 liquidity / activity anomaly context from local snapshots."""

from __future__ import annotations

from typing import Any

from app.context_intelligence.freshness import apply_stale_nulling, compute_freshness
from app.context_intelligence.types import FreshnessMode, SourceStatus


def safe_volume_to_liquidity_ratio(volume: float | None, liquidity: float | None) -> float | None:
    if volume is None or liquidity is None:
        return None
    if liquidity <= 0:
        return None
    return round(volume / liquidity, 6)


def _pct_change(current: float | None, prior: float | None) -> float | None:
    if current is None or prior is None or prior == 0:
        return None
    return round((current - prior) / abs(prior), 6)


def build_liquidity_activity_context(
    *,
    snapshot_row: dict[str, Any] | None,
    prior_snapshot_row: dict[str, Any] | None,
    prior_6h_snapshot_row: dict[str, Any] | None,
    signal_row: dict[str, Any] | None,
    as_of_timestamp: str,
    freshness_reference_timestamp: str,
    freshness_mode: FreshnessMode | str,
    threshold_minutes: float,
) -> tuple[dict[str, Any], dict[str, Any], str, list[str]]:
    warnings: list[str] = []
    empty: dict[str, Any] = {
        "liquidity_usd": None,
        "liquidity_change_1h": None,
        "liquidity_change_6h": None,
        "liquidity_anomaly_score": None,
        "volume_h24": None,
        "volume_change_1h": None,
        "volume_to_liquidity_ratio": None,
        "txns_h24_total": None,
        "buy_sell_ratio_h24": None,
        "activity_spike_score": None,
        "activity_decay_score": None,
        "price_change_m5": None,
        "price_change_h1": None,
        "price_change_h6": None,
        "price_change_h24": None,
        "liquidity_activity_freshness_minutes": None,
        "liquidity_activity_missingness_flag": True,
    }

    if not snapshot_row:
        warnings.append("LIQUIDITY_ACTIVITY_CONTEXT_NOT_AVAILABLE")
        freshness = compute_freshness(
            source_timestamp=None,
            freshness_reference_timestamp=freshness_reference_timestamp,
            freshness_mode=freshness_mode,
            threshold_minutes=threshold_minutes,
            family_key="liquidity_activity",
        )
        return empty, freshness, SourceStatus.SOURCE_NOT_AVAILABLE.value, warnings

    liq = snapshot_row.get("liquidity")
    vol = snapshot_row.get("volume_24h")
    buys = snapshot_row.get("txns_buys") or 0
    sells = snapshot_row.get("txns_sells") or 0
    total_txns = snapshot_row.get("txns_total")
    if total_txns is None:
        total_txns = int(buys or 0) + int(sells or 0)

    prior_liq = (prior_snapshot_row or {}).get("liquidity")
    prior_6h_liq = (prior_6h_snapshot_row or {}).get("liquidity")
    prior_vol = (prior_snapshot_row or {}).get("volume_24h")

    liq_f = float(liq) if liq is not None else None
    vol_f = float(vol) if vol is not None else None
    ratio = safe_volume_to_liquidity_ratio(vol_f, liq_f)

    liq_change_1h = _pct_change(liq_f, float(prior_liq) if prior_liq is not None else None)
    liq_change_6h = _pct_change(liq_f, float(prior_6h_liq) if prior_6h_liq is not None else None)
    vol_change_1h = _pct_change(vol_f, float(prior_vol) if prior_vol is not None else None)

    buy_sell = None
    if sells and float(sells) > 0:
        buy_sell = round(float(buys or 0) / float(sells), 6)
    elif buys:
        buy_sell = round(float(buys), 6)

    activity_spike = None
    if vol_change_1h is not None and vol_change_1h > 0.5:
        activity_spike = round(min(vol_change_1h, 10.0), 6)
    activity_decay = None
    if vol_change_1h is not None and vol_change_1h < -0.3:
        activity_decay = round(abs(vol_change_1h), 6)

    liq_anomaly = None
    if liq_change_1h is not None and abs(liq_change_1h) > 0.25:
        liq_anomaly = round(abs(liq_change_1h), 6)

    features: dict[str, Any] = {
        "liquidity_usd": liq_f,
        "liquidity_change_1h": liq_change_1h,
        "liquidity_change_6h": liq_change_6h,
        "liquidity_anomaly_score": liq_anomaly,
        "volume_h24": vol_f,
        "volume_change_1h": vol_change_1h,
        "volume_to_liquidity_ratio": ratio,
        "txns_h24_total": int(total_txns) if total_txns is not None else None,
        "buy_sell_ratio_h24": buy_sell,
        "activity_spike_score": activity_spike,
        "activity_decay_score": activity_decay,
        "price_change_m5": snapshot_row.get("price_change_m5"),
        "price_change_h1": snapshot_row.get("price_change_h1"),
        "price_change_h6": snapshot_row.get("price_change_h6"),
        "price_change_h24": snapshot_row.get("price_change_h24"),
        "liquidity_activity_freshness_minutes": None,
        "liquidity_activity_missingness_flag": False,
    }

    freshness = compute_freshness(
        source_timestamp=str(snapshot_row.get("timestamp") or ""),
        freshness_reference_timestamp=freshness_reference_timestamp,
        freshness_mode=freshness_mode,
        threshold_minutes=threshold_minutes,
        family_key="liquidity_activity",
    )
    features["liquidity_activity_freshness_minutes"] = freshness.get("freshness_minutes")

    if freshness.get("freshness_status") == "STALE":
        source_status = SourceStatus.SOURCE_STALE.value
    elif freshness.get("freshness_status") == "INVALID_FUTURE_TIMESTAMP":
        source_status = SourceStatus.SOURCE_ERROR.value
    else:
        source_status = SourceStatus.SOURCE_OK.value

    features = apply_stale_nulling(
        features,
        freshness,
        missingness_flag_key="liquidity_activity_missingness_flag",
    )
    return features, freshness, source_status, warnings
