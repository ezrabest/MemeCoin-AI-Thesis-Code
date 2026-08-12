"""AE8 whale / wallet intelligence context."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Any

from app.context_intelligence.freshness import apply_stale_nulling, compute_freshness, parse_timestamp
from app.context_intelligence.types import FreshnessMode, SourceStatus

WHALE_SCORE_RESEARCH_METADATA = {
    "whale_score_status": "RESEARCH_ONLY_PLAUSIBLE_FEATURE_CANDIDATE",
    "not_rule": True,
    "not_runtime_approved_as_standalone_signal": True,
}


def build_whale_context(
    conn: sqlite3.Connection | None,
    *,
    coin_id: int | None,
    pair_address: str | None,
    snapshot_row: dict[str, Any] | None,
    as_of_timestamp: str,
    freshness_reference_timestamp: str,
    freshness_mode: FreshnessMode | str,
    threshold_minutes: float,
    stats: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any], str, list[str], dict[str, Any]]:
    warnings: list[str] = []
    empty: dict[str, Any] = {
        "whale_score_asof": None,
        "whale_score_missingness": True,
        "whale_activity_count_1h": None,
        "whale_activity_count_24h": None,
        "whale_buy_pressure_24h": None,
        "whale_sell_pressure_24h": None,
        "whale_net_flow_proxy_24h": None,
        "whale_unique_wallets_24h": None,
        "whale_top_wallet_concentration": None,
        "whale_context_freshness_minutes": None,
        **WHALE_SCORE_RESEARCH_METADATA,
    }

    source_ts = (snapshot_row or {}).get("timestamp")
    whale_score = (snapshot_row or {}).get("whale_score")

    alerts: list[sqlite3.Row] | list[Any] = []
    if conn is not None and coin_id is not None:
        as_of_dt = parse_timestamp(as_of_timestamp)
        cutoff = (as_of_dt - timedelta(hours=24)).isoformat() if as_of_dt else as_of_timestamp
        sql = """
            SELECT timestamp, alert_type, whale_score, tx_summary_json
            FROM whale_alerts
            WHERE coin_id = ? AND timestamp <= ? AND timestamp >= ?
            ORDER BY timestamp DESC
            LIMIT 100
        """
        if stats is not None:
            stats.queries_executed.append(sql)
        try:
            alerts = conn.execute(sql, (coin_id, as_of_timestamp, cutoff)).fetchall()
            if stats is not None:
                stats.record_scan("whale_alerts", len(alerts))
                stats.record_load("whale_alerts", len(alerts))
        except sqlite3.Error:
            alerts = []

    if not snapshot_row and not alerts:
        warnings.append("WHALE_CONTEXT_NOT_AVAILABLE")
        freshness = compute_freshness(
            source_timestamp=None,
            freshness_reference_timestamp=freshness_reference_timestamp,
            freshness_mode=freshness_mode,
            threshold_minutes=threshold_minutes,
            family_key="whale",
        )
        return empty, freshness, SourceStatus.SOURCE_NOT_AVAILABLE.value, warnings, WHALE_SCORE_RESEARCH_METADATA

    if alerts:
        source_ts = str(alerts[0][0])

    count_1h = count_24h = 0
    buy_pressure = sell_pressure = 0.0
    wallets: set[str] = set()
    as_of_dt = parse_timestamp(as_of_timestamp)

    for alert in alerts:
        ts = parse_timestamp(str(alert[0]))
        if ts is None or as_of_dt is None:
            continue
        age_h = (as_of_dt - ts).total_seconds() / 3600.0
        if age_h <= 24:
            count_24h += 1
            alert_type = str(alert[1] or "").lower()
            if "buy" in alert_type:
                buy_pressure += 1.0
            elif "sell" in alert_type:
                sell_pressure += 1.0
            try:
                tx_summary = json.loads(alert[3] or "{}")
                if isinstance(tx_summary, dict):
                    w = tx_summary.get("wallet") or tx_summary.get("address")
                    if w:
                        wallets.add(str(w))
            except (json.JSONDecodeError, TypeError):
                pass
        if age_h <= 1:
            count_1h += 1

    features: dict[str, Any] = {
        "whale_score_asof": float(whale_score) if whale_score is not None else None,
        "whale_score_missingness": whale_score is None,
        "whale_activity_count_1h": count_1h,
        "whale_activity_count_24h": count_24h,
        "whale_buy_pressure_24h": buy_pressure,
        "whale_sell_pressure_24h": sell_pressure,
        "whale_net_flow_proxy_24h": buy_pressure - sell_pressure,
        "whale_unique_wallets_24h": len(wallets),
        "whale_top_wallet_concentration": None,
        "whale_context_freshness_minutes": None,
        **WHALE_SCORE_RESEARCH_METADATA,
    }

    freshness = compute_freshness(
        source_timestamp=str(source_ts) if source_ts else None,
        freshness_reference_timestamp=freshness_reference_timestamp,
        freshness_mode=freshness_mode,
        threshold_minutes=threshold_minutes,
        family_key="whale",
    )
    features["whale_context_freshness_minutes"] = freshness.get("freshness_minutes")

    if freshness.get("freshness_status") == "STALE":
        source_status = SourceStatus.SOURCE_STALE.value
    else:
        source_status = SourceStatus.SOURCE_OK.value if snapshot_row or alerts else SourceStatus.SOURCE_EMPTY.value

    nulled = apply_stale_nulling(
        {k: v for k, v in features.items() if k not in WHALE_SCORE_RESEARCH_METADATA},
        freshness,
        missingness_flag_key="whale_score_missingness",
    )
    features.update(nulled)
    features.update(WHALE_SCORE_RESEARCH_METADATA)
    return features, freshness, source_status, warnings, WHALE_SCORE_RESEARCH_METADATA
