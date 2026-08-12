"""Whale-wave timing audit report (Phase 1 — report only, no runtime decisions)."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from .. import database as db
from .audit_io import utc_timestamp_slug, write_json_report_atomic
from .effective_settings import get_effective_settings
from .event_dedup import deduplicate_events

log = logging.getLogger("whale_wave_audit")

BULLISH_TYPES = frozenset({"LARGE_BUY", "ACCUMULATION", "PUMP_SIGNAL"})
BEARISH_TYPES = frozenset({"LARGE_SELL", "DISTRIBUTION"})


def _rolling_features_for_pair(conn, pair_address: str, limit: int = 50) -> list[dict[str, Any]]:
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
    result = []
    prev = None
    for row in reversed(rows):
        d = dict(row)
        if prev:
            d["volume_delta"] = (d.get("volume_24h") or 0) - (prev.get("volume_24h") or 0)
            d["buy_txn_delta"] = (d.get("txns_buys") or 0) - (prev.get("txns_buys") or 0)
            d["sell_txn_delta"] = (d.get("txns_sells") or 0) - (prev.get("txns_sells") or 0)
            prev_br = prev.get("buy_ratio") or 0.5
            cur_br = d.get("buy_ratio") or 0.5
            d["buy_ratio_slope"] = cur_br - prev_br
            prev_ws = prev.get("whale_score") or 0
            cur_ws = d.get("whale_score") or 0
            d["whale_score_slope"] = cur_ws - prev_ws
            prev_pc = prev.get("price_change_h1") or 0
            cur_pc = d.get("price_change_h1") or 0
            d["price_acceleration"] = cur_pc - prev_pc
            prev_liq = prev.get("liquidity") or 1
            cur_liq = d.get("liquidity") or 1
            d["vol_liq_ratio_change"] = (d.get("volume_24h") or 0) / max(cur_liq, 1) - (prev.get("volume_24h") or 0) / max(prev_liq, 1)
        prev = d
        result.append(d)
    return result


def _classify_alert_timing(features: list[dict[str, Any]], alert_ts: str, alert_type: str) -> str:
    """Classify whether alert was before/during/after price move."""
    if not features:
        return "unknown"
    alert_time = datetime.fromisoformat(alert_ts.replace("Z", "+00:00"))
    post_prices = [f for f in features if f.get("timestamp", "") >= alert_ts]
    pre_prices = [f for f in features if f.get("timestamp", "") < alert_ts]
    if alert_type in BULLISH_TYPES:
        if pre_prices and post_prices:
            pre_pc = pre_prices[-1].get("price_change_h1") or 0
            post_pc = post_prices[0].get("price_change_h1") or 0
            if pre_pc < 2 and post_pc > 5:
                return "before_move"
            if post_pc > pre_pc:
                return "during_move"
            return "after_move"
    return "unknown"


def run_whale_wave_audit(*, pair_limit: int = 30) -> dict[str, Any]:
    eff = get_effective_settings()
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "settings_hash": eff.settings_hash,
        "phase": 1,
        "note": "Audit/report only — whale-wave features not used for runtime decisions",
    }

    with db.get_db() as conn:
        alerts = conn.execute(
            """
            SELECT wa.timestamp, wa.pair_address, wa.chain, wa.symbol, wa.alert_type,
                   wa.whale_score, wa.volume, c.id AS coin_id
            FROM whale_alerts wa
            LEFT JOIN coins c ON c.id = wa.coin_id
            ORDER BY wa.id DESC
            LIMIT ?
            """,
            (pair_limit * 5,),
        ).fetchall()

        pairs_seen: set[str] = set()
        pair_analyses: list[dict[str, Any]] = []
        timing_counts = {"before_move": 0, "during_move": 0, "after_move": 0, "unknown": 0, "late_bullish": 0}

        for alert in alerts:
            pair_addr = alert["pair_address"] or ""
            if not pair_addr or pair_addr in pairs_seen:
                continue
            if len(pairs_seen) >= pair_limit:
                break
            pairs_seen.add(pair_addr)

            features = _rolling_features_for_pair(conn, pair_addr)
            timing = _classify_alert_timing(features, alert["timestamp"], alert["alert_type"])
            timing_counts[timing] = timing_counts.get(timing, 0) + 1
            if alert["alert_type"] in BULLISH_TYPES and timing == "after_move":
                timing_counts["late_bullish"] += 1

            pair_analyses.append({
                "pair_address": pair_addr,
                "symbol": alert["symbol"],
                "chain": alert["chain"],
                "latest_alert_type": alert["alert_type"],
                "alert_timestamp": alert["timestamp"],
                "timing_classification": timing,
                "snapshot_count": len(features),
                "rolling_features_sample": features[-3:] if features else [],
            })

        events = [
            {
                "pair_address": a["pair_address"],
                "chain": a["chain"] or "unknown",
                "event_type": a["alert_type"],
                "timestamp": a["timestamp"],
            }
            for a in alerts
        ]
        deduped = deduplicate_events(events)

    report["pair_analyses"] = pair_analyses
    report["timing_summary"] = timing_counts
    report["bullish_alert_timing"] = {
        "tend_to_occur": max(timing_counts, key=lambda k: timing_counts[k] if k != "late_bullish" else -1),
        "late_large_buy_accumulation_count": timing_counts.get("late_bullish", 0),
    }
    report["event_level_counts"] = {
        "raw_alert_rows": deduped["raw_event_count"],
        "event_level_count": deduped["event_level_count"],
        "dedup_ratio": deduped["dedup_ratio"],
    }
    report["detects_waves_too_late"] = timing_counts.get("late_bullish", 0) > timing_counts.get("before_move", 0)

    ts = utc_timestamp_slug()
    path = write_json_report_atomic(f"whale_wave_audit_{ts}.json", report)
    report["output_path"] = str(path)
    log.info("Whale-wave audit written: %s", path)
    return report
