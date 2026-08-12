#!/usr/bin/env python3
"""Diagnostic 1 — replay whale alert detection on historical snapshots."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.diagnostics._common import (
    DB_PATH,
    DiagnosticReport,
    open_db_readonly,
    parse_ts,
    reason_if_no_alert,
    row_to_dict,
    safe_float,
    snapshot_to_pair,
)


def _find_auto_window(conn) -> tuple[str, str]:
    row = conn.execute(
        """
        SELECT substr(timestamp, 1, 13) AS hour_bucket, COUNT(*) AS cnt
        FROM whale_alerts
        GROUP BY hour_bucket
        HAVING cnt > 0
        ORDER BY hour_bucket DESC
        LIMIT 1
        """
    ).fetchone()
    if not row:
        raise ValueError("No whale_alerts rows found for auto-window")
    hour_str = str(row["hour_bucket"])
    center = datetime.fromisoformat(hour_str.replace(" ", "T") + ":00:00+00:00")
    start = (center - timedelta(hours=2)).isoformat()
    end = (center + timedelta(hours=2)).isoformat()
    return start, end


def run(
    *,
    start: str,
    end: str,
    limit: int,
    output_dir: Path,
) -> DiagnosticReport:
    from app.engine import (
        WHALE_ALERT_MIN_VOLUME_24H,
        WHALE_ALERT_MIN_WHALE_SCORE,
        compute_whale_score,
        detect_whale_alert,
        generate_signal,
    )

    report = DiagnosticReport("replay_whale_alert_detection", output_dir)
    if not DB_PATH.is_file():
        report.add_limitation(f"Database missing: {DB_PATH}")
        report.set_status("FAIL")
        report.data["summary"] = {"error": "database_missing"}
        return report

    conn = open_db_readonly()
    try:
        historical_alerts = conn.execute(
            """
            SELECT id, timestamp, pair_address, symbol, alert_type, whale_score
            FROM whale_alerts
            WHERE timestamp >= ? AND timestamp <= ?
            """,
            (start, end),
        ).fetchall()
        hist_by_pair: dict[str, list[dict]] = {}
        for r in historical_alerts:
            d = row_to_dict(r)
            pa = str(d.get("pair_address") or "")
            hist_by_pair.setdefault(pa, []).append(d)

        snapshots = conn.execute(
            """
            SELECT ms.*, c.symbol AS coin_symbol
            FROM market_snapshots ms
            LEFT JOIN coins c ON c.id = ms.coin_id
            WHERE ms.timestamp >= ? AND ms.timestamp <= ?
              AND ms.filter_status = 'passed'
            ORDER BY ms.timestamp ASC
            LIMIT ?
            """,
            (start, end, limit),
        ).fetchall()
    finally:
        conn.close()

    examples: list[dict] = []
    would_alert = 0
    overlap_pairs = 0
    overlap_rows = 0

    for row in snapshots:
        snap = row_to_dict(row)
        pair = snapshot_to_pair(snap)
        ws = safe_float(snap.get("whale_score"))
        if ws <= 0:
            ws = compute_whale_score(pair)
        alert = detect_whale_alert(pair, ws)
        sig = generate_signal(pair, ws)
        pa = str(snap.get("pair_address") or "")
        hist_exists = pa in hist_by_pair
        detected = alert is not None
        if detected:
            would_alert += 1
        if detected and hist_exists:
            overlap_rows += 1
        ex = {
            "pair_address": pa,
            "symbol": snap.get("coin_symbol") or snap.get("symbol"),
            "timestamp": snap.get("timestamp"),
            "liquidity": safe_float(snap.get("liquidity")),
            "volume": safe_float(snap.get("volume_24h")),
            "buys": int(snap.get("txns_buys") or 0),
            "sells": int(snap.get("txns_sells") or 0),
            "buy_ratio": safe_float(snap.get("buy_ratio")),
            "whale_score_current": round(ws, 4),
            "historical_alert_exists": hist_exists,
            "current_detected_alert": detected,
            "current_alert_type": alert["alert_type"] if alert else None,
            "signal_action": sig["action"],
            "reason_if_no_alert": None if detected else reason_if_no_alert(pair, ws),
        }
        examples.append(ex)

    unique_overlap_pairs = {
        ex["pair_address"] for ex in examples if ex["current_detected_alert"] and ex["historical_alert_exists"]
    }
    overlap_pairs = len(unique_overlap_pairs)
    hist_count = len(historical_alerts)
    ratio = round(would_alert / hist_count, 4) if hist_count else None

    if hist_count > 0 and would_alert == 0:
        status = "FAIL"
    elif hist_count > 0 and ratio is not None and ratio < 0.25:
        status = "WARN"
    elif would_alert > 0:
        status = "PASS"
    else:
        status = "WARN" if not snapshots else "PASS"

    report.set_status(status)
    report.data.update({
        "window": {"start": start, "end": end},
        "input_snapshots_count": len(snapshots),
        "historical_persisted_whale_alerts_count": hist_count,
        "would_have_alert_count": would_alert,
        "overlap_by_pair_count": overlap_pairs,
        "overlap_row_count": overlap_rows,
        "would_have_alert_to_historical_ratio": ratio,
        "current_thresholds": {
            "min_volume_24h": WHALE_ALERT_MIN_VOLUME_24H,
            "min_whale_score": WHALE_ALERT_MIN_WHALE_SCORE,
        },
        "interpretation": (
            "Detector likely regressed if historical alerts existed but replay recreates none."
            if hist_count > 0 and would_alert == 0
            else "Live distribution may have shifted if replay matches history but live creates none."
        ),
        "top_50_replay_examples": sorted(
            examples,
            key=lambda x: (x["current_detected_alert"], x["whale_score_current"]),
            reverse=True,
        )[:50],
    })
    report.write_json("replay_whale_alert_detection.json")
    report.write_md([
        "## Summary",
        f"- Snapshots evaluated: {len(snapshots)}",
        f"- Historical alerts in window: {hist_count}",
        f"- Current detector would-have alerts: {would_alert}",
        f"- Overlap ratio: {ratio}",
        "",
        f"**Interpretation:** {report.data['interpretation']}",
    ], "replay_whale_alert_detection.md")
    top = report.data["top_50_replay_examples"]
    report.write_csv(top, "replay_whale_alert_examples.csv")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay whale alert detection")
    parser.add_argument("--start", default="2026-06-10T15:00:00Z")
    parser.add_argument("--end", default="2026-06-10T20:00:00Z")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--auto-window", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    start, end = args.start, args.end
    if args.auto_window:
        conn = open_db_readonly()
        try:
            start, end = _find_auto_window(conn)
            print(f"Auto window: {start} -> {end}")
        except ValueError as exc:
            print(f"WARN: auto-window failed ({exc}); using CLI window")
        finally:
            conn.close()

    report = run(start=start, end=end, limit=args.limit, output_dir=args.output_dir)
    print(f"Status: {report.data['status']} — {report.output_dir / 'replay_whale_alert_detection.json'}")
    return 0 if report.data["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
