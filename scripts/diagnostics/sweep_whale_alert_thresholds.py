#!/usr/bin/env python3
"""Diagnostic 2 — sweep whale alert thresholds on recent snapshots (counterfactual)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.diagnostics._common import (
    CHUNK_SIZE,
    DB_PATH,
    WHALE_SCORE_HISTOGRAM_EDGES,
    DiagnosticReport,
    distribution_stats,
    histogram_buckets,
    iter_snapshot_chunks,
    open_db_readonly,
    safe_float,
    snapshot_to_pair,
)


def _counterfactual_alert(pair: dict, whale_score: float, min_score: float) -> bool:
    from app.engine import WHALE_ALERT_MIN_VOLUME_24H, _vol

    if whale_score < min_score or _vol(pair) < WHALE_ALERT_MIN_VOLUME_24H:
        return False
    from app.engine import detect_whale_alert

    if whale_score >= 0.30:
        return detect_whale_alert(pair, whale_score) is not None
    return False


def run(*, latest_snapshots: int, thresholds: list[float], output_dir: Path) -> DiagnosticReport:
    from app.engine import (
        SIGNAL_BUY_LIQUIDITY_USD,
        SIGNAL_BUY_WHALE_THRESHOLD,
        SIGNAL_WATCH_WHALE_THRESHOLD,
        WHALE_ALERT_MIN_VOLUME_24H,
        compute_whale_score,
        detect_whale_alert,
        generate_signal,
        _buy_ratio,
        _liq,
        _vol,
    )

    report = DiagnosticReport("sweep_whale_alert_thresholds", output_dir)
    if not DB_PATH.is_file():
        report.add_limitation(f"Database missing: {DB_PATH}")
        report.set_status("FAIL")
        return report

    conn = open_db_readonly()
    whale_scores: list[float] = []
    per_threshold: dict[float, dict] = {
        t: {
            "threshold": t,
            "snapshots_evaluated": 0,
            "unique_pairs": set(),
            "count_gte_threshold": 0,
            "alert_like_count": 0,
            "watch_like_count": 0,
            "buy_like_count": 0,
            "blocked_by_liquidity": 0,
            "blocked_by_buy_ratio": 0,
            "blocked_by_volume": 0,
            "blocked_by_momentum": 0,
            "top_examples": [],
        }
        for t in thresholds
    }

    try:
        processed = 0
        for chunk in iter_snapshot_chunks(conn, limit=latest_snapshots, chunk_size=CHUNK_SIZE):
            for snap in chunk:
                if snap.get("filter_status") not in (None, "passed"):
                    continue
                pair = snapshot_to_pair(snap)
                ws = safe_float(snap.get("whale_score"))
                if ws <= 0:
                    ws = compute_whale_score(pair)
                whale_scores.append(ws)
                liq = _liq(pair)
                vol = _vol(pair)
                br = _buy_ratio(pair)
                sig = generate_signal(pair, ws)
                alert = detect_whale_alert(pair, ws)

                for t in thresholds:
                    bucket = per_threshold[t]
                    bucket["snapshots_evaluated"] += 1
                    pa = str(snap.get("pair_address") or "")
                    bucket["unique_pairs"].add(pa)
                    if ws >= t:
                        bucket["count_gte_threshold"] += 1
                        if alert:
                            bucket["alert_like_count"] += 1
                        elif _counterfactual_alert(pair, ws, t):
                            bucket["alert_like_count"] += 1
                        if sig["action"] == "WATCH" or ws >= SIGNAL_WATCH_WHALE_THRESHOLD:
                            bucket["watch_like_count"] += 1
                        if sig["action"] == "BUY":
                            bucket["buy_like_count"] += 1
                        if liq < SIGNAL_BUY_LIQUIDITY_USD:
                            bucket["blocked_by_liquidity"] += 1
                        if br < 0.50:
                            bucket["blocked_by_buy_ratio"] += 1
                        if vol < WHALE_ALERT_MIN_VOLUME_24H:
                            bucket["blocked_by_volume"] += 1
                        pc1h = safe_float((pair.get("priceChange") or {}).get("h1"))
                        if abs(pc1h) < 1.0 and ws < t + 0.05:
                            bucket["blocked_by_momentum"] += 1
                        if len(bucket["top_examples"]) < 20:
                            bucket["top_examples"].append({
                                "pair_address": pa,
                                "symbol": snap.get("coin_symbol"),
                                "timestamp": snap.get("timestamp"),
                                "whale_score": round(ws, 4),
                                "liquidity": liq,
                                "volume_24h": vol,
                                "buy_ratio": round(br, 4),
                                "alert_type": alert["alert_type"] if alert else None,
                            })
                processed += 1
            if processed >= latest_snapshots:
                break
    finally:
        conn.close()

    threshold_rows = []
    for t in thresholds:
        b = per_threshold[t]
        threshold_rows.append({
            "threshold": t,
            "snapshots_evaluated": b["snapshots_evaluated"],
            "unique_pairs_evaluated": len(b["unique_pairs"]),
            "count_gte_threshold": b["count_gte_threshold"],
            "alert_like_count": b["alert_like_count"],
            "watch_like_count": b["watch_like_count"],
            "buy_like_count": b["buy_like_count"],
            "blocked_by_liquidity": b["blocked_by_liquidity"],
            "blocked_by_buy_ratio": b["blocked_by_buy_ratio"],
            "blocked_by_volume": b["blocked_by_volume"],
            "blocked_by_momentum": b["blocked_by_momentum"],
            "top_examples": sorted(b["top_examples"], key=lambda x: x["whale_score"], reverse=True)[:10],
        })

    dist = distribution_stats(whale_scores)
    hist = histogram_buckets(whale_scores, WHALE_SCORE_HISTOGRAM_EDGES)
    max_ws = dist.get("max") or 0
    p95 = dist.get("p95") or 0
    if max_ws < 0.30:
        status = "FAIL"
    elif p95 < 0.30:
        status = "WARN"
    else:
        status = "PASS"

    report.set_status(status)
    report.data.update({
        "snapshots_processed": len(whale_scores),
        "whale_score_distribution": dist,
        "whale_score_histogram": hist,
        "per_threshold": threshold_rows,
        "note": "Counterfactual only — production thresholds unchanged.",
    })
    report.write_json("sweep_whale_alert_thresholds.json")
    report.write_md([
        "## Whale score distribution",
        f"- count={dist.get('count')} p50={dist.get('p50')} p95={dist.get('p95')} max={dist.get('max')}",
        "",
        "## Histogram",
        *[f"- {k}: {v}" for k, v in hist.items()],
    ], "sweep_whale_alert_thresholds.md")
    flat = []
    for row in threshold_rows:
        for ex in row.pop("top_examples", []):
            flat.append({**ex, "threshold": row["threshold"]})
    report.write_csv(flat, "threshold_sweep_top_examples.csv")
    csv_rows = [{k: v for k, v in r.items() if k != "top_examples"} for r in threshold_rows]
    report.write_csv(csv_rows, "threshold_sweep_summary.csv")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest-snapshots", type=int, default=10000)
    parser.add_argument("--thresholds", default="0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    report = run(latest_snapshots=args.latest_snapshots, thresholds=thresholds, output_dir=args.output_dir)
    print(f"Status: {report.data['status']}")
    return 0 if report.data["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
