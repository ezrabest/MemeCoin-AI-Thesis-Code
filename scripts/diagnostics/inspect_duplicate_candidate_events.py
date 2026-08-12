#!/usr/bin/env python3
"""Diagnostic 8 — duplicate candidate / pair inflation."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.diagnostics._common import (
    CHUNK_SIZE,
    DB_PATH,
    DiagnosticReport,
    iter_candidate_signal_chunks,
    open_db_readonly,
    parse_ts,
    utc_now,
)


def run(*, latest_candidates: int, windows: list[int], output_dir: Path) -> DiagnosticReport:
    report = DiagnosticReport("inspect_duplicate_candidate_events", output_dir)
    if not DB_PATH.is_file():
        report.add_limitation(f"Database missing: {DB_PATH}")
        report.set_status("FAIL")
        return report

    conn = open_db_readonly()
    rows: list[dict] = []
    try:
        for chunk in iter_candidate_signal_chunks(conn, limit=latest_candidates, chunk_size=CHUNK_SIZE):
            rows.extend(chunk)
    finally:
        conn.close()

    pair_counts = Counter(str(r.get("pair_address") or "") for r in rows)
    symbol_counts = Counter(str(r.get("symbol") or "") for r in rows)
    total = len(rows)
    unique_pairs = len([p for p in pair_counts if p])
    dup_ratio = round(1 - unique_pairs / total, 4) if total else 0.0

    window_stats = {}
    for w in windows:
        repeats = 0
        by_pair: dict[str, list] = {}
        for r in rows:
            ts = parse_ts(str(r.get("timestamp") or ""))
            if ts is None:
                continue
            pa = str(r.get("pair_address") or "")
            by_pair.setdefault(pa, []).append(ts)
        for _pa, times in by_pair.items():
            times.sort()
            for i in range(1, len(times)):
                if (times[i] - times[i - 1]).total_seconds() <= w * 60:
                    repeats += 1
                    break
        window_stats[f"{w}m"] = repeats

    top_pairs = [
        {"pair_address": p, "count": c, "symbol": next((r.get("symbol") for r in rows if r.get("pair_address") == p), "")}
        for p, c in pair_counts.most_common(100)
        if p
    ]
    top1_share = (pair_counts.most_common(1)[0][1] / total) if total and pair_counts else 0
    dominated = top1_share > 0.15

    status = "WARN" if dup_ratio > 0.5 or dominated else "PASS"
    if dup_ratio > 0.8:
        status = "FAIL"
    report.set_status(status)

    report.data.update({
        "total_candidates": total,
        "unique_pair_address_count": unique_pairs,
        "unique_symbol_count": len(symbol_counts),
        "duplicate_ratio": dup_ratio,
        "repeated_candidates_by_window": window_stats,
        "top_repeated_pairs": top_pairs[:20],
        "top_pair_dominance_share": round(top1_share, 4),
        "top_candidates_dominated_by_one_pair": dominated,
        "recommendation": (
            "Group candidates by (pair_address, floor(timestamp/window)) before RF ranking and paper execution."
            if dup_ratio > 0.3
            else "Duplicate ratio acceptable for event-level analysis."
        ),
    })
    report.write_json("duplicate_candidate_events.json")
    report.write_md([
        f"- Total candidates: {total}",
        f"- Unique pairs: {unique_pairs} (dup ratio {dup_ratio})",
        f"- Window repeats: {window_stats}",
        f"- Recommendation: {report.data['recommendation']}",
    ], "duplicate_candidate_events.md")
    report.write_csv(top_pairs, "duplicate_pairs.csv")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest-candidates", type=int, default=10000)
    parser.add_argument("--windows", default="5,15,60")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]
    report = run(latest_candidates=args.latest_candidates, windows=windows, output_dir=args.output_dir)
    print(f"Status: {report.data['status']}")
    return 0 if report.data["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
