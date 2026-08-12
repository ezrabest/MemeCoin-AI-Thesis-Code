#!/usr/bin/env python3
"""One-off dataset audit for Qwen readiness."""
from __future__ import annotations

import csv
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.database import DB_PATH, init_pool

DATA_DIR = ROOT / "data"
DECISIONS_CSV = DATA_DIR / "llm_decisions_log.csv"


def parse_ts(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def hours_span(timestamps: list[str]) -> float:
    if len(timestamps) < 2:
        return 0.0
    dts = sorted(parse_ts(t) for t in timestamps if t)
    return max((dts[-1] - dts[0]).total_seconds() / 3600.0, 1e-9)


def main() -> None:
    init_pool()
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row

    out: dict = {}

    # --- 1. CSV vs SQLite gemini ---
    csv_rows = 0
    csv_first = csv_last = None
    if DECISIONS_CSV.exists():
        with open(DECISIONS_CSV, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        csv_rows = len(rows)
        if rows:
            csv_first = rows[0].get("timestamp")
            csv_last = rows[-1].get("timestamp")

    gemini_rows = conn.execute("SELECT COUNT(*) c FROM gemini_decisions").fetchone()["c"]
    gemini_ts = conn.execute(
        "SELECT MIN(timestamp) mn, MAX(timestamp) mx FROM gemini_decisions"
    ).fetchone()
    gemini_first_in_sqlite = conn.execute(
        "SELECT timestamp FROM gemini_decisions ORDER BY id ASC LIMIT 1"
    ).fetchone()

    out["gemini_csv_vs_sqlite"] = {
        "csv_rows": csv_rows,
        "sqlite_rows": gemini_rows,
        "csv_first": csv_first,
        "csv_last": csv_last,
        "sqlite_first": gemini_ts["mn"],
        "sqlite_last": gemini_ts["mx"],
        "sqlite_starts_after_csv_gap": (
            gemini_first_in_sqlite
            and csv_last
            and parse_ts(gemini_first_in_sqlite["timestamp"]) > parse_ts(csv_last)
            if gemini_first_in_sqlite and csv_last
            else None
        ),
    }

    # --- 2. Example rows ---
    examples = {}
    for table, sql in [
        ("market_snapshots", "SELECT * FROM market_snapshots ORDER BY id DESC LIMIT 1"),
        ("signals", "SELECT * FROM signals ORDER BY id DESC LIMIT 1"),
        ("whale_alerts", "SELECT * FROM whale_alerts ORDER BY id DESC LIMIT 1"),
        ("gemini_decisions", "SELECT * FROM gemini_decisions ORDER BY id DESC LIMIT 1"),
    ]:
        row = conn.execute(sql).fetchone()
        examples[table] = dict(row) if row else None
    out["examples"] = examples

    # --- 3. Gemini data completeness in SQLite ---
    gemini_full = conn.execute("""
        SELECT
            COUNT(*) total,
            SUM(CASE WHEN input_context_json IS NOT NULL AND input_context_json != '' AND input_context_json != '{}' THEN 1 ELSE 0 END) with_context,
            SUM(CASE WHEN gemini_response_json IS NOT NULL AND gemini_response_json != '' THEN 1 ELSE 0 END) with_response,
            SUM(CASE WHEN rationale IS NOT NULL AND rationale != '' THEN 1 ELSE 0 END) with_rationale,
            SUM(CASE WHEN prompt_summary IS NOT NULL AND prompt_summary != '' THEN 1 ELSE 0 END) with_prompt_summary
        FROM gemini_decisions
    """).fetchone()
    gemini_raw = conn.execute(
        "SELECT COUNT(*) c FROM raw_provider_payloads WHERE provider='gemini'"
    ).fetchone()["c"]
    out["gemini_completeness"] = {
        "sqlite": dict(gemini_full),
        "raw_gemini_payloads": gemini_raw,
    }

    # CSV fields present
    if DECISIONS_CSV.exists():
        with open(DECISIONS_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames or []
            sample = next(iter(csv.DictReader(open(DECISIONS_CSV, encoding="utf-8"))), {})
        out["csv_decision_fields"] = fields
        out["csv_sample_keys"] = list(sample.keys()) if sample else []

    # --- 4. Collection rates ---
    rates = {}
    for table in ("market_snapshots", "signals", "raw_provider_payloads", "gemini_decisions"):
        ts_list = [
            r["timestamp"]
            for r in conn.execute(f"SELECT timestamp FROM {table} ORDER BY timestamp").fetchall()
        ]
        span_h = hours_span(ts_list)
        rates[table] = {
            "rows": len(ts_list),
            "span_hours": round(span_h, 2),
            "per_hour": round(len(ts_list) / span_h, 2) if span_h > 0 else 0,
            "first": ts_list[0] if ts_list else None,
            "last": ts_list[-1] if ts_list else None,
        }
    out["collection_rates"] = rates

    # --- 5. Duplicates / missing ---
    dup_snapshots = conn.execute("""
        SELECT coin_id, timestamp, COUNT(*) cnt
        FROM market_snapshots
        GROUP BY coin_id, timestamp
        HAVING cnt > 1
    """).fetchall()
    dup_raw_hash = conn.execute("""
        SELECT payload_hash, COUNT(*) cnt
        FROM raw_provider_payloads
        GROUP BY payload_hash
        HAVING cnt > 1
    """).fetchall()
    signals_null_coin = conn.execute(
        "SELECT COUNT(*) c FROM signals WHERE coin_id IS NULL"
    ).fetchone()["c"]
    signals_dup = conn.execute("""
        SELECT coin_id, timestamp, signal_type, COUNT(*) cnt
        FROM signals
        GROUP BY coin_id, timestamp, signal_type
        HAVING cnt > 1
    """).fetchall()
    snapshots_null_price = conn.execute(
        "SELECT COUNT(*) c FROM market_snapshots WHERE price IS NULL OR price = 0"
    ).fetchone()["c"]

    out["duplicates_missing"] = {
        "snapshot_coin_ts_duplicates": len(dup_snapshots),
        "snapshot_null_price": snapshots_null_price,
        "raw_hash_duplicates": len(dup_raw_hash),
        "signals_null_coin_id": signals_null_coin,
        "signals_coin_ts_type_duplicates": len(signals_dup),
    }

    # Raw dedup effectiveness: inserts skipped = would-be dupes not stored
    out["raw_by_provider"] = {
        r["provider"]: r["c"]
        for r in conn.execute(
            "SELECT provider, COUNT(*) c FROM raw_provider_payloads GROUP BY provider"
        ).fetchall()
    }

    # --- 6. Future return labelability ---
    snaps = conn.execute(
        "SELECT coin_id, timestamp, price FROM market_snapshots WHERE price IS NOT NULL AND price > 0 ORDER BY coin_id, timestamp"
    ).fetchall()
    by_coin: dict[int, list[tuple[datetime, float]]] = {}
    for s in snaps:
        by_coin.setdefault(s["coin_id"], []).append((parse_ts(s["timestamp"]), float(s["price"])))

    horizons = {"15m": 15, "1h": 60, "4h": 240}
    labelable = {h: 0 for h in horizons}
    total_snaps = len(snaps)

    for _coin_id, series in by_coin.items():
        for i, (ts, price) in enumerate(series):
            for h_name, h_min in horizons.items():
                target = ts.timestamp() + h_min * 60
                for j in range(i + 1, len(series)):
                    ts_j, price_j = series[j]
                    if ts_j.timestamp() >= target:
                        labelable[h_name] += 1
                        break

    out["future_return_labelability"] = {
        "total_snapshots_with_price": total_snaps,
        **{
            f"pct_{k}": round(100.0 * labelable[k] / total_snaps, 2) if total_snaps else 0
            for k in horizons
        },
        **{f"count_{k}": labelable[k] for k in horizons},
    }

    # Per-coin snapshot density
    snap_per_coin = conn.execute("""
        SELECT coin_id, COUNT(*) cnt, MIN(timestamp) first_ts, MAX(timestamp) last_ts
        FROM market_snapshots GROUP BY coin_id ORDER BY cnt DESC LIMIT 5
    """).fetchall()
    out["top_coins_by_snapshots"] = [dict(r) for r in snap_per_coin]

    # Collection gap: last snapshot vs now
    last_snap = conn.execute("SELECT MAX(timestamp) mx FROM market_snapshots").fetchone()["mx"]
    if last_snap:
        gap_days = (datetime.now(timezone.utc) - parse_ts(last_snap)).total_seconds() / 86400
        out["collection_staleness_days"] = round(gap_days, 2)

    print(json.dumps(out, indent=2, default=str))
    conn.close()


if __name__ == "__main__":
    main()
