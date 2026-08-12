#!/usr/bin/env python3
"""Diagnostic 9 — paper trade source-of-truth comparison."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.diagnostics._common import (
    DATA_DIR,
    DB_PATH,
    DiagnosticReport,
    fetch_json_url,
    open_db_readonly,
    parse_ts,
    safe_float,
)


def _sqlite_paper_stats() -> dict:
    if not DB_PATH.is_file():
        return {"exists": False}
    conn = open_db_readonly()
    try:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(paper_trades)").fetchall()]
        cnt = int(conn.execute("SELECT COUNT(*) AS c FROM paper_trades").fetchone()["c"])
        latest = conn.execute("SELECT MAX(timestamp) AS t FROM paper_trades").fetchone()["t"]
        buys = int(conn.execute("SELECT COUNT(*) AS c FROM paper_trades WHERE lower(side)='buy'").fetchone()["c"])
        sells = int(conn.execute("SELECT COUNT(*) AS c FROM paper_trades WHERE lower(side)='sell'").fetchone()["c"])
        open_pos = int(conn.execute("SELECT COUNT(*) AS c FROM paper_trades WHERE lower(status)='open'").fetchone()["c"])
        fees = conn.execute("SELECT SUM(fee) AS s FROM paper_trades").fetchone()["s"]
        pnl = conn.execute("SELECT SUM(pnl) AS s FROM paper_trades WHERE pnl IS NOT NULL").fetchone()["s"]
    finally:
        conn.close()
    return {
        "exists": True,
        "row_count": cnt,
        "latest_timestamp": latest,
        "total_buys": buys,
        "total_sells": sells,
        "open_positions": open_pos,
        "total_fees": safe_float(fees),
        "realized_pnl": safe_float(pnl),
        "schema_columns": cols,
    }


def _paper_state_stats() -> dict:
    path = DATA_DIR / "paper_state.json"
    if not path.is_file():
        return {"exists": False, "path": str(path)}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        return {"exists": True, "read_error": str(exc)}
    wallet = data.get("wallet") or data
    positions = data.get("positions") or wallet.get("positions") or []
    open_pos = [p for p in positions if str(p.get("status", "")).upper() == "OPEN"]
    return {
        "exists": True,
        "path": str(path),
        "cash": wallet.get("cash") or wallet.get("balance_usd"),
        "equity": wallet.get("equity") or wallet.get("total_equity"),
        "open_positions": len(open_pos),
        "starting_capital": wallet.get("starting_capital"),
        "positions_count": len(positions),
        "latest_timestamp": data.get("updated_at") or data.get("timestamp"),
    }


def _csv_log_stats() -> dict:
    path = DATA_DIR / "paper_trades_log.csv"
    if not path.is_file():
        return {"exists": False, "path": str(path)}
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)
    buys = sum(1 for r in rows if str(r.get("side", "")).lower() == "buy")
    sells = sum(1 for r in rows if str(r.get("side", "")).lower() == "sell")
    latest = max((r.get("timestamp") or "" for r in rows), default=None)
    return {
        "exists": True,
        "path": str(path),
        "row_count": len(rows),
        "latest_timestamp": latest,
        "total_buys": buys,
        "total_sells": sells,
        "schema_columns": cols,
    }


def run(*, output_dir: Path, api_base: str = "http://127.0.0.1:8000") -> DiagnosticReport:
    report = DiagnosticReport("inspect_paper_trade_source_of_truth", output_dir)
    sqlite_stats = _sqlite_paper_stats()
    state_stats = _paper_state_stats()
    csv_stats = _csv_log_stats()
    api_trades = fetch_json_url(f"{api_base}/api/trades?limit=5000")
    api_summary = fetch_json_url(f"{api_base}/api/analytics/summary")

    sources = {
        "sqlite_paper_trades": sqlite_stats,
        "paper_state_json": state_stats,
        "paper_trades_log_csv": csv_stats,
        "api_trades": {
            "exists": api_trades is not None,
            "row_count": len(api_trades) if isinstance(api_trades, list) else None,
        },
        "api_analytics_summary": {
            "exists": api_summary is not None,
            "payload_keys": list(api_summary.keys()) if isinstance(api_summary, dict) else [],
        },
    }

    comparisons = []
    if sqlite_stats.get("exists") and csv_stats.get("exists"):
        comparisons.append({
            "pair": "sqlite_vs_csv_row_count",
            "sqlite": sqlite_stats.get("row_count"),
            "csv": csv_stats.get("row_count"),
            "match": sqlite_stats.get("row_count") == csv_stats.get("row_count"),
        })
    if sqlite_stats.get("exists") and state_stats.get("exists"):
        comparisons.append({
            "pair": "sqlite_open_vs_state_open",
            "sqlite": sqlite_stats.get("open_positions"),
            "state": state_stats.get("open_positions"),
            "match": sqlite_stats.get("open_positions") == state_stats.get("open_positions"),
        })

    status = "PASS"
    only_one = sum(1 for s in (sqlite_stats, csv_stats, state_stats) if s.get("exists")) == 1
    if only_one and (sqlite_stats.get("row_count") or csv_stats.get("row_count")):
        status = "WARN"
        report.add_limitation("Historical paper data exists in only one source")
    if api_summary and sqlite_stats.get("exists"):
        wallet = (api_summary or {}).get("wallet") or {}
        if wallet.get("open_positions_count") is not None and state_stats.get("open_positions") is not None:
            if wallet.get("open_positions_count") != state_stats.get("open_positions"):
                status = "FAIL"
                report.add_limitation("Dashboard open position count differs from paper_state.json")

    report.set_status(status)
    report.data["sources"] = sources
    report.data["comparisons"] = comparisons
    report.write_json("paper_trade_source_of_truth.json")
    comparison_lines = (
        [f"- {c}" for c in comparisons]
        if comparisons
        else ["- No pairwise comparisons available"]
    )
    report.write_md([
        "## Sources",
        *[f"- **{k}**: exists={v.get('exists')}" for k, v in sources.items()],
        "",
        "## Comparisons",
        *comparison_lines,
    ], "paper_trade_source_of_truth.md")
    report.write_csv(
        [{"source": k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))}} for k, v in sources.items()],
        "source_comparison.csv",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    report = run(output_dir=args.output_dir, api_base=args.api_base)
    print(f"Status: {report.data['status']}")
    return 0 if report.data["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
