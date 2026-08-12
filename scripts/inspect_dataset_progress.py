#!/usr/bin/env python3
"""Report SQLite + CSV/JSON collection progress for dataset readiness."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.database import DB_PATH, get_storage_stats, init_pool

DATA_DIR = ROOT / "data"


def _count_csv_rows(path: Path) -> int:
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with open(path, encoding="utf-8") as f:
        return max(0, sum(1 for _ in csv.DictReader(f)))


def _file_meta(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "rows": 0, "size_bytes": 0}
    return {
        "exists": True,
        "rows": _count_csv_rows(path) if path.suffix == ".csv" else None,
        "size_bytes": path.stat().st_size,
    }


def main() -> None:
    init_pool()
    stats = get_storage_stats()

    csv_files = {
        "paper_trades_log.csv": DATA_DIR / "paper_trades_log.csv",
        "whale_trades_log.csv": DATA_DIR / "whale_trades_log.csv",
        "llm_decisions_log.csv": DATA_DIR / "llm_decisions_log.csv",
    }
    json_files = {
        "paper_state.json": DATA_DIR / "paper_state.json",
        "settings.json": DATA_DIR / "settings.json",
        "watchlist.json": DATA_DIR / "watchlist.json",
        "cluster_registry.json": DATA_DIR / "cluster_registry.json",
        "token_transparency_log.json": DATA_DIR / "token_transparency_log.json",
    }

    report = {
        "sqlite_path": str(DB_PATH),
        "sqlite_tables": stats,
        "csv_artifacts": {name: _file_meta(path) for name, path in csv_files.items()},
        "json_artifacts": {name: _file_meta(path) for name, path in json_files.items()},
        "totals": {
            "sqlite_rows": sum(table.get("rows", 0) for table in stats.values()),
            "csv_trade_rows": _count_csv_rows(csv_files["paper_trades_log.csv"]),
            "csv_whale_rows": _count_csv_rows(csv_files["whale_trades_log.csv"]),
            "csv_decision_rows": _count_csv_rows(csv_files["llm_decisions_log.csv"]),
        },
    }

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
