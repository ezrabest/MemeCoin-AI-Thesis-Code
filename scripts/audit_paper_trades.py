#!/usr/bin/env python3
"""Audit paper trade log integrity; optional --fix with backups."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.execution.paper import STARTING_CAPITAL_USD, TRADE_CSV_FIELDS, TRADES_LOG_PATH, STATE_PATH
from app.execution.paper_audit import audit_trade_rows

DEFAULT_TRADES_PATH = TRADES_LOG_PATH
DEFAULT_STATE_PATH = STATE_PATH
ARCHIVE_DIR = ROOT / "data" / "archive"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _read_trade_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_state(path: Path) -> dict | None:
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def run_audit(
    *,
    trades_path: Path,
    state_path: Path,
) -> dict:
    rows = _read_trade_rows(trades_path)
    audit = audit_trade_rows(rows)
    state = _load_state(state_path)
    state_summary = None
    if state is not None:
        state_summary = {
            "cash_usd": state.get("cash_usd"),
            "starting_capital": state.get("starting_capital"),
            "open_positions_count": len(state.get("open_positions") or []),
            "total_net_pnl": state.get("total_net_pnl"),
            "cumulative_total_fees": state.get("cumulative_total_fees"),
        }
    contaminated = audit["invalid_rows"] > 0
    if state_summary and contaminated:
        starting = float(state_summary.get("starting_capital") or STARTING_CAPITAL_USD)
        cash = float(state_summary.get("cash_usd") or 0)
        if cash > starting * 100 or cash < 0:
            state_summary["appears_contaminated"] = True
        else:
            state_summary["appears_contaminated"] = contaminated
    return {
        "trades_path": str(trades_path),
        "state_path": str(state_path) if state_path.is_file() else None,
        **audit,
        "paper_state": state_summary,
    }


def apply_fix(
    *,
    trades_path: Path,
    state_path: Path,
    dry_run: bool,
) -> dict[str, str]:
    report = run_audit(trades_path=trades_path, state_path=state_path)
    stamp = _utc_stamp()
    archive_dir = ARCHIVE_DIR
    archive_dir.mkdir(parents=True, exist_ok=True)

    trades_backup = archive_dir / f"paper_trades_log_corrupted_{stamp}.csv"
    state_backup = archive_dir / f"paper_state_corrupted_{stamp}.json"
    cleaned_trades = ROOT / "data" / "paper_trades_log_valid_only.csv"
    repaired_state_preview = ROOT / "data" / "paper_state_repaired_preview.json"

    valid_rows = report.get("valid_row_details") or []
    repaired_state = {
        "starting_capital": STARTING_CAPITAL_USD,
        "cash_usd": STARTING_CAPITAL_USD,
        "next_position_id": 1,
        "open_positions": [],
        "closed_trades": 0,
        "total_net_pnl": 0.0,
        "cumulative_swap_fees": 0.0,
        "cumulative_priority_fees": 0.0,
        "cumulative_total_fees": 0.0,
        "trading_mode": "DEMO",
        "repair_note": (
            "Reset to initial_capital because corrupted historical rows invalidated "
            "derived wallet balances. Rebuild from valid rows manually if needed."
        ),
    }

    actions = {
        "dry_run": dry_run,
        "invalid_rows_removed": report.get("invalid_rows", 0),
        "valid_rows_kept": report.get("valid_rows", 0),
        "trades_backup": str(trades_backup),
        "state_backup": str(state_backup),
        "cleaned_trades_path": str(cleaned_trades),
        "repaired_state_preview": str(repaired_state_preview),
    }

    if dry_run:
        actions["message"] = "Dry run only — no files modified."
        return actions

    if trades_path.is_file():
        shutil.copy2(trades_path, trades_backup)
    if state_path.is_file():
        shutil.copy2(state_path, state_backup)

    fieldnames = list(TRADE_CSV_FIELDS)
    if valid_rows:
        extra = [key for key in valid_rows[0].keys() if key not in fieldnames]
        fieldnames.extend(extra)

    with open(cleaned_trades, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in valid_rows:
            writer.writerow(row)

    with open(repaired_state_preview, "w", encoding="utf-8") as handle:
        json.dump(repaired_state, handle, indent=2)

    shutil.copy2(cleaned_trades, trades_path)
    with open(state_path, "w", encoding="utf-8") as handle:
        json.dump(repaired_state, handle, indent=2)

    actions["message"] = (
        f"Wrote cleaned log to {trades_path}, reset {state_path}, "
        f"backups in {archive_dir}"
    )
    return actions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit paper trade log integrity.")
    parser.add_argument("--trades-path", type=Path, default=DEFAULT_TRADES_PATH)
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--fix", action="store_true", help="Backup and repair files.")
    parser.add_argument("--dry-run", action="store_true", help="Preview --fix without writes.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.fix:
        result = apply_fix(
            trades_path=args.trades_path,
            state_path=args.state_path,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2, default=str))
        return 0

    report = run_audit(trades_path=args.trades_path, state_path=args.state_path)
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
