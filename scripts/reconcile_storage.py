#!/usr/bin/env python3
"""Storage reconciliation — SQLite is runtime source of truth; CSV is audit export."""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.database import DATA_DIR, DB_PATH, backup_database, get_db, insert_trade  # noqa: E402
from app.observability.audit_io import utc_timestamp_slug, write_json_report_atomic  # noqa: E402
from app.observability.effective_settings import get_effective_settings  # noqa: E402

PAPER_STATE_PATH = DATA_DIR / "paper_state.json"
PAPER_CSV_PATH = DATA_DIR / "paper_trades_log.csv"
TRAINING_DIR = DATA_DIR / "training"
STALE_MARKER = TRAINING_DIR / ".parquet_stale_marker"

CSV_EXPORT_COLUMNS = [
    "timestamp", "symbol", "side", "fill_price", "quantity", "notional_usd",
    "total_fees", "swap_fee", "realized_pnl", "reason_code", "decision_ref_id",
    "position_id", "chain", "cluster_label", "net_roi_pct", "coin_id", "pair_address",
]


def _count_csv_rows(path: Path) -> tuple[int, int]:
    if not path.exists() or path.stat().st_size == 0:
        return 0, 0
    total = 0
    valid = 0
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if row.get("symbol") and (row.get("pair_address") or row.get("coin_id")):
                valid += 1
    return total, valid


def _sqlite_paper_counts() -> tuple[int, int]:
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) AS c FROM paper_trades").fetchone()["c"]
        valid = conn.execute(
            "SELECT COUNT(*) AS c FROM paper_trades WHERE symbol IS NOT NULL AND coin_id IS NOT NULL"
        ).fetchone()["c"]
    return int(total), int(valid)


def _paper_state_consistency() -> dict:
    result: dict = {"exists": PAPER_STATE_PATH.exists(), "consistent": None, "notes": []}
    if not PAPER_STATE_PATH.exists():
        result["notes"].append("paper_state.json missing")
        return result
    try:
        with open(PAPER_STATE_PATH, encoding="utf-8") as f:
            state = json.load(f)
        result["open_positions_count"] = len(state.get("open_positions") or [])
        result["cash_usd"] = float(state.get("cash_usd") or 0)
        result["starting_capital"] = float(state.get("starting_capital") or 10000)
        result["consistent"] = True
    except (json.JSONDecodeError, OSError) as exc:
        result["consistent"] = False
        result["notes"].append(f"paper_state parse error: {exc}")
    return result


def _parquet_staleness() -> dict:
    result: dict = {"training_dir_exists": TRAINING_DIR.exists(), "files": [], "must_rebuild": False}
    if not DB_PATH.exists():
        result["notes"] = "SQLite DB missing"
        return result
    db_mtime = DB_PATH.stat().st_mtime
    if not TRAINING_DIR.exists():
        result["must_rebuild"] = True
        return result
    parquet_files = list(TRAINING_DIR.glob("**/*.parquet"))
    if not parquet_files:
        result["must_rebuild"] = True
        return result
    stale = []
    for pf in parquet_files:
        is_stale = pf.stat().st_mtime < db_mtime
        stale.append({
            "path": str(pf.relative_to(DATA_DIR)),
            "stale_vs_sqlite": is_stale,
        })
        if is_stale:
            result["must_rebuild"] = True
    result["files"] = stale
    result["sqlite_mtime"] = datetime.fromtimestamp(db_mtime, tz=timezone.utc).isoformat()
    return result


def run_check() -> dict:
    eff = get_effective_settings()
    sqlite_total, sqlite_valid = _sqlite_paper_counts()
    csv_total, csv_valid = _count_csv_rows(PAPER_CSV_PATH)
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "settings_hash": eff.settings_hash,
        "phase": 2,
        "sqlite_source_of_truth": True,
        "paper_trades": {
            "sqlite_row_count": sqlite_total,
            "sqlite_valid_row_count": sqlite_valid,
            "csv_row_count": csv_total,
            "csv_valid_row_count": csv_valid,
            "count_mismatch": sqlite_total != csv_total,
            "valid_mismatch": sqlite_valid != csv_valid,
        },
        "paper_state": _paper_state_consistency(),
        "parquet": _parquet_staleness(),
        "csv_is_audit_export_only": True,
        "default_fix_direction": "sqlite_to_csv_export",
        "csv_import_requires": "--import-csv-to-sqlite --yes",
        "stale_marker_exists": STALE_MARKER.exists(),
    }
    ts = utc_timestamp_slug()
    path = write_json_report_atomic(f"storage_reconcile_{ts}.json", report)
    report["output_path"] = str(path)
    return report


def _backup_files() -> list[str]:
    backups: list[str] = []
    ts = utc_timestamp_slug()
    backup_dir = DATA_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    db_backup = backup_database()
    if db_backup:
        backups.append(str(db_backup))
    for src in (PAPER_CSV_PATH, PAPER_STATE_PATH):
        if src.exists():
            dest = backup_dir / f"{src.stem}_{ts}{src.suffix}"
            shutil.copy2(src, dest)
            backups.append(str(dest))
    return backups


def _export_sqlite_to_csv() -> dict:
    """Regenerate CSV audit export from SQLite (default fix direction)."""
    exported = 0
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT t.timestamp, t.symbol, t.side, t.price AS fill_price, t.amount AS quantity,
                   t.value AS notional_usd, t.fee AS total_fees, t.slippage AS swap_fee,
                   t.pnl AS realized_pnl, t.reason AS reason_code, t.decision_ref_id,
                   t.position_id, t.chain, t.cluster_label, t.net_roi_pct, t.coin_id,
                   c.pair_address
            FROM paper_trades t
            LEFT JOIN coins c ON c.id = t.coin_id
            ORDER BY t.timestamp
            """
        ).fetchall()

    PAPER_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PAPER_CSV_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_EXPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            record = dict(row)
            if not record.get("symbol"):
                continue
            writer.writerow({k: record.get(k) for k in CSV_EXPORT_COLUMNS})
            exported += 1

    return {"exported": exported, "csv_path": str(PAPER_CSV_PATH)}


def _import_csv_to_sqlite() -> dict:
    """Explicit CSV → SQLite import with validation — only via --import-csv-to-sqlite --yes."""
    if not PAPER_CSV_PATH.exists():
        return {"imported": 0, "skipped": 0, "ambiguous": [], "note": "no CSV"}

    imported = 0
    skipped = 0
    ambiguous: list[str] = []

    with get_db() as conn:
        existing_ids = {
            int(r["id"]) for r in conn.execute("SELECT id FROM paper_trades").fetchall()
        }
        existing_keys = {
            (r["timestamp"], r["symbol"], r["side"])
            for r in conn.execute("SELECT timestamp, symbol, side FROM paper_trades").fetchall()
        }
        valid_coin_ids = {
            int(r["id"]) for r in conn.execute("SELECT id FROM coins").fetchall()
        }

    with open(PAPER_CSV_PATH, encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            key = (row.get("timestamp"), row.get("symbol"), row.get("side"))
            if key in existing_keys:
                skipped += 1
                continue
            if not row.get("symbol") or not (row.get("pair_address") or row.get("coin_id")):
                ambiguous.append(f"line {i}: missing symbol/pair/coin_id")
                skipped += 1
                continue
            try:
                fill_price = float(row["fill_price"]) if row.get("fill_price") else None
                if fill_price is None or fill_price <= 0:
                    ambiguous.append(f"line {i}: invalid fill_price")
                    skipped += 1
                    continue
            except (ValueError, TypeError):
                ambiguous.append(f"line {i}: invalid fill_price")
                skipped += 1
                continue

            try:
                coin_id = int(row["coin_id"]) if row.get("coin_id") else None
                if coin_id is not None and coin_id not in valid_coin_ids:
                    ambiguous.append(f"line {i}: coin_id {coin_id} not in coins")
                    skipped += 1
                    continue
                trade_id = insert_trade({
                    "timestamp": row.get("timestamp"),
                    "coin_id": coin_id,
                    "symbol": row.get("symbol"),
                    "side": row.get("side"),
                    "price": row.get("fill_price"),
                    "amount": row.get("quantity"),
                    "value": row.get("notional_usd"),
                    "fee": row.get("total_fees"),
                    "slippage": row.get("swap_fee"),
                    "pnl": row.get("realized_pnl"),
                    "status": "filled",
                    "reason": row.get("reason_code"),
                    "decision_ref_id": row.get("decision_ref_id") or None,
                    "position_id": int(row["position_id"]) if row.get("position_id") else None,
                    "chain": row.get("chain"),
                    "cluster_label": row.get("cluster_label"),
                    "net_roi_pct": row.get("net_roi_pct"),
                    "source": "reconcile_csv_import",
                })
                if trade_id:
                    if int(trade_id) in existing_ids:
                        ambiguous.append(f"line {i}: duplicate trade id {trade_id}")
                        skipped += 1
                        continue
                    imported += 1
                    existing_ids.add(int(trade_id))
                    existing_keys.add(key)
                else:
                    skipped += 1
            except (ValueError, TypeError) as exc:
                ambiguous.append(f"line {i}: {exc}")
                skipped += 1
            except Exception as exc:
                ambiguous.append(f"line {i}: {exc}")
                skipped += 1

    return {"imported": imported, "skipped": skipped, "ambiguous": ambiguous}


def _mark_parquet_stale() -> None:
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)
    STALE_MARKER.write_text(
        json.dumps({
            "marked_at": datetime.now(timezone.utc).isoformat(),
            "reason": "sqlite_runtime_newer_than_parquet",
            "action_required": "run build_training_dataset.py to rebuild",
        }, indent=2),
        encoding="utf-8",
    )


def run_fix_dry_run(*, import_csv: bool = False) -> dict:
    check = run_check()
    plan: list[str] = []
    pt = check["paper_trades"]

    if import_csv and pt["csv_row_count"] > pt["sqlite_row_count"]:
        plan.append(
            f"Import up to {pt['csv_row_count'] - pt['sqlite_row_count']} CSV rows into SQLite "
            "(explicit --import-csv-to-sqlite --yes)"
        )
    elif pt["count_mismatch"]:
        plan.append(f"Regenerate CSV export from SQLite ({pt['sqlite_row_count']} rows)")
    else:
        plan.append("Regenerate CSV export from SQLite (sync audit export)")

    if check["parquet"].get("must_rebuild"):
        plan.append("Mark training Parquet stale (write .parquet_stale_marker)")

    if not plan:
        plan.append("No automatic repairs identified")
    return {
        "mode": "dry_run",
        "phase": 2,
        "modifications": "none",
        "repair_plan": plan,
        "check_summary": check,
        "import_csv_mode": import_csv,
    }


def run_fix_apply(*, import_csv: bool = False) -> dict:
    """Explicit repair with --fix --yes — creates backups first."""
    check = run_check()
    plan_result = run_fix_dry_run(import_csv=import_csv)
    backups = _backup_files()
    repairs: dict[str, Any] = {}

    if import_csv:
        repairs["csv_to_sqlite"] = _import_csv_to_sqlite()
    else:
        repairs["sqlite_to_csv"] = _export_sqlite_to_csv()
        repairs["csv_to_sqlite"] = {"imported": 0, "skipped": 0, "note": "skipped — use --import-csv-to-sqlite --yes"}

    if check["parquet"].get("must_rebuild"):
        _mark_parquet_stale()
        repairs["parquet_stale_marker"] = str(STALE_MARKER)

    post_check = run_check()
    result = {
        "mode": "apply",
        "phase": 2,
        "backups": backups,
        "repair_plan": plan_result["repair_plan"],
        "repairs": repairs,
        "pre_check": check,
        "post_check": post_check,
        "import_csv_mode": import_csv,
    }
    ts = utc_timestamp_slug()
    path = write_json_report_atomic(f"storage_reconcile_fix_{ts}.json", result)
    result["output_path"] = str(path)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconcile SQLite (source of truth) vs CSV audit export / Parquet",
        epilog=(
            "Default --fix --yes regenerates CSV from SQLite. "
            "CSV → SQLite import requires --import-csv-to-sqlite --yes."
        ),
    )
    parser.add_argument("--check", action="store_true", help="Run consistency check (read-only)")
    parser.add_argument("--fix", action="store_true", help="Repair plan (dry-run) or apply with --yes")
    parser.add_argument("--yes", action="store_true", help="Apply repairs (requires --fix)")
    parser.add_argument(
        "--import-csv-to-sqlite",
        action="store_true",
        help="Import CSV rows into SQLite (requires --fix --yes; not default)",
    )
    args = parser.parse_args()

    if args.fix and args.yes:
        result = run_fix_apply(import_csv=args.import_csv_to_sqlite)
        print("Storage reconcile FIX applied (Phase 2)")
        for item in result.get("repair_plan", []):
            print(f"  - {item}")
        print(f"Backups: {len(result.get('backups', []))}")
        if result.get("import_csv_mode"):
            print(f"CSV import: {result['repairs'].get('csv_to_sqlite')}")
        else:
            print(f"SQLite→CSV: {result['repairs'].get('sqlite_to_csv')}")
        print(f"Report: {result.get('output_path')}")
        return 0

    if args.fix:
        result = run_fix_dry_run(import_csv=args.import_csv_to_sqlite)
        print("Storage reconcile FIX (dry-run — use --fix --yes to apply)")
        for item in result["repair_plan"]:
            print(f"  - {item}")
        print(f"Check report: {result['check_summary'].get('output_path', 'n/a')}")
        return 0

    result = run_check()
    pt = result["paper_trades"]
    print(
        f"Storage reconcile: SQLite={pt['sqlite_row_count']} CSV={pt['csv_row_count']}"
    )
    print(f"Report: {result['output_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
