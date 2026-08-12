"""AE13I idempotent migration: repair malformed legacy RISK_GUARD_BLOCK rows.

Some older paper_trades_log.csv rows were written before the current
TRADE_CSV_FIELDS header existed, using a different column order. When those
rows are re-read against today's header, the values land under the wrong
column names (column-shift corruption): the coin's symbol lands under
`side`, the buy/sell side lands under `quantity`, the chain lands under
`fill_price`, and the literal marker string `RISK_GUARD_BLOCK` lands under
`coin_id` instead of `event_type`.

This script:
1. Backs up paper_trades_log.csv (and paper_state.json, if present) to
   `data/backups/pre_ae13i_risk_block_repair_<timestamp>/` before touching
   anything.
2. Scans the CSV for malformed RISK_GUARD_BLOCK rows (coin_id ==
   "RISK_GUARD_BLOCK", or any other row where event_type ==
   "RISK_GUARD_BLOCK" but rejection_code is empty -- the structural
   signature of a pre-AE13G/pre-schema row).
3. Repairs each malformed row into a RejectedTradeAttempt-like schema, or
   quarantines it with legacy_malformed=true, event_type=RISK_GUARD_BLOCK,
   preserving the full original row as legacy_raw_record (JSON).
4. Skips rows already repaired (repaired_by == this script's name and
   schema_version already at/above ae13g_rejected_trade_attempt_v1) so
   running the script twice never double-repairs or corrupts data.
5. Writes a JSON repair report describing what was found/changed.

Paper/demo only. This script never touches a live wallet and never
executes trades -- it only rewrites historical CSV rows.
"""
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ae13b_product.rejected_attempt import RejectedTradeAttempt  # noqa: E402
from app.execution.paper import TRADE_CSV_FIELDS  # noqa: E402

REPAIRED_BY = "repair_risk_block_schema.py"
SCHEMA_VERSION = "ae13g_rejected_trade_attempt_v1"
LEGACY_REJECTION_CODE = "LEGACY_UNSTRUCTURED_RISK_GUARD_BLOCK"

#: Columns appended to the CSV header (idempotent -- only added if missing)
#: to carry AE13I repair/quarantine provenance without disturbing existing
#: reader code that keys off TRADE_CSV_FIELDS.
REPAIR_EXTRA_FIELDS = [
    "legacy_malformed",
    "legacy_raw_record",
    "repaired_by",
    "repaired_at_utc",
    "schema_version",
]

_VALID_SIDES = {"buy", "sell", "BUY", "SELL"}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_csv_path() -> Path:
    return ROOT / "data" / "paper_trades_log.csv"


def _default_state_path() -> Path:
    return ROOT / "data" / "paper_state.json"


def _is_already_repaired(row: dict[str, Any]) -> bool:
    if str(row.get("repaired_by") or "") != REPAIRED_BY:
        return False
    version = str(row.get("schema_version") or "")
    # Simple lexical/prefix comparison is sufficient while there is a single
    # schema_version constant; guards against accidental double-repair if
    # this script's SCHEMA_VERSION is ever bumped in the future.
    return version == SCHEMA_VERSION or version.startswith("ae13")


def _is_malformed_risk_guard_block_row(row: dict[str, Any]) -> bool:
    if _is_already_repaired(row):
        return False
    coin_id = str(row.get("coin_id") or "")
    if coin_id == "RISK_GUARD_BLOCK":
        return True
    event_type = str(row.get("event_type") or "")
    reason_code = str(row.get("reason_code") or "")
    rejection_code = str(row.get("rejection_code") or "")
    is_block_marker = event_type == "RISK_GUARD_BLOCK" or reason_code == "RISK_GUARD_BLOCK"
    return bool(is_block_marker and not rejection_code)


def _repair_row(row: dict[str, Any]) -> dict[str, Any]:
    """Best-effort remap of a column-shifted legacy row into the current
    RejectedTradeAttempt-like schema, preserving the original as JSON.
    """
    legacy_raw_record = json.dumps(row, sort_keys=True)

    timestamp = row.get("timestamp") or _utc_now_iso()

    # Column-shift pattern observed in production data: symbol landed under
    # `side`, the buy/sell side landed under `quantity`, and chain landed
    # under `fill_price`. Fall back to the canonical column if the shifted
    # value looks empty/unusable, so partially-shifted rows are not lost.
    shifted_symbol = str(row.get("side") or "").strip()
    shifted_side_raw = str(row.get("quantity") or "").strip()
    shifted_chain = str(row.get("fill_price") or "").strip()

    symbol = shifted_symbol or (str(row.get("symbol") or "").strip() or None)
    side = shifted_side_raw.upper() if shifted_side_raw in _VALID_SIDES else "BUY"
    chain = shifted_chain or (str(row.get("chain") or "").strip() or None)

    notional_hint = row.get("notional_usd") or row.get("notional_requested")
    try:
        notional_requested = float(notional_hint) if notional_hint not in (None, "") else 0.0
    except (TypeError, ValueError):
        notional_requested = 0.0

    attempt = RejectedTradeAttempt(
        event_type="RISK_GUARD_BLOCK",
        timestamp=str(timestamp),
        symbol=symbol,
        pair=None,
        side=side,
        chain=chain,
        rejection_code=LEGACY_REJECTION_CODE,
        rejection_reason=(
            "Legacy pre-AE13G column-shifted RISK_GUARD_BLOCK row; original "
            "structured rejection reason was not captured at write time. "
            "Repaired/quarantined by AE13I schema migration."
        ),
        rejection_reasons=["legacy_row_missing_structured_rejection_code"],
        blocking_guards=["legacy_unstructured_risk_guard_block"],
        risk_guard_passed=False,
        notional_requested=notional_requested,
    )
    repaired = attempt.to_dict()
    repaired["legacy_malformed"] = True
    repaired["legacy_raw_record"] = legacy_raw_record
    repaired["repaired_by"] = REPAIRED_BY
    repaired["repaired_at_utc"] = _utc_now_iso()
    repaired["schema_version"] = SCHEMA_VERSION
    return repaired


def _read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(r) for r in reader]
    return fieldnames, rows


def _write_csv_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    tmp_path.replace(path)


def backup_files(*, csv_path: Path, state_path: Path, backups_root: Path, ts: str) -> dict[str, Any]:
    backup_dir = backups_root / f"pre_ae13i_risk_block_repair_{ts}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backed_up: list[str] = []
    if csv_path.exists():
        shutil.copy2(csv_path, backup_dir / csv_path.name)
        backed_up.append(str(backup_dir / csv_path.name))
    if state_path.exists():
        shutil.copy2(state_path, backup_dir / state_path.name)
        backed_up.append(str(backup_dir / state_path.name))
    return {"backup_dir": str(backup_dir), "backed_up_files": backed_up}


def repair_csv(
    csv_path: Path,
    *,
    dry_run: bool = False,
) -> dict[str, Any]:
    if not csv_path.exists():
        return {
            "csv_path": str(csv_path),
            "csv_found": False,
            "rows_scanned": 0,
            "rows_malformed_found": 0,
            "rows_repaired": 0,
            "rows_already_repaired": 0,
            "rows_unchanged": 0,
            "dry_run": dry_run,
            "wrote_changes": False,
        }

    fieldnames, rows = _read_csv_rows(csv_path)

    already_repaired = 0
    malformed = 0
    repaired_rows: list[dict[str, Any]] = []
    for row in rows:
        if _is_already_repaired(row):
            already_repaired += 1
            repaired_rows.append(row)
            continue
        if _is_malformed_risk_guard_block_row(row):
            malformed += 1
            repaired_rows.append(_repair_row(row))
        else:
            repaired_rows.append(row)

    out_fields = list(fieldnames)
    for extra in REPAIR_EXTRA_FIELDS:
        if extra not in out_fields:
            out_fields.append(extra)
    for extra in TRADE_CSV_FIELDS:
        if extra not in out_fields:
            out_fields.append(extra)

    wrote_changes = False
    if malformed > 0 and not dry_run:
        _write_csv_rows(csv_path, out_fields, repaired_rows)
        wrote_changes = True

    return {
        "csv_path": str(csv_path),
        "csv_found": True,
        "rows_scanned": len(rows),
        "rows_malformed_found": malformed,
        "rows_repaired": malformed if wrote_changes else (0 if dry_run else malformed),
        "rows_already_repaired": already_repaired,
        "rows_unchanged": len(rows) - malformed - already_repaired,
        "dry_run": dry_run,
        "wrote_changes": wrote_changes,
        "header_migrated": out_fields != fieldnames,
        "output_fields": out_fields,
    }


def run(
    *,
    csv_path: Path | None = None,
    state_path: Path | None = None,
    backups_root: Path | None = None,
    reports_root: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    csv_path = csv_path or _default_csv_path()
    state_path = state_path or _default_state_path()
    backups_root = backups_root or (ROOT / "data" / "backups")
    reports_root = reports_root or (ROOT / "data" / "reports")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Pre-scan (read-only) to decide whether a backup is even needed --
    # running the script again after everything is already repaired must
    # not create a redundant backup or touch the file at all.
    pre_scan_malformed = 0
    if csv_path.exists():
        _, rows = _read_csv_rows(csv_path)
        pre_scan_malformed = sum(1 for r in rows if _is_malformed_risk_guard_block_row(r))

    backup_info: dict[str, Any] = {"backup_dir": None, "backed_up_files": []}
    if pre_scan_malformed > 0 and not dry_run:
        backup_info = backup_files(
            csv_path=csv_path, state_path=state_path, backups_root=backups_root, ts=ts,
        )

    csv_result = repair_csv(csv_path, dry_run=dry_run)

    report = {
        "script": REPAIRED_BY,
        "schema_version": SCHEMA_VERSION,
        "ran_at_utc": _utc_now_iso(),
        "dry_run": dry_run,
        "pre_scan_malformed_rows": pre_scan_malformed,
        "backup": backup_info,
        "csv_repair": csv_result,
        "idempotent": True,
        "notes": [
            "Rows already carrying repaired_by=repair_risk_block_schema.py and "
            "schema_version=ae13g_rejected_trade_attempt_v1 are always skipped, "
            "so re-running this script is a safe no-op once fully repaired.",
            "Repaired rows are quarantined with legacy_malformed=true and the "
            "full original row preserved verbatim as legacy_raw_record (JSON) "
            "for audit purposes -- no original data is discarded.",
            "Paper/demo only. This script never touches a live wallet.",
        ],
    }

    reports_root.mkdir(parents=True, exist_ok=True)
    report_path = reports_root / f"repair_risk_block_schema_report_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv-path", type=str, default=None, help="Override paper_trades_log.csv path")
    parser.add_argument("--state-path", type=str, default=None, help="Override paper_state.json path")
    parser.add_argument("--backups-root", type=str, default=None, help="Override backups root directory")
    parser.add_argument("--reports-root", type=str, default=None, help="Override reports output directory")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report only; never write the CSV or create a backup.",
    )
    args = parser.parse_args(argv)

    report = run(
        csv_path=Path(args.csv_path) if args.csv_path else None,
        state_path=Path(args.state_path) if args.state_path else None,
        backups_root=Path(args.backups_root) if args.backups_root else None,
        reports_root=Path(args.reports_root) if args.reports_root else None,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
