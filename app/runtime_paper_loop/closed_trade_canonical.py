"""AE11F canonical closed trades — SQLite-first, one row per position_id."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.runtime_paper_loop.types import utc_now_iso

CANONICAL_CLOSED_FIELDS = [
    "close_event_id",
    "position_id",
    "economic_close_key",
    "paper_order_id",
    "source_decision_id",
    "pair_address",
    "opened_at_utc",
    "closed_at_utc",
    "exit_reason",
    "entry_price",
    "exit_price",
    "quantity",
    "notional_usd",
    "cost_basis_usd",
    "entry_fee_usd",
    "exit_fee_usd",
    "total_fees_usd",
    "gross_pnl_usd",
    "net_pnl_usd",
    "net_return_pct",
    "cash_debited_usd",
    "cash_credited_usd",
    "wallet_configured",
    "real_transaction_attempted",
    "event_quality",
]

HYGIENE_AUDIT_FIELDS = [
    "audit_timestamp_utc",
    "loop_run_id",
    "invocation_id",
    "source_file",
    "source_row_id",
    "position_id",
    "close_event_id",
    "closed_at_utc",
    "classification",
    "included_in_canonical_snapshot",
    "exclusion_reason",
    "missing_required_fields",
    "duplicate_position_count",
    "duplicate_close_event_count",
    "severity",
    "notes",
]

REQUIRED_CANONICAL = (
    "close_event_id",
    "position_id",
    "exit_reason",
    "entry_price",
    "exit_price",
    "quantity",
    "notional_usd",
    "cash_debited_usd",
    "cash_credited_usd",
)

PLACEHOLDER_POSITION_IDS = {
    "pos-old",
    "pos-test",
    "test-pos",
    "dummy",
    "placeholder",
    "test",
}


def _is_placeholder_position_id(pid: str) -> bool:
    """Legacy/test IDs only — do not treat real pos-<uuid> style IDs as placeholders."""
    low = pid.strip().lower()
    if low in PLACEHOLDER_POSITION_IDS:
        return True
    if low.startswith("pos-old") or low.startswith("pos-test"):
        return True
    return False


@dataclass
class CanonicalClosedResult:
    canonical_rows: list[dict[str, Any]] = field(default_factory=list)
    event_history_rows: list[dict[str, Any]] = field(default_factory=list)
    hygiene_rows: list[dict[str, Any]] = field(default_factory=list)
    closed_trade_hygiene_status: str = "PASS"
    canonical_closed_trades_rows: int = 0
    closed_trade_event_history_rows: int = 0
    invalid_closed_trade_rows: int = 0
    duplicate_closed_position_rows: int = 0
    duplicate_close_event_rows: int = 0
    blank_close_event_id_rows: int = 0
    canonical_closed_position_unique_count: int = 0

    def to_meta(self) -> dict[str, Any]:
        return {
            "closed_trade_hygiene_status": self.closed_trade_hygiene_status,
            "canonical_closed_trades_rows": self.canonical_closed_trades_rows,
            "closed_trade_event_history_rows": self.closed_trade_event_history_rows,
            "invalid_closed_trade_rows": self.invalid_closed_trade_rows,
            "duplicate_closed_position_rows": self.duplicate_closed_position_rows,
            "duplicate_close_event_rows": self.duplicate_close_event_rows,
            "blank_close_event_id_rows": self.blank_close_event_id_rows,
            "canonical_closed_position_unique_count": self.canonical_closed_position_unique_count,
        }


def _is_blank(val: Any) -> bool:
    return val is None or str(val).strip() == ""


def _load_jsonl_close_events(project_root: Path) -> list[tuple[dict[str, Any], str, int]]:
    """Return list of (record, source_file, row_index)."""
    out: list[tuple[dict[str, Any], str, int]] = []
    paper_dir = project_root / "data" / "paper_trading"
    if not paper_dir.is_dir():
        return out
    for path in sorted(paper_dir.glob("paper_trades_*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict):
                    continue
                # Include close-like records
                if rec.get("record_type") in (
                    "PAPER_TRADE",
                    "PAPER_TRADE_CLOSE",
                    "POSITION_CLOSED",
                ) or rec.get("exit_reason") or rec.get("close_reason") or rec.get(
                    "status"
                ) in ("CLOSED", "PAPER_CLOSED_TP", "PAPER_CLOSED_SL", "PAPER_CLOSED_TIME_STOP"):
                    out.append((rec, str(path), i))
                elif rec.get("position_id") and (
                    rec.get("closed_at_utc") or rec.get("exit_price") or rec.get("exit_price_usd")
                ):
                    out.append((rec, str(path), i))
    return out


def _sqlite_closed_rows(state_db: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        for r in state_db._conn.execute("SELECT * FROM closed_positions").fetchall():
            rows.append(dict(r))
    except Exception:
        pass
    if rows:
        return rows
    # Fallback: CLOSED active_positions
    try:
        for r in state_db._conn.execute(
            "SELECT * FROM active_positions WHERE status = 'CLOSED'"
        ).fetchall():
            d = dict(r)
            d.setdefault("economic_close_key", d.get("position_id"))
            d.setdefault("event_quality", "VALID_CANONICAL_CLOSE")
            if _is_blank(d.get("close_event_id")):
                d["close_event_id"] = str(uuid4())
            rows.append(d)
    except Exception:
        pass
    return rows


def _normalize_event(rec: dict[str, Any]) -> dict[str, Any]:
    return {
        "close_event_id": rec.get("close_event_id") or rec.get("trade_id"),
        "position_id": rec.get("position_id"),
        "economic_close_key": rec.get("economic_close_key") or rec.get("position_id"),
        "paper_order_id": rec.get("paper_order_id"),
        "source_decision_id": rec.get("source_decision_id"),
        "pair_address": rec.get("pair_address"),
        "opened_at_utc": rec.get("opened_at_utc"),
        "closed_at_utc": rec.get("closed_at_utc"),
        "exit_reason": rec.get("exit_reason") or rec.get("close_reason"),
        "entry_price": rec.get("entry_price") or rec.get("entry_price_usd"),
        "exit_price": rec.get("exit_price") or rec.get("exit_price_usd"),
        "quantity": rec.get("quantity"),
        "notional_usd": rec.get("notional_usd"),
        "cost_basis_usd": rec.get("cost_basis_usd"),
        "entry_fee_usd": rec.get("entry_fee_usd"),
        "exit_fee_usd": rec.get("exit_fee_usd") or rec.get("fee_usd"),
        "total_fees_usd": rec.get("total_fees_usd"),
        "gross_pnl_usd": rec.get("gross_pnl_usd"),
        "net_pnl_usd": rec.get("net_pnl_usd") or rec.get("realized_pnl_usd"),
        "net_return_pct": rec.get("net_return_pct"),
        "cash_debited_usd": rec.get("cash_debited_usd"),
        "cash_credited_usd": rec.get("cash_credited_usd"),
        "wallet_configured": rec.get("wallet_configured", False),
        "real_transaction_attempted": rec.get("real_transaction_attempted", False),
        "event_quality": rec.get("event_quality"),
    }


def _classify_row(row: dict[str, Any], *, from_sqlite: bool) -> tuple[str, str, list[str]]:
    """Return (classification, exclusion_reason, missing_fields)."""
    pid = str(row.get("position_id") or "").strip()
    missing = [f for f in REQUIRED_CANONICAL if _is_blank(row.get(f))]
    if not pid:
        return "LEGACY_TEST_OR_INVALID", "MISSING_POSITION_ID", missing
    if _is_placeholder_position_id(pid):
        return "LEGACY_TEST_OR_INVALID", "PLACEHOLDER_POSITION_ID", missing
    if _is_blank(row.get("close_event_id")):
        return "MISSING_CLOSE_EVENT_ID", "BLANK_CLOSE_EVENT_ID", missing
    if missing:
        return "MISSING_CLOSE_ECONOMICS", ",".join(missing), missing
    if from_sqlite:
        return "VALID_CANONICAL_CLOSE", "", []
    return "VALID_CANONICAL_CLOSE", "", []


def merge_closed_positions(
    *,
    sqlite_positions: list[dict[str, Any]],
    jsonl_history: list[tuple[dict[str, Any], str, int]],
    source_precedence: list[str] | None = None,
    one_row_per_position: bool = True,
    loop_run_id: str = "",
    invocation_id: str = "",
) -> CanonicalClosedResult:
    """Prefer SQLite for one-row-per-position; JSONL enriches and forms event history."""
    _ = source_precedence  # documented contract
    result = CanonicalClosedResult()
    now = utc_now_iso()

    # Index JSONL by position_id / close_event_id / paper_order_id for enrichment
    by_pid: dict[str, dict[str, Any]] = {}
    by_event: dict[str, dict[str, Any]] = {}
    by_order: dict[str, dict[str, Any]] = {}
    position_counts: dict[str, int] = {}
    event_counts: dict[str, int] = {}

    for rec, source_file, row_idx in jsonl_history:
        norm = _normalize_event(rec)
        result.event_history_rows.append(
            {
                **norm,
                "source_file": source_file,
                "source_row_id": row_idx,
                "event_quality": norm.get("event_quality") or "EVENT_HISTORY",
            }
        )
        pid = str(norm.get("position_id") or "")
        eid = str(norm.get("close_event_id") or "")
        oid = str(norm.get("paper_order_id") or "")
        if pid:
            position_counts[pid] = position_counts.get(pid, 0) + 1
            by_pid.setdefault(pid, norm)
        if eid:
            event_counts[eid] = event_counts.get(eid, 0) + 1
            by_event.setdefault(eid, norm)
        if oid:
            by_order.setdefault(oid, norm)

        classification, exclusion, missing = _classify_row(norm, from_sqlite=False)
        if classification != "VALID_CANONICAL_CLOSE":
            result.invalid_closed_trade_rows += 1
            if classification == "MISSING_CLOSE_EVENT_ID":
                result.blank_close_event_id_rows += 1
            result.hygiene_rows.append(
                {
                    "audit_timestamp_utc": now,
                    "loop_run_id": loop_run_id,
                    "invocation_id": invocation_id,
                    "source_file": source_file,
                    "source_row_id": str(row_idx),
                    "position_id": pid,
                    "close_event_id": eid,
                    "closed_at_utc": norm.get("closed_at_utc"),
                    "classification": classification,
                    "included_in_canonical_snapshot": False,
                    "exclusion_reason": exclusion,
                    "missing_required_fields": ",".join(missing),
                    "duplicate_position_count": position_counts.get(pid, 0),
                    "duplicate_close_event_count": event_counts.get(eid, 0) if eid else 0,
                    "severity": "WARNING",
                    "notes": "JSONL history row excluded from canonical snapshot",
                }
            )

    for pid, cnt in position_counts.items():
        if cnt > 1:
            result.duplicate_closed_position_rows += cnt - 1
    for eid, cnt in event_counts.items():
        if cnt > 1:
            result.duplicate_close_event_rows += cnt - 1

    # Build canonical from SQLite
    seen_pids: set[str] = set()
    for raw in sqlite_positions:
        row = _normalize_event(raw)
        pid = str(row.get("position_id") or "")
        if not pid or (one_row_per_position and pid in seen_pids):
            if pid in seen_pids:
                result.duplicate_closed_position_rows += 1
                result.hygiene_rows.append(
                    {
                        "audit_timestamp_utc": now,
                        "loop_run_id": loop_run_id,
                        "invocation_id": invocation_id,
                        "source_file": "sqlite:closed_positions",
                        "source_row_id": pid,
                        "position_id": pid,
                        "close_event_id": row.get("close_event_id"),
                        "closed_at_utc": row.get("closed_at_utc"),
                        "classification": "DUPLICATE_POSITION_CLOSE",
                        "included_in_canonical_snapshot": False,
                        "exclusion_reason": "DUPLICATE_POSITION_ID",
                        "missing_required_fields": "",
                        "duplicate_position_count": 2,
                        "duplicate_close_event_count": 0,
                        "severity": "WARNING",
                        "notes": "Duplicate SQLite closed row skipped",
                    }
                )
            continue

        # Enrich blanks from JSONL
        enrich = by_pid.get(pid) or by_event.get(str(row.get("close_event_id") or "")) or by_order.get(
            str(row.get("paper_order_id") or "")
        )
        if enrich:
            for k, v in enrich.items():
                if _is_blank(row.get(k)) and not _is_blank(v):
                    row[k] = v

        if _is_blank(row.get("close_event_id")):
            row["close_event_id"] = str(uuid4())
        row["economic_close_key"] = row.get("economic_close_key") or pid
        row["wallet_configured"] = row.get("wallet_configured", False)
        row["real_transaction_attempted"] = row.get("real_transaction_attempted", False)

        classification, exclusion, missing = _classify_row(row, from_sqlite=True)
        if classification != "VALID_CANONICAL_CLOSE":
            # Still allow SQLite CLOSED with generated close_event_id if economics present after enrich
            if classification == "MISSING_CLOSE_EVENT_ID":
                row["close_event_id"] = str(uuid4())
                classification, exclusion, missing = _classify_row(row, from_sqlite=True)
        if classification != "VALID_CANONICAL_CLOSE":
            result.invalid_closed_trade_rows += 1
            result.hygiene_rows.append(
                {
                    "audit_timestamp_utc": now,
                    "loop_run_id": loop_run_id,
                    "invocation_id": invocation_id,
                    "source_file": "sqlite:closed_positions",
                    "source_row_id": pid,
                    "position_id": pid,
                    "close_event_id": row.get("close_event_id"),
                    "closed_at_utc": row.get("closed_at_utc"),
                    "classification": classification,
                    "included_in_canonical_snapshot": False,
                    "exclusion_reason": exclusion,
                    "missing_required_fields": ",".join(missing),
                    "duplicate_position_count": 1,
                    "duplicate_close_event_count": 0,
                    "severity": "WARNING",
                    "notes": "SQLite closed row excluded from canonical",
                }
            )
            continue

        row["event_quality"] = "VALID_CANONICAL_CLOSE"
        result.canonical_rows.append({k: row.get(k) for k in CANONICAL_CLOSED_FIELDS})
        seen_pids.add(pid)
        result.hygiene_rows.append(
            {
                "audit_timestamp_utc": now,
                "loop_run_id": loop_run_id,
                "invocation_id": invocation_id,
                "source_file": "sqlite:closed_positions",
                "source_row_id": pid,
                "position_id": pid,
                "close_event_id": row.get("close_event_id"),
                "closed_at_utc": row.get("closed_at_utc"),
                "classification": "VALID_CANONICAL_CLOSE",
                "included_in_canonical_snapshot": True,
                "exclusion_reason": "",
                "missing_required_fields": "",
                "duplicate_position_count": position_counts.get(pid, 1),
                "duplicate_close_event_count": event_counts.get(str(row.get("close_event_id") or ""), 0),
                "severity": "INFO",
                "notes": "Canonical closed trade from SQLite",
            }
        )

    result.canonical_closed_trades_rows = len(result.canonical_rows)
    result.canonical_closed_position_unique_count = len({r["position_id"] for r in result.canonical_rows})
    result.closed_trade_event_history_rows = len(result.event_history_rows)

    if result.invalid_closed_trade_rows or result.duplicate_closed_position_rows or result.blank_close_event_id_rows:
        result.closed_trade_hygiene_status = "WARNING_WITH_LEGACY_ROWS_EXCLUDED"
    else:
        result.closed_trade_hygiene_status = "PASS"
    return result


def get_canonical_closed_trades(
    state_db: Any,
    *,
    project_root: Path,
    loop_run_id: str = "",
    invocation_id: str = "",
) -> CanonicalClosedResult:
    """Query SQLite closed state; enrich from JSONL; never duplicate by position_id."""
    sqlite_rows = _sqlite_closed_rows(state_db)
    jsonl_history = _load_jsonl_close_events(project_root)
    return merge_closed_positions(
        sqlite_positions=sqlite_rows,
        jsonl_history=jsonl_history,
        source_precedence=["sqlite", "jsonl_enrichment"],
        one_row_per_position=True,
        loop_run_id=loop_run_id,
        invocation_id=invocation_id,
    )


def write_hygiene_audit(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HYGIENE_AUDIT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
    return path


def write_event_history(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = CANONICAL_CLOSED_FIELDS + ["source_file", "source_row_id"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
    return path
