"""AE11 atomic pre-loop SQLite schema migrations (AE11E + AE11F)."""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.runtime_paper_loop.persistence import state_db_path
from app.runtime_paper_loop.types import utc_now_iso

MIGRATION_NAME_AE11E = "AE11E_POSITION_ECONOMICS_V1"
MIGRATION_VERSION_AE11E = "AE11E_V1"
MIGRATION_NAME = MIGRATION_NAME_AE11E  # backward-compatible export
MIGRATION_VERSION = MIGRATION_VERSION_AE11E

MIGRATION_NAME_AE11F = "AE11F_CLOSED_TRADE_IDEMPOTENCY_V1"
MIGRATION_VERSION_AE11F = "AE11F_V1"

MIGRATION_NAME_AE11G = "AE11G_POSITION_STATE_SEMANTICS_V1"
MIGRATION_VERSION_AE11G = "AE11G_V1"

MIGRATION_NAME_AE11I = "AE11I_MARK_TO_MARKET_V1"
MIGRATION_VERSION_AE11I = "AE11I_V1"

AE11I_MTM_COLUMNS: list[tuple[str, str]] = [
    ("valuation_source", "TEXT"),
    ("price_unrealized_pnl_usd", "TEXT"),
    ("total_unrealized_after_cost_pnl_usd", "TEXT"),
    ("open_entry_cost_drag_usd", "TEXT"),
]

ECONOMICS_COLUMNS: list[tuple[str, str]] = [
    ("candidate_id", "TEXT"),
    ("chain", "TEXT"),
    ("symbol", "TEXT"),
    ("entry_price", "TEXT"),
    ("entry_price_timestamp_utc", "TEXT"),
    ("entry_price_source", "TEXT"),
    ("entry_snapshot_id", "TEXT"),
    ("notional_usd", "TEXT"),
    ("quantity", "TEXT"),
    ("cost_basis_usd", "TEXT"),
    ("entry_fee_usd", "TEXT"),
    ("entry_slippage_usd", "TEXT"),
    ("cash_debited_usd", "TEXT"),
    ("tp_price", "TEXT"),
    ("sl_price", "TEXT"),
    ("time_stop_at_utc", "TEXT"),
    ("take_profit_pct", "TEXT"),
    ("stop_loss_pct", "TEXT"),
    ("time_stop_minutes", "TEXT"),
    ("last_price", "TEXT"),
    ("last_price_timestamp_utc", "TEXT"),
    ("last_valuation_at_utc", "TEXT"),
    ("unrealized_pnl_usd", "TEXT"),
    ("unrealized_return_pct", "TEXT"),
    ("open_market_value_usd", "TEXT"),
    ("closed_at_utc", "TEXT"),
    ("exit_price", "TEXT"),
    ("exit_price_timestamp_utc", "TEXT"),
    ("exit_reason", "TEXT"),
    ("gross_pnl_usd", "TEXT"),
    ("exit_fee_usd", "TEXT"),
    ("total_fees_usd", "TEXT"),
    ("net_pnl_usd", "TEXT"),
    ("net_return_pct", "TEXT"),
    ("cash_credited_usd", "TEXT"),
    ("trade_authority", "TEXT"),
    ("not_model_approved", "TEXT"),
    ("not_live_approved", "TEXT"),
    ("override_type", "TEXT"),
    ("economic_enrichment_status", "TEXT"),
    ("economic_enrichment_missing_fields", "TEXT"),
    ("close_event_id", "TEXT"),
]

SCHEMA_MIGRATIONS_SQL = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    migration_name TEXT PRIMARY KEY,
    migration_version TEXT NOT NULL,
    applied_at_utc TEXT NOT NULL,
    notes TEXT
);
"""

CLOSED_POSITIONS_SQL = """
CREATE TABLE IF NOT EXISTS closed_positions (
    position_id TEXT PRIMARY KEY,
    close_event_id TEXT NOT NULL,
    economic_close_key TEXT NOT NULL,
    paper_order_id TEXT,
    source_decision_id TEXT,
    pair_address TEXT,
    opened_at_utc TEXT,
    closed_at_utc TEXT NOT NULL,
    close_event_created_at_utc TEXT NOT NULL,
    exit_reason TEXT,
    entry_price TEXT,
    exit_price TEXT,
    quantity TEXT,
    notional_usd TEXT,
    cost_basis_usd TEXT,
    entry_fee_usd TEXT,
    exit_fee_usd TEXT,
    total_fees_usd TEXT,
    gross_pnl_usd TEXT,
    net_pnl_usd TEXT,
    net_return_pct TEXT,
    cash_debited_usd TEXT,
    cash_credited_usd TEXT,
    wallet_configured TEXT,
    real_transaction_attempted TEXT,
    event_quality TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_closed_positions_position_id_unique
    ON closed_positions(position_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_closed_positions_close_event_id_unique
    ON closed_positions(close_event_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_closed_positions_economic_close_key_unique
    ON closed_positions(economic_close_key);
"""

# AE11G: status-filtered open queries + lock lookups
POSITION_STATE_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_active_positions_status
    ON active_positions(status);
CREATE INDEX IF NOT EXISTS idx_active_positions_status_pair
    ON active_positions(status, pair_address);
CREATE INDEX IF NOT EXISTS idx_active_positions_status_position_id
    ON active_positions(status, position_id);
CREATE INDEX IF NOT EXISTS idx_active_pair_locks_position_id
    ON active_pair_locks(position_id);
CREATE INDEX IF NOT EXISTS idx_active_positions_open
    ON active_positions(position_id, pair_address) WHERE status = 'OPEN';
"""

MIGRATION_AUDIT_FIELDS = [
    "audit_timestamp_utc",
    "migration_name",
    "migration_version",
    "migration_applied",
    "migration_skipped_reason",
    "tables_changed",
    "columns_added",
    "backfill_required",
    "backfill_started_at_utc",
    "backfill_completed_at_utc",
    "backfill_position_count",
    "backfill_success_count",
    "backfill_partial_count",
    "backfill_missing_count",
    "status",
    "notes",
]


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def _write_migration_audit(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MIGRATION_AUDIT_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def _migration_applied(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM schema_migrations WHERE migration_name = ?",
        (name,),
    ).fetchone()
    return row is not None


def _apply_ae11e(conn: sqlite3.Connection, audit: Path, now: str) -> dict[str, Any]:
    if _migration_applied(conn, MIGRATION_NAME_AE11E):
        row = {
            "audit_timestamp_utc": now,
            "migration_name": MIGRATION_NAME_AE11E,
            "migration_version": MIGRATION_VERSION_AE11E,
            "migration_applied": False,
            "migration_skipped_reason": "ALREADY_APPLIED",
            "tables_changed": "",
            "columns_added": "",
            "backfill_required": False,
            "backfill_started_at_utc": None,
            "backfill_completed_at_utc": None,
            "backfill_position_count": None,
            "backfill_success_count": None,
            "backfill_partial_count": None,
            "backfill_missing_count": None,
            "status": "SKIPPED",
            "notes": "schema_migrations already contains AE11E_POSITION_ECONOMICS_V1",
        }
        _write_migration_audit(audit, row)
        return {
            "migration_applied": False,
            "migration_skipped_reason": "ALREADY_APPLIED",
            "columns_added": [],
            "status": "SKIPPED",
        }

    cols = _table_columns(conn, "active_positions")
    added: list[str] = []
    conn.execute("BEGIN")
    try:
        for col_name, col_type in ECONOMICS_COLUMNS:
            if col_name in cols:
                continue
            conn.execute(f"ALTER TABLE active_positions ADD COLUMN {col_name} {col_type}")
            added.append(col_name)
        conn.execute(
            """
            INSERT INTO schema_migrations (migration_name, migration_version, applied_at_utc, notes)
            VALUES (?, ?, ?, ?)
            """,
            (
                MIGRATION_NAME_AE11E,
                MIGRATION_VERSION_AE11E,
                now,
                f"Added columns: {','.join(added) if added else 'none'}",
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    row = {
        "audit_timestamp_utc": now,
        "migration_name": MIGRATION_NAME_AE11E,
        "migration_version": MIGRATION_VERSION_AE11E,
        "migration_applied": True,
        "migration_skipped_reason": None,
        "tables_changed": "active_positions,schema_migrations",
        "columns_added": ",".join(added),
        "backfill_required": True,
        "backfill_started_at_utc": None,
        "backfill_completed_at_utc": None,
        "backfill_position_count": None,
        "backfill_success_count": None,
        "backfill_partial_count": None,
        "backfill_missing_count": None,
        "status": "APPLIED",
        "notes": "Additive AE11E economics migration",
    }
    _write_migration_audit(audit, row)
    return {
        "migration_applied": True,
        "migration_skipped_reason": None,
        "columns_added": added,
        "status": "APPLIED",
    }


def _backfill_closed_positions_from_active(conn: sqlite3.Connection) -> int:
    """Seed closed_positions from existing CLOSED active_positions (one row per position)."""
    cols = _table_columns(conn, "active_positions")
    rows = conn.execute(
        "SELECT * FROM active_positions WHERE status = 'CLOSED'"
    ).fetchall()
    inserted = 0
    for r in rows:
        d = dict(r)
        pid = d.get("position_id")
        if not pid:
            continue
        existing = conn.execute(
            "SELECT 1 FROM closed_positions WHERE position_id = ?",
            (pid,),
        ).fetchone()
        if existing:
            continue
        close_event_id = d.get("close_event_id") or str(uuid4())
        closed_at = d.get("closed_at_utc") or utc_now_iso()
        conn.execute(
            """
            INSERT OR IGNORE INTO closed_positions (
                position_id, close_event_id, economic_close_key, paper_order_id,
                source_decision_id, pair_address, opened_at_utc, closed_at_utc,
                close_event_created_at_utc, exit_reason, entry_price, exit_price,
                quantity, notional_usd, cost_basis_usd, entry_fee_usd, exit_fee_usd,
                total_fees_usd, gross_pnl_usd, net_pnl_usd, net_return_pct,
                cash_debited_usd, cash_credited_usd, wallet_configured,
                real_transaction_attempted, event_quality
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pid,
                close_event_id,
                pid,
                d.get("paper_order_id"),
                d.get("source_decision_id"),
                d.get("pair_address"),
                d.get("opened_at_utc"),
                closed_at,
                closed_at,
                d.get("exit_reason"),
                d.get("entry_price"),
                d.get("exit_price"),
                d.get("quantity"),
                d.get("notional_usd"),
                d.get("cost_basis_usd"),
                d.get("entry_fee_usd"),
                d.get("exit_fee_usd"),
                d.get("total_fees_usd"),
                d.get("gross_pnl_usd"),
                d.get("net_pnl_usd"),
                d.get("net_return_pct"),
                d.get("cash_debited_usd"),
                d.get("cash_credited_usd"),
                "False",
                "False",
                "VALID_CANONICAL_CLOSE" if d.get("entry_price") and d.get("exit_price") else "MISSING_CLOSE_ECONOMICS",
            ),
        )
        if "close_event_id" in cols and not d.get("close_event_id"):
            conn.execute(
                "UPDATE active_positions SET close_event_id = ? WHERE position_id = ?",
                (close_event_id, pid),
            )
        inserted += 1
    return inserted


def _apply_ae11f(conn: sqlite3.Connection, audit: Path, now: str) -> dict[str, Any]:
    if _migration_applied(conn, MIGRATION_NAME_AE11F):
        row = {
            "audit_timestamp_utc": now,
            "migration_name": MIGRATION_NAME_AE11F,
            "migration_version": MIGRATION_VERSION_AE11F,
            "migration_applied": False,
            "migration_skipped_reason": "ALREADY_APPLIED",
            "tables_changed": "",
            "columns_added": "",
            "backfill_required": False,
            "backfill_started_at_utc": None,
            "backfill_completed_at_utc": None,
            "backfill_position_count": None,
            "backfill_success_count": None,
            "backfill_partial_count": None,
            "backfill_missing_count": None,
            "status": "SKIPPED",
            "notes": "schema_migrations already contains AE11F_CLOSED_TRADE_IDEMPOTENCY_V1",
        }
        _write_migration_audit(audit, row)
        return {
            "migration_applied": False,
            "migration_skipped_reason": "ALREADY_APPLIED",
            "status": "SKIPPED",
            "closed_positions_seeded": 0,
        }

    conn.execute("BEGIN")
    try:
        # Ensure close_event_id on active_positions
        cols = _table_columns(conn, "active_positions")
        if "close_event_id" not in cols:
            conn.execute("ALTER TABLE active_positions ADD COLUMN close_event_id TEXT")
        conn.executescript(CLOSED_POSITIONS_SQL)
        seeded = _backfill_closed_positions_from_active(conn)
        conn.execute(
            """
            INSERT INTO schema_migrations (migration_name, migration_version, applied_at_utc, notes)
            VALUES (?, ?, ?, ?)
            """,
            (
                MIGRATION_NAME_AE11F,
                MIGRATION_VERSION_AE11F,
                now,
                f"closed_positions + unique indexes; seeded={seeded}",
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    row = {
        "audit_timestamp_utc": now,
        "migration_name": MIGRATION_NAME_AE11F,
        "migration_version": MIGRATION_VERSION_AE11F,
        "migration_applied": True,
        "migration_skipped_reason": None,
        "tables_changed": "closed_positions,active_positions",
        "columns_added": "close_event_id",
        "backfill_required": False,
        "backfill_started_at_utc": now,
        "backfill_completed_at_utc": now,
        "backfill_position_count": seeded,
        "backfill_success_count": seeded,
        "backfill_partial_count": 0,
        "backfill_missing_count": 0,
        "status": "APPLIED",
        "notes": "AE11F closed trade idempotency: UNIQUE(position_id) on closed_positions",
    }
    _write_migration_audit(audit, row)
    return {
        "migration_applied": True,
        "migration_skipped_reason": None,
        "status": "APPLIED",
        "closed_positions_seeded": seeded,
    }


def _apply_ae11g(conn: sqlite3.Connection, audit: Path, now: str) -> dict[str, Any]:
    if _migration_applied(conn, MIGRATION_NAME_AE11G):
        row = {
            "audit_timestamp_utc": now,
            "migration_name": MIGRATION_NAME_AE11G,
            "migration_version": MIGRATION_VERSION_AE11G,
            "migration_applied": False,
            "migration_skipped_reason": "ALREADY_APPLIED",
            "tables_changed": "",
            "columns_added": "",
            "backfill_required": False,
            "backfill_started_at_utc": None,
            "backfill_completed_at_utc": None,
            "backfill_position_count": None,
            "backfill_success_count": None,
            "backfill_partial_count": None,
            "backfill_missing_count": None,
            "status": "SKIPPED",
            "notes": "schema_migrations already contains AE11G_POSITION_STATE_SEMANTICS_V1",
        }
        _write_migration_audit(audit, row)
        return {
            "migration_applied": False,
            "migration_skipped_reason": "ALREADY_APPLIED",
            "status": "SKIPPED",
        }

    conn.execute("BEGIN")
    try:
        conn.executescript(POSITION_STATE_INDEXES_SQL)
        conn.execute(
            """
            INSERT INTO schema_migrations (migration_name, migration_version, applied_at_utc, notes)
            VALUES (?, ?, ?, ?)
            """,
            (
                MIGRATION_NAME_AE11G,
                MIGRATION_VERSION_AE11G,
                now,
                "status/open indexes + active_pair_locks(position_id); registry semantics",
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    row = {
        "audit_timestamp_utc": now,
        "migration_name": MIGRATION_NAME_AE11G,
        "migration_version": MIGRATION_VERSION_AE11G,
        "migration_applied": True,
        "migration_skipped_reason": None,
        "tables_changed": "active_positions,active_pair_locks",
        "columns_added": "",
        "backfill_required": False,
        "backfill_started_at_utc": None,
        "backfill_completed_at_utc": None,
        "backfill_position_count": None,
        "backfill_success_count": None,
        "backfill_partial_count": None,
        "backfill_missing_count": None,
        "status": "APPLIED",
        "notes": (
            "AE11G: active_positions is a registry; OPEN filtered via status indexes; "
            "capacity must use status='OPEN' only"
        ),
    }
    _write_migration_audit(audit, row)
    return {
        "migration_applied": True,
        "migration_skipped_reason": None,
        "status": "APPLIED",
    }


def _apply_ae11i(conn: sqlite3.Connection, audit: Path, now: str) -> dict[str, Any]:
    if _migration_applied(conn, MIGRATION_NAME_AE11I):
        row = {
            "audit_timestamp_utc": now,
            "migration_name": MIGRATION_NAME_AE11I,
            "migration_version": MIGRATION_VERSION_AE11I,
            "migration_applied": False,
            "migration_skipped_reason": "ALREADY_APPLIED",
            "tables_changed": "",
            "columns_added": "",
            "backfill_required": False,
            "backfill_started_at_utc": None,
            "backfill_completed_at_utc": None,
            "backfill_position_count": None,
            "backfill_success_count": None,
            "backfill_partial_count": None,
            "backfill_missing_count": None,
            "status": "SKIPPED",
            "notes": "schema_migrations already contains AE11I_MARK_TO_MARKET_V1",
        }
        _write_migration_audit(audit, row)
        return {
            "migration_applied": False,
            "migration_skipped_reason": "ALREADY_APPLIED",
            "columns_added": [],
            "status": "SKIPPED",
        }

    cols = _table_columns(conn, "active_positions")
    added: list[str] = []
    conn.execute("BEGIN")
    try:
        for col_name, col_type in AE11I_MTM_COLUMNS:
            if col_name in cols:
                continue
            conn.execute(f"ALTER TABLE active_positions ADD COLUMN {col_name} {col_type}")
            added.append(col_name)
        conn.execute(
            """
            INSERT INTO schema_migrations (migration_name, migration_version, applied_at_utc, notes)
            VALUES (?, ?, ?, ?)
            """,
            (
                MIGRATION_NAME_AE11I,
                MIGRATION_VERSION_AE11I,
                now,
                f"Added AE11I MTM columns: {','.join(added) if added else 'none'}",
            ),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

    row = {
        "audit_timestamp_utc": now,
        "migration_name": MIGRATION_NAME_AE11I,
        "migration_version": MIGRATION_VERSION_AE11I,
        "migration_applied": True,
        "migration_skipped_reason": None,
        "tables_changed": "active_positions",
        "columns_added": ",".join(added),
        "backfill_required": False,
        "backfill_started_at_utc": None,
        "backfill_completed_at_utc": None,
        "backfill_position_count": None,
        "backfill_success_count": None,
        "backfill_partial_count": None,
        "backfill_missing_count": None,
        "status": "APPLIED",
        "notes": "AE11I mark-to-market: valuation_source + price unrealized PnL decomposition",
    }
    _write_migration_audit(audit, row)
    return {
        "migration_applied": True,
        "migration_skipped_reason": None,
        "columns_added": added,
        "status": "APPLIED",
    }


def migrate_db_schema(
    *,
    db_path: Path | None = None,
    project_root: Path | None = None,
    audit_path: Path | None = None,
) -> dict[str, Any]:
    """
    Run once at script startup before the runtime loop.

    Applies AE11E then AE11F then AE11G then AE11I. Each step is additive and idempotent.
    Never call inside the hot iteration loop.
    """
    path = db_path or state_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    root = project_root or Path(__file__).parents[2]
    audit = audit_path or (root / "audits" / "ae11_db_migration_audit.csv")
    now = utc_now_iso()

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA_MIGRATIONS_SQL)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS active_positions (
                position_id TEXT PRIMARY KEY,
                pair_address TEXT NOT NULL,
                opened_at_utc TEXT NOT NULL,
                status TEXT NOT NULL,
                paper_order_id TEXT,
                source_decision_id TEXT
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS active_pair_locks (
                pair_address TEXT PRIMARY KEY,
                position_id TEXT NOT NULL,
                locked_at_utc TEXT NOT NULL
            )
            """
        )
        ae11e = _apply_ae11e(conn, audit, now)
        ae11f = _apply_ae11f(conn, audit, now)
        ae11g = _apply_ae11g(conn, audit, now)
        ae11i = _apply_ae11i(conn, audit, now)
        any_applied = bool(
            ae11e.get("migration_applied")
            or ae11f.get("migration_applied")
            or ae11g.get("migration_applied")
            or ae11i.get("migration_applied")
        )
        return {
            "migration_applied": any_applied,
            "migration_skipped_reason": None
            if any_applied
            else (
                ae11i.get("migration_skipped_reason")
                or ae11g.get("migration_skipped_reason")
                or ae11f.get("migration_skipped_reason")
                or ae11e.get("migration_skipped_reason")
            ),
            "columns_added": (ae11e.get("columns_added") or [])
            + (ae11i.get("columns_added") or []),
            "backfill_required": True,
            "status": "APPLIED" if any_applied else "SKIPPED",
            "ae11e": ae11e,
            "ae11f": ae11f,
            "ae11g": ae11g,
            "ae11i": ae11i,
        }
    finally:
        conn.close()


def update_migration_audit_backfill(
    *,
    project_root: Path,
    backfill_started_at_utc: str,
    backfill_completed_at_utc: str,
    backfill_position_count: int,
    backfill_success_count: int,
    backfill_partial_count: int,
    backfill_missing_count: int,
    status: str = "BACKFILL_COMPLETE",
    notes: str = "",
) -> None:
    audit = project_root / "audits" / "ae11_db_migration_audit.csv"
    row = {
        "audit_timestamp_utc": utc_now_iso(),
        "migration_name": MIGRATION_NAME_AE11E,
        "migration_version": MIGRATION_VERSION_AE11E,
        "migration_applied": False,
        "migration_skipped_reason": "BACKFILL_PASS",
        "tables_changed": "active_positions",
        "columns_added": "",
        "backfill_required": True,
        "backfill_started_at_utc": backfill_started_at_utc,
        "backfill_completed_at_utc": backfill_completed_at_utc,
        "backfill_position_count": backfill_position_count,
        "backfill_success_count": backfill_success_count,
        "backfill_partial_count": backfill_partial_count,
        "backfill_missing_count": backfill_missing_count,
        "status": status,
        "notes": notes,
    }
    _write_migration_audit(audit, row)
