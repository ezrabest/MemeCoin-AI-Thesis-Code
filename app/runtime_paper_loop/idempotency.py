"""Scalable idempotency index via AE11-owned SQLite state DB."""

from __future__ import annotations

import sqlite3
from collections import OrderedDict
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.runtime_paper_loop.persistence import state_db_path
from app.runtime_paper_loop.types import utc_now_iso

SCHEMA_VERSION = "AE11B_V1"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS processed_decisions (
    source_decision_id TEXT PRIMARY KEY,
    source_event_key TEXT,
    first_seen_at_utc TEXT NOT NULL,
    last_seen_at_utc TEXT NOT NULL,
    action_taken TEXT,
    strict_decision TEXT,
    exploration_decision TEXT,
    paper_order_id TEXT,
    candidate_id TEXT,
    pair_address TEXT,
    event_timestamp TEXT,
    source_snapshot_id TEXT,
    source_snapshot_timestamp TEXT
);
CREATE TABLE IF NOT EXISTS active_positions (
    position_id TEXT PRIMARY KEY,
    pair_address TEXT NOT NULL,
    opened_at_utc TEXT NOT NULL,
    status TEXT NOT NULL,
    paper_order_id TEXT,
    source_decision_id TEXT
);
CREATE TABLE IF NOT EXISTS active_pair_locks (
    pair_address TEXT PRIMARY KEY,
    position_id TEXT NOT NULL,
    locked_at_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS cooldowns (
    pair_address TEXT PRIMARY KEY,
    last_action_at_utc TEXT NOT NULL,
    cooldown_until_utc TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS source_cursors (
    source_name TEXT NOT NULL,
    source_file TEXT NOT NULL,
    cursor_type TEXT,
    cursor_value TEXT,
    last_seen_decision_id TEXT,
    last_seen_created_at_utc TEXT,
    eof_reached_at_utc TEXT,
    updated_at_utc TEXT,
    PRIMARY KEY (source_name, source_file)
);
CREATE TABLE IF NOT EXISTS loop_checkpoints (
    loop_run_id TEXT NOT NULL,
    iteration INTEGER NOT NULL,
    completed_at_utc TEXT NOT NULL,
    checkpoint_json TEXT NOT NULL,
    PRIMARY KEY (loop_run_id, iteration)
);
CREATE INDEX IF NOT EXISTS idx_processed_decisions_pair ON processed_decisions(pair_address);
CREATE INDEX IF NOT EXISTS idx_active_positions_pair_status ON active_positions(pair_address, status);
CREATE INDEX IF NOT EXISTS idx_cooldowns_until ON cooldowns(cooldown_until_utc);
"""

INDEX_MIGRATION_SQL = """
CREATE INDEX IF NOT EXISTS idx_processed_decisions_event_key ON processed_decisions(source_event_key);
CREATE INDEX IF NOT EXISTS idx_processed_decisions_event_ts ON processed_decisions(event_timestamp);
"""

MIGRATION_COLUMNS = [
    ("processed_decisions", "source_event_key", "TEXT"),
    ("processed_decisions", "strict_decision", "TEXT"),
    ("processed_decisions", "exploration_decision", "TEXT"),
    ("processed_decisions", "event_timestamp", "TEXT"),
    ("processed_decisions", "source_snapshot_id", "TEXT"),
    ("processed_decisions", "source_snapshot_timestamp", "TEXT"),
]


class BoundedProcessedCache:
    """Bounded recent cache — SQLite remains source of truth."""

    def __init__(self, max_size: int = 512) -> None:
        self.max_size = max_size
        self._cache: OrderedDict[str, bool] = OrderedDict()

    def remember(self, key: str) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            self._cache[key] = True
            if len(self._cache) > self.max_size:
                self._cache.popitem(last=False)

    def might_contain(self, key: str) -> bool:
        return key in self._cache


class AE11StateDb:
    """AE11-owned SQLite index — durable idempotency, no full JSONL scan on startup."""

    def __init__(self, path: Path | None = None, cache_size: int = 512) -> None:
        self.path = path or state_db_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)
        self._migrate_schema()
        self._conn.executescript(INDEX_MIGRATION_SQL)
        self._conn.commit()
        self._recent_cache = BoundedProcessedCache(max_size=cache_size)

    def _migrate_schema(self) -> None:
        for table, column, col_type in MIGRATION_COLUMNS:
            try:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")
            except sqlite3.OperationalError:
                pass

    def close(self) -> None:
        self._conn.close()

    def is_decision_processed(
        self,
        source_decision_id: str,
        *,
        source_event_key: str | None = None,
    ) -> bool:
        if self._recent_cache.might_contain(source_decision_id):
            return True
        row = self._conn.execute(
            "SELECT 1 FROM processed_decisions WHERE source_decision_id = ?",
            (source_decision_id,),
        ).fetchone()
        if row:
            self._recent_cache.remember(source_decision_id)
            return True
        if source_event_key:
            row = self._conn.execute(
                "SELECT 1 FROM processed_decisions WHERE source_event_key = ?",
                (source_event_key,),
            ).fetchone()
            if row:
                self._recent_cache.remember(source_decision_id)
                return True
        return False

    def mark_decision_processed(
        self,
        source_decision_id: str,
        *,
        action_taken: str,
        source_event_key: str | None = None,
        strict_decision: str | None = None,
        exploration_decision: str | None = None,
        paper_order_id: str | None = None,
        candidate_id: str | None = None,
        pair_address: str | None = None,
        event_timestamp: str | None = None,
        source_snapshot_id: str | None = None,
        source_snapshot_timestamp: str | None = None,
    ) -> None:
        now = utc_now_iso()
        existing = self._conn.execute(
            "SELECT first_seen_at_utc FROM processed_decisions WHERE source_decision_id = ?",
            (source_decision_id,),
        ).fetchone()
        first_seen = existing["first_seen_at_utc"] if existing else now
        self._conn.execute(
            """
            INSERT OR REPLACE INTO processed_decisions
            (source_decision_id, source_event_key, first_seen_at_utc, last_seen_at_utc,
             action_taken, strict_decision, exploration_decision, paper_order_id,
             candidate_id, pair_address, event_timestamp, source_snapshot_id,
             source_snapshot_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_decision_id,
                source_event_key,
                first_seen,
                now,
                action_taken,
                strict_decision,
                exploration_decision,
                paper_order_id,
                candidate_id,
                pair_address,
                event_timestamp,
                source_snapshot_id,
                source_snapshot_timestamp,
            ),
        )
        self._conn.commit()
        self._recent_cache.remember(source_decision_id)

    def get_source_cursor(self, source_name: str, source_file: str) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT * FROM source_cursors WHERE source_name = ? AND source_file = ?",
            (source_name, source_file),
        ).fetchone()
        if not row:
            return {
                "source_name": source_name,
                "source_file": source_file,
                "cursor_type": "byte_offset",
                "cursor_value": "0",
            }
        return dict(row)

    def update_source_cursor(
        self,
        source_name: str,
        source_file: str,
        *,
        cursor_type: str,
        cursor_value: str,
        last_seen_decision_id: str | None = None,
        last_seen_created_at_utc: str | None = None,
        eof_reached: bool = False,
    ) -> None:
        now = utc_now_iso()
        eof_at = now if eof_reached else None
        self._conn.execute(
            """
            INSERT OR REPLACE INTO source_cursors
            (source_name, source_file, cursor_type, cursor_value,
             last_seen_decision_id, last_seen_created_at_utc, eof_reached_at_utc, updated_at_utc)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source_name,
                source_file,
                cursor_type,
                cursor_value,
                last_seen_decision_id,
                last_seen_created_at_utc,
                eof_at,
                now,
            ),
        )
        self._conn.commit()

    def reset_source_cursor(self, source_name: str, source_file: str) -> None:
        self.update_source_cursor(
            source_name,
            source_file,
            cursor_type="byte_offset",
            cursor_value="0",
            eof_reached=False,
        )

    def has_active_pair_lock(self, pair_address: str) -> bool:
        row = self._conn.execute(
            """
            SELECT 1 FROM active_pair_locks apl
            JOIN active_positions ap ON apl.position_id = ap.position_id
            WHERE apl.pair_address = ? AND ap.status = 'OPEN'
            """,
            (pair_address,),
        ).fetchone()
        return row is not None

    def register_position(
        self,
        position_id: str,
        pair_address: str,
        *,
        paper_order_id: str,
        source_decision_id: str | None,
        opened_at_utc: str | None = None,
        economics: dict[str, Any] | None = None,
    ) -> None:
        now = opened_at_utc or utc_now_iso()
        existing = self._conn.execute(
            "SELECT position_id FROM active_positions WHERE position_id = ?",
            (position_id,),
        ).fetchone()
        if existing:
            self._conn.execute(
                """
                UPDATE active_positions
                SET pair_address = ?, opened_at_utc = COALESCE(?, opened_at_utc),
                    status = 'OPEN', paper_order_id = COALESCE(?, paper_order_id),
                    source_decision_id = COALESCE(?, source_decision_id)
                WHERE position_id = ?
                """,
                (pair_address, now, paper_order_id, source_decision_id, position_id),
            )
        else:
            self._conn.execute(
                """
                INSERT INTO active_positions
                (position_id, pair_address, opened_at_utc, status, paper_order_id, source_decision_id)
                VALUES (?, ?, ?, 'OPEN', ?, ?)
                """,
                (position_id, pair_address, now, paper_order_id, source_decision_id),
            )
        self._conn.execute(
            """
            INSERT OR REPLACE INTO active_pair_locks (pair_address, position_id, locked_at_utc)
            VALUES (?, ?, ?)
            """,
            (pair_address, position_id, now),
        )
        self._conn.commit()
        if economics:
            self.update_position_economics(position_id, economics)

    def update_position_economics(self, position_id: str, fields: dict[str, Any]) -> None:
        """Update additive economics columns on active_positions (AE11E)."""
        if not fields:
            return
        # Only update columns that exist on the table
        existing = {
            r[1]
            for r in self._conn.execute("PRAGMA table_info(active_positions)").fetchall()
        }
        allowed = {k: v for k, v in fields.items() if k in existing and k != "position_id"}
        if not allowed:
            return
        assignments = ", ".join(f"{k} = ?" for k in allowed)
        values = list(allowed.values()) + [position_id]
        self._conn.execute(
            f"UPDATE active_positions SET {assignments} WHERE position_id = ?",
            values,
        )
        self._conn.commit()

    def close_position(self, position_id: str, pair_address: str) -> None:
        """
        Transition OPEN → CLOSED in one transaction and remove pair locks.

        AE11G: active_positions is a registry; CLOSED rows may remain.
        Locks must never point at CLOSED positions (ghost-lock prevention).
        """
        self._conn.execute("BEGIN")
        try:
            self._conn.execute(
                "UPDATE active_positions SET status = 'CLOSED' WHERE position_id = ?",
                (position_id,),
            )
            # Delete by position_id (and pair) so no ghost lock can remain
            self._conn.execute(
                "DELETE FROM active_pair_locks WHERE position_id = ?",
                (position_id,),
            )
            if pair_address:
                self._conn.execute(
                    "DELETE FROM active_pair_locks WHERE pair_address = ? AND position_id = ?",
                    (pair_address, position_id),
                )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def get_position_status(self, position_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT status FROM active_positions WHERE position_id = ?",
            (position_id,),
        ).fetchone()
        return str(row["status"]) if row else None

    def count_open_positions(self) -> int:
        """Authoritative OPEN count — use for capacity / reports / ledger."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM active_positions WHERE status = 'OPEN'"
        ).fetchone()
        return int(row["cnt"]) if row else 0

    def get_open_positions(self) -> list[dict[str, Any]]:
        """
        Authoritative open-position repository query.

        Only status='OPEN'. Never returns CLOSED registry rows.
        Preferred business-logic path for open positions (AE11G).
        """
        rows = self._conn.execute(
            "SELECT * FROM active_positions WHERE status = 'OPEN'"
        ).fetchall()
        return [dict(r) for r in rows]

    def count_position_registry_rows(self) -> int:
        """Diagnostic: total active_positions rows (OPEN + CLOSED). Not for capacity."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM active_positions"
        ).fetchone()
        return int(row["cnt"]) if row else 0

    def count_closed_rows_in_registry(self) -> int:
        """Diagnostic: CLOSED rows retained in the position registry table."""
        row = self._conn.execute(
            "SELECT COUNT(*) AS cnt FROM active_positions WHERE status = 'CLOSED'"
        ).fetchone()
        return int(row["cnt"]) if row else 0

    def count_closed_positions_economic(self) -> int:
        try:
            row = self._conn.execute(
                "SELECT COUNT(*) AS cnt FROM closed_positions"
            ).fetchone()
            return int(row["cnt"]) if row else 0
        except sqlite3.OperationalError:
            return 0

    def is_economically_closed(self, position_id: str) -> bool:
        """True if this position_id already has an economic close row (AE11F)."""
        try:
            row = self._conn.execute(
                "SELECT 1 FROM closed_positions WHERE position_id = ?",
                (position_id,),
            ).fetchone()
            return row is not None
        except sqlite3.OperationalError:
            return False

    def record_economic_close(self, close_fields: dict[str, Any]) -> dict[str, Any]:
        """
        Insert one economic close per position_id.

        On UNIQUE / IntegrityError: do not raise — caller must skip cash credit.
        """
        position_id = close_fields.get("position_id")
        if not position_id:
            return {
                "recorded": False,
                "duplicate": False,
                "reason": "MISSING_POSITION_ID",
            }
        if self.is_economically_closed(str(position_id)):
            return {
                "recorded": False,
                "duplicate": True,
                "reason": "DUPLICATE_POSITION_CLOSE_SKIPPED",
                "position_id": position_id,
            }

        close_event_id = close_fields.get("close_event_id") or str(uuid4())
        economic_close_key = close_fields.get("economic_close_key") or position_id
        closed_at = close_fields.get("closed_at_utc") or utc_now_iso()
        created_at = close_fields.get("close_event_created_at_utc") or closed_at
        try:
            self._conn.execute(
                """
                INSERT INTO closed_positions (
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
                    position_id,
                    close_event_id,
                    economic_close_key,
                    close_fields.get("paper_order_id"),
                    close_fields.get("source_decision_id"),
                    close_fields.get("pair_address"),
                    close_fields.get("opened_at_utc"),
                    closed_at,
                    created_at,
                    close_fields.get("exit_reason"),
                    close_fields.get("entry_price"),
                    close_fields.get("exit_price"),
                    close_fields.get("quantity"),
                    close_fields.get("notional_usd"),
                    close_fields.get("cost_basis_usd"),
                    close_fields.get("entry_fee_usd"),
                    close_fields.get("exit_fee_usd"),
                    close_fields.get("total_fees_usd"),
                    close_fields.get("gross_pnl_usd"),
                    close_fields.get("net_pnl_usd"),
                    close_fields.get("net_return_pct"),
                    close_fields.get("cash_debited_usd"),
                    close_fields.get("cash_credited_usd"),
                    str(close_fields.get("wallet_configured", False)),
                    str(close_fields.get("real_transaction_attempted", False)),
                    close_fields.get("event_quality") or "VALID_CANONICAL_CLOSE",
                ),
            )
            self._conn.commit()
            return {
                "recorded": True,
                "duplicate": False,
                "close_event_id": close_event_id,
                "economic_close_key": economic_close_key,
                "position_id": position_id,
            }
        except sqlite3.IntegrityError:
            return {
                "recorded": False,
                "duplicate": True,
                "reason": "DUPLICATE_POSITION_CLOSE_SKIPPED",
                "position_id": position_id,
                "close_event_id": close_event_id,
            }

    def load_closed_positions(self) -> list[dict[str, Any]]:
        try:
            rows = self._conn.execute("SELECT * FROM closed_positions").fetchall()
            return [dict(r) for r in rows]
        except sqlite3.OperationalError:
            return []

    def set_cooldown(self, pair_address: str, cooldown_until_utc: str) -> None:
        now = utc_now_iso()
        self._conn.execute(
            """
            INSERT OR REPLACE INTO cooldowns (pair_address, last_action_at_utc, cooldown_until_utc)
            VALUES (?, ?, ?)
            """,
            (pair_address, now, cooldown_until_utc),
        )
        self._conn.commit()

    def is_cooldown_active(self, pair_address: str, now_utc: str | None = None) -> bool:
        row = self._conn.execute(
            "SELECT cooldown_until_utc FROM cooldowns WHERE pair_address = ?",
            (pair_address,),
        ).fetchone()
        if not row:
            return False
        until = row["cooldown_until_utc"]
        now = now_utc or utc_now_iso()
        return until > now

    def active_open_position_count(self) -> int:
        """Backward-compatible alias for count_open_positions() (status='OPEN' only)."""
        return self.count_open_positions()

    def load_active_positions(self) -> list[dict[str, Any]]:
        """Backward-compatible alias for get_open_positions() (status='OPEN' only)."""
        return self.get_open_positions()

    def find_ghost_locks(self) -> list[dict[str, Any]]:
        """Locks that point to CLOSED or missing positions."""
        rows = self._conn.execute(
            """
            SELECT l.pair_address, l.position_id, l.locked_at_utc, a.status AS position_status
            FROM active_pair_locks l
            LEFT JOIN active_positions a ON a.position_id = l.position_id
            WHERE a.position_id IS NULL OR a.status != 'OPEN'
            """
        ).fetchall()
        return [dict(r) for r in rows]

    def repair_ghost_locks(self) -> dict[str, Any]:
        """
        Delete ghost locks (locks not pointing to OPEN positions). Audited by caller.
        """
        ghosts = self.find_ghost_locks()
        repaired = 0
        self._conn.execute("BEGIN")
        try:
            for g in ghosts:
                self._conn.execute(
                    "DELETE FROM active_pair_locks WHERE pair_address = ? AND position_id = ?",
                    (g["pair_address"], g["position_id"]),
                )
                repaired += 1
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        return {
            "ghost_lock_count": len(ghosts),
            "ghost_lock_repair_count": repaired,
            "repaired_locks": ghosts,
        }

    def count_open_also_in_closed_positions(self) -> int:
        """Invalid overlap: OPEN registry row also present in closed_positions."""
        try:
            row = self._conn.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM active_positions a
                JOIN closed_positions c ON c.position_id = a.position_id
                WHERE a.status = 'OPEN'
                """
            ).fetchone()
            return int(row["cnt"]) if row else 0
        except sqlite3.OperationalError:
            return 0

    def load_cooldowns(self) -> dict[str, str]:
        rows = self._conn.execute(
            "SELECT pair_address, cooldown_until_utc FROM cooldowns"
        ).fetchall()
        return {r["pair_address"]: r["cooldown_until_utc"] for r in rows}

    def processed_count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS cnt FROM processed_decisions").fetchone()
        return int(row["cnt"]) if row else 0

    def save_loop_checkpoint(self, loop_run_id: str, iteration: int, checkpoint_json: str) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO loop_checkpoints
            (loop_run_id, iteration, completed_at_utc, checkpoint_json)
            VALUES (?, ?, ?, ?)
            """,
            (loop_run_id, iteration, utc_now_iso(), checkpoint_json),
        )
        self._conn.commit()

    def status_summary(self) -> dict[str, Any]:
        processed = self.processed_count()
        open_pos = self.count_open_positions()
        locks = self._conn.execute("SELECT COUNT(*) AS cnt FROM active_pair_locks").fetchone()
        return {
            "processed_decisions_count": processed,
            "active_open_positions": open_pos,
            "open_positions_count": open_pos,
            "total_position_registry_rows": self.count_position_registry_rows(),
            "closed_rows_in_active_positions_count": self.count_closed_rows_in_registry(),
            "closed_positions_count": self.count_closed_positions_economic(),
            "active_pair_locks": int(locks["cnt"]) if locks else 0,
            "capacity_count_source": "status_OPEN",
            "db_path": str(self.path),
            "registry_backend": "sqlite",
            "schema_version": SCHEMA_VERSION,
            "position_registry_semantics": (
                "active_positions is a position registry; "
                "status='OPEN' is authoritative for open/tradable positions; "
                "closed_positions is canonical economic close state"
            ),
        }

    def repair_from_reconstruction(self, state: Any) -> None:
        """
        Reconcile OPEN identity from reconstruction.

        AE11G: never re-open a position that is economically closed or already CLOSED
        in the registry — closed_positions must not be overridden into OPEN.
        """
        for pos in state.open_positions:
            pid = pos.get("position_id")
            if not pid:
                continue
            if self.is_economically_closed(pid):
                continue
            status = self.get_position_status(pid)
            if status == "CLOSED":
                continue
            self.register_position(
                pid,
                pos["pair_address"],
                paper_order_id=pos.get("paper_order_id", ""),
                source_decision_id=pos.get("source_decision_id"),
                opened_at_utc=pos.get("opened_at_utc"),
            )
        for pair, until in state.cooldowns.items():
            self.set_cooldown(pair, until)
