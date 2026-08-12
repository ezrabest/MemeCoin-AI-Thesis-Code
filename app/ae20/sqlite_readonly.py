"""Strict read-only SQLite access for AE20 (trader.db).

Hard rule: only URI mode=ro + PRAGMA query_only. Never open writable connections.
"""

from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

WRITE_SQL_RE = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|VACUUM|REPLACE|ATTACH|DETACH)\b",
    re.IGNORECASE,
)


class ReadOnlySqliteError(RuntimeError):
    """Raised when write SQL is attempted against a read-only AE20 connection."""


def build_readonly_sqlite_uri(db_path: Path) -> str:
    return f"file:{Path(db_path).resolve().as_posix()}?mode=ro"


def open_readonly_sqlite(db_path: Path) -> tuple[sqlite3.Connection, dict[str, Any]]:
    """Open trader.db (or any sqlite file) in explicit read-only URI mode.

    Uses only:
      sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    Then enforces PRAGMA query_only = TRUE.
    """
    path = Path(db_path).resolve()
    uri = build_readonly_sqlite_uri(path)
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = TRUE")
    audit = {
        "sqlite_open_mode": "READ_ONLY_URI_MODE_RO",
        "sqlite_uri_used": True,
        "sqlite_query_only_pragma_enabled": True,
        "sqlite_write_sql_detected": False,
        "sqlite_uri": uri,
        "db_path": str(path),
        "raw_mutation": False,
        "db_mutation": False,
        "trader_db_mutation": False,
    }
    return conn, audit


def assert_readonly_sql(sql: str) -> None:
    """Reject write SQL. Detection does not rely on mutating the DB."""
    if WRITE_SQL_RE.match(sql or ""):
        raise ReadOnlySqliteError(f"Write SQL forbidden on AE20 read-only connection: {sql[:120]}")


class ReadOnlyConnection:
    """Thin wrapper that refuses write SQL before execution."""

    def __init__(self, conn: sqlite3.Connection, audit: dict[str, Any]):
        self._conn = conn
        self.audit = audit
        self._write_sql_detected = False

    def execute(self, sql: str, parameters: Any = ()) -> sqlite3.Cursor:
        try:
            assert_readonly_sql(sql)
        except ReadOnlySqliteError:
            self._write_sql_detected = True
            self.audit["sqlite_write_sql_detected"] = True
            raise
        return self._conn.execute(sql, parameters)

    def executemany(self, sql: str, seq_of_parameters: Any) -> sqlite3.Cursor:
        try:
            assert_readonly_sql(sql)
        except ReadOnlySqliteError:
            self._write_sql_detected = True
            self.audit["sqlite_write_sql_detected"] = True
            raise
        return self._conn.executemany(sql, seq_of_parameters)

    def close(self) -> None:
        self._conn.close()

    @property
    def write_sql_detected(self) -> bool:
        return self._write_sql_detected


@contextmanager
def readonly_sqlite(db_path: Path) -> Iterator[ReadOnlyConnection]:
    conn, audit = open_readonly_sqlite(db_path)
    wrapped = ReadOnlyConnection(conn, audit)
    try:
        yield wrapped
    finally:
        wrapped.close()
