"""SQLite connection pragmas shared by runtime and audit code."""
from __future__ import annotations

import sqlite3
from typing import Any

SQLITE_BUSY_TIMEOUT_MS = 5000


def configure_sqlite_connection(conn: sqlite3.Connection) -> None:
    """Apply WAL mode and busy_timeout to every runtime SQLite connection."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
    conn.execute("PRAGMA foreign_keys=ON")


def get_sqlite_pragma_state(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return journal_mode and busy_timeout for connection verification."""
    journal = conn.execute("PRAGMA journal_mode").fetchone()
    busy = conn.execute("PRAGMA busy_timeout").fetchone()
    return {
        "journal_mode": journal[0] if journal else None,
        "busy_timeout_ms": int(busy[0]) if busy else None,
    }
