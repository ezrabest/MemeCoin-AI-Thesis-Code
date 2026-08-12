"""
SQLite-backed persistence — primary source for API reads and Gemini historical memory.
Thread-safe via a process lock + WAL mode. CSV/JSON logs remain as secondary audit trails.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Generator, TypeVar

from .sqlite_util import configure_sqlite_connection

log = logging.getLogger("database")

T = TypeVar("T")

DATA_DIR = Path(__file__).parent.parent / "data"
DB_PATH = Path(os.getenv("TRADER_DB_PATH", str(DATA_DIR / "trader.db")))
SETTINGS_PATH = DATA_DIR / "settings.json"

_db_lock = threading.Lock()

_DEFAULT_SETTINGS: dict[str, Any] = {
    "starting_capital": 10_000.0,
    "max_position_size_pct": 0.05,
    "stop_loss_pct": 0.08,
    "take_profit_pct": 0.15,
    "min_liquidity_usd": 5_000.0,
    "min_whale_score": 0.30,
    "llm_score_threshold": 0.50,
    "max_risk_score": 70,
    "auto_execution_enabled": True,
    "trading_mode": "DEMO",
    "prompt_behavior": "balanced",
    "paper_fee_bps": 150,
    "paper_trading_enabled": True,
    "demo_aggressive_enabled": False,
    "live_trading_enabled": False,
    "demo_acceptance_mode": False,
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS coins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    name TEXT,
    chain TEXT,
    pair_address TEXT UNIQUE,
    token_address TEXT,
    quote_symbol TEXT,
    provider TEXT DEFAULT 'dexscreener',
    provider_url TEXT,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL,
    latest_price REAL,
    latest_liquidity REAL,
    latest_volume_24h REAL,
    latest_fdv REAL,
    latest_pair_age TEXT,
    latest_whale_score REAL,
    raw_ref_id INTEGER
);

CREATE INDEX IF NOT EXISTS idx_coins_symbol ON coins(symbol);
CREATE INDEX IF NOT EXISTS idx_coins_chain ON coins(chain);

CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    coin_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    provider TEXT DEFAULT 'dexscreener',
    chain TEXT,
    pair_address TEXT,
    price REAL,
    liquidity REAL,
    volume_24h REAL,
    fdv REAL,
    txns_buys INTEGER,
    txns_sells INTEGER,
    txns_total INTEGER,
    price_change_m5 REAL,
    price_change_h1 REAL,
    price_change_h6 REAL,
    price_change_h24 REAL,
    whale_score REAL,
    buy_ratio REAL,
    source_query TEXT,
    filter_status TEXT,
    drop_reason TEXT,
    FOREIGN KEY (coin_id) REFERENCES coins(id)
);

CREATE INDEX IF NOT EXISTS idx_snapshots_coin_ts ON market_snapshots(coin_id, timestamp);

CREATE TABLE IF NOT EXISTS raw_provider_payloads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    provider TEXT NOT NULL,
    source_type TEXT,
    query TEXT,
    chain TEXT,
    pair_address TEXT,
    symbol TEXT,
    payload_json_or_text TEXT NOT NULL,
    payload_hash TEXT NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_payload_hash ON raw_provider_payloads(payload_hash);
CREATE INDEX IF NOT EXISTS idx_raw_ts ON raw_provider_payloads(timestamp);

CREATE TABLE IF NOT EXISTS signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    coin_id INTEGER,
    symbol TEXT,
    signal_type TEXT,
    score REAL,
    confidence REAL,
    reason TEXT,
    model_source TEXT DEFAULT 'engine.generate_signal',
    features_json TEXT,
    FOREIGN KEY (coin_id) REFERENCES coins(id)
);

CREATE TABLE IF NOT EXISTS whale_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    coin_id INTEGER,
    symbol TEXT,
    chain TEXT,
    pair_address TEXT,
    alert_type TEXT,
    whale_score REAL,
    liquidity REAL,
    volume REAL,
    tx_summary_json TEXT,
    is_real_wallet_level INTEGER DEFAULT 0,
    provider TEXT DEFAULT 'dexscreener',
    description TEXT,
    raw_ref_id INTEGER,
    FOREIGN KEY (coin_id) REFERENCES coins(id)
);

CREATE TABLE IF NOT EXISTS paper_trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    coin_id INTEGER,
    symbol TEXT,
    side TEXT,
    price REAL,
    amount REAL,
    value REAL,
    fee REAL,
    slippage REAL,
    pnl REAL,
    status TEXT,
    reason TEXT,
    decision_ref_id INTEGER,
    position_id INTEGER,
    chain TEXT,
    cluster_label TEXT,
    net_roi_pct REAL,
    source TEXT DEFAULT 'app_paper',
    FOREIGN KEY (coin_id) REFERENCES coins(id),
    FOREIGN KEY (decision_ref_id) REFERENCES gemini_decisions(id)
);

CREATE TABLE IF NOT EXISTS gemini_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    coin_id INTEGER,
    symbol TEXT,
    prompt_summary TEXT,
    input_context_json TEXT,
    gemini_response_json TEXT,
    action TEXT,
    confidence REAL,
    rationale TEXT,
    strategy_type TEXT,
    risk_score INTEGER,
    linked_trade_id INTEGER,
    outcome_pnl REAL,
    outcome_status TEXT,
    trigger_type TEXT,
    FOREIGN KEY (coin_id) REFERENCES coins(id)
);

CREATE INDEX IF NOT EXISTS idx_gemini_coin_ts ON gemini_decisions(coin_id, timestamp);

CREATE TABLE IF NOT EXISTS sentiment_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    source TEXT,
    title TEXT,
    url TEXT,
    text_excerpt TEXT,
    symbols_json TEXT,
    sentiment_score REAL,
    relevance_score REAL,
    raw_ref_id INTEGER
);

CREATE TABLE IF NOT EXISTS watchlist (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    symbol TEXT,
    chain TEXT,
    pair_address TEXT UNIQUE,
    provider TEXT DEFAULT 'manual',
    active INTEGER DEFAULT 1,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS pipeline_audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    scan_id TEXT,
    coin_id INTEGER,
    symbol TEXT,
    pair_address TEXT,
    stage TEXT NOT NULL,
    filter_status TEXT,
    drop_reason TEXT,
    whale_score REAL,
    alert_type TEXT,
    details_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_pipeline_scan ON pipeline_audit(scan_id, timestamp);
"""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Thread-safe SQLite connection with WAL mode and busy_timeout."""
    _ensure_data_dir()
    with _db_lock:
        conn = sqlite3.connect(str(DB_PATH), timeout=30.0)
        conn.row_factory = sqlite3.Row
        configure_sqlite_connection(conn)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}


def _rows_to_dicts(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    return [_row_to_dict(r) for r in rows]  # type: ignore[misc]


def backup_database() -> Path | None:
    """Create timestamped SQLite backup before schema changes."""
    if not DB_PATH.exists():
        return None
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = DATA_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / f"trader_{ts}.db"
    import shutil

    shutil.copy2(DB_PATH, dest)
    log.info("Database backup created: %s", dest)
    return dest


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _add_column_if_missing(conn: sqlite3.Connection, table: str, column: str, col_type: str) -> None:
    cols = _table_columns(conn, table)
    if column not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_type}")


def _migrate_schema(conn: sqlite3.Connection) -> None:
    """Additive idempotent migrations — preserve existing collected rows."""
    gemini_cols = _table_columns(conn, "gemini_decisions")
    audit_cols = _table_columns(conn, "pipeline_audit")
    needs_change = (
        "provider" not in gemini_cols
        or "model_source" not in gemini_cols
        or "decision_trace_id" not in audit_cols
        or "settings_hash" not in audit_cols
    )
    if needs_change:
        backup_database()

    if "provider" not in gemini_cols:
        conn.execute("ALTER TABLE gemini_decisions ADD COLUMN provider TEXT")
    if "model_source" not in gemini_cols:
        conn.execute("ALTER TABLE gemini_decisions ADD COLUMN model_source TEXT")

    _add_column_if_missing(conn, "pipeline_audit", "decision_trace_id", "TEXT")
    _add_column_if_missing(conn, "pipeline_audit", "settings_hash", "TEXT")
    _add_column_if_missing(conn, "pipeline_audit", "chain", "TEXT")
    _add_column_if_missing(conn, "pipeline_audit", "audit_reasons_json", "TEXT")
    _add_column_if_missing(conn, "pipeline_audit", "threshold_values_json", "TEXT")
    _add_column_if_missing(conn, "pipeline_audit", "model_metadata_json", "TEXT")
    _add_column_if_missing(conn, "pipeline_audit", "model_snapshot_price", "REAL")
    _add_column_if_missing(conn, "pipeline_audit", "current_execution_price", "REAL")
    _add_column_if_missing(conn, "pipeline_audit", "price_drift_from_model_pct", "REAL")


def init_db() -> None:
    _ensure_data_dir()
    with get_db() as conn:
        conn.executescript(SCHEMA)
        _migrate_schema(conn)
    log.info("SQLite initialized at %s", DB_PATH)


def backfill_paper_trades_from_csv() -> int:
    """
    Import legacy paper_trades_log.csv rows into SQLite when the DB table is empty.
    Preserves append-only CSV history; skips rows already present in SQLite.
    """
    import csv

    csv_path = DATA_DIR / "paper_trades_log.csv"
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return 0

    with get_db() as conn:
        existing_rows = conn.execute("SELECT COUNT(*) AS cnt FROM paper_trades").fetchone()
        if existing_rows and int(existing_rows["cnt"]) > 0:
            return 0

    imported = 0
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                coin_id = row.get("coin_id")
                if not coin_id and row.get("pair_address"):
                    coin = get_coin_by_pair_address(str(row["pair_address"]))
                    coin_id = coin["id"] if coin else None
                if not coin_id and row.get("symbol"):
                    coins = get_coins(limit=20, sort_by="last_seen")
                    match = next((c for c in coins if c.get("symbol") == row["symbol"]), None)
                    coin_id = match["id"] if match else None

                position_id = row.get("position_id")
                trade_id = insert_trade({
                    "timestamp": row.get("timestamp"),
                    "coin_id": int(coin_id) if coin_id else None,
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
                    "position_id": int(position_id) if position_id else None,
                    "chain": row.get("chain"),
                    "cluster_label": row.get("cluster_label"),
                    "net_roi_pct": row.get("net_roi_pct"),
                    "source": "csv_backfill",
                })
                if trade_id:
                    imported += 1
            except Exception as exc:
                log.warning("CSV trade backfill skipped row: %s", exc)

    if imported:
        log.info("Backfilled %d paper trades from CSV into SQLite", imported)
    return imported


def init_pool() -> None:
    init_db()
    _load_settings_file()
    try:
        from .execution.paper import _ensure_trade_csv_header

        _ensure_trade_csv_header()
    except Exception as exc:
        log.warning("Trade CSV header migration skipped: %s", exc)
    backfill_paper_trades_from_csv()
    log.info("Persistence: SQLite primary at %s", DB_PATH)


def _load_settings_file() -> dict[str, Any]:
    _ensure_data_dir()
    if SETTINGS_PATH.exists():
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
            return {**_DEFAULT_SETTINGS, **data}
    merged = dict(_DEFAULT_SETTINGS)
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)
    return merged


@contextmanager
def get_conn() -> Generator[sqlite3.Connection, None, None]:
    """Alias for get_db — backward compatible name."""
    with get_db() as conn:
        yield conn


def get_settings() -> dict[str, Any]:
    return _load_settings_file()


def upsert_setting(key: str, value: Any) -> None:
    settings = _load_settings_file()
    settings[key] = value
    with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)


def _payload_hash(payload: Any) -> str:
    text = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def insert_raw_payload(
    *,
    provider: str,
    payload: Any,
    source_type: str = "",
    query: str = "",
    chain: str = "",
    pair_address: str = "",
    symbol: str = "",
) -> int | None:
    """Archive raw provider response; skip duplicate hash within same second bucket."""
    text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    phash = _payload_hash(text)
    ts = _utcnow()
    with get_db() as conn:
        existing = conn.execute(
            "SELECT id FROM raw_provider_payloads WHERE payload_hash = ? LIMIT 1",
            (phash,),
        ).fetchone()
        if existing:
            return int(existing["id"])
        cur = conn.execute(
            """
            INSERT INTO raw_provider_payloads
            (timestamp, provider, source_type, query, chain, pair_address, symbol,
             payload_json_or_text, payload_hash)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (ts, provider, source_type, query, chain, pair_address, symbol, text, phash),
        )
        return int(cur.lastrowid)


def get_recent_raw_payloads(limit: int = 50, provider: str | None = None) -> list[dict[str, Any]]:
    with get_db() as conn:
        if provider:
            rows = conn.execute(
                """
                SELECT id, timestamp, provider, source_type, query, chain, pair_address, symbol,
                       substr(payload_json_or_text, 1, 500) AS payload_preview, payload_hash
                FROM raw_provider_payloads
                WHERE provider = ?
                ORDER BY id DESC LIMIT ?
                """,
                (provider, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT id, timestamp, provider, source_type, query, chain, pair_address, symbol,
                       substr(payload_json_or_text, 1, 500) AS payload_preview, payload_hash
                FROM raw_provider_payloads
                ORDER BY id DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
    return _rows_to_dicts(rows)


def upsert_coin(d: dict[str, Any]) -> dict[str, Any] | None:
    pair_address = (d.get("pair_address") or d.get("contract_address") or "").strip()
    if not pair_address:
        return None

    symbol = d.get("symbol") or "?"
    name = d.get("name") or d.get("base_symbol") or symbol
    chain = (d.get("chain") or "unknown").lower()
    ts = _utcnow()
    base_token = d.get("token_address") or (d.get("base_token_address") or "")

    with get_db() as conn:
        row = conn.execute(
            "SELECT id, first_seen_at FROM coins WHERE pair_address = ?",
            (pair_address,),
        ).fetchone()
        if row:
            coin_id = int(row["id"])
            first_seen = row["first_seen_at"]
            conn.execute(
                """
                UPDATE coins SET
                    symbol=?, name=?, chain=?, token_address=?, quote_symbol=?,
                    provider=?, provider_url=?, last_seen_at=?,
                    latest_price=?, latest_liquidity=?, latest_volume_24h=?,
                    latest_fdv=?, latest_pair_age=?, latest_whale_score=?, raw_ref_id=?
                WHERE id=?
                """,
                (
                    symbol,
                    name,
                    chain,
                    base_token or None,
                    d.get("quote_symbol"),
                    d.get("provider", "dexscreener"),
                    d.get("dex_url") or d.get("provider_url"),
                    ts,
                    d.get("price_usd") or d.get("latest_price"),
                    d.get("liquidity_usd") or d.get("latest_liquidity"),
                    d.get("volume_24h") or d.get("latest_volume_24h"),
                    d.get("fdv") or d.get("latest_fdv"),
                    d.get("pair_age") or d.get("latest_pair_age"),
                    d.get("whale_score") or d.get("latest_whale_score"),
                    d.get("raw_ref_id"),
                    coin_id,
                ),
            )
        else:
            cur = conn.execute(
                """
                INSERT INTO coins
                (symbol, name, chain, pair_address, token_address, quote_symbol,
                 provider, provider_url, first_seen_at, last_seen_at,
                 latest_price, latest_liquidity, latest_volume_24h, latest_fdv,
                 latest_pair_age, latest_whale_score, raw_ref_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    name,
                    chain,
                    pair_address,
                    base_token or None,
                    d.get("quote_symbol"),
                    d.get("provider", "dexscreener"),
                    d.get("dex_url") or d.get("provider_url"),
                    ts,
                    ts,
                    d.get("price_usd") or d.get("latest_price"),
                    d.get("liquidity_usd") or d.get("latest_liquidity"),
                    d.get("volume_24h") or d.get("latest_volume_24h"),
                    d.get("fdv") or d.get("latest_fdv"),
                    d.get("pair_age") or d.get("latest_pair_age"),
                    d.get("whale_score") or d.get("latest_whale_score"),
                    d.get("raw_ref_id"),
                ),
            )
            coin_id = int(cur.lastrowid)
            first_seen = ts

    return {
        "id": coin_id,
        "symbol": symbol,
        "name": name,
        "chain": chain,
        "pair_address": pair_address,
        "price_usd": d.get("price_usd"),
        "volume_24h": d.get("volume_24h"),
        "liquidity_usd": d.get("liquidity_usd"),
        "whale_score": d.get("whale_score"),
        "first_seen_at": first_seen,
        "last_seen_at": ts,
    }


def get_coins(
    limit: int = 50,
    sort_by: str = "whale_score",
    chain: str | None = None,
) -> list[dict[str, Any]]:
    sort_map = {
        "whale_score": "latest_whale_score DESC",
        "price": "latest_price DESC",
        "volume": "latest_volume_24h DESC",
        "liquidity": "latest_liquidity DESC",
        "symbol": "symbol ASC",
        "last_seen": "last_seen_at DESC",
    }
    order = sort_map.get(sort_by, "latest_whale_score DESC")
    with get_db() as conn:
        if chain:
            rows = conn.execute(
                f"""
                SELECT * FROM coins WHERE chain = ?
                ORDER BY {order} LIMIT ?
                """,
                (chain.lower(), limit),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT * FROM coins ORDER BY {order} LIMIT ?",
                (limit,),
            ).fetchall()
    result = []
    for r in rows:
        coin = _row_to_dict(r)
        if coin:
            coin["price_usd"] = coin.get("latest_price")
            coin["liquidity_usd"] = coin.get("latest_liquidity")
            coin["volume_24h"] = coin.get("latest_volume_24h")
            coin["whale_score"] = coin.get("latest_whale_score")
            result.append(coin)
    return result


def get_coin_by_id(coin_id: int) -> dict[str, Any] | None:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM coins WHERE id = ?", (coin_id,)).fetchone()
    coin = _row_to_dict(row)
    if coin:
        coin["price_usd"] = coin.get("latest_price")
        coin["liquidity_usd"] = coin.get("latest_liquidity")
        coin["volume_24h"] = coin.get("latest_volume_24h")
        coin["whale_score"] = coin.get("latest_whale_score")
    return coin


def get_coin_by_pair_address(pair_address: str) -> dict[str, Any] | None:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM coins WHERE pair_address = ?",
            (pair_address.strip(),),
        ).fetchone()
    coin = _row_to_dict(row)
    if coin:
        coin["price_usd"] = coin.get("latest_price")
        coin["liquidity_usd"] = coin.get("latest_liquidity")
        coin["volume_24h"] = coin.get("latest_volume_24h")
        coin["whale_score"] = coin.get("latest_whale_score")
    return coin


def insert_market_snapshot(d: dict[str, Any]) -> int | None:
    coin_id = d.get("coin_id")
    if not coin_id:
        return None
    ts = d.get("timestamp") or _utcnow()
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO market_snapshots
            (coin_id, timestamp, provider, chain, pair_address, price, liquidity,
             volume_24h, fdv, txns_buys, txns_sells, txns_total,
             price_change_m5, price_change_h1, price_change_h6, price_change_h24,
             whale_score, buy_ratio, source_query, filter_status, drop_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                coin_id,
                ts,
                d.get("provider", "dexscreener"),
                d.get("chain"),
                d.get("pair_address"),
                d.get("price") or d.get("price_usd"),
                d.get("liquidity") or d.get("liquidity_usd"),
                d.get("volume_24h"),
                d.get("fdv"),
                d.get("txns_buys"),
                d.get("txns_sells"),
                d.get("txns_total"),
                d.get("price_change_m5"),
                d.get("price_change_h1") or d.get("price_change_1h"),
                d.get("price_change_h6"),
                d.get("price_change_h24") or d.get("price_change_24h"),
                d.get("whale_score"),
                d.get("buy_ratio"),
                d.get("source_query"),
                d.get("filter_status"),
                d.get("drop_reason"),
            ),
        )
        return int(cur.lastrowid)


def get_market_snapshots(
    coin_id: int,
    limit: int = 200,
    since: str | None = None,
) -> list[dict[str, Any]]:
    with get_db() as conn:
        if since:
            rows = conn.execute(
                """
                SELECT * FROM market_snapshots
                WHERE coin_id = ? AND timestamp >= ?
                ORDER BY timestamp ASC LIMIT ?
                """,
                (coin_id, since, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM market_snapshots
                WHERE coin_id = ?
                ORDER BY timestamp DESC LIMIT ?
                """,
                (coin_id, limit),
            ).fetchall()
            rows = list(reversed(rows))
    return _rows_to_dicts(rows)


def insert_signal(d: dict[str, Any]) -> int | None:
    ts = d.get("timestamp") or _utcnow()
    features = d.get("features_json") or d.get("features")
    if isinstance(features, dict):
        features = json.dumps(features)
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO signals
            (timestamp, coin_id, symbol, signal_type, score, confidence, reason,
             model_source, features_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                d.get("coin_id"),
                d.get("symbol"),
                d.get("signal_type") or d.get("action"),
                d.get("score") or d.get("probability_up"),
                d.get("confidence") or d.get("probability_up"),
                d.get("reason") or d.get("explanation"),
                d.get("model_source", "engine.generate_signal"),
                features,
            ),
        )
        return int(cur.lastrowid)


def get_signals(
    limit: int = 50,
    action: str | None = None,
    coin_id: int | None = None,
) -> list[dict[str, Any]]:
    with get_db() as conn:
        clauses: list[str] = []
        params: list[Any] = []
        if coin_id is not None:
            clauses.append("coin_id = ?")
            params.append(coin_id)
        if action:
            clauses.append("signal_type = ?")
            params.append(action)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM signals {where} ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()
    return _rows_to_dicts(rows)


def insert_whale_alert(d: dict[str, Any]) -> int | None:
    ts = d.get("timestamp") or _utcnow()
    tx_summary = d.get("tx_summary_json")
    if isinstance(tx_summary, dict):
        tx_summary = json.dumps(tx_summary)
    elif not tx_summary:
        tx_summary = json.dumps({
            "tx_count": d.get("tx_count"),
            "volume_usd": d.get("volume_usd"),
            "price_impact_pct": d.get("price_impact_pct"),
            "note": "aggregate pool-level flow — not wallet-level whale tracking",
        })
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO whale_alerts
            (timestamp, coin_id, symbol, chain, pair_address, alert_type, whale_score,
             liquidity, volume, tx_summary_json, is_real_wallet_level, provider,
             description, raw_ref_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                d.get("coin_id"),
                d.get("symbol"),
                d.get("chain"),
                d.get("pair_address"),
                d.get("alert_type"),
                d.get("whale_score"),
                d.get("liquidity") or d.get("liquidity_usd"),
                d.get("volume") or d.get("volume_usd"),
                tx_summary,
                1 if d.get("is_real_wallet_level") else 0,
                d.get("provider", "dexscreener"),
                d.get("description"),
                d.get("raw_ref_id"),
            ),
        )
        return int(cur.lastrowid)


def get_whale_alerts(limit: int = 50, coin_id: int | None = None) -> list[dict[str, Any]]:
    with get_db() as conn:
        if coin_id is not None:
            rows = conn.execute(
                "SELECT * FROM whale_alerts WHERE coin_id = ? ORDER BY id DESC LIMIT ?",
                (coin_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM whale_alerts ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    result = []
    for r in rows:
        d = _row_to_dict(r)
        if d:
            d["is_real_wallet_level"] = bool(d.get("is_real_wallet_level"))
            d["terminology"] = (
                "wallet-level whale" if d["is_real_wallet_level"]
                else "aggregate whale-like market flow (pool-level)"
            )
            result.append(d)
    return result


def insert_trade(d: dict[str, Any]) -> int | None:
    """App-generated paper trade — never mixed with market whale events."""
    ts = d.get("timestamp") or _utcnow()
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO paper_trades
            (timestamp, coin_id, symbol, side, price, amount, value, fee, slippage,
             pnl, status, reason, decision_ref_id, position_id, chain,
             cluster_label, net_roi_pct, source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                d.get("coin_id"),
                d.get("symbol"),
                d.get("side"),
                d.get("price") or d.get("fill_price"),
                d.get("amount") or d.get("quantity"),
                d.get("value") or d.get("notional_usd"),
                d.get("fee") or d.get("total_fees"),
                d.get("slippage") or d.get("swap_fee"),
                d.get("pnl") or d.get("realized_pnl"),
                d.get("status", "filled"),
                d.get("reason") or d.get("reason_code"),
                d.get("decision_ref_id"),
                d.get("position_id"),
                d.get("chain"),
                d.get("cluster_label"),
                d.get("net_roi_pct"),
                d.get("source", "app_paper"),
            ),
        )
        return int(cur.lastrowid)


def get_trades(limit: int = 100, coin_id: int | None = None) -> list[dict[str, Any]]:
    with get_db() as conn:
        if coin_id is not None:
            rows = conn.execute(
                """
                SELECT * FROM paper_trades WHERE coin_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (coin_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM paper_trades ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return _rows_to_dicts(rows)


def insert_gemini_decision(d: dict[str, Any]) -> int | None:
    ts = d.get("timestamp") or _utcnow()
    ctx = d.get("input_context_json")
    resp = d.get("gemini_response_json")
    if isinstance(ctx, dict):
        ctx = json.dumps(ctx)
    if isinstance(resp, dict):
        resp = json.dumps(resp)
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO gemini_decisions
            (timestamp, coin_id, symbol, prompt_summary, input_context_json,
             gemini_response_json, action, confidence, rationale, strategy_type,
             risk_score, linked_trade_id, outcome_pnl, outcome_status, trigger_type,
             provider, model_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                d.get("coin_id"),
                d.get("symbol"),
                d.get("prompt_summary"),
                ctx,
                resp,
                d.get("action") or d.get("decision"),
                d.get("confidence"),
                d.get("rationale") or d.get("reasoning"),
                d.get("strategy_type"),
                d.get("risk_score"),
                d.get("linked_trade_id"),
                d.get("outcome_pnl"),
                d.get("outcome_status"),
                d.get("trigger_type"),
                d.get("provider"),
                d.get("model_source"),
            ),
        )
        return int(cur.lastrowid)


def get_gemini_decisions(
    limit: int = 50,
    coin_id: int | None = None,
) -> list[dict[str, Any]]:
    with get_db() as conn:
        if coin_id is not None:
            rows = conn.execute(
                """
                SELECT * FROM gemini_decisions WHERE coin_id = ?
                ORDER BY id DESC LIMIT ?
                """,
                (coin_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM gemini_decisions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
    result = []
    for r in rows:
        d = _row_to_dict(r)
        if d and d.get("input_context_json"):
            try:
                d["input_context"] = json.loads(d["input_context_json"])
            except json.JSONDecodeError:
                pass
        if d and d.get("gemini_response_json"):
            try:
                d["gemini_response"] = json.loads(d["gemini_response_json"])
            except json.JSONDecodeError:
                pass
        result.append(d)
    return result


def update_gemini_decision_outcome(
    decision_id: int,
    *,
    linked_trade_id: int | None = None,
    outcome_pnl: float | None = None,
    outcome_status: str | None = None,
) -> None:
    with get_db() as conn:
        conn.execute(
            """
            UPDATE gemini_decisions SET
                linked_trade_id = COALESCE(?, linked_trade_id),
                outcome_pnl = COALESCE(?, outcome_pnl),
                outcome_status = COALESCE(?, outcome_status)
            WHERE id = ?
            """,
            (linked_trade_id, outcome_pnl, outcome_status, decision_id),
        )


def insert_sentiment_record(d: dict[str, Any]) -> int | None:
    ts = d.get("timestamp") or _utcnow()
    symbols = d.get("symbols_json")
    if isinstance(symbols, list):
        symbols = json.dumps(symbols)
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO sentiment_records
            (timestamp, source, title, url, text_excerpt, symbols_json,
             sentiment_score, relevance_score, raw_ref_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                d.get("source"),
                d.get("title"),
                d.get("url"),
                d.get("text_excerpt") or d.get("title"),
                symbols,
                d.get("sentiment_score"),
                d.get("relevance_score"),
                d.get("raw_ref_id"),
            ),
        )
        return int(cur.lastrowid)


def get_sentiment_records(limit: int = 50) -> list[dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM sentiment_records ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return _rows_to_dicts(rows)


def insert_pipeline_audit(d: dict[str, Any]) -> int | None:
    ts = d.get("timestamp") or _utcnow()
    details = d.get("details_json")
    if isinstance(details, dict):
        details = json.dumps(details)
    audit_reasons = d.get("audit_reasons_json")
    if isinstance(audit_reasons, list):
        audit_reasons = json.dumps(audit_reasons)
    threshold_values = d.get("threshold_values_json")
    if isinstance(threshold_values, dict):
        threshold_values = json.dumps(threshold_values)
    model_metadata = d.get("model_metadata_json")
    if isinstance(model_metadata, dict):
        model_metadata = json.dumps(model_metadata)
    with get_db() as conn:
        cur = conn.execute(
            """
            INSERT INTO pipeline_audit
            (timestamp, scan_id, coin_id, symbol, pair_address, stage, filter_status,
             drop_reason, whale_score, alert_type, details_json,
             decision_trace_id, settings_hash, chain, audit_reasons_json,
             threshold_values_json, model_metadata_json, model_snapshot_price,
             current_execution_price, price_drift_from_model_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                d.get("scan_id"),
                d.get("coin_id"),
                d.get("symbol"),
                d.get("pair_address"),
                d.get("stage"),
                d.get("filter_status"),
                d.get("drop_reason"),
                d.get("whale_score"),
                d.get("alert_type"),
                details,
                d.get("decision_trace_id"),
                d.get("settings_hash"),
                d.get("chain"),
                audit_reasons,
                threshold_values,
                model_metadata,
                d.get("model_snapshot_price"),
                d.get("current_execution_price"),
                d.get("price_drift_from_model_pct"),
            ),
        )
        return int(cur.lastrowid)


def get_pipeline_audit(
    scan_id: str | None = None,
    limit: int = 200,
    filter_status: str | None = None,
) -> list[dict[str, Any]]:
    with get_db() as conn:
        clauses: list[str] = []
        params: list[Any] = []
        if scan_id:
            clauses.append("scan_id = ?")
            params.append(scan_id)
        if filter_status:
            clauses.append("filter_status = ?")
            params.append(filter_status)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        rows = conn.execute(
            f"SELECT * FROM pipeline_audit {where} ORDER BY id DESC LIMIT ?",
            params,
        ).fetchall()
    return _rows_to_dicts(rows)


def count_gemini_decisions_by_action(action: str | None = None) -> int:
    with get_db() as conn:
        if action:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM gemini_decisions WHERE action = ?",
                (action,),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM gemini_decisions").fetchone()
    return int(row["cnt"]) if row else 0


def get_collection_debug_status() -> dict[str, Any]:
    """Snapshot for verifying headless collection and LLM bypass."""
    from .llm_config import get_llm_runtime_status

    stats = get_storage_stats()
    with get_db() as conn:
        newest_snapshot = conn.execute(
            "SELECT MAX(timestamp) AS latest FROM market_snapshots"
        ).fetchone()
        newest_raw = conn.execute(
            "SELECT MAX(timestamp) AS latest FROM raw_provider_payloads"
        ).fetchone()
    return {
        "storage": stats,
        "newest_market_snapshot_timestamp": newest_snapshot["latest"] if newest_snapshot else None,
        "newest_raw_payload_timestamp": newest_raw["latest"] if newest_raw else None,
        "signal_count": stats.get("signals", {}).get("rows", 0),
        "whale_alert_count": stats.get("whale_alerts", {}).get("rows", 0),
        "raw_payload_count": stats.get("raw_provider_payloads", {}).get("rows", 0),
        "market_snapshot_count": stats.get("market_snapshots", {}).get("rows", 0),
        "llm_skipped_count_db": count_gemini_decisions_by_action("SKIPPED"),
        "gemini_decision_count_db": count_gemini_decisions_by_action(),
        "llm_runtime": get_llm_runtime_status(),
    }


def get_storage_stats() -> dict[str, dict[str, Any]]:
    tables = [
        "coins",
        "market_snapshots",
        "raw_provider_payloads",
        "signals",
        "whale_alerts",
        "paper_trades",
        "gemini_decisions",
        "sentiment_records",
        "pipeline_audit",
        "watchlist",
    ]
    stats: dict[str, dict[str, Any]] = {}
    with get_db() as conn:
        for table in tables:
            try:
                if table == "coins":
                    row = conn.execute(
                        f"SELECT COUNT(*) AS cnt, MAX(last_seen_at) AS latest FROM {table}"
                    ).fetchone()
                elif table == "watchlist":
                    row = conn.execute(
                        f"SELECT COUNT(*) AS cnt, MAX(created_at) AS latest FROM {table}"
                    ).fetchone()
                else:
                    row = conn.execute(
                        f"SELECT COUNT(*) AS cnt, MAX(timestamp) AS latest FROM {table}"
                    ).fetchone()
                stats[table] = {
                    "rows": int(row["cnt"]) if row else 0,
                    "latest": row["latest"] if row else None,
                }
            except sqlite3.OperationalError:
                stats[table] = {"rows": 0, "latest": None}
    return stats


def derive_chart_candles(
    coin_id: int,
    interval: str = "1m",
    limit: int = 120,
) -> list[dict[str, Any]]:
    """
    Derive approximate OHLC from stored snapshots (labeled snapshot-derived).
    interval: 1m, 5m, 15m, 1h, raw
    """
    snapshots = get_market_snapshots(coin_id, limit=2000)
    if not snapshots:
        return []

    if interval == "raw":
        return [
            {
                "timestamp": s["timestamp"],
                "open": s["price"],
                "high": s["price"],
                "low": s["price"],
                "close": s["price"],
                "volume": s.get("volume_24h"),
                "liquidity": s.get("liquidity"),
                "derived": True,
            }
            for s in snapshots[-limit:]
        ]

    minutes_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60}
    bucket_min = minutes_map.get(interval, 1)

    def parse_ts(ts: str) -> datetime:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))

    buckets: dict[str, list[dict]] = {}
    for s in snapshots:
        if s.get("price") is None:
            continue
        dt = parse_ts(s["timestamp"])
        epoch = int(dt.timestamp())
        bucket_epoch = (epoch // (bucket_min * 60)) * (bucket_min * 60)
        key = datetime.fromtimestamp(bucket_epoch, tz=timezone.utc).isoformat()
        buckets.setdefault(key, []).append(s)

    candles: list[dict[str, Any]] = []
    for ts_key in sorted(buckets.keys()):
        pts = buckets[ts_key]
        prices = [float(p["price"]) for p in pts if p.get("price") is not None]
        if not prices:
            continue
        last = pts[-1]
        candles.append({
            "timestamp": ts_key,
            "open": prices[0],
            "high": max(prices),
            "low": min(prices),
            "close": prices[-1],
            "volume": last.get("volume_24h"),
            "liquidity": last.get("liquidity"),
            "snapshot_count": len(pts),
            "derived": True,
            "derivation_note": f"snapshot-derived OHLC ({interval} buckets)",
        })

    return candles[-limit:]


def get_coin_detail(coin_id: int) -> dict[str, Any] | None:
    coin = get_coin_by_id(coin_id)
    if not coin:
        return None
    return {
        "coin": coin,
        "snapshot_count": len(get_market_snapshots(coin_id, limit=5000)),
        "latest_snapshot": get_market_snapshots(coin_id, limit=1)[-1] if get_market_snapshots(coin_id, limit=1) else None,
        "recent_whale_alerts": get_whale_alerts(limit=10, coin_id=coin_id),
        "recent_signals": get_signals(limit=10, coin_id=coin_id),
        "recent_gemini_decisions": get_gemini_decisions(limit=10, coin_id=coin_id),
        "recent_paper_trades": get_trades(limit=10, coin_id=coin_id),
        "raw_payload_count": _count_raw_for_coin(coin),
    }


def _count_raw_for_coin(coin: dict[str, Any]) -> int:
    pair = coin.get("pair_address") or ""
    symbol = coin.get("symbol") or ""
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS cnt FROM raw_provider_payloads
            WHERE pair_address = ? OR symbol = ?
            """,
            (pair, symbol),
        ).fetchone()
    return int(row["cnt"]) if row else 0


def normalize_coin_for_trade(coin: dict[str, Any]) -> dict[str, Any]:
    """Map SQLite coin record fields to paper trader expectations."""
    normalized = dict(coin)
    if normalized.get("id") is not None and normalized.get("coin_id") is None:
        normalized["coin_id"] = normalized["id"]
    if normalized.get("price_usd") is None:
        normalized["price_usd"] = normalized.get("latest_price")
    return normalized


def get_dashboard_summary() -> dict[str, Any]:
    from .execution.paper import get_paper_trader

    paper = get_paper_trader()
    roi = paper.net_roi_summary()
    wallet = paper.get_wallet_summary()
    audit = paper._trade_audit_summary()
    stats = get_storage_stats()
    valid_sells = [
        t for t in audit.get("valid_row_details", [])
        if str(t.get("side")).lower() == "sell"
    ]
    wins = sum(1 for t in valid_sells if float(t.get("realized_pnl") or 0) > 0)
    win_rate = (wins / len(valid_sells)) if valid_sells else 0.0
    with get_db() as conn:
        sig_row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM signals WHERE signal_type = 'BUY'"
        ).fetchone()
        watch_row = conn.execute(
            "SELECT COUNT(*) AS cnt FROM signals WHERE signal_type = 'WATCH'"
        ).fetchone()
        alert_row = conn.execute("SELECT COUNT(*) AS cnt FROM whale_alerts").fetchone()
    top = get_coins(limit=5, sort_by="whale_score")
    starting = float(wallet.get("starting_capital", 0))
    equity = float(wallet.get("total_equity_usd", 0))
    portfolio_roi = (equity - starting) / starting if starting > 0 else 0.0
    return {
        "total_equity": wallet["total_equity_usd"],
        "cash_usd": wallet["cash_usd"],
        "positions_value_usd": wallet["positions_value_usd"],
        "total_pnl": roi["total_net_pnl"],
        "pnl_pct": round(portfolio_roi, 6),
        "portfolio_roi_pct": round(portfolio_roi, 6),
        "avg_closed_trade_roi_pct": roi.get("avg_net_roi_pct", 0.0),
        "cumulative_total_fees": wallet["cumulative_total_fees"],
        "open_positions": wallet["open_positions_count"],
        "closed_trades": roi["trade_count"],
        "trading_mode": wallet["trading_mode"],
        "win_rate": round(win_rate, 4),
        "invalid_trade_rows_excluded": audit.get("invalid_rows", 0),
        "raw_aggregate_metrics": audit.get("raw_aggregate"),
        "valid_aggregate_metrics": audit.get("valid_aggregate"),
        "paper_state_contaminated": audit.get("paper_state_contaminated", False),
        "total_signals": stats.get("signals", {}).get("rows", 0),
        "buy_signals": int(sig_row["cnt"]) if sig_row else 0,
        "watch_signals": int(watch_row["cnt"]) if watch_row else 0,
        "active_alerts": int(alert_row["cnt"]) if alert_row else 0,
        "top_coins": top,
        "storage": stats,
    }


def open_position(d: dict) -> dict | None:
    from .execution.paper import get_paper_trader

    coin = normalize_coin_for_trade(d)
    settings = get_settings()
    return get_paper_trader().open_position(
        coin,
        size_usd=d.get("size_usd"),
        cluster_label=d.get("cluster_label", "SOCIALLY_MOTIVATED"),
        settings=settings,
        reason_code=d.get("reason_code", "MANUAL_BUY"),
    )


def close_position(pos_id: int, close_price: float) -> dict | None:
    from .execution.paper import get_paper_trader

    return get_paper_trader().close_position(
        pos_id,
        close_price,
        reason_code="MANUAL_SELL",
    )


def get_positions(status: str | None = None) -> list[dict]:
    from .execution.paper import get_paper_trader

    return get_paper_trader().get_positions(status)
