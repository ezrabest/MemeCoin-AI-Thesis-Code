"""AE8 bounded SQLite queries with memory-safety tracking."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from app.context_intelligence.types import MemorySafetyStatus


_UNBOUNDED_SELECT_STAR = re.compile(
    r"SELECT\s+\*\s+FROM\s+(market_snapshots|signals|raw_provider_payloads|sentiment_records)\b",
    re.IGNORECASE,
)


@dataclass
class QueryStats:
    rows_scanned_by_table: dict[str, int] = field(default_factory=dict)
    rows_loaded_by_table: dict[str, int] = field(default_factory=dict)
    max_records_enforced: int = 0
    lookback_hours_enforced: float = 0.0
    queries_executed: list[str] = field(default_factory=list)

    def record_scan(self, table: str, count: int) -> None:
        self.rows_scanned_by_table[table] = self.rows_scanned_by_table.get(table, 0) + count

    def record_load(self, table: str, count: int) -> None:
        self.rows_loaded_by_table[table] = self.rows_loaded_by_table.get(table, 0) + count


def validate_bounded_sql(sql: str) -> bool:
    """Return True when SQL has WHERE and/or LIMIT guards."""
    normalized = " ".join(sql.split()).upper()
    if _UNBOUNDED_SELECT_STAR.search(sql):
        return False
    has_limit = "LIMIT" in normalized
    has_where = "WHERE" in normalized
    return has_limit or has_where


def assess_memory_safety(stats: QueryStats, queries: list[str]) -> str:
    for sql in queries:
        if not validate_bounded_sql(sql):
            return MemorySafetyStatus.BLOCKED_UNBOUNDED_QUERY.value
    if stats.max_records_enforced <= 0:
        return MemorySafetyStatus.BLOCKED_UNBOUNDED_QUERY.value
    return MemorySafetyStatus.PASS_BOUNDED_QUERIES.value


def fetch_context_seed_rows(
    conn: sqlite3.Connection,
    *,
    limit: int,
    lookback_hours: float,
    stats: QueryStats,
) -> list[dict[str, Any]]:
    """Fetch recent signal rows with bounded per-row snapshot/raw lookups."""
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=lookback_hours)).isoformat()
    stats.max_records_enforced = limit
    stats.lookback_hours_enforced = lookback_hours

    signal_sql = """
        SELECT s.id AS signal_id, s.timestamp AS signal_timestamp, s.coin_id,
               s.symbol, s.signal_type, s.score, s.confidence, s.reason,
               s.model_source, s.features_json,
               c.pair_address, c.chain, c.token_address, c.symbol AS coin_symbol,
               c.quote_symbol
        FROM signals s
        LEFT JOIN coins c ON c.id = s.coin_id
        WHERE s.timestamp >= ?
        ORDER BY s.id DESC
        LIMIT ?
    """
    stats.queries_executed.append(signal_sql)
    rows = conn.execute(signal_sql, (cutoff, limit)).fetchall()
    stats.record_scan("signals", len(rows))
    stats.record_load("signals", len(rows))

    bundles: list[dict[str, Any]] = []
    for row in rows:
        row_dict = dict(row)
        coin_id = row_dict.get("coin_id")
        pair_address = row_dict.get("pair_address")
        signal_ts = row_dict.get("signal_timestamp")
        signal_id = row_dict.get("signal_id")

        snapshot_row = None
        if coin_id is not None and signal_ts:
            snap_sql = """
                SELECT id, timestamp, provider, chain, pair_address, price, liquidity,
                       volume_24h, fdv, txns_buys, txns_sells, txns_total,
                       price_change_m5, price_change_h1, price_change_h6, price_change_h24,
                       whale_score, buy_ratio
                FROM market_snapshots
                WHERE coin_id = ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
            """
            stats.queries_executed.append(snap_sql)
            snap = conn.execute(snap_sql, (coin_id, signal_ts)).fetchone()
            stats.record_scan("market_snapshots", 1 if snap else 0)
            if snap:
                snapshot_row = dict(snap)
                stats.record_load("market_snapshots", 1)

        prior_snap_row = None
        if coin_id is not None and signal_ts:
            prior_sql = """
                SELECT id, timestamp, liquidity, volume_24h
                FROM market_snapshots
                WHERE coin_id = ? AND timestamp <= datetime(?, '-1 hour')
                ORDER BY timestamp DESC
                LIMIT 1
            """
            stats.queries_executed.append(prior_sql)
            prior = conn.execute(prior_sql, (coin_id, signal_ts)).fetchone()
            stats.record_scan("market_snapshots", 1 if prior else 0)
            if prior:
                prior_snap_row = dict(prior)
                stats.record_load("market_snapshots", 1)

        prior_6h_snap_row = None
        if coin_id is not None and signal_ts:
            prior6_sql = """
                SELECT id, timestamp, liquidity
                FROM market_snapshots
                WHERE coin_id = ? AND timestamp <= datetime(?, '-6 hours')
                ORDER BY timestamp DESC
                LIMIT 1
            """
            stats.queries_executed.append(prior6_sql)
            prior6 = conn.execute(prior6_sql, (coin_id, signal_ts)).fetchone()
            stats.record_scan("market_snapshots", 1 if prior6 else 0)
            if prior6:
                prior_6h_snap_row = dict(prior6)
                stats.record_load("market_snapshots", 1)

        raw_row = None
        if pair_address and signal_ts:
            raw_sql = """
                SELECT id, timestamp, provider, source_type, query, chain,
                       pair_address, symbol, payload_hash, payload_json_or_text
                FROM raw_provider_payloads
                WHERE pair_address = ? AND timestamp <= ?
                ORDER BY timestamp DESC
                LIMIT 1
            """
            stats.queries_executed.append(raw_sql)
            raw = conn.execute(raw_sql, (pair_address, signal_ts)).fetchone()
            stats.record_scan("raw_provider_payloads", 1 if raw else 0)
            if raw:
                raw_row = dict(raw)
                stats.record_load("raw_provider_payloads", 1)

        bundles.append(
            {
                "signal_row": {
                    "id": signal_id,
                    "timestamp": signal_ts,
                    "coin_id": coin_id,
                    "symbol": row_dict.get("symbol") or row_dict.get("coin_symbol"),
                    "signal_type": row_dict.get("signal_type"),
                    "score": row_dict.get("score"),
                    "confidence": row_dict.get("confidence"),
                    "reason": row_dict.get("reason"),
                    "model_source": row_dict.get("model_source"),
                    "features_json": row_dict.get("features_json"),
                },
                "snapshot_row": snapshot_row,
                "prior_snapshot_row": prior_snap_row,
                "prior_6h_snapshot_row": prior_6h_snap_row,
                "raw_payload_row": raw_row,
                "coin_row": {
                    "pair_address": pair_address,
                    "chain": row_dict.get("chain"),
                    "token_address": row_dict.get("token_address"),
                    "quote_symbol": row_dict.get("quote_symbol"),
                    "symbol": row_dict.get("symbol") or row_dict.get("coin_symbol"),
                },
            }
        )
    return bundles
