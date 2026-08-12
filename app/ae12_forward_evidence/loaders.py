"""Load AE11/AE6/paper JSONL sources and local SQLite market snapshots (read-only)."""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.ae12_forward_evidence.types import parse_ts


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield line_no, obj


def discover_glob(root: Path, pattern: str) -> list[Path]:
    base = root / "data"
    paths = sorted(base.glob(pattern))
    return [p for p in paths if p.is_file()]


@dataclass
class SourceBundle:
    opportunity_files: list[Path] = field(default_factory=list)
    trade_decision_files: list[Path] = field(default_factory=list)
    runtime_event_files: list[Path] = field(default_factory=list)
    ae6_files: list[Path] = field(default_factory=list)
    paper_order_files: list[Path] = field(default_factory=list)
    paper_position_files: list[Path] = field(default_factory=list)
    paper_trade_files: list[Path] = field(default_factory=list)
    live_dry_run_files: list[Path] = field(default_factory=list)
    ae9_audit_files: list[Path] = field(default_factory=list)
    db_path: Path | None = None


def discover_sources(project_root: Path, db_path: Path | None = None) -> SourceBundle:
    root = Path(project_root)
    db = Path(db_path) if db_path else root / "data" / "trader.db"
    return SourceBundle(
        opportunity_files=discover_glob(root, "runtime_paper_loop/ae11_opportunity_capture_*.jsonl"),
        trade_decision_files=discover_glob(root, "runtime_paper_loop/ae11_trade_decisions_*.jsonl"),
        runtime_event_files=discover_glob(root, "runtime_paper_loop/ae11_runtime_events_*.jsonl"),
        ae6_files=discover_glob(root, "decision_records/ae6_decisions_*.jsonl"),
        paper_order_files=discover_glob(root, "paper_trading/paper_orders_*.jsonl"),
        paper_position_files=discover_glob(root, "paper_trading/paper_positions_*.jsonl"),
        paper_trade_files=discover_glob(root, "paper_trading/paper_trades_*.jsonl"),
        live_dry_run_files=discover_glob(root, "execution/live_dry_run_orders_*.jsonl"),
        ae9_audit_files=discover_glob(root, "llm_audit/ae9_llm_audit_records_*.jsonl"),
        db_path=db if db.is_file() else None,
    )


def index_by_keys(
    files: list[Path],
    *,
    project_root: Path,
    key_fields: tuple[str, ...],
    max_rows: int | None = None,
    only_keys: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build first-seen index by any of the key fields (decision_id / candidate_id / etc.)."""
    import re

    out: dict[str, dict[str, Any]] = {}
    count = 0
    remaining = set(only_keys) if only_keys is not None else None
    # Extract quoted UUID-like / hex ids without full JSON when filtering
    key_extractors = [
        re.compile(rf'"{kf}"\s*:\s*"([^"]+)"') for kf in key_fields
    ]
    if "decision_id" in key_fields:
        key_extractors.append(re.compile(r'"source_decision_id"\s*:\s*"([^"]+)"'))

    for path in files:
        rel = _rel(path, project_root)
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                if only_keys is not None:
                    candidates = []
                    for rx in key_extractors:
                        m = rx.search(line)
                        if m and m.group(1) in only_keys:
                            candidates.append(m.group(1))
                    if not candidates:
                        continue
                    # If all candidate keys already indexed, skip expensive parse
                    if all(c in out for c in candidates) and (
                        remaining is None or not any(c in remaining for c in candidates)
                    ):
                        continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(obj, dict):
                    continue
                wrapped = {
                    "_source_file": rel,
                    "_source_line_no": line_no,
                    **obj,
                }
                for kf in key_fields:
                    val = obj.get(kf)
                    if val is None and kf == "decision_id":
                        val = obj.get("source_decision_id")
                    if val is None:
                        continue
                    key = str(val)
                    if only_keys is not None and key not in only_keys:
                        continue
                    if key not in out:
                        out[key] = wrapped
                        count += 1
                        if remaining is not None:
                            remaining.discard(key)
                if max_rows is not None and count >= max_rows:
                    return out
                if remaining is not None and len(remaining) == 0:
                    return out
    return out


def index_nested_lookup(
    files: list[Path],
    *,
    project_root: Path,
    extract_keys: Any,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in files:
        rel = _rel(path, project_root)
        for line_no, obj in iter_jsonl(path):
            keys = extract_keys(obj)
            wrapped = {"_source_file": rel, "_source_line_no": line_no, **obj}
            for key in keys:
                if key and key not in out:
                    out[str(key)] = wrapped
    return out


def load_opportunity_rows(
    files: list[Path],
    *,
    project_root: Path,
    max_rows: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in files:
        rel = _rel(path, project_root)
        for line_no, obj in iter_jsonl(path):
            rows.append(
                {
                    "_source_file": rel,
                    "_source_line_no": line_no,
                    **obj,
                }
            )
            if max_rows is not None and len(rows) >= max_rows:
                return rows
    return rows


def build_paper_linkage_index(
    orders: list[Path],
    positions: list[Path],
    trades: list[Path],
    *,
    project_root: Path,
) -> dict[str, dict[str, Any]]:
    """Index paper artifacts by candidate_id and source_decision_id."""
    by_key: dict[str, dict[str, Any]] = defaultdict(dict)

    def _merge(obj: dict[str, Any], kind: str) -> None:
        keys = []
        for kf in ("candidate_id", "source_decision_id", "decision_id"):
            v = obj.get(kf)
            if v:
                keys.append(str(v))
        for key in keys:
            slot = by_key[key]
            if kind == "order":
                slot["paper_order_id"] = obj.get("paper_order_id") or slot.get("paper_order_id")
                slot["order"] = obj
            elif kind == "position":
                slot["paper_position_id"] = (
                    obj.get("position_id") or obj.get("paper_position_id") or slot.get("paper_position_id")
                )
                slot["position"] = obj
            elif kind == "trade":
                slot["paper_trade_id"] = (
                    obj.get("close_event_id")
                    or obj.get("paper_trade_id")
                    or obj.get("economic_close_key")
                    or slot.get("paper_trade_id")
                )
                slot["trade"] = obj
            slot["was_traded"] = True

    for path in orders:
        for _, obj in iter_jsonl(path):
            _merge(obj, "order")
    for path in positions:
        for _, obj in iter_jsonl(path):
            _merge(obj, "position")
    for path in trades:
        for _, obj in iter_jsonl(path):
            _merge(obj, "trade")
    return dict(by_key)


@dataclass
class MarketSnapshotStore:
    """Read-only market_snapshots accessor with schema discovery and caches."""

    db_path: Path | None
    pair_col: str | None = None
    ts_col: str | None = None
    price_col: str | None = None
    coin_col: str | None = None
    available: bool = False
    unavailable_reason: str = ""
    _conn: sqlite3.Connection | None = None
    _latest_by_pair: dict[str, datetime | None] = field(default_factory=dict)
    _series_by_pair: dict[str, list[tuple[datetime, float]]] = field(default_factory=dict)
    _window_cache: dict[tuple[str, str, str], list[tuple[datetime, float]]] = field(default_factory=dict)
    global_latest_ts: datetime | None = None
    _prefetched: bool = False

    def open(self) -> None:
        if self.db_path is None or not Path(self.db_path).is_file():
            self.available = False
            self.unavailable_reason = "TRADER_DB_MISSING"
            return
        try:
            uri = f"file:{Path(self.db_path).resolve().as_posix()}?mode=ro"
            self._conn = sqlite3.connect(uri, uri=True, timeout=30.0)
            self._conn.row_factory = sqlite3.Row
            # Faster read-only scans
            try:
                self._conn.execute("PRAGMA query_only=ON")
            except Exception:
                pass
            tables = {
                r[0]
                for r in self._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "market_snapshots" not in tables:
                self.available = False
                self.unavailable_reason = "MARKET_SNAPSHOTS_TABLE_MISSING"
                return
            cols = {r[1] for r in self._conn.execute("PRAGMA table_info(market_snapshots)").fetchall()}
            self.pair_col = next((c for c in ("pair_address", "pair", "address") if c in cols), None)
            self.ts_col = next((c for c in ("timestamp", "ts", "created_at", "ingested_at") if c in cols), None)
            self.price_col = next((c for c in ("price", "close", "price_usd", "current_price") if c in cols), None)
            self.coin_col = "coin_id" if "coin_id" in cols else None
            if not self.pair_col or not self.ts_col or not self.price_col:
                self.available = False
                self.unavailable_reason = "MARKET_SNAPSHOTS_SCHEMA_INCOMPLETE"
                return
            row = self._conn.execute(
                f"SELECT MAX({self.ts_col}) AS mx FROM market_snapshots"
            ).fetchone()
            self.global_latest_ts = parse_ts(row["mx"] if row else None)
            self.available = True
            self.unavailable_reason = ""
        except Exception as exc:  # noqa: BLE001 — fail soft for audit
            self.available = False
            self.unavailable_reason = f"MARKET_SNAPSHOTS_OPEN_FAILED:{type(exc).__name__}"
            self._conn = None

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    def prefetch_pairs(
        self,
        pair_addresses: set[str],
        *,
        t_min: datetime | None,
        t_max: datetime | None,
        coin_ids: set[int] | None = None,
    ) -> None:
        """
        Bulk-load snapshots for needed pairs.

        pair_address is not indexed on trader.db; prefer coin_id index when available,
        otherwise one timestamp-bounded table scan filtered in Python.
        """
        if not self.available or self._conn is None or not pair_addresses:
            return
        pairs = {p for p in pair_addresses if p}
        if not pairs:
            return
        t_lo = t_min
        t_hi = t_max or self.global_latest_ts
        if t_lo is None or t_hi is None:
            return

        # Prefer indexed coin_id lookups
        loaded_pairs: set[str] = set()
        if self.coin_col and coin_ids:
            for cid in coin_ids:
                try:
                    rows = self._conn.execute(
                        f"""
                        SELECT {self.pair_col} AS pair_address, {self.ts_col} AS ts, {self.price_col} AS price
                        FROM market_snapshots
                        WHERE {self.coin_col} = ?
                          AND {self.ts_col} > ?
                          AND {self.ts_col} <= ?
                        ORDER BY {self.ts_col} ASC
                        """,
                        (cid, t_lo.isoformat(), t_hi.isoformat()),
                    ).fetchall()
                except Exception:
                    continue
                for r in rows:
                    pair = r["pair_address"]
                    if not pair:
                        continue
                    sp = str(pair)
                    if sp not in pairs:
                        continue
                    self._ingest_snap(sp, r["ts"], r["price"])
                    loaded_pairs.add(sp)

        remaining = pairs - loaded_pairs
        if remaining:
            # One scan over the time window; filter to remaining pairs in Python.
            rows = self._conn.execute(
                f"""
                SELECT {self.pair_col} AS pair_address, {self.ts_col} AS ts, {self.price_col} AS price
                FROM market_snapshots
                WHERE {self.ts_col} > ?
                  AND {self.ts_col} <= ?
                """,
                (t_lo.isoformat(), t_hi.isoformat()),
            ).fetchall()
            for r in rows:
                pair = r["pair_address"]
                if not pair:
                    continue
                sp = str(pair)
                if sp not in remaining:
                    continue
                self._ingest_snap(sp, r["ts"], r["price"])

        # Derive latest from series; pairs with no snaps → None
        for pair, series in self._series_by_pair.items():
            series.sort(key=lambda x: x[0])
            # de-dupe identical timestamps keeping last price
            dedup: list[tuple[datetime, float]] = []
            for ts, px in series:
                if dedup and dedup[-1][0] == ts:
                    dedup[-1] = (ts, px)
                else:
                    dedup.append((ts, px))
            self._series_by_pair[pair] = dedup
        for pair in pairs:
            series = self._series_by_pair.get(pair) or []
            if series:
                self._latest_by_pair[pair] = series[-1][0]
            else:
                self._latest_by_pair.setdefault(pair, None)
        self._prefetched = True

    def _ingest_snap(self, pair: str, ts_raw: Any, price_raw: Any) -> None:
        ts = parse_ts(ts_raw)
        try:
            price = float(price_raw) if price_raw is not None else None
        except (TypeError, ValueError):
            price = None
        if ts is None or price is None or price <= 0:
            return
        series = self._series_by_pair.setdefault(pair, [])
        series.append((ts, price))

    def latest_for_pair(self, pair_address: str | None) -> datetime | None:
        if not self.available or not pair_address:
            return None
        if pair_address in self._latest_by_pair:
            return self._latest_by_pair[pair_address]
        if self._prefetched:
            return None
        if self._conn is None:
            return None
        # Fallback (slow without pair index) — avoid in bulk runs
        row = self._conn.execute(
            f"SELECT MAX({self.ts_col}) AS mx FROM market_snapshots WHERE {self.pair_col} = ?",
            (pair_address,),
        ).fetchone()
        latest = parse_ts(row["mx"] if row else None)
        self._latest_by_pair[pair_address] = latest
        return latest

    def snapshots_in_window(
        self,
        pair_address: str | None,
        start_exclusive: datetime,
        end_inclusive: datetime,
    ) -> list[tuple[datetime, float]]:
        if not self.available or not pair_address:
            return []
        key = (pair_address, start_exclusive.isoformat(), end_inclusive.isoformat())
        if key in self._window_cache:
            return self._window_cache[key]

        series = self._series_by_pair.get(pair_address)
        if series is None and not self._prefetched and self._conn is not None:
            # Slow fallback path
            rows = self._conn.execute(
                f"""
                SELECT {self.ts_col} AS ts, {self.price_col} AS price
                FROM market_snapshots
                WHERE {self.pair_col} = ?
                  AND {self.ts_col} > ?
                  AND {self.ts_col} <= ?
                ORDER BY {self.ts_col} ASC
                """,
                (pair_address, start_exclusive.isoformat(), end_inclusive.isoformat()),
            ).fetchall()
            out: list[tuple[datetime, float]] = []
            for r in rows:
                ts = parse_ts(r["ts"])
                try:
                    price = float(r["price"]) if r["price"] is not None else None
                except (TypeError, ValueError):
                    price = None
                if ts is None or price is None or price <= 0:
                    continue
                if start_exclusive < ts <= end_inclusive:
                    out.append((ts, price))
            self._window_cache[key] = out
            return out

        out = [
            (ts, px)
            for ts, px in (series or [])
            if start_exclusive < ts <= end_inclusive
        ]
        self._window_cache[key] = out
        return out
