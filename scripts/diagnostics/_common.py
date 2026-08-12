"""Shared utilities for Phase 4 diagnostics (read-only, no production mutation)."""
from __future__ import annotations

import csv
import json
import math
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = Path(__import__("os").getenv("TRADER_DB_PATH", str(DATA_DIR / "trader.db")))
SETTINGS_PATH = DATA_DIR / "settings.json"
CHUNK_SIZE = 1000

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def timestamp_slug(dt: datetime | None = None) -> str:
    dt = dt or utc_now()
    return dt.strftime("%Y%m%dT%H%M%SZ")


def parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def open_db_readonly(db_path: Path | None = None) -> sqlite3.Connection:
    """Open SQLite read-only with uri=True (required on Windows)."""
    path = (db_path or DB_PATH).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Database not found: {path}")
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {k: row[k] for k in row.keys()}


def safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x if x is not None else default)
    except (TypeError, ValueError):
        return default


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    idx = int(round((len(vals) - 1) * p))
    idx = max(0, min(len(vals) - 1, idx))
    return vals[idx]


def distribution_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    vals = sorted(values)
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n if n > 1 else 0.0
    return {
        "count": n,
        "min": vals[0],
        "p10": percentile(vals, 0.10),
        "p25": percentile(vals, 0.25),
        "p50": percentile(vals, 0.50),
        "p75": percentile(vals, 0.75),
        "p90": percentile(vals, 0.90),
        "p95": percentile(vals, 0.95),
        "p99": percentile(vals, 0.99),
        "max": vals[-1],
        "mean": round(mean, 6),
        "std": round(math.sqrt(var), 6),
    }


def histogram_buckets(values: list[float], edges: list[tuple[str, float, float | None]]) -> dict[str, int]:
    counts = {label: 0 for label, _, _ in edges}
    for v in values:
        for label, lo, hi in edges:
            if hi is None:
                if v >= lo:
                    counts[label] += 1
                    break
            elif lo <= v < hi:
                counts[label] += 1
                break
    return counts


WHALE_SCORE_HISTOGRAM_EDGES: list[tuple[str, float, float | None]] = [
    ("0.00-0.05", 0.0, 0.05),
    ("0.05-0.10", 0.05, 0.10),
    ("0.10-0.15", 0.10, 0.15),
    ("0.15-0.20", 0.15, 0.20),
    ("0.20-0.25", 0.20, 0.25),
    ("0.25-0.30", 0.25, 0.30),
    ("0.30-0.40", 0.30, 0.40),
    ("0.40-0.50", 0.40, 0.50),
    ("0.50+", 0.50, None),
]


def parse_audit_reasons_field(raw: Any) -> tuple[list[str], str]:
    """Robust audit reason parsing — never iterate strings as characters."""
    if raw is None:
        return [], "empty"
    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None and str(x)], "list"
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return [], "empty_string"
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(x) for x in parsed if x is not None], "json_array_string"
            except json.JSONDecodeError:
                pass
        return [stripped], "plain_string"
    return [str(raw)], "other"


def count_reasons(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        raw = row.get("audit_reasons")
        if raw is None:
            raw = row.get("audit_reasons_json")
        reasons, _ = parse_audit_reasons_field(raw)
        for reason in reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def snapshot_to_pair(row: dict[str, Any]) -> dict[str, Any]:
    """Build minimal DexScreener-like pair dict from market_snapshots row."""
    return {
        "pairAddress": row.get("pair_address") or "",
        "chainId": (row.get("chain") or "unknown").lower(),
        "priceUsd": str(row.get("price") or 0),
        "liquidity": {"usd": safe_float(row.get("liquidity"))},
        "volume": {
            "h24": safe_float(row.get("volume_24h")),
            "h1": safe_float(row.get("volume_h1") or row.get("volume_1h") or 0),
        },
        "txns": {
            "h24": {
                "buys": int(row.get("txns_buys") or 0),
                "sells": int(row.get("txns_sells") or 0),
            }
        },
        "priceChange": {
            "h1": safe_float(row.get("price_change_h1")),
            "h24": safe_float(row.get("price_change_h24")),
            "m5": safe_float(row.get("price_change_m5")),
            "h6": safe_float(row.get("price_change_h6")),
        },
    }


def reason_if_no_alert(pair: dict[str, Any], whale_score: float) -> str:
    from app.engine import WHALE_ALERT_MIN_VOLUME_24H, WHALE_ALERT_MIN_WHALE_SCORE, _vol, _liq, _buy_ratio

    vol24 = _vol(pair)
    liq = _liq(pair)
    br = _buy_ratio(pair)
    if vol24 < WHALE_ALERT_MIN_VOLUME_24H:
        return f"volume_24h {vol24:.0f} < min {WHALE_ALERT_MIN_VOLUME_24H}"
    if whale_score < WHALE_ALERT_MIN_WHALE_SCORE:
        return f"whale_score {whale_score:.4f} < min {WHALE_ALERT_MIN_WHALE_SCORE}"
    if liq <= 0:
        return "zero liquidity"
    if br < 0.38 and vol24 > 20_000:
        return "possible bearish pattern but no rule matched"
    return "pattern rules not matched (flow/momentum thresholds)"


def iter_snapshot_chunks(
    conn: sqlite3.Connection,
    *,
    limit: int,
    chunk_size: int = CHUNK_SIZE,
    where_sql: str = "",
    params: tuple[Any, ...] = (),
) -> Iterator[list[dict[str, Any]]]:
    """Yield market_snapshots rows in chunks (newest first)."""
    where = f"WHERE {where_sql}" if where_sql else ""
    offset = 0
    fetched = 0
    while fetched < limit:
        batch = min(chunk_size, limit - fetched)
        rows = conn.execute(
            f"""
            SELECT ms.*, c.symbol AS coin_symbol
            FROM market_snapshots ms
            LEFT JOIN coins c ON c.id = ms.coin_id
            {where}
            ORDER BY ms.timestamp DESC, ms.id DESC
            LIMIT ? OFFSET ?
            """,
            (*params, batch, offset),
        ).fetchall()
        if not rows:
            break
        chunk = [row_to_dict(r) for r in rows]  # type: ignore[misc]
        yield chunk  # type: ignore[misc]
        fetched += len(chunk)
        offset += len(chunk)
        if len(chunk) < batch:
            break


def iter_candidate_signal_chunks(
    conn: sqlite3.Connection,
    *,
    limit: int,
    chunk_size: int = CHUNK_SIZE,
    cutoff: str | None = None,
) -> Iterator[list[dict[str, Any]]]:
    """Yield recent signal rows joined with latest snapshot fields (chunked, fast JOIN)."""
    where = "WHERE s.timestamp >= ?" if cutoff else ""
    params: tuple[Any, ...] = (cutoff,) if cutoff else ()
    offset = 0
    fetched = 0
    base_sql = f"""
        WITH latest_snap AS (
            SELECT ms.*
            FROM market_snapshots ms
            INNER JOIN (
                SELECT coin_id, MAX(id) AS max_id
                FROM market_snapshots
                GROUP BY coin_id
            ) snap_max ON ms.id = snap_max.max_id
        ),
        latest_alert AS (
            SELECT wa.coin_id, wa.alert_type
            FROM whale_alerts wa
            INNER JOIN (
                SELECT coin_id, MAX(id) AS max_id
                FROM whale_alerts
                GROUP BY coin_id
            ) alert_max ON wa.id = alert_max.max_id
        )
        SELECT s.id AS signal_id, s.timestamp, s.coin_id, s.symbol, s.signal_type,
               s.score, s.confidence, s.features_json,
               c.pair_address, c.chain,
               ls.price AS snap_price,
               ls.liquidity AS snap_liquidity,
               ls.volume_24h AS snap_volume_24h,
               ls.txns_buys AS snap_txns_buys,
               ls.txns_sells AS snap_txns_sells,
               ls.buy_ratio AS snap_buy_ratio,
               ls.whale_score AS snap_whale_score,
               ls.price_change_h1 AS snap_price_change_h1,
               ls.price_change_h24 AS snap_price_change_h24,
               la.alert_type AS latest_alert_type
        FROM signals s
        LEFT JOIN coins c ON c.id = s.coin_id
        LEFT JOIN latest_snap ls ON ls.coin_id = s.coin_id
        LEFT JOIN latest_alert la ON la.coin_id = s.coin_id
        {where}
        ORDER BY s.timestamp DESC, s.id DESC
        LIMIT ? OFFSET ?
    """
    while fetched < limit:
        batch = min(chunk_size, limit - fetched)
        rows = conn.execute(base_sql, (*params, batch, offset)).fetchall()
        if not rows:
            break
        chunk = [row_to_dict(r) for r in rows]  # type: ignore[misc]
        yield chunk  # type: ignore[misc]
        fetched += len(chunk)
        offset += len(chunk)
        if len(chunk) < batch:
            break


def build_pair_from_signal_row(row: dict[str, Any], feats: dict[str, Any]) -> dict[str, Any]:
    symbol = str(row.get("symbol") or "?")
    base_sym = symbol.split("/")[0] if "/" in symbol else symbol
    buys = int(row.get("snap_txns_buys") or feats.get("txns_buys_24h") or 0)
    sells = int(row.get("snap_txns_sells") or feats.get("txns_sells_24h") or 0)
    return {
        "pairAddress": row.get("pair_address") or "",
        "chainId": (row.get("chain") or "unknown").lower(),
        "baseToken": {"symbol": base_sym},
        "priceUsd": str(row.get("snap_price") or feats.get("price_usd") or 0),
        "liquidity": {"usd": safe_float(row.get("snap_liquidity") or feats.get("liquidity_usd"))},
        "volume": {"h24": safe_float(row.get("snap_volume_24h") or feats.get("volume_24h"))},
        "txns": {"h24": {"buys": buys, "sells": sells}},
        "priceChange": {
            "h1": safe_float(row.get("snap_price_change_h1") or feats.get("price_change_1h")),
            "h24": safe_float(row.get("snap_price_change_h24") or feats.get("price_change_24h")),
        },
    }


def features_dict(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def load_settings_file() -> dict[str, Any]:
    if not SETTINGS_PATH.is_file():
        return {}
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def fetch_json_url(url: str, timeout: float = 3.0) -> dict[str, Any] | None:
    try:
        import urllib.request

        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


class DiagnosticReport:
    """Standard Phase 4 diagnostic report writer."""

    def __init__(self, name: str, output_dir: Path) -> None:
        self.name = name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.data: dict[str, Any] = {
            "diagnostic": name,
            "timestamp": utc_now().isoformat(),
            "status": "PASS",
            "limitations": [],
        }

    def set_status(self, status: str) -> None:
        self.data["status"] = status

    def add_limitation(self, msg: str) -> None:
        self.data.setdefault("limitations", []).append(msg)
        if self.data["status"] == "PASS":
            self.data["status"] = "WARN"

    def write_json(self, filename: str | None = None) -> Path:
        path = self.output_dir / (filename or f"{self.name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=2, default=str)
        return path

    def write_md(self, lines: list[str], filename: str | None = None) -> Path:
        path = self.output_dir / (filename or f"{self.name}.md")
        header = [
            f"# {self.name}",
            "",
            f"**Status:** {self.data.get('status', 'UNKNOWN')}",
            f"**Generated:** {self.data.get('timestamp')}",
            "",
        ]
        if self.data.get("limitations"):
            header.append("## Limitations")
            header.extend(f"- {x}" for x in self.data["limitations"])
            header.append("")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(header + lines))
        return path

    def write_csv(self, rows: list[dict[str, Any]], filename: str) -> Path | None:
        if not rows:
            return None
        path = self.output_dir / filename
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return path
