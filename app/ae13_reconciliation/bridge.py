"""AE13 Virtual Ledger View — non-destructive paper/demo read-model bridge.

Reads legacy paper_state / CSV, daily paper_trading JSONL, AE11 SQLite/snapshots,
and trader.db paper_trades without rewriting historical ledgers.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.decision.persistence import read_jsonl_records_safe

STARTING_CAPITAL_USD = 10_000.0

SOURCE_LEGACY_PAPER_STATE = "legacy_paper_state"
SOURCE_LEGACY_PAPER_TRADES_LOG = "legacy_paper_trades_log"
SOURCE_DAILY_PAPER_TRADING_JSONL = "daily_paper_trading_jsonl"
SOURCE_AE11_RUNTIME_SQLITE = "ae11_runtime_sqlite"
SOURCE_AE11_SNAPSHOT = "ae11_snapshot"
SOURCE_SQLITE_PAPER_TRADES = "sqlite_paper_trades"
SOURCE_AE10_SAMPLE_ARCHIVE = "ae10_sample_archive"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_str(v: Any, default: str = "") -> str:
    if v is None:
        return default
    return str(v)


def _file_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False, "mtime_utc": None, "size_bytes": 0}
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "size_bytes": int(st.st_size),
    }


def _dedupe_key_order(rec: dict[str, Any]) -> str:
    for k in ("paper_order_id", "order_id", "id", "source_decision_id"):
        v = rec.get(k)
        if v not in (None, ""):
            return f"order:{v}"
    return (
        f"order_fb:{rec.get('symbol')}|{rec.get('side')}|{rec.get('timestamp')}"
        f"|{rec.get('notional_usd')}|{rec.get('pair_address')}"
    )


def _dedupe_key_position(rec: dict[str, Any]) -> str:
    for k in ("position_id", "id"):
        v = rec.get(k)
        if v not in (None, ""):
            return f"pos:{v}"
    return (
        f"pos_fb:{rec.get('symbol')}|{rec.get('pair_address')}|{rec.get('opened_at')}"
        f"|{rec.get('entry_price')}"
    )


def _dedupe_key_trade(rec: dict[str, Any]) -> str:
    for k in ("close_event_id", "economic_close_key", "trade_id", "id"):
        v = rec.get(k)
        if v not in (None, ""):
            return f"trade:{v}"
    return (
        f"trade_fb:{rec.get('symbol')}|{rec.get('side')}|{rec.get('timestamp')}"
        f"|{rec.get('notional_usd')}|{rec.get('position_id')}"
    )


@dataclass
class VirtualLedgerView:
    """In-memory canonical demo read model with provenance."""

    project_root: Path
    built_at_utc: str
    demo_balance: dict[str, Any]
    orders: list[dict[str, Any]] = field(default_factory=list)
    open_positions: list[dict[str, Any]] = field(default_factory=list)
    closed_trades: list[dict[str, Any]] = field(default_factory=list)
    balance_timeline: list[dict[str, Any]] = field(default_factory=list)
    reconciliation_rows: list[dict[str, Any]] = field(default_factory=list)
    source_freshness: dict[str, Any] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ui_write_source_of_truth: str = SOURCE_LEGACY_PAPER_STATE
    read_model: str = "virtual_ledger_view"

    def summary(self) -> dict[str, Any]:
        return {
            "built_at_utc": self.built_at_utc,
            "read_model": self.read_model,
            "ui_write_source_of_truth": self.ui_write_source_of_truth,
            "demo_balance": self.demo_balance,
            "orders_count": len(self.orders),
            "open_positions_count": len(self.open_positions),
            "closed_trades_count": len(self.closed_trades),
            "source_freshness": self.source_freshness,
            "warnings": list(self.warnings),
            "conflicts_count": len(self.conflicts),
            "paper_demo_only": True,
            "not_live_approved": True,
            "wallet_configured": False,
        }

    def to_api_payload(self, *, limit_orders: int = 200, limit_positions: int = 200, limit_trades: int = 200) -> dict[str, Any]:
        return {
            **self.summary(),
            "orders": self.orders[:limit_orders],
            "open_positions": self.open_positions[:limit_positions],
            "closed_trades": self.closed_trades[:limit_trades],
            "balance_timeline": self.balance_timeline[-50:],
            "conflicts": self.conflicts[:50],
        }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _load_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    try:
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except OSError:
        return []


def _read_jsonl_glob(directory: Path, prefix: str) -> tuple[list[dict[str, Any]], list[Path]]:
    if not directory.is_dir():
        return [], []
    paths = sorted(directory.glob(f"{prefix}_*.jsonl"))
    rows: list[dict[str, Any]] = []
    for path in paths:
        records, _meta = read_jsonl_records_safe(path)
        rows.extend(records)
    return rows, paths


def _normalize_order(raw: dict[str, Any], source_layer: str, source_path: str) -> dict[str, Any]:
    return {
        "record_kind": "order",
        "source_layer": source_layer,
        "source_path": source_path,
        "paper_order_id": raw.get("paper_order_id") or raw.get("order_id") or raw.get("id"),
        "symbol": raw.get("symbol"),
        "side": _safe_str(raw.get("side"), "").upper(),
        "status": raw.get("status"),
        "notional_usd": _safe_float(raw.get("notional_usd")),
        "quantity": _safe_float(raw.get("quantity")),
        "fill_price": _safe_float(raw.get("filled_price_usd") or raw.get("fill_price") or raw.get("requested_price_usd")),
        "pair_address": raw.get("pair_address"),
        "timestamp": raw.get("created_at_utc") or raw.get("filled_at_utc") or raw.get("timestamp"),
        "trade_authority": raw.get("trade_authority") or "PAPER_DEMO",
        "not_live_approved": bool(raw.get("not_live_approved", True)),
        "demo_acceptance_only": bool(raw.get("demo_acceptance_only", False)),
        "not_strategy_evidence": bool(raw.get("not_strategy_evidence", raw.get("not_model_approved", False))),
        "not_profitability_evidence": bool(raw.get("not_profitability_evidence", True)),
        "paper_demo_only": True,
        "raw_keys": sorted(raw.keys())[:40],
    }


def _normalize_position(raw: dict[str, Any], source_layer: str, source_path: str) -> dict[str, Any]:
    status = _safe_str(raw.get("status"), "OPEN").upper()
    return {
        "record_kind": "position",
        "source_layer": source_layer,
        "source_path": source_path,
        "position_id": raw.get("position_id") or raw.get("id"),
        "symbol": raw.get("symbol"),
        "chain": raw.get("chain"),
        "status": status,
        "entry_price": _safe_float(raw.get("entry_price")),
        "size_usd": _safe_float(raw.get("size_usd") or raw.get("notional_usd") or raw.get("cost_basis_usd")),
        "quantity": _safe_float(raw.get("quantity")),
        "notional_usd": _safe_float(raw.get("notional_usd")),
        "pair_address": raw.get("pair_address"),
        "opened_at": raw.get("opened_at_utc") or raw.get("opened_at") or raw.get("timestamp"),
        "cluster_label": raw.get("cluster_label"),
        "unrealized_pnl_usd": _safe_float(raw.get("unrealized_pnl_usd")),
        "trade_authority": raw.get("trade_authority") or "PAPER_DEMO",
        "not_live_approved": bool(raw.get("not_live_approved", True)),
        "paper_demo_only": True,
        "demo_acceptance_only": bool(raw.get("demo_acceptance_only", False)),
    }


def _normalize_trade(raw: dict[str, Any], source_layer: str, source_path: str) -> dict[str, Any]:
    side = _safe_str(raw.get("side"), "").lower()
    if not side:
        side = "sell" if raw.get("closed_at_utc") or raw.get("exit_price") else "buy"
    notional = _safe_float(
        raw.get("notional_usd")
        or raw.get("value")
        or raw.get("notional_executed")
        or raw.get("cash_credited_usd")
    )
    fees = _safe_float(raw.get("total_fees") or raw.get("total_fees_usd") or raw.get("fee"))
    return {
        "record_kind": "trade",
        "source_layer": source_layer,
        "source_path": source_path,
        "trade_id": raw.get("close_event_id") or raw.get("id") or raw.get("economic_close_key"),
        "position_id": raw.get("position_id"),
        "symbol": raw.get("symbol"),
        "side": side,
        "timestamp": (
            raw.get("closed_at_utc")
            or raw.get("timestamp")
            or raw.get("close_event_created_at_utc")
        ),
        "notional_usd": notional,
        "total_fees": fees,
        "fill_price": _safe_float(raw.get("exit_price") or raw.get("fill_price") or raw.get("price")),
        "quantity": _safe_float(raw.get("quantity") or raw.get("amount")),
        "realized_pnl": _safe_float(raw.get("realized_pnl") or raw.get("net_pnl_usd") or raw.get("pnl")),
        "net_roi_pct": _safe_float(raw.get("net_roi_pct") or raw.get("net_return_pct")),
        "reason_code": raw.get("reason_code") or raw.get("reason") or raw.get("exit_reason") or raw.get("close_reason"),
        "chain": raw.get("chain"),
        "cluster_label": raw.get("cluster_label"),
        "pair_address": raw.get("pair_address"),
        "paper_demo_only": True,
        "not_live_approved": bool(raw.get("not_live_approved", True)),
        "wallet_configured": bool(raw.get("wallet_configured", False)),
        "demo_acceptance_only": bool(raw.get("demo_acceptance_only", False)),
        "not_strategy_evidence": bool(raw.get("not_strategy_evidence", True)),
        "not_profitability_evidence": bool(raw.get("not_profitability_evidence", True)),
    }


def _merge_dedupe(
    records: list[dict[str, Any]],
    key_fn,
    *,
    prefer_layers: list[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prefer_rank = {layer: i for i, layer in enumerate(prefer_layers)}
    best: dict[str, dict[str, Any]] = {}
    groups: dict[str, list[str]] = {}
    conflicts: list[dict[str, Any]] = []

    for rec in records:
        key = key_fn(rec)
        groups.setdefault(key, []).append(rec.get("source_layer", "?"))
        existing = best.get(key)
        if existing is None:
            best[key] = rec
            continue
        old_rank = prefer_rank.get(existing.get("source_layer", ""), 999)
        new_rank = prefer_rank.get(rec.get("source_layer", ""), 999)
        if new_rank < old_rank:
            best[key] = rec

    for key, layers in groups.items():
        uniq = sorted(set(layers))
        if len(uniq) > 1:
            conflicts.append({"dedupe_key": key, "source_layers": uniq, "count": len(layers)})

    return list(best.values()), conflicts


def _load_paper_state_positions(project_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    path = project_root / "data" / "paper_state.json"
    meta = _file_meta(path)
    state = _load_json(path) or {}
    positions = []
    for pos in state.get("open_positions") or []:
        if isinstance(pos, dict):
            positions.append(_normalize_position(pos, SOURCE_LEGACY_PAPER_STATE, str(path)))
    balance = {
        "starting_capital": _safe_float(state.get("starting_capital"), STARTING_CAPITAL_USD),
        "cash_usd": _safe_float(state.get("cash_usd"), STARTING_CAPITAL_USD),
        "open_positions_count": len(positions),
        "closed_trades": int(state.get("closed_trades") or 0),
        "total_net_pnl": _safe_float(state.get("total_net_pnl")),
        "cumulative_total_fees": _safe_float(state.get("cumulative_total_fees")),
        "trading_mode": state.get("trading_mode", "DEMO"),
        "source_layer": SOURCE_LEGACY_PAPER_STATE,
        "source_path": str(path),
    }
    positions_val = sum(_safe_float(p.get("size_usd")) for p in positions)
    balance["positions_value_usd"] = round(positions_val, 2)
    balance["total_equity_usd"] = round(balance["cash_usd"] + positions_val, 2)
    return balance, positions, meta


def _load_csv_trades(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = project_root / "data" / "paper_trades_log.csv"
    meta = _file_meta(path)
    rows = []
    for raw in _load_csv_rows(path):
        rows.append(_normalize_trade(raw, SOURCE_LEGACY_PAPER_TRADES_LOG, str(path)))
    return rows, meta


def _load_daily_jsonl(project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    directory = project_root / "data" / "paper_trading"
    orders_raw, order_paths = _read_jsonl_glob(directory, "paper_orders")
    positions_raw, pos_paths = _read_jsonl_glob(directory, "paper_positions")
    trades_raw, trade_paths = _read_jsonl_glob(directory, "paper_trades")

    orders = [_normalize_order(r, SOURCE_DAILY_PAPER_TRADING_JSONL, str(directory)) for r in orders_raw]
    open_positions = [
        p for p in (
            _normalize_position(r, SOURCE_DAILY_PAPER_TRADING_JSONL, str(directory))
            for r in positions_raw
        )
        if p["status"] == "OPEN"
    ]
    trades = [_normalize_trade(r, SOURCE_DAILY_PAPER_TRADING_JSONL, str(directory)) for r in trades_raw]

    freshness = {
        "orders_files": [_file_meta(p) for p in order_paths[-5:]],
        "positions_files": [_file_meta(p) for p in pos_paths[-5:]],
        "trades_files": [_file_meta(p) for p in trade_paths[-5:]],
        "raw_orders": len(orders_raw),
        "raw_positions": len(positions_raw),
        "raw_trades": len(trades_raw),
        "open_positions": len(open_positions),
    }
    return orders, open_positions, trades, freshness


def _load_ae11_sqlite(project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    path = project_root / "data" / "runtime_paper_loop" / "state" / "ae11_state.sqlite"
    meta = _file_meta(path)
    open_positions: list[dict[str, Any]] = []
    closed_trades: list[dict[str, Any]] = []
    if not path.is_file():
        return open_positions, closed_trades, meta
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            # Open = status OPEN (some CLOSED rows linger in active_positions historically)
            rows = conn.execute(
                "SELECT * FROM active_positions WHERE UPPER(COALESCE(status,'')) = 'OPEN' LIMIT 5000"
            ).fetchall()
            for row in rows:
                open_positions.append(
                    _normalize_position(dict(row), SOURCE_AE11_RUNTIME_SQLITE, str(path))
                )
            closed = conn.execute("SELECT * FROM closed_positions LIMIT 5000").fetchall()
            for row in closed:
                closed_trades.append(
                    _normalize_trade(dict(row), SOURCE_AE11_RUNTIME_SQLITE, str(path))
                )
            meta = {
                **meta,
                "open_count": len(open_positions),
                "closed_count": len(closed_trades),
            }
        finally:
            conn.close()
    except sqlite3.Error as exc:
        meta = {**meta, "error": str(exc)}
    return open_positions, closed_trades, meta


def _load_ae11_snapshots(project_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    open_path = project_root / "data" / "ae11_open_positions_snapshot.csv"
    closed_path = project_root / "data" / "ae11_closed_trades_snapshot.csv"
    open_rows = [
        _normalize_position(r, SOURCE_AE11_SNAPSHOT, str(open_path))
        for r in _load_csv_rows(open_path)
        if _safe_str(r.get("status"), "OPEN").upper() == "OPEN"
    ]
    closed_rows = [
        _normalize_trade(r, SOURCE_AE11_SNAPSHOT, str(closed_path))
        for r in _load_csv_rows(closed_path)
    ]
    return open_rows, closed_rows, {
        "open_snapshot": _file_meta(open_path),
        "closed_snapshot": _file_meta(closed_path),
        "open_count": len(open_rows),
        "closed_count": len(closed_rows),
    }


def _load_sqlite_paper_trades(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = project_root / "data" / "trader.db"
    meta = _file_meta(path)
    trades: list[dict[str, Any]] = []
    if not path.is_file():
        return trades, meta
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT * FROM paper_trades ORDER BY id DESC LIMIT 2000"
            ).fetchall()
            for row in rows:
                trades.append(_normalize_trade(dict(row), SOURCE_SQLITE_PAPER_TRADES, str(path)))
            meta = {**meta, "count": len(trades)}
        finally:
            conn.close()
    except sqlite3.Error as exc:
        meta = {**meta, "error": str(exc)}
    return trades, meta


def _build_balance_timeline(
    write_balance: dict[str, Any],
    closed_trades: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    starting = _safe_float(write_balance.get("starting_capital"), STARTING_CAPITAL_USD)
    timeline = [
        {
            "timestamp": "baseline",
            "event": "STARTING_CAPITAL",
            "cash_usd": starting,
            "equity_usd": starting,
            "source_layer": write_balance.get("source_layer"),
            "paper_demo_only": True,
        }
    ]
    # Chronological closed trades PnL walk (informational; write SoT remains paper_state)
    sorted_trades = sorted(
        [t for t in closed_trades if t.get("timestamp")],
        key=lambda t: str(t.get("timestamp")),
    )
    equity = starting
    for t in sorted_trades[-100:]:
        equity = round(equity + _safe_float(t.get("realized_pnl")), 6)
        timeline.append(
            {
                "timestamp": t.get("timestamp"),
                "event": "CLOSED_TRADE",
                "symbol": t.get("symbol"),
                "realized_pnl": t.get("realized_pnl"),
                "equity_usd_approx": equity,
                "source_layer": t.get("source_layer"),
                "paper_demo_only": True,
                "not_profitability_evidence": True,
            }
        )
    timeline.append(
        {
            "timestamp": _utc_now(),
            "event": "UI_WRITE_BALANCE",
            "cash_usd": write_balance.get("cash_usd"),
            "equity_usd": write_balance.get("total_equity_usd"),
            "source_layer": write_balance.get("source_layer"),
            "note": "Operational UI write SoT is legacy paper_state.json; VLV merges archives.",
            "paper_demo_only": True,
        }
    )
    return timeline


def build_virtual_ledger_view(project_root: Path | None = None) -> VirtualLedgerView:
    """Build non-destructive in-memory Virtual Ledger View."""
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    warnings: list[str] = []
    built_at = _utc_now()

    write_balance, state_positions, state_meta = _load_paper_state_positions(root)
    csv_trades, csv_meta = _load_csv_trades(root)
    jsonl_orders, jsonl_open, jsonl_trades, jsonl_fresh = _load_daily_jsonl(root)
    ae11_open, ae11_closed, ae11_meta = _load_ae11_sqlite(root)
    snap_open, snap_closed, snap_meta = _load_ae11_snapshots(root)
    sqlite_trades, sqlite_meta = _load_sqlite_paper_trades(root)

    if write_balance.get("cash_usd") == STARTING_CAPITAL_USD and not state_positions:
        if jsonl_open or ae11_open or snap_open or jsonl_trades or ae11_closed:
            warnings.append(
                "UI write SoT paper_state.json shows fresh $10k with no open positions, "
                "while AE11/JSONL archives contain paper activity — split-brain confirmed; "
                "VLV exposes archives without mutating them."
            )
    if csv_meta.get("exists") and len(csv_trades) == 0:
        warnings.append("legacy paper_trades_log.csv has header only (no rows).")
    if not ae11_meta.get("exists"):
        warnings.append("ae11_state.sqlite missing.")

    order_pref = [SOURCE_DAILY_PAPER_TRADING_JSONL, SOURCE_AE11_RUNTIME_SQLITE]
    pos_pref = [
        SOURCE_LEGACY_PAPER_STATE,
        SOURCE_AE11_RUNTIME_SQLITE,
        SOURCE_AE11_SNAPSHOT,
        SOURCE_DAILY_PAPER_TRADING_JSONL,
    ]
    trade_pref = [
        SOURCE_LEGACY_PAPER_TRADES_LOG,
        SOURCE_SQLITE_PAPER_TRADES,
        SOURCE_AE11_RUNTIME_SQLITE,
        SOURCE_AE11_SNAPSHOT,
        SOURCE_DAILY_PAPER_TRADING_JSONL,
    ]

    orders, order_conflicts = _merge_dedupe(jsonl_orders, _dedupe_key_order, prefer_layers=order_pref)
    open_positions, pos_conflicts = _merge_dedupe(
        state_positions + ae11_open + snap_open + jsonl_open,
        _dedupe_key_position,
        prefer_layers=pos_pref,
    )
    closed_trades, trade_conflicts = _merge_dedupe(
        csv_trades + sqlite_trades + ae11_closed + snap_closed + jsonl_trades,
        _dedupe_key_trade,
        prefer_layers=trade_pref,
    )

    # Sort newest-first for UI
    def _ts(r: dict[str, Any]) -> str:
        return str(r.get("timestamp") or r.get("opened_at") or "")

    orders.sort(key=_ts, reverse=True)
    open_positions.sort(key=lambda r: str(r.get("opened_at") or ""), reverse=True)
    closed_trades.sort(key=_ts, reverse=True)

    # Merged demo balance: write SoT for cash/equity; archive counts for visibility
    merged_balance = {
        **write_balance,
        "merged_open_positions_count": len(open_positions),
        "merged_closed_trades_count": len(closed_trades),
        "merged_orders_count": len(orders),
        "ui_write_open_positions_count": len(state_positions),
        "archive_open_positions_visible": len(open_positions) > len(state_positions),
        "read_model": "virtual_ledger_view",
        "paper_demo_only": True,
        "not_live_approved": True,
        "wallet_configured": False,
    }

    reconciliation_rows = [
        {
            "layer": SOURCE_LEGACY_PAPER_STATE,
            "role": "UI_WRITE_SOT",
            "open_positions": len(state_positions),
            "orders": 0,
            "trades": 0,
            "freshness": state_meta,
        },
        {
            "layer": SOURCE_LEGACY_PAPER_TRADES_LOG,
            "role": "ROI_CSV",
            "open_positions": 0,
            "orders": 0,
            "trades": len(csv_trades),
            "freshness": csv_meta,
        },
        {
            "layer": SOURCE_DAILY_PAPER_TRADING_JSONL,
            "role": "AE10_AE11_DAILY_JSONL",
            "open_positions": len(jsonl_open),
            "orders": len(jsonl_orders),
            "trades": len(jsonl_trades),
            "freshness": jsonl_fresh,
        },
        {
            "layer": SOURCE_AE11_RUNTIME_SQLITE,
            "role": "AE11_RUNTIME",
            "open_positions": len(ae11_open),
            "orders": 0,
            "trades": len(ae11_closed),
            "freshness": ae11_meta,
        },
        {
            "layer": SOURCE_AE11_SNAPSHOT,
            "role": "AE11_SNAPSHOT",
            "open_positions": len(snap_open),
            "orders": 0,
            "trades": len(snap_closed),
            "freshness": snap_meta,
        },
        {
            "layer": SOURCE_SQLITE_PAPER_TRADES,
            "role": "TRADER_DB_PAPER_TRADES",
            "open_positions": 0,
            "orders": 0,
            "trades": len(sqlite_trades),
            "freshness": sqlite_meta,
        },
    ]

    return VirtualLedgerView(
        project_root=root,
        built_at_utc=built_at,
        demo_balance=merged_balance,
        orders=orders,
        open_positions=open_positions,
        closed_trades=closed_trades,
        balance_timeline=_build_balance_timeline(write_balance, closed_trades),
        reconciliation_rows=reconciliation_rows,
        source_freshness={
            "paper_state": state_meta,
            "paper_trades_log": csv_meta,
            "daily_jsonl": jsonl_fresh,
            "ae11_sqlite": ae11_meta,
            "ae11_snapshots": snap_meta,
            "trader_db_paper_trades": sqlite_meta,
        },
        conflicts=order_conflicts + pos_conflicts + trade_conflicts,
        warnings=warnings,
        ui_write_source_of_truth=SOURCE_LEGACY_PAPER_STATE,
        read_model="virtual_ledger_view",
    )
