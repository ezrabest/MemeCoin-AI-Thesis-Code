"""Buffered append-only JSONL persistence and atomic writes for AE11."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent.parent.parent / "data"
RUNTIME_PAPER_LOOP_DIR = DATA_DIR / "runtime_paper_loop"
STATE_DIR = RUNTIME_PAPER_LOOP_DIR / "state"
PAPER_TRADING_DIR = DATA_DIR / "paper_trading"
EXECUTION_DIR = DATA_DIR / "execution"
REPORTS_DIR = Path(__file__).parent.parent.parent / "reports"
AUDITS_DIR = Path(__file__).parent.parent.parent / "audits"


def runtime_events_path_for_date(dt: datetime | None = None) -> Path:
    dt = dt or datetime.now(timezone.utc)
    return RUNTIME_PAPER_LOOP_DIR / f"ae11_runtime_events_{dt.strftime('%Y%m%d')}.jsonl"


def opportunity_capture_path_for_date(dt: datetime | None = None) -> Path:
    dt = dt or datetime.now(timezone.utc)
    return RUNTIME_PAPER_LOOP_DIR / f"ae11_opportunity_capture_{dt.strftime('%Y%m%d')}.jsonl"


def missed_winners_path_for_date(dt: datetime | None = None) -> Path:
    dt = dt or datetime.now(timezone.utc)
    return RUNTIME_PAPER_LOOP_DIR / f"ae11_missed_winners_{dt.strftime('%Y%m%d')}.jsonl"


def trade_decisions_path_for_date(dt: datetime | None = None) -> Path:
    dt = dt or datetime.now(timezone.utc)
    return RUNTIME_PAPER_LOOP_DIR / f"ae11_trade_decisions_{dt.strftime('%Y%m%d')}.jsonl"


def checkpoint_path() -> Path:
    return STATE_DIR / "ae11_latest_checkpoint.json"


def state_db_path() -> Path:
    return STATE_DIR / "ae11_state.sqlite"


class BufferedJsonlWriter:
    """Append-only JSONL writer that buffers rows and fsyncs at iteration boundary."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self._pending_count = 0
        self._flushed = False
        self._fsynced = False

    def _ensure_open(self):
        if self._file is None:
            self._file = open(self.path, "a", encoding="utf-8")
        return self._file

    def append_dict(self, record: dict[str, Any]) -> None:
        serialized = json.dumps(record, default=str, separators=(",", ":"))
        f = self._ensure_open()
        f.write(serialized + "\n")
        self._pending_count += 1
        self._flushed = False
        self._fsynced = False

    def flush_and_fsync(self) -> dict[str, Any]:
        if self._file is None:
            return {"path": str(self.path), "rows_flushed": 0, "fsynced": True}
        self._file.flush()
        self._flushed = True
        os.fsync(self._file.fileno())
        self._fsynced = True
        rows = self._pending_count
        self._pending_count = 0
        return {"path": str(self.path), "rows_flushed": rows, "fsynced": True}

    @property
    def flush_status(self) -> bool:
        return self._flushed

    @property
    def fsync_status(self) -> bool:
        return self._fsynced

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> BufferedJsonlWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class IterationWriters:
    """Manages all AE11 + paper trading writers for one loop iteration."""

    def __init__(self, dt: datetime | None = None) -> None:
        self.dt = dt or datetime.now(timezone.utc)
        self.runtime_events = BufferedJsonlWriter(runtime_events_path_for_date(self.dt))
        self.opportunity_capture = BufferedJsonlWriter(opportunity_capture_path_for_date(self.dt))
        self.missed_winners = BufferedJsonlWriter(missed_winners_path_for_date(self.dt))
        self.trade_decisions = BufferedJsonlWriter(trade_decisions_path_for_date(self.dt))
        from app.paper_trading.persistence import (
            live_dry_run_orders_path_for_date,
            paper_orders_path_for_date,
            paper_positions_path_for_date,
            paper_trades_path_for_date,
        )

        self.paper_orders = BufferedJsonlWriter(paper_orders_path_for_date(self.dt))
        self.paper_positions = BufferedJsonlWriter(paper_positions_path_for_date(self.dt))
        self.paper_trades = BufferedJsonlWriter(paper_trades_path_for_date(self.dt))
        self.live_dry_run = BufferedJsonlWriter(live_dry_run_orders_path_for_date(self.dt))

    def output_paths(self) -> dict[str, str]:
        return {
            "runtime_events": str(self.runtime_events.path),
            "opportunity_capture": str(self.opportunity_capture.path),
            "missed_winners": str(self.missed_winners.path),
            "trade_decisions": str(self.trade_decisions.path),
            "paper_orders": str(self.paper_orders.path),
            "paper_positions": str(self.paper_positions.path),
            "paper_trades": str(self.paper_trades.path),
            "live_dry_run": str(self.live_dry_run.path),
        }

    def flush_and_fsync_all(self) -> dict[str, Any]:
        results = {}
        for name in (
            "runtime_events",
            "opportunity_capture",
            "missed_winners",
            "trade_decisions",
            "paper_orders",
            "paper_positions",
            "paper_trades",
            "live_dry_run",
        ):
            writer = getattr(self, name)
            results[name] = writer.flush_and_fsync()
        return results

    def close_all(self) -> None:
        for name in (
            "runtime_events",
            "opportunity_capture",
            "missed_winners",
            "trade_decisions",
            "paper_orders",
            "paper_positions",
            "paper_trades",
            "live_dry_run",
        ):
            getattr(self, name).close()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> Path:
    """Write JSON atomically via temp file + os.replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return path


def clean_stale_tmp_files(root: Path | None = None) -> list[str]:
    """Remove stale *.tmp files in AE11-owned runtime folders only."""
    root = root or RUNTIME_PAPER_LOOP_DIR
    removed: list[str] = []
    if not root.exists():
        return removed
    for tmp in root.rglob("*.tmp"):
        try:
            tmp.unlink()
            removed.append(str(tmp))
        except OSError:
            pass
    return removed
