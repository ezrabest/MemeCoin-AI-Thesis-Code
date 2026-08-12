"""Append-only JSONL persistence for AE10 paper trading."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.decision.persistence import read_jsonl_records_safe

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PAPER_TRADING_DIR = DATA_DIR / "paper_trading"
EXECUTION_DIR = DATA_DIR / "execution"
DEMO_ACCOUNT_STATE_PATH = PAPER_TRADING_DIR / "demo_account_state.json"


def paper_orders_path_for_date(dt: datetime | None = None) -> Path:
    dt = dt or datetime.now(timezone.utc)
    return PAPER_TRADING_DIR / f"paper_orders_{dt.strftime('%Y%m%d')}.jsonl"


def paper_positions_path_for_date(dt: datetime | None = None) -> Path:
    dt = dt or datetime.now(timezone.utc)
    return PAPER_TRADING_DIR / f"paper_positions_{dt.strftime('%Y%m%d')}.jsonl"


def paper_trades_path_for_date(dt: datetime | None = None) -> Path:
    dt = dt or datetime.now(timezone.utc)
    return PAPER_TRADING_DIR / f"paper_trades_{dt.strftime('%Y%m%d')}.jsonl"


def live_dry_run_orders_path_for_date(dt: datetime | None = None) -> Path:
    dt = dt or datetime.now(timezone.utc)
    return EXECUTION_DIR / f"live_dry_run_orders_{dt.strftime('%Y%m%d')}.jsonl"


class JsonlWriter:
    """Append-only JSONL writer with flush + fsync per record."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None

    def _ensure_open(self):
        if self._file is None:
            self._file = open(self.path, "a", encoding="utf-8")
        return self._file

    def append_dict(self, record: dict[str, Any]) -> Path:
        serialized = json.dumps(record, default=str, separators=(",", ":"))
        f = self._ensure_open()
        f.write(serialized + "\n")
        f.flush()
        os.fsync(f.fileno())
        return self.path

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> JsonlWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def save_demo_account_state(account_dict: dict[str, Any], path: Path | None = None) -> Path:
    target = path or DEMO_ACCOUNT_STATE_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as f:
        json.dump(account_dict, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    return target


def load_demo_account_state(path: Path | None = None) -> dict[str, Any] | None:
    target = path or DEMO_ACCOUNT_STATE_PATH
    if not target.is_file():
        return None
    with open(target, "r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl_safe(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    return read_jsonl_records_safe(path)
