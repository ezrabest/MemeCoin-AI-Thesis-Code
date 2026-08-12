"""Append-only JSONL persistence with per-record flush + fsync."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from app.decision.types import DecisionRecord

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DECISION_RECORDS_DIR = DATA_DIR / "decision_records"


def decision_records_path_for_date(dt: datetime | None = None) -> Path:
    """Return daily JSONL path: data/decision_records/ae6_decisions_YYYYMMDD.jsonl."""
    dt = dt or datetime.now(timezone.utc)
    day = dt.strftime("%Y%m%d")
    return DECISION_RECORDS_DIR / f"ae6_decisions_{day}.jsonl"


def serialize_decision_record(record: DecisionRecord) -> str:
    """Serialize a full decision record to compact JSON."""
    return json.dumps(record.model_dump(mode="json"), default=str, separators=(",", ":"))


class DecisionJsonlWriter:
    """Append-only AE6 decision JSONL writer with flush + fsync per record."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or decision_records_path_for_date()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None

    def _ensure_open(self):
        if self._file is None:
            self._file = open(self.path, "a", encoding="utf-8")
        return self._file

    def append_record(self, record: DecisionRecord) -> Path:
        """Write one complete JSON object per line with flush + fsync."""
        serialized = serialize_decision_record(record)
        f = self._ensure_open()
        f.write(serialized + "\n")
        f.flush()
        os.fsync(f.fileno())
        return self.path

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> DecisionJsonlWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def write_decision_record_jsonl(
    record: DecisionRecord,
    *,
    path: Path | None = None,
) -> Path:
    """Convenience one-shot append with fsync."""
    writer = DecisionJsonlWriter(path=path)
    try:
        return writer.append_record(record)
    finally:
        writer.close()


def read_jsonl_records_safe(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read JSONL records; tolerate a final incomplete/corrupt line.

    Returns (records, diagnostics) where diagnostics may include incomplete_line info.
    """
    records: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {"path": str(path), "complete_lines": 0, "incomplete_line": None}

    if not path.is_file():
        diagnostics["status"] = "file_not_found"
        return records, diagnostics

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines[:-1] if lines else []):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            records.append(json.loads(stripped))
            diagnostics["complete_lines"] += 1
        except json.JSONDecodeError as exc:
            diagnostics.setdefault("parse_errors", []).append(
                {"line_number": idx + 1, "error": str(exc)}
            )

    if lines:
        last = lines[-1].strip()
        if last:
            try:
                records.append(json.loads(last))
                diagnostics["complete_lines"] += 1
            except json.JSONDecodeError as exc:
                diagnostics["incomplete_line"] = {
                    "line_number": len(lines),
                    "preview": last[:200],
                    "error": str(exc),
                }

    diagnostics["status"] = "ok"
    diagnostics["record_count"] = len(records)
    return records, diagnostics


def iter_jsonl_records(path: Path) -> Iterator[dict[str, Any]]:
    """Yield parsed records; skip blank lines; stop before corrupt final line."""
    records, _ = read_jsonl_records_safe(path)
    yield from records
