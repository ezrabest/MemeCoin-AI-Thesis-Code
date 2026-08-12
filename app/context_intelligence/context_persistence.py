"""AE8 append-only JSONL persistence for context feature records."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent.parent.parent / "data"
CONTEXT_INTELLIGENCE_DIR = DATA_DIR / "context_intelligence"


def context_jsonl_path_for_date(dt: datetime | None = None, output_root: Path | None = None) -> Path:
    dt = dt or datetime.now(timezone.utc)
    day = dt.strftime("%Y%m%d")
    base = output_root or CONTEXT_INTELLIGENCE_DIR
    return base / f"ae8_context_features_{day}.jsonl"


class ContextJsonlWriter:
    """Append-only JSONL writer with flush + fsync per record."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = None
        self.records_written = 0

    def _ensure_open(self):
        if self._file is None:
            self._file = open(self.path, "a", encoding="utf-8")
        return self._file

    def append_record(self, record: dict[str, Any]) -> Path:
        serialized = json.dumps(record, default=str, separators=(",", ":"), ensure_ascii=False)
        f = self._ensure_open()
        f.write(serialized + "\n")
        f.flush()
        os.fsync(f.fileno())
        self.records_written += 1
        return self.path

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self) -> ContextJsonlWriter:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def read_context_jsonl_safe(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
