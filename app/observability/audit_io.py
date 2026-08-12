"""Atomic JSON reports and append-only JSONL audit writers (daily partition)."""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

log = logging.getLogger("audit_io")

DATA_DIR = Path(__file__).parent.parent.parent / "data"
AUDITS_DIR = DATA_DIR / "audits"


def utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_date_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def ensure_audits_dir() -> Path:
    AUDITS_DIR.mkdir(parents=True, exist_ok=True)
    return AUDITS_DIR


def new_decision_trace_id() -> str:
    return str(uuid.uuid4())


def write_json_report_atomic(filename: str, payload: dict[str, Any]) -> Path:
    """Write a static JSON report via temp file + atomic rename."""
    ensure_audits_dir()
    final_path = AUDITS_DIR / filename
    tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, final_path)
    return final_path


class JsonlAuditWriter:
    """Append-only JSONL writer — one file per UTC day, not per candidate/scan."""

    def __init__(self, basename: str, *, date_slug: str | None = None) -> None:
        ensure_audits_dir()
        day = date_slug or utc_date_slug()
        self.path = AUDITS_DIR / f"{basename}_{day}.jsonl"
        self._file: TextIO | None = None

    def _ensure_open(self) -> TextIO:
        if self._file is None:
            self._file = open(self.path, "a", encoding="utf-8")
        return self._file

    def append(self, record: dict[str, Any]) -> None:
        try:
            line = json.dumps(record, default=str, separators=(",", ":"))
            f = self._ensure_open()
            f.write(line + "\n")
            f.flush()
        except Exception as exc:
            log.warning("JSONL audit append failed (%s): %s", self.path.name, exc)

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None


_pipeline_writer: JsonlAuditWriter | None = None
_trace_writer: JsonlAuditWriter | None = None


def get_pipeline_reasons_writer() -> JsonlAuditWriter:
    global _pipeline_writer
    if _pipeline_writer is None:
        _pipeline_writer = JsonlAuditWriter("pipeline_reasons")
    return _pipeline_writer


def get_decision_trace_writer() -> JsonlAuditWriter:
    global _trace_writer
    if _trace_writer is None:
        _trace_writer = JsonlAuditWriter("decision_trace")
    return _trace_writer


def reset_audit_writers_for_tests() -> None:
    """Close and clear singleton writers (test helper)."""
    global _pipeline_writer, _trace_writer
    if _pipeline_writer is not None:
        _pipeline_writer.close()
        _pipeline_writer = None
    if _trace_writer is not None:
        _trace_writer.close()
        _trace_writer = None
