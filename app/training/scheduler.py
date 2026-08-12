"""
Background scheduler for daily training-dataset builds (offline, non-blocking).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

from .dataset_builder import TRAINING_DIR, SUMMARY_FILENAME, build_training_datasets

log = logging.getLogger("training.scheduler")

STATUS_FILENAME = "last_build_status.json"
MANUAL_COMMAND = "python scripts/build_training_dataset.py"
STARTUP_DELAY_SECONDS = 10
POLL_SECONDS = 60


def _truthy(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def is_auto_build_enabled() -> bool:
    return _truthy(os.getenv("AUTO_BUILD_TRAINING_DATASET"), default=False)


def is_build_on_startup() -> bool:
    return _truthy(os.getenv("TRAINING_DATASET_BUILD_ON_STARTUP"), default=False)


def get_build_hour() -> int:
    try:
        return max(0, min(23, int(os.getenv("TRAINING_DATASET_BUILD_HOUR", "3"))))
    except ValueError:
        return 3


def get_build_minute() -> int:
    try:
        return max(0, min(59, int(os.getenv("TRAINING_DATASET_BUILD_MINUTE", "0"))))
    except ValueError:
        return 0


def get_build_interval_hours() -> int:
    try:
        return max(1, int(os.getenv("TRAINING_DATASET_BUILD_INTERVAL_HOURS", "24")))
    except ValueError:
        return 24


def status_file_path() -> Path:
    return TRAINING_DIR / STATUS_FILENAME


def summary_file_path() -> Path:
    return TRAINING_DIR / SUMMARY_FILENAME


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def compute_next_build_at(from_dt: datetime | None = None) -> datetime:
    """Next scheduled build at configured hour/minute, rolling forward by interval."""
    now = from_dt or _utcnow()
    hour = get_build_hour()
    minute = get_build_minute()
    interval = timedelta(hours=get_build_interval_hours())
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if target <= now:
        target += interval
    while target <= now:
        target += interval
    return target


def default_build_status() -> dict[str, Any]:
    return {
        "auto_build_enabled": is_auto_build_enabled(),
        "is_running": False,
        "last_started_at": None,
        "last_finished_at": None,
        "last_success": None,
        "last_error": "",
        "last_duration_seconds": None,
        "last_summary_path": str(summary_file_path()),
        "last_rows_model_ready": None,
        "last_pending_rows": None,
        "next_scheduled_build_at": _iso(compute_next_build_at()) if is_auto_build_enabled() else None,
        "manual_command": MANUAL_COMMAND,
    }


def load_build_status(path: Path | None = None) -> dict[str, Any]:
    file_path = path or status_file_path()
    if not file_path.is_file():
        return default_build_status()
    try:
        with open(file_path, encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            merged = default_build_status()
            merged.update(data)
            merged["auto_build_enabled"] = is_auto_build_enabled()
            merged["manual_command"] = MANUAL_COMMAND
            merged["last_summary_path"] = str(summary_file_path())
            if not is_auto_build_enabled():
                merged["next_scheduled_build_at"] = None
            elif not merged.get("next_scheduled_build_at"):
                merged["next_scheduled_build_at"] = _iso(compute_next_build_at())
            return merged
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not load build status from %s: %s", file_path, exc)
    return default_build_status()


def save_build_status(status: dict[str, Any], path: Path | None = None) -> None:
    file_path = path or status_file_path()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2, default=str)


class TrainingDatasetScheduler:
    """Thread-safe background scheduler; never blocks the async event loop."""

    def __init__(
        self,
        *,
        status_path: Path | None = None,
        build_fn: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self._status_path = status_path
        self._build_fn = build_fn or build_training_datasets
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._scheduler_thread: threading.Thread | None = None
        self._active_thread: threading.Thread | None = None
        self._status = load_build_status(self._status_path)

    def _persist_status(self) -> None:
        save_build_status(self._status, self._status_path)

    def _set_running(self, running: bool) -> None:
        with self._lock:
            self._status["is_running"] = running
        self._persist_status()

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._status.get("is_running"))

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            status = dict(self._status)
        status["auto_build_enabled"] = is_auto_build_enabled()
        status["manual_command"] = MANUAL_COMMAND
        status["last_summary_path"] = str(summary_file_path())
        if is_auto_build_enabled():
            next_at = _parse_iso(status.get("next_scheduled_build_at"))
            if next_at is None or next_at <= _utcnow():
                next_at = compute_next_build_at()
                status["next_scheduled_build_at"] = _iso(next_at)
        else:
            status["next_scheduled_build_at"] = None
        return status

    def request_build(self, reason: str = "manual") -> str:
        """Start a background build unless one is already running."""
        with self._lock:
            if self._status.get("is_running"):
                return "already_running"
            self._status["is_running"] = True
            self._status["last_started_at"] = _iso(_utcnow())
            self._status["last_error"] = ""
            self._persist_status()

        worker = threading.Thread(
            target=self._run_build,
            name=f"training-dataset-build-{reason}",
            daemon=True,
        )
        with self._lock:
            self._active_thread = worker
        worker.start()
        log.info("Training dataset build started (reason=%s)", reason)
        return "started"

    def _run_build(self) -> None:
        started = _parse_iso(self._status.get("last_started_at")) or _utcnow()
        success = False
        error = ""
        duration = 0.0
        rows_ready: int | None = None
        pending_rows: int | None = None

        try:
            report = self._build_fn()
            success = True
            summary = report.get("summary") or {}
            rows_ready = summary.get("rows_model_ready")
            pending_rows = summary.get("pending_outcome_rows")
        except Exception as exc:
            error = str(exc)
            log.error("Training dataset build failed: %s", exc, exc_info=True)
        finally:
            finished = _utcnow()
            duration = (finished - started).total_seconds()
            with self._lock:
                self._status["is_running"] = False
                self._status["last_finished_at"] = _iso(finished)
                self._status["last_success"] = success
                self._status["last_error"] = error
                self._status["last_duration_seconds"] = round(duration, 3)
                self._status["last_rows_model_ready"] = rows_ready
                self._status["last_pending_rows"] = pending_rows
                self._status["last_summary_path"] = str(summary_file_path())
                if is_auto_build_enabled():
                    self._status["next_scheduled_build_at"] = _iso(compute_next_build_at(finished))
                self._active_thread = None
                self._persist_status()

            if success:
                log.info(
                    "Training dataset build completed (%.1fs) — rows model-ready=%s pending=%s summary=%s",
                    duration,
                    rows_ready,
                    pending_rows,
                    summary_file_path(),
                )
            else:
                log.info("Training dataset build failed after %.1fs: %s", duration, error)

    def _scheduler_loop(self) -> None:
        while not self._stop_event.is_set():
            if not is_auto_build_enabled():
                break
            status = self.get_status()
            next_at = _parse_iso(status.get("next_scheduled_build_at"))
            now = _utcnow()
            if next_at and now >= next_at and not self.is_running():
                self.request_build("scheduled")
            self._stop_event.wait(POLL_SECONDS)

    def _startup_build(self) -> None:
        time.sleep(STARTUP_DELAY_SECONDS)
        if not self._stop_event.is_set():
            self.request_build("startup")

    def start(self) -> None:
        self._stop_event.clear()
        self._status = load_build_status(self._status_path)
        self._status["auto_build_enabled"] = is_auto_build_enabled()
        self._status["manual_command"] = MANUAL_COMMAND
        if is_auto_build_enabled():
            self._status["next_scheduled_build_at"] = _iso(compute_next_build_at())
            log.info("Training dataset auto-build scheduler enabled")
            log.info("Next scheduled training dataset build time: %s", self._status["next_scheduled_build_at"])
            self._scheduler_thread = threading.Thread(
                target=self._scheduler_loop,
                name="training-dataset-scheduler",
                daemon=True,
            )
            self._scheduler_thread.start()
        else:
            self._status["next_scheduled_build_at"] = None
            log.info("Training dataset auto-build scheduler disabled")

        self._persist_status()

        if is_build_on_startup():
            threading.Thread(
                target=self._startup_build,
                name="training-dataset-startup-build",
                daemon=True,
            ).start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._scheduler_thread.join(timeout=2.0)


_scheduler: TrainingDatasetScheduler | None = None


def get_training_scheduler() -> TrainingDatasetScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = TrainingDatasetScheduler()
    return _scheduler


def public_build_status(status: dict[str, Any] | None = None) -> dict[str, Any]:
    """API-facing build status with the documented fields only."""
    raw = status or get_training_scheduler().get_status()
    return {
        "auto_build_enabled": raw.get("auto_build_enabled", False),
        "is_running": raw.get("is_running", False),
        "last_started_at": raw.get("last_started_at"),
        "last_finished_at": raw.get("last_finished_at"),
        "last_success": raw.get("last_success"),
        "last_error": raw.get("last_error") or "",
        "last_duration_seconds": raw.get("last_duration_seconds") or 0,
        "last_summary_path": raw.get("last_summary_path") or str(summary_file_path()),
        "next_scheduled_build_at": raw.get("next_scheduled_build_at"),
        "manual_command": raw.get("manual_command") or MANUAL_COMMAND,
    }


def reset_training_scheduler_for_tests() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.stop()
    _scheduler = None
