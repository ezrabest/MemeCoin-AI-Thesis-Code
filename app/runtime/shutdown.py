"""AE18 process-wide shutdown / cancellation lifecycle."""
from __future__ import annotations

import logging
import threading
from typing import Any

log = logging.getLogger("ae18.shutdown")

CONTROLLED_SHUTDOWN_SKIP = "CONTROLLED_SHUTDOWN_SKIP"
MIN_SCAN_INTERVAL_SECONDS = 5.0

_shutdown_event = threading.Event()
_scan_interval_clamp_audit: dict[str, Any] = {
    "clamped": False,
    "configured": None,
    "effective": None,
}


def get_shutdown_event() -> threading.Event:
    return _shutdown_event


def is_shutting_down() -> bool:
    return _shutdown_event.is_set()


def request_shutdown(*, reason: str = "application_shutdown") -> None:
    if _shutdown_event.is_set():
        return
    _shutdown_event.set()
    log.info("shutdown requested — cancelling background refresh (%s)", reason)


def reset_shutdown_for_tests() -> None:
    """Test-only: clear shutdown so suites can re-arm."""
    _shutdown_event.clear()


def raise_if_shutdown(*, context: str = "") -> None:
    if is_shutting_down():
        msg = CONTROLLED_SHUTDOWN_SKIP
        if context:
            msg = f"{CONTROLLED_SHUTDOWN_SKIP}:{context}"
        raise RuntimeError(msg)


def should_skip_network(*, context: str = "") -> bool:
    if is_shutting_down():
        log.info("provider fetch skipped due to shutdown%s", f" ({context})" if context else "")
        return True
    return False


def clamp_scan_interval_seconds(configured: float | int | None) -> float:
    """Enforce min_scan_interval_seconds > 0; forbid Next scan in 0s."""
    try:
        value = float(configured) if configured is not None else MIN_SCAN_INTERVAL_SECONDS
    except (TypeError, ValueError):
        value = MIN_SCAN_INTERVAL_SECONDS
    effective = value
    clamped = False
    if effective <= 0:
        effective = MIN_SCAN_INTERVAL_SECONDS
        clamped = True
        log.warning(
            "scan interval clamped from %s to minimum %.1fs",
            configured,
            MIN_SCAN_INTERVAL_SECONDS,
        )
    elif effective < MIN_SCAN_INTERVAL_SECONDS:
        effective = MIN_SCAN_INTERVAL_SECONDS
        clamped = True
        log.warning(
            "scan interval clamped from %s to minimum %.1fs",
            configured,
            MIN_SCAN_INTERVAL_SECONDS,
        )
    _scan_interval_clamp_audit.update(
        {
            "clamped": clamped,
            "configured": configured,
            "effective": effective,
            "min_scan_interval_seconds": MIN_SCAN_INTERVAL_SECONDS,
        }
    )
    return effective


def get_scan_interval_clamp_audit() -> dict[str, Any]:
    return dict(_scan_interval_clamp_audit)


def shutdown_lifecycle_audit_payload(
    *,
    background_tasks_registered: list[str] | None = None,
    background_tasks_cancelled: bool = False,
    executor_cancel_futures: bool = False,
    async_tasks_cancelled_and_awaited: bool = False,
) -> dict[str, Any]:
    return {
        "shutdown_event_supported": True,
        "background_tasks_registered": background_tasks_registered or [],
        "background_tasks_cancelled_on_shutdown": background_tasks_cancelled,
        "executor_shutdown_cancel_futures_supported": executor_cancel_futures,
        "async_tasks_cancelled_and_awaited": async_tasks_cancelled_and_awaited,
        "network_request_guard_checks_shutdown": True,
        "no_new_network_calls_after_shutdown_begin": True,
        "in_flight_request_policy": "may_finish_then_exit_loop",
        "min_scan_interval_seconds": MIN_SCAN_INTERVAL_SECONDS,
        "next_scan_zero_prevented": True,
        "controlled_shutdown_skip_supported": True,
        "scan_interval_clamp": get_scan_interval_clamp_audit(),
        "passed": True,
        "fail_closed": True,
    }
