"""AE18 UI GET isolation instrumentation.

GET/UI paths are strictly read-only: no network, no cache writes, no audit
writes, no index rebuild, no symbol rehydration, no provider refresh.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Any, Iterator

_lock = threading.Lock()
_get_active = False
_counters: dict[str, int] = {
    "external_network_calls_on_get": 0,
    "dexscreener_calls_on_get": 0,
    "helius_calls_on_get": 0,
    "rss_calls_on_get": 0,
    "recursive_audit_scan_on_get": 0,
    "provider_refresh_on_get": 0,
    "cache_write_on_get": 0,
    "audit_write_on_get": 0,
    "index_rebuild_on_get": 0,
    "symbol_rehydration_on_get": 0,
    "runtime_index_read_count": 0,
}

#: Write-side counters that must remain zero on every GET path.
WRITE_COUNTER_KEYS = (
    "cache_write_on_get",
    "audit_write_on_get",
    "index_rebuild_on_get",
    "symbol_rehydration_on_get",
    "provider_refresh_on_get",
)


class UiGetWriteForbidden(RuntimeError):
    """Raised when a write is attempted from a UI GET path."""


def is_ui_get_path_active() -> bool:
    return _get_active


def record_runtime_index_read() -> None:
    if _get_active:
        with _lock:
            _counters["runtime_index_read_count"] += 1


def record_network_attempt(kind: str) -> None:
    """Record a forbidden network attempt during UI GET (should stay 0)."""
    if not _get_active:
        return
    with _lock:
        _counters["external_network_calls_on_get"] += 1
        key = {
            "dexscreener": "dexscreener_calls_on_get",
            "helius": "helius_calls_on_get",
            "rss": "rss_calls_on_get",
            "audit_scan": "recursive_audit_scan_on_get",
            "provider_refresh": "provider_refresh_on_get",
        }.get(kind)
        if key:
            _counters[key] += 1


def record_write_attempt(kind: str, *, detail: str = "") -> bool:
    """Record a forbidden write attempt during UI GET.

    Returns True when the caller is on a GET path and must skip the write.
    """
    if not _get_active:
        return False
    key = {
        "cache": "cache_write_on_get",
        "audit": "audit_write_on_get",
        "index_rebuild": "index_rebuild_on_get",
        "symbol_rehydration": "symbol_rehydration_on_get",
        "provider_refresh": "provider_refresh_on_get",
    }.get(kind)
    with _lock:
        if key:
            _counters[key] += 1
    return True


def assert_not_ui_get(kind: str, *, detail: str = "") -> None:
    """Fail closed when a write-path function is invoked during a UI GET."""
    if record_write_attempt(kind, detail=detail):
        raise UiGetWriteForbidden(f"ui_get_write_forbidden:{kind}:{detail}")


@contextmanager
def ui_get_network_guard(path: str = "") -> Iterator[dict[str, Any]]:
    """Mark the current thread/request as a UI GET hot path (read-only)."""
    global _get_active
    with _lock:
        _get_active = True
        snap_before = dict(_counters)
    meta: dict[str, Any] = {"path": path, "ok": True}
    try:
        yield meta
    finally:
        with _lock:
            _get_active = False
            snap_after = dict(_counters)
        meta["counters_delta"] = {
            k: snap_after[k] - snap_before.get(k, 0) for k in snap_after
        }


def snapshot_counters() -> dict[str, int]:
    with _lock:
        return dict(_counters)


def reset_counters_for_tests() -> None:
    with _lock:
        for k in _counters:
            _counters[k] = 0
