"""AE18 structured provider refresh failure reporting.

Replaces generic "refresh unavailable / signal is aborted without reason"
messages with explicit error codes, reasons and recovery instructions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

PROVIDER_REFRESH_DISABLED = "PROVIDER_REFRESH_DISABLED"
PROVIDER_URL_MISSING = "PROVIDER_URL_MISSING"
PROVIDER_HTTP_404 = "PROVIDER_HTTP_404"
PROVIDER_HTTP_429 = "PROVIDER_HTTP_429"
PROVIDER_HTTP_5XX = "PROVIDER_HTTP_5XX"
PROVIDER_RESPONSE_EMPTY = "PROVIDER_RESPONSE_EMPTY"
PROVIDER_PAIR_NOT_FOUND = "PROVIDER_PAIR_NOT_FOUND"
CONTROLLED_SHUTDOWN_SKIP = "CONTROLLED_SHUTDOWN_SKIP"
IDENTITY_UNRESOLVED = "IDENTITY_UNRESOLVED"
NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
URL_FINAL_SEGMENT_CASE_UNSAFE = "URL_FINAL_SEGMENT_CASE_UNSAFE"
UNKNOWN_PROVIDER_REFRESH_ERROR = "UNKNOWN_PROVIDER_REFRESH_ERROR"

ERROR_CODES = (
    PROVIDER_REFRESH_DISABLED,
    PROVIDER_URL_MISSING,
    PROVIDER_HTTP_404,
    PROVIDER_HTTP_429,
    PROVIDER_HTTP_5XX,
    PROVIDER_RESPONSE_EMPTY,
    PROVIDER_PAIR_NOT_FOUND,
    CONTROLLED_SHUTDOWN_SKIP,
    IDENTITY_UNRESOLVED,
    NETWORK_TIMEOUT,
    URL_FINAL_SEGMENT_CASE_UNSAFE,
    UNKNOWN_PROVIDER_REFRESH_ERROR,
)

_RECOVERY: dict[str, str] = {
    PROVIDER_REFRESH_DISABLED: "Enable manual provider refresh, then retry Force Provider Refresh.",
    PROVIDER_URL_MISSING: "Add a provider pair URL for this market in SeedTargets, then rebuild the runtime index.",
    PROVIDER_HTTP_404: "Provider has no record for this pair URL. Verify the URL on the provider site or remove the target.",
    PROVIDER_HTTP_429: "Provider rate limit reached. Wait ~60s and retry the refresh.",
    PROVIDER_HTTP_5XX: "Provider is temporarily unavailable. Retry the refresh in a few minutes.",
    PROVIDER_RESPONSE_EMPTY: "Provider returned no pair payload. Retry with Force Provider Refresh to bypass the cache.",
    PROVIDER_PAIR_NOT_FOUND: "Pair not listed by the provider. Cached display fields are kept; verify the market URL.",
    CONTROLLED_SHUTDOWN_SKIP: "Shutdown in progress. Restart the app and retry the refresh.",
    IDENTITY_UNRESOLVED: "Canonical URL identity could not be resolved. Rebuild the runtime index with the provider URL present.",
    NETWORK_TIMEOUT: "Network timed out before the provider responded. Check connectivity and retry.",
    URL_FINAL_SEGMENT_CASE_UNSAFE: "Provider URL case would be altered. Fix the seed URL casing, then retry.",
    UNKNOWN_PROVIDER_REFRESH_ERROR: "Unclassified refresh error. Check server logs for the exception class and retry.",
}

_USER_MESSAGE: dict[str, str] = {
    PROVIDER_REFRESH_DISABLED: "Provider refresh is disabled.",
    PROVIDER_URL_MISSING: "No provider market URL for this row.",
    PROVIDER_HTTP_404: "Provider does not know this market URL (404).",
    PROVIDER_HTTP_429: "Provider rate limit hit (429).",
    PROVIDER_HTTP_5XX: "Provider server error.",
    PROVIDER_RESPONSE_EMPTY: "Provider returned an empty response.",
    PROVIDER_PAIR_NOT_FOUND: "Provider has no data for this pair.",
    CONTROLLED_SHUTDOWN_SKIP: "Refresh skipped because the app is shutting down.",
    IDENTITY_UNRESOLVED: "Canonical market URL identity unresolved.",
    NETWORK_TIMEOUT: "Provider request timed out.",
    URL_FINAL_SEGMENT_CASE_UNSAFE: "Provider URL casing is unsafe to use.",
    UNKNOWN_PROVIDER_REFRESH_ERROR: "Provider refresh failed for an unclassified reason.",
}

_RETRYABLE = frozenset(
    {
        PROVIDER_HTTP_429,
        PROVIDER_HTTP_5XX,
        PROVIDER_RESPONSE_EMPTY,
        NETWORK_TIMEOUT,
        CONTROLLED_SHUTDOWN_SKIP,
        UNKNOWN_PROVIDER_REFRESH_ERROR,
    }
)

GENERIC_ABORT_MARKERS = (
    "signal is aborted without reason",
    "aborted without reason",
    "the operation was aborted",
    "refresh unavailable",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_refresh_exception(exc: BaseException | None, *, http_status: int | None = None) -> str:
    """Map an exception / HTTP status onto an explicit refresh error code."""
    if http_status is not None:
        if http_status == 404:
            return PROVIDER_HTTP_404
        if http_status == 429:
            return PROVIDER_HTTP_429
        if 500 <= int(http_status) < 600:
            return PROVIDER_HTTP_5XX

    if exc is None:
        return UNKNOWN_PROVIDER_REFRESH_ERROR

    name = type(exc).__name__.lower()
    text = str(exc).lower()

    if CONTROLLED_SHUTDOWN_SKIP.lower() in text:
        return CONTROLLED_SHUTDOWN_SKIP
    if "timeout" in name or "timeout" in text or "timed out" in text:
        return NETWORK_TIMEOUT
    if "abort" in name or "abort" in text or "cancelled" in name:
        return NETWORK_TIMEOUT
    if "404" in text:
        return PROVIDER_HTTP_404
    if "429" in text or "rate limit" in text:
        return PROVIDER_HTTP_429
    if any(code in text for code in ("500", "502", "503", "504")):
        return PROVIDER_HTTP_5XX
    if "not found" in text:
        return PROVIDER_PAIR_NOT_FOUND
    if "empty" in text:
        return PROVIDER_RESPONSE_EMPTY
    return UNKNOWN_PROVIDER_REFRESH_ERROR


def build_refresh_failure(
    *,
    error_code: str,
    provider_url: str = "",
    canonical_market_identity: str = "",
    provider: str = "dexscreener",
    chain: str = "",
    attempted_endpoint: str = "",
    http_status: int | None = None,
    exception: BaseException | None = None,
    reason: str = "",
    shutdown_event_set: bool = False,
) -> dict[str, Any]:
    """Build a structured refresh failure object (never a bare message)."""
    code = error_code if error_code in ERROR_CODES else UNKNOWN_PROVIDER_REFRESH_ERROR
    detail = reason or (str(exception) if exception else "")
    for marker in GENERIC_ABORT_MARKERS:
        if marker in detail.lower():
            detail = f"{_USER_MESSAGE[code]} ({marker})"
            break
    return {
        "refresh_status": "FAILED",
        "refresh_error_code": code,
        "refresh_error_reason": detail or _USER_MESSAGE[code],
        "user_message": _USER_MESSAGE[code],
        "provider_url": provider_url,
        "provider": provider,
        "chain": chain,
        "canonical_market_identity": canonical_market_identity or provider_url,
        "attempted_endpoint": attempted_endpoint,
        "http_status": http_status,
        "exception_class": type(exception).__name__ if exception is not None else "",
        "retryable": code in _RETRYABLE,
        "recovery_instruction": _RECOVERY[code],
        "shutdown_event_set": bool(shutdown_event_set),
        "controlled_shutdown_skip": code == CONTROLLED_SHUTDOWN_SKIP,
        "failed_at": _utc_now(),
    }


def summarize_failures(failures: list[dict[str, Any]]) -> dict[str, Any]:
    """Audit summary for a list of structured failures."""
    generic = 0
    for f in failures:
        reason = str(f.get("refresh_error_reason") or "").lower()
        has_code = bool(f.get("refresh_error_code"))
        if not has_code:
            generic += 1
        elif any(m in reason for m in GENERIC_ABORT_MARKERS) and not f.get("recovery_instruction"):
            generic += 1
    return {
        "refresh_failures_checked": len(failures),
        "generic_abort_without_reason_count": generic,
        "structured_error_count": sum(1 for f in failures if f.get("refresh_error_code")),
        "retryable_count": sum(1 for f in failures if f.get("retryable")),
        "recovery_instruction_present_count": sum(
            1 for f in failures if f.get("recovery_instruction")
        ),
        "controlled_shutdown_skip_count": sum(
            1 for f in failures if f.get("refresh_error_code") == CONTROLLED_SHUTDOWN_SKIP
        ),
        "unknown_provider_refresh_error_count": sum(
            1 for f in failures if f.get("refresh_error_code") == UNKNOWN_PROVIDER_REFRESH_ERROR
        ),
    }
