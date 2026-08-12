"""Types and constants for AE12-SentimentFix dual-axis taxonomy."""

from __future__ import annotations

from typing import Any, TypedDict

AE12_SENTIMENTFIX_PHASE = "AE12-SentimentFix"
AE12_SENTIMENTFIX_SCHEMA = "AE12_SENTIMENTFIX_V1"

SEMANTIC_SIGNAL_FAMILIES: tuple[str, ...] = (
    "SOCIAL",
    "NEWS",
    "ONCHAIN",
    "PRICE_MOMENTUM",
    "LIQUIDITY",
    "WHALE",
    "LLM_CONTEXT",
    "MIXED",
    "UNKNOWN",
    "UNCLASSIFIED",
)

TRADING_OPPORTUNITY_STATES: tuple[str, ...] = (
    "OPPORTUNISTIC",
    "EXPLORATION",
    "STRICT_BLOCKED",
    "NO_TRADE",
    "PAPER_TRADED",
    "UNKNOWN",
)

SEMANTIC_UNKNOWN_SHARE_THRESHOLD = 0.50

GATE_STATUSES: tuple[str, ...] = (
    "PASS_DUAL_AXIS_READY",
    "PASS_DERIVED_ONLY_RUNTIME_UPDATE_PENDING",
    "HOLD_RUNTIME_WRITER_UPDATE_REQUIRED",
    "HOLD_SEMANTIC_LINKAGE_GAP",
    "HOLD_MANUAL_REVIEW_REQUIRED",
    "FAIL_DEFAULT_FALLBACK_STILL_PRESENT",
    "FAIL_STICKY_CLUSTER_STILL_AUTHORITATIVE",
    "FAIL_CONFLATED_AXIS_STILL_AUTHORITATIVE",
)


class DualAxisResult(TypedDict):
    semantic_signal_family: str
    semantic_signal_source: str
    semantic_signal_confidence: float
    semantic_signal_reason: str
    trading_opportunity_state: str
    trading_state_source: str
    legacy_cluster_label: str | None
    taxonomy_status: str


def null_safe_str(value: Any, default: str = "UNKNOWN") -> str:
    if value is None:
        return default
    text = str(value).strip()
    return text if text else default


def null_safe_warning_code(row: dict[str, Any] | None) -> str:
    """Null-safe warning_code extraction - never KeyError."""
    if not row:
        return "UNKNOWN"
    raw = (
        row.get("warning_code")
        or row.get("missing_field")
        or row.get("warning")
        or "UNKNOWN"
    )
    return null_safe_str(raw, "UNKNOWN").upper()
