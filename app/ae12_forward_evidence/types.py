"""Shared types and constants for AE12 forward-evidence maturation."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


AE12_PHASE = "AE12_FORWARD_EVIDENCE_MATURATION"
AE12_SCHEMA_VERSION = "AE12_V1"

HORIZON_SECONDS: dict[str, int] = {
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "6h": 6 * 60 * 60,
    "24h": 24 * 60 * 60,
}

DEFAULT_MISSED_WINNER_THRESHOLDS: dict[str, float] = {
    "5m": 0.10,
    "15m": 0.15,
    "1h": 0.25,
    "6h": 0.50,
    "24h": 1.00,
}

TRADED_ACTIONS = frozenset(
    {
        "FILLED",
        "OPENED",
        "TRADE",
        "TRADE_EXPLORATION_OVERRIDE",
        "PAPER_FILLED",
        "PAPER_OPEN",
    }
)


class ReasonRecoveryStatus(StrEnum):
    RECOVERED_FROM_OPPORTUNITY = "RECOVERED_FROM_OPPORTUNITY"
    RECOVERED_FROM_TRADE_DECISION = "RECOVERED_FROM_TRADE_DECISION"
    RECOVERED_FROM_AE6 = "RECOVERED_FROM_AE6"
    RECOVERED_FROM_RUNTIME_EVENT = "RECOVERED_FROM_RUNTIME_EVENT"
    RECOVERED_FROM_PAPER = "RECOVERED_FROM_PAPER"
    MISSING_IN_SOURCE = "MISSING_IN_SOURCE"
    TRADED_NO_REJECTION = "TRADED_NO_REJECTION"


class QwenLinkageStatus(StrEnum):
    ROW_LINKED_AE9_RECORD = "ROW_LINKED_AE9_RECORD"
    ROW_LINKED_AE6_DECISION = "ROW_LINKED_AE6_DECISION"
    LOG_ONLY_NOT_ROW_LINKED = "LOG_ONLY_NOT_ROW_LINKED"
    MENTION_ONLY = "MENTION_ONLY"
    ABSENT = "ABSENT"


class InterpretationStatus(StrEnum):
    TOO_EARLY_NO_MATURED_HORIZONS = "TOO_EARLY_NO_MATURED_HORIZONS"
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    TRADED_OUTPERFORMED = "TRADED_OUTPERFORMED"
    NOT_TRADED_OUTPERFORMED = "NOT_TRADED_OUTPERFORMED"
    MIXED = "MIXED"
    DATA_GAP = "DATA_GAP"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


@dataclass
class Ae12RunConfig:
    project_root: Any
    output_root: Any
    resume: bool = False
    fail_if_output_exists: bool = True
    max_rows: int | None = None
    horizons: list[str] = field(default_factory=lambda: list(HORIZON_SECONDS.keys()))
    missed_winner_thresholds: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_MISSED_WINNER_THRESHOLDS)
    )
    no_external_apis: bool = True
    no_real_wallet: bool = True
    db_path: Any = None


@dataclass
class HorizonOutcome:
    horizon: str
    horizon_row_id: str
    evidence_row_id: str
    matured: bool
    max_return: float | None = None
    min_return: float | None = None
    last_return: float | None = None
    snapshot_count: int = 0
    computed_at: str | None = None
    price_source: str | None = None
    no_lookahead_status: str = "NOT_MATURED"
    maturity_deadline_utc: str | None = None
    latest_snapshot_utc: str | None = None
