"""AE8 freshness validation utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.context_intelligence.types import (
    DEFAULT_FRESHNESS_THRESHOLDS_MINUTES,
    FreshnessMode,
    FreshnessStatus,
)

STALE_SOURCE_REASON = "STALE_SOURCE"
FUTURE_CONTEXT_LEAKAGE_RISK = "FUTURE_CONTEXT_LEAKAGE_RISK"


def parse_timestamp(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def minutes_between(earlier: datetime, later: datetime) -> float:
    return max(0.0, (later - earlier).total_seconds() / 60.0)


def compute_freshness(
    *,
    source_timestamp: str | None,
    freshness_reference_timestamp: str,
    freshness_mode: FreshnessMode | str,
    threshold_minutes: float,
    family_key: str = "default",
) -> dict[str, Any]:
    """Compute freshness block for a context source family."""
    ref_dt = parse_timestamp(freshness_reference_timestamp)
    src_dt = parse_timestamp(source_timestamp)
    mode = FreshnessMode(freshness_mode) if isinstance(freshness_mode, str) else freshness_mode

    block: dict[str, Any] = {
        "source_timestamp": source_timestamp,
        "freshness_reference_timestamp": freshness_reference_timestamp,
        "freshness_threshold_minutes": threshold_minutes,
        "freshness_minutes": None,
        "freshness_status": FreshnessStatus.NOT_AVAILABLE.value,
        "stale_reason": None,
        "missingness_flag": False,
        "missingness_reason": None,
        "family_key": family_key,
    }

    if ref_dt is None:
        block["freshness_status"] = FreshnessStatus.MISSING_TIMESTAMP.value
        block["stale_reason"] = "missing_freshness_reference_timestamp"
        block["missingness_flag"] = True
        block["missingness_reason"] = "MISSING_REFERENCE_TIMESTAMP"
        return block

    if src_dt is None:
        block["freshness_status"] = FreshnessStatus.MISSING_TIMESTAMP.value
        block["stale_reason"] = "missing_source_timestamp"
        block["missingness_flag"] = True
        block["missingness_reason"] = "MISSING_SOURCE_TIMESTAMP"
        return block

    if src_dt > ref_dt:
        block["freshness_minutes"] = minutes_between(ref_dt, src_dt)
        block["freshness_status"] = FreshnessStatus.INVALID_FUTURE_TIMESTAMP.value
        block["stale_reason"] = "source_timestamp_after_reference"
        block["missingness_flag"] = True
        block["missingness_reason"] = FUTURE_CONTEXT_LEAKAGE_RISK
        return block

    age_minutes = minutes_between(src_dt, ref_dt)
    block["freshness_minutes"] = round(age_minutes, 4)

    if mode == FreshnessMode.HISTORICAL_REPLAY_OR_AUDIT:
        if age_minutes <= threshold_minutes:
            block["freshness_status"] = FreshnessStatus.REPLAY_AS_OF_FRESH.value
        else:
            block["freshness_status"] = FreshnessStatus.FRESH.value
        return block

    if age_minutes <= threshold_minutes:
        block["freshness_status"] = FreshnessStatus.FRESH.value
    else:
        block["freshness_status"] = FreshnessStatus.STALE.value
        block["stale_reason"] = f"source_older_than_{threshold_minutes}_minutes"
        block["missingness_flag"] = True
        block["missingness_reason"] = STALE_SOURCE_REASON

    return block


def apply_stale_nulling(
    features: dict[str, Any],
    freshness_block: dict[str, Any],
    *,
    missingness_flag_key: str,
) -> dict[str, Any]:
    """Null feature values when source is stale or future-leaking in live mode."""
    status = freshness_block.get("freshness_status")
    if status not in {
        FreshnessStatus.STALE.value,
        FreshnessStatus.INVALID_FUTURE_TIMESTAMP.value,
        FreshnessStatus.MISSING_TIMESTAMP.value,
        FreshnessStatus.NOT_AVAILABLE.value,
    }:
        return features

    nulled = dict(features)
    for key in list(nulled.keys()):
        if key in {missingness_flag_key, "freshness_minutes"}:
            continue
        if key.endswith("_missingness_flag"):
            nulled[key] = True
            continue
        if key.endswith("_freshness_minutes"):
            continue
        nulled[key] = None

    nulled[missingness_flag_key] = True
    return nulled


def default_threshold_for_family(family: str) -> float:
    return DEFAULT_FRESHNESS_THRESHOLDS_MINUTES.get(
        family,
        DEFAULT_FRESHNESS_THRESHOLDS_MINUTES["liquidity_activity"],
    )
