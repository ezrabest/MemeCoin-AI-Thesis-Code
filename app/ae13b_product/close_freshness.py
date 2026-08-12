"""AE13I Smoke Addendum (Part A) — manual close freshness hard guard.

This module is the single source of truth for whether a paper/demo position
close used a genuinely fresh, provider-backed market price or a last-known /
fallback / proposed price. It is intentionally conservative: a close can only
be classified "fresh" when ALL of close_price, price_timestamp, a recognized
fresh source, and a known age within the freshness threshold are present.
Anything else — including missing data, unknown sources, or ages that cannot
be computed — is treated as "unknown_or_fallback". This function must never
be bypassed by a caller-supplied claim of freshness.
"""
from __future__ import annotations

from typing import Any

#: Source strings that are always treated as a fallback / non-provider price.
FALLBACK_SOURCES = {
    "proposed_price",
    "fallback",
    "entry_price",
    "last_known",
    "entry",
    "proposed",
}

#: Source strings that represent a genuine, provider-backed live market price.
FRESH_SOURCES = {
    "provider_live",
    "provider_recent",
    "fresh_market_row",
    "live_market",
    "dexscreener",
    "provider",
}

#: Internal fill-price / API source labels that are equivalent to a fresh
#: provider source. These are real system source values (see
#: app/execution/fill_price.py and app/api.py) that were not literally
#: enumerated in FRESH_SOURCES but carry the same provenance guarantee.
_FRESH_SOURCE_ALIASES = {
    "market_pair_address": "provider_live",
    "market_coin_id": "provider_recent",
    "db": "provider",
    "mark": "provider",
}

#: Internal source labels equivalent to a fallback source.
_FALLBACK_SOURCE_ALIASES = {
    "coin_record_fallback": "fallback",
    "proposed_price_same_pair": "proposed_price",
}

UNKNOWN_OR_FALLBACK = "unknown_or_fallback"
FRESH = "fresh"

REASON_CODE_STALE_OR_FALLBACK = "MANUAL_CLOSE_WITH_STALE_OR_FALLBACK_PRICE"
REASON_CODE_MANUAL_SELL = "MANUAL_SELL"

MANUAL_CLOSE_FALLBACK_WARNING = (
    "Manual close will use last-known / fallback price. This price is not "
    "validated as fresh market data."
)


def _normalize_source(close_price_source: Any) -> str | None:
    if not close_price_source:
        return None
    source = str(close_price_source).strip().lower()
    if not source:
        return None
    return source


def _is_fallback_source(source: str | None) -> bool:
    if source is None:
        return False
    aliased = _FALLBACK_SOURCE_ALIASES.get(source, source)
    return aliased in FALLBACK_SOURCES


def _is_fresh_source(source: str | None) -> bool:
    if source is None:
        return False
    aliased = _FRESH_SOURCE_ALIASES.get(source, source)
    return aliased in FRESH_SOURCES


def classify_manual_close_freshness(
    *,
    close_price: float | None,
    price_timestamp: Any,
    close_price_source: str | None,
    close_price_age_seconds: float | None = None,
    freshness_threshold_seconds: float = 900,
    warning_shown: bool | None = None,
) -> dict[str, Any]:
    """Hard guard: never mark fresh without timestamp + valid fresh source + age within threshold.

    Returns a dict with the authoritative freshness fields for a manual (or
    any) position close. Callers MUST use the returned
    ``close_freshness_status`` / ``close_used_fallback_price`` /
    ``manual_close_warning_shown`` / ``reason_code`` values rather than
    re-deriving their own — this function cannot be overridden by a caller
    claiming a price is fresh.
    """
    source_norm = _normalize_source(close_price_source)

    has_close_price = close_price is not None
    has_timestamp = bool(price_timestamp)
    is_fallback_source = _is_fallback_source(source_norm)
    is_fresh_source = _is_fresh_source(source_norm)
    age = close_price_age_seconds
    has_known_age = age is not None
    age_within_threshold = has_known_age and float(age) <= float(freshness_threshold_seconds)

    is_fresh = (
        has_close_price
        and has_timestamp
        and is_fresh_source
        and not is_fallback_source
        and age_within_threshold
    )

    if is_fresh:
        return {
            "close_freshness_status": FRESH,
            "close_used_fallback_price": False,
            "manual_close_warning_shown": bool(warning_shown),
            "reason_code": REASON_CODE_MANUAL_SELL,
            "close_price_age_seconds": age,
            "close_price_source": source_norm or close_price_source,
            "freshness_threshold_seconds": freshness_threshold_seconds,
            "warning_text": None,
        }

    return {
        "close_freshness_status": UNKNOWN_OR_FALLBACK,
        "close_used_fallback_price": True,
        "manual_close_warning_shown": True,
        "reason_code": REASON_CODE_STALE_OR_FALLBACK,
        "close_price_age_seconds": age,
        "close_price_source": source_norm or close_price_source,
        "freshness_threshold_seconds": freshness_threshold_seconds,
        "warning_text": MANUAL_CLOSE_FALLBACK_WARNING,
    }
