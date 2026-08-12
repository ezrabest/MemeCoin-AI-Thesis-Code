"""Scoped stale-price status — never a vague global warning when candidate-only."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

DEFAULT_FRESHNESS_LIMIT_SECONDS = 120.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def build_stale_price_status(
    *,
    applies_to: str = "unknown",
    last_price_timestamp: str | None = None,
    freshness_limit_seconds: float = DEFAULT_FRESHNESS_LIMIT_SECONDS,
    source: str | None = None,
    next_refresh_eta: str | None = None,
    affected_symbol: str | None = None,
    affected_pair: str | None = None,
    blocks_demo_trade: bool | None = None,
    market_feed_active: bool | None = None,
) -> dict[str, Any]:
    """Build an actionable stale-price payload."""
    now = datetime.now(timezone.utc)
    pts = _parse_ts(last_price_timestamp)
    age = (now - pts).total_seconds() if pts else None
    limit = float(freshness_limit_seconds or DEFAULT_FRESHNESS_LIMIT_SECONDS)
    is_stale = age is None or age > limit
    if blocks_demo_trade is None:
        blocks_demo_trade = bool(is_stale and applies_to in (
            "selected_candidate",
            "demo_queue",
            "bot_decision",
        ))

    if age is None:
        age_label = "no timestamp"
        age_minutes = None
    else:
        age_minutes = round(age / 60.0, 1)
        if age < 120:
            age_label = f"{int(age)}s"
        else:
            age_label = f"{int(age / 60)}m"

    limit_label = f"{int(limit)}s" if limit < 300 else f"{int(limit / 60)}m"
    sym = affected_symbol or affected_pair or "candidate"

    if applies_to == "global_market" and market_feed_active and is_stale:
        label = (
            f"Some candidates have stale prices. Live Market still updating. "
            f"(oldest/selected age {age_label}, limit {limit_label})"
        )
    elif applies_to == "global_market" and not is_stale:
        label = f"Market feed active. Latest update {age_label} ago."
    elif applies_to in ("selected_candidate", "demo_queue", "bot_decision") and is_stale:
        label = (
            f"Demo trading blocked for {sym}: price age {age_label} exceeds {limit_label} limit."
        )
    elif applies_to == "selected_candidate" and not is_stale:
        label = f"Selected candidate {sym}: price fresh ({age_label})."
    elif is_stale and market_feed_active:
        label = (
            f"Market feed active. Selected demo candidate price is too old "
            f"(age {age_label}, limit {limit_label})."
        )
    elif is_stale:
        label = f"Price too old for confident demo trade ({sym}: age {age_label}, limit {limit_label})."
    else:
        label = f"Price fresh for {sym} ({age_label})."

    return {
        "stale_price_applies_to": applies_to,
        "last_price_timestamp": last_price_timestamp,
        "price_age_seconds": age,
        "price_age_label": age_label,
        "price_age_minutes": age_minutes,
        "freshness_limit_seconds": limit,
        "freshness_limit_label": limit_label,
        "source": source or "unknown",
        "next_refresh_eta": next_refresh_eta,
        "blocks_demo_trade": bool(blocks_demo_trade and is_stale),
        "is_stale": is_stale,
        "affected_symbol": affected_symbol,
        "affected_pair": affected_pair,
        "market_feed_active": market_feed_active,
        "label": label,
        "checked_at": _utc_now(),
    }


def row_price_freshness(
    *,
    price: Any,
    timestamp: str | None,
    symbol: str | None = None,
    pair: str | None = None,
    source: str = "live_market",
    freshness_limit_seconds: float = DEFAULT_FRESHNESS_LIMIT_SECONDS,
) -> dict[str, Any]:
    has_price = price is not None and float(price or 0) > 0
    status = build_stale_price_status(
        applies_to="selected_candidate",
        last_price_timestamp=timestamp,
        freshness_limit_seconds=freshness_limit_seconds,
        source=source,
        affected_symbol=symbol,
        affected_pair=pair,
        blocks_demo_trade=True,
        market_feed_active=True,
    )
    if not has_price:
        status["is_stale"] = True
        status["blocks_demo_trade"] = True
        status["label"] = (
            f"No current price available for {symbol or pair or 'candidate'} paper evaluation."
        )
        status["price_missing"] = True
    else:
        status["price_missing"] = False
    return status
