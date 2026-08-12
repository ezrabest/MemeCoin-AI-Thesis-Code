"""AE18 provider / display / market / identity / trade-readiness status contracts.

Hard separation enforced here:
  * Display metadata failures (missing symbol_pair_display) are display-only.
  * Market data failures (price/liquidity/freshness) drive trade blocking.
  * Identity + position continuity failures drive trade blocking.

A missing symbol alone must NEVER produce an ENTRY_BLOCKED_* status, and no
block reason may ever be SYMBOL_MISSING_ONLY.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# ---------------------------------------------------------------------------
# provider_resolution_status
# ---------------------------------------------------------------------------
RESOLVED = "RESOLVED"
RESOLVED_WITH_LAST_GOOD_DISPLAY = "RESOLVED_WITH_LAST_GOOD_DISPLAY"
RESOLVED_WITH_MANUAL_DISPLAY_OVERRIDE = "RESOLVED_WITH_MANUAL_DISPLAY_OVERRIDE"
MARKET_DATA_AVAILABLE_SYMBOLS_MISSING = "MARKET_DATA_AVAILABLE_SYMBOLS_MISSING"
PROVIDER_PAIR_NOT_FOUND = "PROVIDER_PAIR_NOT_FOUND"
PROVIDER_API_DEGRADED = "PROVIDER_API_DEGRADED"
PROVIDER_RATE_LIMITED = "PROVIDER_RATE_LIMITED"
PROVIDER_FORBIDDEN_WEBPAGE = "PROVIDER_FORBIDDEN_WEBPAGE"
PROVIDER_RESPONSE_AMBIGUOUS = "PROVIDER_RESPONSE_AMBIGUOUS"
PROVIDER_RESPONSE_PARTIAL = "PROVIDER_RESPONSE_PARTIAL"
UNRESOLVED = "UNRESOLVED"

PROVIDER_RESOLUTION_STATUSES = frozenset(
    {
        RESOLVED,
        RESOLVED_WITH_LAST_GOOD_DISPLAY,
        RESOLVED_WITH_MANUAL_DISPLAY_OVERRIDE,
        MARKET_DATA_AVAILABLE_SYMBOLS_MISSING,
        PROVIDER_PAIR_NOT_FOUND,
        PROVIDER_API_DEGRADED,
        PROVIDER_RATE_LIMITED,
        PROVIDER_FORBIDDEN_WEBPAGE,
        PROVIDER_RESPONSE_AMBIGUOUS,
        PROVIDER_RESPONSE_PARTIAL,
        UNRESOLVED,
    }
)

# ---------------------------------------------------------------------------
# display_metadata_status  (display degradation only — never blocks trading)
# ---------------------------------------------------------------------------
DISPLAY_READY = "DISPLAY_READY"
SYMBOL_PAIR_UNAVAILABLE_AFTER_PROVIDER_PROBE = "SYMBOL_PAIR_UNAVAILABLE_AFTER_PROVIDER_PROBE"
SYMBOL_PAIR_FROM_LAST_GOOD = "SYMBOL_PAIR_FROM_LAST_GOOD"
SYMBOL_PAIR_FROM_MANUAL_OVERRIDE = "SYMBOL_PAIR_FROM_MANUAL_OVERRIDE"

DISPLAY_METADATA_STATUSES = frozenset(
    {
        DISPLAY_READY,
        SYMBOL_PAIR_UNAVAILABLE_AFTER_PROVIDER_PROBE,
        SYMBOL_PAIR_FROM_LAST_GOOD,
        SYMBOL_PAIR_FROM_MANUAL_OVERRIDE,
    }
)

# ---------------------------------------------------------------------------
# symbol_resolution_status
# ---------------------------------------------------------------------------
SYMBOL_PAIR_RESOLVED = "SYMBOL_PAIR_RESOLVED"
PARTIAL_PROVIDER_SYMBOLS_MISSING = "PARTIAL_PROVIDER_SYMBOLS_MISSING"
SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING = "SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING"

SYMBOL_RESOLUTION_STATUSES = frozenset(
    {
        SYMBOL_PAIR_RESOLVED,
        SYMBOL_PAIR_FROM_LAST_GOOD,
        SYMBOL_PAIR_FROM_MANUAL_OVERRIDE,
        SYMBOL_PAIR_UNAVAILABLE_AFTER_PROVIDER_PROBE,
        PARTIAL_PROVIDER_SYMBOLS_MISSING,
        SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING,
    }
)

# ---------------------------------------------------------------------------
# market_data_status
# ---------------------------------------------------------------------------
MARKET_DATA_READY = "MARKET_DATA_READY"
MARKET_DATA_STALE = "MARKET_DATA_STALE"
MARKET_DATA_MISSING = "MARKET_DATA_MISSING"
MARKET_DATA_UNVERIFIABLE = "MARKET_DATA_UNVERIFIABLE"
MARKET_DATA_PROVIDER_UNAVAILABLE = "MARKET_DATA_PROVIDER_UNAVAILABLE"

MARKET_DATA_STATUSES = frozenset(
    {
        MARKET_DATA_READY,
        MARKET_DATA_STALE,
        MARKET_DATA_MISSING,
        MARKET_DATA_UNVERIFIABLE,
        MARKET_DATA_PROVIDER_UNAVAILABLE,
    }
)

# ---------------------------------------------------------------------------
# identity_readiness_status
# ---------------------------------------------------------------------------
IDENTITY_READY = "IDENTITY_READY"
IDENTITY_PARTIAL = "IDENTITY_PARTIAL"
IDENTITY_UNRESOLVED = "IDENTITY_UNRESOLVED"
IDENTITY_PROVIDER_URL_ONLY = "IDENTITY_PROVIDER_URL_ONLY"
IDENTITY_UNVERIFIABLE = "IDENTITY_UNVERIFIABLE"

IDENTITY_READINESS_STATUSES = frozenset(
    {
        IDENTITY_READY,
        IDENTITY_PARTIAL,
        IDENTITY_UNRESOLVED,
        IDENTITY_PROVIDER_URL_ONLY,
        IDENTITY_UNVERIFIABLE,
    }
)

#: Identity states that are sufficient to price/trade a canonical market.
IDENTITY_TRADEABLE_STATUSES = frozenset(
    {IDENTITY_READY, IDENTITY_PARTIAL, IDENTITY_PROVIDER_URL_ONLY}
)

# ---------------------------------------------------------------------------
# trade_readiness_status
# ---------------------------------------------------------------------------
PAPER_ELIGIBLE = "PAPER_ELIGIBLE"
WATCH_ONLY = "WATCH_ONLY"
ENTRY_BLOCKED_MARKET_DATA_MISSING = "ENTRY_BLOCKED_MARKET_DATA_MISSING"
ENTRY_BLOCKED_MARKET_DATA_STALE = "ENTRY_BLOCKED_MARKET_DATA_STALE"
ENTRY_BLOCKED_MARKET_DATA_UNVERIFIABLE = "ENTRY_BLOCKED_MARKET_DATA_UNVERIFIABLE"
ENTRY_BLOCKED_IDENTITY_UNRESOLVED = "ENTRY_BLOCKED_IDENTITY_UNRESOLVED"
ENTRY_BLOCKED_POSITION_CONTINUITY_UNSAFE = "ENTRY_BLOCKED_POSITION_CONTINUITY_UNSAFE"
MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"

#: Legacy alias retained for backward compatibility; no longer emitted.
ENTRY_BLOCKED_PROVIDER_DEGRADED = "ENTRY_BLOCKED_PROVIDER_DEGRADED"
#: Legacy position-context alias retained for backward compatibility.
EXIT_ONLY_CONTEXT_REQUIRED = "EXIT_ONLY_CONTEXT_REQUIRED"

TRADE_READINESS_STATUSES = frozenset(
    {
        PAPER_ELIGIBLE,
        WATCH_ONLY,
        ENTRY_BLOCKED_MARKET_DATA_MISSING,
        ENTRY_BLOCKED_MARKET_DATA_STALE,
        ENTRY_BLOCKED_MARKET_DATA_UNVERIFIABLE,
        ENTRY_BLOCKED_IDENTITY_UNRESOLVED,
        ENTRY_BLOCKED_POSITION_CONTINUITY_UNSAFE,
        MANUAL_REVIEW_REQUIRED,
    }
)

#: Statuses that forbid opening a new paper/demo entry.
ENTRY_BLOCKED_STATUSES = frozenset(
    {
        WATCH_ONLY,
        ENTRY_BLOCKED_MARKET_DATA_MISSING,
        ENTRY_BLOCKED_MARKET_DATA_STALE,
        ENTRY_BLOCKED_MARKET_DATA_UNVERIFIABLE,
        ENTRY_BLOCKED_IDENTITY_UNRESOLVED,
        ENTRY_BLOCKED_POSITION_CONTINUITY_UNSAFE,
        ENTRY_BLOCKED_PROVIDER_DEGRADED,
        EXIT_ONLY_CONTEXT_REQUIRED,
        MANUAL_REVIEW_REQUIRED,
    }
)

#: Block reason vocabulary. SYMBOL_MISSING_ONLY is deliberately absent and is
#: rejected by assert_block_reason_not_symbol_only().
FORBIDDEN_BLOCK_REASON = "SYMBOL_MISSING_ONLY"

BLOCK_REASON_BY_STATUS = {
    ENTRY_BLOCKED_MARKET_DATA_MISSING: "MARKET_DATA_MISSING",
    ENTRY_BLOCKED_MARKET_DATA_STALE: "MARKET_DATA_STALE",
    ENTRY_BLOCKED_MARKET_DATA_UNVERIFIABLE: "MARKET_DATA_UNVERIFIABLE",
    ENTRY_BLOCKED_IDENTITY_UNRESOLVED: "IDENTITY_UNRESOLVED_OR_UNVERIFIABLE",
    ENTRY_BLOCKED_POSITION_CONTINUITY_UNSAFE: "POSITION_CONTINUITY_UNSAFE",
    MANUAL_REVIEW_REQUIRED: "MANUAL_REVIEW_REQUIRED",
    WATCH_ONLY: "WATCH_ONLY_RISK_RULES",
}

# ---------------------------------------------------------------------------
# position_market_data_state
# ---------------------------------------------------------------------------
DATA_OK = "DATA_OK"
DATA_STALE = "DATA_STALE"
DATA_DEGRADED = "DATA_DEGRADED"
PRICE_UNAVAILABLE = "PRICE_UNAVAILABLE"
POSITION_EXIT_ONLY_CONTEXT_REQUIRED = "EXIT_ONLY_CONTEXT_REQUIRED"
POSITION_MANUAL_REVIEW_REQUIRED = "MANUAL_REVIEW_REQUIRED"

POSITION_MARKET_DATA_STATES = frozenset(
    {
        DATA_OK,
        DATA_STALE,
        DATA_DEGRADED,
        PRICE_UNAVAILABLE,
        POSITION_EXIT_ONLY_CONTEXT_REQUIRED,
        POSITION_MANUAL_REVIEW_REQUIRED,
    }
)

DEFAULT_MAX_STALENESS_SECONDS = 86400.0  # align with runtime index 24h threshold
POSITION_MAX_STALENESS_SECONDS = 900.0  # align with PaperTrader MAX_FRESH_MARK_AGE_SECONDS


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _has_usable_price(row: dict[str, Any]) -> bool:
    price = row.get("price_usd")
    if price is None or price == "":
        price = row.get("price")
    if price is None or price == "":
        return False
    try:
        return float(price) > 0
    except (TypeError, ValueError):
        return False


def _parse_age_seconds(ts: Any) -> float | None:
    if not ts:
        return None
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return max(0.0, (datetime.now(timezone.utc) - t).total_seconds())
    except ValueError:
        return None


def is_proper_symbol_pair_display(display: Any) -> bool:
    """True only for a real BASE/QUOTE symbol pair (never a status token)."""
    text = _cell(display)
    if not text or text == "-":
        return False
    if "/" not in text:
        return False
    banned = ("UNAVAILABLE", "MISSING", "PARTIAL", "SYMBOL_PAIR_", "SYMBOLS_")
    return not any(tok in text for tok in banned)


def classify_display_metadata_status(
    *,
    has_proper_display: bool,
    from_last_good: bool = False,
    from_manual_override: bool = False,
) -> str:
    """Display-only classification. Never consulted for trade blocking."""
    if from_manual_override:
        return SYMBOL_PAIR_FROM_MANUAL_OVERRIDE
    if from_last_good:
        return SYMBOL_PAIR_FROM_LAST_GOOD
    if has_proper_display:
        return DISPLAY_READY
    return SYMBOL_PAIR_UNAVAILABLE_AFTER_PROVIDER_PROBE


def classify_identity_readiness(row: dict[str, Any]) -> str:
    """Identity readiness from URL-first canonical fields only (no symbols)."""
    from app.clean_forward.provider_url_key import try_normalize_provider_pair_url_key

    exact = _cell(
        row.get("provider_pair_url_exact")
        or row.get("canonical_market_identity")
        or row.get("provider_pair_url")
    )
    if not exact:
        return IDENTITY_UNRESOLVED

    key, _ = try_normalize_provider_pair_url_key(exact, require_dexscreener=False)
    if not key:
        return IDENTITY_UNVERIFIABLE

    final_segment = _cell(row.get("provider_pair_url_final_segment_exact"))
    chain = _cell(row.get("chain"))
    base_addr = _cell(
        row.get("provider_base_token_address") or row.get("base_token_address_derived")
    )
    quote_addr = _cell(
        row.get("provider_quote_token_address") or row.get("quote_token_address_derived")
    )

    if final_segment and chain and base_addr and quote_addr:
        return IDENTITY_READY
    if final_segment and chain and (base_addr or quote_addr):
        return IDENTITY_PARTIAL
    if final_segment and chain:
        return IDENTITY_PROVIDER_URL_ONLY
    return IDENTITY_PARTIAL


def classify_market_data_status(
    row: dict[str, Any],
    *,
    max_staleness_seconds: float = DEFAULT_MAX_STALENESS_SECONDS,
) -> str:
    """Classify market data independently of symbol display availability."""
    freshness = _cell(row.get("freshness_status")).lower()
    has_price = _has_usable_price(row)
    age = _parse_age_seconds(
        row.get("last_market_update_at")
        or row.get("price_timestamp")
        or row.get("current_price_timestamp")
    )

    if not has_price:
        if _cell(row.get("provider_resolution_status")) in {
            PROVIDER_API_DEGRADED,
            PROVIDER_RATE_LIMITED,
            PROVIDER_FORBIDDEN_WEBPAGE,
            PROVIDER_PAIR_NOT_FOUND,
        }:
            return MARKET_DATA_PROVIDER_UNAVAILABLE
        return MARKET_DATA_MISSING

    # Price exists but cannot be tied to the canonical market identity.
    if row.get("safe_for_price_lookup") is False:
        return MARKET_DATA_UNVERIFIABLE
    lookup_status = _cell(row.get("mark_price_lookup_status")).upper()
    if lookup_status in {"PRICE_NOT_AVAILABLE", "LEGACY_POSITION_IDENTITY_REPAIR_NEEDED"}:
        return MARKET_DATA_UNVERIFIABLE

    if freshness in {"stale", "degraded"}:
        return MARKET_DATA_STALE
    if age is not None and age > max_staleness_seconds:
        return MARKET_DATA_STALE
    return MARKET_DATA_READY


def classify_trade_readiness(
    *,
    market_data_status: str,
    identity_readiness_status: str = IDENTITY_READY,
    position_continuity_safe: bool = True,
    manual_review_required: bool = False,
    identity_ok: bool | None = None,
    provider_resolution_status: str | None = None,
) -> str:
    """Derive trade readiness from market data + identity + continuity ONLY.

    Symbol/display state is intentionally not an input: a missing
    symbol_pair_display can never produce an ENTRY_BLOCKED_* status.

    ``identity_ok`` and ``provider_resolution_status`` are accepted for
    backward compatibility; ``provider_resolution_status`` is ignored because
    it carries display-resolution meaning.
    """
    identity = _cell(identity_readiness_status) or IDENTITY_READY
    if identity_ok is False:
        identity = IDENTITY_UNRESOLVED

    if manual_review_required:
        return MANUAL_REVIEW_REQUIRED
    if identity in {IDENTITY_UNRESOLVED, IDENTITY_UNVERIFIABLE}:
        return ENTRY_BLOCKED_IDENTITY_UNRESOLVED
    if not position_continuity_safe:
        return ENTRY_BLOCKED_POSITION_CONTINUITY_UNSAFE

    market = _cell(market_data_status)
    if market in {MARKET_DATA_MISSING, MARKET_DATA_PROVIDER_UNAVAILABLE}:
        return ENTRY_BLOCKED_MARKET_DATA_MISSING
    if market == MARKET_DATA_UNVERIFIABLE:
        return ENTRY_BLOCKED_MARKET_DATA_UNVERIFIABLE
    if market == MARKET_DATA_STALE:
        return ENTRY_BLOCKED_MARKET_DATA_STALE
    if market == MARKET_DATA_READY:
        return PAPER_ELIGIBLE
    return WATCH_ONLY


def block_reason_for(trade_readiness_status: str) -> str:
    """Explicit non-symbol block reason for a trade readiness status."""
    status = _cell(trade_readiness_status)
    if status in {PAPER_ELIGIBLE, ""}:
        return ""
    return BLOCK_REASON_BY_STATUS.get(status, status)


def assert_block_reason_not_symbol_only(reason: str) -> bool:
    """True when the block reason is a legitimate non-display reason."""
    return _cell(reason).upper() != FORBIDDEN_BLOCK_REASON


def classify_position_market_data_state(
    *,
    has_fresh_current_price: bool,
    has_any_price: bool,
    is_stale: bool,
    is_partial: bool = False,
    require_exit_only: bool = False,
    require_manual_review: bool = False,
) -> str:
    if require_manual_review:
        return POSITION_MANUAL_REVIEW_REQUIRED
    if require_exit_only:
        return POSITION_EXIT_ONLY_CONTEXT_REQUIRED
    if has_fresh_current_price:
        return DATA_OK
    if is_stale and has_any_price:
        return DATA_STALE
    if is_partial and has_any_price:
        return DATA_DEGRADED
    if not has_any_price:
        return PRICE_UNAVAILABLE
    return DATA_DEGRADED


def entry_blocked(trade_readiness_status: str) -> bool:
    return _cell(trade_readiness_status) in ENTRY_BLOCKED_STATUSES
