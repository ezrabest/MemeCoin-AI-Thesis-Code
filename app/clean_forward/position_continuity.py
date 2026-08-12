"""AE18 position continuity snapshot + financial DTO safety.

Contract/stabilization only — no AE20 live risk engine / live exit logic.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from app.clean_forward.provider_resilience_statuses import (
    DATA_OK,
    DATA_STALE,
    DATA_DEGRADED,
    DEFAULT_MAX_STALENESS_SECONDS,
    ENTRY_BLOCKED_MARKET_DATA_MISSING,
    MARKET_DATA_MISSING,
    MARKET_DATA_READY,
    MARKET_DATA_STALE,
    PAPER_ELIGIBLE,
    POSITION_EXIT_ONLY_CONTEXT_REQUIRED,
    POSITION_MANUAL_REVIEW_REQUIRED,
    POSITION_MAX_STALENESS_SECONDS,
    PRICE_UNAVAILABLE,
    assert_block_reason_not_symbol_only,
    block_reason_for,
    classify_position_market_data_state,
    entry_blocked,
)
from app.clean_forward.provider_url_key import try_normalize_provider_pair_url_key

SNAPSHOT_FIELDS = [
    "position_id",
    "order_id",
    "candidate_id",
    "provider_pair_url_exact",
    "normalized_provider_pair_url_key",
    "canonical_market_identity",
    "canonical_market_identity_type",
    "chain",
    "provider_pair_url_final_segment_exact",
    "provider_base_token_address",
    "provider_quote_token_address",
    "provider_base_token_symbol",
    "provider_quote_token_symbol",
    "symbol_pair_display_at_entry",
    "entry_price",
    "entry_price_source",
    "entry_price_timestamp",
    "entry_liquidity_usd",
    "entry_volume_h24",
    "entry_market_data_status",
    "entry_provider_resolution_status",
    "entry_symbol_resolution_status",
    "entry_trade_readiness_status",
    "last_good_price",
    "last_good_price_timestamp",
    "last_good_market_data_source",
    "max_data_staleness_allowed_seconds",
    "data_loss_policy",
    "provenance",
]


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_entry_continuity_snapshot(
    *,
    position: dict[str, Any],
    coin: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build local entry snapshot so identity survives provider metadata loss."""
    coin = coin or {}
    exact = _cell(
        position.get("provider_pair_url_exact")
        or position.get("canonical_market_identity")
        or coin.get("provider_pair_url_exact")
        or coin.get("canonical_market_identity")
        or coin.get("provider_pair_url")
    )
    key, _ = try_normalize_provider_pair_url_key(exact, require_dexscreener=False)
    if key is None:
        key, _ = try_normalize_provider_pair_url_key(exact, require_dexscreener=True)

    entry_price = position.get("entry_price")
    entry_ts = _cell(position.get("opened_at") or position.get("entry_price_timestamp")) or _utc_now()
    display = _cell(
        position.get("symbol_pair_display")
        or coin.get("symbol_pair_display")
        or position.get("symbol")
    )

    snapshot = {
        "position_id": position.get("id") or position.get("position_id"),
        "order_id": position.get("order_id") or coin.get("order_id") or "",
        "candidate_id": position.get("candidate_id")
        or coin.get("candidate_id")
        or coin.get("decision_ref_id")
        or "",
        "provider_pair_url_exact": exact,
        "normalized_provider_pair_url_key": key or "",
        "canonical_market_identity": _cell(
            position.get("canonical_market_identity") or exact
        ),
        "canonical_market_identity_type": _cell(
            position.get("canonical_market_identity_type")
            or coin.get("canonical_market_identity_type")
            or "PROVIDER_URL"
        ),
        "chain": _cell(position.get("chain") or coin.get("chain")),
        "provider_pair_url_final_segment_exact": _cell(
            position.get("provider_pair_url_final_segment_exact")
            or coin.get("provider_pair_url_final_segment_exact")
        ),
        "provider_base_token_address": _cell(
            position.get("provider_base_token_address")
            or coin.get("provider_base_token_address")
            or position.get("base_token_address")
            or coin.get("base_token_address")
        ),
        "provider_quote_token_address": _cell(
            position.get("provider_quote_token_address")
            or coin.get("provider_quote_token_address")
            or position.get("quote_token_address")
            or coin.get("quote_token_address")
        ),
        "provider_base_token_symbol": _cell(
            position.get("provider_base_token_symbol")
            or coin.get("provider_base_token_symbol")
        ),
        "provider_quote_token_symbol": _cell(
            position.get("provider_quote_token_symbol")
            or coin.get("provider_quote_token_symbol")
        ),
        "symbol_pair_display_at_entry": display,
        "entry_price": entry_price,
        "entry_price_source": _cell(
            position.get("fill_price_source")
            or position.get("entry_price_source")
            or coin.get("entry_price_source")
        ),
        "entry_price_timestamp": entry_ts,
        "entry_liquidity_usd": position.get("liquidity_at_entry")
        or coin.get("liquidity_usd")
        or coin.get("latest_liquidity"),
        "entry_volume_h24": coin.get("volume_h24") or coin.get("volume_24h"),
        "entry_market_data_status": _cell(
            coin.get("market_data_status") or MARKET_DATA_READY
        ),
        "entry_provider_resolution_status": _cell(
            coin.get("provider_resolution_status") or "RESOLVED"
        ),
        "entry_symbol_resolution_status": _cell(
            coin.get("symbol_resolution_status") or "SYMBOL_PAIR_RESOLVED"
        ),
        "entry_trade_readiness_status": _cell(
            coin.get("trade_readiness_status") or PAPER_ELIGIBLE
        ),
        "last_good_price": entry_price,
        "last_good_price_timestamp": entry_ts,
        "last_good_market_data_source": _cell(
            position.get("fill_price_source") or "entry_fill"
        ),
        "max_data_staleness_allowed_seconds": POSITION_MAX_STALENESS_SECONDS,
        "data_loss_policy": (
            "KEEP_POSITION_VISIBLE;"
            "MISSING_SYMBOL_IS_DISPLAY_ONLY;"
            "STALE_OR_MISSING_PRICE_SETS_POSITION_MARKET_DATA_STATE;"
            "NEVER_USE_LAST_GOOD_AS_CURRENT_TRADABLE_PRICE"
        ),
        "provenance": "ae18_position_continuity_entry_snapshot",
    }
    return snapshot


def attach_entry_snapshot_to_position(
    position: dict[str, Any],
    coin: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mutate position dict with continuity snapshot fields (local only)."""
    snap = build_entry_continuity_snapshot(position=position, coin=coin)
    position["entry_continuity_snapshot"] = snap
    for field in SNAPSHOT_FIELDS:
        # Flatten key identity fields onto position for resilient lookups.
        if field in {
            "provider_pair_url_exact",
            "normalized_provider_pair_url_key",
            "canonical_market_identity",
            "canonical_market_identity_type",
            "chain",
            "provider_pair_url_final_segment_exact",
            "provider_base_token_address",
            "provider_quote_token_address",
            "last_good_price",
            "last_good_price_timestamp",
            "last_good_market_data_source",
            "max_data_staleness_allowed_seconds",
            "data_loss_policy",
        }:
            if snap.get(field) not in (None, ""):
                position[field] = snap[field]
    position["symbol_pair_display_at_entry"] = snap.get("symbol_pair_display_at_entry")
    position["entry_market_data_status"] = snap.get("entry_market_data_status")
    position["entry_provider_resolution_status"] = snap.get("entry_provider_resolution_status")
    position["entry_symbol_resolution_status"] = snap.get("entry_symbol_resolution_status")
    position["entry_trade_readiness_status"] = snap.get("entry_trade_readiness_status")
    position["provenance"] = snap.get("provenance")
    return position


def resolve_position_market_data_state(
    position: dict[str, Any],
    *,
    current_price: float | None,
    mark_fresh: bool,
    price_age_seconds: float | None = None,
    max_staleness_seconds: float | None = None,
) -> str:
    """Classify position market-data state without inventing prices."""
    max_age = float(
        max_staleness_seconds
        if max_staleness_seconds is not None
        else position.get("max_data_staleness_allowed_seconds")
        or DEFAULT_MAX_STALENESS_SECONDS
    )
    has_any = current_price is not None
    is_stale = bool(
        has_any
        and (
            not mark_fresh
            or (price_age_seconds is not None and price_age_seconds > max_age)
        )
    )
    # Missing symbol must NOT become missing price — symbol fields ignored here.
    return classify_position_market_data_state(
        has_fresh_current_price=bool(mark_fresh and has_any),
        has_any_price=has_any,
        is_stale=is_stale,
        is_partial=False,
        require_exit_only=False,
        require_manual_review=False,
    )


def build_position_financial_dto(
    position: dict[str, Any],
    *,
    position_market_data_state: str,
    current_price: float | None = None,
    position_value: float | None = None,
    unrealized_pnl: float | None = None,
    unrealized_pnl_pct: float | None = None,
) -> dict[str, Any]:
    """Strict financial DTO — null numerics when stale/unavailable."""
    state = _cell(position_market_data_state)
    last_good = position.get("last_good_price")
    last_good_ts = _cell(position.get("last_good_price_timestamp"))

    if state == DATA_STALE:
        return {
            "position_market_data_state": DATA_STALE,
            "financial_data_status": "STALE",
            "current_price_display": "N/A (STALE)",
            "position_value_display": "N/A (STALE)",
            "unrealized_pnl_display": "N/A (STALE)",
            "unrealized_pnl_pct_display": "N/A (STALE)",
            "current_price_numeric": None,
            "position_value_numeric": None,
            "unrealized_pnl_numeric": None,
            "unrealized_pnl_pct_numeric": None,
            "last_good_price_display": (
                f"{last_good} @ {last_good_ts} (STALE — not current tradable price)"
                if last_good is not None
                else ""
            ),
            "last_good_price": last_good,
            "last_good_price_timestamp": last_good_ts,
            "price_status_detail": (
                "last_good_price is not current tradable price; "
                "do not compute PnL from stale or last_good price"
            ),
            "frontend_must_not_compute_pnl": True,
            "frontend_must_not_treat_null_as_zero": True,
        }

    if state in {
        PRICE_UNAVAILABLE,
        POSITION_EXIT_ONLY_CONTEXT_REQUIRED,
        POSITION_MANUAL_REVIEW_REQUIRED,
        DATA_DEGRADED,
    } and state != DATA_OK:
        # PRICE_UNAVAILABLE and non-OK states that lack usable current price
        if state != DATA_DEGRADED or current_price is None:
            label = "UNAVAILABLE" if state != DATA_DEGRADED else "DEGRADED"
            if state == PRICE_UNAVAILABLE or current_price is None:
                label = "UNAVAILABLE"
            return {
                "position_market_data_state": state,
                "financial_data_status": label,
                "current_price_display": f"N/A ({label})",
                "position_value_display": f"N/A ({label})",
                "unrealized_pnl_display": f"N/A ({label})",
                "unrealized_pnl_pct_display": f"N/A ({label})",
                "current_price_numeric": None,
                "position_value_numeric": None,
                "unrealized_pnl_numeric": None,
                "unrealized_pnl_pct_numeric": None,
                "last_good_price_display": (
                    f"{last_good} @ {last_good_ts} (not current tradable price)"
                    if last_good is not None
                    else ""
                ),
                "last_good_price": last_good,
                "last_good_price_timestamp": last_good_ts,
                "price_status_detail": (
                    "no usable current price; last_good_price is not current tradable price"
                ),
                "frontend_must_not_compute_pnl": True,
                "frontend_must_not_treat_null_as_zero": True,
            }

    # DATA_OK — populate only from validated current market data
    def _fmt_price(v: float | None) -> str:
        if v is None:
            return "N/A"
        return f"{v:.8g}"

    def _fmt_usd(v: float | None) -> str:
        if v is None:
            return "N/A"
        return f"{v:.4f}"

    def _fmt_pct(v: float | None) -> str:
        if v is None:
            return "N/A"
        return f"{v * 100:.2f}%"

    return {
        "position_market_data_state": DATA_OK,
        "financial_data_status": "OK",
        "current_price_display": _fmt_price(current_price),
        "position_value_display": _fmt_usd(position_value),
        "unrealized_pnl_display": _fmt_usd(unrealized_pnl),
        "unrealized_pnl_pct_display": _fmt_pct(unrealized_pnl_pct),
        "current_price_numeric": current_price,
        "position_value_numeric": position_value,
        "unrealized_pnl_numeric": unrealized_pnl,
        "unrealized_pnl_pct_numeric": unrealized_pnl_pct,
        "last_good_price_display": (
            f"{last_good} @ {last_good_ts}" if last_good is not None else ""
        ),
        "last_good_price": last_good,
        "last_good_price_timestamp": last_good_ts,
        "price_status_detail": "current price from validated fresh market data",
        "frontend_must_not_compute_pnl": False,
        "frontend_must_not_treat_null_as_zero": True,
    }


def assert_new_entry_allowed(trade_readiness_status: str) -> tuple[bool, str]:
    """Block new paper/demo buys when trade readiness is blocked.

    The returned reason is always a market-data, identity, or
    position-continuity reason — never a missing-symbol reason.
    """
    status = _cell(trade_readiness_status)
    if not status or status == PAPER_ELIGIBLE:
        return True, ""
    if entry_blocked(status) or status == MARKET_DATA_MISSING:
        reason = block_reason_for(status) or status
        if not assert_block_reason_not_symbol_only(reason):
            raise AssertionError("block reason must never be SYMBOL_MISSING_ONLY")
        return False, f"new_entry_blocked:{status}:{reason}"
    return True, ""
