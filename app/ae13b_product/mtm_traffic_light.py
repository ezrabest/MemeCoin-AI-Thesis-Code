"""Mark-to-market traffic light for open positions (AE13I semantics).

RED: no current price / stale mark / missing market row / invalid price /
     MTM error / ambiguous address role / historical-only.
GREEN: take-profit reached, trailing stop armed/actionable, manual close
       available with a fresh price, or unrealized PnL is non-negative with
       a fresh price.
YELLOW: active/waiting — fresh price but no exit trigger yet (min hold not
        reached, no TP/SL hit, time-stop not reached).
"""
from __future__ import annotations

from typing import Any

MAX_FRESH_PRICE_AGE_SECONDS = 900.0


def _f(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _light(color: str, label: str, reason: str) -> dict[str, Any]:
    return {
        "traffic_light_status": color,
        "traffic_light_label": label,
        "traffic_light_reason": reason,
        # Backward-compatible aliases for any pre-AE13I callers.
        "color": color,
        "traffic_light": color,
        "label": label,
        "reason": reason,
    }


def compute_traffic_light(position_marked: dict[str, Any]) -> dict[str, Any]:
    pos = dict(position_marked or {})

    mtm_error = bool(pos.get("mtm_error") or pos.get("mark_to_market_error"))
    historical_only = bool(
        pos.get("historical_only")
        or pos.get("is_historical_only")
        or pos.get("data_mode") == "historical"
    )
    address_role_status = str(pos.get("address_role_status") or "")
    ambiguous_address = address_role_status == "ambiguous" or bool(pos.get("is_ambiguous"))

    market_row_missing = bool(
        pos.get("market_row_missing")
        or pos.get("matched_market_pair_status") == "unmatched_no_current_price"
    )

    current_price = _f(
        pos.get("current_price") or pos.get("marked_price") or pos.get("mark_price")
    )
    price_age = _f(pos.get("price_age_seconds") or pos.get("mark_age_seconds"))

    mark_stale = bool(pos.get("mark_stale") or pos.get("price_stale") or pos.get("is_stale"))
    if price_age is not None and price_age > MAX_FRESH_PRICE_AGE_SECONDS:
        mark_stale = True

    invalid_price = current_price is not None and current_price <= 0

    # --- RED: hard blockers on trustworthy MTM, checked first -------------
    if mtm_error:
        return _light("red", "At risk", "Mark-to-market error - treat position health as unknown.")
    if historical_only:
        return _light("red", "At risk", "Historical-only market data - no live mark available.")
    if ambiguous_address:
        return _light(
            "red", "At risk", "Address role is ambiguous - cannot confirm this is the traded pair."
        )
    if current_price is None or market_row_missing:
        return _light("red", "At risk", "No current mark price for this position.")
    if invalid_price:
        return _light("red", "At risk", "Invalid mark price (must be > 0).")
    if mark_stale:
        return _light("red", "At risk", "Mark price is stale - treat position health as uncertain.")

    price_fresh = True  # reached only when current_price is set and not stale

    take_profit = _f(pos.get("take_profit"))
    stop_loss = _f(pos.get("stop_loss"))
    tp_reached = take_profit is not None and current_price >= take_profit
    sl_hit = stop_loss is not None and current_price <= stop_loss
    trailing_status = str(pos.get("trailing_stop_status") or "")
    trailing_armed = trailing_status in ("active", "triggered")
    bot_would_exit = bool(pos.get("bot_would_exit_now"))
    manual_close_available = bool(pos.get("manual_close_allowed", True))

    pnl_pct = _f(
        pos.get("unrealized_pnl_pct") or pos.get("mtm_pnl_pct") or pos.get("pnl_pct")
    )
    pnl_usd = _f(pos.get("unrealized_pnl_usd"))
    pnl_known = pnl_pct is not None or pnl_usd is not None
    pnl_positive = (pnl_pct is not None and pnl_pct >= 0) or (pnl_usd is not None and pnl_usd >= 0)

    # --- GREEN: an exit is reachable/actionable now, or PnL is healthy ----
    if tp_reached:
        return _light("green", "Exit available", "Take-profit level reached.")
    if trailing_armed and price_fresh:
        return _light("green", "Exit available", "Trailing stop is armed and actionable.")
    if bot_would_exit:
        return _light("green", "Exit available", pos.get("bot_exit_reason") or "Bot exit condition met.")
    if manual_close_available and price_fresh and pnl_known and pnl_positive:
        return _light(
            "green",
            "Healthy",
            "Manual close available with a fresh price and non-negative unrealized PnL.",
        )
    if price_fresh and pnl_known and pnl_positive:
        return _light("green", "Healthy", "Fresh mark price and non-negative unrealized PnL.")

    # --- YELLOW: active/waiting, no exit trigger yet -----------------------
    if sl_hit:
        return _light("yellow", "Caution", "Stop-loss level reached - awaiting exit execution.")
    if price_fresh:
        return _light(
            "yellow",
            "Waiting",
            "Fresh price - waiting on min hold / time-stop; no exit trigger yet.",
        )
    return _light("yellow", "Caution", "Mixed MTM signals; review before action.")
