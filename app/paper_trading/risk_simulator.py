"""Minimal risk simulation for paper positions (SL/TP/time-stop)."""

from __future__ import annotations

from app.paper_trading.types import PaperOrderStatus


def determine_close_status(
    entry_price: float,
    current_price: float,
    *,
    take_profit_pct: float = 0.10,
    stop_loss_pct: float = 0.05,
) -> str | None:
    if entry_price <= 0:
        return None
    change_pct = (current_price - entry_price) / entry_price
    if change_pct >= take_profit_pct:
        return PaperOrderStatus.PAPER_CLOSED_TP.value
    if change_pct <= -stop_loss_pct:
        return PaperOrderStatus.PAPER_CLOSED_SL.value
    return None
