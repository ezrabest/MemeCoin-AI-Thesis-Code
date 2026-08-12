"""Paper position lifecycle management."""

from __future__ import annotations

from app.paper_trading.order_simulator import compute_execution_latency
from app.paper_trading.types import (
    PaperOrder,
    PaperOrderStatus,
    PaperPosition,
    PaperTradeRecord,
    utc_now_iso,
)


def open_position_from_order(order: PaperOrder) -> PaperPosition | None:
    if order.status != PaperOrderStatus.PAPER_FILLED.value:
        return None
    return PaperPosition(
        paper_order_id=order.paper_order_id,
        candidate_id=order.candidate_id,
        symbol=order.symbol,
        pair_address=order.pair_address,
        side="LONG",
        entry_price_usd=order.filled_price_usd or 0.0,
        quantity=order.quantity,
        notional_usd=order.notional_usd,
        status="OPEN",
        opened_at_utc=order.filled_at_utc or order.created_at_utc,
        source_decision_id=order.source_decision_id,
        source_context_record_id=order.source_context_record_id,
        source_llm_audit_record_id=order.source_llm_audit_record_id,
    )


def close_position_manual(
    position: PaperPosition,
    order: PaperOrder,
    exit_price_usd: float,
    close_status: str = PaperOrderStatus.PAPER_CLOSED_MANUAL.value,
) -> tuple[PaperTradeRecord, PaperOrder]:
    pnl = (exit_price_usd - position.entry_price_usd) * position.quantity
    closed_at = utc_now_iso()
    trade = PaperTradeRecord(
        paper_order_id=order.paper_order_id,
        position_id=position.position_id,
        candidate_id=position.candidate_id,
        symbol=position.symbol,
        side="BUY",
        entry_price_usd=position.entry_price_usd,
        exit_price_usd=exit_price_usd,
        quantity=position.quantity,
        notional_usd=position.notional_usd,
        realized_pnl_usd=pnl,
        close_reason=close_status,
        opened_at_utc=position.opened_at_utc,
        closed_at_utc=closed_at,
        decision_created_at_utc=order.decision_created_at_utc,
        order_created_at_utc=order.created_at_utc,
        filled_at_utc=order.filled_at_utc,
        source_decision_id=order.source_decision_id,
        source_context_record_id=order.source_context_record_id,
        source_llm_audit_record_id=order.source_llm_audit_record_id,
    )
    latency_ms, latency_status = compute_execution_latency(
        order.decision_created_at_utc, order.filled_at_utc
    )
    trade.execution_latency_ms = latency_ms
    trade.execution_latency_status = latency_status

    order.status = close_status
    order.closed_at_utc = closed_at
    return trade, order
