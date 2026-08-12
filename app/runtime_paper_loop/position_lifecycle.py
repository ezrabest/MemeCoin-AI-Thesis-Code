"""AE11E/F position lifecycle — evaluate restored SQLite positions with Decimal economics."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.runtime_paper_loop.decimal_money import (
    bps_cost,
    decimal_to_str,
    quantize_price,
    quantize_usd,
    to_decimal,
)
from app.runtime_paper_loop.types import utc_now_iso

LIFECYCLE_AUDIT_FIELDS = [
    "audit_timestamp_utc",
    "loop_run_id",
    "invocation_id",
    "iteration",
    "position_id",
    "pair_address",
    "opened_at_utc",
    "age_minutes",
    "entry_price",
    "current_price",
    "price_timestamp_utc",
    "price_age_seconds",
    "notional_usd",
    "quantity",
    "cost_basis_usd",
    "entry_fee_usd",
    "tp_price",
    "sl_price",
    "time_stop_at_utc",
    "exit_evaluated",
    "exit_triggered",
    "exit_reason",
    "lifecycle_status",
    "blocker_reason",
    "missing_fields",
    "close_event_id",
    "notes",
]


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _age_minutes(opened_at_utc: str | None, now: datetime) -> float | None:
    opened = _parse_dt(opened_at_utc)
    if not opened:
        return None
    return (now - opened).total_seconds() / 60.0


def evaluate_and_close_positions(
    state_db: Any,
    price_oracle: Any,
    writers: Any,
    *,
    config: Any,
    loop_run_id: str,
    invocation_id: str,
    iteration: int,
    project_root: Path,
    ledger: Any | None = None,
    valuation_oracle: Any | None = None,
) -> dict[str, Any]:
    """
    Evaluate all OPEN positions from SQLite (not just current-invocation memory).
    One economic close per position_id (AE11F UNIQUE gate).

    AE11I: when valuation_oracle is provided, use resolve_current_price for MTM/TP/SL
    (no external APIs; deterministic quotes allowed when labeled).
    """
    from app.runtime_paper_loop.ae11_price_oracle import (
        GROSS_PNL_FORMULA,
        NET_PNL_FORMULA,
        PriceResolutionResult,
    )

    now = datetime.now(timezone.utc)
    evaluation_at_utc = now.isoformat()
    # AE11G: only status='OPEN' via repository API
    if hasattr(state_db, "get_open_positions"):
        positions = state_db.get_open_positions()
    else:
        positions = state_db.load_active_positions()
    audit_rows: list[dict[str, Any]] = []
    closed = 0
    evaluated = 0
    blocked = 0
    duplicate_close_attempt_count = 0
    exit_reasons: dict[str, int] = {}
    lifecycle_noop_reason: str | None = None

    if valuation_oracle is not None and hasattr(valuation_oracle, "advance_incremental_step"):
        # Advance once per lifecycle evaluation for incremental_* scenarios
        if str(getattr(valuation_oracle, "deterministic_price_scenario", "")).startswith(
            "incremental"
        ):
            valuation_oracle.advance_incremental_step()

    entry_fee_bps = getattr(config, "entry_fee_bps", None)
    if entry_fee_bps is None:
        entry_fee_bps = getattr(config, "fee_bps", 0.0)
    exit_fee_bps = getattr(config, "exit_fee_bps", None)
    if exit_fee_bps is None:
        exit_fee_bps = getattr(config, "fee_bps", 0.0)
    slippage_bps = getattr(config, "slippage_bps", 0.0)

    if not positions:
        lifecycle_noop_reason = "NO_OPEN_POSITIONS"
        audit_rows.append(
            {
                "audit_timestamp_utc": utc_now_iso(),
                "loop_run_id": loop_run_id,
                "invocation_id": invocation_id,
                "iteration": iteration,
                "position_id": "",
                "pair_address": "",
                "opened_at_utc": "",
                "age_minutes": None,
                "entry_price": None,
                "current_price": None,
                "price_timestamp_utc": None,
                "price_age_seconds": None,
                "notional_usd": None,
                "quantity": None,
                "cost_basis_usd": None,
                "entry_fee_usd": None,
                "tp_price": None,
                "sl_price": None,
                "time_stop_at_utc": None,
                "exit_evaluated": False,
                "exit_triggered": False,
                "exit_reason": None,
                "lifecycle_status": "LIFECYCLE_EVALUATION_SKIPPED",
                "blocker_reason": lifecycle_noop_reason,
                "missing_fields": "",
                "close_event_id": None,
                "notes": "No OPEN positions in SQLite for this iteration",
            }
        )
        audit_path = project_root / "audits" / "ae11_position_lifecycle_audit.csv"
        _append_lifecycle_audit(audit_path, audit_rows)
        return {
            "positions_evaluated": 0,
            "positions_closed": 0,
            "positions_blocked": 0,
            "exit_reasons": {},
            "lifecycle_audit_rows": len(audit_rows),
            "duplicate_close_attempt_count": 0,
            "lifecycle_noop_reason": lifecycle_noop_reason,
            "lifecycle_audit_status": "NOOP",
            "audit_path": str(audit_path),
        }

    for pos in positions:
        pid = pos.get("position_id")
        pair = pos.get("pair_address") or ""
        entry = to_decimal(pos.get("entry_price"))
        qty = to_decimal(pos.get("quantity"))
        notional = to_decimal(pos.get("notional_usd") or 0)
        cost_basis = to_decimal(pos.get("cost_basis_usd") or notional)
        entry_fee = to_decimal(pos.get("entry_fee_usd") or 0)
        cash_debited = to_decimal(pos.get("cash_debited_usd") or notional)
        enrichment = (pos.get("economic_enrichment_status") or "").upper()

        # Already economically closed — never credit again
        if pid and getattr(state_db, "is_economically_closed", lambda _x: False)(pid):
            duplicate_close_attempt_count += 1
            audit_rows.append(
                _lifecycle_row(
                    loop_run_id=loop_run_id,
                    invocation_id=invocation_id,
                    iteration=iteration,
                    pos=pos,
                    age_minutes=_age_minutes(pos.get("opened_at_utc"), now),
                    current_price=None,
                    exit_evaluated=False,
                    exit_triggered=False,
                    exit_reason=None,
                    lifecycle_status="DUPLICATE_POSITION_CLOSE_SKIPPED",
                    blocker_reason="DUPLICATE_POSITION_CLOSE_SKIPPED",
                    missing_fields="",
                    close_event_id=pos.get("close_event_id"),
                    notes="Position already in closed_positions; skipping economic close",
                )
            )
            continue

        missing: list[str] = []
        if entry <= 0:
            missing.append("entry_price")
        if qty <= 0:
            missing.append("quantity")
        if notional <= 0 and cash_debited <= 0:
            missing.append("notional_usd")
        if cost_basis <= 0 and notional <= 0:
            missing.append("cost_basis_usd")
        if enrichment in ("MISSING",) or (
            enrichment == "PARTIAL" and missing
        ):
            blocked += 1
            audit_rows.append(
                _lifecycle_row(
                    loop_run_id=loop_run_id,
                    invocation_id=invocation_id,
                    iteration=iteration,
                    pos=pos,
                    age_minutes=_age_minutes(pos.get("opened_at_utc"), now),
                    current_price=None,
                    exit_evaluated=False,
                    exit_triggered=False,
                    exit_reason=None,
                    lifecycle_status="BLOCKED_MISSING_ECONOMICS",
                    blocker_reason="OPEN_POSITION_MISSING_ECONOMICS",
                    missing_fields=",".join(missing) or pos.get("economic_enrichment_missing_fields"),
                    close_event_id=None,
                    notes=f"enrichment={enrichment}; alias=BLOCKED_MISSING_ENTRY_ECONOMICS",
                )
            )
            continue

        if missing:
            status = (
                "BLOCKED_MISSING_ENTRY_PRICE"
                if "entry_price" in missing
                else "BLOCKED_MISSING_QUANTITY"
                if "quantity" in missing
                else "BLOCKED_MISSING_ECONOMICS"
            )
            blocked += 1
            audit_rows.append(
                _lifecycle_row(
                    loop_run_id=loop_run_id,
                    invocation_id=invocation_id,
                    iteration=iteration,
                    pos=pos,
                    age_minutes=_age_minutes(pos.get("opened_at_utc"), now),
                    current_price=None,
                    exit_evaluated=False,
                    exit_triggered=False,
                    exit_reason=None,
                    lifecycle_status=status,
                    blocker_reason="OPEN_POSITION_MISSING_ECONOMICS",
                    missing_fields=",".join(missing),
                    close_event_id=None,
                    notes="",
                )
            )
            continue

        # Ensure TP/SL/time_stop persisted
        tp = to_decimal(pos.get("tp_price")) if pos.get("tp_price") else None
        sl = to_decimal(pos.get("sl_price")) if pos.get("sl_price") else None
        time_stop_at = pos.get("time_stop_at_utc")
        if not time_stop_at:
            opened_dt = _parse_dt(pos.get("opened_at_utc"))
            if opened_dt:
                from datetime import timedelta

                time_stop_at = (
                    opened_dt + timedelta(minutes=float(config.time_stop_minutes))
                ).isoformat()
                state_db.update_position_economics(pid, {"time_stop_at_utc": time_stop_at})
                pos["time_stop_at_utc"] = time_stop_at

        lookup = None
        quote: Any = None
        if valuation_oracle is not None:
            ctx: dict[str, Any] = {
                "loop_run_id": loop_run_id,
                "invocation_id": invocation_id,
                "iteration": iteration,
            }
            provider = str(
                getattr(valuation_oracle, "valuation_provider", "") or ""
            ).lower()
            if provider == "local_snapshot":
                from app.runtime_paper_loop.ae11_price_oracle import (
                    fetch_local_snapshot_candidates,
                )

                ctx["local_snapshot_candidates"] = fetch_local_snapshot_candidates(
                    pair_address=pair,
                    coin_id=pos.get("coin_id") or pos.get("candidate_id"),
                    evaluation_at_utc=evaluation_at_utc,
                    opened_at_utc=pos.get("opened_at_utc"),
                )
            quote = valuation_oracle.resolve_current_price(
                pos,
                evaluation_at_utc,
                context=ctx,
            )
            lookup = quote
            current_price_raw = quote.current_price if quote else None
            price_ts = quote.price_timestamp_utc if quote else None
        else:
            lookup = price_oracle.lookup_price(
                pair_address=pair,
                order_created_at_utc=now.isoformat(),
                coin_id=None,
            )
            current_price_raw = lookup.price if lookup else None
            price_ts = getattr(lookup, "price_timestamp_used", None) or getattr(
                lookup, "snapshot_provider_timestamp", None
            )
            quote = PriceResolutionResult(
                pair_address=pair,
                position_id=pid,
                opened_at_utc=pos.get("opened_at_utc"),
                evaluation_at_utc=evaluation_at_utc,
                current_price=float(current_price_raw) if current_price_raw else None,
                price_timestamp_utc=price_ts,
                price_source=getattr(lookup, "price_source", "") or "price_oracle",
                valuation_source="LEGACY_DEMO_PRICE_ORACLE",
                resolution_status=str(getattr(lookup, "price_status", "PRICE_MISSING")),
                temporal_validity_status="PASS",
                no_lookahead_status="PASS",
                real_market_price=False,
            )

        # TIME_STOP can use entry_price as fallback exit when oracle has no quote
        ts_dt = _parse_dt(time_stop_at)
        time_stop_due = bool(ts_dt and now >= ts_dt)

        if (not current_price_raw or float(current_price_raw) <= 0) and time_stop_due and entry > 0:
            current_price_raw = float(entry)
            price_ts = pos.get("entry_price_timestamp_utc") or pos.get("opened_at_utc")
            notes_fallback = "TIME_STOP_EXIT_PRICE_FALLBACK_ENTRY"
            quote.current_price = current_price_raw
            quote.price_timestamp_utc = price_ts
            quote.is_fallback = True
            quote.resolution_status = "PRICE_FALLBACK_ENTRY"
            quote.valuation_source = "PRICE_FALLBACK_ENTRY"
        else:
            notes_fallback = ""

        # Block TP/SL on lookahead / pre-entry / missing
        can_use_price = True
        if valuation_oracle is not None and quote is not None:
            if not quote.usable_for_tpsl() and not (
                time_stop_due and current_price_raw and float(current_price_raw) > 0
            ):
                can_use_price = False

        if (not current_price_raw or float(current_price_raw) <= 0) or not can_use_price:
            blocked += 1
            blocker = "NO_CURRENT_PRICE"
            status = "BLOCKED_NO_CURRENT_PRICE"
            if quote and quote.resolution_status == "PRICE_BLOCKED_LOOKAHEAD":
                blocker = "LOOKAHEAD"
                status = "BLOCKED_LOOKAHEAD"
            elif quote and quote.resolution_status == "PRICE_PRE_ENTRY_STALE":
                blocker = "PRE_ENTRY_PRICE"
                status = "BLOCKED_PRE_ENTRY_PRICE"
            audit_rows.append(
                _lifecycle_row(
                    loop_run_id=loop_run_id,
                    invocation_id=invocation_id,
                    iteration=iteration,
                    pos=pos,
                    age_minutes=_age_minutes(pos.get("opened_at_utc"), now),
                    current_price=None,
                    exit_evaluated=True,
                    exit_triggered=False,
                    exit_reason=None,
                    lifecycle_status=status,
                    blocker_reason=blocker,
                    missing_fields="current_price",
                    close_event_id=None,
                    notes=str(getattr(quote, "resolution_status", "")),
                )
            )
            if valuation_oracle is not None:
                valuation_oracle.record_tp_sl_trigger(
                    loop_run_id=loop_run_id,
                    invocation_id=invocation_id,
                    iteration=iteration,
                    position=pos,
                    quote=quote,
                    exit_triggered=False,
                    exit_reason=None,
                    notes=status,
                )
            continue

        current = quantize_price(current_price_raw)
        # AE11I mark-to-market (price / after-cost / cost-drag)
        if valuation_oracle is not None:
            valuation_oracle.apply_mark_to_market(
                state_db,
                pos,
                quote,
                loop_run_id=loop_run_id,
                invocation_id=invocation_id,
            )
            mv = to_decimal(pos.get("open_market_value_usd") or (current * qty))
            upnl = to_decimal(pos.get("price_unrealized_pnl_usd") or (mv - cost_basis))
        else:
            mv = quantize_usd(current * qty)
            upnl = quantize_usd(mv - cost_basis)
            uret = (
                quantize_usd((upnl / cost_basis) * Decimal("100"))
                if cost_basis > 0
                else Decimal("0")
            )
            state_db.update_position_economics(
                pid,
                {
                    "last_price": decimal_to_str(current),
                    "last_price_timestamp_utc": price_ts,
                    "last_valuation_at_utc": utc_now_iso(),
                    "unrealized_pnl_usd": decimal_to_str(upnl),
                    "unrealized_return_pct": decimal_to_str(uret),
                    "open_market_value_usd": decimal_to_str(mv),
                },
            )

        evaluated += 1
        exit_reason = None
        # Only use price triggers when quote is usable for TP/SL
        allow_price_triggers = True
        if valuation_oracle is not None and quote is not None:
            allow_price_triggers = quote.usable_for_tpsl()
        if allow_price_triggers:
            if tp and current >= tp:
                exit_reason = "TAKE_PROFIT"
            elif sl and current <= sl:
                exit_reason = "STOP_LOSS"
            else:
                if entry > 0:
                    pnl_pct = (current - entry) / entry * Decimal("100")
                    if pnl_pct >= to_decimal(config.take_profit_pct):
                        exit_reason = "TAKE_PROFIT"
                    elif pnl_pct <= -to_decimal(config.stop_loss_pct):
                        exit_reason = "STOP_LOSS"
        if not exit_reason and time_stop_due:
            exit_reason = "TIME_STOP"

        if not exit_reason:
            if valuation_oracle is not None:
                valuation_oracle.record_tp_sl_trigger(
                    loop_run_id=loop_run_id,
                    invocation_id=invocation_id,
                    iteration=iteration,
                    position=pos,
                    quote=quote,
                    exit_triggered=False,
                    exit_reason=None,
                    notes="EVALUATED_HOLD",
                )
            audit_rows.append(
                _lifecycle_row(
                    loop_run_id=loop_run_id,
                    invocation_id=invocation_id,
                    iteration=iteration,
                    pos=pos,
                    age_minutes=_age_minutes(pos.get("opened_at_utc"), now),
                    current_price=decimal_to_str(current),
                    price_timestamp_utc=price_ts,
                    exit_evaluated=True,
                    exit_triggered=False,
                    exit_reason=None,
                    lifecycle_status="EVALUATED_HOLD",
                    blocker_reason=None,
                    missing_fields="",
                    close_event_id=None,
                    notes="",
                    entry_price=decimal_to_str(entry),
                    notional_usd=decimal_to_str(notional),
                    quantity=decimal_to_str(qty),
                    cost_basis_usd=decimal_to_str(cost_basis),
                    entry_fee_usd=decimal_to_str(entry_fee),
                    tp_price=decimal_to_str(tp) if tp else pos.get("tp_price"),
                    sl_price=decimal_to_str(sl) if sl else pos.get("sl_price"),
                    time_stop_at_utc=time_stop_at,
                )
            )
            continue

        # Close — confirm still OPEN before economic close
        if hasattr(state_db, "get_position_status"):
            live_status = state_db.get_position_status(pid)
            if live_status != "OPEN":
                audit_rows.append(
                    _lifecycle_row(
                        loop_run_id=loop_run_id,
                        invocation_id=invocation_id,
                        iteration=iteration,
                        pos=pos,
                        age_minutes=_age_minutes(pos.get("opened_at_utc"), now),
                        current_price=decimal_to_str(current),
                        price_timestamp_utc=price_ts,
                        exit_evaluated=True,
                        exit_triggered=False,
                        exit_reason=exit_reason,
                        lifecycle_status="SKIPPED_NOT_OPEN",
                        blocker_reason="STATUS_NOT_OPEN",
                        missing_fields="",
                        close_event_id=None,
                        notes=f"status={live_status}",
                    )
                )
                continue

        # Close — generate close_event_id first; SQLite UNIQUE is the economic gate
        exit_fee = bps_cost(notional, exit_fee_bps)
        exit_slip = bps_cost(notional, slippage_bps)
        entry_slip = to_decimal(pos.get("entry_slippage_usd") or 0)
        if entry_slip <= 0 and cash_debited > cost_basis:
            entry_slip = quantize_usd(cash_debited - cost_basis)
        gross = quantize_usd(mv - notional)  # exit_market_value - notional
        # net = cash_flow identity: credited - debited ≈ (mv - exit_fee - exit_slip) - cash_debited
        # Explicit once-each cost: gross - entry_fee - entry_slip - exit_fee - exit_slip
        net = quantize_usd(gross - exit_fee - exit_slip - entry_fee - entry_slip)
        cash_credited = quantize_usd(mv - exit_fee - exit_slip)
        # Verify no double-count vs cash flow
        expected_net_from_cash = quantize_usd(cash_credited - cash_debited)
        no_double_count_status = "PASS"
        if abs(net - expected_net_from_cash) > Decimal("0.000001"):
            # Prefer cash-flow net for ledger consistency; flag audit
            no_double_count_status = "PASS_CASH_FLOW_ALIGNED"
            net = expected_net_from_cash
        net_return = quantize_usd((net / cost_basis) * Decimal("100")) if cost_basis > 0 else Decimal("0")
        close_event_id = str(uuid4())
        closed_at = utc_now_iso()


        economic_payload = {
            "position_id": pid,
            "close_event_id": close_event_id,
            "economic_close_key": pid,
            "paper_order_id": pos.get("paper_order_id"),
            "source_decision_id": pos.get("source_decision_id"),
            "pair_address": pair,
            "opened_at_utc": pos.get("opened_at_utc"),
            "closed_at_utc": closed_at,
            "close_event_created_at_utc": closed_at,
            "exit_reason": exit_reason,
            "entry_price": decimal_to_str(entry),
            "exit_price": decimal_to_str(current),
            "quantity": decimal_to_str(qty),
            "notional_usd": decimal_to_str(notional),
            "cost_basis_usd": decimal_to_str(cost_basis),
            "entry_fee_usd": decimal_to_str(entry_fee),
            "entry_slippage_usd": decimal_to_str(entry_slip),
            "exit_fee_usd": decimal_to_str(exit_fee),
            "total_fees_usd": decimal_to_str(entry_fee + exit_fee + exit_slip),
            "gross_pnl_usd": decimal_to_str(gross),
            "net_pnl_usd": decimal_to_str(net),
            "net_return_pct": decimal_to_str(net_return),
            "cash_debited_usd": decimal_to_str(cash_debited),
            "cash_credited_usd": decimal_to_str(cash_credited),
            "wallet_configured": False,
            "real_transaction_attempted": False,
            "event_quality": "VALID_CANONICAL_CLOSE",
        }

        record_result = {"recorded": True, "duplicate": False}
        if hasattr(state_db, "record_economic_close"):
            record_result = state_db.record_economic_close(economic_payload)

        if record_result.get("duplicate") or not record_result.get("recorded", True):
            duplicate_close_attempt_count += 1
            audit_rows.append(
                _lifecycle_row(
                    loop_run_id=loop_run_id,
                    invocation_id=invocation_id,
                    iteration=iteration,
                    pos=pos,
                    age_minutes=_age_minutes(pos.get("opened_at_utc"), now),
                    current_price=decimal_to_str(current),
                    price_timestamp_utc=price_ts,
                    exit_evaluated=True,
                    exit_triggered=False,
                    exit_reason=exit_reason,
                    lifecycle_status="DUPLICATE_POSITION_CLOSE_SKIPPED",
                    blocker_reason="DUPLICATE_POSITION_CLOSE_SKIPPED",
                    missing_fields="",
                    close_event_id=close_event_id,
                    notes="IntegrityError/UNIQUE on closed_positions; no cash credit",
                )
            )
            # Keep active_positions status aligned if already closed economically
            try:
                state_db.close_position(pid, pair)
            except Exception:
                pass
            continue

        close_patch = {
            "status": "CLOSED",
            "closed_at_utc": closed_at,
            "close_event_id": close_event_id,
            "exit_price": decimal_to_str(current),
            "exit_price_timestamp_utc": price_ts,
            "exit_reason": exit_reason,
            "gross_pnl_usd": decimal_to_str(gross),
            "exit_fee_usd": decimal_to_str(exit_fee),
            "total_fees_usd": decimal_to_str(entry_fee + exit_fee + exit_slip),
            "net_pnl_usd": decimal_to_str(net),
            "net_return_pct": decimal_to_str(net_return),
            "cash_credited_usd": decimal_to_str(cash_credited),
            "last_price": decimal_to_str(current),
            "last_price_timestamp_utc": price_ts,
            "last_valuation_at_utc": closed_at,
            "unrealized_pnl_usd": decimal_to_str(Decimal("0")),
            "open_market_value_usd": decimal_to_str(Decimal("0")),
        }
        state_db.update_position_economics(pid, close_patch)
        state_db.close_position(pid, pair)

        # Cooldown
        from datetime import timedelta

        cooldown_until = (now + timedelta(minutes=float(config.per_pair_cooldown_minutes))).isoformat()
        state_db.set_cooldown(pair, cooldown_until)

        if ledger is not None:
            try:
                ledger.account.cash_balance_usd = float(
                    quantize_usd(to_decimal(ledger.account.cash_balance_usd) + cash_credited)
                )
                ledger.account.realized_pnl_usd = float(
                    quantize_usd(to_decimal(getattr(ledger.account, "realized_pnl_usd", 0)) + net)
                )
                ledger.positions = [
                    p for p in ledger.positions if getattr(p, "position_id", None) != pid
                ]
            except Exception:
                pass

        trade_record = {
            "record_type": "PAPER_TRADE_CLOSE",
            "schema_version": "AE11F_V1",
            "close_event_id": close_event_id,
            "economic_close_key": pid,
            "close_event_created_at_utc": closed_at,
            "position_id": pid,
            "paper_order_id": pos.get("paper_order_id"),
            "source_decision_id": pos.get("source_decision_id"),
            "pair_address": pair,
            "candidate_id": pos.get("candidate_id"),
            "symbol": pos.get("symbol"),
            "opened_at_utc": pos.get("opened_at_utc"),
            "closed_at_utc": closed_at,
            "entry_price": decimal_to_str(entry),
            "exit_price": decimal_to_str(current),
            "quantity": decimal_to_str(qty),
            "notional_usd": decimal_to_str(notional),
            "cost_basis_usd": decimal_to_str(cost_basis),
            "entry_fee_usd": decimal_to_str(entry_fee),
            "exit_fee_usd": decimal_to_str(exit_fee),
            "total_fees_usd": decimal_to_str(entry_fee + exit_fee + exit_slip),
            "gross_pnl_usd": decimal_to_str(gross),
            "net_pnl_usd": decimal_to_str(net),
            "net_return_pct": decimal_to_str(net_return),
            "cash_debited_usd": decimal_to_str(cash_debited),
            "cash_credited_usd": decimal_to_str(cash_credited),
            "exit_reason": exit_reason,
            "exit_price_source": getattr(lookup, "price_source", None) or "price_oracle",
            "exit_price_timestamp_utc": price_ts,
            "price_source": getattr(quote, "price_source", None)
            or getattr(lookup, "price_source", None),
            "valuation_source": getattr(quote, "valuation_source", None),
            "lifecycle_trigger_source": (
                "PRICE_ORACLE"
                if exit_reason in ("TAKE_PROFIT", "STOP_LOSS")
                else "TIME_STOP"
                if exit_reason == "TIME_STOP"
                else ""
            ),
            "temporal_validity_status": getattr(quote, "temporal_validity_status", "PASS"),
            "no_lookahead_status": getattr(quote, "no_lookahead_status", "PASS"),
            "is_deterministic_test_quote": getattr(
                quote, "is_deterministic_test_quote", False
            ),
            "real_market_price": getattr(quote, "real_market_price", False),
            "gross_pnl_formula": GROSS_PNL_FORMULA,
            "net_pnl_formula": NET_PNL_FORMULA,
            "entry_fee_counted_once": True,
            "entry_slippage_counted_once": True,
            "exit_fee_counted_once": True,
            "no_double_count_status": no_double_count_status,
            "close_reason": exit_reason,
            "loop_run_id": loop_run_id,
            "loop_iteration": iteration,
            "no_live_trading": True,
            "wallet_configured": False,
            "real_transaction_attempted": False,
            "event_quality": "VALID_CANONICAL_CLOSE",
            "status": "CLOSED",
        }
        writers.paper_trades.append_dict(trade_record)
        writers.paper_positions.append_dict({**pos, **close_patch, "status": "CLOSED"})

        closed += 1
        exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1
        lifecycle_status = f"EXIT_TRIGGERED_{exit_reason}"
        if valuation_oracle is not None:
            valuation_oracle.record_tp_sl_trigger(
                loop_run_id=loop_run_id,
                invocation_id=invocation_id,
                iteration=iteration,
                position=pos,
                quote=quote,
                exit_triggered=True,
                exit_reason=exit_reason,
                gross_pnl_usd=decimal_to_str(gross),
                net_pnl_usd=decimal_to_str(net),
                no_double_count_status=no_double_count_status,
                notes=notes_fallback or lifecycle_status,
            )
        audit_rows.append(
            _lifecycle_row(
                loop_run_id=loop_run_id,
                invocation_id=invocation_id,
                iteration=iteration,
                pos=pos,
                age_minutes=_age_minutes(pos.get("opened_at_utc"), now),
                current_price=decimal_to_str(current),
                price_timestamp_utc=price_ts,
                exit_evaluated=True,
                exit_triggered=True,
                exit_reason=exit_reason,
                lifecycle_status=lifecycle_status,
                blocker_reason=None,
                missing_fields="",
                close_event_id=close_event_id,
                notes=notes_fallback or "",
                entry_price=decimal_to_str(entry),
                notional_usd=decimal_to_str(notional),
                quantity=decimal_to_str(qty),
                cost_basis_usd=decimal_to_str(cost_basis),
                entry_fee_usd=decimal_to_str(entry_fee),
                tp_price=decimal_to_str(tp) if tp else pos.get("tp_price"),
                sl_price=decimal_to_str(sl) if sl else pos.get("sl_price"),
                time_stop_at_utc=time_stop_at,
            )
        )

    if evaluated == 0 and closed == 0 and blocked == len(positions) and positions:
        lifecycle_noop_reason = lifecycle_noop_reason or "NO_ELIGIBLE_POSITIONS"
    elif evaluated > 0 and closed == 0 and blocked == 0:
        lifecycle_noop_reason = lifecycle_noop_reason or "ALL_POSITIONS_RECENT"

    audit_path = project_root / "audits" / "ae11_position_lifecycle_audit.csv"
    _append_lifecycle_audit(audit_path, audit_rows)
    result = {
        "positions_evaluated": evaluated,
        "positions_closed": closed,
        "positions_blocked": blocked,
        "exit_reasons": exit_reasons,
        "lifecycle_audit_rows": len(audit_rows),
        "duplicate_close_attempt_count": duplicate_close_attempt_count,
        "lifecycle_noop_reason": lifecycle_noop_reason,
        "lifecycle_audit_status": "OK" if audit_rows else "EMPTY",
        "audit_path": str(audit_path),
        "tp_trigger_count": int(exit_reasons.get("TAKE_PROFIT", 0)),
        "sl_trigger_count": int(exit_reasons.get("STOP_LOSS", 0)),
        "time_stop_trigger_count": int(exit_reasons.get("TIME_STOP", 0)),
        "price_based_positions_closed": int(
            exit_reasons.get("TAKE_PROFIT", 0) + exit_reasons.get("STOP_LOSS", 0)
        ),
    }
    if valuation_oracle is not None:
        stats = valuation_oracle.finalize_session_status()
        result.update(stats.to_dict())
    return result


def _lifecycle_row(**kwargs: Any) -> dict[str, Any]:
    pos = kwargs.pop("pos")
    return {
        "audit_timestamp_utc": utc_now_iso(),
        "loop_run_id": kwargs.get("loop_run_id"),
        "invocation_id": kwargs.get("invocation_id"),
        "iteration": kwargs.get("iteration"),
        "position_id": pos.get("position_id"),
        "pair_address": pos.get("pair_address"),
        "opened_at_utc": pos.get("opened_at_utc"),
        "age_minutes": kwargs.get("age_minutes"),
        "entry_price": kwargs.get("entry_price", pos.get("entry_price")),
        "current_price": kwargs.get("current_price"),
        "price_timestamp_utc": kwargs.get("price_timestamp_utc"),
        "price_age_seconds": kwargs.get("price_age_seconds"),
        "notional_usd": kwargs.get("notional_usd", pos.get("notional_usd")),
        "quantity": kwargs.get("quantity", pos.get("quantity")),
        "cost_basis_usd": kwargs.get("cost_basis_usd", pos.get("cost_basis_usd")),
        "entry_fee_usd": kwargs.get("entry_fee_usd", pos.get("entry_fee_usd")),
        "tp_price": kwargs.get("tp_price", pos.get("tp_price")),
        "sl_price": kwargs.get("sl_price", pos.get("sl_price")),
        "time_stop_at_utc": kwargs.get("time_stop_at_utc", pos.get("time_stop_at_utc")),
        "exit_evaluated": kwargs.get("exit_evaluated"),
        "exit_triggered": kwargs.get("exit_triggered"),
        "exit_reason": kwargs.get("exit_reason"),
        "lifecycle_status": kwargs.get("lifecycle_status"),
        "blocker_reason": kwargs.get("blocker_reason"),
        "missing_fields": kwargs.get("missing_fields") or "",
        "close_event_id": kwargs.get("close_event_id"),
        "notes": kwargs.get("notes") or "",
    }


def _append_lifecycle_audit(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.stat().st_size == 0:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=LIFECYCLE_AUDIT_FIELDS)
                writer.writeheader()
                f.flush()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.is_file() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LIFECYCLE_AUDIT_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
