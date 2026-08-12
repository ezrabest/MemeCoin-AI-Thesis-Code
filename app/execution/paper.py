"""
Paper trading with strict Solana transaction costs, $10K demo wallet, and net ROI logging.
"""
from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .fill_price import (
    MAX_FEE_TO_NOTIONAL_PCT,
    PAPER_PRICE_SANITY_MAX_DEVIATION_PCT,
    build_market_price_maps,
    parse_market_price_usd,
    resolve_buy_fill_price,
    resolve_sell_fill_price,
)
from .paper_audit import (
    audit_trade_rows,
    notional_exceeds_equity_guard,
    portfolio_roi_from_equity,
)

from app.clean_forward.position_continuity import (
    assert_new_entry_allowed,
    attach_entry_snapshot_to_position,
    build_position_financial_dto,
    resolve_position_market_data_state,
)
from app.clean_forward.provider_resilience_statuses import entry_blocked
from app.llm_config import (
    LLM_PROVIDER_AUDIT_ONLY_REASON,
    extract_explicit_llm_origin_provider,
)

log = logging.getLogger("paper")

#: AE13I: a mark price older than this (seconds) is never treated as fresh for
#: PnL display purposes, matching MarketDataGateKeeper's default freshness window.
MAX_FRESH_MARK_AGE_SECONDS = 900.0


def _llm_audit_only_block_result(provider: str, action: str) -> dict[str, Any]:
    """Benign structured rejection for explicit LLM-origin paper attempts."""
    log.info(
        "LLM audit-only decision stored; execution not attempted provider=%s action=%s",
        provider,
        action,
    )
    return {
        "ok": False,
        "opened": False,
        "closed": False,
        "execution_attempted": False,
        "reason": LLM_PROVIDER_AUDIT_ONLY_REASON,
        "provider": provider,
        "action": action,
        "authority_status": "AUDIT_ONLY_NO_TRADE_AUTHORITY",
    }


#: current_price_source values that count as a real provider-backed mark for PnL.
_FRESH_PRICE_SOURCES = {
    "market_canonical_url",
    "market_provider_pair_url",
    "market_pair_address",
    "market_coin_id",
}

DATA_DIR = Path(__file__).parent.parent.parent / "data"
STATE_PATH = DATA_DIR / "paper_state.json"
TRADES_LOG_PATH = DATA_DIR / "paper_trades_log.csv"

STARTING_CAPITAL_USD: float = 10_000.0
SLIPPAGE_DEX_RATE = 0.015
SCALPING_EQUITY_PCT = 0.10
WHALE_RIDER_EQUITY_PCT = 0.30
PRIORITY_FEE_RATE = float(os.getenv("SOLANA_PRIORITY_FEE_RATE", "0.0003"))
PRIORITY_FEE_MIN = float(os.getenv("SOLANA_PRIORITY_FEE_MIN", "0.001"))
PRIORITY_FEE_MAX = float(os.getenv("SOLANA_PRIORITY_FEE_MAX", "2.0"))

TRADE_CSV_FIELDS = [
    "timestamp",
    "position_id",
    "symbol",
    "chain",
    "side",
    "quantity",
    "fill_price",
    "notional_usd",
    "swap_fee",
    "priority_fee",
    "total_fees",
    "gross_pnl",
    "realized_pnl",
    "net_roi_pct",
    "cluster_label",
    "reason_code",
    "coin_id",
    "pair_address",
    "decision_ref_id",
    "fill_price_source",
    "market_price_usd",
    "price_timestamp",
    "cash_before",
    "equity_before",
    "notional_requested",
    "notional_executed",
    "rejection_reason",
    "rejection_reasons",
    "blocking_guards",
    "rejection_code",
    "strategy_lane",
    "preset_id",
    "risk_mode",
    "event_type",
    "pair",
    "closed_by",
    "close_reason",
    "close_note",
    "paper_demo_only",
    "not_live_approved",
    "not_profitability_evidence",
    # AE13I close-freshness / manual-close disclosure fields
    "manual_close",
    "close_price_age_seconds",
    "close_freshness_status",
    "close_used_fallback_price",
    "manual_close_warning_shown",
    "close_price_source",
    # AE14 canonical instrument identity (mode-agnostic; paper now / live later)
    "instrument_id",
    "execution_instrument_id",
    "instrument_source",
    "candidate_source",
    "provider_pair_id",
    "base_token_address",
    "quote_token_address",
    "liquidity_at_entry",
    "price_updated_at",
    "liquidity_updated_at",
    "execution_mode",
    "live_trading_ready",
    "live_execution_enabled",
    "wallet_required",
    "wallet_connected",
    "clean_forward_bridge_used",
    "legacy_market_snapshots_used",
]


def _resolve_coin_id(coin: dict[str, Any]) -> int | None:
    raw = coin.get("coin_id") or coin.get("id")
    return int(raw) if raw is not None else None


def _format_age_label(seconds: float) -> str:
    """Human-readable age label, ASCII-safe."""
    secs = max(0.0, float(seconds))
    if secs < 60:
        return f"{int(secs)}s"
    minutes = secs / 60.0
    if minutes < 60:
        return f"{int(minutes)}m"
    hours = minutes / 60.0
    if hours < 24:
        h = int(hours)
        m = int(round(minutes - h * 60))
        return f"{h}h {m}m" if m else f"{h}h"
    days = hours / 24.0
    d = int(days)
    h = int(round(hours - d * 24))
    return f"{d}d {h}h" if h else f"{d}d"


def _parse_iso_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def _ensure_trade_csv_header() -> list[str]:
    """Migrate existing trade CSV to include all known columns without losing rows.

    Also guards against column-shift corruption: if every canonical column is
    already present but the on-disk header order differs from
    TRADE_CSV_FIELDS, appends written with the canonical order would silently
    land under the wrong header (e.g. symbol="", side=<pair value>,
    fill_price=<chain value>). When that happens, rewrite the file in
    canonical order first.
    """
    if not TRADES_LOG_PATH.exists() or TRADES_LOG_PATH.stat().st_size == 0:
        return list(TRADE_CSV_FIELDS)

    with open(TRADES_LOG_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing = list(reader.fieldnames or [])
        rows = list(reader)

    missing = [field for field in TRADE_CSV_FIELDS if field not in existing]
    if not missing:
        if existing == TRADE_CSV_FIELDS:
            return list(TRADE_CSV_FIELDS)
        with open(TRADES_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRADE_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in TRADE_CSV_FIELDS})
        return list(TRADE_CSV_FIELDS)

    merged_fields = existing + missing
    with open(TRADES_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=merged_fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return merged_fields


@dataclass(frozen=True)
class TransactionCosts:
    swap_fee: float
    priority_fee: float

    @property
    def total(self) -> float:
        return self.swap_fee + self.priority_fee


def compute_transaction_costs(notional_usd: float, chain: str) -> TransactionCosts:
    swap_fee = notional_usd * SLIPPAGE_DEX_RATE
    priority_fee = 0.0
    if chain.lower() == "solana":
        priority_fee = max(
            PRIORITY_FEE_MIN,
            min(PRIORITY_FEE_MAX, notional_usd * PRIORITY_FEE_RATE),
        )
    return TransactionCosts(swap_fee=round(swap_fee, 6), priority_fee=round(priority_fee, 6))


def _default_state() -> dict[str, Any]:
    return {
        "starting_capital": STARTING_CAPITAL_USD,
        "cash_usd": STARTING_CAPITAL_USD,
        "next_position_id": 1,
        "open_positions": [],
        "closed_trades": 0,
        "total_net_pnl": 0.0,
        "cumulative_swap_fees": 0.0,
        "cumulative_priority_fees": 0.0,
        "cumulative_total_fees": 0.0,
        "trading_mode": "DEMO",
    }


def _normalize_state(state: dict[str, Any]) -> dict[str, Any]:
    base = _default_state()
    merged = {**base, **state}
    merged["starting_capital"] = float(merged.get("starting_capital") or STARTING_CAPITAL_USD)
    for key in (
        "cash_usd",
        "total_net_pnl",
        "cumulative_swap_fees",
        "cumulative_priority_fees",
        "cumulative_total_fees",
    ):
        merged[key] = float(merged.get(key) or 0.0)
    merged.setdefault("open_positions", [])
    merged.setdefault("trading_mode", "DEMO")
    return merged


class PaperTrader:
    def __init__(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._state = self._load_state()
        self._market_prices_by_pair: dict[str, float] = {}
        self._market_prices_by_coin_id: dict[int, float] = {}
        self._market_prices_by_canonical: dict[str, float] = {}
        self._market_price_timestamp: str | None = None

    def set_market_prices(
        self,
        entries: list[dict[str, Any]],
        *,
        price_timestamp: str | None = None,
    ) -> None:
        by_pair, by_coin, by_canonical = build_market_price_maps(entries)
        self._market_prices_by_pair = by_pair
        self._market_prices_by_coin_id = by_coin
        self._market_prices_by_canonical = by_canonical
        self._market_price_timestamp = price_timestamp

    def _wallet_equity(self) -> float:
        return float(self.get_wallet_summary().get("total_equity_usd", 0))

    def _trade_audit_summary(self) -> dict[str, Any]:
        rows = self.get_trades_from_log(limit=100_000)
        audit = audit_trade_rows(rows)
        wallet = self.get_wallet_summary()
        starting = float(wallet.get("starting_capital", STARTING_CAPITAL_USD))
        equity = float(wallet.get("total_equity_usd", 0))
        return {
            **audit,
            "portfolio_roi_pct": round(portfolio_roi_from_equity(
                current_equity=equity,
                starting_capital=starting,
            ), 6),
            "avg_closed_trade_roi_pct": round(
                self.net_roi_summary().get("avg_net_roi_pct", 0.0), 6
            ),
            "paper_state_contaminated": audit["invalid_rows"] > 0,
        }

    def _load_state(self) -> dict[str, Any]:
        if STATE_PATH.exists():
            with open(STATE_PATH, encoding="utf-8") as f:
                state = _normalize_state(json.load(f))
                self._save_state(state)
                return state
        state = _default_state()
        self._save_state(state)
        return state

    def _save_state(self, state: dict[str, Any] | None = None) -> None:
        if state is not None:
            self._state = _normalize_state(state)
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2)

    def _record_fees(self, swap_fee: float, priority_fee: float) -> None:
        total = swap_fee + priority_fee
        self._state["cumulative_swap_fees"] = round(
            float(self._state.get("cumulative_swap_fees", 0)) + swap_fee, 6
        )
        self._state["cumulative_priority_fees"] = round(
            float(self._state.get("cumulative_priority_fees", 0)) + priority_fee, 6
        )
        self._state["cumulative_total_fees"] = round(
            float(self._state.get("cumulative_total_fees", 0)) + total, 6
        )

    def _append_trade_row(self, row: dict[str, Any]) -> int | None:
        import logging

        trade_logger = logging.getLogger("paper")
        try:
            fields = _ensure_trade_csv_header()
            exists = TRADES_LOG_PATH.exists() and TRADES_LOG_PATH.stat().st_size > 0
            serializable: dict[str, Any] = {}
            for key, value in row.items():
                if value is None:
                    serializable[key] = ""
                elif isinstance(value, (list, dict)):
                    serializable[key] = json.dumps(value)
                else:
                    serializable[key] = value
            with open(TRADES_LOG_PATH, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                if not exists:
                    writer.writeheader()
                writer.writerow({field: serializable.get(field, "") for field in fields})
        except Exception as exc:
            trade_logger.warning("CSV trade log write failed: %s", exc)

        try:
            from .. import database as db

            coin_id = row.get("coin_id")
            if not coin_id and row.get("pair_address"):
                coin = db.get_coin_by_pair_address(str(row["pair_address"]))
                coin_id = coin["id"] if coin else None
            if not coin_id and row.get("symbol"):
                coins = db.get_coins(limit=5, sort_by="last_seen")
                match = next((c for c in coins if c.get("symbol") == row["symbol"]), None)
                coin_id = match["id"] if match else None

            event_type = str(row.get("event_type") or "").upper()
            reason_code = str(row.get("reason_code") or "").upper()
            is_rejected = (
                event_type == "RISK_GUARD_BLOCK"
                or reason_code == "RISK_GUARD_BLOCK"
                or reason_code.startswith("REJECTED")
            )
            status = "rejected" if is_rejected else "filled"

            trade_id = db.insert_trade({
                "timestamp": row.get("timestamp"),
                "coin_id": coin_id,
                "symbol": row.get("symbol"),
                "side": row.get("side"),
                "price": row.get("fill_price"),
                "amount": row.get("quantity"),
                "value": row.get("notional_usd"),
                "fee": row.get("total_fees"),
                "slippage": row.get("swap_fee"),
                "pnl": row.get("realized_pnl"),
                "status": status,
                "reason": row.get("reason_code") or row.get("rejection_code") or row.get("event_type"),
                "decision_ref_id": row.get("decision_ref_id"),
                "position_id": row.get("position_id"),
                "chain": row.get("chain"),
                "cluster_label": row.get("cluster_label"),
                "net_roi_pct": row.get("net_roi_pct"),
                "source": "app_paper",
            })
            if trade_id and row.get("decision_ref_id") and row.get("side") == "sell":
                db.update_gemini_decision_outcome(
                    int(row["decision_ref_id"]),
                    linked_trade_id=trade_id,
                    outcome_pnl=float(row.get("realized_pnl") or 0),
                    outcome_status="closed",
                )
            return trade_id
        except Exception as exc:
            trade_logger.warning("SQLite trade sync failed: %s", exc)
            return None

    def positions_market_value(self, price_map: dict[str, float] | None = None) -> float:
        total = 0.0
        for pos in self._state.get("open_positions", []):
            pair_address = str(pos.get("pair_address") or "").strip()
            coin_id = pos.get("coin_id")
            price = None
            if pair_address and pair_address in self._market_prices_by_pair:
                price = self._market_prices_by_pair[pair_address]
            elif coin_id is not None and int(coin_id) in self._market_prices_by_coin_id:
                price = self._market_prices_by_coin_id[int(coin_id)]
            elif price_map and pair_address and pair_address in price_map:
                price = price_map[pair_address]
            else:
                price = float(pos.get("entry_price", 0))
            total += float(price) * float(pos.get("quantity", 0))
        return round(total, 6)

    def get_wallet_summary(self) -> dict[str, Any]:
        cash = float(self._state.get("cash_usd", 0))
        positions_val = self.positions_market_value()
        starting = float(self._state.get("starting_capital", STARTING_CAPITAL_USD))
        equity = cash + positions_val
        return {
            "trading_mode": self._state.get("trading_mode", "DEMO"),
            "starting_capital": starting,
            "cash_usd": round(cash, 2),
            "positions_value_usd": round(positions_val, 2),
            "total_equity_usd": round(equity, 2),
            "open_positions_count": len(self._state.get("open_positions", [])),
            "closed_trades": int(self._state.get("closed_trades", 0)),
            "total_net_pnl": round(float(self._state.get("total_net_pnl", 0)), 6),
            "cumulative_swap_fees": round(float(self._state.get("cumulative_swap_fees", 0)), 6),
            "cumulative_priority_fees": round(
                float(self._state.get("cumulative_priority_fees", 0)), 6
            ),
            "cumulative_total_fees": round(float(self._state.get("cumulative_total_fees", 0)), 6),
            "unrealized_pnl_usd": round(equity - starting - float(self._state.get("total_net_pnl", 0)), 6),
        }

    def get_positions(self, status: str | None = None) -> list[dict[str, Any]]:
        positions = self._state.get("open_positions", [])
        if status and status.upper() != "OPEN":
            return []
        return list(positions)

    def get_last_open_result(self) -> dict[str, Any] | None:
        """Structured result of the most recent open_position() call (success or reject)."""
        result = self._state.get("last_open_result")
        return dict(result) if isinstance(result, dict) else None

    def mark_positions_to_market(
        self, positions: list[dict[str, Any]] | None = None
    ) -> list[dict[str, Any]]:
        """Enrich open positions with live mark price, PnL, and exit-eligibility.

        Never mutates stored state — returns enriched copies. If no current
        mark price is available for a position's pair/coin_id, current_price
        stays None and mark_price_unavailable_reason explains why.
        """
        now = datetime.now(timezone.utc)
        source = positions if positions is not None else self._state.get("open_positions", [])
        marked: list[dict[str, Any]] = []

        for raw in source:
            row = dict(raw)
            pair_address = str(row.get("pair_address") or "").strip()
            coin_id = row.get("coin_id")
            canonical_key = str(
                row.get("canonical_market_identity")
                or row.get("provider_pair_url_exact")
                or row.get("provider_pair_url")
                or row.get("mark_price_lookup_key")
                or ""
            ).strip()

            current_price: float | None = None
            current_price_source: str | None = None
            if canonical_key and canonical_key in self._market_prices_by_canonical:
                current_price = self._market_prices_by_canonical[canonical_key]
                current_price_source = "market_canonical_url"
            elif coin_id is not None:
                try:
                    cid = int(coin_id)
                except (TypeError, ValueError):
                    cid = None
                if cid is not None and cid in self._market_prices_by_coin_id:
                    current_price = self._market_prices_by_coin_id[cid]
                    current_price_source = "market_coin_id"
            elif pair_address and pair_address in self._market_prices_by_pair:
                current_price = self._market_prices_by_pair[pair_address]
                current_price_source = "market_pair_address"

            mark_price_unavailable_reason: str | None = None
            mark_price_lookup_status: str | None = None
            price_resolution_failure_reason: str | None = None
            current_price_timestamp: str | None = None
            price_age_seconds: float | None = None
            price_age_label: str | None = None
            if current_price is None:
                if canonical_key and not pair_address:
                    mark_price_unavailable_reason = "PRICE_NOT_AVAILABLE"
                    mark_price_lookup_status = "PRICE_NOT_AVAILABLE"
                    price_resolution_failure_reason = "no_index_price_for_canonical_url"
                else:
                    mark_price_unavailable_reason = "PRICE_NOT_AVAILABLE"
                    mark_price_lookup_status = "PRICE_NOT_AVAILABLE"
                    price_resolution_failure_reason = "no_mark_price_in_runtime_index"
            else:
                mark_price_lookup_status = "PRICE_AVAILABLE"
                current_price_timestamp = self._market_price_timestamp
                ts = _parse_iso_ts(current_price_timestamp)
                if ts is not None:
                    price_age_seconds = max(0.0, (now - ts).total_seconds())
                    price_age_label = _format_age_label(price_age_seconds)

            opened_ts = _parse_iso_ts(row.get("opened_at"))
            age_seconds: float | None = None
            age_minutes: float | None = None
            age_label: str | None = None
            if opened_ts is not None:
                age_seconds = max(0.0, (now - opened_ts).total_seconds())
                age_minutes = round(age_seconds / 60.0, 2)
                age_label = _format_age_label(age_seconds)

            entry_price = float(row.get("entry_price") or 0)
            quantity = float(row.get("quantity") or 0)

            # AE13I PnL honesty: a mark is only "fresh" when it comes from a
            # live provider-backed price map AND has a known, recent timestamp.
            # Positions using a fallback/entry price or a stale/unknown-age
            # timestamp must never render a computed (including 0.00%) PnL.
            mark_fresh = (
                current_price is not None
                and current_price_source in _FRESH_PRICE_SOURCES
                and price_age_seconds is not None
                and price_age_seconds <= MAX_FRESH_MARK_AGE_SECONDS
            )

            unrealized_pnl_usd: float | None = None
            unrealized_pnl_pct: float | None = None
            distance_to_take_profit_pct: float | None = None
            distance_to_stop_loss_pct: float | None = None
            if mark_fresh and entry_price > 0:
                unrealized_pnl_usd = round((current_price - entry_price) * quantity, 6)
                unrealized_pnl_pct = round((current_price - entry_price) / entry_price, 6)
                take_profit = row.get("take_profit")
                if take_profit:
                    distance_to_take_profit_pct = round(
                        (float(take_profit) - current_price) / current_price, 6
                    )
                stop_loss = row.get("stop_loss")
                if stop_loss:
                    distance_to_stop_loss_pct = round(
                        (current_price - float(stop_loss)) / current_price, 6
                    )

            if current_price is None:
                pnl_display_status = "unavailable"
                pnl_display_message = "PnL unavailable - no fresh mark price"
            elif not mark_fresh:
                pnl_display_status = "stale_estimated"
                pnl_display_message = (
                    "PnL unavailable - mark price is stale or not confirmed fresh"
                )
            else:
                pnl_display_status = "fresh"
                pnl_display_message = None

            peak_price = float(row.get("peak_price") or entry_price or 0)
            if current_price is not None and current_price > peak_price:
                peak_price = current_price
            trailing_pct = row.get("trailing_stop_pct")
            if current_price is not None and trailing_pct and peak_price > 0:
                trailing_trigger = peak_price * (1 - float(trailing_pct))
                trailing_stop_status = "triggered" if current_price <= trailing_trigger else "active"
            else:
                trailing_stop_status = "not_configured"

            time_stop_seconds = row.get("time_stop_seconds")
            time_stop_remaining_seconds: float | None = None
            if time_stop_seconds and age_seconds is not None:
                time_stop_remaining_seconds = max(0.0, float(time_stop_seconds) - age_seconds)

            min_hold_seconds = row.get("min_hold_seconds")
            exit_eligible_now = True
            exit_blocker: str | None = None
            bot_would_exit_now = False
            bot_exit_reason: str | None = None
            if current_price is None:
                exit_eligible_now = False
                exit_blocker = mark_price_unavailable_reason or "No current price"
                bot_exit_reason = "No current price"
            elif (
                min_hold_seconds is not None
                and age_seconds is not None
                and age_seconds < float(min_hold_seconds)
            ):
                exit_eligible_now = False
                exit_blocker = (
                    f"Min hold not reached ({age_label} of "
                    f"{_format_age_label(float(min_hold_seconds))} required)"
                )
                bot_exit_reason = "Min hold not reached"
            else:
                take_profit_hit = (
                    row.get("take_profit") is not None
                    and current_price >= float(row["take_profit"])
                )
                stop_loss_hit = (
                    row.get("stop_loss") is not None
                    and current_price <= float(row["stop_loss"])
                )
                trailing_hit = trailing_stop_status == "triggered"
                time_stop_hit = (
                    time_stop_remaining_seconds is not None
                    and time_stop_remaining_seconds <= 0
                )
                if take_profit_hit:
                    bot_would_exit_now = True
                    bot_exit_reason = "Take-profit reached"
                elif stop_loss_hit:
                    bot_would_exit_now = True
                    bot_exit_reason = "Stop-loss reached"
                elif trailing_hit:
                    bot_would_exit_now = True
                    bot_exit_reason = "Trailing stop triggered"
                elif time_stop_hit:
                    bot_would_exit_now = True
                    bot_exit_reason = "Time-stop reached"
                elif time_stop_remaining_seconds is not None:
                    bot_exit_reason = (
                        f"Waiting for time-stop "
                        f"({_format_age_label(time_stop_remaining_seconds)} left); "
                        "price has not reached TP/SL"
                    )
                else:
                    bot_exit_reason = "Price has not reached TP/SL"

            exit_plan = row.get("exit_plan") if isinstance(row.get("exit_plan"), dict) else {}
            tp_pct = exit_plan.get("take_profit_pct")
            sl_pct = exit_plan.get("stop_loss_pct")
            trail_pct = exit_plan.get("trailing_stop_pct", row.get("trailing_stop_pct"))
            hold_secs = exit_plan.get("min_hold_seconds", min_hold_seconds)
            time_secs = exit_plan.get("time_stop_seconds", time_stop_seconds)
            # Infer pct from absolute levels when exit_plan is absent.
            if tp_pct is None and row.get("take_profit") and entry_price > 0:
                tp_pct = round(float(row["take_profit"]) / entry_price - 1.0, 4)
            if sl_pct is None and row.get("stop_loss") and entry_price > 0:
                sl_pct = round(1.0 - float(row["stop_loss"]) / entry_price, 4)
            exit_plan_parts: list[str] = []
            if tp_pct is not None:
                exit_plan_parts.append(f"TP +{float(tp_pct) * 100:.0f}%")
            if sl_pct is not None:
                exit_plan_parts.append(f"SL -{float(sl_pct) * 100:.0f}%")
            if trail_pct is not None:
                exit_plan_parts.append(f"trailing stop {float(trail_pct) * 100:.0f}%")
            if time_secs is not None:
                exit_plan_parts.append(f"time stop {_format_age_label(float(time_secs))}")
            if hold_secs is not None:
                exit_plan_parts.append(f"min hold {_format_age_label(float(hold_secs))}")
            exit_plan_summary = " · ".join(exit_plan_parts) if exit_plan_parts else "Exit plan not configured"

            matched_market_pair_status = (
                "matched" if current_price is not None else "unmatched_no_current_price"
            )

            row.update({
                "current_price": current_price,
                "current_price_timestamp": current_price_timestamp,
                "price_age_seconds": price_age_seconds,
                "price_age_label": price_age_label,
                "current_price_source": current_price_source,
                "canonical_market_identity": canonical_key or row.get("canonical_market_identity"),
                "mark_price_lookup_key": canonical_key or row.get("mark_price_lookup_key"),
                "mark_price_lookup_status": mark_price_lookup_status,
                "price_resolution_failure_reason": price_resolution_failure_reason,
                "pair_address_derived": pair_address or None,
                "age_seconds": age_seconds,
                "age_minutes": age_minutes,
                "age_label": age_label,
                "unrealized_pnl_usd": unrealized_pnl_usd,
                "unrealized_pnl_pct": unrealized_pnl_pct,
                "distance_to_take_profit_pct": distance_to_take_profit_pct,
                "distance_to_stop_loss_pct": distance_to_stop_loss_pct,
                "trailing_stop_status": trailing_stop_status,
                "time_stop_remaining_seconds": time_stop_remaining_seconds,
                "exit_eligible_now": exit_eligible_now,
                "exit_blocker": exit_blocker,
                "bot_would_exit_now": bot_would_exit_now,
                "bot_exit_reason": bot_exit_reason,
                "exit_plan_summary": exit_plan_summary,
                "manual_close_allowed": True,
                "manual_close_note": "You can manually close this demo position now.",
                "matched_market_pair_status": matched_market_pair_status,
                "mark_price_unavailable_reason": mark_price_unavailable_reason,
                # AE13I PnL honesty fields
                "mark_fresh": mark_fresh,
                "pnl_display_status": pnl_display_status,
                "pnl_display_message": pnl_display_message,
                "close_freshness_status": None,
                "paper_demo_only": True,
                "not_live_approved": True,
                "not_profitability_evidence": True,
            })
            _pmds = resolve_position_market_data_state(
                row,
                current_price=current_price,
                mark_fresh=mark_fresh,
                price_age_seconds=price_age_seconds,
            )
            _qty = float(row.get("quantity") or 0)
            _pos_val = (
                (current_price * _qty) if (mark_fresh and current_price is not None) else None
            )
            _fin = build_position_financial_dto(
                row,
                position_market_data_state=_pmds,
                current_price=current_price if mark_fresh else None,
                position_value=_pos_val,
                unrealized_pnl=unrealized_pnl_usd,
                unrealized_pnl_pct=unrealized_pnl_pct,
            )
            row["position_market_data_state"] = _pmds
            row.update(_fin)
            # Financial DTO numerics are authoritative for UI; do not wipe
            # current_price (mark lookup result) — last_good is never copied
            # into current_price_numeric / display when state is not DATA_OK.

            try:
                from app.ae13b_product.address_role import enrich_row_with_address_role

                row.update(enrich_row_with_address_role(row))
            except Exception:
                pass

            try:
                from app.ae13b_product.mtm_traffic_light import compute_traffic_light

                row.update(compute_traffic_light(row))
            except Exception:
                row["traffic_light_status"] = "yellow"
                row["traffic_light_label"] = "Caution"
                row["traffic_light_reason"] = "Traffic light unavailable."

            marked.append(row)

        return marked

    def get_marked_positions(self, status: str = "OPEN") -> list[dict[str, Any]]:
        """Open positions enriched via mark_positions_to_market()."""
        return self.mark_positions_to_market(self.get_positions(status=status))

    def get_trades_from_log(self, limit: int = 100) -> list[dict[str, Any]]:
        if not TRADES_LOG_PATH.exists():
            return []
        with open(TRADES_LOG_PATH, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return rows[-limit:]

    def get_state_snapshot(self) -> dict[str, Any]:
        w = self.get_wallet_summary()
        return {
            **w,
            "open_positions": list(self._state.get("open_positions", [])),
        }

    def net_roi_summary(self) -> dict[str, Any]:
        rows = self.get_trades_from_log(500)
        audit = audit_trade_rows(rows)
        valid_sells = [
            t for t in audit["valid_row_details"] if str(t.get("side")).lower() == "sell"
        ]
        wallet = self.get_wallet_summary()
        starting = float(wallet.get("starting_capital", STARTING_CAPITAL_USD))
        equity = float(wallet.get("total_equity_usd", 0))
        portfolio_roi = portfolio_roi_from_equity(
            current_equity=equity,
            starting_capital=starting,
        )
        if not valid_sells:
            return {
                "trade_count": 0,
                "avg_net_roi_pct": 0.0,
                "portfolio_roi_pct": round(portfolio_roi, 6),
                "total_net_pnl": wallet["total_net_pnl"],
                "cumulative_total_fees": wallet["cumulative_total_fees"],
                "invalid_trade_rows_excluded": audit["invalid_rows"],
            }
        rois = [float(t.get("net_roi_pct") or 0) for t in valid_sells]
        return {
            "trade_count": len(valid_sells),
            "avg_net_roi_pct": round(sum(rois) / len(rois), 6),
            "portfolio_roi_pct": round(portfolio_roi, 6),
            "total_net_pnl": wallet["total_net_pnl"],
            "cumulative_total_fees": wallet["cumulative_total_fees"],
            "invalid_trade_rows_excluded": audit["invalid_rows"],
        }

    def chart_series(self) -> dict[str, Any]:
        """Data for frontend Chart.js widgets."""
        sells = [t for t in self.get_trades_from_log(500) if t.get("side") == "sell"]
        roi_labels = [t.get("timestamp", "")[:16] for t in sells]
        roi_values = [round(float(t.get("net_roi_pct") or 0) * 100, 4) for t in sells]
        cumulative_pnl: list[float] = []
        running = 0.0
        for t in sells:
            running += float(t.get("realized_pnl") or 0)
            cumulative_pnl.append(round(running, 2))
        return {
            "net_roi_labels": roi_labels,
            "net_roi_pct": roi_values,
            "cumulative_pnl": cumulative_pnl,
        }

    def compute_strategy_notional(self, strategy_type: str) -> float:
        """
        Dual-strategy portfolio allocation based on total wallet equity.
        SCALPING_OPPORTUNITY → 10% | WHALE_RIDER → 30%
        """
        wallet = self.get_wallet_summary()
        equity = float(wallet.get("total_equity_usd", 0))
        cash = float(wallet.get("cash_usd", 0))
        if strategy_type == "WHALE_RIDER":
            pct = WHALE_RIDER_EQUITY_PCT
        else:
            pct = SCALPING_EQUITY_PCT
        target = round(equity * pct, 2)
        # Cap by available cash (reserve ~2% for entry fees)
        return round(min(target, cash * 0.98), 2)

    def reset_demo_wallet(self) -> dict[str, Any]:
        self._state = _default_state()
        self._save_state()
        return self.get_wallet_summary()

    def set_trading_mode(self, mode: str) -> str:
        mode_upper = mode.upper()
        if mode_upper not in ("DEMO", "LIVE"):
            mode_upper = "DEMO"
        self._state["trading_mode"] = mode_upper
        self._save_state()
        return mode_upper

    def _assert_execution_guard(self, *, reason_code: str = "SIGNAL") -> None:
        """Fail-closed DEMO/PAPER guard inside the execution path (not frontend-only)."""
        from app.ae13b_product.execution_guard import (
            DemoExecutionGuardError,
            assert_paper_demo_allowed,
            resolve_runtime_guard_context,
        )

        ctx = resolve_runtime_guard_context()
        mode = str(self._state.get("trading_mode") or ctx.get("trading_mode") or "DEMO").upper()
        if mode == "LIVE" or bool(ctx.get("live_trading_enabled")):
            raise DemoExecutionGuardError(
                ["live_trading_enabled_true"],
                detail={"reason_code": reason_code, "trading_mode": mode},
            )
        acceptance = "DEMO_TEST" in str(reason_code).upper() or "ACCEPTANCE" in str(reason_code).upper()
        flags = {
            "paper_demo_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
        }
        if acceptance:
            flags["demo_acceptance_only"] = True
            flags["not_strategy_evidence"] = True
        assert_paper_demo_allowed(
            trading_mode="DEMO" if mode not in ("DEMO", "PAPER") else mode,
            live_trading_enabled=False,
            wallet_configured=False,
            private_key_accessed=False,
            real_signing_enabled=False,
            real_submission_enabled=False,
            order_flags=flags,
            demo_acceptance_mode_enabled=True if acceptance else None,
        )

    def open_position(
        self,
        coin: dict[str, Any],
        *,
        size_usd: float | None = None,
        cluster_label: str = "SOCIALLY_MOTIVATED",
        settings: dict[str, Any] | None = None,
        reason_code: str = "SIGNAL",
        strategy_type: str = "SCALPING_OPPORTUNITY",
        allow_coin_price_fallback: bool = False,
        skip_execution_guard: bool = False,
        bot_state: dict[str, Any] | None = None,
        pair_cooldowns: dict[str, Any] | None = None,
        risk_mode: str | None = None,
        preset_id: str | None = None,
        gate_result: dict[str, Any] | None = None,
        skip_market_data_gate: bool = False,
    ) -> dict[str, Any] | None:
        # AE19 defense-in-depth: refuse explicit LLM-origin open attempts.
        llm_provider = extract_explicit_llm_origin_provider(coin, settings or {})
        if llm_provider:
            block = _llm_audit_only_block_result(llm_provider, "BUY")
            self._state["last_open_result"] = block
            self._save_state()
            return None

        if not skip_execution_guard:
            try:
                self._assert_execution_guard(reason_code=reason_code)
            except Exception as exc:
                log.warning("BUY rejected by execution guard: %s", exc)
                return None

        # AE13I defense-in-depth: the PRIMARY freshness/provenance/reentry gate
        # is MarketDataGateKeeper, run upstream of PaperTrader by demo_bot /
        # demo_queue / watchlist. This secondary call exists only so a manual
        # /api/demo/buy (or any other direct open_position() call) cannot
        # bypass the gate — it reuses the same modular gatekeeper rather than
        # duplicating freshness/provenance logic here.
        if not skip_market_data_gate:
            _gate = gate_result
            if _gate is None:
                try:
                    from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

                    # AE13I fix: stagnant_price_guard now returns passed=True
                    # (momentum_evidence="unknown_insufficient_delta_fields")
                    # when a coin dict carries no delta fields at all, rather
                    # than hard-blocking on missing data, so it is safe to
                    # run the guard here (skip_stagnant=False).
                    _gate = validate_market_data_gate(coin, for_open=True, skip_stagnant=False)
                except Exception as exc:
                    log.warning("BUY defense-in-depth gate check failed (fail-open): %s", exc)
                    _gate = None
            if _gate is not None and not _gate.get("passed"):
                rejection_reason = (
                    (_gate.get("rejection_reasons") or ["Blocked by market data gate"])[0]
                )
                log.warning(
                    "BUY rejected by market data gate (defense-in-depth): %s pair=%s",
                    rejection_reason,
                    coin.get("pair_address"),
                )
                self._state["last_open_result"] = {
                    "opened": False,
                    "symbol": coin.get("symbol"),
                    "pair_address": coin.get("pair_address"),
                    "rejection_code": _gate.get("rejection_code") or "NOT_OPENED_STALE_MARKET_DATA",
                    "rejection_reason": rejection_reason,
                    "rejection_reasons": list(_gate.get("rejection_reasons") or []),
                    "blocking_guards": list(_gate.get("blocking_guards") or []),
                    "primary_blocker": _gate.get("primary_blocker"),
                    "tradability_status": _gate.get("tradability_status"),
                    "decision": _gate.get("decision"),
                    "paper_demo_only": True,
                    "not_live_approved": True,
                    "not_profitability_evidence": True,
                }
                self._save_state()
                return None

        # AE18: block new paper/demo entry when trade_readiness is blocked.
        _tr = str(coin.get("trade_readiness_status") or "").strip()
        if _tr:
            _allowed, _block_reason = assert_new_entry_allowed(_tr)
            if (not _allowed) or entry_blocked(_tr):
                log.warning(
                    "BUY rejected by trade_readiness_status: %s pair=%s",
                    _tr,
                    coin.get("pair_address"),
                )
                self._state["last_open_result"] = {
                    "opened": False,
                    "symbol": coin.get("symbol"),
                    "pair_address": coin.get("pair_address"),
                    "rejection_code": "ENTRY_BLOCKED_TRADE_READINESS",
                    "rejection_reason": _block_reason or _tr,
                    "trade_readiness_status": _tr,
                    "paper_demo_only": True,
                    "not_live_approved": True,
                    "not_profitability_evidence": True,
                }
                self._save_state()
                return None

        settings = settings or {}
        capital = float(settings.get("starting_capital", self._state.get("starting_capital", STARTING_CAPITAL_USD)))
        size_pct = float(settings.get("max_position_size_pct", 0.05))
        stop_loss_pct = float(settings.get("stop_loss_pct", 0.08))
        take_profit_pct = float(settings.get("take_profit_pct", 0.15))
        max_position_usd = float(settings.get("max_position_size_usd", capital * size_pct))

        if size_usd is None:
            size_usd = self.compute_strategy_notional(strategy_type)
        requested_notional = float(size_usd or (capital * size_pct))
        requested_notional = min(requested_notional, max_position_usd)

        # Backend portfolio risk guard — cannot be bypassed by UI / Demo Queue
        bot_state_ctx = bot_state if bot_state is not None else settings.get("bot_state")
        pair_cooldowns_ctx = (
            pair_cooldowns if pair_cooldowns is not None else settings.get("pair_cooldowns")
        )
        risk_mode_ctx = risk_mode if risk_mode is not None else settings.get("risk_mode")
        preset_id_ctx = preset_id if preset_id is not None else settings.get("preset_id")
        try:
            from app.ae13b_product.demo_risk_guard import evaluate_demo_risk_guard
            from app.ae13b_product.rejected_attempt import RejectedTradeAttempt

            risk = evaluate_demo_risk_guard(
                requested_notional=requested_notional,
                demo_equity=float(self._wallet_equity() or capital),
                open_positions=self.get_positions(status="OPEN"),
                recent_trades=self.get_trades_from_log(limit=500),
                pair_address=str(coin.get("pair_address") or ""),
                symbol=coin.get("symbol"),
                chain=coin.get("chain"),
                price=coin.get("latest_price") or coin.get("price_usd") or coin.get("price"),
                price_timestamp=coin.get("last_seen_at") or coin.get("price_timestamp"),
                liquidity=coin.get("latest_liquidity") or coin.get("liquidity_usd"),
                strategy_lane=strategy_type,
                settings=settings,
                bot_state=bot_state_ctx,
                pair_cooldowns=pair_cooldowns_ctx,
                risk_mode=risk_mode_ctx,
                preset_id=preset_id_ctx,
                token_contract_address=coin.get("token_contract_address")
                or coin.get("contract_address"),
            )
            self._state["last_risk_guard"] = risk
            if not risk.get("risk_guard_passed"):
                rejection_reason = (
                    risk.get("rejection_reason")
                    or risk.get("risk_guard_reason")
                    or "risk_guard_blocked"
                )
                log.warning(
                    "BUY rejected by demo risk guard: %s pair=%s",
                    rejection_reason,
                    coin.get("pair_address"),
                )
                attempt = RejectedTradeAttempt.from_risk_guard(
                    coin,
                    risk,
                    strategy_lane=strategy_type,
                    preset_id=preset_id_ctx or risk.get("preset_id"),
                    risk_mode=risk_mode_ctx or risk.get("risk_mode"),
                    notional_requested=requested_notional,
                )
                self._append_trade_row(attempt.to_dict())
                self._state["last_open_result"] = {
                    "opened": False,
                    "symbol": coin.get("symbol"),
                    "pair_address": coin.get("pair_address"),
                    "rejection_code": risk.get("rejection_code"),
                    "rejection_reason": rejection_reason,
                    "rejection_reasons": list(risk.get("rejection_reasons") or []),
                    "blocking_guards": list(risk.get("blocking_guards") or []),
                    "primary_blocker": risk.get("primary_blocker"),
                    "requested_notional": requested_notional,
                    "checked_at_utc": risk.get("checked_at_utc"),
                    "paper_demo_only": True,
                    "not_live_approved": True,
                    "not_profitability_evidence": True,
                }
                self._save_state()
                return None
            requested_notional = float(risk.get("approved_notional") or requested_notional)
        except Exception as exc:
            log.warning("BUY rejected — risk guard error (fail-closed): %s", exc)
            self._state["last_open_result"] = {
                "opened": False,
                "symbol": coin.get("symbol"),
                "pair_address": coin.get("pair_address"),
                "rejection_code": "RISK_GUARD_ERROR",
                "rejection_reason": f"risk guard error (fail-closed): {exc}",
                "rejection_reasons": [f"risk guard error (fail-closed): {exc}"],
                "blocking_guards": ["risk_guard_error"],
                "paper_demo_only": True,
                "not_live_approved": True,
                "not_profitability_evidence": True,
            }
            self._save_state()
            return None

        cash_before = float(self._state.get("cash_usd", 0))
        equity_before = self._wallet_equity()

        resolution = resolve_buy_fill_price(
            coin,
            market_prices_by_pair=self._market_prices_by_pair,
            market_prices_by_coin_id=self._market_prices_by_coin_id,
            price_timestamp=self._market_price_timestamp,
            allow_coin_price_fallback=allow_coin_price_fallback,
        )
        if not resolution.ok or resolution.price is None:
            log.warning(
                "BUY rejected %s pair=%s coin_id=%s reason=%s",
                coin.get("symbol"),
                resolution.pair_address,
                resolution.coin_id,
                resolution.rejection_reason,
            )
            self._append_trade_row({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "position_id": "",
                "symbol": coin.get("symbol"),
                "chain": coin.get("chain", "solana"),
                "side": "buy",
                "quantity": 0,
                "fill_price": "",
                "notional_usd": 0,
                "swap_fee": 0,
                "priority_fee": 0,
                "total_fees": 0,
                "gross_pnl": 0,
                "realized_pnl": 0,
                "net_roi_pct": 0,
                "cluster_label": cluster_label,
                "reason_code": f"REJECTED_{reason_code}",
                "coin_id": resolution.coin_id,
                "pair_address": resolution.pair_address,
                "decision_ref_id": coin.get("decision_ref_id"),
                "fill_price_source": resolution.source,
                "market_price_usd": resolution.market_price_usd,
                "price_timestamp": resolution.price_timestamp,
                "cash_before": round(cash_before, 6),
                "equity_before": round(equity_before, 6),
                "notional_requested": round(requested_notional, 6),
                "notional_executed": 0,
                "rejection_reason": resolution.rejection_reason,
            })
            return None

        price = float(resolution.price)
        chain = str(coin.get("chain", "solana"))
        size_usd = requested_notional
        quantity = size_usd / price

        if quantity <= 0 or notional_exceeds_equity_guard(size_usd, equity_before):
            log.warning("BUY rejected %s invalid quantity/notional guard", coin.get("symbol"))
            return None

        costs = compute_transaction_costs(size_usd, chain)
        total_fees = costs.total
        cash = cash_before

        if cash < size_usd + total_fees:
            log.warning("BUY rejected %s insufficient cash", coin.get("symbol"))
            return None
        if total_fees / max(size_usd, 1e-9) > MAX_FEE_TO_NOTIONAL_PCT:
            log.warning("BUY rejected %s fee/notional anomaly", coin.get("symbol"))
            return None

        pos_id = int(self._state.get("next_position_id", 1))
        coin_id = resolution.coin_id
        pair_address = resolution.pair_address
        instrument_id = (
            str(coin.get("instrument_id") or coin.get("execution_instrument_id") or "").strip()
            or None
        )
        instrument_source = coin.get("instrument_source") or coin.get("candidate_source")
        pos = {
            "id": pos_id,
            "symbol": coin["symbol"],
            "chain": chain,
            "quantity": quantity,
            "entry_price": price,
            "fill_price": price,
            "size_usd": size_usd,
            "stop_loss": price * (1 - stop_loss_pct),
            "take_profit": price * (1 + take_profit_pct),
            "entry_swap_fee": costs.swap_fee,
            "entry_priority_fee": costs.priority_fee,
            "entry_fees": total_fees,
            "cluster_label": cluster_label,
            "strategy_type": strategy_type,
            "status": "OPEN",
            "pair_address": pair_address,
            "coin_id": coin_id,
            "decision_ref_id": coin.get("decision_ref_id"),
            "opened_at": datetime.now(timezone.utc).isoformat(),
            "fill_price_source": resolution.source,
        }
        # Persist URL-first identity when available (display + mark-price key).
        canon = str(
            coin.get("canonical_market_identity")
            or coin.get("provider_pair_url_exact")
            or coin.get("provider_pair_url")
            or coin.get("open_chart_url")
            or ""
        ).strip()
        if canon:
            pos["canonical_market_identity"] = canon
            pos["provider_pair_url_exact"] = canon
            pos["mark_price_lookup_key"] = canon
            pos["open_chart_url"] = canon
        for fld in (
            "provider_pair_url_final_segment_exact",
            "symbol_pair_display",
            "provider_base_token_symbol",
            "provider_quote_token_symbol",
            "pair_address_derived",
        ):
            if coin.get(fld):
                pos[fld] = coin.get(fld)
        attach_entry_snapshot_to_position(pos, coin)
        # Stamp canonical instrument identity when present (Clean Forward /
        # future adapters). Never invent coin_id; leave it None for CF.
        if instrument_id:
            pos.update(
                {
                    "instrument_id": instrument_id,
                    "execution_instrument_id": instrument_id,
                    "instrument_source": instrument_source,
                    "candidate_source": coin.get("candidate_source") or instrument_source,
                    "provider_pair_id": coin.get("provider_pair_id"),
                    "base_token_address": coin.get("base_token_address"),
                    "quote_token_address": coin.get("quote_token_address"),
                    "pair": coin.get("pair"),
                    "liquidity_at_entry": coin.get("liquidity_at_entry")
                    or coin.get("latest_liquidity")
                    or coin.get("liquidity_usd"),
                    "price_updated_at": coin.get("price_updated_at"),
                    "liquidity_updated_at": coin.get("liquidity_updated_at"),
                    "clean_forward_bridge_used": bool(
                        coin.get("clean_forward_bridge_used")
                    ),
                    "legacy_market_snapshots_used": bool(
                        coin.get("legacy_market_snapshots_used") or False
                    ),
                    "execution_mode": coin.get("execution_mode") or "paper",
                    "live_trading_ready": False,
                    "live_execution_enabled": False,
                    "wallet_required": False,
                    "wallet_connected": False,
                    "paper_demo_only": True,
                    "not_live_approved": True,
                    "not_profitability_evidence": True,
                }
            )
        self._state.setdefault("open_positions", []).append(pos)
        self._state["next_position_id"] = pos_id + 1
        self._state["cash_usd"] = round(cash - size_usd - total_fees, 6)
        if self._state["cash_usd"] < 0:
            self._state["cash_usd"] = 0.0
        self._record_fees(costs.swap_fee, costs.priority_fee)
        self._state["last_open_result"] = {
            "opened": True,
            "position_id": pos_id,
            "symbol": coin.get("symbol"),
            "pair_address": pair_address,
            "instrument_id": instrument_id,
            "coin_id": coin_id,
            "notional_usd": round(size_usd, 6),
            "fill_price": price,
            "rejection_reasons": [],
            "blocking_guards": [],
            "execution_mode": coin.get("execution_mode") or ("paper" if instrument_id else None),
            "live_trading_ready": False,
            "live_execution_enabled": False,
            "paper_demo_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
            "clean_forward_bridge_used": bool(coin.get("clean_forward_bridge_used")),
            "legacy_market_snapshots_used": bool(
                coin.get("legacy_market_snapshots_used") or False
            ),
        }
        self._save_state()

        trade_row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "position_id": pos_id,
            "symbol": coin["symbol"],
            "chain": chain,
            "side": "buy",
            "quantity": round(quantity, 10),
            "fill_price": price,
            "notional_usd": round(size_usd, 6),
            "swap_fee": costs.swap_fee,
            "priority_fee": costs.priority_fee,
            "total_fees": total_fees,
            "gross_pnl": 0.0,
            "realized_pnl": 0.0,
            "net_roi_pct": 0.0,
            "cluster_label": cluster_label,
            "reason_code": reason_code,
            "coin_id": coin_id if coin_id is not None else "",
            "pair_address": pair_address,
            "decision_ref_id": coin.get("decision_ref_id"),
            "fill_price_source": resolution.source,
            "market_price_usd": resolution.market_price_usd,
            "price_timestamp": resolution.price_timestamp,
            "cash_before": round(cash_before, 6),
            "equity_before": round(equity_before, 6),
            "notional_requested": round(requested_notional, 6),
            "notional_executed": round(size_usd, 6),
            "rejection_reason": "",
        }
        if instrument_id:
            trade_row.update(
                {
                    "instrument_id": instrument_id,
                    "execution_instrument_id": instrument_id,
                    "instrument_source": instrument_source,
                    "candidate_source": coin.get("candidate_source") or instrument_source,
                    "provider_pair_id": coin.get("provider_pair_id"),
                    "base_token_address": coin.get("base_token_address"),
                    "quote_token_address": coin.get("quote_token_address"),
                    "pair": coin.get("pair"),
                    "liquidity_at_entry": coin.get("liquidity_at_entry")
                    or coin.get("latest_liquidity"),
                    "price_updated_at": coin.get("price_updated_at"),
                    "liquidity_updated_at": coin.get("liquidity_updated_at"),
                    "execution_mode": coin.get("execution_mode") or "paper",
                    "live_trading_ready": False,
                    "live_execution_enabled": False,
                    "wallet_required": False,
                    "wallet_connected": False,
                    "clean_forward_bridge_used": True
                    if coin.get("clean_forward_bridge_used")
                    else False,
                    "legacy_market_snapshots_used": False,
                    "paper_demo_only": True,
                    "not_live_approved": True,
                    "not_profitability_evidence": True,
                }
            )
        self._append_trade_row(trade_row)
        return pos

    def try_autonomous_buy(
        self,
        coin: dict[str, Any],
        cluster_label: str,
        settings: dict[str, Any],
        *,
        risk_score: int = 50,
        max_risk_score: int = 70,
        strategy_type: str = "SCALPING_OPPORTUNITY",
        size_usd: float | None = None,
    ) -> dict[str, Any] | None:
        llm_provider = extract_explicit_llm_origin_provider(coin, settings)
        if llm_provider:
            block = _llm_audit_only_block_result(llm_provider, "BUY")
            self._state["last_open_result"] = block
            self._save_state()
            return None
        if not settings.get("auto_execution_enabled", True):
            return None
        if self._state.get("trading_mode", "DEMO") == "LIVE":
            return None
        if settings.get("enforce_risk_gate", False) and risk_score > int(
            settings.get("max_risk_score", max_risk_score)
        ):
            return None
        symbol = coin.get("symbol", "")
        open_positions = self._state.get("open_positions", [])
        pair_address = str(coin.get("pair_address") or "").strip()
        coin_id = _resolve_coin_id(coin)
        if any(
            str(p.get("pair_address") or "").strip() == pair_address and pair_address
            for p in open_positions
        ):
            return None
        if coin_id is not None and any(p.get("coin_id") == coin_id for p in open_positions):
            return None
        if symbol and any(p.get("symbol") == symbol for p in open_positions):
            return None
        notional = size_usd if size_usd is not None else self.compute_strategy_notional(strategy_type)
        return self.open_position(
            coin,
            size_usd=notional,
            cluster_label=cluster_label,
            settings=settings,
            reason_code=f"AGENT_BUY_{strategy_type}",
            strategy_type=strategy_type,
        )

    def _resolve_open_position(
        self,
        *,
        position_id: int | None = None,
        pair_address: str | None = None,
        coin_id: int | None = None,
        symbol: str | None = None,
    ) -> dict[str, Any] | None:
        positions = self._state.get("open_positions", [])
        if not positions:
            return None
        if position_id is not None:
            return next((p for p in positions if p["id"] == position_id), None)
        norm_pair = str(pair_address or "").strip()
        if norm_pair:
            matches = [
                p for p in positions
                if str(p.get("pair_address") or "").strip() == norm_pair
            ]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                return None
        if coin_id is not None:
            matches = [p for p in positions if p.get("coin_id") == coin_id]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                return None
        if symbol:
            matches = [p for p in positions if p.get("symbol") == symbol]
            if len(matches) == 1:
                return matches[0]
        return None

    def try_autonomous_sell(
        self,
        *,
        symbol: str | None = None,
        position_id: int | None = None,
        pair_address: str | None = None,
        coin_id: int | None = None,
        cur_price: float | None = None,
        settings: dict[str, Any] | None = None,
        provider: str | None = None,
        llm_provider: str | None = None,
        model_provider: str | None = None,
        source_provider: str | None = None,
        decision_provider: str | None = None,
        model_source: str | None = None,
    ) -> dict[str, Any] | None:
        """Close an open paper position per autonomous SELL decision."""
        settings = settings or {}
        origin = {
            "provider": provider,
            "llm_provider": llm_provider,
            "model_provider": model_provider,
            "source_provider": source_provider,
            "decision_provider": decision_provider,
            "model_source": model_source,
        }
        llm_origin = extract_explicit_llm_origin_provider(origin, settings)
        if llm_origin:
            block = _llm_audit_only_block_result(llm_origin, "SELL")
            self._state["last_close_result"] = block
            self._save_state()
            return None
        if not settings.get("auto_execution_enabled", True):
            return None
        if self._state.get("trading_mode", "DEMO") == "LIVE":
            return None

        target = self._resolve_open_position(
            position_id=position_id,
            pair_address=pair_address,
            coin_id=coin_id,
            symbol=symbol,
        )
        if target is None:
            return None

        return self.close_position(
            int(target["id"]),
            cur_price,
            reason_code="AGENT_SELL",
            proposed_pair_address=pair_address,
            proposed_coin_id=coin_id,
            provider=provider,
            llm_provider=llm_provider,
            model_provider=model_provider,
            source_provider=source_provider,
            decision_provider=decision_provider,
            model_source=model_source,
        )

    def close_position(
        self,
        pos_id: int,
        cur_price: float | None = None,
        *,
        reason_code: str = "MANUAL",
        proposed_pair_address: str | None = None,
        proposed_coin_id: int | None = None,
        skip_execution_guard: bool = False,
        close_reason: str | None = None,
        close_note: str | None = None,
        closed_by: str = "user_manual",
        close_price_source: str | None = None,
        close_price_age_seconds: float | None = None,
        close_freshness_status: str | None = None,
        close_used_fallback_price: bool | None = None,
        manual_close_warning_shown: bool | None = None,
        provider: str | None = None,
        llm_provider: str | None = None,
        model_provider: str | None = None,
        source_provider: str | None = None,
        decision_provider: str | None = None,
        model_source: str | None = None,
    ) -> dict[str, Any] | None:
        # AE19 defense-in-depth: refuse explicit LLM-origin close attempts.
        llm_origin = extract_explicit_llm_origin_provider(
            {
                "provider": provider,
                "llm_provider": llm_provider,
                "model_provider": model_provider,
                "source_provider": source_provider,
                "decision_provider": decision_provider,
                "model_source": model_source,
            }
        )
        if llm_origin:
            block = _llm_audit_only_block_result(llm_origin, "SELL")
            self._state["last_close_result"] = block
            self._save_state()
            return None

        if not skip_execution_guard:
            try:
                self._assert_execution_guard(reason_code=reason_code)
            except Exception as exc:
                log.warning("SELL rejected by execution guard: %s", exc)
                return None
        positions = self._state.get("open_positions", [])
        idx = next((i for i, p in enumerate(positions) if p["id"] == pos_id), None)
        if idx is None:
            return None

        pos = positions[idx]
        entry = float(pos["entry_price"])
        qty = float(pos["quantity"])
        chain = str(pos["chain"])
        cash_before = float(self._state.get("cash_usd", 0))
        equity_before = self._wallet_equity()

        # MANUAL_DEMO_SELL_ENTRY_DEVIATION_GUARD_BYPASS_V1
        # The default sell resolver compares the candidate close price to the entry price
        # and rejects moves beyond PAPER_PRICE_SANITY_MAX_DEVIATION_PCT, historically 50%.
        # That is appropriate as a general sanity guard, but it is wrong for an explicit
        # user manual demo close: a user must be able to close a paper/demo position after
        # a large adverse move. Identity, positive-price, min/max price, and DEMO-only
        # guards still apply; this only disables the entry-price deviation rejection for
        # manual demo closes.
        manual_sell_entry_deviation_limit = (
            1_000_000_000.0
            if (
                str(reason_code or "").upper() == "MANUAL_SELL"
                or str(closed_by or "").lower() == "user_manual"
            )
            else PAPER_PRICE_SANITY_MAX_DEVIATION_PCT
        )

        resolution = resolve_sell_fill_price(
            pos,
            market_prices_by_pair=self._market_prices_by_pair,
            market_prices_by_coin_id=self._market_prices_by_coin_id,
            proposed_price=cur_price,
            proposed_pair_address=proposed_pair_address or pos.get("pair_address"),
            proposed_coin_id=proposed_coin_id if proposed_coin_id is not None else pos.get("coin_id"),
            price_timestamp=self._market_price_timestamp,
            max_deviation_pct=manual_sell_entry_deviation_limit,
        )
        if not resolution.ok or resolution.price is None:
            log.warning(
                "SELL rejected #%s %s reason=%s",
                pos_id,
                pos.get("symbol"),
                resolution.rejection_reason,
            )
            self._append_trade_row({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "position_id": pos_id,
                "symbol": pos.get("symbol"),
                "chain": chain,
                "side": "sell",
                "quantity": 0,
                "fill_price": "",
                "notional_usd": 0,
                "swap_fee": 0,
                "priority_fee": 0,
                "total_fees": 0,
                "gross_pnl": 0,
                "realized_pnl": 0,
                "net_roi_pct": 0,
                "cluster_label": pos.get("cluster_label", ""),
                "reason_code": f"REJECTED_{reason_code}",
                "coin_id": pos.get("coin_id"),
                "pair_address": pos.get("pair_address"),
                "decision_ref_id": pos.get("decision_ref_id"),
                "fill_price_source": resolution.source,
                "market_price_usd": resolution.market_price_usd,
                "price_timestamp": resolution.price_timestamp,
                "cash_before": round(cash_before, 6),
                "equity_before": round(equity_before, 6),
                "notional_requested": round(qty * (cur_price or 0), 6) if cur_price else 0,
                "notional_executed": 0,
                "rejection_reason": resolution.rejection_reason,
            })
            return None

        cur_price = float(resolution.price)
        if qty <= 0:
            return None

        exit_notional = cur_price * qty
        if notional_exceeds_equity_guard(exit_notional, equity_before):
            log.warning("SELL rejected #%s notional exceeds equity guard", pos_id)
            return None

        exit_costs = compute_transaction_costs(exit_notional, chain)
        entry_fees = float(pos.get("entry_fees", 0))
        if exit_costs.total / max(exit_notional, 1e-9) > MAX_FEE_TO_NOTIONAL_PCT:
            log.warning("SELL rejected #%s fee/notional anomaly", pos_id)
            return None

        gross_pnl = (cur_price - entry) * qty
        total_fees = entry_fees + exit_costs.total
        realized_pnl = gross_pnl - total_fees
        cost_basis = entry * qty + entry_fees
        net_roi_pct = realized_pnl / cost_basis if cost_basis > 0 else 0.0
        realized_pnl_pct = (cur_price - entry) / entry if entry > 0 else 0.0

        resolved_close_reason = str(close_reason or reason_code or "user_exit").strip() or "user_exit"
        resolved_close_note = str(close_note or "").strip()
        resolved_closed_by = str(closed_by or "user_manual").strip() or "user_manual"
        is_manual_close = resolved_closed_by == "user_manual"

        # AE13I Smoke Addendum (Part A): close_freshness.classify_manual_close_freshness
        # is the single, hard-guard source of truth for whether this close used a
        # genuinely fresh provider price. It CANNOT be bypassed by a caller-supplied
        # close_freshness_status/close_used_fallback_price claiming "fresh" — those
        # legacy parameters are accepted for backward compatibility but are no longer
        # used to derive freshness themselves.
        from app.ae13b_product.close_freshness import classify_manual_close_freshness

        resolved_price_source = close_price_source or resolution.source
        resolved_price_age = close_price_age_seconds
        if resolved_price_age is None:
            ts = _parse_iso_ts(resolution.price_timestamp)
            if ts is not None:
                resolved_price_age = max(
                    0.0, (datetime.now(timezone.utc) - ts).total_seconds()
                )

        freshness = classify_manual_close_freshness(
            close_price=cur_price,
            price_timestamp=resolution.price_timestamp,
            close_price_source=resolved_price_source,
            close_price_age_seconds=resolved_price_age,
            freshness_threshold_seconds=MAX_FRESH_MARK_AGE_SECONDS,
            warning_shown=manual_close_warning_shown,
        )
        resolved_freshness_status = freshness["close_freshness_status"]
        used_fallback = freshness["close_used_fallback_price"]
        manual_close_warning_shown = freshness["manual_close_warning_shown"]

        # AE13I: manual closes using a fallback/stale price get a distinct
        # reason_code so activity feeds and Virtual Ledger surface the caveat.
        resolved_reason_code = reason_code
        if is_manual_close and resolved_freshness_status == "unknown_or_fallback":
            resolved_reason_code = freshness["reason_code"]
        elif is_manual_close and reason_code in (None, "", "MANUAL"):
            resolved_reason_code = freshness["reason_code"] or "MANUAL_SELL"

        closed = {
            **pos,
            "status": "CLOSED",
            "close_price": cur_price,
            "close_price_source": resolved_price_source,
            "gross_pnl": round(gross_pnl, 6),
            "realized_pnl": round(realized_pnl, 6),
            "realized_pnl_usd": round(realized_pnl, 6),
            "realized_pnl_pct": round(realized_pnl_pct, 6),
            "net_roi_pct": round(net_roi_pct, 6),
            "exit_fees": exit_costs.total,
            "fees": round(total_fees, 6),
            "closed_at": datetime.now(timezone.utc).isoformat(),
            "fill_price_source": resolution.source,
            "closed_by": resolved_closed_by,
            "close_reason": resolved_close_reason,
            "close_note": resolved_close_note,
            "reason_code": resolved_reason_code,
            # AE13I close-freshness / manual-close disclosure fields
            "manual_close": is_manual_close,
            "close_price_age_seconds": resolved_price_age,
            "close_freshness_status": resolved_freshness_status,
            "close_used_fallback_price": bool(used_fallback),
            "manual_close_warning_shown": bool(manual_close_warning_shown),
            "paper_demo_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
            "trade_authority": "PAPER_DEMO_ONLY",
        }

        positions.pop(idx)
        self._state["open_positions"] = positions
        self._state["closed_trades"] = int(self._state.get("closed_trades", 0)) + 1
        self._state["total_net_pnl"] = round(
            float(self._state.get("total_net_pnl", 0)) + realized_pnl, 6
        )
        self._state["cash_usd"] = round(
            float(self._state.get("cash_usd", 0)) + exit_notional - exit_costs.total, 6
        )
        self._record_fees(exit_costs.swap_fee, exit_costs.priority_fee)
        self._save_state()

        self._append_trade_row({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "position_id": pos_id,
            "symbol": pos["symbol"],
            "chain": chain,
            "side": "sell",
            "quantity": round(qty, 10),
            "fill_price": cur_price,
            "notional_usd": round(exit_notional, 6),
            "swap_fee": exit_costs.swap_fee,
            "priority_fee": exit_costs.priority_fee,
            "total_fees": round(total_fees, 6),
            "gross_pnl": round(gross_pnl, 6),
            "realized_pnl": round(realized_pnl, 6),
            "net_roi_pct": round(net_roi_pct, 6),
            "cluster_label": pos.get("cluster_label", ""),
            "reason_code": resolved_reason_code,
            "coin_id": pos.get("coin_id"),
            "pair_address": pos.get("pair_address"),
            "decision_ref_id": pos.get("decision_ref_id"),
            "fill_price_source": resolution.source,
            "market_price_usd": resolution.market_price_usd,
            "price_timestamp": resolution.price_timestamp,
            "cash_before": round(cash_before, 6),
            "equity_before": round(equity_before, 6),
            "notional_requested": round(exit_notional, 6),
            "notional_executed": round(exit_notional, 6),
            "rejection_reason": "",
            "strategy_lane": pos.get("strategy_lane"),
            "preset_id": pos.get("risk_mode") or pos.get("preset_id"),
            "closed_by": resolved_closed_by,
            "close_reason": resolved_close_reason,
            "close_note": resolved_close_note,
            "paper_demo_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
            "event_type": "MANUAL_CLOSE" if resolved_closed_by == "user_manual" else "CLOSE",
            "manual_close": is_manual_close,
            "close_price_age_seconds": resolved_price_age,
            "close_freshness_status": resolved_freshness_status,
            "close_used_fallback_price": bool(used_fallback),
            "manual_close_warning_shown": bool(manual_close_warning_shown),
            "close_price_source": resolved_price_source,
        })

        # AE13I: create the reentry cooldown block after a successful close.
        # Manual user closes get a longer (1h) cooldown; system/bot closes get
        # a short (5min) cooldown so the same pair cannot churn immediately.
        try:
            from app.ae13b_product.reentry_blocks import (
                add_manual_close_block,
                add_system_close_block,
            )

            if is_manual_close:
                add_manual_close_block(closed, resolved_close_reason, duration_seconds=3600)
            else:
                add_system_close_block(closed, resolved_close_reason, duration_seconds=300)
        except Exception:
            log.exception("reentry block creation failed for closed position #%s", pos_id)

        return closed


_paper_trader: PaperTrader | None = None


def get_paper_trader() -> PaperTrader:
    global _paper_trader
    if _paper_trader is None:
        _paper_trader = PaperTrader()
    return _paper_trader


# BEGIN MANUAL_AUTO_BUY_SCHEMA_CANONICALIZATION_V2
def _manual_auto_buy_first_nonempty_v2(*vals):
    for v in vals:
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return ""


def _manual_auto_buy_parse_chain_pair_from_url_v2(v):
    import re as _re
    s = str(v or "").strip()
    m = _re.search(r"dexscreener\.com/([^/\s]+)/([^?\s#]+)", s, flags=_re.I)
    if not m:
        return "", ""
    return m.group(1).lower(), m.group(2)


def _manual_auto_buy_num_v2(*vals):
    for v in vals:
        try:
            if v is None:
                continue
            s = str(v).strip()
            if not s or s.upper() in {"N/A", "NA", "NONE", "NULL", "UNAVAILABLE"}:
                continue
            x = float(s)
            if x > 0:
                return x
        except Exception:
            pass
    return None


def _manual_auto_buy_normalize_position_identity_v2(target, source=None):
    """
    Ensure auto-buy/demo positions have the same canonical identity fields
    as manual demo candidate buys. This prevents portfolio mark-price from
    downgrading valid pair-address positions into legacy orphan positions.
    """
    from datetime import datetime as _dt, timezone as _tz

    if not isinstance(target, dict):
        return target

    if not isinstance(source, dict):
        source = {}

    urls = [
        target.get("canonical_market_identity"),
        target.get("provider_pair_url_exact"),
        target.get("normalized_provider_pair_url_key"),
        target.get("open_chart_url"),
        target.get("mark_price_lookup_key"),
        source.get("canonical_market_identity"),
        source.get("provider_pair_url_exact"),
        source.get("normalized_provider_pair_url_key"),
        source.get("open_chart_url"),
        source.get("url"),
        source.get("pair_url"),
    ]

    url_chain, url_pair = "", ""
    for u in urls:
        c, p = _manual_auto_buy_parse_chain_pair_from_url_v2(u)
        if c and p:
            url_chain, url_pair = c, p
            break

    chain = _manual_auto_buy_first_nonempty_v2(
        url_chain,
        target.get("chain"),
        target.get("network"),
        source.get("chain"),
        source.get("network"),
        source.get("chainId"),
        source.get("chain_id"),
    ).lower()

    pair = _manual_auto_buy_first_nonempty_v2(
        url_pair,
        target.get("pair_address"),
        target.get("pair_address_derived"),
        target.get("provider_pair_url_final_segment_exact"),
        source.get("pair_address"),
        source.get("pairAddress"),
        source.get("provider_pair_address"),
        source.get("resolved_pair_address"),
    )

    if not chain or not pair:
        return target

    canonical_url = f"https://dexscreener.com/{chain}/{pair}"

    price = _manual_auto_buy_num_v2(
        target.get("current_price"),
        target.get("current_price_usd"),
        target.get("mark_price_usd"),
        target.get("latest_price"),
        source.get("current_price"),
        source.get("current_price_usd"),
        source.get("mark_price_usd"),
        source.get("price_usd"),
        source.get("priceUsd"),
        source.get("price"),
        target.get("entry_price"),
        target.get("fill_price"),
    )

    fields = {
        "canonical_market_identity": canonical_url,
        "provider_pair_url_exact": canonical_url,
        "normalized_provider_pair_url_key": canonical_url,
        "open_chart_url": canonical_url,
        "canonical_market_identity_type": "PROVIDER_URL",
        "provider_pair_url_final_segment_exact": pair,
        "pair_address": pair,
        "pair_address_derived": pair,
        "mark_price_lookup_key": canonical_url,
        "price_source_key": canonical_url,
        "identity_repair_method": "AUTO_BUY_SCHEMA_CANONICALIZED_TO_MANUAL_DEMO_SCHEMA",
        "identity_repaired_at": _dt.now(_tz.utc).isoformat(),
    }

    if price is not None:
        fields.update({
            "current_price": price,
            "current_price_numeric": price,
            "current_price_usd": price,
            "mark_price_usd": price,
            "latest_price": price,
            "current_price_status": "PRICE_OK_CANONICAL_IDENTITY_READY",
            "mark_price_status": "PRICE_OK_CANONICAL_IDENTITY_READY",
            "mark_price_lookup_status": "PRICE_OK_CANONICAL_IDENTITY_READY",
            "mark_price_unavailable_reason": "",
            "price_resolution_failure_reason": "",
            "price_status_detail": "price alias available and canonical identity repaired",
        })

    for k, v in fields.items():
        target[k] = v

    snap = target.get("entry_continuity_snapshot")
    if not isinstance(snap, dict):
        snap = {}
        target["entry_continuity_snapshot"] = snap

    snap.update({
        "position_id": target.get("id") or target.get("position_id"),
        "chain": chain,
        "provider_pair_url_exact": canonical_url,
        "normalized_provider_pair_url_key": canonical_url,
        "canonical_market_identity": canonical_url,
        "canonical_market_identity_type": "PROVIDER_URL",
        "provider_pair_url_final_segment_exact": pair,
        "entry_provider_resolution_status": "RESOLVED",
        "entry_trade_readiness_status": "PAPER_ELIGIBLE",
        "entry_market_data_status": "MARKET_DATA_READY",
        "provenance": "auto_buy_schema_identity_repair_runtime",
    })

    return target


def _manual_auto_buy_save_state_v2(self):
    for name in ("_save_state", "save_state", "_persist_state", "persist_state"):
        fn = getattr(self, name, None)
        if callable(fn):
            try:
                fn()
                return True
            except TypeError:
                pass
            except Exception:
                return False

    try:
        state = getattr(self, "_state", None) or getattr(self, "state", None)
        if isinstance(state, dict):
            STATE_PATH.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
            return True
    except Exception:
        pass

    return False


if "PaperTrader" in globals() and not getattr(PaperTrader, "_manual_auto_buy_schema_v2_patched", False):
    _manual_auto_buy_original_open_position_v2 = PaperTrader.open_position

    def _manual_auto_buy_open_position_wrapper_v2(self, *args, **kwargs):
        source_coin = None

        if args and isinstance(args[0], dict):
            source_coin = args[0]
            _manual_auto_buy_normalize_position_identity_v2(source_coin, source_coin)
        elif isinstance(kwargs.get("coin"), dict):
            source_coin = kwargs.get("coin")
            _manual_auto_buy_normalize_position_identity_v2(source_coin, source_coin)

        pos = _manual_auto_buy_original_open_position_v2(self, *args, **kwargs)

        if isinstance(pos, dict):
            _manual_auto_buy_normalize_position_identity_v2(pos, source_coin or {})
            _manual_auto_buy_save_state_v2(self)

        return pos

    PaperTrader.open_position = _manual_auto_buy_open_position_wrapper_v2
    PaperTrader._manual_auto_buy_schema_v2_patched = True
# END MANUAL_AUTO_BUY_SCHEMA_CANONICALIZATION_V2

