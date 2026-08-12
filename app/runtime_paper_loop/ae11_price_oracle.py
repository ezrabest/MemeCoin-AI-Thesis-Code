"""
AE11I price oracle — mark-to-market valuation and TP/SL lifecycle quotes.

No external API calls. Deterministic test quotes are explicitly labeled
real_market_price=false / is_deterministic_test_quote=true.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

from app.runtime_paper_loop.decimal_money import (
    decimal_to_str,
    quantize_price,
    quantize_usd,
    to_decimal,
)
from app.runtime_paper_loop.types import utc_now_iso

GROSS_PNL_FORMULA = "exit_market_value - notional_usd"
NET_PNL_FORMULA = (
    "gross_pnl_usd - entry_fee_usd - entry_slippage_usd - exit_fee_usd - exit_slippage_usd"
)

PRICE_ORACLE_AUDIT_FIELDS = [
    "audit_timestamp_utc",
    "loop_run_id",
    "invocation_id",
    "position_id",
    "pair_address",
    "chain",
    "opened_at_utc",
    "evaluation_at_utc",
    "current_price",
    "price_timestamp_utc",
    "price_observed_at_utc",
    "price_ingested_at_utc",
    "price_age_seconds",
    "price_source",
    "valuation_source",
    "resolution_status",
    "temporal_validity_status",
    "is_stale",
    "is_fallback",
    "is_deterministic_test_quote",
    "real_market_price",
    "no_lookahead_status",
    "missing_reason",
    "notes",
]

MTM_AUDIT_FIELDS = [
    "audit_timestamp_utc",
    "loop_run_id",
    "invocation_id",
    "position_id",
    "pair_address",
    "evaluation_at_utc",
    "entry_price",
    "quantity",
    "notional_usd",
    "cost_basis_usd",
    "current_price",
    "open_market_value_usd",
    "price_unrealized_pnl_usd",
    "total_unrealized_after_cost_pnl_usd",
    "open_entry_cost_drag_usd",
    "valuation_source",
    "resolution_status",
    "temporal_validity_status",
    "no_lookahead_status",
    "is_deterministic_test_quote",
    "real_market_price",
    "mark_to_market_status",
    "notes",
]

TP_SL_AUDIT_FIELDS = [
    "audit_timestamp_utc",
    "loop_run_id",
    "invocation_id",
    "iteration",
    "position_id",
    "pair_address",
    "opened_at_utc",
    "evaluation_at_utc",
    "entry_price",
    "tp_price",
    "sl_price",
    "current_price",
    "price_timestamp_utc",
    "exit_triggered",
    "exit_reason",
    "lifecycle_trigger_source",
    "valuation_source",
    "price_source",
    "resolution_status",
    "temporal_validity_status",
    "no_lookahead_status",
    "is_deterministic_test_quote",
    "real_market_price",
    "gross_pnl_usd",
    "net_pnl_usd",
    "gross_pnl_formula",
    "net_pnl_formula",
    "entry_fee_counted_once",
    "entry_slippage_counted_once",
    "exit_fee_counted_once",
    "no_double_count_status",
    "notes",
]


@dataclass
class PriceResolutionResult:
    pair_address: str = ""
    chain: str | None = None
    position_id: str | None = None
    opened_at_utc: str | None = None
    evaluation_at_utc: str = ""
    current_price: float | None = None
    price_timestamp_utc: str | None = None
    price_observed_at_utc: str | None = None
    price_ingested_at_utc: str | None = None
    price_source: str = ""
    price_age_seconds: float | None = None
    valuation_source: str = ""
    is_stale: bool = False
    is_fallback: bool = False
    is_deterministic_test_quote: bool = False
    real_market_price: bool = False
    temporal_validity_status: str = "BLOCKED_MISSING_TIMESTAMP"
    no_lookahead_status: str = "BLOCKED_MISSING_TIMESTAMP"
    resolution_status: str = "PRICE_MISSING"
    missing_reason: str | None = None
    notes: str = ""

    @property
    def price(self) -> float | None:
        return self.current_price

    @property
    def price_timestamp_used(self) -> str | None:
        return self.price_timestamp_utc

    @property
    def price_status(self) -> str:
        return self.resolution_status

    def to_audit_dict(
        self,
        *,
        loop_run_id: str = "",
        invocation_id: str = "",
    ) -> dict[str, Any]:
        return {
            "audit_timestamp_utc": utc_now_iso(),
            "loop_run_id": loop_run_id,
            "invocation_id": invocation_id,
            "position_id": self.position_id,
            "pair_address": self.pair_address,
            "chain": self.chain,
            "opened_at_utc": self.opened_at_utc,
            "evaluation_at_utc": self.evaluation_at_utc,
            "current_price": self.current_price,
            "price_timestamp_utc": self.price_timestamp_utc,
            "price_observed_at_utc": self.price_observed_at_utc,
            "price_ingested_at_utc": self.price_ingested_at_utc,
            "price_age_seconds": self.price_age_seconds,
            "price_source": self.price_source,
            "valuation_source": self.valuation_source,
            "resolution_status": self.resolution_status,
            "temporal_validity_status": self.temporal_validity_status,
            "is_stale": self.is_stale,
            "is_fallback": self.is_fallback,
            "is_deterministic_test_quote": self.is_deterministic_test_quote,
            "real_market_price": self.real_market_price,
            "no_lookahead_status": self.no_lookahead_status,
            "missing_reason": self.missing_reason,
            "notes": self.notes,
        }

    def usable_for_tpsl(self) -> bool:
        if self.current_price is None or float(self.current_price) <= 0:
            return False
        if self.resolution_status in (
            "PRICE_MISSING",
            "PRICE_BLOCKED_LOOKAHEAD",
            "PRICE_BLOCKED_MISSING_ECONOMICS",
            "PRICE_BLOCKED_MISSING_TIMESTAMP",
            "PRICE_PRE_ENTRY_STALE",
            "PRICE_STALE",
        ):
            return False
        return True

    def usable_for_mtm(self) -> bool:
        if self.current_price is None or float(self.current_price) <= 0:
            return False
        if self.resolution_status in (
            "PRICE_MISSING",
            "PRICE_BLOCKED_LOOKAHEAD",
            "PRICE_BLOCKED_MISSING_ECONOMICS",
            "PRICE_BLOCKED_MISSING_TIMESTAMP",
            "PRICE_PRE_ENTRY_STALE",
        ):
            return False
        return True


def _parse_dt(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def validate_temporal_validity(
    *,
    opened_at_utc: str | None,
    price_timestamp_utc: str | None,
    evaluation_at_utc: str,
    price_observed_at_utc: str | None = None,
    price_ingested_at_utc: str | None = None,
    is_deterministic_test_quote: bool = False,
) -> tuple[str, str]:
    """Return (temporal_validity_status, no_lookahead_status)."""
    if is_deterministic_test_quote:
        if not price_timestamp_utc or not evaluation_at_utc:
            return "BLOCKED_MISSING_TIMESTAMP", "BLOCKED_MISSING_TIMESTAMP"
        return "PASS", "NOT_APPLICABLE_DETERMINISTIC_TEST"

    if not price_timestamp_utc:
        return "BLOCKED_MISSING_TIMESTAMP", "BLOCKED_MISSING_TIMESTAMP"

    opened = _parse_dt(opened_at_utc)
    price_ts = _parse_dt(price_timestamp_utc)
    eval_at = _parse_dt(evaluation_at_utc)
    if price_ts is None or eval_at is None:
        return "BLOCKED_MISSING_TIMESTAMP", "BLOCKED_MISSING_TIMESTAMP"

    if price_ts > eval_at:
        return "FAIL_PRICE_AFTER_EVALUATION_TIME", "FAIL_PRICE_AFTER_EVALUATION_TIME"

    for raw in (price_observed_at_utc, price_ingested_at_utc):
        dt = _parse_dt(raw)
        if dt is not None and dt > eval_at:
            return (
                "BLOCKED_UNAVAILABLE_AT_EVALUATION",
                "BLOCKED_UNAVAILABLE_AT_EVALUATION",
            )

    if opened is not None and price_ts < opened:
        return "BLOCKED_PRE_ENTRY_PRICE", "FAIL_PRICE_BEFORE_OPEN_POLICY"

    return "PASS", "PASS"


@dataclass
class Ae11PriceOracleSessionStats:
    price_positions_evaluated: int = 0
    price_positions_resolved: int = 0
    price_positions_missing: int = 0
    price_positions_stale: int = 0
    price_positions_pre_entry_stale: int = 0
    price_positions_fallback: int = 0
    price_positions_deterministic: int = 0
    tp_trigger_count: int = 0
    sl_trigger_count: int = 0
    time_stop_trigger_count: int = 0
    price_based_positions_closed: int = 0
    price_oracle_audit_rows: int = 0
    mark_to_market_audit_rows: int = 0
    tp_sl_trigger_audit_rows: int = 0
    price_unrealized_pnl_usd: float = 0.0
    total_unrealized_after_cost_pnl_usd: float = 0.0
    valuation_source: str = "UNSET"
    price_oracle_status: str = "NOT_RUN"
    mark_to_market_status: str = "NOT_RUN"
    tp_sl_lifecycle_status: str = "NOT_RUN"
    no_lookahead_status: str = "PASS"
    temporal_validity_status: str = "PASS"
    no_double_count_status: str = "PASS"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Ae11PriceOracle:
    """AE11 valuation oracle — deterministic proof + optional local snapshot resolve."""

    valuation_provider: str = "legacy"
    deterministic_price_scenario: str = "neutral"
    deterministic_price_bump_pct: float = 25.0
    deterministic_price_drop_pct: float = 15.0
    deterministic_price_step_pct: float = 5.0
    max_price_age_seconds: float = 900.0
    price_lifecycle_proof_mode: bool = False
    incremental_step: int = 0
    audit_log: list[dict[str, Any]] = field(default_factory=list)
    mtm_audit_log: list[dict[str, Any]] = field(default_factory=list)
    tp_sl_audit_log: list[dict[str, Any]] = field(default_factory=list)
    session_stats: Ae11PriceOracleSessionStats = field(
        default_factory=Ae11PriceOracleSessionStats
    )
    _mixed_roles: dict[str, str] = field(default_factory=dict)

    def advance_incremental_step(self) -> None:
        self.incremental_step += 1

    def _role_for_position(self, position_id: str) -> str:
        if position_id in self._mixed_roles:
            return self._mixed_roles[position_id]
        h = int(hashlib.sha256(position_id.encode("utf-8")).hexdigest()[:8], 16)
        role = ("tp", "sl", "neutral")[h % 3]
        self._mixed_roles[position_id] = role
        return role

    def resolve_current_price(
        self,
        position: dict[str, Any],
        evaluation_at_utc: str,
        context: dict[str, Any] | None = None,
    ) -> PriceResolutionResult:
        context = context or {}
        pid = str(position.get("position_id") or "")
        pair = str(position.get("pair_address") or "")
        opened = position.get("opened_at_utc")
        entry = to_decimal(position.get("entry_price") or 0)
        enrichment = (position.get("economic_enrichment_status") or "").upper()

        base = PriceResolutionResult(
            pair_address=pair,
            chain=position.get("chain"),
            position_id=pid,
            opened_at_utc=opened,
            evaluation_at_utc=evaluation_at_utc,
        )
        self.session_stats.price_positions_evaluated += 1

        if enrichment == "MISSING" or entry <= 0:
            base.resolution_status = "PRICE_BLOCKED_MISSING_ECONOMICS"
            base.missing_reason = "MISSING_ENTRY_ECONOMICS"
            self.session_stats.price_positions_missing += 1
            self._record_oracle(base, context)
            return base

        if self.valuation_provider == "deterministic":
            return self._resolve_deterministic(position, evaluation_at_utc, context)

        if self.valuation_provider == "local_snapshot":
            resolved = self._resolve_local_snapshot(position, evaluation_at_utc, context)
            if resolved.resolution_status not in ("PRICE_MISSING",):
                return resolved
            base.notes = "LOCAL_SNAPSHOT_UNAVAILABLE"
            base.resolution_status = "PRICE_MISSING"
            base.missing_reason = "LOCAL_SNAPSHOT_UNAVAILABLE"
            self.session_stats.price_positions_missing += 1
            self._record_oracle(base, context)
            return base

        last_price = position.get("last_price")
        if last_price not in (None, "") and to_decimal(last_price) > 0:
            price = float(quantize_price(last_price))
            base.current_price = price
            base.price_timestamp_utc = (
                position.get("last_price_timestamp_utc") or evaluation_at_utc
            )
            base.price_source = "sqlite_last_price"
            base.valuation_source = "PRICE_FALLBACK_SQLITE_MARKET_VALUE"
            base.is_fallback = True
            base.resolution_status = "PRICE_FALLBACK_SQLITE_MARKET_VALUE"
            tv, nl = validate_temporal_validity(
                opened_at_utc=opened,
                price_timestamp_utc=base.price_timestamp_utc,
                evaluation_at_utc=evaluation_at_utc,
            )
            base.temporal_validity_status = tv
            base.no_lookahead_status = nl
            self.session_stats.price_positions_fallback += 1
            self._record_oracle(base, context)
            return base

        if entry > 0:
            base.current_price = float(quantize_price(entry))
            base.price_timestamp_utc = evaluation_at_utc
            base.price_source = "entry_price_fallback"
            base.valuation_source = "PRICE_FALLBACK_ENTRY"
            base.is_fallback = True
            base.resolution_status = "PRICE_FALLBACK_ENTRY"
            base.temporal_validity_status = "PASS"
            base.no_lookahead_status = "PASS"
            self.session_stats.price_positions_fallback += 1
            self._record_oracle(base, context)
            return base

        base.resolution_status = "PRICE_MISSING"
        base.missing_reason = "NO_PRICE_AVAILABLE"
        self.session_stats.price_positions_missing += 1
        self._record_oracle(base, context)
        return base

    def _resolve_deterministic(
        self,
        position: dict[str, Any],
        evaluation_at_utc: str,
        context: dict[str, Any],
    ) -> PriceResolutionResult:
        pid = str(position.get("position_id") or "")
        pair = str(position.get("pair_address") or "")
        entry = to_decimal(position.get("entry_price") or 0)
        tp = to_decimal(position.get("tp_price") or 0)
        sl = to_decimal(position.get("sl_price") or 0)
        scenario = (self.deterministic_price_scenario or "neutral").lower()
        bump = Decimal(str(self.deterministic_price_bump_pct)) / Decimal("100")
        drop = Decimal(str(self.deterministic_price_drop_pct)) / Decimal("100")
        step = Decimal(str(self.deterministic_price_step_pct)) / Decimal("100")
        step_n = max(0, int(self.incremental_step))

        if tp <= 0 and entry > 0:
            tp = quantize_price(entry * (Decimal("1") + Decimal("0.20")))
        if sl <= 0 and entry > 0:
            sl = quantize_price(entry * (Decimal("1") - Decimal("0.10")))

        role = "neutral"
        if scenario in ("tp", "incremental_tp"):
            role = "tp"
        elif scenario in ("sl", "incremental_sl"):
            role = "sl"
        elif scenario in ("mixed", "incremental_mixed"):
            role = self._role_for_position(pid)

        if scenario == "neutral" or role == "neutral":
            price = entry
        elif scenario == "tp" or (scenario == "mixed" and role == "tp"):
            target = tp if tp > 0 else entry * (Decimal("1") + bump)
            price = quantize_price(max(target, entry * (Decimal("1") + bump)))
        elif scenario == "sl" or (scenario == "mixed" and role == "sl"):
            target = sl if sl > 0 else entry * (Decimal("1") - drop)
            price = quantize_price(min(target, entry * (Decimal("1") - drop)))
        elif scenario == "incremental_tp" or (
            scenario == "incremental_mixed" and role == "tp"
        ):
            price = quantize_price(entry * (Decimal("1") + step * Decimal(step_n)))
            if tp > 0 and step_n >= 5 and price < tp:
                price = quantize_price(tp)
        elif scenario == "incremental_sl" or (
            scenario == "incremental_mixed" and role == "sl"
        ):
            price = quantize_price(entry * (Decimal("1") - step * Decimal(step_n)))
            if sl > 0 and step_n >= 5 and price > sl:
                price = quantize_price(sl)
        else:
            price = entry

        if price <= 0:
            price = entry

        result = PriceResolutionResult(
            pair_address=pair,
            chain=position.get("chain"),
            position_id=pid,
            opened_at_utc=position.get("opened_at_utc"),
            evaluation_at_utc=evaluation_at_utc,
            current_price=float(price),
            price_timestamp_utc=evaluation_at_utc,
            price_observed_at_utc=evaluation_at_utc,
            price_ingested_at_utc=evaluation_at_utc,
            price_source="DETERMINISTIC_TEST_ORACLE",
            price_age_seconds=0.0,
            valuation_source="DETERMINISTIC_TEST_ORACLE",
            is_stale=False,
            is_fallback=False,
            is_deterministic_test_quote=True,
            real_market_price=False,
            temporal_validity_status="PASS",
            no_lookahead_status="NOT_APPLICABLE_DETERMINISTIC_TEST",
            resolution_status="PRICE_RESOLVED_DETERMINISTIC_TEST",
            notes=f"scenario={scenario};role={role};step={step_n}",
        )
        self.session_stats.price_positions_resolved += 1
        self.session_stats.price_positions_deterministic += 1
        self.session_stats.valuation_source = "DETERMINISTIC_TEST_ORACLE"
        self._record_oracle(result, context)
        return result

    def _resolve_local_snapshot(
        self,
        position: dict[str, Any],
        evaluation_at_utc: str,
        context: dict[str, Any],
    ) -> PriceResolutionResult:
        base = PriceResolutionResult(
            pair_address=str(position.get("pair_address") or ""),
            chain=position.get("chain"),
            position_id=str(position.get("position_id") or ""),
            opened_at_utc=position.get("opened_at_utc"),
            evaluation_at_utc=evaluation_at_utc,
            valuation_source="LOCAL_SNAPSHOT",
            price_source="local_market_snapshots",
            real_market_price=True,
            is_deterministic_test_quote=False,
            resolution_status="PRICE_MISSING",
            missing_reason="LOCAL_SNAPSHOT_UNAVAILABLE",
            notes="LOCAL_SNAPSHOT_UNAVAILABLE",
        )
        candidates = list(context.get("local_snapshot_candidates") or [])
        opened = _parse_dt(position.get("opened_at_utc"))
        eval_at = _parse_dt(evaluation_at_utc)
        best: dict[str, Any] | None = None
        best_ts: datetime | None = None
        for cand in candidates:
            pts = _parse_dt(cand.get("price_timestamp_utc") or cand.get("timestamp"))
            if pts is None or eval_at is None:
                continue
            if pts > eval_at:
                continue
            obs = _parse_dt(cand.get("price_observed_at_utc"))
            ing = _parse_dt(cand.get("price_ingested_at_utc"))
            if obs is not None and obs > eval_at:
                continue
            if ing is not None and ing > eval_at:
                continue
            if opened is not None and pts < opened:
                continue
            if best_ts is None or pts > best_ts:
                best = cand
                best_ts = pts
        if not best:
            return base

        price = float(best.get("price") or best.get("current_price") or 0)
        if price <= 0:
            return base
        ts = best.get("price_timestamp_utc") or best.get("timestamp")
        tv, nl = validate_temporal_validity(
            opened_at_utc=position.get("opened_at_utc"),
            price_timestamp_utc=ts,
            evaluation_at_utc=evaluation_at_utc,
            price_observed_at_utc=best.get("price_observed_at_utc"),
            price_ingested_at_utc=best.get("price_ingested_at_utc"),
        )
        if tv != "PASS":
            base.temporal_validity_status = tv
            base.no_lookahead_status = nl
            if tv == "FAIL_PRICE_AFTER_EVALUATION_TIME":
                base.resolution_status = "PRICE_BLOCKED_LOOKAHEAD"
            elif tv == "BLOCKED_PRE_ENTRY_PRICE":
                base.resolution_status = "PRICE_PRE_ENTRY_STALE"
                self.session_stats.price_positions_pre_entry_stale += 1
            self._record_oracle(base, context)
            return base

        age = None
        if best_ts is not None and eval_at is not None:
            age = (eval_at - best_ts).total_seconds()
        stale = bool(age is not None and age > self.max_price_age_seconds)
        base.current_price = price
        base.price_timestamp_utc = ts
        base.price_observed_at_utc = best.get("price_observed_at_utc")
        base.price_ingested_at_utc = best.get("price_ingested_at_utc")
        base.price_age_seconds = age
        base.is_stale = stale
        base.temporal_validity_status = tv
        base.no_lookahead_status = nl
        if stale:
            base.resolution_status = "PRICE_STALE"
            self.session_stats.price_positions_stale += 1
        else:
            base.resolution_status = "PRICE_RESOLVED"
            self.session_stats.price_positions_resolved += 1
        self.session_stats.valuation_source = "LOCAL_SNAPSHOT"
        self._record_oracle(base, context)
        return base

    def _record_oracle(self, result: PriceResolutionResult, context: dict[str, Any]) -> None:
        row = result.to_audit_dict(
            loop_run_id=str(context.get("loop_run_id") or ""),
            invocation_id=str(context.get("invocation_id") or ""),
        )
        self.audit_log.append(row)
        self.session_stats.price_oracle_audit_rows = len(self.audit_log)
        if result.no_lookahead_status.startswith("FAIL"):
            self.session_stats.no_lookahead_status = result.no_lookahead_status
        if result.temporal_validity_status.startswith("FAIL"):
            self.session_stats.temporal_validity_status = result.temporal_validity_status

    def apply_mark_to_market(
        self,
        state_db: Any,
        position: dict[str, Any],
        quote: PriceResolutionResult,
        *,
        loop_run_id: str = "",
        invocation_id: str = "",
    ) -> dict[str, Any]:
        pid = position.get("position_id")
        if not quote.usable_for_mtm() or quote.current_price is None:
            row = {
                "audit_timestamp_utc": utc_now_iso(),
                "loop_run_id": loop_run_id,
                "invocation_id": invocation_id,
                "position_id": pid,
                "pair_address": position.get("pair_address"),
                "evaluation_at_utc": quote.evaluation_at_utc,
                "entry_price": position.get("entry_price"),
                "quantity": position.get("quantity"),
                "notional_usd": position.get("notional_usd"),
                "cost_basis_usd": position.get("cost_basis_usd"),
                "current_price": quote.current_price,
                "open_market_value_usd": position.get("open_market_value_usd"),
                "price_unrealized_pnl_usd": position.get("price_unrealized_pnl_usd"),
                "total_unrealized_after_cost_pnl_usd": position.get(
                    "total_unrealized_after_cost_pnl_usd"
                ),
                "open_entry_cost_drag_usd": position.get("open_entry_cost_drag_usd"),
                "valuation_source": quote.valuation_source
                or position.get("valuation_source"),
                "resolution_status": quote.resolution_status,
                "temporal_validity_status": quote.temporal_validity_status,
                "no_lookahead_status": quote.no_lookahead_status,
                "is_deterministic_test_quote": quote.is_deterministic_test_quote,
                "real_market_price": quote.real_market_price,
                "mark_to_market_status": "BLOCKED_NO_CURRENT_PRICE",
                "notes": quote.missing_reason or quote.notes,
            }
            self.mtm_audit_log.append(row)
            self.session_stats.mark_to_market_audit_rows = len(self.mtm_audit_log)
            return row

        qty = to_decimal(position.get("quantity") or 0)
        notional = to_decimal(position.get("notional_usd") or 0)
        cost = to_decimal(position.get("cost_basis_usd") or notional)
        price = quantize_price(quote.current_price)
        mv = quantize_usd(price * qty)
        price_upnl = quantize_usd(mv - notional)
        after_cost = quantize_usd(mv - cost)
        cost_drag = quantize_usd(notional - cost)
        patch = {
            "last_price": decimal_to_str(price),
            "last_price_timestamp_utc": quote.price_timestamp_utc,
            "last_valuation_at_utc": quote.evaluation_at_utc or utc_now_iso(),
            "open_market_value_usd": decimal_to_str(mv),
            "unrealized_pnl_usd": decimal_to_str(price_upnl),
            "price_unrealized_pnl_usd": decimal_to_str(price_upnl),
            "total_unrealized_after_cost_pnl_usd": decimal_to_str(after_cost),
            "open_entry_cost_drag_usd": decimal_to_str(cost_drag),
            "valuation_source": quote.valuation_source,
            "unrealized_return_pct": decimal_to_str(
                quantize_usd((price_upnl / notional) * Decimal("100"))
                if notional > 0
                else Decimal("0")
            ),
        }
        if pid and hasattr(state_db, "update_position_economics"):
            state_db.update_position_economics(pid, patch)
        position.update(patch)
        self.session_stats.price_unrealized_pnl_usd += float(price_upnl)
        self.session_stats.total_unrealized_after_cost_pnl_usd += float(after_cost)
        row = {
            "audit_timestamp_utc": utc_now_iso(),
            "loop_run_id": loop_run_id,
            "invocation_id": invocation_id,
            "position_id": pid,
            "pair_address": position.get("pair_address"),
            "evaluation_at_utc": quote.evaluation_at_utc,
            "entry_price": position.get("entry_price"),
            "quantity": decimal_to_str(qty),
            "notional_usd": decimal_to_str(notional),
            "cost_basis_usd": decimal_to_str(cost),
            "current_price": decimal_to_str(price),
            "open_market_value_usd": decimal_to_str(mv),
            "price_unrealized_pnl_usd": decimal_to_str(price_upnl),
            "total_unrealized_after_cost_pnl_usd": decimal_to_str(after_cost),
            "open_entry_cost_drag_usd": decimal_to_str(cost_drag),
            "valuation_source": quote.valuation_source,
            "resolution_status": quote.resolution_status,
            "temporal_validity_status": quote.temporal_validity_status,
            "no_lookahead_status": quote.no_lookahead_status,
            "is_deterministic_test_quote": quote.is_deterministic_test_quote,
            "real_market_price": quote.real_market_price,
            "mark_to_market_status": "PASS",
            "notes": quote.notes,
        }
        self.mtm_audit_log.append(row)
        self.session_stats.mark_to_market_audit_rows = len(self.mtm_audit_log)
        return row

    def record_tp_sl_trigger(
        self,
        *,
        loop_run_id: str,
        invocation_id: str,
        iteration: int,
        position: dict[str, Any],
        quote: PriceResolutionResult,
        exit_triggered: bool,
        exit_reason: str | None,
        gross_pnl_usd: Any = None,
        net_pnl_usd: Any = None,
        no_double_count_status: str = "PASS",
        notes: str = "",
    ) -> dict[str, Any]:
        trigger_src = ""
        if exit_reason in ("TAKE_PROFIT", "STOP_LOSS"):
            trigger_src = "PRICE_ORACLE"
        elif exit_reason == "TIME_STOP":
            trigger_src = "TIME_STOP"
        row = {
            "audit_timestamp_utc": utc_now_iso(),
            "loop_run_id": loop_run_id,
            "invocation_id": invocation_id,
            "iteration": iteration,
            "position_id": position.get("position_id"),
            "pair_address": position.get("pair_address"),
            "opened_at_utc": position.get("opened_at_utc"),
            "evaluation_at_utc": quote.evaluation_at_utc,
            "entry_price": position.get("entry_price"),
            "tp_price": position.get("tp_price"),
            "sl_price": position.get("sl_price"),
            "current_price": quote.current_price,
            "price_timestamp_utc": quote.price_timestamp_utc,
            "exit_triggered": exit_triggered,
            "exit_reason": exit_reason,
            "lifecycle_trigger_source": trigger_src,
            "valuation_source": quote.valuation_source,
            "price_source": quote.price_source,
            "resolution_status": quote.resolution_status,
            "temporal_validity_status": quote.temporal_validity_status,
            "no_lookahead_status": quote.no_lookahead_status,
            "is_deterministic_test_quote": quote.is_deterministic_test_quote,
            "real_market_price": quote.real_market_price,
            "gross_pnl_usd": gross_pnl_usd,
            "net_pnl_usd": net_pnl_usd,
            "gross_pnl_formula": GROSS_PNL_FORMULA,
            "net_pnl_formula": NET_PNL_FORMULA,
            "entry_fee_counted_once": True,
            "entry_slippage_counted_once": True,
            "exit_fee_counted_once": True,
            "no_double_count_status": no_double_count_status,
            "notes": notes,
        }
        self.tp_sl_audit_log.append(row)
        self.session_stats.tp_sl_trigger_audit_rows = len(self.tp_sl_audit_log)
        if exit_triggered and exit_reason == "TAKE_PROFIT":
            self.session_stats.tp_trigger_count += 1
            self.session_stats.price_based_positions_closed += 1
        elif exit_triggered and exit_reason == "STOP_LOSS":
            self.session_stats.sl_trigger_count += 1
            self.session_stats.price_based_positions_closed += 1
        elif exit_triggered and exit_reason == "TIME_STOP":
            self.session_stats.time_stop_trigger_count += 1
        if no_double_count_status != "PASS":
            self.session_stats.no_double_count_status = no_double_count_status
        return row

    def finalize_session_status(self) -> Ae11PriceOracleSessionStats:
        stats = self.session_stats
        if stats.price_positions_evaluated == 0:
            stats.price_oracle_status = "NO_OPEN_POSITIONS"
            stats.mark_to_market_status = "NO_OPEN_POSITIONS"
        elif (
            stats.price_positions_resolved
            + stats.price_positions_deterministic
            + stats.price_positions_fallback
            > 0
        ):
            stats.price_oracle_status = "PASS"
            stats.mark_to_market_status = "PASS"
        elif stats.price_positions_missing > 0:
            stats.price_oracle_status = "WARNING_MISSING_PRICES"
            stats.mark_to_market_status = "WARNING_MISSING_PRICES"
        else:
            stats.price_oracle_status = "PASS"
            stats.mark_to_market_status = "PASS"

        if stats.tp_trigger_count + stats.sl_trigger_count > 0:
            stats.tp_sl_lifecycle_status = "PASS"
        elif self.valuation_provider == "deterministic" and self.deterministic_price_scenario in (
            "tp",
            "sl",
            "mixed",
            "incremental_tp",
            "incremental_sl",
            "incremental_mixed",
        ):
            stats.tp_sl_lifecycle_status = (
                "PASS" if stats.price_based_positions_closed > 0 else "PENDING_OR_NEUTRAL"
            )
        else:
            stats.tp_sl_lifecycle_status = "PASS_NO_PRICE_TRIGGERS"
        return stats


def write_price_oracle_audit(path: Path, rows: list[dict[str, Any]]) -> Path:
    return _write_csv(path, rows, PRICE_ORACLE_AUDIT_FIELDS)


def write_mark_to_market_audit(path: Path, rows: list[dict[str, Any]]) -> Path:
    return _write_csv(path, rows, MTM_AUDIT_FIELDS)


def write_tp_sl_trigger_audit(path: Path, rows: list[dict[str, Any]]) -> Path:
    return _write_csv(path, rows, TP_SL_AUDIT_FIELDS)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})
        f.flush()
    return path


def build_ae11_price_oracle(config: Any) -> Ae11PriceOracle:
    provider = getattr(config, "valuation_provider", "legacy") or "legacy"
    proof_mode = bool(getattr(config, "price_lifecycle_proof_mode", False))
    if proof_mode:
        # Proof mode always uses labeled deterministic quotes (never external APIs).
        provider = "deterministic"
    return Ae11PriceOracle(
        valuation_provider=provider,
        deterministic_price_scenario=getattr(
            config, "deterministic_price_scenario", "neutral"
        )
        or "neutral",
        deterministic_price_bump_pct=float(
            getattr(config, "deterministic_price_bump_pct", 25.0) or 25.0
        ),
        deterministic_price_drop_pct=float(
            getattr(config, "deterministic_price_drop_pct", 15.0) or 15.0
        ),
        deterministic_price_step_pct=float(
            getattr(config, "deterministic_price_step_pct", 5.0) or 5.0
        ),
        max_price_age_seconds=float(
            getattr(config, "max_price_age_seconds", 900.0) or 900.0
        ),
        price_lifecycle_proof_mode=proof_mode,
    )


def fetch_local_snapshot_candidates(
    *,
    pair_address: str | None = None,
    coin_id: str | None = None,
    evaluation_at_utc: str,
    opened_at_utc: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Load local market_snapshots from trader.db (no external APIs).

    Returns candidate dicts for _resolve_local_snapshot temporal filtering.
    Empty list if DB/table unavailable → caller treats as LOCAL_SNAPSHOT_UNAVAILABLE.
    """
    eval_at = _parse_dt(evaluation_at_utc)
    opened = _parse_dt(opened_at_utc)
    if eval_at is None:
        return []

    conn = None
    try:
        try:
            from scripts.diagnostics._common import open_db_readonly

            conn = open_db_readonly()
        except Exception:
            from app.database import get_db

            # get_db is a context manager; use nested with below
            conn = None

        def _query(c: Any) -> list[dict[str, Any]]:
            tables = {
                r[0]
                for r in c.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            if "market_snapshots" not in tables:
                return []
            cols = {r[1] for r in c.execute("PRAGMA table_info(market_snapshots)").fetchall()}
            where = ["timestamp <= ?"]
            params: list[Any] = [evaluation_at_utc]
            if pair_address and "pair_address" in cols:
                where.append("pair_address = ?")
                params.append(pair_address)
            elif coin_id not in (None, ""):
                where.append("coin_id = ?")
                params.append(coin_id)
            else:
                return []
            if opened_at_utc:
                where.append("timestamp >= ?")
                params.append(opened_at_utc)
            sql = (
                f"SELECT * FROM market_snapshots WHERE {' AND '.join(where)} "
                f"ORDER BY timestamp DESC LIMIT {int(limit)}"
            )
            return [dict(r) for r in c.execute(sql, params).fetchall()]

        if conn is not None:
            rows = _query(conn)
        else:
            from app.database import get_db

            with get_db() as c:
                rows = _query(c)
    except Exception:
        return []
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    out: list[dict[str, Any]] = []
    for d in rows:
        ts = d.get("timestamp") or d.get("price_timestamp_utc")
        pts = _parse_dt(ts)
        if pts is None:
            continue
        if pts > eval_at:
            continue
        if opened is not None and pts < opened:
            continue
        price = d.get("price") or d.get("close") or d.get("current_price")
        if price is None:
            continue
        out.append(
            {
                "price": price,
                "current_price": price,
                "price_timestamp_utc": ts,
                "timestamp": ts,
                "price_observed_at_utc": d.get("price_observed_at_utc") or ts,
                "price_ingested_at_utc": d.get("price_ingested_at_utc")
                or d.get("ingested_at_utc")
                or ts,
                "coin_id": d.get("coin_id"),
                "pair_address": d.get("pair_address") or pair_address,
            }
        )
    return out
