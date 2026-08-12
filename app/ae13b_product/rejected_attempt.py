"""Structured record of a rejected (risk-guard-blocked) paper trade attempt.

Paper/demo only. A RejectedTradeAttempt is never a filled trade — it exists so
UI / audit / CSV / SQLite consumers can see *why* a candidate did not open a
position, with the same structured RiskGuardResult fields the backend used.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RejectedTradeAttempt:
    """A single rejected/blocked paper trade attempt (never a live order)."""

    event_type: str = "RISK_GUARD_BLOCK"
    timestamp: str = field(default_factory=_utc_now)
    symbol: str | None = None
    pair: str | None = None
    side: str = "BUY"
    chain: str | None = None
    pair_address: str | None = None
    token_contract_address: str | None = None
    coin_id: int | None = None
    strategy_lane: str | None = None
    preset_id: str | None = None
    risk_mode: str | None = None
    fill_price: float | None = None
    quantity: float = 0
    notional_usd: float = 0
    notional_requested: float = 0
    notional_executed: float = 0
    rejection_code: str | None = None
    rejection_reason: str | None = None
    rejection_reasons: list[str] = field(default_factory=list)
    blocking_guards: list[str] = field(default_factory=list)
    risk_guard_passed: bool = False
    paper_demo_only: bool = True
    not_live_approved: bool = True
    not_profitability_evidence: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Explicit keyed serialization - never positional field access."""
        event = self.event_type or "RISK_GUARD_BLOCK"
        return {
            "event_type": event,
            # CSV / portfolio consumers historically key off reason_code
            "reason_code": event,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "pair": self.pair,
            "side": self.side,
            "chain": self.chain,
            "pair_address": self.pair_address,
            "token_contract_address": self.token_contract_address,
            "coin_id": self.coin_id,
            "strategy_lane": self.strategy_lane,
            "preset_id": self.preset_id,
            "risk_mode": self.risk_mode,
            "fill_price": self.fill_price,
            "quantity": self.quantity,
            "notional_usd": self.notional_usd,
            "notional_requested": self.notional_requested,
            "notional_executed": self.notional_executed,
            "rejection_code": self.rejection_code,
            "rejection_reason": self.rejection_reason,
            "rejection_reasons": list(self.rejection_reasons or []),
            "blocking_guards": list(self.blocking_guards or []),
            "risk_guard_passed": self.risk_guard_passed,
            "paper_demo_only": self.paper_demo_only,
            "not_live_approved": self.not_live_approved,
            "not_profitability_evidence": self.not_profitability_evidence,
            "position_id": "",
            "swap_fee": 0,
            "priority_fee": 0,
            "total_fees": 0,
            "gross_pnl": 0,
            "realized_pnl": 0,
            "net_roi_pct": 0,
        }

    @classmethod
    def from_risk_guard(
        cls,
        coin: dict[str, Any],
        risk: dict[str, Any],
        **kwargs: Any,
    ) -> "RejectedTradeAttempt":
        """Build a RejectedTradeAttempt from a candidate coin dict and a
        RiskGuardResult dict (as returned by evaluate_demo_risk_guard).

        Any explicit kwargs override the values derived from coin/risk.
        """
        coin = coin or {}
        risk = risk or {}

        pair_address = (
            kwargs.pop("pair_address", None)
            or coin.get("pair_address")
            or coin.get("contract_address")
        )
        token_contract_address = kwargs.pop("token_contract_address", None) or (
            coin.get("token_contract_address")
            or coin.get("contract_address")
            or coin.get("token_address")
        )
        coin_id_raw = kwargs.pop("coin_id", None)
        if coin_id_raw is None:
            coin_id_raw = coin.get("coin_id") or coin.get("id")
        try:
            coin_id = int(coin_id_raw) if coin_id_raw is not None else None
        except (TypeError, ValueError):
            coin_id = None

        reasons = list(risk.get("rejection_reasons") or risk.get("risk_guard_reasons") or [])
        guards = list(risk.get("blocking_guards") or [])
        rejection_reason = (
            risk.get("rejection_reason")
            or risk.get("risk_guard_reason")
            or (reasons[0] if reasons else "risk_guard_blocked")
        )
        rejection_code = risk.get("rejection_code")

        notional_requested = kwargs.pop("notional_requested", None)
        if notional_requested is None:
            notional_requested = risk.get("requested_notional") or 0

        defaults: dict[str, Any] = {
            "symbol": coin.get("symbol"),
            "pair": pair_address,
            "side": "BUY",
            "chain": coin.get("chain"),
            "pair_address": pair_address,
            "token_contract_address": token_contract_address,
            "coin_id": coin_id,
            "strategy_lane": risk.get("strategy_lane") or coin.get("strategy_lane"),
            "preset_id": risk.get("preset_id"),
            "risk_mode": risk.get("risk_mode") or risk.get("preset_id"),
            "fill_price": None,
            "quantity": 0,
            "notional_usd": 0,
            "notional_requested": notional_requested,
            "notional_executed": 0,
            "rejection_code": rejection_code,
            "rejection_reason": rejection_reason,
            "rejection_reasons": reasons,
            "blocking_guards": guards,
            "risk_guard_passed": bool(risk.get("risk_guard_passed") or risk.get("passed") or False),
        }
        defaults.update(kwargs)
        return cls(**defaults)
