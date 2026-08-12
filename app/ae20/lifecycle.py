"""AE20-local paper/demo lifecycle ledger (does not mutate live paper_state by default)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


FEE_ASSUMPTION_BPS = 30.0  # 0.30% swap fee assumption for reporting
SLIPPAGE_ASSUMPTION_BPS = 25.0  # 0.25% slippage assumption for reporting


def maybe_create_paper_lifecycle(
    decision: dict[str, Any],
    candidate: dict[str, Any],
    *,
    notional_usd: float = 25.0,
) -> dict[str, Any] | None:
    """Create AE20-local order/position/outcome when final decision is paper/demo open.

    Shadow ledger only — does not write live paper_state.json.
    """
    if decision.get("final_paper_demo_decision") != "PAPER_DEMO_OPEN":
        return None
    if decision.get("trade_authority"):
        # Hard refuse — AE20 never has trade authority for live.
        return None
    if decision.get("live_trading_enabled"):
        return None

    try:
        price = float(candidate.get("price_usd"))
    except (TypeError, ValueError):
        price = None
    if price is None or price <= 0:
        decision["skip_reason"] = decision.get("skip_reason") or "MISSING_PRICE"
        decision["final_paper_demo_decision"] = "NO_TRADE"
        return None

    order_id = _new_id("ae20ord")
    position_id = _new_id("ae20pos")
    outcome_id = _new_id("ae20out")
    qty = notional_usd / price
    fee = notional_usd * (FEE_ASSUMPTION_BPS / 10000.0)
    slip = notional_usd * (SLIPPAGE_ASSUMPTION_BPS / 10000.0)
    now = _utc()

    order = {
        "ae20_run_id": decision.get("ae20_run_id"),
        "ae20_cycle_id": decision.get("ae20_cycle_id"),
        "order_id": order_id,
        "ae20_decision_id": decision.get("ae20_decision_id"),
        "candidate_id": decision.get("candidate_id"),
        "provider_pair_url_exact": decision.get("provider_pair_url_exact"),
        "canonical_market_identity": decision.get("canonical_market_identity"),
        "price_source_key": decision.get("price_source_key"),
        "chain": decision.get("chain"),
        "pair_address": decision.get("pair_address"),
        "side": "BUY",
        "notional_usd": notional_usd,
        "fill_price": price,
        "fill_price_source": "clean_forward_canonical_runtime_index",
        "mark_price_source": "clean_forward_canonical_runtime_index",
        "fees_assumption_usd": fee,
        "slippage_assumption_usd": slip,
        "fees_assumption_bps": FEE_ASSUMPTION_BPS,
        "slippage_assumption_bps": SLIPPAGE_ASSUMPTION_BPS,
        "entry_time": now,
        "status": "FILLED_PAPER_DEMO",
        "decision_path": (
            "STRICT"
            if str(decision.get("strict_decision", "")).startswith("STRICT_APPROVED")
            else "EXPLORATION"
        ),
        "created_during_ae20": True,
        "preexisting_baseline": False,
        "trade_authority": False,
        "live_trading_enabled": False,
        "wallet_connected": False,
        "profitability_claim": False,
        "position_id": position_id,
    }
    position = {
        "ae20_run_id": decision.get("ae20_run_id"),
        "ae20_cycle_id": decision.get("ae20_cycle_id"),
        "position_id": position_id,
        "order_id": order_id,
        "ae20_decision_id": decision.get("ae20_decision_id"),
        "candidate_id": decision.get("candidate_id"),
        "provider_pair_url_exact": decision.get("provider_pair_url_exact"),
        "canonical_market_identity": decision.get("canonical_market_identity"),
        "price_source_key": decision.get("price_source_key"),
        "chain": decision.get("chain"),
        "pair_address": decision.get("pair_address"),
        "entry_price": price,
        "mark_price": price,
        "quantity": qty,
        "notional_usd": notional_usd,
        "entry_time": now,
        "status": "OPEN",
        "maturity_status": "NOT_MATURED",
        "decision_path": order["decision_path"],
        "fees_assumption_usd": fee,
        "slippage_assumption_usd": slip,
        "unrealized_pnl": 0.0,
        "realized_pnl": 0.0,
        "created_during_ae20": True,
        "preexisting_baseline": False,
        "outcome_id": outcome_id,
        "trade_authority": False,
        "live_trading_enabled": False,
        "wallet_connected": False,
        "profitability_claim": False,
    }
    outcome = {
        "ae20_run_id": decision.get("ae20_run_id"),
        "ae20_cycle_id": decision.get("ae20_cycle_id"),
        "outcome_id": outcome_id,
        "position_id": position_id,
        "order_id": order_id,
        "ae20_decision_id": decision.get("ae20_decision_id"),
        "candidate_id": decision.get("candidate_id"),
        "provider_pair_url_exact": decision.get("provider_pair_url_exact"),
        "canonical_market_identity": decision.get("canonical_market_identity"),
        "status": "OUTCOME_UNAVAILABLE",
        "maturity_status": "NOT_MATURED",
        "outcome_unavailable_reason": "SMOKE_HORIZON_NOT_ELAPSED",
        "realized_pnl": 0.0,
        "unrealized_pnl": 0.0,
        "decision_path": order["decision_path"],
        "created_during_ae20": True,
        "preexisting_baseline": False,
        "profitability_claim": False,
    }
    decision["order_id"] = order_id
    decision["position_id"] = position_id
    decision["outcome_id"] = outcome_id
    return {"order": order, "position": position, "outcome": outcome}


def audit_lineage(
    decisions: list[dict[str, Any]],
    orders: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
    *,
    preexisting_positions: list[dict[str, Any]],
    preexisting_trades: list[dict[str, Any]],
    preexisting_orders: list[dict[str, Any]],
) -> dict[str, Any]:
    """Orphan checks for AE20-created records only; baseline excluded."""
    ae20_orders = [o for o in orders if o.get("created_during_ae20")]
    ae20_positions = [p for p in positions if p.get("created_during_ae20")]
    ae20_outcomes = [o for o in outcomes if o.get("created_during_ae20")]
    decision_ids = {d.get("ae20_decision_id") for d in decisions if d.get("created_during_ae20")}

    orphan_orders = [
        o for o in ae20_orders if o.get("ae20_decision_id") not in decision_ids
    ]
    order_ids = {o.get("order_id") for o in ae20_orders}
    orphan_positions = [
        p
        for p in ae20_positions
        if p.get("order_id") not in order_ids or p.get("ae20_decision_id") not in decision_ids
    ]
    position_ids = {p.get("position_id") for p in ae20_positions}
    orphan_outcomes = [
        o
        for o in ae20_outcomes
        if o.get("position_id") not in position_ids
        or (not o.get("outcome_unavailable_reason") and o.get("status") not in {"OUTCOME_UNAVAILABLE", "CLOSED", "OPEN_MARK"})
    ]
    # Outcomes must link to position; OUTCOME_UNAVAILABLE with reason is valid.
    orphan_outcomes = [
        o
        for o in ae20_outcomes
        if o.get("position_id") not in position_ids
        or (
            o.get("status") == "OUTCOME_UNAVAILABLE"
            and not o.get("outcome_unavailable_reason")
            and o.get("position_id") not in position_ids
        )
        or o.get("position_id") not in position_ids
    ]

    # Baseline must not appear in AE20-created sets
    baseline_leak_positions = [
        p for p in positions if p.get("preexisting_baseline") and p.get("created_during_ae20")
    ]
    baseline_excluded_from_orphan = True

    passed = (
        len(orphan_orders) == 0
        and len(orphan_positions) == 0
        and len(orphan_outcomes) == 0
        and len(baseline_leak_positions) == 0
    )
    return {
        "lineage_pass": passed,
        "ae20_created_orders": len(ae20_orders),
        "ae20_created_positions": len(ae20_positions),
        "ae20_created_outcomes": len(ae20_outcomes),
        "orphan_orders_count": len(orphan_orders),
        "orphan_positions_count": len(orphan_positions),
        "orphan_outcomes_count": len(orphan_outcomes),
        "preexisting_positions_excluded_count": len(preexisting_positions),
        "preexisting_trades_excluded_count": len(preexisting_trades),
        "preexisting_orders_excluded_count": len(preexisting_orders),
        "baseline_excluded_from_orphan_checks": baseline_excluded_from_orphan,
        "baseline_leak_positions_count": len(baseline_leak_positions),
        "decision_to_order_links": sum(1 for d in decisions if d.get("order_id")),
        "order_to_position_links": sum(1 for o in ae20_orders if o.get("position_id")),
        "position_to_outcome_links": sum(1 for p in ae20_positions if p.get("outcome_id")),
    }
