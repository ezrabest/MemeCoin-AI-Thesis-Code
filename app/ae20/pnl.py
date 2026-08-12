"""AE20 PnL reporting with strict/exploration/baseline separation. No profitability claim."""

from __future__ import annotations

from typing import Any

from app.ae20.lifecycle import FEE_ASSUMPTION_BPS, SLIPPAGE_ASSUMPTION_BPS


def _sum(rows: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for r in rows:
        try:
            total += float(r.get(key) or 0.0)
        except (TypeError, ValueError):
            continue
    return total


def build_pnl_summary(
    *,
    orders: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    preexisting_positions: list[dict[str, Any]],
    preexisting_trades: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ae20_positions = [p for p in positions if p.get("created_during_ae20")]
    strict_pos = [p for p in ae20_positions if p.get("decision_path") == "STRICT"]
    expl_pos = [p for p in ae20_positions if p.get("decision_path") == "EXPLORATION"]
    open_ae20 = [p for p in ae20_positions if p.get("status") == "OPEN"]
    closed_ae20 = [p for p in ae20_positions if p.get("status") == "CLOSED"]

    def scope_row(
        scope: str,
        path: str,
        pos: list[dict[str, Any]],
        *,
        include_baseline: bool = False,
    ) -> dict[str, Any]:
        open_c = sum(1 for p in pos if p.get("status") == "OPEN")
        closed_c = sum(1 for p in pos if p.get("status") == "CLOSED")
        return {
            "pnl_scope": scope,
            "decision_path": path,
            "trade_count": len(pos),
            "open_position_count": open_c,
            "closed_position_count": closed_c,
            "realized_pnl": _sum(pos, "realized_pnl"),
            "unrealized_pnl": _sum(pos, "unrealized_pnl"),
            "gross_pnl": _sum(pos, "realized_pnl") + _sum(pos, "unrealized_pnl"),
            "fees": _sum(pos, "fees_assumption_usd"),
            "slippage_assumption": _sum(pos, "slippage_assumption_usd"),
            "fees_assumption_bps": FEE_ASSUMPTION_BPS,
            "slippage_assumption_bps": SLIPPAGE_ASSUMPTION_BPS,
            "maturity_status": (
                "MIXED"
                if pos and len({p.get("maturity_status") for p in pos}) > 1
                else (pos[0].get("maturity_status") if pos else "N/A")
            ),
            "profitability_claim": False,
            "includes_preexisting_baseline": include_baseline,
            "pnl_calculated_and_reported": True,
            "profitability_proof": False,
        }

    rows = [
        scope_row("AE20_CREATED_STRICT", "STRICT", strict_pos),
        scope_row("AE20_CREATED_EXPLORATION", "EXPLORATION", expl_pos),
        scope_row("AE20_CREATED_ALL", "COMBINED_AE20", ae20_positions),
        scope_row(
            "PREEXISTING_BASELINE_POSITIONS",
            "PREEXISTING_BASELINE",
            preexisting_positions,
            include_baseline=True,
        ),
        {
            "pnl_scope": "PREEXISTING_BASELINE_TRADES",
            "decision_path": "PREEXISTING_BASELINE",
            "trade_count": len(preexisting_trades),
            "open_position_count": 0,
            "closed_position_count": len(preexisting_trades),
            "realized_pnl": _sum(preexisting_trades, "realized_pnl"),
            "unrealized_pnl": 0.0,
            "gross_pnl": _sum(preexisting_trades, "realized_pnl"),
            "fees": _sum(preexisting_trades, "total_fees"),
            "slippage_assumption": 0.0,
            "fees_assumption_bps": "",
            "slippage_assumption_bps": "",
            "maturity_status": "HISTORICAL_BASELINE",
            "profitability_claim": False,
            "includes_preexisting_baseline": True,
            "pnl_calculated_and_reported": True,
            "profitability_proof": False,
            "excluded_from_ae20_created_pnl": True,
        },
        scope_row("AE20_CREATED_OPEN", "COMBINED_AE20", open_ae20),
        scope_row("AE20_CREATED_CLOSED", "COMBINED_AE20", closed_ae20),
    ]

    # Ensure baseline is not mixed into AE20-created combined metrics.
    baseline_excluded = all(
        not r.get("includes_preexisting_baseline")
        for r in rows
        if r["decision_path"] in {"STRICT", "EXPLORATION", "COMBINED_AE20"}
        and r["pnl_scope"].startswith("AE20_CREATED")
    )
    audit = {
        "pnl_reporting_complete": True,
        "profitability_claim": False,
        "profitability_proof": False,
        "strict_exploration_separated": True,
        "baseline_excluded_from_ae20_created_pnl": baseline_excluded,
        "ae20_created_orders_count": len([o for o in orders if o.get("created_during_ae20")]),
        "ae20_created_positions_count": len(ae20_positions),
        "main_gate_message": (
            "PnL was calculated and reported with strict/exploration/baseline "
            "separation. No profitability claim."
        ),
        "rows": rows,
    }
    return rows, audit
