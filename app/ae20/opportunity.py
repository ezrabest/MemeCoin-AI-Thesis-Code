"""Opportunity capture, missed winners, trade-vs-no-trade, strict-vs-exploration."""

from __future__ import annotations

from typing import Any


def build_opportunity_rows(
    decisions: list[dict[str, Any]],
    outcomes: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    outcome_by_decision = {
        o.get("ae20_decision_id"): o for o in outcomes if o.get("ae20_decision_id")
    }
    skipped: list[dict[str, Any]] = []
    opportunity: list[dict[str, Any]] = []
    missed: list[dict[str, Any]] = []
    trade_vs: list[dict[str, Any]] = []
    strict_vs: list[dict[str, Any]] = []

    for d in decisions:
        traded = bool(d.get("order_id")) and d.get("final_paper_demo_decision") == "PAPER_DEMO_OPEN"
        skip_reason = d.get("skip_reason") or ""
        outcome = outcome_by_decision.get(d.get("ae20_decision_id")) or {}
        matured = (outcome.get("maturity_status") == "MATURED") if outcome else False
        outcome_status = outcome.get("status") if outcome else "NO_OUTCOME_RECORD"

        row = {
            "ae20_run_id": d.get("ae20_run_id"),
            "ae20_cycle_id": d.get("ae20_cycle_id"),
            "ae20_decision_id": d.get("ae20_decision_id"),
            "candidate_id": d.get("candidate_id"),
            "provider_pair_url_exact": d.get("provider_pair_url_exact"),
            "canonical_market_identity": d.get("canonical_market_identity"),
            "traded": traded,
            "non_traded": not traded,
            "skipped": bool(skip_reason) and not traded,
            "skip_reason": skip_reason,
            "strict_decision": d.get("strict_decision"),
            "exploration_decision": d.get("exploration_decision"),
            "final_paper_demo_decision": d.get("final_paper_demo_decision"),
            "outcome_status": outcome_status,
            "maturity_status": outcome.get("maturity_status") if outcome else "N/A",
            "false_positive": False,
            "correct_reject": (not traded) and skip_reason in {
                "AE16_REJECT",
                "GATEKEEPER_BLOCK",
                "RISKGUARD_BLOCK",
                "CLEAN_FORWARD_IDENTITY_INCOMPLETE",
            },
            "missed_winner": False,
            "outcome_unavailable": outcome_status == "OUTCOME_UNAVAILABLE" or not matured,
            "created_during_ae20": True,
            "preexisting_baseline": False,
            "profitability_claim": False,
        }
        opportunity.append(row)

        if row["skipped"]:
            skipped.append(
                {
                    "ae20_decision_id": d.get("ae20_decision_id"),
                    "candidate_id": d.get("candidate_id"),
                    "canonical_market_identity": d.get("canonical_market_identity"),
                    "skip_reason": skip_reason,
                    "strict_decision": d.get("strict_decision"),
                    "exploration_decision": d.get("exploration_decision"),
                    "ae16_status": d.get("ae16_status"),
                    "ae17_status": d.get("ae17_status"),
                    "ae18_status": d.get("ae18_status"),
                    "ae19_status": d.get("ae19_status"),
                }
            )

        # Missed winner only when matured positive outcome exists for non-traded.
        if (
            not traded
            and matured
            and float(outcome.get("realized_pnl") or 0) > 0
        ):
            mw = {
                **row,
                "missed_winner": True,
                "missed_winner_reason": "NON_TRADED_WITH_POSITIVE_MATURED_OUTCOME",
            }
            missed.append(mw)
            row["missed_winner"] = True

        trade_vs.append(
            {
                "ae20_decision_id": d.get("ae20_decision_id"),
                "candidate_id": d.get("candidate_id"),
                "trade_class": "TRADED" if traded else "NO_TRADE",
                "skip_reason": skip_reason,
                "strict_decision": d.get("strict_decision"),
                "exploration_decision": d.get("exploration_decision"),
                "outcome_unavailable": row["outcome_unavailable"],
                "profitability_claim": False,
            }
        )

        disagreement = ""
        if str(d.get("strict_decision", "")).startswith("STRICT_BLOCKED") and str(
            d.get("exploration_decision", "")
        ).startswith("EXPLORATION"):
            disagreement = "STRICT_BLOCKED_VS_EXPLORATION_ALLOWED"
        if d.get("ae19_llm_action_label") in {"BUY", "SELL"} and not traded:
            disagreement = (disagreement + "|LLM_LABEL_IGNORED").strip("|")

        strict_vs.append(
            {
                "ae20_decision_id": d.get("ae20_decision_id"),
                "candidate_id": d.get("candidate_id"),
                "strict_approved": str(d.get("strict_decision", "")).startswith("STRICT_APPROVED"),
                "strict_blocked": str(d.get("strict_decision", "")).startswith("STRICT_BLOCKED"),
                "exploration_watch": "WATCH" in str(d.get("exploration_decision", "")),
                "exploration_paper_demo": "PAPER_DEMO" in str(d.get("exploration_decision", "")),
                "strict_decision": d.get("strict_decision"),
                "exploration_decision": d.get("exploration_decision"),
                "ae16_tier": d.get("ae16_consensus_tier"),
                "ae17_meta_decision": d.get("ae17_meta_decision"),
                "ae18_status": d.get("ae18_status"),
                "ae19_status": d.get("ae19_status"),
                "ae19_llm_action_label": d.get("ae19_llm_action_label"),
                "disagreement_case": disagreement,
                "non_executed_opportunity": (not traded)
                and str(d.get("exploration_decision", "")).startswith("EXPLORATION"),
                "backtest_claim": False,
                "profitability_claim": False,
            }
        )

    return {
        "skipped": skipped,
        "opportunity": opportunity,
        "missed_winners": missed,
        "trade_vs_no_trade": trade_vs,
        "strict_vs_exploration": strict_vs,
    }
