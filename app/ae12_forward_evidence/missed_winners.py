"""Missed-winner detection from matured forward outcomes."""

from __future__ import annotations

from typing import Any

from app.ae12_forward_evidence.types import DEFAULT_MISSED_WINNER_THRESHOLDS, TRADED_ACTIONS


def _was_traded(row: dict[str, Any]) -> bool:
    if row.get("was_traded") is True:
        return True
    action = str(row.get("paper_action_taken") or "").upper()
    return action in TRADED_ACTIONS


def _strict_blocked(row: dict[str, Any]) -> bool:
    strict = str(row.get("strict_shadow_decision") or "").upper()
    return strict in {"NO_TRADE", "BLOCKED", "REJECT"}


def _exploration_only_possible(row: dict[str, Any]) -> bool:
    exploration = str(row.get("exploration_decision") or "").upper()
    return exploration in {"TRADE", "TRADE_EXPLORATION_OVERRIDE", "PAPER_BUY", "NO_TRADE"}


def detect_missed_winners(
    evidence_rows: list[dict[str, Any]],
    *,
    thresholds: dict[str, float] | None = None,
    horizons: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Missed winner:
    - seen by system
    - not traded OR strict-blocked while exploration-only was possible
    - matured horizon return >= threshold
    - outcome computed without lookahead
    """
    th = dict(DEFAULT_MISSED_WINNER_THRESHOLDS)
    if thresholds:
        th.update(thresholds)
    hs = horizons or list(th.keys())
    missed: list[dict[str, Any]] = []

    for row in evidence_rows:
        traded = _was_traded(row)
        strict_blocked = _strict_blocked(row)
        exploration_possible = _exploration_only_possible(row)

        eligible = (not traded) or (strict_blocked and exploration_possible)
        if not eligible:
            continue

        for h in hs:
            matured = row.get(f"horizon_{h}_matured")
            status = row.get(f"horizon_{h}_no_lookahead_status")
            max_ret = row.get(f"horizon_{h}_max_return")
            if matured is not True:
                continue
            if status not in {"NO_LOOKAHEAD_OK"}:
                continue
            if max_ret is None:
                continue
            try:
                ret_f = float(max_ret)
            except (TypeError, ValueError):
                continue
            need = float(th.get(h, 9e9))
            if ret_f < need:
                continue
            missed.append(
                {
                    "evidence_row_id": row.get("evidence_row_id"),
                    "candidate_id": row.get("candidate_id"),
                    "decision_id": row.get("decision_id") or row.get("source_decision_id"),
                    "pair_address": row.get("pair_address"),
                    "first_seen_timestamp": row.get("first_seen_timestamp"),
                    "horizon": h,
                    "max_return": ret_f,
                    "threshold": need,
                    "was_traded": traded,
                    "strict_shadow_decision": row.get("strict_shadow_decision"),
                    "exploration_decision": row.get("exploration_decision"),
                    "reason_not_traded": row.get("reason_not_traded"),
                    "reason_recovery_status": row.get("reason_recovery_status"),
                    "rejection_reason": row.get("rejection_reason"),
                    "price_freshness_status": row.get("price_freshness_status"),
                    "context_missingness": row.get("ae8_context_status"),
                    "audit_blockers": row.get("ae9_audit_blockers"),
                    "cooldown_active": row.get("cooldown_active"),
                    "max_open_positions_hit": row.get("max_open_positions_hit"),
                    "duplicate_active_pair": row.get("duplicate_active_pair"),
                    "duplicate_reason": row.get("duplicate_reason"),
                    "no_lookahead_status": status,
                }
            )
    return missed


def missed_winner_counts_by_horizon(missed: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in missed:
        h = str(row.get("horizon") or "")
        out[h] = out.get(h, 0) + 1
    return out
