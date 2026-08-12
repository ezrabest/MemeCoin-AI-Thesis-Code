"""Rejection reason analysis for AE11."""

from __future__ import annotations

from collections import Counter
from typing import Any


def _get_field(rec: Any, name: str) -> Any:
    val = getattr(rec, name, None)
    if val is not None:
        return val
    if isinstance(rec, dict):
        return rec.get(name)
    return None


def summarize_rejection_reasons(
    capture_records: list[Any],
    trade_decisions: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Aggregate rejection reasons from opportunity capture and trade decisions."""
    counter: Counter[str] = Counter()
    for rec in capture_records:
        reason = _get_field(rec, "reason_for_no_trade")
        if reason:
            counter[str(reason)] += 1
        dup = _get_field(rec, "duplicate_reason")
        if dup:
            counter[str(dup)] += 1
    if trade_decisions:
        for td in trade_decisions:
            reason = td.get("reason_for_no_trade") or td.get("rejection_reason")
            if reason:
                counter[str(reason)] += 1
    return [{"reason": k, "count": v} for k, v in counter.most_common()]


def build_trade_vs_no_trade_comparison(
    capture_records: list[Any],
) -> list[dict[str, Any]]:
    """Compare traded vs not-traded candidates."""
    rows: list[dict[str, Any]] = []
    traded_actions = ("FILLED", "OPENED", "TRADE", "TRADE_EXPLORATION_OVERRIDE")
    for rec in capture_records:
        action = _get_field(rec, "paper_action_taken")
        traded = action in traded_actions
        rows.append(
            {
                "source_decision_id": _get_field(rec, "source_decision_id"),
                "candidate_id": _get_field(rec, "candidate_id"),
                "pair_address": _get_field(rec, "pair_address"),
                "traded": traded,
                "strict_shadow_decision": _get_field(rec, "strict_shadow_decision"),
                "exploration_decision": _get_field(rec, "exploration_decision"),
                "reason_for_no_trade": _get_field(rec, "reason_for_no_trade"),
                "max_return_1h": _get_field(rec, "max_return_1h"),
                "max_return_24h": _get_field(rec, "max_return_24h"),
            }
        )
    return rows
