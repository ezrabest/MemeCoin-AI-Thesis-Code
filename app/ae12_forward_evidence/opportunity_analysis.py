"""Trade vs no-trade and strict vs exploration opportunity analysis."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any

from app.ae12_forward_evidence.types import InterpretationStatus, TRADED_ACTIONS


def _is_traded(row: dict[str, Any]) -> bool:
    if row.get("was_traded") is True:
        return True
    return str(row.get("paper_action_taken") or "").upper() in TRADED_ACTIONS


def _is_strict_approved(row: dict[str, Any]) -> bool:
    return str(row.get("strict_shadow_decision") or "").upper() in {"TRADE", "APPROVED", "PASS"}


def _is_exploration_trade(row: dict[str, Any]) -> bool:
    exploration = str(row.get("exploration_decision") or "").upper()
    return exploration in {"TRADE", "TRADE_EXPLORATION_OVERRIDE", "PAPER_BUY"} and _is_traded(row)


def build_trade_vs_no_trade_comparison(
    rows: list[dict[str, Any]],
    *,
    horizons: list[str],
    min_sample: int = 5,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for h in horizons:
        traded_rets: list[float] = []
        not_traded_rets: list[float] = []
        traded_count = 0
        not_traded_count = 0
        strict_approved = 0
        strict_blocked = 0
        exploration_only = 0
        missed = 0
        matured_n = 0

        for row in rows:
            matured = row.get(f"horizon_{h}_matured") is True
            ret = row.get(f"horizon_{h}_max_return")
            status = row.get(f"horizon_{h}_no_lookahead_status")
            traded = _is_traded(row)
            if traded:
                traded_count += 1
            else:
                not_traded_count += 1
            if _is_strict_approved(row):
                strict_approved += 1
            else:
                strict_blocked += 1
            if (
                not _is_strict_approved(row)
                and str(row.get("exploration_decision") or "").upper()
                in {"TRADE", "TRADE_EXPLORATION_OVERRIDE", "NO_TRADE"}
                and not traded
            ):
                exploration_only += 1

            if not matured or status != "NO_LOOKAHEAD_OK" or ret is None:
                continue
            matured_n += 1
            try:
                rf = float(ret)
            except (TypeError, ValueError):
                continue
            if traded:
                traded_rets.append(rf)
            else:
                not_traded_rets.append(rf)

        # missed winners already tagged externally; count rows with was_missed_winner flag if present
        for row in rows:
            if row.get(f"is_missed_winner_{h}") or (
                row.get("missed_winner_horizons") and h in (row.get("missed_winner_horizons") or [])
            ):
                missed += 1

        interpretation = _interpret(traded_rets, not_traded_rets, matured_n, min_sample)
        out.append(
            {
                "horizon": h,
                "traded_count": traded_count,
                "not_traded_count": not_traded_count,
                "strict_approved_count": strict_approved,
                "strict_blocked_count": strict_blocked,
                "exploration_only_count": exploration_only,
                "median_forward_return_traded": median(traded_rets) if traded_rets else None,
                "median_forward_return_not_traded": median(not_traded_rets) if not_traded_rets else None,
                "max_forward_return_traded": max(traded_rets) if traded_rets else None,
                "max_forward_return_not_traded": max(not_traded_rets) if not_traded_rets else None,
                "missed_winner_count": missed,
                "sample_size_matured": matured_n,
                "interpretation_status": interpretation,
            }
        )
    return out


def _interpret(
    traded_rets: list[float],
    not_traded_rets: list[float],
    matured_n: int,
    min_sample: int,
) -> str:
    if matured_n == 0:
        return InterpretationStatus.TOO_EARLY_NO_MATURED_HORIZONS.value
    if matured_n < min_sample:
        return InterpretationStatus.INSUFFICIENT_SAMPLE.value
    if not traded_rets and not not_traded_rets:
        return InterpretationStatus.DATA_GAP.value
    if not traded_rets or not not_traded_rets:
        return InterpretationStatus.DATA_GAP.value
    mt = median(traded_rets)
    mn = median(not_traded_rets)
    if mt > mn * 1.05:
        return InterpretationStatus.TRADED_OUTPERFORMED.value
    if mn > mt * 1.05:
        return InterpretationStatus.NOT_TRADED_OUTPERFORMED.value
    return InterpretationStatus.MIXED.value


def build_strict_vs_exploration_comparison(
    rows: list[dict[str, Any]],
    *,
    horizons: list[str],
) -> dict[str, Any]:
    total = len(rows)
    strict_approved = sum(1 for r in rows if _is_strict_approved(r))
    strict_blocked = total - strict_approved
    exploration_traded = sum(1 for r in rows if _is_exploration_trade(r))
    exploration_only_trades = sum(
        1
        for r in rows
        if _is_exploration_trade(r) and not _is_strict_approved(r)
    )
    strict_approved_trades = sum(1 for r in rows if _is_strict_approved(r) and _is_traded(r))
    strict_blocked_but_exploration_traded = sum(
        1 for r in rows if (not _is_strict_approved(r)) and _is_exploration_trade(r)
    )

    blocker_counter: Counter[str] = Counter()
    for r in rows:
        reason = r.get("rejection_reason") or r.get("reason_not_traded")
        if reason and not _is_strict_approved(r):
            blocker_counter[str(reason)] += 1

    return_rows: list[dict[str, Any]] = []
    for h in horizons:
        sa_rets: list[float] = []
        eo_rets: list[float] = []
        for r in rows:
            if r.get(f"horizon_{h}_matured") is not True:
                continue
            if r.get(f"horizon_{h}_no_lookahead_status") != "NO_LOOKAHEAD_OK":
                continue
            ret = r.get(f"horizon_{h}_max_return")
            if ret is None:
                continue
            try:
                rf = float(ret)
            except (TypeError, ValueError):
                continue
            if _is_strict_approved(r):
                sa_rets.append(rf)
            elif not _is_traded(r) or _is_exploration_trade(r):
                eo_rets.append(rf)
        return_rows.append(
            {
                "horizon": h,
                "strict_approved_median_return": median(sa_rets) if sa_rets else None,
                "exploration_only_median_return": median(eo_rets) if eo_rets else None,
                "strict_approved_n": len(sa_rets),
                "exploration_only_n": len(eo_rets),
            }
        )

    return {
        "total_candidates": total,
        "strict_approved": strict_approved,
        "strict_blocked": strict_blocked,
        "exploration_traded": exploration_traded,
        "exploration_only_trades": exploration_only_trades,
        "strict_approved_trades": strict_approved_trades,
        "strict_blocked_but_exploration_traded": strict_blocked_but_exploration_traded,
        "top_strict_blockers": [
            {"reason": k, "count": v} for k, v in blocker_counter.most_common(20)
        ],
        "return_comparison_by_horizon": return_rows,
    }


def build_rejection_reason_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counter: Counter[str] = Counter()
    recovery: Counter[str] = Counter()
    for r in rows:
        recovery[str(r.get("reason_recovery_status") or "UNKNOWN")] += 1
        if _is_traded(r):
            continue
        reason = r.get("rejection_reason") or r.get("reason_not_traded") or "UNKNOWN_NOT_RECORDED"
        counter[str(reason)] += 1
    summary = [{"reason": k, "count": v} for k, v in counter.most_common()]
    for k, v in recovery.most_common():
        summary.append({"reason": f"RECOVERY_STATUS::{k}", "count": v})
    return summary
