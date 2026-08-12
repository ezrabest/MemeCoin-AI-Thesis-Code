"""AE13I Smoke Addendum (Part E) — AE14 readiness assessment.

Read-only, paper/demo-only diagnostic that tells the operator whether the
current live-market snapshot has enough fresh, tradable rows to run AE14
trading-validation smoke tests, vs. only being able to run AE14
negative-control tests (proving the freshness gate correctly blocks stale
data). Never mutates state, never places trades, never weakens
MarketDataGateKeeper.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

DEFAULT_MIN_TRADABLE_ROWS_FOR_AE14 = 10

NEGATIVE_CONTROL_REASON = (
    "Freshness gate is working, but no fresh tradable market rows are available."
)


def _is_stale_row(row: dict[str, Any]) -> bool:
    tradability = str(row.get("tradability_status") or "")
    if tradability == "stale_market_data":
        return True
    freshness = row.get("price_freshness")
    if isinstance(freshness, dict) and freshness.get("is_stale"):
        return True
    if row.get("mark_fresh") is False:
        return True
    return False


def compute_ae14_readiness(
    *,
    market_rows: list[dict[str, Any]] | None = None,
    min_tradable_rows_for_ae14: int = DEFAULT_MIN_TRADABLE_ROWS_FOR_AE14,
) -> dict[str, Any]:
    """Assess whether AE14 negative-control / trading-validation smoke can run.

    ``market_rows`` should be the ``rows`` list from
    :func:`app.ae13b_product.live_market.build_live_market` (or an equivalent
    coin/gatekeeper-enriched row list). Missing/empty input is treated as
    "no tradable rows available" rather than raising.
    """
    rows = market_rows if isinstance(market_rows, list) else []
    total_rows = len(rows)

    tradable_now_count = sum(
        1 for r in rows if isinstance(r, dict) and str(r.get("tradability_status") or "") == "tradable_now"
    )
    stale_count = sum(1 for r in rows if isinstance(r, dict) and _is_stale_row(r))
    other_count = max(0, total_rows - tradable_now_count - stale_count)

    threshold = int(min_tradable_rows_for_ae14)
    all_stale_or_none_tradable = tradable_now_count == 0

    # Negative-control (proving the freshness gate blocks stale/fallback data)
    # only requires the gate itself to be operational - it does not need any
    # fresh tradable rows, so it is always considered ready.
    ready_for_negative_control = True
    ready_for_trading_validation = tradable_now_count >= threshold

    if all_stale_or_none_tradable:
        reason = NEGATIVE_CONTROL_REASON
        recommended_next_action = (
            "Run AE14 negative-control tests now (freshness-gate-blocks-stale-data). "
            "Wait for fresh, tradable market rows before running AE14 "
            "trading-validation smoke tests."
        )
    elif ready_for_trading_validation:
        reason = (
            f"{tradable_now_count} fresh tradable market row(s) available "
            f"(threshold {threshold}) - AE14 trading-validation smoke can proceed."
        )
        recommended_next_action = (
            "Proceed with AE14 trading-validation smoke tests using currently "
            "tradable_now market rows."
        )
    else:
        reason = (
            f"Only {tradable_now_count} fresh tradable market row(s) available; "
            f"AE14 trading-validation smoke requires at least {threshold}."
        )
        recommended_next_action = (
            f"Wait for at least {threshold} tradable_now market rows, or lower "
            "min_tradable_rows_for_ae14 for a smaller smoke run. AE14 "
            "negative-control tests can run now."
        )

    return {
        "ready_for_negative_control": ready_for_negative_control,
        "ready_for_trading_validation": ready_for_trading_validation,
        "reason": reason,
        "recommended_next_action": recommended_next_action,
        "tradable_now_count": tradable_now_count,
        "fresh_count": tradable_now_count,
        "stale_count": stale_count,
        "other_count": other_count,
        "total_rows": total_rows,
        "min_tradable_rows_for_ae14": threshold,
        "all_stale_or_none_tradable": all_stale_or_none_tradable,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "paper_demo_only": True,
        "not_live_approved": True,
        "not_profitability_evidence": True,
        "live_trading_implied": False,
    }
