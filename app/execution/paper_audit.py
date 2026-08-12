"""Shared validation helpers for paper trade rows and aggregates."""
from __future__ import annotations

import math
from typing import Any

from app.execution.fill_price import MAX_FEE_TO_NOTIONAL_PCT, MAX_NOTIONAL_TO_EQUITY_MULTIPLIER

DOGE_STYLE_RATIO_THRESHOLD = 50.0
EXTREME_ROI_THRESHOLD = 5.0
IMPOSSIBLE_NOTIONAL_USD = 1_000_000.0


def _float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def row_issue_codes(row: dict[str, Any], *, open_qty_by_position: dict[int, float]) -> list[str]:
    issues: list[str] = []
    side = str(row.get("side") or "").lower()
    fill_price = _float_or_none(row.get("fill_price"))
    quantity = _float_or_none(row.get("quantity"))
    notional = _float_or_none(row.get("notional_usd"))
    total_fees = _float_or_none(row.get("total_fees"))
    net_roi = _float_or_none(row.get("net_roi_pct"))
    pos_id = _int_or_none(row.get("position_id"))
    pair_address = str(row.get("pair_address") or "").strip()
    coin_id = _int_or_none(row.get("coin_id"))
    instrument_id = str(
        row.get("instrument_id") or row.get("execution_instrument_id") or ""
    ).strip()

    if fill_price is None or fill_price <= 0:
        issues.append("invalid_fill_price")
    if quantity is None or quantity <= 0:
        issues.append("invalid_quantity")
    if notional is None or notional < 0:
        issues.append("invalid_notional")
    if not pair_address:
        issues.append("missing_pair_address")
    # Legacy rows require coin_id. Canonical instrument identity is an
    # accepted alternative (Clean Forward / future live adapters).
    if coin_id is None and not instrument_id:
        issues.append("missing_coin_id")

    if fill_price and quantity and notional is not None:
        implied = fill_price * quantity
        if notional > 0 and abs(implied - notional) / notional > 0.05:
            issues.append("notional_price_quantity_mismatch")

    if notional is not None and notional > IMPOSSIBLE_NOTIONAL_USD:
        issues.append("impossible_notional")

    if total_fees is not None and notional and notional > 0:
        if total_fees / notional > MAX_FEE_TO_NOTIONAL_PCT:
            issues.append("fee_notional_anomaly")

    if side == "sell":
        if pos_id is None:
            issues.append("sell_missing_position_id")
        elif pos_id not in open_qty_by_position:
            issues.append("sell_without_matching_open_position")
        else:
            open_qty = open_qty_by_position[pos_id]
            if quantity is not None and quantity > open_qty * 1.000001:
                issues.append("sell_quantity_exceeds_open_position")
            open_qty_by_position[pos_id] = max(0.0, open_qty - (quantity or 0.0))
        if net_roi is not None and abs(net_roi) > EXTREME_ROI_THRESHOLD:
            issues.append("extreme_roi")

    if side == "buy" and pos_id is not None:
        open_qty_by_position[pos_id] = (open_qty_by_position.get(pos_id, 0.0) + (quantity or 0.0))

    return issues


def detect_doge_style_pattern(
    buy_row: dict[str, Any] | None,
    sell_row: dict[str, Any] | None,
) -> bool:
    if not buy_row or not sell_row:
        return False
    buy_price = _float_or_none(buy_row.get("fill_price"))
    sell_price = _float_or_none(sell_row.get("fill_price"))
    if buy_price is None or sell_price is None or buy_price <= 0:
        return False
    ratio = sell_price / buy_price
    return ratio >= DOGE_STYLE_RATIO_THRESHOLD


def audit_trade_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    open_qty_by_position: dict[int, float] = {}
    valid_rows: list[dict[str, Any]] = []
    invalid_rows: list[dict[str, Any]] = []
    issue_counts: dict[str, int] = {}
    buys_by_position: dict[int, dict[str, Any]] = {}
    first_corrupted_index: int | None = None
    doge_style_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        side = str(row.get("side") or "").lower()
        pos_id = _int_or_none(row.get("position_id"))
        if side == "buy" and pos_id is not None:
            buys_by_position[pos_id] = row

        issues = row_issue_codes(row, open_qty_by_position=open_qty_by_position)
        if side == "sell" and pos_id is not None:
            if detect_doge_style_pattern(buys_by_position.get(pos_id), row):
                issues.append("doge_style_wrong_buy_tiny_sell_realistic")
                doge_style_rows.append({
                    "position_id": pos_id,
                    "symbol": row.get("symbol"),
                    "buy_fill_price": buys_by_position.get(pos_id, {}).get("fill_price"),
                    "sell_fill_price": row.get("fill_price"),
                    "buy_notional": buys_by_position.get(pos_id, {}).get("notional_usd"),
                    "sell_notional": row.get("notional_usd"),
                })

        if issues:
            enriched = {**row, "issues": issues, "row_index": idx}
            invalid_rows.append(enriched)
            if first_corrupted_index is None:
                first_corrupted_index = idx
            for code in issues:
                issue_counts[code] = issue_counts.get(code, 0) + 1
        else:
            valid_rows.append(row)

    def _aggregate(target_rows: list[dict[str, Any]]) -> dict[str, Any]:
        sells = [r for r in target_rows if str(r.get("side")).lower() == "sell"]
        realized = sum(_float_or_none(r.get("realized_pnl")) or 0.0 for r in sells)
        fees = sum(_float_or_none(r.get("total_fees")) or 0.0 for r in target_rows)
        swap_fees = sum(_float_or_none(r.get("swap_fee")) or 0.0 for r in target_rows)
        return {
            "row_count": len(target_rows),
            "sell_count": len(sells),
            "total_realized_pnl": round(realized, 6),
            "total_fees": round(fees, 6),
            "total_swap_fees": round(swap_fees, 6),
        }

    return {
        "total_rows": len(rows),
        "valid_rows": len(valid_rows),
        "invalid_rows": len(invalid_rows),
        "first_corrupted_row_index": first_corrupted_index,
        "first_corrupted_row": invalid_rows[0] if invalid_rows else None,
        "issue_counts": issue_counts,
        "doge_style_rows": doge_style_rows,
        "valid_aggregate": _aggregate(valid_rows),
        "raw_aggregate": _aggregate(rows),
        "invalid_row_details": invalid_rows,
        "valid_row_details": valid_rows,
    }


def portfolio_roi_from_equity(
    *,
    current_equity: float,
    starting_capital: float,
) -> float:
    if starting_capital <= 0:
        return 0.0
    return (current_equity - starting_capital) / starting_capital


def notional_exceeds_equity_guard(
    notional_usd: float,
    current_equity: float,
    *,
    multiplier: float = MAX_NOTIONAL_TO_EQUITY_MULTIPLIER,
) -> bool:
    if current_equity <= 0:
        return notional_usd > 0
    return notional_usd > current_equity * multiplier
