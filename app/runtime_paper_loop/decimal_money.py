"""Decimal money helpers for AE11E live-like paper ledger accounting."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_EVEN, InvalidOperation
from typing import Any

USD_QUANT = Decimal("0.000001")
PRICE_QUANT = Decimal("0.000000000001")
QTY_QUANT = Decimal("0.00000001")
LEDGER_TOLERANCE = Decimal("0.000001")
BPS_DENOM = Decimal("10000")


def to_decimal(value: Any) -> Decimal:
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, bool):
        return Decimal(int(value))
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError, TypeError):
        return Decimal("0")


def decimal_to_str(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value, "f")


def quantize_usd(value: Any) -> Decimal:
    return to_decimal(value).quantize(USD_QUANT, rounding=ROUND_HALF_EVEN)


def quantize_price(value: Any) -> Decimal:
    return to_decimal(value).quantize(PRICE_QUANT, rounding=ROUND_HALF_EVEN)


def quantize_quantity(value: Any) -> Decimal:
    return to_decimal(value).quantize(QTY_QUANT, rounding=ROUND_HALF_EVEN)


def bps_cost(notional: Any, bps: Any) -> Decimal:
    return quantize_usd(to_decimal(notional) * to_decimal(bps) / BPS_DENOM)


def decimal_almost_equal(a: Any, b: Any, tol: Decimal = LEDGER_TOLERANCE) -> bool:
    return abs(quantize_usd(a) - quantize_usd(b)) <= tol


DECIMAL_PRECISION_POLICY = (
    f"usd={USD_QUANT} price={PRICE_QUANT} qty={QTY_QUANT} "
    f"tolerance={LEDGER_TOLERANCE} rounding=ROUND_HALF_EVEN"
)
