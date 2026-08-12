"""Dynamic slippage estimation — AMM-inspired, decimal-fraction units, rejection not cap."""
from __future__ import annotations

from typing import Any

from .audit_reasons import AuditReason


def estimate_slippage_per_side_pct(
    *,
    position_size_usd: float,
    liquidity_usd: float,
    volume_24h: float | None = None,
    baseline_slippage_pct: float = 0.015,
    baseline_slippage_is_per_side: bool = True,
    dynamic_slippage_enabled: bool = True,
    effective_liquidity_conservative_factor: float = 1.0,
    slippage_volume_liquidity_multiplier: float = 0.5,
) -> tuple[float | None, list[str]]:
    """
    Compute estimated slippage per side as decimal fraction (0.015 = 1.5%).

    Formula (constant-product AMM x*y=k approximation):
        trade_fraction = position_size_usd / effective_liquidity_usd
        amm_price_impact_pct = trade_fraction / (1 + trade_fraction)

    effective_liquidity_usd = liquidity_usd / effective_liquidity_conservative_factor

    estimated_slippage_per_side_pct =
        baseline_slippage_pct
        + amm_price_impact_pct
        + volume_liquidity_penalty_pct

    volume_liquidity_penalty_pct =
        (volume_24h / effective_liquidity_usd) * slippage_volume_liquidity_multiplier / 100
    """
    reasons: list[str] = []
    if liquidity_usd is None or liquidity_usd <= 0:
        reasons.append(AuditReason.BLOCKED_BY_MISSING_SLIPPAGE_INPUTS.value)
        return None, reasons

    base = float(baseline_slippage_pct)
    if not baseline_slippage_is_per_side:
        base = base / 2.0

    if not dynamic_slippage_enabled:
        return round(base, 8), reasons

    factor = max(float(effective_liquidity_conservative_factor), 1e-9)
    effective_liq = liquidity_usd / factor
    if effective_liq <= 0:
        reasons.append(AuditReason.BLOCKED_BY_MISSING_SLIPPAGE_INPUTS.value)
        return None, reasons

    trade_fraction = position_size_usd / effective_liq
    amm_price_impact_pct = trade_fraction / (1.0 + trade_fraction)

    vol_penalty = 0.0
    if volume_24h is not None and volume_24h > 0:
        vol_penalty = (volume_24h / effective_liq) * float(slippage_volume_liquidity_multiplier) / 100.0

    estimated = base + amm_price_impact_pct + vol_penalty
    return round(max(estimated, 0.0), 8), reasons


def compute_total_cost_pct(
    *,
    round_trip_fee_pct: float,
    round_trip_slippage_pct: float,
    gas_or_priority_cost_pct: float = 0.0,
) -> float:
    """Total round-trip cost as decimal fraction (0.03 = 3%). All inputs are decimal fractions."""
    return round(
        float(round_trip_fee_pct) + float(round_trip_slippage_pct) + float(gas_or_priority_cost_pct),
        8,
    )


def check_slippage_limit(
    estimated_slippage_per_side_pct: float | None,
    max_slippage_pct: float,
) -> tuple[bool, list[str]]:
    """True if trade passes slippage rejection threshold (not a cap)."""
    if estimated_slippage_per_side_pct is None:
        return False, [AuditReason.BLOCKED_BY_MISSING_SLIPPAGE_INPUTS.value]
    if estimated_slippage_per_side_pct > max_slippage_pct:
        return False, [AuditReason.BLOCKED_BY_SLIPPAGE_LIMIT.value]
    return True, []


def check_price_drift(
    *,
    model_snapshot_price: float | None,
    current_execution_price: float | None,
    max_price_drift_from_model_pct: float,
) -> tuple[bool, float | None, list[str]]:
    """
    price_drift_from_model_pct = abs(exec - snapshot) / snapshot (decimal fraction).
    max_price_drift_from_model_pct is also a decimal fraction (0.01 = 1%).
    """
    reasons: list[str] = []
    if model_snapshot_price is None or model_snapshot_price <= 0:
        reasons.append(AuditReason.MISSING_MODEL_SNAPSHOT_PRICE.value)
        return False, None, reasons
    if current_execution_price is None or current_execution_price <= 0:
        reasons.append(AuditReason.PRICE_FILL_RESOLUTION_FAILED.value)
        return False, None, reasons

    drift = abs(current_execution_price - model_snapshot_price) / model_snapshot_price
    drift_pct = round(drift, 8)
    if drift > max_price_drift_from_model_pct:
        reasons.append(AuditReason.BLOCKED_BY_PRICE_DRIFT.value)
        return False, drift_pct, reasons
    return True, drift_pct, reasons
