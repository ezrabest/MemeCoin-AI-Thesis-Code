"""AE18 market activity / tradability axis.

This module is deliberately orthogonal to display rendering:
- Missing symbols are display metadata only.
- Missing/stale/zero transaction activity is market activity / tradability.
- Liquidity and market cap alone are never treated as proof of tradable activity.
"""
from __future__ import annotations

from typing import Any

ACTIVE_PROVIDER_TXNS = "ACTIVE_PROVIDER_TXNS"
NO_RECENT_PROVIDER_TXNS = "NO_RECENT_PROVIDER_TXNS"
ACTIVITY_STAGNANT = "ACTIVITY_STAGNANT"
ACTIVITY_UNKNOWN = "ACTIVITY_UNKNOWN"

WATCH_ONLY_NO_RECENT_PROVIDER_TXNS = "WATCH_ONLY_NO_RECENT_PROVIDER_TXNS"
WATCH_ONLY_ACTIVITY_STAGNANT = "WATCH_ONLY_ACTIVITY_STAGNANT"
WATCH_ONLY_ACTIVITY_UNKNOWN = "WATCH_ONLY_ACTIVITY_UNKNOWN"

TXN_FIELDS = (
    "txns_m5_buys",
    "txns_m5_sells",
    "txns_h1_buys",
    "txns_h1_sells",
    "txns_h6_buys",
    "txns_h6_sells",
    "txns_h24_buys",
    "txns_h24_sells",
)

VOLUME_FIELDS = (
    "volume_m5",
    "volume_h1",
    "volume_h6",
    "volume_h24",
    "volume_24h",
)

DELTA_FIELDS = (
    "price_change_m5",
    "price_change_h1",
    "price_change_h6",
    "price_change_h24",
    "price_change_5m",
    "price_change_1h",
    "price_change_6h",
    "price_change_24h",
)

def _num(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None

def _values(row: dict[str, Any], fields: tuple[str, ...]) -> list[float]:
    out: list[float] = []
    for f in fields:
        n = _num(row.get(f))
        if n is not None:
            out.append(n)
    return out

def _any_positive(vals: list[float]) -> bool:
    return any(v > 0 for v in vals)

def _any_nonzero(vals: list[float]) -> bool:
    return any(abs(v) > 1e-12 for v in vals)

def evaluate_market_activity(row: dict[str, Any]) -> dict[str, Any]:
    """Classify provider market activity without using symbols as evidence.

    ACTIVE requires provider transaction flow and non-zero volume.
    Liquidity/market_cap/FDV alone never upgrades a row to active.
    """
    tx_vals = _values(row, TXN_FIELDS)
    vol_vals = _values(row, VOLUME_FIELDS)
    delta_vals = _values(row, DELTA_FIELDS)

    tx_total = sum(max(0.0, v) for v in tx_vals)
    volume_total = sum(max(0.0, v) for v in vol_vals)
    has_txns = tx_total > 0
    has_volume = volume_total > 0
    has_delta = _any_nonzero(delta_vals)

    liquidity = _num(row.get("liquidity_usd") or row.get("liquidity"))
    market_cap = _num(row.get("market_cap"))
    fdv = _num(row.get("fdv"))
    price = _num(row.get("price_usd") or row.get("price"))

    # Price alone is not enough to classify a market as stagnant.
    # Stagnant means provider reports static market context such as liquidity,
    # market cap, or FDV, but no txns/volume/deltas.
    has_static_market_metadata = any(
        v is not None and v > 0 for v in (liquidity, market_cap, fdv)
    )
    observed_any_activity_field = bool(tx_vals or vol_vals or delta_vals)

    if has_txns and has_volume:
        status = ACTIVE_PROVIDER_TXNS
        readiness = "PAPER_ELIGIBLE"
        block_reason = ""
    elif tx_vals and tx_total <= 0:
        status = NO_RECENT_PROVIDER_TXNS
        readiness = WATCH_ONLY_NO_RECENT_PROVIDER_TXNS
        block_reason = (
            "NO_RECENT_PROVIDER_TXNS — provider reports zero transaction flow; "
            "liquidity/market-cap alone is not treated as tradable activity."
        )
    elif has_static_market_metadata and not has_txns and not has_volume and not has_delta:
        status = ACTIVITY_STAGNANT
        readiness = WATCH_ONLY_ACTIVITY_STAGNANT
        block_reason = (
            "ACTIVITY_STAGNANT — static market metadata exists, but provider "
            "transactions, volume, and deltas do not show active flow."
        )
    elif observed_any_activity_field:
        status = ACTIVITY_STAGNANT
        readiness = WATCH_ONLY_ACTIVITY_STAGNANT
        block_reason = (
            "ACTIVITY_STAGNANT — provider activity fields are present but do not "
            "meet active transaction + non-zero volume criteria."
        )
    else:
        status = ACTIVITY_UNKNOWN
        readiness = WATCH_ONLY_ACTIVITY_UNKNOWN
        block_reason = (
            "ACTIVITY_UNKNOWN — insufficient provider transaction/volume/delta "
            "metadata to classify this market as active."
        )

    return {
        "market_activity_status": status,
        "activity_trade_readiness_status": readiness,
        "activity_trade_block_reason": block_reason,
        "market_activity_blocks_demo_entry": status != ACTIVE_PROVIDER_TXNS,
        "provider_txns_observed_field_count": len(tx_vals),
        "provider_txns_recent_total": tx_total,
        "provider_volume_observed_field_count": len(vol_vals),
        "provider_volume_recent_total": volume_total,
        "provider_price_delta_observed_field_count": len(delta_vals),
        "provider_price_delta_any_nonzero": has_delta,
        "market_activity_provenance": "ae18_market_activity_axis_from_provider_runtime_fields",
        "activity_uses_symbol_display": False,
        "activity_uses_liquidity_or_market_cap_as_activity_proxy": False,
    }
