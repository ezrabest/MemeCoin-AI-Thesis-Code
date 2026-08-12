"""Strict paper fill-price resolution keyed by pair_address / coin_id."""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

PAPER_PRICE_SANITY_MAX_DEVIATION_PCT = float(
    os.getenv("PAPER_PRICE_SANITY_MAX_DEVIATION_PCT", "0.50")
)
MIN_FILL_PRICE_USD = float(os.getenv("PAPER_MIN_FILL_PRICE_USD", "1e-12"))
MAX_FILL_PRICE_USD = float(os.getenv("PAPER_MAX_FILL_PRICE_USD", "1e9"))
MAX_FEE_TO_NOTIONAL_PCT = float(os.getenv("PAPER_MAX_FEE_TO_NOTIONAL_PCT", "0.25"))
MAX_NOTIONAL_TO_EQUITY_MULTIPLIER = float(
    os.getenv("PAPER_MAX_NOTIONAL_TO_EQUITY_MULTIPLIER", "1.05")
)


@dataclass(frozen=True)
class FillPriceResolution:
    ok: bool
    price: float | None
    source: str
    market_price_usd: float | None
    pair_address: str | None
    coin_id: int | None
    price_timestamp: str | None
    rejection_reason: str | None = None


def parse_market_price_usd(value: Any) -> float | None:
    """Parse DexScreener priceUsd / stored price_usd safely."""
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(price):
        return None
    if price <= 0:
        return None
    if price < MIN_FILL_PRICE_USD or price > MAX_FILL_PRICE_USD:
        return None
    return price


def parse_dexscreener_pair_price(pair: dict[str, Any]) -> float | None:
    return parse_market_price_usd(pair.get("priceUsd"))


def _normalize_pair_address(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_coin_id(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _lookup_market_price(
    *,
    pair_address: str | None,
    coin_id: int | None,
    market_prices_by_pair: dict[str, float] | None,
    market_prices_by_coin_id: dict[int, float] | None,
) -> tuple[float | None, str]:
    if pair_address and market_prices_by_pair:
        key = pair_address.strip()
        if key in market_prices_by_pair:
            return market_prices_by_pair[key], "market_pair_address"
    if coin_id is not None and market_prices_by_coin_id:
        if coin_id in market_prices_by_coin_id:
            return market_prices_by_coin_id[coin_id], "market_coin_id"
    return None, "missing"


def validate_price_deviation(
    *,
    candidate_price: float,
    reference_price: float | None,
    max_deviation_pct: float = PAPER_PRICE_SANITY_MAX_DEVIATION_PCT,
) -> str | None:
    if reference_price is None or reference_price <= 0:
        return None
    deviation = abs(candidate_price - reference_price) / reference_price
    if deviation > max_deviation_pct:
        return (
            f"price_deviation_{deviation:.4f}_exceeds_{max_deviation_pct:.4f}"
        )
    return None


def _canonical_instrument_id(payload: dict[str, Any] | None) -> str | None:
    if not isinstance(payload, dict):
        return None
    raw = payload.get("instrument_id") or payload.get("execution_instrument_id")
    text = str(raw or "").strip()
    if not text:
        return None
    # Accept clean_forward:<chain>:<pair> (and future non-demo instrument prefixes
    # that are similarly structured). Require at least prefix:scope:key.
    parts = text.split(":", 2)
    if len(parts) != 3 or not parts[0] or not parts[1] or not parts[2]:
        return None
    return text


def resolve_buy_fill_price(
    coin: dict[str, Any],
    *,
    market_prices_by_pair: dict[str, float] | None = None,
    market_prices_by_coin_id: dict[int, float] | None = None,
    price_timestamp: str | None = None,
    allow_coin_price_fallback: bool = False,
) -> FillPriceResolution:
    pair_address = _normalize_pair_address(
        coin.get("pair_address") or coin.get("contract_address")
    )
    coin_id = _normalize_coin_id(coin.get("coin_id") or coin.get("id"))
    instrument_id = _canonical_instrument_id(coin)

    if not pair_address:
        return FillPriceResolution(
            ok=False,
            price=None,
            source="rejected",
            market_price_usd=None,
            pair_address=None,
            coin_id=coin_id,
            price_timestamp=price_timestamp,
            rejection_reason="missing_pair_address",
        )
    # Legacy path requires coin_id. Canonical instrument identity may proceed
    # without inventing a legacy coin_id (Clean Forward / future adapters).
    if coin_id is None and instrument_id is None:
        return FillPriceResolution(
            ok=False,
            price=None,
            source="rejected",
            market_price_usd=None,
            pair_address=pair_address,
            coin_id=None,
            price_timestamp=price_timestamp,
            rejection_reason="missing_coin_id",
        )

    market_price, source = _lookup_market_price(
        pair_address=pair_address,
        coin_id=coin_id,
        market_prices_by_pair=market_prices_by_pair,
        market_prices_by_coin_id=market_prices_by_coin_id,
    )
    if market_price is None and allow_coin_price_fallback:
        market_price = parse_market_price_usd(
            coin.get("price_usd") or coin.get("latest_price") or coin.get("entry_price")
        )
        if market_price is not None:
            if instrument_id and str(instrument_id).startswith("clean_forward:"):
                source = "clean_forward_market_feed"
            elif str(coin.get("candidate_source") or "") == "clean_forward_market_feed":
                source = "clean_forward_market_feed"
            else:
                source = "coin_record_fallback"

    if market_price is None:
        return FillPriceResolution(
            ok=False,
            price=None,
            source="rejected",
            market_price_usd=None,
            pair_address=pair_address,
            coin_id=coin_id,
            price_timestamp=price_timestamp,
            rejection_reason="missing_market_price_for_pair",
        )

    if instrument_id and source in ("market_pair_address", "instrument_pair_address"):
        if str(coin.get("candidate_source") or coin.get("instrument_source") or "") == (
            "clean_forward_market_feed"
        ) or str(instrument_id).startswith("clean_forward:"):
            source = "clean_forward_market_feed"
        else:
            source = "instrument_pair_address"

    return FillPriceResolution(
        ok=True,
        price=market_price,
        source=source,
        market_price_usd=market_price,
        pair_address=pair_address,
        coin_id=coin_id,
        price_timestamp=price_timestamp,
    )


def resolve_sell_fill_price(
    position: dict[str, Any],
    *,
    market_prices_by_pair: dict[str, float] | None = None,
    market_prices_by_coin_id: dict[int, float] | None = None,
    proposed_price: float | None = None,
    proposed_pair_address: str | None = None,
    proposed_coin_id: int | None = None,
    price_timestamp: str | None = None,
    max_deviation_pct: float = PAPER_PRICE_SANITY_MAX_DEVIATION_PCT,
) -> FillPriceResolution:
    pos_pair = _normalize_pair_address(position.get("pair_address"))
    pos_coin_id = _normalize_coin_id(position.get("coin_id"))
    pos_instrument_id = _canonical_instrument_id(position)
    entry_price = parse_market_price_usd(position.get("entry_price"))

    if not pos_pair:
        return FillPriceResolution(
            ok=False,
            price=None,
            source="rejected",
            market_price_usd=None,
            pair_address=None,
            coin_id=pos_coin_id,
            price_timestamp=price_timestamp,
            rejection_reason="open_position_missing_pair_address",
        )
    if pos_coin_id is None and pos_instrument_id is None:
        return FillPriceResolution(
            ok=False,
            price=None,
            source="rejected",
            market_price_usd=None,
            pair_address=pos_pair,
            coin_id=None,
            price_timestamp=price_timestamp,
            rejection_reason="open_position_missing_coin_id",
        )

    if proposed_pair_address:
        norm_prop_pair = _normalize_pair_address(proposed_pair_address)
        if norm_prop_pair and norm_prop_pair != pos_pair:
            return FillPriceResolution(
                ok=False,
                price=None,
                source="rejected",
                market_price_usd=None,
                pair_address=pos_pair,
                coin_id=pos_coin_id,
                price_timestamp=price_timestamp,
                rejection_reason="sell_pair_address_mismatch",
            )
    if (
        proposed_coin_id is not None
        and pos_coin_id is not None
        and int(proposed_coin_id) != pos_coin_id
    ):
        return FillPriceResolution(
            ok=False,
            price=None,
            source="rejected",
            market_price_usd=None,
            pair_address=pos_pair,
            coin_id=pos_coin_id,
            price_timestamp=price_timestamp,
            rejection_reason="sell_coin_id_mismatch",
        )

    market_price, source = _lookup_market_price(
        pair_address=pos_pair,
        coin_id=pos_coin_id,
        market_prices_by_pair=market_prices_by_pair,
        market_prices_by_coin_id=market_prices_by_coin_id,
    )
    if market_price is None:
        parsed_proposed = parse_market_price_usd(proposed_price)
        if parsed_proposed is not None:
            market_price = parsed_proposed
            source = "proposed_price"
        else:
            return FillPriceResolution(
                ok=False,
                price=None,
                source="rejected",
                market_price_usd=None,
                pair_address=pos_pair,
                coin_id=pos_coin_id,
                price_timestamp=price_timestamp,
                rejection_reason="missing_market_price_for_open_position_pair",
            )
    elif proposed_price is not None:
        parsed_proposed = parse_market_price_usd(proposed_price)
        if parsed_proposed is not None and abs(parsed_proposed - market_price) / market_price > 1e-9:
            if (
                _normalize_pair_address(proposed_pair_address) == pos_pair
                or (proposed_coin_id is not None and proposed_coin_id == pos_coin_id)
                or pos_instrument_id is not None
            ):
                market_price = parsed_proposed
                source = "proposed_price_same_pair"

    deviation_reason = validate_price_deviation(
        candidate_price=market_price,
        reference_price=entry_price,
        max_deviation_pct=max_deviation_pct,
    )
    if deviation_reason:
        return FillPriceResolution(
            ok=False,
            price=None,
            source="rejected",
            market_price_usd=market_price,
            pair_address=pos_pair,
            coin_id=pos_coin_id,
            price_timestamp=price_timestamp,
            rejection_reason=deviation_reason,
        )

    return FillPriceResolution(
        ok=True,
        price=market_price,
        source=source,
        market_price_usd=market_price,
        pair_address=pos_pair,
        coin_id=pos_coin_id,
        price_timestamp=price_timestamp,
    )


def build_market_price_maps(
    entries: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[int, float], dict[str, float]]:
    by_pair: dict[str, float] = {}
    by_coin: dict[int, float] = {}
    by_canonical: dict[str, float] = {}
    for entry in entries:
        pair_address = _normalize_pair_address(
            entry.get("pair_address") or entry.get("contract_address")
        )
        coin_id = _normalize_coin_id(entry.get("coin_id") or entry.get("id"))
        price = parse_market_price_usd(
            entry.get("price_usd") or entry.get("market_price_usd") or entry.get("price")
        )
        if price is None:
            continue
        canonical_key = str(
            entry.get("mark_price_lookup_key")
            or entry.get("canonical_market_identity")
            or entry.get("provider_pair_url_exact")
            or entry.get("provider_pair_url")
            or ""
        ).strip()
        if canonical_key:
            by_canonical[canonical_key] = price
        if pair_address:
            by_pair[pair_address] = price
        if coin_id is not None:
            by_coin[coin_id] = price
    return by_pair, by_coin, by_canonical
