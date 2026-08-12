"""AE14 — Clean Forward Market Feed → GateKeeper / demo candidate bridge.

Dependency-light: standard library only. Must not pull in demo bot, demo queue,
API, or paper modules. Does not read market_snapshots. Does not invent coin_id,
price, liquidity, or timestamps.
"""
from __future__ import annotations

import math
from typing import Any

CANDIDATE_SOURCE = "clean_forward_market_feed"

_REQUIRED_VERIFICATION = "provider_pair_verified"
_REQUIRED_FRESHNESS = "fresh"
_REQUIRED_IDENTITY = "pair_and_tokens_separated"


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def _norm_addr(value: Any) -> str:
    return str(value or "").strip()


def _norm_addr_key(value: Any) -> str:
    return _norm_addr(value).lower()


def _norm_chain(value: Any) -> str:
    return str(value or "").strip().lower()


def find_matching_clean_forward_row(
    rows: list[dict[str, Any]] | None,
    *,
    chain: Any = None,
    pair_address: Any = None,
) -> dict[str, Any] | None:
    """Locate a Clean Forward row by chain + pair/provider address. No network."""
    if not isinstance(rows, list) or not rows:
        return None
    want_pair = _norm_addr_key(pair_address)
    if not want_pair:
        return None
    want_chain = _norm_chain(chain)
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_pair = _norm_addr_key(
            _first_non_empty(row.get("pair_address"), row.get("provider_pair_id"))
        )
        if row_pair != want_pair:
            continue
        if want_chain:
            row_chain = _norm_chain(
                _first_non_empty(
                    row.get("chain"),
                    row.get("normalized_chain_id"),
                    row.get("chain_id"),
                )
            )
            if row_chain and row_chain != want_chain:
                continue
        return dict(row)
    return None


def is_clean_forward_queue_item(item: dict[str, Any] | None) -> bool:
    """True when a demo-queue entry is Clean Forward-backed."""
    if not isinstance(item, dict):
        return False
    source = str(item.get("source") or "")
    if source.startswith("ae14_clean_forward"):
        return True
    if source == CANDIDATE_SOURCE:
        return True
    if str(item.get("market_match_status") or "") == _REQUIRED_VERIFICATION:
        return True
    return False


def _fail(
    *block_reasons: str,
    block_reason: str | None = None,
) -> dict[str, Any]:
    reasons = [str(r) for r in block_reasons if r]
    primary = block_reason or (reasons[0] if reasons else "CLEAN_FORWARD_BRIDGE_REJECTED")
    return {
        "ok": False,
        "candidate": None,
        "block_reason": primary,
        "block_reasons": reasons or [primary],
        "legacy_market_snapshots_used": False,
        "source": CANDIDATE_SOURCE,
    }


def build_clean_forward_gatekeeper_candidate(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize a Clean Forward feed row into a GateKeeper/demo candidate.

    Returns a structured result. Never raises on bad/missing market fields.
    Never invents price, liquidity, timestamps, or coin_id.
    Never falls back to market_snapshots.
    """
    if not isinstance(row, dict):
        return _fail("CLEAN_FORWARD_ROW_INVALID", block_reason="CLEAN_FORWARD_ROW_INVALID")

    block_reasons: list[str] = []

    verification = str(row.get("verification_status") or "")
    if verification != _REQUIRED_VERIFICATION:
        block_reasons.append(
            f"verification_status must be {_REQUIRED_VERIFICATION}, got {verification!r}"
        )

    freshness = str(row.get("freshness_status") or "")
    if freshness != _REQUIRED_FRESHNESS:
        block_reasons.append(
            f"freshness_status must be {_REQUIRED_FRESHNESS}, got {freshness!r}"
        )

    identity = str(row.get("identity_status") or "")
    if identity != _REQUIRED_IDENTITY:
        block_reasons.append(
            f"identity_status must be {_REQUIRED_IDENTITY}, got {identity!r}"
        )

    shown_as_token = _safe_bool(row.get("shown_as_token_contract"))
    if shown_as_token is not False:
        block_reasons.append(
            "shown_as_token_contract must be false "
            f"(got {row.get('shown_as_token_contract')!r})"
        )

    paper_demo_only = _safe_bool(row.get("paper_demo_only"))
    if paper_demo_only is not True:
        block_reasons.append(
            f"paper_demo_only must be true (got {row.get('paper_demo_only')!r})"
        )

    live_ready = _safe_bool(row.get("live_trading_ready"))
    if live_ready is not False:
        block_reasons.append(
            "live_trading_ready must be false "
            f"(got {row.get('live_trading_ready')!r})"
        )

    price = _safe_float(_first_non_empty(row.get("price_usd"), row.get("price")))
    if price is None or price <= 0:
        block_reasons.append("missing/zero/invalid price_usd")

    liquidity = _safe_float(
        _first_non_empty(row.get("liquidity_usd"), row.get("liquidity"))
    )
    if liquidity is None or liquidity <= 0:
        block_reasons.append("missing/zero/invalid liquidity_usd")

    timestamp = _first_non_empty(
        row.get("observed_at"),
        row.get("fetched_at"),
        row.get("last_fetched"),
        row.get("ingested_at"),
    )
    if timestamp is None:
        block_reasons.append("missing timestamp (observed_at/fetched_at/last_fetched/ingested_at)")

    if block_reasons:
        return _fail(*block_reasons)

    chain = _first_non_empty(
        row.get("chain"),
        row.get("normalized_chain_id"),
        row.get("chain_id"),
    )
    chain_norm = _norm_chain(chain) if chain is not None else None
    pair_address = _norm_addr(
        _first_non_empty(row.get("pair_address"), row.get("provider_pair_id"))
    ) or None
    provider_pair_id = _norm_addr(
        _first_non_empty(row.get("provider_pair_id"), row.get("pair_address"))
    ) or None
    base_token_address = _norm_addr(row.get("base_token_address")) or None
    quote_token_address = _norm_addr(row.get("quote_token_address")) or None
    symbol = _first_non_empty(row.get("base_token_symbol"), row.get("pair"), row.get("pair_label"))
    pair_label = _first_non_empty(row.get("pair"), row.get("pair_label"))
    source_provider = str(
        _first_non_empty(row.get("source_provider"), row.get("provider"), CANDIDATE_SOURCE)
    )
    address_role = str(_first_non_empty(row.get("address_role"), "pair_contract"))
    ts = str(timestamp)

    candidate: dict[str, Any] = {
        "latest_price": float(price),  # type: ignore[arg-type]
        "price_usd": float(price),  # type: ignore[arg-type]
        "price": float(price),  # type: ignore[arg-type]
        "latest_liquidity": float(liquidity),  # type: ignore[arg-type]
        "liquidity_usd": float(liquidity),  # type: ignore[arg-type]
        "liquidity": float(liquidity),  # type: ignore[arg-type]
        "price_updated_at": ts,
        "liquidity_updated_at": ts,
        "last_seen_at": ts,
        "observed_at": ts,
        "source_provider": source_provider,
        "chain": chain_norm,
        "pair": pair_label,
        "symbol": str(symbol).strip().upper() if symbol else None,
        "pair_address": pair_address,
        "provider_pair_id": provider_pair_id,
        "primary_address": pair_address or provider_pair_id,
        "base_token_address": base_token_address,
        "token_contract_address": base_token_address,
        "quote_token_address": quote_token_address,
        "address_role": address_role,
        "paper_demo_only": True,
        "not_live_approved": True,
        "live_trading_ready": False,
        "candidate_source": CANDIDATE_SOURCE,
        "source": CANDIDATE_SOURCE,
        "market_match_status": _REQUIRED_VERIFICATION,
        "legacy_market_snapshots_used": False,
        "clean_forward_bridge_used": True,
        "verification_status": verification,
        "freshness_status": freshness,
        "identity_status": identity,
        "shown_as_token_contract": False,
        # Explicit: never invent a DB coin_id for Clean Forward rows.
        "id": None,
        "coin_id": None,
    }

    # Deterministic instrument id (mode-agnostic identity; policy applied later).
    if chain_norm and (pair_address or provider_pair_id):
        instrument_id = (
            f"clean_forward:{chain_norm}:{pair_address or provider_pair_id}"
        )
        candidate["instrument_id"] = instrument_id
        candidate["execution_instrument_id"] = instrument_id
        candidate["instrument_source"] = CANDIDATE_SOURCE

    if chain_norm == "solana" and base_token_address:
        candidate["token_mint_address"] = base_token_address

    # Preserve optional momentum / activity fields when present (no invention).
    for key in (
        "price_change_5m",
        "price_change_1h",
        "price_change_6h",
        "price_change_24h",
        "volume_24h",
        "volume_1h",
        "volume_5m",
        "txns_24h_buys",
        "txns_24h_sells",
        "row_id",
        "provider_pair_url",
        "dexscreener_url",
        "base_token_symbol",
        "quote_token_symbol",
        "pair_label",
    ):
        if row.get(key) is not None:
            candidate[key] = row.get(key)

    return {
        "ok": True,
        "candidate": candidate,
        "block_reason": None,
        "block_reasons": [],
        "legacy_market_snapshots_used": False,
        "source": CANDIDATE_SOURCE,
    }
