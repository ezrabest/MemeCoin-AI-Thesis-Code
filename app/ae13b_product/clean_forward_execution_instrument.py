"""Canonical Clean Forward execution instrument (mode-agnostic identity).

Instrument identity is independent of legacy coin_id and of execution policy.
Paper adapters may execute against it now; live adapters can route later.
This module does not enable live trading, wallets, or transaction submission.
"""
from __future__ import annotations

import math
from typing import Any

INSTRUMENT_SOURCE = "clean_forward_market_feed"
INSTRUMENT_ID_PREFIX = "clean_forward"


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
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


def _norm_chain(value: Any) -> str:
    return str(value or "").strip().lower()


def _norm_addr(value: Any) -> str:
    return str(value or "").strip()


def make_clean_forward_instrument_id(*, chain: str, pair_address: str) -> str:
    """Deterministic instrument id: clean_forward:<chain>:<pair_address>."""
    return f"{INSTRUMENT_ID_PREFIX}:{_norm_chain(chain)}:{_norm_addr(pair_address)}"


def is_clean_forward_instrument_id(value: Any) -> bool:
    text = str(value or "").strip()
    if not text.startswith(f"{INSTRUMENT_ID_PREFIX}:"):
        return False
    parts = text.split(":", 2)
    return len(parts) == 3 and bool(parts[1]) and bool(parts[2])


def has_canonical_instrument_identity(payload: dict[str, Any] | None) -> bool:
    """True when a payload carries a usable Clean Forward instrument id."""
    if not isinstance(payload, dict):
        return False
    return is_clean_forward_instrument_id(
        _first_non_empty(
            payload.get("instrument_id"),
            payload.get("execution_instrument_id"),
        )
    )


def _fail(*reasons: str, block_reason: str | None = None) -> dict[str, Any]:
    cleaned = [str(r) for r in reasons if r]
    primary = block_reason or (cleaned[0] if cleaned else "CLEAN_FORWARD_INSTRUMENT_REJECTED")
    return {
        "ok": False,
        "instrument": None,
        "block_reason": primary,
        "block_reasons": cleaned or [primary],
        "legacy_market_snapshots_used": False,
        "live_execution_enabled": False,
    }


def build_clean_forward_execution_instrument(
    candidate: dict[str, Any],
    *,
    execution_mode: str = "paper",
) -> dict[str, Any]:
    """Build a canonical execution instrument from a normalized CF candidate.

    Identity is mode-agnostic. ``execution_mode`` / live flags are policy
    overlays for the current adapter call — they do not redefine identity.
    Never invents coin_id. Never enables live execution.
    """
    if not isinstance(candidate, dict):
        return _fail("CLEAN_FORWARD_CANDIDATE_INVALID")

    block_reasons: list[str] = []

    candidate_source = str(candidate.get("candidate_source") or candidate.get("source") or "")
    if candidate_source != INSTRUMENT_SOURCE:
        block_reasons.append(
            f"candidate_source must be {INSTRUMENT_SOURCE}, got {candidate_source!r}"
        )

    if _safe_bool(candidate.get("clean_forward_bridge_used")) is not True:
        block_reasons.append("clean_forward_bridge_used must be true")

    if _safe_bool(candidate.get("legacy_market_snapshots_used")) is not False:
        block_reasons.append("legacy_market_snapshots_used must be false")

    chain = _norm_chain(
        _first_non_empty(
            candidate.get("chain"),
            candidate.get("normalized_chain_id"),
            candidate.get("chain_id"),
        )
    )
    if not chain:
        block_reasons.append("missing chain")

    pair_address = _norm_addr(
        _first_non_empty(candidate.get("pair_address"), candidate.get("provider_pair_id"))
    )
    provider_pair_id = _norm_addr(
        _first_non_empty(candidate.get("provider_pair_id"), candidate.get("pair_address"))
    )
    if not pair_address and not provider_pair_id:
        block_reasons.append("missing pair_address/provider_pair_id")

    price = _safe_float(
        _first_non_empty(
            candidate.get("latest_price"),
            candidate.get("price_usd"),
            candidate.get("price"),
        )
    )
    if price is None or price <= 0:
        block_reasons.append("missing/zero/invalid latest_price")

    liquidity = _safe_float(
        _first_non_empty(
            candidate.get("latest_liquidity"),
            candidate.get("liquidity_usd"),
            candidate.get("liquidity"),
        )
    )
    if liquidity is None or liquidity <= 0:
        block_reasons.append("missing/zero/invalid latest_liquidity")

    price_updated_at = _first_non_empty(
        candidate.get("price_updated_at"),
        candidate.get("observed_at"),
        candidate.get("last_seen_at"),
        candidate.get("fetched_at"),
    )
    liquidity_updated_at = _first_non_empty(
        candidate.get("liquidity_updated_at"),
        price_updated_at,
    )
    if price_updated_at is None or liquidity_updated_at is None:
        block_reasons.append("missing price/liquidity timestamps")

    mode = str(execution_mode or "paper").strip().lower() or "paper"
    if mode != "paper":
        # Structural identity exists, but this builder only authorizes paper now.
        block_reasons.append(
            f"execution_mode {mode!r} is not enabled; only paper is authorized"
        )

    if block_reasons:
        return _fail(*block_reasons)

    instrument_id = make_clean_forward_instrument_id(
        chain=chain, pair_address=pair_address or provider_pair_id
    )
    symbol = _first_non_empty(candidate.get("symbol"), candidate.get("base_token_symbol"))
    pair_label = _first_non_empty(candidate.get("pair"), candidate.get("pair_label"))

    instrument: dict[str, Any] = {
        # Canonical identity (mode-agnostic)
        "instrument_id": instrument_id,
        "execution_instrument_id": instrument_id,
        "instrument_source": INSTRUMENT_SOURCE,
        "candidate_source": INSTRUMENT_SOURCE,
        "market_match_status": str(
            candidate.get("market_match_status") or "provider_pair_verified"
        ),
        "chain": chain,
        "pair": pair_label,
        "symbol": str(symbol).strip().upper() if symbol else None,
        "pair_address": pair_address or provider_pair_id,
        "provider_pair_id": provider_pair_id or pair_address,
        "primary_address": pair_address or provider_pair_id,
        "base_token_address": _norm_addr(candidate.get("base_token_address")) or None,
        "quote_token_address": _norm_addr(candidate.get("quote_token_address")) or None,
        "token_contract_address": _norm_addr(
            candidate.get("token_contract_address") or candidate.get("base_token_address")
        )
        or None,
        "token_mint_address": candidate.get("token_mint_address"),
        "address_role": candidate.get("address_role") or "pair_contract",
        "latest_price": float(price),  # type: ignore[arg-type]
        "price_usd": float(price),  # type: ignore[arg-type]
        "price": float(price),  # type: ignore[arg-type]
        "entry_price": float(price),  # type: ignore[arg-type]
        "latest_liquidity": float(liquidity),  # type: ignore[arg-type]
        "liquidity_usd": float(liquidity),  # type: ignore[arg-type]
        "liquidity_at_entry": float(liquidity),  # type: ignore[arg-type]
        "price_updated_at": str(price_updated_at),
        "liquidity_updated_at": str(liquidity_updated_at),
        "last_seen_at": str(price_updated_at),
        "source_provider": candidate.get("source_provider") or INSTRUMENT_SOURCE,
        "clean_forward_bridge_used": True,
        "legacy_market_snapshots_used": False,
        # Explicit: never invent legacy DB coin_id
        "id": None,
        "coin_id": None,
        # Execution policy overlay (current adapter authorization)
        "execution_mode": "paper",
        "live_trading_ready": False,
        "live_execution_enabled": False,
        "wallet_required": False,
        "wallet_connected": False,
        "paper_demo_only": True,
        "not_live_approved": True,
        "not_profitability_evidence": True,
    }

    return {
        "ok": True,
        "instrument": instrument,
        "block_reason": None,
        "block_reasons": [],
        "legacy_market_snapshots_used": False,
        "live_execution_enabled": False,
        "instrument_id": instrument_id,
    }
