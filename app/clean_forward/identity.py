"""Chain-aware Clean Forward identity helpers."""
from __future__ import annotations

from typing import Any

from app.ae13b_product.contract_resolver import normalize_address, normalize_chain
from app.clean_forward.schema import CleanForwardInstrumentIdentity

EVM_CHAINS = frozenset(
    {
        "ethereum",
        "bsc",
        "base",
        "arbitrum",
        "polygon",
        "avalanche",
    }
)


def normalize_address_for_chain(addr: str | None, *, chain: str | None) -> str:
    """Solana: preserve case. EVM-compatible: lowercase only."""
    return normalize_address(addr, chain=chain)


def pair_address_for_id(pair_address: str | None, *, chain: str | None) -> str:
    """Address form used inside deterministic IDs (chain-aware)."""
    return normalize_address_for_chain(pair_address, chain=chain)


def is_evm_chain(chain: str | None) -> bool:
    return normalize_chain(chain) in EVM_CHAINS


def _first(*values: Any) -> Any:
    for value in values:
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return default


def build_instrument_identity(row: dict[str, Any]) -> CleanForwardInstrumentIdentity:
    """Build CleanForwardInstrumentIdentity from a clean-forward market row."""
    chain_raw = _first(row.get("normalized_chain_id"), row.get("chain"), row.get("chain_id"))
    chain = normalize_chain(str(chain_raw or ""))
    pair_address = str(_first(row.get("pair_address"), row.get("provider_pair_id")) or "").strip()
    base = str(_first(row.get("base_token_address")) or "").strip()
    quote = str(_first(row.get("quote_token_address")) or "").strip()
    provider = str(_first(row.get("source_provider"), row.get("provider"), "dexscreener") or "dexscreener")

    return CleanForwardInstrumentIdentity(
        chain=chain,
        provider=provider,
        provider_pair_url=_first(row.get("provider_pair_url"), row.get("dexscreener_url")),
        pair_address=pair_address,
        pair_address_normalized=pair_address_for_id(pair_address, chain=chain),
        base_token_address=base,
        base_token_symbol=_first(row.get("base_token_symbol")),
        base_token_name=_first(row.get("base_token_name")),
        quote_token_address=quote,
        quote_token_symbol=_first(row.get("quote_token_symbol")),
        quote_token_name=_first(row.get("quote_token_name")),
        dex_id=_first(row.get("dex_id"), row.get("dex")),
        pair_created_at=row.get("pair_created_at"),
        shown_as_token_contract=_as_bool(row.get("shown_as_token_contract"), False),
        identity_status=str(row.get("identity_status") or ""),
        verification_status=str(row.get("verification_status") or ""),
        freshness_status=str(row.get("freshness_status") or ""),
        paper_demo_only=_as_bool(row.get("paper_demo_only"), True),
        live_trading_ready=_as_bool(row.get("live_trading_ready"), False),
    )
