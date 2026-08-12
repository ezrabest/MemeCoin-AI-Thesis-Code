"""Coin/token identity resolution for AE12 Gemini pair-asset adjudications."""

from __future__ import annotations

import re
from typing import Any

QUOTE_OR_STABLE_NATIVE: frozenset[str] = frozenset(
    {
        "USDC",
        "USDC.E",
        "USDT",
        "DAI",
        "FDUSD",
        "TUSD",
        "WETH",
        "ETH",
        "WBNB",
        "BNB",
        "WSOL",
        "SOL",
        "WBTC",
        "BTC",
        "WMATIC",
        "MATIC",
        "WAVAX",
        "AVAX",
    }
)

WRAP_TO_NATIVE: dict[str, str] = {
    "WETH": "ETH",
    "WBNB": "BNB",
    "WSOL": "SOL",
    "WBTC": "BTC",
    "WMATIC": "MATIC",
    "WAVAX": "AVAX",
}


def normalize_symbol(symbol: str | None) -> str:
    return str(symbol or "").strip().upper()


def normalize_wrapped_native(symbol: str | None) -> str:
    s = normalize_symbol(symbol)
    return WRAP_TO_NATIVE.get(s, s)


def parse_pair_symbol(symbol: str | None) -> tuple[str, str | None]:
    """Return (base_symbol, quote_symbol) for pair-like symbols."""
    raw = str(symbol or "").strip()
    if not raw:
        return "", None
    if "/" in raw:
        left, right = raw.split("/", 1)
        base = normalize_symbol(left)
        quote = normalize_symbol(right)
        return base, quote or None
    return normalize_symbol(raw), None


def _clean_address(value: Any) -> str:
    return str(value or "").strip()


def _looks_like_token_address(value: str, *, pair_address: str = "") -> bool:
    addr = _clean_address(value)
    if not addr:
        return False
    pair = _clean_address(pair_address)
    if pair and addr.lower() == pair.lower():
        return False
    if addr.startswith("0x") and len(addr) >= 42:
        return True
    if not addr.startswith("0x") and 32 <= len(addr) <= 64:
        return True
    return False


def resolve_coin_identity(row: dict[str, Any]) -> dict[str, Any]:
    """Resolve a pair-asset row into a coin identity key + audit fields."""
    chain = str(row.get("chain") or "").strip().lower()
    symbol_raw = str(row.get("symbol") or "")
    name = str(row.get("name") or "").strip()
    base_sym, quote_sym = parse_pair_symbol(symbol_raw)
    base_norm = normalize_wrapped_native(base_sym)
    quote_norm = normalize_wrapped_native(quote_sym) if quote_sym else None

    identity_warning = ""
    if quote_sym and base_norm in QUOTE_OR_STABLE_NATIVE and quote_norm not in QUOTE_OR_STABLE_NATIVE:
        identity_warning = "BASE_LOOKS_LIKE_QUOTE"

    token_address = _clean_address(row.get("token_address") or row.get("base_token_address"))
    pair_address = _clean_address(row.get("pair_address"))
    contract_address = _clean_address(row.get("contract_address"))
    base_token_address = _clean_address(row.get("base_token_address") or token_address)

    warnings: list[str] = []
    if identity_warning:
        warnings.append(identity_warning)
    if not base_norm:
        warnings.append("MISSING_BASE_SYMBOL")

    if base_token_address and chain and _looks_like_token_address(base_token_address, pair_address=pair_address):
        method = "base_token_address+chain"
        confidence = "HIGH"
        coin_id = f"{chain}:{base_token_address.lower()}"
    elif token_address and chain and _looks_like_token_address(token_address, pair_address=pair_address):
        method = "token_address+chain"
        confidence = "HIGH"
        coin_id = f"{chain}:{token_address.lower()}"
    elif (
        contract_address
        and chain
        and _looks_like_token_address(contract_address, pair_address=pair_address)
        and contract_address.lower() != pair_address.lower()
    ):
        method = "contract_address+chain"
        confidence = "MEDIUM"
        coin_id = f"{chain}:{contract_address.lower()}"
    elif base_norm and chain and name:
        method = "normalized_base_symbol+chain+normalized_name"
        confidence = "MEDIUM"
        name_key = re.sub(r"\s+", "_", name.strip().upper())
        coin_id = f"{chain}:{base_norm}:{name_key}"
    elif base_norm and chain:
        method = "normalized_base_symbol+chain"
        confidence = "MEDIUM"
        coin_id = f"{chain}:{base_norm}"
    else:
        method = "normalized_base_symbol_only"
        confidence = "LOW_CONFIDENCE_IDENTITY"
        warnings.append("LOW_CONFIDENCE_IDENTITY")
        coin_id = f"SYM:{base_norm}" if base_norm else f"UNKNOWN:{row.get('asset_id') or 'NA'}"

    if pair_address and coin_id.lower() in {pair_address.lower(), f"{chain}:{pair_address.lower()}"}:
        method = "normalized_base_symbol+chain" if (base_norm and chain) else "normalized_base_symbol_only"
        confidence = "LOW_CONFIDENCE_IDENTITY"
        warnings.append("REFUSED_PAIR_ADDRESS_AS_IDENTITY")
        coin_id = f"{chain}:{base_norm}" if (base_norm and chain) else f"SYM:{base_norm or 'UNKNOWN'}"

    return {
        "coin_id": coin_id,
        "base_symbol": base_sym,
        "quote_symbol": quote_sym or "",
        "normalized_base_symbol": base_norm,
        "normalized_quote_symbol": quote_norm or "",
        "chain": chain,
        "identity_resolution_method": method,
        "identity_confidence": confidence,
        "identity_warnings": ",".join(warnings),
        "name": name,
        "token_address": token_address,
        "pair_address": pair_address,
    }
