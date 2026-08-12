"""Address role classifier for AE13I."""
from __future__ import annotations

from typing import Any

ADDRESS_ROLES = (
    "token_mint",
    "token_contract",
    "pair_contract",
    "pool_address",
    "market_account",
    "token_account",
    "provider_pair_id",
    "explorer_account",
    "unknown_or_provider_pair",
    "ambiguous",
)

SOLANA_CHAINS = {"solana", "sol", "solana-mainnet"}
EVM_CHAINS = {
    "ethereum", "eth", "base", "bsc", "bnb", "arbitrum",
    "polygon", "avalanche", "optimism", "robinhood", "robinhood_chain",
}


def _norm_chain(chain: Any) -> str:
    return str(chain or "").strip().lower()


def _is_evm_hex(addr: str) -> bool:
    a = str(addr or "").strip()
    return a.startswith("0x") and len(a) == 42


def _is_solana_base58(addr: str) -> bool:
    a = str(addr or "").strip()
    if not a or a.startswith("0x"):
        return False
    return 32 <= len(a) <= 44


def classify_address_role(
    *,
    chain: str | None = None,
    address: str | None = None,
    pair_address: str | None = None,
    pool_address: str | None = None,
    token_mint_address: str | None = None,
    token_contract_address: str | None = None,
    provider_pair_id: str | None = None,
    hint: str | None = None,
    source_field: str | None = None,
) -> dict[str, Any]:
    """Classify address roles. Does not invent identity."""
    chain_n = _norm_chain(chain)
    hint_n = str(hint or source_field or "").strip().lower()

    token_mint = (token_mint_address or "").strip() or None
    token_contract = (token_contract_address or "").strip() or None
    pair = (pair_address or "").strip() or None
    pool = (pool_address or "").strip() or None
    provider_id = (provider_pair_id or "").strip() or None
    primary = (address or pair or pool or token_mint or token_contract or "").strip() or None

    role = "unknown_or_provider_pair"
    role_note = "Address role not fully resolved."
    is_ambiguous = False

    if hint_n in ("token_mint", "mint"):
        role = "token_mint"
        token_mint = token_mint or primary
        role_note = "Classified as token mint from explicit hint."
    elif hint_n in ("token_contract", "token", "contract"):
        role = "token_contract"
        token_contract = token_contract or primary
        role_note = "Classified as token contract from explicit hint."
    elif hint_n in ("pair", "pair_address", "pair_contract", "lp"):
        role = "pair_contract" if chain_n in EVM_CHAINS or _is_evm_hex(primary or "") else "pool_address"
        pair = pair or primary
        if role == "pool_address":
            pool = pool or primary
        role_note = "Classified as pair/pool from explicit hint."
    elif hint_n in ("pool", "pool_address", "market_account"):
        role = "pool_address" if chain_n in SOLANA_CHAINS or _is_solana_base58(primary or "") else "pair_contract"
        pool = pool or primary
        pair = pair or primary
        role_note = "Classified as pool/market account from explicit hint."
    elif hint_n in ("provider_pair_id", "dexscreener", "provider"):
        role = "provider_pair_id"
        provider_id = provider_id or primary
        role_note = "Provider pair id — not necessarily on-chain token mint."
    elif pair and token_mint and pair == token_mint:
        role = "ambiguous"
        is_ambiguous = True
        role_note = "Same value used as both pair and token mint — ambiguous."
    elif pair and token_contract and pair.lower() == token_contract.lower():
        role = "ambiguous"
        is_ambiguous = True
        role_note = "Same value used as both pair and token contract — ambiguous."
    elif pair or pool:
        if chain_n in SOLANA_CHAINS or _is_solana_base58(pair or pool or ""):
            role = "pool_address"
            pool = pool or pair
            role_note = "This is a pool/pair address, not necessarily the token mint."
        elif chain_n in EVM_CHAINS or _is_evm_hex(pair or pool or ""):
            role = "pair_contract"
            role_note = "This is a pair/pool contract address, not necessarily the token contract."
        else:
            role = "unknown_or_provider_pair"
            role_note = "Pair/pool address present; chain role uncertain."
    elif token_mint:
        role = "token_mint"
        role_note = "Token mint address only; pair/pool not resolved."
    elif token_contract:
        role = "token_contract"
        role_note = "Token contract address only; pair/pool not resolved."
    elif provider_id:
        role = "provider_pair_id"
        role_note = "Only provider pair id available."
    elif primary:
        if chain_n in SOLANA_CHAINS or _is_solana_base58(primary):
            role = "unknown_or_provider_pair"
            pair = pair or primary
            role_note = "Solana address without role proof — treat as provider pair/pool candidate."
        elif _is_evm_hex(primary):
            role = "ambiguous"
            is_ambiguous = True
            role_note = "EVM address without role proof — ambiguous token vs pair."
        else:
            role = "ambiguous"
            is_ambiguous = True
            role_note = "Unrecognized address form."
    else:
        role = "ambiguous"
        is_ambiguous = True
        role_note = "No address fields available."

    conflict = False
    if pair and token_mint and pair == token_mint:
        conflict = True
        is_ambiguous = True
        role = "ambiguous"
    if pair and token_contract and pair.lower() == str(token_contract).lower():
        conflict = True
        is_ambiguous = True
        role = "ambiguous"

    tradable_identity_ok = (
        role in ("pair_contract", "pool_address", "market_account", "provider_pair_id")
        and bool(pair or pool or provider_id)
        and not conflict
        and not is_ambiguous
    )

    return {
        "address_role": role,
        "address_role_status": (
            "ambiguous"
            if is_ambiguous
            else ("ok" if role != "unknown_or_provider_pair" else "uncertain")
        ),
        "address_role_note": role_note,
        "token_mint_address": token_mint,
        "token_contract_address": token_contract,
        "pair_address": pair,
        "pool_address": pool,
        "market_account": pool if role in ("pool_address", "market_account") else None,
        "provider_pair_id": provider_id,
        "primary_address": primary,
        "pair_token_identity_conflict": conflict,
        "is_ambiguous": is_ambiguous,
        "tradable_identity_ok": tradable_identity_ok,
        "ui_warning": (
            "This is a pool/pair address, not necessarily the token mint."
            if role in ("pool_address", "pair_contract", "market_account")
            else None
        ),
        "paper_demo_only": True,
        "not_live_approved": True,
    }


def enrich_row_with_address_role(row: dict[str, Any]) -> dict[str, Any]:
    """Attach address-role fields onto a market/candidate/position row (copy)."""
    out = dict(row or {})
    classified = classify_address_role(
        chain=out.get("chain") or out.get("network"),
        address=out.get("address") or out.get("contract_address"),
        pair_address=out.get("pair_address") or out.get("matched_pair_address"),
        pool_address=out.get("pool_address"),
        token_mint_address=out.get("token_mint_address") or out.get("token_address"),
        token_contract_address=out.get("token_contract_address")
        or out.get("token_address")
        or out.get("base_token_address"),
        provider_pair_id=out.get("provider_pair_id") or out.get("pair_id"),
        hint=out.get("address_role_hint") or out.get("address_role"),
        source_field=out.get("address_source_field"),
    )
    out.update(classified)
    if out.get("pair_address") and out.get("token_mint_address") == out.get("pair_address"):
        if classified.get("address_role") in ("pool_address", "pair_contract"):
            out["token_mint_address"] = None
            out["contract_address_ambiguous"] = True
    return out