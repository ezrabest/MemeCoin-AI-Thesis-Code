"""Clean Forward eligibility and identity-separation validation."""
from __future__ import annotations

from typing import Any

from app.clean_forward.identity import (
    build_instrument_identity,
    is_evm_chain,
    normalize_address_for_chain,
    pair_address_for_id,
)

CLEAN_FEED_ELIGIBILITY_RULES = {
    "verification_status": "provider_pair_verified",
    "freshness_status": "fresh",
    "identity_status": "pair_and_tokens_separated",
    "shown_as_token_contract": False,
    "paper_demo_only": True,
    "live_trading_ready": False,
}


def _as_bool(value: Any, default: bool | None = None) -> bool | None:
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


def evaluate_clean_feed_eligibility(row: dict[str, Any]) -> dict[str, Any]:
    """Apply AE15 clean-feed eligibility rule. Local row fields only."""
    reasons: list[str] = []
    verification_status = str(row.get("verification_status") or "")
    freshness_status = str(row.get("freshness_status") or "")
    identity_status = str(row.get("identity_status") or "")
    shown_as_token_contract = _as_bool(row.get("shown_as_token_contract"), False)
    paper_demo_only = _as_bool(row.get("paper_demo_only"), False)
    live_trading_ready = _as_bool(row.get("live_trading_ready"), True)

    if verification_status != CLEAN_FEED_ELIGIBILITY_RULES["verification_status"]:
        reasons.append("verification_status_not_provider_pair_verified")
    if freshness_status != CLEAN_FEED_ELIGIBILITY_RULES["freshness_status"]:
        reasons.append("freshness_status_not_fresh")
    if identity_status != CLEAN_FEED_ELIGIBILITY_RULES["identity_status"]:
        reasons.append("identity_status_not_pair_and_tokens_separated")
    if shown_as_token_contract is True:
        reasons.append("shown_as_token_contract_true")
    if paper_demo_only is not True:
        reasons.append("paper_demo_only_false")
    if live_trading_ready is True:
        reasons.append("live_trading_ready_true")

    eligible = len(reasons) == 0
    return {
        "clean_feed_eligible": eligible,
        "rejection_reasons": reasons,
        "verification_status": verification_status,
        "freshness_status": freshness_status,
        "identity_status": identity_status,
        "shown_as_token_contract": shown_as_token_contract,
        "paper_demo_only": paper_demo_only,
        "live_trading_ready": live_trading_ready,
    }


def validate_identity_separation(row: dict[str, Any]) -> dict[str, Any]:
    """Validate pair/token separation and chain-aware address normalization."""
    identity = build_instrument_identity(row)
    chain = identity.chain
    pair = identity.pair_address
    base = identity.base_token_address
    quote = identity.quote_token_address

    pair_norm = pair_address_for_id(pair, chain=chain)
    base_norm = normalize_address_for_chain(base, chain=chain)
    quote_norm = normalize_address_for_chain(quote, chain=chain)

    failures: list[str] = []
    if not pair:
        failures.append("pair_address_missing")
    if not base:
        failures.append("base_token_address_missing")
    if not quote:
        failures.append("quote_token_address_missing")

    if pair and base and pair_norm == base_norm:
        failures.append("pair_address_equals_base_token_address")
    if pair and quote and pair_norm == quote_norm:
        failures.append("pair_address_equals_quote_token_address")
    if base and quote and base_norm == quote_norm:
        failures.append("base_token_address_equals_quote_token_address")

    # coin_id must not be invented for clean-forward rows
    coin_id = row.get("coin_id")
    coin_id_invented = coin_id is not None and str(coin_id).strip() != ""
    if coin_id_invented:
        failures.append("coin_id_invented")

    if identity.shown_as_token_contract:
        failures.append("shown_as_token_contract")

    if identity.identity_status and identity.identity_status != "pair_and_tokens_separated":
        failures.append("identity_status_not_pair_and_tokens_separated")

    # Solana must preserve case; EVM may lowercase
    solana_case_preserved = True
    evm_lowercased = True
    if chain == "solana" and pair:
        solana_case_preserved = pair == pair.strip() and pair_norm == pair.strip()
        if pair != pair_norm:
            failures.append("solana_pair_address_case_mutated")
            solana_case_preserved = False
    if is_evm_chain(chain) and pair:
        expected = pair.strip().lower()
        evm_lowercased = pair_norm == expected
        if not evm_lowercased:
            failures.append("evm_pair_address_not_lowercased")

    passed = len(failures) == 0
    return {
        "passed": passed,
        "failures": failures,
        "pair_address_present": bool(pair),
        "base_token_address_present": bool(base),
        "quote_token_address_present": bool(quote),
        "pair_address_ne_base_token_address": bool(pair and base and pair_norm != base_norm),
        "pair_address_ne_quote_token_address": bool(pair and quote and pair_norm != quote_norm),
        "base_token_address_ne_quote_token_address": bool(base and quote and base_norm != quote_norm),
        "coin_id_not_invented": not coin_id_invented,
        "coin_id": coin_id,
        "chain": chain,
        "pair_address": pair,
        "pair_address_normalized": pair_norm,
        "base_token_address": base,
        "quote_token_address": quote,
        "solana_case_preserved": solana_case_preserved if chain == "solana" else None,
        "evm_lowercased": evm_lowercased if is_evm_chain(chain) else None,
        "identity": identity.to_dict(),
    }
