"""AE18 display identity — SYMBOL/PAIR derivation and social classification.

Display identity is never canonical identity. Raw token addresses may only
appear in address/detail columns, never as the primary SYMBOL/PAIR.
"""
from __future__ import annotations

from typing import Any

SYMBOL_PAIR_UNAVAILABLE = "SYMBOL_PAIR_UNAVAILABLE"
SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING = "SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING"
PARTIAL_PROVIDER_SYMBOLS_MISSING = "PARTIAL_PROVIDER_SYMBOLS_MISSING"

#: symbol_pair_display_status values
STATUS_FULL_PAIR = "FULL_PAIR"
STATUS_PARTIAL = PARTIAL_PROVIDER_SYMBOLS_MISSING
STATUS_SYMBOLS_MISSING = SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING
STATUS_UNAVAILABLE = SYMBOL_PAIR_UNAVAILABLE

#: Explicit statuses that are not real symbol pairs.
UNAVAILABLE_STATUSES = frozenset(
    {
        SYMBOL_PAIR_UNAVAILABLE,
        SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING,
        PARTIAL_PROVIDER_SYMBOLS_MISSING,
    }
)

#: Deterministic source tokens indicating direct social evidence.
_DIRECT_SOCIAL_TOKENS = (
    "social",
    "socialfi",
    "twitter",
    "telegram",
    "reddit",
    "discord",
    "manual_social",
)
#: Community-oriented curated collections (social group, explicitly labelled).
_COMMUNITY_SOCIAL_TOKENS = (
    "community",
    "community_dao",
    "fan_token",
    "dao",
)
#: Tokens that mark a confirmed social classification.
_CONFIRMED_SOCIAL_TOKENS = (
    "social_confirmed",
    "socialconfirmed",
)
#: Tokens marking manual/user curation.
_MANUAL_TOKENS = (
    "user_seed",
    "user_dexscreener_seed",
    "manual",
)

_SOCIAL_EVIDENCE_FIELDS = (
    "target_source",
    "seed_collection",
    "linked_sources",
    "semantic_status",
    "acceptance_status",
    "recovery_status",
    "manual_curation_status",
    "social_source",
    "clean_forward_candidate_ready",
)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def short_addr_display(addr: Any, *, n: int = 6) -> str:
    """Shortened address for address/detail columns only (never primary SYMBOL/PAIR)."""
    text = _cell(addr)
    if not text:
        return ""
    if len(text) <= n * 2 + 1:
        return text
    return f"{text[:n]}…{text[-4:]}"


def _pick(*values: Any) -> str:
    for value in values:
        text = _cell(value)
        if text and text not in UNAVAILABLE_STATUSES:
            return text
    return ""


def derive_symbol_pair_display(
    row: dict[str, Any],
    identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strict SYMBOL/PAIR derivation.

    Returns display, status, reason, and a details-only address fallback.
    Never returns a base-only symbol when a quote symbol/address exists, and
    never returns a raw address pair as the primary display.
    """
    ident = identity or {}
    base_sym = _pick(
        row.get("provider_base_token_symbol"),
        ident.get("base_token_symbol"),
        row.get("base_token_symbol"),
        row.get("base_symbol"),
    )
    quote_sym = _pick(
        row.get("provider_quote_token_symbol"),
        ident.get("quote_token_symbol"),
        row.get("quote_token_symbol"),
        row.get("quote_symbol"),
    )
    base_addr = _pick(
        row.get("provider_base_token_address"),
        ident.get("base_token_address_derived"),
        row.get("base_token_address_derived"),
        row.get("base_token_address"),
    )
    quote_addr = _pick(
        row.get("provider_quote_token_address"),
        ident.get("quote_token_address_derived"),
        row.get("quote_token_address_derived"),
        row.get("quote_token_address"),
    )

    address_pair_fallback = ""
    if base_addr and quote_addr:
        address_pair_fallback = f"{short_addr_display(base_addr)}/{short_addr_display(quote_addr)}"
    elif base_addr:
        address_pair_fallback = short_addr_display(base_addr)
    elif quote_addr:
        address_pair_fallback = short_addr_display(quote_addr)

    # 1 & 2: both symbols known
    if base_sym and quote_sym:
        return {
            "symbol_pair_display": f"{base_sym}/{quote_sym}",
            "symbol_pair_display_status": STATUS_FULL_PAIR,
            "symbol_pair_display_reason": "",
            "symbol_pair_address_fallback": address_pair_fallback,
            "base_token_symbol": base_sym,
            "quote_token_symbol": quote_sym,
        }

    # 3: exactly one side known after rehydration — explicit partial status.
    # A half-symbol / half-address string is never shown as the primary pair.
    if base_sym or quote_sym:
        known_side = "quote" if quote_sym else "base"
        return {
            "symbol_pair_display": PARTIAL_PROVIDER_SYMBOLS_MISSING,
            "symbol_pair_display_status": STATUS_PARTIAL,
            "symbol_pair_display_reason": (
                f"only_{known_side}_token_symbol_available_from_provider"
            ),
            "symbol_pair_address_fallback": address_pair_fallback,
            "symbol_pair_known_side_symbol": base_sym or quote_sym,
            "base_token_symbol": base_sym,
            "quote_token_symbol": quote_sym,
        }

    # 4: no symbols but addresses exist — details-only fallback, explicit status
    if base_addr or quote_addr:
        return {
            "symbol_pair_display": SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING,
            "symbol_pair_display_status": STATUS_SYMBOLS_MISSING,
            "symbol_pair_display_reason": "provider_token_symbols_missing_in_cache",
            "symbol_pair_address_fallback": address_pair_fallback,
            "base_token_symbol": "",
            "quote_token_symbol": quote_sym,
        }

    # 5: nothing usable
    return {
        "symbol_pair_display": SYMBOL_PAIR_UNAVAILABLE,
        "symbol_pair_display_status": STATUS_UNAVAILABLE,
        "symbol_pair_display_reason": "no_token_symbols_or_addresses_available",
        "symbol_pair_address_fallback": "",
        "base_token_symbol": "",
        "quote_token_symbol": "",
    }


def is_symbol_pair_available(display: Any) -> bool:
    return bool(_cell(display)) and _cell(display) not in UNAVAILABLE_STATUSES


def _evidence_blob(row: dict[str, Any]) -> str:
    parts = [_cell(row.get(field)) for field in _SOCIAL_EVIDENCE_FIELDS]
    return "|".join(p for p in parts if p).lower()


def classify_social_candidate(row: dict[str, Any]) -> dict[str, Any]:
    """Deterministic, source-based social/manual classification (no LLM)."""
    blob = _evidence_blob(row)
    matched_sources: list[str] = []
    for field in _SOCIAL_EVIDENCE_FIELDS:
        value = _cell(row.get(field)).lower()
        if not value:
            continue
        if any(tok in value for tok in _DIRECT_SOCIAL_TOKENS + _COMMUNITY_SOCIAL_TOKENS):
            matched_sources.append(f"{field}={_cell(row.get(field))}")

    has_direct = any(tok in blob for tok in _DIRECT_SOCIAL_TOKENS)
    has_community = any(tok in blob for tok in _COMMUNITY_SOCIAL_TOKENS)
    has_confirmed = any(tok in blob for tok in _CONFIRMED_SOCIAL_TOKENS)
    is_manual = any(tok in blob for tok in _MANUAL_TOKENS)

    if has_confirmed:
        classification = "SOCIAL_CONFIRMED"
        reason = "source_fields_indicate_social_confirmed"
    elif has_direct:
        classification = "SOCIAL_CANDIDATE_UNCONFIRMED"
        reason = "source_fields_indicate_social_seed"
    elif has_community:
        classification = "SOCIAL_COMMUNITY_ADJACENT"
        reason = "source_fields_indicate_community_or_fan_collection"
    elif is_manual:
        classification = "MANUAL_CURATED_NON_SOCIAL"
        reason = "manual_curation_without_social_source_evidence"
    else:
        classification = "NON_SOCIAL_OR_UNCLASSIFIED"
        reason = "no_social_source_evidence"

    is_social = classification in {
        "SOCIAL_CONFIRMED",
        "SOCIAL_CANDIDATE_UNCONFIRMED",
        "SOCIAL_COMMUNITY_ADJACENT",
    }

    if classification == "SOCIAL_CONFIRMED":
        semantic_status = "SOCIAL_CONFIRMED"
    elif is_social:
        semantic_status = "SOCIAL_CANDIDATE_UNCONFIRMED"
    else:
        semantic_status = _cell(row.get("semantic_status")) or "UNKNOWN_UNRESOLVED"

    return {
        "social_classification": classification,
        "is_social_candidate": is_social,
        "is_social_confirmed": classification == "SOCIAL_CONFIRMED",
        "social_source": "|".join(matched_sources),
        "social_reason": reason,
        "semantic_status": semantic_status,
        "linked_sources": _cell(row.get("linked_sources")),
        "seed_collection": _cell(row.get("seed_collection")),
        "target_source": _cell(row.get("target_source")),
        "manual_curation_status": _cell(row.get("manual_curation_status"))
        or ("USER_MANUAL_SEED" if is_manual else ""),
    }
