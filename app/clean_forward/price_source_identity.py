"""Stable market identity helpers: URL suffix extraction and price_source_key.

Rules:
- ae16b_* / combined_target_id / candidate_id are lineage IDs, never pair addresses.
- Display fields preserve original case; normalized join keys are lowercase.
- URL suffix extraction strips query, fragment, and trailing slashes.
- Does not call DexScreener.
"""
from __future__ import annotations

from typing import Any
from urllib.parse import urlparse, urlunparse

INTERNAL_ID_PREFIX = "ae16b_"
DEFAULT_PROVIDER = "dexscreener"

PAIR_LIKE_FIELDS = (
    "pair_address",
    "resolved_pair_address",
    "user_supplied_pair_address",
    "refetch_pair_id",
    "provider_pair_address",
    "display_real_pair_address",
    "normalized_real_pair_address",
)


def cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_internal_lineage_id(value: Any) -> bool:
    text = cell(value)
    return bool(text) and text.lower().startswith(INTERNAL_ID_PREFIX)


def clean_provider_pair_url(url: str) -> str:
    """Strip whitespace, query, fragment, and trailing slashes from a provider URL."""
    raw = cell(url)
    if not raw:
        return ""
    parsed = urlparse(raw)
    path = parsed.path.rstrip("/")
    cleaned = urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
    return cleaned


def extract_chain_and_pair_from_provider_url(url: str) -> tuple[str, str]:
    """Extract (display_chain, display_real_pair_address) from a DexScreener-style URL.

    Final non-empty path segment = real_pair_address.
    Immediately previous path segment = chain.
    Query params, fragments, empty segments, and trailing slashes are excluded.
    """
    cleaned = clean_provider_pair_url(url)
    if not cleaned:
        return "", ""
    parsed = urlparse(cleaned)
    segments = [s for s in parsed.path.split("/") if s]
    if len(segments) < 2:
        if len(segments) == 1:
            return "", segments[0]
        return "", ""
    return segments[-2], segments[-1]


def synthesize_dexscreener_url(chain: str, pair_address: str) -> str:
    ch = cell(chain)
    pair = cell(pair_address)
    if not ch or not pair or is_internal_lineage_id(pair):
        return ""
    return f"https://dexscreener.com/{ch}/{pair}"


def build_price_source_key(
    provider: str,
    normalized_chain: str,
    normalized_real_pair_address: str,
) -> str:
    prov = cell(provider).lower() or DEFAULT_PROVIDER
    chain = cell(normalized_chain).lower()
    pair = cell(normalized_real_pair_address).lower()
    if not chain or not pair or is_internal_lineage_id(pair):
        return ""
    return f"{prov}|{chain}|{pair}"


def _first_nonempty(*values: Any) -> str:
    for value in values:
        text = cell(value)
        if text:
            return text
    return ""


def _reject_internal(value: str) -> str:
    return "" if is_internal_lineage_id(value) else value


def resolve_selected_target_identity(row: dict[str, Any]) -> dict[str, Any]:
    """Resolve a curated/selected target row into stable display + join identity.

    Prefer provider_pair_url suffix. Never use ae16b_* as real_pair_address.
    Fall back to provider_pair_address / refetch_pair_id / other pair fields.
    """
    selected_target_id = _first_nonempty(
        row.get("selected_target_id"),
        row.get("internal_target_id"),
        row.get("combined_target_id"),
        row.get("candidate_id"),
    )
    combined_target_id = cell(row.get("combined_target_id"))
    candidate_id = cell(row.get("candidate_id"))
    provider = _first_nonempty(row.get("provider"), DEFAULT_PROVIDER).lower() or DEFAULT_PROVIDER

    raw_url = _first_nonempty(row.get("provider_pair_url"), row.get("provider_url"))
    url_chain, url_pair = extract_chain_and_pair_from_provider_url(raw_url)

    pair_address_field = cell(row.get("pair_address"))
    pair_like_candidates = [
        ("provider_pair_url_suffix", url_pair),
        ("provider_pair_address", cell(row.get("provider_pair_address"))),
        ("refetch_pair_id", cell(row.get("refetch_pair_id"))),
        ("resolved_pair_address", cell(row.get("resolved_pair_address"))),
        ("user_supplied_pair_address", cell(row.get("user_supplied_pair_address"))),
        ("pair_address", pair_address_field),
    ]

    method = ""
    display_pair = ""
    for name, value in pair_like_candidates:
        cleaned = _reject_internal(value)
        if cleaned:
            display_pair = cleaned
            method = name
            break

    display_chain = _first_nonempty(
        url_chain if method == "provider_pair_url_suffix" else "",
        row.get("provider_chain_id"),
        row.get("chain"),
        row.get("display_chain"),
    )

    # If URL exists and pair_address was an internal id, URL wins (explicit rule).
    if raw_url and is_internal_lineage_id(pair_address_field) and url_pair and not is_internal_lineage_id(url_pair):
        display_pair = url_pair
        display_chain = url_chain or display_chain
        method = "provider_pair_url_overrides_ae16b_pair_address"

    provider_pair_url = clean_provider_pair_url(raw_url) if raw_url else ""
    if not provider_pair_url and display_chain and display_pair:
        provider_pair_url = synthesize_dexscreener_url(display_chain, display_pair)
        if method and "synthesized_url" not in method:
            method = f"{method}+synthesized_provider_pair_url"

    normalized_chain = display_chain.lower() if display_chain else ""
    normalized_pair = display_pair.lower() if display_pair else ""
    price_source_key = build_price_source_key(provider, normalized_chain, normalized_pair)

    error = ""
    status = "RESOLVED"
    if is_internal_lineage_id(display_pair) or (
        not display_pair
        and (
            is_internal_lineage_id(pair_address_field)
            or is_internal_lineage_id(combined_target_id)
        )
    ):
        if not display_pair:
            status = "UNRESOLVED_INTERNAL_ID_ONLY"
            error = "ae16b_internal_id_without_recoverable_provider_pair"
            method = method or "internal_id_misused_as_pair"
            display_pair = ""
            normalized_pair = ""
            price_source_key = ""
        else:
            # Should not happen after _reject_internal; defensive.
            status = "UNRESOLVED_INTERNAL_ID_ONLY"
            error = "ae16b_rejected_as_real_pair_address"
            display_pair = ""
            normalized_pair = ""
            price_source_key = ""
    elif not display_pair or not display_chain:
        status = "UNRESOLVED_MISSING_PAIR_OR_CHAIN"
        error = "missing_display_chain_or_real_pair_address"
        price_source_key = ""

    if method == "provider_pair_url_suffix" or method.startswith("provider_pair_url"):
        identity_resolution_method = method
    elif method:
        identity_resolution_method = method
    else:
        identity_resolution_method = "unresolved"

    ready_flag = _first_nonempty(row.get("clean_forward_candidate_ready"), "true").lower()
    default_selected = "ACTIVE" if ready_flag in {"1", "true", "yes", "y"} else "INACTIVE"
    selected_status = _first_nonempty(
        row.get("selected_status"),
        row.get("active_status"),
        row.get("acceptance_status"),
        default_selected,
    )

    return {
        "selected_target_id": selected_target_id,
        "internal_target_id": selected_target_id,
        "combined_target_id": combined_target_id,
        "candidate_id": candidate_id,
        "provider": provider,
        "display_chain": display_chain,
        "display_real_pair_address": display_pair,
        "normalized_chain": normalized_chain,
        "normalized_real_pair_address": normalized_pair,
        "provider_pair_url": provider_pair_url,
        "price_source_key": price_source_key,
        "base_token_symbol": _first_nonempty(
            row.get("base_token_symbol"),
            row.get("provider_base_token_symbol"),
        ),
        "quote_token_symbol": _first_nonempty(
            row.get("quote_token_symbol"),
            row.get("provider_quote_token_symbol"),
        ),
        "target_source": cell(row.get("target_source")),
        "selected_status": selected_status,
        "active_status": selected_status,
        "identity_resolution_method": identity_resolution_method,
        "identity_resolution_status": status,
        "identity_resolution_error": error,
    }


def scan_ae16b_pair_field_misuse(row: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any] | None:
    """Return an audit row if ae16b_* was misused as a pair identity (actual or prior-map class)."""
    misused_fields: list[str] = []
    for field in PAIR_LIKE_FIELDS:
        if is_internal_lineage_id(row.get(field)):
            misused_fields.append(field)

    combined = cell(row.get("combined_target_id"))
    # Empty/absent pair_address while combined_target_id is ae16b_* is the prior-map misuse class:
    # audits incorrectly treated combined_target_id as pair_address / price_source_key suffix.
    pair_address_empty = not cell(row.get("pair_address"))
    prior_map_misuse = bool(is_internal_lineage_id(combined) and pair_address_empty)

    if not misused_fields and not prior_map_misuse:
        return None

    corrected = bool(resolved.get("price_source_key")) and not is_internal_lineage_id(
        resolved.get("display_real_pair_address")
    )
    method = cell(resolved.get("identity_resolution_method"))
    url_corrected = corrected and "provider_pair_url" in method
    return {
        "selected_target_id": resolved.get("selected_target_id") or combined,
        "combined_target_id": combined,
        "misused_pair_like_fields": "|".join(misused_fields)
        if misused_fields
        else ("prior_map_used_combined_target_id_as_pair" if prior_map_misuse else ""),
        "provider_pair_url_present": "true"
        if cell(row.get("provider_pair_url") or row.get("provider_url"))
        else "false",
        "provider_pair_url_corrected_identity": "true" if url_corrected else "false",
        "corrected_by_provider_fields": "true" if corrected else "false",
        "corrected_display_real_pair_address": resolved.get("display_real_pair_address") or "",
        "corrected_normalized_real_pair_address": resolved.get("normalized_real_pair_address") or "",
        "corrected_price_source_key": resolved.get("price_source_key") or "",
        "identity_resolution_status": resolved.get("identity_resolution_status") or "",
        "unresolved": "true"
        if resolved.get("identity_resolution_status") == "UNRESOLVED_INTERNAL_ID_ONLY"
        else "false",
    }
