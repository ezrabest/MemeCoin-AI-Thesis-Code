"""AE18 URL-first canonical market identity.

Canonical identity is provider_pair_url_exact (preserved casing).
pair_address / provider_pair_address are derived helpers only.
Display fields are preserved separately from canonical identity.
"""
from __future__ import annotations

from typing import Any

from app.clean_forward.display_identity import (
    classify_social_candidate,
    derive_symbol_pair_display,
    is_symbol_pair_available,
)
from app.clean_forward.display_resilience import apply_display_resilience
from app.clean_forward.price_source_identity import (
    DEFAULT_PROVIDER,
    cell,
    clean_provider_pair_url,
    extract_chain_and_pair_from_provider_url,
    is_internal_lineage_id,
    resolve_selected_target_identity,
    synthesize_dexscreener_url,
)
from app.clean_forward.provider_url_key import try_normalize_provider_pair_url_key
from app.clean_forward.market_activity import ACTIVE_PROVIDER_TXNS, evaluate_market_activity

CANONICAL_IDENTITY_TYPE = "PROVIDER_URL"


def extract_final_url_segment_exact(url: str) -> str:
    """Return the final non-empty path segment with original case preserved."""
    _, pair = extract_chain_and_pair_from_provider_url(url)
    return pair


def _short_addr_display(addr: str, *, n: int = 6) -> str:
    text = cell(addr)
    if not text:
        return ""
    if len(text) <= n * 2 + 1:
        return text
    return f"{text[:n]}…{text[-4:]}"


def _pick(*values: Any) -> str:
    for value in values:
        text = cell(value)
        if text:
            return text
    return ""


def _num_or_blank(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def resolve_canonical_market_identity(row: dict[str, Any]) -> dict[str, Any]:
    """Resolve URL-first canonical identity fields from a source row."""
    resolved = resolve_selected_target_identity(row)
    raw_url = cell(row.get("provider_pair_url") or row.get("provider_url") or row.get("open_chart_url"))
    provider_pair_url_exact = clean_provider_pair_url(raw_url) if raw_url else ""

    if not provider_pair_url_exact:
        chain = cell(resolved.get("display_chain") or row.get("chain"))
        pair = cell(resolved.get("display_real_pair_address"))
        if chain and pair and not is_internal_lineage_id(pair):
            provider_pair_url_exact = synthesize_dexscreener_url(chain, pair)

    final_segment = (
        extract_final_url_segment_exact(provider_pair_url_exact)
        if provider_pair_url_exact
        else ""
    )

    pair_derived = final_segment or cell(resolved.get("display_real_pair_address"))
    if is_internal_lineage_id(pair_derived):
        pair_derived = ""

    derivation_source = ""
    derivation_status = "MISSING"
    if pair_derived:
        if final_segment:
            derivation_source = "provider_pair_url_final_segment"
            derivation_status = "DERIVED_FROM_URL"
        elif cell(resolved.get("identity_resolution_method")):
            derivation_source = cell(resolved.get("identity_resolution_method"))
            derivation_status = "DERIVED_FROM_HELPER_FIELD"
        else:
            derivation_source = "unknown_helper"
            derivation_status = "DERIVED_FROM_HELPER_FIELD"

    rpc_address = pair_derived if pair_derived else ""
    rpc_source = derivation_source if rpc_address else ""

    canonical = provider_pair_url_exact
    case_preserved = bool(canonical and final_segment and canonical.endswith(final_segment))

    missingness = ""
    identity_status = "VALID"
    identity_source = cell(row.get("target_source") or row.get("canonical_identity_source"))
    if not canonical:
        identity_status = "MISSING_URL_IDENTITY"
        missingness = "provider_pair_url_exact_empty"
    elif not final_segment:
        identity_status = "INVALID_URL_SEGMENT"
        missingness = "provider_pair_url_final_segment_empty"

    price_usd = row.get("price_usd")
    mark_status = "PRICE_AVAILABLE" if price_usd not in (None, "") else "PRICE_NOT_AVAILABLE"
    price_failure = "" if mark_status == "PRICE_AVAILABLE" else "price_usd_missing_in_source_row"

    return {
        "provider_pair_url": provider_pair_url_exact,
        "provider_pair_url_exact": provider_pair_url_exact,
        "provider_pair_url_final_segment_exact": final_segment,
        "open_chart_url": provider_pair_url_exact,
        "canonical_market_identity": canonical,
        "canonical_market_identity_type": CANONICAL_IDENTITY_TYPE if canonical else "",
        "canonical_identity_status": identity_status,
        "canonical_identity_source": identity_source or cell(resolved.get("target_source")),
        "canonical_identity_case_preserved": case_preserved,
        "canonical_identity_missingness_reason": missingness,
        "pair_address_derived": pair_derived,
        "pair_address_derivation_source": derivation_source,
        "pair_address_derivation_status": derivation_status,
        "pair_address_for_rpc": rpc_address,
        "rpc_address_source": rpc_source,
        "chain": cell(resolved.get("display_chain") or row.get("chain") or row.get("provider_chain_id")),
        "provider": cell(resolved.get("provider") or row.get("provider") or DEFAULT_PROVIDER),
        "base_token_symbol": cell(
            resolved.get("base_token_symbol")
            or row.get("provider_base_token_symbol")
            or row.get("base_token_symbol")
        ),
        "quote_token_symbol": cell(
            resolved.get("quote_token_symbol")
            or row.get("provider_quote_token_symbol")
            or row.get("quote_token_symbol")
        ),
        "base_token_address_derived": cell(
            row.get("provider_base_token_address") or row.get("base_token_address_derived")
        ),
        "quote_token_address_derived": cell(
            row.get("provider_quote_token_address") or row.get("quote_token_address_derived")
        ),
        "price_source_key": cell(resolved.get("price_source_key")),
        "mark_price_lookup_key": canonical,
        "mark_price_lookup_status": mark_status,
        "price_resolution_failure_reason": price_failure,
        "identity_resolution_status": cell(resolved.get("identity_resolution_status")),
        "identity_resolution_method": cell(resolved.get("identity_resolution_method")),
    }


def build_symbol_pair_display(row: dict[str, Any], identity: dict[str, Any] | None = None) -> str:
    """Human display SYMBOL/PAIR — never canonical identity, never a raw address pair.

    Returns an explicit unavailable status instead of a base-only symbol or a
    raw address pair. See derive_symbol_pair_display for the full record.
    """
    return derive_symbol_pair_display(row, identity)["symbol_pair_display"]


def build_index_row(
    row: dict[str, Any],
    *,
    last_identity_rebuild_at: str,
    last_market_update_at: str | None = None,
) -> dict[str, Any]:
    """Build a runtime index row with identity + full display/market fields."""
    identity = resolve_canonical_market_identity(row)
    display = derive_symbol_pair_display(row, identity)
    social = classify_social_candidate(row)
    base_sym = _pick(display.get("base_token_symbol"), identity.get("base_token_symbol"))
    quote_sym = _pick(display.get("quote_token_symbol"), identity.get("quote_token_symbol"))
    symbol_pair = display["symbol_pair_display"]

    dex_id = _pick(row.get("provider_dex_id"), row.get("dex_id"), row.get("dex"))
    price = _num_or_blank(row.get("price_usd") if row.get("price_usd") not in (None, "") else row.get("price"))
    liquidity = _num_or_blank(row.get("liquidity_usd") if row.get("liquidity_usd") not in (None, "") else row.get("liquidity"))
    volume_h24 = _num_or_blank(row.get("volume_h24") or row.get("volume_24h"))
    whale = _num_or_blank(row.get("whale_score") or row.get("latest_whale_score"))

    has_url = bool(identity.get("canonical_market_identity"))
    has_price = price not in (None, "")
    norm_key, _ = try_normalize_provider_pair_url_key(
        cell(identity.get("provider_pair_url_exact") or identity.get("canonical_market_identity")),
        require_dexscreener=True,
    )

    out = {
        **identity,
        # Display identity (not canonical)
        "symbol_pair_display": symbol_pair,
        "symbol_pair_display_status": display["symbol_pair_display_status"],
        "symbol_pair_display_reason": display["symbol_pair_display_reason"],
        "symbol_pair_address_fallback": display["symbol_pair_address_fallback"],
        "base_token_symbol": base_sym,
        "quote_token_symbol": quote_sym,
        "provider_base_token_symbol": _pick(row.get("provider_base_token_symbol"), base_sym),
        "provider_quote_token_symbol": _pick(row.get("provider_quote_token_symbol"), quote_sym),
        "provider_base_token_name": cell(row.get("provider_base_token_name") or row.get("base_token_name")),
        "provider_quote_token_name": cell(row.get("provider_quote_token_name") or row.get("quote_token_name")),
        "provider_base_token_address": _pick(
            row.get("provider_base_token_address"), identity.get("base_token_address_derived")
        ),
        "provider_quote_token_address": _pick(
            row.get("provider_quote_token_address"), identity.get("quote_token_address_derived")
        ),
        "base_token_address_derived": identity.get("base_token_address_derived") or "",
        "quote_token_address_derived": identity.get("quote_token_address_derived") or "",
        "normalized_provider_pair_url_key": norm_key or "",
        "dex_id": dex_id,
        "provider_dex_id": dex_id,
        "target_source": social["target_source"] or cell(identity.get("canonical_identity_source")),
        # Social / manual curation classification (deterministic, source-based)
        "social_classification": social["social_classification"],
        "is_social_candidate": social["is_social_candidate"],
        "is_social_confirmed": social["is_social_confirmed"],
        "social_source": social["social_source"],
        "social_reason": social["social_reason"],
        "linked_sources": social["linked_sources"],
        "seed_collection": social["seed_collection"],
        "manual_curation_status": social["manual_curation_status"],
        # Market metrics — preserve source values when present
        "price_usd": price,
        "liquidity_usd": liquidity,
        "fdv": _num_or_blank(row.get("fdv")),
        "market_cap": _num_or_blank(row.get("market_cap")),
        "volume_m5": _num_or_blank(row.get("volume_m5")),
        "volume_h1": _num_or_blank(row.get("volume_h1")),
        "volume_h6": _num_or_blank(row.get("volume_h6")),
        "volume_h24": volume_h24,
        "txns_m5_buys": _num_or_blank(row.get("txns_m5_buys")),
        "txns_m5_sells": _num_or_blank(row.get("txns_m5_sells")),
        "txns_h1_buys": _num_or_blank(row.get("txns_h1_buys")),
        "txns_h1_sells": _num_or_blank(row.get("txns_h1_sells")),
        "txns_h6_buys": _num_or_blank(row.get("txns_h6_buys")),
        "txns_h6_sells": _num_or_blank(row.get("txns_h6_sells")),
        "txns_h24_buys": _num_or_blank(row.get("txns_h24_buys")),
        "txns_h24_sells": _num_or_blank(row.get("txns_h24_sells")),
        "price_change_m5": _num_or_blank(row.get("price_change_m5") or row.get("price_change_5m")),
        "price_change_h1": _num_or_blank(row.get("price_change_h1") or row.get("price_change_1h")),
        "price_change_h6": _num_or_blank(row.get("price_change_h6") or row.get("price_change_6h")),
        "price_change_h24": _num_or_blank(row.get("price_change_h24") or row.get("price_change_24h")),
        "pair_created_at": cell(row.get("pair_created_at")),
        "whale_score": whale,
        "semantic_status": social["semantic_status"]
        or cell(row.get("semantic_label") or row.get("semantic_signal_family")),
        "feed_status": cell(row.get("feed_status")) or ("INDEXED" if has_url else "MISSING_IDENTITY"),
        "freshness_status": cell(row.get("freshness_status")) or ("fresh" if has_price else "unknown"),
        "tradability_status": cell(row.get("tradability_status") or row.get("acceptance_status")),
        "verification_status": cell(
            row.get("verification_status") or row.get("acceptance_status") or ("INDEXED" if has_url else "")
        ),
        "last_market_update_at": last_market_update_at or cell(row.get("last_market_update_at")),
        "last_identity_rebuild_at": last_identity_rebuild_at,
        "identity_status": identity.get("canonical_identity_status") or "",
        "safe_for_price_lookup": has_url,
        "safe_for_rpc_derivation": bool(identity.get("pair_address_for_rpc")),
    }
    # Preserve any pre-set resilience statuses from refresh/rehydration.
    for fld in (
        "provider_resolution_status",
        "symbol_resolution_status",
        "market_data_status",
        "trade_readiness_status",
        "unresolved_reason",
        "display_provenance",
        "provider_resolution_reason",
    ):
        if row.get(fld) not in (None, ""):
            out[fld] = row.get(fld)
    out = apply_display_resilience(out, allow_cache_lookup=True)
    out["display_status"] = (
        out.get("display_metadata_status")
        or out.get("symbol_resolution_status")
        or out.get("symbol_pair_display_status")
        or ""
    )

    # Normalize timestamp aliases for all UI/API/bot consumers.
    # fetched_at / last_fetched means provider data was fetched/refreshed.
    # This does not assert that the numeric price changed.
    fetch_ts = (
        out.get("fetched_at")
        or out.get("last_fetched")
        or out.get("provider_fetch_at")
        or out.get("loaded_at")
        or out.get("last_market_update_at")
        or out.get("price_updated_at")
    )
    if fetch_ts:
        out["provider_fetch_at"] = fetch_ts
        out["market_data_refreshed_at"] = fetch_ts
        out["last_market_update_at"] = fetch_ts
        out["price_updated_at"] = fetch_ts

    activity = evaluate_market_activity(out)
    out.update(activity)

    # Missing symbols alone do not block trading. Market inactivity does.
    # Keep the functional readiness axis synchronized everywhere.
    if activity.get("market_activity_status") != ACTIVE_PROVIDER_TXNS:
        out["trade_readiness_status"] = activity.get("activity_trade_readiness_status")
        out["trade_block_reason"] = activity.get("activity_trade_block_reason")
    elif not out.get("trade_readiness_status"):
        out["trade_readiness_status"] = "PAPER_ELIGIBLE"

    return out
