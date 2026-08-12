"""AE18 display resilience — last-good + manual override (display only).

Applies after provider derivation. Read-only: never writes caches or audits,
so it is safe on UI GET paths. Display state never feeds trade blocking.
"""
from __future__ import annotations

from typing import Any

from app.clean_forward.last_good_display_cache import lookup_last_good_display
from app.clean_forward.manual_display_overrides import lookup_manual_override
from app.clean_forward.provider_resilience_statuses import (
    DISPLAY_READY,
    MARKET_DATA_AVAILABLE_SYMBOLS_MISSING,
    MARKET_DATA_READY,
    PROVIDER_PAIR_NOT_FOUND,
    RESOLVED,
    RESOLVED_WITH_LAST_GOOD_DISPLAY,
    RESOLVED_WITH_MANUAL_DISPLAY_OVERRIDE,
    SYMBOL_PAIR_FROM_LAST_GOOD,
    SYMBOL_PAIR_FROM_MANUAL_OVERRIDE,
    SYMBOL_PAIR_RESOLVED,
    SYMBOL_PAIR_UNAVAILABLE_AFTER_PROVIDER_PROBE,
    SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING,
    UNRESOLVED,
    block_reason_for,
    classify_display_metadata_status,
    classify_identity_readiness,
    classify_market_data_status,
    classify_trade_readiness,
    is_proper_symbol_pair_display,
)
from app.clean_forward.provider_url_key import try_normalize_provider_pair_url_key

#: Fields written by apply_display_resilience — used by surfaces to copy state.
RESILIENCE_FIELDS = (
    "normalized_provider_pair_url_key",
    "symbol_pair_display",
    "symbol_pair_display_status",
    "symbol_pair_display_reason",
    "display_metadata_status",
    "provider_resolution_status",
    "symbol_resolution_status",
    "market_data_status",
    "identity_readiness_status",
    "trade_readiness_status",
    "trade_block_reason",
    "unresolved_reason",
    "display_provenance",
    "provider_base_token_symbol",
    "provider_quote_token_symbol",
    "base_token_symbol",
    "quote_token_symbol",
)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _looks_like_raw_address(text: str) -> bool:
    value = _cell(text)
    if not value or "/" in value:
        return False
    if value.startswith("0x") and len(value) >= 40:
        return True
    return len(value) >= 32 and value.isalnum()


def apply_display_resilience(
    row: dict[str, Any],
    *,
    overrides: dict[str, dict[str, Any]] | None = None,
    allow_cache_lookup: bool = True,
    provider_probe_attempted: bool = False,
) -> dict[str, Any]:
    """Attach resilience statuses, optionally filling display from override/last-good.

    Never writes caches or audits. Safe to call on GET paths.
    """
    # The probe flag is persisted on the row so re-deriving the index does not
    # downgrade an "after provider probe" status back to "cache missing".
    provider_probe_attempted = bool(provider_probe_attempted) or bool(
        row.get("provider_probe_attempted")
    )
    if provider_probe_attempted:
        row["provider_probe_attempted"] = True

    exact = _cell(
        row.get("provider_pair_url_exact")
        or row.get("canonical_market_identity")
        or row.get("provider_pair_url")
    )
    key, _ = try_normalize_provider_pair_url_key(exact, require_dexscreener=True)
    if key:
        row["normalized_provider_pair_url_key"] = key

    display = _cell(row.get("symbol_pair_display"))
    has_proper = is_proper_symbol_pair_display(display)
    from_last_good = False
    from_override = False

    price = row.get("price_usd") if row.get("price_usd") not in (None, "") else row.get("price")
    try:
        has_price = price not in (None, "") and float(price) > 0
    except (TypeError, ValueError):
        has_price = False

    provenance_parts: list[str] = []
    provider_status = _cell(row.get("provider_resolution_status"))
    symbol_status = _cell(row.get("symbol_resolution_status"))

    # Manual override has highest display priority when display is missing.
    if allow_cache_lookup and exact and not has_proper:
        override = None
        try:
            override = lookup_manual_override(exact, overrides=overrides)
        except Exception:
            override = None
        if override and is_proper_symbol_pair_display(override.get("symbol_pair_display")):
            row["symbol_pair_display"] = override["symbol_pair_display"]
            if override.get("provider_base_token_symbol"):
                row["provider_base_token_symbol"] = override["provider_base_token_symbol"]
                row["base_token_symbol"] = override["provider_base_token_symbol"]
            if override.get("provider_quote_token_symbol"):
                row["provider_quote_token_symbol"] = override["provider_quote_token_symbol"]
                row["quote_token_symbol"] = override["provider_quote_token_symbol"]
            row["symbol_pair_display_status"] = "FULL_PAIR"
            row["symbol_pair_display_reason"] = "manual_display_override"
            provider_status = RESOLVED_WITH_MANUAL_DISPLAY_OVERRIDE
            symbol_status = SYMBOL_PAIR_FROM_MANUAL_OVERRIDE
            provenance_parts.append("manual_display_override")
            has_proper = True
            from_override = True

    # Last-good display cache (display only).
    if allow_cache_lookup and exact and not has_proper:
        last_good = None
        try:
            last_good = lookup_last_good_display(exact)
        except Exception:
            last_good = None
        if last_good and is_proper_symbol_pair_display(last_good.get("symbol_pair_display")):
            row["symbol_pair_display"] = last_good["symbol_pair_display"]
            for fld in (
                "provider_base_token_symbol",
                "provider_quote_token_symbol",
                "provider_base_token_address",
                "provider_quote_token_address",
                "provider_dex_id",
                "chain",
            ):
                if last_good.get(fld) and not row.get(fld):
                    row[fld] = last_good[fld]
            if last_good.get("provider_base_token_symbol"):
                row["base_token_symbol"] = last_good["provider_base_token_symbol"]
            if last_good.get("provider_quote_token_symbol"):
                row["quote_token_symbol"] = last_good["provider_quote_token_symbol"]
            row["symbol_pair_display_status"] = "FULL_PAIR"
            row["symbol_pair_display_reason"] = "last_good_display_cache"
            provider_status = RESOLVED_WITH_LAST_GOOD_DISPLAY
            symbol_status = SYMBOL_PAIR_FROM_LAST_GOOD
            provenance_parts.append("last_good_display")
            has_proper = True
            from_last_good = True

    if has_proper and not (from_last_good or from_override):
        symbol_status = SYMBOL_PAIR_RESOLVED
        if not provider_status or provider_status == UNRESOLVED:
            provider_status = RESOLVED
        provenance_parts.append("provider_or_index_symbols")
    elif not has_proper:
        if has_price:
            provider_status = MARKET_DATA_AVAILABLE_SYMBOLS_MISSING
            symbol_status = (
                SYMBOL_PAIR_UNAVAILABLE_AFTER_PROVIDER_PROBE
                if provider_probe_attempted
                else SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING
            )
        elif provider_probe_attempted:
            provider_status = provider_status or PROVIDER_PAIR_NOT_FOUND
            symbol_status = SYMBOL_PAIR_UNAVAILABLE_AFTER_PROVIDER_PROBE
        else:
            provider_status = provider_status or UNRESOLVED
            symbol_status = SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING
        # A raw address must never survive as the primary SYMBOL/PAIR.
        if _looks_like_raw_address(row.get("symbol_pair_display")):
            row["symbol_pair_display"] = SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING
            row["symbol_pair_display_status"] = SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING
        provenance_parts.append(
            _cell(row.get("symbol_pair_display_reason")) or "symbols_unavailable"
        )

    if provider_status == RESOLVED and not has_proper and has_price:
        provider_status = MARKET_DATA_AVAILABLE_SYMBOLS_MISSING

    # Display metadata status is display-only and never blocks trading.
    row["display_metadata_status"] = classify_display_metadata_status(
        has_proper_display=has_proper,
        from_last_good=from_last_good,
        from_manual_override=from_override,
    )

    identity_status = classify_identity_readiness(row)
    market_status = classify_market_data_status(row)
    trade_status = classify_trade_readiness(
        market_data_status=market_status,
        identity_readiness_status=identity_status,
        position_continuity_safe=True,
    )

    row["provider_resolution_status"] = provider_status
    row["symbol_resolution_status"] = symbol_status
    row["market_data_status"] = market_status
    row["identity_readiness_status"] = identity_status
    row["trade_readiness_status"] = trade_status
    row["trade_block_reason"] = block_reason_for(trade_status)
    row["display_provenance"] = "|".join(p for p in provenance_parts if p)
    row["unresolved_reason"] = (
        ""
        if has_proper
        else (
            _cell(row.get("symbol_pair_display_reason"))
            or _cell(row.get("provider_resolution_reason"))
            or "provider_symbols_unavailable"
        )
    )
    return row
