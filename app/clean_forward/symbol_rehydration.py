"""AE18 provider symbol rehydration — cold rebuild / manual refresh only.

Fills missing provider display fields (symbols, names, dex, metrics) for rows
that already have a canonical provider URL identity. The DexScreener lookup key
is derived from the canonical URL itself, so pair_address is never a canonical
or primary lookup key. Never runs on a UI GET path.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx


from app.ae13b_product import provider_refresh_errors as refresh_errors
from app.clean_forward.price_source_identity import (
    cell,
    clean_provider_pair_url,
    extract_chain_and_pair_from_provider_url,
)

log = logging.getLogger("ae18.symbol_rehydration")

DEXSCREENER_API_BASE = "https://api.dexscreener.com"
_DEX_HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}
_DEX_TIMEOUT = httpx.Timeout(12.0)


#: Fields populated from a provider pair payload (display/market only).
REHYDRATED_FIELDS = (
    "provider_base_token_symbol",
    "provider_quote_token_symbol",
    "provider_base_token_name",
    "provider_quote_token_name",
    "provider_base_token_address",
    "provider_quote_token_address",
    "provider_dex_id",
    "price_usd",
    "liquidity_usd",
    "fdv",
    "market_cap",
    "volume_m5",
    "volume_h1",
    "volume_h6",
    "volume_h24",
    "txns_m5_buys",
    "txns_m5_sells",
    "txns_h1_buys",
    "txns_h1_sells",
    "txns_h6_buys",
    "txns_h6_sells",
    "txns_h24_buys",
    "txns_h24_sells",
    "price_change_m5",
    "price_change_h1",
    "price_change_h6",
    "price_change_h24",
    "pair_created_at",
)


def row_provider_url(row: dict[str, Any]) -> str:
    return cell(
        row.get("provider_pair_url_exact")
        or row.get("canonical_market_identity")
        or row.get("provider_pair_url")
        or row.get("provider_url")
        or row.get("open_chart_url")
    )


def row_needs_symbol_rehydration(row: dict[str, Any]) -> bool:
    """True when a canonical URL exists but provider symbols are incomplete."""
    if not row_provider_url(row):
        return False
    base = cell(row.get("provider_base_token_symbol") or row.get("base_token_symbol"))
    quote = cell(row.get("provider_quote_token_symbol") or row.get("quote_token_symbol"))
    return not (base and quote)


def _txns(raw: dict[str, Any], window: str, side: str) -> Any:
    txns = raw.get("txns")
    if not isinstance(txns, dict):
        return None
    entry = txns.get(window)
    if not isinstance(entry, dict):
        return None
    return entry.get(side)


def _nested(raw: dict[str, Any], key: str, sub: str) -> Any:
    entry = raw.get(key)
    if isinstance(entry, dict):
        return entry.get(sub)
    return None


def map_provider_pair_payload(raw: dict[str, Any], *, symbols_only: bool = False) -> dict[str, Any]:
    """Map a DexScreener pair object onto runtime index display fields.

    pairAddress is mapped to pair_address_derived only — never to canonical
    identity. pair.url is returned separately for equivalence checking.

    When symbols_only=True, only token identity/display fields are mapped.
    This is used for token-pairs fallback where the provider confirms token
    symbols but does not prove that the returned pool is the exact canonical
    URL market.
    """
    if not isinstance(raw, dict):
        return {}
    base = raw.get("baseToken") if isinstance(raw.get("baseToken"), dict) else {}
    quote = raw.get("quoteToken") if isinstance(raw.get("quoteToken"), dict) else {}
    mapped: dict[str, Any] = {
        "provider_base_token_symbol": base.get("symbol"),
        "provider_quote_token_symbol": quote.get("symbol"),
        "provider_base_token_name": base.get("name"),
        "provider_quote_token_name": quote.get("name"),
        "provider_base_token_address": base.get("address"),
        "provider_quote_token_address": quote.get("address"),
        "_provider_reported_url": raw.get("url"),
    }
    if symbols_only:
        return {k: v for k, v in mapped.items() if v not in (None, "")}

    mapped.update({
        "provider_dex_id": raw.get("dexId"),
        "pair_address_derived": raw.get("pairAddress"),
        "price_usd": raw.get("priceUsd"),
        "liquidity_usd": _nested(raw, "liquidity", "usd"),
        "fdv": raw.get("fdv"),
        "market_cap": raw.get("marketCap"),
        "volume_m5": _nested(raw, "volume", "m5"),
        "volume_h1": _nested(raw, "volume", "h1"),
        "volume_h6": _nested(raw, "volume", "h6"),
        "volume_h24": _nested(raw, "volume", "h24"),
        "txns_m5_buys": _txns(raw, "m5", "buys"),
        "txns_m5_sells": _txns(raw, "m5", "sells"),
        "txns_h1_buys": _txns(raw, "h1", "buys"),
        "txns_h1_sells": _txns(raw, "h1", "sells"),
        "txns_h6_buys": _txns(raw, "h6", "buys"),
        "txns_h6_sells": _txns(raw, "h6", "sells"),
        "txns_h24_buys": _txns(raw, "h24", "buys"),
        "txns_h24_sells": _txns(raw, "h24", "sells"),
        "price_change_m5": _nested(raw, "priceChange", "m5"),
        "price_change_h1": _nested(raw, "priceChange", "h1"),
        "price_change_h6": _nested(raw, "priceChange", "h6"),
        "price_change_h24": _nested(raw, "priceChange", "h24"),
        "pair_created_at": raw.get("pairCreatedAt"),
    })
    return {k: v for k, v in mapped.items() if v not in (None, "")}


def _map_verify_result(result_dict: dict[str, Any]) -> dict[str, Any]:
    """Fallback mapping from the flattened pair-verify result."""
    aliases = {
        "provider_base_token_symbol": "base_token_symbol",
        "provider_quote_token_symbol": "quote_token_symbol",
        "provider_base_token_name": "base_token_name",
        "provider_quote_token_name": "quote_token_name",
        "provider_base_token_address": "base_token_address",
        "provider_quote_token_address": "quote_token_address",
        "provider_dex_id": "dex_id",
        "price_usd": "price_usd",
        "liquidity_usd": "liquidity_usd",
        "volume_m5": "volume_5m",
        "volume_h1": "volume_1h",
        "volume_h6": "volume_6h",
        "volume_h24": "volume_24h",
        "txns_m5_buys": "txns_5m_buys",
        "txns_m5_sells": "txns_5m_sells",
        "txns_h1_buys": "txns_1h_buys",
        "txns_h1_sells": "txns_1h_sells",
        "txns_h24_buys": "txns_24h_buys",
        "txns_h24_sells": "txns_24h_sells",
        "price_change_m5": "price_change_5m",
        "price_change_h1": "price_change_1h",
        "price_change_h6": "price_change_6h",
        "price_change_h24": "price_change_24h",
        "pair_created_at": "pair_created_at",
        "pair_address_derived": "pair_address",
    }
    out: dict[str, Any] = {}
    for dest, src in aliases.items():
        value = result_dict.get(src)
        if value not in (None, ""):
            out[dest] = value
    return out


def _urls_equivalent(a: str, b: str) -> bool:
    return clean_provider_pair_url(a).strip().lower() == clean_provider_pair_url(b).strip().lower()


def _http_get_json(path: str) -> dict[str, Any]:
    """GET a DexScreener API path for explicit cold/manual rehydration only."""
    from app.runtime.shutdown import CONTROLLED_SHUTDOWN_SKIP, should_skip_network
    from app.runtime.ui_get_network_guard import is_ui_get_path_active, record_network_attempt

    if is_ui_get_path_active():
        record_network_attempt("dexscreener")
        return {"ok": False, "status_code": 0, "data": None, "error": "ui_get_network_forbidden"}
    if should_skip_network(context=f"symbol_rehydration:{path}"):
        return {"ok": False, "status_code": 0, "data": None, "error": CONTROLLED_SHUTDOWN_SKIP}

    try:
        with httpx.Client(timeout=_DEX_TIMEOUT) as client:
            response = client.get(f"{DEXSCREENER_API_BASE}{path}", headers=_DEX_HEADERS)
        status = response.status_code
        if status == 429:
            return {"ok": False, "status_code": status, "data": None, "error": "too_many_requests"}
        if status >= 500:
            return {"ok": False, "status_code": status, "data": None, "error": f"provider_5xx_{status}"}
        if status >= 400:
            return {"ok": False, "status_code": status, "data": None, "error": f"http_{status}"}
        return {"ok": True, "status_code": status, "data": response.json(), "error": ""}
    except httpx.TimeoutException as exc:
        return {"ok": False, "status_code": None, "data": None, "error": f"timeout:{exc}"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status_code": None, "data": None, "error": f"network:{exc}"}


def _pairs_from_payload(data: Any) -> list[dict[str, Any]]:
    """Normalize DexScreener pair/token-pairs/tokens/search payloads to pair objects."""
    if not data:
        return []
    if isinstance(data, list):
        return [p for p in data if isinstance(p, dict)]
    if isinstance(data, dict):
        pair = data.get("pair")
        if isinstance(pair, dict):
            return [pair]
        pairs = data.get("pairs")
        if isinstance(pairs, list):
            return [p for p in pairs if isinstance(p, dict)]
        if data.get("pairAddress") or data.get("baseToken") or data.get("quoteToken"):
            return [data]
    return []


def _token_address_set(row: dict[str, Any]) -> set[str]:
    values = {
        cell(row.get("provider_base_token_address")),
        cell(row.get("provider_quote_token_address")),
        cell(row.get("base_token_address_derived")),
        cell(row.get("quote_token_address_derived")),
        cell(row.get("base_token_address")),
        cell(row.get("quote_token_address")),
    }
    return {v.lower() for v in values if v}


def _pair_token_address_set(pair: dict[str, Any]) -> set[str]:
    base = pair.get("baseToken") if isinstance(pair.get("baseToken"), dict) else {}
    quote = pair.get("quoteToken") if isinstance(pair.get("quoteToken"), dict) else {}
    values = {cell(base.get("address")), cell(quote.get("address"))}
    return {v.lower() for v in values if v}


def _pair_has_symbols(pair: dict[str, Any]) -> bool:
    base = pair.get("baseToken") if isinstance(pair.get("baseToken"), dict) else {}
    quote = pair.get("quoteToken") if isinstance(pair.get("quoteToken"), dict) else {}
    return bool(cell(base.get("symbol")) and cell(quote.get("symbol")))


def _canonical_match_kind(row: dict[str, Any], pair: dict[str, Any], *, segment: str) -> str:
    """Return exact/symbol-only suitability of a provider pair object.

    EXACT_* means the payload can update full market metrics. TOKEN_ADDRESS_PAIR_MATCH
    means it can update token symbols/names/addresses only, because a token-pairs
    endpoint may return another market for the same token pair.
    """
    provider_url = row_provider_url(row)
    returned_url = cell(pair.get("url"))
    returned_pair = cell(pair.get("pairAddress"))
    if returned_url and provider_url and _urls_equivalent(returned_url, provider_url):
        return "EXACT_PROVIDER_URL_MATCH"
    if returned_pair and segment and returned_pair.lower() == segment.lower():
        return "EXACT_PAIR_ID_MATCH"
    known_tokens = _token_address_set(row)
    returned_tokens = _pair_token_address_set(pair)
    if len(known_tokens) >= 2 and known_tokens.issubset(returned_tokens):
        return "TOKEN_ADDRESS_PAIR_MATCH"
    return "NO_MATCH"


def _pick_best_pair_from_payload(
    row: dict[str, Any],
    data: Any,
    *,
    segment: str,
) -> tuple[dict[str, Any] | None, str, bool]:
    """Pick a pair and whether only symbol fields are safe to use."""
    candidates = [p for p in _pairs_from_payload(data) if _pair_has_symbols(p)]
    exact: list[tuple[str, dict[str, Any]]] = []
    token_match: list[tuple[str, dict[str, Any]]] = []
    for pair in candidates:
        kind = _canonical_match_kind(row, pair, segment=segment)
        if kind.startswith("EXACT_"):
            exact.append((kind, pair))
        elif kind == "TOKEN_ADDRESS_PAIR_MATCH":
            token_match.append((kind, pair))
    if exact:
        return exact[0][1], exact[0][0], False
    if token_match:
        return token_match[0][1], token_match[0][0], True
    return None, "NO_MATCH", False


def _lookup_pair_payload_multistage(
    row: dict[str, Any],
    *,
    chain: str,
    segment: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Multi-stage provider resolver for display symbols.

    Stages:
    1. Direct pair endpoint without EVM/Solana address-format prevalidation.
    2. token-pairs endpoint for base and quote token addresses.
    3. tokens endpoint for both token addresses.

    The resolver never changes canonical identity. Non-exact token-address matches
    are symbol-only, so market price/liquidity/delta fields are not overwritten
    from a different pool.
    """
    attempts: list[dict[str, Any]] = []

    def attempt(path: str, method: str) -> tuple[dict[str, Any] | None, str, bool]:
        result = _http_get_json(path)
        attempts.append({"method": method, "path": path, "status_code": result.get("status_code"), "error": result.get("error")})
        if not result.get("ok"):
            return None, str(result.get("error") or "PROVIDER_HTTP_ERROR"), False
        pair, match_kind, symbols_only = _pick_best_pair_from_payload(row, result.get("data"), segment=segment)
        if pair:
            return pair, match_kind, symbols_only
        return None, "NO_MATCHING_PAIR_WITH_SYMBOLS", False

    pair, match_kind, symbols_only = attempt(f"/latest/dex/pairs/{chain}/{segment}", "pair_by_url_segment")
    if pair:
        return map_provider_pair_payload(pair, symbols_only=symbols_only), {
            "success": True,
            "method": "pair_by_url_segment",
            "match_kind": match_kind,
            "symbols_only": symbols_only,
            "attempts": attempts,
        }

    token_addresses = [cell(row.get("provider_base_token_address")), cell(row.get("provider_quote_token_address"))]
    seen: set[str] = set()
    for token in [t for t in token_addresses if t]:
        token_key = token.lower()
        if token_key in seen:
            continue
        seen.add(token_key)
        pair, match_kind, symbols_only = attempt(f"/token-pairs/v1/{chain}/{token}", f"token_pairs:{token}")
        if pair:
            return map_provider_pair_payload(pair, symbols_only=symbols_only), {
                "success": True,
                "method": f"token_pairs:{token}",
                "match_kind": match_kind,
                "symbols_only": symbols_only,
                "attempts": attempts,
            }

    token_csv = ",".join([t for t in token_addresses if t])
    if token_csv:
        pair, match_kind, symbols_only = attempt(f"/tokens/v1/{chain}/{token_csv}", "tokens_batch")
        if pair:
            return map_provider_pair_payload(pair, symbols_only=symbols_only), {
                "success": True,
                "method": "tokens_batch",
                "match_kind": match_kind,
                "symbols_only": symbols_only,
                "attempts": attempts,
            }

    return {}, {
        "success": False,
        "method": "multi_stage_provider_symbol_resolver",
        "match_kind": "NO_MATCH",
        "symbols_only": False,
        "attempts": attempts,
    }


def apply_rehydration_payload(row: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    """Merge provider payload into a source row, preserving canonical URL casing."""
    out = dict(row)
    canonical = row_provider_url(row)
    reported = cell(payload.pop("_provider_reported_url", ""))
    derived = payload.pop("pair_address_derived", None)
    if derived not in (None, ""):
        # Derived helper only. Canonical identity stays the provider URL.
        out["pair_address_derived"] = derived
        out.setdefault("provider_pair_address", derived)

    for key, value in payload.items():
        if value in (None, ""):
            continue
        out[key] = value

    # pair.url adopted only when trusted and equivalent to canonical identity.
    if canonical:
        out["provider_pair_url"] = clean_provider_pair_url(canonical)
        out["provider_pair_url_trusted_equivalent"] = bool(
            reported and _urls_equivalent(reported, canonical)
        )
    return out


def rehydrate_row_symbols(
    row: dict[str, Any],
    *,
    use_cache: bool = True,
    validator: Any = None,
) -> dict[str, Any]:
    """Rehydrate one row from the provider. Explicit cold/manual paths only.

    Returns {row, attempted, success, failure_reason, failure_code, provider_url}.
    """
    from app.runtime.shutdown import is_shutting_down
    from app.runtime.ui_get_network_guard import is_ui_get_path_active

    url = row_provider_url(row)
    outcome: dict[str, Any] = {
        "row": dict(row),
        "provider_url": url,
        "attempted": False,
        "success": False,
        "failure_code": "",
        "failure_reason": "",
    }

    if not url:
        outcome["failure_code"] = refresh_errors.PROVIDER_URL_MISSING
        outcome["failure_reason"] = "row has no canonical provider pair URL"
        return outcome

    if is_ui_get_path_active():
        outcome["failure_code"] = refresh_errors.PROVIDER_REFRESH_DISABLED
        outcome["failure_reason"] = "symbol rehydration is forbidden on UI GET paths"
        return outcome

    if is_shutting_down():
        outcome["failure_code"] = refresh_errors.CONTROLLED_SHUTDOWN_SKIP
        outcome["failure_reason"] = "shutdown in progress"
        return outcome

    chain, segment = extract_chain_and_pair_from_provider_url(url)
    chain = chain or cell(row.get("chain") or row.get("provider_chain_id"))
    if not chain or not segment:
        outcome["failure_code"] = refresh_errors.IDENTITY_UNRESOLVED
        outcome["failure_reason"] = f"chain/segment not extractable from canonical URL: {url}"
        return outcome

    explicit_validator_supplied = validator is not None
    if validator is None:
        from app.ae13b_product.dexscreener_pair_verify import validate_dexscreener_pair

        validator = validate_dexscreener_pair

    outcome["attempted"] = True
    data: dict[str, Any] = {}
    primary_failure = ""
    try:
        result = validator(chain, segment, use_cache=use_cache)
        data = result.to_dict(include_raw=True) if hasattr(result, "to_dict") else dict(result or {})
    except Exception as exc:  # noqa: BLE001 - classified below
        primary_failure = f"{type(exc).__name__}: {str(exc)[:200]}"

    raw_pair = data.get("raw_pair") if isinstance(data.get("raw_pair"), dict) else None

    payload = map_provider_pair_payload(raw_pair) if raw_pair else {}
    if not payload:
        payload = _map_verify_result(data)

    resolver_meta: dict[str, Any] = {}
    # The legacy verifier performs chain/address-family prevalidation and may
    # reject provider URL final segments that are valid DexScreener market IDs
    # but not canonical on-chain pair addresses. For URL-first identity, fall
    # through to provider-URL and token-address based resolver instead of
    # treating that as final.
    if (
        not explicit_validator_supplied
        and not (cell(payload.get("provider_base_token_symbol")) and cell(payload.get("provider_quote_token_symbol")))
    ):
        payload, resolver_meta = _lookup_pair_payload_multistage(row, chain=chain, segment=segment)

    if not payload:
        status = cell(data.get("verification_status")) or "provider_response_empty"
        http_status = data.get("verification_http_status")
        code = refresh_errors.classify_refresh_exception(None, http_status=http_status)
        if code == refresh_errors.UNKNOWN_PROVIDER_REFRESH_ERROR:
            code = (
                refresh_errors.PROVIDER_PAIR_NOT_FOUND
                if "not_found" in status or resolver_meta.get("attempts")
                else refresh_errors.PROVIDER_RESPONSE_EMPTY
            )
        outcome["failure_code"] = code
        outcome["failure_reason"] = (
            f"verification_status={status}"
            + (f" http_status={http_status}" if http_status else "")
            + (f" error={cell(data.get('verification_error'))}" if data.get("verification_error") else "")
            + (f" primary_error={primary_failure}" if primary_failure else "")
            + (f" resolver_attempts={resolver_meta.get('attempts')}" if resolver_meta.get("attempts") else "")
        )
        return outcome

    updated = apply_rehydration_payload(row, payload)
    if resolver_meta:
        updated["symbol_rehydration_method"] = resolver_meta.get("method", "")
        updated["symbol_rehydration_match_kind"] = resolver_meta.get("match_kind", "")
        updated["symbol_rehydration_symbols_only"] = bool(resolver_meta.get("symbols_only"))
    outcome["row"] = updated
    base = cell(updated.get("provider_base_token_symbol"))
    quote = cell(updated.get("provider_quote_token_symbol"))
    if base and quote:
        outcome["success"] = True
    else:
        outcome["failure_code"] = refresh_errors.PROVIDER_RESPONSE_EMPTY
        missing = "base" if not base else "quote"
        outcome["failure_reason"] = (
            f"provider payload did not contain {missing}Token.symbol "
            f"(verification_status={cell(data.get('verification_status'))}; "
            f"resolver_method={resolver_meta.get('method', '')}; "
            f"resolver_match={resolver_meta.get('match_kind', '')})"
        )
    return outcome


def rehydrate_rows(
    rows: list[dict[str, Any]],
    *,
    enabled: bool,
    use_cache: bool = True,
    validator: Any = None,
    stop_check: Any = None,
) -> dict[str, Any]:
    """Rehydrate all rows needing provider symbols. Returns rows + audit stats."""
    out_rows: list[dict[str, Any]] = []
    needed = 0
    attempted = 0
    success = 0
    failed = 0
    failed_urls: list[str] = []
    failed_reasons: dict[str, int] = {}
    per_row: list[dict[str, Any]] = []

    for row in rows:
        working = dict(row)
        needs = row_needs_symbol_rehydration(working)
        record = {
            "canonical_market_identity": row_provider_url(working),
            "rehydration_needed": needs,
            "rehydration_attempted": False,
            "rehydration_success": False,
            "rehydration_failure_reason": "",
        }
        if needs:
            needed += 1
        if needs and enabled:
            if stop_check is not None and stop_check():
                record["rehydration_failure_reason"] = refresh_errors.CONTROLLED_SHUTDOWN_SKIP
                per_row.append(record)
                out_rows.append(working)
                continue
            outcome = rehydrate_row_symbols(working, use_cache=use_cache, validator=validator)
            working = outcome["row"]
            record["rehydration_attempted"] = outcome["attempted"]
            record["rehydration_success"] = outcome["success"]
            if outcome["attempted"]:
                attempted += 1
            if outcome["success"]:
                success += 1
            else:
                failed += 1
                reason = f"{outcome['failure_code']}: {outcome['failure_reason']}"
                record["rehydration_failure_reason"] = reason
                failed_urls.append(outcome["provider_url"])
                failed_reasons[outcome["failure_code"] or "UNKNOWN"] = (
                    failed_reasons.get(outcome["failure_code"] or "UNKNOWN", 0) + 1
                )
                log.info("symbol rehydration failed: %s (%s)", outcome["provider_url"], reason)
        per_row.append(record)
        out_rows.append(working)

    return {
        "rows": out_rows,
        "per_row": per_row,
        "rows_rehydration_needed": needed,
        "dex_rehydration_enabled": bool(enabled),
        "dex_rehydration_attempted_count": attempted,
        "dex_rehydration_success_count": success,
        "dex_rehydration_failed_count": failed,
        "failed_rehydration_urls": failed_urls,
        "failed_rehydration_reasons": failed_reasons,
    }
