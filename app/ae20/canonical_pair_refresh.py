
"""AE20 canonical 45 exact-pair provider refresh.

Refresh/enrich the existing Clean Forward canonical universe.
This module must not replace candidate selection with provider-feed display rows.

AE20_VALIDATOR_GAP_DIRECT_PROVIDER_FALLBACK_V1:
If local format validation rejects a pair but direct DexScreener pair lookup
returns a valid pair payload, treat it as provider verified via explicit fallback.
The canonical AE20/AE16 identity key is preserved during AE20; any provider-returned
case-corrected pairAddress is only recorded as a canonicalization candidate.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from app.ae13b_product.dexscreener_pair_verify import (
    _http_get_pair,
    validate_dexscreener_pair,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cell(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in {"", "nan", "NaN", "None", "null"}:
        return ""
    return text


def _parse_dexscreener_exact_url(url: str) -> tuple[str, str]:
    parsed = urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if parsed.netloc != "dexscreener.com" or len(parts) < 2:
        return "", ""
    return parts[0].strip(), parts[1].strip()


def _getattr_any(obj: Any, names: list[str]) -> Any:
    for name in names:
        if hasattr(obj, name):
            value = getattr(obj, name)
            if value not in (None, ""):
                return value
    return None


def _pair_payload_price(pair_payload: dict[str, Any]) -> Any:
    return (
        pair_payload.get("priceUsd")
        or pair_payload.get("price_usd")
        or pair_payload.get("price")
    )


def _pair_payload_liquidity(pair_payload: dict[str, Any]) -> Any:
    liquidity = pair_payload.get("liquidity") or {}
    if isinstance(liquidity, dict):
        return liquidity.get("usd") or liquidity.get("liquidity_usd")
    return pair_payload.get("liquidity_usd") or pair_payload.get("liquidity")


def refresh_canonical_clean_forward_rows(
    rows: list[dict[str, Any]],
    *,
    use_cache: bool = False,
    max_rows: int | None = None,
) -> dict[str, Any]:
    started_at = _utc_now()

    out_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []

    attempted = 0
    verified = 0
    failed = 0
    skipped = 0
    price_updated = 0
    liquidity_updated = 0
    identity_mutation_count = 0
    validator_gap_direct_fallback_verified = 0
    provider_pair_case_mismatch_count = 0

    selected = rows[: max_rows or len(rows)]

    for idx, row in enumerate(selected, start=1):
        r = dict(row)

        exact_url_before = _cell(
            r.get("provider_pair_url_exact")
            or r.get("canonical_market_identity")
            or r.get("raw_provider_pair_url")
        )
        chain, pair = _parse_dexscreener_exact_url(exact_url_before)

        audit = {
            "row_idx": idx,
            "provider_pair_url_exact": exact_url_before,
            "chain": chain,
            "pair_address_from_exact_url": pair,
            "attempted": False,
            "lookup_ok": False,
            "verification_status": "",
            "http_status": "",
            "error": "",
            "validator_gap_direct_fallback_used": False,
            "provider_returned_pair_address": "",
            "provider_returned_url": "",
            "provider_pair_case_mismatch_detected": False,
            "identity_preserved": True,
            "price_before": r.get("price_usd"),
            "price_after": r.get("price_usd"),
            "liquidity_before": r.get("liquidity_usd"),
            "liquidity_after": r.get("liquidity_usd"),
        }

        if not chain or not pair:
            skipped += 1
            audit["verification_status"] = "SKIPPED_BAD_EXACT_URL"
            r["ae20_canonical_pair_refresh_status"] = "SKIPPED_BAD_EXACT_URL"
            r["ae20_canonical_pair_refresh_lookup_ok"] = False
            audit_rows.append(audit)
            out_rows.append(r)
            continue

        attempted += 1
        audit["attempted"] = True

        try:
            res = validate_dexscreener_pair(chain, pair, use_cache=use_cache)

            lookup_ok = bool(getattr(res, "lookup_ok", False))
            status = _cell(getattr(res, "verification_status", ""))
            http_status = getattr(res, "verification_http_status", "")
            err = _cell(getattr(res, "verification_error", ""))
            provider_display_url = _cell(getattr(res, "provider_pair_url", ""))

            direct_pair_payload: dict[str, Any] = {}
            fallback_used = False
            provider_returned_pair_address = ""
            provider_returned_url = ""

            if (not lookup_ok) and status == "chain_address_format_mismatch":
                direct_resp = _http_get_pair(chain, pair)
                direct_pair_payload = direct_resp.get("pair") or {}

                provider_returned_pair_address = _cell(
                    direct_pair_payload.get("pairAddress")
                    or direct_pair_payload.get("pair_address")
                )
                provider_returned_url = _cell(direct_pair_payload.get("url"))

                if direct_resp.get("ok") and direct_pair_payload:
                    lookup_ok = True
                    status = "provider_pair_verified_direct_fallback_validator_gap"
                    http_status = direct_resp.get("status_code")
                    err = ""
                    provider_display_url = provider_returned_url or provider_display_url
                    fallback_used = True
                    validator_gap_direct_fallback_verified += 1

                    audit["validator_gap_direct_fallback_used"] = True
                    audit["provider_returned_pair_address"] = provider_returned_pair_address
                    audit["provider_returned_url"] = provider_returned_url

                    if provider_returned_pair_address and provider_returned_pair_address != pair:
                        audit["provider_pair_case_mismatch_detected"] = True
                        provider_pair_case_mismatch_count += 1
                        r["ae20_provider_pair_case_mismatch_detected"] = True
                        r["ae20_provider_returned_pair_address"] = provider_returned_pair_address
                        r["ae20_provider_pair_url_exact_canonicalization_candidate"] = (
                            f"https://dexscreener.com/{chain}/{provider_returned_pair_address}"
                        )

            audit["lookup_ok"] = lookup_ok
            audit["verification_status"] = status
            audit["http_status"] = http_status
            audit["error"] = err

            r["ae20_canonical_pair_refresh_attempted"] = True
            r["ae20_canonical_pair_refresh_status"] = status
            r["ae20_canonical_pair_refresh_lookup_ok"] = lookup_ok
            r["ae20_canonical_pair_refresh_http_status"] = http_status
            r["ae20_canonical_pair_refresh_error"] = err
            r["ae20_canonical_pair_refresh_at"] = _utc_now()
            r["ae20_validator_gap_direct_fallback_used"] = fallback_used
            r["ae20_canonical_pair_refresh_source"] = (
                "direct_provider_fallback_after_validator_gap"
                if fallback_used
                else "validate_dexscreener_pair_exact_chain_pair"
            )

            # Preserve current AE20/AE16 identity during this stage.
            r["provider_pair_url_exact"] = exact_url_before
            r["canonical_market_identity"] = exact_url_before
            r["raw_provider_pair_url"] = exact_url_before
            r["pair_address"] = pair
            r["chain"] = chain

            if provider_display_url and provider_display_url != exact_url_before:
                r["provider_pair_url_display"] = provider_display_url

            price = _getattr_any(res, ["price_usd", "price"])
            if direct_pair_payload:
                price = _pair_payload_price(direct_pair_payload) or price
            if price not in (None, ""):
                r["price_usd"] = price
                audit["price_after"] = price
                if str(audit["price_before"]) != str(price):
                    price_updated += 1

            liq = _getattr_any(res, ["liquidity_usd", "liquidity"])
            if direct_pair_payload:
                liq = _pair_payload_liquidity(direct_pair_payload) or liq
            if liq not in (None, ""):
                r["liquidity_usd"] = liq
                audit["liquidity_after"] = liq
                if str(audit["liquidity_before"]) != str(liq):
                    liquidity_updated += 1

            fetched_at = _getattr_any(res, ["fetched_at", "observed_at", "last_fetched"])
            if fetched_at not in (None, ""):
                r["fetched_at"] = fetched_at
                r["observed_at"] = fetched_at
                r["ingested_at"] = fetched_at

            if lookup_ok:
                verified += 1
            else:
                failed += 1

        except Exception as exc:
            failed += 1
            audit["verification_status"] = "EXCEPTION"
            audit["error"] = f"{type(exc).__name__}: {exc}"
            r["ae20_canonical_pair_refresh_attempted"] = True
            r["ae20_canonical_pair_refresh_status"] = "EXCEPTION"
            r["ae20_canonical_pair_refresh_lookup_ok"] = False
            r["ae20_canonical_pair_refresh_error"] = audit["error"]

        exact_url_after = _cell(
            r.get("provider_pair_url_exact")
            or r.get("canonical_market_identity")
            or r.get("raw_provider_pair_url")
        )
        if exact_url_after != exact_url_before:
            identity_mutation_count += 1
            audit["identity_preserved"] = False

        audit_rows.append(audit)
        out_rows.append(r)

    if max_rows is not None and max_rows < len(rows):
        out_rows.extend(rows[max_rows:])

    summary = {
        "refresh_mode": "canonical_45_exact_pair_verify_with_validator_gap_fallback",
        "started_at_utc": started_at,
        "completed_at_utc": _utc_now(),
        "input_rows": len(rows),
        "processed_rows": len(selected),
        "attempted_rows": attempted,
        "verified_rows": verified,
        "failed_rows": failed,
        "skipped_rows": skipped,
        "price_updated_rows": price_updated,
        "liquidity_updated_rows": liquidity_updated,
        "validator_gap_direct_fallback_verified_rows": validator_gap_direct_fallback_verified,
        "provider_pair_case_mismatch_count": provider_pair_case_mismatch_count,
        "identity_mutation_count": identity_mutation_count,
        "identity_preserved": identity_mutation_count == 0,
        "candidate_universe_replaced": False,
        "candidate_universe_source": "canonical_market_identity_index",
        "provider_refresh_enrichment_only": True,
        "lowercase_join_used": False,
        "casefold_join_used": False,
        "case_insensitive_join_used": False,
        "symbol_only_join_used": False,
    }

    return {
        "rows": out_rows,
        "audit_rows": audit_rows,
        "summary": summary,
    }
