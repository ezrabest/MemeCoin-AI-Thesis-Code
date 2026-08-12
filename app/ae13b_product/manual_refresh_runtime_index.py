"""AE18 manual refresh — explicit POST-only provider refresh updating runtime index.

URL-first identity. Never promotes pair_address to canonical.
Respects shutdown cancellation. Atomic index update.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.clean_forward.canonical_market_identity import build_index_row
from app.clean_forward.display_identity import is_symbol_pair_available
from app.clean_forward.last_good_display_cache import upsert_last_good_display
from app.clean_forward.provider_resilience_statuses import is_proper_symbol_pair_display
from app.clean_forward.symbol_rehydration import (
    rehydrate_row_symbols,
    row_needs_symbol_rehydration,
)
from app.clean_forward.price_source_identity import cell, clean_provider_pair_url
from app.clean_forward.runtime_identity_index import (
    INDEX_CSV_PATH,
    INDEX_JSONL_PATH,
    RuntimeIndexValidationError,
    load_runtime_identity_index,
    write_runtime_index_validated,
)
from app.ae13b_product import provider_refresh_errors as errors
from app.ae13b_product.provider_refresh_errors import build_refresh_failure, summarize_failures
from app.runtime.shutdown import CONTROLLED_SHUTDOWN_SKIP, is_shutting_down, should_skip_network

log = logging.getLogger("ae18.manual_refresh")


class ProviderRefreshError(RuntimeError):
    """Carries a structured provider refresh failure object."""

    def __init__(self, failure: dict[str, Any]):
        super().__init__(failure.get("refresh_error_code", errors.UNKNOWN_PROVIDER_REFRESH_ERROR))
        self.failure = failure

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SEED_CSV = ROOT / "data" / "SeedTargets" / "clean_forward_curated_ready_targets_active.csv"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_seed_rows(path: Path) -> list[dict[str, str]]:
    import csv

    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _merge_verify_into_row(
    src: dict[str, Any], verified: dict[str, Any], *, overwrite: bool = False
) -> dict[str, Any]:
    """Merge DexScreener verify payload into source row without changing URL casing.

    With overwrite=True (force refresh) provider values replace stale/degraded
    cached display fields instead of only filling blanks.
    """
    out = dict(src)
    # Preserve exact URL — never lowercase final segment
    url = cell(src.get("provider_pair_url") or src.get("provider_url") or verified.get("provider_pair_url"))
    if url:
        out["provider_pair_url"] = clean_provider_pair_url(url)

    mapping = {
        "provider_dex_id": ("dex_id", "provider_dex_id"),
        "provider_base_token_symbol": ("base_token_symbol", "base_symbol"),
        "provider_quote_token_symbol": ("quote_token_symbol", "quote_symbol"),
        "provider_base_token_address": ("base_token_address",),
        "provider_quote_token_address": ("quote_token_address",),
        "provider_base_token_name": ("base_token_name",),
        "provider_quote_token_name": ("quote_token_name",),
        "price_usd": ("price_usd",),
        "liquidity_usd": ("liquidity_usd",),
        "volume_h24": ("volume_24h", "volume_h24"),
        "txns_h24_buys": ("txns_24h_buys",),
        "txns_h24_sells": ("txns_24h_sells",),
        "price_change_m5": ("price_change_m5", "price_change_5m"),
        "price_change_h1": ("price_change_h1", "price_change_1h"),
        "price_change_h6": ("price_change_h6", "price_change_6h"),
        "price_change_h24": ("price_change_h24", "price_change_24h"),
        "freshness_status": ("freshness_status",),
        "verification_status": ("verification_status", "status"),
        "tradability_status": ("tradability_status",),
    }
    for dest, keys in mapping.items():
        if cell(out.get(dest)) and not overwrite:
            continue
        for k in keys:
            if verified.get(k) not in (None, ""):
                out[dest] = verified.get(k)
                break

    # txns nested dict fallback
    tx = verified.get("txns_24h")
    if isinstance(tx, dict):
        if not cell(out.get("txns_h24_buys")) and tx.get("buys") is not None:
            out["txns_h24_buys"] = tx.get("buys")
        if not cell(out.get("txns_h24_sells")) and tx.get("sells") is not None:
            out["txns_h24_sells"] = tx.get("sells")
    return out


def _verify_one_row(
    src: dict[str, Any], *, use_cache: bool, overwrite: bool = False
) -> dict[str, Any]:
    """Verify one row against the provider.

    Raises ProviderRefreshError carrying a structured failure object so callers
    never surface a generic "aborted without reason" message.
    """
    url = cell(src.get("provider_pair_url") or src.get("provider_url"))
    chain = cell(src.get("provider_chain_id") or src.get("chain"))

    if should_skip_network(context="manual_refresh_row"):
        raise ProviderRefreshError(
            build_refresh_failure(
                error_code=errors.CONTROLLED_SHUTDOWN_SKIP,
                provider_url=url,
                chain=chain,
                shutdown_event_set=True,
            )
        )

    from app.ae13b_product.dexscreener_pair_verify import validate_dexscreener_pair

    # Derived helper for RPC/API only — never canonical
    pair = cell(
        src.get("provider_pair_address")
        or src.get("refetch_pair_id")
        or src.get("pair_address_derived")
    )
    if url:
        from app.clean_forward.price_source_identity import extract_chain_and_pair_from_provider_url

        u_chain, u_pair = extract_chain_and_pair_from_provider_url(url)
        chain = chain or u_chain
        pair = pair or u_pair

    if not url and not (chain and pair):
        raise ProviderRefreshError(
            build_refresh_failure(
                error_code=errors.PROVIDER_URL_MISSING,
                provider_url=url,
                chain=chain,
                reason="row has neither provider pair URL nor chain+derived helper address",
            )
        )
    if not chain or not pair:
        raise ProviderRefreshError(
            build_refresh_failure(
                error_code=errors.IDENTITY_UNRESOLVED,
                provider_url=url,
                chain=chain,
                reason="chain or derived pair helper could not be extracted from the market URL",
            )
        )

    endpoint = f"https://api.dexscreener.com/latest/dex/pairs/{chain}/{pair}"
    try:
        result = validate_dexscreener_pair(chain, pair, use_cache=use_cache)
    except Exception as exc:  # noqa: BLE001 - classified into structured failure
        raise ProviderRefreshError(
            build_refresh_failure(
                error_code=errors.classify_refresh_exception(exc),
                provider_url=url,
                chain=chain,
                attempted_endpoint=endpoint,
                exception=exc,
                shutdown_event_set=is_shutting_down(),
            )
        ) from exc

    d = result.to_dict(include_raw=True) if hasattr(result, "to_dict") else {}
    if not d:
        raise ProviderRefreshError(
            build_refresh_failure(
                error_code=errors.PROVIDER_RESPONSE_EMPTY,
                provider_url=url,
                chain=chain,
                attempted_endpoint=endpoint,
            )
        )

    status = str(d.get("status") or d.get("verification_status") or "").lower()
    if "not_found" in status or status == "provider_pair_not_found":
        raise ProviderRefreshError(
            build_refresh_failure(
                error_code=errors.PROVIDER_PAIR_NOT_FOUND,
                provider_url=url,
                chain=chain,
                attempted_endpoint=endpoint,
                reason=f"provider verification status={status}",
            )
        )

    # Ensure URL casing from source URL wins over any lowercased API address
    if url:
        d["provider_pair_url"] = clean_provider_pair_url(url)
    else:
        # Synthesize with exact pair case from helper (display), not lowercased
        d["provider_pair_url"] = f"https://dexscreener.com/{chain}/{pair}"
    return _merge_verify_into_row(src, d, overwrite=overwrite)


def manual_refresh_runtime_index(
    *,
    force: bool = False,
    clear_cache: bool = False,
    max_rows: int | None = None,
    seed_csv: Path | None = None,
    allow_dexscreener: bool = True,
    refresh_requested_by: str = "ui_manual_refresh",
) -> dict[str, Any]:
    """Explicit manual refresh path — may call DexScreener; updates runtime index atomically."""
    started = _utc_now()
    meta: dict[str, Any] = {
        "refresh_mode": "force_provider_refresh" if clear_cache or force else "manual_refresh",
        "refresh_requested_by": refresh_requested_by,
        "provider_refresh_started_at": started,
        "provider_refresh_completed_at": None,
        "provider_refresh_cancelled": False,
        "provider_refresh_cancel_reason": "",
        "rows_refreshed": 0,
        "rows_skipped_due_to_shutdown": 0,
        "runtime_index_updated": False,
        "runtime_index_update_status": "NOT_STARTED",
        "canonical_identity_type": "PROVIDER_URL",
        "pair_address_used_as_canonical_count": 0,
        "provider_pair_url_exact_preserved_count": 0,
        "url_final_segment_lowercased_count": 0,
        "refresh_failures": [],
        "display_repaired_rows": 0,
        "display_still_unavailable_rows": 0,
        "unresolved_display_reason_counts": {},
        "rows_checked": 0,
        "rows_rehydration_needed": 0,
        "dex_rehydration_attempted_count": 0,
        "dex_rehydration_success_count": 0,
        "dex_rehydration_failed_count": 0,
        "failed_rehydration_urls": [],
        "failed_rehydration_reasons": {},
    }

    if is_shutting_down():
        failure = build_refresh_failure(
            error_code=errors.CONTROLLED_SHUTDOWN_SKIP, shutdown_event_set=True
        )
        meta["provider_refresh_cancelled"] = True
        meta["provider_refresh_cancel_reason"] = CONTROLLED_SHUTDOWN_SKIP
        meta["runtime_index_update_status"] = "SKIPPED_SHUTDOWN"
        meta["refresh_failures"] = [failure]
        meta.update({k: v for k, v in failure.items() if k.startswith("refresh_") or k in ("recovery_instruction", "retryable", "user_message")})
        meta["failure_summary"] = summarize_failures([failure])
        return meta

    if clear_cache:
        try:
            from app.ae13b_product.dexscreener_pair_verify import get_pair_verify_limiter

            get_pair_verify_limiter().clear_cache()
        except Exception:
            pass

    seed = seed_csv or DEFAULT_SEED_CSV
    # Prefer existing index as base so we keep prior display fields when refresh skips
    existing = load_runtime_identity_index()
    base_by_url: dict[str, dict[str, Any]] = {}
    if existing.get("ok"):
        for r in existing.get("rows") or []:
            u = cell(r.get("provider_pair_url_exact") or r.get("canonical_market_identity"))
            if u:
                base_by_url[u] = r

    source_rows = _read_seed_rows(seed) if seed.exists() else []
    if max_rows is not None:
        source_rows = source_rows[:max_rows]

    rebuild_at = _utc_now()
    index_rows: list[dict[str, Any]] = []
    seen: set[str] = set()

    for src in source_rows:
        if is_shutting_down():
            meta["provider_refresh_cancelled"] = True
            meta["provider_refresh_cancel_reason"] = CONTROLLED_SHUTDOWN_SKIP
            meta["rows_skipped_due_to_shutdown"] += 1
            log.info("background scanner stopped / refresh cancelled due to shutdown")
            break

        working = dict(src)
        # Seed URL from helpers if missing (does not make pair_address canonical)
        if not cell(working.get("provider_pair_url")):
            chain = cell(working.get("provider_chain_id") or working.get("chain"))
            pair = cell(working.get("provider_pair_address") or working.get("refetch_pair_id"))
            if chain and pair:
                working["provider_pair_url"] = f"https://dexscreener.com/{chain}/{pair}"

        if allow_dexscreener:
            needs_symbols = row_needs_symbol_rehydration(working)
            if needs_symbols:
                meta["rows_rehydration_needed"] += 1
            try:
                working = _verify_one_row(
                    working,
                    use_cache=not force and not clear_cache,
                    overwrite=bool(force or clear_cache),
                )
                meta["rows_refreshed"] += 1
                if needs_symbols:
                    # Conditional symbol rehydration for rows still missing symbols.
                    outcome = rehydrate_row_symbols(
                        working, use_cache=not force and not clear_cache
                    )
                    working = outcome["row"]
                    if outcome["attempted"]:
                        meta["dex_rehydration_attempted_count"] += 1
                    if outcome["success"]:
                        meta["dex_rehydration_success_count"] += 1
                    elif outcome["attempted"]:
                        meta["dex_rehydration_failed_count"] += 1
                        meta["failed_rehydration_urls"].append(outcome["provider_url"])
                        reason = f"{outcome['failure_code']}: {outcome['failure_reason']}"
                        meta["failed_rehydration_reasons"][outcome["failure_code"] or "UNKNOWN"] = (
                            meta["failed_rehydration_reasons"].get(
                                outcome["failure_code"] or "UNKNOWN", 0
                            )
                            + 1
                        )
                        log.info("manual refresh rehydration failed: %s", reason)
            except ProviderRefreshError as exc:
                failure = dict(exc.failure)
                failure.setdefault("canonical_market_identity", cell(working.get("provider_pair_url")))

                rescued = False
                if needs_symbols and not failure.get("controlled_shutdown_skip"):
                    # Do not stop after pair verification fails. Some DexScreener URL
                    # final segments are opaque market IDs or provider-specific IDs,
                    # not strict EVM/Solana pair-address values. The symbol resolver
                    # can still recover BASE/QUOTE through provider URL and token-pair
                    # endpoints while keeping URL as canonical identity.
                    outcome = rehydrate_row_symbols(
                        working, use_cache=not force and not clear_cache
                    )
                    working = outcome["row"]
                    if outcome["attempted"]:
                        meta["dex_rehydration_attempted_count"] += 1
                    if outcome["success"]:
                        meta["dex_rehydration_success_count"] += 1
                        rescued = True
                    elif outcome["attempted"]:
                        meta["dex_rehydration_failed_count"] += 1
                        meta["failed_rehydration_urls"].append(outcome["provider_url"])
                        meta["failed_rehydration_reasons"][outcome["failure_code"] or "UNKNOWN"] = (
                            meta["failed_rehydration_reasons"].get(outcome["failure_code"] or "UNKNOWN", 0) + 1
                        )

                if not rescued:
                    meta["refresh_failures"].append(failure)
                    if failure.get("controlled_shutdown_skip"):
                        meta["provider_refresh_cancelled"] = True
                        meta["provider_refresh_cancel_reason"] = CONTROLLED_SHUTDOWN_SKIP
                        meta["rows_skipped_due_to_shutdown"] += 1
                        break
                    log.warning(
                        "manual refresh row failed: %s (%s)",
                        failure.get("refresh_error_code"),
                        failure.get("refresh_error_reason"),
                    )
            except Exception as exc:  # noqa: BLE001 - classified into structured failure
                failure = build_refresh_failure(
                    error_code=errors.classify_refresh_exception(exc),
                    provider_url=cell(working.get("provider_pair_url")),
                    chain=cell(working.get("provider_chain_id") or working.get("chain")),
                    exception=exc,
                    shutdown_event_set=is_shutting_down(),
                )
                meta["refresh_failures"].append(failure)
                log.warning("manual refresh row failed: %s", failure.get("refresh_error_code"))

        row = build_index_row(
            working,
            last_identity_rebuild_at=rebuild_at,
            last_market_update_at=rebuild_at,
        )
        key = cell(row.get("canonical_market_identity"))
        if not key or key in seen:
            continue
        seen.add(key)

        prior = base_by_url.get(key) or {}
        prior_display = cell(prior.get("symbol_pair_display"))
        new_display = cell(row.get("symbol_pair_display"))
        if not is_symbol_pair_available(new_display) and is_symbol_pair_available(prior_display):
            # Keep the better cached display rather than degrading the row.
            for field in (
                "symbol_pair_display",
                "symbol_pair_display_status",
                "symbol_pair_display_reason",
                "symbol_pair_address_fallback",
                "base_token_symbol",
                "quote_token_symbol",
                "provider_base_token_symbol",
                "provider_quote_token_symbol",
            ):
                if prior.get(field) not in (None, ""):
                    row[field] = prior.get(field)
            new_display = cell(row.get("symbol_pair_display"))
        if is_symbol_pair_available(new_display):
            if not is_symbol_pair_available(prior_display):
                meta["display_repaired_rows"] += 1
            if is_proper_symbol_pair_display(new_display):
                try:
                    upsert_last_good_display({
                        "provider_pair_url_exact": cell(row.get("provider_pair_url_exact") or key),
                        "symbol_pair_display": new_display,
                        "provider_base_token_symbol": cell(row.get("provider_base_token_symbol")),
                        "provider_quote_token_symbol": cell(row.get("provider_quote_token_symbol")),
                        "provider_base_token_address": cell(row.get("provider_base_token_address")),
                        "provider_quote_token_address": cell(row.get("provider_quote_token_address")),
                        "provider_dex_id": cell(row.get("provider_dex_id") or row.get("dex_id")),
                        "chain": cell(row.get("chain")),
                        "source": "manual_refresh_provider_resolution",
                    })
                except Exception as _lg_exc:
                    log.warning("last-good display cache upsert failed: %s", _lg_exc)
        else:
            meta["display_still_unavailable_rows"] += 1
            reason = cell(row.get("symbol_pair_display_reason")) or "unspecified"
            counts = meta["unresolved_display_reason_counts"]
            counts[reason] = counts.get(reason, 0) + 1

        # Case preservation checks
        final = cell(row.get("provider_pair_url_final_segment_exact"))
        if final and key.endswith(final):
            meta["provider_pair_url_exact_preserved_count"] += 1
        if final and final != final.lower() and key.lower().endswith(final.lower()) and not key.endswith(final):
            meta["url_final_segment_lowercased_count"] += 1

        # Never allow pair_address as canonical
        if row.get("canonical_market_identity_type") != "PROVIDER_URL":
            meta["pair_address_used_as_canonical_count"] += 1
            continue

        index_rows.append(row)

    if not index_rows and not meta["provider_refresh_cancelled"]:
        # Fall back to rebuilding from seed without network results already attempted
        meta["runtime_index_update_status"] = "NO_ROWS"
        if not meta["refresh_failures"]:
            meta["refresh_failures"].append(
                build_refresh_failure(
                    error_code=errors.IDENTITY_UNRESOLVED,
                    reason="no seed rows resolved to a canonical provider URL identity",
                )
            )
        meta["failure_summary"] = summarize_failures(meta["refresh_failures"])
        meta["provider_refresh_completed_at"] = _utc_now()
        return meta

    meta["rows_checked"] = len(index_rows)
    if index_rows:
        try:
            report = write_runtime_index_validated(
                index_rows, jsonl_path=INDEX_JSONL_PATH, csv_path=INDEX_CSV_PATH
            )
            meta["runtime_index_updated"] = True
            meta["runtime_index_update_status"] = "ATOMIC_OK"
            meta["atomic_update_report"] = report
        except RuntimeIndexValidationError as exc:
            meta["runtime_index_update_status"] = "VALIDATION_FAILED_INDEX_PRESERVED"
            meta["atomic_update_report"] = exc.report
            meta["refresh_failures"].append(
                build_refresh_failure(
                    error_code=errors.UNKNOWN_PROVIDER_REFRESH_ERROR,
                    exception=exc,
                    reason="candidate index failed validation; existing runtime index preserved",
                )
            )
            log.error("runtime index validation failed: %s", exc.report.get("problems"))
        except Exception as exc:
            meta["runtime_index_update_status"] = f"FAILED:{exc}"
            meta["refresh_failures"].append(
                build_refresh_failure(
                    error_code=errors.classify_refresh_exception(exc),
                    exception=exc,
                    reason=f"atomic runtime index write failed: {exc}",
                )
            )
            log.error("atomic runtime index update failed: %s", exc)

    meta["provider_refresh_completed_at"] = _utc_now()
    meta["index_rows"] = len(index_rows)
    meta["failure_summary"] = summarize_failures(meta["refresh_failures"])
    meta["refresh_status"] = "FAILED" if not meta["runtime_index_updated"] else "OK"
    if meta["refresh_failures"] and not meta["runtime_index_updated"]:
        first = meta["refresh_failures"][0]
        meta["refresh_error_code"] = first.get("refresh_error_code")
        meta["refresh_error_reason"] = first.get("refresh_error_reason")
        meta["recovery_instruction"] = first.get("recovery_instruction")
        meta["retryable"] = first.get("retryable")
        meta["user_message"] = first.get("user_message")
    return meta
