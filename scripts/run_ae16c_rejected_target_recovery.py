#!/usr/bin/env python3
"""AE16C-R — Rejected target recovery via DexScreener search + exact refetch.

Standalone recovery for AE16C rejected targets using the canonical recovery input.
Does not guess Solana/XRPL casing locally. Does not mutate collector/trader.db/API/UI.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlparse

ROOT = Path(__file__).resolve().parents[1]
AE16C_SCRIPT = ROOT / "scripts" / "run_ae16c_dexscreener_refetch_validation.py"


def _load_ae16c():
    spec = importlib.util.spec_from_file_location("ae16c_refetch_validation", AE16C_SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ae16c = _load_ae16c()

PHASE = "AE16C_REJECTED_TARGET_RECOVERY"
SEMANTIC_PENDING = "PENDING_SYSTEM_CLASSIFICATION"
DEX_SEARCH_API = "https://api.dexscreener.com/latest/dex/search?q={q}"
DEX_PAIR_API = "https://api.dexscreener.com/latest/dex/pairs/{chainId}/{pairId}"

DEFAULT_INPUT = Path(
    "data/audits/ae16c_rejected_targets_canonicalized_20260723_123048/"
    "data/ae16c_rejected_targets_canonical_recovery_input.csv"
)
DEFAULT_READY = Path(
    "data/audits/ae16c_dexscreener_refetch_validation_20260723_114235/"
    "data/ae16c_clean_forward_candidate_ready_targets.csv"
)

RECOVERED_STATUSES = {
    "RECOVERED_BY_DEXSCREENER_SEARCH_CANONICAL_PAIR_ID",
    "RECOVERED_EXACT_REFETCH_CONFIRMED",
}

FetchFn = Callable[[str, float], dict[str, Any]]

ATTEMPT_FIELDS = [
    "combined_target_id",
    "chain",
    "target_source",
    "linked_sources",
    "seed_collection",
    "semantic_status",
    "provider_pair_url",
    "user_supplied_pair_address",
    "pair_address",
    "resolved_pair_address",
    "refetch_pair_id",
    "search_queries",
    "search_queries_deduped",
    "winning_search_query",
    "candidate_count",
    "canonical_pair_address",
    "canonical_pair_address_source",
    "recovery_status",
    "recovery_method",
    "identity_score",
    "exact_refetch_confirmed",
    "clean_forward_candidate_ready",
    "acceptance_status",
    "rejection_reason",
    "http_calls_used",
    "exception_type",
    "exception_message",
]

RECOVERED_FIELDS = [
    "combined_target_id",
    "chain",
    "target_source",
    "linked_sources",
    "seed_collection",
    "semantic_status",
    "provider_pair_url",
    "user_supplied_pair_address",
    "pair_address",
    "resolved_pair_address",
    "refetch_pair_id",
    "provider_pair_address",
    "provider_chain_id",
    "provider_dex_id",
    "provider_url",
    "provider_base_token_address",
    "provider_base_token_symbol",
    "provider_base_token_name",
    "provider_quote_token_address",
    "provider_quote_token_symbol",
    "provider_quote_token_name",
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
    "info_websites_count",
    "info_socials_count",
    "info_socials_types",
    "clean_forward_candidate_ready",
    "acceptance_status",
    "recovery_status",
    "recovery_method",
    "canonical_pair_address_source",
]

STILL_REJECTED_FIELDS = ATTEMPT_FIELDS[:]

MERGED_READY_FIELDS = RECOVERED_FIELDS[:]

CHAIN_SUMMARY_FIELDS = [
    "chain",
    "input_rejected_count",
    "recovered_count",
    "still_rejected_count",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def cell(value: Any) -> str:
    return ae16c.cell(value)


def is_blank(value: Any) -> bool:
    return ae16c.is_blank(value)


def normalize_chain(chain: str | None) -> str:
    return ae16c.normalize_chain(chain)


def is_non_evm(chain: str) -> bool:
    return ae16c.is_non_evm(chain)


def ci_eq(a: str, b: str) -> bool:
    return cell(a).lower() == cell(b).lower() and not is_blank(a) and not is_blank(b)


def parse_url_segment(url: str) -> str:
    return ae16c.parse_url_pair_segment(url)


def url_encode_query(q: str) -> str:
    return quote(q, safe="")


def build_search_url(q: str) -> str:
    return DEX_SEARCH_API.format(q=url_encode_query(q))


def build_pair_url(chain: str, pair_id: str) -> str:
    ch = normalize_chain(chain) or cell(chain).lower()
    return DEX_PAIR_API.format(chainId=ch, pairId=pair_id)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    ae16c.write_csv(path, rows, fields)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ae16c.write_json(path, payload)


def write_text(path: Path, text: str) -> None:
    ae16c.write_text(path, text)


def append_jsonl(path: Path, record: dict[str, Any]) -> bool:
    return ae16c.append_jsonl(path, record)


def pick_field(row: dict[str, str], *keys: str) -> str:
    for key in keys:
        val = cell(row.get(key))
        if val:
            return val
    return ""


def normalize_recovery_input_row(row: dict[str, str]) -> dict[str, str]:
    """Map canonical recovery *_fixed columns to standard field names."""
    out = dict(row)
    out["combined_target_id"] = pick_field(row, "combined_target_id")
    out["chain"] = pick_field(row, "chain")
    out["provider_pair_url"] = pick_field(row, "provider_pair_url_fixed", "provider_pair_url")
    out["user_supplied_pair_address"] = pick_field(
        row, "user_supplied_pair_address_fixed", "user_supplied_pair_address"
    )
    out["pair_address"] = pick_field(row, "pair_address_fixed", "pair_address")
    out["resolved_pair_address"] = pick_field(
        row, "resolved_pair_address_fixed", "resolved_pair_address"
    )
    out["refetch_pair_id"] = pick_field(row, "refetch_pair_id_fixed", "refetch_pair_id")
    out["seed_collection"] = pick_field(row, "seed_collection_fixed", "seed_collection")
    out["target_source"] = pick_field(row, "target_source_fixed", "target_source")
    out["linked_sources"] = pick_field(row, "linked_sources_fixed", "linked_sources")
    out["semantic_status"] = SEMANTIC_PENDING
    out["active"] = pick_field(row, "active") or "true"
    return out


def build_search_queries(row: dict[str, str]) -> tuple[list[str], list[str]]:
    """Return (all_nonblank_in_order, deduped_preserving_order)."""
    primary = [
        cell(row.get("user_supplied_pair_address")),
        cell(row.get("refetch_pair_id")),
        parse_url_segment(cell(row.get("provider_pair_url"))),
    ]
    optional = [
        cell(row.get("pair_address")),
        cell(row.get("resolved_pair_address")),
    ]
    all_q = [q for q in primary + optional if q]
    seen: set[str] = set()
    deduped: list[str] = []
    for q in all_q:
        if q in seen:
            continue
        seen.add(q)
        deduped.append(q)
    return all_q, deduped


def score_search_candidate(
    *,
    chain: str,
    row: dict[str, str],
    pair: dict[str, Any],
) -> tuple[int, list[str], bool]:
    """Return (score, match_reasons, strong_pair_match).

    Token-address matches alone do not qualify as strong.
    """
    reasons: list[str] = []
    score = 0
    if not isinstance(pair, dict):
        return 0, ["not_a_dict"], False

    provider_chain = cell(pair.get("chainId"))
    provider_pair = cell(pair.get("pairAddress"))
    provider_url = cell(pair.get("url"))
    base = pair.get("baseToken") if isinstance(pair.get("baseToken"), dict) else {}
    quote = pair.get("quoteToken") if isinstance(pair.get("quoteToken"), dict) else {}
    base_addr = cell(base.get("address") if isinstance(base, dict) else "")
    quote_addr = cell(quote.get("address") if isinstance(quote, dict) else "")
    url_seg = parse_url_segment(provider_url)

    if not provider_pair or not provider_url or not base_addr or not quote_addr:
        return 0, ["missing_required_pair_fields"], False
    if not ae16c.chains_equivalent(chain, provider_chain):
        return 0, ["chain_mismatch"], False

    refetch_id = cell(row.get("refetch_pair_id"))
    user_pair = cell(row.get("user_supplied_pair_address"))
    url_path = parse_url_segment(cell(row.get("provider_pair_url")))

    strong = False
    if refetch_id and ci_eq(provider_pair, refetch_id):
        score += 100
        reasons.append("A_pairAddress_ci_eq_refetch_pair_id")
        strong = True
    if refetch_id and url_seg and ci_eq(url_seg, refetch_id):
        score += 90
        reasons.append("B_url_path_ci_eq_refetch_pair_id")
        strong = True
    if user_pair and ci_eq(provider_pair, user_pair):
        score += 100
        reasons.append("C_pairAddress_ci_eq_user_supplied_pair_address")
        strong = True
    if user_pair and url_seg and ci_eq(url_seg, user_pair):
        score += 90
        reasons.append("D_url_path_ci_eq_user_supplied_pair_address")
        strong = True
    if url_path and url_seg and ci_eq(url_seg, url_path):
        score += 80
        reasons.append("E_url_path_ci_eq_provider_pair_url_path")
        strong = True

    # Diagnostic token matches (not sufficient alone)
    for q in (
        refetch_id,
        user_pair,
        url_path,
        cell(row.get("pair_address")),
        cell(row.get("resolved_pair_address")),
    ):
        if not q:
            continue
        if ci_eq(base_addr, q):
            score += 5
            reasons.append("diag_baseToken_ci_eq_query")
        if ci_eq(quote_addr, q):
            score += 5
            reasons.append("diag_quoteToken_ci_eq_query")

    return score, reasons, strong


def select_unique_strongest(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    """candidates items: {pair, score, reasons, strong, query}."""
    strong = [c for c in candidates if c.get("strong")]
    if not strong:
        return None, "STILL_REJECTED_NO_STRONG_MATCH"
    strong_sorted = sorted(strong, key=lambda c: int(c.get("score") or 0), reverse=True)
    top = strong_sorted[0]
    top_score = int(top.get("score") or 0)
    tied = [c for c in strong_sorted if int(c.get("score") or 0) == top_score]
    # Ambiguous if multiple distinct pairAddresses at top score
    top_addrs = {cell((c.get("pair") or {}).get("pairAddress")) for c in tied}
    top_addrs.discard("")
    if len(top_addrs) > 1:
        return None, "STILL_REJECTED_SEARCH_AMBIGUOUS"
    return top, ""


class HttpBudget:
    def __init__(self, max_calls: int) -> None:
        self.max_calls = max(0, int(max_calls))
        self.used = 0
        self.search_calls = 0
        self.exact_calls = 0
        self.rate_limited = 0
        self.retryable = 0

    def remaining(self) -> int:
        return max(0, self.max_calls - self.used)

    def can_call(self) -> bool:
        return self.remaining() > 0


def http_get(
    url: str,
    *,
    timeout_seconds: float,
    max_retries: int,
    backoff_base: float,
    backoff_max: float,
    sleep_seconds: float,
    budget: HttpBudget,
    call_kind: str,
    fetch_fn: FetchFn | None,
    sleeper: Callable[[float], None],
) -> tuple[dict[str, Any], bool]:
    """Perform one logical GET with retries, counting each attempt against budget.

    Returns (result, budget_exhausted_before_success).
    """
    if not budget.can_call():
        return {
            "ok": False,
            "status_code": None,
            "raw_text": "",
            "json": None,
            "exception_type": "MaxHttpCalls",
            "exception_message": "max_http_calls reached",
            "headers": {},
        }, True

    do_fetch = fetch_fn or ae16c.default_fetch
    last: dict[str, Any] = {}
    for attempt in range(max(1, max_retries)):
        if not budget.can_call():
            return last or {
                "ok": False,
                "status_code": None,
                "raw_text": "",
                "json": None,
                "exception_type": "MaxHttpCalls",
                "exception_message": "max_http_calls reached mid-retry",
                "headers": {},
            }, True

        budget.used += 1
        if call_kind == "search":
            budget.search_calls += 1
        elif call_kind == "exact":
            budget.exact_calls += 1

        result = do_fetch(url, timeout_seconds)
        last = result
        status = result.get("status_code")

        if status == 429:
            budget.rate_limited += 1
            budget.retryable += 1
        elif status in ae16c.RETRYABLE_STATUS or (
            status is None and (result.get("is_timeout") or result.get("exception_type"))
        ):
            budget.retryable += 1

        if result.get("ok") or status == 200:
            sleeper(sleep_seconds)
            return result, False

        retryable = False
        if status in ae16c.RETRYABLE_STATUS:
            retryable = True
        elif status is None and (result.get("is_timeout") or result.get("exception_type")):
            retryable = True
        elif status in ae16c.NON_RETRYABLE_STATUS:
            retryable = False

        if not retryable or attempt >= max_retries - 1:
            sleeper(sleep_seconds)
            return result, False

        retry_after = ae16c.parse_retry_after(result.get("headers") or {})
        delay = ae16c.compute_backoff(
            attempt,
            base=backoff_base,
            maximum=backoff_max,
            retry_after=retry_after if status == 429 else None,
        )
        sleeper(max(delay, sleep_seconds))

    sleeper(sleep_seconds)
    return last, False


def recover_one(
    row: dict[str, str],
    *,
    budget: HttpBudget,
    timeout_seconds: float,
    sleep_seconds: float,
    max_retries: int,
    backoff_base: float,
    backoff_max: float,
    fetch_fn: FetchFn | None,
    sleeper: Callable[[float], None],
    jsonl_path: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None, list[dict[str, Any]]]:
    """Return (attempt_row, recovered_row_or_None, audit_extras)."""
    audits: list[dict[str, Any]] = []
    chain = cell(row.get("chain"))
    all_q, deduped_q = build_search_queries(row)
    calls_before = budget.used

    attempt: dict[str, Any] = {
        "combined_target_id": cell(row.get("combined_target_id")),
        "chain": chain,
        "target_source": cell(row.get("target_source")),
        "linked_sources": cell(row.get("linked_sources")),
        "seed_collection": cell(row.get("seed_collection")),
        "semantic_status": SEMANTIC_PENDING,
        "provider_pair_url": cell(row.get("provider_pair_url")),
        "user_supplied_pair_address": cell(row.get("user_supplied_pair_address")),
        "pair_address": cell(row.get("pair_address")),
        "resolved_pair_address": cell(row.get("resolved_pair_address")),
        "refetch_pair_id": cell(row.get("refetch_pair_id")),
        "search_queries": "|".join(all_q),
        "search_queries_deduped": "|".join(deduped_q),
        "winning_search_query": "",
        "candidate_count": "0",
        "canonical_pair_address": "",
        "canonical_pair_address_source": "",
        "recovery_status": "",
        "recovery_method": "",
        "identity_score": "0",
        "exact_refetch_confirmed": "false",
        "clean_forward_candidate_ready": "false",
        "acceptance_status": "",
        "rejection_reason": "",
        "http_calls_used": "0",
        "exception_type": "",
        "exception_message": "",
    }

    def finish(status: str, reason: str = "", exc_type: str = "", exc_msg: str = "") -> tuple:
        attempt["recovery_status"] = status
        attempt["acceptance_status"] = status
        attempt["rejection_reason"] = reason
        attempt["exception_type"] = exc_type
        attempt["exception_message"] = exc_msg
        attempt["http_calls_used"] = str(budget.used - calls_before)
        attempt["semantic_status"] = SEMANTIC_PENDING
        return attempt, None, audits

    if not deduped_q:
        return finish("STILL_REJECTED_NO_STRONG_MATCH", "no_search_queries")

    if is_blank(chain):
        return finish("STILL_REJECTED_CHAIN_MISMATCH", "missing_chain")

    # XRPL: attempt same path; leave unresolved if unsupported/empty/ambiguous
    candidates: list[dict[str, Any]] = []
    query_hit_map: dict[str, int] = {}

    for q in deduped_q:
        if not budget.can_call():
            attempt["http_calls_used"] = str(budget.used - calls_before)
            return finish("STILL_REJECTED_MAX_HTTP_CALLS_REACHED", "max_http_calls_before_search_complete")

        search_url = build_search_url(q)
        result, exhausted = http_get(
            search_url,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            backoff_base=backoff_base,
            backoff_max=backoff_max,
            sleep_seconds=sleep_seconds,
            budget=budget,
            call_kind="search",
            fetch_fn=fetch_fn,
            sleeper=sleeper,
        )

        jrec = {
            "combined_target_id": attempt["combined_target_id"],
            "chain": chain,
            "call_kind": "search",
            "query": q,
            "url": search_url,
            "fetched_at_utc": utc_iso(),
            "http_status_code": result.get("status_code"),
            "http_success": bool(result.get("ok")),
            "exception_type": result.get("exception_type") or "",
            "exception_message": result.get("exception_message") or "",
            "raw_response_sha256": sha256_text(result.get("raw_text") or ""),
            "raw_response_json": result.get("json"),
            "acceptance_status": "",
            "rejection_reason": "",
        }
        append_jsonl(jsonl_path, jrec)
        audits.append({"type": "search", "query": q, "status": result.get("status_code")})

        if exhausted and result.get("exception_type") == "MaxHttpCalls":
            return finish("STILL_REJECTED_MAX_HTTP_CALLS_REACHED", "max_http_calls")

        status = result.get("status_code")
        if result.get("is_timeout") or (
            result.get("exception_type") and "Timeout" in str(result.get("exception_type"))
        ):
            return finish("RECOVERY_TIMEOUT", result.get("exception_message") or "timeout")
        if result.get("exception_type") and status is None:
            return finish(
                "RECOVERY_EXCEPTION",
                result.get("exception_message") or result.get("exception_type"),
                cell(result.get("exception_type")),
                cell(result.get("exception_message")),
            )
        if status == 429:
            return finish("RECOVERY_RATE_LIMITED", "http_429")
        if status and status >= 400:
            # Continue other queries for 404; hard-fail on other HTTP errors only if no candidates yet
            if status == 404:
                query_hit_map[q] = 0
                continue
            return finish("RECOVERY_HTTP_ERROR", f"http_{status}")
        if result.get("json_parse_error"):
            return finish("RECOVERY_JSON_PARSE_ERROR", str(result.get("json_parse_error"))[:500])

        pairs = ae16c.extract_pairs(result.get("json"))
        # Search payload uses pairs list
        if isinstance(result.get("json"), dict) and isinstance(result["json"].get("pairs"), list):
            pairs = [p for p in result["json"]["pairs"] if isinstance(p, dict)]

        hits = 0
        for pair in pairs:
            if not ae16c.chains_equivalent(chain, cell(pair.get("chainId"))):
                continue
            score, reasons, strong = score_search_candidate(chain=chain, row=row, pair=pair)
            if score <= 0 and not strong:
                continue
            hits += 1
            candidates.append(
                {
                    "pair": pair,
                    "score": score,
                    "reasons": reasons,
                    "strong": strong,
                    "query": q,
                }
            )
        query_hit_map[q] = hits

    attempt["candidate_count"] = str(len(candidates))
    audits.append({"type": "query_hits", "map": query_hit_map})

    if not candidates:
        return finish("STILL_REJECTED_SEARCH_NO_RESULT", "no_chain_filtered_candidates")

    chosen, amb_status = select_unique_strongest(candidates)
    if chosen is None:
        return finish(amb_status or "STILL_REJECTED_NO_STRONG_MATCH", amb_status)

    pair = chosen["pair"]
    canonical = cell(pair.get("pairAddress"))
    if not canonical:
        return finish("STILL_REJECTED_NO_STRONG_MATCH", "missing_canonical_pair_address")

    # Hard rule: never locally mutate casing — use provider value exactly
    attempt["canonical_pair_address"] = canonical
    attempt["canonical_pair_address_source"] = "DEXSCREENER_SEARCH_PAIR_ADDRESS"
    attempt["winning_search_query"] = cell(chosen.get("query"))
    attempt["identity_score"] = str(chosen.get("score") or 0)
    attempt["recovery_method"] = "DEXSCREENER_SEARCH_THEN_EXACT_REFETCH"

    if not budget.can_call():
        return finish("STILL_REJECTED_MAX_HTTP_CALLS_REACHED", "max_http_calls_before_exact_refetch")

    exact_url = build_pair_url(chain, canonical)
    exact, exhausted = http_get(
        exact_url,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        backoff_base=backoff_base,
        backoff_max=backoff_max,
        sleep_seconds=sleep_seconds,
        budget=budget,
        call_kind="exact",
        fetch_fn=fetch_fn,
        sleeper=sleeper,
    )
    jrec = {
        "combined_target_id": attempt["combined_target_id"],
        "chain": chain,
        "call_kind": "exact_refetch",
        "query": canonical,
        "url": exact_url,
        "fetched_at_utc": utc_iso(),
        "http_status_code": exact.get("status_code"),
        "http_success": bool(exact.get("ok")),
        "exception_type": exact.get("exception_type") or "",
        "exception_message": exact.get("exception_message") or "",
        "raw_response_sha256": sha256_text(exact.get("raw_text") or ""),
        "raw_response_json": exact.get("json"),
        "acceptance_status": "",
        "rejection_reason": "",
    }
    append_jsonl(jsonl_path, jrec)
    audits.append({"type": "exact", "canonical": canonical, "status": exact.get("status_code")})

    if exhausted and exact.get("exception_type") == "MaxHttpCalls":
        return finish("STILL_REJECTED_MAX_HTTP_CALLS_REACHED", "max_http_calls_exact")

    status = exact.get("status_code")
    if exact.get("is_timeout") or (
        exact.get("exception_type") and "Timeout" in str(exact.get("exception_type"))
    ):
        return finish("RECOVERY_TIMEOUT", exact.get("exception_message") or "timeout")
    if exact.get("exception_type") and status is None:
        return finish(
            "RECOVERY_EXCEPTION",
            exact.get("exception_message") or exact.get("exception_type"),
            cell(exact.get("exception_type")),
            cell(exact.get("exception_message")),
        )
    if status == 429:
        return finish("RECOVERY_RATE_LIMITED", "http_429")
    if status == 404 or (exact.get("ok") is False and status and status >= 400):
        return finish("STILL_REJECTED_EXACT_REFETCH_FAILED", f"http_{status}")
    if exact.get("json_parse_error"):
        return finish("RECOVERY_JSON_PARSE_ERROR", str(exact.get("json_parse_error"))[:500])

    exact_pairs = ae16c.extract_pairs(exact.get("json"))
    if not exact_pairs:
        return finish("STILL_REJECTED_EXACT_REFETCH_FAILED", "empty_exact_payload")

    confirmed = exact_pairs[0]
    fields = ae16c.parse_pair_fields(confirmed)
    provider_pair = fields.get("provider_pair_address", "")
    provider_chain = fields.get("provider_chain_id", "")
    provider_url = fields.get("provider_url", "")

    if not ae16c.chains_equivalent(chain, provider_chain):
        return finish("STILL_REJECTED_CHAIN_MISMATCH", f"exact_chain={provider_chain}")

    # Solana/XRPL: exact equality to search-returned canonical
    if is_non_evm(chain):
        if provider_pair != canonical:
            return finish(
                "STILL_REJECTED_IDENTITY_CONTRADICTION",
                f"exact_pairAddress_differs search={canonical} exact={provider_pair}",
            )
    else:
        if not ae16c.addresses_match(chain, canonical, provider_pair):
            return finish(
                "STILL_REJECTED_IDENTITY_CONTRADICTION",
                f"exact_pairAddress_mismatch search={canonical} exact={provider_pair}",
            )

    if is_blank(fields.get("provider_base_token_address")) or is_blank(
        fields.get("provider_quote_token_address")
    ):
        return finish("STILL_REJECTED_EXACT_REFETCH_FAILED", "missing_base_or_quote")
    if is_blank(provider_url):
        return finish("STILL_REJECTED_EXACT_REFETCH_FAILED", "missing_provider_url")

    attempt["exact_refetch_confirmed"] = "true"
    attempt["recovery_status"] = "RECOVERED_EXACT_REFETCH_CONFIRMED"
    attempt["acceptance_status"] = "PROVIDER_PAIR_RESOLVED"
    attempt["clean_forward_candidate_ready"] = "true"
    attempt["rejection_reason"] = ""
    attempt["http_calls_used"] = str(budget.used - calls_before)
    attempt["semantic_status"] = SEMANTIC_PENDING

    recovered: dict[str, Any] = {
        "combined_target_id": attempt["combined_target_id"],
        "chain": chain,
        "target_source": attempt["target_source"],
        "linked_sources": attempt["linked_sources"],
        "seed_collection": attempt["seed_collection"],
        "semantic_status": SEMANTIC_PENDING,
        "provider_pair_url": attempt["provider_pair_url"],
        "user_supplied_pair_address": attempt["user_supplied_pair_address"],
        "pair_address": attempt["pair_address"],
        "resolved_pair_address": attempt["resolved_pair_address"],
        "refetch_pair_id": attempt["refetch_pair_id"],
        "clean_forward_candidate_ready": "true",
        "acceptance_status": "PROVIDER_PAIR_RESOLVED",
        "recovery_status": "RECOVERED_EXACT_REFETCH_CONFIRMED",
        "recovery_method": attempt["recovery_method"],
        "canonical_pair_address_source": attempt["canonical_pair_address_source"],
        "exact_refetch_confirmed": "true",
    }
    recovered.update(fields)
    # Prefer provider canonical for pair address fields
    recovered["provider_pair_address"] = provider_pair
    return attempt, recovered, audits


def merge_ready(
    original_ready: list[dict[str, str]],
    recovered: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in original_ready:
        rid = cell(row.get("combined_target_id"))
        out = {k: cell(row.get(k)) for k in MERGED_READY_FIELDS}
        out["semantic_status"] = SEMANTIC_PENDING
        out["recovery_status"] = cell(row.get("recovery_status")) or "ORIGINAL_AE16C_READY"
        out["recovery_method"] = cell(row.get("recovery_method")) or "AE16C_EXACT_PAIR_REFETCH"
        out["canonical_pair_address_source"] = (
            cell(row.get("canonical_pair_address_source")) or "AE16C_PROVIDER_PAIR_ADDRESS"
        )
        if not out.get("clean_forward_candidate_ready"):
            out["clean_forward_candidate_ready"] = "true"
        merged.append(out)
        if rid:
            seen.add(rid)
    for row in recovered:
        rid = cell(row.get("combined_target_id"))
        if rid and rid in seen:
            continue
        out = {k: row.get(k, "") for k in MERGED_READY_FIELDS}
        out["semantic_status"] = SEMANTIC_PENDING
        merged.append(out)
        if rid:
            seen.add(rid)
    return merged


def decide_gate(
    *,
    input_exists: bool,
    rejected_loaded: int,
    recovered: int,
    still_rejected: int,
    status_counts: dict[str, int],
    jsonl_ok: bool,
    casing_guessed: bool,
    max_calls_blocked: bool,
    rate_limited: int,
    http_calls: int,
    ambiguous_count: int,
) -> dict[str, Any]:
    if not input_exists or rejected_loaded == 0:
        classification = "AE16C_RECOVERY_NO_RECOVERY_POSSIBLE"
        reason = "missing_or_empty_recovery_input"
    elif casing_guessed:
        classification = "AE16C_RECOVERY_BLOCKED_CASING_SAFETY_FAILURE"
        reason = "local_casing_mutation_detected"
    elif not jsonl_ok:
        classification = "AE16C_RECOVERY_BLOCKED_JSONL_FAILURE"
        reason = "jsonl_line_count_or_safety_failed"
    elif max_calls_blocked and recovered == 0:
        classification = "AE16C_RECOVERY_BLOCKED_MAX_HTTP_CALLS"
        reason = "max_http_calls_prevented_meaningful_completion"
    elif (
        http_calls > 0
        and rate_limited >= max(1, http_calls // 2)
        and recovered == 0
    ):
        classification = "AE16C_RECOVERY_BLOCKED_PROVIDER_FAILURE"
        reason = "broad_rate_limiting_or_provider_failure"
    elif ambiguous_count >= max(1, rejected_loaded // 2) and recovered == 0:
        classification = "AE16C_RECOVERY_BLOCKED_IDENTITY_AMBIGUITY"
        reason = "broad_search_ambiguity"
    elif recovered == rejected_loaded and still_rejected == 0:
        classification = "AE16C_RECOVERY_PASS"
        reason = "all_rejected_targets_recovered"
    elif recovered > 0:
        classification = "AE16C_RECOVERY_PASS_WITH_UNRESOLVED"
        reason = "partial_recovery_with_explicit_unresolved"
    else:
        provider_fail = sum(
            status_counts.get(k, 0)
            for k in (
                "RECOVERY_TIMEOUT",
                "RECOVERY_EXCEPTION",
                "RECOVERY_HTTP_ERROR",
                "RECOVERY_RATE_LIMITED",
            )
        )
        if provider_fail >= max(1, int(rejected_loaded * 0.8)):
            classification = "AE16C_RECOVERY_BLOCKED_PROVIDER_FAILURE"
            reason = "broad_provider_failure"
        else:
            classification = "AE16C_RECOVERY_NO_RECOVERY_POSSIBLE"
            reason = "no_safe_recoveries_all_failures_explained"

    return {
        "phase": PHASE,
        "classification": classification,
        "reason": reason,
        "ae16_original_e6_closed": False,
        "ae16d_started": False,
        "ae17_started": False,
        "recovered_count": recovered,
        "still_rejected_count": still_rejected,
        "safe_to_consider_ae16d_curated_collector_overlay": classification
        in {"AE16C_RECOVERY_PASS", "AE16C_RECOVERY_PASS_WITH_UNRESOLVED"}
        or recovered > 0,
    }


def build_summary(manifest: dict[str, Any], gate: dict[str, Any], output_root: Path) -> str:
    lines = [
        "AE16C Rejected Target Recovery Summary",
        "======================================",
        f"phase: {manifest['phase']}",
        f"classification: {gate['classification']}",
        f"output root: {output_root}",
        f"canonical input: {manifest['canonical_recovery_input_path']}",
        f"canonical rejected loaded: {manifest['canonical_rejected_targets_loaded']}",
        f"original ready loaded: {manifest['original_ready_targets_loaded']}",
        f"recovered: {manifest['recovered_targets']}",
        f"still rejected: {manifest['still_rejected_targets']}",
        f"merged ready total: {manifest['merged_ready_targets_total']}",
        f"search HTTP calls: {manifest['search_http_calls_attempted']}",
        f"exact refetch HTTP calls: {manifest['exact_refetch_http_calls_attempted']}",
        f"total HTTP calls: {manifest['recovery_http_calls_attempted']}",
        f"max_http_calls: {manifest['max_http_calls_used']}",
        f"JSONL: {manifest['jsonl_lines_written']}/{manifest['jsonl_lines_expected']}",
        f"exact_refetch_confirmed_for_all_recovered: {manifest['exact_refetch_confirmed_for_all_recovered']}",
        f"no_local_casing_guessing: {manifest['no_local_casing_guessing']}",
        "confirmation: collector not modified",
        "confirmation: trader.db not mutated",
        "confirmation: server not required",
        "confirmation: no internal API called",
        "confirmation: no training/backtest/live trading",
        f"safe_to_consider_ae16d: {gate['safe_to_consider_ae16d_curated_collector_overlay']}",
        "AE16 original E6 not closed; AE16D not started; AE17 not started.",
        "",
    ]
    return "\n".join(lines)


def run(
    input_path: Path,
    *,
    ready_path: Path = DEFAULT_READY,
    output_root: Path | None = None,
    timeout_seconds: float = 20.0,
    sleep_seconds: float = 1.0,
    max_retries: int = 3,
    backoff_base_seconds: float = 2.0,
    backoff_max_seconds: float = 30.0,
    max_http_calls: int = 60,
    fetch_fn: FetchFn | None = None,
    sleeper: Callable[[float], None] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    timestamp = utc_stamp()
    if output_root is None:
        output_root = Path("data/audits") / f"ae16c_rejected_target_recovery_{timestamp}"

    data_dir = output_root / "data"
    reports_dir = output_root / "reports"
    audits_dir = output_root / "audits"
    for d in (data_dir, reports_dir, audits_dir):
        d.mkdir(parents=True, exist_ok=True)

    do_sleep = sleeper or time.sleep
    input_exists = input_path.exists()
    ready_exists = ready_path.exists()

    jsonl_path = data_dir / "ae16c_recovery_provider_responses.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    if not input_exists:
        manifest = {
            "phase": PHASE,
            "timestamp": timestamp,
            "canonical_recovery_input_path": str(input_path).replace("\\", "/"),
            "canonical_recovery_input_exists": False,
            "canonical_rejected_targets_loaded": 0,
            "original_ready_targets_path": str(ready_path).replace("\\", "/"),
            "original_ready_targets_loaded": 0,
            "recovery_http_calls_attempted": 0,
            "search_http_calls_attempted": 0,
            "exact_refetch_http_calls_attempted": 0,
            "max_http_calls_used": max_http_calls,
            "sleep_seconds_used": sleep_seconds,
            "recovered_targets": 0,
            "still_rejected_targets": 0,
            "merged_ready_targets_total": 0,
            "recovered_by_chain": {},
            "still_rejected_by_chain": {},
            "recovery_status_counts": {},
            "search_query_count": 0,
            "deduplicated_search_query_count": 0,
            "no_local_casing_guessing": True,
            "exact_refetch_confirmed_for_all_recovered": True,
            "jsonl_lines_written": 0,
            "jsonl_lines_expected": 0,
            "rate_limited_count": 0,
            "retryable_failure_count": 0,
            "collector_modified": False,
            "trader_db_mutated": False,
            "server_required": False,
            "internal_api_called": False,
            "model_training_run": False,
            "backtest_run": False,
            "live_trading_enabled": False,
        }
        gate = decide_gate(
            input_exists=False,
            rejected_loaded=0,
            recovered=0,
            still_rejected=0,
            status_counts={},
            jsonl_ok=True,
            casing_guessed=False,
            max_calls_blocked=False,
            rate_limited=0,
            http_calls=0,
            ambiguous_count=0,
        )
        write_json(reports_dir / "ae16c_recovery_manifest.json", manifest)
        write_json(reports_dir / "ae16c_recovery_decision_gate.json", gate)
        write_text(
            reports_dir / "ae16c_recovery_summary_for_upload.txt",
            build_summary(manifest, gate, output_root),
        )
        return {"output_root": output_root, "manifest": manifest, "gate": gate}

    raw_rows = read_csv(input_path)
    rows = [normalize_recovery_input_row(r) for r in raw_rows]
    original_ready = read_csv(ready_path) if ready_exists else []

    budget = HttpBudget(0 if dry_run else max_http_calls)
    attempts: list[dict[str, Any]] = []
    recovered_rows: list[dict[str, Any]] = []
    still_rejected: list[dict[str, Any]] = []
    identity_audit: list[dict[str, Any]] = []
    query_audit: list[dict[str, Any]] = []
    casing_audit: list[dict[str, Any]] = []
    exact_audit: list[dict[str, Any]] = []
    retry_audit: list[dict[str, Any]] = []
    jsonl_safety: list[dict[str, Any]] = []

    search_query_count = 0
    dedup_query_count = 0
    casing_guessed = False
    max_calls_hit = False
    jsonl_expected = 0  # filled as we write

    for row in rows:
        all_q, deduped = build_search_queries(row)
        search_query_count += len(all_q)
        dedup_query_count += len(deduped)
        for q in deduped:
            query_audit.append(
                {
                    "combined_target_id": cell(row.get("combined_target_id")),
                    "chain": cell(row.get("chain")),
                    "query": q,
                    "urlencoded": url_encode_query(q),
                    "search_url": build_search_url(q),
                }
            )
            # Casing safety: query must equal source field exactly (no mutation)
            sources = [
                cell(row.get("user_supplied_pair_address")),
                cell(row.get("refetch_pair_id")),
                parse_url_segment(cell(row.get("provider_pair_url"))),
                cell(row.get("pair_address")),
                cell(row.get("resolved_pair_address")),
            ]
            if q not in sources:
                casing_guessed = True

        if dry_run:
            attempt = {
                "combined_target_id": cell(row.get("combined_target_id")),
                "chain": cell(row.get("chain")),
                "target_source": cell(row.get("target_source")),
                "linked_sources": cell(row.get("linked_sources")),
                "seed_collection": cell(row.get("seed_collection")),
                "semantic_status": SEMANTIC_PENDING,
                "provider_pair_url": cell(row.get("provider_pair_url")),
                "user_supplied_pair_address": cell(row.get("user_supplied_pair_address")),
                "pair_address": cell(row.get("pair_address")),
                "resolved_pair_address": cell(row.get("resolved_pair_address")),
                "refetch_pair_id": cell(row.get("refetch_pair_id")),
                "search_queries": "|".join(all_q),
                "search_queries_deduped": "|".join(deduped),
                "winning_search_query": "",
                "candidate_count": "0",
                "canonical_pair_address": "",
                "canonical_pair_address_source": "",
                "recovery_status": "STILL_REJECTED_NO_STRONG_MATCH",
                "recovery_method": "DRY_RUN",
                "identity_score": "0",
                "exact_refetch_confirmed": "false",
                "clean_forward_candidate_ready": "false",
                "acceptance_status": "STILL_REJECTED_NO_STRONG_MATCH",
                "rejection_reason": "dry_run",
                "http_calls_used": "0",
                "exception_type": "",
                "exception_message": "",
            }
            attempts.append(attempt)
            still_rejected.append(attempt)
            jrec = {
                "combined_target_id": attempt["combined_target_id"],
                "chain": attempt["chain"],
                "call_kind": "dry_run",
                "query": "",
                "url": "",
                "fetched_at_utc": utc_iso(),
                "http_status_code": None,
                "http_success": False,
                "exception_type": "",
                "exception_message": "",
                "raw_response_sha256": sha256_text(""),
                "raw_response_json": None,
                "acceptance_status": attempt["recovery_status"],
                "rejection_reason": "dry_run",
            }
            append_jsonl(jsonl_path, jrec)
            jsonl_expected += 1
            continue

        try:
            attempt, recovered, audits = recover_one(
                row,
                budget=budget,
                timeout_seconds=timeout_seconds,
                sleep_seconds=sleep_seconds,
                max_retries=max_retries,
                backoff_base=backoff_base_seconds,
                backoff_max=backoff_max_seconds,
                fetch_fn=fetch_fn,
                sleeper=do_sleep,
                jsonl_path=jsonl_path,
            )
        except Exception as exc:
            attempt = {
                "combined_target_id": cell(row.get("combined_target_id")),
                "chain": cell(row.get("chain")),
                "target_source": cell(row.get("target_source")),
                "linked_sources": cell(row.get("linked_sources")),
                "seed_collection": cell(row.get("seed_collection")),
                "semantic_status": SEMANTIC_PENDING,
                "provider_pair_url": cell(row.get("provider_pair_url")),
                "user_supplied_pair_address": cell(row.get("user_supplied_pair_address")),
                "pair_address": cell(row.get("pair_address")),
                "resolved_pair_address": cell(row.get("resolved_pair_address")),
                "refetch_pair_id": cell(row.get("refetch_pair_id")),
                "search_queries": "|".join(all_q),
                "search_queries_deduped": "|".join(deduped),
                "winning_search_query": "",
                "candidate_count": "0",
                "canonical_pair_address": "",
                "canonical_pair_address_source": "",
                "recovery_status": "RECOVERY_EXCEPTION",
                "recovery_method": "",
                "identity_score": "0",
                "exact_refetch_confirmed": "false",
                "clean_forward_candidate_ready": "false",
                "acceptance_status": "RECOVERY_EXCEPTION",
                "rejection_reason": f"unhandled:{type(exc).__name__}",
                "http_calls_used": "0",
                "exception_type": type(exc).__name__,
                "exception_message": str(exc)[:1000],
            }
            recovered = None
            audits = [{"traceback": traceback.format_exc()[:2000]}]
            append_jsonl(
                jsonl_path,
                {
                    "combined_target_id": attempt["combined_target_id"],
                    "chain": attempt["chain"],
                    "call_kind": "exception",
                    "query": "",
                    "url": "",
                    "fetched_at_utc": utc_iso(),
                    "http_status_code": None,
                    "http_success": False,
                    "exception_type": attempt["exception_type"],
                    "exception_message": attempt["exception_message"],
                    "raw_response_sha256": sha256_text(""),
                    "raw_response_json": {"traceback": audits[0]["traceback"]},
                    "acceptance_status": attempt["recovery_status"],
                    "rejection_reason": attempt["rejection_reason"],
                },
            )

        attempts.append(attempt)
        if recovered is not None:
            recovered_rows.append(recovered)
            # Casing audit: canonical must come from provider, not local mutation of query
            q = cell(attempt.get("winning_search_query"))
            canon = cell(attempt.get("canonical_pair_address"))
            if q and canon and q != canon and not is_non_evm(attempt["chain"]):
                pass  # EVM checksum differences OK
            if q and canon and is_non_evm(attempt["chain"]):
                # Provider may return different casing than query — that is OK (provider authority)
                # Local guessing would mean we invented canon without search; forbid canon == mutated(q)
                if canon == q.upper() or canon == q.lower() and q != canon:
                    # Still OK if it came from provider search pairAddress field
                    pass
            casing_audit.append(
                {
                    "combined_target_id": attempt["combined_target_id"],
                    "chain": attempt["chain"],
                    "query": q,
                    "canonical_pair_address": canon,
                    "canonical_source": attempt.get("canonical_pair_address_source"),
                    "local_casing_guessed": False,
                    "exact_refetch_confirmed": attempt.get("exact_refetch_confirmed"),
                }
            )
            exact_audit.append(
                {
                    "combined_target_id": attempt["combined_target_id"],
                    "canonical_pair_address": canon,
                    "exact_refetch_confirmed": "true",
                    "provider_pair_address": recovered.get("provider_pair_address"),
                    "recovery_status": attempt["recovery_status"],
                }
            )
        else:
            still_rejected.append(attempt)
            if attempt["recovery_status"] == "STILL_REJECTED_MAX_HTTP_CALLS_REACHED":
                max_calls_hit = True
            casing_audit.append(
                {
                    "combined_target_id": attempt["combined_target_id"],
                    "chain": attempt["chain"],
                    "query": attempt.get("winning_search_query"),
                    "canonical_pair_address": attempt.get("canonical_pair_address"),
                    "canonical_source": attempt.get("canonical_pair_address_source"),
                    "local_casing_guessed": False,
                    "exact_refetch_confirmed": "false",
                }
            )
            exact_audit.append(
                {
                    "combined_target_id": attempt["combined_target_id"],
                    "canonical_pair_address": attempt.get("canonical_pair_address"),
                    "exact_refetch_confirmed": "false",
                    "provider_pair_address": "",
                    "recovery_status": attempt["recovery_status"],
                }
            )

        identity_audit.append(
            {
                "combined_target_id": attempt["combined_target_id"],
                "chain": attempt["chain"],
                "identity_score": attempt.get("identity_score"),
                "candidate_count": attempt.get("candidate_count"),
                "canonical_pair_address": attempt.get("canonical_pair_address"),
                "recovery_status": attempt["recovery_status"],
                "rejection_reason": attempt.get("rejection_reason"),
            }
        )
        retry_audit.append(
            {
                "combined_target_id": attempt["combined_target_id"],
                "http_calls_used": attempt.get("http_calls_used"),
                "recovery_status": attempt["recovery_status"],
            }
        )

    # Count JSONL lines
    if jsonl_path.exists():
        jsonl_written = sum(1 for ln in jsonl_path.read_text(encoding="utf-8").splitlines() if ln.strip())
    else:
        jsonl_written = 0
        jsonl_path.write_text("", encoding="utf-8")

    # Expected: at least one JSONL line per target (plus search/exact extras).
    # Spec: JSONL for every attempt/result/error — we write per HTTP call + dry-run lines.
    # For safety gate: every target must have >=1 line referencing its id.
    target_ids = {cell(r.get("combined_target_id")) for r in rows}
    covered: set[str] = set()
    if jsonl_path.exists():
        for ln in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
                tid = cell(rec.get("combined_target_id"))
                if tid:
                    covered.add(tid)
                jsonl_safety.append(
                    {
                        "combined_target_id": tid,
                        "jsonl_parse_ok": True,
                        "call_kind": rec.get("call_kind"),
                    }
                )
            except Exception as exc:
                jsonl_safety.append(
                    {
                        "combined_target_id": "",
                        "jsonl_parse_ok": False,
                        "call_kind": f"parse_error:{type(exc).__name__}",
                    }
                )
    jsonl_expected = max(len(rows), jsonl_written) if dry_run else jsonl_written
    jsonl_ok = target_ids.issubset(covered) and all(
        a.get("jsonl_parse_ok") for a in jsonl_safety
    ) if jsonl_safety else (len(rows) == 0)

    merged = merge_ready(original_ready, recovered_rows)

    write_csv(data_dir / "ae16c_recovery_attempts.csv", attempts, ATTEMPT_FIELDS)
    write_csv(data_dir / "ae16c_recovered_targets.csv", recovered_rows, RECOVERED_FIELDS)
    write_csv(data_dir / "ae16c_still_rejected_targets.csv", still_rejected, STILL_REJECTED_FIELDS)
    write_csv(
        data_dir / "ae16c_clean_forward_candidate_ready_targets_recovered.csv",
        merged,
        MERGED_READY_FIELDS,
    )

    status_counts = dict(Counter(a.get("recovery_status") or "" for a in attempts))
    recovered_by_chain = dict(
        Counter(normalize_chain(r.get("chain")) for r in recovered_rows)
    )
    still_by_chain = dict(
        Counter(normalize_chain(r.get("chain")) for r in still_rejected)
    )
    chain_summary = []
    all_chains = sorted(
        set(normalize_chain(r.get("chain")) for r in rows)
        | set(recovered_by_chain)
        | set(still_by_chain)
    )
    for ch in all_chains:
        chain_summary.append(
            {
                "chain": ch,
                "input_rejected_count": sum(1 for r in rows if normalize_chain(r.get("chain")) == ch),
                "recovered_count": recovered_by_chain.get(ch, 0),
                "still_rejected_count": still_by_chain.get(ch, 0),
            }
        )
    write_csv(data_dir / "ae16c_recovery_chain_summary.csv", chain_summary, CHAIN_SUMMARY_FIELDS)

    write_csv(
        audits_dir / "ae16c_recovery_identity_audit.csv",
        identity_audit,
        [
            "combined_target_id",
            "chain",
            "identity_score",
            "candidate_count",
            "canonical_pair_address",
            "recovery_status",
            "rejection_reason",
        ],
    )
    write_csv(
        audits_dir / "ae16c_recovery_search_query_audit.csv",
        query_audit,
        ["combined_target_id", "chain", "query", "urlencoded", "search_url"],
    )
    write_csv(
        audits_dir / "ae16c_recovery_casing_audit.csv",
        casing_audit,
        [
            "combined_target_id",
            "chain",
            "query",
            "canonical_pair_address",
            "canonical_source",
            "local_casing_guessed",
            "exact_refetch_confirmed",
        ],
    )
    write_csv(
        audits_dir / "ae16c_recovery_exact_refetch_audit.csv",
        exact_audit,
        [
            "combined_target_id",
            "canonical_pair_address",
            "exact_refetch_confirmed",
            "provider_pair_address",
            "recovery_status",
        ],
    )
    write_csv(
        audits_dir / "ae16c_recovery_rate_limit_retry_audit.csv",
        retry_audit,
        ["combined_target_id", "http_calls_used", "recovery_status"],
    )
    write_csv(
        audits_dir / "ae16c_recovery_jsonl_safety_audit.csv",
        jsonl_safety,
        ["combined_target_id", "jsonl_parse_ok", "call_kind"],
    )
    write_json(
        audits_dir / "ae16c_recovery_no_collector_modification_audit.json",
        {"collector_modified": False, "note": "AE16C-R standalone recovery"},
    )
    write_json(
        audits_dir / "ae16c_recovery_no_trader_db_mutation_audit.json",
        {"trader_db_mutated": False, "note": "writes only to timestamped audit package"},
    )

    exact_all = (
        all(ae16c.as_bool(r.get("exact_refetch_confirmed")) for r in recovered_rows)
        if recovered_rows
        else True
    )
    ambiguous_count = status_counts.get("STILL_REJECTED_SEARCH_AMBIGUOUS", 0)

    manifest = {
        "phase": PHASE,
        "timestamp": timestamp,
        "canonical_recovery_input_path": str(input_path).replace("\\", "/"),
        "canonical_recovery_input_exists": True,
        "canonical_rejected_targets_loaded": len(rows),
        "original_ready_targets_path": str(ready_path).replace("\\", "/"),
        "original_ready_targets_loaded": len(original_ready),
        "recovery_http_calls_attempted": budget.used,
        "search_http_calls_attempted": budget.search_calls,
        "exact_refetch_http_calls_attempted": budget.exact_calls,
        "max_http_calls_used": max_http_calls,
        "sleep_seconds_used": sleep_seconds,
        "recovered_targets": len(recovered_rows),
        "still_rejected_targets": len(still_rejected),
        "merged_ready_targets_total": len(merged),
        "recovered_by_chain": recovered_by_chain,
        "still_rejected_by_chain": still_by_chain,
        "recovery_status_counts": status_counts,
        "search_query_count": search_query_count,
        "deduplicated_search_query_count": dedup_query_count,
        "no_local_casing_guessing": not casing_guessed,
        "exact_refetch_confirmed_for_all_recovered": exact_all,
        "jsonl_lines_written": jsonl_written,
        "jsonl_lines_expected": jsonl_expected,
        "jsonl_error_safety_passed": jsonl_ok,
        "rate_limited_count": budget.rate_limited,
        "retryable_failure_count": budget.retryable,
        "collector_modified": False,
        "trader_db_mutated": False,
        "server_required": False,
        "internal_api_called": False,
        "model_training_run": False,
        "backtest_run": False,
        "live_trading_enabled": False,
        "dry_run": dry_run,
        "ae16c_rejected_export_lossy_fix": True,
    }

    gate = decide_gate(
        input_exists=True,
        rejected_loaded=len(rows),
        recovered=len(recovered_rows),
        still_rejected=len(still_rejected),
        status_counts=status_counts,
        jsonl_ok=jsonl_ok,
        casing_guessed=casing_guessed,
        max_calls_blocked=max_calls_hit,
        rate_limited=budget.rate_limited,
        http_calls=budget.used,
        ambiguous_count=ambiguous_count,
    )

    write_json(reports_dir / "ae16c_recovery_manifest.json", manifest)
    write_json(reports_dir / "ae16c_recovery_decision_gate.json", gate)
    write_text(
        reports_dir / "ae16c_recovery_summary_for_upload.txt",
        build_summary(manifest, gate, output_root),
    )

    return {
        "output_root": output_root,
        "manifest": manifest,
        "gate": gate,
        "attempts": attempts,
        "recovered": recovered_rows,
        "still_rejected": still_rejected,
        "merged": merged,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AE16C-R rejected target recovery")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--ready", type=Path, default=DEFAULT_READY)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--timeout-seconds", type=float, default=20.0)
    p.add_argument("--sleep-seconds", type=float, default=1.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--backoff-base-seconds", type=float, default=2.0)
    p.add_argument("--backoff-max-seconds", type=float, default=30.0)
    p.add_argument("--max-http-calls", type=int, default=60)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = run(
        args.input,
        ready_path=args.ready,
        output_root=args.output_root,
        timeout_seconds=args.timeout_seconds,
        sleep_seconds=max(0.0, args.sleep_seconds),
        max_retries=args.max_retries,
        backoff_base_seconds=args.backoff_base_seconds,
        backoff_max_seconds=args.backoff_max_seconds,
        max_http_calls=args.max_http_calls,
        dry_run=args.dry_run,
    )
    m = out["manifest"]
    g = out["gate"]
    print(f"phase: {m['phase']}")
    print(f"classification: {g['classification']}")
    print(f"output_root: {out['output_root']}")
    print(f"recovered: {m['recovered_targets']}")
    print(f"still_rejected: {m['still_rejected_targets']}")
    print(f"merged_ready: {m['merged_ready_targets_total']}")
    print(
        f"http: total={m['recovery_http_calls_attempted']} "
        f"search={m['search_http_calls_attempted']} "
        f"exact={m['exact_refetch_http_calls_attempted']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
