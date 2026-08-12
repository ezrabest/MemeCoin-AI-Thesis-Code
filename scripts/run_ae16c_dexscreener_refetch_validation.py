#!/usr/bin/env python3
"""AE16C — Combined Universe DexScreener Refetch Validation (standalone).

Reads AE16B combined clean universe CSV, refetches each target via DexScreener
exact pair endpoint, and writes an isolated audit package.

Does not modify collector, trader.db, API, UI, or existing AE15/AE16/AE16B artifacts.
Does not start a server or call internal API endpoints.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


PHASE = "AE16C_DEXSCREENER_REFETCH_VALIDATION"
SEMANTIC_PENDING = "PENDING_SYSTEM_CLASSIFICATION"
DEX_PAIR_API = "https://api.dexscreener.com/latest/dex/pairs/{chainId}/{pairId}"

DEFAULT_INPUT = Path(
    "data/audits/ae16b_combined_clean_universe_20260723_100640/data/ae16b_combined_clean_universe.csv"
)

NON_EVM_CHAINS = {"solana", "xrpl"}
CHAIN_ALIASES: dict[str, str] = {
    "solana": "solana",
    "sol": "solana",
    "ethereum": "ethereum",
    "eth": "ethereum",
    "bsc": "bsc",
    "binance-smart-chain": "bsc",
    "bnb": "bsc",
    "base": "base",
    "arbitrum": "arbitrum",
    "polygon": "polygon",
    "matic": "polygon",
    "optimism": "optimism",
    "avalanche": "avalanche",
    "robinhood": "robinhood",
    "xrpl": "xrpl",
    "xrp": "xrpl",
}

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
NON_RETRYABLE_STATUS = {400, 401, 403, 404}

HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

VALIDATION_FIELDS = [
    "combined_target_id",
    "active",
    "chain",
    "target_source",
    "linked_sources",
    "seed_collection",
    "semantic_status",
    "provider_pair_url",
    "pair_address",
    "user_supplied_pair_address",
    "resolved_pair_address",
    "refetch_pair_id",
    "refetch_pair_id_source",
    "refetch_url",
    "http_attempted",
    "http_status_code",
    "http_success",
    "response_pair_count",
    "provider_chain_id",
    "provider_dex_id",
    "provider_pair_address",
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
    "raw_response_sha256",
    "provider_identity_match",
    "provider_identity_match_reason",
    "clean_forward_candidate_ready",
    "acceptance_status",
    "rejection_reason",
    "retry_count",
    "final_backoff_seconds",
    "exception_type",
    "exception_message",
]

RESOLVED_FIELDS = [
    "combined_target_id",
    "chain",
    "refetch_pair_id",
    "provider_pair_address",
    "provider_chain_id",
    "provider_base_token_address",
    "provider_quote_token_address",
    "price_usd",
    "liquidity_usd",
    "acceptance_status",
    "clean_forward_candidate_ready",
    "target_source",
    "seed_collection",
    "semantic_status",
]

REJECTED_FIELDS = [
    "combined_target_id",
    "active",
    "chain",
    "target_source",
    "linked_sources",
    "seed_collection",
    "semantic_status",
    "provider_pair_url",
    "pair_address",
    "user_supplied_pair_address",
    "resolved_pair_address",
    "refetch_pair_id",
    "refetch_pair_id_source",
    "refetch_url",
    "acceptance_status",
    "rejection_reason",
    "http_status_code",
    "http_success",
    "provider_identity_match",
    "provider_identity_match_reason",
    "clean_forward_candidate_ready",
    "exception_type",
    "exception_message",
    "retry_count",
    "raw_response_sha256",
]

READY_FIELDS = RESOLVED_FIELDS[:]

CHAIN_SUMMARY_FIELDS = [
    "chain",
    "input_count",
    "resolved_count",
    "ready_count",
    "rejected_count",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def is_blank(value: Any) -> bool:
    return cell(value) == ""


def as_bool(value: Any) -> bool:
    return cell(value).lower() in {"1", "true", "yes", "y"}


def normalize_chain(chain: str | None) -> str:
    raw = cell(chain).lower()
    if not raw:
        return ""
    return CHAIN_ALIASES.get(raw, raw)


def is_non_evm(chain: str) -> bool:
    return normalize_chain(chain) in NON_EVM_CHAINS


def chains_equivalent(expected: str, provider: str) -> bool:
    e = normalize_chain(expected)
    p = normalize_chain(provider)
    if not e or not p:
        return False
    return e == p


def addresses_match(chain: str, expected: str, actual: str) -> bool:
    exp = cell(expected)
    act = cell(actual)
    if not exp or not act:
        return False
    if is_non_evm(chain):
        return exp == act
    return exp.lower() == act.lower()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def safe_jsonl_line(record: dict[str, Any]) -> str:
    """Serialize one JSONL record; never raise to caller for common issues."""
    try:
        return json.dumps(record, ensure_ascii=False, default=str)
    except Exception as exc:  # pragma: no cover
        fallback = {
            "combined_target_id": record.get("combined_target_id"),
            "acceptance_status": record.get("acceptance_status"),
            "jsonl_serialize_error": type(exc).__name__,
            "jsonl_serialize_message": str(exc)[:500],
        }
        return json.dumps(fallback, ensure_ascii=False)


def append_jsonl(path: Path, record: dict[str, Any]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("a", encoding="utf-8", newline="\n") as f:
            f.write(safe_jsonl_line(record) + "\n")
        return True
    except Exception:  # pragma: no cover
        return False


def parse_url_pair_segment(url: str) -> str:
    u = cell(url)
    if not u:
        return ""
    path = urlparse(u).path.rstrip("/")
    if not path:
        return ""
    return path.split("/")[-1]


def has_clean_forward_source(row: dict[str, str]) -> bool:
    ts = cell(row.get("target_source"))
    ls = cell(row.get("linked_sources"))
    return "CLEAN_FORWARD_EXISTING" in ts or "CLEAN_FORWARD_EXISTING" in ls


def select_refetch_pair_id(row: dict[str, str]) -> tuple[str, str]:
    """Return (refetch_pair_id, refetch_pair_id_source)."""
    pair_address = cell(row.get("pair_address"))
    user_pair = cell(row.get("user_supplied_pair_address"))
    resolved = cell(row.get("resolved_pair_address"))
    url = cell(row.get("provider_pair_url"))

    if has_clean_forward_source(row) and pair_address:
        return pair_address, "PAIR_ADDRESS_CLEAN_FORWARD"
    if user_pair:
        return user_pair, "USER_SUPPLIED_PAIR_ADDRESS"
    if resolved:
        return resolved, "RESOLVED_PAIR_ADDRESS"
    segment = parse_url_pair_segment(url)
    if segment:
        return segment, "PROVIDER_PAIR_URL_PATH_SEGMENT"
    return "", "MISSING"


def build_refetch_url(chain: str, pair_id: str) -> str:
    ch = normalize_chain(chain) or cell(chain).lower()
    return DEX_PAIR_API.format(chainId=ch, pairId=pair_id)


def empty_validation_row(row: dict[str, str]) -> dict[str, Any]:
    out = {k: "" for k in VALIDATION_FIELDS}
    for key in (
        "combined_target_id",
        "active",
        "chain",
        "target_source",
        "linked_sources",
        "seed_collection",
        "provider_pair_url",
        "pair_address",
        "user_supplied_pair_address",
        "resolved_pair_address",
    ):
        out[key] = cell(row.get(key))
    out["semantic_status"] = SEMANTIC_PENDING
    out["http_attempted"] = "false"
    out["http_success"] = "false"
    out["clean_forward_candidate_ready"] = "false"
    out["provider_identity_match"] = "false"
    out["retry_count"] = "0"
    out["final_backoff_seconds"] = "0"
    out["response_pair_count"] = "0"
    return out


def extract_pairs(payload: Any) -> list[dict[str, Any]]:
    if payload is None:
        return []
    if isinstance(payload, dict):
        pair = payload.get("pair")
        if isinstance(pair, dict):
            return [pair]
        pairs = payload.get("pairs")
        if isinstance(pairs, list):
            return [p for p in pairs if isinstance(p, dict)]
        if payload.get("pairAddress"):
            return [payload]
        return []
    if isinstance(payload, list):
        return [p for p in payload if isinstance(p, dict)]
    return []


def safe_num(value: Any) -> str:
    if value is None or value == "":
        return ""
    try:
        return str(value)
    except Exception:
        return ""


def parse_pair_fields(pair: dict[str, Any] | None) -> dict[str, str]:
    if not isinstance(pair, dict):
        pair = {}
    base = pair.get("baseToken") if isinstance(pair.get("baseToken"), dict) else {}
    quote = pair.get("quoteToken") if isinstance(pair.get("quoteToken"), dict) else {}
    volume = pair.get("volume") if isinstance(pair.get("volume"), dict) else {}
    txns = pair.get("txns") if isinstance(pair.get("txns"), dict) else {}
    price_change = pair.get("priceChange") if isinstance(pair.get("priceChange"), dict) else {}
    info = pair.get("info") if isinstance(pair.get("info"), dict) else {}
    websites = info.get("websites") if isinstance(info.get("websites"), list) else []
    socials = info.get("socials") if isinstance(info.get("socials"), list) else []
    social_types: list[str] = []
    for s in socials:
        if isinstance(s, dict) and s.get("type"):
            social_types.append(str(s.get("type")))

    def txn_side(window: str, side: str) -> str:
        bucket = txns.get(window) if isinstance(txns.get(window), dict) else {}
        return safe_num(bucket.get(side) if isinstance(bucket, dict) else None)

    return {
        "provider_chain_id": cell(pair.get("chainId")),
        "provider_dex_id": cell(pair.get("dexId")),
        "provider_pair_address": cell(pair.get("pairAddress")),
        "provider_url": cell(pair.get("url")),
        "provider_base_token_address": cell(base.get("address") if isinstance(base, dict) else ""),
        "provider_base_token_symbol": cell(base.get("symbol") if isinstance(base, dict) else ""),
        "provider_base_token_name": cell(base.get("name") if isinstance(base, dict) else ""),
        "provider_quote_token_address": cell(quote.get("address") if isinstance(quote, dict) else ""),
        "provider_quote_token_symbol": cell(quote.get("symbol") if isinstance(quote, dict) else ""),
        "provider_quote_token_name": cell(quote.get("name") if isinstance(quote, dict) else ""),
        "price_usd": safe_num(pair.get("priceUsd")),
        "liquidity_usd": safe_num(
            (pair.get("liquidity") or {}).get("usd")
            if isinstance(pair.get("liquidity"), dict)
            else pair.get("liquidity")
        ),
        "fdv": safe_num(pair.get("fdv")),
        "market_cap": safe_num(pair.get("marketCap")),
        "volume_m5": safe_num(volume.get("m5") if isinstance(volume, dict) else None),
        "volume_h1": safe_num(volume.get("h1") if isinstance(volume, dict) else None),
        "volume_h6": safe_num(volume.get("h6") if isinstance(volume, dict) else None),
        "volume_h24": safe_num(volume.get("h24") if isinstance(volume, dict) else None),
        "txns_m5_buys": txn_side("m5", "buys"),
        "txns_m5_sells": txn_side("m5", "sells"),
        "txns_h1_buys": txn_side("h1", "buys"),
        "txns_h1_sells": txn_side("h1", "sells"),
        "txns_h6_buys": txn_side("h6", "buys"),
        "txns_h6_sells": txn_side("h6", "sells"),
        "txns_h24_buys": txn_side("h24", "buys"),
        "txns_h24_sells": txn_side("h24", "sells"),
        "price_change_m5": safe_num(price_change.get("m5") if isinstance(price_change, dict) else None),
        "price_change_h1": safe_num(price_change.get("h1") if isinstance(price_change, dict) else None),
        "price_change_h6": safe_num(price_change.get("h6") if isinstance(price_change, dict) else None),
        "price_change_h24": safe_num(price_change.get("h24") if isinstance(price_change, dict) else None),
        "pair_created_at": safe_num(pair.get("pairCreatedAt")),
        "info_websites_count": str(len(websites)),
        "info_socials_count": str(len(socials)),
        "info_socials_types": ";".join(social_types),
    }


def evaluate_identity(
    *,
    chain: str,
    expected_pair_id: str,
    pair_fields: dict[str, str],
) -> tuple[bool, str, str]:
    """Return (identity_ok, match_reason, status_if_fail_or_empty)."""
    provider_chain = pair_fields.get("provider_chain_id", "")
    provider_pair = pair_fields.get("provider_pair_address", "")
    base = pair_fields.get("provider_base_token_address", "")
    quote = pair_fields.get("provider_quote_token_address", "")

    if not provider_pair and not provider_chain:
        return False, "empty_provider_identity", "PROVIDER_PAYLOAD_EMPTY"

    if provider_chain and not chains_equivalent(chain, provider_chain):
        return False, f"chain_mismatch expected={normalize_chain(chain)} got={provider_chain}", "PROVIDER_CHAIN_MISMATCH"

    if provider_pair and not addresses_match(chain, expected_pair_id, provider_pair):
        # Solana/XRPL casing mismatch without auto-correct
        if is_non_evm(chain) and cell(expected_pair_id).lower() == cell(provider_pair).lower():
            return False, "solana_or_xrpl_casing_mismatch", "CASE_SENSITIVE_PAIR_ID_UNRESOLVED"
        return False, "pair_address_mismatch", "PROVIDER_PAIR_ADDRESS_MISMATCH"

    if not base or not quote:
        return False, "missing_base_or_quote_token", "PROVIDER_PAYLOAD_SCHEMA_UNSUPPORTED"

    # price/liquidity parse without crashing
    for field in ("price_usd", "liquidity_usd"):
        raw = pair_fields.get(field, "")
        if raw == "":
            continue
        try:
            float(raw)
        except (TypeError, ValueError):
            return False, f"{field}_unparseable", "PROVIDER_PAYLOAD_SCHEMA_UNSUPPORTED"

    if provider_chain and provider_pair and addresses_match(chain, expected_pair_id, provider_pair):
        return True, "chain_and_pair_match", ""
    return True, "identity_accepted_with_partial_provider_fields", ""


def classify_http_status(status: int | None) -> str | None:
    if status is None:
        return None
    if status == 404:
        return "PROVIDER_PAIR_NOT_FOUND"
    if status == 429:
        return "PROVIDER_RATE_LIMITED"
    if status in RETRYABLE_STATUS:
        return "PROVIDER_HTTP_ERROR"
    if status in NON_RETRYABLE_STATUS:
        return "PROVIDER_HTTP_ERROR"
    if status >= 400:
        return "PROVIDER_HTTP_ERROR"
    return None


# ---------------------------------------------------------------------------
# HTTP fetch with retries
# ---------------------------------------------------------------------------


FetchResult = dict[str, Any]
FetchFn = Callable[[str, float], FetchResult]


def default_fetch(url: str, timeout_seconds: float) -> FetchResult:
    if httpx is None:
        return {
            "ok": False,
            "status_code": None,
            "raw_text": "",
            "json": None,
            "exception_type": "ImportError",
            "exception_message": "httpx is not installed",
            "headers": {},
        }
    try:
        with httpx.Client(timeout=timeout_seconds, headers=HEADERS, follow_redirects=True) as client:
            resp = client.get(url)
            raw_text = resp.text or ""
            parsed: Any = None
            parse_error = ""
            try:
                parsed = resp.json()
            except Exception as exc:
                parse_error = f"{type(exc).__name__}: {exc}"
                parsed = {
                    "raw_text_preview": raw_text[:2000],
                    "json_parse_error": parse_error,
                }
            return {
                "ok": 200 <= resp.status_code < 300,
                "status_code": resp.status_code,
                "raw_text": raw_text,
                "json": parsed,
                "json_parse_error": parse_error,
                "exception_type": "",
                "exception_message": "",
                "headers": {k.lower(): v for k, v in resp.headers.items()},
            }
    except Exception as exc:
        etype = type(exc).__name__
        is_timeout = "Timeout" in etype or "timeout" in str(exc).lower()
        return {
            "ok": False,
            "status_code": None,
            "raw_text": "",
            "json": None,
            "exception_type": etype,
            "exception_message": str(exc)[:1000],
            "is_timeout": is_timeout,
            "headers": {},
        }


def parse_retry_after(headers: dict[str, str]) -> float | None:
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(str(raw).strip())
    except ValueError:
        return None


def compute_backoff(
    attempt_index: int,
    *,
    base: float,
    maximum: float,
    retry_after: float | None = None,
) -> float:
    if retry_after is not None and retry_after >= 0:
        delay = min(maximum, retry_after)
    else:
        delay = min(maximum, base * (2 ** max(0, attempt_index)))
    jitter = random.uniform(0, min(0.25, delay * 0.1)) if delay > 0 else 0.0
    return min(maximum, delay + jitter)


def fetch_with_retries(
    url: str,
    *,
    timeout_seconds: float,
    max_retries: int,
    backoff_base: float,
    backoff_max: float,
    fetch_fn: FetchFn | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> tuple[FetchResult, int, float, list[dict[str, Any]]]:
    """Returns (final_result, retry_count, final_backoff_seconds, attempt_audit)."""
    do_fetch = fetch_fn or default_fetch
    do_sleep = sleeper or time.sleep
    attempts: list[dict[str, Any]] = []
    retry_count = 0
    final_backoff = 0.0
    last: FetchResult = {
        "ok": False,
        "status_code": None,
        "raw_text": "",
        "json": None,
        "exception_type": "",
        "exception_message": "",
        "headers": {},
    }

    # max_retries = number of attempts (default 3)
    for attempt in range(max(1, max_retries)):
        result = do_fetch(url, timeout_seconds)
        last = result
        status = result.get("status_code")
        attempts.append(
            {
                "attempt": attempt + 1,
                "status_code": status,
                "exception_type": result.get("exception_type") or "",
                "ok": bool(result.get("ok")),
            }
        )

        if result.get("ok"):
            return result, retry_count, final_backoff, attempts

        # JSON parse error on HTTP 200 is not retryable at transport layer;
        # still return immediately if status is 200.
        if status == 200:
            return result, retry_count, final_backoff, attempts

        retryable = False
        if status in RETRYABLE_STATUS:
            retryable = True
        elif status is None and (result.get("is_timeout") or result.get("exception_type")):
            # timeout / connection errors
            retryable = True
        elif status in NON_RETRYABLE_STATUS:
            retryable = False

        if not retryable or attempt >= max_retries - 1:
            return result, retry_count, final_backoff, attempts

        retry_after = parse_retry_after(result.get("headers") or {})
        delay = compute_backoff(
            attempt,
            base=backoff_base,
            maximum=backoff_max,
            retry_after=retry_after if status == 429 else None,
        )
        final_backoff = delay
        retry_count += 1
        do_sleep(delay)

    return last, retry_count, final_backoff, attempts


# ---------------------------------------------------------------------------
# Per-target validation
# ---------------------------------------------------------------------------


def validate_target(
    row: dict[str, str],
    *,
    dry_run: bool = False,
    timeout_seconds: float = 20.0,
    max_retries: int = 3,
    backoff_base: float = 2.0,
    backoff_max: float = 30.0,
    fetch_fn: FetchFn | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return (validation_row, jsonl_record, attempt_meta)."""
    out = empty_validation_row(row)
    fetched_at = utc_iso()
    chain = cell(row.get("chain"))
    pair_id, pair_source = select_refetch_pair_id(row)
    out["refetch_pair_id"] = pair_id
    out["refetch_pair_id_source"] = pair_source
    out["refetch_url"] = build_refetch_url(chain, pair_id) if pair_id and chain else ""

    jsonl: dict[str, Any] = {
        "combined_target_id": out["combined_target_id"],
        "chain": chain,
        "refetch_pair_id": pair_id,
        "refetch_url": out["refetch_url"],
        "fetched_at_utc": fetched_at,
        "http_attempted": False,
        "http_status_code": None,
        "http_success": False,
        "acceptance_status": "",
        "rejection_reason": "",
        "exception_type": "",
        "exception_message": "",
        "raw_response_sha256": sha256_text(""),
        "raw_response_json": None,
    }
    meta: dict[str, Any] = {
        "retryable_failure": False,
        "rate_limited": False,
        "retry_after_seen": False,
        "attempts": [],
        "used_provider_pair_url_for_clean_solana": False,
        "casing_mutated": False,
        "authoritative_id_preserved": True,
    }

    # Track whether Clean Solana incorrectly used URL path
    if (
        has_clean_forward_source(row)
        and normalize_chain(chain) == "solana"
        and cell(row.get("pair_address"))
        and pair_source == "PROVIDER_PAIR_URL_PATH_SEGMENT"
    ):
        meta["used_provider_pair_url_for_clean_solana"] = True

    # Casing safety: authoritative id must equal selected source field exactly
    if pair_source == "PAIR_ADDRESS_CLEAN_FORWARD":
        meta["authoritative_id_preserved"] = pair_id == cell(row.get("pair_address"))
    elif pair_source == "USER_SUPPLIED_PAIR_ADDRESS":
        meta["authoritative_id_preserved"] = pair_id == cell(row.get("user_supplied_pair_address"))
    elif pair_source == "RESOLVED_PAIR_ADDRESS":
        meta["authoritative_id_preserved"] = pair_id == cell(row.get("resolved_pair_address"))
    meta["casing_mutated"] = not meta["authoritative_id_preserved"]

    if not as_bool(row.get("active", "true")):
        out["acceptance_status"] = "TARGET_INACTIVE_SKIPPED"
        out["rejection_reason"] = "target_inactive"
        jsonl["acceptance_status"] = out["acceptance_status"]
        jsonl["rejection_reason"] = out["rejection_reason"]
        return out, jsonl, meta

    if dry_run:
        out["acceptance_status"] = "DRY_RUN_NOT_FETCHED"
        out["rejection_reason"] = "dry_run"
        jsonl["acceptance_status"] = out["acceptance_status"]
        jsonl["rejection_reason"] = out["rejection_reason"]
        return out, jsonl, meta

    if is_blank(pair_id):
        out["acceptance_status"] = "REFRESH_PAIR_ID_MISSING"
        out["rejection_reason"] = "no_refetch_pair_id"
        jsonl["acceptance_status"] = out["acceptance_status"]
        jsonl["rejection_reason"] = out["rejection_reason"]
        return out, jsonl, meta

    if is_blank(chain):
        out["acceptance_status"] = "PROVIDER_EXCEPTION"
        out["rejection_reason"] = "missing_chain"
        jsonl["acceptance_status"] = out["acceptance_status"]
        jsonl["rejection_reason"] = out["rejection_reason"]
        return out, jsonl, meta

    out["http_attempted"] = "true"
    jsonl["http_attempted"] = True

    result, retry_count, final_backoff, attempts = fetch_with_retries(
        out["refetch_url"],
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        backoff_base=backoff_base,
        backoff_max=backoff_max,
        fetch_fn=fetch_fn,
        sleeper=sleeper,
    )
    meta["attempts"] = attempts
    out["retry_count"] = str(retry_count)
    out["final_backoff_seconds"] = f"{final_backoff:.3f}"

    status = result.get("status_code")
    out["http_status_code"] = "" if status is None else str(status)
    jsonl["http_status_code"] = status
    raw_text = result.get("raw_text") or ""
    raw_sha = sha256_text(raw_text)
    out["raw_response_sha256"] = raw_sha
    jsonl["raw_response_sha256"] = raw_sha
    jsonl["raw_response_json"] = result.get("json")

    if any(a.get("status_code") == 429 for a in attempts):
        meta["rate_limited"] = True
    if status == 429:
        meta["retryable_failure"] = True
    if status in RETRYABLE_STATUS:
        meta["retryable_failure"] = True
    if result.get("is_timeout") or (
        result.get("exception_type") and status is None and retry_count > 0
    ):
        meta["retryable_failure"] = True
    headers = result.get("headers") or {}
    if parse_retry_after(headers) is not None:
        meta["retry_after_seen"] = True

    out["exception_type"] = cell(result.get("exception_type"))
    out["exception_message"] = cell(result.get("exception_message"))
    jsonl["exception_type"] = out["exception_type"]
    jsonl["exception_message"] = out["exception_message"]

    # Transport / exception outcomes
    if result.get("is_timeout") or (
        result.get("exception_type") and "Timeout" in str(result.get("exception_type"))
    ):
        out["acceptance_status"] = "PROVIDER_TIMEOUT"
        out["rejection_reason"] = out["exception_message"] or "timeout"
        out["http_success"] = "false"
        jsonl["http_success"] = False
        jsonl["acceptance_status"] = out["acceptance_status"]
        jsonl["rejection_reason"] = out["rejection_reason"]
        return out, jsonl, meta

    if result.get("exception_type") and status is None:
        out["acceptance_status"] = "PROVIDER_EXCEPTION"
        out["rejection_reason"] = out["exception_message"] or out["exception_type"]
        out["http_success"] = "false"
        jsonl["http_success"] = False
        jsonl["acceptance_status"] = out["acceptance_status"]
        jsonl["rejection_reason"] = out["rejection_reason"]
        return out, jsonl, meta

    if status == 429:
        out["acceptance_status"] = "PROVIDER_RATE_LIMITED"
        out["rejection_reason"] = "http_429"
        out["http_success"] = "false"
        jsonl["http_success"] = False
        jsonl["acceptance_status"] = out["acceptance_status"]
        jsonl["rejection_reason"] = out["rejection_reason"]
        return out, jsonl, meta

    if status == 404:
        out["acceptance_status"] = "PROVIDER_PAIR_NOT_FOUND"
        out["rejection_reason"] = (
            "http_404_no_auto_case_correct"
            if is_non_evm(chain)
            else "http_404"
        )
        out["http_success"] = "false"
        jsonl["http_success"] = False
        jsonl["acceptance_status"] = out["acceptance_status"]
        jsonl["rejection_reason"] = out["rejection_reason"]
        return out, jsonl, meta

    http_status_class = classify_http_status(status if isinstance(status, int) else None)
    if http_status_class and status and status >= 400:
        out["acceptance_status"] = http_status_class
        out["rejection_reason"] = f"http_{status}"
        out["http_success"] = "false"
        jsonl["http_success"] = False
        jsonl["acceptance_status"] = out["acceptance_status"]
        jsonl["rejection_reason"] = out["rejection_reason"]
        return out, jsonl, meta

    # HTTP 200 path
    out["http_success"] = "true" if result.get("ok") else "false"
    jsonl["http_success"] = bool(result.get("ok"))

    if result.get("json_parse_error"):
        out["acceptance_status"] = "PROVIDER_JSON_PARSE_ERROR"
        out["rejection_reason"] = str(result.get("json_parse_error"))[:500]
        jsonl["acceptance_status"] = out["acceptance_status"]
        jsonl["rejection_reason"] = out["rejection_reason"]
        return out, jsonl, meta

    payload = result.get("json")
    pairs = extract_pairs(payload)
    out["response_pair_count"] = str(len(pairs))

    if not pairs:
        out["acceptance_status"] = "PROVIDER_PAIR_NOT_FOUND"
        out["rejection_reason"] = "empty_pairs_payload"
        jsonl["acceptance_status"] = out["acceptance_status"]
        jsonl["rejection_reason"] = out["rejection_reason"]
        return out, jsonl, meta

    # Prefer exact address match among returned pairs
    chosen: dict[str, Any] | None = None
    for p in pairs:
        addr = cell(p.get("pairAddress")) if isinstance(p, dict) else ""
        if addr and addresses_match(chain, pair_id, addr):
            chosen = p
            break
    if chosen is None:
        chosen = pairs[0]

    fields = parse_pair_fields(chosen)
    out.update(fields)

    identity_ok, match_reason, fail_status = evaluate_identity(
        chain=chain, expected_pair_id=pair_id, pair_fields=fields
    )
    out["provider_identity_match"] = "true" if identity_ok else "false"
    out["provider_identity_match_reason"] = match_reason

    if not identity_ok:
        out["acceptance_status"] = fail_status or "PROVIDER_PAIR_ADDRESS_MISMATCH"
        out["rejection_reason"] = match_reason
        out["clean_forward_candidate_ready"] = "false"
        jsonl["acceptance_status"] = out["acceptance_status"]
        jsonl["rejection_reason"] = out["rejection_reason"]
        return out, jsonl, meta

    # Ready check
    ready = (
        identity_ok
        and not is_blank(fields.get("provider_base_token_address"))
        and not is_blank(fields.get("provider_quote_token_address"))
        and len(pairs) >= 1
    )
    warning = False
    if len(pairs) > 1:
        warning = True
        match_reason = f"{match_reason};multiple_pairs_returned={len(pairs)}"

    out["provider_identity_match_reason"] = match_reason
    if ready and not warning:
        out["acceptance_status"] = "PROVIDER_PAIR_RESOLVED"
        out["rejection_reason"] = ""
        out["clean_forward_candidate_ready"] = "true"
    elif ready and warning:
        out["acceptance_status"] = "PROVIDER_PAIR_RESOLVED_IDENTITY_WARNING"
        out["rejection_reason"] = "multiple_pairs_in_payload"
        out["clean_forward_candidate_ready"] = "true"
    else:
        out["acceptance_status"] = "PROVIDER_PAYLOAD_SCHEMA_UNSUPPORTED"
        out["rejection_reason"] = match_reason or "not_ready"
        out["clean_forward_candidate_ready"] = "false"

    jsonl["acceptance_status"] = out["acceptance_status"]
    jsonl["rejection_reason"] = out["rejection_reason"]
    return out, jsonl, meta


# ---------------------------------------------------------------------------
# Package writer / decision gate
# ---------------------------------------------------------------------------


def decide_gate(
    *,
    input_exists: bool,
    targets_loaded: int,
    ready_count: int,
    resolved_count: int,
    rejected_count: int,
    status_counts: dict[str, int],
    rate_limited_count: int,
    http_calls: int,
    casing_loss: bool,
    used_url_for_clean_solana: bool,
    jsonl_ok: bool,
    identity_contradiction_count: int,
) -> dict[str, Any]:
    if not input_exists or targets_loaded == 0:
        classification = "AE16C_REFETCH_VALIDATION_BLOCKED_NO_TARGETS"
        reason = "missing_or_empty_input"
    elif casing_loss or used_url_for_clean_solana:
        classification = "AE16C_REFETCH_VALIDATION_BLOCKED_CASING_LOSS"
        reason = "solana_or_xrpl_casing_corrupted_or_url_used_as_authoritative_for_clean_solana"
    elif not jsonl_ok:
        classification = "AE16C_REFETCH_VALIDATION_BLOCKED_JSONL_WRITE_FAILURE"
        reason = "jsonl_line_count_mismatch_or_write_failure"
    elif identity_contradiction_count > 0 and resolved_count == 0:
        classification = "AE16C_REFETCH_VALIDATION_BLOCKED_IDENTITY_CONTRADICTION"
        reason = "provider_identity_contradictions_with_no_safe_resolutions"
    elif (
        http_calls > 0
        and rate_limited_count >= max(1, http_calls // 2)
        and resolved_count == 0
    ):
        classification = "AE16C_REFETCH_VALIDATION_BLOCKED_RATE_LIMITED"
        reason = "broad_rate_limiting_prevented_validation"
    elif http_calls > 0 and resolved_count == 0 and ready_count == 0:
        provider_fail = sum(
            status_counts.get(k, 0)
            for k in (
                "PROVIDER_TIMEOUT",
                "PROVIDER_EXCEPTION",
                "PROVIDER_HTTP_ERROR",
                "PROVIDER_RATE_LIMITED",
            )
        )
        if provider_fail >= max(1, int(targets_loaded * 0.8)):
            classification = "AE16C_REFETCH_VALIDATION_BLOCKED_PROVIDER_FAILURE"
            reason = "broad_provider_or_network_failure"
        else:
            classification = "AE16C_REFETCH_VALIDATION_PASS_WITH_REJECTIONS"
            reason = "no_ready_targets_but_explicit_rejections"
    elif ready_count > 0 and rejected_count == 0:
        classification = "AE16C_REFETCH_VALIDATION_PASS"
        reason = "all_active_targets_resolved"
    elif ready_count > 0 and rejected_count > 0:
        classification = "AE16C_REFETCH_VALIDATION_PASS_WITH_REJECTIONS"
        reason = "partial_resolution_with_explicit_rejections"
    elif resolved_count > 0:
        classification = "AE16C_REFETCH_VALIDATION_PASS_WITH_REJECTIONS"
        reason = "some_resolved_some_rejected"
    else:
        classification = "AE16C_REFETCH_VALIDATION_PASS_WITH_REJECTIONS"
        reason = "validation_completed_with_rejections_only"

    return {
        "phase": PHASE,
        "classification": classification,
        "reason": reason,
        "ae16_original_e6_closed": False,
        "model_evidence_attached": False,
        "rf_xgb_tab_consensus_repaired": False,
        "ae17_started": False,
        "ready_count": ready_count,
        "resolved_count": resolved_count,
        "rejected_count": rejected_count,
        "safe_to_consider_ae16d_curated_collector_overlay": classification
        in {
            "AE16C_REFETCH_VALIDATION_PASS",
            "AE16C_REFETCH_VALIDATION_PASS_WITH_REJECTIONS",
        }
        and ready_count > 0,
    }


def build_summary(manifest: dict[str, Any], gate: dict[str, Any], output_root: Path) -> str:
    lines = [
        "AE16C DexScreener Refetch Validation Summary",
        "============================================",
        f"phase: {manifest['phase']}",
        f"classification: {gate['classification']}",
        f"output root: {output_root}",
        f"input: {manifest['input_combined_universe_path']} (exists={manifest['input_exists']})",
        f"input targets loaded: {manifest['input_targets_loaded']}",
        f"active targets loaded: {manifest['active_targets_loaded']}",
        f"HTTP calls attempted: {manifest['http_calls_attempted']}",
        f"sleep seconds used: {manifest['sleep_seconds_used']}",
        f"max retries used: {manifest['max_retries_used']}",
        f"provider pairs resolved: {manifest['provider_pairs_resolved']}",
        f"clean_forward_candidate_ready: {manifest['provider_pairs_ready_for_clean_forward']}",
        f"rejected targets: {manifest['rejected_targets']}",
        f"status counts: {json.dumps(manifest['status_counts'], ensure_ascii=False)}",
        f"rate-limited count: {manifest['rate_limited_count']}",
        f"retryable failures: {manifest['retryable_failures_count']}",
        f"JSONL expected/written: {manifest['jsonl_lines_expected']}/{manifest['jsonl_lines_written']}",
        f"JSONL error safety passed: {manifest['jsonl_error_safety_passed']}",
        f"Solana casing preserved: {manifest['solana_casing_preserved']}",
        f"XRPL casing preserved: {manifest['xrpl_casing_preserved']}",
        f"used provider_pair_url as authoritative for Clean Solana: "
        f"{manifest['used_provider_pair_url_as_authoritative_for_clean_solana']}",
        "confirmation: collector was not modified",
        "confirmation: trader.db was not mutated",
        "confirmation: server was not required",
        "confirmation: internal API endpoints were not called",
        "confirmation: no model training / backtest / live trading occurred",
        f"semantic_status remains: {SEMANTIC_PENDING}",
        f"safe_to_consider_ae16d: {gate['safe_to_consider_ae16d_curated_collector_overlay']}",
        "AE16 original E6 not closed; AE17 not started.",
        "",
    ]
    return "\n".join(lines)


def run(
    input_path: Path,
    *,
    output_root: Path | None = None,
    dry_run: bool = False,
    limit: int | None = None,
    timeout_seconds: float = 20.0,
    sleep_seconds: float = 1.0,
    max_retries: int = 3,
    backoff_base_seconds: float = 2.0,
    backoff_max_seconds: float = 30.0,
    fetch_fn: FetchFn | None = None,
    sleeper: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    timestamp = utc_stamp()
    if output_root is None:
        output_root = Path("data/audits") / f"ae16c_dexscreener_refetch_validation_{timestamp}"

    data_dir = output_root / "data"
    reports_dir = output_root / "reports"
    audits_dir = output_root / "audits"
    for d in (data_dir, reports_dir, audits_dir):
        d.mkdir(parents=True, exist_ok=True)

    input_exists = input_path.exists()
    if not input_exists:
        manifest = {
            "phase": PHASE,
            "timestamp": timestamp,
            "input_combined_universe_path": str(input_path).replace("\\", "/"),
            "input_exists": False,
            "input_targets_loaded": 0,
            "active_targets_loaded": 0,
            "http_calls_attempted": 0,
            "provider_pairs_resolved": 0,
            "provider_pairs_ready_for_clean_forward": 0,
            "rejected_targets": 0,
            "status_counts": {},
            "chain_counts_input": {},
            "chain_counts_resolved": {},
            "chain_counts_rejected": {},
            "clean_forward_existing_targets": 0,
            "user_seed_targets": 0,
            "merged_targets": 0,
            "solana_targets_checked": 0,
            "xrpl_targets_checked": 0,
            "solana_casing_preserved": True,
            "xrpl_casing_preserved": True,
            "used_provider_pair_url_as_authoritative_for_clean_solana": False,
            "sleep_seconds_used": sleep_seconds,
            "max_retries_used": max_retries,
            "retryable_failures_count": 0,
            "rate_limited_count": 0,
            "retry_after_header_seen_count": 0,
            "jsonl_lines_written": 0,
            "jsonl_lines_expected": 0,
            "jsonl_error_safety_passed": True,
            "collector_modified": False,
            "trader_db_mutated": False,
            "server_required": False,
            "internal_api_called": False,
            "model_training_run": False,
            "backtest_run": False,
            "live_trading_enabled": False,
            "dry_run": dry_run,
        }
        gate = decide_gate(
            input_exists=False,
            targets_loaded=0,
            ready_count=0,
            resolved_count=0,
            rejected_count=0,
            status_counts={},
            rate_limited_count=0,
            http_calls=0,
            casing_loss=False,
            used_url_for_clean_solana=False,
            jsonl_ok=True,
            identity_contradiction_count=0,
        )
        write_json(reports_dir / "ae16c_manifest.json", manifest)
        write_json(reports_dir / "ae16c_decision_gate.json", gate)
        write_text(reports_dir / "ae16c_summary_for_upload.txt", build_summary(manifest, gate, output_root))
        write_csv(data_dir / "ae16c_refetch_validation_rows.csv", [], VALIDATION_FIELDS)
        (data_dir / "ae16c_provider_responses.jsonl").write_text("", encoding="utf-8")
        return {"output_root": output_root, "manifest": manifest, "gate": gate, "rows": []}

    rows_in = read_csv(input_path)
    if limit is not None and limit >= 0:
        rows_in = rows_in[:limit]

    jsonl_path = data_dir / "ae16c_provider_responses.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    validation_rows: list[dict[str, Any]] = []
    identity_audit: list[dict[str, Any]] = []
    fetch_audit: list[dict[str, Any]] = []
    casing_audit: list[dict[str, Any]] = []
    retry_audit: list[dict[str, Any]] = []
    jsonl_safety_audit: list[dict[str, Any]] = []

    http_calls = 0
    retryable_failures = 0
    rate_limited = 0
    retry_after_seen = 0
    jsonl_written = 0
    solana_casing_ok = True
    xrpl_casing_ok = True
    used_url_clean_solana = False
    identity_contradictions = 0
    do_sleep = sleeper or time.sleep

    for idx, row in enumerate(rows_in):
        try:
            vrow, jrec, meta = validate_target(
                row,
                dry_run=dry_run,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                backoff_base=backoff_base_seconds,
                backoff_max=backoff_max_seconds,
                fetch_fn=fetch_fn,
                sleeper=sleeper,
            )
        except Exception as exc:  # never crash whole run
            vrow = empty_validation_row(row)
            vrow["acceptance_status"] = "PROVIDER_EXCEPTION"
            vrow["rejection_reason"] = f"unhandled:{type(exc).__name__}"
            vrow["exception_type"] = type(exc).__name__
            vrow["exception_message"] = str(exc)[:1000]
            jrec = {
                "combined_target_id": vrow["combined_target_id"],
                "chain": vrow["chain"],
                "refetch_pair_id": vrow.get("refetch_pair_id", ""),
                "refetch_url": vrow.get("refetch_url", ""),
                "fetched_at_utc": utc_iso(),
                "http_attempted": False,
                "http_status_code": None,
                "http_success": False,
                "acceptance_status": vrow["acceptance_status"],
                "rejection_reason": vrow["rejection_reason"],
                "exception_type": vrow["exception_type"],
                "exception_message": vrow["exception_message"],
                "raw_response_sha256": sha256_text(""),
                "raw_response_json": {"traceback": traceback.format_exc()[:2000]},
            }
            meta = {
                "retryable_failure": False,
                "rate_limited": False,
                "retry_after_seen": False,
                "attempts": [],
                "used_provider_pair_url_for_clean_solana": False,
                "casing_mutated": False,
                "authoritative_id_preserved": True,
            }

        validation_rows.append(vrow)

        if as_bool(vrow.get("http_attempted")):
            http_calls += 1
        if meta.get("retryable_failure"):
            retryable_failures += 1
        if meta.get("rate_limited"):
            rate_limited += 1
        if meta.get("retry_after_seen"):
            retry_after_seen += 1
        if meta.get("used_provider_pair_url_for_clean_solana"):
            used_url_clean_solana = True
        if meta.get("casing_mutated"):
            ch = normalize_chain(vrow.get("chain", ""))
            if ch == "solana":
                solana_casing_ok = False
            if ch == "xrpl":
                xrpl_casing_ok = False
        if vrow.get("acceptance_status") in {
            "PROVIDER_CHAIN_MISMATCH",
            "PROVIDER_PAIR_ADDRESS_MISMATCH",
        }:
            identity_contradictions += 1

        wrote = append_jsonl(jsonl_path, jrec)
        if wrote:
            jsonl_written += 1
        jsonl_safety_audit.append(
            {
                "combined_target_id": vrow.get("combined_target_id"),
                "jsonl_write_ok": wrote,
                "acceptance_status": vrow.get("acceptance_status"),
                "has_required_keys": all(
                    k in jrec
                    for k in (
                        "combined_target_id",
                        "chain",
                        "refetch_pair_id",
                        "refetch_url",
                        "fetched_at_utc",
                        "http_attempted",
                        "http_status_code",
                        "http_success",
                        "acceptance_status",
                        "rejection_reason",
                        "exception_type",
                        "exception_message",
                        "raw_response_sha256",
                        "raw_response_json",
                    )
                ),
            }
        )

        identity_audit.append(
            {
                "combined_target_id": vrow.get("combined_target_id"),
                "chain": vrow.get("chain"),
                "refetch_pair_id": vrow.get("refetch_pair_id"),
                "refetch_pair_id_source": vrow.get("refetch_pair_id_source"),
                "provider_pair_address": vrow.get("provider_pair_address"),
                "provider_identity_match": vrow.get("provider_identity_match"),
                "provider_identity_match_reason": vrow.get("provider_identity_match_reason"),
                "acceptance_status": vrow.get("acceptance_status"),
            }
        )
        fetch_audit.append(
            {
                "combined_target_id": vrow.get("combined_target_id"),
                "refetch_url": vrow.get("refetch_url"),
                "http_attempted": vrow.get("http_attempted"),
                "http_status_code": vrow.get("http_status_code"),
                "http_success": vrow.get("http_success"),
                "retry_count": vrow.get("retry_count"),
                "acceptance_status": vrow.get("acceptance_status"),
            }
        )
        casing_audit.append(
            {
                "combined_target_id": vrow.get("combined_target_id"),
                "chain": vrow.get("chain"),
                "refetch_pair_id": vrow.get("refetch_pair_id"),
                "refetch_pair_id_source": vrow.get("refetch_pair_id_source"),
                "pair_address": vrow.get("pair_address"),
                "user_supplied_pair_address": vrow.get("user_supplied_pair_address"),
                "provider_pair_url": vrow.get("provider_pair_url"),
                "authoritative_id_preserved": meta.get("authoritative_id_preserved"),
                "used_provider_pair_url_for_clean_solana": meta.get(
                    "used_provider_pair_url_for_clean_solana"
                ),
            }
        )
        retry_audit.append(
            {
                "combined_target_id": vrow.get("combined_target_id"),
                "retry_count": vrow.get("retry_count"),
                "final_backoff_seconds": vrow.get("final_backoff_seconds"),
                "rate_limited": meta.get("rate_limited"),
                "retry_after_seen": meta.get("retry_after_seen"),
                "attempts_json": json.dumps(meta.get("attempts") or []),
                "acceptance_status": vrow.get("acceptance_status"),
            }
        )

        # Pace between successful live requests (not dry-run, not last)
        if (
            not dry_run
            and as_bool(vrow.get("http_attempted"))
            and idx < len(rows_in) - 1
            and sleep_seconds > 0
        ):
            do_sleep(sleep_seconds)

    # Derived outputs
    resolved_statuses = {
        "PROVIDER_PAIR_RESOLVED",
        "PROVIDER_PAIR_RESOLVED_IDENTITY_WARNING",
    }
    resolved_rows = [r for r in validation_rows if r.get("acceptance_status") in resolved_statuses]
    ready_rows = [r for r in validation_rows if as_bool(r.get("clean_forward_candidate_ready"))]
    rejected_rows = [
        r
        for r in validation_rows
        if r.get("acceptance_status") not in resolved_statuses
        and r.get("acceptance_status") not in {"DRY_RUN_NOT_FETCHED", "TARGET_INACTIVE_SKIPPED"}
    ]
    # For dry-run, treat non-ready as informational rather than hard rejects in CSV
    if dry_run:
        rejected_rows = []

    status_counts = dict(Counter(r.get("acceptance_status") or "" for r in validation_rows))
    chain_input = Counter(normalize_chain(r.get("chain")) for r in validation_rows)
    chain_resolved = Counter(normalize_chain(r.get("chain")) for r in resolved_rows)
    chain_rejected = Counter(normalize_chain(r.get("chain")) for r in rejected_rows)

    chain_summary = []
    for ch in sorted(set(chain_input) | set(chain_resolved) | set(chain_rejected)):
        chain_summary.append(
            {
                "chain": ch,
                "input_count": chain_input.get(ch, 0),
                "resolved_count": chain_resolved.get(ch, 0),
                "ready_count": sum(
                    1
                    for r in ready_rows
                    if normalize_chain(r.get("chain")) == ch
                ),
                "rejected_count": chain_rejected.get(ch, 0),
            }
        )

    write_csv(data_dir / "ae16c_refetch_validation_rows.csv", validation_rows, VALIDATION_FIELDS)
    write_csv(data_dir / "ae16c_resolved_pairs.csv", resolved_rows, RESOLVED_FIELDS)
    write_csv(data_dir / "ae16c_rejected_targets.csv", rejected_rows, REJECTED_FIELDS)
    write_csv(data_dir / "ae16c_clean_forward_candidate_ready_targets.csv", ready_rows, READY_FIELDS)
    write_csv(data_dir / "ae16c_chain_summary.csv", chain_summary, CHAIN_SUMMARY_FIELDS)

    write_csv(
        audits_dir / "ae16c_identity_resolution_audit.csv",
        identity_audit,
        [
            "combined_target_id",
            "chain",
            "refetch_pair_id",
            "refetch_pair_id_source",
            "provider_pair_address",
            "provider_identity_match",
            "provider_identity_match_reason",
            "acceptance_status",
        ],
    )
    write_csv(
        audits_dir / "ae16c_provider_fetch_audit.csv",
        fetch_audit,
        [
            "combined_target_id",
            "refetch_url",
            "http_attempted",
            "http_status_code",
            "http_success",
            "retry_count",
            "acceptance_status",
        ],
    )
    write_csv(
        audits_dir / "ae16c_casing_safety_audit.csv",
        casing_audit,
        [
            "combined_target_id",
            "chain",
            "refetch_pair_id",
            "refetch_pair_id_source",
            "pair_address",
            "user_supplied_pair_address",
            "provider_pair_url",
            "authoritative_id_preserved",
            "used_provider_pair_url_for_clean_solana",
        ],
    )
    write_csv(
        audits_dir / "ae16c_rate_limit_retry_audit.csv",
        retry_audit,
        [
            "combined_target_id",
            "retry_count",
            "final_backoff_seconds",
            "rate_limited",
            "retry_after_seen",
            "attempts_json",
            "acceptance_status",
        ],
    )
    write_csv(
        audits_dir / "ae16c_jsonl_error_safety_audit.csv",
        jsonl_safety_audit,
        ["combined_target_id", "jsonl_write_ok", "acceptance_status", "has_required_keys"],
    )

    write_json(
        audits_dir / "ae16c_no_collector_modification_audit.json",
        {
            "collector_modified": False,
            "note": "AE16C standalone script; Clean Forward collector not imported or altered",
        },
    )
    write_json(
        audits_dir / "ae16c_no_trader_db_mutation_audit.json",
        {
            "trader_db_mutated": False,
            "note": "AE16C writes only to its timestamped audit package",
        },
    )

    jsonl_expected = len(rows_in)
    jsonl_ok = jsonl_written == jsonl_expected and all(
        a.get("jsonl_write_ok") and a.get("has_required_keys") for a in jsonl_safety_audit
    )

    active_loaded = sum(1 for r in rows_in if as_bool(r.get("active", "true")))
    ts_counter = Counter(cell(r.get("target_source")) for r in rows_in)

    manifest = {
        "phase": PHASE,
        "timestamp": timestamp,
        "input_combined_universe_path": str(input_path).replace("\\", "/"),
        "input_exists": True,
        "input_targets_loaded": len(rows_in),
        "active_targets_loaded": active_loaded,
        "http_calls_attempted": http_calls,
        "provider_pairs_resolved": len(resolved_rows),
        "provider_pairs_ready_for_clean_forward": len(ready_rows),
        "rejected_targets": len(rejected_rows),
        "status_counts": status_counts,
        "chain_counts_input": dict(chain_input),
        "chain_counts_resolved": dict(chain_resolved),
        "chain_counts_rejected": dict(chain_rejected),
        "clean_forward_existing_targets": ts_counter.get("CLEAN_FORWARD_EXISTING", 0),
        "user_seed_targets": ts_counter.get("USER_DEXSCREENER_SEED", 0),
        "merged_targets": ts_counter.get("MERGED", 0),
        "solana_targets_checked": chain_input.get("solana", 0),
        "xrpl_targets_checked": chain_input.get("xrpl", 0),
        "solana_casing_preserved": solana_casing_ok,
        "xrpl_casing_preserved": xrpl_casing_ok,
        "used_provider_pair_url_as_authoritative_for_clean_solana": used_url_clean_solana,
        "sleep_seconds_used": sleep_seconds,
        "max_retries_used": max_retries,
        "retryable_failures_count": retryable_failures,
        "rate_limited_count": rate_limited,
        "retry_after_header_seen_count": retry_after_seen,
        "jsonl_lines_written": jsonl_written,
        "jsonl_lines_expected": jsonl_expected,
        "jsonl_error_safety_passed": jsonl_ok,
        "collector_modified": False,
        "trader_db_mutated": False,
        "server_required": False,
        "internal_api_called": False,
        "model_training_run": False,
        "backtest_run": False,
        "live_trading_enabled": False,
        "dry_run": dry_run,
        "timeout_seconds": timeout_seconds,
        "backoff_base_seconds": backoff_base_seconds,
        "backoff_max_seconds": backoff_max_seconds,
    }

    gate = decide_gate(
        input_exists=True,
        targets_loaded=len(rows_in),
        ready_count=len(ready_rows),
        resolved_count=len(resolved_rows),
        rejected_count=len(rejected_rows),
        status_counts=status_counts,
        rate_limited_count=rate_limited,
        http_calls=http_calls,
        casing_loss=(not solana_casing_ok) or (not xrpl_casing_ok),
        used_url_for_clean_solana=used_url_clean_solana,
        jsonl_ok=jsonl_ok,
        identity_contradiction_count=identity_contradictions,
    )

    write_json(reports_dir / "ae16c_manifest.json", manifest)
    write_json(reports_dir / "ae16c_decision_gate.json", gate)
    write_text(reports_dir / "ae16c_summary_for_upload.txt", build_summary(manifest, gate, output_root))

    return {
        "output_root": output_root,
        "manifest": manifest,
        "gate": gate,
        "rows": validation_rows,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AE16C DexScreener refetch validation (standalone)")
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-root", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--timeout-seconds", type=float, default=20.0)
    p.add_argument("--sleep-seconds", type=float, default=1.0)
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--backoff-base-seconds", type=float, default=2.0)
    p.add_argument("--backoff-max-seconds", type=float, default=30.0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sleep = args.sleep_seconds
    if sleep < 0:
        sleep = 0.0
    # Default is 1.0; allow override (including test/smoke below 1.0) but never negative.

    out = run(
        args.input,
        output_root=args.output_root,
        dry_run=args.dry_run,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
        sleep_seconds=sleep,
        max_retries=args.max_retries,
        backoff_base_seconds=args.backoff_base_seconds,
        backoff_max_seconds=args.backoff_max_seconds,
    )
    manifest = out["manifest"]
    gate = out["gate"]
    print(f"phase: {manifest['phase']}")
    print(f"classification: {gate['classification']}")
    print(f"output_root: {out['output_root']}")
    print(f"input_targets_loaded: {manifest['input_targets_loaded']}")
    print(f"http_calls_attempted: {manifest['http_calls_attempted']}")
    print(f"provider_pairs_resolved: {manifest['provider_pairs_resolved']}")
    print(f"ready: {manifest['provider_pairs_ready_for_clean_forward']}")
    print(f"rejected: {manifest['rejected_targets']}")
    print(
        f"jsonl: {manifest['jsonl_lines_written']}/{manifest['jsonl_lines_expected']} "
        f"safety={manifest['jsonl_error_safety_passed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
