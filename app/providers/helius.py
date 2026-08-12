"""
Helius enhanced-transaction validation provider (signature-level only, Phase 2).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from .helius_budget import HeliusBudgetManager, estimate_credit_cost

log = logging.getLogger("helius")

HELIUS_BASE = "https://mainnet.helius-rpc.com"
HELIUS_TRANSACTIONS_PATH = "/v0/transactions"
DEFAULT_TIMEOUT_SECONDS = 20.0
CACHE_TTL_SECONDS = 24 * 60 * 60

HELIUS_OK = "HELIUS_OK"
HELIUS_SKIPPED_NOT_ENABLED = "HELIUS_SKIPPED_NOT_ENABLED"
HELIUS_SKIPPED_NO_API_KEY = "HELIUS_SKIPPED_NO_API_KEY"
HELIUS_SKIPPED_BUDGET = "HELIUS_SKIPPED_BUDGET"
HELIUS_RATE_LIMITED = "HELIUS_RATE_LIMITED"
HELIUS_FORBIDDEN = "HELIUS_FORBIDDEN"
HELIUS_TIMEOUT = "HELIUS_TIMEOUT"
HELIUS_UNAVAILABLE = "HELIUS_UNAVAILABLE"
HELIUS_JSON_ERROR = "HELIUS_JSON_ERROR"
HELIUS_CACHE_HIT = "HELIUS_CACHE_HIT"

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_CACHE_DIR = DATA_DIR / "cache" / "helius"

KNOWN_AGGREGATOR_SOURCES = frozenset({"JUPITER", "JUPITER_V6", "RAYDIUM", "ORCA"})


def get_helius_api_key() -> str | None:
    key = os.getenv("HELIUS_API_KEY", "").strip()
    if not key or key.startswith("$"):
        return None
    return key


def build_helius_transactions_url(api_key: str | None = None) -> str | None:
    key = api_key or get_helius_api_key()
    if not key:
        return None
    return f"{HELIUS_BASE}{HELIUS_TRANSACTIONS_PATH}?api-key={quote(key, safe='')}"


def _time_bucket() -> str:
    return str(int(time.time() // CACHE_TTL_SECONDS))


def _cache_key(signatures: list[str], limit: int, endpoint: str) -> str:
    payload = json.dumps(
        {
            "endpoint": endpoint,
            "signatures": sorted(signatures),
            "limit": limit,
            "bucket": _time_bucket(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class HeliusStats:
    signatures_requested: int = 0
    signatures_validated: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    budget_used: int = 0


@dataclass
class HeliusClient:
    enabled: bool = False
    cache_dir: Path = field(default_factory=lambda: DEFAULT_CACHE_DIR)
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    budget_manager: HeliusBudgetManager | None = None
    stats: HeliusStats = field(default_factory=HeliusStats)

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> list[dict[str, Any]] | None:
        path = self._cache_path(key)
        try:
            if not path.is_file():
                return None
            raw = json.loads(path.read_text(encoding="utf-8"))
            cached_at = float(raw.get("cached_at", 0))
            if time.time() - cached_at > CACHE_TTL_SECONDS:
                return None
            data = raw.get("response")
            if isinstance(data, list):
                return data
        except Exception as exc:
            log.debug("Helius cache read failed: %s", exc)
        return None

    def _write_cache(self, key: str, response: list[dict[str, Any]]) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            payload = {"cached_at": time.time(), "response": response}
            self._cache_path(key).write_text(json.dumps(payload), encoding="utf-8")
        except Exception as exc:
            log.debug("Helius cache write failed: %s", exc)

    def fetch_transactions(self, signatures: list[str]) -> dict[str, Any]:
        if not self.enabled:
            return {"status": HELIUS_SKIPPED_NOT_ENABLED, "transactions": []}

        url = build_helius_transactions_url()
        if not url:
            return {"status": HELIUS_SKIPPED_NO_API_KEY, "transactions": []}

        if not signatures:
            return {"status": HELIUS_OK, "transactions": []}

        endpoint = HELIUS_TRANSACTIONS_PATH
        cache_key = _cache_key(signatures, len(signatures), endpoint)
        cached = self._read_cache(cache_key)
        if cached is not None:
            self.stats.cache_hits += 1
            return {"status": HELIUS_CACHE_HIT, "transactions": cached, "cache_hit": True}

        self.stats.cache_misses += 1
        cost = estimate_credit_cost(len(signatures))
        if self.budget_manager and not self.budget_manager.can_spend(cost):
            return {"status": HELIUS_SKIPPED_BUDGET, "transactions": []}

        body = {"transactions": signatures}
        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(url, json=body, headers={"Content-Type": "application/json"})
        except httpx.TimeoutException:
            return {"status": HELIUS_TIMEOUT, "transactions": []}
        except httpx.HTTPError:
            return {"status": HELIUS_UNAVAILABLE, "transactions": []}

        if response.status_code == 429:
            return {"status": HELIUS_RATE_LIMITED, "transactions": []}
        if response.status_code == 403:
            return {"status": HELIUS_FORBIDDEN, "transactions": []}
        if response.status_code >= 400:
            return {"status": HELIUS_UNAVAILABLE, "transactions": [], "http_status": response.status_code}

        try:
            payload = response.json()
        except json.JSONDecodeError:
            return {"status": HELIUS_JSON_ERROR, "transactions": []}

        if not isinstance(payload, list):
            return {"status": HELIUS_JSON_ERROR, "transactions": []}

        self._write_cache(cache_key, payload)
        if self.budget_manager:
            self.budget_manager.record_spend(cost, endpoint=endpoint.lstrip("/"))
            self.stats.budget_used += cost

        return {"status": HELIUS_OK, "transactions": payload, "cache_hit": False}


def _helius_tx_by_signature(transactions: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for tx in transactions:
        sig = tx.get("signature")
        if sig:
            indexed[sig] = tx
    return indexed


def _extract_helius_side(tx: dict[str, Any], base_mint: str | None, quote_mint: str | None) -> str | None:
    tx_type = str(tx.get("type") or "").upper()
    if "SWAP" in tx_type:
        events = tx.get("events") or {}
        swap = events.get("swap") or {}
        if swap.get("nativeInput") and not swap.get("nativeOutput"):
            return "SELL_BASE"
        if swap.get("nativeOutput") and not swap.get("nativeInput"):
            return "BUY_BASE"
    transfers = tx.get("tokenTransfers") or []
    if not base_mint or not quote_mint:
        return None
    base_in = Decimal("0")
    base_out = Decimal("0")
    quote_in = Decimal("0")
    quote_out = Decimal("0")
    for transfer in transfers:
        mint = transfer.get("mint")
        amount = transfer.get("tokenAmount")
        if amount is None:
            continue
        try:
            amt = Decimal(str(amount))
        except Exception:
            continue
        if mint == base_mint:
            if transfer.get("toUserAccount"):
                base_in += amt
            if transfer.get("fromUserAccount"):
                base_out += amt
        elif mint == quote_mint:
            if transfer.get("toUserAccount"):
                quote_in += amt
            if transfer.get("fromUserAccount"):
                quote_out += amt
    if base_in > base_out and quote_out > quote_in:
        return "BUY_BASE"
    if base_out > base_in and quote_in > quote_out:
        return "SELL_BASE"
    return None


def compare_raw_with_helius(raw_parsed: dict[str, Any], helius_tx: dict[str, Any] | None) -> dict[str, Any]:
    signature = raw_parsed.get("signature")
    comparison: dict[str, Any] = {
        "signature": signature,
        "raw_side": raw_parsed.get("side"),
        "raw_fee_payer": raw_parsed.get("fee_payer"),
        "raw_trader_wallet": raw_parsed.get("trader_wallet"),
        "raw_approx_usd_value": raw_parsed.get("approx_usd_value"),
        "raw_quote_amount_native": raw_parsed.get("quote_amount_native"),
        "helius_available": helius_tx is not None,
    }
    if not helius_tx:
        comparison["mismatch"] = True
        comparison["mismatch_fields"] = ["helius_missing"]
        return comparison

    helius_fee_payer = helius_tx.get("feePayer")
    helius_source = helius_tx.get("source")
    helius_transfers = helius_tx.get("tokenTransfers") or []
    helius_mints = sorted({t.get("mint") for t in helius_transfers if t.get("mint")})
    helius_side = _extract_helius_side(
        helius_tx,
        raw_parsed.get("base_token_mint"),
        raw_parsed.get("quote_token_mint"),
    )
    helius_owners = sorted(
        {
            account
            for transfer in helius_transfers
            for account in (transfer.get("fromUserAccount"), transfer.get("toUserAccount"))
            if account
        }
    )

    comparison.update(
        {
            "helius_fee_payer": helius_fee_payer,
            "helius_source": helius_source,
            "helius_token_mints": helius_mints,
            "helius_token_transfer_count": len(helius_transfers),
            "helius_inferred_side": helius_side,
            "helius_token_owner_candidates": helius_owners,
        }
    )

    mismatches: list[str] = []
    if helius_fee_payer and raw_parsed.get("fee_payer") and helius_fee_payer != raw_parsed.get("fee_payer"):
        mismatches.append("fee_payer")
    raw_side = raw_parsed.get("side")
    if helius_side and raw_side in {"BUY_BASE", "SELL_BASE"} and helius_side != raw_side:
        mismatches.append("side")
    raw_base = raw_parsed.get("base_token_mint")
    raw_quote = raw_parsed.get("quote_token_mint")
    if raw_base and raw_base not in helius_mints and helius_mints:
        mismatches.append("base_mint")
    if raw_quote and raw_quote not in helius_mints and helius_mints:
        mismatches.append("quote_mint")

    comparison["mismatch"] = bool(mismatches)
    comparison["mismatch_fields"] = mismatches
    return comparison


def select_validation_signatures(
    parsed_rows: list[dict[str, Any]],
    *,
    limit: int = 5,
) -> list[str]:
    eligible_statuses = {"OK", "PARTIAL", "UNKNOWN_FORMAT"}
    selected: list[str] = []
    for row in parsed_rows:
        if row.get("failed_transaction"):
            continue
        if row.get("parse_status") not in eligible_statuses:
            continue
        sig = row.get("signature")
        if sig and sig not in selected:
            selected.append(sig)
        if len(selected) >= limit:
            break
    return selected


def validate_signatures_against_raw(
    parsed_rows: list[dict[str, Any]],
    *,
    enabled: bool,
    validation_limit: int = 5,
    cache_dir: Path | None = None,
    budget_manager: HeliusBudgetManager | None = None,
) -> dict[str, Any]:
    client = HeliusClient(
        enabled=enabled,
        cache_dir=cache_dir or DEFAULT_CACHE_DIR,
        budget_manager=budget_manager,
    )

    if not enabled:
        return {
            "helius_validation_enabled": False,
            "helius_signatures_requested": 0,
            "helius_signatures_validated": 0,
            "helius_cache_hits": 0,
            "helius_cache_misses": 0,
            "helius_budget_used": 0,
            "helius_comparisons": [],
            "helius_mismatches": [],
            "helius_status": HELIUS_SKIPPED_NOT_ENABLED,
        }

    signatures = select_validation_signatures(parsed_rows, limit=validation_limit)
    client.stats.signatures_requested = len(signatures)

    fetch_result = client.fetch_transactions(signatures)
    status = fetch_result.get("status", HELIUS_UNAVAILABLE)
    helius_txs = fetch_result.get("transactions") or []
    helius_index = _helius_tx_by_signature(helius_txs)

    raw_by_sig = {row.get("signature"): row for row in parsed_rows if row.get("signature")}
    comparisons: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []

    for sig in signatures:
        raw = raw_by_sig.get(sig, {})
        comparison = compare_raw_with_helius(raw, helius_index.get(sig))
        comparisons.append(comparison)
        if comparison.get("mismatch"):
            mismatches.append(comparison)
        if comparison.get("helius_available"):
            client.stats.signatures_validated += 1

    budget_used = client.stats.budget_used
    if status == HELIUS_CACHE_HIT:
        budget_used = 0

    result = {
        "helius_validation_enabled": True,
        "helius_signatures_requested": client.stats.signatures_requested,
        "helius_signatures_validated": client.stats.signatures_validated,
        "helius_cache_hits": client.stats.cache_hits,
        "helius_cache_misses": client.stats.cache_misses,
        "helius_budget_used": budget_used,
        "helius_comparisons": comparisons,
        "helius_mismatches": mismatches,
        "helius_status": status,
    }
    if budget_manager:
        result["helius_budget_snapshot"] = budget_manager.budget_snapshot()
    return result
