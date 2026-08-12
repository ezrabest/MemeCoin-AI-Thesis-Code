"""AE13K — Chain-aware DexScreener pair URL builder + rate-limited verifier.

Rules:
- Do NOT construct provider_pair_url until provider pair lookup succeeds.
- Validate Solana/EVM address formats before lookup.
- Cache + pace DexScreener pair verification; never treat 429 as a clean valid row.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import httpx

BASE_API = "https://api.dexscreener.com/latest/dex"
PAIR_PAGE_TEMPLATE = "https://dexscreener.com/{chain}/{pair}"

SOURCE_PROVIDER = "dexscreener"

# Configurable settings (env overrides)
def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


DEXSCREENER_PAIR_VERIFY_CACHE_TTL_SECONDS = _env_float(
    "DEXSCREENER_PAIR_VERIFY_CACHE_TTL_SECONDS", 20.0
)
DEXSCREENER_PAIR_VERIFY_MIN_INTERVAL_MS = _env_int(
    "DEXSCREENER_PAIR_VERIFY_MIN_INTERVAL_MS", 250
)
DEXSCREENER_PAIR_VERIFY_MAX_RETRIES = _env_int(
    "DEXSCREENER_PAIR_VERIFY_MAX_RETRIES", 2
)
DEXSCREENER_PAIR_VERIFY_MAX_CONCURRENCY = _env_int(
    "DEXSCREENER_PAIR_VERIFY_MAX_CONCURRENCY", 2
)

_CHAIN_ALIASES: dict[str, str] = {
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
}

SOLANA_LIKE_CHAINS = {"solana"}
EVM_LIKE_CHAINS = {
    "ethereum",
    "bsc",
    "base",
    "arbitrum",
    "polygon",
    "optimism",
    "avalanche",
}

_BASE58_ALPHABET = set("123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz")

_HEADERS = {
    "Accept": "application/json",
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}
_TIMEOUT = httpx.Timeout(12.0)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_chain_id(chain_id: str | None) -> str | None:
    """Normalize chain IDs to DexScreener chain IDs."""
    raw = str(chain_id or "").strip().lower()
    if not raw:
        return None
    if raw in _CHAIN_ALIASES:
        return _CHAIN_ALIASES[raw]
    # Pass through unknown chains (e.g. robinhood) after lowercasing —
    # address format validation will still apply when family is known.
    return raw


def is_evm_address(address: str | None) -> bool:
    a = str(address or "").strip()
    if not (a.startswith("0x") or a.startswith("0X")):
        return False
    body = a[2:]
    if len(body) != 40:
        return False
    try:
        int(body, 16)
        return True
    except ValueError:
        return False


def is_solana_address(address: str | None) -> bool:
    """Preliminary Solana-like base58 check (not cryptographic proof)."""
    a = str(address or "").strip()
    if not a or a.startswith("0x") or a.startswith("0X"):
        return False
    if not (32 <= len(a) <= 44):
        return False
    return all(ch in _BASE58_ALPHABET for ch in a)


def address_format_for_chain(chain_id: str | None, address: str | None) -> dict[str, Any]:
    """Preliminary chain/address format validation before provider lookup."""
    chain = normalize_chain_id(chain_id)
    addr = str(address or "").strip()
    result: dict[str, Any] = {
        "normalized_chain_id": chain,
        "address": addr or None,
        "format_ok": False,
        "address_family": None,
        "reason": None,
    }
    if not chain:
        result["reason"] = "missing_chain_id"
        return result
    if not addr:
        result["reason"] = "missing_pair_address"
        return result

    if chain in SOLANA_LIKE_CHAINS:
        if is_evm_address(addr):
            result["address_family"] = "evm_hex"
            result["reason"] = "chain_address_format_mismatch"
            result["verification_status"] = "chain_address_format_mismatch"
            return result
        if not is_solana_address(addr):
            result["address_family"] = "unknown"
            result["reason"] = "invalid_solana_address_format"
            result["verification_status"] = "chain_address_format_mismatch"
            return result
        result["format_ok"] = True
        result["address_family"] = "solana_base58"
        return result

    if chain in EVM_LIKE_CHAINS:
        if is_solana_address(addr) and not is_evm_address(addr):
            result["address_family"] = "solana_base58"
            result["reason"] = "chain_address_format_mismatch"
            result["verification_status"] = "chain_address_format_mismatch"
            return result
        if not is_evm_address(addr):
            result["address_family"] = "unknown"
            result["reason"] = "invalid_evm_address_format"
            result["verification_status"] = "chain_address_format_mismatch"
            return result
        result["format_ok"] = True
        result["address_family"] = "evm_hex"
        return result

    # Unknown chain family: accept either format preliminarily; provider decides.
    if is_evm_address(addr):
        result["format_ok"] = True
        result["address_family"] = "evm_hex"
        return result
    if is_solana_address(addr):
        result["format_ok"] = True
        result["address_family"] = "solana_base58"
        return result
    result["reason"] = "unrecognized_address_format"
    result["verification_status"] = "chain_address_format_mismatch"
    return result


def build_dexscreener_pair_url(chain_id: str, pair_address: str) -> str:
    """Construct DexScreener pair chart URL.

    Call ONLY after provider pair verification succeeds.
    """
    chain = normalize_chain_id(chain_id)
    pair = str(pair_address or "").strip()
    if not chain or not pair:
        return ""
    return PAIR_PAGE_TEMPLATE.format(chain=chain, pair=pair)


def _payload_hash(fields: dict[str, Any]) -> str:
    blob = json.dumps(fields, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _extract_pair(data: Any) -> dict | None:
    if not data:
        return None
    if isinstance(data, dict):
        pair = data.get("pair")
        if isinstance(pair, dict) and pair.get("pairAddress"):
            return pair
        pairs = data.get("pairs")
        if isinstance(pairs, list):
            for p in pairs:
                if isinstance(p, dict) and p.get("pairAddress"):
                    return p
        if data.get("pairAddress"):
            return data
    if isinstance(data, list):
        for p in data:
            if isinstance(p, dict) and p.get("pairAddress"):
                return p
    return None


def _as_float(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _txn_window(txns: Any, key: str) -> dict[str, Any]:
    if not isinstance(txns, dict):
        return {"buys": None, "sells": None, "total": None}
    window = txns.get(key) if isinstance(txns.get(key), dict) else {}
    buys = window.get("buys")
    sells = window.get("sells")
    total = None
    if buys is not None or sells is not None:
        total = int(buys or 0) + int(sells or 0)
    return {"buys": buys, "sells": sells, "total": total}


def _volume_window(volume: Any, key: str) -> float | None:
    if isinstance(volume, dict):
        return _as_float(volume.get(key))
    return None


@dataclass
class DexScreenerPairVerificationResult:
    source_provider: str = SOURCE_PROVIDER
    requested_chain_id: str | None = None
    normalized_chain_id: str | None = None
    requested_pair_address: str | None = None
    provider_pair_id: str | None = None
    pair_address: str | None = None
    provider_pair_url: str | None = None
    provider_pair_url_source: str | None = None
    dex_id: str | None = None
    base_token_address: str | None = None
    base_token_symbol: str | None = None
    base_token_name: str | None = None
    quote_token_address: str | None = None
    quote_token_symbol: str | None = None
    quote_token_name: str | None = None
    price_usd: Any = None
    liquidity_usd: Any = None
    volume_5m: float | None = None
    volume_1h: float | None = None
    volume_6h: float | None = None
    volume_24h: float | None = None
    volume: Any = None
    txns_5m_buys: Any = None
    txns_5m_sells: Any = None
    txns_1h_buys: Any = None
    txns_1h_sells: Any = None
    txns_24h_buys: Any = None
    txns_24h_sells: Any = None
    txns: Any = None
    price_change_5m: Any = None
    price_change_1h: Any = None
    price_change_6h: Any = None
    price_change_24h: Any = None
    price_change: Any = None
    pair_created_at: Any = None
    fetched_at: str | None = None
    ingested_at: str | None = None
    provider_payload_hash: str | None = None
    verification_status: str = "provider_pair_not_found"
    tradability_status: str = "not_tradable_without_provider_pair"
    freshness_status: str = "stale_or_unknown"
    address_role: str | None = None
    identity_status: str = "unresolved"
    clean_feed_eligible: bool = False
    exclusion_reason: str | None = None
    lookup_ok: bool = False
    verification_attempted_at: str | None = None
    verification_http_status: int | None = None
    verification_error: str | None = None
    verification_cache_hit: bool = False
    verification_retry_after_seconds: float | None = None
    verification_attempt_count: int = 0
    address_format: dict[str, Any] = field(default_factory=dict)
    pair_label: str | None = None
    age_seconds: float | None = None
    raw_pair: dict[str, Any] | None = None

    def to_dict(self, *, include_raw: bool = False) -> dict[str, Any]:
        d = asdict(self)
        if not include_raw:
            d.pop("raw_pair", None)
        return d


class _PairVerifyLimiter:
    """In-process cache + request pacing + bounded concurrency for pair verify."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[float, DexScreenerPairVerificationResult]] = {}
        self._last_request_at = 0.0
        self._inflight = 0
        self._stats = {
            "cache_hits": 0,
            "cache_misses": 0,
            "http_calls": 0,
            "rate_limited_responses": 0,
            "provider_unavailable_responses": 0,
            "retries": 0,
            "max_inflight_observed": 0,
        }

    def settings_snapshot(self) -> dict[str, Any]:
        return {
            "DEXSCREENER_PAIR_VERIFY_CACHE_TTL_SECONDS": DEXSCREENER_PAIR_VERIFY_CACHE_TTL_SECONDS,
            "DEXSCREENER_PAIR_VERIFY_MIN_INTERVAL_MS": DEXSCREENER_PAIR_VERIFY_MIN_INTERVAL_MS,
            "DEXSCREENER_PAIR_VERIFY_MAX_RETRIES": DEXSCREENER_PAIR_VERIFY_MAX_RETRIES,
            "DEXSCREENER_PAIR_VERIFY_MAX_CONCURRENCY": DEXSCREENER_PAIR_VERIFY_MAX_CONCURRENCY,
        }

    def stats_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                **dict(self._stats),
                "cache_entries": len(self._cache),
                "inflight": self._inflight,
                "settings": self.settings_snapshot(),
            }

    def cache_snapshot(self) -> dict[str, Any]:
        with self._lock:
            now = time.monotonic()
            entries = []
            for key, (expires, result) in list(self._cache.items()):
                entries.append(
                    {
                        "cache_key": key,
                        "ttl_remaining_sec": max(0.0, round(expires - now, 3)),
                        "verification_status": result.verification_status,
                        "clean_feed_eligible": result.clean_feed_eligible,
                        "pair_address": result.pair_address,
                        "normalized_chain_id": result.normalized_chain_id,
                        "verification_cache_hit_on_store": result.verification_cache_hit,
                    }
                )
            return {
                "timestamp_utc": _utc_now(),
                "entry_count": len(entries),
                "entries": entries,
                "stats": self.stats_snapshot(),
            }

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def cache_age_seconds(self, chain: str, pair: str) -> float | None:
        """Seconds since this pair entry was cached (None if not cached/expired)."""
        key = self._cache_key(chain, pair)
        with self._lock:
            item = self._cache.get(key)
            if not item:
                return None
            expires, _result = item
            now = time.monotonic()
            if now > expires:
                return None
            ttl_used = max(expires - now, 0.001)
            # Entry was stored at (expires - ttl_at_store); approximate age from remaining TTL.
            remaining = expires - now
            configured_ttl = DEXSCREENER_PAIR_VERIFY_CACHE_TTL_SECONDS
            age = max(0.0, configured_ttl - remaining)
            return round(age, 3)

    def reset_stats(self) -> None:
        with self._lock:
            for k in self._stats:
                self._stats[k] = 0

    def _cache_key(self, chain: str, pair: str) -> str:
        return f"{SOURCE_PROVIDER}|{chain.lower()}|{pair.lower()}"

    def get_cached(self, chain: str, pair: str) -> DexScreenerPairVerificationResult | None:
        key = self._cache_key(chain, pair)
        with self._lock:
            item = self._cache.get(key)
            if not item:
                self._stats["cache_misses"] += 1
                return None
            expires, result = item
            if time.monotonic() > expires:
                del self._cache[key]
                self._stats["cache_misses"] += 1
                return None
            self._stats["cache_hits"] += 1
            raw = result.raw_pair
            payload = result.to_dict(include_raw=False)
            cached = DexScreenerPairVerificationResult(**payload)
            cached.verification_cache_hit = True
            cached.raw_pair = raw
            return cached

    def put_cache(self, chain: str, pair: str, result: DexScreenerPairVerificationResult) -> None:
        # Do not cache deferred/rate-limited/unavailable as "valid" — but DO cache
        # briefly so we do not stampede; use short TTL for deferred.
        key = self._cache_key(chain, pair)
        ttl = DEXSCREENER_PAIR_VERIFY_CACHE_TTL_SECONDS
        if result.verification_status in (
            "provider_rate_limited",
            "provider_unavailable",
            "verification_deferred",
        ):
            ttl = min(ttl, 5.0)
        with self._lock:
            self._cache[key] = (time.monotonic() + ttl, result)

    def acquire_slot(self) -> None:
        """Block until concurrency + min-interval allow a request."""
        while True:
            with self._lock:
                if self._inflight < DEXSCREENER_PAIR_VERIFY_MAX_CONCURRENCY:
                    now = time.monotonic()
                    wait = (
                        self._last_request_at
                        + (DEXSCREENER_PAIR_VERIFY_MIN_INTERVAL_MS / 1000.0)
                        - now
                    )
                    if wait <= 0:
                        self._inflight += 1
                        self._last_request_at = now
                        self._stats["max_inflight_observed"] = max(
                            self._stats["max_inflight_observed"], self._inflight
                        )
                        return
                else:
                    wait = 0.05
            time.sleep(max(0.01, min(wait, 0.25)))

    def release_slot(self) -> None:
        with self._lock:
            self._inflight = max(0, self._inflight - 1)

    def note_http(self) -> None:
        with self._lock:
            self._stats["http_calls"] += 1

    def note_retry(self) -> None:
        with self._lock:
            self._stats["retries"] += 1

    def note_rate_limited(self) -> None:
        with self._lock:
            self._stats["rate_limited_responses"] += 1

    def note_unavailable(self) -> None:
        with self._lock:
            self._stats["provider_unavailable_responses"] += 1


_LIMITER = _PairVerifyLimiter()


def get_pair_verify_limiter() -> _PairVerifyLimiter:
    return _LIMITER


def _http_get_pair(chain: str, pair: str) -> dict[str, Any]:
    """Perform one DexScreener pair GET; returns status metadata + optional pair."""
    try:
        from app.runtime.shutdown import CONTROLLED_SHUTDOWN_SKIP, should_skip_network
        from app.runtime.ui_get_network_guard import is_ui_get_path_active, record_network_attempt

        if is_ui_get_path_active():
            record_network_attempt("dexscreener")
            return {
                "ok": False,
                "status_code": 0,
                "pair": None,
                "error": "ui_get_network_forbidden",
                "retry_after_seconds": None,
                "rate_limited": False,
                "provider_unavailable": True,
            }
        if should_skip_network(context="dexscreener_pair_verify"):
            return {
                "ok": False,
                "status_code": 0,
                "pair": None,
                "error": CONTROLLED_SHUTDOWN_SKIP,
                "retry_after_seconds": None,
                "rate_limited": False,
                "provider_unavailable": True,
            }
    except Exception:
        pass
    path = f"/pairs/{chain}/{pair}"
    url = f"{BASE_API}{path}"
    try:
        with httpx.Client(timeout=_TIMEOUT) as client:
            r = client.get(url, headers=_HEADERS)
        status = r.status_code
        retry_after = None
        if "Retry-After" in r.headers:
            try:
                retry_after = float(r.headers["Retry-After"])
            except ValueError:
                retry_after = None
        if status == 429:
            return {
                "ok": False,
                "status_code": status,
                "pair": None,
                "error": "too_many_requests",
                "retry_after_seconds": retry_after,
                "rate_limited": True,
                "provider_unavailable": False,
            }
        if status >= 500:
            return {
                "ok": False,
                "status_code": status,
                "pair": None,
                "error": f"provider_5xx_{status}",
                "retry_after_seconds": retry_after,
                "rate_limited": False,
                "provider_unavailable": True,
            }
        if status == 404:
            return {
                "ok": False,
                "status_code": status,
                "pair": None,
                "error": "not_found",
                "retry_after_seconds": None,
                "rate_limited": False,
                "provider_unavailable": False,
            }
        if status >= 400:
            return {
                "ok": False,
                "status_code": status,
                "pair": None,
                "error": f"http_{status}",
                "retry_after_seconds": retry_after,
                "rate_limited": False,
                "provider_unavailable": True,
            }
        data = r.json()
        pair_obj = _extract_pair(data)
        if not pair_obj:
            return {
                "ok": False,
                "status_code": status,
                "pair": None,
                "error": "empty_pair_payload",
                "retry_after_seconds": None,
                "rate_limited": False,
                "provider_unavailable": False,
            }
        return {
            "ok": True,
            "status_code": status,
            "pair": pair_obj,
            "error": None,
            "retry_after_seconds": None,
            "rate_limited": False,
            "provider_unavailable": False,
        }
    except httpx.TimeoutException as exc:
        return {
            "ok": False,
            "status_code": None,
            "pair": None,
            "error": f"timeout:{exc}",
            "retry_after_seconds": None,
            "rate_limited": False,
            "provider_unavailable": True,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status_code": None,
            "pair": None,
            "error": f"network:{exc}",
            "retry_after_seconds": None,
            "rate_limited": False,
            "provider_unavailable": True,
        }


def _freshness(fetched_at: str | None) -> dict[str, Any]:
    if not fetched_at:
        return {"freshness_status": "stale_or_unknown", "age_seconds": None}
    try:
        t = datetime.fromisoformat(str(fetched_at).replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - t).total_seconds()
    except ValueError:
        return {"freshness_status": "stale_or_unknown", "age_seconds": None}
    if age < 120:
        status = "fresh"
    elif age < 900:
        status = "soft_stale"
    else:
        status = "stale_or_unknown"
    return {"freshness_status": status, "age_seconds": age}


def _address_role_for_pair(chain_id: str | None, dex_id: str | None) -> str:
    ch = str(chain_id or "").lower()
    dex = str(dex_id or "").lower()
    if "pump" in dex:
        return "pool_address"
    if ch in SOLANA_LIKE_CHAINS or ch.endswith("solana"):
        return "pool_address"
    if ch in EVM_LIKE_CHAINS:
        return "pair_contract"
    return "provider_pair_id"


def _fill_from_pair(
    result: DexScreenerPairVerificationResult,
    payload: dict[str, Any],
    *,
    requested_chain: str,
    requested_pair: str,
) -> DexScreenerPairVerificationResult:
    fetched_at = _utc_now()
    returned_pair = str(payload.get("pairAddress") or "").strip()
    returned_chain = normalize_chain_id(payload.get("chainId")) or requested_chain
    base = payload.get("baseToken") if isinstance(payload.get("baseToken"), dict) else {}
    quote = payload.get("quoteToken") if isinstance(payload.get("quoteToken"), dict) else {}
    liq = payload.get("liquidity") if isinstance(payload.get("liquidity"), dict) else {}
    vol = payload.get("volume")
    txns = payload.get("txns")
    price_change = (
        payload.get("priceChange") if isinstance(payload.get("priceChange"), dict) else {}
    )
    price_usd = payload.get("priceUsd")
    liq_usd = liq.get("usd") if isinstance(liq, dict) else None
    base_addr = str(base.get("address") or "").strip()
    quote_addr = str(quote.get("address") or "").strip()

    provider_url = str(payload.get("url") or "").strip()
    url_source = None
    if provider_url:
        url_source = "provider_returned_url"
    # URL construction ONLY after we know lookup returned a pair object —
    # still gated by admission checks below before clean_feed_eligible.

    hash_fields = {
        "priceUsd": price_usd,
        "liquidity.usd": liq_usd,
        "volume": vol,
        "txns": txns,
        "priceChange": price_change,
    }
    ph = _payload_hash(hash_fields)

    m5 = _txn_window(txns, "m5")
    h1 = _txn_window(txns, "h1")
    h24 = _txn_window(txns, "h24")

    result.fetched_at = fetched_at
    result.ingested_at = fetched_at
    result.normalized_chain_id = returned_chain
    result.pair_address = returned_pair or None
    result.provider_pair_id = returned_pair or None
    result.dex_id = payload.get("dexId")
    result.base_token_address = base_addr or None
    result.base_token_symbol = base.get("symbol")
    result.base_token_name = base.get("name")
    result.quote_token_address = quote_addr or None
    result.quote_token_symbol = quote.get("symbol")
    result.quote_token_name = quote.get("name")
    result.price_usd = price_usd
    result.liquidity_usd = liq_usd
    result.volume = vol
    result.volume_5m = _volume_window(vol, "m5")
    result.volume_1h = _volume_window(vol, "h1")
    result.volume_6h = _volume_window(vol, "h6")
    result.volume_24h = _volume_window(vol, "h24")
    result.txns = txns
    result.txns_5m_buys = m5["buys"]
    result.txns_5m_sells = m5["sells"]
    result.txns_1h_buys = h1["buys"]
    result.txns_1h_sells = h1["sells"]
    result.txns_24h_buys = h24["buys"]
    result.txns_24h_sells = h24["sells"]
    result.price_change = price_change
    result.price_change_5m = price_change.get("m5")
    result.price_change_1h = price_change.get("h1")
    result.price_change_6h = price_change.get("h6")
    result.price_change_24h = price_change.get("h24")
    result.pair_created_at = payload.get("pairCreatedAt")
    result.provider_payload_hash = ph
    result.raw_pair = payload
    result.pair_label = f"{base.get('symbol') or '?'}/{quote.get('symbol') or '?'}"

    missing: list[str] = []
    if not returned_pair:
        missing.append("pairAddress")
    elif returned_pair.lower() != requested_pair.lower():
        missing.append("pair_address_mismatch")
    if returned_chain and requested_chain and returned_chain.lower() != requested_chain.lower():
        missing.append("chain_id_mismatch")
    if not base_addr:
        missing.append("baseToken.address")
    if not quote_addr:
        missing.append("quoteToken.address")
    if price_usd is None or price_usd == "":
        missing.append("priceUsd")
    if liq_usd is None:
        missing.append("liquidity.usd")

    if missing:
        result.lookup_ok = False
        result.clean_feed_eligible = False
        result.verification_status = "provider_pair_incomplete"
        result.tradability_status = "not_tradable_without_provider_pair"
        result.identity_status = "incomplete_provider_pair"
        result.exclusion_reason = "missing_required_fields:" + ",".join(missing)
        # Do NOT attach constructed URL when verification failed
        result.provider_pair_url = None
        result.provider_pair_url_source = None
        return result

    # Verification succeeded — prefer provider URL, else construct AFTER success
    if not provider_url:
        provider_url = build_dexscreener_pair_url(returned_chain, returned_pair)
        url_source = "constructed_after_verified_lookup"
    result.provider_pair_url = provider_url
    result.provider_pair_url_source = url_source

    fresh = _freshness(fetched_at)
    result.freshness_status = fresh["freshness_status"]
    result.age_seconds = fresh["age_seconds"]
    result.address_role = _address_role_for_pair(returned_chain, payload.get("dexId"))
    result.lookup_ok = True
    result.clean_feed_eligible = True
    result.verification_status = "provider_pair_verified"
    result.tradability_status = "provider_pair_verified_display_only"
    result.identity_status = "pair_and_tokens_separated"
    result.exclusion_reason = None
    return result


def validate_dexscreener_pair(
    chain_id: str,
    pair_address: str,
    *,
    use_cache: bool = True,
    _http_get=_http_get_pair,
) -> DexScreenerPairVerificationResult:
    """Validate address format then verify via DexScreener pair lookup.

    Never constructs provider_pair_url before successful verification.
    Never promotes 429 / 5xx / timeout rows into clean_feed_eligible.
    """
    attempted_at = _utc_now()
    chain = normalize_chain_id(chain_id)
    pair = str(pair_address or "").strip()
    result = DexScreenerPairVerificationResult(
        requested_chain_id=str(chain_id or "").strip() or None,
        normalized_chain_id=chain,
        requested_pair_address=pair or None,
        provider_pair_id=pair or None,
        verification_attempted_at=attempted_at,
        provider_pair_url=None,
        provider_pair_url_source=None,
    )

    fmt = address_format_for_chain(chain, pair)
    result.address_format = fmt
    if not fmt.get("format_ok"):
        result.verification_status = "chain_address_format_mismatch"
        result.tradability_status = "ambiguous_address_role"
        result.identity_status = "chain_address_format_mismatch"
        result.exclusion_reason = fmt.get("reason") or "chain_address_format_mismatch"
        result.clean_feed_eligible = False
        result.lookup_ok = False
        result.verification_attempt_count = 0
        return result

    assert chain and pair

    if use_cache:
        cached = _LIMITER.get_cached(chain, pair)
        if cached is not None:
            cached.verification_attempted_at = attempted_at
            return cached

    attempt = 0
    last_http: dict[str, Any] | None = None
    max_retries = DEXSCREENER_PAIR_VERIFY_MAX_RETRIES

    while attempt <= max_retries:
        attempt += 1
        _LIMITER.acquire_slot()
        try:
            _LIMITER.note_http()
            last_http = _http_get(chain, pair)
        finally:
            _LIMITER.release_slot()

        result.verification_attempt_count = attempt
        result.verification_http_status = last_http.get("status_code")
        result.verification_error = last_http.get("error")
        result.verification_retry_after_seconds = last_http.get("retry_after_seconds")

        if last_http.get("ok") and last_http.get("pair"):
            result = _fill_from_pair(
                result,
                last_http["pair"],
                requested_chain=chain,
                requested_pair=pair,
            )
            result.verification_attempt_count = attempt
            result.verification_http_status = last_http.get("status_code")
            result.verification_error = None
            result.verification_cache_hit = False
            _LIMITER.put_cache(chain, pair, result)
            return result

        if last_http.get("rate_limited"):
            _LIMITER.note_rate_limited()
            retry_after = last_http.get("retry_after_seconds")
            if attempt <= max_retries:
                _LIMITER.note_retry()
                if retry_after is not None:
                    sleep_s = float(retry_after)
                else:
                    sleep_s = (0.5 * (2 ** (attempt - 1))) + random.uniform(0, 0.25)
                time.sleep(min(sleep_s, 10.0))
                continue
            result.verification_status = "provider_rate_limited"
            result.tradability_status = "verification_deferred"
            result.identity_status = "verification_deferred"
            result.exclusion_reason = "provider_rate_limited"
            result.clean_feed_eligible = False
            result.lookup_ok = False
            result.provider_pair_url = None
            result.provider_pair_url_source = None
            _LIMITER.put_cache(chain, pair, result)
            return result

        if last_http.get("provider_unavailable"):
            _LIMITER.note_unavailable()
            if attempt <= max_retries:
                _LIMITER.note_retry()
                sleep_s = (0.4 * (2 ** (attempt - 1))) + random.uniform(0, 0.2)
                time.sleep(min(sleep_s, 8.0))
                continue
            result.verification_status = "provider_unavailable"
            result.tradability_status = "verification_deferred"
            result.identity_status = "verification_deferred"
            result.exclusion_reason = last_http.get("error") or "provider_unavailable"
            result.clean_feed_eligible = False
            result.lookup_ok = False
            result.provider_pair_url = None
            result.provider_pair_url_source = None
            _LIMITER.put_cache(chain, pair, result)
            return result

        # Not found / incomplete — no retry storm
        break

    result.verification_status = "provider_pair_not_found"
    result.tradability_status = "not_tradable_without_provider_pair"
    result.identity_status = "unresolved"
    result.exclusion_reason = (last_http or {}).get("error") or "provider_pair_not_found"
    result.clean_feed_eligible = False
    result.lookup_ok = False
    result.provider_pair_url = None
    result.provider_pair_url_source = None
    result.verification_cache_hit = False
    _LIMITER.put_cache(chain, pair, result)
    return result
