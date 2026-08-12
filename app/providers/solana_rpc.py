"""
Shared Solana JSON-RPC client with retries, pacing, and optional response cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger("solana_rpc")

DEFAULT_RPC_URL = "https://api.mainnet-beta.solana.com"
PUBLIC_RPC_URL = DEFAULT_RPC_URL
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_RETRIES = 5
DEFAULT_GET_TX_PACE_SECONDS = 0.5
CACHE_TTL_SECONDS = 24 * 60 * 60

SOLANA_RPC_OK = "SOLANA_RPC_OK"
SOLANA_RPC_RATE_LIMITED = "SOLANA_RPC_RATE_LIMITED"
SOLANA_RPC_FORBIDDEN = "SOLANA_RPC_FORBIDDEN"
SOLANA_RPC_TIMEOUT = "SOLANA_RPC_TIMEOUT"
SOLANA_RPC_UNAVAILABLE = "SOLANA_RPC_UNAVAILABLE"
SOLANA_RPC_JSON_ERROR = "SOLANA_RPC_JSON_ERROR"
SOLANA_RPC_JSONRPC_ERROR = "SOLANA_RPC_JSONRPC_ERROR"
SOLANA_RPC_NULL_RESULT = "SOLANA_RPC_NULL_RESULT"

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
CACHE_DIR = DATA_DIR / "cache" / "solana_rpc"


@dataclass
class RpcCallResult:
    status: str
    result: Any = None
    error: dict[str, Any] | None = None
    http_status: int | None = None
    retries: int = 0


@dataclass
class SolanaRpcStats:
    rpc_calls_attempted: int = 0
    rpc_calls_succeeded: int = 0
    rpc_rate_limited_count: int = 0
    rpc_forbidden_count: int = 0
    rpc_retry_count: int = 0
    rpc_timeout_count: int = 0
    rpc_null_result_count: int = 0
    cache_hits: int = 0
    cache_misses: int = 0


@dataclass
class SolanaRpcClient:
    rpc_url: str | None = None
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_retries: int = DEFAULT_MAX_RETRIES
    get_tx_pace_seconds: float = DEFAULT_GET_TX_PACE_SECONDS
    cache_enabled: bool = True
    stats: SolanaRpcStats = field(default_factory=SolanaRpcStats)
    _last_get_tx_at: float = field(default=0.0, repr=False)

    def get_rpc_url(self) -> str:
        if self.rpc_url:
            return self.rpc_url
        return os.getenv("SOLANA_RPC_URL", DEFAULT_RPC_URL).strip() or DEFAULT_RPC_URL

    def is_public_rpc(self) -> bool:
        url = self.get_rpc_url().rstrip("/")
        return url == PUBLIC_RPC_URL.rstrip("/")

    def _cache_key(self, method: str, params: list | dict | None) -> str:
        payload = json.dumps(
            {"endpoint": self.get_rpc_url(), "method": method, "params": params},
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        return CACHE_DIR / f"{cache_key}.json"

    def _read_cache(self, cache_key: str) -> dict[str, Any] | None:
        if not self.cache_enabled:
            return None
        path = self._cache_path(cache_key)
        try:
            if not path.is_file():
                return None
            raw = json.loads(path.read_text(encoding="utf-8"))
            cached_at = float(raw.get("cached_at", 0))
            if time.time() - cached_at > CACHE_TTL_SECONDS:
                return None
            return raw.get("response")
        except Exception as exc:
            log.debug("Solana RPC cache read failed: %s", exc)
            return None

    def _write_cache(self, cache_key: str, response: dict[str, Any]) -> None:
        if not self.cache_enabled:
            return
        try:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            payload = {
                "cached_at": time.time(),
                "response": response,
            }
            self._cache_path(cache_key).write_text(
                json.dumps(payload, separators=(",", ":")),
                encoding="utf-8",
            )
        except Exception as exc:
            log.debug("Solana RPC cache write failed: %s", exc)

    def _sleep_with_jitter(self, base_seconds: float) -> None:
        jitter = random.uniform(0.0, min(base_seconds * 0.25, 1.0))
        time.sleep(base_seconds + jitter)

    def _parse_retry_after(self, response: httpx.Response) -> float | None:
        header = response.headers.get("Retry-After")
        if not header:
            return None
        try:
            return max(float(header), 0.0)
        except ValueError:
            return None

    def rpc_call(
        self,
        method: str,
        params: list | dict | None = None,
        *,
        use_cache: bool = False,
    ) -> RpcCallResult:
        if use_cache and method == "getTransaction":
            cache_key = self._cache_key(method, params)
            cached = self._read_cache(cache_key)
            if cached is not None:
                self.stats.cache_hits += 1
                return RpcCallResult(status=SOLANA_RPC_OK, result=cached.get("result"))
            self.stats.cache_misses += 1

        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params if params is not None else [],
        }
        url = self.get_rpc_url()
        retries = 0
        last_status = SOLANA_RPC_UNAVAILABLE

        while retries < self.max_retries:
            self.stats.rpc_calls_attempted += 1
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    response = client.post(
                        url,
                        json=body,
                        headers={"Content-Type": "application/json"},
                    )
            except httpx.TimeoutException:
                self.stats.rpc_timeout_count += 1
                last_status = SOLANA_RPC_TIMEOUT
                retries += 1
                self.stats.rpc_retry_count += 1
                if retries >= self.max_retries:
                    break
                self._sleep_with_jitter(min(2 ** retries, 16))
                continue
            except httpx.HTTPError:
                last_status = SOLANA_RPC_UNAVAILABLE
                retries += 1
                self.stats.rpc_retry_count += 1
                if retries >= self.max_retries:
                    break
                self._sleep_with_jitter(min(2 ** retries, 16))
                continue

            if response.status_code == 429:
                self.stats.rpc_rate_limited_count += 1
                last_status = SOLANA_RPC_RATE_LIMITED
                retries += 1
                self.stats.rpc_retry_count += 1
                if retries >= self.max_retries:
                    break
                delay = self._parse_retry_after(response) or min(2 ** retries, 16)
                self._sleep_with_jitter(delay)
                continue

            if response.status_code == 403:
                self.stats.rpc_forbidden_count += 1
                return RpcCallResult(
                    status=SOLANA_RPC_FORBIDDEN,
                    http_status=403,
                    retries=retries,
                )

            if response.status_code >= 500:
                last_status = SOLANA_RPC_UNAVAILABLE
                retries += 1
                self.stats.rpc_retry_count += 1
                if retries >= self.max_retries:
                    break
                self._sleep_with_jitter(min(2 ** retries, 16))
                continue

            if response.status_code >= 400:
                return RpcCallResult(
                    status=SOLANA_RPC_UNAVAILABLE,
                    http_status=response.status_code,
                    retries=retries,
                )

            try:
                payload = response.json()
            except json.JSONDecodeError:
                return RpcCallResult(
                    status=SOLANA_RPC_JSON_ERROR,
                    http_status=response.status_code,
                    retries=retries,
                )

            if not isinstance(payload, dict):
                return RpcCallResult(
                    status=SOLANA_RPC_JSON_ERROR,
                    http_status=response.status_code,
                    retries=retries,
                )

            if "error" in payload:
                return RpcCallResult(
                    status=SOLANA_RPC_JSONRPC_ERROR,
                    error=payload.get("error"),
                    http_status=response.status_code,
                    retries=retries,
                )

            if payload.get("result") is None:
                self.stats.rpc_null_result_count += 1
                return RpcCallResult(
                    status=SOLANA_RPC_NULL_RESULT,
                    result=None,
                    http_status=response.status_code,
                    retries=retries,
                )

            self.stats.rpc_calls_succeeded += 1
            if use_cache and method == "getTransaction":
                self._write_cache(
                    self._cache_key(method, params),
                    {"result": payload.get("result")},
                )
            return RpcCallResult(
                status=SOLANA_RPC_OK,
                result=payload.get("result"),
                http_status=response.status_code,
                retries=retries,
            )

        return RpcCallResult(status=last_status, retries=retries)

    def _pace_get_transaction(self) -> None:
        if self.get_tx_pace_seconds <= 0:
            return
        if not self.is_public_rpc():
            return
        elapsed = time.time() - self._last_get_tx_at
        if elapsed < self.get_tx_pace_seconds:
            time.sleep(self.get_tx_pace_seconds - elapsed)
        self._last_get_tx_at = time.time()

    def get_account_info(self, address: str) -> dict[str, Any]:
        result = self.rpc_call(
            "getAccountInfo",
            [address, {"encoding": "jsonParsed"}],
        )
        return {
            "status": result.status,
            "result": result.result,
            "error": result.error,
        }

    def get_signatures_for_address(self, address: str, limit: int = 25) -> list[dict[str, Any]]:
        result = self.rpc_call(
            "getSignaturesForAddress",
            [address, {"limit": limit}],
        )
        if result.status != SOLANA_RPC_OK or not isinstance(result.result, list):
            return []
        return result.result

    def get_transaction(self, signature: str) -> dict[str, Any]:
        self._pace_get_transaction()
        params = [
            signature,
            {
                "encoding": "jsonParsed",
                "maxSupportedTransactionVersion": 0,
            },
        ]
        result = self.rpc_call("getTransaction", params, use_cache=True)
        return {
            "status": result.status,
            "result": result.result,
            "error": result.error,
            "signature": signature,
        }


_default_client: SolanaRpcClient | None = None


def get_default_client() -> SolanaRpcClient:
    global _default_client
    if _default_client is None:
        _default_client = SolanaRpcClient()
    return _default_client


def reset_default_client() -> None:
    global _default_client
    _default_client = None


def get_rpc_url() -> str:
    return get_default_client().get_rpc_url()


def rpc_call(method: str, params: list | dict | None = None) -> dict[str, Any]:
    result = get_default_client().rpc_call(method, params)
    return {
        "status": result.status,
        "result": result.result,
        "error": result.error,
    }


def get_account_info(address: str) -> dict[str, Any]:
    return get_default_client().get_account_info(address)


def get_signatures_for_address(address: str, limit: int = 25) -> list[dict[str, Any]]:
    return get_default_client().get_signatures_for_address(address, limit=limit)


def get_transaction(signature: str) -> dict[str, Any]:
    return get_default_client().get_transaction(signature)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
