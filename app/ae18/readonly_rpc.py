"""AE18 read-only Solana/Helius JSON-RPC client — allowlist enforced."""

from __future__ import annotations

import hashlib
import json
import random
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

import httpx

ALLOWED_RPC_METHODS: frozenset[str] = frozenset(
    {
        "getAccountInfo",
        "getSignaturesForAddress",
        "getTransaction",
        "getTokenLargestAccounts",
        "getTokenAccountsByOwner",
        "getBalance",
        "getBlockTime",
        "getTokenSupply",
    }
)

FORBIDDEN_RPC_METHODS: frozenset[str] = frozenset(
    {
        "sendTransaction",
        "sendRawTransaction",
        "simulateTransaction",
        "requestAirdrop",
        "getFeeForMessage",
    }
)


class AE18ReadOnlyViolation(Exception):
    """Raised when a forbidden or non-allowlisted RPC method is requested."""

    def __init__(self, method: str, reason: str = "forbidden_or_not_allowlisted"):
        self.method = method
        self.reason = reason
        super().__init__(f"AE18ReadOnlyViolation: method={method} reason={reason}")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _params_hash(params: Any) -> str:
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _response_hash(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _is_forbidden_method(method: str) -> bool:
    m = (method or "").strip()
    if not m:
        return True
    if m in FORBIDDEN_RPC_METHODS:
        return True
    if m.startswith("send"):
        return True
    # Block signing methods, but not getSignatures* / getSignatureStatuses*
    lower = m.lower()
    if "sign" in lower and not lower.startswith("getsignature"):
        return True
    if m not in ALLOWED_RPC_METHODS:
        return True
    return False


@dataclass
class AE18RpcStats:
    rpc_calls_attempted: int = 0
    rpc_calls_successful: int = 0
    rpc_calls_failed: int = 0
    rpc_calls_skipped_by_cache: int = 0
    retry_after_used_count: int = 0
    backoff_retry_count: int = 0
    rate_limit_count: int = 0
    max_retries_reached_count: int = 0
    total_delay_ms: float = 0.0
    delay_samples: int = 0
    forbidden_method_attempts: list[str] = field(default_factory=list)

    @property
    def average_delay_ms(self) -> float:
        if self.delay_samples <= 0:
            return 0.0
        return round(self.total_delay_ms / self.delay_samples, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rpc_calls_attempted": self.rpc_calls_attempted,
            "rpc_calls_successful": self.rpc_calls_successful,
            "rpc_calls_failed": self.rpc_calls_failed,
            "rpc_calls_skipped_by_cache": self.rpc_calls_skipped_by_cache,
            "retry_after_used_count": self.retry_after_used_count,
            "backoff_retry_count": self.backoff_retry_count,
            "rate_limit_count": self.rate_limit_count,
            "average_delay_ms": self.average_delay_ms,
            "max_retries_reached_count": self.max_retries_reached_count,
            "forbidden_method_attempts": list(self.forbidden_method_attempts),
        }


@dataclass
class AE18ReadOnlyRpcClient:
    """Allowlisted JSON-RPC client with in-run cache, throttle, and 429 backoff."""

    rpc_url: str
    provider_used: str = "HELIUS_RPC"
    min_delay_ms: int = 250
    max_calls: int = 250
    max_retries: int = 3
    timeout_seconds: float = 20.0
    sleep_fn: Callable[[float], None] = field(default=time.sleep, repr=False)
    transport: httpx.BaseTransport | None = None
    stats: AE18RpcStats = field(default_factory=AE18RpcStats)
    _cache: dict[str, dict[str, Any]] = field(default_factory=dict, repr=False)
    _last_call_at: float = field(default=0.0, repr=False)
    raw_call_log: list[dict[str, Any]] = field(default_factory=list)
    budget_exhausted: bool = False

    def call(
        self,
        method: str,
        params: list[Any] | None = None,
        *,
        clean_forward_candidate_id: str = "",
        price_source_key: str = "",
        chain: str = "solana",
        pair_address: str = "",
    ) -> dict[str, Any]:
        if _is_forbidden_method(method):
            self.stats.forbidden_method_attempts.append(method)
            raise AE18ReadOnlyViolation(method)

        params = params if params is not None else []
        cache_key = f"{method}|{_params_hash(params)}"
        if cache_key in self._cache:
            self.stats.rpc_calls_skipped_by_cache += 1
            cached = self._cache[cache_key]
            entry = {
                "rpc_call_id": str(uuid.uuid4()),
                "clean_forward_candidate_id": clean_forward_candidate_id,
                "price_source_key": price_source_key,
                "chain": chain,
                "pair_address": pair_address,
                "method": method,
                "params_hash": _params_hash(params),
                "provider_used": self.provider_used,
                "attempted_at": utc_now(),
                "success": cached.get("success", False),
                "http_status": cached.get("http_status"),
                "rpc_error_code": cached.get("rpc_error_code"),
                "rate_limit_status": "NOT_RATE_LIMITED",
                "retry_count": 0,
                "response_hash": cached.get("response_hash"),
                "compact_response": cached.get("compact_response"),
                "provenance_status": "CACHE_HIT",
                "read_only_enforced": True,
                "cache_hit": True,
            }
            self.raw_call_log.append(entry)
            return {
                "success": cached.get("success", False),
                "result": cached.get("result"),
                "error": cached.get("error"),
                "http_status": cached.get("http_status"),
                "rpc_error_code": cached.get("rpc_error_code"),
                "rate_limit_status": "NOT_RATE_LIMITED",
                "retry_count": 0,
                "response_hash": cached.get("response_hash"),
                "cache_hit": True,
                "rpc_call_id": entry["rpc_call_id"],
            }

        if self.stats.rpc_calls_attempted >= self.max_calls:
            self.budget_exhausted = True
            entry = self._log_failure(
                method=method,
                params=params,
                clean_forward_candidate_id=clean_forward_candidate_id,
                price_source_key=price_source_key,
                chain=chain,
                pair_address=pair_address,
                http_status=None,
                rpc_error_code="RPC_BUDGET_EXCEEDED",
                rate_limit_status="NOT_RATE_LIMITED",
                retry_count=0,
                reason="RPC_BUDGET_EXCEEDED",
            )
            return {
                "success": False,
                "result": None,
                "error": {"code": "RPC_BUDGET_EXCEEDED"},
                "http_status": None,
                "rpc_error_code": "RPC_BUDGET_EXCEEDED",
                "rate_limit_status": "NOT_RATE_LIMITED",
                "retry_count": 0,
                "response_hash": None,
                "cache_hit": False,
                "rpc_call_id": entry["rpc_call_id"],
            }

        self._throttle()
        self.stats.rpc_calls_attempted += 1

        retry_count = 0
        last_http: int | None = None
        last_error: Any = None
        rate_limit_status = "NOT_RATE_LIMITED"

        while retry_count <= self.max_retries:
            try:
                with httpx.Client(timeout=self.timeout_seconds, transport=self.transport) as client:
                    resp = client.post(
                        self.rpc_url,
                        json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params},
                        headers={"Content-Type": "application/json"},
                    )
                last_http = resp.status_code
                if resp.status_code == 429:
                    self.stats.rate_limit_count += 1
                    rate_limit_status = "RPC_RATE_LIMITED"
                    retry_after = resp.headers.get("Retry-After")
                    delay = self._compute_backoff(retry_count, retry_after)
                    if retry_after:
                        self.stats.retry_after_used_count += 1
                    else:
                        self.stats.backoff_retry_count += 1
                    if retry_count >= self.max_retries:
                        self.stats.max_retries_reached_count += 1
                        self.stats.rpc_calls_failed += 1
                        entry = self._log_failure(
                            method=method,
                            params=params,
                            clean_forward_candidate_id=clean_forward_candidate_id,
                            price_source_key=price_source_key,
                            chain=chain,
                            pair_address=pair_address,
                            http_status=429,
                            rpc_error_code="RPC_RATE_LIMITED",
                            rate_limit_status=rate_limit_status,
                            retry_count=retry_count,
                            reason="RPC_RATE_LIMITED",
                        )
                        return {
                            "success": False,
                            "result": None,
                            "error": {"code": "RPC_RATE_LIMITED"},
                            "http_status": 429,
                            "rpc_error_code": "RPC_RATE_LIMITED",
                            "rate_limit_status": rate_limit_status,
                            "retry_count": retry_count,
                            "response_hash": None,
                            "cache_hit": False,
                            "rpc_call_id": entry["rpc_call_id"],
                        }
                    self.sleep_fn(delay)
                    retry_count += 1
                    continue

                if resp.status_code >= 400:
                    self.stats.rpc_calls_failed += 1
                    entry = self._log_failure(
                        method=method,
                        params=params,
                        clean_forward_candidate_id=clean_forward_candidate_id,
                        price_source_key=price_source_key,
                        chain=chain,
                        pair_address=pair_address,
                        http_status=resp.status_code,
                        rpc_error_code=f"HTTP_{resp.status_code}",
                        rate_limit_status=rate_limit_status,
                        retry_count=retry_count,
                        reason="HTTP_ERROR",
                    )
                    return {
                        "success": False,
                        "result": None,
                        "error": {"code": f"HTTP_{resp.status_code}"},
                        "http_status": resp.status_code,
                        "rpc_error_code": f"HTTP_{resp.status_code}",
                        "rate_limit_status": rate_limit_status,
                        "retry_count": retry_count,
                        "response_hash": None,
                        "cache_hit": False,
                        "rpc_call_id": entry["rpc_call_id"],
                    }

                payload = resp.json()
                if isinstance(payload, dict) and payload.get("error"):
                    err = payload["error"]
                    code = err.get("code") if isinstance(err, dict) else "JSONRPC_ERROR"
                    self.stats.rpc_calls_failed += 1
                    entry = self._log_failure(
                        method=method,
                        params=params,
                        clean_forward_candidate_id=clean_forward_candidate_id,
                        price_source_key=price_source_key,
                        chain=chain,
                        pair_address=pair_address,
                        http_status=resp.status_code,
                        rpc_error_code=str(code),
                        rate_limit_status=rate_limit_status,
                        retry_count=retry_count,
                        reason="JSONRPC_ERROR",
                        compact=err,
                    )
                    return {
                        "success": False,
                        "result": None,
                        "error": err,
                        "http_status": resp.status_code,
                        "rpc_error_code": str(code),
                        "rate_limit_status": rate_limit_status,
                        "retry_count": retry_count,
                        "response_hash": _response_hash(payload),
                        "cache_hit": False,
                        "rpc_call_id": entry["rpc_call_id"],
                    }

                result = payload.get("result") if isinstance(payload, dict) else payload
                compact = _compact_result(method, result)
                rh = _response_hash(compact)
                self.stats.rpc_calls_successful += 1
                cached_obj = {
                    "success": True,
                    "result": result,
                    "error": None,
                    "http_status": resp.status_code,
                    "rpc_error_code": None,
                    "response_hash": rh,
                    "compact_response": compact,
                }
                self._cache[cache_key] = cached_obj
                entry = {
                    "rpc_call_id": str(uuid.uuid4()),
                    "clean_forward_candidate_id": clean_forward_candidate_id,
                    "price_source_key": price_source_key,
                    "chain": chain,
                    "pair_address": pair_address,
                    "method": method,
                    "params_hash": _params_hash(params),
                    "provider_used": self.provider_used,
                    "attempted_at": utc_now(),
                    "success": True,
                    "http_status": resp.status_code,
                    "rpc_error_code": None,
                    "rate_limit_status": rate_limit_status,
                    "retry_count": retry_count,
                    "response_hash": rh,
                    "compact_response": compact,
                    "provenance_status": "RPC_FETCH_SUCCEEDED",
                    "read_only_enforced": True,
                    "cache_hit": False,
                }
                self.raw_call_log.append(entry)
                return {
                    "success": True,
                    "result": result,
                    "error": None,
                    "http_status": resp.status_code,
                    "rpc_error_code": None,
                    "rate_limit_status": rate_limit_status,
                    "retry_count": retry_count,
                    "response_hash": rh,
                    "cache_hit": False,
                    "rpc_call_id": entry["rpc_call_id"],
                }
            except AE18ReadOnlyViolation:
                raise
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if retry_count >= self.max_retries:
                    self.stats.max_retries_reached_count += 1
                    self.stats.rpc_calls_failed += 1
                    entry = self._log_failure(
                        method=method,
                        params=params,
                        clean_forward_candidate_id=clean_forward_candidate_id,
                        price_source_key=price_source_key,
                        chain=chain,
                        pair_address=pair_address,
                        http_status=last_http,
                        rpc_error_code="RPC_EXCEPTION",
                        rate_limit_status=rate_limit_status,
                        retry_count=retry_count,
                        reason=str(last_error),
                    )
                    return {
                        "success": False,
                        "result": None,
                        "error": {"code": "RPC_EXCEPTION", "message": str(last_error)},
                        "http_status": last_http,
                        "rpc_error_code": "RPC_EXCEPTION",
                        "rate_limit_status": rate_limit_status,
                        "retry_count": retry_count,
                        "response_hash": None,
                        "cache_hit": False,
                        "rpc_call_id": entry["rpc_call_id"],
                    }
                self.stats.backoff_retry_count += 1
                self.sleep_fn(self._compute_backoff(retry_count, None))
                retry_count += 1

        self.stats.rpc_calls_failed += 1
        entry = self._log_failure(
            method=method,
            params=params,
            clean_forward_candidate_id=clean_forward_candidate_id,
            price_source_key=price_source_key,
            chain=chain,
            pair_address=pair_address,
            http_status=last_http,
            rpc_error_code="RPC_FAILED",
            rate_limit_status=rate_limit_status,
            retry_count=retry_count,
            reason="RPC_FAILED",
        )
        return {
            "success": False,
            "result": None,
            "error": {"code": "RPC_FAILED"},
            "http_status": last_http,
            "rpc_error_code": "RPC_FAILED",
            "rate_limit_status": rate_limit_status,
            "retry_count": retry_count,
            "response_hash": None,
            "cache_hit": False,
            "rpc_call_id": entry["rpc_call_id"],
        }

    def _throttle(self) -> None:
        if self.min_delay_ms <= 0:
            return
        now = time.monotonic()
        elapsed_ms = (now - self._last_call_at) * 1000.0 if self._last_call_at else self.min_delay_ms
        wait_ms = max(0.0, float(self.min_delay_ms) - elapsed_ms)
        if wait_ms > 0:
            self.sleep_fn(wait_ms / 1000.0)
            self.stats.total_delay_ms += wait_ms
            self.stats.delay_samples += 1
        else:
            self.stats.total_delay_ms += float(self.min_delay_ms)
            self.stats.delay_samples += 1
        self._last_call_at = time.monotonic()

    def _compute_backoff(self, retry_count: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
        base = 0.5 * (2**retry_count)
        jitter = random.uniform(0, 0.25)
        return base + jitter

    def _log_failure(
        self,
        *,
        method: str,
        params: list[Any],
        clean_forward_candidate_id: str,
        price_source_key: str,
        chain: str,
        pair_address: str,
        http_status: int | None,
        rpc_error_code: str | None,
        rate_limit_status: str,
        retry_count: int,
        reason: str,
        compact: Any = None,
    ) -> dict[str, Any]:
        entry = {
            "rpc_call_id": str(uuid.uuid4()),
            "clean_forward_candidate_id": clean_forward_candidate_id,
            "price_source_key": price_source_key,
            "chain": chain,
            "pair_address": pair_address,
            "method": method,
            "params_hash": _params_hash(params),
            "provider_used": self.provider_used,
            "attempted_at": utc_now(),
            "success": False,
            "http_status": http_status,
            "rpc_error_code": rpc_error_code,
            "rate_limit_status": rate_limit_status,
            "retry_count": retry_count,
            "response_hash": _response_hash(compact) if compact is not None else None,
            "compact_response": compact,
            "provenance_status": reason,
            "read_only_enforced": True,
            "cache_hit": False,
        }
        self.raw_call_log.append(entry)
        return entry


def _compact_result(method: str, result: Any) -> Any:
    """Reduce payload size for archival while keeping provenance-useful fields."""
    if result is None:
        return None
    if method == "getAccountInfo" and isinstance(result, dict):
        value = result.get("value")
        if value is None:
            return {"value": None}
        if isinstance(value, dict):
            data = value.get("data")
            data_present = data is not None and data != "" and data != []
            return {
                "value": {
                    "lamports": value.get("lamports"),
                    "owner": value.get("owner"),
                    "executable": value.get("executable"),
                    "rentEpoch": value.get("rentEpoch"),
                    "data_present": bool(data_present),
                    "data_len": (
                        len(data[0]) if isinstance(data, list) and data else (len(data) if isinstance(data, str) else None)
                    ),
                }
            }
        return {"value": "present"}
    if method == "getSignaturesForAddress" and isinstance(result, list):
        return [
            {
                "signature": (item or {}).get("signature") if isinstance(item, dict) else None,
                "blockTime": (item or {}).get("blockTime") if isinstance(item, dict) else None,
                "err": (item or {}).get("err") if isinstance(item, dict) else None,
            }
            for item in result[:50]
        ]
    if method == "getTransaction" and isinstance(result, dict):
        meta = result.get("meta") or {}
        message = ((result.get("transaction") or {}).get("message") or {})
        return {
            "blockTime": result.get("blockTime"),
            "slot": result.get("slot"),
            "fee": meta.get("fee"),
            "err": meta.get("err"),
            "preTokenBalances_count": len(meta.get("preTokenBalances") or []),
            "postTokenBalances_count": len(meta.get("postTokenBalances") or []),
            "accountKeys_count": len(message.get("accountKeys") or []),
            "signatures_count": len((result.get("transaction") or {}).get("signatures") or []),
        }
    if isinstance(result, (dict, list)):
        blob = json.dumps(result, default=str)
        if len(blob) > 4000:
            return {"truncated": True, "sha256": hashlib.sha256(blob.encode()).hexdigest(), "len": len(blob)}
    return result
