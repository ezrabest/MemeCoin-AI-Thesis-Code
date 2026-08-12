"""Stable serialization helpers for AE15 Clean Forward records."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any


def _normalize_for_json(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: _normalize_for_json(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): _normalize_for_json(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_normalize_for_json(v) for v in value]
    if isinstance(value, float):
        # Keep finite floats; stringify non-finite for stability.
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        return value
    return value


def stable_json_dumps(obj: Any) -> str:
    """Deterministic JSON string (sorted keys, no whitespace variance)."""
    return json.dumps(_normalize_for_json(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def stable_payload_hash(obj: Any) -> str:
    return hashlib.sha256(stable_json_dumps(obj).encode("utf-8")).hexdigest()


def record_to_dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "to_dict") and callable(record.to_dict):
        return dict(record.to_dict())
    if is_dataclass(record) and not isinstance(record, type):
        return asdict(record)
    if isinstance(record, dict):
        return dict(record)
    raise TypeError(f"Unsupported record type: {type(record)!r}")


def candidate_id_material_dict(
    *,
    chain: str,
    provider: str,
    pair_address_for_id: str,
    base_token_address: str,
    quote_token_address: str,
    observed_at_or_fetched_at: str,
    provider_payload_hash: str,
) -> dict[str, str]:
    """Only the fields allowed in clean_forward_candidate_id material."""
    return {
        "chain": chain,
        "provider": provider,
        "pair_address_for_id": pair_address_for_id,
        "base_token_address": base_token_address,
        "quote_token_address": quote_token_address,
        "observed_at_or_fetched_at": observed_at_or_fetched_at,
        "provider_payload_hash": provider_payload_hash,
    }
