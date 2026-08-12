"""Validation helpers for unified candidate schema (Phase E2)."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

_ARTIFACT_ID_RE = re.compile(r"^[0-9a-f]{64}$")
_EVM_CHAINS = frozenset({"ethereum", "eth", "evm", "bsc", "polygon", "arbitrum", "base"})
_SOLANA_CHAINS = frozenset({"solana", "sol"})


def normalize_pair_address(pair_address: str, chain: str | None = None) -> str:
    """Normalize pair address for deterministic hashing."""
    normalized = pair_address.strip()
    chain_key = (chain or "").strip().lower()
    if chain_key in _EVM_CHAINS:
        return normalized.lower()
    if chain_key in _SOLANA_CHAINS:
        return normalized
    if normalized.startswith("0x"):
        return normalized.lower()
    return normalized


def normalize_event_timestamp(
    value: datetime | str | int | float,
    *,
    precision: str = "seconds",
) -> str:
    """Normalize event timestamps to ISO-8601 UTC strings.

    Naive datetimes are assumed UTC. Equivalent inputs must produce the same output.
    """
    if precision not in {"seconds", "milliseconds"}:
        raise ValueError(f"Unsupported timestamp precision: {precision}")

    dt = _parse_event_timestamp(value)
    dt = dt.astimezone(timezone.utc)
    if precision == "seconds":
        dt = dt.replace(microsecond=0)
    else:
        ms = dt.microsecond // 1000
        dt = dt.replace(microsecond=ms * 1000)
    iso = dt.isoformat()
    if iso.endswith("+00:00"):
        return iso[:-6] + "Z"
    return iso


def _parse_event_timestamp(value: datetime | str | int | float) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value

    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError("event_timestamp must be finite")
        numeric = float(value)
        if abs(numeric) >= 1e12:
            numeric /= 1000.0
        return datetime.fromtimestamp(numeric, tz=timezone.utc)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("event_timestamp string must not be empty")
        if text.isdigit() or (
            text.replace(".", "", 1).isdigit() and text.count(".") <= 1
        ):
            return _parse_event_timestamp(float(text))
        normalized = text.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed

    raise TypeError(f"Unsupported event_timestamp type: {type(value)!r}")


def is_finite_number(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def validate_finite_numeric(
    value: Any,
    *,
    allow_nan_for_research: bool = False,
) -> float | None:
    """Validate numeric input; reject non-finite unless research mode is enabled."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text == "":
            return None
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError(f"Invalid numeric string: {value!r}") from exc
    if isinstance(value, bool):
        raise ValueError("Boolean is not a valid numeric value")
    if not isinstance(value, (int, float)):
        raise ValueError(f"Expected numeric value, got {type(value)!r}")
    numeric = float(value)
    if not math.isfinite(numeric):
        if allow_nan_for_research:
            return numeric
        raise ValueError("Non-finite numeric values are not allowed")
    return numeric


def compute_vote_count(
    in_xgb: bool | None,
    in_tab: bool | None,
    in_rf: bool | None,
) -> int:
    return sum(1 for flag in (in_xgb, in_tab, in_rf) if flag is True)


def is_syntactically_valid_artifact_id(artifact_id: str | None) -> bool:
    """Check whether an artifact id matches the E1 SHA-256 hex format."""
    if artifact_id is None:
        return False
    text = artifact_id.strip()
    return bool(_ARTIFACT_ID_RE.fullmatch(text))
