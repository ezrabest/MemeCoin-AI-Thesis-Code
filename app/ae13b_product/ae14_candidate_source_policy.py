"""AE14 candidate-source policy — Clean Forward Market Feed only.

Central flag/helper used by AE14 closure mode, demo_bot, demo_queue, and
paper fill-price attribution. Does not enable live execution.
"""
from __future__ import annotations

import contextvars
import os
import re
from typing import Any

AE14_CANDIDATE_SOURCE_POLICY = "clean_forward_market_feed_only"
CANDIDATE_SOURCE = "clean_forward_market_feed"
PAPER_PRICE_SOURCE = "clean_forward_market_feed"

_SYNTHETIC_MARKERS = (
    "ae14",
    "paperlifecycle",
    "fixture",
    "synthetic",
    "mock",
    "test",
    "dummy",
    "sample",
)

_EVM_CHAINS = frozenset(
    {
        "base",
        "ethereum",
        "eth",
        "bsc",
        "arbitrum",
        "optimism",
        "polygon",
    }
)

_EVM_PAIR_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
_BASE58_ALPHABET = frozenset(
    "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
)

_ae14_closure_mode: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "ae14_closure_mode", default=False
)


def ae14_candidate_source_policy() -> str:
    return AE14_CANDIDATE_SOURCE_POLICY


def is_ae14_closure_mode() -> bool:
    if _ae14_closure_mode.get():
        return True
    flag = str(os.environ.get("AE14_CLOSURE_MODE") or "").strip().lower()
    return flag in {"1", "true", "yes", "on", "closure"}


def set_ae14_closure_mode(enabled: bool) -> None:
    _ae14_closure_mode.set(bool(enabled))
    if enabled:
        os.environ["AE14_CLOSURE_MODE"] = "1"
    else:
        os.environ.pop("AE14_CLOSURE_MODE", None)


def enable_ae14_closure_mode() -> None:
    set_ae14_closure_mode(True)


def disable_ae14_closure_mode() -> None:
    set_ae14_closure_mode(False)


def requires_clean_forward_only() -> bool:
    """True when AE14 closure mode forbids legacy candidate fallbacks."""
    return is_ae14_closure_mode()


def candidate_source_for_ae14() -> str:
    return CANDIDATE_SOURCE


def paper_price_source_for_ae14() -> str:
    return PAPER_PRICE_SOURCE


def _norm_text(value: Any) -> str:
    return str(value or "").strip()


def _haystack_for_synthetic_check(row: dict[str, Any]) -> str:
    parts = [
        row.get("pair_address"),
        row.get("provider_pair_id"),
        row.get("row_id"),
        row.get("row_key"),
        row.get("pair"),
        row.get("pair_label"),
        row.get("symbol"),
        row.get("base_token_symbol"),
        row.get("quote_token_symbol"),
    ]
    return " ".join(_norm_text(p) for p in parts).lower()


def is_synthetic_or_fixture_row(row: dict[str, Any] | None) -> bool:
    """Reject synthetic/test/mock fixture identifiers used by prior smokes."""
    if not isinstance(row, dict):
        return True
    hay = _haystack_for_synthetic_check(row)
    if not hay.strip():
        return True
    for marker in _SYNTHETIC_MARKERS:
        if marker in hay:
            return True
    return False


def _looks_like_base58(value: str) -> bool:
    text = value.strip()
    if len(text) < 32 or len(text) > 64:
        return False
    if text.startswith("0x") or text.startswith("0X"):
        return False
    return all(ch in _BASE58_ALPHABET for ch in text)


def pair_address_plausible(*, chain: Any, pair_address: Any, provider_pair_id: Any = None) -> bool:
    """Validate pair/pool address shape for EVM-like and Solana-like chains."""
    chain_norm = _norm_text(chain).lower()
    addr = _norm_text(pair_address) or _norm_text(provider_pair_id)
    if not addr:
        return False
    if chain_norm in _EVM_CHAINS:
        return bool(_EVM_PAIR_RE.match(addr))
    if chain_norm in {"solana", "sol", "svm"}:
        return _looks_like_base58(addr)
    # Unknown chain: accept EVM hex or plausible base58, reject 0x fakes shorter/longer.
    if addr.startswith("0x") or addr.startswith("0X"):
        return bool(_EVM_PAIR_RE.match(addr))
    return _looks_like_base58(addr)


def _safe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _safe_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None


def is_valid_ae14_clean_forward_row(row: dict[str, Any] | None) -> bool:
    """Hard gates for AE14 real Clean Forward closure candidates."""
    if not isinstance(row, dict):
        return False
    if is_synthetic_or_fixture_row(row):
        return False
    if str(row.get("verification_status") or "") != "provider_pair_verified":
        return False
    if str(row.get("freshness_status") or "") != "fresh":
        return False
    if str(row.get("identity_status") or "") != "pair_and_tokens_separated":
        return False
    if _safe_bool(row.get("shown_as_token_contract")) is not False:
        return False
    if _safe_bool(row.get("paper_demo_only")) is not True:
        return False
    if _safe_bool(row.get("live_trading_ready")) is not False:
        return False
    pair = _norm_text(row.get("pair") or row.get("pair_label"))
    if not pair or "/" not in pair:
        return False
    if not _norm_text(row.get("base_token_symbol")):
        return False
    chain = _norm_text(
        row.get("chain") or row.get("normalized_chain_id") or row.get("chain_id")
    )
    if not chain:
        return False
    pair_address = _norm_text(row.get("pair_address"))
    provider_pair_id = _norm_text(row.get("provider_pair_id"))
    if not pair_address and not provider_pair_id:
        return False
    if not pair_address_plausible(
        chain=chain, pair_address=pair_address, provider_pair_id=provider_pair_id
    ):
        return False
    price = _safe_float(row.get("price_usd") if row.get("price_usd") is not None else row.get("price"))
    liquidity = _safe_float(
        row.get("liquidity_usd") if row.get("liquidity_usd") is not None else row.get("liquidity")
    )
    if price is None or price <= 0:
        return False
    if liquidity is None or liquidity <= 0:
        return False
    timestamp = (
        row.get("observed_at")
        or row.get("fetched_at")
        or row.get("last_fetched")
        or row.get("ingested_at")
    )
    if not timestamp:
        return False
    return True


def select_ae14_clean_forward_candidates(
    rows: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if not isinstance(rows, list):
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        if is_valid_ae14_clean_forward_row(row):
            out.append(dict(row))
    return out


def ae14_policy_audit_flags() -> dict[str, Any]:
    return {
        "ae14_candidate_source_policy": AE14_CANDIDATE_SOURCE_POLICY,
        "candidate_source": CANDIDATE_SOURCE,
        "paper_execution_price_source": PAPER_PRICE_SOURCE,
        "legacy_market_snapshots_used": False,
        "old_watchlist_candidates_used": False,
        "local_db_candidate_universe_used": False,
        "synthetic_fixture_used": False,
        "all_bots_use_clean_forward_feed": True,
        "clean_forward_market_feed_used": True,
    }
