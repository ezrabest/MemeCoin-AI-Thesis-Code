"""Persistent reentry cooldown blocks (survives process restart)."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_LOCK = threading.RLock()

#: AE13I scope taxonomy — order reflects match priority (most specific first).
SCOPE_EXACT_PAIR = "exact_pair"
SCOPE_ASSET_CONTRACT = "asset_contract"
SCOPE_TOKEN_MINT = "token_mint"
SCOPE_SYMBOL_CHAIN = "symbol_chain"


def _determine_scope(
    *,
    pair_address: str | None,
    token_contract: str | None,
    token_mint: str | None,
    symbol: str | None,
) -> str:
    if pair_address:
        return SCOPE_EXACT_PAIR
    if token_contract:
        return SCOPE_ASSET_CONTRACT
    if token_mint:
        return SCOPE_TOKEN_MINT
    if symbol:
        return SCOPE_SYMBOL_CHAIN
    return "unknown"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def blocks_file_path() -> Path:
    return _project_root() / "data" / "runtime" / "reentry_blocks.json"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(value: Any) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _identity_key(
    *,
    pair_address: str | None,
    chain: str | None,
    token_contract: str | None,
    token_mint: str | None,
    symbol: str | None,
) -> str:
    parts = [
        _norm(chain),
        _norm(pair_address),
        _norm(token_contract),
        _norm(token_mint),
        _norm(symbol).split("/")[0],
    ]
    return "|".join(parts)


def load_blocks() -> dict[str, Any]:
    path = blocks_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        return {"version": 1, "blocks": []}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"version": 1, "blocks": []}
    if not isinstance(raw, dict):
        return {"version": 1, "blocks": []}
    blocks = raw.get("blocks")
    if not isinstance(blocks, list):
        raw["blocks"] = []
    raw.setdefault("version", 1)
    return raw


def save_blocks(store: dict[str, Any]) -> None:
    path = blocks_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(store or {})
    payload.setdefault("version", 1)
    payload["saved_at_utc"] = _utc_now().isoformat()
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def clear_expired_blocks(store: dict[str, Any] | None = None) -> dict[str, Any]:
    with _LOCK:
        data = dict(store if store is not None else load_blocks())
        now = _utc_now()
        kept: list[dict[str, Any]] = []
        for block in data.get("blocks") or []:
            if not isinstance(block, dict):
                continue
            expires = _parse_ts(block.get("expires_at_utc"))
            if expires is not None and expires <= now:
                continue
            kept.append(block)
        data["blocks"] = kept
        save_blocks(data)
        return data


def _add_block(
    *,
    position: dict[str, Any],
    close_reason: str,
    duration_seconds: int,
    block_kind: str,
) -> dict[str, Any]:
    with _LOCK:
        data = clear_expired_blocks()
        now = _utc_now()
        pos = dict(position or {})
        pair_address = pos.get("pair_address") or pos.get("matched_pair_address")
        chain = pos.get("chain") or pos.get("network")
        token_contract = pos.get("token_contract_address") or pos.get("token_contract")
        token_mint = pos.get("token_mint_address") or pos.get("token_mint")
        symbol = pos.get("symbol") or pos.get("market_symbol")
        key = _identity_key(
            pair_address=pair_address,
            chain=chain,
            token_contract=token_contract,
            token_mint=token_mint,
            symbol=symbol,
        )
        expires = now.timestamp() + max(1, int(duration_seconds))
        expires_at = datetime.fromtimestamp(expires, tz=timezone.utc).isoformat()
        is_manual = block_kind == "manual_close"
        scope = _determine_scope(
            pair_address=pair_address,
            token_contract=token_contract,
            token_mint=token_mint,
            symbol=symbol,
        )
        block = {
            "block_id": str(uuid.uuid4()),
            "identity_key": key,
            "block_kind": block_kind,
            "scope": scope,
            "source": "user_manual_close" if is_manual else "system_close",
            "position_id": pos.get("id") or pos.get("position_id"),
            "block_reason": "manual_user_exit" if is_manual else "system_exit",
            "close_reason": str(close_reason or "unknown"),
            "created_at_utc": now.isoformat(),
            "created_at": now.isoformat(),
            "expires_at_utc": expires_at,
            "expires_at": expires_at,
            "duration_seconds": int(duration_seconds),
            "pair_address": pair_address,
            "chain": chain,
            "token_contract": token_contract,
            "token_mint": token_mint,
            "symbol": symbol,
            "paper_demo_only": True,
            "not_live_approved": True,
            "close_snapshot": {
                "price": pos.get("exit_price") or pos.get("marked_price") or pos.get("latest_price"),
                "volume_24h": pos.get("volume_24h") or pos.get("latest_volume_24h"),
                "liquidity": pos.get("liquidity") or pos.get("latest_liquidity"),
                "closed_at_utc": pos.get("closed_at") or pos.get("closed_at_utc") or now.isoformat(),
            },
        }
        blocks = [b for b in (data.get("blocks") or []) if b.get("identity_key") != key]
        blocks.append(block)
        data["blocks"] = blocks
        save_blocks(data)
        return block


def add_manual_close_block(
    position: dict[str, Any],
    close_reason: str,
    duration_seconds: int = 3600,
) -> dict[str, Any]:
    return _add_block(
        position=position,
        close_reason=close_reason,
        duration_seconds=duration_seconds,
        block_kind="manual_close",
    )


def add_system_close_block(
    position: dict[str, Any],
    close_reason: str,
    duration_seconds: int = 300,
) -> dict[str, Any]:
    return _add_block(
        position=position,
        close_reason=close_reason,
        duration_seconds=duration_seconds,
        block_kind="system_close",
    )


def check_reentry_block(
    pair_address: str | None = None,
    chain: str | None = None,
    token_contract: str | None = None,
    token_mint: str | None = None,
    symbol: str | None = None,
) -> dict[str, Any] | None:
    """Look up an active reentry block.

    Primary match is the full identity key (chain + pair/contract/mint/symbol).
    AE13I: when a pair_address is present, also fall back to matching by
    pair_address alone (exact_pair scope) even if other identity fields
    (e.g. chain casing, symbol) differ slightly between candidate rows.
    """
    data = clear_expired_blocks()
    key = _identity_key(
        pair_address=pair_address,
        chain=chain,
        token_contract=token_contract,
        token_mint=token_mint,
        symbol=symbol,
    )
    now = _utc_now()
    norm_pair = _norm(pair_address)
    blocks = list(reversed(data.get("blocks") or []))

    def _live(block: dict[str, Any]) -> dict[str, Any] | None:
        expires = _parse_ts(block.get("expires_at_utc"))
        if expires is not None and expires <= now:
            return None
        remaining = (expires - now).total_seconds() if expires else None
        out = dict(block)
        out["seconds_remaining"] = max(0.0, remaining) if remaining is not None else None
        out["active"] = True
        return out

    for block in blocks:
        if block.get("identity_key") != key:
            continue
        live = _live(block)
        if live is not None:
            return live

    if norm_pair:
        for block in blocks:
            if _norm(block.get("pair_address")) != norm_pair:
                continue
            live = _live(block)
            if live is not None:
                live["matched_by"] = "pair_address_only"
                return live
    return None


def get_manual_cooldown_fields(
    pair_address: str | None = None,
    chain: str | None = None,
    token_contract: str | None = None,
    token_mint: str | None = None,
    symbol: str | None = None,
) -> dict[str, Any]:
    block = check_reentry_block(pair_address, chain, token_contract, token_mint, symbol)
    if not block:
        return {
            "reentry_blocked": False,
            "manual_cooldown_active": False,
            "cooldown_seconds_remaining": 0,
            "cooldown_expires_at_utc": None,
            "cooldown_reason": None,
            "block_kind": None,
            "manual_cooldown_expiry": None,
            "manual_cooldown_remaining_seconds": 0,
            "manual_cooldown_reason": None,
            "manual_cooldown_scope": None,
        }
    is_manual = block.get("block_kind") == "manual_close"
    remaining_seconds = int(block.get("seconds_remaining") or 0)
    scope = block.get("scope") or _determine_scope(
        pair_address=block.get("pair_address"),
        token_contract=block.get("token_contract"),
        token_mint=block.get("token_mint"),
        symbol=block.get("symbol"),
    )
    return {
        "reentry_blocked": True,
        "manual_cooldown_active": is_manual,
        "cooldown_seconds_remaining": remaining_seconds,
        "cooldown_expires_at_utc": block.get("expires_at_utc"),
        "cooldown_reason": block.get("close_reason"),
        "block_kind": block.get("block_kind"),
        "close_snapshot": block.get("close_snapshot"),
        # AE13I additions
        "manual_cooldown_expiry": block.get("expires_at_utc"),
        "manual_cooldown_remaining_seconds": remaining_seconds if is_manual else 0,
        "manual_cooldown_reason": block.get("close_reason") if is_manual else None,
        "manual_cooldown_scope": scope,
        "block_id": block.get("block_id"),
        "matched_by": block.get("matched_by") or "identity_key",
        "paper_demo_only": True,
        "not_live_approved": True,
    }