"""Local identity store — user-entered token identity, keyed by chain+address.

Paper/demo research aid only. This store NEVER fabricates price, liquidity,
or any market data — it only persists identity fields the user explicitly
entered (symbol/name/chain/contract) so they can be looked up instantly by
the contract resolver without waiting on a live market match.

system_verified is always False: nothing here has been confirmed by an
external/authoritative source, it is simply what the user told the system.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "runtime"
STORE_PATH = DATA_DIR / "watchlist_identity_store.json"
_LOCK = threading.RLock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_chain(chain: str | None) -> str:
    try:
        from app.ae13b_product.contract_resolver import normalize_chain

        return normalize_chain(chain)
    except Exception:
        return str(chain or "").strip().lower()


def _normalize_address(addr: str | None, *, chain: str | None = None) -> str:
    try:
        from app.ae13b_product.contract_resolver import normalize_address

        return normalize_address(addr, chain=chain)
    except Exception:
        raw = str(addr or "").strip()
        return raw.lower() if raw.startswith("0x") else raw


def identity_key(chain: str | None, address: str | None) -> str:
    """Stable lookup key: normalized chain + normalized address."""
    ch = _normalize_chain(chain)
    addr = _normalize_address(address, chain=ch or chain)
    return f"{ch}:{addr}"


def _load() -> dict[str, dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not STORE_PATH.exists():
        return {}
    try:
        with open(STORE_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _save(store: dict[str, dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2)


def upsert_identity(
    *,
    chain: str | None,
    address: str | None,
    symbol: str | None = None,
    name: str | None = None,
    display_label: str | None = None,
    pair_address: str | None = None,
    source: str = "watchlist_user_input",
    watchlist_item_id: str | None = None,
) -> dict[str, Any]:
    """Insert or update a user-entered identity record for chain+address.

    Never writes price/liquidity/volume — identity fields only.
    """
    if not address and not pair_address:
        raise ValueError("address or pair_address required")

    ch = _normalize_chain(chain)
    addr = _normalize_address(address or pair_address, chain=ch or chain)
    key = f"{ch}:{addr}"

    with _LOCK:
        store = _load()
        existing = store.get(key) or {}
        record = {
            "id": existing.get("id") or str(uuid.uuid4())[:10],
            "identity_key": key,
            "chain": ch or existing.get("chain") or "",
            "address": addr or existing.get("address") or "",
            "pair_address": (pair_address or existing.get("pair_address") or None),
            "symbol": (symbol.strip() if symbol else None) or existing.get("symbol"),
            "name": (name.strip() if name else None) or existing.get("name"),
            "display_label": (display_label.strip() if display_label else None)
            or existing.get("display_label"),
            "source": source or existing.get("source") or "watchlist_user_input",
            "watchlist_item_id": watchlist_item_id or existing.get("watchlist_item_id"),
            "paper_demo_only": True,
            "system_verified": False,
            "created_at": existing.get("created_at") or _utc_now(),
            "updated_at": _utc_now(),
        }
        store[key] = record
        _save(store)
        return dict(record)


def get_identity(chain: str | None, address: str | None) -> dict[str, Any] | None:
    """Exact chain+address lookup. Returns None if not found."""
    if not address:
        return None
    key = identity_key(chain, address)
    with _LOCK:
        store = _load()
        rec = store.get(key)
        return dict(rec) if rec else None


def list_identities() -> list[dict[str, Any]]:
    with _LOCK:
        store = _load()
        return [dict(v) for v in store.values()]


def lookup_for_resolver(
    chain: str | None,
    address: str | None,
    symbol: str | None = None,
) -> dict[str, Any] | None:
    """Best-effort lookup for the contract resolver: exact chain+address first,
    then fall back to a chain-scoped symbol match. Never fabricates market data.
    """
    rec = get_identity(chain, address)
    if rec:
        return rec

    if not symbol:
        return None

    sym_norm = str(symbol).strip().upper()
    if "/" in sym_norm:
        sym_norm = sym_norm.split("/")[0].strip()
    if not sym_norm:
        return None

    ch = _normalize_chain(chain)
    with _LOCK:
        store = _load()
        for rec in store.values():
            if ch and rec.get("chain") and rec.get("chain") != ch:
                continue
            rec_sym = str(rec.get("symbol") or "").strip().upper()
            if rec_sym == sym_norm:
                return dict(rec)
    return None


def remove_identity(chain: str | None, address: str | None) -> bool:
    key = identity_key(chain, address)
    with _LOCK:
        store = _load()
        if key not in store:
            return False
        del store[key]
        _save(store)
        return True
