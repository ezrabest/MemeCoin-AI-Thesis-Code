"""
Manual watchlist — user-supplied symbols/contracts for persistent tracking.

Paper/demo research aid only — does not imply live trading.

Display coalescing: user-entered identity is always primary.
Market match is enrichment only and must never replace user identity fields.

Statuses are independent:
  identity_resolution_status / market_match_status / semantic_status / demo_queue_status
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent.parent.parent / "data"
WATCHLIST_PATH = DATA_DIR / "watchlist.json"
_LOCK = threading.RLock()

EXPECTED_CATEGORIES = frozenset(
    {
        "user thinks social",
        "user thinks opportunistic",
        "user wants investigation",
        "social",
        "opportunistic",
        "investigation",
        "",
    }
)

MARKET_MATCH_EXPLAIN = (
    "Tracked from user input. Not found in current local market feed. "
    "External lookup not enabled. Semantic check can still run from available evidence."
)

COLLECTION_ACTIVE_LOCAL = "active_local_only"
COLLECTION_ACTIVE_EXTERNAL = "active_external_enabled"
COLLECTION_WAITING = "waiting_for_market_match"
COLLECTION_NO_PROVIDER = "no_supported_provider"
COLLECTION_PROVIDER_UNAVAIL = "provider_unavailable"
COLLECTION_UNSUPPORTED = "unsupported_chain"
COLLECTION_ERROR = "error"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _norm_addr(value: str | None) -> str:
    raw = str(value or "").strip()
    if raw.startswith("0x"):
        return raw.lower()
    return raw


def _normalize_symbol(symbol: str | None) -> str:
    raw = str(symbol or "").strip().upper()
    if "/" in raw:
        raw = raw.split("/")[0].strip()
    return raw


def _short_contract(addr: str | None) -> str:
    raw = str(addr or "").strip()
    if not raw:
        return ""
    if len(raw) <= 14:
        return raw
    return f"{raw[:10]}…{raw[-6:]}"


def _coalesce(*vals: Any, fallback: str = "Unknown") -> str:
    for v in vals:
        if v is None:
            continue
        s = str(v).strip()
        if s and s != "—" and s.lower() != "none":
            return s
    return fallback


def display_coalesce(entry: dict[str, Any]) -> dict[str, Any]:
    """Display identity priority — user input first, market as enrichment only."""
    user_symbol = entry.get("user_entered_symbol") or entry.get("user_symbol") or entry.get("symbol")
    user_name = entry.get("user_entered_name") or entry.get("user_name")
    user_pair = entry.get("user_entered_pair") or entry.get("user_pair") or entry.get("pair")
    user_contract = (
        entry.get("user_entered_contract_or_pair_address")
        or entry.get("user_contract_address")
        or entry.get("contract_address")
    )
    user_chain = entry.get("user_entered_chain") or entry.get("chain")
    user_label = entry.get("user_display_label") or entry.get("user_nickname")
    resolved_symbol = entry.get("resolved_symbol")
    resolved_name = entry.get("resolved_name")
    resolved_chain = entry.get("resolved_chain")
    resolved_addr = entry.get("resolved_contract_or_pair_address")
    market_symbol = entry.get("market_symbol")
    market_name = entry.get("market_name")
    market_chain = entry.get("market_chain")
    matched_addr = entry.get("matched_pair_address") or entry.get("matched_market_address")

    display_name = _coalesce(
        user_name,
        user_label,
        resolved_name,
        market_name,
        user_symbol,
        resolved_symbol,
        market_symbol,
        f"Contract {_short_contract(user_contract)}" if user_contract else None,
        fallback="Unknown",
    )
    # Prefer short contract for symbol when user only entered an address (pair==contract)
    symbol_fallback = None
    if user_contract:
        if user_pair and _norm_addr(user_pair) == _norm_addr(user_contract):
            symbol_fallback = _short_contract(user_contract)
        else:
            symbol_fallback = user_pair or _short_contract(user_contract)

    display_symbol = _coalesce(
        user_symbol,
        resolved_symbol,
        market_symbol,
        symbol_fallback,
        _short_contract(user_contract) if user_contract else None,
        fallback="—",
    )
    # Never show "—" for identity the user provided
    if (user_symbol or user_name or user_contract or user_pair) and display_symbol == "—":
        display_symbol = _coalesce(
            user_symbol, user_pair, _short_contract(user_contract), fallback="Unknown"
        )
    if (user_name or user_symbol or user_contract) and display_name == "Unknown" and user_contract:
        display_name = f"Contract {_short_contract(user_contract)}"

    display_chain = _coalesce(user_chain, resolved_chain, market_chain, fallback="Unknown")
    display_id = _coalesce(
        user_contract,
        user_pair,
        resolved_addr,
        matched_addr,
        fallback="Unknown",
    )

    market_match = entry.get("market_match_status") or entry.get("data_collection_status") or ""
    market_explain = None
    if str(market_match) in ("waiting_for_market_match", ""):
        market_explain = MARKET_MATCH_EXPLAIN

    return {
        "user_entered_symbol": user_symbol,
        "user_entered_name": user_name,
        "user_entered_pair": user_pair,
        "user_entered_contract_or_pair_address": user_contract,
        "user_entered_chain": user_chain,
        "user_display_label": user_label,
        "user_symbol": user_symbol,
        "user_pair": user_pair,
        "user_contract_address": user_contract,
        "display_name": display_name,
        "display_symbol": display_symbol,
        "display_chain": display_chain,
        "display_id": display_id,
        "display_pair": user_pair or matched_addr or user_contract,
        "market_symbol": market_symbol,
        "market_name": market_name,
        "market_enrichment": bool(market_symbol or market_name or matched_addr),
        "market_match_explanation": market_explain,
        "full_id_copyable": display_id if display_id != "Unknown" else "",
    }


# Backward-compatible alias
def _display_fields(entry: dict[str, Any]) -> dict[str, Any]:
    return display_coalesce(entry)


def _default_statuses(entry: dict[str, Any]) -> dict[str, Any]:
    user_has_id = bool(
        entry.get("user_entered_symbol")
        or entry.get("user_symbol")
        or entry.get("user_entered_contract_or_pair_address")
        or entry.get("user_contract_address")
        or entry.get("user_entered_pair")
        or entry.get("user_pair")
        or entry.get("user_entered_name")
    )
    identity = entry.get("identity_resolution_status")
    if not identity:
        if entry.get("matched_pair_address") or entry.get("last_seen_in_market"):
            identity = "local_match"
        elif user_has_id:
            identity = "user_entered_identity"
        else:
            identity = "unresolved_local_only"
    # Normalize legacy
    if identity == "matched_live_market":
        identity = "local_match"
    elif identity == "user_entered_only":
        identity = "user_entered_identity"
    elif identity == "unresolved":
        identity = "user_entered_identity" if user_has_id else "unresolved_local_only"

    market = entry.get("market_match_status") or entry.get("data_collection_status")
    if not market:
        market = (
            "seen_in_live_market"
            if entry.get("last_seen_in_market")
            else "waiting_for_market_match"
        )
    # Normalize legacy
    if market == "matched_in_live_market":
        market = "seen_in_live_market"

    semantic = entry.get("semantic_status")
    if not semantic:
        fam = entry.get("semantic_classification") or entry.get("semantic_signal_family")
        if fam in ("UNKNOWN_INSUFFICIENT_EVIDENCE", None, ""):
            if entry.get("user_evidence_url") or entry.get("user_evidence_note"):
                semantic = "evidence_provided_pending_check"
            elif fam is None:
                semantic = "not_checked"
            else:
                semantic = "unknown_insufficient_evidence"
        elif fam in ("NEEDS_REVIEW", "SOCIAL_CANDIDATE_NEEDS_VERIFICATION"):
            semantic = "needs_review"
        else:
            semantic = "classified"

    demo_q = entry.get("demo_queue_status") or "not_in_queue"

    tracking_enabled = entry.get("tracking_enabled")
    if tracking_enabled is None:
        tracking_enabled = bool(entry.get("enabled", True)) and not bool(entry.get("disabled"))
    collection = entry.get("collection_status")
    if not collection:
        if not tracking_enabled:
            collection = COLLECTION_WAITING
        elif entry.get("last_seen_in_market") or entry.get("latest_price") is not None:
            collection = COLLECTION_ACTIVE_LOCAL
        elif entry.get("external_lookup_enabled"):
            collection = COLLECTION_ACTIVE_EXTERNAL
        else:
            collection = COLLECTION_WAITING

    return {
        "identity_resolution_status": identity,
        "market_match_status": market,
        "semantic_status": semantic,
        "demo_queue_status": demo_q,
        "data_collection_status": market,  # keep legacy field aligned to market_match
        "tracking_enabled": bool(tracking_enabled),
        "collection_status": collection,
    }


def _migrate_item(item: dict[str, Any]) -> dict[str, Any]:
    """Ensure legacy rows have ids and immutable user-identity fields."""
    if not item.get("id"):
        item["id"] = str(uuid.uuid4())[:10]
    item["watchlist_id"] = item.get("watchlist_id") or item.get("id")

    # Freeze user identity on first load if missing
    if "user_symbol" not in item:
        item["user_symbol"] = item.get("symbol")
    if "user_pair" not in item:
        item["user_pair"] = item.get("pair")
    if "user_contract_address" not in item:
        item["user_contract_address"] = item.get("contract_address")

    item.setdefault("user_entered_symbol", item.get("user_symbol"))
    item.setdefault("user_entered_name", item.get("user_name") or item.get("name"))
    item.setdefault("user_entered_pair", item.get("user_pair"))
    item.setdefault(
        "user_entered_contract_or_pair_address",
        item.get("user_contract_address") or item.get("user_pair"),
    )
    item.setdefault("user_entered_chain", item.get("chain"))
    item.setdefault("user_display_label", item.get("user_nickname") or item.get("user_display_label"))
    item.setdefault("user_expected_category", item.get("expected_category") or "")
    item.setdefault("user_note", item.get("user_note") or "")
    item.setdefault("user_evidence_url", item.get("user_evidence_url") or "")
    item.setdefault("user_evidence_note", item.get("user_evidence_note") or "")
    item.setdefault("user_claimed_social_mission", item.get("user_claimed_social_mission") or "")

    # Keep legacy fields aligned to user identity
    if item.get("user_entered_symbol") is not None:
        item["user_symbol"] = item["user_entered_symbol"]
        item["symbol"] = item["user_entered_symbol"]
    if item.get("user_entered_pair") is not None:
        item["user_pair"] = item["user_entered_pair"]
        item["pair"] = item["user_entered_pair"]
    if item.get("user_entered_contract_or_pair_address") is not None:
        item["user_contract_address"] = item["user_entered_contract_or_pair_address"]
        item["contract_address"] = item["user_entered_contract_or_pair_address"]
    if item.get("user_entered_chain"):
        item["chain"] = item["user_entered_chain"]

    item.setdefault("enabled", not bool(item.get("disabled")))
    item.setdefault("disabled", False)
    item.setdefault("removed", False)
    item.setdefault("pinned", False)
    item.setdefault("active_demo_candidate", False)
    item.setdefault("status", "registered")
    item.setdefault("created_at", item.get("first_added_at") or item.get("added_at") or _utc_now())
    item.setdefault("updated_at", item.get("last_checked_at") or item.get("created_at"))
    item.setdefault("paper_demo_only", True)
    item.setdefault("not_live_approved", True)
    item.setdefault("live_trading_implied", False)
    item.setdefault("tracking_enabled", bool(item.get("enabled", True)) and not bool(item.get("disabled")))
    item.setdefault("collection_status", COLLECTION_WAITING)
    item.setdefault("last_collection_attempt_at", None)
    item.setdefault("last_collection_success_at", None)
    item.setdefault("last_collection_error", None)
    item.setdefault("external_lookup_enabled", False)
    item.setdefault("latest_price", None)
    item.setdefault("latest_price_source", None)
    item.setdefault("latest_price_timestamp", None)
    item.setdefault("latest_liquidity", None)
    item.setdefault("latest_volume_24h", None)
    item.setdefault("latest_delta_5m", None)
    item.setdefault("latest_delta_1h", None)
    item.setdefault("latest_delta_6h", None)
    item.setdefault("latest_delta_24h", None)
    item.update(_default_statuses(item))
    item.update(display_coalesce(item))
    try:
        from app.ae13b_product.identity_model import attach_identity_objects

        attach_identity_objects(item)
    except Exception:
        pass
    return item


def _load(*, persist_migrations: bool = False) -> list[dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not WATCHLIST_PATH.exists():
        return []
    with open(WATCHLIST_PATH, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        return []
    migrated: list[dict[str, Any]] = []
    needs_persist = False
    for raw in data:
        if not isinstance(raw, dict):
            continue
        before_id = raw.get("id")
        before_user = "user_symbol" in raw or "user_entered_symbol" in raw
        item = _migrate_item(dict(raw))
        if not before_id or not before_user:
            needs_persist = True
        if item.get("removed"):
            continue
        migrated.append(item)
    if persist_migrations and needs_persist:
        _save(migrated)
    return migrated


def _save(items: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    payload = []
    for i in items:
        row = dict(i)
        row.update(_default_statuses(row))
        row.update(display_coalesce(row))
        try:
            from app.ae13b_product.identity_model import attach_identity_objects

            attach_identity_objects(row)
        except Exception:
            pass
        payload.append(row)
    with open(WATCHLIST_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def add_to_watchlist(contract_address: str, chain: str | None = None) -> dict[str, Any]:
    """Backward-compatible: add by contract address only."""
    return upsert_watchlist_item(
        contract_address=contract_address,
        chain=chain or "solana",
    )


def upsert_watchlist_item(
    *,
    symbol: str | None = None,
    pair: str | None = None,
    contract_address: str | None = None,
    chain: str | None = None,
    note: str | None = None,
    expected_category: str | None = None,
    name: str | None = None,
    display_label: str | None = None,
) -> dict[str, Any]:
    addr = (contract_address or "").strip()
    sym = (symbol or "").strip()
    pair_s = (pair or "").strip()
    if not addr and not sym and not pair_s:
        raise ValueError("symbol, pair, or contract_address required")

    cat = (expected_category or "").strip().lower()
    if cat and cat not in EXPECTED_CATEGORIES:
        cat = "user wants investigation"

    chain_l = (chain or "solana").lower()
    addr_l = _norm_addr(addr)

    with _LOCK:
        items = _load()
        existing = None
        if addr_l:
            existing = next(
                (
                    i
                    for i in items
                    if _norm_addr(
                        i.get("user_entered_contract_or_pair_address")
                        or i.get("user_contract_address")
                        or i.get("contract_address")
                    )
                    == addr_l
                ),
                None,
            )
        if existing is None and pair_s:
            pair_l = _norm_addr(pair_s)
            existing = next(
                (
                    i
                    for i in items
                    if _norm_addr(i.get("user_entered_pair") or i.get("user_pair") or i.get("pair"))
                    == pair_l
                    and str(i.get("chain") or "").lower() == chain_l
                ),
                None,
            )
        if existing is None and sym:
            existing = next(
                (
                    i
                    for i in items
                    if _normalize_symbol(
                        i.get("user_entered_symbol") or i.get("user_symbol") or i.get("symbol")
                    )
                    == _normalize_symbol(sym)
                    and str(i.get("chain") or "").lower() == chain_l
                    and not i.get("disabled")
                ),
                None,
            )
        if existing is not None:
            if sym:
                existing["user_entered_symbol"] = sym
                existing["user_symbol"] = sym
                existing["symbol"] = sym
            if name:
                existing["user_entered_name"] = name
            if display_label:
                existing["user_display_label"] = display_label
            if pair_s:
                existing["user_entered_pair"] = pair_s
                existing["user_pair"] = pair_s
                existing["pair"] = pair_s
            if addr:
                existing["user_entered_contract_or_pair_address"] = addr
                existing["user_contract_address"] = addr
                existing["contract_address"] = addr
            if note is not None:
                existing["user_note"] = note
            if cat:
                existing["expected_category"] = cat
                existing["user_expected_category"] = cat
            existing["user_entered_chain"] = chain_l
            existing["chain"] = chain_l
            existing["enabled"] = True
            existing["disabled"] = False
            existing["removed"] = False
            if existing.get("status") == "disabled":
                existing["status"] = "registered"
            existing["last_checked_at"] = _utc_now()
            existing["updated_at"] = _utc_now()
            existing.update(_default_statuses(existing))
            existing.update(display_coalesce(existing))
            _save(items)
            _feed_registry(existing)
            _upsert_identity_store(existing)
            return existing

        entry = {
            "id": str(uuid.uuid4())[:10],
            "watchlist_id": None,
            "user_entered_symbol": sym or None,
            "user_entered_name": name or None,
            "user_entered_pair": pair_s or None,
            "user_entered_contract_or_pair_address": addr or pair_s or None,
            "user_entered_chain": chain_l,
            "user_display_label": display_label or None,
            "user_symbol": sym or None,
            "user_pair": pair_s or None,
            "user_contract_address": addr or pair_s or None,
            "symbol": sym or None,
            "pair": pair_s or None,
            "contract_address": addr or pair_s or None,
            "chain": chain_l,
            "user_note": note or "",
            "expected_category": cat or "",
            "user_expected_category": cat or "",
            "user_evidence_url": "",
            "user_evidence_note": "",
            "user_claimed_social_mission": "",
            "status": "registered",
            "data_collection_status": "waiting_for_market_match",
            "identity_resolution_status": "user_entered_identity",
            "market_match_status": "waiting_for_market_match",
            "semantic_status": "not_checked",
            "demo_queue_status": "not_in_queue",
            "tracking_enabled": True,
            "collection_status": COLLECTION_WAITING,
            "last_collection_attempt_at": None,
            "last_collection_success_at": None,
            "last_collection_error": None,
            "external_lookup_enabled": False,
            "latest_price": None,
            "latest_price_source": None,
            "latest_price_timestamp": None,
            "latest_liquidity": None,
            "latest_volume_24h": None,
            "latest_delta_5m": None,
            "latest_delta_1h": None,
            "latest_delta_6h": None,
            "latest_delta_24h": None,
            "semantic_classification": None,
            "semantic_signal_family": None,
            "evidence_summary": None,
            "market_symbol": None,
            "market_name": None,
            "matched_pair_address": None,
            "first_added_at": _utc_now(),
            "added_at": _utc_now(),
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "last_checked_at": None,
            "last_seen_in_market": None,
            "enabled": True,
            "disabled": False,
            "removed": False,
            "pinned": False,
            "active_demo_candidate": False,
            "paper_demo_only": True,
            "not_live_approved": True,
            "live_trading_implied": False,
        }
        entry["watchlist_id"] = entry["id"]
        entry.update(display_coalesce(entry))
        items.append(entry)
        _save(items)
        _feed_registry(entry)
        _upsert_identity_store(entry)
        return entry


def remove_watchlist_item(item_id: str | None = None, *, contract_address: str | None = None) -> bool:
    with _LOCK:
        items = _load()
        before = len(items)
        if item_id:
            items = [
                i
                for i in items
                if str(i.get("id")) != str(item_id) and str(i.get("watchlist_id")) != str(item_id)
            ]
        elif contract_address:
            addr_l = _norm_addr(contract_address)
            items = [
                i
                for i in items
                if _norm_addr(
                    i.get("user_entered_contract_or_pair_address")
                    or i.get("user_contract_address")
                    or i.get("contract_address")
                )
                != addr_l
                and _norm_addr(i.get("user_entered_pair") or i.get("user_pair") or i.get("pair"))
                != addr_l
            ]
        else:
            return False
        if len(items) == before:
            return False
        _save(items)
        return True


def disable_watchlist_item(item_id: str) -> dict[str, Any] | None:
    with _LOCK:
        items = _load()
        for i in items:
            if str(i.get("id")) == str(item_id) or str(i.get("watchlist_id")) == str(item_id):
                i["enabled"] = False
                i["disabled"] = True
                i["status"] = "disabled"
                i["updated_at"] = _utc_now()
                i.update(display_coalesce(i))
                _save(items)
                return i
        return None


def enable_watchlist_item(item_id: str) -> dict[str, Any] | None:
    with _LOCK:
        items = _load()
        for i in items:
            if str(i.get("id")) == str(item_id) or str(i.get("watchlist_id")) == str(item_id):
                i["enabled"] = True
                i["disabled"] = False
                if i.get("status") == "disabled":
                    i["status"] = "registered"
                i["updated_at"] = _utc_now()
                i.update(display_coalesce(i))
                _save(items)
                return i
        return None


def pin_watchlist_item(item_id: str, pinned: bool = True) -> dict[str, Any] | None:
    with _LOCK:
        items = _load()
        for i in items:
            if str(i.get("id")) == str(item_id) or str(i.get("watchlist_id")) == str(item_id):
                i["pinned"] = bool(pinned)
                i["updated_at"] = _utc_now()
                i.update(display_coalesce(i))
                _save(items)
                return i
        return None


def mark_active_demo_candidate(item_id: str, *, active: bool = True, demo_queue_status: str = "queued_for_evaluation") -> dict[str, Any] | None:
    with _LOCK:
        items = _load()
        for i in items:
            if str(i.get("id")) == str(item_id) or str(i.get("watchlist_id")) == str(item_id):
                i["active_demo_candidate"] = bool(active)
                i["demo_queue_status"] = demo_queue_status if active else "not_in_queue"
                i["updated_at"] = _utc_now()
                i.update(_default_statuses(i))
                i.update(display_coalesce(i))
                _save(items)
                return i
        return None


def set_watchlist_evidence(
    item_id: str,
    *,
    user_evidence_url: str | None = None,
    user_evidence_note: str | None = None,
    user_claimed_social_mission: str | None = None,
    user_expected_category: str | None = None,
) -> dict[str, Any] | None:
    """Add/edit user evidence. Does NOT auto-classify as SOCIAL_CONFIRMED."""
    with _LOCK:
        items = _load()
        for i in items:
            if str(i.get("id")) == str(item_id) or str(i.get("watchlist_id")) == str(item_id):
                if user_evidence_url is not None:
                    i["user_evidence_url"] = user_evidence_url
                if user_evidence_note is not None:
                    i["user_evidence_note"] = user_evidence_note
                if user_claimed_social_mission is not None:
                    i["user_claimed_social_mission"] = user_claimed_social_mission
                if user_expected_category is not None:
                    i["user_expected_category"] = user_expected_category
                    i["expected_category"] = user_expected_category
                i["semantic_status"] = "evidence_provided_pending_check"
                # Never fabricate SOCIAL_CONFIRMED from user hypothesis alone
                note_l = str(user_evidence_note or i.get("user_evidence_note") or "").lower()
                mission_l = str(
                    user_claimed_social_mission or i.get("user_claimed_social_mission") or ""
                ).lower()
                cat_l = str(
                    user_expected_category or i.get("user_expected_category") or ""
                ).lower()
                socialish = (
                    "social" in cat_l
                    or "charit" in note_l
                    or "educat" in note_l
                    or "charit" in mission_l
                    or "educat" in mission_l
                    or "social" in mission_l
                )
                if i.get("semantic_classification") == "SOCIAL_CONFIRMED":
                    pass  # keep only if previously evidence-backed
                elif socialish:
                    i["semantic_classification"] = "SOCIAL_CANDIDATE_NEEDS_VERIFICATION"
                    i["semantic_signal_family"] = "SOCIAL_CANDIDATE_NEEDS_VERIFICATION"
                    i["semantic_label_human"] = "Social candidate - needs verification"
                    i["evidence_summary"] = (
                        "user_supplied_social_claim_requires_validation: User-provided "
                        "social/educational claim; requires source validation. "
                        "Displayed as user supplied evidence - not system truth."
                    )
                elif not i.get("semantic_classification"):
                    i["semantic_classification"] = "NEEDS_REVIEW"
                    i["semantic_signal_family"] = "NEEDS_REVIEW"
                    i["semantic_label_human"] = "Needs review"
                    i["evidence_summary"] = (
                        "User evidence provided — pending validation. "
                        "SOCIAL_CONFIRMED requires explicit supporting evidence."
                    )
                i["updated_at"] = _utc_now()
                i["last_checked_at"] = _utc_now()
                i.update(_default_statuses(i))
                i.update(display_coalesce(i))
                _save(items)
                return i
        return None


def _attach_manual_cooldown(row: dict[str, Any]) -> dict[str, Any]:
    """Best-effort AE13I manual-cooldown badge fields for watchlist views."""
    try:
        from app.ae13b_product.reentry_blocks import get_manual_cooldown_fields

        addr = (
            row.get("matched_pair_address")
            or row.get("user_entered_contract_or_pair_address")
            or row.get("user_contract_address")
            or row.get("contract_address")
        )
        cooldown = get_manual_cooldown_fields(
            pair_address=addr,
            chain=row.get("display_chain") or row.get("chain"),
            symbol=row.get("display_symbol") or row.get("user_entered_symbol") or row.get("symbol"),
        )
        row["manual_cooldown_active"] = cooldown.get("manual_cooldown_active")
        row["manual_cooldown_expiry"] = cooldown.get("manual_cooldown_expiry")
        row["manual_cooldown_remaining_seconds"] = cooldown.get("manual_cooldown_remaining_seconds")
        row["manual_cooldown_reason"] = cooldown.get("manual_cooldown_reason")
        row["manual_cooldown_scope"] = cooldown.get("manual_cooldown_scope")
        row["reentry_blocked"] = cooldown.get("reentry_blocked")
    except Exception:
        row.setdefault("manual_cooldown_active", False)
    return row


def list_watchlist(*, include_disabled: bool = True) -> list[dict[str, Any]]:
    with _LOCK:
        items = _load(persist_migrations=True)
        out = []
        for i in items:
            row = dict(i)
            row.update(_default_statuses(row))
            row.update(display_coalesce(row))
            _attach_manual_cooldown(row)
            if include_disabled or (row.get("enabled", True) and not row.get("disabled")):
                out.append(row)
        # Pinned first
        out.sort(key=lambda r: (0 if r.get("pinned") else 1, str(r.get("created_at") or "")))
        return out


def get_watchlist_item(item_id: str) -> dict[str, Any] | None:
    for i in list_watchlist(include_disabled=True):
        if str(i.get("id")) == str(item_id) or str(i.get("watchlist_id")) == str(item_id):
            return i
    return None


def mark_analyzed(contract_address: str, **extra: Any) -> None:
    with _LOCK:
        items = _load()
        addr_l = _norm_addr(contract_address)
        for i in items:
            if _norm_addr(
                i.get("user_entered_contract_or_pair_address")
                or i.get("user_contract_address")
                or i.get("contract_address")
            ) == addr_l:
                i["last_analyzed_at"] = _utc_now()
                i["last_checked_at"] = _utc_now()
                i["updated_at"] = _utc_now()
                for key, val in extra.items():
                    if val is not None and key not in (
                        "user_symbol",
                        "user_pair",
                        "user_contract_address",
                        "user_entered_symbol",
                        "user_entered_pair",
                        "user_entered_contract_or_pair_address",
                        "user_entered_name",
                        "symbol",
                        "pair",
                        "contract_address",
                    ):
                        i[key] = val
                i.update(_default_statuses(i))
                i.update(display_coalesce(i))
        _save(items)


def match_market_to_watchlist(coin: dict[str, Any]) -> dict[str, Any] | None:
    items = list_watchlist(include_disabled=False)
    pair = _norm_addr(coin.get("pair_address"))
    addr = _norm_addr(coin.get("contract_address") or coin.get("token_address"))
    sym = _normalize_symbol(coin.get("symbol"))
    chain = str(coin.get("chain") or "").lower()
    for i in items:
        i_chain = str(i.get("chain") or "").lower()
        i_pair = _norm_addr(
            i.get("user_entered_pair")
            or i.get("user_pair")
            or i.get("pair")
            or i.get("matched_pair_address")
        )
        i_addr = _norm_addr(
            i.get("user_entered_contract_or_pair_address")
            or i.get("user_contract_address")
            or i.get("contract_address")
        )
        if pair and i_pair == pair and (not chain or not i_chain or chain == i_chain):
            return i
        if addr and i_addr == addr and (not chain or not i_chain or chain == i_chain):
            return i
        if (
            sym
            and _normalize_symbol(
                i.get("user_entered_symbol") or i.get("user_symbol") or i.get("symbol")
            )
            == sym
            and (not chain or not i_chain or chain == i_chain)
        ):
            return i
    return None


def refresh_watchlist_against_market(coins: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Match market rows to watchlist; enrich only — never overwrite user identity."""
    with _LOCK:
        items = _load()
        changed = False
        for coin in coins:
            match = None
            pair = _norm_addr(coin.get("pair_address"))
            addr = _norm_addr(coin.get("contract_address") or coin.get("token_address"))
            sym = _normalize_symbol(coin.get("symbol"))
            chain = str(coin.get("chain") or "").lower()
            for i in items:
                if i.get("disabled"):
                    continue
                i_chain = str(i.get("chain") or "").lower()
                i_pair = _norm_addr(
                    i.get("user_entered_pair")
                    or i.get("user_pair")
                    or i.get("pair")
                    or i.get("matched_pair_address")
                )
                i_addr = _norm_addr(
                    i.get("user_entered_contract_or_pair_address")
                    or i.get("user_contract_address")
                    or i.get("contract_address")
                )
                if pair and i_pair == pair and (not chain or not i_chain or chain == i_chain):
                    match = i
                    break
                if addr and i_addr == addr and (not chain or not i_chain or chain == i_chain):
                    match = i
                    break
                if (
                    sym
                    and _normalize_symbol(
                        i.get("user_entered_symbol") or i.get("user_symbol") or i.get("symbol")
                    )
                    == sym
                    and (not chain or not i_chain or chain == i_chain)
                ):
                    match = i
                    break
            if not match:
                continue
            match["last_seen_in_market"] = _utc_now()
            match["last_checked_at"] = _utc_now()
            match["updated_at"] = _utc_now()
            match["data_collection_status"] = "seen_in_live_market"
            match["market_match_status"] = "seen_in_live_market"
            match["identity_resolution_status"] = "matched_live_market"
            match["status"] = "seen_in_market"
            if coin.get("symbol"):
                match["market_symbol"] = coin.get("symbol")
            if coin.get("name"):
                match["market_name"] = coin.get("name")
            if coin.get("chain"):
                match["market_chain"] = coin.get("chain")
            if coin.get("pair_address"):
                match["matched_pair_address"] = coin.get("pair_address")
            match.update(_default_statuses(match))
            match.update(display_coalesce(match))
            changed = True
            _feed_registry(match, coin=coin)
        if changed:
            _save(items)
        return [dict(i, **display_coalesce(i)) for i in items]


def resolve_watchlist_identity(
    item_id: str,
    *,
    allow_external: bool = False,
    user_confirmed_external: bool = False,
) -> dict[str, Any]:
    """Explicit Resolve Identity action — local sources only unless explicitly allowed."""
    item = get_watchlist_item(item_id)
    if not item:
        return {"ok": False, "error": "watchlist_item_not_found"}

    # Snapshot user identity BEFORE resolve — must remain intact
    user_before = {
        "user_entered_symbol": item.get("user_entered_symbol"),
        "user_entered_name": item.get("user_entered_name"),
        "user_entered_pair": item.get("user_entered_pair"),
        "user_entered_contract_or_pair_address": item.get(
            "user_entered_contract_or_pair_address"
        ),
        "user_entered_chain": item.get("user_entered_chain"),
        "user_display_label": item.get("user_display_label"),
        "user_expected_category": item.get("user_expected_category"),
        "user_note": item.get("user_note"),
        "user_evidence_url": item.get("user_evidence_url"),
        "user_evidence_note": item.get("user_evidence_note"),
        "user_claimed_social_mission": item.get("user_claimed_social_mission"),
    }

    from app.ae13b_product.contract_resolver import resolve_identity
    from app.ae13b_product.identity_model import apply_resolved_only, attach_identity_objects

    allow_ext = bool(allow_external or item.get("external_lookup_enabled"))
    if allow_ext and not user_confirmed_external and not item.get("external_lookup_enabled"):
        allow_ext = False

    resolution = resolve_identity(
        chain=item.get("user_entered_chain") or item.get("chain"),
        contract_or_pair_address=item.get("user_entered_contract_or_pair_address")
        or item.get("user_contract_address")
        or item.get("contract_address"),
        symbol=item.get("user_entered_symbol") or item.get("user_symbol") or item.get("symbol"),
        pair_address=item.get("user_entered_pair") or item.get("user_pair"),
        allow_external=allow_ext,
    )

    with _LOCK:
        items = _load()
        for i in items:
            if str(i.get("id")) != str(item_id) and str(i.get("watchlist_id")) != str(item_id):
                continue
            apply_resolved_only(i, resolution)
            status = resolution.get("resolution_status") or "unresolved_local_only"
            if status == "local_match":
                i["market_match_status"] = "seen_in_live_market"
                i["data_collection_status"] = "seen_in_live_market"
                i["last_seen_in_market"] = i.get("last_seen_in_market") or _utc_now()
                if i.get("tracking_enabled"):
                    i["collection_status"] = COLLECTION_ACTIVE_LOCAL
                    i["last_collection_success_at"] = _utc_now()
            elif status == "user_entered_identity":
                i["market_match_status"] = i.get("market_match_status") or "waiting_for_market_match"
                i["data_collection_status"] = i["market_match_status"]
                if i.get("tracking_enabled"):
                    i["collection_status"] = COLLECTION_WAITING
                    i["last_collection_error"] = (
                        "This token is registered and tracked locally, but the current data "
                        "provider does not fetch assets outside the market feed."
                    )
            elif status == "provider_unavailable":
                if i.get("tracking_enabled"):
                    i["collection_status"] = COLLECTION_PROVIDER_UNAVAIL
            elif status == "invalid_address":
                i["collection_status"] = COLLECTION_ERROR
            # Restore protected user fields if anything drifted
            for k, v in user_before.items():
                if v is not None:
                    i[k] = v
            i["last_checked_at"] = _utc_now()
            i["last_collection_attempt_at"] = _utc_now()
            i["updated_at"] = _utc_now()
            i.update(_default_statuses(i))
            i.update(display_coalesce(i))
            attach_identity_objects(i)
            _save(items)
            # Verify non-destructive
            destructive = False
            for k, v in user_before.items():
                if v is not None and i.get(k) != v:
                    destructive = True
                    break
            return {
                "ok": True,
                "item": dict(i),
                "resolution": resolution,
                "user_entered_identity": i.get("user_entered_identity"),
                "resolved_identity": i.get("resolved_identity"),
                "market_enrichment": i.get("market_enrichment"),
                "identity_preserved": not destructive,
                "paper_demo_only": True,
            }
    return {"ok": False, "error": "watchlist_item_not_found"}


def update_watchlist_identity(
    item_id: str,
    *,
    name: str | None = None,
    symbol: str | None = None,
    pair: str | None = None,
    chain: str | None = None,
    contract_or_pair_address: str | None = None,
    display_label: str | None = None,
) -> dict[str, Any] | None:
    """Explicit user edit of UserEnteredIdentity — the only path that mutates it."""
    with _LOCK:
        items = _load()
        for i in items:
            if str(i.get("id")) != str(item_id) and str(i.get("watchlist_id")) != str(item_id):
                continue
            if name is not None:
                i["user_entered_name"] = name.strip() or None
            if symbol is not None:
                i["user_entered_symbol"] = symbol.strip() or None
                i["user_symbol"] = i["user_entered_symbol"]
                i["symbol"] = i["user_entered_symbol"]
            if pair is not None:
                i["user_entered_pair"] = pair.strip() or None
                i["user_pair"] = i["user_entered_pair"]
                i["pair"] = i["user_entered_pair"]
            if contract_or_pair_address is not None:
                addr = contract_or_pair_address.strip() or None
                i["user_entered_contract_or_pair_address"] = addr
                i["user_contract_address"] = addr
                i["contract_address"] = addr
            if chain is not None:
                i["user_entered_chain"] = (chain or "").strip().lower() or i.get("user_entered_chain")
                i["chain"] = i["user_entered_chain"]
            if display_label is not None:
                i["user_display_label"] = display_label.strip() or None
            i["user_identity_updated_at"] = _utc_now()
            i["updated_at"] = _utc_now()
            i["identity_resolution_status"] = "user_entered_identity"
            i.update(_default_statuses(i))
            i.update(display_coalesce(i))
            try:
                from app.ae13b_product.identity_model import attach_identity_objects

                attach_identity_objects(i)
            except Exception:
                pass
            _save(items)
            _upsert_identity_store(i)
            return dict(i)
        return None


def set_tracking_enabled(item_id: str, enabled: bool = True) -> dict[str, Any] | None:
    """Track Continuously / Stop Tracking — does not open live trades."""
    with _LOCK:
        items = _load()
        for i in items:
            if str(i.get("id")) != str(item_id) and str(i.get("watchlist_id")) != str(item_id):
                continue
            i["tracking_enabled"] = bool(enabled)
            if enabled:
                i["enabled"] = True
                i["disabled"] = False
                if i.get("status") == "disabled":
                    i["status"] = "registered"
                if i.get("last_seen_in_market") or i.get("latest_price") is not None:
                    i["collection_status"] = COLLECTION_ACTIVE_LOCAL
                elif i.get("external_lookup_enabled"):
                    i["collection_status"] = COLLECTION_ACTIVE_EXTERNAL
                else:
                    i["collection_status"] = COLLECTION_WAITING
                    i["last_collection_error"] = (
                        "This token is registered and tracked locally, but the current data "
                        "provider does not fetch assets outside the market feed."
                    )
            else:
                # Stop tracking ≠ remove; keep identity
                i["collection_status"] = COLLECTION_WAITING
            i["updated_at"] = _utc_now()
            i.update(_default_statuses(i))
            i.update(display_coalesce(i))
            _save(items)
            return dict(i)
        return None


def run_watchlist_collection_attempt(item_id: str) -> dict[str, Any]:
    """Attempt local data collection for a tracked watchlist item."""
    item = get_watchlist_item(item_id)
    if not item:
        return {"ok": False, "error": "watchlist_item_not_found"}
    resolve_result = resolve_watchlist_identity(item_id)
    item = (resolve_result or {}).get("item") or get_watchlist_item(item_id) or item
    resolution = (resolve_result or {}).get("resolution") or {}
    collected = resolution.get("resolution_status") == "local_match" and resolution.get(
        "matched_price"
    ) is not None
    explanation = resolution.get("reason") or (
        "This token is registered and tracked locally, but the current data "
        "provider does not fetch assets outside the market feed."
    )
    return {
        "ok": True,
        "item": item,
        "collected": collected,
        "collection_status": item.get("collection_status"),
        "tracking_enabled": item.get("tracking_enabled"),
        "explanation": explanation,
        "resolution": resolution,
        "paper_demo_only": True,
        "live_trading_implied": False,
    }


def enable_external_lookup_for_watchlist_item(item_id: str) -> dict[str, Any]:
    from app.ae13b_product.external_resolver import enable_external_lookup_for_item

    with _LOCK:
        items = _load()
        found = None
        for i in items:
            if str(i.get("id")) == str(item_id) or str(i.get("watchlist_id")) == str(item_id):
                i["external_lookup_enabled"] = True
                if i.get("tracking_enabled"):
                    i["collection_status"] = COLLECTION_ACTIVE_EXTERNAL
                i["updated_at"] = _utc_now()
                found = i
                break
        if not found:
            return {"ok": False, "error": "watchlist_item_not_found"}
        _save(items)
    flag = enable_external_lookup_for_item(str(item_id))
    return {
        "ok": True,
        "item": dict(found),
        "external": flag,
        "paper_demo_only": True,
    }


def run_watchlist_semantic_check(item_id: str) -> dict[str, Any]:
    """Run local taxonomy / registry check — independent of market match."""
    item = get_watchlist_item(item_id)
    if not item:
        return {"ok": False, "error": "watchlist_item_not_found"}

    with _LOCK:
        items = _load()
        for i in items:
            if str(i.get("id")) != str(item_id) and str(i.get("watchlist_id")) != str(item_id):
                continue
            try:
                from app.ae13_semantic.runtime_registry import get_semantic_registry

                cand = {
                    "symbol": i.get("user_entered_symbol")
                    or i.get("user_symbol")
                    or i.get("resolved_symbol")
                    or i.get("market_symbol"),
                    "name": i.get("user_entered_name")
                    or i.get("resolved_name")
                    or i.get("market_name")
                    or i.get("user_entered_symbol"),
                    "chain": i.get("user_entered_chain") or i.get("chain"),
                    "pair_address": i.get("matched_pair_address")
                    or i.get("user_entered_pair")
                    or i.get("user_entered_contract_or_pair_address"),
                    "coin_id": f"watchlist:{i.get('id')}",
                    "user_expected_category": i.get("user_expected_category")
                    or i.get("expected_category"),
                    "user_evidence_url": i.get("user_evidence_url"),
                    "user_evidence_note": i.get("user_evidence_note"),
                    "user_claimed_social_mission": i.get("user_claimed_social_mission"),
                    "force_reclassify": True,
                    "pinned": True,
                }
                rec = get_semantic_registry().observe_candidate(cand)
                fam = rec.get("semantic_signal_family") or "UNKNOWN_INSUFFICIENT_EVIDENCE"
                hyp = str(i.get("user_expected_category") or i.get("expected_category") or "").lower()
                evidence_note = str(i.get("user_evidence_note") or "").lower()
                mission = str(i.get("user_claimed_social_mission") or "").lower()
                social_claim = (
                    "social" in hyp
                    or "charit" in evidence_note
                    or "educat" in evidence_note
                    or "public" in evidence_note
                    or "charit" in mission
                    or "educat" in mission
                    or "social" in mission
                    or "giggle" in str(i.get("user_entered_name") or "").lower()
                )

                # User hypothesis / claim alone must never become SOCIAL_CONFIRMED
                if fam == "SOCIAL_CONFIRMED":
                    validated = "social_confirmed" in str(rec.get("evidence_summary") or "").lower()
                    if not validated:
                        fam = "SOCIAL_CANDIDATE_NEEDS_VERIFICATION"
                        rec = dict(rec)
                        rec["semantic_signal_family"] = fam
                        rec["semantic_label_human"] = "Social candidate - needs verification"
                        rec["evidence_summary"] = (
                            "user_supplied_social_claim_requires_validation: User-provided "
                            "social/educational claim; requires source validation. "
                            "SOCIAL_CONFIRMED is not auto-assigned from user hypothesis."
                        )
                elif social_claim and fam in (
                    "UNKNOWN_INSUFFICIENT_EVIDENCE",
                    "UNKNOWN_UNRESOLVED",
                    "NEEDS_REVIEW",
                    None,
                    "",
                ):
                    fam = "SOCIAL_CANDIDATE_NEEDS_VERIFICATION"
                    rec = dict(rec)
                    rec["semantic_signal_family"] = fam
                    rec["semantic_label_human"] = "Social candidate - needs verification"
                    rec["evidence_summary"] = (
                        "user_supplied_social_claim_requires_validation: User-provided "
                        "social/educational claim; requires source validation."
                    )
                elif "social" in hyp and not (
                    i.get("user_evidence_url")
                    or i.get("user_evidence_note")
                    or i.get("user_claimed_social_mission")
                ):
                    fam = "NEEDS_REVIEW"
                    rec = dict(rec)
                    rec["semantic_signal_family"] = fam
                    rec["semantic_label_human"] = "Needs review"
                    rec["evidence_summary"] = (
                        "No social/public-good evidence found from current local data. "
                        "Add evidence or enable provider-backed check. "
                        "User hypothesis is not SOCIAL_CONFIRMED."
                    )

                i["semantic_classification"] = fam
                i["semantic_signal_family"] = fam
                i["semantic_label_human"] = rec.get("semantic_label_human")
                i["evidence_summary"] = rec.get("evidence_summary") or (
                    "No social/public-good evidence found from current local data. "
                    "Add evidence or enable provider-backed check."
                )
                i["trading_opportunity_state"] = rec.get("trading_opportunity_state")
                # Market match is NOT required — mark semantic independent
                i["semantic_independent_of_market_match"] = True
                if fam in ("UNKNOWN_INSUFFICIENT_EVIDENCE", "UNKNOWN_UNRESOLVED"):
                    i["semantic_status"] = "unknown_insufficient_evidence"
                elif fam in ("NEEDS_REVIEW", "SOCIAL_CANDIDATE_NEEDS_VERIFICATION"):
                    i["semantic_status"] = "needs_review"
                else:
                    i["semantic_status"] = "classified"
            except Exception as exc:
                i["semantic_status"] = "provider_unavailable"
                i["semantic_classification"] = (
                    i.get("semantic_classification") or "UNKNOWN_INSUFFICIENT_EVIDENCE"
                )
                i["evidence_summary"] = f"Semantic check unavailable: {exc}"[:240]
                i["semantic_independent_of_market_match"] = True

            i["last_checked_at"] = _utc_now()
            i["updated_at"] = _utc_now()
            i.update(_default_statuses(i))
            i.update(display_coalesce(i))
            try:
                from app.ae13b_product.identity_model import attach_identity_objects

                attach_identity_objects(i)
            except Exception:
                pass
            _save(items)
            return {
                "ok": True,
                "item": dict(i),
                "semantic_signal_family": i.get("semantic_signal_family")
                or i.get("semantic_classification"),
                "semantic_status": i.get("semantic_status"),
                "user_hypothesis": i.get("user_expected_category") or i.get("expected_category"),
                "requires_market_match": False,
                "paper_demo_only": True,
            }
    return {"ok": False, "error": "watchlist_item_not_found"}


def evaluate_watchlist_item(item_id: str) -> dict[str, Any]:
    """One safe paper/demo evaluation — does not force a trade."""
    item = get_watchlist_item(item_id)
    if not item:
        return {"ok": False, "error": "watchlist_item_not_found"}

    resolve_result = resolve_watchlist_identity(item_id)
    semantic_result = run_watchlist_semantic_check(item_id)
    item = get_watchlist_item(item_id) or item

    from app.ae13b_product.demo_risk_guard import evaluate_demo_risk_guard
    from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate
    from app.ae13b_product.reentry_blocks import get_manual_cooldown_fields
    from app.ae13b_product.stale_price_status import build_stale_price_status, row_price_freshness

    resolution = (resolve_result or {}).get("resolution") or {}
    price = resolution.get("matched_price")
    if price is None:
        price = item.get("latest_price")
    price_ts = resolution.get("matched_price_ts") or item.get("latest_price_timestamp")

    pair_addr = (
        item.get("matched_pair_address")
        or item.get("user_entered_contract_or_pair_address")
        or item.get("user_contract_address")
    )
    chain = item.get("display_chain") or item.get("chain")
    symbol = item.get("display_symbol") or item.get("user_entered_symbol")

    # AE13I: manual reentry cooldown precheck — before the risk guard, same
    # ordering as the Demo Queue evaluation path.
    cooldown = get_manual_cooldown_fields(pair_address=pair_addr, chain=chain, symbol=symbol)
    if cooldown.get("manual_cooldown_active") or cooldown.get("reentry_blocked"):
        return {
            "ok": True,
            "item": item,
            "resolution": resolution,
            "semantic": semantic_result,
            "decision": "BLOCKED_MANUAL_REENTRY_COOLDOWN",
            "rejection_code": "MANUAL_REENTRY_BLOCK_ACTIVE",
            "blocking_guards": ["manual_reentry_block"],
            "selected": False,
            "blocked": True,
            "can_demo_trade": False,
            "reason": "Manual re-entry cooldown active for this pair.",
            "last_decision": "BLOCKED_MANUAL_REENTRY_COOLDOWN",
            "manual_cooldown_active": True,
            "manual_cooldown_expiry": cooldown.get("manual_cooldown_expiry"),
            "manual_cooldown_remaining_seconds": cooldown.get("manual_cooldown_remaining_seconds"),
            "manual_cooldown_scope": cooldown.get("manual_cooldown_scope"),
            "demo_queue_status": "blocked_by_manual_reentry_cooldown",
            "next_action": "Wait for the re-entry cooldown to expire before demo evaluation.",
            "next_possible_action": "Wait for the re-entry cooldown to expire before demo evaluation.",
            "paper_demo_only": True,
            "not_live_approved": True,
            "live_trading_implied": False,
        }

    # AE13I: MarketDataGateKeeper — freshness/provenance/address-role/historical
    # checks. semantic_status stays a fully separate field and never sets passed.
    gate = validate_market_data_gate(
        {
            "chain": chain,
            "symbol": symbol,
            "pair_address": pair_addr,
            "latest_price": price,
            "price_updated_at": price_ts,
            "latest_liquidity": resolution.get("matched_liquidity") or item.get("latest_liquidity"),
            "liquidity_updated_at": price_ts,
            "source_provider": resolution.get("resolution_source") or "watchlist_eval",
        },
        for_open=True,
        # AE13I fix: stagnant_price_guard now returns passed=True (with
        # momentum_evidence="unknown_insufficient_delta_fields") when no
        # delta fields are present, so it is safe to run here
        # (skip_stagnant=False) even though watchlist identity resolution
        # does not carry 1h/4h activity deltas.
        skip_stagnant=False,
    )

    risk = evaluate_demo_risk_guard(
        requested_notional=75.0,
        pair_address=item.get("matched_pair_address")
        or item.get("user_entered_contract_or_pair_address"),
        symbol=item.get("display_symbol") or item.get("user_entered_symbol"),
        chain=item.get("display_chain") or item.get("chain"),
        price=price,
        price_timestamp=price_ts,
        liquidity=resolution.get("matched_liquidity") or item.get("latest_liquidity"),
        strategy_lane="manual_watchlist_scout",
    )

    stale = row_price_freshness(
        price=price,
        timestamp=price_ts,
        symbol=item.get("display_symbol") or item.get("user_entered_symbol"),
        pair=item.get("display_id"),
        source=resolution.get("resolution_source") or "watchlist_eval",
    )

    blockers: list[str] = []
    can_demo = False
    decision = "WATCH"
    dq_status = "evaluated_watch"

    if not gate.get("passed"):
        blockers.extend(gate.get("rejection_reasons") or ["Blocked by market data gate"])
        decision = "BLOCKED"
        dq_status = "blocked_by_market_data_gate"
    elif price is None or float(price or 0) <= 0:
        blockers.append("No current price available for paper evaluation.")
        decision = "NOT_ENOUGH_DATA"
        dq_status = "blocked_by_missing_price"
    elif stale.get("is_stale"):
        blockers.append(
            f"Price is stale: last update {stale.get('price_age_label')}, "
            f"limit {stale.get('freshness_limit_label')}."
        )
        decision = "BLOCKED"
        dq_status = "blocked_by_stale_price"
    if not risk["risk_guard_passed"]:
        blockers.append(risk["risk_guard_reason"])
        decision = "BLOCKED"
        if "stale" in str(risk["risk_guard_reason"]).lower():
            dq_status = "blocked_by_stale_price"
        else:
            dq_status = "blocked_by_risk"
    fam = item.get("semantic_signal_family") or item.get("semantic_classification")
    if fam in (
        "UNKNOWN_INSUFFICIENT_EVIDENCE",
        "UNKNOWN_UNRESOLVED",
        "NEEDS_REVIEW",
        "SOCIAL_CANDIDATE_NEEDS_VERIFICATION",
        None,
        "",
    ):
        if decision not in ("NOT_ENOUGH_DATA", "BLOCKED"):
            blockers.append("blocked_by_semantic_uncertainty")
            dq_status = "blocked_by_semantic_uncertainty"
    elif risk["risk_guard_passed"] and decision not in ("NOT_ENOUGH_DATA", "BLOCKED"):
        decision = "DEMO_CANDIDATE"
        can_demo = True
        dq_status = "eligible_for_demo"
        # paper_buy_allowed only if risk says so — still not placing order
        if can_demo:
            dq_status = "paper_buy_allowed"

    return {
        "ok": True,
        "item": item,
        "resolution": resolution,
        "semantic": semantic_result,
        "risk_guard": risk,
        "stale_price_status": stale,
        "gate_result": gate,
        "tradability_status": gate.get("tradability_status"),
        "freshness_gate_status": gate.get("freshness_gate_status"),
        "address_role": gate.get("address_role_status"),
        "address_role_status": gate.get("address_role_status"),
        "provenance_status": gate.get("provenance_status"),
        "manual_cooldown_active": cooldown.get("manual_cooldown_active"),
        "manual_cooldown_expiry": cooldown.get("manual_cooldown_expiry"),
        "manual_cooldown_scope": cooldown.get("manual_cooldown_scope"),
        "selected": can_demo,
        "blocked": decision == "BLOCKED",
        "decision": decision,
        "reason": "; ".join(blockers) if blockers else "eligible_for_demo_evaluation",
        "market_data_availability": item.get("market_match_status"),
        "latest_price_status": "missing"
        if price is None
        else ("stale" if stale.get("is_stale") else "fresh"),
        # semantic_status stays independent of tradability_status/gate — it never
        # sets passed / can_demo_trade by itself.
        "semantic_status": fam,
        "risk_guard_result": risk,
        "data_status": item.get("market_match_status"),
        "semantic_label": fam,
        "price_freshness": stale.get("label"),
        "demo_queue_status": dq_status,
        "risk_status": "passed" if risk["risk_guard_passed"] else "blocked",
        "last_decision": decision,
        "can_demo_trade": can_demo,
        "next_action": (
            "Add to Demo Queue for Manual Watchlist Scout"
            if can_demo or decision in ("WATCH", "NOT_ENOUGH_DATA", "DEMO_CANDIDATE")
            else "Address blocker before demo consideration"
        ),
        "next_possible_action": (
            "Add to Demo Queue for Manual Watchlist Scout"
            if can_demo or decision in ("WATCH", "NOT_ENOUGH_DATA", "DEMO_CANDIDATE")
            else "Address blocker before demo consideration"
        ),
        "paper_demo_only": True,
        "not_live_approved": True,
        "live_trading_implied": False,
    }


def _upsert_identity_store(entry: dict[str, Any]) -> None:
    """Persist user-entered identity into the local Identity Store.

    Best-effort and non-fatal — never blocks watchlist add/update, never
    writes price/liquidity, only identity fields the user provided.
    """
    addr = (
        entry.get("user_entered_contract_or_pair_address")
        or entry.get("user_contract_address")
        or entry.get("contract_address")
    )
    pair = entry.get("user_entered_pair") or entry.get("user_pair") or entry.get("pair")
    if not addr and not pair:
        return
    try:
        from app.ae13b_product import identity_store

        identity_store.upsert_identity(
            chain=entry.get("user_entered_chain") or entry.get("chain"),
            address=addr or pair,
            symbol=entry.get("user_entered_symbol") or entry.get("user_symbol") or entry.get("symbol"),
            name=entry.get("user_entered_name") or entry.get("user_name"),
            display_label=entry.get("user_display_label"),
            pair_address=pair or (addr if addr == pair else None),
            source="watchlist_user_input",
            watchlist_item_id=entry.get("id") or entry.get("watchlist_id"),
        )
    except Exception:
        pass


def _feed_registry(entry: dict[str, Any], coin: dict[str, Any] | None = None) -> None:
    try:
        from app.ae13_semantic.runtime_registry import get_semantic_registry

        cand = {
            "symbol": entry.get("user_entered_symbol")
            or entry.get("user_symbol")
            or entry.get("symbol")
            or (coin or {}).get("symbol"),
            "name": entry.get("user_entered_name")
            or (coin or {}).get("name")
            or entry.get("market_name")
            or entry.get("user_entered_symbol")
            or entry.get("symbol"),
            "chain": entry.get("user_entered_chain") or entry.get("chain"),
            "pair_address": entry.get("matched_pair_address")
            or entry.get("user_entered_pair")
            or entry.get("user_pair")
            or entry.get("pair")
            or entry.get("user_entered_contract_or_pair_address")
            or entry.get("user_contract_address")
            or entry.get("contract_address"),
            "coin_id": f"watchlist:{entry.get('id')}",
            "price_usd": (coin or {}).get("latest_price") or (coin or {}).get("price_usd"),
            "liquidity_usd": (coin or {}).get("latest_liquidity"),
            "volume_24h": (coin or {}).get("latest_volume_24h"),
            "whale_score": (coin or {}).get("latest_whale_score"),
            "user_expected_category": entry.get("user_expected_category")
            or entry.get("expected_category"),
            "force_reclassify": True,
            "pinned": True,
        }
        rec = get_semantic_registry().observe_candidate(cand)
        entry["semantic_classification"] = rec.get("semantic_signal_family")
        entry["semantic_signal_family"] = rec.get("semantic_signal_family")
        entry["evidence_summary"] = rec.get("evidence_summary")
        entry["semantic_label_human"] = rec.get("semantic_label_human")
        entry["trading_opportunity_state"] = rec.get("trading_opportunity_state")
        fam = rec.get("semantic_signal_family")
        if fam in ("UNKNOWN_INSUFFICIENT_EVIDENCE", "UNKNOWN_UNRESOLVED", None):
            entry["semantic_status"] = (
                "not_checked" if fam is None else "unknown_insufficient_evidence"
            )
        elif fam == "NEEDS_REVIEW":
            entry["semantic_status"] = "needs_review"
        else:
            entry["semantic_status"] = "classified"
    except Exception:
        entry["data_collection_status"] = entry.get("data_collection_status") or "registry_unavailable"
        entry["semantic_status"] = entry.get("semantic_status") or "provider_unavailable"
