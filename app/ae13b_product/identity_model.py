"""Non-destructive watchlist identity model.

UserEnteredIdentity is immutable except via explicit user edit.
ResolvedIdentity and MarketEnrichment are enrichment-only and must never
overwrite user-entered fields.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _s(v: Any) -> str | None:
    if v is None:
        return None
    t = str(v).strip()
    return t if t else None


def build_user_entered_identity(entry: dict[str, Any]) -> dict[str, Any]:
    """Snapshot of user-authored identity fields (never resolver-written)."""
    return {
        "user_entered_symbol": _s(
            entry.get("user_entered_symbol") or entry.get("user_symbol") or entry.get("symbol")
        ),
        "user_entered_name": _s(entry.get("user_entered_name") or entry.get("user_name")),
        "user_entered_pair": _s(
            entry.get("user_entered_pair") or entry.get("user_pair") or entry.get("pair")
        ),
        "user_entered_contract_or_pair_address": _s(
            entry.get("user_entered_contract_or_pair_address")
            or entry.get("user_contract_address")
            or entry.get("contract_address")
        ),
        "user_entered_chain": _s(entry.get("user_entered_chain") or entry.get("chain")),
        "user_display_label": _s(entry.get("user_display_label") or entry.get("user_nickname")),
        "user_expected_category": _s(
            entry.get("user_expected_category") or entry.get("expected_category")
        )
        or "",
        "user_note": _s(entry.get("user_note")) or "",
        "user_evidence_url": _s(entry.get("user_evidence_url")) or "",
        "user_evidence_note": _s(entry.get("user_evidence_note")) or "",
        "user_claimed_social_mission": _s(entry.get("user_claimed_social_mission")) or "",
        "created_at": entry.get("created_at") or entry.get("first_added_at") or entry.get("added_at"),
        "updated_at": entry.get("user_identity_updated_at")
        or entry.get("updated_at")
        or entry.get("created_at"),
    }


def build_resolved_identity(entry: dict[str, Any]) -> dict[str, Any]:
    """Resolver output — separate from user-entered identity."""
    resolution = entry.get("resolution") if isinstance(entry.get("resolution"), dict) else {}
    return {
        "resolved_symbol": _s(entry.get("resolved_symbol") or resolution.get("matched_symbol")),
        "resolved_name": _s(entry.get("resolved_name") or resolution.get("matched_name")),
        "resolved_pair": _s(
            entry.get("resolved_pair")
            or resolution.get("matched_pair_address")
            or entry.get("matched_pair_address")
        ),
        "resolved_chain": _s(entry.get("resolved_chain") or resolution.get("matched_chain")),
        "resolved_contract_address": _s(
            entry.get("resolved_contract_address")
            or resolution.get("matched_contract_address")
            or entry.get("resolved_contract_or_pair_address")
        ),
        "resolved_pair_address": _s(
            entry.get("resolved_pair_address")
            or resolution.get("matched_pair_address")
            or entry.get("matched_pair_address")
        ),
        "resolution_status": (
            entry.get("identity_resolution_status")
            or resolution.get("resolution_status")
            or "unresolved_local_only"
        ),
        "resolution_source": resolution.get("resolution_source") or entry.get("resolution_source") or "none",
        "confidence": float(resolution.get("confidence") or entry.get("resolution_confidence") or 0.0),
        "reason": resolution.get("reason")
        or entry.get("resolution_reason")
        or "Not resolved against local market feed.",
        "checked_at": resolution.get("checked_at") or entry.get("last_checked_at") or entry.get("updated_at"),
    }


def build_market_enrichment(entry: dict[str, Any]) -> dict[str, Any]:
    """Live/local market enrichment — never replaces UserEnteredIdentity."""
    resolution = entry.get("resolution") if isinstance(entry.get("resolution"), dict) else {}
    return {
        "matched_live_market_status": entry.get("market_match_status")
        or entry.get("data_collection_status")
        or "waiting_for_market_match",
        "market_name": _s(entry.get("market_name")),
        "market_symbol": _s(entry.get("market_symbol")),
        "market_chain": _s(entry.get("market_chain")),
        "matched_market_address": _s(
            entry.get("matched_pair_address") or entry.get("matched_market_address")
        ),
        "latest_price": entry.get("latest_price")
        if entry.get("latest_price") is not None
        else resolution.get("matched_price"),
        "latest_price_source": entry.get("latest_price_source")
        or (
            "live_market"
            if entry.get("last_seen_in_market")
            else (resolution.get("resolution_source") or "none")
        ),
        "latest_price_timestamp": entry.get("latest_price_timestamp")
        or resolution.get("matched_price_ts")
        or entry.get("last_seen_in_market"),
        "latest_liquidity": entry.get("latest_liquidity")
        if entry.get("latest_liquidity") is not None
        else resolution.get("matched_liquidity"),
        "latest_volume_24h": entry.get("latest_volume_24h"),
        "latest_delta_5m": entry.get("latest_delta_5m"),
        "latest_delta_1h": entry.get("latest_delta_1h"),
        "latest_delta_6h": entry.get("latest_delta_6h"),
        "latest_delta_24h": entry.get("latest_delta_24h"),
        "first_seen_in_market": entry.get("first_seen_in_market"),
        "last_seen_in_market": entry.get("last_seen_in_market"),
    }


def identity_conflict(user: dict[str, Any], resolved: dict[str, Any]) -> dict[str, Any] | None:
    """Detect when resolver disagrees with user-entered identity."""
    conflicts: list[str] = []
    u_sym = (_s(user.get("user_entered_symbol")) or "").upper()
    r_sym = (_s(resolved.get("resolved_symbol")) or "").upper()
    if u_sym and r_sym and u_sym.split("/")[0] != r_sym.split("/")[0]:
        conflicts.append(f"symbol: user={u_sym} vs resolved={r_sym}")
    u_name = (_s(user.get("user_entered_name")) or "").lower()
    r_name = (_s(resolved.get("resolved_name")) or "").lower()
    if u_name and r_name and u_name != r_name:
        conflicts.append(f"name: user={user.get('user_entered_name')} vs resolved={resolved.get('resolved_name')}")
    u_chain = (_s(user.get("user_entered_chain")) or "").lower()
    r_chain = (_s(resolved.get("resolved_chain")) or "").lower()
    if u_chain and r_chain and u_chain != r_chain:
        conflicts.append(f"chain: user={u_chain} vs resolved={r_chain}")
    if not conflicts:
        return None
    return {
        "has_conflict": True,
        "conflicts": conflicts,
        "resolution_status": "conflict",
        "note": "Resolver disagrees with user-entered identity. User-entered remains primary.",
    }


def attach_identity_objects(entry: dict[str, Any]) -> dict[str, Any]:
    """Attach nested identity objects without mutating user-entered flat fields."""
    user = build_user_entered_identity(entry)
    resolved = build_resolved_identity(entry)
    market = build_market_enrichment(entry)
    conflict = identity_conflict(user, resolved)
    entry["user_entered_identity"] = user
    entry["resolved_identity"] = resolved
    entry["market_enrichment"] = market
    entry["identity_conflict"] = conflict
    # Keep flat user fields aligned FROM user object only (never from resolved)
    for k, v in user.items():
        if k in ("created_at", "updated_at"):
            continue
        if v is not None and v != "":
            entry[k] = v
    return entry


def apply_resolved_only(entry: dict[str, Any], resolution: dict[str, Any]) -> None:
    """Write resolver output into ResolvedIdentity / enrichment fields only.

    Must NOT overwrite any user_entered_* field.
    """
    status = resolution.get("resolution_status") or "unresolved_local_only"
    entry["identity_resolution_status"] = status
    entry["resolution"] = dict(resolution)
    entry["resolved_symbol"] = resolution.get("matched_symbol")
    entry["resolved_name"] = resolution.get("matched_name")
    entry["resolved_chain"] = resolution.get("matched_chain")
    entry["resolved_pair"] = resolution.get("matched_pair_address")
    entry["resolved_contract_address"] = resolution.get("matched_contract_address")
    entry["resolved_pair_address"] = resolution.get("matched_pair_address")
    entry["resolved_contract_or_pair_address"] = (
        resolution.get("matched_contract_address") or resolution.get("matched_pair_address")
    )
    entry["resolution_source"] = resolution.get("resolution_source")
    entry["resolution_reason"] = resolution.get("reason")
    entry["resolution_confidence"] = resolution.get("confidence")

    # Market enrichment only
    if resolution.get("matched_price") is not None:
        entry["latest_price"] = resolution.get("matched_price")
        entry["latest_price_source"] = resolution.get("resolution_source") or "local_resolver"
        entry["latest_price_timestamp"] = resolution.get("matched_price_ts")
    if resolution.get("matched_liquidity") is not None:
        entry["latest_liquidity"] = resolution.get("matched_liquidity")
    if resolution.get("matched_symbol"):
        entry["market_symbol"] = resolution["matched_symbol"]
    if resolution.get("matched_name"):
        entry["market_name"] = resolution["matched_name"]
    if resolution.get("matched_chain"):
        entry["market_chain"] = resolution["matched_chain"]
    if resolution.get("matched_pair_address"):
        entry["matched_pair_address"] = resolution["matched_pair_address"]

    # Explicitly do not touch user_entered_* keys
    protected = (
        "user_entered_symbol",
        "user_entered_name",
        "user_entered_pair",
        "user_entered_contract_or_pair_address",
        "user_entered_chain",
        "user_display_label",
        "user_expected_category",
        "user_note",
        "user_evidence_url",
        "user_evidence_note",
        "user_claimed_social_mission",
        "user_symbol",
        "user_pair",
        "user_contract_address",
    )
    # No-op assert for callers — values must remain
    _ = protected
    entry["updated_at"] = _utc_now()
