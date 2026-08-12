"""Contract / identity resolver — local/runtime data only unless explicitly allowed.

Does not call external providers silently.
Market match is enrichment; unresolved identity can still show user-entered fields.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

CHAIN_ALIASES: dict[str, str] = {
    "eth": "ethereum",
    "ethereum": "ethereum",
    "sol": "solana",
    "solana": "solana",
    "bsc": "bsc",
    "binance": "bsc",
    "bnb": "bsc",
    "base": "base",
    "arbitrum": "arbitrum",
    "arb": "arbitrum",
    "polygon": "polygon",
    "matic": "polygon",
    "avalanche": "avalanche",
    "avax": "avalanche",
    "robinhood": "robinhood",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_chain(chain: str | None) -> str:
    raw = str(chain or "").strip().lower()
    if not raw:
        return ""
    return CHAIN_ALIASES.get(raw, raw)


def normalize_address(addr: str | None, *, chain: str | None = None) -> str:
    """EVM: lowercase; Solana: exact base58 string (preserve case)."""
    raw = str(addr or "").strip()
    if not raw:
        return ""
    ch = normalize_chain(chain)
    if raw.startswith("0x") or ch in (
        "ethereum",
        "bsc",
        "base",
        "arbitrum",
        "polygon",
        "avalanche",
    ):
        return raw.lower()
    # Solana / other — preserve exact string for matching, strip whitespace
    return raw


def addresses_equal(a: str | None, b: str | None, *, chain: str | None = None) -> bool:
    na = normalize_address(a, chain=chain)
    nb = normalize_address(b, chain=chain)
    if not na or not nb:
        return False
    ch = normalize_chain(chain)
    if ch == "solana" or (not na.startswith("0x") and not nb.startswith("0x")):
        return na == nb
    return na.lower() == nb.lower()


def classify_contract_format(addr: str | None, *, chain: str | None = None) -> str:
    raw = str(addr or "").strip()
    if not raw:
        return "user_entered_only"
    ch = normalize_chain(chain) or ("ethereum" if raw.startswith("0x") else "solana")
    if ch == "solana":
        # Base58 rough check
        if 32 <= len(raw) <= 44 and not raw.startswith("0x"):
            return "contract_format_valid"
        return "contract_format_invalid"
    if raw.startswith("0x") and len(raw) == 42:
        try:
            int(raw[2:], 16)
            return "contract_format_valid"
        except ValueError:
            return "contract_format_invalid"
    if len(raw) >= 40:  # possible pair / long id
        return "contract_format_valid"
    return "contract_format_invalid"


# Canonical AE13F resolution statuses
STATUS_LOCAL_MATCH = "local_match"
STATUS_USER_ENTERED = "user_entered_identity"
STATUS_EXTERNAL = "external_match"
STATUS_UNRESOLVED_LOCAL = "unresolved_local_only"
STATUS_PROVIDER_UNAVAILABLE = "provider_unavailable"
STATUS_UNSUPPORTED_CHAIN = "unsupported_chain"
STATUS_INVALID = "invalid_address"
STATUS_CONFLICT = "conflict"
STATUS_ERROR = "error"

# Legacy aliases still accepted in UI
_LEGACY_STATUS_MAP = {
    "matched_live_market": STATUS_LOCAL_MATCH,
    "matched_registry": STATUS_LOCAL_MATCH,
    "matched_static_snapshot": STATUS_LOCAL_MATCH,
    "matched_db": STATUS_LOCAL_MATCH,
    "unresolved": STATUS_UNRESOLVED_LOCAL,
    "user_entered_only": STATUS_USER_ENTERED,
    "resolver_unavailable": STATUS_PROVIDER_UNAVAILABLE,
}


def normalize_resolution_status(status: str | None) -> str:
    raw = str(status or "").strip()
    if not raw:
        return STATUS_UNRESOLVED_LOCAL
    return _LEGACY_STATUS_MAP.get(raw, raw)


def _empty_result(**overrides: Any) -> dict[str, Any]:
    base = {
        "resolution_status": STATUS_UNRESOLVED_LOCAL,
        "resolution_source": "none",
        "matched_symbol": None,
        "matched_name": None,
        "matched_chain": None,
        "matched_contract_address": None,
        "matched_pair_address": None,
        "matched_price": None,
        "matched_price_ts": None,
        "matched_liquidity": None,
        "confidence": 0.0,
        "reason": (
            "No match in local market/DB/registry. "
            "External resolver not enabled. This item is tracked from user input and local data only."
        ),
        "checked_at": _utc_now(),
        "external_resolver_attempted": False,
        "external_resolver_available": False,
        "paper_demo_only": True,
    }
    base.update(overrides)
    if "resolution_status" in base:
        base["resolution_status"] = normalize_resolution_status(base["resolution_status"])
        # Keep legacy alias for older UI
        base["resolution_status_legacy"] = overrides.get("resolution_status") or base["resolution_status"]
    return base


def resolve_identity(
    *,
    chain: str | None = None,
    contract_or_pair_address: str | None = None,
    symbol: str | None = None,
    pair_address: str | None = None,
    token_address: str | None = None,
    allow_external: bool = False,
) -> dict[str, Any]:
    """Resolve identity using local sources only (unless allow_external explicitly True).

    Order:
      1. Identity Store exact match (local, user-entered — never fabricates price)
      2. Watchlist user-entered identity
      3. Current live market (coins table)
      4. Local DB
      5. Runtime semantic registry
      6. Static snapshot context
      7. Configured external provider only if allow_external and available
    """
    ch = normalize_chain(chain)
    addr = (contract_or_pair_address or pair_address or token_address or "").strip()
    sym = str(symbol or "").strip().upper()
    if "/" in sym:
        sym = sym.split("/")[0].strip()

    fmt = classify_contract_format(addr, chain=ch or None)
    if not addr and not sym:
        return _empty_result(
            resolution_status="unresolved",
            reason="No contract/pair address or symbol provided.",
        )

    # 1: Identity Store exact match (before live market / DB / registry).
    # Never fabricates price — identity-only, so matched_price stays None here.
    try:
        from app.ae13b_product import identity_store

        store_rec = identity_store.lookup_for_resolver(
            chain=ch or None, address=addr or None, symbol=sym or None
        )
    except Exception:
        store_rec = None
    if store_rec:
        return _empty_result(
            resolution_status=STATUS_USER_ENTERED,
            resolution_source="local_identity_store",
            matched_symbol=store_rec.get("symbol"),
            matched_name=store_rec.get("name"),
            matched_chain=store_rec.get("chain") or ch,
            matched_contract_address=store_rec.get("address"),
            matched_pair_address=store_rec.get("pair_address"),
            matched_price=None,
            confidence=0.85,
            reason=(
                "Identity known from local store. Market price unavailable until "
                "feed/resolver supplies data."
            ),
        )

    # 2: Watchlist user-entered identity (checked before live market/DB).
    try:
        from app.analytics.watchlist import list_watchlist

        for w in list_watchlist(include_disabled=True):
            w_chain = normalize_chain(w.get("chain"))
            if ch and w_chain and ch != w_chain:
                continue
            w_addr = w.get("user_contract_address") or w.get("contract_address") or ""
            w_pair = w.get("user_pair") or w.get("pair") or w.get("matched_pair_address") or ""
            w_sym = str(w.get("user_symbol") or w.get("symbol") or "").strip().upper()
            if addr and (
                addresses_equal(addr, w_addr, chain=w_chain or ch)
                or addresses_equal(addr, w_pair, chain=w_chain or ch)
            ):
                if w.get("market_symbol") or w.get("matched_pair_address") or w_addr:
                    return _empty_result(
                        resolution_status=STATUS_LOCAL_MATCH
                        if w.get("last_seen_in_market")
                        else STATUS_USER_ENTERED,
                        resolution_source="watchlist_user_input",
                        matched_symbol=w.get("market_symbol")
                        or w.get("user_entered_symbol")
                        or w.get("user_symbol"),
                        matched_name=w.get("market_name") or w.get("user_entered_name"),
                        matched_chain=w_chain or ch,
                        matched_contract_address=w_addr or None,
                        matched_pair_address=w.get("matched_pair_address") or w_pair,
                        confidence=0.6 if w.get("last_seen_in_market") else 0.85,
                        reason=(
                            "Tracked from user input with prior market enrichment."
                            if w.get("market_symbol")
                            else (
                                "Tracked from user input. Not found in current local market feed. "
                                "External lookup not enabled."
                            )
                        ),
                    )
            if sym and w_sym and w_sym.split("/")[0] == sym:
                return _empty_result(
                    resolution_status=STATUS_USER_ENTERED,
                    resolution_source="watchlist_user_input",
                    matched_symbol=w.get("user_entered_symbol") or w.get("user_symbol") or w.get("symbol"),
                    matched_name=w.get("user_entered_name"),
                    matched_chain=w_chain or ch,
                    matched_contract_address=w_addr or None,
                    confidence=0.4,
                    reason="Found symbol in watchlist history; not matched to live market.",
                )
    except Exception:
        pass

    # 3-4: Current live market / local DB (coins table)
    try:
        from app import database as db

        coins = db.get_coins(limit=500, sort_by="whale_score")
    except Exception as exc:
        coins = []
        db_err = str(exc)[:200]
    else:
        db_err = None

    for c in coins:
        c_chain = normalize_chain(c.get("chain"))
        if ch and c_chain and ch != c_chain:
            continue
        c_pair = c.get("pair_address") or ""
        c_token = c.get("contract_address") or c.get("token_address") or ""
        c_sym = str(c.get("symbol") or "").strip().upper()
        if "/" in c_sym:
            c_sym = c_sym.split("/")[0].strip()

        match_how = None
        if addr and addresses_equal(addr, c_pair, chain=c_chain or ch):
            match_how = "pair_address"
        elif addr and c_token and addresses_equal(addr, c_token, chain=c_chain or ch):
            match_how = "token_contract"
        elif sym and c_sym == sym:
            match_how = "symbol"

        if not match_how:
            continue

        return _empty_result(
            resolution_status=STATUS_LOCAL_MATCH,
            resolution_source="live_market" if match_how != "symbol" else "local_db",
            matched_symbol=c.get("symbol"),
            matched_name=c.get("name"),
            matched_chain=c_chain or ch,
            matched_contract_address=c_token or None,
            matched_pair_address=c_pair or None,
            matched_price=c.get("latest_price"),
            matched_price_ts=c.get("last_seen_at"),
            matched_liquidity=c.get("latest_liquidity"),
            confidence=0.95 if match_how != "symbol" else 0.7,
            reason=f"Exact match in current live market table via {match_how}.",
            match_how=match_how,
        )

    # 5: runtime semantic registry
    try:
        from app.ae13_semantic.runtime_registry import get_semantic_registry

        snap = get_semantic_registry().snapshot()
        records = (snap.get("records") or {}) if isinstance(snap, dict) else {}
        for _key, rec in records.items():
            if not isinstance(rec, dict):
                continue
            r_chain = normalize_chain(rec.get("chain"))
            if ch and r_chain and ch != r_chain:
                continue
            r_pair = rec.get("pair_address") or ""
            r_sym = str(rec.get("symbol") or "").strip().upper()
            if addr and addresses_equal(addr, r_pair, chain=r_chain or ch):
                return _empty_result(
                    resolution_status=STATUS_LOCAL_MATCH,
                    resolution_source="runtime_registry",
                    matched_symbol=rec.get("symbol"),
                    matched_name=rec.get("name"),
                    matched_chain=r_chain or ch,
                    matched_pair_address=r_pair,
                    matched_contract_address=rec.get("contract_address"),
                    confidence=0.8,
                    reason="Matched runtime semantic registry by pair/contract.",
                )
            if sym and r_sym == sym:
                return _empty_result(
                    resolution_status=STATUS_LOCAL_MATCH,
                    resolution_source="runtime_registry",
                    matched_symbol=rec.get("symbol"),
                    matched_name=rec.get("name"),
                    matched_chain=r_chain or ch,
                    matched_pair_address=r_pair or None,
                    confidence=0.55,
                    reason="Matched runtime semantic registry by symbol (lower confidence).",
                )
    except Exception:
        pass

    # 6: static snapshot (semantic coverage) — optional soft match by symbol
    try:
        from pathlib import Path

        from app.ae13_reconciliation.semantic_coverage import build_semantic_coverage

        static = build_semantic_coverage(Path(__file__).resolve().parents[2])
        coins_static = static.get("coins") or static.get("coin_rows") or []
        if isinstance(coins_static, list) and sym:
            for sc in coins_static[:500]:
                if not isinstance(sc, dict):
                    continue
                s_sym = str(sc.get("symbol") or "").strip().upper()
                if s_sym.split("/")[0] == sym:
                    return _empty_result(
                        resolution_status=STATUS_LOCAL_MATCH,
                        resolution_source="static_snapshot",
                        matched_symbol=sc.get("symbol"),
                        matched_name=sc.get("name"),
                        matched_chain=normalize_chain(sc.get("chain")) or ch,
                        confidence=0.35,
                        reason="Matched static research snapshot by symbol (not live market).",
                    )
    except Exception:
        pass

    # 7: external — only if explicitly allowed (never silent)
    if allow_external:
        try:
            from app.ae13b_product.external_resolver import attempt_external_lookup

            ext = attempt_external_lookup(
                chain=ch,
                contract_or_pair_address=addr,
                symbol=sym or None,
                user_confirmed=True,
            )
            return _empty_result(
                resolution_status=ext.get("resolution_status") or STATUS_PROVIDER_UNAVAILABLE,
                resolution_source=ext.get("resolution_source") or "external_provider",
                matched_symbol=ext.get("matched_symbol"),
                matched_name=ext.get("matched_name"),
                matched_chain=ext.get("matched_chain") or ch,
                matched_contract_address=ext.get("matched_contract_address") or addr,
                confidence=float(ext.get("confidence") or 0.0),
                reason=ext.get("reason")
                or "External resolver not enabled. This item is tracked from user input and local data only.",
                external_resolver_attempted=bool(ext.get("external_resolver_attempted")),
                external_resolver_available=bool(ext.get("external_resolver_available")),
                cache_hit=bool(ext.get("cache_hit")),
                provider_name=ext.get("provider_name"),
            )
        except Exception as exc:
            return _empty_result(
                resolution_status=STATUS_ERROR,
                resolution_source="external_provider",
                matched_chain=ch or None,
                matched_contract_address=addr or None,
                reason=f"External resolver error: {exc}"[:240],
                external_resolver_attempted=True,
                external_resolver_available=False,
            )

    # Layer 2 — user-provided identity without market match
    if addr or sym:
        reason_parts = []
        if fmt == "contract_format_invalid":
            reason_parts.append("Contract format looks invalid for the selected chain.")
            status = STATUS_INVALID
        elif fmt == "contract_format_valid" or addr:
            reason_parts.append(
                "Tracked from user input. Not found in current local market feed. "
                "External lookup not enabled."
            )
            status = STATUS_USER_ENTERED if (sym or addr) else STATUS_UNRESOLVED_LOCAL
        else:
            status = STATUS_USER_ENTERED
            reason_parts.append("Tracked from user input (symbol/name only).")

        if db_err:
            reason_parts.append(f"DB lookup issue: {db_err}")
        reason_parts.append(
            "External resolver not called (intentionally disabled / not silently invoked)."
        )
        if ch:
            reason_parts.append(f"Checked chain alias: {ch}.")

        return _empty_result(
            resolution_status=status,
            resolution_source="watchlist_user_input" if (addr or sym) else "none",
            matched_chain=ch or None,
            matched_contract_address=addr or None,
            matched_symbol=sym or None,
            confidence=0.9 if (addr or sym) else 0.0,
            reason=" ".join(reason_parts),
            identity_format=fmt,
        )

    return _empty_result(
        resolution_status=STATUS_UNRESOLVED_LOCAL,
        resolution_source="none",
        matched_chain=ch or None,
        reason="No contract/pair address or symbol provided.",
        identity_format=fmt,
    )
