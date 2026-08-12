"""AE13B Live Market + RSS snapshot builder (paper/demo product view)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _freshness(ts: str | None) -> dict[str, Any]:
    if not ts:
        return {
            "label": "Market data: no timestamp available",
            "fresh": False,
            "age_seconds": None,
        }
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        age = (datetime.now(timezone.utc) - t).total_seconds()
    except ValueError:
        return {"label": "Market data stale: bad timestamp", "fresh": False, "age_seconds": None}
    if age < 120:
        return {
            "label": f"Market data fresh: updated {int(age)} seconds ago",
            "fresh": True,
            "age_seconds": age,
        }
    if age < 900:
        return {
            "label": f"Market data stale: last update {int(age / 60)} minutes ago",
            "fresh": False,
            "age_seconds": age,
        }
    return {
        "label": (
            f"Some candidates have stale prices. Market Snapshot Feed still updating. "
            f"(age {int(age / 60)}m, limit 2m)"
        ),
        "fresh": False,
        "age_seconds": age,
    }


#: Address roles that mean "this is a pool/pair address, not a token contract".
_POOL_OR_PAIR_ROLES = ("pool_address", "pair_contract", "market_account", "provider_pair_id")

CONTRACT_ADDRESS_LEGACY_ALIAS_WARNING = (
    "Legacy alias. This may be a pool/pair address, not a token contract."
)


def compute_contract_address_disclosure(
    *,
    raw_contract_address: str | None,
    address_role: str | None,
    token_contract_address: str | None,
    token_mint_address: str | None,
    pair_address: str | None,
) -> dict[str, Any]:
    """AE13I Smoke Addendum (Part C): never silently alias contract_address = pair.

    Returns the legacy ``contract_address`` value (kept for backward compat)
    plus explicit disclosure fields so callers/UI never mistake a pool/pair
    address for an actual token-mint/token-contract address.
    """
    is_actual_token_address = address_role in ("token_mint", "token_contract")
    contract_address_value = (
        raw_contract_address or token_contract_address or token_mint_address or pair_address
    )
    if is_actual_token_address:
        contract_address_role = None
        contract_address_warning = None
    else:
        if address_role == "pair_contract":
            contract_address_role = "pair_address_alias"
        elif address_role in ("pool_address", "market_account"):
            contract_address_role = "pool_address_alias"
        else:
            contract_address_role = "unknown_legacy_alias"
        contract_address_warning = CONTRACT_ADDRESS_LEGACY_ALIAS_WARNING
    address_display_label = (
        "Pool / Pair address" if address_role in _POOL_OR_PAIR_ROLES else "Contract address"
    )
    return {
        "contract_address": contract_address_value,
        "contract_address_deprecated": True,
        "contract_address_role": contract_address_role,
        "contract_address_warning": contract_address_warning,
        "address_display_label": address_display_label,
        "token_contract_address": token_contract_address,
        "token_mint_address": token_mint_address,
    }


def _latest_snapshot_map(coin_ids: list[int]) -> dict[int, dict[str, Any]]:
    if not coin_ids:
        return {}
    from app import database as db

    out: dict[int, dict[str, Any]] = {}
    try:
        with db.get_db() as conn:
            for cid in coin_ids:
                row = conn.execute(
                    """
                    SELECT * FROM market_snapshots
                    WHERE coin_id = ?
                    ORDER BY timestamp DESC LIMIT 1
                    """,
                    (cid,),
                ).fetchone()
                if row:
                    out[int(cid)] = db._row_to_dict(row) or {}
    except Exception:
        return out
    return out


# Semantic family filters — exact enum/equality match (API-level, not frontend-only)
# Do not mix with trading_opportunity_state / legacy_cluster_label / user_expected_category
_SEMANTIC_FILTERS: dict[str, frozenset[str]] = {
    "social": frozenset({"SOCIAL_CONFIRMED"}),
    "opportunistic": frozenset(
        {
            "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
            "OPPORTUNISTIC_SUSPECTED",
        }
    ),
    "unknown": frozenset({"UNKNOWN_INSUFFICIENT_EVIDENCE", "NEEDS_REVIEW"}),
    "unresolved": frozenset({"UNKNOWN_UNRESOLVED", "UNKNOWN_INSUFFICIENT_EVIDENCE", "NEEDS_REVIEW"}),
    "infrastructure": frozenset({"NON_SOCIAL_INFRASTRUCTURE_CONFIRMED"}),
}

# Default filter behavior: hide non-matching rows (not dim)
DEFAULT_FILTER_MODE = "hide"


def build_live_market(
    limit: int = 50,
    status_filter: str | None = None,
    *,
    filter_mode: str | None = None,
) -> dict[str, Any]:
    from app import database as db
    from app.ae13_semantic.runtime_registry import get_semantic_registry
    from app.ae13b_product.address_role import enrich_row_with_address_role
    from app.ae13b_product.market_data_gatekeeper import compute_tradability_status
    from app.ae13b_product.provenance_enricher import enrich_market_provenance
    from app.ae13b_product.stale_price_status import build_stale_price_status, row_price_freshness

    registry = get_semantic_registry()
    mode = (filter_mode or DEFAULT_FILTER_MODE).lower().strip()
    if mode not in ("hide", "highlight"):
        mode = DEFAULT_FILTER_MODE
    try:
        from app.live import get_token_transparency_logs

        transparency = get_token_transparency_logs()
    except Exception:
        transparency = {"passed_count": 0, "dropped_count": 0, "scan_at": None}

    # When a semantic/status filter is active, pull a broader candidate set so
    # filtering happens at API level over more than the top-N whale rows.
    filt_early = (status_filter or "all").lower().strip()
    fetch_limit = max(limit * 8, 200) if filt_early and filt_early != "all" else limit
    try:
        coins = db.get_coins(limit=fetch_limit, sort_by="whale_score")
    except Exception:
        coins = []

    snaps = _latest_snapshot_map([int(c["id"]) for c in coins if c.get("id") is not None])
    open_pairs: set[str] = set()
    try:
        from app.execution.paper import get_paper_trader

        open_pairs = {
            str(p.get("pair_address") or "")
            for p in get_paper_trader().get_positions(status="OPEN")
        }
    except Exception:
        open_pairs = set()

    rows = []
    passed = 0
    blocked = 0
    watch = 0
    # Global freshness from unfiltered feed (not filtered subset)
    global_latest_ts = None
    for c in coins:
        ts_c = c.get("last_seen_at")
        if ts_c and (global_latest_ts is None or str(ts_c) > str(global_latest_ts)):
            global_latest_ts = ts_c

    for c in coins:
        cid = c.get("id")
        snap = snaps.get(int(cid), {}) if cid is not None else {}
        rec = registry.observe_candidate(
            {
                "id": cid,
                "coin_id": cid,
                "symbol": c.get("symbol"),
                "name": c.get("name"),
                "chain": c.get("chain"),
                "pair_address": c.get("pair_address"),
                "price_usd": c.get("latest_price"),
                "liquidity_usd": c.get("latest_liquidity"),
                "volume_24h": c.get("latest_volume_24h"),
                "whale_score": c.get("latest_whale_score"),
                "cluster_label": c.get("cluster_label"),
            }
        )
        family = str(rec.get("semantic_signal_family") or "UNKNOWN_INSUFFICIENT_EVIDENCE")
        pair = str(c.get("pair_address") or "")
        price = c.get("latest_price")
        row_ts = snap.get("timestamp") or c.get("last_seen_at")
        price_fresh = row_price_freshness(
            price=price,
            timestamp=row_ts,
            symbol=c.get("symbol"),
            pair=pair,
            source="live_market",
        )
        if pair in open_pairs:
            row_status = "Demo candidate"
            reason = "Already held as a current demo position."
        elif not price:
            row_status = "Blocked"
            reason = "Price data too old or missing for a confident demo trade."
            blocked += 1
        elif price_fresh.get("is_stale"):
            row_status = "Blocked"
            reason = price_fresh.get("label") or "Price stale for demo trade."
            blocked += 1
        elif family in ("UNKNOWN_UNRESOLVED", "UNKNOWN_INSUFFICIENT_EVIDENCE"):
            row_status = "Watch"
            reason = rec.get("unresolved_reason") or "Unknown — not enough evidence yet."
            watch += 1
        else:
            row_status = "Passed"
            reason = f"{rec.get('semantic_label_human')} — eligible for bounded paper/demo exploration."
            passed += 1

        # AE13I: lightweight per-row enrichment (address role + provenance +
        # tradability_status). Uses only local enrich helpers - no reentry/
        # stagnant checks here to keep the per-row cost low for the full feed.
        addr_enriched = enrich_row_with_address_role(c)
        prov_enriched = enrich_market_provenance(c)

        # AE13I Smoke Addendum (Part C): contract_address must never silently
        # become "the pair address" with no disclosure. token_contract_address
        # / token_mint_address stay separate (see AddressRoleClassifier); the
        # legacy contract_address field is kept ONLY for backward compat and is
        # always flagged as a deprecated alias when it is not an actual
        # token-mint/token-contract address.
        address_role = addr_enriched.get("address_role")
        contract_disclosure = compute_contract_address_disclosure(
            raw_contract_address=c.get("contract_address"),
            address_role=address_role,
            token_contract_address=addr_enriched.get("token_contract_address"),
            token_mint_address=addr_enriched.get("token_mint_address"),
            pair_address=pair,
        )
        row_freshness_status = "fail" if (not price or price_fresh.get("is_stale")) else "pass"
        tradability_status = compute_tradability_status(
            passed=row_status == "Passed" or pair in open_pairs,
            for_open=True,
            row={**c, **addr_enriched},
            freshness_status=row_freshness_status,
            blocking_guards=[] if price else ["freshness_missing_price"],
        )

        # Stable row key for keyed UI updates (never array index)
        row_key = None
        chain_s = str(c.get("chain") or "").lower()
        if chain_s and pair:
            row_key = f"{chain_s}|pair|{pair}"
        elif chain_s and c.get("contract_address"):
            row_key = f"{chain_s}|contract|{c.get('contract_address')}"
        elif rec.get("coin_identity") or rec.get("registry_key"):
            row_key = str(rec.get("coin_identity") or rec.get("registry_key"))
        else:
            row_key = f"live|{c.get('symbol')}|{rec.get('first_seen_at') or cid}"

        rows.append(
            {
                "row_key": row_key,
                "candidate_id": rec.get("coin_identity") or rec.get("registry_key") or row_key,
                "time": row_ts,
                "symbol": c.get("symbol"),
                "pair": pair,
                "pair_address": pair,
                **contract_disclosure,
                "chain": c.get("chain"),
                "price": price,
                "liquidity": c.get("latest_liquidity"),
                "volume_24h": c.get("latest_volume_24h"),
                "price_change_5m": snap.get("price_change_m5"),
                "price_change_1h": snap.get("price_change_h1"),
                "price_change_6h": snap.get("price_change_h6"),
                "price_change_24h": snap.get("price_change_h24"),
                "buy_ratio": snap.get("buy_ratio"),
                "whale_score": c.get("latest_whale_score"),
                "semantic_label": rec.get("semantic_label_human"),
                "semantic_signal_family": family,
                "semantic_status": rec.get("semantic_status")
                or (
                    "Unresolved"
                    if family in ("UNKNOWN_UNRESOLVED", "UNKNOWN_INSUFFICIENT_EVIDENCE", "NEEDS_REVIEW")
                    else "Classified"
                ),
                "semantic_family": family,
                "trading_opportunity_state": rec.get("trading_opportunity_state"),
                "opportunity_state": rec.get("trading_opportunity_state"),
                "legacy_cluster_label": c.get("cluster_label"),
                "status": row_status,
                "reason": reason,
                "seen_count": rec.get("seen_count"),
                "first_seen_at": rec.get("first_seen_at"),
                "last_seen_at": rec.get("last_seen_at"),
                "source": "live_market",
                "price_timestamp": row_ts,
                "price_age_seconds": price_fresh.get("price_age_seconds"),
                "price_freshness": price_fresh,
                "stale_price_applies_to": "selected_candidate",
                "blocks_demo_trade": price_fresh.get("blocks_demo_trade"),
                # AE13I enrichment
                "address_role": addr_enriched.get("address_role"),
                "address_role_status": addr_enriched.get("address_role_status"),
                "address_role_note": addr_enriched.get("address_role_note"),
                "ui_warning": addr_enriched.get("ui_warning"),
                "provenance_status": prov_enriched.get("provenance_status"),
                "source_provider": prov_enriched.get("source_provider"),
                "tradability_status": tradability_status,
            }
        )

    total_before_filter = len(rows)
    filt = filt_early
    applied_filter = "all"
    matching_keys: set[str] = set()
    if filt and filt != "all":
        status_mapping = {
            "passed": "Passed",
            "blocked": "Blocked",
            "watch": "Watch",
            "demo": "Demo candidate",
            "demo candidates": "Demo candidate",
        }
        # Demo Candidate filter uses trading_opportunity_state only (not semantic family)
        if filt in ("demo_candidate", "demo-candidate", "demo"):
            matched = [
                r
                for r in rows
                if str(r.get("trading_opportunity_state") or r.get("opportunity_state") or "")
                == "DEMO_CANDIDATE"
                or r.get("status") == "Demo candidate"
            ]
            applied_filter = "demo" if filt == "demo" else "demo_candidate"
        elif filt in _SEMANTIC_FILTERS:
            allowed = _SEMANTIC_FILTERS[filt]
            matched = [
                r
                for r in rows
                if str(r.get("semantic_signal_family") or r.get("semantic_family") or "")
                in allowed
            ]
            applied_filter = filt
        elif filt == "registered":
            matched = [
                r
                for r in rows
                if r.get("semantic_status")
                in ("Registered", "Classified", "Unresolved", "Needs Review")
            ]
            applied_filter = filt
        elif status_mapping.get(filt):
            # Passed/Blocked/Watch use actionability/status only
            matched = [r for r in rows if r.get("status") == status_mapping[filt]]
            applied_filter = filt
        else:
            matched = list(rows)

        matching_keys = {str(r.get("row_key")) for r in matched}
        if mode == "highlight":
            for r in rows:
                r["filter_match"] = str(r.get("row_key")) in matching_keys
                r["filter_highlight"] = r["filter_match"]
            # highlight mode still returns all rows (UI may dim non-matches)
            # but default is hide — so highlight is optional
        else:
            # DEFAULT: hide non-matching rows
            rows = matched
            for r in rows:
                r["filter_match"] = True
                r["filter_highlight"] = False
    else:
        for r in rows:
            r["filter_match"] = True
            r["filter_highlight"] = False
        matching_keys = {str(r.get("row_key")) for r in rows}

    shown_before_cap = len(rows)
    # Cap result set after API-level filter so UI limit still applies
    rows = rows[:limit]

    latest_ts = global_latest_ts
    for r in rows:
        if r.get("time") and (latest_ts is None or str(r["time"]) > str(latest_ts)):
            latest_ts = r["time"]
            break
    fresh = _freshness(global_latest_ts or latest_ts)

    stale_status = build_stale_price_status(
        applies_to="global_market",
        last_price_timestamp=global_latest_ts or latest_ts,
        source="live_market_feed",
        market_feed_active=bool(coins),
        blocks_demo_trade=False,
    )
    if not fresh.get("fresh") and coins:
        stale_status["label"] = (
            "Some candidates have stale prices. Market Snapshot Feed still updating. "
            f"Market feed active ({len(coins)} pairs)."
        )
        stale_status["blocks_demo_trade"] = False

    whale_scores = [float(r["whale_score"]) for r in rows if r.get("whale_score") is not None]
    avg_whale = round(sum(whale_scores) / len(whale_scores), 4) if whale_scores else None

    return {
        "latest_market_update": global_latest_ts or latest_ts,
        "freshness": fresh,
        "stale_price_status": stale_status,
        "live_pairs_count": len(coins),
        "passed_filter": transparency.get("passed_count", passed),
        "dropped_blocked": transparency.get("dropped_count", blocked),
        "average_whale_score": avg_whale,
        "demo_mode_badge": "LIVE DISABLED / DEMO ONLY",
        "rows": rows,
        "count": len(rows),
        "total_before_filter": total_before_filter,
        "shown_count": len(rows),
        "matching_count": shown_before_cap if applied_filter != "all" else total_before_filter,
        "filter_result_label": (
            f"Showing {len(rows)} of {total_before_filter}"
            if applied_filter != "all"
            else f"Showing {len(rows)}"
        ),
        "empty_state": "No matching rows." if not rows else None,
        "status_filter_applied": applied_filter,
        "filter_mode": mode,
        "filter_mode_default": DEFAULT_FILTER_MODE,
        "filter_hides_non_matching": mode == "hide",
        "filter_applied_at": "api",
        "filter_backend_authoritative": True,
        "filter_match_logic": "strict_enum_equality",
        "scan_at": transparency.get("scan_at"),
        "paper_demo_only": True,
        "wallet_configured": False,
        "live_trading_ready": False,
        "built_at_utc": _utc_now(),
        "reconciliation": {
            "strategy": "keyed_reconciliation",
            "row_key_preferred": "chain+pair_address",
            "row_key_fallback": ["chain+contract_address", "candidate_id"],
            "no_innerhtml_full_rebuild_required": True,
            "array_index_as_key": False,
        },
    }
