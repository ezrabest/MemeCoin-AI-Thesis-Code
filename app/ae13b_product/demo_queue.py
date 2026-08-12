"""Demo Trade Queue — paper/demo candidates from watchlist / live market.

Does not place live trades. Does not bypass risk controls.
Strategy lane: Manual Watchlist Scout.
"""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).parent.parent.parent / "data"
QUEUE_PATH = DATA_DIR / "demo_trade_queue.json"
_LOCK = threading.RLock()

STRATEGY_LANE = "Manual Watchlist Scout"
STRATEGY_LANE_ID = "manual_watchlist_scout"

#: AE13I Smoke Addendum (Part B): a GateKeeper evaluation older than this is
#: treated as stale and must be re-run before being trusted for a decision.
EVALUATION_STALE_SECONDS = 900
EVALUATION_STALE_MESSAGE = "Evaluation stale - click Evaluate Now."


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> list[dict[str, Any]]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not QUEUE_PATH.exists():
        return []
    try:
        with open(QUEUE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
    except Exception:
        return []
    return []


def _save(items: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)


def _attach_manual_cooldown(item: dict[str, Any]) -> dict[str, Any]:
    """Best-effort AE13I manual-cooldown badge fields for queue list/detail views."""
    try:
        from app.ae13b_product.reentry_blocks import get_manual_cooldown_fields

        cooldown = get_manual_cooldown_fields(
            pair_address=item.get("contract_or_pair_address") or item.get("pair"),
            chain=item.get("chain"),
            symbol=item.get("symbol"),
        )
        item["manual_cooldown_active"] = cooldown.get("manual_cooldown_active")
        item["manual_cooldown_expiry"] = cooldown.get("manual_cooldown_expiry")
        item["manual_cooldown_remaining_seconds"] = cooldown.get("manual_cooldown_remaining_seconds")
        item["manual_cooldown_reason"] = cooldown.get("manual_cooldown_reason")
        item["manual_cooldown_scope"] = cooldown.get("manual_cooldown_scope")
        item["reentry_blocked"] = cooldown.get("reentry_blocked")
    except Exception:
        item.setdefault("manual_cooldown_active", False)
    return item


def _attach_gatekeeper_freshness(item: dict[str, Any]) -> dict[str, Any]:
    """AE13I Smoke Addendum (Part B): attach GateKeeper evaluation freshness fields.

    Best-effort, never raises. Items evaluated before AE13I (no gate_result /
    gatekeeper_status persisted) are always treated as stale so the UI prompts
    a fresh re-evaluation rather than silently trusting an old balanced-preset
    snapshot.
    """
    gate_result = item.get("gate_result") if isinstance(item.get("gate_result"), dict) else None

    last_gatekeeper_evaluated_at = item.get("last_gatekeeper_evaluated_at")
    if not last_gatekeeper_evaluated_at and gate_result is not None:
        last_gatekeeper_evaluated_at = item.get("last_evaluated_at")

    gatekeeper_status = item.get("gatekeeper_status")
    if not gatekeeper_status and gate_result is not None:
        gatekeeper_status = "pass" if gate_result.get("passed") else "fail"

    tradability_status = item.get("tradability_status")
    freshness_gate_status = item.get("freshness_gate_status")
    provenance_status = item.get("provenance_status")
    address_role = item.get("address_role") or item.get("address_role_status")
    market_data_status = item.get("market_data_status")

    has_pre_ae13i_gap = bool(item.get("last_evaluated_at")) and gate_result is None

    age_seconds: float | None = None
    if last_gatekeeper_evaluated_at:
        try:
            ts = datetime.fromisoformat(str(last_gatekeeper_evaluated_at).replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_seconds = max(0.0, (datetime.now(timezone.utc) - ts).total_seconds())
        except (ValueError, TypeError):
            age_seconds = None

    evaluation_stale = (
        not last_gatekeeper_evaluated_at
        or age_seconds is None
        or age_seconds > EVALUATION_STALE_SECONDS
        or has_pre_ae13i_gap
    )

    item["last_gatekeeper_evaluated_at"] = last_gatekeeper_evaluated_at
    item["gatekeeper_status"] = gatekeeper_status
    item["tradability_status"] = tradability_status
    item["freshness_gate_status"] = freshness_gate_status
    item["provenance_status"] = provenance_status
    item["address_role"] = address_role
    item["market_data_status"] = market_data_status
    item["gatekeeper_evaluated"] = bool(gate_result is not None or gatekeeper_status)
    item["evaluation_stale"] = evaluation_stale
    item["evaluation_stale_reason"] = EVALUATION_STALE_MESSAGE if evaluation_stale else None
    item["evaluation_age_seconds"] = age_seconds
    return item


def list_demo_queue(*, include_disabled: bool = True) -> list[dict[str, Any]]:
    with _LOCK:
        items = _load()
        if include_disabled:
            return [_attach_gatekeeper_freshness(_attach_manual_cooldown(dict(i))) for i in items]
        return [
            _attach_gatekeeper_freshness(_attach_manual_cooldown(dict(i)))
            for i in items
            if i.get("enabled", True)
        ]


def get_queue_item(queue_id: str) -> dict[str, Any] | None:
    with _LOCK:
        for i in _load():
            if str(i.get("queue_id")) == str(queue_id):
                return _attach_gatekeeper_freshness(_attach_manual_cooldown(dict(i)))
    return None


def get_active_demo_risk_profile() -> dict[str, Any]:
    """Read the active demo bot preset for Demo Queue / Manual Watchlist Scout inheritance.

    Best-effort, never raises. Falls back to the balanced preset if the demo
    bot state file cannot be read (e.g. bot never started).
    """
    from app.ae13b_product.presets import get_preset

    preset_id = "balanced"
    try:
        from app.ae13b_product.demo_bot import STATE_PATH

        if STATE_PATH.is_file():
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("preset_id"):
                preset_id = str(data["preset_id"])
    except Exception:
        preset_id = "balanced"

    preset = get_preset(preset_id)
    return {
        "preset_id": preset["id"],
        "risk_mode": preset["id"],
        "max_open_positions": preset["max_open_positions"],
        "max_trades_per_hour": preset["max_trades_per_hour"],
        "max_notional_usd": preset["max_notional_usd"],
        "cooldown_seconds": preset["cooldown_seconds"],
        "inherits_active_bot_preset": True,
    }


def add_to_demo_queue(
    *,
    watchlist_id: str | None = None,
    symbol: str | None = None,
    pair: str | None = None,
    chain: str | None = None,
    contract_or_pair_address: str | None = None,
    source: str = "watchlist_manual",
    semantic_label: str | None = None,
    market_match_status: str | None = None,
    max_notional: float | None = None,
    risk_mode: str | None = None,
    user_hypothesis: str | None = None,
    user_evidence_note: str | None = None,
    user_claimed_social_mission: str | None = None,
) -> dict[str, Any]:
    addr = (contract_or_pair_address or pair or "").strip()
    sym = (symbol or "").strip()
    if not addr and not sym and not watchlist_id:
        raise ValueError("watchlist_id, symbol, or contract_or_pair_address required")

    # Preset propagation (AE13G): explicit risk_mode always wins. Otherwise inherit
    # the currently active demo bot preset so a Lotto-preset bot does not silently
    # queue Manual Watchlist Scout candidates under a mismatched "balanced" risk tier.
    explicit_risk_mode = risk_mode is not None
    profile = get_active_demo_risk_profile()
    effective_risk_mode = risk_mode if explicit_risk_mode else profile["risk_mode"]
    effective_max_notional = (
        float(max_notional) if max_notional is not None else float(profile["max_notional_usd"])
    )
    risk_profile_source = "queue_item_explicit" if explicit_risk_mode else "active_bot_preset"
    inherits_active_bot_preset = not explicit_risk_mode

    with _LOCK:
        items = _load()
        # Dedupe by watchlist_id or address
        existing = None
        if watchlist_id:
            existing = next(
                (i for i in items if str(i.get("watchlist_id")) == str(watchlist_id) and i.get("enabled", True)),
                None,
            )
        if existing is None and addr:
            addr_l = addr.lower()
            existing = next(
                (
                    i
                    for i in items
                    if str(i.get("contract_or_pair_address") or "").lower() == addr_l
                    and i.get("enabled", True)
                ),
                None,
            )
        if existing is not None:
            existing["enabled"] = True
            existing["updated_at"] = _utc_now()
            if semantic_label:
                existing["semantic_label"] = semantic_label
            if market_match_status:
                existing["market_match_status"] = market_match_status
            if user_hypothesis is not None:
                existing["user_hypothesis"] = user_hypothesis
            if user_evidence_note is not None:
                existing["user_evidence_note"] = user_evidence_note
            if user_claimed_social_mission is not None:
                existing["user_claimed_social_mission"] = user_claimed_social_mission
            if explicit_risk_mode:
                existing["risk_mode"] = effective_risk_mode
                existing["risk_profile_source"] = "queue_item_explicit"
                existing["inherits_active_bot_preset"] = False
            if max_notional is not None:
                existing["max_notional"] = float(max_notional)
            existing["eligibility_status"] = existing.get("eligibility_status") or "queued_for_evaluation"
            existing["demo_queue_status"] = "queued_for_evaluation"
            _save(items)
            return dict(existing)

        entry = {
            "queue_id": str(uuid.uuid4())[:12],
            "watchlist_id": watchlist_id,
            "symbol": sym or None,
            "pair": (pair or addr or None),
            "chain": (chain or "solana").lower(),
            "contract_or_pair_address": addr or None,
            "source": source or "watchlist_manual",
            "added_at": _utc_now(),
            "updated_at": _utc_now(),
            "enabled": True,
            "strategy_lane": STRATEGY_LANE,
            "strategy_lane_id": STRATEGY_LANE_ID,
            "max_notional": effective_max_notional,
            "risk_mode": effective_risk_mode,
            "risk_profile_source": risk_profile_source,
            "inherits_active_bot_preset": inherits_active_bot_preset,
            "semantic_label": semantic_label,
            "market_match_status": market_match_status or "waiting_for_market_match",
            "eligibility_status": "queued_for_evaluation",
            "demo_queue_status": "queued_for_evaluation",
            "last_evaluated_at": None,
            "last_decision": "WATCH",
            "last_blocker": None,
            "user_hypothesis": user_hypothesis,
            "user_evidence_note": user_evidence_note,
            "user_claimed_social_mission": user_claimed_social_mission,
            "not_live_approved": True,
            "paper_demo_only": True,
            "live_trading_implied": False,
        }
        items.append(entry)
        _save(items)
        return dict(entry)


def remove_from_demo_queue(queue_id: str) -> bool:
    with _LOCK:
        items = _load()
        before = len(items)
        items = [i for i in items if str(i.get("queue_id")) != str(queue_id)]
        if len(items) == before:
            return False
        _save(items)
        return True


def disable_queue_item(queue_id: str) -> dict[str, Any] | None:
    with _LOCK:
        items = _load()
        for i in items:
            if str(i.get("queue_id")) == str(queue_id):
                i["enabled"] = False
                i["demo_queue_status"] = "blocked"
                i["updated_at"] = _utc_now()
                _save(items)
                return dict(i)
        return None


def update_queue_evaluation(
    queue_id: str,
    *,
    last_decision: str,
    last_blocker: str | None = None,
    eligibility_status: str | None = None,
    semantic_label: str | None = None,
    market_match_status: str | None = None,
    demo_queue_status: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    with _LOCK:
        items = _load()
        for i in items:
            if str(i.get("queue_id")) == str(queue_id):
                i["last_evaluated_at"] = _utc_now()
                i["last_decision"] = last_decision
                i["last_blocker"] = last_blocker
                if eligibility_status:
                    i["eligibility_status"] = eligibility_status
                if semantic_label is not None:
                    i["semantic_label"] = semantic_label
                if market_match_status is not None:
                    i["market_match_status"] = market_match_status
                if demo_queue_status:
                    i["demo_queue_status"] = demo_queue_status
                if extra:
                    for k, v in extra.items():
                        if v is not None:
                            i[k] = v
                i["updated_at"] = _utc_now()
                i["paper_demo_only"] = True
                i["not_live_approved"] = True
                _save(items)
                return dict(i)
        return None


def evaluate_queue_item(queue_id: str) -> dict[str, Any]:
    """Safe paper/demo evaluation — does not force a trade."""
    item = get_queue_item(queue_id)
    if not item:
        return {
            "ok": False,
            "error": "queue_item_not_found",
            "paper_demo_only": True,
            "not_live_approved": True,
        }

    from app.ae13b_product.contract_resolver import resolve_identity
    from app.ae13b_product.demo_risk_guard import evaluate_demo_risk_guard
    from app.ae13b_product.presets import get_preset

    resolution = resolve_identity(
        chain=item.get("chain"),
        contract_or_pair_address=item.get("contract_or_pair_address") or item.get("pair"),
        symbol=item.get("symbol"),
        allow_external=False,
    )

    # AE13I: manual reentry cooldown precheck — runs before the risk guard and
    # short-circuits evaluation entirely while a manual-close cooldown is active.
    from app.ae13b_product.reentry_blocks import get_manual_cooldown_fields

    cooldown = get_manual_cooldown_fields(
        pair_address=item.get("contract_or_pair_address")
        or resolution.get("matched_pair_address")
        or item.get("pair"),
        chain=item.get("chain") or resolution.get("matched_chain"),
        token_contract=resolution.get("matched_token_contract_address"),
        token_mint=resolution.get("matched_token_mint_address"),
        symbol=item.get("symbol") or resolution.get("matched_symbol"),
    )
    if cooldown.get("manual_cooldown_active") or cooldown.get("reentry_blocked"):
        # AE13I Smoke Addendum (Part B): the cooldown precheck runs before
        # GateKeeper/RiskGuard by design, so no gate evaluation happened this
        # cycle. gatekeeper_evaluated stays whatever it was; do not claim a
        # fresh gate evaluation occurred here.
        updated = update_queue_evaluation(
            queue_id,
            last_decision="BLOCKED_MANUAL_REENTRY_COOLDOWN",
            last_blocker="manual_reentry_block",
            eligibility_status="blocked",
            demo_queue_status="blocked_by_manual_reentry_cooldown",
            extra={
                "resolution": resolution,
                "manual_cooldown_active": cooldown.get("manual_cooldown_active"),
                "manual_cooldown_expiry": cooldown.get("manual_cooldown_expiry"),
                "manual_cooldown_remaining_seconds": cooldown.get("manual_cooldown_remaining_seconds"),
                "manual_cooldown_reason": cooldown.get("manual_cooldown_reason"),
                "manual_cooldown_scope": cooldown.get("manual_cooldown_scope"),
                "can_demo_trade": False,
            },
        )
        return {
            "ok": True,
            "queue_item": updated,
            "resolution": resolution,
            "decision": "BLOCKED_MANUAL_REENTRY_COOLDOWN",
            "rejection_code": "MANUAL_REENTRY_BLOCK_ACTIVE",
            "blocking_guards": ["manual_reentry_block"],
            "reason": "Manual re-entry cooldown active for this pair.",
            "last_decision": "BLOCKED_MANUAL_REENTRY_COOLDOWN",
            "last_blocker": "manual_reentry_block",
            "selected": False,
            "blocked": True,
            "can_demo_trade": False,
            "manual_cooldown_active": True,
            "manual_cooldown_expiry": cooldown.get("manual_cooldown_expiry")
            or cooldown.get("cooldown_expires_at_utc"),
            "manual_cooldown_remaining_seconds": cooldown.get("manual_cooldown_remaining_seconds"),
            "manual_cooldown_reason": cooldown.get("manual_cooldown_reason"),
            "manual_cooldown_scope": cooldown.get("manual_cooldown_scope"),
            "strategy_lane": STRATEGY_LANE,
            "strategy_lane_id": STRATEGY_LANE_ID,
            "next_action": "Wait for the re-entry cooldown to expire before demo evaluation.",
            "next_possible_action": "Wait for the re-entry cooldown to expire before demo evaluation.",
            "paper_demo_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
            "live_trading_implied": False,
        }

    # AE13I: MarketDataGateKeeper precheck — runs before the risk guard. A gate
    # failure is the primary blocked decision; the risk guard only evaluates
    # portfolio limits once the market data gate has passed.
    from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate
    from app.ae13b_product.clean_forward_bridge import (
        build_clean_forward_gatekeeper_candidate,
        find_matching_clean_forward_row,
        is_clean_forward_queue_item,
    )

    clean_forward_bridge_used = False
    legacy_market_snapshots_used = False
    cf_candidate: dict[str, Any] | None = None
    eval_price = resolution.get("matched_price")
    eval_price_ts = resolution.get("matched_price_ts")
    eval_liquidity = resolution.get("matched_liquidity")
    eval_source = resolution.get("resolution_source") or "demo_queue"

    if is_clean_forward_queue_item(item):
        # AE14: Clean Forward rows carry price/liquidity/provenance. Do not use
        # resolution.matched_price or market_snapshots for these entries.
        from app.ae13b_product.ae14_candidate_source_policy import (
            AE14_CANDIDATE_SOURCE_POLICY,
            CANDIDATE_SOURCE as AE14_CANDIDATE_SOURCE,
            is_ae14_closure_mode,
            is_synthetic_or_fixture_row,
            is_valid_ae14_clean_forward_row,
        )
        from app.ae13b_product.clean_forward_market_feed import get_cached_clean_forward_rows

        cf_row = find_matching_clean_forward_row(
            get_cached_clean_forward_rows(),
            chain=item.get("chain") or resolution.get("matched_chain"),
            pair_address=item.get("contract_or_pair_address")
            or resolution.get("matched_pair_address")
            or item.get("pair"),
        )
        if cf_row is None:
            updated = update_queue_evaluation(
                queue_id,
                last_decision="BLOCKED",
                last_blocker="CLEAN_FORWARD_ROW_NOT_FOUND",
                eligibility_status="blocked",
                demo_queue_status="blocked_by_clean_forward_row_not_found",
                market_match_status=item.get("market_match_status") or "provider_pair_verified",
                extra={
                    "resolution": resolution,
                    "can_demo_trade": False,
                    "clean_forward_bridge_used": True,
                    "legacy_market_snapshots_used": False,
                    "old_watchlist_candidates_used": False,
                    "local_db_candidate_universe_used": False,
                    "candidate_source": AE14_CANDIDATE_SOURCE,
                    "ae14_candidate_source_policy": AE14_CANDIDATE_SOURCE_POLICY,
                },
            )
            return {
                "ok": True,
                "queue_item": updated,
                "resolution": resolution,
                "decision": "BLOCKED",
                "rejection_code": "CLEAN_FORWARD_ROW_NOT_FOUND",
                "blocking_guards": ["clean_forward_row_lookup"],
                "reason": "Matching Clean Forward Market Feed row not found in process cache.",
                "last_decision": "BLOCKED",
                "last_blocker": "CLEAN_FORWARD_ROW_NOT_FOUND",
                "selected": False,
                "blocked": True,
                "can_demo_trade": False,
                "clean_forward_bridge_used": True,
                "legacy_market_snapshots_used": False,
                "old_watchlist_candidates_used": False,
                "local_db_candidate_universe_used": False,
                "candidate_source": AE14_CANDIDATE_SOURCE,
                "ae14_candidate_source_policy": AE14_CANDIDATE_SOURCE_POLICY,
                "strategy_lane": STRATEGY_LANE,
                "strategy_lane_id": STRATEGY_LANE_ID,
                "next_action": "Refresh Clean Forward Market Feed, then Evaluate again.",
                "next_possible_action": "Refresh Clean Forward Market Feed, then Evaluate again.",
                "paper_demo_only": True,
                "not_live_approved": True,
                "not_profitability_evidence": True,
                "live_trading_implied": False,
            }

        if is_ae14_closure_mode() and (
            is_synthetic_or_fixture_row(cf_row) or not is_valid_ae14_clean_forward_row(cf_row)
        ):
            updated = update_queue_evaluation(
                queue_id,
                last_decision="BLOCKED",
                last_blocker="SYNTHETIC_OR_INVALID_CLEAN_FORWARD_ROW",
                eligibility_status="blocked",
                demo_queue_status="blocked_by_ae14_source_policy",
                market_match_status="provider_pair_verified",
                extra={
                    "resolution": resolution,
                    "can_demo_trade": False,
                    "clean_forward_bridge_used": True,
                    "legacy_market_snapshots_used": False,
                    "candidate_source": AE14_CANDIDATE_SOURCE,
                    "ae14_candidate_source_policy": AE14_CANDIDATE_SOURCE_POLICY,
                    "synthetic_fixture_used": is_synthetic_or_fixture_row(cf_row),
                },
            )
            return {
                "ok": True,
                "queue_item": updated,
                "resolution": resolution,
                "decision": "BLOCKED",
                "rejection_code": "SYNTHETIC_OR_INVALID_CLEAN_FORWARD_ROW",
                "blocking_guards": ["ae14_candidate_source_policy"],
                "reason": "AE14 rejects synthetic/fixture/invalid Clean Forward rows.",
                "last_decision": "BLOCKED",
                "last_blocker": "SYNTHETIC_OR_INVALID_CLEAN_FORWARD_ROW",
                "selected": False,
                "blocked": True,
                "can_demo_trade": False,
                "clean_forward_bridge_used": True,
                "legacy_market_snapshots_used": False,
                "candidate_source": AE14_CANDIDATE_SOURCE,
                "ae14_candidate_source_policy": AE14_CANDIDATE_SOURCE_POLICY,
                "strategy_lane": STRATEGY_LANE,
                "strategy_lane_id": STRATEGY_LANE_ID,
                "paper_demo_only": True,
                "not_live_approved": True,
                "not_profitability_evidence": True,
                "live_trading_implied": False,
            }

        bridge = build_clean_forward_gatekeeper_candidate(cf_row)
        clean_forward_bridge_used = True
        if not bridge.get("ok") or not isinstance(bridge.get("candidate"), dict):
            reasons = list(bridge.get("block_reasons") or [bridge.get("block_reason") or "bridge_rejected"])
            primary = str(bridge.get("block_reason") or reasons[0])
            updated = update_queue_evaluation(
                queue_id,
                last_decision="BLOCKED",
                last_blocker=primary,
                eligibility_status="blocked",
                demo_queue_status="blocked_by_clean_forward_bridge",
                market_match_status="provider_pair_verified",
                extra={
                    "resolution": resolution,
                    "clean_forward_bridge": bridge,
                    "can_demo_trade": False,
                    "clean_forward_bridge_used": True,
                    "legacy_market_snapshots_used": False,
                    "candidate_source": AE14_CANDIDATE_SOURCE,
                    "ae14_candidate_source_policy": AE14_CANDIDATE_SOURCE_POLICY,
                },
            )
            return {
                "ok": True,
                "queue_item": updated,
                "resolution": resolution,
                "clean_forward_bridge": bridge,
                "decision": "BLOCKED",
                "rejection_code": primary,
                "blocking_guards": ["clean_forward_bridge"],
                "reason": "; ".join(str(r) for r in reasons),
                "last_decision": "BLOCKED",
                "last_blocker": primary,
                "selected": False,
                "blocked": True,
                "can_demo_trade": False,
                "clean_forward_bridge_used": True,
                "legacy_market_snapshots_used": False,
                "candidate_source": AE14_CANDIDATE_SOURCE,
                "ae14_candidate_source_policy": AE14_CANDIDATE_SOURCE_POLICY,
                "strategy_lane": STRATEGY_LANE,
                "strategy_lane_id": STRATEGY_LANE_ID,
                "next_action": "Clean Forward row failed bridge hard gates — refresh or pick another row.",
                "next_possible_action": "Clean Forward row failed bridge hard gates — refresh or pick another row.",
                "paper_demo_only": True,
                "not_live_approved": True,
                "not_profitability_evidence": True,
                "live_trading_implied": False,
            }

        cf_candidate = dict(bridge["candidate"])
        cf_candidate["ae14_candidate_source_policy"] = AE14_CANDIDATE_SOURCE_POLICY
        gate_row = cf_candidate
        eval_price = cf_candidate.get("latest_price")
        eval_price_ts = cf_candidate.get("price_updated_at")
        eval_liquidity = cf_candidate.get("latest_liquidity")
        eval_source = AE14_CANDIDATE_SOURCE
    else:
        from app.ae13b_product.ae14_candidate_source_policy import requires_clean_forward_only

        if requires_clean_forward_only():
            updated = update_queue_evaluation(
                queue_id,
                last_decision="BLOCKED",
                last_blocker="AE14_REQUIRES_CLEAN_FORWARD_MARKET_FEED",
                eligibility_status="blocked",
                demo_queue_status="blocked_by_ae14_source_policy",
                extra={
                    "resolution": resolution,
                    "can_demo_trade": False,
                    "clean_forward_bridge_used": False,
                    "legacy_market_snapshots_used": False,
                    "old_watchlist_candidates_used": False,
                    "local_db_candidate_universe_used": False,
                    "candidate_source": "clean_forward_market_feed",
                    "ae14_candidate_source_policy": "clean_forward_market_feed_only",
                },
            )
            return {
                "ok": True,
                "queue_item": updated,
                "resolution": resolution,
                "decision": "BLOCKED",
                "rejection_code": "AE14_REQUIRES_CLEAN_FORWARD_MARKET_FEED",
                "blocking_guards": ["ae14_candidate_source_policy"],
                "reason": "AE14 closure mode rejects non-Clean-Forward queue evaluation sources.",
                "last_decision": "BLOCKED",
                "last_blocker": "AE14_REQUIRES_CLEAN_FORWARD_MARKET_FEED",
                "selected": False,
                "blocked": True,
                "can_demo_trade": False,
                "clean_forward_bridge_used": False,
                "legacy_market_snapshots_used": False,
                "candidate_source": "clean_forward_market_feed",
                "ae14_candidate_source_policy": "clean_forward_market_feed_only",
                "strategy_lane": STRATEGY_LANE,
                "strategy_lane_id": STRATEGY_LANE_ID,
                "paper_demo_only": True,
                "not_live_approved": True,
                "not_profitability_evidence": True,
                "live_trading_implied": False,
            }
        gate_row = {
            "chain": item.get("chain") or resolution.get("matched_chain"),
            "symbol": item.get("symbol") or resolution.get("matched_symbol"),
            "pair_address": item.get("contract_or_pair_address")
            or resolution.get("matched_pair_address")
            or item.get("pair"),
            "latest_price": resolution.get("matched_price"),
            "price_updated_at": resolution.get("matched_price_ts"),
            "latest_liquidity": resolution.get("matched_liquidity"),
            "liquidity_updated_at": resolution.get("matched_price_ts"),
            "source_provider": resolution.get("resolution_source") or "demo_queue",
        }
    # AE13I fix: stagnant_price_guard now returns passed=True (with
    # momentum_evidence="unknown_insufficient_delta_fields") when no delta
    # fields are present, rather than hard-blocking on missing data, so the
    # guard can safely run here (skip_stagnant=False) even though identity
    # resolution does not carry 1h/4h activity deltas.
    gate = validate_market_data_gate(gate_row, for_open=True, skip_stagnant=False)
    gate_checked_at = gate.get("checked_at_utc") or _utc_now()
    if not gate.get("passed"):
        updated = update_queue_evaluation(
            queue_id,
            last_decision="BLOCKED",
            last_blocker=(gate.get("rejection_reasons") or ["Blocked by market data gate"])[0],
            eligibility_status="blocked",
            demo_queue_status="blocked_by_market_data_gate",
            extra={
                "resolution": resolution,
                "gate_result": gate,
                "tradability_status": gate.get("tradability_status"),
                "freshness_gate_status": gate.get("freshness_gate_status"),
                "provenance_status": gate.get("provenance_status"),
                "address_role": gate.get("address_role_status"),
                "address_role_status": gate.get("address_role_status"),
                "market_data_status": gate.get("market_data_status"),
                "last_gatekeeper_evaluated_at": gate_checked_at,
                "gatekeeper_status": "pass" if gate.get("passed") else "fail",
                "gatekeeper_evaluated": True,
                "can_demo_trade": False,
                "clean_forward_bridge_used": clean_forward_bridge_used,
                "legacy_market_snapshots_used": legacy_market_snapshots_used,
                "candidate_source": (
                    "clean_forward_market_feed" if clean_forward_bridge_used else None
                ),
                "clean_forward_candidate": cf_candidate,
            },
        )
        return {
            "ok": True,
            "queue_item": updated,
            "resolution": resolution,
            "gate_result": gate,
            "decision": "BLOCKED",
            "rejection_code": gate.get("rejection_code"),
            "blocking_guards": gate.get("blocking_guards"),
            "reason": "; ".join(gate.get("rejection_reasons") or []) or "Blocked by market data gate",
            "last_decision": "BLOCKED",
            "last_blocker": gate.get("primary_blocker"),
            "selected": False,
            "blocked": True,
            "can_demo_trade": False,
            "tradability_status": gate.get("tradability_status"),
            "freshness_gate_status": gate.get("freshness_gate_status"),
            "address_role_status": gate.get("address_role_status"),
            "provenance_status": gate.get("provenance_status"),
            "market_data_status": gate.get("market_data_status"),
            "gatekeeper_status": "fail",
            "gatekeeper_evaluated": True,
            "last_gatekeeper_evaluated_at": gate_checked_at,
            "manual_cooldown_active": cooldown.get("manual_cooldown_active"),
            "manual_cooldown_expiry": cooldown.get("manual_cooldown_expiry"),
            "clean_forward_bridge_used": clean_forward_bridge_used,
            "legacy_market_snapshots_used": legacy_market_snapshots_used,
            "candidate_source": (
                "clean_forward_market_feed" if clean_forward_bridge_used else None
            ),
            "ae14_candidate_source_policy": (
                "clean_forward_market_feed_only" if clean_forward_bridge_used else None
            ),
            "strategy_lane": STRATEGY_LANE,
            "strategy_lane_id": STRATEGY_LANE_ID,
            "next_action": "Blocked by market data gate - wait for fresh data or resolve identity.",
            "next_possible_action": "Blocked by market data gate - wait for fresh data or resolve identity.",
            "paper_demo_only": True,
            "not_live_approved": True,
            "not_profitability_evidence": True,
            "live_trading_implied": False,
        }

    market_status = "waiting_for_market_match"
    if clean_forward_bridge_used:
        market_status = "provider_pair_verified"
    else:
        res_status = resolution.get("resolution_status")
        if res_status in ("local_match", "matched_live_market"):
            market_status = "seen_in_live_market"
        elif res_status in (
            "matched_registry",
            "matched_static_snapshot",
            "matched_db",
            "user_entered_identity",
        ):
            if res_status == "user_entered_identity":
                market_status = "waiting_for_market_match"
            else:
                market_status = res_status

    # AE13G: identity store / user-supplied social claim fields flow into the
    # candidate even when there is no market match yet. A user hypothesis alone
    # must never become SOCIAL_CONFIRMED — only SOCIAL_CANDIDATE_NEEDS_VERIFICATION.
    semantic = item.get("semantic_label")
    evidence = None
    try:
        from app.ae13_semantic.runtime_registry import get_semantic_registry

        rec = get_semantic_registry().observe_candidate(
            {
                "symbol": item.get("symbol") or resolution.get("matched_symbol"),
                "name": resolution.get("matched_name"),
                "chain": item.get("chain") or resolution.get("matched_chain"),
                "pair_address": item.get("contract_or_pair_address")
                or resolution.get("matched_pair_address"),
                "coin_id": f"demo_queue:{item.get('queue_id')}",
                "user_expected_category": item.get("user_hypothesis"),
                "user_evidence_note": item.get("user_evidence_note"),
                "user_claimed_social_mission": item.get("user_claimed_social_mission"),
                "force_reclassify": True,
                "pinned": True,
            }
        )
        semantic = rec.get("semantic_signal_family") or semantic
        evidence = rec.get("evidence_summary")

        note_l = str(item.get("user_evidence_note") or "").lower()
        mission_l = str(item.get("user_claimed_social_mission") or "").lower()
        hyp_l = str(item.get("user_hypothesis") or "").lower()
        social_claim = (
            "social" in hyp_l
            or "charit" in note_l
            or "educat" in note_l
            or "charit" in mission_l
            or "educat" in mission_l
            or "social" in mission_l
        )
        if semantic == "SOCIAL_CONFIRMED" and "social_confirmed" not in str(evidence or "").lower():
            semantic = "SOCIAL_CANDIDATE_NEEDS_VERIFICATION"
            evidence = (
                "user_supplied_social_claim_requires_validation: SOCIAL_CONFIRMED is not "
                "auto-assigned from user hypothesis alone; requires source validation."
            )
        elif social_claim and semantic in (
            "UNKNOWN_INSUFFICIENT_EVIDENCE",
            "UNKNOWN_UNRESOLVED",
            "NEEDS_REVIEW",
            None,
            "",
        ):
            semantic = "SOCIAL_CANDIDATE_NEEDS_VERIFICATION"
            evidence = (
                "user_supplied_social_claim_requires_validation: User-provided "
                "social/educational claim; requires source validation. Works without a "
                "market match."
            )
    except Exception:
        evidence = "registry_unavailable"

    # Risk guard preview (no order) — inherits the active demo bot preset unless the
    # queue item carries an explicit risk_mode (AE13G preset propagation).
    from app.ae13b_product.stale_price_status import row_price_freshness

    # AE13I Smoke Addendum (Part B): once an item inherits the active bot
    # preset (no explicit risk_mode ever set on the item), always re-read the
    # CURRENT active preset here rather than trusting whatever preset id was
    # snapshotted onto the item at add-time. Otherwise a bot preset switch
    # (e.g. balanced -> aggressive) would silently leave old queue items
    # evaluating under a stale "balanced" risk_mode forever.
    inherits_active_bot_preset = bool(item.get("inherits_active_bot_preset", True))
    active_profile = get_active_demo_risk_profile()
    if inherits_active_bot_preset:
        item_risk_mode = str(active_profile["risk_mode"])
    else:
        item_risk_mode = str(item.get("risk_mode") or active_profile["risk_mode"])
    risk_preset = get_preset(item_risk_mode)
    bot_state = {
        "preset_id": risk_preset["id"],
        "risk_mode": risk_preset["id"],
        "max_open_positions": risk_preset["max_open_positions"],
        "max_trades_per_hour": risk_preset["max_trades_per_hour"],
        "max_notional_usd": item.get("max_notional") or risk_preset["max_notional_usd"],
        "cooldown_seconds": risk_preset["cooldown_seconds"],
    }
    risk = evaluate_demo_risk_guard(
        requested_notional=float(item.get("max_notional") or 75),
        pair_address=item.get("contract_or_pair_address")
        or (cf_candidate or {}).get("pair_address")
        or resolution.get("matched_pair_address"),
        symbol=item.get("symbol")
        or (cf_candidate or {}).get("symbol")
        or resolution.get("matched_symbol"),
        chain=item.get("chain") or (cf_candidate or {}).get("chain"),
        price=eval_price,
        price_timestamp=eval_price_ts,
        liquidity=eval_liquidity,
        strategy_lane=STRATEGY_LANE_ID,
        risk_mode=item_risk_mode,
        bot_state=bot_state,
    )

    stale = row_price_freshness(
        price=eval_price,
        timestamp=eval_price_ts,
        symbol=item.get("symbol")
        or (cf_candidate or {}).get("symbol")
        or resolution.get("matched_symbol"),
        pair=item.get("contract_or_pair_address")
        or (cf_candidate or {}).get("pair_address")
        or item.get("pair"),
        source=eval_source,
    )

    blockers: list[str] = []
    decision = "WATCH"
    eligibility = "queued_for_evaluation"
    dq_status = "queued_for_evaluation"

    price = eval_price
    if price is None or float(price or 0) <= 0:
        blockers.append(
            "Queued, but cannot be evaluated for paper trade until a current price is available."
        )
        decision = "NOT_ENOUGH_DATA"
        eligibility = "awaiting_market_data"
        dq_status = "blocked_by_missing_price"
    elif market_status == "waiting_for_market_match":
        blockers.append("awaiting_market_data")
        decision = "NOT_ENOUGH_DATA"
        eligibility = "awaiting_market_data"
        dq_status = "awaiting_market_data"
    elif stale.get("is_stale"):
        blockers.append(
            f"Price is stale: last update {stale.get('price_age_label')}, "
            f"limit {stale.get('freshness_limit_label')}."
        )
        decision = "BLOCKED"
        eligibility = "blocked"
        dq_status = "blocked_by_stale_price"
    elif not risk["risk_guard_passed"]:
        blockers.append(risk["risk_guard_reason"])
        decision = "BLOCKED"
        eligibility = "blocked"
        if "stale" in str(risk["risk_guard_reason"]).lower():
            dq_status = "blocked_by_stale_price"
        else:
            dq_status = "blocked_by_risk"
    elif semantic in (
        "UNKNOWN_INSUFFICIENT_EVIDENCE",
        "NEEDS_REVIEW",
        "SOCIAL_CANDIDATE_NEEDS_VERIFICATION",
        None,
        "",
    ):
        decision = "WATCH"
        eligibility = "queued_for_evaluation"
        dq_status = "blocked_by_semantic_uncertainty"
        blockers.append("semantic_needs_review_or_insufficient_evidence")
    else:
        decision = "DEMO_CANDIDATE"
        eligibility = "eligible_for_demo"
        dq_status = "eligible_for_demo"

    risk_profile_source = item.get("risk_profile_source") or "active_bot_preset"
    risk_profile_disclosure: str | None = None
    if not inherits_active_bot_preset:
        if str(item_risk_mode) != str(active_profile["risk_mode"]):
            risk_profile_disclosure = (
                f"Manual Watchlist Scout risk profile: {item_risk_mode}, "
                f"independent from active {active_profile['risk_mode']} bot preset."
            )

    updated = update_queue_evaluation(
        queue_id,
        last_decision=decision,
        last_blocker="; ".join(blockers) if blockers else None,
        eligibility_status=eligibility,
        semantic_label=semantic,
        market_match_status=market_status,
        demo_queue_status=dq_status,
        extra={
            "resolution": resolution,
            "risk_guard": risk,
            "stale_price_status": stale,
            "evidence_summary": evidence,
            "gate_result": gate,
            "tradability_status": gate.get("tradability_status"),
            "freshness_gate_status": gate.get("freshness_gate_status"),
            "address_role_status": gate.get("address_role_status"),
            "address_role": gate.get("address_role_status"),
            "provenance_status": gate.get("provenance_status"),
            "market_data_status": gate.get("market_data_status"),
            "last_gatekeeper_evaluated_at": gate_checked_at,
            "gatekeeper_status": "pass" if gate.get("passed") else "fail",
            "gatekeeper_evaluated": True,
            "risk_mode": item_risk_mode,
            "inherits_active_bot_preset": inherits_active_bot_preset,
            "manual_cooldown_active": cooldown.get("manual_cooldown_active"),
            "manual_cooldown_expiry": cooldown.get("manual_cooldown_expiry"),
            "manual_cooldown_scope": cooldown.get("manual_cooldown_scope"),
            "can_demo_trade": decision in ("DEMO_CANDIDATE", "PAPER_BUY_ALLOWED")
            and risk["risk_guard_passed"],
            "clean_forward_bridge_used": clean_forward_bridge_used,
            "legacy_market_snapshots_used": legacy_market_snapshots_used,
            "candidate_source": (
                "clean_forward_market_feed" if clean_forward_bridge_used else None
            ),
            "clean_forward_candidate": cf_candidate,
        },
    )

    return {
        "ok": True,
        "queue_item": updated,
        "resolution": resolution,
        "risk_guard": risk,
        "stale_price_status": stale,
        "decision": decision,
        "reason": "; ".join(blockers) if blockers else "eligible_for_demo_evaluation",
        "market_data_availability": market_status,
        "latest_price_status": "missing"
        if price is None
        else ("stale" if stale.get("is_stale") else "fresh"),
        "semantic_status": semantic,
        "risk_guard_result": risk,
        "gate_result": gate,
        "tradability_status": gate.get("tradability_status"),
        "freshness_gate_status": gate.get("freshness_gate_status"),
        "address_role_status": gate.get("address_role_status"),
        "provenance_status": gate.get("provenance_status"),
        "market_data_status": gate.get("market_data_status"),
        "gatekeeper_status": "pass" if gate.get("passed") else "fail",
        "gatekeeper_evaluated": True,
        "last_gatekeeper_evaluated_at": gate_checked_at,
        "evaluation_stale": False,
        "manual_cooldown_active": cooldown.get("manual_cooldown_active"),
        "manual_cooldown_expiry": cooldown.get("manual_cooldown_expiry"),
        "manual_cooldown_remaining_seconds": cooldown.get("manual_cooldown_remaining_seconds"),
        "manual_cooldown_scope": cooldown.get("manual_cooldown_scope"),
        "last_decision": decision,
        "last_blocker": "; ".join(blockers) if blockers else None,
        "selected": decision in ("DEMO_CANDIDATE", "PAPER_BUY_ALLOWED"),
        "blocked": decision == "BLOCKED",
        "semantic_label": semantic,
        "market_match_status": market_status,
        "data_status": market_status,
        "price_freshness": stale.get("label")
        if stale
        else ("ok" if eval_price else "missing"),
        "risk_status": "passed" if risk["risk_guard_passed"] else "blocked",
        "risk_mode": item_risk_mode,
        "risk_profile_source": risk_profile_source,
        "inherits_active_bot_preset": inherits_active_bot_preset,
        "risk_profile_disclosure": risk_profile_disclosure,
        "clean_forward_bridge_used": clean_forward_bridge_used,
        "legacy_market_snapshots_used": legacy_market_snapshots_used,
        "candidate_source": (
            "clean_forward_market_feed" if clean_forward_bridge_used else None
        ),
        "ae14_candidate_source_policy": (
            "clean_forward_market_feed_only" if clean_forward_bridge_used else None
        ),
        "strategy_lane": STRATEGY_LANE,
        "strategy_lane_id": STRATEGY_LANE_ID,
        "next_action": (
            "Await market match / add evidence"
            if decision == "NOT_ENOUGH_DATA"
            else (
                "Blocked by risk guard - adjust size or wait"
                if decision == "BLOCKED"
                else (
                    "Eligible for bounded paper/demo evaluation next bot cycle"
                    if decision == "DEMO_CANDIDATE"
                    else "Continue watching - paper/demo only"
                )
            )
        ),
        "next_possible_action": (
            "Await market match / add evidence"
            if decision == "NOT_ENOUGH_DATA"
            else (
                "Blocked by risk guard - adjust size or wait"
                if decision == "BLOCKED"
                else (
                    "Eligible for bounded paper/demo evaluation next bot cycle"
                    if decision == "DEMO_CANDIDATE"
                    else "Continue watching - paper/demo only"
                )
            )
        ),
        "paper_demo_only": True,
        "not_live_approved": True,
        "not_profitability_evidence": True,
        "live_trading_implied": False,
    }
