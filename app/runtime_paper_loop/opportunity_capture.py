"""Opportunity capture audit — every candidate persisted, no-lookahead outcomes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from app.paper_trading.price_oracle import DemoPriceOracle, parse_ts
from app.runtime_paper_loop.decision_policy import PolicyDecision
from app.runtime_paper_loop.types import OpportunityCaptureRecord, utc_now_iso

HORIZON_SECONDS = {
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "6h": 6 * 60 * 60,
    "24h": 24 * 60 * 60,
}

MISSED_WINNER_THRESHOLD = 0.10  # 10% forward return


def build_opportunity_capture_record(
    *,
    decision: dict[str, Any] | None,
    context: dict[str, Any] | None,
    audit: dict[str, Any] | None,
    traceability: dict[str, Any],
    policy: PolicyDecision,
    price_result: dict[str, Any],
    loop_run_id: str,
    loop_iteration: int,
    paper_order_id: str | None = None,
    position_id: str | None = None,
    paper_action_taken: str | None = None,
) -> OpportunityCaptureRecord:
    identity = (decision or {}).get("candidate_identity") or {}
    market = (decision or {}).get("market_context") or {}
    liq_ctx = (context or {}).get("liquidity_activity_context") or {}

    # AE12-SentimentFix: additive dual-axis taxonomy (does not rewrite legacy cluster sources)
    from app.ae12_sentimentfix.dual_axis_mapper import map_dual_axis

    seed_row = {
        "pair_address": identity.get("pair_address") or (context or {}).get("pair_address"),
        "cluster_label": (
            (decision or {}).get("cluster_label")
            or identity.get("cluster_label")
            or market.get("cluster_label")
        ),
        "exploration_decision": policy.exploration_decision,
        "strict_shadow_decision": policy.strict_shadow_decision,
        "paper_action_taken": paper_action_taken,
        "reason_for_no_trade": policy.reason_for_no_trade,
        "blocked_by_ae9": policy.blocked_by_ae9,
        "stale_price": policy.stale_price,
        "missing_context": policy.missing_context,
        "max_open_positions_hit": policy.max_open_positions_hit,
        "cooldown_active": policy.cooldown_active,
        "duplicate_active_pair": policy.duplicate_active_pair,
        "llm_verdict": traceability.get("audit_verdict"),
        "ae8_context_status": "PRESENT" if context else "MISSING",
    }
    # Include light context text clues when present (local only; no LLM call)
    if isinstance(context, dict):
        for key in ("narrative", "summary", "source_family", "context_family"):
            if context.get(key) is not None:
                seed_row[key] = context.get(key)
    dual = map_dual_axis(seed_row)

    return OpportunityCaptureRecord(
        loop_run_id=loop_run_id,
        loop_iteration=loop_iteration,
        source_decision_id=traceability.get("source_decision_id"),
        candidate_id=traceability.get("candidate_id"),
        pair_address=identity.get("pair_address") or (context or {}).get("pair_address"),
        first_seen_timestamp=(decision or {}).get("created_at_utc") or utc_now_iso(),
        source_context_record_id=traceability.get("source_context_record_id"),
        source_llm_audit_record_id=traceability.get("source_llm_audit_record_id"),
        price_at_first_seen=price_result.get("price") or market.get("price"),
        liquidity_at_first_seen=float(market.get("liquidity") or liq_ctx.get("liquidity_usd") or 0),
        whale_score_at_first_seen=float(market.get("whale_score") or 0),
        volume_at_first_seen=float(market.get("volume_h24") or liq_ctx.get("volume_24h") or 0),
        ae6_decision_status=(decision or {}).get("decision_status"),
        ae8_context_status="PRESENT" if context else "MISSING",
        ae9_audit_verdict=traceability.get("audit_verdict"),
        ae9_audit_blockers=list(traceability.get("audit_blockers") or []),
        paper_action_taken=paper_action_taken,
        reason_for_no_trade=policy.reason_for_no_trade,
        strict_shadow_decision=policy.strict_shadow_decision,
        exploration_decision=policy.exploration_decision,
        paper_order_id=paper_order_id,
        position_id=position_id,
        blocked_by_ae9=policy.blocked_by_ae9,
        stale_price=policy.stale_price,
        missing_context=policy.missing_context,
        max_open_positions_hit=policy.max_open_positions_hit,
        cooldown_active=policy.cooldown_active,
        duplicate_active_pair=policy.duplicate_active_pair,
        missing_identity=policy.missing_identity,
        paper_policy_prevented=policy.paper_policy_prevented,
        duplicate_reason=policy.duplicate_reason,
        semantic_signal_family=dual["semantic_signal_family"],
        semantic_signal_source=dual["semantic_signal_source"],
        semantic_signal_confidence=float(dual["semantic_signal_confidence"]),
        semantic_signal_reason=dual["semantic_signal_reason"],
        trading_opportunity_state=dual["trading_opportunity_state"],
        trading_state_source=dual["trading_state_source"],
        legacy_cluster_label=dual["legacy_cluster_label"],
        taxonomy_status=dual["taxonomy_status"],
    )


def compute_forward_returns_no_lookahead(
    record: OpportunityCaptureRecord,
    *,
    price_oracle: DemoPriceOracle,
    coin_id: int | str | None,
    pair_address: str,
    now_utc: str | None = None,
) -> OpportunityCaptureRecord:
    """Compute forward returns only after horizons have matured — audit labels only."""
    first_seen = record.first_seen_timestamp
    entry_price = record.price_at_first_seen
    if not first_seen or not entry_price or float(entry_price) <= 0:
        return record

    first_dt = parse_ts(first_seen)
    if first_dt is None:
        return record

    now_dt = parse_ts(now_utc) or datetime.now(timezone.utc)
    elapsed = (now_dt - first_dt).total_seconds()
    snapshot_count = 0

    for label, seconds in HORIZON_SECONDS.items():
        matured_flag = f"horizon_matured_{label}"
        return_field = f"max_return_{label}"
        if elapsed < seconds:
            setattr(record, matured_flag, False)
            continue
        setattr(record, matured_flag, True)
        horizon_end = first_dt + timedelta(seconds=seconds)
        horizon_end_iso = horizon_end.isoformat()
        lookup = price_oracle.lookup_price(
            coin_id=coin_id,
            pair_address=pair_address,
            order_created_at_utc=horizon_end_iso,
            decision_created_at_utc=first_seen,
        )
        if lookup.price and lookup.price_status == "PRICE_OK":
            snapshot_count += 1
            max_ret = (float(lookup.price) - float(entry_price)) / float(entry_price)
            setattr(record, return_field, max_ret)

    record.outcome_computed_at = utc_now_iso()
    record.outcome_source_snapshot_count = snapshot_count
    return record


def is_missed_winner(record: OpportunityCaptureRecord, threshold: float = MISSED_WINNER_THRESHOLD) -> bool:
    """Candidate not traded but achieved large forward return."""
    if record.paper_action_taken in ("FILLED", "OPENED", "TRADE", "TRADE_EXPLORATION_OVERRIDE"):
        return False
    for label in ("5m", "15m", "1h", "6h", "24h"):
        matured = getattr(record, f"horizon_matured_{label}", False)
        ret = getattr(record, f"max_return_{label}", None)
        if matured and ret is not None and float(ret) >= threshold:
            return True
    return False


def build_missed_winner_record(record: OpportunityCaptureRecord) -> dict[str, Any]:
    best_return = None
    best_horizon = None
    for label in ("24h", "6h", "1h", "15m", "5m"):
        matured = getattr(record, f"horizon_matured_{label}", False)
        ret = getattr(record, f"max_return_{label}", None)
        if matured and ret is not None:
            if best_return is None or float(ret) > float(best_return):
                best_return = ret
                best_horizon = label

    return {
        "record_type": "MISSED_WINNER",
        "schema_version": record.schema_version,
        "created_at_utc": utc_now_iso(),
        "loop_run_id": record.loop_run_id,
        "loop_iteration": record.loop_iteration,
        "source_decision_id": record.source_decision_id,
        "candidate_id": record.candidate_id,
        "pair_address": record.pair_address,
        "first_seen_timestamp": record.first_seen_timestamp,
        "reason_not_traded": record.reason_for_no_trade,
        "blocked_by_ae9": record.blocked_by_ae9,
        "stale_price": record.stale_price,
        "missing_context": record.missing_context,
        "max_open_positions": record.max_open_positions_hit,
        "cooldown": record.cooldown_active,
        "duplicate_active_pair": record.duplicate_active_pair,
        "missing_identity_linkage": record.missing_identity,
        "paper_policy_prevented": record.paper_policy_prevented,
        "best_forward_return": best_return,
        "best_forward_horizon": best_horizon,
        "strict_shadow_decision": record.strict_shadow_decision,
        "exploration_decision": record.exploration_decision,
    }
