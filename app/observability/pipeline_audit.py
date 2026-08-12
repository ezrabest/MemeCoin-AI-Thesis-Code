"""Pipeline decision audit recorder — non-fatal to market collection."""
from __future__ import annotations

import json
import logging
from typing import Any

from .. import database as db
from ..engine import (
    SIGNAL_BUY_LIQUIDITY_USD,
    SIGNAL_BUY_PROB_THRESHOLD,
    SIGNAL_BUY_WHALE_THRESHOLD,
    SIGNAL_WATCH_PROB_THRESHOLD,
    SIGNAL_WATCH_WHALE_THRESHOLD,
    WHALE_ALERT_MIN_VOLUME_24H,
    WHALE_ALERT_MIN_WHALE_SCORE,
)
from .audit_io import (
    get_decision_trace_writer,
    get_pipeline_reasons_writer,
    new_decision_trace_id,
)
from .audit_reasons import AuditReason
from .effective_settings import get_effective_settings

log = logging.getLogger("pipeline_audit")

BEARISH_ALERT_TYPES = frozenset({"LARGE_SELL", "DISTRIBUTION"})


def _default_thresholds(settings_hash: str) -> dict[str, Any]:
    eff = get_effective_settings()
    return {
        "min_liquidity_usd": eff.canonical.get("min_liquidity_usd"),
        "min_whale_score": eff.canonical.get("min_whale_score"),
        "llm_score_threshold": eff.canonical.get("llm_score_threshold"),
        "signal_buy_whale": SIGNAL_BUY_WHALE_THRESHOLD,
        "signal_buy_liquidity": SIGNAL_BUY_LIQUIDITY_USD,
        "signal_buy_prob": SIGNAL_BUY_PROB_THRESHOLD,
        "signal_watch_prob": SIGNAL_WATCH_PROB_THRESHOLD,
        "signal_watch_whale": SIGNAL_WATCH_WHALE_THRESHOLD,
        "alert_min_volume": WHALE_ALERT_MIN_VOLUME_24H,
        "alert_min_whale": WHALE_ALERT_MIN_WHALE_SCORE,
        "settings_hash": settings_hash,
    }


def derive_signal_audit_reasons(
    *,
    signal_action: str,
    prob_up: float,
    whale_score: float,
    liquidity_usd: float,
    alert_type: str | None = None,
) -> list[str]:
    """Non-invasive audit reasons for engine signal outcomes."""
    reasons: list[str] = []

    if signal_action == "BUY":
        reasons.append(AuditReason.BUY_SIGNAL_CREATED.value)
        if alert_type in BEARISH_ALERT_TYPES:
            reasons.append(AuditReason.CONFLICT_ENGINE_BUY_BEARISH_ALERT.value)
        return reasons

    if signal_action == "WATCH":
        reasons.append(AuditReason.WATCH_NOT_ACTIONABLE.value)
        if liquidity_usd < SIGNAL_BUY_LIQUIDITY_USD:
            reasons.append(AuditReason.BELOW_LIQUIDITY_THRESHOLD.value)
        if whale_score < SIGNAL_BUY_WHALE_THRESHOLD:
            reasons.append(AuditReason.BELOW_WHALE_THRESHOLD.value)
        if prob_up < SIGNAL_BUY_PROB_THRESHOLD:
            reasons.append(AuditReason.NO_ACTIONABLE_RULE_MATCH.value)
        if alert_type is None:
            reasons.append(AuditReason.ALERT_REQUIRED_BUT_MISSING.value)
        if alert_type in BEARISH_ALERT_TYPES:
            reasons.append(AuditReason.BLOCKED_BY_BEARISH_ALERT.value)
        return reasons

    if signal_action == "NO_TRADE":
        reasons.append(AuditReason.NO_ACTIONABLE_RULE_MATCH.value)
        if whale_score < SIGNAL_WATCH_WHALE_THRESHOLD and prob_up < SIGNAL_WATCH_PROB_THRESHOLD:
            reasons.append(AuditReason.BELOW_WHALE_THRESHOLD.value)
        return reasons

    return reasons


def derive_alert_audit_reasons(
    *,
    alert: dict[str, Any] | None,
    whale_score: float,
    volume_24h: float,
) -> list[str]:
    if alert is not None:
        return []
    reasons = [AuditReason.ALERT_REQUIRED_BUT_MISSING.value]
    if volume_24h < WHALE_ALERT_MIN_VOLUME_24H:
        reasons.append(AuditReason.NO_ACTIONABLE_RULE_MATCH.value)
    if whale_score < WHALE_ALERT_MIN_WHALE_SCORE:
        reasons.append(AuditReason.BELOW_WHALE_THRESHOLD.value)
    return reasons


def record_pipeline_decision(
    *,
    pair_address: str,
    coin_id: int | None = None,
    chain: str = "unknown",
    symbol: str = "",
    audit_reasons: list[str],
    scan_id: str | None = None,
    signal_action: str | None = None,
    alert_type: str | None = None,
    whale_score: float | None = None,
    model_metadata: dict[str, Any] | None = None,
    model_snapshot_price: float | None = None,
    current_execution_price: float | None = None,
    price_drift_from_model_pct: float | None = None,
    decision_trace_id: str | None = None,
    settings_hash: str | None = None,
    stage: str = "decision",
) -> str | None:
    """
    Persist audit event to JSONL + SQLite. Never raises — failures are logged.
    Returns decision_trace_id.
    """
    trace_id = decision_trace_id or new_decision_trace_id()
    eff = get_effective_settings()
    s_hash = settings_hash or eff.settings_hash
    thresholds = _default_thresholds(s_hash)

    record: dict[str, Any] = {
        "decision_trace_id": trace_id,
        "timestamp": db._utcnow(),  # type: ignore[attr-defined]
        "pair_address": pair_address,
        "coin_id": coin_id,
        "chain": chain,
        "symbol": symbol,
        "settings_hash": s_hash,
        "threshold_values": thresholds,
        "audit_reasons": audit_reasons,
        "signal_action": signal_action,
        "alert_type": alert_type,
        "whale_score": whale_score,
        "scan_id": scan_id,
        "stage": stage,
    }
    if model_metadata:
        record["model_metadata"] = model_metadata
    if model_snapshot_price is not None:
        record["model_snapshot_price"] = model_snapshot_price
    if current_execution_price is not None:
        record["current_execution_price"] = current_execution_price
    if price_drift_from_model_pct is not None:
        record["price_drift_from_model_pct"] = price_drift_from_model_pct

    try:
        get_pipeline_reasons_writer().append(record)
        get_decision_trace_writer().append(record)
    except Exception as exc:
        log.warning("JSONL pipeline audit write failed: %s", exc)

    try:
        db.insert_pipeline_audit({
            "scan_id": scan_id,
            "coin_id": coin_id,
            "symbol": symbol,
            "pair_address": pair_address,
            "chain": chain,
            "stage": stage,
            "filter_status": signal_action or "audit",
            "whale_score": whale_score,
            "alert_type": alert_type,
            "decision_trace_id": trace_id,
            "settings_hash": s_hash,
            "audit_reasons_json": audit_reasons,
            "threshold_values_json": thresholds,
            "model_metadata_json": model_metadata,
            "model_snapshot_price": model_snapshot_price,
            "current_execution_price": current_execution_price,
            "price_drift_from_model_pct": price_drift_from_model_pct,
            "details_json": {
                "audit_reasons": audit_reasons,
                "signal_action": signal_action,
            },
        })
    except Exception as exc:
        log.warning("SQLite pipeline audit insert failed (non-fatal): %s", exc)

    return trace_id


def safe_record_pipeline_decision(**kwargs: Any) -> str | None:
    """Wrapper that never propagates exceptions."""
    try:
        return record_pipeline_decision(**kwargs)
    except Exception as exc:
        log.warning("Pipeline audit recording skipped: %s", exc)
        return None
