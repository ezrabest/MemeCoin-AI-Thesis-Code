"""Decision trace persistence to JSONL + pipeline_audit."""
from __future__ import annotations

import logging
from typing import Any

from .. import database as db
from .audit_io import get_decision_trace_writer, get_pipeline_reasons_writer
from .candidate import TradeCandidate
from .economic_gate import DecisionResult
from .effective_settings import get_effective_settings

log = logging.getLogger("decision_trace")


def persist_decision_trace(
    candidate: TradeCandidate,
    result: DecisionResult | None = None,
    *,
    stage: str = "economic_gate",
    llm_status: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Non-fatal JSONL + SQLite decision trace write."""
    try:
        eff = get_effective_settings()
        s_hash = candidate.settings_hash or eff.settings_hash
        record: dict[str, Any] = {
            "decision_trace_id": candidate.decision_trace_id,
            "timestamp": candidate.event_timestamp,
            "pair_address": candidate.pair_address,
            "coin_id": candidate.coin_id,
            "chain": candidate.chain,
            "symbol": candidate.symbol,
            "settings_hash": s_hash,
            "stage": stage,
            "actionability_decision": result.action if result else candidate.actionability_decision,
            "audit_reasons": (result.reasons if result else candidate.audit_reasons) or [],
            "model_metadata": candidate.model_metadata,
            "model_snapshot_price": candidate.model_snapshot_price,
            "current_execution_price": candidate.current_execution_price or candidate.price,
            "price_drift_from_model_pct": (
                result.price_drift_from_model_pct if result else candidate.price_drift_from_model_pct
            ),
            "signal_type": candidate.signal_type,
            "alert_type": candidate.alert_type,
            "whale_score": candidate.whale_score,
            "scan_id": candidate.scan_id,
        }
        if result:
            record["cost_calculation"] = {
                "total_cost_pct": result.total_cost_pct,
                "round_trip_slippage_pct": result.round_trip_slippage_pct,
                "estimated_slippage_per_side_pct": result.estimated_slippage_per_side_pct,
                "expected_return_pct": result.expected_return_pct,
                "expected_net_return_pct": result.expected_net_return_pct,
                "expected_net_return": result.expected_net_return_pct,
                "probability_profitable_after_costs": result.probability_profitable_after_costs,
                "required_margin_after_costs_pct": result.required_margin_after_costs_pct,
                "position_size_multiplier": result.position_size_multiplier,
            }
            record.update(result.audit_payload)
        if llm_status:
            record["llm_status"] = llm_status
        if extra:
            record.update(extra)

        get_decision_trace_writer().append(record)
        get_pipeline_reasons_writer().append(record)

        db.insert_pipeline_audit({
            "scan_id": candidate.scan_id,
            "coin_id": candidate.coin_id,
            "symbol": candidate.symbol,
            "pair_address": candidate.pair_address,
            "chain": candidate.chain,
            "stage": stage,
            "filter_status": record.get("actionability_decision"),
            "whale_score": candidate.whale_score,
            "alert_type": candidate.alert_type,
            "decision_trace_id": candidate.decision_trace_id,
            "settings_hash": s_hash,
            "audit_reasons_json": record["audit_reasons"],
            "model_metadata_json": candidate.model_metadata,
            "model_snapshot_price": candidate.model_snapshot_price,
            "current_execution_price": record.get("current_execution_price"),
            "price_drift_from_model_pct": record.get("price_drift_from_model_pct"),
            "details_json": record,
        })
    except Exception as exc:
        log.warning("Decision trace persist failed (non-fatal): %s", exc)


def safe_persist_decision_trace(**kwargs: Any) -> None:
    try:
        persist_decision_trace(**kwargs)
    except Exception as exc:
        log.warning("Decision trace skipped: %s", exc)
