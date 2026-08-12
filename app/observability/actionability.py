"""Controlled DEMO/PAPER actionability orchestration."""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

from ..execution.paper import get_paper_trader
from ..models.predictor import normalize_execution_settings
from .audit_reasons import AuditReason
from .candidate import TradeCandidate
from .decision_trace import safe_persist_decision_trace
from .economic_gate import DecisionResult, evaluate_economic_trade_candidate
from .effective_settings import EffectiveSettings, get_effective_settings
from .llm_gate import BEARISH_ALERT_TYPES, evaluate_llm_short_circuit_phase2
from .whale_wave_features import compute_rolling_whale_wave_features, load_snapshots_for_pair

log = logging.getLogger("actionability")

_LLM_CACHE: dict[str, float] = {}


def build_candidate_from_pair(
    pair: dict[str, Any],
    *,
    whale_score: float,
    signal_action: str,
    signal_score: float,
    coin_id: int | None,
    chain: str,
    symbol: str,
    price: float,
    liquidity_usd: float,
    alert_type: str | None = None,
    cluster_label: str = "OPPORTUNISTIC_SPECULATIVE",
    cluster_is_default: bool = True,
    sentiment_score: float | None = None,
    sentiment_available: bool = False,
    has_open_position: bool = False,
    scan_id: str | None = None,
    settings_hash: str | None = None,
) -> TradeCandidate:
    txns = (pair.get("txns") or {}).get("h24") or {}
    buys = int(txns.get("buys") or 0)
    sells = int(txns.get("sells") or 0)
    vol = pair.get("volume") or {}
    pair_address = (pair.get("pairAddress") or "").strip()
    br = buys / max(buys + sells, 1)

    snapshots = load_snapshots_for_pair(pair_address)
    wave_features = compute_rolling_whale_wave_features(snapshots) if snapshots else {}

    return TradeCandidate(
        pair_address=pair_address,
        chain=chain,
        symbol=symbol,
        price=price,
        liquidity_usd=liquidity_usd,
        whale_score=whale_score,
        signal_score=signal_score,
        signal_type=signal_action,
        coin_id=coin_id,
        volume_5m=float(vol.get("m5") or 0) or None,
        volume_15m=None,
        volume_1h=float(vol.get("h1") or 0) or None,
        volume_24h=float(vol.get("h24") or 0) or None,
        buy_count=buys,
        sell_count=sells,
        buy_ratio=round(br, 4),
        alert_type=alert_type,
        bearish_alert_active=alert_type in BEARISH_ALERT_TYPES,
        existing_open_position_for_pair=has_open_position,
        sentiment_available=sentiment_available,
        sentiment_score=sentiment_score,
        cluster_label=cluster_label,
        cluster_is_default=cluster_is_default,
        cluster_confidence=0.5 if cluster_is_default else 0.8,
        current_execution_price=price,
        scan_id=scan_id,
        settings_hash=settings_hash,
        whale_wave_features=wave_features or None,
    )


def _llm_cache_key(candidate: TradeCandidate, settings_hash: str) -> str:
    payload = {
        "pair": candidate.pair_address,
        "signal": candidate.signal_type,
        "alert": candidate.alert_type,
        "price": candidate.price,
        "settings_hash": settings_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _llm_enabled_for_mode(settings: dict[str, Any]) -> bool:
    mode = str(settings.get("trading_mode", "DEMO")).upper()
    if mode == "LIVE":
        return bool(settings.get("llm_enabled_for_live", False))
    return bool(settings.get("llm_enabled_for_demo", False))


async def evaluate_and_execute_candidate(
    candidate: TradeCandidate,
    *,
    pair: dict[str, Any],
    settings: dict[str, Any] | None = None,
    alert: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Full DEMO/PAPER actionability path:
    economic gate -> optional LLM veto -> paper execution.
    """
    settings = normalize_execution_settings(settings or {})
    eff = get_effective_settings(settings)
    candidate.settings_hash = eff.settings_hash

    if not bool(settings.get("economic_gate_enabled", False)):
        return {"action": "SKIPPED", "reason": "economic_gate_disabled"}

    gate_result = evaluate_economic_trade_candidate(candidate, eff, pair=pair)
    candidate.actionability_decision = gate_result.action
    candidate.audit_reasons = gate_result.reasons
    candidate.estimated_slippage_per_side_pct = gate_result.estimated_slippage_per_side_pct
    candidate.round_trip_slippage_pct = gate_result.round_trip_slippage_pct
    candidate.total_cost_pct = gate_result.total_cost_pct
    candidate.expected_net_return = gate_result.expected_net_return_pct
    candidate.probability_profitable_after_costs = gate_result.probability_profitable_after_costs
    candidate.price_drift_from_model_pct = gate_result.price_drift_from_model_pct

    safe_persist_decision_trace(candidate=candidate, result=gate_result, stage="economic_gate")

    if gate_result.action != "PAPER_BUY_CANDIDATE":
        return {
            "action": gate_result.action,
            "reasons": gate_result.reasons,
            "decision_trace_id": candidate.decision_trace_id,
        }

    # LLM layer — only after economic approval
    should_llm, llm_reasons = evaluate_llm_short_circuit_phase2(
        economic_approved=True,
        candidate=candidate,
        gate_result=gate_result,
        settings=settings,
        alert=alert,
    )
    llm_status = "skipped"
    decision = None
    decision_id = None

    if should_llm and _llm_enabled_for_mode(settings):
        cache_key = _llm_cache_key(candidate, eff.settings_hash)
        if cache_key in _LLM_CACHE:
            llm_status = "cached_skip"
            gate_result.reasons.append(AuditReason.LLM_SHORT_CIRCUITED.value)
        else:
            from ..analytics.features import ClusterLabel, build_feature_row
            from ..models import MarketState, TokenMetadata, Network
            from ..models.predictor import analyze_market_state

            _LLM_CACHE[cache_key] = 1.0
            llm_status = "attempted"
            gate_result.reasons.append(AuditReason.LLM_CALL_ATTEMPTED.value)

            base = pair.get("baseToken") or {}
            chain = (pair.get("chainId") or "unknown").lower()
            _CHAIN_MAP = {
                "solana": Network.SOLANA, "ethereum": Network.ETHEREUM,
                "bsc": Network.BSC, "base": Network.BASE,
            }
            token = TokenMetadata(
                contract_address=candidate.pair_address,
                symbol=base.get("symbol", "?"),
                name=base.get("name", "?"),
                network=_CHAIN_MAP.get(chain, Network.UNKNOWN),
            )
            liq = candidate.liquidity_usd
            txns = (pair.get("txns") or {}).get("h24") or {}
            state = MarketState(
                contract_address=candidate.pair_address,
                price_usd=candidate.price,
                liquidity_usd=liq,
                volume_24h=candidate.volume_24h or 0,
                price_change_24h=float((pair.get("priceChange") or {}).get("h24") or 0),
                price_change_1h=float((pair.get("priceChange") or {}).get("h1") or 0),
                txns_buys_24h=int(txns.get("buys") or 0),
                txns_sells_24h=int(txns.get("sells") or 0),
            )
            try:
                cluster_enum = ClusterLabel(candidate.cluster_label)
            except ValueError:
                cluster_enum = ClusterLabel.OPPORTUNISTIC_SPECULATIVE
            metrics = build_feature_row(pair, state, cluster_enum, candidate.whale_score)
            open_positions = get_paper_trader().get_positions("OPEN")
            decision, decision_id = await analyze_market_state(
                metrics,
                candidate.cluster_label,
                candidate.sentiment_score or 0.0,
                open_positions=open_positions,
                coin_id=candidate.coin_id,
                trigger_type="economic_gate_candidate",
            )
            if decision.decision in ("HOLD", "SELL"):
                gate_result.action = "BLOCKED"
                gate_result.reasons.append(AuditReason.BLOCKED_BY_LLM_VETO.value)
                gate_result.reasons.append(AuditReason.LLM_VETO_REASON.value)
                llm_status = "vetoed"
                safe_persist_decision_trace(
                    candidate=candidate, result=gate_result, stage="llm_veto", llm_status=llm_status,
                )
                return {
                    "action": "BLOCKED",
                    "reasons": gate_result.reasons,
                    "decision_trace_id": candidate.decision_trace_id,
                    "llm_decision": decision.model_dump(),
                }
    else:
        if not _llm_enabled_for_mode(settings):
            gate_result.reasons.append(AuditReason.LLM_SKIPPED_DISABLED.value)
        else:
            gate_result.reasons.extend(llm_reasons)
            if AuditReason.LLM_BUDGET_BLOCKED.value in llm_reasons:
                gate_result.reasons.append(AuditReason.LLM_SKIPPED_BUDGET.value)

    gate_result.action = "PAPER_BUY_APPROVED"
    gate_result.reasons.append(AuditReason.PAPER_BUY_APPROVED.value)
    safe_persist_decision_trace(candidate=candidate, result=gate_result, stage="paper_buy_approved", llm_status=llm_status)

    # Paper execution via safe service path
    trader = get_paper_trader()
    capital = float(settings.get("starting_capital", 10_000))
    size_pct = float(settings.get("max_position_size_pct", 0.05))
    base_notional = capital * size_pct * gate_result.position_size_multiplier
    base_notional = min(base_notional, capital * size_pct * 2.0)

    coin = {
        "symbol": candidate.symbol,
        "chain": candidate.chain,
        "price_usd": candidate.current_execution_price or candidate.price,
        "pair_address": candidate.pair_address,
        "coin_id": candidate.coin_id,
        "decision_ref_id": decision_id,
    }
    exec_result = trader.try_autonomous_buy(
        coin,
        cluster_label=candidate.cluster_label,
        settings=settings,
        strategy_type="WHALE_RIDER" if candidate.whale_score >= 0.6 else "SCALPING_OPPORTUNITY",
        size_usd=round(base_notional, 2),
    )
    if exec_result:
        gate_result.reasons.append(AuditReason.PAPER_BUY_EXECUTED.value)
        gate_result.reasons.append(AuditReason.PAPER_TRADE_CREATED.value)
        safe_persist_decision_trace(
            candidate=candidate, result=gate_result, stage="paper_executed", llm_status=llm_status,
            extra={"position_id": (exec_result.get("position") or {}).get("id")},
        )
        return {
            "action": "PAPER_BUY_EXECUTED",
            "reasons": gate_result.reasons,
            "decision_trace_id": candidate.decision_trace_id,
            "execution": exec_result,
        }

    gate_result.reasons.append(AuditReason.BLOCKED_BY_FILL_PRICE.value)
    gate_result.action = "BLOCKED"
    safe_persist_decision_trace(candidate=candidate, result=gate_result, stage="paper_blocked", llm_status=llm_status)
    return {
        "action": "BLOCKED",
        "reasons": gate_result.reasons,
        "decision_trace_id": candidate.decision_trace_id,
    }
