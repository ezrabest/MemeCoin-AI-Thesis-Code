"""Economic trade candidate evaluation — fail-closed DEMO/PAPER gate."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .audit_reasons import AuditReason
from .candidate import TradeCandidate
from .effective_settings import EffectiveSettings
from .model_lookup import get_model_lookup
from .model_runtime_inference import (
    PRIMARY_RF_TARGET,
    VALIDATED_RF_THRESHOLD,
    get_runtime_model_inference,
)
from .slippage import (
    check_price_drift,
    check_slippage_limit,
    compute_total_cost_pct,
    estimate_slippage_per_side_pct,
)

BEARISH_ALERT_TYPES = frozenset({"LARGE_SELL", "DISTRIBUTION"})
BULLISH_ALERT_TYPES = frozenset({"LARGE_BUY", "ACCUMULATION", "PUMP_SIGNAL"})


@dataclass
class DecisionResult:
    action: str  # HOLD / WATCH / PAPER_BUY_CANDIDATE / PAPER_BUY_APPROVED / BLOCKED
    expected_return_pct: float | None = None
    expected_net_return_pct: float | None = None
    probability_profitable_after_costs: float | None = None
    estimated_slippage_per_side_pct: float | None = None
    round_trip_slippage_pct: float | None = None
    total_cost_pct: float | None = None
    required_margin_after_costs_pct: float | None = None
    position_size_multiplier: float = 1.0
    price_drift_from_model_pct: float | None = None
    reasons: list[str] = field(default_factory=list)
    audit_payload: dict[str, Any] = field(default_factory=dict)


def _settings_dict(eff: EffectiveSettings | dict[str, Any]) -> dict[str, Any]:
    if isinstance(eff, EffectiveSettings):
        return eff.canonical
    return eff


def _parse_event_ts(ts_str: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(ts_str).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _freshness_audit(
    candidate: TradeCandidate,
    settings: dict[str, Any],
    *,
    inference_meta: dict[str, Any],
    prediction_generated_at: str | None,
    now: datetime,
) -> tuple[list[str], dict[str, Any]]:
    """Separate market snapshot, prediction, artifact, and offline file ages."""
    reasons: list[str] = []
    audit: dict[str, Any] = {}

    event_ts = _parse_event_ts(candidate.event_timestamp)
    if event_ts is not None:
        market_age = (now - event_ts).total_seconds()
        audit["market_snapshot_age_seconds"] = round(market_age, 3)
        max_market = float(settings.get("max_market_snapshot_age_seconds", 300))
        if market_age > max_market:
            reasons.append(AuditReason.MARKET_SNAPSHOT_TOO_OLD.value)

    if prediction_generated_at:
        pred_ts = _parse_event_ts(prediction_generated_at)
        if pred_ts is not None:
            pred_age = (now - pred_ts).total_seconds()
            audit["model_prediction_age_seconds"] = round(pred_age, 3)
            max_pred = float(settings.get("max_model_prediction_age_seconds", 300))
            if pred_age > max_pred:
                reasons.append(AuditReason.MODEL_PREDICTION_TOO_OLD.value)

    runtime = get_runtime_model_inference()
    artifact_age = runtime.artifact_age_seconds(now)
    if artifact_age is not None:
        audit["model_artifact_age_seconds"] = round(artifact_age, 3)
        max_hours = float(settings.get("max_model_artifact_age_hours", 168))
        if artifact_age > max_hours * 3600:
            reasons.append(AuditReason.MODEL_ARTIFACT_TOO_OLD.value)

    lookup = get_model_lookup()
    offline_age = lookup.offline_prediction_file_age_seconds()
    if offline_age is not None:
        audit["offline_prediction_file_age_seconds"] = round(offline_age, 3)

    lookbacks = settings.get("whale_wave_lookback_minutes", [5, 15, 60, 240])
    if isinstance(lookbacks, (list, tuple)) and lookbacks:
        audit["whale_wave_window_minutes_used"] = list(lookbacks)

    audit.update(inference_meta)
    return reasons, audit


def enrich_candidate_with_model(
    candidate: TradeCandidate,
    settings: dict[str, Any],
    *,
    pair: dict[str, Any] | None = None,
    now: datetime | None = None,
) -> TradeCandidate:
    """Attach live RF inference and optional Tab offline lookup."""
    now = now or datetime.now(timezone.utc)
    runtime = get_runtime_model_inference()
    inference = runtime.predict_for_candidate(candidate, pair, now=now)

    candidate.audit_reasons.extend(inference.audit_reasons)
    candidate.rf_prediction = inference.rf_prediction
    candidate.model_metadata = {
        "runtime_inference": inference.runtime_metadata,
        "inference_status": inference.status,
        "calibration": _load_calibration_metadata(),
        "expected_return_calibration_available": False,
    }

    if inference.model_snapshot_price is not None:
        candidate.model_snapshot_price = float(inference.model_snapshot_price)
    elif candidate.price > 0 and inference.status == "ok":
        candidate.model_snapshot_price = candidate.price

    freshness_reasons, freshness_audit = _freshness_audit(
        candidate,
        settings,
        inference_meta=inference.runtime_metadata,
        prediction_generated_at=inference.prediction_generated_at,
        now=now,
    )
    candidate.audit_reasons.extend(freshness_reasons)
    if candidate.model_metadata:
        candidate.model_metadata["freshness"] = freshness_audit

    tab_lookup = get_model_lookup().lookup_tab_exact(
        candidate.pair_address,
        candidate.event_timestamp,
    )
    if tab_lookup:
        candidate.tab_prediction = tab_lookup

    return candidate


def _load_calibration_metadata() -> dict[str, Any]:
    return get_model_lookup().get_calibration()


def _load_expected_return_calibration() -> dict[str, Any] | None:
    """Offline calibrated expected payoff table — not derived from future returns at runtime."""
    path = get_model_lookup().expected_return_calibration_path()
    if path is None or not path.is_file():
        return None
    try:
        import json
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if data.get("payoff_table"):
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return None


def evaluate_tab_confidence_boost(
    candidate: TradeCandidate,
    settings: dict[str, Any],
    *,
    rf_probability: float,
    trading_mode: str,
) -> tuple[float, list[str]]:
    """Position-size multiplier overlay only — never standalone BUY authority."""
    reasons: list[str] = []
    mode = str(trading_mode).upper()
    demo_enabled = bool(settings.get("tab_confidence_boost_enabled_demo", False))
    live_enabled = bool(settings.get("tab_confidence_boost_enabled_live", False))
    boost_on = demo_enabled if mode != "LIVE" else live_enabled

    if not bool(settings.get("tab_confidence_boost_enabled", False)) and not boost_on:
        reasons.append(AuditReason.TAB_CONFIDENCE_BOOST_DISABLED.value)
        return 1.0, reasons

    if mode == "LIVE" and not live_enabled:
        reasons.append(AuditReason.TAB_CONFIDENCE_BOOST_DISABLED.value)
        return 1.0, reasons

    if bool(settings.get("tab_standalone_trading_enabled", False)):
        reasons.append(AuditReason.SETTINGS_BLOCKED.value)

    if bool(settings.get("tab_rescue_enabled", False)):
        reasons.append(AuditReason.SETTINGS_BLOCKED.value)

    threshold = float(settings.get("rf_probability_threshold", VALIDATED_RF_THRESHOLD))
    if rf_probability < threshold:
        return 1.0, reasons

    tab = candidate.tab_prediction
    if not tab:
        reasons.append(AuditReason.TAB_CONFIDENCE_UNAVAILABLE.value)
        return 1.0, reasons

    suffix = str(settings.get("tab_confidence_suffix", "nearest_neighbors_context_4096"))
    if tab.get("tab_suffix") and suffix not in str(tab.get("tab_suffix", "")):
        reasons.append(AuditReason.TAB_CONFIDENCE_UNAVAILABLE.value)
        return 1.0, reasons

    pct_threshold = float(settings.get("tab_confidence_percentile_threshold", 0.98))
    tab_score = float(tab.get("tab_score", 0))
    p98 = tab.get("percentile_threshold")
    meets = tab.get("meets_percentile", False)
    if p98 is not None and tab_score < float(p98):
        meets = False
    if not meets and p98 is None and tab_score < pct_threshold:
        reasons.append(AuditReason.TAB_CONFIDENCE_UNAVAILABLE.value)
        return 1.0, reasons

    multiplier = float(settings.get("tab_position_size_multiplier", 1.5))
    reasons.append(AuditReason.TAB_CONFIDENCE_BOOST_APPLIED.value)
    return min(multiplier, 2.0), reasons


def evaluate_economic_trade_candidate(
    candidate: TradeCandidate,
    effective_settings: EffectiveSettings | dict[str, Any],
    *,
    position_size_usd: float | None = None,
    pair: dict[str, Any] | None = None,
) -> DecisionResult:
    """
    Fail-closed economic gate for DEMO/PAPER.
    Does not execute trades — returns DecisionResult only.
    """
    settings = _settings_dict(effective_settings)
    reasons: list[str] = list(candidate.audit_reasons)
    now = datetime.now(timezone.utc)

    if not bool(settings.get("economic_gate_enabled", False)):
        reasons.append(AuditReason.SETTINGS_BLOCKED.value)
        return DecisionResult(action="WATCH", reasons=reasons)

    trading_mode = str(settings.get("trading_mode", "DEMO")).upper()
    if trading_mode == "LIVE" and not bool(settings.get("live_trading_enabled", False)):
        reasons.append(AuditReason.SETTINGS_BLOCKED.value)
        return DecisionResult(action="BLOCKED", reasons=reasons)

    paper_ok = bool(settings.get("paper_trading_enabled", True)) and trading_mode == "DEMO"
    demo_ok = bool(settings.get("demo_aggressive_enabled", False))
    if not paper_ok and not demo_ok:
        reasons.append(AuditReason.SETTINGS_BLOCKED.value)
        return DecisionResult(action="HOLD", reasons=reasons)

    if bool(settings.get("allow_watch_to_buy_promotion", False)):
        reasons.append(AuditReason.SETTINGS_BLOCKED.value)

    # Hard blockers — price, liquidity, pair, coin_id
    if not candidate.pair_address:
        reasons.append(AuditReason.MISSING_PRICE_OR_PAIR.value)
        return DecisionResult(action="HOLD", reasons=reasons)
    if candidate.coin_id is None:
        reasons.append(AuditReason.MISSING_PRICE_OR_PAIR.value)
        return DecisionResult(action="HOLD", reasons=reasons)
    if candidate.price is None or candidate.price <= 0:
        reasons.append(AuditReason.MISSING_PRICE_OR_PAIR.value)
        return DecisionResult(action="HOLD", reasons=reasons)
    if candidate.liquidity_usd is None or candidate.liquidity_usd <= 0:
        reasons.append(AuditReason.MISSING_PRICE_OR_PAIR.value)
        return DecisionResult(action="HOLD", reasons=reasons)

    # Bearish veto — hard block
    if candidate.bearish_alert_active or candidate.alert_type in BEARISH_ALERT_TYPES:
        reasons.append(AuditReason.BLOCKED_BY_BEARISH_ALERT.value)
        return DecisionResult(action="BLOCKED", reasons=reasons)

    # Missing bullish alert — audit only, not a blocker in DEMO/PAPER
    if not candidate.alert_type or candidate.alert_type not in BULLISH_ALERT_TYPES:
        reasons.append(AuditReason.MISSING_BULLISH_ALERT.value)

    if candidate.existing_open_position_for_pair:
        reasons.append(AuditReason.BLOCKED_BY_DUPLICATE_PAIR.value)
        return DecisionResult(action="BLOCKED", reasons=reasons)

    min_liq = float(settings.get("min_liquidity_usd", 5000))
    min_whale = float(settings.get("min_whale_score", 0.30))
    min_signal = float(settings.get("min_signal_score", 0.55))
    min_buy_ratio = float(settings.get("min_buy_ratio", 0.50))

    if candidate.liquidity_usd < min_liq:
        reasons.append(AuditReason.BELOW_LIQUIDITY_THRESHOLD.value)
        return DecisionResult(action="HOLD", reasons=reasons)
    if candidate.whale_score < min_whale:
        reasons.append(AuditReason.BELOW_WHALE_THRESHOLD.value)
        return DecisionResult(action="HOLD", reasons=reasons)
    if candidate.signal_score < min_signal:
        reasons.append(AuditReason.NO_ACTIONABLE_RULE_MATCH.value)
        return DecisionResult(action="HOLD", reasons=reasons)
    if candidate.buy_ratio is not None and candidate.buy_ratio < min_buy_ratio:
        reasons.append(AuditReason.NO_ACTIONABLE_RULE_MATCH.value)
        return DecisionResult(action="HOLD", reasons=reasons)

    # Runtime RF inference (mandatory when gate enabled)
    enrich_candidate_with_model(candidate, settings, pair=pair, now=now)
    rf = candidate.rf_prediction
    runtime_status = (candidate.model_metadata or {}).get("inference_status")

    blocking_inference_reasons = {
        AuditReason.MODEL_RUNTIME_INFERENCE_NOT_AVAILABLE.value,
        AuditReason.MODEL_ARTIFACT_LOAD_FAILED.value,
        AuditReason.MODEL_SCHEMA_MISMATCH.value,
        AuditReason.MODEL_FEATURE_MISSING.value,
        AuditReason.MODEL_FEATURE_EXTRA.value,
        AuditReason.MODEL_SCHEMA_METADATA_MISSING.value,
        AuditReason.MODEL_PREPROCESSOR_MISSING.value,
        AuditReason.MODEL_TRAINED_WITH_TARGET_LEAKAGE.value,
    }
    if runtime_status != "ok" or rf is None:
        for r in candidate.audit_reasons:
            if r in blocking_inference_reasons:
                reasons.append(r)
        if not any(r in blocking_inference_reasons for r in reasons):
            reasons.append(AuditReason.MODEL_RUNTIME_INFERENCE_NOT_AVAILABLE.value)
        return DecisionResult(action="HOLD", reasons=reasons)

    freshness_block = {
        AuditReason.MARKET_SNAPSHOT_TOO_OLD.value,
        AuditReason.MODEL_PREDICTION_TOO_OLD.value,
        AuditReason.MODEL_ARTIFACT_TOO_OLD.value,
    }
    if any(r in freshness_block for r in candidate.audit_reasons):
        reasons.extend(r for r in candidate.audit_reasons if r in freshness_block)
        return DecisionResult(action="HOLD", reasons=reasons)

    rf_prob = float(rf.get("predicted_probability", 0))
    threshold = float(settings.get("rf_probability_threshold", VALIDATED_RF_THRESHOLD))
    if rf_prob < threshold:
        reasons.append(AuditReason.PROBABILITY_BELOW_THRESHOLD.value)
        return DecisionResult(action="HOLD", reasons=reasons)

    # Price drift — before slippage / LLM / execution
    max_drift = float(settings.get("max_price_drift_from_model_pct", 0.01))
    candidate.current_execution_price = candidate.current_execution_price or candidate.price
    candidate.max_price_drift_from_model_pct = max_drift

    drift_ok, drift_pct, drift_reasons = check_price_drift(
        model_snapshot_price=candidate.model_snapshot_price,
        current_execution_price=candidate.current_execution_price,
        max_price_drift_from_model_pct=max_drift,
    )
    reasons.extend(drift_reasons)
    if not drift_ok:
        return DecisionResult(
            action="HOLD",
            price_drift_from_model_pct=drift_pct,
            reasons=reasons,
        )

    # Slippage
    capital = float(settings.get("starting_capital", 10_000))
    size_pct = float(settings.get("max_position_size_pct", 0.05))
    notional = position_size_usd or (capital * size_pct)

    slip_per_side, slip_reasons = estimate_slippage_per_side_pct(
        position_size_usd=notional,
        liquidity_usd=candidate.liquidity_usd,
        volume_24h=candidate.volume_24h,
        baseline_slippage_pct=float(settings.get("baseline_slippage_pct", 0.015)),
        baseline_slippage_is_per_side=bool(settings.get("baseline_slippage_is_per_side", True)),
        dynamic_slippage_enabled=bool(settings.get("dynamic_slippage_enabled", True)),
        effective_liquidity_conservative_factor=float(
            settings.get("effective_liquidity_conservative_factor", 1.0)
        ),
        slippage_volume_liquidity_multiplier=float(
            settings.get("slippage_volume_liquidity_multiplier", 0.5)
        ),
    )
    reasons.extend(slip_reasons)
    if slip_per_side is None:
        return DecisionResult(action="HOLD", reasons=reasons)

    round_trip_slip_pp = round(2.0 * slip_per_side, 6)
    max_slip = float(settings.get("max_slippage_pct", 0.015))
    slip_ok, slip_block = check_slippage_limit(slip_per_side, max_slip)
    reasons.extend(slip_block)
    if not slip_ok:
        return DecisionResult(
            action="HOLD",
            estimated_slippage_per_side_pct=slip_per_side,
            round_trip_slippage_pct=round_trip_slip_pp,
            reasons=reasons,
        )

    fee_pp = float(settings.get("round_trip_fee_pct", 0.03))
    gas_pp = float(settings.get("gas_or_priority_cost_pct", 0.0))
    total_cost = compute_total_cost_pct(
        round_trip_fee_pct=fee_pp,
        round_trip_slippage_pct=round_trip_slip_pp,
        gas_or_priority_cost_pct=gas_pp,
    )

    required_margin = float(settings.get("required_margin_after_costs_pct", 0.005))
    if settings.get("required_margin_after_costs_pct") is None and settings.get("required_margin_after_costs") is not None:
        from .settings_normalize import normalize_decimal_fraction_pct
        legacy = normalize_decimal_fraction_pct(settings.get("required_margin_after_costs"))
        if legacy is not None:
            required_margin = legacy

    calib = _load_expected_return_calibration()
    expected_return_pct: float | None = None
    expected_net_return_pct: float | None = None

    if calib and calib.get("payoff_table"):
        bucket = calib["payoff_table"].get(f"{rf_prob:.2f}") or calib["payoff_table"].get(str(round(rf_prob, 2)))
        if bucket is not None:
            expected_return_pct = float(bucket.get("expected_return_pct", bucket))
            expected_net_return_pct = round(expected_return_pct - total_cost, 6)
            if expected_net_return_pct <= required_margin:
                reasons.append(AuditReason.EXPECTED_RETURN_BELOW_MARGIN.value)
                return DecisionResult(
                    action="HOLD",
                    expected_return_pct=expected_return_pct,
                    expected_net_return_pct=expected_net_return_pct,
                    probability_profitable_after_costs=rf_prob,
                    estimated_slippage_per_side_pct=slip_per_side,
                    round_trip_slippage_pct=round_trip_slip_pp,
                    total_cost_pct=total_cost,
                    required_margin_after_costs_pct=required_margin,
                    price_drift_from_model_pct=drift_pct,
                    reasons=reasons,
                )
    else:
        reasons.append(AuditReason.EXPECTED_RETURN_CALIBRATION_UNAVAILABLE.value)

    tab_mult, tab_reasons = evaluate_tab_confidence_boost(
        candidate,
        settings,
        rf_probability=rf_prob,
        trading_mode=trading_mode,
    )
    reasons.extend(tab_reasons)

    reasons.append(AuditReason.ECONOMIC_GATE_APPROVED.value)
    reasons.append(AuditReason.PAPER_BUY_CANDIDATE_CREATED.value)

    return DecisionResult(
        action="PAPER_BUY_CANDIDATE",
        expected_return_pct=expected_return_pct,
        expected_net_return_pct=expected_net_return_pct,
        probability_profitable_after_costs=rf_prob,
        estimated_slippage_per_side_pct=slip_per_side,
        round_trip_slippage_pct=round_trip_slip_pp,
        total_cost_pct=total_cost,
        required_margin_after_costs_pct=required_margin,
        position_size_multiplier=tab_mult,
        price_drift_from_model_pct=drift_pct,
        reasons=reasons,
        audit_payload={
            "rf_probability": rf_prob,
            "rf_threshold": threshold,
            "model_metadata": candidate.model_metadata,
            "tab_multiplier": tab_mult,
            "target_name": PRIMARY_RF_TARGET,
        },
    )
