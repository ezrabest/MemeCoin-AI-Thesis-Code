"""Single source for runtime-effective settings, aliases, and hidden thresholds."""
from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
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
from ..models.predictor import normalize_execution_settings
from .settings_normalize import normalize_canonical_settings
from .audit_io import utc_timestamp_slug, write_json_report_atomic

DATA_DIR = Path(__file__).parent.parent.parent / "data"

# UI alias → canonical backend key
SETTING_ALIASES: dict[str, str] = {
    "minLiquidity": "min_liquidity_usd",
    "positionSizePct": "max_position_size_pct",
    "stopLossPct": "stop_loss_pct",
    "takeProfitPct": "take_profit_pct",
    "mode": "trading_mode",
    "tradingFee": "paper_fee_bps",
}

CANONICAL_KEYS: list[str] = [
    "trading_mode",
    "mode",
    "min_liquidity_usd",
    "min_whale_score",
    "llm_score_threshold",
    "max_position_size_pct",
    "stop_loss_pct",
    "take_profit_pct",
    "max_slippage_pct",
    "round_trip_fee_pct",
    "max_open_positions",
    "cooldown_minutes",
    "demo_aggressive_enabled",
    "required_margin_after_costs",
    "probability_profitable_threshold",
    "max_llm_calls_per_hour",
    "max_llm_calls_per_scan",
    "llm_cache_window_minutes",
    "llm_enabled_for_demo",
    "llm_enabled_for_live",
    "auto_execution_enabled",
    "max_risk_score",
    "enforce_risk_gate",
    "starting_capital",
    "paper_fee_bps",
    "prompt_behavior",
    # Phase 2 feature switches
    "paper_trading_enabled",
    "live_trading_enabled",
    "economic_gate_enabled",
    "allow_watch_to_buy_promotion",
    "allow_model_unavailable_fallback",
    "rf_gate_enabled",
    "rf_probability_threshold",
    "tab_confidence_boost_enabled",
    "tab_confidence_boost_enabled_demo",
    "tab_confidence_boost_enabled_live",
    "tab_confidence_suffix",
    "tab_confidence_percentile_threshold",
    "tab_position_size_multiplier",
    "tab_standalone_trading_enabled",
    "tab_rescue_enabled",
    "baseline_slippage_pct",
    "baseline_slippage_is_per_side",
    "dynamic_slippage_enabled",
    "slippage_liquidity_impact_multiplier",
    "slippage_volume_liquidity_multiplier",
    "gas_or_priority_cost_pct",
    "max_price_drift_from_model_pct",
    "min_market_read_interval_seconds",
    "max_market_snapshot_age_seconds",
    "max_model_prediction_age_seconds",
    "max_model_artifact_age_hours",
    "whale_wave_lookback_minutes",
    "effective_liquidity_conservative_factor",
    "required_margin_after_costs_pct",
    # Legacy alias — mapped to max_market_snapshot_age_seconds
    "max_model_snapshot_age_seconds",
    "min_signal_score",
    "min_buy_ratio",
    "max_daily_loss_pct",
    "max_drawdown_pct",
    "trailing_stop_pct",
    "time_stop_minutes",
    "duplicate_pair_guard_enabled",
]

# Defaults — decimal-fraction *_pct (0.015 = 1.5%, 0.03 = 3%)
_EXTENDED_DEFAULTS: dict[str, Any] = {
    "max_slippage_pct": 0.015,
    "round_trip_fee_pct": 0.03,
    "max_open_positions": 10,
    "cooldown_minutes": 0,
    "demo_aggressive_enabled": False,
    "required_margin_after_costs": 0.5,
    "required_margin_after_costs_pct": 0.005,
    "probability_profitable_threshold": 0.55,
    "max_llm_calls_per_hour": 60,
    "max_llm_calls_per_scan": 5,
    "llm_cache_window_minutes": 15,
    "llm_enabled_for_demo": False,
    "llm_enabled_for_live": False,
    "enforce_risk_gate": False,
    "mode": "DEMO",
    # Phase 2
    "paper_trading_enabled": True,
    "live_trading_enabled": False,
    "economic_gate_enabled": False,
    "allow_watch_to_buy_promotion": False,
    "allow_model_unavailable_fallback": False,
    "rf_gate_enabled": True,
    "rf_probability_threshold": 0.70,
    "tab_confidence_boost_enabled": False,
    "tab_confidence_boost_enabled_demo": False,
    "tab_confidence_boost_enabled_live": False,
    "tab_confidence_suffix": "nearest_neighbors_context_4096",
    "tab_confidence_percentile_threshold": 0.98,
    "tab_position_size_multiplier": 1.5,
    "tab_standalone_trading_enabled": False,
    "tab_rescue_enabled": False,
    "baseline_slippage_pct": 0.015,
    "baseline_slippage_is_per_side": True,
    "dynamic_slippage_enabled": True,
    "slippage_liquidity_impact_multiplier": 1.0,
    "slippage_volume_liquidity_multiplier": 0.5,
    "gas_or_priority_cost_pct": 0.0,
    "max_price_drift_from_model_pct": 0.01,
    "min_market_read_interval_seconds": 60,
    "max_market_snapshot_age_seconds": 300,
    "max_model_prediction_age_seconds": 300,
    "max_model_artifact_age_hours": 168.0,
    "whale_wave_lookback_minutes": [5, 15, 60, 240],
    "effective_liquidity_conservative_factor": 1.0,
    "max_model_snapshot_age_seconds": 300,
    "min_signal_score": 0.55,
    "min_buy_ratio": 0.50,
    "max_daily_loss_pct": 0.05,
    "max_drawdown_pct": 0.15,
    "trailing_stop_pct": 0.0,
    "time_stop_minutes": 0,
    "duplicate_pair_guard_enabled": True,
}


def _env_overrides() -> dict[str, tuple[Any, str]]:
    """Environment variables that override settings at runtime."""
    overrides: dict[str, tuple[Any, str]] = {}
    if os.getenv("MIN_WHALE_SCORE"):
        try:
            overrides["min_whale_score"] = (float(os.getenv("MIN_WHALE_SCORE", "")), "env:MIN_WHALE_SCORE")
        except ValueError:
            pass
    if os.getenv("POLL_INTERVAL"):
        overrides["poll_interval_seconds"] = (int(os.getenv("POLL_INTERVAL", "60")), "env:POLL_INTERVAL")
    return overrides


def _compute_settings_hash(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


class EffectiveSettings:
    """Resolved runtime settings with source tracking and hidden thresholds."""

    def __init__(self, raw: dict[str, Any] | None = None) -> None:
        self._raw = raw if raw is not None else db.get_settings()
        self._resolved: dict[str, Any] = {}
        self._sources: dict[str, str] = {}
        self._aliases_resolved: dict[str, Any] = {}
        self._build()

    def _build(self) -> None:
        base = {**db._DEFAULT_SETTINGS, **_EXTENDED_DEFAULTS}  # type: ignore[attr-defined]
        merged = deepcopy(base)
        sources: dict[str, str] = {k: "default" for k in base}

        # Apply stored settings (canonical keys)
        for key, value in self._raw.items():
            if key in SETTING_ALIASES:
                canonical = SETTING_ALIASES[key]
                self._aliases_resolved[key] = value
                merged[canonical] = value
                sources[canonical] = f"alias:{key}"
            elif key in CANONICAL_KEYS or key in base:
                merged[key] = value
                sources[key] = "settings.json"

        # Sync mode ↔ trading_mode
        if merged.get("mode") and not self._raw.get("trading_mode"):
            merged["trading_mode"] = merged["mode"]
            sources["trading_mode"] = sources.get("mode", "alias:mode")
        if merged.get("trading_mode"):
            merged["mode"] = merged["trading_mode"]

        # Env overrides
        for key, (value, source) in _env_overrides().items():
            merged[key] = value
            sources[key] = source

        # Legacy alias: max_model_snapshot_age_seconds → max_market_snapshot_age_seconds
        legacy_snap = merged.get("max_model_snapshot_age_seconds")
        if legacy_snap is not None:
            merged.setdefault("max_market_snapshot_age_seconds", legacy_snap)

        self._resolved = normalize_canonical_settings(normalize_execution_settings(merged))
        self._sources = sources

    @property
    def canonical(self) -> dict[str, Any]:
        return dict(self._resolved)

    @property
    def sources(self) -> dict[str, str]:
        return dict(self._sources)

    @property
    def aliases_resolved(self) -> dict[str, Any]:
        return dict(self._aliases_resolved)

    @property
    def defaults(self) -> dict[str, Any]:
        return {**db._DEFAULT_SETTINGS, **_EXTENDED_DEFAULTS}  # type: ignore[attr-defined]

    @property
    def hidden_thresholds(self) -> dict[str, Any]:
        """Hard-coded engine/live gates made explicit (Phase 1 — read-only)."""
        from ..llm_config import get_ollama_max_calls_per_scan

        live_min_whale = float(os.getenv("MIN_WHALE_SCORE", "0.30"))
        return {
            "generate_signal": {
                "buy_prob_threshold": SIGNAL_BUY_PROB_THRESHOLD,
                "buy_whale_score_threshold": SIGNAL_BUY_WHALE_THRESHOLD,
                "buy_liquidity_usd": SIGNAL_BUY_LIQUIDITY_USD,
                "watch_prob_threshold": SIGNAL_WATCH_PROB_THRESHOLD,
                "watch_whale_score_threshold": SIGNAL_WATCH_WHALE_THRESHOLD,
                "source": "engine.generate_signal",
            },
            "detect_whale_alert": {
                "min_volume_24h": WHALE_ALERT_MIN_VOLUME_24H,
                "min_whale_score": WHALE_ALERT_MIN_WHALE_SCORE,
                "source": "engine.detect_whale_alert",
            },
            "live_scan_gates": {
                "min_liquidity_usd_effective": float(self._resolved.get("min_liquidity_usd", 5000)),
                "min_whale_score_effective": float(self._resolved.get("min_whale_score", live_min_whale)),
                "llm_score_threshold": float(self._resolved.get("llm_score_threshold", 0.50)),
                "llm_requires_whale_alert": True,
                "auto_execution_enabled": bool(self._resolved.get("auto_execution_enabled", True)),
                "trading_mode": str(self._resolved.get("trading_mode", "DEMO")),
                "live_trading_blocked": str(self._resolved.get("trading_mode", "DEMO")) != "DEMO",
                "source": "live.scan_once",
            },
            "llm_budget": {
                "max_llm_calls_per_scan": int(self._resolved.get("max_llm_calls_per_scan", get_ollama_max_calls_per_scan())),
                "ollama_max_calls_per_scan_env": get_ollama_max_calls_per_scan(),
                "source": "llm_config",
            },
            "economic_gate": {
                "economic_gate_enabled": bool(self._resolved.get("economic_gate_enabled", False)),
                "paper_trading_enabled": bool(self._resolved.get("paper_trading_enabled", True)),
                "live_trading_enabled": bool(self._resolved.get("live_trading_enabled", False)),
                "demo_aggressive_enabled": bool(self._resolved.get("demo_aggressive_enabled", False)),
                "rf_probability_threshold": float(self._resolved.get("rf_probability_threshold", 0.70)),
                "rf_gate_enabled": bool(self._resolved.get("rf_gate_enabled", True)),
                "source": "phase2.economic_gate",
            },
            "tab_confidence_boost": {
                "tab_confidence_boost_enabled": bool(self._resolved.get("tab_confidence_boost_enabled", False)),
                "tab_confidence_boost_enabled_demo": bool(self._resolved.get("tab_confidence_boost_enabled_demo", False)),
                "tab_confidence_boost_enabled_live": bool(self._resolved.get("tab_confidence_boost_enabled_live", False)),
                "tab_confidence_suffix": str(self._resolved.get("tab_confidence_suffix", "nearest_neighbors_context_4096")),
                "tab_position_size_multiplier": float(self._resolved.get("tab_position_size_multiplier", 1.5)),
                "tab_standalone_trading_enabled": bool(self._resolved.get("tab_standalone_trading_enabled", False)),
                "tab_rescue_enabled": bool(self._resolved.get("tab_rescue_enabled", False)),
                "source": "phase2.tab_boost",
            },
            "slippage_and_drift": {
                "max_slippage_pct": float(self._resolved.get("max_slippage_pct", 0.015)),
                "baseline_slippage_pct": float(self._resolved.get("baseline_slippage_pct", 0.015)),
                "round_trip_fee_pct": float(self._resolved.get("round_trip_fee_pct", 0.03)),
                "max_price_drift_from_model_pct": float(self._resolved.get("max_price_drift_from_model_pct", 0.01)),
                "min_market_read_interval_seconds": float(self._resolved.get("min_market_read_interval_seconds", 60)),
                "max_market_snapshot_age_seconds": float(self._resolved.get("max_market_snapshot_age_seconds", 300)),
                "max_model_prediction_age_seconds": float(self._resolved.get("max_model_prediction_age_seconds", 300)),
                "max_model_artifact_age_hours": float(self._resolved.get("max_model_artifact_age_hours", 168)),
                "whale_wave_lookback_minutes": self._resolved.get("whale_wave_lookback_minutes", [5, 15, 60, 240]),
                "source": "phase2.costs",
            },
            "api_refresh_hardcoded": {
                "min_liquidity_usd": 5000.0,
                "note": "POST /api/coins/refresh uses hard-coded MIN_LIQ=5000",
                "source": "api.refresh_coins",
            },
        }

    @property
    def settings_hash(self) -> str:
        payload = {
            "canonical": self.canonical,
            "hidden_thresholds": self.hidden_thresholds,
        }
        return _compute_settings_hash(payload)

    def to_api_response(self) -> dict[str, Any]:
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "settings_hash": self.settings_hash,
            "canonical": self.canonical,
            "aliases_resolved": self.aliases_resolved,
            "defaults": self.defaults,
            "sources": self.sources,
            "hidden_thresholds": self.hidden_thresholds,
        }

    def write_audit_report(self) -> Path:
        ts = utc_timestamp_slug()
        path = write_json_report_atomic(
            f"settings_effective_{ts}.json",
            self.to_api_response(),
        )
        return path


def get_effective_settings(raw: dict[str, Any] | None = None) -> EffectiveSettings:
    return EffectiveSettings(raw)
