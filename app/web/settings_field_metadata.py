"""Metadata for System Configuration form fields and inspector rows."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    group: str
    kind: str
    consumer: str = ""
    notes: str = ""
    read_only: bool = False
    select_options: tuple[str, ...] | None = None
    step: float = 0.1
    decimals: int = 4


FIELD_SPECS: list[FieldSpec] = [
    FieldSpec("economic_gate_enabled", "Economic Gate", "gates", "bool", "economic_gate / RF approval"),
    FieldSpec("demo_aggressive_enabled", "Demo Aggressive", "gates", "bool", "economic_gate demo path"),
    FieldSpec("paper_trading_enabled", "Paper Trading", "gates", "bool", "paper execution"),
    FieldSpec(
        "demo_acceptance_mode",
        "DEMO Acceptance Mode",
        "gates",
        "bool",
        "ae13 demo acceptance",
        notes="Off by default. Bounded paper-only UI wiring proof — not strategy/profitability evidence.",
    ),
    FieldSpec("allow_watch_to_buy_promotion", "Watch→Buy Promotion", "gates", "bool", "actionability"),
    FieldSpec("rf_gate_enabled", "RF Gate", "gates", "bool", "model_runtime_inference"),
    FieldSpec("rf_probability_threshold", "RF Probability Threshold", "gates", "number", "RF gate", notes="Internal 0.70 = 70%", step=0.1, decimals=2),
    FieldSpec("tab_confidence_boost_enabled", "TAB Confidence Boost", "tab", "bool", "TAB overlay"),
    FieldSpec("tab_confidence_boost_enabled_demo", "TAB DEMO Boost", "tab", "bool", "TAB overlay demo"),
    FieldSpec("tab_confidence_boost_enabled_live", "TAB LIVE Boost", "tab", "bool", "TAB overlay live"),
    FieldSpec("tab_confidence_suffix", "TAB Confidence Suffix", "tab", "select", "TAB model lookup", select_options=("nearest_neighbors_context_4096", "nearest_neighbors_context_2048")),
    FieldSpec("tab_confidence_percentile_threshold", "TAB Percentile Threshold", "tab", "number", "TAB overlay", notes="Internal 0.98 = 98th percentile", step=0.1, decimals=2),
    FieldSpec("tab_position_size_multiplier", "TAB Position Size Multiplier", "tab", "number", "TAB overlay sizing", step=0.1, decimals=2),
    FieldSpec("tab_standalone_trading_enabled", "TAB Standalone Trading", "tab", "bool", "blocked — overlay only", read_only=True),
    FieldSpec("tab_rescue_enabled", "TAB Rescue", "tab", "bool", "blocked — overlay only", read_only=True),
    FieldSpec("max_position_size_pct", "Max Position Size", "costs", "number", "economic gate / risk"),
    FieldSpec("stop_loss_pct", "Stop Loss", "costs", "number", "economic gate / exits"),
    FieldSpec("take_profit_pct", "Take Profit", "costs", "number", "economic gate / exits"),
    FieldSpec("max_slippage_pct", "Max Slippage", "costs", "number", "slippage / economic gate"),
    FieldSpec("baseline_slippage_pct", "Baseline Slippage", "costs", "number", "slippage model"),
    FieldSpec("dynamic_slippage_enabled", "Dynamic Slippage", "costs", "bool", "slippage model"),
    FieldSpec("max_price_drift_from_model_pct", "Max Price Drift", "costs", "number", "economic gate"),
    FieldSpec("round_trip_fee_pct", "Round-Trip Fee", "costs", "number", "economic gate costs"),
    FieldSpec("required_margin_after_costs_pct", "Required Margin After Costs", "costs", "number", "economic gate"),
    FieldSpec("min_liquidity_usd", "Min Liquidity", "costs", "number", "live scan / economic gate"),
    FieldSpec("max_open_positions", "Max Open Positions", "costs", "int", "paper / risk"),
    FieldSpec("duplicate_pair_guard_enabled", "Duplicate Pair Guard", "costs", "bool", "paper execution"),
    FieldSpec("llm_enabled_for_demo", "LLM Enabled (Demo)", "llm", "bool", "llm_gate"),
    FieldSpec("llm_enabled_for_live", "LLM Enabled (Live)", "llm", "bool", "llm_gate"),
    FieldSpec("max_llm_calls_per_hour", "Max LLM Calls / Hour", "llm", "int", "llm_gate budget"),
    FieldSpec("max_llm_calls_per_scan", "Max LLM Calls / Scan", "llm", "int", "llm_gate budget"),
    FieldSpec("llm_cache_window_minutes", "LLM Cache Window (min)", "llm", "int", "llm_gate cache"),
    FieldSpec("live_trading_enabled", "LIVE Trading", "safety", "bool", "live execution", read_only=True),
    FieldSpec("auto_execution_enabled", "Auto Execution", "safety", "bool", "live.scan_once"),
    FieldSpec("enforce_risk_gate", "Enforce Risk Gate", "safety", "bool", "risk gate"),
    FieldSpec("trading_mode", "Trading Mode", "safety", "readonly", "live.scan_once", read_only=True),
    FieldSpec("mode", "Mode Alias", "safety", "readonly", "alias:mode", read_only=True),
    FieldSpec("prompt_behavior", "Prompt Behavior", "safety", "select", "LLM prompts", select_options=("conservative", "balanced", "aggressive")),
    FieldSpec("cooldown_minutes", "Cooldown (minutes)", "safety", "int", "risk"),
    FieldSpec("max_daily_loss_pct", "Max Daily Loss", "safety", "number", "risk gate"),
    FieldSpec("max_drawdown_pct", "Max Drawdown", "safety", "number", "risk gate"),
]

FIELD_SPEC_BY_KEY: dict[str, FieldSpec] = {f.key: f for f in FIELD_SPECS}

GROUP_TITLES: dict[str, str] = {
    "gates": "Trading Gates & Actionability",
    "tab": "Model & TabICL Overlay",
    "costs": "Costs, Slippage & Risk",
    "llm": "LLM Controls",
    "safety": "Runtime Safety",
}

EDITABLE_KEYS: set[str] = {f.key for f in FIELD_SPECS if not f.read_only and f.kind != "readonly"}

INSPECTOR_COLUMNS: tuple[str, ...] = (
    "ui_label",
    "canonical_key",
    "displayed_value",
    "internal_value",
    "unit",
    "source",
    "default_value",
    "alias_resolved",
    "active_status",
    "backend_consumer",
    "notes_warnings",
)
