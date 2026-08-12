"""Demo presets (paper/demo only) — hold horizons + trade volume bounds."""

from __future__ import annotations

from typing import Any

# Strategy lane ids used by the continuous demo bot
STRATEGY_LANES: list[dict[str, Any]] = [
    {"id": "momentum_scout", "label": "Momentum Scout", "enabled_default": True},
    {"id": "liquidity_whale_scout", "label": "Liquidity/Whale Scout", "enabled_default": True},
    {"id": "meme_opportunistic_scout", "label": "Meme Opportunistic Scout", "enabled_default": True},
    {"id": "lotto_scout", "label": "Lotto Scout", "enabled_default": False},
    {"id": "rss_sentiment_watcher", "label": "RSS/Sentiment Watcher", "enabled_default": True},
    {"id": "manual_watchlist_scout", "label": "Manual Watchlist Scout", "enabled_default": True},
]

PRESETS: dict[str, dict[str, Any]] = {
    "reasoning_demo": {
        "id": "reasoning_demo",
        "label": "Reasoning Demo",
        "summary": "Reasoning/context arm. Independent from model consensus. Blocks missing/conflicted/scam/stale context.",
        "max_open_positions": 2,
        "max_trades_per_hour": 4,
        "max_notional_usd": 25.0,
        "cooldown_seconds": 900,
        "exploration_enabled": False,
        "demo_acceptance_mode": False,
        "min_hold_seconds": 1800,
        "time_stop_seconds": 172800,
        "take_profit_pct": 0.40,
        "stop_loss_pct": 0.12,
        "trailing_stop_pct": 0.15,
        "expected_hold_profile": "reasoning_demo_30m_to_48h",
        "lanes_enabled": [
            "momentum_scout",
            "liquidity_whale_scout",
            "meme_opportunistic_scout",
            "rss_sentiment_watcher",
            "manual_watchlist_scout",
        ],
    },
    "strict_consensus": {
        "id": "strict_consensus",
        "label": "Strict Consensus Demo",
        "summary": "Consensus-only entries: TAB_XGB_RF_ALL3 / TAB_RF_ONLY. No exploration.",
        "max_open_positions": 2,
        "max_trades_per_hour": 4,
        "max_notional_usd": 25.0,
        "cooldown_seconds": 900,
        "exploration_enabled": False,
        "demo_acceptance_mode": False,
        "min_hold_seconds": 1800,
        "time_stop_seconds": 172800,
        "take_profit_pct": 0.40,
        "stop_loss_pct": 0.12,
        "trailing_stop_pct": 0.15,
        "expected_hold_profile": "strict_consensus_30m_to_48h",
        "strict_consensus_only": True,
        "strict_consensus_allowed_tiers": ["TAB_XGB_RF_ALL3", "TAB_RF_ONLY"],
        "lanes_enabled": [
            "momentum_scout",
            "liquidity_whale_scout",
            "meme_opportunistic_scout",
            "manual_watchlist_scout",
        ],
    },
    "conservative": {
        "id": "conservative",
        "label": "Conservative Demo",
        "summary": "Low trade frequency, stricter gates, longer min hold.",
        "max_open_positions": 2,
        "max_trades_per_hour": 6,
        "max_notional_usd": 50.0,
        "cooldown_seconds": 120,
        "exploration_enabled": False,
        "demo_acceptance_mode": False,
        "min_hold_seconds": 600,  # 10 min
        "time_stop_seconds": 7200,  # 2h
        "take_profit_pct": 0.12,
        "stop_loss_pct": 0.06,
        "trailing_stop_pct": 0.08,
        "expected_hold_profile": "conservative_10m_to_4h",
        "lanes_enabled": [
            "momentum_scout",
            "liquidity_whale_scout",
            "rss_sentiment_watcher",
            "manual_watchlist_scout",
        ],
    },
    "balanced": {
        "id": "balanced",
        "label": "Balanced Demo",
        "summary": "Moderate trade frequency, paper-only exploration, still bounded.",
        "max_open_positions": 3,
        "max_trades_per_hour": 12,
        "max_notional_usd": 75.0,
        "cooldown_seconds": 60,
        "exploration_enabled": True,
        "demo_acceptance_mode": False,
        "min_hold_seconds": 300,  # 5 min
        "time_stop_seconds": 21600,  # 6h
        "take_profit_pct": 0.18,
        "stop_loss_pct": 0.08,
        "trailing_stop_pct": 0.10,
        "expected_hold_profile": "balanced_5m_to_12h",
        "lanes_enabled": [
            "momentum_scout",
            "liquidity_whale_scout",
            "meme_opportunistic_scout",
            "rss_sentiment_watcher",
            "manual_watchlist_scout",
        ],
    },
    "aggressive": {
        "id": "aggressive",
        "label": "Aggressive Demo",
        "summary": "Higher demo volume, shorter min hold, longer time-stop for upside.",
        "max_open_positions": 6,
        "max_trades_per_hour": 30,
        "max_notional_usd": 100.0,
        "cooldown_seconds": 30,
        "exploration_enabled": True,
        "demo_acceptance_mode": False,
        "min_hold_seconds": 180,  # 3 min
        "time_stop_seconds": 43200,  # 12h
        "take_profit_pct": 0.25,
        "stop_loss_pct": 0.10,
        "trailing_stop_pct": 0.12,
        "expected_hold_profile": "aggressive_2m_to_24h",
        "lanes_enabled": [
            "momentum_scout",
            "liquidity_whale_scout",
            "meme_opportunistic_scout",
            "lotto_scout",
            "rss_sentiment_watcher",
            "manual_watchlist_scout",
        ],
    },
    "lotto": {
        "id": "lotto",
        "label": "Lotto Scout",
        "summary": "Very small notional, long hold windows, high-volatility tolerant (demo only).",
        "max_open_positions": 8,
        "max_trades_per_hour": 20,
        "max_notional_usd": 25.0,
        "cooldown_seconds": 45,
        "exploration_enabled": True,
        "demo_acceptance_mode": False,
        "min_hold_seconds": 900,  # 15 min
        "time_stop_seconds": 86400,  # 24h
        "take_profit_pct": 0.50,
        "stop_loss_pct": 0.20,
        "trailing_stop_pct": 0.15,
        "expected_hold_profile": "lotto_15m_to_48h",
        "lanes_enabled": [
            "lotto_scout",
            "meme_opportunistic_scout",
            "manual_watchlist_scout",
        ],
    },
    "acceptance": {
        "id": "acceptance",
        "label": "Demo Acceptance / Test Mode",
        "summary": "For UI/ledger testing only. Not strategy evidence. Not profitability evidence.",
        "max_open_positions": 1,
        "max_trades_per_hour": 4,
        "max_notional_usd": 25.0,
        "cooldown_seconds": 15,
        "exploration_enabled": True,
        "demo_acceptance_mode": True,
        "min_hold_seconds": 5,
        "time_stop_seconds": 60,
        "take_profit_pct": 0.05,
        "stop_loss_pct": 0.05,
        "trailing_stop_pct": 0.05,
        "expected_hold_profile": "acceptance_test_only",
        "lanes_enabled": ["momentum_scout"],
    },
}

# Absolute ceiling for user-configurable paper/demo trades per hour
MAX_TRADES_PER_HOUR_CAP = 50


def list_presets() -> list[dict[str, Any]]:
    return [dict(v) for v in PRESETS.values()]


def get_preset(preset_id: str) -> dict[str, Any]:
    key = str(preset_id or "balanced").lower().strip()
    if key not in PRESETS:
        key = "balanced"
    return dict(PRESETS[key])


def list_strategy_lanes() -> list[dict[str, Any]]:
    return [dict(v) for v in STRATEGY_LANES]


def clamp_trades_per_hour(value: Any, *, default: int = 12) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = default
    return max(1, min(MAX_TRADES_PER_HOUR_CAP, n))
