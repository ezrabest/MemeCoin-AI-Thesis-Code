"""Runtime trade candidate object for economic gate decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .audit_io import new_decision_trace_id


@dataclass
class TradeCandidate:
    """Single runtime decision input — event_timestamp is audit-only, not an ML feature."""

    pair_address: str
    chain: str
    symbol: str
    price: float
    liquidity_usd: float
    whale_score: float
    signal_score: float
    signal_type: str
    decision_trace_id: str = field(default_factory=new_decision_trace_id)
    event_timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    coin_id: int | None = None
    volume_5m: float | None = None
    volume_15m: float | None = None
    volume_1h: float | None = None
    volume_24h: float | None = None
    buy_count: int | None = None
    sell_count: int | None = None
    buy_ratio: float | None = None
    alert_type: str | None = None
    bearish_alert_active: bool = False
    existing_open_position_for_pair: bool = False
    recent_duplicate_event: bool = False
    sentiment_available: bool = False
    sentiment_score: float | None = None
    sentiment_age_minutes: float | None = None
    # Legacy single-axis label (audit comparison only after AE12-SentimentFix).
    # Prefer semantic_signal_family + trading_opportunity_state for taxonomy.
    cluster_label: str = "OPPORTUNISTIC_SPECULATIVE"
    cluster_confidence: float | None = None
    cluster_is_default: bool = True
    # AE12-SentimentFix additive dual-axis fields (defaults never invent social/opportunistic semantics)
    semantic_signal_family: str = "UNKNOWN"
    semantic_signal_source: str = "none"
    semantic_signal_confidence: float = 0.0
    semantic_signal_reason: str = "not_evaluated"
    trading_opportunity_state: str = "UNKNOWN"
    trading_state_source: str = "none"
    legacy_cluster_label: str | None = None
    taxonomy_status: str = "UNKNOWN_NOT_EVALUATED"
    rf_prediction: dict[str, Any] | None = None
    tab_prediction: dict[str, Any] | None = None
    model_metadata: dict[str, Any] | None = None
    model_snapshot_price: float | None = None
    current_execution_price: float | None = None
    price_drift_from_model_pct: float | None = None
    max_price_drift_from_model_pct: float | None = None
    estimated_slippage_per_side_pct: float | None = None
    round_trip_slippage_pct: float | None = None
    total_cost_pct: float | None = None
    expected_net_return: float | None = None
    probability_profitable_after_costs: float | None = None
    actionability_decision: str | None = None
    audit_reasons: list[str] = field(default_factory=list)
    settings_hash: str | None = None
    whale_wave_features: dict[str, Any] | None = None
    scan_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}
