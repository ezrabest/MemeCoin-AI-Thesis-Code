"""AE11 runtime paper loop types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

AE11_PHASE = "AE11_RUNTIME_PAPER_LOOP"
AE11_SCHEMA_VERSION = "AE11_V1"
AE11B_SCHEMA_VERSION = "AE11B_v1"

RUNTIME_INFERENCE_STATUS = "BLOCKED_NOT_APPROVED"
TRADING_AUTHORIZATION_STATUS = "PAPER_EXPLORATION_ONLY"

EXPLORATION_OVERRIDE_TYPE = "DEMO_ONLY_USER_APPROVED_EXPLORATION"
EXPLORATION_TRADE_AUTHORITY = "PAPER_EXPLORATION_ONLY"


class DuplicateReason(StrEnum):
    DUPLICATE_DECISION_ID = "DUPLICATE_DECISION_ID"
    ACTIVE_PAIR_LOCK = "ACTIVE_PAIR_LOCK"
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
    ALREADY_OPEN_POSITION = "ALREADY_OPEN_POSITION"
    STALE_REPLAYED_DECISION = "STALE_REPLAYED_DECISION"


class Ae11FinalStatus(StrEnum):
    AE11_LOOP_OPERATIONAL = "AE11_LOOP_OPERATIONAL"
    AE11_LOOP_PARTIAL = "AE11_LOOP_PARTIAL"
    AE11_LOOP_BLOCKED_NO_INPUT = "AE11_LOOP_BLOCKED_NO_INPUT"
    AE11_LOOP_BLOCKED_SAFETY = "AE11_LOOP_BLOCKED_SAFETY"
    AE11_STATE_RECONSTRUCTION_MISMATCH = "AE11_STATE_RECONSTRUCTION_MISMATCH"
    AE11_LEDGER_INVARIANT_FAILED = "AE11_LEDGER_INVARIANT_FAILED"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Ae11LoopConfig:
    project_root: Any = None
    duration_minutes: float = 60.0
    loop_interval_seconds: float = 30.0
    enable_paper_demo_orders: bool = False
    allow_paper_trades_with_audit_blockers: bool = False
    exploration_mode: bool = False
    strict_shadow_mode: bool = True
    enable_live_dry_run: bool = False
    starting_balance_usd: float = 10_000.0
    notional_usd: float = 100.0
    max_open_positions: int = 10
    per_pair_cooldown_minutes: float = 30.0
    max_price_age_seconds: float = 900.0
    strict_shadow_max_price_age_seconds: float = 30.0
    max_scan_records_per_iteration: int = 5000
    take_profit_pct: float = 20.0
    stop_loss_pct: float = 10.0
    time_stop_minutes: float = 240.0
    entry_fee_bps: float | None = None
    exit_fee_bps: float | None = None
    fee_bps: float = 30.0
    slippage_bps: float = 50.0
    provider: str = "mock"
    allow_local_qwen: bool = False
    allow_ollama: bool = False
    allow_gemini: bool = False
    max_records_per_iteration: int = 50
    heartbeat_every_n_iterations: int = 5
    no_real_wallet: bool = True
    allow_duplicate_pair: bool = False
    allow_negative_cash: bool = False
    resume_loop_run_id: str | None = None
    resume_latest_loop: bool = False
    # AE11I valuation / deterministic TP-SL proof (no external APIs)
    valuation_provider: str = "legacy"  # legacy | deterministic | local_snapshot
    deterministic_price_scenario: str = "neutral"
    deterministic_price_bump_pct: float = 25.0
    deterministic_price_drop_pct: float = 15.0
    deterministic_price_step_pct: float = 5.0
    price_lifecycle_proof_mode: bool = False

    def resolved_entry_fee_bps(self) -> float:
        return float(self.entry_fee_bps if self.entry_fee_bps is not None else self.fee_bps)

    def resolved_exit_fee_bps(self) -> float:
        return float(self.exit_fee_bps if self.exit_fee_bps is not None else self.fee_bps)


@dataclass
class Ae11BaseRecord:
    record_type: str = ""
    schema_version: str = AE11_SCHEMA_VERSION
    created_at_utc: str = field(default_factory=utc_now_iso)
    loop_run_id: str = ""
    loop_iteration: int = 0
    source_decision_id: str | None = None
    candidate_id: str | None = None
    pair_address: str | None = None

    def base_dict(self) -> dict[str, Any]:
        return {
            "record_type": self.record_type,
            "schema_version": self.schema_version,
            "created_at_utc": self.created_at_utc,
            "loop_run_id": self.loop_run_id,
            "loop_iteration": self.loop_iteration,
            "source_decision_id": self.source_decision_id,
            "candidate_id": self.candidate_id,
            "pair_address": self.pair_address,
        }


@dataclass
class OpportunityCaptureRecord(Ae11BaseRecord):
    record_type: str = "OPPORTUNITY_CAPTURE"
    first_seen_timestamp: str = field(default_factory=utc_now_iso)
    source_context_record_id: str | None = None
    source_llm_audit_record_id: str | None = None
    price_at_first_seen: float | None = None
    liquidity_at_first_seen: float | None = None
    whale_score_at_first_seen: float | None = None
    volume_at_first_seen: float | None = None
    ae6_decision_status: str | None = None
    ae8_context_status: str | None = None
    ae9_audit_verdict: str | None = None
    ae9_audit_blockers: list[str] = field(default_factory=list)
    paper_action_taken: str | None = None
    reason_for_no_trade: str | None = None
    strict_shadow_decision: str | None = None
    exploration_decision: str | None = None
    paper_order_id: str | None = None
    position_id: str | None = None
    max_return_5m: float | None = None
    max_return_15m: float | None = None
    max_return_1h: float | None = None
    max_return_6h: float | None = None
    max_return_24h: float | None = None
    horizon_matured_5m: bool = False
    horizon_matured_15m: bool = False
    horizon_matured_1h: bool = False
    horizon_matured_6h: bool = False
    horizon_matured_24h: bool = False
    outcome_computed_at: str | None = None
    outcome_source_snapshot_count: int = 0
    blocked_by_ae9: bool = False
    stale_price: bool = False
    missing_context: bool = False
    max_open_positions_hit: bool = False
    cooldown_active: bool = False
    duplicate_active_pair: bool = False
    missing_identity: bool = False
    paper_policy_prevented: bool = False
    duplicate_reason: str | None = None
    # AE12-SentimentFix additive dual-axis fields (optional; old consumers ignore)
    semantic_signal_family: str = "UNKNOWN"
    semantic_signal_source: str = "none"
    semantic_signal_confidence: float = 0.0
    semantic_signal_reason: str = "not_evaluated"
    trading_opportunity_state: str = "UNKNOWN"
    trading_state_source: str = "none"
    legacy_cluster_label: str | None = None
    taxonomy_status: str = "UNKNOWN_NOT_EVALUATED"

    def to_dict(self) -> dict[str, Any]:
        d = self.base_dict()
        d.update(
            {
                "first_seen_timestamp": self.first_seen_timestamp,
                "source_context_record_id": self.source_context_record_id,
                "source_llm_audit_record_id": self.source_llm_audit_record_id,
                "price_at_first_seen": self.price_at_first_seen,
                "liquidity_at_first_seen": self.liquidity_at_first_seen,
                "whale_score_at_first_seen": self.whale_score_at_first_seen,
                "volume_at_first_seen": self.volume_at_first_seen,
                "ae6_decision_status": self.ae6_decision_status,
                "ae8_context_status": self.ae8_context_status,
                "ae9_audit_verdict": self.ae9_audit_verdict,
                "ae9_audit_blockers": self.ae9_audit_blockers,
                "paper_action_taken": self.paper_action_taken,
                "reason_for_no_trade": self.reason_for_no_trade,
                "strict_shadow_decision": self.strict_shadow_decision,
                "exploration_decision": self.exploration_decision,
                "paper_order_id": self.paper_order_id,
                "position_id": self.position_id,
                "max_return_5m": self.max_return_5m,
                "max_return_15m": self.max_return_15m,
                "max_return_1h": self.max_return_1h,
                "max_return_6h": self.max_return_6h,
                "max_return_24h": self.max_return_24h,
                "horizon_matured_5m": self.horizon_matured_5m,
                "horizon_matured_15m": self.horizon_matured_15m,
                "horizon_matured_1h": self.horizon_matured_1h,
                "horizon_matured_6h": self.horizon_matured_6h,
                "horizon_matured_24h": self.horizon_matured_24h,
                "outcome_computed_at": self.outcome_computed_at,
                "outcome_source_snapshot_count": self.outcome_source_snapshot_count,
                "blocked_by_ae9": self.blocked_by_ae9,
                "stale_price": self.stale_price,
                "missing_context": self.missing_context,
                "max_open_positions_hit": self.max_open_positions_hit,
                "cooldown_active": self.cooldown_active,
                "duplicate_active_pair": self.duplicate_active_pair,
                "missing_identity": self.missing_identity,
                "paper_policy_prevented": self.paper_policy_prevented,
                "duplicate_reason": self.duplicate_reason,
                "semantic_signal_family": self.semantic_signal_family,
                "semantic_signal_source": self.semantic_signal_source,
                "semantic_signal_confidence": self.semantic_signal_confidence,
                "semantic_signal_reason": self.semantic_signal_reason,
                "trading_opportunity_state": self.trading_opportunity_state,
                "trading_state_source": self.trading_state_source,
                "legacy_cluster_label": self.legacy_cluster_label,
                "taxonomy_status": self.taxonomy_status,
            }
        )
        return d


@dataclass
class ReconstructedAccountState:
    cash_balance_usd: float = 0.0
    reserved_cash_usd: float = 0.0
    open_positions: list[dict[str, Any]] = field(default_factory=list)
    closed_positions: list[dict[str, Any]] = field(default_factory=list)
    active_pair_locks: dict[str, str] = field(default_factory=dict)
    cooldowns: dict[str, str] = field(default_factory=dict)
    processed_decision_ids: set[str] = field(default_factory=set)
    realized_pnl_usd: float = 0.0
    gross_pnl_usd: float = 0.0
    net_pnl_usd: float = 0.0
    reconstruction_status: str = "OK"
    mismatches: list[str] = field(default_factory=list)
    no_wallet_path: bool = True
