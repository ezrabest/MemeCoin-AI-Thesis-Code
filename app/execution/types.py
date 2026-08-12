"""AE10 execution layer types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal
from uuid import uuid4

from app.paper_trading.types import utc_now_iso


@dataclass
class OrderIntent:
    candidate_id: str
    symbol: str = ""
    pair_address: str = ""
    side: str = "BUY"
    order_type: str = "MARKET"
    notional_usd: float = 100.0
    requested_price_usd: float | None = None
    source_decision_id: str | None = None
    source_context_record_id: str | None = None
    source_llm_audit_record_id: str | None = None
    decision_status: str | None = None
    consensus_family: str | None = None
    context_schema_id: str | None = None
    audit_verdict: str | None = None
    audit_blockers: list[str] = field(default_factory=list)
    audit_warnings: list[str] = field(default_factory=list)
    scoring_policy_id: str | None = None
    decision_created_at_utc: str | None = None
    order_created_at_utc: str | None = None
    coin_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "symbol": self.symbol,
            "pair_address": self.pair_address,
            "side": self.side,
            "order_type": self.order_type,
            "notional_usd": self.notional_usd,
            "requested_price_usd": self.requested_price_usd,
            "source_decision_id": self.source_decision_id,
            "source_context_record_id": self.source_context_record_id,
            "source_llm_audit_record_id": self.source_llm_audit_record_id,
            "decision_status": self.decision_status,
            "consensus_family": self.consensus_family,
            "context_schema_id": self.context_schema_id,
            "audit_verdict": self.audit_verdict,
            "audit_blockers": self.audit_blockers,
            "audit_warnings": self.audit_warnings,
            "scoring_policy_id": self.scoring_policy_id,
            "decision_created_at_utc": self.decision_created_at_utc,
            "order_created_at_utc": self.order_created_at_utc,
            "coin_id": self.coin_id,
        }


@dataclass
class ExecutionResult:
    success: bool
    execution_mode: str
    live_submission_status: str | None = None
    wallet_required: bool = False
    wallet_configured: bool = False
    real_transaction_attempted: Literal[False] = False
    no_wallet_dry_run: bool = False
    no_live_submission: bool = True
    order_id: str | None = None
    message: str = ""
    record: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "execution_mode": self.execution_mode,
            "live_submission_status": self.live_submission_status,
            "wallet_required": self.wallet_required,
            "wallet_configured": self.wallet_configured,
            "real_transaction_attempted": self.real_transaction_attempted,
            "no_wallet_dry_run": self.no_wallet_dry_run,
            "no_live_submission": self.no_live_submission,
            "order_id": self.order_id,
            "message": self.message,
            "record": self.record,
        }


@dataclass
class LiveDryRunOrder:
    live_order_id: str = field(default_factory=lambda: str(uuid4()))
    candidate_id: str = ""
    symbol: str = ""
    pair_address: str = ""
    side: str = "BUY"
    order_type: str = "MARKET"
    notional_usd: float = 0.0
    requested_price_usd: float | None = None
    execution_mode: str = "LIVE_NO_WALLET_DRY_RUN"
    live_submission_status: str = "NOT_SUBMITTED_NO_WALLET"
    wallet_required: bool = True
    wallet_configured: bool = False
    real_transaction_attempted: Literal[False] = False
    no_wallet_dry_run: bool = True
    no_live_submission: bool = True
    source_decision_id: str | None = None
    source_context_record_id: str | None = None
    source_llm_audit_record_id: str | None = None
    decision_status: str | None = None
    consensus_family: str | None = None
    context_schema_id: str | None = None
    audit_verdict: str | None = None
    audit_blockers: list[str] = field(default_factory=list)
    audit_warnings: list[str] = field(default_factory=list)
    scoring_policy_id: str | None = None
    decision_created_at_utc: str | None = None
    order_created_at_utc: str = field(default_factory=utc_now_iso)
    execution_latency_ms: float | None = None
    execution_latency_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "live_order_id": self.live_order_id,
            "candidate_id": self.candidate_id,
            "symbol": self.symbol,
            "pair_address": self.pair_address,
            "side": self.side,
            "order_type": self.order_type,
            "notional_usd": self.notional_usd,
            "requested_price_usd": self.requested_price_usd,
            "execution_mode": self.execution_mode,
            "live_submission_status": self.live_submission_status,
            "wallet_required": self.wallet_required,
            "wallet_configured": self.wallet_configured,
            "real_transaction_attempted": self.real_transaction_attempted,
            "no_wallet_dry_run": self.no_wallet_dry_run,
            "no_live_submission": self.no_live_submission,
            "source_decision_id": self.source_decision_id,
            "source_context_record_id": self.source_context_record_id,
            "source_llm_audit_record_id": self.source_llm_audit_record_id,
            "decision_status": self.decision_status,
            "consensus_family": self.consensus_family,
            "context_schema_id": self.context_schema_id,
            "audit_verdict": self.audit_verdict,
            "audit_blockers": self.audit_blockers,
            "audit_warnings": self.audit_warnings,
            "scoring_policy_id": self.scoring_policy_id,
            "decision_created_at_utc": self.decision_created_at_utc,
            "order_created_at_utc": self.order_created_at_utc,
            "execution_latency_ms": self.execution_latency_ms,
            "execution_latency_status": self.execution_latency_status,
        }
