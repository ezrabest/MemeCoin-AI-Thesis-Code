"""AE10 paper trading types."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal
from uuid import uuid4

AE10_PHASE = "AE10_TRADING_ORCHESTRATION"

RUNTIME_INFERENCE_STATUS = "BLOCKED_NOT_APPROVED"
TRADING_AUTHORIZATION_STATUS = "PAPER_DEMO_AND_LIVE_DRY_RUN_ONLY"


class PaperOrderStatus(StrEnum):
    PAPER_PENDING = "PAPER_PENDING"
    PAPER_FILLED = "PAPER_FILLED"
    PAPER_REJECTED = "PAPER_REJECTED"
    PAPER_CANCELED = "PAPER_CANCELED"
    PAPER_CLOSED_TP = "PAPER_CLOSED_TP"
    PAPER_CLOSED_SL = "PAPER_CLOSED_SL"
    PAPER_CLOSED_TIME_STOP = "PAPER_CLOSED_TIME_STOP"
    PAPER_CLOSED_MANUAL = "PAPER_CLOSED_MANUAL"
    PAPER_EXPIRED = "PAPER_EXPIRED"


class PaperTradeDecisionStatus(StrEnum):
    DEMO_WATCH_ONLY = "DEMO_WATCH_ONLY"
    DEMO_PAPER_BUY_CANDIDATE = "DEMO_PAPER_BUY_CANDIDATE"
    DEMO_PAPER_REJECTED_AUDIT_BLOCKED = "DEMO_PAPER_REJECTED_AUDIT_BLOCKED"
    DEMO_PAPER_REJECTED_MISSING_PRICE = "DEMO_PAPER_REJECTED_MISSING_PRICE"
    DEMO_PAPER_REJECTED_MISSING_IDENTITY = "DEMO_PAPER_REJECTED_MISSING_IDENTITY"
    DEMO_PAPER_REJECTED_STATE_MACHINE = "DEMO_PAPER_REJECTED_STATE_MACHINE"
    DEMO_PAPER_FILLED = "DEMO_PAPER_FILLED"
    LIVE_NO_WALLET_DRY_RUN_NOT_SUBMITTED = "LIVE_NO_WALLET_DRY_RUN_NOT_SUBMITTED"


class PriceStatus(StrEnum):
    PRICE_OK = "PRICE_OK"
    PRICE_MISSING = "PRICE_MISSING"
    PRICE_STALE = "PRICE_STALE"
    PRICE_LOOKAHEAD_REJECTED = "PRICE_LOOKAHEAD_REJECTED"
    PRICE_PROVIDER_TIME_SKEW_REJECTED = "PRICE_PROVIDER_TIME_SKEW_REJECTED"
    PRICE_INVALID_ZERO_OR_NEGATIVE = "PRICE_INVALID_ZERO_OR_NEGATIVE"


class ExecutionLatencyStatus(StrEnum):
    OK = "OK"
    MISSING_DECISION_TIMESTAMP = "MISSING_DECISION_TIMESTAMP"
    NOT_FILLED = "NOT_FILLED"


class Ae10FinalStatus(StrEnum):
    AE10_TRACEABILITY_READY = "AE10_TRACEABILITY_READY"
    AE10_PAPER_DEMO_READY = "AE10_PAPER_DEMO_READY"
    AE10_PAPER_DEMO_PARTIAL_NO_ORDERS = "AE10_PAPER_DEMO_PARTIAL_NO_ORDERS"
    AE10_LIVE_DRY_RUN_WIRED_NO_WALLET = "AE10_LIVE_DRY_RUN_WIRED_NO_WALLET"
    AE10_BLOCKED_NO_INPUT_ARTIFACTS = "AE10_BLOCKED_NO_INPUT_ARTIFACTS"
    AE10_BLOCKED_MISSING_PRICE = "AE10_BLOCKED_MISSING_PRICE"
    AE10_BLOCKED_PRICE_LOOKAHEAD = "AE10_BLOCKED_PRICE_LOOKAHEAD"
    AE10_BLOCKED_TRACEABILITY_GAP = "AE10_BLOCKED_TRACEABILITY_GAP"
    AE10_BLOCKED_STATE_MACHINE = "AE10_BLOCKED_STATE_MACHINE"
    AE10_BLOCKED_REAL_WALLET_RISK = "AE10_BLOCKED_REAL_WALLET_RISK"
    AE10_BLOCKED_WITH_EXACT_REASONS = "AE10_BLOCKED_WITH_EXACT_REASONS"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DemoAccount:
    account_id: str = field(default_factory=lambda: str(uuid4()))
    starting_balance_usd: float = 10_000.0
    cash_balance_usd: float = 10_000.0
    equity_usd: float = 10_000.0
    open_position_count: int = 0
    closed_trade_count: int = 0
    realized_pnl_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    reset_count: int = 0
    created_at_utc: str = field(default_factory=utc_now_iso)
    updated_at_utc: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "account_id": self.account_id,
            "starting_balance_usd": self.starting_balance_usd,
            "cash_balance_usd": self.cash_balance_usd,
            "equity_usd": self.equity_usd,
            "open_position_count": self.open_position_count,
            "closed_trade_count": self.closed_trade_count,
            "realized_pnl_usd": self.realized_pnl_usd,
            "unrealized_pnl_usd": self.unrealized_pnl_usd,
            "reset_count": self.reset_count,
            "created_at_utc": self.created_at_utc,
            "updated_at_utc": self.updated_at_utc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DemoAccount:
        return cls(
            account_id=data.get("account_id", str(uuid4())),
            starting_balance_usd=float(data.get("starting_balance_usd", 10_000.0)),
            cash_balance_usd=float(data.get("cash_balance_usd", 10_000.0)),
            equity_usd=float(data.get("equity_usd", 10_000.0)),
            open_position_count=int(data.get("open_position_count", 0)),
            closed_trade_count=int(data.get("closed_trade_count", 0)),
            realized_pnl_usd=float(data.get("realized_pnl_usd", 0.0)),
            unrealized_pnl_usd=float(data.get("unrealized_pnl_usd", 0.0)),
            reset_count=int(data.get("reset_count", 0)),
            created_at_utc=data.get("created_at_utc", utc_now_iso()),
            updated_at_utc=data.get("updated_at_utc", utc_now_iso()),
        )


@dataclass
class TraceabilityRecord:
    traceability_id: str = field(default_factory=lambda: str(uuid4()))
    candidate_id: str = ""
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
    execution_mode: str = "PAPER_DEMO"
    no_wallet_dry_run: bool = False
    no_live_submission: bool = True
    traceability_status: str = "COMPLETE"
    created_at_utc: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "traceability_id": self.traceability_id,
            "candidate_id": self.candidate_id,
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
            "execution_mode": self.execution_mode,
            "no_wallet_dry_run": self.no_wallet_dry_run,
            "no_live_submission": self.no_live_submission,
            "traceability_status": self.traceability_status,
            "created_at_utc": self.created_at_utc,
        }


@dataclass
class PaperOrder:
    paper_order_id: str = field(default_factory=lambda: str(uuid4()))
    candidate_id: str = ""
    symbol: str = ""
    pair_address: str = ""
    side: str = "BUY"
    order_type: str = "MARKET"
    requested_price_usd: float | None = None
    filled_price_usd: float | None = None
    quantity: float = 0.0
    notional_usd: float = 0.0
    status: str = PaperOrderStatus.PAPER_PENDING.value
    created_at_utc: str = field(default_factory=utc_now_iso)
    filled_at_utc: str | None = None
    closed_at_utc: str | None = None
    source_decision_id: str | None = None
    source_context_record_id: str | None = None
    source_llm_audit_record_id: str | None = None
    decision_created_at_utc: str | None = None
    traceability_status: str = "COMPLETE"
    execution_latency_ms: float | None = None
    execution_latency_status: str | None = None
    no_live_trading: Literal[True] = True
    paper_trade_reason: str | None = None
    not_model_approved: bool = False
    not_live_approved: bool = True
    override_type: str | None = None
    decision_status: str | None = None
    consensus_family: str | None = None
    context_schema_id: str | None = None
    audit_verdict: str | None = None
    audit_blockers: list[str] = field(default_factory=list)
    audit_warnings: list[str] = field(default_factory=list)
    scoring_policy_id: str | None = None
    execution_mode: str = "PAPER_DEMO"
    no_wallet_dry_run: bool = False
    no_live_submission: bool = True
    price_source: str | None = None
    price_snapshot_id: int | str | None = None
    price_timestamp: str | None = None
    price_age_seconds: float | None = None
    max_price_age_seconds: float = 30.0
    price_status: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "paper_order_id": self.paper_order_id,
            "candidate_id": self.candidate_id,
            "symbol": self.symbol,
            "pair_address": self.pair_address,
            "side": self.side,
            "order_type": self.order_type,
            "requested_price_usd": self.requested_price_usd,
            "filled_price_usd": self.filled_price_usd,
            "quantity": self.quantity,
            "notional_usd": self.notional_usd,
            "status": self.status,
            "created_at_utc": self.created_at_utc,
            "filled_at_utc": self.filled_at_utc,
            "closed_at_utc": self.closed_at_utc,
            "source_decision_id": self.source_decision_id,
            "source_context_record_id": self.source_context_record_id,
            "source_llm_audit_record_id": self.source_llm_audit_record_id,
            "decision_created_at_utc": self.decision_created_at_utc,
            "traceability_status": self.traceability_status,
            "execution_latency_ms": self.execution_latency_ms,
            "execution_latency_status": self.execution_latency_status,
            "no_live_trading": self.no_live_trading,
            "paper_trade_reason": self.paper_trade_reason,
            "not_model_approved": self.not_model_approved,
            "not_live_approved": self.not_live_approved,
            "override_type": self.override_type,
            "decision_status": self.decision_status,
            "consensus_family": self.consensus_family,
            "context_schema_id": self.context_schema_id,
            "audit_verdict": self.audit_verdict,
            "audit_blockers": self.audit_blockers,
            "audit_warnings": self.audit_warnings,
            "scoring_policy_id": self.scoring_policy_id,
            "execution_mode": self.execution_mode,
            "no_wallet_dry_run": self.no_wallet_dry_run,
            "no_live_submission": self.no_live_submission,
            "price_source": self.price_source,
            "price_snapshot_id": self.price_snapshot_id,
            "price_timestamp": self.price_timestamp,
            "price_age_seconds": self.price_age_seconds,
            "max_price_age_seconds": self.max_price_age_seconds,
            "price_status": self.price_status,
        }


@dataclass
class PaperPosition:
    position_id: str = field(default_factory=lambda: str(uuid4()))
    paper_order_id: str = ""
    candidate_id: str = ""
    symbol: str = ""
    pair_address: str = ""
    side: str = "LONG"
    entry_price_usd: float = 0.0
    quantity: float = 0.0
    notional_usd: float = 0.0
    status: str = "OPEN"
    opened_at_utc: str = field(default_factory=utc_now_iso)
    closed_at_utc: str | None = None
    exit_price_usd: float | None = None
    realized_pnl_usd: float = 0.0
    source_decision_id: str | None = None
    source_context_record_id: str | None = None
    source_llm_audit_record_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_id": self.position_id,
            "paper_order_id": self.paper_order_id,
            "candidate_id": self.candidate_id,
            "symbol": self.symbol,
            "pair_address": self.pair_address,
            "side": self.side,
            "entry_price_usd": self.entry_price_usd,
            "quantity": self.quantity,
            "notional_usd": self.notional_usd,
            "status": self.status,
            "opened_at_utc": self.opened_at_utc,
            "closed_at_utc": self.closed_at_utc,
            "exit_price_usd": self.exit_price_usd,
            "realized_pnl_usd": self.realized_pnl_usd,
            "source_decision_id": self.source_decision_id,
            "source_context_record_id": self.source_context_record_id,
            "source_llm_audit_record_id": self.source_llm_audit_record_id,
        }


@dataclass
class PaperTradeRecord:
    trade_id: str = field(default_factory=lambda: str(uuid4()))
    paper_order_id: str = ""
    position_id: str = ""
    candidate_id: str = ""
    symbol: str = ""
    side: str = "BUY"
    entry_price_usd: float = 0.0
    exit_price_usd: float = 0.0
    quantity: float = 0.0
    notional_usd: float = 0.0
    realized_pnl_usd: float = 0.0
    close_reason: str = ""
    opened_at_utc: str = ""
    closed_at_utc: str = field(default_factory=utc_now_iso)
    execution_latency_ms: float | None = None
    execution_latency_status: str | None = None
    decision_created_at_utc: str | None = None
    order_created_at_utc: str | None = None
    filled_at_utc: str | None = None
    source_decision_id: str | None = None
    source_context_record_id: str | None = None
    source_llm_audit_record_id: str | None = None
    no_live_trading: Literal[True] = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "trade_id": self.trade_id,
            "paper_order_id": self.paper_order_id,
            "position_id": self.position_id,
            "candidate_id": self.candidate_id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price_usd": self.entry_price_usd,
            "exit_price_usd": self.exit_price_usd,
            "quantity": self.quantity,
            "notional_usd": self.notional_usd,
            "realized_pnl_usd": self.realized_pnl_usd,
            "close_reason": self.close_reason,
            "opened_at_utc": self.opened_at_utc,
            "closed_at_utc": self.closed_at_utc,
            "execution_latency_ms": self.execution_latency_ms,
            "execution_latency_status": self.execution_latency_status,
            "decision_created_at_utc": self.decision_created_at_utc,
            "order_created_at_utc": self.order_created_at_utc,
            "filled_at_utc": self.filled_at_utc,
            "source_decision_id": self.source_decision_id,
            "source_context_record_id": self.source_context_record_id,
            "source_llm_audit_record_id": self.source_llm_audit_record_id,
            "no_live_trading": self.no_live_trading,
        }


@dataclass
class PaperLedgerSnapshot:
    snapshot_id: str = field(default_factory=lambda: str(uuid4()))
    account_id: str = ""
    cash_balance_usd: float = 0.0
    equity_usd: float = 0.0
    open_position_count: int = 0
    closed_trade_count: int = 0
    realized_pnl_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0
    created_at_utc: str = field(default_factory=utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "account_id": self.account_id,
            "cash_balance_usd": self.cash_balance_usd,
            "equity_usd": self.equity_usd,
            "open_position_count": self.open_position_count,
            "closed_trade_count": self.closed_trade_count,
            "realized_pnl_usd": self.realized_pnl_usd,
            "unrealized_pnl_usd": self.unrealized_pnl_usd,
            "created_at_utc": self.created_at_utc,
        }
