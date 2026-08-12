"""AE12.7 canonical intelligent-agent record types (audit/explanation only)."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

AE12_7_PHASE = "AE12.7"
AE12_7_SCHEMA_VERSION = "AE12_7_AGENT_RECORD_V1"
OUTPUT_PREFIX = "ae12_7_intelligent_agent_operational_demo"


class AgentType(str, Enum):
    QWEN_LOCAL_MEMO = "QWEN_LOCAL_MEMO"
    GEMINI_SELECTIVE_AUDIT = "GEMINI_SELECTIVE_AUDIT"
    HELIUS_READONLY_ENRICHMENT = "HELIUS_READONLY_ENRICHMENT"
    RSS_CONTEXT = "RSS_CONTEXT"
    SEMANTIC_CLASSIFICATION = "SEMANTIC_CLASSIFICATION"
    AGENT_AGGREGATE_SUMMARY = "AGENT_AGGREGATE_SUMMARY"


class SourceMode(str, Enum):
    DISABLED = "disabled"
    LOCAL = "local"
    EXTERNAL_API = "external_api"
    ARTIFACT_ONLY = "artifact_only"


class AgentStatus(str, Enum):
    NOT_CONFIGURED = "NOT_CONFIGURED"
    SKIPPED = "SKIPPED"
    GENERATED = "GENERATED"
    ERROR = "ERROR"
    REJECTED_SAFETY = "REJECTED_SAFETY"
    READ_FROM_EXISTING_ARTIFACT = "READ_FROM_EXISTING_ARTIFACT"
    UNKNOWN_UNRESOLVED = "UNKNOWN_UNRESOLVED"


class DecisionEffect(str, Enum):
    EXPLANATION_ONLY = "explanation_only"
    AUDIT_ONLY = "audit_only"
    CONTEXT_ONLY = "context_only"
    SOFT_WARNING_ONLY = "soft_warning_only"
    SOFT_VETO_RECOMMENDATION_ONLY = "soft_veto_recommendation_only"
    NO_EFFECT = "no_effect"


class OperatingMode(str, Enum):
    AGENT_DEMO_DISABLED = "agent_demo_disabled"
    QWEN_LOCAL_DEMO = "qwen_local_demo"
    GEMINI_SELECTIVE_AUDIT_DEMO = "gemini_selective_audit_demo"
    HELIUS_READONLY_ENRICHMENT_DEMO = "helius_readonly_enrichment_demo"
    FULL_AGENT_OBSERVABILITY_DEMO = "full_agent_observability_demo"


MODE_ALIASES: dict[str, OperatingMode] = {
    "disabled": OperatingMode.AGENT_DEMO_DISABLED,
    "artifact-only": OperatingMode.AGENT_DEMO_DISABLED,
    "artifact_only": OperatingMode.AGENT_DEMO_DISABLED,
    "agent_demo_disabled": OperatingMode.AGENT_DEMO_DISABLED,
    "qwen-local": OperatingMode.QWEN_LOCAL_DEMO,
    "qwen_local": OperatingMode.QWEN_LOCAL_DEMO,
    "qwen_local_demo": OperatingMode.QWEN_LOCAL_DEMO,
    "gemini-selective": OperatingMode.GEMINI_SELECTIVE_AUDIT_DEMO,
    "gemini_selective": OperatingMode.GEMINI_SELECTIVE_AUDIT_DEMO,
    "gemini_selective_audit_demo": OperatingMode.GEMINI_SELECTIVE_AUDIT_DEMO,
    "helius-readonly": OperatingMode.HELIUS_READONLY_ENRICHMENT_DEMO,
    "helius_readonly": OperatingMode.HELIUS_READONLY_ENRICHMENT_DEMO,
    "helius_readonly_enrichment_demo": OperatingMode.HELIUS_READONLY_ENRICHMENT_DEMO,
    "full-demo": OperatingMode.FULL_AGENT_OBSERVABILITY_DEMO,
    "full_demo": OperatingMode.FULL_AGENT_OBSERVABILITY_DEMO,
    "full_agent_observability_demo": OperatingMode.FULL_AGENT_OBSERVABILITY_DEMO,
}


def resolve_operating_mode(raw: str | OperatingMode | None) -> OperatingMode:
    if isinstance(raw, OperatingMode):
        return raw
    key = str(raw or "artifact-only").strip().lower().replace(" ", "_")
    if key in MODE_ALIASES:
        return MODE_ALIASES[key]
    raise ValueError(f"Unknown AE12.7 operating mode: {raw!r}")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def new_agent_record_id(prefix: str = "ae127") -> str:
    return f"{prefix}_{uuid4().hex[:16]}"


def base_safety_flags() -> dict[str, Any]:
    return {
        "trade_authority_used": False,
        "live_authority_used": False,
        "wallet_accessed": False,
        "private_key_accessed": False,
        "real_transaction_attempted": False,
        "live_ready": False,
        "profitability_proven": False,
        "wallet_status": "NOT_CONFIGURED",
    }


def make_agent_record(
    *,
    agent_type: AgentType | str,
    source_mode: SourceMode | str,
    agent_status: AgentStatus | str,
    agent_summary: str,
    decision_effect: DecisionEffect | str = DecisionEffect.NO_EFFECT,
    candidate_id: str | None = None,
    source_decision_id: str | None = None,
    source_context_record_id: str | None = None,
    source_llm_audit_record_id: str | None = None,
    paper_order_id: str | None = None,
    position_id: str | None = None,
    pair_address: str | None = None,
    symbol: str | None = None,
    chain: str | None = None,
    input_artifact_refs: list[str] | None = None,
    output_artifact_refs: list[str] | None = None,
    warnings: list[str] | None = None,
    soft_veto_flags: list[str] | None = None,
    missing_context_flags: list[str] | None = None,
    semantic_label: str | None = None,
    confidence: float | None = None,
    safety_flags: dict[str, Any] | None = None,
    raw_response_ref: str | None = None,
    redacted_response_ref: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a canonical AE12.7 agent record. Authority fields are always false."""
    safety = base_safety_flags()
    if safety_flags:
        # Never allow callers to flip authority/wallet flags to True via normal paths.
        for k, v in safety_flags.items():
            if k in (
                "trade_authority_used",
                "live_authority_used",
                "wallet_accessed",
                "private_key_accessed",
                "real_transaction_attempted",
                "live_ready",
                "profitability_proven",
            ):
                safety[k] = False
            else:
                safety[k] = v

    record: dict[str, Any] = {
        "agent_record_id": new_agent_record_id(),
        "created_at": utc_now_iso(),
        "phase": AE12_7_PHASE,
        "schema_version": AE12_7_SCHEMA_VERSION,
        "agent_type": agent_type.value if isinstance(agent_type, AgentType) else str(agent_type),
        "source_mode": source_mode.value if isinstance(source_mode, SourceMode) else str(source_mode),
        "candidate_id": candidate_id,
        "source_decision_id": source_decision_id,
        "source_context_record_id": source_context_record_id,
        "source_llm_audit_record_id": source_llm_audit_record_id,
        "paper_order_id": paper_order_id,
        "position_id": position_id,
        "pair_address": pair_address,
        "symbol": symbol,
        "chain": chain,
        "input_artifact_refs": list(input_artifact_refs or []),
        "output_artifact_refs": list(output_artifact_refs or []),
        "agent_status": agent_status.value if isinstance(agent_status, AgentStatus) else str(agent_status),
        "agent_summary": agent_summary,
        "warnings": list(warnings or []),
        "soft_veto_flags": list(soft_veto_flags or []),
        "missing_context_flags": list(missing_context_flags or []),
        "semantic_label": semantic_label,
        "confidence": confidence,
        "trade_authority_used": False,
        "live_authority_used": False,
        "wallet_accessed": False,
        "private_key_accessed": False,
        "real_transaction_attempted": False,
        "decision_effect": (
            decision_effect.value if isinstance(decision_effect, DecisionEffect) else str(decision_effect)
        ),
        "safety_flags": safety,
        "raw_response_ref": raw_response_ref,
        "redacted_response_ref": redacted_response_ref,
    }
    if extra:
        for k, v in extra.items():
            if k not in record:
                record[k] = v
    return record


LinkIdField = Literal[
    "candidate_id",
    "candidate_policy_id",
    "target_row_id",
    "source_decision_id",
    "source_context_record_id",
    "source_llm_audit_record_id",
    "paper_order_id",
    "position_id",
    "pair_address",
    "event_timestamp",
    "first_seen_timestamp",
]
