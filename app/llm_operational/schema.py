"""AE19 LLM Operational Layer schemas and status constants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

PHASE = "AE19"
PROMPT_TEMPLATE_VERSION = "ae19_llm_operational_v1"
ENGINE_VERSION = "ae19_llm_operational_layer_v1"

# Task types
TASK_CANDIDATE_MEMO = "CANDIDATE_MEMO"
TASK_RISK_EXPLANATION = "RISK_EXPLANATION"
TASK_MISSED_WINNER_REVIEW = "MISSED_WINNER_REVIEW"
TASK_SEMANTIC_CONFLICT_REVIEW = "SEMANTIC_CONFLICT_REVIEW"
TASK_CONTEXT_SUMMARY = "CONTEXT_SUMMARY"
TASK_AUDIT = "AUDIT"

TASK_TYPES: tuple[str, ...] = (
    TASK_CANDIDATE_MEMO,
    TASK_RISK_EXPLANATION,
    TASK_MISSED_WINNER_REVIEW,
    TASK_SEMANTIC_CONFLICT_REVIEW,
    TASK_CONTEXT_SUMMARY,
    TASK_AUDIT,
)

# Provider statuses
PROVIDER_AVAILABLE = "LLM_PROVIDER_AVAILABLE"
PROVIDER_UNAVAILABLE = "LLM_PROVIDER_UNAVAILABLE"
PROVIDER_DISABLED = "LLM_PROVIDER_DISABLED"
PROVIDER_ERROR = "LLM_PROVIDER_ERROR"
GEMINI_UNAVAILABLE_OR_DISABLED = "GEMINI_PROVIDER_UNAVAILABLE_OR_DISABLED"
MOCK_PROVIDER_DIAGNOSTIC = "MOCK_PROVIDER_USED_DIAGNOSTIC_ONLY"

# Task statuses
TASK_SUCCEEDED = "LLM_TASK_SUCCEEDED"
TASK_FAILED = "LLM_TASK_FAILED"
TASK_SKIPPED_PROVIDER = "LLM_TASK_SKIPPED_PROVIDER_UNAVAILABLE"
TASK_SKIPPED_INPUT = "LLM_TASK_SKIPPED_INPUT_UNAVAILABLE"
TASK_SKIPPED_SAFETY = "LLM_TASK_SKIPPED_SAFETY_BOUNDARY"
TASK_MOCK_DIAGNOSTIC = "LLM_TASK_MOCK_DIAGNOSTIC_ONLY"

# Missed-winner unavailable
MISSED_WINNER_UNAVAILABLE = "MISSED_WINNER_REVIEW_UNAVAILABLE_NO_OUTCOME_EVIDENCE"

# Safety statuses
PASS_NO_TRADE_AUTHORITY = "PASS_NO_TRADE_AUTHORITY"
PASS_NO_LIVE_APPROVAL = "PASS_NO_LIVE_APPROVAL"
PASS_NO_RISK_OVERRIDE = "PASS_NO_RISK_OVERRIDE"
PASS_NO_WALLET_ACCESS = "PASS_NO_WALLET_ACCESS"
FAIL_TRADE_AUTHORITY_LANGUAGE = "FAIL_TRADE_AUTHORITY_LANGUAGE"
FAIL_LIVE_APPROVAL_LANGUAGE = "FAIL_LIVE_APPROVAL_LANGUAGE"
FAIL_RISK_OVERRIDE_LANGUAGE = "FAIL_RISK_OVERRIDE_LANGUAGE"
FAIL_WALLET_ACCESS = "FAIL_WALLET_ACCESS"
FAIL_FORBIDDEN_AUTHORITY_LANGUAGE = "FAIL_FORBIDDEN_AUTHORITY_LANGUAGE"
OUTPUT_REJECTED_AND_QUARANTINED = "OUTPUT_REJECTED_AND_QUARANTINED"

# Identity statuses
IDENTITY_APPROVED_SPINE = "IDENTITY_APPROVED_SPINE"
IDENTITY_UNRESOLVED = "IDENTITY_UNRESOLVED"
IDENTITY_AMBIGUOUS = "IDENTITY_AMBIGUOUS"
IDENTITY_SYMBOL_ONLY_REJECTED = "IDENTITY_SYMBOL_ONLY_JOIN_REJECTED"
IDENTITY_INVENTED_REJECTED = "IDENTITY_LLM_INVENTED_REJECTED"

# Classifications
CLASSIFICATION_PASS = "AE19_LLM_OPERATIONAL_LAYER_PASS_WITH_AUDIT"
CLASSIFICATION_PASS_LIMITATIONS = "AE19_PASS_WITH_PROVIDER_LIMITATIONS"
CLASSIFICATION_PARTIAL = "AE19_PARTIAL_PASS_LLM_INFRASTRUCTURE_ONLY"
CLASSIFICATION_BLOCKED_PROVIDER = "AE19_BLOCKED_LLM_PROVIDER_UNAVAILABLE"
CLASSIFICATION_BLOCKED_MISSING_TASK = "AE19_BLOCKED_MISSING_REQUIRED_TASK"
CLASSIFICATION_BLOCKED_AUTHORITY = "AE19_BLOCKED_AUTHORITY_ESCALATION"
CLASSIFICATION_BLOCKED_IDENTITY = "AE19_BLOCKED_IDENTITY_OR_LINEAGE_FAILURE"
CLASSIFICATION_BLOCKED_FALSE_SUCCESS = "AE19_BLOCKED_FALSE_PROVIDER_SUCCESS_REPORTING"
CLASSIFICATION_BLOCKED_QUARANTINE = "AE19_BLOCKED_DOWNSTREAM_QUARANTINE_FAILURE"

SAFETY_BOUNDARY = {
    "trade_authority_used": False,
    "live_trading_approved": False,
    "risk_override_used": False,
    "wallet_accessed": False,
    "llms_have_no_trade_authority": True,
    "llms_do_not_approve_buy_live_buy": True,
    "llms_do_not_override_gatekeeper": True,
    "llms_do_not_override_riskguard": True,
    "llms_do_not_connect_wallet": True,
    "llms_do_not_create_live_orders": True,
    "llms_do_not_change_model_scores": True,
    "llms_do_not_claim_profitability": True,
}

TASK_RECORD_FIELDS: tuple[str, ...] = (
    "ae19_task_id",
    "task_type",
    "provider",
    "provider_model",
    "provider_status",
    "task_status",
    "candidate_id",
    "clean_forward_candidate_id",
    "decision_input_id",
    "price_source_key",
    "provider_pair_url_exact",
    "canonical_market_identity",
    "normalized_provider_pair_url_key",
    "pair_address",
    "chain",
    "base_token_address",
    "quote_token_address",
    "symbol_pair",
    "evidence_refs",
    "model_evidence_refs",
    "consensus_refs",
    "meta_refs",
    "context_refs",
    "prompt_template_version",
    "prompt_text_hash",
    "response_text_hash",
    "created_at",
    "completed_at",
    "failure_reason",
    "mock_used",
    "counted_as_real_provider_success",
    "downstream_eligible",
    "downstream_quarantined",
    "safety_status",
    "trade_authority_used",
    "live_trading_approved",
    "risk_override_used",
    "wallet_accessed",
    "identity_status",
    "resolver_status",
    "output_text",
    "output_summary",
    "accepted_for_downstream",
    "safety_failed",
    "forbidden_language_hits",
    "missed_winner_status",
    "allowed_language_tags",
)


@dataclass
class AE19TaskRecord:
    """One AE19 operational LLM task record."""

    ae19_task_id: str
    task_type: str
    provider: str
    provider_model: str = ""
    provider_status: str = PROVIDER_UNAVAILABLE
    task_status: str = TASK_SKIPPED_PROVIDER
    candidate_id: str = ""
    clean_forward_candidate_id: str = ""
    decision_input_id: str = ""
    price_source_key: str = ""
    provider_pair_url_exact: str = ""
    canonical_market_identity: str = ""
    normalized_provider_pair_url_key: str = ""
    pair_address: str = ""
    chain: str = ""
    base_token_address: str = ""
    quote_token_address: str = ""
    symbol_pair: str = ""
    evidence_refs: list[str] = field(default_factory=list)
    model_evidence_refs: list[str] = field(default_factory=list)
    consensus_refs: list[str] = field(default_factory=list)
    meta_refs: list[str] = field(default_factory=list)
    context_refs: list[str] = field(default_factory=list)
    prompt_template_version: str = PROMPT_TEMPLATE_VERSION
    prompt_text_hash: str = ""
    response_text_hash: str = ""
    created_at: str = ""
    completed_at: str = ""
    failure_reason: str = ""
    mock_used: bool = False
    counted_as_real_provider_success: bool = False
    downstream_eligible: bool = False
    downstream_quarantined: bool = True
    safety_status: str = PASS_NO_TRADE_AUTHORITY
    trade_authority_used: bool = False
    live_trading_approved: bool = False
    risk_override_used: bool = False
    wallet_accessed: bool = False
    identity_status: str = IDENTITY_UNRESOLVED
    resolver_status: str = IDENTITY_UNRESOLVED
    output_text: str = ""
    output_summary: str = ""
    accepted_for_downstream: bool = False
    safety_failed: bool = False
    forbidden_language_hits: list[str] = field(default_factory=list)
    missed_winner_status: str = ""
    allowed_language_tags: list[str] = field(default_factory=list)
    raw_response_preserved: str = ""
    identity_invention_detected: bool = False
    symbol_only_join_attempted: bool = False
    symbol_only_join_rejected: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for list_key in (
            "evidence_refs",
            "model_evidence_refs",
            "consensus_refs",
            "meta_refs",
            "context_refs",
            "forbidden_language_hits",
            "allowed_language_tags",
        ):
            val = d.get(list_key)
            if isinstance(val, list):
                d[list_key] = "|".join(str(x) for x in val)
        return d

    def to_json_dict(self) -> dict[str, Any]:
        return asdict(self)


def empty_task_dict(**overrides: Any) -> dict[str, Any]:
    rec = AE19TaskRecord(ae19_task_id="", task_type="", provider="")
    d = rec.to_json_dict()
    d.update(overrides)
    return d
