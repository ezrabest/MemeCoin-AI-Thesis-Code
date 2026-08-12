"""AE9 LLM audit layer types and canonical LLMAuditRecord schema."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

AE9_PHASE = "AE9_QWEN_GEMINI_AUDIT_LAYER"
AE9_AUDIT_SCHEMA_VERSION = "AE9_LLM_AUDIT_V1"
PARSER_VERSION = "AE9_RESPONSE_PARSER_V1"

RUNTIME_INFERENCE_STATUS = "BLOCKED_NOT_APPROVED"
TRADING_AUTHORIZATION_STATUS = "NOT_APPROVED"


class LLMAuditVerdict(str, Enum):
  AUDIT_PASS_NO_ACTION = "AUDIT_PASS_NO_ACTION"
  AUDIT_WARN_REVIEW_REQUIRED = "AUDIT_WARN_REVIEW_REQUIRED"
  AUDIT_BLOCK_INSUFFICIENT_CONTEXT = "AUDIT_BLOCK_INSUFFICIENT_CONTEXT"
  AUDIT_BLOCK_STALE_CONTEXT = "AUDIT_BLOCK_STALE_CONTEXT"
  AUDIT_BLOCK_WEAK_LINEAGE = "AUDIT_BLOCK_WEAK_LINEAGE"
  AUDIT_BLOCK_POLICY_PLACEHOLDER = "AUDIT_BLOCK_POLICY_PLACEHOLDER"
  AUDIT_BLOCK_PARITY_NOT_PROVEN = "AUDIT_BLOCK_PARITY_NOT_PROVEN"
  AUDIT_BLOCK_CONCENTRATION_RISK = "AUDIT_BLOCK_CONCENTRATION_RISK"
  AUDIT_BLOCK_RESEARCH_ONLY_SIGNAL = "AUDIT_BLOCK_RESEARCH_ONLY_SIGNAL"
  AUDIT_ERROR_UNPARSEABLE_RESPONSE = "AUDIT_ERROR_UNPARSEABLE_RESPONSE"
  AUDIT_ERROR_DISALLOWED_VERDICT = "AUDIT_ERROR_DISALLOWED_VERDICT"
  AUDIT_NOT_RUN = "AUDIT_NOT_RUN"


ALLOWED_VERDICTS = frozenset(v.value for v in LLMAuditVerdict)

DISALLOWED_VERDICTS = frozenset(
  {
    "BUY",
    "SELL",
    "LIVE_BUY",
    "PAPER_BUY",
    "EXECUTE",
    "TRADE",
    "APPROVE_TRADE",
  }
)

FORBIDDEN_OUTPUT_PATTERNS = [
  r"\bBUY\b",
  r"\bSELL\b",
  r"\bEXECUTE\b",
  r"\bAPPROVE_TRADE\b",
  r"\bLIVE_BUY\b",
  r"\bPAPER_BUY\b",
]


class CrossAuditAlignmentStatus(str, Enum):
  CROSS_AUDIT_ALIGNED = "CROSS_AUDIT_ALIGNED"
  CROSS_AUDIT_LLM_MISSED_WEAK_LINEAGE = "CROSS_AUDIT_LLM_MISSED_WEAK_LINEAGE"
  CROSS_AUDIT_LLM_MISSED_STALE_CONTEXT = "CROSS_AUDIT_LLM_MISSED_STALE_CONTEXT"
  CROSS_AUDIT_LLM_MISSED_MISSING_CONTEXT = "CROSS_AUDIT_LLM_MISSED_MISSING_CONTEXT"
  CROSS_AUDIT_LLM_OVERSTATED_CONFIDENCE = "CROSS_AUDIT_LLM_OVERSTATED_CONFIDENCE"
  CROSS_AUDIT_SOURCE_DECISION_MISSING = "CROSS_AUDIT_SOURCE_DECISION_MISSING"
  CROSS_AUDIT_NOT_APPLICABLE = "CROSS_AUDIT_NOT_APPLICABLE"


class Ae9FinalStatus(str, Enum):
  AE9_AUDIT_LAYER_READY_FOR_UI_TRACEABILITY = "AE9_AUDIT_LAYER_READY_FOR_UI_TRACEABILITY"
  AE9_AUDIT_LAYER_PARTIAL_MOCK_ONLY = "AE9_AUDIT_LAYER_PARTIAL_MOCK_ONLY"
  AE9_AUDIT_LAYER_BLOCKED_NO_INPUT_RECORDS = "AE9_AUDIT_LAYER_BLOCKED_NO_INPUT_RECORDS"
  AE9_AUDIT_LAYER_BLOCKED_UNSAFE_PROMPT_PAYLOAD = "AE9_AUDIT_LAYER_BLOCKED_UNSAFE_PROMPT_PAYLOAD"
  AE9_AUDIT_LAYER_BLOCKED_DISALLOWED_LLM_OUTPUT = "AE9_AUDIT_LAYER_BLOCKED_DISALLOWED_LLM_OUTPUT"
  AE9_AUDIT_LAYER_BLOCKED_EXTERNAL_CONFIG = "AE9_AUDIT_LAYER_BLOCKED_EXTERNAL_CONFIG"
  AE9_AUDIT_LAYER_BLOCKED_PARSER_VALIDATION = "AE9_AUDIT_LAYER_BLOCKED_PARSER_VALIDATION"
  AE9_AUDIT_LAYER_BLOCKED_SOURCE_DECISION_LINKAGE = "AE9_AUDIT_LAYER_BLOCKED_SOURCE_DECISION_LINKAGE"
  AE9_AUDIT_LAYER_BLOCKED_WITH_EXACT_REASONS = "AE9_AUDIT_LAYER_BLOCKED_WITH_EXACT_REASONS"


class LLMCallStatus(str, Enum):
  SUCCESS = "SUCCESS"
  DISABLED_BY_DEFAULT = "DISABLED_BY_DEFAULT"
  LLM_SOURCE_CONFIG_MISSING = "LLM_SOURCE_CONFIG_MISSING"
  ERROR = "ERROR"
  MOCK = "MOCK"
  NOT_RUN = "NOT_RUN"


class LLMAuditRecord(BaseModel):
  """Canonical AE9 LLM audit record — audit-only, no trade authority."""

  model_config = ConfigDict(extra="forbid")

  audit_record_id: str = Field(default_factory=lambda: str(uuid4()))
  audit_schema_id: str
  audit_schema_version: str = AE9_AUDIT_SCHEMA_VERSION

  source_decision_id: str | None = None
  source_decision_record_path: str | None = None
  source_decision_record_line_no: int | None = None
  source_decision_link_status: str | None = None
  source_context_record_id: str | None = None
  source_context_record_path: str | None = None
  source_context_record_line_no: int | None = None

  candidate_id: str
  pair_address: str | None = None
  symbol: str | None = None
  chain: str | None = None
  as_of_timestamp: str
  audit_created_at_utc: str = Field(
    default_factory=lambda: datetime.now(timezone.utc).isoformat()
  )

  input_summary: dict[str, Any] = Field(default_factory=dict)
  model_score_summary: dict[str, Any] = Field(default_factory=dict)
  meta_layer_summary: dict[str, Any] = Field(default_factory=dict)
  context_summary: dict[str, Any] = Field(default_factory=dict)
  freshness_summary: dict[str, Any] = Field(default_factory=dict)
  lineage_summary: dict[str, Any] = Field(default_factory=dict)
  missingness_summary: dict[str, Any] = Field(default_factory=dict)
  robustness_summary: dict[str, Any] = Field(default_factory=dict)
  policy_summary: dict[str, Any] = Field(default_factory=dict)
  authorization_summary: dict[str, Any] = Field(default_factory=dict)

  llm_provider: str = "mock"
  llm_model: str | None = None
  llm_call_status: str = LLMCallStatus.NOT_RUN.value
  llm_response_raw: str | None = None
  llm_response_parsed: dict[str, Any] | None = None
  llm_verdict: str = LLMAuditVerdict.AUDIT_NOT_RUN.value
  llm_confidence: float | None = None
  llm_decision_authority: Literal[False] = False
  no_trade_authority: Literal[True] = True
  runtime_inference_status: str = RUNTIME_INFERENCE_STATUS
  trading_authorization_status: str = TRADING_AUTHORIZATION_STATUS

  safety_flags: list[str] = Field(default_factory=list)
  parse_errors: list[str] = Field(default_factory=list)
  audit_warnings: list[str] = Field(default_factory=list)
  audit_blockers: list[str] = Field(default_factory=list)

  cross_audit_alignment_status: str = CrossAuditAlignmentStatus.CROSS_AUDIT_NOT_APPLICABLE.value
  cross_audit_alignment_reasons: list[str] = Field(default_factory=list)

  phase: Literal["AE9_QWEN_GEMINI_AUDIT_LAYER"] = AE9_PHASE


def compute_schema_hash(content: dict[str, Any]) -> str:
  serialized = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
  return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_audit_schema(*, prompt_template_hash: str) -> dict[str, Any]:
  """Build deterministic AE9 audit schema with content-derived audit_schema_id."""
  fields = [
    "audit_record_id",
    "audit_schema_id",
    "audit_schema_version",
    "source_decision_id",
    "source_context_record_id",
    "candidate_id",
    "pair_address",
    "symbol",
    "chain",
    "as_of_timestamp",
    "input_summary",
    "model_score_summary",
    "meta_layer_summary",
    "context_summary",
    "freshness_summary",
    "lineage_summary",
    "missingness_summary",
    "robustness_summary",
    "policy_summary",
    "authorization_summary",
    "llm_provider",
    "llm_model",
    "llm_call_status",
    "llm_verdict",
    "llm_decision_authority",
    "no_trade_authority",
    "runtime_inference_status",
    "trading_authorization_status",
    "cross_audit_alignment_status",
  ]
  core = {
    "version": AE9_AUDIT_SCHEMA_VERSION,
    "fields": fields,
    "verdict_enum": sorted(ALLOWED_VERDICTS),
    "forbidden_output_patterns": FORBIDDEN_OUTPUT_PATTERNS,
    "prompt_template_hash": prompt_template_hash,
    "parser_version": PARSER_VERSION,
  }
  schema_hash = compute_schema_hash(core)
  audit_schema_id = hashlib.sha256(
    f"{AE9_AUDIT_SCHEMA_VERSION}|{schema_hash}".encode("utf-8")
  ).hexdigest()
  return {
    "audit_schema_id": audit_schema_id,
    "audit_schema_version": AE9_AUDIT_SCHEMA_VERSION,
    "fields": fields,
    "verdict_enum": sorted(ALLOWED_VERDICTS),
    "forbidden_output_patterns": FORBIDDEN_OUTPUT_PATTERNS,
    "prompt_template_hash": prompt_template_hash,
    "parser_version": PARSER_VERSION,
    "schema_hash": schema_hash,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
  }
