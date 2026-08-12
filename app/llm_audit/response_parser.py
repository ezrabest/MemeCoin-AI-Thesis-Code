"""AE9 strict Pydantic v2 LLM response parser."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.llm_audit.types import (
  ALLOWED_VERDICTS,
  LLMAuditVerdict,
  PARSER_VERSION,
)

ALLOWED_VERDICT_LITERAL = Literal[
  "AUDIT_PASS_NO_ACTION",
  "AUDIT_WARN_REVIEW_REQUIRED",
  "AUDIT_BLOCK_INSUFFICIENT_CONTEXT",
  "AUDIT_BLOCK_STALE_CONTEXT",
  "AUDIT_BLOCK_WEAK_LINEAGE",
  "AUDIT_BLOCK_POLICY_PLACEHOLDER",
  "AUDIT_BLOCK_PARITY_NOT_PROVEN",
  "AUDIT_BLOCK_CONCENTRATION_RISK",
  "AUDIT_BLOCK_RESEARCH_ONLY_SIGNAL",
  "AUDIT_ERROR_UNPARSEABLE_RESPONSE",
  "AUDIT_ERROR_DISALLOWED_VERDICT",
  "AUDIT_NOT_RUN",
]


class LLMAuditResponse(BaseModel):
  """Strict Pydantic v2 model for required LLM audit JSON response."""

  model_config = ConfigDict(strict=True, extra="forbid")

  verdict: ALLOWED_VERDICT_LITERAL
  summary: str
  main_blockers: list[str] = Field(default_factory=list)
  warnings: list[str] = Field(default_factory=list)
  missing_context_families: list[str] = Field(default_factory=list)
  stale_context_families: list[str] = Field(default_factory=list)
  lineage_assessment: str
  freshness_assessment: str
  research_signal_assessment: str
  trading_authorization_respected: bool
  runtime_inference_authorization_respected: bool
  requires_human_review: bool


class ParseResult:
  """Result of parsing an LLM response."""

  def __init__(
    self,
    *,
    verdict: str,
    parsed: dict[str, Any] | None = None,
    parse_errors: list[str] | None = None,
    raw: str | None = None,
  ) -> None:
    self.verdict = verdict
    self.parsed = parsed
    self.parse_errors = parse_errors or []
    self.raw = raw


def parse_llm_audit_response(raw: str) -> ParseResult:
  """Parse LLM response with strict Pydantic v2 validation.

  Returns AUDIT_ERROR_UNPARSEABLE_RESPONSE on any structural failure.
  """
  if not raw or not raw.strip():
    return ParseResult(
      verdict=LLMAuditVerdict.AUDIT_ERROR_UNPARSEABLE_RESPONSE.value,
      parse_errors=["empty_response"],
      raw=raw,
    )

  text = raw.strip()

  # Attempt strict JSON parse only — no markdown extraction in strict mode
  try:
    data = json.loads(text)
  except json.JSONDecodeError as exc:
    return ParseResult(
      verdict=LLMAuditVerdict.AUDIT_ERROR_UNPARSEABLE_RESPONSE.value,
      parse_errors=[f"json_decode_error: {exc}"],
      raw=raw,
    )

  if not isinstance(data, dict):
    return ParseResult(
      verdict=LLMAuditVerdict.AUDIT_ERROR_UNPARSEABLE_RESPONSE.value,
      parse_errors=["response_not_object"],
      raw=raw,
    )

  try:
    model = LLMAuditResponse.model_validate(data)
  except ValidationError as exc:
    return ParseResult(
      verdict=LLMAuditVerdict.AUDIT_ERROR_UNPARSEABLE_RESPONSE.value,
      parse_errors=[f"pydantic_validation_error: {exc}"],
      raw=raw,
    )

  verdict = model.verdict
  if verdict not in ALLOWED_VERDICTS:
    return ParseResult(
      verdict=LLMAuditVerdict.AUDIT_ERROR_UNPARSEABLE_RESPONSE.value,
      parse_errors=[f"disallowed_verdict_value: {verdict}"],
      raw=raw,
    )

  return ParseResult(
    verdict=verdict,
    parsed=model.model_dump(mode="json"),
    raw=raw,
  )


def get_parser_version() -> str:
  return PARSER_VERSION
