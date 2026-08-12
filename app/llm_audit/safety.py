"""AE9 prompt payload safety and disallowed verdict detection."""

from __future__ import annotations

import re
from typing import Any

from app.llm_audit.types import DISALLOWED_VERDICTS, FORBIDDEN_OUTPUT_PATTERNS, LLMAuditVerdict

SECRET_PATTERNS = [
  re.compile(r"(?i)(api[_-]?key|apikey)\s*[:=]\s*['\"]?\w{8,}"),
  re.compile(r"(?i)(secret|password|passwd|token)\s*[:=]\s*['\"]?\S{8,}"),
  re.compile(r"(?i)(private[_-]?key|wallet[_-]?key)\s*[:=]"),
  re.compile(r"(?i)sk-[a-zA-Z0-9]{20,}"),
  re.compile(r"(?i)AIza[0-9A-Za-z\-_]{35}"),
  re.compile(r"(?i)-----BEGIN (RSA |EC )?PRIVATE KEY-----"),
]

ACTION_PATTERN = re.compile(
  r"\b(" + "|".join(DISALLOWED_VERDICTS) + r")\b",
  re.IGNORECASE,
)

# Negated / audit-context phrases that mention action words without recommending them
_NEGATION_PREFIX = re.compile(r"\b(not|no|never|without|forbid|forbidden|disallow|blocked|audit-only)\b", re.IGNORECASE)


def _contains_action_recommendation(text: str) -> bool:
  """Return True only when action-like tokens appear outside negation/audit context."""
  for match in ACTION_PATTERN.finditer(text):
    start = match.start()
    prefix = text[max(0, start - 40):start]
    if _NEGATION_PREFIX.search(prefix):
      continue
    token = match.group(1).upper()
    # Allow audit-schema field names and metadata keys in JSON-ish text
    if token == "TRADE" and re.search(r"(no_trade|not_trade|_trade_|trade_authorization|trade_execution)", text, re.I):
      continue
    return True
  return False


def detect_secrets(text: str) -> list[str]:
  hits: list[str] = []
  for pattern in SECRET_PATTERNS:
    if pattern.search(text):
      hits.append(pattern.pattern)
  return hits


def check_payload_safety(
  *,
  audit_record_id: str,
  source_decision_id: str | None,
  payload: dict[str, Any],
  payload_text: str,
  raw_payload_included: bool = False,
) -> dict[str, Any]:
  """Run prompt payload safety audit for one record."""
  secrets = detect_secrets(payload_text)
  redaction_applied = False
  safe_for_external = True
  reasons: list[str] = []

  if secrets:
    safe_for_external = False
    reasons.append("secrets_detected")
  if raw_payload_included:
    safe_for_external = False
    reasons.append("raw_payload_included")
  if len(payload_text) > 50000:
    safe_for_external = False
    reasons.append("payload_too_large")

  if not reasons:
    reasons.append("payload_safe")

  return {
    "audit_record_id": audit_record_id,
    "source_decision_id": source_decision_id,
    "payload_size_chars": len(payload_text),
    "raw_payload_included": raw_payload_included,
    "secrets_detected": bool(secrets),
    "redaction_applied": redaction_applied,
    "payload_safe_for_external_llm": safe_for_external,
    "reason": ";".join(reasons),
  }


def check_disallowed_content(
  *,
  verdict: str,
  parsed: dict[str, Any] | None,
  raw: str | None,
) -> tuple[str, list[str]]:
  """Check for disallowed trading verdicts or action-like content.

  Returns (final_verdict, reasons).
  """
  reasons: list[str] = []

  if verdict.upper() in DISALLOWED_VERDICTS:
    reasons.append(f"disallowed_verdict: {verdict}")
    return LLMAuditVerdict.AUDIT_ERROR_DISALLOWED_VERDICT.value, reasons

  texts_to_check: list[str] = []
  if raw:
    texts_to_check.append(raw)
  if parsed:
    for key in ("summary", "lineage_assessment", "freshness_assessment", "research_signal_assessment"):
      val = parsed.get(key)
      if isinstance(val, str):
        texts_to_check.append(val)
    for key in ("main_blockers", "warnings"):
      for item in parsed.get(key, []):
        if isinstance(item, str):
          texts_to_check.append(item)

  for text in texts_to_check:
    if _contains_action_recommendation(text):
      reasons.append("action_like_content_in_response")
      return LLMAuditVerdict.AUDIT_ERROR_DISALLOWED_VERDICT.value, reasons

  for pattern in FORBIDDEN_OUTPUT_PATTERNS:
    for text in texts_to_check:
      if re.search(pattern, text, re.IGNORECASE):
        # Skip if match is in negated audit-context phrasing
        for m in re.finditer(pattern, text, re.IGNORECASE):
          prefix = text[max(0, m.start() - 40):m.start()]
          if _NEGATION_PREFIX.search(prefix):
            continue
          reasons.append(f"forbidden_pattern_match: {pattern}")
          return LLMAuditVerdict.AUDIT_ERROR_DISALLOWED_VERDICT.value, reasons

  return verdict, reasons
