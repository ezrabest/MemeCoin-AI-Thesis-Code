"""AE9 mock LLM adapter — deterministic audit-only responses."""

from __future__ import annotations

import json
from typing import Any

from app.llm_audit.types import LLMAuditVerdict, LLMCallStatus


def choose_highest_priority_verdict(
  *,
  stale_context: bool,
  weak_lineage: bool,
  missing_context_families: bool,
  research_only_signals: bool,
) -> str:
  """Return one primary verdict; highest-priority material caveat wins."""
  del research_only_signals  # surfaced in warnings, not primary verdict escalation
  if stale_context:
    return LLMAuditVerdict.AUDIT_BLOCK_STALE_CONTEXT.value
  if weak_lineage:
    return LLMAuditVerdict.AUDIT_BLOCK_WEAK_LINEAGE.value
  if missing_context_families:
    return LLMAuditVerdict.AUDIT_WARN_REVIEW_REQUIRED.value
  return LLMAuditVerdict.AUDIT_PASS_NO_ACTION.value


def _detect_payload_flags(payload: dict[str, Any]) -> dict[str, Any]:
  blockers = payload.get("blockers") or []
  stale_families = (payload.get("freshness_audit") or {}).get("stale_families") or []
  missing_families = (payload.get("context_summary") or {}).get("missing_families") or []
  lineage_audit = payload.get("lineage_audit") or {}
  auth = payload.get("authorization_status") or {}
  whale = (payload.get("research_signal_caveats") or {}).get("whale_score_asof") or {}

  stale_context = "STALE_CONTEXT" in blockers or bool(stale_families)
  weak_lineage = "WEAK_LINEAGE" in blockers or bool(lineage_audit.get("weak_lineage_detected"))
  missing_context = "MISSING_CONTEXT_FAMILIES" in blockers or bool(missing_families)
  research_only = bool(
    whale.get("not_rule")
    or whale.get("not_runtime_approved_as_standalone_signal")
    or (payload.get("research_signal_caveats") or {}).get("research_context")
  )
  runtime_blocked = auth.get("runtime_inference_status") != "APPROVED"
  trading_blocked = auth.get("trading_authorization_status") != "APPROVED"

  return {
    "stale_context": stale_context,
    "weak_lineage": weak_lineage,
    "missing_context": missing_context,
    "research_only": research_only,
    "runtime_blocked": runtime_blocked,
    "trading_blocked": trading_blocked,
    "stale_families": stale_families,
    "missing_families": missing_families,
    "lineage_audit": lineage_audit,
  }


def run_mock_audit(payload: dict[str, Any]) -> dict[str, Any]:
  """Return deterministic JSON audit with multi-caveat coverage and one primary verdict."""
  flags = _detect_payload_flags(payload)
  caveats: list[str] = []
  warnings: list[str] = list((payload.get("context_summary") or {}).get("source_warnings") or [])

  if flags["stale_context"]:
    caveats.append("stale context present")
  if flags["weak_lineage"]:
    caveats.append("weak best-effort lineage present")
  if flags["missing_context"]:
    caveats.append("missing context families present")
  if flags["research_only"]:
    warnings.append("research-only signal caveats present")
  if flags["runtime_blocked"]:
    caveats.append("runtime inference not approved")
  if flags["trading_blocked"]:
    caveats.append("trading not approved")

  verdict = choose_highest_priority_verdict(
    stale_context=flags["stale_context"],
    weak_lineage=flags["weak_lineage"],
    missing_context_families=flags["missing_context"],
    research_only_signals=flags["research_only"],
  )

  lineage_audit = flags["lineage_audit"]
  stale_families = flags["stale_families"]
  missing_families = flags["missing_families"]

  if verdict == LLMAuditVerdict.AUDIT_BLOCK_STALE_CONTEXT.value:
    summary = (
      "Stale context families detected; primary audit block pending freshness review. "
      f"Additional caveats captured: {len(caveats)}."
    )
  elif verdict == LLMAuditVerdict.AUDIT_BLOCK_WEAK_LINEAGE.value:
    summary = (
      "Weak implicit lineage detected; primary audit block pending lineage review. "
      f"Additional caveats captured: {len(caveats)}."
    )
  elif verdict == LLMAuditVerdict.AUDIT_WARN_REVIEW_REQUIRED.value:
    summary = "Missing context families require human review before any forward action."
  else:
    summary = "No blocking audit issues detected; audit-only pass with no action."

  response = {
    "verdict": verdict,
    "summary": summary,
    "main_blockers": caveats,
    "warnings": warnings,
    "missing_context_families": missing_families,
    "stale_context_families": stale_families,
    "lineage_assessment": (
      f"ae6={lineage_audit.get('ae6_lineage_strength')}; "
      f"ae8={lineage_audit.get('ae8_lineage_strength')}; "
      f"weak={lineage_audit.get('weak_lineage_detected')}"
    ),
    "freshness_assessment": f"stale_families={stale_families}",
    "research_signal_assessment": json.dumps(
      payload.get("research_signal_caveats", {}).get("whale_score_asof", {}),
      default=str,
    ),
    "trading_authorization_respected": True,
    "runtime_inference_authorization_respected": True,
    "requires_human_review": verdict != LLMAuditVerdict.AUDIT_PASS_NO_ACTION.value,
  }

  return {
    "llm_provider": "mock",
    "llm_model": "mock-deterministic-v1",
    "llm_call_status": LLMCallStatus.MOCK.value,
    "llm_response_raw": json.dumps(response, separators=(",", ":")),
    "llm_response_parsed": response,
    "llm_verdict": verdict,
    "llm_confidence": None,
    "external_call_made": False,
  }
