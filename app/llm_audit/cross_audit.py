"""AE9 cross-audit alignment between AE6/AE8 caveats and LLM verdicts."""

from __future__ import annotations

from typing import Any

from app.llm_audit.audit_payload import WEAK_LINEAGE_STRENGTHS
from app.llm_audit.types import CrossAuditAlignmentStatus, LLMAuditVerdict

BLOCKING_VERDICTS = frozenset(
  {
    LLMAuditVerdict.AUDIT_BLOCK_WEAK_LINEAGE.value,
    LLMAuditVerdict.AUDIT_BLOCK_STALE_CONTEXT.value,
    LLMAuditVerdict.AUDIT_BLOCK_INSUFFICIENT_CONTEXT.value,
    LLMAuditVerdict.AUDIT_WARN_REVIEW_REQUIRED.value,
  }
)

WEAK_LINEAGE_VERDICTS = frozenset(
  {
    LLMAuditVerdict.AUDIT_BLOCK_WEAK_LINEAGE.value,
    LLMAuditVerdict.AUDIT_WARN_REVIEW_REQUIRED.value,
  }
)

STALE_VERDICTS = frozenset(
  {
    LLMAuditVerdict.AUDIT_BLOCK_STALE_CONTEXT.value,
    LLMAuditVerdict.AUDIT_WARN_REVIEW_REQUIRED.value,
  }
)


def _llm_lineage_assessment(parsed: dict[str, Any] | None) -> str:
  if not parsed:
    return "NOT_ASSESSED"
  return str(parsed.get("lineage_assessment", "NOT_ASSESSED"))


def _llm_freshness_assessment(parsed: dict[str, Any] | None) -> str:
  if not parsed:
    return "NOT_ASSESSED"
  return str(parsed.get("freshness_assessment", "NOT_ASSESSED"))


def _llm_missing_context_assessment(parsed: dict[str, Any] | None) -> str:
  if not parsed:
    return "NOT_ASSESSED"
  families = parsed.get("missing_context_families") or []
  return f"missing_families={families}"


def _blockers_text(parsed: dict[str, Any] | None) -> str:
  if not parsed:
    return ""
  blockers = parsed.get("main_blockers") or []
  return " ".join(str(b) for b in blockers).lower()


def _llm_covers_weak_lineage(llm_verdict: str, parsed: dict[str, Any] | None) -> bool:
  if llm_verdict in WEAK_LINEAGE_VERDICTS:
    return True
  if not parsed:
    return False
  blockers = _blockers_text(parsed)
  if "weak" in blockers and "lineage" in blockers:
    return True
  assessment = str(parsed.get("lineage_assessment", "")).lower()
  return "weak" in assessment


def _llm_covers_stale_context(llm_verdict: str, parsed: dict[str, Any] | None) -> bool:
  if llm_verdict in STALE_VERDICTS:
    return True
  if not parsed:
    return False
  blockers = _blockers_text(parsed)
  if "stale" in blockers and "context" in blockers:
    return True
  stale_families = parsed.get("stale_context_families") or []
  if stale_families:
    return True
  assessment = str(parsed.get("freshness_assessment", "")).lower()
  return "stale" in assessment


def _llm_covers_missing_context(llm_verdict: str, parsed: dict[str, Any] | None) -> bool:
  if llm_verdict in BLOCKING_VERDICTS:
    return True
  if not parsed:
    return False
  blockers = _blockers_text(parsed)
  if "missing" in blockers and "context" in blockers:
    return True
  missing_families = parsed.get("missing_context_families") or []
  return bool(missing_families)


def compute_cross_audit_alignment(
  *,
  audit_record_id: str,
  source_decision_id: str | None,
  source_context_record_id: str | None,
  ae6_lineage: dict[str, Any] | None,
  ae8_lineage: dict[str, Any] | None,
  ae8_freshness_status_summary: dict[str, Any] | None,
  ae8_missing_context_families: list[str],
  llm_verdict: str,
  llm_response_parsed: dict[str, Any] | None,
) -> dict[str, Any]:
  """Compare AE6/AE8 caveats against LLM verdict and multi-caveat response coverage."""
  ae6_strength = (ae6_lineage or {}).get("lineage_strength")
  ae6_mode = (ae6_lineage or {}).get("lineage_mode")
  ae8_strength = (ae8_lineage or {}).get("lineage_strength")
  ae8_mode = (ae8_lineage or {}).get("lineage_mode")

  ae6_weak = ae6_strength in WEAK_LINEAGE_STRENGTHS
  ae8_weak = ae8_strength in WEAK_LINEAGE_STRENGTHS
  has_weak_lineage = ae6_weak or ae8_weak

  stale_families = (ae8_freshness_status_summary or {}).get("stale_families") or []
  if not stale_families and ae8_freshness_status_summary:
    stale_count = (ae8_freshness_status_summary or {}).get("stale_source_count", 0)
    has_stale = stale_count > 0
  else:
    has_stale = bool(stale_families)

  has_missing = bool(ae8_missing_context_families)

  reasons: list[str] = []

  if source_decision_id is None:
    status = CrossAuditAlignmentStatus.CROSS_AUDIT_SOURCE_DECISION_MISSING.value
    reasons.append("source_decision_id_missing")
    return _row(
      audit_record_id=audit_record_id,
      source_decision_id=source_decision_id,
      source_context_record_id=source_context_record_id,
      ae6_strength=ae6_strength,
      ae6_mode=ae6_mode,
      ae8_strength=ae8_strength,
      ae8_mode=ae8_mode,
      ae8_freshness=ae8_freshness_status_summary,
      ae8_missing=ae8_missing_context_families,
      llm_verdict=llm_verdict,
      parsed=llm_response_parsed,
      status=status,
      reasons=reasons,
    )

  missed_weak = has_weak_lineage and not _llm_covers_weak_lineage(llm_verdict, llm_response_parsed)
  missed_stale = has_stale and not _llm_covers_stale_context(llm_verdict, llm_response_parsed)
  missed_missing = has_missing and not _llm_covers_missing_context(llm_verdict, llm_response_parsed)

  if missed_weak:
    status = CrossAuditAlignmentStatus.CROSS_AUDIT_LLM_MISSED_WEAK_LINEAGE.value
    reasons.append("llm_did_not_cover_weak_lineage_in_verdict_or_blockers")
  elif missed_stale:
    status = CrossAuditAlignmentStatus.CROSS_AUDIT_LLM_MISSED_STALE_CONTEXT.value
    reasons.append("llm_did_not_cover_stale_context_in_verdict_or_blockers")
  elif missed_missing:
    status = CrossAuditAlignmentStatus.CROSS_AUDIT_LLM_MISSED_MISSING_CONTEXT.value
    reasons.append("llm_did_not_cover_missing_context_in_verdict_or_blockers")
  else:
    status = CrossAuditAlignmentStatus.CROSS_AUDIT_ALIGNED.value
    reasons.append("llm_verdict_and_multi_caveat_coverage_align_with_system_caveats")

  return _row(
    audit_record_id=audit_record_id,
    source_decision_id=source_decision_id,
    source_context_record_id=source_context_record_id,
    ae6_strength=ae6_strength,
    ae6_mode=ae6_mode,
    ae8_strength=ae8_strength,
    ae8_mode=ae8_mode,
    ae8_freshness=ae8_freshness_status_summary,
    ae8_missing=ae8_missing_context_families,
    llm_verdict=llm_verdict,
    parsed=llm_response_parsed,
    status=status,
    reasons=reasons,
  )


def _row(
  *,
  audit_record_id: str,
  source_decision_id: str | None,
  source_context_record_id: str | None,
  ae6_strength: str | None,
  ae6_mode: str | None,
  ae8_strength: str | None,
  ae8_mode: str | None,
  ae8_freshness: dict[str, Any] | None,
  ae8_missing: list[str],
  llm_verdict: str,
  parsed: dict[str, Any] | None,
  status: str,
  reasons: list[str],
) -> dict[str, Any]:
  freshness_summary = ""
  if ae8_freshness:
    stale = ae8_freshness.get("stale_families") or []
    freshness_summary = f"stale_families={stale};stale_count={ae8_freshness.get('stale_source_count', 0)}"

  return {
    "audit_record_id": audit_record_id,
    "source_decision_id": source_decision_id,
    "source_context_record_id": source_context_record_id,
    "ae6_lineage_strength": ae6_strength,
    "ae6_lineage_mode": ae6_mode,
    "ae8_lineage_strength": ae8_strength,
    "ae8_lineage_mode": ae8_mode,
    "ae8_freshness_status_summary": freshness_summary,
    "ae8_missing_context_families": ",".join(ae8_missing) if ae8_missing else "",
    "llm_verdict": llm_verdict,
    "llm_lineage_assessment": _llm_lineage_assessment(parsed),
    "llm_freshness_assessment": _llm_freshness_assessment(parsed),
    "llm_missing_context_assessment": _llm_missing_context_assessment(parsed),
    "cross_audit_alignment_status": status,
    "cross_audit_alignment_reasons": ";".join(reasons),
    "cross_audit_alignment_status_enum": status,
    "cross_audit_alignment_reasons_list": reasons,
  }
