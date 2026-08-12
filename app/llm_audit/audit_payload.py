"""AE9 audit payload construction from AE6/AE7/AE8 sources."""

from __future__ import annotations

from typing import Any

WEAK_LINEAGE_STRENGTHS = frozenset(
  {
    "WEAK_IMPLICIT_TIME_PAIR_LINKS",
    "WEAK",
    "BEST_EFFORT_IMPLICIT",
  }
)

STALE_FRESHNESS_STATUSES = frozenset(
  {
    "STALE",
    "INVALID_FUTURE_TIMESTAMP",
    "MISSING_TIMESTAMP",
  }
)


def _extract_stale_families(context_record: dict[str, Any] | None) -> list[str]:
  if not context_record:
    return []
  stale: list[str] = []
  freshness = context_record.get("context_freshness") or {}
  for fam, block in freshness.items():
    if isinstance(block, dict) and block.get("freshness_status") in STALE_FRESHNESS_STATUSES:
      stale.append(fam)
    if isinstance(block, dict) and block.get("missingness_reason") == "STALE_SOURCE":
      if fam not in stale:
        stale.append(fam)
  return sorted(set(stale))


def _extract_missing_families(context_record: dict[str, Any] | None) -> list[str]:
  if not context_record:
    return ["rss", "onchain", "whale", "reputation", "liquidity_activity"]
  miss = context_record.get("context_missingness") or {}
  families = miss.get("missing_families")
  if isinstance(families, list):
    return sorted(families)
  flags = miss.get("family_missingness_flags") or {}
  return sorted(f for f, v in flags.items() if v)


def _whale_research_caveat(decision: dict[str, Any] | None, context: dict[str, Any] | None) -> dict[str, Any]:
  whale_ctx = (context or {}).get("whale_context") or {}
  research = (decision or {}).get("research_context") or {}
  return {
    "whale_score_asof_status": whale_ctx.get("whale_score_status")
    or research.get("whale_score_asof_status")
    or "RESEARCH_ONLY_PLAUSIBLE_FEATURE_CANDIDATE",
    "not_rule": whale_ctx.get("not_rule", research.get("whale_score_asof_not_rule", True)),
    "not_runtime_approved_as_standalone_signal": whale_ctx.get(
      "not_runtime_approved_as_standalone_signal",
      research.get("whale_score_asof_not_runtime_approved", True),
    ),
    "description": "whale_score_asof is RESEARCH_ONLY — not a standalone approval rule",
  }


def build_audit_payload(
  *,
  decision: dict[str, Any] | None,
  context: dict[str, Any] | None,
  ae7_gate: dict[str, Any] | None,
  ae8_gate: dict[str, Any] | None,
  decision_line_no: int | None = None,
  context_line_no: int | None = None,
) -> dict[str, Any]:
  """Construct compact structured audit payload for LLM auditor."""
  identity = (decision or {}).get("candidate_identity") or {}
  if context and not identity.get("candidate_id"):
    identity = {
      "candidate_id": context.get("candidate_id"),
      "pair_address": context.get("pair_address"),
      "symbol": context.get("symbol"),
      "chain": context.get("chain"),
    }

  source_decision_id = (decision or {}).get("decision_id")
  ae6_lineage = (decision or {}).get("lineage") or {}
  ae8_lineage = (context or {}).get("lineage") or {}

  stale_families = _extract_stale_families(context)
  missing_families = _extract_missing_families(context)

  ae6_strength = ae6_lineage.get("lineage_strength")
  ae8_strength = ae8_lineage.get("lineage_strength")
  weak_lineage = (
    ae6_strength in WEAK_LINEAGE_STRENGTHS
    or ae8_strength in WEAK_LINEAGE_STRENGTHS
  )

  blockers: list[str] = []
  if weak_lineage:
    blockers.append("WEAK_LINEAGE")
  if stale_families:
    blockers.append("STALE_CONTEXT")
  if missing_families:
    blockers.append("MISSING_CONTEXT_FAMILIES")

  model_scores = (decision or {}).get("model_scores") or {}
  consensus = (decision or {}).get("consensus") or {}

  return {
    "audit_mode": "AUDIT_ONLY_NO_TRADE_AUTHORITY",
    "llm_must_not_recommend_trade_execution": True,
    "llm_must_not_output_buy_sell": True,
    "llm_is_audit_only": True,
    "source_decision_id": source_decision_id,
    "source_decision_record_line_no": decision_line_no,
    "source_context_record_id": (context or {}).get("context_record_id"),
    "source_context_record_line_no": context_line_no,
    "candidate_identity": {
      "candidate_id": identity.get("candidate_id") or context.get("candidate_id") if context else None,
      "pair_address": identity.get("pair_address") or (context or {}).get("pair_address"),
      "symbol": identity.get("symbol") or (context or {}).get("symbol"),
      "chain": identity.get("chain") or (context or {}).get("chain"),
      "as_of_timestamp": (context or {}).get("as_of_timestamp")
      or identity.get("event_timestamp")
      or (decision or {}).get("created_at_utc"),
    },
    "model_meta_summary": {
      "model_scores": {
        k: {"available": v.get("available"), "missing_reason": v.get("missing_reason")}
        for k, v in model_scores.items()
        if isinstance(v, dict)
      },
      "consensus": consensus,
      "ae7_final_status": (ae7_gate or {}).get("final_status"),
      "ae7_runtime_inference_status": (ae7_gate or {}).get("runtime_inference_status"),
      "ae7_trading_authorization_status": (ae7_gate or {}).get("trading_authorization_status"),
    },
    "context_summary": {
      "context_schema_id": (context or {}).get("context_schema_id"),
      "source_statuses": (context or {}).get("source_statuses") or {},
      "source_warnings": (context or {}).get("source_warnings") or [],
      "missing_families": missing_families,
      "stale_families": stale_families,
      "ae8_final_status": (ae8_gate or {}).get("final_status"),
    },
    "source_statuses": (context or {}).get("source_statuses") or {},
    "freshness_audit": {
      "stale_families": stale_families,
      "context_freshness": (context or {}).get("context_freshness") or {},
      "ae8_freshness_summary": (ae8_gate or {}).get("freshness_summary") or {},
    },
    "lineage_audit": {
      "ae6_lineage_mode": ae6_lineage.get("lineage_mode"),
      "ae6_lineage_strength": ae6_strength,
      "ae6_lineage_warning": ae6_lineage.get("lineage_warning"),
      "ae8_lineage_mode": ae8_lineage.get("lineage_mode"),
      "ae8_lineage_strength": ae8_strength,
      "ae8_lineage_warning": ae8_lineage.get("lineage_warning"),
      "weak_lineage_detected": weak_lineage,
    },
    "research_signal_caveats": {
      "whale_score_asof": _whale_research_caveat(decision, context),
      "decision_caveats": (decision or {}).get("caveats") or [],
      "research_context": (decision or {}).get("research_context") or {},
    },
    "authorization_status": {
      "runtime_inference_status": "BLOCKED_NOT_APPROVED",
      "trading_authorization_status": "NOT_APPROVED",
      "llm_decision_authority": False,
      "no_trade_authority": True,
    },
    "blockers": blockers,
    "policy_summary": {
      "ae7_policy_config_status": (ae7_gate or {}).get("policy_config_status"),
      "ae7_blocking_reasons": (ae7_gate or {}).get("blocking_reasons") or [],
    },
    "robustness_summary": {
      "ae7_robustness": (ae7_gate or {}).get("robustness_summary") or {},
      "decision_robustness_status": ((decision or {}).get("research_context") or {}).get(
        "robustness_status"
      ),
    },
    "missingness_summary": {
      "decision_missingness": (decision or {}).get("missingness") or [],
      "context_missing_families": missing_families,
      "context_source_warnings": (context or {}).get("source_warnings") or [],
    },
  }


def payload_to_summary_sections(payload: dict[str, Any]) -> dict[str, Any]:
  """Map payload into LLMAuditRecord summary field blocks."""
  return {
    "input_summary": {
      "source_decision_id": payload.get("source_decision_id"),
      "candidate_identity": payload.get("candidate_identity"),
      "audit_mode": payload.get("audit_mode"),
    },
    "model_score_summary": payload.get("model_meta_summary", {}),
    "meta_layer_summary": {
      "ae7_final_status": payload.get("model_meta_summary", {}).get("ae7_final_status"),
      "policy_summary": payload.get("policy_summary", {}),
    },
    "context_summary": payload.get("context_summary", {}),
    "freshness_summary": payload.get("freshness_audit", {}),
    "lineage_summary": payload.get("lineage_audit", {}),
    "missingness_summary": payload.get("missingness_summary", {}),
    "robustness_summary": payload.get("robustness_summary", {}),
    "policy_summary": payload.get("policy_summary", {}),
    "authorization_summary": payload.get("authorization_status", {}),
  }
