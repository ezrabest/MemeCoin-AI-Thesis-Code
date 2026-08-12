"""AE9 provider-neutral prompt templates."""

from __future__ import annotations

import hashlib
import json
from typing import Any

SYSTEM_PROMPT = """You are an audit-only reviewer for a memecoin trading research system.
You CANNOT approve trades.
You CANNOT output BUY, SELL, EXECUTE, LIVE_BUY, PAPER_BUY, TRADE, or APPROVE_TRADE.
You have ZERO decision authority.
You must return strict JSON only — no markdown, no free text outside JSON.
You must respect lineage, freshness, and missingness caveats from the input.
You must treat whale_score_asof as RESEARCH_ONLY — never as a standalone buy rule.
You must not recommend trade execution.
Your verdict must be one of the allowed audit verdict values only."""

RESPONSE_JSON_SCHEMA = {
  "verdict": "<one of allowed audit verdicts>",
  "summary": "<brief audit summary>",
  "main_blockers": [],
  "warnings": [],
  "missing_context_families": [],
  "stale_context_families": [],
  "lineage_assessment": "<lineage assessment>",
  "freshness_assessment": "<freshness assessment>",
  "research_signal_assessment": "<research signal assessment>",
  "trading_authorization_respected": True,
  "runtime_inference_authorization_respected": True,
  "requires_human_review": True,
}


def compute_prompt_template_hash() -> str:
  content = {
    "system": SYSTEM_PROMPT,
    "response_schema": RESPONSE_JSON_SCHEMA,
  }
  serialized = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
  return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_user_prompt(payload: dict[str, Any]) -> str:
  """Build provider-neutral user prompt from audit payload."""
  sections = [
    "AUDIT REQUEST — audit-only, no trade authority",
    "",
    f"source_decision_id: {payload.get('source_decision_id', 'null')}",
    "",
    "CANDIDATE IDENTITY:",
    json.dumps(payload.get("candidate_identity", {}), indent=2, default=str),
    "",
    "MODEL / META SUMMARY:",
    json.dumps(payload.get("model_meta_summary", {}), indent=2, default=str),
    "",
    "CONTEXT SUMMARY:",
    json.dumps(payload.get("context_summary", {}), indent=2, default=str),
    "",
    "SOURCE STATUSES:",
    json.dumps(payload.get("source_statuses", {}), indent=2, default=str),
    "",
    "FRESHNESS AUDIT:",
    json.dumps(payload.get("freshness_audit", {}), indent=2, default=str),
    "",
    "LINEAGE AUDIT:",
    json.dumps(payload.get("lineage_audit", {}), indent=2, default=str),
    "",
    "RESEARCH SIGNAL CAVEATS:",
    json.dumps(payload.get("research_signal_caveats", {}), indent=2, default=str),
    "",
    "AUTHORIZATION STATUS:",
    json.dumps(payload.get("authorization_status", {}), indent=2, default=str),
    "",
    "BLOCKERS:",
    json.dumps(payload.get("blockers", []), indent=2, default=str),
    "",
    "REQUIRED RESPONSE JSON SCHEMA:",
    json.dumps(RESPONSE_JSON_SCHEMA, indent=2),
    "",
    "REMINDER: Return strict JSON only. No BUY/SELL/EXECUTE. Audit-only.",
  ]
  return "\n".join(sections)


def build_prompt_messages(payload: dict[str, Any]) -> list[dict[str, str]]:
  return [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": build_user_prompt(payload)},
  ]
