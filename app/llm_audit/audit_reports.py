"""AE9 audit reports, decision gate, and CSV audit writers."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.llm_audit.types import (
  AE9_PHASE,
  Ae9FinalStatus,
  LLMAuditVerdict,
  RUNTIME_INFERENCE_STATUS,
  TRADING_AUTHORIZATION_STATUS,
)


def _write_json(path: Path, data: dict[str, Any]) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, default=str)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
  if not rows:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)


def _write_jsonl_sample(path: Path, records: list[dict[str, Any]], max_rows: int = 10) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w", encoding="utf-8") as f:
    for rec in records[:max_rows]:
      f.write(json.dumps(rec, default=str, separators=(",", ":")) + "\n")


def decide_ae9_status(
  *,
  records_created: int,
  unsafe_payload_count: int,
  disallowed_verdict_count: int,
  parse_error_count: int,
  provider: str,
  external_calls_made: int,
) -> Ae9FinalStatus:
  if records_created == 0:
    return Ae9FinalStatus.AE9_AUDIT_LAYER_BLOCKED_NO_INPUT_RECORDS
  if unsafe_payload_count > 0:
    return Ae9FinalStatus.AE9_AUDIT_LAYER_BLOCKED_UNSAFE_PROMPT_PAYLOAD
  if disallowed_verdict_count > 0:
    return Ae9FinalStatus.AE9_AUDIT_LAYER_BLOCKED_DISALLOWED_LLM_OUTPUT
  if parse_error_count > 0:
    return Ae9FinalStatus.AE9_AUDIT_LAYER_BLOCKED_PARSER_VALIDATION
  if provider == "mock":
    return Ae9FinalStatus.AE9_AUDIT_LAYER_PARTIAL_MOCK_ONLY
  if external_calls_made == 0 and provider != "mock":
    return Ae9FinalStatus.AE9_AUDIT_LAYER_PARTIAL_MOCK_ONLY
  return Ae9FinalStatus.AE9_AUDIT_LAYER_READY_FOR_UI_TRACEABILITY


def write_ae9_audits(
  *,
  output_root: Path,
  audit_records: list[dict[str, Any]],
  payload_samples: list[dict[str, Any]],
  source_inventory: list[dict[str, Any]],
  safety_audit_rows: list[dict[str, Any]],
  parse_audit_rows: list[dict[str, Any]],
  disallowed_verdict_rows: list[dict[str, Any]],
  lineage_freshness_rows: list[dict[str, Any]],
  cross_audit_rows: list[dict[str, Any]],
  external_call_safety: dict[str, Any],
  audit_schema: dict[str, Any],
  provider: str,
  allow_flags: dict[str, bool],
) -> dict[str, str]:
  """Write all required AE9 audit outputs. Returns full concrete paths."""
  ts = datetime.now(timezone.utc).isoformat()
  paths: dict[str, str] = {}

  paths["output_root"] = str(output_root.resolve())
  paths["summary"] = str((output_root / "reports" / "ae9_llm_audit_summary.json").resolve())
  paths["decision_gate"] = str((output_root / "reports" / "ae9_decision_gate.json").resolve())
  paths["records_sample"] = str(
    (output_root / "data" / "ae9_llm_audit_records_sample.jsonl").resolve()
  )
  paths["payloads_sample"] = str(
    (output_root / "data" / "ae9_prompt_payloads_sample.jsonl").resolve()
  )
  paths["source_inventory"] = str(
    (output_root / "data" / "ae9_source_artifact_inventory.csv").resolve()
  )
  paths["safety_audit"] = str(
    (output_root / "audits" / "ae9_prompt_payload_safety_audit.csv").resolve()
  )
  paths["parse_audit"] = str((output_root / "audits" / "ae9_response_parse_audit.csv").resolve())
  paths["disallowed_audit"] = str(
    (output_root / "audits" / "ae9_disallowed_verdict_audit.csv").resolve()
  )
  paths["external_call_safety"] = str(
    (output_root / "audits" / "ae9_external_call_safety_audit.json").resolve()
  )
  paths["lineage_freshness_audit"] = str(
    (output_root / "audits" / "ae9_lineage_freshness_caveat_audit.csv").resolve()
  )
  paths["cross_audit"] = str(
    (output_root / "audits" / "ae9_cross_audit_alignment.csv").resolve()
  )

  provider_dist = dict(Counter(r.get("llm_provider", "unknown") for r in audit_records))
  call_status_dist = dict(Counter(r.get("llm_call_status", "unknown") for r in audit_records))
  verdict_dist = dict(Counter(r.get("llm_verdict", "unknown") for r in audit_records))
  cross_dist = dict(
    Counter(r.get("cross_audit_alignment_status", "unknown") for r in audit_records)
  )

  with_source = sum(1 for r in audit_records if r.get("source_decision_id"))
  missing_source = sum(
    1 for r in audit_records if r.get("source_decision_link_status") == "SOURCE_DECISION_ID_MISSING"
  )

  unsafe = sum(1 for r in safety_audit_rows if not r.get("payload_safe_for_external_llm"))
  secrets = sum(1 for r in safety_audit_rows if r.get("secrets_detected"))
  raw_included = sum(1 for r in safety_audit_rows if r.get("raw_payload_included"))
  safe_external = sum(1 for r in safety_audit_rows if r.get("payload_safe_for_external_llm"))

  parse_errors = sum(1 for r in parse_audit_rows if r.get("parse_error"))
  pydantic_errors = sum(
    1
    for r in parse_audit_rows
    if "pydantic_validation_error" in str(r.get("error_detail", ""))
  )
  disallowed_count = len(disallowed_verdict_rows)

  weak_lineage = sum(
    1
    for r in lineage_freshness_rows
    if r.get("weak_lineage")
    or r.get("ae6_lineage_strength") == "WEAK_IMPLICIT_TIME_PAIR_LINKS"
    or r.get("ae8_lineage_strength") == "WEAK_IMPLICIT_TIME_PAIR_LINKS"
  )
  stale_context = sum(1 for r in lineage_freshness_rows if r.get("stale_context"))
  missing_context = sum(1 for r in lineage_freshness_rows if r.get("missing_context_families"))
  research_only = sum(1 for r in lineage_freshness_rows if r.get("whale_research_only"))

  external_calls = external_call_safety.get("external_calls_made", 0)

  final_status = decide_ae9_status(
    records_created=len(audit_records),
    unsafe_payload_count=unsafe,
    disallowed_verdict_count=disallowed_count,
    parse_error_count=parse_errors,
    provider=provider,
    external_calls_made=external_calls,
  )

  decision_gate = {
    "phase": AE9_PHASE,
    "final_status": final_status.value,
    "blocking_reasons": [],
    "audit_records_created": len(audit_records),
    "records_with_source_decision_id": with_source,
    "records_missing_source_decision_id": missing_source,
    "audit_schema_id": audit_schema.get("audit_schema_id"),
    "llm_provider_distribution": provider_dist,
    "llm_call_status_distribution": call_status_dist,
    "verdict_distribution": verdict_dist,
    "prompt_payload_safety_status": {
      "payloads_checked": len(safety_audit_rows),
      "unsafe_payloads": unsafe,
      "raw_payloads_included": raw_included,
      "secrets_detected": secrets,
      "redactions_applied": sum(1 for r in safety_audit_rows if r.get("redaction_applied")),
      "safe_for_external_llm_count": safe_external,
    },
    "disallowed_verdict_count": disallowed_count,
    "parse_error_count": parse_errors,
    "pydantic_validation_error_count": pydantic_errors,
    "cross_audit_alignment_distribution": cross_dist,
    "lineage_caveat_summary": {"weak_lineage_count": weak_lineage},
    "freshness_caveat_summary": {"stale_context_count": stale_context},
    "missing_context_family_summary": {"missing_context_family_count": missing_context},
    "research_signal_caveat_summary": {"research_only_signal_count": research_only},
    "external_call_status": external_call_safety,
    "recommended_next_phase": "AE10_UI_PAPER_DEMO_TRACEABILITY",
    "runtime_inference_status": RUNTIME_INFERENCE_STATUS,
    "trading_authorization_status": TRADING_AUTHORIZATION_STATUS,
    "allow_flags": allow_flags,
    "created_at_utc": ts,
    "output_paths": paths,
  }

  summary = {
    "phase": AE9_PHASE,
    "final_status": final_status.value,
    "audit_records_created": len(audit_records),
    "audit_schema_id": audit_schema.get("audit_schema_id"),
    "provider": provider,
    "output_root": str(output_root.resolve()),
    "paths": paths,
    "created_at_utc": ts,
  }

  _write_json(Path(paths["decision_gate"]), decision_gate)
  _write_json(Path(paths["summary"]), summary)
  _write_jsonl_sample(Path(paths["records_sample"]), audit_records)
  _write_jsonl_sample(Path(paths["payloads_sample"]), payload_samples)
  _write_csv(Path(paths["source_inventory"]), source_inventory)
  _write_csv(Path(paths["safety_audit"]), safety_audit_rows)
  _write_csv(Path(paths["parse_audit"]), parse_audit_rows)
  _write_csv(Path(paths["disallowed_audit"]), disallowed_verdict_rows)
  _write_json(Path(paths["external_call_safety"]), external_call_safety)
  _write_csv(Path(paths["lineage_freshness_audit"]), lineage_freshness_rows)

  cross_csv_rows = [
    {k: v for k, v in row.items() if not k.endswith("_list") and k != "cross_audit_alignment_status_enum"}
    for row in cross_audit_rows
  ]
  _write_csv(Path(paths["cross_audit"]), cross_csv_rows)

  return paths
