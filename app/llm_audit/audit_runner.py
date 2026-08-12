"""AE9 LLM audit orchestration runner."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.decision.persistence import read_jsonl_records_safe
from app.llm_audit.audit_payload import build_audit_payload, payload_to_summary_sections
from app.llm_audit.audit_persistence import LLMAuditJsonlWriter, read_audit_jsonl_safe
from app.llm_audit.audit_reports import write_ae9_audits
from app.llm_audit.cross_audit import compute_cross_audit_alignment
from app.llm_audit.gemini_adapter import run_gemini_audit
from app.llm_audit.mock_adapter import run_mock_audit
from app.llm_audit.prompt_templates import build_user_prompt, compute_prompt_template_hash
from app.llm_audit.qwen_adapter import run_ollama_audit, run_qwen_audit
from app.llm_audit.response_parser import parse_llm_audit_response
from app.llm_audit.safety import check_disallowed_content, check_payload_safety
from app.llm_audit.types import (
  AE9_PHASE,
  LLMAuditRecord,
  LLMAuditVerdict,
  LLMCallStatus,
  RUNTIME_INFERENCE_STATUS,
  TRADING_AUTHORIZATION_STATUS,
  build_audit_schema,
)


def _load_json(path: Path | None) -> tuple[dict[str, Any] | None, str]:
  if path is None or not path.is_file():
    return None, "MISSING"
  try:
    with open(path, "r", encoding="utf-8") as f:
      return json.load(f), "OK"
  except (json.JSONDecodeError, OSError):
    return None, "ERROR"


def discover_latest_glob(project_root: Path, pattern: str) -> Path | None:
  matches = sorted(project_root.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
  return matches[0] if matches else None


def discover_sources(
  project_root: Path,
  *,
  ae6_jsonl: Path | None = None,
  ae8_context_jsonl: Path | None = None,
  ae7_decision_gate: Path | None = None,
  ae8_decision_gate: Path | None = None,
) -> dict[str, Any]:
  """Discover or use explicit source artifact paths."""
  inventory: list[dict[str, Any]] = []

  ae6_path = ae6_jsonl or discover_latest_glob(
    project_root, "data/decision_records/ae6_decisions_*.jsonl"
  )
  ae8_ctx_path = ae8_context_jsonl or discover_latest_glob(
    project_root, "data/context_intelligence/ae8_context_features_*.jsonl"
  )
  ae7_gate_path = ae7_decision_gate or discover_latest_glob(
    project_root,
    "data/training/manual_verified_results/ae7_final_meta_layer_*/reports/ae7_final_meta_layer_decision_gate.json",
  )
  ae8_gate_path = ae8_decision_gate
  if ae8_gate_path is None:
    ae8_gate_path = discover_latest_glob(
      project_root, "data/audits/ae8_context_intelligence_*/reports/ae8_decision_gate.json"
    )
  if ae8_gate_path is None:
    candidate = project_root / "reports" / "ae8_decision_gate.json"
    ae8_gate_path = candidate if candidate.is_file() else None

  for label, path in [
    ("ae6_decisions_jsonl", ae6_path),
    ("ae8_context_features_jsonl", ae8_ctx_path),
    ("ae7_final_meta_layer_decision_gate", ae7_gate_path),
    ("ae8_decision_gate", ae8_gate_path),
  ]:
    status = "OK" if path and path.is_file() else "MISSING"
    inventory.append(
      {
        "artifact_label": label,
        "path": str(path.resolve()) if path else "",
        "status": status,
      }
    )

  ae6_records: list[dict[str, Any]] = []
  ae6_diag: dict[str, Any] = {}
  if ae6_path and ae6_path.is_file():
    ae6_records, ae6_diag = read_jsonl_records_safe(ae6_path)

  ae8_records: list[dict[str, Any]] = []
  ae8_diag: dict[str, Any] = {}
  if ae8_ctx_path and ae8_ctx_path.is_file():
    ae8_records, ae8_diag = read_context_jsonl_safe(ae8_ctx_path)

  ae7_gate, ae7_status = _load_json(ae7_gate_path)
  ae8_gate, ae8_status = _load_json(ae8_gate_path)

  return {
    "ae6_path": ae6_path,
    "ae8_context_path": ae8_ctx_path,
    "ae7_gate_path": ae7_gate_path,
    "ae8_gate_path": ae8_gate_path,
    "ae6_records": ae6_records,
    "ae6_diag": ae6_diag,
    "ae8_records": ae8_records,
    "ae8_diag": ae8_diag,
    "ae7_gate": ae7_gate,
    "ae7_gate_status": ae7_status,
    "ae8_gate": ae8_gate,
    "ae8_gate_status": ae8_status,
    "inventory": inventory,
  }


def read_context_jsonl_safe(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
  return read_jsonl_records_safe(path)


def _index_context_by_key(records: list[dict[str, Any]]) -> dict[str, tuple[dict[str, Any], int]]:
  index: dict[str, tuple[dict[str, Any], int]] = {}
  for idx, rec in enumerate(records, start=1):
    for key in (
      rec.get("candidate_id"),
      rec.get("pair_address"),
    ):
      if key and key not in index:
        index[str(key)] = (rec, idx)
  return index


def _match_context(
  decision: dict[str, Any],
  context_index: dict[str, tuple[dict[str, Any], int]],
) -> tuple[dict[str, Any] | None, int | None]:
  identity = decision.get("candidate_identity") or {}
  for key in (
    identity.get("candidate_id"),
    identity.get("pair_address"),
    decision.get("pair_address"),
  ):
    if key and str(key) in context_index:
      return context_index[str(key)]
  return None, None


def _run_provider(
  provider: str,
  payload: dict[str, Any],
  *,
  allow_local_qwen: bool,
  allow_ollama: bool,
  allow_gemini: bool,
  audit_only: bool,
) -> dict[str, Any]:
  if provider == "mock":
    return run_mock_audit(payload)
  if provider == "qwen":
    return run_qwen_audit(
      payload,
      allow_local_qwen=allow_local_qwen,
      allow_ollama=False,
      audit_only=audit_only,
    )
  if provider == "ollama":
    return run_ollama_audit(payload, allow_ollama=allow_ollama, audit_only=audit_only)
  if provider == "gemini":
    return run_gemini_audit(payload, allow_gemini=allow_gemini, audit_only=audit_only)
  return run_mock_audit(payload)


def run_ae9_llm_audit(
  *,
  project_root: Path,
  max_records: int = 50,
  audit_only: bool = True,
  no_db_write: bool = True,
  provider: str = "mock",
  allow_local_qwen: bool = False,
  allow_ollama: bool = False,
  allow_gemini: bool = False,
  output_root: Path | None = None,
  ae6_jsonl: Path | None = None,
  ae8_context_jsonl: Path | None = None,
  ae7_decision_gate: Path | None = None,
  ae8_decision_gate: Path | None = None,
) -> dict[str, Any]:
  """Main AE9 orchestration — audit-only, no trade authority."""
  del no_db_write  # explicit no-op; AE9 never writes DB

  from scripts.diagnostics._common import timestamp_slug

  ts_slug = timestamp_slug()
  audit_dir = output_root or (project_root / "data" / "audits" / f"ae9_llm_audit_{ts_slug}")
  audit_dir.mkdir(parents=True, exist_ok=True)

  prompt_hash = compute_prompt_template_hash()
  audit_schema = build_audit_schema(prompt_template_hash=prompt_hash)

  sources = discover_sources(
    project_root,
    ae6_jsonl=ae6_jsonl,
    ae8_context_jsonl=ae8_context_jsonl,
    ae7_decision_gate=ae7_decision_gate,
    ae8_decision_gate=ae8_decision_gate,
  )

  ae6_records = sources["ae6_records"][:max_records]
  ae8_records = sources["ae8_records"]
  context_index = _index_context_by_key(ae8_records)

  # If no AE6 records, try AE8-only
  candidates: list[tuple[dict[str, Any] | None, dict[str, Any] | None, int | None, int | None]] = []
  if ae6_records:
    for d_idx, decision in enumerate(ae6_records, start=1):
      context, c_idx = _match_context(decision, context_index)
      candidates.append((decision, context, d_idx, c_idx))
  elif ae8_records:
    for c_idx, context in enumerate(ae8_records[:max_records], start=1):
      candidates.append((None, context, None, c_idx))

  audit_records: list[dict[str, Any]] = []
  payload_samples: list[dict[str, Any]] = []
  safety_rows: list[dict[str, Any]] = []
  parse_rows: list[dict[str, Any]] = []
  disallowed_rows: list[dict[str, Any]] = []
  lineage_rows: list[dict[str, Any]] = []
  cross_rows: list[dict[str, Any]] = []
  external_calls_made = 0

  writer = LLMAuditJsonlWriter()

  for decision, context, d_line, c_line in candidates:
    payload = build_audit_payload(
      decision=decision,
      context=context,
      ae7_gate=sources["ae7_gate"],
      ae8_gate=sources["ae8_gate"],
      decision_line_no=d_line,
      context_line_no=c_line,
    )
    payload_text = build_user_prompt(payload)
    payload_samples.append(payload)

    audit_record_id = str(uuid4())
    safety_row = check_payload_safety(
      audit_record_id=audit_record_id,
      source_decision_id=payload.get("source_decision_id"),
      payload=payload,
      payload_text=payload_text,
      raw_payload_included=False,
    )
    safety_rows.append(safety_row)

    provider_result = _run_provider(
      provider,
      payload,
      allow_local_qwen=allow_local_qwen,
      allow_ollama=allow_ollama,
      allow_gemini=allow_gemini,
      audit_only=audit_only,
    )

    if provider_result.get("external_call_made"):
      external_calls_made += 1

    raw_response = provider_result.get("llm_response_raw")
    llm_verdict = provider_result.get("llm_verdict")
    llm_parsed = provider_result.get("llm_response_parsed")
    parse_errors: list[str] = []

    if raw_response and provider == "mock":
      parse_result = parse_llm_audit_response(raw_response)
      llm_verdict = parse_result.verdict
      llm_parsed = parse_result.parsed
      parse_errors = parse_result.parse_errors
    elif raw_response:
      parse_result = parse_llm_audit_response(raw_response)
      llm_verdict = parse_result.verdict
      llm_parsed = parse_result.parsed
      parse_errors = parse_result.parse_errors
    elif llm_verdict is None:
      llm_verdict = LLMAuditVerdict.AUDIT_NOT_RUN.value

    if llm_verdict and llm_parsed is None and raw_response is None:
      pass
    elif llm_verdict:
      final_verdict, disallowed_reasons = check_disallowed_content(
        verdict=llm_verdict,
        parsed=llm_parsed,
        raw=raw_response,
      )
      if disallowed_reasons:
        disallowed_rows.append(
          {
            "audit_record_id": audit_record_id,
            "source_decision_id": payload.get("source_decision_id"),
            "original_verdict": llm_verdict,
            "forced_verdict": final_verdict,
            "reasons": ";".join(disallowed_reasons),
          }
        )
        llm_verdict = final_verdict

    parse_rows.append(
      {
        "audit_record_id": audit_record_id,
        "source_decision_id": payload.get("source_decision_id"),
        "parse_error": bool(parse_errors),
        "error_detail": ";".join(parse_errors) if parse_errors else "",
        "final_verdict": llm_verdict,
      }
    )

    identity = payload.get("candidate_identity") or {}
    source_decision_id = payload.get("source_decision_id")
    source_link_status = None
    if source_decision_id is None:
      source_link_status = "SOURCE_DECISION_ID_MISSING"

    summaries = payload_to_summary_sections(payload)
    ae6_lineage = (decision or {}).get("lineage")
    ae8_lineage = (context or {}).get("lineage")

    cross = compute_cross_audit_alignment(
      audit_record_id=audit_record_id,
      source_decision_id=source_decision_id,
      source_context_record_id=payload.get("source_context_record_id"),
      ae6_lineage=ae6_lineage,
      ae8_lineage=ae8_lineage,
      ae8_freshness_status_summary=payload.get("freshness_audit"),
      ae8_missing_context_families=(payload.get("context_summary") or {}).get(
        "missing_families", []
      ),
      llm_verdict=llm_verdict or LLMAuditVerdict.AUDIT_NOT_RUN.value,
      llm_response_parsed=llm_parsed,
    )
    cross_rows.append(cross)

    whale = (payload.get("research_signal_caveats") or {}).get("whale_score_asof") or {}
    lineage_rows.append(
      {
        "audit_record_id": audit_record_id,
        "source_decision_id": source_decision_id,
        "ae6_lineage_strength": (ae6_lineage or {}).get("lineage_strength"),
        "ae8_lineage_strength": (ae8_lineage or {}).get("lineage_strength"),
        "weak_lineage": (payload.get("lineage_audit") or {}).get("weak_lineage_detected"),
        "stale_context": bool((payload.get("freshness_audit") or {}).get("stale_families")),
        "missing_context_families": ",".join(
          (payload.get("context_summary") or {}).get("missing_families") or []
        ),
        "whale_research_only": whale.get("not_rule", True),
      }
    )

    record = LLMAuditRecord(
      audit_record_id=audit_record_id,
      audit_schema_id=audit_schema["audit_schema_id"],
      source_decision_id=source_decision_id,
      source_decision_record_path=str(sources["ae6_path"].resolve())
      if sources["ae6_path"]
      else None,
      source_decision_record_line_no=d_line,
      source_decision_link_status=source_link_status,
      source_context_record_id=payload.get("source_context_record_id"),
      source_context_record_path=str(sources["ae8_context_path"].resolve())
      if sources["ae8_context_path"]
      else None,
      source_context_record_line_no=c_line,
      candidate_id=identity.get("candidate_id") or "UNKNOWN",
      pair_address=identity.get("pair_address"),
      symbol=identity.get("symbol"),
      chain=identity.get("chain"),
      as_of_timestamp=identity.get("as_of_timestamp")
      or datetime.now(timezone.utc).isoformat(),
      input_summary=summaries["input_summary"],
      model_score_summary=summaries["model_score_summary"],
      meta_layer_summary=summaries["meta_layer_summary"],
      context_summary=summaries["context_summary"],
      freshness_summary=summaries["freshness_summary"],
      lineage_summary=summaries["lineage_summary"],
      missingness_summary=summaries["missingness_summary"],
      robustness_summary=summaries["robustness_summary"],
      policy_summary=summaries["policy_summary"],
      authorization_summary=summaries["authorization_summary"],
      llm_provider=provider_result.get("llm_provider", provider),
      llm_model=provider_result.get("llm_model"),
      llm_call_status=provider_result.get("llm_call_status", LLMCallStatus.NOT_RUN.value),
      llm_response_raw=raw_response,
      llm_response_parsed=llm_parsed,
      llm_verdict=llm_verdict or LLMAuditVerdict.AUDIT_NOT_RUN.value,
      llm_confidence=provider_result.get("llm_confidence"),
      runtime_inference_status=RUNTIME_INFERENCE_STATUS,
      trading_authorization_status=TRADING_AUTHORIZATION_STATUS,
      parse_errors=parse_errors,
      audit_warnings=list((payload.get("context_summary") or {}).get("source_warnings") or []),
      audit_blockers=list(payload.get("blockers") or []),
      cross_audit_alignment_status=cross["cross_audit_alignment_status"],
      cross_audit_alignment_reasons=cross["cross_audit_alignment_reasons_list"],
    )

    writer.append_record(record)
    audit_records.append(record.model_dump(mode="json"))

  writer.close()

  allow_flags = {
    "allow_local_qwen": allow_local_qwen,
    "allow_ollama": allow_ollama,
    "allow_gemini": allow_gemini,
  }
  external_call_safety = {
    "provider_selected": provider,
    "allow_flags": allow_flags,
    "external_calls_made": external_calls_made,
    "local_llm_calls_made": external_calls_made if provider in ("qwen", "ollama") else 0,
    "audit_only": audit_only,
  }

  output_paths = write_ae9_audits(
    output_root=audit_dir,
    audit_records=audit_records,
    payload_samples=[{"payload": p} for p in payload_samples],
    source_inventory=sources["inventory"],
    safety_audit_rows=safety_rows,
    parse_audit_rows=parse_rows,
    disallowed_verdict_rows=disallowed_rows,
    lineage_freshness_rows=lineage_rows,
    cross_audit_rows=cross_rows,
    external_call_safety=external_call_safety,
    audit_schema=audit_schema,
    provider=provider,
    allow_flags=allow_flags,
  )

  jsonl_path = writer.path

  return {
    "phase": AE9_PHASE,
    "final_status": json.loads(
      open(output_paths["decision_gate"], encoding="utf-8").read()
    ).get("final_status"),
    "audit_records_created": len(audit_records),
    "records_with_source_decision_id": sum(1 for r in audit_records if r.get("source_decision_id")),
    "records_missing_source_decision_id": sum(
      1 for r in audit_records if r.get("source_decision_link_status") == "SOURCE_DECISION_ID_MISSING"
    ),
    "audit_schema_id": audit_schema["audit_schema_id"],
    "provider": provider,
    "llm_provider_distribution": dict(
      __import__("collections").Counter(r.get("llm_provider") for r in audit_records)
    ),
    "llm_call_status_distribution": dict(
      __import__("collections").Counter(r.get("llm_call_status") for r in audit_records)
    ),
    "verdict_distribution": dict(
      __import__("collections").Counter(r.get("llm_verdict") for r in audit_records)
    ),
    "source_paths": {
      "ae6_jsonl": str(sources["ae6_path"].resolve()) if sources["ae6_path"] else None,
      "ae8_context_jsonl": str(sources["ae8_context_path"].resolve())
      if sources["ae8_context_path"]
      else None,
      "ae7_decision_gate": str(sources["ae7_gate_path"].resolve())
      if sources["ae7_gate_path"]
      else None,
      "ae8_decision_gate": str(sources["ae8_gate_path"].resolve())
      if sources["ae8_gate_path"]
      else None,
    },
    "output_root": str(audit_dir.resolve()),
    "jsonl_path": str(jsonl_path.resolve()),
    "output_paths": output_paths,
    "external_call_safety": external_call_safety,
    "runtime_inference_status": RUNTIME_INFERENCE_STATUS,
    "trading_authorization_status": TRADING_AUTHORIZATION_STATUS,
  }
