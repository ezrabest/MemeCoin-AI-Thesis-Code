"""Tests for AE9 LLM audit layer."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from app.llm_audit.audit_payload import build_audit_payload
from app.llm_audit.audit_persistence import LLMAuditJsonlWriter, read_audit_jsonl_safe
from app.llm_audit.audit_runner import run_ae9_llm_audit
from app.llm_audit.cross_audit import compute_cross_audit_alignment
from app.llm_audit.gemini_adapter import run_gemini_audit
from app.llm_audit.mock_adapter import choose_highest_priority_verdict, run_mock_audit
from app.llm_audit.prompt_templates import compute_prompt_template_hash
from app.llm_audit.qwen_adapter import run_ollama_audit, run_qwen_audit
from app.llm_audit.response_parser import LLMAuditResponse, parse_llm_audit_response
from app.llm_audit.safety import check_disallowed_content, check_payload_safety
from app.llm_audit.types import (
  LLMAuditRecord,
  LLMAuditVerdict,
  RUNTIME_INFERENCE_STATUS,
  TRADING_AUTHORIZATION_STATUS,
  build_audit_schema,
)


def _sample_decision(decision_id: str = "dec-001") -> dict:
  return {
    "decision_id": decision_id,
    "created_at_utc": "2026-07-10T09:00:52+00:00",
    "candidate_identity": {
      "candidate_id": "cand-001",
      "pair_address": "0xABC",
      "symbol": "PEPE/WETH",
      "chain": "ethereum",
      "event_timestamp": "2026-07-10T09:00:00+00:00",
    },
    "lineage": {
      "lineage_mode": "BEST_EFFORT_IMPLICIT_LINKAGE",
      "lineage_strength": "WEAK_IMPLICIT_TIME_PAIR_LINKS",
      "lineage_warning": "weak",
    },
    "model_scores": {
      "RF": {"available": False, "missing_reason": "NOT_AVAILABLE"},
    },
    "consensus": {"consensus_family": "NO_MODEL_CONSENSUS_AVAILABLE"},
    "research_context": {
      "whale_score_asof_status": "RESEARCH_ONLY_PLAUSIBLE_FEATURE_CANDIDATE",
      "whale_score_asof_not_rule": True,
    },
    "caveats": ["weak lineage"],
    "missingness": [],
  }


def _sample_context() -> dict:
  return {
    "context_record_id": "ctx-001",
    "candidate_id": "cand-001",
    "pair_address": "0xABC",
    "symbol": "PEPE/WETH",
    "chain": "ethereum",
    "as_of_timestamp": "2026-07-10T15:55:00+00:00",
    "context_schema_id": "schema-001",
    "source_statuses": {"rss": "SOURCE_EMPTY", "onchain": "SOURCE_NOT_AVAILABLE"},
    "source_warnings": ["RSS_CONTEXT_NOT_AVAILABLE"],
    "context_missingness": {
      "missing_families": ["rss", "onchain"],
      "family_missingness_flags": {"rss": True, "onchain": True, "whale": False},
    },
    "context_freshness": {
      "rss": {
        "freshness_status": "MISSING_TIMESTAMP",
        "missingness_reason": "MISSING_SOURCE_TIMESTAMP",
      },
      "whale": {"freshness_status": "FRESH"},
    },
    "lineage": {
      "lineage_mode": "BEST_EFFORT_IMPLICIT_LINKAGE",
      "lineage_strength": "WEAK_IMPLICIT_TIME_PAIR_LINKS",
      "lineage_warning": "weak context lineage",
    },
    "whale_context": {
      "whale_score_status": "RESEARCH_ONLY_PLAUSIBLE_FEATURE_CANDIDATE",
      "not_rule": True,
      "not_runtime_approved_as_standalone_signal": True,
    },
  }


def _valid_response_dict() -> dict:
  return {
    "verdict": "AUDIT_PASS_NO_ACTION",
    "summary": "audit ok",
    "main_blockers": [],
    "warnings": [],
    "missing_context_families": [],
    "stale_context_families": [],
    "lineage_assessment": "explicit",
    "freshness_assessment": "fresh",
    "research_signal_assessment": "research-only",
    "trading_authorization_respected": True,
    "runtime_inference_authorization_respected": True,
    "requires_human_review": False,
  }


def test_audit_schema_id_deterministic_and_content_derived():
  h = compute_prompt_template_hash()
  s1 = build_audit_schema(prompt_template_hash=h)
  s2 = build_audit_schema(prompt_template_hash=h)
  assert s1["audit_schema_id"] == s2["audit_schema_id"]
  assert len(s1["audit_schema_id"]) == 64
  assert s1["audit_schema_id"] != s1["created_at_utc"]


def test_prompt_template_hash_deterministic():
  assert compute_prompt_template_hash() == compute_prompt_template_hash()


def test_mock_adapter_returns_deterministic_json():
  payload = build_audit_payload(
    decision=_sample_decision(),
    context=_sample_context(),
    ae7_gate=None,
    ae8_gate=None,
  )
  r1 = run_mock_audit(payload)
  r2 = run_mock_audit(payload)
  assert r1["llm_verdict"] == r2["llm_verdict"]
  assert r1["llm_provider"] == "mock"
  parsed = json.loads(r1["llm_response_raw"])
  assert parsed["trading_authorization_respected"] is True


def test_mock_adapter_stale_context_priority():
  ctx = _sample_context()
  ctx["context_freshness"]["whale"] = {
    "freshness_status": "STALE",
    "missingness_reason": "STALE_SOURCE",
  }
  payload = build_audit_payload(decision=_sample_decision(), context=ctx, ae7_gate=None, ae8_gate=None)
  result = run_mock_audit(payload)
  assert result["llm_verdict"] == LLMAuditVerdict.AUDIT_BLOCK_STALE_CONTEXT.value


def test_mock_adapter_includes_all_material_caveats_in_blockers():
  ctx = _sample_context()
  ctx["context_freshness"]["whale"] = {
    "freshness_status": "STALE",
    "missingness_reason": "STALE_SOURCE",
  }
  payload = build_audit_payload(decision=_sample_decision(), context=ctx, ae7_gate=None, ae8_gate=None)
  parsed = json.loads(run_mock_audit(payload)["llm_response_raw"])
  blockers = parsed["main_blockers"]
  assert "stale context present" in blockers
  assert "weak best-effort lineage present" in blockers
  assert "missing context families present" in blockers
  assert "runtime inference not approved" in blockers
  assert "trading not approved" in blockers
  assert "research-only signal caveats present" in parsed["warnings"]


def test_choose_highest_priority_verdict_stale_over_weak():
  verdict = choose_highest_priority_verdict(
    stale_context=True,
    weak_lineage=True,
    missing_context_families=True,
    research_only_signals=True,
  )
  assert verdict == LLMAuditVerdict.AUDIT_BLOCK_STALE_CONTEXT.value


def test_cross_audit_aligned_when_stale_verdict_covers_all_caveats():
  parsed = {
    "main_blockers": [
      "stale context present",
      "weak best-effort lineage present",
      "missing context families present",
    ],
    "stale_context_families": ["whale"],
    "missing_context_families": ["rss", "onchain"],
    "lineage_assessment": "ae6=WEAK_IMPLICIT_TIME_PAIR_LINKS; weak=True",
    "freshness_assessment": "stale_families=['whale']",
  }
  cross = compute_cross_audit_alignment(
    audit_record_id="a1",
    source_decision_id="d1",
    source_context_record_id="c1",
    ae6_lineage={"lineage_strength": "WEAK_IMPLICIT_TIME_PAIR_LINKS"},
    ae8_lineage={"lineage_strength": "WEAK_IMPLICIT_TIME_PAIR_LINKS"},
    ae8_freshness_status_summary={"stale_families": ["whale"]},
    ae8_missing_context_families=["rss", "onchain"],
    llm_verdict=LLMAuditVerdict.AUDIT_BLOCK_STALE_CONTEXT.value,
    llm_response_parsed=parsed,
  )
  assert cross["cross_audit_alignment_status"] == "CROSS_AUDIT_ALIGNED"


def test_pydantic_parser_accepts_valid_strict_audit_json():
  raw = json.dumps(_valid_response_dict())
  result = parse_llm_audit_response(raw)
  assert result.verdict == LLMAuditVerdict.AUDIT_PASS_NO_ACTION.value
  assert result.parsed is not None


def test_pydantic_parser_rejects_free_text():
  result = parse_llm_audit_response("This is not JSON at all.")
  assert result.verdict == LLMAuditVerdict.AUDIT_ERROR_UNPARSEABLE_RESPONSE.value


def test_pydantic_parser_rejects_markdown_wrapped_json():
  wrapped = "```json\n" + json.dumps(_valid_response_dict()) + "\n```"
  result = parse_llm_audit_response(wrapped)
  assert result.verdict == LLMAuditVerdict.AUDIT_ERROR_UNPARSEABLE_RESPONSE.value


def test_pydantic_parser_rejects_string_booleans():
  data = _valid_response_dict()
  data["trading_authorization_respected"] = "true"
  result = parse_llm_audit_response(json.dumps(data))
  assert result.verdict == LLMAuditVerdict.AUDIT_ERROR_UNPARSEABLE_RESPONSE.value


def test_pydantic_parser_rejects_list_fields_as_strings():
  data = _valid_response_dict()
  data["warnings"] = "not a list"
  result = parse_llm_audit_response(json.dumps(data))
  assert result.verdict == LLMAuditVerdict.AUDIT_ERROR_UNPARSEABLE_RESPONSE.value


def test_pydantic_parser_rejects_extra_fields():
  data = _valid_response_dict()
  data["extra_field"] = "bad"
  result = parse_llm_audit_response(json.dumps(data))
  assert result.verdict == LLMAuditVerdict.AUDIT_ERROR_UNPARSEABLE_RESPONSE.value


def test_pydantic_parser_rejects_missing_required_fields():
  data = {"verdict": "AUDIT_PASS_NO_ACTION", "summary": "x"}
  result = parse_llm_audit_response(json.dumps(data))
  assert result.verdict == LLMAuditVerdict.AUDIT_ERROR_UNPARSEABLE_RESPONSE.value


def test_pydantic_parser_rejects_invalid_json():
  result = parse_llm_audit_response("{invalid")
  assert result.verdict == LLMAuditVerdict.AUDIT_ERROR_UNPARSEABLE_RESPONSE.value


def test_disallowed_buy_verdict_becomes_error():
  verdict, reasons = check_disallowed_content(
    verdict="BUY",
    parsed=None,
    raw='{"verdict":"BUY"}',
  )
  assert verdict == LLMAuditVerdict.AUDIT_ERROR_DISALLOWED_VERDICT.value
  assert reasons


def test_disallowed_execute_in_summary():
  parsed = _valid_response_dict()
  parsed["summary"] = "Recommend EXECUTE immediately"
  verdict, reasons = check_disallowed_content(
    verdict="AUDIT_PASS_NO_ACTION",
    parsed=parsed,
    raw=json.dumps(parsed),
  )
  assert verdict == LLMAuditVerdict.AUDIT_ERROR_DISALLOWED_VERDICT.value


def test_llm_decision_authority_always_false():
  record = LLMAuditRecord(
    audit_schema_id="a" * 64,
    candidate_id="c1",
    as_of_timestamp="2026-07-10T00:00:00+00:00",
  )
  assert record.llm_decision_authority is False


def test_no_trade_authority_always_true():
  record = LLMAuditRecord(
    audit_schema_id="a" * 64,
    candidate_id="c1",
    as_of_timestamp="2026-07-10T00:00:00+00:00",
  )
  assert record.no_trade_authority is True


def test_trading_authorization_status_always_not_approved():
  record = LLMAuditRecord(
    audit_schema_id="a" * 64,
    candidate_id="c1",
    as_of_timestamp="2026-07-10T00:00:00+00:00",
  )
  assert record.trading_authorization_status == TRADING_AUTHORIZATION_STATUS


def test_runtime_inference_status_blocked():
  record = LLMAuditRecord(
    audit_schema_id="a" * 64,
    candidate_id="c1",
    as_of_timestamp="2026-07-10T00:00:00+00:00",
  )
  assert record.runtime_inference_status == RUNTIME_INFERENCE_STATUS


def test_source_decision_id_carried_into_payload_and_record():
  payload = build_audit_payload(
    decision=_sample_decision("dec-xyz"),
    context=_sample_context(),
    ae7_gate=None,
    ae8_gate=None,
  )
  assert payload["source_decision_id"] == "dec-xyz"
  record = LLMAuditRecord(
    audit_schema_id="a" * 64,
    source_decision_id="dec-xyz",
    candidate_id="c1",
    as_of_timestamp="2026-07-10T00:00:00+00:00",
  )
  assert record.source_decision_id == "dec-xyz"


def test_missing_source_decision_id_explicit():
  payload = build_audit_payload(
    decision=None,
    context=_sample_context(),
    ae7_gate=None,
    ae8_gate=None,
  )
  assert payload["source_decision_id"] is None
  record = LLMAuditRecord(
    audit_schema_id="a" * 64,
    source_decision_link_status="SOURCE_DECISION_ID_MISSING",
    candidate_id="c1",
    as_of_timestamp="2026-07-10T00:00:00+00:00",
  )
  assert record.source_decision_link_status == "SOURCE_DECISION_ID_MISSING"


def test_cross_audit_aligned_weak_lineage():
  cross = compute_cross_audit_alignment(
    audit_record_id="a1",
    source_decision_id="d1",
    source_context_record_id="c1",
    ae6_lineage={"lineage_strength": "WEAK_IMPLICIT_TIME_PAIR_LINKS"},
    ae8_lineage={"lineage_strength": "WEAK_IMPLICIT_TIME_PAIR_LINKS"},
    ae8_freshness_status_summary={},
    ae8_missing_context_families=[],
    llm_verdict=LLMAuditVerdict.AUDIT_BLOCK_WEAK_LINEAGE.value,
    llm_response_parsed={"lineage_assessment": "weak"},
  )
  assert cross["cross_audit_alignment_status"] == "CROSS_AUDIT_ALIGNED"


def test_cross_audit_llm_missed_weak_lineage():
  cross = compute_cross_audit_alignment(
    audit_record_id="a1",
    source_decision_id="d1",
    source_context_record_id="c1",
    ae6_lineage={"lineage_strength": "WEAK_IMPLICIT_TIME_PAIR_LINKS"},
    ae8_lineage={},
    ae8_freshness_status_summary={},
    ae8_missing_context_families=[],
    llm_verdict=LLMAuditVerdict.AUDIT_PASS_NO_ACTION.value,
    llm_response_parsed={"lineage_assessment": "strong"},
  )
  assert cross["cross_audit_alignment_status"] == "CROSS_AUDIT_LLM_MISSED_WEAK_LINEAGE"


def test_cross_audit_llm_missed_stale_context():
  cross = compute_cross_audit_alignment(
    audit_record_id="a1",
    source_decision_id="d1",
    source_context_record_id="c1",
    ae6_lineage={},
    ae8_lineage={},
    ae8_freshness_status_summary={"stale_families": ["whale"], "stale_source_count": 1},
    ae8_missing_context_families=[],
    llm_verdict=LLMAuditVerdict.AUDIT_PASS_NO_ACTION.value,
    llm_response_parsed={},
  )
  assert cross["cross_audit_alignment_status"] == "CROSS_AUDIT_LLM_MISSED_STALE_CONTEXT"


def test_weak_lineage_in_payload():
  payload = build_audit_payload(
    decision=_sample_decision(),
    context=_sample_context(),
    ae7_gate=None,
    ae8_gate=None,
  )
  assert payload["lineage_audit"]["weak_lineage_detected"] is True


def test_stale_context_in_payload():
  ctx = _sample_context()
  ctx["context_freshness"]["whale"] = {
    "freshness_status": "STALE",
    "missingness_reason": "STALE_SOURCE",
  }
  payload = build_audit_payload(decision=_sample_decision(), context=ctx, ae7_gate=None, ae8_gate=None)
  assert "whale" in payload["freshness_audit"]["stale_families"]


def test_missing_context_families_in_payload():
  payload = build_audit_payload(
    decision=_sample_decision(),
    context=_sample_context(),
    ae7_gate=None,
    ae8_gate=None,
  )
  assert "rss" in payload["context_summary"]["missing_families"]


def test_whale_score_asof_research_only():
  payload = build_audit_payload(
    decision=_sample_decision(),
    context=_sample_context(),
    ae7_gate=None,
    ae8_gate=None,
  )
  whale = payload["research_signal_caveats"]["whale_score_asof"]
  assert whale["not_rule"] is True
  assert whale["not_runtime_approved_as_standalone_signal"] is True


def test_raw_payloads_not_included_by_default():
  payload = build_audit_payload(
    decision=_sample_decision(),
    context=_sample_context(),
    ae7_gate=None,
    ae8_gate=None,
  )
  safety = check_payload_safety(
    audit_record_id="a1",
    source_decision_id="d1",
    payload=payload,
    payload_text=json.dumps(payload),
    raw_payload_included=False,
  )
  assert safety["raw_payload_included"] is False


def test_prompt_payload_safety_detects_secrets():
  safety = check_payload_safety(
    audit_record_id="a1",
    source_decision_id="d1",
    payload={},
    payload_text='api_key="DUMMY"',
    raw_payload_included=False,
  )
  assert safety["secrets_detected"] is True
  assert safety["payload_safe_for_external_llm"] is False


def test_gemini_adapter_disabled_by_default():
  result = run_gemini_audit({}, allow_gemini=False, audit_only=True)
  assert result["llm_call_status"] == "DISABLED_BY_DEFAULT"
  assert result["external_call_made"] is False


def test_qwen_adapter_disabled_by_default():
  result = run_qwen_audit({}, allow_local_qwen=False, allow_ollama=False, audit_only=True)
  assert result["llm_call_status"] == "DISABLED_BY_DEFAULT"
  assert result["external_call_made"] is False


def test_ollama_adapter_requires_explicit_allow_flag():
  result = run_ollama_audit({}, allow_ollama=False, audit_only=True)
  assert result["llm_call_status"] == "DISABLED_BY_DEFAULT"


def test_gemini_adapter_requires_explicit_allow_flag():
  result = run_gemini_audit({}, allow_gemini=False, audit_only=True)
  assert result["llm_call_status"] == "DISABLED_BY_DEFAULT"


def test_jsonl_writer_flush_fsync():
  with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "test.jsonl"
    writer = LLMAuditJsonlWriter(path=path)
    record = LLMAuditRecord(
      audit_schema_id="a" * 64,
      candidate_id="c1",
      as_of_timestamp="2026-07-10T00:00:00+00:00",
    )
    with mock.patch("os.fsync") as fsync_mock:
      writer.append_record(record)
      fsync_mock.assert_called_once()
    writer.close()
    records, diag = read_audit_jsonl_safe(path)
    assert len(records) == 1
    assert diag["status"] == "ok"


def test_response_parser_requires_required_fields():
  with pytest.raises(Exception):
    LLMAuditResponse.model_validate({"verdict": "AUDIT_PASS_NO_ACTION"})


def test_default_smoke_mock_no_external_calls(tmp_path):
  ae6 = tmp_path / "ae6.jsonl"
  ae8 = tmp_path / "ae8.jsonl"
  ae6.write_text(
    json.dumps(_sample_decision("smoke-dec-1"), separators=(",", ":")) + "\n",
    encoding="utf-8",
  )
  ae8.write_text(
    json.dumps(_sample_context(), separators=(",", ":")) + "\n",
    encoding="utf-8",
  )
  out = tmp_path / "audit_out"
  summary = run_ae9_llm_audit(
    project_root=tmp_path,
    max_records=5,
    audit_only=True,
    no_db_write=True,
    provider="mock",
    output_root=out,
    ae6_jsonl=ae6,
    ae8_context_jsonl=ae8,
  )
  assert summary["audit_records_created"] >= 1
  assert summary["external_call_safety"]["external_calls_made"] == 0
  assert summary["provider"] == "mock"
  assert summary["output_root"]
  assert Path(summary["output_paths"]["decision_gate"]).is_file()
  assert Path(summary["output_paths"]["cross_audit"]).is_file()


def test_smoke_reports_full_concrete_paths(tmp_path):
  ae6 = tmp_path / "ae6.jsonl"
  ae6.write_text(
    json.dumps(_sample_decision(), separators=(",", ":")) + "\n",
    encoding="utf-8",
  )
  out = tmp_path / "audit_out"
  summary = run_ae9_llm_audit(
    project_root=tmp_path,
    max_records=1,
    output_root=out,
    ae6_jsonl=ae6,
  )
  for key, val in summary["output_paths"].items():
    assert val
    assert not val.startswith("reports/")
    assert str(out.resolve()) in val or key == "output_root"


def test_no_model_training_inference_trading_in_smoke_script():
  script = Path(__file__).resolve().parents[1] / "scripts" / "run_ae9_llm_audit_smoke.py"
  text = script.read_text(encoding="utf-8")
  assert "train" not in text.lower() or "retrain" not in text.lower()
  assert "paper_trade" not in text
  assert "live_trade" not in text
  assert "DROP TABLE" not in text

