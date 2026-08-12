#!/usr/bin/env python3
"""AE9 LLM audit smoke script — mock by default, no external calls."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
  sys.path.insert(0, str(ROOT))

from app.llm_audit.audit_runner import run_ae9_llm_audit  # noqa: E402


def main() -> None:
  parser = argparse.ArgumentParser(description="AE9 LLM audit smoke")
  parser.add_argument("--max-records", type=int, default=50)
  parser.add_argument("--audit-only", action="store_true", default=True)
  parser.add_argument("--no-db-write", action="store_true", default=True)
  parser.add_argument(
    "--provider",
    choices=["mock", "qwen", "ollama", "gemini"],
    default="mock",
  )
  parser.add_argument("--allow-local-qwen", action="store_true", default=False)
  parser.add_argument("--allow-ollama", action="store_true", default=False)
  parser.add_argument("--allow-gemini", action="store_true", default=False)
  parser.add_argument("--output-root", type=Path, default=None)
  parser.add_argument("--ae6-jsonl", type=Path, default=None)
  parser.add_argument("--ae8-context-jsonl", type=Path, default=None)
  parser.add_argument("--ae7-decision-gate", type=Path, default=None)
  parser.add_argument("--ae8-decision-gate", type=Path, default=None)
  args = parser.parse_args()

  summary = run_ae9_llm_audit(
    project_root=ROOT,
    max_records=args.max_records,
    audit_only=args.audit_only,
    no_db_write=args.no_db_write,
    provider=args.provider,
    allow_local_qwen=args.allow_local_qwen,
    allow_ollama=args.allow_ollama,
    allow_gemini=args.allow_gemini,
    output_root=args.output_root,
    ae6_jsonl=args.ae6_jsonl,
    ae8_context_jsonl=args.ae8_context_jsonl,
    ae7_decision_gate=args.ae7_decision_gate,
    ae8_decision_gate=args.ae8_decision_gate,
  )

  compact = {
    "phase": summary.get("phase"),
    "final_status": summary.get("final_status"),
    "audit_records_created": summary.get("audit_records_created"),
    "records_with_source_decision_id": summary.get("records_with_source_decision_id"),
    "records_missing_source_decision_id": summary.get("records_missing_source_decision_id"),
    "audit_schema_id": summary.get("audit_schema_id"),
    "provider": summary.get("provider"),
    "llm_provider_distribution": summary.get("llm_provider_distribution"),
    "llm_call_status_distribution": summary.get("llm_call_status_distribution"),
    "verdict_distribution": summary.get("verdict_distribution"),
    "source_paths": summary.get("source_paths"),
    "output_root": summary.get("output_root"),
    "jsonl_path": summary.get("jsonl_path"),
    "output_paths": summary.get("output_paths"),
    "external_call_safety": summary.get("external_call_safety"),
    "runtime_inference_status": summary.get("runtime_inference_status"),
    "trading_authorization_status": summary.get("trading_authorization_status"),
  }
  print(json.dumps(compact, indent=2))


if __name__ == "__main__":
  main()
