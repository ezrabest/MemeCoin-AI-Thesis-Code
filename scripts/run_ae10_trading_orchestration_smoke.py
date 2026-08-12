#!/usr/bin/env python3
"""AE10 trading orchestration smoke script — paper/demo and live dry-run wiring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.execution.execution_orchestrator import run_ae10_trading_orchestration  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="AE10 trading orchestration smoke")
    parser.add_argument("--max-records", type=int, default=50)
    parser.add_argument("--audit-only", action="store_true", default=True)
    parser.add_argument("--no-db-write", action="store_true", default=True)
    parser.add_argument("--enable-paper-demo-orders", action="store_true", default=False)
    parser.add_argument(
        "--allow-paper-trades-with-audit-blockers",
        action="store_true",
        default=False,
    )
    parser.add_argument("--enable-live-dry-run-orders", action="store_true", default=False)
    parser.add_argument("--reset-demo-account", action="store_true", default=False)
    parser.add_argument("--clear-history", action="store_true", default=False)
    parser.add_argument("--starting-balance-usd", type=float, default=10_000.0)
    parser.add_argument("--max-price-age-seconds", type=float, default=30.0)
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
    parser.add_argument("--ae9-audit-jsonl", type=Path, default=None)
    args = parser.parse_args()

    summary = run_ae10_trading_orchestration(
        project_root=ROOT,
        max_records=args.max_records,
        audit_only=args.audit_only,
        no_db_write=args.no_db_write,
        enable_paper_demo_orders=args.enable_paper_demo_orders,
        allow_paper_trades_with_audit_blockers=args.allow_paper_trades_with_audit_blockers,
        enable_live_dry_run_orders=args.enable_live_dry_run_orders,
        reset_demo_account_flag=args.reset_demo_account,
        clear_history=args.clear_history,
        starting_balance_usd=args.starting_balance_usd,
        max_price_age_seconds=args.max_price_age_seconds,
        provider=args.provider,
        allow_local_qwen=args.allow_local_qwen,
        allow_ollama=args.allow_ollama,
        allow_gemini=args.allow_gemini,
        output_root=args.output_root,
        ae6_jsonl=args.ae6_jsonl,
        ae8_context_jsonl=args.ae8_context_jsonl,
        ae9_audit_jsonl=args.ae9_audit_jsonl,
    )

    compact = {
        "phase": summary.get("phase"),
        "final_status": summary.get("final_status"),
        "traceability_records_created": summary.get("traceability_records_created"),
        "records_with_source_decision_id": summary.get("records_with_source_decision_id"),
        "records_with_source_context_record_id": summary.get("records_with_source_context_record_id"),
        "records_with_source_llm_audit_record_id": summary.get("records_with_source_llm_audit_record_id"),
        "paper_orders_created": summary.get("paper_orders_created"),
        "live_dry_run_orders_created": summary.get("live_dry_run_orders_created"),
        "demo_account_balance": summary.get("demo_account_balance"),
        "reset_demo_account_executed": summary.get("reset_demo_account_executed"),
        "wallet_configured": summary.get("wallet_configured"),
        "real_transaction_attempted": summary.get("real_transaction_attempted"),
        "live_submission_status": summary.get("live_submission_status"),
        "source_paths": summary.get("source_paths"),
        "output_root": summary.get("output_root"),
        "output_paths": summary.get("output_paths"),
        "decision_gate": summary.get("decision_gate"),
        "paper_demo_summary": summary.get("paper_demo_summary"),
        "state_machine_summary": summary.get("state_machine_summary"),
        "price_oracle_summary": summary.get("price_oracle_summary"),
        "execution_latency_summary": summary.get("execution_latency_summary"),
        "live_adapter_summary": summary.get("live_adapter_summary"),
        "llm_provider_summary": summary.get("llm_provider_summary"),
        "safety_confirmation": summary.get("safety_confirmation"),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
