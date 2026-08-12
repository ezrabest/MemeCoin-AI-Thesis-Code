#!/usr/bin/env python3
"""AE19 — Original E9 Repair: LLM Operational Layer.

Does NOT: start AE20, train, backtest, mutate trader.db, connect wallet,
load private keys, sign/submit transactions, enable live trading, claim profitability,
grant trade authority, override RiskGuard/GateKeeper, or invent identity links.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
    """Load .env into process env if present. Never print secret values."""
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    try:
        for line in env_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, _, val = raw.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in __import__("os").environ:
                __import__("os").environ[key] = val
    except OSError:
        pass


_load_dotenv()

from app.llm_operational import PHASE  # noqa: E402
from app.llm_operational.orchestrator import run_ae19_llm_operational_layer  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AE19 LLM Operational Layer")
    p.add_argument("--ae17-root", type=str, default=None)
    p.add_argument("--ae16-root", type=str, default=None)
    p.add_argument("--ae18-root", type=str, default=None)
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--max-candidates", type=int, default=20)
    p.add_argument("--max-tasks-per-type", type=int, default=20)
    p.add_argument(
        "--allow-qwen",
        action="store_true",
        default=False,
        help="Explicitly enable Qwen/Ollama attempts (default still probes unless disabled)",
    )
    p.add_argument(
        "--disallow-qwen",
        action="store_true",
        default=False,
        help="Disable Qwen/Ollama for this run",
    )
    p.add_argument(
        "--allow-gemini",
        action="store_true",
        default=False,
        help="Explicitly enable selective Gemini audit (opt-in)",
    )
    p.add_argument(
        "--force-qwen-unavailable",
        action="store_true",
        default=False,
    )
    p.add_argument(
        "--force-gemini-unavailable",
        action="store_true",
        default=False,
    )
    p.add_argument(
        "--allow-mock-diagnostic",
        action="store_true",
        default=False,
        help="Use mock diagnostic responses only (never counted as provider success)",
    )
    p.add_argument("--gemini-selective-budget", type=int, default=3)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    allow_qwen: bool | None
    if args.disallow_qwen:
        allow_qwen = False
    elif args.allow_qwen:
        allow_qwen = True
    else:
        allow_qwen = None

    allow_gemini: bool | None = True if args.allow_gemini else None

    try:
        result = run_ae19_llm_operational_layer(
            ROOT,
            output_root=args.output_root,
            ae17_root=args.ae17_root,
            ae16_root=args.ae16_root,
            ae18_root=args.ae18_root,
            max_candidates=args.max_candidates,
            max_tasks_per_type=args.max_tasks_per_type,
            allow_qwen=allow_qwen,
            allow_gemini=allow_gemini,
            force_qwen_unavailable=args.force_qwen_unavailable,
            force_gemini_unavailable=args.force_gemini_unavailable,
            use_mock_diagnostic=args.allow_mock_diagnostic,
            gemini_selective_budget=args.gemini_selective_budget,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[{PHASE}] unexpected error: {exc}", file=sys.stderr)
        traceback.print_exc()
        return 1

    classification = result.get("classification")
    print(f"[{PHASE}] classification: {classification}")
    print(f"[{PHASE}] output_root: {result.get('output_root')}")
    pc = result.get("provider_counts") or {}
    tc = result.get("task_counts") or {}
    print(f"[{PHASE}] real_provider_success_count: {pc.get('real_provider_success_count')}")
    print(f"[{PHASE}] provider_unavailable_count: {pc.get('provider_unavailable_count')}")
    print(f"[{PHASE}] mock_diagnostic_count: {pc.get('mock_diagnostic_count')}")
    print(f"[{PHASE}] mock_counted_as_real_success_count: {pc.get('mock_counted_as_real_success_count')}")
    print(f"[{PHASE}] candidate_memo_count: {tc.get('candidate_memo_count')}")
    print(f"[{PHASE}] risk_explanation_count: {tc.get('risk_explanation_count')}")
    print(f"[{PHASE}] missed_winner_review_count: {tc.get('missed_winner_review_count')}")
    print(f"[{PHASE}] semantic_conflict_review_count: {tc.get('semantic_conflict_review_count')}")
    print(f"[{PHASE}] context_summary_count: {tc.get('context_summary_count')}")
    print(f"[{PHASE}] audit_record_count: {result.get('audit_record_count')}")
    print(f"[{PHASE}] ae20_status: {result.get('ae20_status')}")

    if str(classification).startswith("AE19_BLOCKED_"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
