#!/usr/bin/env python3
"""AE20 — Integrated Clean Forward Validation.

Does NOT: start AE21, train, backtest, connect wallet, enable live trading,
grant trade authority, override risk gates, claim profitability, claim live readiness,
use legacy market_snapshots as source of truth, use symbol-only identity joins,
let LLM invent identity links, or silently backfill Clean Forward identity from legacy.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_dotenv() -> None:
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


def _str2bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean, got {value!r}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AE20 Integrated Clean Forward Validation")
    p.add_argument("--smoke-cycles", type=int, default=None)
    p.add_argument("--duration-hours", type=float, default=None)
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--no-external-llm", action="store_true", default=False)
    p.add_argument("--llm-provider", type=str, default="ollama")
    p.add_argument("--max-llm-calls-per-cycle", type=int, default=2)
    p.add_argument("--llm-timeout-seconds", type=float, default=45.0)
    p.add_argument("--paper-demo-only", type=_str2bool, default=True)
    p.add_argument("--clean-forward-only", type=_str2bool, default=True)
    p.add_argument("--strict-baseline-partition", type=_str2bool, default=True)
    p.add_argument("--fail-on-missing-canonical-identity", type=_str2bool, default=True)
    p.add_argument("--separate-strict-exploration-pnl", type=_str2bool, default=True)
    p.add_argument("--max-candidates-per-cycle", type=int, default=8)
    p.add_argument("--cycle-sleep-seconds", type=float, default=0.0)
    p.add_argument("--force-llm-unavailable", action="store_true", default=False)
    p.add_argument(
        "--ae16-bridge-source",
        type=str,
        default=None,
        help="Optional AE16 bridge CSV path (relative to project root or absolute). "
        "Overrides AE20_AE16_BRIDGE_SOURCE env and default relative path.",
    )
    p.add_argument(
        "--ae20-ae16-exact-bridge",
        type=str,
        default=None,
        help="Preferred AE20↔AE16 exact derived bridge CSV "
        "(or set AE20_AE16_EXACT_BRIDGE). Supersedes legacy AE16 bridge authority.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from app.ae20.orchestrator import run_ae20_integrated_clean_forward_validation

    result = run_ae20_integrated_clean_forward_validation(
        ROOT,
        smoke_cycles=args.smoke_cycles,
        duration_hours=args.duration_hours,
        output_root=args.output_root,
        no_external_llm=args.no_external_llm,
        llm_provider=args.llm_provider,
        max_llm_calls_per_cycle=args.max_llm_calls_per_cycle,
        llm_timeout_seconds=args.llm_timeout_seconds,
        paper_demo_only=args.paper_demo_only,
        clean_forward_only=args.clean_forward_only,
        strict_baseline_partition=args.strict_baseline_partition,
        fail_on_missing_canonical_identity=args.fail_on_missing_canonical_identity,
        separate_strict_exploration_pnl=args.separate_strict_exploration_pnl,
        max_candidates_per_cycle=args.max_candidates_per_cycle,
        cycle_sleep_seconds=args.cycle_sleep_seconds,
        force_llm_unavailable=args.force_llm_unavailable,
        ae16_bridge_source=args.ae16_bridge_source,
        ae20_ae16_exact_bridge=args.ae20_ae16_exact_bridge,
    )
    print(json.dumps(
        {
            "classification": result.get("classification"),
            "output_root": result.get("output_root"),
            "smoke_cycles_completed": result.get("smoke_cycles_completed"),
            "profitability_claim": False,
        },
        indent=2,
    ))
    print(f"\noutput_root:\n{result.get('output_root')}")
    print(f"classification:\n{result.get('classification')}")
    classification = str(result.get("classification") or "")
    return 2 if classification.startswith("AE20_SMOKE_BLOCKED") else 0


if __name__ == "__main__":
    raise SystemExit(main())
