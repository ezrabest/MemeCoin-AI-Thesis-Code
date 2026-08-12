"""CLI: AE12.7 intelligent-agent operational demo layer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.intelligent_agents import run_ae12_7_agent_demo


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AE12.7 Intelligent-Agent Operational Demo (paper/demo only; no trade authority)"
    )
    p.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument(
        "--mode",
        type=str,
        default="artifact-only",
        choices=[
            "artifact-only",
            "disabled",
            "qwen-local",
            "gemini-selective",
            "helius-readonly",
            "full-demo",
        ],
        help="Operating mode",
    )
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--provider", type=str, default="none", help="ollama|qwen|local|none")
    p.add_argument("--enable-gemini", action="store_true")
    p.add_argument("--enable-helius", action="store_true")
    p.add_argument("--enable-qwen", action="store_true")
    p.add_argument(
        "--no-external-api",
        action="store_true",
        help="Force no external API calls (default unless gemini/helius enabled without this flag)",
    )
    p.add_argument("--allow-external-api", action="store_true", help="Permit external APIs when also enabled")
    p.add_argument("--no-real-wallet", action="store_true", default=True)
    p.add_argument("--readonly", action="store_true", help="Alias: Helius read-only (always true)")
    p.add_argument("--attempt-ollama-live", action="store_true")
    p.add_argument("--gemini-budget", type=int, default=5)
    p.add_argument("--helius-budget", type=int, default=10)
    p.add_argument("--qwen-budget", type=int, default=20)
    p.add_argument("--no-append-daily", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    raw = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv)

    # Default: no external. Explicit --enable-gemini/--enable-helius opens external
    # unless --no-external-api is also present. --allow-external-api always opens.
    no_external = True
    if args.allow_external_api:
        no_external = False
    elif (args.enable_gemini or args.enable_helius) and "--no-external-api" not in raw:
        no_external = False
    if "--no-external-api" in raw and not args.allow_external_api:
        no_external = True

    result = run_ae12_7_agent_demo(
        project_root=Path(args.project_root),
        output_root=Path(args.output_root) if args.output_root else None,
        mode=args.mode,
        limit=args.limit,
        enable_gemini=bool(args.enable_gemini),
        enable_helius=bool(args.enable_helius),
        enable_qwen=bool(args.enable_qwen) or args.provider in {"ollama", "qwen", "local"},
        no_external_api=no_external,
        no_real_wallet=True,
        provider=args.provider,
        attempt_ollama_live=bool(args.attempt_ollama_live),
        gemini_budget=args.gemini_budget,
        helius_budget=args.helius_budget,
        qwen_budget=args.qwen_budget,
        append_daily=not args.no_append_daily,
    )
    print(
        json.dumps(
            {
                "phase": "AE12.7",
                "output_root": result.get("output_root"),
                "classification": result.get("classification"),
                "gate_status": result.get("gate_status"),
                "record_count": result.get("record_count"),
                "trade_authority_used": False,
                "live_ready": False,
                "profitability_proven": False,
                "ae12_closed": False,
            },
            indent=2,
        )
    )
    if str(result.get("classification", "")).startswith("AE12_7_FAIL"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
