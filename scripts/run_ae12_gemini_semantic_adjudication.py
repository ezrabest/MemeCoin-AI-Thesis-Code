"""CLI: AE12-SentimentFix Gemini semantic adjudication (reporting only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ae12_sentimentfix.adjudication_reports import run_gemini_semantic_adjudication
from app.env_bootstrap import load_environment


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AE12-SentimentFix Gemini semantic adjudication (not AE12.6)")
    p.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    p.add_argument("--classifier-root", type=str, required=True)
    p.add_argument("--max-assets", type=int, default=100)
    p.add_argument("--use-gemini", action="store_true", default=False)
    p.add_argument("--allow-external-apis", action="store_true", default=False)
    p.add_argument("--semantic-reporting-only", action="store_true", default=True)
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument("--force-refresh", action="store_true", default=False)
    p.add_argument("--only-suspected", action="store_true", default=False)
    p.add_argument("--only-unknown", action="store_true", default=False)
    p.add_argument("--no-web-grounding", action="store_true", default=False)
    p.add_argument("--allow-model-knowledge-fallback", action="store_true", default=False)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    load_environment(PROJECT_ROOT)
    summary = run_gemini_semantic_adjudication(
        project_root=Path(args.project_root),
        classifier_root=Path(args.classifier_root),
        max_assets=int(args.max_assets),
        use_gemini=bool(args.use_gemini),
        allow_external_apis=bool(args.allow_external_apis),
        semantic_reporting_only=bool(args.semantic_reporting_only),
        dry_run=bool(args.dry_run),
        force_refresh=bool(args.force_refresh),
        only_suspected=bool(args.only_suspected),
        only_unknown=bool(args.only_unknown),
        no_web_grounding=bool(args.no_web_grounding),
        allow_model_knowledge_fallback=bool(args.allow_model_knowledge_fallback),
    )
    gate_path = Path(summary["output_root"]) / "audits" / "ae12_gemini_semantic_adjudication_gate.json"
    gate = json.loads(gate_path.read_text(encoding="utf-8")) if gate_path.is_file() else {}
    print(
        json.dumps(
            {
                "phase": "AE12-SentimentFix",
                "output_root": summary.get("output_root"),
                "gate_status": summary.get("gate_status"),
                "unique_assets_input": gate.get("unique_assets_input"),
                "unique_assets_adjudicated": gate.get("unique_assets_adjudicated"),
                "social_confirmed_count": gate.get("social_confirmed_count"),
                "opportunistic_suspected_count": gate.get("opportunistic_suspected_count"),
                "non_social_opportunistic_confirmed_count": gate.get("non_social_opportunistic_confirmed_count"),
                "external_api_used": gate.get("external_api_used"),
                "gemini_used": gate.get("gemini_used"),
                "trade_authority_used": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
