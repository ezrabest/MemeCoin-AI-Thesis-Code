"""CLI: AE12-SentimentFix local semantic coin classifier (read-only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ae12_sentimentfix.classification_reports import run_semantic_coin_classifier


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AE12-SentimentFix semantic coin classifier (not AE12.6)"
    )
    p.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    p.add_argument("--ae12-root", type=str, required=True)
    p.add_argument("--sentimentfix-root", type=str, default=None)
    p.add_argument("--max-assets", type=int, default=1000)
    p.add_argument("--local-llm-only", action="store_true", default=True)
    p.add_argument("--no-external-apis", action="store_true", default=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_semantic_coin_classifier(
        project_root=Path(args.project_root),
        ae12_root=Path(args.ae12_root),
        sentimentfix_root=Path(args.sentimentfix_root) if args.sentimentfix_root else None,
        max_assets=int(args.max_assets),
        local_llm_only=bool(args.local_llm_only),
        no_external_apis=bool(args.no_external_apis),
    )
    print(
        json.dumps(
            {
                "phase": "AE12-SentimentFix",
                "output_root": summary.get("output_root"),
                "gate_status": summary.get("gate_status"),
                "unique_assets_found": summary.get("unique_assets_found"),
                "unique_assets_classified": summary.get("unique_assets_classified"),
                "unknown_share": summary.get("unknown_share"),
                "trade_authority_used": False,
                "external_api_used": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
