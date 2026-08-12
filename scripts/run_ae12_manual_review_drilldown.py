"""CLI: AE12-SentimentFix local manual-review drilldown (no Gemini / no external APIs)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ae12_sentimentfix.manual_review_drilldown import run_manual_review_drilldown


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AE12-SentimentFix local manual-review drilldown (not AE12.6)"
    )
    p.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    p.add_argument(
        "--gemini-root",
        type=str,
        required=True,
        help="Existing Gemini adjudication root (read-only source)",
    )
    p.add_argument(
        "--no-external-apis",
        action="store_true",
        default=True,
        help="Require local-only mode (default True)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_manual_review_drilldown(
        project_root=Path(args.project_root),
        gemini_root=Path(args.gemini_root),
        no_external_apis=bool(args.no_external_apis),
    )
    print(
        json.dumps(
            {
                "phase": "AE12-SentimentFix",
                "not_ae12_6": True,
                "output_root": summary.get("output_root"),
                "gate_status": summary.get("gate_status"),
                "drilldown_rule_version": summary.get("drilldown_rule_version"),
                "manual_review_input_count": summary.get("manual_review_input_count"),
                "manual_review_resolved_count": summary.get("manual_review_resolved_count"),
                "unknown_unresolved_count": summary.get("unknown_unresolved_count"),
                "manual_review_remaining_count": summary.get("manual_review_remaining_count"),
                "updated_coin_level_counts": summary.get("updated_coin_level_counts"),
                "resolution_rule_distribution": summary.get("resolution_rule_distribution"),
                "external_api_used": False,
                "gemini_called_again": False,
                "trade_authority_used": False,
                "live_ready": False,
                "profitability_proven": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
