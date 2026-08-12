"""CLI: AE12 opportunity analysis over an existing maturation output or fresh smoke."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ae12_forward_evidence.maturation import run_forward_evidence_maturation
from app.ae12_forward_evidence.opportunity_analysis import (
    build_rejection_reason_summary,
    build_strict_vs_exploration_comparison,
    build_trade_vs_no_trade_comparison,
)
from app.ae12_forward_evidence.reports import write_csv, write_json
from app.ae12_forward_evidence.types import Ae12RunConfig, HORIZON_SECONDS, DEFAULT_MISSED_WINNER_THRESHOLDS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AE12 opportunity analysis / comparison layer")
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--from-maturation-root", type=str, default=None, help="Reuse existing evidence CSV/JSON")
    p.add_argument("--max-rows", type=int, default=5000)
    p.add_argument("--horizons", nargs="+", default=["5m", "15m", "1h"])
    p.add_argument("--missed-winner-thresholds-default", action="store_true")
    p.add_argument("--no-external-apis", action="store_true", default=True)
    p.add_argument("--no-real-wallet", action="store_true", default=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args(argv)

    if args.from_maturation_root:
        root = Path(args.from_maturation_root)
        summary_path = root / "reports" / "ae12_forward_evidence_summary.json"
        evidence_path = root / "data" / "ae12_candidate_evidence_rows.csv"
        if not summary_path.is_file():
            print(f"Missing summary: {summary_path}", file=sys.stderr)
            return 1
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        # Re-emit comparison artifacts from summary if evidence CSV too heavy; prefer summary
        out = root / "reports" / "ae12_opportunity_analysis_refresh.json"
        write_json(out, {
            "source_summary": str(summary_path),
            "trade_vs_no_trade": summary.get("trade_vs_no_trade"),
            "strict_vs_exploration": summary.get("strict_vs_exploration"),
            "note": "Refreshed view from existing maturation summary; no recomputation.",
            "evidence_csv": str(evidence_path) if evidence_path.is_file() else None,
        })
        print(f"Wrote {out}")
        return 0

    config = Ae12RunConfig(
        project_root=PROJECT_ROOT,
        output_root=Path(args.output_root) if args.output_root else None,
        resume=bool(args.resume),
        fail_if_output_exists=not bool(args.resume),
        max_rows=args.max_rows,
        horizons=list(args.horizons or HORIZON_SECONDS.keys()),
        missed_winner_thresholds=dict(DEFAULT_MISSED_WINNER_THRESHOLDS),
        no_external_apis=True,
        no_real_wallet=True,
    )
    summary = run_forward_evidence_maturation(config)
    print(f"output_root={summary.get('output_root')}")
    print(f"trade_vs_no_trade={summary.get('trade_vs_no_trade_interpretations')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
