"""CLI: AE13 Live Demo Runtime Acceptance + Virtual Ledger View package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ae13_reconciliation.run import run_ae13_live_demo_runtime_acceptance


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AE13 — Live Demo Runtime Acceptance + Virtual Ledger View + Dynamic Semantic Coverage"
    )
    p.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    p.add_argument(
        "--no-demo-acceptance",
        action="store_true",
        help="Skip DEMO_ACCEPTANCE_MODE paper trade (diagnosis/audit only)",
    )
    p.add_argument(
        "--keep-open",
        action="store_true",
        help="Leave demo acceptance position open (default closes for lifecycle visibility)",
    )
    p.add_argument("--output-root", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_ae13_live_demo_runtime_acceptance(
        project_root=Path(args.project_root),
        enable_demo_acceptance=not bool(args.no_demo_acceptance),
        close_acceptance_trade=not bool(args.keep_open),
        output_root=Path(args.output_root) if args.output_root else None,
    )
    print(
        json.dumps(
            {
                "phase": summary.get("phase"),
                "classification": summary.get("classification"),
                "output_root": summary.get("output_root"),
                "semantic_source_label": summary.get("semantic_source_label"),
                "demo_acceptance_status": (summary.get("demo_acceptance") or {}).get("status"),
                "virtual_ledger_summary": summary.get("virtual_ledger_summary"),
                "live_trading_ready": False,
                "profitability_proven": False,
                "wallet_configured": False,
            },
            indent=2,
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
