"""CLI: AE12.5 Signal Taxonomy Audit (read-only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ae12_signal_taxonomy import run_signal_taxonomy_audit


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AE12.5 Signal Taxonomy Audit - dual axis QA")
    p.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    p.add_argument("--ae12-root", type=str, default=None)
    p.add_argument("--max-rows-per-source", type=int, default=5000)
    p.add_argument("--no-external-apis", action="store_true", default=True)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_signal_taxonomy_audit(
        project_root=Path(args.project_root),
        ae12_root=Path(args.ae12_root) if args.ae12_root else None,
        max_rows_per_source=int(args.max_rows_per_source),
        no_external_apis=bool(args.no_external_apis),
    )
    print(json.dumps(
        {
            "output_root": summary.get("output_root"),
            "gate_status": summary.get("gate_status"),
            "rows_sampled_total": summary.get("rows_sampled_total"),
            "live_trading_ready": False,
            "profitability_proven": False,
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
