"""CLI: AE12.6 ML / meta-layer evaluation summary (read-only reporting)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ae12_ml_meta_layer_evaluation import run_ae12_ml_meta_layer_evaluation


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AE12.6 ML/meta-layer evaluation (not AE13)")
    p.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument(
        "--simulate-missing-critical",
        action="store_true",
        help="Test hook: force HOLD_MISSING_CRITICAL_ARTIFACTS",
    )
    p.add_argument(
        "--simulate-safety-failure",
        action="store_true",
        help="Test hook: force safety/gate failure",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_ae12_ml_meta_layer_evaluation(
        project_root=Path(args.project_root),
        output_root=Path(args.output_root) if args.output_root else None,
        simulate_missing_critical_artifacts=bool(args.simulate_missing_critical),
        simulate_safety_failure=bool(args.simulate_safety_failure),
    )
    if summary.get("gate_status") == "HOLD_OUTPUT_ROOT_WRITE_FAILED":
        print(json.dumps(summary, indent=2), file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "phase": "AE12.6",
                "output_root": summary.get("output_root"),
                "gate_status": summary.get("gate_status"),
                "safety_audit_status": summary.get("safety_audit_status"),
                "ae12_closed": False,
                "live_ready": False,
                "profitability_proven": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
