#!/usr/bin/env python3
"""Phase E8C — analyze E8B clean RF prediction tails and economic policy robustness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.artifacts.registry import detect_project_root  # noqa: E402
from app.training.clean_rf_policy_tail_audit import AuditConfig, run_audit  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase E8C prediction tail and economic policy audit (offline only).",
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        help="E8B run directory containing predictions/ and reports/",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional explicit output directory (default: timestamped under run-dir).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = detect_project_root(ROOT)
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = project_root / run_dir

    if not run_dir.is_dir():
        print(f"Missing run directory: {run_dir}", file=sys.stderr)
        return 1

    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir

    config = AuditConfig(run_dir=run_dir, output_dir=output_dir)
    print(f"Phase E8C audit on E8B run: {run_dir}")

    try:
        result = run_audit(config)
    except Exception as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        return 2

    counts = result["classification_counts"]
    print(f"Output: {result['output_dir']}")
    print(f"Prediction files analyzed: {result['prediction_files_analyzed']}")
    print(f"Validation rows: {result['validation_rows']}")
    print(f"Test rows: {result['test_rows']}")
    print(f"Validation policies selected: {result['validation_policies_selected']}")
    print(f"Join performed: {result['join_performed']}")
    for label, n in sorted(counts.items()):
        if n:
            print(f"  {label}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
