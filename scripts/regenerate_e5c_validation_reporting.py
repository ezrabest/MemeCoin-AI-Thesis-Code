#!/usr/bin/env python3
"""Regenerate E5C validation-selected consensus reporting from existing E5 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.artifacts.registry import detect_project_root  # noqa: E402
from app.training.direct_target_tabicl import regenerate_e5c_reporting_from_artifacts  # noqa: E402

DEFAULT_OUTPUT_ROOT = (
    "data/training/manual_verified_results/phase_e5_direct_target_tabicl_20260703_203824"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate E5C validation-selected consensus reporting (no TAB inference).",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_OUTPUT_ROOT,
        help="Existing E5 output directory containing consensus artifacts.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = detect_project_root(ROOT)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = project_root / str(args.output_root).replace("\\", "/")

    if not output_root.is_dir():
        print(f"Missing E5 output root: {output_root}", file=sys.stderr)
        return 1

    try:
        result = regenerate_e5c_reporting_from_artifacts(output_root)
    except Exception as exc:
        print(f"E5C regeneration failed: {exc}", file=sys.stderr)
        return 2

    print(json.dumps(result, indent=2))
    print(f"Validation-selected rows: {result['validation_selected_rows']}")
    print(f"Wrote: {result['validation_selected_path']}")
    print(f"Wrote: {result['decision_summary_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
