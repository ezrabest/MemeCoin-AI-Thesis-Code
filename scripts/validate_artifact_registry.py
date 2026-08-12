#!/usr/bin/env python3
"""Validate the Phase E1 artifact registry."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.artifacts.registry import detect_project_root, validate_registry, write_validation_report  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate artifact registry JSONL.")
    parser.add_argument(
        "--registry-path",
        default="data/training/artifact_registry/artifact_registry.jsonl",
        help="Path to registry JSONL file.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        help="Override validation report output path.",
    )
    parser.add_argument("--fail-on-error", action="store_true", help="Exit nonzero when errors exist.")
    parser.add_argument("--verbose", action="store_true", help="Print validation details.")
    return parser.parse_args()


def default_report_path(project_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return project_root / "data" / "audits" / f"artifact_registry_validation_{stamp}.json"


def main() -> int:
    args = parse_args()
    project_root = detect_project_root(ROOT)
    registry_path = Path(args.registry_path)
    if not registry_path.is_absolute():
        registry_path = project_root / registry_path

    report_path = Path(args.report_path) if args.report_path else default_report_path(project_root)
    if not report_path.is_absolute():
        report_path = project_root / report_path

    report = validate_registry(registry_path, project_root=project_root, verbose=args.verbose)
    write_validation_report(report, report_path)

    if args.verbose:
        print(f"Validation status: {report['status']}")
        print(f"Total records: {report['total_records']}")
        print(f"Errors: {len(report['errors'])}")
        print(f"Warnings: {len(report['warnings'])}")
        print(f"Report: {report_path}")

    if args.fail_on_error and report["errors"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
