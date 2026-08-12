#!/usr/bin/env python3
"""Register existing research artifacts into the Phase E1 file-based registry."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.artifacts.registry import (  # noqa: E402
    DEFAULT_SCAN_ROOTS,
    build_registry_summary,
    detect_project_root,
    get_git_commit_hash,
    load_registry,
    scan_artifacts,
    write_registry,
    write_registry_csv,
    write_registry_jsonl,
    write_registry_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register existing research artifacts.")
    parser.add_argument("--dry-run", action="store_true", help="Scan only; do not write registry files.")
    parser.add_argument(
        "--output-dir",
        default="data/training/artifact_registry",
        help="Directory for registry outputs.",
    )
    parser.add_argument(
        "--include-root",
        action="append",
        default=[],
        help="Additional scan root relative to project root (repeatable).",
    )
    parser.add_argument(
        "--registry-path",
        default=None,
        help="Override JSONL registry path.",
    )
    parser.add_argument("--write-csv", action="store_true", help="Write artifact_registry.csv.")
    parser.add_argument("--write-summary", action="store_true", help="Write artifact_registry_summary.json.")
    parser.add_argument(
        "--max-file-size-mb",
        type=float,
        default=None,
        help="Skip content hashing above this size (MB).",
    )
    parser.add_argument("--force-rehash", action="store_true", help="Ignore cached hashes from existing registry.")
    parser.add_argument("--verbose", action="store_true", help="Print progress details.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project_root = detect_project_root(ROOT)
    output_dir = project_root / args.output_dir.replace("\\", "/")
    registry_path = (
        Path(args.registry_path)
        if args.registry_path
        else output_dir / "artifact_registry.jsonl"
    )
    if not registry_path.is_absolute():
        registry_path = project_root / registry_path

    scan_roots = list(DEFAULT_SCAN_ROOTS)
    for extra in args.include_root:
        if extra not in scan_roots:
            scan_roots.append(extra.replace("\\", "/"))

    git_commit_hash, git_warnings = get_git_commit_hash(project_root)
    previous_registry = {} if args.force_rehash else load_registry(registry_path)

    records, scan_warnings = scan_artifacts(
        project_root=project_root,
        scan_roots=scan_roots,
        branch_name="phase_e1_artifact_registry",
        generated_by_script="scripts/register_existing_artifacts.py",
        previous_registry=previous_registry,
        force_rehash=args.force_rehash,
        max_file_size_mb=args.max_file_size_mb,
        git_commit_hash=git_commit_hash,
        git_warnings=git_warnings,
    )

    summary = build_registry_summary(
        records,
        registry_path=str(registry_path.relative_to(project_root)).replace("\\", "/"),
        scan_roots=scan_roots,
        git_commit_hash=git_commit_hash,
        warnings_count=len(scan_warnings),
        errors_count=0,
    )

    if args.verbose:
        print(f"Project root: {project_root}")
        print(f"Scan roots: {len(scan_roots)}")
        print(f"Artifacts found: {len(records)}")
        print(f"Hash status: {summary['count_by_hash_status']}")
        for warning in scan_warnings[:20]:
            print(f"WARNING: {warning}")
        if len(scan_warnings) > 20:
            print(f"... and {len(scan_warnings) - 20} more scan warnings")
        for record in records[:3]:
            print("SAMPLE:", json.dumps(record.to_dict(), ensure_ascii=False, default=str)[:500])

    if args.dry_run:
        print(f"Dry run complete. Would register {len(records)} artifacts.")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "artifact_registry.csv" if args.write_csv else None
    summary_path = output_dir / "artifact_registry_summary.json" if args.write_summary else None

    write_registry_jsonl(records, registry_path)
    if csv_path is not None:
        write_registry_csv(records, csv_path)
    if summary_path is not None:
        write_registry_summary(summary, summary_path)

    if args.verbose:
        print(f"Wrote registry: {registry_path}")
        if csv_path:
            print(f"Wrote CSV: {csv_path}")
        if summary_path:
            print(f"Wrote summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
