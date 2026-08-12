#!/usr/bin/env python3
"""Phase E8E — unified rare-winner context forensics audit (offline, read-only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.artifacts.registry import detect_project_root  # noqa: E402
from app.database import DB_PATH  # noqa: E402
from app.training.rare_winner_context_forensics import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    WINDOWS,
    ForensicsConfig,
    make_output_dir,
    run_forensics,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase E8E rare-winner context forensics audit (offline only).",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true", help="Smoke mode: tiny sample, all schemas.")
    mode.add_argument("--full", action="store_true", help="Full mode: all rare-winner rows with checkpointing.")
    parser.add_argument("--e8b-run-dir", required=True)
    parser.add_argument("--e8c-dir", required=True)
    parser.add_argument("--sqlite-db", default=None)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--max-controls-per-candidate", type=int, default=20)
    parser.add_argument("--windows", nargs="+", default=list(WINDOWS))
    parser.add_argument("--force", action="store_true", help="Ignore checkpoints and reprocess.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = detect_project_root(ROOT)
    e8b_run_dir = Path(args.e8b_run_dir)
    e8c_dir = Path(args.e8c_dir)
    if not e8b_run_dir.is_absolute():
        e8b_run_dir = project_root / e8b_run_dir
    if not e8c_dir.is_absolute():
        e8c_dir = project_root / e8c_dir
    if not e8b_run_dir.is_dir():
        print(f"Missing E8B run dir: {e8b_run_dir}", file=sys.stderr)
        return 1
    if not e8c_dir.is_dir():
        print(f"Missing E8C dir: {e8c_dir}", file=sys.stderr)
        return 2

    sqlite_db = Path(args.sqlite_db) if args.sqlite_db else DB_PATH
    if args.sqlite_db and not sqlite_db.is_absolute():
        sqlite_db = project_root / sqlite_db

    output_root = project_root / args.output_root.replace("\\", "/")
    output_dir = make_output_dir(output_root)

    config = ForensicsConfig(
        e8b_run_dir=e8b_run_dir,
        e8c_dir=e8c_dir,
        output_dir=output_dir,
        sqlite_db=sqlite_db,
        smoke=args.smoke,
        full=args.full,
        force=args.force,
        max_candidates=args.max_candidates,
        max_controls_per_candidate=args.max_controls_per_candidate,
        windows=tuple(args.windows),
    )

    print(f"Phase E8E forensics ({'smoke' if config.smoke else 'full'})")
    print(f"  E8B: {e8b_run_dir}")
    print(f"  E8C: {e8c_dir}")
    print(f"  SQLite: {sqlite_db}")
    print(f"  Output: {output_dir}")

    try:
        result = run_forensics(config, project_root=project_root)
    except Exception as exc:
        print(f"Forensics failed: {exc}", file=sys.stderr)
        return 3

    print(f"Output: {result['output_dir']}")
    print(f"Candidate rows: {result['candidate_rows']}")
    print(f"SQLite tables inventoried: {result['sqlite_tables']}")
    print(f"Reservoir files: {result['reservoir_files']}")
    print(f"Final classification: {result['final_classification'].get('final_classification')}")
    if result.get("fatal_blockers"):
        print("Fatal blockers for full mode:")
        for blocker in result["fatal_blockers"]:
            print(f"  - {blocker}")
        return 4 if config.full else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
