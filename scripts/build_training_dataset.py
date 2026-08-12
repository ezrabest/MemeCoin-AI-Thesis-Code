#!/usr/bin/env python3
"""Build outcome-labeled training datasets from SQLite (offline batch)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import database as db
from app.training.dataset_builder import build_training_datasets, print_run_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Build training outcome datasets from SQLite.")
    parser.add_argument(
        "--include-pending",
        action="store_true",
        help="Include rows with pending_outcome=true in model_ready_dataset",
    )
    args = parser.parse_args()

    db.init_pool()
    report = build_training_datasets(include_pending=args.include_pending)
    print_run_report(report)
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
