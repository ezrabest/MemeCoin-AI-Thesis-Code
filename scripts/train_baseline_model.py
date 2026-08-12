#!/usr/bin/env python3
"""Train baseline sklearn classifiers on model_ready_dataset.parquet."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.training.baseline_model import train_baseline_models


def main() -> int:
    report = train_baseline_models()
    print(json.dumps({
        "status": "ok",
        "targets_trained": report.get("targets_trained"),
        "targets_skipped": report.get("targets_skipped"),
        "best_model_by_target": report.get("best_model_by_target"),
        "features_used_count": report.get("features_used_count"),
        "output_files": report.get("output_files"),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
