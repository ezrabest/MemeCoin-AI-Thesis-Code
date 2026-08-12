#!/usr/bin/env python3
"""Offline stored Qwen/Ollama overlay evaluation on strict RF policy candidates."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.database import DB_PATH
from app.training.baseline_model import MODELS_DIR
from app.training.llm_overlay import run_llm_overlay_evaluation

VALIDATION_PREDICTIONS_PATH = MODELS_DIR / "predictions_validation.parquet"
TEST_PREDICTIONS_PATH = MODELS_DIR / "predictions_test.parquet"
BACKTEST_DIR = ROOT / "data" / "training" / "policy_backtests"
REPORT_PATH = BACKTEST_DIR / "llm_overlay_report.json"


def main() -> int:
    report = run_llm_overlay_evaluation(
        validation_predictions_path=VALIDATION_PREDICTIONS_PATH,
        test_predictions_path=TEST_PREDICTIONS_PATH,
        db_path=DB_PATH,
        models_dir=MODELS_DIR,
    )
    BACKTEST_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
