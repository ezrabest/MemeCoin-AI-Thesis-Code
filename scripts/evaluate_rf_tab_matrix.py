#!/usr/bin/env python3
"""Evaluate RF + TabICLv2 policy matrix (offline, validation-ranked)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.training.baseline_model import MODELS_DIR, TRAINING_DIR
from app.training.rf_tab_matrix import (
    COMBINATION_METHODS,
    DEFAULT_BACKTEST_DIR,
    DEFAULT_GRID_PATH,
    DEFAULT_REPORT_PATH,
    discover_tab_suffixes,
    run_rf_tab_matrix,
    write_rf_tab_matrix_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate RF + TabICLv2 combination policies across Tab prediction "
            "suffixes. Validation ranks policies; test metrics report stability."
        )
    )
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=MODELS_DIR,
        help="Directory containing RF and TabICL prediction parquet files.",
    )
    parser.add_argument(
        "--backtest-dir",
        type=Path,
        default=DEFAULT_BACKTEST_DIR,
        help="Directory containing optional TabICL report JSON files.",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--grid-path",
        type=Path,
        default=DEFAULT_GRID_PATH,
        help="Output flattened policy grid parquet path.",
    )
    parser.add_argument(
        "--tab-suffix",
        action="append",
        dest="tab_suffixes",
        default=None,
        help=(
            "Evaluate only this Tab output suffix (repeatable). "
            "Omit to auto-discover all tabicl_v2_predictions_validation_* files."
        ),
    )
    parser.add_argument(
        "--combination-method",
        action="append",
        dest="combination_methods",
        choices=list(COMBINATION_METHODS),
        default=None,
        help="Combination method to evaluate (repeatable). Default: all methods.",
    )
    parser.add_argument(
        "--join-mismatch-threshold",
        type=float,
        default=0.0,
        help=(
            "Maximum allowed timestamp/pair mismatch rate for row-order fallback join. "
            "Default 0.0 (strict)."
        ),
    )
    parser.add_argument(
        "--min-stability-trade-count",
        type=int,
        default=5,
        help="Minimum validation and test trades required for stable_positive flag.",
    )
    parser.add_argument(
        "--list-suffixes",
        action="store_true",
        help="List discovered Tab prediction suffixes and exit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_suffixes:
        suffixes = discover_tab_suffixes(args.models_dir)
        payload = {
            "models_dir": str(args.models_dir),
            "tab_suffixes": [suffix or "" for suffix in suffixes],
            "count": len(suffixes),
        }
        print(json.dumps(payload, indent=2))
        return 0

    tab_suffixes = args.tab_suffixes
    if tab_suffixes is not None:
        tab_suffixes = [suffix if suffix != "" else None for suffix in tab_suffixes]

    report = run_rf_tab_matrix(
        models_dir=args.models_dir,
        backtest_dir=args.backtest_dir,
        tab_suffixes=tab_suffixes,
        combination_methods=args.combination_methods,
        mismatch_threshold=args.join_mismatch_threshold,
        min_stability_trade_count=args.min_stability_trade_count,
    )
    output_files = write_rf_tab_matrix_outputs(
        report,
        report_path=args.report_path,
        grid_path=args.grid_path,
    )

    summary = {
        "candidates_evaluated": report.get("candidates_evaluated"),
        "candidates_skipped": len(report.get("candidates_skipped") or []),
        "best_candidate": {
            "tab_suffix": (report.get("best_candidate") or {}).get("tab_suffix"),
            "combination_method": (report.get("best_candidate") or {}).get("combination_method"),
            "best_policy": (report.get("best_candidate") or {}).get("best_policy"),
        },
        "output_files": output_files,
    }
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {output_files['report_json']}")
    if "grid_parquet" in output_files:
        print(f"Wrote {output_files['grid_parquet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
