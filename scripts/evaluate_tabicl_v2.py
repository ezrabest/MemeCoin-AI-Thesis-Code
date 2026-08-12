#!/usr/bin/env python3
"""Offline TabICLv2 evaluation on model_ready_dataset.parquet (train-only context)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.training.baseline_model import DEFAULT_DATASET_PATH
from app.training.tabicl_v2_eval import (
    CONTEXT_STRATEGIES,
    DEFAULT_TARGET,
    evaluate_tabicl_v2,
    tabicl_available,
)


def _parse_top_pcts(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--top-pcts must contain at least one value.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline TabICLv2 evaluation (no live trading).")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--target", default=DEFAULT_TARGET)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--validation-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--top-pcts", default="0.01,0.02,0.05")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "training")
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-train-context-rows", type=int, default=1024)
    parser.add_argument("--scaler", choices=("none", "standard", "robust"), default="standard")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional local TabICLv2 checkpoint (.ckpt). Defaults to Hugging Face cache.",
    )
    parser.add_argument(
        "--partial-reason",
        default=None,
        help="Optional note stored in the report; does not alone mark evaluation partial.",
    )
    parser.add_argument(
        "--context-strategy",
        choices=sorted(CONTEXT_STRATEGIES),
        default=None,
        help="Context selection strategy. Omit to preserve legacy default sampling.",
    )
    parser.add_argument(
        "--output-suffix",
        default=None,
        help="Suffix for output filenames. Defaults to context strategy name when set.",
    )
    parser.add_argument("--ensemble-count", type=int, default=4)
    parser.add_argument("--ensemble-context-size", type=int, default=2048)
    parser.add_argument("--positive-context-ratio", type=float, default=None)
    parser.add_argument(
        "--nearest-neighbor-metric",
        default="euclidean",
        help="Metric for nearest_neighbors_context and whale_wave_context.",
    )
    parser.add_argument(
        "--overwrite-outputs",
        action="store_true",
        help="Allow overwriting existing strategy output files.",
    )
    parser.add_argument(
        "--knn-rolling-days",
        type=int,
        default=None,
        help="Enable rolling KNN for nearest_neighbors_context when >0. Omit for fixed full-train KNN.",
    )
    parser.add_argument(
        "--knn-min-context-rows",
        type=int,
        default=None,
        help=f"Minimum temporal slice rows before window expansion (default {512} when rolling enabled).",
    )
    parser.add_argument(
        "--knn-expand-window",
        default=None,
        help="Expand rolling window up to max days when slice is too small (default true).",
    )
    parser.add_argument(
        "--knn-max-rolling-days",
        type=int,
        default=None,
        help=f"Maximum rolling window days (default {90} when rolling enabled).",
    )
    parser.add_argument(
        "--knn-time-decay-alpha",
        type=float,
        default=None,
        help="Optional time-decay reranking weight (0 disables).",
    )
    parser.add_argument(
        "--knn-cache-time-bucket",
        default=None,
        choices=("day", "hour"),
        help="Cache fitted KNN indexes by this time bucket (default day).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not tabicl_available():
        print(
            "ERROR: tabicl is not installed in this Python environment.\n"
            "Run from .venv-tabicl:\n"
            "  .venv-tabicl\\Scripts\\python.exe scripts/evaluate_tabicl_v2.py",
            file=sys.stderr,
        )
        return 1

    partial_reason = args.partial_reason
    if args.max_rows is not None and partial_reason is None:
        partial_reason = f"max_rows capped at {args.max_rows}"

    try:
        top_pcts = _parse_top_pcts(args.top_pcts)
        report = evaluate_tabicl_v2(
            dataset_path=args.dataset,
            target=args.target,
            max_rows=args.max_rows,
            max_features=args.max_features,
            validation_frac=args.validation_frac,
            test_frac=args.test_frac,
            top_pcts=top_pcts,
            device=args.device,
            output_dir=args.output_dir,
            context_size=args.context_size,
            batch_size=args.batch_size,
            max_train_context_rows=args.max_train_context_rows,
            scaler=args.scaler,
            model_path=args.model_path,
            partial_evaluation_reason=partial_reason,
            context_strategy=args.context_strategy,
            output_suffix=args.output_suffix,
            ensemble_count=args.ensemble_count,
            ensemble_context_size=args.ensemble_context_size,
            positive_context_ratio=args.positive_context_ratio,
            nearest_neighbor_metric=args.nearest_neighbor_metric,
            overwrite_outputs=args.overwrite_outputs,
            knn_rolling_days=args.knn_rolling_days,
            knn_min_context_rows=args.knn_min_context_rows,
            knn_expand_window=args.knn_expand_window,
            knn_max_rolling_days=args.knn_max_rolling_days,
            knn_time_decay_alpha=args.knn_time_decay_alpha,
            knn_cache_time_bucket=args.knn_cache_time_bucket,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except FileExistsError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    summary = {
        "status": "ok",
        "context_strategy": report.get("context_strategy"),
        "output_label": report.get("output_label"),
        "full_evaluation": report.get("full_evaluation"),
        "partial_evaluation_reason": report.get("partial_evaluation_reason"),
        "target": report.get("target"),
        "device_used": report.get("device_used"),
        "cuda_available": report.get("cuda_available"),
        "context_size_used": report.get("context_size_used"),
        "batch_size_used": report.get("batch_size_used"),
        "feature_count": report.get("feature_count"),
        "max_features": report.get("max_features"),
        "train_row_count": report.get("train_row_count"),
        "validation_row_count": report.get("validation_row_count"),
        "test_row_count": report.get("test_row_count"),
        "validation": report.get("tabicl_metrics", {}).get("validation"),
        "test": report.get("tabicl_metrics", {}).get("test"),
        "tabicl_vs_rf": report.get("tabicl_vs_rf"),
        "knn_index_fit_count": report.get("knn_index_fit_count"),
        "knn_cache_hit_count": report.get("knn_cache_hit_count"),
        "knn_cache_miss_count": report.get("knn_cache_miss_count"),
        "output_files": report.get("output_files"),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
