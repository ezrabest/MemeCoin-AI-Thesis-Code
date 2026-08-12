#!/usr/bin/env python3
"""Offline TabICLv2 context-strategy sweep (memory-safe, no live trading)."""
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
    SWEEP_STRATEGIES,
    run_context_strategy_sweep,
    tabicl_available,
)


def _parse_top_pcts(raw: str) -> list[float]:
    values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("--top-pcts must contain at least one value.")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep TabICLv2 context strategies and rank by validation metrics.",
    )
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
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--partial-reason", default=None)
    parser.add_argument(
        "--strategy-sweep",
        action="store_true",
        help="Run the full context strategy sweep (default behavior of this script).",
    )
    parser.add_argument(
        "--strategies",
        default=",".join(SWEEP_STRATEGIES),
        help="Comma-separated subset of context strategies to evaluate.",
    )
    parser.add_argument("--ensemble-count", type=int, default=4)
    parser.add_argument("--ensemble-context-size", type=int, default=2048)
    parser.add_argument("--positive-context-ratio", type=float, default=None)
    parser.add_argument("--nearest-neighbor-metric", default="euclidean")
    parser.add_argument(
        "--overwrite-outputs",
        action="store_true",
        help="Allow overwriting existing sweep and per-strategy outputs.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if not tabicl_available():
        print(
            "ERROR: tabicl is not installed in this Python environment.\n"
            "Run from .venv-tabicl:\n"
            "  .venv-tabicl\\Scripts\\python.exe scripts/sweep_tabicl_v2_context_strategies.py",
            file=sys.stderr,
        )
        return 1

    strategies = [part.strip() for part in args.strategies.split(",") if part.strip()]
    unknown = [s for s in strategies if s not in CONTEXT_STRATEGIES]
    if unknown:
        print(
            f"ERROR: Unknown strategies: {unknown}. "
            f"Valid: {', '.join(sorted(CONTEXT_STRATEGIES))}",
            file=sys.stderr,
        )
        return 1

    partial_reason = args.partial_reason
    if args.max_rows is not None and partial_reason is None:
        partial_reason = f"max_rows capped at {args.max_rows}"

    try:
        top_pcts = _parse_top_pcts(args.top_pcts)
        sweep = run_context_strategy_sweep(
            strategies=strategies,
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
            ensemble_count=args.ensemble_count,
            ensemble_context_size=args.ensemble_context_size,
            positive_context_ratio=args.positive_context_ratio,
            nearest_neighbor_metric=args.nearest_neighbor_metric,
            overwrite_outputs=args.overwrite_outputs,
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
        "best_strategy": sweep.get("best_strategy"),
        "ranking_basis": sweep.get("ranking_basis"),
        "results": sweep.get("results"),
        "output_files": sweep.get("output_files"),
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
