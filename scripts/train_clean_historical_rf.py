#!/usr/bin/env python3
"""Phase E8B — train clean historical Random Forest on direct-target datasets (offline only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.artifacts.registry import detect_project_root  # noqa: E402
from app.training.clean_historical_rf import (  # noqa: E402
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_ROOT,
    TrainConfig,
    discover_direct_target_datasets,
    filter_descriptors,
    make_output_dir,
    run_training,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase E8B clean historical RF training (offline research only).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Smoke mode: one dataset, row cap, n_jobs=1.")
    mode.add_argument("--full", action="store_true", help="Full mode: all RAW_ALL_VERIFIED horizons and policies.")
    parser.add_argument(
        "--filters",
        nargs="+",
        default=["RAW_ALL_VERIFIED"],
        help="Dataset filters (default: RAW_ALL_VERIFIED).",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        default=["30m", "1h", "4h", "8h", "24h"],
        help="Horizons to include.",
    )
    parser.add_argument(
        "--exit-policies",
        nargs="+",
        default=[
            "TP20308_SL075_FEE0308_TIME_BY_HORIZON",
            "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
        ],
        help="Exit policy ids.",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument(
        "--class-weight",
        default="balanced_subsample",
        choices=["none", "balanced", "balanced_subsample"],
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.smoke and not args.full:
        print("Specify --smoke or --full.", file=sys.stderr)
        return 2

    project_root = detect_project_root(ROOT)
    dataset_root = project_root / args.dataset_root.replace("\\", "/")
    output_root = project_root / args.output_root.replace("\\", "/")

    if not dataset_root.is_dir():
        print(f"Missing dataset root: {dataset_root}", file=sys.stderr)
        return 1

    n_jobs = args.n_jobs if args.n_jobs is not None else (1 if args.smoke else -1)
    output_dir = make_output_dir(output_root)

    descriptors = discover_direct_target_datasets(dataset_root)
    descriptors = filter_descriptors(
        descriptors,
        filters=tuple(args.filters),
        horizons=tuple(args.horizons),
        exit_policies=tuple(args.exit_policies),
        smoke=args.smoke,
    )
    if not descriptors:
        print("No matching direct-target datasets found.", file=sys.stderr)
        return 3

    config = TrainConfig(
        dataset_root=dataset_root,
        output_dir=output_dir,
        filters=tuple(args.filters),
        horizons=tuple(args.horizons),
        exit_policies=tuple(args.exit_policies),
        smoke=args.smoke,
        full=args.full,
        max_rows=args.max_rows,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=args.class_weight,
        random_state=args.random_state,
        n_jobs=n_jobs,
        selected_descriptors=descriptors,
    )

    print(f"Phase E8B clean historical RF on {len(descriptors)} dataset(s)")
    print(f"  dataset_root: {dataset_root}")
    print(f"  output_dir:   {output_dir}")
    print(f"  smoke:        {config.smoke}")
    print(f"  full:         {config.full}")
    print(f"  n_jobs:       {config.n_jobs}")

    try:
        result = run_training(config)
    except Exception as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 4

    print(f"Completed datasets: {result['datasets_completed']}")
    print(f"Failed datasets:    {result['datasets_failed']}")
    print(f"Models trained:     {result['models_trained']}")
    print(f"Output:             {result['output_dir']}")
    return 0 if result["datasets_failed"] == 0 else 5


if __name__ == "__main__":
    raise SystemExit(main())
