#!/usr/bin/env python3
"""Phase E4A — train and evaluate XGB/RF on E3 direct-target datasets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.artifacts.registry import detect_project_root  # noqa: E402
from app.training.direct_target_ids import (  # noqa: E402
    DEFAULT_EXIT_POLICIES,
    DEFAULT_FILTERS,
    DEFAULT_HORIZONS,
)
from app.training.direct_target_xgb_rf import (  # noqa: E402
    TrainConfig,
    _parse_bool,
    discover_direct_target_datasets,
    filter_descriptors,
    load_e3_manifest,
    run_training,
)

E3_DATASET_DIR = "data/training/manual_verified_datasets_direct_target_v1"
E4_OUTPUT_DIR = "data/training/manual_verified_results/phase_e4_direct_target_xgb_rf_v1"
E3_MANIFEST = "data/training/manual_verified_results/phase_e3_direct_targets_v1/phase_e3_direct_target_manifest.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate XGB and RF on Phase E3 direct-target datasets (offline only).",
    )
    parser.add_argument("--input-dir", default=E3_DATASET_DIR)
    parser.add_argument("--output-dir", default=E4_OUTPUT_DIR)
    parser.add_argument("--filter", default=None, help="Filter name, e.g. LIQ_5K_HIGH_ACTIVITY")
    parser.add_argument("--horizon", default=None, help="Horizon, e.g. 1h")
    parser.add_argument("--exit-policy", default=None, dest="exit_policy", help="Exit policy id")
    parser.add_argument(
        "--model",
        default="both",
        choices=["xgb", "rf", "both"],
        help="Model to train (default: both).",
    )
    parser.add_argument("--all", action="store_true", help="Train all discovered E3 direct-target datasets.")
    parser.add_argument("--smoke", action="store_true", help="Smoke mode (single dataset + row limit).")
    parser.add_argument("--overwrite", action="store_true", help="Clean E4 output subfolders before run.")
    parser.add_argument(
        "--register-artifacts",
        default="true",
        help="Register E4 outputs in artifact registry (default: true).",
    )
    parser.add_argument("--xgb-device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--allow-cpu-fallback",
        default="false",
        help="Allow XGB CPU fallback if CUDA fails (default: false).",
    )
    parser.add_argument("--min-train-positives", type=int, default=10)
    parser.add_argument("--min-validation-positives", type=int, default=3)
    parser.add_argument("--min-test-positives", type=int, default=3)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args(argv)


def resolve_models(model_arg: str) -> tuple[str, ...]:
    if model_arg == "both":
        return ("XGB", "RF")
    return (model_arg.upper(),)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = detect_project_root(ROOT)
    input_dir = project_root / args.input_dir.replace("\\", "/")
    output_dir = project_root / args.output_dir.replace("\\", "/")
    manifest_path = project_root / E3_MANIFEST

    if not input_dir.is_dir():
        print(f"Missing input directory: {input_dir}", file=sys.stderr)
        return 1

    manifest = load_e3_manifest(manifest_path)
    descriptors = discover_direct_target_datasets(input_dir, manifest=manifest)

    if not args.all and not args.smoke:
        if not args.filter or not args.horizon or not args.exit_policy:
            print(
                "Specify --filter, --horizon, and --exit-policy, or use --all / --smoke.",
                file=sys.stderr,
            )
            return 2

    filter_name = args.filter
    horizon = args.horizon
    exit_policy = args.exit_policy
    if args.all:
        filter_name = horizon = exit_policy = None
    elif args.smoke and not filter_name:
        filter_name = "LIQ_5K_HIGH_ACTIVITY"
        horizon = horizon or "1h"
        exit_policy = exit_policy or "TP20308_SL080_FEE0308_TIME_BY_HORIZON"

    descriptors = filter_descriptors(
        descriptors,
        filter_name=filter_name,
        horizon=horizon,
        exit_policy=exit_policy,
        smoke=args.smoke and not args.all,
    )
    if not descriptors:
        print("No matching E3 direct-target datasets found.", file=sys.stderr)
        return 3

    config = TrainConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        models=resolve_models(args.model),
        smoke=args.smoke,
        overwrite=args.overwrite,
        register_artifacts=_parse_bool(args.register_artifacts),
        xgb_device=args.xgb_device,
        allow_cpu_fallback=_parse_bool(args.allow_cpu_fallback),
        min_train_positives=args.min_train_positives,
        min_validation_positives=args.min_validation_positives,
        min_test_positives=args.min_test_positives,
        max_rows=args.max_rows,
        random_state=args.random_state,
        selected_descriptors=descriptors,
    )

    print(f"Phase E4A training on {len(descriptors)} dataset(s)")
    print(f"  input:  {input_dir}")
    print(f"  output: {output_dir}")
    print(f"  models: {config.models}")
    print(f"  smoke:  {config.smoke}")

    try:
        result = run_training(config, e3_manifest_path=manifest_path if manifest_path.exists() else None)
    except Exception as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 4

    print(f"Completed datasets: {result['datasets_completed']}")
    if result.get("registration") and not result["registration"].get("success"):
        print(f"WARNING: artifact registration failed: {result['registration'].get('error')}")
        print(f"Repair: {result['registration'].get('repair_command')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
