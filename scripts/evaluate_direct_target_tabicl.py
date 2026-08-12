#!/usr/bin/env python3
"""Phase E5 — evaluate TabICL / TabICLv2 on E3 direct-target datasets (offline only)."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.artifacts.registry import detect_project_root  # noqa: E402
from app.training.direct_target_tabicl import (  # noqa: E402
    DEFAULT_CONTEXT_STRATEGIES,
    DEFAULT_MAX_WORKERS,
    SMOKE_DEFAULT_MAX_CONTEXT_SIZE,
    SMOKE_DEFAULT_MAX_ROWS,
    SMOKE_DEFAULT_QUERY_BATCH_SIZE,
    EvalConfig,
    _parse_bool,
    discover_direct_target_datasets,
    load_e3_manifest,
    run_evaluation,
)
from app.training.tabicl_v2_eval import CONTEXT_STRATEGIES  # noqa: E402

E3_DATASET_DIR = "data/training/manual_verified_datasets_direct_target_v1"
E4A_OUTPUT_DIR = (
    "data/training/manual_verified_results/phase_e4_direct_target_xgb_rf_full_20260630_195312"
)
E3_MANIFEST = (
    "data/training/manual_verified_results/phase_e3_direct_targets_v1/phase_e3_direct_target_manifest.json"
)


def _default_output_root(project_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return project_root / f"data/training/manual_verified_results/phase_e5_direct_target_tabicl_{stamp}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase E5 offline direct-target TabICL evaluation (smoke/focused/full).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Smoke mode (default if no mode set).")
    mode.add_argument("--focused", action="store_true", help="Focused regimes from E4A evidence.")
    mode.add_argument("--full", action="store_true", help="All 40 E3 dataset combinations.")

    parser.add_argument("--filters", default=None, help="Comma-separated filter names.")
    parser.add_argument("--horizons", default=None, help="Comma-separated horizons.")
    parser.add_argument("--exit-policies", default=None, dest="exit_policies")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument(
        "--max-context-size",
        type=int,
        default=SMOKE_DEFAULT_MAX_CONTEXT_SIZE,
        help="Hard cap on context size (default smoke: 75).",
    )
    parser.add_argument(
        "--context-sizes",
        default=None,
        help="Comma-separated context sizes (required for focused/full unless max-context-size set).",
    )
    parser.add_argument("--query-batch-size", type=int, default=SMOKE_DEFAULT_QUERY_BATCH_SIZE)
    parser.add_argument("--max-query-batch-size", type=int, default=SMOKE_DEFAULT_QUERY_BATCH_SIZE)
    parser.add_argument(
        "--context-strategies",
        default=",".join(DEFAULT_CONTEXT_STRATEGIES),
        help=f"Comma-separated strategies from {sorted(CONTEXT_STRATEGIES)}",
    )
    parser.add_argument("--output-root", default=None, dest="output_root")
    parser.add_argument("--e3-dataset-root", default=E3_DATASET_DIR, dest="e3_dataset_root")
    parser.add_argument("--e4a-output-root", default=E4A_OUTPUT_DIR, dest="e4a_output_root")
    parser.add_argument(
        "--register-artifacts",
        default="true",
        help="Register E5 outputs in artifact registry (default: true).",
    )
    parser.add_argument(
        "--skip-registry-registration",
        action="store_true",
        help="Skip E5 artifact registry registration.",
    )
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--concurrency", type=int, default=None, help="Alias for --max-workers.")
    parser.add_argument(
        "--context-cache-mode",
        default="off",
        choices=["off", "cpu_only", "disk"],
    )
    parser.add_argument("--max-context-cache-entries", type=int, default=1)
    parser.add_argument(
        "--fail-on-missing-e4a-registry",
        default="true",
        help="Fail dependency audit if E4A registry entries missing.",
    )
    parser.add_argument(
        "--allow-registry-warnings",
        action="store_true",
        help="Allow dependency audit warnings for registry gaps.",
    )
    parser.add_argument(
        "--skip-tab-inference",
        action="store_true",
        help="Skip TabICL GPU inference (smoke/tests; uses deterministic fallback scores).",
    )
    return parser.parse_args(argv)


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = detect_project_root(ROOT)
    input_dir = project_root / args.e3_dataset_root.replace("\\", "/")
    e4a_root = project_root / args.e4a_output_root.replace("\\", "/")
    output_dir = (
        Path(args.output_root)
        if args.output_root
        else _default_output_root(project_root)
    )
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir

    smoke = not args.focused and not args.full
    if args.smoke:
        smoke = True

    max_workers = args.concurrency if args.concurrency is not None else args.max_workers
    max_workers = max(1, int(max_workers))

    context_strategies = tuple(_split_csv(args.context_strategies) or list(DEFAULT_CONTEXT_STRATEGIES))
    context_sizes_raw = _split_csv(args.context_sizes)
    if context_sizes_raw:
        context_sizes = tuple(int(x) for x in context_sizes_raw)
    else:
        context_sizes = (int(args.max_context_size),)

    if (args.focused or args.full) and not context_sizes_raw and args.max_context_size == SMOKE_DEFAULT_MAX_CONTEXT_SIZE:
        print(
            "WARNING: focused/full run using smoke default max_context_size; "
            "pass --context-sizes or explicit --max-context-size for production runs.",
            file=sys.stderr,
        )

    if not input_dir.is_dir():
        print(f"Missing E3 dataset root: {input_dir}", file=sys.stderr)
        return 1

    manifest_path = project_root / E3_MANIFEST
    manifest = load_e3_manifest(manifest_path) if manifest_path.exists() else None
    if not discover_direct_target_datasets(input_dir, manifest=manifest):
        print("No E3 direct-target datasets discovered.", file=sys.stderr)
        return 3

    config = EvalConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        e4a_root=e4a_root,
        smoke=smoke,
        focused=bool(args.focused),
        full=bool(args.full),
        context_strategies=context_strategies,
        context_sizes=context_sizes,
        max_context_size=int(args.max_context_size),
        query_batch_size=int(args.query_batch_size),
        max_query_batch_size=int(args.max_query_batch_size),
        max_workers=max_workers,
        context_cache_mode=args.context_cache_mode,
        max_context_cache_entries=int(args.max_context_cache_entries),
        register_artifacts=not args.skip_registry_registration and _parse_bool(args.register_artifacts),
        random_state=int(args.random_state),
        max_rows=args.max_rows if args.max_rows is not None else (SMOKE_DEFAULT_MAX_ROWS if smoke else None),
        device=args.device,
        fail_on_missing_e4a_registry=_parse_bool(args.fail_on_missing_e4a_registry),
        allow_registry_warnings=bool(args.allow_registry_warnings),
        skip_tab_inference=bool(args.skip_tab_inference),
    )

    print("Phase E5 direct-target TabICL evaluation")
    print(f"  mode:     {'smoke' if smoke else 'focused' if args.focused else 'full'}")
    print(f"  input:    {input_dir}")
    print(f"  e4a:      {e4a_root}")
    print(f"  output:   {output_dir}")
    print(f"  workers:  {max_workers}")
    print(f"  context:  max={config.max_context_size} sizes={context_sizes}")
    print(f"  strategies: {context_strategies}")

    try:
        result = run_evaluation(
            config,
            e3_manifest_path=manifest_path if manifest_path.exists() else None,
            filters=_split_csv(args.filters),
            horizons=_split_csv(args.horizons),
            exit_policies=_split_csv(args.exit_policies),
        )
    except Exception as exc:
        print(f"Run failed: {exc}", file=sys.stderr)
        return 4

    if result.get("status") == "dependency_failed":
        print(f"Dependency audit failed: {result['dependency_audit'].get('failure_reason')}", file=sys.stderr)
        print(f"Audit written under: {output_dir / 'audit'}")
        return 5

    print(f"Successful jobs: {result.get('successful_jobs', 0)}")
    print(f"Processed jobs: {result.get('processed_jobs', 0)}")
    if result.get('skipped_jobs'):
        print(f"Skipped jobs: {result.get('skipped_jobs')}")
    if result.get('failed_jobs'):
        print(f"Failed jobs: {result.get('failed_jobs')}")
    if result.get("registration") and not result["registration"].get("success"):
        print(f"WARNING: artifact registration failed: {result['registration'].get('error')}")
        print(f"Repair: {result['registration'].get('repair_command')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
