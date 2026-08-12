#!/usr/bin/env python3
"""Build Phase E3 direct net-profitable exit-policy target datasets."""

from __future__ import annotations

import argparse
import json
import sys
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.artifacts.registry import detect_project_root  # noqa: E402
from app.training.direct_target_builder import (  # noqa: E402
    DEFAULT_CHAIN,
    DEFAULT_SOURCE,
    EXIT_COMPARE_EPSILON,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_parquet,
    atomic_write_text,
    build_gap_diagnostic,
    build_invalid_diagnostic,
    build_summary_rows,
    discover_input_files,
    process_combination,
    register_e3_artifacts,
    sort_canonical_df,
    utc_now_iso,
    validate_sqlite_readonly,
    write_canonical_dual,
)
from app.training.direct_target_ids import (  # noqa: E402
    DEFAULT_EXIT_POLICIES,
    DEFAULT_FILTERS,
    DEFAULT_HORIZONS,
    TARGET_NAME,
    output_dataset_basename,
    resolve_time_stop_minutes,
)

E3_DATASET_DIR = "data/training/manual_verified_datasets_direct_target_v1"
E3_REPORT_DIR = "data/training/manual_verified_results/phase_e3_direct_targets_v1"


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "t"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase E3 direct exit-policy target datasets.")
    parser.add_argument(
        "--input-dir",
        default="data/training/manual_verified_datasets_clean_for_model",
    )
    parser.add_argument("--sqlite-db", default="data/trader.db")
    parser.add_argument("--output-dataset-dir", default=E3_DATASET_DIR)
    parser.add_argument("--output-report-dir", default=E3_REPORT_DIR)
    parser.add_argument("--target-version", default="v1")
    parser.add_argument(
        "--filters",
        default=",".join(DEFAULT_FILTERS),
        help="Comma-separated filter names.",
    )
    parser.add_argument(
        "--horizons",
        default=",".join(DEFAULT_HORIZONS),
        help="Comma-separated horizons.",
    )
    parser.add_argument(
        "--exit-policy",
        action="append",
        default=[],
        help="Specific exit_policy_id (repeatable).",
    )
    parser.add_argument(
        "--all-default-exit-policies",
        default="true",
        help="Use default exit policies when --exit-policy is not set.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--register-artifacts",
        default="true",
        help="Register outputs in Phase E1 artifact registry.",
    )
    parser.add_argument("--max-rows-per-dataset", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument(
        "--snapshot-prefetch-mode",
        default="chunked_pair_cache",
        choices=["chunked_pair_cache"],
    )
    parser.add_argument("--max-future-gap-minutes", type=float, default=20.0)
    parser.add_argument("--measure-memory", default="false")
    return parser.parse_args()


def resolve_exit_policies(args: argparse.Namespace) -> list[dict]:
    if args.exit_policy:
        selected = {pid.strip() for pid in args.exit_policy if pid.strip()}
        policies = [p for p in DEFAULT_EXIT_POLICIES if p["exit_policy_id"] in selected]
        missing = selected - {p["exit_policy_id"] for p in policies}
        if missing:
            raise SystemExit(f"Unknown exit_policy_id values: {sorted(missing)}")
        return policies
    if _parse_bool(str(args.all_default_exit_policies)):
        return list(DEFAULT_EXIT_POLICIES)
    return list(DEFAULT_EXIT_POLICIES)


def planned_output_paths(
    output_dataset_dir: Path,
    filters: list[str],
    horizons: list[str],
    policies: list[dict],
    target_version: str,
) -> list[Path]:
    paths: list[Path] = []
    for filter_name in filters:
        for horizon in horizons:
            for policy in policies:
                basename = output_dataset_basename(
                    filter_name,
                    horizon,
                    str(policy["exit_policy_id"]),
                    target_version,
                )
                paths.append(output_dataset_dir / f"{basename}.parquet")
                paths.append(output_dataset_dir / f"{basename}.csv")
    return paths


def main() -> int:
    args = parse_args()
    project_root = detect_project_root(ROOT)
    input_dir = project_root / args.input_dir.replace("\\", "/")
    sqlite_db = project_root / args.sqlite_db.replace("\\", "/")
    output_dataset_dir = project_root / args.output_dataset_dir.replace("\\", "/")
    output_report_dir = project_root / args.output_report_dir.replace("\\", "/")
    filters = [f.strip() for f in args.filters.split(",") if f.strip()]
    horizons = [h.strip() for h in args.horizons.split(",") if h.strip()]
    policies = resolve_exit_policies(args)
    measure_memory = _parse_bool(str(args.measure_memory))
    register_artifacts = _parse_bool(str(args.register_artifacts))

    combos = discover_input_files(input_dir, filters, horizons)
    missing_inputs = [(f, h, p) for f, h, p in combos if not p.exists()]

    sqlite_info = validate_sqlite_readonly(sqlite_db)
    planned = planned_output_paths(
        output_dataset_dir, filters, horizons, policies, args.target_version
    )

    print(f"Phase E3 direct exit target builder")
    print(f"Project root: {project_root}")
    print(f"Input dir: {input_dir}")
    print(f"SQLite DB: {sqlite_db} (read-only)")
    print(f"Output dataset dir: {output_dataset_dir}")
    print(f"Output report dir: {output_report_dir}")
    print(f"Filters: {filters}")
    print(f"Horizons: {horizons}")
    print(f"Exit policies: {[p['exit_policy_id'] for p in policies]}")
    print(f"Combinations: {len(filters) * len(horizons) * len(policies)} output datasets")
    print(f"EXIT_COMPARE_EPSILON: {EXIT_COMPARE_EPSILON}")
    print(f"chunk_size: {args.chunk_size}")
    print(f"snapshot_prefetch_mode: {args.snapshot_prefetch_mode}")
    print(f"max_future_gap_minutes: {args.max_future_gap_minutes}")

    if missing_inputs:
        print("\nMissing input files:")
        for f, h, p in missing_inputs:
            print(f"  - {p}")
        if not args.dry_run:
            return 1

    if not sqlite_info["readable"]:
        print(f"\nSQLite read-only validation failed: {sqlite_info.get('error')}")
        if not args.dry_run:
            return 1
    else:
        print(f"\nSQLite market_snapshots rows: {sqlite_info['row_count_sample']:,}")

    print("\nPlanned output files:")
    for path in planned:
        try:
            display = path.relative_to(project_root)
        except ValueError:
            display = path
        print(f"  - {display}")

    estimated_rows = 0
    for f, h, p in combos:
        if p.exists():
            try:
                import pyarrow.parquet as pq

                estimated_rows += pq.ParquetFile(p).metadata.num_rows * len(policies)
            except Exception:
                pass
    if estimated_rows:
        print(f"\nEstimated total target rows (all policies): {estimated_rows:,}")

    if args.dry_run:
        print("\nDry run complete. No outputs written.")
        return 0

    if args.overwrite:
        output_dataset_dir.mkdir(parents=True, exist_ok=True)
        output_report_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dataset_dir.mkdir(parents=True, exist_ok=True)
        output_report_dir.mkdir(parents=True, exist_ok=True)

    peak_memory_bytes: int | None = None
    if measure_memory:
        tracemalloc.start()

    def memory_tick() -> None:
        nonlocal peak_memory_bytes
        if measure_memory:
            _, peak = tracemalloc.get_traced_memory()
            peak_memory_bytes = max(peak_memory_bytes or 0, peak)

    all_frames = []
    build_results = []

    for filter_name, horizon, input_path in combos:
        if not input_path.exists():
            continue
        for policy in policies:
            print(
                f"Building {filter_name} / {horizon} / {policy['exit_policy_id']} ..."
            )
            frame, result = process_combination(
                filter_name=filter_name,
                horizon=horizon,
                exit_policy=policy,
                input_path=input_path,
                output_dataset_dir=output_dataset_dir,
                sqlite_db=sqlite_db,
                target_version=args.target_version,
                chunk_size=args.chunk_size,
                max_future_gap_minutes=args.max_future_gap_minutes,
                max_rows=args.max_rows_per_dataset,
                chain=DEFAULT_CHAIN,
                source=DEFAULT_SOURCE,
                memory_callback=memory_tick,
            )
            build_results.append(result)
            if not frame.empty:
                all_frames.append(frame)
            memory_tick()

    if measure_memory:
        tracemalloc.stop()

    if all_frames:
        import pandas as pd

        audit_df = sort_canonical_df(pd.concat(all_frames, ignore_index=True))
    else:
        import pandas as pd

        audit_df = pd.DataFrame()

    audit_csv = output_report_dir / "phase_e3_direct_target_audit_rows.csv"
    audit_parquet = output_report_dir / "phase_e3_direct_target_audit_rows.parquet"
    write_canonical_dual(audit_df, audit_csv, audit_parquet)

    summary_df = build_summary_rows(build_results)
    atomic_write_csv(summary_df, output_report_dir / "phase_e3_direct_target_summary.csv")

    invalid_df = build_invalid_diagnostic(audit_df)
    atomic_write_csv(invalid_df, output_report_dir / "phase_e3_invalid_label_diagnostic.csv")

    gap_df = build_gap_diagnostic(audit_df)
    atomic_write_csv(gap_df, output_report_dir / "phase_e3_future_gap_diagnostic.csv")

    total_rows = len(audit_df)
    valid_count = int(audit_df["label_valid"].fillna(False).astype(bool).sum()) if total_rows else 0
    invalid_count = total_rows - valid_count
    positive_count = 0
    if valid_count:
        positive_count = int(
            audit_df.loc[audit_df["label_valid"].fillna(False), "target_net_profitable_after_exit"]
            .fillna(False)
            .astype(bool)
            .sum()
        )
    gap_count = int((audit_df["label_error_code"] == "GAP_IN_FUTURE_DATA").sum()) if total_rows else 0
    if total_rows:
        codes = audit_df["label_error_code"].fillna("").astype(str)
        error_counts = codes[codes != ""].value_counts().to_dict()
    else:
        error_counts = {}

    registry_status: dict = {"attempted": False, "success": False}
    if register_artifacts:
        registry_status = register_e3_artifacts(
            project_root, output_dataset_dir, output_report_dir
        )
        if not registry_status.get("success"):
            print(
                f"WARNING: Artifact registry registration failed: "
                f"{registry_status.get('error')}"
            )

    manifest = {
        "phase": "E3",
        "target_name": TARGET_NAME,
        "target_version": args.target_version,
        "created_at_utc": utc_now_iso(),
        "input_dataset_dir": str(input_dir.relative_to(project_root)).replace("\\", "/"),
        "sqlite_db_path": str(sqlite_db.relative_to(project_root)).replace("\\", "/"),
        "sqlite_access_mode": "read_only",
        "output_dataset_dir": str(output_dataset_dir.relative_to(project_root)).replace("\\", "/"),
        "output_report_dir": str(output_report_dir.relative_to(project_root)).replace("\\", "/"),
        "filters": filters,
        "horizons": horizons,
        "exit_policies": [
            {
                **policy,
                "time_stop_minutes_by_horizon": {
                    h: resolve_time_stop_minutes(h, policy) for h in horizons
                },
            }
            for policy in policies
        ],
        "EXIT_COMPARE_EPSILON": EXIT_COMPARE_EPSILON,
        "max_future_gap_minutes": args.max_future_gap_minutes,
        "chunk_size": args.chunk_size,
        "snapshot_prefetch_mode": args.snapshot_prefetch_mode,
        "measure_memory": measure_memory,
        "row_counts": {
            "total": total_rows,
            "valid_labels": valid_count,
            "invalid_labels": invalid_count,
            "positive_targets": positive_count,
            "positive_target_rate": (positive_count / valid_count) if valid_count else 0.0,
            "gap_in_future_data": gap_count,
        },
        "label_error_code_counts": error_counts,
        "dataset_hashes": [
            {
                "filter": r.filter_name,
                "horizon": r.horizon,
                "exit_policy_id": r.exit_policy_id,
                "schema_hash": r.schema_hash,
                "content_hash": r.content_hash,
                "row_count": r.row_count,
            }
            for r in build_results
        ],
        "artifact_registry_status": registry_status,
        "peak_memory_bytes": peak_memory_bytes,
        "no_runtime_change_confirmation": True,
        "no_sqlite_mutation_confirmation": True,
    }
    atomic_write_json(manifest, output_report_dir / "phase_e3_direct_target_manifest.json")

    upload_lines = [
        "Phase E3 Direct Exit Target Builder Summary",
        f"created_at_utc: {manifest['created_at_utc']}",
        f"total_rows: {total_rows}",
        f"valid_labels: {valid_count}",
        f"invalid_labels: {invalid_count}",
        f"positive_targets: {positive_count}",
        f"positive_target_rate: {manifest['row_counts']['positive_target_rate']:.6f}",
        f"GAP_IN_FUTURE_DATA: {gap_count}",
        f"output_dataset_dir: {manifest['output_dataset_dir']}",
        f"output_report_dir: {manifest['output_report_dir']}",
        "no_runtime_change: true",
        "no_sqlite_mutation: true",
    ]
    atomic_write_text("\n".join(upload_lines) + "\n", output_report_dir / "phase_e3_summary_for_upload.txt")

    if measure_memory and peak_memory_bytes is not None:
        atomic_write_json(
            {"peak_memory_bytes": peak_memory_bytes, "chunk_size": args.chunk_size},
            output_report_dir / "phase_e3_memory_report.json",
        )

    print("\nPhase E3 build complete.")
    print(f"Total rows: {total_rows:,}")
    print(f"Valid labels: {valid_count:,}")
    print(f"Invalid labels: {invalid_count:,}")
    print(f"Positive targets: {positive_count:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
