"""Phase E3 direct net-profitable exit-policy target dataset builder."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
import pandas as pd

from app.artifacts.hash_utils import compute_content_hash, read_tabular_schema, sha256_hex
from app.candidates.schema import compute_candidate_id
from app.candidates.validation import normalize_event_timestamp
from app.training.direct_target_ids import (
    DEFAULT_EXIT_POLICIES,
    DEFAULT_FILTERS,
    DEFAULT_HORIZONS,
    NOT_APPLICABLE,
    TARGET_NAME,
    compute_candidate_policy_id,
    compute_target_row_id,
    input_dataset_filename,
    label_source_artifact_id_for_input,
    output_dataset_basename,
    resolve_time_stop_minutes,
)
from app.training.exit_path_simulation import (
    EXIT_COMPARE_EPSILON,
    ExitSimulationResult,
    simulate_exit_path,
)

__all__ = ["EXIT_COMPARE_EPSILON"]

DEFAULT_CHAIN = "solana"
DEFAULT_SOURCE = "manual_verified_clean_model_input"

SORT_COLUMNS = [
    "filter",
    "horizon",
    "exit_policy_id",
    "pair_address",
    "event_timestamp",
    "candidate_id",
    "candidate_policy_id",
    "target_row_id",
]

OUTCOME_COLUMNS = frozenset(
    {
        "target_net_profitable_after_exit",
        "sim_net_return",
        "sim_exit_status",
        "exit_ratio",
        "max_future_ratio",
        "min_future_ratio",
        "label_valid",
        "label_error_code",
        "label_error_detail",
        "entry_snapshot_timestamp",
        "entry_price_raw",
        "entry_price",
        "entry_snapshot_id",
        "future_window_start_timestamp",
        "future_window_end_timestamp",
        "first_future_snapshot_timestamp",
        "last_future_snapshot_timestamp",
        "future_snapshot_count",
        "max_future_gap_minutes",
        "gap_detected",
        "gap_start_timestamp",
        "gap_end_timestamp",
        "gap_minutes",
        "exit_timestamp",
        "target_name",
        "target_version",
        "candidate_policy_id",
        "target_row_id",
        "label_source_artifact_id",
    }
)


def utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_json(payload: dict[str, Any], path: Path) -> None:
    atomic_write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), path)


def sort_canonical_df(df: pd.DataFrame) -> pd.DataFrame:
    present = [col for col in SORT_COLUMNS if col in df.columns]
    if not present:
        return df
    return df.sort_values(present, kind="mergesort").reset_index(drop=True)


def write_canonical_dual(df: pd.DataFrame, csv_path: Path, parquet_path: Path) -> pd.DataFrame:
    """Write CSV and Parquet from the same canonical dataframe."""
    canonical = sort_canonical_df(df)
    atomic_write_csv(canonical, csv_path)
    atomic_write_parquet(canonical, parquet_path)
    return canonical


@dataclass
class PairSnapshotSeries:
    ts_ns: np.ndarray
    prices: np.ndarray
    snapshot_ids: np.ndarray


class SnapshotPairCache:
    """Read-only SQLite snapshot loader with bounded pair batches."""

    def __init__(self, db_path: Path, *, batch_size: int = 700) -> None:
        self.db_path = db_path
        self.batch_size = batch_size
        self._cache: dict[str, PairSnapshotSeries] = {}

    def _connect(self) -> sqlite3.Connection:
        uri = f"file:{self.db_path.as_posix()}?mode=ro"
        return sqlite3.connect(uri, uri=True)

    def prefetch_pairs(self, pairs: list[str]) -> None:
        missing = [p for p in pairs if p not in self._cache]
        if not missing:
            return
        for i in range(0, len(missing), self.batch_size):
            chunk = missing[i : i + self.batch_size]
            loaded = self._load_pairs(chunk)
            self._cache.update(loaded)

    def release_pairs(self, pairs: list[str]) -> None:
        for pair in pairs:
            self._cache.pop(pair, None)

    def get(self, pair: str) -> PairSnapshotSeries | None:
        return self._cache.get(pair)

    def _load_pairs(self, pairs: list[str]) -> dict[str, PairSnapshotSeries]:
        if not pairs:
            return {}
        if not self.db_path.exists():
            return {}

        frames: list[pd.DataFrame] = []
        conn = self._connect()
        try:
            for i in range(0, len(pairs), self.batch_size):
                chunk = pairs[i : i + self.batch_size]
                placeholders = ",".join(["?"] * len(chunk))
                query = f"""
                    SELECT id, pair_address, timestamp, price
                    FROM market_snapshots
                    WHERE pair_address IN ({placeholders})
                      AND timestamp IS NOT NULL
                """
                frames.append(pd.read_sql_query(query, conn, params=chunk))
        finally:
            conn.close()

        if not frames:
            return {}

        snap = pd.concat(frames, ignore_index=True)
        if snap.empty:
            return {}

        snap["pair_address"] = snap["pair_address"].astype(str).str.strip()
        snap["ts"] = pd.to_datetime(snap["timestamp"], utc=True, errors="coerce")
        snap["price"] = pd.to_numeric(snap["price"], errors="coerce")

        snap = snap[snap["pair_address"].str.len().gt(0) & snap["ts"].notna()].copy()
        snap = snap.sort_values(["pair_address", "ts"]).reset_index(drop=True)

        groups: dict[str, PairSnapshotSeries] = {}
        for pair, group in snap.groupby("pair_address", sort=False):
            ts_ns = group["ts"].map(lambda x: int(x.value)).to_numpy(dtype=np.int64)
            prices = group["price"].to_numpy(dtype=float)
            ids = group["id"].to_numpy()
            order = np.argsort(ts_ns)
            groups[str(pair)] = PairSnapshotSeries(
                ts_ns=ts_ns[order],
                prices=prices[order],
                snapshot_ids=ids[order],
            )
        return groups


def _event_ns(value: Any) -> int | None:
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except (TypeError, ValueError):
        return None
    if pd.isna(ts):
        return None
    return int(ts.value)


def _resolve_candidate_id(
    row: pd.Series,
    *,
    chain: str,
    source: str,
) -> tuple[str | None, str | None]:
    pair = str(row.get("pair_address", "")).strip()
    if not pair:
        return None, "BAD_PAIR_ADDRESS"
    event_raw = row.get("event_timestamp")
    if event_raw is None or (isinstance(event_raw, float) and math.isnan(event_raw)):
        return None, "BAD_TIMESTAMP"
    try:
        normalized_ts = normalize_event_timestamp(event_raw)
    except (TypeError, ValueError):
        return None, "BAD_TIMESTAMP"
    source_row_id = row.get("source_row_id")
    if source_row_id is not None and str(source_row_id).strip() == "":
        source_row_id = None
    row_chain = row.get("chain")
    effective_chain = str(row_chain).strip() if row_chain is not None and str(row_chain).strip() else chain
    row_source = row.get("source")
    effective_source = str(row_source).strip() if row_source is not None and str(row_source).strip() else source
    cid = compute_candidate_id(
        chain=effective_chain,
        pair_address=pair,
        event_timestamp_normalized=normalized_ts,
        source=effective_source,
        source_row_id=str(source_row_id).strip() if source_row_id is not None else None,
    )
    return cid, None


def _simulation_to_audit_row(
    sim: ExitSimulationResult,
    *,
    base: dict[str, Any],
) -> dict[str, Any]:
    row = dict(base)
    row.update(
        {
            "entry_snapshot_timestamp": sim.entry_snapshot_timestamp,
            "entry_price_raw": sim.entry_price_raw,
            "entry_price": sim.entry_price,
            "entry_snapshot_id": sim.entry_snapshot_id,
            "future_window_start_timestamp": sim.future_window_start_timestamp,
            "future_window_end_timestamp": sim.future_window_end_timestamp,
            "first_future_snapshot_timestamp": sim.first_future_snapshot_timestamp,
            "last_future_snapshot_timestamp": sim.last_future_snapshot_timestamp,
            "future_snapshot_count": sim.future_snapshot_count,
            "max_future_ratio": sim.max_future_ratio,
            "min_future_ratio": sim.min_future_ratio,
            "gap_detected": sim.gap.gap_detected,
            "gap_start_timestamp": sim.gap.gap_start_timestamp,
            "gap_end_timestamp": sim.gap.gap_end_timestamp,
            "gap_minutes": sim.gap.gap_minutes,
            "exit_timestamp": sim.exit_timestamp,
            "sim_exit_status": sim.sim_exit_status,
            "exit_ratio": sim.exit_ratio,
            "sim_net_return": sim.sim_net_return,
            "label_valid": sim.label_valid,
            "label_error_code": sim.label_error_code,
            "label_error_detail": sim.label_error_detail,
            "target_net_profitable_after_exit": sim.target_net_profitable_after_exit,
        }
    )
    return row


def build_audit_row(
    row: pd.Series,
    *,
    filter_name: str,
    horizon: str,
    exit_policy: dict[str, Any],
    target_version: str,
    label_source_artifact_id: str,
    chain: str,
    source: str,
    max_future_gap_minutes: float,
    snapshot_cache: SnapshotPairCache,
) -> dict[str, Any]:
    pair = str(row.get("pair_address", "")).strip()
    event_raw = row.get("event_timestamp")
    exit_policy_id = str(exit_policy["exit_policy_id"])
    tp_ratio = float(exit_policy["tp_ratio"])
    sl_ratio = float(exit_policy["sl_ratio"])
    round_trip_fee_pct = float(exit_policy["round_trip_fee_pct"])
    time_stop_minutes = resolve_time_stop_minutes(horizon, exit_policy)

    candidate_id, id_error = _resolve_candidate_id(row, chain=chain, source=source)
    candidate_policy_id = None
    target_row_id = None
    if candidate_id is not None:
        candidate_policy_id = compute_candidate_policy_id(
            candidate_id=candidate_id,
            filter_name=filter_name,
            horizon=horizon,
            exit_policy_id=exit_policy_id,
            tp_ratio=tp_ratio,
            sl_ratio=sl_ratio,
            time_stop_minutes=time_stop_minutes,
            round_trip_fee_pct=round_trip_fee_pct,
            top_pct=NOT_APPLICABLE,
            pair_cap=NOT_APPLICABLE,
        )
        target_row_id = compute_target_row_id(
            candidate_policy_id=candidate_policy_id,
            target_version=target_version,
            label_source_artifact_id=label_source_artifact_id,
        )

    base: dict[str, Any] = {
        "candidate_id": candidate_id,
        "candidate_policy_id": candidate_policy_id,
        "target_row_id": target_row_id,
        "pair_address": pair,
        "event_timestamp": event_raw,
        "filter": filter_name,
        "horizon": horizon,
        "exit_policy_id": exit_policy_id,
        "tp_ratio": tp_ratio,
        "sl_ratio": sl_ratio,
        "round_trip_fee_pct": round_trip_fee_pct,
        "time_stop_minutes": time_stop_minutes,
        "top_pct": NOT_APPLICABLE,
        "pair_cap": NOT_APPLICABLE,
        "target_name": TARGET_NAME,
        "target_version": target_version,
        "label_source_artifact_id": label_source_artifact_id,
        "max_future_gap_minutes": max_future_gap_minutes,
    }

    for col in row.index:
        if col not in base and col not in OUTCOME_COLUMNS:
            base[col] = row[col]

    if id_error is not None:
        sim = ExitSimulationResult(
            sim_exit_status=id_error,
            label_error_code=id_error,
            label_error_detail=f"identity resolution failed: {id_error}",
        )
        return _simulation_to_audit_row(sim, base=base)

    event_ns = _event_ns(event_raw)
    if event_ns is None:
        sim = ExitSimulationResult(
            sim_exit_status="BAD_TIMESTAMP",
            label_error_code="BAD_TIMESTAMP",
            label_error_detail="event_timestamp could not be parsed",
        )
        return _simulation_to_audit_row(sim, base=base)

    series = snapshot_cache.get(pair)
    if series is None:
        sim = ExitSimulationResult(
            sim_exit_status="NO_PAIR",
            label_error_code="NO_PAIR",
            label_error_detail="pair not found in snapshot cache",
        )
        return _simulation_to_audit_row(sim, base=base)

    sim = simulate_exit_path(
        pair=pair,
        event_ns=event_ns,
        time_stop_minutes=time_stop_minutes,
        tp_ratio=tp_ratio,
        sl_ratio=sl_ratio,
        round_trip_fee_pct=round_trip_fee_pct,
        max_future_gap_minutes=max_future_gap_minutes,
        ts_ns=series.ts_ns,
        prices=series.prices,
        snapshot_ids=series.snapshot_ids,
    )
    return _simulation_to_audit_row(sim, base=base)


def iter_input_chunks(
    input_path: Path,
    chunk_size: int,
    max_rows: int | None = None,
) -> Iterator[pd.DataFrame]:
    total_read = 0
    try:
        import pyarrow.parquet as pq
    except ImportError:
        df = pd.read_parquet(input_path)
        if max_rows is not None:
            df = df.iloc[:max_rows]
        for start in range(0, len(df), chunk_size):
            yield df.iloc[start : start + chunk_size].copy()
        return

    pf = pq.ParquetFile(input_path)
    batch_rows: list[pd.DataFrame] = []
    batch_len = 0
    for batch in pf.iter_batches(batch_size=chunk_size):
        frame = batch.to_pandas()
        if max_rows is not None:
            remaining = max_rows - total_read
            if remaining <= 0:
                return
            if len(frame) > remaining:
                frame = frame.iloc[:remaining].copy()
        batch_rows.append(frame)
        batch_len += len(frame)
        total_read += len(frame)
        if batch_len >= chunk_size:
            yield pd.concat(batch_rows, ignore_index=True)
            batch_rows = []
            batch_len = 0
        if max_rows is not None and total_read >= max_rows:
            if batch_rows:
                yield pd.concat(batch_rows, ignore_index=True)
            return
    if batch_rows:
        yield pd.concat(batch_rows, ignore_index=True)


@dataclass
class BuildCombinationResult:
    filter_name: str
    horizon: str
    exit_policy_id: str
    input_path: str
    output_csv: str
    output_parquet: str
    row_count: int = 0
    valid_count: int = 0
    invalid_count: int = 0
    positive_count: int = 0
    gap_count: int = 0
    schema_hash: str | None = None
    content_hash: str | None = None
    error_code_counts: dict[str, int] = field(default_factory=dict)


def process_combination(
    *,
    filter_name: str,
    horizon: str,
    exit_policy: dict[str, Any],
    input_path: Path,
    output_dataset_dir: Path,
    sqlite_db: Path,
    target_version: str,
    chunk_size: int,
    max_future_gap_minutes: float,
    max_rows: int | None,
    chain: str = DEFAULT_CHAIN,
    source: str = DEFAULT_SOURCE,
    memory_callback: Callable[[], None] | None = None,
) -> tuple[pd.DataFrame, BuildCombinationResult]:
    label_source_id = label_source_artifact_id_for_input(
        input_path.as_posix().replace("\\", "/")
    )
    exit_policy_id = str(exit_policy["exit_policy_id"])
    basename = output_dataset_basename(filter_name, horizon, exit_policy_id, target_version)
    csv_path = output_dataset_dir / f"{basename}.csv"
    parquet_path = output_dataset_dir / f"{basename}.parquet"

    cache = SnapshotPairCache(sqlite_db)
    chunk_frames: list[pd.DataFrame] = []

    for chunk in iter_input_chunks(input_path, chunk_size, max_rows):
        pairs = chunk["pair_address"].astype(str).str.strip().unique().tolist()
        cache.prefetch_pairs(pairs)
        rows = [
            build_audit_row(
                row,
                filter_name=filter_name,
                horizon=horizon,
                exit_policy=exit_policy,
                target_version=target_version,
                label_source_artifact_id=label_source_id,
                chain=chain,
                source=source,
                max_future_gap_minutes=max_future_gap_minutes,
                snapshot_cache=cache,
            )
            for _, row in chunk.iterrows()
        ]
        chunk_frames.append(pd.DataFrame(rows))
        cache.release_pairs(pairs)
        if memory_callback:
            memory_callback()

    if chunk_frames:
        combined = pd.concat(chunk_frames, ignore_index=True)
    else:
        combined = pd.DataFrame()

    canonical = write_canonical_dual(combined, csv_path, parquet_path)

    result = BuildCombinationResult(
        filter_name=filter_name,
        horizon=horizon,
        exit_policy_id=exit_policy_id,
        input_path=str(input_path),
        output_csv=str(csv_path),
        output_parquet=str(parquet_path),
        row_count=len(canonical),
    )
    if not canonical.empty:
        result.valid_count = int(canonical["label_valid"].fillna(False).astype(bool).sum())
        result.invalid_count = result.row_count - result.valid_count
        positive = canonical.loc[canonical["label_valid"].fillna(False), "target_net_profitable_after_exit"]
        result.positive_count = int(positive.fillna(False).astype(bool).sum())
        gap_mask = canonical["label_error_code"] == "GAP_IN_FUTURE_DATA"
        result.gap_count = int(gap_mask.sum())
        codes = canonical["label_error_code"].fillna("").astype(str)
        result.error_code_counts = codes[codes != ""].value_counts().to_dict()
        try:
            result.schema_hash = read_tabular_schema(parquet_path, "parquet")["schema_hash"]
            result.content_hash = compute_content_hash(parquet_path)
        except OSError:
            pass

    return canonical, result


def discover_input_files(input_dir: Path, filters: list[str], horizons: list[str]) -> list[tuple[str, str, Path]]:
    combos: list[tuple[str, str, Path]] = []
    for filter_name in filters:
        for horizon in horizons:
            path = input_dir / input_dataset_filename(filter_name, horizon)
            combos.append((filter_name, horizon, path))
    return combos


def build_summary_rows(results: list[BuildCombinationResult]) -> pd.DataFrame:
    records = []
    for r in results:
        rate = (r.positive_count / r.valid_count) if r.valid_count else 0.0
        records.append(
            {
                "filter": r.filter_name,
                "horizon": r.horizon,
                "exit_policy_id": r.exit_policy_id,
                "input_path": r.input_path,
                "output_parquet": r.output_parquet,
                "row_count": r.row_count,
                "valid_label_count": r.valid_count,
                "invalid_label_count": r.invalid_count,
                "positive_target_count": r.positive_count,
                "positive_target_rate": rate,
                "gap_in_future_data_count": r.gap_count,
                "schema_hash": r.schema_hash,
                "content_hash": r.content_hash,
            }
        )
    return pd.DataFrame(records)


def build_invalid_diagnostic(all_rows: pd.DataFrame) -> pd.DataFrame:
    if all_rows.empty:
        return pd.DataFrame(
            columns=[
                "filter",
                "horizon",
                "exit_policy_id",
                "label_error_code",
                "label_error_detail",
                "count",
            ]
        )
    invalid = all_rows[~all_rows["label_valid"].fillna(False)].copy()
    if invalid.empty:
        return pd.DataFrame(
            columns=[
                "filter",
                "horizon",
                "exit_policy_id",
                "label_error_code",
                "label_error_detail",
                "count",
            ]
        )
    grouped = (
        invalid.groupby(
            ["filter", "horizon", "exit_policy_id", "label_error_code", "label_error_detail"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
    )
    return grouped.sort_values(
        ["filter", "horizon", "exit_policy_id", "count"],
        ascending=[True, True, True, False],
        kind="mergesort",
    )


def build_gap_diagnostic(all_rows: pd.DataFrame) -> pd.DataFrame:
    if all_rows.empty:
        return pd.DataFrame()
    gap = all_rows[all_rows["label_error_code"] == "GAP_IN_FUTURE_DATA"].copy()
    cols = [
        "filter",
        "horizon",
        "exit_policy_id",
        "candidate_id",
        "pair_address",
        "event_timestamp",
        "gap_start_timestamp",
        "gap_end_timestamp",
        "gap_minutes",
        "max_future_gap_minutes",
        "label_error_detail",
    ]
    present = [c for c in cols if c in gap.columns]
    return gap[present].sort_values(
        ["filter", "horizon", "exit_policy_id", "pair_address", "event_timestamp"],
        kind="mergesort",
    )


def validate_sqlite_readonly(db_path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"readable": False, "row_count_sample": None, "error": None}
    if not db_path.exists():
        info["error"] = f"SQLite DB not found: {db_path}"
        return info
    try:
        conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
        try:
            cur = conn.execute("SELECT COUNT(*) FROM market_snapshots")
            info["row_count_sample"] = int(cur.fetchone()[0])
            info["readable"] = True
        finally:
            conn.close()
    except sqlite3.Error as exc:
        info["error"] = str(exc)
    return info


def register_e3_artifacts(
    project_root: Path,
    output_dataset_dir: Path,
    output_report_dir: Path,
) -> dict[str, Any]:
    """Register E3 outputs via Phase E1 registry; report failures clearly."""
    status: dict[str, Any] = {
        "attempted": True,
        "success": False,
        "error": None,
        "artifacts_registered": 0,
        "validation_report_path": None,
    }
    try:
        from app.artifacts.registry import (
            detect_project_root,
            get_git_commit_hash,
            load_registry,
            scan_artifacts,
            validate_registry,
            write_registry_jsonl,
            write_validation_report,
        )
        from app.observability.audit_io import utc_timestamp_slug

        root = detect_project_root(project_root)
        rel_dataset = output_dataset_dir.relative_to(root).as_posix()
        rel_report = output_report_dir.relative_to(root).as_posix()
        scan_roots = [rel_dataset, rel_report]
        registry_path = root / "data" / "training" / "artifact_registry" / "artifact_registry.jsonl"
        git_commit_hash, git_warnings = get_git_commit_hash(root)
        previous = load_registry(registry_path)
        records, scan_warnings = scan_artifacts(
            project_root=root,
            scan_roots=scan_roots,
            branch_name="phase_e3_direct_exit_target_builder",
            generated_by_script="scripts/build_direct_exit_targets.py",
            previous_registry=previous,
            git_commit_hash=git_commit_hash,
            git_warnings=git_warnings,
        )
        merged = {r.artifact_id: r for r in previous.values()}
        for record in records:
            merged[record.artifact_id] = record
        write_registry_jsonl(list(merged.values()), registry_path)
        report = validate_registry(registry_path, root, verbose=False)
        validation_path = root / "data" / "audits" / f"phase_e3_artifact_registration_validation_{utc_timestamp_slug()}.json"
        write_validation_report(report, validation_path)
        status["success"] = True
        status["artifacts_registered"] = len(records)
        status["validation_report_path"] = str(validation_path)
        status["scan_warnings"] = scan_warnings
        if git_warnings:
            status["git_warnings"] = git_warnings
    except Exception as exc:
        status["error"] = str(exc)
    return status
