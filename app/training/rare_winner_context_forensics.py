"""Phase E8E — unified rare-winner context forensics audit (offline, read-only, resumable)."""

from __future__ import annotations

import hashlib
import json
import os
import random
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from app.database import DB_PATH
from app.training.clean_rf_policy_tail_audit import (
    PAIR_COL,
    RETURN_COL,
    SCORE_COL,
    TARGET_COL,
    load_predictions,
    rank_frame,
    select_top_rows,
    selected_count_from_rows,
)
from app.training.direct_target_xgb_rf import (
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    utc_now_iso,
)

PHASE = "E8E"
SCRIPT_PATH = "scripts/run_rare_winner_context_forensics.py"
DEFAULT_OUTPUT_ROOT = "data/training/manual_verified_results"

WINDOWS: tuple[str, ...] = ("5m", "15m", "30m", "60m", "4h", "8h", "24h")
WINDOW_MINUTES: dict[str, int] = {
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "60m": 60,
    "4h": 240,
    "8h": 480,
    "24h": 1440,
}

LEAKAGE_TOKENS: tuple[str, ...] = (
    "post_",
    "after",
    "future",
    "target_",
    "label_",
    "sim_",
    "exit_",
    "gap_",
    "max_future",
    "min_future",
    "realized",
    "outcome",
)

CANDIDATE_GROUPS = (
    "rare_winner_selected_positives",
    "rare_winner_selected_losers",
    "high_score_nonselected_near_tail",
    "matched_random_controls",
    "reservoir_overlap",
)

FINAL_CLASSIFICATIONS = (
    "CONTEXT_PATTERN_FOUND_FOR_E9",
    "RARE_WINNER_RESEARCH_ONLY",
    "PAIR_IDENTITY_ARTIFACT",
    "NO_CONTEXT_SIGNAL",
    "INSUFFICIENT_CONTEXT_DATA",
)

MARKET_SNAPSHOT_COLS = (
    "id",
    "pair_address",
    "timestamp",
    "price",
    "liquidity",
    "volume_24h",
    "fdv",
    "txns_buys",
    "txns_sells",
    "txns_total",
    "price_change_m5",
    "price_change_h1",
    "price_change_h6",
    "price_change_h24",
    "whale_score",
    "buy_ratio",
)

STAGES = (
    "discover_inputs",
    "sqlite_inventory",
    "joinability_audit",
    "reservoir_discovery",
    "build_candidate_groups",
    "market_context",
    "liquidity_dynamics",
    "whale_wallet_context",
    "raw_payload_inventory",
    "reservoir_crosscheck",
    "matched_control_comparison",
    "forensic_timeline",
    "pattern_discovery",
    "feature_candidate_map",
    "final_classification",
    "finalize_outputs",
)


def normalize_id(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def normalize_pair(value: Any) -> str:
    return normalize_id(value).lower()


def parse_utc_timestamp(value: Any) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return ts


def identity_key(row: dict[str, Any]) -> str:
    for col in ("target_row_id", "candidate_policy_id"):
        val = normalize_id(row.get(col))
        if val:
            return f"{col}:{val}"
    cid = normalize_id(row.get("candidate_id"))
    if cid:
        return (
            f"candidate:{cid}|{normalize_id(row.get('horizon'))}|"
            f"{normalize_id(row.get('exit_policy_id'))}"
        )
    pair = normalize_pair(row.get("pair_address"))
    event = row.get("event_timestamp")
    if pair and event is not None:
        return f"pair_ts:{pair}|{parse_utc_timestamp(event)}"
    return f"row:{row.get('_row_order', '')}|{row.get('split', '')}|{row.get('dataset_name', '')}"


def is_leaky_feature_name(feature_name: str) -> tuple[bool, str | None]:
    lower = feature_name.lower()
    for token in LEAKAGE_TOKENS:
        if token in lower:
            return True, f"forbidden_token:{token}"
    return False, None


def recommended_status_for_feature(
    *,
    feature_name: str,
    pre_entry_legal: bool,
    missingness_rate: float,
    pair_identity_risk: str,
) -> str:
    leaky, reason = is_leaky_feature_name(feature_name)
    if leaky:
        return "REJECT_LEAKAGE"
    if not pre_entry_legal:
        return "RESEARCH_ONLY"
    if missingness_rate >= 0.95:
        return "REJECT_NOT_AVAILABLE"
    if pair_identity_risk == "high":
        return "REJECT_PAIR_IDENTITY_ARTIFACT"
    return "KEEP_FOR_E9"


@dataclass
class ForensicsConfig:
    e8b_run_dir: Path
    e8c_dir: Path
    output_dir: Path
    sqlite_db: Path | None = None
    smoke: bool = False
    full: bool = False
    force: bool = False
    max_candidates: int | None = None
    max_controls_per_candidate: int = 20
    windows: tuple[str, ...] = WINDOWS
    random_state: int = 42


@dataclass
class ForensicsState:
    rare_winner_datasets: list[dict[str, Any]] = field(default_factory=list)
    prediction_files: list[str] = field(default_factory=list)
    reservoir_files: list[str] = field(default_factory=list)
    candidate_rows: list[dict[str, Any]] = field(default_factory=list)
    candidate_identity_map: list[dict[str, Any]] = field(default_factory=list)
    candidate_group_summary: list[dict[str, Any]] = field(default_factory=list)
    joinability_audit: list[dict[str, Any]] = field(default_factory=list)
    sqlite_inventory: list[dict[str, Any]] = field(default_factory=list)
    market_context: list[dict[str, Any]] = field(default_factory=list)
    liquidity_dynamics: list[dict[str, Any]] = field(default_factory=list)
    whale_wallet_context: list[dict[str, Any]] = field(default_factory=list)
    raw_payload_inventory: list[dict[str, Any]] = field(default_factory=list)
    raw_payload_extract: list[dict[str, Any]] = field(default_factory=list)
    reservoir_overlap: list[dict[str, Any]] = field(default_factory=list)
    reservoir_pattern: list[dict[str, Any]] = field(default_factory=list)
    matched_control_comparison: list[dict[str, Any]] = field(default_factory=list)
    forensic_timeline: list[dict[str, Any]] = field(default_factory=list)
    pattern_candidates: list[dict[str, Any]] = field(default_factory=list)
    feature_candidate_map: list[dict[str, Any]] = field(default_factory=list)
    final_classification: dict[str, Any] = field(default_factory=dict)
    context_availability: dict[str, bool] = field(default_factory=dict)
    fatal_blockers: list[str] = field(default_factory=list)
    completed_stages: list[str] = field(default_factory=list)
    failed_stages: list[str] = field(default_factory=list)


class E8EAuditLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, **fields: Any) -> None:
        payload = {"created_at_utc": utc_now_iso(), "event_type": event_type, "phase": PHASE, **fields}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def error(self, message: str, **fields: Any) -> None:
        self.log("error", message=message, **fields)


class CheckpointManager:
    def __init__(self, audit_dir: Path, *, force: bool = False) -> None:
        self.audit_dir = audit_dir
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.force = force
        self.stage_path = audit_dir / "e8e_progress_checkpoints.jsonl"
        self.candidate_path = audit_dir / "e8e_candidate_checkpoints.jsonl"
        self.completed_stages: set[str] = set()
        self.completed_candidates: set[str] = set()
        if not force:
            self._load()

    def _load(self) -> None:
        if self.stage_path.exists():
            for line in self.stage_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") == "completed":
                    self.completed_stages.add(str(row.get("stage_name")))
        if self.candidate_path.exists():
            for line in self.candidate_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("status") == "completed":
                    self.completed_candidates.add(str(row.get("identity_key")))

    def stage_complete(self, stage_name: str) -> bool:
        return not self.force and stage_name in self.completed_stages

    def candidate_complete(self, key: str) -> bool:
        return not self.force and key in self.completed_candidates

    def mark_stage(self, stage_name: str, *, status: str, **fields: Any) -> None:
        payload = {
            "stage_name": stage_name,
            "status": status,
            "started_at": fields.pop("started_at", utc_now_iso()),
            "completed_at": utc_now_iso(),
            **fields,
        }
        with self.stage_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def mark_candidate(self, key: str, **fields: Any) -> None:
        payload = {"identity_key": key, "status": "completed", "completed_at": utc_now_iso(), **fields}
        with self.candidate_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


def make_output_dir(output_root: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = output_root / f"phase_e8e_rare_winner_context_forensics_{ts}"
    (out / "reports").mkdir(parents=True, exist_ok=True)
    (out / "audit").mkdir(parents=True, exist_ok=True)
    return out


def discover_reservoir_files(project_root: Path) -> list[Path]:
    patterns = (
        "E7E_168h_cap250_confirmed_then_activity.csv",
        "E7E_72h_cap250_confirmed_then_activity.csv",
        "*168h*cap250*confirmed_then_activity*.csv",
        "*72h*cap250*confirmed_then_activity*.csv",
        "e8_*reservoir*.csv",
    )
    found: dict[str, Path] = {}
    for pattern in patterns:
        for path in project_root.rglob(pattern):
            if path.is_file():
                found[str(path.resolve())] = path.resolve()
    return sorted(found.values(), key=lambda p: str(p))


def discover_prediction_file(run_dir: Path, dataset_name: str, split: str) -> Path | None:
    pred_dir = run_dir / "predictions"
    for suffix in (".parquet", ".csv"):
        path = pred_dir / f"{dataset_name}_{split}_predictions{suffix}"
        if path.exists():
            return path
    return None


def load_e8c_rare_winner_datasets(e8c_dir: Path) -> list[dict[str, Any]]:
    path = e8c_dir / "reports" / "e8c_final_classification.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing E8C classification: {path}")
    df = pd.read_csv(path)
    rare = df[df["final_classification"] == "RARE_WINNER_DETECTOR"].copy()
    policies = pd.read_csv(e8c_dir / "reports" / "e8c_validation_selected_policies.csv")
    rows: list[dict[str, Any]] = []
    for _, row in rare.iterrows():
        ds = str(row["dataset_name"])
        policy = policies[policies["dataset_name"] == ds]
        top_pct = float(policy.iloc[0]["top_pct_percent"]) if not policy.empty else None
        rows.append(
            {
                "dataset_name": ds,
                "filter": row.get("filter"),
                "horizon": row.get("horizon"),
                "exit_policy_id": row.get("exit_policy_id"),
                "final_classification": row.get("final_classification"),
                "validation_selected_top_pct_percent": top_pct,
            }
        )
    return rows


def enrich_row(row: pd.Series, *, dataset_meta: dict[str, Any], group: str, split: str) -> dict[str, Any]:
    out = row.to_dict()
    out["group"] = group
    out["split"] = split
    out["dataset_name"] = dataset_meta["dataset_name"]
    out["horizon"] = dataset_meta.get("horizon")
    out["exit_policy_id"] = dataset_meta.get("exit_policy_id")
    out["filter"] = dataset_meta.get("filter")
    out["identity_key"] = identity_key(out)
    if "event_timestamp" in out:
        out["event_timestamp"] = parse_utc_timestamp(out["event_timestamp"])
        out["timestamp_source"] = "event_timestamp"
    out["pair_address_norm"] = normalize_pair(out.get("pair_address"))
    return out


def build_candidate_groups_for_dataset(
    dataset_meta: dict[str, Any],
    *,
    run_dir: Path,
    max_controls_per_candidate: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    top_pct = float(dataset_meta["validation_selected_top_pct_percent"])
    rows: list[dict[str, Any]] = []
    for split in ("validation", "test"):
        path = discover_prediction_file(run_dir, dataset_meta["dataset_name"], split)
        if path is None:
            continue
        df = load_predictions(path)
        selected, k, _ = select_top_rows(df, top_pct)
        ranked = rank_frame(df)
        near_tail = ranked.iloc[k : k + max(1, min(50, k))].copy()

        pos = selected[selected[TARGET_COL] == 1]
        neg = selected[selected[TARGET_COL] == 0]
        for _, r in pos.iterrows():
            rows.append(enrich_row(r, dataset_meta=dataset_meta, group="rare_winner_selected_positives", split=split))
        for _, r in neg.iterrows():
            rows.append(enrich_row(r, dataset_meta=dataset_meta, group="rare_winner_selected_losers", split=split))
        for _, r in near_tail.iterrows():
            rows.append(
                enrich_row(r, dataset_meta=dataset_meta, group="high_score_nonselected_near_tail", split=split)
            )

        pool = df.copy()
        selected_keys = set(selected.index.tolist())
        pool = pool[~pool.index.isin(selected_keys)]
        if not pool.empty:
            n_controls = min(max_controls_per_candidate, len(pool))
            sample_idx = rng.sample(pool.index.tolist(), n_controls)
            for idx in sample_idx:
                rows.append(
                    enrich_row(pool.loc[idx], dataset_meta=dataset_meta, group="matched_random_controls", split=split)
                )
    return rows


def sqlite_connect_ro(db_path: Path) -> sqlite3.Connection:
    uri = f"file:{db_path.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def inventory_sqlite_tables(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    conn = sqlite_connect_ro(db_path)
    try:
        tables = pd.read_sql_query(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", conn
        )["name"].tolist()
        for table in tables:
            cols_df = pd.read_sql_query(f"PRAGMA table_info({table})", conn)
            columns = cols_df["name"].tolist()
            row_count = None
            try:
                row_count = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
            except sqlite3.Error:
                row_count = None
            ts_cols = [c for c in columns if "timestamp" in c.lower() or c.lower().endswith("_at")]
            pair_cols = [c for c in columns if "pair" in c.lower() or "address" in c.lower()]
            id_cols = [c for c in columns if "candidate" in c.lower() or "target_row" in c.lower()]
            usable_asof = table == "market_snapshots" or "timestamp" in columns
            usable_pair_time = bool(pair_cols and ts_cols)
            notes = ""
            if table == "raw_provider_payloads":
                notes = "bounded payload key inventory only"
            rows.append(
                {
                    "table_name": table,
                    "row_count": row_count,
                    "columns": "|".join(columns),
                    "timestamp_columns": "|".join(ts_cols),
                    "pair_address_columns": "|".join(pair_cols),
                    "candidate_target_id_columns": "|".join(id_cols),
                    "usable_for_asof_join": usable_asof,
                    "usable_for_pair_time_join": usable_pair_time,
                    "notes": notes,
                }
            )
    finally:
        conn.close()
    return rows


def load_pair_snapshots(db_path: Path, pairs: list[str], *, batch_size: int = 200) -> pd.DataFrame:
    if not db_path.exists() or not pairs:
        return pd.DataFrame(columns=list(MARKET_SNAPSHOT_COLS))
    conn = sqlite_connect_ro(db_path)
    try:
        info = pd.read_sql_query("PRAGMA table_info(market_snapshots)", conn)
        available = set(info["name"].tolist()) if not info.empty else set()
        if "pair_address" not in available:
            return pd.DataFrame(columns=list(MARKET_SNAPSHOT_COLS))
        select_cols = [c for c in MARKET_SNAPSHOT_COLS if c in available]
        cols = ", ".join(select_cols)
        frames: list[pd.DataFrame] = []
        for i in range(0, len(pairs), batch_size):
            chunk = [normalize_pair(p) for p in pairs[i : i + batch_size]]
            placeholders = ",".join(["?"] * len(chunk))
            q = f"SELECT {cols} FROM market_snapshots WHERE LOWER(pair_address) IN ({placeholders})"
            frames.append(pd.read_sql_query(q, conn, params=chunk))
    finally:
        conn.close()
    if not frames:
        return pd.DataFrame(columns=list(MARKET_SNAPSHOT_COLS))
    out = pd.concat(frames, ignore_index=True)
    for col in MARKET_SNAPSHOT_COLS:
        if col not in out.columns:
            out[col] = np.nan
    out["pair_address"] = out["pair_address"].map(normalize_pair)
    out["ts"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    return out.dropna(subset=["ts"]).sort_values(["pair_address", "ts"])


def asof_snapshot(snapshots: pd.DataFrame, pair: str, event_ts: pd.Timestamp) -> pd.Series | None:
    sub = snapshots[(snapshots["pair_address"] == normalize_pair(pair)) & (snapshots["ts"] <= event_ts)]
    if sub.empty:
        return None
    return sub.iloc[-1]


def snapshot_at_or_before(
    snapshots: pd.DataFrame, pair: str, cutoff_ts: pd.Timestamp
) -> pd.Series | None:
    return asof_snapshot(snapshots, pair, cutoff_ts)


def run_joinability_audit(
    sample_rows: list[dict[str, Any]],
    *,
    db_path: Path | None,
    reservoir_files: list[Path],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    sample_size = len(sample_rows)
    pairs = sorted({normalize_pair(r.get("pair_address")) for r in sample_rows if normalize_pair(r.get("pair_address"))})

    sqlite_match = 0
    sqlite_possible = False
    failure_reason = ""
    if db_path and db_path.exists() and pairs:
        snaps = load_pair_snapshots(db_path, pairs[: min(100, len(pairs))])
        pair_set = set(snaps["pair_address"].unique()) if not snaps.empty else set()
        for row in sample_rows[:100]:
            if normalize_pair(row.get("pair_address")) in pair_set:
                sqlite_match += 1
        sqlite_possible = sqlite_match > 0
        if sqlite_match == 0:
            failure_reason = "zero market_snapshots pair matches in sample"
    elif db_path is None or not db_path.exists():
        failure_reason = "sqlite unavailable"
    results.append(
        {
            "join_strategy": "pair_address_to_market_snapshots",
            "sample_size": min(100, sample_size),
            "key_columns": "pair_address",
            "normalized_key_nonnull_rate": float(
                np.mean([bool(normalize_pair(r.get("pair_address"))) for r in sample_rows[:100]])
            )
            if sample_rows
            else 0.0,
            "sqlite_match_count": sqlite_match,
            "sqlite_match_rate": sqlite_match / min(100, sample_size) if sample_size else 0.0,
            "prediction_to_sqlite_join_possible": sqlite_possible,
            "reservoir_match_count": 0,
            "reservoir_match_rate": 0.0,
            "failure_reason": failure_reason,
        }
    )

    reservoir_pairs: set[str] = set()
    for path in reservoir_files:
        try:
            cols = pd.read_csv(path, nrows=0).columns.tolist()
            if "pair_address" not in cols:
                continue
            sub = pd.read_csv(path, usecols=["pair_address"])
            reservoir_pairs.update(sub["pair_address"].astype(str).str.strip().str.lower())
        except Exception:
            continue
    res_match = sum(1 for r in sample_rows[:100] if normalize_pair(r.get("pair_address")) in reservoir_pairs)
    results.append(
        {
            "join_strategy": "pair_address_to_reservoir",
            "sample_size": min(100, sample_size),
            "key_columns": "pair_address",
            "normalized_key_nonnull_rate": results[0]["normalized_key_nonnull_rate"],
            "sqlite_match_count": None,
            "sqlite_match_rate": None,
            "prediction_to_sqlite_join_possible": sqlite_possible,
            "reservoir_match_count": res_match,
            "reservoir_match_rate": res_match / min(100, sample_size) if sample_size else 0.0,
            "failure_reason": "" if reservoir_pairs else "no reservoir pair index built",
        }
    )
    return results


def compute_market_context_row(
    row: dict[str, Any],
    snapshots: pd.DataFrame,
    *,
    windows: tuple[str, ...],
) -> dict[str, Any]:
    event_ts = row.get("event_timestamp")
    if not isinstance(event_ts, pd.Timestamp):
        event_ts = parse_utc_timestamp(event_ts)
    pair = normalize_pair(row.get("pair_address"))
    out: dict[str, Any] = {
        "identity_key": row.get("identity_key"),
        "group": row.get("group"),
        "dataset_name": row.get("dataset_name"),
        "split": row.get("split"),
        "pair_address": row.get("pair_address"),
        "event_timestamp": event_ts.isoformat() if event_ts is not None else None,
        "timestamp_source": row.get("timestamp_source"),
        TARGET_COL: row.get(TARGET_COL),
        RETURN_COL: row.get("sim_net_return"),
        "market_context_available": False,
    }
    if event_ts is None or not pair or snapshots.empty:
        return out

    asof = asof_snapshot(snapshots, pair, event_ts)
    if asof is None:
        return out
    out["market_context_available"] = True
    numeric_cols = (
        "price",
        "liquidity",
        "volume_24h",
        "fdv",
        "txns_buys",
        "txns_sells",
        "txns_total",
        "price_change_m5",
        "price_change_h1",
        "price_change_h6",
        "price_change_h24",
        "whale_score",
        "buy_ratio",
    )
    for col in numeric_cols:
        val = asof.get(col)
        out[f"{col}_asof"] = float(val) if pd.notna(val) else None
    liq = out.get("liquidity_asof")
    vol = out.get("volume_24h_asof")
    out["volume_to_liquidity_ratio_asof"] = float(vol / liq) if liq and vol is not None and liq > 0 else None

    for window in windows:
        minutes = WINDOW_MINUTES[window]
        pre_cutoff = event_ts - timedelta(minutes=minutes)
        pre_snap = snapshot_at_or_before(snapshots, pair, pre_cutoff)
        post_cutoff = event_ts + timedelta(minutes=minutes)
        post_snaps = snapshots[
            (snapshots["pair_address"] == pair)
            & (snapshots["ts"] > event_ts)
            & (snapshots["ts"] <= post_cutoff)
        ]
        if pre_snap is not None:
            for col in ("liquidity", "volume_24h", "price", "txns_total", "whale_score", "buy_ratio"):
                a = asof.get(col)
                b = pre_snap.get(col)
                if pd.notna(a) and pd.notna(b):
                    out[f"{col}_delta_{window}_pre"] = float(a) - float(b)
        if not post_snaps.empty:
            post_last = post_snaps.iloc[-1]
            for col in ("liquidity", "volume_24h", "price", "txns_total"):
                a = asof.get(col)
                b = post_last.get(col)
                if pd.notna(a) and pd.notna(b):
                    out[f"{col}_delta_{window}_post_diag"] = float(b) - float(a)
    return out


def compute_liquidity_dynamics(market_row: dict[str, Any]) -> dict[str, Any]:
    return {
        "identity_key": market_row.get("identity_key"),
        "group": market_row.get("group"),
        "dataset_name": market_row.get("dataset_name"),
        "liquidity_asof": market_row.get("liquidity_asof"),
        "liquidity_known_flag": market_row.get("liquidity_asof") is not None,
        "liquidity_missing_flag": market_row.get("liquidity_asof") is None,
        "liquidity_recovery_pre_entry": market_row.get("liquidity_delta_60m_pre"),
        "liquidity_collapse_post_entry": market_row.get("liquidity_delta_60m_post_diag"),
        "volume_to_liquidity_ratio_asof": market_row.get("volume_to_liquidity_ratio_asof"),
        "volume_spike_ratio": (
            (market_row.get("volume_24h_delta_15m_pre") / market_row.get("volume_24h_asof"))
            if market_row.get("volume_24h_asof") not in (None, 0)
            and market_row.get("volume_24h_delta_15m_pre") is not None
            else None
        ),
    }


def inventory_whale_wallet(db_path: Path, pairs: list[str]) -> tuple[list[dict[str, Any]], dict[str, bool]]:
    rows: list[dict[str, Any]] = []
    availability = {
        "whale_context_available": False,
        "wallet_context_available": False,
        "signal_context_available": False,
    }
    if not db_path.exists():
        return rows, availability
    conn = sqlite_connect_ro(db_path)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "whale_alerts" in tables and pairs:
            availability["whale_context_available"] = True
            norm_pairs = [normalize_pair(p) for p in pairs[:100]]
            placeholders = ",".join(["?"] * len(norm_pairs))
            q = f"""
                SELECT pair_address, timestamp, alert_type, whale_score, liquidity, volume
                FROM whale_alerts
                WHERE LOWER(pair_address) IN ({placeholders})
            """
            wa = pd.read_sql_query(q, conn, params=norm_pairs)
            for pair, grp in wa.groupby("pair_address"):
                rows.append(
                    {
                        "pair_address": pair,
                        "whale_event_count_24h": int(len(grp)),
                        "whale_buy_count": int((grp["alert_type"].astype(str).str.lower() == "buy").sum()),
                        "whale_sell_count": int((grp["alert_type"].astype(str).str.lower() == "sell").sum()),
                        "whale_context_available": True,
                    }
                )
        if "signals" in tables:
            availability["signal_context_available"] = True
    finally:
        conn.close()
    return rows, availability


def inventory_raw_payload_keys(db_path: Path, pairs: list[str], *, sample_limit: int = 50) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not db_path.exists() or not pairs:
        return rows
    conn = sqlite_connect_ro(db_path)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "raw_provider_payloads" not in tables:
            return rows
        norm_pairs = [normalize_pair(p) for p in pairs[:20]]
        placeholders = ",".join(["?"] * len(norm_pairs))
        q = f"""
            SELECT pair_address, payload_json_or_text
            FROM raw_provider_payloads
            WHERE LOWER(pair_address) IN ({placeholders})
            LIMIT {sample_limit}
        """
        df = pd.read_sql_query(q, conn, params=norm_pairs)
        key_counts: dict[str, int] = {}
        for text in df["payload_json_or_text"].dropna().astype(str).head(sample_limit):
            try:
                payload = json.loads(text)
                if isinstance(payload, dict):
                    for key in payload.keys():
                        key_counts[str(key)] = key_counts.get(str(key), 0) + 1
            except json.JSONDecodeError:
                continue
        for key, count in sorted(key_counts.items(), key=lambda x: (-x[1], x[0]))[:100]:
            rows.append({"payload_key": key, "sample_count": count})
    finally:
        conn.close()
    return rows


def reservoir_crosscheck(
    candidate_rows: list[dict[str, Any]],
    reservoir_files: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overlap_rows: list[dict[str, Any]] = []
    pattern_rows: list[dict[str, Any]] = []
    reservoir_index: dict[str, list[dict[str, Any]]] = {}
    for path in reservoir_files:
        try:
            df = pd.read_csv(path)
            if "pair_address" not in df.columns:
                continue
            for _, r in df.iterrows():
                pair = normalize_pair(r["pair_address"])
                reservoir_index.setdefault(pair, []).append({**r.to_dict(), "reservoir_file": str(path)})
        except Exception:
            continue
    seen_pairs: set[str] = set()
    for row in candidate_rows:
        pair = normalize_pair(row.get("pair_address"))
        if not pair:
            continue
        hits = reservoir_index.get(pair, [])
        overlap_rows.append(
            {
                "identity_key": row.get("identity_key"),
                "group": row.get("group"),
                "pair_address": row.get("pair_address"),
                "reservoir_hit_count": len(hits),
                "reservoir_files": "|".join(sorted({str(h.get("reservoir_file", "")) for h in hits}))[:500],
                "in_e7_reservoir": bool(hits),
            }
        )
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            pattern_rows.append(
                {
                    "pair_address": row.get("pair_address"),
                    "rare_winner_groups_present": "|".join(
                        sorted(
                            {
                                str(r.get("group"))
                                for r in candidate_rows
                                if normalize_pair(r.get("pair_address")) == pair
                            }
                        )
                    ),
                    "reservoir_hit_count": len(hits),
                    "pattern_supported_by_reservoir": bool(hits),
                }
            )
    return overlap_rows, pattern_rows


def matched_control_comparison(candidate_rows: list[dict[str, Any]], market_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    market_by_key = {r["identity_key"]: r for r in market_rows if r.get("identity_key")}
    out: list[dict[str, Any]] = []
    for group in CANDIDATE_GROUPS:
        keys = [r["identity_key"] for r in candidate_rows if r.get("group") == group]
        mrows = [market_by_key[k] for k in keys if k in market_by_key]
        if not mrows:
            out.append({"group": group, "count": 0})
            continue
        liqs = [r.get("liquidity_asof") for r in mrows if r.get("liquidity_asof") is not None]
        vol_ratios = [
            r.get("volume_to_liquidity_ratio_asof")
            for r in mrows
            if r.get("volume_to_liquidity_ratio_asof") is not None
        ]
        out.append(
            {
                "group": group,
                "count": len(mrows),
                "unique_pairs": len({normalize_pair(r.get("pair_address")) for r in mrows}),
                "positive_count": int(sum(1 for r in mrows if r.get(TARGET_COL) == 1)),
                "mean_liquidity_asof": float(np.mean(liqs)) if liqs else None,
                "median_liquidity_asof": float(np.median(liqs)) if liqs else None,
                "mean_volume_to_liquidity_ratio_asof": float(np.mean(vol_ratios)) if vol_ratios else None,
                "median_volume_to_liquidity_ratio_asof": float(np.median(vol_ratios)) if vol_ratios else None,
                "top_pair_share": None,
            }
        )
    return out


def build_forensic_timeline(
    row: dict[str, Any],
    snapshots: pd.DataFrame,
    *,
    windows: tuple[str, ...],
) -> list[dict[str, Any]]:
    event_ts = row.get("event_timestamp")
    if not isinstance(event_ts, pd.Timestamp):
        event_ts = parse_utc_timestamp(event_ts)
    pair = normalize_pair(row.get("pair_address"))
    if event_ts is None or not pair or snapshots.empty:
        return []
    sub = snapshots[snapshots["pair_address"] == pair].copy()
    rows: list[dict[str, Any]] = []
    for window in windows:
        minutes = WINDOW_MINUTES[window]
        pre_ts = event_ts - timedelta(minutes=minutes)
        pre = snapshot_at_or_before(sub, pair, pre_ts)
        if pre is not None:
            rows.append(
                {
                    "identity_key": row.get("identity_key"),
                    "group": row.get("group"),
                    "dataset_name": row.get("dataset_name"),
                    "split": row.get("split"),
                    "event_timestamp": event_ts.isoformat(),
                    "relative_window": f"-{window}",
                    "pre_or_post": "pre",
                    "is_legal_feature_window": True,
                    "price": pre.get("price"),
                    "liquidity": pre.get("liquidity"),
                    "volume_24h": pre.get("volume_24h"),
                    "txns_total": pre.get("txns_total"),
                    "buy_ratio": pre.get("buy_ratio"),
                    "whale_score": pre.get("whale_score"),
                }
            )
        post_ts = event_ts + timedelta(minutes=minutes)
        post = sub[(sub["ts"] > event_ts) & (sub["ts"] <= post_ts)]
        if not post.empty:
            p = post.iloc[-1]
            rows.append(
                {
                    "identity_key": row.get("identity_key"),
                    "group": row.get("group"),
                    "dataset_name": row.get("dataset_name"),
                    "split": row.get("split"),
                    "event_timestamp": event_ts.isoformat(),
                    "relative_window": f"+{window}",
                    "pre_or_post": "post",
                    "is_legal_feature_window": False,
                    "price": p.get("price"),
                    "liquidity": p.get("liquidity"),
                    "volume_24h": p.get("volume_24h"),
                    "txns_total": p.get("txns_total"),
                    "buy_ratio": p.get("buy_ratio"),
                    "whale_score": p.get("whale_score"),
                }
            )
    return rows


def discover_pattern_candidates(
    comparison: list[dict[str, Any]],
    market_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    patterns: list[dict[str, Any]] = []
    winners = next((r for r in comparison if r.get("group") == "rare_winner_selected_positives"), {})
    losers = next((r for r in comparison if r.get("group") == "rare_winner_selected_losers"), {})
    controls = next((r for r in comparison if r.get("group") == "matched_random_controls"), {})
    if not winners or not losers:
        return patterns
    for metric in ("mean_liquidity_asof", "mean_volume_to_liquidity_ratio_asof"):
        w = winners.get(metric)
        l = losers.get(metric)
        c = controls.get(metric)
        if w is None or l is None:
            continue
        direction = "higher_in_winners" if (w or 0) > (l or 0) else "lower_in_winners"
        patterns.append(
            {
                "pattern_name": metric,
                "winner_value": w,
                "loser_value": l,
                "control_value": c,
                "effect_direction": direction,
                "pre_entry_legal": True,
                "pair_identity_risk": "medium",
            }
        )
    unique_pairs = len({normalize_pair(r.get("pair_address")) for r in market_rows})
    if unique_pairs <= 1:
        patterns.append(
            {
                "pattern_name": "single_pair_concentration",
                "winner_value": unique_pairs,
                "loser_value": None,
                "control_value": None,
                "effect_direction": "pair_identity",
                "pre_entry_legal": True,
                "pair_identity_risk": "high",
            }
        )
    return patterns


def build_feature_candidate_map(
    market_rows: list[dict[str, Any]],
    *,
    sqlite_available: bool,
) -> list[dict[str, Any]]:
    if not market_rows:
        return []
    sample = market_rows[0]
    features: list[dict[str, Any]] = []
    for key, val in sample.items():
        if not key.endswith("_asof") and "_delta_" not in key and key not in (
            "price_asof",
            "liquidity_asof",
            "volume_24h_asof",
            "volume_to_liquidity_ratio_asof",
        ):
            continue
        if key.endswith("_post_diag"):
            pre_entry = False
        elif "_delta_" in key and key.endswith("_pre"):
            pre_entry = True
        elif key.endswith("_asof") or key == "volume_to_liquidity_ratio_asof":
            pre_entry = True
        else:
            continue
        missing = float(np.mean([r.get(key) is None for r in market_rows]))
        leaky, leakage_reason = is_leaky_feature_name(key)
        pair_risk = "high" if "pair" in key else "low"
        status = recommended_status_for_feature(
            feature_name=key,
            pre_entry_legal=pre_entry and not leaky,
            missingness_rate=missing,
            pair_identity_risk=pair_risk,
        )
        features.append(
            {
                "feature_name": key,
                "source_table_or_file": "market_snapshots" if sqlite_available else "predictions_only",
                "computation_window": "asof_or_pre_entry_delta",
                "pre_entry_legal": pre_entry and not leaky,
                "is_leaky": leaky,
                "leakage_reason": leakage_reason,
                "requires_sqlite_table": "market_snapshots",
                "requires_external_api": False,
                "missingness_rate": missing,
                "available_for_runtime_estimate": sqlite_available and missing < 0.5,
                "distinguishes_winners_from_controls": None,
                "pair_identity_risk": pair_risk,
                "leakage_risk": "high" if leaky else "low",
                "recommended_status": status,
            }
        )
    return features


def classify_e8e_final(
    *,
    patterns: list[dict[str, Any]],
    feature_map: list[dict[str, Any]],
    market_available: bool,
    joinability: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    keep = [f for f in feature_map if f.get("recommended_status") == "KEEP_FOR_E9"]
    unique_pairs = len({normalize_pair(r.get("pair_address")) for r in candidate_rows})
    sqlite_join = next((j for j in joinability if j.get("join_strategy") == "pair_address_to_market_snapshots"), {})
    reservoir_supported = any(p.get("pattern_supported_by_reservoir") for p in patterns if "reservoir" in str(p))

    if not market_available and not sqlite_join.get("prediction_to_sqlite_join_possible"):
        classification = "INSUFFICIENT_CONTEXT_DATA"
        reason = "market snapshot context unavailable and sqlite join not possible"
    elif unique_pairs <= 1 and not reservoir_supported:
        classification = "PAIR_IDENTITY_ARTIFACT"
        reason = "signal concentrated in one pair without reservoir support"
    elif keep:
        classification = "CONTEXT_PATTERN_FOUND_FOR_E9"
        reason = f"{len(keep)} pre-entry legal feature candidates marked KEEP_FOR_E9"
    elif patterns and unique_pairs > 1:
        classification = "RARE_WINNER_RESEARCH_ONLY"
        reason = "patterns exist but are pair-concentrated or not strong enough for E9"
    elif patterns:
        classification = "RARE_WINNER_RESEARCH_ONLY"
        reason = "interesting context differences but pair-concentrated"
    else:
        classification = "NO_CONTEXT_SIGNAL"
        reason = "no distinguishing pre-entry context pattern found"
    return {
        "final_classification": classification,
        "classification_reason": reason,
        "keep_for_e9_feature_count": len(keep),
        "unique_pairs_analyzed": unique_pairs,
        "recommend_e9_training": classification == "CONTEXT_PATTERN_FOUND_FOR_E9",
    }


def limit_rows_for_smoke(rows: list[dict[str, Any]], max_rows: int) -> list[dict[str, Any]]:
    if len(rows) <= max_rows:
        return rows
    positives = [r for r in rows if r.get("group") == "rare_winner_selected_positives"]
    others = [r for r in rows if r.get("group") != "rare_winner_selected_positives"]
    keep = positives[: max_rows // 2]
    keep.extend(others[: max_rows - len(keep)])
    return keep[:max_rows]


def write_all_reports(config: ForensicsConfig, state: ForensicsState) -> None:
    reports = config.output_dir / "reports"
    writers = {
        "e8e_candidate_group_summary.csv": state.candidate_group_summary,
        "e8e_candidate_identity_map.csv": state.candidate_identity_map,
        "e8e_joinability_audit.csv": state.joinability_audit,
        "e8e_sqlite_table_inventory.csv": state.sqlite_inventory,
        "e8e_market_context_by_candidate.csv": state.market_context,
        "e8e_liquidity_dynamics.csv": state.liquidity_dynamics,
        "e8e_whale_wallet_context.csv": state.whale_wallet_context,
        "e8e_raw_payload_key_inventory.csv": state.raw_payload_inventory,
        "e8e_reservoir_overlap.csv": state.reservoir_overlap,
        "e8e_reservoir_pattern_crosscheck.csv": state.reservoir_pattern,
        "e8e_matched_control_comparison.csv": state.matched_control_comparison,
        "e8e_forensic_timeline_by_candidate.csv": state.forensic_timeline,
        "e8e_pattern_candidates.csv": state.pattern_candidates,
        "e8e_context_feature_candidate_map.csv": state.feature_candidate_map,
    }
    for name, rows in writers.items():
        atomic_write_csv(pd.DataFrame(rows), reports / name)
    if state.raw_payload_extract:
        atomic_write_csv(pd.DataFrame(state.raw_payload_extract), reports / "e8e_raw_payload_candidate_extract.csv")
    atomic_write_csv(pd.DataFrame([state.final_classification]), reports / "e8e_final_classification.csv")

    summary_lines = [
        "Phase E8E Unified Rare-Winner Context Forensics Audit",
        f"output_dir: {config.output_dir}",
        f"mode: {'smoke' if config.smoke else 'full'}",
        f"candidate_rows: {len(state.candidate_rows)}",
        f"sqlite_path: {config.sqlite_db}",
        f"reservoir_files: {len(state.reservoir_files)}",
        f"final_classification: {state.final_classification.get('final_classification')}",
        f"reason: {state.final_classification.get('classification_reason')}",
        "",
        "Safety:",
        "  no_training_performed = true",
        "  no_runtime_changes = true",
        "  no_db_writes = true",
        "  reservoir_scoring_performed = false",
    ]
    if state.fatal_blockers:
        summary_lines.extend(["", "Fatal blockers for full mode:"])
        summary_lines.extend(f"  - {b}" for b in state.fatal_blockers)
    atomic_write_text("\n".join(summary_lines) + "\n", reports / "e8e_decision_summary.txt")

    manifest = {
        "phase": PHASE,
        "created_at": utc_now_iso(),
        "input_e8b_run_dir": str(config.e8b_run_dir),
        "input_e8c_dir": str(config.e8c_dir),
        "output_directory": str(config.output_dir),
        "mode": "smoke" if config.smoke else "full",
        "no_training_performed": True,
        "no_runtime_changes": True,
        "no_db_writes": True,
        "reservoir_scoring_performed": False,
        "sqlite_path": str(config.sqlite_db) if config.sqlite_db else None,
        "sqlite_read_only": True,
        "reservoir_files_found": state.reservoir_files,
        "prediction_files_used": state.prediction_files,
        "candidate_groups": list(CANDIDATE_GROUPS),
        "windows": list(config.windows),
        "joinability_summary": state.joinability_audit,
        "context_availability_summary": state.context_availability,
        "checkpoint_resume_enabled": True,
        "completed_stages": state.completed_stages,
        "failed_stages": state.failed_stages,
        "final_classification": state.final_classification,
        "python_version": sys.version,
        "script_path": SCRIPT_PATH,
    }
    atomic_write_json(manifest, reports / "e8e_run_manifest.json")


def run_forensics(config: ForensicsConfig, *, project_root: Path) -> dict[str, Any]:
    audit_dir = config.output_dir / "audit"
    logger = E8EAuditLogger(audit_dir / "e8e_run_audit.jsonl")
    error_logger = E8EAuditLogger(audit_dir / "e8e_errors.jsonl")
    checkpoints = CheckpointManager(audit_dir, force=config.force)
    state = ForensicsState()
    rng = random.Random(config.random_state)
    logger.log("run_started", smoke=config.smoke, full=config.full)

    if not checkpoints.stage_complete("discover_inputs"):
        checkpoints.mark_stage("discover_inputs", status="completed", output_count=0)
    state.rare_winner_datasets = load_e8c_rare_winner_datasets(config.e8c_dir)
    state.reservoir_files = [str(p) for p in discover_reservoir_files(project_root)]
    state.prediction_files = []
    for ds in state.rare_winner_datasets:
        for split in ("validation", "test"):
            p = discover_prediction_file(config.e8b_run_dir, ds["dataset_name"], split)
            if p:
                state.prediction_files.append(str(p))
    state.completed_stages.append("discover_inputs")

    db_path = config.sqlite_db
    if db_path is None:
        db_path = DB_PATH

    if not checkpoints.stage_complete("sqlite_inventory"):
        state.sqlite_inventory = inventory_sqlite_tables(db_path)
        state.context_availability["sqlite_available"] = bool(state.sqlite_inventory)
        checkpoints.mark_stage("sqlite_inventory", status="completed", output_count=len(state.sqlite_inventory))
    state.completed_stages.append("sqlite_inventory")

    if not checkpoints.stage_complete("build_candidate_groups"):
        checkpoints.mark_stage("build_candidate_groups", status="completed", output_count=0)
    all_rows: list[dict[str, Any]] = []
    for ds in state.rare_winner_datasets:
        all_rows.extend(
            build_candidate_groups_for_dataset(
                ds,
                run_dir=config.e8b_run_dir,
                max_controls_per_candidate=config.max_controls_per_candidate,
                rng=rng,
            )
        )
    max_rows = config.max_candidates if config.smoke else config.max_candidates
    if config.smoke:
        max_rows = max_rows or 20
        all_rows = limit_rows_for_smoke(all_rows, max_rows)
    elif max_rows:
        all_rows = all_rows[:max_rows]
    state.candidate_rows = all_rows
    state.candidate_identity_map = [
        {
            "identity_key": r.get("identity_key"),
            "group": r.get("group"),
            "dataset_name": r.get("dataset_name"),
            "pair_address": r.get("pair_address"),
            "event_timestamp": r.get("event_timestamp"),
            "split": r.get("split"),
        }
        for r in all_rows
    ]
    summary: dict[tuple[str, str], int] = {}
    for r in all_rows:
        summary[(str(r.get("dataset_name")), str(r.get("group")))] = (
            summary.get((str(r.get("dataset_name")), str(r.get("group"))), 0) + 1
        )
    state.candidate_group_summary = [
        {"dataset_name": k[0], "group": k[1], "row_count": v} for k, v in sorted(summary.items())
    ]
    state.completed_stages.append("build_candidate_groups")

    if not checkpoints.stage_complete("joinability_audit"):
        state.joinability_audit = run_joinability_audit(
            state.candidate_rows,
            db_path=db_path,
            reservoir_files=[Path(p) for p in state.reservoir_files],
        )
        essential = next(
            (j for j in state.joinability_audit if j.get("join_strategy") == "pair_address_to_market_snapshots"),
            {},
        )
        if not essential.get("prediction_to_sqlite_join_possible") and not config.smoke:
            msg = essential.get("failure_reason") or "essential sqlite join impossible"
            state.fatal_blockers.append(msg)
            error_logger.error(msg, stage="joinability_audit")
            logger.log("joinability_failed_closed", message=msg)
        checkpoints.mark_stage("joinability_audit", status="completed")
    state.completed_stages.append("joinability_audit")

    pairs = sorted({normalize_pair(r.get("pair_address")) for r in state.candidate_rows if normalize_pair(r.get("pair_address"))})
    if db_path.exists() and pairs:
        if checkpoints.stage_complete("market_context") and not config.force:
            existing = config.output_dir / "reports" / "e8e_market_context_by_candidate.csv"
            if existing.exists():
                state.market_context = pd.read_csv(existing).to_dict("records")
                state.liquidity_dynamics = (
                    pd.read_csv(config.output_dir / "reports" / "e8e_liquidity_dynamics.csv").to_dict("records")
                    if (config.output_dir / "reports" / "e8e_liquidity_dynamics.csv").exists()
                    else [compute_liquidity_dynamics(r) for r in state.market_context]
                )
        else:
            snapshots = load_pair_snapshots(db_path, pairs)
            for row in state.candidate_rows:
                key = row.get("identity_key", "")
                if checkpoints.candidate_complete(str(key)):
                    continue
                mrow = compute_market_context_row(row, snapshots, windows=config.windows)
                state.market_context.append(mrow)
                state.liquidity_dynamics.append(compute_liquidity_dynamics(mrow))
                state.forensic_timeline.extend(build_forensic_timeline(row, snapshots, windows=config.windows))
                checkpoints.mark_candidate(str(key), group=row.get("group"))
            checkpoints.mark_stage("market_context", status="completed", output_count=len(state.market_context))
        state.context_availability["market_context_available"] = any(
            r.get("market_context_available") for r in state.market_context
        )
    state.completed_stages.append("market_context")

    if not checkpoints.stage_complete("whale_wallet_context"):
        whale_rows, whale_avail = inventory_whale_wallet(db_path, pairs)
        state.whale_wallet_context = whale_rows
        state.context_availability.update(whale_avail)
        checkpoints.mark_stage("whale_wallet_context", status="completed", output_count=len(whale_rows))
    state.completed_stages.append("whale_wallet_context")

    if not checkpoints.stage_complete("raw_payload_inventory"):
        state.raw_payload_inventory = inventory_raw_payload_keys(db_path, pairs)
        checkpoints.mark_stage("raw_payload_inventory", status="completed", output_count=len(state.raw_payload_inventory))
    state.completed_stages.append("raw_payload_inventory")

    if not checkpoints.stage_complete("reservoir_crosscheck"):
        overlap, pattern = reservoir_crosscheck(state.candidate_rows, [Path(p) for p in state.reservoir_files])
        state.reservoir_overlap = overlap
        state.reservoir_pattern = pattern
        checkpoints.mark_stage("reservoir_crosscheck", status="completed", output_count=len(overlap))
    state.completed_stages.append("reservoir_crosscheck")

    if not checkpoints.stage_complete("matched_control_comparison"):
        state.matched_control_comparison = matched_control_comparison(state.candidate_rows, state.market_context)
        checkpoints.mark_stage("matched_control_comparison", status="completed")
    state.completed_stages.append("matched_control_comparison")

    if not checkpoints.stage_complete("pattern_discovery"):
        state.pattern_candidates = discover_pattern_candidates(state.matched_control_comparison, state.market_context)
        checkpoints.mark_stage("pattern_discovery", status="completed", output_count=len(state.pattern_candidates))
    state.completed_stages.append("pattern_discovery")

    if not checkpoints.stage_complete("feature_candidate_map"):
        state.feature_candidate_map = build_feature_candidate_map(
            state.market_context,
            sqlite_available=db_path.exists(),
        )
        checkpoints.mark_stage("feature_candidate_map", status="completed", output_count=len(state.feature_candidate_map))
    state.completed_stages.append("feature_candidate_map")

    if not checkpoints.stage_complete("final_classification"):
        state.final_classification = classify_e8e_final(
            patterns=state.pattern_candidates,
            feature_map=state.feature_candidate_map,
            market_available=state.context_availability.get("market_context_available", False),
            joinability=state.joinability_audit,
            candidate_rows=state.candidate_rows,
        )
        checkpoints.mark_stage("final_classification", status="completed")
    state.completed_stages.append("final_classification")

    write_all_reports(config, state)
    checkpoints.mark_stage("finalize_outputs", status="completed")
    state.completed_stages.append("finalize_outputs")
    logger.log("run_completed", final_classification=state.final_classification.get("final_classification"))

    return {
        "output_dir": str(config.output_dir),
        "candidate_rows": len(state.candidate_rows),
        "sqlite_tables": len(state.sqlite_inventory),
        "reservoir_files": len(state.reservoir_files),
        "final_classification": state.final_classification,
        "fatal_blockers": state.fatal_blockers,
        "context_availability": state.context_availability,
        "joinability_audit": state.joinability_audit,
        "completed_stages": state.completed_stages,
    }
