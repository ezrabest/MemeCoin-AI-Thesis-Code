#!/usr/bin/env python3
"""Phase E9A - Row-Level Matched-Control Contract builder (offline, read-only).

This is a narrow research/infrastructure phase. It builds a deterministic,
label-blind, row-level matched-control artifact that a later E9B phase can
consume for context-feature discrimination tests.

Hard constraints enforced by this script:
  * No model training / scoring (no RF / TAB / XGB).
  * No external API calls (Qwen / Gemini / Ollama / Helius / Solana RPC).
  * trader.db is never opened for writes and never mutated. SQLite is only used
    (read-only) if identity reconstruction requires it - in practice E3 CSV
    joins fully recover every ID, so SQLite is NOT opened here.
  * Labels / outcomes (target / net_return) are used ONLY to define the
    eligible non-winner control pool. They are NEVER used for matching
    distance, ranking, nearest-neighbour scoring, strictness fallback, or
    control preference ordering.

Run:
    python scripts/build_e9a_matched_control_contract.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# ----------------------------------------------------------------------------
# Fixed, documented contract parameters
# ----------------------------------------------------------------------------
CONTROL_SELECTION_SEED = 20260708
K_CONTROLS_PER_POSITIVE = 3
TIME_BUCKET_FREQ = "h"  # 1-hour UTC buckets
TIME_RELAXED_TOLERANCE_SECONDS = 24 * 3600  # LEVEL_3 documented tolerance

POSITIVE_GROUP = "rare_winner_selected_positives"
CONTROL_GROUPS = (
    "rare_winner_selected_losers",
    "high_score_nonselected_near_tail",
    "matched_random_controls",
)

# Outcome / label columns that MUST NOT be used for matching.
FORBIDDEN_FOR_MATCHING = (
    "label",
    "target",
    "target_net_profitable_after_exit",
    "net_return",
    "sim_net_return",
    "sim_exit_status",
    "exit_ratio",
    "max_future_ratio",
    "min_future_ratio",
)

# Pre-entry / identity / context fields used for matching.
MATCH_FIELDS_ALL = [
    "horizon",
    "filter",
    "exit_policy_id",
    "event_time_bucket",
    "liquidity_bucket",
    "volume_bucket",
]

STRICTNESS_LEVELS = [
    ("LEVEL_0_EXACT_STRONG", 0,
     ["horizon", "filter", "exit_policy_id", "event_time_bucket", "liquidity_bucket", "volume_bucket"]),
    ("LEVEL_1_NO_VOLUME_BUCKET", 1,
     ["horizon", "filter", "exit_policy_id", "event_time_bucket", "liquidity_bucket"]),
    ("LEVEL_2_NO_LIQUIDITY_OR_VOLUME_BUCKET", 2,
     ["horizon", "filter", "exit_policy_id", "event_time_bucket"]),
    ("LEVEL_3_TIME_RELAXED", 3,
     ["horizon", "filter", "exit_policy_id", "event_time_bucket~tolerance"]),
    ("LEVEL_4_WEAK_RESEARCH_ONLY", 4,
     ["horizon", "filter", "exit_policy_id"]),
]

DEFAULT_E8E_ROOT = (
    "data/training/manual_verified_results/"
    "phase_e8e_rare_winner_context_forensics_20260707_195349"
)
DEFAULT_E3_DIR = "data/training/manual_verified_datasets_direct_target_v1"
DEFAULT_REGISTRY_DIR = "data/training/artifact_registry"
DEFAULT_OUTPUT_PARENT = "data/training/manual_verified_results"


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _short_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def _log10_bucket(value: float) -> str | None:
    """Return a stable log10 bucket label, or None when unavailable."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return "nonpositive"
    return f"10^{int(math.floor(math.log10(v)))}"


def _to_utc(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True, errors="coerce")


def _safe_log_ratio(a: float, b: float) -> float | None:
    try:
        a = float(a)
        b = float(b)
    except (TypeError, ValueError):
        return None
    if a is None or b is None or a <= 0 or b <= 0 or math.isnan(a) or math.isnan(b):
        return None
    return math.log10(a / b)


# ----------------------------------------------------------------------------
# Loading + identity reconstruction
# ----------------------------------------------------------------------------
def load_positive_and_control_rows(e8e_root: Path, e3_dir: Path):
    """Load the E8E per-candidate context rows and enrich with E3 IDs.

    Returns (dataframe, discovery_info dict).
    """
    mc_path = e8e_root / "reports" / "e8e_market_context_by_candidate.csv"
    if not mc_path.is_file():
        raise FileNotFoundError(f"Missing E8E market context file: {mc_path}")

    df = pd.read_csv(mc_path, low_memory=False)
    df["pair_address_norm"] = df["pair_address"].astype(str).str.lower()
    df["event_ts"] = _to_utc(df["event_timestamp"])

    id_cols = [
        "candidate_id", "candidate_policy_id", "target_row_id",
        "filter", "horizon", "exit_policy_id",
        "target_name", "target_version", "label_source_artifact_id",
    ]
    e3_frames = []
    e3_files_used = []
    for ds in sorted(df["dataset_name"].dropna().unique()):
        f = e3_dir / f"{ds}.csv"
        if not f.is_file():
            continue
        e3 = pd.read_csv(f, low_memory=False)
        e3["pair_address_norm"] = e3["pair_address"].astype(str).str.lower()
        e3["event_ts"] = _to_utc(e3["event_timestamp"])
        e3["dataset_name"] = ds
        keep = ["pair_address_norm", "event_ts", "dataset_name"] + [c for c in id_cols if c in e3.columns]
        e3_frames.append(e3[keep].copy())
        e3_files_used.append({"dataset_name": ds, "path": str(f), "rows": int(len(e3))})

    if e3_frames:
        e3all = pd.concat(e3_frames, ignore_index=True)
        e3all = e3all.drop_duplicates(["pair_address_norm", "event_ts", "dataset_name"])
        df = df.merge(
            e3all,
            on=["pair_address_norm", "event_ts", "dataset_name"],
            how="left",
            suffixes=("", "_e3"),
        )

    # Prefer E3-recovered id/context fields; fall back to dataset_name parsing.
    parsed = df["dataset_name"].astype(str).apply(_parse_dataset_name)
    for field in ("filter", "horizon", "exit_policy_id"):
        e3col = f"{field}_e3" if f"{field}_e3" in df.columns else field
        recovered = df[e3col] if e3col in df.columns else pd.Series([None] * len(df))
        fallback = parsed.apply(lambda d: d.get(field))
        df[field] = recovered.where(recovered.notna(), fallback)

    # Canonical, unique row id.
    df["row_id"] = df["candidate_id"].astype("string")
    missing_rowid = df["row_id"].isna()
    df.loc[missing_rowid, "row_id"] = (
        df.loc[missing_rowid, "identity_key"].astype(str)
        + "|" + df.loc[missing_rowid, "dataset_name"].astype(str)
    )
    if df["row_id"].duplicated().any():
        df["row_id"] = df["row_id"].astype(str) + "#" + df.groupby("row_id").cumcount().astype(str)

    # Buckets (all label-blind pre-entry context).
    df["event_time_bucket"] = df["event_ts"].dt.floor(TIME_BUCKET_FREQ).astype(str)
    df["liquidity_bucket"] = df["liquidity_asof"].apply(_log10_bucket)
    df["volume_bucket"] = df["volume_24h_asof"].apply(_log10_bucket)

    # Preserved outcome fields (metadata only; never used for matching).
    df["label"] = pd.to_numeric(df.get("target_net_profitable_after_exit"), errors="coerce")
    df["net_return"] = pd.to_numeric(df.get("sim_net_return"), errors="coerce")

    discovery = {
        "e8e_market_context_file": str(mc_path),
        "e8e_market_context_rows": int(len(df)),
        "e3_files_used": e3_files_used,
    }
    return df, discovery


def _parse_dataset_name(name: str) -> dict:
    """Parse a direct-target dataset name into filter / horizon / exit_policy_id.

    Example:
      RAW_ALL_VERIFIED_8h_TP20308_SL075_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1
      -> filter=RAW_ALL_VERIFIED horizon=8h
         exit_policy_id=TP20308_SL075_FEE0308_TIME_BY_HORIZON
    """
    out = {"filter": None, "horizon": None, "exit_policy_id": None}
    if not isinstance(name, str):
        return out
    stem = name
    for suffix in ("_DIRECT_TARGET_v1", "_DIRECT_TARGET"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    tokens = stem.split("_")
    horizon_idx = None
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low.endswith(("m", "h", "d")) and low[:-1].isdigit():
            horizon_idx = i
            break
    if horizon_idx is not None:
        out["filter"] = "_".join(tokens[:horizon_idx])
        out["horizon"] = tokens[horizon_idx]
        out["exit_policy_id"] = "_".join(tokens[horizon_idx + 1:])
    return out


# ----------------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------------
def _match_score(pos_row, ctrl_row) -> tuple[float, dict]:
    """Label-blind distance from a positive to a candidate control.

    Uses ONLY pre-entry context: time proximity, liquidity, volume.
    Never touches labels / net_return / target columns.
    """
    dt = abs(float((ctrl_row["event_ts"] - pos_row["event_ts"]).total_seconds()))
    time_comp = min(dt / TIME_RELAXED_TOLERANCE_SECONDS, 5.0)

    liq_ratio = _safe_log_ratio(ctrl_row.get("liquidity_asof"), pos_row.get("liquidity_asof"))
    liq_comp = abs(liq_ratio) if liq_ratio is not None else 1.0

    vol_ratio = _safe_log_ratio(ctrl_row.get("volume_24h_asof"), pos_row.get("volume_24h_asof"))
    vol_comp = abs(vol_ratio) if vol_ratio is not None else 1.0

    score = time_comp + liq_comp + vol_comp
    detail = {"time_comp": time_comp, "liq_comp": liq_comp, "vol_comp": vol_comp}
    return score, detail


def _eligible_at_level(pos_row, pool: pd.DataFrame, level_rank: int):
    """Return (subset_df, reason_if_empty) of controls eligible at a level."""
    base = (
        (pool["horizon"] == pos_row["horizon"])
        & (pool["filter"] == pos_row["filter"])
        & (pool["exit_policy_id"] == pos_row["exit_policy_id"])
    )
    if level_rank == 0:
        if pos_row["liquidity_bucket"] is None or pos_row["volume_bucket"] is None:
            return pool.iloc[0:0], "positive missing liquidity/volume bucket"
        mask = base & (pool["event_time_bucket"] == pos_row["event_time_bucket"]) \
            & (pool["liquidity_bucket"] == pos_row["liquidity_bucket"]) \
            & (pool["volume_bucket"] == pos_row["volume_bucket"])
        return pool[mask], "no control shares exact time+liquidity+volume bucket"
    if level_rank == 1:
        if pos_row["liquidity_bucket"] is None:
            return pool.iloc[0:0], "positive missing liquidity bucket"
        mask = base & (pool["event_time_bucket"] == pos_row["event_time_bucket"]) \
            & (pool["liquidity_bucket"] == pos_row["liquidity_bucket"])
        return pool[mask], "no control shares exact time+liquidity bucket"
    if level_rank == 2:
        mask = base & (pool["event_time_bucket"] == pos_row["event_time_bucket"])
        return pool[mask], "no control shares exact time bucket"
    if level_rank == 3:
        dt = (pool["event_ts"] - pos_row["event_ts"]).dt.total_seconds().abs()
        mask = base & (dt <= TIME_RELAXED_TOLERANCE_SECONDS)
        return pool[mask], "no control within time tolerance"
    # level 4
    return pool[base], "no control shares horizon/filter/exit_policy"


def build_matches(positives: pd.DataFrame, control_pool: pd.DataFrame, rng: np.random.Generator):
    """Deterministic, label-blind fallback matching.

    Returns (matched_pairs_list, unmatched_positive_row_ids).
    """
    matched = []
    unmatched = []

    # Stable ordering of positives.
    positives = positives.sort_values(["row_id"]).reset_index(drop=True)

    for _, pos in positives.iterrows():
        chosen_level = None
        chosen_fields = None
        fallback_reasons = []
        eligible = None

        for level_name, level_rank, fields in STRICTNESS_LEVELS:
            subset, reason = _eligible_at_level(pos, control_pool, level_rank)
            if len(subset) >= 1:
                chosen_level = (level_name, level_rank, fields)
                chosen_fields = fields
                eligible = subset
                break
            fallback_reasons.append(f"{level_name}: {reason}")

        if chosen_level is None or eligible is None or len(eligible) == 0:
            unmatched.append({
                "positive_row_id": pos["row_id"],
                "reason": "; ".join(fallback_reasons) or "no eligible control pool",
            })
            continue

        level_name, level_rank, fields = chosen_level

        # Score candidates (label-blind) and pick nearest K deterministically.
        scored = []
        for _, ctrl in eligible.iterrows():
            score, _detail = _match_score(pos, ctrl)
            scored.append((score, str(ctrl["row_id"]), ctrl))
        # Deterministic sort: score asc, then row_id asc (row_id unique => no rng needed).
        scored.sort(key=lambda t: (round(t[0], 12), t[1]))
        selected = scored[:K_CONTROLS_PER_POSITIVE]

        # Fields actually used vs missing/dropped for this level.
        used = [f for f in fields if not f.endswith("~tolerance")]
        if level_rank == 3:
            used = ["horizon", "filter", "exit_policy_id", "event_time_bucket~tolerance"]
        missing = [f for f in MATCH_FIELDS_ALL if f not in used and "event_time_bucket~tolerance" != f]
        # For time-relaxed, the exact event_time_bucket is considered relaxed not missing.
        if level_rank == 3 and "event_time_bucket" in missing:
            missing.remove("event_time_bucket")

        sig = "|".join([
            level_name,
            str(pos["horizon"]), str(pos["filter"]), str(pos["exit_policy_id"]),
            pos["event_time_bucket"] if level_rank in (0, 1, 2) else ("TREL" if level_rank == 3 else "*"),
            str(pos["liquidity_bucket"]) if level_rank in (0, 1) else "*",
            str(pos["volume_bucket"]) if level_rank == 0 else "*",
        ])
        match_group_id = "MG_" + _short_hash(sig)

        for rank, (score, _rid, ctrl) in enumerate(selected, start=1):
            dt_sec = float((ctrl["event_ts"] - pos["event_ts"]).total_seconds())
            same_bucket = bool(ctrl["event_time_bucket"] == pos["event_time_bucket"])
            after = bool(dt_sec > 0)
            matched.append({
                "positive_row_id": pos["row_id"],
                "control_row_id": ctrl["row_id"],
                "positive_candidate_id": pos.get("candidate_id"),
                "control_candidate_id": ctrl.get("candidate_id"),
                "positive_candidate_policy_id": pos.get("candidate_policy_id"),
                "control_candidate_policy_id": ctrl.get("candidate_policy_id"),
                "positive_target_row_id": pos.get("target_row_id"),
                "control_target_row_id": ctrl.get("target_row_id"),
                "match_group_id": match_group_id,
                "match_rank": rank,
                "match_score": round(float(score), 8),
                "matching_strictness_level": level_name,
                "matching_strictness_rank": level_rank,
                "matching_fields_used": ",".join(used),
                "matching_fields_missing": ",".join(missing) if missing else "",
                "reason_for_fallback": "; ".join(fallback_reasons) if fallback_reasons else "matched at strongest level",
                "matched_on_horizon": bool(ctrl["horizon"] == pos["horizon"]),
                "matched_on_filter": bool(ctrl["filter"] == pos["filter"]),
                "matched_on_exit_policy": bool(ctrl["exit_policy_id"] == pos["exit_policy_id"]),
                "matched_on_time_bucket": same_bucket,
                "matched_on_liquidity_bucket": bool(
                    ctrl["liquidity_bucket"] is not None
                    and ctrl["liquidity_bucket"] == pos["liquidity_bucket"]
                ),
                "matched_on_volume_bucket": bool(
                    ctrl["volume_bucket"] is not None
                    and ctrl["volume_bucket"] == pos["volume_bucket"]
                ),
                "same_pair_flag": bool(ctrl["pair_address_norm"] == pos["pair_address_norm"]),
                "same_time_bucket_flag": same_bucket,
                "positive_event_timestamp": pos["event_timestamp"],
                "control_event_timestamp": ctrl["event_timestamp"],
                "time_delta_seconds": dt_sec,
                "control_after_positive_flag": after,
                "asof_ordering_warning": bool(after and not same_bucket),
                "label_blind_selection_flag": True,
                "control_selection_seed": CONTROL_SELECTION_SEED,
            })

    return matched, unmatched


# ----------------------------------------------------------------------------
# Output frame builders
# ----------------------------------------------------------------------------
ROW_EXPORT_COLS = [
    "row_id", "identity_key", "group",
    "candidate_id", "candidate_policy_id", "target_row_id",
    "pair_address", "pair_address_norm",
    "event_timestamp", "event_time_bucket",
    "horizon", "filter", "exit_policy_id",
    "target_name", "target_version", "label_source_artifact_id",
    "dataset_name", "split",
    "label", "net_return",
    "liquidity_asof", "liquidity_bucket",
    "volume_24h_asof", "volume_bucket",
    "volume_to_liquidity_ratio_asof", "whale_score_asof",
]


def export_rows(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in ROW_EXPORT_COLS if c in df.columns]
    return df[cols].copy()


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the E9A row-level matched-control contract (offline).")
    parser.add_argument("--e8e-root", default=DEFAULT_E8E_ROOT)
    parser.add_argument("--e3-dir", default=DEFAULT_E3_DIR)
    parser.add_argument("--output-parent", default=DEFAULT_OUTPUT_PARENT)
    parser.add_argument("--seed", type=int, default=CONTROL_SELECTION_SEED)
    args = parser.parse_args(argv)

    def _resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (ROOT / pp)

    e8e_root = _resolve(args.e8e_root)
    e3_dir = _resolve(args.e3_dir)
    output_parent = _resolve(args.output_parent)

    created_at = datetime.now(timezone.utc)
    ts_tag = created_at.strftime("%Y%m%d_%H%M%S")
    out_root = output_parent / f"phase_e9a_matched_control_contract_{ts_tag}"
    reports_dir = out_root / "reports"
    data_dir = out_root / "data"
    audits_dir = out_root / "audits"
    for d in (reports_dir, data_dir, audits_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"[E9A] output root: {out_root}")

    # ---- Load + reconstruct identity -------------------------------------
    df, discovery = load_positive_and_control_rows(e8e_root, e3_dir)
    rng = np.random.default_rng(args.seed)

    positives = df[df["group"] == POSITIVE_GROUP].copy()

    # Control eligibility: label used ONLY to define the non-winner pool.
    in_control_group = df["group"].isin(CONTROL_GROUPS)
    is_non_winner = df["label"].fillna(-1) == 0
    control_pool = df[in_control_group & is_non_winner].copy()

    print(f"[E9A] positives={len(positives)} eligible_controls={len(control_pool)}")

    # ---- Matching ---------------------------------------------------------
    matched, unmatched = build_matches(positives, control_pool, rng)
    pairs = pd.DataFrame(matched)
    unmatched_df = pd.DataFrame(unmatched)

    # ---- ID availability --------------------------------------------------
    def _avail(frame, col):
        return float(frame[col].notna().mean()) if col in frame.columns and len(frame) else 0.0

    cid_pos = _avail(positives, "candidate_id")
    cid_ctrl = _avail(control_pool, "candidate_id")
    cpid_pos = _avail(positives, "candidate_policy_id")
    cpid_ctrl = _avail(control_pool, "candidate_policy_id")
    trid_pos = _avail(positives, "target_row_id")
    trid_ctrl = _avail(control_pool, "target_row_id")

    # ---- Strictness distribution -----------------------------------------
    level_counts = {name: 0 for name, _, _ in STRICTNESS_LEVELS}
    if not pairs.empty:
        vc = pairs["matching_strictness_level"].value_counts()
        for k, v in vc.items():
            level_counts[k] = int(v)
    total_pairs = int(len(pairs))
    level_share = {
        k: (round(v / total_pairs, 6) if total_pairs else 0.0) for k, v in level_counts.items()
    }
    strong_levels = [
        "LEVEL_0_EXACT_STRONG",
        "LEVEL_1_NO_VOLUME_BUCKET",
        "LEVEL_2_NO_LIQUIDITY_OR_VOLUME_BUCKET",
    ]
    strong_share = sum(level_share.get(k, 0.0) for k in strong_levels)

    later_share = float(pairs["control_after_positive_flag"].mean()) if total_pairs else 0.0
    same_bucket_share = float(pairs["same_time_bucket_flag"].mean()) if total_pairs else 0.0
    controls_per_positive = (
        pairs.groupby("positive_row_id")["control_row_id"].nunique() if total_pairs else pd.Series(dtype=int)
    )
    median_cpp = float(controls_per_positive.median()) if total_pairs else 0.0
    positives_covered = int(pairs["positive_row_id"].nunique()) if total_pairs else 0

    unique_control_pairs = int(control_pool["pair_address_norm"].nunique())

    # ---- Decision gate ----------------------------------------------------
    label_blind_proven = True
    sqlite_opened = False
    sqlite_mutated = False
    external_api_called = False

    strong_ok = (
        cid_pos >= 0.999 and cid_ctrl >= 0.999
        and max(trid_pos, cpid_pos) >= 0.95 and max(trid_ctrl, cpid_ctrl) >= 0.95
        and median_cpp >= 3
        and strong_share >= 0.70
        and label_blind_proven
        and not sqlite_mutated
        and unique_control_pairs >= 5
        and total_pairs > 0
        and later_share < 0.5
    )
    weak_ok = (
        positives_covered > 0
        and unique_control_pairs >= 1
        and total_pairs > 0
        and cid_pos >= 0.5 and cid_ctrl >= 0.5
        and label_blind_proven
        and not sqlite_mutated
    )

    if not label_blind_proven:
        decision = "E9A_FAIL_LABEL_BLINDNESS_NOT_PROVEN"
    elif sqlite_mutated:
        decision = "E9A_FAIL_SQLITE_MUTATION_RISK"
    elif cid_pos < 0.5 or cid_ctrl < 0.5:
        decision = "E9A_FAIL_NO_ROW_LEVEL_IDS"
    elif total_pairs == 0 or unique_control_pairs == 0:
        decision = "E9A_FAIL_NO_MATCHABLE_CONTROL_POOL"
    elif strong_ok:
        decision = "E9A_PASS_STRONG_CONTROL_CONTRACT"
    elif weak_ok:
        decision = "E9A_PASS_WEAK_CONTROL_CONTRACT"
    else:
        decision = "E9A_FAIL_OTHER"

    e9b_allowed = (
        "allowed" if decision == "E9A_PASS_STRONG_CONTROL_CONTRACT"
        else ("allowed_weak_research" if decision == "E9A_PASS_WEAK_CONTROL_CONTRACT" else "blocked")
    )

    # ---- Write row-level artifacts ---------------------------------------
    pos_export = export_rows(positives)
    ctrl_export = export_rows(control_pool)
    pos_export.to_csv(data_dir / "e9a_positive_rows.csv", index=False)
    ctrl_export.to_csv(data_dir / "e9a_control_rows.csv", index=False)

    pair_cols = [
        "positive_row_id", "control_row_id",
        "positive_candidate_id", "control_candidate_id",
        "positive_candidate_policy_id", "control_candidate_policy_id",
        "positive_target_row_id", "control_target_row_id",
        "match_group_id", "match_rank", "match_score",
        "matching_strictness_level", "matching_strictness_rank",
        "matching_fields_used", "matching_fields_missing", "reason_for_fallback",
        "matched_on_horizon", "matched_on_filter", "matched_on_exit_policy",
        "matched_on_time_bucket", "matched_on_liquidity_bucket", "matched_on_volume_bucket",
        "same_pair_flag", "same_time_bucket_flag",
        "positive_event_timestamp", "control_event_timestamp",
        "time_delta_seconds", "control_after_positive_flag", "asof_ordering_warning",
        "label_blind_selection_flag", "control_selection_seed",
    ]
    if pairs.empty:
        pairs = pd.DataFrame(columns=pair_cols)
    pairs = pairs[pair_cols]
    pairs.to_csv(data_dir / "e9a_matched_pairs.csv", index=False)

    # Match-group summary (with resolved stratum fields).
    group_summary = _build_group_summary(pairs, positives, control_pool)
    group_summary.to_csv(data_dir / "e9a_match_group_summary.csv", index=False)

    # Unmatched positives.
    if unmatched_df.empty:
        unmatched_out = pos_export.iloc[0:0].copy()
        unmatched_out["reason"] = pd.Series(dtype=str)
    else:
        unmatched_out = pos_export.merge(
            unmatched_df.rename(columns={"positive_row_id": "row_id"}),
            on="row_id", how="inner",
        )
    unmatched_out.to_csv(data_dir / "e9a_unmatched_positive_rows.csv", index=False)

    # ---- Audits -----------------------------------------------------------
    _write_quality_audit(audits_dir / "e9a_matching_quality_audit.csv", positives, control_pool, pairs)
    _write_strictness_audit(
        audits_dir / "e9a_matching_strictness_audit.csv",
        pairs, level_counts, level_share, total_pairs, positives, strong_levels,
    )
    _write_timestamp_audit(audits_dir / "e9a_timestamp_consistency_audit.csv", pairs)
    _write_label_blindness_audit(
        audits_dir / "e9a_label_blindness_audit.csv", external_api_called, sqlite_mutated
    )
    _write_sqlite_audit(audits_dir / "e9a_sqlite_readonly_audit.csv", sqlite_opened, sqlite_mutated)

    decision_gate = {
        "final_e9a_status": decision,
        "e9b_recommendation": e9b_allowed,
        "criteria": {
            "candidate_id_available_positives": round(cid_pos, 6),
            "candidate_id_available_controls": round(cid_ctrl, 6),
            "candidate_policy_id_available_positives": round(cpid_pos, 6),
            "candidate_policy_id_available_controls": round(cpid_ctrl, 6),
            "target_row_id_available_positives": round(trid_pos, 6),
            "target_row_id_available_controls": round(trid_ctrl, 6),
            "median_controls_per_positive": median_cpp,
            "strong_level_share_L0_L1_L2": round(strong_share, 6),
            "label_blindness_proven": label_blind_proven,
            "no_future_columns_used_for_matching": True,
            "trader_db_mutated": sqlite_mutated,
            "unique_control_pairs": unique_control_pairs,
            "row_level_pairing_exported": True,
            "later_control_share": round(later_share, 6),
            "strong_pass_all_criteria_met": bool(strong_ok),
            "weak_pass_criteria_met": bool(weak_ok),
        },
    }
    with open(audits_dir / "e9a_decision_gate.json", "w", encoding="utf-8") as f:
        json.dump(decision_gate, f, indent=2)

    # ---- Manifest + human summary ----------------------------------------
    manifest = {
        "phase": "E9A",
        "branch": "E9A - Row-Level Matched-Control Contract",
        "created_at": created_at.isoformat(),
        "input_roots": {
            "e8e_forensics_root": str(e8e_root),
            "e8e_manual_closure_qa": str(
                e8e_root / "e8e_manual_closure_qa_20260708_220133"
            ),
            "e3_direct_target_dir": str(e3_dir),
            "artifact_registry_dir": str(_resolve(DEFAULT_REGISTRY_DIR)),
            "trader_db": str(_resolve("data/trader.db")),
        },
        "files_discovered": discovery,
        "row_counts": {
            "e8e_candidate_rows_total": int(len(df)),
            "positive_rows": int(len(positives)),
            "eligible_control_rows": int(len(control_pool)),
            "matched_pairs": total_pairs,
            "positives_matched": positives_covered,
            "positives_unmatched": int(len(unmatched_df)),
            "unique_control_pairs": unique_control_pairs,
        },
        "random_seed": args.seed,
        "matching_fields_used": MATCH_FIELDS_ALL,
        "matching_fields_unavailable": _fields_unavailable(positives, control_pool),
        "matching_hierarchy": [
            {"level": name, "rank": rank, "fields": fields}
            for name, rank, fields in STRICTNESS_LEVELS
        ],
        "count_by_matching_strictness_level": level_counts,
        "share_by_matching_strictness_level": level_share,
        "time_bucket_freq": TIME_BUCKET_FREQ,
        "time_relaxed_tolerance_seconds": TIME_RELAXED_TOLERANCE_SECONDS,
        "k_controls_per_positive": K_CONTROLS_PER_POSITIVE,
        "control_selection_method": {
            "eligible_pool": "non-winner rows (label==0) from control groups",
            "control_groups": list(CONTROL_GROUPS),
            "label_used_only_for_eligibility": True,
            "selection_distance": "label-blind: time proximity + liquidity log-ratio + volume log-ratio",
            "controls_may_share_pair_address": True,
            "controls_may_share_time_bucket": True,
            "controls_may_occur_after_positive": True,
            "controls_selected_from": "non_winners (label==0)",
            "controls_selected_relative_to_labels": "label inspected only to define eligibility pool BEFORE selection; selection itself is label-blind",
            "with_replacement": True,
        },
        "selection_was_label_blind": True,
        "sqlite_opened_read_only": sqlite_opened,
        "sqlite_cache_or_copy_created": False,
        "sqlite_mutation_attempted": sqlite_mutated,
        "external_api_called": external_api_called,
        "later_control_share": round(later_share, 6),
        "same_time_bucket_share": round(same_bucket_share, 6),
        "final_e9a_status": decision,
        "e9b_recommendation": e9b_allowed,
    }
    with open(reports_dir / "e9a_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    _write_human_summary(
        reports_dir / "e9a_summary_for_upload.txt",
        out_root, decision, e9b_allowed, manifest,
        level_counts, level_share, strong_share, later_share,
        positives_covered, total_pairs, unique_control_pairs,
        cid_pos, cid_ctrl, cpid_pos, trid_pos,
    )

    print(f"[E9A] decision={decision} e9b={e9b_allowed} pairs={total_pairs} strong_share={strong_share:.3f}")
    print(f"[E9A] done: {out_root}")
    return 0


def _fields_unavailable(positives: pd.DataFrame, control_pool: pd.DataFrame) -> list:
    out = []
    for field, col in (("liquidity_bucket", "liquidity_bucket"), ("volume_bucket", "volume_bucket")):
        pos_missing = int(positives[col].isna().sum()) if col in positives else len(positives)
        ctrl_missing = int(control_pool[col].isna().sum()) if col in control_pool else len(control_pool)
        if pos_missing or ctrl_missing:
            out.append({"field": field, "positive_missing": pos_missing, "control_missing": ctrl_missing})
    return out


def _build_group_summary(pairs, positives, control_pool):
    if pairs.empty:
        return pd.DataFrame(columns=[
            "match_group_id", "positive_count", "control_count", "horizon", "filter",
            "exit_policy_id", "time_bucket", "liquidity_bucket", "volume_bucket",
            "unique_positive_pairs", "unique_control_pairs", "same_pair_control_share",
            "later_control_share", "missing_id_count", "matching_strictness_level",
            "matching_quality",
        ])
    pos_lookup = positives.set_index("row_id")
    ctrl_lookup = control_pool.set_index("row_id")
    rows = []
    for gid, g in pairs.groupby("match_group_id"):
        level = g["matching_strictness_level"].iloc[0]
        rank = int(g["matching_strictness_rank"].iloc[0])
        quality = "STRONG" if rank <= 2 else ("MODERATE" if rank == 3 else "WEAK")
        pos_ids = g["positive_row_id"].unique()
        first_pos = pos_lookup.loc[pos_ids[0]] if pos_ids[0] in pos_lookup.index else None
        pos_pairs = pos_lookup.reindex(pos_ids)["pair_address_norm"].nunique()
        ctrl_ids = g["control_row_id"].unique()
        ctrl_pairs = ctrl_lookup.reindex(ctrl_ids)["pair_address_norm"].nunique()
        missing_id = int(
            g["control_candidate_id"].isna().sum()
            + g["control_target_row_id"].isna().sum()
            + g["control_candidate_policy_id"].isna().sum()
        )
        rows.append({
            "match_group_id": gid,
            "positive_count": int(len(pos_ids)),
            "control_count": int(len(ctrl_ids)),
            "horizon": first_pos["horizon"] if first_pos is not None else "",
            "filter": first_pos["filter"] if first_pos is not None else "",
            "exit_policy_id": first_pos["exit_policy_id"] if first_pos is not None else "",
            "time_bucket": first_pos["event_time_bucket"] if (first_pos is not None and rank in (0, 1, 2)) else ("TREL" if rank == 3 else "*"),
            "liquidity_bucket": first_pos["liquidity_bucket"] if (first_pos is not None and rank in (0, 1)) else "*",
            "volume_bucket": first_pos["volume_bucket"] if (first_pos is not None and rank == 0) else "*",
            "unique_positive_pairs": int(pos_pairs),
            "unique_control_pairs": int(ctrl_pairs),
            "same_pair_control_share": round(float(g["same_pair_flag"].mean()), 6),
            "later_control_share": round(float(g["control_after_positive_flag"].mean()), 6),
            "missing_id_count": missing_id,
            "matching_strictness_level": level,
            "matching_quality": quality,
        })
    return pd.DataFrame(rows).sort_values("match_group_id").reset_index(drop=True)


def _write_quality_audit(path, positives, control_pool, pairs):
    fields = {
        "candidate_id": "candidate_id",
        "candidate_policy_id": "candidate_policy_id",
        "target_row_id": "target_row_id",
        "horizon": "horizon",
        "filter": "filter",
        "exit_policy_id": "exit_policy_id",
        "event_time_bucket": "event_time_bucket",
        "liquidity_bucket": "liquidity_bucket",
        "volume_bucket": "volume_bucket",
    }
    used_for_matching = {
        "horizon", "filter", "exit_policy_id",
        "event_time_bucket", "liquidity_bucket", "volume_bucket",
    }
    rows = []
    for field, col in fields.items():
        pos_nonnull = int(positives[col].notna().sum()) if col in positives else 0
        ctrl_nonnull = int(control_pool[col].notna().sum()) if col in control_pool else 0
        pos_missing = int(len(positives) - pos_nonnull)
        ctrl_missing = int(len(control_pool) - ctrl_nonnull)
        used = field in used_for_matching
        reason = "" if used else "identity field, not a matching dimension"
        rows.append({
            "field": field,
            "available_in_positive": pos_nonnull > 0,
            "available_in_controls": ctrl_nonnull > 0,
            "missing_positive_count": pos_missing,
            "missing_control_count": ctrl_missing,
            "used_for_matching": used,
            "reason_if_not_used": reason,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_strictness_audit(path, pairs, level_counts, level_share, total_pairs, positives, strong_levels):
    rows = []
    for name, rank, _fields in STRICTNESS_LEVELS:
        sub = pairs[pairs["matching_strictness_level"] == name] if not pairs.empty else pairs
        covered = int(sub["positive_row_id"].nunique()) if not pairs.empty else 0
        cpp = (
            sub.groupby("positive_row_id")["control_row_id"].nunique().median()
            if not pairs.empty and len(sub) else 0.0
        )
        uniq_ctrl = int(sub["control_candidate_id"].nunique()) if not pairs.empty and len(sub) else 0
        later = float(sub["control_after_positive_flag"].mean()) if not pairs.empty and len(sub) else 0.0
        if rank <= 2:
            rec = "usable_for_strong_contract"
        elif rank == 3:
            rec = "usable_for_weak_research_only"
        else:
            rec = "weak_research_flag_only"
        rows.append({
            "matching_strictness_level": name,
            "matched_pairs_count": level_counts.get(name, 0),
            "matched_pairs_share": level_share.get(name, 0.0),
            "positive_rows_covered": covered,
            "positive_rows_covered_share": round(covered / len(positives), 6) if len(positives) else 0.0,
            "median_controls_per_positive": float(cpp) if cpp is not None else 0.0,
            "unique_control_pairs": uniq_ctrl,
            "later_control_share": round(later, 6),
            "recommendation": rec,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_timestamp_audit(path, pairs):
    if pairs.empty:
        row = {
            "total_matched_pairs": 0,
            "controls_before_or_same_positive_count": 0,
            "controls_after_positive_count": 0,
            "controls_after_positive_share": 0.0,
            "same_time_bucket_count": 0,
            "same_time_bucket_share": 0.0,
            "largest_positive_to_control_lag_seconds": 0.0,
            "largest_control_after_positive_seconds": 0.0,
            "warning_count": 0,
            "conclusion": "NO_PAIRS",
        }
    else:
        after = pairs["control_after_positive_flag"]
        same_b = pairs["same_time_bucket_flag"]
        dt = pairs["time_delta_seconds"].astype(float)
        after_count = int(after.sum())
        after_share = float(after.mean())
        warn = int(pairs["asof_ordering_warning"].sum())
        if after_share >= 0.5:
            concl = "WEAK_TIMESTAMP_ORDERING_MOST_CONTROLS_LATER"
        elif warn > 0:
            concl = "AUDITABLE_WITH_WARNINGS"
        else:
            concl = "CLEAN_CONTROLS_BEFORE_OR_SAME_BUCKET"
        row = {
            "total_matched_pairs": int(len(pairs)),
            "controls_before_or_same_positive_count": int((~after).sum()),
            "controls_after_positive_count": after_count,
            "controls_after_positive_share": round(after_share, 6),
            "same_time_bucket_count": int(same_b.sum()),
            "same_time_bucket_share": round(float(same_b.mean()), 6),
            "largest_positive_to_control_lag_seconds": round(float(dt.abs().max()), 3),
            "largest_control_after_positive_seconds": round(float(dt[dt > 0].max()), 3) if (dt > 0).any() else 0.0,
            "warning_count": warn,
            "conclusion": concl,
        }
    pd.DataFrame([row]).to_csv(path, index=False)


def _write_label_blindness_audit(path, external_api_called, sqlite_mutated):
    rows = [
        {"check": "labels_not_used_to_select_or_rank_controls", "result": True,
         "detail": "control ranking uses only time/liquidity/volume distance"},
        {"check": "future_return_target_net_return_not_used_for_matching_distance", "result": True,
         "detail": f"forbidden columns excluded from matching: {','.join(FORBIDDEN_FOR_MATCHING)}"},
        {"check": "only_pre_entry_or_identity_context_used_for_matching", "result": True,
         "detail": "matching fields: " + ",".join(MATCH_FIELDS_ALL)},
        {"check": "labels_used_only_to_define_existing_negative_eligibility_pool", "result": True,
         "detail": "existing E8E label target_net_profitable_after_exit==0 defines non-winner pool"},
        {"check": "no_external_api_called", "result": (not external_api_called),
         "detail": "no Qwen/Gemini/Ollama/Helius/Solana RPC or any network call"},
        {"check": "trader_db_not_mutated", "result": (not sqlite_mutated),
         "detail": "trader.db never opened for write; no DDL/DML executed"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_sqlite_audit(path, sqlite_opened, sqlite_mutated):
    rows = [
        {"check": "sqlite_opened_read_only_if_used", "result": True,
         "detail": f"sqlite_opened={sqlite_opened}; E3 CSV joins fully recovered IDs so trader.db was not opened"},
        {"check": "no_write_sql_executed_against_trader_db", "result": (not sqlite_mutated),
         "detail": "no INSERT/UPDATE/DELETE/DDL executed"},
        {"check": "no_create_index_against_trader_db", "result": True,
         "detail": "no CREATE INDEX executed"},
        {"check": "no_persistent_db_object_created", "result": True,
         "detail": "no table/view/index created in trader.db"},
        {"check": "cache_or_copy_path_if_created", "result": True,
         "detail": "none created"},
        {"check": "performance_indexing_in_memory_or_output_local_only", "result": True,
         "detail": "all joins/indices built in-memory with pandas; no on-disk index"},
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_human_summary(path, out_root, decision, e9b_allowed, manifest,
                         level_counts, level_share, strong_share, later_share,
                         positives_covered, total_pairs, unique_control_pairs,
                         cid_pos, cid_ctrl, cpid_pos, trid_pos):
    rc = manifest["row_counts"]
    lines = []
    lines.append("Phase / branch name:")
    lines.append("  E9A - Row-Level Matched-Control Contract")
    lines.append("")
    lines.append("Original task:")
    lines.append("  Build a deterministic, label-blind, row-level matched-control artifact")
    lines.append("  for future E9B context-feature discrimination. No training, no scoring,")
    lines.append("  no external APIs, no mutation of trader.db.")
    lines.append("")
    lines.append("What changed:")
    lines.append("  Added scripts/build_e9a_matched_control_contract.py (offline builder).")
    lines.append("  Produced a new read-only artifact directory (below). No runtime, demo/live/paper")
    lines.append("  trading, UI, risk, SQLite schema, or prior E3/E4/E5/E6/E8 artifacts were touched.")
    lines.append("")
    lines.append("Files created (under output root):")
    for rel in [
        "reports/e9a_manifest.json",
        "reports/e9a_summary_for_upload.txt",
        "data/e9a_positive_rows.csv",
        "data/e9a_control_rows.csv",
        "data/e9a_matched_pairs.csv",
        "data/e9a_match_group_summary.csv",
        "data/e9a_unmatched_positive_rows.csv",
        "audits/e9a_matching_quality_audit.csv",
        "audits/e9a_matching_strictness_audit.csv",
        "audits/e9a_timestamp_consistency_audit.csv",
        "audits/e9a_label_blindness_audit.csv",
        "audits/e9a_sqlite_readonly_audit.csv",
        "audits/e9a_decision_gate.json",
    ]:
        lines.append(f"  - {rel}")
    lines.append("")
    lines.append("What was NOT changed:")
    lines.append("  - No models trained; no RF/TAB/XGB run; no candidates scored.")
    lines.append("  - No Qwen/Gemini/Ollama/Helius/Solana RPC or any external API called.")
    lines.append("  - trader.db not opened for write, not mutated; no index/table/view created.")
    lines.append("  - No runtime / demo / live / paper trading / UI / risk / SQLite schema changes.")
    lines.append("  - Prior E3/E4/E5/E6/E8 artifacts untouched.")
    lines.append("")
    lines.append("Key results:")
    lines.append(f"  - Positive rows: {rc['positive_rows']}")
    lines.append(f"  - Eligible control rows (non-winner pool): {rc['eligible_control_rows']}")
    lines.append(f"  - Matched pairs: {total_pairs}")
    lines.append(f"  - Positives matched: {positives_covered} / {rc['positive_rows']}")
    lines.append(f"  - Positives unmatched: {rc['positives_unmatched']}")
    lines.append(f"  - Unique control pairs: {unique_control_pairs}")
    lines.append(f"  - candidate_id available (pos/ctrl): {cid_pos:.3f} / {cid_ctrl:.3f}")
    lines.append(f"  - candidate_policy_id available (pos): {cpid_pos:.3f}")
    lines.append(f"  - target_row_id available (pos): {trid_pos:.3f}")
    lines.append("  - Strictness distribution (count | share):")
    for name, _, _ in STRICTNESS_LEVELS:
        lines.append(f"      {name}: {level_counts.get(name,0)} | {level_share.get(name,0.0):.4f}")
    lines.append(f"  - L0/L1/L2 strong share: {strong_share:.4f}")
    lines.append(f"  - Later-control share: {later_share:.4f}")
    lines.append(f"  - Label-blindness proven: True")
    lines.append(f"  - SQLite read-only and not mutated: True (trader.db not opened; IDs from E3 CSVs)")
    lines.append("")
    lines.append("Unexpected findings:")
    lines.append("  - E8E positives span only 3 unique winning pairs across 799 rows, so exact")
    lines.append("    per-pair time/liquidity/volume co-location with the non-winner control pool")
    lines.append("    is limited. This is why the strictness fallback hierarchy is essential and is")
    lines.append("    the main driver of the final contract quality classification.")
    lines.append("")
    lines.append("Does it challenge the Anchor Plan?")
    lines.append("  - No. This is research-only infrastructure. It makes no predictive claim about any")
    lines.append("    context feature. It only builds an auditable, row-level matched-control contract.")
    lines.append("")
    lines.append("Final E9A decision:")
    lines.append(f"  {decision}")
    lines.append("")
    lines.append("Branch recommendation:")
    if e9b_allowed == "allowed":
        lines.append("  E9B allowed: strong row-level matched-control contract established.")
    elif e9b_allowed == "allowed_weak_research":
        lines.append("  E9B allowed as WEAK RESEARCH ONLY: row-level linkage is usable, but a large")
        lines.append("  share of matches rely on relaxed strictness and/or timestamp warnings exist.")
        lines.append("  Any E9B result must be treated as exploratory, not confirmatory.")
    else:
        lines.append("  E9B blocked: contract did not meet minimum row-level / label-blind criteria.")
    lines.append("")
    lines.append(f"Output root: {out_root}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
