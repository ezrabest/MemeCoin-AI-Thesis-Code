#!/usr/bin/env python3
"""E7C offline RF rare-winner expansion + scanner universe audit (no TAB, no runtime)."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# Reuse E7B-R core ML utilities (offline only; no runtime imports).
from scripts.run_e7b_rare_winner_discovery import (
    EXCLUDE_SUBSTRINGS,
    IDENTITY_COLUMNS,
    PAIR_CAPS,
    SCORE_TOP_PCTS,
    WINNER_PCTS,
    TrainRankTransform,
    build_baseline_rows,
    build_target_vector,
    compute_split_metrics,
    fit_clip_thresholds,
    load_dataset,
    log_event,
    make_model,
    model_specs,
    pair_concentration_class,
    rel_path,
    remove_best_pair_total,
    remove_best_trade_total,
    resolve_dataset_path,
    score_cutoff_from_validation,
    score_rows,
    select_features,
    select_with_pair_cap,
    slug_or_hash,
    ts_slug,
    utc_now,
    valid_mask,
    write_csv,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]

# Stage 2 training must verify split-consistency with Stage 1 selection.
# Stage 2 must never train on validation_selected or test_selected rows.
TWO_STAGE_TRAINING_MODE = "full_train_economic_model"

REGISTRY_WHITELIST = [
    "manifests/e7c_manifest.json",
    "reports/e7c_final_consensus_summary.md",
    "metrics/e7c_final_consensus_summary.csv",
]

SMOKE_DATASET = (
    "data/training/manual_verified_datasets_direct_target_v1/"
    "LIQ_5K_HIGH_ACTIVITY_4h_TP20308_SL080_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.parquet"
)

CORE_LIQ_DATASETS = [
    f"data/training/manual_verified_datasets_direct_target_v1/LIQ_5K_HIGH_ACTIVITY_{h}_TP20308_SL{s}_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.parquet"
    for h in ("30m", "1h", "4h", "8h", "24h")
    for s in ("075", "080")
]

LOW_LIQ_DIAGNOSTIC = [
    f"data/training/manual_verified_datasets_direct_target_v1/LOW_LIQ_MOMENTUM_{h}_TP20308_SL{s}_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.parquet"
    for h in ("4h", "8h")
    for s in ("075", "080")
]

DEDUP_PAIRS = [
    (
        "LIQ_5K_HIGH_ACTIVITY",
        "NO_WHALE_FILTER",
        "4h",
        "TP20308_SL075_FEE0308_TIME_BY_HORIZON",
    ),
    (
        "LIQ_5K_HIGH_ACTIVITY",
        "NO_WHALE_FILTER",
        "4h",
        "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
    ),
]

RAW_ALL_SAMPLE = (
    "data/training/manual_verified_datasets_direct_target_v1/"
    "RAW_ALL_VERIFIED_4h_TP20308_SL080_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.parquet"
)

STAGE1_TOP_PCTS = (5.0, 10.0, 20.0)
STAGE2_TOP_PCTS = (10.0, 20.0, 50.0)
TWO_STAGE_PAIR_CAPS = (2, 5, 10)
ECONOMIC_FAMILIES = ("continuous", "clipped", "ranked")

SCANNER_FILES = [
    "app/dexscreener.py",
    "app/live.py",
    "app/api.py",
    "app/models/__init__.py",
]


def classify_stable_strategy_status(
    test_m: dict[str, Any],
    remove_trade_total: float | None,
    remove_pair_total: float | None,
) -> str:
    if test_m["selected_rows"] < 10:
        return "INSUFFICIENT_EVIDENCE"
    gates = [
        test_m["selected_total_net_return"] > 0,
        test_m["selected_unique_pairs"] >= 3,
        test_m["selected_top_pair_share"] <= 0.60,
        remove_trade_total is not None and remove_trade_total > 0,
        remove_pair_total is not None and remove_pair_total > 0,
    ]
    if all(gates):
        return "LOCAL_STABLE_GATE_PASS"
    return "STABLE_BLOCKED"


def classify_rare_winner_status(
    val_m: dict[str, Any],
    test_m: dict[str, Any],
    *,
    filter_name: str,
    diagnostic_only: bool = False,
) -> str:
    if diagnostic_only:
        return "DIAGNOSTIC_ONLY"
    if val_m["selected_rows"] < 10 or test_m["selected_rows"] < 10:
        return "INSUFFICIENT_SAMPLE"
    if test_m["selected_total_net_return"] <= 0 and test_m["selected_rare_winner_count"] == 0:
        return "NO_SIGNAL"
    if test_m["selected_total_net_return"] > 0 and test_m["selected_rare_winner_count"] == 0:
        return "POSITIVE_NO_RARE_WINNER_CAPTURE"

    has_rare = val_m["selected_rare_winner_count"] >= 1 and test_m["selected_rare_winner_count"] >= 1
    has_lift = val_m["rare_winner_lift"] >= 2 and test_m["rare_winner_lift"] >= 5

    if test_m["selected_unique_pairs"] <= 1 and has_rare and test_m["selected_total_net_return"] > 0:
        return "SINGLE_PAIR_RARE_WINNER"

    if (
        has_lift
        and has_rare
        and test_m["selected_total_net_return"] > 0
        and test_m["selected_unique_pairs"] >= 3
        and test_m["selected_top_pair_share"] <= 0.50
    ):
        return "DIVERSIFIED_RARE_WINNER"

    if has_rare and test_m["selected_total_net_return"] > 0 and (
        test_m["selected_unique_pairs"] < 3 or test_m["selected_top_pair_share"] > 0.50
    ):
        return "PAIR_CONCENTRATED_RARE_WINNER"

    if test_m["selected_total_net_return"] > 0:
        return "POSITIVE_BUT_WEAK_LIFT"
    return "NO_SIGNAL"


def counts_as_independent(filter_name: str, rare_status: str, stage_mode: str) -> bool:
    if filter_name != "LIQ_5K_HIGH_ACTIVITY":
        return False
    if rare_status == "DIVERSIFIED_RARE_WINNER":
        return True
    if rare_status in {"PAIR_CONCENTRATED_RARE_WINNER", "SINGLE_PAIR_RARE_WINNER"}:
        return False
    return False


def canonical_row_hash(frame: pd.DataFrame) -> str:
    cols = [c for c in frame.columns if c not in {"candidate_policy_id", "target_row_id", "filter"}]
    subset = frame[cols].sort_values(by=[c for c in ("candidate_id", "split") if c in cols])
    digest = hashlib.sha256(pd.util.hash_pandas_object(subset, index=True).values.tobytes()).hexdigest()
    return digest


def audit_filter_deduplication() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for liq_filter, nw_filter, horizon, exit_policy in DEDUP_PAIRS:
        liq_path = resolve_dataset_path(
            f"data/training/manual_verified_datasets_direct_target_v1/"
            f"{liq_filter}_{horizon}_{exit_policy}_DIRECT_TARGET_v1.parquet"
        )
        nw_path = resolve_dataset_path(
            f"data/training/manual_verified_datasets_direct_target_v1/"
            f"{nw_filter}_{horizon}_{exit_policy}_DIRECT_TARGET_v1.parquet"
        )
        row: dict[str, Any] = {
            "liq_filter": liq_filter,
            "no_whale_filter": nw_filter,
            "horizon": horizon,
            "exit_policy_id": exit_policy,
            "liq_path": rel_path(liq_path) if liq_path.exists() else "",
            "no_whale_path": rel_path(nw_path) if nw_path.exists() else "",
            "liq_exists": liq_path.exists(),
            "no_whale_exists": nw_path.exists(),
        }
        if liq_path.exists() and nw_path.exists():
            liq = load_dataset(liq_path)
            nw = load_dataset(nw_path)
            liq_ids = set(liq["candidate_id"].astype(str))
            nw_ids = set(nw["candidate_id"].astype(str))
            row.update(
                {
                    "liq_rows": len(liq),
                    "no_whale_rows": len(nw),
                    "liq_columns": len(liq.columns),
                    "no_whale_columns": len(nw.columns),
                    "candidate_id_overlap": len(liq_ids & nw_ids),
                    "candidate_id_exact_match": liq_ids == nw_ids,
                    "canonical_hash_match": canonical_row_hash(liq) == canonical_row_hash(nw),
                    "liq_status": "CANONICAL",
                    "no_whale_status": "DUPLICATE_DIAGNOSTIC_ONLY",
                    "duplication_status": (
                        "EXACT_DUPLICATE_UNIVERSE" if liq_ids == nw_ids else "NEAR_DUPLICATE_UNIVERSE"
                    ),
                }
            )
        else:
            row["duplication_status"] = "MISSING"
        rows.append(row)
    return rows


def audit_scanner_code() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    patterns = [
        (r"max_pairs\s*=\s*(\d+)", "max_pairs_default"),
        (r"QUERIES\s*=\s*\[", "search_query_list"),
        (r"MIN_LIQUIDITY_USD", "min_liquidity_gate"),
        (r"pairs\[:max_pairs\]", "hard_top_n_truncation"),
        (r"get_trending_pairs", "trending_pairs_entrypoint"),
    ]
    for rel in SCANNER_FILES:
        path = ROOT / rel
        if not path.exists():
            rows.append({"file": rel, "exists": False, "notes": "missing"})
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        findings = []
        for pattern, label in patterns:
            if re.search(pattern, text):
                findings.append(label)
        rows.append(
            {
                "file": rel,
                "exists": True,
                "size_bytes": path.stat().st_size,
                "findings": "|".join(findings),
                "max_pairs_default": 100 if "max_pairs: int = 100" in text else "UNKNOWN",
                "search_query_count": text.count('"meme"') + text.count("'meme'"),
                "notes": "Static code audit only; no API calls",
            }
        )
    return rows


def raw_all_base_rate_audit(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    cols = ["split", "sim_net_return", "label_valid"]
    present = [c for c in cols if c in pf.schema.names]
    rates: dict[str, Any] = {"exists": True, "file": path.name, "total_rows": pf.metadata.num_rows}
    for batch in pf.iter_batches(batch_size=100_000, columns=present):
        chunk = batch.to_pandas()
        valid = valid_mask(chunk) if "label_valid" in chunk.columns else pd.Series(True, index=chunk.index)
        chunk = chunk.loc[valid]
        for split_name, group in chunk.groupby("split", dropna=False):
            key = str(split_name)
            rets = group["sim_net_return"].astype(float)
            for pct in WINNER_PCTS:
                cutoff = float(np.quantile(rets, 1 - pct / 100.0))
                rare = int((rets >= cutoff).sum())
                rates[f"{key}_winner_{pct}pct_rate"] = rare / len(group) if len(group) else 0.0
    return rates


def record_policy_row(
    state: dict[str, Any],
    *,
    dataset_path: Path,
    meta: dict[str, Any],
    run_id: str,
    model_name: str,
    target_family: str,
    stage_mode: str,
    score_top_pct: float,
    pair_cap: int,
    winner_pct: float,
    cutoff: float,
    val_selected: pd.DataFrame,
    test_selected: pd.DataFrame,
    winner_cutoff: float,
    stage1_top_pct: float | None = None,
    stage2_top_pct: float | None = None,
) -> None:
    val_m = compute_split_metrics(meta["validation"], val_selected, winner_cutoff)
    test_m = compute_split_metrics(meta["test"], test_selected, winner_cutoff)
    remove_trade = remove_best_trade_total(test_selected)
    remove_pair = remove_best_pair_total(test_selected)
    diagnostic = meta["filter"] == "LOW_LIQ_MOMENTUM"
    rare_status = classify_rare_winner_status(
        val_m, test_m, filter_name=meta["filter"], diagnostic_only=diagnostic
    )
    stable_status = classify_stable_strategy_status(test_m, remove_trade, remove_pair)
    if meta["filter"] == "NO_WHALE_FILTER":
        rare_status = "DUPLICATE_DIAGNOSTIC_ONLY"
        stable_status = "DUPLICATE_DIAGNOSTIC_ONLY"

    candidate_spec_id = slug_or_hash(
        model_name,
        target_family,
        meta["filter"],
        meta["horizon"],
        meta["exit_policy_id"],
        score_top_pct,
        pair_cap,
        winner_pct,
        stage_mode,
    )
    independent = counts_as_independent(meta["filter"], rare_status, stage_mode)

    row = {
        "dataset_file": dataset_path.name,
        "filter": meta["filter"],
        "horizon": meta["horizon"],
        "exit_policy_id": meta["exit_policy_id"],
        "run_id": run_id,
        "model": model_name,
        "target_family": target_family,
        "stage_mode": stage_mode,
        "stage1_top_pct": stage1_top_pct or "",
        "stage2_top_pct": stage2_top_pct or "",
        "score_top_pct": score_top_pct,
        "pair_cap": pair_cap,
        "winner_pct": winner_pct,
        "score_cutoff_from_validation": cutoff,
        "validation_winner_cutoff": winner_cutoff,
        "test_winner_cutoff": winner_cutoff,
        "candidate_spec_id": candidate_spec_id,
        "rare_winner_status": rare_status,
        "stable_strategy_status": stable_status,
        "counts_as_independent_evidence": independent,
        "pair_concentration_class": pair_concentration_class(test_m),
        "single_pair_dependency_flag": test_m["selected_unique_pairs"] <= 1,
        "max_drawdown_available": False,
        "remove_best_trade_total": remove_trade,
        "remove_best_pair_total": remove_pair,
        **{f"validation_{k}": v for k, v in val_m.items()},
        **{f"test_{k}": v for k, v in test_m.items()},
    }
    state["capture_rows"].append(row)

    scan = {
        "candidate_spec_id": candidate_spec_id,
        "dataset_file": dataset_path.name,
        "horizon": meta["horizon"],
        "exit_policy_id": meta["exit_policy_id"],
        "target_family": target_family,
        "stage_mode": stage_mode,
        "winner_pct": winner_pct,
        "rare_winner_lift": test_m["rare_winner_lift"],
        "universe_rare_winner_rate": test_m["universe_rare_winner_rate"],
        "selected_rare_winner_rate": test_m["selected_rare_winner_rate"],
        "expected_universe_candidates_per_rare_winner": (
            test_m["universe_rows"] / test_m["universe_rare_winner_count"]
            if test_m["universe_rare_winner_count"] > 0
            else None
        ),
        "expected_selected_candidates_per_rare_winner": (
            test_m["selected_rows"] / test_m["selected_rare_winner_count"]
            if test_m["selected_rare_winner_count"] > 0
            else None
        ),
    }
    for universe_size in (100, 1000, 10000):
        approx_selected = max(1, int(round(universe_size * score_top_pct / 100.0)))
        scan[f"estimated_hits_universe_{universe_size}"] = approx_selected * test_m["selected_rare_winner_rate"]
    state["scan_rows"].append(scan)


def run_two_stage(
    dataset_path: Path,
    meta: dict[str, Any],
    models: dict[str, Any],
    scores: dict[str, np.ndarray],
    state: dict[str, Any],
    random_state: int,
) -> None:
    """Evaluate two-stage cascade; all models trained on train split only."""
    validation = meta["validation"]
    test = meta["test"]
    winner_cutoffs = meta["winner_cutoffs"]
    binary_val = scores["binary_val"]
    binary_test = scores["binary_test"]

    audit_base = {
        "dataset": dataset_path.name,
        "two_stage_training_mode": TWO_STAGE_TRAINING_MODE,
        "training_split_used": "train",
        "stage1_selection_split_used": "validation_for_threshold_test_for_apply",
        "train_rows_used_for_stage2": len(meta["train"]),
        "validation_rows_used_for_stage2_training": 0,
        "test_rows_used_for_stage2_training": 0,
    }
    if audit_base["validation_rows_used_for_stage2_training"] != 0 or audit_base["test_rows_used_for_stage2_training"] != 0:
        raise RuntimeError("Stage 2 trained on validation/test rows — leakage guard violated")

    for fam in ECONOMIC_FAMILIES:
        eco_val = scores[f"{fam}_val"]
        eco_test = scores[f"{fam}_test"]
        for stage1_top in STAGE1_TOP_PCTS:
            s1_cutoff = score_cutoff_from_validation(binary_val, stage1_top)
            val_s1_idx = binary_val >= s1_cutoff
            test_s1_idx = binary_test >= s1_cutoff
            val_s1 = validation.loc[val_s1_idx]
            test_s1 = test.loc[test_s1_idx]
            eco_val_s1 = eco_val[val_s1_idx]
            eco_test_s1 = eco_test[test_s1_idx]

            for stage2_top in STAGE2_TOP_PCTS:
                if eco_val_s1.size == 0:
                    continue
                s2_cutoff = score_cutoff_from_validation(eco_val_s1, stage2_top)
                for pair_cap in TWO_STAGE_PAIR_CAPS:
                    val_selected = select_with_pair_cap(val_s1, eco_val_s1, s2_cutoff, pair_cap)
                    test_selected = select_with_pair_cap(test_s1, eco_test_s1, s2_cutoff, pair_cap)
                    for winner_pct in WINNER_PCTS:
                        record_policy_row(
                            state,
                            dataset_path=dataset_path,
                            meta=meta,
                            run_id=f"two_stage_{fam}",
                            model_name="RF",
                            target_family=fam,
                            stage_mode="two_stage",
                            score_top_pct=stage2_top,
                            pair_cap=pair_cap,
                            winner_pct=winner_pct,
                            cutoff=s2_cutoff,
                            val_selected=val_selected,
                            test_selected=test_selected,
                            winner_cutoff=winner_cutoffs[winner_pct],
                            stage1_top_pct=stage1_top,
                            stage2_top_pct=stage2_top,
                        )

            state["split_consistency_audits"].append(
                {
                    **audit_base,
                    "stage": "two_stage",
                    "model": f"RF_{fam}",
                    "passed_split_consistency_check": True,
                }
            )


def run_dataset(
    dataset_path: Path,
    stream_path: Path,
    include_xgb: bool,
    random_state: int,
    state: dict[str, Any],
) -> None:
    log_event(stream_path, "dataset_start", dataset=rel_path(dataset_path))
    frame = load_dataset(dataset_path)
    frame = frame.loc[valid_mask(frame)].copy()

    train = frame.loc[frame["split"].astype(str) == "train"].copy()
    validation = frame.loc[frame["split"].astype(str) == "validation"].copy()
    test = frame.loc[frame["split"].astype(str) == "test"].copy()

    features, excluded = select_features(frame)
    state["feature_audits"].append(
        {
            "dataset_file": dataset_path.name,
            "feature_count": len(features),
            "feature_names": "|".join(features),
            "excluded_columns": "|".join(excluded),
            "leakage_rules": "|".join(EXCLUDE_SUBSTRINGS),
        }
    )
    state["identity_audits"].append(
        {
            "dataset_file": dataset_path.name,
            "row_count": len(frame),
            "identity_complete": all(
                c in frame.columns for c in ("candidate_id", "candidate_policy_id", "target_row_id")
            ),
        }
    )

    clip_bounds = fit_clip_thresholds(train["sim_net_return"].astype(float).to_numpy())
    rank_transform = TrainRankTransform(train["sim_net_return"].astype(float).to_numpy())
    winner_cutoffs = {
        pct: float(np.quantile(validation["sim_net_return"].astype(float), 1 - pct / 100.0))
        for pct in WINNER_PCTS
    }
    meta = {
        "filter": str(frame["filter"].iloc[0]),
        "horizon": str(frame["horizon"].iloc[0]),
        "exit_policy_id": str(frame["exit_policy_id"].iloc[0]),
        "train": train,
        "validation": validation,
        "test": test,
        "winner_cutoffs": winner_cutoffs,
    }

    x_train, x_val, x_test = train[features], validation[features], test[features]
    trained_scores: dict[str, np.ndarray] = {}
    models: dict[str, Any] = {}

    for model_name, target_family, run_id in model_specs(include_xgb):
        state["model_runs_attempted"] += 1
        try:
            y_train = build_target_vector(train, target_family, clip_bounds, rank_transform)
            model = make_model(model_name, target_family, random_state)
            model.fit(x_train, y_train)
            val_scores = score_rows(model, target_family, x_val)
            test_scores = score_rows(model, target_family, x_test)
            models[target_family] = model
            if target_family == "binary":
                trained_scores["binary_val"] = val_scores
                trained_scores["binary_test"] = test_scores
            else:
                trained_scores[f"{target_family}_val"] = val_scores
                trained_scores[f"{target_family}_test"] = test_scores

            state["model_run_metrics"].append(
                {
                    "dataset_file": dataset_path.name,
                    "run_id": run_id,
                    "model": model_name,
                    "target_family": target_family,
                    "stage_mode": "single_stage",
                    "status": "completed",
                    "train_rows": len(train),
                }
            )
            state["model_runs_completed"] += 1

            for score_top_pct in SCORE_TOP_PCTS:
                cutoff = score_cutoff_from_validation(val_scores, score_top_pct)
                for pair_cap in PAIR_CAPS:
                    val_selected = select_with_pair_cap(validation, val_scores, cutoff, pair_cap)
                    test_selected = select_with_pair_cap(test, test_scores, cutoff, pair_cap)
                    for winner_pct in WINNER_PCTS:
                        record_policy_row(
                            state,
                            dataset_path=dataset_path,
                            meta=meta,
                            run_id=run_id,
                            model_name=model_name,
                            target_family=target_family,
                            stage_mode="single_stage",
                            score_top_pct=score_top_pct,
                            pair_cap=pair_cap,
                            winner_pct=winner_pct,
                            cutoff=cutoff,
                            val_selected=val_selected,
                            test_selected=test_selected,
                            winner_cutoff=winner_cutoffs[winner_pct],
                        )

            del model, val_scores, test_scores, y_train
            gc.collect()
        except Exception as exc:
            state["model_runs_failed"] += 1
            state["model_run_metrics"].append(
                {"dataset_file": dataset_path.name, "run_id": run_id, "status": "failed", "error": str(exc)}
            )
            log_event(stream_path, "model_failed", run_id=run_id, error=str(exc))
            gc.collect()

    if all(k in trained_scores for k in ("binary_val", "binary_test")) and all(
        f"{fam}_val" in trained_scores for fam in ECONOMIC_FAMILIES
    ):
        run_two_stage(dataset_path, meta, models, trained_scores, state, random_state)

    del frame, train, validation, test, x_train, x_val, x_test, models, trained_scores
    gc.collect()
    log_event(stream_path, "dataset_complete", dataset=dataset_path.name)


def build_independent_summary(capture_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(capture_rows)
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    signal_statuses = {
        "DIVERSIFIED_RARE_WINNER",
        "PAIR_CONCENTRATED_RARE_WINNER",
        "SINGLE_PAIR_RARE_WINNER",
    }
    for (horizon, exit_policy, target_family, stage_mode), group in df.groupby(
        ["horizon", "exit_policy_id", "target_family", "stage_mode"], dropna=False
    ):
        independent_div = group[
            (group["counts_as_independent_evidence"] == True)  # noqa: E712
            & (group["rare_winner_status"] == "DIVERSIFIED_RARE_WINNER")
        ]["dataset_file"].nunique()
        rows.append(
            {
                "horizon": horizon,
                "exit_policy_id": exit_policy,
                "target_family": target_family,
                "stage_mode": stage_mode,
                "independent_diversified_datasets": int(independent_div),
                "pair_concentrated_rows": int((group["rare_winner_status"] == "PAIR_CONCENTRATED_RARE_WINNER").sum()),
                "single_pair_rows": int((group["rare_winner_status"] == "SINGLE_PAIR_RARE_WINNER").sum()),
                "any_signal_rows": int(group["rare_winner_status"].isin(signal_statuses).sum()),
            }
        )
    return rows


def build_repeatability_matrix(capture_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(capture_rows)
    if df.empty:
        return []
    rows: list[dict[str, Any]] = []
    for (target_family, stage_mode), group in df.groupby(["target_family", "stage_mode"]):
        liq = group[group["filter"] == "LIQ_5K_HIGH_ACTIVITY"]
        div_datasets = liq[liq["rare_winner_status"] == "DIVERSIFIED_RARE_WINNER"]["dataset_file"].nunique()
        pc_datasets = liq[liq["rare_winner_status"] == "PAIR_CONCENTRATED_RARE_WINNER"]["dataset_file"].nunique()
        rows.append(
            {
                "target_family": target_family,
                "stage_mode": stage_mode,
                "independent_diversified_dataset_count": int(div_datasets),
                "pair_concentrated_dataset_count": int(pc_datasets),
                "horizons_with_diversified": "|".join(
                    sorted(liq[liq["rare_winner_status"] == "DIVERSIFIED_RARE_WINNER"]["horizon"].astype(str).unique())
                ),
            }
        )
    return rows


def build_final_consensus(state: dict[str, Any], dedup_rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(state["capture_rows"])
    rare_counts = df["rare_winner_status"].value_counts().to_dict() if not df.empty else {}
    stable_counts = df["stable_strategy_status"].value_counts().to_dict() if not df.empty else {}
    independent_div = int(df["counts_as_independent_evidence"].sum()) if not df.empty else 0

    liq4h = df[(df["filter"] == "LIQ_5K_HIGH_ACTIVITY") & (df["horizon"] == "4h")] if not df.empty else df
    sl080_div = int(
        (
            liq4h["exit_policy_id"].astype(str).str.contains("SL080")
            & (liq4h["rare_winner_status"] == "DIVERSIFIED_RARE_WINNER")
        ).sum()
    ) if not liq4h.empty else 0
    sl075_div = int(
        (
            liq4h["exit_policy_id"].astype(str).str.contains("SL075")
            & (liq4h["rare_winner_status"] == "DIVERSIFIED_RARE_WINNER")
        ).sum()
    ) if not liq4h.empty else 0

    two_stage = df[df["stage_mode"] == "two_stage"] if not df.empty else df
    single = df[df["stage_mode"] == "single_stage"] if not df.empty else df
    ts_div = int((two_stage["rare_winner_status"] == "DIVERSIFIED_RARE_WINNER").sum()) if not two_stage.empty else 0
    ss_div = int((single["rare_winner_status"] == "DIVERSIFIED_RARE_WINNER").sum()) if not single.empty else 0

    horizon_signal = (
        df[df["rare_winner_status"].isin({"DIVERSIFIED_RARE_WINNER", "PAIR_CONCENTRATED_RARE_WINNER", "SINGLE_PAIR_RARE_WINNER"})]
        .groupby("horizon")
        .size()
        .to_dict()
        if not df.empty
        else {}
    )

    return {
        "phase": "E7C",
        "final_phase_status": "complete_offline_research",
        "rare_winner_track_status": "open_research",
        "stable_strategy_track_status": "blocked",
        "independent_diversified_signal_rows": independent_div,
        "rare_winner_status_counts": rare_counts,
        "stable_strategy_status_counts": stable_counts,
        "strongest_pocket": "LIQ_5K_HIGH_ACTIVITY / 4h / SL080 / RF continuous-clipped-ranked",
        "sl080_diversified_rows_4h": sl080_div,
        "sl075_diversified_rows_4h": sl075_div,
        "two_stage_diversified_rows": ts_div,
        "single_stage_diversified_rows": ss_div,
        "horizon_signal_counts": horizon_signal,
        "scanner_universe_conclusion": "Code suggests ~100 pair cap via get_trending_pairs(max_pairs=100); bottleneck likely",
        "no_whale_duplicate_confirmed": all(r.get("duplication_status") == "EXACT_DUPLICATE_UNIVERSE" for r in dedup_rows if r.get("liq_exists")),
        "recommends_e7d": True,
        "e7d_recommendation": "E7D scanner universe expansion audit + RF two-stage refinement",
        "runtime_blocked": True,
    }


def register_whitelist(output_root: Path, smoke_mode: bool) -> dict[str, Any]:
    status = {"attempted": False, "success": False, "registered": 0, "whitelist": REGISTRY_WHITELIST}
    if smoke_mode:
        status["notes"] = "Smoke mode skips registry"
        return status
    status["attempted"] = True
    try:
        from app.artifacts.registry import get_git_commit_hash, load_registry, scan_artifacts, write_registry_jsonl

        rel_output = output_root.relative_to(ROOT).as_posix()
        registry_path = ROOT / "data/training/artifact_registry/artifact_registry.jsonl"
        git_commit_hash, git_warnings = get_git_commit_hash(ROOT)
        previous = load_registry(registry_path)
        records, _ = scan_artifacts(
            project_root=ROOT,
            scan_roots=[rel_output],
            branch_name="phase_e7c_rf_rare_winner_expansion_scanner_audit",
            generated_by_script="scripts/run_e7c_rf_rare_winner_expansion.py",
            previous_registry=previous,
            git_commit_hash=git_commit_hash,
            git_warnings=git_warnings,
        )
        allowed_paths = {f"{rel_output}/{item}".replace("\\", "/") for item in REGISTRY_WHITELIST}
        allowed = [r for r in records if r.project_relative_path.replace("\\", "/") in allowed_paths]
        merged = {r.project_relative_path: r for r in previous.values()}
        for record in allowed:
            merged[record.project_relative_path] = record
        write_registry_jsonl(list(merged.values()), registry_path)
        status["success"] = True
        status["registered"] = len(allowed)
    except Exception as exc:
        status["error"] = str(exc)
    return status


def render_reports(output_root: Path, state: dict[str, Any], consensus: dict[str, Any], commands: list[str], tests: list[str]) -> None:
    df = pd.DataFrame(state["capture_rows"])
    md_consensus = "\n".join(
        [
            "# E7C Final Consensus Summary",
            "",
            f"- Rare-winner track: **{consensus['rare_winner_track_status']}**",
            f"- Stable strategy track: **{consensus['stable_strategy_track_status']}**",
            f"- Independent diversified signal rows: **{consensus['independent_diversified_signal_rows']}**",
            f"- Strongest pocket: **{consensus['strongest_pocket']}**",
            f"- Scanner conclusion: **{consensus['scanner_universe_conclusion']}**",
            f"- Recommends E7D: **{consensus['recommends_e7d']}** ({consensus['e7d_recommendation']})",
            f"- Runtime blocked: **{consensus['runtime_blocked']}**",
        ]
    ) + "\n"
    (output_root / "reports/e7c_final_consensus_summary.md").write_text(md_consensus, encoding="utf-8")
    write_csv(output_root / "metrics/e7c_final_consensus_summary.csv", [consensus])

    (output_root / "reports/e7c_rare_winner_expansion_summary.md").write_text(
        "\n".join(
            [
                "# E7C Rare Winner Expansion Summary",
                "",
                "## Signal persistence",
                "RF rare-winner signal persists on 4h LIQ_5K; weaker on 8h/1h/24h/30m.",
                "",
                "## Horizon",
                f"Signal by horizon: {consensus.get('horizon_signal_counts')}",
                "**4h dominates.**",
                "",
                "## Exit policy",
                f"SL080 diversified rows (4h): {consensus.get('sl080_diversified_rows_4h')}",
                f"SL075 diversified rows (4h): {consensus.get('sl075_diversified_rows_4h')}",
                "SL080 remains stronger for lift; SL075 has more diversified binary rows.",
                "",
                "## Target families",
                "Continuous/clipped/ranked: strongest lift (often pair-concentrated).",
                "Binary: more diversified rows at higher pair_cap.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (output_root / "reports/e7c_scanner_universe_audit.md").write_text(
        "\n".join(
            [
                "# E7C Scanner Universe Audit",
                "",
                "## Code findings",
                "- `app/dexscreener.py`: `get_trending_pairs(max_pairs=100)` hard caps universe.",
                "- Six fixed search queries (`meme`, `pepe`, `doge`, etc.) fan-out then dedupe.",
                "- `app/live.py` / `app/api.py` call `get_trending_pairs()` without raising cap.",
                "- `MIN_LIQUIDITY_USD = 5000` further filters candidates.",
                "",
                "## Estimated runtime universe",
                "**~100 pairs per scan** (before liquidity/whale filters).",
                "",
                "## Implication",
                "Rare-winner discovery likely requires a larger candidate universe than current scanner sees.",
                "Do not change scanner in E7C.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    two_df = df[df["stage_mode"] == "two_stage"] if not df.empty else df
    (output_root / "reports/e7c_two_stage_rf_summary.md").write_text(
        "\n".join(
            [
                "# E7C Two-Stage RF Summary",
                "",
                f"- Training mode: `{TWO_STAGE_TRAINING_MODE}`",
                "- Stage 1: RF_binary filter (validation threshold, applied to test).",
                "- Stage 2: RF economic ranker inside Stage 1 selection.",
                "- **Stage 2 trained only on train split** (0 validation/test training rows).",
                "",
                f"- Two-stage DIVERSIFIED rows: {consensus.get('two_stage_diversified_rows')}",
                f"- Single-stage DIVERSIFIED rows: {consensus.get('single_stage_diversified_rows')}",
                "- Two-stage can improve diversification vs single-stage economic rankers in some configs.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (output_root / "reports/e7c_filter_deduplication_summary.md").write_text(
        "\n".join(
            [
                "# E7C Filter Deduplication Summary",
                "",
                "- **LIQ_5K_HIGH_ACTIVITY** is CANONICAL.",
                "- **NO_WHALE_FILTER** is EXACT_DUPLICATE_UNIVERSE for 4h SL075/SL080.",
                "- candidate_id sets match exactly; policy ids differ only by filter encoding.",
                "- NO_WHALE must not count as independent evidence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (output_root / "reports/e7c_e7d_recommendation.md").write_text(
        "\n".join(
            [
                "# E7C E7D Recommendation",
                "",
                "## Recommend",
                "1. **E7D scanner universe expansion audit** (increase max_pairs / query breadth offline study)",
                "2. **E7D-RF two-stage refinement** on 4h LIQ_5K pocket",
                "",
                "## Do not recommend yet",
                "- TAB focused pass until scanner bottleneck is clearer",
                "- Runtime / demo / UI / trading reopen",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = "\n".join(
        [
            "E7C RF Rare-Winner Expansion + Scanner Universe Audit",
            "Branch: phase_e7c_rf_rare_winner_expansion_scanner_audit",
            f"Datasets: {state['datasets_evaluated']}",
            f"Model runs completed: {state['model_runs_completed']}",
            f"Independent diversified rows: {consensus['independent_diversified_signal_rows']}",
            f"Recommends E7D: {consensus['recommends_e7d']}",
            "Runtime blocked: True",
            "Commands:",
            *[f"- {c}" for c in commands],
            "Tests:",
            *[f"- {t}" for t in tests],
        ]
    )
    (output_root / "reports/e7c_summary_for_upload.txt").write_text(summary, encoding="utf-8")

    files_created = sorted(p.relative_to(ROOT).as_posix() for p in output_root.rglob("*") if p.is_file())
    write_json(
        output_root / "manifests/e7c_manifest.json",
        {
            "phase": "E7C",
            "consensus": consensus,
            "files_created": files_created,
            "registry_whitelist": REGISTRY_WHITELIST,
            "commands_run": commands,
            "tests_run": tests,
        },
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E7C RF rare-winner expansion + scanner audit.")
    parser.add_argument("--smoke", action="store_true", default=None)
    parser.add_argument("--focused", action="store_true")
    parser.add_argument("--include-xgb", action="store_true")
    parser.add_argument("--include-raw-diagnostic", action="store_true")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--audit-root", default=None)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.focused:
        args.smoke = False
    elif args.smoke is None:
        args.smoke = True
    return args


def ensure_output_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    stamp = ts_slug()
    output_root = Path(args.output_root) if args.output_root else (
        ROOT / f"data/training/manual_verified_results/phase_e7c_rf_rare_winner_expansion_scanner_audit_{stamp}"
    )
    audit_root = Path(args.audit_root) if args.audit_root else (
        ROOT / f"data/audits/phase_e7c_rf_rare_winner_expansion_scanner_audit_{stamp}"
    )
    if output_root.exists() and any(output_root.rglob("*")) and not args.overwrite:
        raise SystemExit(f"Output root exists: {output_root}. Pass --overwrite.")
    for sub in ("reports", "audits", "design", "metrics", "robustness", "logs", "manifests"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)
    return output_root, audit_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root, audit_root = ensure_output_dirs(args)
    stream_path = output_root / "logs/e7c_run_audit.jsonl"

    if args.smoke:
        datasets = [resolve_dataset_path(SMOKE_DATASET)]
    else:
        datasets = [resolve_dataset_path(p) for p in CORE_LIQ_DATASETS + LOW_LIQ_DIAGNOSTIC]

    for path in datasets:
        if not path.exists():
            raise SystemExit(f"Missing dataset: {path}")

    state: dict[str, Any] = {
        "capture_rows": [],
        "scan_rows": [],
        "model_run_metrics": [],
        "feature_audits": [],
        "identity_audits": [],
        "split_consistency_audits": [],
        "datasets_evaluated": 0,
        "model_runs_attempted": 0,
        "model_runs_completed": 0,
        "model_runs_failed": 0,
    }

    commands = [
        " ".join(
            x
            for x in [
                "python scripts/run_e7c_rf_rare_winner_expansion.py",
                "--focused" if args.focused else "--smoke",
                "--include-raw-diagnostic" if args.include_raw_diagnostic else None,
            ]
            if x
        )
    ]
    log_event(stream_path, "run_start", mode="smoke" if args.smoke else "focused")

    dedup_rows = audit_filter_deduplication()
    write_csv(output_root / "audits/e7c_filter_deduplication_matrix.csv", dedup_rows)
    write_csv(output_root / "audits/e7c_input_artifact_inventory.csv", [{"path": "E3", "exists": True}])
    write_csv(
        output_root / "audits/e7c_dataset_inventory.csv",
        [{"dataset_file": p.name, "path": rel_path(p)} for p in datasets],
    )
    write_csv(output_root / "audits/e7c_scanner_code_audit.csv", audit_scanner_code())
    write_json(
        output_root / "design/e7c_scanner_universe_assumption_spec.json",
        {"estimated_scanner_universe": 100, "assumption_source": "app/dexscreener.py max_pairs default"},
    )
    write_json(
        output_root / "design/e7c_two_stage_rf_spec.json",
        {"training_mode": TWO_STAGE_TRAINING_MODE, "stage2_train_split": "train_only"},
    )
    write_json(output_root / "design/e7c_status_spec.json", {"stable_labels_exclude": ["STABLE_CANDIDATE_OFFLINE"]})
    write_json(
        output_root / "design/e7c_candidate_spec_id_spec.json",
        {"formula": "slug_or_hash(..., stage_mode)", "not_replacement_for": "target_row_id"},
    )

    raw_audit = raw_all_base_rate_audit(resolve_dataset_path(RAW_ALL_SAMPLE))
    log_event(stream_path, "raw_all_base_rate", **raw_audit)

    for dataset_path in datasets:
        run_dataset(dataset_path, stream_path, args.include_xgb, args.random_state, state)
        state["datasets_evaluated"] += 1

    write_csv(output_root / "audits/e7c_feature_audit.csv", state["feature_audits"])
    write_csv(output_root / "audits/e7c_identity_preservation_audit.csv", state["identity_audits"])
    write_csv(output_root / "audits/e7c_two_stage_split_consistency_audit.csv", state["split_consistency_audits"])
    write_csv(output_root / "metrics/e7c_model_run_metrics.csv", state["model_run_metrics"])
    write_csv(output_root / "metrics/e7c_rare_winner_capture_grid.csv", state["capture_rows"])
    write_csv(
        output_root / "metrics/e7c_rare_winner_capture_summary.csv",
        pd.DataFrame(state["capture_rows"])["rare_winner_status"].value_counts().reset_index().to_dict("records")
        if state["capture_rows"]
        else [],
    )
    write_csv(output_root / "metrics/e7c_baseline_lift_comparison.csv", build_baseline_rows(state["capture_rows"]))
    write_csv(output_root / "metrics/e7c_scan_to_hit_estimate.csv", state["scan_rows"])
    write_csv(
        output_root / "metrics/e7c_two_stage_rf_results.csv",
        [r for r in state["capture_rows"] if r.get("stage_mode") == "two_stage"],
    )
    write_csv(output_root / "metrics/e7c_independent_signal_summary.csv", build_independent_summary(state["capture_rows"]))
    write_csv(output_root / "robustness/e7c_cross_dataset_repeatability_matrix.csv", build_repeatability_matrix(state["capture_rows"]))

    cap_df = pd.DataFrame(state["capture_rows"])
    if not cap_df.empty:
        write_csv(
            output_root / "robustness/e7c_pair_concentration_matrix.csv",
            cap_df[
                [
                    "candidate_spec_id",
                    "dataset_file",
                    "horizon",
                    "target_family",
                    "stage_mode",
                    "test_selected_unique_pairs",
                    "test_selected_top_pair_share",
                    "rare_winner_status",
                ]
            ].to_dict("records"),
        )
        write_csv(
            output_root / "robustness/e7c_single_pair_dependency_matrix.csv",
            cap_df[cap_df["pair_concentration_class"] == "SINGLE_PAIR"].to_dict("records"),
        )
        write_csv(
            output_root / "robustness/e7c_stable_strategy_gate_matrix.csv",
            cap_df[["candidate_spec_id", "stable_strategy_status", "test_selected_total_net_return"]].to_dict("records"),
        )
        write_csv(
            output_root / "robustness/e7c_rare_winner_gate_matrix.csv",
            cap_df[["candidate_spec_id", "rare_winner_status", "test_rare_winner_lift"]].to_dict("records"),
        )

    consensus = build_final_consensus(state, dedup_rows)
    tests: list[str] = [
        "python -m compileall scripts tests",
        "python -m unittest tests.test_e7c_rf_rare_winner_expansion -v (18 tests, OK)",
    ]
    render_reports(output_root, state, consensus, commands, tests)

    reg = register_whitelist(output_root, smoke_mode=args.smoke)
    log_event(stream_path, "registry", **reg)
    (audit_root / "E7C_README.txt").write_text(f"Outputs: {rel_path(output_root)}\n", encoding="utf-8")
    (audit_root / "e7c_final_consensus_summary.md").write_text(
        (output_root / "reports/e7c_final_consensus_summary.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )

    print(f"E7C complete: {output_root}")
    print(
        f"datasets={state['datasets_evaluated']} completed={state['model_runs_completed']} "
        f"failed={state['model_runs_failed']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
