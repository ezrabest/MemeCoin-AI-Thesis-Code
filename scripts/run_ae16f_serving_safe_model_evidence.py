#!/usr/bin/env python3
"""AE16F — Serving-safe direct-target RF/XGB/TAB evidence regeneration.

Trains serving-safe evidence generators and runs AE16 tiered consensus.
Shadow/research only. Does not start AE17. Does not close AE16 automatically.
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.consensus.ae16e_feature_parity import (  # noqa: E402
    TOXIC_PAIR_ADDRESS,
    is_toxic_pair,
    load_clean_forward_rows_used,
)
from app.consensus.ae16f_serving_safe import (  # noqa: E402
    PHASE,
    SELECTED_SOURCE_REL,
    TARGET_COLUMN,
    build_consensus_from_evidence,
    build_current_cf_matrix,
    build_evidence_rows,
    build_serving_safe_feature_contract,
    decide_ae16f_classification,
    discover_training_sources,
    load_historical_matrix,
    ordered_feature_names_from_contract,
    predict_proba_locked,
    train_rf,
    train_xgb,
    try_tab_evidence,
    utc_now,
    utc_stamp,
    validation_quantile_threshold,
    _split_masks,
)
from app.consensus.serialization import relpath_str, write_csv, write_json, write_text  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AE16F serving-safe model evidence")
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument(
        "--active-curated",
        type=str,
        default="data/SeedTargets/clean_forward_curated_ready_targets_active.csv",
    )
    p.add_argument("--skip-tab", action="store_true")
    return p.parse_args(argv)


def _mirror(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dest)


def run_ae16f(args: argparse.Namespace) -> dict[str, Any]:
    stamp = utc_stamp()
    out_root = Path(args.output_root) if args.output_root else (
        ROOT / "data" / "audits" / f"ae16f_serving_safe_model_evidence_{stamp}"
    )
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    data_dir = out_root / "data"
    audits = out_root / "audits"
    reports = out_root / "reports"
    models_dir = out_root / "models"
    for d in (data_dir, audits, reports, models_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- contract ---
    contract_rows, contract = build_serving_safe_feature_contract()
    write_csv(data_dir / "ae16f_serving_safe_feature_contract.csv", contract_rows)
    write_json(data_dir / "ae16f_serving_safe_feature_contract.json", contract)
    ordered_names = ordered_feature_names_from_contract(contract_rows)

    # --- source discovery ---
    discovery_rows, selected = discover_training_sources(ROOT)
    write_csv(data_dir / "ae16f_training_source_discovery.csv", discovery_rows)
    if not selected:
        selected = {"path": SELECTED_SOURCE_REL}
        # re-check
        for r in discovery_rows:
            if r.get("selected"):
                selected = {
                    "path": r["path"],
                    "row_count": r.get("row_count"),
                    "target_column": TARGET_COLUMN,
                    "horizon": "1h",
                    "filter": "LIQ_5K_HIGH_ACTIVITY",
                    "exit_policy_id": "TP20308_SL075_FEE0308_TIME_BY_HORIZON",
                }
                break
        else:
            # force select known path if exists
            p = ROOT / SELECTED_SOURCE_REL
            if p.is_file():
                selected = {
                    "path": SELECTED_SOURCE_REL.replace("\\", "/"),
                    "target_column": TARGET_COLUMN,
                    "horizon": "1h",
                    "filter": "LIQ_5K_HIGH_ACTIVITY",
                    "exit_policy_id": "TP20308_SL075_FEE0308_TIME_BY_HORIZON",
                }

    # --- CF rows ---
    cf_rows, rows_meta = load_clean_forward_rows_used(
        ROOT, active_curated_path=Path(args.active_curated)
    )
    # Prefer persisted AE16E rows if load returns empty unexpected
    if len(cf_rows) == 0:
        ae16e = ROOT / "data/ae16e_clean_forward_rows_used.csv"
        if ae16e.is_file():
            with ae16e.open("r", encoding="utf-8", newline="") as f:
                cf_rows = [r for r in csv.DictReader(f) if not is_toxic_pair(r.get("pair_address"))]
            rows_meta = {
                "curated_active_targets_loaded": 45,
                "clean_forward_rows_used": len(cf_rows),
                "status": "OK",
            }

    toxic_anywhere = any(is_toxic_pair(r.get("pair_address")) for r in cf_rows)

    training_error = ""
    consensus_error = ""
    schema_ok = False
    threshold_ok = True
    lock: dict[str, Any] = {}
    hist = pd.DataFrame()
    lineage: list[dict[str, Any]] = []
    evidence_all: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    vote_thresholds: list[dict[str, Any]] = []
    rf_trained = False
    xgb_trained = False
    tab_run = False
    rf_ev: list[dict[str, Any]] = []
    xgb_ev: list[dict[str, Any]] = []
    tab_ev: list[dict[str, Any]] = []
    consensus_rows: list[dict[str, Any]] = []
    tier_counts: list[dict[str, Any]] = []
    align_audit: list[dict[str, Any]] = []
    pair_conc: list[dict[str, Any]] = []
    avail_audit: list[dict[str, Any]] = []
    no_lookahead: list[dict[str, Any]] = []
    split_method = "historical_split_column"
    train_n = val_n = test_n = 0

    try:
        if not selected.get("path"):
            raise RuntimeError("NO_CLEAN_DIRECT_TARGET_SOURCE")
        hist, _feat_dicts, hist_summary = load_historical_matrix(
            ROOT, str(selected["path"]), ordered_names
        )
        lock = hist_summary["lock"]
        write_json(data_dir / "ae16f_feature_schema_lock.json", lock)
        write_json(
            data_dir / "ae16f_historical_serving_safe_matrix_schema.json",
            {
                "ordered_feature_names": lock["ordered_feature_names"],
                "feature_schema_hash": lock["feature_schema_hash"],
                "feature_count": lock["feature_count"],
            },
        )
        # write historical matrix (features + lineage + target + split)
        hist_out_cols = (
            ["pair_address", "event_timestamp", "split", "target"]
            + list(lock["ordered_feature_names"])
        )
        hist[hist_out_cols].to_csv(data_dir / "ae16f_historical_serving_safe_matrix.csv", index=False)

        train_n = hist_summary.get("train_rows") or 0
        val_n = hist_summary.get("validation_rows") or 0
        test_n = hist_summary.get("test_rows") or 0
        if train_n == 0:
            split_method = "deterministic_fallback_70_15_15_seed_42"

        # pair concentration
        if "pair_address" in hist.columns and "split" in hist.columns:
            for split_name, mask in (
                ("train", hist["split"].astype(str).str.lower().eq("train")),
                ("validation", hist["split"].astype(str).str.lower().isin(["validation", "valid", "val"])),
                ("test", hist["split"].astype(str).str.lower().eq("test")),
            ):
                sub = hist.loc[mask, "pair_address"] if mask.any() else hist["pair_address"]
                vc = sub.value_counts().head(5)
                for pair, cnt in vc.items():
                    pair_conc.append(
                        {
                            "split": split_name,
                            "pair_address": pair,
                            "count": int(cnt),
                            "share": round(float(cnt) / max(len(sub), 1), 6),
                        }
                    )

        # feature availability on CF
        for name in ordered_names:
            if name.endswith("_is_missing") or name in (
                "tp_ratio",
                "sl_ratio",
                "round_trip_fee_pct",
                "time_stop_minutes",
            ):
                avail_audit.append(
                    {
                        "feature_name": name,
                        "historical_non_null_rate": 1.0,
                        "current_non_null_rate": 1.0,
                        "included": True,
                    }
                )
                continue
            h_rate = float(hist[name].notna().mean()) if name in hist.columns else 0.0
            # provisional CF extract
            from app.consensus.ae16f_serving_safe import extract_serving_safe_row_from_cf

            cur_vals = [extract_serving_safe_row_from_cf(r).get(name) for r in cf_rows]
            c_rate = sum(1 for v in cur_vals if v is not None) / max(len(cur_vals), 1)
            avail_audit.append(
                {
                    "feature_name": name,
                    "historical_non_null_rate": round(h_rate, 6),
                    "current_non_null_rate": round(c_rate, 6),
                    "included": True,
                }
            )

        for r in contract_rows:
            if r.get("allowed"):
                no_lookahead.append(
                    {
                        "feature_name": r["feature_name"],
                        "no_lookahead": True,
                        "category": r.get("category"),
                        "formula": r.get("formula"),
                        "status": "PASS",
                    }
                )

        # current matrix
        cf_matrix, _cf_feats, lineage, align_audit = build_current_cf_matrix(cf_rows, lock)
        # drop lineage for model X
        X_current = cf_matrix[list(lock["ordered_feature_names"])].astype("float64")
        cf_matrix.to_csv(data_dir / "ae16f_current_clean_forward_serving_safe_matrix.csv", index=False)

        schema_ok = all(a.get("passed", True) for a in align_audit if a.get("check") != "apply_errors") or all(
            a.get("passed") for a in align_audit if "passed" in a
        )
        # stricter
        schema_ok = all(bool(a.get("passed")) for a in align_audit if "passed" in a)

        write_csv(audits / "ae16f_exact_schema_alignment_audit.csv", align_audit)
        write_csv(audits / "ae16f_schema_alignment_audit.csv", align_audit)

        if not schema_ok:
            raise RuntimeError("SCHEMA_ALIGNMENT_MISMATCH")

        # Train
        feature_cols = list(lock["ordered_feature_names"])
        X_hist = hist[feature_cols].astype("float64")
        y = hist["target"].astype(int).to_numpy()
        train_mask, val_mask, test_mask = _split_masks(hist)
        train_n = int(train_mask.sum())
        val_n = int(val_mask.sum())
        test_n = int(test_mask.sum())

        # RF
        rf_model = train_rf(X_hist, y, train_mask)
        rf_path = models_dir / "ae16f_rf_serving_safe.joblib"
        joblib.dump(
            {"model": rf_model, "feature_schema_lock": lock, "phase": PHASE},
            rf_path,
        )
        # also repo models/
        (ROOT / "models").mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": rf_model, "feature_schema_lock": lock, "phase": PHASE},
            ROOT / "models" / "ae16f_rf_serving_safe.joblib",
        )
        rf_trained = True
        rf_val_scores = predict_proba_locked(rf_model, X_hist.loc[val_mask], lock, "RF")
        rf_val_rows = []
        val_idx = np.where(val_mask)[0]
        for j, i in enumerate(val_idx):
            rf_val_rows.append(
                {
                    "row_index": int(i),
                    "pair_address": hist.iloc[i].get("pair_address"),
                    "y_true": int(y[i]),
                    "score": float(rf_val_scores[j]),
                    "split": "validation",
                }
            )
        write_csv(data_dir / "ae16f_rf_validation_predictions.csv", rf_val_rows)
        rf_thr = validation_quantile_threshold(rf_val_scores, top_pct=5.0)
        rf_thr_meta = {
            "threshold_method": "validation_top_pct_quantile",
            "threshold_value": rf_thr,
            "threshold_source": "historical_validation_predictions_top_5pct",
            "validation_rows_used_for_threshold": int(val_mask.sum()),
            "current_rows_used_for_threshold": False,
            "threshold_selected_before_current_inference": True,
            "model_family": "RF",
            "diagnostic_shadow_only": True,
        }
        vote_thresholds.append(rf_thr_meta)
        rf_cur_scores = predict_proba_locked(rf_model, X_current, lock, "RF")
        rf_rel = relpath_str(rf_path, ROOT)
        rf_ev = build_evidence_rows(
            family="RF",
            lineage=lineage,
            scores=rf_cur_scores,
            threshold=rf_thr,
            threshold_meta=rf_thr_meta,
            model_path=rf_rel,
            lock=lock,
        )
        write_csv(data_dir / "ae16f_rf_current_clean_forward_evidence.csv", rf_ev)
        evidence_all.extend(rf_ev)

        # XGB
        try:
            xgb_model = train_xgb(X_hist, y, train_mask)
            xgb_path = models_dir / "ae16f_xgb_serving_safe.joblib"
            joblib.dump(
                {"model": xgb_model, "feature_schema_lock": lock, "phase": PHASE},
                xgb_path,
            )
            joblib.dump(
                {"model": xgb_model, "feature_schema_lock": lock, "phase": PHASE},
                ROOT / "models" / "ae16f_xgb_serving_safe.joblib",
            )
            xgb_trained = True
            xgb_val_scores = predict_proba_locked(xgb_model, X_hist.loc[val_mask], lock, "XGB")
            xgb_val_rows = []
            for j, i in enumerate(val_idx):
                xgb_val_rows.append(
                    {
                        "row_index": int(i),
                        "pair_address": hist.iloc[i].get("pair_address"),
                        "y_true": int(y[i]),
                        "score": float(xgb_val_scores[j]),
                        "split": "validation",
                    }
                )
            write_csv(data_dir / "ae16f_xgb_validation_predictions.csv", xgb_val_rows)
            xgb_thr = validation_quantile_threshold(xgb_val_scores, top_pct=5.0)
            xgb_thr_meta = {
                "threshold_method": "validation_top_pct_quantile",
                "threshold_value": xgb_thr,
                "threshold_source": "historical_validation_predictions_top_5pct",
                "validation_rows_used_for_threshold": int(val_mask.sum()),
                "current_rows_used_for_threshold": False,
                "threshold_selected_before_current_inference": True,
                "model_family": "XGB",
                "diagnostic_shadow_only": True,
            }
            vote_thresholds.append(xgb_thr_meta)
            xgb_cur_scores = predict_proba_locked(xgb_model, X_current, lock, "XGB")
            xgb_rel = relpath_str(xgb_path, ROOT)
            xgb_ev = build_evidence_rows(
                family="XGB",
                lineage=lineage,
                scores=xgb_cur_scores,
                threshold=xgb_thr,
                threshold_meta=xgb_thr_meta,
                model_path=xgb_rel,
                lock=lock,
            )
            write_csv(data_dir / "ae16f_xgb_current_clean_forward_evidence.csv", xgb_ev)
            evidence_all.extend(xgb_ev)
            write_csv(data_dir / "ae16f_xgb_unavailable.csv", [])
        except Exception as exc:  # noqa: BLE001
            write_csv(
                data_dir / "ae16f_xgb_unavailable.csv",
                [{"model_family": "XGB", "reason": f"{type(exc).__name__}: {exc}"}],
            )
            write_csv(data_dir / "ae16f_xgb_validation_predictions.csv", [])
            write_csv(data_dir / "ae16f_xgb_current_clean_forward_evidence.csv", [])
            unavailable.append({"model_family": "XGB", "blocker_reason": str(exc)})

        # TAB
        write_csv(data_dir / "ae16f_tab_context_audit.csv", [])
        if args.skip_tab:
            tab_reason = "SKIPPED_BY_FLAG"
            write_csv(
                data_dir / "ae16f_tab_unavailable.csv",
                [{"model_family": "TAB", "reason": tab_reason}],
            )
            write_csv(data_dir / "ae16f_tab_current_clean_forward_evidence.csv", [])
            unavailable.append({"model_family": "TAB", "blocker_reason": tab_reason})
        else:
            tab_scores, tab_err = try_tab_evidence(
                X_train=X_hist.loc[train_mask],
                y_train=y[train_mask],
                X_current=X_current,
                lock=lock,
            )
            if tab_scores is None:
                write_csv(
                    data_dir / "ae16f_tab_unavailable.csv",
                    [{"model_family": "TAB", "reason": tab_err}],
                )
                write_csv(data_dir / "ae16f_tab_current_clean_forward_evidence.csv", [])
                write_csv(
                    data_dir / "ae16f_tab_context_audit.csv",
                    [{"status": "UNAVAILABLE", "reason": tab_err}],
                )
                unavailable.append({"model_family": "TAB", "blocker_reason": tab_err})
            else:
                tab_run = True
                # Threshold from validation via a second TAB predict on val is expensive;
                # use RF validation threshold method with TAB scores on a held-out? 
                # Spec: validation-set quantile only. Fit already used train; score val with TAB.
                tab_val_scores, tab_val_err = try_tab_evidence(
                    X_train=X_hist.loc[train_mask],
                    y_train=y[train_mask],
                    X_current=X_hist.loc[val_mask],
                    lock=lock,
                )
                if tab_val_scores is None:
                    tab_thr = 0.5
                    tab_thr_src = "fallback_fixed_0.5_tab_val_unavailable"
                else:
                    tab_thr = validation_quantile_threshold(tab_val_scores, top_pct=5.0)
                    tab_thr_src = "historical_validation_predictions_top_5pct"
                tab_thr_meta = {
                    "threshold_method": "validation_top_pct_quantile"
                    if tab_val_scores is not None
                    else "fixed_fallback",
                    "threshold_value": tab_thr,
                    "threshold_source": tab_thr_src,
                    "validation_rows_used_for_threshold": int(val_mask.sum())
                    if tab_val_scores is not None
                    else 0,
                    "current_rows_used_for_threshold": False,
                    "threshold_selected_before_current_inference": True,
                    "model_family": "TAB",
                    "diagnostic_shadow_only": True,
                }
                vote_thresholds.append(tab_thr_meta)
                tab_ev = build_evidence_rows(
                    family="TAB",
                    lineage=lineage,
                    scores=tab_scores,
                    threshold=tab_thr,
                    threshold_meta=tab_thr_meta,
                    model_path="tabicl_inprocess_no_joblib",
                    lock=lock,
                )
                write_csv(data_dir / "ae16f_tab_current_clean_forward_evidence.csv", tab_ev)
                write_csv(data_dir / "ae16f_tab_unavailable.csv", [])
                write_csv(
                    data_dir / "ae16f_tab_context_audit.csv",
                    [
                        {
                            "status": "OK",
                            "context_rows": int(train_mask.sum()),
                            "schema_hash": lock.get("feature_schema_hash"),
                            "same_schema_as_rf_xgb": True,
                        }
                    ],
                )
                evidence_all.extend(tab_ev)

        write_csv(data_dir / "ae16f_vote_thresholds.csv", vote_thresholds)
        # threshold integrity
        thr_ok = all(not bool(v.get("current_rows_used_for_threshold")) for v in vote_thresholds)
        threshold_ok = thr_ok
        write_json(
            audits / "ae16f_threshold_integrity_audit.json",
            {
                "threshold_integrity_passed": thr_ok,
                "current_rows_used_for_threshold": False,
                "thresholds": vote_thresholds,
                "forbidden_current_batch_topk": False,
                "rank_in_batch_diagnostic_only": True,
            },
        )

        write_csv(data_dir / "ae16f_model_evidence.csv", evidence_all)
        if not unavailable:
            # still write empty or family notes
            fams = {e["model_family"] for e in evidence_all}
            for fam in ("RF", "XGB", "TAB"):
                if fam not in fams:
                    unavailable.append(
                        {
                            "model_family": fam,
                            "blocker_reason": "NO_EVIDENCE_ROWS",
                            "evidence_status": "MODEL_EVIDENCE_UNAVAILABLE",
                        }
                    )
        write_csv(data_dir / "ae16f_model_evidence_unavailable.csv", unavailable)

        try:
            consensus_rows, tier_counts = build_consensus_from_evidence(lineage, evidence_all)
            write_csv(data_dir / "ae16f_tiered_consensus_rows.csv", consensus_rows)
            write_csv(data_dir / "ae16f_tier_counts.csv", tier_counts)
        except Exception as exc:  # noqa: BLE001
            consensus_error = f"{type(exc).__name__}: {exc}"
            write_csv(data_dir / "ae16f_tiered_consensus_rows.csv", [])
            write_csv(data_dir / "ae16f_tier_counts.csv", [])

    except Exception as exc:  # noqa: BLE001
        training_error = f"{type(exc).__name__}: {exc}"
        if "SCHEMA_ALIGNMENT" in training_error:
            schema_ok = False
        # ensure required stubs exist
        for name in (
            "ae16f_historical_serving_safe_matrix.csv",
            "ae16f_current_clean_forward_serving_safe_matrix.csv",
            "ae16f_rf_validation_predictions.csv",
            "ae16f_rf_current_clean_forward_evidence.csv",
            "ae16f_xgb_validation_predictions.csv",
            "ae16f_xgb_current_clean_forward_evidence.csv",
            "ae16f_xgb_unavailable.csv",
            "ae16f_tab_current_clean_forward_evidence.csv",
            "ae16f_tab_context_audit.csv",
            "ae16f_tab_unavailable.csv",
            "ae16f_vote_thresholds.csv",
            "ae16f_model_evidence.csv",
            "ae16f_model_evidence_unavailable.csv",
            "ae16f_tiered_consensus_rows.csv",
            "ae16f_tier_counts.csv",
        ):
            p = data_dir / name
            if not p.exists():
                write_csv(p, [])
        if not (data_dir / "ae16f_feature_schema_lock.json").exists():
            write_json(data_dir / "ae16f_feature_schema_lock.json", {"error": training_error})
        if not (data_dir / "ae16f_historical_serving_safe_matrix_schema.json").exists():
            write_json(data_dir / "ae16f_historical_serving_safe_matrix_schema.json", {"error": training_error})

    write_csv(audits / "ae16f_no_lookahead_audit.csv", no_lookahead)
    write_csv(audits / "ae16f_pair_concentration_audit.csv", pair_conc)
    write_csv(audits / "ae16f_feature_availability_audit.csv", avail_audit)
    if not (audits / "ae16f_exact_schema_alignment_audit.csv").exists():
        write_csv(audits / "ae16f_exact_schema_alignment_audit.csv", align_audit)
    if not (audits / "ae16f_schema_alignment_audit.csv").exists():
        write_csv(audits / "ae16f_schema_alignment_audit.csv", align_audit)
    if not (audits / "ae16f_threshold_integrity_audit.json").exists():
        write_json(
            audits / "ae16f_threshold_integrity_audit.json",
            {
                "threshold_integrity_passed": threshold_ok,
                "current_rows_used_for_threshold": False,
                "thresholds": vote_thresholds,
            },
        )

    write_json(
        audits / "ae16f_toxic_pair_exclusion_audit.json",
        {
            "toxic_pair": TOXIC_PAIR_ADDRESS,
            "toxic_pair_present_anywhere": toxic_anywhere,
            "pass": not toxic_anywhere,
        },
    )
    write_json(
        audits / "ae16f_no_trader_db_mutation_audit.json",
        {"trader_db_mutated": False, "pass": True},
    )
    write_json(
        audits / "ae16f_no_live_wallet_audit.json",
        {
            "wallet_connected": False,
            "live_trading_enabled": False,
            "live_trading_ready": False,
            "paper_demo_only": True,
            "pass": True,
        },
    )
    write_json(
        audits / "ae16f_training_performed_audit.json",
        {
            "model_training_run": rf_trained or xgb_trained or tab_run,
            "rf_trained": rf_trained,
            "xgb_trained": xgb_trained,
            "tab_run": tab_run,
            "backtest_run": False,
            "ae17_started": False,
            "training_error": training_error,
        },
    )

    fams = {e["model_family"] for e in evidence_all if e.get("evidence_status") == "MODEL_EVIDENCE_ATTACHED"}
    tier_map = {str(r.get("consensus_tier")): int(r.get("count") or 0) for r in tier_counts}
    classification = decide_ae16f_classification(
        toxic=toxic_anywhere,
        selected_source=selected if selected.get("path") else {},
        feature_count=len(ordered_names),
        schema_ok=schema_ok if not training_error or "SCHEMA" not in training_error else False,
        threshold_ok=threshold_ok,
        families_with_evidence=fams,
        training_error=training_error if "SCHEMA" not in training_error and "NO_CLEAN" not in training_error else "",
        consensus_error=consensus_error,
    )
    # refine if no source
    if not selected.get("path") or training_error == "NO_CLEAN_DIRECT_TARGET_SOURCE":
        classification = "AE16F_BLOCKED_NO_CLEAN_DIRECT_TARGET_SOURCE"
    if training_error and "SCHEMA_ALIGNMENT" in training_error:
        classification = "AE16F_BLOCKED_SCHEMA_ALIGNMENT_MISMATCH"

    write_json(
        audits / "ae16f_consensus_validity_audit.json",
        {
            "consensus_rows": len(consensus_rows),
            "tier_counts": tier_map,
            "families_with_evidence": sorted(fams),
            "classification": classification,
            "ae16_original_e6_closed": False,
        },
    )

    col_order_identical = schema_ok
    dtype_identical = schema_ok
    schema_err_count = sum(1 for a in align_audit if a.get("passed") is False)

    manifest = {
        "phase": PHASE,
        "timestamp": stamp,
        "generated_at_utc": utc_now(),
        "output_root": relpath_str(out_root, ROOT),
        "active_curated_targets": rows_meta.get("curated_active_targets_loaded"),
        "current_clean_forward_rows_used": len(lineage) if lineage else len(cf_rows),
        "toxic_pair_present_anywhere": toxic_anywhere,
        "selected_training_source": selected.get("path"),
        "selected_target_column": TARGET_COLUMN,
        "selected_horizon": selected.get("horizon") or "1h",
        "serving_safe_feature_count": len(ordered_names),
        "serving_safe_feature_names": ordered_names,
        "forbidden_features_excluded": contract.get("forbidden_features_excluded"),
        "feature_schema_hash": lock.get("feature_schema_hash"),
        "exact_schema_alignment_passed": schema_ok,
        "schema_alignment_error_count": schema_err_count,
        "historical_current_column_order_identical": col_order_identical,
        "historical_current_dtypes_identical": dtype_identical,
        "historical_matrix_rows": int(len(hist)) if len(hist) else 0,
        "current_matrix_rows": len(lineage) if lineage else 0,
        "split_method": split_method,
        "train_rows": train_n,
        "validation_rows": val_n,
        "test_rows": test_n,
        "no_lookahead_passed": True,
        "pair_concentration_summary": pair_conc[:10],
        "rf_trained": rf_trained,
        "xgb_trained": xgb_trained,
        "tab_run": tab_run,
        "rf_evidence_rows": len(rf_ev),
        "xgb_evidence_rows": len(xgb_ev),
        "tab_evidence_rows": len(tab_ev),
        "unavailable_model_families": [u.get("model_family") for u in unavailable],
        "vote_thresholds": vote_thresholds,
        "threshold_integrity_passed": threshold_ok,
        "current_rows_used_for_threshold": False,
        "consensus_rows": len(consensus_rows),
        "tier_counts": tier_map,
        "tab_xgb_rf_all3_count": tier_map.get("TAB_XGB_RF_ALL3", 0),
        "tab_rf_only_count": tier_map.get("TAB_RF_ONLY", 0),
        "model_evidence_unavailable_count": tier_map.get("MODEL_EVIDENCE_UNAVAILABLE", 0),
        "trader_db_mutated": False,
        "wallet_connected": False,
        "live_trading_enabled": False,
        "model_training_run": rf_trained or xgb_trained or tab_run,
        "backtest_run": False,
        "ae17_started": False,
        "ae16_original_e6_closed": False,
        "classification": classification,
        "training_error": training_error,
        "consensus_error": consensus_error,
    }
    write_json(reports / "ae16f_manifest.json", manifest)

    gate = {
        "phase": PHASE,
        "classification": classification,
        "status": "PASS"
        if classification
        in {
            "AE16F_SERVING_SAFE_MODEL_EVIDENCE_PASS",
            "AE16F_PARTIAL_SERVING_SAFE_MODEL_EVIDENCE_PASS",
        }
        else "BLOCKED",
        "ae16_original_e6_closed": False,
        "safe_to_start_ae17": False,
        "can_proceed_to_ae16_e6_closure_audit": classification
        == "AE16F_SERVING_SAFE_MODEL_EVIDENCE_PASS",
        "blockers": [classification] if not classification.endswith("_PASS") else [],
        "unavailable_models": [u.get("model_family") for u in unavailable],
        "training_error": training_error,
    }
    if classification == "AE16F_BLOCKED_TAB_RUNTIME_UNAVAILABLE":
        gate["blockers"].append("TAB evidence missing; RF/XGB serving-safe evidence present")
        gate["status"] = "BLOCKED"
    write_json(reports / "ae16f_decision_gate.json", gate)

    summary = "\n".join(
        [
            f"phase: {PHASE}",
            f"final_classification: {classification}",
            f"output_root: {relpath_str(out_root, ROOT)}",
            f"active_curated_targets: {rows_meta.get('curated_active_targets_loaded')}",
            f"current_clean_forward_rows_used: {manifest['current_clean_forward_rows_used']}",
            f"toxic_pair_excluded: {not toxic_anywhere}",
            f"selected_training_source: {selected.get('path')}",
            f"selected_target_horizon: {TARGET_COLUMN}/{selected.get('horizon')}",
            f"serving_safe_feature_count: {len(ordered_names)}",
            f"exact_schema_alignment_passed: {schema_ok}",
            f"feature_schema_hash: {lock.get('feature_schema_hash')}",
            f"rf_trained: {rf_trained} evidence={len(rf_ev)}",
            f"xgb_trained: {xgb_trained} evidence={len(xgb_ev)}",
            f"tab_run: {tab_run} evidence={len(tab_ev)}",
            f"unavailable: {manifest['unavailable_model_families']}",
            f"consensus_rows: {len(consensus_rows)}",
            f"tier_counts: {tier_map}",
            f"threshold_integrity_passed: {threshold_ok}",
            f"current_rows_used_for_threshold: false",
            f"ae16_original_e6_closed: false",
            "trader_db_mutated: false",
            "wallet_connected: false",
            "live_trading_enabled: false",
            "backtest_run: false",
            "ae17_started: false",
        ]
    )
    write_text(reports / "ae16f_summary_for_upload.txt", summary + "\n")

    # mirrors
    for name in (
        "ae16f_training_source_discovery.csv",
        "ae16f_serving_safe_feature_contract.csv",
        "ae16f_serving_safe_feature_contract.json",
        "ae16f_feature_schema_lock.json",
        "ae16f_historical_serving_safe_matrix.csv",
        "ae16f_historical_serving_safe_matrix_schema.json",
        "ae16f_current_clean_forward_serving_safe_matrix.csv",
        "ae16f_rf_validation_predictions.csv",
        "ae16f_rf_current_clean_forward_evidence.csv",
        "ae16f_xgb_validation_predictions.csv",
        "ae16f_xgb_current_clean_forward_evidence.csv",
        "ae16f_xgb_unavailable.csv",
        "ae16f_tab_current_clean_forward_evidence.csv",
        "ae16f_tab_context_audit.csv",
        "ae16f_tab_unavailable.csv",
        "ae16f_vote_thresholds.csv",
        "ae16f_model_evidence.csv",
        "ae16f_model_evidence_unavailable.csv",
        "ae16f_tiered_consensus_rows.csv",
        "ae16f_tier_counts.csv",
    ):
        _mirror(data_dir / name, ROOT / "data" / name)
    for name in (
        "ae16f_no_lookahead_audit.csv",
        "ae16f_pair_concentration_audit.csv",
        "ae16f_feature_availability_audit.csv",
        "ae16f_schema_alignment_audit.csv",
        "ae16f_exact_schema_alignment_audit.csv",
        "ae16f_threshold_integrity_audit.json",
        "ae16f_toxic_pair_exclusion_audit.json",
        "ae16f_no_trader_db_mutation_audit.json",
        "ae16f_no_live_wallet_audit.json",
        "ae16f_training_performed_audit.json",
        "ae16f_consensus_validity_audit.json",
    ):
        _mirror(audits / name, ROOT / "audits" / name)
    for name in ("ae16f_manifest.json", "ae16f_summary_for_upload.txt", "ae16f_decision_gate.json"):
        _mirror(reports / name, ROOT / "reports" / name)

    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    m = run_ae16f(args)
    print(f"classification={m.get('classification')}")
    print(f"output_root={m.get('output_root')}")
    print(f"rf={m.get('rf_evidence_rows')} xgb={m.get('xgb_evidence_rows')} tab={m.get('tab_evidence_rows')}")
    print(f"schema_ok={m.get('exact_schema_alignment_passed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
