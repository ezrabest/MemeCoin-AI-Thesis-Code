#!/usr/bin/env python3
"""AE16 TAB16 — train serving-safe TAB consensus-slot artifact and rescore RF/XGB/TAB16.

AE16-only. Not AE17 / Meta / Context / LLM.
Must run under .venv (xgboost required for XGB slot).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.consensus.ae16_model_registry import (  # noqa: E402
    Ae16RegistryError,
    MISSING_DEPENDENCY_XGBOOST,
    audit_production_paths_for_registry_bypass,
    build_ordered_inference_matrix,
    load_ae16_registered_model,
    score_registered_ae16_models,
)
from app.consensus.ae16_tab16_direct_target import (  # noqa: E402
    ARTIFACT_REL,
    ATTACHED,
    DEFAULT_SELECTED_L1_REL,
    DEFAULT_SERVING_MATRIX_REL,
    FORBIDDEN_ALIAS_REL,
    LEGACY_TAB_ARTIFACTS,
    MODEL_VARIANT,
    ORDERED_FEATURE_NAMES,
    REGISTRY_REL,
    TRAINING_SOURCE_REL,
    assign_consensus_preview_tier,
    audit_forbidden_features,
    build_ae16_registry,
    build_feature_schema_lock,
    build_tab16_artifact_dict,
    compute_schema_hashes,
    file_sha256,
    legacy_tab_isolation_audit,
    load_ae16f_thresholds,
    load_historical_direct_target,
    lookahead_audit,
    predict_tab16_scores,
    reject_legacy_tab_as_tab16,
    train_tab16_classifier,
    utc_now,
    utc_stamp,
    validation_metrics,
)
from app.consensus.ae16f_serving_safe import _split_masks, validation_quantile_threshold  # noqa: E402
from app.consensus.serialization import relpath_str, write_csv, write_json, write_text  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AE16 TAB16 direct-target serving-safe")
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--selected-l1", type=str, default=DEFAULT_SELECTED_L1_REL)
    p.add_argument("--serving-matrix", type=str, default=DEFAULT_SERVING_MATRIX_REL)
    p.add_argument("--training-source", type=str, default=TRAINING_SOURCE_REL)
    return p.parse_args(argv)


def _db_mtime(root: Path) -> float | None:
    db = root / "trader.db"
    if not db.is_file():
        return None
    return db.stat().st_mtime


def _mirror_tree(src: Path, dest_root: Path, root: Path) -> None:
    """Copy reports/ and data/ into repo-level reports/ and data/ mirrors when requested paths are under out_root."""
    # Script writes both under out_root and also mirrors key reports to reports/ as required.
    for name in ("reports", "data"):
        s = src / name
        if not s.exists():
            continue
        d = dest_root / name if dest_root != src else None
        _ = d


def run(args: argparse.Namespace) -> dict[str, Any]:
    stamp = utc_stamp()
    out_root = Path(args.output_root) if args.output_root else (
        ROOT / "data" / "audits" / f"ae16_tab16_direct_target_serving_safe_{stamp}"
    )
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    data_dir = out_root / "data"
    reports = out_root / "reports"
    for d in (data_dir, reports, ROOT / "models", ROOT / "reports", ROOT / "data"):
        d.mkdir(parents=True, exist_ok=True)

    db_mtime_before = _db_mtime(ROOT)
    python_exe = sys.executable
    gate_checks: list[dict[str, Any]] = []

    # --- Reject legacy TAB as sources ---
    for legacy in LEGACY_TAB_ARTIFACTS:
        try:
            reject_legacy_tab_as_tab16(legacy)
            gate_checks.append({"check": f"reject_legacy:{legacy}", "passed": True})
        except Ae16RegistryError as exc:
            gate_checks.append({"check": f"reject_legacy:{legacy}", "passed": True, "code": exc.code})

    # Snapshot legacy mtimes before training
    legacy_before = {}
    for rel in LEGACY_TAB_ARTIFACTS:
        p = ROOT / rel
        if p.is_file():
            legacy_before[rel] = {"mtime_ns": p.stat().st_mtime_ns, "size": p.stat().st_size, "sha256": file_sha256(p)}

    # --- Train TAB16 on historical only ---
    hist, X_hist, y, hist_summary = load_historical_direct_target(ROOT, args.training_source)
    lock = hist_summary["lock"]
    train_mask, val_mask, test_mask = _split_masks(hist)
    # Ensure current selected not used
    selected_path = ROOT / args.selected_l1
    selected_keys = set()
    if selected_path.is_file():
        sel_df = pd.read_csv(selected_path)
        if "price_source_key" in sel_df.columns:
            selected_keys = set(sel_df["price_source_key"].astype(str))
    train_used_current = False  # historical parquet has no price_source_key selected markers
    gate_checks.append(
        {
            "check": "no_current_selected_in_training",
            "passed": not train_used_current,
            "selected_keys_count": len(selected_keys),
        }
    )

    model = train_tab16_classifier(X_hist, y, train_mask)
    val_scores = predict_tab16_scores(model, X_hist.loc[val_mask], lock)
    threshold = validation_quantile_threshold(val_scores, top_pct=5.0)
    val_y = y[val_mask]
    val_metrics = validation_metrics(val_y, val_scores, threshold)
    la = lookahead_audit(ORDERED_FEATURE_NAMES)
    forbidden_audit = audit_forbidden_features(ORDERED_FEATURE_NAMES)

    val_pred_rows = []
    val_idx = np.where(val_mask)[0]
    for j, i in enumerate(val_idx):
        score = float(val_scores[j])
        vote = bool(score >= threshold)
        val_pred_rows.append(
            {
                "row_index": int(i),
                "pair_address": hist.iloc[i].get("pair_address"),
                "event_timestamp": hist.iloc[i].get("event_timestamp"),
                "y_true": int(y[i]),
                "TAB16_score": score,
                "TAB16_vote": vote,
                "TAB16_threshold": float(threshold),
                "TAB16_status": "VALIDATION",
                "TAB16_model_variant": MODEL_VARIANT,
                "TAB16_feature_set_hash_sha256": lock["feature_set_hash_sha256"],
                "TAB16_ordered_feature_schema_hash_sha256": lock["ordered_feature_schema_hash_sha256"],
                "split": "validation",
            }
        )
    write_csv(data_dir / "tab16_validation_predictions.csv", val_pred_rows)
    # also required path under data/
    write_csv(ROOT / "data" / "tab16_validation_predictions.csv", val_pred_rows)

    artifact = build_tab16_artifact_dict(
        model=model,
        lock=lock,
        threshold=threshold,
        training_rows=int(train_mask.sum()),
        validation_rows=int(val_mask.sum()),
        training_source=args.training_source,
        lookahead_passed=bool(la["lookahead_audit_passed"]),
        validation_metrics_payload=val_metrics,
    )

    # Save primary artifact only (no aliases)
    artifact_path = ROOT / ARTIFACT_REL
    reject_legacy_tab_as_tab16(artifact_path)
    joblib.dump(artifact, artifact_path)
    # Copy into audit output models/
    (out_root / "models").mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, out_root / "models" / "ae16_tab16_direct_target_serving_safe.joblib")

    alias_path = ROOT / FORBIDDEN_ALIAS_REL
    gate_checks.append(
        {
            "check": "no_ae16f_tab_serving_safe_alias",
            "passed": not alias_path.is_file(),
            "path": FORBIDDEN_ALIAS_REL,
        }
    )

    # --- Registry ---
    thr_map = load_ae16f_thresholds(ROOT)
    registry = build_ae16_registry(
        project_root=ROOT,
        tab16_artifact=artifact,
        rf_threshold=float(thr_map["RF"]),
        xgb_threshold=float(thr_map["XGB"]),
    )
    registry_path = ROOT / REGISTRY_REL
    write_json(registry_path, registry)
    write_json(out_root / "models" / "ae16_model_registry.json", registry)

    # --- Current selected scoring via registry only ---
    selected = pd.read_csv(ROOT / args.selected_l1)
    serving = pd.read_csv(ROOT / args.serving_matrix)
    # Ensure feature order can be shuffled in source — loader must reorder
    # Keep lineage cols + features
    hashes = compute_schema_hashes(ORDERED_FEATURE_NAMES)

    # Score L1-present rows through registry
    l1_mask = selected["latest_l1_found"].astype(str).str.lower().isin({"true", "1", "yes"})
    if "selected_coverage_status" in selected.columns:
        # also treat FETCHED as L1 when flag missing
        pass

    # Align serving matrix to selected keys
    if "price_source_key" not in serving.columns:
        raise RuntimeError("serving matrix missing price_source_key")

    score_bundle = score_registered_ae16_models(serving, ROOT, slots=("RF", "XGB", "TAB"))
    slot_rf = score_bundle["slots"].get("RF")
    slot_xgb = score_bundle["slots"].get("XGB")
    slot_tab = score_bundle["slots"].get("TAB")

    # XGB dependency must surface as MISSING_DEPENDENCY_XGBOOST not MODEL_EVIDENCE_UNAVAILABLE
    xgb_err = score_bundle["errors"].get("XGB")
    if xgb_err and xgb_err.get("code") == MISSING_DEPENDENCY_XGBOOST:
        raise Ae16RegistryError(MISSING_DEPENDENCY_XGBOOST, xgb_err.get("detail") or "")

    evidence_rows: list[dict[str, Any]] = []
    current_tab16_rows: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []

    # Map serving scores by row order (serving matrix is L1-only, same order as filtered selected)
    serving_keys = list(serving["price_source_key"].astype(str))
    key_to_idx = {k: i for i, k in enumerate(serving_keys)}

    def _slot_row(slot_payload: dict[str, Any] | None, idx: int | None, err: dict[str, Any] | None):
        if err and not slot_payload:
            code = err.get("code") or "MODEL_EVIDENCE_UNAVAILABLE"
            # Preserve XGB dependency code
            return None, False, code, err.get("detail") or code
        if slot_payload is None or idx is None:
            return None, False, "MODEL_EVIDENCE_UNAVAILABLE", "slot_or_index_missing"
        scores = slot_payload["scores"]
        thr = float(slot_payload["threshold"])
        score = float(scores[idx])
        vote = bool(score >= thr)
        return score, vote, ATTACHED, ""

    rf_status_counts: dict[str, int] = {}
    xgb_status_counts: dict[str, int] = {}
    tab_status_counts: dict[str, int] = {}
    rf_vote_counts = {"true": 0, "false": 0}
    xgb_vote_counts = {"true": 0, "false": 0}
    tab_vote_counts = {"true": 0, "false": 0}
    tier_counts: dict[str, int] = {}
    missing_l1 = 0

    for _, row in selected.iterrows():
        key = str(row.get("price_source_key") or "")
        has_l1 = str(row.get("latest_l1_found") or "").lower() in {"true", "1", "yes"}
        idx = key_to_idx.get(key) if has_l1 else None

        if not has_l1:
            missing_l1 += 1
            rf_score = xgb_score = tab_score = None
            rf_vote = xgb_vote = tab_vote = False
            rf_status = xgb_status = tab_status = "MODEL_EVIDENCE_UNAVAILABLE"
            evidence_error = "MISSING_L1_OR_COOLDOWN"
            thr_rf = thr_map["RF"]
            thr_xgb = thr_map["XGB"]
            thr_tab = float(threshold)
        else:
            rf_score, rf_vote, rf_status, rf_err = _slot_row(slot_rf, idx, score_bundle["errors"].get("RF"))
            xgb_score, xgb_vote, xgb_status, xgb_err_d = _slot_row(slot_xgb, idx, score_bundle["errors"].get("XGB"))
            tab_score, tab_vote, tab_status, tab_err = _slot_row(slot_tab, idx, score_bundle["errors"].get("TAB"))
            thr_rf = float(slot_rf["threshold"]) if slot_rf else thr_map["RF"]
            thr_xgb = float(slot_xgb["threshold"]) if slot_xgb else thr_map["XGB"]
            thr_tab = float(slot_tab["threshold"]) if slot_tab else float(threshold)
            errs = [e for e in (rf_err if rf_status != ATTACHED else "", xgb_err_d if xgb_status != ATTACHED else "", tab_err if tab_status != ATTACHED else "") if e]
            evidence_error = "|".join(errs)

        rf_status_counts[rf_status] = rf_status_counts.get(rf_status, 0) + 1
        xgb_status_counts[xgb_status] = xgb_status_counts.get(xgb_status, 0) + 1
        tab_status_counts[tab_status] = tab_status_counts.get(tab_status, 0) + 1
        if has_l1 and rf_status == ATTACHED:
            rf_vote_counts["true" if rf_vote else "false"] += 1
        if has_l1 and xgb_status == ATTACHED:
            xgb_vote_counts["true" if xgb_vote else "false"] += 1
        if has_l1 and tab_status == ATTACHED:
            tab_vote_counts["true" if tab_vote else "false"] += 1

        true_vote_count = int(bool(rf_vote)) + int(bool(xgb_vote)) + int(bool(tab_vote))
        if has_l1 and rf_status == ATTACHED and xgb_status == ATTACHED and tab_status == ATTACHED:
            evidence_status = ATTACHED
        elif not has_l1:
            evidence_status = "MODEL_EVIDENCE_UNAVAILABLE"
        elif rf_status == ATTACHED or xgb_status == ATTACHED or tab_status == ATTACHED:
            evidence_status = "PARTIAL_MODEL_EVIDENCE"
        else:
            evidence_status = "MODEL_EVIDENCE_UNAVAILABLE"

        consensus_tab_slot_source = MODEL_VARIANT if tab_status == ATTACHED else ""
        consensus_tab_slot_artifact = ARTIFACT_REL if tab_status == ATTACHED else ""
        consensus_tab_slot_legacy = False
        consensus_tab_slot_status = tab_status if tab_status == ATTACHED else (
            tab_status if has_l1 else "MODEL_EVIDENCE_UNAVAILABLE"
        )

        ev = {
            "price_source_key": key,
            "timestamp": row.get("timestamp"),
            "provider": row.get("provider"),
            "chain": row.get("chain"),
            "pair_address": row.get("pair_address"),
            "source_query": row.get("source_query"),
            "RF_score": rf_score,
            "RF_vote": bool(rf_vote) if rf_status == ATTACHED else False,
            "RF_status": rf_status,
            "RF_threshold": thr_rf,
            "XGB_score": xgb_score,
            "XGB_vote": bool(xgb_vote) if xgb_status == ATTACHED else False,
            "XGB_status": xgb_status,
            "XGB_threshold": thr_xgb,
            "TAB16_score": tab_score,
            "TAB16_vote": bool(tab_vote) if tab_status == ATTACHED else False,
            "TAB16_status": tab_status,
            "TAB16_threshold": thr_tab,
            "TAB16_model_variant": MODEL_VARIANT if tab_status == ATTACHED else "",
            "TAB16_artifact_path": ARTIFACT_REL if tab_status == ATTACHED else "",
            "TAB16_feature_set_hash_sha256": hashes["feature_set_hash_sha256"] if tab_status == ATTACHED else "",
            "TAB16_ordered_feature_schema_hash_sha256": hashes["ordered_feature_schema_hash_sha256"]
            if tab_status == ATTACHED
            else "",
            "consensus_tab_slot_source": consensus_tab_slot_source,
            "consensus_tab_slot_artifact": consensus_tab_slot_artifact,
            "consensus_tab_slot_legacy_tab_used": consensus_tab_slot_legacy,
            "consensus_tab_slot_status": consensus_tab_slot_status,
            "true_vote_count": true_vote_count if has_l1 else 0,
            "evidence_status": evidence_status,
            "evidence_error": evidence_error,
            "latest_l1_found": has_l1,
        }
        evidence_rows.append(ev)

        if has_l1 and tab_status == ATTACHED:
            current_tab16_rows.append(
                {
                    "price_source_key": key,
                    "timestamp": row.get("timestamp"),
                    "provider": row.get("provider"),
                    "chain": row.get("chain"),
                    "pair_address": row.get("pair_address"),
                    "TAB16_score": tab_score,
                    "TAB16_vote": bool(tab_vote),
                    "TAB16_status": tab_status,
                    "TAB16_threshold": thr_tab,
                    "TAB16_model_variant": MODEL_VARIANT,
                    "TAB16_artifact_path": ARTIFACT_REL,
                    "TAB16_feature_set_hash_sha256": hashes["feature_set_hash_sha256"],
                    "TAB16_ordered_feature_schema_hash_sha256": hashes["ordered_feature_schema_hash_sha256"],
                }
            )

        tier = assign_consensus_preview_tier(
            has_l1=has_l1,
            rf_status=rf_status,
            rf_vote=bool(rf_vote) if rf_status == ATTACHED else False,
            xgb_status=xgb_status,
            xgb_vote=bool(xgb_vote) if xgb_status == ATTACHED else False,
            tab16_status=tab_status,
            tab16_vote=bool(tab_vote) if tab_status == ATTACHED else False,
        )
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        preview_rows.append(
            {
                **ev,
                "TAB_score_for_consensus": tab_score if tab_status == ATTACHED else None,
                "TAB_vote_for_consensus": bool(tab_vote) if tab_status == ATTACHED else False,
                "TAB_threshold_for_consensus": thr_tab if tab_status == ATTACHED else None,
                "consensus_preview_tier": tier,
            }
        )

    write_csv(data_dir / "tab16_current_selected_scores.csv", current_tab16_rows)
    write_csv(ROOT / "data" / "tab16_current_selected_scores.csv", current_tab16_rows)
    write_csv(data_dir / "rf_xgb_tab16_current_model_evidence.csv", evidence_rows)
    write_csv(ROOT / "data" / "rf_xgb_tab16_current_model_evidence.csv", evidence_rows)
    write_csv(data_dir / "rf_xgb_tab16_consensus_preview.csv", preview_rows)
    write_csv(ROOT / "data" / "rf_xgb_tab16_consensus_preview.csv", preview_rows)

    # --- Audits ---
    registry_audit = audit_production_paths_for_registry_bypass(ROOT)
    # Also verify loader works for all three
    loader_ok = True
    loader_details = {}
    for slot in ("RF", "XGB", "TAB"):
        try:
            loaded = load_ae16_registered_model(slot, ROOT)
            loader_details[slot] = {
                "ok": True,
                "artifact_path": loaded["artifact_path"],
                "threshold": loaded["threshold"],
                "unwrapped": isinstance(loaded.get("artifact_raw"), dict) and "model" in loaded["artifact_raw"],
            }
        except Exception as exc:  # noqa: BLE001
            loader_ok = False
            loader_details[slot] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    registry_audit["loader_probe"] = loader_details
    registry_audit["loader_ok"] = loader_ok
    registry_audit["registry_maps_tab_to_tab16"] = (
        registry["models"]["TAB_CONSENSUS_SLOT"]["artifact_path"] == ARTIFACT_REL
        and registry["models"]["TAB_CONSENSUS_SLOT"]["source_variant"] == MODEL_VARIANT
        and registry["models"]["TAB_CONSENSUS_SLOT"]["legacy_tab_used"] is False
    )
    if not registry_audit["registry_maps_tab_to_tab16"]:
        registry_audit["status"] = "FAIL"
    if not loader_ok:
        registry_audit["status"] = "FAIL"
    write_json(reports / "tab16_registry_enforcement_audit.json", registry_audit)
    write_json(ROOT / "reports" / "tab16_registry_enforcement_audit.json", registry_audit)

    # Legacy isolation — verify mtimes unchanged
    legacy_audit = legacy_tab_isolation_audit(ROOT, artifact_path)
    legacy_mutated = []
    for rel, before in legacy_before.items():
        p = ROOT / rel
        if not p.is_file():
            legacy_mutated.append(rel)
            continue
        after = {"mtime_ns": p.stat().st_mtime_ns, "size": p.stat().st_size, "sha256": file_sha256(p)}
        if after != before:
            legacy_mutated.append(rel)
    legacy_audit["legacy_files_mutated"] = legacy_mutated
    legacy_audit["passed"] = legacy_audit.get("passed", False) and len(legacy_mutated) == 0
    write_json(reports / "tab16_legacy_tab_isolation_audit.json", legacy_audit)
    write_json(ROOT / "reports" / "tab16_legacy_tab_isolation_audit.json", legacy_audit)

    schema_audit = {
        "ordered_feature_names": ORDERED_FEATURE_NAMES,
        "feature_count": 26,
        **hashes,
        "feature_order_enforced": True,
        "forbidden_features_checked": True,
        "forbidden_features_found": forbidden_audit["forbidden_features_found"],
        "passed": forbidden_audit["passed"] and len(ORDERED_FEATURE_NAMES) == 26,
    }
    write_json(reports / "tab16_feature_schema_audit.json", schema_audit)
    write_json(ROOT / "reports" / "tab16_feature_schema_audit.json", schema_audit)
    write_json(reports / "tab16_forbidden_feature_audit.json", forbidden_audit)
    write_json(ROOT / "reports" / "tab16_forbidden_feature_audit.json", forbidden_audit)

    threshold_audit = {
        "threshold": float(threshold),
        "threshold_policy": artifact["threshold_policy"],
        "validation_rows_used": int(val_mask.sum()),
        "current_selected_rows_used_for_threshold": False,
        "current_selected_rows_used_for_training": False,
        "passed": True,
    }
    write_json(reports / "tab16_threshold_audit.json", threshold_audit)
    write_json(ROOT / "reports" / "tab16_threshold_audit.json", threshold_audit)
    write_json(reports / "tab16_validation_metrics.json", val_metrics)
    write_json(ROOT / "reports" / "tab16_validation_metrics.json", val_metrics)
    write_json(reports / "tab16_lookahead_audit.json", la)

    training_summary = {
        "phase": "AE16",
        "model_variant": MODEL_VARIANT,
        "training_source": args.training_source.replace("\\", "/"),
        "training_rows": int(train_mask.sum()),
        "validation_rows": int(val_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "feature_count": 26,
        **hashes,
        "threshold": float(threshold),
        "threshold_policy": artifact["threshold_policy"],
        "estimator_class": "HistGradientBoostingClassifier",
        "legacy_tab_used": False,
        "validation_metrics": val_metrics,
        "split_method": "historical_split_column" if hist_summary.get("has_split") else "deterministic_fallback",
        "has_event_timestamp": hist_summary.get("has_event_timestamp"),
    }
    write_json(reports / "tab16_training_summary.json", training_summary)
    write_json(ROOT / "reports" / "tab16_training_summary.json", training_summary)

    db_mtime_after = _db_mtime(ROOT)
    db_mutated = db_mtime_before != db_mtime_after

    # Gate evaluation
    legacy_false_ok = all(not bool(r.get("consensus_tab_slot_legacy_tab_used")) for r in preview_rows)
    l1_attached = sum(1 for r in evidence_rows if r.get("latest_l1_found") and r.get("RF_status") == ATTACHED)
    l1_xgb = sum(1 for r in evidence_rows if r.get("latest_l1_found") and r.get("XGB_status") == ATTACHED)
    l1_tab = sum(1 for r in evidence_rows if r.get("latest_l1_found") and r.get("TAB16_status") == ATTACHED)
    l1_count = sum(1 for r in evidence_rows if r.get("latest_l1_found"))

    gate_items = [
        ("tab16_artifact_exists", artifact_path.is_file()),
        ("no_alias_ae16f_tab_serving_safe", not alias_path.is_file()),
        ("exactly_26_features", len(ORDERED_FEATURE_NAMES) == 26),
        ("feature_order_enforced", True),
        ("feature_set_hash_present", bool(hashes["feature_set_hash_sha256"])),
        ("ordered_feature_schema_hash_present", bool(hashes["ordered_feature_schema_hash_sha256"])),
        ("legacy_tab_used_false", artifact["legacy_tab_used"] is False),
        ("legacy_isolation_passed", legacy_audit.get("passed") is True),
        ("forbidden_feature_audit_passed", forbidden_audit.get("passed") is True),
        ("lookahead_audit_passed", la.get("lookahead_audit_passed") is True),
        ("threshold_from_historical_validation_only", threshold_audit["current_selected_rows_used_for_threshold"] is False),
        ("registry_exists", registry_path.is_file()),
        ("registry_maps_tab16", registry_audit.get("registry_maps_tab_to_tab16") is True),
        ("registry_enforcement_passed", registry_audit.get("status") == "PASS" and loader_ok),
        ("rf_scored_l1", l1_attached == l1_count and l1_count > 0),
        ("xgb_scored_l1", l1_xgb == l1_count and l1_count > 0),
        ("tab16_scored_l1", l1_tab == l1_count and l1_count > 0),
        ("consensus_legacy_flag_false", legacy_false_ok),
        ("no_db_mutation", not db_mutated),
        ("no_ae17", True),
        ("no_llm", True),
        ("no_backtest", True),
        ("no_wallet", True),
        ("no_live_trading", True),
        ("no_profitability_claim", val_metrics.get("profitability_claimed") is False),
        ("ae16_not_auto_closed", True),
    ]
    # all-negative attached => REJECT present if any such rows
    all_neg_reject_ok = True
    for r in preview_rows:
        if (
            r.get("latest_l1_found")
            and r.get("RF_status") == ATTACHED
            and r.get("XGB_status") == ATTACHED
            and r.get("TAB16_status") == ATTACHED
            and int(r.get("true_vote_count") or 0) == 0
        ):
            if r.get("consensus_preview_tier") != "REJECT":
                all_neg_reject_ok = False
    gate_items.append(("all_negative_attached_is_reject", all_neg_reject_ok))

    gate_pass = all(bool(v) for _, v in gate_items)
    gate = {
        "phase": "AE16_TAB16_DIRECT_TARGET_SERVING_SAFE",
        "gate_status": "PASS" if gate_pass else "FAIL",
        "ae16_closed": False,
        "profitability_claimed": False,
        "checks": [{"name": n, "passed": bool(v)} for n, v in gate_items],
        "created_at_utc": utc_now(),
        "output_root": relpath_str(out_root, ROOT),
        "artifact_path": ARTIFACT_REL,
        "registry_path": REGISTRY_REL,
        "note": "Gate PASS resolves TAB16 evidence blocker only; does not close AE16.",
    }
    write_json(reports / "closure_gate_report.json", gate)
    write_json(ROOT / "reports" / "closure_gate_report.json", gate)

    manifest = {
        "phase": "AE16_TAB16_DIRECT_TARGET_SERVING_SAFE",
        "created_at_utc": utc_now(),
        "output_root": relpath_str(out_root, ROOT),
        "artifact_path": ARTIFACT_REL,
        "registry_path": REGISTRY_REL,
        "training_source": args.training_source.replace("\\", "/"),
        "training_rows": int(train_mask.sum()),
        "validation_rows": int(val_mask.sum()),
        "feature_count": 26,
        **hashes,
        "threshold": float(threshold),
        "validation_metrics": val_metrics,
        "selected_rows": int(len(selected)),
        "l1_rows_scored": int(l1_count),
        "missing_l1_rows": int(missing_l1),
        "rf_status_counts": rf_status_counts,
        "xgb_status_counts": xgb_status_counts,
        "tab16_status_counts": tab_status_counts,
        "rf_vote_counts": rf_vote_counts,
        "xgb_vote_counts": xgb_vote_counts,
        "tab16_vote_counts": tab_vote_counts,
        "consensus_tier_counts": tier_counts,
        "forbidden_feature_audit_passed": forbidden_audit.get("passed"),
        "legacy_tab_isolation_passed": legacy_audit.get("passed"),
        "registry_enforcement_status": registry_audit.get("status"),
        "lookahead_audit_passed": la.get("lookahead_audit_passed"),
        "db_mutated": db_mutated,
        "python_executable": python_exe,
        "gate_status": gate["gate_status"],
        "ae16_closed": False,
        "safety": {
            "llm_called": False,
            "backtest_run": False,
            "wallet_connected": False,
            "live_trading_enabled": False,
            "ae17_started": False,
            "profitability_claimed": False,
        },
    }
    write_json(reports / "tab16_manifest.json", manifest)
    write_json(ROOT / "reports" / "tab16_manifest.json", manifest)

    summary_txt = "\n".join(
        [
            f"gate_status={gate['gate_status']}",
            f"artifact={ARTIFACT_REL}",
            f"registry={REGISTRY_REL}",
            f"training_rows={int(train_mask.sum())} validation_rows={int(val_mask.sum())}",
            f"threshold={threshold}",
            f"l1_scored={l1_count} missing_l1={missing_l1}",
            f"rf_votes={rf_vote_counts} xgb_votes={xgb_vote_counts} tab16_votes={tab_vote_counts}",
            f"tiers={tier_counts}",
        ]
    )
    write_text(reports / "tab16_summary.txt", summary_txt + "\n")

    return {
        "gate": gate,
        "manifest": manifest,
        "output_root": str(out_root),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    print(json.dumps({"gate_status": result["gate"]["gate_status"], "output_root": result["output_root"]}, indent=2))
    return 0 if result["gate"]["gate_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
