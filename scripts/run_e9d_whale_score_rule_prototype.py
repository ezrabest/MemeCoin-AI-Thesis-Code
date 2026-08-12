#!/usr/bin/env python
"""
E9D — Non-Training Whale-Score Rule Prototype

Purpose:
  Offline, read-only diagnostic prototype for a simple whale_score_asof rule.

Hypothesis from E9B/E9C:
  rare-winner positives tend to have LOWER whale_score_asof than matched controls.

Strict boundaries:
  - No model training
  - No RF/TAB/XGB retraining
  - No runtime changes
  - No UI changes
  - No trading/demo/live changes
  - No SQLite writes
  - No external API calls
  - No Qwen/Gemini/Ollama calls
  - No reservoir scoring deployment

This is not a trading rule approval.
This is only a research diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_E9A_ROOT = ROOT / "data" / "training" / "manual_verified_results" / "phase_e9a_matched_control_contract_20260708_202222"
DEFAULT_E9B_ROOT = ROOT / "data" / "training" / "manual_verified_results" / "phase_e9b_context_discrimination_20260709_081445"
DEFAULT_E9C_ROOT = ROOT / "data" / "training" / "manual_verified_results" / "phase_e9c_whale_score_contract_20260709_082522"

FEATURE = "whale_score_asof"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_csv_required(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {label}: {path}")
    return pd.read_csv(path, low_memory=False)


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc), "_path": str(path)}


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8")


def norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


def find_col(df: pd.DataFrame, name: str) -> str | None:
    target = norm_col(name)
    for c in df.columns:
        if norm_col(c) == target:
            return c
    return None


def detect_pair_col(df: pd.DataFrame) -> str | None:
    for name in ["pair_address", "pairAddress", "pair", "contract_address", "token_address"]:
        c = find_col(df, name)
        if c:
            return c
    return None


def detect_target_row_col(df: pd.DataFrame) -> str | None:
    for name in ["target_row_id", "targetRowId"]:
        c = find_col(df, name)
        if c:
            return c
    return None


def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_div(a: float, b: float) -> float | None:
    if b == 0:
        return None
    return float(a / b)


def safe_rate(n: int, d: int) -> float | None:
    if d == 0:
        return None
    return float(n / d)


def build_combined_rows(positives: pd.DataFrame, controls: pd.DataFrame, feature_col: str) -> pd.DataFrame:
    pos = positives.copy()
    ctrl = controls.copy()

    pos["_e9d_group"] = "positive"
    ctrl["_e9d_group"] = "control"

    common = sorted(set(pos.columns).intersection(set(ctrl.columns)))
    needed = list(dict.fromkeys(common + ["_e9d_group"]))

    combined = pd.concat([pos[needed], ctrl[needed]], ignore_index=True)
    combined["_feature_value"] = coerce_numeric(combined[feature_col])
    combined["_is_positive"] = combined["_e9d_group"].eq("positive").astype(int)
    combined["_is_control"] = combined["_e9d_group"].eq("control").astype(int)
    return combined


def threshold_grid(values: pd.Series) -> list[float]:
    vals = coerce_numeric(values).dropna()
    if vals.empty:
        return []

    quantiles = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
    qs = [float(vals.quantile(q)) for q in quantiles]

    unique = []
    for q in qs:
        if not math.isnan(q) and q not in unique:
            unique.append(q)
    return unique


def evaluate_rule_grid(
    combined: pd.DataFrame,
    pair_col: str | None,
    min_positive_capture: float,
    min_enrichment: float,
) -> pd.DataFrame:
    thresholds = threshold_grid(combined["_feature_value"])
    rows = []

    total_pos = int(combined["_is_positive"].sum())
    total_ctrl = int(combined["_is_control"].sum())
    base_positive_rate = safe_rate(total_pos, len(combined))

    for t in thresholds:
        selected = combined[combined["_feature_value"] <= t].copy()

        selected_pos = int(selected["_is_positive"].sum())
        selected_ctrl = int(selected["_is_control"].sum())
        selected_total = int(len(selected))

        pos_capture = safe_rate(selected_pos, total_pos)
        ctrl_capture = safe_rate(selected_ctrl, total_ctrl)
        selected_positive_rate = safe_rate(selected_pos, selected_total)
        enrichment_vs_base = safe_div(selected_positive_rate, base_positive_rate) if selected_positive_rate is not None and base_positive_rate else None
        pos_ctrl_capture_ratio = safe_div(pos_capture, ctrl_capture) if pos_capture is not None and ctrl_capture is not None else None

        unique_pos_pairs = None
        top_pos_pair_share = None
        if pair_col and pair_col in selected.columns:
            pos_sel = selected[selected["_e9d_group"] == "positive"]
            if len(pos_sel):
                vc = pos_sel[pair_col].astype(str).value_counts()
                unique_pos_pairs = int(vc.shape[0])
                top_pos_pair_share = float(vc.iloc[0] / len(pos_sel)) if len(vc) else None
            else:
                unique_pos_pairs = 0
                top_pos_pair_share = None

        passes_basic = bool(
            pos_capture is not None
            and enrichment_vs_base is not None
            and pos_capture >= min_positive_capture
            and enrichment_vs_base >= min_enrichment
        )

        rows.append({
            "feature_name": FEATURE,
            "rule_direction": "LOWER_OR_EQUAL_IS_BETTER",
            "threshold": t,
            "selected_total": selected_total,
            "selected_positives": selected_pos,
            "selected_controls": selected_ctrl,
            "total_positives": total_pos,
            "total_controls": total_ctrl,
            "positive_capture_rate": pos_capture,
            "control_capture_rate": ctrl_capture,
            "pos_ctrl_capture_ratio": pos_ctrl_capture_ratio,
            "base_positive_rate": base_positive_rate,
            "selected_positive_rate": selected_positive_rate,
            "enrichment_vs_base": enrichment_vs_base,
            "unique_selected_positive_pairs": unique_pos_pairs,
            "top_selected_positive_pair_share": top_pos_pair_share,
            "passes_basic_rule_gate": passes_basic,
        })

    return pd.DataFrame(rows)


def apply_rule(df: pd.DataFrame, feature_col: str, threshold: float) -> pd.Series:
    vals = coerce_numeric(df[feature_col])
    return vals <= threshold


def top_pair_removal_rule_audit(
    positives: pd.DataFrame,
    controls: pd.DataFrame,
    feature_col: str,
    pair_col: str | None,
    threshold: float,
) -> pd.DataFrame:
    if not pair_col or pair_col not in positives.columns:
        return pd.DataFrame([{"audit_status": "SKIPPED_NO_PAIR_COLUMN"}])

    if len(positives) == 0:
        return pd.DataFrame([{"audit_status": "SKIPPED_NO_POSITIVES"}])

    vc = positives[pair_col].astype(str).value_counts()
    if vc.empty:
        return pd.DataFrame([{"audit_status": "SKIPPED_NO_PAIR_VALUES"}])

    top_pair = str(vc.index[0])
    reduced_pos = positives[positives[pair_col].astype(str) != top_pair].copy()

    original_combined = build_combined_rows(positives, controls, feature_col)
    reduced_combined = build_combined_rows(reduced_pos, controls, feature_col)

    def eval_one(combined: pd.DataFrame) -> dict[str, Any]:
        selected = combined[combined["_feature_value"] <= threshold]
        total_pos = int(combined["_is_positive"].sum())
        total_ctrl = int(combined["_is_control"].sum())
        selected_pos = int(selected["_is_positive"].sum())
        selected_ctrl = int(selected["_is_control"].sum())
        selected_total = int(len(selected))
        base_rate = safe_rate(total_pos, len(combined))
        selected_rate = safe_rate(selected_pos, selected_total)
        return {
            "total_positives": total_pos,
            "total_controls": total_ctrl,
            "selected_total": selected_total,
            "selected_positives": selected_pos,
            "selected_controls": selected_ctrl,
            "positive_capture_rate": safe_rate(selected_pos, total_pos),
            "control_capture_rate": safe_rate(selected_ctrl, total_ctrl),
            "selected_positive_rate": selected_rate,
            "enrichment_vs_base": safe_div(selected_rate, base_rate) if selected_rate is not None and base_rate else None,
        }

    original = eval_one(original_combined)
    reduced = eval_one(reduced_combined)

    return pd.DataFrame([{
        "audit_status": "OK",
        "threshold": threshold,
        "removed_top_pair": top_pair,
        "top_pair_rows": int(vc.iloc[0]),
        "top_pair_share": float(vc.iloc[0] / len(positives)),
        **{f"original_{k}": v for k, v in original.items()},
        **{f"after_top_pair_removal_{k}": v for k, v in reduced.items()},
    }])


def leave_one_pair_out_rule_audit(
    positives: pd.DataFrame,
    controls: pd.DataFrame,
    feature_col: str,
    pair_col: str | None,
    threshold: float,
) -> pd.DataFrame:
    if not pair_col or pair_col not in positives.columns:
        return pd.DataFrame([{"audit_status": "SKIPPED_NO_PAIR_COLUMN"}])

    pairs = sorted(positives[pair_col].dropna().astype(str).unique())
    if not pairs:
        return pd.DataFrame([{"audit_status": "SKIPPED_NO_PAIR_VALUES"}])

    rows = []
    for pair in pairs:
        reduced_pos = positives[positives[pair_col].astype(str) != pair].copy()
        combined = build_combined_rows(reduced_pos, controls, feature_col)
        selected = combined[combined["_feature_value"] <= threshold]

        total_pos = int(combined["_is_positive"].sum())
        total_ctrl = int(combined["_is_control"].sum())
        selected_pos = int(selected["_is_positive"].sum())
        selected_ctrl = int(selected["_is_control"].sum())
        selected_total = int(len(selected))

        base_rate = safe_rate(total_pos, len(combined))
        selected_rate = safe_rate(selected_pos, selected_total)

        rows.append({
            "audit_status": "OK",
            "threshold": threshold,
            "removed_pair": pair,
            "remaining_positive_rows": total_pos,
            "selected_total": selected_total,
            "selected_positives": selected_pos,
            "selected_controls": selected_ctrl,
            "positive_capture_rate": safe_rate(selected_pos, total_pos),
            "control_capture_rate": safe_rate(selected_ctrl, total_ctrl),
            "selected_positive_rate": selected_rate,
            "enrichment_vs_base": safe_div(selected_rate, base_rate) if selected_rate is not None and base_rate else None,
        })

    return pd.DataFrame(rows)


def matched_pair_rule_audit(
    pairs: pd.DataFrame,
    positives: pd.DataFrame,
    controls: pd.DataFrame,
    feature_col: str,
    threshold: float,
) -> pd.DataFrame:
    pos_ref = None
    ctrl_ref = None

    for c in pairs.columns:
        n = norm_col(c)
        if pos_ref is None and "positive" in n and "target_row_id" in n:
            pos_ref = c
        if ctrl_ref is None and "control" in n and "target_row_id" in n:
            ctrl_ref = c

    pos_key = detect_target_row_col(positives)
    ctrl_key = detect_target_row_col(controls)

    if not pos_ref or not ctrl_ref or not pos_key or not ctrl_key:
        return pd.DataFrame([{"audit_status": "SKIPPED_COULD_NOT_FIND_PAIR_REFERENCES"}])

    pos = positives[[pos_key, feature_col]].copy()
    ctrl = controls[[ctrl_key, feature_col]].copy()

    pos["_pos_ref"] = pos[pos_key].astype(str)
    ctrl["_ctrl_ref"] = ctrl[ctrl_key].astype(str)
    pos["_pos_rule_hit"] = apply_rule(pos, feature_col, threshold)
    ctrl["_ctrl_rule_hit"] = apply_rule(ctrl, feature_col, threshold)

    merged = pairs.copy()
    merged["_pos_ref"] = merged[pos_ref].astype(str)
    merged["_ctrl_ref"] = merged[ctrl_ref].astype(str)

    merged = merged.merge(pos[["_pos_ref", feature_col, "_pos_rule_hit"]].rename(columns={feature_col: "positive_feature_value"}), on="_pos_ref", how="left")
    merged = merged.merge(ctrl[["_ctrl_ref", feature_col, "_ctrl_rule_hit"]].rename(columns={feature_col: "control_feature_value"}), on="_ctrl_ref", how="left")

    valid = merged["_pos_rule_hit"].notna() & merged["_ctrl_rule_hit"].notna()
    if valid.sum() == 0:
        return pd.DataFrame([{"audit_status": "NO_VALID_MATCHED_RULE_ROWS"}])

    sub = merged[valid].copy()
    pos_hit = sub["_pos_rule_hit"].astype(bool)
    ctrl_hit = sub["_ctrl_rule_hit"].astype(bool)

    return pd.DataFrame([{
        "audit_status": "OK",
        "threshold": threshold,
        "valid_matched_pairs": int(len(sub)),
        "positive_rule_hit_share": float(pos_hit.mean()),
        "control_rule_hit_share": float(ctrl_hit.mean()),
        "positive_only_hit_share": float((pos_hit & ~ctrl_hit).mean()),
        "control_only_hit_share": float((~pos_hit & ctrl_hit).mean()),
        "both_hit_share": float((pos_hit & ctrl_hit).mean()),
        "neither_hit_share": float((~pos_hit & ~ctrl_hit).mean()),
    }])


def choose_best_rule(grid: pd.DataFrame) -> dict[str, Any]:
    if grid.empty:
        return {}

    candidates = grid[grid["passes_basic_rule_gate"] == True].copy()
    if candidates.empty:
        candidates = grid.copy()

    candidates["_rank_enrichment"] = pd.to_numeric(candidates["enrichment_vs_base"], errors="coerce").fillna(-999)
    candidates["_rank_capture"] = pd.to_numeric(candidates["positive_capture_rate"], errors="coerce").fillna(-999)
    candidates["_rank_control"] = pd.to_numeric(candidates["control_capture_rate"], errors="coerce").fillna(999)

    candidates = candidates.sort_values(
        by=["passes_basic_rule_gate", "_rank_enrichment", "_rank_capture", "_rank_control"],
        ascending=[False, False, False, True],
    )

    return candidates.iloc[0].drop(labels=[c for c in candidates.columns if c.startswith("_rank_")], errors="ignore").to_dict()


def decide_e9d(
    best: dict[str, Any],
    top_pair_audit: pd.DataFrame,
    loo_audit: pd.DataFrame,
    matched_audit: pd.DataFrame,
    e9c_decision: dict[str, Any],
    min_enrichment: float,
    min_positive_capture: float,
) -> dict[str, Any]:
    blockers = []
    warnings = []

    if not best:
        blockers.append("NO_RULE_GRID")
        return {
            "decision": "E9D_FAIL_NO_RULE_GRID",
            "blockers": blockers,
            "warnings": warnings,
            "approved_for_e9e_modeling_feasibility": False,
            "approved_for_runtime": False,
            "approved_for_training": False,
        }

    if not e9c_decision.get("approved_for_e9d_non_training_rule_prototype", False):
        blockers.append("E9C_DID_NOT_APPROVE_E9D")

    for w in e9c_decision.get("warnings", []) or []:
        warnings.append(f"E9C_{w}")

    best_enrich = best.get("enrichment_vs_base")
    best_capture = best.get("positive_capture_rate")

    if best_enrich is None or pd.isna(best_enrich) or float(best_enrich) < min_enrichment:
        blockers.append("BEST_RULE_ENRICHMENT_BELOW_THRESHOLD")

    if best_capture is None or pd.isna(best_capture) or float(best_capture) < min_positive_capture:
        blockers.append("BEST_RULE_POSITIVE_CAPTURE_BELOW_THRESHOLD")

    top_ok = False
    if not top_pair_audit.empty and "after_top_pair_removal_enrichment_vs_base" in top_pair_audit.columns:
        val = top_pair_audit.iloc[0].get("after_top_pair_removal_enrichment_vs_base")
        cap = top_pair_audit.iloc[0].get("after_top_pair_removal_positive_capture_rate")
        top_ok = (
            val is not None and not pd.isna(val) and float(val) >= 1.0
            and cap is not None and not pd.isna(cap) and float(cap) > 0
        )
    if not top_ok:
        blockers.append("RULE_DOES_NOT_SURVIVE_TOP_PAIR_REMOVAL")

    loo_ok = False
    if not loo_audit.empty and "enrichment_vs_base" in loo_audit.columns:
        vals = pd.to_numeric(loo_audit["enrichment_vs_base"], errors="coerce").dropna()
        caps = pd.to_numeric(loo_audit["positive_capture_rate"], errors="coerce").dropna()
        if len(vals) and len(caps):
            loo_ok = bool(vals.min() >= 1.0 and caps.min() > 0)
    if not loo_ok:
        blockers.append("RULE_DOES_NOT_SURVIVE_LEAVE_ONE_PAIR_OUT")

    matched_ok = False
    if not matched_audit.empty and "positive_rule_hit_share" in matched_audit.columns and "control_rule_hit_share" in matched_audit.columns:
        p = matched_audit.iloc[0].get("positive_rule_hit_share")
        c = matched_audit.iloc[0].get("control_rule_hit_share")
        matched_ok = (
            p is not None and c is not None
            and not pd.isna(p) and not pd.isna(c)
            and float(p) > float(c)
        )
    if not matched_ok:
        warnings.append("MATCHED_PAIR_RULE_HIT_SHARE_NOT_STRONGLY_POSITIVE_OVER_CONTROL")

    if blockers:
        decision = "E9D_RESEARCH_ONLY_RULE_NOT_ROBUST_ENOUGH"
        approve_e9e = False
    else:
        decision = "E9D_RESEARCH_ONLY_RULE_CANDIDATE"
        approve_e9e = False

    return {
        "decision": decision,
        "feature_name": FEATURE,
        "rule": {
            "direction": "whale_score_asof <= threshold",
            "threshold": best.get("threshold"),
        },
        "best_rule": best,
        "blockers": blockers,
        "warnings": warnings,
        "approved_for_e9e_modeling_feasibility": approve_e9e,
        "approved_for_modeling": False,
        "approved_for_training": False,
        "approved_for_runtime": False,
        "approved_for_ui": False,
        "approved_for_trading": False,
        "reason": (
            "Rule remains research-only. It may inform forward collection or final E9 summary, but does not justify modeling/runtime."
            if not blockers
            else "Rule did not pass enough robustness gates to justify E9E."
        ),
        "created_at_utc": utc_now_iso(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="E9D Non-Training Whale-Score Rule Prototype")
    parser.add_argument("--e9a-root", type=Path, default=DEFAULT_E9A_ROOT)
    parser.add_argument("--e9b-root", type=Path, default=DEFAULT_E9B_ROOT)
    parser.add_argument("--e9c-root", type=Path, default=DEFAULT_E9C_ROOT)
    parser.add_argument("--feature", default=FEATURE)
    parser.add_argument("--min-positive-capture", type=float, default=0.20)
    parser.add_argument("--min-enrichment", type=float, default=1.20)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    if norm_col(args.feature) != norm_col(FEATURE):
        raise ValueError("E9D currently supports only whale_score_asof.")

    output_root = args.output_root or (
        ROOT / "data" / "training" / "manual_verified_results" / f"phase_e9d_whale_score_rule_prototype_{timestamp_slug()}"
    )
    reports_dir = output_root / "reports"
    data_dir = output_root / "data"
    audits_dir = output_root / "audits"
    for d in [reports_dir, data_dir, audits_dir]:
        ensure_dir(d)

    positives = read_csv_required(args.e9a_root / "data" / "e9a_positive_rows.csv", "E9A positive rows")
    controls = read_csv_required(args.e9a_root / "data" / "e9a_control_rows.csv", "E9A control rows")
    matched_pairs = read_csv_required(args.e9a_root / "data" / "e9a_matched_pairs.csv", "E9A matched pairs")

    e9b_decision = read_json_optional(args.e9b_root / "audits" / "e9b_decision_gate.json")
    e9c_decision = read_json_optional(args.e9c_root / "audits" / "e9c_decision_gate.json")

    feature_pos = find_col(positives, args.feature)
    feature_ctrl = find_col(controls, args.feature)
    if feature_pos is None or feature_ctrl is None:
        raise ValueError("whale_score_asof missing from positives or controls.")

    if feature_pos != feature_ctrl:
        # Normalize for downstream combined operations.
        controls = controls.rename(columns={feature_ctrl: feature_pos})
    feature_col = feature_pos

    pair_col = detect_pair_col(positives)

    combined = build_combined_rows(positives, controls, feature_col)
    grid = evaluate_rule_grid(
        combined=combined,
        pair_col=pair_col,
        min_positive_capture=args.min_positive_capture,
        min_enrichment=args.min_enrichment,
    )
    write_csv(data_dir / "e9d_whale_score_rule_grid.csv", grid)

    best = choose_best_rule(grid)
    threshold = best.get("threshold")
    if threshold is None or pd.isna(threshold):
        threshold = float("nan")

    best_df = pd.DataFrame([best]) if best else pd.DataFrame()
    write_csv(data_dir / "e9d_selected_rule.csv", best_df)

    top_pair_audit = top_pair_removal_rule_audit(
        positives=positives,
        controls=controls,
        feature_col=feature_col,
        pair_col=pair_col,
        threshold=float(threshold),
    )
    write_csv(audits_dir / "e9d_top_pair_removal_rule_audit.csv", top_pair_audit)

    loo_audit = leave_one_pair_out_rule_audit(
        positives=positives,
        controls=controls,
        feature_col=feature_col,
        pair_col=pair_col,
        threshold=float(threshold),
    )
    write_csv(audits_dir / "e9d_leave_one_pair_out_rule_audit.csv", loo_audit)

    matched_audit = matched_pair_rule_audit(
        pairs=matched_pairs,
        positives=positives,
        controls=controls,
        feature_col=feature_col,
        threshold=float(threshold),
    )
    write_csv(audits_dir / "e9d_matched_pair_rule_audit.csv", matched_audit)

    decision = decide_e9d(
        best=best,
        top_pair_audit=top_pair_audit,
        loo_audit=loo_audit,
        matched_audit=matched_audit,
        e9c_decision=e9c_decision,
        min_enrichment=args.min_enrichment,
        min_positive_capture=args.min_positive_capture,
    )

    decision["e9b_decision"] = e9b_decision.get("decision")
    decision["e9c_decision"] = e9c_decision.get("decision")
    decision["positive_rows"] = int(len(positives))
    decision["control_rows"] = int(len(controls))
    decision["matched_pairs"] = int(len(matched_pairs))
    if pair_col and pair_col in positives.columns:
        decision["unique_positive_pairs"] = int(positives[pair_col].astype(str).nunique())
        vc = positives[pair_col].astype(str).value_counts()
        decision["top_positive_pair_share"] = float(vc.iloc[0] / len(positives)) if len(vc) else None

    write_json(audits_dir / "e9d_decision_gate.json", decision)

    manifest = {
        "phase": "E9D",
        "branch_name": "phase_e9d_non_training_whale_score_rule_prototype",
        "created_at_utc": utc_now_iso(),
        "status": "completed",
        "boundaries": {
            "model_training": False,
            "runtime_changes": False,
            "ui_changes": False,
            "trading_changes": False,
            "sqlite_writes": False,
            "external_api_calls": False,
            "llm_calls": False,
            "reservoir_scoring_deployment": False,
        },
        "inputs": {
            "e9a_root": str(args.e9a_root),
            "e9b_root": str(args.e9b_root),
            "e9c_root": str(args.e9c_root),
            "feature": args.feature,
        },
        "outputs": {
            "output_root": str(output_root),
            "manifest": str(reports_dir / "e9d_manifest.json"),
            "summary": str(reports_dir / "e9d_summary_for_upload.txt"),
            "rule_grid": str(data_dir / "e9d_whale_score_rule_grid.csv"),
            "selected_rule": str(data_dir / "e9d_selected_rule.csv"),
            "top_pair_audit": str(audits_dir / "e9d_top_pair_removal_rule_audit.csv"),
            "leave_one_pair_audit": str(audits_dir / "e9d_leave_one_pair_out_rule_audit.csv"),
            "matched_pair_rule_audit": str(audits_dir / "e9d_matched_pair_rule_audit.csv"),
            "decision_gate": str(audits_dir / "e9d_decision_gate.json"),
        },
        "decision": decision.get("decision"),
    }
    write_json(reports_dir / "e9d_manifest.json", manifest)

    summary_lines = [
        "Phase / branch name",
        "",
        "E9D — Non-Training Whale-Score Rule Prototype",
        "",
        "Run status",
        "",
        "COMPLETED",
        "",
        "Decision",
        "",
        str(decision.get("decision")),
        "",
        "Scope",
        "",
        "Offline read-only non-training rule prototype.",
        "No model training, no runtime, no UI, no trading, no SQLite writes, no external APIs, no LLM calls.",
        "",
        "Hypothesis",
        "",
        "Rare-winner positives may have lower whale_score_asof than matched controls.",
        "",
        "Selected rule",
        "",
        f"- rule: whale_score_asof <= {decision.get('rule', {}).get('threshold')}",
        f"- direction: LOWER_OR_EQUAL_IS_BETTER",
        "",
        "Best rule metrics",
        "",
    ]

    for k in [
        "selected_total",
        "selected_positives",
        "selected_controls",
        "positive_capture_rate",
        "control_capture_rate",
        "selected_positive_rate",
        "enrichment_vs_base",
        "unique_selected_positive_pairs",
        "top_selected_positive_pair_share",
    ]:
        summary_lines.append(f"- {k}: {best.get(k)}")

    summary_lines.extend([
        "",
        "Approvals",
        "",
        f"- approved_for_e9e_modeling_feasibility: {decision.get('approved_for_e9e_modeling_feasibility')}",
        f"- approved_for_modeling: {decision.get('approved_for_modeling')}",
        f"- approved_for_training: {decision.get('approved_for_training')}",
        f"- approved_for_runtime: {decision.get('approved_for_runtime')}",
        "",
        "Warnings",
        "",
    ])

    warnings = decision.get("warnings") or []
    if warnings:
        for w in warnings:
            summary_lines.append(f"- {w}")
    else:
        summary_lines.append("None.")

    summary_lines.extend([
        "",
        "Blockers",
        "",
    ])

    blockers = decision.get("blockers") or []
    if blockers:
        for b in blockers:
            summary_lines.append(f"- {b}")
    else:
        summary_lines.append("None.")

    summary_lines.extend([
        "",
        "Final interpretation",
        "",
        str(decision.get("reason")),
    ])

    (reports_dir / "e9d_summary_for_upload.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(json.dumps({
        "status": "completed",
        "phase": "E9D",
        "output_root": str(output_root),
        "decision": decision.get("decision"),
        "selected_rule_threshold": decision.get("rule", {}).get("threshold"),
        "approved_for_e9e_modeling_feasibility": decision.get("approved_for_e9e_modeling_feasibility"),
        "approved_for_modeling": decision.get("approved_for_modeling"),
        "approved_for_runtime": decision.get("approved_for_runtime"),
        "summary": str(reports_dir / "e9d_summary_for_upload.txt"),
        "decision_gate": str(audits_dir / "e9d_decision_gate.json"),
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
