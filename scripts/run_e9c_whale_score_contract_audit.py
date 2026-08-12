#!/usr/bin/env python
"""
E9C — Whale-Score Context Feature Contract Audit

Purpose:
  Audit whether whale_score_asof is a legitimate context feature candidate
  for continued research after E9B.

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

This script reads existing E9A/E9B/E8 artifacts only.
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
DEFAULT_E8_ROOT = ROOT / "data" / "training" / "manual_verified_results" / "phase_e8e_rare_winner_context_forensics_20260707_195349"

FORBIDDEN_PATTERNS = [
    r"target", r"label", r"future", r"outcome", r"net_return", r"return_after",
    r"realized", r"pnl", r"profit", r"profitable", r"exit", r"simulation",
    r"sim_", r"tp_hit", r"sl_hit", r"time_stop", r"max_return", r"min_return",
    r"drawdown_after", r"price_after", r"winner", r"positive_label",
]

SOURCE_SCAN_DIRS = [
    "app",
    "scripts",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_csv_required(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required {name}: {path}")
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


def forbidden_name(name: str) -> bool:
    n = norm_col(name)
    return any(re.search(p, n, flags=re.I) for p in FORBIDDEN_PATTERNS)


def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def safe_stats(s: pd.Series) -> dict[str, Any]:
    vals = coerce_numeric(s)
    non_missing = vals.dropna()
    if len(non_missing) == 0:
        return {
            "count": int(len(s)),
            "non_missing": 0,
            "missing": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()) if len(s) else None,
            "min": None,
            "p01": None,
            "p05": None,
            "median": None,
            "mean": None,
            "p95": None,
            "p99": None,
            "max": None,
            "std": None,
        }

    return {
        "count": int(len(s)),
        "non_missing": int(non_missing.shape[0]),
        "missing": int(vals.isna().sum()),
        "missing_rate": float(vals.isna().mean()) if len(vals) else None,
        "min": float(non_missing.min()),
        "p01": float(non_missing.quantile(0.01)),
        "p05": float(non_missing.quantile(0.05)),
        "median": float(non_missing.median()),
        "mean": float(non_missing.mean()),
        "p95": float(non_missing.quantile(0.95)),
        "p99": float(non_missing.quantile(0.99)),
        "max": float(non_missing.max()),
        "std": float(non_missing.std()) if len(non_missing) > 1 else 0.0,
    }


def cliffs_delta(x: pd.Series, y: pd.Series) -> float | None:
    xv = coerce_numeric(x).dropna().to_numpy(dtype=float)
    yv = coerce_numeric(y).dropna().to_numpy(dtype=float)
    if len(xv) == 0 or len(yv) == 0:
        return None

    y_sorted = np.sort(yv)
    greater = 0
    less = 0
    for value in xv:
        greater += int(np.searchsorted(y_sorted, value, side="left"))
        less += int(len(y_sorted) - np.searchsorted(y_sorted, value, side="right"))

    denom = len(xv) * len(yv)
    return float((greater - less) / denom) if denom else None


def detect_pair_col(df: pd.DataFrame) -> str | None:
    candidates = ["pair_address", "pairAddress", "pair", "contract_address", "token_address"]
    norm_map = {norm_col(c): c for c in df.columns}
    for c in candidates:
        if norm_col(c) in norm_map:
            return norm_map[norm_col(c)]
    return None


def detect_target_row_col(df: pd.DataFrame) -> str | None:
    norm_map = {norm_col(c): c for c in df.columns}
    for c in ["target_row_id", "targetRowId"]:
        if norm_col(c) in norm_map:
            return norm_map[norm_col(c)]
    return None


def find_feature_case_insensitive(df: pd.DataFrame, feature: str) -> str | None:
    target = norm_col(feature)
    for c in df.columns:
        if norm_col(c) == target:
            return c
    return None


def search_text_files_for_terms(root: Path, terms: list[str], max_file_mb: float = 8.0) -> pd.DataFrame:
    rows = []
    if not root.exists():
        return pd.DataFrame(rows)

    suffixes = {".py", ".md", ".txt", ".json", ".csv", ".jsonl", ".yaml", ".yml"}
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in suffixes:
            continue
        try:
            if p.stat().st_size > max_file_mb * 1024 * 1024:
                continue
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        for i, line in enumerate(txt.splitlines(), start=1):
            lower = line.lower()
            for term in terms:
                if term.lower() in lower:
                    rows.append({
                        "path": str(p),
                        "line_number": i,
                        "term": term,
                        "line_snippet": line.strip()[:500],
                    })
                    break

    return pd.DataFrame(rows)


def search_repo_source_terms(terms: list[str]) -> pd.DataFrame:
    rows = []
    for d in SOURCE_SCAN_DIRS:
        base = ROOT / d
        if base.exists():
            df = search_text_files_for_terms(base, terms=terms, max_file_mb=4.0)
            if not df.empty:
                rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["path", "line_number", "term", "line_snippet"])
    return pd.concat(rows, ignore_index=True)


def load_e9b_candidate_feature(e9b_decision: dict[str, Any], feature: str) -> dict[str, Any]:
    for item in e9b_decision.get("candidate_features", []) or []:
        if norm_col(item.get("feature_name", "")) == norm_col(feature):
            return item
    return {}


def build_matched_pair_direction_audit(
    pairs: pd.DataFrame,
    positives: pd.DataFrame,
    controls: pd.DataFrame,
    feature_col: str,
) -> tuple[pd.DataFrame, str]:
    if pairs.empty:
        return pd.DataFrame([{"audit_status": "SKIPPED_NO_MATCHED_PAIRS"}]), "SKIPPED_NO_MATCHED_PAIRS"

    pos_ref = None
    ctrl_ref = None
    for c in pairs.columns:
        n = norm_col(c)
        if pos_ref is None and "positive" in n and "target_row_id" in n:
            pos_ref = c
        if ctrl_ref is None and "control" in n and "target_row_id" in n:
            ctrl_ref = c

    if pos_ref is None or ctrl_ref is None:
        return pd.DataFrame([{"audit_status": "SKIPPED_COULD_NOT_FIND_POSITIVE_CONTROL_TARGET_ROW_REFS"}]), "SKIPPED_COULD_NOT_FIND_REFS"

    pos_key = detect_target_row_col(positives)
    ctrl_key = detect_target_row_col(controls)
    if not pos_key or not ctrl_key:
        return pd.DataFrame([{"audit_status": "SKIPPED_COULD_NOT_FIND_TARGET_ROW_ID_IN_POS_OR_CTRL"}]), "SKIPPED_NO_TARGET_ROW_KEYS"

    pos_m = positives[[pos_key, feature_col]].copy()
    ctrl_m = controls[[ctrl_key, feature_col]].copy()
    pos_m["_pos_ref"] = pos_m[pos_key].astype(str)
    ctrl_m["_ctrl_ref"] = ctrl_m[ctrl_key].astype(str)

    merged = pairs.copy()
    merged["_pos_ref"] = merged[pos_ref].astype(str)
    merged["_ctrl_ref"] = merged[ctrl_ref].astype(str)

    merged = merged.merge(
        pos_m[["_pos_ref", feature_col]].rename(columns={feature_col: "positive_feature_value"}),
        on="_pos_ref",
        how="left",
    )
    merged = merged.merge(
        ctrl_m[["_ctrl_ref", feature_col]].rename(columns={feature_col: "control_feature_value"}),
        on="_ctrl_ref",
        how="left",
    )

    p = coerce_numeric(merged["positive_feature_value"])
    c = coerce_numeric(merged["control_feature_value"])
    valid = p.notna() & c.notna()
    if valid.sum() == 0:
        return pd.DataFrame([{"audit_status": "NO_VALID_NUMERIC_MATCHED_PAIR_VALUES"}]), "NO_VALID_NUMERIC_MATCHED_PAIR_VALUES"

    delta = p[valid] - c[valid]
    out = pd.DataFrame([{
        "feature_name": feature_col,
        "matched_pair_rows": int(len(merged)),
        "valid_numeric_pairs": int(valid.sum()),
        "median_positive_minus_control": float(delta.median()),
        "mean_positive_minus_control": float(delta.mean()),
        "positive_gt_control_share": float((delta > 0).mean()),
        "positive_lt_control_share": float((delta < 0).mean()),
        "positive_eq_control_share": float((delta == 0).mean()),
        "direction": "POSITIVE_LOWER_THAN_CONTROL" if float(delta.median()) < 0 else "POSITIVE_HIGHER_THAN_CONTROL" if float(delta.median()) > 0 else "NO_MEDIAN_DIRECTION",
        "audit_status": "OK",
    }])
    return out, "OK"


def main() -> int:
    parser = argparse.ArgumentParser(description="E9C Whale-Score Context Feature Contract Audit")
    parser.add_argument("--e9a-root", type=Path, default=DEFAULT_E9A_ROOT)
    parser.add_argument("--e9b-root", type=Path, default=DEFAULT_E9B_ROOT)
    parser.add_argument("--e8-root", type=Path, default=DEFAULT_E8_ROOT)
    parser.add_argument("--feature", default="whale_score_asof")
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    output_root = args.output_root or (
        ROOT / "data" / "training" / "manual_verified_results" / f"phase_e9c_whale_score_contract_{timestamp_slug()}"
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
    e9b_manifest = read_json_optional(args.e9b_root / "reports" / "e9b_manifest.json")
    e9b_feature_scope = read_csv_optional(args.e9b_root / "data" / "e9b_feature_scope.csv")
    e9b_matrix = read_csv_optional(args.e9b_root / "data" / "e9b_feature_discrimination_matrix.csv")
    e9b_missingness = read_csv_optional(args.e9b_root / "data" / "e9b_feature_missingness_summary.csv")
    e9b_top_pair = read_csv_optional(args.e9b_root / "audits" / "e9b_top_pair_removal_audit.csv")
    e9b_leave_one = read_csv_optional(args.e9b_root / "audits" / "e9b_leave_one_pair_out_audit.csv")
    e9b_leakage = read_csv_optional(args.e9b_root / "audits" / "e9b_label_leakage_audit.csv")

    feature_pos = find_feature_case_insensitive(positives, args.feature)
    feature_ctrl = find_feature_case_insensitive(controls, args.feature)

    blockers = []
    warnings = []

    if not feature_pos:
        blockers.append("FEATURE_MISSING_FROM_POSITIVE_ROWS")
    if not feature_ctrl:
        blockers.append("FEATURE_MISSING_FROM_CONTROL_ROWS")
    if forbidden_name(args.feature):
        blockers.append("FEATURE_NAME_MATCHES_FORBIDDEN_LEAKAGE_PATTERN")

    feature_col = feature_pos or feature_ctrl or args.feature

    distribution_rows = []
    if feature_pos and feature_ctrl:
        pos_stats = safe_stats(positives[feature_pos])
        ctrl_stats = safe_stats(controls[feature_ctrl])

        distribution_rows.append({"group": "positive", "feature_name": feature_col, **pos_stats})
        distribution_rows.append({"group": "control", "feature_name": feature_col, **ctrl_stats})

        missing_diff = None
        if pos_stats["missing_rate"] is not None and ctrl_stats["missing_rate"] is not None:
            missing_diff = abs(pos_stats["missing_rate"] - ctrl_stats["missing_rate"])
            if missing_diff > 0.05:
                warnings.append("FEATURE_MISSINGNESS_DIFF_ABOVE_5_PERCENT")

        cd = cliffs_delta(positives[feature_pos], controls[feature_ctrl])
        direction = "POSITIVE_LOWER_THAN_CONTROL" if cd is not None and cd < 0 else "POSITIVE_HIGHER_THAN_CONTROL" if cd is not None and cd > 0 else "NO_DIRECTION"

        direction_df = pd.DataFrame([{
            "feature_name": feature_col,
            "positive_median": pos_stats["median"],
            "control_median": ctrl_stats["median"],
            "median_diff_positive_minus_control": None if pos_stats["median"] is None or ctrl_stats["median"] is None else pos_stats["median"] - ctrl_stats["median"],
            "positive_mean": pos_stats["mean"],
            "control_mean": ctrl_stats["mean"],
            "mean_diff_positive_minus_control": None if pos_stats["mean"] is None or ctrl_stats["mean"] is None else pos_stats["mean"] - ctrl_stats["mean"],
            "cliffs_delta": cd,
            "abs_cliffs_delta": None if cd is None else abs(cd),
            "direction": direction,
            "missing_rate_diff_abs": missing_diff,
        }])
    else:
        direction_df = pd.DataFrame([{
            "feature_name": args.feature,
            "audit_status": "FEATURE_MISSING_IN_POSITIVE_OR_CONTROL_ROWS",
        }])

    distribution_df = pd.DataFrame(distribution_rows)
    write_csv(data_dir / "e9c_whale_score_distribution_by_group.csv", distribution_df)
    write_csv(data_dir / "e9c_whale_score_directionality_audit.csv", direction_df)

    pair_col = detect_pair_col(positives)
    unique_positive_pairs = int(positives[pair_col].astype(str).nunique()) if pair_col and pair_col in positives.columns else None
    top_positive_pair_share = None
    if pair_col and pair_col in positives.columns and len(positives):
        vc = positives[pair_col].astype(str).value_counts()
        top_positive_pair_share = float(vc.iloc[0] / len(positives)) if len(vc) else None

    matched_direction_df, matched_status = build_matched_pair_direction_audit(
        matched_pairs,
        positives,
        controls,
        feature_col=feature_col if feature_pos and feature_ctrl else args.feature,
    )
    write_csv(data_dir / "e9c_whale_score_matched_pair_direction_audit.csv", matched_direction_df)

    # E9B lineage extraction for this feature.
    e9b_candidate = load_e9b_candidate_feature(e9b_decision, args.feature)

    scope_rows = []
    if not e9b_feature_scope.empty and "feature_name" in e9b_feature_scope.columns:
        scope_rows = e9b_feature_scope[
            e9b_feature_scope["feature_name"].astype(str).str.lower() == args.feature.lower()
        ].to_dict("records")

    matrix_rows = []
    if not e9b_matrix.empty and "feature_name" in e9b_matrix.columns:
        matrix_rows = e9b_matrix[
            e9b_matrix["feature_name"].astype(str).str.lower() == args.feature.lower()
        ].to_dict("records")

    missing_rows = []
    if not e9b_missingness.empty and "feature_name" in e9b_missingness.columns:
        missing_rows = e9b_missingness[
            e9b_missingness["feature_name"].astype(str).str.lower() == args.feature.lower()
        ].to_dict("records")

    top_pair_rows = []
    if not e9b_top_pair.empty and "feature_name" in e9b_top_pair.columns:
        top_pair_rows = e9b_top_pair[
            e9b_top_pair["feature_name"].astype(str).str.lower() == args.feature.lower()
        ].to_dict("records")

    leave_one_rows = []
    if not e9b_leave_one.empty and "feature_name" in e9b_leave_one.columns:
        leave_one_rows = e9b_leave_one[
            e9b_leave_one["feature_name"].astype(str).str.lower() == args.feature.lower()
        ].to_dict("records")

    source_trace_e8 = search_text_files_for_terms(args.e8_root, [args.feature, "whale_score"], max_file_mb=8.0)
    source_trace_repo = search_repo_source_terms([args.feature, "whale_score", "compute_whale_score"])
    write_csv(data_dir / "e9c_e8_source_trace.csv", source_trace_e8)
    write_csv(data_dir / "e9c_repo_source_trace.csv", source_trace_repo)

    leakage_rows = []
    if not e9b_leakage.empty:
        colname = None
        for c in e9b_leakage.columns:
            if norm_col(c) in {"column", "feature_name"}:
                colname = c
                break
        if colname:
            leakage_rows = e9b_leakage[
                e9b_leakage[colname].astype(str).str.lower() == args.feature.lower()
            ].to_dict("records")

    leakage_audit = pd.DataFrame([{
        "feature_name": args.feature,
        "forbidden_name_pattern": forbidden_name(args.feature),
        "present_in_e9b_leakage_audit": bool(leakage_rows),
        "e9b_leakage_rows_json": json.dumps(leakage_rows, ensure_ascii=False),
        "audit_status": "PASS_NO_NAME_BASED_LEAKAGE" if not forbidden_name(args.feature) else "FAIL_FORBIDDEN_NAME_PATTERN",
    }])
    write_csv(audits_dir / "e9c_whale_score_leakage_name_audit.csv", leakage_audit)

    asof_safety = pd.DataFrame([{
        "feature_name": args.feature,
        "has_asof_suffix": args.feature.lower().endswith("_asof"),
        "present_in_positive_rows": bool(feature_pos),
        "present_in_control_rows": bool(feature_ctrl),
        "present_in_e9b_candidate_features": bool(e9b_candidate),
        "present_in_e9b_feature_scope": bool(scope_rows),
        "feature_scope_fallback_used_in_e9b": bool(e9b_decision.get("feature_scope_fallback_used", None)),
        "e8_trace_rows_found": int(len(source_trace_e8)),
        "repo_trace_rows_found": int(len(source_trace_repo)),
        "asof_status": (
            "PASS_RESEARCH_ASOF_CONTRACT"
            if args.feature.lower().endswith("_asof")
            and feature_pos and feature_ctrl
            and bool(e9b_candidate)
            and not forbidden_name(args.feature)
            else "WEAK_OR_INCOMPLETE_ASOF_CONTRACT"
        ),
    }])
    write_csv(audits_dir / "e9c_asof_safety_audit.csv", asof_safety)

    runtime_feasibility = pd.DataFrame([{
        "feature_name": args.feature,
        "requires_external_api_for_e9c": False,
        "available_in_e9a_artifact": bool(feature_pos and feature_ctrl),
        "existing_repo_whale_score_code_found": bool(len(source_trace_repo)),
        "compute_cost_estimate": "LOW_IF_COMPUTED_FROM_EXISTING_SNAPSHOT_FIELDS",
        "runtime_status": "NOT_RUNTIME_APPROVED",
        "research_status": "MAY_BE_USED_FOR_OFFLINE_RESEARCH_ONLY" if feature_pos and feature_ctrl else "BLOCKED_FEATURE_MISSING",
    }])
    write_csv(data_dir / "e9c_runtime_feasibility_audit.csv", runtime_feasibility)

    # Decision gate
    e9b_decision_name = e9b_decision.get("decision")
    e9a_weak = bool(
        (e9b_decision.get("e9a_weak_contract_details") or {}).get("weak_contract_inferred", True)
    )

    if e9b_decision_name != "E9B_RESEARCH_ONLY_FEATURE_CANDIDATES":
        warnings.append(f"UNEXPECTED_E9B_DECISION_{e9b_decision_name}")

    if not e9b_candidate:
        blockers.append("FEATURE_NOT_LISTED_AS_E9B_CANDIDATE")

    if e9a_weak:
        warnings.append("E9A_WEAK_CONTROL_CONTRACT_CAPS_E9C_AT_RESEARCH_ONLY")

    if unique_positive_pairs is not None and unique_positive_pairs < 5:
        warnings.append("UNIQUE_POSITIVE_PAIRS_BELOW_5")

    if top_positive_pair_share is not None and top_positive_pair_share > 0.35:
        warnings.append("TOP_POSITIVE_PAIR_SHARE_HIGH")

    if not bool(scope_rows):
        warnings.append("FEATURE_SCOPE_ROW_NOT_FOUND_IN_E9B_FEATURE_SCOPE")

    if len(blockers) > 0:
        decision_name = "E9C_FAIL_WHALE_SCORE_CONTRACT"
        approved_for_e9d = False
    else:
        decision_name = "E9C_RESEARCH_ONLY_WHALE_SCORE_CONTRACT_CANDIDATE"
        approved_for_e9d = True

    decision = {
        "decision": decision_name,
        "feature_name": args.feature,
        "blockers": blockers,
        "warnings": warnings,
        "approved_for_e9d_non_training_rule_prototype": approved_for_e9d,
        "approved_for_modeling": False,
        "approved_for_training": False,
        "approved_for_runtime": False,
        "approved_for_ui": False,
        "approved_for_trading": False,
        "reason": (
            "Feature appears usable for offline non-training context rule research only."
            if approved_for_e9d
            else "Feature contract failed; do not proceed to E9D."
        ),
        "e9b_candidate": e9b_candidate,
        "e9b_decision": e9b_decision_name,
        "e9a_weak_contract": e9a_weak,
        "positive_rows": int(len(positives)),
        "control_rows": int(len(controls)),
        "matched_pairs": int(len(matched_pairs)),
        "unique_positive_pairs": unique_positive_pairs,
        "top_positive_pair_share": top_positive_pair_share,
        "matched_pair_direction_status": matched_status,
        "created_at_utc": utc_now_iso(),
    }
    write_json(audits_dir / "e9c_decision_gate.json", decision)

    contract = pd.DataFrame([{
        "feature_name": args.feature,
        "status": decision_name,
        "present_in_positives": bool(feature_pos),
        "present_in_controls": bool(feature_ctrl),
        "numeric_in_positives": bool(feature_pos and coerce_numeric(positives[feature_pos]).notna().sum() > 0),
        "numeric_in_controls": bool(feature_ctrl and coerce_numeric(controls[feature_ctrl]).notna().sum() > 0),
        "forbidden_name_pattern": forbidden_name(args.feature),
        "e9b_candidate_feature": bool(e9b_candidate),
        "e9b_feature_scope_rows": len(scope_rows),
        "e8_trace_rows": len(source_trace_e8),
        "repo_trace_rows": len(source_trace_repo),
        "e9a_weak_contract": e9a_weak,
        "approved_for_e9d_non_training_rule_prototype": approved_for_e9d,
        "approved_for_modeling": False,
        "approved_for_runtime": False,
    }])
    write_csv(data_dir / "e9c_whale_score_contract.csv", contract)

    lineage = {
        "feature_name": args.feature,
        "e9b_candidate": e9b_candidate,
        "e9b_feature_scope_rows": scope_rows,
        "e9b_discrimination_matrix_rows": matrix_rows,
        "e9b_missingness_rows": missing_rows,
        "e9b_top_pair_rows": top_pair_rows,
        "e9b_leave_one_rows_count": len(leave_one_rows),
        "e9b_manifest_decision": e9b_manifest.get("decision"),
    }
    write_json(data_dir / "e9c_whale_score_lineage.json", lineage)

    manifest = {
        "phase": "E9C",
        "branch_name": "phase_e9c_whale_score_context_feature_contract",
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
            "e8_root": str(args.e8_root),
            "feature": args.feature,
        },
        "outputs": {
            "output_root": str(output_root),
            "manifest": str(reports_dir / "e9c_manifest.json"),
            "summary": str(reports_dir / "e9c_summary_for_upload.txt"),
            "contract": str(data_dir / "e9c_whale_score_contract.csv"),
            "distribution": str(data_dir / "e9c_whale_score_distribution_by_group.csv"),
            "directionality": str(data_dir / "e9c_whale_score_directionality_audit.csv"),
            "matched_pair_direction": str(data_dir / "e9c_whale_score_matched_pair_direction_audit.csv"),
            "lineage": str(data_dir / "e9c_whale_score_lineage.json"),
            "e8_source_trace": str(data_dir / "e9c_e8_source_trace.csv"),
            "repo_source_trace": str(data_dir / "e9c_repo_source_trace.csv"),
            "runtime_feasibility": str(data_dir / "e9c_runtime_feasibility_audit.csv"),
            "asof_safety": str(audits_dir / "e9c_asof_safety_audit.csv"),
            "leakage_name_audit": str(audits_dir / "e9c_whale_score_leakage_name_audit.csv"),
            "decision_gate": str(audits_dir / "e9c_decision_gate.json"),
        },
        "decision": decision_name,
    }
    write_json(reports_dir / "e9c_manifest.json", manifest)

    summary_lines = [
        "Phase / branch name",
        "",
        "E9C — Whale-Score Context Feature Contract Audit",
        "",
        "Run status",
        "",
        "COMPLETED",
        "",
        "Decision",
        "",
        decision_name,
        "",
        "Scope",
        "",
        "Offline read-only contract/feasibility audit for whale_score_asof.",
        "No model training, no runtime, no UI, no trading, no SQLite writes, no external APIs, no LLM calls.",
        "",
        "Key results",
        "",
        f"- feature: {args.feature}",
        f"- present in positives: {bool(feature_pos)}",
        f"- present in controls: {bool(feature_ctrl)}",
        f"- E9B candidate feature: {bool(e9b_candidate)}",
        f"- E9A weak contract: {e9a_weak}",
        f"- unique positive pairs: {unique_positive_pairs}",
        f"- top positive pair share: {top_positive_pair_share}",
        f"- approved for E9D non-training prototype: {approved_for_e9d}",
        f"- approved for modeling: False",
        f"- approved for runtime: False",
        "",
        "Directionality",
        "",
    ]

    if not direction_df.empty and "direction" in direction_df.columns:
        row = direction_df.iloc[0].to_dict()
        summary_lines.extend([
            f"- direction: {row.get('direction')}",
            f"- cliffs_delta: {row.get('cliffs_delta')}",
            f"- median positive-control diff: {row.get('median_diff_positive_minus_control')}",
        ])
    else:
        summary_lines.append("- direction: unavailable")

    summary_lines.extend([
        "",
        "Warnings",
        "",
    ])
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
    if blockers:
        for b in blockers:
            summary_lines.append(f"- {b}")
    else:
        summary_lines.append("None.")

    summary_lines.extend([
        "",
        "Final interpretation",
        "",
    ])
    if approved_for_e9d:
        summary_lines.append("whale_score_asof may proceed to E9D as an offline non-training rule/composite prototype only.")
        summary_lines.append("It does not approve modeling, training, runtime, UI, or trading.")
    else:
        summary_lines.append("whale_score_asof does not have a sufficient contract to proceed.")

    (reports_dir / "e9c_summary_for_upload.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    print(json.dumps({
        "status": "completed",
        "phase": "E9C",
        "output_root": str(output_root),
        "decision": decision_name,
        "approved_for_e9d_non_training_rule_prototype": approved_for_e9d,
        "approved_for_modeling": False,
        "approved_for_runtime": False,
        "summary": str(reports_dir / "e9c_summary_for_upload.txt"),
        "decision_gate": str(audits_dir / "e9c_decision_gate.json"),
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
