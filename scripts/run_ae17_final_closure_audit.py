#!/usr/bin/env python3
"""AE17 final roadmap closure audit (Path B: explicit meta-combination).

Read-only consumer of a durable AE17 real-meta evidence output root.
Does NOT: start AE18/AE19, train/fit, backtest, mutate trader.db, call LLMs,
call external APIs/Helius/Solana, connect wallet, enable live trading, or claim
profitability / live readiness.

Does NOT accept prior tier-only outputs as sufficient AE17 closure.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE = "AE17_FINAL_CLOSURE_AUDIT"
CONTEXT_STATUS = "AE17_CONTEXT_NOT_AVAILABLE_PENDING_AE18"
CONTEXT_MISSINGNESS_REASON = "AE18_CONTEXT_LAYER_NOT_STARTED"
CONTEXT_SCORE_WEIGHT = 0.0
META_LAYER_TYPE = "NON_LEARNED_EXPLICIT_META_COMBINATION"
META_FORMULA_VERSION = "AE17_EXPLICIT_META_COMBINATION_V1"

CLASSIFICATION_PASS = "AE17_PASS_WITH_NOTES_EXPLICIT_META_COMBINATION"
CLASSIFICATION_SUBSTANCE = "AE17_INCOMPLETE_META_STACKING_SUBSTANCE"
CLASSIFICATION_CONTEXT = "AE17_INCOMPLETE_CONTEXT_MISSINGNESS_GAP"
CLASSIFICATION_LOOKAHEAD = "AE17_INCOMPLETE_LOOKAHEAD_GAP"
CLASSIFICATION_FEATURE = "AE17_INCOMPLETE_FEATURE_PARITY_GAP"
CLASSIFICATION_LINEAGE = "AE17_INCOMPLETE_LINEAGE_GAP"
CLASSIFICATION_AUTHORITY = "AE17_INCOMPLETE_AUTHORITY_GAP"
CLASSIFICATION_TRAINING = "AE17_BLOCKED_TRAINING_REQUIRED"
CLASSIFICATION_INCOMPLETE = "AE17_INCOMPLETE"

PASS_NOTES = [
    "AE17 is a non-learned explicit meta-combination layer (Path B).",
    "This is not a learned trained stacking model.",
    "This is not profitability proof.",
    "This is not live readiness.",
    "This is not execution authority.",
    "Context is explicitly missing and pending AE18.",
    "Large source files were skipped by default and audited as source warnings.",
    (
        "Historical E5 selected_trades evidence is not the same as Clean Forward "
        "live forward validation; that belongs to AE20."
    ),
    "Baseline tier-only scores are retained only for comparison.",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def relpath(path: Path, root: Path = ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            return next(reader)
        except StopIteration:
            return []


def count_csv_rows(path: Path) -> int:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def sample_csv_rows(path: Path, limit: int = 5) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            if i >= limit:
                break
            rows.append(row)
        return rows


def latest_ae17_output_root(project_root: Path) -> Path | None:
    pattern = "data/audits/ae17_real_meta_evidence_run_*"
    matches = sorted(
        [p for p in project_root.glob(pattern) if p.is_dir()],
        key=lambda p: p.name,
    )
    return matches[-1] if matches else None


def checklist_item(
    requirement_id: str,
    requirement: str,
    implementation_status: str,
    exact_evidence: dict[str, Any] | list[Any] | str,
    verified: bool,
    remaining_gap: str,
) -> dict[str, Any]:
    return {
        "id": requirement_id,
        "REQUIREMENT": requirement,
        "IMPLEMENTATION_STATUS": implementation_status,
        "EXACT_EVIDENCE": exact_evidence,
        "VERIFIED": "YES" if verified else "NO",
        "REMAINING_GAP": remaining_gap if not verified else "none",
    }


def _as_bool_false(value: Any) -> bool:
    return value in (False, "False", "false", 0, "0")


def _as_bool_true(value: Any) -> bool:
    return value in (True, "True", "true", 1, "1")


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def audit_ae17_01(
    *,
    runner_path: Path,
    ae17_root: Path,
    feature_path: Path,
    outputs_path: Path,
    manifest: dict[str, Any],
    decision_gate: dict[str, Any],
    formula_audit: dict[str, Any] | None,
    baseline_comparison: dict[str, Any] | None,
) -> dict[str, Any]:
    feature_rows = count_csv_rows(feature_path) if feature_path.exists() else 0
    output_rows = count_csv_rows(outputs_path) if outputs_path.exists() else 0
    feature_cols = read_csv_header(feature_path) if feature_path.exists() else []
    sample = sample_csv_rows(feature_path, limit=3) if feature_path.exists() else []
    decision_dist = (
        decision_gate.get("decision_distribution")
        or manifest.get("decision_distribution")
        or {}
    )

    meta_layer_ok = (
        manifest.get("meta_layer_type") == META_LAYER_TYPE
        and decision_gate.get("meta_layer_type") == META_LAYER_TYPE
        and all(r.get("meta_layer_type") == META_LAYER_TYPE for r in sample)
    )
    formula_version_ok = (
        manifest.get("meta_formula_version") == META_FORMULA_VERSION
        and decision_gate.get("meta_formula_version") == META_FORMULA_VERSION
    )
    flags_ok = all(
        [
            _as_bool_true(r.get("numeric_scores_used_in_meta_score"))
            and _as_bool_true(r.get("votes_used_in_meta_score"))
            and _as_bool_true(r.get("consensus_feature_used_in_meta_score"))
            and _as_bool_true(r.get("context_missingness_used_in_meta_score"))
            and _as_bool_false(r.get("tier_only_scoring"))
            for r in sample
        ]
    ) if sample else False

    formula_exists = formula_audit is not None
    baseline_exists = baseline_comparison is not None
    historical_differs = bool(
        baseline_comparison
        and baseline_comparison.get("classification")
        == "AE17_EXPLICIT_META_COMBINATION_DIFFERS_FROM_TIER_BASELINE"
    )
    synthetic_ok = bool(
        formula_audit
        and (
            formula_audit.get("synthetic_same_tier_score_sensitivity", {}).get(
                "synthetic_same_tier_score_sensitivity_pass"
            )
            or baseline_comparison
            and baseline_comparison.get("classification")
            == "AE17_EXPLICIT_META_COMBINATION_PROVEN_BY_SYNTHETIC_FORMULA_TESTS_WITH_HISTORICAL_VARIANCE_LIMITATION"
        )
    )
    substance_ok = (
        formula_exists
        and baseline_exists
        and meta_layer_ok
        and formula_version_ok
        and flags_ok
        and (historical_differs or synthetic_ok)
        and formula_audit.get("tier_only_scoring") is False
        and formula_audit.get("classification") != CLASSIFICATION_SUBSTANCE
    )

    evidence = {
        "runner_exists": runner_path.is_file(),
        "output_root_exists": ae17_root.is_dir(),
        "feature_matrix_rows": feature_rows,
        "meta_output_rows": output_rows,
        "meta_score_in_feature_matrix": "meta_score" in feature_cols,
        "meta_decision_in_feature_matrix": "meta_decision" in feature_cols,
        "weighted_model_score_present": "weighted_model_score" in feature_cols,
        "decision_distribution": decision_dist,
        "meta_layer_type": manifest.get("meta_layer_type"),
        "meta_formula_version": manifest.get("meta_formula_version"),
        "tier_only_scoring": sample[0].get("tier_only_scoring") if sample else None,
        "formula_audit_exists": formula_exists,
        "baseline_comparison_exists": baseline_exists,
        "formula_audit_classification": (formula_audit or {}).get("classification"),
        "baseline_comparison_classification": (baseline_comparison or {}).get(
            "classification"
        ),
        "historical_differs_from_baseline": historical_differs,
        "synthetic_formula_sensitivity_pass": synthetic_ok,
    }
    verified = all(
        [
            evidence["runner_exists"],
            evidence["output_root_exists"],
            feature_rows > 0,
            output_rows > 0,
            evidence["meta_score_in_feature_matrix"],
            evidence["meta_decision_in_feature_matrix"],
            bool(decision_dist),
            substance_ok,
        ]
    )
    return checklist_item(
        "AE17-01",
        "Real explicit meta-combination exists and produces meta_score/meta_decision beyond tier-only mapping.",
        "implemented_explicit_meta_combination" if verified else "incomplete_tier_only_or_missing_formula",
        evidence,
        verified,
        "Tier-only outputs, missing formula/baseline audits, or missing explicit meta substance.",
    )


def audit_ae17_02(
    *,
    feature_path: Path,
    outputs_path: Path,
    manifest: dict[str, Any],
    decision_gate: dict[str, Any],
    feature_parity: dict[str, Any],
) -> dict[str, Any]:
    feature_cols = set(read_csv_header(feature_path)) if feature_path.exists() else set()
    output_cols = set(read_csv_header(outputs_path)) if outputs_path.exists() else set()
    sample = sample_csv_rows(feature_path, limit=3) if feature_path.exists() else []
    required_model = [
        "rf_score",
        "rf_vote",
        "xgb_score",
        "xgb_vote",
        "tab_score",
        "tab_vote",
        "scoring_tier",
        "weighted_model_score",
        "vote_ratio",
        "consensus_strength",
    ]
    context_fields = [
        "context_feature_available",
        "context_status",
        "context_missingness_reason",
        "context_score_weight",
        "context_component",
    ]
    model_ok = all(c in feature_cols for c in required_model)
    context_cols_ok = all(c in feature_cols for c in context_fields) and all(
        c in output_cols for c in context_fields
    )
    row_values_ok = True
    for row in sample:
        weight = _as_float(row.get("context_score_weight"))
        if not (
            _as_bool_false(row.get("context_feature_available"))
            and row.get("context_status") == CONTEXT_STATUS
            and row.get("context_missingness_reason") == CONTEXT_MISSINGNESS_REASON
            and weight == CONTEXT_SCORE_WEIGHT
            and _as_float(row.get("context_component")) == 0.0
        ):
            row_values_ok = False
            break
    report_ok = all(
        [
            manifest.get("context_feature_contract_present") is True,
            decision_gate.get("context_feature_contract_present") is True,
            feature_parity.get("context_feature_contract_present") is True,
            _as_bool_false(manifest.get("context_feature_available")),
            _as_bool_false(decision_gate.get("context_feature_available")),
            manifest.get("context_status") == CONTEXT_STATUS,
            decision_gate.get("context_status") == CONTEXT_STATUS,
        ]
    )
    verified = model_ok and context_cols_ok and row_values_ok and report_ok and bool(sample)
    return checklist_item(
        "AE17-02",
        "AE17 consumes RF/XGB/TAB/Consensus/context fields or explicit context-missingness fields.",
        "implemented_with_explicit_context_missingness" if verified else "incomplete",
        {
            "rf_xgb_tab_and_formula_fields_present": model_ok,
            "context_fields_ok": context_cols_ok,
            "context_values_ok": row_values_ok,
            "report_contract_ok": report_ok,
        },
        verified,
        "Context missingness fields missing/inconsistent, or RF/XGB/TAB/formula fields absent.",
    )


def audit_ae17_03(
    *,
    feature_path: Path,
    feature_parity: dict[str, Any],
    no_lookahead: dict[str, Any],
) -> dict[str, Any]:
    feature_cols = set(read_csv_header(feature_path)) if feature_path.exists() else set()
    forbidden = {
        "target_net_profitable",
        "target_net_profitable_x",
        "target_net_profitable_y",
        "outcome_label_value",
        "outcome_label_name",
        "outcome_label_available",
        "sim_net_return",
        "sim_net_return_x",
        "sim_net_return_y",
        "future_return",
        "max_upside",
        "max_drawdown",
        "realized_pnl",
        "pnl",
        "profit",
        "closed_at",
        "exit_status",
    }
    forbidden_present = sorted(feature_cols & forbidden)
    required_explicit = [
        "weighted_model_score",
        "baseline_tier_score",
        "meta_layer_type",
        "tier_only_scoring",
    ]
    explicit_present = all(c in feature_cols for c in required_explicit)
    verified = (
        bool(feature_parity.get("feature_parity_pass"))
        and bool(no_lookahead.get("no_lookahead_pass"))
        and len(forbidden_present) == 0
        and explicit_present
    )
    return checklist_item(
        "AE17-03",
        "Feature parity and no-lookahead audits pass.",
        "pass" if verified else "fail",
        {
            "feature_parity_pass": feature_parity.get("feature_parity_pass"),
            "no_lookahead_pass": no_lookahead.get("no_lookahead_pass"),
            "forbidden_columns_present_in_feature_matrix": forbidden_present,
            "explicit_formula_fields_present": explicit_present,
        },
        verified,
        "Feature parity and/or no-lookahead failed, or forbidden/explicit fields missing.",
    )


def audit_ae17_04(
    *,
    feature_path: Path,
    lineage_path: Path,
    pair_conc: dict[str, Any],
) -> dict[str, Any]:
    feature_cols = set(read_csv_header(feature_path)) if feature_path.exists() else set()
    required = [
        "target_row_id",
        "candidate_id",
        "candidate_policy_id",
        "pair_address",
        "event_timestamp",
    ]
    lineage_exists = lineage_path.is_file()
    lineage_note = ""
    lineage_all_present = False
    if lineage_exists:
        with lineage_path.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        lineage_all_present = bool(rows) and all(
            str(r.get("lineage_fields_present")).lower() in {"true", "1"} for r in rows
        )
        if rows:
            lineage_note = rows[0].get("pair_address_identity_note") or ""
    note_ok = "price-source / pool-observation identity only" in lineage_note or (
        "not tradability proof" in str(pair_conc.get("note") or "")
    )
    verified = (
        all(c in feature_cols for c in required)
        and lineage_exists
        and lineage_all_present
        and note_ok
    )
    return checklist_item(
        "AE17-04",
        "Lineage is preserved for the historical evidence proof.",
        "pass" if verified else "fail",
        {
            "lineage_fields_in_feature_matrix": {c: (c in feature_cols) for c in required},
            "lineage_audit_exists": lineage_exists,
            "lineage_fields_present_all_rows": lineage_all_present,
            "pair_address_identity_note": lineage_note or pair_conc.get("note") or "",
        },
        verified,
        "Lineage fields/audit incomplete or pair_address identity note missing.",
    )


def audit_ae17_05(*, pair_conc: dict[str, Any], pair_path: Path) -> dict[str, Any]:
    exists = pair_path.is_file() and bool(pair_conc)
    keys_ok = all(
        k in pair_conc for k in ("total_rows", "unique_pairs", "top_pair_share", "hhi")
    )
    interpretation = {
        "top_pair_share_status": pair_conc.get("top_pair_share_status"),
        "hhi_status": pair_conc.get("hhi_status"),
    }
    verified = exists and keys_ok and any(interpretation.values())
    return checklist_item(
        "AE17-05",
        "Pair-concentration audit exists.",
        "pass" if verified else "fail",
        {
            "pair_concentration_audit_exists": exists,
            "total_rows": pair_conc.get("total_rows"),
            "unique_pairs": pair_conc.get("unique_pairs"),
            "top_pair_share": pair_conc.get("top_pair_share"),
            "hhi": pair_conc.get("hhi"),
            "concentration_interpretation": interpretation,
        },
        verified,
        "Pair concentration audit missing required metrics/interpretation.",
    )


def audit_ae17_06(*, authority: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "trade_authority": False,
        "live_trading_ready": False,
        "paper_demo_only": True,
        "risk_override_authority": False,
        "wallet_access": False,
        "private_key_access": False,
        "db_mutation": False,
        "orders_opened": 0,
        "positions_opened": 0,
        "llm_calls": 0,
        "external_api_calls": 0,
        "training_or_fit": False,
    }
    mismatches = {}
    for k, v in expected.items():
        actual = authority.get(k)
        if actual != v:
            mismatches[k] = {"expected": v, "actual": actual}
    verified = len(mismatches) == 0 and bool(authority)
    return checklist_item(
        "AE17-06",
        "Authority boundary is enforced.",
        "pass" if verified else "fail",
        {"authority": authority, "mismatches": mismatches},
        verified,
        "Authority audit violated expected false/no-mutation/no-training boundary.",
    )


def classify(items: list[dict[str, Any]], *, formula_audit: dict[str, Any] | None) -> str:
    by_id = {i["id"]: i for i in items}
    if by_id["AE17-01"]["VERIFIED"] != "YES":
        return CLASSIFICATION_SUBSTANCE
    if by_id["AE17-02"]["VERIFIED"] != "YES":
        return CLASSIFICATION_CONTEXT
    if by_id["AE17-03"]["VERIFIED"] != "YES":
        evidence = by_id["AE17-03"]["EXACT_EVIDENCE"]
        if not evidence.get("no_lookahead_pass"):
            return CLASSIFICATION_LOOKAHEAD
        if not evidence.get("feature_parity_pass"):
            return CLASSIFICATION_FEATURE
        return CLASSIFICATION_INCOMPLETE
    if by_id["AE17-04"]["VERIFIED"] != "YES":
        return CLASSIFICATION_LINEAGE
    if by_id["AE17-06"]["VERIFIED"] != "YES":
        return CLASSIFICATION_AUTHORITY
    if formula_audit and formula_audit.get("learned_model_used") is True:
        return CLASSIFICATION_TRAINING
    if formula_audit and formula_audit.get("training_or_fit") is True:
        return CLASSIFICATION_TRAINING
    if all(i["VERIFIED"] == "YES" for i in items):
        return CLASSIFICATION_PASS
    return CLASSIFICATION_INCOMPLETE


def run_ae17_final_closure_audit(
    project_root: Path | None = None,
    *,
    ae17_output_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(project_root) if project_root else ROOT
    if ae17_output_root is None:
        ae17_root = latest_ae17_output_root(root)
        if ae17_root is None:
            raise FileNotFoundError(
                "No AE17 output root found under data/audits/ae17_real_meta_evidence_run_*"
            )
    else:
        ae17_root = Path(ae17_output_root)
        if not ae17_root.is_absolute():
            ae17_root = root / ae17_root

    if output_root is None:
        out = root / "data" / "audits" / f"ae17_final_roadmap_closure_audit_{timestamp_tag()}"
    else:
        out = Path(output_root)
        if not out.is_absolute():
            out = root / out

    reports_dir = out / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    feature_path = ae17_root / "data" / "ae17_real_meta_feature_matrix.csv"
    outputs_path = ae17_root / "data" / "ae17_real_meta_outputs.csv"
    manifest_path = ae17_root / "reports" / "ae17_real_meta_manifest.json"
    gate_path = ae17_root / "reports" / "ae17_real_meta_decision_gate.json"
    parity_path = ae17_root / "audits" / "ae17_real_meta_feature_parity_audit.json"
    lookahead_path = ae17_root / "audits" / "ae17_real_meta_no_lookahead_audit.json"
    score_path = ae17_root / "audits" / "ae17_real_meta_score_integrity_audit.json"
    lineage_path = ae17_root / "audits" / "ae17_real_meta_lineage_audit.csv"
    pair_path = ae17_root / "audits" / "ae17_real_meta_pair_concentration_audit.json"
    semantic_path = ae17_root / "audits" / "ae17_real_meta_partial_evidence_semantic_audit.json"
    authority_path = ae17_root / "audits" / "ae17_real_meta_authority_audit.json"
    null_path = ae17_root / "audits" / "ae17_real_meta_null_safety_audit.json"
    formula_path = ae17_root / "audits" / "ae17_real_meta_formula_audit.json"
    baseline_path = ae17_root / "audits" / "ae17_real_meta_baseline_comparison_audit.json"
    runner_path = root / "scripts" / "run_ae17_real_meta_evidence.py"

    required_files = [
        feature_path,
        outputs_path,
        manifest_path,
        gate_path,
        parity_path,
        lookahead_path,
        score_path,
        lineage_path,
        pair_path,
        semantic_path,
        authority_path,
        null_path,
        formula_path,
        baseline_path,
    ]
    missing = [relpath(p, root) for p in required_files if not p.exists()]
    if missing:
        # Missing formula/baseline on old tier-only packages => substance incomplete.
        classification = (
            CLASSIFICATION_SUBSTANCE
            if any("formula_audit" in m or "baseline_comparison" in m for m in missing)
            else CLASSIFICATION_INCOMPLETE
        )
        payload = {
            "created_at_utc": utc_now_iso(),
            "classification": classification,
            "ae17_output_root": relpath(ae17_root, root),
            "final_closure_audit_root": relpath(out, root),
            "ae18_started": False,
            "missing_required_inputs": missing,
            "checklist": [],
            "notes": [
                "Mandatory AE17 explicit-meta evidence artifacts missing; cannot close AE17.",
                "Tier-only packages without formula/baseline audits are AE17_INCOMPLETE_META_STACKING_SUBSTANCE.",
            ],
        }
        write_json(reports_dir / "ae17_final_roadmap_closure_audit.json", payload)
        write_text(
            reports_dir / "ae17_final_roadmap_closure_summary.txt",
            f"AE17 FINAL ROADMAP CLOSURE AUDIT\n\nclassification={classification}\n"
            f"missing={missing}\nae18_started=false\n",
        )
        return payload

    manifest = load_json(manifest_path)
    decision_gate = load_json(gate_path)
    feature_parity = load_json(parity_path)
    no_lookahead = load_json(lookahead_path)
    score_integrity = load_json(score_path)
    pair_conc = load_json(pair_path)
    semantic = load_json(semantic_path)
    authority = load_json(authority_path)
    null_safety = load_json(null_path)
    formula_audit = load_json(formula_path)
    baseline_comparison = load_json(baseline_path)

    items = [
        audit_ae17_01(
            runner_path=runner_path,
            ae17_root=ae17_root,
            feature_path=feature_path,
            outputs_path=outputs_path,
            manifest=manifest,
            decision_gate=decision_gate,
            formula_audit=formula_audit,
            baseline_comparison=baseline_comparison,
        ),
        audit_ae17_02(
            feature_path=feature_path,
            outputs_path=outputs_path,
            manifest=manifest,
            decision_gate=decision_gate,
            feature_parity=feature_parity,
        ),
        audit_ae17_03(
            feature_path=feature_path,
            feature_parity=feature_parity,
            no_lookahead=no_lookahead,
        ),
        audit_ae17_04(
            feature_path=feature_path,
            lineage_path=lineage_path,
            pair_conc=pair_conc,
        ),
        audit_ae17_05(pair_conc=pair_conc, pair_path=pair_path),
        audit_ae17_06(authority=authority),
    ]
    classification = classify(items, formula_audit=formula_audit)
    all_verified = all(i["VERIFIED"] == "YES" for i in items)

    payload = {
        "created_at_utc": utc_now_iso(),
        "classification": classification,
        "ae17_output_root": relpath(ae17_root, root),
        "final_closure_audit_root": relpath(out, root),
        "ae17_closed": all_verified and classification == CLASSIFICATION_PASS,
        "ae18_started": False,
        "ae19_started": False,
        "rows_processed": decision_gate.get("rows_processed") or manifest.get("rows_processed"),
        "feature_matrix_rows": decision_gate.get("feature_matrix_rows")
        or manifest.get("feature_matrix_rows"),
        "meta_output_rows": decision_gate.get("meta_output_rows")
        or manifest.get("meta_output_rows"),
        "decision_distribution": decision_gate.get("decision_distribution")
        or manifest.get("decision_distribution"),
        "baseline_decision_distribution": decision_gate.get("baseline_decision_distribution")
        or manifest.get("baseline_decision_distribution"),
        "tier_distribution": decision_gate.get("tier_distribution")
        or manifest.get("tier_distribution"),
        "partial_evidence_distribution": decision_gate.get("partial_evidence_distribution")
        or manifest.get("partial_evidence_distribution"),
        "meta_layer_type": META_LAYER_TYPE,
        "meta_formula_version": META_FORMULA_VERSION,
        "tier_only_scoring": False,
        "formula_audit_classification": formula_audit.get("classification"),
        "baseline_comparison_classification": baseline_comparison.get("classification"),
        "same_tier_different_scores_observed": formula_audit.get(
            "same_tier_different_scores_observed"
        ),
        "historical_variance_sufficient": baseline_comparison.get(
            "historical_variance_sufficient"
        ),
        "decision_changed_count": baseline_comparison.get("decision_changed_count"),
        "decision_changed_share": baseline_comparison.get("decision_changed_share"),
        "mean_score_delta": baseline_comparison.get("mean_score_delta"),
        "context_contract": {
            "context_feature_contract_present": True,
            "context_feature_available": False,
            "context_status": CONTEXT_STATUS,
            "context_missingness_reason": CONTEXT_MISSINGNESS_REASON,
            "context_score_weight": CONTEXT_SCORE_WEIGHT,
        },
        "feature_parity_pass": feature_parity.get("feature_parity_pass"),
        "no_lookahead_pass": no_lookahead.get("no_lookahead_pass"),
        "score_integrity_pass": score_integrity.get("score_integrity_pass"),
        "lineage_pass": decision_gate.get("lineage_pass"),
        "pair_concentration": {
            "top_pair_share": pair_conc.get("top_pair_share"),
            "hhi": pair_conc.get("hhi"),
            "unique_pairs": pair_conc.get("unique_pairs"),
            "top_pair_share_status": pair_conc.get("top_pair_share_status"),
            "hhi_status": pair_conc.get("hhi_status"),
        },
        "null_safety": null_safety,
        "authority": authority,
        "partial_evidence_semantic_pass": semantic.get("partial_evidence_semantic_pass"),
        "checklist": items,
        "notes": PASS_NOTES if classification == CLASSIFICATION_PASS else [
            "AE17 final closure incomplete; see checklist VERIFIED=NO items.",
            "AE18 remains not started.",
            "Tier-only outputs are not sufficient for AE17 closure.",
        ],
    }
    write_json(reports_dir / "ae17_final_roadmap_closure_audit.json", payload)

    lines = [
        "AE17 FINAL ROADMAP CLOSURE AUDIT (EXPLICIT META-COMBINATION)",
        "",
        f"AE17 evidence output root:\n{relpath(ae17_root, root)}",
        "",
        f"Final closure audit root:\n{relpath(out, root)}",
        "",
        f"Classification:\n{classification}",
        "",
        f"AE17 closed:\n{payload['ae17_closed']}",
        "",
        "AE18 started:\nfalse",
        "",
        f"meta_layer_type:\n{META_LAYER_TYPE}",
        "",
        f"meta_formula_version:\n{META_FORMULA_VERSION}",
        "",
        "Checklist:",
    ]
    for item in items:
        lines.extend(
            [
                "",
                f"{item['id']}: VERIFIED={item['VERIFIED']}",
                f"  REQUIREMENT: {item['REQUIREMENT']}",
                f"  IMPLEMENTATION_STATUS: {item['IMPLEMENTATION_STATUS']}",
                f"  REMAINING_GAP: {item['REMAINING_GAP']}",
            ]
        )
    lines.extend(["", "Notes:"])
    for note in payload["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    write_text(reports_dir / "ae17_final_roadmap_closure_summary.txt", "\n".join(lines))
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AE17 final roadmap closure audit (explicit meta-combination)"
    )
    p.add_argument(
        "--ae17-output-root",
        type=str,
        default=None,
        help="Default: latest data/audits/ae17_real_meta_evidence_run_*",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Default: data/audits/ae17_final_roadmap_closure_audit_<timestamp>",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_ae17_final_closure_audit(
            ROOT,
            ae17_output_root=args.ae17_output_root,
            output_root=args.output_root,
        )
    except FileNotFoundError as exc:
        print(f"[{PHASE}] controlled blocker: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"[{PHASE}] unexpected error: {exc}", file=sys.stderr)
        return 1

    print(f"[{PHASE}] classification: {result.get('classification')}")
    print(f"[{PHASE}] ae17_output_root: {result.get('ae17_output_root')}")
    print(f"[{PHASE}] final_closure_audit_root: {result.get('final_closure_audit_root')}")
    print(f"[{PHASE}] ae17_closed: {result.get('ae17_closed')}")
    print(f"[{PHASE}] ae18_started: false")
    if str(result.get("classification")).startswith("AE17_INCOMPLETE") or str(
        result.get("classification")
    ).startswith("AE17_BLOCKED"):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
