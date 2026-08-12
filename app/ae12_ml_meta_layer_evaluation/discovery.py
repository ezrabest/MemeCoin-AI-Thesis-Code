"""Read-only artifact discovery for AE12.6."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ArtifactSpec:
    artifact_family: str
    expected_layer: str
    evidence_role: str
    criticality: str  # CRITICAL | IMPORTANT | OPTIONAL | LEGACY_OR_DIAGNOSTIC
    relative_glob: str
    search_roots: tuple[str, ...]
    pick: str = "newest"  # newest | any | exact_subpath
    notes: str = ""


def _mtime_utc(p: Path) -> str:
    if not p.is_file():
        return ""
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()


def _pick_newest(matches: list[Path]) -> Path | None:
    if not matches:
        return None
    return sorted(matches, key=lambda x: x.stat().st_mtime, reverse=True)[0]


def discover_artifact(project_root: Path, spec: ArtifactSpec) -> dict[str, Any]:
    matches: list[Path] = []
    for root_rel in spec.search_roots:
        base = project_root / root_rel
        if not base.is_dir():
            continue
        matches.extend(base.glob(spec.relative_glob))

    chosen: Path | None = None
    if matches:
        if spec.pick == "newest":
            chosen = _pick_newest(matches)
        else:
            chosen = matches[0]

    exists = chosen is not None and chosen.is_file()
    if exists:
        missing_status = "FOUND"
    elif spec.criticality == "CRITICAL":
        missing_status = "MISSING_CRITICAL_ARTIFACT"
    elif spec.criticality == "IMPORTANT":
        missing_status = "MISSING_EXPECTED_ARTIFACT"
    elif spec.criticality in {"OPTIONAL", "LEGACY_OR_DIAGNOSTIC"}:
        missing_status = "MISSING_NONCRITICAL_ARTIFACT"
    else:
        missing_status = "MISSING_EXPECTED_ARTIFACT"

    return {
        "artifact_family": spec.artifact_family,
        "expected_layer": spec.expected_layer,
        "discovered_path": str(chosen.resolve()) if chosen else "",
        "exists": exists,
        "file_type": chosen.suffix.lstrip(".") if chosen else "",
        "last_modified_utc": _mtime_utc(chosen) if chosen else "",
        "evidence_role": spec.evidence_role,
        "criticality": spec.criticality,
        "missing_status": missing_status if not exists else "FOUND",
        "notes": spec.notes,
    }


def build_artifact_specs() -> list[ArtifactSpec]:
    audits = ("data/audits",)
    training = ("data/training/manual_verified_results",)
    training_clean = ("data/training/manual_verified_datasets_clean_for_model",)
    docs = ("docs/msc_final",)
    reports = ("reports",)
    data_root = ("data",)

    return [
        ArtifactSpec(
            "ae12_forward_evidence_summary",
            "Forward evidence maturation (AE12.3)",
            "FORWARD_EVIDENCE",
            "CRITICAL",
            "ae12_forward_evidence_maturation_*/reports/ae12_forward_evidence_summary.json",
            audits,
            notes="Primary no-lookahead forward evidence maturation summary.",
        ),
        ArtifactSpec(
            "ae12_semantic_coverage_reconciliation",
            "Semantic reporting / SentimentFix",
            "SEMANTIC_REPORTING",
            "CRITICAL",
            "ae12_semantic_coverage_reconciliation_*/reports/ae12_semantic_coverage_reconciliation_summary.json",
            audits,
        ),
        ArtifactSpec(
            "ae12_gemini_semantic_adjudication_summary",
            "Semantic reporting / SentimentFix",
            "SEMANTIC_REPORTING",
            "CRITICAL",
            "ae12_gemini_semantic_adjudication_*/reports/ae12_gemini_semantic_adjudication_summary.json",
            audits,
        ),
        ArtifactSpec(
            "ae12_manual_review_drilldown_summary",
            "Semantic reporting / SentimentFix",
            "SEMANTIC_REPORTING",
            "CRITICAL",
            "ae12_sentimentfix_manual_review_drilldown_*/reports/ae12_manual_review_drilldown_summary.json",
            audits,
        ),
        ArtifactSpec(
            "ae12_runtime_data_census",
            "Runtime observability (AE12.1)",
            "PAPER_RUNTIME",
            "IMPORTANT",
            "ae12_runtime_data_census_*/reports/ae12_data_census_summary.json",
            audits,
        ),
        ArtifactSpec(
            "ae12_missed_winners_full",
            "Opportunity capture (AE12.4)",
            "FORWARD_EVIDENCE",
            "IMPORTANT",
            "ae12_forward_evidence_maturation_*/data/ae12_missed_winners_full.csv",
            audits,
        ),
        ArtifactSpec(
            "ae6_consensus_decision_summary",
            "Consensus / DecisionRecord (AE6)",
            "META_LAYER",
            "IMPORTANT",
            "ae6_consensus_decision_layer_*/ae6_consensus_decision_summary.json",
            audits,
        ),
        ArtifactSpec(
            "ae6_decision_records",
            "Consensus / DecisionRecord (AE6)",
            "META_LAYER",
            "IMPORTANT",
            "ae6_decisions_*.jsonl",
            ("data/decision_records",),
            notes="Runtime/persisted AE6 decision JSONL (read-only).",
        ),
        ArtifactSpec(
            "exit_sim_xgb_clean_full_test_policies",
            "XGB clean full exit simulation",
            "EXIT_SIMULATION",
            "IMPORTANT",
            "exit_sim_xgb_full/strict_validation_selected_policies_XGB_CLEAN_FULL_applied_to_test.csv",
            training,
        ),
        ArtifactSpec(
            "exit_sim_xgb_clean_full_summary",
            "XGB clean full exit simulation",
            "MODEL_PERFORMANCE",
            "IMPORTANT",
            "exit_sim_xgb_full/xgb_full_summary_for_upload.txt",
            training,
        ),
        ArtifactSpec(
            "exit_sim_tab_rf_xgb_comparison",
            "Consensus / exit simulation",
            "EXIT_SIMULATION",
            "IMPORTANT",
            "exit_sim_fixed/strict_validation_selected_policies_applied_to_test.csv",
            training,
        ),
        ArtifactSpec(
            "clean_rf_test_policies",
            "RF clean historical (E8B)",
            "MODEL_PERFORMANCE",
            "IMPORTANT",
            "phase_e8b_clean_historical_rf_*/reports/clean_rf_test_applied_selected_policies.csv",
            training,
        ),
        ArtifactSpec(
            "clean_rf_leakage_audit",
            "RF clean historical (E8B)",
            "MODEL_PERFORMANCE",
            "IMPORTANT",
            "phase_e8b_clean_historical_rf_*/reports/clean_rf_leakage_audit.csv",
            training,
        ),
        ArtifactSpec(
            "xgb_clean_full_feature_schemas",
            "XGB clean full",
            "MODEL_PERFORMANCE",
            "OPTIONAL",
            "xgb_clean_full/xgb_features_CLEAN_RAW_ALL_VERIFIED_x2_24h_XGB.json",
            training,
        ),
        ArtifactSpec(
            "e4_direct_target_xgb_rf_policy_test",
            "XGB/RF direct target (E4)",
            "MODEL_PERFORMANCE",
            "IMPORTANT",
            "phase_e4_direct_target_xgb_rf_full_*/policy_evaluation/validation_selected_policies_direct_target_xgb_rf_applied_to_test.csv",
            training,
        ),
        ArtifactSpec(
            "e6r_tabicl_consensus_comparison",
            "TAB / TabICL consensus (E6R)",
            "META_LAYER",
            "IMPORTANT",
            "phase_e6r_tabicl_full_recheck_*/consensus/direct_target_tab_xgb_rf_comparison.csv",
            training,
        ),
        ArtifactSpec(
            "e6r_tabicl_test_policies",
            "TAB / TabICL (E6R)",
            "MODEL_PERFORMANCE",
            "IMPORTANT",
            "phase_e6r_tabicl_full_recheck_*/policy_evaluation/validation_selected_policies_direct_target_tabicl_applied_to_test.csv",
            training,
        ),
        ArtifactSpec(
            "e8e_context_forensics",
            "Context intelligence (E8E forensics)",
            "CONTEXT_LAYER",
            "IMPORTANT",
            "phase_e8e_rare_winner_context_forensics_*/reports/e8e_market_context_by_candidate.csv",
            training,
            notes="Evidence under E8E forensics; not a closed original AE8 runtime layer.",
        ),
        ArtifactSpec(
            "ae11_wallet_safety_audit",
            "Safety / no-wallet (AE11)",
            "SAFETY",
            "IMPORTANT",
            "ae11_wallet_safety_audit.json",
            ("audits", "reports", "data/audits"),
        ),
        ArtifactSpec(
            "ae11_runtime_paper_loop_gate",
            "Paper/demo execution (AE11)",
            "PAPER_RUNTIME",
            "IMPORTANT",
            "ae11_runtime_paper_loop_*/reports/ae11_decision_gate.json",
            audits,
        ),
        ArtifactSpec(
            "ae11_llm_audit_linkage",
            "LLM audit layer (AE9/AE11 linkage)",
            "LLM_AUDIT",
            "IMPORTANT",
            "ae11_llm_audit_linkage_audit.csv",
            ("audits",),
        ),
        ArtifactSpec(
            "ae11_context_availability_audit",
            "Context intelligence availability (AE11)",
            "CONTEXT_LAYER",
            "OPTIONAL",
            "ae11_context_availability_audit.csv",
            ("audits",),
        ),
        ArtifactSpec(
            "ae7_meta_model_stacking",
            "Meta-model / stacking (AE7)",
            "META_LAYER",
            "OPTIONAL",
            "ae7_*/*stack*.json",
            audits + training,
            notes="Original AE7 stacking may be absent; diagnostic only if missing.",
        ),
        ArtifactSpec(
            "ae9_llm_operational_audit",
            "LLM operational audit (AE9)",
            "LLM_AUDIT",
            "OPTIONAL",
            "ae9_*/*audit*.json",
            audits + training,
        ),
        ArtifactSpec(
            "msc_final_docs",
            "MSc final documentation",
            "UI_REPORTING",
            "OPTIONAL",
            "*.md",
            docs,
        ),
        ArtifactSpec(
            "legacy_cluster_diagnostic",
            "Legacy cluster counters (diagnostic)",
            "LEGACY_OR_DIAGNOSTIC",
            "LEGACY_OR_DIAGNOSTIC",
            "ae12_sentimentfix_*/data/ae12_trading_opportunity_state_distribution.csv",
            audits,
            notes="Legacy runtime cluster diagnostics; not final semantic coin authority.",
        ),
    ]


def resolve_maturation_root(project_root: Path, *, preferred_suffix: str = "20260714_235401") -> Path | None:
    audits = project_root / "data" / "audits"
    preferred = audits / f"ae12_forward_evidence_maturation_{preferred_suffix}"
    if (preferred / "reports" / "ae12_forward_evidence_summary.json").is_file():
        return preferred
    mats = sorted(audits.glob("ae12_forward_evidence_maturation_*"), key=lambda p: p.name, reverse=True)
    for m in mats:
        if (m / "reports" / "ae12_forward_evidence_summary.json").is_file():
            return m
    return None


def inventory_by_path(inventory: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {row["artifact_family"]: row for row in inventory}
