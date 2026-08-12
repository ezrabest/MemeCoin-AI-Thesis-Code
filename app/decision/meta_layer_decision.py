"""AE7 FINAL meta-layer decision gate."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.decision.meta_layer_policy import PolicyConfigStatus


AE7_FINAL_PHASE = "AE7_FINAL_META_LAYER"


class MetaLayerFinalStatus(StrEnum):
    READY_FOR_RESEARCH_ONLY_FORWARD_VALIDATION = (
        "AE7_META_LAYER_READY_FOR_RESEARCH_ONLY_FORWARD_VALIDATION"
    )
    NOT_SUPERIOR_TO_RULE_BASELINE = "AE7_META_LAYER_NOT_SUPERIOR_TO_RULE_BASELINE"
    NOT_ROBUST_ENOUGH = "AE7_META_LAYER_NOT_ROBUST_ENOUGH"
    BLOCKED_INSUFFICIENT_DATA = "AE7_META_LAYER_BLOCKED_INSUFFICIENT_DATA"
    BLOCKED_SCHEMA_GAP = "AE7_META_LAYER_BLOCKED_SCHEMA_GAP"
    BLOCKED_LEAKAGE_RISK = "AE7_META_LAYER_BLOCKED_LEAKAGE_RISK"
    BLOCKED_POLICY_CONFIG = "AE7_META_LAYER_BLOCKED_POLICY_CONFIG"
    BLOCKED_WITH_EXACT_REASONS = "AE7_META_LAYER_BLOCKED_WITH_EXACT_REASONS"


def decide_meta_layer(
    *,
    audits: dict[str, Any],
    rule_result: dict[str, Any],
    logistic_result: dict[str, Any],
    calibrated_result: dict[str, Any],
    xgb_result: dict[str, Any],
    robustness: dict[str, Any],
    ablation_findings: dict[str, Any],
    policy_audit: dict[str, Any],
    dataset_summary: dict[str, Any],
) -> dict[str, Any]:
    blocking_reasons: list[str] = []

    if policy_audit.get("policy_config_status") in {
        PolicyConfigStatus.POLICY_CONFIG_MISSING.value,
        PolicyConfigStatus.POLICY_CONFIG_INVALID.value,
        PolicyConfigStatus.POLICY_CONFIG_VALIDATION_FAILED.value,
    }:
        return _decision(
            MetaLayerFinalStatus.BLOCKED_POLICY_CONFIG,
            audits=audits,
            blocking_reasons=[policy_audit.get("policy_config_status", "POLICY_CONFIG_ERROR")],
            best_approach="none",
            rule_result=rule_result,
            logistic_result=logistic_result,
            calibrated_result=calibrated_result,
            xgb_result=xgb_result,
            robustness=robustness,
            ablation_findings=ablation_findings,
            policy_audit=policy_audit,
        )

    if audits.get("leakage_status") != "PASS":
        blocking_reasons.append("leakage_audit_failed")
    if audits.get("target_availability_status") != "PASS":
        blocking_reasons.append("target_unavailable")
    if dataset_summary.get("rows", 0) < 100:
        blocking_reasons.append("insufficient_rows")

    if blocking_reasons:
        status = (
            MetaLayerFinalStatus.BLOCKED_LEAKAGE_RISK
            if "leakage_audit_failed" in blocking_reasons
            else MetaLayerFinalStatus.BLOCKED_INSUFFICIENT_DATA
        )
        if len(blocking_reasons) > 1:
            status = MetaLayerFinalStatus.BLOCKED_WITH_EXACT_REASONS
        return _decision(
            status,
            audits=audits,
            blocking_reasons=blocking_reasons,
            best_approach="none",
            rule_result=rule_result,
            logistic_result=logistic_result,
            calibrated_result=calibrated_result,
            xgb_result=xgb_result,
            robustness=robustness,
            ablation_findings=ablation_findings,
            policy_audit=policy_audit,
        )

    rule_auc = (rule_result.get("metrics") or {}).get("auc")
    candidates = [
        ("rule_based_tier_comparator", rule_auc, rule_result.get("status")),
        (
            "logistic_regression_baseline",
            (logistic_result.get("metrics") or {}).get("auc"),
            logistic_result.get("status"),
        ),
        (
            "calibrated_logistic",
            (calibrated_result.get("metrics") or {}).get("auc"),
            calibrated_result.get("status"),
        ),
        ("xgb_meta_model", (xgb_result.get("metrics") or {}).get("auc"), xgb_result.get("status")),
    ]
    scored = [
        (name, auc) for name, auc, status in candidates if auc is not None and status == "PASS"
    ]
    best_name = "rule_based_tier_comparator"
    best_auc = rule_auc
    if scored:
        best_name, best_auc = max(scored, key=lambda x: x[1] if x[1] is not None else -1)

    if not robustness.get("robustness_pass_flag", False):
        return _decision(
            MetaLayerFinalStatus.NOT_ROBUST_ENOUGH,
            audits=audits,
            blocking_reasons=["robustness_audit_failed"],
            best_approach=best_name,
            rule_result=rule_result,
            logistic_result=logistic_result,
            calibrated_result=calibrated_result,
            xgb_result=xgb_result,
            robustness=robustness,
            ablation_findings=ablation_findings,
            policy_audit=policy_audit,
        )

    if (
        best_name == "rule_based_tier_comparator"
        or best_auc is None
        or (rule_auc is not None and best_auc <= rule_auc + 0.01)
    ):
        return _decision(
            MetaLayerFinalStatus.NOT_SUPERIOR_TO_RULE_BASELINE,
            audits=audits,
            blocking_reasons=[],
            best_approach=best_name,
            rule_result=rule_result,
            logistic_result=logistic_result,
            calibrated_result=calibrated_result,
            xgb_result=xgb_result,
            robustness=robustness,
            ablation_findings=ablation_findings,
            policy_audit=policy_audit,
        )

    return _decision(
        MetaLayerFinalStatus.READY_FOR_RESEARCH_ONLY_FORWARD_VALIDATION,
        audits=audits,
        blocking_reasons=[],
        best_approach=best_name,
        rule_result=rule_result,
        logistic_result=logistic_result,
        calibrated_result=calibrated_result,
        xgb_result=xgb_result,
        robustness=robustness,
        ablation_findings=ablation_findings,
        policy_audit=policy_audit,
    )


def _research_signal_findings(ablation_findings: dict[str, Any]) -> dict[str, Any]:
    whale_delta = None
    context_status = "RESEARCH_SIGNAL_NOT_AVAILABLE_YET"
    whale_status = "RESEARCH_SIGNAL_NOT_AVAILABLE_YET"
    liquidity_status = "RESEARCH_SIGNAL_LOW_CURRENT_INCREMENTAL_VALUE"

    plus_whale = ablation_findings.get("plus_whale_family", {})
    plus_policy = ablation_findings.get("plus_policy_context", {})
    if plus_whale.get("status") == "PASS":
        whale_delta = plus_whale.get("delta_auc_vs_previous_layer")
        if whale_delta and whale_delta > 0.005:
            whale_status = "RESEARCH_SIGNAL_PROMISING_CONTEXT_DEPENDENT"
        elif whale_delta is not None:
            whale_status = "RESEARCH_SIGNAL_LOW_CURRENT_INCREMENTAL_VALUE"
        else:
            whale_status = "RESEARCH_SIGNAL_RETAIN_FOR_FORWARD_COLLECTION"
    if plus_policy.get("delta_auc_vs_previous_layer"):
        liquidity_status = "RESEARCH_SIGNAL_PROMISING_CONTEXT_DEPENDENT"

    return {
        "whale_family": whale_status,
        "rss_context_family": context_status,
        "liquidity_activity_family": liquidity_status,
        "robustness_concentration_family": "RESEARCH_SIGNAL_RETAIN_FOR_FORWARD_COLLECTION",
        "weak_signal_strengthens_in_combination": bool(whale_delta and whale_delta > 0),
    }


def _decision(
    status: MetaLayerFinalStatus,
    *,
    audits: dict[str, Any],
    blocking_reasons: list[str],
    best_approach: str,
    rule_result: dict[str, Any],
    logistic_result: dict[str, Any],
    calibrated_result: dict[str, Any],
    xgb_result: dict[str, Any],
    robustness: dict[str, Any],
    ablation_findings: dict[str, Any],
    policy_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "final_status": status.value,
        "best_approach": best_approach,
        "rule_baseline_metrics": rule_result.get("metrics", {}),
        "logistic_metrics": logistic_result.get("metrics", {}),
        "calibrated_logistic_metrics": calibrated_result.get("metrics", {}),
        "xgb_meta_metrics": xgb_result.get("metrics", {}),
        "selected_meta_feature_families": list(ablation_findings.keys()),
        "research_signal_family_findings": _research_signal_findings(ablation_findings),
        "robustness_summary": robustness,
        "concentration_summary": {
            "top_pair_share": robustness.get("top_pair_share"),
            "unique_pairs": robustness.get("unique_pairs"),
        },
        "leakage_audit_status": audits.get("leakage_status", "UNKNOWN"),
        "policy_config_status": policy_audit.get("policy_config_status"),
        "policy_content_hash_status": policy_audit.get("policy_content_hash", ""),
        "blocking_reasons": blocking_reasons,
        "recommended_next_phase": "AE7C2_RUNTIME_INFERENCE_DRY_RUN_AFTER_EXACT_PARITY",
        "offline_meta_layer_status": (
            "PASS"
            if status == MetaLayerFinalStatus.READY_FOR_RESEARCH_ONLY_FORWARD_VALIDATION
            else "BLOCKED"
        ),
        "runtime_inference_status": "BLOCKED_PENDING_RUNTIME_PARITY_AND_LINEAGE",
        "trading_authorization_status": "NOT_APPROVED",
        "explicit_no_runtime_trading_approval": True,
    }


def write_decision_gate(decision: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2, default=str)
