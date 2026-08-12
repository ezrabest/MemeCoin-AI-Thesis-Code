#!/usr/bin/env python3
"""AE7 FINAL — offline RF/XGB/TAB meta-layer evaluation (research only)."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.decision.meta_layer_audits import (  # noqa: E402
    run_meta_audits,
    write_audit_json,
)
from app.decision.meta_layer_dataset import (  # noqa: E402
    build_meta_dataset,
    save_meta_dataset,
)
from app.decision.meta_layer_decision import (  # noqa: E402
    AE7_FINAL_PHASE,
    MetaLayerFinalStatus,
    decide_meta_layer,
    write_decision_gate,
)
from app.decision.meta_layer_models import (  # noqa: E402
    evaluate_rule_baseline,
    run_robustness_audits,
    run_signal_family_ablations,
    train_calibrated_logistic,
    train_logistic_baseline,
    train_xgb_meta_model,
)
from app.decision.meta_layer_policy import (  # noqa: E402
    PolicyConfigError,
    load_scoring_policy_config_strict,
)


def _default_output_root(project_root: Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return project_root / "data" / "training" / "manual_verified_results" / f"ae7_final_meta_layer_{ts}"


def _ensure_layout(output_root: Path) -> dict[str, Path]:
    paths = {
        "root": output_root,
        "reports": output_root / "reports",
        "data": output_root / "data",
        "models": output_root / "models",
        "audits": output_root / "audits",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def run_ae7_final_meta_layer(
    *,
    project_root: Path,
    output_root: Path | None = None,
    max_rows: int = 50_000,
    include_xgb_meta: bool = False,
    audit_only: bool = True,
    smoke: bool = True,
    full: bool = False,
    scoring_policy_config: Path | None = None,
) -> dict[str, Any]:
    paths = _ensure_layout(output_root or _default_output_root(project_root))
    output_root = paths["root"]

    if full:
        smoke = False
        audit_only = False
        include_xgb_meta = True

    try:
        policy_audit = load_scoring_policy_config_strict(
            scoring_policy_config,
            offline_smoke_mode=smoke and scoring_policy_config is None,
        )
    except PolicyConfigError as exc:
        decision = {
            "phase": AE7_FINAL_PHASE,
            "final_status": MetaLayerFinalStatus.BLOCKED_POLICY_CONFIG.value,
            "policy_config_status": exc.status,
            "blocking_reasons": [exc.reason],
            "runtime_inference_status": "BLOCKED_PENDING_RUNTIME_PARITY_AND_LINEAGE",
            "trading_authorization_status": "NOT_APPROVED",
            "explicit_no_runtime_trading_approval": True,
        }
        write_decision_gate(decision, paths["reports"] / "ae7_final_meta_layer_decision_gate.json")
        return decision

    dataset = build_meta_dataset(
        project_root=project_root,
        max_rows=max_rows,
        policy_audit=policy_audit,
    )
    save_meta_dataset(dataset, paths["data"] / "meta_dataset.parquet")
    write_audit_json(
        dataset.to_summary_dict(),
        paths["audits"] / "meta_dataset_summary.json",
    )

    audits = run_meta_audits(
        frame=dataset.frame,
        feature_columns=dataset.feature_columns,
        target_column=dataset.target_column,
        policy_audit=policy_audit,
    )
    audit_dict = audits.to_dict()
    write_audit_json(audit_dict, paths["audits"] / "meta_layer_audits.json")

    rule_result = evaluate_rule_baseline(dataset.frame, dataset.target_column)
    logistic_result = {"approach": "logistic_regression_baseline", "status": "SKIPPED_AUDIT_ONLY"}
    calibrated_result = {"approach": "calibrated_logistic", "status": "SKIPPED_AUDIT_ONLY"}
    xgb_result = {"approach": "xgb_meta_model", "status": "SKIPPED_AUDIT_ONLY"}
    ablation_findings: dict[str, Any] = {}
    robustness: dict[str, Any] = {"status": "SKIPPED_AUDIT_ONLY"}

    if not audit_only:
        logistic_result = train_logistic_baseline(
            dataset.frame, dataset.feature_columns, dataset.target_column
        ).to_dict()
        calibrated_result = train_calibrated_logistic(
            dataset.frame, dataset.feature_columns, dataset.target_column
        ).to_dict()
        xgb_result = train_xgb_meta_model(
            dataset.frame,
            dataset.feature_columns,
            dataset.target_column,
            include_xgb_meta=include_xgb_meta or full,
        ).to_dict()
        ablation_findings = run_signal_family_ablations(
            dataset.frame,
            dataset.target_column,
            {
                "model_score_family": dataset.signal_families_present.get("model_score_family", []),
                "consensus_family": dataset.signal_families_present.get("consensus_family", []),
                "policy_family": dataset.signal_families_present.get("policy_family", []),
                "liquidity_activity_family": dataset.signal_families_present.get(
                    "liquidity_activity_family", []
                ),
                "concentration_robustness_family": dataset.signal_families_present.get(
                    "concentration_robustness_family", []
                ),
                "whale_family": [c for c in dataset.frame.columns if "whale" in c],
            },
        )
        write_audit_json(ablation_findings, paths["audits"] / "signal_family_ablations.json")

    robustness = run_robustness_audits(
        dataset.frame,
        dataset.target_column,
        rule_result=rule_result,
    )
    write_audit_json(robustness, paths["audits"] / "robustness_audits.json")

    split_counts = (
        dataset.frame["split"].value_counts().to_dict() if "split" in dataset.frame.columns else {}
    )
    dataset_summary = {
        **dataset.to_summary_dict(),
        "split_counts": split_counts,
        "mode": "smoke" if smoke else "full",
        "audit_only": audit_only,
    }

    decision = decide_meta_layer(
        audits=audit_dict,
        rule_result=rule_result.to_dict() if hasattr(rule_result, "to_dict") else rule_result,
        logistic_result=logistic_result,
        calibrated_result=calibrated_result,
        xgb_result=xgb_result,
        robustness=robustness,
        ablation_findings=ablation_findings,
        policy_audit=policy_audit,
        dataset_summary=dataset_summary,
    )
    decision["phase"] = AE7_FINAL_PHASE
    decision["output_root"] = str(output_root)
    decision["dataset_summary"] = dataset_summary
    decision["rule_baseline_status"] = rule_result.status if hasattr(rule_result, "status") else rule_result.get("status")
    write_decision_gate(decision, paths["reports"] / "ae7_final_meta_layer_decision_gate.json")
    write_audit_json(decision, paths["reports"] / "ae7_final_meta_layer_summary.json")
    return decision


def main() -> None:
    parser = argparse.ArgumentParser(description="AE7 FINAL meta-layer offline evaluation")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-rows", type=int, default=50_000)
    parser.add_argument("--include-xgb-meta", action="store_true", default=False)
    parser.add_argument("--audit-only", action="store_true", default=True)
    parser.add_argument("--smoke", action="store_true", default=True)
    parser.add_argument("--full", action="store_true", default=False)
    parser.add_argument("--scoring-policy-config", type=Path, default=None)
    args = parser.parse_args()

    if args.full:
        args.audit_only = False
        args.smoke = False

    decision = run_ae7_final_meta_layer(
        project_root=ROOT,
        output_root=args.output_root,
        max_rows=args.max_rows,
        include_xgb_meta=args.include_xgb_meta,
        audit_only=args.audit_only,
        smoke=args.smoke,
        full=args.full,
        scoring_policy_config=args.scoring_policy_config,
    )
    compact = {
        "phase": decision.get("phase"),
        "final_status": decision.get("final_status"),
        "best_approach": decision.get("best_approach"),
        "leakage_audit_status": decision.get("leakage_audit_status"),
        "policy_config_status": decision.get("policy_config_status"),
        "runtime_inference_status": decision.get("runtime_inference_status"),
        "output_root": decision.get("output_root"),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
