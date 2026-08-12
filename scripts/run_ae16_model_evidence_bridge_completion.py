#!/usr/bin/env python3
"""AE16 continuation — model-evidence bridge completion (Original E6 repair).

Attempts safe exact-ID attachment or feature-parity inference for RF/XGB/TAB
against cleaned Clean Forward candidates. Fails closed with an explicit AE16
blocker when neither path is safe. Does not train, backtest, or start AE17.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.consensus import (  # noqa: E402
    CONSENSUS_ENGINE_VERSION_V2,
    DEFAULT_CLEANED_INPUT_ROOT,
    PHASE,
)
from app.consensus.audits import (  # noqa: E402
    audit_authority,
    audit_consensus_tier_logic,
    audit_no_invented_scores,
    audit_no_legacy_source,
)
from app.consensus.evidence_bridge import (  # noqa: E402
    audit_exact_id_joins,
    audit_feature_parity,
    build_attachment_v2,
    decide_completion_classification,
    discover_direct_target_artifacts,
)
from app.consensus.model_evidence import AttachmentResult  # noqa: E402
from app.consensus.serialization import (  # noqa: E402
    read_csv_dicts,
    relpath_str,
    write_csv,
    write_json,
    write_jsonl,
    write_text,
)
from app.consensus.tiered_engine import (  # noqa: E402
    build_all_consensus_decisions,
    summarize_consensus_tiers,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AE16 model-evidence bridge completion")
    p.add_argument("--input-root", type=str, default=DEFAULT_CLEANED_INPUT_ROOT)
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument(
        "--prior-ae16-root",
        type=str,
        default="data/audits/ae16_tiered_consensus_engine_20260722_202855",
    )
    return p.parse_args(argv)


def _attachment_dicts_to_results(rows: list[dict[str, Any]]) -> list[AttachmentResult]:
    out: list[AttachmentResult] = []
    for r in rows:
        score = r.get("score")
        attached = bool(r.get("evidence_attached")) and r.get("attachment_status") == "MODEL_EVIDENCE_ATTACHED"
        out.append(
            AttachmentResult(
                clean_forward_candidate_id=str(r.get("clean_forward_candidate_id") or ""),
                clean_forward_decision_input_id=str(r.get("clean_forward_decision_input_id") or ""),
                pair_address=str(r.get("pair_address") or ""),
                base_token_address=str(r.get("base_token_address") or ""),
                quote_token_address=str(r.get("quote_token_address") or ""),
                model_family=str(r.get("model_family") or ""),
                evidence_attached=attached,
                score=score if attached else None,
                rank=r.get("rank") if attached else None,
                percentile_rank=r.get("percentile_rank") if attached else None,
                source_artifact_path=str(r.get("source_artifact_path") or ""),
                source_run_id=str(r.get("source_run_id") or ""),
                source_prediction_file=str(r.get("source_prediction_file") or ""),
                source_model_artifact=str(r.get("source_model_artifact") or ""),
                candidate_policy_id=str(r.get("candidate_policy_id") or ""),
                target_row_id=str(r.get("target_row_id") or ""),
                target_name=str(r.get("target_name") or ""),
                target_version=str(r.get("target_version") or ""),
                horizon=str(r.get("horizon") or ""),
                filter_name=str(r.get("filter_name") or ""),
                exit_policy_id=str(r.get("exit_policy_id") or ""),
                evidence_type=str(r.get("evidence_source_type") or ""),
                attachment_status=str(r.get("attachment_status") or "MODEL_EVIDENCE_UNAVAILABLE"),
                attachment_failure_reason=str(r.get("attachment_failure_reason") or ""),
            )
        )
    return out


def run_ae16_bridge_completion(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.output_root) if args.output_root else (
        ROOT / "data" / "audits" / f"ae16_model_evidence_bridge_completion_{utc_stamp()}"
    )
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    reports = out_root / "reports"
    data_dir = out_root / "data"
    audits = out_root / "audits"
    for d in (reports, data_dir, audits):
        d.mkdir(parents=True, exist_ok=True)

    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = ROOT / input_root

    candidates = read_csv_dicts(input_root / "ae16_clean_forward_candidates.csv")
    decision_inputs = read_csv_dicts(input_root / "ae16_clean_forward_decision_inputs.csv")
    decision_by_candidate = {
        str(r.get("clean_forward_candidate_id") or ""): r for r in decision_inputs
    }

    # --- Part 1: discovery ---
    discovery_rows, discovery_manifest = discover_direct_target_artifacts(
        ROOT,
        exclude_roots=(relpath_str(out_root, ROOT),),
    )
    write_json(reports / "ae16_model_evidence_discovery_manifest.json", discovery_manifest)
    write_csv(data_dir / "ae16_discovered_model_assets.csv", discovery_rows)
    write_csv(audits / "ae16_direct_target_artifact_discovery_audit.csv", discovery_rows)
    if not (data_dir / "ae16_discovered_model_assets.csv").is_file():
        raise RuntimeError("Failed to write ae16_discovered_model_assets.csv")

    # --- Part 2: exact-ID join ---
    join_rows, join_summary = audit_exact_id_joins(
        project_root=ROOT,
        candidates=candidates,
        discovery_rows=discovery_rows,
    )
    write_csv(audits / "ae16_exact_id_join_audit.csv", join_rows)
    write_json(reports / "ae16_exact_id_join_summary.json", join_summary)

    # --- Part 3: feature parity ---
    parity_rows, compat_rows, feature_matrix, parity_summary = audit_feature_parity(
        project_root=ROOT,
        candidates=candidates,
        discovery_rows=discovery_rows,
    )
    write_csv(data_dir / "ae16_clean_forward_feature_matrix.csv", feature_matrix)
    write_csv(audits / "ae16_feature_parity_audit.csv", parity_rows)
    write_csv(audits / "ae16_model_artifact_compatibility_audit.csv", compat_rows)
    write_json(reports / "ae16_feature_parity_summary.json", parity_summary)

    # Inference intentionally not run when parity fails / not allowed.
    inference_ran = {
        fam: bool((parity_summary.get("by_family") or {}).get(fam, {}).get("inference_allowed"))
        and False  # never true in fail-closed path without explicit attach
        for fam in ("RF", "XGB", "TAB")
    }
    for fam, info in (parity_summary.get("by_family") or {}).items():
        info["inference_run"] = False
        inference_ran[fam] = False

    # --- Part 4/5: attachment + vote policy ---
    attachments, vote_policy_rows = build_attachment_v2(
        candidates=candidates,
        decision_by_candidate=decision_by_candidate,
        join_summary=join_summary,
        parity_summary=parity_summary,
        discovery_rows=discovery_rows,
    )
    write_csv(data_dir / "ae16_model_evidence_attachment_v2.csv", attachments)
    write_csv(audits / "ae16_vote_policy_audit.csv", vote_policy_rows)

    # --- Part 6: consensus v2 ---
    attachment_results = _attachment_dicts_to_results(attachments)
    decisions = build_all_consensus_decisions(
        candidates=candidates,
        decision_by_candidate=decision_by_candidate,
        attachments=attachment_results,
    )
    # Ensure engine version v2 marker
    for d in decisions:
        d["consensus_engine_version"] = CONSENSUS_ENGINE_VERSION_V2
        d["trade_authority"] = False
        d["live_trading_ready"] = False
        d["wallet_authority"] = False
        d["risk_gate_override_authority"] = False
        d["paper_demo_only"] = True
        d["authority_status"] = "RESEARCH_SHADOW_ONLY"

    write_csv(data_dir / "ae16_clean_forward_consensus_decisions_v2.csv", decisions)
    write_jsonl(data_dir / "ae16_clean_forward_consensus_decisions_v2.jsonl", decisions)
    tier_summary = summarize_consensus_tiers(decisions)
    write_csv(data_dir / "ae16_consensus_tier_summary_v2.csv", tier_summary)
    write_csv(audits / "ae16_consensus_tier_logic_audit_v2.csv", audit_consensus_tier_logic(decisions))

    # --- Part 7: safety audits v2 ---
    invented_audit, missing_handling = audit_no_invented_scores(
        decision_inputs=decision_inputs,
        attachments=attachment_results,
        decisions=decisions,
    )
    # Strengthen: ensure no attachment has invented zero
    for a in attachments:
        if a.get("score") == 0 and not a.get("evidence_attached"):
            invented_audit["passed"] = False
            invented_audit.setdefault("violations", []).append("defaulted_zero_on_unattached")
    write_json(audits / "ae16_no_invented_scores_audit_v2.json", invented_audit)
    write_csv(audits / "ae16_missing_score_handling_audit_v2.csv", missing_handling)

    legacy_audit = audit_no_legacy_source(
        used_paths=[relpath_str(input_root, ROOT)]
        + [str(r.get("path") or "") for r in discovery_rows if r.get("path_exists")],
        candidates_from_market_snapshots=False,
        decision_inputs_from_old_feed=False,
    )
    write_json(audits / "ae16_no_legacy_source_audit_v2.json", legacy_audit)

    authority_audit = audit_authority(decisions)
    write_json(audits / "ae16_authority_audit_v2.json", authority_audit)

    # --- Part 8: decision gate ---
    classification = decide_completion_classification(
        join_summary=join_summary,
        parity_summary=parity_summary,
        attachments=attachments,
        invented_ok=bool(invented_audit.get("passed")),
        legacy_ok=bool(legacy_audit.get("passed")),
        authority_ok=bool(authority_audit.get("passed")),
    )

    attached_counts = Counter(
        a["model_family"]
        for a in attachments
        if a.get("evidence_attached") and a.get("attachment_status") == "MODEL_EVIDENCE_ATTACHED"
    )
    status_counts = Counter(a["attachment_status"] for a in attachments)
    tier_counts = {r["consensus_tier"]: r["count"] for r in tier_summary}

    ae16_closeable = classification in {
        "AE16_TIERED_CONSENSUS_ENGINE_PASS_WITH_MODEL_EVIDENCE",
        "AE16_TIERED_CONSENSUS_ENGINE_PARTIAL_PASS_WITH_MODEL_EVIDENCE",
    }

    gate = {
        "phase": PHASE,
        "continuation": "ae16_model_evidence_bridge_completion",
        "status": "PASS" if ae16_closeable else "BLOCKED",
        "classification": classification,
        "generated_at_utc": utc_now(),
        "prior_ae16_root": args.prior_ae16_root,
        "input_root": relpath_str(input_root, ROOT),
        "output_root": relpath_str(out_root, ROOT),
        "input_counts": {
            "candidates": len(candidates),
            "decision_inputs": len(decision_inputs),
        },
        "discovery": {
            "canonical_expected_roots": discovery_manifest.get("canonical_expected_roots"),
            "fallback_discovered_roots": discovery_manifest.get("fallback_discovered_roots"),
            "counts_by_family": discovery_manifest.get("counts_by_family"),
            "counts_by_type": discovery_manifest.get("counts_by_type"),
            "usable_for_attachment_count": discovery_manifest.get("usable_for_attachment_count"),
        },
        "exact_id_join": join_summary,
        "feature_parity": {
            "any_passed": parity_summary.get("any_feature_parity_passed"),
            "any_inference_allowed": parity_summary.get("any_inference_allowed"),
            "by_family": parity_summary.get("by_family"),
            "inference_run_by_family": inference_ran,
        },
        "model_evidence_attached_counts": {
            "RF": attached_counts.get("RF", 0),
            "XGB": attached_counts.get("XGB", 0),
            "TAB": attached_counts.get("TAB", 0),
        },
        "attachment_status_counts": dict(status_counts),
        "vote_policy": vote_policy_rows,
        "consensus_tier_counts": tier_counts,
        "no_invented_scores_passed": invented_audit.get("passed"),
        "no_legacy_source_passed": legacy_audit.get("passed"),
        "authority_audit_passed": authority_audit.get("passed"),
        "retraining_required": classification == "AE16_BLOCKED_RETRAINING_REQUIRED"
        or (
            not join_summary.get("any_exact_join_safe")
            and not parity_summary.get("any_inference_allowed")
        ),
        "ae16_can_close_as_original_e6_repair": ae16_closeable,
        "hard_blocker": None if ae16_closeable else classification,
        "hard_blocker_proof": {
            "exact_id_join_safe": join_summary.get("any_exact_join_safe"),
            "feature_parity_passed": parity_summary.get("any_feature_parity_passed"),
            "inference_allowed": parity_summary.get("any_inference_allowed"),
            "attached_any": sum(attached_counts.values()) > 0,
            "rf_missing_features": (parity_summary.get("by_family") or {}).get("RF", {}).get(
                "missing_required_feature_count"
            ),
            "xgb_missing_features": (parity_summary.get("by_family") or {}).get("XGB", {}).get(
                "missing_required_feature_count"
            ),
            "tab_missing_features": (parity_summary.get("by_family") or {}).get("TAB", {}).get(
                "missing_required_feature_count"
            ),
        },
        "trade_authority": False,
        "live_trading_ready": False,
        "wallet_authority": False,
        "risk_gate_override_authority": False,
        "paper_demo_only": True,
        "model_authority_approved": False,
        "not_profitability_evidence": True,
    }
    write_json(reports / "ae16_completion_decision_gate.json", gate)

    summary_lines = [
        "AE16 Model Evidence Bridge Completion",
        f"classification: {classification}",
        f"status: {gate['status']}",
        f"output_root: {gate['output_root']}",
        f"RF attached: {attached_counts.get('RF', 0)}",
        f"XGB attached: {attached_counts.get('XGB', 0)}",
        f"TAB attached: {attached_counts.get('TAB', 0)}",
        f"exact_id_join_safe: {join_summary.get('any_exact_join_safe')}",
        f"feature_parity_passed: {parity_summary.get('any_feature_parity_passed')}",
        f"inference_allowed: {parity_summary.get('any_inference_allowed')}",
        f"inference_run: {inference_ran}",
        f"retraining_required_flag: {gate['retraining_required']}",
        f"consensus_tier_counts: {tier_counts}",
        f"no_invented_scores: {invented_audit.get('passed')}",
        f"no_legacy_source: {legacy_audit.get('passed')}",
        f"authority_ok: {authority_audit.get('passed')}",
        f"ae16_can_close_as_original_e6_repair: {ae16_closeable}",
        f"hard_blocker: {gate['hard_blocker']}",
        f"hard_blocker_proof: {gate['hard_blocker_proof']}",
        "trade_authority: false",
        "live_trading_ready: false",
    ]
    write_text(reports / "ae16_completion_summary_for_upload.txt", "\n".join(summary_lines) + "\n")

    write_json(
        reports / "ae16_completion_manifest.json",
        {
            "phase": PHASE,
            "classification": classification,
            "output_root": gate["output_root"],
            "generated_at_utc": utc_now(),
            "files": {
                "discovery_manifest": relpath_str(reports / "ae16_model_evidence_discovery_manifest.json", ROOT),
                "discovered_assets": relpath_str(data_dir / "ae16_discovered_model_assets.csv", ROOT),
                "join_summary": relpath_str(reports / "ae16_exact_id_join_summary.json", ROOT),
                "attachment_v2": relpath_str(data_dir / "ae16_model_evidence_attachment_v2.csv", ROOT),
                "consensus_v2": relpath_str(data_dir / "ae16_clean_forward_consensus_decisions_v2.csv", ROOT),
                "decision_gate": relpath_str(reports / "ae16_completion_decision_gate.json", ROOT),
                "summary": relpath_str(reports / "ae16_completion_summary_for_upload.txt", ROOT),
            },
        },
    )

    return {
        "classification": classification,
        "output_root": out_root,
        "gate": gate,
        "ae16_closeable": ae16_closeable,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_ae16_bridge_completion(args)
    print(result["classification"])
    print(f"output_root={result['output_root']}")
    print(f"ae16_closeable={result['ae16_closeable']}")
    return 0 if result["ae16_closeable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
