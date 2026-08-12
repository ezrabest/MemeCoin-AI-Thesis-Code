#!/usr/bin/env python3
"""AE16 — Tiered Consensus Engine on Direct Target / Clean Forward Bridge.

Schema/evidence/consensus-engine phase only.
Does NOT: train, backtest, mutate trader.db, grant live authority,
connect wallet, call Gemini/Qwen/Helius, or claim profitability.
"""
from __future__ import annotations

import argparse
import sys
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.consensus import (  # noqa: E402
    DEFAULT_CLEANED_INPUT_ROOT,
    EXPECTED_INPUT_COUNTS,
    PHASE,
    REQUIRED_INPUT_FILES,
)
from app.consensus.audits import (  # noqa: E402
    audit_authority,
    audit_consensus_tier_logic,
    audit_input_contract,
    audit_no_invented_scores,
    audit_no_legacy_source,
    run_input_path_preflight,
)
from app.consensus.model_evidence import (  # noqa: E402
    attach_all_model_evidence,
    discover_model_artifacts,
    summarize_model_availability,
)
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
    p = argparse.ArgumentParser(description="AE16 Tiered Consensus Engine")
    p.add_argument(
        "--input-root",
        type=str,
        default=DEFAULT_CLEANED_INPUT_ROOT,
        help="Directory containing cleaned AE16 input CSVs",
    )
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument(
        "--skip-artifact-discovery",
        action="store_true",
        help="Skip scanning training artifacts (tests / fast fail-closed)",
    )
    return p.parse_args(argv)


def _make_output_dirs(out_root: Path) -> dict[str, Path]:
    reports = out_root / "reports"
    data_dir = out_root / "data"
    audits = out_root / "audits"
    for d in (reports, data_dir, audits):
        d.mkdir(parents=True, exist_ok=True)
    return {"reports": reports, "data": data_dir, "audits": audits}


def _write_blocked_minimal(
    *,
    out_root: Path,
    dirs: dict[str, Path],
    classification: str,
    blockers: list[str],
    preflight: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gate = {
        "phase": PHASE,
        "status": "BLOCKED",
        "classification": classification,
        "blockers": blockers,
        "limitations": [],
        "generated_at_utc": utc_now(),
        "trade_authority": False,
        "live_trading_ready": False,
        "wallet_authority": False,
        "risk_gate_override_authority": False,
        "paper_demo_only": True,
        "model_authority_approved": False,
        "not_profitability_evidence": True,
        "ae17_blocked": True,
        "ae17_blocker": classification,
        "preflight_passed": bool(preflight.get("passed")),
        "missing_files": preflight.get("missing_files", []),
    }
    if extra:
        gate.update(extra)

    write_csv(dirs["audits"] / "ae16_input_path_preflight_audit.csv", preflight.get("rows") or [])
    write_json(dirs["reports"] / "ae16_decision_gate.json", gate)

    summary_lines = [
        "AE16 Tiered Consensus Engine",
        f"classification: {classification}",
        f"status: BLOCKED",
        f"output_root: {relpath_str(out_root, ROOT)}",
        f"preflight_passed: {preflight.get('passed')}",
        f"missing_files: {', '.join(preflight.get('missing_files') or []) or '(none)'}",
        "trade_authority: false",
        "live_trading_ready: false",
        "AE17 blocked: true",
    ]
    write_text(dirs["reports"] / "ae16_summary_for_upload.txt", "\n".join(summary_lines) + "\n")

    manifest = {
        "phase": PHASE,
        "classification": classification,
        "status": "BLOCKED",
        "output_root": relpath_str(out_root, ROOT),
        "generated_at_utc": utc_now(),
        "files": {
            "decision_gate": relpath_str(dirs["reports"] / "ae16_decision_gate.json", ROOT),
            "summary": relpath_str(dirs["reports"] / "ae16_summary_for_upload.txt", ROOT),
            "preflight_audit": relpath_str(dirs["audits"] / "ae16_input_path_preflight_audit.csv", ROOT),
        },
    }
    write_json(dirs["reports"] / "ae16_manifest.json", manifest)
    return {"classification": classification, "output_root": out_root, "gate": gate, "manifest": manifest}


def decide_classification(
    *,
    blockers: list[str],
    any_model_evidence_attached: bool,
) -> tuple[str, str]:
    if blockers:
        return blockers[0], "BLOCKED"
    if any_model_evidence_attached:
        return "AE16_TIERED_CONSENSUS_ENGINE_PASS_WITH_MODEL_EVIDENCE", "PASS"
    return "AE16_TIERED_CONSENSUS_ENGINE_PASS_SCHEMA_ONLY_NO_MODEL_EVIDENCE", "PASS"


def run_ae16(args: argparse.Namespace) -> dict[str, Any]:
    out_root = Path(args.output_root) if args.output_root else (
        ROOT / "data" / "audits" / f"ae16_tiered_consensus_engine_{utc_stamp()}"
    )
    if not out_root.is_absolute():
        out_root = ROOT / out_root

    dirs = _make_output_dirs(out_root)
    input_root = Path(args.input_root)
    if not input_root.is_absolute():
        input_root = ROOT / input_root

    # --- 1. Preflight ---
    preflight = run_input_path_preflight(input_root, required_files=REQUIRED_INPUT_FILES)
    write_csv(dirs["audits"] / "ae16_input_path_preflight_audit.csv", preflight["rows"])

    if not preflight["passed"]:
        return _write_blocked_minimal(
            out_root=out_root,
            dirs=dirs,
            classification="AE16_BLOCKED_INPUT_FILES_MISSING",
            blockers=["AE16_BLOCKED_INPUT_FILES_MISSING"],
            preflight=preflight,
        )

    blockers: list[str] = []
    limitations: list[str] = []

    try:
        candidates = read_csv_dicts(input_root / "ae16_clean_forward_candidates.csv")
        decision_inputs = read_csv_dicts(input_root / "ae16_clean_forward_decision_inputs.csv")
        outcomes = read_csv_dicts(input_root / "ae16_clean_forward_outcome_label_contract.csv")
        execution_links = read_csv_dicts(input_root / "ae16_clean_forward_paper_execution_links.csv")
    except Exception as exc:  # noqa: BLE001
        blockers.append("AE16_BLOCKED_INPUT_CONTRACT_FAILED")
        return _write_blocked_minimal(
            out_root=out_root,
            dirs=dirs,
            classification="AE16_BLOCKED_INPUT_CONTRACT_FAILED",
            blockers=blockers,
            preflight=preflight,
            extra={"read_error": f"{type(exc).__name__}: {exc}"},
        )

    # --- 2. Input contract ---
    contract = audit_input_contract(
        candidates=candidates,
        decision_inputs=decision_inputs,
        outcomes=outcomes,
        execution_links=execution_links,
    )
    write_json(dirs["audits"] / "ae16_input_contract_audit.json", contract)
    if not contract["passed"]:
        blockers.append("AE16_BLOCKED_INPUT_CONTRACT_FAILED")
        gate = {
            "phase": PHASE,
            "status": "BLOCKED",
            "classification": "AE16_BLOCKED_INPUT_CONTRACT_FAILED",
            "blockers": blockers,
            "limitations": contract.get("failures") or [],
            "generated_at_utc": utc_now(),
            "trade_authority": False,
            "live_trading_ready": False,
            "wallet_authority": False,
            "risk_gate_override_authority": False,
            "paper_demo_only": True,
            "model_authority_approved": False,
            "not_profitability_evidence": True,
            "ae17_blocked": True,
            "ae17_blocker": "AE16_BLOCKED_INPUT_CONTRACT_FAILED",
            "input_counts": contract.get("counts"),
        }
        write_json(dirs["reports"] / "ae16_decision_gate.json", gate)
        write_text(
            dirs["reports"] / "ae16_summary_for_upload.txt",
            "\n".join(
                [
                    "AE16 Tiered Consensus Engine",
                    "classification: AE16_BLOCKED_INPUT_CONTRACT_FAILED",
                    f"failures: {contract.get('failures')}",
                    f"counts: {contract.get('counts')}",
                    "AE17 blocked: true",
                ]
            )
            + "\n",
        )
        write_json(
            dirs["reports"] / "ae16_manifest.json",
            {
                "phase": PHASE,
                "classification": "AE16_BLOCKED_INPUT_CONTRACT_FAILED",
                "status": "BLOCKED",
                "output_root": relpath_str(out_root, ROOT),
                "generated_at_utc": utc_now(),
                "input_counts": contract.get("counts"),
            },
        )
        return {
            "classification": "AE16_BLOCKED_INPUT_CONTRACT_FAILED",
            "output_root": out_root,
            "gate": gate,
        }

    decision_by_candidate = {
        str(r.get("clean_forward_candidate_id") or ""): r for r in decision_inputs
    }

    # --- 3. Model-evidence adapter ---
    try:
        if args.skip_artifact_discovery:
            discovered = {f: [] for f in ("RF", "XGB", "TAB")}
        else:
            discovered = discover_model_artifacts(ROOT)
        attachments = attach_all_model_evidence(
            candidates=candidates,
            decision_by_candidate=decision_by_candidate,
            project_root=ROOT,
            discovered=discovered,
        )
        if len(attachments) != len(candidates) * 3:
            blockers.append("AE16_BLOCKED_MODEL_EVIDENCE_ADAPTER")
    except Exception as exc:  # noqa: BLE001
        blockers.append("AE16_BLOCKED_MODEL_EVIDENCE_ADAPTER")
        write_json(
            dirs["audits"] / "ae16_model_evidence_adapter_error.json",
            {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
        )
        # Still emit empty attachment schema for auditability
        attachments = []
        discovered = {f: [] for f in ("RF", "XGB", "TAB")}

    attachment_dicts = [a.to_dict() for a in attachments]
    write_csv(dirs["data"] / "ae16_model_evidence_attachment.csv", attachment_dicts)

    attachment_audit_rows = [
        {
            "clean_forward_candidate_id": a.clean_forward_candidate_id,
            "model_family": a.model_family,
            "evidence_attached": a.evidence_attached,
            "attachment_status": a.attachment_status,
            "attachment_failure_reason": a.attachment_failure_reason,
            "score_is_null": a.score is None,
            "source_prediction_file": a.source_prediction_file,
            "candidate_policy_id": a.candidate_policy_id,
            "target_row_id": a.target_row_id,
        }
        for a in attachments
    ]
    write_csv(dirs["audits"] / "ae16_model_evidence_attachment_audit.csv", attachment_audit_rows)

    availability = summarize_model_availability(attachments)
    write_csv(dirs["data"] / "ae16_model_availability_summary.csv", availability)

    discovery_paths: list[str] = []
    for fam_arts in (discovered or {}).values():
        for art in fam_arts:
            discovery_paths.append(str(art.path).replace("\\", "/"))

    # --- 4/6. Consensus engine ---
    decisions = build_all_consensus_decisions(
        candidates=candidates,
        decision_by_candidate=decision_by_candidate,
        attachments=attachments,
    )
    write_csv(dirs["data"] / "ae16_clean_forward_consensus_decisions.csv", decisions)
    write_jsonl(dirs["data"] / "ae16_clean_forward_consensus_decisions.jsonl", decisions)

    tier_summary = summarize_consensus_tiers(decisions)
    write_csv(dirs["data"] / "ae16_consensus_tier_summary.csv", tier_summary)

    tier_logic_audit = audit_consensus_tier_logic(decisions)
    write_csv(dirs["audits"] / "ae16_consensus_tier_logic_audit.csv", tier_logic_audit)
    if any(not r.get("logic_ok") for r in tier_logic_audit):
        limitations.append("AE16_CONSENSUS_TIER_LOGIC_AUDIT_WARNINGS")

    # --- 5. No invented scores ---
    invented_audit, missing_handling = audit_no_invented_scores(
        decision_inputs=decision_inputs,
        attachments=attachments,
        decisions=decisions,
    )
    write_json(dirs["audits"] / "ae16_no_invented_scores_audit.json", invented_audit)
    write_csv(dirs["audits"] / "ae16_missing_score_handling_audit.csv", missing_handling)
    if not invented_audit["passed"]:
        blockers.append("AE16_BLOCKED_INVENTED_OR_DEFAULTED_SCORES")

    # --- 7. Authority ---
    authority_audit = audit_authority(decisions)
    write_json(dirs["audits"] / "ae16_authority_audit.json", authority_audit)
    if not authority_audit["passed"]:
        blockers.append("AE16_BLOCKED_AUTHORITY_ESCALATION")

    # --- 8. No legacy SoT ---
    legacy_audit = audit_no_legacy_source(
        used_paths=[relpath_str(input_root, ROOT)] + discovery_paths,
        candidates_from_market_snapshots=False,
        decision_inputs_from_old_feed=False,
    )
    write_json(dirs["audits"] / "ae16_no_legacy_source_audit.json", legacy_audit)
    if not legacy_audit["passed"]:
        blockers.append("AE16_BLOCKED_LEGACY_CONTAMINATION")

    any_attached = any(
        a.evidence_attached and a.attachment_status == "MODEL_EVIDENCE_ATTACHED" for a in attachments
    )
    classification, status = decide_classification(
        blockers=blockers,
        any_model_evidence_attached=any_attached,
    )

    attached_counts = Counter(
        a.model_family
        for a in attachments
        if a.evidence_attached and a.attachment_status == "MODEL_EVIDENCE_ATTACHED"
    )
    tier_counts = {r["consensus_tier"]: r["count"] for r in tier_summary}

    if not any_attached:
        limitations.append("AE15_SCHEMA_ONLY_MODEL_EVIDENCE_PENDING_RESOLVED_AS_NO_SAFE_JOIN")
        limitations.append("MODEL_EVIDENCE_UNAVAILABLE_FOR_ALL_CLEAN_FORWARD_CANDIDATES")

    gate = {
        "phase": PHASE,
        "status": status,
        "classification": classification,
        "blockers": blockers,
        "limitations": limitations,
        "generated_at_utc": utc_now(),
        "input_root": relpath_str(input_root, ROOT),
        "input_counts": {
            "candidates": len(candidates),
            "decision_inputs": len(decision_inputs),
            "outcome_contracts": len(outcomes),
            "execution_links": len(execution_links),
            "expected": EXPECTED_INPUT_COUNTS,
        },
        "model_evidence_attached_counts": {
            "RF": attached_counts.get("RF", 0),
            "XGB": attached_counts.get("XGB", 0),
            "TAB": attached_counts.get("TAB", 0),
            "any_attached": any_attached,
        },
        "consensus_tier_counts": tier_counts,
        "no_invented_scores_passed": invented_audit["passed"],
        "no_legacy_source_passed": legacy_audit["passed"],
        "authority_audit_passed": authority_audit["passed"],
        "trade_authority": False,
        "live_trading_ready": False,
        "wallet_authority": False,
        "risk_gate_override_authority": False,
        "paper_demo_only": True,
        "model_authority_approved": False,
        "not_profitability_evidence": True,
        "ae17_blocked": status != "PASS",
        "ae17_blocker": blockers[0] if blockers else None,
        "ae17_recommended_inputs": [
            relpath_str(dirs["data"] / "ae16_clean_forward_consensus_decisions.csv", ROOT),
            relpath_str(dirs["data"] / "ae16_model_evidence_attachment.csv", ROOT),
            relpath_str(dirs["data"] / "ae16_consensus_tier_summary.csv", ROOT),
            relpath_str(dirs["reports"] / "ae16_decision_gate.json", ROOT),
        ]
        if status == "PASS"
        else [],
    }
    write_json(dirs["reports"] / "ae16_decision_gate.json", gate)

    summary = "\n".join(
        [
            "AE16 Tiered Consensus Engine",
            f"classification: {classification}",
            f"status: {status}",
            f"output_root: {relpath_str(out_root, ROOT)}",
            f"input_root: {relpath_str(input_root, ROOT)}",
            f"candidates: {len(candidates)}",
            f"decision_inputs: {len(decision_inputs)}",
            f"outcome_contracts: {len(outcomes)}",
            f"execution_links: {len(execution_links)}",
            f"RF attached: {attached_counts.get('RF', 0)}",
            f"XGB attached: {attached_counts.get('XGB', 0)}",
            f"TAB attached: {attached_counts.get('TAB', 0)}",
            f"consensus_tier_counts: {tier_counts}",
            f"no_invented_scores: {invented_audit['passed']}",
            f"no_legacy_source: {legacy_audit['passed']}",
            f"authority_ok: {authority_audit['passed']}",
            "trade_authority: false",
            "live_trading_ready: false",
            f"AE17 blocked: {gate['ae17_blocked']}",
        ]
    ) + "\n"
    write_text(dirs["reports"] / "ae16_summary_for_upload.txt", summary)

    files_map = {
        "manifest": relpath_str(dirs["reports"] / "ae16_manifest.json", ROOT),
        "summary": relpath_str(dirs["reports"] / "ae16_summary_for_upload.txt", ROOT),
        "decision_gate": relpath_str(dirs["reports"] / "ae16_decision_gate.json", ROOT),
        "model_evidence_attachment": relpath_str(dirs["data"] / "ae16_model_evidence_attachment.csv", ROOT),
        "consensus_decisions_csv": relpath_str(
            dirs["data"] / "ae16_clean_forward_consensus_decisions.csv", ROOT
        ),
        "consensus_decisions_jsonl": relpath_str(
            dirs["data"] / "ae16_clean_forward_consensus_decisions.jsonl", ROOT
        ),
        "consensus_tier_summary": relpath_str(dirs["data"] / "ae16_consensus_tier_summary.csv", ROOT),
        "model_availability_summary": relpath_str(
            dirs["data"] / "ae16_model_availability_summary.csv", ROOT
        ),
        "preflight_audit": relpath_str(dirs["audits"] / "ae16_input_path_preflight_audit.csv", ROOT),
        "input_contract_audit": relpath_str(dirs["audits"] / "ae16_input_contract_audit.json", ROOT),
        "model_evidence_attachment_audit": relpath_str(
            dirs["audits"] / "ae16_model_evidence_attachment_audit.csv", ROOT
        ),
        "no_invented_scores_audit": relpath_str(
            dirs["audits"] / "ae16_no_invented_scores_audit.json", ROOT
        ),
        "missing_score_handling_audit": relpath_str(
            dirs["audits"] / "ae16_missing_score_handling_audit.csv", ROOT
        ),
        "consensus_tier_logic_audit": relpath_str(
            dirs["audits"] / "ae16_consensus_tier_logic_audit.csv", ROOT
        ),
        "authority_audit": relpath_str(dirs["audits"] / "ae16_authority_audit.json", ROOT),
        "no_legacy_source_audit": relpath_str(dirs["audits"] / "ae16_no_legacy_source_audit.json", ROOT),
    }
    manifest = {
        "phase": PHASE,
        "classification": classification,
        "status": status,
        "output_root": relpath_str(out_root, ROOT),
        "input_root": relpath_str(input_root, ROOT),
        "generated_at_utc": utc_now(),
        "input_counts": gate["input_counts"],
        "model_evidence_attached_counts": gate["model_evidence_attached_counts"],
        "consensus_tier_counts": tier_counts,
        "files": files_map,
        "ae17_recommended_inputs": gate["ae17_recommended_inputs"],
    }
    write_json(dirs["reports"] / "ae16_manifest.json", manifest)

    return {
        "classification": classification,
        "status": status,
        "output_root": out_root,
        "gate": gate,
        "manifest": manifest,
        "attached_counts": dict(attached_counts),
        "tier_counts": tier_counts,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_ae16(args)
    except Exception as exc:  # noqa: BLE001 — last-resort controlled failure
        out_root = Path(args.output_root) if args.output_root else (
            ROOT / "data" / "audits" / f"ae16_tiered_consensus_engine_{utc_stamp()}"
        )
        if not out_root.is_absolute():
            out_root = ROOT / out_root
        dirs = _make_output_dirs(out_root)
        gate = {
            "phase": PHASE,
            "status": "BLOCKED",
            "classification": "AE16_BLOCKED_MODEL_EVIDENCE_ADAPTER",
            "blockers": ["AE16_BLOCKED_MODEL_EVIDENCE_ADAPTER"],
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "generated_at_utc": utc_now(),
            "trade_authority": False,
            "live_trading_ready": False,
            "paper_demo_only": True,
            "ae17_blocked": True,
        }
        write_json(dirs["reports"] / "ae16_decision_gate.json", gate)
        write_json(dirs["reports"] / "ae16_manifest.json", {"phase": PHASE, "classification": gate["classification"]})
        write_text(
            dirs["reports"] / "ae16_summary_for_upload.txt",
            f"AE16 blocked: {gate['classification']}\nerror: {gate['error']}\n",
        )
        print(gate["classification"], file=sys.stderr)
        return 2

    print(result["classification"])
    print(f"output_root={result['output_root']}")
    classification = result["classification"]
    if classification.startswith("AE16_TIERED_CONSENSUS_ENGINE_PASS"):
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
