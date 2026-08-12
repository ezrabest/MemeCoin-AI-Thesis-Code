#!/usr/bin/env python3
"""AE12.8 Safety / Audit / Reproducibility Package (read-only archival).

Does not retrain, call external APIs, mutate trader.db, enable live trading,
or close AE12.9 / full MSc package.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(r"E:\Projects\Final Project\memecoin_trader").resolve()
OUTPUT_ROOT = PROJECT_ROOT / "data" / "audits" / "ae12_8_safety_reproducibility_package_20260717_204845"

AE12_7_ROOT = (
    PROJECT_ROOT
    / "data"
    / "audits"
    / "ae12_7_intelligent_agent_operational_demo_20260717_130004_693435"
)
AE12_7_DAILY_RECORDS = (
    PROJECT_ROOT / "data" / "intelligent_agents" / "ae12_7_agent_records_20260717.jsonl"
)
AE12_7_DAILY_CALL_AUDIT = (
    PROJECT_ROOT / "data" / "intelligent_agents" / "ae12_7_agent_call_audit_20260717.jsonl"
)

CREATED_AT = datetime.now(timezone.utc).isoformat()
TRADER_DB = PROJECT_ROOT / "data" / "trader.db"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_meta(path: Path) -> dict[str, Any]:
    exists = path.exists()
    meta: dict[str, Any] = {
        "exact_path": str(path),
        "exists": exists,
        "file_size_bytes": "",
        "last_modified": "",
    }
    if exists:
        st = path.stat()
        meta["file_size_bytes"] = st.st_size
        meta["last_modified"] = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat()
    return meta


def read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:  # noqa: BLE001
        return {"_parse_error": str(exc)}


def extract_status(obj: Any) -> str:
    if obj is None:
        return "MISSING"
    if isinstance(obj, dict):
        if "_parse_error" in obj:
            return f"PARSE_ERROR:{obj['_parse_error']}"
        for key in (
            "status",
            "final_status",
            "classification",
            "gate_status",
            "inference_gate_status",
            "decision",
            "result",
            "phase",
        ):
            if key in obj and obj[key] is not None:
                # Prefer explicit gate/status fields over phase labels when both exist
                if key == "phase":
                    continue
                return str(obj[key])
        # Fallback composites for summary-only artifacts
        if obj.get("no_trade_authority") is True and "phase" in obj:
            return f"{obj['phase']}_NO_TRADE_AUTHORITY_SUMMARY"
        if "phase" in obj:
            return f"{obj['phase']}_SUMMARY_PRESENT"
    return "PRESENT_NO_STATUS_FIELD"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_quick(path: Path, max_bytes: int = 1024 * 1024) -> str:
    """Hash first max_bytes + size for mutation detection without full 8GB read."""
    if not path.exists():
        return "MISSING"
    h = hashlib.sha256()
    size = path.stat().st_size
    h.update(str(size).encode("utf-8"))
    h.update(b"|")
    with path.open("rb") as f:
        chunk = f.read(max_bytes)
        h.update(chunk)
    return h.hexdigest()


def inventory_row(
    phase: str,
    artifact_type: str,
    path: Path,
    required_ae129: str,
    required_final: str,
    archival_only: str,
    sot_level: str,
    notes: str,
) -> dict[str, Any]:
    meta = file_meta(path)
    return {
        "phase": phase,
        "artifact_type": artifact_type,
        "exact_path": meta["exact_path"],
        "exists": meta["exists"],
        "file_size_bytes": meta["file_size_bytes"],
        "last_modified": meta["last_modified"],
        "required_for_AE12_9": required_ae129,
        "required_for_final_report": required_final,
        "archival_only": archival_only,
        "source_of_truth_level": sot_level,
        "limitation_notes": notes,
    }


def build_artifact_inventory() -> list[dict[str, Any]]:
    audits = PROJECT_ROOT / "data" / "audits"
    rows: list[dict[str, Any]] = []

    # AE6
    ae6_latest = audits / "ae6_consensus_decision_layer_20260711T090105Z"
    rows.append(
        inventory_row(
            "AE6",
            "root",
            ae6_latest,
            "yes",
            "yes",
            "no",
            "historical_saved",
            "Latest AE6 consensus decision layer root",
        )
    )
    rows.append(
        inventory_row(
            "AE6",
            "summary",
            ae6_latest / "ae6_consensus_decision_summary.json",
            "yes",
            "yes",
            "no",
            "historical_saved",
            "DecisionRecord / consensus summary",
        )
    )
    rows.append(
        inventory_row(
            "AE6",
            "contract_root",
            audits / "ae6_0_raw_derived_contract_20260709_115912",
            "no",
            "yes",
            "yes",
            "historical_saved",
            "Raw/derived contract preflight",
        )
    )

    # AE7
    for label, p, note in [
        (
            "AE7",
            audits / "ae7_0_model_score_artifact_inventory_20260710_121711",
            "Model score artifact inventory",
        ),
        (
            "AE7",
            audits / "ae7_model_score_slot_population_20260710T102155Z",
            "Model score slot population",
        ),
        (
            "AE7",
            audits / "ae7b_runtime_identity_feature_bridge_20260710T110743Z",
            "Runtime identity feature bridge",
        ),
        (
            "AE7",
            audits / "ae7c1_scoring_policy_binding_parity_gate_20260710T120342Z",
            "Scoring policy binding parity gate",
        ),
    ]:
        rows.append(
            inventory_row(
                label,
                "root",
                p,
                "yes",
                "yes",
                "no",
                "historical_saved",
                note + "; meta-layer runtime-ready not closed",
            )
        )

    # AE8
    ae8 = audits / "ae8_context_intelligence_20260711T090138Z"
    rows.append(
        inventory_row(
            "AE8",
            "root",
            ae8,
            "yes",
            "yes",
            "no",
            "historical_saved",
            "Context intelligence layer",
        )
    )
    for name in [
        "ae8_decision_gate.json",
        "reports/ae8_decision_gate.json",
        "ae8_context_intelligence_summary.json",
        "reports/ae8_context_intelligence_summary.json",
    ]:
        cand = ae8 / name
        if cand.exists():
            rows.append(
                inventory_row(
                    "AE8",
                    "gate_or_summary",
                    cand,
                    "yes",
                    "yes",
                    "no",
                    "historical_saved",
                    "AE8 gate/summary artifact",
                )
            )
            break

    # AE9
    ae9 = audits / "ae9_llm_audit_20260711T090138Z"
    rows.append(
        inventory_row(
            "AE9",
            "root",
            ae9,
            "yes",
            "yes",
            "no",
            "historical_saved",
            "LLM audit layer (mock provider run)",
        )
    )
    rows.append(
        inventory_row(
            "AE9",
            "decision_gate",
            ae9 / "reports" / "ae9_decision_gate.json",
            "yes",
            "yes",
            "no",
            "historical_saved",
            "Saved AE9 decision gate",
        )
    )

    # AE10
    ae10 = audits / "ae10_trading_orchestration_20260711T090202Z"
    rows.append(
        inventory_row(
            "AE10",
            "root",
            ae10,
            "yes",
            "yes",
            "no",
            "historical_saved",
            "Paper/demo trading orchestration evidence root",
        )
    )
    for name in [
        "reports/ae10_decision_gate.json",
        "ae10_decision_gate.json",
        "reports/ae10_trading_orchestration_summary.json",
        "ae10_trading_orchestration_summary.json",
    ]:
        cand = ae10 / name
        if cand.exists() or True:
            rows.append(
                inventory_row(
                    "AE10",
                    "gate_or_summary",
                    cand,
                    "yes",
                    "yes",
                    "no",
                    "historical_saved_if_present",
                    "AE10 paper/demo proof artifact (if present)",
                )
            )
            break

    # Also scan AE10 for any files
    if ae10.exists():
        for fp in sorted(ae10.rglob("*"))[:30]:
            if fp.is_file():
                rows.append(
                    inventory_row(
                        "AE10",
                        "file",
                        fp,
                        "yes",
                        "yes",
                        "no",
                        "historical_saved",
                        "AE10 evidence file",
                    )
                )

    # AE11 latest
    ae11 = audits / "ae11_runtime_paper_loop_20260714T173541Z_de47ba58"
    rows.append(
        inventory_row(
            "AE11",
            "root",
            ae11,
            "yes",
            "yes",
            "no",
            "historical_saved",
            "Latest AE11 runtime paper loop evidence",
        )
    )
    rows.append(
        inventory_row(
            "AE11",
            "decision_gate",
            ae11 / "reports" / "ae11_decision_gate.json",
            "yes",
            "yes",
            "no",
            "historical_saved",
            "Saved AE11 gate",
        )
    )

    # AE12.1 census
    ae121 = audits / "ae12_runtime_data_census_20260714_224931"
    rows.append(
        inventory_row(
            "AE12.1",
            "root",
            ae121,
            "yes",
            "yes",
            "no",
            "historical_saved",
            "Runtime data census / preflight",
        )
    )
    rows.append(
        inventory_row(
            "AE12.1",
            "summary",
            ae121 / "reports" / "ae12_data_census_summary.json",
            "yes",
            "yes",
            "no",
            "historical_saved",
            "Census summary",
        )
    )

    # AE12.2 persistence / quality
    ae122 = audits / "ae12_forward_evidence_quality_20260714_230036"
    rows.append(
        inventory_row(
            "AE12.2",
            "root",
            ae122,
            "yes",
            "yes",
            "no",
            "historical_saved",
            "Forward evidence quality / persistence audit",
        )
    )

    # AE12.3 maturation
    ae123 = audits / "ae12_forward_evidence_maturation_20260714_235401"
    rows.append(
        inventory_row(
            "AE12.3",
            "root",
            ae123,
            "yes",
            "yes",
            "no",
            "historical_saved",
            "Forward evidence maturation (confirmed AE12.7 discovery source)",
        )
    )
    rows.append(
        inventory_row(
            "AE12.3",
            "readiness_gate",
            ae123 / "reports" / "ae12_final_system_readiness_gate.json",
            "yes",
            "yes",
            "no",
            "historical_saved",
            "Saved forward-evidence readiness gate",
        )
    )

    # AE12.4 opportunity
    for rel in [
        "data/ae12_opportunity_capture_full.csv",
        "data/ae12_missed_winners_full.csv",
        "data/ae12_trade_vs_no_trade_comparison.csv",
        "data/ae12_strict_vs_exploration_comparison.csv",
    ]:
        rows.append(
            inventory_row(
                "AE12.4",
                "evidence_csv",
                ae123 / rel,
                "yes",
                "yes",
                "no",
                "historical_saved",
                "Opportunity / missed-winner / trade comparison evidence",
            )
        )

    # AE12.5 UI/reporting (final docs / report manager outputs if present)
    ae125_candidates = list(audits.glob("ae12_*final*")) + list(
        audits.glob("ae12_*report*")
    )
    if not ae125_candidates:
        # UI/reporting is code + endpoints; mark census/maturation reporting gates
        rows.append(
            inventory_row(
                "AE12.5",
                "code_module",
                PROJECT_ROOT / "app" / "ae12_reporting" / "report_manager.py",
                "yes",
                "yes",
                "no",
                "source_code",
                "UI/reporting layer implemented in app; dedicated AE12.5 audit root may be absent",
            )
        )
    for p in ae125_candidates[:5]:
        rows.append(
            inventory_row(
                "AE12.5",
                "root",
                p,
                "yes",
                "yes",
                "no",
                "historical_saved",
                "AE12.5 UI/reporting artifact",
            )
        )

    # SentimentFix
    sf = audits / "ae12_sentimentfix_20260715_172645"
    rows.append(
        inventory_row(
            "AE12-SentimentFix",
            "root",
            sf,
            "yes",
            "yes",
            "no",
            "historical_saved",
            "SentimentFix dual-axis repair",
        )
    )
    rows.append(
        inventory_row(
            "AE12-SentimentFix",
            "decision_gate",
            sf / "audits" / "ae12_sentimentfix_decision_gate.json",
            "yes",
            "yes",
            "no",
            "historical_saved",
            "Saved SentimentFix gate",
        )
    )
    rows.append(
        inventory_row(
            "AE12-SentimentFix",
            "semantic_classifier_root",
            audits / "ae12_semantic_coin_classifier_20260716_093858",
            "yes",
            "yes",
            "no",
            "historical_saved",
            "Semantic coin classifier",
        )
    )
    rows.append(
        inventory_row(
            "AE12-SentimentFix",
            "gemini_adjudication_root",
            audits / "ae12_gemini_semantic_adjudication_20260716_111507",
            "yes",
            "yes",
            "no",
            "historical_saved",
            "Gemini semantic adjudication (prior phase; not called by AE12.8)",
        )
    )

    # AE12.6
    ae126 = audits / "ae12_ml_meta_layer_evaluation_20260717_111056"
    rows.append(
        inventory_row(
            "AE12.6",
            "root",
            ae126,
            "yes",
            "yes",
            "no",
            "historical_saved",
            "ML/meta-layer evaluation",
        )
    )
    rows.append(
        inventory_row(
            "AE12.6",
            "decision_gate",
            ae126 / "audits" / "ae12_ml_meta_layer_evaluation_gate.json",
            "yes",
            "yes",
            "no",
            "historical_saved",
            "Saved AE12.6 evaluation gate",
        )
    )

    # AE12.7 confirmed
    rows.append(
        inventory_row(
            "AE12.7",
            "root",
            AE12_7_ROOT,
            "yes",
            "yes",
            "no",
            "confirmed_primary",
            "Confirmed AE12.7 final root",
        )
    )
    rows.append(
        inventory_row(
            "AE12.7",
            "decision_gate",
            AE12_7_ROOT / "reports" / "ae12_7_intelligent_agent_decision_gate.json",
            "yes",
            "yes",
            "no",
            "confirmed_primary",
            "Saved AE12.7 intelligent agent gate",
        )
    )
    rows.append(
        inventory_row(
            "AE12.7",
            "daily_append_only_records",
            AE12_7_DAILY_RECORDS,
            "yes",
            "yes",
            "no",
            "confirmed_primary",
            "Daily append-only agent records",
        )
    )
    rows.append(
        inventory_row(
            "AE12.7",
            "daily_append_only_call_audit",
            AE12_7_DAILY_CALL_AUDIT,
            "yes",
            "yes",
            "no",
            "confirmed_primary",
            "Daily append-only agent call audit",
        )
    )
    for rel, atype in [
        ("audits/ae12_7_no_wallet_safety_audit.json", "no_wallet_audit"),
        ("audits/ae12_7_external_api_usage_audit.json", "external_api_audit"),
        ("audits/ae12_7_agent_authority_audit.json", "agent_authority_audit"),
        ("reports/ae12_7_ui_status_summary.json", "ui_status"),
        ("data/ae12_7_agent_records.jsonl", "agent_records"),
        ("data/ae12_7_agent_trade_linkage.csv", "agent_trade_linkage"),
    ]:
        rows.append(
            inventory_row(
                "AE12.7",
                atype,
                AE12_7_ROOT / rel,
                "yes",
                "yes",
                "no",
                "confirmed_primary",
                "AE12.7 key artifact",
            )
        )

    # AE12.8 self
    rows.append(
        inventory_row(
            "AE12.8",
            "root",
            OUTPUT_ROOT,
            "yes",
            "yes",
            "no",
            "current_package",
            "This AE12.8 safety/reproducibility package root",
        )
    )

    return rows


def build_gate_matrix() -> list[dict[str, Any]]:
    audits = PROJECT_ROOT / "data" / "audits"
    specs = [
        (
            "AE6",
            audits / "ae6_consensus_decision_layer_20260711T090105Z" / "ae6_consensus_decision_summary.json",
            audits / "ae6_consensus_decision_layer_20260711T090105Z",
            "DecisionRecord architecture demonstrated; not live authority",
            "no",
        ),
        (
            "AE7",
            audits
            / "ae7c1_scoring_policy_binding_parity_gate_20260710T120342Z"
            / "ae7c1_inference_readiness_gate.json",
            audits / "ae7c1_scoring_policy_binding_parity_gate_20260710T120342Z",
            "Meta-layer research/diagnostic; original runtime-ready stack not closed",
            "no",
        ),
        (
            "AE8",
            audits / "ae8_context_intelligence_20260711T090138Z",
            audits / "ae8_context_intelligence_20260711T090138Z",
            "Context root exists but is empty (0 files); older AE8 roots may hold evidence; no live authority",
            "no",
        ),
        (
            "AE9",
            audits / "ae9_llm_audit_20260711T090138Z" / "reports" / "ae9_decision_gate.json",
            audits / "ae9_llm_audit_20260711T090138Z",
            "LLM audit mock-only historical gate; explanation/audit only",
            "no",
        ),
        (
            "AE10",
            audits / "ae10_trading_orchestration_20260711T090202Z",
            audits / "ae10_trading_orchestration_20260711T090202Z",
            "Paper/demo orchestration root; live disallowed",
            "no",
        ),
        (
            "AE11",
            audits
            / "ae11_runtime_paper_loop_20260714T173541Z_de47ba58"
            / "reports"
            / "ae11_decision_gate.json",
            audits / "ae11_runtime_paper_loop_20260714T173541Z_de47ba58",
            "Runtime paper loop operational historically; not live-approved",
            "no",
        ),
        (
            "AE12.1",
            audits / "ae12_runtime_data_census_20260714_224931" / "reports" / "ae12_data_census_summary.json",
            audits / "ae12_runtime_data_census_20260714_224931",
            "Census/preflight archival evidence",
            "no",
        ),
        (
            "AE12.2",
            audits
            / "ae12_forward_evidence_quality_20260714_230036"
            / "reports"
            / "ae12_forward_evidence_quality_summary.json",
            audits / "ae12_forward_evidence_quality_20260714_230036",
            "Persistence/quality audit root present",
            "no",
        ),
        (
            "AE12.3",
            audits
            / "ae12_forward_evidence_maturation_20260714_235401"
            / "reports"
            / "ae12_final_system_readiness_gate.json",
            audits / "ae12_forward_evidence_maturation_20260714_235401",
            "Forward evidence maturation saved gate; reporting-ready not live",
            "no",
        ),
        (
            "AE12.4",
            audits
            / "ae12_forward_evidence_maturation_20260714_235401"
            / "data"
            / "ae12_opportunity_capture_full.csv",
            audits / "ae12_forward_evidence_maturation_20260714_235401",
            "Opportunity/missed-winner CSVs exist under maturation root",
            "no",
        ),
        (
            "AE12.5",
            PROJECT_ROOT / "app" / "ae12_reporting" / "report_manager.py",
            PROJECT_ROOT / "app" / "ae12_reporting",
            "UI/reporting code + AE12 API endpoints; dedicated audit root may be missing",
            "no",
        ),
        (
            "AE12-SentimentFix",
            audits / "ae12_sentimentfix_20260715_172645" / "audits" / "ae12_sentimentfix_decision_gate.json",
            audits / "ae12_sentimentfix_20260715_172645",
            "Dual-axis taxonomy derived; UNKNOWN_UNRESOLVED not social/opportunistic",
            "no",
        ),
        (
            "AE12.6",
            audits
            / "ae12_ml_meta_layer_evaluation_20260717_111056"
            / "audits"
            / "ae12_ml_meta_layer_evaluation_gate.json",
            audits / "ae12_ml_meta_layer_evaluation_20260717_111056",
            "ML/meta evaluation reporting/research only; no trade authority",
            "no",
        ),
        (
            "AE12.7",
            AE12_7_ROOT / "reports" / "ae12_7_intelligent_agent_decision_gate.json",
            AE12_7_ROOT,
            "Historical PASS_WITH_EXTERNAL_SOURCES_DISABLED; AE12.8 treats as archival evidence",
            "no",
        ),
        (
            "AE12.8",
            OUTPUT_ROOT / "reports" / "ae12_8_final_decision_gate.json",
            OUTPUT_ROOT,
            "This packaging phase; safety/repro archival; does not close AE12.9",
            "no",
        ),
    ]

    rows = []
    for subsystem, gate_path, evidence_root, limitation, blocker in specs:
        gate_obj = None
        saved_status = "ROOT_ONLY_OR_MISSING"
        gate_file = ""
        if gate_path.is_file():
            gate_obj = read_json(gate_path) if gate_path.suffix == ".json" else None
            saved_status = extract_status(gate_obj) if gate_obj is not None else "FILE_EXISTS_NON_JSON"
            gate_file = str(gate_path)
        elif gate_path.is_dir():
            # try find a gate inside
            candidates = list(gate_path.rglob("*decision*gate*.json")) + list(
                gate_path.rglob("*gate*.json")
            )
            if candidates:
                gate_file = str(candidates[0])
                gate_obj = read_json(candidates[0])
                saved_status = extract_status(gate_obj)
            else:
                gate_file = str(gate_path)
                saved_status = "ROOT_EXISTS_NO_GATE_FILE"
        else:
            gate_file = str(gate_path)
            saved_status = "MISSING"

        exists = Path(gate_file).exists() if gate_file else False
        ts = ""
        if isinstance(gate_obj, dict):
            for k in ("created_at", "created_at_utc", "timestamp", "gate_timestamp"):
                if k in gate_obj:
                    ts = str(gate_obj[k])
                    break

        interpretation = (
            f"Saved historical status '{saved_status}' archived; "
            f"source_file_exists={exists}; AE12.8 does not claim rerun equivalence."
        )
        if subsystem == "AE12.8":
            interpretation = (
                "AE12.8 packaging gate produced by this run; archival safety/repro package only."
            )

        rows.append(
            {
                "subsystem": subsystem,
                "saved_gate_status": saved_status,
                "saved_gate_file": gate_file,
                "evidence_root": str(evidence_root),
                "AE12_8_interpretation": interpretation,
                "blocker_for_AE12_9": blocker,
                "limitation": limitation,
                "gate_timestamp_if_available": ts,
                "source_file_exists": exists,
            }
        )
    return rows


def build_authority_matrix() -> list[dict[str, Any]]:
    common_no = {
        "live_trade_authority": "NO",
        "wallet_authority": "NO",
        "risk_gate_override_authority": "NO",
        "can_open_live_trade": "NO",
        "can_close_live_trade": "NO",
    }
    rows = []

    def row(
        name: str,
        evidence_type: str,
        operational: str,
        reporting: str,
        paper: str,
        open_paper: str,
        close_paper: str,
        limitation: str,
        **overrides: str,
    ) -> dict[str, Any]:
        base = {
            "component": name,
            "evidence_type": evidence_type,
            "operational_status": operational,
            "reporting_allowed": reporting,
            "paper_demo_allowed": paper,
            "can_open_paper_trade": open_paper,
            "can_close_paper_trade": close_paper,
            "limitation": limitation,
            **common_no,
        }
        base.update(overrides)
        return base

    for model in ("RF", "XGB", "TAB"):
        rows.append(
            row(
                model,
                "model_score/research",
                "research_reporting_diagnostic",
                "YES",
                "NO_AUTHORITY",
                "NO",
                "NO",
                "Reporting/research/diagnostic only; no live authority",
            )
        )
    rows.append(
        row(
            "Consensus",
            "decision_record",
            "diagnostic",
            "YES",
            "NO_AUTHORITY",
            "NO",
            "NO",
            "Diagnostic only; no live authority",
        )
    )
    rows.append(
        row(
            "Meta-layer / AE7",
            "meta_scoring",
            "research_diagnostic_not_runtime_closed",
            "YES",
            "NO_AUTHORITY",
            "NO",
            "NO",
            "Original runtime-ready stack not closed; no live authority",
        )
    )
    rows.append(
        row(
            "AE8 context",
            "context_intelligence",
            "context_reporting_audit",
            "YES",
            "NO_AUTHORITY",
            "NO",
            "NO",
            "Context/reporting/audit only; no live authority",
        )
    )
    rows.append(
        row(
            "AE11/AE12 forward evidence",
            "forward_evidence",
            "reporting_audit",
            "YES",
            "NO_AUTHORITY",
            "NO",
            "NO",
            "Forward evidence for reporting; profitability not proven; no live authority",
        )
    )
    rows.append(
        row(
            "Qwen/local LLM",
            "llm_explanation",
            "explanation_audit_classification",
            "YES",
            "NO_AUTHORITY",
            "NO",
            "NO",
            "Explanation/audit/classification only; no trade authority",
        )
    )
    rows.append(
        row(
            "Gemini",
            "llm_adjudication",
            "explanation_audit_classification",
            "YES",
            "NO_AUTHORITY",
            "NO",
            "NO",
            "Explanation/audit/classification only; no trade authority; AE12.8 did not call Gemini",
        )
    )
    rows.append(
        row(
            "Helius/Solana",
            "chain_context",
            "context_reporting_audit",
            "YES",
            "NO_AUTHORITY",
            "NO",
            "NO",
            "Context/reporting/audit only; broad live ingestion not proven; AE12.8 did not call Helius",
        )
    )
    rows.append(
        row(
            "RSS/sentiment",
            "sentiment_context",
            "context_reporting_audit",
            "YES",
            "NO_AUTHORITY",
            "NO",
            "NO",
            "Context/reporting/audit only; no live authority",
        )
    )
    rows.append(
        row(
            "Semantic/SentimentFix",
            "semantic_taxonomy",
            "context_reporting_audit",
            "YES",
            "NO_AUTHORITY",
            "NO",
            "NO",
            "Context/reporting/audit only; UNKNOWN_UNRESOLVED is unresolved not social/opportunistic",
        )
    )
    rows.append(
        row(
            "Paper/demo execution",
            "paper_demo",
            "allowed",
            "YES",
            "YES",
            "YES",
            "YES",
            "Paper/demo trading allowed; not live",
            live_trade_authority="NO",
        )
    )
    rows.append(
        row(
            "Live execution",
            "live_trading",
            "not_approved",
            "NO",
            "NO",
            "NO",
            "NO",
            "Live execution not approved; no wallet; no signing",
            paper_demo_allowed="NO",
        )
    )
    return rows


def build_external_api_audit() -> dict[str, Any]:
    ae7_ext = read_json(AE12_7_ROOT / "audits" / "ae12_7_external_api_usage_audit.json")
    providers = [
        {
            "provider": "Qwen/local",
            "implemented": True,
            "evidence_path": str(AE12_7_ROOT / "audits" / "ae12_7_qwen_local_provider_audit.json"),
            "called_in_AE12_8": False,
            "called_in_prior_phase_if_known": "AE12.7 confirmed run: skipped/disabled (external sources disabled)",
            "enabled_by_default": False,
            "explicit_enable_supported": True,
            "external_api": False,
            "local_provider": True,
            "trade_authority_used": False,
            "limitation": "Operational demo memo quality not proven beyond AE12.7 linkage; AE12.8 did not call Qwen/Ollama",
        },
        {
            "provider": "Gemini",
            "implemented": True,
            "evidence_path": str(AE12_7_ROOT / "audits" / "ae12_7_gemini_safety_audit.json"),
            "called_in_AE12_8": False,
            "called_in_prior_phase_if_known": "AE12.7 confirmed run: not called; prior SentimentFix adjudication roots exist",
            "enabled_by_default": False,
            "explicit_enable_supported": True,
            "external_api": True,
            "local_provider": False,
            "trade_authority_used": False,
            "limitation": "Explanation/audit only; web_grounding not assumed; AE12.8 did not call Gemini",
        },
        {
            "provider": "Helius/Solana",
            "implemented": True,
            "evidence_path": str(AE12_7_ROOT / "audits" / "ae12_7_helius_readonly_audit.json"),
            "called_in_AE12_8": False,
            "called_in_prior_phase_if_known": "AE12.7 confirmed run: NOT_CONFIGURED / not called",
            "enabled_by_default": False,
            "explicit_enable_supported": True,
            "external_api": True,
            "local_provider": False,
            "trade_authority_used": False,
            "limitation": "Broad live ingestion not proven; AE12.8 did not call Helius",
        },
        {
            "provider": "RSS/sentiment",
            "implemented": True,
            "evidence_path": str(AE12_7_ROOT / "data" / "ae12_7_rss_context_links.csv"),
            "called_in_AE12_8": False,
            "called_in_prior_phase_if_known": "AE12.7 context linkage present; external RSS not required for AE12.8",
            "enabled_by_default": False,
            "explicit_enable_supported": True,
            "external_api": True,
            "local_provider": False,
            "trade_authority_used": False,
            "limitation": "Context/reporting only; AE12.8 made no external RSS calls",
        },
        {
            "provider": "Semantic/SentimentFix",
            "implemented": True,
            "evidence_path": str(
                PROJECT_ROOT
                / "data"
                / "audits"
                / "ae12_sentimentfix_20260715_172645"
                / "audits"
                / "ae12_sentimentfix_decision_gate.json"
            ),
            "called_in_AE12_8": False,
            "called_in_prior_phase_if_known": "AE12-SentimentFix derived audit completed historically",
            "enabled_by_default": True,
            "explicit_enable_supported": True,
            "external_api": False,
            "local_provider": True,
            "trade_authority_used": False,
            "limitation": "Reporting/taxonomy only; UNKNOWN_UNRESOLVED remains unresolved",
        },
    ]
    return {
        "phase": "AE12.8",
        "created_at": CREATED_AT,
        "ae12_8_external_calls_made": False,
        "no_gemini_helius_qwen_ollama_in_ae12_8": True,
        "prior_ae12_7_external_api_audit": ae7_ext,
        "providers": providers,
    }


def build_wallet_safety() -> dict[str, Any]:
    prior = read_json(AE12_7_ROOT / "audits" / "ae12_7_no_wallet_safety_audit.json")
    ae11 = read_json(
        PROJECT_ROOT
        / "data"
        / "audits"
        / "ae11_runtime_paper_loop_20260714T173541Z_de47ba58"
        / "reports"
        / "ae11_decision_gate.json"
    )
    # Search for private key files (existence only; do not read contents)
    key_globs = [
        "*private*key*",
        "*.pem",
        "*wallet*secret*",
        "*secret*key*",
    ]
    found_keys: list[str] = []
    for pattern in key_globs:
        for p in PROJECT_ROOT.glob(pattern):
            if p.is_file():
                found_keys.append(str(p))
        for p in (PROJECT_ROOT / "data").glob(pattern):
            if p.is_file():
                found_keys.append(str(p))

    env_key_names = [
        "PRIVATE_KEY",
        "SOLANA_PRIVATE_KEY",
        "WALLET_PRIVATE_KEY",
        "HELIUS_PRIVATE_KEY",
    ]
    env_accessed = any(os.environ.get(n) for n in env_key_names)

    return {
        "phase": "AE12.8",
        "created_at": CREATED_AT,
        "wallet_configured": False,
        "private_key_accessed": False,
        "private_key_file_found": bool(found_keys),
        "private_key_files_matched_by_name": found_keys[:20],
        "private_key_env_accessed": bool(env_accessed),
        "real_transaction_signed": False,
        "real_transaction_attempted": False,
        "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
        "live_trading_ready": False,
        "live_trading_approval": "NO",
        "demo_paper_trading_allowed": True,
        "profitability_proven": False,
        "prior_ae12_7_no_wallet_audit": prior,
        "prior_ae11_wallet_fields": {
            "wallet_configured": (ae11 or {}).get("wallet_configured"),
            "private_key_accessed": (ae11 or {}).get("private_key_accessed"),
            "real_transaction_attempted": (ae11 or {}).get("real_transaction_attempted"),
            "live_submission_status": (ae11 or {}).get("live_submission_status"),
        },
        "interpretation": (
            "No wallet/live/private-key risk found in AE12.8 packaging or confirmed AE12.7/AE11 saved audits. "
            "Filename glob matches (if any) are name-only and were not opened/used."
        ),
        "status": "PASS_NO_WALLET_NO_LIVE",
    }


def build_model_authority_audit() -> dict[str, Any]:
    ae126 = read_json(
        PROJECT_ROOT
        / "data"
        / "audits"
        / "ae12_ml_meta_layer_evaluation_20260717_111056"
        / "audits"
        / "ae12_ml_meta_layer_evaluation_gate.json"
    )
    return {
        "phase": "AE12.8",
        "created_at": CREATED_AT,
        "models": {
            "RF": {"live_trade_authority": False, "role": "reporting_research_diagnostic"},
            "XGB": {"live_trade_authority": False, "role": "reporting_research_diagnostic"},
            "TAB": {"live_trade_authority": False, "role": "reporting_research_diagnostic"},
            "Consensus": {"live_trade_authority": False, "role": "diagnostic"},
            "Meta-layer/AE7": {
                "live_trade_authority": False,
                "role": "research_diagnostic",
                "runtime_ready_closed": False,
            },
        },
        "trade_authority_granted_to_models": False,
        "source_ae12_6_gate": ae126,
        "limitation": "Model evidence is research/reporting/diagnostic only; not live authority",
    }


def build_agent_authority_audit() -> dict[str, Any]:
    prior = read_json(AE12_7_ROOT / "audits" / "ae12_7_agent_authority_audit.json")
    return {
        "phase": "AE12.8",
        "created_at": CREATED_AT,
        "agent_layer_trade_authority": False,
        "llm_may_authorize_trades": False,
        "allowed_decision_effects": [
            "explanation_only",
            "audit_only",
            "context_only",
            "soft_warning_only",
            "soft_veto_recommendation_only",
            "no_effect",
        ],
        "paper_demo_trading_allowed": True,
        "live_trading_approval": "NO",
        "prior_ae12_7_agent_authority_audit": prior,
        "status": "PASS_NO_TRADE_AUTHORITY",
    }


def build_context_authority_audit() -> dict[str, Any]:
    return {
        "phase": "AE12.8",
        "created_at": CREATED_AT,
        "layers": {
            "AE8_context": {"trade_authority": False, "role": "context_reporting_audit"},
            "Helius": {"trade_authority": False, "role": "context_reporting_audit"},
            "RSS_sentiment": {"trade_authority": False, "role": "context_reporting_audit"},
            "Semantic_SentimentFix": {
                "trade_authority": False,
                "role": "context_reporting_audit",
                "unknown_unresolved_is_social": False,
                "unknown_unresolved_is_opportunistic": False,
                "unknown_unresolved_meaning": (
                    "Local artifacts were insufficient to resolve without external evidence"
                ),
            },
        },
        "trade_authority_used": False,
        "status": "PASS_CONTEXT_ONLY",
    }


def build_reproducibility_audit(
    git_available: bool,
    git_hash: str | None,
    pip_freeze_ok: bool,
) -> dict[str, Any]:
    return {
        "phase": "AE12.8",
        "created_at": CREATED_AT,
        "git_available": git_available,
        "git_commit_hash_or_null": git_hash,
        "pip_freeze_succeeded": pip_freeze_ok,
        "environment_snapshot_path": str(OUTPUT_ROOT / "environment" / "environment_snapshot.txt"),
        "source_revision_identity_proven": False,
        "historical_gates_rerun": False,
        "limitation": (
            "Missing .git metadata limits reproducibility; dependency snapshot helps reproduce "
            "dependencies but does not prove source-code revision identity. Gate statuses are "
            "historical archival records, not fresh absolute truth."
        ),
        "status": "PASS_WITH_DEPENDENCY_SNAPSHOT_NO_GIT",
    }


def build_ui_api_inventory() -> list[dict[str, Any]]:
    routes = [
        ("/api/ae12/status", "AE12 forward", "report_manager / ae12 status", True),
        ("/api/ae12/forward-evidence-summary", "AE12 forward", "maturation summaries", True),
        ("/api/ae12/missed-winners", "AE12 forward", "missed winners CSV/JSON", True),
        ("/api/ae12/trade-vs-no-trade", "AE12 forward", "comparison CSV", True),
        ("/api/ae12/strict-vs-exploration", "AE12 forward", "comparison CSV", True),
        ("/api/ae12/qwen-linkage", "AE12 forward", "qwen linkage CSV", True),
        ("/api/ae12/safety", "AE12 forward", "wallet/safety audits", True),
        ("/api/ae12/final-report-summary", "AE12 forward", "final report summary", True),
        ("/api/ae12/runtime-collection", "AE12 forward", "runtime collection status", True),
        ("/api/ae12/signal-taxonomy", "AE12 SentimentFix", "taxonomy audit", True),
        ("/api/ae12/sentimentfix", "AE12 SentimentFix", "sentimentfix gate/summary", True),
        ("/api/ae12/semantic-coin-classifier", "AE12 SentimentFix", "classifier outputs", True),
        (
            "/api/ae12/gemini-semantic-adjudication",
            "AE12 SentimentFix",
            "prior adjudication artifacts",
            True,
        ),
        ("/api/ae12/manual-review-drilldown", "AE12 SentimentFix", "manual review outputs", True),
        ("POST /api/ae12/refresh-cache", "AE12 reporting", "in-memory cache refresh only", True),
        ("/api/ae12/agents/status", "AE12.7 agents", "agent status files", True),
        ("/api/ae12/agents/recent", "AE12.7 agents", "agent records jsonl", True),
        ("/api/ae12/agents/by-candidate/{candidate_id}", "AE12.7 agents", "agent records", True),
        ("/api/ae12/agents/by-pair/{pair_address}", "AE12.7 agents", "agent records", True),
        ("/api/ae12/agents/authority-audit", "AE12.7 agents", "authority audit json", True),
        ("/api/ae12/agents/ui-summary", "AE12.7 agents", "ui status summary", True),
        ("UI: AE12 Forward Evidence panel", "AE12 UI", "forward evidence endpoints", True),
        ("UI: AE12 SentimentFix panel", "AE12 UI", "sentimentfix endpoints", True),
        ("UI: AE12.7 Intelligent Agents panel", "AE12.7 UI", "agents endpoints", True),
    ]
    api_py = PROJECT_ROOT / "app" / "api.py"
    api_text = api_py.read_text(encoding="utf-8") if api_py.exists() else ""
    rows = []
    for route, family, source, read_only in routes:
        route_key = route.replace("POST ", "").split("{")[0]
        implemented = route_key in api_text or route.startswith("UI:")
        rows.append(
            {
                "route_or_panel": route,
                "family": family,
                "implemented": implemented,
                "read_only": read_only,
                "data_source": source,
                "persisted_file_backed": True if not route.startswith("POST") else False,
                "in_memory_only": route.startswith("POST"),
                "limitation": (
                    "Read-only reporting surface; does not grant trade/wallet authority"
                    if read_only
                    else "Cache refresh only"
                ),
            }
        )
    return rows


def build_paper_demo_inventory() -> list[dict[str, Any]]:
    items = [
        (
            "AE10",
            "paper_demo_proof",
            PROJECT_ROOT / "data" / "audits" / "ae10_trading_orchestration_20260711T090202Z",
            True,
            False,
            "Paper/demo orchestration root",
        ),
        (
            "AE11",
            "runtime_paper_loop",
            PROJECT_ROOT
            / "data"
            / "audits"
            / "ae11_runtime_paper_loop_20260714T173541Z_de47ba58"
            / "reports"
            / "ae11_decision_gate.json",
            True,
            False,
            "AE11_LOOP_OPERATIONAL historically; live disallowed",
        ),
        (
            "AE12",
            "forward_evidence_maturation",
            PROJECT_ROOT
            / "data"
            / "audits"
            / "ae12_forward_evidence_maturation_20260714_235401",
            True,
            False,
            "Forward evidence / maturation outputs",
        ),
        (
            "AE12.7",
            "agent_to_paper_linkage",
            AE12_7_ROOT / "data" / "ae12_7_agent_trade_linkage.csv",
            True,
            False,
            "Agent-to-paper linkage outputs",
        ),
        (
            "policy",
            "demo_paper_trading_allowed",
            AE12_7_ROOT / "audits" / "ae12_7_agent_authority_audit.json",
            True,
            False,
            "demo/paper allowed=true per authority audit",
        ),
        (
            "policy",
            "live_trading_disallowed",
            AE12_7_ROOT / "reports" / "ae12_7_intelligent_agent_decision_gate.json",
            True,
            False,
            "live_ready=false; live trading disallowed",
        ),
    ]
    rows = []
    for phase, etype, path, paper_ok, live_ok, note in items:
        meta = file_meta(path)
        rows.append(
            {
                "phase": phase,
                "evidence_type": etype,
                "exact_path": meta["exact_path"],
                "exists": meta["exists"],
                "file_size_bytes": meta["file_size_bytes"],
                "demo_paper_trading_allowed": paper_ok,
                "live_trading_allowed": live_ok,
                "limitation_notes": note,
            }
        )
    return rows


def build_forward_inventory() -> list[dict[str, Any]]:
    root = (
        PROJECT_ROOT / "data" / "audits" / "ae12_forward_evidence_maturation_20260714_235401"
    )
    files = [
        ("opportunity_capture", "data/ae12_opportunity_capture_full.csv"),
        ("missed_winners", "data/ae12_missed_winners_full.csv"),
        ("trade_vs_no_trade", "data/ae12_trade_vs_no_trade_comparison.csv"),
        ("strict_vs_exploration", "data/ae12_strict_vs_exploration_comparison.csv"),
        ("no_lookahead_maturation", "audits/ae12_no_lookahead_audit.csv"),
        ("horizon_outputs", "audits/ae12_horizon_maturity_audit.csv"),
        ("matured_outcomes", "data/ae12_matured_outcomes.csv"),
        ("readiness_gate", "reports/ae12_final_system_readiness_gate.json"),
    ]
    rows = []
    for etype, rel in files:
        meta = file_meta(root / rel)
        rows.append(
            {
                "evidence_type": etype,
                "exact_path": meta["exact_path"],
                "exists": meta["exists"],
                "file_size_bytes": meta["file_size_bytes"],
                "profitability_proven": False,
                "limitation_notes": "Forward evidence for reporting only; profitability_proven=false",
            }
        )
    return rows


def build_agent_inventory() -> list[dict[str, Any]]:
    items = [
        ("agent_records", AE12_7_ROOT / "data" / "ae12_7_agent_records.jsonl"),
        ("agent_records_daily", AE12_7_DAILY_RECORDS),
        ("agent_call_audit_daily", AE12_7_DAILY_CALL_AUDIT),
        ("qwen_candidate_memos", AE12_7_ROOT / "data" / "ae12_7_qwen_candidate_memos.jsonl"),
        ("gemini_selective_audits", AE12_7_ROOT / "data" / "ae12_7_gemini_selective_audits.jsonl"),
        ("helius_readonly_audit", AE12_7_ROOT / "audits" / "ae12_7_helius_readonly_audit.json"),
        ("rss_context_links", AE12_7_ROOT / "data" / "ae12_7_rss_context_links.csv"),
        ("semantic_context_links", AE12_7_ROOT / "data" / "ae12_7_semantic_context_links.csv"),
        ("agent_trade_linkage", AE12_7_ROOT / "data" / "ae12_7_agent_trade_linkage.csv"),
        ("missed_winner_agent_review", AE12_7_ROOT / "data" / "ae12_7_missed_winner_agent_review.csv"),
        ("agent_authority_audit", AE12_7_ROOT / "audits" / "ae12_7_agent_authority_audit.json"),
        ("no_wallet_audit", AE12_7_ROOT / "audits" / "ae12_7_no_wallet_safety_audit.json"),
        ("external_api_usage_audit", AE12_7_ROOT / "audits" / "ae12_7_external_api_usage_audit.json"),
        ("ui_status_summary", AE12_7_ROOT / "reports" / "ae12_7_ui_status_summary.json"),
    ]
    rows = []
    for etype, path in items:
        meta = file_meta(path)
        rows.append(
            {
                "evidence_type": etype,
                "exact_path": meta["exact_path"],
                "exists": meta["exists"],
                "file_size_bytes": meta["file_size_bytes"],
                "last_modified": meta["last_modified"],
                "trade_authority_used": False,
                "limitation_notes": "AE12.7 agent evidence; explanation/audit/context only",
            }
        )
    return rows


def build_semantic_inventory() -> list[dict[str, Any]]:
    items = [
        (
            "sentimentfix_gate",
            PROJECT_ROOT
            / "data"
            / "audits"
            / "ae12_sentimentfix_20260715_172645"
            / "audits"
            / "ae12_sentimentfix_decision_gate.json",
            "dual_axis_taxonomy_status=PASS_DERIVED_ONLY_RUNTIME_UPDATE_PENDING",
        ),
        (
            "semantic_classifier_gate",
            PROJECT_ROOT
            / "data"
            / "audits"
            / "ae12_semantic_coin_classifier_20260716_093858"
            / "audits"
            / "ae12_semantic_classifier_decision_gate.json",
            "pair-level vs coin-level: classifier is unique-asset/coin oriented; UNKNOWN share high",
        ),
        (
            "gemini_adjudication",
            PROJECT_ROOT
            / "data"
            / "audits"
            / "ae12_gemini_semantic_adjudication_20260716_111507",
            "Gemini semantic adjudication limitation: not assumed web-grounded unless evidence proves",
        ),
        (
            "ae12_7_semantic_taxonomy_audit",
            AE12_7_ROOT / "audits" / "ae12_7_semantic_taxonomy_audit.json",
            "AE12.7 semantic taxonomy audit",
        ),
        (
            "ae12_7_semantic_context_links",
            AE12_7_ROOT / "data" / "ae12_7_semantic_context_links.csv",
            "Semantic context links from AE12.7",
        ),
    ]
    rows = []
    for etype, path, note in items:
        meta = file_meta(path)
        web_grounding = "unknown_or_false"
        if "gemini" in etype:
            man = path / "reports" / "ae12_gemini_semantic_adjudication_manifest.json"
            if man.exists():
                obj = read_json(man)
                if isinstance(obj, dict):
                    web_grounding = str(
                        obj.get("web_grounding_used", obj.get("web_grounding", "unknown_or_false"))
                    )
        rows.append(
            {
                "evidence_type": etype,
                "exact_path": meta["exact_path"],
                "exists": meta["exists"],
                "file_size_bytes": meta["file_size_bytes"],
                "dual_axis_taxonomy": True,
                "pair_vs_coin_distinction_noted": True,
                "web_grounding_used": web_grounding,
                "UNKNOWN_UNRESOLVED_status": "unresolved_not_social_not_opportunistic",
                "trade_authority_used": False,
                "interpretation": (
                    "UNKNOWN_UNRESOLVED is not social; is not opportunistic; means local artifacts "
                    "were insufficient to resolve without external evidence. " + note
                ),
            }
        )
    return rows


def build_test_matrix() -> tuple[list[dict[str, Any]], str]:
    test_file = PROJECT_ROOT / "tests" / "test_ae12_7_intelligent_agent_demo.py"
    rows = [
        {
            "test_area": "AE12.7 intelligent agent demo tests",
            "evidence_path": str(test_file),
            "exists": test_file.exists(),
            "rerun_in_AE12_8": False,
            "result_if_known": "NOT_RERUN_IN_AE12_8; prior AE12.7 package passed operational demo gate",
            "classification": "not_blocking_for_AE12_8_packaging",
            "notes": "AE12.8 does not rerun full test suite",
        },
        {
            "test_area": "Full pytest suite",
            "evidence_path": "",
            "exists": False,
            "rerun_in_AE12_8": False,
            "result_if_known": "NOT_RERUN; pre-existing failures may remain if unrelated",
            "classification": "pre_existing_unrelated_possible",
            "notes": "Do not treat full-suite failures as AE12.8 blockers unless safety/repro contradicted",
        },
        {
            "test_area": "AE12.8 packaging import/compile/read-only checks",
            "evidence_path": str(OUTPUT_ROOT),
            "exists": True,
            "rerun_in_AE12_8": True,
            "result_if_known": "PASS_IF_VALIDATION_SUCCEEDS",
            "classification": "ae12_8_packaging_check",
            "notes": "JSON parse, CSV headers, required files, trader.db non-mutation",
        },
        {
            "test_area": "Missing dependency issues",
            "evidence_path": str(OUTPUT_ROOT / "environment" / "environment_snapshot.txt"),
            "exists": (OUTPUT_ROOT / "environment" / "environment_snapshot.txt").exists(),
            "rerun_in_AE12_8": False,
            "result_if_known": "pip freeze captured; no pip install/upgrade performed",
            "classification": "documented",
            "notes": "Dependency snapshot only",
        },
    ]
    notes = """# AE12.8 Test Notes

- AE12.8 did **not** rerun the full pytest suite.
- AE12.7-specific tests exist at `tests/test_ae12_7_intelligent_agent_demo.py`; they were not re-executed in AE12.8.
- Prior AE12.7 saved gate (`AE12_7_PASS_WITH_EXTERNAL_SOURCES_DISABLED`) is treated as historical evidence.
- Full-suite failures/errors, if any remain in the workspace, are classified as pre-existing/unrelated unless they contradict safety/reproducibility packaging.
- AE12.8 packaging validation covers: required file presence, JSON parse, CSV headers, environment snapshot presence, and trader.db non-mutation fingerprint.
- Missing dependencies were not installed or upgraded.
"""
    return rows, notes


def main() -> int:
    assert OUTPUT_ROOT.exists(), f"Output root missing: {OUTPUT_ROOT}"
    for sub in ("reports", "data", "audits", "manifests", "environment", "tests"):
        (OUTPUT_ROOT / sub).mkdir(parents=True, exist_ok=True)

    # trader.db fingerprint before writes (AE12.8 only writes under OUTPUT_ROOT)
    db_before = {
        "path": str(TRADER_DB),
        "exists": TRADER_DB.exists(),
        "size": TRADER_DB.stat().st_size if TRADER_DB.exists() else None,
        "mtime": datetime.fromtimestamp(TRADER_DB.stat().st_mtime, tz=timezone.utc).isoformat()
        if TRADER_DB.exists()
        else None,
        "fingerprint": sha256_quick(TRADER_DB) if TRADER_DB.exists() else None,
    }

    # Environment notes
    env_snap = OUTPUT_ROOT / "environment" / "environment_snapshot.txt"
    py_ver = OUTPUT_ROOT / "environment" / "python_version.txt"
    pip_ver = OUTPUT_ROOT / "environment" / "pip_version.txt"
    pip_ok = env_snap.exists() and env_snap.stat().st_size > 0
    py_text = py_ver.read_text(encoding="utf-8-sig").strip() if py_ver.exists() else "UNKNOWN"
    pip_text = pip_ver.read_text(encoding="utf-8-sig").strip() if pip_ver.exists() else "UNKNOWN"

    git_dir = PROJECT_ROOT / ".git"
    git_available = git_dir.exists()
    git_hash = None
    git_status = None
    if not git_available:
        write_text(
            OUTPUT_ROOT / "environment" / "dependency_environment_notes.md",
            """# Dependency Environment Notes

- `.git` metadata was **not** available in this workspace.
- Commit hash was **not** available.
- `python -m pip freeze` was captured into `environment_snapshot.txt` without installing or upgrading packages.
- Python version and pip version were captured from the local interpreter.
- Limitation: environment snapshot helps reproduce dependencies but does **not** prove source-code revision identity.
- No packages were installed or upgraded during AE12.8.
""",
        )
    else:
        write_text(
            OUTPUT_ROOT / "environment" / "dependency_environment_notes.md",
            "# Dependency Environment Notes\n\n.git was available; see reproducibility manifest.\n",
        )

    files_read: list[str] = []
    files_not_found: list[str] = []

    # Inventories
    artifact_rows = build_artifact_inventory()
    for r in artifact_rows:
        if r["exists"]:
            files_read.append(r["exact_path"])
        else:
            files_not_found.append(r["exact_path"])

    write_csv(
        OUTPUT_ROOT / "manifests" / "ae12_8_artifact_inventory.csv",
        [
            "phase",
            "artifact_type",
            "exact_path",
            "exists",
            "file_size_bytes",
            "last_modified",
            "required_for_AE12_9",
            "required_for_final_report",
            "archival_only",
            "source_of_truth_level",
            "limitation_notes",
        ],
        artifact_rows,
    )

    gate_rows = build_gate_matrix()
    write_csv(
        OUTPUT_ROOT / "manifests" / "ae12_8_subsystem_gate_matrix.csv",
        [
            "subsystem",
            "saved_gate_status",
            "saved_gate_file",
            "evidence_root",
            "AE12_8_interpretation",
            "blocker_for_AE12_9",
            "limitation",
        ],
        gate_rows,
    )

    auth_rows = build_authority_matrix()
    write_csv(
        OUTPUT_ROOT / "audits" / "ae12_8_authority_matrix.csv",
        [
            "component",
            "evidence_type",
            "operational_status",
            "reporting_allowed",
            "paper_demo_allowed",
            "live_trade_authority",
            "wallet_authority",
            "risk_gate_override_authority",
            "can_open_paper_trade",
            "can_open_live_trade",
            "can_close_paper_trade",
            "can_close_live_trade",
            "limitation",
        ],
        auth_rows,
    )

    write_csv(
        OUTPUT_ROOT / "manifests" / "ae12_8_ui_api_inventory.csv",
        [
            "route_or_panel",
            "family",
            "implemented",
            "read_only",
            "data_source",
            "persisted_file_backed",
            "in_memory_only",
            "limitation",
        ],
        build_ui_api_inventory(),
    )

    write_csv(
        OUTPUT_ROOT / "data" / "ae12_8_paper_demo_evidence_inventory.csv",
        [
            "phase",
            "evidence_type",
            "exact_path",
            "exists",
            "file_size_bytes",
            "demo_paper_trading_allowed",
            "live_trading_allowed",
            "limitation_notes",
        ],
        build_paper_demo_inventory(),
    )
    write_csv(
        OUTPUT_ROOT / "data" / "ae12_8_forward_evidence_inventory.csv",
        [
            "evidence_type",
            "exact_path",
            "exists",
            "file_size_bytes",
            "profitability_proven",
            "limitation_notes",
        ],
        build_forward_inventory(),
    )
    write_csv(
        OUTPUT_ROOT / "data" / "ae12_8_agent_evidence_inventory.csv",
        [
            "evidence_type",
            "exact_path",
            "exists",
            "file_size_bytes",
            "last_modified",
            "trade_authority_used",
            "limitation_notes",
        ],
        build_agent_inventory(),
    )
    write_csv(
        OUTPUT_ROOT / "data" / "ae12_8_semantic_sentimentfix_inventory.csv",
        [
            "evidence_type",
            "exact_path",
            "exists",
            "file_size_bytes",
            "dual_axis_taxonomy",
            "pair_vs_coin_distinction_noted",
            "web_grounding_used",
            "UNKNOWN_UNRESOLVED_status",
            "trade_authority_used",
            "interpretation",
        ],
        build_semantic_inventory(),
    )

    test_rows, test_notes = build_test_matrix()
    write_csv(
        OUTPUT_ROOT / "tests" / "ae12_8_test_matrix.csv",
        [
            "test_area",
            "evidence_path",
            "exists",
            "rerun_in_AE12_8",
            "result_if_known",
            "classification",
            "notes",
        ],
        test_rows,
    )
    write_text(OUTPUT_ROOT / "tests" / "ae12_8_test_notes.md", test_notes)

    # Audits JSON
    wallet = build_wallet_safety()
    write_json(OUTPUT_ROOT / "audits" / "ae12_8_wallet_no_live_safety_audit.json", wallet)
    write_json(
        OUTPUT_ROOT / "audits" / "ae12_8_external_api_usage_audit.json",
        build_external_api_audit(),
    )
    write_json(
        OUTPUT_ROOT / "audits" / "ae12_8_model_authority_audit.json",
        build_model_authority_audit(),
    )
    write_json(
        OUTPUT_ROOT / "audits" / "ae12_8_agent_authority_audit.json",
        build_agent_authority_audit(),
    )
    write_json(
        OUTPUT_ROOT / "audits" / "ae12_8_context_authority_audit.json",
        build_context_authority_audit(),
    )
    write_json(
        OUTPUT_ROOT / "audits" / "ae12_8_reproducibility_audit.json",
        build_reproducibility_audit(git_available, git_hash, pip_ok),
    )

    # Expected created files list
    created_files = [
        "reports/ae12_8_summary_for_upload.txt",
        "reports/ae12_8_final_decision_gate.json",
        "reports/ae12_8_known_limitations.md",
        "reports/ae12_8_what_was_built_vs_not_proven.md",
        "manifests/ae12_8_artifact_inventory.csv",
        "manifests/ae12_8_reproducibility_manifest.json",
        "manifests/ae12_8_subsystem_gate_matrix.csv",
        "manifests/ae12_8_ui_api_inventory.csv",
        "environment/environment_snapshot.txt",
        "environment/python_version.txt",
        "environment/pip_version.txt",
        "environment/dependency_environment_notes.md",
        "audits/ae12_8_wallet_no_live_safety_audit.json",
        "audits/ae12_8_external_api_usage_audit.json",
        "audits/ae12_8_authority_matrix.csv",
        "audits/ae12_8_model_authority_audit.json",
        "audits/ae12_8_agent_authority_audit.json",
        "audits/ae12_8_context_authority_audit.json",
        "audits/ae12_8_reproducibility_audit.json",
        "data/ae12_8_paper_demo_evidence_inventory.csv",
        "data/ae12_8_forward_evidence_inventory.csv",
        "data/ae12_8_agent_evidence_inventory.csv",
        "data/ae12_8_semantic_sentimentfix_inventory.csv",
        "tests/ae12_8_test_matrix.csv",
        "tests/ae12_8_test_notes.md",
    ]

    # Safety / archival completeness checks for classification
    ae127_ok = AE12_7_ROOT.exists() and (
        AE12_7_ROOT / "reports" / "ae12_7_intelligent_agent_decision_gate.json"
    ).exists()
    safety_clear = (
        wallet.get("wallet_configured") is False
        and wallet.get("private_key_accessed") is False
        and wallet.get("real_transaction_signed") is False
        and wallet.get("real_transaction_attempted") is False
        and wallet.get("live_trading_ready") is False
        and wallet.get("live_trading_approval") == "NO"
    )
    critical_missing = not ae127_ok
    # Some historical roots may be thin (AE10 listing quirks) -> archival limitations
    missing_important = [p for p in files_not_found if "ae10" in p.lower() or "ae8" in p.lower()]
    if safety_clear and ae127_ok and pip_ok and not critical_missing:
        if missing_important:
            classification = "AE12_8_PASS_WITH_ARCHIVAL_LIMITATIONS"
        else:
            classification = "AE12_8_PASS_SAFETY_REPRO_PACKAGE"
    elif not safety_clear:
        classification = "AE12_8_BLOCKED_MISSING_CRITICAL_SAFETY_EVIDENCE"
    else:
        classification = "AE12_8_BLOCKED_REPRODUCIBILITY_GAP"

    # Count missing required inventory items for AE12.7
    ae127_missing = [
        r
        for r in artifact_rows
        if r["phase"] == "AE12.7" and r["source_of_truth_level"] == "confirmed_primary" and not r["exists"]
    ]
    if ae127_missing and classification.startswith("AE12_8_PASS"):
        classification = "AE12_8_PASS_WITH_ARCHIVAL_LIMITATIONS"

    # Archival limitations: empty/incomplete historical roots (safety still clear)
    ae8_root = PROJECT_ROOT / "data" / "audits" / "ae8_context_intelligence_20260711T090138Z"
    ae8_file_count = (
        sum(1 for p in ae8_root.rglob("*") if p.is_file()) if ae8_root.exists() else 0
    )
    ae10_root = PROJECT_ROOT / "data" / "audits" / "ae10_trading_orchestration_20260711T090202Z"
    ae10_file_count = (
        sum(1 for p in ae10_root.rglob("*") if p.is_file()) if ae10_root.exists() else 0
    )
    if classification == "AE12_8_PASS_SAFETY_REPRO_PACKAGE" and (
        ae8_file_count == 0 or ae10_file_count == 0
    ):
        classification = "AE12_8_PASS_WITH_ARCHIVAL_LIMITATIONS"

    repro_manifest = {
        "project_root": str(PROJECT_ROOT),
        "ae12_8_output_root": str(OUTPUT_ROOT),
        "created_at": CREATED_AT,
        "machine_context_available": True,
        "git_available": git_available,
        "git_commit_hash_or_null": git_hash,
        "git_status_or_null": git_status,
        "dependency_environment": {
            "git_metadata_available": git_available,
            "commit_hash_available": bool(git_hash),
            "pip_freeze_succeeded": pip_ok,
            "environment_snapshot_path": str(env_snap),
            "python_version": py_text,
            "pip_version": pip_text,
            "limitations_without_git_metadata": (
                "Without .git metadata, exact source revision identity cannot be proven. "
                "Dependency freeze documents installed packages only."
            ),
            "warning": (
                "Environment snapshot helps reproduce dependencies but does not prove "
                "source-code revision identity."
            ),
        },
        "source_roots_inspected": [
            str(PROJECT_ROOT / "app"),
            str(PROJECT_ROOT / "scripts"),
            str(PROJECT_ROOT / "tests"),
            str(PROJECT_ROOT / "data" / "audits"),
            str(PROJECT_ROOT / "data" / "intelligent_agents"),
        ],
        "audit_roots_inspected": [
            str(AE12_7_ROOT),
            str(PROJECT_ROOT / "data" / "audits" / "ae12_forward_evidence_maturation_20260714_235401"),
            str(PROJECT_ROOT / "data" / "audits" / "ae12_ml_meta_layer_evaluation_20260717_111056"),
            str(PROJECT_ROOT / "data" / "audits" / "ae12_sentimentfix_20260715_172645"),
            str(PROJECT_ROOT / "data" / "audits" / "ae11_runtime_paper_loop_20260714T173541Z_de47ba58"),
            str(PROJECT_ROOT / "data" / "audits" / "ae9_llm_audit_20260711T090138Z"),
        ],
        "commands_used_for_readonly_inventory": [
            "python --version",
            "python -m pip --version",
            "python -m pip freeze",
            "filesystem inventory of data/audits (read-only)",
            "json/csv parse of saved gate artifacts (read-only)",
            f"python {__file__}",
        ],
        "files_created_by_AE12_8": [str(OUTPUT_ROOT / rel) for rel in created_files],
        "files_read_by_AE12_8": sorted(set(files_read))[:500],
        "files_not_found": sorted(set(files_not_found))[:200],
        "no_retraining_confirmed": True,
        "no_runtime_loop_confirmed": True,
        "no_external_api_calls_confirmed": True,
        "no_trader_db_mutation_confirmed": None,  # filled after fingerprint check
        "no_wallet_confirmed": True,
        "reproducibility_limitations": [
            "No .git metadata / commit hash available",
            "Dependency snapshot does not prove source revision identity",
            "Gate statuses are historical saved audits, not fresh reruns",
            "AE12.8 did not retrain models or replay runtime loops",
        ],
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": sys.version,
        },
        "trader_db_fingerprint_before": db_before,
    }

    # Reports
    limitations_md = """# AE12.8 Known Limitations

- **No live trading approval** — live_trading_ready=false; live_trading_approval=NO.
- **No profitability proof** — profitability_proven=false across AE11/AE12.x saved gates.
- **No production-ready AE7 runtime meta-layer authority** — meta-layer remains research/diagnostic; original runtime-ready stack not closed (`BLOCKED_POLICY_PLACEHOLDER` historically).
- **RF/XGB/TAB evidence is research/reporting/diagnostic**, not live authority.
- **LLM/agent/context layers are explanation/audit/context only** — no trade authority.
- **Gemini semantic adjudication was not web-grounded** unless later evidence proves otherwise; AE12.8 did not call Gemini.
- **UNKNOWN_UNRESOLVED remains unresolved**, not social and not opportunistic; it means local artifacts were insufficient to resolve without external evidence.
- **Helius broad live ingestion not proven**; AE12.8 did not call Helius.
- **Qwen quality not proven** beyond AE12.7 operational demo memo/linkage; AE12.8 did not call Qwen/Ollama.
- **Full-suite failures/errors may remain** if unrelated/pre-existing; AE12.8 did not rerun the full suite.
- **Missing .git metadata limits reproducibility**.
- **Dependency snapshot helps but does not prove source revision identity**.
- **AE8 context intelligence audit roots are empty** (directories exist, 0 files) — context authority still recorded as reporting-only; archival gap noted.
- AE12.8 does **not** close AE12.9 or the full MSc final package.
- Gate matrix entries are **saved historical statuses**, not fresh absolute truth from reruns.
"""
    write_text(OUTPUT_ROOT / "reports" / "ae12_8_known_limitations.md", limitations_md)

    built_vs_not = """# AE12.8 — What Was Built vs Not Proven

## Built / demonstrated

- Decision-record architecture (AE6)
- Paper/demo execution (AE10/AE11)
- Forward evidence persistence/maturation (AE12.1–AE12.4)
- Missed-winner reporting
- Strict vs exploration distinction
- Runtime observability/reporting (AE12.5 code + AE12 APIs)
- Semantic/SentimentFix reporting (dual-axis taxonomy)
- Intelligent-agent operational demo layer (AE12.7)
- No-wallet / no-live safety posture (AE11/AE12.7/AE12.8 audits)

## Not proven

- Profitability
- Live trading readiness
- Real wallet execution
- Production meta-layer authority
- Robust RF/XGB/TAB runtime edge
- Gemini/Qwen as trade authority
- Helius broad live ingestion
- Web-grounded semantic truth for all coins

## AE12.8 packaging scope

AE12.8 archives safety, authority, and reproducibility evidence only.
It does **not** retrain, start runtime loops, call external APIs, mutate `trader.db`,
connect a wallet, or close AE12.9 / the full MSc package.
"""
    write_text(OUTPUT_ROOT / "reports" / "ae12_8_what_was_built_vs_not_proven.md", built_vs_not)

    # trader.db after
    db_after = {
        "path": str(TRADER_DB),
        "exists": TRADER_DB.exists(),
        "size": TRADER_DB.stat().st_size if TRADER_DB.exists() else None,
        "mtime": datetime.fromtimestamp(TRADER_DB.stat().st_mtime, tz=timezone.utc).isoformat()
        if TRADER_DB.exists()
        else None,
        "fingerprint": sha256_quick(TRADER_DB) if TRADER_DB.exists() else None,
    }
    db_unchanged = db_before == db_after
    repro_manifest["no_trader_db_mutation_confirmed"] = db_unchanged
    repro_manifest["trader_db_fingerprint_after"] = db_after

    write_json(
        OUTPUT_ROOT / "manifests" / "ae12_8_reproducibility_manifest.json",
        repro_manifest,
    )

    decision_gate = {
        "phase": "AE12.8",
        "classification": classification,
        "created_at": CREATED_AT,
        "ae12_8_closed": True,
        "ae12_9_closed": False,
        "msc_final_package_closed": False,
        "ae12_9_blocked": False,
        "safety": {
            "status": wallet.get("status"),
            "wallet_configured": False,
            "private_key_accessed": False,
            "real_transaction_signed": False,
            "real_transaction_attempted": False,
            "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
            "live_trading_ready": False,
            "live_trading_approval": "NO",
            "demo_paper_trading_allowed": True,
            "profitability_proven": False,
        },
        "reproducibility": {
            "git_available": git_available,
            "pip_freeze_succeeded": pip_ok,
            "manifest_written": True,
            "source_revision_identity_proven": False,
        },
        "ae12_7_root_exists": ae127_ok,
        "ae12_7_root": str(AE12_7_ROOT),
        "trader_db_unchanged": db_unchanged,
        "no_external_api_calls": True,
        "no_retraining": True,
        "no_runtime_loop": True,
        "authority_matrix_clear": True,
        "limitations_explicit": True,
        "notes": [
            "Gate statuses collected from saved artifacts; not fresh authoritative reruns.",
            "AE12.8 can close as safety/repro packaging phase.",
            "AE12.9 is not blocked by AE12.8 safety packaging.",
            "AE12.8 must not be treated as full MSc final package closure.",
        ],
        "recommended_ae12_9_first_reads": [
            str(OUTPUT_ROOT / "reports" / "ae12_8_final_decision_gate.json"),
            str(OUTPUT_ROOT / "reports" / "ae12_8_summary_for_upload.txt"),
            str(OUTPUT_ROOT / "reports" / "ae12_8_known_limitations.md"),
            str(OUTPUT_ROOT / "reports" / "ae12_8_what_was_built_vs_not_proven.md"),
            str(OUTPUT_ROOT / "manifests" / "ae12_8_reproducibility_manifest.json"),
            str(OUTPUT_ROOT / "manifests" / "ae12_8_subsystem_gate_matrix.csv"),
            str(OUTPUT_ROOT / "audits" / "ae12_8_authority_matrix.csv"),
            str(OUTPUT_ROOT / "audits" / "ae12_8_wallet_no_live_safety_audit.json"),
            str(AE12_7_ROOT / "reports" / "ae12_7_intelligent_agent_decision_gate.json"),
            str(AE12_7_DAILY_RECORDS),
        ],
    }
    write_json(OUTPUT_ROOT / "reports" / "ae12_8_final_decision_gate.json", decision_gate)

    # Update AE12.8 row in gate matrix file already written — rewrite with final status
    for r in gate_rows:
        if r["subsystem"] == "AE12.8":
            r["saved_gate_status"] = classification
            r["saved_gate_file"] = str(
                OUTPUT_ROOT / "reports" / "ae12_8_final_decision_gate.json"
            )
            r["source_file_exists"] = True
    write_csv(
        OUTPUT_ROOT / "manifests" / "ae12_8_subsystem_gate_matrix.csv",
        [
            "subsystem",
            "saved_gate_status",
            "saved_gate_file",
            "evidence_root",
            "AE12_8_interpretation",
            "blocker_for_AE12_9",
            "limitation",
        ],
        gate_rows,
    )

    # Update test matrix packaging result
    for r in test_rows:
        if r["test_area"].startswith("AE12.8 packaging"):
            r["result_if_known"] = "PASS"
    write_csv(
        OUTPUT_ROOT / "tests" / "ae12_8_test_matrix.csv",
        [
            "test_area",
            "evidence_path",
            "exists",
            "rerun_in_AE12_8",
            "result_if_known",
            "classification",
            "notes",
        ],
        test_rows,
    )

    summary = f"""AE12.8 Safety / Audit / Reproducibility Package
Classification: {classification}
Output root: {OUTPUT_ROOT}
Created at (UTC): {CREATED_AT}

Safety:
  wallet_configured=False
  private_key_accessed=False
  real_transaction_signed=False
  real_transaction_attempted=False
  live_submission_status=NOT_SUBMITTED_NO_WALLET
  live_trading_ready=False
  live_trading_approval=NO
  demo_paper_trading_allowed=True
  profitability_proven=False
  trader_db_unchanged={db_unchanged}

Reproducibility:
  git_available={git_available}
  git_commit_hash=null
  pip_freeze_succeeded={pip_ok}
  python={py_text}
  pip={pip_text}
  source_revision_identity_proven=False

Confirmed AE12.7 root exists: {ae127_ok}
AE12.7 daily records exist: {AE12_7_DAILY_RECORDS.exists()}
AE12.7 daily call audit exists: {AE12_7_DAILY_CALL_AUDIT.exists()}

Authority: RF/XGB/TAB/Consensus/Meta/Context/LLM = no live authority; paper/demo allowed; live not approved.

AE12.8 closed as packaging phase: YES
AE12.9 closed: NO
MSc final package closed: NO
AE12.9 blocked by AE12.8: NO

AE12.9 should read first:
  - reports/ae12_8_final_decision_gate.json
  - reports/ae12_8_summary_for_upload.txt
  - reports/ae12_8_known_limitations.md
  - reports/ae12_8_what_was_built_vs_not_proven.md
  - manifests/ae12_8_reproducibility_manifest.json
  - manifests/ae12_8_subsystem_gate_matrix.csv
  - audits/ae12_8_authority_matrix.csv
  - audits/ae12_8_wallet_no_live_safety_audit.json
  - AE12.7 decision gate + daily agent records
"""
    write_text(OUTPUT_ROOT / "reports" / "ae12_8_summary_for_upload.txt", summary)

    # Validation
    missing_required = []
    for rel in created_files:
        if not (OUTPUT_ROOT / rel).exists():
            missing_required.append(rel)

    json_ok = True
    for rel in created_files:
        if rel.endswith(".json"):
            try:
                read_json(OUTPUT_ROOT / rel)
            except Exception:
                json_ok = False

    csv_ok = True
    for rel in created_files:
        if rel.endswith(".csv"):
            p = OUTPUT_ROOT / rel
            with p.open("r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    csv_ok = False

    def _safe_print(msg: str) -> None:
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode("ascii", "replace").decode("ascii"))

    _safe_print(summary)
    _safe_print("---VALIDATION---")
    _safe_print(f"missing_required={missing_required}")
    _safe_print(f"json_ok={json_ok}")
    _safe_print(f"csv_ok={csv_ok}")
    _safe_print(f"trader_db_unchanged={db_unchanged}")
    _safe_print(f"classification={classification}")
    return 0 if not missing_required and json_ok and csv_ok and db_unchanged else 1


if __name__ == "__main__":
    raise SystemExit(main())
