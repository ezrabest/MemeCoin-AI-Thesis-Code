#!/usr/bin/env python
"""
E9H - Final E9 Decision Gate

Purpose:
  Close Phase E9 by reading E9A-E9G artifacts and producing a final main-thread
  decision package.

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

This is a final decision/reporting script only.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_E9A_ROOT = ROOT / "data" / "training" / "manual_verified_results" / "phase_e9a_matched_control_contract_20260708_202222"
DEFAULT_E9B_ROOT = ROOT / "data" / "training" / "manual_verified_results" / "phase_e9b_context_discrimination_20260709_081445"
DEFAULT_E9C_ROOT = ROOT / "data" / "training" / "manual_verified_results" / "phase_e9c_whale_score_contract_20260709_082522"
DEFAULT_E9D_ROOT = ROOT / "data" / "training" / "manual_verified_results" / "phase_e9d_whale_score_rule_prototype_20260709_082958"
DEFAULT_E9F_ROOT = ROOT / "data" / "training" / "manual_verified_results" / "phase_e9f_forward_context_collection_design_20260709_083922"
DEFAULT_E9G_ROOT = ROOT / "data" / "training" / "manual_verified_results" / "phase_e9g_llm_reasoning_design_20260709_084233"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_read_error": str(exc), "_path": str(path)}


def read_text_optional(path: Path) -> str:
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def bool_from_gate(gate: dict[str, Any], key: str) -> bool:
    return bool(gate.get(key, False))


def build_phase_inventory(args: argparse.Namespace) -> pd.DataFrame:
    rows = [
        {
            "phase": "E9A",
            "root": str(args.e9a_root),
            "decision_file": str(args.e9a_root / "audits" / "e9a_decision_gate.json"),
            "summary_file": str(args.e9a_root / "reports" / "e9a_summary_for_upload.txt"),
            "role": "row-level matched-control contract",
        },
        {
            "phase": "E9B",
            "root": str(args.e9b_root),
            "decision_file": str(args.e9b_root / "audits" / "e9b_decision_gate.json"),
            "summary_file": str(args.e9b_root / "reports" / "e9b_summary_for_upload.txt"),
            "role": "context feature discrimination audit",
        },
        {
            "phase": "E9C",
            "root": str(args.e9c_root),
            "decision_file": str(args.e9c_root / "audits" / "e9c_decision_gate.json"),
            "summary_file": str(args.e9c_root / "reports" / "e9c_summary_for_upload.txt"),
            "role": "whale_score_asof feature contract",
        },
        {
            "phase": "E9D",
            "root": str(args.e9d_root),
            "decision_file": str(args.e9d_root / "audits" / "e9d_decision_gate.json"),
            "summary_file": str(args.e9d_root / "reports" / "e9d_summary_for_upload.txt"),
            "role": "non-training whale-score rule prototype",
        },
        {
            "phase": "E9F",
            "root": str(args.e9f_root),
            "decision_file": str(args.e9f_root / "audits" / "e9f_no_runtime_decision_gate.json"),
            "summary_file": str(args.e9f_root / "reports" / "e9f_summary_for_upload.txt"),
            "role": "forward context collection design",
        },
        {
            "phase": "E9G",
            "root": str(args.e9g_root),
            "decision_file": str(args.e9g_root / "audits" / "e9g_no_llm_execution_confirmation.json"),
            "summary_file": str(args.e9g_root / "reports" / "e9g_summary_for_upload.txt"),
            "role": "Qwen/Gemini reasoning design only",
        },
    ]

    for r in rows:
        r["root_exists"] = Path(r["root"]).exists()
        r["decision_file_exists"] = Path(r["decision_file"]).exists()
        r["summary_file_exists"] = Path(r["summary_file"]).exists()

    return pd.DataFrame(rows)


def derive_final_decision(
    e9a: dict[str, Any],
    e9b: dict[str, Any],
    e9c: dict[str, Any],
    e9d: dict[str, Any],
    e9f: dict[str, Any],
    e9g: dict[str, Any],
) -> dict[str, Any]:
    blockers = []
    approvals = []
    warnings = []

    e9a_status = e9a.get("final_e9a_status") or e9a.get("decision")
    e9b_status = e9b.get("decision")
    e9c_status = e9c.get("decision")
    e9d_status = e9d.get("decision")
    e9f_status = e9f.get("decision")
    e9g_status = e9g.get("decision")

    if e9a_status != "E9A_PASS_WEAK_CONTROL_CONTRACT":
        warnings.append(f"E9A_UNEXPECTED_STATUS_{e9a_status}")
    else:
        blockers.append("E9A control contract is weak, not strong.")

    if e9b_status != "E9B_RESEARCH_ONLY_FEATURE_CANDIDATES":
        warnings.append(f"E9B_UNEXPECTED_STATUS_{e9b_status}")
    else:
        blockers.append("E9B feature candidates are research-only.")

    if e9c_status != "E9C_RESEARCH_ONLY_WHALE_SCORE_CONTRACT_CANDIDATE":
        warnings.append(f"E9C_UNEXPECTED_STATUS_{e9c_status}")
    else:
        blockers.append("E9C whale_score_asof contract is research-only.")

    if e9d_status != "E9D_RESEARCH_ONLY_RULE_NOT_ROBUST_ENOUGH":
        warnings.append(f"E9D_UNEXPECTED_STATUS_{e9d_status}")
    else:
        blockers.append("E9D whale-score rule failed robustness and leave-one-pair-out.")

    if e9f_status == "E9F_FORWARD_COLLECTION_DESIGN_ONLY_NO_RUNTIME":
        approvals.append("Forward context collection design is approved.")
    else:
        warnings.append(f"E9F_UNEXPECTED_STATUS_{e9f_status}")

    if e9g_status == "E9G_LLM_REASONING_DESIGN_ONLY_NO_EXECUTION":
        approvals.append("LLM reasoning/audit design is approved, with no execution.")
    else:
        warnings.append(f"E9G_UNEXPECTED_STATUS_{e9g_status}")

    final_decision = "E9_RESEARCH_ONLY_FORWARD_COLLECTION_DESIGN"
    if not e9f.get("approved_for_forward_collection_design", False):
        final_decision = "E9_RESEARCH_ONLY_NO_FORWARD_COLLECTION_APPROVAL"
        blockers.append("E9F did not approve forward collection design.")

    return {
        "decision": final_decision,
        "created_at_utc": utc_now_iso(),
        "phase_e9_closed": True,
        "approved_for_forward_context_collection_design": bool(e9f.get("approved_for_forward_collection_design", False)),
        "approved_for_actual_collection_code_changes": False,
        "approved_for_context_modeling": False,
        "approved_for_model_training": False,
        "approved_for_rf_tab_xgb_training": False,
        "approved_for_runtime": False,
        "approved_for_ui": False,
        "approved_for_trading": False,
        "approved_for_reservoir_scoring": False,
        "approved_for_llm_execution": False,
        "approved_for_qwen_execution": False,
        "approved_for_gemini_execution": False,
        "approved_for_ollama_execution": False,
        "recommended_next_phase": "Main-thread decision required: either stop at research-only archive, or open a new forward-context collection implementation phase with explicit approval.",
        "source_decisions": {
            "e9a": e9a_status,
            "e9b": e9b_status,
            "e9c": e9c_status,
            "e9d": e9d_status,
            "e9f": e9f_status,
            "e9g": e9g_status,
        },
        "key_findings": [
            "Row-level matched-control contract was built, but it is weak.",
            "E9B found whale_score_asof as a research-only candidate feature.",
            "E9C confirmed whale_score_asof can proceed only as offline research.",
            "E9D showed the whale-score rule is not robust enough and fails leave-one-pair-out.",
            "The evidence remains concentrated in too few positive pairs.",
            "E9F produced a forward context collection design only.",
            "E9G produced Qwen/Gemini reasoning design only, with no LLM execution.",
        ],
        "blockers_to_modeling_or_runtime": blockers,
        "approved_outputs": approvals,
        "warnings": warnings,
        "non_negotiable_closure_rules": [
            "Do not open E9E modeling from this E9 evidence.",
            "Do not train RF/TAB/XGB from this E9 evidence.",
            "Do not connect context features to runtime.",
            "Do not run reservoir scoring.",
            "Do not use Qwen/Gemini/Ollama in runtime from this phase.",
            "Do not use whale_score_asof as a trading rule.",
            "Do not treat E9D enrichment as a deployable signal.",
            "Future progress requires new forward collection over more unique positive pairs.",
        ],
    }


def build_final_report(
    final_gate: dict[str, Any],
    phase_inventory: pd.DataFrame,
    e9a: dict[str, Any],
    e9b: dict[str, Any],
    e9c: dict[str, Any],
    e9d: dict[str, Any],
    e9f: dict[str, Any],
    e9g: dict[str, Any],
) -> str:
    lines: list[str] = []

    lines.extend([
        "Phase / branch name",
        "",
        "E9H — Final E9 Decision Gate",
        "",
        "Run status",
        "",
        "COMPLETED",
        "",
        "Final E9 decision",
        "",
        str(final_gate["decision"]),
        "",
        "Final approval matrix",
        "",
        f"- Phase E9 closed: {final_gate['phase_e9_closed']}",
        f"- approved_for_forward_context_collection_design: {final_gate['approved_for_forward_context_collection_design']}",
        f"- approved_for_actual_collection_code_changes: {final_gate['approved_for_actual_collection_code_changes']}",
        f"- approved_for_context_modeling: {final_gate['approved_for_context_modeling']}",
        f"- approved_for_model_training: {final_gate['approved_for_model_training']}",
        f"- approved_for_rf_tab_xgb_training: {final_gate['approved_for_rf_tab_xgb_training']}",
        f"- approved_for_runtime: {final_gate['approved_for_runtime']}",
        f"- approved_for_ui: {final_gate['approved_for_ui']}",
        f"- approved_for_trading: {final_gate['approved_for_trading']}",
        f"- approved_for_reservoir_scoring: {final_gate['approved_for_reservoir_scoring']}",
        f"- approved_for_llm_execution: {final_gate['approved_for_llm_execution']}",
        "",
        "Source decisions",
        "",
    ])

    for k, v in final_gate["source_decisions"].items():
        lines.append(f"- {k.upper()}: {v}")

    lines.extend([
        "",
        "Key findings",
        "",
    ])
    for item in final_gate["key_findings"]:
        lines.append(f"- {item}")

    lines.extend([
        "",
        "Detailed interpretation",
        "",
        "E9 successfully tested whether context features can help explain or discriminate rare winners after E8 remained research-only.",
        "The phase found one interesting feature candidate, whale_score_asof, but it did not survive robustness strongly enough to justify modeling or runtime.",
        "",
        "The most important E9D result is that the selected whale-score rule captured only one unique selected positive pair and failed leave-one-pair-out.",
        "Therefore the correct interpretation is not 'context signal approved', but 'forward context collection needed'.",
        "",
        "Forward collection design",
        "",
        "E9F approved only design for future forward context collection.",
        "It did not approve implementation, runtime writes, modeling, or trading.",
        "",
        "LLM design",
        "",
        "E9G approved only design documentation for future Qwen/Gemini roles.",
        "Qwen may later become a local explanation/memo layer only.",
        "Gemini may later become a selective offline auditor only.",
        "Neither is approved for execution now.",
        "",
        "Blockers to modeling/runtime",
        "",
    ])

    for b in final_gate["blockers_to_modeling_or_runtime"]:
        lines.append(f"- {b}")

    lines.extend([
        "",
        "Approved outputs",
        "",
    ])

    if final_gate["approved_outputs"]:
        for a in final_gate["approved_outputs"]:
            lines.append(f"- {a}")
    else:
        lines.append("None.")

    lines.extend([
        "",
        "Non-negotiable closure rules",
        "",
    ])

    for r in final_gate["non_negotiable_closure_rules"]:
        lines.append(f"- {r}")

    lines.extend([
        "",
        "Artifact inventory",
        "",
        "| phase | role | root | decision_file_exists | summary_file_exists |",
        "|---|---|---|---:|---:|",
    ])

    for _, r in phase_inventory.iterrows():
        lines.append(
            f"| {r['phase']} | {r['role']} | `{r['root']}` | {r['decision_file_exists']} | {r['summary_file_exists']} |"
        )

    lines.extend([
        "",
        "Recommended next action",
        "",
        final_gate["recommended_next_phase"],
        "",
        "Final management note",
        "",
        "Phase E9 answered its intended question. It refined the Anchor Plan but does not approve runtime, model training, or context-enhanced deployment.",
        "The system should not proceed to runtime or E9E modeling from this evidence.",
        "The only clean continuation is a separately approved forward-context collection implementation phase, if the main thread wants to collect more evidence.",
        "",
    ])

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="E9H Final E9 Decision Gate")
    parser.add_argument("--e9a-root", type=Path, default=DEFAULT_E9A_ROOT)
    parser.add_argument("--e9b-root", type=Path, default=DEFAULT_E9B_ROOT)
    parser.add_argument("--e9c-root", type=Path, default=DEFAULT_E9C_ROOT)
    parser.add_argument("--e9d-root", type=Path, default=DEFAULT_E9D_ROOT)
    parser.add_argument("--e9f-root", type=Path, default=DEFAULT_E9F_ROOT)
    parser.add_argument("--e9g-root", type=Path, default=DEFAULT_E9G_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    output_root = args.output_root or (
        ROOT / "data" / "training" / "manual_verified_results" / f"phase_e9h_final_decision_gate_{timestamp_slug()}"
    )

    reports_dir = output_root / "reports"
    data_dir = output_root / "data"
    audits_dir = output_root / "audits"

    for d in [reports_dir, data_dir, audits_dir]:
        ensure_dir(d)

    e9a_decision = read_json_optional(args.e9a_root / "audits" / "e9a_decision_gate.json")
    e9b_decision = read_json_optional(args.e9b_root / "audits" / "e9b_decision_gate.json")
    e9c_decision = read_json_optional(args.e9c_root / "audits" / "e9c_decision_gate.json")
    e9d_decision = read_json_optional(args.e9d_root / "audits" / "e9d_decision_gate.json")
    e9f_gate = read_json_optional(args.e9f_root / "audits" / "e9f_no_runtime_decision_gate.json")
    e9g_gate = read_json_optional(args.e9g_root / "audits" / "e9g_no_llm_execution_confirmation.json")

    phase_inventory = build_phase_inventory(args)
    write_csv(data_dir / "e9h_phase_artifact_inventory.csv", phase_inventory)

    final_gate = derive_final_decision(
        e9a=e9a_decision,
        e9b=e9b_decision,
        e9c=e9c_decision,
        e9d=e9d_decision,
        e9f=e9f_gate,
        e9g=e9g_gate,
    )
    write_json(audits_dir / "e9h_final_decision_gate.json", final_gate)

    final_report = build_final_report(
        final_gate=final_gate,
        phase_inventory=phase_inventory,
        e9a=e9a_decision,
        e9b=e9b_decision,
        e9c=e9c_decision,
        e9d=e9d_decision,
        e9f=e9f_gate,
        e9g=e9g_gate,
    )
    write_text(reports_dir / "e9h_summary_for_upload.txt", final_report)

    manifest = {
        "phase": "E9H",
        "branch_name": "phase_e9h_final_e9_decision_gate",
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
            "e9d_root": str(args.e9d_root),
            "e9f_root": str(args.e9f_root),
            "e9g_root": str(args.e9g_root),
        },
        "outputs": {
            "output_root": str(output_root),
            "manifest": str(reports_dir / "e9h_manifest.json"),
            "summary": str(reports_dir / "e9h_summary_for_upload.txt"),
            "phase_artifact_inventory": str(data_dir / "e9h_phase_artifact_inventory.csv"),
            "final_decision_gate": str(audits_dir / "e9h_final_decision_gate.json"),
        },
        "decision": final_gate["decision"],
    }
    write_json(reports_dir / "e9h_manifest.json", manifest)

    print(json.dumps({
        "status": "completed",
        "phase": "E9H",
        "output_root": str(output_root),
        "decision": final_gate["decision"],
        "phase_e9_closed": final_gate["phase_e9_closed"],
        "approved_for_forward_context_collection_design": final_gate["approved_for_forward_context_collection_design"],
        "approved_for_actual_collection_code_changes": final_gate["approved_for_actual_collection_code_changes"],
        "approved_for_context_modeling": final_gate["approved_for_context_modeling"],
        "approved_for_model_training": final_gate["approved_for_model_training"],
        "approved_for_runtime": final_gate["approved_for_runtime"],
        "approved_for_llm_execution": final_gate["approved_for_llm_execution"],
        "summary": str(reports_dir / "e9h_summary_for_upload.txt"),
        "decision_gate": str(audits_dir / "e9h_final_decision_gate.json"),
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
