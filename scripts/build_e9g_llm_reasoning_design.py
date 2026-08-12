#!/usr/bin/env python
"""
E9G - Qwen/Gemini Reasoning Design Only

Purpose:
  Create design-only contracts for future Qwen/Gemini usage after E9F.
  No LLM execution is performed.

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
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_E9A_ROOT = ROOT / "data" / "training" / "manual_verified_results" / "phase_e9a_matched_control_contract_20260708_202222"
DEFAULT_E9B_ROOT = ROOT / "data" / "training" / "manual_verified_results" / "phase_e9b_context_discrimination_20260709_081445"
DEFAULT_E9C_ROOT = ROOT / "data" / "training" / "manual_verified_results" / "phase_e9c_whale_score_contract_20260709_082522"
DEFAULT_E9D_ROOT = ROOT / "data" / "training" / "manual_verified_results" / "phase_e9d_whale_score_rule_prototype_20260709_082958"
DEFAULT_E9F_ROOT = ROOT / "data" / "training" / "manual_verified_results" / "phase_e9f_forward_context_collection_design_20260709_083922"


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


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def build_qwen_contract() -> str:
    return "\n".join([
        "# E9G - Qwen Candidate Memo Contract",
        "",
        "## Status",
        "",
        "Design-only. No Qwen/Ollama execution is approved in E9G.",
        "",
        "## Intended future role",
        "",
        "Qwen may be used in a future approved phase as a local reasoning and explanation layer only.",
        "",
        "Allowed future roles:",
        "",
        "- summarize candidate context",
        "- explain why a candidate is interesting or risky",
        "- describe market/liquidity/activity context",
        "- summarize forward-collected context features",
        "- generate an audit memo for human review",
        "- flag soft concerns such as missing context, inconsistent features, or suspected hype/scam patterns",
        "",
        "Forbidden roles:",
        "",
        "- no BUY decision authority",
        "- no SELL decision authority",
        "- no position sizing authority",
        "- no direct trading execution",
        "- no override of quantitative gates",
        "- no use as a replacement for RF/TAB/XGB",
        "- no runtime execution unless a later decision gate explicitly approves it",
        "",
        "## Required Qwen input object",
        "",
        "A future Qwen memo must receive only a bounded candidate packet:",
        "",
        "- candidate_id",
        "- candidate_policy_id",
        "- target_row_id if available",
        "- pair_address",
        "- event_timestamp",
        "- model tier/rank metadata if already approved",
        "- context features collected as-of entry time",
        "- missingness flags",
        "- source lineage",
        "- explicit instruction that this is not a trading command",
        "",
        "## Required Qwen output object",
        "",
        "Qwen output must be structured:",
        "",
        "- memo_id",
        "- candidate_id",
        "- context_summary",
        "- risk_summary",
        "- supporting_factors",
        "- opposing_factors",
        "- missing_data_warnings",
        "- soft_veto_flags",
        "- reasoning_confidence",
        "- no_trade_authority_confirmation",
        "",
        "## Safety rule",
        "",
        "Every Qwen memo must include: `This memo is explanatory only and is not an execution decision.`",
        "",
    ])


def build_gemini_contract() -> str:
    return "\n".join([
        "# E9G - Gemini Selective Audit Contract",
        "",
        "## Status",
        "",
        "Design-only. No Gemini call is approved in E9G.",
        "",
        "## Intended future role",
        "",
        "Gemini may be used only as a selective offline auditor in a future approved phase.",
        "",
        "Allowed future roles:",
        "",
        "- review a small number of high-risk or ambiguous candidate memos",
        "- challenge Qwen explanations",
        "- inspect false-positive risk",
        "- provide scam/reputation reasoning if sufficient context is supplied",
        "- generate an independent audit note for human review",
        "",
        "Forbidden roles:",
        "",
        "- no default runtime calls",
        "- no continuous scan-level execution",
        "- no BUY/SELL decision authority",
        "- no execution approval",
        "- no replacement of quantitative gates",
        "- no hidden internet/API expansion unless explicitly approved",
        "",
        "## Trigger policy for future approved use",
        "",
        "Gemini should only be considered when one of these happens:",
        "",
        "- candidate is unusually high-ranked but context is contradictory",
        "- Qwen flags soft-veto uncertainty",
        "- missingness pattern is suspicious",
        "- candidate is near a future paper/demo threshold",
        "- manual reviewer requests audit",
        "",
        "## Output requirements",
        "",
        "- gemini_audit_id",
        "- candidate_id",
        "- audit_reason",
        "- agreement_or_disagreement_with_qwen",
        "- key_risks",
        "- key_supporting_evidence",
        "- missing_information",
        "- no_trade_authority_confirmation",
        "",
    ])


def build_llm_safety_rules() -> str:
    return "\n".join([
        "# E9G - LLM Safety Rules",
        "",
        "## Global rules",
        "",
        "- LLMs are not quantitative models.",
        "- LLMs do not replace RF/TAB/XGB.",
        "- LLMs do not approve runtime.",
        "- LLMs do not approve trading.",
        "- LLMs must be fail-closed.",
        "- If LLM output is missing, invalid, late, or malformed, the system must continue without LLM approval.",
        "- No LLM output may change risk settings.",
        "- No LLM output may write to trading state.",
        "- No LLM output may execute a trade.",
        "- No LLM output may be treated as a label or target.",
        "",
        "## Runtime protection",
        "",
        "Future runtime use, if ever approved, must be short-circuited:",
        "",
        "- do not call LLM for every candidate",
        "- call only after strict quantitative/context gates",
        "- enforce max calls per scan",
        "- enforce timeout",
        "- enforce structured JSON schema",
        "- log every call and every skipped call",
        "- default to HOLD / no action on failure",
        "",
        "## Research protection",
        "",
        "- LLM memos are research artifacts.",
        "- LLM memos must not be used as direct training labels.",
        "- LLM memos can be audited for explanatory quality only.",
        "- LLM calls must be reproducibly linked to candidate_id and source artifacts.",
        "",
    ])


def build_no_execution_gate(
    e9a_decision: dict[str, Any],
    e9b_decision: dict[str, Any],
    e9c_decision: dict[str, Any],
    e9d_decision: dict[str, Any],
    e9f_gate: dict[str, Any],
) -> dict[str, Any]:
    return {
        "decision": "E9G_LLM_REASONING_DESIGN_ONLY_NO_EXECUTION",
        "created_at_utc": utc_now_iso(),
        "approved_for_llm_design": True,
        "approved_for_qwen_execution": False,
        "approved_for_gemini_execution": False,
        "approved_for_ollama_execution": False,
        "approved_for_modeling": False,
        "approved_for_training": False,
        "approved_for_runtime": False,
        "approved_for_ui": False,
        "approved_for_trading": False,
        "approved_for_reservoir_scoring": False,
        "reason": "E9G defines future LLM reasoning/audit contracts only. E9 evidence remains research-only and does not approve LLM execution or runtime integration.",
        "source_decisions": {
            "e9a": e9a_decision.get("final_e9a_status") or e9a_decision.get("decision"),
            "e9b": e9b_decision.get("decision"),
            "e9c": e9c_decision.get("decision"),
            "e9d": e9d_decision.get("decision"),
            "e9f": e9f_gate.get("decision"),
        },
        "non_negotiable_rules": [
            "Qwen is explanation/audit only, not BUY/SELL authority.",
            "Gemini is selective offline auditor only, not runtime default.",
            "No LLM calls are approved by E9G.",
            "No LLM output may approve trading.",
            "No LLM output may override quantitative gates.",
            "No LLM output may be used as target/label.",
            "Any future LLM use requires a separate implementation decision gate.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="E9G Qwen/Gemini Reasoning Design Only")
    parser.add_argument("--e9a-root", type=Path, default=DEFAULT_E9A_ROOT)
    parser.add_argument("--e9b-root", type=Path, default=DEFAULT_E9B_ROOT)
    parser.add_argument("--e9c-root", type=Path, default=DEFAULT_E9C_ROOT)
    parser.add_argument("--e9d-root", type=Path, default=DEFAULT_E9D_ROOT)
    parser.add_argument("--e9f-root", type=Path, default=DEFAULT_E9F_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    output_root = args.output_root or (
        ROOT / "data" / "training" / "manual_verified_results" / f"phase_e9g_llm_reasoning_design_{timestamp_slug()}"
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

    qwen_contract = build_qwen_contract()
    gemini_contract = build_gemini_contract()
    safety_rules = build_llm_safety_rules()
    no_execution_gate = build_no_execution_gate(
        e9a_decision=e9a_decision,
        e9b_decision=e9b_decision,
        e9c_decision=e9c_decision,
        e9d_decision=e9d_decision,
        e9f_gate=e9f_gate,
    )

    write_text(reports_dir / "e9g_qwen_candidate_memo_contract.md", qwen_contract)
    write_text(reports_dir / "e9g_gemini_selective_audit_contract.md", gemini_contract)
    write_text(reports_dir / "e9g_llm_safety_rules.md", safety_rules)
    write_json(audits_dir / "e9g_no_llm_execution_confirmation.json", no_execution_gate)

    llm_role_matrix = [
        {
            "actor": "Qwen/Ollama local",
            "future_role": "candidate memo and explanation",
            "default_runtime_role": "none",
            "allowed_now": False,
            "future_possible": True,
            "buy_sell_authority": False,
            "execution_authority": False,
            "requires_future_gate": True,
        },
        {
            "actor": "Gemini",
            "future_role": "selective offline audit",
            "default_runtime_role": "none",
            "allowed_now": False,
            "future_possible": True,
            "buy_sell_authority": False,
            "execution_authority": False,
            "requires_future_gate": True,
        },
    ]
    write_json(data_dir / "e9g_llm_role_matrix.json", llm_role_matrix)

    manifest = {
        "phase": "E9G",
        "branch_name": "phase_e9g_qwen_gemini_reasoning_design",
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
        },
        "outputs": {
            "output_root": str(output_root),
            "manifest": str(reports_dir / "e9g_manifest.json"),
            "summary": str(reports_dir / "e9g_summary_for_upload.txt"),
            "qwen_contract": str(reports_dir / "e9g_qwen_candidate_memo_contract.md"),
            "gemini_contract": str(reports_dir / "e9g_gemini_selective_audit_contract.md"),
            "llm_safety_rules": str(reports_dir / "e9g_llm_safety_rules.md"),
            "llm_role_matrix": str(data_dir / "e9g_llm_role_matrix.json"),
            "no_llm_execution_confirmation": str(audits_dir / "e9g_no_llm_execution_confirmation.json"),
        },
        "decision": no_execution_gate["decision"],
    }

    write_json(reports_dir / "e9g_manifest.json", manifest)

    summary = "\n".join([
        "Phase / branch name",
        "",
        "E9G - Qwen/Gemini Reasoning Design Only",
        "",
        "Run status",
        "",
        "COMPLETED",
        "",
        "Decision",
        "",
        no_execution_gate["decision"],
        "",
        "Scope",
        "",
        "Design-only LLM reasoning/audit contracts.",
        "No Qwen, Gemini, Ollama, external API, model training, runtime, UI, trading, or SQLite writes.",
        "",
        "Source decisions",
        "",
        f"- E9A: {no_execution_gate['source_decisions']['e9a']}",
        f"- E9B: {no_execution_gate['source_decisions']['e9b']}",
        f"- E9C: {no_execution_gate['source_decisions']['e9c']}",
        f"- E9D: {no_execution_gate['source_decisions']['e9d']}",
        f"- E9F: {no_execution_gate['source_decisions']['e9f']}",
        "",
        "Qwen future role",
        "",
        "- local candidate memo",
        "- context explanation",
        "- risk summary",
        "- missingness warnings",
        "- soft-veto explanation",
        "- no BUY/SELL authority",
        "",
        "Gemini future role",
        "",
        "- selective offline audit",
        "- false-positive review",
        "- Qwen challenge/review",
        "- scam/reputation reasoning if context exists",
        "- no default runtime calls",
        "- no BUY/SELL authority",
        "",
        "Approvals",
        "",
        f"- approved_for_llm_design: {no_execution_gate['approved_for_llm_design']}",
        f"- approved_for_qwen_execution: {no_execution_gate['approved_for_qwen_execution']}",
        f"- approved_for_gemini_execution: {no_execution_gate['approved_for_gemini_execution']}",
        f"- approved_for_ollama_execution: {no_execution_gate['approved_for_ollama_execution']}",
        f"- approved_for_modeling: {no_execution_gate['approved_for_modeling']}",
        f"- approved_for_training: {no_execution_gate['approved_for_training']}",
        f"- approved_for_runtime: {no_execution_gate['approved_for_runtime']}",
        f"- approved_for_ui: {no_execution_gate['approved_for_ui']}",
        f"- approved_for_trading: {no_execution_gate['approved_for_trading']}",
        "",
        "Final interpretation",
        "",
        "E9G approves only design documentation for future LLM reasoning/audit roles. It does not approve LLM execution, modeling, runtime, UI, or trading.",
    ])

    write_text(reports_dir / "e9g_summary_for_upload.txt", summary)

    print(json.dumps({
        "status": "completed",
        "phase": "E9G",
        "output_root": str(output_root),
        "decision": no_execution_gate["decision"],
        "approved_for_llm_design": no_execution_gate["approved_for_llm_design"],
        "approved_for_qwen_execution": no_execution_gate["approved_for_qwen_execution"],
        "approved_for_gemini_execution": no_execution_gate["approved_for_gemini_execution"],
        "approved_for_modeling": no_execution_gate["approved_for_modeling"],
        "approved_for_runtime": no_execution_gate["approved_for_runtime"],
        "summary": str(reports_dir / "e9g_summary_for_upload.txt"),
        "decision_gate": str(audits_dir / "e9g_no_llm_execution_confirmation.json"),
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
