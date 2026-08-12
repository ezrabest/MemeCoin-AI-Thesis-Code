#!/usr/bin/env python
"""
E9F - Forward Context Collection Design

Purpose:
  Create a forward-collection design package after E9D showed that whale_score_asof
  is research-interesting but not robust enough for modeling/runtime.

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

This script only reads E9A/E9B/E9C/E9D artifacts and writes planning artifacts.
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


def build_feature_plan(
    e9b_decision: dict[str, Any],
    e9c_decision: dict[str, Any],
    e9d_decision: dict[str, Any],
) -> pd.DataFrame:
    candidate_features = e9b_decision.get("candidate_features") or []
    e9b_candidates = {str(x.get("feature_name")) for x in candidate_features if x.get("feature_name")}

    rows: list[dict[str, Any]] = []

    def add(
        feature_name: str,
        category: str,
        priority: str,
        source_class: str,
        collect_forward: bool,
        current_evidence_status: str,
        reason: str,
        fields_needed: str,
        compute_cost: str = "LOW",
        external_api_required: bool = False,
        missing_policy: str = "log_missing_and_do_not_impute_for_primary_audit",
        leakage_rule: str = "must_be_computed_before_entry_decision",
        asof_rule: str = "feature_asof_time <= candidate_event_timestamp",
    ) -> None:
        rows.append({
            "feature_name": feature_name,
            "category": category,
            "priority": priority,
            "source_class": source_class,
            "collect_forward": collect_forward,
            "current_evidence_status": current_evidence_status,
            "reason": reason,
            "fields_needed": fields_needed,
            "compute_cost": compute_cost,
            "external_api_required": external_api_required,
            "missing_policy": missing_policy,
            "leakage_rule": leakage_rule,
            "asof_rule": asof_rule,
            "approved_for_modeling": False,
            "approved_for_runtime": False,
            "approved_for_trading": False,
        })

    whale_status = "E9B_CANDIDATE_E9C_CONTRACT_E9D_NOT_ROBUST"
    if "whale_score_asof" not in e9b_candidates:
        whale_status = "NOT_CONFIRMED_BY_E9B"

    add(
        feature_name="whale_score_asof",
        category="market_activity_context",
        priority="P0_REQUIRED_FORWARD_COLLECTION",
        source_class="existing_snapshot_or_engine_computable",
        collect_forward=True,
        current_evidence_status=whale_status,
        reason="E9B/E9C found research-only discrimination, but E9D showed the rule was pair-specific and did not survive leave-one-pair-out.",
        fields_needed="liquidity_usd_asof, volume_h24_asof, txns_h24_buys_asof, txns_h24_sells_asof, price_change_fields_asof",
    )

    add(
        feature_name="liquidity_usd_asof",
        category="market_liquidity",
        priority="P0_REQUIRED_FORWARD_COLLECTION",
        source_class="dexscreener_snapshot",
        collect_forward=True,
        current_evidence_status="required_base_context",
        reason="Needed to interpret whale_score_asof and control for liquidity differences.",
        fields_needed="pair_address, event_timestamp, liquidity.usd",
    )

    add(
        feature_name="volume_h24_asof",
        category="market_activity",
        priority="P0_REQUIRED_FORWARD_COLLECTION",
        source_class="dexscreener_snapshot",
        collect_forward=True,
        current_evidence_status="required_base_context",
        reason="Needed to distinguish low-whale-score quiet pairs from low-whale-score early activity candidates.",
        fields_needed="pair_address, event_timestamp, volume.h24",
    )

    add(
        feature_name="txns_h24_buys_asof",
        category="market_activity",
        priority="P0_REQUIRED_FORWARD_COLLECTION",
        source_class="dexscreener_snapshot",
        collect_forward=True,
        current_evidence_status="required_base_context",
        reason="Buy-side activity is needed to interpret whale_score_asof direction.",
        fields_needed="pair_address, event_timestamp, txns.h24.buys",
    )

    add(
        feature_name="txns_h24_sells_asof",
        category="market_activity",
        priority="P0_REQUIRED_FORWARD_COLLECTION",
        source_class="dexscreener_snapshot",
        collect_forward=True,
        current_evidence_status="required_base_context",
        reason="Sell-side activity is needed for buy/sell imbalance and whale-score interpretation.",
        fields_needed="pair_address, event_timestamp, txns.h24.sells",
    )

    add(
        feature_name="buy_sell_ratio_h24_asof",
        category="market_activity",
        priority="P1_DERIVED_FORWARD_COLLECTION",
        source_class="derived_from_snapshot",
        collect_forward=True,
        current_evidence_status="not_validated_yet",
        reason="Derived feature for activity imbalance; must be tested later against matched controls.",
        fields_needed="txns_h24_buys_asof, txns_h24_sells_asof",
    )

    add(
        feature_name="price_change_m5_asof",
        category="momentum",
        priority="P1_FORWARD_COLLECTION",
        source_class="dexscreener_snapshot_if_available",
        collect_forward=True,
        current_evidence_status="not_validated_yet",
        reason="Short-term momentum may explain lottery-ticket behavior and should be logged separately from whale_score_asof.",
        fields_needed="priceChange.m5",
    )

    add(
        feature_name="price_change_h1_asof",
        category="momentum",
        priority="P1_FORWARD_COLLECTION",
        source_class="dexscreener_snapshot_if_available",
        collect_forward=True,
        current_evidence_status="not_validated_yet",
        reason="Medium-term momentum is needed as a control/context covariate.",
        fields_needed="priceChange.h1",
    )

    add(
        feature_name="price_change_h24_asof",
        category="momentum",
        priority="P1_FORWARD_COLLECTION",
        source_class="dexscreener_snapshot_if_available",
        collect_forward=True,
        current_evidence_status="not_validated_yet",
        reason="Longer momentum context is needed to avoid interpreting whale_score_asof in isolation.",
        fields_needed="priceChange.h24",
    )

    add(
        feature_name="pair_age_minutes_asof",
        category="token_age",
        priority="P1_FORWARD_COLLECTION",
        source_class="dexscreener_pair_created_at_if_available",
        collect_forward=True,
        current_evidence_status="not_validated_yet",
        reason="Rare winners may be concentrated in young-pair regimes; age must be logged for future controls.",
        fields_needed="pairCreatedAt or first_seen_timestamp",
    )

    add(
        feature_name="rss_sentiment_asof",
        category="news_context",
        priority="P2_FORWARD_COLLECTION_OPTIONAL",
        source_class="rss_sentiment_snapshot",
        collect_forward=True,
        current_evidence_status="not_validated_yet",
        reason="RSS context belongs to Anchor Plan, but no current E9 evidence approves it as discriminative.",
        fields_needed="rss_fetch_timestamp, sentiment_score, headline_count, source_count",
        compute_cost="LOW",
        external_api_required=False,
    )

    add(
        feature_name="wallet_level_whale_context_asof",
        category="onchain_wallet_context",
        priority="P3_FUTURE_EXTERNAL_DEFERRED",
        source_class="future_solana_helius_optional",
        collect_forward=False,
        current_evidence_status="deferred_requires_separate_provider_phase",
        reason="Potentially important, but should not be introduced in E9F without an explicit external-provider phase.",
        fields_needed="wallet flows, token balance deltas, signer/fee-payer activity",
        compute_cost="MEDIUM_TO_HIGH",
        external_api_required=True,
        missing_policy="not_available_until_future_provider_phase",
    )

    return pd.DataFrame(rows)


def build_logging_contract(feature_plan: pd.DataFrame) -> dict[str, Any]:
    collect_features = feature_plan[feature_plan["collect_forward"] == True]["feature_name"].tolist()

    return {
        "contract_name": "E9F_forward_context_logging_contract_v1",
        "created_at_utc": utc_now_iso(),
        "status": "research_collection_design_only",
        "not_approved_for_runtime": True,
        "not_approved_for_trading": True,
        "not_approved_for_modeling": True,
        "identity_fields_required": [
            "candidate_id",
            "candidate_policy_id",
            "target_row_id",
            "pair_address",
            "chain",
            "event_timestamp",
            "snapshot_asof_time",
            "feature_asof_time",
            "source_provider",
            "source_payload_id",
        ],
        "feature_fields_to_collect_forward": collect_features,
        "required_safety_fields": [
            "feature_available_before_entry",
            "feature_asof_time_lte_event_timestamp",
            "raw_payload_preserved",
            "missing_reason",
            "collection_error",
            "schema_version",
        ],
        "storage_recommendation": {
            "preferred_table_or_artifact": "forward_context_features",
            "format": "append-only parquet/csv or SQLite table in a future explicitly approved collection phase",
            "write_policy_for_this_phase": "do_not_write_runtime_database_in_E9F",
            "raw_data_policy": "preserve raw payloads; never delete raw data",
        },
        "future_validation_requirements": {
            "minimum_unique_positive_pairs_before_modeling": 20,
            "max_top_positive_pair_share_before_modeling": 0.20,
            "minimum_controls_per_positive": 3,
            "minimum_strong_control_match_share": 0.70,
            "must_pass_leave_one_pair_out": True,
            "must_compare_against_matched_controls": True,
            "must_compare_against_selected_losers_if_available": True,
        },
    }


def build_schema_markdown(feature_plan: pd.DataFrame, logging_contract: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# E9F - Forward Context Collection Schema")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append("Define a forward context collection schema after E9B/E9C found `whale_score_asof` interesting but E9D showed it was not robust enough for modeling or runtime.")
    lines.append("")
    lines.append("This document is design-only. It does not approve model training, runtime integration, UI changes, trading, external API calls, or SQLite writes.")
    lines.append("")
    lines.append("## Required identity fields")
    lines.append("")
    for f in logging_contract["identity_fields_required"]:
        lines.append(f"- `{f}`")
    lines.append("")
    lines.append("## Required safety fields")
    lines.append("")
    for f in logging_contract["required_safety_fields"]:
        lines.append(f"- `{f}`")
    lines.append("")
    lines.append("## Forward feature plan")
    lines.append("")
    lines.append("| feature | priority | category | collect_forward | status | source |")
    lines.append("|---|---:|---|---:|---|---|")
    for _, r in feature_plan.iterrows():
        lines.append(
            f"| `{r['feature_name']}` | {r['priority']} | {r['category']} | {r['collect_forward']} | {r['current_evidence_status']} | {r['source_class']} |"
        )
    lines.append("")
    lines.append("## Leakage rules")
    lines.append("")
    lines.append("- Every feature must be computed before the candidate entry decision.")
    lines.append("- `feature_asof_time` must be less than or equal to `candidate_event_timestamp`.")
    lines.append("- Labels, future returns, target columns, exit simulations, and post-entry fields must never be used as context features.")
    lines.append("- Missingness must be logged explicitly and must not be silently imputed in primary audits.")
    lines.append("- Raw payloads must be preserved; do not delete or overwrite raw data.")
    lines.append("")
    lines.append("## Future modeling prerequisites")
    lines.append("")
    req = logging_contract["future_validation_requirements"]
    for k, v in req.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    return "\n".join(lines)


def build_storage_plan_markdown() -> str:
    return "\n".join([
        "# E9F - Storage Plan",
        "",
        "## Status",
        "",
        "Design-only. No runtime or SQLite write is approved in E9F.",
        "",
        "## Recommended future storage pattern",
        "",
        "Use an append-only forward context artifact/table with stable row identity.",
        "",
        "Recommended columns:",
        "",
        "- candidate_id",
        "- candidate_policy_id",
        "- target_row_id",
        "- pair_address",
        "- chain",
        "- event_timestamp",
        "- snapshot_asof_time",
        "- feature_asof_time",
        "- source_provider",
        "- source_payload_id",
        "- feature_name",
        "- feature_value_numeric",
        "- feature_value_text",
        "- feature_available_before_entry",
        "- missing_reason",
        "- collection_error",
        "- schema_version",
        "",
        "## Raw data policy",
        "",
        "Preserve raw provider payloads. Do not delete raw data. Derived features must be reproducible from raw payloads and schema metadata.",
        "",
        "## No-divergence rule",
        "",
        "If future collection writes both SQLite and CSV/Parquet, a canonical feature event object must be created first, then exported to each sink. Do not compute SQLite and CSV feature rows independently.",
        "",
        "## Current phase restriction",
        "",
        "E9F does not implement this storage. It only defines the contract.",
        "",
    ])


def build_no_runtime_gate(
    e9a_decision: dict[str, Any],
    e9b_decision: dict[str, Any],
    e9c_decision: dict[str, Any],
    e9d_decision: dict[str, Any],
) -> dict[str, Any]:
    blockers = [
        "E9A weak control contract",
        "E9B research-only feature candidates",
        "E9C research-only whale-score contract",
        "E9D rule not robust enough",
        "E9D failed leave-one-pair-out",
        "Only 3 unique positive pairs in current rare-winner evidence",
        "Selected E9D rule captured only 1 unique selected positive pair",
    ]

    return {
        "decision": "E9F_FORWARD_COLLECTION_DESIGN_ONLY_NO_RUNTIME",
        "created_at_utc": utc_now_iso(),
        "approved_for_forward_collection_design": True,
        "approved_for_actual_collection_code_changes": False,
        "approved_for_modeling": False,
        "approved_for_training": False,
        "approved_for_runtime": False,
        "approved_for_ui": False,
        "approved_for_trading": False,
        "approved_for_reservoir_scoring": False,
        "reason": "E9 supports forward context collection design only. Existing evidence is research-only and not robust enough for modeling or runtime.",
        "blockers_to_runtime_or_modeling": blockers,
        "source_decisions": {
            "e9a": e9a_decision.get("final_e9a_status") or e9a_decision.get("decision"),
            "e9b": e9b_decision.get("decision"),
            "e9c": e9c_decision.get("decision"),
            "e9d": e9d_decision.get("decision"),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="E9F Forward Context Collection Design")
    parser.add_argument("--e9a-root", type=Path, default=DEFAULT_E9A_ROOT)
    parser.add_argument("--e9b-root", type=Path, default=DEFAULT_E9B_ROOT)
    parser.add_argument("--e9c-root", type=Path, default=DEFAULT_E9C_ROOT)
    parser.add_argument("--e9d-root", type=Path, default=DEFAULT_E9D_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    output_root = args.output_root or (
        ROOT / "data" / "training" / "manual_verified_results" / f"phase_e9f_forward_context_collection_design_{timestamp_slug()}"
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

    e9d_selected_rule = read_csv_optional(args.e9d_root / "data" / "e9d_selected_rule.csv")

    feature_plan = build_feature_plan(
        e9b_decision=e9b_decision,
        e9c_decision=e9c_decision,
        e9d_decision=e9d_decision,
    )
    write_csv(data_dir / "e9f_forward_context_feature_plan.csv", feature_plan)

    logging_contract = build_logging_contract(feature_plan)
    write_json(data_dir / "e9f_context_logging_contract.json", logging_contract)

    schema_md = build_schema_markdown(feature_plan, logging_contract)
    write_text(reports_dir / "e9f_forward_context_collection_schema.md", schema_md)

    storage_plan_md = build_storage_plan_markdown()
    write_text(reports_dir / "e9f_storage_plan.md", storage_plan_md)

    no_runtime_gate = build_no_runtime_gate(
        e9a_decision=e9a_decision,
        e9b_decision=e9b_decision,
        e9c_decision=e9c_decision,
        e9d_decision=e9d_decision,
    )
    write_json(audits_dir / "e9f_no_runtime_decision_gate.json", no_runtime_gate)

    manifest = {
        "phase": "E9F",
        "branch_name": "phase_e9f_forward_context_collection_design",
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
        },
        "source_decisions": {
            "e9a": e9a_decision.get("final_e9a_status") or e9a_decision.get("decision"),
            "e9b": e9b_decision.get("decision"),
            "e9c": e9c_decision.get("decision"),
            "e9d": e9d_decision.get("decision"),
        },
        "outputs": {
            "output_root": str(output_root),
            "manifest": str(reports_dir / "e9f_manifest.json"),
            "summary": str(reports_dir / "e9f_summary_for_upload.txt"),
            "forward_context_collection_schema": str(reports_dir / "e9f_forward_context_collection_schema.md"),
            "storage_plan": str(reports_dir / "e9f_storage_plan.md"),
            "feature_plan": str(data_dir / "e9f_forward_context_feature_plan.csv"),
            "logging_contract": str(data_dir / "e9f_context_logging_contract.json"),
            "no_runtime_decision_gate": str(audits_dir / "e9f_no_runtime_decision_gate.json"),
        },
        "decision": no_runtime_gate["decision"],
    }
    write_json(reports_dir / "e9f_manifest.json", manifest)

    selected_rule_text = "not_available"
    if not e9d_selected_rule.empty:
        selected_rule_text = e9d_selected_rule.iloc[0].to_dict()

    summary_lines = [
        "Phase / branch name",
        "",
        "E9F - Forward Context Collection Design",
        "",
        "Run status",
        "",
        "COMPLETED",
        "",
        "Decision",
        "",
        no_runtime_gate["decision"],
        "",
        "Scope",
        "",
        "Design-only forward context collection plan.",
        "No model training, no runtime, no UI, no trading, no SQLite writes, no external APIs, no LLM calls.",
        "",
        "Why E9F is needed",
        "",
        "E9D showed that whale_score_asof has a research-only rule signal, but it did not survive leave-one-pair-out and was concentrated in one selected positive pair.",
        "Therefore the correct next step is not modeling, but forward collection design.",
        "",
        "Source decisions",
        "",
        f"- E9A: {manifest['source_decisions']['e9a']}",
        f"- E9B: {manifest['source_decisions']['e9b']}",
        f"- E9C: {manifest['source_decisions']['e9c']}",
        f"- E9D: {manifest['source_decisions']['e9d']}",
        "",
        "Forward collection priorities",
        "",
    ]

    for _, r in feature_plan.iterrows():
        if bool(r["collect_forward"]):
            summary_lines.append(f"- {r['priority']}: {r['feature_name']} ({r['category']})")

    summary_lines.extend([
        "",
        "Deferred features",
        "",
    ])

    deferred = feature_plan[feature_plan["collect_forward"] == False]
    if deferred.empty:
        summary_lines.append("None.")
    else:
        for _, r in deferred.iterrows():
            summary_lines.append(f"- {r['priority']}: {r['feature_name']} ({r['reason']})")

    summary_lines.extend([
        "",
        "Future validation requirements",
        "",
    ])

    for k, v in logging_contract["future_validation_requirements"].items():
        summary_lines.append(f"- {k}: {v}")

    summary_lines.extend([
        "",
        "Approvals",
        "",
        f"- approved_for_forward_collection_design: {no_runtime_gate['approved_for_forward_collection_design']}",
        f"- approved_for_actual_collection_code_changes: {no_runtime_gate['approved_for_actual_collection_code_changes']}",
        f"- approved_for_modeling: {no_runtime_gate['approved_for_modeling']}",
        f"- approved_for_training: {no_runtime_gate['approved_for_training']}",
        f"- approved_for_runtime: {no_runtime_gate['approved_for_runtime']}",
        f"- approved_for_ui: {no_runtime_gate['approved_for_ui']}",
        f"- approved_for_trading: {no_runtime_gate['approved_for_trading']}",
        "",
        "E9D selected rule reference",
        "",
        str(selected_rule_text),
        "",
        "Final interpretation",
        "",
        "E9F approves only a forward context collection design. It does not approve implementation, modeling, runtime, UI, or trading.",
    ])

    write_text(reports_dir / "e9f_summary_for_upload.txt", "\n".join(summary_lines))

    print(json.dumps({
        "status": "completed",
        "phase": "E9F",
        "output_root": str(output_root),
        "decision": no_runtime_gate["decision"],
        "approved_for_forward_collection_design": no_runtime_gate["approved_for_forward_collection_design"],
        "approved_for_actual_collection_code_changes": no_runtime_gate["approved_for_actual_collection_code_changes"],
        "approved_for_modeling": no_runtime_gate["approved_for_modeling"],
        "approved_for_runtime": no_runtime_gate["approved_for_runtime"],
        "summary": str(reports_dir / "e9f_summary_for_upload.txt"),
        "decision_gate": str(audits_dir / "e9f_no_runtime_decision_gate.json"),
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
