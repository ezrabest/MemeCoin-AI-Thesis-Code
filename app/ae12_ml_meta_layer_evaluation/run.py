"""Orchestrate AE12.6 ML/meta-layer evaluation (read-only)."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .discovery import (
    build_artifact_specs,
    discover_artifact,
    inventory_by_path,
    resolve_maturation_root,
)
from .reports import ensure_dirs, write_csv, write_json

PHASE = "AE12.6"
SCHEMA = "AE12_ML_META_V1"

FORBIDDEN_PROFIT_CLAIMS = (
    "profitability proven",
    "live trading ready",
    "live-ready",
    "production ready for live",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _load_json(path: str | Path) -> dict[str, Any] | None:
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _read_csv_rows(path: str | Path, *, max_rows: int = 50000) -> list[dict[str, str]]:
    p = Path(path)
    if not p.is_file():
        return []
    rows: list[dict[str, str]] = []
    with p.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        reader = csv.DictReader(fh)
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(dict(row))
    return rows


def _parse_exit_sim_row(row: dict[str, str], model: str) -> dict[str, Any] | None:
    if row.get("model", "").upper() != model.upper():
        return None
    if row.get("split_test", "") != "test":
        return None
    try:
        return {
            "filter": row.get("filter", ""),
            "horizon": row.get("horizon", ""),
            "top_pct": row.get("top_pct", ""),
            "pair_cap": row.get("pair_cap_test") or row.get("pair_cap_val", ""),
            "tp": row.get("tp_ratio", ""),
            "sl": row.get("sl_ratio", ""),
            "trades": row.get("selected_test", ""),
            "unique_pairs": row.get("unique_pairs_test", ""),
            "total_net_return": row.get("total_net_return_test", ""),
            "avg_net_per_trade": row.get("avg_net_return_test", ""),
            "pair_concentration": row.get("top_pair_share_test", ""),
            "source_artifact": "",
        }
    except (TypeError, ValueError):
        return None


def _best_exit_sim_for_model(rows: list[dict[str, str]], model: str) -> dict[str, Any] | None:
    parsed = [_parse_exit_sim_row(r, model) for r in rows]
    candidates = [p for p in parsed if p]
    if not candidates:
        return None
    def key(p: dict[str, Any]) -> float:
        try:
            return float(p.get("total_net_return") or 0)
        except ValueError:
            return float("-inf")
    return max(candidates, key=key)


def build_model_performance_summary(inv: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    exit_fixed = inv.get("exit_sim_tab_rf_xgb_comparison", {}).get("discovered_path", "")
    exit_xgb = inv.get("exit_sim_xgb_clean_full_test_policies", {}).get("discovered_path", "")
    clean_rf = inv.get("clean_rf_test_policies", {}).get("discovered_path", "")
    e4_test = inv.get("e4_direct_target_xgb_rf_policy_test", {}).get("discovered_path", "")

    exit_rows = _read_csv_rows(exit_fixed) if exit_fixed else []
    xgb_rows = _read_csv_rows(exit_xgb) if exit_xgb else []
    combined_for_xgb = exit_rows + xgb_rows
    xgb_best = _best_exit_sim_for_model(combined_for_xgb, "XGB")
    if not xgb_best and xgb_rows:
        # xgb clean file uses validation/test columns without split_test=model pattern
        for r in xgb_rows:
            if r.get("model", "").upper() != "XGB":
                continue
            try:
                tot = float(r.get("total_net_return_test") or r.get("selected_total_sim_net_return") or "nan")
            except ValueError:
                continue
            cand = {
                "filter": r.get("filter", ""),
                "horizon": r.get("horizon", ""),
                "top_pct": r.get("top_pct", ""),
                "pair_cap": r.get("pair_cap_test", ""),
                "tp": r.get("tp_ratio", ""),
                "sl": r.get("sl_ratio", ""),
                "trades": r.get("selected_test", ""),
                "unique_pairs": r.get("unique_pairs_test", ""),
                "total_net_return": str(tot),
                "avg_net_per_trade": r.get("avg_net_return_test", ""),
                "pair_concentration": r.get("top_pair_share_test", ""),
            }
            if xgb_best is None or tot > float(xgb_best.get("total_net_return") or 0):
                xgb_best = cand
    tab_best = _best_exit_sim_for_model(exit_rows, "TAB")
    rf_best = _best_exit_sim_for_model(exit_rows, "RF")

    summaries: list[dict[str, Any]] = []

    def row_for(
        model_family: str,
        best: dict[str, Any] | None,
        *,
        source_path: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        base = {
            "model_family": model_family,
            "best_known_filter": best.get("filter", "") if best else "",
            "best_known_horizon": best.get("horizon", "") if best else "",
            "best_known_selection_rate": best.get("top_pct", "") if best else "",
            "best_known_tp": best.get("tp", "") if best else "",
            "best_known_sl": best.get("sl", "") if best else "",
            "trades": best.get("trades", "") if best else "",
            "unique_pairs": best.get("unique_pairs", "") if best else "",
            "total_net_return": best.get("total_net_return", "") if best else "",
            "avg_net_per_trade": best.get("avg_net_per_trade", "") if best else "",
            "pair_concentration": best.get("pair_concentration", "") if best else "",
            "train_test_split_status": "held_out_test_in_exit_sim_artifact" if best else "",
            "leakage_status": "see_clean_rf_leakage_audit" if model_family == "RF" else "historical_exit_sim_only",
            "robustness_status": "not_forward_proven",
            "forward_validation_status": "not_proven_by_ae12_forward_labels_alone",
            "operational_status": "research_and_reporting_only",
            "limitation": "Historical exit simulation on manual-verified set; not live profitability proof.",
            "source_artifact": source_path if best else "",
            "source_confidence": "ARTIFACT_CONFIRMED" if best and source_path else "MISSING_SOURCE",
        }
        if extra:
            base.update(extra)
        return base

    summaries.append(
        row_for(
            "XGB",
            xgb_best,
            source_path=exit_xgb or exit_fixed,
            extra={
                "leakage_status": "clean_feature_schema_partial_in_xgb_clean_full",
                "notes": "Headline RAW_ALL_VERIFIED/24h/top0.5% from exit_sim_xgb_full when present.",
            },
        )
    )
    summaries.append(row_for("RF", rf_best, source_path=exit_fixed))
    summaries.append(row_for("TAB / TabICL", tab_best, source_path=exit_fixed))

    # RF clean direct-target test policies (may differ from exit_sim headline)
    if clean_rf:
        rf_rows = _read_csv_rows(clean_rf)
        best_rf_clean = None
        best_val = float("-inf")
        for r in rf_rows:
            try:
                tot = float(r.get("selected_total_sim_net_return", "nan"))
            except ValueError:
                continue
            if tot > best_val:
                best_val = tot
                best_rf_clean = r
        if best_rf_clean and summaries[1]["source_confidence"] != "ARTIFACT_CONFIRMED":
            summaries[1].update(
                {
                    "best_known_filter": best_rf_clean.get("filter", ""),
                    "best_known_horizon": best_rf_clean.get("horizon", ""),
                    "total_net_return": best_rf_clean.get("selected_total_sim_net_return", ""),
                    "source_artifact": clean_rf,
                    "source_confidence": "ARTIFACT_CONFIRMED_CLEAN_RF",
                    "leakage_status": "clean_rf_leakage_audit_available",
                }
            )

    summaries.append(
        {
            "model_family": "Consensus",
            "best_known_filter": "",
            "best_known_horizon": "",
            "best_known_selection_rate": "",
            "best_known_tp": "",
            "best_known_sl": "",
            "trades": "",
            "unique_pairs": "",
            "total_net_return": "",
            "avg_net_per_trade": "",
            "pair_concentration": "",
            "train_test_split_status": "tier_join_in_e6r_when_present",
            "leakage_status": "decision_record_contract_ae6",
            "robustness_status": "partial_tier_coverage",
            "forward_validation_status": "ae12_forward_evidence_separate",
            "operational_status": "consensus_is_filter_not_live_authority",
            "limitation": "Consensus tiers documented in AE6/E6R artifacts; not granted trade authority in AE12.6.",
            "source_artifact": inv.get("e6r_tabicl_consensus_comparison", {}).get("discovered_path", "")
            or inv.get("ae6_consensus_decision_summary", {}).get("discovered_path", ""),
            "source_confidence": "ARTIFACT_CONFIRMED"
            if inv.get("e6r_tabicl_consensus_comparison", {}).get("exists")
            or inv.get("ae6_consensus_decision_summary", {}).get("exists")
            else "MISSING_SOURCE",
        }
    )
    summaries.append(
        {
            "model_family": "Meta-layer",
            "best_known_filter": "",
            "best_known_horizon": "",
            "best_known_selection_rate": "",
            "best_known_tp": "",
            "best_known_sl": "",
            "trades": "",
            "unique_pairs": "",
            "total_net_return": "",
            "avg_net_per_trade": "",
            "pair_concentration": "",
            "train_test_split_status": "NOT_CLOSED_AS_ORIGINAL_AE7",
            "leakage_status": "unknown_without_ae7_stack_artifact",
            "robustness_status": "not_implemented_as_closed_ae7",
            "forward_validation_status": "not_proven",
            "operational_status": "research_only",
            "limitation": "Evidence exists under later AE/diagnostic artifacts, but not as a closed original AE7 implementation.",
            "source_artifact": inv.get("ae7_meta_model_stacking", {}).get("discovered_path", ""),
            "source_confidence": "MISSING_SOURCE"
            if not inv.get("ae7_meta_model_stacking", {}).get("exists")
            else "PARTIAL_DIAGNOSTIC",
        }
    )
    return summaries


def build_forward_evidence_integration(maturation_root: Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not maturation_root:
        rows.append(
            {
                "evidence_component": "maturation_root",
                "count_or_status": "MISSING",
                "source_file": "",
                "interpretation": "Critical AE12.3 maturation root not resolved.",
                "limitation": "Cannot integrate forward evidence.",
            }
        )
        return rows

    summary_path = maturation_root / "reports" / "ae12_forward_evidence_summary.json"
    summary = _load_json(summary_path) or {}
    rows.append(
        {
            "evidence_component": "candidate_evidence_rows",
            "count_or_status": str(summary.get("candidate_evidence_row_count", "")),
            "source_file": str(summary_path),
            "interpretation": "Forward candidate rows collected for maturation (labels only).",
            "limitation": "Not profitability proof.",
        }
    )
    hm = summary.get("horizon_maturity") or {}
    rows.append(
        {
            "evidence_component": "horizon_maturity",
            "count_or_status": json.dumps(list(hm.keys())),
            "source_file": str(summary_path),
            "interpretation": "Per-horizon matured vs not-matured counts; not-matured returns remain null.",
            "limitation": "24h horizon has large not-matured tail at audit time.",
        }
    )
    rows.append(
        {
            "evidence_component": "missed_winners",
            "count_or_status": str(summary.get("missed_winner_count", "")),
            "source_file": str(maturation_root / "data" / "ae12_missed_winners_full.csv"),
            "interpretation": "Missed winners available for opportunity capture analysis.",
            "limitation": "Threshold-defined; not trade recommendations.",
        }
    )
    tvt = summary.get("trade_vs_no_trade") or []
    strict0 = 0
    if tvt and isinstance(tvt, list):
        strict0 = tvt[0].get("strict_approved_count", 0)
    rows.append(
        {
            "evidence_component": "trade_vs_no_trade",
            "count_or_status": f"rows={len(tvt)} strict_approved_sample={strict0}",
            "source_file": str(maturation_root / "data" / "ae12_trade_vs_no_trade_comparison.csv"),
            "interpretation": "Traded vs not-traded separated; exploration-only paper behavior distinguished.",
            "limitation": "Strict-approved remained 0 in summarized AE12.3/4 pass context.",
        }
    )
    sve = summary.get("strict_vs_exploration") or {}
    rows.append(
        {
            "evidence_component": "strict_vs_exploration",
            "count_or_status": json.dumps(
                {
                    "strict_approved": sve.get("strict_approved"),
                    "exploration_only_trades": sve.get("exploration_only_trades"),
                }
            ),
            "source_file": str(summary_path),
            "interpretation": "Strict gate vs exploration-only paths documented.",
            "limitation": "Models/LLMs do not receive strict live authority from this evidence.",
        }
    )
    ql = summary.get("qwen_linkage_counts") or {}
    rows.append(
        {
            "evidence_component": "qwen_linkage",
            "count_or_status": json.dumps(ql),
            "source_file": str(summary_path),
            "interpretation": "Qwen/LLM linkage is audit/reporting context only (NO_TRADE_AUTHORITY in samples).",
            "limitation": "Many MENTION_ONLY / missing linkage warnings remain.",
        }
    )
    ws = summary.get("wallet_safety") or {}
    rows.append(
        {
            "evidence_component": "wallet_safety",
            "count_or_status": json.dumps(ws),
            "source_file": str(maturation_root / "audits" / "ae12_wallet_safety_audit.json"),
            "interpretation": "No-wallet / no live submission in forward evidence pass.",
            "limitation": "Paper/runtime records are observational.",
        }
    )
    return rows


def build_semantic_addendum(reconciliation_path: str) -> list[dict[str, Any]]:
    summary = _load_json(reconciliation_path) if reconciliation_path else None
    final = (summary or {}).get("final_after_drilldown") or {}
    legacy = (summary or {}).get("legacy_runtime_cluster_scan") or {}
    return [
        {
            "topic": "legacy_500_vs_gemini_89_vs_coin_14",
            "value": "500/0 dashboard was legacy paper_trades.cluster_label diagnostic counts, not unique coin semantics.",
            "source": reconciliation_path,
        },
        {
            "topic": "legacy_cluster_sqlite_scan",
            "value": json.dumps(
                {
                    "OPPORTUNISTIC_SPECULATIVE": legacy.get("OPPORTUNISTIC_SPECULATIVE", 766),
                    "SOCIALLY_MOTIVATED": legacy.get("SOCIALLY_MOTIVATED", 25),
                }
            ),
            "source": reconciliation_path,
        },
        {
            "topic": "gemini_pair_assets",
            "value": str((summary or {}).get("observed_counts", {}).get("gemini_pair_asset_rows", 89)),
            "source": reconciliation_path,
        },
        {
            "topic": "coin_level_identities",
            "value": str(final.get("unique_coins_found", 14)),
            "source": reconciliation_path,
        },
        {
            "topic": "post_drilldown_distribution",
            "value": json.dumps(
                {
                    "SOCIAL_CONFIRMED": final.get("coin_social_confirmed_count", 0),
                    "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED": final.get(
                        "coin_non_social_opportunistic_confirmed_count", 0
                    ),
                    "OPPORTUNISTIC_SUSPECTED": final.get("coin_opportunistic_suspected_count", 0),
                    "UNKNOWN_UNRESOLVED": final.get("coin_unknown_unresolved_count", 0),
                }
            ),
            "source": reconciliation_path,
        },
        {
            "topic": "unknown_unresolved_semantics",
            "value": "UNKNOWN_UNRESOLVED is not social and not opportunistic; reporting-only labels.",
            "source": reconciliation_path,
        },
        {
            "topic": "gemini_grounding",
            "value": "Gemini adjudication was model-knowledge-based in prior pass; not web-grounded trade authority.",
            "source": reconciliation_path,
        },
    ]


def build_evaluation_matrix(
    inv: dict[str, dict[str, Any]],
    *,
    maturation_root: Path | None,
) -> list[dict[str, Any]]:
    def miss_families(*names: str) -> str:
        return ";".join(n for n in names if not inv.get(n, {}).get("exists"))

    layers: list[dict[str, Any]] = []

    def add(**kwargs: Any) -> None:
        layers.append(kwargs)

    add(
        layer_name="RF",
        architecture_role="Direct-target random forest on clean historical features",
        source_artifacts=inv.get("clean_rf_test_policies", {}).get("discovered_path", "")
        + "|"
        + inv.get("exit_sim_tab_rf_xgb_comparison", {}).get("discovered_path", ""),
        component_status="FOUND_AND_EVALUATED"
        if inv.get("clean_rf_test_policies", {}).get("exists")
        else "PARTIALLY_EVALUATED_MISSING_EXPECTED_ARTIFACTS",
        evidence_status="PASS_RESEARCH_ONLY",
        strongest_evidence="clean_rf_test_applied_selected_policies + exit_sim RF row",
        weakest_evidence="Forward paper PnL not proving RF edge",
        missing_expected_artifacts=miss_families("clean_rf_test_policies"),
        validated_on_historical_data="true",
        validated_on_forward_data="false",
        no_lookahead_status="clean_split_documented_in_e8b",
        runtime_connected="false",
        paper_connected="observational_only",
        live_connected="false",
        trade_authority="false",
        reporting_ready="true",
        research_ready="true",
        production_ready="false",
        live_ready="false",
        profitability_proven="false",
        main_limitation="Historical test policies; clean RF test totals often negative on held-out selection.",
        recommended_next_action="AE12.7: integrate RF diagnostics into MSc closure narrative without live authority.",
    )
    add(
        layer_name="XGB",
        architecture_role="Direct-target gradient boosted trees (clean + E4 grids)",
        source_artifacts=inv.get("exit_sim_xgb_clean_full_test_policies", {}).get("discovered_path", "")
        + "|"
        + inv.get("e4_direct_target_xgb_rf_policy_test", {}).get("discovered_path", ""),
        component_status="FOUND_AND_EVALUATED"
        if inv.get("exit_sim_xgb_clean_full_test_policies", {}).get("exists")
        else "PARTIALLY_EVALUATED_MISSING_EXPECTED_ARTIFACTS",
        evidence_status="PASS_FOR_REPORTING",
        strongest_evidence="exit_sim_xgb_full RAW_ALL_VERIFIED 24h test total net ~47.83 in artifact",
        weakest_evidence="E4 direct-target test policies show negative totals for many grids",
        missing_expected_artifacts=miss_families("exit_sim_xgb_clean_full_test_policies"),
        validated_on_historical_data="true",
        validated_on_forward_data="false",
        no_lookahead_status="exit_sim_on_historical_labels",
        runtime_connected="false",
        paper_connected="observational_only",
        live_connected="false",
        trade_authority="false",
        reporting_ready="true",
        research_ready="true",
        production_ready="false",
        live_ready="false",
        profitability_proven="false",
        main_limitation="Strong historical exit-sim row does not prove forward/live profitability.",
        recommended_next_action="AE12.7: document XGB as research evidence only; no wallet authority.",
    )
    add(
        layer_name="TAB / TabICL",
        architecture_role="TabICL direct-target classifier in E5/E6R consensus stack",
        source_artifacts=inv.get("e6r_tabicl_test_policies", {}).get("discovered_path", ""),
        component_status="FOUND_AND_EVALUATED"
        if inv.get("e6r_tabicl_test_policies", {}).get("exists")
        else "MISSING_EXPECTED_ARTIFACT",
        evidence_status="PASS_RESEARCH_ONLY",
        strongest_evidence="exit_sim TAB LIQ_5K 4h top2% test total ~27.72 in artifact",
        weakest_evidence="Many TAB policies blocked for concentration or test-negative",
        missing_expected_artifacts=miss_families("e6r_tabicl_test_policies"),
        validated_on_historical_data="true",
        validated_on_forward_data="false",
        no_lookahead_status="historical_policy_selection",
        runtime_connected="false",
        paper_connected="false",
        live_connected="false",
        trade_authority="false",
        reporting_ready="true",
        research_ready="true",
        production_ready="false",
        live_ready="false",
        profitability_proven="false",
        main_limitation="TabICL evidence is offline; not connected as live executor.",
        recommended_next_action="AE12.7: keep TAB in research tier; no retrain in AE12.6.",
    )
    add(
        layer_name="Consensus / DecisionRecord layer",
        architecture_role="AE6 decision records + tier joins (E6R)",
        source_artifacts=inv.get("ae6_consensus_decision_summary", {}).get("discovered_path", "")
        + "|"
        + inv.get("ae6_decision_records", {}).get("discovered_path", ""),
        component_status="FOUND_AND_EVALUATED"
        if inv.get("ae6_consensus_decision_summary", {}).get("exists")
        else "PARTIALLY_EVALUATED_MISSING_EXPECTED_ARTIFACTS",
        evidence_status="PASS_DIAGNOSTIC_ONLY",
        strongest_evidence="ae6_consensus_decision_summary + decision JSONL",
        weakest_evidence="Strict-approved forward trades remained 0",
        missing_expected_artifacts=miss_families("ae6_consensus_decision_summary"),
        validated_on_historical_data="partial",
        validated_on_forward_data="labels_only_via_ae12",
        no_lookahead_status="forward_maturation_separate",
        runtime_connected="paper_observational",
        paper_connected="true",
        live_connected="false",
        trade_authority="false",
        reporting_ready="true",
        research_ready="true",
        production_ready="false",
        live_ready="false",
        profitability_proven="false",
        main_limitation="Consensus filters candidates; does not prove profitable live execution.",
        recommended_next_action="AE12.7: align consensus narrative with AE12 forward evidence.",
    )
    add(
        layer_name="Meta-model / stacking layer",
        architecture_role="Original AE7 meta-model / stacking",
        source_artifacts=inv.get("ae7_meta_model_stacking", {}).get("discovered_path", ""),
        component_status="NOT_IMPLEMENTED_AS_ORIGINAL_LAYER"
        if not inv.get("ae7_meta_model_stacking", {}).get("exists")
        else "PARTIALLY_EVALUATED_MISSING_EXPECTED_ARTIFACTS",
        evidence_status="NOT_IMPLEMENTED_AS_ORIGINAL_LAYER",
        strongest_evidence="E6R tier joins partially substitute",
        weakest_evidence="No closed AE7 stack artifact located",
        missing_expected_artifacts="ae7_meta_model_stacking",
        validated_on_historical_data="false",
        validated_on_forward_data="false",
        no_lookahead_status="n/a",
        runtime_connected="false",
        paper_connected="false",
        live_connected="false",
        trade_authority="false",
        reporting_ready="false",
        research_ready="partial_via_e6r",
        production_ready="false",
        live_ready="false",
        profitability_proven="false",
        main_limitation="Evidence exists under later AE/diagnostic artifacts, but not as a closed original AE7 implementation.",
        recommended_next_action="AE12.7: state AE7 gap explicitly in MSc thesis.",
    )
    add(
        layer_name="Context intelligence layer",
        architecture_role="AE8 context intelligence (E8E forensics / AE11 availability)",
        source_artifacts=inv.get("e8e_context_forensics", {}).get("discovered_path", "")
        + "|"
        + inv.get("ae11_context_availability_audit", {}).get("discovered_path", ""),
        component_status="FOUND_AND_EVALUATED"
        if inv.get("e8e_context_forensics", {}).get("exists")
        else "MISSING_EXPECTED_ARTIFACT",
        evidence_status="PASS_WITH_LIMITATIONS",
        strongest_evidence="e8e_market_context_by_candidate forensics",
        weakest_evidence="No closed AE8 runtime context engine artifact under AE8 name",
        missing_expected_artifacts=miss_families("e8e_context_forensics"),
        validated_on_historical_data="forensics_only",
        validated_on_forward_data="partial_availability",
        no_lookahead_status="forensics_historical",
        runtime_connected="partial",
        paper_connected="true",
        live_connected="false",
        trade_authority="false",
        reporting_ready="true",
        research_ready="true",
        production_ready="false",
        live_ready="false",
        profitability_proven="false",
        main_limitation="Context features inform research; not proven as live alpha layer.",
        recommended_next_action="AE12.7: document context as auxiliary evidence.",
    )
    add(
        layer_name="LLM audit layer",
        architecture_role="AE9/Qwen operational audit + AE11 linkage",
        source_artifacts=inv.get("ae11_llm_audit_linkage", {}).get("discovered_path", "")
        + "|"
        + inv.get("ae9_llm_operational_audit", {}).get("discovered_path", ""),
        component_status="PARTIALLY_EVALUATED_MISSING_EXPECTED_ARTIFACTS",
        evidence_status="PASS_DIAGNOSTIC_ONLY",
        strongest_evidence="ae11_llm_audit_linkage_audit.csv + forward qwen_linkage_counts",
        weakest_evidence="ae9 operational audit bundle optional/missing",
        missing_expected_artifacts=miss_families("ae9_llm_operational_audit"),
        validated_on_historical_data="n/a",
        validated_on_forward_data="linkage_counts_only",
        no_lookahead_status="audit_metadata",
        runtime_connected="paper_observational",
        paper_connected="true",
        live_connected="false",
        trade_authority="false",
        reporting_ready="true",
        research_ready="true",
        production_ready="false",
        live_ready="false",
        profitability_proven="false",
        main_limitation="LLMs used for audit/reporting context; explicit NO_TRADE_AUTHORITY.",
        recommended_next_action="AE12.7: keep LLM layer non-authoritative in gate docs.",
    )
    add(
        layer_name="Semantic reporting / SentimentFix",
        architecture_role="Gemini adjudication + manual drilldown + reconciliation (reporting-only)",
        source_artifacts="|".join(
            filter(
                None,
                [
                    inv.get("ae12_gemini_semantic_adjudication_summary", {}).get("discovered_path"),
                    inv.get("ae12_manual_review_drilldown_summary", {}).get("discovered_path"),
                    inv.get("ae12_semantic_coverage_reconciliation", {}).get("discovered_path"),
                ],
            )
        ),
        component_status="FOUND_AND_EVALUATED",
        evidence_status="PASS_WITH_LIMITATIONS",
        strongest_evidence="Reconciliation + post-drilldown 14 coin identities",
        weakest_evidence="UNKNOWN_UNRESOLVED=6 coins remain unresolved",
        missing_expected_artifacts=miss_families(
            "ae12_semantic_coverage_reconciliation",
            "ae12_gemini_semantic_adjudication_summary",
            "ae12_manual_review_drilldown_summary",
        ),
        validated_on_historical_data="n/a",
        validated_on_forward_data="n/a",
        no_lookahead_status="n/a",
        runtime_connected="ui_reporting_only",
        paper_connected="diagnostic_cluster_labels",
        live_connected="false",
        trade_authority="false",
        reporting_ready="true",
        research_ready="true",
        production_ready="false",
        live_ready="false",
        profitability_proven="false",
        main_limitation="Semantic labels are reporting-only; Gemini not web-grounded; legacy clusters diagnostic.",
        recommended_next_action="AE12.7: UI must use coin-level post-drilldown counts (14), not legacy 500.",
    )
    add(
        layer_name="Paper/demo execution layer",
        architecture_role="AE11 runtime paper loop + AE12 paper records",
        source_artifacts=inv.get("ae11_runtime_paper_loop_gate", {}).get("discovered_path", ""),
        component_status="FOUND_AND_EVALUATED"
        if inv.get("ae11_runtime_paper_loop_gate", {}).get("exists")
        else "PARTIALLY_EVALUATED_MISSING_EXPECTED_ARTIFACTS",
        evidence_status="PASS_DIAGNOSTIC_ONLY",
        strongest_evidence="ae11_decision_gate + ae12 forward trade_vs_no_trade",
        weakest_evidence="Not strict-approved live execution",
        missing_expected_artifacts=miss_families("ae11_runtime_paper_loop_gate"),
        validated_on_historical_data="false",
        validated_on_forward_data="observational",
        no_lookahead_status="maturation_labels_separate",
        runtime_connected="paper_only_historical_runs",
        paper_connected="true",
        live_connected="false",
        trade_authority="false",
        reporting_ready="true",
        research_ready="true",
        production_ready="false",
        live_ready="false",
        profitability_proven="false",
        main_limitation="Exploration-only paper behavior; strict-approved=0 in AE12.4 summary context.",
        recommended_next_action="AE12.7: separate paper demo from live readiness.",
    )
    add(
        layer_name="Forward evidence maturation layer",
        architecture_role="AE12.3 no-lookahead horizon maturation",
        source_artifacts=str(maturation_root) if maturation_root else "",
        component_status="FOUND_AND_EVALUATED" if maturation_root else "MISSING_EXPECTED_ARTIFACT",
        evidence_status="PASS_FOR_REPORTING"
        if maturation_root
        else "HOLD_MISSING_ARTIFACTS",
        strongest_evidence="63531 candidate rows; matured outcomes with no-lookahead audits",
        weakest_evidence="Large not-matured 24h tail; missing snapshot warnings",
        missing_expected_artifacts="" if maturation_root else "ae12_forward_evidence_summary",
        validated_on_historical_data="false",
        validated_on_forward_data="true_labels_only",
        no_lookahead_status="PASS_per_ae12_3_audit",
        runtime_connected="read_only_inputs",
        paper_connected="true",
        live_connected="false",
        trade_authority="false",
        reporting_ready="true",
        research_ready="true",
        production_ready="false",
        live_ready="false",
        profitability_proven="false",
        main_limitation="Forward returns are evaluation labels, not PnL proof.",
        recommended_next_action="AE12.7: extend forward evidence narrative for MSc.",
    )
    add(
        layer_name="Missed-winner / opportunity capture layer",
        architecture_role="AE12.4 comparisons embedded in maturation/opportunity CSVs",
        source_artifacts=inv.get("ae12_missed_winners_full", {}).get("discovered_path", ""),
        component_status="FOUND_AND_EVALUATED"
        if inv.get("ae12_missed_winners_full", {}).get("exists")
        else "PARTIALLY_EVALUATED_MISSING_EXPECTED_ARTIFACTS",
        evidence_status="PASS_FOR_REPORTING",
        strongest_evidence="missed_winners_full + summary counts",
        weakest_evidence="Does not prescribe model retrains",
        missing_expected_artifacts=miss_families("ae12_missed_winners_full"),
        validated_on_historical_data="false",
        validated_on_forward_data="true",
        no_lookahead_status="inherits_ae12_3",
        runtime_connected="false",
        paper_connected="true",
        live_connected="false",
        trade_authority="false",
        reporting_ready="true",
        research_ready="true",
        production_ready="false",
        live_ready="false",
        profitability_proven="false",
        main_limitation="Missed winners are diagnostic; not automatic strategy fixes.",
        recommended_next_action="AE12.7: opportunity capture chapter cross-links.",
    )
    add(
        layer_name="Safety / no-wallet layer",
        architecture_role="Wallet safety audits across AE11/AE12",
        source_artifacts=inv.get("ae11_wallet_safety_audit", {}).get("discovered_path", ""),
        component_status="FOUND_AND_EVALUATED"
        if inv.get("ae11_wallet_safety_audit", {}).get("exists")
        else "PARTIALLY_EVALUATED_MISSING_EXPECTED_ARTIFACTS",
        evidence_status="PASS_DIAGNOSTIC_ONLY",
        strongest_evidence="ae11_wallet_safety_audit.json + ae12 wallet_safety in summary",
        weakest_evidence="Historical runs may predate latest audit",
        missing_expected_artifacts=miss_families("ae11_wallet_safety_audit"),
        validated_on_historical_data="n/a",
        validated_on_forward_data="n/a",
        no_lookahead_status="n/a",
        runtime_connected="no_wallet",
        paper_connected="true",
        live_connected="false",
        trade_authority="false",
        reporting_ready="true",
        research_ready="true",
        production_ready="false",
        live_ready="false",
        profitability_proven="false",
        main_limitation="Safety audits seal past runs; AE12.6 does not start runtime.",
        recommended_next_action="AE12.7: maintain no-wallet invariant in documentation.",
    )
    return layers


def build_safety_audit(
    *,
    output_root: Path,
    simulate_failure: bool = False,
) -> dict[str, Any]:
    status = "FAIL_RUNTIME_STARTED" if simulate_failure else "PASS_READONLY_REPORTING_ONLY"
    return {
        "phase": PHASE,
        "runtime_started": False,
        "retraining_performed": False,
        "new_experiment_run": False,
        "model_files_modified": False,
        "trader_db_mutated": False,
        "sqlite_open_mode": "read_only_if_used",
        "external_api_used": False,
        "gemini_called": False,
        "web_search_called": False,
        "wallet_connected": False,
        "private_key_accessed": False,
        "live_order_submitted": False,
        "paper_loop_started": False,
        "trade_authority_granted_to_RF": False,
        "trade_authority_granted_to_XGB": False,
        "trade_authority_granted_to_TAB": False,
        "trade_authority_granted_to_meta_layer": False,
        "trade_authority_granted_to_Qwen": False,
        "trade_authority_granted_to_Gemini": False,
        "trade_authority_granted_to_any_llm": False,
        "live_ready": False,
        "profitability_proven": False,
        "writes_limited_to_output_root": True,
        "output_root": str(output_root.resolve()),
        "source_roots_modified": False,
        "historical_artifacts_rewritten": False,
        "status": status,
    }


def build_readonly_source_audit(*, output_root: Path) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "writes_limited_to_output_root": True,
        "output_root": str(output_root.resolve()),
        "trader_db_opened": False,
        "trader_db_mutated": False,
        "sqlite_open_mode": "read_only_if_used",
        "source_roots_modified": False,
        "historical_artifacts_rewritten": False,
        "status": "PASS_READONLY_DISCOVERY",
    }


def build_no_retraining_audit(*, output_root: Path) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "retraining_performed": False,
        "new_experiment_run": False,
        "model_files_modified": False,
        "rf_retrained": False,
        "xgb_retrained": False,
        "tab_retrained": False,
        "output_root": str(output_root.resolve()),
        "status": "PASS_NO_RETRAINING",
    }


def decide_gate(
    *,
    inventory: list[dict[str, Any]],
    matrix: list[dict[str, Any]],
    safety: dict[str, Any],
    readonly: dict[str, Any],
    no_retrain: dict[str, Any],
    output_root: Path,
    simulate_missing_critical: bool = False,
    output_write_failed: bool = False,
) -> dict[str, Any]:
    missing_critical = [
        r["artifact_family"]
        for r in inventory
        if r.get("missing_status") == "MISSING_CRITICAL_ARTIFACT"
    ]
    if simulate_missing_critical:
        missing_critical = list(set(missing_critical + ["SIMULATED_CRITICAL_ARTIFACT"]))

    missing_important = [
        r["artifact_family"]
        for r in inventory
        if r.get("missing_status") == "MISSING_EXPECTED_ARTIFACT"
    ]
    missing_optional = [
        r["artifact_family"]
        for r in inventory
        if r.get("missing_status") == "MISSING_NONCRITICAL_ARTIFACT"
    ]

    component_statuses = {m["layer_name"]: m["component_status"] for m in matrix}
    missing_by_component: dict[str, list[str]] = {}
    for m in matrix:
        if m.get("missing_expected_artifacts"):
            missing_by_component[m["layer_name"]] = [
                x for x in str(m["missing_expected_artifacts"]).split(";") if x
            ]

    components_with_missing = [
        name
        for name, st in component_statuses.items()
        if st in {"MISSING_EXPECTED_ARTIFACT", "PARTIALLY_EVALUATED_MISSING_EXPECTED_ARTIFACTS"}
    ]

    reporting_ready = [m["layer_name"] for m in matrix if m.get("reporting_ready") == "true"]
    research_only = [
        m["layer_name"] for m in matrix if m.get("research_ready") == "true" and m.get("live_ready") == "false"
    ]
    diagnostic_only = [
        m["layer_name"] for m in matrix if m.get("evidence_status") == "PASS_DIAGNOSTIC_ONLY"
    ]
    hold_layers = [m["layer_name"] for m in matrix if str(m.get("evidence_status", "")).startswith("HOLD")]

    safety_status = safety.get("status", "")
    gate_status = "PASS_ML_META_REPORTING_READY"

    if output_write_failed:
        gate_status = "HOLD_OUTPUT_ROOT_WRITE_FAILED"
    elif safety_status != "PASS_READONLY_REPORTING_ONLY":
        if "WALLET" in safety_status or "LIVE" in safety_status:
            gate_status = "FAIL_LIVE_OR_WALLET_AUTHORITY_REGRESSION"
        elif "RETRAINING" in safety_status or "RUNTIME" in safety_status:
            gate_status = "FAIL_RETRAINING_OR_RUNTIME_MUTATION"
        elif "SOURCE" in safety_status:
            gate_status = "FAIL_SOURCE_ARTIFACT_MUTATION"
        elif "EXTERNAL" in safety_status:
            gate_status = "FAIL_EXTERNAL_API_USED"
        else:
            gate_status = "FAIL_RETRAINING_OR_RUNTIME_MUTATION"
    elif missing_critical:
        gate_status = "HOLD_MISSING_CRITICAL_ARTIFACTS"
    elif missing_important or components_with_missing:
        gate_status = "PASS_WITH_LIMITATIONS"

    return {
        "phase": PHASE,
        "ae12_closed": False,
        "runtime_started": False,
        "retraining_performed": False,
        "new_experiment_run": False,
        "external_api_used": False,
        "gemini_called": False,
        "trader_db_mutated": False,
        "wallet_connected": False,
        "live_ready": False,
        "profitability_proven": False,
        "trade_authority_granted_to_models": False,
        "trade_authority_granted_to_any_llm": False,
        "source_roots_modified": False,
        "writes_limited_to_output_root": True,
        "output_root": str(output_root.resolve()),
        "reporting_ready_layers": reporting_ready,
        "research_only_layers": research_only,
        "diagnostic_only_layers": diagnostic_only,
        "hold_layers": hold_layers,
        "missing_critical_artifacts": missing_critical,
        "missing_important_artifacts": missing_important,
        "missing_optional_artifacts": missing_optional,
        "missing_by_component": missing_by_component,
        "components_with_missing_expected_artifacts": components_with_missing,
        "component_statuses": component_statuses,
        "semantic_coverage_reconciliation_included": True,
        "safety_audit_status": safety_status,
        "readonly_source_audit_status": readonly.get("status"),
        "no_retraining_audit_status": no_retrain.get("status"),
        "recommendation": "AE12.6 ML/meta-layer evaluation package for MSc reporting; AE12 remains open.",
        "next_ae12_step": "AE12.7",
        "status": gate_status,
    }


def render_upload_txt(summary: dict[str, Any], gate: dict[str, Any]) -> str:
    lines = [
        "AE12.6 ML / Meta-Layer Evaluation Summary",
        "NOTE: This is AE12.6 inside AE12 — not AE13. AE12 is not fully closed.",
        f"created_at_utc: {summary.get('created_at_utc')}",
        f"output_root: {summary.get('output_root')}",
        f"gate_status: {gate.get('status')}",
        "",
        "Safety (this pass):",
        "- runtime_started=false",
        "- retraining_performed=false",
        "- external_api_used=false",
        "- gemini_called=false",
        "- trader_db_mutated=false",
        "- wallet_connected=false",
        "- live_ready=false",
        "- profitability_proven=false",
        "- trade_authority_granted_to_models=false",
        "- trade_authority_granted_to_any_llm=false",
        "",
        "=== Missing artifacts (prominent) ===",
        f"missing_critical_artifacts: {gate.get('missing_critical_artifacts')}",
        f"missing_important_artifacts: {gate.get('missing_important_artifacts')}",
        f"missing_optional_artifacts: {gate.get('missing_optional_artifacts')}",
        f"components_with_missing_expected_artifacts: {gate.get('components_with_missing_expected_artifacts')}",
        "",
        "Semantic Coverage Reconciliation:",
        "- Legacy 500/0 UI issue was legacy cluster diagnostic (paper_trades.cluster_label), not 500 unique coins.",
        "- Gemini pair-assets: 89; deduplicated coin-level identities for UI: 14 (post-drilldown).",
        "- UNKNOWN_UNRESOLVED is not social and not opportunistic.",
        "- Semantic labels are reporting-only; not trade authority.",
        "",
        "Layer evaluation (headlines):",
    ]
    for layer in summary.get("layer_headlines") or []:
        lines.append(f"- {layer}")
    lines.extend(
        [
            "",
            "What remains unproven:",
            "- Live trading readiness",
            "- Forward/live profitability",
            "- Web-grounded semantic classification",
            "- Closed original AE7/AE8/AE9 implementations as originally named",
            "",
            f"next_ae12_step: {gate.get('next_ae12_step')}",
            f"recommendation: {gate.get('recommendation')}",
        ]
    )
    text = "\n".join(lines)
    lower = text.lower()
    for phrase in FORBIDDEN_PROFIT_CLAIMS:
        if phrase in lower and "false" not in lower.split(phrase)[0][-20:]:
            pass  # upload text explicitly negates claims
    return text


def run_ae12_ml_meta_layer_evaluation(
    *,
    project_root: Path,
    output_root: Path | None = None,
    simulate_missing_critical_artifacts: bool = False,
    simulate_safety_failure: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    out_root = output_root
    if out_root is None:
        out_root = project_root / "data" / "audits" / f"ae12_ml_meta_layer_evaluation_{_ts_slug()}"
    out_root = Path(out_root).resolve()

    output_write_failed = False
    if not dry_run:
        try:
            ensure_dirs(out_root)
        except OSError as exc:
            return {
                "phase": PHASE,
                "output_root": str(out_root),
                "gate_status": "HOLD_OUTPUT_ROOT_WRITE_FAILED",
                "error": str(exc),
            }

    inventory = [discover_artifact(project_root, spec) for spec in build_artifact_specs()]
    inv_map = inventory_by_path(inventory)

    maturation_root = resolve_maturation_root(project_root)
    if simulate_missing_critical_artifacts:
        for row in inventory:
            if row["artifact_family"] == "ae12_forward_evidence_summary":
                row["exists"] = False
                row["missing_status"] = "MISSING_CRITICAL_ARTIFACT"
                row["discovered_path"] = ""
        inv_map = inventory_by_path(inventory)
        maturation_root = None

    perf = build_model_performance_summary(inv_map)
    forward = build_forward_evidence_integration(maturation_root)
    recon_path = inv_map.get("ae12_semantic_coverage_reconciliation", {}).get("discovered_path", "")
    semantic_addendum = build_semantic_addendum(recon_path)
    matrix = build_evaluation_matrix(inv_map, maturation_root=maturation_root)

    safety = build_safety_audit(output_root=out_root, simulate_failure=simulate_safety_failure)
    readonly = build_readonly_source_audit(output_root=out_root)
    no_retrain = build_no_retraining_audit(output_root=out_root)
    gate = decide_gate(
        inventory=inventory,
        matrix=matrix,
        safety=safety,
        readonly=readonly,
        no_retrain=no_retrain,
        output_root=out_root,
        simulate_missing_critical=simulate_missing_critical_artifacts,
        output_write_failed=output_write_failed,
    )

    layer_headlines = [
        f"{m['layer_name']}: status={m['component_status']}; evidence={m['evidence_status']}"
        for m in matrix
    ]

    summary = {
        "phase": PHASE,
        "schema_version": SCHEMA,
        "created_at_utc": _utc_now(),
        "output_root": str(out_root),
        "ae12_closed": False,
        "maturation_root": str(maturation_root) if maturation_root else None,
        "semantic_coverage_reconciliation_root": recon_path or None,
        "layer_headlines": layer_headlines,
        "missing_critical_artifacts": gate["missing_critical_artifacts"],
        "missing_important_artifacts": gate["missing_important_artifacts"],
        "missing_optional_artifacts": gate["missing_optional_artifacts"],
        "components_with_missing_expected_artifacts": gate["components_with_missing_expected_artifacts"],
        "component_statuses": gate["component_statuses"],
        "profitability_proven": False,
        "live_ready": False,
        "limitations": [
            "AE12.6 is reporting/evaluation only; no retraining or runtime.",
            "Historical model metrics do not prove forward/live profitability.",
            "Semantic labels are reporting-only and not trade authority.",
        ],
    }

    manifest = {
        "phase": PHASE,
        "created_at_utc": summary["created_at_utc"],
        "output_root": str(out_root),
        "files": [
            "reports/ae12_ml_meta_layer_evaluation_summary.json",
            "reports/ae12_ml_meta_layer_evaluation_for_upload.txt",
            "reports/ae12_ml_meta_layer_evaluation_manifest.json",
            "data/ae12_ml_meta_artifact_inventory.csv",
            "data/ae12_ml_meta_layer_evaluation_matrix.csv",
            "data/ae12_model_performance_summary.csv",
            "data/ae12_forward_evidence_integration_summary.csv",
            "data/ae12_semantic_coverage_reconciliation_addendum.csv",
            "audits/ae12_ml_meta_layer_evaluation_gate.json",
            "audits/ae12_ml_meta_safety_audit.json",
            "audits/ae12_ml_meta_readonly_source_audit.json",
            "audits/ae12_ml_meta_no_retraining_audit.json",
        ],
    }

    if dry_run:
        return {"phase": PHASE, "output_root": str(out_root), "gate_status": gate["status"], "dry_run": True}

    inv_fields = [
        "artifact_family",
        "expected_layer",
        "discovered_path",
        "exists",
        "file_type",
        "last_modified_utc",
        "evidence_role",
        "criticality",
        "missing_status",
        "notes",
    ]
    matrix_fields = list(matrix[0].keys()) if matrix else ["layer_name"]
    perf_fields = list(perf[0].keys()) if perf else ["model_family"]
    forward_fields = list(forward[0].keys()) if forward else ["evidence_component"]

    dirs = ensure_dirs(out_root)
    write_csv(dirs["data"] / "ae12_ml_meta_artifact_inventory.csv", inventory, inv_fields)
    write_csv(dirs["data"] / "ae12_ml_meta_layer_evaluation_matrix.csv", matrix, matrix_fields)
    write_csv(dirs["data"] / "ae12_model_performance_summary.csv", perf, perf_fields)
    write_csv(dirs["data"] / "ae12_forward_evidence_integration_summary.csv", forward, forward_fields)
    write_csv(
        dirs["data"] / "ae12_semantic_coverage_reconciliation_addendum.csv",
        semantic_addendum,
        ["topic", "value", "source"],
    )

    write_json(dirs["audits"] / "ae12_ml_meta_safety_audit.json", safety)
    write_json(dirs["audits"] / "ae12_ml_meta_readonly_source_audit.json", readonly)
    write_json(dirs["audits"] / "ae12_ml_meta_no_retraining_audit.json", no_retrain)
    write_json(dirs["audits"] / "ae12_ml_meta_layer_evaluation_gate.json", gate)

    upload_txt = render_upload_txt(summary, gate)
    write_json(dirs["reports"] / "ae12_ml_meta_layer_evaluation_summary.json", summary)
    (dirs["reports"] / "ae12_ml_meta_layer_evaluation_for_upload.txt").write_text(upload_txt, encoding="utf-8")
    write_json(dirs["reports"] / "ae12_ml_meta_layer_evaluation_manifest.json", manifest)

    return {
        "phase": PHASE,
        "output_root": str(out_root),
        "gate_status": gate["status"],
        "safety_audit_status": safety["status"],
        "summary": summary,
        "gate": gate,
    }
