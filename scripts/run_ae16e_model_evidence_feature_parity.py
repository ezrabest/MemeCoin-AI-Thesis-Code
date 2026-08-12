#!/usr/bin/env python3
"""AE16E — Direct-target model evidence attachment + strict feature parity recheck.

Uses existing RF/XGB/TAB artifacts only. No training, backtest, trader.db mutation,
wallet connection, or live trading. Does not start AE17.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.consensus.ae16e_feature_parity import (  # noqa: E402
    PHASE,
    TOXIC_PAIR_ADDRESS,
    assert_no_toxic_in_outputs,
    audit_feature_parity_ae16e,
    build_ae16e_consensus,
    build_feature_matrix_for_family,
    build_model_evidence,
    decide_ae16e_classification,
    discover_ae16e_artifacts,
    load_clean_forward_rows_used,
    run_inference_if_allowed,
    utc_now,
    utc_stamp,
)
from app.consensus.serialization import (  # noqa: E402
    relpath_str,
    write_csv,
    write_json,
    write_text,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AE16E model evidence + feature parity")
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument(
        "--active-curated",
        type=str,
        default="data/SeedTargets/clean_forward_curated_ready_targets_active.csv",
    )
    p.add_argument("--ae16d-rows", type=str, default=None)
    p.add_argument("--skip-inference", action="store_true")
    return p.parse_args(argv)


def _empty_matrix_headers() -> list[str]:
    return [
        "row_id",
        "combined_target_id",
        "chain",
        "pair_address",
        "provider_pair_url",
        "base_token_address",
        "quote_token_address",
        "base_token_symbol",
        "quote_token_symbol",
        "feature_parity_status",
        "no_lookahead_status",
        "model_family",
    ]


def run_ae16e(args: argparse.Namespace) -> dict[str, Any]:
    stamp = utc_stamp()
    out_root = Path(args.output_root) if args.output_root else (
        ROOT / "data" / "audits" / f"ae16e_model_evidence_feature_parity_{stamp}"
    )
    if not out_root.is_absolute():
        out_root = ROOT / out_root
    data_dir = out_root / "data"
    audits = out_root / "audits"
    reports = out_root / "reports"
    for d in (data_dir, audits, reports):
        d.mkdir(parents=True, exist_ok=True)

    discovery_crashed = False
    discovery_error = ""
    try:
        discovery_rows, extraction_audit, discovery_manifest = discover_ae16e_artifacts(
            ROOT,
            exclude_roots=(relpath_str(out_root, ROOT),),
        )
    except Exception as exc:  # noqa: BLE001
        discovery_crashed = True
        discovery_error = f"{type(exc).__name__}: {exc}"
        discovery_rows, extraction_audit, discovery_manifest = [], [], {"error": discovery_error}

    selection = (discovery_manifest.get("selection") or {}) if discovery_manifest else {}

    active_path = Path(args.active_curated)
    ae16d_path = Path(args.ae16d_rows) if args.ae16d_rows else None
    rows, rows_meta = load_clean_forward_rows_used(
        ROOT,
        active_curated_path=active_path,
        ae16d_rows_path=ae16d_path,
    )
    write_csv(data_dir / "ae16e_clean_forward_rows_used.csv", rows)
    write_csv(data_dir / "ae16e_artifact_discovery.csv", discovery_rows)
    write_csv(audits / "ae16e_artifact_feature_name_extraction_audit.csv", extraction_audit)

    controlled_history = False  # AE16D provides single current snapshot only
    parity = audit_feature_parity_ae16e(
        project_root=ROOT,
        rows=rows,
        selection=selection,
        controlled_snapshot_history=controlled_history,
    )
    fam_rows = parity.get("by_family_rows") or {}
    fam_sum = parity.get("by_family_summary") or {}
    for family in ("RF", "XGB", "TAB"):
        write_csv(data_dir / f"ae16e_feature_parity_{family.lower()}.csv", fam_rows.get(family) or [])

    write_csv(audits / "ae16e_sequential_feature_questionnaire.csv", parity.get("sequential_questionnaire") or [])
    write_csv(audits / "ae16e_no_lookahead_audit.csv", parity.get("derived_feature_audit") or [])

    # Feature matrices — only non-empty when parity allows inference
    matrices: dict[str, list[dict[str, Any]]] = {}
    inference_by: dict[str, list[dict[str, Any]]] = {}
    for family in ("RF", "XGB", "TAB"):
        info = fam_sum.get(family) or {}
        sel = selection.get(family) or {}
        required = list(sel.get("feature_names") or [])
        matrix = build_feature_matrix_for_family(
            rows=rows,
            family=family,
            required=required,
            parity_rows=fam_rows.get(family) or [],
            inference_allowed=bool(info.get("inference_allowed")),
        )
        matrices[family] = matrix
        out_path = data_dir / f"ae16e_clean_forward_model_features_{family.lower()}.csv"
        if matrix:
            write_csv(out_path, matrix)
        else:
            write_csv(out_path, [], fieldnames=_empty_matrix_headers())

        if info.get("inference_allowed") and matrix and not args.skip_inference:
            scores, err = run_inference_if_allowed(
                project_root=ROOT,
                family=family,
                model_path=str(sel.get("model_path") or ""),
                matrix=matrix,
                required=required,
            )
            if err:
                info["inference_error"] = err
                info["inference_allowed"] = False
                fam_sum[family] = info
            else:
                inference_by[family] = scores

    evidence, unavailable = build_model_evidence(
        rows=rows,
        parity_summary=parity,
        selection=selection,
        inference_by_family=inference_by,
    )
    write_csv(data_dir / "ae16e_model_evidence.csv", evidence)
    write_csv(data_dir / "ae16e_model_evidence_unavailable.csv", unavailable)

    consensus_rows, tier_counts = build_ae16e_consensus(rows, evidence)
    write_csv(data_dir / "ae16e_tiered_consensus_rows.csv", consensus_rows)
    write_csv(data_dir / "ae16e_tier_counts.csv", tier_counts)

    toxic_in_outputs = assert_no_toxic_in_outputs(
        [rows, evidence, consensus_rows, matrices.get("RF") or [], matrices.get("XGB") or [], matrices.get("TAB") or []]
    )

    # Lineage audit
    lineage = []
    for row in rows:
        lineage.append(
            {
                "row_id": row.get("row_id"),
                "combined_target_id": row.get("combined_target_id"),
                "chain": row.get("chain"),
                "pair_address": row.get("pair_address"),
                "provider_pair_url": row.get("provider_pair_url"),
                "base_token_address": row.get("base_token_address"),
                "quote_token_address": row.get("quote_token_address"),
                "target_source": row.get("target_source"),
                "semantic_status": row.get("semantic_status"),
                "source_ae16d_path": row.get("source_ae16d_path"),
                "source_active_curated_path": row.get("source_active_curated_path"),
            }
        )
    write_csv(audits / "ae16e_lineage_audit.csv", lineage)

    write_json(
        audits / "ae16e_toxic_pair_exclusion_audit.json",
        {
            "toxic_pair_address": TOXIC_PAIR_ADDRESS,
            "toxic_pair_present_in_used_rows": toxic_in_outputs,
            "excluded_from_ae16d_source_count": rows_meta.get("excluded_toxic_count"),
            "rows_used": len(rows),
            "pass": not toxic_in_outputs,
        },
    )
    write_json(
        audits / "ae16e_no_trader_db_mutation_audit.json",
        {
            "trader_db_mutated": False,
            "operations": "read_only_csv_artifact_inspection_inference_optional",
            "pass": True,
        },
    )
    write_json(
        audits / "ae16e_no_live_wallet_audit.json",
        {
            "wallet_connected": False,
            "live_trading_enabled": False,
            "live_trading_ready": False,
            "paper_demo_only": True,
            "trade_authority": False,
            "pass": True,
        },
    )
    write_json(
        audits / "ae16e_no_training_backtest_audit.json",
        {
            "model_training_run": False,
            "backtest_run": False,
            "ae17_started": False,
            "synthetic_labels": False,
            "pass": True,
        },
    )

    attached_rf = sum(1 for e in evidence if e.get("model_family") == "RF")
    attached_xgb = sum(1 for e in evidence if e.get("model_family") == "XGB")
    attached_tab = sum(1 for e in evidence if e.get("model_family") == "TAB")
    tier_map = {str(r.get("consensus_tier")): int(r.get("count") or 0) for r in tier_counts}
    me_unavail = tier_map.get("MODEL_EVIDENCE_UNAVAILABLE", 0)

    classification = decide_ae16e_classification(
        rows_meta=rows_meta,
        parity=parity,
        evidence=evidence,
        discovery_crashed=discovery_crashed,
        toxic_in_outputs=toxic_in_outputs,
    )
    ae16_e6_closed = classification == "AE16E_MODEL_EVIDENCE_ATTACHMENT_PASS"

    write_json(
        audits / "ae16e_consensus_validity_audit.json",
        {
            "consensus_rows": len(consensus_rows),
            "tier_counts": tier_map,
            "evidence_rows": len(evidence),
            "families_attached": sorted(
                {e["model_family"] for e in evidence if e.get("evidence_status") == "MODEL_EVIDENCE_ATTACHED"}
            ),
            "meaningful_tiers_require_attached_votes": True,
            "model_evidence_unavailable_count": me_unavail,
            "tab_xgb_rf_all3_exists": tier_map.get("TAB_XGB_RF_ALL3", 0) > 0,
            "tab_rf_only_exists": tier_map.get("TAB_RF_ONLY", 0) > 0,
            "ae16_original_e6_closed": ae16_e6_closed,
            "classification": classification,
            "pass": classification
            in {
                "AE16E_MODEL_EVIDENCE_ATTACHMENT_PASS",
                "AE16E_PARTIAL_MODEL_EVIDENCE_ATTACHMENT_PASS",
            }
            or (
                classification == "AE16E_BLOCKED_FEATURE_PARITY_GAP"
                and me_unavail == len(consensus_rows)
                and len(evidence) == 0
            ),
        },
    )

    seq_q = parity.get("sequential_questionnaire") or []
    seq_blocking = [r for r in seq_q if r.get("final_classification") == "MISSING_BLOCKING"]

    manifest = {
        "phase": PHASE,
        "timestamp": stamp,
        "generated_at_utc": utc_now(),
        "output_root": relpath_str(out_root, ROOT),
        "clean_forward_rows_used": len(rows),
        "curated_active_targets_loaded": rows_meta.get("curated_active_targets_loaded"),
        "valid_provider_pairs": rows_meta.get("valid_provider_pairs"),
        "toxic_pair_present_anywhere": toxic_in_outputs,
        "rf_artifacts_discovered": sum(1 for r in discovery_rows if r.get("model_family") == "RF" or r.get("artifact_family") == "RF"),
        "xgb_artifacts_discovered": sum(1 for r in discovery_rows if r.get("model_family") == "XGB" or r.get("artifact_family") == "XGB"),
        "tab_artifacts_discovered": sum(1 for r in discovery_rows if r.get("model_family") == "TAB" or r.get("artifact_family") == "TAB"),
        "rf_artifact_selected": (selection.get("RF") or {}).get("model_path") or "",
        "xgb_artifact_selected": (selection.get("XGB") or {}).get("model_path") or "",
        "tab_artifact_selected": (selection.get("TAB") or {}).get("schema_path") or "",
        "rf_artifact_rejected_reason": (selection.get("RF") or {}).get("rejected_reason") or "",
        "xgb_artifact_rejected_reason": (selection.get("XGB") or {}).get("rejected_reason") or "",
        "tab_artifact_rejected_reason": (selection.get("TAB") or {}).get("rejected_reason") or "",
        "rf_feature_names_extraction_status": (fam_sum.get("RF") or {}).get("feature_names_extraction_status"),
        "xgb_feature_names_extraction_status": (fam_sum.get("XGB") or {}).get("feature_names_extraction_status"),
        "tab_feature_names_extraction_status": (fam_sum.get("TAB") or {}).get("feature_names_extraction_status")
        or (selection.get("TAB") or {}).get("feature_names_extraction_status"),
        "feature_name_extraction_errors_count": discovery_manifest.get("extraction_error_count", 0)
        if discovery_manifest
        else 0,
        "rf_required_features_count": (fam_sum.get("RF") or {}).get("required_features_count"),
        "xgb_required_features_count": (fam_sum.get("XGB") or {}).get("required_features_count"),
        "tab_required_features_count": (fam_sum.get("TAB") or {}).get("required_features_count"),
        "rf_missing_blocking_count": (fam_sum.get("RF") or {}).get("missing_blocking_count"),
        "xgb_missing_blocking_count": (fam_sum.get("XGB") or {}).get("missing_blocking_count"),
        "tab_missing_blocking_count": (fam_sum.get("TAB") or {}).get("missing_blocking_count"),
        "sequential_features_detected_count": len(seq_q),
        "sequential_features_available_from_controlled_history_count": 0,
        "sequential_features_missing_blocking_count": len(seq_blocking),
        "controlled_snapshot_history_used": False,
        "legacy_market_snapshots_used": False,
        "unsafe_pair_timestamp_join_used": False,
        "rf_evidence_rows_attached": attached_rf,
        "xgb_evidence_rows_attached": attached_xgb,
        "tab_evidence_rows_attached": attached_tab,
        "consensus_rows_produced": len(consensus_rows),
        "tier_counts": tier_map,
        "tab_xgb_rf_all3_count": tier_map.get("TAB_XGB_RF_ALL3", 0),
        "tab_rf_only_count": tier_map.get("TAB_RF_ONLY", 0),
        "model_evidence_unavailable_count": me_unavail,
        "trader_db_mutated": False,
        "wallet_connected": False,
        "live_trading_enabled": False,
        "model_training_run": False,
        "backtest_run": False,
        "ae17_started": False,
        "ae16_original_e6_closed": ae16_e6_closed,
        "classification": classification,
        "rows_meta": rows_meta,
        "parity_by_family": fam_sum,
        "discovery_error": discovery_error,
    }
    write_json(reports / "ae16e_manifest.json", manifest)

    gate = {
        "phase": PHASE,
        "classification": classification,
        "status": "PASS"
        if classification
        in {
            "AE16E_MODEL_EVIDENCE_ATTACHMENT_PASS",
            "AE16E_PARTIAL_MODEL_EVIDENCE_ATTACHMENT_PASS",
        }
        else "BLOCKED",
        "ae16_original_e6_closed": ae16_e6_closed,
        "safe_to_start_ae17": False,
        "blockers": [],
        "rf_missing_blocking": (fam_sum.get("RF") or {}).get("missing_blocking_features"),
        "xgb_missing_blocking": (fam_sum.get("XGB") or {}).get("missing_blocking_features"),
        "tab_blocker": (fam_sum.get("TAB") or {}).get("blocker_reason")
        or (selection.get("TAB") or {}).get("rejected_reason"),
    }
    if classification != "AE16E_MODEL_EVIDENCE_ATTACHMENT_PASS":
        gate["blockers"].append(classification)
        if (fam_sum.get("RF") or {}).get("blocker_reason"):
            gate["blockers"].append((fam_sum.get("RF") or {}).get("blocker_reason"))
        if (selection.get("TAB") or {}).get("rejected_reason"):
            gate["blockers"].append((selection.get("TAB") or {}).get("rejected_reason"))
    write_json(reports / "ae16e_decision_gate.json", gate)

    summary_lines = [
        f"phase: {PHASE}",
        f"final_classification: {classification}",
        f"output_root: {relpath_str(out_root, ROOT)}",
        f"clean_forward_rows_used: {len(rows)}",
        f"curated_active_targets_loaded: {rows_meta.get('curated_active_targets_loaded')}",
        f"toxic_pair_excluded: {not toxic_in_outputs}",
        f"rf_artifact_selected: {(selection.get('RF') or {}).get('model_path') or (selection.get('RF') or {}).get('rejected_reason')}",
        f"xgb_artifact_selected: {(selection.get('XGB') or {}).get('model_path') or (selection.get('XGB') or {}).get('rejected_reason')}",
        f"tab_artifact_selected: {(selection.get('TAB') or {}).get('rejected_reason')}",
        f"rf_feature_names_extraction_status: {(fam_sum.get('RF') or {}).get('feature_names_extraction_status')}",
        f"xgb_feature_names_extraction_status: {(fam_sum.get('XGB') or {}).get('feature_names_extraction_status')}",
        f"tab_feature_names_extraction_status: {(selection.get('TAB') or {}).get('feature_names_extraction_status')}",
        f"rf_missing_blocking_count: {(fam_sum.get('RF') or {}).get('missing_blocking_count')}",
        f"xgb_missing_blocking_count: {(fam_sum.get('XGB') or {}).get('missing_blocking_count')}",
        f"tab_missing_blocking_count: {(fam_sum.get('TAB') or {}).get('missing_blocking_count')}",
        f"sequential_features_detected: {len(seq_q)}",
        f"sequential_features_blocked: {len(seq_blocking)}",
        f"controlled_snapshot_history_used: false",
        f"rf_evidence_rows_attached: {attached_rf}",
        f"xgb_evidence_rows_attached: {attached_xgb}",
        f"tab_evidence_rows_attached: {attached_tab}",
        f"consensus_rows_produced: {len(consensus_rows)}",
        f"tier_counts: {tier_map}",
        f"TAB_XGB_RF_ALL3_exists: {tier_map.get('TAB_XGB_RF_ALL3', 0) > 0}",
        f"TAB_RF_ONLY_exists: {tier_map.get('TAB_RF_ONLY', 0) > 0}",
        f"ae16_original_e6_closed: {ae16_e6_closed}",
        "trader_db_mutated: false",
        "wallet_connected: false",
        "live_trading_enabled: false",
        "model_training_run: false",
        "backtest_run: false",
        "ae17_started: false",
    ]
    write_text(reports / "ae16e_summary_for_upload.txt", "\n".join(summary_lines) + "\n")

    # Mirror key reports to repo-level reports/ for upload convenience (read-only copies)
    repo_reports = ROOT / "reports"
    repo_reports.mkdir(parents=True, exist_ok=True)
    for name in ("ae16e_manifest.json", "ae16e_summary_for_upload.txt", "ae16e_decision_gate.json"):
        shutil.copy2(reports / name, repo_reports / name)

    # Mirror data CSVs to repo data/ root as required deliverable paths
    repo_data = ROOT / "data"
    for name in (
        "ae16e_clean_forward_rows_used.csv",
        "ae16e_artifact_discovery.csv",
        "ae16e_feature_parity_rf.csv",
        "ae16e_feature_parity_xgb.csv",
        "ae16e_feature_parity_tab.csv",
        "ae16e_clean_forward_model_features_rf.csv",
        "ae16e_clean_forward_model_features_xgb.csv",
        "ae16e_clean_forward_model_features_tab.csv",
        "ae16e_model_evidence.csv",
        "ae16e_model_evidence_unavailable.csv",
        "ae16e_tiered_consensus_rows.csv",
        "ae16e_tier_counts.csv",
    ):
        src = data_dir / name
        if src.is_file():
            shutil.copy2(src, repo_data / name)

    repo_audits = ROOT / "audits"
    repo_audits.mkdir(parents=True, exist_ok=True)
    for name in (
        "ae16e_no_lookahead_audit.csv",
        "ae16e_lineage_audit.csv",
        "ae16e_toxic_pair_exclusion_audit.json",
        "ae16e_artifact_feature_name_extraction_audit.csv",
        "ae16e_sequential_feature_questionnaire.csv",
        "ae16e_no_trader_db_mutation_audit.json",
        "ae16e_no_live_wallet_audit.json",
        "ae16e_no_training_backtest_audit.json",
        "ae16e_consensus_validity_audit.json",
    ):
        src = audits / name
        if src.is_file():
            shutil.copy2(src, repo_audits / name)

    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = run_ae16e(args)
    print(f"classification={manifest.get('classification')}")
    print(f"output_root={manifest.get('output_root')}")
    print(f"clean_forward_rows_used={manifest.get('clean_forward_rows_used')}")
    print(f"ae16_original_e6_closed={manifest.get('ae16_original_e6_closed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
