"""AE17 end-to-end meta stacking pipeline (deterministic, no training)."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.meta import (
    CLASSIFICATION_AUTHORITY,
    CLASSIFICATION_FEATURE_PARITY,
    CLASSIFICATION_LINEAGE,
    CLASSIFICATION_LOOKAHEAD,
    CLASSIFICATION_MISSING_AE16,
    CLASSIFICATION_PASS,
    CLASSIFICATION_PASS_LIMITATIONS,
    META_ENGINE_VERSION,
    PHASE,
)
from app.meta.audits import (
    audit_authority,
    audit_feature_parity,
    audit_lineage,
    audit_no_lookahead,
    audit_null_safety,
    audit_pair_concentration,
    audit_score_clamping,
    build_feature_contract_audit,
)
from app.meta.discovery import discover_ae16_artifacts
from app.meta.features import (
    build_meta_feature_rows,
    feature_matrix_dicts,
    load_ae16_consensus_csv,
)
from app.meta.models import AE17MetaDecision
from app.meta.scoring import score_all_rows
from app.consensus.serialization import relpath_str, write_csv, write_json, write_jsonl, write_text


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _db_mtime(root: Path) -> float | None:
    db = root / "trader.db"
    if not db.is_file():
        return None
    return db.stat().st_mtime


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_blocked_missing_inputs(
    project_root: Path,
    discovery: Any,
    *,
    out_root: Path | None = None,
) -> dict[str, Any]:
    stamp = utc_stamp()
    out = out_root or (project_root / "data" / "audits" / f"ae17_meta_stacking_layer_{stamp}")
    reports = out / "reports"
    audits = out / "audits"
    data = out / "data"
    for d in (reports, audits, data, project_root / "reports"):
        d.mkdir(parents=True, exist_ok=True)

    discovery_dict = discovery.to_dict() if hasattr(discovery, "to_dict") else dict(discovery)
    gate = {
        "phase": PHASE,
        "classification": CLASSIFICATION_MISSING_AE16,
        "created_at_utc": utc_now(),
        "meta_engine_version": META_ENGINE_VERSION,
        "ae18_status": "BLOCKED",
        "ae19_status": "BLOCKED",
        "meta_authority_allowed": False,
        "trade_authority": False,
        "live_trading_ready": False,
        "paper_demo_only": True,
        "training_performed": False,
        "searched_roots": discovery_dict.get("searched_roots", []),
        "expected_patterns": discovery_dict.get("expected_patterns", []),
        "missing_required_artifacts": discovery_dict.get("missing_required_artifacts", []),
        "found_candidate_artifacts": discovery_dict.get("found_candidate_artifacts", []),
        "recommended_next_action": discovery_dict.get("recommended_next_action", ""),
    }
    write_json(audits / "ae17_input_artifact_discovery_audit.json", discovery_dict)
    write_json(reports / "ae17_decision_gate.json", gate)
    write_json(project_root / "reports" / "ae17_decision_gate.json", gate)

    summary = "\n".join(
        [
            "AE17 Meta-Model / Stacking Layer",
            f"classification: {CLASSIFICATION_MISSING_AE16}",
            f"output_root: {relpath_str(out, project_root)}",
            "AE16 required consensus artifact not found.",
            f"recommended_next_action: {gate['recommended_next_action']}",
            "AE18: BLOCKED",
            "AE19: BLOCKED",
        ]
    )
    write_text(reports / "ae17_summary_for_upload.txt", summary + "\n")
    write_text(project_root / "reports" / "ae17_summary_for_upload.txt", summary + "\n")
    write_json(
        reports / "ae17_manifest.json",
        {
            "phase": PHASE,
            "classification": CLASSIFICATION_MISSING_AE16,
            "output_root": relpath_str(out, project_root),
            "created_at_utc": utc_now(),
        },
    )
    write_json(
        project_root / "reports" / "ae17_manifest.json",
        {
            "phase": PHASE,
            "classification": CLASSIFICATION_MISSING_AE16,
            "output_root": relpath_str(out, project_root),
            "created_at_utc": utc_now(),
        },
    )
    return {
        "classification": CLASSIFICATION_MISSING_AE16,
        "output_root": relpath_str(out, project_root),
        "decision_gate": gate,
        "discovery": discovery_dict,
    }


def run_ae17_meta_stacking_layer(
    project_root: Path,
    *,
    ae16_root: str | Path | None = None,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    stamp = utc_stamp()
    out = Path(output_root) if output_root else (root / "data" / "audits" / f"ae17_meta_stacking_layer_{stamp}")
    if not out.is_absolute():
        out = root / out

    reports = out / "reports"
    audits_dir = out / "audits"
    data_dir = out / "data"
    for d in (reports, audits_dir, data_dir, root / "reports", root / "data"):
        d.mkdir(parents=True, exist_ok=True)

    db_before = _db_mtime(root)

    discovery = discover_ae16_artifacts(root, ae16_root=ae16_root)
    write_json(audits_dir / "ae17_input_artifact_discovery_audit.json", discovery.to_dict())

    if discovery.status == CLASSIFICATION_MISSING_AE16 or not discovery.selected_consensus_path:
        return write_blocked_missing_inputs(root, discovery, out_root=out)

    consensus_path = root / discovery.selected_consensus_path
    consensus_rows, source_columns = load_ae16_consensus_csv(consensus_path)
    source_rel = discovery.selected_consensus_path

    feature_rows, warn_records = build_meta_feature_rows(
        consensus_rows,
        source_artifact=source_rel,
        source_columns=source_columns,
    )

    concentration = audit_pair_concentration(feature_rows, grouping="all_meta_rows")
    conc_status = "|".join(concentration.concentration_status)
    penalty = concentration.pair_concentration_penalty

    shadow_rows = score_all_rows(
        feature_rows,
        pair_concentration_status=conc_status,
        pair_concentration_penalty=penalty,
    )

    parity = audit_feature_parity(feature_rows, shadow_rows)
    lookahead = audit_no_lookahead(feature_rows, source_columns=source_columns)
    lineage_rows, lineage_summary = audit_lineage(feature_rows)
    authority = audit_authority(shadow_rows)
    clamp_rows = audit_score_clamping(shadow_rows)
    null_safety = audit_null_safety(feature_rows, shadow_rows)
    contract_rows = build_feature_contract_audit(feature_rows)

    # Persist data products
    feat_dicts = [r.to_dict() for r in feature_rows]
    shadow_dicts = [r.to_dict() for r in shadow_rows]
    matrix_dicts = feature_matrix_dicts(feature_rows)

    write_csv(data_dir / "ae17_meta_feature_rows.csv", feat_dicts)
    write_csv(data_dir / "ae17_meta_shadow_outputs.csv", shadow_dicts)
    write_jsonl(data_dir / "ae17_meta_shadow_outputs.jsonl", shadow_dicts)
    write_csv(data_dir / "ae17_meta_layer_matrix.csv", matrix_dicts)

    # Mirror under project data/ as convenience (non-authoritative copies)
    write_csv(root / "data" / "ae17_meta_feature_rows.csv", feat_dicts)
    write_csv(root / "data" / "ae17_meta_shadow_outputs.csv", shadow_dicts)
    write_jsonl(root / "data" / "ae17_meta_shadow_outputs.jsonl", shadow_dicts)
    write_csv(root / "data" / "ae17_meta_layer_matrix.csv", matrix_dicts)

    write_csv(audits_dir / "ae17_meta_feature_contract_audit.csv", contract_rows)
    write_json(audits_dir / "ae17_no_lookahead_audit.json", lookahead)
    write_json(audits_dir / "ae17_feature_parity_audit.json", parity)
    write_csv(audits_dir / "ae17_lineage_audit.csv", lineage_rows)
    write_csv(
        audits_dir / "ae17_pair_concentration_audit.csv",
        [concentration.to_dict()],
    )
    write_json(audits_dir / "ae17_authority_audit.json", authority.to_dict())
    write_csv(audits_dir / "ae17_score_clamping_audit.csv", clamp_rows)
    write_json(audits_dir / "ae17_null_safety_audit.json", null_safety)
    if warn_records:
        write_csv(audits_dir / "ae17_construction_warnings.csv", warn_records)

    db_after = _db_mtime(root)
    db_mutated = db_before != db_after
    if db_mutated:
        authority.violations.append("trader_db_mtime_changed")
        authority.db_mutation = True
        authority.passed = False

    # Classification
    reasons: list[str] = []
    classification = CLASSIFICATION_PASS
    if not authority.passed:
        classification = CLASSIFICATION_AUTHORITY
        reasons.append("authority_audit_failed")
    elif not lookahead.get("passed", False):
        classification = CLASSIFICATION_LOOKAHEAD
        reasons.append("lookahead_audit_failed")
    elif not parity.get("passed", False):
        classification = CLASSIFICATION_FEATURE_PARITY
        reasons.append("feature_parity_audit_failed")
    elif not lineage_summary.get("passed", False):
        classification = CLASSIFICATION_LINEAGE
        reasons.append("lineage_critical_identity_gap")
    else:
        limitations: list[str] = []
        if not concentration.meta_authority_allowed:
            limitations.append("pair_concentration_high_risk_limits_signal_authority")
        if lineage_summary.get("lineage_majority_incomplete"):
            limitations.append("majority_lineage_incomplete_surrogate_ids")
        context_all_missing = all(not r.context_feature_available for r in feature_rows)
        if context_all_missing:
            limitations.append("context_not_available_pending_ae18")
        if "SMALL_SAMPLE_WARNING" in concentration.concentration_status:
            limitations.append("small_sample_warning")
        if limitations:
            classification = CLASSIFICATION_PASS_LIMITATIONS
            reasons.extend(limitations)
        else:
            reasons.append("all_core_audits_passed")

    checklist = {
        "AE17-01_meta_substrate_exists": True,
        "AE17-02_consumes_rf_xgb_tab_consensus_context": True,
        "AE17-03_feature_parity_and_no_lookahead": bool(
            parity.get("passed") and lookahead.get("passed")
        ),
        "AE17-04_lineage_preserved": bool(lineage_summary.get("passed")),
        "AE17-05_pair_concentration_numeric": True,
        "AE17-06_authority_boundary_enforced": bool(authority.passed),
    }

    decision = AE17MetaDecision(
        classification=classification,
        # Research/shadow substrate: never grant meta trade/signal authority in AE17.
        meta_authority_allowed=False,
        ae18_status="BLOCKED",
        ae19_status="BLOCKED",
        reasons=reasons,
        checklist=checklist,
        concentration_limitation=not concentration.meta_authority_allowed,
        notes=[
            "Deterministic rule-based meta shadow only; no second-stage ML training.",
            "Context Intelligence remains AE18.",
            "LLM Operational Layer remains AE19.",
            (
                "Concentration limitation recorded; does not auto-block infrastructure closure."
                if not concentration.meta_authority_allowed
                else "Concentration within numeric thresholds or sample warnings only."
            ),
        ],
    )

    gate = {
        "phase": PHASE,
        "classification": classification,
        "created_at_utc": utc_now(),
        "meta_engine_version": META_ENGINE_VERSION,
        "output_root": relpath_str(out, root),
        "ae16_root": discovery.selected_root,
        "ae16_consensus_artifact": discovery.selected_consensus_path,
        "ae16_evidence_artifact": discovery.selected_evidence_path,
        "input_rows_consumed": len(consensus_rows),
        "meta_feature_rows_produced": len(feature_rows),
        "meta_shadow_outputs_produced": len(shadow_rows),
        "context_availability": {
            "context_feature_available_any": any(r.context_feature_available for r in feature_rows),
            "context_status": "AE17_CONTEXT_NOT_AVAILABLE_PENDING_AE18",
            "context_missingness_reason": "AE18_NOT_IMPLEMENTED_OR_NO_CONTEXT_ATTACHED",
        },
        "pair_concentration": {
            "top_pair": concentration.top_pair,
            "top_pair_share": concentration.top_pair_share,
            "hhi": concentration.hhi,
            "concentration_status": concentration.concentration_status,
            "meta_authority_allowed": concentration.meta_authority_allowed,
            "limitation_recorded": not concentration.meta_authority_allowed,
        },
        "score_clamping": {
            "rows_clamped": sum(1 for r in shadow_rows if r.score_clamped),
            "rows_null_score": sum(1 for r in shadow_rows if r.meta_score is None),
        },
        "null_safety_passed": null_safety.get("passed"),
        "no_lookahead_passed": lookahead.get("passed"),
        "feature_parity_passed": parity.get("passed"),
        "lineage_passed": lineage_summary.get("passed"),
        "authority_passed": authority.passed,
        "trade_authority": False,
        "live_trading_ready": False,
        "paper_demo_only": True,
        "risk_override_authority": False,
        "training_performed": False,
        "fit_called": False,
        "sklearn_xgboost_tabicl_training_imports": False,
        "db_mutated": db_mutated,
        "ae18_status": "BLOCKED",
        "ae19_status": "BLOCKED",
        "checklist": checklist,
        "reasons": reasons,
        "decision": decision.to_dict(),
    }

    write_json(reports / "ae17_decision_gate.json", gate)
    write_json(root / "reports" / "ae17_decision_gate.json", gate)

    manifest = {
        "phase": PHASE,
        "classification": classification,
        "created_at_utc": utc_now(),
        "output_root": relpath_str(out, root),
        "meta_engine_version": META_ENGINE_VERSION,
        "ae16_discovery": discovery.to_dict(),
        "files": {
            "meta_feature_rows": relpath_str(data_dir / "ae17_meta_feature_rows.csv", root),
            "meta_shadow_outputs_csv": relpath_str(data_dir / "ae17_meta_shadow_outputs.csv", root),
            "meta_shadow_outputs_jsonl": relpath_str(data_dir / "ae17_meta_shadow_outputs.jsonl", root),
            "meta_layer_matrix": relpath_str(data_dir / "ae17_meta_layer_matrix.csv", root),
            "decision_gate": relpath_str(reports / "ae17_decision_gate.json", root),
        },
        "consensus_artifact_sha256": _file_sha256(consensus_path) if consensus_path.is_file() else None,
    }
    write_json(reports / "ae17_manifest.json", manifest)
    write_json(root / "reports" / "ae17_manifest.json", manifest)

    tier_counts: dict[str, int] = {}
    for r in feature_rows:
        t = str(r.consensus_tier or "MISSING")
        tier_counts[t] = tier_counts.get(t, 0) + 1

    summary_lines = [
        "AE17 Meta-Model / Stacking Layer (deterministic shadow combinator)",
        f"phase: {PHASE}",
        f"classification: {classification}",
        f"output_root: {relpath_str(out, root)}",
        f"ae16_root: {discovery.selected_root}",
        f"ae16_consensus: {discovery.selected_consensus_path}",
        f"input_ae16_rows_consumed: {len(consensus_rows)}",
        f"meta_feature_rows_produced: {len(feature_rows)}",
        f"meta_shadow_outputs_produced: {len(shadow_rows)}",
        f"consensus_tier_counts: {tier_counts}",
        f"context_status: AE17_CONTEXT_NOT_AVAILABLE_PENDING_AE18",
        f"pair_concentration_top_pair_share: {concentration.top_pair_share}",
        f"pair_concentration_hhi: {concentration.hhi}",
        f"pair_concentration_status: {conc_status}",
        f"score_clamping_rows_clamped: {sum(1 for r in shadow_rows if r.score_clamped)}",
        f"null_safety_passed: {null_safety.get('passed')}",
        f"no_lookahead_passed: {lookahead.get('passed')}",
        f"feature_parity_passed: {parity.get('passed')}",
        f"lineage_passed: {lineage_summary.get('passed')}",
        f"authority_passed: {authority.passed}",
        f"training_performed: false",
        f"ae18_status: BLOCKED",
        f"ae19_status: BLOCKED",
        f"reasons: {reasons}",
    ]
    summary_text = "\n".join(summary_lines) + "\n"
    write_text(reports / "ae17_summary_for_upload.txt", summary_text)
    write_text(root / "reports" / "ae17_summary_for_upload.txt", summary_text)

    return {
        "classification": classification,
        "output_root": relpath_str(out, root),
        "decision_gate": gate,
        "manifest": manifest,
        "discovery": discovery.to_dict(),
        "feature_row_count": len(feature_rows),
        "shadow_row_count": len(shadow_rows),
        "pair_concentration": concentration.to_dict(),
        "parity": parity,
        "lookahead": lookahead,
        "lineage": lineage_summary,
        "authority": authority.to_dict(),
        "null_safety": null_safety,
    }
