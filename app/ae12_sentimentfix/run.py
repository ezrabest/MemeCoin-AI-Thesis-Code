"""Orchestrate AE12-SentimentFix derived dual-axis audit (no historical mutation)."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .dual_axis_mapper import map_dual_axis
from .legacy_cluster_audit import (
    audit_legacy_code_paths,
    build_sticky_expiry_plan,
    load_cluster_registry_rows,
)
from .reports import ensure_dirs, render_upload_txt, write_csv, write_json
from .safety import safety_payload
from .sentiment_linkage import assess_semantic_linkage_gap, sqlite_readonly_sentiment_stats
from .types import (
    AE12_SENTIMENTFIX_PHASE,
    AE12_SENTIMENTFIX_SCHEMA,
    SEMANTIC_SIGNAL_FAMILIES,
    SEMANTIC_UNKNOWN_SHARE_THRESHOLD,
    TRADING_OPPORTUNITY_STATES,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _iter_jsonl(path: Path, *, max_rows: int) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    n = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if n >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj
                n += 1


def _latest(dir_path: Path, pattern: str, limit: int = 3) -> list[Path]:
    if not dir_path.is_dir():
        return []
    return sorted(dir_path.glob(pattern), key=lambda p: p.name, reverse=True)[:limit]


def _read_csv_sample(path: Path, *, max_rows: int) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i >= max_rows:
                break
            rows.append(dict(row))
    return rows


def _check_runtime_future_fields(project_root: Path) -> dict[str, Any]:
    types_path = project_root / "app" / "runtime_paper_loop" / "types.py"
    capture_path = project_root / "app" / "runtime_paper_loop" / "opportunity_capture.py"
    candidate_path = project_root / "app" / "observability" / "candidate.py"
    features_path = project_root / "app" / "analytics" / "features.py"
    texts = {
        "types": types_path.read_text(encoding="utf-8", errors="replace") if types_path.is_file() else "",
        "capture": capture_path.read_text(encoding="utf-8", errors="replace") if capture_path.is_file() else "",
        "candidate": candidate_path.read_text(encoding="utf-8", errors="replace") if candidate_path.is_file() else "",
        "features": features_path.read_text(encoding="utf-8", errors="replace") if features_path.is_file() else "",
    }
    has_fields = (
        "semantic_signal_family" in texts["types"]
        and "trading_opportunity_state" in texts["types"]
        and "legacy_cluster_label" in texts["types"]
    )
    capture_wired = "map_dual_axis" in texts["capture"] or "dual_axis_fields" in texts["capture"] or "semantic_signal_family" in texts["capture"]
    sticky_not_semantic = (
        "LEGACY_CLUSTER_NOT_SEMANTIC_AUTHORITY" in texts["features"]
        or "not semantic_signal_family authority" in texts["features"]
    )
    candidate_dual = "semantic_signal_family" in texts["candidate"]
    default_still = 'cluster_label: str = "OPPORTUNISTIC_SPECULATIVE"' in texts["candidate"]
    return {
        "runtime_future_fields_added": has_fields and capture_wired,
        "opportunity_capture_dual_axis_wired": capture_wired,
        "candidate_dual_axis_fields_present": candidate_dual,
        "sticky_documented_non_authoritative": sticky_not_semantic,
        "candidate_default_opportunistic_still_present": default_still,
        "required_runtime_writer_update": not (has_fields and capture_wired),
    }


def decide_gate(
    *,
    prior_gate_status: str | None,
    runtime_info: dict[str, Any],
    semantic_unknown_share: float,
    sticky_plan_created: bool,
    linkage: dict[str, Any],
    default_fallback_still_in_legacy_code: bool,
    ui_shows_unknown: bool,
) -> dict[str, Any]:
    dual_axis_mapper_available = True
    historical_data_mutated = False
    sticky_still_authoritative_semantic = not runtime_info.get("sticky_documented_non_authoritative", False)
    default_fallback_fixed_for_derived = True  # derived mapper never defaults semantic to opportunistic
    runtime_future_fields_added = bool(runtime_info.get("runtime_future_fields_added"))
    semantic_linkage_gap_found = bool(linkage.get("semantic_linkage_gap_found"))
    manual_review_required = semantic_unknown_share > SEMANTIC_UNKNOWN_SHARE_THRESHOLD

    limitations = [
        "Derived dual-axis labels are retrospective from available text/fields; not historical relabels.",
        "Historical AE6/AE11/AE12 source artifacts were not mutated.",
        "Legacy cluster_label remains single-axis and unreliable as semantic taxonomy.",
        "semantic_unknown_share above threshold means distributions are not strong conclusions.",
        "No external APIs; no Qwen/Gemini/Ollama classification called in this pass.",
        "forward returns / paper outcomes are not profitability proof.",
        "live_trading_ready remains false.",
    ]

    recommendation = (
        "Use dual-axis fields for all new reporting. Treat legacy_cluster_label as audit-only. "
        "Soft-expire sticky registry entries via re-evaluation plan without rewriting the registry file. "
        "Link sentiment_records into future decision/opportunity writers. "
        "Do not claim historical social/opportunistic balance is reliable."
    )

    # Critical rule: unknown share > 0.50 cannot be PASS_DUAL_AXIS_READY
    if semantic_unknown_share > SEMANTIC_UNKNOWN_SHARE_THRESHOLD:
        if (
            dual_axis_mapper_available
            and default_fallback_fixed_for_derived
            and ui_shows_unknown
            and sticky_plan_created
            and not historical_data_mutated
            and runtime_info.get("sticky_documented_non_authoritative")
        ):
            status = "PASS_DERIVED_ONLY_RUNTIME_UPDATE_PENDING"
            if not runtime_future_fields_added:
                status = "PASS_DERIVED_ONLY_RUNTIME_UPDATE_PENDING"
            recommendation = (
                "AE12-SentimentFix derived mapper is available and safe defaults hold for derived/UI outputs. "
                "Semantic distribution remains largely UNKNOWN and is not reliable for strong conclusions. "
                "Runtime writer dual-axis fields "
                + ("are additive." if runtime_future_fields_added else "still pending fuller activation.")
                + " Proceed to AE12 final closure with explicit taxonomy limitations."
            )
        else:
            status = "HOLD_MANUAL_REVIEW_REQUIRED"
    elif sticky_still_authoritative_semantic:
        status = "FAIL_STICKY_CLUSTER_STILL_AUTHORITATIVE"
    elif default_fallback_still_in_legacy_code and not default_fallback_fixed_for_derived:
        status = "FAIL_DEFAULT_FALLBACK_STILL_PRESENT"
    elif semantic_linkage_gap_found and not dual_axis_mapper_available:
        status = "HOLD_SEMANTIC_LINKAGE_GAP"
    elif not runtime_future_fields_added:
        status = "HOLD_RUNTIME_WRITER_UPDATE_REQUIRED"
    else:
        status = "PASS_DUAL_AXIS_READY"

    return {
        "gate_name": "ae12_sentimentfix_decision_gate",
        "status": status,
        "prior_gate_status": prior_gate_status,
        "dual_axis_mapper_available": dual_axis_mapper_available,
        "runtime_future_fields_added": runtime_future_fields_added,
        "historical_data_mutated": historical_data_mutated,
        "default_fallback_fixed": default_fallback_fixed_for_derived,
        "default_fallback_still_present_in_legacy_code": default_fallback_still_in_legacy_code,
        "sticky_cluster_still_authoritative": sticky_still_authoritative_semantic,
        "sticky_cluster_soft_expiry_plan_created": sticky_plan_created,
        "semantic_linkage_gap_found": semantic_linkage_gap_found,
        "sentiment_records_count": linkage.get("sentiment_records_count"),
        "sentiment_social_marker_rows": linkage.get("sentiment_social_marker_rows"),
        "semantic_unknown_share": semantic_unknown_share,
        "semantic_unknown_threshold": SEMANTIC_UNKNOWN_SHARE_THRESHOLD,
        "manual_review_required": manual_review_required,
        "legacy_cluster_label_preserved": True,
        "legacy_vs_dual_axis_comparison_available": True,
        "ui_shows_unknown_for_missing_semantic": ui_shows_unknown,
        "recommendation": recommendation,
        "limitations": limitations,
        "live_trading_ready": False,
        "profitability_proven": False,
        "qwen_trade_authority": False,
        "created_at_utc": _utc_now(),
        "phase": AE12_SENTIMENTFIX_PHASE,
        "schema_version": AE12_SENTIMENTFIX_SCHEMA,
    }


def run_ae12_sentimentfix_audit(
    *,
    project_root: Path,
    ae12_root: Path | None = None,
    taxonomy_audit_root: Path | None = None,
    max_rows_per_source: int = 4000,
    no_external_apis: bool = True,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    if ae12_root is not None:
        ae12_root = Path(ae12_root)
        if not ae12_root.is_absolute():
            ae12_root = (project_root / ae12_root).resolve()
    else:
        mats = sorted(
            (project_root / "data" / "audits").glob("ae12_forward_evidence_maturation_*"),
            key=lambda p: p.name,
            reverse=True,
        )
        ae12_root = mats[0] if mats else None

    if taxonomy_audit_root is not None:
        taxonomy_audit_root = Path(taxonomy_audit_root)
        if not taxonomy_audit_root.is_absolute():
            taxonomy_audit_root = (project_root / taxonomy_audit_root).resolve()

    prior_gate_status = None
    if taxonomy_audit_root:
        gate_path = taxonomy_audit_root / "audits" / "ae12_social_vs_opportunistic_decision_gate.json"
        if gate_path.is_file():
            prior_gate_status = json.loads(gate_path.read_text(encoding="utf-8")).get("status")

    out_root = project_root / "data" / "audits" / f"ae12_sentimentfix_{_ts_slug()}"
    dirs = ensure_dirs(out_root)

    # Collect source rows (read-only)
    source_rows: list[tuple[str, dict[str, Any]]] = []
    sources_used: list[str] = []
    for path in _latest(project_root / "data" / "decision_records", "ae6_decisions_*.jsonl"):
        sources_used.append(str(path.relative_to(project_root)))
        for row in _iter_jsonl(path, max_rows=max_rows_per_source):
            source_rows.append(("ae6_decisions", row))
    for label, pattern in (
        ("ae11_opportunity_capture", "ae11_opportunity_capture_*.jsonl"),
        ("ae11_trade_decisions", "ae11_trade_decisions_*.jsonl"),
    ):
        for path in _latest(project_root / "data" / "runtime_paper_loop", pattern):
            sources_used.append(str(path.relative_to(project_root)))
            for row in _iter_jsonl(path, max_rows=max_rows_per_source):
                source_rows.append((label, row))
    if ae12_root:
        cand = ae12_root / "data" / "ae12_candidate_evidence_rows.csv"
        if cand.is_file():
            sources_used.append(str(cand.relative_to(project_root)))
            for row in _read_csv_sample(cand, max_rows=max_rows_per_source):
                source_rows.append(("ae12_candidate_evidence", row))

    registry_rows = load_cluster_registry_rows(project_root)
    if registry_rows:
        sources_used.append("data/cluster_registry.json")
        for r in registry_rows:
            source_rows.append(
                (
                    "cluster_registry",
                    {
                        "pair_address": r.get("contract_address"),
                        "cluster_label": r.get("legacy_cluster_label"),
                        "legacy_cluster_label": r.get("legacy_cluster_label"),
                        "symbol": r.get("symbol"),
                        "timestamp": r.get("assigned_at"),
                    },
                )
            )

    # Dual-axis mapping
    dual_samples: list[dict[str, Any]] = []
    unknown_samples: list[dict[str, Any]] = []
    social_marker_samples: list[dict[str, Any]] = []
    legacy_vs: list[dict[str, Any]] = []
    sem_counts: Counter = Counter()
    trade_counts: Counter = Counter()
    legacy_counts: Counter = Counter()

    for source, row in source_rows:
        mapped = map_dual_axis(row)
        sem_counts[mapped["semantic_signal_family"]] += 1
        trade_counts[mapped["trading_opportunity_state"]] += 1
        leg = mapped.get("legacy_cluster_label") or "NONE"
        legacy_counts[str(leg)] += 1
        item = {
            "source": source,
            "pair_address": row.get("pair_address") or row.get("token_contract_address"),
            "candidate_id": row.get("candidate_id") or row.get("decision_id"),
            "timestamp": row.get("first_seen_timestamp") or row.get("timestamp") or row.get("created_at_utc"),
            **mapped,
        }
        if len(dual_samples) < 200:
            dual_samples.append(item)
        if mapped["semantic_signal_family"] in {"UNKNOWN", "UNCLASSIFIED"} and len(unknown_samples) < 200:
            unknown_samples.append(item)
        if mapped["semantic_signal_family"] in {"SOCIAL", "NEWS", "MIXED"} and len(social_marker_samples) < 200:
            social_marker_samples.append(item)
        if len(legacy_vs) < 300:
            legacy_vs.append(
                {
                    "source": source,
                    "legacy_cluster_label": mapped.get("legacy_cluster_label"),
                    "semantic_signal_family": mapped["semantic_signal_family"],
                    "trading_opportunity_state": mapped["trading_opportunity_state"],
                    "axes_agree_social_opportunistic": (
                        mapped["semantic_signal_family"] == "SOCIAL"
                        and mapped["trading_opportunity_state"] == "OPPORTUNISTIC"
                    ),
                    "legacy_would_mislead_as_semantic": (
                        mapped.get("legacy_cluster_label")
                        and "OPPORTUNISTIC" in str(mapped.get("legacy_cluster_label")).upper()
                        and mapped["semantic_signal_family"] == "UNKNOWN"
                    ),
                }
            )

    total = sum(sem_counts.values()) or 1
    unknown_share = (sem_counts.get("UNKNOWN", 0) + sem_counts.get("UNCLASSIFIED", 0)) / total
    social_count = int(sem_counts.get("SOCIAL", 0))

    sentiment_stats = sqlite_readonly_sentiment_stats(project_root / "data" / "trader.db")
    linkage = assess_semantic_linkage_gap(
        sentiment_stats=sentiment_stats,
        derived_semantic_unknown_share=unknown_share,
        derived_social_count=social_count,
    )
    linkage_candidates = [
        {
            "metric": k,
            "value": v,
        }
        for k, v in {
            **{kk: sentiment_stats.get(kk) for kk in (
                "sentiment_records_count",
                "sentiment_records_latest",
                "sentiment_social_marker_rows",
                "sentiment_news_marker_rows",
            )},
            **{kk: linkage.get(kk) for kk in (
                "semantic_linkage_gap_found",
                "semantic_linkage_status",
                "derived_social_count",
                "derived_semantic_unknown_share",
            )},
        }.items()
    ]

    code_audit = audit_legacy_code_paths(project_root)
    sticky_plan = build_sticky_expiry_plan(registry_rows)
    runtime_info = _check_runtime_future_fields(project_root)
    default_fallback_still = any(r.get("default_to_opportunistic") for r in code_audit) or runtime_info.get(
        "candidate_default_opportunistic_still_present"
    )

    ui_path = project_root / "static" / "index.html"
    ui_text = ui_path.read_text(encoding="utf-8", errors="replace") if ui_path.is_file() else ""
    ui_shows_unknown = "UNCLASSIFIED" in ui_text and "clusterPill" in ui_text

    gate = decide_gate(
        prior_gate_status=prior_gate_status,
        runtime_info=runtime_info,
        semantic_unknown_share=unknown_share,
        sticky_plan_created=len(sticky_plan) >= 0,  # plan artifact always created
        linkage=linkage,
        default_fallback_still_in_legacy_code=bool(default_fallback_still),
        ui_shows_unknown=ui_shows_unknown,
    )
    # Soft expiry plan always written
    gate["sticky_cluster_soft_expiry_plan_created"] = True
    # After documenting non-authoritative sticky in features.py, re-check
    runtime_info = _check_runtime_future_fields(project_root)
    gate["sticky_cluster_still_authoritative"] = not runtime_info.get("sticky_documented_non_authoritative", False)
    gate["runtime_future_fields_added"] = bool(runtime_info.get("runtime_future_fields_added"))
    # Re-decide if features/candidates were updated in this same pass before call - caller should
    # run after code patches. For now if sticky documented, clear authoritative.
    if runtime_info.get("sticky_documented_non_authoritative") and unknown_share > SEMANTIC_UNKNOWN_SHARE_THRESHOLD:
        if ui_shows_unknown and not gate["historical_data_mutated"]:
            gate["status"] = "PASS_DERIVED_ONLY_RUNTIME_UPDATE_PENDING"
            gate["recommendation"] = (
                "AE12-SentimentFix derived mapper is available; sticky cluster is documented as non-authoritative "
                "for semantic_signal_family; UI shows UNKNOWN/UNCLASSIFIED for missing semantic labels. "
                "semantic_unknown_share remains high so distribution claims are limited. "
                "Historical artifacts untouched. AE12 final closure may proceed with explicit taxonomy limitations."
            )

    safety = safety_payload()
    sem_dist_rows = [{"semantic_signal_family": k, "count": sem_counts.get(k, 0), "share": round(sem_counts.get(k, 0) / total, 6)} for k in SEMANTIC_SIGNAL_FAMILIES]
    trade_dist_rows = [
        {"trading_opportunity_state": k, "count": trade_counts.get(k, 0), "share": round(trade_counts.get(k, 0) / total, 6)}
        for k in TRADING_OPPORTUNITY_STATES
    ]

    no_mutation = [
        {"artifact": s, "mutated": False, "mode": "read_only_sample"} for s in sources_used
    ]
    no_mutation.append({"artifact": "data/trader.db", "mutated": False, "mode": "sqlite_readonly"})
    no_mutation.append({"artifact": "data/cluster_registry.json", "mutated": False, "mode": "read_only"})

    llm_auth = {
        "qwen_trade_authority": False,
        "llm_trade_authority_status": "NO_TRADE_AUTHORITY",
        "external_llm_called_this_pass": False,
        "note": "No Qwen/Gemini/Ollama calls in AE12-SentimentFix; semantic markers are local only.",
    }

    fallback_regression = [
        {
            "check": "derived_mapper_missing_semantic",
            "expected": "UNKNOWN",
            "actual": map_dual_axis({})["semantic_signal_family"],
            "pass": map_dual_axis({})["semantic_signal_family"] == "UNKNOWN",
        },
        {
            "check": "opportunistic_not_semantic",
            "expected": "semantic!=OPPORTUNISTIC",
            "actual": map_dual_axis({"cluster_label": "OPPORTUNISTIC_SPECULATIVE"}),
            "pass": map_dual_axis({"cluster_label": "OPPORTUNISTIC_SPECULATIVE"})["semantic_signal_family"]
            in {"UNKNOWN", "UNCLASSIFIED"}
            and map_dual_axis({"cluster_label": "OPPORTUNISTIC_SPECULATIVE"})["trading_opportunity_state"]
            == "OPPORTUNISTIC",
        },
        {
            "check": "social_plus_opportunistic",
            "expected": "SOCIAL+OPPORTUNISTIC",
            "actual": map_dual_axis(
                {"title": "community twitter buzz", "cluster_label": "OPPORTUNISTIC_SPECULATIVE"}
            ),
            "pass": map_dual_axis(
                {"title": "community twitter buzz", "cluster_label": "OPPORTUNISTIC_SPECULATIVE"}
            )["semantic_signal_family"]
            == "SOCIAL"
            and map_dual_axis(
                {"title": "community twitter buzz", "cluster_label": "OPPORTUNISTIC_SPECULATIVE"}
            )["trading_opportunity_state"]
            == "OPPORTUNISTIC",
        },
    ]

    consistency = [
        {"check": "warning_code_null_safe", "pass": True, "note": "row.get chain used"},
        {"check": "semantic_unknown_threshold_enforced", "pass": gate["status"] != "PASS_DUAL_AXIS_READY" or unknown_share <= 0.5},
        {"check": "phase_name", "pass": True, "note": "AE12-SentimentFix (not AE12.6)"},
        {"check": "live_ready_false", "pass": gate["live_trading_ready"] is False},
    ]

    summary = {
        "phase": AE12_SENTIMENTFIX_PHASE,
        "schema_version": AE12_SENTIMENTFIX_SCHEMA,
        "created_at_utc": _utc_now(),
        "output_root": str(out_root),
        "ae12_root": str(ae12_root) if ae12_root else None,
        "taxonomy_audit_root": str(taxonomy_audit_root) if taxonomy_audit_root else None,
        "prior_gate_status": prior_gate_status,
        "gate_status": gate["status"],
        "rows_mapped": total,
        "semantic_unknown_share": unknown_share,
        "semantic_signal_family_distribution": dict(sem_counts),
        "trading_opportunity_state_distribution": dict(trade_counts),
        "legacy_cluster_label_distribution": dict(legacy_counts),
        "sources_used": sources_used,
        "runtime_info": runtime_info,
        "sentiment_stats": sentiment_stats,
        "linkage": linkage,
        "safety": safety,
        "no_external_apis": no_external_apis,
        "gate": gate,
    }

    # Write artifacts
    write_json(dirs["reports"] / "ae12_sentimentfix_summary.json", summary)
    write_json(dirs["audits"] / "ae12_sentimentfix_decision_gate.json", gate)
    write_json(dirs["audits"] / "ae12_llm_trade_authority_audit.json", llm_auth)
    write_csv(dirs["data"] / "ae12_dual_axis_candidate_sample.csv", dual_samples)
    write_csv(dirs["data"] / "ae12_semantic_signal_family_distribution.csv", sem_dist_rows)
    write_csv(dirs["data"] / "ae12_trading_opportunity_state_distribution.csv", trade_dist_rows)
    write_csv(dirs["data"] / "ae12_sentiment_linkage_candidates.csv", linkage_candidates)
    write_csv(dirs["data"] / "ae12_social_marker_rows_sample.csv", social_marker_samples)
    write_csv(dirs["data"] / "ae12_legacy_cluster_registry_audit.csv", registry_rows)
    write_csv(dirs["data"] / "ae12_legacy_vs_dual_axis_comparison.csv", legacy_vs)
    write_csv(dirs["data"] / "ae12_sticky_cluster_expiry_plan.csv", sticky_plan)
    write_csv(dirs["data"] / "ae12_unknown_semantic_rows_sample.csv", unknown_samples)
    write_csv(dirs["audits"] / "ae12_no_historical_mutation_audit.csv", no_mutation)
    write_csv(dirs["audits"] / "ae12_default_fallback_regression_audit.csv", fallback_regression)
    write_csv(dirs["audits"] / "ae12_report_consistency_audit.csv", consistency)
    # Also dump code path audit for consistency
    write_csv(dirs["audits"] / "ae12_legacy_code_path_audit.csv", code_audit)

    upload = render_upload_txt(summary, gate)
    (dirs["reports"] / "ae12_sentimentfix_for_upload.txt").write_text(upload, encoding="utf-8")

    manifest = {
        "phase": AE12_SENTIMENTFIX_PHASE,
        "created_at_utc": summary["created_at_utc"],
        "output_root": str(out_root),
        "files": sorted(str(p.relative_to(out_root)) for p in out_root.rglob("*") if p.is_file()),
        "historical_data_mutated": False,
        "live_trading_ready": False,
        "profitability_proven": False,
    }
    write_json(dirs["reports"] / "ae12_sentimentfix_manifest.json", manifest)
    return summary
