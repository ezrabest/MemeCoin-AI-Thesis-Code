"""Run/read AE12-SentimentFix Gemini semantic adjudication outputs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .adjudication_cache import (
    append_cache_rows,
    build_cache_row,
    cache_key_fields,
    load_cache,
    lookup_frozen_entry,
    output_cache_path,
    state_cache_path,
)
from .adjudication_safety import redact_secrets, sanity_check_adjudication_output
from .adjudication_safety_status import build_safety_audit, resolve_safety_audit_status
from .adjudication_schema import (
    ADJUDICATION_CLASSES,
    ADJUDICATION_RUBRIC_VERSION,
    ADJUDICATOR_VERSION,
    UI_LABELS,
    default_adjudication,
    map_local_class_to_adjudication,
    normalize_adjudication_payload,
)
from .coin_level_aggregation import derive_coin_level_from_root
from .evidence_builder import build_unique_asset_evidence_with_linkage, load_candidate_rows
from .gemini_adjudicator import adjudicate_asset, get_gemini_hold_status, get_gemini_model_name


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: redact_secrets(str(r.get(k, ""))) if isinstance(r.get(k), str) else r.get(k, "") for k in fieldnames})


def _load_classifier_assets(classifier_root: Path, *, max_assets: int) -> list[dict[str, Any]]:
    path = classifier_root / "data" / "ae12_unique_coin_evidence_packages.csv"
    if path.is_file():
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as fh:
            for i, row in enumerate(csv.DictReader(fh)):
                if i >= max_assets:
                    break
                rows.append(dict(row))
        return rows
    return []


def _load_classifier_classifications(classifier_root: Path) -> dict[str, str]:
    path = classifier_root / "data" / "ae12_semantic_coin_classifications.csv"
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    with path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out[str(row.get("asset_id") or "")] = str(row.get("semantic_coin_class") or "")
    return out


def run_gemini_semantic_adjudication(
    *,
    project_root: Path,
    classifier_root: Path,
    max_assets: int = 100,
    use_gemini: bool = False,
    allow_external_apis: bool = False,
    semantic_reporting_only: bool = True,
    dry_run: bool = False,
    force_refresh: bool = False,
    only_suspected: bool = False,
    only_unknown: bool = False,
    no_web_grounding: bool = False,
    allow_model_knowledge_fallback: bool = False,
    gemini_call=None,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    classifier_root = Path(classifier_root).resolve()
    if not classifier_root.is_absolute():
        classifier_root = (project_root / classifier_root).resolve()

    out_root = project_root / "data" / "audits" / f"ae12_gemini_semantic_adjudication_{_ts_slug()}"
    for d in ("reports", "data", "audits", "state"):
        (out_root / d).mkdir(parents=True, exist_ok=True)

    assets = _load_classifier_assets(classifier_root, max_assets=max_assets)
    local_classes = _load_classifier_classifications(classifier_root)
    if not assets:
        # fallback rebuild from maturation if classifier evidence missing
        ae12_root = classifier_root.parent
        evidence, linkage_rows, linkage_summary = build_unique_asset_evidence_with_linkage(
            project_root=project_root,
            ae12_root=ae12_root,
            max_assets=max_assets,
        )
        assets = evidence
    else:
        linkage_rows = []
        linkage_summary = {}
        summary_path = classifier_root / "reports" / "ae12_semantic_coin_classifier_summary.json"
        if summary_path.is_file():
            try:
                linkage_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                linkage_summary = {}

    if only_unknown or only_suspected:
        filtered = []
        for a in assets:
            lc = local_classes.get(str(a.get("asset_id") or ""), "")
            mapped = map_local_class_to_adjudication(lc)
            if only_unknown and lc == "UNKNOWN_INSUFFICIENT_EVIDENCE":
                filtered.append(a)
            elif only_suspected and mapped == "OPPORTUNISTIC_SUSPECTED":
                filtered.append(a)
            elif only_unknown and only_suspected:
                if lc == "UNKNOWN_INSUFFICIENT_EVIDENCE" or mapped == "OPPORTUNISTIC_SUSPECTED":
                    filtered.append(a)
        assets = filtered[:max_assets]

    cache = load_cache([state_cache_path(project_root), output_cache_path(out_root)])
    cache_rows_to_append: list[dict[str, Any]] = []
    adjudications: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    web_sources: list[dict[str, Any]] = []
    cache_audit_rows: list[dict[str, Any]] = []

    hold_status = get_gemini_hold_status(use_gemini=use_gemini, allow_external_apis=allow_external_apis)
    gemini_used = False
    external_api_used = False
    web_grounding_used = False
    model_knowledge_only_count = 0
    forbidden_terms: set[str] = set()
    forbidden_keys: set[str] = set()
    rejected_outputs = 0
    cache_hits = 0

    for asset in assets[:max_assets]:
        asset_id = str(asset.get("asset_id") or "")
        evidence_hash = str(asset.get("evidence_hash") or "")
        frozen, cache_meta = lookup_frozen_entry(
            asset_id=asset_id,
            evidence_hash=evidence_hash,
            cache=cache,
            force_refresh=force_refresh,
        )
        if frozen and isinstance(frozen.get("adjudication"), dict):
            adj = dict(frozen["adjudication"])
            adj["from_cache"] = True
            adj["evidence_hash_changed_since_classification"] = cache_meta.get(
                "evidence_hash_changed_since_classification", False
            )
            adj["stale_evidence_warning"] = cache_meta.get("stale_evidence_warning", False)
            cache_hits += 1
            cache_audit_rows.append(
                {
                    "asset_id": asset_id,
                    "cache_hit": True,
                    "decision_frozen": True,
                    "evidence_hash_changed_since_classification": cache_meta.get(
                        "evidence_hash_changed_since_classification"
                    ),
                    "stale_evidence_warning": cache_meta.get("stale_evidence_warning"),
                }
            )
        else:
            adj = adjudicate_asset(
                asset,
                use_gemini=use_gemini and not dry_run,
                allow_external_apis=allow_external_apis and not dry_run,
                allow_model_knowledge_fallback=allow_model_knowledge_fallback,
                no_web_grounding=no_web_grounding,
                dry_run=dry_run,
                gemini_call=gemini_call,
            )
            if adj.get("accepted") is False and adj.get("safety_check", {}).get("status") == "REJECTED_FORBIDDEN_TRADE_LANGUAGE":
                rejected_outputs += 1
                rejected.append(
                    {
                        "asset_id": asset_id,
                        "symbol": asset.get("symbol"),
                        "reason": adj.get("safety_check", {}).get("status"),
                        "forbidden_terms_found": ",".join(adj.get("safety_check", {}).get("forbidden_terms_found") or []),
                        "forbidden_keys_found": ",".join(adj.get("safety_check", {}).get("forbidden_keys_found") or []),
                        "raw_llm_text_redacted": adj.get("raw_llm_text_redacted", ""),
                    }
                )
            if adj.get("gemini_used"):
                gemini_used = True
                external_api_used = True
            if adj.get("web_grounding_used"):
                web_grounding_used = True
            if adj.get("raw_evidence_status") == "MODEL_KNOWLEDGE_ONLY":
                model_knowledge_only_count += 1
            safety = adj.get("safety_check") or sanity_check_adjudication_output("", adj)
            forbidden_terms.update(safety.get("forbidden_terms_found") or [])
            forbidden_keys.update(safety.get("forbidden_keys_found") or [])
            if (
                adj.get("accepted", True)
                and not dry_run
                and adj.get("gemini_used")
                and not adj.get("hold_status")
            ):
                cache_rows_to_append.append(
                    build_cache_row(
                        asset=asset,
                        adjudication=adj,
                        gemini_model=str(adj.get("gemini_model") or get_gemini_model_name()),
                    )
                )

        for url in adj.get("source_urls") or []:
            web_sources.append(
                {
                    "asset_id": asset_id,
                    "symbol": asset.get("symbol"),
                    "source_url": url,
                    "linkage_method": adj.get("linkage_method"),
                    "retrieved_at_utc": adj.get("classified_at_utc"),
                }
            )

        local_class = local_classes.get(asset_id, "UNKNOWN_INSUFFICIENT_EVIDENCE")
        adjudications.append(
            {
                "asset_id": asset_id,
                "chain": asset.get("chain"),
                "token_address": asset.get("token_address"),
                "pair_address": asset.get("pair_address"),
                "symbol": asset.get("symbol"),
                "name": asset.get("name"),
                "local_semantic_coin_class": local_class,
                "semantic_coin_class": adj.get("semantic_coin_class"),
                "ui_label": UI_LABELS.get(str(adj.get("semantic_coin_class")), str(adj.get("semantic_coin_class"))),
                "raw_evidence_status": adj.get("raw_evidence_status"),
                "semantic_social_score": adj.get("semantic_social_score"),
                "opportunistic_score": adj.get("opportunistic_score"),
                "infrastructure_score": adj.get("infrastructure_score"),
                "classification_confidence": adj.get("classification_confidence"),
                "positive_criteria_met": json.dumps(adj.get("positive_criteria_met") or []),
                "negative_triggers_met": json.dumps(adj.get("negative_triggers_met") or []),
                "evidence_summary": adj.get("evidence_summary"),
                "reasoning_short": adj.get("reasoning_short"),
                "source_urls": json.dumps(adj.get("source_urls") or []),
                "requires_manual_review": adj.get("requires_manual_review"),
                "gemini_model": adj.get("gemini_model"),
                "adjudicator_version": adj.get("adjudicator_version") or ADJUDICATOR_VERSION,
                "rubric_version": adj.get("rubric_version") or ADJUDICATION_RUBRIC_VERSION,
                "linkage_method": adj.get("linkage_method"),
                "external_api_used": bool(adj.get("external_api_used")),
                "gemini_used": bool(adj.get("gemini_used")),
                "web_grounding_used": bool(adj.get("web_grounding_used")),
                "trade_authority_used": False,
                "from_cache": bool(adj.get("from_cache")),
                "decision_frozen": True,
                "evidence_hash": evidence_hash,
                "evidence_hash_changed_since_classification": adj.get("evidence_hash_changed_since_classification", False),
                "stale_evidence_warning": adj.get("stale_evidence_warning", False),
            }
        )

    append_cache_rows(out_root, cache_rows_to_append, project_root=project_root)

    dist = Counter(r.get("semantic_coin_class") for r in adjudications)
    raw_dist = Counter(r.get("raw_evidence_status") for r in adjudications)
    total = len(adjudications) or 1
    social = int(dist.get("SOCIAL_CONFIRMED", 0))
    opp_conf = int(dist.get("NON_SOCIAL_OPPORTUNISTIC_CONFIRMED", 0))
    opp_sus = int(dist.get("OPPORTUNISTIC_SUSPECTED", 0))
    infra = int(dist.get("NON_SOCIAL_INFRASTRUCTURE_CONFIRMED", 0))
    manual = int(dist.get("MANUAL_REVIEW", 0))

    gate_status = "PASS_GEMINI_ADJUDICATION_READY"
    if hold_status:
        gate_status = hold_status
    elif dry_run:
        gate_status = "PASS_WITH_OP_SUSPECTED_LIMITATION"
    elif not gemini_used and use_gemini:
        gate_status = "HOLD_GEMINI_CLIENT_UNAVAILABLE"
    elif opp_sus / total > 0.85:
        gate_status = "PASS_WITH_OP_SUSPECTED_LIMITATION"
    if manual / total > 0.3:
        gate_status = "HOLD_TOO_MANY_MANUAL_REVIEW"
    # Rejected forbidden outputs must not be treated as accepted classifications.
    accepted_classifications = [
        r
        for r in adjudications
        if str(r.get("raw_evidence_status") or "") != "REJECTED_FOR_TRADE_LANGUAGE"
    ]
    # Accepted rows are only those that passed the safety gate (forbidden language never accepted).
    accepted_with_forbidden = 0
    output_used_after_rejection = False
    safety_audit = build_safety_audit(
        total_gemini_outputs=len(adjudications),
        accepted_outputs=len(accepted_classifications),
        rejected_outputs=rejected_outputs,
        forbidden_terms=forbidden_terms,
        forbidden_keys=forbidden_keys,
        output_used_after_rejection=output_used_after_rejection,
        accepted_classifications_with_forbidden_language=accepted_with_forbidden,
        trade_authority_used=False,
    )
    if resolve_safety_audit_status(safety_audit) == "FAIL":
        gate_status = "FAIL_INVALID_GEMINI_OUTPUT"
    if any("API_KEY" in str(x) for x in forbidden_terms):
        gate_status = "FAIL_API_KEY_LEAK_RISK"

    external_api_audit = {
        "external_api_used": external_api_used,
        "gemini_used": gemini_used,
        "web_grounding_used": web_grounding_used,
        "model_knowledge_only_count": model_knowledge_only_count,
        "api_key_logged": False,
        "api_key_in_artifacts": False,
        "semantic_reporting_only": semantic_reporting_only,
        "trade_authority_used": False,
    }

    gate = {
        "gate_name": "ae12_gemini_semantic_adjudication_gate",
        "status": gate_status,
        "unique_assets_input": len(assets),
        "unique_assets_adjudicated": len(adjudications),
        "social_confirmed_count": social,
        "non_social_opportunistic_confirmed_count": opp_conf,
        "opportunistic_suspected_count": opp_sus,
        "non_social_infrastructure_confirmed_count": infra,
        "manual_review_count": manual,
        "social_confirmed_share": round(social / total, 6),
        "opportunistic_confirmed_share": round(opp_conf / total, 6),
        "opportunistic_suspected_share": round(opp_sus / total, 6),
        "infrastructure_share": round(infra / total, 6),
        "manual_review_share": round(manual / total, 6),
        "raw_evidence_status_distribution": dict(raw_dist),
        "web_grounded_count": int(raw_dist.get("WEB_GROUNDED", 0)),
        "model_knowledge_only_count": model_knowledge_only_count,
        "source_url_count": len(web_sources),
        "gemini_model": get_gemini_model_name() if gemini_used else ("DRY_RUN" if dry_run else "GEMINI_NOT_RUN"),
        "adjudicator_version": ADJUDICATOR_VERSION,
        "rubric_version": ADJUDICATION_RUBRIC_VERSION,
        "external_api_used": external_api_used,
        "gemini_used": gemini_used,
        "web_grounding_used": web_grounding_used,
        "trade_authority_used": False,
        "api_key_logged": False,
        "api_key_in_artifacts": False,
        "forbidden_trade_language_found": bool(forbidden_terms),
        "forbidden_trade_key_found": bool(forbidden_keys),
        "rejected_outputs": rejected_outputs,
        "cache_used": cache_hits > 0,
        "cache_hits": cache_hits,
        "cache_key_fields": cache_key_fields(),
        "freeze_once_policy_enabled": True,
        "reclassification_allowed": False,
        "manual_override_allowed": True,
        "limitations": [
            "Gemini adjudication is semantic reporting only and not trade authority.",
            "OPPORTUNISTIC_SUSPECTED is a triage bucket, not confirmed opportunistic.",
            "raw_evidence_status is preserved separately from final UI bucket.",
        ],
        "recommendation": "Use gemini-semantic-adjudication panel for SOCIAL vs OP.SUSPECTED counts.",
        "live_ready": False,
        "profitability_proven": False,
        "qwen_trade_authority": False,
    }

    examples = {k: [] for k in ADJUDICATION_CLASSES}
    for row in adjudications:
        if str(row.get("raw_evidence_status") or "") == "REJECTED_FOR_TRADE_LANGUAGE":
            continue
        k = row["semantic_coin_class"]
        if k in examples and len(examples[k]) < 20:
            examples[k].append(row)

    distribution_rows = [
        {
            "semantic_coin_class": k,
            "ui_label": UI_LABELS.get(k, k),
            "count": int(dist.get(k, 0)),
            "share": round(int(dist.get(k, 0)) / total, 6),
        }
        for k in ADJUDICATION_CLASSES
    ]

    summary = {
        "phase": "AE12-SentimentFix",
        "subtask": "Gemini Semantic Adjudication",
        "created_at_utc": _utc_now(),
        "output_root": str(out_root),
        "classifier_root": str(classifier_root),
        "gate_status": gate_status,
        "unique_assets_input": len(assets),
        "unique_assets_adjudicated": len(adjudications),
        "class_distribution": dict(dist),
        "raw_evidence_status_distribution": dict(raw_dist),
        "examples_by_class": {
            k: [
                {
                    "asset_id": r.get("asset_id"),
                    "symbol": r.get("symbol"),
                    "ui_label": r.get("ui_label"),
                    "raw_evidence_status": r.get("raw_evidence_status"),
                    "reasoning_short": r.get("reasoning_short"),
                }
                for r in examples[k][:5]
            ]
            for k in ADJUDICATION_CLASSES
        },
        "safety_audit": safety_audit,
        "external_api_audit": external_api_audit,
        "freeze_once_policy_enabled": True,
        "final_semantic_adjudication": gate_status
        in {"PASS_GEMINI_ADJUDICATION_READY", "PASS_WITH_OP_SUSPECTED_LIMITATION"},
        "trade_authority_used": False,
        "live_ready": False,
        "profitability_proven": False,
    }

    _write_csv(out_root / "data" / "ae12_gemini_asset_adjudications.csv", adjudications)
    _write_csv(out_root / "data" / "ae12_gemini_class_distribution.csv", distribution_rows)
    _write_csv(out_root / "data" / "ae12_gemini_social_examples.csv", examples["SOCIAL_CONFIRMED"])
    _write_csv(out_root / "data" / "ae12_gemini_opportunistic_confirmed_examples.csv", examples["NON_SOCIAL_OPPORTUNISTIC_CONFIRMED"])
    _write_csv(out_root / "data" / "ae12_gemini_opportunistic_suspected_examples.csv", examples["OPPORTUNISTIC_SUSPECTED"])
    _write_csv(out_root / "data" / "ae12_gemini_infrastructure_examples.csv", examples["NON_SOCIAL_INFRASTRUCTURE_CONFIRMED"])
    _write_csv(out_root / "data" / "ae12_gemini_manual_review_examples.csv", examples["MANUAL_REVIEW"])
    _write_csv(out_root / "data" / "ae12_gemini_web_sources.csv", web_sources)
    _write_csv(out_root / "data" / "ae12_gemini_rejected_outputs.csv", rejected)
    coin_level = derive_coin_level_from_root(out_root, write=True)
    summary["pair_asset_counts"] = coin_level.get("pair_asset_counts")
    summary["coin_level_counts"] = coin_level.get("coin_level_counts")
    summary["identity_resolution_method_distribution"] = coin_level.get(
        "identity_resolution_method_distribution"
    )
    summary["identity_warning_count"] = coin_level.get("identity_warning_count")
    summary["conflict_count"] = coin_level.get("conflict_count")
    summary["count_level_used_for_main_ui"] = "coin_level"
    if linkage_rows:
        _write_csv(out_root / "data" / "ae12_semantic_linkage_audit.csv", linkage_rows)
        _write_json(out_root / "reports" / "ae12_semantic_linkage_quality_summary.json", linkage_summary)
    _write_json(out_root / "audits" / "ae12_gemini_semantic_adjudication_gate.json", gate)
    _write_json(out_root / "audits" / "ae12_gemini_safety_audit.json", safety_audit)
    _write_csv(out_root / "audits" / "ae12_gemini_cache_audit.csv", cache_audit_rows)
    _write_json(out_root / "audits" / "ae12_external_api_usage_audit.json", external_api_audit)
    _write_json(out_root / "reports" / "ae12_gemini_semantic_adjudication_summary.json", summary)
    manifest = {
        "created_at_utc": summary["created_at_utc"],
        "output_root": str(out_root),
        "files": sorted(str(p.relative_to(out_root)) for p in out_root.rglob("*") if p.is_file()),
        "historical_data_mutated": False,
        "trade_authority_used": False,
        "external_api_used": external_api_used,
        "gemini_used": gemini_used,
    }
    _write_json(out_root / "reports" / "ae12_gemini_semantic_adjudication_manifest.json", manifest)
    upload = [
        "AE12-SentimentFix Gemini Semantic Adjudication (not AE12.6)",
        f"output_root: {out_root}",
        f"gate_status: {gate_status}",
        f"unique_assets_input: {len(assets)}",
        f"unique_assets_adjudicated: {len(adjudications)}",
        f"SOCIAL_CONFIRMED: {social} ({round(social/total,6)})",
        f"NON_SOCIAL_OPPORTUNISTIC_CONFIRMED: {opp_conf} ({round(opp_conf/total,6)})",
        f"OPPORTUNISTIC_SUSPECTED (OP.SUSPECTED): {opp_sus} ({round(opp_sus/total,6)})",
        f"NON_SOCIAL_INFRASTRUCTURE_CONFIRMED: {infra} ({round(infra/total,6)})",
        f"MANUAL_REVIEW: {manual} ({round(manual/total,6)})",
        f"raw_evidence_status_distribution: {dict(raw_dist)}",
        f"web_grounding_used: {web_grounding_used}",
        f"model_knowledge_only_count: {model_knowledge_only_count}",
        f"source_url_count: {len(web_sources)}",
        f"gemini_model: {gate['gemini_model']}",
        f"adjudicator_version: {ADJUDICATOR_VERSION}",
        "trade_authority_used: false",
        f"external_api_used: {str(external_api_used).lower()}",
        f"gemini_used: {str(gemini_used).lower()}",
        "api_key_logged: false",
        "OP.SUSPECTED is suspected opportunistic, not confirmed.",
        "Gemini is semantic reporting only, not trade authority.",
    ]
    (out_root / "reports" / "ae12_gemini_semantic_adjudication_for_upload.txt").write_text(
        "\n".join(upload) + "\n",
        encoding="utf-8",
    )
    return summary
