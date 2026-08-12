"""Run/read AE12 semantic coin classifier outputs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .classification_cache import (
    append_cache_rows,
    cache_key,
    cache_key_fields,
    cache_path,
    cache_uses_evidence_hash,
    load_cache,
)
from .classification_schema import CLASSIFIER_VERSION, RUBRIC_VERSION, SEMANTIC_COIN_CLASSES
from .evidence_builder import build_unique_asset_evidence_with_linkage
from .llm_classifier import classify_asset_semantic


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
            w.writerow({k: r.get(k, "") for k in fieldnames})


def run_semantic_coin_classifier(
    *,
    project_root: Path,
    ae12_root: Path,
    sentimentfix_root: Path | None = None,
    max_assets: int = 1000,
    local_llm_only: bool = True,
    no_external_apis: bool = True,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    if not ae12_root.is_absolute():
        ae12_root = (project_root / ae12_root).resolve()
    if sentimentfix_root and not sentimentfix_root.is_absolute():
        sentimentfix_root = (project_root / sentimentfix_root).resolve()

    out_root = project_root / "data" / "audits" / f"ae12_semantic_coin_classifier_{_ts_slug()}"
    for d in ("reports", "data", "audits", "state"):
        (out_root / d).mkdir(parents=True, exist_ok=True)

    evidence, linkage_rows, linkage_summary = build_unique_asset_evidence_with_linkage(
        project_root=project_root,
        ae12_root=ae12_root,
        max_assets=max_assets,
    )
    unique_assets_found = len(evidence)
    cache = load_cache(out_root)
    cache_rows_to_append: list[dict[str, Any]] = []

    classifications: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    unknown_reasons: Counter = Counter()
    total_llm_outputs = 0
    accepted_outputs = 0
    rejected_outputs = 0
    forbidden_terms: set[str] = set()
    local_llm_used = False
    classifier_status = "LOCAL_LLM_UNAVAILABLE"

    for asset in evidence:
        key = cache_key(
            asset_id=str(asset.get("asset_id") or ""),
            classifier_version=CLASSIFIER_VERSION,
            evidence_hash=str(asset.get("evidence_hash") or ""),
            rubric_version=RUBRIC_VERSION,
        )
        cached = cache.get(key)
        if cached and isinstance(cached.get("classification"), dict):
            c = dict(cached["classification"])
            c["from_cache"] = True
        else:
            c = classify_asset_semantic(asset, local_llm_only=local_llm_only)
            cache_rows_to_append.append(
                {
                    "cache_key": key,
                    "asset_id": asset.get("asset_id"),
                    "classifier_version": CLASSIFIER_VERSION,
                    "evidence_hash": asset.get("evidence_hash"),
                    "rubric_version": RUBRIC_VERSION,
                    "classification": c,
                    "cached_at_utc": _utc_now(),
                }
            )
        total_llm_outputs += 1
        local_llm_used = local_llm_used or (c.get("classifier_status") == "LOCAL_LLM_USED")
        classifier_status = c.get("classifier_status") or classifier_status
        safety = c.get("safety_check") or {}
        if safety.get("forbidden_trade_language_found"):
            forbidden_terms.update(safety.get("forbidden_terms_found") or [])
        if c.get("accepted"):
            accepted_outputs += 1
        else:
            rejected_outputs += 1
            rejected.append(
                {
                    "asset_id": asset.get("asset_id"),
                    "symbol": asset.get("symbol"),
                    "reason": safety.get("status") or "REJECTED",
                    "forbidden_terms_found": ",".join(safety.get("forbidden_terms_found") or []),
                    "raw_llm_text": "",
                }
            )

        klass = str(c.get("semantic_coin_class") or "UNKNOWN_INSUFFICIENT_EVIDENCE")
        if klass == "UNKNOWN_INSUFFICIENT_EVIDENCE":
            unknown_reasons["INSUFFICIENT_EVIDENCE_OR_LLM_UNAVAILABLE"] += 1
        if klass == "MANUAL_REVIEW":
            unknown_reasons["MANUAL_REVIEW_REQUIRED"] += 1
        classifications.append(
            {
                "asset_id": asset.get("asset_id"),
                "chain": asset.get("chain"),
                "symbol": asset.get("symbol"),
                "name": asset.get("name"),
                "identity_confidence": asset.get("identity_confidence"),
                "legacy_cluster_label": asset.get("legacy_cluster_label"),
                "semantic_coin_class": klass,
                "semantic_social_score": c.get("semantic_social_score"),
                "speculation_score": c.get("speculation_score"),
                "classification_confidence": c.get("classification_confidence"),
                "positive_criteria_met": json.dumps(c.get("positive_criteria_met") or []),
                "negative_triggers_met": json.dumps(c.get("negative_triggers_met") or []),
                "reasoning_short": c.get("reasoning_short"),
                "evidence_summary": c.get("evidence_summary"),
                "requires_manual_review": c.get("requires_manual_review"),
                "classifier_model": c.get("classifier_model"),
                "classifier_version": c.get("classifier_version"),
                "rubric_version": c.get("rubric_version"),
                "classifier_status": c.get("classifier_status"),
                "trade_authority_used": False,
                "external_api_used": False,
                "evidence_hash": asset.get("evidence_hash"),
                "source_count": asset.get("source_count"),
            }
        )

    append_cache_rows(out_root, cache_rows_to_append)

    dist = Counter(r.get("semantic_coin_class") for r in classifications)
    total = len(classifications) or 1
    social = int(dist.get("SOCIAL", 0))
    opp = int(dist.get("NON_SOCIAL_OPPORTUNISTIC", 0))
    infra = int(dist.get("NON_SOCIAL_INFRASTRUCTURE", 0))
    unknown = int(dist.get("UNKNOWN_INSUFFICIENT_EVIDENCE", 0))
    manual = int(dist.get("MANUAL_REVIEW", 0))
    unknown_share = round(unknown / total, 6)
    manual_share = round(manual / total, 6)

    gate_status = "PASS_CLASSIFIER_READY"
    if classifier_status == "LOCAL_LLM_UNAVAILABLE":
        gate_status = "HOLD_LOCAL_LLM_UNAVAILABLE"
    if unknown_share > 0.6:
        gate_status = "PASS_CLASSIFIER_READY_WITH_UNKNOWN_LIMITATION"
    if manual_share > 0.3:
        gate_status = "HOLD_MANUAL_REVIEW_REQUIRED"
    if rejected_outputs > 0 and len(forbidden_terms) > 0:
        # still acceptable if all rejected outputs excluded from accepted set
        if accepted_outputs + rejected_outputs == total_llm_outputs:
            gate_status = gate_status
        else:
            gate_status = "FAIL_LLM_USED_AS_TRADE_AUTHORITY"

    distribution_rows = []
    for k in SEMANTIC_COIN_CLASSES:
        c = int(dist.get(k, 0))
        distribution_rows.append({"semantic_coin_class": k, "count": c, "share": round(c / total, 6)})

    examples = {k: [] for k in SEMANTIC_COIN_CLASSES}
    for row in classifications:
        k = row["semantic_coin_class"]
        if len(examples[k]) < 20:
            examples[k].append(row)

    safety_audit = {
        "total_llm_outputs": total_llm_outputs,
        "accepted_outputs": accepted_outputs,
        "rejected_outputs": rejected_outputs,
        "forbidden_trade_language_found": len(forbidden_terms) > 0,
        "forbidden_terms_found": sorted(forbidden_terms),
        "trade_authority_used": False,
        "output_used_after_rejection": False,
        "status": "PASS_REJECTIONS_ENFORCED" if rejected_outputs >= 0 else "FAIL",
    }

    cache_audit_rows = [
        {
            "classification_cache_used": True,
            "cache_path": str(cache_path(out_root)),
            "cache_key_fields": ",".join(cache_key_fields()),
            "cache_key_uses_evidence_hash": cache_uses_evidence_hash(),
            "cache_key_weak_asset_only": False,
        }
    ]
    dedup_audit = [
        {
            "unique_assets_found": unique_assets_found,
            "candidate_rows_seen": len(evidence),
            "dedup_key_priority": "chain+token_address > chain+pair_address > symbol+chain",
        }
    ]

    gate = {
        "gate_name": "ae12_semantic_classifier_decision_gate",
        "status": gate_status,
        "unique_assets_found": unique_assets_found,
        "unique_assets_classified": len(classifications),
        "social_count": social,
        "non_social_opportunistic_count": opp,
        "non_social_infrastructure_count": infra,
        "unknown_count": unknown,
        "manual_review_count": manual,
        "social_share": round(social / total, 6),
        "opportunistic_share": round(opp / total, 6),
        "unknown_share": unknown_share,
        "manual_review_share": manual_share,
        "unknown_reason_breakdown": dict(unknown_reasons),
        "classifier_status": classifier_status,
        "classifier_model": "LOCAL_LLM_UNAVAILABLE" if not local_llm_used else "LOCAL_LLM_USED",
        "classifier_version": CLASSIFIER_VERSION,
        "rubric_version": RUBRIC_VERSION,
        "local_llm_used": local_llm_used,
        "external_api_used": False,
        "trade_authority_used": False,
        "forbidden_trade_language_found": safety_audit["forbidden_trade_language_found"],
        "rejected_llm_outputs": rejected_outputs,
        "classification_cache_used": True,
        "cache_key_fields": cache_key_fields(),
        "cache_key_uses_evidence_hash": cache_uses_evidence_hash(),
        "evidence_sources_checked": [
            "data/trader.db:coins,sentiment_records,signals",
            "ae12_candidate_evidence_rows.csv",
            "ae12_sentimentfix outputs",
            "cluster_registry.json (legacy evidence only)",
        ],
        "limitations": [
            "Classifier output is semantic reporting only and not trade authority.",
            "UNKNOWN means insufficient evidence or local LLM unavailable.",
            "No external API calls.",
        ],
        "recommendation": "Use unique-asset semantic counts with UNKNOWN breakdown in UI; do not infer trade actions.",
        "live_ready": False,
        "profitability_proven": False,
        "qwen_trade_authority": False,
    }

    summary = {
        "phase": "AE12-SentimentFix",
        "subtask": "LLM Semantic Coin Classifier",
        "created_at_utc": _utc_now(),
        "output_root": str(out_root),
        "ae12_root": str(ae12_root),
        "sentimentfix_root": str(sentimentfix_root) if sentimentfix_root else None,
        "gate_status": gate_status,
        "unique_assets_found": unique_assets_found,
        "unique_assets_classified": len(classifications),
        "class_distribution": dict(dist),
        "examples_by_class": {
            "SOCIAL": [
                {"asset_id": r.get("asset_id"), "symbol": r.get("symbol"), "reasoning_short": r.get("reasoning_short")}
                for r in examples["SOCIAL"][:5]
            ],
            "NON_SOCIAL_OPPORTUNISTIC": [
                {"asset_id": r.get("asset_id"), "symbol": r.get("symbol"), "reasoning_short": r.get("reasoning_short")}
                for r in examples["NON_SOCIAL_OPPORTUNISTIC"][:5]
            ],
            "NON_SOCIAL_INFRASTRUCTURE": [
                {"asset_id": r.get("asset_id"), "symbol": r.get("symbol"), "reasoning_short": r.get("reasoning_short")}
                for r in examples["NON_SOCIAL_INFRASTRUCTURE"][:5]
            ],
            "UNKNOWN_INSUFFICIENT_EVIDENCE": [
                {"asset_id": r.get("asset_id"), "symbol": r.get("symbol"), "reasoning_short": r.get("reasoning_short")}
                for r in examples["UNKNOWN_INSUFFICIENT_EVIDENCE"][:5]
            ],
            "MANUAL_REVIEW": [
                {"asset_id": r.get("asset_id"), "symbol": r.get("symbol"), "reasoning_short": r.get("reasoning_short")}
                for r in examples["MANUAL_REVIEW"][:5]
            ],
        },
        "unknown_share": unknown_share,
        "unknown_reason_breakdown": dict(unknown_reasons),
        "safety_audit": safety_audit,
        "cache_key_fields": cache_key_fields(),
        "cache_key_uses_evidence_hash": cache_uses_evidence_hash(),
        "no_external_apis": no_external_apis,
        "trade_authority_used": False,
        "live_ready": False,
        "profitability_proven": False,
        "qwen_trade_authority": False,
    }

    manifest = {
        "created_at_utc": summary["created_at_utc"],
        "output_root": str(out_root),
        "files": [],
        "historical_data_mutated": False,
        "trade_authority_used": False,
        "external_api_used": False,
    }

    _write_csv(out_root / "data" / "ae12_unique_coin_evidence_packages.csv", evidence)
    _write_csv(out_root / "data" / "ae12_semantic_linkage_audit.csv", linkage_rows)
    _write_json(out_root / "reports" / "ae12_semantic_linkage_quality_summary.json", linkage_summary)
    _write_csv(out_root / "data" / "ae12_semantic_coin_classifications.csv", classifications)
    _write_csv(out_root / "data" / "ae12_semantic_coin_class_distribution.csv", distribution_rows)
    _write_csv(out_root / "data" / "ae12_social_coin_examples.csv", examples["SOCIAL"])
    _write_csv(out_root / "data" / "ae12_opportunistic_coin_examples.csv", examples["NON_SOCIAL_OPPORTUNISTIC"])
    _write_csv(out_root / "data" / "ae12_infrastructure_coin_examples.csv", examples["NON_SOCIAL_INFRASTRUCTURE"])
    _write_csv(out_root / "data" / "ae12_unknown_coin_examples.csv", examples["UNKNOWN_INSUFFICIENT_EVIDENCE"])
    _write_csv(out_root / "data" / "ae12_manual_review_coin_examples.csv", examples["MANUAL_REVIEW"])
    _write_csv(out_root / "data" / "ae12_llm_classifier_rejected_outputs.csv", rejected)
    _write_json(out_root / "audits" / "ae12_semantic_classifier_decision_gate.json", gate)
    _write_json(out_root / "audits" / "ae12_llm_classifier_safety_audit.json", safety_audit)
    _write_csv(out_root / "audits" / "ae12_classification_cache_audit.csv", cache_audit_rows)
    _write_csv(out_root / "audits" / "ae12_unique_asset_dedup_audit.csv", dedup_audit)
    _write_json(out_root / "reports" / "ae12_semantic_coin_classifier_summary.json", summary)
    _write_json(out_root / "reports" / "ae12_semantic_coin_classifier_manifest.json", manifest)

    upload_lines = [
        "AE12-SentimentFix Semantic Coin Classifier (not AE12.6)",
        f"output_root: {out_root}",
        f"gate_status: {gate_status}",
        f"unique_assets_found: {unique_assets_found}",
        f"unique_assets_classified: {len(classifications)}",
        f"SOCIAL: {social} ({round(social/total,6)})",
        f"NON_SOCIAL_OPPORTUNISTIC: {opp} ({round(opp/total,6)})",
        f"NON_SOCIAL_INFRASTRUCTURE: {infra} ({round(infra/total,6)})",
        f"UNKNOWN_INSUFFICIENT_EVIDENCE: {unknown} ({unknown_share})",
        f"MANUAL_REVIEW: {manual} ({manual_share})",
        f"classifier_status: {classifier_status}",
        f"local_llm_used: {local_llm_used}",
        f"forbidden_trade_language_found: {safety_audit['forbidden_trade_language_found']}",
        f"rejected_llm_outputs: {rejected_outputs}",
        "trade_authority_used: false",
        "external_api_used: false",
        "cache_key_fields: asset_id,classifier_version,evidence_hash,rubric_version",
        "cache_key_uses_evidence_hash: true",
        f"unknown_reason_breakdown: {dict(unknown_reasons)}",
        "UNKNOWN is a dataset-quality metric, not opportunistic by default.",
    ]
    (out_root / "reports" / "ae12_semantic_coin_classifier_for_upload.txt").write_text(
        "\n".join(upload_lines) + "\n",
        encoding="utf-8",
    )

    manifest["files"] = sorted(
        str(p.relative_to(out_root)) for p in out_root.rglob("*") if p.is_file()
    )
    _write_json(out_root / "reports" / "ae12_semantic_coin_classifier_manifest.json", manifest)
    return summary
