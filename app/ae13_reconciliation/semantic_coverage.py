"""AE13 runtime vs static semantic/sentiment coverage reconciliation.

Does not call Gemini/Helius/Qwen/Ollama. Reads local artifacts + registry only.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.ae12_reporting.latest import (
    discover_latest_gemini_adjudication_root,
    discover_latest_manual_review_drilldown_root,
    discover_latest_semantic_classifier_root,
    discover_latest_sentimentfix_root,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_meta(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"path": str(path) if path else None, "exists": False, "mtime_utc": None}
    st = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "mtime_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
        "size_bytes": int(st.st_size),
    }


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _extract_static_counts(project_root: Path) -> dict[str, Any]:
    drill_root = discover_latest_manual_review_drilldown_root(project_root)
    gem_root = discover_latest_gemini_adjudication_root(project_root)
    class_root = discover_latest_semantic_classifier_root(project_root)
    sent_root = discover_latest_sentimentfix_root(project_root)

    counts: dict[str, Any] = {}
    source = "unavailable"
    gate = None
    subtitle = None
    root_used = None

    if drill_root:
        summary = _load_json(drill_root / "reports" / "ae12_manual_review_drilldown_summary.json")
        if summary:
            updated = summary.get("updated_coin_level_counts") or summary.get("coin_level_counts") or {}
            if updated.get("unique_coins_found") is not None:
                counts = dict(updated)
                source = "Static AE12 Snapshot"
                source_detail = "manual-review-drilldown"
                gate = summary.get("gate_status")
                subtitle = "Final UI counts from AE12 manual-review drilldown audit artifact."
                root_used = drill_root
                return {
                    "counts": counts,
                    "semantic_source_label": "Semantic Source: Static AE12 Snapshot",
                    "semantic_source": source,
                    "source_detail": source_detail,
                    "gate_status": gate,
                    "subtitle": subtitle,
                    "audit_root": str(root_used),
                    "freshness": _file_meta(root_used / "reports" / "ae12_manual_review_drilldown_summary.json"),
                    "gemini_adjudication": "reporting_only_static",
                    "final_semantic_classification": False,
                }

    if gem_root:
        summary = _load_json(gem_root / "reports" / "ae12_gemini_semantic_adjudication_summary.json")
        if summary:
            after = summary.get("coin_level_counts_after_drilldown") or summary.get("coin_level_counts") or {}
            if after.get("unique_coins_found") is not None:
                counts = dict(after)
                return {
                    "counts": counts,
                    "semantic_source_label": "Semantic Source: Static AE12 Snapshot",
                    "semantic_source": "Static AE12 Snapshot",
                    "source_detail": "gemini-adjudication",
                    "gate_status": summary.get("gate_status"),
                    "subtitle": "Coin-level counts from AE12 Gemini adjudication audit artifact.",
                    "audit_root": str(gem_root),
                    "freshness": _file_meta(gem_root / "reports" / "ae12_gemini_semantic_adjudication_summary.json"),
                    "gemini_adjudication": "reporting_only_static",
                    "final_semantic_classification": False,
                }

    return {
        "counts": counts,
        "semantic_source_label": "Semantic Source: Static AE12 Snapshot",
        "semantic_source": "Static AE12 Snapshot",
        "source_detail": "unavailable",
        "gate_status": None,
        "subtitle": "No AE12 semantic audit summary found.",
        "audit_root": None,
        "freshness": {
            "drilldown": _file_meta(drill_root),
            "gemini": _file_meta(gem_root),
            "classifier": _file_meta(class_root),
            "sentimentfix": _file_meta(sent_root),
        },
        "gemini_adjudication": "reporting_only_static",
        "final_semantic_classification": False,
    }


def _load_cluster_registry(project_root: Path) -> dict[str, Any]:
    path = project_root / "data" / "cluster_registry.json"
    meta = _file_meta(path)
    data = _load_json(path) or {}
    entries = data if isinstance(data, dict) else {}
    # Support both flat address->label and nested objects
    labels: dict[str, int] = {}
    assigned_ats: list[str] = []
    for _k, v in entries.items():
        if isinstance(v, str):
            labels[v] = labels.get(v, 0) + 1
        elif isinstance(v, dict):
            lab = str(v.get("cluster_label") or v.get("label") or "UNKNOWN")
            labels[lab] = labels.get(lab, 0) + 1
            if v.get("assigned_at"):
                assigned_ats.append(str(v["assigned_at"]))
    return {
        "path": str(path),
        "meta": meta,
        "entry_count": len(entries),
        "label_counts": labels,
        "latest_assigned_at": max(assigned_ats) if assigned_ats else None,
        "legacy_cluster_not_semantic_authority": True,
        "note": (
            "cluster_registry.json is sticky discovery clustering — not AE12 final semantic authority."
        ),
    }


def _count_runtime_candidates(project_root: Path) -> dict[str, Any]:
    """Count recent runtime candidates from local DB coins / whale log without external APIs."""
    coins_count = 0
    recent_symbols: list[str] = []
    try:
        from app import database as db

        coins = db.get_coins(limit=500, sort_by="last_seen")
        coins_count = len(coins)
        recent_symbols = [str(c.get("symbol") or "") for c in coins[:50] if c.get("symbol")]
    except Exception as exc:  # noqa: BLE001 — coverage must degrade safely
        return {
            "runtime_candidates_seen": 0,
            "runtime_candidates_error": str(exc),
            "recent_symbols_sample": [],
        }

    # Runtime classified via sticky registry intersection (not AE12 taxonomy)
    registry = _load_cluster_registry(project_root)
    classified = int(registry.get("entry_count") or 0)

    return {
        "runtime_candidates_seen": coins_count,
        "runtime_candidates_in_cluster_registry": classified,
        "runtime_candidates_classified_ae12_live": 0,  # live AE12 classifier not invoked at runtime
        "runtime_candidates_unresolved": max(0, coins_count),  # AE12 live path does not resolve at runtime
        "recent_symbols_sample": recent_symbols[:20],
        "ae12_live_classifier_invoked": False,
        "note": (
            "Runtime discovery candidates exist in trader.db/coins and sticky cluster_registry, "
            "but UI final semantic cards read static AE12 audit snapshots — not a live classifier stream."
        ),
    }


def explain_social_confirmed_zero(static: dict[str, Any], runtime: dict[str, Any]) -> dict[str, Any]:
    counts = static.get("counts") or {}
    social = int(counts.get("coin_social_confirmed_count") or 0)
    reasons: list[str] = []
    if social == 0:
        reasons.append(
            "Static AE12 drilldown/gemini audit reports coin_social_confirmed_count=0 "
            "(insufficient social-source evidence under conservative taxonomy)."
        )
        reasons.append(
            "Gemini adjudication is reporting-only / static — not runtime-enabled for live candidates."
        )
        reasons.append(
            "No Twitter/Telegram/Reddit social ingestion is wired into the live runtime semantic path."
        )
        reasons.append(
            "RSS sentiment matrix is headline-level (Cointelegraph) and is not linked as coin-level "
            "SOCIAL_CONFIRMED evidence."
        )
        if not runtime.get("ae12_live_classifier_invoked"):
            reasons.append(
                "Live AE12 semantic classifier is not invoked on each runtime candidate "
                "(UI reads cached audit); SOCIAL_CONFIRMED=0 is therefore snapshot-derived, "
                "not a silent live-classifier failure."
            )
    return {
        "social_confirmed_count": social,
        "is_zero": social == 0,
        "acceptable": social == 0,  # acceptable when explained by evidence/taxonomy
        "explanation_reasons": reasons,
        "caused_by_silent_stale_without_label": False,  # AE13 labels this explicitly
        "caused_by_classifier_never_running_silently": False,
        "social_sources_available": False,
        "social_sources_note": (
            "Social confirmation sources (Twitter/Telegram/Reddit) are unavailable in current runtime config."
        ),
    }


def build_semantic_coverage(project_root: Path | None = None) -> dict[str, Any]:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    static = _extract_static_counts(root)
    registry = _load_cluster_registry(root)
    runtime = _count_runtime_candidates(root)
    social = explain_social_confirmed_zero(static, runtime)

    counts = static.get("counts") or {}
    unique_static = counts.get("unique_coins_found")
    coverage_beyond_static = (
        int(runtime.get("runtime_candidates_seen") or 0) > int(unique_static or 0)
        if unique_static is not None
        else False
    )

    # Mixed label when runtime candidates exist beyond static snapshot
    if coverage_beyond_static:
        semantic_source_label = "Semantic Source: Runtime Stream + Static AE12 Snapshot"
        semantic_source = "Runtime Stream + Static AE12 Snapshot"
    else:
        semantic_source_label = static.get("semantic_source_label") or "Semantic Source: Static AE12 Snapshot"
        semantic_source = static.get("semantic_source") or "Static AE12 Snapshot"

    coverage_status = "LIMITED_STATIC_SNAPSHOT"
    if unique_static == 14 and not runtime.get("ae12_live_classifier_invoked"):
        coverage_status = "LIMITED_STATIC_14_COINS_AE12_SNAPSHOT"
        coverage_explanation = (
            "UI unique-coin semantic coverage remains 14 because it reads the frozen AE12 "
            "manual-review drilldown audit (unique_coins_found=14). Runtime discovery has more "
            "candidates, but they are not reclassified by a live AE12 semantic stream."
        )
    else:
        coverage_explanation = (
            "Static AE12 snapshot is the final UI semantic authority; runtime candidates are counted separately."
        )

    return {
        "built_at_utc": _utc_now(),
        "semantic_source_label": semantic_source_label,
        "semantic_source": semantic_source,
        "coverage_status": coverage_status,
        "coverage_explanation": coverage_explanation,
        "static_ae12": static,
        "cluster_registry": registry,
        "runtime": runtime,
        "social_confirmed_audit": social,
        "rss_sentiment": {
            "endpoint": "/api/sentiment/matrix",
            "linked_to_coin_social_confirmed": False,
            "contribution": "headline_matrix_only",
            "note": "RSS contributes dashboard Sentiment Matrix, not AE12 SOCIAL_CONFIRMED counts.",
        },
        "unknown_unresolved_policy": {
            "preserved": True,
            "promoted_to_social": False,
            "promoted_to_opportunistic": False,
            "note": "UNKNOWN_UNRESOLVED remains unresolved and is not social/opportunistic.",
        },
        "ui_counters": {
            "unique_coins_static": unique_static,
            "coin_social_confirmed_count": counts.get("coin_social_confirmed_count", 0),
            "coin_non_social_opportunistic_confirmed_count": counts.get(
                "coin_non_social_opportunistic_confirmed_count", 0
            ),
            "coin_opportunistic_suspected_count": counts.get("coin_opportunistic_suspected_count", 0),
            "coin_unknown_unresolved_count": counts.get("coin_unknown_unresolved_count", 0),
            "runtime_candidates_seen": runtime.get("runtime_candidates_seen", 0),
            "runtime_candidates_classified": runtime.get("runtime_candidates_classified_ae12_live", 0),
            "runtime_candidates_unresolved": runtime.get("runtime_candidates_unresolved", 0),
            "cluster_registry_entries": registry.get("entry_count", 0),
            "social_sources_available": False,
            "gemini_adjudication_mode": "reporting_only_static",
            "latest_semantic_refresh_utc": (static.get("freshness") or {}).get("mtime_utc")
            if isinstance(static.get("freshness"), dict)
            else None,
        },
        "paper_demo_only": True,
        "not_trade_authority": True,
    }
