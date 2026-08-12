"""AE12.7 UI/API summary builder for Intelligent Agent Layer panel."""

from __future__ import annotations

from collections import Counter
from typing import Any


def build_ui_summary(
    *,
    records: list[dict[str, Any]],
    policy_snapshot: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    """Read-only payload for GET /api/ae12/agents/ui-summary and UI panel."""
    by_type: Counter[str] = Counter()
    by_status: Counter[str] = Counter()
    warnings: list[str] = []
    soft_vetoes: list[str] = []
    linked_paper = 0
    linked_missed = 0
    linked_opp = 0

    qwen_status = "NOT_SEEN"
    gemini_status = "NOT_SEEN"
    helius_status = "NOT_SEEN"
    rss_status = "NOT_SEEN"
    semantic_status = "NOT_SEEN"

    for r in records:
        t = str(r.get("agent_type") or "")
        s = str(r.get("agent_status") or "")
        by_type[t] += 1
        by_status[s] += 1
        warnings.extend(list(r.get("warnings") or [])[:3])
        soft_vetoes.extend(list(r.get("soft_veto_flags") or [])[:3])
        link = r.get("linkage") if isinstance(r.get("linkage"), dict) else {}
        if link.get("linked_to_paper_trade"):
            linked_paper += 1
        if link.get("linked_to_missed_winner"):
            linked_missed += 1
        if link.get("linked_to_opportunity"):
            linked_opp += 1
        if t == "QWEN_LOCAL_MEMO":
            qwen_status = s
        elif t == "GEMINI_SELECTIVE_AUDIT":
            gemini_status = s
        elif t == "HELIUS_READONLY_ENRICHMENT":
            helius_status = s
        elif t == "RSS_CONTEXT":
            rss_status = s
        elif t == "SEMANTIC_CLASSIFICATION":
            semantic_status = s

    return {
        "phase": "AE12.7",
        "panel": "Intelligent Agent Layer",
        "mode": policy_snapshot.get("mode"),
        "qwen_memo_status": qwen_status,
        "gemini_audit_status": gemini_status,
        "helius_enrichment_status": helius_status,
        "rss_context_status": rss_status,
        "semantic_classification_status": semantic_status,
        "warnings_blockers": sorted(set(warnings))[:40],
        "soft_veto_recommendations": sorted(set(soft_vetoes))[:20],
        "strict_vs_exploration_note": "Agent layer explains strict vs exploration; does not override AE10/AE11 gates.",
        "linkage": {
            "linked_to_paper_trade_count": linked_paper,
            "linked_to_missed_winner_count": linked_missed,
            "linked_to_opportunity_count": linked_opp,
        },
        "counts_by_type": dict(by_type),
        "counts_by_status": dict(by_status),
        "record_count": len(records),
        "trade_authority_used": False,
        "wallet_status": "NOT_CONFIGURED",
        "live_ready": False,
        "profitability_proven": False,
        "ae12_closed": False,
        "gate_status": gate.get("status"),
        "decision_gate": gate,
        "read_only": True,
    }


def build_status_payload(
    *,
    latest_root: str | None,
    ui_summary: dict[str, Any],
    policy_snapshot: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": "AE12.7",
        "latest_ae12_7_output_root": latest_root,
        "operating_mode": policy_snapshot.get("mode"),
        "qwen_memo_status": ui_summary.get("qwen_memo_status"),
        "gemini_audit_status": ui_summary.get("gemini_audit_status"),
        "helius_enrichment_status": ui_summary.get("helius_enrichment_status"),
        "rss_context_status": ui_summary.get("rss_context_status"),
        "semantic_classification_status": ui_summary.get("semantic_classification_status"),
        "trade_authority_used": False,
        "wallet_status": "NOT_CONFIGURED",
        "live_ready": False,
        "profitability_proven": False,
        "ae12_closed": False,
        "external_api_used": bool(policy_snapshot.get("external_api_used")),
        "no_real_wallet": True,
        "paper_demo_ok": True,
    }
