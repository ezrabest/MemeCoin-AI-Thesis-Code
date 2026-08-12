"""AE19 authority/safety language filter — no trade authority."""

from __future__ import annotations

import re
from typing import Any

from app.llm_operational.schema import (
    FAIL_FORBIDDEN_AUTHORITY_LANGUAGE,
    FAIL_LIVE_APPROVAL_LANGUAGE,
    FAIL_RISK_OVERRIDE_LANGUAGE,
    FAIL_TRADE_AUTHORITY_LANGUAGE,
    FAIL_WALLET_ACCESS,
    OUTPUT_REJECTED_AND_QUARANTINED,
    PASS_NO_LIVE_APPROVAL,
    PASS_NO_RISK_OVERRIDE,
    PASS_NO_TRADE_AUTHORITY,
    PASS_NO_WALLET_ACCESS,
    TASK_SKIPPED_SAFETY,
)

# Exact forbidden concepts from AE19 hard rule
FORBIDDEN_PHRASES: tuple[str, ...] = (
    "approve live trade",
    "approved for live trade",
    "execute live buy",
    "execute live sell",
    "live buy approved",
    "live sell approved",
    "override risk gate",
    "ignore RiskGuard",
    "ignore GateKeeper",
    "bypass safety",
    "safe to trade live",
    "guaranteed profit",
    "guaranteed return",
    "wallet signing",
    "sign transaction",
    "submit transaction",
    "send live transaction",
    "connect wallet",
    "private key",
    "ignore stop loss",
    "override drawdown",
    "trade authority granted",
)

ALLOWED_LANGUAGE_TAGS: tuple[str, ...] = (
    "WATCH",
    "REVIEW",
    "EXPLAIN",
    "RISK_NOTE",
    "CONTEXT_SUMMARY",
    "RESEARCH_ONLY",
    "PAPER_DEMO_OBSERVATION",
    "NOT_TRADE_AUTHORITY",
)

_NEGATION = re.compile(
    r"\b(not|no|never|without|forbid|forbidden|disallow|blocked|audit-only|do\s+not|must\s+not)\b",
    re.I,
)

_CATEGORY_MAP: dict[str, str] = {
    "approve live trade": FAIL_LIVE_APPROVAL_LANGUAGE,
    "approved for live trade": FAIL_LIVE_APPROVAL_LANGUAGE,
    "execute live buy": FAIL_LIVE_APPROVAL_LANGUAGE,
    "execute live sell": FAIL_LIVE_APPROVAL_LANGUAGE,
    "live buy approved": FAIL_LIVE_APPROVAL_LANGUAGE,
    "live sell approved": FAIL_LIVE_APPROVAL_LANGUAGE,
    "safe to trade live": FAIL_LIVE_APPROVAL_LANGUAGE,
    "override risk gate": FAIL_RISK_OVERRIDE_LANGUAGE,
    "ignore riskguard": FAIL_RISK_OVERRIDE_LANGUAGE,
    "ignore gatekeeper": FAIL_RISK_OVERRIDE_LANGUAGE,
    "bypass safety": FAIL_RISK_OVERRIDE_LANGUAGE,
    "ignore stop loss": FAIL_RISK_OVERRIDE_LANGUAGE,
    "override drawdown": FAIL_RISK_OVERRIDE_LANGUAGE,
    "wallet signing": FAIL_WALLET_ACCESS,
    "sign transaction": FAIL_WALLET_ACCESS,
    "submit transaction": FAIL_WALLET_ACCESS,
    "send live transaction": FAIL_WALLET_ACCESS,
    "connect wallet": FAIL_WALLET_ACCESS,
    "private key": FAIL_WALLET_ACCESS,
    "trade authority granted": FAIL_TRADE_AUTHORITY_LANGUAGE,
    "guaranteed profit": FAIL_FORBIDDEN_AUTHORITY_LANGUAGE,
    "guaranteed return": FAIL_FORBIDDEN_AUTHORITY_LANGUAGE,
}


def find_forbidden_authority_language(text: str | None) -> list[dict[str, str]]:
    """Return forbidden phrase hits outside negation context."""
    if not text:
        return []
    hits: list[dict[str, str]] = []
    lower = text.lower()
    for phrase in FORBIDDEN_PHRASES:
        needle = phrase.lower()
        start = 0
        while True:
            idx = lower.find(needle, start)
            if idx < 0:
                break
            prefix = text[max(0, idx - 40) : idx]
            if _NEGATION.search(prefix):
                start = idx + len(needle)
                continue
            category = _CATEGORY_MAP.get(needle, FAIL_FORBIDDEN_AUTHORITY_LANGUAGE)
            hits.append({"phrase": phrase, "category": category, "offset": str(idx)})
            start = idx + len(needle)
    return hits


def scan_output_safety(text: str | None) -> dict[str, Any]:
    """Scan textual LLM/mock output for forbidden authority language."""
    hits = find_forbidden_authority_language(text)
    phrases = [h["phrase"] for h in hits]
    categories = sorted({h["category"] for h in hits})
    rejected = bool(hits)
    primary = categories[0] if categories else PASS_NO_TRADE_AUTHORITY
    if rejected and primary not in {
        FAIL_TRADE_AUTHORITY_LANGUAGE,
        FAIL_LIVE_APPROVAL_LANGUAGE,
        FAIL_RISK_OVERRIDE_LANGUAGE,
        FAIL_WALLET_ACCESS,
        FAIL_FORBIDDEN_AUTHORITY_LANGUAGE,
    }:
        primary = FAIL_FORBIDDEN_AUTHORITY_LANGUAGE

    return {
        "forbidden_language_found": rejected,
        "forbidden_language_hits": phrases,
        "forbidden_categories": categories,
        "safety_status": OUTPUT_REJECTED_AND_QUARANTINED if rejected else primary if not rejected else FAIL_FORBIDDEN_AUTHORITY_LANGUAGE,
        "pass_statuses": [
            PASS_NO_TRADE_AUTHORITY,
            PASS_NO_LIVE_APPROVAL,
            PASS_NO_RISK_OVERRIDE,
            PASS_NO_WALLET_ACCESS,
        ]
        if not rejected
        else [],
        "trade_authority_used": False,
        "live_trading_approved": False,
        "risk_override_used": False,
        "wallet_accessed": False,
        "downstream_eligible": not rejected,
        "downstream_quarantined": rejected,
        "safety_failed": rejected,
        "accepted_for_downstream": not rejected,
        "rejection_task_status": TASK_SKIPPED_SAFETY if rejected else None,
    }


def apply_safety_to_record(record: dict[str, Any], text: str | None = None) -> dict[str, Any]:
    """Mutate/return record with safety fields applied. Never grants authority."""
    out = dict(record)
    scan_text = text if text is not None else str(out.get("output_text") or out.get("output_summary") or "")
    safety = scan_output_safety(scan_text)

    out["trade_authority_used"] = False
    out["live_trading_approved"] = False
    out["risk_override_used"] = False
    out["wallet_accessed"] = False
    out["forbidden_language_hits"] = safety["forbidden_language_hits"]
    out["safety_failed"] = safety["safety_failed"]

    if safety["forbidden_language_found"]:
        out["safety_status"] = FAIL_FORBIDDEN_AUTHORITY_LANGUAGE
        out["downstream_eligible"] = False
        out["downstream_quarantined"] = True
        out["accepted_for_downstream"] = False
        out["task_status"] = TASK_SKIPPED_SAFETY
        out["failure_reason"] = "FORBIDDEN_AUTHORITY_LANGUAGE"
        # Preserve raw for audit if already set
        if not out.get("raw_response_preserved"):
            out["raw_response_preserved"] = scan_text
        out["output_text"] = ""
        out["output_summary"] = "OUTPUT_REJECTED_AND_QUARANTINED"
    else:
        # Keep existing status; mark pass authority flags
        if not out.get("safety_status") or str(out.get("safety_status")).startswith("PASS_"):
            out["safety_status"] = PASS_NO_TRADE_AUTHORITY
        out.setdefault("allowed_language_tags", list(ALLOWED_LANGUAGE_TAGS))

    return out


def authority_boundary_snapshot() -> dict[str, Any]:
    return {
        "trade_authority_used": False,
        "live_trading_approved": False,
        "risk_override_used": False,
        "wallet_accessed": False,
        "transaction_signing": False,
        "profitability_claimed": False,
        "live_readiness_claimed": False,
        "gatekeeper_override": False,
        "riskguard_override": False,
        "pass_no_trade_authority": PASS_NO_TRADE_AUTHORITY,
        "pass_no_live_approval": PASS_NO_LIVE_APPROVAL,
        "pass_no_risk_override": PASS_NO_RISK_OVERRIDE,
        "pass_no_wallet_access": PASS_NO_WALLET_ACCESS,
    }
