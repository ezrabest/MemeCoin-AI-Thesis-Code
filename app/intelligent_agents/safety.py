"""AE12.7 safety helpers — reject trade-authority language; never grant execution."""

from __future__ import annotations

import re
from typing import Any

FORBIDDEN_TRADE_PATTERNS = [
    re.compile(r"\bBUY\b", re.I),
    re.compile(r"\bSELL\b", re.I),
    re.compile(r"\bEXECUTE\b", re.I),
    re.compile(r"\bAPPROVE[_\s-]?TRADE\b", re.I),
    re.compile(r"\bLIVE[_\s-]?BUY\b", re.I),
    re.compile(r"\bPAPER[_\s-]?BUY\b", re.I),
    re.compile(r"\bSUBMIT[_\s-]?TRANSACTION\b", re.I),
    re.compile(r"\bSIGN[_\s-]?TRANSACTION\b", re.I),
    re.compile(r"\bCONNECT[_\s-]?WALLET\b", re.I),
    re.compile(r"\bLIVE[_\s-]?TRADING\s+APPROVED\b", re.I),
]

_NEGATION = re.compile(
    r"\b(not|no|never|without|forbid|forbidden|disallow|blocked|audit-only|do\s+not)\b",
    re.I,
)


def find_forbidden_trade_language(text: str | None) -> list[str]:
    """Return forbidden trade-authority phrases found outside negation context."""
    if not text:
        return []
    hits: list[str] = []
    for pattern in FORBIDDEN_TRADE_PATTERNS:
        for m in pattern.finditer(text):
            prefix = text[max(0, m.start() - 40) : m.start()]
            if _NEGATION.search(prefix):
                continue
            hits.append(m.group(0))
    return hits


def reject_authority_language(text: str | None) -> dict[str, Any]:
    """Evaluate LLM/agent text for prohibited execution wording."""
    hits = find_forbidden_trade_language(text)
    rejected = bool(hits)
    return {
        "forbidden_trade_language_found": hits,
        "rejection_status": "REJECTED_TRADE_AUTHORITY_LANGUAGE" if rejected else "NONE",
        "output_used_after_rejection": False,
        "safety_status": "PASS_REJECTIONS_ENFORCED" if rejected else "PASS_NO_FORBIDDEN_LANGUAGE",
        "trade_authority_used": False,
        "decision_effect_allowed": not rejected,
    }


def assert_no_authority_mutation(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, Any]:
    """Verify agent processing did not mutate BUY/SELL/live authority fields."""
    before = before or {}
    after = after or {}
    keys = (
        "action",
        "trade_action",
        "buy_sell",
        "live_authority",
        "trade_authority",
        "execution_intent",
        "wallet_connected",
    )
    mutated = []
    for k in keys:
        if k in after and after.get(k) != before.get(k):
            mutated.append(k)
    return {
        "authority_fields_mutated": mutated,
        "pass": len(mutated) == 0,
        "trade_authority_used": False,
    }
