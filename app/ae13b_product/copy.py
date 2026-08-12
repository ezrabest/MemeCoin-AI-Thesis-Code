"""AE13B human-facing copy helpers - replace internal gate jargon."""

from __future__ import annotations

COPY_MAP: dict[str, str] = {
    "FORWARD_EVIDENCE_READY_FOR_REPORTING": (
        "Research evidence is ready for reporting, but not a live trading signal."
    ),
    "UNKNOWN_UNRESOLVED": "Unknown - not enough evidence yet.",
    "price_price_stale": "Price data too old for a confident trade.",
    "PRICE_STALE": "Price data too old for a confident trade.",
    "NO_TRADE_AUTHORITY": "AI can explain, but cannot place trades.",
    "NOT_SUBMITTED_NO_WALLET": "No wallet connected - nothing submitted.",
    "SOCIAL_CONFIRMED": "Socially confirmed",
    "OPPORTUNISTIC_CONFIRMED": "Opportunistic (confirmed)",
    "OP.SUSPECTED": "Possibly opportunistic (not confirmed)",
    "DEMO_ACCEPTANCE_MODE": "Demo acceptance / test trade only",
    "PAPER_EXPLORATION_ONLY": "Paper exploration only - not live approved",
}


def humanize(code: str | None, *, default: str | None = None) -> str:
    if not code:
        return default or "-"
    key = str(code).strip()
    if key in COPY_MAP:
        return COPY_MAP[key]
    # soften common snake/SCREAMING patterns for UI
    if key.isupper() and "_" in key:
        return key.replace("_", " ").title()
    return key


def semantic_label_human(family: str | None) -> str:
    fam = str(family or "UNKNOWN_INSUFFICIENT_EVIDENCE").upper()
    mapping = {
        "SOCIAL_CONFIRMED": "Socially confirmed",
        "SOCIALLY_MOTIVATED": "Legacy social cluster (diagnostic)",
        "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED": "Opportunistic (confirmed)",
        "OPPORTUNISTIC_CONFIRMED": "Opportunistic (confirmed)",
        "OPPORTUNISTIC_SPECULATIVE": "Legacy opportunistic cluster (diagnostic)",
        "OPPORTUNISTIC_SUSPECTED": "Possibly opportunistic (not confirmed)",
        "OP.SUSPECTED": "Possibly opportunistic (not confirmed)",
        "NON_SOCIAL_INFRASTRUCTURE_CONFIRMED": "Infrastructure / non-meme utility",
        "INFRASTRUCTURE_CONFIRMED": "Infrastructure / non-meme utility",
        "UNKNOWN_INSUFFICIENT_EVIDENCE": "Unknown - not enough evidence yet.",
        "UNKNOWN_UNRESOLVED": "Unknown - not enough evidence yet.",
        "NEEDS_REVIEW": "Needs review",
    }
    return mapping.get(fam, humanize(fam))
