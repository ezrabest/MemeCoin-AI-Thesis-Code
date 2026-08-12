"""Semantic signal family derivation from explicit evidence only (no opportunistic default)."""

from __future__ import annotations

from typing import Any

from .types import SEMANTIC_SIGNAL_FAMILIES, null_safe_str

SOCIAL_MARKERS = (
    "social",
    "community",
    "hype",
    "reddit",
    "twitter",
    "telegram",
    "buzz",
    "influencer",
    "discord",
)
NEWS_MARKERS = ("rss", "news", "article", "headline", "press", "media")
WHALE_MARKERS = ("whale", "holder")
ONCHAIN_MARKERS = ("onchain", "on-chain", "on_chain", "wallet", "transfer")
LIQUIDITY_MARKERS = ("liquidity", "volume", "pool")
MOMENTUM_MARKERS = ("momentum", "pump", "dump", "breakout", "price_change", "trend")
LLM_MARKERS = ("qwen", "gemini", "ollama", "llm_context", "narrative", "llm_")

TEXT_KEYS = (
    "title",
    "text",
    "body",
    "content",
    "summary",
    "reasoning",
    "narrative",
    "llm_context",
    "llm_summary",
    "llm_verdict",
    "description",
    "source_family",
    "context_family",
    "semantic_signal_family",
    "semantic_family",
    "signal_family",
    "ae8_context_status",
    "reason_for_no_trade",
    "reason_not_traded",
)


def _row_text_blob(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for key in TEXT_KEYS:
        val = row.get(key)
        if val is None or val == "":
            continue
        parts.append(str(val).lower())
    # Also scan nested context snippets lightly
    for key in ("context", "llm_audit", "sentiment"):
        val = row.get(key)
        if isinstance(val, dict):
            for vv in val.values():
                if vv is not None and not isinstance(vv, (dict, list)):
                    parts.append(str(vv).lower())
    return " ".join(parts)


def detect_semantic_families_from_markers(row: dict[str, Any]) -> list[str]:
    blob = _row_text_blob(row)
    found: list[str] = []
    if any(m in blob for m in SOCIAL_MARKERS):
        found.append("SOCIAL")
    if any(m in blob for m in NEWS_MARKERS):
        found.append("NEWS")
    if any(m in blob for m in WHALE_MARKERS):
        found.append("WHALE")
    if any(m in blob for m in ONCHAIN_MARKERS):
        found.append("ONCHAIN")
    if any(m in blob for m in LIQUIDITY_MARKERS):
        found.append("LIQUIDITY")
    if any(m in blob for m in MOMENTUM_MARKERS):
        found.append("PRICE_MOMENTUM")
    if any(m in blob for m in LLM_MARKERS):
        found.append("LLM_CONTEXT")
    # Deduplicate preserving order
    out: list[str] = []
    for f in found:
        if f not in out:
            out.append(f)
    return out


def extract_explicit_semantic_raw(row: dict[str, Any]) -> str | None:
    raw = (
        row.get("semantic_signal_family")
        or row.get("semantic_family")
        or row.get("signal_family")
        or row.get("source_family")
        or None
    )
    if raw is None:
        return None
    text = str(raw).strip()
    return text if text else None


def derive_semantic_signal_family(row: dict[str, Any]) -> tuple[str, str, float, str]:
    """
    Returns (family, source, confidence, reason).
    Missing evidence -> UNKNOWN. Never returns OPPORTUNISTIC.
    Sticky/legacy cluster alone is NOT sufficient for SOCIAL/NEWS etc.
    """
    explicit = extract_explicit_semantic_raw(row)
    if explicit:
        upper = explicit.upper()
        if upper in SEMANTIC_SIGNAL_FAMILIES:
            if upper in {"UNKNOWN", "UNCLASSIFIED"}:
                return upper, "explicit_field", 0.9, f"explicit semantic field={upper}"
            return upper, "explicit_field", 0.95, f"explicit semantic field={upper}"
        if "SOCIAL" in upper:
            return "SOCIAL", "explicit_field", 0.9, "explicit field contains SOCIAL"
        if "NEWS" in upper or "RSS" in upper:
            return "NEWS", "explicit_field", 0.9, "explicit field contains NEWS"
        if "MIXED" in upper:
            return "MIXED", "explicit_field", 0.9, "explicit field MIXED"

    markers = detect_semantic_families_from_markers(row)
    # SOCIALLY_MOTIVATED in cluster_label alone must not invent SOCIAL semantic without text markers
    # unless explicit SOCIALLY_MOTIVATED and user wants narrative - spec says semantic from social evidence.
    # Only treat SOCIALLY_MOTIVATED as weak SOCIAL when present with semantic markers OR as legacy hint
    # Spec: "sticky contract-level cluster_label must not be used as final semantic_signal_family"
    # So we ignore cluster_label for semantic entirely except when already covered by markers.

    if len(markers) > 1:
        return (
            "MIXED",
            "text_markers",
            0.7,
            f"multiple semantic markers: {','.join(markers)}",
        )
    if len(markers) == 1:
        return markers[0], "text_markers", 0.75, f"marker family={markers[0]}"

    # No evidence
    return "UNKNOWN", "none", 0.0, "no explicit semantic evidence; UNKNOWN (not OPPORTUNISTIC)"
