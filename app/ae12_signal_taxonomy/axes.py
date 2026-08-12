"""Taxonomy axis definitions and conservative marker/classification helpers."""

from __future__ import annotations

import re
from typing import Any

SEMANTIC_FAMILIES = (
    "SOCIAL",
    "NEWS",
    "ONCHAIN",
    "PRICE_MOMENTUM",
    "LIQUIDITY",
    "WHALE",
    "LLM_CONTEXT",
    "UNKNOWN",
    "UNCLASSIFIED",
)

TRADING_STATES = (
    "OPPORTUNISTIC",
    "EXPLORATION",
    "STRICT_BLOCKED",
    "NO_TRADE",
    "PAPER_TRADED",
    "UNKNOWN",
)

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
NEWS_MARKERS = ("rss", "news", "article", "headline", "press")
WHALE_ONCHAIN_MARKERS = ("whale", "wallet", "holder", "transfer", "onchain", "on-chain")
MOMENTUM_MARKERS = ("momentum", "pump", "dump", "price_change", "trend")
LIQUIDITY_MARKERS = ("liquidity", "volume", "pool")

# Patterns that assign opportunistic as a silent default for missing/unknown category.
DANGEROUS_FALLBACK_PATTERNS = (
    r'cluster_label\s*=\s*["\']OPPORTUNISTIC_SPECULATIVE["\']',
    r'\.get\(\s*["\']cluster_label["\']\s*,\s*["\']OPPORTUNISTIC_SPECULATIVE["\']\s*\)',
    r'or\s+["\']OPPORTUNISTIC_SPECULATIVE["\']',
    r'return\s+ClusterLabel\.OPPORTUNISTIC_SPECULATIVE',
    r'DEFAULT_CLUSTER\s*=\s*["\']OPPORTUNISTIC_SPECULATIVE["\']',
    r'category\s*=\s*["\']opportunistic["\']',
    r'signal_type\s*=\s*.*["\']opportunistic["\']',
    r'or\s+["\']opportunistic["\']',
)


def empty_semantic_counts() -> dict[str, int]:
    return {k: 0 for k in SEMANTIC_FAMILIES}


def empty_trading_counts() -> dict[str, int]:
    return {k: 0 for k in TRADING_STATES}


def normalize_semantic_family(raw: Any) -> str:
    """Map a raw label to semantic_signal_family. Missing -> UNKNOWN (never OPPORTUNISTIC)."""
    if raw is None:
        return "UNKNOWN"
    text = str(raw).strip()
    if not text:
        return "UNKNOWN"
    upper = text.upper()
    if upper in {"UNKNOWN", "UNCLASSIFIED", "NONE", "NULL", "N/A"}:
        return "UNKNOWN" if upper != "UNCLASSIFIED" else "UNCLASSIFIED"
    if "SOCIAL" in upper:
        return "SOCIAL"
    if any(m in upper for m in ("NEWS", "RSS", "HEADLINE")):
        return "NEWS"
    if any(m in upper for m in ("WHALE",)):
        return "WHALE"
    if any(m in upper for m in ("ONCHAIN", "ON-CHAIN", "ON_CHAIN")):
        return "ONCHAIN"
    if any(m in upper for m in ("LIQUID", "VOLUME")):
        return "LIQUIDITY"
    if any(m in upper for m in ("MOMENTUM", "PRICE", "TREND")):
        return "PRICE_MOMENTUM"
    if any(m in upper for m in ("LLM", "QWEN", "GEMINI", "OLLAMA", "NARRATIVE")):
        return "LLM_CONTEXT"
    # Opportunistic/speculative is NOT a semantic family - treat as UNCLASSIFIED on this axis
    if "OPPORTUNISTIC" in upper or "SPECULATIVE" in upper:
        return "UNCLASSIFIED"
    return "UNCLASSIFIED"


def normalize_trading_state(raw: Any, *, row: dict[str, Any] | None = None) -> str:
    """Map payload fields to trading_opportunity_state."""
    row = row or {}
    if raw is not None and str(raw).strip():
        upper = str(raw).strip().upper()
        if "OPPORTUNISTIC" in upper or "SPECULATIVE" in upper:
            return "OPPORTUNISTIC"
        if "EXPLORATION" in upper:
            return "EXPLORATION"
        if "STRICT" in upper and "BLOCK" in upper:
            return "STRICT_BLOCKED"
        if "NO_TRADE" in upper or upper == "NO-TRADE":
            return "NO_TRADE"
        if "PAPER" in upper and "TRADE" in upper:
            return "PAPER_TRADED"
        if upper in TRADING_STATES:
            return upper

    # Derive from explicit trading fields only (not from missing semantic category)
    exploration = str(row.get("exploration_decision") or "").upper()
    strict = str(row.get("strict_shadow_decision") or "").upper()
    paper = str(row.get("paper_action_taken") or row.get("was_traded") or "").upper()
    cluster = str(row.get("cluster_label") or "").upper()

    if "TRADE_EXPLORATION" in exploration or exploration in {"TRADE", "BUY"}:
        if "TRUE" in paper or paper in {"TRADE", "BUY", "OPEN"}:
            return "PAPER_TRADED"
        return "EXPLORATION"
    if strict in {"NO_TRADE", "BLOCK", "BLOCKED"} or row.get("strict_approved") in (0, "0", False):
        if exploration in {"NO_TRADE", ""} and paper in {"", "NO_TRADE", "FALSE", "NONE"}:
            # Prefer explicit opportunistic cluster as trading state when present
            if "OPPORTUNISTIC" in cluster or "SPECULATIVE" in cluster:
                return "OPPORTUNISTIC"
            return "STRICT_BLOCKED" if strict else "NO_TRADE"
    if "OPPORTUNISTIC" in cluster or "SPECULATIVE" in cluster:
        return "OPPORTUNISTIC"
    if paper in {"TRADE", "BUY", "TRUE", "1"}:
        return "PAPER_TRADED"
    if exploration == "NO_TRADE" or paper == "NO_TRADE":
        return "NO_TRADE"
    return "UNKNOWN"


def classify_row_axes(row: dict[str, Any]) -> dict[str, str]:
    """
    Dual-axis classification from a single row.

    semantic_signal_family and trading_opportunity_state are independent:
    a row may be SOCIAL and OPPORTUNISTIC simultaneously.
    """
    semantic_raw = (
        row.get("semantic_signal_family")
        or row.get("signal_category")
        or row.get("signal_type")
        or row.get("source_family")
        or row.get("context_family")
        or row.get("strategy_family")
        or row.get("category")
    )
    # cluster_label alone does not define semantic family when it is opportunistic/speculative
    cluster = row.get("cluster_label")
    if semantic_raw is None and cluster:
        c_up = str(cluster).upper()
        if "SOCIAL" in c_up:
            semantic_raw = cluster
        # else leave None -> UNKNOWN / marker-based below

    marker_hit = detect_text_markers(row)
    if semantic_raw is None and marker_hit.get("primary_semantic"):
        semantic_raw = marker_hit["primary_semantic"]

    semantic = normalize_semantic_family(semantic_raw)

    trading_raw = (
        row.get("trading_opportunity_state")
        or row.get("opportunity_type")
        or row.get("candidate_type")
    )
    trading = normalize_trading_state(trading_raw, row=row)

    return {
        "semantic_signal_family": semantic,
        "trading_opportunity_state": trading,
        "marker_primary_semantic": marker_hit.get("primary_semantic") or "",
        "marker_hits": ",".join(marker_hit.get("hits") or []),
    }


def detect_text_markers(row: dict[str, Any]) -> dict[str, Any]:
    """Conservative text-marker audit only - does not invent classification authority."""
    parts: list[str] = []
    for key in (
        "title",
        "text",
        "body",
        "summary",
        "reasoning",
        "narrative",
        "llm_context",
        "llm_summary",
        "llm_verdict",
        "reason_for_no_trade",
        "reason_not_traded",
        "rejection_reason",
        "description",
        "source_family",
        "context_family",
        "ae8_context_status",
        "cluster_label",
    ):
        val = row.get(key)
        if val is None:
            continue
        if isinstance(val, (dict, list)):
            parts.append(str(val).lower())
        else:
            parts.append(str(val).lower())
    blob = " ".join(parts)
    hits: list[str] = []
    primary = None
    if any(m in blob for m in SOCIAL_MARKERS):
        hits.append("SOCIAL")
        primary = primary or "SOCIAL"
    if any(m in blob for m in NEWS_MARKERS):
        hits.append("NEWS")
        primary = primary or "NEWS"
    if any(m in blob for m in WHALE_ONCHAIN_MARKERS):
        label = "WHALE" if "whale" in blob else "ONCHAIN"
        hits.append(label)
        primary = primary or label
    if any(m in blob for m in LIQUIDITY_MARKERS):
        hits.append("LIQUIDITY")
        primary = primary or "LIQUIDITY"
    if any(m in blob for m in MOMENTUM_MARKERS):
        hits.append("PRICE_MOMENTUM")
        primary = primary or "PRICE_MOMENTUM"
    return {"hits": hits, "primary_semantic": primary, "text_len": len(blob)}


def scan_code_for_dangerous_fallbacks(source_text: str, file_path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for i, line in enumerate(source_text.splitlines(), start=1):
        for pat in DANGEROUS_FALLBACK_PATTERNS:
            if re.search(pat, line, flags=re.IGNORECASE):
                rows.append(
                    {
                        "file_path": file_path,
                        "line_no": str(i),
                        "pattern": pat,
                        "line_snippet": line.strip()[:240],
                        "severity": "HIGH",
                        "note": "Dangerous opportunistic default / fallback pattern",
                    }
                )
                break
    return rows
