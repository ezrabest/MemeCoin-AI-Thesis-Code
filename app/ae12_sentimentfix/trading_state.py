"""Trading opportunity state derivation (separate from semantic family)."""

from __future__ import annotations

from typing import Any

from .types import null_safe_str


def extract_raw_trading_state(row: dict[str, Any]) -> str | None:
    raw = (
        row.get("trading_opportunity_state")
        or row.get("opportunity_state")
        or row.get("trade_state")
        or None
    )
    if raw is None:
        return None
    text = str(raw).strip()
    return text if text else None


def derive_trading_opportunity_state(row: dict[str, Any]) -> tuple[str, str]:
    """
    Returns (state, source).
    Opportunistic/speculative labels map here only - never force semantic family.
    """
    explicit = extract_raw_trading_state(row)
    if explicit:
        upper = explicit.upper()
        if "OPPORTUNISTIC" in upper or "SPECULATIVE" in upper:
            return "OPPORTUNISTIC", "explicit_trading_state"
        if "EXPLORATION" in upper:
            return "EXPLORATION", "explicit_trading_state"
        if "STRICT" in upper and "BLOCK" in upper:
            return "STRICT_BLOCKED", "explicit_trading_state"
        if "PAPER" in upper:
            return "PAPER_TRADED", "explicit_trading_state"
        if "NO_TRADE" in upper or upper.replace("-", "_") == "NO_TRADE":
            return "NO_TRADE", "explicit_trading_state"
        if upper == "UNKNOWN":
            return "UNKNOWN", "explicit_trading_state"

    exploration = null_safe_str(row.get("exploration_decision"), "").upper()
    strict = null_safe_str(row.get("strict_shadow_decision"), "").upper()
    paper = null_safe_str(
        row.get("paper_action_taken") or row.get("was_traded"), ""
    ).upper()
    legacy = null_safe_str(
        row.get("legacy_cluster_label") or row.get("cluster_label"), ""
    ).upper()

    if paper in {"FILLED", "OPENED", "TRADE", "TRUE", "1", "BUY"} or "PAPER" in paper:
        if "EXPLORATION" in exploration or "OVERRIDE" in exploration:
            return "PAPER_TRADED", "paper_action+exploration"
        return "PAPER_TRADED", "paper_action"

    if "TRADE_EXPLORATION" in exploration or (
        "TRADE" in exploration and "NO_TRADE" not in exploration
    ):
        return "EXPLORATION", "exploration_decision"

    if row.get("max_open_positions_hit") or row.get("cooldown_active") or row.get("duplicate_active_pair"):
        return "STRICT_BLOCKED", "runtime_blockers"
    if row.get("blocked_by_ae9") or row.get("stale_price") or row.get("missing_context"):
        return "STRICT_BLOCKED", "runtime_blockers"
    if strict in {"NO_TRADE", "BLOCK", "BLOCKED"} and exploration in {"NO_TRADE", "", "UNKNOWN"}:
        # Prefer opportunistic legacy as trading state when present
        if "OPPORTUNISTIC" in legacy or "SPECULATIVE" in legacy:
            return "OPPORTUNISTIC", "legacy_cluster_as_trading_state"
        return "STRICT_BLOCKED", "strict_shadow"

    if "OPPORTUNISTIC" in legacy or "SPECULATIVE" in legacy:
        return "OPPORTUNISTIC", "legacy_cluster_as_trading_state"

    if exploration == "NO_TRADE" or paper in {"NO_TRADE", "NONE"}:
        return "NO_TRADE", "no_trade_fields"

    reason = null_safe_str(row.get("reason_for_no_trade") or row.get("reason_not_traded"), "").upper()
    if reason and reason not in {"UNKNOWN", ""}:
        return "STRICT_BLOCKED", "reason_for_no_trade"

    return "UNKNOWN", "none"
