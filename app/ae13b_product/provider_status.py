"""AE13B provider / LLM / RSS / semantic runtime status (honest, non-scary).

Canonical provider_health values:
  - active
  - unavailable_metrics_helper  ("Unavailable — Metrics Helper Only")
  - inactive

Never expose a vague "Provider error" label.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

# Canonical health codes (UI + API contract)
HEALTH_ACTIVE = "active"
HEALTH_UNAVAILABLE_METRICS = "unavailable_metrics_helper"
HEALTH_INACTIVE = "inactive"

LABEL_ACTIVE = "Active"
LABEL_UNAVAILABLE_METRICS = "Unavailable — Metrics Helper Only"
LABEL_INACTIVE = "Inactive"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_provider_status() -> dict[str, Any]:
    from app.env_bootstrap import resolve_active_mode_label
    from app.llm_config import (
        get_llm_provider,
        get_llm_runtime_status,
        get_ollama_base_url,
        get_ollama_model,
        is_gemini_provider_active,
        is_headless_data_collection,
        is_ollama_provider_active,
    )

    provider = get_llm_provider()
    ollama_active = is_ollama_provider_active()
    gemini_active = is_gemini_provider_active()
    headless = is_headless_data_collection()
    snap = {}
    try:
        snap = get_llm_runtime_status()
    except Exception:
        snap = {}

    ollama_errors = int(snap.get("ollama_error_count") or 0)
    last_err_reason = str(snap.get("ollama_last_error") or "").strip() or None

    # Resolve three-state health — never "error" / "Provider error"
    if headless or provider == "none":
        health = HEALTH_INACTIVE
        health_label = LABEL_INACTIVE
        selected = "none" if provider == "none" else provider
        assistant_mode = "inactive"
    elif ollama_active:
        selected = "ollama"
        if ollama_errors > 0:
            health = HEALTH_UNAVAILABLE_METRICS
            health_label = LABEL_UNAVAILABLE_METRICS
            assistant_mode = "metrics_helper"
        else:
            health = HEALTH_ACTIVE
            health_label = LABEL_ACTIVE
            assistant_mode = "provider_backed"
    elif gemini_active:
        selected = "gemini"
        health = HEALTH_ACTIVE
        health_label = LABEL_ACTIVE
        assistant_mode = "provider_backed"
    elif provider == "ollama":
        # Selected but not considered active (e.g. headless race) → metrics helper
        selected = "ollama"
        health = HEALTH_UNAVAILABLE_METRICS
        health_label = LABEL_UNAVAILABLE_METRICS
        assistant_mode = "metrics_helper"
    elif provider == "gemini":
        selected = "gemini"
        health = HEALTH_UNAVAILABLE_METRICS
        health_label = LABEL_UNAVAILABLE_METRICS
        assistant_mode = "metrics_helper"
    else:
        health = HEALTH_INACTIVE
        health_label = LABEL_INACTIVE
        selected = provider or "local-rules"
        assistant_mode = "inactive"

    last_error = None
    last_error_detail = None
    if ollama_errors > 0 and selected == "ollama":
        reason = last_err_reason or "request failed"
        last_error = f"Ollama unavailable: {reason}"
        last_error_detail = {
            "provider": "ollama",
            "error_count": ollama_errors,
            "reason": reason,
            "fallback": LABEL_UNAVAILABLE_METRICS,
        }

    runtime_mode = "Demo"
    try:
        from app.execution.paper import get_paper_trader

        mode = str(get_paper_trader().get_wallet_summary().get("trading_mode") or "DEMO").upper()
        if mode == "PAPER":
            runtime_mode = "Paper"
        elif mode == "DEMO":
            runtime_mode = "Demo"
        else:
            runtime_mode = "Safe"
    except Exception:
        runtime_mode = "Demo"

    semantic = {}
    try:
        from app.ae13_semantic.runtime_registry import get_semantic_registry

        semantic = get_semantic_registry().snapshot()
    except Exception:
        semantic = {}

    def _provider_row(name: str, *, selected_here: bool, active: bool) -> dict[str, Any]:
        if selected_here and active and health == HEALTH_ACTIVE:
            row_health, row_label = HEALTH_ACTIVE, LABEL_ACTIVE
        elif selected_here and health == HEALTH_UNAVAILABLE_METRICS:
            row_health, row_label = HEALTH_UNAVAILABLE_METRICS, LABEL_UNAVAILABLE_METRICS
        elif selected_here:
            row_health, row_label = HEALTH_INACTIVE, LABEL_INACTIVE
        else:
            row_health, row_label = HEALTH_INACTIVE, LABEL_INACTIVE
        return {
            "selected": selected_here,
            "active": active and health == HEALTH_ACTIVE,
            "configured": selected_here,
            "health": row_health,
            "health_label": row_label,
        }

    ollama_row = _provider_row("ollama", selected_here=selected == "ollama", active=ollama_active)
    ollama_row.update(
        {
            "base_url": get_ollama_base_url() if selected == "ollama" else None,
            "model": get_ollama_model() if selected == "ollama" else None,
            "call_count": snap.get("ollama_call_count"),
            "error_count": snap.get("ollama_error_count"),
        }
    )
    gemini_row = _provider_row("gemini", selected_here=selected == "gemini", active=gemini_active)
    gemini_row["call_count"] = snap.get("gemini_call_count")
    if gemini_active and health == HEALTH_ACTIVE:
        gemini_row["health_label"] = "Active (audit/explanation only — no trade authority)"

    return {
        "runtime_mode": runtime_mode,
        "cli_mode_label": resolve_active_mode_label(),
        "llm_provider_selected": selected,
        "provider_selected": selected,
        "provider_reachable": health == HEALTH_ACTIVE,
        "model_name": (
            get_ollama_model() if selected == "ollama" else ("gemini" if selected == "gemini" else None)
        ),
        "model_available": health == HEALTH_ACTIVE,
        "llm_provider_actually_active": health == HEALTH_ACTIVE,
        "provider_configured": selected not in ("none", "", None),
        "provider_health": health,
        "provider_health_label": health_label,
        "provider_health_codes": {
            "active": LABEL_ACTIVE,
            "unavailable_metrics_helper": LABEL_UNAVAILABLE_METRICS,
            "inactive": LABEL_INACTIVE,
        },
        "assistant_mode": assistant_mode,
        "local_rules_active": True,
        "rss_active": True,
        "demo_trading_blocked_by_provider": False,
        "last_provider_error": last_error,
        "last_provider_success": snap.get("ollama_last_success") or snap.get("last_success_at"),
        "last_health_check_at": _utc_now(),
        "provider_status_explanation": (
            f"{selected or 'none'} selected"
            + (
                ", reachable. Local rules active."
                if health == HEALTH_ACTIVE
                else (
                    f", not reachable. {LABEL_UNAVAILABLE_METRICS}. "
                    "Local rules still active. Demo trading not blocked by LLM."
                    if health == HEALTH_UNAVAILABLE_METRICS
                    else ". Local rules active — no LLM assistant. Demo trading not blocked by provider."
                )
            )
        ),
        "ollama": ollama_row,
        "gemini": gemini_row,
        "rss": {
            "status": "available_via_/api/sentiment/matrix",
            "health_label": "RSS active (headline lexicon — not SOCIAL_CONFIRMED)",
            "active": True,
        },
        "local_rules": {
            "active": True,
            "health_label": "Local rules active — no LLM assistant"
            if health != HEALTH_ACTIVE
            else "Local rules active",
        },
        "semantic_registry": {
            "status": semantic.get("semantic_source_label"),
            "runtime_unique_identities": semantic.get("runtime_unique_identities"),
            "max_entries": semantic.get("max_entries"),
            "eviction_count": semantic.get("eviction_count"),
            "last_update": semantic.get("updated_at_utc"),
            "social_confirmed_explanation": semantic.get("social_confirmed_explanation"),
        },
        "trade_authority": "AI explanation only, no trade authority",
        "assistant": {
            "label": (
                "AI Assistant — explanation only, no trade authority"
                if health == HEALTH_ACTIVE
                else (
                    "Metrics Assistant — limited demo metrics only"
                    if health == HEALTH_UNAVAILABLE_METRICS
                    else "Assistant inactive"
                )
            ),
            "path": "provider_backed_when_active_else_metrics_helper",
            "can_place_trades": False,
        },
        "last_provider_error_detail": last_error_detail,
        "checked_at_utc": _utc_now(),
        "scary_error_reserved_for_real_failures": True,
        "never_show_vague_provider_error": True,
        "fail_soft": True,
    }


def build_ai_assistant_status() -> dict[str, Any]:
    status = build_provider_status()
    health = status.get("provider_health")
    if health == HEALTH_ACTIVE:
        mode = "real_provider_backed_assistant"
        label = "AI Assistant — explanation only, no trade authority"
        capability = (
            "Ollama/Qwen or Gemini answers natural questions about the demo bot, blockers, "
            "positions, sentiment, and registry. It cannot place trades."
        )
    elif health == HEALTH_UNAVAILABLE_METRICS:
        mode = "limited_metrics_helper"
        label = "Metrics Assistant — limited demo metrics only"
        capability = (
            "Provider is unavailable. This assistant answers demo metrics questions only."
        )
    else:
        mode = "inactive"
        label = "Assistant inactive"
        capability = "No LLM provider selected. Local classification rules remain active."
    return {
        "mode": mode,
        "label": label,
        "capability": capability,
        "examples": [
            "average whale score",
            "net ROI after fees",
            "open positions",
            "why no trade",
            "recent blockers",
            "demo bot status",
            "semantic registry status",
        ],
        "can_place_trades": False,
        "provider": status.get("llm_provider_selected"),
        "provider_active": status.get("llm_provider_actually_active"),
        "provider_health": health,
        "provider_health_label": status.get("provider_health_label"),
    }
