"""
LLM provider flags, Ollama settings, and in-process call counters.
"""
from __future__ import annotations

import os
import threading
from typing import Any

SKIP_REASON = "LLM evaluation skipped - Headless Data Collection mode active"
SKIP_REASON_NONE = "LLM evaluation skipped - LLM_PROVIDER=none"
SKIP_REASON_BUDGET = "LLM skipped: local call budget reached"
OLLAMA_FALLBACK_REASON = "Local LLM unreachable or invalid response"

# AE19: LLM providers are audit/shadow only — never trade authority.
LLM_AUDIT_ONLY_PROVIDERS = frozenset({"gemini", "qwen", "ollama"})
LLM_PROVIDER_AUDIT_ONLY_REASON = "LLM_PROVIDER_AUDIT_ONLY"
LLM_AUTHORITY_STATUS = "AUDIT_ONLY_NO_TRADE_AUTHORITY"

_lock = threading.Lock()
_gemini_call_count = 0
_llm_skipped_count = 0
_ollama_call_count = 0
_ollama_skipped_count = 0
_ollama_error_count = 0
_ollama_calls_this_scan = 0
_scan_llm_decisions_stored = 0
_scan_gemini_decisions_stored = 0


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def _falsy(value: str | None) -> bool:
    return str(value or "").strip().lower() in ("0", "false", "no", "off")


def is_headless_data_collection() -> bool:
    return _truthy(os.getenv("HEADLESS_DATA_COLLECTION"))


def is_gemini_env_enabled() -> bool:
    enable = os.getenv("ENABLE_GEMINI", "true")
    return not _falsy(enable)


def get_llm_provider() -> str:
    """Resolved provider: none | gemini | ollama."""
    raw = os.getenv("LLM_PROVIDER", "").strip().lower()
    if raw in ("none", "gemini", "ollama"):
        return raw
    if is_headless_data_collection():
        return "none"
    if is_gemini_env_enabled():
        return "gemini"
    return "none"


def get_ollama_base_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")


def get_ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "qwen3:8b")


def get_ollama_max_calls_per_scan() -> int:
    try:
        return max(0, int(os.getenv("OLLAMA_MAX_CALLS_PER_SCAN", "5")))
    except ValueError:
        return 5


def get_ollama_timeout_seconds() -> float:
    try:
        return max(1.0, float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60")))
    except ValueError:
        return 60.0


def is_gemini_provider_active() -> bool:
    return (
        not is_headless_data_collection()
        and get_llm_provider() == "gemini"
        and is_gemini_env_enabled()
    )


def is_ollama_provider_active() -> bool:
    return not is_headless_data_collection() and get_llm_provider() == "ollama"


def is_gemini_enabled() -> bool:
    """Backward-compatible alias."""
    return is_gemini_provider_active()


def is_llm_enabled() -> bool:
    if is_headless_data_collection():
        return False
    provider = get_llm_provider()
    if provider == "none":
        return False
    if provider == "ollama":
        return True
    if provider == "gemini":
        return is_gemini_env_enabled()
    return False


def normalize_llm_provider_name(provider: str | None) -> str:
    return str(provider or "").strip().lower()


def is_llm_audit_only_provider(provider: str | None = None) -> bool:
    """True when provider is gemini/qwen/ollama (explicit or resolved)."""
    if provider is not None and str(provider).strip():
        resolved = normalize_llm_provider_name(provider)
    else:
        resolved = get_llm_provider()
    return resolved in LLM_AUDIT_ONLY_PROVIDERS


def extract_explicit_llm_origin_provider(*sources: Any) -> str | None:
    """Detect explicit LLM-origin markers only (defense-in-depth for paper.py).

    Does not infer from ambient LLM_PROVIDER env — only fields present on payloads.
    """
    keys = (
        "provider",
        "llm_provider",
        "model_provider",
        "source_provider",
        "decision_provider",
    )
    for src in sources:
        if not isinstance(src, dict):
            continue
        for key in keys:
            raw = src.get(key)
            if raw is None or raw == "":
                continue
            normalized = normalize_llm_provider_name(str(raw))
            if normalized in LLM_AUDIT_ONLY_PROVIDERS:
                return normalized
        model_source = src.get("model_source")
        if model_source is None or model_source == "":
            continue
        ms = normalize_llm_provider_name(str(model_source))
        for name in LLM_AUDIT_ONLY_PROVIDERS:
            if ms == name or ms.startswith(f"{name}") or ms.startswith(f"{name}:") or ms.startswith(f"{name}/") or ms.startswith(f"{name}-"):
                return name
    return None


def build_llm_authority_boundary(*, execution_attempted: bool = False) -> dict[str, Any]:
    """Fields persisted on LLM decision JSON — audit only, no trade authority."""
    return {
        "authority_status": LLM_AUTHORITY_STATUS,
        "execution_allowed": False,
        "paper_execution_allowed": False,
        "live_execution_allowed": False,
        "risk_override_allowed": False,
        "execution_attempted": bool(execution_attempted),
        "blocked_reason": LLM_PROVIDER_AUDIT_ONLY_REASON,
    }


def reset_scan_llm_decision_counters() -> None:
    """Reset per-scan LLM decision storage counters (call at scan start)."""
    global _scan_llm_decisions_stored, _scan_gemini_decisions_stored
    with _lock:
        _scan_llm_decisions_stored = 0
        _scan_gemini_decisions_stored = 0


def record_scan_llm_decision_stored(provider: str | None) -> None:
    """Increment per-scan counters after a provider decision row is stored."""
    global _scan_llm_decisions_stored, _scan_gemini_decisions_stored
    normalized = normalize_llm_provider_name(provider)
    with _lock:
        _scan_llm_decisions_stored += 1
        if normalized == "gemini":
            _scan_gemini_decisions_stored += 1


def get_scan_llm_decisions_stored() -> int:
    with _lock:
        return _scan_llm_decisions_stored


def get_scan_gemini_decisions_stored() -> int:
    with _lock:
        return _scan_gemini_decisions_stored


def reset_ollama_scan_budget() -> None:
    global _ollama_calls_this_scan
    with _lock:
        _ollama_calls_this_scan = 0


def try_consume_ollama_call() -> bool:
    """Reserve one Ollama call slot for the current scan."""
    global _ollama_calls_this_scan
    limit = get_ollama_max_calls_per_scan()
    with _lock:
        if _ollama_calls_this_scan >= limit:
            return False
        _ollama_calls_this_scan += 1
        return True


def ollama_budget_remaining() -> int:
    with _lock:
        return max(0, get_ollama_max_calls_per_scan() - _ollama_calls_this_scan)


def record_gemini_call() -> None:
    global _gemini_call_count
    with _lock:
        _gemini_call_count += 1


def record_llm_skipped() -> None:
    global _llm_skipped_count
    with _lock:
        _llm_skipped_count += 1


def record_ollama_call() -> None:
    global _ollama_call_count
    with _lock:
        _ollama_call_count += 1


def record_ollama_skipped() -> None:
    global _ollama_skipped_count
    with _lock:
        _ollama_skipped_count += 1


def record_ollama_error() -> None:
    global _ollama_error_count
    with _lock:
        _ollama_error_count += 1


def reset_llm_counters() -> None:
    """Test helper — reset in-process counters."""
    global _gemini_call_count, _llm_skipped_count
    global _ollama_call_count, _ollama_skipped_count, _ollama_error_count
    global _ollama_calls_this_scan
    with _lock:
        _gemini_call_count = 0
        _llm_skipped_count = 0
        _ollama_call_count = 0
        _ollama_skipped_count = 0
        _ollama_error_count = 0
        _ollama_calls_this_scan = 0
    reset_scan_llm_decision_counters()


def get_llm_runtime_status() -> dict[str, Any]:
    return {
        "llm_provider": get_llm_provider(),
        "headless_data_collection": is_headless_data_collection(),
        "enable_gemini": is_gemini_env_enabled(),
        "llm_enabled": is_llm_enabled(),
        "ollama_enabled": is_ollama_provider_active(),
        "ollama_model": get_ollama_model(),
        "ollama_base_url": get_ollama_base_url(),
        "ollama_max_calls_per_scan": get_ollama_max_calls_per_scan(),
        "ollama_budget_remaining": ollama_budget_remaining(),
        "ollama_call_count": _ollama_call_count,
        "ollama_skipped_count": _ollama_skipped_count,
        "ollama_error_count": _ollama_error_count,
        "gemini_call_count": _gemini_call_count,
        "llm_skipped_count": _llm_skipped_count,
        "skip_reason": SKIP_REASON,
        "skip_reason_none": SKIP_REASON_NONE,
        "skip_reason_budget": SKIP_REASON_BUDGET,
    }
