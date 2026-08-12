"""AE19 provider availability probing — no import-time side effects."""

from __future__ import annotations

import os
from typing import Any

from app.llm_config import (
    get_ollama_base_url,
    get_ollama_model,
    get_ollama_timeout_seconds,
    is_gemini_env_enabled,
    is_headless_data_collection,
)
from app.llm_operational.schema import (
    GEMINI_UNAVAILABLE_OR_DISABLED,
    PROVIDER_AVAILABLE,
    PROVIDER_DISABLED,
    PROVIDER_ERROR,
    PROVIDER_UNAVAILABLE,
)


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")


def gemini_api_key_present() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def gemini_model_name() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def qwen_model_name() -> str:
    return get_ollama_model()


def qwen_base_url() -> str:
    return get_ollama_base_url()


def is_qwen_explicitly_enabled(*, allow_qwen: bool | None = None) -> bool:
    """Qwen/Ollama is used when explicitly allowed for AE19 or LLM_PROVIDER=ollama."""
    if allow_qwen is False:
        return False
    if allow_qwen is True:
        return True
    if is_headless_data_collection():
        return False
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider == "ollama":
        return True
    if _truthy(os.getenv("ENABLE_QWEN")) or _truthy(os.getenv("ENABLE_OLLAMA")):
        return True
    # Default AE19: attempt local Ollama probe unless explicitly disabled
    if _truthy(os.getenv("AE19_DISABLE_QWEN")):
        return False
    return True


def is_gemini_explicitly_enabled(*, allow_gemini: bool | None = None) -> bool:
    if allow_gemini is False:
        return False
    if allow_gemini is True:
        return True
    if is_headless_data_collection():
        return False
    if not is_gemini_env_enabled():
        return False
    provider = os.getenv("LLM_PROVIDER", "").strip().lower()
    if provider == "gemini":
        return True
    if _truthy(os.getenv("AE19_ENABLE_GEMINI")):
        return True
    # Gemini is selective/opt-in for AE19 unless AE19_ENABLE_GEMINI or LLM_PROVIDER=gemini
    return False


def probe_ollama_reachable(*, timeout_s: float | None = None) -> dict[str, Any]:
    """Best-effort local Ollama probe. Never raises."""
    import urllib.error
    import urllib.request

    timeout = timeout_s if timeout_s is not None else min(2.0, get_ollama_timeout_seconds())
    base = qwen_base_url()
    host = base.replace("/v1", "").rstrip("/")
    url = f"{host}/api/tags"
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = int(getattr(resp, "status", 200) or 200)
            reachable = status < 400
        return {
            "reachable": reachable,
            "probe_url": url,
            "base_url": base,
            "model": qwen_model_name(),
            "error": "" if reachable else f"http_status_{status}",
        }
    except urllib.error.URLError as exc:
        return {
            "reachable": False,
            "probe_url": url,
            "base_url": base,
            "model": qwen_model_name(),
            "error": f"url_error:{exc}",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "reachable": False,
            "probe_url": url,
            "base_url": base,
            "model": qwen_model_name(),
            "error": f"{type(exc).__name__}:{exc}",
        }


def resolve_qwen_provider_status(
    *,
    allow_qwen: bool | None = None,
    force_unavailable: bool = False,
    probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve Qwen/Ollama provider status without faking success."""
    enabled = is_qwen_explicitly_enabled(allow_qwen=allow_qwen)
    if force_unavailable:
        return {
            "provider": "qwen",
            "provider_model": qwen_model_name(),
            "provider_status": PROVIDER_UNAVAILABLE,
            "enabled": enabled,
            "reachable": False,
            "detail": "forced_unavailable",
            "probe": probe or {},
        }
    if not enabled:
        return {
            "provider": "qwen",
            "provider_model": qwen_model_name(),
            "provider_status": PROVIDER_DISABLED,
            "enabled": False,
            "reachable": False,
            "detail": "qwen_disabled_by_config",
            "probe": {},
        }
    probe_result = probe if probe is not None else probe_ollama_reachable()
    if probe_result.get("reachable"):
        return {
            "provider": "qwen",
            "provider_model": qwen_model_name(),
            "provider_status": PROVIDER_AVAILABLE,
            "enabled": True,
            "reachable": True,
            "detail": "ollama_reachable",
            "probe": probe_result,
        }
    err = str(probe_result.get("error") or "unreachable")
    status = PROVIDER_ERROR if "error" in err.lower() and "url_error" not in err else PROVIDER_UNAVAILABLE
    return {
        "provider": "qwen",
        "provider_model": qwen_model_name(),
        "provider_status": status,
        "enabled": True,
        "reachable": False,
        "detail": err,
        "probe": probe_result,
    }


def resolve_gemini_provider_status(
    *,
    allow_gemini: bool | None = None,
    force_unavailable: bool = False,
) -> dict[str, Any]:
    """Resolve Gemini provider status. Selective/opt-in only."""
    enabled = is_gemini_explicitly_enabled(allow_gemini=allow_gemini)
    key_present = gemini_api_key_present()
    model = gemini_model_name()

    if force_unavailable:
        return {
            "provider": "gemini",
            "provider_model": model,
            "provider_status": GEMINI_UNAVAILABLE_OR_DISABLED,
            "enabled": enabled,
            "key_present": key_present,
            "detail": "forced_unavailable",
        }
    if not enabled:
        return {
            "provider": "gemini",
            "provider_model": model,
            "provider_status": GEMINI_UNAVAILABLE_OR_DISABLED,
            "enabled": False,
            "key_present": key_present,
            "detail": "gemini_disabled_or_not_opted_in",
        }
    if not key_present:
        return {
            "provider": "gemini",
            "provider_model": model,
            "provider_status": GEMINI_UNAVAILABLE_OR_DISABLED,
            "enabled": True,
            "key_present": False,
            "detail": "gemini_api_key_missing",
        }
    return {
        "provider": "gemini",
        "provider_model": model,
        "provider_status": PROVIDER_AVAILABLE,
        "enabled": True,
        "key_present": True,
        "detail": "gemini_configured",
    }


def assert_no_false_provider_success(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Fail closed if mock/unavailable is counted as real provider success."""
    violations: list[dict[str, Any]] = []
    for rec in records:
        mock_used = bool(rec.get("mock_used"))
        counted = bool(rec.get("counted_as_real_provider_success"))
        provider_status = str(rec.get("provider_status") or "")
        task_status = str(rec.get("task_status") or "")

        if mock_used and counted:
            violations.append(
                {
                    "ae19_task_id": rec.get("ae19_task_id"),
                    "reason": "mock_counted_as_real_provider_success",
                }
            )
        if mock_used and provider_status == PROVIDER_AVAILABLE:
            violations.append(
                {
                    "ae19_task_id": rec.get("ae19_task_id"),
                    "reason": "mock_reported_as_provider_available",
                }
            )
        if mock_used and task_status == "LLM_TASK_SUCCEEDED":
            violations.append(
                {
                    "ae19_task_id": rec.get("ae19_task_id"),
                    "reason": "mock_reported_as_task_succeeded",
                }
            )
        if provider_status in {PROVIDER_UNAVAILABLE, PROVIDER_DISABLED, GEMINI_UNAVAILABLE_OR_DISABLED} and counted:
            violations.append(
                {
                    "ae19_task_id": rec.get("ae19_task_id"),
                    "reason": "unavailable_counted_as_real_success",
                }
            )

    return {
        "pass": len(violations) == 0,
        "violation_count": len(violations),
        "violations": violations,
        "block_code": "AE19_BLOCKED_FALSE_PROVIDER_SUCCESS_REPORTING" if violations else None,
    }
