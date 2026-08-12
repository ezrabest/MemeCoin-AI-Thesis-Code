"""AE19 Gemini selective external audit runtime — opt-in only."""

from __future__ import annotations

from typing import Any

from app.llm_operational.providers import gemini_model_name, resolve_gemini_provider_status
from app.llm_operational.schema import (
    GEMINI_UNAVAILABLE_OR_DISABLED,
    MOCK_PROVIDER_DIAGNOSTIC,
    PROVIDER_AVAILABLE,
    PROVIDER_ERROR,
    TASK_FAILED,
    TASK_MOCK_DIAGNOSTIC,
    TASK_SKIPPED_PROVIDER,
    TASK_SUCCEEDED,
)
from app.llm_operational.qwen_runtime import SYSTEM_PROMPT, build_mock_diagnostic_response


def call_gemini_generate(prompt: str, *, model: str | None = None) -> dict[str, Any]:
    """Call Gemini generate_content when configured. Never raises."""
    import os

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    model_name = model or gemini_model_name()
    if not api_key:
        return {
            "ok": False,
            "text": "",
            "provider_status": GEMINI_UNAVAILABLE_OR_DISABLED,
            "error": "gemini_api_key_missing",
        }
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as exc:
        return {
            "ok": False,
            "text": "",
            "provider_status": GEMINI_UNAVAILABLE_OR_DISABLED,
            "error": f"google_generativeai_missing:{exc}",
        }
    try:
        genai.configure(api_key=api_key)
        gm = genai.GenerativeModel(model_name)
        full_prompt = f"{SYSTEM_PROMPT}\n\n{prompt}"
        response = gm.generate_content(full_prompt)
        text = ""
        if hasattr(response, "text") and response.text:
            text = str(response.text).strip()
        elif hasattr(response, "candidates") and response.candidates:
            parts = getattr(response.candidates[0].content, "parts", None) or []
            text = " ".join(str(getattr(p, "text", "") or "") for p in parts).strip()
        if not text:
            return {
                "ok": False,
                "text": "",
                "provider_status": PROVIDER_ERROR,
                "error": "empty_gemini_response",
            }
        return {
            "ok": True,
            "text": text,
            "provider_status": PROVIDER_AVAILABLE,
            "error": "",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "text": "",
            "provider_status": PROVIDER_ERROR,
            "error": f"gemini_error:{type(exc).__name__}:{exc}",
        }


def run_gemini_operational(
    prompt: str,
    *,
    task_type: str,
    candidate: dict[str, Any],
    allow_gemini: bool | None = None,
    force_unavailable: bool = False,
    use_mock_diagnostic: bool = False,
    provider_status_cache: dict[str, Any] | None = None,
    selective: bool = True,
) -> dict[str, Any]:
    """
    Execute selective Gemini audit call or return explicit unavailable/disabled status.

    Gemini is selective audit only. Disabled/unconfigured → explicit skip, never fake success.
    """
    del selective  # documented contract; all AE19 Gemini calls are selective by policy

    if use_mock_diagnostic:
        text = build_mock_diagnostic_response(f"GEMINI_{task_type}", candidate)
        return {
            "provider": "mock_gemini",
            "provider_model": "mock_diagnostic",
            "provider_status": MOCK_PROVIDER_DIAGNOSTIC,
            "task_status": TASK_MOCK_DIAGNOSTIC,
            "text": text,
            "mock_used": True,
            "counted_as_real_provider_success": False,
            "downstream_eligible": False,
            "downstream_quarantined": True,
            "accepted_for_downstream": False,
            "failure_reason": "MOCK_PROVIDER_USED_DIAGNOSTIC_ONLY",
            "error": "",
        }

    status = provider_status_cache or resolve_gemini_provider_status(
        allow_gemini=allow_gemini,
        force_unavailable=force_unavailable,
    )
    provider_status = str(status.get("provider_status") or GEMINI_UNAVAILABLE_OR_DISABLED)
    model = str(status.get("provider_model") or gemini_model_name())

    if provider_status != PROVIDER_AVAILABLE:
        return {
            "provider": "gemini",
            "provider_model": model,
            "provider_status": GEMINI_UNAVAILABLE_OR_DISABLED
            if provider_status != PROVIDER_ERROR
            else provider_status,
            "task_status": TASK_SKIPPED_PROVIDER,
            "text": "",
            "mock_used": False,
            "counted_as_real_provider_success": False,
            "downstream_eligible": False,
            "downstream_quarantined": True,
            "accepted_for_downstream": False,
            "failure_reason": "GEMINI_PROVIDER_UNAVAILABLE_OR_DISABLED",
            "error": str(status.get("detail") or provider_status),
            "provider_resolution": status,
        }

    call = call_gemini_generate(prompt, model=model)
    if call.get("ok"):
        return {
            "provider": "gemini",
            "provider_model": model,
            "provider_status": PROVIDER_AVAILABLE,
            "task_status": TASK_SUCCEEDED,
            "text": call.get("text") or "",
            "mock_used": False,
            "counted_as_real_provider_success": True,
            "downstream_eligible": True,
            "downstream_quarantined": False,
            "accepted_for_downstream": True,
            "failure_reason": "",
            "error": "",
            "provider_resolution": status,
        }

    return {
        "provider": "gemini",
        "provider_model": model,
        "provider_status": call.get("provider_status") or PROVIDER_ERROR,
        "task_status": TASK_FAILED
        if call.get("provider_status") == PROVIDER_ERROR
        else TASK_SKIPPED_PROVIDER,
        "text": "",
        "mock_used": False,
        "counted_as_real_provider_success": False,
        "downstream_eligible": False,
        "downstream_quarantined": True,
        "accepted_for_downstream": False,
        "failure_reason": str(call.get("error") or "GEMINI_PROVIDER_UNAVAILABLE_OR_DISABLED"),
        "error": str(call.get("error") or ""),
        "provider_resolution": status,
    }
