"""AE9 Gemini LLM adapter — interface only, no default calls."""

from __future__ import annotations

import os
from typing import Any

from app.llm_audit.prompt_templates import build_prompt_messages
from app.llm_audit.types import LLMCallStatus


def _gemini_config_present() -> bool:
  return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def run_gemini_audit(
  payload: dict[str, Any],
  *,
  allow_gemini: bool = False,
  audit_only: bool = True,
) -> dict[str, Any]:
  """Run Gemini audit only when explicitly allowed and configured."""
  if not allow_gemini:
    return {
      "llm_provider": "gemini",
      "llm_model": None,
      "llm_call_status": LLMCallStatus.DISABLED_BY_DEFAULT.value,
      "llm_response_raw": None,
      "llm_response_parsed": None,
      "llm_verdict": None,
      "llm_confidence": None,
      "external_call_made": False,
    }

  if not audit_only:
    return {
      "llm_provider": "gemini",
      "llm_model": None,
      "llm_call_status": LLMCallStatus.DISABLED_BY_DEFAULT.value,
      "llm_response_raw": None,
      "llm_response_parsed": None,
      "llm_verdict": None,
      "llm_confidence": None,
      "external_call_made": False,
      "error": "audit_mode_must_be_non_trading",
    }

  if not _gemini_config_present():
    return {
      "llm_provider": "gemini",
      "llm_model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
      "llm_call_status": LLMCallStatus.LLM_SOURCE_CONFIG_MISSING.value,
      "llm_response_raw": None,
      "llm_response_parsed": None,
      "llm_verdict": None,
      "llm_confidence": None,
      "external_call_made": False,
    }

  messages = build_prompt_messages(payload)
  return {
    "llm_provider": "gemini",
    "llm_model": os.getenv("GEMINI_MODEL", "gemini-2.0-flash"),
    "llm_call_status": LLMCallStatus.LLM_SOURCE_CONFIG_MISSING.value,
    "llm_response_raw": None,
    "llm_response_parsed": None,
    "llm_verdict": None,
    "llm_confidence": None,
    "external_call_made": False,
    "messages_prepared": len(messages),
  }
