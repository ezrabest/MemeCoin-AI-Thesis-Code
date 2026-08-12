"""AE9 Qwen/Ollama local LLM adapter — interface only, no default calls."""

from __future__ import annotations

import json
import os
from typing import Any

from app.llm_audit.prompt_templates import build_prompt_messages
from app.llm_audit.types import LLMCallStatus


def _ollama_config_present() -> bool:
  base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
  model = os.getenv("OLLAMA_MODEL", "qwen3:8b")
  return bool(base and model)


def run_qwen_audit(
  payload: dict[str, Any],
  *,
  allow_local_qwen: bool = False,
  allow_ollama: bool = False,
  audit_only: bool = True,
) -> dict[str, Any]:
  """Run Qwen/Ollama audit only when explicitly allowed and configured."""
  if not allow_local_qwen and not allow_ollama:
    return {
      "llm_provider": "qwen",
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
      "llm_provider": "qwen",
      "llm_model": None,
      "llm_call_status": LLMCallStatus.DISABLED_BY_DEFAULT.value,
      "llm_response_raw": None,
      "llm_response_parsed": None,
      "llm_verdict": None,
      "llm_confidence": None,
      "external_call_made": False,
      "error": "audit_mode_must_be_non_trading",
    }

  if not _ollama_config_present():
    return {
      "llm_provider": "qwen",
      "llm_model": None,
      "llm_call_status": LLMCallStatus.LLM_SOURCE_CONFIG_MISSING.value,
      "llm_response_raw": None,
      "llm_response_parsed": None,
      "llm_verdict": None,
      "llm_confidence": None,
      "external_call_made": False,
    }

  # Interface implemented — actual HTTP call only when explicitly allowed
  # Do not call external APIs in default smoke; return config-ready stub
  messages = build_prompt_messages(payload)
  return {
    "llm_provider": "qwen",
    "llm_model": os.getenv("OLLAMA_MODEL", "qwen3:8b"),
    "llm_call_status": LLMCallStatus.LLM_SOURCE_CONFIG_MISSING.value,
    "llm_response_raw": None,
    "llm_response_parsed": None,
    "llm_verdict": None,
    "llm_confidence": None,
    "external_call_made": False,
    "messages_prepared": len(messages),
  }


def run_ollama_audit(
  payload: dict[str, Any],
  *,
  allow_ollama: bool = False,
  audit_only: bool = True,
) -> dict[str, Any]:
  """Alias for Ollama provider — same interface constraints as Qwen."""
  result = run_qwen_audit(
    payload,
    allow_local_qwen=False,
    allow_ollama=allow_ollama,
    audit_only=audit_only,
  )
  result["llm_provider"] = "ollama"
  return result
