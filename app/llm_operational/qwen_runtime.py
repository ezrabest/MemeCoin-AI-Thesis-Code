"""AE19 Qwen/Ollama operational runtime — real calls or explicit unavailable."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from app.llm_config import get_ollama_timeout_seconds, record_ollama_call, record_ollama_error
from app.llm_operational.providers import (
    qwen_base_url,
    qwen_model_name,
    resolve_qwen_provider_status,
)
from app.llm_operational.schema import (
    MOCK_PROVIDER_DIAGNOSTIC,
    PROVIDER_AVAILABLE,
    PROVIDER_ERROR,
    PROVIDER_UNAVAILABLE,
    TASK_FAILED,
    TASK_MOCK_DIAGNOSTIC,
    TASK_SKIPPED_PROVIDER,
    TASK_SUCCEEDED,
)


SYSTEM_PROMPT = (
    "You are an operational explanation/audit assistant for a paper/demo memecoin research system. "
    "Roles allowed: WATCH, REVIEW, EXPLAIN, RISK_NOTE, CONTEXT_SUMMARY, RESEARCH_ONLY, "
    "PAPER_DEMO_OBSERVATION, NOT_TRADE_AUTHORITY. "
    "You have NO trade authority. Never approve live trades, never override RiskGuard or GateKeeper, "
    "never connect wallets, never sign/submit transactions, never claim guaranteed profit or live readiness. "
    "Never invent pair addresses, token addresses, price_source_key, provider URLs, or resolver links. "
    "If identity is missing, say unresolved. Respond in plain text."
)


def build_mock_diagnostic_response(task_type: str, candidate: dict[str, Any]) -> str:
    """Explicit mock/diagnostic text — must never count as provider success."""
    cid = candidate.get("clean_forward_candidate_id") or candidate.get("candidate_id") or "UNKNOWN"
    pair = candidate.get("pair_address") or "UNRESOLVED"
    return (
        f"[MOCK_DIAGNOSTIC_ONLY] {task_type} for candidate={cid} pair={pair}. "
        f"RESEARCH_ONLY / PAPER_DEMO_OBSERVATION / NOT_TRADE_AUTHORITY. "
        f"This is diagnostic mock output, not a real provider response."
    )


def call_ollama_chat(prompt: str, *, timeout_s: float | None = None) -> dict[str, Any]:
    """Call local Ollama native /api/chat. Never raises."""
    timeout = timeout_s if timeout_s is not None else get_ollama_timeout_seconds()
    host = qwen_base_url().replace("/v1", "").rstrip("/")
    model = qwen_model_name()
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        msg = (data.get("message") or {}).get("content")
        if isinstance(msg, str) and msg.strip():
            record_ollama_call()
            return {
                "ok": True,
                "text": msg.strip(),
                "provider": "qwen",
                "provider_model": model,
                "provider_status": PROVIDER_AVAILABLE,
                "error": "",
            }
        record_ollama_error()
        return {
            "ok": False,
            "text": "",
            "provider": "qwen",
            "provider_model": model,
            "provider_status": PROVIDER_ERROR,
            "error": "empty_ollama_response",
        }
    except urllib.error.URLError as exc:
        record_ollama_error()
        return {
            "ok": False,
            "text": "",
            "provider": "qwen",
            "provider_model": model,
            "provider_status": PROVIDER_UNAVAILABLE,
            "error": f"ollama_unreachable:{exc}",
        }
    except Exception as exc:  # noqa: BLE001
        record_ollama_error()
        return {
            "ok": False,
            "text": "",
            "provider": "qwen",
            "provider_model": model,
            "provider_status": PROVIDER_ERROR,
            "error": f"ollama_error:{type(exc).__name__}:{exc}",
        }


def run_qwen_operational(
    prompt: str,
    *,
    task_type: str,
    candidate: dict[str, Any],
    allow_qwen: bool | None = None,
    force_unavailable: bool = False,
    use_mock_diagnostic: bool = False,
    provider_status_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Execute one Qwen/Ollama operational call or return explicit skip/mock status.

    Mock path is diagnostic-only and never counts as real provider success.
    """
    if use_mock_diagnostic:
        text = build_mock_diagnostic_response(task_type, candidate)
        return {
            "provider": "mock",
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

    status = provider_status_cache or resolve_qwen_provider_status(
        allow_qwen=allow_qwen,
        force_unavailable=force_unavailable,
    )
    provider_status = str(status.get("provider_status") or PROVIDER_UNAVAILABLE)
    model = str(status.get("provider_model") or qwen_model_name())

    if provider_status != PROVIDER_AVAILABLE:
        return {
            "provider": "qwen",
            "provider_model": model,
            "provider_status": provider_status,
            "task_status": TASK_SKIPPED_PROVIDER,
            "text": "",
            "mock_used": False,
            "counted_as_real_provider_success": False,
            "downstream_eligible": False,
            "downstream_quarantined": True,
            "accepted_for_downstream": False,
            "failure_reason": "LLM_PROVIDER_UNAVAILABLE",
            "error": str(status.get("detail") or provider_status),
            "provider_resolution": status,
        }

    call = call_ollama_chat(prompt)
    if call.get("ok"):
        return {
            "provider": "qwen",
            "provider_model": call.get("provider_model") or model,
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
        "provider": "qwen",
        "provider_model": model,
        "provider_status": call.get("provider_status") or PROVIDER_ERROR,
        "task_status": TASK_FAILED if call.get("provider_status") == PROVIDER_ERROR else TASK_SKIPPED_PROVIDER,
        "text": "",
        "mock_used": False,
        "counted_as_real_provider_success": False,
        "downstream_eligible": False,
        "downstream_quarantined": True,
        "accepted_for_downstream": False,
        "failure_reason": str(call.get("error") or "LLM_PROVIDER_ERROR"),
        "error": str(call.get("error") or ""),
        "provider_resolution": status,
    }
