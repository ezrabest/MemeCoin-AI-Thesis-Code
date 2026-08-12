"""
Unified local LLM client — Ollama via native REST /api/chat.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

from .llm_config import (
    OLLAMA_FALLBACK_REASON,
    get_ollama_base_url,
    get_ollama_model,
    get_ollama_timeout_seconds,
    record_ollama_call,
    record_ollama_error,
)

log = logging.getLogger("llm_client")

OLLAMA_PROVIDER_TAG = "ollama_qwen3_8b"

VALID_ACTIONS = frozenset({"BUY", "SELL", "HOLD", "SKIPPED"})
VALID_STRATEGIES = frozenset({"MOMENTUM", "WHALE_FLOW", "SENTIMENT", "RISK_OFF", "UNKNOWN"})


def normalize_strategy_type(raw: Any) -> str:
    """Map LLM strategy labels to the allowed Ollama schema set."""
    strategy = str(raw or "UNKNOWN").upper().strip()
    if strategy in VALID_STRATEGIES:
        return strategy
    if strategy in ("WHALE_RIDER", "WHALE", "SCALPING_OPPORTUNITY") or "WHALE" in strategy or "RIDER" in strategy:
        return "WHALE_FLOW"
    return "UNKNOWN"
VALID_RISK_LEVELS = frozenset({"LOW", "MEDIUM", "HIGH"})
VALID_DIRECTIONS = frozenset({"UP", "DOWN", "SIDEWAYS"})


def _clamp_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse clean JSON or extract the first valid JSON object from surrounding text."""
    cleaned = _strip_markdown_fences(text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            parsed, _end = decoder.raw_decode(cleaned, match.start())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue
    raise ValueError("No valid JSON object found in LLM response")


def normalize_ollama_response(raw: dict[str, Any]) -> dict[str, Any]:
    action = str(raw.get("action", "SKIPPED")).upper()
    if action not in VALID_ACTIONS:
        action = "SKIPPED"

    strategy = normalize_strategy_type(raw.get("strategy_type", "UNKNOWN"))

    risk_level = str(raw.get("risk_level", "MEDIUM")).upper()
    if risk_level not in VALID_RISK_LEVELS:
        risk_level = "MEDIUM"

    direction = str(raw.get("expected_direction", "SIDEWAYS")).upper()
    if direction not in VALID_DIRECTIONS:
        direction = "SIDEWAYS"

    return {
        "action": action,
        "strategy_type": strategy,
        "confidence": _clamp_confidence(raw.get("confidence", 0.0)),
        "rationale": str(raw.get("rationale") or raw.get("reasoning") or "").strip() or OLLAMA_FALLBACK_REASON,
        "risk_level": risk_level,
        "expected_direction": direction,
        "provider": OLLAMA_PROVIDER_TAG,
    }


def fallback_ollama_response(reason: str = OLLAMA_FALLBACK_REASON) -> dict[str, Any]:
    return {
        "action": "SKIPPED",
        "strategy_type": "UNKNOWN",
        "confidence": 0.0,
        "rationale": reason,
        "risk_level": "MEDIUM",
        "expected_direction": "SIDEWAYS",
        "provider": OLLAMA_PROVIDER_TAG,
    }


def parse_ollama_response_text(text: str) -> dict[str, Any]:
    try:
        raw = extract_json_object(text)
        return normalize_ollama_response(raw)
    except Exception as exc:
        log.warning("Local LLM unreachable or invalid response: %s", exc)
        return fallback_ollama_response(f"{OLLAMA_FALLBACK_REASON}: {exc}")


def _build_messages(prompt_text: str, context_json: dict[str, Any]) -> list[dict[str, str]]:
    schema_hint = (
        'Return ONLY a JSON object with keys: '
        '"action" (BUY|SELL|HOLD|SKIPPED), '
        '"strategy_type" (MOMENTUM|WHALE_FLOW|SENTIMENT|RISK_OFF|UNKNOWN), '
        '"confidence" (0.0-1.0), '
        '"rationale" (string), '
        '"risk_level" (LOW|MEDIUM|HIGH), '
        '"expected_direction" (UP|DOWN|SIDEWAYS), '
        '"provider" ("ollama_qwen3_8b").'
    )
    return [
        {
            "role": "system",
            "content": (
                "You are a crypto trading decision engine. "
                f"{schema_hint}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"{prompt_text}\n\n"
                f"Structured context JSON:\n{json.dumps(context_json, default=str)[:12000]}"
            ),
        },
    ]


def _ollama_chat_endpoint(base_url: str | None = None) -> str:
    """Normalize OLLAMA_BASE_URL and append native /api/chat."""
    raw = get_ollama_base_url() if base_url is None else base_url
    normalized = raw.strip().rstrip("/").removesuffix("/v1")
    return f"{normalized}/api/chat"


def _post_ollama_chat(payload: dict[str, Any]) -> dict[str, Any]:
    """
    POST to native Ollama /api/chat.
    Raises httpx / JSON errors for callers to handle.
    """
    endpoint = _ollama_chat_endpoint()
    timeout = get_ollama_timeout_seconds()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(endpoint, json=payload)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected Ollama response type: {type(data).__name__}")
        return data


def _decision_failure(reason: str, exc: BaseException | None = None) -> dict[str, Any]:
    record_ollama_error()
    if exc is not None:
        log.warning("Local LLM unreachable or invalid response: %s", exc)
        return fallback_ollama_response(f"{OLLAMA_FALLBACK_REASON}: {exc}")
    log.warning("Local LLM unreachable or invalid response: %s", reason)
    return fallback_ollama_response(reason)


def generate_decision(prompt_text: str, context_json: dict[str, Any]) -> dict[str, Any]:
    """
    Call local Ollama via native REST /api/chat and return normalized decision dict.
    Never raises — returns SKIPPED fallback on failure.
    """
    payload = {
        "model": get_ollama_model(),
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": 0,
        },
        "messages": _build_messages(prompt_text, context_json),
    }

    try:
        response_json = _post_ollama_chat(payload)

        if response_json.get("error"):
            err = response_json.get("error")
            return _decision_failure(f"ollama_error_response: {err}")

        message = response_json.get("message") or {}
        if not isinstance(message, dict):
            message = {}
        content = message.get("content") or ""
        thinking = message.get("thinking") or ""
        if not isinstance(content, str):
            content = str(content)
        if not isinstance(thinking, str):
            thinking = str(thinking)

        content = content.strip()
        thinking = thinking.strip()

        if content:
            record_ollama_call()
            return parse_ollama_response_text(content)

        if thinking:
            record_ollama_error()
            log.warning("Local LLM returned thinking-only response with empty content")
            return fallback_ollama_response("ollama_empty_content_thinking_only")

        return _decision_failure("ollama_empty_content")

    except httpx.TimeoutException as exc:
        return _decision_failure(f"{OLLAMA_FALLBACK_REASON}: {exc}", exc)
    except httpx.HTTPStatusError as exc:
        return _decision_failure(f"{OLLAMA_FALLBACK_REASON}: {exc}", exc)
    except httpx.RequestError as exc:
        return _decision_failure(f"{OLLAMA_FALLBACK_REASON}: {exc}", exc)
    except (ValueError, json.JSONDecodeError) as exc:
        return _decision_failure(f"{OLLAMA_FALLBACK_REASON}: {exc}", exc)
    except (KeyError, TypeError) as exc:
        return _decision_failure(f"{OLLAMA_FALLBACK_REASON}: {exc}", exc)
    except Exception as exc:
        return _decision_failure(f"{OLLAMA_FALLBACK_REASON}: {exc}", exc)


def generate_assistant_reply(
    *,
    user_message: str,
    context_json_text: str,
    history_text: str = "",
) -> str:
    """
    Free-form explanation assistant via Ollama native REST.
    No trade authority — text only. Raises on hard failures so caller can fallback.
    """
    system = (
        "You are the AI Assistant for a paper/demo memecoin trading workstation. "
        "You explain bot status, blockers, market context, sentiment, and demo PnL. "
        "You have NO trade authority — never place, approve, or claim to execute trades. "
        "Never claim profitability or live trading readiness. Never ask for private keys. "
        "Answer in clear, concise natural language."
    )
    user = (
        f"Runtime context JSON:\n{context_json_text[:14000]}\n\n"
        f"Conversation:\n{history_text[-4000:]}\n\n"
        f"user: {user_message}"
    )
    payload = {
        "model": get_ollama_model(),
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.3,
        },
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    response_json = _post_ollama_chat(payload)

    if response_json.get("error"):
        err = response_json.get("error")
        record_ollama_error()
        raise RuntimeError(f"Ollama error response: {err}")

    message = response_json.get("message") or {}
    if not isinstance(message, dict):
        message = {}
    content = message.get("content") or ""
    thinking = message.get("thinking") or ""
    if not isinstance(content, str):
        content = str(content)
    if not isinstance(thinking, str):
        thinking = str(thinking)

    content = content.strip()
    thinking = thinking.strip()

    if content:
        record_ollama_call()
        return content

    if thinking:
        record_ollama_error()
        raise RuntimeError("Ollama returned thinking-only response with no usable final content")

    record_ollama_error()
    return "No response from Ollama."
