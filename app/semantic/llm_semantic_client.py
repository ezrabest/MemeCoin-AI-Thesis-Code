"""Audit-only LLM client for social/opportunistic semantic classification.

Routes to Qwen/Ollama or Gemini based on env. Never grants trade authority.
Does not import/initialize Gemini when the active provider is Qwen/Ollama.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from app.llm_config import (
    get_llm_provider,
    get_ollama_model,
    is_headless_data_collection,
    is_ollama_provider_active,
)

PROMPT_VERSION = "SOCIAL_OPP_SEMANTIC_V1"

SYSTEM_MESSAGE = (
    "You are classifying a crypto token for an academic meme-coin trading thesis. "
    "Your task is to classify behavioral identity, not current price action. "
    "Use only the evidence bundle provided. "
    "Do not invent facts. "
    "Do not use the user seed label as proof. "
    "If evidence is insufficient, say so. "
    "Return only valid JSON. "
    "The verdict has no trade authority."
)

CLASSIFICATION_CRITERIA = """
Classification criteria:
SOCIAL_CONFIRMED: credible evidence of real social/community/DAO/ReFi/public-goods/fan-community/
creator-economy/social-network/charity/ecosystem-purpose identity. More than symbol/name/hype/volume.
OPPORTUNISTIC_CONFIRMED: evidence of meme/parody, speculative launch, pump-oriented narrative,
celebrity/narrative opportunism, copycat, or no credible social purpose. Do not default when evidence missing.
INSUFFICIENT_EVIDENCE: unresolved identity, missing cache, ambiguous symbol, absent/weak/conflicting evidence.
""".strip()


def resolve_semantic_llm_provider() -> str:
    """Return active semantic LLM provider key: none | ollama | gemini."""
    if is_headless_data_collection():
        return "none"
    local = str(os.getenv("LOCAL_LLM_PROVIDER", "")).strip().lower()
    raw = str(os.getenv("LLM_PROVIDER", "")).strip().lower()
    if raw == "none" or get_llm_provider() == "none":
        return "none"
    if raw in ("ollama", "qwen") or local == "qwen" or is_ollama_provider_active():
        return "ollama"
    if raw == "gemini" or str(os.getenv("ENABLE_GEMINI", "")).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        # Only when not forced to ollama
        if raw not in ("ollama", "qwen") and local != "qwen":
            return "gemini"
    provider = get_llm_provider()
    if provider == "ollama":
        return "ollama"
    if provider == "gemini":
        return "gemini"
    return "none"


def build_user_message(
    *,
    identity: dict[str, Any],
    user_hypothesis: str,
    evidence_items: list[dict[str, Any]],
    counter_evidence: list[dict[str, Any]],
) -> str:
    payload = {
        "token_identity": {
            "symbol": identity.get("symbol"),
            "name": identity.get("name"),
            "chain": identity.get("chain"),
            "pair_address": identity.get("pair_address"),
            "token_address": identity.get("token_address"),
            "provider_url": identity.get("provider_url"),
        },
        "user_hypothesis": user_hypothesis or "",
        "evidence_items": evidence_items,
        "counter_evidence": counter_evidence,
        "classification_criteria": CLASSIFICATION_CRITERIA,
        "required_json_schema": {
            "semantic_status": "SOCIAL_CONFIRMED | OPPORTUNISTIC_CONFIRMED | INSUFFICIENT_EVIDENCE",
            "cluster_label": "SOCIALLY_MOTIVATED | OPPORTUNISTIC_SPECULATIVE | UNKNOWN",
            "confidence": 0.0,
            "evidence_quality": "HIGH | MEDIUM | LOW | NONE",
            "reasoning": "brief explanation",
            "supporting_evidence_ids": [],
            "counter_evidence_ids": [],
            "risk_notes": [],
        },
        "constraints": [
            "Do not invent facts.",
            "Do not treat user seed as proof.",
            "Do not classify from symbol/name/price/volume alone.",
            "no_trade_authority=true",
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _extract_json(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        raise ValueError("no_json_object_in_llm_response")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("llm_json_not_object")
    return data


def _normalize_llm_payload(data: dict[str, Any]) -> dict[str, Any]:
    status = str(data.get("semantic_status") or "").strip().upper()
    allowed_status = {
        "SOCIAL_CONFIRMED",
        "OPPORTUNISTIC_CONFIRMED",
        "INSUFFICIENT_EVIDENCE",
    }
    if status not in allowed_status:
        raise ValueError(f"invalid_semantic_status:{status}")

    label = str(data.get("cluster_label") or "").strip().upper()
    allowed_label = {
        "SOCIALLY_MOTIVATED",
        "OPPORTUNISTIC_SPECULATIVE",
        "UNKNOWN",
    }
    if label not in allowed_label:
        # Map consistent pairs
        if status == "SOCIAL_CONFIRMED":
            label = "SOCIALLY_MOTIVATED"
        elif status == "OPPORTUNISTIC_CONFIRMED":
            label = "OPPORTUNISTIC_SPECULATIVE"
        else:
            label = "UNKNOWN"

    # Consistency guard: status/label pairing
    if status == "SOCIAL_CONFIRMED" and label != "SOCIALLY_MOTIVATED":
        label = "SOCIALLY_MOTIVATED"
    if status == "OPPORTUNISTIC_CONFIRMED" and label != "OPPORTUNISTIC_SPECULATIVE":
        label = "OPPORTUNISTIC_SPECULATIVE"
    if status == "INSUFFICIENT_EVIDENCE":
        label = "UNKNOWN"

    try:
        confidence = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    eq = str(data.get("evidence_quality") or "NONE").strip().upper()
    if eq not in {"HIGH", "MEDIUM", "LOW", "NONE"}:
        eq = "NONE"

    return {
        "semantic_status": status,
        "cluster_label": label,
        "confidence": confidence,
        "evidence_quality": eq,
        "reasoning": str(data.get("reasoning") or "")[:600],
        "supporting_evidence_ids": list(data.get("supporting_evidence_ids") or [])[:20],
        "counter_evidence_ids": list(data.get("counter_evidence_ids") or [])[:20],
        "risk_notes": list(data.get("risk_notes") or [])[:20],
    }


def _call_ollama(system: str, user: str) -> dict[str, Any]:
    """Call Ollama/Qwen without importing Gemini."""
    import urllib.error
    import urllib.request

    from app.llm_config import get_ollama_base_url, get_ollama_timeout_seconds, record_ollama_call, record_ollama_error

    host = get_ollama_base_url().replace("/v1", "").rstrip("/")
    model = get_ollama_model()
    timeout = get_ollama_timeout_seconds()
    body = json.dumps(
        {
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "format": "json",
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
        if not isinstance(msg, str) or not msg.strip():
            record_ollama_error()
            return {
                "ok": False,
                "error": "empty_ollama_response",
                "provider": "QWEN_OLLAMA",
                "model": model,
            }
        record_ollama_call()
        parsed = _normalize_llm_payload(_extract_json(msg))
        return {
            "ok": True,
            "provider": "QWEN_OLLAMA",
            "model": model,
            "parsed": parsed,
            "raw_text": msg,
        }
    except Exception as exc:  # noqa: BLE001
        try:
            record_ollama_error()
        except Exception:
            pass
        return {
            "ok": False,
            "error": f"ollama_error:{type(exc).__name__}:{exc}",
            "provider": "QWEN_OLLAMA",
            "model": model,
        }


def _call_gemini(system: str, user: str) -> dict[str, Any]:
    """Call Gemini only when provider is gemini. Lazy import."""
    from app.llm_config import is_gemini_provider_active, record_gemini_call

    if not is_gemini_provider_active():
        return {
            "ok": False,
            "error": "gemini_provider_not_active",
            "provider": "GEMINI",
            "model": "",
        }
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return {
            "ok": False,
            "error": "GEMINI_API_KEY_missing",
            "provider": "GEMINI",
            "model": "",
        }
    try:
        import google.generativeai as genai

        genai.configure(api_key=key)
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        record_gemini_call()
        model = genai.GenerativeModel(
            model_name,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
            system_instruction=system,
        )
        response = model.generate_content(user)
        text = response.text or ""
        parsed = _normalize_llm_payload(_extract_json(text))
        return {
            "ok": True,
            "provider": "GEMINI",
            "model": model_name,
            "parsed": parsed,
            "raw_text": text,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "error": f"gemini_error:{type(exc).__name__}:{exc}",
            "provider": "GEMINI",
            "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        }


def call_semantic_llm(
    *,
    identity: dict[str, Any],
    user_hypothesis: str,
    evidence_items: list[dict[str, Any]],
    counter_evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Audit-only LLM classification call.

    Returns:
      ok, provider, model, parsed|error, prompt_version
    """
    provider = resolve_semantic_llm_provider()
    if provider == "none":
        return {
            "ok": False,
            "provider": "NONE",
            "model": "",
            "error": "llm_disabled",
            "prompt_version": PROMPT_VERSION,
        }

    user = build_user_message(
        identity=identity,
        user_hypothesis=user_hypothesis,
        evidence_items=evidence_items,
        counter_evidence=counter_evidence,
    )

    if provider == "ollama":
        result = _call_ollama(SYSTEM_MESSAGE, user)
    elif provider == "gemini":
        result = _call_gemini(SYSTEM_MESSAGE, user)
    else:
        result = {
            "ok": False,
            "provider": "NONE",
            "model": "",
            "error": f"unsupported_provider:{provider}",
        }
    result["prompt_version"] = PROMPT_VERSION
    return result
