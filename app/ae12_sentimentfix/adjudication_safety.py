"""Safety gate for Gemini semantic adjudication outputs."""

from __future__ import annotations

import re
from typing import Any

from .adjudication_schema import FORBIDDEN_TRADE_KEYS
from .classification_schema import FORBIDDEN_TRADE_TERMS


def _extract_strings(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            out.extend(_extract_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_extract_strings(v))
    return out


def _find_forbidden_keys(obj: Any, *, prefix: str = "") -> list[str]:
    found: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k).lower()
            if key in FORBIDDEN_TRADE_KEYS:
                found.append(key)
            found.extend(_find_forbidden_keys(v, prefix=key))
    elif isinstance(obj, list):
        for v in obj:
            found.extend(_find_forbidden_keys(v, prefix=prefix))
    return found


def sanity_check_adjudication_output(raw_llm_text: str, parsed_json: dict[str, Any] | None) -> dict[str, Any]:
    hay = [raw_llm_text or ""]
    if parsed_json:
        hay.extend(_extract_strings(parsed_json))
    blob = "\n".join(hay).upper()
    found_terms = [t for t in FORBIDDEN_TRADE_TERMS if re.search(rf"\b{re.escape(t)}\b", blob)]
    found_keys = sorted(set(_find_forbidden_keys(parsed_json or {})))
    forbidden = bool(found_terms or found_keys)
    status = "REJECTED_FORBIDDEN_TRADE_LANGUAGE" if forbidden else "OK"
    return {
        "forbidden_trade_language_found": bool(found_terms),
        "forbidden_trade_key_found": bool(found_keys),
        "forbidden_terms_found": sorted(set(found_terms)),
        "forbidden_keys_found": found_keys,
        "status": status,
        "trade_authority_used": False,
    }


def redact_secrets(text: str) -> str:
    if not text:
        return text
    out = re.sub(r"AIza[0-9A-Za-z\-_]{20,}", "[REDACTED_API_KEY]", text)
    out = re.sub(r"(?i)(GEMINI_API_KEY|GOOGLE_API_KEY)\s*=\s*\S+", r"\1=[REDACTED]", out)
    return out
