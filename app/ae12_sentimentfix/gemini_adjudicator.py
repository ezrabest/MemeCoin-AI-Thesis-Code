"""Gemini semantic adjudicator for AE12-SentimentFix (reporting only, not trade authority)."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Callable

from .adjudication_safety import redact_secrets, sanity_check_adjudication_output
from .adjudication_schema import (
    ADJUDICATION_RUBRIC_VERSION,
    ADJUDICATOR_VERSION,
    default_adjudication,
    normalize_adjudication_payload,
)
from .web_evidence import extract_grounding_metadata, normalize_source_urls, web_grounding_available


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _api_key_present() -> bool:
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))


def gemini_client_available() -> bool:
    try:
        import google.generativeai as genai  # noqa: F401

        return True
    except ImportError:
        return False


def get_gemini_model_name() -> str:
    return os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


def get_gemini_hold_status(*, use_gemini: bool, allow_external_apis: bool) -> str | None:
    if not use_gemini or not allow_external_apis:
        return None
    if not _api_key_present():
        return "HOLD_GEMINI_API_KEY_MISSING"
    if not gemini_client_available():
        return "HOLD_GEMINI_CLIENT_UNAVAILABLE"
    return None


def _parse_json_response(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def build_adjudication_prompt(asset: dict[str, Any]) -> str:
    return f"""You are a semantic coin classifier for academic research reporting only.
You must NOT provide trade advice, buy/sell signals, position sizing, or execution instructions.
Return JSON only matching this schema:
{{
  "semantic_coin_class": "SOCIAL_CONFIRMED | NON_SOCIAL_OPPORTUNISTIC_CONFIRMED | NON_SOCIAL_INFRASTRUCTURE_CONFIRMED | OPPORTUNISTIC_SUSPECTED | MANUAL_REVIEW",
  "raw_evidence_status": "WEB_GROUNDED | MODEL_KNOWLEDGE_ONLY | INSUFFICIENT_EVIDENCE | IDENTITY_AMBIGUOUS | CONFLICTING_EVIDENCE",
  "semantic_social_score": 0.0,
  "opportunistic_score": 0.0,
  "infrastructure_score": 0.0,
  "classification_confidence": 0.0,
  "positive_criteria_met": [],
  "negative_triggers_met": [],
  "evidence_summary": "...",
  "reasoning_short": "...",
  "evidence_quotes_or_markers": [],
  "source_urls": [],
  "requires_manual_review": false
}}

Rubric:
- SOCIAL_CONFIRMED: community utility/payment/reward, DAO/governance, charity/impact mechanism, or pro-social rewards supported by evidence.
- NON_SOCIAL_INFRASTRUCTURE_CONFIRMED: broad L1/L2/base infrastructure asset (e.g. Bitcoin/Ethereum-like), not a specific social-impact token.
- NON_SOCIAL_OPPORTUNISTIC_CONFIRMED: clear evidence of meme/speculation-only, return-promise marketing, extreme launch centralization, or ordinary memecoin with no social mechanism.
- OPPORTUNISTIC_SUSPECTED: insufficient/conflicting/ambiguous evidence; default when not confirmable as social or infrastructure.
- MANUAL_REVIEW: conflicting sources, uncertain identity, impersonation/scam risk.

Asset identity:
- asset_id: {asset.get("asset_id")}
- chain: {asset.get("chain")}
- symbol: {asset.get("symbol")}
- name: {asset.get("name")}
- token_address: {asset.get("token_address")}
- pair_address: {asset.get("pair_address")}
- legacy_cluster_label: {asset.get("legacy_cluster_label")}

Local evidence package:
{asset.get("evidence_text") or "(none)"}

If public web evidence is available, cite source_urls. If using model knowledge without web grounding, set raw_evidence_status=MODEL_KNOWLEDGE_ONLY and lower confidence.
"""


def _heuristic_adjudicate(asset: dict[str, Any], *, raw_status: str = "INSUFFICIENT_EVIDENCE") -> dict[str, Any]:
    text = (asset.get("evidence_text") or "").lower()
    symbol = str(asset.get("symbol") or "").upper()
    name = str(asset.get("name") or "").lower()
    out = default_adjudication(raw_evidence_status=raw_status)
    pos: list[str] = []
    neg: list[str] = []
    if any(k in text for k in ("charity", "donation", "ngo", "restoration")):
        pos.append("charity_impact_mechanism")
    if any(k in text for k in ("governance", "dao", "vote", "voting")):
        pos.append("governance_dao")
    if any(k in text for k in ("community reward", "access token", "cooperative", "internal payment")):
        pos.append("community_utility")
    if any(k in text for k in ("meme", "moon", "pump", "100x", "roi", "no utility", "hodl")):
        neg.append("speculation_or_meme")
    if symbol in {"BTC", "ETH"} or any(k in name for k in ("bitcoin", "ethereum")):
        out.update(
            {
                "semantic_coin_class": "NON_SOCIAL_INFRASTRUCTURE_CONFIRMED",
                "infrastructure_score": 0.8,
                "classification_confidence": 0.75,
                "reasoning_short": "Infrastructure/base asset heuristic.",
            }
        )
    elif pos and not neg:
        out.update(
            {
                "semantic_coin_class": "SOCIAL_CONFIRMED",
                "semantic_social_score": 0.8,
                "classification_confidence": 0.75,
                "positive_criteria_met": pos,
                "reasoning_short": "Social mechanism markers in evidence.",
            }
        )
    elif neg and not pos:
        out.update(
            {
                "semantic_coin_class": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
                "opportunistic_score": 0.8,
                "classification_confidence": 0.75,
                "negative_triggers_met": neg,
                "reasoning_short": "Speculative/memecoin markers in evidence.",
            }
        )
    else:
        out.update(
            {
                "semantic_coin_class": "OPPORTUNISTIC_SUSPECTED",
                "opportunistic_score": 0.5,
                "classification_confidence": 0.3,
                "reasoning_short": "Insufficient evidence for confirmed class; default suspected opportunistic.",
            }
        )
    return normalize_adjudication_payload(out)


def adjudicate_asset(
    asset: dict[str, Any],
    *,
    use_gemini: bool = False,
    allow_external_apis: bool = False,
    allow_model_knowledge_fallback: bool = False,
    no_web_grounding: bool = False,
    dry_run: bool = False,
    gemini_call: Callable[[str], tuple[str, Any]] | None = None,
) -> dict[str, Any]:
    hold = get_gemini_hold_status(use_gemini=use_gemini, allow_external_apis=allow_external_apis)
    if dry_run or hold:
        raw = "GEMINI_NOT_RUN" if hold or dry_run else "INSUFFICIENT_EVIDENCE"
        result = _heuristic_adjudicate(asset, raw_status=raw)
        if hold:
            result["semantic_coin_class"] = "OPPORTUNISTIC_SUSPECTED"
            result["raw_evidence_status"] = "GEMINI_NOT_RUN"
            result["reasoning_short"] = f"Gemini not run ({hold}). Default OP.SUSPECTED."
        result.update(
            {
                "gemini_model": "DRY_RUN" if dry_run else "GEMINI_NOT_RUN",
                "adjudicator_version": ADJUDICATOR_VERSION,
                "rubric_version": ADJUDICATION_RUBRIC_VERSION,
                "classified_at_utc": _utc_now(),
                "linkage_method": "NO_LINK",
                "external_api_used": False,
                "gemini_used": False,
                "web_grounding_used": False,
                "trade_authority_used": False,
                "hold_status": hold,
            }
        )
        return result

    model_name = get_gemini_model_name()
    prompt = build_adjudication_prompt(asset)
    raw_text = ""
    response_obj: Any = None
    linkage_method = "GEMINI_MODEL_KNOWLEDGE_ONLY"
    web_grounding_used = False
    try:
        if gemini_call is not None:
            raw_text, response_obj = gemini_call(prompt)
        else:
            import google.generativeai as genai

            genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
            model = genai.GenerativeModel(
                model_name,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json",
                    temperature=0.2,
                ),
            )
            response_obj = model.generate_content(prompt)
            raw_text = redact_secrets(getattr(response_obj, "text", "") or "")
        grounding = extract_grounding_metadata(response_obj) if response_obj is not None else {}
        web_grounding_used = bool(grounding.get("web_grounding_used"))
        if web_grounding_used:
            linkage_method = "GEMINI_WEB_GROUNDED"
        elif allow_model_knowledge_fallback:
            linkage_method = "GEMINI_MODEL_KNOWLEDGE_ONLY"
        else:
            result = default_adjudication(raw_evidence_status="WEB_EVIDENCE_MISSING")
            result["semantic_coin_class"] = "OPPORTUNISTIC_SUSPECTED"
            result["reasoning_short"] = "Web grounding unavailable and model-knowledge fallback not allowed."
            result.update(
                {
                    "gemini_model": model_name,
                    "adjudicator_version": ADJUDICATOR_VERSION,
                    "rubric_version": ADJUDICATION_RUBRIC_VERSION,
                    "classified_at_utc": _utc_now(),
                    "linkage_method": "NO_LINK",
                    "external_api_used": True,
                    "gemini_used": True,
                    "web_grounding_used": False,
                    "trade_authority_used": False,
                }
            )
            return result
        parsed = normalize_adjudication_payload(_parse_json_response(raw_text))
        if web_grounding_used:
            parsed["raw_evidence_status"] = "WEB_GROUNDED"
            parsed["source_urls"] = normalize_source_urls(
                list(parsed.get("source_urls") or []) + list(grounding.get("source_urls") or [])
            )
        elif allow_model_knowledge_fallback:
            parsed["raw_evidence_status"] = "MODEL_KNOWLEDGE_ONLY"
            parsed["classification_confidence"] = min(float(parsed.get("classification_confidence") or 0.0), 0.55)
            limitation = "Gemini model knowledge used without web grounding."
            parsed["reasoning_short"] = (
                ((parsed.get("reasoning_short") or "") + " " + limitation).strip()
            )
        safety = sanity_check_adjudication_output(raw_text, parsed)
        accepted = safety["status"] == "OK"
        if not accepted:
            parsed = default_adjudication(raw_evidence_status="REJECTED_FOR_TRADE_LANGUAGE")
            parsed["semantic_coin_class"] = "MANUAL_REVIEW"
            parsed["requires_manual_review"] = True
            parsed["reasoning_short"] = "Rejected by trade-language safety gate."
        parsed.update(
            {
                "gemini_model": model_name,
                "adjudicator_version": ADJUDICATOR_VERSION,
                "rubric_version": ADJUDICATION_RUBRIC_VERSION,
                "classified_at_utc": _utc_now(),
                "linkage_method": linkage_method,
                "external_api_used": True,
                "gemini_used": True,
                "web_grounding_used": web_grounding_used,
                "trade_authority_used": False,
                "safety_check": safety,
                "accepted": accepted,
                "raw_llm_text_redacted": redact_secrets(raw_text)[:2000],
            }
        )
        return parsed
    except json.JSONDecodeError:
        # retry once with repair prompt
        try:
            repair_prompt = prompt + "\nReturn valid JSON only. No markdown."
            if gemini_call is not None:
                raw_text, response_obj = gemini_call(repair_prompt)
            else:
                import google.generativeai as genai

                genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
                model = genai.GenerativeModel(
                    model_name,
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        temperature=0.0,
                    ),
                )
                response_obj = model.generate_content(repair_prompt)
                raw_text = redact_secrets(getattr(response_obj, "text", "") or "")
            parsed = normalize_adjudication_payload(_parse_json_response(raw_text))
            safety = sanity_check_adjudication_output(raw_text, parsed)
            if safety["status"] != "OK":
                raise ValueError("safety rejected")
            parsed.update(
                {
                    "gemini_model": model_name,
                    "adjudicator_version": ADJUDICATOR_VERSION,
                    "rubric_version": ADJUDICATION_RUBRIC_VERSION,
                    "classified_at_utc": _utc_now(),
                    "linkage_method": linkage_method,
                    "external_api_used": True,
                    "gemini_used": True,
                    "web_grounding_used": web_grounding_used,
                    "trade_authority_used": False,
                    "safety_check": safety,
                    "accepted": True,
                }
            )
            return parsed
        except Exception:
            bad = default_adjudication(raw_evidence_status="INVALID_LLM_OUTPUT")
            bad["semantic_coin_class"] = "OPPORTUNISTIC_SUSPECTED"
            bad["requires_manual_review"] = True
            bad["reasoning_short"] = "Invalid Gemini JSON after retry."
            bad.update(
                {
                    "gemini_model": model_name,
                    "external_api_used": True,
                    "gemini_used": True,
                    "web_grounding_used": False,
                    "trade_authority_used": False,
                    "accepted": False,
                    "raw_llm_text_redacted": redact_secrets(raw_text)[:2000],
                }
            )
            return bad
    except Exception:
        bad = default_adjudication(raw_evidence_status="INSUFFICIENT_EVIDENCE")
        bad["semantic_coin_class"] = "OPPORTUNISTIC_SUSPECTED"
        bad["reasoning_short"] = "Gemini adjudication error; default OP.SUSPECTED."
        bad.update(
            {
                "gemini_model": model_name,
                "external_api_used": True,
                "gemini_used": True,
                "web_grounding_used": False,
                "trade_authority_used": False,
                "accepted": False,
            }
        )
        return bad
