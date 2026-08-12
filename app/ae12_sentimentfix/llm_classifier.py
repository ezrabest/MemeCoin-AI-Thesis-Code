"""Local semantic classifier with strict safety gate; never trade authority."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from .classification_schema import (
    CLASSIFIER_VERSION,
    FORBIDDEN_TRADE_TERMS,
    RUBRIC_VERSION,
    normalize_classification_payload,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_strings(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_extract_strings(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_extract_strings(v))
    return out


def sanity_check_output(raw_llm_text: str, parsed_json: dict[str, Any] | None) -> dict[str, Any]:
    hay = [raw_llm_text or ""]
    if parsed_json:
        hay.extend(_extract_strings(parsed_json))
    blob = "\n".join(hay).upper()
    found = [t for t in FORBIDDEN_TRADE_TERMS if re.search(rf"\b{re.escape(t)}\b", blob)]
    forbidden = len(found) > 0
    return {
        "forbidden_trade_language_found": forbidden,
        "forbidden_terms_found": sorted(set(found)),
        "status": "REJECTED_FORBIDDEN_TRADE_LANGUAGE" if forbidden else "OK",
    }


def local_llm_available(*, local_llm_only: bool = True) -> dict[str, Any]:
    # Safe default for this phase: local LLM optional and not required.
    # We intentionally do not auto-call external APIs.
    if local_llm_only:
        return {"available": False, "model": "LOCAL_LLM_UNAVAILABLE"}
    return {"available": False, "model": "LOCAL_LLM_UNAVAILABLE"}


def _heuristic_classify(asset: dict[str, Any]) -> dict[str, Any]:
    text = (asset.get("evidence_text") or "").lower()
    symbol = str(asset.get("symbol") or "").upper()
    name = str(asset.get("name") or "").lower()

    pos: list[str] = []
    neg: list[str] = []
    markers: list[str] = []

    if any(k in text for k in ("charity", "donation", "ngo", "restoration", "public-good")):
        pos.append("charity_impact_mechanism")
        markers.append("charity/donation/ngo")
    if any(k in text for k in ("governance", "dao", "vote", "voting", "treasury")):
        pos.append("governance_dao")
        markers.append("dao/governance/voting")
    if any(k in text for k in ("community reward", "reward", "access token", "internal payment", "cooperative")):
        pos.append("community_utility")
        markers.append("community utility/reward")
    if any(k in text for k in ("volunteer", "civic", "environmental action")):
        pos.append("prosocial_rewards")
        markers.append("prosocial reward")

    if any(k in text for k in ("no utility", "meme", "speculation", "hype only", "hodl", "pump", "moon", "100x", "roi")):
        neg.append("no_utility_or_return_promise")
        markers.append("meme/speculation/roi language")
    if any(k in text for k in ("founder 30%", "insider 30%", "vc 30%", "centralized supply")):
        neg.append("extreme_launch_centralization")
        markers.append("launch centralization marker")

    infra_syms = {"BTC", "ETH"}
    if symbol in infra_syms or any(k in name for k in ("bitcoin", "ethereum", "infrastructure", "layer 1", "l1")):
        klass = "NON_SOCIAL_INFRASTRUCTURE"
        social = 0.2
        spec = 0.2
        conf = 0.75
    elif pos and not neg:
        klass = "SOCIAL"
        social = min(1.0, 0.7 + 0.1 * len(pos))
        spec = 0.2
        conf = 0.75
    elif neg:
        klass = "NON_SOCIAL_OPPORTUNISTIC"
        social = 0.1
        spec = min(1.0, 0.7 + 0.1 * len(neg))
        conf = 0.75
    else:
        klass = "UNKNOWN_INSUFFICIENT_EVIDENCE"
        social = 0.0
        spec = 0.0
        conf = 0.3

    manual = False
    if pos and neg:
        klass = "MANUAL_REVIEW"
        manual = True
        conf = 0.45

    return {
        "semantic_coin_class": klass,
        "semantic_social_score": round(float(social), 4),
        "speculation_score": round(float(spec), 4),
        "classification_confidence": round(float(conf), 4),
        "positive_criteria_met": pos,
        "negative_triggers_met": neg,
        "evidence_summary": "; ".join(markers[:4]),
        "reasoning_short": "heuristic local rubric classification (LLM optional/unavailable).",
        "evidence_quotes_or_markers": markers[:8],
        "requires_manual_review": manual,
    }


def classify_asset_semantic(
    asset: dict[str, Any],
    *,
    local_llm_only: bool = True,
) -> dict[str, Any]:
    llm = local_llm_available(local_llm_only=local_llm_only)
    classifier_status = "LOCAL_LLM_UNAVAILABLE"
    raw_llm_text = ""
    parsed_llm: dict[str, Any] | None = None

    # Current implementation intentionally uses deterministic local rubric when no local LLM.
    result = _heuristic_classify(asset)
    if llm.get("available"):
        classifier_status = "LOCAL_LLM_USED"
        # Placeholder path: if a local model is wired in future, parse JSON and pass through sanity gate.
        try:
            parsed_llm = json.loads(raw_llm_text) if raw_llm_text else None
        except json.JSONDecodeError:
            parsed_llm = None
        if parsed_llm:
            result = normalize_classification_payload(parsed_llm)
    else:
        result = normalize_classification_payload(result)

    safety = sanity_check_output(raw_llm_text, result)
    accepted = safety["status"] == "OK"
    if not accepted:
        result = normalize_classification_payload(
            {
                "semantic_coin_class": "MANUAL_REVIEW",
                "classification_confidence": 0.0,
                "reasoning_short": "Rejected by forbidden trade language safety gate.",
                "requires_manual_review": True,
            }
        )

    result.update(
        {
            "classifier_model": str(llm.get("model") or "LOCAL_RUBRIC_HEURISTIC"),
            "classifier_version": CLASSIFIER_VERSION,
            "rubric_version": RUBRIC_VERSION,
            "classified_at_utc": _utc_now(),
            "classifier_status": classifier_status,
            "trade_authority_used": False,
            "external_api_used": False,
            "safety_check": safety,
            "accepted": accepted,
        }
    )
    return result
