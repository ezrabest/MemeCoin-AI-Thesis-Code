"""Schema for AE12-SentimentFix Gemini semantic adjudication (reporting only)."""

from __future__ import annotations

from typing import Any

ADJUDICATION_CLASSES: tuple[str, ...] = (
    "SOCIAL_CONFIRMED",
    "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
    "NON_SOCIAL_INFRASTRUCTURE_CONFIRMED",
    "OPPORTUNISTIC_SUSPECTED",
    "MANUAL_REVIEW",
)

RAW_EVIDENCE_STATUSES: tuple[str, ...] = (
    "WEB_GROUNDED",
    "MODEL_KNOWLEDGE_ONLY",
    "INSUFFICIENT_EVIDENCE",
    "IDENTITY_AMBIGUOUS",
    "CONFLICTING_EVIDENCE",
    "UNKNOWN_INSUFFICIENT_EVIDENCE",
    "GEMINI_NOT_RUN",
    "INVALID_LLM_OUTPUT",
    "REJECTED_FOR_TRADE_LANGUAGE",
    "WEB_EVIDENCE_MISSING",
    "LLM_UNAVAILABLE",
)

ADJUDICATOR_VERSION = "AE12_SENTIMENTFIX_GEMINI_ADJUDICATOR_V1"
ADJUDICATION_RUBRIC_VERSION = "AE12_SENTIMENTFIX_ADJUDICATION_RUBRIC_V1"

FORBIDDEN_TRADE_KEYS: tuple[str, ...] = (
    "trade_direction",
    "position_size",
    "entry_signal",
    "exit_signal",
    "buy_signal",
    "sell_signal",
    "order_type",
    "order_size",
    "stop_loss",
    "take_profit",
    "leverage",
    "wallet",
    "private_key",
    "execution_action",
    "trade_action",
    "paper_trade_action",
    "live_trade_action",
)

UI_LABELS: dict[str, str] = {
    "SOCIAL_CONFIRMED": "SOCIAL_CONFIRMED",
    "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
    "OPPORTUNISTIC_SUSPECTED": "OP.SUSPECTED",
    "NON_SOCIAL_INFRASTRUCTURE_CONFIRMED": "NON_SOCIAL_INFRASTRUCTURE_CONFIRMED",
    "MANUAL_REVIEW": "MANUAL_REVIEW",
}


def default_adjudication(*, raw_evidence_status: str = "GEMINI_NOT_RUN") -> dict[str, Any]:
    return {
        "semantic_coin_class": "OPPORTUNISTIC_SUSPECTED",
        "raw_evidence_status": raw_evidence_status,
        "semantic_social_score": 0.0,
        "opportunistic_score": 0.5,
        "infrastructure_score": 0.0,
        "classification_confidence": 0.25,
        "positive_criteria_met": [],
        "negative_triggers_met": [],
        "evidence_summary": "",
        "reasoning_short": "",
        "evidence_quotes_or_markers": [],
        "source_urls": [],
        "requires_manual_review": False,
    }


def map_local_class_to_adjudication(local_class: str | None) -> str:
    """Map legacy local classifier buckets to adjudication buckets."""
    klass = str(local_class or "").strip().upper()
    mapping = {
        "SOCIAL": "SOCIAL_CONFIRMED",
        "NON_SOCIAL_OPPORTUNISTIC": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
        "NON_SOCIAL_INFRASTRUCTURE": "NON_SOCIAL_INFRASTRUCTURE_CONFIRMED",
        "MANUAL_REVIEW": "MANUAL_REVIEW",
        "UNKNOWN_INSUFFICIENT_EVIDENCE": "OPPORTUNISTIC_SUSPECTED",
    }
    return mapping.get(klass, "OPPORTUNISTIC_SUSPECTED")


def normalize_adjudication_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    out = default_adjudication()
    payload = payload or {}
    out.update({k: payload.get(k, out[k]) for k in out.keys()})
    klass = str(out.get("semantic_coin_class") or "").strip().upper()
    if klass not in ADJUDICATION_CLASSES:
        klass = "OPPORTUNISTIC_SUSPECTED"
    out["semantic_coin_class"] = klass
    raw = str(out.get("raw_evidence_status") or "INSUFFICIENT_EVIDENCE").strip().upper()
    if raw not in RAW_EVIDENCE_STATUSES:
        raw = "INSUFFICIENT_EVIDENCE"
    out["raw_evidence_status"] = raw
    out["semantic_social_score"] = float(out.get("semantic_social_score") or 0.0)
    out["opportunistic_score"] = float(out.get("opportunistic_score") or 0.0)
    out["infrastructure_score"] = float(out.get("infrastructure_score") or 0.0)
    out["classification_confidence"] = float(out.get("classification_confidence") or 0.0)
    out["positive_criteria_met"] = [str(x) for x in (out.get("positive_criteria_met") or [])]
    out["negative_triggers_met"] = [str(x) for x in (out.get("negative_triggers_met") or [])]
    out["evidence_quotes_or_markers"] = [str(x) for x in (out.get("evidence_quotes_or_markers") or [])]
    out["source_urls"] = [str(x) for x in (out.get("source_urls") or [])]
    out["requires_manual_review"] = bool(out.get("requires_manual_review", False))
    out["evidence_summary"] = str(out.get("evidence_summary") or "")
    out["reasoning_short"] = str(out.get("reasoning_short") or "")
    return out


def is_confirmed_opportunistic(semantic_coin_class: str) -> bool:
    return semantic_coin_class == "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED"


def is_suspected_opportunistic(semantic_coin_class: str) -> bool:
    return semantic_coin_class == "OPPORTUNISTIC_SUSPECTED"
