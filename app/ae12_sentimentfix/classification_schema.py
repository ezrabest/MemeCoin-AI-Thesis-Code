"""Schema helpers for AE12-SentimentFix semantic coin classification."""

from __future__ import annotations

from typing import Any

SEMANTIC_COIN_CLASSES: tuple[str, ...] = (
    "SOCIAL",
    "NON_SOCIAL_OPPORTUNISTIC",
    "NON_SOCIAL_INFRASTRUCTURE",
    "UNKNOWN_INSUFFICIENT_EVIDENCE",
    "MANUAL_REVIEW",
)

RUBRIC_VERSION = "AE12_SENTIMENTFIX_RUBRIC_V1"
CLASSIFIER_VERSION = "AE12_SENTIMENTFIX_SEMANTIC_CLASSIFIER_V1"

FORBIDDEN_TRADE_TERMS: tuple[str, ...] = (
    "BUY",
    "SELL",
    "EXECUTE",
    "PLACE_ORDER",
    "LIVE_BUY",
    "LIVE_SELL",
    "PAPER_BUY",
    "PAPER_SELL",
    "ENTRY_SIGNAL",
    "EXIT_SIGNAL",
    "TAKE_PROFIT",
    "STOP_LOSS",
    "LEVERAGE",
    "WALLET",
    "PRIVATE_KEY",
    "TRADE_NOW",
    "OPEN_POSITION",
    "CLOSE_POSITION",
)


def default_classification() -> dict[str, Any]:
    return {
        "semantic_coin_class": "UNKNOWN_INSUFFICIENT_EVIDENCE",
        "semantic_social_score": 0.0,
        "speculation_score": 0.0,
        "classification_confidence": 0.0,
        "positive_criteria_met": [],
        "negative_triggers_met": [],
        "evidence_summary": "",
        "reasoning_short": "",
        "evidence_quotes_or_markers": [],
        "requires_manual_review": False,
    }


def normalize_classification_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    out = default_classification()
    payload = payload or {}
    out.update({k: payload.get(k, out[k]) for k in out.keys()})
    klass = str(out.get("semantic_coin_class") or "").strip().upper()
    if klass not in SEMANTIC_COIN_CLASSES:
        klass = "UNKNOWN_INSUFFICIENT_EVIDENCE"
    out["semantic_coin_class"] = klass
    out["semantic_social_score"] = float(out.get("semantic_social_score") or 0.0)
    out["speculation_score"] = float(out.get("speculation_score") or 0.0)
    out["classification_confidence"] = float(out.get("classification_confidence") or 0.0)
    out["positive_criteria_met"] = [str(x) for x in (out.get("positive_criteria_met") or [])]
    out["negative_triggers_met"] = [str(x) for x in (out.get("negative_triggers_met") or [])]
    out["evidence_quotes_or_markers"] = [str(x) for x in (out.get("evidence_quotes_or_markers") or [])]
    out["requires_manual_review"] = bool(out.get("requires_manual_review", False))
    out["evidence_summary"] = str(out.get("evidence_summary") or "")
    out["reasoning_short"] = str(out.get("reasoning_short") or "")
    return out
