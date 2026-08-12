"""Safety status resolution for Gemini semantic adjudication audits."""

from __future__ import annotations

from typing import Any


PASS_REJECTIONS_ENFORCED = "PASS_REJECTIONS_ENFORCED"
FAIL = "FAIL"


def resolve_safety_audit_status(safety: dict[str, Any] | None) -> str:
    """
    Derive safety_audit_status from enforcement outcomes.

    Forbidden language/keys that were rejected and never used as valid
    classifications must report PASS_REJECTIONS_ENFORCED, not FAIL.
    """
    s = safety or {}
    used_after = bool(s.get("output_used_after_rejection"))
    accepted_forbidden = int(s.get("accepted_classifications_with_forbidden_language") or 0)
    trade_authority = bool(s.get("trade_authority_used"))

    if used_after or accepted_forbidden > 0 or trade_authority:
        return FAIL

    forbidden_found = bool(s.get("forbidden_trade_language_found")) or bool(
        s.get("forbidden_trade_key_found")
    )
    rejected = int(s.get("rejected_outputs") or 0)

    if forbidden_found:
        # Offending outputs must have been rejected and unused.
        if rejected > 0 and not used_after and accepted_forbidden == 0:
            return PASS_REJECTIONS_ENFORCED
        return FAIL

    return PASS_REJECTIONS_ENFORCED


def build_safety_audit(
    *,
    total_gemini_outputs: int,
    accepted_outputs: int,
    rejected_outputs: int,
    forbidden_terms: set[str] | list[str],
    forbidden_keys: set[str] | list[str],
    output_used_after_rejection: bool = False,
    accepted_classifications_with_forbidden_language: int = 0,
    trade_authority_used: bool = False,
) -> dict[str, Any]:
    terms = sorted(set(forbidden_terms or []))
    keys = sorted(set(forbidden_keys or []))
    payload = {
        "total_gemini_outputs": total_gemini_outputs,
        "accepted_outputs": accepted_outputs,
        "rejected_outputs": rejected_outputs,
        "forbidden_trade_language_found": bool(terms),
        "forbidden_trade_key_found": bool(keys),
        "forbidden_terms_found": terms,
        "forbidden_keys_found": keys,
        "trade_authority_used": bool(trade_authority_used),
        "output_used_after_rejection": bool(output_used_after_rejection),
        "accepted_classifications_with_forbidden_language": int(
            accepted_classifications_with_forbidden_language
        ),
        "api_key_logged": False,
    }
    payload["status"] = resolve_safety_audit_status(payload)
    return payload


def gate_allowed_with_safety(gate_status: str, safety: dict[str, Any] | None) -> bool:
    """PASS adjudication gates require non-FAIL safety enforcement."""
    if resolve_safety_audit_status(safety) == FAIL:
        return False
    return gate_status in {
        "PASS_GEMINI_ADJUDICATION_READY",
        "PASS_WITH_OP_SUSPECTED_LIMITATION",
    }
