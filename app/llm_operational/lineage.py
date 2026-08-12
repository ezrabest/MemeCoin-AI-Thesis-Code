"""AE19 identity spine and lineage discipline — no symbol-only joins."""

from __future__ import annotations

from typing import Any

from app.clean_forward.price_source_identity import (
    build_price_source_key,
    synthesize_dexscreener_url,
)
from app.clean_forward.provider_url_key import try_normalize_provider_pair_url_key
from app.llm_operational.schema import (
    IDENTITY_AMBIGUOUS,
    IDENTITY_APPROVED_SPINE,
    IDENTITY_INVENTED_REJECTED,
    IDENTITY_SYMBOL_ONLY_REJECTED,
    IDENTITY_UNRESOLVED,
)

APPROVED_SPINE_FIELDS: tuple[str, ...] = (
    "provider_pair_url_exact",
    "canonical_market_identity",
    "normalized_provider_pair_url_key",
    "price_source_key",
    "candidate_id",
    "clean_forward_candidate_id",
    "decision_input_id",
    "pair_address",
    "chain",
    "base_token_address",
    "quote_token_address",
)

# Fields an LLM must never invent
PROTECTED_IDENTITY_FIELDS: tuple[str, ...] = (
    "pair_address",
    "base_token_address",
    "quote_token_address",
    "price_source_key",
    "provider_pair_url_exact",
    "canonical_market_identity",
    "normalized_provider_pair_url_key",
    "candidate_id",
    "clean_forward_candidate_id",
    "decision_input_id",
)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def extract_identity_spine(candidate: dict[str, Any]) -> dict[str, Any]:
    """Extract approved identity spine from candidate/context input. Never invent."""
    chain = _cell(candidate.get("chain"))
    pair = _cell(candidate.get("pair_address"))
    provider = _cell(candidate.get("provider") or "dexscreener") or "dexscreener"

    url_exact = _cell(
        candidate.get("provider_pair_url_exact")
        or candidate.get("provider_pair_url")
        or candidate.get("canonical_market_identity")
    )
    if not url_exact and chain and pair:
        url_exact = synthesize_dexscreener_url(chain, pair)

    psk = _cell(candidate.get("price_source_key"))
    if not psk and chain and pair:
        psk = build_price_source_key(provider, chain, pair)

    norm_key = ""
    if url_exact:
        try:
            key, _reason = try_normalize_provider_pair_url_key(url_exact)
            norm_key = key or ""
        except Exception:  # noqa: BLE001
            norm_key = ""

    canonical = _cell(candidate.get("canonical_market_identity")) or url_exact
    candidate_id = _cell(
        candidate.get("candidate_id")
        or candidate.get("clean_forward_candidate_id")
        or candidate.get("combined_target_id")
    )
    decision_input_id = _cell(
        candidate.get("decision_input_id")
        or candidate.get("clean_forward_decision_input_id")
    )

    symbol_pair = _cell(
        candidate.get("symbol_pair")
        or candidate.get("token_symbol")
        or candidate.get("symbol")
    )

    spine_present = bool(psk or (chain and pair) or candidate_id or url_exact or canonical)
    ambiguous = bool(symbol_pair and not spine_present)

    if spine_present:
        identity_status = IDENTITY_APPROVED_SPINE
        resolver_status = "RESOLVER_LINKED_APPROVED_SPINE"
    elif ambiguous:
        identity_status = IDENTITY_AMBIGUOUS
        resolver_status = IDENTITY_AMBIGUOUS
    else:
        identity_status = IDENTITY_UNRESOLVED
        resolver_status = IDENTITY_UNRESOLVED

    return {
        "candidate_id": candidate_id,
        "clean_forward_candidate_id": _cell(candidate.get("clean_forward_candidate_id")) or candidate_id,
        "decision_input_id": decision_input_id,
        "price_source_key": psk,
        "provider_pair_url_exact": url_exact,
        "canonical_market_identity": canonical,
        "normalized_provider_pair_url_key": norm_key,
        "pair_address": pair,
        "chain": chain,
        "base_token_address": _cell(candidate.get("base_token_address")),
        "quote_token_address": _cell(candidate.get("quote_token_address")),
        "symbol_pair": symbol_pair,  # display only
        "identity_status": identity_status,
        "resolver_status": resolver_status,
        "has_approved_spine": spine_present,
        "symbol_only": bool(symbol_pair and not spine_present),
    }


def reject_symbol_only_join(
    *,
    join_key_claimed: str | None,
    candidate: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject any attempt to join by symbol alone."""
    claimed = _cell(join_key_claimed).lower()
    symbol_keys = {"symbol", "symbol_pair", "token_symbol", "ticker", "name"}
    attempted = claimed in symbol_keys or claimed.startswith("symbol")
    if not attempted and candidate:
        # Implicit: only symbol present, no spine
        spine = extract_identity_spine(candidate)
        attempted = bool(spine.get("symbol_only"))

    return {
        "symbol_only_join_attempted": attempted,
        "symbol_only_join_rejected": attempted,
        "identity_status": IDENTITY_SYMBOL_ONLY_REJECTED if attempted else None,
        "resolver_status": IDENTITY_SYMBOL_ONLY_REJECTED if attempted else None,
        "join_key_claimed": join_key_claimed,
    }


def detect_llm_invented_identity(
    *,
    input_spine: dict[str, Any],
    llm_payload: dict[str, Any] | None,
    llm_text: str | None = None,
) -> dict[str, Any]:
    """Detect if LLM output invents new identity fields not present in input spine.

    Rules:
    - Empty LLM identity fields are fine
    - Matching input spine values are fine
    - New non-empty identity values that differ from input are invented → reject
    - Parsing free text for address-like invention is best-effort only
    """
    invented: list[dict[str, str]] = []
    payload = llm_payload or {}

    for field_name in PROTECTED_IDENTITY_FIELDS:
        incoming = _cell(payload.get(field_name))
        if not incoming:
            continue
        expected = _cell(input_spine.get(field_name))
        if not expected:
            invented.append(
                {
                    "field": field_name,
                    "llm_value": incoming,
                    "input_value": "",
                    "reason": "llm_invented_missing_input_identity",
                }
            )
        elif incoming.lower() != expected.lower():
            invented.append(
                {
                    "field": field_name,
                    "llm_value": incoming,
                    "input_value": expected,
                    "reason": "llm_identity_mismatch_invented",
                }
            )

    # Free-text heuristic: do not treat display symbol mentions as invention
    del llm_text  # reserved; identity invention is structured-field based

    detected = bool(invented)
    return {
        "llm_invented_identity_detected": detected,
        "llm_invented_identity_rejected": detected,
        "invented_fields": invented,
        "identity_status": IDENTITY_INVENTED_REJECTED if detected else input_spine.get("identity_status"),
        "resolver_status": IDENTITY_INVENTED_REJECTED if detected else input_spine.get("resolver_status"),
    }


def apply_lineage_to_record(
    record: dict[str, Any],
    spine: dict[str, Any],
    *,
    symbol_only_result: dict[str, Any] | None = None,
    invention_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach identity spine and enforce rejection flags on a task record."""
    out = dict(record)
    for key in APPROVED_SPINE_FIELDS:
        if key in spine:
            out[key] = spine.get(key) or out.get(key) or ""
    out["symbol_pair"] = spine.get("symbol_pair") or out.get("symbol_pair") or ""
    out["identity_status"] = spine.get("identity_status") or IDENTITY_UNRESOLVED
    out["resolver_status"] = spine.get("resolver_status") or IDENTITY_UNRESOLVED

    if symbol_only_result and symbol_only_result.get("symbol_only_join_rejected"):
        out["symbol_only_join_attempted"] = True
        out["symbol_only_join_rejected"] = True
        out["identity_status"] = IDENTITY_SYMBOL_ONLY_REJECTED
        out["resolver_status"] = IDENTITY_SYMBOL_ONLY_REJECTED
        out["downstream_eligible"] = False
        out["downstream_quarantined"] = True
        out["accepted_for_downstream"] = False
        out["failure_reason"] = out.get("failure_reason") or "SYMBOL_ONLY_JOIN_REJECTED"

    if invention_result and invention_result.get("llm_invented_identity_rejected"):
        out["identity_invention_detected"] = True
        out["identity_status"] = IDENTITY_INVENTED_REJECTED
        out["resolver_status"] = IDENTITY_INVENTED_REJECTED
        out["downstream_eligible"] = False
        out["downstream_quarantined"] = True
        out["accepted_for_downstream"] = False
        out["failure_reason"] = out.get("failure_reason") or "LLM_INVENTED_IDENTITY_REJECTED"
        out["safety_failed"] = True

    return out


def build_no_identity_invention_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    symbol_attempt = sum(1 for r in records if r.get("symbol_only_join_attempted"))
    symbol_rejected = sum(1 for r in records if r.get("symbol_only_join_rejected"))
    invented_detected = any(r.get("identity_invention_detected") for r in records)
    invented_rejected = sum(1 for r in records if r.get("identity_invention_detected"))
    unresolved = sum(1 for r in records if r.get("identity_status") == IDENTITY_UNRESOLVED)
    ambiguous = sum(1 for r in records if r.get("identity_status") == IDENTITY_AMBIGUOUS)
    approved = sum(1 for r in records if r.get("identity_status") == IDENTITY_APPROVED_SPINE)
    missing_spine = sum(
        1
        for r in records
        if r.get("identity_status") in {IDENTITY_UNRESOLVED, IDENTITY_AMBIGUOUS}
        or not (
            r.get("price_source_key")
            or (r.get("chain") and r.get("pair_address"))
            or r.get("clean_forward_candidate_id")
            or r.get("provider_pair_url_exact")
            or r.get("canonical_market_identity")
        )
    )

    # Hard fail if any accepted symbol-only join or accepted invented identity
    accepted_symbol_only = [
        r
        for r in records
        if r.get("symbol_only_join_attempted")
        and r.get("accepted_for_downstream")
        and not r.get("symbol_only_join_rejected")
    ]
    accepted_invented = [
        r
        for r in records
        if r.get("identity_invention_detected") and r.get("accepted_for_downstream")
    ]

    blocked = bool(accepted_symbol_only or accepted_invented)
    # Also block if invented identity was not rejected
    unrejected_invented = [
        r for r in records if r.get("identity_invention_detected") and not r.get("downstream_quarantined")
    ]
    if unrejected_invented:
        blocked = True

    return {
        "audit": "ae19_no_identity_invention_audit",
        "symbol_only_join_attempt_count": symbol_attempt,
        "symbol_only_join_rejected_count": symbol_rejected,
        "llm_invented_identity_detected": invented_detected,
        "llm_invented_identity_rejected_count": invented_rejected,
        "unresolved_identity_count": unresolved,
        "ambiguous_identity_count": ambiguous,
        "records_with_approved_identity_spine_count": approved,
        "records_missing_identity_spine_count": missing_spine,
        "accepted_symbol_only_join_count": len(accepted_symbol_only),
        "accepted_invented_identity_count": len(accepted_invented),
        "pass": not blocked,
        "block_code": "AE19_BLOCKED_IDENTITY_OR_LINEAGE_FAILURE" if blocked else None,
    }
