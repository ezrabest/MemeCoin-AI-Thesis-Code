"""Qwen/Ollama linkage classification from existing records only (no LLM calls)."""

from __future__ import annotations

from typing import Any

from app.ae12_forward_evidence.types import QwenLinkageStatus, parse_ts


def _has_qwen_mention(obj: dict[str, Any] | None) -> bool:
    if not obj:
        return False
    blob = str(obj).lower()
    return "qwen" in blob


def _has_ollama_mention(obj: dict[str, Any] | None) -> bool:
    if not obj:
        return False
    blob = str(obj).lower()
    return "ollama" in blob


def _ae6_qwen_row_linked(ae6: dict[str, Any] | None) -> bool:
    if not ae6:
        return False
    llm = ae6.get("llm_context") if isinstance(ae6.get("llm_context"), dict) else {}
    # Require substantive linkage: memo available or text present, plus decision_id on row
    if not ae6.get("decision_id"):
        return False
    if llm.get("qwen_memo_available") is True:
        return True
    memo = llm.get("qwen_memo") or llm.get("qwen_text") or llm.get("qwen_summary")
    if memo:
        return True
    return False


def _ae6_qwen_mention_only(ae6: dict[str, Any] | None) -> bool:
    if not ae6:
        return False
    llm = ae6.get("llm_context") if isinstance(ae6.get("llm_context"), dict) else {}
    if "qwen_memo_available" in llm or "qwen" in str(llm).lower():
        return True
    return _has_qwen_mention(ae6)


def classify_qwen_ollama_linkage(
    *,
    opportunity: dict[str, Any],
    ae6: dict[str, Any] | None,
    ae9: dict[str, Any] | None,
    log_mentions_present: bool = False,
) -> dict[str, Any]:
    """
    Classify linkage accurately:
    - ROW_LINKED_AE9_RECORD when AE9 audit record ties to candidate/decision
    - ROW_LINKED_AE6_DECISION when AE6 has substantive Qwen content + decision_id
    - MENTION_ONLY when qwen fields exist but memo/content not linked as usable output
    - LOG_ONLY_NOT_ROW_LINKED when only log text exists without ids
    - ABSENT otherwise
    """
    candidate_id = opportunity.get("candidate_id")
    decision_id = opportunity.get("source_decision_id") or opportunity.get("decision_id")
    pair_address = opportunity.get("pair_address")
    ts = opportunity.get("first_seen_timestamp") or opportunity.get("created_at_utc")

    qwen_status = QwenLinkageStatus.ABSENT.value
    ollama_status = QwenLinkageStatus.ABSENT.value
    llm_record_source = None
    llm_verdict = opportunity.get("ae9_audit_verdict")
    llm_blockers = opportunity.get("ae9_audit_blockers") or []
    llm_summary_available = False
    llm_trade_authority_status = "NOT_TRADE_AUTHORITY"
    ts_diff_seconds = None
    pair_match_status = "UNKNOWN"

    # AE9 row linkage via explicit audit record id or matched AE9 index row
    ae9_id = opportunity.get("source_llm_audit_record_id")
    if ae9 and (ae9_id or ae9.get("audit_record_id")):
        ae9_cand = ae9.get("candidate_id")
        ae9_dec = ae9.get("source_decision_id") or ae9.get("decision_id")
        ids_match = False
        if candidate_id and ae9_cand and str(candidate_id) == str(ae9_cand):
            ids_match = True
        if decision_id and ae9_dec and str(decision_id) == str(ae9_dec):
            ids_match = True
        if ae9_id and str(ae9.get("audit_record_id") or "") == str(ae9_id):
            ids_match = True or ids_match
        # Opportunity points at AE9 record id — treat as row-linked when id present,
        # even if AE9 file unavailable; strengthen when object loaded.
        if ae9_id or ids_match:
            provider = str(ae9.get("llm_provider") or "").lower()
            if "ollama" in provider:
                ollama_status = QwenLinkageStatus.ROW_LINKED_AE9_RECORD.value
            else:
                # local qwen / mock / gemini adapters still may be under AE9
                if "qwen" in provider or provider in {"", "mock", "local_qwen"} or ae9_id:
                    qwen_status = QwenLinkageStatus.ROW_LINKED_AE9_RECORD.value
            llm_record_source = "ae9_llm_audit"
            llm_verdict = ae9.get("llm_verdict") or llm_verdict
            llm_blockers = ae9.get("audit_blockers") or llm_blockers
            llm_summary_available = bool(ae9.get("llm_response_parsed") or ae9.get("llm_verdict"))
            if ae9.get("llm_decision_authority") is True:
                llm_trade_authority_status = "CLAIMED_AUTHORITY_IN_RECORD"
            else:
                llm_trade_authority_status = "NO_TRADE_AUTHORITY"
            ae9_ts = parse_ts(ae9.get("audit_created_at_utc") or ae9.get("as_of_timestamp"))
            ev_ts = parse_ts(ts)
            if ae9_ts and ev_ts:
                ts_diff_seconds = abs((ae9_ts - ev_ts).total_seconds())
            ae9_pair = ae9.get("pair_address")
            if pair_address and ae9_pair:
                pair_match_status = "MATCH" if str(pair_address) == str(ae9_pair) else "MISMATCH"
            elif pair_address:
                pair_match_status = "AE9_PAIR_MISSING"
    elif ae9_id and not ae9:
        # Pointer exists but record not loaded — still row-linked pointer without body
        qwen_status = QwenLinkageStatus.ROW_LINKED_AE9_RECORD.value
        llm_record_source = "opportunity_source_llm_audit_record_id"
        llm_summary_available = bool(llm_verdict)

    # AE6 substantive qwen
    if qwen_status == QwenLinkageStatus.ABSENT.value and _ae6_qwen_row_linked(ae6):
        qwen_status = QwenLinkageStatus.ROW_LINKED_AE6_DECISION.value
        llm_record_source = llm_record_source or "ae6_decisions"
        llm_ctx = ae6.get("llm_context") if isinstance(ae6.get("llm_context"), dict) else {}
        llm_summary_available = bool(llm_ctx.get("qwen_memo") or llm_ctx.get("qwen_memo_available"))
        if llm_ctx.get("llm_decision_authority") is True:
            llm_trade_authority_status = "CLAIMED_AUTHORITY_IN_RECORD"
        ae6_ts = parse_ts(ae6.get("created_at_utc"))
        ev_ts = parse_ts(ts)
        if ae6_ts and ev_ts:
            ts_diff_seconds = abs((ae6_ts - ev_ts).total_seconds())
        ae6_pair = (ae6.get("candidate_identity") or {}).get("pair_address") if isinstance(
            ae6.get("candidate_identity"), dict
        ) else ae6.get("pair_address")
        if pair_address and ae6_pair:
            pair_match_status = "MATCH" if str(pair_address) == str(ae6_pair) else "MISMATCH"

    # Mention-only if AE6 has qwen keys but no memo
    if qwen_status == QwenLinkageStatus.ABSENT.value and _ae6_qwen_mention_only(ae6):
        qwen_status = QwenLinkageStatus.MENTION_ONLY.value
        llm_record_source = llm_record_source or "ae6_llm_context_fields"

    if ollama_status == QwenLinkageStatus.ABSENT.value and _has_ollama_mention(ae6):
        ollama_status = QwenLinkageStatus.MENTION_ONLY.value

    if qwen_status == QwenLinkageStatus.ABSENT.value and log_mentions_present and not (candidate_id or decision_id):
        qwen_status = QwenLinkageStatus.LOG_ONLY_NOT_ROW_LINKED.value
    elif qwen_status == QwenLinkageStatus.ABSENT.value and log_mentions_present:
        # logs mention qwen but this row has no attached record
        qwen_status = QwenLinkageStatus.LOG_ONLY_NOT_ROW_LINKED.value

    if ollama_status == QwenLinkageStatus.ABSENT.value and log_mentions_present and "ollama" in str(opportunity).lower():
        ollama_status = QwenLinkageStatus.MENTION_ONLY.value

    return {
        "candidate_id": candidate_id,
        "decision_id": decision_id,
        "pair_address": pair_address,
        "timestamp": ts,
        "qwen_linkage_status": qwen_status,
        "ollama_linkage_status": ollama_status,
        "llm_record_source": llm_record_source,
        "llm_verdict": llm_verdict,
        "llm_blockers": llm_blockers,
        "llm_summary_available": llm_summary_available,
        "llm_trade_authority_status": llm_trade_authority_status,
        "timestamp_diff_seconds": ts_diff_seconds,
        "pair_address_match_status": pair_match_status,
        "ae6_qwen_marker_present": bool(ae6 and _ae6_qwen_mention_only(ae6)),
    }


def summarize_linkage(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in rows:
        st = str(r.get("qwen_linkage_status") or "ABSENT")
        counts[st] = counts.get(st, 0) + 1
    return counts
