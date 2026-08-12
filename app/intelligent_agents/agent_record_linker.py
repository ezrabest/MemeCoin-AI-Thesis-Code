"""AE12.7 agent record linker — preserve IDs; never hallucinate missing ones."""

from __future__ import annotations

from typing import Any

LINK_ID_FIELDS = (
    "candidate_id",
    "candidate_policy_id",
    "target_row_id",
    "source_decision_id",
    "source_context_record_id",
    "source_llm_audit_record_id",
    "paper_order_id",
    "position_id",
    "pair_address",
    "event_timestamp",
    "first_seen_timestamp",
)


def extract_link_ids(row: dict[str, Any]) -> dict[str, Any]:
    """Pull linkage IDs from a candidate/opportunity/trade row. Missing → explicit None."""
    out: dict[str, Any] = {}
    for field in LINK_ID_FIELDS:
        val = row.get(field)
        if field == "source_decision_id" and val is None:
            val = row.get("decision_id")
        if field == "event_timestamp" and val is None:
            val = row.get("first_seen_timestamp") or row.get("created_at") or row.get("created_at_utc")
        if field == "first_seen_timestamp" and val is None:
            val = row.get("event_timestamp") or row.get("created_at") or row.get("created_at_utc")
        if val is None or str(val).strip() == "":
            out[field] = None
            out[f"missing_{field}"] = True
        else:
            out[field] = str(val)
            out[f"missing_{field}"] = False
    return out


def link_agent_to_demo_row(agent_record: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Merge linkage IDs into agent record without inventing values."""
    ids = extract_link_ids(row)
    linked = dict(agent_record)
    for field in LINK_ID_FIELDS:
        if linked.get(field) in (None, ""):
            linked[field] = ids.get(field)
        # Never overwrite a real ID with None from a partial row if already set
    missing = [f for f in LINK_ID_FIELDS if ids.get(f"missing_{f}")]
    linked["missing_id_flags"] = missing
    linked["linkage"] = {
        "linked_to_paper_trade": bool(ids.get("paper_order_id") or ids.get("position_id")),
        "linked_to_opportunity": bool(ids.get("candidate_id") or row.get("evidence_row_id")),
        "linked_to_missed_winner": bool(row.get("is_missed_winner") or row.get("missed_winner_horizons")),
        "linked_to_ae6_decision": bool(ids.get("source_decision_id")),
        "linked_to_ae8_context": bool(ids.get("source_context_record_id")),
        "linked_to_ae9_llm_audit": bool(ids.get("source_llm_audit_record_id")),
        "ids_hallucinated": False,
    }
    return linked


def build_linkage_row(agent_record: dict[str, Any], row: dict[str, Any] | None = None) -> dict[str, Any]:
    """Flat CSV-oriented linkage audit row."""
    ids = extract_link_ids(row or {})
    return {
        "agent_record_id": agent_record.get("agent_record_id"),
        "agent_type": agent_record.get("agent_type"),
        "agent_status": agent_record.get("agent_status"),
        "candidate_id": agent_record.get("candidate_id") or ids.get("candidate_id"),
        "source_decision_id": agent_record.get("source_decision_id") or ids.get("source_decision_id"),
        "source_context_record_id": agent_record.get("source_context_record_id") or ids.get("source_context_record_id"),
        "source_llm_audit_record_id": agent_record.get("source_llm_audit_record_id")
        or ids.get("source_llm_audit_record_id"),
        "paper_order_id": agent_record.get("paper_order_id") or ids.get("paper_order_id"),
        "position_id": agent_record.get("position_id") or ids.get("position_id"),
        "pair_address": agent_record.get("pair_address") or ids.get("pair_address"),
        "missing_ids": ";".join(
            f for f in LINK_ID_FIELDS if (agent_record.get(f) in (None, "") and ids.get(f) in (None, ""))
        ),
        "ids_hallucinated": False,
        "trade_authority_used": False,
        "decision_effect": agent_record.get("decision_effect"),
    }
