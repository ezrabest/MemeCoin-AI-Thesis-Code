"""AE19 LLM Operational Layer orchestrator."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.consensus.serialization import write_csv, write_json, write_jsonl, write_text
from app.llm_operational import audit as audit_mod
from app.llm_operational.gemini_runtime import run_gemini_operational
from app.llm_operational.lineage import (
    apply_lineage_to_record,
    detect_llm_invented_identity,
    extract_identity_spine,
)
from app.llm_operational.providers import (
    resolve_gemini_provider_status,
    resolve_qwen_provider_status,
)
from app.llm_operational.qwen_runtime import run_qwen_operational
from app.llm_operational.safety import apply_safety_to_record
from app.llm_operational.schema import (
    ENGINE_VERSION,
    PHASE,
    PROMPT_TEMPLATE_VERSION,
    PROVIDER_AVAILABLE,
    SAFETY_BOUNDARY,
    TASK_AUDIT,
    TASK_CANDIDATE_MEMO,
    TASK_CONTEXT_SUMMARY,
    TASK_MISSED_WINNER_REVIEW,
    TASK_RECORD_FIELDS,
    TASK_RISK_EXPLANATION,
    TASK_SEMANTIC_CONFLICT_REVIEW,
    TASK_SKIPPED_INPUT,
    TASK_SUCCEEDED,
    TASK_TYPES,
    MISSED_WINNER_UNAVAILABLE,
)
from app.llm_operational.task_builder import (
    build_prompt_for_task,
    discover_task_candidates,
    match_outcome_for_candidate,
    sha256_text,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _new_task_id(task_type: str) -> str:
    return f"ae19_{task_type.lower()}_{uuid4().hex[:12]}"


def _base_record(
    *,
    task_type: str,
    candidate: dict[str, Any],
    prompt_meta: dict[str, Any],
    evidence_refs: list[str],
) -> dict[str, Any]:
    spine = prompt_meta.get("spine") or extract_identity_spine(candidate)
    now = utc_now()
    return {
        "ae19_task_id": _new_task_id(task_type),
        "task_type": task_type,
        "provider": "",
        "provider_model": "",
        "provider_status": "",
        "task_status": "",
        "candidate_id": spine.get("candidate_id") or "",
        "clean_forward_candidate_id": spine.get("clean_forward_candidate_id") or "",
        "decision_input_id": spine.get("decision_input_id") or "",
        "price_source_key": spine.get("price_source_key") or "",
        "provider_pair_url_exact": spine.get("provider_pair_url_exact") or "",
        "canonical_market_identity": spine.get("canonical_market_identity") or "",
        "normalized_provider_pair_url_key": spine.get("normalized_provider_pair_url_key") or "",
        "pair_address": spine.get("pair_address") or "",
        "chain": spine.get("chain") or "",
        "base_token_address": spine.get("base_token_address") or "",
        "quote_token_address": spine.get("quote_token_address") or "",
        "symbol_pair": spine.get("symbol_pair") or "",
        "evidence_refs": evidence_refs,
        "model_evidence_refs": [r for r in evidence_refs if "model" in r.lower() or "ae16" in r.lower() or "ae17" in r.lower()],
        "consensus_refs": [r for r in evidence_refs if "consensus" in r.lower()],
        "meta_refs": [r for r in evidence_refs if "meta" in r.lower() or "ae17" in r.lower()],
        "context_refs": [r for r in evidence_refs if "ae18" in r.lower() or "context" in r.lower()],
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "prompt_text_hash": prompt_meta.get("prompt_text_hash") or "",
        "response_text_hash": "",
        "created_at": now,
        "completed_at": "",
        "failure_reason": "",
        "mock_used": False,
        "counted_as_real_provider_success": False,
        "downstream_eligible": False,
        "downstream_quarantined": True,
        "safety_status": "PASS_NO_TRADE_AUTHORITY",
        "trade_authority_used": False,
        "live_trading_approved": False,
        "risk_override_used": False,
        "wallet_accessed": False,
        "identity_status": spine.get("identity_status") or "",
        "resolver_status": spine.get("resolver_status") or "",
        "output_text": "",
        "output_summary": "",
        "accepted_for_downstream": False,
        "safety_failed": False,
        "forbidden_language_hits": [],
        "missed_winner_status": prompt_meta.get("missed_winner_status") or "",
        "allowed_language_tags": [],
        "raw_response_preserved": "",
        "identity_invention_detected": False,
        "symbol_only_join_attempted": bool((prompt_meta.get("symbol_check") or {}).get("symbol_only_join_attempted")),
        "symbol_only_join_rejected": bool((prompt_meta.get("symbol_check") or {}).get("symbol_only_join_rejected")),
    }


def _finalize_record(
    record: dict[str, Any],
    provider_result: dict[str, Any],
    *,
    spine: dict[str, Any],
    symbol_check: dict[str, Any] | None = None,
    parse_identity_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    out = dict(record)
    out["provider"] = provider_result.get("provider") or out.get("provider")
    out["provider_model"] = provider_result.get("provider_model") or ""
    out["provider_status"] = provider_result.get("provider_status") or ""
    out["task_status"] = provider_result.get("task_status") or ""
    out["mock_used"] = bool(provider_result.get("mock_used"))
    out["counted_as_real_provider_success"] = bool(provider_result.get("counted_as_real_provider_success"))
    out["downstream_eligible"] = bool(provider_result.get("downstream_eligible"))
    out["downstream_quarantined"] = bool(provider_result.get("downstream_quarantined", True))
    out["accepted_for_downstream"] = bool(provider_result.get("accepted_for_downstream"))
    out["failure_reason"] = provider_result.get("failure_reason") or out.get("failure_reason") or ""
    text = provider_result.get("text") or ""
    out["output_text"] = text
    out["output_summary"] = text[:500] if text else out.get("output_summary") or ""
    out["response_text_hash"] = sha256_text(text) if text else ""
    out["completed_at"] = utc_now()
    out["raw_response_preserved"] = text

    # Identity invention check on structured payload only (LLMs shouldn't invent identity)
    invention = detect_llm_invented_identity(
        input_spine=spine,
        llm_payload=parse_identity_payload,
        llm_text=text,
    )
    out = apply_lineage_to_record(out, spine, symbol_only_result=symbol_check, invention_result=invention)
    out = apply_safety_to_record(out, text)

    # Mock hard rules (re-assert after safety)
    if out.get("mock_used"):
        out["counted_as_real_provider_success"] = False
        out["downstream_eligible"] = False
        out["downstream_quarantined"] = True
        out["accepted_for_downstream"] = False

    # Authority hard false
    out["trade_authority_used"] = False
    out["live_trading_approved"] = False
    out["risk_override_used"] = False
    out["wallet_accessed"] = False
    return out


def _run_provider_for_task(
    *,
    prompt: str,
    task_type: str,
    candidate: dict[str, Any],
    prefer_gemini: bool,
    allow_qwen: bool | None,
    allow_gemini: bool | None,
    force_qwen_unavailable: bool,
    force_gemini_unavailable: bool,
    use_mock_diagnostic: bool,
    qwen_status: dict[str, Any],
    gemini_status: dict[str, Any],
) -> dict[str, Any]:
    if use_mock_diagnostic:
        # Prefer qwen mock path for primary operational tasks
        return run_qwen_operational(
            prompt,
            task_type=task_type,
            candidate=candidate,
            use_mock_diagnostic=True,
        )

    if prefer_gemini:
        return run_gemini_operational(
            prompt,
            task_type=task_type,
            candidate=candidate,
            allow_gemini=allow_gemini,
            force_unavailable=force_gemini_unavailable,
            provider_status_cache=gemini_status,
        )

    # Default operational provider: Qwen/Ollama
    result = run_qwen_operational(
        prompt,
        task_type=task_type,
        candidate=candidate,
        allow_qwen=allow_qwen,
        force_unavailable=force_qwen_unavailable,
        provider_status_cache=qwen_status,
    )
    # If Qwen unavailable and Gemini available, optional selective fallback for audit-ish tasks
    if (
        result.get("provider_status") != PROVIDER_AVAILABLE
        and str(gemini_status.get("provider_status")) == PROVIDER_AVAILABLE
        and task_type in {TASK_SEMANTIC_CONFLICT_REVIEW, TASK_AUDIT}
    ):
        return run_gemini_operational(
            prompt,
            task_type=task_type,
            candidate=candidate,
            allow_gemini=allow_gemini,
            force_unavailable=force_gemini_unavailable,
            provider_status_cache=gemini_status,
        )
    return result


def run_ae19_llm_operational_layer(
    project_root: Path,
    *,
    output_root: str | Path | None = None,
    ae17_root: str | Path | None = None,
    ae16_root: str | Path | None = None,
    ae18_root: str | Path | None = None,
    max_candidates: int = 20,
    max_tasks_per_type: int = 20,
    allow_qwen: bool | None = None,
    allow_gemini: bool | None = None,
    force_qwen_unavailable: bool = False,
    force_gemini_unavailable: bool = False,
    use_mock_diagnostic: bool = False,
    fixture_candidates: list[dict[str, Any]] | None = None,
    gemini_selective_budget: int = 3,
) -> dict[str, Any]:
    """
    Run AE19 LLM Operational Layer.

    LLMs have no trade authority. Never mutates trader.db. Never connects wallet.
    """
    root = project_root.resolve()
    stamp = utc_stamp()
    out = Path(output_root) if output_root else root / "data" / "audits" / f"ae19_llm_operational_layer_{stamp}"
    if not out.is_absolute():
        out = root / out
    data_dir = out / "data"
    audits_dir = out / "audits"
    reports_dir = out / "reports"
    for d in (data_dir, audits_dir, reports_dir):
        d.mkdir(parents=True, exist_ok=True)

    inputs = discover_task_candidates(
        root,
        ae17_root=ae17_root,
        ae16_root=ae16_root,
        ae18_root=ae18_root,
        max_candidates=max_candidates,
        fixture_candidates=fixture_candidates,
    )
    candidates = list(inputs.get("candidates") or [])[:max_candidates]
    context_by = inputs.get("context_by_candidate") or {}
    outcome_rows = inputs.get("outcome_rows") or []
    outcome_meta = inputs.get("outcome_evidence") or {}
    outcome_available_global = bool(outcome_meta.get("found"))

    qwen_status = resolve_qwen_provider_status(
        allow_qwen=allow_qwen,
        force_unavailable=force_qwen_unavailable,
    )
    gemini_status = resolve_gemini_provider_status(
        allow_gemini=allow_gemini,
        force_unavailable=force_gemini_unavailable,
    )

    records: list[dict[str, Any]] = []
    gemini_calls = 0
    operational_types = [
        TASK_CANDIDATE_MEMO,
        TASK_RISK_EXPLANATION,
        TASK_MISSED_WINNER_REVIEW,
        TASK_SEMANTIC_CONFLICT_REVIEW,
        TASK_CONTEXT_SUMMARY,
    ]

    for task_type in operational_types:
        type_count = 0
        for cand in candidates:
            if type_count >= max_tasks_per_type:
                break
            cid = (
                cand.get("clean_forward_candidate_id")
                or cand.get("candidate_id")
                or cand.get("combined_target_id")
                or ""
            )
            ctx_rows = context_by.get(cid) or []
            outcome_row = match_outcome_for_candidate(cand, outcome_rows) if outcome_available_global else None
            outcome_ok = bool(outcome_row)

            prompt_meta = build_prompt_for_task(
                task_type,
                cand,
                context_rows=ctx_rows,
                outcome_row=outcome_row,
                outcome_available=outcome_ok,
            )
            evidence_refs = list(
                filter(
                    None,
                    [
                        cand.get("source_artifact"),
                        (inputs.get("ae18_context") or {}).get("context_csv"),
                        outcome_meta.get("path"),
                    ],
                )
            )
            record = _base_record(
                task_type=task_type,
                candidate=cand,
                prompt_meta=prompt_meta,
                evidence_refs=[str(x) for x in evidence_refs],
            )

            if prompt_meta.get("input_unavailable") and task_type == TASK_MISSED_WINNER_REVIEW:
                record["task_status"] = TASK_SKIPPED_INPUT
                record["provider"] = "none"
                record["provider_status"] = qwen_status.get("provider_status") or "LLM_PROVIDER_UNAVAILABLE"
                record["provider_model"] = ""
                record["missed_winner_status"] = MISSED_WINNER_UNAVAILABLE
                record["failure_reason"] = MISSED_WINNER_UNAVAILABLE
                record["output_summary"] = MISSED_WINNER_UNAVAILABLE
                record["completed_at"] = utc_now()
                record["mock_used"] = False
                record["counted_as_real_provider_success"] = False
                record["downstream_eligible"] = False
                record["downstream_quarantined"] = True
                record["accepted_for_downstream"] = False
                record = apply_lineage_to_record(
                    record,
                    prompt_meta.get("spine") or extract_identity_spine(cand),
                    symbol_only_result=prompt_meta.get("symbol_check"),
                )
                records.append(record)
                type_count += 1
                continue

            prefer_gemini = (
                task_type == TASK_SEMANTIC_CONFLICT_REVIEW
                and str(gemini_status.get("provider_status")) == PROVIDER_AVAILABLE
                and gemini_calls < gemini_selective_budget
            )
            provider_result = _run_provider_for_task(
                prompt=prompt_meta.get("prompt_text") or "",
                task_type=task_type,
                candidate=cand,
                prefer_gemini=prefer_gemini,
                allow_qwen=allow_qwen,
                allow_gemini=allow_gemini,
                force_qwen_unavailable=force_qwen_unavailable,
                force_gemini_unavailable=force_gemini_unavailable,
                use_mock_diagnostic=use_mock_diagnostic,
                qwen_status=qwen_status,
                gemini_status=gemini_status,
            )
            if provider_result.get("provider") == "gemini" and provider_result.get("counted_as_real_provider_success"):
                gemini_calls += 1

            record = _finalize_record(
                record,
                provider_result,
                spine=prompt_meta.get("spine") or extract_identity_spine(cand),
                symbol_check=prompt_meta.get("symbol_check"),
            )
            records.append(record)
            type_count += 1

    # AUDIT tasks — one machine-readable audit record per candidate (bounded)
    audit_count = 0
    for cand in candidates:
        if audit_count >= max_tasks_per_type:
            break
        prompt_meta = build_prompt_for_task(TASK_AUDIT, cand, context_rows=[])
        evidence_refs = list(filter(None, [cand.get("source_artifact")]))
        record = _base_record(
            task_type=TASK_AUDIT,
            candidate=cand,
            prompt_meta=prompt_meta,
            evidence_refs=[str(x) for x in evidence_refs],
        )
        # AUDIT family is primarily machine-built; still go through provider status channel
        if use_mock_diagnostic:
            provider_result = run_qwen_operational(
                "Emit AUDIT status only.",
                task_type=TASK_AUDIT,
                candidate=cand,
                use_mock_diagnostic=True,
            )
        elif str(qwen_status.get("provider_status")) == PROVIDER_AVAILABLE:
            # Lightweight real call for audit trail proof
            provider_result = run_qwen_operational(
                prompt_meta.get("prompt_text") or "Emit AUDIT research-only note.",
                task_type=TASK_AUDIT,
                candidate=cand,
                allow_qwen=allow_qwen,
                force_unavailable=force_qwen_unavailable,
                provider_status_cache=qwen_status,
            )
        else:
            # Explicit unavailable audit record (still counts as AUDIT family present)
            provider_result = {
                "provider": "qwen",
                "provider_model": qwen_status.get("provider_model") or "",
                "provider_status": qwen_status.get("provider_status"),
                "task_status": "LLM_TASK_SKIPPED_PROVIDER_UNAVAILABLE",
                "text": json.dumps(
                    {
                        "audit": "ae19_task_audit",
                        "provider_available": False,
                        "trade_authority_used": False,
                        "live_trading_approved": False,
                        "risk_override_used": False,
                        "wallet_accessed": False,
                        "mock_used": False,
                        "identity_invention": False,
                    },
                    sort_keys=True,
                ),
                "mock_used": False,
                "counted_as_real_provider_success": False,
                "downstream_eligible": False,
                "downstream_quarantined": True,
                "accepted_for_downstream": False,
                "failure_reason": "LLM_PROVIDER_UNAVAILABLE",
                "error": str(qwen_status.get("detail") or ""),
            }
            # For unavailable path, still produce machine audit JSON as output_summary content
            # but do not claim provider success.
        record = _finalize_record(
            record,
            provider_result,
            spine=prompt_meta.get("spine") or extract_identity_spine(cand),
            symbol_check=prompt_meta.get("symbol_check"),
        )
        # Ensure AUDIT always has structured audit payload in summary when skipped
        if not record.get("output_text") and record.get("task_status") != TASK_SUCCEEDED:
            record["output_summary"] = "AUDIT_RECORD_PROVIDER_UNAVAILABLE"
        records.append(record)
        audit_count += 1

    # If no candidates, still emit one AUDIT infrastructure record so family exists
    if not any(r.get("task_type") == TASK_AUDIT for r in records):
        empty_cand = {
            "clean_forward_candidate_id": "",
            "pair_address": "",
            "chain": "",
            "price_source_key": "",
        }
        prompt_meta = build_prompt_for_task(TASK_AUDIT, empty_cand)
        record = _base_record(
            task_type=TASK_AUDIT,
            candidate=empty_cand,
            prompt_meta=prompt_meta,
            evidence_refs=[],
        )
        record["task_status"] = TASK_SKIPPED_INPUT
        record["provider"] = "none"
        record["provider_status"] = qwen_status.get("provider_status") or "LLM_PROVIDER_UNAVAILABLE"
        record["failure_reason"] = "LLM_TASK_SKIPPED_INPUT_UNAVAILABLE"
        record["output_summary"] = "NO_CANDIDATES_FOR_AUDIT"
        record["completed_at"] = utc_now()
        record["downstream_quarantined"] = True
        record["identity_status"] = "IDENTITY_UNRESOLVED"
        records.append(record)

    # Ensure each required family has at least a placeholder if candidates existed but loop skipped
    present = {r.get("task_type") for r in records}
    for required in TASK_TYPES:
        if required in present:
            continue
        if required == TASK_AUDIT:
            continue
        # Create explicit skipped-input placeholder
        empty_cand = {"clean_forward_candidate_id": "", "chain": "", "pair_address": ""}
        prompt_meta = build_prompt_for_task(required, empty_cand)
        record = _base_record(
            task_type=required,
            candidate=empty_cand,
            prompt_meta=prompt_meta,
            evidence_refs=[],
        )
        if required == TASK_MISSED_WINNER_REVIEW:
            record["missed_winner_status"] = MISSED_WINNER_UNAVAILABLE
            record["failure_reason"] = MISSED_WINNER_UNAVAILABLE
            record["output_summary"] = MISSED_WINNER_UNAVAILABLE
        else:
            record["failure_reason"] = "LLM_TASK_SKIPPED_INPUT_UNAVAILABLE"
            record["output_summary"] = "NO_CANDIDATES"
        record["task_status"] = TASK_SKIPPED_INPUT
        record["provider"] = "none"
        record["provider_status"] = qwen_status.get("provider_status") or "LLM_PROVIDER_UNAVAILABLE"
        record["completed_at"] = utc_now()
        record["downstream_quarantined"] = True
        records.append(record)

    # --- Persist data artifacts ---
    json_records = []
    for r in records:
        jd = dict(r)
        # Keep lists as lists in JSONL
        json_records.append(jd)

    csv_rows = []
    for r in records:
        row = {}
        for k in TASK_RECORD_FIELDS:
            val = r.get(k)
            if isinstance(val, list):
                row[k] = "|".join(str(x) for x in val)
            else:
                row[k] = val
        csv_rows.append(row)

    write_csv(data_dir / "ae19_llm_tasks.csv", csv_rows, fieldnames=list(TASK_RECORD_FIELDS))
    write_jsonl(data_dir / "ae19_llm_tasks.jsonl", json_records)

    def _accepted_family_csv(task_type: str, path: Path) -> list[dict[str, Any]]:
        rows = []
        for r in records:
            if r.get("task_type") != task_type:
                continue
            # Accepted tables exclude rejected/quarantined except as audit (not here)
            if r.get("safety_failed") or (r.get("mock_used") and not r.get("accepted_for_downstream")):
                # Rejected/mock do not appear in accepted family CSVs
                continue
            if r.get("downstream_quarantined") and not r.get("downstream_eligible"):
                # Allow skipped/unavailable observational rows that are not safety-failed
                if r.get("safety_failed") or r.get("identity_invention_detected"):
                    continue
            rows.append(
                {
                    "ae19_task_id": r.get("ae19_task_id"),
                    "candidate_id": r.get("candidate_id") or r.get("clean_forward_candidate_id"),
                    "clean_forward_candidate_id": r.get("clean_forward_candidate_id"),
                    "price_source_key": r.get("price_source_key"),
                    "provider_pair_url_exact": r.get("provider_pair_url_exact"),
                    "canonical_market_identity": r.get("canonical_market_identity"),
                    "pair_address": r.get("pair_address"),
                    "chain": r.get("chain"),
                    "symbol_pair": r.get("symbol_pair"),
                    "provider": r.get("provider"),
                    "provider_status": r.get("provider_status"),
                    "task_status": r.get("task_status"),
                    "output_summary": r.get("output_summary"),
                    "missed_winner_status": r.get("missed_winner_status"),
                    "safety_status": r.get("safety_status"),
                    "downstream_eligible": r.get("downstream_eligible"),
                    "downstream_quarantined": r.get("downstream_quarantined"),
                    "mock_used": r.get("mock_used"),
                    "counted_as_real_provider_success": r.get("counted_as_real_provider_success"),
                    "prompt_text_hash": r.get("prompt_text_hash"),
                    "response_text_hash": r.get("response_text_hash"),
                    "created_at": r.get("created_at"),
                    "completed_at": r.get("completed_at"),
                }
            )
        write_csv(path, rows)
        return rows

    memos = _accepted_family_csv(TASK_CANDIDATE_MEMO, data_dir / "ae19_candidate_memos.csv")
    risks = _accepted_family_csv(TASK_RISK_EXPLANATION, data_dir / "ae19_risk_explanations.csv")
    missed = _accepted_family_csv(TASK_MISSED_WINNER_REVIEW, data_dir / "ae19_missed_winner_reviews.csv")
    semantic = _accepted_family_csv(
        TASK_SEMANTIC_CONFLICT_REVIEW, data_dir / "ae19_semantic_conflict_reviews.csv"
    )
    contexts = _accepted_family_csv(TASK_CONTEXT_SUMMARY, data_dir / "ae19_context_summaries.csv")

    audit_records = [
        {
            "ae19_task_id": r.get("ae19_task_id"),
            "task_type": r.get("task_type"),
            "provider": r.get("provider"),
            "provider_model": r.get("provider_model"),
            "provider_status": r.get("provider_status"),
            "task_status": r.get("task_status"),
            "prompt_template_version": r.get("prompt_template_version"),
            "prompt_text_hash": r.get("prompt_text_hash"),
            "response_text_hash": r.get("response_text_hash"),
            "created_at": r.get("created_at"),
            "completed_at": r.get("completed_at"),
            "lineage": {
                "price_source_key": r.get("price_source_key"),
                "provider_pair_url_exact": r.get("provider_pair_url_exact"),
                "canonical_market_identity": r.get("canonical_market_identity"),
                "clean_forward_candidate_id": r.get("clean_forward_candidate_id"),
                "decision_input_id": r.get("decision_input_id"),
                "pair_address": r.get("pair_address"),
                "chain": r.get("chain"),
                "identity_status": r.get("identity_status"),
                "resolver_status": r.get("resolver_status"),
            },
            "missingness": {
                "failure_reason": r.get("failure_reason"),
                "missed_winner_status": r.get("missed_winner_status"),
            },
            "safety_status": r.get("safety_status"),
            "authority": {
                "trade_authority_used": False,
                "live_trading_approved": False,
                "risk_override_used": False,
                "wallet_accessed": False,
            },
            "mock_used": r.get("mock_used"),
            "counted_as_real_provider_success": r.get("counted_as_real_provider_success"),
            "downstream_eligible": r.get("downstream_eligible"),
            "downstream_quarantined": r.get("downstream_quarantined"),
            "forbidden_language_hits": r.get("forbidden_language_hits"),
            "raw_response_preserved_present": bool(r.get("raw_response_preserved")),
        }
        for r in records
    ]
    write_jsonl(data_dir / "ae19_llm_audit_records.jsonl", audit_records)

    # --- Audits ---
    identity_audit = audit_mod.build_no_identity_invention_audit(records)
    authority_audit = audit_mod.build_authority_safety_audit(records)
    mock_audit = audit_mod.build_mock_provider_audit(records)
    quarantine_audit = audit_mod.build_downstream_quarantine_audit(records)
    failure_audit = audit_mod.build_failure_modes_audit(
        records, qwen_status=qwen_status, gemini_status=gemini_status
    )
    provider_audit = audit_mod.build_provider_runtime_audit(
        qwen_status=qwen_status, gemini_status=gemini_status, records=records
    )
    lineage_audit = audit_mod.build_input_lineage_audit(
        discovery={
            "status": inputs.get("status"),
            "candidate_count": inputs.get("candidate_count"),
            "discovery": inputs.get("discovery"),
            "ae18_context": inputs.get("ae18_context"),
            "outcome_evidence": outcome_meta,
        },
        output_root=out,
        records=records,
    )

    write_json(audits_dir / "ae19_input_lineage_audit.json", lineage_audit)
    write_json(audits_dir / "ae19_provider_runtime_audit.json", provider_audit)
    write_csv(
        audits_dir / "ae19_candidate_memo_audit.csv",
        audit_mod.task_family_audit_rows(records, TASK_CANDIDATE_MEMO),
    )
    write_csv(
        audits_dir / "ae19_risk_explanation_audit.csv",
        audit_mod.task_family_audit_rows(records, TASK_RISK_EXPLANATION),
    )
    write_csv(
        audits_dir / "ae19_missed_winner_review_audit.csv",
        audit_mod.task_family_audit_rows(records, TASK_MISSED_WINNER_REVIEW),
    )
    write_csv(
        audits_dir / "ae19_semantic_conflict_review_audit.csv",
        audit_mod.task_family_audit_rows(records, TASK_SEMANTIC_CONFLICT_REVIEW),
    )
    write_csv(
        audits_dir / "ae19_context_summary_audit.csv",
        audit_mod.task_family_audit_rows(records, TASK_CONTEXT_SUMMARY),
    )
    write_json(audits_dir / "ae19_no_identity_invention_audit.json", identity_audit)
    write_json(audits_dir / "ae19_authority_safety_audit.json", authority_audit)
    write_json(audits_dir / "ae19_failure_modes_audit.json", failure_audit)
    write_json(audits_dir / "ae19_mock_provider_audit.json", mock_audit)
    write_json(audits_dir / "ae19_downstream_quarantine_audit.json", quarantine_audit)

    counts = audit_mod.provider_runtime_counts(records)
    task_counts = audit_mod.task_type_counts(records)
    classification = audit_mod.decide_classification(
        records=records,
        identity_audit=identity_audit,
        authority_audit=authority_audit,
        mock_audit=mock_audit,
        quarantine_audit=quarantine_audit,
        qwen_status=qwen_status,
        gemini_status=gemini_status,
    )

    artifact_paths = {
        "output_root": str(out.resolve()),
        "ae19_manifest": str((reports_dir / "ae19_manifest.json").resolve()),
        "ae19_summary_for_upload": str((reports_dir / "ae19_summary_for_upload.txt").resolve()),
        "ae19_decision_gate": str((reports_dir / "ae19_decision_gate.json").resolve()),
        "ae19_llm_tasks_csv": str((data_dir / "ae19_llm_tasks.csv").resolve()),
        "ae19_llm_tasks_jsonl": str((data_dir / "ae19_llm_tasks.jsonl").resolve()),
        "ae19_candidate_memos_csv": str((data_dir / "ae19_candidate_memos.csv").resolve()),
        "ae19_risk_explanations_csv": str((data_dir / "ae19_risk_explanations.csv").resolve()),
        "ae19_missed_winner_reviews_csv": str((data_dir / "ae19_missed_winner_reviews.csv").resolve()),
        "ae19_semantic_conflict_reviews_csv": str((data_dir / "ae19_semantic_conflict_reviews.csv").resolve()),
        "ae19_context_summaries_csv": str((data_dir / "ae19_context_summaries.csv").resolve()),
        "ae19_llm_audit_records_jsonl": str((data_dir / "ae19_llm_audit_records.jsonl").resolve()),
        "ae19_input_lineage_audit": str((audits_dir / "ae19_input_lineage_audit.json").resolve()),
        "ae19_provider_runtime_audit": str((audits_dir / "ae19_provider_runtime_audit.json").resolve()),
        "ae19_no_identity_invention_audit": str((audits_dir / "ae19_no_identity_invention_audit.json").resolve()),
        "ae19_authority_safety_audit": str((audits_dir / "ae19_authority_safety_audit.json").resolve()),
        "ae19_failure_modes_audit": str((audits_dir / "ae19_failure_modes_audit.json").resolve()),
        "ae19_mock_provider_audit": str((audits_dir / "ae19_mock_provider_audit.json").resolve()),
        "ae19_downstream_quarantine_audit": str((audits_dir / "ae19_downstream_quarantine_audit.json").resolve()),
    }

    # AE20 gate: unblocked only for non-blocked pass variants (informational)
    ae20_blocked = classification.startswith("AE19_BLOCKED_") or classification == audit_mod.CLASSIFICATION_PARTIAL
    if classification in {
        audit_mod.CLASSIFICATION_PASS,
        audit_mod.CLASSIFICATION_PASS_LIMITATIONS,
    }:
        ae20_status = "UNBLOCKED_FOR_HANDOFF"
    else:
        ae20_status = "BLOCKED"

    decision_gate = {
        "phase": PHASE,
        "classification": classification,
        "engine_version": ENGINE_VERSION,
        "output_root": str(out.resolve()),
        "artifact_paths": artifact_paths,
        "provider_counts": counts,
        "task_counts": task_counts,
        "identity_audit_pass": identity_audit.get("pass"),
        "authority_audit_pass": authority_audit.get("pass"),
        "mock_audit_pass": mock_audit.get("pass"),
        "quarantine_audit_pass": quarantine_audit.get("pass"),
        "safety_boundary": SAFETY_BOUNDARY,
        "ae20_status": ae20_status,
        "ae20_blocked": ae20_blocked if classification.startswith("AE19_BLOCKED_") else ae20_status == "BLOCKED",
        "checklist": {
            "AE19-01_provider_runtime": True,
            "AE19-02_candidate_memo": task_counts.get("candidate_memo_count", 0) > 0,
            "AE19-03_risk_explanation": task_counts.get("risk_explanation_count", 0) > 0,
            "AE19-04_missed_winner": task_counts.get("missed_winner_review_count", 0) > 0,
            "AE19-05_semantic_conflict": task_counts.get("semantic_conflict_review_count", 0) > 0,
            "AE19-06_context_summary": task_counts.get("context_summary_count", 0) > 0,
            "AE19-07_audit_trail": len(audit_records) > 0,
            "AE19-08_authority_boundary": authority_audit.get("pass"),
            "AE19-09_provider_truthfulness": mock_audit.get("pass"),
            "AE19-10_identity_lineage": identity_audit.get("pass"),
        },
    }
    write_json(reports_dir / "ae19_decision_gate.json", decision_gate)

    manifest = {
        "phase": PHASE,
        "classification": classification,
        "engine_version": ENGINE_VERSION,
        "created_at": utc_now(),
        "output_root": str(out.resolve()),
        "artifact_paths": artifact_paths,
        "qwen_status": qwen_status,
        "gemini_status": gemini_status,
        "provider_counts": counts,
        "task_counts": task_counts,
        "candidate_count": len(candidates),
        "task_record_count": len(records),
        "audit_record_count": len(audit_records),
        "accepted_memo_count": len(memos),
        "accepted_risk_count": len(risks),
        "accepted_missed_count": len(missed),
        "accepted_semantic_count": len(semantic),
        "accepted_context_count": len(contexts),
        "max_candidates": max_candidates,
        "max_tasks_per_type": max_tasks_per_type,
        "use_mock_diagnostic": use_mock_diagnostic,
        "safety_boundary": SAFETY_BOUNDARY,
        "ae20_status": ae20_status,
    }
    write_json(reports_dir / "ae19_manifest.json", manifest)

    summary = audit_mod.build_summary_text(
        classification=classification,
        output_root=out,
        counts=counts,
        task_counts=task_counts,
        identity_audit=identity_audit,
        authority_audit=authority_audit,
        quarantine_audit=quarantine_audit,
        artifact_paths=artifact_paths,
    )
    write_text(reports_dir / "ae19_summary_for_upload.txt", summary)

    return {
        "phase": PHASE,
        "classification": classification,
        "output_root": str(out.resolve()),
        "artifact_paths": artifact_paths,
        "provider_counts": counts,
        "task_counts": task_counts,
        "task_records": records,
        "audit_record_count": len(audit_records),
        "identity_audit": identity_audit,
        "authority_audit": authority_audit,
        "mock_audit": mock_audit,
        "quarantine_audit": quarantine_audit,
        "lineage_audit": lineage_audit,
        "qwen_status": qwen_status,
        "gemini_status": gemini_status,
        "decision_gate": decision_gate,
        "ae20_status": ae20_status,
        "manifest": manifest,
    }
