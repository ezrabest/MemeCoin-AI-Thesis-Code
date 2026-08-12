"""AE19 audit builders and classification decision."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from app.llm_operational.lineage import build_no_identity_invention_audit
from app.llm_operational.providers import assert_no_false_provider_success
from app.llm_operational.schema import (
    CLASSIFICATION_BLOCKED_AUTHORITY,
    CLASSIFICATION_BLOCKED_FALSE_SUCCESS,
    CLASSIFICATION_BLOCKED_IDENTITY,
    CLASSIFICATION_BLOCKED_MISSING_TASK,
    CLASSIFICATION_BLOCKED_PROVIDER,
    CLASSIFICATION_BLOCKED_QUARANTINE,
    CLASSIFICATION_PARTIAL,
    CLASSIFICATION_PASS,
    CLASSIFICATION_PASS_LIMITATIONS,
    ENGINE_VERSION,
    MOCK_PROVIDER_DIAGNOSTIC,
    PHASE,
    PROVIDER_AVAILABLE,
    PROVIDER_DISABLED,
    PROVIDER_ERROR,
    PROVIDER_UNAVAILABLE,
    GEMINI_UNAVAILABLE_OR_DISABLED,
    SAFETY_BOUNDARY,
    TASK_AUDIT,
    TASK_CANDIDATE_MEMO,
    TASK_CONTEXT_SUMMARY,
    TASK_MISSED_WINNER_REVIEW,
    TASK_RISK_EXPLANATION,
    TASK_SEMANTIC_CONFLICT_REVIEW,
    TASK_SUCCEEDED,
    TASK_TYPES,
)
from app.llm_operational.safety import authority_boundary_snapshot


REQUIRED_TASK_FAMILIES = (
    TASK_CANDIDATE_MEMO,
    TASK_RISK_EXPLANATION,
    TASK_MISSED_WINNER_REVIEW,
    TASK_SEMANTIC_CONFLICT_REVIEW,
    TASK_CONTEXT_SUMMARY,
    TASK_AUDIT,
)


def _full(path: Path) -> str:
    return str(path.resolve())


def provider_runtime_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(r.get("provider_status") or "") for r in records)
    return {
        "provider_available_count": statuses.get(PROVIDER_AVAILABLE, 0),
        "provider_unavailable_count": statuses.get(PROVIDER_UNAVAILABLE, 0)
        + statuses.get(GEMINI_UNAVAILABLE_OR_DISABLED, 0),
        "provider_disabled_count": statuses.get(PROVIDER_DISABLED, 0),
        "provider_error_count": statuses.get(PROVIDER_ERROR, 0),
        "mock_diagnostic_count": statuses.get(MOCK_PROVIDER_DIAGNOSTIC, 0),
        "real_provider_success_count": sum(
            1 for r in records if r.get("counted_as_real_provider_success")
        ),
        "mock_counted_as_real_success_count": sum(
            1 for r in records if r.get("mock_used") and r.get("counted_as_real_provider_success")
        ),
        "status_histogram": dict(statuses),
    }


def task_type_counts(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_type = Counter(str(r.get("task_type") or "") for r in records)
    by_status = Counter(str(r.get("task_status") or "") for r in records)
    by_type_status: dict[str, dict[str, int]] = {}
    for r in records:
        tt = str(r.get("task_type") or "")
        ts = str(r.get("task_status") or "")
        by_type_status.setdefault(tt, {})
        by_type_status[tt][ts] = by_type_status[tt].get(ts, 0) + 1
    return {
        "by_type": dict(by_type),
        "by_status": dict(by_status),
        "by_type_status": by_type_status,
        "candidate_memo_count": by_type.get(TASK_CANDIDATE_MEMO, 0),
        "risk_explanation_count": by_type.get(TASK_RISK_EXPLANATION, 0),
        "missed_winner_review_count": by_type.get(TASK_MISSED_WINNER_REVIEW, 0),
        "semantic_conflict_review_count": by_type.get(TASK_SEMANTIC_CONFLICT_REVIEW, 0),
        "context_summary_count": by_type.get(TASK_CONTEXT_SUMMARY, 0),
        "audit_task_count": by_type.get(TASK_AUDIT, 0),
    }


def build_authority_safety_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    forbidden_hits = 0
    rejected = 0
    failures: list[dict[str, Any]] = []
    for r in records:
        hits = r.get("forbidden_language_hits") or []
        if isinstance(hits, str):
            hits = [h for h in hits.split("|") if h]
        if hits:
            forbidden_hits += len(hits)
            failures.append(
                {
                    "ae19_task_id": r.get("ae19_task_id"),
                    "task_type": r.get("task_type"),
                    "hits": hits,
                    "safety_status": r.get("safety_status"),
                    "downstream_quarantined": r.get("downstream_quarantined"),
                    "downstream_eligible": r.get("downstream_eligible"),
                }
            )
        if r.get("safety_failed") or str(r.get("safety_status") or "").startswith("FAIL_"):
            rejected += 1

    authority_escalation = any(
        bool(r.get("trade_authority_used"))
        or bool(r.get("live_trading_approved"))
        or bool(r.get("risk_override_used"))
        or bool(r.get("wallet_accessed"))
        for r in records
    )
    # Also escalation if forbidden language accepted downstream
    accepted_forbidden = [
        r
        for r in records
        if (r.get("forbidden_language_hits") or r.get("safety_failed"))
        and r.get("accepted_for_downstream")
        and r.get("downstream_eligible")
    ]
    if accepted_forbidden:
        authority_escalation = True

    return {
        "audit": "ae19_authority_safety_audit",
        "authority_boundary": authority_boundary_snapshot(),
        "safety_boundary": SAFETY_BOUNDARY,
        "forbidden_language_hit_count": forbidden_hits,
        "rejected_quarantined_output_count": rejected,
        "failure_details": failures,
        "authority_escalation": authority_escalation,
        "pass": not authority_escalation,
        "block_code": CLASSIFICATION_BLOCKED_AUTHORITY if authority_escalation else None,
    }


def build_mock_provider_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    mock_recs = [r for r in records if r.get("mock_used")]
    false_success = assert_no_false_provider_success(records)
    return {
        "audit": "ae19_mock_provider_audit",
        "mock_diagnostic_count": len(mock_recs),
        "mock_counted_as_real_success_count": sum(
            1 for r in mock_recs if r.get("counted_as_real_provider_success")
        ),
        "all_mocks_quarantined": all(r.get("downstream_quarantined") for r in mock_recs) if mock_recs else True,
        "all_mocks_not_downstream_eligible": all(not r.get("downstream_eligible") for r in mock_recs)
        if mock_recs
        else True,
        "false_provider_success_check": false_success,
        "pass": false_success["pass"]
        and all(not r.get("downstream_eligible") for r in mock_recs)
        and all(r.get("downstream_quarantined") for r in mock_recs if mock_recs),
        "block_code": false_success.get("block_code"),
    }


def build_downstream_quarantine_audit(records: list[dict[str, Any]]) -> dict[str, Any]:
    quarantined = [r for r in records if r.get("downstream_quarantined")]
    eligible = [r for r in records if r.get("downstream_eligible") and r.get("accepted_for_downstream")]
    bad = [
        r
        for r in records
        if (r.get("mock_used") or r.get("safety_failed") or r.get("identity_invention_detected"))
        and (r.get("downstream_eligible") or r.get("accepted_for_downstream"))
    ]
    return {
        "audit": "ae19_downstream_quarantine_audit",
        "downstream_quarantined_count": len(quarantined),
        "downstream_eligible_count": len(eligible),
        "quarantine_failures": [
            {
                "ae19_task_id": r.get("ae19_task_id"),
                "task_type": r.get("task_type"),
                "mock_used": r.get("mock_used"),
                "safety_failed": r.get("safety_failed"),
                "identity_invention_detected": r.get("identity_invention_detected"),
                "downstream_eligible": r.get("downstream_eligible"),
                "accepted_for_downstream": r.get("accepted_for_downstream"),
            }
            for r in bad
        ],
        "pass": len(bad) == 0,
        "block_code": CLASSIFICATION_BLOCKED_QUARANTINE if bad else None,
    }


def build_failure_modes_audit(
    records: list[dict[str, Any]],
    *,
    qwen_status: dict[str, Any],
    gemini_status: dict[str, Any],
) -> dict[str, Any]:
    failures = []
    for r in records:
        if r.get("failure_reason") or str(r.get("task_status") or "").startswith("LLM_TASK_FAILED"):
            failures.append(
                {
                    "ae19_task_id": r.get("ae19_task_id"),
                    "task_type": r.get("task_type"),
                    "provider": r.get("provider"),
                    "provider_status": r.get("provider_status"),
                    "task_status": r.get("task_status"),
                    "failure_reason": r.get("failure_reason"),
                }
            )
    return {
        "audit": "ae19_failure_modes_audit",
        "qwen_status": qwen_status,
        "gemini_status": gemini_status,
        "failure_count": len(failures),
        "failures": failures,
        "provider_unavailable_recorded": (
            str(qwen_status.get("provider_status"))
            in {PROVIDER_UNAVAILABLE, PROVIDER_DISABLED, PROVIDER_ERROR}
            or str(gemini_status.get("provider_status"))
            in {GEMINI_UNAVAILABLE_OR_DISABLED, PROVIDER_UNAVAILABLE, PROVIDER_DISABLED, PROVIDER_ERROR}
        ),
    }


def build_provider_runtime_audit(
    *,
    qwen_status: dict[str, Any],
    gemini_status: dict[str, Any],
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = provider_runtime_counts(records)
    return {
        "audit": "ae19_provider_runtime_audit",
        "qwen": qwen_status,
        "gemini": gemini_status,
        "counts": counts,
        "engine_version": ENGINE_VERSION,
        "phase": PHASE,
    }


def build_input_lineage_audit(
    *,
    discovery: dict[str, Any],
    output_root: Path,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "audit": "ae19_input_lineage_audit",
        "discovery": discovery,
        "output_root": _full(output_root),
        "task_record_count": len(records),
        "records_with_price_source_key": sum(1 for r in records if r.get("price_source_key")),
        "records_with_pair_address": sum(1 for r in records if r.get("pair_address")),
        "records_with_candidate_id": sum(
            1 for r in records if r.get("candidate_id") or r.get("clean_forward_candidate_id")
        ),
        "symbol_pair_display_only": True,
        "join_by_symbol_alone_forbidden": True,
    }


def task_family_audit_rows(records: list[dict[str, Any]], task_type: str) -> list[dict[str, Any]]:
    rows = []
    for r in records:
        if r.get("task_type") != task_type:
            continue
        rows.append(
            {
                "ae19_task_id": r.get("ae19_task_id"),
                "candidate_id": r.get("candidate_id") or r.get("clean_forward_candidate_id"),
                "provider": r.get("provider"),
                "provider_status": r.get("provider_status"),
                "task_status": r.get("task_status"),
                "mock_used": r.get("mock_used"),
                "counted_as_real_provider_success": r.get("counted_as_real_provider_success"),
                "downstream_eligible": r.get("downstream_eligible"),
                "downstream_quarantined": r.get("downstream_quarantined"),
                "safety_status": r.get("safety_status"),
                "identity_status": r.get("identity_status"),
                "failure_reason": r.get("failure_reason"),
                "missed_winner_status": r.get("missed_winner_status"),
                "prompt_text_hash": r.get("prompt_text_hash"),
                "response_text_hash": r.get("response_text_hash"),
                "accepted_for_downstream": r.get("accepted_for_downstream"),
            }
        )
    return rows


def decide_classification(
    *,
    records: list[dict[str, Any]],
    identity_audit: dict[str, Any],
    authority_audit: dict[str, Any],
    mock_audit: dict[str, Any],
    quarantine_audit: dict[str, Any],
    qwen_status: dict[str, Any],
    gemini_status: dict[str, Any],
) -> str:
    present_types = {str(r.get("task_type") or "") for r in records}
    missing = [t for t in REQUIRED_TASK_FAMILIES if t not in present_types]
    if missing:
        return CLASSIFICATION_BLOCKED_MISSING_TASK

    if not identity_audit.get("pass"):
        return CLASSIFICATION_BLOCKED_IDENTITY
    if not authority_audit.get("pass"):
        return CLASSIFICATION_BLOCKED_AUTHORITY
    if not mock_audit.get("pass"):
        return CLASSIFICATION_BLOCKED_FALSE_SUCCESS
    if not quarantine_audit.get("pass"):
        return CLASSIFICATION_BLOCKED_QUARANTINE

    false_success = assert_no_false_provider_success(records)
    if not false_success["pass"]:
        return CLASSIFICATION_BLOCKED_FALSE_SUCCESS

    qwen_avail = str(qwen_status.get("provider_status")) == PROVIDER_AVAILABLE
    gemini_avail = str(gemini_status.get("provider_status")) == PROVIDER_AVAILABLE
    real_success = any(r.get("counted_as_real_provider_success") for r in records)
    explicit_unavailable = any(
        str(r.get("provider_status"))
        in {
            PROVIDER_UNAVAILABLE,
            PROVIDER_DISABLED,
            PROVIDER_ERROR,
            GEMINI_UNAVAILABLE_OR_DISABLED,
            MOCK_PROVIDER_DIAGNOSTIC,
        }
        for r in records
    )

    # Operational runtime exercised (real success or explicit unavailable handling)
    runtime_ok = real_success or explicit_unavailable
    if not runtime_ok:
        return CLASSIFICATION_PARTIAL

    if not qwen_avail and not gemini_avail and not real_success:
        # Both unavailable but explicit handling present → limitations pass
        if explicit_unavailable and all(t in present_types for t in REQUIRED_TASK_FAMILIES):
            return CLASSIFICATION_PASS_LIMITATIONS
        return CLASSIFICATION_BLOCKED_PROVIDER

    if real_success and all(t in present_types for t in REQUIRED_TASK_FAMILIES):
        if qwen_avail and gemini_avail:
            return CLASSIFICATION_PASS
        return CLASSIFICATION_PASS_LIMITATIONS

    if explicit_unavailable and all(t in present_types for t in REQUIRED_TASK_FAMILIES):
        return CLASSIFICATION_PASS_LIMITATIONS

    return CLASSIFICATION_PARTIAL


def build_summary_text(
    *,
    classification: str,
    output_root: Path,
    counts: dict[str, Any],
    task_counts: dict[str, Any],
    identity_audit: dict[str, Any],
    authority_audit: dict[str, Any],
    quarantine_audit: dict[str, Any],
    artifact_paths: dict[str, str],
) -> str:
    lines = [
        f"AE19 LLM Operational Layer — {classification}",
        f"output_root={_full(output_root)}",
        f"engine_version={ENGINE_VERSION}",
        "",
        "Provider counts:",
        f"  available={counts.get('provider_available_count')}",
        f"  unavailable={counts.get('provider_unavailable_count')}",
        f"  disabled={counts.get('provider_disabled_count')}",
        f"  error={counts.get('provider_error_count')}",
        f"  mock_diagnostic={counts.get('mock_diagnostic_count')}",
        f"  real_provider_success={counts.get('real_provider_success_count')}",
        f"  mock_counted_as_real_success={counts.get('mock_counted_as_real_success_count')}",
        "",
        "Task counts:",
        f"  candidate_memo={task_counts.get('candidate_memo_count')}",
        f"  risk_explanation={task_counts.get('risk_explanation_count')}",
        f"  missed_winner_review={task_counts.get('missed_winner_review_count')}",
        f"  semantic_conflict_review={task_counts.get('semantic_conflict_review_count')}",
        f"  context_summary={task_counts.get('context_summary_count')}",
        f"  audit_tasks={task_counts.get('audit_task_count')}",
        "",
        "Safety / identity:",
        f"  identity_pass={identity_audit.get('pass')}",
        f"  symbol_only_join_rejected={identity_audit.get('symbol_only_join_rejected_count')}",
        f"  invented_identity_rejected={identity_audit.get('llm_invented_identity_rejected_count')}",
        f"  authority_pass={authority_audit.get('pass')}",
        f"  forbidden_language_hits={authority_audit.get('forbidden_language_hit_count')}",
        f"  rejected_quarantined={authority_audit.get('rejected_quarantined_output_count')}",
        f"  downstream_eligible={quarantine_audit.get('downstream_eligible_count')}",
        f"  downstream_quarantined={quarantine_audit.get('downstream_quarantined_count')}",
        "",
        "Artifact paths (exact):",
    ]
    for key, path in sorted(artifact_paths.items()):
        lines.append(f"  {key}={path}")
    lines.extend(
        [
            "",
            "Proves: operational LLM task families, provider/unavailable handling, authority/identity safety audits.",
            "Does not prove: profitability, live readiness, trade authority, wallet connectivity.",
            "AE20: BLOCKED until AE19 classification is a non-blocked pass variant and product owners unblock.",
        ]
    )
    return "\n".join(lines) + "\n"
