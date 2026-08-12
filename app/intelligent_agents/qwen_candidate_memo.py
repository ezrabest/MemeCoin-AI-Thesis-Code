"""AE12.7 Qwen / local LLM candidate memo generation (no trade authority)."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

from app.intelligent_agents.agent_policy import AgentDemoPolicy
from app.intelligent_agents.safety import reject_authority_language
from app.intelligent_agents.types import (
    AgentStatus,
    AgentType,
    DecisionEffect,
    SourceMode,
    make_agent_record,
)


def _ollama_base() -> str:
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def _ollama_model() -> str:
    return os.getenv("OLLAMA_MODEL", "qwen3:8b")


def ollama_available(*, timeout_s: float = 1.5) -> bool:
    """Best-effort local probe; never raises."""
    try:
        base = _ollama_base()
        # Support both /v1 OpenAI-compat and native Ollama host
        host = base.replace("/v1", "")
        req = urllib.request.Request(f"{host}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return int(getattr(resp, "status", 200) or 200) < 400
    except Exception:
        return False


def _build_local_memo(candidate: dict[str, Any]) -> str:
    """Deterministic offline memo when Ollama is unavailable or not called."""
    symbol = candidate.get("symbol") or "UNKNOWN"
    pair = candidate.get("pair_address") or "MISSING_PAIR"
    strict = candidate.get("strict_shadow_decision") or "UNKNOWN"
    exploration = candidate.get("exploration_decision") or "UNKNOWN"
    reason = candidate.get("reason_not_traded") or candidate.get("reason_for_no_trade") or "unspecified"
    stale = candidate.get("price_freshness_status") or "UNKNOWN"
    return (
        f"Candidate memo (local template): {symbol} @ {pair}. "
        f"Strict={strict}; Exploration={exploration}. "
        f"Why PAPER/WATCH/BLOCK context: reason={reason}; price_freshness={stale}. "
        f"This is explanation-only — not BUY/SELL authority. Soft-veto only if lineage/stale risks dominate."
    )


def _call_ollama_chat(prompt: str, *, timeout_s: float = 20.0) -> tuple[str | None, str | None]:
    """Optional native Ollama chat. Returns (text, error)."""
    host = _ollama_base().replace("/v1", "")
    body = json.dumps(
        {
            "model": _ollama_model(),
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an audit/explanation agent for a paper-trading research demo. "
                        "Never recommend BUY, SELL, EXECUTE, or live trading. "
                        "Produce a short candidate memo covering context, missingness, stale-price risk, "
                        "strict-vs-exploration explanation, and optional soft-veto recommendation only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        f"{host}/api/chat",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        msg = (data.get("message") or {}).get("content")
        if isinstance(msg, str) and msg.strip():
            return msg.strip(), None
        return None, "empty_ollama_response"
    except urllib.error.URLError as e:
        return None, f"ollama_unreachable:{e}"
    except Exception as e:
        return None, f"ollama_error:{type(e).__name__}"


def generate_qwen_candidate_memo(
    candidate: dict[str, Any],
    *,
    policy: AgentDemoPolicy,
    attempt_live_call: bool = False,
) -> dict[str, Any]:
    """
    Generate a Qwen/local candidate memo record.

    Never fails the run: unavailable → NOT_CONFIGURED / SKIPPED.
    Never grants trade authority.
    """
    link_ids = {
        "candidate_id": candidate.get("candidate_id"),
        "source_decision_id": candidate.get("source_decision_id") or candidate.get("decision_id"),
        "source_context_record_id": candidate.get("source_context_record_id"),
        "source_llm_audit_record_id": candidate.get("source_llm_audit_record_id"),
        "paper_order_id": candidate.get("paper_order_id"),
        "position_id": candidate.get("position_id"),
        "pair_address": candidate.get("pair_address"),
        "symbol": candidate.get("symbol"),
        "chain": candidate.get("chain") or "solana",
    }
    refs = list(filter(None, [candidate.get("_source_ref")]))

    if not policy.qwen_allowed or policy.force_qwen_unavailable:
        status = AgentStatus.SKIPPED if not policy.qwen_allowed else AgentStatus.NOT_CONFIGURED
        return make_agent_record(
            agent_type=AgentType.QWEN_LOCAL_MEMO,
            source_mode=SourceMode.DISABLED if not policy.qwen_allowed else SourceMode.LOCAL,
            agent_status=status,
            agent_summary="Qwen/local memo not generated (disabled or forced unavailable).",
            decision_effect=DecisionEffect.NO_EFFECT,
            warnings=["qwen_not_invoked"],
            missing_context_flags=["qwen_provider_unavailable"] if policy.force_qwen_unavailable else [],
            input_artifact_refs=refs,
            **link_ids,
            extra={
                "provider": policy.provider,
                "external_call_made": False,
                "ollama_probed": False,
            },
        )

    if policy.qwen_calls_used >= policy.qwen_budget:
        return make_agent_record(
            agent_type=AgentType.QWEN_LOCAL_MEMO,
            source_mode=SourceMode.LOCAL,
            agent_status=AgentStatus.SKIPPED,
            agent_summary="Qwen budget exhausted; memo skipped.",
            decision_effect=DecisionEffect.NO_EFFECT,
            warnings=["qwen_budget_exhausted"],
            input_artifact_refs=refs,
            **link_ids,
            extra={"external_call_made": False},
        )

    memo_text: str | None = None
    call_error: str | None = None
    external_call = False
    source_mode = SourceMode.LOCAL

    if attempt_live_call and policy.provider in {"ollama", "qwen", "local"}:
        available = ollama_available()
        if not available:
            return make_agent_record(
                agent_type=AgentType.QWEN_LOCAL_MEMO,
                source_mode=SourceMode.LOCAL,
                agent_status=AgentStatus.NOT_CONFIGURED,
                agent_summary="Ollama/Qwen not reachable; memo not generated.",
                decision_effect=DecisionEffect.NO_EFFECT,
                warnings=["ollama_not_reachable"],
                missing_context_flags=["qwen_provider_unavailable"],
                input_artifact_refs=refs,
                **link_ids,
                extra={"external_call_made": False, "ollama_probed": True},
            )
        prompt = (
            f"Candidate: {json.dumps({k: candidate.get(k) for k in ('symbol','pair_address','strict_shadow_decision','exploration_decision','reason_not_traded','price_freshness_status','max_return')}, default=str)}"
        )
        memo_text, call_error = _call_ollama_chat(prompt)
        external_call = True
        policy.qwen_calls_used += 1
        # Local Ollama is not an external cloud API; do not mark external_api_used
        if call_error or not memo_text:
            return make_agent_record(
                agent_type=AgentType.QWEN_LOCAL_MEMO,
                source_mode=SourceMode.LOCAL,
                agent_status=AgentStatus.ERROR,
                agent_summary=f"Ollama call failed: {call_error or 'empty'}",
                decision_effect=DecisionEffect.NO_EFFECT,
                warnings=["qwen_call_failed"],
                input_artifact_refs=refs,
                **link_ids,
                extra={"external_call_made": False, "ollama_error": call_error},
            )
    else:
        # Demo/template memo — operational proof without requiring Ollama
        memo_text = _build_local_memo(candidate)
        policy.qwen_calls_used += 1

    rejection = reject_authority_language(memo_text)
    if rejection["forbidden_trade_language_found"]:
        return make_agent_record(
            agent_type=AgentType.QWEN_LOCAL_MEMO,
            source_mode=source_mode,
            agent_status=AgentStatus.REJECTED_SAFETY,
            agent_summary="Qwen memo rejected: forbidden trade-authority language.",
            decision_effect=DecisionEffect.NO_EFFECT,
            warnings=["forbidden_trade_language_rejected"],
            soft_veto_flags=[],
            input_artifact_refs=refs,
            **link_ids,
            extra={
                "external_call_made": False,
                "gemini_called": False,
                **{k: rejection[k] for k in ("forbidden_trade_language_found", "rejection_status", "safety_status", "output_used_after_rejection")},
                "memo_redacted": True,
            },
        )

    soft_veto: list[str] = []
    if str(candidate.get("price_freshness_status") or "").upper().startswith("STALE"):
        soft_veto.append("soft_veto_stale_price")
    if "WEAK_LINEAGE" in str(candidate.get("audit_blockers") or ""):
        soft_veto.append("soft_veto_weak_lineage")

    effect = DecisionEffect.SOFT_VETO_RECOMMENDATION_ONLY if soft_veto else DecisionEffect.EXPLANATION_ONLY
    return make_agent_record(
        agent_type=AgentType.QWEN_LOCAL_MEMO,
        source_mode=source_mode,
        agent_status=AgentStatus.GENERATED,
        agent_summary=memo_text[:2000],
        decision_effect=effect,
        warnings=["stale_price_warning"] if soft_veto and "soft_veto_stale_price" in soft_veto else [],
        soft_veto_flags=soft_veto,
        confidence=0.55 if external_call else 0.4,
        input_artifact_refs=refs,
        **link_ids,
        extra={
            "external_call_made": False,
            "ollama_live_call": external_call,
            "provider": policy.provider or "local_template",
            "memo_full": memo_text,
            "why_paper_watch_block": {
                "strict": candidate.get("strict_shadow_decision"),
                "exploration": candidate.get("exploration_decision"),
                "reason": candidate.get("reason_not_traded") or candidate.get("reason_for_no_trade"),
            },
        },
    )
