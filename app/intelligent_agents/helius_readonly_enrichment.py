"""AE12.7 Helius / Solana read-only enrichment — no wallet, no signing, no submit."""

from __future__ import annotations

import os
from typing import Any

from app.intelligent_agents.agent_policy import AgentDemoPolicy
from app.intelligent_agents.types import (
    AgentStatus,
    AgentType,
    DecisionEffect,
    SourceMode,
    make_agent_record,
)


def _helius_key_present() -> bool:
    key = (os.getenv("HELIUS_API_KEY") or "").strip()
    return bool(key) and not key.startswith("$")


def run_helius_readonly_enrichment(
    candidate: dict[str, Any],
    *,
    policy: AgentDemoPolicy,
) -> dict[str, Any]:
    """
    Attach read-only enrichment status.

    Never accesses wallet/private keys. Never signs or submits.
    Unconfigured → NOT_CONFIGURED.
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

    enrichment_skeleton = {
        "fee_payer": None,
        "signer_wallets": [],
        "token_owner_wallet": None,
        "pool_owner_hint": None,
        "vault_hint": None,
        "transaction_flow_summary": None,
        "large_wallet_activity_hints": [],
        "buy_sell_flow_pressure": None,
        "enrichment_status": "NOT_RUN",
        "readonly": True,
        "wallet_connected": False,
        "signing_attempted": False,
        "transaction_submitted": False,
    }

    if policy.force_helius_unavailable or not policy.enable_helius or policy.no_external_api:
        status = AgentStatus.NOT_CONFIGURED if (policy.force_helius_unavailable or not _helius_key_present()) else AgentStatus.SKIPPED
        if not policy.enable_helius or policy.no_external_api:
            status = AgentStatus.NOT_CONFIGURED if not policy.enable_helius else AgentStatus.SKIPPED
        enrichment_skeleton["enrichment_status"] = status.value
        return make_agent_record(
            agent_type=AgentType.HELIUS_READONLY_ENRICHMENT,
            source_mode=SourceMode.DISABLED if (not policy.enable_helius or policy.no_external_api) else SourceMode.EXTERNAL_API,
            agent_status=status,
            agent_summary="Helius read-only enrichment not configured or disabled.",
            decision_effect=DecisionEffect.NO_EFFECT,
            warnings=["helius_not_configured_or_disabled"],
            missing_context_flags=["helius_unavailable"],
            input_artifact_refs=refs,
            **link_ids,
            extra={
                "helius_called": False,
                "external_api_used": False,
                "enrichment": enrichment_skeleton,
                "wallet_accessed": False,
                "private_key_accessed": False,
            },
        )

    if not policy.helius_allowed:
        enrichment_skeleton["enrichment_status"] = "SKIPPED"
        return make_agent_record(
            agent_type=AgentType.HELIUS_READONLY_ENRICHMENT,
            source_mode=SourceMode.EXTERNAL_API,
            agent_status=AgentStatus.SKIPPED,
            agent_summary="Helius budget exhausted or mode does not allow enrichment.",
            decision_effect=DecisionEffect.NO_EFFECT,
            warnings=["helius_budget_or_mode"],
            input_artifact_refs=refs,
            **link_ids,
            extra={"helius_called": False, "external_api_used": False, "enrichment": enrichment_skeleton},
        )

    if not _helius_key_present():
        enrichment_skeleton["enrichment_status"] = "NOT_CONFIGURED"
        return make_agent_record(
            agent_type=AgentType.HELIUS_READONLY_ENRICHMENT,
            source_mode=SourceMode.EXTERNAL_API,
            agent_status=AgentStatus.NOT_CONFIGURED,
            agent_summary="HELIUS_API_KEY not configured; enrichment skipped.",
            decision_effect=DecisionEffect.NO_EFFECT,
            warnings=["helius_api_key_missing"],
            missing_context_flags=["helius_unavailable"],
            input_artifact_refs=refs,
            **link_ids,
            extra={
                "helius_called": False,
                "external_api_used": False,
                "enrichment": enrichment_skeleton,
                "wallet_accessed": False,
                "private_key_accessed": False,
            },
        )

    # Configured + enabled: produce read-only placeholder enrichment without broad ingestion.
    # No live HTTP by default in AE12.7 smoke (avoids accidental credit spend); record intent.
    policy.helius_calls_used += 1
    policy.record_external_call(provider="helius", purpose="readonly_enrichment_stub", success=True)
    enrichment = {
        **enrichment_skeleton,
        "fee_payer": None,
        "signer_wallets": [],
        "token_owner_wallet": None,
        "pool_owner_hint": "UNRESOLVED_READONLY",
        "vault_hint": "UNRESOLVED_READONLY",
        "transaction_flow_summary": (
            f"Read-only enrichment stub for pair={candidate.get('pair_address')}; "
            "no wallet connect; no sign; no submit."
        ),
        "large_wallet_activity_hints": [],
        "buy_sell_flow_pressure": "UNKNOWN_UNRESOLVED",
        "enrichment_status": "GENERATED_READONLY_STUB",
    }
    return make_agent_record(
        agent_type=AgentType.HELIUS_READONLY_ENRICHMENT,
        source_mode=SourceMode.EXTERNAL_API,
        agent_status=AgentStatus.GENERATED,
        agent_summary=enrichment["transaction_flow_summary"],
        decision_effect=DecisionEffect.AUDIT_ONLY,
        warnings=["helius_readonly_stub_no_live_http"],
        input_artifact_refs=refs,
        **link_ids,
        extra={
            "helius_called": True,
            "external_api_used": True,
            "live_http": False,
            "enrichment": enrichment,
            "wallet_accessed": False,
            "private_key_accessed": False,
            "real_transaction_attempted": False,
            "readonly": True,
        },
    )
