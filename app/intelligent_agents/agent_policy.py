"""AE12.7 agent policy — operating modes, budgets, and hard authority bans."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from app.intelligent_agents.types import OperatingMode, resolve_operating_mode


@dataclass
class AgentDemoPolicy:
    """Explicit enablement gates. Defaults are safe (no external, no wallet)."""

    mode: OperatingMode = OperatingMode.AGENT_DEMO_DISABLED
    enable_qwen: bool = False
    enable_gemini: bool = False
    enable_helius: bool = False
    enable_rss_external: bool = False
    no_external_api: bool = True
    no_real_wallet: bool = True
    provider: str = "none"
    gemini_budget: int = 5
    helius_budget: int = 10
    qwen_budget: int = 20
    limit: int = 50
    allow_append_daily: bool = True
    # Test hooks
    inject_gemini_response: str | None = None
    force_qwen_unavailable: bool = False
    force_helius_unavailable: bool = False

    # Runtime counters (mutated during run)
    gemini_calls_used: int = 0
    helius_calls_used: int = 0
    qwen_calls_used: int = 0
    external_api_calls: list[dict[str, Any]] = field(default_factory=list)

    @property
    def qwen_allowed(self) -> bool:
        if self.no_external_api and self.mode == OperatingMode.AGENT_DEMO_DISABLED:
            return False
        if self.mode == OperatingMode.AGENT_DEMO_DISABLED:
            return False
        if self.mode in (OperatingMode.QWEN_LOCAL_DEMO, OperatingMode.FULL_AGENT_OBSERVABILITY_DEMO):
            return bool(self.enable_qwen or self.provider in {"ollama", "qwen", "local"})
        return False

    @property
    def gemini_allowed(self) -> bool:
        if self.no_external_api:
            return False
        if not self.enable_gemini:
            return False
        if self.mode not in (
            OperatingMode.GEMINI_SELECTIVE_AUDIT_DEMO,
            OperatingMode.FULL_AGENT_OBSERVABILITY_DEMO,
        ):
            return False
        return self.gemini_calls_used < self.gemini_budget

    @property
    def helius_allowed(self) -> bool:
        if self.no_external_api:
            return False
        if not self.enable_helius:
            return False
        if self.mode not in (
            OperatingMode.HELIUS_READONLY_ENRICHMENT_DEMO,
            OperatingMode.FULL_AGENT_OBSERVABILITY_DEMO,
        ):
            return False
        return self.helius_calls_used < self.helius_budget

    def record_external_call(self, *, provider: str, purpose: str, success: bool) -> None:
        self.external_api_calls.append(
            {
                "provider": provider,
                "purpose": purpose,
                "success": success,
                "trade_authority_used": False,
            }
        )

    def authority_ban_contract(self) -> dict[str, Any]:
        return {
            "llm_trade_authority": False,
            "gemini_trade_authority": False,
            "qwen_trade_authority": False,
            "helius_trade_authority": False,
            "rss_trade_authority": False,
            "semantic_trade_authority": False,
            "agent_layer_trade_authority": False,
            "wallet_allowed": False,
            "private_key_access_allowed": False,
            "real_transaction_allowed": False,
            "no_real_wallet": self.no_real_wallet,
            "no_external_api": self.no_external_api,
            "paper_demo_trading_allowed": True,
            "decision_effects_allowed": [
                "explanation_only",
                "audit_only",
                "context_only",
                "soft_warning_only",
                "soft_veto_recommendation_only",
                "no_effect",
            ],
        }


def build_policy_from_args(
    *,
    mode: str,
    enable_gemini: bool = False,
    enable_helius: bool = False,
    enable_qwen: bool = False,
    no_external_api: bool = True,
    no_real_wallet: bool = True,
    provider: str = "none",
    limit: int = 50,
    gemini_budget: int = 5,
    helius_budget: int = 10,
    qwen_budget: int = 20,
    inject_gemini_response: str | None = None,
    force_qwen_unavailable: bool = False,
    force_helius_unavailable: bool = False,
) -> AgentDemoPolicy:
    op = resolve_operating_mode(mode)
    # Convenience: qwen-local / full-demo with provider implies qwen enable
    if op in (OperatingMode.QWEN_LOCAL_DEMO, OperatingMode.FULL_AGENT_OBSERVABILITY_DEMO):
        if provider in {"ollama", "qwen", "local"} or enable_qwen:
            enable_qwen = True
    # Force external off when flag set
    if no_external_api:
        enable_gemini = False
        enable_helius = False
    # Env cannot override no_real_wallet
    if os.getenv("AE12_7_FORCE_WALLET", "").lower() in {"1", "true"}:
        # Explicitly ignore — wallet remains banned
        pass
    return AgentDemoPolicy(
        mode=op,
        enable_qwen=enable_qwen,
        enable_gemini=enable_gemini,
        enable_helius=enable_helius,
        no_external_api=no_external_api,
        no_real_wallet=True if no_real_wallet else True,  # always True in AE12.7
        provider=provider,
        gemini_budget=gemini_budget,
        helius_budget=helius_budget,
        qwen_budget=qwen_budget,
        limit=limit,
        inject_gemini_response=inject_gemini_response,
        force_qwen_unavailable=force_qwen_unavailable,
        force_helius_unavailable=force_helius_unavailable,
    )
