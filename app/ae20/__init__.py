"""AE20 — Integrated Clean Forward Validation (shadow/exploration/reporting)."""

from __future__ import annotations

PHASE = "AE20_INTEGRATED_CLEAN_FORWARD_VALIDATION"
ENGINE_VERSION = "ae20_integrated_clean_forward_validation_v1"
OUTPUT_ROOT_PREFIX = "ae20_integrated_clean_forward_validation_"

# Smoke LLM budget
SMOKE_DEFAULT_MAX_LLM_CALLS_PER_CYCLE = 2
SMOKE_HARD_MAX_LLM_CALLS_PER_CYCLE = 3
SMOKE_DEFAULT_LLM_TIMEOUT_SECONDS = 45

ALLOWED_SMOKE_CLASSIFICATIONS = frozenset(
    {
        "AE20_SMOKE_INTEGRATION_PASS_READY_FOR_24H",
        "AE20_SMOKE_PASS_WITH_RUNTIME_LIMITATIONS",
        "AE20_SMOKE_BLOCKED_CLEAN_FORWARD_INPUT_FAILURE",
        "AE20_SMOKE_BLOCKED_INTEGRATION_LAYER_FAILURE",
        "AE20_SMOKE_BLOCKED_LINEAGE_FAILURE",
        "AE20_SMOKE_BLOCKED_LEGACY_CONTAMINATION",
        "AE20_SMOKE_BLOCKED_AUTHORITY_ESCALATION",
    }
)

REQUIRED_CANONICAL_IDENTITY_FIELDS = (
    "provider_pair_url_exact",
    "canonical_market_identity",
    "normalized_provider_pair_url_key",
    "chain",
    "pair_address",
)

SKIP_REASONS = frozenset(
    {
        "AE16_REJECT",
        "AE16_MODEL_EVIDENCE_UNAVAILABLE",
        "AE17_META_LOW_CONFIDENCE",
        "AE18_CONTEXT_RISK",
        "AE18_CONTEXT_UNAVAILABLE",
        "AE19_LLM_AUDIT_WARNING",
        "GATEKEEPER_BLOCK",
        "RISKGUARD_BLOCK",
        "STALE_PRICE",
        "MISSING_PRICE",
        "MARKET_ACTIVITY_BLOCK",
        "COOLDOWN",
        "DEAD_PAIR",
        "MAX_EXPOSURE",
        "MAX_OPEN_POSITIONS",
        "EXPLORATION_ONLY",
        "CLEAN_FORWARD_IDENTITY_INCOMPLETE",
        "EVIDENCE_UNAVAILABLE",
    }
)

__all__ = [
    "PHASE",
    "ENGINE_VERSION",
    "OUTPUT_ROOT_PREFIX",
    "ALLOWED_SMOKE_CLASSIFICATIONS",
    "REQUIRED_CANONICAL_IDENTITY_FIELDS",
    "SKIP_REASONS",
]
