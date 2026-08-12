"""AE18 constants, schemas, and safety contracts."""

from __future__ import annotations

PHASE = "AE18"

# Final classifications
CLASSIFICATION_PASS = "AE18_CONTEXT_INTELLIGENCE_PASS_WITH_PROVENANCE"
CLASSIFICATION_PASS_LIMITATIONS = "AE18_CONTEXT_INTELLIGENCE_PASS_WITH_MISSINGNESS_LIMITATIONS"
CLASSIFICATION_PASS_REAL_HELIUS = "AE18_CONTEXT_INTELLIGENCE_PASS_WITH_REAL_HELIUS_CONTEXT"
CLASSIFICATION_PASS_REAL_FETCH_LIMITATIONS = "AE18_CONTEXT_INTELLIGENCE_PASS_WITH_REAL_FETCH_LIMITATIONS"
CLASSIFICATION_INFRA_ONLY = "AE18_CONTEXT_INFRASTRUCTURE_AND_MISSINGNESS_CONTRACT_PASS"
CLASSIFICATION_PARTIAL_INFRA = "AE18_PARTIAL_PASS_CONTEXT_INFRASTRUCTURE_ONLY"
CLASSIFICATION_BLOCKED_NO_SOURCES = "AE18_BLOCKED_NO_CONTEXT_SOURCES_AVAILABLE"
CLASSIFICATION_BLOCKED_RESOLVER = "AE18_BLOCKED_RESOLVER_UNSAFE"
CLASSIFICATION_BLOCKED_SYMBOL_ONLY = "AE18_BLOCKED_SYMBOL_ONLY_JOIN_RISK"
CLASSIFICATION_BLOCKED_WHALE_CONFLATION = "AE18_BLOCKED_WHALE_SCORE_CONFLATION"
CLASSIFICATION_BLOCKED_HELIUS_WRITE = "AE18_BLOCKED_HELIUS_SOLANA_WRITE_RISK"
CLASSIFICATION_BLOCKED_MISSINGNESS = "AE18_BLOCKED_MISSINGNESS_CONTRACT_FAILURE"
CLASSIFICATION_BLOCKED_AUTHORITY = "AE18_BLOCKED_AUTHORITY_ESCALATION"
CLASSIFICATION_BLOCKED_HELIUS_NOT_CONFIGURED = "AE18_BLOCKED_HELIUS_NOT_CONFIGURED"
CLASSIFICATION_BLOCKED_SOLANA_PREFLIGHT = "AE18_BLOCKED_SOLANA_PREFLIGHT_SAFETY"
CLASSIFICATION_BLOCKED_RPC_FAILURE = "AE18_BLOCKED_RPC_FAILURE"

CONTEXT_ENGINE_VERSION = "ae18_context_intelligence_v1_real_helius"

# Resolver statuses
RESOLVER_LINKED = "RESOLVER_LINKED"
RESOLVER_UNRESOLVED = "IDENTITY_UNRESOLVED"
RESOLVER_SYMBOL_REJECTED = "SYMBOL_ONLY_JOIN_REJECTED"
RESOLVER_AMBIGUOUS = "RESOLVER_AMBIGUOUS"

# Allowed identity join paths
ALLOWED_JOIN_PATHS: tuple[str, ...] = (
    "price_source_key",
    "chain_pair_address",
    "chain_token_address",
    "clean_forward_candidate_id",
    "target_lineage_id",
)

# Whale evidence
WHALE_SIGNAL_POOL_FLOW_PROXY = "POOL_FLOW_PROXY"
WHALE_SIGNAL_WALLET_LEVEL = "WALLET_LEVEL_EVIDENCE"

# Missingness reasons (subset; extensible)
MISSINGNESS_REASONS: frozenset[str] = frozenset(
    {
        "SOURCE_UNAVAILABLE_PENDING_FETCH",
        "SOURCE_DISABLED_BY_CONFIG",
        "SOURCE_NOT_CONFIGURED",
        "SOURCE_EMPTY_RESPONSE",
        "SOURCE_RATE_LIMITED",
        "SOURCE_TIMEOUT",
        "SOURCE_PARSE_FAILED",
        "IDENTITY_UNRESOLVED",
        "SYMBOL_ONLY_JOIN_REJECTED",
        "WALLET_LEVEL_DATA_NOT_AVAILABLE",
        "PROVIDER_NOT_CALLED_IN_THIS_MODE",
        "RPC_RATE_LIMITED",
        "RPC_FETCH_FAILED",
        "RPC_BUDGET_EXCEEDED",
        "TRANSACTIONS_NOT_FOUND",
        "ACCOUNT_NOT_FOUND",
    }
)

ALLOWED_IDENTITY_BASES: frozenset[str] = frozenset(
    {
        "PRICE_SOURCE_KEY",
        "CHAIN_PAIR_ADDRESS",
        "CHAIN_TOKEN_ADDRESS",
        "CLEAN_FORWARD_CANDIDATE_ID",
        "AE16_EVIDENCE_ID",
        "AE17_EVIDENCE_ID",
        "price_source_key",
        "chain_pair_address",
        "chain_token_address",
        "clean_forward_candidate_id",
        "target_lineage_id",
    }
)

FORBIDDEN_IDENTITY_BASES: frozenset[str] = frozenset(
    {
        "SYMBOL",
        "TOKEN_SYMBOL",
        "TOKEN_NAME",
        "DISPLAY_NAME",
        "SYMBOL_FALLBACK",
        "LLM_INFERRED_SYMBOL",
        "symbol_only",
    }
)

# Safety boundary — always enforced
SAFETY_BOUNDARY: dict[str, bool] = {
    "wallet_access": False,
    "private_key_access": False,
    "signer_available": False,
    "transaction_builder_available": False,
    "transaction_signing_available": False,
    "transaction_submission_available": False,
    "live_trading_enabled": False,
    "trade_authority": False,
    "no_trade_authority": True,
    "risk_gate_override": False,
    "profitability_claim": False,
    "live_readiness_claim": False,
}

# Forbidden write-capability patterns for read-only audit
FORBIDDEN_HELIUS_SOLANA_PATTERNS: tuple[str, ...] = (
    "sendTransaction",
    "sendRawTransaction",
    "Keypair",
    "private_key",
    "sign_transaction",
    "wallet_client",
    "jupiter_swap",
    "execute_swap",
)

# Known AE17 / AE16 discovery patterns
KNOWN_AE17_ROOTS: tuple[str, ...] = (
    "data/audits/ae17_meta_stacking_layer_20260725T080549Z",
    "data/audits/ae17_real_meta_evidence_run_20260726_202057",
)

AE17_FEATURE_PATTERNS: tuple[str, ...] = (
    "**/ae17_meta_feature_rows.csv",
    "**/ae17_real_meta_feature_matrix.csv",
)

AE16_CONSENSUS_PATTERNS: tuple[str, ...] = (
    "**/ae16f_tiered_consensus_rows.csv",
    "**/rf_xgb_tab16_consensus_preview.csv",
    "**/ae16_tiered_consensus_rows.csv",
)

CURATED_TARGETS_PATH = "data/SeedTargets/clean_forward_curated_ready_targets_active.csv"

# Context record schema fields
CONTEXT_RECORD_FIELDS: tuple[str, ...] = (
    "context_record_id",
    "clean_forward_candidate_id",
    "clean_forward_decision_input_id",
    "price_source_key",
    "chain",
    "pair_address",
    "base_token_address",
    "quote_token_address",
    "combined_target_id",
    "context_family",
    "context_status",
    "source_name",
    "source_type",
    "attempted",
    "available",
    "missingness_reason",
    "provenance_status",
    "resolver_status",
    "resolver_join_path",
    "resolver_confidence",
    "whale_signal_type",
    "observed_at",
    "fetched_at",
    "attempted_at",
    "no_trade_authority",
    "wallet_access",
    "private_key_access",
    "signer_available",
    "transaction_signing_available",
    "transaction_submission_available",
    "live_trading_enabled",
    "trade_authority",
    "context_engine_version",
    "lineage_tier",
)

RESOLVER_LINK_FIELDS: tuple[str, ...] = (
    "resolver_link_id",
    "context_record_id",
    "clean_forward_candidate_id",
    "price_source_key",
    "chain",
    "pair_address",
    "token_address",
    "text_item_id",
    "join_path",
    "resolver_status",
    "resolver_confidence",
    "ambiguous",
    "symbol_only_rejected",
    "provenance_status",
    "observed_at",
)

MISSINGNESS_RECORD_FIELDS: tuple[str, ...] = (
    "missingness_record_id",
    "clean_forward_candidate_id",
    "price_source_key",
    "pair_address",
    "chain",
    "source_name",
    "source_type",
    "attempted",
    "available",
    "context_status",
    "missingness_reason",
    "provenance_status",
    "attempted_at",
    "no_trade_authority",
)
