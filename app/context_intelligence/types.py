"""AE8 context intelligence types and constants."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

AE8_PHASE = "AE8_CONTEXT_INTELLIGENCE_COLLECTION_INTEGRATION_LAYER"
AE8_CONTEXT_SCHEMA_VERSION = "AE8_CONTEXT_V1"

AE8_LINEAGE_WARNING = (
    "Lineage warning: context source was linked by provider/pair/timestamp "
    "best-effort matching, not by explicit source ID."
)
AE8_LINEAGE_FALLBACK_REASON = (
    "Best-effort linkage via provider/pair/timestamp; no explicit source ID chain."
)

FORBIDDEN_CONTEXT_FEATURE_PATTERNS: tuple[str, ...] = (
    "target",
    "label",
    "future",
    "realized",
    "outcome",
    "profit",
    "profitable",
    "net_return",
    "future_return",
    "realized_return",
    "exit_result",
    "exit_outcome",
    "simulation_result",
    "test",
    "train",
    "validation",
)

DEFAULT_FRESHNESS_THRESHOLDS_MINUTES: dict[str, float] = {
    "market_snapshot": 15.0,
    "liquidity_activity": 15.0,
    "raw_provider_payloads": 15.0,
    "onchain": 15.0,
    "whale": 15.0,
    "reputation": 24.0 * 60.0,
    "rss": 6.0 * 60.0,
    "rss_article_window": 24.0 * 60.0,
}


class FreshnessMode(StrEnum):
    LIVE_OR_CURRENT_RUNTIME = "LIVE_OR_CURRENT_RUNTIME"
    HISTORICAL_REPLAY_OR_AUDIT = "HISTORICAL_REPLAY_OR_AUDIT"


class FreshnessStatus(StrEnum):
    FRESH = "FRESH"
    STALE = "STALE"
    MISSING_TIMESTAMP = "MISSING_TIMESTAMP"
    NOT_AVAILABLE = "NOT_AVAILABLE"
    REPLAY_AS_OF_FRESH = "REPLAY_AS_OF_FRESH"
    DISABLED_BY_DEFAULT = "DISABLED_BY_DEFAULT"
    INVALID_FUTURE_TIMESTAMP = "INVALID_FUTURE_TIMESTAMP"


class SourceStatus(StrEnum):
    SOURCE_OK = "SOURCE_OK"
    SOURCE_EMPTY = "SOURCE_EMPTY"
    SOURCE_CONFIG_MISSING = "SOURCE_CONFIG_MISSING"
    SOURCE_NOT_AVAILABLE = "SOURCE_NOT_AVAILABLE"
    SOURCE_STALE = "SOURCE_STALE"
    SOURCE_ERROR = "SOURCE_ERROR"
    SOURCE_DISABLED_BY_DEFAULT = "SOURCE_DISABLED_BY_DEFAULT"


class LineageValidationStatus(StrEnum):
    PASS_EXPLICIT_ID_LINKAGE = "PASS_EXPLICIT_ID_LINKAGE"
    PASS_WEAK_BEST_EFFORT_WITH_WARNING = "PASS_WEAK_BEST_EFFORT_WITH_WARNING"
    BLOCKED_MISSING_STABLE_IDENTITY = "BLOCKED_MISSING_STABLE_IDENTITY"
    BLOCKED_UNRESOLVED_SOURCE = "BLOCKED_UNRESOLVED_SOURCE"
    BLOCKED_AMBIGUOUS_MATCH = "BLOCKED_AMBIGUOUS_MATCH"
    BLOCKED_FUTURE_TIMESTAMP = "BLOCKED_FUTURE_TIMESTAMP"


class MemorySafetyStatus(StrEnum):
    PASS_BOUNDED_QUERIES = "PASS_BOUNDED_QUERIES"
    PASS_CHUNKED_OR_STREAMING = "PASS_CHUNKED_OR_STREAMING"
    BLOCKED_FULL_TABLE_LOAD_RISK = "BLOCKED_FULL_TABLE_LOAD_RISK"
    BLOCKED_UNBOUNDED_QUERY = "BLOCKED_UNBOUNDED_QUERY"


class Ae8FinalStatus(StrEnum):
    READY_FOR_FORWARD_COLLECTION = "AE8_CONTEXT_LAYER_READY_FOR_FORWARD_COLLECTION"
    PARTIAL_LOCAL_ONLY = "AE8_CONTEXT_LAYER_PARTIAL_LOCAL_ONLY"
    BLOCKED_NO_LOCAL_CONTEXT = "AE8_CONTEXT_LAYER_BLOCKED_NO_LOCAL_CONTEXT"
    BLOCKED_SCHEMA_LEAKAGE = "AE8_CONTEXT_LAYER_BLOCKED_SCHEMA_LEAKAGE"
    BLOCKED_EXTERNAL_CONFIG = "AE8_CONTEXT_LAYER_BLOCKED_EXTERNAL_CONFIG"
    BLOCKED_STALE_CONTEXT = "AE8_CONTEXT_LAYER_BLOCKED_STALE_CONTEXT"
    BLOCKED_LINEAGE_VALIDATION = "AE8_CONTEXT_LAYER_BLOCKED_LINEAGE_VALIDATION"
    BLOCKED_MEMORY_SAFETY = "AE8_CONTEXT_LAYER_BLOCKED_MEMORY_SAFETY"
    BLOCKED_WITH_EXACT_REASONS = "AE8_CONTEXT_LAYER_BLOCKED_WITH_EXACT_REASONS"


RSS_FEATURE_NAMES: tuple[str, ...] = (
    "rss_article_count_1h",
    "rss_article_count_6h",
    "rss_article_count_24h",
    "rss_positive_count_24h",
    "rss_negative_count_24h",
    "rss_neutral_count_24h",
    "rss_sentiment_mean_24h",
    "rss_sentiment_min_24h",
    "rss_sentiment_max_24h",
    "rss_keyword_hit_count_24h",
    "rss_source_count_24h",
    "rss_freshness_minutes",
    "rss_missingness_flag",
)

ONCHAIN_FEATURE_NAMES: tuple[str, ...] = (
    "onchain_txn_count_1h",
    "onchain_txn_count_24h",
    "onchain_unique_wallets_24h",
    "onchain_new_wallet_ratio_24h",
    "onchain_large_transfer_count_24h",
    "onchain_large_transfer_usd_24h",
    "onchain_wallet_concentration_top1",
    "onchain_wallet_concentration_top5",
    "onchain_contract_age_minutes",
    "onchain_lp_lock_signal",
    "onchain_authority_risk_flag",
    "onchain_freshness_minutes",
    "onchain_missingness_flag",
)

WHALE_FEATURE_NAMES: tuple[str, ...] = (
    "whale_score_asof",
    "whale_score_missingness",
    "whale_activity_count_1h",
    "whale_activity_count_24h",
    "whale_buy_pressure_24h",
    "whale_sell_pressure_24h",
    "whale_net_flow_proxy_24h",
    "whale_unique_wallets_24h",
    "whale_top_wallet_concentration",
    "whale_context_freshness_minutes",
)

REPUTATION_FEATURE_NAMES: tuple[str, ...] = (
    "reputation_known_token_flag",
    "reputation_new_token_flag",
    "reputation_blacklist_flag",
    "reputation_contract_verified_flag",
    "reputation_mint_authority_flag",
    "reputation_freeze_authority_flag",
    "reputation_honeypot_risk_flag",
    "reputation_source_count",
    "reputation_freshness_minutes",
    "reputation_missingness_flag",
)

LIQUIDITY_ACTIVITY_FEATURE_NAMES: tuple[str, ...] = (
    "liquidity_usd",
    "liquidity_change_1h",
    "liquidity_change_6h",
    "liquidity_anomaly_score",
    "volume_h24",
    "volume_change_1h",
    "volume_to_liquidity_ratio",
    "txns_h24_total",
    "buy_sell_ratio_h24",
    "activity_spike_score",
    "activity_decay_score",
    "price_change_m5",
    "price_change_h1",
    "price_change_h6",
    "price_change_h24",
    "liquidity_activity_freshness_minutes",
    "liquidity_activity_missingness_flag",
)

FEATURE_FAMILIES: dict[str, tuple[str, ...]] = {
    "rss": RSS_FEATURE_NAMES,
    "onchain": ONCHAIN_FEATURE_NAMES,
    "whale": WHALE_FEATURE_NAMES,
    "reputation": REPUTATION_FEATURE_NAMES,
    "liquidity_activity": LIQUIDITY_ACTIVITY_FEATURE_NAMES,
}

ALL_CONTEXT_FEATURE_NAMES: tuple[str, ...] = (
    *RSS_FEATURE_NAMES,
    *ONCHAIN_FEATURE_NAMES,
    *WHALE_FEATURE_NAMES,
    *REPUTATION_FEATURE_NAMES,
    *LIQUIDITY_ACTIVITY_FEATURE_NAMES,
)


def is_forbidden_context_feature(name: str) -> bool:
    lower = name.lower()
    if lower in {"test", "train", "validation"}:
        return True
    return any(pat in lower for pat in FORBIDDEN_CONTEXT_FEATURE_PATTERNS)


@dataclass
class FreshnessBlock:
    source_timestamp: str | None = None
    freshness_reference_timestamp: str | None = None
    freshness_minutes: float | None = None
    freshness_threshold_minutes: float | None = None
    freshness_status: str = FreshnessStatus.NOT_AVAILABLE.value
    stale_reason: str | None = None
    missingness_flag: bool = False
    missingness_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_timestamp": self.source_timestamp,
            "freshness_reference_timestamp": self.freshness_reference_timestamp,
            "freshness_minutes": self.freshness_minutes,
            "freshness_threshold_minutes": self.freshness_threshold_minutes,
            "freshness_status": self.freshness_status,
            "stale_reason": self.stale_reason,
            "missingness_flag": self.missingness_flag,
            "missingness_reason": self.missingness_reason,
        }


@dataclass
class ContextLineageBlock:
    lineage_mode: str
    lineage_strength: str
    lineage_confidence_score: float
    exact_id_match: bool
    lineage_validation_status: str
    source_tables: list[str] = field(default_factory=list)
    source_files: list[str] = field(default_factory=list)
    source_payload_ids: list[int | str] = field(default_factory=list)
    source_signal_ids: list[int | str] = field(default_factory=list)
    source_snapshot_ids: list[int | str] = field(default_factory=list)
    resolution_methods: list[str] = field(default_factory=list)
    lineage_warning: str | None = None
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lineage_mode": self.lineage_mode,
            "lineage_strength": self.lineage_strength,
            "lineage_confidence_score": self.lineage_confidence_score,
            "exact_id_match": self.exact_id_match,
            "source_tables": self.source_tables,
            "source_files": self.source_files,
            "source_payload_ids": self.source_payload_ids,
            "source_signal_ids": self.source_signal_ids,
            "source_snapshot_ids": self.source_snapshot_ids,
            "resolution_methods": self.resolution_methods,
            "lineage_warning": self.lineage_warning,
            "fallback_reason": self.fallback_reason,
            "lineage_validation_status": self.lineage_validation_status,
        }


@dataclass
class ContextFeatureRecord:
    context_record_id: str
    candidate_id: str
    pair_address: str | None
    symbol: str | None
    chain: str | None
    as_of_timestamp: str
    context_schema_id: str
    context_schema_version: str
    freshness_mode: str
    run_started_at_utc: str
    rss_context: dict[str, Any]
    onchain_context: dict[str, Any]
    whale_context: dict[str, Any]
    reputation_context: dict[str, Any]
    liquidity_activity_context: dict[str, Any]
    context_missingness: dict[str, Any]
    context_freshness: dict[str, Any]
    lineage: dict[str, Any]
    source_statuses: dict[str, str]
    source_warnings: list[str]
    no_trade_authority: Literal[True] = True
    llm_decision_authority: Literal[False] = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_record_id": self.context_record_id,
            "candidate_id": self.candidate_id,
            "pair_address": self.pair_address,
            "symbol": self.symbol,
            "chain": self.chain,
            "as_of_timestamp": self.as_of_timestamp,
            "context_schema_id": self.context_schema_id,
            "context_schema_version": self.context_schema_version,
            "freshness_mode": self.freshness_mode,
            "run_started_at_utc": self.run_started_at_utc,
            "rss_context": self.rss_context,
            "onchain_context": self.onchain_context,
            "whale_context": self.whale_context,
            "reputation_context": self.reputation_context,
            "liquidity_activity_context": self.liquidity_activity_context,
            "context_missingness": self.context_missingness,
            "context_freshness": self.context_freshness,
            "lineage": self.lineage,
            "source_statuses": self.source_statuses,
            "source_warnings": self.source_warnings,
            "no_trade_authority": self.no_trade_authority,
            "llm_decision_authority": self.llm_decision_authority,
            "phase": AE8_PHASE,
        }
