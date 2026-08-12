"""AE18 dataclass models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class AE18CandidateTarget:
    """Clean Forward / Selected candidate used as context target."""

    clean_forward_candidate_id: str
    clean_forward_decision_input_id: str = ""
    price_source_key: str = ""
    provider: str = "dexscreener"
    chain: str = ""
    pair_address: str = ""
    base_token_address: str = ""
    quote_token_address: str = ""
    combined_target_id: str = ""
    provider_pair_url: str = ""
    provider_payload_hash: str = ""
    token_symbol: str = ""
    token_name: str = ""
    observed_at: str = ""
    fetched_at: str = ""
    ingested_at: str = ""
    whale_score: str | float | None = None
    source_artifact: str = ""
    lineage_status: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AE18ContextRecord:
    context_record_id: str
    clean_forward_candidate_id: str
    context_family: str
    context_status: str
    source_name: str
    source_type: str
    attempted: bool
    available: bool
    clean_forward_decision_input_id: str = ""
    price_source_key: str = ""
    chain: str = ""
    pair_address: str = ""
    base_token_address: str = ""
    quote_token_address: str = ""
    combined_target_id: str = ""
    missingness_reason: str = ""
    provenance_status: str = ""
    resolver_status: str = ""
    resolver_join_path: str = ""
    resolver_confidence: float | None = None
    whale_signal_type: str = ""
    observed_at: str = ""
    fetched_at: str = ""
    attempted_at: str = ""
    lineage_tier: str = "T4"
    evidence_payload: dict[str, Any] = field(default_factory=dict)
    no_trade_authority: bool = True
    wallet_access: bool = False
    private_key_access: bool = False
    signer_available: bool = False
    transaction_signing_available: bool = False
    transaction_submission_available: bool = False
    live_trading_enabled: bool = False
    trade_authority: bool = False
    context_engine_version: str = "ae18_context_intelligence_v1"

    def to_dict(self) -> dict[str, Any]:
        d = {
            "context_record_id": self.context_record_id,
            "clean_forward_candidate_id": self.clean_forward_candidate_id,
            "clean_forward_decision_input_id": self.clean_forward_decision_input_id,
            "price_source_key": self.price_source_key,
            "chain": self.chain,
            "pair_address": self.pair_address,
            "base_token_address": self.base_token_address,
            "quote_token_address": self.quote_token_address,
            "combined_target_id": self.combined_target_id,
            "context_family": self.context_family,
            "context_status": self.context_status,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "attempted": self.attempted,
            "available": self.available,
            "missingness_reason": self.missingness_reason,
            "provenance_status": self.provenance_status,
            "resolver_status": self.resolver_status,
            "resolver_join_path": self.resolver_join_path,
            "resolver_confidence": self.resolver_confidence,
            "whale_signal_type": self.whale_signal_type,
            "observed_at": self.observed_at,
            "fetched_at": self.fetched_at,
            "attempted_at": self.attempted_at,
            "lineage_tier": self.lineage_tier,
            "context_engine_version": self.context_engine_version,
            "no_trade_authority": True,
            "wallet_access": False,
            "private_key_access": False,
            "signer_available": False,
            "transaction_signing_available": False,
            "transaction_submission_available": False,
            "live_trading_enabled": False,
            "trade_authority": False,
        }
        return d


@dataclass
class AE18ResolverLink:
    resolver_link_id: str
    context_record_id: str
    clean_forward_candidate_id: str
    join_path: str
    resolver_status: str
    price_source_key: str = ""
    chain: str = ""
    pair_address: str = ""
    token_address: str = ""
    text_item_id: str = ""
    resolver_confidence: float | None = None
    ambiguous: bool = False
    symbol_only_rejected: bool = False
    provenance_status: str = ""
    observed_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AE18MissingnessRecord:
    missingness_record_id: str
    source_name: str
    source_type: str
    attempted: bool
    available: bool
    context_status: str
    missingness_reason: str
    clean_forward_candidate_id: str = ""
    price_source_key: str = ""
    pair_address: str = ""
    chain: str = ""
    provenance_status: str = ""
    attempted_at: str = ""
    no_trade_authority: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AE18InputDiscoveryResult:
    status: str
    candidate_csv: str = ""
    candidate_count: int = 0
    source_kind: str = ""
    searched_roots: list[str] = field(default_factory=list)
    found_artifacts: list[dict[str, Any]] = field(default_factory=list)
    missing_artifacts: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
