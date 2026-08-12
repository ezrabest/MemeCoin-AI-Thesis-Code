"""AE17 canonical dataclasses for meta feature rows and shadow outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


def _record_to_dict(obj: Any) -> dict[str, Any]:
    return asdict(obj)


@dataclass
class AE17MetaFeatureRow:
    clean_forward_candidate_id: str = ""
    clean_forward_decision_input_id: str = ""
    price_source_key: str = ""
    provider: str = ""
    chain: str = ""
    pair_address: str = ""
    base_token_address: str = ""
    quote_token_address: str = ""
    provider_pair_url: str = ""
    provider_payload_hash: str = ""
    rf_evidence_status: str = "MODEL_EVIDENCE_UNAVAILABLE"
    xgb_evidence_status: str = "MODEL_EVIDENCE_UNAVAILABLE"
    tab_evidence_status: str = "MODEL_EVIDENCE_UNAVAILABLE"
    rf_score: float | None = None
    xgb_score: float | None = None
    tab_score: float | None = None
    rf_vote: bool = False
    xgb_vote: bool = False
    tab_vote: bool = False
    attached_model_count: int = 0
    model_vote_count: int = 0
    consensus_tier: str | None = None
    consensus_reason: str = ""
    context_status: str = "AE17_CONTEXT_NOT_AVAILABLE_PENDING_AE18"
    context_feature_available: bool = False
    context_missingness_reason: str = "AE18_NOT_IMPLEMENTED_OR_NO_CONTEXT_ATTACHED"
    context_score_weight: float = 0.0
    observed_at: str = ""
    fetched_at: str = ""
    ingested_at: str = ""
    source_ae16_artifact: str = ""
    source_schema_hash: str = ""
    lineage_status: str = "AE17_LINEAGE_INCOMPLETE"
    # Non-feature evaluation reference only (never written into feature matrix CSV columns beyond schema).
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = _record_to_dict(self)
        d.pop("warnings", None)
        return d

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls) if f.name != "warnings"]


@dataclass
class AE17MetaShadowOutput:
    clean_forward_candidate_id: str = ""
    clean_forward_decision_input_id: str = ""
    price_source_key: str = ""
    pair_address: str = ""
    consensus_tier: str | None = None
    pre_clamp_meta_score: float | None = None
    meta_score: float | None = None
    score_clamped: bool = False
    score_clamp_reason: str = ""
    meta_decision: str = "META_UNAVAILABLE"
    meta_reason: str = ""
    meta_mode: str = "rule_based_meta_shadow"
    context_score_weight: float = 0.0
    pair_concentration_status: str = "PAIR_CONCENTRATION_OK"
    authority_status: str = "AE17_RESEARCH_SHADOW_ONLY"
    trade_authority: bool = False
    live_trading_ready: bool = False
    paper_demo_only: bool = True
    risk_override_authority: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _record_to_dict(self)

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]


@dataclass
class AE17MetaDecision:
    classification: str
    meta_authority_allowed: bool = False
    ae18_status: str = "BLOCKED"
    ae19_status: str = "BLOCKED"
    reasons: list[str] = field(default_factory=list)
    checklist: dict[str, bool] = field(default_factory=dict)
    concentration_limitation: bool = False
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _record_to_dict(self)


@dataclass
class AE17MetaAuthorityStatus:
    authority_status: str = "AE17_RESEARCH_SHADOW_ONLY"
    trade_authority: bool = False
    live_trading_ready: bool = False
    paper_demo_only: bool = True
    risk_override_authority: bool = False
    wallet_access: bool = False
    private_key_access: bool = False
    live_trading_enabled: bool = False
    db_mutation: bool = False
    order_opened: bool = False
    position_opened: bool = False
    external_llm_call: bool = False
    helius_solana_call: bool = False
    training_performed: bool = False
    fit_called: bool = False
    passed: bool = True
    violations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _record_to_dict(self)


@dataclass
class AE17PairConcentrationResult:
    grouping: str
    total_rows: int = 0
    unique_pairs: int = 0
    top_pair: str = ""
    top_pair_count: int = 0
    top_pair_share: float | None = None
    pair_count_distribution: dict[str, int] = field(default_factory=dict)
    hhi: float | None = None
    top_pair_share_status: str = "PAIR_CONCENTRATION_OK"
    hhi_status: str = "HHI_LOW_CONCENTRATION"
    concentration_status: list[str] = field(default_factory=list)
    meta_authority_allowed: bool = True
    pair_concentration_penalty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d = _record_to_dict(self)
        # Flatten list for CSV friendliness when needed by callers.
        d["concentration_status"] = "|".join(self.concentration_status)
        return d


@dataclass
class AE17InputDiscoveryResult:
    status: str
    searched_roots: list[str] = field(default_factory=list)
    expected_patterns: list[str] = field(default_factory=list)
    missing_required_artifacts: list[str] = field(default_factory=list)
    found_candidate_artifacts: list[dict[str, Any]] = field(default_factory=list)
    selected_consensus_path: str | None = None
    selected_evidence_path: str | None = None
    selected_root: str | None = None
    recommended_next_action: str = ""
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _record_to_dict(self)
