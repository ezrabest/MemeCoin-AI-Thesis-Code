"""AE7B runtime candidate identity and scoring policy generation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.decision.types import (
    LINEAGE_FALLBACK_REASON,
    LineageMode,
    LineageResolutionMethod,
    LineageStrength,
)

RUNTIME_CANDIDATE_ID_NAMESPACE = "RUNTIME_CANDIDATE_ID_V1"
SCORING_POLICY_NAMESPACE = "AE7B_DEFAULT_NON_TRADING_SCORING_POLICY_V1"
SCORING_POLICY_VERSION = "AE7B_V1"
RUNTIME_INFERENCE_PLACEHOLDER = "AE7B_RUNTIME_INFERENCE_NOT_IMPLEMENTED"


class CandidateIdentityStatus(StrEnum):
    OK = "OK"
    BLOCKED_MISSING_STABLE_IDENTITY = "BLOCKED_MISSING_STABLE_IDENTITY"


class ScoringPolicyStatus(StrEnum):
    PLACEHOLDER_NO_MODEL_INFERENCE = "PLACEHOLDER_NO_MODEL_INFERENCE"


def normalize_address(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = str(value).strip()
    if not stripped:
        return None
    return stripped.lower()


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_identity_payload(
    *,
    chain: str | None = None,
    pair_address: str | None = None,
    base_token_address: str | None = None,
    quote_token_address: str | None = None,
    symbol: str | None = None,
    event_timestamp: str | None = None,
    source_table: str | None = None,
    source_row_id: int | str | None = None,
    provider: str | None = None,
) -> dict[str, str]:
    """Build deterministic identity payload with sorted keys."""
    payload: dict[str, str] = {}
    if chain:
        payload["chain"] = str(chain).strip()
    pair = normalize_address(pair_address)
    if pair:
        payload["pair_address"] = pair
    base = normalize_address(base_token_address)
    if base:
        payload["base_token_address"] = base
    quote = normalize_address(quote_token_address)
    if quote:
        payload["quote_token_address"] = quote
    if symbol:
        payload["symbol"] = str(symbol).strip()
    if event_timestamp:
        payload["event_timestamp"] = str(event_timestamp).strip()
    if source_table:
        payload["source_table"] = str(source_table).strip()
    if source_row_id is not None:
        payload["source_row_id"] = str(source_row_id)
    if provider:
        payload["provider"] = str(provider).strip()
    return dict(sorted(payload.items()))


def has_stable_identity_fields(payload: dict[str, str]) -> bool:
    """At least pair_address or token address combination required."""
    if payload.get("pair_address"):
        return True
    if payload.get("base_token_address") and payload.get("event_timestamp"):
        return True
    if payload.get("symbol") and payload.get("chain") and payload.get("event_timestamp"):
        return True
    return False


def generate_candidate_id(payload: dict[str, str]) -> tuple[str | None, CandidateIdentityStatus, list[str]]:
    """Return (candidate_id, status, caveats)."""
    caveats: list[str] = []
    if not payload:
        return None, CandidateIdentityStatus.BLOCKED_MISSING_STABLE_IDENTITY, [
            "no_identity_fields_available"
        ]
    if not has_stable_identity_fields(payload):
        caveats.append("pair_address_absent_using_fallback_token_identity")
        if not (
            (payload.get("base_token_address") and payload.get("event_timestamp"))
            or (payload.get("symbol") and payload.get("chain"))
        ):
            return None, CandidateIdentityStatus.BLOCKED_MISSING_STABLE_IDENTITY, [
                "no_stable_identity_fields"
            ]
    if not payload.get("pair_address"):
        caveats.append("pair_address_absent_using_next_safest_token_identity")
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = _sha256_hex(f"{RUNTIME_CANDIDATE_ID_NAMESPACE}|{serialized}")
    return digest, CandidateIdentityStatus.OK, caveats


def generate_scoring_policy_id() -> str:
    return _sha256_hex(SCORING_POLICY_NAMESPACE)


@dataclass
class ScoringPolicyIdentity:
    scoring_policy_id: str
    scoring_policy_version: str
    model_family_targets: list[str] = field(default_factory=lambda: ["RF", "XGB", "TAB"])
    horizon_candidates: list[str] = field(default_factory=lambda: ["UNKNOWN"])
    exit_policy_candidates: list[str] = field(default_factory=lambda: ["UNKNOWN"])
    policy_source: str = "AE7B_RUNTIME_BRIDGE_DEFAULT"
    scoring_policy_status: str = ScoringPolicyStatus.PLACEHOLDER_NO_MODEL_INFERENCE.value


def default_scoring_policy() -> ScoringPolicyIdentity:
    return ScoringPolicyIdentity(
        scoring_policy_id=generate_scoring_policy_id(),
        scoring_policy_version=SCORING_POLICY_VERSION,
    )


def generate_as_of_feature_row_id(
    *,
    candidate_id: str,
    scoring_policy_id: str,
    feature_schema_id: str,
    as_of_timestamp: str,
    source_snapshot_id: int | str | None = None,
    source_signal_id: int | str | None = None,
) -> str:
    parts = [
        candidate_id,
        scoring_policy_id,
        feature_schema_id,
        as_of_timestamp,
        str(source_snapshot_id or ""),
        str(source_signal_id or ""),
    ]
    return _sha256_hex("|".join(parts))


def generate_runtime_inference_id_placeholder(candidate_id: str, as_of_feature_row_id: str) -> str:
    return _sha256_hex(f"{RUNTIME_INFERENCE_PLACEHOLDER}|{candidate_id}|{as_of_feature_row_id}")


@dataclass
class BridgeLineageBlock:
    lineage_mode: str
    lineage_strength: str
    lineage_confidence_score: float
    exact_id_match: bool
    source_signal_id: int | str | None = None
    source_snapshot_id: int | str | None = None
    source_raw_payload_id: int | str | None = None
    signal_resolution_method: str = LineageResolutionMethod.MISSING.value
    snapshot_resolution_method: str = LineageResolutionMethod.MISSING.value
    raw_payload_resolution_method: str = LineageResolutionMethod.MISSING.value
    lineage_warning: str | None = None
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "lineage_mode": self.lineage_mode,
            "lineage_strength": self.lineage_strength,
            "lineage_confidence_score": self.lineage_confidence_score,
            "exact_id_match": self.exact_id_match,
            "source_signal_id": self.source_signal_id,
            "source_snapshot_id": self.source_snapshot_id,
            "source_raw_payload_id": self.source_raw_payload_id,
            "signal_resolution_method": self.signal_resolution_method,
            "snapshot_resolution_method": self.snapshot_resolution_method,
            "raw_payload_resolution_method": self.raw_payload_resolution_method,
            "lineage_warning": self.lineage_warning,
            "fallback_reason": self.fallback_reason,
        }


def compute_lineage_confidence_score(
    *,
    signal_method: LineageResolutionMethod,
    snapshot_method: LineageResolutionMethod,
    raw_method: LineageResolutionMethod,
) -> float:
    """Lineage confidence only — not trading or model confidence."""
    explicit = {
        LineageResolutionMethod.EXPLICIT_COLUMN,
        LineageResolutionMethod.FOREIGN_KEY,
        LineageResolutionMethod.DIRECT_SOURCE_REFERENCE,
    }
    stored_id = {LineageResolutionMethod.EXPLICIT_COLUMN}
    best_effort = {
        LineageResolutionMethod.BEST_EFFORT_PAIR_TIME_MATCH,
        LineageResolutionMethod.BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH,
    }

    methods = [signal_method, snapshot_method, raw_method]
    if all(m in explicit for m in methods):
        return 1.0
    if any(m == LineageResolutionMethod.MISSING for m in methods):
        return 0.0
    if all(m in stored_id or m in explicit for m in methods):
        return 0.7
    if any(m in best_effort for m in methods):
        return 0.35
    return 0.0


def build_bridge_lineage(
    *,
    signal_id: int | str | None,
    snapshot_id: int | str | None,
    raw_payload_id: int | str | None,
    signal_method: LineageResolutionMethod = LineageResolutionMethod.EXPLICIT_COLUMN,
    snapshot_method: LineageResolutionMethod = LineageResolutionMethod.BEST_EFFORT_PAIR_TIME_MATCH,
    raw_method: LineageResolutionMethod = LineageResolutionMethod.BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH,
) -> BridgeLineageBlock:
    """Build truthful weak-implicit lineage when best-effort matching is used."""
    explicit_methods = {
        LineageResolutionMethod.EXPLICIT_COLUMN,
        LineageResolutionMethod.FOREIGN_KEY,
        LineageResolutionMethod.DIRECT_SOURCE_REFERENCE,
    }
    methods = [signal_method, snapshot_method, raw_method]
    all_explicit = all(m in explicit_methods for m in methods)
    has_best_effort = any(
        m
        in {
            LineageResolutionMethod.BEST_EFFORT_PAIR_TIME_MATCH,
            LineageResolutionMethod.BEST_EFFORT_PROVIDER_PAIR_TIME_MATCH,
        }
        for m in methods
    )

    if all_explicit:
        mode = LineageMode.EXPLICIT_LINKAGE.value
        strength = LineageStrength.STRONG_EXPLICIT_LINKS.value
        exact = True
        warning = None
        fallback = None
    elif has_best_effort:
        mode = LineageMode.BEST_EFFORT_IMPLICIT_LINKAGE.value
        strength = LineageStrength.WEAK_IMPLICIT_TIME_PAIR_LINKS.value
        exact = False
        warning = LINEAGE_FALLBACK_REASON
        fallback = LINEAGE_FALLBACK_REASON
    else:
        mode = LineageMode.BEST_EFFORT_IMPLICIT_LINKAGE.value
        strength = LineageStrength.WEAK_IMPLICIT_TIME_PAIR_LINKS.value
        exact = False
        warning = LINEAGE_FALLBACK_REASON
        fallback = LINEAGE_FALLBACK_REASON

    confidence = compute_lineage_confidence_score(
        signal_method=signal_method,
        snapshot_method=snapshot_method,
        raw_method=raw_method,
    )
    if not exact and confidence >= 0.5:
        confidence = 0.35

    return BridgeLineageBlock(
        lineage_mode=mode,
        lineage_strength=strength,
        lineage_confidence_score=confidence,
        exact_id_match=exact,
        source_signal_id=signal_id,
        source_snapshot_id=snapshot_id,
        source_raw_payload_id=raw_payload_id,
        signal_resolution_method=signal_method.value,
        snapshot_resolution_method=snapshot_method.value,
        raw_payload_resolution_method=raw_method.value,
        lineage_warning=warning,
        fallback_reason=fallback,
    )
