"""AE8 deterministic context schema builder and validation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.context_intelligence.types import (
    AE8_CONTEXT_SCHEMA_VERSION,
    ALL_CONTEXT_FEATURE_NAMES,
    DEFAULT_FRESHNESS_THRESHOLDS_MINUTES,
    FEATURE_FAMILIES,
    FORBIDDEN_CONTEXT_FEATURE_PATTERNS,
    is_forbidden_context_feature,
)


@dataclass
class ContextSchema:
    context_schema_id: str
    context_schema_version: str
    feature_names: list[str]
    feature_families: dict[str, list[str]]
    feature_dtypes: dict[str, str]
    source_map: dict[str, dict[str, str]]
    forbidden_feature_patterns: list[str]
    freshness_thresholds: dict[str, float]
    schema_hash: str
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    rejected_features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_schema_id": self.context_schema_id,
            "context_schema_version": self.context_schema_version,
            "feature_names": self.feature_names,
            "feature_families": self.feature_families,
            "feature_dtypes": self.feature_dtypes,
            "source_map": self.source_map,
            "forbidden_feature_patterns": self.forbidden_feature_patterns,
            "freshness_thresholds": self.freshness_thresholds,
            "schema_hash": self.schema_hash,
            "created_at_utc": self.created_at_utc,
            "rejected_features": self.rejected_features,
        }


_FEATURE_DTYPES: dict[str, str] = {
    name: (
        "bool"
        if name.endswith("_flag")
        else "int"
        if "count" in name or name.endswith("_total")
        else "float"
    )
    for name in ALL_CONTEXT_FEATURE_NAMES
}

_SOURCE_MAP: dict[str, dict[str, str]] = {
    **{n: {"source_table": "sentiment_records", "source_family": "rss"} for n in FEATURE_FAMILIES["rss"]},
    **{n: {"source_table": "raw_provider_payloads", "source_family": "onchain"} for n in FEATURE_FAMILIES["onchain"]},
    **{
        n: {"source_table": "market_snapshots|whale_alerts", "source_family": "whale"}
        for n in FEATURE_FAMILIES["whale"]
    },
    **{
        n: {"source_table": "raw_provider_payloads|coins", "source_family": "reputation"}
        for n in FEATURE_FAMILIES["reputation"]
    },
    **{
        n: {"source_table": "market_snapshots|signals", "source_family": "liquidity_activity"}
        for n in FEATURE_FAMILIES["liquidity_activity"]
    },
}


def compute_schema_hash(content: dict[str, Any]) -> str:
    serialized = json.dumps(content, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_context_schema_id(core: dict[str, Any]) -> str:
    schema_hash = compute_schema_hash(core)
    return hashlib.sha256(f"{AE8_CONTEXT_SCHEMA_VERSION}|{schema_hash}".encode("utf-8")).hexdigest()


def validate_feature_names(names: list[str]) -> tuple[list[str], list[dict[str, str]]]:
    safe: list[str] = []
    rejected: list[dict[str, str]] = []
    for name in names:
        if is_forbidden_context_feature(name):
            rejected.append({"feature_name": name, "reason": "forbidden_pattern_match"})
        else:
            safe.append(name)
    return safe, rejected


def build_context_schema(
    *,
    freshness_thresholds: dict[str, float] | None = None,
) -> ContextSchema:
    thresholds = dict(DEFAULT_FRESHNESS_THRESHOLDS_MINUTES)
    if freshness_thresholds:
        thresholds.update(freshness_thresholds)

    candidate_names = sorted(ALL_CONTEXT_FEATURE_NAMES)
    safe_names, rejected_audit = validate_feature_names(candidate_names)
    rejected = [r["feature_name"] for r in rejected_audit]

    core = {
        "version": AE8_CONTEXT_SCHEMA_VERSION,
        "feature_names": safe_names,
        "feature_families": {k: list(v) for k, v in FEATURE_FAMILIES.items()},
        "feature_dtypes": {n: _FEATURE_DTYPES.get(n, "float") for n in safe_names},
        "freshness_thresholds": thresholds,
        "forbidden_feature_patterns": list(FORBIDDEN_CONTEXT_FEATURE_PATTERNS),
    }
    schema_hash = compute_schema_hash(core)
    schema_id = compute_context_schema_id(core)

    return ContextSchema(
        context_schema_id=schema_id,
        context_schema_version=AE8_CONTEXT_SCHEMA_VERSION,
        feature_names=safe_names,
        feature_families={k: list(v) for k, v in FEATURE_FAMILIES.items()},
        feature_dtypes=core["feature_dtypes"],
        source_map={n: _SOURCE_MAP[n] for n in safe_names if n in _SOURCE_MAP},
        forbidden_feature_patterns=list(FORBIDDEN_CONTEXT_FEATURE_PATTERNS),
        freshness_thresholds=thresholds,
        schema_hash=schema_hash,
        rejected_features=rejected,
    )
