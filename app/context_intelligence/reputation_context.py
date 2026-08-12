"""AE8 reputation / scam indicator context."""

from __future__ import annotations

import json
from typing import Any

from app.context_intelligence.freshness import apply_stale_nulling, compute_freshness
from app.context_intelligence.types import FreshnessMode, SourceStatus


def _parse_payload_flags(payload_text: str | None) -> dict[str, Any]:
    if not payload_text:
        return {}
    try:
        data = json.loads(payload_text)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def build_reputation_context(
    *,
    raw_payload_row: dict[str, Any] | None,
    coin_row: dict[str, Any] | None,
    as_of_timestamp: str,
    freshness_reference_timestamp: str,
    freshness_mode: FreshnessMode | str,
    threshold_minutes: float,
    allow_external_fetch: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], str, list[str]]:
    warnings: list[str] = []
    empty: dict[str, Any] = {
        "reputation_known_token_flag": None,
        "reputation_new_token_flag": None,
        "reputation_blacklist_flag": None,
        "reputation_contract_verified_flag": None,
        "reputation_mint_authority_flag": None,
        "reputation_freeze_authority_flag": None,
        "reputation_honeypot_risk_flag": None,
        "reputation_source_count": 0,
        "reputation_freshness_minutes": None,
        "reputation_missingness_flag": True,
    }

    if allow_external_fetch:
        warnings.append("external_reputation_fetch_disabled_by_default")
        freshness = compute_freshness(
            source_timestamp=None,
            freshness_reference_timestamp=freshness_reference_timestamp,
            freshness_mode=freshness_mode,
            threshold_minutes=threshold_minutes,
            family_key="reputation",
        )
        return empty, freshness, SourceStatus.SOURCE_DISABLED_BY_DEFAULT.value, warnings

    source_count = 0
    source_ts = None
    flags: dict[str, Any] = {}

    if raw_payload_row:
        source_ts = raw_payload_row.get("timestamp")
        source_count += 1
        payload = _parse_payload_flags(raw_payload_row.get("payload_json_or_text"))
        flags.update(
            {
                "reputation_contract_verified_flag": payload.get("verified") or payload.get("isVerified"),
                "reputation_mint_authority_flag": payload.get("mintAuthority") or payload.get("mint_authority"),
                "reputation_freeze_authority_flag": payload.get("freezeAuthority")
                or payload.get("freeze_authority"),
                "reputation_honeypot_risk_flag": payload.get("honeypot") or payload.get("honeypot_risk"),
                "reputation_blacklist_flag": payload.get("blacklisted") or payload.get("blacklist"),
            }
        )

    if coin_row and coin_row.get("token_address"):
        source_count += 1
        if source_ts is None:
            source_ts = as_of_timestamp

    if source_count == 0:
        warnings.append("REPUTATION_CONTEXT_NOT_AVAILABLE")
        freshness = compute_freshness(
            source_timestamp=None,
            freshness_reference_timestamp=freshness_reference_timestamp,
            freshness_mode=freshness_mode,
            threshold_minutes=threshold_minutes,
            family_key="reputation",
        )
        return empty, freshness, SourceStatus.SOURCE_NOT_AVAILABLE.value, warnings

    features: dict[str, Any] = {
        **empty,
        **flags,
        "reputation_known_token_flag": bool(coin_row and coin_row.get("token_address")),
        "reputation_new_token_flag": not bool(coin_row and coin_row.get("token_address")),
        "reputation_source_count": source_count,
        "reputation_missingness_flag": False,
    }

    freshness = compute_freshness(
        source_timestamp=str(source_ts) if source_ts else None,
        freshness_reference_timestamp=freshness_reference_timestamp,
        freshness_mode=freshness_mode,
        threshold_minutes=threshold_minutes,
        family_key="reputation",
    )
    features["reputation_freshness_minutes"] = freshness.get("freshness_minutes")

    if freshness.get("freshness_status") == "STALE":
        source_status = SourceStatus.SOURCE_STALE.value
    else:
        source_status = SourceStatus.SOURCE_OK.value

    features = apply_stale_nulling(features, freshness, missingness_flag_key="reputation_missingness_flag")
    return features, freshness, source_status, warnings
