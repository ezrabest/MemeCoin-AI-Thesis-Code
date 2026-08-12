"""AE8 Helius / Solana on-chain context from local payloads."""

from __future__ import annotations

import json
import os
from typing import Any

from app.context_intelligence.freshness import apply_stale_nulling, compute_freshness
from app.context_intelligence.types import FreshnessMode, SourceStatus

ONCHAIN_PROVIDERS = frozenset({"helius", "solana_rpc", "solana", "helius_rpc"})


def _helius_config_present() -> bool:
    return bool(os.getenv("HELIUS_API_KEY", "").strip())


def _parse_onchain_payload(payload_text: str | None) -> dict[str, Any]:
    if not payload_text:
        return {}
    try:
        data = json.loads(payload_text)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _extract_onchain_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    """Best-effort extraction from stored provider payloads."""
    txs = payload.get("transactions") or payload.get("txns") or []
    if isinstance(txs, dict):
        txs = txs.get("items") or txs.get("data") or []
    if not isinstance(txs, list):
        txs = []

    wallets: set[str] = set()
    large_count = 0
    large_usd = 0.0
    amounts: list[float] = []

    for tx in txs[:500]:
        if not isinstance(tx, dict):
            continue
        signer = tx.get("feePayer") or tx.get("wallet") or tx.get("from")
        if signer:
            wallets.add(str(signer))
        amt = tx.get("amountUsd") or tx.get("usd_amount") or tx.get("value")
        if amt is not None:
            try:
                val = float(amt)
                amounts.append(val)
                if val >= 10_000:
                    large_count += 1
                    large_usd += val
            except (TypeError, ValueError):
                pass

    amounts_sorted = sorted(amounts, reverse=True)
    total_amt = sum(amounts_sorted) or 1.0
    top1 = amounts_sorted[0] / total_amt if amounts_sorted else None
    top5 = sum(amounts_sorted[:5]) / total_amt if amounts_sorted else None

    return {
        "onchain_txn_count_1h": len(txs),
        "onchain_txn_count_24h": len(txs),
        "onchain_unique_wallets_24h": len(wallets),
        "onchain_new_wallet_ratio_24h": None,
        "onchain_large_transfer_count_24h": large_count,
        "onchain_large_transfer_usd_24h": round(large_usd, 4) if large_usd else None,
        "onchain_wallet_concentration_top1": round(top1, 6) if top1 is not None else None,
        "onchain_wallet_concentration_top5": round(top5, 6) if top5 is not None else None,
        "onchain_contract_age_minutes": payload.get("contract_age_minutes"),
        "onchain_lp_lock_signal": payload.get("lp_locked"),
        "onchain_authority_risk_flag": payload.get("authority_risk"),
    }


def build_onchain_context(
    *,
    raw_payload_row: dict[str, Any] | None,
    as_of_timestamp: str,
    freshness_reference_timestamp: str,
    freshness_mode: FreshnessMode | str,
    threshold_minutes: float,
    allow_external_fetch: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], str, list[str]]:
    warnings: list[str] = []
    empty = {
        "onchain_txn_count_1h": None,
        "onchain_txn_count_24h": None,
        "onchain_unique_wallets_24h": None,
        "onchain_new_wallet_ratio_24h": None,
        "onchain_large_transfer_count_24h": None,
        "onchain_large_transfer_usd_24h": None,
        "onchain_wallet_concentration_top1": None,
        "onchain_wallet_concentration_top5": None,
        "onchain_contract_age_minutes": None,
        "onchain_lp_lock_signal": None,
        "onchain_authority_risk_flag": None,
        "onchain_freshness_minutes": None,
        "onchain_missingness_flag": True,
    }

    if allow_external_fetch and not _helius_config_present():
        warnings.append("ONCHAIN_CONTEXT_CONFIG_MISSING")
        freshness = compute_freshness(
            source_timestamp=None,
            freshness_reference_timestamp=freshness_reference_timestamp,
            freshness_mode=freshness_mode,
            threshold_minutes=threshold_minutes,
            family_key="onchain",
        )
        return empty, freshness, SourceStatus.SOURCE_CONFIG_MISSING.value, warnings

    if allow_external_fetch:
        warnings.append("external_onchain_fetch_not_implemented_use_local_payloads")
        return (
            empty,
            compute_freshness(
                source_timestamp=None,
                freshness_reference_timestamp=freshness_reference_timestamp,
                freshness_mode=freshness_mode,
                threshold_minutes=threshold_minutes,
                family_key="onchain",
            ),
            SourceStatus.SOURCE_DISABLED_BY_DEFAULT.value,
            warnings,
        )

    if not raw_payload_row:
        warnings.append("ONCHAIN_CONTEXT_NOT_AVAILABLE")
        freshness = compute_freshness(
            source_timestamp=None,
            freshness_reference_timestamp=freshness_reference_timestamp,
            freshness_mode=freshness_mode,
            threshold_minutes=threshold_minutes,
            family_key="onchain",
        )
        return empty, freshness, SourceStatus.SOURCE_NOT_AVAILABLE.value, warnings

    provider = str(raw_payload_row.get("provider") or "").lower()
    source_type = str(raw_payload_row.get("source_type") or "").lower()
    is_onchain = provider in ONCHAIN_PROVIDERS or "helius" in provider or "solana" in source_type

    if not is_onchain:
        warnings.append("ONCHAIN_CONTEXT_NOT_AVAILABLE")
        freshness = compute_freshness(
            source_timestamp=raw_payload_row.get("timestamp"),
            freshness_reference_timestamp=freshness_reference_timestamp,
            freshness_mode=freshness_mode,
            threshold_minutes=threshold_minutes,
            family_key="onchain",
        )
        return empty, freshness, SourceStatus.SOURCE_NOT_AVAILABLE.value, warnings

    payload = _parse_onchain_payload(raw_payload_row.get("payload_json_or_text"))
    metrics = _extract_onchain_metrics(payload)
    features = {**empty, **metrics, "onchain_missingness_flag": False}

    freshness = compute_freshness(
        source_timestamp=str(raw_payload_row.get("timestamp") or ""),
        freshness_reference_timestamp=freshness_reference_timestamp,
        freshness_mode=freshness_mode,
        threshold_minutes=threshold_minutes,
        family_key="onchain",
    )
    features["onchain_freshness_minutes"] = freshness.get("freshness_minutes")

    if freshness.get("freshness_status") == "STALE":
        source_status = SourceStatus.SOURCE_STALE.value
    elif freshness.get("freshness_status") == "INVALID_FUTURE_TIMESTAMP":
        source_status = SourceStatus.SOURCE_ERROR.value
    else:
        source_status = SourceStatus.SOURCE_OK.value

    features = apply_stale_nulling(features, freshness, missingness_flag_key="onchain_missingness_flag")
    return features, freshness, source_status, warnings
