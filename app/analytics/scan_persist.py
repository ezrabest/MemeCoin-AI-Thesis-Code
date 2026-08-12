"""
Scan-cycle persistence helpers — archive raw data, upsert coins, snapshots, signals, alerts.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, TypeVar

from .. import database as db

T = TypeVar("T")
from ..engine import compute_whale_score, detect_whale_alert, generate_signal
from ..observability.pipeline_audit import (
    derive_alert_audit_reasons,
    derive_signal_audit_reasons,
    safe_record_pipeline_decision,
)

log = logging.getLogger("scan_persist")


def _safe_persist(stage: str, symbol: str, fn: Callable[[], T]) -> T | None:
    try:
        return fn()
    except Exception as exc:
        log.warning("persist stage %s failed symbol=%s: %s", stage, symbol, exc)
        return None


def new_scan_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]


def _count_pairs_in_response(response_data: dict[str, Any] | list | None) -> int:
    if response_data is None:
        return 0
    if isinstance(response_data, list):
        return len(response_data)
    if isinstance(response_data, dict):
        if "pairs" in response_data and isinstance(response_data["pairs"], list):
            return len(response_data["pairs"])
        if "pair_count" in response_data:
            try:
                return int(response_data["pair_count"])
            except (TypeError, ValueError):
                return 0
    return 0


def archive_dexscreener_search(
    query: str,
    response_data: dict[str, Any] | list | None,
    *,
    source_type: str = "search_response",
) -> int | None:
    if response_data is None:
        return None
    try:
        ref_id = db.insert_raw_payload(
            provider="dexscreener",
            payload=response_data,
            source_type=source_type,
            query=query,
        )
        if ref_id:
            pairs = _count_pairs_in_response(response_data)
            log.info("raw payload archived: provider=dexscreener query=%s pairs=%s", query, pairs)
        return ref_id
    except Exception as exc:
        log.warning("Dexscreener raw archival failed query=%s: %s", query, exc)
        return None


def archive_dexscreener_pair(pair: dict[str, Any], *, query: str = "") -> int | None:
    base = pair.get("baseToken") or {}
    quote = pair.get("quoteToken") or {}
    symbol = f"{base.get('symbol', '?')}/{quote.get('symbol', '?')}"
    try:
        return db.insert_raw_payload(
            provider="dexscreener",
            payload=pair,
            source_type="pair",
            query=query,
            chain=(pair.get("chainId") or "").lower(),
            pair_address=pair.get("pairAddress") or "",
            symbol=symbol,
        )
    except Exception as exc:
        log.warning("Dexscreener pair archival failed symbol=%s: %s", symbol, exc)
        return None


def persist_pair_pipeline(
    pair: dict[str, Any],
    *,
    scan_id: str,
    source_query: str = "",
    filter_status: str = "passed",
    drop_reason: str = "",
    cluster_label: str = "",
    sentiment_score: float = 0.0,
) -> dict[str, Any] | None:
    """
    Full persistence pipeline for one DexScreener pair (before/after filter stages logged).
    Returns dict with coin_id and optional whale alert info. Does NOT call Gemini.
    """
    try:
        return _persist_pair_pipeline_impl(
            pair,
            scan_id=scan_id,
            source_query=source_query,
            filter_status=filter_status,
            drop_reason=drop_reason,
            cluster_label=cluster_label,
            sentiment_score=sentiment_score,
        )
    except Exception as exc:
        base = pair.get("baseToken") or {}
        quote = pair.get("quoteToken") or {}
        symbol = f"{base.get('symbol', '?')}/{quote.get('symbol', '?')}"
        log.error("persist_pair_pipeline failed symbol=%s: %s", symbol, exc, exc_info=True)
        return None


def _persist_pair_pipeline_impl(
    pair: dict[str, Any],
    *,
    scan_id: str,
    source_query: str = "",
    filter_status: str = "passed",
    drop_reason: str = "",
    cluster_label: str = "",
    sentiment_score: float = 0.0,
) -> dict[str, Any] | None:
    base = pair.get("baseToken") or {}
    quote = pair.get("quoteToken") or {}
    symbol = f"{base.get('symbol', '?')}/{quote.get('symbol', '?')}"
    chain = (pair.get("chainId") or "unknown").lower()
    pair_address = (pair.get("pairAddress") or "").strip()
    txns = (pair.get("txns") or {}).get("h24") or {}
    buys = int(txns.get("buys") or 0)
    sells = int(txns.get("sells") or 0)
    whale_score = compute_whale_score(pair)

    raw_ref = archive_dexscreener_pair(pair, query=source_query)

    _safe_persist(
        "pipeline_audit_before",
        symbol,
        lambda: db.insert_pipeline_audit({
            "scan_id": scan_id,
            "symbol": symbol,
            "pair_address": pair_address,
            "stage": "before_filter",
            "filter_status": filter_status,
            "drop_reason": drop_reason or None,
            "whale_score": whale_score,
            "details_json": {"source_query": source_query},
        }),
    )

    if filter_status == "dropped":
        coin = db.upsert_coin({
            "symbol": symbol,
            "name": base.get("name", base.get("symbol", "?")),
            "chain": chain,
            "pair_address": pair_address,
            "token_address": base.get("address"),
            "price_usd": float(pair.get("priceUsd") or 0),
            "volume_24h": float((pair.get("volume") or {}).get("h24") or 0),
            "liquidity_usd": float((pair.get("liquidity") or {}).get("usd") or 0),
            "whale_score": whale_score,
            "raw_ref_id": raw_ref,
        })
        coin_id = coin["id"] if coin else None
        if coin_id:
            _safe_persist(
                "snapshot_dropped",
                symbol,
                lambda: db.insert_market_snapshot({
                    "coin_id": coin_id,
                    "chain": chain,
                    "pair_address": pair_address,
                    "price": float(pair.get("priceUsd") or 0),
                    "liquidity": float((pair.get("liquidity") or {}).get("usd") or 0),
                    "volume_24h": float((pair.get("volume") or {}).get("h24") or 0),
                    "whale_score": whale_score,
                    "filter_status": "dropped",
                    "drop_reason": drop_reason,
                    "source_query": source_query,
                }),
            )
        _safe_persist(
            "pipeline_audit_dropped",
            symbol,
            lambda: db.insert_pipeline_audit({
            "scan_id": scan_id,
            "coin_id": coin_id,
            "symbol": symbol,
            "pair_address": pair_address,
            "stage": "after_filter",
            "filter_status": "dropped",
            "drop_reason": drop_reason,
            "whale_score": whale_score,
            }),
        )
        return {
            "dropped": True,
            "coin_id": coin_id,
            "symbol": symbol,
            "drop_reason": drop_reason,
            "whale_score": whale_score,
        }

    coin = db.upsert_coin({
        "symbol": symbol,
        "name": base.get("name", base.get("symbol", "?")),
        "base_symbol": base.get("symbol"),
        "quote_symbol": quote.get("symbol"),
        "chain": chain,
        "pair_address": pair_address,
        "token_address": base.get("address"),
        "price_usd": float(pair.get("priceUsd") or 0),
        "volume_24h": float((pair.get("volume") or {}).get("h24") or 0),
        "liquidity_usd": float((pair.get("liquidity") or {}).get("usd") or 0),
        "price_change_24h": float((pair.get("priceChange") or {}).get("h24") or 0),
        "price_change_1h": float((pair.get("priceChange") or {}).get("h1") or 0),
        "price_change_6h": float((pair.get("priceChange") or {}).get("h6") or 0),
        "fdv": float(pair.get("fdv") or 0) or None,
        "whale_score": whale_score,
        "dex_url": pair.get("url"),
        "raw_ref_id": raw_ref,
        "pair_age": pair.get("pairCreatedAt"),
    })
    if not coin:
        return None

    coin_id = coin["id"]
    buy_ratio = buys / max(buys + sells, 1)

    _safe_persist(
        "snapshot",
        symbol,
        lambda: db.insert_market_snapshot({
            "coin_id": coin_id,
            "chain": chain,
            "pair_address": pair_address,
            "price": float(pair.get("priceUsd") or 0),
            "liquidity": float((pair.get("liquidity") or {}).get("usd") or 0),
            "volume_24h": float((pair.get("volume") or {}).get("h24") or 0),
            "fdv": float(pair.get("fdv") or 0) or None,
            "txns_buys": buys,
            "txns_sells": sells,
            "txns_total": buys + sells,
            "price_change_m5": float((pair.get("priceChange") or {}).get("m5") or 0),
            "price_change_h1": float((pair.get("priceChange") or {}).get("h1") or 0),
            "price_change_h6": float((pair.get("priceChange") or {}).get("h6") or 0),
            "price_change_h24": float((pair.get("priceChange") or {}).get("h24") or 0),
            "whale_score": whale_score,
            "buy_ratio": round(buy_ratio, 4),
            "source_query": source_query,
            "filter_status": "passed",
        }),
    )

    sig = generate_signal(pair, whale_score)
    alert = detect_whale_alert(pair, whale_score)
    alert_type = alert["alert_type"] if alert else None

    liq_usd = float((pair.get("liquidity") or {}).get("usd") or 0)
    signal_reasons = derive_signal_audit_reasons(
        signal_action=sig["action"],
        prob_up=sig["probability_up"],
        whale_score=whale_score,
        liquidity_usd=liq_usd,
        alert_type=alert_type,
    )
    if not alert:
        signal_reasons.extend(
            derive_alert_audit_reasons(
                alert=None,
                whale_score=whale_score,
                volume_24h=float((pair.get("volume") or {}).get("h24") or 0),
            )
        )
    safe_record_pipeline_decision(
        pair_address=pair_address,
        coin_id=None,
        chain=chain,
        symbol=symbol,
        audit_reasons=signal_reasons,
        scan_id=scan_id,
        signal_action=sig["action"],
        alert_type=alert_type,
        whale_score=whale_score,
        current_execution_price=float(pair.get("priceUsd") or 0) or None,
        stage="signal_generated",
    )

    _safe_persist(
        "signal",
        symbol,
        lambda: db.insert_signal({
            "coin_id": coin_id,
            "symbol": symbol,
            "action": sig["action"],
            "probability_up": sig["probability_up"],
            "score": sig["probability_up"],
            "confidence": sig["probability_up"],
            "explanation": sig["explanation"],
            "features_json": {
                "volume_24h": coin.get("volume_24h"),
                "liquidity_usd": coin.get("liquidity_usd"),
                "cluster_label": cluster_label,
                "sentiment_score": sentiment_score,
            },
        }),
    )

    alert_info: dict[str, Any] | None = None
    if alert:
        _safe_persist(
            "whale_alert",
            symbol,
            lambda: db.insert_whale_alert({
                "coin_id": coin_id,
                "symbol": symbol,
                "chain": chain,
                "pair_address": pair_address,
                "alert_type": alert["alert_type"],
                "volume_usd": alert["volume_usd"],
                "price_impact_pct": alert["price_impact_pct"],
                "tx_count": alert["tx_count"],
                "description": alert["description"],
                "whale_score": whale_score,
                "liquidity_usd": coin.get("liquidity_usd"),
                "is_real_wallet_level": False,
                "raw_ref_id": raw_ref,
            }),
        )
        alert_info = alert

    if coin_id:
        safe_record_pipeline_decision(
            pair_address=pair_address,
            coin_id=coin_id,
            chain=chain,
            symbol=symbol,
            audit_reasons=signal_reasons,
            scan_id=scan_id,
            signal_action=sig["action"],
            alert_type=alert_type,
            whale_score=whale_score,
            current_execution_price=float(pair.get("priceUsd") or 0) or None,
            stage="after_persist",
        )

    _safe_persist(
        "pipeline_audit_after",
        symbol,
        lambda: db.insert_pipeline_audit({
            "scan_id": scan_id,
            "coin_id": coin_id,
            "symbol": symbol,
            "pair_address": pair_address,
            "stage": "after_filter",
            "filter_status": "passed",
            "whale_score": whale_score,
            "alert_type": alert["alert_type"] if alert else None,
            "details_json": {"cluster_label": cluster_label, "signal": sig["action"]},
        }),
    )

    return {
        "coin_id": coin_id,
        "coin": coin,
        "symbol": symbol,
        "chain": chain,
        "whale_score": whale_score,
        "alert": alert_info,
        "signal_action": sig["action"],
    }
