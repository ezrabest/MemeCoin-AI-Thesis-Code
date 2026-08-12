"""AE18 context collectors: Helius/Solana read-only, RSS, reputation, semantic, whale separation."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.ae18.constants import (
    CONTEXT_ENGINE_VERSION,
    WHALE_SIGNAL_POOL_FLOW_PROXY,
    WHALE_SIGNAL_WALLET_LEVEL,
)
from app.ae18.models import AE18CandidateTarget, AE18ContextRecord, AE18MissingnessRecord
from app.context_intelligence.onchain_context import ONCHAIN_PROVIDERS, build_onchain_context
from app.context_intelligence.reputation_context import build_reputation_context
from app.context_intelligence.types import FreshnessMode

READONLY_RPC_METHODS = frozenset(
    {
        "getAccountInfo",
        "getTokenLargestAccounts",
        "getSignaturesForAddress",
        "getTransaction",
        "getTokenAccountsByOwner",
        "getBalance",
        "getSlot",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _record_id(*parts: str) -> str:
    return hashlib.sha256(f"AE18_CTX|{'|'.join(parts)}|{uuid.uuid4()}".encode()).hexdigest()


def _missingness_id(*parts: str) -> str:
    return hashlib.sha256(f"AE18_MISS|{'|'.join(parts)}|{uuid.uuid4()}".encode()).hexdigest()


def _base_record(candidate: AE18CandidateTarget, family: str, **kwargs: Any) -> AE18ContextRecord:
    return AE18ContextRecord(
        context_record_id=kwargs.pop("context_record_id", _record_id(candidate.clean_forward_candidate_id, family)),
        clean_forward_candidate_id=candidate.clean_forward_candidate_id,
        clean_forward_decision_input_id=candidate.clean_forward_decision_input_id,
        price_source_key=candidate.price_source_key,
        chain=candidate.chain,
        pair_address=candidate.pair_address,
        base_token_address=candidate.base_token_address,
        quote_token_address=candidate.quote_token_address,
        combined_target_id=candidate.combined_target_id,
        context_family=family,
        context_status=kwargs.pop("context_status", "CONTEXT_PENDING"),
        source_name=kwargs.pop("source_name", family),
        source_type=kwargs.pop("source_type", "local"),
        attempted=kwargs.pop("attempted", True),
        available=kwargs.pop("available", False),
        missingness_reason=kwargs.pop("missingness_reason", ""),
        provenance_status=kwargs.pop("provenance_status", ""),
        resolver_status=kwargs.pop("resolver_status", ""),
        resolver_join_path=kwargs.pop("resolver_join_path", ""),
        resolver_confidence=kwargs.pop("resolver_confidence", None),
        whale_signal_type=kwargs.pop("whale_signal_type", ""),
        observed_at=candidate.observed_at or utc_now(),
        fetched_at=candidate.fetched_at,
        attempted_at=utc_now(),
        context_engine_version=CONTEXT_ENGINE_VERSION,
        evidence_payload=kwargs.pop("evidence_payload", {}),
    )


def _missingness(
    candidate: AE18CandidateTarget,
    *,
    source_name: str,
    source_type: str,
    reason: str,
    attempted: bool = True,
) -> AE18MissingnessRecord:
    return AE18MissingnessRecord(
        missingness_record_id=_missingness_id(candidate.clean_forward_candidate_id, source_name),
        clean_forward_candidate_id=candidate.clean_forward_candidate_id,
        price_source_key=candidate.price_source_key,
        pair_address=candidate.pair_address,
        chain=candidate.chain,
        source_name=source_name,
        source_type=source_type,
        attempted=attempted,
        available=False,
        context_status="CONTEXT_UNAVAILABLE",
        missingness_reason=reason,
        provenance_status="MISSINGNESS_EMITTED",
        attempted_at=utc_now(),
    )


def collect_helius_solana_readonly(
    candidate: AE18CandidateTarget,
    *,
    raw_payload_row: dict[str, Any] | None = None,
    allow_external: bool = False,
) -> tuple[AE18ContextRecord, AE18MissingnessRecord | None, dict[str, Any]]:
    """Read-only Helius/Solana context — no wallet/signer/transaction submission."""
    rid = _record_id(candidate.clean_forward_candidate_id, "helius_solana")
    safety = {
        "wallet_access": False,
        "private_key_access": False,
        "signer_available": False,
        "transaction_signing_available": False,
        "transaction_submission_available": False,
        "readonly_rpc_methods_only": True,
        "allowed_rpc_methods": sorted(READONLY_RPC_METHODS),
    }

    if candidate.chain.lower() != "solana" and not raw_payload_row:
        miss = _missingness(
            candidate,
            source_name="helius_solana",
            source_type="solana_rpc_readonly",
            reason="PROVIDER_NOT_CALLED_IN_THIS_MODE",
            attempted=False,
        )
        rec = _base_record(
            candidate,
            "helius_solana",
            context_record_id=rid,
            context_status="CONTEXT_NOT_APPLICABLE",
            source_name="helius_solana",
            source_type="solana_rpc_readonly",
            attempted=False,
            available=False,
            missingness_reason="PROVIDER_NOT_CALLED_IN_THIS_MODE",
            provenance_status="NON_SOLANA_CANDIDATE",
            evidence_payload=safety,
        )
        return rec, miss, safety

    helius_configured = bool(os.getenv("HELIUS_API_KEY", "").strip())
    if allow_external and not helius_configured:
        miss = _missingness(
            candidate,
            source_name="helius_solana",
            source_type="solana_rpc_readonly",
            reason="SOURCE_NOT_CONFIGURED",
        )
        rec = _base_record(
            candidate,
            "helius_solana",
            context_record_id=rid,
            context_status="CONTEXT_UNAVAILABLE",
            source_name="helius_solana",
            source_type="solana_rpc_readonly",
            missingness_reason="SOURCE_NOT_CONFIGURED",
            provenance_status="HELIUS_KEY_NOT_CONFIGURED",
            evidence_payload={**safety, "helius_configured": False},
        )
        return rec, miss, safety

    # Local payload path (default — no external calls)
    features, freshness, status, warnings = build_onchain_context(
        raw_payload_row=raw_payload_row,
        as_of_timestamp=candidate.observed_at or utc_now(),
        freshness_reference_timestamp=utc_now(),
        freshness_mode=FreshnessMode.HISTORICAL_REPLAY_OR_AUDIT,
        threshold_minutes=360.0,
        allow_external_fetch=False,
    )

    available = status not in {"SOURCE_NOT_AVAILABLE", "SOURCE_CONFIG_MISSING", "SOURCE_EMPTY"}
    reason = ""
    if not available:
        reason = "SOURCE_EMPTY_RESPONSE" if status == "SOURCE_EMPTY" else "SOURCE_UNAVAILABLE_PENDING_FETCH"

    wallet_evidence = _extract_wallet_level_evidence(raw_payload_row)
    rec = _base_record(
        candidate,
        "helius_solana",
        context_record_id=rid,
        context_status="CONTEXT_AVAILABLE" if available else "CONTEXT_UNAVAILABLE",
        source_name="helius_solana",
        source_type="solana_rpc_readonly",
        available=available,
        missingness_reason=reason,
        provenance_status="LOCAL_PAYLOAD_READ_ONLY" if available else "MISSINGNESS_EMITTED",
        evidence_payload={
            **safety,
            "features": features,
            "freshness": freshness,
            "source_status": status,
            "warnings": warnings,
            "wallet_level_evidence": wallet_evidence,
            "provider": (raw_payload_row or {}).get("provider", ""),
        },
    )
    miss = _missingness(candidate, source_name="helius_solana", source_type="solana_rpc_readonly", reason=reason) if reason else None
    return rec, miss, safety


def _extract_wallet_level_evidence(raw_payload_row: dict[str, Any] | None) -> dict[str, Any]:
    """Genuine wallet-level evidence requires explicit provenance fields."""
    if not raw_payload_row:
        return {
            "available": False,
            "missingness_reason": "WALLET_LEVEL_DATA_NOT_AVAILABLE",
            "whale_signal_type": WHALE_SIGNAL_WALLET_LEVEL,
        }
    provider = str(raw_payload_row.get("provider") or "").lower()
    if provider not in ONCHAIN_PROVIDERS and "helius" not in provider and "solana" not in provider:
        return {
            "available": False,
            "missingness_reason": "WALLET_LEVEL_DATA_NOT_AVAILABLE",
            "whale_signal_type": WHALE_SIGNAL_WALLET_LEVEL,
        }
    try:
        payload = json.loads(raw_payload_row.get("payload_json_or_text") or "{}")
    except (json.JSONDecodeError, TypeError):
        payload = {}

    wallet = payload.get("wallet") or payload.get("feePayer") or payload.get("signer")
    sig = payload.get("signature") or payload.get("tx_signature")
    if wallet and sig:
        return {
            "available": True,
            "whale_signal_type": WHALE_SIGNAL_WALLET_LEVEL,
            "wallet_address": str(wallet),
            "source_transaction_signature": str(sig),
            "source_rpc_method": str(payload.get("rpc_method") or "getTransaction"),
            "source_timestamp": str(raw_payload_row.get("timestamp") or ""),
            "read_only_provider": provider or "helius",
            "confidence": payload.get("confidence"),
        }
    return {
        "available": False,
        "missingness_reason": "WALLET_LEVEL_DATA_NOT_AVAILABLE",
        "whale_signal_type": WHALE_SIGNAL_WALLET_LEVEL,
    }


def collect_rss_news_context(
    candidate: AE18CandidateTarget,
    conn: sqlite3.Connection | None,
    *,
    text_items: list[dict[str, Any]] | None = None,
) -> tuple[AE18ContextRecord, AE18MissingnessRecord | None, list[dict[str, Any]]]:
    """RSS/news context — never joins by symbol alone; uses explicit candidate identity."""
    rid = _record_id(candidate.clean_forward_candidate_id, "rss_news")
    normalized_items: list[dict[str, Any]] = []

    if text_items:
        for item in text_items:
            normalized_items.append(_normalize_rss_item(item, candidate))

    elif conn is not None and candidate.price_source_key:
        # Explicit identity query — chain+token or price_source_key metadata, NOT symbol alone
        sql = """
            SELECT id, timestamp, sentiment_score, source, title, symbols_json, url
            FROM sentiment_records
            WHERE (symbols_json LIKE ? OR symbols_json LIKE ?)
              AND timestamp <= ?
            ORDER BY timestamp DESC
            LIMIT 50
        """
        token_hint = f"%{candidate.base_token_address}%"
        pair_hint = f"%{candidate.pair_address}%"
        as_of = candidate.observed_at or utc_now()
        try:
            rows = conn.execute(sql, (token_hint, pair_hint, as_of)).fetchall()
            for row in rows:
                normalized_items.append(
                    {
                        "text_item_id": str(row[0]),
                        "timestamp": str(row[1]),
                        "sentiment_score": row[2],
                        "source": str(row[3] or ""),
                        "title": str(row[4] or ""),
                        "symbols_json": str(row[5] or "[]"),
                        "url": str(row[6] or ""),
                        "price_source_key": candidate.price_source_key,
                        "chain": candidate.chain,
                        "pair_address": candidate.pair_address,
                        "token_address": candidate.base_token_address,
                        "clean_forward_candidate_id": candidate.clean_forward_candidate_id,
                    }
                )
        except sqlite3.Error:
            pass

    available = len(normalized_items) > 0
    reason = "" if available else "SOURCE_EMPTY_RESPONSE"
    if conn is None and not text_items:
        reason = "SOURCE_NOT_AVAILABLE_PENDING_FETCH"

    rec = _base_record(
        candidate,
        "rss_news",
        context_record_id=rid,
        context_status="CONTEXT_AVAILABLE" if available else "CONTEXT_UNAVAILABLE",
        source_name="rss_news",
        source_type="rss_normalized",
        available=available,
        missingness_reason=reason,
        provenance_status="T1_NORMALIZED_ARTICLE" if available else "MISSINGNESS_EMITTED",
        resolver_join_path="price_source_key" if available else "",
        evidence_payload={"article_count": len(normalized_items), "items": normalized_items[:10]},
    )
    miss = (
        _missingness(candidate, source_name="rss_news", source_type="rss_normalized", reason=reason)
        if reason
        else None
    )
    return rec, miss, normalized_items


def _normalize_rss_item(item: dict[str, Any], candidate: AE18CandidateTarget) -> dict[str, Any]:
    return {
        "text_item_id": str(item.get("text_item_id") or item.get("id") or uuid.uuid4()),
        "timestamp": str(item.get("timestamp") or item.get("published_at") or ""),
        "title": str(item.get("title") or ""),
        "source": str(item.get("source") or item.get("feed_name") or ""),
        "sentiment_score": item.get("sentiment_score"),
        "url": str(item.get("url") or ""),
        "price_source_key": item.get("price_source_key") or candidate.price_source_key,
        "chain": item.get("chain") or candidate.chain,
        "pair_address": item.get("pair_address") or candidate.pair_address,
        "token_address": item.get("token_address") or candidate.base_token_address,
        "clean_forward_candidate_id": item.get("clean_forward_candidate_id") or candidate.clean_forward_candidate_id,
        "entity_link_confidence": item.get("entity_link_confidence"),
        "ambiguous_entity_link": bool(item.get("ambiguous_entity_link", False)),
    }


def collect_reputation_scam_context(
    candidate: AE18CandidateTarget,
    *,
    raw_payload_row: dict[str, Any] | None = None,
) -> tuple[AE18ContextRecord, AE18MissingnessRecord | None]:
    rid = _record_id(candidate.clean_forward_candidate_id, "reputation_scam")
    features, freshness, status, warnings = build_reputation_context(
        raw_payload_row=raw_payload_row,
        coin_row={"token_address": candidate.base_token_address} if candidate.base_token_address else None,
        as_of_timestamp=candidate.observed_at or utc_now(),
        freshness_reference_timestamp=utc_now(),
        freshness_mode=FreshnessMode.HISTORICAL_REPLAY_OR_AUDIT,
        threshold_minutes=360.0,
        allow_external_fetch=False,
    )
    available = status not in {"SOURCE_NOT_AVAILABLE", "SOURCE_EMPTY"}
    reason = "" if available else "SOURCE_UNAVAILABLE_PENDING_FETCH"
    rec = _base_record(
        candidate,
        "reputation_scam",
        context_record_id=rid,
        context_status="CONTEXT_AVAILABLE" if available else "CONTEXT_UNAVAILABLE",
        source_name="reputation_scam",
        source_type="reputation_provider",
        available=available,
        missingness_reason=reason,
        provenance_status="REPUTATION_EVIDENCE" if available else "MISSINGNESS_EMITTED",
        evidence_payload={"features": features, "freshness": freshness, "warnings": warnings},
    )
    miss = (
        _missingness(candidate, source_name="reputation_scam", source_type="reputation_provider", reason=reason)
        if reason
        else None
    )
    return rec, miss


def collect_semantic_context(candidate: AE18CandidateTarget) -> tuple[AE18ContextRecord, AE18MissingnessRecord | None]:
    rid = _record_id(candidate.clean_forward_candidate_id, "semantic")
    # Semantic context from candidate metadata only — no LLM entity invention
    semantic_status = "PENDING_SYSTEM_CLASSIFICATION"
    available = bool(candidate.token_name or candidate.token_symbol or candidate.combined_target_id)
    evidence = {
        "semantic_status": semantic_status,
        "token_symbol": candidate.token_symbol,
        "token_name": candidate.token_name,
        "combined_target_id": candidate.combined_target_id,
        "llm_entity_links_invented": False,
        "unresolved_flags": not available,
    }
    reason = "" if available else "SOURCE_EMPTY_RESPONSE"
    rec = _base_record(
        candidate,
        "semantic",
        context_record_id=rid,
        context_status="CONTEXT_PARTIAL" if available else "CONTEXT_UNAVAILABLE",
        source_name="semantic",
        source_type="candidate_metadata",
        available=available,
        missingness_reason=reason,
        provenance_status="T3_CANDIDATE_METADATA_ONLY",
        evidence_payload=evidence,
    )
    miss = (
        _missingness(candidate, source_name="semantic", source_type="candidate_metadata", reason=reason)
        if reason
        else None
    )
    return rec, miss


def collect_whale_evidence_separated(
    candidate: AE18CandidateTarget,
    *,
    snapshot_whale_score: float | str | None = None,
    wallet_evidence: dict[str, Any] | None = None,
) -> tuple[AE18ContextRecord, AE18ContextRecord | None, AE18MissingnessRecord | None]:
    """Separate legacy whale_score (POOL_FLOW_PROXY) from wallet-level whale evidence."""
    rid = _record_id(candidate.clean_forward_candidate_id, "whale_pool_flow")
    score = snapshot_whale_score if snapshot_whale_score is not None else candidate.whale_score
    # Legacy whale_score is always POOL_FLOW_PROXY
    pool_rec = _base_record(
        candidate,
        "whale_pool_flow_proxy",
        context_record_id=rid,
        context_status="CONTEXT_AVAILABLE" if score is not None and str(score).strip() != "" else "CONTEXT_UNAVAILABLE",
        source_name="legacy_whale_score",
        source_type="pool_flow_proxy",
        available=score is not None and str(score).strip() != "",
        whale_signal_type=WHALE_SIGNAL_POOL_FLOW_PROXY,
        missingness_reason="" if score is not None and str(score).strip() != "" else "SOURCE_EMPTY_RESPONSE",
        provenance_status="POOL_FLOW_PROXY_NOT_WALLET_EVIDENCE",
        evidence_payload={
            "whale_score": score,
            "whale_signal_type": WHALE_SIGNAL_POOL_FLOW_PROXY,
            "description": "DexScreener/pool-flow proxy derived from volume/liquidity/txns — NOT wallet-level whale evidence",
            "not_wallet_level_whale_evidence": True,
            "not_on_chain_whale_evidence": True,
        },
    )
    pool_miss = (
        _missingness(
            candidate,
            source_name="legacy_whale_score",
            source_type="pool_flow_proxy",
            reason="SOURCE_EMPTY_RESPONSE",
        )
        if not pool_rec.available
        else None
    )

    wallet_rec: AE18ContextRecord | None = None
    wallet_miss: AE18MissingnessRecord | None = None
    we = wallet_evidence or {}
    if we.get("available"):
        wallet_rec = _base_record(
            candidate,
            "wallet_whale",
            context_record_id=_record_id(candidate.clean_forward_candidate_id, "wallet_whale"),
            context_status="CONTEXT_AVAILABLE",
            source_name="helius_solana",
            source_type="wallet_level_readonly",
            available=True,
            whale_signal_type=WHALE_SIGNAL_WALLET_LEVEL,
            provenance_status="WALLET_LEVEL_READONLY_PROVENANCE",
            evidence_payload=we,
        )
    else:
        wallet_miss = _missingness(
            candidate,
            source_name="wallet_whale",
            source_type="wallet_level_readonly",
            reason=str(we.get("missingness_reason") or "WALLET_LEVEL_DATA_NOT_AVAILABLE"),
        )
        wallet_rec = _base_record(
            candidate,
            "wallet_whale",
            context_record_id=_record_id(candidate.clean_forward_candidate_id, "wallet_whale"),
            context_status="CONTEXT_UNAVAILABLE",
            source_name="helius_solana",
            source_type="wallet_level_readonly",
            available=False,
            whale_signal_type=WHALE_SIGNAL_WALLET_LEVEL,
            missingness_reason=str(we.get("missingness_reason") or "WALLET_LEVEL_DATA_NOT_AVAILABLE"),
            provenance_status="WALLET_LEVEL_MISSINGNESS_EMITTED",
            evidence_payload=we,
        )

    return pool_rec, wallet_rec, pool_miss or wallet_miss


def open_readonly_db(project_root: Path) -> sqlite3.Connection | None:
    db = project_root / "trader.db"
    if not db.is_file():
        return None
    uri = f"file:{db.resolve()}?mode=ro"
    try:
        return sqlite3.connect(uri, uri=True)
    except sqlite3.Error:
        return None


def fetch_local_onchain_payload(
    conn: sqlite3.Connection | None,
    candidate: AE18CandidateTarget,
) -> dict[str, Any] | None:
    if conn is None:
        return None
    pair = candidate.pair_address
    if not pair:
        return None
    sql = """
        SELECT id, provider, source_type, timestamp, payload_json_or_text
        FROM raw_provider_payloads
        WHERE (pair_address = ? OR payload_json_or_text LIKE ?)
          AND (provider IN ('helius','solana_rpc','solana','helius_rpc')
               OR source_type LIKE '%solana%' OR source_type LIKE '%helius%')
        ORDER BY timestamp DESC
        LIMIT 1
    """
    try:
        row = conn.execute(sql, (pair, f"%{pair}%")).fetchone()
        if not row:
            return None
        return {
            "id": row[0],
            "provider": row[1],
            "source_type": row[2],
            "timestamp": row[3],
            "payload_json_or_text": row[4],
            "pair_address": pair,
        }
    except sqlite3.Error:
        return None
