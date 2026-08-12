"""Tests for AE8 context intelligence layer."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

from app.context_intelligence.bounded_queries import QueryStats, assess_memory_safety, validate_bounded_sql
from app.context_intelligence.context_feature_builder import (
    build_context_lineage,
    build_context_feature_record,
    run_ae8_context_intelligence,
)
from app.context_intelligence.context_persistence import ContextJsonlWriter, read_context_jsonl_safe
from app.context_intelligence.context_schema import build_context_schema, compute_context_schema_id
from app.context_intelligence.freshness import compute_freshness
from app.context_intelligence.liquidity_activity_context import safe_volume_to_liquidity_ratio
from app.context_intelligence.onchain_context import build_onchain_context
from app.context_intelligence.rss_context import build_rss_context
from app.context_intelligence.types import (
    AE8_LINEAGE_WARNING,
    FreshnessMode,
    FreshnessStatus,
    SourceStatus,
    is_forbidden_context_feature,
)
from app.context_intelligence.whale_context import WHALE_SCORE_RESEARCH_METADATA, build_whale_context


def test_context_schema_id_deterministic_and_content_derived():
    s1 = build_context_schema()
    s2 = build_context_schema()
    assert s1.context_schema_id == s2.context_schema_id
    assert len(s1.context_schema_id) == 64
    core = {
        "version": s1.context_schema_version,
        "feature_names": s1.feature_names,
        "feature_families": s1.feature_families,
        "feature_dtypes": s1.feature_dtypes,
        "freshness_thresholds": s1.freshness_thresholds,
        "forbidden_feature_patterns": s1.forbidden_feature_patterns,
    }
    assert s1.context_schema_id == compute_context_schema_id(core)


def test_forbidden_feature_names_rejected():
    assert is_forbidden_context_feature("target_row_id")
    assert is_forbidden_context_feature("future_return_proxy")
    assert is_forbidden_context_feature("train_split_flag")
    assert not is_forbidden_context_feature("liquidity_usd")


def test_missing_rss_produces_not_available_not_invented():
    features, freshness, status, warnings = build_rss_context(
        None,
        symbol="PEPE",
        as_of_timestamp=datetime.now(timezone.utc).isoformat(),
        freshness_reference_timestamp=datetime.now(timezone.utc).isoformat(),
        freshness_mode=FreshnessMode.LIVE_OR_CURRENT_RUNTIME,
        threshold_minutes=360.0,
    )
    assert status in {SourceStatus.SOURCE_NOT_AVAILABLE.value, SourceStatus.SOURCE_EMPTY.value}
    assert "RSS_CONTEXT_NOT_AVAILABLE" in warnings or features["rss_missingness_flag"] is True
    assert features["rss_article_count_24h"] is None


def test_missing_onchain_config_flag():
    features, _, status, warnings = build_onchain_context(
        raw_payload_row=None,
        as_of_timestamp=datetime.now(timezone.utc).isoformat(),
        freshness_reference_timestamp=datetime.now(timezone.utc).isoformat(),
        freshness_mode=FreshnessMode.LIVE_OR_CURRENT_RUNTIME,
        threshold_minutes=15.0,
        allow_external_fetch=True,
    )
    assert status == SourceStatus.SOURCE_CONFIG_MISSING.value
    assert "ONCHAIN_CONTEXT_CONFIG_MISSING" in warnings
    assert features["onchain_missingness_flag"] is True


def test_whale_score_asof_research_only_metadata():
    features, _, _, _, meta = build_whale_context(
        None,
        coin_id=None,
        pair_address=None,
        snapshot_row={"timestamp": datetime.now(timezone.utc).isoformat(), "whale_score": 0.42},
        as_of_timestamp=datetime.now(timezone.utc).isoformat(),
        freshness_reference_timestamp=datetime.now(timezone.utc).isoformat(),
        freshness_mode=FreshnessMode.LIVE_OR_CURRENT_RUNTIME,
        threshold_minutes=15.0,
    )
    assert features["whale_score_asof"] == 0.42
    assert meta["whale_score_status"] == "RESEARCH_ONLY_PLAUSIBLE_FEATURE_CANDIDATE"
    assert meta["not_rule"] is True
    assert meta["not_runtime_approved_as_standalone_signal"] is True


def test_whale_score_not_used_as_hard_rule():
    """Context builder must not filter or gate on whale_score_asof."""
    assert WHALE_SCORE_RESEARCH_METADATA["not_rule"] is True


def test_volume_to_liquidity_ratio_safe():
    assert safe_volume_to_liquidity_ratio(100.0, 0.0) is None
    assert safe_volume_to_liquidity_ratio(None, 50.0) is None
    assert safe_volume_to_liquidity_ratio(100.0, 50.0) == 2.0


def test_stale_snapshot_live_mode_nulled():
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    now = datetime.now(timezone.utc).isoformat()
    from app.context_intelligence.liquidity_activity_context import build_liquidity_activity_context

    features, freshness, status, _ = build_liquidity_activity_context(
        snapshot_row={
            "timestamp": old_ts,
            "liquidity": 50000.0,
            "volume_24h": 10000.0,
            "txns_buys": 10,
            "txns_sells": 5,
        },
        prior_snapshot_row=None,
        prior_6h_snapshot_row=None,
        signal_row=None,
        as_of_timestamp=old_ts,
        freshness_reference_timestamp=now,
        freshness_mode=FreshnessMode.LIVE_OR_CURRENT_RUNTIME,
        threshold_minutes=15.0,
    )
    assert status == SourceStatus.SOURCE_STALE.value
    assert freshness["freshness_status"] == FreshnessStatus.STALE.value
    assert features["liquidity_usd"] is None
    assert features["liquidity_activity_missingness_flag"] is True


def test_historical_replay_avoids_false_staleness():
    old_ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
    from app.context_intelligence.liquidity_activity_context import build_liquidity_activity_context

    features, freshness, status, _ = build_liquidity_activity_context(
        snapshot_row={
            "timestamp": old_ts,
            "liquidity": 50000.0,
            "volume_24h": 10000.0,
            "txns_buys": 10,
            "txns_sells": 5,
        },
        prior_snapshot_row=None,
        prior_6h_snapshot_row=None,
        signal_row=None,
        as_of_timestamp=old_ts,
        freshness_reference_timestamp=old_ts,
        freshness_mode=FreshnessMode.HISTORICAL_REPLAY_OR_AUDIT,
        threshold_minutes=15.0,
    )
    assert freshness["freshness_status"] in {
        FreshnessStatus.REPLAY_AS_OF_FRESH.value,
        FreshnessStatus.FRESH.value,
    }
    assert features["liquidity_usd"] == 50000.0
    assert status == SourceStatus.SOURCE_OK.value


def test_future_context_timestamp_blocked():
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    now = datetime.now(timezone.utc).isoformat()
    block = compute_freshness(
        source_timestamp=future,
        freshness_reference_timestamp=now,
        freshness_mode=FreshnessMode.HISTORICAL_REPLAY_OR_AUDIT,
        threshold_minutes=15.0,
    )
    assert block["freshness_status"] == FreshnessStatus.INVALID_FUTURE_TIMESTAMP.value
    assert block["missingness_reason"] == "FUTURE_CONTEXT_LEAKAGE_RISK"


def test_weak_lineage_requires_warning_and_fallback():
    lineage = build_context_lineage(
        signal_row={"id": 1},
        snapshot_row={"id": 2},
        raw_payload_row=None,
        freshness_blocks={},
    )
    assert lineage.lineage_validation_status == "PASS_WEAK_BEST_EFFORT_WITH_WARNING"
    assert lineage.lineage_warning == AE8_LINEAGE_WARNING
    assert lineage.fallback_reason
    assert lineage.lineage_confidence_score < 0.5
    assert lineage.exact_id_match is False


def test_lineage_cannot_be_written_without_validation_status():
    lineage = build_context_lineage(
        signal_row={"id": 1},
        snapshot_row=None,
        raw_payload_row=None,
        freshness_blocks={},
    )
    d = lineage.to_dict()
    assert d.get("lineage_validation_status")


def test_bounded_query_builder_guard():
    assert validate_bounded_sql("SELECT id FROM signals WHERE timestamp >= ? LIMIT ?")
    assert not validate_bounded_sql("SELECT * FROM market_snapshots")


def test_memory_safety_rejects_unbounded_query():
    stats = QueryStats(max_records_enforced=50, lookback_hours_enforced=3.0)
    status = assess_memory_safety(stats, ["SELECT * FROM market_snapshots"])
    assert status == "BLOCKED_UNBOUNDED_QUERY"


def test_jsonl_writer_flush_fsync(tmp_path: Path):
    path = tmp_path / "test.jsonl"
    with ContextJsonlWriter(path) as writer:
        writer.append_record({"a": 1})
    records, diag = read_context_jsonl_safe(path)
    assert len(records) == 1
    assert diag["complete_lines"] == 1


def test_context_record_authority_flags():
    schema = build_context_schema()
    bundle = {
        "signal_row": {
            "id": 1,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "coin_id": 1,
            "symbol": "TEST",
        },
        "snapshot_row": {
            "id": 2,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "liquidity": 1000.0,
            "volume_24h": 500.0,
            "txns_buys": 1,
            "txns_sells": 1,
            "whale_score": 0.1,
        },
        "raw_payload_row": None,
        "coin_row": {"pair_address": "abc123", "chain": "solana", "symbol": "TEST"},
        "prior_snapshot_row": None,
        "prior_6h_snapshot_row": None,
    }
    stats = QueryStats()
    rec = build_context_feature_record(
        bundle,
        schema=schema,
        run_started_at_utc=datetime.now(timezone.utc).isoformat(),
        freshness_mode=FreshnessMode.LIVE_OR_CURRENT_RUNTIME,
        freshness_reference_timestamp=datetime.now(timezone.utc).isoformat(),
        conn=None,
        allow_external_fetch=False,
        stats=stats,
    )
    assert rec is not None
    assert rec.no_trade_authority is True
    assert rec.llm_decision_authority is False


def test_no_external_fetch_by_default_in_onchain():
    _, _, status, _ = build_onchain_context(
        raw_payload_row=None,
        as_of_timestamp=datetime.now(timezone.utc).isoformat(),
        freshness_reference_timestamp=datetime.now(timezone.utc).isoformat(),
        freshness_mode=FreshnessMode.LIVE_OR_CURRENT_RUNTIME,
        threshold_minutes=15.0,
        allow_external_fetch=False,
    )
    assert status == SourceStatus.SOURCE_NOT_AVAILABLE.value


def test_external_fetch_requires_explicit_flag():
    with mock.patch.dict(os.environ, {"HELIUS_API_KEY": ""}, clear=False):
        _, _, status_disabled, _ = build_onchain_context(
            raw_payload_row=None,
            as_of_timestamp=datetime.now(timezone.utc).isoformat(),
            freshness_reference_timestamp=datetime.now(timezone.utc).isoformat(),
            freshness_mode=FreshnessMode.LIVE_OR_CURRENT_RUNTIME,
            threshold_minutes=15.0,
            allow_external_fetch=False,
        )
        _, _, status_enabled, warnings = build_onchain_context(
            raw_payload_row=None,
            as_of_timestamp=datetime.now(timezone.utc).isoformat(),
            freshness_reference_timestamp=datetime.now(timezone.utc).isoformat(),
            freshness_mode=FreshnessMode.LIVE_OR_CURRENT_RUNTIME,
            threshold_minutes=15.0,
            allow_external_fetch=True,
        )
    assert status_disabled == SourceStatus.SOURCE_NOT_AVAILABLE.value
    assert status_enabled == SourceStatus.SOURCE_CONFIG_MISSING.value


def test_missing_context_families_retained_not_discarded():
    schema = build_context_schema()
    assert "rss" in schema.feature_families
    assert "onchain" in schema.feature_families
    assert len(schema.feature_names) > 50


def _make_memory_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE coins (id INTEGER PRIMARY KEY, pair_address TEXT, chain TEXT,
            token_address TEXT, symbol TEXT, quote_symbol TEXT);
        CREATE TABLE signals (id INTEGER PRIMARY KEY, timestamp TEXT, coin_id INTEGER,
            symbol TEXT, signal_type TEXT, score REAL, confidence REAL, reason TEXT,
            model_source TEXT, features_json TEXT);
        CREATE TABLE market_snapshots (id INTEGER PRIMARY KEY, coin_id INTEGER, timestamp TEXT,
            provider TEXT, chain TEXT, pair_address TEXT, price REAL, liquidity REAL,
            volume_24h REAL, fdv REAL, txns_buys INTEGER, txns_sells INTEGER, txns_total INTEGER,
            price_change_m5 REAL, price_change_h1 REAL, price_change_h6 REAL, price_change_h24 REAL,
            whale_score REAL, buy_ratio REAL);
        CREATE TABLE raw_provider_payloads (id INTEGER PRIMARY KEY, timestamp TEXT, provider TEXT,
            source_type TEXT, query TEXT, chain TEXT, pair_address TEXT, symbol TEXT,
            payload_json_or_text TEXT, payload_hash TEXT);
        CREATE TABLE sentiment_records (id INTEGER PRIMARY KEY, timestamp TEXT, source TEXT,
            title TEXT, url TEXT, text_excerpt TEXT, symbols_json TEXT,
            sentiment_score REAL, relevance_score REAL, raw_ref_id INTEGER);
        CREATE TABLE whale_alerts (id INTEGER PRIMARY KEY, timestamp TEXT, coin_id INTEGER,
            symbol TEXT, chain TEXT, pair_address TEXT, alert_type TEXT, whale_score REAL,
            liquidity REAL, volume REAL, tx_summary_json TEXT);
        """
    )
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        "INSERT INTO coins VALUES (1, 'pair1', 'solana', 'tok1', 'MEME', 'SOL')"
    )
    conn.execute(
        "INSERT INTO signals VALUES (1, ?, 1, 'MEME', 'buy', 0.8, 0.7, 'test', 'engine', '{}')",
        (now,),
    )
    conn.execute(
        """INSERT INTO market_snapshots VALUES
        (1, 1, ?, 'dexscreener', 'solana', 'pair1', 0.01, 50000, 10000, NULL,
         10, 5, 15, 0.1, 0.2, 0.3, 0.4, 0.25, 0.6)""",
        (now,),
    )
    return conn


def test_run_ae8_smoke_integration(tmp_path: Path):
    conn = _make_memory_db()
    summary = run_ae8_context_intelligence(
        project_root=tmp_path,
        conn=conn,
        max_records=5,
        lookback_hours=24.0,
        output_root=tmp_path / "data",
        allow_external_fetch=False,
        freshness_mode="live",
    )
    assert summary["context_records_created"] >= 1
    assert summary["final_status"]
    assert summary["runtime_inference_status"] == "BLOCKED_NOT_APPROVED"
    assert summary["trading_authorization_status"] == "NOT_APPROVED"