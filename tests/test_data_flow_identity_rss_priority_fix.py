"""Tests for data-flow identity, RSS normalization, and collection priority fix."""
from __future__ import annotations

import csv
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

from app.clean_forward.price_source_identity import (
    build_price_source_key,
    extract_chain_and_pair_from_provider_url,
    is_internal_lineage_id,
    resolve_selected_target_identity,
)
from app.clean_forward.rss_article_normalization import (
    normalize_raw_rss_payload,
    parse_rss_payload_items,
)

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_data_flow_identity_rss_priority_fix.py"


def _load_runner():
    spec = importlib.util.spec_from_file_location("df_identity_runner", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_url_suffix_strips_trailing_slash_query_fragment():
    cases = [
        ("https://dexscreener.com/solana/ABC123/", "solana", "ABC123"),
        ("https://dexscreener.com/solana/ABC123?ref=x", "solana", "ABC123"),
        ("https://dexscreener.com/xrpl/ABC.DEF#section", "xrpl", "ABC.DEF"),
        (" https://dexscreener.com/base/0xAbC/ ", "base", "0xAbC"),
    ]
    for url, chain, pair in cases:
        got_chain, got_pair = extract_chain_and_pair_from_provider_url(url)
        assert got_chain == chain
        assert got_pair == pair
        assert "?" not in got_pair and "#" not in got_pair and not got_pair.endswith("/")


def test_url_suffix_beats_ae16b_internal_id():
    row = {
        "combined_target_id": "ae16b_deadbeefdeadbeef",
        "chain": "solana",
        "pair_address": "ae16b_deadbeefdeadbeef",
        "provider_pair_url": "https://dexscreener.com/solana/RealPairAddress123/",
    }
    resolved = resolve_selected_target_identity(row)
    assert resolved["display_real_pair_address"] == "RealPairAddress123"
    assert resolved["normalized_real_pair_address"] == "realpairaddress123"
    assert "ae16b_" not in resolved["price_source_key"]
    assert resolved["identity_resolution_status"] == "RESOLVED"


def test_ae16b_cannot_be_real_pair_address():
    assert is_internal_lineage_id("ae16b_14a6d70ca3ca4b95")
    row = {
        "combined_target_id": "ae16b_14a6d70ca3ca4b95",
        "chain": "solana",
        "pair_address": "ae16b_14a6d70ca3ca4b95",
    }
    resolved = resolve_selected_target_identity(row)
    assert resolved["identity_resolution_status"] == "UNRESOLVED_INTERNAL_ID_ONLY"
    assert resolved["display_real_pair_address"] == ""
    assert resolved["price_source_key"] == ""


def test_display_preserves_case_normalized_lowercases():
    row = {
        "combined_target_id": "ae16b_abc",
        "chain": "solana",
        "provider_pair_address": "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd",
        "provider_pair_url": "",
    }
    resolved = resolve_selected_target_identity(row)
    assert resolved["display_real_pair_address"] == "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd"
    assert resolved["normalized_real_pair_address"] == "2uf4xh61rdwxng9woyxsvqp7zua6klfpb3nvnrqeoisd"
    assert resolved["price_source_key"] == build_price_source_key(
        "dexscreener",
        "solana",
        "2uf4xh61rdwxng9woyxsvqp7zua6klfpb3nvnrqeoisd",
    )


def test_selected_target_count_is_dynamic_not_hardcoded(tmp_path: Path):
    csv_path = tmp_path / "selected.csv"
    rows = [
        {
            "combined_target_id": f"ae16b_{i:016x}",
            "chain": "solana",
            "provider_pair_address": f"Pair{i}AbCdEfGhIjKlMnOpQrStUvWxYz0123456789",
            "clean_forward_candidate_ready": "true",
            "acceptance_status": "ACTIVE",
        }
        for i in range(3)
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    runner = _load_runner()
    loaded = runner.load_selected_rows(csv_path)
    assert len(loaded) == 3
    # Must not assume 45
    assert len(loaded) != 45
    source = SCRIPT.read_text(encoding="utf-8")
    # No hard-coded selected universe size gate
    assert "selected_count == 45" not in source
    assert "assert len(selected" not in source.lower() or "45" not in source


def test_collection_priority_open_and_selected_before_discovery(tmp_path: Path):
    runner = _load_runner()
    resolved = [
        resolve_selected_target_identity(
            {
                "combined_target_id": "ae16b_sel1",
                "chain": "solana",
                "provider_pair_address": "SelectedPairAAAAABBBBBCCCCCDDDDDEEEEEFFFFF",
            }
        )
    ]
    open_positions = [
        {
            "status": "OPEN",
            "chain": "robinhood",
            "pair_address": "0x6Ac7857561c6aB70aad1aef504CC8B585E3fa6a1",
        }
    ]
    global_rows = [
        {
            "price_source_key": "dexscreener|solana|trendingpair00000000000000000000000001",
            "provider": "dexscreener",
            "display_chain": "solana",
            "display_real_pair_address": "TrendingPair00000000000000000000000001",
            "normalized_chain": "solana",
            "normalized_real_pair_address": "trendingpair00000000000000000000000001",
            "provider_pair_url": "",
            "L1_source_queries": "trending",
        }
    ]
    out = runner.part_c_collection_priority(
        resolved_rows=resolved,
        open_positions=open_positions,
        global_rows=global_rows,
        out_dir=tmp_path,
    )
    ranks = [r["priority_rank"] for r in out["plan_rows"]]
    assert ranks.index("0A") < ranks.index("2")
    assert ranks.index("0B") < ranks.index("2")
    legacy = [r for r in out["plan_rows"] if r["priority_rank"] == "0A"][0]
    assert legacy["open_position_status"] == "LEGACY_OR_OUT_OF_SELECTED_POSITION"
    assert legacy["collection_reason"] == "MARK_PRICE_ONLY"
    assert legacy["eligible_for_new_trade_candidate"] == "false"
    assert "does not promote" in legacy["notes"].lower()
    assert out["collection_priority_audit_pass"] is True


def test_rss_reconstruction_preserves_fetched_at_and_trace():
    xml = """<?xml version='1.0'?>
    <rss><channel>
      <item>
        <title>Solana ETF buzz</title>
        <link>https://example.com/a</link>
        <pubDate>Mon, 01 Jun 2026 18:33:41 +0000</pubDate>
        <description>Something about $SOL</description>
      </item>
    </channel></rss>
    """
    result = normalize_raw_rss_payload(
        raw_payload_id=42,
        provider="rss_cointelegraph",
        source_type="rss_feed",
        query="https://cointelegraph.com/rss",
        fetched_at="2026-06-01T19:40:16+00:00",
        payload_hash="abc",
        payload_text=xml,
    )
    assert result["t0"]["parse_status"] == "PARSED"
    assert len(result["traces"]) >= 1
    assert all(t["normalization_status"] == "PARSED" for t in result["traces"])
    assert result["articles"][0]["fetched_at"] == "2026-06-01T19:40:16+00:00"
    assert result["articles"][0]["raw_payload_id"] == "42"
    assert result["articles"][0]["article_id"]
    assert result["articles"][0]["article_hash"]


def test_every_raw_rss_payload_has_trace_including_parse_failed():
    bad = normalize_raw_rss_payload(
        raw_payload_id=7,
        provider="rss_decrypt",
        source_type="rss_feed",
        query="https://decrypt.co/feed",
        fetched_at="2026-07-01T00:00:00+00:00",
        payload_hash="badhash",
        payload_text="<not-valid-xml",
    )
    assert bad["t0"]["parse_status"] == "PARSE_FAILED"
    assert len(bad["traces"]) == 1
    assert bad["traces"][0]["normalization_status"] == "PARSE_FAILED"
    assert bad["traces"][0]["parse_error"]

    empty = normalize_raw_rss_payload(
        raw_payload_id=8,
        provider="rss_decrypt",
        source_type="rss_feed",
        query="https://decrypt.co/feed",
        fetched_at="2026-07-01T00:00:00+00:00",
        payload_hash="emptyhash",
        payload_text="<rss><channel></channel></rss>",
    )
    assert empty["t0"]["parse_status"] == "NO_ITEMS_EXTRACTED"
    assert empty["traces"][0]["normalization_status"] == "NO_ITEMS_EXTRACTED"


def test_t3_corpus_has_no_llm_output_fields(tmp_path: Path):
    runner = _load_runner()
    # Minimal in-memory sqlite with one good RSS payload
    db = tmp_path / "t.db"
    conn = sqlite3.connect(db)
    conn.execute(
        """
        CREATE TABLE raw_provider_payloads (
          id INTEGER PRIMARY KEY,
          timestamp TEXT,
          provider TEXT,
          source_type TEXT,
          query TEXT,
          chain TEXT,
          pair_address TEXT,
          symbol TEXT,
          payload_json_or_text TEXT,
          payload_hash TEXT
        )
        """
    )
    xml = """<?xml version='1.0'?><rss><channel>
      <item><title>Hello BTC</title><link>https://ex/a</link>
      <pubDate>Mon, 01 Jun 2026 18:33:41 +0000</pubDate>
      <description>Bitcoin moves</description></item>
    </channel></rss>"""
    conn.execute(
        "INSERT INTO raw_provider_payloads VALUES (1,?,?,?,?,?,?,?,?,?)",
        (
            "2026-06-01T19:40:16+00:00",
            "rss_cointelegraph",
            "rss_feed",
            "https://cointelegraph.com/rss",
            "",
            "",
            "",
            xml,
            "ph1",
        ),
    )
    conn.commit()
    info = runner.part_d_rss(conn, [], tmp_path / "rss")
    conn.close()
    assert info["llm_corpus_items"] >= 1
    forbidden = {"llm_summary", "llm_sentiment", "gemini_response", "openai_response", "qwen_response"}
    for item in info["corpus"]:
        assert not (forbidden & set(item.keys()))
        assert item["corpus_status"] == "READY_FOR_FUTURE_LLM"
    assert info["all_payloads_traced"] is True
    for art in info["articles"]:
        assert art["raw_payload_id"]
        assert art["payload_hash"]


def test_safety_flags_in_runner_source():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "llm_calls_made\": False" in source or "llm_calls_made': False" in source or '"llm_calls_made": False' in source
    assert "model_training_run" in source
    assert "backtest_run" in source
    assert "wallet_connected" in source
    assert "live_trading_enabled" in source
    assert "mode=ro" in source
    # Must not import LLM client SDKs in this runner
    assert "openai" not in source.lower().split("import")[0] or "openai" not in source
    assert "google.generativeai" not in source
    assert "from openai" not in source
    assert "import openai" not in source
