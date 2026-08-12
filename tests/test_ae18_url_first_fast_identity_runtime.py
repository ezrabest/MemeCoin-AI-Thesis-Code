"""AE18 URL-first identity + fast runtime index hot-path tests."""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from app.clean_forward.canonical_market_identity import (
    CANONICAL_IDENTITY_TYPE,
    build_index_row,
    extract_final_url_segment_exact,
    resolve_canonical_market_identity,
)
from app.clean_forward.runtime_identity_index import (
    INDEX_JSONL_PATH,
    INDEX_MISSING_CODE,
    load_runtime_identity_index,
    resolve_position_canonical_key,
    write_runtime_index,
)
from app.ae13b_product.runtime_market_feed import (
    build_clean_forward_from_index,
    build_live_market_from_index,
)
from app.execution.fill_price import build_market_price_maps
from app.execution.paper import PaperTrader

ROOT = Path(__file__).resolve().parents[1]
REBUILD_SCRIPT = ROOT / "scripts" / "rebuild_canonical_market_identity_index.py"


def _load_rebuild_runner():
    spec = importlib.util.spec_from_file_location("rebuild_index", REBUILD_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_url_is_canonical_identity():
    row = {
        "provider_pair_url": "https://dexscreener.com/solana/AbCdEfGh123/",
        "chain": "solana",
    }
    identity = resolve_canonical_market_identity(row)
    assert identity["canonical_market_identity"] == identity["provider_pair_url_exact"]
    assert identity["canonical_market_identity_type"] == CANONICAL_IDENTITY_TYPE
    assert identity["mark_price_lookup_key"] == identity["canonical_market_identity"]


def test_provider_pair_url_exact_case_preserved():
    url = "https://dexscreener.com/solana/2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd"
    row = {"provider_pair_url": url, "chain": "solana"}
    identity = resolve_canonical_market_identity(row)
    assert identity["provider_pair_url_exact"] == url
    assert "2uf4xh" not in identity["provider_pair_url_exact"]


def test_final_url_segment_exact_case_preserved():
    url = "https://dexscreener.com/base/0xAbC123"
    assert extract_final_url_segment_exact(url) == "0xAbC123"
    identity = resolve_canonical_market_identity({"provider_pair_url": url, "chain": "base"})
    assert identity["provider_pair_url_final_segment_exact"] == "0xAbC123"


def test_row_valid_with_empty_pair_address_when_url_exists():
    row = {
        "provider_pair_url": "https://dexscreener.com/solana/OnlyUrlIdentity123",
        "chain": "solana",
        "pair_address": "",
        "price_usd": 0.001,
    }
    identity = resolve_canonical_market_identity(row)
    assert identity["canonical_market_identity"]
    assert identity["pair_address_derived"] == "OnlyUrlIdentity123"
    index_row = build_index_row(row, last_identity_rebuild_at="2026-01-01T00:00:00+00:00")
    assert index_row["safe_for_price_lookup"] is True


def test_pair_address_not_canonical_market_identity():
    row = {
        "provider_pair_address": "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd",
        "chain": "solana",
    }
    identity = resolve_canonical_market_identity(row)
    assert identity["canonical_market_identity"].startswith("https://")
    assert identity["pair_address_derived"] != identity["canonical_market_identity"]


def test_clean_feed_loader_reads_only_runtime_index(tmp_path: Path, monkeypatch):
    rows = [
        build_index_row(
            {
                "provider_pair_url": "https://dexscreener.com/solana/HotPathRow1",
                "chain": "solana",
                "provider_base_token_symbol": "TEST",
                "provider_quote_token_symbol": "SOL",
                "price_usd": 1.23,
            },
            last_identity_rebuild_at="2026-07-27T12:00:00+00:00",
        )
    ]
    jsonl = tmp_path / "index.jsonl"
    csv = tmp_path / "index.csv"
    write_runtime_index(rows, jsonl_path=jsonl, csv_path=csv)

    monkeypatch.setattr(
        "app.clean_forward.runtime_identity_index.INDEX_JSONL_PATH",
        jsonl,
    )
    monkeypatch.setattr(
        "app.clean_forward.runtime_identity_index.INDEX_CSV_PATH",
        csv,
    )

    with patch("app.ae13b_product.dexscreener_pair_verify.validate_dexscreener_pair") as mock_verify:
        feed = build_clean_forward_from_index(limit=10)
        assert feed["ok"] is True
        assert feed["runtime_index_sourced"] is True
        assert feed["external_network_calls_on_load"] is False
        assert len(feed["rows"]) == 1
        mock_verify.assert_not_called()


def test_clean_feed_loader_does_not_scan_audit_roots(tmp_path: Path, monkeypatch):
    audit_root = tmp_path / "audits"
    audit_root.mkdir()
    (audit_root / "deep_audit.json").write_text('{"secret": true}', encoding="utf-8")

    rows = [
        build_index_row(
            {"provider_pair_url": "https://dexscreener.com/solana/NoScan1", "chain": "solana"},
            last_identity_rebuild_at="2026-07-27T12:00:00+00:00",
        )
    ]
    jsonl = tmp_path / "index.jsonl"
    csv = tmp_path / "index.csv"
    write_runtime_index(rows, jsonl_path=jsonl, csv_path=csv)
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_JSONL_PATH", jsonl)
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_CSV_PATH", csv)

    loaded = load_runtime_identity_index()
    assert loaded["recursive_audit_scan_used"] is False
    assert loaded["ok"] is True


def test_clean_feed_loader_no_dexscreener_helius_on_load(tmp_path: Path, monkeypatch):
    rows = [
        build_index_row(
            {"provider_pair_url": "https://dexscreener.com/solana/NetFree1", "chain": "solana"},
            last_identity_rebuild_at="2026-07-27T12:00:00+00:00",
        )
    ]
    jsonl = tmp_path / "index.jsonl"
    csv = tmp_path / "index.csv"
    write_runtime_index(rows, jsonl_path=jsonl, csv_path=csv)
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_JSONL_PATH", jsonl)
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_CSV_PATH", csv)

    loaded = load_runtime_identity_index()
    assert loaded["dexscreener_calls_on_load"] is False
    assert loaded["helius_calls_on_load"] is False
    assert loaded["external_network_calls_on_load"] is False


def test_portfolio_mark_price_uses_canonical_url_identity():
    trader = PaperTrader()
    url = "https://dexscreener.com/solana/CanonicalPriceKey123"
    trader.set_market_prices(
        [
            {
                "canonical_market_identity": url,
                "mark_price_lookup_key": url,
                "price_usd": 0.42,
            }
        ],
        price_timestamp="2026-07-27T12:00:00+00:00",
    )
    positions = [
        {
            "id": 1,
            "symbol": "TEST",
            "chain": "solana",
            "quantity": 100,
            "entry_price": 0.40,
            "size_usd": 40,
            "opened_at": "2026-07-27T11:00:00+00:00",
            "canonical_market_identity": url,
            "pair_address": "",
        }
    ]
    marked = trader.mark_positions_to_market(positions)
    assert marked[0]["current_price"] == 0.42
    assert marked[0]["current_price_source"] == "market_canonical_url"
    assert marked[0]["mark_price_lookup_status"] == "PRICE_AVAILABLE"


def test_no_mark_price_not_only_because_missing_pair_address():
    trader = PaperTrader()
    url = "https://dexscreener.com/solana/UrlOnlyNoPair"
    trader.set_market_prices([], price_timestamp="2026-07-27T12:00:00+00:00")
    marked = trader.mark_positions_to_market(
        [
            {
                "id": 2,
                "symbol": "TEST",
                "chain": "solana",
                "quantity": 1,
                "entry_price": 1.0,
                "size_usd": 1,
                "opened_at": "2026-07-27T11:00:00+00:00",
                "canonical_market_identity": url,
                "pair_address": "",
            }
        ]
    )
    assert marked[0]["current_price"] is None
    assert marked[0]["mark_price_lookup_status"] == "PRICE_NOT_AVAILABLE"
    assert marked[0]["price_resolution_failure_reason"] == "no_index_price_for_canonical_url"
    assert "missing pair" not in (marked[0].get("mark_price_unavailable_reason") or "").lower()


def test_pair_address_never_primary_lookup_key_in_price_maps():
    url = "https://dexscreener.com/solana/PrimaryUrlKey"
    by_pair, by_coin, by_canonical = build_market_price_maps(
        [
            {
                "canonical_market_identity": url,
                "mark_price_lookup_key": url,
                "pair_address": "DerivedPairOnly",
                "price_usd": 9.99,
            }
        ]
    )
    assert url in by_canonical
    assert by_canonical[url] == 9.99
    # pair map may exist as secondary alias but canonical is authoritative for URL-first
    trader = PaperTrader()
    trader.set_market_prices(
        [{"canonical_market_identity": url, "price_usd": 9.99}],
        price_timestamp="2026-07-27T12:00:00+00:00",
    )
    marked = trader.mark_positions_to_market(
        [{"id": 3, "canonical_market_identity": url, "pair_address": "", "entry_price": 1, "quantity": 1, "opened_at": "2026-07-27T11:00:00+00:00"}]
    )
    assert marked[0]["current_price_source"] == "market_canonical_url"


def test_runtime_index_no_duplicate_canonical_identity(tmp_path: Path):
    url = "https://dexscreener.com/solana/DupeTest"
    rows = [
        build_index_row({"provider_pair_url": url, "chain": "solana"}, last_identity_rebuild_at="2026-01-01T00:00:00+00:00"),
        build_index_row({"provider_pair_url": url, "chain": "solana"}, last_identity_rebuild_at="2026-01-01T00:00:00+00:00"),
    ]
    rebuild = _load_rebuild_runner()
    deduped = rebuild._dedupe_by_canonical(rows)
    assert len(deduped) == 1


def test_stale_index_warning_does_not_block_ui(tmp_path: Path, monkeypatch):
    old_ts = "2020-01-01T00:00:00+00:00"
    rows = [
        build_index_row(
            {"provider_pair_url": "https://dexscreener.com/solana/StaleRow1", "chain": "solana", "price_usd": 1.0},
            last_identity_rebuild_at=old_ts,
        )
    ]
    jsonl = tmp_path / "index.jsonl"
    csv = tmp_path / "index.csv"
    write_runtime_index(rows, jsonl_path=jsonl, csv_path=csv)
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_JSONL_PATH", jsonl)
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_CSV_PATH", csv)

    feed = build_clean_forward_from_index()
    assert feed["ok"] is True
    assert feed["stale_warning"] is True
    assert len(feed["rows"]) == 1


def test_index_missing_shows_clear_message(tmp_path: Path, monkeypatch):
    missing_jsonl = tmp_path / "missing.jsonl"
    missing_csv = tmp_path / "missing.csv"
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_JSONL_PATH", missing_jsonl)
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_CSV_PATH", missing_csv)

    feed = build_clean_forward_from_index()
    assert feed["ok"] is False
    assert feed["error_code"] == INDEX_MISSING_CODE


def test_load_time_under_500ms_for_current_scale(tmp_path: Path, monkeypatch):
    seed = ROOT / "data" / "SeedTargets" / "clean_forward_curated_ready_targets_active.csv"
    if not seed.exists():
        pytest.skip("seed CSV not available")
    rebuild = _load_rebuild_runner()
    rows = rebuild.build_index_from_sources(
        seed_csv=seed,
        max_rows=None,
        allow_dexscreener=False,
        allow_helius=False,
    )
    if isinstance(rows, tuple):
        rows = rows[0]
    jsonl = tmp_path / "perf.jsonl"
    csv = tmp_path / "perf.csv"
    write_runtime_index(rows, jsonl_path=jsonl, csv_path=csv)
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_JSONL_PATH", jsonl)
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_CSV_PATH", csv)

    start = time.perf_counter()
    loaded = load_runtime_identity_index()
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert loaded["ok"] is True
    assert elapsed_ms < 500, f"load took {elapsed_ms:.1f}ms"


def test_resolve_position_canonical_key_from_index():
    url = "https://dexscreener.com/ethereum/0xA43fe16908251ee70EF74718545e4FE6C5cCEc9f"
    index_rows = [
        {
            "canonical_market_identity": url,
            "provider_pair_url_exact": url,
            "pair_address_derived": "0xA43fe16908251ee70EF74718545e4FE6C5cCEc9f",
            "chain": "ethereum",
        }
    ]
    pos = {"pair_address": "0xA43fe16908251ee70EF74718545e4FE6C5cCEc9f", "chain": "ethereum"}
    assert resolve_position_canonical_key(pos, index_rows) == url


def test_symbol_only_join_forbidden_in_rebuild_audits(tmp_path: Path, monkeypatch):
    rebuild = _load_rebuild_runner()
    monkeypatch.setattr(rebuild, "AUDITS_DIR", tmp_path)
    rows = [
        build_index_row(
            {"provider_pair_url": "https://dexscreener.com/solana/SymOnlyGuard", "chain": "solana"},
            last_identity_rebuild_at="2026-07-27T12:00:00+00:00",
        )
    ]
    audit = rebuild.write_url_first_identity_audit(rows)
    assert audit["symbol_only_join_attempt_count"] == 0
    assert (tmp_path / "ae18_url_first_identity_audit.json").exists()
