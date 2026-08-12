"""AE18 URL-first display + GET isolation + refresh + shutdown tests."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.clean_forward.canonical_market_identity import (
    CANONICAL_IDENTITY_TYPE,
    build_index_row,
    build_symbol_pair_display,
    resolve_canonical_market_identity,
)
from app.clean_forward.runtime_identity_index import (
    INDEX_MISSING_CODE,
    load_runtime_identity_index,
    write_runtime_index,
)
from app.ae13b_product.runtime_market_feed import (
    build_clean_forward_from_index,
    build_live_market_from_index,
    repair_legacy_position_identity,
)
from app.execution.paper import PaperTrader
from app.runtime.shutdown import (
    CONTROLLED_SHUTDOWN_SKIP,
    MIN_SCAN_INTERVAL_SECONDS,
    clamp_scan_interval_seconds,
    is_shutting_down,
    request_shutdown,
    reset_shutdown_for_tests,
    should_skip_network,
)
from app.runtime.ui_get_network_guard import (
    is_ui_get_path_active,
    record_network_attempt,
    reset_counters_for_tests,
    snapshot_counters,
    ui_get_network_guard,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset_runtime_flags():
    reset_shutdown_for_tests()
    reset_counters_for_tests()
    yield
    reset_shutdown_for_tests()
    reset_counters_for_tests()


def _sample_source(**overrides):
    row = {
        "provider_pair_url": "https://dexscreener.com/solana/AbCdEfGhIjKlMnOp",
        "chain": "solana",
        "provider_dex_id": "raydium",
        "provider_base_token_symbol": "PEPE",
        "provider_quote_token_symbol": "SOL",
        "provider_base_token_address": "TokenBase1111111111111111111111111111111",
        "provider_quote_token_address": "So11111111111111111111111111111111111111112",
        "price_usd": "0.001",
        "liquidity_usd": "50000",
        "volume_h24": "12000",
        "txns_h24_buys": "10",
        "txns_h24_sells": "8",
        "price_change_m5": "1.5",
        "price_change_h1": "-0.2",
        "price_change_h6": "3.0",
        "price_change_h24": "5.5",
        "acceptance_status": "PROVIDER_PAIR_RESOLVED",
    }
    row.update(overrides)
    return row


def test_runtime_index_keeps_display_fields():
    row = build_index_row(_sample_source(), last_identity_rebuild_at="2026-07-27T12:00:00+00:00")
    assert row["symbol_pair_display"] == "PEPE/SOL"
    assert row["provider_dex_id"] == "raydium"
    assert row["dex_id"] == "raydium"
    assert row["price_change_m5"] is not None
    assert row["price_change_h1"] is not None
    assert row["price_change_h6"] is not None
    assert row["price_change_h24"] is not None
    assert row["open_chart_url"] == row["provider_pair_url_exact"]
    assert row["canonical_market_identity_type"] == CANONICAL_IDENTITY_TYPE


def test_symbol_pair_not_dash_when_provider_symbols_exist():
    display = build_symbol_pair_display(_sample_source())
    assert display == "PEPE/SOL"
    assert display != "-"


def test_dex_not_dash_when_provider_dex_exists(tmp_path, monkeypatch):
    rows = [build_index_row(_sample_source(), last_identity_rebuild_at="2026-07-27T12:00:00+00:00")]
    jsonl = tmp_path / "i.jsonl"
    csv = tmp_path / "i.csv"
    write_runtime_index(rows, jsonl_path=jsonl, csv_path=csv)
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_JSONL_PATH", jsonl)
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_CSV_PATH", csv)
    feed = build_clean_forward_from_index()
    assert feed["rows"][0]["dex"] == "raydium"
    assert feed["rows"][0]["dex"] != "-"


def test_deltas_populated_when_source_has_price_change(tmp_path, monkeypatch):
    rows = [build_index_row(_sample_source(), last_identity_rebuild_at="2026-07-27T12:00:00+00:00")]
    jsonl = tmp_path / "i.jsonl"
    csv = tmp_path / "i.csv"
    write_runtime_index(rows, jsonl_path=jsonl, csv_path=csv)
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_JSONL_PATH", jsonl)
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_CSV_PATH", csv)
    live = build_live_market_from_index()
    r = live["rows"][0]
    assert r["price_change_5m"] is not None
    assert r["price_change_1h"] is not None
    assert r["price_change_6h"] is not None
    assert r["price_change_24h"] is not None


def test_clean_feed_cards_not_dash_when_index_exists(tmp_path, monkeypatch):
    rows = [build_index_row(_sample_source(), last_identity_rebuild_at="2026-07-27T12:00:00+00:00")]
    jsonl = tmp_path / "i.jsonl"
    csv = tmp_path / "i.csv"
    write_runtime_index(rows, jsonl_path=jsonl, csv_path=csv)
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_JSONL_PATH", jsonl)
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_CSV_PATH", csv)
    feed = build_clean_forward_from_index()
    st = feed["stats"]
    for key in (
        "total_candidates_seen",
        "valid_provider_pairs",
        "unique_base_tokens",
        "unique_canonical_markets",
        "duplicate_pools_suppressed",
        "invalid_or_unresolved_addresses",
        "clean_rows_displayed",
    ):
        assert st.get(key) is not None
        assert st.get(key) != "-"


def test_market_url_opens_provider_pair_url_exact():
    row = build_index_row(_sample_source(), last_identity_rebuild_at="2026-07-27T12:00:00+00:00")
    assert row["open_chart_url"] == "https://dexscreener.com/solana/AbCdEfGhIjKlMnOp"
    assert row["open_chart_url"] == row["provider_pair_url_exact"]


def test_pair_address_not_canonical():
    identity = resolve_canonical_market_identity(_sample_source())
    assert identity["canonical_market_identity"].startswith("https://")
    assert identity["pair_address_derived"] != identity["canonical_market_identity"]


def test_portfolio_mark_uses_canonical_url():
    trader = PaperTrader()
    url = "https://dexscreener.com/solana/AbCdEfGhIjKlMnOp"
    trader.set_market_prices(
        [{"canonical_market_identity": url, "price_usd": 0.5}],
        price_timestamp="2026-07-27T12:00:00+00:00",
    )
    marked = trader.mark_positions_to_market(
        [{
            "id": 1,
            "canonical_market_identity": url,
            "pair_address": "",
            "entry_price": 0.4,
            "quantity": 1,
            "opened_at": "2026-07-27T11:00:00+00:00",
        }]
    )
    assert marked[0]["current_price"] == 0.5
    assert marked[0]["current_price_source"] == "market_canonical_url"


def test_legacy_position_repair_needed():
    repaired = repair_legacy_position_identity(
        {"id": 9, "pair_address": "UnknownPairNotInIndex", "chain": "solana"},
        [],
    )
    assert repaired["mark_price_lookup_status"] == "LEGACY_POSITION_IDENTITY_REPAIR_NEEDED"


def test_get_reads_index_only_no_dexscreener(tmp_path, monkeypatch):
    rows = [build_index_row(_sample_source(), last_identity_rebuild_at="2026-07-27T12:00:00+00:00")]
    jsonl = tmp_path / "i.jsonl"
    csv = tmp_path / "i.csv"
    write_runtime_index(rows, jsonl_path=jsonl, csv_path=csv)
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_JSONL_PATH", jsonl)
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_CSV_PATH", csv)
    with ui_get_network_guard("/api/ae13b/clean-forward-market-feed"):
        with patch("app.ae13b_product.dexscreener_pair_verify.validate_dexscreener_pair") as mock_v:
            feed = build_clean_forward_from_index()
            live = build_live_market_from_index()
            assert feed["ok"] and live["ok"]
            mock_v.assert_not_called()
    snap = snapshot_counters()
    assert snap["dexscreener_calls_on_get"] == 0
    assert snap["external_network_calls_on_get"] == 0


def test_get_missing_index_no_pair_fallback(tmp_path, monkeypatch):
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_JSONL_PATH", tmp_path / "missing.jsonl")
    monkeypatch.setattr("app.clean_forward.runtime_identity_index.INDEX_CSV_PATH", tmp_path / "missing.csv")
    feed = build_clean_forward_from_index()
    assert feed["ok"] is False
    assert feed["error_code"] == INDEX_MISSING_CODE


def test_manual_refresh_is_post_and_url_first():
    from app.ae13b_product.manual_refresh_runtime_index import manual_refresh_runtime_index

    # Shutdown path must not call network
    request_shutdown(reason="test")
    meta = manual_refresh_runtime_index(allow_dexscreener=True, max_rows=1)
    assert meta["provider_refresh_cancelled"] is True
    assert meta["canonical_identity_type"] == "PROVIDER_URL"


def test_manual_refresh_preserves_url_case(tmp_path, monkeypatch):
    from app.ae13b_product.manual_refresh_runtime_index import _merge_verify_into_row

    src = {
        "provider_pair_url": "https://dexscreener.com/solana/AbCdEfGhIjKlMnOp",
        "chain": "solana",
    }
    verified = {"dex_id": "raydium", "base_token_symbol": "X", "quote_token_symbol": "Y", "price_usd": 1}
    merged = _merge_verify_into_row(src, verified)
    assert "AbCdEfGhIjKlMnOp" in merged["provider_pair_url"]
    assert "abcdefghijklmnop" not in merged["provider_pair_url"]


def test_shutdown_blocks_provider_calls():
    assert should_skip_network(context="before") is False
    request_shutdown(reason="test_block")
    assert is_shutting_down()
    assert should_skip_network(context="after") is True


def test_provider_wrapper_blocks_after_shutdown():
    from app.ae13b_product.dexscreener_pair_verify import _http_get_pair

    request_shutdown(reason="test_http")
    result = _http_get_pair("solana", "AbCdEf")
    assert result["ok"] is False
    assert CONTROLLED_SHUTDOWN_SKIP in str(result.get("error"))


def test_scan_interval_zero_clamped():
    assert clamp_scan_interval_seconds(0) >= MIN_SCAN_INTERVAL_SECONDS
    assert clamp_scan_interval_seconds(-5) >= MIN_SCAN_INTERVAL_SECONDS
    assert clamp_scan_interval_seconds(1) >= MIN_SCAN_INTERVAL_SECONDS


def test_ui_get_guard_records_forbidden_network():
    with ui_get_network_guard("/test"):
        assert is_ui_get_path_active()
        record_network_attempt("dexscreener")
    snap = snapshot_counters()
    assert snap["dexscreener_calls_on_get"] == 1
    assert snap["external_network_calls_on_get"] == 1


def test_atomic_index_write(tmp_path):
    rows = [build_index_row(_sample_source(), last_identity_rebuild_at="2026-07-27T12:00:00+00:00")]
    jsonl = tmp_path / "final.jsonl"
    csv = tmp_path / "final.csv"
    write_runtime_index(rows, jsonl_path=jsonl, csv_path=csv, atomic=True)
    assert jsonl.exists() and csv.exists()
    loaded = json.loads(jsonl.read_text(encoding="utf-8").strip().splitlines()[0])
    assert loaded["canonical_market_identity"].startswith("https://")


def test_symbol_address_fallback_display_only():
    """Addresses are never the primary SYMBOL/PAIR — only a details-column fallback."""
    from app.clean_forward.display_identity import (
        SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING,
        derive_symbol_pair_display,
    )

    row = {
        "provider_pair_url": "https://dexscreener.com/solana/OnlyAddrFallback",
        "chain": "solana",
        "provider_base_token_address": "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456",
        "provider_quote_token_address": "So11111111111111111111111111111111111111112",
    }
    display = build_symbol_pair_display(row)
    assert display == SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING
    derived = derive_symbol_pair_display(row)
    assert derived["symbol_pair_display_reason"]
    assert "/" in derived["symbol_pair_address_fallback"]
    identity = resolve_canonical_market_identity(row)
    assert identity["canonical_market_identity"].startswith("https://")
