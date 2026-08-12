"""AE13L — backend refresh change counter fix."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset_refresh_state():
    from app.ae13b_product.clean_forward_market_feed import reset_clean_forward_refresh_baseline
    from app.ae13b_product.dexscreener_pair_verify import get_pair_verify_limiter

    reset_clean_forward_refresh_baseline()
    lim = get_pair_verify_limiter()
    lim.clear_cache()
    lim.reset_stats()
    yield
    reset_clean_forward_refresh_baseline()
    lim.clear_cache()
    lim.reset_stats()


def _sample_pair(**kwargs: Any) -> dict[str, Any]:
    base = {
        "chainId": "solana",
        "dexId": "pumpswap",
        "pairAddress": "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd",
        "url": "https://dexscreener.com/solana/2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd",
        "baseToken": {
            "address": "pumpCmXqMfrsAkQ5r49WcJnRayYRqmXz6ae8H7hCNwe",
            "symbol": "PUMP",
            "name": "PUMP",
        },
        "quoteToken": {
            "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
            "symbol": "USDC",
            "name": "USD Coin",
        },
        "priceUsd": "0.01",
        "liquidity": {"usd": 100000.0},
        "volume": {"m5": 1.0, "h1": 10.0, "h6": 50.0, "h24": 100.0},
        "txns": {"h24": {"buys": 50, "sells": 40}},
        "priceChange": {"m5": 0.1, "h1": 1.0, "h6": 2.0, "h24": 3.0},
    }
    base.update(kwargs)
    return base


def _verified_dict(pair: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    base = {
        "lookup_ok": True,
        "clean_feed_eligible": True,
        "verification_status": "provider_pair_verified",
        "status": "provider_pair_verified",
        "tradability_status": "provider_pair_verified_display_only",
        "identity_status": "pair_and_tokens_separated",
        "normalized_chain_id": "solana",
        "chain_id": "solana",
        "pair_address": pair["pairAddress"],
        "provider_pair_id": pair["pairAddress"],
        "provider_pair_url": pair["url"],
        "provider_pair_url_source": "provider_returned_url",
        "dex_id": "pumpswap",
        "base_token_address": pair["baseToken"]["address"],
        "base_token_symbol": "PUMP",
        "quote_token_address": pair["quoteToken"]["address"],
        "quote_token_symbol": "USDC",
        "price_usd": pair["priceUsd"],
        "liquidity_usd": 100000,
        "volume_24h": 100,
        "txns_24h_buys": 50,
        "txns_24h_sells": 40,
        "txns_24h": {"buys": 50, "sells": 40, "total": 90},
        "price_change_5m": 0.1,
        "price_change_1h": 1.0,
        "price_change_6h": 2.0,
        "price_change_24h": 3.0,
        "provider_payload_hash": "hash1",
        "payload_hash": "hash1",
        "fetched_at": "2026-07-21T00:00:00+00:00",
        "ingested_at": "2026-07-21T00:00:00+00:00",
        "observed_at": "2026-07-21T00:00:00+00:00",
        "freshness_status": "fresh",
        "address_role": "pool_address",
        "pair_label": "PUMP/USDC",
        "verification_cache_hit": False,
        "exclusion_reason": None,
    }
    base.update(overrides)
    return base


def _search_from_pair(pair: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "chainId": "solana",
            "pairAddress": pair["pairAddress"],
            "priceUsd": pair["priceUsd"],
            "liquidity": {"usd": 100000},
            "volume": {"h24": 100},
            "baseToken": pair["baseToken"],
            "quoteToken": pair["quoteToken"],
        }
    ]


def test_01_provider_values_changed_when_price_changes():
    from app.ae13b_product.clean_forward_market_feed import (
        _compute_refresh_change_counters,
        _row_from_verification,
    )

    prev = _row_from_verification(_verified_dict(_sample_pair()))
    curr = _row_from_verification(
        _verified_dict(_sample_pair(), price_usd="0.02", provider_payload_hash="hash2")
    )
    out = _compute_refresh_change_counters(
        [curr],
        previous_rows=[prev],
        provider_refetch_completed=True,
    )
    assert out["provider_values_changed_count"] > 0


def test_02_payload_hash_changed_count():
    from app.ae13b_product.clean_forward_market_feed import (
        _compute_refresh_change_counters,
        _row_from_verification,
    )

    prev = _row_from_verification(_verified_dict(_sample_pair()))
    curr = _row_from_verification(
        _verified_dict(_sample_pair(), provider_payload_hash="hash2", payload_hash="hash2")
    )
    out = _compute_refresh_change_counters(
        [curr],
        previous_rows=[prev],
        provider_refetch_completed=True,
    )
    assert out["payload_hash_changed_count"] > 0


def test_03_provider_unchanged_but_refetched_count():
    from app.ae13b_product.clean_forward_market_feed import (
        _compute_refresh_change_counters,
        _row_from_verification,
    )

    prev = _row_from_verification(_verified_dict(_sample_pair()))
    curr = _row_from_verification(_verified_dict(_sample_pair()))
    out = _compute_refresh_change_counters(
        [curr],
        previous_rows=[prev],
        provider_refetch_completed=True,
    )
    assert out["provider_unchanged_but_refetched_count"] > 0
    assert out["provider_values_changed_count"] == 0
    assert out["payload_hash_changed_count"] == 0


def test_04_first_poll_no_baseline():
    from app.ae13b_product.clean_forward_market_feed import refresh_clean_forward_market_feed

    pair = _sample_pair()
    with patch(
        "app.ae13b_product.clean_forward_market_feed.get_trending_pairs_sync",
        return_value=_search_from_pair(pair),
    ), patch(
        "app.ae13b_product.clean_forward_market_feed.verify_provider_pair",
        return_value={**_verified_dict(pair), "verification_cache_hit": False},
    ):
        out = refresh_clean_forward_market_feed(force=True, max_verify=5, max_candidates=5)
    meta = out["refresh"]
    assert meta["comparison_baseline_available"] is False
    assert meta["provider_values_changed_count"] == 0
    assert meta["payload_hash_changed_count"] == 0
    assert meta["provider_unchanged_but_refetched_count"] == 0


def test_05_second_poll_has_baseline_and_counts():
    from app.ae13b_product.clean_forward_market_feed import refresh_clean_forward_market_feed

    pair = _sample_pair()
    verified_v1 = {**_verified_dict(pair), "verification_cache_hit": False}
    verified_v2 = {
        **_verified_dict(pair, price_usd="0.02", provider_payload_hash="hash2"),
        "verification_cache_hit": False,
    }

    with patch(
        "app.ae13b_product.clean_forward_market_feed.get_trending_pairs_sync",
        return_value=_search_from_pair(pair),
    ), patch(
        "app.ae13b_product.clean_forward_market_feed.verify_provider_pair",
        side_effect=[verified_v1, verified_v2],
    ):
        first = refresh_clean_forward_market_feed(force=True, max_verify=5, max_candidates=5)
        second = refresh_clean_forward_market_feed(force=True, max_verify=5, max_candidates=5)

    assert first["refresh"]["comparison_baseline_available"] is False
    meta = second["refresh"]
    assert meta["comparison_baseline_available"] is True
    assert meta["provider_values_changed_count"] > 0


def test_06_stable_key_uses_chain_and_pair_address():
    from app.ae13b_product.clean_forward_market_feed import (
        _compute_refresh_change_counters,
        _row_from_verification,
    )

    prev = _row_from_verification(
        _verified_dict(_sample_pair(), normalized_chain_id="solana", pair_address="PairA")
    )
    curr = _row_from_verification(
        _verified_dict(
            _sample_pair(pairAddress="PairA"),
            normalized_chain_id="solana",
            pair_address="PairA",
            price_usd="0.99",
        )
    )
    out = _compute_refresh_change_counters(
        [curr],
        previous_rows=[prev],
        provider_refetch_completed=True,
    )
    assert out["provider_values_changed_count"] == 1


def test_07_rows_entered_and_exited_main_feed():
    from app.ae13b_product.clean_forward_market_feed import (
        _compute_refresh_change_counters,
        _row_from_verification,
    )

    prev_a = _row_from_verification(
        _verified_dict(_sample_pair(pairAddress="PairA"), pair_address="PairA")
    )
    prev_b = _row_from_verification(
        _verified_dict(_sample_pair(pairAddress="PairB"), pair_address="PairB")
    )
    curr_b = _row_from_verification(
        _verified_dict(_sample_pair(pairAddress="PairB"), pair_address="PairB")
    )
    curr_c = _row_from_verification(
        _verified_dict(_sample_pair(pairAddress="PairC"), pair_address="PairC")
    )
    out = _compute_refresh_change_counters(
        [curr_b, curr_c],
        previous_rows=[prev_a, prev_b],
        provider_refetch_completed=True,
    )
    assert out["rows_entered_main_feed"] == 1
    assert out["rows_exited_main_feed"] == 1


def test_08_no_old_data_touched_scope_safety():
    from fastapi.testclient import TestClient

    from app.api import app

    with patch(
        "app.ae13b_product.clean_forward_market_feed.get_trending_pairs_sync",
        return_value=[],
    ):
        client = TestClient(app)
        r = client.post("/api/clean-forward-feed/refresh", json={"force": True})
    data = r.json()
    assert data.get("old_data_touched") is False
    assert data.get("training_run") is False
    assert data.get("backtest_run") is False
    assert data.get("ae14_run") is False
    assert data.get("paper_positions_opened_from_clean_feed") == 0
    assert data.get("live_trading_enabled") is False

    for p in (
        ROOT / "app" / "ae13b_product" / "clean_forward_market_feed.py",
        ROOT / "app" / "ae13b_product" / "dexscreener_pair_verify.py",
    ):
        text = p.read_text(encoding="utf-8").lower()
        assert "wallet_private_key" not in text
        assert "run_ae14(" not in text
