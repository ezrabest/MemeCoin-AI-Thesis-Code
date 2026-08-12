"""AE13K UI Refresh Addendum — refresh semantics, metadata, rate-limit, visual polish."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset_limiter():
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


def _verified_dict(pair: dict[str, Any]) -> dict[str, Any]:
    return {
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


def test_01_refresh_endpoint_exists_and_returns_metadata():
    from fastapi.testclient import TestClient

    from app.api import app

    pair = _sample_pair()
    search = [
        {
            "chainId": "solana",
            "pairAddress": pair["pairAddress"],
            "priceUsd": "0.01",
            "liquidity": {"usd": 100000},
            "volume": {"h24": 10},
            "baseToken": pair["baseToken"],
            "quoteToken": pair["quoteToken"],
        }
    ]

    with patch(
        "app.ae13b_product.clean_forward_market_feed.get_trending_pairs_sync",
        return_value=search,
    ), patch(
        "app.ae13b_product.clean_forward_market_feed.verify_provider_pair",
        return_value=_verified_dict(pair),
    ):
        client = TestClient(app)
        r = client.post(
            "/api/clean-forward-feed/refresh",
            json={"force": True, "clear_cache": False, "limit": 10, "max_verify": 5},
        )
    assert r.status_code == 200
    data = r.json()
    assert data.get("ok") is True
    meta = data.get("refresh") or data.get("refresh_metadata")
    assert meta
    for key in (
        "refresh_mode",
        "provider_refetch_attempted",
        "provider_refetch_completed",
        "cache_hit_count",
        "cache_miss_count",
        "cache_ttl_seconds",
        "payload_hash_changed_count",
        "provider_values_changed_count",
        "rendered_at",
    ):
        assert key in meta, f"missing {key}"


def test_02_refresh_uses_verifier_not_snapshot_feed():
    from app.ae13b_product.clean_forward_market_feed import refresh_clean_forward_market_feed

    calls = {"verify": 0}

    def fake_verify(**_k):
        calls["verify"] += 1
        return {
            **_verified_dict(_sample_pair()),
            "verification_cache_hit": False,
        }

    with patch(
        "app.ae13b_product.clean_forward_market_feed.get_trending_pairs_sync",
        return_value=[],
    ), patch(
        "app.ae13b_product.clean_forward_market_feed.verify_provider_pair",
        fake_verify,
    ):
        refresh_clean_forward_market_feed(force=True, max_verify=1, max_candidates=1)
    # trending still called; verify only if candidates — empty search means 0 verify
    assert calls["verify"] == 0


def test_04_refresh_not_ui_rerender_only_mode():
    from app.ae13b_product.clean_forward_market_feed import refresh_clean_forward_market_feed

    pair = _sample_pair()
    search = [{"chainId": "solana", "pairAddress": pair["pairAddress"], "priceUsd": "0.01",
               "liquidity": {"usd": 1}, "volume": {"h24": 1},
               "baseToken": pair["baseToken"], "quoteToken": pair["quoteToken"]}]

    with patch(
        "app.ae13b_product.clean_forward_market_feed.get_trending_pairs_sync",
        return_value=search,
    ), patch(
        "app.ae13b_product.clean_forward_market_feed.verify_provider_pair",
        return_value={**_verified_dict(pair), "verification_cache_hit": False},
    ):
        out = refresh_clean_forward_market_feed(force=True, max_verify=5, max_candidates=5)
    meta = out["refresh"]
    assert meta["refresh_mode"] != "ui_rerender_only"
    assert meta["provider_refetch_attempted"] is True


def test_05_force_provider_refresh_supported():
    from fastapi.testclient import TestClient

    from app.api import app

    pair = _sample_pair()
    search = [{"chainId": "solana", "pairAddress": pair["pairAddress"], "priceUsd": "0.01",
               "liquidity": {"usd": 1}, "volume": {"h24": 1},
               "baseToken": pair["baseToken"], "quoteToken": pair["quoteToken"]}]

    with patch(
        "app.ae13b_product.clean_forward_market_feed.get_trending_pairs_sync",
        return_value=search,
    ), patch(
        "app.ae13b_product.clean_forward_market_feed.verify_provider_pair",
        return_value={**_verified_dict(pair), "verification_cache_hit": False},
    ):
        client = TestClient(app)
        r = client.post(
            "/api/clean-forward-feed/refresh",
            json={"force": True, "clear_cache": True},
        )
    assert r.status_code == 200
    meta = r.json().get("refresh") or {}
    assert meta.get("clear_cache_used") is True
    assert meta.get("force_refresh_supported") is True


def test_06_07_force_respects_rate_limits():
    from app.ae13b_product.dexscreener_pair_verify import (
        get_pair_verify_limiter,
        validate_dexscreener_pair,
    )

    def fake_get(_c, _p):
        return {
            "ok": True,
            "status_code": 200,
            "pair": _sample_pair(),
            "error": None,
            "retry_after_seconds": None,
            "rate_limited": False,
            "provider_unavailable": False,
        }

    validate_dexscreener_pair(
        "solana", "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd",
        use_cache=False, _http_get=fake_get,
    )
    stats = get_pair_verify_limiter().stats_snapshot()
    assert stats["max_inflight_observed"] <= stats["settings"]["DEXSCREENER_PAIR_VERIFY_MAX_CONCURRENCY"]


def test_08_429_deferred_not_clean():
    from app.ae13b_product.dexscreener_pair_verify import validate_dexscreener_pair

    def fake_get(_c, _p):
        return {
            "ok": False,
            "status_code": 429,
            "pair": None,
            "error": "too_many_requests",
            "retry_after_seconds": 0.01,
            "rate_limited": True,
            "provider_unavailable": False,
        }

    with patch(
        "app.ae13b_product.dexscreener_pair_verify.DEXSCREENER_PAIR_VERIFY_MAX_RETRIES",
        0,
    ):
        r = validate_dexscreener_pair(
            "solana", "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd",
            use_cache=False, _http_get=fake_get,
        )
    assert r.verification_status == "provider_rate_limited"
    assert r.clean_feed_eligible is False


def test_10_5xx_deferred():
    from app.ae13b_product.dexscreener_pair_verify import validate_dexscreener_pair

    def fake_get(_c, _p):
        return {
            "ok": False,
            "status_code": 503,
            "pair": None,
            "error": "provider_5xx_503",
            "rate_limited": False,
            "provider_unavailable": True,
        }

    with patch(
        "app.ae13b_product.dexscreener_pair_verify.DEXSCREENER_PAIR_VERIFY_MAX_RETRIES",
        0,
    ):
        r = validate_dexscreener_pair(
            "solana", "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd",
            use_cache=False, _http_get=fake_get,
        )
    assert r.verification_status == "provider_unavailable"
    assert r.tradability_status == "verification_deferred"


def test_11_14_chain_and_url_validation_on_refresh_path():
    from app.ae13b_product.dexscreener_pair_verify import (
        address_format_for_chain,
        build_dexscreener_pair_url,
        validate_dexscreener_pair,
    )

    assert address_format_for_chain("solana", "0x" + "a" * 40)["format_ok"] is False
    assert address_format_for_chain("ethereum", "4dtsp9bx38gwytmdpgmqu5yx5bkatrfck1akgn1pujjm")["format_ok"] is False

    r = validate_dexscreener_pair("solana", "0x" + "b" * 40, use_cache=False)
    assert r.provider_pair_url is None

    pair = _sample_pair(url="")

    def fake_get(_c, _p):
        return {
            "ok": True,
            "status_code": 200,
            "pair": pair,
            "error": None,
            "rate_limited": False,
            "provider_unavailable": False,
        }

    r2 = validate_dexscreener_pair(
        "solana", pair["pairAddress"], use_cache=False, _http_get=fake_get
    )
    assert r2.provider_pair_url_source == "constructed_after_verified_lookup"
    assert r2.pair_address.lower() == pair["pairAddress"].lower()
    assert r2.normalized_chain_id == "solana"
    assert build_dexscreener_pair_url("solana", pair["pairAddress"]).startswith(
        "https://dexscreener.com/solana/"
    )


def test_18_20_ui_metadata_and_messages_present():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "static" / "product_demo.js").read_text(encoding="utf-8")
    css = (ROOT / "static" / "product_demo.css").read_text(encoding="utf-8")
    assert "cf-refresh-meta-panel" in html
    assert "cf-meta-mode" in html
    assert "cfForceProviderRefresh" in js
    assert "cfRenderRefreshMetadata" in js
    assert "cached provider verification" in js.lower() or "cache_hit" in js
    assert "no market value changes" in js.lower()
    assert "rate limit" in js.lower()


def test_21_24_visual_readability_no_blue_links():
    css = (ROOT / "static" / "product_demo.css").read_text(encoding="utf-8")
    js = (ROOT / "static" / "product_demo.js").read_text(encoding="utf-8")
    assert ".cf-link" in css
    assert "#5eead4" in css or "5eead4" in css
    assert ".cf-pct-pos" in css
    assert ".cf-pct-neg" in css
    assert ".cf-pct-neu" in css
    assert "cf-link" in js
    assert "cf-pct-pos" in js
    assert "3b82f6" not in css  # default accent blue not used for cf links


def test_25_30_scope_and_safety_flags():
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


def test_cache_hit_mode_reported():
    from app.ae13b_product.clean_forward_market_feed import refresh_clean_forward_market_feed

    pair = _sample_pair()
    search = [{"chainId": "solana", "pairAddress": pair["pairAddress"], "priceUsd": "0.01",
               "liquidity": {"usd": 1}, "volume": {"h24": 1},
               "baseToken": pair["baseToken"], "quoteToken": pair["quoteToken"]}]

    verified = {**_verified_dict(pair), "verification_cache_hit": True}

    with patch(
        "app.ae13b_product.clean_forward_market_feed.get_trending_pairs_sync",
        return_value=search,
    ), patch(
        "app.ae13b_product.clean_forward_market_feed.verify_provider_pair",
        return_value=verified,
    ):
        out = refresh_clean_forward_market_feed(force=False, clear_cache=False, max_verify=5)
    meta = out["refresh"]
    assert meta["cache_hit_count"] >= 1
    assert meta["refresh_mode"] == "cache_hit"
    assert "cached provider verification" in meta["ui_message"].lower()
