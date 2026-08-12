"""AE13K targeted validation — clean forward feed, DexScreener verify, diversity, safety."""
from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _reset_verify_limiter():
    from app.ae13b_product.dexscreener_pair_verify import get_pair_verify_limiter

    lim = get_pair_verify_limiter()
    lim.clear_cache()
    lim.reset_stats()
    yield
    lim.clear_cache()
    lim.reset_stats()


def _sample_pair(
    *,
    chain: str = "solana",
    pair: str = "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd",
    base: str = "pumpCmXqMfrsAkQ5r49WcJnRayYRqmXz6ae8H7hCNwe",
    quote: str = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
    symbol: str = "PUMP",
    url: str | None = None,
    price: str = "0.01",
    liq: float = 100000.0,
) -> dict[str, Any]:
    return {
        "chainId": chain,
        "dexId": "pumpswap",
        "pairAddress": pair,
        "url": url or f"https://dexscreener.com/{chain}/{pair}",
        "baseToken": {"address": base, "symbol": symbol, "name": symbol},
        "quoteToken": {"address": quote, "symbol": "USDC", "name": "USD Coin"},
        "priceUsd": price,
        "liquidity": {"usd": liq},
        "volume": {"m5": 1.0, "h1": 10.0, "h6": 50.0, "h24": 100.0},
        "txns": {
            "m5": {"buys": 1, "sells": 1},
            "h1": {"buys": 5, "sells": 4},
            "h24": {"buys": 50, "sells": 40},
        },
        "priceChange": {"m5": 0.1, "h1": 1.0, "h6": 2.0, "h24": 3.0},
        "pairCreatedAt": 1700000000000,
    }


def test_01_clean_feed_does_not_import_or_write_old_db():
    import app.ae13b_product.clean_forward_market_feed as cf
    import app.ae13b_product.dexscreener_pair_verify as pv

    src_cf = Path(cf.__file__).read_text(encoding="utf-8")
    src_pv = Path(pv.__file__).read_text(encoding="utf-8")
    for forbidden in (
        'open("data/trader.db"',
        "sqlite3.connect",
        "manual_verified_datasets",
        "INSERT INTO",
        "persist_pair",
    ):
        assert forbidden not in src_cf
        assert forbidden not in src_pv
    assert "from app.database" not in src_cf
    assert "from app.database" not in src_pv


def test_07_solana_rejects_0x_address():
    from app.ae13b_product.dexscreener_pair_verify import (
        address_format_for_chain,
        validate_dexscreener_pair,
    )

    fmt = address_format_for_chain("solana", "0xd239aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    assert fmt["format_ok"] is False
    assert fmt["reason"] == "chain_address_format_mismatch"

    r = validate_dexscreener_pair(
        "solana", "0xd239aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", use_cache=False
    )
    assert r.verification_status == "chain_address_format_mismatch"
    assert r.tradability_status == "ambiguous_address_role"
    assert r.clean_feed_eligible is False
    assert r.provider_pair_url is None


def test_08_evm_rejects_solana_base58():
    from app.ae13b_product.dexscreener_pair_verify import (
        address_format_for_chain,
        validate_dexscreener_pair,
    )

    sol = "4dtsp9bx38gwytmdpgmqu5yx5bkatrfck1akgn1pujjm"
    fmt = address_format_for_chain("ethereum", sol)
    assert fmt["format_ok"] is False
    assert fmt["reason"] == "chain_address_format_mismatch"

    r = validate_dexscreener_pair("bsc", sol, use_cache=False)
    assert r.verification_status == "chain_address_format_mismatch"
    assert r.clean_feed_eligible is False
    assert r.provider_pair_url is None


def test_chain_normalization():
    from app.ae13b_product.dexscreener_pair_verify import normalize_chain_id

    assert normalize_chain_id("eth") == "ethereum"
    assert normalize_chain_id("binance-smart-chain") == "bsc"
    assert normalize_chain_id("SOL") == "solana"
    assert normalize_chain_id("matic") == "polygon"


def test_04_url_not_constructed_before_verification():
    from app.ae13b_product.dexscreener_pair_verify import validate_dexscreener_pair

    def boom(*_a, **_k):
        raise AssertionError("HTTP should not be called for format mismatch")

    r = validate_dexscreener_pair(
        "solana",
        "0xd239aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        use_cache=False,
        _http_get=boom,
    )
    assert r.provider_pair_url is None
    assert r.provider_pair_url_source is None


def test_04b_url_constructed_only_after_success_when_provider_omits_url():
    from app.ae13b_product.dexscreener_pair_verify import validate_dexscreener_pair

    pair = _sample_pair(url="")
    pair["url"] = ""

    def fake_get(_c, _p):
        return {
            "ok": True,
            "status_code": 200,
            "pair": pair,
            "error": None,
            "retry_after_seconds": None,
            "rate_limited": False,
            "provider_unavailable": False,
        }

    r = validate_dexscreener_pair(
        "solana", pair["pairAddress"], use_cache=False, _http_get=fake_get
    )
    assert r.clean_feed_eligible is True
    assert r.provider_pair_url_source == "constructed_after_verified_lookup"
    assert r.provider_pair_url.startswith("https://dexscreener.com/solana/")


def test_04c_provider_returned_url_preferred():
    from app.ae13b_product.dexscreener_pair_verify import validate_dexscreener_pair

    pair = _sample_pair()

    def fake_get(_c, _p):
        return {
            "ok": True,
            "status_code": 200,
            "pair": pair,
            "error": None,
            "retry_after_seconds": None,
            "rate_limited": False,
            "provider_unavailable": False,
        }

    r = validate_dexscreener_pair(
        "solana", pair["pairAddress"], use_cache=False, _http_get=fake_get
    )
    assert r.provider_pair_url_source == "provider_returned_url"
    assert r.provider_pair_url == pair["url"]


def test_verified_row_fields_and_matches():
    from app.ae13b_product.dexscreener_pair_verify import validate_dexscreener_pair

    pair = _sample_pair()

    def fake_get(chain, pid):
        assert chain == "solana"
        assert pid == pair["pairAddress"]
        return {
            "ok": True,
            "status_code": 200,
            "pair": pair,
            "error": None,
            "retry_after_seconds": None,
            "rate_limited": False,
            "provider_unavailable": False,
        }

    r = validate_dexscreener_pair(
        "solana", pair["pairAddress"], use_cache=False, _http_get=fake_get
    )
    assert r.lookup_ok is True
    assert r.clean_feed_eligible is True
    assert r.pair_address.lower() == pair["pairAddress"].lower()
    assert r.normalized_chain_id == "solana"
    assert r.base_token_address
    assert r.quote_token_address
    assert r.price_usd is not None
    assert r.liquidity_usd is not None
    assert r.provider_pair_url
    assert r.verification_status == "provider_pair_verified"


def test_11_unresolved_not_clean_eligible():
    from app.ae13b_product.dexscreener_pair_verify import validate_dexscreener_pair

    def fake_get(_c, _p):
        return {
            "ok": False,
            "status_code": 404,
            "pair": None,
            "error": "not_found",
            "retry_after_seconds": None,
            "rate_limited": False,
            "provider_unavailable": False,
        }

    r = validate_dexscreener_pair(
        "solana",
        "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd",
        use_cache=False,
        _http_get=fake_get,
    )
    assert r.clean_feed_eligible is False
    assert r.verification_status == "provider_pair_not_found"
    assert r.provider_pair_url is None


def test_09_10_base_quote_separate_pair_not_token_contract():
    from app.ae13b_product.clean_forward_market_feed import _row_from_verification

    pair = _sample_pair()
    v = {
        "normalized_chain_id": "solana",
        "pair_address": pair["pairAddress"],
        "provider_pair_id": pair["pairAddress"],
        "provider_pair_url": pair["url"],
        "provider_pair_url_source": "provider_returned_url",
        "dex_id": "pumpswap",
        "base_token_address": pair["baseToken"]["address"],
        "base_token_symbol": "PUMP",
        "quote_token_address": pair["quoteToken"]["address"],
        "quote_token_symbol": "USDC",
        "price_usd": "0.01",
        "liquidity_usd": 100000,
        "address_role": "pool_address",
        "verification_status": "provider_pair_verified",
        "identity_status": "pair_and_tokens_separated",
        "tradability_status": "provider_pair_verified_display_only",
        "freshness_status": "fresh",
        "clean_feed_eligible": True,
        "provider_payload_hash": "abc",
        "fetched_at": "2026-07-20T00:00:00+00:00",
    }
    row = _row_from_verification(v)
    assert row["shown_as_token_contract"] is False
    assert row["address_role_label"] == "Pair / Pool address"
    assert row["base_token_address_label"] == "Base token address"
    assert row["quote_token_address_label"] == "Quote token address"
    assert row["pair_address"] != row["base_token_address"]
    assert row["pair_address"] != row["quote_token_address"]


def test_12_13_14_diversity_suppresses_duplicate_bases_and_wif():
    from app.ae13b_product.clean_forward_market_feed import _apply_diversity

    def mk(pair, base, sym, liq, vol):
        return {
            "pair_address": pair,
            "base_token_address": base,
            "base_token_symbol": sym,
            "liquidity_usd": liq,
            "volume_24h": vol,
            "fetched_at": "2026-07-20T00:00:00+00:00",
            "txns_24h_buys": 10,
            "txns_24h_sells": 5,
            "provider_pair_url": f"https://dexscreener.com/solana/{pair}",
            "verification_status": "provider_pair_verified",
        }

    wif_mint = "EKpQGSJjmqWmDvhiUHmM9BgRb8jxeyN35xC4RGomoon"
    rows = [
        mk("pairWIF1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxABC", wif_mint, "WIF", 1_000_000, 500),
        mk("pairWIF2xxxxxxxxxxxxxxxxxxxxxxxxxxxxxDEF", wif_mint, "WIF", 500_000, 200),
        mk("pairWIF3xxxxxxxxxxxxxxxxxxxxxxxxxxxxxGHI", wif_mint, "WIF", 100_000, 50),
        mk(
            "pairBONKxxxxxxxxxxxxxxxxxxxxxxxxxxxxJKL",
            "BonkMintxxxxxxxxxxxxxxxxxxxxxxxxxx",
            "BONK",
            800_000,
            100,
        ),
        mk("pairWIF1xxxxxxxxxxxxxxxxxxxxxxxxxxxxxABC", wif_mint, "WIF", 1_000_000, 500),
    ]
    main, alts, _events = _apply_diversity(
        rows, max_rows_per_base_token=1, max_rows_per_symbol=1, limit=25
    )
    wif_main = [r for r in main if r["base_token_symbol"] == "WIF"]
    wif_alts = [r for r in alts if r["base_token_symbol"] == "WIF"]
    assert len(wif_main) == 1
    assert len(wif_alts) >= 2
    assert len({r["pair_address"].lower() for r in main}) == len(main)
    bases = [r["base_token_address"].lower() for r in main]
    assert len(bases) == len(set(bases))


def test_15_16_cache_and_bounded_concurrency():
    from app.ae13b_product.dexscreener_pair_verify import (
        get_pair_verify_limiter,
        validate_dexscreener_pair,
    )

    pair = _sample_pair()
    calls = {"n": 0}

    def fake_get(_c, _p):
        calls["n"] += 1
        return {
            "ok": True,
            "status_code": 200,
            "pair": pair,
            "error": None,
            "retry_after_seconds": None,
            "rate_limited": False,
            "provider_unavailable": False,
        }

    r1 = validate_dexscreener_pair(
        "solana", pair["pairAddress"], use_cache=True, _http_get=fake_get
    )
    r2 = validate_dexscreener_pair(
        "solana", pair["pairAddress"], use_cache=True, _http_get=fake_get
    )
    assert r1.clean_feed_eligible is True
    assert r2.verification_cache_hit is True
    assert calls["n"] == 1
    stats = get_pair_verify_limiter().stats_snapshot()
    assert stats["cache_hits"] >= 1
    assert stats["http_calls"] == 1
    assert stats["max_inflight_observed"] <= stats["settings"][
        "DEXSCREENER_PAIR_VERIFY_MAX_CONCURRENCY"
    ]


def test_17_429_is_deferred_not_clean():
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
            "solana",
            "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd",
            use_cache=False,
            _http_get=fake_get,
        )
    assert r.verification_status == "provider_rate_limited"
    assert r.tradability_status == "verification_deferred"
    assert r.clean_feed_eligible is False
    assert r.provider_pair_url is None


def test_18_5xx_timeout_deferred_not_clean():
    from app.ae13b_product.dexscreener_pair_verify import validate_dexscreener_pair

    def fake_get(_c, _p):
        return {
            "ok": False,
            "status_code": 503,
            "pair": None,
            "error": "provider_5xx_503",
            "retry_after_seconds": None,
            "rate_limited": False,
            "provider_unavailable": True,
        }

    with patch(
        "app.ae13b_product.dexscreener_pair_verify.DEXSCREENER_PAIR_VERIFY_MAX_RETRIES",
        0,
    ):
        r = validate_dexscreener_pair(
            "solana",
            "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd",
            use_cache=False,
            _http_get=fake_get,
        )
    assert r.verification_status == "provider_unavailable"
    assert r.tradability_status == "verification_deferred"
    assert r.clean_feed_eligible is False


def test_02_clean_feed_only_verified_rows():
    from app.ae13b_product import clean_forward_market_feed as cf

    good = _sample_pair()
    bad_pair = "BadPairAddressThatIsLongEnoughBase58xxxx"

    search_pairs = [
        {
            "chainId": "solana",
            "pairAddress": good["pairAddress"],
            "priceUsd": "0.01",
            "liquidity": {"usd": 100000},
            "volume": {"h24": 10},
            "baseToken": good["baseToken"],
            "quoteToken": good["quoteToken"],
        },
        {
            "chainId": "solana",
            "pairAddress": bad_pair,
            "priceUsd": "0.02",
            "liquidity": {"usd": 50},
            "volume": {"h24": 1},
            "baseToken": {"address": "x", "symbol": "X"},
            "quoteToken": {"address": "y", "symbol": "Y"},
        },
    ]

    def fake_trending(**_k):
        return search_pairs

    def fake_verify(*, chain_id, pair_address, **_k):
        if pair_address == good["pairAddress"]:
            return {
                "lookup_ok": True,
                "clean_feed_eligible": True,
                "verification_status": "provider_pair_verified",
                "status": "provider_pair_verified",
                "tradability_status": "provider_pair_verified_display_only",
                "identity_status": "pair_and_tokens_separated",
                "normalized_chain_id": "solana",
                "chain_id": "solana",
                "pair_address": good["pairAddress"],
                "provider_pair_id": good["pairAddress"],
                "provider_pair_url": good["url"],
                "provider_pair_url_source": "provider_returned_url",
                "dex_id": "pumpswap",
                "base_token_address": good["baseToken"]["address"],
                "base_token_symbol": "PUMP",
                "quote_token_address": good["quoteToken"]["address"],
                "quote_token_symbol": "USDC",
                "price_usd": "0.01",
                "liquidity_usd": 100000,
                "volume_24h": 100,
                "txns_24h_buys": 50,
                "txns_24h_sells": 40,
                "txns_24h": {"buys": 50, "sells": 40, "total": 90},
                "provider_payload_hash": "hash1",
                "payload_hash": "hash1",
                "fetched_at": "2026-07-20T00:00:00+00:00",
                "ingested_at": "2026-07-20T00:00:00+00:00",
                "observed_at": "2026-07-20T00:00:00+00:00",
                "freshness_status": "fresh",
                "address_role": "pool_address",
                "pair_label": "PUMP/USDC",
                "exclusion_reason": None,
            }
        return {
            "lookup_ok": False,
            "clean_feed_eligible": False,
            "verification_status": "provider_pair_not_found",
            "status": "provider_pair_not_found",
            "tradability_status": "not_tradable_without_provider_pair",
            "identity_status": "unresolved",
            "provider_pair_url": None,
            "provider_pair_url_source": None,
            "exclusion_reason": "not_found",
            "reject_reason": "not_found",
        }

    with patch.object(cf, "get_trending_pairs_sync", fake_trending), patch.object(
        cf, "verify_provider_pair", fake_verify
    ):
        feed = cf.build_clean_forward_market_feed(limit=10, max_verify=10, max_candidates=10)

    assert feed["stats"]["clean_rows_displayed"] == 1
    assert all(r.get("verification_status") == "provider_pair_verified" for r in feed["rows"])
    assert all(r.get("provider_pair_url") for r in feed["rows"])
    assert all(r.get("shown_as_token_contract") is False for r in feed["rows"])
    assert feed["training_run"] is False
    assert feed["backtest_run"] is False
    assert feed["ae14_run"] is False
    assert feed["paper_positions_opened_from_clean_feed"] == 0
    assert feed["live_trading_enabled"] is False


def test_19_classify_refresh():
    from app.ae13b_product.clean_forward_market_feed import classify_refresh

    prev = {
        "price_usd": "1",
        "liquidity_usd": 10,
        "volume": {"h24": 1},
        "txns": {},
        "payload_hash": "aaa",
        "fetched_at": "t1",
    }
    same = {**prev, "fetched_at": "t2"}
    changed = {**prev, "price_usd": "2", "payload_hash": "bbb", "fetched_at": "t2"}
    assert classify_refresh(prev, same, lookup_ok=True) == "provider_unchanged_but_refetched"
    assert classify_refresh(prev, changed, lookup_ok=True) == "provider_updated"
    assert (
        classify_refresh(
            prev,
            same,
            lookup_ok=False,
            verification_status="provider_rate_limited",
        )
        == "provider_rate_limited"
    )


def test_20_ui_separates_clean_and_legacy():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "static" / "product_demo.js").read_text(encoding="utf-8")
    assert "Clean Forward Market Feed" in html
    assert "Market Snapshot Feed" in html
    assert 'id="tab-clean-forward"' in html
    assert 'id="tab-live-market"' in html
    assert "loadCleanForwardFeedTab" in js
    assert "Pair/pool addresses are not token contracts" in html
    assert "No clean provider-verified market rows available yet." in js


def test_21_25_no_training_trading_wallet_in_clean_modules():
    paths = [
        ROOT / "app" / "ae13b_product" / "clean_forward_market_feed.py",
        ROOT / "app" / "ae13b_product" / "dexscreener_pair_verify.py",
    ]
    banned = [
        "train_model(",
        "run_ae14(",
        "wallet_private_key",
        "sign_transaction(",
        "submit_transaction(",
        "open_position(",
        "run_backtest(",
    ]
    for p in paths:
        text = p.read_text(encoding="utf-8").lower()
        for b in banned:
            assert b not in text, f"{p.name} contains banned token {b}"

    from app.ae13b_product.clean_forward_market_feed import build_clean_forward_market_feed

    with patch(
        "app.ae13b_product.clean_forward_market_feed.get_trending_pairs_sync",
        return_value=[],
    ):
        feed = build_clean_forward_market_feed(limit=1, max_verify=1, max_candidates=1)
    assert feed["training_run"] is False
    assert feed["backtest_run"] is False
    assert feed["ae14_run"] is False
    assert feed["paper_positions_opened_from_clean_feed"] == 0
    assert feed["live_trading_enabled"] is False


def test_build_dexscreener_pair_url():
    from app.ae13b_product.dexscreener_pair_verify import build_dexscreener_pair_url

    url = build_dexscreener_pair_url(
        "solana", "4dtsp9bx38gwytmdpgmqu5yx5bkatrfck1akgn1pujjm"
    )
    assert url == (
        "https://dexscreener.com/solana/4dtsp9bx38gwytmdpgmqu5yx5bkatrfck1akgn1pujjm"
    )


def test_candidate_does_not_preconstruct_url():
    from app.ae13b_product.clean_forward_market_feed import candidate_from_search_pair

    c = candidate_from_search_pair(_sample_pair())
    assert c["provider_pair_url"] is None
