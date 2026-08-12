from app.clean_forward.market_activity import (
    ACTIVE_PROVIDER_TXNS,
    ACTIVITY_STAGNANT,
    ACTIVITY_UNKNOWN,
    NO_RECENT_PROVIDER_TXNS,
    evaluate_market_activity,
)

def test_missing_symbols_do_not_affect_active_market_activity():
    row = {
        "symbol_pair_display": "SYMBOLS_UNAVAILABLE_PROVIDER_CACHE_MISSING",
        "price_usd": 1.0,
        "liquidity_usd": 10000,
        "txns_h24_buys": 10,
        "txns_h24_sells": 5,
        "volume_h24": 1234,
        "price_change_h24": 2.5,
    }
    out = evaluate_market_activity(row)
    assert out["market_activity_status"] == ACTIVE_PROVIDER_TXNS
    assert out["activity_uses_symbol_display"] is False
    assert out["market_activity_blocks_demo_entry"] is False

def test_zero_txns_are_not_actionable_even_with_liquidity():
    row = {
        "symbol_pair_display": "PRETTY/USDC",
        "price_usd": 1.0,
        "liquidity_usd": 999999,
        "market_cap": 5000000,
        "txns_h24_buys": 0,
        "txns_h24_sells": 0,
        "volume_h24": 0,
        "price_change_m5": 0,
        "price_change_h1": 0,
        "price_change_h6": 0,
        "price_change_h24": 0,
    }
    out = evaluate_market_activity(row)
    assert out["market_activity_status"] == NO_RECENT_PROVIDER_TXNS
    assert out["market_activity_blocks_demo_entry"] is True
    assert out["activity_uses_liquidity_or_market_cap_as_activity_proxy"] is False

def test_static_metadata_without_activity_is_stagnant():
    row = {
        "price_usd": 0.01,
        "liquidity_usd": 10000,
        "volume_h24": 0,
        "price_change_h24": 0,
    }
    out = evaluate_market_activity(row)
    assert out["market_activity_status"] == ACTIVITY_STAGNANT
    assert out["market_activity_blocks_demo_entry"] is True

def test_missing_activity_metadata_is_unknown_not_active():
    out = evaluate_market_activity({"price_usd": 1.0})
    assert out["market_activity_status"] == ACTIVITY_UNKNOWN
    assert out["market_activity_blocks_demo_entry"] is True
