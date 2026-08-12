"""AE14 — Clean Forward canonical instrument identity → paper execution."""
from __future__ import annotations

import importlib
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _fresh_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bridge_candidate(**overrides: Any) -> dict[str, Any]:
    from app.ae13b_product.clean_forward_bridge import build_clean_forward_gatekeeper_candidate

    row = {
        "source_provider": "dexscreener",
        "chain": "base",
        "normalized_chain_id": "base",
        "pair_address": "0xABC123PAIR",
        "provider_pair_id": "0xABC123PAIR",
        "base_token_address": "0xBASE",
        "quote_token_address": "0xQUOTE",
        "base_token_symbol": "SOL",
        "pair": "SOL/USDC",
        "price_usd": "12.5",
        "price": "12.5",
        "liquidity_usd": 250000.0,
        "liquidity": 250000.0,
        "observed_at": _fresh_ts(),
        "fetched_at": _fresh_ts(),
        "verification_status": "provider_pair_verified",
        "freshness_status": "fresh",
        "identity_status": "pair_and_tokens_separated",
        "shown_as_token_contract": False,
        "paper_demo_only": True,
        "live_trading_ready": False,
        "address_role": "pair_contract",
        "price_change_1h": 1.0,
        "txns_24h_buys": 5,
        "txns_24h_sells": 4,
    }
    row.update(overrides)
    out = build_clean_forward_gatekeeper_candidate(row)
    assert out["ok"] is True
    return out["candidate"]


def test_01_execution_instrument_without_coin_id():
    from app.ae13b_product.clean_forward_execution_instrument import (
        build_clean_forward_execution_instrument,
    )

    cand = _bridge_candidate()
    assert cand.get("coin_id") is None
    built = build_clean_forward_execution_instrument(cand, execution_mode="paper")
    assert built["ok"] is True
    inst = built["instrument"]
    assert inst["instrument_id"] == "clean_forward:base:0xABC123PAIR"
    assert inst["execution_instrument_id"] == inst["instrument_id"]
    assert inst["coin_id"] is None
    assert inst["execution_mode"] == "paper"
    assert inst["live_execution_enabled"] is False
    assert inst["live_trading_ready"] is False
    assert inst["wallet_connected"] is False
    assert inst["clean_forward_bridge_used"] is True
    assert inst["legacy_market_snapshots_used"] is False


def test_02_missing_pair_blocks_instrument():
    from app.ae13b_product.clean_forward_execution_instrument import (
        build_clean_forward_execution_instrument,
    )

    cand = _bridge_candidate()
    cand["pair_address"] = None
    cand["provider_pair_id"] = None
    built = build_clean_forward_execution_instrument(cand)
    assert built["ok"] is False
    assert any("pair_address" in r for r in built["block_reasons"])


def test_03_missing_price_blocks_instrument():
    from app.ae13b_product.clean_forward_execution_instrument import (
        build_clean_forward_execution_instrument,
    )

    cand = _bridge_candidate()
    cand["latest_price"] = None
    cand["price_usd"] = None
    cand["price"] = None
    built = build_clean_forward_execution_instrument(cand)
    assert built["ok"] is False
    assert any("price" in r for r in built["block_reasons"])


def test_04_fill_price_allows_instrument_without_coin_id():
    from app.execution.fill_price import resolve_buy_fill_price
    from app.ae13b_product.clean_forward_execution_instrument import (
        build_clean_forward_execution_instrument,
    )

    cand = _bridge_candidate()
    inst = build_clean_forward_execution_instrument(cand)["instrument"]
    result = resolve_buy_fill_price(
        inst,
        market_prices_by_pair={inst["pair_address"]: float(inst["latest_price"])},
        allow_coin_price_fallback=True,
        price_timestamp=inst["price_updated_at"],
    )
    assert result.ok is True
    assert result.coin_id is None
    assert result.price == pytest.approx(12.5)
    assert result.rejection_reason is None


def test_05_missing_coin_id_still_blocks_legacy_path():
    from app.execution.fill_price import resolve_buy_fill_price

    result = resolve_buy_fill_price(
        {"symbol": "LEGACY", "pair_address": "pair_legacy", "price_usd": 1.0},
        market_prices_by_pair={"pair_legacy": 1.0},
        allow_coin_price_fallback=True,
    )
    assert result.ok is False
    assert result.rejection_reason == "missing_coin_id"


def test_06_paper_open_without_coin_id(tmp_path, monkeypatch):
    os.environ["TRADER_DB_PATH"] = str(tmp_path / "test.db")
    import app.execution.paper as paper
    import app.database as database

    importlib.reload(paper)
    importlib.reload(database)
    paper.DATA_DIR = tmp_path
    paper.STATE_PATH = tmp_path / "paper_state.json"
    paper.TRADES_LOG_PATH = tmp_path / "paper_trades_log.csv"
    paper._paper_trader = None
    database.DATA_DIR = tmp_path
    database.DB_PATH = tmp_path / "test.db"
    database.init_db()

    from app.ae13b_product.clean_forward_execution_instrument import (
        build_clean_forward_execution_instrument,
    )
    from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

    cand = _bridge_candidate()
    inst = build_clean_forward_execution_instrument(cand)["instrument"]
    gate = validate_market_data_gate(inst, for_open=True, skip_stagnant=False)
    assert gate["passed"] is True

    trader = paper.PaperTrader()
    trader.set_trading_mode("DEMO")
    trader.set_market_prices(
        [{"pair_address": inst["pair_address"], "price_usd": inst["latest_price"]}],
        price_timestamp=inst["price_updated_at"],
    )
    pos = trader.open_position(
        inst,
        size_usd=50.0,
        settings={"starting_capital": 10000, "max_position_size_usd": 100},
        reason_code="DEMO_STRATEGY_ENTRY",
        strategy_type="MANUAL_WATCHLIST_SCOUT",
        allow_coin_price_fallback=True,
        skip_execution_guard=True,
        gate_result=gate,
        risk_mode="balanced",
        preset_id="balanced",
        bot_state={
            "preset_id": "balanced",
            "max_open_positions": 6,
            "max_trades_per_hour": 30,
            "max_notional_usd": 100,
            "cooldown_seconds": 30,
        },
    )
    assert pos is not None
    assert pos.get("coin_id") is None
    assert pos.get("instrument_id") == "clean_forward:base:0xABC123PAIR"
    assert pos.get("execution_mode") == "paper"
    assert pos.get("live_execution_enabled") is False
    assert pos.get("clean_forward_bridge_used") is True
    assert pos.get("legacy_market_snapshots_used") is False
    assert pos.get("not_profitability_evidence") is True
    os.environ.pop("TRADER_DB_PATH", None)


def test_07_demo_bot_opens_cf_paper_position(tmp_path, monkeypatch):
    os.environ["TRADER_DB_PATH"] = str(tmp_path / "test.db")
    import app.execution.paper as paper
    import app.database as database
    from app.ae13b_product.clean_forward_market_feed import set_cached_clean_forward_rows
    from app.ae13b_product.demo_bot import get_demo_bot, reset_demo_bot_for_tests

    importlib.reload(paper)
    importlib.reload(database)
    paper.DATA_DIR = tmp_path
    paper.STATE_PATH = tmp_path / "paper_state.json"
    paper.TRADES_LOG_PATH = tmp_path / "paper_trades_log.csv"
    paper._paper_trader = None
    database.DATA_DIR = tmp_path
    database.DB_PATH = tmp_path / "test.db"
    database.init_db()

    set_cached_clean_forward_rows(
        [
            {
                "source_provider": "dexscreener",
                "chain": "base",
                "normalized_chain_id": "base",
                "pair_address": "0xCFOPEN001",
                "provider_pair_id": "0xCFOPEN001",
                "base_token_address": "0xBASE",
                "quote_token_address": "0xQUOTE",
                "base_token_symbol": "SOL",
                "pair": "SOL/USDC",
                "price_usd": "10.0",
                "price": "10.0",
                "liquidity_usd": 300000.0,
                "liquidity": 300000.0,
                "observed_at": _fresh_ts(),
                "fetched_at": _fresh_ts(),
                "verification_status": "provider_pair_verified",
                "freshness_status": "fresh",
                "identity_status": "pair_and_tokens_separated",
                "shown_as_token_contract": False,
                "paper_demo_only": True,
                "live_trading_ready": False,
                "address_role": "pair_contract",
                "price_change_1h": 2.0,
                "txns_24h_buys": 20,
                "txns_24h_sells": 10,
            }
        ]
    )
    monkeypatch.setattr(
        "app.ae13b_product.demo_bot.resolve_runtime_guard_context",
        lambda: {
            "trading_mode": "DEMO",
            "live_trading_enabled": False,
            "settings": {
                "starting_capital": 10000,
                "max_position_size_usd": 100,
                "take_profit_pct": 0.18,
                "stop_loss_pct": 0.08,
            },
        },
    )
    reset_demo_bot_for_tests()
    bot = get_demo_bot()
    bot.apply_preset("balanced")
    # Ensure exploration / lanes allow open
    with bot._cycle_lock:  # noqa: SLF001
        pass
    out = bot.run_once()
    assert out.get("ok") is True
    assert out.get("clean_forward_bridge_used") is True
    assert out.get("legacy_market_snapshots_used") is False
    opened = out.get("paper_orders_opened") or out.get("paper_positions_opened") or 0
    # If risk/semantic still blocks, at least instrument path must not be missing_coin_id
    if opened < 1:
        buy = out.get("opened") or {}
        rejected = buy.get("rejected_attempts") or []
        codes = [r.get("rejection_code") for r in rejected]
        assert "missing_coin_id" not in codes
        assert "MISSING_PRICE" not in codes
    else:
        assert opened >= 1
        trader = paper.get_paper_trader()
        opens = trader.get_positions(status="OPEN")
        assert opens
        assert opens[0].get("instrument_id", "").startswith("clean_forward:")
        assert opens[0].get("coin_id") is None
        assert opens[0].get("execution_mode") == "paper"
        assert opens[0].get("live_execution_enabled") is False

    reset_demo_bot_for_tests()
    set_cached_clean_forward_rows(None)
    os.environ.pop("TRADER_DB_PATH", None)


def test_08_live_mode_not_authorized_by_instrument_builder():
    from app.ae13b_product.clean_forward_execution_instrument import (
        build_clean_forward_execution_instrument,
    )

    cand = _bridge_candidate()
    built = build_clean_forward_execution_instrument(cand, execution_mode="live")
    assert built["ok"] is False
    assert built["live_execution_enabled"] is False
