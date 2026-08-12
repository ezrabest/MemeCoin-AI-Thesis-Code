"""AE14 Real Clean Forward Closure — focused acceptance tests."""
from __future__ import annotations

import importlib
import importlib.util
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_ae14_real_clean_forward_closure.py"
TEST_FILE = ROOT / "tests" / "test_ae14_real_clean_forward_closure.py"


def _fresh_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _real_cf_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "row_id": "base|pair|0x1111111111111111111111111111111111111111",
        "source_provider": "dexscreener",
        "normalized_chain_id": "base",
        "chain": "base",
        "chain_id": "base",
        "provider_pair_id": "0x1111111111111111111111111111111111111111",
        "pair_address": "0x1111111111111111111111111111111111111111",
        "base_token_address": "0x2222222222222222222222222222222222222222",
        "base_token_symbol": "SOL",
        "quote_token_address": "0x3333333333333333333333333333333333333333",
        "quote_token_symbol": "cbBTC",
        "pair": "SOL/cbBTC",
        "pair_label": "SOL/cbBTC",
        "price": "25.5",
        "price_usd": "25.5",
        "liquidity": 420000.0,
        "liquidity_usd": 420000.0,
        "volume_24h": 100000.0,
        "observed_at": _fresh_ts(),
        "fetched_at": _fresh_ts(),
        "last_fetched": _fresh_ts(),
        "ingested_at": _fresh_ts(),
        "verification_status": "provider_pair_verified",
        "freshness_status": "fresh",
        "identity_status": "pair_and_tokens_separated",
        "shown_as_token_contract": False,
        "paper_demo_only": True,
        "live_trading_ready": False,
        "address_role": "pair_contract",
        "price_change_1h": 1.5,
        "txns_24h_buys": 40,
        "txns_24h_sells": 30,
    }
    row.update(overrides)
    return row


def _synthetic_fixture_row() -> dict[str, Any]:
    return _real_cf_row(
        row_id="base|pair|0xAE14PaperLifecycle01",
        provider_pair_id="0xAE14PaperLifecycle01",
        pair_address="0xAE14PaperLifecycle01",
        pair="SOL/cbBTC",
    )


def test_01_runner_and_test_paths_exist():
    assert RUNNER.is_file()
    assert TEST_FILE.is_file()


def test_02_rejects_synthetic_fixture_rows():
    from app.ae13b_product.ae14_candidate_source_policy import (
        is_synthetic_or_fixture_row,
        is_valid_ae14_clean_forward_row,
        select_ae14_clean_forward_candidates,
    )

    synth = _synthetic_fixture_row()
    real = _real_cf_row()
    assert is_synthetic_or_fixture_row(synth) is True
    assert is_valid_ae14_clean_forward_row(synth) is False
    selected = select_ae14_clean_forward_candidates([synth, real])
    assert len(selected) == 1
    assert selected[0]["pair_address"] == real["pair_address"]
    assert selected[0]["pair"] == "SOL/cbBTC"


def test_03_rejects_0xAE14PaperLifecycle01():
    from app.ae13b_product.ae14_candidate_source_policy import is_synthetic_or_fixture_row

    assert is_synthetic_or_fixture_row(
        {"pair_address": "0xAE14PaperLifecycle01", "pair": "SOL/cbBTC"}
    )


def test_04_selects_only_clean_forward_feed_rows(tmp_path):
    spec = importlib.util.spec_from_file_location("ae14_closure_runner", RUNNER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    real = _real_cf_row()
    audit = mod.run_closure(
        out_dir=tmp_path / "closure",
        skip_runtime_check=True,
        feed_rows_override=[_synthetic_fixture_row(), real],
        refresh_feed=False,
    )
    assert audit["status"] == "PASS"
    assert audit["real_clean_forward_row_used"] is True
    assert audit["synthetic_fixture_used"] is False
    assert audit["selected_pair"] == "SOL/cbBTC"
    assert audit["selected_pair_address"] == real["pair_address"]
    assert audit["ae14_candidate_source_policy"] == "clean_forward_market_feed_only"
    assert audit["legacy_market_snapshots_used"] is False
    assert audit["old_watchlist_candidates_used"] is False
    assert audit["local_db_candidate_universe_used"] is False


def test_05_selected_pair_preserved_into_paper_position(tmp_path):
    spec = importlib.util.spec_from_file_location("ae14_closure_runner", RUNNER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    row = _real_cf_row(pair="WETH/USDC", base_token_symbol="WETH", quote_token_symbol="USDC")
    audit = mod.run_closure(
        out_dir=tmp_path / "closure_pair",
        skip_runtime_check=True,
        feed_rows_override=[row],
        refresh_feed=False,
    )
    assert audit["status"] == "PASS"
    assert audit["opened_position_pair"] == "WETH/USDC"
    assert audit["selected_pair"] == "WETH/USDC"
    assert audit["opened_position_entry_price"] == pytest.approx(25.5)
    assert audit["opened_position_liquidity_at_entry"] == pytest.approx(420000.0)
    assert audit["coin_id"] is None
    assert audit["opened_position_coin_id"] is None
    assert str(audit["instrument_id"]).startswith("clean_forward:")
    assert audit["execution_mode"] == "paper"
    assert audit["live_execution_enabled"] is False
    assert audit["wallet_connected"] is False
    assert audit["not_profitability_evidence"] is True


def test_06_price_liquidity_preserved_through_gatekeeper_and_paper(tmp_path):
    from app.ae13b_product.clean_forward_bridge import build_clean_forward_gatekeeper_candidate
    from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

    row = _real_cf_row(price_usd="77.77", price="77.77", liquidity_usd=558118.71)
    bridge = build_clean_forward_gatekeeper_candidate(row)
    assert bridge["ok"] is True
    cand = bridge["candidate"]
    assert cand["latest_price"] == pytest.approx(77.77)
    assert cand["latest_liquidity"] == pytest.approx(558118.71)
    gate = validate_market_data_gate(cand, for_open=True, skip_stagnant=False)
    assert gate["passed"] is True

    spec = importlib.util.spec_from_file_location("ae14_closure_runner", RUNNER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    audit = mod.run_closure(
        out_dir=tmp_path / "closure_px",
        skip_runtime_check=True,
        feed_rows_override=[row],
        refresh_feed=False,
    )
    assert audit["selected_price_usd"] == pytest.approx(77.77)
    assert audit["selected_liquidity_usd"] == pytest.approx(558118.71)
    assert audit["opened_position_entry_price"] == pytest.approx(77.77)
    assert audit["gatekeeper_pass_count"] >= 1


def test_07_demo_bot_ae14_mode_uses_clean_forward_only(tmp_path, monkeypatch):
    os.environ["TRADER_DB_PATH"] = str(tmp_path / "bot.db")
    import app.execution.paper as paper
    import app.database as database
    from app.ae13b_product.ae14_candidate_source_policy import (
        disable_ae14_closure_mode,
        enable_ae14_closure_mode,
    )
    from app.ae13b_product.clean_forward_market_feed import set_cached_clean_forward_rows
    from app.ae13b_product.demo_bot import get_demo_bot, reset_demo_bot_for_tests

    importlib.reload(paper)
    importlib.reload(database)
    paper.DATA_DIR = tmp_path
    paper.STATE_PATH = tmp_path / "paper_state.json"
    paper.TRADES_LOG_PATH = tmp_path / "paper_trades_log.csv"
    paper._paper_trader = None
    database.DATA_DIR = tmp_path
    database.DB_PATH = tmp_path / "bot.db"
    database.init_db()

    set_cached_clean_forward_rows([_real_cf_row()])
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
    enable_ae14_closure_mode()
    try:
        reset_demo_bot_for_tests()
        bot = get_demo_bot()
        bot.apply_preset("balanced")
        out = bot.run_once()
        assert out.get("candidate_source") == "clean_forward_market_feed"
        assert out.get("ae14_candidate_source_policy") == "clean_forward_market_feed_only"
        assert out.get("legacy_market_snapshots_used") is False
        assert out.get("old_watchlist_candidates_used") is False
        assert out.get("local_db_candidate_universe_used") is False
        assert out.get("clean_forward_bridge_used") is True
    finally:
        disable_ae14_closure_mode()
        set_cached_clean_forward_rows(None)
        reset_demo_bot_for_tests()
        os.environ.pop("TRADER_DB_PATH", None)


def test_08_demo_queue_ae14_mode_uses_clean_forward_only(tmp_path):
    import app.ae13b_product.demo_queue as demo_queue
    from app.ae13b_product.ae14_candidate_source_policy import (
        disable_ae14_closure_mode,
        enable_ae14_closure_mode,
    )
    from app.ae13b_product.clean_forward_market_feed import set_cached_clean_forward_rows

    demo_queue.DATA_DIR = tmp_path
    demo_queue.QUEUE_PATH = tmp_path / "demo_trade_queue.json"
    row = _real_cf_row()
    set_cached_clean_forward_rows([row])
    enable_ae14_closure_mode()
    try:
        entry = demo_queue.add_to_demo_queue(
            symbol="SOL",
            pair="SOL/cbBTC",
            chain="base",
            contract_or_pair_address=row["pair_address"],
            source="clean_forward_market_feed",
            market_match_status="provider_pair_verified",
            risk_mode="balanced",
            max_notional=50.0,
            user_hypothesis="ae14 test",
        )
        out = demo_queue.evaluate_queue_item(entry["queue_id"])
        assert out.get("candidate_source") == "clean_forward_market_feed"
        assert out.get("ae14_candidate_source_policy") == "clean_forward_market_feed_only"
        assert out.get("legacy_market_snapshots_used") is False
        assert out.get("clean_forward_bridge_used") is True
    finally:
        disable_ae14_closure_mode()
        set_cached_clean_forward_rows(None)


def test_09_legacy_snapshots_and_watchlist_not_used_in_closure(tmp_path):
    spec = importlib.util.spec_from_file_location("ae14_closure_runner", RUNNER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    audit = mod.run_closure(
        out_dir=tmp_path / "closure_legacy",
        skip_runtime_check=True,
        feed_rows_override=[_real_cf_row()],
        refresh_feed=False,
    )
    assert audit["legacy_market_snapshots_used"] is False
    assert audit["old_watchlist_candidates_used"] is False
    assert audit["local_db_candidate_universe_used"] is False
    assert audit["clean_forward_market_feed_used"] is True


def test_10_fail_closed_runtime_not_exclusive(tmp_path):
    spec = importlib.util.spec_from_file_location("ae14_closure_runner", RUNNER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with patch.object(
        mod,
        "check_runtime_exclusive",
        return_value={
            "ok": False,
            "runtime_exclusive": False,
            "preexisting_demo_bot_loop_active": True,
            "blocker": "RUNTIME_NOT_EXCLUSIVE",
            "foreign_processes": ["python main.py --mode ollama"],
        },
    ):
        audit = mod.run_closure(
            out_dir=tmp_path / "closure_runtime",
            feed_rows_override=[_real_cf_row()],
            refresh_feed=False,
        )
    assert audit["status"] == "FAIL_CLOSED"
    assert audit["blocker"] == "RUNTIME_NOT_EXCLUSIVE"
    assert audit["paper_orders_opened"] == 0
    assert audit["paper_positions_opened"] == 0


def test_11_fail_closed_audit_file_locked(tmp_path):
    spec = importlib.util.spec_from_file_location("ae14_closure_runner", RUNNER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    with patch.object(
        mod,
        "check_audit_files_writable",
        return_value={
            "ok": False,
            "audit_files_lock_free": False,
            "blocker": "AUDIT_FILE_LOCKED",
            "locked_paths": ["locked.json"],
        },
    ):
        audit = mod.run_closure(
            out_dir=tmp_path / "closure_lock",
            skip_runtime_check=True,
            feed_rows_override=[_real_cf_row()],
            refresh_feed=False,
        )
    assert audit["status"] == "FAIL_CLOSED"
    assert audit["blocker"] == "AUDIT_FILE_LOCKED"


def test_12_fail_closed_when_no_real_cf_row(tmp_path):
    spec = importlib.util.spec_from_file_location("ae14_closure_runner", RUNNER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    audit = mod.run_closure(
        out_dir=tmp_path / "closure_empty",
        skip_runtime_check=True,
        feed_rows_override=[_synthetic_fixture_row()],
        refresh_feed=False,
    )
    assert audit["status"] == "FAIL_CLOSED"
    assert audit["blocker"] == "NO_REAL_CLEAN_FORWARD_ROW_AVAILABLE"
    assert audit["paper_positions_opened"] == 0


def test_13_legacy_non_ae14_paths_remain_backward_compatible():
    from app.ae13b_product.ae14_candidate_source_policy import (
        disable_ae14_closure_mode,
        requires_clean_forward_only,
    )
    from app.ae13b_product.clean_forward_bridge import is_clean_forward_queue_item

    disable_ae14_closure_mode()
    assert requires_clean_forward_only() is False
    # Legacy queue items are still recognized as non-CF.
    assert is_clean_forward_queue_item({"source": "manual_watchlist"}) is False
    assert is_clean_forward_queue_item({"source": "clean_forward_market_feed"}) is True


def test_14_live_execution_remains_disabled(tmp_path):
    from app.ae13b_product.clean_forward_execution_instrument import (
        build_clean_forward_execution_instrument,
    )
    from app.ae13b_product.clean_forward_bridge import build_clean_forward_gatekeeper_candidate

    bridge = build_clean_forward_gatekeeper_candidate(_real_cf_row())
    built = build_clean_forward_execution_instrument(bridge["candidate"], execution_mode="paper")
    inst = built["instrument"]
    assert inst["live_execution_enabled"] is False
    assert inst["live_trading_ready"] is False
    assert inst["wallet_connected"] is False
    assert inst["execution_mode"] == "paper"

    live = build_clean_forward_execution_instrument(bridge["candidate"], execution_mode="live")
    assert live["ok"] is False
    assert live["live_execution_enabled"] is False


def test_15_paper_fill_price_source_is_clean_forward_market_feed():
    from app.execution.fill_price import resolve_buy_fill_price

    result = resolve_buy_fill_price(
        {
            "pair_address": "0x1111111111111111111111111111111111111111",
            "instrument_id": "clean_forward:base:0x1111111111111111111111111111111111111111",
            "candidate_source": "clean_forward_market_feed",
            "price_usd": 25.5,
            "coin_id": None,
        },
        allow_coin_price_fallback=True,
    )
    assert result.ok is True
    assert result.source == "clean_forward_market_feed"
    assert result.coin_id is None
