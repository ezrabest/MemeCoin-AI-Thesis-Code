"""AE13I targeted tests — Data Trust GateKeeper, freshness/provenance/address-role
enforcement, reentry cooldown persistence, stagnant-price guard, and the
RISK_GUARD_BLOCK schema repair migration.

Paper/demo only — no wallet, no live trading, no private keys. These tests
never start a live server; any FastAPI usage goes through TestClient only.
"""
from __future__ import annotations

import csv
import importlib
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_minutes_ago(minutes: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


# ---------------------------------------------------------------------------
# 1-2: GateKeeper exists as reusable middleware, not ad-hoc inside PaperTrader
# ---------------------------------------------------------------------------


class GateKeeperModuleIsReusableMiddlewareTests(unittest.TestCase):
    def test_gatekeeper_module_importable_standalone(self) -> None:
        from app.ae13b_product import market_data_gatekeeper

        self.assertTrue(hasattr(market_data_gatekeeper, "validate_market_data_gate"))
        self.assertTrue(hasattr(market_data_gatekeeper, "MarketDataGateResult"))

    def test_call_sites_reuse_the_same_module_instead_of_reimplementing(self) -> None:
        """demo_bot / demo_queue / watchlist / paper.py all import the shared
        gatekeeper rather than duplicating freshness/provenance/reentry logic.
        """
        for rel_path in (
            "app/ae13b_product/demo_bot.py",
            "app/ae13b_product/demo_queue.py",
            "app/analytics/watchlist.py",
            "app/execution/paper.py",
        ):
            src = (ROOT / rel_path).read_text(encoding="utf-8")
            self.assertIn(
                "from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate",
                src,
                msg=f"{rel_path} should reuse the shared gatekeeper module",
            )

    def test_paper_trader_gate_is_defense_in_depth_not_primary(self) -> None:
        src = (ROOT / "app" / "execution" / "paper.py").read_text(encoding="utf-8")
        self.assertIn("defense-in-depth", src.lower())


# ---------------------------------------------------------------------------
# 3-8: Freshness gate blocks missing/stale price/liquidity/provider/ambiguous
#      role; semantic labels cannot bypass the gate.
# ---------------------------------------------------------------------------


class FreshnessGateTests(unittest.TestCase):
    def _fresh_row(self, **overrides) -> dict:
        row = {
            "chain": "solana",
            "symbol": "TEST/SOL",
            "pair_address": "poolAAA111",
            "latest_price": 1.23,
            "price_updated_at": _utc_now_iso(),
            "latest_liquidity": 50000.0,
            "liquidity_updated_at": _utc_now_iso(),
            "source_provider": "dexscreener",
        }
        row.update(overrides)
        return row

    def test_missing_price_blocks(self) -> None:
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        gate = validate_market_data_gate(self._fresh_row(latest_price=None), for_open=True)
        self.assertFalse(gate["passed"])
        self.assertIn("freshness_missing_price", gate["blocking_guards"])
        self.assertEqual(gate["tradability_status"], "missing_price")

    def test_missing_liquidity_blocks(self) -> None:
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        gate = validate_market_data_gate(self._fresh_row(latest_liquidity=None), for_open=True)
        self.assertFalse(gate["passed"])
        self.assertIn("freshness_missing_liquidity", gate["blocking_guards"])

    def test_missing_provider_blocks(self) -> None:
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        gate = validate_market_data_gate(self._fresh_row(source_provider=None), for_open=True)
        self.assertFalse(gate["passed"])
        self.assertIn("freshness_missing_source_provider", gate["blocking_guards"])

    def test_stale_price_blocks(self) -> None:
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        row = self._fresh_row(
            price_updated_at=_iso_minutes_ago(60),
            price_age_seconds=3600.0,
        )
        gate = validate_market_data_gate(row, for_open=True)
        self.assertFalse(gate["passed"])
        self.assertIn("freshness_stale_price", gate["blocking_guards"])
        self.assertEqual(gate["freshness_gate_status"], "fail")

    def test_ambiguous_address_role_blocks(self) -> None:
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        row = self._fresh_row(pair_address="0xabc", token_mint_address="0xabc", chain="ethereum")
        gate = validate_market_data_gate(row, for_open=True)
        self.assertFalse(gate["passed"])
        self.assertIn(gate["address_role_status"], ("ambiguous",))

    def test_semantic_label_cannot_bypass_freshness_block(self) -> None:
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        row = self._fresh_row(
            latest_price=None,
            semantic_status="NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
            semantic_signal_family="NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
        )
        gate = validate_market_data_gate(row, for_open=True)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["semantic_status"], "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED")
        self.assertNotEqual(gate["decision"], "TRADABLE_NOW")

    def test_fully_fresh_row_passes(self) -> None:
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        gate = validate_market_data_gate(self._fresh_row(), for_open=True)
        self.assertTrue(gate["passed"], gate)
        self.assertEqual(gate["tradability_status"], "tradable_now")
        self.assertEqual(gate["decision"], "TRADABLE_NOW")


# ---------------------------------------------------------------------------
# 9: tradability_status must not be derived purely from "historical_seen"
# ---------------------------------------------------------------------------


class TradabilityNotFromHistoricalSeenAloneTests(unittest.TestCase):
    def test_historical_seen_alone_does_not_imply_tradable(self) -> None:
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        row = {
            "chain": "solana",
            "symbol": "OLD/SOL",
            "pair_address": "poolOLD111",
            # No latest_price/liquidity/provider -- only a "historical_seen" style flag.
            "historical_seen": True,
            "last_seen_in_market": "2020-01-01T00:00:00+00:00",
        }
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate as _v

        gate = _v(row, for_open=True)
        self.assertFalse(gate["passed"])
        self.assertNotEqual(gate["tradability_status"], "tradable_now")


# ---------------------------------------------------------------------------
# 10-12: address_role distinctions (pair/pool vs token mint/contract vs
#         provider pair id vs ambiguous)
# ---------------------------------------------------------------------------


class AddressRoleDistinctionTests(unittest.TestCase):
    def test_solana_pair_address_classified_as_pool_not_mint(self) -> None:
        from app.ae13b_product.address_role import classify_address_role

        result = classify_address_role(
            chain="solana",
            pair_address="9VW8yfZaf2GcEpVb4apuk63oGVnebYZ4pr7ymc8Ftx3i",
        )
        self.assertEqual(result["address_role"], "pool_address")
        self.assertFalse(result["is_ambiguous"])

    def test_evm_pair_address_classified_as_pair_contract(self) -> None:
        from app.ae13b_product.address_role import classify_address_role

        result = classify_address_role(
            chain="ethereum",
            pair_address="0xd2391dB4D7B9841b989521088c3Bf8C4cFe404d8",
        )
        self.assertEqual(result["address_role"], "pair_contract")

    def test_same_address_as_pair_and_mint_is_ambiguous_conflict(self) -> None:
        from app.ae13b_product.address_role import classify_address_role

        result = classify_address_role(
            chain="solana",
            pair_address="SAMEADDR111",
            token_mint_address="SAMEADDR111",
        )
        self.assertTrue(result["is_ambiguous"])
        self.assertTrue(result["pair_token_identity_conflict"])

    def test_provider_pair_id_hint_is_distinct_role(self) -> None:
        from app.ae13b_product.address_role import classify_address_role

        result = classify_address_role(
            chain="solana",
            provider_pair_id="dexscreener-123",
            hint="provider_pair_id",
        )
        self.assertEqual(result["address_role"], "provider_pair_id")


# ---------------------------------------------------------------------------
# 13: PnL must be null (not a fabricated number) when the mark price is stale
# ---------------------------------------------------------------------------


class PnlNullWhenStaleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        data_dir = Path(self._tmpdir.name) / "data"
        data_dir.mkdir(parents=True)
        os.environ["TRADER_DB_PATH"] = str(data_dir / "test.db")

        import app.database as database
        import app.execution.paper as paper

        importlib.reload(paper)
        importlib.reload(database)
        paper.DATA_DIR = data_dir
        paper.STATE_PATH = data_dir / "paper_state.json"
        paper.TRADES_LOG_PATH = data_dir / "paper_trades_log.csv"
        database.DATA_DIR = data_dir
        database.DB_PATH = data_dir / "test.db"
        database.init_db()

        self.paper = paper
        self.trader = paper.PaperTrader()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        os.environ.pop("TRADER_DB_PATH", None)

    def test_pnl_is_none_when_mark_price_stale(self) -> None:
        self.trader.set_market_prices(
            [{"pair_address": "pool_stale_1", "coin_id": 900, "price_usd": 1.0}],
            price_timestamp=_utc_now_iso(),
        )
        pos = self.trader.open_position(
            {
                "symbol": "STALE/SOL",
                "chain": "solana",
                "pair_address": "pool_stale_1",
                "coin_id": 900,
                "latest_price": 1.0,
                "price_updated_at": _utc_now_iso(),
                "latest_liquidity": 50000.0,
                "liquidity_updated_at": _utc_now_iso(),
                "source_provider": "dexscreener",
                "activity_delta_1h_pct": 5.0,
            },
            size_usd=10.0,
            settings={},
            reason_code="TEST",
        )
        self.assertIsNotNone(pos)
        # Push the market price timestamp far into the past -> stale mark.
        self.trader.set_market_prices(
            [{"pair_address": "pool_stale_1", "coin_id": 900, "price_usd": 1.5}],
            price_timestamp=_iso_minutes_ago(60),
        )
        marked = self.trader.get_marked_positions()
        row = next(p for p in marked if p["id"] == pos["id"])
        self.assertIsNone(row["unrealized_pnl_usd"])
        self.assertIsNone(row["unrealized_pnl_pct"])

    def test_pnl_is_present_when_mark_price_fresh(self) -> None:
        self.trader.set_market_prices(
            [{"pair_address": "pool_fresh_1", "coin_id": 901, "price_usd": 1.0}],
            price_timestamp=_utc_now_iso(),
        )
        pos = self.trader.open_position(
            {
                "symbol": "FRESH/SOL",
                "chain": "solana",
                "pair_address": "pool_fresh_1",
                "coin_id": 901,
                "latest_price": 1.0,
                "price_updated_at": _utc_now_iso(),
                "latest_liquidity": 50000.0,
                "liquidity_updated_at": _utc_now_iso(),
                "source_provider": "dexscreener",
                "activity_delta_1h_pct": 5.0,
            },
            size_usd=10.0,
            settings={},
            reason_code="TEST",
        )
        self.assertIsNotNone(pos)
        self.trader.set_market_prices(
            [{"pair_address": "pool_fresh_1", "coin_id": 901, "price_usd": 1.1}],
            price_timestamp=_utc_now_iso(),
        )
        marked = self.trader.get_marked_positions()
        row = next(p for p in marked if p["id"] == pos["id"])
        self.assertIsNotNone(row["unrealized_pnl_usd"])


# ---------------------------------------------------------------------------
# 14-15: manual close metadata
# ---------------------------------------------------------------------------


class ManualCloseMetadataTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        data_dir = Path(self._tmpdir.name) / "data"
        data_dir.mkdir(parents=True)
        os.environ["TRADER_DB_PATH"] = str(data_dir / "test.db")

        import app.ae13b_product.reentry_blocks as reentry_blocks
        import app.database as database
        import app.execution.paper as paper

        importlib.reload(paper)
        importlib.reload(database)
        paper.DATA_DIR = data_dir
        paper.STATE_PATH = data_dir / "paper_state.json"
        paper.TRADES_LOG_PATH = data_dir / "paper_trades_log.csv"
        database.DATA_DIR = data_dir
        database.DB_PATH = data_dir / "test.db"
        database.init_db()

        # Isolate reentry-block persistence too, so this test never writes a
        # manual-close cooldown into the real project's
        # data/runtime/reentry_blocks.json (which would leak across test runs
        # and eventually block real pair_addresses/symbols).
        self._reentry_blocks = reentry_blocks
        self._orig_blocks_path = reentry_blocks.blocks_file_path
        reentry_blocks.blocks_file_path = lambda: data_dir / "reentry_blocks.json"  # type: ignore[assignment]

        self.paper = paper
        self.trader = paper.PaperTrader()
        self.trader.set_market_prices(
            [{"pair_address": "pool_manual_1", "coin_id": 902, "price_usd": 1.0}],
            price_timestamp=_utc_now_iso(),
        )

    def tearDown(self) -> None:
        self._reentry_blocks.blocks_file_path = self._orig_blocks_path
        self._tmpdir.cleanup()
        os.environ.pop("TRADER_DB_PATH", None)

    def test_manual_close_records_metadata_and_flags(self) -> None:
        pos = self.trader.open_position(
            {
                "symbol": "MANUAL/SOL",
                "chain": "solana",
                "pair_address": "pool_manual_1",
                "coin_id": 902,
                "latest_price": 1.0,
                "price_updated_at": _utc_now_iso(),
                "latest_liquidity": 50000.0,
                "liquidity_updated_at": _utc_now_iso(),
                "source_provider": "dexscreener",
                "activity_delta_1h_pct": 5.0,
            },
            size_usd=10.0,
            settings={},
            reason_code="TEST",
        )
        self.assertIsNotNone(pos)
        closed = self.trader.close_position(
            int(pos["id"]),
            1.05,
            reason_code="MANUAL_SELL",
            proposed_pair_address="pool_manual_1",
            proposed_coin_id=902,
            close_reason="manual_take_profit",
            close_note="ae13i unit test",
            closed_by="user_manual",
        )
        self.assertIsNotNone(closed)
        assert closed is not None
        self.assertEqual(closed["closed_by"], "user_manual")
        self.assertEqual(closed["close_reason"], "manual_take_profit")
        self.assertEqual(closed["close_note"], "ae13i unit test")
        self.assertTrue(closed["manual_close"])
        self.assertIn("close_freshness_status", closed)
        self.assertTrue(closed["paper_demo_only"])
        self.assertTrue(closed["not_live_approved"])
        self.assertTrue(closed["not_profitability_evidence"])


# ---------------------------------------------------------------------------
# 16-17: persistent reentry block survives reload
# ---------------------------------------------------------------------------


class PersistentReentryBlockSurvivesReloadTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._orig_cwd_root = None

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_manual_close_block_persists_and_reloads_from_disk(self) -> None:
        import app.ae13b_product.reentry_blocks as reentry_blocks

        tmp_path = Path(self._tmpdir.name) / "reentry_blocks.json"
        original_path_fn = reentry_blocks.blocks_file_path
        reentry_blocks.blocks_file_path = lambda: tmp_path  # type: ignore[assignment]
        try:
            position = {
                "id": 999,
                "pair_address": "pool_reentry_111",
                "chain": "solana",
                "symbol": "REENTRY/SOL",
                "exit_price": 1.0,
            }
            block = reentry_blocks.add_manual_close_block(position, "manual_take_profit", duration_seconds=3600)
            self.assertTrue(tmp_path.exists())
            self.assertEqual(block["block_kind"], "manual_close")

            # Simulate a process restart by reloading the module fresh and
            # re-pointing it at the same on-disk file.
            importlib.reload(reentry_blocks)
            reentry_blocks.blocks_file_path = lambda: tmp_path  # type: ignore[assignment]

            found = reentry_blocks.check_reentry_block(
                pair_address="pool_reentry_111", chain="solana", symbol="REENTRY/SOL",
            )
            self.assertIsNotNone(found)
            assert found is not None
            self.assertTrue(found["active"])
            self.assertEqual(found["block_kind"], "manual_close")
        finally:
            reentry_blocks.blocks_file_path = original_path_fn

    def test_expired_block_does_not_block(self) -> None:
        import app.ae13b_product.reentry_blocks as reentry_blocks

        tmp_path = Path(self._tmpdir.name) / "reentry_blocks_expired.json"
        original_path_fn = reentry_blocks.blocks_file_path
        reentry_blocks.blocks_file_path = lambda: tmp_path  # type: ignore[assignment]
        try:
            position = {"id": 998, "pair_address": "pool_expired_111", "chain": "solana", "symbol": "EXP/SOL"}
            reentry_blocks.add_manual_close_block(position, "manual_take_profit", duration_seconds=1)

            # Force the stored block into the past rather than sleeping in a
            # unit test -- directly rewrite its expires_at_utc on disk.
            store = reentry_blocks.load_blocks()
            past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
            for block in store.get("blocks") or []:
                block["expires_at_utc"] = past
                block["expires_at"] = past
            reentry_blocks.save_blocks(store)

            found = reentry_blocks.check_reentry_block(
                pair_address="pool_expired_111", chain="solana", symbol="EXP/SOL",
            )
            self.assertIsNone(found)
        finally:
            reentry_blocks.blocks_file_path = original_path_fn


# ---------------------------------------------------------------------------
# 18-20: watchlist cooldown + demo queue precheck + bot cannot reopen via gate
# ---------------------------------------------------------------------------


class ReentryCooldownPrecheckTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_gatekeeper_blocks_reentry_during_manual_cooldown(self) -> None:
        import app.ae13b_product.reentry_blocks as reentry_blocks
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        tmp_path = Path(self._tmpdir.name) / "reentry_blocks.json"
        original_path_fn = reentry_blocks.blocks_file_path
        reentry_blocks.blocks_file_path = lambda: tmp_path  # type: ignore[assignment]
        try:
            position = {
                "id": 997,
                "pair_address": "pool_cooldown_111",
                "chain": "solana",
                "symbol": "COOL/SOL",
                "exit_price": 1.0,
            }
            reentry_blocks.add_manual_close_block(position, "manual_take_profit", duration_seconds=3600)

            row = {
                "chain": "solana",
                "symbol": "COOL/SOL",
                "pair_address": "pool_cooldown_111",
                "latest_price": 1.0,
                "price_updated_at": _utc_now_iso(),
                "latest_liquidity": 50000.0,
                "liquidity_updated_at": _utc_now_iso(),
                "source_provider": "dexscreener",
            }
            gate = validate_market_data_gate(row, for_open=True)
            self.assertFalse(gate["passed"])
            self.assertIn("manual_reentry_block", gate["blocking_guards"])
            self.assertEqual(gate["rejection_code"], "MANUAL_REENTRY_BLOCK_ACTIVE")
            self.assertEqual(gate["decision"], "BLOCKED_MANUAL_REENTRY_COOLDOWN")
        finally:
            reentry_blocks.blocks_file_path = original_path_fn

    def test_manual_cooldown_fields_helper_used_by_watchlist_and_queue(self) -> None:
        watchlist_src = (ROOT / "app" / "analytics" / "watchlist.py").read_text(encoding="utf-8")
        queue_src = (ROOT / "app" / "ae13b_product" / "demo_queue.py").read_text(encoding="utf-8")
        self.assertIn("get_manual_cooldown_fields", watchlist_src)
        self.assertIn("get_manual_cooldown_fields", queue_src)

    def test_bot_buy_path_runs_through_shared_gatekeeper(self) -> None:
        # Point 20: the bot cannot reopen a reentry-blocked pair because its
        # buy path routes through the same validate_market_data_gate() as
        # watchlist/queue, rather than a separate ad-hoc check.
        demo_bot_src = (ROOT / "app" / "ae13b_product" / "demo_bot.py").read_text(encoding="utf-8")
        self.assertIn("validate_market_data_gate", demo_bot_src)
        self.assertIn("skip_stagnant=False", demo_bot_src)


# ---------------------------------------------------------------------------
# 21-22: system reentry needs a new, meaningful signal
# ---------------------------------------------------------------------------


class SystemReentrySignalTests(unittest.TestCase):
    def test_no_new_signal_blocks(self) -> None:
        from app.ae13b_product.system_reentry_signal import check_system_reentry_signal

        candidate = {"latest_price": 1.0, "latest_volume_24h": 10000.0, "latest_liquidity": 50000.0}
        close_snapshot = {"price": 1.0005, "volume_24h": 10001.0, "liquidity": 50000.5}
        result = check_system_reentry_signal(candidate, close_snapshot)
        self.assertFalse(result["passed"])
        self.assertEqual(result["rejection_code"], "REENTRY_BLOCK_NO_NEW_SIGNAL")

    def test_meaningful_price_move_passes(self) -> None:
        from app.ae13b_product.system_reentry_signal import check_system_reentry_signal

        candidate = {"latest_price": 1.10, "latest_volume_24h": 10000.0, "latest_liquidity": 50000.0}
        close_snapshot = {"price": 1.00, "volume_24h": 10000.0, "liquidity": 50000.0}
        result = check_system_reentry_signal(candidate, close_snapshot)
        self.assertTrue(result["passed"])
        self.assertTrue(result["new_signal_detected"])

    def test_no_snapshot_skips_check(self) -> None:
        from app.ae13b_product.system_reentry_signal import check_system_reentry_signal

        result = check_system_reentry_signal({"latest_price": 1.0}, None)
        self.assertTrue(result["passed"])


# ---------------------------------------------------------------------------
# 23-24: stagnant price guard (AE13I Fix A)
# ---------------------------------------------------------------------------


class StagnantPriceGuardTests(unittest.TestCase):
    def test_missing_all_deltas_does_not_block(self) -> None:
        from app.ae13b_product.stagnant_price_guard import (
            MOMENTUM_EVIDENCE_UNKNOWN,
            evaluate_stagnant_price,
        )

        result = evaluate_stagnant_price({"symbol": "NEW/SOL"})
        self.assertTrue(result["passed"])
        self.assertEqual(result["momentum_evidence"], MOMENTUM_EVIDENCE_UNKNOWN)
        self.assertEqual(result["blocking_guards"], [])

    def test_low_4h_delta_blocks(self) -> None:
        from app.ae13b_product.stagnant_price_guard import evaluate_stagnant_price

        result = evaluate_stagnant_price({"activity_delta_4h_pct": 0.1})
        self.assertFalse(result["passed"])
        self.assertEqual(result["rejection_code"], "PRICE_STAGNANT_NO_RECENT_MOMENTUM")
        self.assertIn("stagnant_price_guard", result["blocking_guards"])
        self.assertIn("no_recent_momentum", result["blocking_guards"])

    def test_low_1h_delta_with_no_4h_blocks(self) -> None:
        from app.ae13b_product.stagnant_price_guard import evaluate_stagnant_price

        result = evaluate_stagnant_price({"activity_delta_1h_pct": 0.1})
        self.assertFalse(result["passed"])
        self.assertIn("stagnant_price_guard_1h", result["blocking_guards"])

    def test_healthy_momentum_passes(self) -> None:
        from app.ae13b_product.stagnant_price_guard import evaluate_stagnant_price

        result = evaluate_stagnant_price({"activity_delta_1h_pct": 5.0, "activity_delta_4h_pct": 8.0})
        self.assertTrue(result["passed"])
        self.assertEqual(result["momentum_evidence"], "recent_momentum_present")

    def test_fresh_catalyst_bypasses_low_momentum(self) -> None:
        from app.ae13b_product.stagnant_price_guard import evaluate_stagnant_price

        result = evaluate_stagnant_price(
            {"activity_delta_4h_pct": 0.05, "fresh_whale_signal": True},
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["fresh_catalyst_bypass"])

    def test_allow_stagnant_buy_only_with_fresh_catalyst_flag_can_disable_bypass(self) -> None:
        from app.ae13b_product.stagnant_price_guard import evaluate_stagnant_price

        result = evaluate_stagnant_price(
            {"activity_delta_4h_pct": 0.05, "fresh_catalyst": True},
            allow_stagnant_buy_only_with_fresh_catalyst=False,
        )
        self.assertFalse(result["passed"])

    def test_volume_spike_counts_as_catalyst(self) -> None:
        from app.ae13b_product.stagnant_price_guard import evaluate_stagnant_price

        result = evaluate_stagnant_price({"activity_delta_1h_pct": 0.1, "volume_spike": True})
        self.assertTrue(result["passed"])

    def test_extra_window_fields_recognized_as_present(self) -> None:
        from app.ae13b_product.stagnant_price_guard import evaluate_stagnant_price

        result = evaluate_stagnant_price({"price_change_6h": 10.0, "change_24h": 12.0})
        self.assertTrue(result["passed"])
        self.assertNotEqual(result.get("momentum_evidence"), "unknown_insufficient_delta_fields")

    def test_gatekeeper_runs_stagnant_guard_by_default(self) -> None:
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        row = {
            "chain": "solana",
            "symbol": "FLAT/SOL",
            "pair_address": "pool_flat_111",
            "latest_price": 1.0,
            "price_updated_at": _utc_now_iso(),
            "latest_liquidity": 50000.0,
            "liquidity_updated_at": _utc_now_iso(),
            "source_provider": "dexscreener",
            "activity_delta_4h_pct": 0.05,
        }
        gate = validate_market_data_gate(row, for_open=True)  # skip_stagnant defaults False
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["rejection_code"], "PRICE_STAGNANT_NO_RECENT_MOMENTUM")

    def test_gatekeeper_does_not_blackout_rows_with_no_deltas(self) -> None:
        """Coins-table rows with no per-row deltas must not be hard-blocked."""
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        row = {
            "chain": "solana",
            "symbol": "NODELTA/SOL",
            "pair_address": "pool_nodelta_111",
            "latest_price": 1.0,
            "price_updated_at": _utc_now_iso(),
            "latest_liquidity": 50000.0,
            "liquidity_updated_at": _utc_now_iso(),
            "source_provider": "dexscreener",
        }
        gate = validate_market_data_gate(row, for_open=True)
        self.assertTrue(gate["passed"], gate)


class NoRemainingSkipStagnantTrueCallSitesTests(unittest.TestCase):
    def test_no_skip_stagnant_true_left_in_call_sites(self) -> None:
        for rel_path in (
            "app/ae13b_product/demo_bot.py",
            "app/ae13b_product/demo_queue.py",
            "app/analytics/watchlist.py",
            "app/execution/paper.py",
        ):
            src = (ROOT / rel_path).read_text(encoding="utf-8")
            self.assertNotIn("skip_stagnant=True", src, msg=f"{rel_path} still hard-disables the stagnant guard")


# ---------------------------------------------------------------------------
# 25: RiskGuard blockers include reentry/stagnation/freshness when merged
# ---------------------------------------------------------------------------


class RiskGuardMergesGateBlockersTests(unittest.TestCase):
    def test_gate_result_blockers_merged_into_risk_guard_output(self) -> None:
        from app.ae13b_product.demo_risk_guard import evaluate_demo_risk_guard

        gate_result = {
            "passed": False,
            "rejection_code": "PRICE_STAGNANT_NO_RECENT_MOMENTUM",
            "blocking_guards": ["stagnant_price_guard", "no_recent_momentum"],
            "rejection_reasons": ["No recent price momentum detected."],
        }
        result = evaluate_demo_risk_guard(
            requested_notional=25.0,
            demo_equity=10000.0,
            open_positions=[],
            recent_trades=[],
            pair_address="pool_x",
            symbol="X/SOL",
            chain="solana",
            price=1.0,
            gate_result=gate_result,
        )
        self.assertFalse(result.get("passed", True))
        guards = result.get("blocking_guards") or []
        self.assertTrue(
            any("stagnant" in g for g in guards) or any("stagnant" in g for g in (result.get("rejection_reasons") or [])),
            msg=result,
        )


# ---------------------------------------------------------------------------
# 26-29: repair script backup + idempotent
# ---------------------------------------------------------------------------


class RepairRiskBlockSchemaScriptTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_root = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def _write_sample_csv(self, path: Path) -> None:
        header = [
            "timestamp", "position_id", "symbol", "chain", "side", "quantity",
            "fill_price", "notional_usd", "swap_fee", "priority_fee", "total_fees",
            "gross_pnl", "realized_pnl", "net_roi_pct", "cluster_label", "reason_code",
            "coin_id", "pair_address", "decision_ref_id", "fill_price_source",
            "market_price_usd", "price_timestamp", "cash_before", "equity_before",
            "notional_requested", "notional_executed", "rejection_reason",
            "rejection_reasons", "blocking_guards", "rejection_code", "strategy_lane",
            "preset_id", "risk_mode", "event_type", "pair", "closed_by", "close_reason",
            "close_note", "paper_demo_only", "not_live_approved",
            "not_profitability_evidence", "manual_close", "close_price_age_seconds",
            "close_freshness_status", "close_used_fallback_price",
            "manual_close_warning_shown",
        ]
        # A malformed, column-shifted RISK_GUARD_BLOCK row (marker literal
        # lands under coin_id, matching the production corruption pattern).
        malformed_row = [
            "2026-07-01T00:00:00+00:00", "", "", "", "WIF/SOL", "buy", "solana",
            "0", "0", "", "", "", "0", "", "", "", "RISK_GUARD_BLOCK", "",
        ] + [""] * (len(header) - 18)
        # A normal, well-formed buy fill row that must be left untouched.
        normal_row = [
            "2026-07-01T00:05:00+00:00", "1", "GOOD/SOL", "solana", "buy", "100",
            "1.0", "100.0", "1.5", "0.03", "1.53", "0", "0", "0",
            "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED", "DEMO_STRATEGY_ENTRY", "5",
            "poolGOOD111",
        ] + [""] * (len(header) - 17)

        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(malformed_row)
            writer.writerow(normal_row)

    def test_repair_backs_up_and_repairs_malformed_rows(self) -> None:
        import scripts.repair_risk_block_schema as repair_mod

        csv_path = self.tmp_root / "paper_trades_log.csv"
        self._write_sample_csv(csv_path)
        backups_root = self.tmp_root / "backups"
        reports_root = self.tmp_root / "reports"

        report = repair_mod.run(
            csv_path=csv_path,
            state_path=self.tmp_root / "paper_state.json",
            backups_root=backups_root,
            reports_root=reports_root,
        )
        self.assertEqual(report["csv_repair"]["rows_malformed_found"], 1)
        self.assertTrue(report["csv_repair"]["wrote_changes"])
        self.assertIsNotNone(report["backup"]["backup_dir"])
        self.assertTrue(Path(report["backup"]["backup_dir"]).exists())
        backed_up_csv = Path(report["backup"]["backup_dir"]) / "paper_trades_log.csv"
        self.assertTrue(backed_up_csv.exists())

        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        repaired = [r for r in rows if r.get("repaired_by") == "repair_risk_block_schema.py"]
        self.assertEqual(len(repaired), 1)
        self.assertEqual(repaired[0]["symbol"], "WIF/SOL")
        self.assertEqual(repaired[0]["chain"], "solana")
        self.assertEqual(repaired[0]["event_type"], "RISK_GUARD_BLOCK")
        self.assertEqual(repaired[0]["legacy_malformed"], "True")
        self.assertTrue(repaired[0]["legacy_raw_record"])

        untouched = [r for r in rows if r.get("symbol") == "GOOD/SOL"]
        self.assertEqual(len(untouched), 1)
        self.assertNotEqual(untouched[0].get("repaired_by"), "repair_risk_block_schema.py")

    def test_repair_is_idempotent_on_second_run(self) -> None:
        import scripts.repair_risk_block_schema as repair_mod

        csv_path = self.tmp_root / "paper_trades_log.csv"
        self._write_sample_csv(csv_path)
        backups_root = self.tmp_root / "backups"
        reports_root = self.tmp_root / "reports"

        first = repair_mod.run(
            csv_path=csv_path, state_path=self.tmp_root / "paper_state.json",
            backups_root=backups_root, reports_root=reports_root,
        )
        self.assertEqual(first["csv_repair"]["rows_repaired"], 1)

        after_first = csv_path.read_text(encoding="utf-8")

        second = repair_mod.run(
            csv_path=csv_path, state_path=self.tmp_root / "paper_state.json",
            backups_root=backups_root, reports_root=reports_root,
        )
        self.assertEqual(second["csv_repair"]["rows_malformed_found"], 0)
        self.assertFalse(second["csv_repair"]["wrote_changes"])
        self.assertIsNone(second["backup"]["backup_dir"])

        after_second = csv_path.read_text(encoding="utf-8")
        self.assertEqual(after_first, after_second)

    def test_repair_handles_missing_csv_gracefully(self) -> None:
        import scripts.repair_risk_block_schema as repair_mod

        report = repair_mod.run(
            csv_path=self.tmp_root / "does_not_exist.csv",
            state_path=self.tmp_root / "paper_state.json",
            backups_root=self.tmp_root / "backups",
            reports_root=self.tmp_root / "reports",
        )
        self.assertFalse(report["csv_repair"]["csv_found"])


# ---------------------------------------------------------------------------
# 30: traffic light
# ---------------------------------------------------------------------------


class TrafficLightTests(unittest.TestCase):
    def test_missing_price_is_red(self) -> None:
        from app.ae13b_product.mtm_traffic_light import compute_traffic_light

        result = compute_traffic_light({"current_price": None})
        self.assertEqual(result["traffic_light_status"], "red")

    def test_ambiguous_address_is_red(self) -> None:
        from app.ae13b_product.mtm_traffic_light import compute_traffic_light

        result = compute_traffic_light({"current_price": 1.0, "address_role_status": "ambiguous"})
        self.assertEqual(result["traffic_light_status"], "red")

    def test_take_profit_reached_is_green(self) -> None:
        from app.ae13b_product.mtm_traffic_light import compute_traffic_light

        result = compute_traffic_light({"current_price": 2.0, "take_profit": 1.5})
        self.assertEqual(result["traffic_light_status"], "green")

    def test_fresh_waiting_position_is_yellow(self) -> None:
        from app.ae13b_product.mtm_traffic_light import compute_traffic_light

        result = compute_traffic_light(
            {"current_price": 1.0, "take_profit": 5.0, "stop_loss": 0.1, "unrealized_pnl_pct": None}
        )
        self.assertEqual(result["traffic_light_status"], "yellow")


# ---------------------------------------------------------------------------
# 31: retrospective JSON covers required addresses
# ---------------------------------------------------------------------------


class RetrospectiveDecisionTraceTests(unittest.TestCase):
    def test_retrospective_trace_file_exists_and_covers_required_addresses(self) -> None:
        path = ROOT / "data" / "ae13i_retrospective_decision_trace.json"
        self.assertTrue(path.exists(), msg="data/ae13i_retrospective_decision_trace.json is missing")
        data = json.loads(path.read_text(encoding="utf-8"))
        blob = json.dumps(data)
        for needle in (
            "9VW8yfZaf2GcEpVb4apuk63oGVnebYZ4pr7ymc8Ftx3i",
            "0xd2391dB4D7B9841b989521088c3Bf8C4cFe404d8",
            "0x20d6015660b3fe52e6690a889b5c51f69902ce0e",
            "817",
            "1626",
        ):
            self.assertIn(needle, blob)
        self.assertIn("pre_AE13I_provenance_not_guaranteed", blob)

    def test_retrospective_trace_never_invents_timestamps_marker_present(self) -> None:
        path = ROOT / "data" / "ae13i_retrospective_decision_trace.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn("methodology_notes", data)
        self.assertTrue(any("invented" in note.lower() for note in data["methodology_notes"]))


# ---------------------------------------------------------------------------
# 32: bot activity summary helpers (if testable)
# ---------------------------------------------------------------------------


class BotActivitySummaryTests(unittest.TestCase):
    def test_demo_bot_status_returns_safety_flagged_summary(self) -> None:
        from app.ae13b_product.demo_bot import get_demo_bot, reset_demo_bot_for_tests

        reset_demo_bot_for_tests()
        bot = get_demo_bot()
        status = bot.status()
        self.assertIn("demo_mode_active", status)
        self.assertIn("live_trading_disabled", status)
        self.assertIn("wallet_not_connected", status)
        self.assertTrue(status.get("demo_mode_active"))
        self.assertTrue(status.get("live_trading_disabled"))
        reset_demo_bot_for_tests()


# ---------------------------------------------------------------------------
# 33: live market keyed update preserved in JS
# ---------------------------------------------------------------------------


class LiveMarketKeyedUpdateStaticTests(unittest.TestCase):
    def test_render_live_market_keyed_present(self) -> None:
        js = (ROOT / "static" / "product_demo.js").read_text(encoding="utf-8")
        self.assertIn("renderLiveMarketKeyed", js)


# ---------------------------------------------------------------------------
# 34: safety flags
# ---------------------------------------------------------------------------


class SafetyFlagsTests(unittest.TestCase):
    def test_gate_result_has_no_live_trading_signal(self) -> None:
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        row = {
            "chain": "solana",
            "symbol": "SAFE/SOL",
            "pair_address": "pool_safe_111",
            "latest_price": 1.0,
            "price_updated_at": _utc_now_iso(),
            "latest_liquidity": 50000.0,
            "liquidity_updated_at": _utc_now_iso(),
            "source_provider": "dexscreener",
        }
        gate = validate_market_data_gate(row, for_open=True)
        self.assertIn("candidate_context", gate)
        self.assertTrue(gate["candidate_context"].get("paper_demo_only"))
        self.assertTrue(gate["candidate_context"].get("not_live_approved"))

    def test_repair_script_never_touches_wallet_or_live_paths(self) -> None:
        src = (ROOT / "scripts" / "repair_risk_block_schema.py").read_text(encoding="utf-8")
        self.assertNotRegex(src, r"private_key|signTransaction|sendRawTransaction|live_wallet")
        self.assertIn("Paper/demo only", src)

    def test_stagnant_guard_module_has_no_live_execution_path(self) -> None:
        src = (ROOT / "app" / "ae13b_product" / "stagnant_price_guard.py").read_text(encoding="utf-8")
        self.assertNotRegex(src, r"private_key|signTransaction|sendRawTransaction")


# ---------------------------------------------------------------------------
# Fix G: /api/trades legacy_malformed filter + manual close reentry block
# ---------------------------------------------------------------------------


class ApiTradesLegacyMalformedFilterTests(unittest.TestCase):
    def test_legacy_malformed_row_detected_and_hidden_by_default(self) -> None:
        from app.api import _alias_trade_row_for_ui, _is_legacy_malformed_trade_row

        malformed = {"event_type": "RISK_GUARD_BLOCK", "reason_code": "RISK_GUARD_BLOCK", "rejection_code": ""}
        self.assertTrue(_is_legacy_malformed_trade_row(malformed))

        structured = {"event_type": "RISK_GUARD_BLOCK", "rejection_code": "PRICE_STAGNANT_NO_RECENT_MOMENTUM"}
        self.assertFalse(_is_legacy_malformed_trade_row(structured))

        aliased = _alias_trade_row_for_ui(malformed)
        self.assertTrue(aliased["legacy_malformed"])

    def test_list_trades_hides_legacy_malformed_by_default(self) -> None:
        import app.api as api_mod

        rows = [
            {"event_type": "RISK_GUARD_BLOCK", "reason_code": "RISK_GUARD_BLOCK", "rejection_code": "", "symbol": "OLD/SOL"},
            {"event_type": "BUY", "reason_code": "DEMO_STRATEGY_ENTRY", "rejection_code": "", "symbol": "GOOD/SOL"},
        ]
        aliased = [api_mod._alias_trade_row_for_ui(r) for r in rows]
        visible_default = [r for r in aliased if not r.get("legacy_malformed")]
        self.assertEqual(len(visible_default), 1)
        self.assertEqual(visible_default[0]["symbol"], "GOOD/SOL")


class ManualCloseCreatesReentryBlockTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        data_dir = Path(self._tmpdir.name) / "data"
        data_dir.mkdir(parents=True)
        os.environ["TRADER_DB_PATH"] = str(data_dir / "test.db")

        import app.ae13b_product.reentry_blocks as reentry_blocks
        import app.database as database
        import app.execution.paper as paper

        importlib.reload(paper)
        importlib.reload(database)
        paper.DATA_DIR = data_dir
        paper.STATE_PATH = data_dir / "paper_state.json"
        paper.TRADES_LOG_PATH = data_dir / "paper_trades_log.csv"
        database.DATA_DIR = data_dir
        database.DB_PATH = data_dir / "test.db"
        database.init_db()

        self._reentry_blocks = reentry_blocks
        self._orig_blocks_path = reentry_blocks.blocks_file_path
        reentry_blocks.blocks_file_path = lambda: data_dir / "reentry_blocks.json"  # type: ignore[assignment]

        self.paper = paper
        self.trader = paper.PaperTrader()
        self.trader.set_market_prices(
            [{"pair_address": "pool_g_111", "coin_id": 903, "price_usd": 1.0}],
            price_timestamp=_utc_now_iso(),
        )

    def tearDown(self) -> None:
        self._reentry_blocks.blocks_file_path = self._orig_blocks_path
        self._tmpdir.cleanup()
        os.environ.pop("TRADER_DB_PATH", None)

    def test_manual_close_creates_reentry_block(self) -> None:
        pos = self.trader.open_position(
            {
                "symbol": "GBLK/SOL",
                "chain": "solana",
                "pair_address": "pool_g_111",
                "coin_id": 903,
                "latest_price": 1.0,
                "price_updated_at": _utc_now_iso(),
                "latest_liquidity": 50000.0,
                "liquidity_updated_at": _utc_now_iso(),
                "source_provider": "dexscreener",
                "activity_delta_1h_pct": 5.0,
            },
            size_usd=10.0,
            settings={},
            reason_code="TEST",
        )
        self.assertIsNotNone(pos)
        self.trader.close_position(
            int(pos["id"]),
            1.05,
            reason_code="MANUAL_SELL",
            proposed_pair_address="pool_g_111",
            proposed_coin_id=903,
            close_reason="manual_take_profit",
            closed_by="user_manual",
        )
        block = self._reentry_blocks.check_reentry_block(
            pair_address="pool_g_111", chain="solana", symbol="GBLK/SOL",
        )
        self.assertIsNotNone(block)
        assert block is not None
        self.assertEqual(block["block_kind"], "manual_close")


if __name__ == "__main__":
    unittest.main()
