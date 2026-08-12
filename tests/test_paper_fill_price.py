"""Paper fill-price resolver and accounting guard tests."""
from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from app.execution.fill_price import (
    resolve_buy_fill_price,
    resolve_sell_fill_price,
)
from app.execution.paper_audit import (
    audit_trade_rows,
    detect_doge_style_pattern,
    portfolio_roi_from_equity,
)
from app.training.baseline_model import (
    assert_chronological_splits,
    chronological_split,
    select_feature_columns,
)
from app.training.tabicl_v2_eval import (
    assert_chronological_splits as tab_assert_splits,
    chronological_split as tab_chronological_split,
    select_tabicl_feature_columns,
    select_rolling_temporal_indices,
)


class FillPriceResolverTests(unittest.TestCase):
    def test_rejects_missing_pair_address(self) -> None:
        result = resolve_buy_fill_price({"symbol": "DOGE/USDC", "coin_id": 1})
        self.assertFalse(result.ok)
        self.assertEqual(result.rejection_reason, "missing_pair_address")

    def test_rejects_missing_market_price_without_fallback(self) -> None:
        result = resolve_buy_fill_price(
            {
                "symbol": "DOGE/USDC",
                "pair_address": "pair_a",
                "coin_id": 1,
                "price_usd": 8.6e-05,
            },
            allow_coin_price_fallback=False,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.rejection_reason, "missing_market_price_for_pair")

    def test_buy_uses_pair_keyed_market_price(self) -> None:
        result = resolve_buy_fill_price(
            {
                "symbol": "DOGE/USDC",
                "pair_address": "pair_a",
                "coin_id": 1,
                "price_usd": 8.6e-05,
            },
            market_prices_by_pair={"pair_a": 0.2839},
        )
        self.assertTrue(result.ok)
        self.assertAlmostEqual(result.price or 0, 0.2839)

    def test_sell_rejects_symbol_only_price_from_other_pair(self) -> None:
        position = {
            "pair_address": "pair_scam",
            "coin_id": 10,
            "entry_price": 8.6e-05,
            "quantity": 1000,
        }
        result = resolve_sell_fill_price(
            position,
            market_prices_by_pair={
                "pair_scam": 8.6e-05,
                "pair_real": 0.2839,
            },
            proposed_price=0.2839,
            proposed_pair_address="pair_real",
            proposed_coin_id=99,
        )
        self.assertFalse(result.ok)
        self.assertEqual(result.rejection_reason, "sell_pair_address_mismatch")

    def test_sell_uses_open_position_pair_price(self) -> None:
        position = {
            "pair_address": "pair_scam",
            "coin_id": 10,
            "entry_price": 8.6e-05,
            "quantity": 1000,
        }
        result = resolve_sell_fill_price(
            position,
            market_prices_by_pair={"pair_scam": 8.7e-05},
        )
        self.assertTrue(result.ok)
        self.assertAlmostEqual(result.price or 0, 8.7e-05)

    def test_sell_rejects_large_deviation_from_entry(self) -> None:
        position = {
            "pair_address": "pair_a",
            "coin_id": 1,
            "entry_price": 0.10,
            "quantity": 10,
        }
        result = resolve_sell_fill_price(
            position,
            market_prices_by_pair={"pair_a": 0.2839},
        )
        self.assertFalse(result.ok)
        self.assertIn("price_deviation", result.rejection_reason or "")


class PaperAuditTests(unittest.TestCase):
    def test_detects_doge_style_pattern(self) -> None:
        buy = {"fill_price": "8.6e-05", "notional_usd": "1455.4"}
        sell = {"fill_price": "0.2839", "notional_usd": "4804512"}
        self.assertTrue(detect_doge_style_pattern(buy, sell))

    def test_portfolio_roi_from_equity(self) -> None:
        roi = portfolio_roi_from_equity(current_equity=12_000, starting_capital=10_000)
        self.assertAlmostEqual(roi, 0.2)


class PaperTraderGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        data_dir = Path(self._tmpdir.name) / "data"
        data_dir.mkdir(parents=True)
        os.environ["TRADER_DB_PATH"] = str(data_dir / "test.db")

        import importlib
        import app.execution.paper as paper
        import app.database as database

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

    def test_rejects_buy_without_market_price(self) -> None:
        pos = self.trader.open_position({
            "symbol": "DOGE/USDC",
            "chain": "solana",
            "pair_address": "pair_a",
            "coin_id": 1,
            "price_usd": 0.2839,
        })
        self.assertIsNone(pos)

    def test_rejects_cross_pair_sell_price(self) -> None:
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).isoformat()
        self.trader.set_market_prices([
            {"pair_address": "pair_a", "coin_id": 1, "price_usd": 0.000086},
        ], price_timestamp=ts)
        pos = self.trader.open_position(
            {
                "symbol": "DOGE/USDC",
                "chain": "solana",
                "pair_address": "pair_a",
                "coin_id": 1,
                "price_usd": 0.000086,
                "latest_price": 0.000086,
                "latest_liquidity": 100000.0,
                "price_updated_at": ts,
                "liquidity_updated_at": ts,
                "source_provider": "dexscreener",
            },
            size_usd=100.0,
            allow_coin_price_fallback=True,
            skip_execution_guard=True,
        )
        self.assertIsNotNone(pos)
        self.trader.set_market_prices([
            {"pair_address": "pair_b", "coin_id": 2, "price_usd": 0.2839},
        ])
        closed = self.trader.close_position(
            int(pos["id"]),
            0.2839,
            proposed_pair_address="pair_b",
            proposed_coin_id=2,
        )
        self.assertIsNone(closed)

    def test_cannot_buy_above_cash(self) -> None:
        self.trader.set_market_prices([
            {"pair_address": "pair_a", "coin_id": 1, "price_usd": 1.0},
        ])
        pos = self.trader.open_position({
            "symbol": "AAA/USDC",
            "chain": "solana",
            "pair_address": "pair_a",
            "coin_id": 1,
            "price_usd": 1.0,
        }, size_usd=50_000.0, settings={"max_position_size_usd": 100_000.0})
        self.assertIsNone(pos)

    def test_cannot_sell_more_than_held(self) -> None:
        self.trader._state["open_positions"] = [{
            "id": 99,
            "symbol": "X/Y",
            "chain": "solana",
            "quantity": 10,
            "entry_price": 1.0,
            "pair_address": "pair_x",
            "coin_id": 5,
            "entry_fees": 0.1,
        }]
        self.trader.set_market_prices([
            {"pair_address": "pair_x", "coin_id": 5, "price_usd": 1.0},
        ])
        closed = self.trader.close_position(99, 1.0)
        self.assertIsNotNone(closed)
        self.assertAlmostEqual(float(closed["quantity"]), 10)


class AuditScriptTests(unittest.TestCase):
    def test_audit_script_dry_run_does_not_modify(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trades = root / "paper_trades_log.csv"
            state = root / "paper_state.json"
            shutil.copyfile(
                Path(__file__).resolve().parents[1] / "data" / "paper_trades_log.csv",
                trades,
            )
            if (Path(__file__).resolve().parents[1] / "data" / "paper_state.json").is_file():
                shutil.copyfile(
                    Path(__file__).resolve().parents[1] / "data" / "paper_state.json",
                    state,
                )
            trades_mtime = trades.stat().st_mtime
            import scripts.audit_paper_trades as audit_script

            audit_script.apply_fix(trades_path=trades, state_path=state, dry_run=True)
            self.assertEqual(trades.stat().st_mtime, trades_mtime)

    def test_audit_script_fix_creates_backup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            archive_dir = data_dir / "archive"
            data_dir.mkdir()
            trades = data_dir / "paper_trades_log.csv"
            state = data_dir / "paper_state.json"
            with open(trades, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=[
                    "timestamp", "position_id", "symbol", "chain", "side", "quantity",
                    "fill_price", "notional_usd", "swap_fee", "priority_fee", "total_fees",
                    "gross_pnl", "realized_pnl", "net_roi_pct", "cluster_label", "reason_code",
                    "coin_id", "pair_address", "decision_ref_id",
                ])
                writer.writeheader()
                writer.writerow({
                    "timestamp": "2026-01-01T00:00:00+00:00",
                    "position_id": "1",
                    "symbol": "BAD/USDC",
                    "chain": "solana",
                    "side": "sell",
                    "quantity": "1",
                    "fill_price": "1",
                    "notional_usd": "1",
                    "swap_fee": "0",
                    "priority_fee": "0",
                    "total_fees": "0",
                    "gross_pnl": "0",
                    "realized_pnl": "0",
                    "net_roi_pct": "0",
                    "cluster_label": "X",
                    "reason_code": "X",
                    "coin_id": "",
                    "pair_address": "",
                    "decision_ref_id": "",
                })
            with open(state, "w", encoding="utf-8") as handle:
                json.dump({"cash_usd": 999999}, handle)

            import scripts.audit_paper_trades as audit_script

            with patch.object(audit_script, "ARCHIVE_DIR", archive_dir):
                result = audit_script.apply_fix(
                    trades_path=trades,
                    state_path=state,
                    dry_run=False,
                )
            self.assertFalse(result["dry_run"])
            self.assertTrue(archive_dir.exists())
            backups = list(archive_dir.glob("paper_trades_log_corrupted_*.csv"))
            self.assertEqual(len(backups), 1)


class TemporalSplitAuditTests(unittest.TestCase):
    def _frame(self, n: int = 100) -> pd.DataFrame:
        ts = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame({
            "event_timestamp": ts,
            "whale_wave_score": range(n),
            "target_profitable_4h": [i % 2 for i in range(n)],
        })

    def test_rf_split_chronological(self) -> None:
        frame = self._frame()
        train, val, test = chronological_split(frame)
        assert_chronological_splits(train, val, test)

    def test_tabicl_split_chronological(self) -> None:
        frame = self._frame()
        train, val, test = tab_chronological_split(frame)
        tab_assert_splits(train, val, test)

    def test_event_timestamp_excluded_from_rf_features(self) -> None:
        frame = self._frame()
        numeric, _, excluded = select_feature_columns(frame)
        self.assertNotIn("event_timestamp", numeric)
        self.assertIn("event_timestamp", excluded)

    def test_event_timestamp_excluded_from_tabicl_features(self) -> None:
        frame = self._frame()
        numeric, excluded = select_tabicl_feature_columns(frame)
        self.assertNotIn("event_timestamp", numeric)
        self.assertIn("event_timestamp", excluded)

    def test_rolling_context_timestamp_before_prediction(self) -> None:
        train_ts = pd.date_range("2026-01-01", periods=20, freq="h", tz="UTC").to_numpy()
        batch_min = pd.Timestamp("2026-01-15", tz="UTC")
        indices, _, _ = select_rolling_temporal_indices(
            train_ts,
            batch_min,
            rolling_days=14,
            min_context_rows=1,
            expand_window=True,
            max_rolling_days=90,
        )
        eligible_ts = pd.to_datetime(train_ts[indices], utc=True)
        self.assertTrue((eligible_ts < batch_min).all())


if __name__ == "__main__":
    unittest.main()
