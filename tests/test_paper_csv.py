"""Paper trade CSV schema resilience tests."""
from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path


class PaperCsvTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        data_dir = Path(self._tmpdir.name) / "data"
        data_dir.mkdir(parents=True)
        os.environ["TRADER_DB_PATH"] = str(data_dir / "test.db")

        import importlib
        import app.execution.paper as paper

        importlib.reload(paper)
        paper.DATA_DIR = data_dir
        paper.STATE_PATH = data_dir / "paper_state.json"
        paper.TRADES_LOG_PATH = data_dir / "paper_trades_log.csv"
        self.paper = paper

        import app.database as database

        importlib.reload(database)
        database.DATA_DIR = data_dir
        database.DB_PATH = data_dir / "test.db"
        database.init_db()
        self.db = database

        self.trader = paper.PaperTrader()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        os.environ.pop("TRADER_DB_PATH", None)

    def test_trade_row_with_linkage_fields_does_not_crash(self) -> None:
        coin = self.db.upsert_coin({
            "symbol": "LINK/SOL",
            "pair_address": "0xlink",
            "chain": "solana",
            "price_usd": 1.0,
        })
        decision_id = self.db.insert_gemini_decision({
            "coin_id": coin["id"],
            "symbol": "LINK/SOL",
            "action": "BUY",
            "confidence": 0.8,
            "rationale": "test",
        })
        self.trader.set_market_prices([
            {"pair_address": coin["pair_address"], "coin_id": coin["id"], "price_usd": coin["price_usd"]},
        ])
        pos = self.trader.open_position(
            {**coin, "coin_id": coin["id"], "decision_ref_id": decision_id},
            reason_code="TEST",
        )
        self.assertIsNotNone(pos)

        with open(self.paper.TRADES_LOG_PATH, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertIn("coin_id", rows[0])
        self.assertIn("pair_address", rows[0])
        self.assertIn("decision_ref_id", rows[0])

        trades = self.db.get_trades()
        self.assertEqual(len(trades), 1)

    def test_legacy_csv_header_migrates_safely(self) -> None:
        legacy_header = [
            "timestamp",
            "position_id",
            "symbol",
            "chain",
            "side",
            "quantity",
            "fill_price",
            "notional_usd",
            "swap_fee",
            "priority_fee",
            "total_fees",
            "gross_pnl",
            "realized_pnl",
            "net_roi_pct",
            "cluster_label",
            "reason_code",
        ]
        with open(self.paper.TRADES_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=legacy_header)
            writer.writeheader()
            writer.writerow({field: "1" for field in legacy_header})

        fields = self.paper._ensure_trade_csv_header()
        self.assertIn("coin_id", fields)
        self.assertIn("pair_address", fields)
        self.assertIn("decision_ref_id", fields)

        self.trader._append_trade_row({
            "timestamp": "2026-06-05T00:00:00+00:00",
            "position_id": 99,
            "symbol": "MIG/SOL",
            "chain": "solana",
            "side": "buy",
            "quantity": 1,
            "fill_price": 1,
            "notional_usd": 1,
            "swap_fee": 0.01,
            "priority_fee": 0.01,
            "total_fees": 0.02,
            "gross_pnl": 0,
            "realized_pnl": 0,
            "net_roi_pct": 0,
            "cluster_label": "TEST",
            "reason_code": "TEST",
            "coin_id": 1,
            "pair_address": "0xmig",
            "decision_ref_id": 7,
        })

        with open(self.paper.TRADES_LOG_PATH, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
