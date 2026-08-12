"""Persistence layer tests — uses isolated SQLite file via TRADER_DB_PATH."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path


class DatabasePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._db_path = Path(self._tmpdir.name) / "test.db"
        os.environ["TRADER_DB_PATH"] = str(self._db_path)
        # Force reimport with new path
        import importlib
        import app.database as database

        importlib.reload(database)
        self.db = database
        self.db.init_db()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        os.environ.pop("TRADER_DB_PATH", None)

    def test_upsert_and_get_coin(self) -> None:
        coin = self.db.upsert_coin({
            "symbol": "PEPE/WETH",
            "name": "Pepe",
            "chain": "ethereum",
            "pair_address": "0xabc123",
            "price_usd": 0.000001,
            "liquidity_usd": 50000,
            "volume_24h": 100000,
            "whale_score": 0.55,
        })
        self.assertIsNotNone(coin)
        self.assertEqual(coin["symbol"], "PEPE/WETH")
        fetched = self.db.get_coin_by_id(coin["id"])
        self.assertIsNotNone(fetched)
        self.assertNotEqual(fetched.get("symbol"), "DEMO/SOL")
        self.assertEqual(fetched["price_usd"], 0.000001)

    def test_market_snapshot_persists(self) -> None:
        coin = self.db.upsert_coin({
            "symbol": "DOGE/BTCB",
            "pair_address": "0xdef456",
            "chain": "bsc",
            "price_usd": 0.1,
            "liquidity_usd": 8000,
            "volume_24h": 20000,
        })
        snap_id = self.db.insert_market_snapshot({
            "coin_id": coin["id"],
            "price": 0.1,
            "liquidity": 8000,
            "volume_24h": 20000,
            "whale_score": 0.4,
            "filter_status": "passed",
        })
        self.assertIsNotNone(snap_id)
        snaps = self.db.get_market_snapshots(coin["id"])
        self.assertEqual(len(snaps), 1)

    def test_raw_payload_dedup(self) -> None:
        rid1 = self.db.insert_raw_payload(provider="dexscreener", payload={"a": 1}, query="meme")
        rid2 = self.db.insert_raw_payload(provider="dexscreener", payload={"a": 1}, query="meme")
        self.assertEqual(rid1, rid2)

    def test_whale_alert_aggregate_flag(self) -> None:
        coin = self.db.upsert_coin({"symbol": "X/Y", "pair_address": "0x111", "chain": "solana"})
        aid = self.db.insert_whale_alert({
            "coin_id": coin["id"],
            "symbol": "X/Y",
            "alert_type": "LARGE_BUY",
            "whale_score": 0.6,
            "is_real_wallet_level": False,
            "description": "aggregate flow",
        })
        self.assertIsNotNone(aid)
        alerts = self.db.get_whale_alerts(coin_id=coin["id"])
        self.assertFalse(alerts[0]["is_real_wallet_level"])

    def test_gemini_decision_persists(self) -> None:
        coin = self.db.upsert_coin({"symbol": "A/B", "pair_address": "0x222", "chain": "solana"})
        did = self.db.insert_gemini_decision({
            "coin_id": coin["id"],
            "symbol": "A/B",
            "action": "HOLD",
            "confidence": 0.5,
            "rationale": "test",
            "input_context_json": {"churn_guard": {"status": "no_history"}},
            "gemini_response_json": {"decision": "HOLD"},
        })
        self.assertIsNotNone(did)
        decs = self.db.get_gemini_decisions(coin_id=coin["id"])
        self.assertEqual(len(decs), 1)

    def test_paper_trade_persists(self) -> None:
        coin = self.db.upsert_coin({"symbol": "T/SOL", "pair_address": "0x333", "chain": "solana"})
        tid = self.db.insert_trade({
            "coin_id": coin["id"],
            "symbol": "T/SOL",
            "side": "buy",
            "price": 1.0,
            "amount": 100,
            "value": 100,
            "source": "app_paper",
        })
        self.assertIsNotNone(tid)
        trades = self.db.get_trades(coin_id=coin["id"])
        self.assertEqual(len(trades), 1)

    def test_storage_stats(self) -> None:
        stats = self.db.get_storage_stats()
        self.assertIn("coins", stats)
        self.assertGreaterEqual(stats["coins"]["rows"], 0)


if __name__ == "__main__":
    unittest.main()
