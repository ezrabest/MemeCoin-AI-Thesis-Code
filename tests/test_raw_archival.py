"""Raw payload archival tests."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path


class RawArchivalTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["TRADER_DB_PATH"] = str(Path(self._tmpdir.name) / "test.db")
        import importlib
        import app.database as database
        import app.analytics.scan_persist as sp

        importlib.reload(database)
        importlib.reload(sp)
        self.db = database
        self.sp = sp
        self.db.init_db()

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        os.environ.pop("TRADER_DB_PATH", None)

    def test_dexscreener_search_pair_count_from_summary(self) -> None:
        ref_id = self.sp.archive_dexscreener_search(
            "trending_batch",
            {"pair_count": 17, "scan_id": "test-scan"},
            source_type="trending_summary",
        )
        self.assertIsNotNone(ref_id)
        payloads = self.db.get_recent_raw_payloads(limit=5, provider="dexscreener")
        self.assertTrue(any(p.get("source_type") == "trending_summary" for p in payloads))

    def test_dexscreener_pair_archived_on_persist(self) -> None:
        pair = {
            "pairAddress": "0xraw1",
            "chainId": "solana",
            "priceUsd": "0.01",
            "baseToken": {"symbol": "RAW", "name": "Raw"},
            "quoteToken": {"symbol": "SOL"},
            "liquidity": {"usd": 8000},
            "volume": {"h24": 20000},
            "priceChange": {"h1": 1, "h24": 5},
            "txns": {"h24": {"buys": 100, "sells": 80}},
        }
        result = self.sp.persist_pair_pipeline(
            pair,
            scan_id="test-scan",
            filter_status="passed",
        )
        self.assertIsNotNone(result)
        self.assertIn("coin_id", result)
        stats = self.db.get_storage_stats()
        self.assertGreaterEqual(stats["raw_provider_payloads"]["rows"], 1)
        self.assertGreaterEqual(stats["market_snapshots"]["rows"], 1)


if __name__ == "__main__":
    unittest.main()
