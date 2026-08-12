"""API storage endpoint smoke tests."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient


class ApiStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["TRADER_DB_PATH"] = str(Path(self._tmpdir.name) / "test.db")
        import importlib
        import app.database as database

        importlib.reload(database)
        database.init_db()
        database.upsert_coin({
            "symbol": "API/TEST",
            "pair_address": "0xapi",
            "chain": "solana",
            "price_usd": 1.0,
        })
        import main

        importlib.reload(main)
        self.client = TestClient(main.app)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        os.environ.pop("TRADER_DB_PATH", None)

    def test_debug_storage(self) -> None:
        r = self.client.get("/api/debug/storage")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("coins", data)
        self.assertGreaterEqual(data["coins"]["rows"], 1)

    def test_list_coins_real_data(self) -> None:
        r = self.client.get("/api/coins")
        self.assertEqual(r.status_code, 200)
        coins = r.json()
        self.assertTrue(any(c.get("symbol") == "API/TEST" for c in coins))


if __name__ == "__main__":
    unittest.main()
