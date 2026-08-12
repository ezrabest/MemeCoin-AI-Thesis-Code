"""Gemini context builder tests."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path


class GeminiContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["TRADER_DB_PATH"] = str(Path(self._tmpdir.name) / "test.db")
        import importlib
        import app.database as database

        importlib.reload(database)
        self.db = database
        self.db.init_db()

        coin = self.db.upsert_coin({
            "symbol": "MEME/SOL",
            "pair_address": "0xmeme",
            "chain": "solana",
            "price_usd": 0.01,
            "whale_score": 0.5,
        })
        self.coin_id = coin["id"]
        self.db.insert_market_snapshot({"coin_id": self.coin_id, "price": 0.01, "liquidity": 6000})
        self.db.insert_gemini_decision({
            "coin_id": self.coin_id,
            "symbol": "MEME/SOL",
            "action": "SELL",
            "confidence": 0.8,
            "rationale": "prior exit",
            "input_context_json": {},
            "gemini_response_json": {"decision": "SELL"},
        })
        self.db.insert_trade({
            "coin_id": self.coin_id,
            "symbol": "MEME/SOL",
            "side": "sell",
            "value": 50,
            "pnl": -2.5,
            "source": "app_paper",
        })

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        os.environ.pop("TRADER_DB_PATH", None)

    def test_build_gemini_context_includes_history(self) -> None:
        import importlib
        import app.gemini_context as gc

        importlib.reload(gc)
        ctx = gc.build_gemini_context(self.coin_id)
        self.assertEqual(ctx["coin_id"], self.coin_id)
        self.assertTrue(ctx.get("prior_gemini_decisions"))
        self.assertTrue(ctx.get("app_paper_trades"))
        self.assertIn("churn_guard", ctx)


if __name__ == "__main__":
    unittest.main()
