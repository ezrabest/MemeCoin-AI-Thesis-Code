"""Headless data collection mode — Gemini bypass tests."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class HeadlessModeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["TRADER_DB_PATH"] = str(Path(self._tmpdir.name) / "test.db")
        os.environ["HEADLESS_DATA_COLLECTION"] = "true"
        os.environ["ENABLE_GEMINI"] = "false"

        import importlib
        import app.llm_config as llm_config
        import app.database as database
        import app.models.predictor as predictor

        llm_config.reset_llm_counters()
        importlib.reload(llm_config)
        importlib.reload(database)
        importlib.reload(predictor)

        llm_config.reset_llm_counters()
        self.llm_config = llm_config
        self.db = database
        self.predictor = predictor
        self.db.init_db()

        self.coin = self.db.upsert_coin({
            "symbol": "HEAD/SOL",
            "pair_address": "0xhead",
            "chain": "solana",
            "price_usd": 1.0,
        })

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        for key in ("TRADER_DB_PATH", "HEADLESS_DATA_COLLECTION", "ENABLE_GEMINI", "LLM_PROVIDER"):
            os.environ.pop(key, None)

    def test_llm_disabled_flags(self) -> None:
        self.assertTrue(self.llm_config.is_headless_data_collection())
        self.assertFalse(self.llm_config.is_gemini_provider_active())
        self.assertFalse(self.llm_config.is_ollama_provider_active())
        self.assertFalse(self.llm_config.is_llm_enabled())
        self.assertEqual(self.llm_config.get_llm_provider(), "none")

    def test_analyze_market_state_skips_gemini_and_stores_skipped(self) -> None:
        metrics = {
            "symbol": "HEAD/SOL",
            "token_contract_address": "0xhead",
            "price_usd": 1.0,
            "whale_score": 0.6,
            "price_change_1h": 1.0,
        }

        with patch.object(self.predictor, "_gemini_json") as mock_gemini:
            import asyncio

            decision, decision_id = asyncio.run(
                self.predictor.analyze_market_state(
                    metrics,
                    "OPPORTUNISTIC_SPECULATIVE",
                    0.0,
                    coin_id=self.coin["id"],
                    trigger_type="test_headless",
                )
            )
            mock_gemini.assert_not_called()

        self.assertEqual(decision.decision, "HOLD")
        self.assertIn("Headless Data Collection", decision.reasoning)
        self.assertIsNotNone(decision_id)

        stored = self.db.get_gemini_decisions(coin_id=self.coin["id"])
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["action"], "SKIPPED")
        self.assertEqual(stored[0]["rationale"], self.llm_config.SKIP_REASON)
        self.assertTrue(stored[0].get("input_context_json"))

        status = self.llm_config.get_llm_runtime_status()
        self.assertEqual(status["gemini_call_count"], 0)
        self.assertGreaterEqual(status["llm_skipped_count"], 1)

    def test_classify_token_cluster_skips_gemini(self) -> None:
        with patch.object(self.predictor, "_gemini_json") as mock_gemini:
            import asyncio

            label, reason = asyncio.run(
                self.predictor.classify_token_cluster(
                    symbol="NEW",
                    name="New Token",
                    network="solana",
                    contract_address="0xnewtoken",
                )
            )
            mock_gemini.assert_not_called()

        self.assertEqual(label.value, "OPPORTUNISTIC_SPECULATIVE")
        self.assertIn("Headless Data Collection", reason)
        self.assertEqual(self.llm_config.get_llm_runtime_status()["gemini_call_count"], 0)

    def test_collection_debug_status(self) -> None:
        import asyncio

        metrics = {
            "symbol": "HEAD/SOL",
            "token_contract_address": "0xhead",
            "price_usd": 1.0,
            "whale_score": 0.6,
        }
        asyncio.run(
            self.predictor.analyze_market_state(
                metrics,
                "OPPORTUNISTIC_SPECULATIVE",
                0.0,
                coin_id=self.coin["id"],
                trigger_type="test_debug",
            )
        )
        debug = self.db.get_collection_debug_status()
        self.assertGreaterEqual(debug["llm_skipped_count_db"], 1)
        self.assertEqual(debug["llm_runtime"]["gemini_call_count"], 0)


if __name__ == "__main__":
    unittest.main()
