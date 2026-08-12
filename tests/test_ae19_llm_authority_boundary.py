"""AE19: LLM providers are audit/shadow only — no paper execution authority."""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


AUTHORITY_FIELDS = {
    "authority_status": "AUDIT_ONLY_NO_TRADE_AUTHORITY",
    "execution_allowed": False,
    "paper_execution_allowed": False,
    "live_execution_allowed": False,
    "risk_override_allowed": False,
    "execution_attempted": False,
    "blocked_reason": "LLM_PROVIDER_AUDIT_ONLY",
}


class GeminiAuditOnlyAuthorityTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["TRADER_DB_PATH"] = str(Path(self._tmpdir.name) / "test.db")
        os.environ["HEADLESS_DATA_COLLECTION"] = "false"
        os.environ["ENABLE_GEMINI"] = "true"
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["GEMINI_API_KEY"] = "test-key-not-real"

        import app.database as database
        import app.llm_config as llm_config
        import app.models.predictor as predictor

        importlib.reload(llm_config)
        importlib.reload(database)
        importlib.reload(predictor)
        llm_config.reset_llm_counters()

        self.llm_config = llm_config
        self.predictor = predictor
        self.db = database
        self.db.init_db()
        self.coin = self.db.upsert_coin({
            "symbol": "PUMP/USDC",
            "pair_address": "0xpump_ae19",
            "chain": "solana",
            "price_usd": 0.01,
        })

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        for key in (
            "TRADER_DB_PATH",
            "HEADLESS_DATA_COLLECTION",
            "ENABLE_GEMINI",
            "LLM_PROVIDER",
            "GEMINI_API_KEY",
        ):
            os.environ.pop(key, None)

    def _mock_gemini_decision(self, action: str) -> dict:
        return {
            "decision": action,
            "strategy_type": "SCALPING_OPPORTUNITY",
            "risk_score": 40,
            "confidence": 0.81,
            "reasoning": f"mock gemini {action}",
            "position_id": None,
            "symbol": "PUMP/USDC",
        }

    def test_gemini_sell_stored_no_paper_execution_no_exec_fail(self) -> None:
        metrics = {
            "symbol": "PUMP/USDC",
            "token_contract_address": "0xpump_ae19",
            "price_usd": 0.01,
            "whale_score": 0.7,
            "price_change_1h": -1.0,
        }
        gemini_payload = self._mock_gemini_decision("SELL")

        with patch.object(self.predictor, "is_gemini_provider_active", return_value=True), patch.object(
            self.predictor, "_configure_genai", return_value=True
        ), patch.object(self.predictor, "_gemini_json", return_value=gemini_payload), patch.object(
            self.predictor, "get_paper_trader"
        ) as mock_get_trader:
            trader = MagicMock()
            trader.get_positions.return_value = []
            trader.get_wallet_summary.return_value = {
                "total_equity_usd": 10000.0,
                "cash_usd": 10000.0,
                "trading_mode": "DEMO",
            }
            trader.try_autonomous_sell = MagicMock(return_value={"id": 99})
            trader.try_autonomous_buy = MagicMock(return_value={"id": 1})
            trader.open_position = MagicMock(return_value={"id": 1})
            trader.close_position = MagicMock(return_value={"id": 99})
            mock_get_trader.return_value = trader

            with self.assertLogs("predictor", level="INFO") as cm:
                decision, decision_id = asyncio.run(
                    self.predictor.analyze_market_state(
                        metrics,
                        "OPPORTUNISTIC_SPECULATIVE",
                        0.0,
                        coin_id=self.coin["id"],
                        trigger_type="test_ae19_sell",
                        open_positions=[],
                    )
                )
                exec_result = self.predictor.execute_trade_decision(
                    decision,
                    {
                        "symbol": "PUMP/USDC",
                        "pair_address": "0xpump_ae19",
                        "coin_id": self.coin["id"],
                        "price_usd": 0.01,
                    },
                    "OPPORTUNISTIC_SPECULATIVE",
                    {"auto_execution_enabled": True},
                    cur_price=0.01,
                    decision_ref_id=decision_id,
                    coin_id=self.coin["id"],
                    provider="gemini",
                )

        self.assertEqual(decision.decision, "SELL")
        self.assertIsNotNone(decision_id)
        stored = self.db.get_gemini_decisions(coin_id=self.coin["id"])[0]
        self.assertEqual(stored["provider"], "gemini")
        self.assertEqual(stored["action"], "SELL")
        response = stored.get("gemini_response") or stored.get("gemini_response_json") or {}
        if isinstance(response, str):
            import json

            response = json.loads(response)
        for key, expected in AUTHORITY_FIELDS.items():
            self.assertEqual(response.get(key), expected, f"missing/wrong {key}")
        self.assertFalse(exec_result.get("execution_attempted", True))
        self.assertTrue(exec_result.get("audit_only"))
        self.assertEqual(exec_result.get("reason"), "LLM_PROVIDER_AUDIT_ONLY")
        trader.try_autonomous_sell.assert_not_called()
        trader.close_position.assert_not_called()
        joined = "\n".join(cm.output)
        self.assertIn("LLM audit-only decision stored; execution not attempted provider=gemini action=SELL", joined)
        self.assertNotIn("EXEC_FAIL", joined)

    def test_gemini_buy_stored_no_paper_execution_no_exec_fail(self) -> None:
        metrics = {
            "symbol": "PUMP/USDC",
            "token_contract_address": "0xpump_ae19",
            "price_usd": 0.01,
            "whale_score": 0.2,
            "price_change_1h": 0.0,
        }
        gemini_payload = self._mock_gemini_decision("BUY")

        with patch.object(self.predictor, "is_gemini_provider_active", return_value=True), patch.object(
            self.predictor, "_configure_genai", return_value=True
        ), patch.object(self.predictor, "_gemini_json", return_value=gemini_payload), patch.object(
            self.predictor, "get_paper_trader"
        ) as mock_get_trader:
            trader = MagicMock()
            trader.get_positions.return_value = []
            trader.get_wallet_summary.return_value = {
                "total_equity_usd": 10000.0,
                "cash_usd": 10000.0,
                "trading_mode": "DEMO",
            }
            trader.try_autonomous_buy = MagicMock(return_value={"id": 1})
            trader.open_position = MagicMock(return_value={"id": 1})
            mock_get_trader.return_value = trader

            with self.assertLogs("predictor", level="INFO") as cm:
                decision, decision_id = asyncio.run(
                    self.predictor.analyze_market_state(
                        metrics,
                        "OPPORTUNISTIC_SPECULATIVE",
                        0.0,
                        coin_id=self.coin["id"],
                        trigger_type="test_ae19_buy",
                        open_positions=[],
                    )
                )
                exec_result = self.predictor.execute_trade_decision(
                    decision,
                    {
                        "symbol": "PUMP/USDC",
                        "pair_address": "0xpump_ae19",
                        "coin_id": self.coin["id"],
                        "price_usd": 0.01,
                    },
                    "OPPORTUNISTIC_SPECULATIVE",
                    {"auto_execution_enabled": True},
                    provider="gemini",
                )

        self.assertEqual(decision.decision, "BUY")
        self.assertIsNotNone(decision_id)
        stored = self.db.get_gemini_decisions(coin_id=self.coin["id"])[0]
        self.assertEqual(stored["action"], "BUY")
        response = stored.get("gemini_response") or stored.get("gemini_response_json") or {}
        if isinstance(response, str):
            import json

            response = json.loads(response)
        self.assertEqual(response.get("authority_status"), "AUDIT_ONLY_NO_TRADE_AUTHORITY")
        self.assertFalse(response.get("execution_allowed"))
        self.assertTrue(exec_result.get("audit_only"))
        trader.try_autonomous_buy.assert_not_called()
        trader.open_position.assert_not_called()
        joined = "\n".join(cm.output)
        self.assertIn("LLM audit-only decision stored; execution not attempted provider=gemini action=BUY", joined)
        self.assertNotIn("EXEC_FAIL", joined)

    def test_scan_gemini_decision_counter(self) -> None:
        self.llm_config.reset_scan_llm_decision_counters()
        self.assertEqual(self.llm_config.get_scan_gemini_decisions_stored(), 0)

        metrics = {
            "symbol": "PUMP/USDC",
            "token_contract_address": "0xpump_ae19",
            "price_usd": 0.01,
            "whale_score": 0.2,
            "price_change_1h": 0.0,
        }
        with patch.object(self.predictor, "is_gemini_provider_active", return_value=True), patch.object(
            self.predictor, "_configure_genai", return_value=True
        ), patch.object(self.predictor, "_gemini_json", return_value=self._mock_gemini_decision("HOLD")):
            asyncio.run(
                self.predictor.analyze_market_state(
                    metrics,
                    "OPPORTUNISTIC_SPECULATIVE",
                    0.0,
                    coin_id=self.coin["id"],
                    trigger_type="test_counter",
                )
            )

        self.assertEqual(self.llm_config.get_scan_gemini_decisions_stored(), 1)
        self.assertEqual(self.llm_config.get_scan_llm_decisions_stored(), 1)

    def test_live_whale_event_skips_execute_for_gemini(self) -> None:
        import app.live as live

        importlib.reload(live)
        decision = self.predictor.TradeDecision(
            decision="SELL",
            strategy_type="SCALPING_OPPORTUNITY",
            risk_score=40,
            confidence=0.8,
            reasoning="audit",
            symbol="PUMP/USDC",
        )

        async def _fake_analyze(*_a, **_k):
            return decision, 12345

        with patch.object(live, "analyze_market_state", side_effect=_fake_analyze), patch.object(
            live, "execute_trade_decision"
        ) as mock_exec, patch.object(live, "get_llm_provider", return_value="gemini"), patch.object(
            live, "is_llm_audit_only_provider", return_value=True
        ), patch.object(live, "build_feature_row", return_value={"symbol": "PUMP/USDC"}), patch.object(
            live, "normalize_execution_settings", side_effect=lambda s: s
        ), patch.object(live, "get_paper_trader") as mock_trader, self.assertLogs("live", level="INFO") as cm:
            mock_trader.return_value.get_positions.return_value = []
            token = MagicMock()
            token.symbol = "PUMP"
            token.contract_address = "0xpump_ae19"
            state = MagicMock()
            state.price_usd = 0.01
            result = asyncio.run(
                live._evaluate_whale_event(
                    {},
                    state,
                    token,
                    "PUMP/USDC",
                    "solana",
                    0.9,
                    "OPPORTUNISTIC_SPECULATIVE",
                    0.0,
                    {"auto_execution_enabled": True, "llm_score_threshold": 0.1},
                    coin_id=self.coin["id"],
                    alert_type="LARGE_BUY",
                )
            )

        mock_exec.assert_not_called()
        self.assertTrue(result.get("audit_only"))
        self.assertEqual(result.get("reason"), "LLM_PROVIDER_AUDIT_ONLY")
        self.assertIn("LLM audit-only decision stored; execution not attempted provider=gemini", "\n".join(cm.output))


class PaperDefensiveLlmGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        state_path = Path(self._tmpdir.name) / "paper_state.json"
        trades_path = Path(self._tmpdir.name) / "paper_trades.csv"
        import app.execution.paper as paper_mod

        self._paper_mod = paper_mod
        self._state_patch = patch.object(paper_mod, "STATE_PATH", state_path)
        self._trades_patch = patch.object(paper_mod, "TRADES_LOG_PATH", trades_path)
        self._state_patch.start()
        self._trades_patch.start()
        self.trader = paper_mod.PaperTrader()

    def tearDown(self) -> None:
        self._state_patch.stop()
        self._trades_patch.stop()
        self._tmpdir.cleanup()

    def _coin(self, **extra):
        base = {
            "symbol": "DEMO/SOL",
            "chain": "solana",
            "price_usd": 1.0,
            "pair_address": "0xdemo_pair",
        }
        base.update(extra)
        return base

    def test_open_blocked_for_explicit_llm_providers(self) -> None:
        for provider in ("gemini", "qwen", "ollama"):
            with self.subTest(provider=provider):
                with self.assertLogs("paper", level="INFO") as cm:
                    pos = self.trader.open_position(
                        self._coin(provider=provider),
                        size_usd=100.0,
                        settings={"auto_execution_enabled": True},
                        skip_market_data_gate=True,
                    )
                self.assertIsNone(pos)
                last = self.trader._state.get("last_open_result") or {}
                self.assertEqual(last.get("reason"), "LLM_PROVIDER_AUDIT_ONLY")
                self.assertFalse(last.get("execution_attempted", True))
                self.assertIn(
                    f"LLM audit-only decision stored; execution not attempted provider={provider} action=BUY",
                    "\n".join(cm.output),
                )
                self.assertNotIn("EXEC_FAIL", "\n".join(cm.output))

    def test_close_blocked_for_explicit_llm_providers(self) -> None:
        # Seed an open position without going through open_position gates.
        self.trader._state["open_positions"] = [
            {
                "id": 42,
                "symbol": "DEMO/SOL",
                "chain": "solana",
                "quantity": 100.0,
                "entry_price": 1.0,
                "size_usd": 100.0,
                "pair_address": "0xdemo_pair",
                "coin_id": 7,
                "cluster_label": "OPPORTUNISTIC_SPECULATIVE",
                "opened_at": "2026-08-03T00:00:00+00:00",
            }
        ]
        self.trader._save_state()
        open_count = len(self.trader.get_positions("OPEN"))
        self.assertEqual(open_count, 1)

        for provider in ("gemini", "qwen", "ollama"):
            with self.subTest(provider=provider):
                with self.assertLogs("paper", level="INFO") as cm:
                    closed = self.trader.close_position(
                        42,
                        cur_price=1.05,
                        provider=provider,
                        skip_execution_guard=True,
                    )
                self.assertIsNone(closed)
                last = self.trader._state.get("last_close_result") or {}
                self.assertEqual(last.get("reason"), "LLM_PROVIDER_AUDIT_ONLY")
                self.assertEqual(len(self.trader.get_positions("OPEN")), open_count)
                self.assertIn(
                    f"LLM audit-only decision stored; execution not attempted provider={provider} action=SELL",
                    "\n".join(cm.output),
                )
                self.assertNotIn("EXEC_FAIL", "\n".join(cm.output))

    def test_non_llm_paper_execution_still_allowed(self) -> None:
        from app.llm_config import extract_explicit_llm_origin_provider

        coin = self._coin(coin_id=7)
        self.assertIsNone(extract_explicit_llm_origin_provider(coin, {}))
        self.assertIsNone(extract_explicit_llm_origin_provider({"source": "economic_gate"}))

        # Without explicit LLM provider markers, try_autonomous_buy must not LLM-block.
        # auto_execution_enabled=False causes a normal early None (not LLM_PROVIDER_AUDIT_ONLY).
        result = self.trader.try_autonomous_buy(
            coin,
            "OPPORTUNISTIC_SPECULATIVE",
            {"auto_execution_enabled": False},
        )
        self.assertIsNone(result)
        last = self.trader._state.get("last_open_result") or {}
        self.assertNotEqual(last.get("reason"), "LLM_PROVIDER_AUDIT_ONLY")

        # sell without provider markers also must not LLM-block.
        sell = self.trader.try_autonomous_sell(
            symbol="DEMO/SOL",
            settings={"auto_execution_enabled": False},
        )
        self.assertIsNone(sell)
        last_close = self.trader._state.get("last_close_result") or {}
        self.assertNotEqual(last_close.get("reason"), "LLM_PROVIDER_AUDIT_ONLY")


class GeminiModeConfigStillWorks(unittest.TestCase):
    def test_mode_gemini_env(self) -> None:
        os.environ["LLM_PROVIDER"] = "gemini"
        os.environ["ENABLE_GEMINI"] = "true"
        os.environ["HEADLESS_DATA_COLLECTION"] = "false"
        import app.llm_config as llm_config

        importlib.reload(llm_config)
        self.assertEqual(llm_config.get_llm_provider(), "gemini")
        self.assertTrue(llm_config.is_gemini_provider_active())
        self.assertTrue(llm_config.is_llm_audit_only_provider("gemini"))
        for key in ("LLM_PROVIDER", "ENABLE_GEMINI", "HEADLESS_DATA_COLLECTION"):
            os.environ.pop(key, None)


if __name__ == "__main__":
    unittest.main()
