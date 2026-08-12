"""AE13H targeted tests — position MTM UI mapping, identity disambiguation,
per-position Sell Demo close, paper/demo safety, preset consistency.

Paper/demo only — no wallet, no live trading, no private keys.
"""
from __future__ import annotations

import importlib
import os
import re
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coin(i: int, *, symbol: str | None = None, pair: str | None = None) -> dict:
    return {
        "symbol": symbol or f"SYM{i}",
        "chain": "solana",
        "pair_address": pair or f"pair_{i}",
        "coin_id": i,
        "latest_price": 1.0,
        "latest_liquidity": 50000 + i,
        "latest_volume_24h": 12000 + i,
        "latest_whale_score": 0.7 + i * 0.01,
    }


class _IsolatedPaperTraderCase(unittest.TestCase):
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
        self.db = database
        self.trader = paper.PaperTrader()
        self.trader.set_market_prices(
            [{"pair_address": f"pair_{i}", "coin_id": i, "price_usd": 1.0} for i in range(1, 12)],
            price_timestamp=_utc_now_iso(),
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        os.environ.pop("TRADER_DB_PATH", None)


class PortfolioMtmFieldTests(_IsolatedPaperTraderCase):
    """Portfolio API / mark_positions expose MTM fields used by UI."""

    def test_marked_positions_include_mtm_and_exit_context(self) -> None:
        pos = self.trader.open_position(
            _coin(1),
            size_usd=25.0,
            settings={"take_profit_pct": 0.5, "stop_loss_pct": 0.2},
            reason_code="TEST",
        )
        self.assertIsNotNone(pos)
        # Annotate like demo_bot would.
        opens = self.trader.get_positions(status="OPEN")
        opens[0].update(
            {
                "strategy_lane": "meme_opportunistic_scout",
                "pair_address": "pair_1",
                "coin_id": 1,
                "min_hold_seconds": 60,
                "time_stop_seconds": 86400,
                "trailing_stop_pct": 0.15,
                "take_profit": 1.5,
                "stop_loss": 0.8,
                "exit_plan": {
                    "take_profit_pct": 0.5,
                    "stop_loss_pct": 0.2,
                    "trailing_stop_pct": 0.15,
                    "min_hold_seconds": 60,
                    "time_stop_seconds": 86400,
                },
            }
        )
        self.trader._save_state()  # noqa: SLF001
        self.trader.set_market_prices(
            [{"pair_address": "pair_1", "coin_id": 1, "price_usd": 1.1}],
            price_timestamp=_utc_now_iso(),
        )
        marked = self.trader.get_marked_positions()
        row = marked[0]
        for key in (
            "current_price",
            "unrealized_pnl_usd",
            "unrealized_pnl_pct",
            "age_label",
            "distance_to_take_profit_pct",
            "distance_to_stop_loss_pct",
            "exit_plan_summary",
            "bot_exit_reason",
            "manual_close_allowed",
            "paper_demo_only",
            "not_live_approved",
        ):
            self.assertIn(key, row)
        self.assertEqual(row["current_price"], 1.1)
        self.assertGreater(row["unrealized_pnl_usd"], 0)
        self.assertIn("TP", row["exit_plan_summary"])
        self.assertTrue(row["manual_close_allowed"])
        self.assertTrue(row["paper_demo_only"])


class DuplicateSymbolIdentityTests(_IsolatedPaperTraderCase):
    def test_same_symbol_different_pools_remain_distinct(self) -> None:
        a = self.trader.open_position(
            _coin(1, symbol="WIF/SOL", pair="pool_aaa_111"),
            size_usd=10.0,
            settings={},
            reason_code="TEST",
        )
        b = self.trader.open_position(
            _coin(2, symbol="WIF/SOL", pair="pool_bbb_222"),
            size_usd=10.0,
            settings={},
            reason_code="TEST",
        )
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        opens = self.trader.get_positions(status="OPEN")
        wif = [p for p in opens if p["symbol"] == "WIF/SOL"]
        self.assertEqual(len(wif), 2)
        pairs = {p["pair_address"] for p in wif}
        ids = {p["id"] for p in wif}
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len(ids), 2)


class PerPositionCloseTests(_IsolatedPaperTraderCase):
    def test_close_only_selected_position_with_reason(self) -> None:
        p1 = self.trader.open_position(_coin(1), size_usd=10.0, settings={}, reason_code="TEST")
        p2 = self.trader.open_position(_coin(2), size_usd=10.0, settings={}, reason_code="TEST")
        self.assertIsNotNone(p1)
        self.assertIsNotNone(p2)
        closed = self.trader.close_position(
            int(p1["id"]),
            1.2,
            reason_code="MANUAL_SELL",
            proposed_pair_address="pair_1",
            proposed_coin_id=1,
            close_reason="manual_take_profit",
            close_note="ae13h unit test",
            closed_by="user_manual",
        )
        self.assertIsNotNone(closed)
        assert closed is not None
        self.assertEqual(closed["id"], p1["id"])
        self.assertEqual(closed["closed_by"], "user_manual")
        self.assertEqual(closed["close_reason"], "manual_take_profit")
        self.assertEqual(closed["close_note"], "ae13h unit test")
        self.assertTrue(closed["paper_demo_only"])
        self.assertTrue(closed["not_live_approved"])
        self.assertTrue(closed["not_profitability_evidence"])
        self.assertEqual(closed["trade_authority"], "PAPER_DEMO_ONLY")
        remaining = self.trader.get_positions(status="OPEN")
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["id"], p2["id"])


class CloseEndpointApiTests(_IsolatedPaperTraderCase):
    def test_put_close_endpoint_scopes_to_one_position(self) -> None:
        from fastapi.testclient import TestClient

        import app.api as api_mod
        import app.execution.paper as paper_mod

        paper_mod._paper_trader = self.trader  # noqa: SLF001
        api_mod.get_paper_trader = lambda: self.trader

        p1 = self.trader.open_position(_coin(3), size_usd=10.0, settings={}, reason_code="TEST")
        p2 = self.trader.open_position(_coin(4), size_usd=10.0, settings={}, reason_code="TEST")
        self.assertIsNotNone(p1)
        self.assertIsNotNone(p2)

        client = TestClient(api_mod.app)
        resp = client.put(
            f"/api/positions/{p1['id']}/close",
            json={"close_reason": "testing", "close_note": "api scope test"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        data = resp.json()
        self.assertEqual(data["position_id"], p1["id"])
        self.assertEqual(data["close_reason"], "testing")
        self.assertTrue(data["paper_demo_only"])
        self.assertTrue(data["not_live_approved"])
        self.assertIn("closed manually", data["message"].lower())

        remaining = {p["id"] for p in self.trader.get_positions(status="OPEN")}
        self.assertNotIn(p1["id"], remaining)
        self.assertIn(p2["id"], remaining)


class UiMappingStaticTests(unittest.TestCase):
    def test_portfolio_table_maps_mtm_and_sell_button(self) -> None:
        html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
        js = (ROOT / "static" / "product_demo.js").read_text(encoding="utf-8")
        for col in (
            "Current Price",
            "Unrealized PnL $",
            "Unrealized PnL %",
            "TP Distance",
            "SL Distance",
            "Exit Status",
            "Pool / Pair Address",
            "Strategy Lane",
            "Actions",
        ):
            self.assertIn(col, html)
        self.assertIn("Sell Demo", js)
        self.assertIn("pdOpenSellDemo", js)
        self.assertIn("renderPortfolioOpenRow", js)
        self.assertIn("current_price", js)
        self.assertIn("unrealized_pnl_usd", js)
        self.assertIn("distance_to_take_profit_pct", js)
        self.assertIn("pair_address", js)
        self.assertIn("/api/positions/", js)
        self.assertIn("close_reason", js)
        self.assertIn("Paper / demo only", html)
        # Must not introduce live wallet sell path.
        self.assertNotRegex(js, re.compile(r"signTransaction|private[_ ]?key|sendRawTransaction", re.I))


class PresetConsistencyTests(unittest.TestCase):
    def test_aggressive_and_lotto_caps(self) -> None:
        from app.ae13b_product.presets import PRESETS

        self.assertEqual(PRESETS["aggressive"]["max_open_positions"], 6)
        self.assertEqual(PRESETS["aggressive"]["max_trades_per_hour"], 30)
        self.assertEqual(PRESETS["aggressive"]["max_notional_usd"], 100.0)
        self.assertEqual(PRESETS["lotto"]["max_open_positions"], 8)

    def test_risk_guard_honours_aggressive_bot_state(self) -> None:
        from app.ae13b_product.demo_risk_guard import evaluate_demo_risk_guard

        opens = [
            {"pair_address": f"p{i}", "symbol": f"S{i}", "size_usd": 25, "chain": "solana"}
            for i in range(6)
        ]
        blocked = evaluate_demo_risk_guard(
            requested_notional=10.0,
            demo_equity=10000.0,
            open_positions=opens,
            recent_trades=[],
            pair_address="p_new",
            symbol="NEW",
            chain="solana",
            price=1.0,
            bot_state={
                "preset_id": "aggressive",
                "max_open_positions": 6,
                "max_trades_per_hour": 30,
                "max_notional_usd": 100.0,
            },
            preset_id="aggressive",
            risk_mode="aggressive",
        )
        self.assertFalse(blocked.get("passed", True))
        self.assertFalse(blocked.get("risk_guard_passed", True))
        self.assertIn("max_open_positions", blocked.get("blocking_guards") or [])
        self.assertEqual(blocked.get("max_open_positions"), 6)
        self.assertEqual(blocked.get("candidate_context", {}).get("available_slots"), 0)


class Ae13gRegressionSmokeTests(unittest.TestCase):
    def test_rejected_attempt_still_has_structured_fields(self) -> None:
        from app.ae13b_product.rejected_attempt import RejectedTradeAttempt

        attempt = RejectedTradeAttempt(
            symbol="X",
            rejection_reason="max_open_positions_reached",
            rejection_reasons=["Blocked: max open positions reached (6)"],
            blocking_guards=["max_open_positions"],
        )
        d = attempt.to_dict()
        self.assertEqual(d["blocking_guards"], ["max_open_positions"])
        self.assertTrue(d["paper_demo_only"])
        self.assertTrue(d["not_live_approved"])


if __name__ == "__main__":
    unittest.main()
