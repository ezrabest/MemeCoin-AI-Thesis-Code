"""AE13G targeted tests -- bot decision explainability, preset propagation,
identity-store semantics, and mark-to-market. Paper/demo only.

Covers:
 1. Risk guard returns rejection_reasons[] and blocking_guards[]
 2. Lotto preset via bot_state allows max_open=8 (fixes the "6/8 paradox")
 3. Default (no bot_state) risk guard still blocks the 7th open at max_open=6
 4. RejectedTradeAttempt.to_dict() keyed fields are not column-shifted
 5. Trade CSV header migration repairs a mismatched on-disk column order
 6. Identity Store upsert + resolver finds it without a market match (price None)
 7. Semantic check from Identity Store fields -> SOCIAL_CANDIDATE_NEEDS_VERIFICATION,
    never SOCIAL_CONFIRMED from user hypothesis alone, without a market match
 8. format_top_rejection_summary() is actionable, not a generic placeholder
 9. mark_positions_to_market() / get_marked_positions() add current_price,
    unrealized PnL, and age fields
10. Safety flags (paper_demo_only / not_live_approved) are present
"""
from __future__ import annotations

import csv
import importlib
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _coin(i: int) -> dict:
    return {
        "symbol": f"SYM{i}",
        "chain": "solana",
        "pair_address": f"pair_{i}",
        "coin_id": i,
        "latest_price": 1.0,
    }


class _IsolatedPaperTraderCase(unittest.TestCase):
    """Base case: isolated PaperTrader + SQLite DB in a tempdir (no shared state)."""

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
        # Preload market prices for pair_1..pair_9 so open_position() never
        # rejects purely on a missing price during these tests.
        self.trader.set_market_prices(
            [{"pair_address": f"pair_{i}", "coin_id": i, "price_usd": 1.0} for i in range(1, 10)],
            price_timestamp=_utc_now_iso(),
        )

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        os.environ.pop("TRADER_DB_PATH", None)


class RiskGuardStructuredResultTests(unittest.TestCase):
    """1) Risk guard returns rejection_reasons[] and blocking_guards[]."""

    def test_rejection_reasons_and_blocking_guards_present(self) -> None:
        from app.ae13b_product.demo_risk_guard import evaluate_demo_risk_guard

        result = evaluate_demo_risk_guard(
            requested_notional=0,
            pair_address="p1",
            symbol="ABC",
            chain="solana",
            price=1.0,
            price_timestamp=_utc_now_iso(),
        )
        self.assertFalse(result["risk_guard_passed"])
        self.assertIsInstance(result["rejection_reasons"], list)
        self.assertIsInstance(result["blocking_guards"], list)
        self.assertGreaterEqual(len(result["rejection_reasons"]), 1)
        self.assertIn("missing_notional", result["blocking_guards"])
        self.assertEqual(result["primary_blocker"], "missing_notional")

    def test_safety_flags_present_on_risk_guard_result(self) -> None:
        """10) Safety flags paper_demo_only / not_live_approved (risk guard surface)."""
        from app.ae13b_product.demo_risk_guard import evaluate_demo_risk_guard

        result = evaluate_demo_risk_guard(
            requested_notional=10,
            pair_address="p1",
            symbol="ABC",
            chain="solana",
            price=1.0,
            price_timestamp=_utc_now_iso(),
        )
        self.assertTrue(result["paper_demo_only"])
        self.assertTrue(result["not_live_approved"])
        self.assertTrue(result["not_profitability_evidence"])


class FormatTopRejectionSummaryTests(unittest.TestCase):
    """8) format_top_rejection_summary is actionable, not generic."""

    def test_summary_is_actionable_not_generic(self) -> None:
        from app.ae13b_product.demo_risk_guard import (
            aggregate_rejection_counts,
            format_top_rejection_summary,
        )

        attempts = [
            {
                "blocking_guards": ["duplicate_pair_guard"],
                "rejection_reasons": ["Blocked: duplicate pair already open"],
            },
            {
                "blocking_guards": ["duplicate_pair_guard"],
                "rejection_reasons": ["Blocked: duplicate pair already open"],
            },
            {
                "blocking_guards": ["cooldown"],
                "rejection_reasons": ["Blocked: pair cooldown active"],
            },
        ]
        dist = aggregate_rejection_counts(attempts)
        self.assertEqual(dist[0]["guard"], "duplicate_pair_guard")
        self.assertEqual(dist[0]["count"], 2)

        summary = format_top_rejection_summary(attempts, candidates_selected=3)
        self.assertNotIn("open_position_rejected", summary.lower())
        self.assertIn("duplicate pair already open", summary.lower())
        self.assertIn("3 candidates rejected", summary)

    def test_empty_attempts_summary_is_not_misleading(self) -> None:
        from app.ae13b_product.demo_risk_guard import format_top_rejection_summary

        summary = format_top_rejection_summary([], candidates_selected=0)
        self.assertIn("No trade attempts", summary)


class MaxOpenPositionsPresetTests(_IsolatedPaperTraderCase):
    """2 & 3) The "6/8 paradox" fix: bot_state.max_open_positions must override
    the risk guard's DEFAULTS max_open_positions (6), while the no-bot_state
    default path must still enforce the 6-open ceiling."""

    def test_lotto_preset_bot_state_allows_max_open_8(self) -> None:
        bot_state = {
            "preset_id": "lotto",
            "risk_mode": "lotto",
            "max_open_positions": 8,
            "max_notional_usd": 25,
            "cooldown_seconds": 45,
        }
        for i in range(1, 7):  # 6 opens
            pos = self.trader.open_position(
                _coin(i),
                size_usd=10.0,
                settings={},
                reason_code="TEST",
                strategy_type="LOTTO_SCOUT",
                bot_state=bot_state,
                risk_mode="lotto",
                preset_id="lotto",
            )
            self.assertIsNotNone(pos, f"open #{i} should succeed under lotto preset")
        self.assertEqual(len(self.trader.get_positions(status="OPEN")), 6)

        # 7th open: under the OLD default (max_open=6) this would be wrongly
        # blocked. With bot_state.max_open_positions=8 propagated, it must open.
        pos7 = self.trader.open_position(
            _coin(7),
            size_usd=10.0,
            settings={},
            reason_code="TEST",
            strategy_type="LOTTO_SCOUT",
            bot_state=bot_state,
            risk_mode="lotto",
            preset_id="lotto",
        )
        self.assertIsNotNone(
            pos7, "7th open must not be blocked by max_open when bot_state.max_open_positions=8"
        )
        self.assertEqual(len(self.trader.get_positions(status="OPEN")), 7)

    def test_default_no_bot_state_blocks_seventh_at_six(self) -> None:
        for i in range(1, 7):  # 6 opens, no bot_state -> risk guard DEFAULTS max_open_positions=6
            pos = self.trader.open_position(
                _coin(i),
                size_usd=10.0,
                settings={},
                reason_code="TEST",
                strategy_type="MOMENTUM_SCOUT",
            )
            self.assertIsNotNone(pos, f"open #{i} should succeed under default risk guard settings")
        self.assertEqual(len(self.trader.get_positions(status="OPEN")), 6)

        pos7 = self.trader.open_position(
            _coin(7),
            size_usd=10.0,
            settings={},
            reason_code="TEST",
            strategy_type="MOMENTUM_SCOUT",
        )
        self.assertIsNone(pos7, "7th open should be blocked by the default max_open_positions ceiling")
        last = self.trader.get_last_open_result()
        self.assertIsNotNone(last)
        self.assertFalse(last["opened"] if "opened" in last else True)
        self.assertIn("max_open_positions", last.get("blocking_guards") or [])


class RejectedTradeAttemptSerializationTests(unittest.TestCase):
    """4) RejectedTradeAttempt.to_dict() keyed fields must not be column-shifted."""

    def test_to_dict_keyed_fields_not_shifted(self) -> None:
        from app.ae13b_product.rejected_attempt import RejectedTradeAttempt

        attempt = RejectedTradeAttempt(
            symbol="ABCXYZ",
            side="SELL",
            fill_price=1.2345,
            quantity=45.6,
            rejection_reason="max_open_positions_reached",
            chain="solana",
            pair_address="pair_abc",
            rejection_reasons=["Blocked: max open positions reached (6)"],
            blocking_guards=["max_open_positions"],
        )
        d = attempt.to_dict()
        self.assertEqual(d["symbol"], "ABCXYZ")
        self.assertEqual(d["side"], "SELL")
        self.assertEqual(d["fill_price"], 1.2345)
        self.assertEqual(d["quantity"], 45.6)
        self.assertEqual(d["rejection_reason"], "max_open_positions_reached")
        self.assertEqual(d["chain"], "solana")
        self.assertEqual(d["pair_address"], "pair_abc")
        self.assertEqual(d["rejection_reasons"], ["Blocked: max open positions reached (6)"])
        self.assertEqual(d["blocking_guards"], ["max_open_positions"])
        self.assertTrue(d["paper_demo_only"])
        self.assertTrue(d["not_live_approved"])


class TradeCsvHeaderMigrationTests(_IsolatedPaperTraderCase):
    """5) CSV: RISK_GUARD_BLOCK written under a mismatched header order must
    be repaired by _ensure_trade_csv_header() so DictReader gets correct
    symbol/side (not column-shifted) for both old and newly-appended rows."""

    def test_mismatched_header_order_rewritten_correctly(self) -> None:
        fields = list(self.paper.TRADE_CSV_FIELDS)
        shuffled = list(fields)
        i_sym, i_side = shuffled.index("symbol"), shuffled.index("side")
        shuffled[i_sym], shuffled[i_side] = shuffled[i_side], shuffled[i_sym]

        row1 = {f: "" for f in fields}
        row1.update(
            {
                "timestamp": "2024-01-01T00:00:00+00:00",
                "symbol": "ABC",
                "side": "BUY",
                "event_type": "PAPER_BUY",
            }
        )
        with open(self.paper.TRADES_LOG_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=shuffled)
            writer.writeheader()
            writer.writerow(row1)

        fixed_fields = self.paper._ensure_trade_csv_header()
        self.assertEqual(fixed_fields, fields)

        with open(self.paper.TRADES_LOG_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, fields)
            rows_after_migration = list(reader)
        self.assertEqual(len(rows_after_migration), 1)
        self.assertEqual(rows_after_migration[0]["symbol"], "ABC")
        self.assertEqual(rows_after_migration[0]["side"], "BUY")

        # Append a new rejected attempt row through the real write path.
        self.trader._append_trade_row(
            {
                "timestamp": "2024-01-01T01:00:00+00:00",
                "symbol": "XYZ",
                "side": "buy",
                "chain": "solana",
                "event_type": "RISK_GUARD_BLOCK",
                "rejection_reason": "max_open_positions_reached",
            }
        )

        with open(self.paper.TRADES_LOG_PATH, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, fields)
            rows = list(reader)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["symbol"], "ABC")
        self.assertEqual(rows[0]["side"], "BUY")
        self.assertEqual(rows[1]["symbol"], "XYZ")
        self.assertEqual(rows[1]["side"], "buy")
        self.assertEqual(rows[1]["event_type"], "RISK_GUARD_BLOCK")


class IdentityStoreResolverTests(unittest.TestCase):
    """6) Identity Store upsert + resolver finds it without a market match;
    price stays None; reason references the local store."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        data_dir = Path(self._tmpdir.name) / "runtime"
        data_dir.mkdir(parents=True)

        import app.ae13b_product.identity_store as identity_store

        self._identity_store = identity_store
        self._orig_data_dir = identity_store.DATA_DIR
        self._orig_store_path = identity_store.STORE_PATH
        identity_store.DATA_DIR = data_dir
        identity_store.STORE_PATH = data_dir / "watchlist_identity_store.json"

    def tearDown(self) -> None:
        self._identity_store.DATA_DIR = self._orig_data_dir
        self._identity_store.STORE_PATH = self._orig_store_path
        self._tmpdir.cleanup()

    def test_upsert_and_resolve_without_market_price_is_none(self) -> None:
        from app.ae13b_product.contract_resolver import STATUS_USER_ENTERED, resolve_identity

        addr = "0xidentitytest0000000000000000000000001"
        self._identity_store.upsert_identity(
            chain="bsc",
            address=addr,
            symbol="IDTEST",
            name="Identity Test Coin",
            source="watchlist_user_input",
        )

        looked_up = self._identity_store.get_identity("bsc", addr)
        self.assertIsNotNone(looked_up)
        self.assertFalse(looked_up["system_verified"])

        result = resolve_identity(chain="bsc", contract_or_pair_address=addr, allow_external=False)
        self.assertEqual(result["resolution_status"], STATUS_USER_ENTERED)
        self.assertIsNone(result.get("matched_price"))
        self.assertEqual(result["resolution_source"], "local_identity_store")
        self.assertIn("local store", (result.get("reason") or "").lower())


class SemanticIdentityStoreClassificationTests(unittest.TestCase):
    """7) Identity Store / user-claimed social mission + evidence, without a
    market match, must classify as SOCIAL_CANDIDATE_NEEDS_VERIFICATION and
    NEVER SOCIAL_CONFIRMED from user hypothesis alone."""

    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        data_dir = Path(self._tmpdir.name)

        import app.ae13_semantic.runtime_registry as runtime_registry
        import app.analytics.watchlist as watchlist
        import app.ae13b_product.identity_store as identity_store

        self._watchlist = watchlist
        self._orig_wl_data_dir = watchlist.DATA_DIR
        self._orig_wl_path = watchlist.WATCHLIST_PATH
        watchlist.DATA_DIR = data_dir
        watchlist.WATCHLIST_PATH = data_dir / "watchlist.json"

        self._identity_store = identity_store
        self._orig_id_data_dir = identity_store.DATA_DIR
        self._orig_id_store_path = identity_store.STORE_PATH
        identity_store.DATA_DIR = data_dir
        identity_store.STORE_PATH = data_dir / "watchlist_identity_store.json"

        self._registry_module = runtime_registry
        self._fake_registry = runtime_registry.SemanticRegistry(path=data_dir / "semantic_registry.json")
        self._orig_get_registry = runtime_registry.get_semantic_registry
        runtime_registry.get_semantic_registry = lambda: self._fake_registry

    def tearDown(self) -> None:
        self._watchlist.DATA_DIR = self._orig_wl_data_dir
        self._watchlist.WATCHLIST_PATH = self._orig_wl_path
        self._identity_store.DATA_DIR = self._orig_id_data_dir
        self._identity_store.STORE_PATH = self._orig_id_store_path
        self._registry_module.get_semantic_registry = self._orig_get_registry
        self._tmpdir.cleanup()

    def test_social_claim_without_market_match_needs_verification_not_confirmed(self) -> None:
        entry = self._watchlist.upsert_watchlist_item(
            symbol="GIVEBACK",
            chain="bsc",
            contract_address="0xgiveback00000000000000000000000000001",
            expected_category="user thinks social",
        )
        self._watchlist.set_watchlist_evidence(
            entry["id"],
            user_evidence_note="This project donates 10% of proceeds to verified charities.",
            user_claimed_social_mission="Charitable giving mission per founder statement.",
        )

        # No market match performed at all -- semantic check must still work.
        result = self._watchlist.run_watchlist_semantic_check(entry["id"])
        self.assertTrue(result["ok"])
        item = result["item"]

        self.assertEqual(item["semantic_classification"], "SOCIAL_CANDIDATE_NEEDS_VERIFICATION")
        self.assertNotEqual(item["semantic_classification"], "SOCIAL_CONFIRMED")
        self.assertIn(
            "user_supplied_social_claim_requires_validation",
            item.get("evidence_summary") or "",
        )
        self.assertTrue(item.get("semantic_independent_of_market_match"))


class MarkPositionsToMarketTests(_IsolatedPaperTraderCase):
    """9) mark_positions_to_market()/get_marked_positions() add current_price,
    unrealized PnL, and age fields to open positions."""

    def test_marked_positions_add_current_price_and_pnl_fields(self) -> None:
        pos = self.trader.open_position(
            _coin(1),
            size_usd=10.0,
            settings={},
            reason_code="TEST",
            strategy_type="MOMENTUM_SCOUT",
        )
        self.assertIsNotNone(pos)

        # Move the market price so unrealized PnL is non-trivial.
        self.trader.set_market_prices(
            [{"pair_address": "pair_1", "coin_id": 1, "price_usd": 1.5}],
            price_timestamp=_utc_now_iso(),
        )
        marked = self.trader.get_marked_positions()
        self.assertEqual(len(marked), 1)
        row = marked[0]
        self.assertAlmostEqual(row["current_price"], 1.5)
        self.assertIsNotNone(row["unrealized_pnl_usd"])
        self.assertGreater(row["unrealized_pnl_usd"], 0)
        self.assertIsNotNone(row["age_label"])
        self.assertIn("distance_to_take_profit_pct", row)
        self.assertIn("distance_to_stop_loss_pct", row)

    def test_missing_mark_price_leaves_current_price_none_with_reason(self) -> None:
        pos = self.trader.open_position(
            _coin(2),
            size_usd=10.0,
            settings={},
            reason_code="TEST",
            strategy_type="MOMENTUM_SCOUT",
        )
        self.assertIsNotNone(pos)
        # Clear all market prices so no mark is available.
        self.trader.set_market_prices([], price_timestamp=_utc_now_iso())
        marked = self.trader.get_marked_positions()
        row = next(p for p in marked if p["symbol"] == "SYM2")
        self.assertIsNone(row["current_price"])
        self.assertIsNotNone(row["mark_price_unavailable_reason"])


class SafetyFlagsRejectedAttemptTests(unittest.TestCase):
    """10) Safety flags paper_demo_only / not_live_approved (RejectedTradeAttempt surface)."""

    def test_rejected_trade_attempt_defaults_are_safe(self) -> None:
        from app.ae13b_product.rejected_attempt import RejectedTradeAttempt

        attempt = RejectedTradeAttempt(symbol="X")
        d = attempt.to_dict()
        self.assertTrue(d["paper_demo_only"])
        self.assertTrue(d["not_live_approved"])
        self.assertTrue(d["not_profitability_evidence"])
        self.assertFalse(d["risk_guard_passed"])


if __name__ == "__main__":
    unittest.main()
