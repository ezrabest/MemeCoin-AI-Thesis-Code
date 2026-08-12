"""AE13I Smoke Addendum - close freshness hard guard, demo queue GateKeeper
re-evaluation freshness, address alias cleanup, global text sanitizer, and
AE14 readiness.

Paper/demo only - no wallet, no live trading, no private keys, no
rebuild/retrain. These tests never start a long-lived server; any FastAPI
usage goes through TestClient only.
"""
from __future__ import annotations

import importlib
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_minutes_ago(minutes: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


# ---------------------------------------------------------------------------
# Part A (1-8): close_freshness hard guard
# ---------------------------------------------------------------------------


class CloseFreshnessHardGuardTests(unittest.TestCase):
    def test_proposed_price_with_empty_timestamp_is_unknown_or_fallback(self) -> None:
        """The exact smoke-test bug: proposed_price + no timestamp must NEVER be fresh."""
        from app.ae13b_product.close_freshness import classify_manual_close_freshness

        result = classify_manual_close_freshness(
            close_price=1.23,
            price_timestamp="",
            close_price_source="proposed_price",
            close_price_age_seconds=None,
        )
        self.assertEqual(result["close_freshness_status"], "unknown_or_fallback")
        self.assertTrue(result["close_used_fallback_price"])
        self.assertTrue(result["manual_close_warning_shown"])
        self.assertEqual(result["reason_code"], "MANUAL_CLOSE_WITH_STALE_OR_FALLBACK_PRICE")

    def test_proposed_price_source_alone_is_never_fresh_even_with_timestamp(self) -> None:
        from app.ae13b_product.close_freshness import classify_manual_close_freshness

        result = classify_manual_close_freshness(
            close_price=1.23,
            price_timestamp=_utc_now_iso(),
            close_price_source="proposed_price",
            close_price_age_seconds=1.0,
        )
        self.assertEqual(result["close_freshness_status"], "unknown_or_fallback")

    def test_missing_source_is_unknown_or_fallback(self) -> None:
        from app.ae13b_product.close_freshness import classify_manual_close_freshness

        result = classify_manual_close_freshness(
            close_price=1.0, price_timestamp=_utc_now_iso(), close_price_source=None,
            close_price_age_seconds=1.0,
        )
        self.assertEqual(result["close_freshness_status"], "unknown_or_fallback")

    def test_no_close_price_is_unknown_or_fallback(self) -> None:
        from app.ae13b_product.close_freshness import classify_manual_close_freshness

        result = classify_manual_close_freshness(
            close_price=None, price_timestamp=_utc_now_iso(), close_price_source="provider",
            close_price_age_seconds=1.0,
        )
        self.assertEqual(result["close_freshness_status"], "unknown_or_fallback")

    def test_age_over_threshold_is_unknown_or_fallback(self) -> None:
        from app.ae13b_product.close_freshness import classify_manual_close_freshness

        result = classify_manual_close_freshness(
            close_price=1.0,
            price_timestamp=_iso_minutes_ago(30),
            close_price_source="provider",
            close_price_age_seconds=1800.0,
            freshness_threshold_seconds=900,
        )
        self.assertEqual(result["close_freshness_status"], "unknown_or_fallback")

    def test_age_none_is_unknown_or_fallback(self) -> None:
        from app.ae13b_product.close_freshness import classify_manual_close_freshness

        result = classify_manual_close_freshness(
            close_price=1.0, price_timestamp=_utc_now_iso(), close_price_source="provider",
            close_price_age_seconds=None,
        )
        self.assertEqual(result["close_freshness_status"], "unknown_or_fallback")

    def test_all_fallback_sources_are_never_fresh(self) -> None:
        from app.ae13b_product.close_freshness import (
            FALLBACK_SOURCES,
            classify_manual_close_freshness,
        )

        for source in FALLBACK_SOURCES:
            result = classify_manual_close_freshness(
                close_price=1.0,
                price_timestamp=_utc_now_iso(),
                close_price_source=source,
                close_price_age_seconds=1.0,
            )
            self.assertEqual(
                result["close_freshness_status"], "unknown_or_fallback",
                msg=f"source={source} must never be fresh",
            )

    def test_genuinely_fresh_provider_price_is_fresh(self) -> None:
        from app.ae13b_product.close_freshness import classify_manual_close_freshness

        result = classify_manual_close_freshness(
            close_price=1.0,
            price_timestamp=_utc_now_iso(),
            close_price_source="provider",
            close_price_age_seconds=5.0,
        )
        self.assertEqual(result["close_freshness_status"], "fresh")
        self.assertFalse(result["close_used_fallback_price"])
        self.assertEqual(result["reason_code"], "MANUAL_SELL")

    def test_fresh_source_aliases_market_pair_address_and_db(self) -> None:
        from app.ae13b_product.close_freshness import classify_manual_close_freshness

        for source in ("market_pair_address", "market_coin_id", "db", "mark"):
            result = classify_manual_close_freshness(
                close_price=1.0,
                price_timestamp=_utc_now_iso(),
                close_price_source=source,
                close_price_age_seconds=5.0,
            )
            self.assertEqual(
                result["close_freshness_status"], "fresh", msg=f"source={source} should be fresh-equivalent"
            )

    def test_warning_text_matches_spec(self) -> None:
        from app.ae13b_product.close_freshness import MANUAL_CLOSE_FALLBACK_WARNING

        self.assertEqual(
            MANUAL_CLOSE_FALLBACK_WARNING,
            "Manual close will use last-known / fallback price. This price is not "
            "validated as fresh market data.",
        )


class PaperClosePositionFreshnessWiringTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        data_dir = Path(self._tmpdir.name) / "data"
        data_dir.mkdir(parents=True)
        os.environ["TRADER_DB_PATH"] = str(data_dir / "test.db")

        import app.ae13b_product.reentry_blocks as reentry_blocks
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

        self._reentry_blocks = reentry_blocks
        self._orig_blocks_path = reentry_blocks.blocks_file_path
        reentry_blocks.blocks_file_path = lambda: data_dir / "reentry_blocks.json"  # type: ignore[assignment]

        self.paper = paper
        self.trader = paper.PaperTrader()
        self.trader.set_market_prices(
            [{"pair_address": "pool_close_1", "coin_id": 910, "price_usd": 1.0}],
            price_timestamp=_utc_now_iso(),
        )

    def tearDown(self) -> None:
        self._reentry_blocks.blocks_file_path = self._orig_blocks_path
        self._tmpdir.cleanup()
        os.environ.pop("TRADER_DB_PATH", None)

    def _open(self) -> dict:
        pos = self.trader.open_position(
            {
                "symbol": "CLOSE/SOL",
                "chain": "solana",
                "pair_address": "pool_close_1",
                "coin_id": 910,
                "latest_price": 1.0,
                "price_updated_at": _utc_now_iso(),
                "latest_liquidity": 50000.0,
                "liquidity_updated_at": _utc_now_iso(),
                "source_provider": "dexscreener",
                "activity_delta_1h_pct": 5.0,
            },
            size_usd=10.0,
            settings={},
            reason_code="TEST",
        )
        self.assertIsNotNone(pos)
        return pos

    def test_manual_close_with_proposed_price_and_no_timestamp_is_never_fresh(self) -> None:
        """Reproduces the exact smoke-test bug at the PaperTrader layer."""
        pos = self._open()
        # Clear the market price timestamp so resolution.price_timestamp is
        # empty, and force fill-price resolution down the proposed_price path
        # by clearing the pair/coin price maps.
        self.trader._market_prices_by_pair = {}
        self.trader._market_prices_by_coin_id = {}
        self.trader._market_price_timestamp = None

        closed = self.trader.close_position(
            int(pos["id"]),
            1.10,
            reason_code="MANUAL_SELL",
            proposed_pair_address="pool_close_1",
            proposed_coin_id=910,
            closed_by="user_manual",
            close_price_source="proposed_price",
        )
        self.assertIsNotNone(closed)
        assert closed is not None
        self.assertEqual(closed["fill_price_source"], "proposed_price")
        self.assertEqual(closed["close_freshness_status"], "unknown_or_fallback")
        self.assertTrue(closed["close_used_fallback_price"])
        self.assertTrue(closed["manual_close_warning_shown"])
        self.assertEqual(closed["reason_code"], "MANUAL_CLOSE_WITH_STALE_OR_FALLBACK_PRICE")

    def test_manual_close_with_fresh_provider_price_is_fresh(self) -> None:
        pos = self._open()
        self.trader.set_market_prices(
            [{"pair_address": "pool_close_1", "coin_id": 910, "price_usd": 1.1}],
            price_timestamp=_utc_now_iso(),
        )
        closed = self.trader.close_position(
            int(pos["id"]), None, reason_code="MANUAL_SELL", closed_by="user_manual",
        )
        self.assertIsNotNone(closed)
        assert closed is not None
        self.assertEqual(closed["close_freshness_status"], "fresh")
        self.assertFalse(closed["close_used_fallback_price"])

    def test_caller_cannot_spoof_fresh_via_close_freshness_status_param(self) -> None:
        """Hard guard: passing close_freshness_status='fresh' explicitly must not
        override the real classification when the underlying data is not fresh."""
        pos = self._open()
        self.trader._market_prices_by_pair = {}
        self.trader._market_prices_by_coin_id = {}
        self.trader._market_price_timestamp = None

        closed = self.trader.close_position(
            int(pos["id"]),
            1.10,
            reason_code="MANUAL_SELL",
            proposed_pair_address="pool_close_1",
            proposed_coin_id=910,
            closed_by="user_manual",
            close_price_source="proposed_price",
            close_freshness_status="fresh",  # attempted spoof
            close_used_fallback_price=False,  # attempted spoof
        )
        self.assertIsNotNone(closed)
        assert closed is not None
        self.assertEqual(closed["close_freshness_status"], "unknown_or_fallback")
        self.assertTrue(closed["close_used_fallback_price"])

    def test_csv_row_persists_close_price_source(self) -> None:
        pos = self._open()
        self.trader.close_position(
            int(pos["id"]), 1.05, reason_code="MANUAL_SELL", closed_by="user_manual",
        )
        rows = self.trader.get_trades_from_log(limit=10)
        sell_rows = [r for r in rows if r.get("side") == "sell"]
        self.assertTrue(sell_rows)
        self.assertIn("close_price_source", sell_rows[-1])
        self.assertTrue(sell_rows[-1]["close_price_source"])


class ApiManualClosePriceSourceTests(unittest.TestCase):
    def test_resolve_manual_close_price_source_labels_explicit_price_as_proposed(self) -> None:
        from app.api import _resolve_manual_close_price_source
        from app.execution.paper import get_paper_trader

        info = _resolve_manual_close_price_source(
            {"pair_address": "pool_api_1", "coin_id": 1, "entry_price": 1.0},
            get_paper_trader(),
            2.5,
        )
        self.assertEqual(info["close_price_source"], "proposed_price")
        self.assertIsNone(info["price_timestamp"])
        self.assertEqual(info["close_price"], 2.5)

    def test_manual_close_response_warning_matches_spec_text(self) -> None:
        from app.ae13b_product.close_freshness import MANUAL_CLOSE_FALLBACK_WARNING
        from app.api import _manual_close_response

        closed = {
            "id": 5,
            "closed_by": "user_manual",
            "close_used_fallback_price": True,
            "close_freshness_status": "unknown_or_fallback",
        }
        payload = _manual_close_response(closed)
        self.assertEqual(payload["warning"], MANUAL_CLOSE_FALLBACK_WARNING)


# ---------------------------------------------------------------------------
# Part B (9-15): demo queue GateKeeper re-evaluation freshness
# ---------------------------------------------------------------------------


class DemoQueueEvaluationFreshnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        import app.ae13b_product.demo_queue as demo_queue

        importlib.reload(demo_queue)
        self.demo_queue = demo_queue
        self.demo_queue.QUEUE_PATH = Path(self._tmpdir.name) / "demo_trade_queue.json"

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_never_evaluated_item_is_stale(self) -> None:
        entry = self.demo_queue.add_to_demo_queue(symbol="NEW", pair="pool_new_1", chain="solana")
        item = self.demo_queue.get_queue_item(entry["queue_id"])
        self.assertIsNotNone(item)
        assert item is not None
        self.assertTrue(item["evaluation_stale"])
        self.assertEqual(item["evaluation_stale_reason"], "Evaluation stale - click Evaluate Now.")
        self.assertFalse(item["gatekeeper_evaluated"])

    def test_pre_ae13i_evaluated_item_without_gate_result_is_stale(self) -> None:
        entry = self.demo_queue.add_to_demo_queue(symbol="OLD", pair="pool_old_1", chain="solana")
        # Simulate a pre-AE13I evaluation: last_evaluated_at set, but no
        # gate_result / gatekeeper_status persisted (old schema).
        self.demo_queue.update_queue_evaluation(
            entry["queue_id"], last_decision="WATCH", extra={"legacy_note": "pre_ae13i"},
        )
        item = self.demo_queue.get_queue_item(entry["queue_id"])
        assert item is not None
        self.assertTrue(item["evaluation_stale"])

    def test_recently_gate_evaluated_item_is_not_stale(self) -> None:
        entry = self.demo_queue.add_to_demo_queue(symbol="FRESHQ", pair="pool_freshq_1", chain="solana")
        self.demo_queue.update_queue_evaluation(
            entry["queue_id"],
            last_decision="BLOCKED",
            extra={
                "gate_result": {"passed": False},
                "last_gatekeeper_evaluated_at": _utc_now_iso(),
                "gatekeeper_status": "fail",
                "gatekeeper_evaluated": True,
            },
        )
        item = self.demo_queue.get_queue_item(entry["queue_id"])
        assert item is not None
        self.assertFalse(item["evaluation_stale"])
        self.assertTrue(item["gatekeeper_evaluated"])
        self.assertEqual(item["gatekeeper_status"], "fail")

    def test_old_gate_evaluation_becomes_stale_again(self) -> None:
        entry = self.demo_queue.add_to_demo_queue(symbol="OLDQ", pair="pool_oldq_1", chain="solana")
        self.demo_queue.update_queue_evaluation(
            entry["queue_id"],
            last_decision="BLOCKED",
            extra={
                "gate_result": {"passed": False},
                "last_gatekeeper_evaluated_at": _iso_minutes_ago(30),
                "gatekeeper_status": "fail",
                "gatekeeper_evaluated": True,
            },
        )
        item = self.demo_queue.get_queue_item(entry["queue_id"])
        assert item is not None
        self.assertTrue(item["evaluation_stale"])

    def test_list_demo_queue_attaches_freshness_fields_to_every_item(self) -> None:
        self.demo_queue.add_to_demo_queue(symbol="A", pair="pool_a_1", chain="solana")
        self.demo_queue.add_to_demo_queue(symbol="B", pair="pool_b_1", chain="solana")
        items = self.demo_queue.list_demo_queue()
        self.assertEqual(len(items), 2)
        for item in items:
            for field in (
                "last_gatekeeper_evaluated_at", "gatekeeper_status", "tradability_status",
                "freshness_gate_status", "provenance_status", "address_role",
                "market_data_status", "evaluation_stale", "evaluation_stale_reason",
                "gatekeeper_evaluated",
            ):
                self.assertIn(field, item, msg=f"missing {field}")

    def test_evaluate_queue_item_runs_manual_cooldown_before_gate(self) -> None:
        import app.ae13b_product.reentry_blocks as reentry_blocks

        tmp_path = Path(self._tmpdir.name) / "reentry_blocks.json"
        orig = reentry_blocks.blocks_file_path
        reentry_blocks.blocks_file_path = lambda: tmp_path  # type: ignore[assignment]
        try:
            reentry_blocks.add_manual_close_block(
                {"id": 1, "pair_address": "pool_cool_q1", "chain": "solana", "symbol": "COOLQ"},
                "manual_take_profit",
                duration_seconds=3600,
            )
            entry = self.demo_queue.add_to_demo_queue(
                symbol="COOLQ", pair="pool_cool_q1", chain="solana",
            )
            result = self.demo_queue.evaluate_queue_item(entry["queue_id"])
            self.assertEqual(result["decision"], "BLOCKED_MANUAL_REENTRY_COOLDOWN")
            self.assertTrue(result["manual_cooldown_active"])
            # gate_result must NOT be populated - cooldown precheck short-circuits
            # before GateKeeper/RiskGuard ever run.
            self.assertNotIn("gate_result", result)
        finally:
            reentry_blocks.blocks_file_path = orig

    def test_evaluate_queue_item_persists_gatekeeper_fields(self) -> None:
        entry = self.demo_queue.add_to_demo_queue(
            symbol="GATEQ", pair="pool_gate_q1", chain="solana",
        )
        result = self.demo_queue.evaluate_queue_item(entry["queue_id"])
        self.assertIn("gate_result", result)
        item = result["queue_item"]
        self.assertIn("gatekeeper_status", item)
        self.assertIn("last_gatekeeper_evaluated_at", item)
        self.assertTrue(item.get("gatekeeper_evaluated"))
        self.assertFalse(item.get("evaluation_stale"))

    def _mock_resolution(self, *, pair: str, symbol: str) -> dict:
        return {
            "matched_chain": "solana",
            "matched_symbol": symbol,
            "matched_pair_address": pair,
            "matched_price": 1.0,
            "matched_price_ts": _utc_now_iso(),
            "matched_liquidity": 50000.0,
            "resolution_source": "dexscreener",
            "resolution_status": "matched_live_market",
            "matched_name": symbol,
            "matched_token_contract_address": None,
            "matched_token_mint_address": None,
        }

    def test_evaluate_queue_item_refreshes_risk_mode_from_active_preset(self) -> None:
        from unittest.mock import patch

        entry = self.demo_queue.add_to_demo_queue(
            symbol="RISKQ", pair="pool_risk_q1", chain="solana",
        )
        # Item inherits active preset (default). Force a stale snapshot value
        # onto the item directly, simulating an old bot-preset switch that the
        # item never picked up.
        self.demo_queue.update_queue_evaluation(
            entry["queue_id"], last_decision="WATCH", extra={"risk_mode": "stale_balanced_snapshot"},
        )
        # Fresh market data is required for evaluation to reach the risk-guard
        # stage (a GateKeeper block short-circuits before risk_mode is read).
        with patch(
            "app.ae13b_product.contract_resolver.resolve_identity",
            return_value=self._mock_resolution(pair="pool_risk_q1", symbol="RISKQ"),
        ):
            result = self.demo_queue.evaluate_queue_item(entry["queue_id"])
        active_profile = self.demo_queue.get_active_demo_risk_profile()
        self.assertIn("risk_mode", result)
        self.assertEqual(result["risk_mode"], active_profile["risk_mode"])
        self.assertNotEqual(result["risk_mode"], "stale_balanced_snapshot")

    def test_explicit_risk_mode_is_not_overridden_by_active_preset(self) -> None:
        from unittest.mock import patch

        entry = self.demo_queue.add_to_demo_queue(
            symbol="EXPLICITQ", pair="pool_explicit_q1", chain="solana", risk_mode="aggressive",
        )
        self.assertFalse(entry["inherits_active_bot_preset"])
        with patch(
            "app.ae13b_product.contract_resolver.resolve_identity",
            return_value=self._mock_resolution(pair="pool_explicit_q1", symbol="EXPLICITQ"),
        ):
            result = self.demo_queue.evaluate_queue_item(entry["queue_id"])
        self.assertIn("risk_mode", result)
        self.assertEqual(result["risk_mode"], "aggressive")


# ---------------------------------------------------------------------------
# Part C (16-18): address alias cleanup
# ---------------------------------------------------------------------------


class AddressAliasCleanupTests(unittest.TestCase):
    def test_pool_address_role_produces_deprecated_alias_disclosure(self) -> None:
        from app.ae13b_product.live_market import compute_contract_address_disclosure

        result = compute_contract_address_disclosure(
            raw_contract_address=None,
            address_role="pool_address",
            token_contract_address=None,
            token_mint_address=None,
            pair_address="9VW8yfZaf2GcEpVb4apuk63oGVnebYZ4pr7ymc8Ftx3i",
        )
        self.assertEqual(result["contract_address"], "9VW8yfZaf2GcEpVb4apuk63oGVnebYZ4pr7ymc8Ftx3i")
        self.assertTrue(result["contract_address_deprecated"])
        self.assertEqual(result["contract_address_role"], "pool_address_alias")
        self.assertIn("pool/pair address", result["contract_address_warning"])
        self.assertEqual(result["address_display_label"], "Pool / Pair address")

    def test_pair_contract_role_produces_pair_alias(self) -> None:
        from app.ae13b_product.live_market import compute_contract_address_disclosure

        result = compute_contract_address_disclosure(
            raw_contract_address=None,
            address_role="pair_contract",
            token_contract_address=None,
            token_mint_address=None,
            pair_address="0xd2391dB4D7B9841b989521088c3Bf8C4cFe404d8",
        )
        self.assertEqual(result["contract_address_role"], "pair_address_alias")
        self.assertEqual(result["address_display_label"], "Pool / Pair address")

    def test_actual_token_contract_role_has_no_warning(self) -> None:
        from app.ae13b_product.live_market import compute_contract_address_disclosure

        result = compute_contract_address_disclosure(
            raw_contract_address=None,
            address_role="token_contract",
            token_contract_address="0xTOKENCONTRACT",
            token_mint_address=None,
            pair_address="0xPAIRADDR",
        )
        self.assertIsNone(result["contract_address_role"])
        self.assertIsNone(result["contract_address_warning"])
        self.assertEqual(result["contract_address"], "0xTOKENCONTRACT")
        self.assertEqual(result["address_display_label"], "Contract address")

    def test_unknown_role_uses_unknown_legacy_alias(self) -> None:
        from app.ae13b_product.live_market import compute_contract_address_disclosure

        result = compute_contract_address_disclosure(
            raw_contract_address="somevalue",
            address_role="unknown_or_provider_pair",
            token_contract_address=None,
            token_mint_address=None,
            pair_address=None,
        )
        self.assertEqual(result["contract_address_role"], "unknown_legacy_alias")

    def test_does_not_silently_conflate_pair_and_contract_without_disclosure(self) -> None:
        """Regression for the exact smoke finding: contract_address must always
        carry disclosure metadata when it is really just the pair address."""
        from app.ae13b_product.live_market import compute_contract_address_disclosure

        result = compute_contract_address_disclosure(
            raw_contract_address=None,
            address_role="pool_address",
            token_contract_address=None,
            token_mint_address=None,
            pair_address="poolABC123",
        )
        self.assertEqual(result["contract_address"], "poolABC123")
        self.assertTrue(result["contract_address_deprecated"])
        self.assertIsNotNone(result["contract_address_role"])
        self.assertIsNotNone(result["contract_address_warning"])


# ---------------------------------------------------------------------------
# Part D (19-20): global text sanitizer
# ---------------------------------------------------------------------------


class TextSanitizerTests(unittest.TestCase):
    def test_sanitize_text_replaces_unicode_dashes_and_ellipsis(self) -> None:
        from app.ae13b_product.text_sanitizer import sanitize_text

        self.assertEqual(sanitize_text("a \u2014 b"), "a - b")
        self.assertEqual(sanitize_text("a \u2013 b"), "a - b")
        self.assertEqual(sanitize_text("loading\u2026"), "loading...")

    def test_sanitize_text_repairs_mojibake(self) -> None:
        from app.ae13b_product.text_sanitizer import sanitize_text

        mojibake = "paper demo \u00e2\u0080\u0094 not live"
        out = sanitize_text(mojibake)
        self.assertNotIn("\u00e2", out)
        self.assertIn("-", out)

    def test_sanitize_payload_recurses_dict_and_list(self) -> None:
        from app.ae13b_product.text_sanitizer import sanitize_payload

        payload = {
            "note": "paper \u2014 demo",
            "rows": [{"label": "a \u2026 b"}, {"count": 3, "flag": True, "empty": None}],
            "count": 5,
        }
        out = sanitize_payload(payload)
        self.assertEqual(out["note"], "paper - demo")
        self.assertEqual(out["rows"][0]["label"], "a ... b")
        self.assertEqual(out["rows"][1]["count"], 3)
        self.assertTrue(out["rows"][1]["flag"])
        self.assertIsNone(out["rows"][1]["empty"])
        self.assertEqual(out["count"], 5)

    def test_sanitize_payload_preserves_non_string_leaves(self) -> None:
        from app.ae13b_product.text_sanitizer import sanitize_payload

        self.assertEqual(sanitize_payload(42), 42)
        self.assertEqual(sanitize_payload(3.14), 3.14)
        self.assertIsNone(sanitize_payload(None))
        self.assertEqual(sanitize_payload(True), True)

    def test_static_ui_files_are_ascii_safe(self) -> None:
        for rel_path in ("static/index.html", "static/product_demo.js"):
            text = (ROOT / rel_path).read_text(encoding="utf-8")
            for bad_char in ("\u2014", "\u2013", "\u2026"):
                self.assertNotIn(
                    bad_char, text, msg=f"{rel_path} still contains unicode char {bad_char!r}"
                )


# ---------------------------------------------------------------------------
# Part E (21): AE14 readiness
# ---------------------------------------------------------------------------


class Ae14ReadinessTests(unittest.TestCase):
    def test_no_tradable_rows_yields_negative_control_only(self) -> None:
        from app.ae13b_product.ae14_readiness import (
            NEGATIVE_CONTROL_REASON,
            compute_ae14_readiness,
        )

        result = compute_ae14_readiness(market_rows=[{"tradability_status": "stale_market_data"}] * 5)
        self.assertTrue(result["ready_for_negative_control"])
        self.assertFalse(result["ready_for_trading_validation"])
        self.assertEqual(result["reason"], NEGATIVE_CONTROL_REASON)
        self.assertEqual(result["tradable_now_count"], 0)

    def test_empty_rows_yields_negative_control_only(self) -> None:
        from app.ae13b_product.ae14_readiness import compute_ae14_readiness

        result = compute_ae14_readiness(market_rows=[])
        self.assertTrue(result["ready_for_negative_control"])
        self.assertFalse(result["ready_for_trading_validation"])
        self.assertEqual(result["total_rows"], 0)

    def test_none_rows_treated_as_empty_not_an_error(self) -> None:
        from app.ae13b_product.ae14_readiness import compute_ae14_readiness

        result = compute_ae14_readiness(market_rows=None)
        self.assertTrue(result["ready_for_negative_control"])
        self.assertFalse(result["ready_for_trading_validation"])

    def test_enough_tradable_rows_enables_trading_validation(self) -> None:
        from app.ae13b_product.ae14_readiness import compute_ae14_readiness

        rows = [{"tradability_status": "tradable_now"} for _ in range(12)]
        result = compute_ae14_readiness(market_rows=rows, min_tradable_rows_for_ae14=10)
        self.assertTrue(result["ready_for_negative_control"])
        self.assertTrue(result["ready_for_trading_validation"])
        self.assertEqual(result["tradable_now_count"], 12)

    def test_below_threshold_tradable_rows_blocks_trading_validation(self) -> None:
        from app.ae13b_product.ae14_readiness import compute_ae14_readiness

        rows = [{"tradability_status": "tradable_now"} for _ in range(3)]
        result = compute_ae14_readiness(market_rows=rows, min_tradable_rows_for_ae14=10)
        self.assertTrue(result["ready_for_negative_control"])
        self.assertFalse(result["ready_for_trading_validation"])
        self.assertEqual(result["tradable_now_count"], 3)

    def test_safety_flags_present(self) -> None:
        from app.ae13b_product.ae14_readiness import compute_ae14_readiness

        result = compute_ae14_readiness(market_rows=[])
        self.assertTrue(result["paper_demo_only"])
        self.assertTrue(result["not_live_approved"])
        self.assertFalse(result["live_trading_implied"])


# ---------------------------------------------------------------------------
# API-level tests (TestClient only, no live server)
# ---------------------------------------------------------------------------


class ApiEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from app.api import app

        self.client = TestClient(app)

    def test_ae14_readiness_endpoint_returns_required_fields(self) -> None:
        resp = self.client.get("/api/ae14/readiness")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        for field in (
            "ready_for_negative_control", "ready_for_trading_validation", "reason",
            "tradable_now_count", "stale_count", "recommended_next_action",
            "paper_demo_only", "not_live_approved",
        ):
            self.assertIn(field, data, msg=f"missing {field}")

    def test_demo_bot_status_includes_ae14_readiness(self) -> None:
        resp = self.client.get("/api/ae13b/demo-bot/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("ae14_readiness", data)

    def test_demo_queue_endpoint_is_ascii_safe(self) -> None:
        resp = self.client.get("/api/demo-queue")
        self.assertEqual(resp.status_code, 200)
        body_text = resp.text
        for bad_char in ("\u2014", "\u2013", "\u2026"):
            self.assertNotIn(bad_char, body_text)

    def test_watchlist_endpoint_is_ascii_safe(self) -> None:
        resp = self.client.get("/api/watchlist")
        self.assertEqual(resp.status_code, 200)
        for bad_char in ("\u2014", "\u2013", "\u2026"):
            self.assertNotIn(bad_char, resp.text)

    def test_live_market_endpoint_returns_address_disclosure_fields_when_rows_present(self) -> None:
        resp = self.client.get("/api/ae13b/live-market", params={"limit": 5})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        rows = data.get("rows") or []
        for row in rows:
            self.assertIn("contract_address_deprecated", row)
            self.assertIn("address_display_label", row)


# ---------------------------------------------------------------------------
# Part F/G: regression - do not weaken GateKeeper / reentry / stagnant guards
# ---------------------------------------------------------------------------


class RegressionGuardsStillBlockTests(unittest.TestCase):
    def setUp(self) -> None:
        import app.ae13b_product.reentry_blocks as reentry_blocks

        self._tmpdir = tempfile.TemporaryDirectory()
        self._reentry_blocks = reentry_blocks
        self._orig_blocks_path = reentry_blocks.blocks_file_path
        tmp_path = Path(self._tmpdir.name) / "reentry_blocks.json"
        reentry_blocks.blocks_file_path = lambda: tmp_path  # type: ignore[assignment]

    def tearDown(self) -> None:
        self._reentry_blocks.blocks_file_path = self._orig_blocks_path
        self._tmpdir.cleanup()

    def _fresh_row(self, **overrides) -> dict:
        row = {
            "chain": "solana",
            "symbol": "REG/SOL",
            "pair_address": "pool_reg_111",
            "latest_price": 1.23,
            "price_updated_at": _utc_now_iso(),
            "latest_liquidity": 50000.0,
            "liquidity_updated_at": _utc_now_iso(),
            "source_provider": "dexscreener",
        }
        row.update(overrides)
        return row

    def test_stale_price_still_blocks(self) -> None:
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        row = self._fresh_row(price_updated_at=_iso_minutes_ago(60), price_age_seconds=3600.0)
        gate = validate_market_data_gate(row, for_open=True)
        self.assertFalse(gate["passed"])
        self.assertIn("freshness_stale_price", gate["blocking_guards"])

    def test_missing_price_still_blocks(self) -> None:
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        gate = validate_market_data_gate(self._fresh_row(latest_price=None), for_open=True)
        self.assertFalse(gate["passed"])

    def test_manual_reentry_cooldown_still_blocks(self) -> None:
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        self._reentry_blocks.add_manual_close_block(
            {"id": 1, "pair_address": "pool_reg_cooldown", "chain": "solana", "symbol": "RCOOL"},
            "manual_take_profit",
            duration_seconds=3600,
        )
        row = self._fresh_row(pair_address="pool_reg_cooldown", symbol="RCOOL")
        gate = validate_market_data_gate(row, for_open=True)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["rejection_code"], "MANUAL_REENTRY_BLOCK_ACTIVE")

    def test_stagnant_price_guard_still_blocks(self) -> None:
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        row = self._fresh_row(activity_delta_4h_pct=0.05)
        gate = validate_market_data_gate(row, for_open=True)  # skip_stagnant defaults False
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["rejection_code"], "PRICE_STAGNANT_NO_RECENT_MOMENTUM")

    def test_fully_fresh_row_with_momentum_still_passes(self) -> None:
        from app.ae13b_product.market_data_gatekeeper import validate_market_data_gate

        row = self._fresh_row(activity_delta_1h_pct=5.0, activity_delta_4h_pct=8.0)
        gate = validate_market_data_gate(row, for_open=True)
        self.assertTrue(gate["passed"], gate)
        self.assertEqual(gate["tradability_status"], "tradable_now")

    def test_demo_queue_evaluate_still_gate_first_before_risk(self) -> None:
        src = (ROOT / "app" / "ae13b_product" / "demo_queue.py").read_text(encoding="utf-8")
        gate_idx = src.index("validate_market_data_gate(gate_row")
        risk_idx = src.index("evaluate_demo_risk_guard(")
        self.assertLess(gate_idx, risk_idx, msg="GateKeeper must run before RiskGuard in demo_queue")

    def test_close_freshness_hard_guard_cannot_be_weakened_by_callers(self) -> None:
        """paper.close_position must not accept a caller override that marks
        a fallback price as fresh (this was the original smoke bug)."""
        from app.ae13b_product.close_freshness import classify_manual_close_freshness

        # Even if a hypothetical caller tries every fallback source, none can
        # ever classify as fresh.
        for source in ("proposed_price", "fallback", "entry_price", "last_known", "entry", "proposed"):
            result = classify_manual_close_freshness(
                close_price=1.0,
                price_timestamp=_utc_now_iso(),
                close_price_source=source,
                close_price_age_seconds=1.0,
            )
            self.assertNotEqual(result["close_freshness_status"], "fresh")


if __name__ == "__main__":
    unittest.main()
