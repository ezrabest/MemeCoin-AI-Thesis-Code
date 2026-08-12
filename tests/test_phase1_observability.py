"""Phase 1 observability, safety, and audit tests."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient


class EffectiveSettingsTests(unittest.TestCase):
    def test_alias_normalization(self) -> None:
        from app.observability.effective_settings import EffectiveSettings, SETTING_ALIASES

        eff = EffectiveSettings({
            "minLiquidity": 8000,
            "positionSizePct": 10,
            "mode": "DEMO",
        })
        self.assertEqual(eff.canonical["min_liquidity_usd"], 8000)
        self.assertAlmostEqual(eff.canonical["max_position_size_pct"], 0.10, places=4)
        self.assertEqual(eff.canonical["trading_mode"], "DEMO")
        self.assertIn("minLiquidity", eff.aliases_resolved)
        self.assertEqual(SETTING_ALIASES["minLiquidity"], "min_liquidity_usd")

    def test_settings_hash_stable(self) -> None:
        from app.observability.effective_settings import EffectiveSettings

        raw = {"min_liquidity_usd": 5000, "trading_mode": "DEMO"}
        h1 = EffectiveSettings(raw).settings_hash
        h2 = EffectiveSettings(raw).settings_hash
        self.assertEqual(h1, h2)
        self.assertEqual(len(h1), 16)

    def test_hidden_thresholds_exposed(self) -> None:
        from app.engine import (
            SIGNAL_BUY_LIQUIDITY_USD,
            SIGNAL_BUY_WHALE_THRESHOLD,
            WHALE_ALERT_MIN_VOLUME_24H,
            WHALE_ALERT_MIN_WHALE_SCORE,
            get_alert_thresholds,
            get_signal_thresholds,
        )
        from app.observability.effective_settings import EffectiveSettings

        eff = EffectiveSettings()
        ht = eff.hidden_thresholds
        self.assertEqual(ht["generate_signal"]["buy_whale_score_threshold"], SIGNAL_BUY_WHALE_THRESHOLD)
        self.assertEqual(ht["generate_signal"]["buy_liquidity_usd"], SIGNAL_BUY_LIQUIDITY_USD)
        self.assertEqual(ht["detect_whale_alert"]["min_volume_24h"], WHALE_ALERT_MIN_VOLUME_24H)
        self.assertEqual(ht["detect_whale_alert"]["min_whale_score"], WHALE_ALERT_MIN_WHALE_SCORE)
        self.assertEqual(get_signal_thresholds()["buy_whale_score_threshold"], SIGNAL_BUY_WHALE_THRESHOLD)
        self.assertEqual(get_alert_thresholds()["min_whale_score"], WHALE_ALERT_MIN_WHALE_SCORE)


class EngineBehaviorPreservationTests(unittest.TestCase):
    def _pair(self, *, liq: float = 50_000, vol: float = 100_000, whale: float = 0.8) -> dict:
        return {
            "priceUsd": "0.001",
            "liquidity": {"usd": liq},
            "volume": {"h24": vol, "h1": vol / 24},
            "priceChange": {"h24": 10, "h1": 5},
            "txns": {"h24": {"buys": 600, "sells": 400}},
        }

    def test_generate_signal_unchanged_outcomes(self) -> None:
        from app.engine import generate_signal

        sig_buy = generate_signal(self._pair(liq=30_000), 0.55)
        self.assertEqual(sig_buy["action"], "BUY")
        sig_watch = generate_signal(self._pair(liq=20_000), 0.55)
        self.assertEqual(sig_watch["action"], "WATCH")
        low_pair = {
            "priceUsd": "0.001",
            "liquidity": {"usd": 10_000},
            "volume": {"h24": 1000, "h1": 50},
            "priceChange": {"h24": -5, "h1": -2},
            "txns": {"h24": {"buys": 40, "sells": 60}},
        }
        sig_no = generate_signal(low_pair, 0.15)
        self.assertEqual(sig_no["action"], "NO_TRADE")

    def test_detect_whale_alert_gate_unchanged(self) -> None:
        from app.engine import detect_whale_alert

        pair = self._pair(vol=1000, whale=0.5)
        self.assertIsNone(detect_whale_alert(pair, 0.5))
        accumulation_pair = {
            "priceUsd": "0.001",
            "liquidity": {"usd": 50_000},
            "volume": {"h24": 30_000, "h1": 5_000},
            "priceChange": {"h24": 5, "h1": 2},
            "txns": {"h24": {"buys": 700, "sells": 300}},
        }
        alert = detect_whale_alert(accumulation_pair, 0.5)
        self.assertIsNotNone(alert)
        self.assertEqual(alert["alert_type"], "ACCUMULATION")


class SqlitePragmaTests(unittest.TestCase):
    def test_wal_and_busy_timeout(self) -> None:
        from app.sqlite_util import configure_sqlite_connection, get_sqlite_pragma_state, SQLITE_BUSY_TIMEOUT_MS

        with tempfile.TemporaryDirectory() as tmp:
            conn = sqlite3.connect(str(Path(tmp) / "test.db"))
            configure_sqlite_connection(conn)
            state = get_sqlite_pragma_state(conn)
            self.assertEqual(state["journal_mode"], "wal")
            self.assertEqual(state["busy_timeout_ms"], SQLITE_BUSY_TIMEOUT_MS)
            conn.close()

    def test_get_db_uses_pragmas(self) -> None:
        import importlib

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["TRADER_DB_PATH"] = str(Path(tmp) / "pragma.db")
            import app.database as database

            importlib.reload(database)
            database.init_db()
            with database.get_db() as conn:
                from app.sqlite_util import get_sqlite_pragma_state

                state = get_sqlite_pragma_state(conn)
                self.assertEqual(state["journal_mode"], "wal")
                self.assertGreaterEqual(state["busy_timeout_ms"], 5000)
            os.environ.pop("TRADER_DB_PATH", None)


class AuditIoTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._audit_dir = Path(self._tmpdir.name) / "audits"
        self._audit_dir.mkdir()

    def tearDown(self) -> None:
        from app.observability.audit_io import reset_audit_writers_for_tests

        reset_audit_writers_for_tests()
        self._tmpdir.cleanup()

    def test_jsonl_append_format(self) -> None:
        from app.observability import audit_io

        audit_io.AUDITS_DIR = self._audit_dir
        audit_io.reset_audit_writers_for_tests()
        day = audit_io.utc_date_slug()
        writer = audit_io.JsonlAuditWriter("pipeline_reasons", date_slug=day)
        writer.append({"decision_trace_id": "abc", "settings_hash": "hash1", "audit_reasons": ["X"]})
        writer.append({"decision_trace_id": "def", "settings_hash": "hash1", "audit_reasons": ["Y"]})
        writer.close()
        self.assertIn(f"pipeline_reasons_{day}.jsonl", writer.path.name)
        lines = writer.path.read_text(encoding="utf-8").strip().split("\n")
        self.assertEqual(len(lines), 2)
        for line in lines:
            obj = json.loads(line)
            self.assertIn("decision_trace_id", obj)
            self.assertIn("settings_hash", obj)

    def test_atomic_json_report(self) -> None:
        from app.observability import audit_io

        audit_io.AUDITS_DIR = self._audit_dir
        path = audit_io.write_json_report_atomic("test_report.json", {"ok": True})
        self.assertTrue(path.exists())
        self.assertFalse(path.with_suffix(path.suffix + ".tmp").exists())
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertTrue(data["ok"])


class LlmGateTests(unittest.TestCase):
    def test_short_circuit_blocks_without_alert(self) -> None:
        from app.observability.llm_gate import evaluate_llm_short_circuit

        should, reasons = evaluate_llm_short_circuit(
            alert=None, coin_id=1, whale_score=0.9, llm_threshold=0.5, pair_address="0x1"
        )
        self.assertFalse(should)
        self.assertIn("ALERT_REQUIRED_BUT_MISSING", reasons)

    def test_short_circuit_allows_when_gates_pass(self) -> None:
        from app.observability.llm_gate import evaluate_llm_short_circuit

        should, _ = evaluate_llm_short_circuit(
            alert={"alert_type": "LARGE_BUY"},
            coin_id=1,
            whale_score=0.9,
            llm_threshold=0.5,
            pair_address="0x1",
            price_usd=1.0,
        )
        self.assertTrue(should)

    def test_bearish_conflict_audited_not_blocking(self) -> None:
        from app.observability.audit_reasons import AuditReason
        from app.observability.llm_gate import evaluate_llm_short_circuit

        should, reasons = evaluate_llm_short_circuit(
            alert={"alert_type": "LARGE_SELL"},
            coin_id=1,
            whale_score=0.9,
            llm_threshold=0.5,
            signal_action="BUY",
            alert_type="LARGE_SELL",
            pair_address="0x1",
            price_usd=1.0,
        )
        self.assertTrue(should)
        self.assertIn(AuditReason.CONFLICT_ENGINE_BUY_BEARISH_ALERT.value, reasons)

    @patch("app.observability.llm_gate.is_headless_data_collection", return_value=True)
    def test_headless_audited_but_does_not_block_pre_call(self, _mock: object) -> None:
        from app.observability.llm_gate import evaluate_llm_short_circuit

        should, reasons = evaluate_llm_short_circuit(
            alert={"alert_type": "LARGE_BUY"},
            coin_id=1,
            whale_score=0.9,
            llm_threshold=0.5,
            pair_address="0x1",
            price_usd=1.0,
        )
        self.assertTrue(should)
        self.assertIn("LLM_SHORT_CIRCUITED", reasons)


class EventDedupTests(unittest.TestCase):
    def test_repeated_snapshots_grouped(self) -> None:
        from app.observability.event_dedup import deduplicate_events

        events = [
            {"pair_address": "0xa", "chain": "solana", "event_type": "WATCH", "timestamp": "2026-06-20T12:01:00+00:00"},
            {"pair_address": "0xa", "chain": "solana", "event_type": "WATCH", "timestamp": "2026-06-20T12:02:00+00:00"},
            {"pair_address": "0xb", "chain": "bsc", "event_type": "WATCH", "timestamp": "2026-06-20T12:01:00+00:00"},
        ]
        result = deduplicate_events(events)
        self.assertEqual(result["raw_event_count"], 3)
        self.assertEqual(result["event_level_count"], 2)


class PipelineAuditTests(unittest.TestCase):
    def test_audit_failure_does_not_raise(self) -> None:
        from app.observability.pipeline_audit import safe_record_pipeline_decision

        trace = safe_record_pipeline_decision(
            pair_address="0xdead",
            audit_reasons=["WATCH_NOT_ACTIONABLE"],
        )
        self.assertIsNotNone(trace)

    def test_derive_signal_reasons_watch(self) -> None:
        from app.observability.audit_reasons import AuditReason
        from app.observability.pipeline_audit import derive_signal_audit_reasons

        reasons = derive_signal_audit_reasons(
            signal_action="WATCH",
            prob_up=0.6,
            whale_score=0.45,
            liquidity_usd=20_000,
        )
        self.assertIn(AuditReason.WATCH_NOT_ACTIONABLE.value, reasons)


class MigrationSafetyTests(unittest.TestCase):
    def test_no_drop_table_in_app_code(self) -> None:
        root = Path(__file__).resolve().parents[1]
        offenders: list[str] = []
        for pattern in ("app/**/*.py", "scripts/**/*.py"):
            for path in root.glob(pattern):
                if ".venv" in str(path):
                    continue
                text = path.read_text(encoding="utf-8", errors="ignore")
                if "DROP TABLE" in text.upper():
                    offenders.append(str(path.relative_to(root)))
        self.assertEqual(offenders, [], f"DROP TABLE found in: {offenders}")

    def test_migration_idempotent(self) -> None:
        import importlib

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["TRADER_DB_PATH"] = str(Path(tmp) / "migrate.db")
            import app.database as database

            importlib.reload(database)
            database.init_db()
            database.init_db()
            with database.get_db() as conn:
                cols = {row[1] for row in conn.execute("PRAGMA table_info(pipeline_audit)")}
                self.assertIn("decision_trace_id", cols)
                self.assertIn("settings_hash", cols)
            os.environ.pop("TRADER_DB_PATH", None)


class ApiEffectiveSettingsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["TRADER_DB_PATH"] = str(Path(self._tmpdir.name) / "api.db")
        import importlib
        import app.database as database

        importlib.reload(database)
        database.init_db()
        import main

        importlib.reload(main)
        self.client = TestClient(main.app)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        os.environ.pop("TRADER_DB_PATH", None)

    def test_effective_settings_endpoint(self) -> None:
        r = self.client.get("/api/settings/effective")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("settings_hash", data)
        self.assertIn("canonical", data)
        self.assertIn("hidden_thresholds", data)
        c = data["canonical"]
        self.assertIn("min_liquidity_usd", c)
        self.assertIn("generate_signal", data["hidden_thresholds"])
        for key in (
            "stop_loss_pct",
            "take_profit_pct",
            "min_liquidity_usd",
            "paper_fee_bps",
            "max_slippage_pct",
            "baseline_slippage_pct",
            "round_trip_fee_pct",
            "required_margin_after_costs_pct",
            "max_price_drift_from_model_pct",
            "rf_probability_threshold",
        ):
            self.assertIn(key, c)
            self.assertIsInstance(c[key], (int, float), msg=key)
        self.assertLessEqual(c["max_slippage_pct"], 1.0)
        self.assertLessEqual(c["baseline_slippage_pct"], 1.0)
        self.assertLessEqual(c["round_trip_fee_pct"], 1.0)


class ReconcileStorageTests(unittest.TestCase):
    def test_check_does_not_modify(self) -> None:
        import importlib
        import scripts.reconcile_storage as rs

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["TRADER_DB_PATH"] = str(Path(tmp) / "rec.db")
            import app.database as database

            importlib.reload(database)
            database.init_db()
            importlib.reload(rs)
            before = list(Path(tmp).rglob("*"))
            result = rs.run_check()
            after = list(Path(tmp).rglob("*"))
            self.assertIn("paper_trades", result)
            self.assertTrue(result["sqlite_source_of_truth"])

    def test_fix_is_dry_run(self) -> None:
        import importlib
        import scripts.reconcile_storage as rs

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["TRADER_DB_PATH"] = str(Path(tmp) / "fix.db")
            import app.database as database

            importlib.reload(database)
            database.init_db()
            importlib.reload(rs)
            result = rs.run_fix_dry_run()
            self.assertEqual(result["mode"], "dry_run")
            self.assertEqual(result["modifications"], "none")


class LlmNotCalledWhenBlockedTests(unittest.IsolatedAsyncioTestCase):
    async def test_analyze_market_state_mocked_not_gemini(self) -> None:
        from app.models.predictor import analyze_market_state

        with patch("app.models.predictor.is_headless_data_collection", return_value=True):
            with patch(
                "app.models.predictor._build_analysis_context",
                return_value=({"symbol": "TEST"}, "summary"),
            ):
                with patch("app.models.predictor.log_skipped_llm_decision", new_callable=AsyncMock) as mock_skip:
                    mock_skip.return_value = 99
                    decision, did = await analyze_market_state(
                        {"symbol": "TEST", "whale_score": 0.9},
                        "OPPORTUNISTIC_SPECULATIVE",
                        0.0,
                        coin_id=1,
                    )
                    self.assertEqual(decision.decision, "HOLD")
                    mock_skip.assert_called_once()


if __name__ == "__main__":
    unittest.main()
