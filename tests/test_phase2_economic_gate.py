"""Phase 2 economic gate, actionability, runtime inference, and storage fix tests."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from app.observability.audit_reasons import AuditReason
from app.observability.candidate import TradeCandidate
from app.observability.effective_settings import EffectiveSettings
from app.observability.economic_gate import (
    DecisionResult,
    evaluate_economic_trade_candidate,
    evaluate_tab_confidence_boost,
)
from app.observability.llm_gate import evaluate_llm_short_circuit_phase2
from app.observability.model_runtime_inference import (
    RuntimeInferenceResult,
    reset_runtime_model_inference_for_tests,
)
from app.observability.slippage import (
    check_price_drift,
    check_slippage_limit,
    compute_total_cost_pct,
    estimate_slippage_per_side_pct,
)
from app.observability.whale_wave_features import compute_rolling_whale_wave_features


def _base_settings(**overrides) -> dict:
    s = {
        "economic_gate_enabled": True,
        "paper_trading_enabled": True,
        "demo_aggressive_enabled": False,
        "live_trading_enabled": False,
        "trading_mode": "DEMO",
        "min_liquidity_usd": 5000,
        "min_whale_score": 0.30,
        "min_signal_score": 0.55,
        "min_buy_ratio": 0.50,
        "rf_probability_threshold": 0.70,
        "rf_gate_enabled": True,
        "allow_model_unavailable_fallback": False,
        "max_slippage_pct": 0.05,
        "baseline_slippage_pct": 0.015,
        "baseline_slippage_is_per_side": True,
        "dynamic_slippage_enabled": True,
        "round_trip_fee_pct": 0.03,
        "max_price_drift_from_model_pct": 0.01,
        "max_market_snapshot_age_seconds": 999999,
        "max_model_prediction_age_seconds": 999999,
        "max_model_artifact_age_hours": 999999,
        "required_margin_after_costs_pct": 0.02,
        "max_position_size_pct": 0.05,
        "starting_capital": 10000,
        "tab_confidence_boost_enabled": False,
        "tab_confidence_boost_enabled_demo": False,
    }
    s.update(overrides)
    return s


def _candidate(**kwargs) -> TradeCandidate:
    base = dict(
        pair_address="0xtest",
        chain="bsc",
        symbol="TEST/WBNB",
        price=1.0,
        liquidity_usd=100_000,
        whale_score=0.8,
        signal_score=0.7,
        signal_type="WATCH",
        coin_id=1,
        buy_ratio=0.6,
        volume_24h=50_000,
        current_execution_price=1.0,
        model_snapshot_price=1.0,
        event_timestamp=datetime.now(timezone.utc).isoformat(),
    )
    base.update(kwargs)
    return TradeCandidate(**base)


def _mock_inference_ok(prob: float = 0.85) -> RuntimeInferenceResult:
    now = datetime.now(timezone.utc).isoformat()
    return RuntimeInferenceResult(
        status="ok",
        predicted_probability=prob,
        prediction_generated_at=now,
        model_snapshot_price=1.0,
        audit_reasons=[AuditReason.MODEL_RUNTIME_INFERENCE_OK.value],
        runtime_metadata={
            "model_path": "/data/training/models/label_profitable_after_fees_4h_best.joblib",
            "model_name": "random_forest",
            "is_full_pipeline": True,
            "target_name": "label_profitable_after_fees_4h",
            "horizon": "4h",
        },
        rf_prediction={
            "predicted_probability": prob,
            "event_timestamp": now,
            "prediction_source": "runtime_inference",
            "prediction_generated_at": now,
        },
    )


def _patch_runtime_inference(result: RuntimeInferenceResult):
    mock_runtime = MagicMock()
    mock_runtime.predict_for_candidate.return_value = result
    mock_runtime.artifact_age_seconds.return_value = 3600.0
    return patch(
        "app.observability.economic_gate.get_runtime_model_inference",
        return_value=mock_runtime,
    )


class Phase1PrerequisiteTests(unittest.TestCase):
    def test_phase1_artifacts_exist(self) -> None:
        root = Path(__file__).resolve().parents[1]
        required = [
            "app/observability/effective_settings.py",
            "app/observability/audit_reasons.py",
            "app/observability/llm_gate.py",
            "app/observability/audit_io.py",
            "app/observability/model_runtime_inference.py",
            "scripts/reconcile_storage.py",
            "scripts/dry_run_economic_gate_recent.py",
        ]
        for rel in required:
            self.assertTrue((root / rel).exists(), rel)
        eff = EffectiveSettings()
        self.assertIn("economic_gate", eff.hidden_thresholds)
        self.assertEqual(eff.canonical.get("max_market_snapshot_age_seconds"), 300)


class SlippageTests(unittest.TestCase):
    def test_per_side_baseline(self) -> None:
        slip, _ = estimate_slippage_per_side_pct(
            position_size_usd=500,
            liquidity_usd=100_000,
            baseline_slippage_pct=0.015,
            baseline_slippage_is_per_side=True,
            dynamic_slippage_enabled=False,
        )
        self.assertAlmostEqual(slip or 0, 0.015, places=4)

    def test_amm_nonlinear_increase(self) -> None:
        small, _ = estimate_slippage_per_side_pct(
            position_size_usd=100, liquidity_usd=100_000, dynamic_slippage_enabled=True,
        )
        large, _ = estimate_slippage_per_side_pct(
            position_size_usd=5000, liquidity_usd=100_000, dynamic_slippage_enabled=True,
        )
        self.assertGreater(large or 0, small or 0)
        mid, _ = estimate_slippage_per_side_pct(
            position_size_usd=1000, liquidity_usd=100_000, dynamic_slippage_enabled=True,
        )
        self.assertGreater(mid or 0, small or 0)
        self.assertLess(mid or 0, large or 0)

    def test_slippage_rejects_not_caps(self) -> None:
        slip = 0.025
        ok, reasons = check_slippage_limit(slip, max_slippage_pct=0.015)
        self.assertFalse(ok)
        self.assertIn(AuditReason.BLOCKED_BY_SLIPPAGE_LIMIT.value, reasons)
        self.assertEqual(slip, 0.025)

    def test_missing_liquidity_fails_closed(self) -> None:
        slip, reasons = estimate_slippage_per_side_pct(
            position_size_usd=500, liquidity_usd=0,
        )
        self.assertIsNone(slip)
        self.assertIn(AuditReason.BLOCKED_BY_MISSING_SLIPPAGE_INPUTS.value, reasons)

    def test_total_cost_decimal_fraction(self) -> None:
        total = compute_total_cost_pct(round_trip_fee_pct=0.03, round_trip_slippage_pct=0.03, gas_or_priority_cost_pct=0.001)
        self.assertAlmostEqual(total, 0.061, places=3)

    def test_slippage_above_max_rejects(self) -> None:
        slip, _ = estimate_slippage_per_side_pct(
            position_size_usd=50_000, liquidity_usd=10_000, dynamic_slippage_enabled=True,
        )
        ok, reasons = check_slippage_limit(slip, max_slippage_pct=0.015)
        self.assertFalse(ok)
        self.assertIn(AuditReason.BLOCKED_BY_SLIPPAGE_LIMIT.value, reasons)


class PriceDriftTests(unittest.TestCase):
    def test_rejects_high_drift(self) -> None:
        ok, drift, reasons = check_price_drift(
            model_snapshot_price=1.0,
            current_execution_price=1.02,
            max_price_drift_from_model_pct=0.01,
        )
        self.assertFalse(ok)
        self.assertIn(AuditReason.BLOCKED_BY_PRICE_DRIFT.value, reasons)
        self.assertGreater(drift or 0, 0.01)

    def test_missing_snapshot_price(self) -> None:
        ok, _, reasons = check_price_drift(
            model_snapshot_price=None,
            current_execution_price=1.0,
            max_price_drift_from_model_pct=0.01,
        )
        self.assertFalse(ok)
        self.assertIn(AuditReason.MISSING_MODEL_SNAPSHOT_PRICE.value, reasons)

    def test_missing_execution_price(self) -> None:
        ok, _, reasons = check_price_drift(
            model_snapshot_price=1.0,
            current_execution_price=None,
            max_price_drift_from_model_pct=0.01,
        )
        self.assertFalse(ok)
        self.assertIn(AuditReason.PRICE_FILL_RESOLUTION_FAILED.value, reasons)


class ExpectedReturnUnitTests(unittest.TestCase):
    def test_margin_pass_decimal_units(self) -> None:
        total_cost = 0.03
        expected_return = 0.08
        margin = 0.02
        self.assertTrue(expected_return - total_cost > margin)

    def test_margin_fail_decimal_units(self) -> None:
        total_cost = 0.03
        expected_return = 0.04
        margin = 0.02
        self.assertFalse(expected_return - total_cost > margin)

    def test_rf_probability_not_expected_return(self) -> None:
        with _patch_runtime_inference(_mock_inference_ok(prob=0.85)):
            with patch("app.observability.economic_gate._load_expected_return_calibration", return_value=None):
                c = _candidate(liquidity_usd=1_000_000)
                result = evaluate_economic_trade_candidate(c, _base_settings(max_slippage_pct=0.05))
        self.assertEqual(result.action, "PAPER_BUY_CANDIDATE")
        self.assertIsNone(result.expected_return_pct)
        self.assertIn(AuditReason.EXPECTED_RETURN_CALIBRATION_UNAVAILABLE.value, result.reasons)


class EconomicGateTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_runtime_model_inference_for_tests()

    def test_gate_disabled_by_default(self) -> None:
        eff = EffectiveSettings()
        self.assertFalse(eff.canonical.get("economic_gate_enabled", False))

    def test_fail_closed_no_runtime_inference(self) -> None:
        c = _candidate()
        unavailable = RuntimeInferenceResult(
            status="not_available",
            audit_reasons=[AuditReason.MODEL_RUNTIME_INFERENCE_NOT_AVAILABLE.value],
            runtime_metadata={"model_path": None},
        )
        with _patch_runtime_inference(unavailable):
            result = evaluate_economic_trade_candidate(c, _base_settings())
        self.assertEqual(result.action, "HOLD")
        self.assertIn(AuditReason.MODEL_RUNTIME_INFERENCE_NOT_AVAILABLE.value, result.reasons)

    def test_fail_closed_low_rf_probability(self) -> None:
        c = _candidate()
        with _patch_runtime_inference(_mock_inference_ok(prob=0.50)):
            result = evaluate_economic_trade_candidate(c, _base_settings())
        self.assertEqual(result.action, "HOLD")
        self.assertIn(AuditReason.PROBABILITY_BELOW_THRESHOLD.value, result.reasons)

    def test_bearish_veto_blocks(self) -> None:
        c = _candidate(bearish_alert_active=True, alert_type="LARGE_SELL")
        result = evaluate_economic_trade_candidate(c, _base_settings())
        self.assertEqual(result.action, "BLOCKED")
        self.assertIn(AuditReason.BLOCKED_BY_BEARISH_ALERT.value, result.reasons)

    def test_strong_signal_no_alert_rf_approved(self) -> None:
        c = _candidate(alert_type=None, liquidity_usd=1_000_000)
        with _patch_runtime_inference(_mock_inference_ok()):
            with patch("app.observability.economic_gate._load_expected_return_calibration", return_value=None):
                result = evaluate_economic_trade_candidate(c, _base_settings(max_slippage_pct=0.05))
        self.assertEqual(result.action, "PAPER_BUY_CANDIDATE")
        self.assertIn(AuditReason.MISSING_BULLISH_ALERT.value, result.reasons)
        self.assertIn(AuditReason.PAPER_BUY_CANDIDATE_CREATED.value, result.reasons)

    def test_missing_price_blocks(self) -> None:
        c = _candidate(price=0, coin_id=1)
        result = evaluate_economic_trade_candidate(c, _base_settings())
        self.assertEqual(result.action, "HOLD")
        self.assertIn(AuditReason.MISSING_PRICE_OR_PAIR.value, result.reasons)

    def test_missing_coin_id_blocks(self) -> None:
        c = _candidate(coin_id=None)
        result = evaluate_economic_trade_candidate(c, _base_settings())
        self.assertEqual(result.action, "HOLD")
        self.assertIn(AuditReason.MISSING_PRICE_OR_PAIR.value, result.reasons)

    def test_economic_gate_disabled_old_behavior(self) -> None:
        c = _candidate()
        result = evaluate_economic_trade_candidate(c, _base_settings(economic_gate_enabled=False))
        self.assertEqual(result.action, "WATCH")
        self.assertIn(AuditReason.SETTINGS_BLOCKED.value, result.reasons)

    def test_stale_market_snapshot_rejects(self) -> None:
        c = _candidate(event_timestamp="2020-01-01T00:00:00+00:00")
        with _patch_runtime_inference(_mock_inference_ok()):
            result = evaluate_economic_trade_candidate(
                c, _base_settings(max_market_snapshot_age_seconds=300),
            )
        self.assertEqual(result.action, "HOLD")
        self.assertIn(AuditReason.MARKET_SNAPSHOT_TOO_OLD.value, result.reasons)


class TabBoostTests(unittest.TestCase):
    def test_default_disabled(self) -> None:
        eff = EffectiveSettings()
        self.assertFalse(eff.canonical.get("tab_confidence_boost_enabled", True))

    def test_boost_applies_multiplier(self) -> None:
        c = _candidate(tab_prediction={
            "tab_score": 0.99, "tab_suffix": "nearest_neighbors_context_4096",
            "meets_percentile": True, "percentile_threshold": 0.90,
        })
        settings = _base_settings(tab_confidence_boost_enabled_demo=True, tab_confidence_boost_enabled=True)
        mult, reasons = evaluate_tab_confidence_boost(c, settings, rf_probability=0.75, trading_mode="DEMO")
        self.assertGreater(mult, 1.0)
        self.assertIn(AuditReason.TAB_CONFIDENCE_BOOST_APPLIED.value, reasons)


class LlmPhase2GateTests(unittest.TestCase):
    def test_not_called_before_economic_approval(self) -> None:
        c = _candidate()
        gate = DecisionResult(action="HOLD", reasons=[AuditReason.BLOCKED_BY_ECONOMIC_MODEL.value])
        should, _ = evaluate_llm_short_circuit_phase2(
            economic_approved=False, candidate=c, gate_result=gate, settings=_base_settings(), alert=None,
        )
        self.assertFalse(should)

    def test_not_called_for_runtime_inference_unavailable(self) -> None:
        c = _candidate()
        gate = DecisionResult(
            action="HOLD",
            reasons=[AuditReason.MODEL_RUNTIME_INFERENCE_NOT_AVAILABLE.value],
        )
        should, _ = evaluate_llm_short_circuit_phase2(
            economic_approved=True, candidate=c, gate_result=gate, settings=_base_settings(), alert=None,
        )
        self.assertFalse(should)

    def test_not_called_for_slippage_block(self) -> None:
        c = _candidate()
        gate = DecisionResult(action="HOLD", reasons=[AuditReason.BLOCKED_BY_SLIPPAGE_LIMIT.value])
        should, _ = evaluate_llm_short_circuit_phase2(
            economic_approved=True, candidate=c, gate_result=gate, settings=_base_settings(), alert={"alert_type": "LARGE_BUY"},
        )
        self.assertFalse(should)


class AuditIoDailyPartitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._audit_dir = Path(self._tmpdir.name) / "audits"
        self._audit_dir.mkdir()

    def tearDown(self) -> None:
        from app.observability.audit_io import reset_audit_writers_for_tests
        reset_audit_writers_for_tests()
        self._tmpdir.cleanup()

    def test_same_day_same_file(self) -> None:
        from app.observability import audit_io

        audit_io.AUDITS_DIR = self._audit_dir
        audit_io.reset_audit_writers_for_tests()
        day = "2026-06-20"
        w1 = audit_io.JsonlAuditWriter("pipeline_reasons", date_slug=day)
        w1.append({"n": 1})
        w1.close()
        w2 = audit_io.JsonlAuditWriter("pipeline_reasons", date_slug=day)
        w2.append({"n": 2})
        w2.close()
        self.assertEqual(w1.path, w2.path)
        self.assertIn("pipeline_reasons_2026-06-20.jsonl", w1.path.name)
        lines = w1.path.read_text(encoding="utf-8").strip().split("\n")
        self.assertEqual(len(lines), 2)

    def test_static_json_remains_timestamped(self) -> None:
        from app.observability import audit_io

        audit_io.AUDITS_DIR = self._audit_dir
        path = audit_io.write_json_report_atomic("settings_effective_20260620T120000Z.json", {"ok": True})
        self.assertTrue(path.name.startswith("settings_effective_"))
        self.assertTrue(path.suffix == ".json")


class ReconcileStoragePhase2Tests(unittest.TestCase):
    def test_fix_without_yes_no_modify(self) -> None:
        import importlib
        import scripts.reconcile_storage as rs

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["TRADER_DB_PATH"] = str(Path(tmp) / "fix.db")
            import app.database as database
            importlib.reload(database)
            database.init_db()
            importlib.reload(rs)
            before = list(Path(tmp).rglob("*"))
            rs.run_fix_dry_run()
            after = list(Path(tmp).rglob("*"))
            self.assertEqual(len(before), len(after))

    def test_fix_yes_default_sqlite_to_csv_not_import(self) -> None:
        import importlib
        import scripts.reconcile_storage as rs

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["TRADER_DB_PATH"] = str(Path(tmp) / "apply.db")
            import app.database as database
            importlib.reload(database)
            database.init_db()
            importlib.reload(rs)
            result = rs.run_fix_apply(import_csv=False)
            self.assertEqual(result["mode"], "apply")
            self.assertIn("sqlite_to_csv", result["repairs"])
            self.assertEqual(result["repairs"]["csv_to_sqlite"]["imported"], 0)
            self.assertFalse(result["import_csv_mode"])

    def test_csv_import_requires_explicit_flag(self) -> None:
        import importlib
        import scripts.reconcile_storage as rs

        with tempfile.TemporaryDirectory() as tmp:
            os.environ["TRADER_DB_PATH"] = str(Path(tmp) / "import.db")
            import app.database as database
            importlib.reload(database)
            database.init_db()
            importlib.reload(rs)
            plan = rs.run_fix_dry_run(import_csv=True)
            self.assertTrue(plan["import_csv_mode"])
            os.environ.pop("TRADER_DB_PATH", None)


class RuntimeInferenceLeakageTests(unittest.TestCase):
    def test_outcome_columns_excluded_from_inference_matrix(self) -> None:
        from app.observability.model_runtime_inference import _is_outcome_column

        self.assertTrue(_is_outcome_column("target_return_4h"))
        self.assertTrue(_is_outcome_column("future_return_4h"))
        self.assertTrue(_is_outcome_column("label_profitable_after_fees_4h"))
        self.assertFalse(_is_outcome_column("whale_score"))
        self.assertFalse(_is_outcome_column("buy_ratio"))

    def test_baseline_metrics_preserves_outcome_columns(self) -> None:
        metrics_path = Path(__file__).resolve().parents[1] / "data" / "training" / "models" / "baseline_metrics.json"
        if not metrics_path.is_file():
            self.skipTest("baseline_metrics.json not present")
        with open(metrics_path, encoding="utf-8") as f:
            metrics = json.load(f)
        excluded = metrics.get("excluded_features") or []
        self.assertTrue(any("target_return" in str(c) or "label_" in str(c) for c in excluded))


class ActionabilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_mocked_not_called_when_disabled(self) -> None:
        from app.observability.actionability import evaluate_and_execute_candidate

        c = _candidate(liquidity_usd=1_000_000)
        settings = _base_settings(llm_enabled_for_demo=False, economic_gate_enabled=True, max_slippage_pct=0.05)
        pair = {
            "pairAddress": "0xtest",
            "chainId": "bsc",
            "baseToken": {"symbol": "TEST"},
            "priceUsd": "1.0",
            "liquidity": {"usd": 100000},
            "volume": {"h24": 50000},
            "txns": {"h24": {"buys": 60, "sells": 40}},
            "priceChange": {"h24": 5, "h1": 2},
        }
        with _patch_runtime_inference(_mock_inference_ok()):
            with patch("app.observability.actionability.get_paper_trader") as mock_trader:
                mock_trader.return_value.try_autonomous_buy.return_value = {"ok": True, "position": {"id": 1}}
                with patch("app.models.predictor.analyze_market_state", new_callable=AsyncMock) as mock_llm:
                    result = await evaluate_and_execute_candidate(c, pair=pair, settings=settings)
                    mock_llm.assert_not_called()
        self.assertIn(result.get("action"), ("PAPER_BUY_EXECUTED", "BLOCKED", "HOLD"))


if __name__ == "__main__":
    unittest.main()
