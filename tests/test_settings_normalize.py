"""Settings normalization and decimal-fraction unit tests."""
from __future__ import annotations

import unittest

from app.observability.effective_settings import EffectiveSettings
from app.observability.settings_normalize import normalize_canonical_settings, normalize_decimal_fraction_pct
from app.observability.slippage import compute_total_cost_pct, estimate_slippage_per_side_pct


class SettingsNormalizeTests(unittest.TestCase):
    def test_string_ui_values_normalize(self) -> None:
        raw = {
            "stopLossPct": "7",
            "takeProfitPct": "30",
            "minLiquidity": "5000",
            "positionSizePct": "5",
            "tradingFee": "1.5",
            "max_slippage_pct": "1.5",
            "baseline_slippage_pct": "1.5",
            "round_trip_fee_pct": "3",
        }
        eff = EffectiveSettings(raw)
        c = eff.canonical
        self.assertIsInstance(c["stop_loss_pct"], float)
        self.assertIsInstance(c["take_profit_pct"], float)
        self.assertIsInstance(c["min_liquidity_usd"], float)
        self.assertAlmostEqual(c["stop_loss_pct"], 0.07, places=6)
        self.assertAlmostEqual(c["take_profit_pct"], 0.30, places=6)
        self.assertAlmostEqual(c["max_position_size_pct"], 0.05, places=6)
        self.assertAlmostEqual(c["min_liquidity_usd"], 5000.0, places=2)
        self.assertAlmostEqual(c["paper_fee_bps"], 150.0, places=2)

    def test_economic_pct_fields_are_decimal_fractions(self) -> None:
        eff = EffectiveSettings({
            "max_slippage_pct": 1.5,
            "baseline_slippage_pct": 1.5,
            "round_trip_fee_pct": 3.0,
            "max_price_drift_from_model_pct": 0.01,
            "required_margin_after_costs_pct": 0.5,
        })
        c = eff.canonical
        self.assertAlmostEqual(c["max_slippage_pct"], 0.015, places=6)
        self.assertAlmostEqual(c["baseline_slippage_pct"], 0.015, places=6)
        self.assertAlmostEqual(c["round_trip_fee_pct"], 0.03, places=6)
        self.assertAlmostEqual(c["max_price_drift_from_model_pct"], 0.01, places=6)
        self.assertAlmostEqual(c["required_margin_after_costs_pct"], 0.005, places=6)

    def test_probability_fields_not_scaled(self) -> None:
        eff = EffectiveSettings({"rf_probability_threshold": "0.70"})
        self.assertAlmostEqual(eff.canonical["rf_probability_threshold"], 0.70, places=6)

    def test_slippage_receives_decimal_baseline(self) -> None:
        slip, _ = estimate_slippage_per_side_pct(
            position_size_usd=500,
            liquidity_usd=100_000,
            baseline_slippage_pct=0.015,
            dynamic_slippage_enabled=False,
        )
        self.assertAlmostEqual(slip or 0, 0.015, places=6)

    def test_total_cost_realistic_not_100x(self) -> None:
        slip, _ = estimate_slippage_per_side_pct(
            position_size_usd=500,
            liquidity_usd=100_000,
            baseline_slippage_pct=0.015,
            dynamic_slippage_enabled=False,
        )
        total = compute_total_cost_pct(
            round_trip_fee_pct=0.03,
            round_trip_slippage_pct=2 * (slip or 0),
            gas_or_priority_cost_pct=0.001,
        )
        self.assertLess(total, 0.10)
        self.assertGreater(total, 0.05)

    def test_required_margin_legacy_and_decimal(self) -> None:
        eff_legacy = EffectiveSettings({"required_margin_after_costs_pct": 0.5})
        self.assertAlmostEqual(eff_legacy.canonical["required_margin_after_costs_pct"], 0.005, places=6)
        eff_decimal = EffectiveSettings({"required_margin_after_costs_pct": 0.02})
        self.assertAlmostEqual(eff_decimal.canonical["required_margin_after_costs_pct"], 0.02, places=6)


class JsonlDailyPartitionGuardTests(unittest.TestCase):
    def test_daily_filename_has_no_run_timestamp(self) -> None:
        from app.observability.audit_io import JsonlAuditWriter

        w = JsonlAuditWriter("decision_trace", date_slug="2026-06-20")
        self.assertEqual(w.path.name, "decision_trace_2026-06-20.jsonl")
        self.assertNotIn("T", w.path.name)

    def test_same_day_appends_to_one_file(self) -> None:
        import tempfile
        from pathlib import Path

        from app.observability import audit_io

        with tempfile.TemporaryDirectory() as tmp:
            audit_io.AUDITS_DIR = Path(tmp)
            day = "2026-06-20"
            w1 = audit_io.JsonlAuditWriter("pipeline_reasons", date_slug=day)
            w1.append({"n": 1})
            w1.close()
            w2 = audit_io.JsonlAuditWriter("pipeline_reasons", date_slug=day)
            w2.append({"n": 2})
            w2.close()
            self.assertEqual(w1.path, w2.path)
            lines = w1.path.read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 2)
            self.assertNotRegex(w1.path.name, r"\d{8}T\d{6}Z")


if __name__ == "__main__":
    unittest.main()
