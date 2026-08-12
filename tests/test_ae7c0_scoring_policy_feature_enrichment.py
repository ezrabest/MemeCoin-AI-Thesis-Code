"""Tests for AE7C-0 scoring policy feature enrichment."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.decision.ae7c0_feature_enrichment import summarize_compatibility_delta  # noqa: E402
from app.decision.feature_parity import FeatureParityStatus, run_feature_parity_check  # noqa: E402
from app.decision.feature_schema import (  # noqa: E402
    build_enriched_runtime_feature_schema,
    build_feature_values,
    build_runtime_feature_schema,
    is_forbidden_feature_name,
)
from app.decision.runtime_feature_bridge import build_model_schema_compatibility_matrix  # noqa: E402
from app.decision.scoring_policy_features import resolve_scoring_policy_context  # noqa: E402


class AE7C0ForbiddenRulesTests(unittest.TestCase):
    def test_tp_ratio_allowed(self) -> None:
        self.assertFalse(is_forbidden_feature_name("tp_ratio"))

    def test_sl_ratio_allowed(self) -> None:
        self.assertFalse(is_forbidden_feature_name("sl_ratio"))

    def test_time_stop_minutes_allowed(self) -> None:
        self.assertFalse(is_forbidden_feature_name("time_stop_minutes"))

    def test_round_trip_fee_pct_allowed(self) -> None:
        self.assertFalse(is_forbidden_feature_name("round_trip_fee_pct"))

    def test_outcome_returns_rejected(self) -> None:
        self.assertTrue(is_forbidden_feature_name("realized_return"))
        self.assertTrue(is_forbidden_feature_name("future_return"))
        self.assertTrue(is_forbidden_feature_name("net_return"))


class AE7C0DerivedFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.schema = build_enriched_runtime_feature_schema()
        self.policy = resolve_scoring_policy_context(settings_path=Path("/nonexistent"))

    def test_volume_to_liquidity_ratio_computes(self) -> None:
        result = build_feature_values(
            snapshot_row={"price": 1.0, "liquidity": 1000.0, "volume_24h": 5000.0},
            signal_row=None,
            sentiment_agg=None,
            schema=self.schema,
            policy_context=self.policy,
        )
        self.assertAlmostEqual(result.feature_values["volume_to_liquidity_ratio"], 5.0)

    def test_volume_to_liquidity_ratio_null_when_missing(self) -> None:
        result = build_feature_values(
            snapshot_row={"price": 1.0, "liquidity": 1000.0},
            signal_row=None,
            sentiment_agg=None,
            schema=self.schema,
            policy_context=self.policy,
        )
        self.assertIsNone(result.feature_values.get("volume_to_liquidity_ratio"))
        self.assertIn("volume_to_liquidity_ratio", result.feature_missingness)
        self.assertEqual(
            result.feature_missingness_reasons.get("volume_to_liquidity_ratio"),
            "MISSING_SOURCE_FEATURE",
        )

    def test_volume_to_liquidity_ratio_null_when_liquidity_zero(self) -> None:
        result = build_feature_values(
            snapshot_row={"price": 1.0, "liquidity": 0.0, "volume_24h": 100.0},
            signal_row=None,
            sentiment_agg=None,
            schema=self.schema,
            policy_context=self.policy,
        )
        self.assertIsNone(result.feature_values.get("volume_to_liquidity_ratio"))
        self.assertEqual(
            result.feature_missingness_reasons.get("volume_to_liquidity_ratio"),
            "INVALID_LIQUIDITY_USD_LE_ZERO",
        )

    def test_txns_h24_total_from_buys_sells(self) -> None:
        result = build_feature_values(
            snapshot_row={
                "price": 1.0,
                "liquidity": 1000.0,
                "txns_buys": 10,
                "txns_sells": 5,
            },
            signal_row=None,
            sentiment_agg=None,
            schema=self.schema,
            policy_context=self.policy,
        )
        self.assertEqual(result.feature_values["txns_h24_total"], 15.0)

    def test_buy_sell_ratio_handles_zero_sells(self) -> None:
        result = build_feature_values(
            snapshot_row={
                "price": 1.0,
                "liquidity": 1000.0,
                "txns_buys": 10,
                "txns_sells": 0,
            },
            signal_row=None,
            sentiment_agg=None,
            schema=self.schema,
            policy_context=self.policy,
        )
        self.assertIsNone(result.feature_values.get("buy_sell_ratio_h24"))
        self.assertEqual(
            result.feature_missingness_reasons.get("buy_sell_ratio_h24"),
            "INVALID_TXNS_H24_SELLS_LE_ZERO",
        )

    def test_missing_required_not_filled_with_zero(self) -> None:
        result = build_feature_values(
            snapshot_row=None,
            signal_row=None,
            sentiment_agg=None,
            schema=self.schema,
            policy_context=self.policy,
        )
        self.assertIsNone(result.feature_values.get("price_usd"))
        self.assertNotEqual(result.feature_values.get("price_usd"), 0)


class AE7C0SchemaTests(unittest.TestCase):
    def test_feature_schema_id_changes_with_enrichment(self) -> None:
        before = build_runtime_feature_schema(enriched=False)
        after = build_enriched_runtime_feature_schema()
        self.assertNotEqual(before.feature_schema_id, after.feature_schema_id)
        self.assertGreater(len(after.feature_names), len(before.feature_names))

    def test_policy_features_populated_from_placeholder(self) -> None:
        schema = build_enriched_runtime_feature_schema()
        policy = resolve_scoring_policy_context(settings_path=Path("/nonexistent"))
        result = build_feature_values(
            snapshot_row={"price": 1.0, "liquidity": 1000.0},
            signal_row=None,
            sentiment_agg=None,
            schema=schema,
            policy_context=policy,
        )
        self.assertIsNotNone(result.feature_values.get("tp_ratio"))
        self.assertIsNotNone(result.policy_feature_metadata)
        meta = result.policy_feature_metadata["tp_ratio"]
        self.assertTrue(meta["not_label"])
        self.assertTrue(meta["not_future_outcome"])
        self.assertFalse(meta["used_for_inference"])


class AE7C0ParityAndCompatibilityTests(unittest.TestCase):
    def test_parity_blocked_no_overlap(self) -> None:
        result = run_feature_parity_check(
            runtime_bridge_records=[{"candidate_id": "only-runtime", "feature_values": {}}],
            offline_rows_by_exact_id=None,
        )
        self.assertEqual(result.feature_parity_status, FeatureParityStatus.BLOCKED_NO_OVERLAP.value)

    def test_safe_for_future_inference_false_when_parity_blocked(self) -> None:
        delta = summarize_compatibility_delta(
            before_rows=[{"compatibility_status": "PARTIAL_MISSING_FEATURES", "missing_features_sample": "tp_ratio"}],
            after_rows=[{"compatibility_status": "COMPATIBLE", "missing_features_sample": "", "safe_for_future_inference": False}],
            parity_status=FeatureParityStatus.BLOCKED_NO_OVERLAP.value,
        )
        self.assertFalse(delta["safe_for_future_inference"])

    def test_compatibility_matrix_safe_false_with_weak_lineage(self) -> None:
        schema = build_enriched_runtime_feature_schema()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "models" / "test_clean_rf_schema.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                '{"feature_columns":["tp_ratio","price_usd","liquidity_usd"]}',
                encoding="utf-8",
            )
            with mock.patch(
                "app.decision.runtime_feature_bridge.load_model_schema_from_json",
                return_value=(["tp_ratio", "price_usd", "liquidity_usd"], "model_schema"),
            ):
                rows = build_model_schema_compatibility_matrix(
                    runtime_schema=schema,
                    schema_candidate_paths=[Path("models/test_clean_rf_schema.json")],
                    project_root=root,
                    parity_status=FeatureParityStatus.BLOCKED_NO_OVERLAP.value,
                    weak_lineage=True,
                )
        self.assertFalse(rows[0]["safe_for_future_inference"])


class AE7C0SafetyTests(unittest.TestCase):
    def test_target_row_id_not_required_in_enriched_schema(self) -> None:
        schema = build_enriched_runtime_feature_schema()
        self.assertNotIn("target_row_id", schema.feature_names)


if __name__ == "__main__":
    unittest.main()
