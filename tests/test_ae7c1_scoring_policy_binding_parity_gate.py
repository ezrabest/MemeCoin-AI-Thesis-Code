"""Tests for AE7C-1 scoring policy binding, parity harness, and inference gate."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.decision.feature_parity_harness import (  # noqa: E402
    CANONICAL_SYNTHETIC_FIXTURE,
    HarnessParityStatus,
    ParityHarnessMode,
    run_feature_parity_harness,
    run_synthetic_fixture_parity,
)
from app.decision.inference_readiness_gate import (  # noqa: E402
    InferenceGateStatus,
    attempt_local_inference_if_allowed,
    evaluate_inference_readiness_gate,
)
from app.decision.scoring_policy_binding import (  # noqa: E402
    ScoringPolicyBindingStatus,
    bind_scoring_policy,
    generate_scoring_policy_id_from_content,
)


class AE7C1PolicyBindingTests(unittest.TestCase):
    def test_binds_from_explicit_config(self) -> None:
        settings = {
            "tp_ratio": 2.0308,
            "sl_ratio": 0.8,
            "round_trip_fee_pct": 0.0308,
            "time_stop_minutes": 60,
            "horizon": "1h",
            "exit_policy_id": "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
        }
        binding = bind_scoring_policy(settings=settings, signal_row=None)
        self.assertEqual(
            binding.scoring_policy_binding_status,
            ScoringPolicyBindingStatus.PASS_CONFIG_BOUND.value,
        )
        self.assertAlmostEqual(binding.policy_features["tp_ratio"], 2.0308)

    def test_binds_from_signal_context(self) -> None:
        signal = {
            "features_json": {
                "horizon": "4h",
                "exit_policy_id": "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
            }
        }
        binding = bind_scoring_policy(settings={}, signal_row=signal)
        self.assertEqual(
            binding.scoring_policy_binding_status,
            ScoringPolicyBindingStatus.PASS_SIGNAL_CONTEXT_BOUND.value,
        )
        self.assertEqual(binding.horizon, "4h")

    def test_placeholder_marked_honestly(self) -> None:
        binding = bind_scoring_policy(settings={}, signal_row={"features_json": {}})
        self.assertEqual(
            binding.scoring_policy_binding_status,
            ScoringPolicyBindingStatus.PLACEHOLDER_BOUND.value,
        )

    def test_inconsistent_context_blocks(self) -> None:
        settings = {
            "tp_ratio": 2.0308,
            "sl_ratio": 0.8,
            "round_trip_fee_pct": 0.0308,
            "time_stop_minutes": 60,
            "horizon": "1h",
        }
        signal = {
            "features_json": {
                "horizon": "4h",
                "exit_policy_id": "TP20308_SL075_FEE0308_TIME_BY_HORIZON",
                "tp_ratio": 2.0308,
                "sl_ratio": 0.75,
                "round_trip_fee_pct": 0.0308,
                "time_stop_minutes": 240,
            }
        }
        binding = bind_scoring_policy(settings=settings, signal_row=signal)
        self.assertEqual(
            binding.scoring_policy_binding_status,
            ScoringPolicyBindingStatus.BLOCKED_INCONSISTENT_POLICY_CONTEXT.value,
        )

    def test_scoring_policy_id_deterministic(self) -> None:
        features = {
            "tp_ratio": 2.0308,
            "sl_ratio": 0.8,
            "time_stop_minutes": 60,
            "round_trip_fee_pct": 0.0308,
        }
        id1 = generate_scoring_policy_id_from_content(
            exit_policy="P1", horizon="1h", policy_features=features
        )
        id2 = generate_scoring_policy_id_from_content(
            exit_policy="P1", horizon="1h", policy_features=features
        )
        self.assertEqual(id1, id2)

    def test_scoring_policy_id_changes_with_content(self) -> None:
        features_a = {
            "tp_ratio": 2.0308,
            "sl_ratio": 0.8,
            "time_stop_minutes": 60,
            "round_trip_fee_pct": 0.0308,
        }
        features_b = dict(features_a)
        features_b["sl_ratio"] = 0.75
        id_a = generate_scoring_policy_id_from_content(
            exit_policy="P1", horizon="1h", policy_features=features_a
        )
        id_b = generate_scoring_policy_id_from_content(
            exit_policy="P1", horizon="1h", policy_features=features_b
        )
        self.assertNotEqual(id_a, id_b)


class AE7C1ParityHarnessTests(unittest.TestCase):
    def _policy_context(self) -> dict:
        binding = bind_scoring_policy(settings={}, signal_row=None)
        return binding.policy_context

    def test_synthetic_fixture_pass(self) -> None:
        result = run_synthetic_fixture_parity(policy_context=self._policy_context())
        self.assertEqual(
            result.feature_parity_status,
            HarnessParityStatus.PASS_SYNTHETIC_FIXTURE_ONLY.value,
        )

    def test_synthetic_fixture_fail_mismatch(self) -> None:
        fixture = dict(CANONICAL_SYNTHETIC_FIXTURE)
        fixture["snapshot_row"] = dict(fixture["snapshot_row"])
        fixture["snapshot_row"]["liquidity"] = 0.0
        result = run_synthetic_fixture_parity(
            policy_context=self._policy_context(),
            raw_fixture=fixture,
        )
        self.assertIn(
            result.feature_parity_status,
            {
                HarnessParityStatus.FAIL_MISMATCH.value,
                HarnessParityStatus.PASS_SYNTHETIC_FIXTURE_ONLY.value,
            },
        )

    def test_exact_parity_requires_exact_ids(self) -> None:
        result = run_feature_parity_harness(
            mode=ParityHarnessMode.EXACT_ONLY.value,
            runtime_bridge_records=[{"candidate_id": "a", "feature_values": {"price_usd": 1.0}}],
            offline_rows_by_exact_id={"b": {"price_usd": 1.0}},
        )
        self.assertEqual(
            result.feature_parity_status,
            HarnessParityStatus.BLOCKED_NO_OVERLAP.value,
        )

    def test_synthetic_does_not_fully_unblock_inference(self) -> None:
        result = run_synthetic_fixture_parity(policy_context=self._policy_context())
        self.assertEqual(
            result.future_inference_readiness,
            "BLOCKED_PENDING_EXACT_OR_APPROVED_PARITY_SET",
        )


class AE7C1InferenceGateTests(unittest.TestCase):
    def _pass_gate_inputs(self) -> dict:
        return dict(
            schema_compatibility_status="PASS",
            feature_parity_status=HarnessParityStatus.PASS_EXACT_OVERLAP.value,
            scoring_policy_binding_status=ScoringPolicyBindingStatus.PASS_CONFIG_BOUND.value,
            missing_required_features_count=0,
            forbidden_feature_check="PASS",
            lineage_confidence_score=1.0,
            exact_id_match=True,
            model_artifacts_reproducible=True,
            target_row_id_required=False,
            external_calls_required=False,
        )

    def test_blocks_placeholder_policy(self) -> None:
        gate = evaluate_inference_readiness_gate(
            **{
                **self._pass_gate_inputs(),
                "scoring_policy_binding_status": ScoringPolicyBindingStatus.PLACEHOLDER_BOUND.value,
            }
        )
        self.assertFalse(gate.inference_allowed)
        self.assertEqual(
            gate.inference_gate_status,
            InferenceGateStatus.BLOCKED_POLICY_PLACEHOLDER.value,
        )

    def test_blocks_no_overlap_parity(self) -> None:
        gate = evaluate_inference_readiness_gate(
            schema_compatibility_status="PASS",
            feature_parity_status=HarnessParityStatus.BLOCKED_NO_OVERLAP.value,
            scoring_policy_binding_status=ScoringPolicyBindingStatus.PASS_CONFIG_BOUND.value,
            missing_required_features_count=0,
            forbidden_feature_check="PASS",
            lineage_confidence_score=1.0,
            exact_id_match=True,
            model_artifacts_reproducible=True,
        )
        self.assertFalse(gate.inference_allowed)
        self.assertEqual(gate.inference_gate_status, InferenceGateStatus.BLOCKED_PARITY.value)

    def test_smoke_defaults_no_inference(self) -> None:
        gate = evaluate_inference_readiness_gate(
            schema_compatibility_status="PASS",
            feature_parity_status=HarnessParityStatus.BLOCKED_NO_OVERLAP.value,
            scoring_policy_binding_status=ScoringPolicyBindingStatus.PLACEHOLDER_BOUND.value,
            missing_required_features_count=0,
            forbidden_feature_check="PASS",
            lineage_confidence_score=0.35,
            exact_id_match=False,
            model_artifacts_reproducible=True,
        )
        attempt = attempt_local_inference_if_allowed(
            gate_result=gate,
            allow_local_inference_if_gates_pass=False,
        )
        self.assertFalse(attempt["inference_executed"])

    def test_blocks_missing_required_features(self) -> None:
        gate = evaluate_inference_readiness_gate(
            **{**self._pass_gate_inputs(), "missing_required_features_count": 2}
        )
        self.assertFalse(gate.inference_allowed)

    def test_blocks_target_row_id_required(self) -> None:
        gate = evaluate_inference_readiness_gate(
            **{**self._pass_gate_inputs(), "target_row_id_required": True}
        )
        self.assertFalse(gate.inference_allowed)

    def test_blocks_unreproducible_artifacts(self) -> None:
        gate = evaluate_inference_readiness_gate(
            **{**self._pass_gate_inputs(), "model_artifacts_reproducible": False}
        )
        self.assertFalse(gate.inference_allowed)

    def test_pass_when_all_gates_pass(self) -> None:
        gate = evaluate_inference_readiness_gate(**self._pass_gate_inputs())
        self.assertTrue(gate.inference_allowed)

    def test_allow_flag_still_refuses_when_gates_fail(self) -> None:
        gate = evaluate_inference_readiness_gate(
            schema_compatibility_status="PASS",
            feature_parity_status=HarnessParityStatus.BLOCKED_NO_OVERLAP.value,
            scoring_policy_binding_status=ScoringPolicyBindingStatus.PLACEHOLDER_BOUND.value,
            missing_required_features_count=0,
            forbidden_feature_check="PASS",
            lineage_confidence_score=0.35,
            exact_id_match=False,
            model_artifacts_reproducible=True,
        )
        attempt = attempt_local_inference_if_allowed(
            gate_result=gate,
            allow_local_inference_if_gates_pass=True,
        )
        self.assertFalse(attempt["inference_executed"])

    def test_no_predict_in_inference_attempt(self) -> None:
        gate = evaluate_inference_readiness_gate(**self._pass_gate_inputs())
        attempt = attempt_local_inference_if_allowed(
            gate_result=gate,
            allow_local_inference_if_gates_pass=True,
        )
        self.assertFalse(attempt["inference_executed"])
        self.assertIn("AE7C1", attempt.get("reason", ""))


if __name__ == "__main__":
    unittest.main()
