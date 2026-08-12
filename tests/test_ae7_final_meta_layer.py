"""Tests for AE7 FINAL offline meta-layer."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.decision.meta_layer_audits import run_meta_audits  # noqa: E402
from app.decision.meta_layer_dataset import (  # noqa: E402
    build_meta_dataset,
    is_forbidden_meta_feature,
)
from app.decision.meta_layer_decision import (  # noqa: E402
    MetaLayerFinalStatus,
    decide_meta_layer,
)
from app.decision.meta_layer_models import (  # noqa: E402
    evaluate_rule_baseline,
    run_robustness_audits,
    train_calibrated_logistic,
    train_logistic_baseline,
    train_xgb_meta_model,
)
from app.decision.meta_layer_policy import (  # noqa: E402
    PolicyConfigError,
    canonical_policy_json,
    load_scoring_policy_config_strict,
    policy_content_hash,
    scoring_policy_id_from_policy_content,
)


def _valid_policy() -> dict:
    return {
        "tp_ratio": 2.0308,
        "sl_ratio": 0.8,
        "round_trip_fee_pct": 0.0308,
        "time_stop_minutes": 60,
        "horizon": "1h",
        "exit_policy": "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
        "policy_version": "AE7_FINAL_TEST",
        "allow_trading": False,
        "allow_paper_trading": False,
        "allow_model_inference": False,
    }


def _synthetic_meta_frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    pairs = [f"pair_{i % 10}" for i in range(n)]
    split = ["train"] * (n // 2) + ["validation"] * (n // 4) + ["test"] * (n - n // 2 - n // 4)
    return pd.DataFrame(
        {
            "candidate_id": [f"c{i}" for i in range(n)],
            "pair_address": pairs,
            "split": split,
            "rf_score": rng.random(n),
            "xgb_score": rng.random(n),
            "tab_score": rng.random(n),
            "vote_count": rng.integers(0, 4, n),
            "consensus_tier": rng.choice(["ALL3", "TAB_XGB", "RF_ONLY"], n),
            "tp_ratio": 2.03,
            "sl_ratio": 0.8,
            "round_trip_fee_pct": 0.0308,
            "time_stop_minutes": 60,
            "horizon": "1h",
            "filter_name": "direct",
            "exit_policy": "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
            "meta_target_y": rng.integers(0, 2, n),
        }
    )


class MetaDatasetTests(unittest.TestCase):
    def test_excludes_label_future_outcome_from_x(self) -> None:
        self.assertTrue(is_forbidden_meta_feature("target_net_profitable"))
        self.assertTrue(is_forbidden_meta_feature("sim_net_return"))
        self.assertFalse(is_forbidden_meta_feature("tp_ratio"))

    def test_pair_address_excluded_from_predictive_features(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "consensus.csv"
            frame = _synthetic_meta_frame(50)
            frame["target_net_profitable"] = frame["meta_target_y"]
            frame.drop(columns=["meta_target_y"]).to_csv(path, index=False)
            result = build_meta_dataset(project_root=Path(tmp), consensus_artifact=path, max_rows=50)
            self.assertNotIn("pair_address", result.feature_columns)

    def test_vote_count_and_consensus_included(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "consensus.csv"
            frame = _synthetic_meta_frame(50)
            frame["target_net_profitable"] = frame["meta_target_y"]
            frame.drop(columns=["meta_target_y"]).to_csv(path, index=False)
            result = build_meta_dataset(project_root=Path(tmp), consensus_artifact=path, max_rows=50)
            self.assertIn("vote_count", result.feature_columns)
            self.assertIn("consensus_tier", result.feature_columns)

    def test_missing_context_recorded_not_invented(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "consensus.csv"
            frame = _synthetic_meta_frame(20)
            frame["target_net_profitable"] = frame["meta_target_y"]
            frame.drop(columns=["meta_target_y"]).to_csv(path, index=False)
            result = build_meta_dataset(project_root=Path(tmp), consensus_artifact=path, max_rows=20)
            self.assertIn("context_family", result.signal_families_missing)
            self.assertNotIn("rss_sentiment_score", result.feature_columns)


class PolicyConfigTests(unittest.TestCase):
    def test_missing_config_fails_hard(self) -> None:
        with self.assertRaises(PolicyConfigError) as ctx:
            load_scoring_policy_config_strict(Path("/nonexistent/policy.json"))
        self.assertEqual(ctx.exception.status, "POLICY_CONFIG_MISSING")

    def test_malformed_json_fails_hard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.json"
            bad.write_text("{not json", encoding="utf-8")
            with self.assertRaises(PolicyConfigError) as ctx:
                load_scoring_policy_config_strict(bad)
            self.assertEqual(ctx.exception.status, "POLICY_CONFIG_INVALID")

    def test_invalid_config_no_silent_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "invalid.json"
            bad.write_text(json.dumps({"horizon": "1h"}), encoding="utf-8")
            with self.assertRaises(PolicyConfigError) as ctx:
                load_scoring_policy_config_strict(bad)
            self.assertEqual(ctx.exception.status, "POLICY_CONFIG_VALIDATION_FAILED")

    def test_policy_hash_stable_on_key_order(self) -> None:
        a = {"tp_ratio": 2.0, "sl_ratio": 0.8, "round_trip_fee_pct": 0.03, "time_stop_minutes": 60, "horizon": "1h"}
        b = {"horizon": "1h", "time_stop_minutes": 60, "round_trip_fee_pct": 0.03, "sl_ratio": 0.8, "tp_ratio": 2.0}
        self.assertEqual(canonical_policy_json(a), canonical_policy_json(b))
        self.assertEqual(policy_content_hash(a), policy_content_hash(b))

    def test_policy_hash_changes_on_material_field(self) -> None:
        base = _valid_policy()
        changed = dict(base)
        changed["tp_ratio"] = 2.5
        self.assertNotEqual(policy_content_hash(base), policy_content_hash(changed))
        self.assertNotEqual(
            scoring_policy_id_from_policy_content(base),
            scoring_policy_id_from_policy_content(changed),
        )

    def test_meta_rows_include_policy_audit_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "consensus.csv"
            frame = _synthetic_meta_frame(20)
            frame["target_net_profitable"] = frame["meta_target_y"]
            frame.drop(columns=["meta_target_y"]).to_csv(path, index=False)
            policy = load_scoring_policy_config_strict(None)
            cfg_path = Path(tmp) / "policy.json"
            cfg_path.write_text(json.dumps(_valid_policy()), encoding="utf-8")
            policy = load_scoring_policy_config_strict(cfg_path)
            result = build_meta_dataset(
                project_root=Path(tmp),
                consensus_artifact=path,
                policy_audit=policy,
                max_rows=20,
            )
            self.assertIn("scoring_policy_id", result.frame.columns)
            self.assertIn("policy_content_hash", result.frame.columns)


class MetaModelTests(unittest.TestCase):
    def test_rule_comparator_without_training(self) -> None:
        frame = _synthetic_meta_frame(120)
        result = evaluate_rule_baseline(frame, "meta_target_y")
        self.assertEqual(result.status, "PASS")

    def test_logistic_baseline_on_synthetic(self) -> None:
        frame = _synthetic_meta_frame(200)
        features = ["rf_score", "xgb_score", "tab_score", "vote_count", "tp_ratio"]
        result = train_logistic_baseline(frame, features, "meta_target_y")
        self.assertEqual(result.status, "PASS")

    def test_calibrated_blocks_insufficient_calibration_data(self) -> None:
        frame = _synthetic_meta_frame(80)
        frame["split"] = "test"
        frame.loc[:10, "split"] = "validation"
        result = train_calibrated_logistic(frame, ["rf_score", "xgb_score"], "meta_target_y")
        self.assertEqual(result.status, "BLOCKED_INSUFFICIENT_CALIBRATION_DATA")

    def test_xgb_meta_blocks_when_dependency_missing(self) -> None:
        frame = _synthetic_meta_frame(120)
        import app.decision.meta_layer_models as mlm

        real_import = __import__

        def fake_import(name, *args, **kwargs):
            if name == "xgboost":
                raise ImportError("xgboost not installed")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            result = mlm.train_xgb_meta_model(
                frame, ["rf_score", "xgb_score"], "meta_target_y", include_xgb_meta=True
            )
        self.assertEqual(result.status, "XGB_META_BLOCKED_DEPENDENCY_MISSING")


class RobustnessAndDecisionTests(unittest.TestCase):
    def test_top_pair_removed_audit(self) -> None:
        frame = _synthetic_meta_frame(200)
        rule = evaluate_rule_baseline(frame, "meta_target_y")
        robustness = run_robustness_audits(frame, "meta_target_y", rule_result=rule)
        self.assertIn("top_pair_share", robustness)
        self.assertIn("leave_one_pair_out_precision_sample", robustness)

    def test_decision_blocks_on_leakage(self) -> None:
        decision = decide_meta_layer(
            audits={"leakage_status": "FAIL", "target_availability_status": "PASS"},
            rule_result={"status": "PASS", "metrics": {"auc": 0.6}},
            logistic_result={"status": "PASS", "metrics": {"auc": 0.7}},
            calibrated_result={"status": "PASS", "metrics": {}},
            xgb_result={"status": "PASS", "metrics": {}},
            robustness={"robustness_pass_flag": True},
            ablation_findings={},
            policy_audit={"policy_config_status": "NOT_PROVIDED_ARTIFACT_EMBEDDED"},
            dataset_summary={"rows": 200},
        )
        self.assertEqual(decision["final_status"], MetaLayerFinalStatus.BLOCKED_LEAKAGE_RISK.value)

    def test_decision_blocks_on_robustness_failure(self) -> None:
        decision = decide_meta_layer(
            audits={"leakage_status": "PASS", "target_availability_status": "PASS"},
            rule_result={"status": "PASS", "metrics": {"auc": 0.55}},
            logistic_result={"status": "PASS", "metrics": {"auc": 0.9}},
            calibrated_result={"status": "PASS", "metrics": {}},
            xgb_result={"status": "PASS", "metrics": {}},
            robustness={"robustness_pass_flag": False, "top_pair_share": 0.6},
            ablation_findings={},
            policy_audit={"policy_config_status": "NOT_PROVIDED_ARTIFACT_EMBEDDED"},
            dataset_summary={"rows": 200},
        )
        self.assertEqual(decision["final_status"], MetaLayerFinalStatus.NOT_ROBUST_ENOUGH.value)

    def test_decision_blocks_invalid_policy_config(self) -> None:
        decision = decide_meta_layer(
            audits={"leakage_status": "PASS", "target_availability_status": "PASS"},
            rule_result={"status": "PASS", "metrics": {}},
            logistic_result={"status": "PASS", "metrics": {}},
            calibrated_result={"status": "PASS", "metrics": {}},
            xgb_result={"status": "PASS", "metrics": {}},
            robustness={"robustness_pass_flag": True},
            ablation_findings={},
            policy_audit={"policy_config_status": "POLICY_CONFIG_INVALID"},
            dataset_summary={"rows": 200},
        )
        self.assertEqual(decision["final_status"], MetaLayerFinalStatus.BLOCKED_POLICY_CONFIG.value)

    def test_runtime_trading_disallowed(self) -> None:
        decision = decide_meta_layer(
            audits={"leakage_status": "PASS", "target_availability_status": "PASS"},
            rule_result={"status": "PASS", "metrics": {"auc": 0.6}},
            logistic_result={"status": "PASS", "metrics": {"auc": 0.62}},
            calibrated_result={"status": "PASS", "metrics": {}},
            xgb_result={"status": "PASS", "metrics": {}},
            robustness={"robustness_pass_flag": True},
            ablation_findings={},
            policy_audit={"policy_config_status": "NOT_PROVIDED_ARTIFACT_EMBEDDED"},
            dataset_summary={"rows": 200},
        )
        self.assertEqual(decision["runtime_inference_status"], "BLOCKED_PENDING_RUNTIME_PARITY_AND_LINEAGE")
        self.assertEqual(decision["trading_authorization_status"], "NOT_APPROVED")
        self.assertTrue(decision["explicit_no_runtime_trading_approval"])

    def test_leakage_audit_excludes_pair_address(self) -> None:
        frame = _synthetic_meta_frame(50)
        audits = run_meta_audits(
            frame=frame,
            feature_columns=["rf_score", "vote_count"],
            target_column="meta_target_y",
            policy_audit={"policy_config_status": "NOT_PROVIDED_ARTIFACT_EMBEDDED"},
        )
        self.assertEqual(audits.leakage_status, "PASS")
        self.assertFalse(audits.pair_address_predictive_use)


class SafetyTests(unittest.TestCase):
    def test_no_external_api_calls_in_run_script(self) -> None:
        source = (ROOT / "scripts" / "run_ae7_final_meta_layer.py").read_text(encoding="utf-8")
        for token in ("requests.", "httpx.", "openai.", "ollama", "helius", "gemini", "qwen"):
            self.assertNotIn(token, source.lower())

    def test_no_base_model_retraining_hooks(self) -> None:
        source = (ROOT / "app" / "decision" / "meta_layer_models.py").read_text(encoding="utf-8")
        self.assertNotIn("fit_base", source)
        self.assertNotIn("retrain", source.lower())


if __name__ == "__main__":
    unittest.main()
