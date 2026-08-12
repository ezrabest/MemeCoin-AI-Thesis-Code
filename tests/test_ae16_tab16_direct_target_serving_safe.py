"""Tests for AE16 TAB16 direct-target serving-safe artifact and registry."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

from app.consensus.ae16_model_registry import (  # noqa: E402
    Ae16RegistryError,
    FEATURE_ORDER_MISMATCH,
    FEATURE_SCHEMA_HASH_MISMATCH,
    MISSING_DEPENDENCY_XGBOOST,
    REGISTRY_BYPASS_CODE,
    audit_production_paths_for_registry_bypass,
    build_ordered_inference_matrix,
    feature_set_hash_sha256,
    ordered_feature_schema_hash_sha256,
    verify_feature_schema_hashes,
)
from app.consensus.ae16_tab16_direct_target import (  # noqa: E402
    ARTIFACT_REL,
    FORBIDDEN_ALIAS_REL,
    LEGACY_TAB_ARTIFACTS,
    ORDERED_FEATURE_NAMES,
    assign_consensus_preview_tier,
    audit_forbidden_features,
    build_feature_schema_lock,
    build_tab16_artifact_dict,
    compute_schema_hashes,
    is_forbidden_feature,
    reject_legacy_tab_as_tab16,
    train_tab16_classifier,
)


class TestTab16Schema(unittest.TestCase):
    def test_exactly_26_features(self):
        self.assertEqual(len(ORDERED_FEATURE_NAMES), 26)
        lock = build_feature_schema_lock()
        self.assertEqual(lock["feature_count"], 26)
        self.assertEqual(lock["ordered_feature_names"], ORDERED_FEATURE_NAMES)

    def test_feature_hashes_stable(self):
        h = compute_schema_hashes()
        self.assertEqual(h["feature_set_hash_sha256"], feature_set_hash_sha256(ORDERED_FEATURE_NAMES))
        self.assertEqual(
            h["ordered_feature_schema_hash_sha256"],
            ordered_feature_schema_hash_sha256(ORDERED_FEATURE_NAMES),
        )
        # set hash differs from ordered hash (unless already sorted identically — still may equal)
        shuffled = list(reversed(ORDERED_FEATURE_NAMES))
        self.assertNotEqual(
            ordered_feature_schema_hash_sha256(ORDERED_FEATURE_NAMES),
            ordered_feature_schema_hash_sha256(shuffled),
        )
        self.assertEqual(
            feature_set_hash_sha256(ORDERED_FEATURE_NAMES),
            feature_set_hash_sha256(shuffled),
        )

    def test_feature_set_hash_mismatch_fails(self):
        with self.assertRaises(Ae16RegistryError) as ctx:
            verify_feature_schema_hashes(
                ordered_feature_names=ORDERED_FEATURE_NAMES,
                expected_feature_set_hash="deadbeef",
                expected_ordered_hash=None,
            )
        self.assertEqual(ctx.exception.code, FEATURE_SCHEMA_HASH_MISMATCH)

    def test_ordered_feature_schema_hash_mismatch_fails(self):
        with self.assertRaises(Ae16RegistryError) as ctx:
            verify_feature_schema_hashes(
                ordered_feature_names=ORDERED_FEATURE_NAMES,
                expected_feature_set_hash=None,
                expected_ordered_hash="deadbeef",
            )
        self.assertEqual(ctx.exception.code, FEATURE_SCHEMA_HASH_MISMATCH)

    def test_shuffled_inference_reorders(self):
        lock = build_feature_schema_lock()
        rng = np.random.default_rng(0)
        data = {n: rng.normal(size=5) for n in ORDERED_FEATURE_NAMES}
        df = pd.DataFrame(data)
        shuffled_cols = list(reversed(ORDERED_FEATURE_NAMES))
        df_shuf = df[shuffled_cols]
        entry = {
            "ordered_feature_names": ORDERED_FEATURE_NAMES,
            "feature_set_hash_sha256": lock["feature_set_hash_sha256"],
            "ordered_feature_schema_hash_sha256": lock["ordered_feature_schema_hash_sha256"],
        }
        X = build_ordered_inference_matrix(df_shuf, entry, {"feature_schema_lock": lock})
        self.assertEqual(list(X.columns), ORDERED_FEATURE_NAMES)
        np.testing.assert_allclose(X.to_numpy(), df[ORDERED_FEATURE_NAMES].to_numpy())

    def test_missing_feature_fails_order_mismatch(self):
        lock = build_feature_schema_lock()
        df = pd.DataFrame({n: [1.0] for n in ORDERED_FEATURE_NAMES[:-1]})
        entry = {
            "ordered_feature_names": ORDERED_FEATURE_NAMES,
            "feature_set_hash_sha256": lock["feature_set_hash_sha256"],
            "ordered_feature_schema_hash_sha256": lock["ordered_feature_schema_hash_sha256"],
        }
        with self.assertRaises(Ae16RegistryError) as ctx:
            build_ordered_inference_matrix(df, entry, {"feature_schema_lock": lock})
        self.assertEqual(ctx.exception.code, FEATURE_ORDER_MISMATCH)

    def test_forbidden_features_rejected(self):
        self.assertTrue(is_forbidden_feature("whale_score"))
        self.assertTrue(is_forbidden_feature("llm_confidence"))
        self.assertTrue(is_forbidden_feature("price_return_1h"))
        self.assertTrue(is_forbidden_feature("verified_exit"))
        audit = audit_forbidden_features(["tp_ratio", "whale_score", "sentiment_score"])
        self.assertFalse(audit["passed"])
        self.assertIn("whale_score", audit["forbidden_features_found"])

    def test_legacy_tab_paths_rejected(self):
        for p in LEGACY_TAB_ARTIFACTS:
            with self.assertRaises(Ae16RegistryError):
                reject_legacy_tab_as_tab16(p)
        with self.assertRaises(Ae16RegistryError):
            reject_legacy_tab_as_tab16("models/ae16f_tab_serving_safe.joblib")

    def test_no_alias_constant(self):
        self.assertEqual(FORBIDDEN_ALIAS_REL, "models/ae16f_tab_serving_safe.joblib")
        self.assertEqual(ARTIFACT_REL, "models/ae16_tab16_direct_target_serving_safe.joblib")
        self.assertNotEqual(ARTIFACT_REL, FORBIDDEN_ALIAS_REL)

    def test_all_negative_attached_maps_to_reject(self):
        tier = assign_consensus_preview_tier(
            has_l1=True,
            rf_status="MODEL_EVIDENCE_ATTACHED",
            rf_vote=False,
            xgb_status="MODEL_EVIDENCE_ATTACHED",
            xgb_vote=False,
            tab16_status="MODEL_EVIDENCE_ATTACHED",
            tab16_vote=False,
        )
        self.assertEqual(tier, "REJECT")

    def test_partial_when_tab_missing(self):
        tier = assign_consensus_preview_tier(
            has_l1=True,
            rf_status="MODEL_EVIDENCE_ATTACHED",
            rf_vote=False,
            xgb_status="MODEL_EVIDENCE_ATTACHED",
            xgb_vote=True,
            tab16_status="TAB_DIRECT_TARGET_SERVING_ARTIFACT_MISSING",
            tab16_vote=False,
        )
        self.assertEqual(tier, "PARTIAL_MODEL_EVIDENCE")

    def test_artifact_schema_fields(self):
        # Tiny fit
        X = pd.DataFrame(
            np.random.default_rng(1).normal(size=(40, 26)),
            columns=ORDERED_FEATURE_NAMES,
        )
        y = np.array([0, 1] * 20)
        mask = np.ones(40, dtype=bool)
        model = train_tab16_classifier(X, y, mask)
        lock = build_feature_schema_lock()
        art = build_tab16_artifact_dict(
            model=model,
            lock=lock,
            threshold=0.9,
            training_rows=40,
            validation_rows=10,
            training_source="data/training/manual_verified_datasets_direct_target_v1/LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL075_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.parquet",
            lookahead_passed=True,
            validation_metrics_payload={"profitability_claimed": False},
        )
        self.assertEqual(art["model_variant"], "TAB16_DIRECT_TARGET_SERVING_SAFE")
        self.assertEqual(art["consensus_slot"], "TAB")
        self.assertTrue(art["allowed_to_feed_ae16_consensus_tab_slot"])
        self.assertFalse(art["allowed_to_replace_legacy_tab"])
        self.assertFalse(art["legacy_tab_used"])
        self.assertIsNone(art["legacy_tab_artifact_path"])
        self.assertEqual(art["feature_schema_lock"]["feature_count"], 26)
        self.assertTrue(hasattr(art["model"], "predict_proba"))
        # Not the same as a RandomForest dump
        est = art["model"].named_steps["model"]
        self.assertEqual(type(est).__name__, "HistGradientBoostingClassifier")

    def test_registry_bypass_audit_runs(self):
        result = audit_production_paths_for_registry_bypass(ROOT)
        self.assertIn("status", result)
        self.assertIn("bypass_detected", result)

    def test_xgb_missing_dependency_code(self):
        self.assertEqual(MISSING_DEPENDENCY_XGBOOST, "MISSING_DEPENDENCY_XGBOOST")
        self.assertEqual(REGISTRY_BYPASS_CODE, "AE16_MODEL_REGISTRY_BYPASS")


class TestRegistryLoadIntegration(unittest.TestCase):
    def test_rf_dict_unwrap_through_registry_if_present(self):
        rf_path = ROOT / "models" / "ae16f_rf_serving_safe.joblib"
        reg_path = ROOT / "models" / "ae16_model_registry.json"
        if not rf_path.is_file() or not reg_path.is_file():
            self.skipTest("RF artifact or registry not present yet")
        from app.consensus.ae16_model_registry import load_ae16_registered_model

        loaded = load_ae16_registered_model("RF", ROOT)
        self.assertTrue(hasattr(loaded["model"], "predict_proba"))
        self.assertTrue(isinstance(loaded["artifact_raw"], dict))
        self.assertIn("model", loaded["artifact_raw"])

    def test_tab16_scores_serving_matrix_if_present(self):
        art = ROOT / "models" / "ae16_tab16_direct_target_serving_safe.joblib"
        reg = ROOT / "models" / "ae16_model_registry.json"
        matrix = ROOT / (
            "data/audits/manual_post_collection_rf_xgb_tab_sanity_20260724T193531Z/"
            "data/serving_feature_matrix_preview.csv"
        )
        if not art.is_file() or not reg.is_file() or not matrix.is_file():
            self.skipTest("TAB16 artifact/registry/matrix not present yet")
        from app.consensus.ae16_model_registry import load_ae16_registered_model, build_ordered_inference_matrix

        loaded = load_ae16_registered_model("TAB", ROOT)
        df = pd.read_csv(matrix)
        X = build_ordered_inference_matrix(
            df,
            loaded["registry_entry"],
            {"feature_schema_lock": loaded["feature_schema_lock"], **loaded["artifact_metadata"]},
        )
        scores = loaded["model"].predict_proba(X)[:, 1]
        self.assertEqual(len(scores), len(df))

    def test_no_current_selected_in_training_source_path(self):
        from app.consensus.ae16_tab16_direct_target import TRAINING_SOURCE_REL

        self.assertIn("manual_verified_datasets_direct_target_v1", TRAINING_SOURCE_REL)
        self.assertNotIn("selected", TRAINING_SOURCE_REL.lower())
        self.assertNotIn("runtime_selected", TRAINING_SOURCE_REL)


if __name__ == "__main__":
    unittest.main()
