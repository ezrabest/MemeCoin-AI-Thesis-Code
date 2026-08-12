"""Tests for Phase E4A direct-target XGB/RF training infrastructure."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.training.direct_target_xgb_rf import (  # noqa: E402
    CANONICAL_TARGET,
    IncrementalAuditLogger,
    TargetNormalizationAudit,
    apply_deterministic_row_limit,
    assert_row_count_invariants,
    atomic_write_json,
    build_feature_columns,
    build_rf_classifier,
    build_training_pipeline,
    build_xgb_rf_agreement_diagnostic,
    discover_direct_target_datasets,
    extract_imputer_metadata,
    extract_positive_probability,
    filter_descriptors,
    normalize_target_column,
    prepare_output_dirs,
    register_e4_artifacts,
    select_top_with_pair_cap,
    DatasetDescriptor,
    derive_valid_label_mask,
    rank_validation_policies,
    validate_split_column,
)


def _synthetic_dataset(
  n_train: int = 80,
  n_val: int = 30,
  n_test: int = 30,
  *,
  target_col: str = "target_net_profitable_after_exit",
  include_ambiguous_target: bool = False,
) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    for split, n in (("train", n_train), ("validation", n_val), ("test", n_test)):
        for i in range(n):
            label = int(rng.random() > 0.7)
            rows.append(
                {
                    "candidate_id": f"cand_{split}_{i}",
                    "candidate_policy_id": f"cp_{split}_{i}",
                    "target_row_id": f"tr_{split}_{i}",
                    "pair_address": f"pair_{i % 10}",
                    "event_timestamp": f"2026-06-01T12:{i % 60:02d}:00Z",
                    "filter": "LIQ_5K_HIGH_ACTIVITY",
                    "horizon": "1h",
                    "exit_policy_id": "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
                    "split": split,
                    "label_valid": True,
                    target_col: label,
                    "sim_net_return": 0.05 if label else -0.02,
                    "feature_a": float(rng.normal()),
                    "feature_b": float(rng.normal()),
                    "feature_c": rng.random() > 0.5,
                    "target_name": "net_profitable_after_exit_policy",
                    "target_version": "v1",
                }
            )
    df = pd.DataFrame(rows)
    if include_ambiguous_target:
        df["target_net_profitable_after_exit_policy"] = df[target_col]
    return df


class DirectTargetDiscoveryTests(unittest.TestCase):
    def test_discover_direct_target_datasets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            name = "LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL080_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.csv"
            _synthetic_dataset().to_csv(root / name, index=False)
            found = discover_direct_target_datasets(root)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].filter_name, "LIQ_5K_HIGH_ACTIVITY")
            self.assertEqual(found[0].horizon, "1h")

    def test_prefers_parquet_over_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stem = "LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL080_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1"
            _synthetic_dataset().to_csv(root / f"{stem}.csv", index=False)
            _synthetic_dataset(n_train=1, n_val=1, n_test=1).to_parquet(root / f"{stem}.parquet", index=False)
            found = discover_direct_target_datasets(root)
            self.assertEqual(found[0].dataset_path.suffix, ".parquet")


class TargetNormalizationTests(unittest.TestCase):
    def test_normalize_to_canonical_target(self) -> None:
        df = _synthetic_dataset()
        desc = DatasetDescriptor(
            dataset_name="ds",
            dataset_path=Path("x.csv"),
            filter_name="LIQ_5K_HIGH_ACTIVITY",
            horizon="1h",
            exit_policy_id="TP20308_SL080_FEE0308_TIME_BY_HORIZON",
            target_name="net_profitable_after_exit_policy",
            target_version="v1",
        )
        out, audit = normalize_target_column(df, desc)
        self.assertEqual(audit["normalization_status"], "ok")
        self.assertEqual(audit["target_column_canonical"], CANONICAL_TARGET)
        self.assertIn(CANONICAL_TARGET, out.columns)

    def test_normalize_boolean_strings(self) -> None:
        df = _synthetic_dataset()
        df["target_net_profitable_after_exit"] = df["target_net_profitable_after_exit"].map(
            {0: "False", 1: "True"}
        )
        desc = DatasetDescriptor(
            dataset_name="ds",
            dataset_path=Path("x.csv"),
            filter_name="LIQ_5K_HIGH_ACTIVITY",
            horizon="1h",
            exit_policy_id="TP20308_SL080_FEE0308_TIME_BY_HORIZON",
            target_name="net_profitable_after_exit_policy",
            target_version="v1",
        )
        out, audit = normalize_target_column(df, desc)
        self.assertEqual(audit["normalization_status"], "ok")
        self.assertSetEqual(set(out[CANONICAL_TARGET].dropna().unique()), {0.0, 1.0})

    def test_ambiguous_target_alias(self) -> None:
        df = _synthetic_dataset(include_ambiguous_target=True)
        desc = DatasetDescriptor(
            dataset_name="ds",
            dataset_path=Path("x.csv"),
            filter_name="LIQ_5K_HIGH_ACTIVITY",
            horizon="1h",
            exit_policy_id="TP20308_SL080_FEE0308_TIME_BY_HORIZON",
            target_name="net_profitable_after_exit_policy",
            target_version="v1",
        )
        _, audit = normalize_target_column(df, desc)
        self.assertEqual(audit["normalization_status"], "AMBIGUOUS_TARGET_ALIAS")

    def test_target_normalization_audit_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "phase_e4_target_normalization_audit.csv"
            audit_writer = TargetNormalizationAudit(path)
            audit_writer.append_row(
                {
                    "dataset_name": "ds",
                    "dataset_path": "x.csv",
                    "filter": "F",
                    "horizon": "1h",
                    "exit_policy_id": "P",
                    "target_column_original": "target_net_profitable_after_exit",
                    "target_column_canonical": CANONICAL_TARGET,
                    "normalization_status": "ok",
                }
            )
            text = path.read_text(encoding="utf-8")
            self.assertIn("target_column_canonical", text)
            self.assertIn(CANONICAL_TARGET, text)


class ValidLabelAndSplitTests(unittest.TestCase):
    def test_valid_label_filtering(self) -> None:
        df = _synthetic_dataset()
        df.loc[0, "label_valid"] = False
        df, _ = normalize_target_column(
            df,
            DatasetDescriptor("d", Path("x"), "F", "1h", "P", "t", "v1"),
        )
        mask = derive_valid_label_mask(df)
        self.assertEqual(int(mask.sum()), len(df) - 1)

    def test_identity_columns_preserved(self) -> None:
        df = _synthetic_dataset()
        for col in ("candidate_id", "candidate_policy_id", "target_row_id"):
            self.assertIn(col, df.columns)

    def test_split_preservation(self) -> None:
        df = _synthetic_dataset()
        limited = apply_deterministic_row_limit(df, 50)
        ok, err = validate_split_column(limited)
        self.assertTrue(ok)
        self.assertIsNone(err)
        self.assertTrue(set(limited["split"]).issubset({"train", "validation", "test"}))


class RowCountInvariantTests(unittest.TestCase):
    def test_row_count_invariant_pass(self) -> None:
        counts = {
            "post_valid_filter_row_count": 140,
            "train_row_count": 80,
            "validation_row_count": 30,
            "test_row_count": 30,
            "feature_matrix_train_row_count": 80,
            "feature_matrix_validation_row_count": 30,
            "feature_matrix_test_row_count": 30,
            "prediction_validation_row_count": 30,
            "prediction_test_row_count": 30,
        }
        assert_row_count_invariants(counts)

    def test_row_count_invariant_fail(self) -> None:
        counts = {
            "post_valid_filter_row_count": 140,
            "train_row_count": 80,
            "validation_row_count": 30,
            "test_row_count": 30,
            "feature_matrix_train_row_count": 79,
            "feature_matrix_validation_row_count": 30,
            "feature_matrix_test_row_count": 30,
            "prediction_validation_row_count": 30,
            "prediction_test_row_count": 30,
        }
        with self.assertRaises(RuntimeError) as ctx:
            assert_row_count_invariants(counts)
        self.assertIn("ROW_COUNT_INVARIANT_FAILED", str(ctx.exception))


class FeatureMatrixTests(unittest.TestCase):
    def test_leakage_exclusion(self) -> None:
        df = _synthetic_dataset()
        df, _ = normalize_target_column(
            df,
            DatasetDescriptor("d", Path("x"), "F", "1h", "P", "t", "v1"),
        )
        features, excluded_leakage, excluded_identity, _ = build_feature_columns(df)
        self.assertNotIn(CANONICAL_TARGET, features)
        self.assertNotIn("sim_net_return", features)
        self.assertIn("candidate_id", excluded_identity)

    def test_schema_alignment_missing_column(self) -> None:
        df = _synthetic_dataset()
        df, _ = normalize_target_column(
            df,
            DatasetDescriptor("d", Path("x"), "F", "1h", "P", "t", "v1"),
        )
        features, _, _, _ = build_feature_columns(df)
        val = df[df["split"] == "validation"].drop(columns=["feature_a"])
        with self.assertRaises(RuntimeError) as ctx:
            from app.training.direct_target_xgb_rf import build_feature_matrix

            build_feature_matrix(val, features, split_name="validation")
        self.assertIn("MODEL_SCHEMA_MISMATCH", str(ctx.exception))


class PipelineArtifactTests(unittest.TestCase):
    def test_pipeline_not_raw_estimator(self) -> None:
        df = _synthetic_dataset()
        df, _ = normalize_target_column(
            df,
            DatasetDescriptor("d", Path("x"), "F", "1h", "P", "t", "v1"),
        )
        features, _, _, _ = build_feature_columns(df)
        train = df[df["split"] == "train"]
        x_train = train[features]
        y_train = train[CANONICAL_TARGET].astype(int).to_numpy()
        pipeline = build_training_pipeline("RF", build_rf_classifier(random_state=42))
        pipeline.fit(x_train, y_train)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump(pipeline, path)
            loaded = joblib.load(path)
            self.assertIsInstance(loaded, Pipeline)
            self.assertIn("imputer", loaded.named_steps)
            self.assertIn("model", loaded.named_steps)

    def test_imputer_statistics_by_feature(self) -> None:
        df = _synthetic_dataset()
        df.loc[df.index[0], "feature_a"] = np.nan
        df, _ = normalize_target_column(
            df,
            DatasetDescriptor("d", Path("x"), "F", "1h", "P", "t", "v1"),
        )
        features, _, _, _ = build_feature_columns(df)
        train = df[df["split"] == "train"]
        x_train = train[features]
        y_train = train[CANONICAL_TARGET].astype(int).to_numpy()
        pipeline = build_training_pipeline("RF", build_rf_classifier(random_state=42))
        pipeline.fit(x_train, y_train)
        stats, missing, medians = extract_imputer_metadata(pipeline, features, x_train)
        self.assertEqual(set(stats.keys()), set(features))
        self.assertIn("feature_a", medians)
        self.assertGreater(missing["feature_a"], 0)

    def test_preprocessing_sidecar_sklearn_version(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sidecar.json"
            atomic_write_json({"sklearn_version": sklearn.__version__}, path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["sklearn_version"], sklearn.__version__)

    def test_no_validation_imputer_refit(self) -> None:
        df = _synthetic_dataset()
        df, _ = normalize_target_column(
            df,
            DatasetDescriptor("d", Path("x"), "F", "1h", "P", "t", "v1"),
        )
        features, _, _, _ = build_feature_columns(df)
        train = df[df["split"] == "train"]
        val = df[df["split"] == "validation"]
        x_train = train[features]
        x_val = val[features].copy()
        x_val.iloc[0, 0] = np.nan
        y_train = train[CANONICAL_TARGET].astype(int).to_numpy()
        pipeline = build_training_pipeline("RF", build_rf_classifier(random_state=42))
        pipeline.fit(x_train, y_train)
        train_stats = pipeline.named_steps["imputer"].statistics_.copy()
        pipeline.predict_proba(x_val)
        self.assertTrue(np.allclose(pipeline.named_steps["imputer"].statistics_, train_stats))

    def test_positive_probability_uses_classes(self) -> None:
        df = _synthetic_dataset()
        df, _ = normalize_target_column(
            df,
            DatasetDescriptor("d", Path("x"), "F", "1h", "P", "t", "v1"),
        )
        features, _, _, _ = build_feature_columns(df)
        train = df[df["split"] == "train"]
        x_train = train[features]
        y_train = train[CANONICAL_TARGET].astype(int).to_numpy()
        pipeline = build_training_pipeline("RF", build_rf_classifier(random_state=42))
        pipeline.fit(x_train, y_train)
        proba = extract_positive_probability(pipeline, x_train)
        self.assertEqual(len(proba), len(x_train))


class PolicyEvaluationTests(unittest.TestCase):
    def test_pair_cap_selection(self) -> None:
        df = pd.DataFrame(
            {
                "pair_address": ["a", "a", "b", "c"],
                "predicted_probability": [0.9, 0.8, 0.7, 0.6],
            }
        )
        selected = select_top_with_pair_cap(df, score_col="predicted_probability", k=3, pair_cap=1)
        self.assertEqual(len(selected), 3)
        self.assertEqual(selected["pair_address"].nunique(), 3)

    def test_validation_policy_ranking(self) -> None:
        grid = pd.DataFrame(
            [
                {
                    "model": "RF",
                    "filter": "F",
                    "horizon": "1h",
                    "exit_policy_id": "P",
                    "split": "validation",
                    "top_pct": 1.0,
                    "pair_cap": 5,
                    "selected_count": 60,
                    "total_net_return": 1.5,
                    "avg_net_return": 0.02,
                    "target_precision": 0.4,
                    "unique_pairs": 10,
                    "top_pair_share": 0.1,
                },
                {
                    "model": "RF",
                    "filter": "F",
                    "horizon": "1h",
                    "exit_policy_id": "P",
                    "split": "validation",
                    "top_pct": 2.0,
                    "pair_cap": "none",
                    "selected_count": 100,
                    "total_net_return": 2.0,
                    "avg_net_return": 0.03,
                    "target_precision": 0.5,
                    "unique_pairs": 20,
                    "top_pair_share": 0.1,
                },
            ]
        )
        ranked = rank_validation_policies(grid)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked.iloc[0]["pair_cap"], 5)

    def test_xgb_rf_agreement_diagnostic(self) -> None:
        base = _synthetic_dataset(n_train=0, n_val=20, n_test=0)
        base, _ = normalize_target_column(
            base,
            DatasetDescriptor("d", Path("x"), "F", "1h", "P", "t", "v1"),
        )
        xgb = base.copy()
        rf = base.copy()
        xgb["predicted_probability"] = np.linspace(0.9, 0.1, len(xgb))
        rf["predicted_probability"] = np.linspace(0.8, 0.2, len(rf))
        rows = build_xgb_rf_agreement_diagnostic(
            xgb,
            rf,
            filter_name="F",
            horizon="1h",
            exit_policy_id="P",
            split_name="validation",
            return_col="sim_net_return",
        )
        self.assertTrue(rows)
        slices = {r["agreement_slice"] for r in rows}
        self.assertIn("XGB_AND_RF", slices)


class AuditAndAtomicTests(unittest.TestCase):
    def test_incremental_jsonl_audit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            logger = IncrementalAuditLogger(path)
            logger.log("run_started", status="started")
            logger.log("dataset_loaded", status="ok")
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            for line in lines:
                json.loads(line)

    def test_audit_failure_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            logger = IncrementalAuditLogger(path)
            logger.log("dataset_started", status="started")
            try:
                raise RuntimeError("ROW_COUNT_INVARIANT_FAILED: test")
            except RuntimeError as exc:
                logger.log("row_count_invariant_failed", status="failed", error_message=str(exc))
            lines = path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            self.assertIn("ROW_COUNT_INVARIANT_FAILED", lines[1])

    def test_atomic_json_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.json"
            atomic_write_json({"ok": True}, path)
            self.assertTrue(path.exists())
            self.assertFalse(list(Path(tmp).glob(".*.tmp.*")))


class SmokeBehaviorTests(unittest.TestCase):
    def test_deterministic_row_limit(self) -> None:
        df = _synthetic_dataset(n_train=100, n_val=40, n_test=40)
        a = apply_deterministic_row_limit(df, 50)
        b = apply_deterministic_row_limit(df, 50)
        pd.testing.assert_frame_equal(a, b)

    def test_random_state_in_rf(self) -> None:
        rf1 = build_rf_classifier(random_state=7)
        rf2 = build_rf_classifier(random_state=7)
        self.assertEqual(rf1.get_params()["random_state"], rf2.get_params()["random_state"])


class RegistrationTests(unittest.TestCase):
    def test_register_reports_repair_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "data/training/manual_verified_results/phase_e4_direct_target_xgb_rf_v1/reports"
            out.mkdir(parents=True)
            (out / "phase_e4_manifest.json").write_text("{}", encoding="utf-8")
            with mock.patch(
                "app.artifacts.registry.scan_artifacts",
                side_effect=RuntimeError("boom"),
            ):
                status = register_e4_artifacts(root, out.parent)
            self.assertFalse(status["success"])
            self.assertIn("register_existing_artifacts.py", status["repair_command"])


class OutputDirTests(unittest.TestCase):
    def test_prepare_output_dirs_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            models = out / "models"
            models.mkdir(parents=True)
            stale = models / "stale.joblib"
            stale.write_text("x", encoding="utf-8")
            prepare_output_dirs(out, overwrite=True)
            self.assertFalse(stale.exists())
            self.assertTrue((out / "audit").is_dir())


class IntegrationSmokeTests(unittest.TestCase):
    def test_end_to_end_rf_only_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            name = "LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL080_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.csv"
            _synthetic_dataset().to_csv(input_dir / name, index=False)

            from app.training.direct_target_xgb_rf import TrainConfig, run_training

            config = TrainConfig(
                input_dir=input_dir,
                output_dir=output_dir,
                models=("RF",),
                smoke=True,
                overwrite=True,
                register_artifacts=False,
                max_rows=120,
                random_state=42,
                selected_descriptors=discover_direct_target_datasets(input_dir),
            )
            result = run_training(config)
            self.assertEqual(result["datasets_completed"], 1)
            self.assertTrue(any(output_dir.glob("models/*.joblib")))
            self.assertTrue((output_dir / "audit/phase_e4_run_audit.jsonl").exists())
            self.assertTrue((output_dir / "audit/phase_e4_target_normalization_audit.csv").exists())


if __name__ == "__main__":
    unittest.main()
