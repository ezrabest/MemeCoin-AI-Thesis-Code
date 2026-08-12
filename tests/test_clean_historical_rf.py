"""Tests for Phase E8B clean historical RF training infrastructure."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.training.clean_historical_rf import (  # noqa: E402
    SAFE_CORE_FEATURES,
    TARGET_COL,
    CleanRFForbiddenFeatureError,
    TrainConfig,
    build_rf_pipeline,
    compute_top_pct_metrics,
    filter_descriptors,
    make_output_dir,
    prepare_dataset,
    recall_at_top_pct,
    resolve_safe_features,
    run_training,
    select_validation_policy,
    set_forbidden_audit_path,
    temporal_split,
    validate_feature_schema,
    pair_overlap_diagnostics,
    robustness_diagnostics,
    seen_unseen_pair_diagnostics,
    DatasetDescriptor,
)
from app.training.direct_target_xgb_rf import discover_direct_target_datasets  # noqa: E402


def _synthetic_rows(n: int, *, start_hour: int = 0) -> list[dict]:
    rng = np.random.default_rng(42)
    rows: list[dict] = []
    for i in range(n):
        label = int(rng.random() > 0.85)
        row = {
            "pair_address": f"pair_{i % 7}",
            "event_timestamp": f"2026-06-01T{((start_hour + i) % 24):02d}:00:00Z",
            "label_valid": True,
            TARGET_COL: label,
            "sim_net_return": 0.5 if label else -0.03,
            "gap_detected": False,
            "target": 0,
        }
        for feat in SAFE_CORE_FEATURES:
            row[feat] = float(rng.normal()) if feat not in {"time_stop_minutes", "txns_buys", "txns_sells", "txns_total"} else int(rng.integers(1, 100))
        rows.append(row)
    return rows


def _synthetic_dataset(n: int = 120) -> pd.DataFrame:
    return pd.DataFrame(_synthetic_rows(n))


class FeatureValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        set_forbidden_audit_path(None)

    def test_rejects_forbidden_exact_and_pattern_columns(self) -> None:
        forbidden = [
            "gap_detected",
            "exit_timestamp",
            "future_window_start_timestamp",
            TARGET_COL,
            "sim_net_return",
            "label_valid",
            "synthetic_future_col",
            "synthetic_target_x",
            "synthetic_label_x",
            "synthetic_sim_x",
            "synthetic_exit_x",
            "synthetic_gap_x",
        ]
        for col in forbidden:
            with self.subTest(col=col):
                with self.assertRaises(CleanRFForbiddenFeatureError):
                    validate_feature_schema([col])

    def test_accepts_safe_core_feature_list(self) -> None:
        result = validate_feature_schema(list(SAFE_CORE_FEATURES))
        self.assertTrue(result["valid"])
        self.assertEqual(result["accepted_feature_list"], list(SAFE_CORE_FEATURES))

    def test_resolve_safe_features_excludes_non_feature_columns(self) -> None:
        df = _synthetic_dataset(10)
        features = resolve_safe_features(list(df.columns))
        self.assertEqual(features, list(SAFE_CORE_FEATURES))
        self.assertNotIn(TARGET_COL, features)
        self.assertNotIn("pair_address", features)
        self.assertNotIn("event_timestamp", features)
        self.assertNotIn("label_valid", features)

    def test_writes_forbidden_audit_on_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "forbidden.csv"
            set_forbidden_audit_path(audit_path)
            with self.assertRaises(CleanRFForbiddenFeatureError):
                validate_feature_schema(["gap_detected"])
            self.assertTrue(audit_path.exists())
            audit_df = pd.read_csv(audit_path)
            self.assertIn("gap_detected", audit_df["feature"].tolist())


class TemporalSplitTests(unittest.TestCase):
    def test_chronological_split_order(self) -> None:
        df = prepare_dataset(_synthetic_dataset(100), max_rows=None)
        train, val, test, meta = temporal_split(df)
        self.assertGreater(len(train), 0)
        self.assertGreater(len(val), 0)
        self.assertGreater(len(test), 0)
        self.assertLessEqual(train["event_timestamp"].max(), val["event_timestamp"].min())
        self.assertLessEqual(val["event_timestamp"].max(), test["event_timestamp"].min())
        self.assertEqual(meta["train_rows"] + meta["validation_rows"] + meta["test_rows"], meta["total_rows"])

    def test_fit_uses_train_only(self) -> None:
        df = prepare_dataset(_synthetic_dataset(90), max_rows=None)
        train, val, test, _ = temporal_split(df)
        features = resolve_safe_features(list(df.columns))
        config = TrainConfig(
            dataset_root=Path("."),
            output_dir=Path("."),
            n_estimators=10,
            n_jobs=1,
            random_state=42,
        )
        pipeline = build_rf_pipeline(config)
        x_train = train[features].apply(pd.to_numeric, errors="coerce")
        pipeline.fit(x_train, train[TARGET_COL].to_numpy())
        self.assertIsNotNone(pipeline.named_steps["model"])


class MetricsTests(unittest.TestCase):
    def test_precision_and_recall_at_top_pct(self) -> None:
        y_true = np.array([1, 0, 1, 0, 0, 0, 1, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        frame = pd.DataFrame({TARGET_COL: y_true, "pair_address": [f"p{i}" for i in range(10)]})
        metrics = compute_top_pct_metrics(frame, y_true, y_score, 10.0, return_col=None)
        self.assertEqual(metrics["selected_count"], 1)
        self.assertEqual(metrics["precision_at_top_pct"], 1.0)
        recall = recall_at_top_pct(y_true, y_score, 10.0)
        self.assertAlmostEqual(recall or 0.0, 1 / 3, places=4)

    def test_degenerate_roc_auc_is_null(self) -> None:
        y_true = np.zeros(10, dtype=int)
        y_score = np.linspace(0.1, 0.9, 10)
        from app.training.clean_historical_rf import compute_split_metrics

        rows = compute_split_metrics(
            pd.DataFrame({TARGET_COL: y_true}),
            y_true,
            y_score,
            split_name="test",
            return_col=None,
        )
        self.assertIsNone(rows[0]["roc_auc"])
        self.assertIsNone(rows[0]["pr_auc"])


class PolicySelectionTests(unittest.TestCase):
    def test_validation_policy_selected_on_validation_only(self) -> None:
        val_metrics = [
            {"split": "validation", "top_pct": 5.0, "precision_at_top_pct": 0.2, "selected_count": 10, "selected_unique_pairs": 5, "selected_average_sim_net_return": 0.01, "selected_total_sim_net_return": 0.1},
            {"split": "validation", "top_pct": 1.0, "precision_at_top_pct": 0.5, "selected_count": 8, "selected_unique_pairs": 4, "selected_average_sim_net_return": 0.05, "selected_total_sim_net_return": 0.4},
            {"split": "test", "top_pct": 10.0, "precision_at_top_pct": 0.9, "selected_count": 20, "selected_unique_pairs": 10},
        ]
        selected = select_validation_policy(val_metrics)
        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected["top_pct"], 1.0)


class PairDiagnosticsTests(unittest.TestCase):
    def test_pair_overlap_diagnostics(self) -> None:
        df = prepare_dataset(_synthetic_dataset(60), max_rows=None)
        train, val, test, _ = temporal_split(df)
        desc = DatasetDescriptor(
            dataset_name="test",
            dataset_path=Path("x.csv"),
            filter_name="RAW_ALL_VERIFIED",
            horizon="1h",
            exit_policy_id="TP20308_SL080_FEE0308_TIME_BY_HORIZON",
            target_name="net_profitable_after_exit_policy",
            target_version="v1",
        )
        diag = pair_overlap_diagnostics(train, val, test, descriptor=desc)
        self.assertIn("train_val_pair_overlap_count", diag)

    def test_seen_unseen_empty_group_does_not_crash(self) -> None:
        df = prepare_dataset(_synthetic_dataset(30), max_rows=None)
        train, _, test, _ = temporal_split(df)
        test = test.copy()
        test["pair_address"] = "brand_new_pair_only"
        desc = DatasetDescriptor(
            dataset_name="test",
            dataset_path=Path("x.csv"),
            filter_name="RAW_ALL_VERIFIED",
            horizon="1h",
            exit_policy_id="TP20308_SL080_FEE0308_TIME_BY_HORIZON",
            target_name="net_profitable_after_exit_policy",
            target_version="v1",
        )
        scores = np.linspace(0.9, 0.1, len(test))
        rows = seen_unseen_pair_diagnostics(test, train, scores, 5.0, descriptor=desc, return_col="sim_net_return")
        groups = {r["group"] for r in rows}
        self.assertIn("unseen_pair", groups)

    def test_robustness_small_selected_set(self) -> None:
        df = prepare_dataset(_synthetic_dataset(40), max_rows=None)
        _, val, _, _ = temporal_split(df)
        features = resolve_safe_features(list(df.columns))
        config = TrainConfig(dataset_root=Path("."), output_dir=Path("."), n_estimators=10, n_jobs=1)
        pipeline = build_rf_pipeline(config)
        x_val = val[features].apply(pd.to_numeric, errors="coerce")
        pipeline.fit(x_val, val[TARGET_COL].to_numpy())
        scores = pipeline.predict_proba(x_val)[:, 1]
        desc = DatasetDescriptor(
            dataset_name="test",
            dataset_path=Path("x.csv"),
            filter_name="RAW_ALL_VERIFIED",
            horizon="1h",
            exit_policy_id="TP20308_SL080_FEE0308_TIME_BY_HORIZON",
            target_name="net_profitable_after_exit_policy",
            target_version="v1",
        )
        rows = robustness_diagnostics(val, scores, 10.0, descriptor=desc, split_name="validation", return_col="sim_net_return")
        self.assertGreater(len(rows), 0)


class SmokeTrainingTests(unittest.TestCase):
    def test_smoke_training_writes_manifest_and_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset_dir = root / "datasets"
            dataset_dir.mkdir()
            name = "RAW_ALL_VERIFIED_1h_TP20308_SL080_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.csv"
            _synthetic_dataset(120).to_csv(dataset_dir / name, index=False)
            output_dir = make_output_dir(root / "results")
            config = TrainConfig(
                dataset_root=dataset_dir,
                output_dir=output_dir,
                smoke=True,
                max_rows=100,
                n_estimators=20,
                n_jobs=1,
                random_state=42,
                selected_descriptors=discover_direct_target_datasets(dataset_dir),
            )
            result = run_training(config)
            self.assertEqual(result["datasets_completed"], 1)
            manifest_path = output_dir / "reports" / "clean_rf_run_manifest.json"
            self.assertTrue(manifest_path.exists())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["phase"], "E8B")
            self.assertTrue(manifest["no_runtime_changes"])
            self.assertFalse(manifest["old_rf_sidecars_used"])
            self.assertIn("safe_feature_list", manifest)
            self.assertTrue((output_dir / "reports" / "clean_rf_split_summary.csv").exists())
            self.assertTrue((output_dir / "reports" / "clean_rf_leakage_audit.csv").exists())
            forbidden = output_dir / "reports" / "clean_rf_forbidden_feature_audit.csv"
            self.assertTrue(forbidden.exists())


class DescriptorFilterTests(unittest.TestCase):
    def test_smoke_prefers_raw_all_verified_1h_sl080(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            names = [
                "RAW_ALL_VERIFIED_30m_TP20308_SL075_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.csv",
                "RAW_ALL_VERIFIED_1h_TP20308_SL080_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.csv",
            ]
            for name in names:
                _synthetic_dataset(20).to_csv(root / name, index=False)
            descriptors = discover_direct_target_datasets(root)
            filtered = filter_descriptors(
                descriptors,
                filters=("RAW_ALL_VERIFIED",),
                horizons=("30m", "1h", "4h", "8h", "24h"),
                exit_policies=(
                    "TP20308_SL075_FEE0308_TIME_BY_HORIZON",
                    "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
                ),
                smoke=True,
            )
            self.assertEqual(len(filtered), 1)
            self.assertEqual(filtered[0].horizon, "1h")
            self.assertEqual(filtered[0].exit_policy_id, "TP20308_SL080_FEE0308_TIME_BY_HORIZON")


if __name__ == "__main__":
    unittest.main()
