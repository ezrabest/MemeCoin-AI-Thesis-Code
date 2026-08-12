"""TabICLv2 offline evaluation tests (no GPU required)."""
from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.training.tabicl_v2_eval import (
    NearestNeighborContextIndex,
    RollingKnnConfig,
    RollingKnnContextSelector,
    TrainOnlyPreprocessor,
    base_report_flags,
    build_rolling_knn_config,
    build_static_context_indices,
    cap_context_size,
    chronological_split,
    compute_split_metrics,
    evaluate_tabicl_v2,
    limit_features_by_variance,
    parse_bool_flag,
    precision_at_top_k_with_count,
    prediction_output_paths,
    resolve_full_evaluation,
    resolve_output_label,
    rolling_knn_enabled,
    sample_context_indices,
    sample_positive_enriched_indices,
    sample_stratified_recent_indices,
    select_rolling_temporal_indices,
    select_tabicl_feature_columns,
    select_whale_wave_feature_columns,
    tabicl_available,
    validate_context_strategy,
    whale_wave_feature_indices,
)
from scripts.evaluate_tabicl_v2 import build_parser as build_eval_parser


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _synthetic_dataset(n: int = 200) -> pd.DataFrame:
    t0 = datetime(2026, 6, 1, 0, 0, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        ts = t0 + timedelta(minutes=i)
        rows.append({
            "event_timestamp": _iso(ts),
            "symbol": f"COIN{i % 5}",
            "pair_address": f"pair_{i % 5}",
            "whale_wave_score": 0.4 + (i % 10) * 0.02,
            "volume_spike_ratio_15m_vs_1h": 1.0 + (i % 7) * 0.1,
            "buy_sell_ratio": 1.0 + (i % 5) * 0.05,
            "liquidity_usd": 10000 + i,
            "target_profitable_4h": int(i % 23 == 0),
            "target_return_4h": 0.08 if i % 23 == 0 else -0.02,
            "future_return_4h": 0.08 if i % 23 == 0 else -0.02,
            "label_up_4h": int(i % 29 == 0),
            "big_pump_4h": int(i % 47 == 0),
            "optimal_trade_class_4h": "NO_TRADE",
            "timestamp": _iso(ts),
            "created_at": _iso(ts),
            "reasoning": "exclude me",
        })
    return pd.DataFrame(rows)


class TabICLFeatureTests(unittest.TestCase):
    def test_leakage_columns_excluded(self) -> None:
        frame = _synthetic_dataset(30)
        numeric, excluded = select_tabicl_feature_columns(frame)
        self.assertNotIn("target_return_4h", numeric)
        self.assertNotIn("future_return_4h", numeric)
        self.assertNotIn("label_up_4h", numeric)
        self.assertNotIn("big_pump_4h", numeric)
        self.assertNotIn("optimal_trade_class_4h", numeric)
        self.assertNotIn("timestamp", numeric)
        self.assertNotIn("created_at", numeric)
        self.assertIn("whale_wave_score", numeric)

    def test_numeric_bool_only(self) -> None:
        frame = _synthetic_dataset(10)
        frame["whale_wave_direction"] = ["UP", "DOWN"] * 5
        numeric, excluded = select_tabicl_feature_columns(frame)
        self.assertNotIn("whale_wave_direction", numeric)
        self.assertIn("whale_wave_direction", excluded)


class TabICLSplitTests(unittest.TestCase):
    def test_chronological_split_no_shuffle(self) -> None:
        frame = _synthetic_dataset(100)
        train, val, test = chronological_split(frame)
        self.assertEqual(len(train), 70)
        self.assertEqual(len(val), 15)
        self.assertEqual(len(test), 15)
        combined = pd.concat([train, val, test], ignore_index=True)
        self.assertTrue(combined["event_timestamp"].is_monotonic_increasing)

    def test_context_size_cap(self) -> None:
        self.assertEqual(cap_context_size(1024, 1024, 500), 500)
        self.assertEqual(cap_context_size(2048, 1024, 5000), 1024)
        self.assertEqual(cap_context_size(1024, 1024, 0), 0)


class TabICLContextSamplingTests(unittest.TestCase):
    def test_context_sampling_includes_positives_and_recent_negatives(self) -> None:
        y = np.array([0] * 80 + [1] * 4 + [0] * 16 + [1] * 2 + [0] * 18, dtype=int)
        idx = sample_context_indices(y, context_size=40, random_state=42)
        self.assertLessEqual(len(idx), 40)
        self.assertGreater(len(idx), 0)
        pos_rate = y[idx].mean()
        self.assertGreater(pos_rate, 0.0)
        self.assertLessEqual(pos_rate, 0.35)

    def test_context_sampling_deterministic(self) -> None:
        y = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 0] * 20, dtype=int)
        a = sample_context_indices(y, context_size=50, random_state=42)
        b = sample_context_indices(y, context_size=50, random_state=42)
        np.testing.assert_array_equal(a, b)

    def test_stratified_recent_uses_train_indices_only(self) -> None:
        y = np.array([0, 0, 1, 0, 1, 0, 0, 0, 1, 0] * 20, dtype=int)
        idx = sample_stratified_recent_indices(y, context_size=30, positive_context_ratio=0.25)
        self.assertTrue((idx >= 0).all())
        self.assertTrue((idx < len(y)).all())

    def test_positive_enriched_increases_positive_ratio(self) -> None:
        y = np.array([0] * 180 + [1] * 20, dtype=int)
        stratified = sample_stratified_recent_indices(
            y, context_size=40, positive_context_ratio=0.25, random_state=42,
        )
        enriched = sample_positive_enriched_indices(
            y, context_size=40, positive_context_ratio=0.50, random_state=42,
        )
        self.assertGreater(y[enriched].mean(), y[stratified].mean())


class TabICLNearestNeighborTests(unittest.TestCase):
    def test_nearest_neighbors_fits_once_not_per_batch(self) -> None:
        rng = np.random.RandomState(42)
        x_train = rng.randn(120, 8)
        y_train = (rng.rand(120) > 0.85).astype(int)
        index = NearestNeighborContextIndex(x_train, y_train, metric="euclidean")
        self.assertEqual(index.fit_count, 1)
        with patch.object(index._nn, "fit", wraps=index._nn.fit) as fit_mock:
            for _ in range(4):
                batch = rng.randn(16, 8)
                index.build_context_indices(batch, context_size=32)
            fit_mock.assert_not_called()

    def test_nearest_neighbors_never_uses_validation_or_test_rows(self) -> None:
        rng = np.random.RandomState(7)
        x_train = rng.randn(100, 6)
        y_train = (rng.rand(100) > 0.9).astype(int)
        x_val = rng.randn(20, 6) + 10.0
        index = NearestNeighborContextIndex(x_train, y_train)
        ctx = index.build_context_indices(x_val, context_size=24)
        self.assertTrue((ctx >= 0).all())
        self.assertTrue((ctx < len(x_train)).all())

    def test_self_neighbor_excluded_for_train_queries(self) -> None:
        rng = np.random.RandomState(11)
        x_train = rng.randn(50, 5)
        y_train = (rng.rand(50) > 0.8).astype(int)
        index = NearestNeighborContextIndex(x_train, y_train)
        for row_idx in range(8):
            ctx = index.build_context_indices(
                x_train[row_idx: row_idx + 1],
                context_size=16,
                exclude_zero_distance=True,
            )
            self.assertNotIn(row_idx, ctx)


class TabICLRollingKnnTests(unittest.TestCase):
    def _train_timestamps(self, n: int, start_day: int = 0) -> np.ndarray:
        base = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=start_day)
        return np.array(
            [base + timedelta(hours=i) for i in range(n)],
            dtype="datetime64[ns]",
        )

    def test_cli_accepts_knn_rolling_days(self) -> None:
        parser = build_eval_parser()
        args = parser.parse_args(["--knn-rolling-days", "14"])
        self.assertEqual(args.knn_rolling_days, 14)

    def test_rolling_knn_only_uses_rows_before_batch_min_time(self) -> None:
        ts = self._train_timestamps(100)
        batch_min = pd.Timestamp("2026-01-03T00:00:00Z")
        slice_indices, _, _ = select_rolling_temporal_indices(
            ts,
            batch_min,
            rolling_days=14,
            min_context_rows=10,
            expand_window=False,
            max_rolling_days=90,
        )
        self.assertTrue((ts[slice_indices] < np.datetime64(batch_min.to_datetime64())).all())

    def test_rolling_knn_excludes_rows_at_or_after_batch_min_time(self) -> None:
        ts = self._train_timestamps(50)
        batch_min = pd.Timestamp(ts[25], tz="UTC")
        slice_indices, _, _ = select_rolling_temporal_indices(
            ts,
            batch_min,
            rolling_days=14,
            min_context_rows=5,
            expand_window=False,
            max_rolling_days=90,
        )
        self.assertTrue((ts[slice_indices] < np.datetime64(batch_min.to_datetime64())).all())
        self.assertNotIn(25, slice_indices)

    def test_event_timestamp_used_for_slicing_not_features(self) -> None:
        frame = _synthetic_dataset(20)
        numeric, excluded = select_tabicl_feature_columns(frame)
        self.assertNotIn("event_timestamp", numeric)
        self.assertIn("event_timestamp", excluded)

    def test_fallback_when_temporal_slice_empty(self) -> None:
        ts = self._train_timestamps(20)
        batch_min = pd.Timestamp("2025-01-01T00:00:00Z")
        slice_indices, _, used_fallback = select_rolling_temporal_indices(
            ts,
            batch_min,
            rolling_days=14,
            min_context_rows=512,
            expand_window=True,
            max_rolling_days=90,
        )
        self.assertTrue(used_fallback)
        self.assertEqual(len(slice_indices), 0)

    def test_window_expands_when_too_few_rows(self) -> None:
        ts = self._train_timestamps(200)
        batch_min = pd.Timestamp(ts[-1], tz="UTC")
        small_window, window_small, _ = select_rolling_temporal_indices(
            ts,
            batch_min,
            rolling_days=1,
            min_context_rows=100,
            expand_window=False,
            max_rolling_days=90,
        )
        expanded, window_large, _ = select_rolling_temporal_indices(
            ts,
            batch_min,
            rolling_days=1,
            min_context_rows=100,
            expand_window=True,
            max_rolling_days=90,
        )
        self.assertLess(len(small_window), len(expanded))
        self.assertGreaterEqual(window_large, window_small)

    def test_nearest_neighbors_not_fit_per_row(self) -> None:
        rng = np.random.RandomState(0)
        x_train = rng.randn(80, 4)
        y_train = (rng.rand(80) > 0.9).astype(int)
        ts = self._train_timestamps(80)
        config = build_rolling_knn_config(knn_rolling_days=14)
        selector = RollingKnnContextSelector(
            x_train, y_train, ts, config=config,
        )
        initial_fits = selector.knn_index_fit_count
        for i in range(5):
            batch_ts = np.array([ts[50 + i * 6]], dtype="datetime64[ns]")
            selector.build_global_context_indices(
                x_train[50 + i * 6: 51 + i * 6],
                batch_ts,
                context_size=16,
            )
        self.assertLessEqual(selector.knn_index_fit_count - initial_fits, 5)

    def test_knn_cache_used_for_same_day_bucket(self) -> None:
        rng = np.random.RandomState(1)
        x_train = rng.randn(120, 5)
        y_train = (rng.rand(120) > 0.85).astype(int)
        ts = self._train_timestamps(120, start_day=10)
        config = RollingKnnConfig(rolling_days=14, min_context_rows=8, expand_window=True, max_rolling_days=90)
        selector = RollingKnnContextSelector(
            x_train, y_train, ts, config=config,
        )
        day_ts = np.array([ts[80], ts[81]], dtype="datetime64[ns]")
        selector.build_global_context_indices(x_train[80:82], day_ts, context_size=12)
        selector.build_global_context_indices(x_train[82:84], day_ts, context_size=12)
        self.assertGreater(selector.knn_cache_hit_count, 0)

    def test_train_only_mode_never_uses_validation_indices(self) -> None:
        rng = np.random.RandomState(3)
        x_train = rng.randn(60, 4)
        y_train = (rng.rand(60) > 0.88).astype(int)
        ts = self._train_timestamps(60)
        config = build_rolling_knn_config(knn_rolling_days=14, knn_min_context_rows=8)
        selector = RollingKnnContextSelector(
            x_train, y_train, ts, config=config,
        )
        val_ts = np.array([ts[-1] + np.timedelta64(1, "D")], dtype="datetime64[ns]")
        ctx = selector.build_global_context_indices(rng.randn(1, 4), val_ts, context_size=10)
        self.assertTrue((ctx >= 0).all())
        self.assertTrue((ctx < len(x_train)).all())

    def test_self_neighbor_leakage_still_prevented(self) -> None:
        rng = np.random.RandomState(11)
        x_train = rng.randn(50, 5)
        y_train = (rng.rand(50) > 0.8).astype(int)
        index = NearestNeighborContextIndex(x_train, y_train)
        for row_idx in range(5):
            ctx = index.build_context_indices(
                x_train[row_idx: row_idx + 1],
                context_size=16,
                exclude_zero_distance=True,
            )
            self.assertNotIn(row_idx, ctx)

    def test_rolling_knn_diagnostics_present(self) -> None:
        rng = np.random.RandomState(4)
        x_train = rng.randn(40, 3)
        y_train = (rng.rand(40) > 0.85).astype(int)
        ts = self._train_timestamps(40)
        selector = RollingKnnContextSelector(
            x_train,
            y_train,
            ts,
            config=build_rolling_knn_config(knn_rolling_days=7, knn_min_context_rows=5),
        )
        batch_ts = np.array([ts[30]], dtype="datetime64[ns]")
        selector.build_global_context_indices(x_train[30:31], batch_ts, context_size=8)
        diag = selector.diagnostics()
        self.assertEqual(diag["rolling_context_mode"], "train_only")
        self.assertTrue(diag["event_timestamp_used_for_slicing"])
        self.assertFalse(diag["event_timestamp_used_as_feature"])
        self.assertIn("knn_index_fit_count", diag)

    def test_rolling_enabled_flag(self) -> None:
        self.assertFalse(rolling_knn_enabled(None))
        self.assertFalse(rolling_knn_enabled(0))
        self.assertTrue(rolling_knn_enabled(14))

    def test_parse_bool_flag(self) -> None:
        self.assertTrue(parse_bool_flag("true"))
        self.assertFalse(parse_bool_flag("false"))


class TabICLWhaleWaveTests(unittest.TestCase):
    def test_whale_wave_fallback_when_few_whale_columns(self) -> None:
        feature_cols = ["alpha_metric", "beta_metric", "gamma_metric"]
        selected = select_whale_wave_feature_columns(feature_cols)
        self.assertEqual(selected, feature_cols)

    def test_whale_wave_uses_keyword_columns_when_enough_exist(self) -> None:
        feature_cols = [
            "whale_wave_score",
            "volume_spike_ratio_15m_vs_1h",
            "buy_sell_ratio",
            "liquidity_usd",
            "txns_5m",
            "price_change_1h",
            "other_metric",
        ]
        selected = select_whale_wave_feature_columns(feature_cols)
        self.assertIn("whale_wave_score", selected)
        self.assertNotIn("other_metric", selected)
        idx = whale_wave_feature_indices(feature_cols)
        self.assertEqual(len(idx), len(selected))


class TabICLEnsembleTests(unittest.TestCase):
    def test_ensemble_averages_probabilities(self) -> None:
        member_scores = [
            np.array([0.2, 0.4, 0.6]),
            np.array([0.4, 0.6, 0.8]),
            np.array([0.6, 0.8, 1.0]),
        ]
        avg = np.mean(np.vstack(member_scores), axis=0)
        np.testing.assert_allclose(avg, [0.4, 0.6, 0.8])


class TabICLOutputPathTests(unittest.TestCase):
    def test_output_filenames_include_strategy_or_suffix(self) -> None:
        root = Path("/tmp/training")
        val_path, test_path, report_path, features_path = prediction_output_paths(
            root, "stratified_recent",
        )
        self.assertIn("stratified_recent", val_path.name)
        self.assertIn("stratified_recent", test_path.name)
        self.assertIn("stratified_recent", report_path.name)
        self.assertIsNotNone(features_path)
        self.assertFalse(val_path.name.endswith("_.parquet"))

    def test_legacy_output_paths_without_suffix(self) -> None:
        root = Path("/tmp/training")
        val_path, _, report_path, features_path = prediction_output_paths(root, None)
        self.assertEqual(val_path.name, "tabicl_v2_predictions_validation.parquet")
        self.assertEqual(report_path.name, "tabicl_v2_report.json")
        self.assertIsNone(features_path)


class TabICLFullEvaluationTests(unittest.TestCase):
    def test_full_evaluation_true_when_no_actual_cap(self) -> None:
        full, reason = resolve_full_evaluation(max_rows=None, partial_evaluation_reason="note only")
        self.assertTrue(full)
        self.assertEqual(reason, "note only")

    def test_full_evaluation_false_when_max_rows_used(self) -> None:
        full, reason = resolve_full_evaluation(
            max_rows=1000,
            partial_evaluation_reason="max_rows capped at 1000",
        )
        self.assertFalse(full)


class TabICLMetricTests(unittest.TestCase):
    def test_precision_at_top_k(self) -> None:
        y_true = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        y_score = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        top1 = precision_at_top_k_with_count(y_true, y_score, 10.0)
        self.assertEqual(top1["precision"], 1.0)
        self.assertEqual(top1["trade_count"], 1)

    def test_compute_split_metrics_return_kind(self) -> None:
        frame = _synthetic_dataset(50)
        y_true = frame["target_profitable_4h"].astype(int).to_numpy()
        y_score = np.linspace(0.9, 0.1, len(frame))
        metrics = compute_split_metrics(
            frame.assign(y_true=y_true),
            y_true,
            y_score,
            top_pcts=[0.01, 0.02, 0.05],
            return_col="target_return_4h",
        )
        self.assertEqual(metrics["return_column_kind"], "raw_not_fee_adjusted")
        self.assertIn("precision_at_top_1_percent", metrics)
        self.assertEqual(metrics["row_count"], len(frame))


class TabICLPreprocessorTests(unittest.TestCase):
    def test_scaler_imputer_train_only(self) -> None:
        frame = _synthetic_dataset(120)
        train, val, _ = chronological_split(frame)
        numeric, _ = select_tabicl_feature_columns(frame)
        numeric = numeric[:5]
        prep = TrainOnlyPreprocessor(scaler="standard")
        prep.fit(train, numeric)
        x_val = prep.transform(val)
        self.assertFalse(np.isnan(x_val).any())
        train_medians = prep.imputer.statistics_
        self.assertEqual(len(train_medians), len(numeric))

    def test_max_features_omitted_uses_all_eligible_features(self) -> None:
        frame = _synthetic_dataset(80)
        numeric, _ = select_tabicl_feature_columns(frame)
        limited = limit_features_by_variance(frame, numeric, None)
        self.assertEqual(limited, numeric)


class TabICLReportFlagTests(unittest.TestCase):
    def test_report_flags(self) -> None:
        flags = base_report_flags()
        self.assertFalse(flags["is_oracle_backtest"])
        self.assertTrue(flags["offline_only"])
        self.assertTrue(flags["uses_tabicl_v2"])
        self.assertFalse(flags["uses_new_llm_calls"])
        self.assertFalse(flags["calls_gemini"])
        self.assertFalse(flags["calls_ollama"])
        self.assertFalse(flags["modifies_sqlite"])
        self.assertFalse(flags["changes_live_behavior"])
        self.assertEqual(flags["venv_expected"], ".venv-tabicl")


class TabICLCliTests(unittest.TestCase):
    def test_cli_accepts_context_strategy(self) -> None:
        parser = build_eval_parser()
        args = parser.parse_args(["--context-strategy", "stratified_recent"])
        self.assertEqual(args.context_strategy, "stratified_recent")

    def test_invalid_context_strategy_fails_clearly(self) -> None:
        with self.assertRaises(ValueError):
            validate_context_strategy("not_a_real_strategy")

    def test_resolve_output_label_prefers_suffix(self) -> None:
        self.assertEqual(
            resolve_output_label("stratified_recent", "custom_suffix"),
            "custom_suffix",
        )


class TabICLEvaluateScriptTests(unittest.TestCase):
    def test_evaluate_script_exists(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_tabicl_v2.py"
        self.assertTrue(path.is_file())

    def test_sweep_script_exists(self) -> None:
        path = Path(__file__).resolve().parents[1] / "scripts" / "sweep_tabicl_v2_context_strategies.py"
        self.assertTrue(path.is_file())

    def test_missing_target_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data.parquet"
            frame = _synthetic_dataset(80)
            frame = frame.drop(columns=["target_profitable_4h"])
            frame.to_parquet(data_path, index=False)
            with self.assertRaises(ValueError):
                evaluate_tabicl_v2(dataset_path=data_path, target="target_profitable_4h")

    def test_missing_dataset_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            evaluate_tabicl_v2(dataset_path=Path("/nonexistent/dataset.parquet"))

    @unittest.skipUnless(tabicl_available(), "tabicl not installed in this environment")
    def test_batching_configuration_small_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data.parquet"
            output_dir = Path(tmp) / "out"
            _synthetic_dataset(120).to_parquet(data_path, index=False)
            report = evaluate_tabicl_v2(
                dataset_path=data_path,
                target="target_profitable_4h",
                output_dir=output_dir,
                context_size=32,
                batch_size=16,
                max_train_context_rows=32,
                max_features=10,
                device="cpu",
            )
            self.assertLessEqual(report["context_size_used"], 32)
            self.assertEqual(report["batch_size_used"], 16)
            self.assertTrue(report["full_evaluation"])
            self.assertEqual(report["validation_row_count"], report["tabicl_metrics"]["validation"]["row_count"])
            self.assertTrue(Path(report["output_files"]["report_json"]).is_file())

    def test_skip_when_tabicl_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data.parquet"
            _synthetic_dataset(80).to_parquet(data_path, index=False)
            with patch("app.training.tabicl_v2_eval.tabicl_available", return_value=False):
                with patch(
                    "app.training.tabicl_v2_eval.run_tabicl_with_oom_retry",
                    side_effect=ImportError("tabicl missing"),
                ):
                    with self.assertRaises(ImportError):
                        evaluate_tabicl_v2(dataset_path=data_path, target="target_profitable_4h")

    def test_report_includes_row_counts_and_feature_info(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_path = Path(tmp) / "data.parquet"
            output_dir = Path(tmp) / "out"
            _synthetic_dataset(90).to_parquet(data_path, index=False)

            def fake_run(x_context, y_context, x_val, x_test, **kwargs):
                return (
                    np.linspace(0.1, 0.9, len(x_val)),
                    np.linspace(0.2, 0.8, len(x_test)),
                    {
                        "context_size_used": 32,
                        "batch_size_used": 16,
                        "oom_retry_count": 0,
                    },
                )

            with patch("app.training.tabicl_v2_eval.run_tabicl_with_oom_retry", side_effect=fake_run):
                report = evaluate_tabicl_v2(
                    dataset_path=data_path,
                    target="target_profitable_4h",
                    output_dir=output_dir,
                    context_strategy="stratified_recent",
                    output_suffix="unit_test",
                    max_rows=90,
                )
            self.assertEqual(
                report["train_row_count"] + report["validation_row_count"] + report["test_row_count"],
                report["total_rows_used"],
            )
            self.assertEqual(
                report["validation_row_count"],
                report["tabicl_metrics"]["validation"]["row_count"],
            )
            self.assertEqual(report["feature_count"], report["features_used_count"])
            self.assertIsNone(report["max_features"])
            self.assertFalse(report["full_evaluation"])


class TabICLBatchingConfigTests(unittest.TestCase):
    def test_predict_proba_batched_splits(self) -> None:
        from app.training.tabicl_v2_eval import predict_proba_batched

        class FakeClf:
            def predict_proba(self, x: np.ndarray) -> np.ndarray:
                return np.column_stack([1.0 - x[:, 0], x[:, 0]])

        x = np.arange(100, dtype=float).reshape(-1, 1) / 100.0
        out = predict_proba_batched(FakeClf(), x, batch_size=30)
        self.assertEqual(out.shape, (100, 2))

    def test_static_context_strategy_selection(self) -> None:
        y = np.array([0] * 180 + [1] * 20, dtype=int)
        idx = build_static_context_indices(
            "positive_enriched",
            y,
            context_size=40,
            max_train_context_rows=40,
        )
        self.assertLessEqual(len(idx), 40)
        self.assertTrue((idx < len(y)).all())


if __name__ == "__main__":
    unittest.main()
