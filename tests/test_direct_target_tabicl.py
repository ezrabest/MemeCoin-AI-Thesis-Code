"""Tests for Phase E5 direct-target TabICL evaluation infrastructure."""

from __future__ import annotations

import json
import os
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

from app.training.direct_target_tabicl import (  # noqa: E402
    DEFAULT_MAX_WORKERS,
    SMOKE_DEFAULT_MAX_CONTEXT_SIZE,
    SMOKE_DEFAULT_MAX_ROWS,
    BoundedContextCache,
    E5IncrementalAuditLogger,
    EvalConfig,
    apply_smoke_stratified_split_sampling,
    assert_full_split_positive_minimums,
    assign_consensus_tier,
    build_consensus_frame,
    build_e5c_decision_summary,
    build_validation_selected_consensus_applied_to_test,
    regenerate_e5c_reporting_from_artifacts,
    build_context_cache_key,
    build_context_construction_strategy_hash,
    build_train_identity_hash,
    check_context_drift,
    compute_rank_percentile,
    merge_tab_xgb_rf_predictions,
    normalize_target_column_strict,
    reindex_features,
    run_bounded_executor,
    run_dependency_audit,
    run_evaluation,
    run_memory_cleanup,
    sanitize_join_keys,
    strict_coerce_binary_target,
    strict_merge_one_to_one,
    target_sanity_check,
)
from app.training.direct_target_xgb_rf import (  # noqa: E402
    CANONICAL_TARGET,
    DatasetDescriptor,
    apply_deterministic_row_limit,
    build_feature_columns,
    derive_valid_label_mask,
)


def _descriptor() -> DatasetDescriptor:
    return DatasetDescriptor(
        dataset_name="LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL080_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1",
        dataset_path=Path("x.csv"),
        filter_name="LIQ_5K_HIGH_ACTIVITY",
        horizon="1h",
        exit_policy_id="TP20308_SL080_FEE0308_TIME_BY_HORIZON",
        target_name="net_profitable_after_exit_policy",
        target_version="v1",
    )


def _synthetic_dataset(n_train: int = 80, n_val: int = 30, n_test: int = 30) -> pd.DataFrame:
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
                    "symbol": "SYM",
                    "event_timestamp": f"2026-06-01T12:{i % 60:02d}:00Z",
                    "filter": "LIQ_5K_HIGH_ACTIVITY",
                    "horizon": "1h",
                    "exit_policy_id": "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
                    "split": split,
                    "label_valid": True,
                    "target_net_profitable_after_exit": label,
                    "sim_net_return": 0.05 if label else -0.02,
                    "feature_a": float(rng.normal()),
                    "feature_b": float(rng.normal()),
                    "feature_c": rng.random() > 0.5,
                }
            )
    return pd.DataFrame(rows)


def _late_positive_dataset() -> pd.DataFrame:
    """Positives clustered at the end of each split (head-only sampling loses them)."""
    rows: list[dict] = []
    for split, n_neg, n_pos in (("train", 120, 15), ("validation", 80, 10), ("test", 60, 8)):
        for i in range(n_neg):
            rows.append(
                {
                    "candidate_id": f"cand_{split}_neg_{i}",
                    "candidate_policy_id": f"cp_{split}_neg_{i}",
                    "target_row_id": f"tr_{split}_neg_{i}",
                    "pair_address": f"pair_{i % 5}",
                    "symbol": "SYM",
                    "event_timestamp": f"2026-06-01T12:{i % 60:02d}:00Z",
                    "filter": "LIQ_5K_HIGH_ACTIVITY",
                    "horizon": "1h",
                    "exit_policy_id": "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
                    "split": split,
                    "label_valid": True,
                    "target_net_profitable_after_exit": 0,
                    "sim_net_return": -0.02,
                    "feature_a": float(i),
                    "feature_b": float(i + 1),
                    "feature_c": False,
                }
            )
        for i in range(n_pos):
            rows.append(
                {
                    "candidate_id": f"cand_{split}_pos_{i}",
                    "candidate_policy_id": f"cp_{split}_pos_{i}",
                    "target_row_id": f"tr_{split}_pos_{i}",
                    "pair_address": f"pair_{i % 5}",
                    "symbol": "SYM",
                    "event_timestamp": f"2026-06-02T12:{i % 60:02d}:00Z",
                    "filter": "LIQ_5K_HIGH_ACTIVITY",
                    "horizon": "1h",
                    "exit_policy_id": "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
                    "split": split,
                    "label_valid": True,
                    "target_net_profitable_after_exit": 1,
                    "sim_net_return": 0.05,
                    "feature_a": float(i + 1000),
                    "feature_b": float(i + 1001),
                    "feature_c": True,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_e4a_predictions(df: pd.DataFrame, model: str) -> pd.DataFrame:
    pred = df[
        [
            "candidate_id",
            "candidate_policy_id",
            "target_row_id",
            "pair_address",
            "event_timestamp",
            "filter",
            "horizon",
            "exit_policy_id",
            "split",
        ]
    ].copy()
    pred[CANONICAL_TARGET] = df[CANONICAL_TARGET].values if CANONICAL_TARGET in df.columns else 0
    pred["predicted_probability"] = np.linspace(0.9, 0.1, len(pred))
    pred["model"] = model
    return pred


class StrictTargetTests(unittest.TestCase):
    def test_strict_coerce_mapping(self) -> None:
        series = pd.Series(["True", "false", "FALSE", 1, 0, True, False])
        out, errors = strict_coerce_binary_target(series)
        self.assertEqual(errors, [])
        self.assertListEqual(out.tolist(), [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    def test_invalid_target_coercion_error(self) -> None:
        series = pd.Series(["maybe", 2, -1])
        _, errors = strict_coerce_binary_target(series)
        self.assertTrue(any("INVALID_TARGET_COERCION_ERROR" in e for e in errors))

    def test_target_sanity_check_valid(self) -> None:
        ok, detail = target_sanity_check(pd.Series([0, 1, 0, 1]))
        self.assertTrue(ok)
        self.assertTrue(detail["valid"])

    def test_invalid_target_values(self) -> None:
        ok, detail = target_sanity_check(pd.Series([0, 1, 2, 0.5]))
        self.assertFalse(ok)
        self.assertEqual(detail["error"], "INVALID_TARGET_VALUES")

    def test_normalize_target_column_strict(self) -> None:
        df = _synthetic_dataset()
        out, audit = normalize_target_column_strict(df, _descriptor())
        self.assertEqual(audit["normalization_status"], "ok")
        self.assertIn(CANONICAL_TARGET, out.columns)


class FeatureAndIdentityTests(unittest.TestCase):
    def test_leakage_exclusion(self) -> None:
        df = _synthetic_dataset()
        out, audit = normalize_target_column_strict(df, _descriptor())
        valid = out.loc[derive_valid_label_mask(out)]
        features, leakage, identity, _ = build_feature_columns(valid)
        self.assertNotIn(CANONICAL_TARGET, features)
        self.assertNotIn("sim_net_return", features)
        self.assertTrue(any("target_row_id" in c for c in identity))

    def test_identity_preservation(self) -> None:
        df = _synthetic_dataset(n_train=5, n_val=3, n_test=3)
        for col in ("candidate_id", "candidate_policy_id", "target_row_id"):
            self.assertIn(col, df.columns)
            self.assertEqual(df[col].nunique(), len(df))

    def test_feature_order_reindex(self) -> None:
        df = _synthetic_dataset()
        features, _, _, _ = build_feature_columns(df)
        matrix = df[features]
        reordered = reindex_features(matrix, features)
        self.assertListEqual(list(reordered.columns), features)

    def test_feature_schema_missing_columns(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            reindex_features(pd.DataFrame({"a": [1]}), ["a", "b"])
        self.assertIn("FEATURE_SCHEMA_MISSING_COLUMNS", str(ctx.exception))

    def test_feature_order_reindex_fixes_column_order(self) -> None:
        df = _synthetic_dataset()
        features, _, _, _ = build_feature_columns(df)
        matrix = df[features].copy()
        shuffled = matrix[["feature_b", "feature_a", "feature_c"]] if "feature_c" in features else matrix
        reordered = reindex_features(shuffled, features)
        self.assertListEqual(list(reordered.columns), features)


class JoinTests(unittest.TestCase):
    def _pred_frames(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df = _synthetic_dataset(n_train=10, n_val=8, n_test=8)
        out, _ = normalize_target_column_strict(df, _descriptor())
        val = out[out["split"] == "validation"].head(8)
        tab = val[["target_row_id", "candidate_policy_id", "candidate_id"]].copy()
        tab["tab_score"] = np.linspace(0.9, 0.1, len(tab))
        xgb = _synthetic_e4a_predictions(val, "XGB").rename(
            columns={"predicted_probability": "predicted_probability_xgb"}
        )
        rf = _synthetic_e4a_predictions(val, "RF").rename(
            columns={"predicted_probability": "predicted_probability_rf"}
        )
        return tab, xgb, rf

    def test_join_one_to_one(self) -> None:
        tab, xgb, rf = self._pred_frames()
        merged, diags = merge_tab_xgb_rf_predictions(tab, xgb, rf)
        self.assertEqual(len(merged), len(tab))
        self.assertEqual(len(diags), 2)

    def test_join_key_null_error(self) -> None:
        frame = pd.DataFrame({"target_row_id": ["a", None, ""]})
        with self.assertRaises(RuntimeError) as ctx:
            sanitize_join_keys(frame, "target_row_id")
        self.assertIn("JOIN_KEY_NULL_ERROR", str(ctx.exception))

    def test_join_key_duplicate_error(self) -> None:
        left = pd.DataFrame({"target_row_id": ["a", "a"], "v": [1, 2]})
        right = pd.DataFrame({"target_row_id": ["a"], "v": [3]})
        with self.assertRaises(RuntimeError) as ctx:
            strict_merge_one_to_one(left, right, "target_row_id")
        self.assertIn("JOIN_KEY_DUPLICATE_ERROR", str(ctx.exception))

    def test_join_empty_overlap_produces_zero_rows(self) -> None:
        tab = pd.DataFrame({"target_row_id": ["x"], "tab_score": [0.5]})
        xgb = pd.DataFrame({"target_row_id": ["y"], "predicted_probability_xgb": [0.5]})
        merged, diags = merge_tab_xgb_rf_predictions(tab, xgb, xgb)
        self.assertEqual(len(merged), 0)
        self.assertEqual(diags[0]["actual_matched_rows"], 0)


class ConsensusTests(unittest.TestCase):
    def test_consensus_tier_assignment(self) -> None:
        self.assertEqual(assign_consensus_tier(True, True, True), "TAB_XGB_RF_ALL3")
        self.assertEqual(assign_consensus_tier(True, False, True), "TAB_RF_ONLY")
        self.assertEqual(assign_consensus_tier(True, True, False), "TAB_XGB_ONLY")
        self.assertEqual(assign_consensus_tier(False, True, True), "XGB_RF_ONLY")
        self.assertEqual(assign_consensus_tier(True, False, False), "TAB_ONLY")
        self.assertEqual(assign_consensus_tier(False, False, False), "NONE")

    def test_consensus_frame(self) -> None:
        tab, xgb, rf = JoinTests()._pred_frames()
        merged, _ = merge_tab_xgb_rf_predictions(tab, xgb, rf)
        consensus = build_consensus_frame(merged, top_pct=50.0)
        self.assertIn("consensus_tier", consensus.columns)


class RankAndContextTests(unittest.TestCase):
    def test_rank_percentile_deterministic(self) -> None:
        scores = np.array([0.1, 0.9, 0.5, 0.7])
        ranks = compute_rank_percentile(scores)
        self.assertEqual(len(ranks), 4)
        self.assertAlmostEqual(float(ranks.max()), 100.0)

    def test_context_hash_deterministic(self) -> None:
        cfg = {"strategy": "stratified_recent", "context_size": 64, "random_state": 42}
        h1 = build_context_construction_strategy_hash(cfg)
        h2 = build_context_construction_strategy_hash(cfg)
        self.assertEqual(h1, h2)

    def test_context_cache_key_includes_fingerprint(self) -> None:
        train_df = _synthetic_dataset(n_train=20, n_val=5, n_test=5)
        h1 = build_train_identity_hash(train_df[train_df["split"] == "train"])
        key = build_context_cache_key(
            strategy_hash="abc",
            dataset_rel_path="data/x.csv",
            dataset_content_hash=None,
            train_row_count=20,
            train_identity_hash=h1,
            feature_order_hash="feat",
            target_name="net_profitable_after_exit_policy",
            target_version="v1",
            filter_name="LIQ_5K_HIGH_ACTIVITY",
            horizon="1h",
            exit_policy_id="TP20308_SL080_FEE0308_TIME_BY_HORIZON",
            random_state=42,
        )
        self.assertTrue(len(key) >= 32)

    def test_context_cache_no_reuse_on_fingerprint_change(self) -> None:
        cache = BoundedContextCache(mode="cpu_only", max_entries=2)
        meta_a = {"train_row_count": 10, "train_identity_hash": "a", "feature_order_hash": "f"}
        meta_b = {"train_row_count": 10, "train_identity_hash": "b", "feature_order_hash": "f"}
        cache.put("key1", {"val_scores": np.array([0.1])}, meta_a)
        self.assertIsNone(cache.get("key1", meta_b))

    def test_context_drift_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "prior.json"
            report.write_text(json.dumps({"context_construction_strategy_hash": "old"}), encoding="utf-8")
            result = check_context_drift("new", prior_reports_glob=[report])
            self.assertEqual(result["status"], "TAB_CONTEXT_DRIFT_WARNING")

    def test_context_baseline_not_found(self) -> None:
        result = check_context_drift("hash", prior_reports_glob=[])
        self.assertEqual(result["status"], "TAB_CONTEXT_BASELINE_NOT_FOUND")


class ConcurrencyAndSafetyTests(unittest.TestCase):
    def test_max_workers_default_is_one(self) -> None:
        self.assertEqual(DEFAULT_MAX_WORKERS, 1)

    def test_smoke_context_size_default(self) -> None:
        self.assertGreaterEqual(SMOKE_DEFAULT_MAX_CONTEXT_SIZE, 50)
        self.assertLessEqual(SMOKE_DEFAULT_MAX_CONTEXT_SIZE, 100)

    def test_bounded_executor_serial(self) -> None:
        calls: list[int] = []

        def worker(job: dict) -> dict:
            calls.append(job["id"])
            return {"id": job["id"]}

        jobs = [{"id": i} for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp:
            audit = E5IncrementalAuditLogger(Path(tmp) / "audit.jsonl")
            results = run_bounded_executor(jobs, worker, max_workers=1, audit=audit)
        self.assertEqual(len(results), 3)
        self.assertEqual(calls, [0, 1, 2])

    @mock.patch("app.training.direct_target_tabicl.os.cpu_count", return_value=16)
    def test_max_workers_not_from_cpu_count(self, _mock_cpu: mock.MagicMock) -> None:
        self.assertEqual(DEFAULT_MAX_WORKERS, 1)

    @mock.patch("app.training.direct_target_tabicl.gc.collect")
    def test_memory_cleanup_calls_gc(self, mock_gc: mock.MagicMock) -> None:
        run_memory_cleanup()
        mock_gc.assert_called()

    @mock.patch("app.training.direct_target_tabicl.gc.collect")
    def test_memory_cleanup_torch_cuda(self, mock_gc: mock.MagicMock) -> None:
        fake_torch = mock.MagicMock()
        fake_torch.cuda.is_available.return_value = True
        fake_torch.cuda.memory_allocated.return_value = 100
        fake_torch.cuda.memory_reserved.return_value = 200
        with mock.patch.dict(sys.modules, {"torch": fake_torch}):
            summary = run_memory_cleanup()
        fake_torch.cuda.empty_cache.assert_called()
        self.assertTrue(summary["gc_collect"])


class DependencyAuditTests(unittest.TestCase):
    def _setup_e4a_root(self, root: Path) -> None:
        for pattern in (
            "predictions/direct_target_predictions_validation_XGB_LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL080_FEE0308_TIME_BY_HORIZON.parquet",
            "predictions/direct_target_predictions_test_XGB_LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL080_FEE0308_TIME_BY_HORIZON.parquet",
            "predictions/direct_target_predictions_validation_RF_LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL080_FEE0308_TIME_BY_HORIZON.parquet",
            "predictions/direct_target_predictions_test_RF_LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL080_FEE0308_TIME_BY_HORIZON.parquet",
            "metrics/direct_target_metrics_XGB_LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL080_FEE0308_TIME_BY_HORIZON.json",
            "metrics/direct_target_metrics_RF_LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL080_FEE0308_TIME_BY_HORIZON.json",
            "policy_evaluation/direct_target_policy_grid_xgb_rf.csv",
            "policy_evaluation/validation_selected_policies_direct_target_xgb_rf_applied_to_test.csv",
            "reports/phase_e4_manifest.json",
        ):
            path = root / pattern
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                path.write_text("{}", encoding="utf-8")
            elif path.suffix == ".csv":
                path.write_text("model,filter\nTAB,LIQ_5K_HIGH_ACTIVITY\n", encoding="utf-8")
            else:
                pd.DataFrame({"x": [1]}).to_parquet(path, index=False)

    def test_dependency_audit_passes_with_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            e3 = root / "e3"
            e4a = root / "e4a"
            e3.mkdir()
            e4a.mkdir()
            _synthetic_dataset().to_csv(
                e3 / "LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL080_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.csv",
                index=False,
            )
            self._setup_e4a_root(e4a)
            audit = run_dependency_audit(
                project_root=root,
                e3_root=e3,
                e4a_root=e4a,
                fail_on_missing_registry=False,
                allow_registry_warnings=True,
            )
            self.assertEqual(audit["status"], "pass")

    def test_dependency_audit_fails_missing_predictions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            e3 = root / "e3"
            e4a = root / "e4a"
            e3.mkdir()
            e4a.mkdir()
            audit = run_dependency_audit(
                project_root=root,
                e3_root=e3,
                e4a_root=e4a,
                fail_on_missing_registry=False,
                allow_registry_warnings=True,
            )
            self.assertEqual(audit["status"], "fail")


class SmokeSamplingTests(unittest.TestCase):
    def test_head_only_sampling_drops_late_validation_positives(self) -> None:
        raw = _late_positive_dataset()
        out, audit = normalize_target_column_strict(raw, _descriptor())
        self.assertEqual(audit["normalization_status"], "ok")
        valid = out.loc[derive_valid_label_mask(out)]
        assert_full_split_positive_minimums(
            valid,
            min_train_positives=10,
            min_validation_positives=3,
            min_test_positives=3,
        )
        head_limited = apply_deterministic_row_limit(valid, SMOKE_DEFAULT_MAX_ROWS)
        val_pos = int((head_limited[head_limited["split"] == "validation"][CANONICAL_TARGET] == 1).sum())
        self.assertLess(val_pos, 3)

    def test_stratified_smoke_sampling_preserves_positives(self) -> None:
        raw = _late_positive_dataset()
        out, audit = normalize_target_column_strict(raw, _descriptor())
        valid = out.loc[derive_valid_label_mask(out)]
        sampled = apply_smoke_stratified_split_sampling(
            valid,
            max_rows=SMOKE_DEFAULT_MAX_ROWS,
            min_train_positives=10,
            min_validation_positives=3,
            min_test_positives=3,
            random_state=42,
        )
        self.assertLessEqual(len(sampled), SMOKE_DEFAULT_MAX_ROWS)
        self.assertGreaterEqual(int((sampled[sampled["split"] == "train"][CANONICAL_TARGET] == 1).sum()), 10)
        self.assertGreaterEqual(
            int((sampled[sampled["split"] == "validation"][CANONICAL_TARGET] == 1).sum()), 3
        )
        self.assertGreaterEqual(int((sampled[sampled["split"] == "test"][CANONICAL_TARGET] == 1).sum()), 3)

    def test_stratified_smoke_sampling_is_deterministic(self) -> None:
        raw = _late_positive_dataset()
        out, _ = normalize_target_column_strict(raw, _descriptor())
        valid = out.loc[derive_valid_label_mask(out)]
        kwargs = dict(
            max_rows=SMOKE_DEFAULT_MAX_ROWS,
            min_train_positives=10,
            min_validation_positives=3,
            min_test_positives=3,
            random_state=99,
        )
        a = apply_smoke_stratified_split_sampling(valid, **kwargs)
        b = apply_smoke_stratified_split_sampling(valid, **kwargs)
        pd.testing.assert_frame_equal(
            a.sort_values("target_row_id").reset_index(drop=True),
            b.sort_values("target_row_id").reset_index(drop=True),
        )

    def test_min_validation_only_when_full_split_lacks_positives(self) -> None:
        raw = _late_positive_dataset()
        out, _ = normalize_target_column_strict(raw, _descriptor())
        valid = out.loc[derive_valid_label_mask(out)]
        broken = valid.copy()
        broken.loc[broken["split"] == "validation", CANONICAL_TARGET] = 0
        with self.assertRaises(RuntimeError) as ctx:
            assert_full_split_positive_minimums(
                broken,
                min_train_positives=10,
                min_validation_positives=3,
                min_test_positives=3,
            )
        self.assertEqual(str(ctx.exception), "MIN_VALIDATION_POSITIVES_NOT_MET")


class AuditJsonlTests(unittest.TestCase):
    def test_audit_jsonl_flush(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.jsonl"
            logger = E5IncrementalAuditLogger(path)
            logger.log("test_event", status="ok")
            with path.open(encoding="utf-8") as handle:
                lines = handle.readlines()
            self.assertEqual(len(lines), 1)
            payload = json.loads(lines[0])
            self.assertEqual(payload["event_type"], "test_event")


class IntegrationSmokeTests(unittest.TestCase):
    def test_smoke_run_with_skip_tab(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            e3 = root / "e3"
            e4a = root / "e4a"
            out = root / "e5_out"
            e3.mkdir()
            e4a.mkdir()
            df = _synthetic_dataset()
            ds_name = "LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL080_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.csv"
            df.to_csv(e3 / ds_name, index=False)
            DependencyAuditTests()._setup_e4a_root(e4a)
            val_df = df[df["split"] == "validation"]
            test_df = df[df["split"] == "test"]
            for model in ("XGB", "RF"):
                for split_name, split_df in (("validation", val_df), ("test", test_df)):
                    combo = f"{model}_LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL080_FEE0308_TIME_BY_HORIZON"
                    pred = _synthetic_e4a_predictions(split_df, model)
                    pred[CANONICAL_TARGET] = split_df["target_net_profitable_after_exit"].values
                    path = e4a / "predictions" / f"direct_target_predictions_{split_name}_{combo}.parquet"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pred.to_parquet(path, index=False)

            config = EvalConfig(
                input_dir=e3,
                output_dir=out,
                e4a_root=e4a,
                smoke=True,
                skip_tab_inference=True,
                register_artifacts=False,
                fail_on_missing_e4a_registry=False,
                allow_registry_warnings=True,
                max_rows=SMOKE_DEFAULT_MAX_ROWS,
                max_context_size=SMOKE_DEFAULT_MAX_CONTEXT_SIZE,
            )
            result = run_evaluation(config)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result.get("successful_jobs"), 1)
            self.assertEqual(result.get("processed_jobs"), 1)
            self.assertEqual(result.get("skipped_jobs"), 0)
            self.assertEqual(result.get("failed_jobs"), 0)
            manifest = out / "reports" / "direct_target_tabicl_manifest.json"
            self.assertTrue(manifest.is_file())
            with manifest.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            self.assertEqual(payload.get("datasets_requested"), 1)
            self.assertEqual(payload.get("successful_jobs"), 1)
            self.assertEqual(payload.get("datasets_skipped"), 0)


def _synthetic_consensus_trades(*, n_val: int = 40, n_test: int = 40) -> pd.DataFrame:
    rows: list[dict] = []
    for split, n, base_return in (("validation", n_val, 0.08), ("test", n_test, 0.05)):
        for i in range(n):
            pair = f"pair_{i % 12}"
            rows.append(
                {
                    "filter": "LIQ_5K_HIGH_ACTIVITY",
                    "horizon": "1h",
                    "exit_policy_id": "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
                    "split": split,
                    "top_pct": 1.0,
                    "consensus_tier": "TAB_XGB_RF_ALL3",
                    "tab_score": 1.0 - i * 0.01,
                    CANONICAL_TARGET: 1 if i % 3 == 0 else 0,
                    "sim_net_return": base_return,
                    "pair_address": pair,
                }
            )
    return pd.DataFrame(rows)


class E5CValidationReportingTests(unittest.TestCase):
    def test_validation_selected_non_empty_from_tier_summaries(self) -> None:
        df = _synthetic_consensus_trades()
        applied = build_validation_selected_consensus_applied_to_test(df)
        self.assertFalse(applied.empty)
        self.assertIn("selection_status", applied.columns)
        self.assertIn("concentration_status", applied.columns)
        self.assertIn("validation_total_net_return", applied.columns)
        self.assertIn("test_total_net_return", applied.columns)

    def test_validation_selection_not_test_selection(self) -> None:
        val_only = _synthetic_consensus_trades(n_val=40, n_test=0)
        applied = build_validation_selected_consensus_applied_to_test(val_only)
        self.assertFalse(applied.empty)
        self.assertTrue((applied["selection_status"] == "NO_TEST_MATCH").all())

    def test_no_test_match_explicit(self) -> None:
        df = _synthetic_consensus_trades(n_test=0)
        applied = build_validation_selected_consensus_applied_to_test(df)
        self.assertIn("NO_TEST_MATCH", set(applied["selection_status"]))

    def test_concentration_blocked_label(self) -> None:
        rows: list[dict] = []
        for split in ("validation", "test"):
            for i in range(10):
                rows.append(
                    {
                        "filter": "NO_WHALE_FILTER",
                        "horizon": "4h",
                        "exit_policy_id": "TP20308_SL075_FEE0308_TIME_BY_HORIZON",
                        "split": split,
                        "top_pct": 0.5,
                        "consensus_tier": "TAB_RF_ONLY",
                        "tab_score": 0.9 - i * 0.01,
                        CANONICAL_TARGET: 1,
                        "sim_net_return": 0.2,
                        "pair_address": "same_pair",
                    }
                )
        applied = build_validation_selected_consensus_applied_to_test(pd.DataFrame(rows))
        self.assertFalse(applied.empty)
        self.assertIn(
            "POSITIVE_BUT_CONCENTRATION_BLOCKED",
            set(applied["selection_status"]),
        )

    def test_test_negative_label(self) -> None:
        rows: list[dict] = []
        for split, ret in (("validation", 0.15), ("test", -0.05)):
            for i in range(12):
                rows.append(
                    {
                        "filter": "LIQ_5K_HIGH_ACTIVITY",
                        "horizon": "1h",
                        "exit_policy_id": "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
                        "split": split,
                        "top_pct": 1.0,
                        "consensus_tier": "TAB_XGB_RF_ALL3",
                        "tab_score": 0.9 - i * 0.01,
                        CANONICAL_TARGET: 1,
                        "sim_net_return": ret,
                        "pair_address": f"pair_{i}",
                    }
                )
        applied = build_validation_selected_consensus_applied_to_test(pd.DataFrame(rows))
        self.assertIn("TEST_NEGATIVE", set(applied["selection_status"]))

    def test_e5c_decision_summary_written(self) -> None:
        applied = build_validation_selected_consensus_applied_to_test(_synthetic_consensus_trades())
        summary = build_e5c_decision_summary(applied)
        self.assertIn("TAB_XGB_RF_ALL3", summary)
        self.assertIn("E6:", summary)

    def test_regenerate_from_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            consensus_dir = root / "consensus"
            policy_dir = root / "policy_evaluation"
            reports_dir = root / "reports"
            consensus_dir.mkdir(parents=True)
            policy_dir.mkdir(parents=True)
            reports_dir.mkdir(parents=True)
            trades = _synthetic_consensus_trades()
            trades.to_csv(consensus_dir / "direct_target_selected_trades_by_tier.csv", index=False)
            result = regenerate_e5c_reporting_from_artifacts(root)
            self.assertGreater(result["validation_selected_rows"], 0)
            self.assertTrue(Path(result["validation_selected_path"]).is_file())
            self.assertTrue(Path(result["decision_summary_path"]).is_file())


class ArtifactIdPortabilityTests(unittest.TestCase):
    def test_relative_path_not_absolute_in_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp)
            artifact = project / "data/training/manual_verified_results/phase_e5_x/audit/x.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text("{}", encoding="utf-8")
            rel = artifact.resolve().relative_to(project.resolve()).as_posix()
            self.assertFalse(rel.startswith("E:"))
            self.assertFalse("\\" in rel or rel.startswith("/"))


if __name__ == "__main__":
    unittest.main()
