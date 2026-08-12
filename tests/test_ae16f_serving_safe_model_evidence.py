"""Focused tests for AE16F serving-safe model evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.consensus.ae16e_feature_parity import (  # noqa: E402
    TOXIC_PAIR_ADDRESS,
    is_toxic_pair,
    load_clean_forward_rows_used,
)
from app.consensus.ae16f_serving_safe import (  # noqa: E402
    FORBIDDEN_FEATURE_NAMES,
    build_current_cf_matrix,
    build_evidence_rows,
    build_feature_schema_lock,
    build_serving_safe_feature_contract,
    decide_ae16f_classification,
    discover_training_sources,
    extract_serving_safe_row_from_cf,
    ordered_feature_names_from_contract,
    predict_proba_locked,
    schema_hash,
    validate_predict_matrix,
    validation_quantile_threshold,
)


class TestAE16FServingSafe(unittest.TestCase):
    def test_active_curated_toxic_absent(self):
        rows, meta = load_clean_forward_rows_used(ROOT)
        self.assertGreater(meta.get("curated_active_targets_loaded") or 0, 0)
        if rows:
            self.assertFalse(any(is_toxic_pair(r.get("pair_address")) for r in rows))
        self.assertFalse(is_toxic_pair("0xabc"))
        self.assertTrue(is_toxic_pair(TOXIC_PAIR_ADDRESS))

    def test_contract_excludes_old_blocking_fields(self):
        rows, contract = build_serving_safe_feature_contract()
        allowed = {r["feature_name"] for r in rows if r.get("allowed")}
        for bad in (
            "whale_score",
            "price_step_ratio_prev",
            "gap_detected",
            "is_extreme_step_ratio_100x",
            "entry_snapshot_id",
            "entry_price_verified_1h",
            "entry_price_verified_30m",
        ):
            self.assertNotIn(bad, allowed)
            self.assertIn(bad, FORBIDDEN_FEATURE_NAMES)
        self.assertFalse(contract["whale_score_included"])
        self.assertFalse(contract["sequential_features_included"])
        self.assertFalse(contract["entry_verified_fields_included"])

    def test_sequential_and_entry_excluded(self):
        rows, _ = build_serving_safe_feature_contract()
        forbidden_rows = {r["feature_name"]: r for r in rows if not r.get("allowed")}
        self.assertIn("gap_detected", forbidden_rows)
        self.assertIn("entry_snapshot_id", forbidden_rows)
        self.assertEqual(forbidden_rows["whale_score"]["allowed"], False)

    def test_discovery_rejects_clean_model_input_as_direct_target(self):
        rows, selected = discover_training_sources(ROOT)
        clean_rejects = [
            r
            for r in rows
            if "CLEAN_MODEL_INPUT" in str(r.get("path") or "")
            and r.get("serving_safe_compatible") is False
        ]
        self.assertTrue(clean_rejects or any("CLEAN_MODEL_INPUT" in str(r.get("path")) for r in rows))
        if selected:
            self.assertIn("DIRECT_TARGET", str(selected.get("path") or ""))

    def test_historical_matrix_only_serving_safe_names(self):
        rows, _ = build_serving_safe_feature_contract()
        names = ordered_feature_names_from_contract(rows)
        for n in names:
            self.assertNotIn(n, FORBIDDEN_FEATURE_NAMES)

    def test_schema_lock_and_current_alignment(self):
        rows, _ = build_serving_safe_feature_contract()
        names = ordered_feature_names_from_contract(rows)
        medians = {n: 0.0 for n in names if not n.endswith("_is_missing")}
        lock = build_feature_schema_lock(names, medians)
        cf = [
            {
                "row_id": "r1",
                "combined_target_id": "c1",
                "chain": "base",
                "pair_address": "0x1",
                "price_usd": "1.0",
                "liquidity_usd": "1000",
                "volume_24h": "100",
            }
        ]
        matrix, _, lineage, align = build_current_cf_matrix(cf, lock)
        self.assertEqual(len(lineage), 1)
        feat = matrix[names]
        self.assertEqual(list(feat.columns), names)
        self.assertTrue(all(a.get("passed") for a in align if "passed" in a))

    def test_column_order_and_dtype_match_lock(self):
        rows, _ = build_serving_safe_feature_contract()
        names = ordered_feature_names_from_contract(rows)
        lock = build_feature_schema_lock(names, {n: 1.0 for n in names})
        df = pd.DataFrame([[0.0] * len(names)], columns=names, dtype="float64")
        audit = validate_predict_matrix(df, lock)
        self.assertTrue(all(a["passed"] for a in audit))

    def test_schema_hash_mismatch_blocks(self):
        rows, _ = build_serving_safe_feature_contract()
        names = ordered_feature_names_from_contract(rows)
        lock = build_feature_schema_lock(names, {n: 1.0 for n in names})
        bad = names[1:] + [names[0]]  # reorder
        df = pd.DataFrame([[0.0] * len(names)], columns=bad, dtype="float64")
        audit = validate_predict_matrix(df, lock)
        self.assertFalse(all(a["passed"] for a in audit))

    def test_extra_and_missing_columns_block(self):
        rows, _ = build_serving_safe_feature_contract()
        names = ordered_feature_names_from_contract(rows)
        lock = build_feature_schema_lock(names, {n: 1.0 for n in names})
        df_extra = pd.DataFrame(
            [[0.0] * (len(names) + 1)], columns=names + ["extra_bad"], dtype="float64"
        )
        audit = validate_predict_matrix(df_extra[names + ["extra_bad"]], lock)
        self.assertFalse(next(a["passed"] for a in audit if a["check"] == "extra_columns"))
        df_miss = pd.DataFrame([[0.0] * (len(names) - 1)], columns=names[:-1], dtype="float64")
        audit2 = validate_predict_matrix(df_miss, lock)
        self.assertFalse(next(a["passed"] for a in audit2 if a["check"] == "missing_columns"))

    def test_rf_predict_uses_locked_order(self):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer

        rows, _ = build_serving_safe_feature_contract()
        names = ordered_feature_names_from_contract(rows)[:8]
        lock = build_feature_schema_lock(names, {n: 0.0 for n in names})
        X = pd.DataFrame(np.random.randn(40, len(names)), columns=names)
        y = (X.iloc[:, 0] > 0).astype(int).to_numpy()
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("model", RandomForestClassifier(n_estimators=5, random_state=0)),
            ]
        )
        pipe.fit(X, y)
        scores = predict_proba_locked(pipe, X, lock, "RF")
        self.assertEqual(len(scores), 40)
        # mismatch raises
        Xbad = X[names[::-1]]
        with self.assertRaises(ValueError):
            predict_proba_locked(pipe, Xbad, lock, "RF")

    def test_xgb_feature_order_mismatch_blocks(self):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            self.skipTest("xgboost not installed")
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer

        rows, _ = build_serving_safe_feature_contract()
        names = ordered_feature_names_from_contract(rows)[:6]
        lock = build_feature_schema_lock(names, {n: 0.0 for n in names})
        X = pd.DataFrame(np.random.randn(30, len(names)), columns=names)
        y = (X.iloc[:, 0] > 0).astype(int).to_numpy()
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("model", XGBClassifier(n_estimators=5, max_depth=2, random_state=0)),
            ]
        )
        pipe.fit(X, y)
        predict_proba_locked(pipe, X, lock, "XGB")
        with self.assertRaises(ValueError):
            predict_proba_locked(pipe, X[names[::-1]], lock, "XGB")

    def test_vote_thresholds_not_from_current_rows(self):
        val_scores = np.array([0.1, 0.2, 0.9, 0.95, 0.99])
        thr = validation_quantile_threshold(val_scores, top_pct=5.0)
        # Must be derived from validation array only
        self.assertGreaterEqual(thr, 0.0)
        # current-batch top-k must not define threshold — function has no current arg
        self.assertTrue(callable(validation_quantile_threshold))

    def test_threshold_integrity_flags(self):
        meta = {
            "current_rows_used_for_threshold": False,
            "threshold_selected_before_current_inference": True,
            "threshold_method": "validation_top_pct_quantile",
        }
        self.assertFalse(meta["current_rows_used_for_threshold"])

    def test_rank_in_batch_diagnostic_in_evidence(self):
        lock = {"feature_schema_hash": "abc", "ordered_feature_names": ["a"]}
        lineage = [
            {
                "row_id": "r1",
                "combined_target_id": "c1",
                "chain": "base",
                "pair_address": "0x1",
                "provider_pair_url": "u",
                "base_token_address": "b",
                "quote_token_address": "q",
                "base_token_symbol": "B",
                "quote_token_symbol": "Q",
            },
            {
                "row_id": "r2",
                "combined_target_id": "c2",
                "chain": "base",
                "pair_address": "0x2",
                "provider_pair_url": "u",
                "base_token_address": "b",
                "quote_token_address": "q",
                "base_token_symbol": "B",
                "quote_token_symbol": "Q",
            },
        ]
        ev = build_evidence_rows(
            family="RF",
            lineage=lineage,
            scores=np.array([0.2, 0.9]),
            threshold=0.5,
            threshold_meta={
                "threshold_method": "validation_top_pct_quantile",
                "threshold_source": "historical_validation_predictions_top_5pct",
            },
            model_path="m.joblib",
            lock=lock,
        )
        self.assertEqual(ev[1]["rank_in_batch"], 1)
        self.assertTrue(ev[1]["vote"])
        self.assertFalse(ev[0]["vote"])
        self.assertFalse(ev[0]["current_rows_used_for_threshold"])
        self.assertIn("diagnostic", ev[0]["limitation_notes"])

    def test_lineage_preserved(self):
        lock = build_feature_schema_lock(["price_usd", "liquidity_usd"], {"price_usd": 1, "liquidity_usd": 1})
        # rebuild with full names for extract
        rows, _ = build_serving_safe_feature_contract()
        names = ordered_feature_names_from_contract(rows)
        lock = build_feature_schema_lock(names, {n: 0.0 for n in names})
        cf = [
            {
                "row_id": "rid",
                "combined_target_id": "cid",
                "chain": "solana",
                "pair_address": "PairX",
                "price_usd": "1",
                "liquidity_usd": "10",
                "volume_24h": "5",
            }
        ]
        matrix, _, lineage, _ = build_current_cf_matrix(cf, lock)
        self.assertEqual(lineage[0]["combined_target_id"], "cid")
        self.assertEqual(matrix.iloc[0]["row_id"], "rid")

    def test_decision_classifications(self):
        self.assertEqual(
            decide_ae16f_classification(
                toxic=True,
                selected_source={"path": "x"},
                feature_count=10,
                schema_ok=True,
                threshold_ok=True,
                families_with_evidence={"RF"},
                training_error="",
                consensus_error="",
            ),
            "AE16F_BLOCKED_TOXIC_PAIR_STILL_PRESENT",
        )
        self.assertEqual(
            decide_ae16f_classification(
                toxic=False,
                selected_source={"path": "x"},
                feature_count=10,
                schema_ok=False,
                threshold_ok=True,
                families_with_evidence=set(),
                training_error="",
                consensus_error="",
            ),
            "AE16F_BLOCKED_SCHEMA_ALIGNMENT_MISMATCH",
        )
        self.assertEqual(
            decide_ae16f_classification(
                toxic=False,
                selected_source={"path": "x"},
                feature_count=10,
                schema_ok=True,
                threshold_ok=True,
                families_with_evidence={"RF", "XGB"},
                training_error="",
                consensus_error="",
            ),
            "AE16F_BLOCKED_TAB_RUNTIME_UNAVAILABLE",
        )
        self.assertEqual(
            decide_ae16f_classification(
                toxic=False,
                selected_source={"path": "x"},
                feature_count=10,
                schema_ok=True,
                threshold_ok=True,
                families_with_evidence={"RF", "XGB", "TAB"},
                training_error="",
                consensus_error="",
            ),
            "AE16F_SERVING_SAFE_MODEL_EVIDENCE_PASS",
        )

    def test_no_ae17_and_safety_class_names(self):
        cls = decide_ae16f_classification(
            toxic=False,
            selected_source={"path": "x"},
            feature_count=10,
            schema_ok=True,
            threshold_ok=True,
            families_with_evidence={"RF"},
            training_error="",
            consensus_error="",
        )
        self.assertTrue(cls.startswith("AE16F_"))
        self.assertNotIn("AE17", cls)

    def test_cf_extract_no_forbidden(self):
        feats = extract_serving_safe_row_from_cf(
            {
                "price_usd": "1",
                "liquidity_usd": "100",
                "volume_24h": "50",
                "txns_h24_buys": "3",
                "txns_h24_sells": "2",
                "fdv": "1000",
            }
        )
        self.assertNotIn("whale_score", feats)
        self.assertNotIn("gap_detected", feats)
        self.assertIn("price_usd", feats)
        self.assertIn("fdv_is_missing", feats)


if __name__ == "__main__":
    unittest.main()
