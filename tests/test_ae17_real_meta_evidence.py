"""Tests for AE17 durable real-meta evidence runner."""
from __future__ import annotations

import csv
import importlib.util
import inspect
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_runner():
    path = ROOT / "scripts" / "run_ae17_real_meta_evidence.py"
    spec = importlib.util.spec_from_file_location("run_ae17_real_meta_evidence", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


MOD = _load_runner()

REQUIRED_COLS = list(MOD.REQUIRED_SOURCE_COLUMNS)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else REQUIRED_COLS
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _base_row(**overrides) -> dict:
    row = {
        "target_row_id": "t1",
        "candidate_id": "c1",
        "candidate_policy_id": "p1",
        "pair_address": "0xpairA",
        "event_timestamp": "2026-06-22T00:00:00+00:00",
        "filter": "LIQ_5K_HIGH_ACTIVITY",
        "horizon": "1h",
        "exit_policy_id": "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
        "split": "validation",
        "target_net_profitable": "0",
        "sim_net_return": "-0.03",
        "tab_score": "0.5",
        "predicted_probability_xgb": "0.4",
        "predicted_probability_rf": "0.3",
        "in_tab": "False",
        "in_xgb": "False",
        "in_rf": "False",
        "vote_count": "0",
        "consensus_tier": "NONE",
    }
    row.update(overrides)
    return row


class TestAE17SourceDiscovery(unittest.TestCase):
    def test_empty_source_glob_controlled_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "out"
            result = MOD.run_ae17_real_meta_evidence(
                root,
                source_glob="does_not_exist_*/nope.csv",
                output_root=out,
            )
            self.assertEqual(result["classification"], MOD.CLASSIFICATION_SOURCE_NOT_FOUND)
            gate = json.loads((out / "reports" / "ae17_real_meta_decision_gate.json").read_text(encoding="utf-8"))
            self.assertEqual(gate["classification"], MOD.CLASSIFICATION_SOURCE_NOT_FOUND)
            self.assertEqual(gate["stage_decision"], "AE17_NOT_CLOSED")
            self.assertEqual(gate["rows_processed"], 0)
            self.assertTrue((out / "reports" / "ae17_real_meta_manifest.json").exists())
            self.assertTrue((out / "reports" / "ae17_real_meta_summary_for_upload.txt").exists())
            self.assertTrue((out / "audits" / "ae17_real_meta_source_file_audit.csv").exists())

    def test_missing_column_file_skipped_and_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "data" / "src"
            bad = src / "bad.csv"
            good = src / "good.csv"
            _write_csv(bad, [{"foo": "1", "bar": "2"}], ["foo", "bar"])
            _write_csv(good, [_base_row()])
            out = root / "out"
            result = MOD.run_ae17_real_meta_evidence(
                root,
                source_glob="data/src/*.csv",
                output_root=out,
            )
            audit_path = out / "audits" / "ae17_real_meta_source_file_audit.csv"
            with audit_path.open(encoding="utf-8") as f:
                audits = list(csv.DictReader(f))
            bad_audit = next(a for a in audits if a["path"].endswith("bad.csv"))
            self.assertEqual(bad_audit["skipped_missing_columns"], "True")
            self.assertIn("target_row_id", bad_audit["missing_required_columns"])
            self.assertEqual(result["classification"], MOD.CLASSIFICATION_PASS_WARNINGS)
            self.assertEqual(result["rows_processed"], 1)

    def test_all_files_unusable_blocker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "data" / "src"
            _write_csv(src / "bad.csv", [{"foo": "1"}], ["foo"])
            out = root / "out"
            result = MOD.run_ae17_real_meta_evidence(
                root,
                source_glob="data/src/*.csv",
                output_root=out,
            )
            self.assertEqual(result["classification"], MOD.CLASSIFICATION_NO_USABLE)

    def test_usable_plus_bad_continues_with_warnings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "data" / "src"
            _write_csv(src / "bad.csv", [{"foo": "1"}], ["foo"])
            _write_csv(
                src / "good.csv",
                [
                    _base_row(
                        in_tab="True",
                        in_xgb="True",
                        in_rf="True",
                        consensus_tier="TAB_XGB_RF_ALL3",
                    )
                ],
            )
            out = root / "out"
            result = MOD.run_ae17_real_meta_evidence(
                root,
                source_glob="data/src/*.csv",
                output_root=out,
            )
            self.assertEqual(result["classification"], MOD.CLASSIFICATION_PASS_WARNINGS)
            self.assertGreaterEqual(result["rows_processed"], 1)


class TestAE17StreamingAndSize(unittest.TestCase):
    def test_no_pandas_read_csv_without_chunksize_on_sources(self):
        src = inspect.getsource(MOD)
        # Forbidden pattern for source evidence loading.
        self.assertNotIn("pd.read_csv(", src)
        self.assertNotIn("pandas.read_csv(", src)
        self.assertIn("csv.DictReader", src)

    def test_streaming_uses_dictreader(self):
        self.assertTrue(callable(MOD.iter_csv_rows_streaming))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.csv"
            _write_csv(path, [_base_row(), _base_row(target_row_id="t2")])
            rows = list(MOD.iter_csv_rows_streaming(path))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0][0], 0)
            self.assertEqual(rows[1][1]["target_row_id"], "t2")

    def test_large_file_skipped_unless_include_large(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "data" / "src"
            large = src / "large.csv"
            small = src / "small.csv"
            _write_csv(large, [_base_row(pair_address="0xlarge")])
            _write_csv(small, [_base_row(pair_address="0xsmall")])
            # Pretend large file exceeds limit via monkeypatched stat size through audit.
            real_audit = MOD.audit_source_file

            def wrapped(path, **kwargs):
                rec = real_audit(path, **kwargs)
                if path.name == "large.csv":
                    rec["size_bytes"] = kwargs["max_source_bytes"] + 1
                    rec["size_gb"] = rec["size_bytes"] / (1024**3)
                    if not kwargs.get("include_large"):
                        rec["used_in_run"] = False
                        rec["skipped_large_file"] = True
                        rec["error_message"] = "too large"
                return rec

            out = root / "out"
            with mock.patch.object(MOD, "audit_source_file", side_effect=wrapped):
                result = MOD.run_ae17_real_meta_evidence(
                    root,
                    source_glob="data/src/*.csv",
                    output_root=out,
                    max_source_bytes=10_000_000,
                    include_large=False,
                )
            audit_path = out / "audits" / "ae17_real_meta_source_file_audit.csv"
            with audit_path.open(encoding="utf-8") as f:
                audits = list(csv.DictReader(f))
            large_audit = next(a for a in audits if a["path"].endswith("large.csv"))
            self.assertEqual(large_audit["skipped_large_file"], "True")
            self.assertEqual(result["classification"], MOD.CLASSIFICATION_PASS_WARNINGS)
            self.assertEqual(result["rows_processed"], 1)

            out2 = root / "out2"
            with mock.patch.object(MOD, "audit_source_file", side_effect=wrapped):
                result2 = MOD.run_ae17_real_meta_evidence(
                    root,
                    source_glob="data/src/*.csv",
                    output_root=out2,
                    max_source_bytes=10_000_000,
                    include_large=True,
                )
            self.assertEqual(result2["rows_processed"], 2)


class TestAE17BooleanAndNumeric(unittest.TestCase):
    def setUp(self):
        MOD._reset_bool_stats()

    def test_boolean_normalization_matrix(self):
        cases = [
            (True, True),
            (False, False),
            ("True", True),
            ("False", False),
            ("true", True),
            ("false", False),
            ("TRUE", True),
            ("FALSE", False),
            ("1", True),
            ("0", False),
            (1, True),
            (0, False),
            ("yes", True),
            ("no", False),
            ("y", True),
            ("n", False),
            ("", False),
            (None, False),
            ("null", False),
            ("None", False),
            ("nan", False),
            (float("nan"), False),
        ]
        for value, expected in cases:
            self.assertEqual(MOD.normalize_bool_vote(value), expected, msg=repr(value))

    def test_unrecognized_bool_returns_false_and_counts(self):
        MOD._reset_bool_stats()
        self.assertFalse(MOD.normalize_bool_vote("maybe"))
        self.assertGreaterEqual(MOD._BOOL_STATS["boolean_unrecognized_values_count"], 1)

    def test_missing_model_scores_remain_null(self):
        feature, output = MOD.process_source_row(
            _base_row(
                tab_score="",
                predicted_probability_xgb="",
                predicted_probability_rf="nan",
                in_tab="",
                in_xgb=None,
                in_rf="nan",
            ),
            source_file="t.csv",
            source_row_index=0,
            row_seq=0,
        )
        self.assertIsNone(feature["tab_score"])
        self.assertIsNone(feature["xgb_score"])
        self.assertIsNone(feature["rf_score"])
        self.assertNotEqual(feature["tab_score"], 0)
        self.assertNotEqual(feature["tab_score"], 0.0)
        self.assertEqual(feature["attached_model_count"], 0)
        self.assertEqual(feature["scoring_tier"], "MODEL_EVIDENCE_UNAVAILABLE")
        self.assertIsNone(feature["meta_score"])
        self.assertEqual(feature["meta_decision"], "META_UNAVAILABLE")
        self.assertFalse(feature["tab_vote"])
        self.assertFalse(feature["xgb_vote"])
        self.assertFalse(feature["rf_vote"])

    def test_attached_model_count_uses_score_presence_not_votes(self):
        feature, _ = MOD.process_source_row(
            _base_row(
                tab_score="0.9",
                predicted_probability_xgb="0.8",
                predicted_probability_rf="",
                in_tab="False",
                in_xgb="False",
                in_rf="True",
            ),
            source_file="t.csv",
            source_row_index=0,
            row_seq=0,
        )
        self.assertEqual(feature["attached_model_count"], 2)
        self.assertEqual(feature["model_vote_count"], 1)
        self.assertEqual(feature["partial_evidence_status"], "TWO_MODELS_ATTACHED")


class TestAE17ScoringAndClamping(unittest.TestCase):
    def test_baseline_tier_mappings_remain_separate(self):
        cases = [
            (True, True, True, 3, "TAB_XGB_RF_ALL3", 0.90, "META_STRONG_WATCH"),
            (True, False, True, 3, "TAB_RF_ONLY", 0.75, "META_SECONDARY_WATCH"),
            (True, True, False, 3, "TAB_XGB_ONLY", 0.45, "META_RESEARCH_ONLY"),
            (False, True, True, 3, "RF_XGB_ONLY", 0.40, "META_RESEARCH_ONLY"),
            (True, False, False, 3, "SINGLE_MODEL_ONLY", 0.25, "META_LOW_CONFIDENCE"),
            (False, False, False, 3, "REJECT", 0.0, "META_REJECT"),
            (False, False, False, 0, "MODEL_EVIDENCE_UNAVAILABLE", None, "META_UNAVAILABLE"),
        ]
        for tab, xgb, rf, attached, tier, score, decision in cases:
            out = MOD.compute_baseline_tier_score(
                tab_vote=tab,
                xgb_vote=xgb,
                rf_vote=rf,
                attached_model_count=attached,
            )
            self.assertEqual(out["scoring_tier"], tier)
            self.assertEqual(out["baseline_tier_decision"], decision)
            if score is None:
                self.assertIsNone(out["baseline_tier_score"])
            else:
                self.assertAlmostEqual(out["baseline_tier_score"], score)

    def test_hard_clamping_inside_combinator(self):
        high = MOD.compute_ae17_real_meta_score(
            tab_vote=True,
            xgb_vote=True,
            rf_vote=True,
            rf_score=0.9,
            xgb_score=0.9,
            tab_score=0.9,
            base_score_override=1.7,
        )
        self.assertEqual(high["meta_score"], 1.0)
        self.assertTrue(high["score_clamped"])
        self.assertAlmostEqual(high["pre_clamp_meta_score"], 1.7)

        low = MOD.compute_ae17_real_meta_score(
            tab_vote=False,
            xgb_vote=False,
            rf_vote=False,
            rf_score=0.4,
            xgb_score=0.4,
            tab_score=0.4,
            base_score_override=-0.4,
        )
        self.assertEqual(low["meta_score"], 0.0)
        self.assertTrue(low["score_clamped"])
        self.assertAlmostEqual(low["pre_clamp_meta_score"], -0.4)

        clamped, was, reason = MOD.clamp_meta_score(1.5)
        self.assertEqual(clamped, 1.0)
        self.assertTrue(was)
        clamped2, was2, _ = MOD.clamp_meta_score(-2.0)
        self.assertEqual(clamped2, 0.0)
        self.assertTrue(was2)


class TestAE17ExplicitMetaCombination(unittest.TestCase):
    def test_weighted_model_score_excludes_nulls_and_uses_active_weights(self):
        out = MOD.compute_weighted_model_score(
            rf_score=1.0,
            xgb_score=None,
            tab_score=0.0,
        )
        # weights: rf 0.30 + tab 0.35 = 0.65; numerator 1.0*0.30 + 0.0*0.35 = 0.30
        self.assertAlmostEqual(out["active_model_weight_sum"], 0.65)
        self.assertEqual(out["active_model_score_count"], 2)
        self.assertAlmostEqual(out["weighted_model_score"], 0.30 / 0.65)
        self.assertIsNone(out["xgb_score_component"])
        self.assertEqual(out["weighted_model_score_missing_reason"], "")

    def test_all_missing_scores_yield_null_meta(self):
        out = MOD.compute_ae17_explicit_meta_combination(
            rf_score=None,
            xgb_score=None,
            tab_score=None,
            tab_vote=False,
            xgb_vote=False,
            rf_vote=False,
        )
        self.assertIsNone(out["weighted_model_score"])
        self.assertIsNone(out["meta_score"])
        self.assertEqual(out["meta_decision"], "META_UNAVAILABLE")
        self.assertEqual(
            out["weighted_model_score_missing_reason"], "NO_VALID_MODEL_SCORES_ATTACHED"
        )

    def test_same_tier_different_numeric_scores_differ(self):
        low = MOD.compute_ae17_explicit_meta_combination(
            rf_score=0.05,
            xgb_score=0.05,
            tab_score=0.05,
            tab_vote=True,
            xgb_vote=True,
            rf_vote=True,
        )
        high = MOD.compute_ae17_explicit_meta_combination(
            rf_score=0.99,
            xgb_score=0.99,
            tab_score=0.99,
            tab_vote=True,
            xgb_vote=True,
            rf_vote=True,
        )
        self.assertEqual(low["scoring_tier"], high["scoring_tier"])
        self.assertEqual(low["scoring_tier"], "TAB_XGB_RF_ALL3")
        self.assertNotAlmostEqual(low["meta_score"], high["meta_score"])
        self.assertEqual(low["baseline_tier_score"], high["baseline_tier_score"])
        self.assertNotEqual(low["meta_score"], low["baseline_tier_score"])
        self.assertFalse(low["tier_only_scoring"])
        self.assertEqual(low["meta_layer_type"], "NON_LEARNED_EXPLICIT_META_COMBINATION")

    def test_consensus_tier_alone_not_sufficient(self):
        sens = MOD.prove_synthetic_same_tier_score_sensitivity()
        self.assertTrue(sens["synthetic_same_tier_score_sensitivity_pass"])
        self.assertTrue(sens["same_tier"])
        self.assertTrue(sens["different_from_baseline"])

    def test_formula_components_present(self):
        out = MOD.compute_ae17_explicit_meta_combination(
            rf_score=0.8,
            xgb_score=0.7,
            tab_score=0.6,
            tab_vote=True,
            xgb_vote=False,
            rf_vote=True,
        )
        self.assertIsNotNone(out["weighted_model_score"])
        self.assertAlmostEqual(out["vote_ratio"], 2 / 3)
        self.assertEqual(out["consensus_strength"], 0.75)
        self.assertAlmostEqual(out["evidence_coverage"], 1.0)
        self.assertEqual(out["context_score_weight"], 0.0)
        self.assertEqual(out["context_component"], 0.0)
        self.assertTrue(out["numeric_scores_used_in_meta_score"])
        self.assertTrue(out["votes_used_in_meta_score"])
        self.assertTrue(out["consensus_feature_used_in_meta_score"])
        self.assertTrue(out["context_missingness_used_in_meta_score"])
        self.assertIsNotNone(out["baseline_vs_explicit_score_delta"])
        self.assertEqual(out["meta_formula_version"], "AE17_EXPLICIT_META_COMBINATION_V1")

    def test_invalid_scores_excluded_like_null(self):
        out = MOD.compute_weighted_model_score(
            rf_score=MOD.parse_float_or_none("nan"),
            xgb_score=0.5,
            tab_score=MOD.parse_float_or_none("inf"),
        )
        self.assertEqual(out["active_model_score_count"], 1)
        self.assertAlmostEqual(out["active_model_weight_sum"], 0.35)
        self.assertAlmostEqual(out["weighted_model_score"], 0.5)

    def test_baseline_comparison_audit_handles_low_variance(self):
        # Two identical rows: historical variance false, but synthetic proof remains.
        rows = []
        for i in range(3):
            scored = MOD.compute_ae17_explicit_meta_combination(
                rf_score=0.5,
                xgb_score=0.5,
                tab_score=0.5,
                tab_vote=True,
                xgb_vote=True,
                rf_vote=True,
            )
            scored["historical_meta_row_id"] = f"r{i}"
            rows.append(scored)
        audit = MOD.build_baseline_comparison_audit(rows)
        self.assertFalse(audit["same_tier_different_scores_observed"])
        self.assertEqual(
            audit["major_limitation"], "HISTORICAL_INPUT_LACKS_WITHIN_TIER_SCORE_VARIANCE"
        )
        self.assertNotEqual(audit["classification"], MOD.CLASSIFICATION_INCOMPLETE_SUBSTANCE)
        self.assertEqual(audit["classification"], MOD.CLASSIFICATION_SYNTHETIC_PROVEN)

    def test_formula_audit_reports_substance(self):
        rows = [
            MOD.compute_ae17_explicit_meta_combination(
                rf_score=0.2,
                xgb_score=0.3,
                tab_score=0.4,
                tab_vote=False,
                xgb_vote=False,
                rf_vote=False,
            )
        ]
        audit = MOD.build_formula_audit(rows)
        self.assertFalse(audit["tier_only_scoring"])
        self.assertTrue(audit["numeric_scores_used"])
        self.assertTrue(audit["hard_clamping_inside_combinator"])
        self.assertTrue(audit["formula_substance_pass"])


class TestAE17FinalClosureAuditPathB(unittest.TestCase):
    def _load_closure(self):
        path = ROOT / "scripts" / "run_ae17_final_closure_audit.py"
        spec = importlib.util.spec_from_file_location("run_ae17_final_closure_audit", path)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_closure_rejects_tier_only_missing_formula_audits(self):
        closure = self._load_closure()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Minimal fake AE17 root missing formula/baseline audits.
            ae17 = root / "data" / "audits" / "ae17_real_meta_evidence_run_fake"
            (ae17 / "data").mkdir(parents=True)
            (ae17 / "reports").mkdir(parents=True)
            (ae17 / "audits").mkdir(parents=True)
            _write_csv(
                ae17 / "data" / "ae17_real_meta_feature_matrix.csv",
                [{"meta_score": "0.9", "meta_decision": "META_STRONG_WATCH"}],
            )
            _write_csv(
                ae17 / "data" / "ae17_real_meta_outputs.csv",
                [{"meta_score": "0.9", "meta_decision": "META_STRONG_WATCH"}],
            )
            (ae17 / "reports" / "ae17_real_meta_manifest.json").write_text(
                json.dumps({"classification": "old"}), encoding="utf-8"
            )
            (ae17 / "reports" / "ae17_real_meta_decision_gate.json").write_text(
                json.dumps({"classification": "old"}), encoding="utf-8"
            )
            for name in [
                "ae17_real_meta_feature_parity_audit.json",
                "ae17_real_meta_no_lookahead_audit.json",
                "ae17_real_meta_score_integrity_audit.json",
                "ae17_real_meta_pair_concentration_audit.json",
                "ae17_real_meta_partial_evidence_semantic_audit.json",
                "ae17_real_meta_authority_audit.json",
                "ae17_real_meta_null_safety_audit.json",
            ]:
                (ae17 / "audits" / name).write_text("{}", encoding="utf-8")
            (ae17 / "audits" / "ae17_real_meta_lineage_audit.csv").write_text(
                "historical_meta_row_id\n", encoding="utf-8"
            )
            out = root / "closure"
            # Point closure at fake tree by using project_root=root and relative path.
            # Required files include formula/baseline which are missing.
            result = closure.run_ae17_final_closure_audit(
                root,
                ae17_output_root=ae17,
                output_root=out,
            )
            self.assertEqual(
                result["classification"], closure.CLASSIFICATION_SUBSTANCE
            )

    def test_closure_pass_requires_explicit_formula_package(self):
        # Integration-style: run evidence into temp then closure.
        closure = self._load_closure()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Copy runner constants by importing MOD against real ROOT sources via
            # writing tiny CSV under real project is heavy; instead call MOD with
            # temp project root that includes source files.
            src = root / "data" / "src" / "good.csv"
            rows = [
                _base_row(
                    in_tab="True",
                    in_xgb="True",
                    in_rf="True",
                    tab_score="0.2",
                    predicted_probability_xgb="0.2",
                    predicted_probability_rf="0.2",
                    pair_address="0xA",
                ),
                _base_row(
                    target_row_id="t2",
                    candidate_id="c2",
                    in_tab="True",
                    in_xgb="True",
                    in_rf="True",
                    tab_score="0.95",
                    predicted_probability_xgb="0.95",
                    predicted_probability_rf="0.95",
                    pair_address="0xB",
                ),
            ]
            _write_csv(src, rows)
            ae17_out = root / "ae17_out"
            result = MOD.run_ae17_real_meta_evidence(
                root,
                source_glob="data/src/*.csv",
                output_root=ae17_out,
            )
            self.assertIn(
                result["classification"],
                {MOD.CLASSIFICATION_PASS, MOD.CLASSIFICATION_PASS_WARNINGS},
            )
            self.assertTrue(
                (ae17_out / "audits" / "ae17_real_meta_formula_audit.json").exists()
            )
            self.assertTrue(
                (
                    ae17_out / "audits" / "ae17_real_meta_baseline_comparison_audit.json"
                ).exists()
            )
            # Closure script resolves runner under ROOT (real repo), and audits under ae17_out.
            closure_out = root / "closure_out"
            # Provide a shim scripts path? Closure checks ROOT/scripts/run_ae17... which exists.
            # But ae17_output_root is absolute under tmp.
            # Feature parity / lineage etc. exist from runner.
            # Closure also needs project_root for relpath; use real ROOT so runner_exists passes.
            final = closure.run_ae17_final_closure_audit(
                ROOT,
                ae17_output_root=ae17_out,
                output_root=closure_out,
            )
            self.assertEqual(final["classification"], closure.CLASSIFICATION_PASS)
            self.assertTrue(final["ae17_closed"])
            self.assertFalse(final["ae18_started"])



class TestAE17ContextMissingness(unittest.TestCase):
    def test_context_fields_in_feature_and_output_rows(self):
        feature, output = MOD.process_source_row(
            _base_row(),
            source_file="t.csv",
            source_row_index=0,
            row_seq=0,
        )
        for row in (feature, output):
            self.assertIn("context_feature_available", row)
            self.assertIn("context_status", row)
            self.assertIn("context_missingness_reason", row)
            self.assertIn("context_score_weight", row)
            self.assertIs(row["context_feature_available"], False)
            self.assertEqual(row["context_status"], "AE17_CONTEXT_NOT_AVAILABLE_PENDING_AE18")
            self.assertEqual(
                row["context_missingness_reason"], "AE18_CONTEXT_LAYER_NOT_STARTED"
            )
            self.assertEqual(row["context_score_weight"], 0.0)
            self.assertIn("context_component", row)
            self.assertEqual(row["context_component"], 0.0)

        self.assertIn("context_feature_available", MOD.FEATURE_MATRIX_FIELDS)
        self.assertIn("context_status", MOD.FEATURE_MATRIX_FIELDS)
        self.assertIn("context_missingness_reason", MOD.FEATURE_MATRIX_FIELDS)
        self.assertIn("context_score_weight", MOD.FEATURE_MATRIX_FIELDS)
        self.assertIn("context_component", MOD.FEATURE_MATRIX_FIELDS)
        self.assertIn("weighted_model_score", MOD.FEATURE_MATRIX_FIELDS)
        self.assertIn("tier_only_scoring", MOD.FEATURE_MATRIX_FIELDS)
        for col in (
            "context_feature_available",
            "context_status",
            "context_missingness_reason",
            "context_score_weight",
            "weighted_model_score",
            "meta_layer_type",
            "tier_only_scoring",
        ):
            self.assertIn(col, MOD.FEATURE_PARITY_REQUIRED)

    def test_context_fields_survive_full_run_and_parity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "data" / "src" / "good.csv"
            _write_csv(src, [_base_row(in_tab="True", in_xgb="True", in_rf="True")])
            out = root / "out"
            result = MOD.run_ae17_real_meta_evidence(
                root,
                source_glob="data/src/*.csv",
                output_root=out,
            )
            self.assertIn(
                result["classification"],
                {MOD.CLASSIFICATION_PASS, MOD.CLASSIFICATION_PASS_WARNINGS},
            )
            with (out / "data" / "ae17_real_meta_feature_matrix.csv").open(encoding="utf-8") as f:
                feature_reader = csv.DictReader(f)
                feature_cols = feature_reader.fieldnames or []
                feature_row = next(feature_reader)
            with (out / "data" / "ae17_real_meta_outputs.csv").open(encoding="utf-8") as f:
                output_reader = csv.DictReader(f)
                output_cols = output_reader.fieldnames or []
                output_row = next(output_reader)

            for cols in (feature_cols, output_cols):
                for name in (
                    "context_feature_available",
                    "context_status",
                    "context_missingness_reason",
                    "context_score_weight",
                ):
                    self.assertIn(name, cols)

            for row in (feature_row, output_row):
                self.assertEqual(row["context_feature_available"], "False")
                self.assertEqual(
                    row["context_status"], "AE17_CONTEXT_NOT_AVAILABLE_PENDING_AE18"
                )
                self.assertEqual(
                    row["context_missingness_reason"], "AE18_CONTEXT_LAYER_NOT_STARTED"
                )
                self.assertEqual(float(row["context_score_weight"]), 0.0)

            parity = json.loads(
                (out / "audits" / "ae17_real_meta_feature_parity_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(parity["feature_parity_pass"])
            self.assertTrue(parity["context_feature_contract_present"])
            self.assertFalse(parity["context_feature_available"])
            self.assertEqual(
                parity["context_status"], "AE17_CONTEXT_NOT_AVAILABLE_PENDING_AE18"
            )
            self.assertEqual(
                parity["context_missingness_reason"], "AE18_CONTEXT_LAYER_NOT_STARTED"
            )
            self.assertEqual(parity["context_score_weight"], 0.0)

            nl = json.loads(
                (out / "audits" / "ae17_real_meta_no_lookahead_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(nl["no_lookahead_pass"])

            gate = json.loads(
                (out / "reports" / "ae17_real_meta_decision_gate.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = json.loads(
                (out / "reports" / "ae17_real_meta_manifest.json").read_text(encoding="utf-8")
            )
            for report in (gate, manifest):
                self.assertTrue(report["context_feature_contract_present"])
                self.assertFalse(report["context_feature_available"])
                self.assertEqual(
                    report["context_status"], "AE17_CONTEXT_NOT_AVAILABLE_PENDING_AE18"
                )
                self.assertEqual(
                    report["context_missingness_reason"], "AE18_CONTEXT_LAYER_NOT_STARTED"
                )
                self.assertEqual(report["context_score_weight"], 0.0)


class TestAE17AuditsAndOutputs(unittest.TestCase):
    def test_score_integrity_and_feature_matrix_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "data" / "src" / "good.csv"
            rows = [
                _base_row(
                    in_tab="True",
                    in_xgb="True",
                    in_rf="True",
                    pair_address="0xA",
                    target_net_profitable="1",
                    sim_net_return="0.1",
                ),
                _base_row(
                    target_row_id="t2",
                    candidate_id="c2",
                    pair_address="0xB",
                    in_tab="False",
                    in_xgb="False",
                    in_rf="False",
                ),
            ]
            _write_csv(src, rows)
            out = root / "out"
            result = MOD.run_ae17_real_meta_evidence(
                root,
                source_glob="data/src/*.csv",
                output_root=out,
            )
            self.assertIn(
                result["classification"],
                {MOD.CLASSIFICATION_PASS, MOD.CLASSIFICATION_PASS_WARNINGS},
            )
            score_audit = json.loads(
                (out / "audits" / "ae17_real_meta_score_integrity_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(score_audit["out_of_range_final_scores"], 0)
            self.assertTrue(score_audit["score_integrity_pass"])

            with (out / "data" / "ae17_real_meta_feature_matrix.csv").open(encoding="utf-8") as f:
                feature_cols = next(csv.reader(f))
            for forbidden in MOD.FORBIDDEN_FEATURE_COLUMNS:
                self.assertNotIn(forbidden, feature_cols)
            for name in (
                "context_feature_available",
                "context_status",
                "context_missingness_reason",
                "context_score_weight",
            ):
                self.assertIn(name, feature_cols)

            with (out / "data" / "ae17_real_meta_outputs.csv").open(encoding="utf-8") as f:
                output_cols = next(csv.reader(f))
            self.assertIn("sim_net_return", output_cols)
            self.assertIn("outcome_label_value", output_cols)
            self.assertIn("context_status", output_cols)

            nl = json.loads(
                (out / "audits" / "ae17_real_meta_no_lookahead_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(nl["no_lookahead_pass"])

            auth = json.loads(
                (out / "audits" / "ae17_real_meta_authority_audit.json").read_text(encoding="utf-8")
            )
            self.assertFalse(auth["trade_authority"])
            self.assertFalse(auth["db_mutation"])
            self.assertFalse(auth["training_or_fit"])
            self.assertTrue(auth["paper_demo_only"])

            pair = json.loads(
                (out / "audits" / "ae17_real_meta_pair_concentration_audit.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("top_pair_share", pair)
            self.assertIn("hhi", pair)
            self.assertGreater(pair["hhi"], 0)

    def test_no_lookahead_fails_if_forbidden_columns_present(self):
        contaminated = [
            {
                "candidate_id": "c1",
                "target_net_profitable": 1,
                "sim_net_return": 0.2,
                "meta_score": 0.1,
            }
        ]
        audit = MOD.build_no_lookahead_audit(contaminated, "x.csv")
        self.assertFalse(audit["no_lookahead_pass"])
        self.assertIn("target_net_profitable", audit["forbidden_columns_present"])

    def test_pair_concentration_hhi(self):
        rows = [{"pair_address": "A"}] * 3 + [{"pair_address": "B"}] * 1
        audit = MOD.compute_pair_concentration(rows)
        self.assertEqual(audit["total_rows"], 4)
        self.assertEqual(audit["unique_pairs"], 2)
        self.assertEqual(audit["top_pair"], "A")
        self.assertEqual(audit["top_pair_count"], 3)
        self.assertAlmostEqual(audit["top_pair_share"], 0.75)
        expected_hhi = (0.75**2) + (0.25**2)
        self.assertAlmostEqual(audit["hhi"], expected_hhi)
        self.assertEqual(audit["top_pair_share_status"], "high_risk")
        self.assertIn("not tradability proof", audit["note"])

    def test_no_fit_or_training_calls_in_module(self):
        src = inspect.getsource(MOD)
        lowered = src.lower()
        # Ensure no sklearn/xgboost/tabicl fit/train invocation patterns.
        self.assertNotIn(".fit(", src)
        self.assertNotIn("train_model", lowered)
        self.assertNotIn("import sklearn", src)
        self.assertNotIn("import xgboost", src)
        self.assertNotIn("import tabicl", src)


if __name__ == "__main__":
    unittest.main()
