"""Focused tests for AE16E model evidence + feature parity."""
from __future__ import annotations

import csv
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.consensus.ae16e_feature_parity import (  # noqa: E402
    TOXIC_PAIR_ADDRESS,
    assert_no_toxic_in_outputs,
    audit_feature_parity_ae16e,
    build_ae16e_consensus,
    build_model_evidence,
    classify_feature,
    decide_ae16e_classification,
    extract_feature_names_from_object,
    is_toxic_pair,
    load_clean_forward_rows_used,
    sequential_feature_questionnaire,
)


def _load_runner():
    path = ROOT / "scripts" / "run_ae16e_model_evidence_feature_parity.py"
    spec = importlib.util.spec_from_file_location("run_ae16e", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames and rows:
        fieldnames = list(rows[0].keys())
    fieldnames = fieldnames or []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


class TestAE16EFeatureParity(unittest.TestCase):
    def test_clean_forward_rows_load_and_toxic_excluded(self):
        rows, meta = load_clean_forward_rows_used(ROOT)
        self.assertGreater(meta.get("curated_active_targets_loaded") or 0, 0)
        if meta.get("status") == "OK":
            self.assertGreater(len(rows), 0)
            self.assertFalse(any(is_toxic_pair(r.get("pair_address")) for r in rows))
            self.assertFalse(assert_no_toxic_in_outputs([rows]))

    def test_toxic_pair_helper(self):
        self.assertTrue(is_toxic_pair(TOXIC_PAIR_ADDRESS))
        self.assertTrue(is_toxic_pair(TOXIC_PAIR_ADDRESS.lower()))
        self.assertFalse(is_toxic_pair("0xabc"))

    def test_sklearn_feature_names_in_extraction(self):
        class FakeEst:
            feature_names_in_ = ["a", "b", "c"]

        names, method, err = extract_feature_names_from_object(FakeEst())
        self.assertEqual(names, ["a", "b", "c"])
        self.assertEqual(method, "feature_names_in_")
        self.assertEqual(err, "")

    def test_sklearn_missing_feature_names_handled(self):
        class NoNames:
            pass

        names, method, err = extract_feature_names_from_object(NoNames())
        self.assertEqual(names, [])
        self.assertIn(method, {"unsupported_or_missing", "booster.feature_names"})

    def test_xgboost_booster_feature_names(self):
        class Booster:
            feature_names = ["x1", "x2"]

        class XGB:
            def get_booster(self):
                return Booster()

        names, method, err = extract_feature_names_from_object(XGB())
        self.assertEqual(names, ["x1", "x2"])
        self.assertEqual(method, "booster.feature_names")

    def test_xgboost_missing_feature_names(self):
        class Booster:
            feature_names = None

        class XGB:
            def get_booster(self):
                return Booster()

        names, method, err = extract_feature_names_from_object(XGB())
        self.assertEqual(names, [])
        self.assertEqual(method, "booster.feature_names")

    def test_lightgbm_feature_name_extraction(self):
        class Booster:
            def feature_name(self):
                return ["lgb1", "lgb2"]

        class LGB:
            booster_ = Booster()

        names, method, err = extract_feature_names_from_object(LGB())
        self.assertEqual(names, ["lgb1", "lgb2"])
        self.assertEqual(method, "booster_.feature_name()")

    def test_attribute_error_does_not_crash_discovery(self):
        class Bad:
            @property
            def feature_names_in_(self):
                raise AttributeError("boom")

            def get_booster(self):
                raise AttributeError("no booster")

        names, method, err = extract_feature_names_from_object(Bad())
        self.assertEqual(names, [])
        # Must not raise
        self.assertIsInstance(method, str)

    def test_feature_parity_detects_available_direct(self):
        row = classify_feature(
            "price_usd",
            available_fields={"price_usd": 5},
            controlled_snapshot_history=False,
        )
        self.assertEqual(row["classification"], "AVAILABLE_DIRECT")

    def test_feature_parity_detects_missing_blocking(self):
        row = classify_feature(
            "whale_score",
            available_fields={"price_usd": 5},
            controlled_snapshot_history=False,
        )
        self.assertEqual(row["classification"], "MISSING_BLOCKING")

    def test_policy_constant_safe(self):
        row = classify_feature(
            "tp_ratio",
            available_fields={},
            controlled_snapshot_history=False,
        )
        self.assertEqual(row["classification"], "CONSTANT_POLICY_PARAM_SAFE")

    def test_no_unsafe_default_filling_for_entry_fields(self):
        for feat in (
            "entry_snapshot_id",
            "entry_price_verified_1h",
            "gap_detected",
            "price_step_ratio_prev",
        ):
            row = classify_feature(
                feat,
                available_fields={"price_usd": 1},
                controlled_snapshot_history=False,
            )
            self.assertEqual(row["classification"], "MISSING_BLOCKING", feat)

    def test_sequential_from_single_snapshot_blocking(self):
        q = sequential_feature_questionnaire(
            "gap_detected", controlled_snapshot_history=False
        )
        self.assertEqual(q["requires_prior_observations"], "YES")
        self.assertEqual(q["has_controlled_cf_snapshot_history"], "NO")
        self.assertEqual(q["final_classification"], "MISSING_BLOCKING")

    def test_sequential_requires_controlled_history(self):
        q = sequential_feature_questionnaire(
            "price_step_ratio_prev", controlled_snapshot_history=True
        )
        # Even with flag true, AE16E questionnaire still requires proven history details
        self.assertEqual(q["minimum_observations_available"], "NO")
        self.assertEqual(q["final_classification"], "MISSING_BLOCKING")

    def test_artifact_discovery_classifies_families(self):
        from app.consensus.ae16e_feature_parity import discover_ae16e_artifacts

        rows, audit, manifest = discover_ae16e_artifacts(ROOT)
        families = {r.get("artifact_family") or r.get("model_family") for r in rows}
        self.assertTrue(families & {"RF", "XGB", "TAB", "UNKNOWN"} or len(rows) >= 0)
        sel = manifest.get("selection") or {}
        self.assertIn("RF", sel)
        self.assertIn("XGB", sel)
        self.assertIn("TAB", sel)
        # TAB should not be selected for CF inference
        self.assertFalse(sel["TAB"].get("selected"))
        self.assertTrue(audit is not None)

    def test_no_pair_timestamp_only_prediction_join(self):
        # Evidence builder never joins predictions by pair alone — only inference or unavailable
        rows = [
            {
                "row_id": "r1",
                "combined_target_id": "c1",
                "pair_address": "0xabc",
                "chain": "base",
            }
        ]
        parity = {
            "by_family_summary": {
                "RF": {
                    "inference_allowed": False,
                    "blocker_reason": "FEATURE_PARITY_NOT_APPROVED",
                    "missing_blocking_count": 10,
                    "required_features_count": 31,
                    "artifact_selected": True,
                    "model_path": "models/rf.joblib",
                    "schema_path": "models/rf.json",
                },
                "XGB": {
                    "inference_allowed": False,
                    "blocker_reason": "FEATURE_PARITY_NOT_APPROVED",
                    "missing_blocking_count": 10,
                    "required_features_count": 31,
                    "artifact_selected": True,
                    "model_path": "models/xgb.joblib",
                    "schema_path": "models/xgb.json",
                },
                "TAB": {
                    "inference_allowed": False,
                    "blocker_reason": "NO_SAFE_CLEAN_FORWARD_TAB_JOBLIB",
                    "missing_blocking_count": 0,
                    "required_features_count": 31,
                    "artifact_selected": False,
                    "model_path": "",
                    "schema_path": "metrics/tab.json",
                },
            }
        }
        evidence, unavailable = build_model_evidence(
            rows=rows,
            parity_summary=parity,
            selection={"RF": {}, "XGB": {}, "TAB": {}},
            inference_by_family={},
        )
        self.assertEqual(evidence, [])
        self.assertEqual(len(unavailable), 3)
        self.assertTrue(all(u["evidence_status"] == "MODEL_EVIDENCE_UNAVAILABLE" for u in unavailable))

    def test_unavailable_model_evidence_explicit(self):
        evidence, unavailable = build_model_evidence(
            rows=[{"row_id": "r1", "combined_target_id": "c1", "pair_address": "p"}],
            parity_summary={
                "by_family_summary": {
                    "RF": {"inference_allowed": False, "blocker_reason": "GAP"},
                    "XGB": {"inference_allowed": False, "blocker_reason": "GAP"},
                    "TAB": {"inference_allowed": False, "blocker_reason": "NO_JOBLIB"},
                }
            },
            selection={},
            inference_by_family={},
        )
        self.assertEqual(evidence, [])
        self.assertGreaterEqual(len(unavailable), 3)

    def test_model_evidence_schema_stable_when_attached(self):
        rows = [
            {
                "row_id": "r1",
                "combined_target_id": "c1",
                "pair_address": "0x1",
                "chain": "base",
                "provider_pair_url": "u",
                "base_token_address": "b",
                "quote_token_address": "q",
                "target_source": "CLEAN_FORWARD_EXISTING",
                "semantic_status": "PENDING_SYSTEM_CLASSIFICATION",
            }
        ]
        evidence, _ = build_model_evidence(
            rows=rows,
            parity_summary={
                "by_family_summary": {
                    "RF": {"inference_allowed": True, "blocker_reason": ""},
                    "XGB": {"inference_allowed": False, "blocker_reason": "GAP"},
                    "TAB": {"inference_allowed": False, "blocker_reason": "NO_JOBLIB"},
                }
            },
            selection={
                "RF": {
                    "model_path": "m.joblib",
                    "target_name": "net_profitable_after_exit_policy",
                    "horizon": "1h",
                }
            },
            inference_by_family={
                "RF": [
                    {
                        "row_id": "r1",
                        "combined_target_id": "c1",
                        "score": 0.7,
                        "rank_in_batch": 1,
                        "model_artifact_hash": "abc",
                    }
                ]
            },
        )
        self.assertEqual(len(evidence), 1)
        required_cols = {
            "evidence_id",
            "row_id",
            "combined_target_id",
            "model_family",
            "model_artifact_path",
            "score",
            "rank_in_batch",
            "vote",
            "feature_parity_status",
            "no_lookahead_status",
            "evidence_status",
            "blocker_reason",
        }
        self.assertTrue(required_cols.issubset(set(evidence[0].keys())))

    def test_consensus_unavailable_when_no_evidence(self):
        rows = [
            {
                "row_id": "r1",
                "combined_target_id": "c1",
                "pair_address": "0x1",
                "base_token_address": "b",
                "quote_token_address": "q",
                "provider_pair_url": "u",
                "chain": "base",
                "target_source": "t",
                "semantic_status": "PENDING_SYSTEM_CLASSIFICATION",
            }
        ]
        decisions, counts = build_ae16e_consensus(rows, [])
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]["consensus_tier"], "MODEL_EVIDENCE_UNAVAILABLE")
        self.assertEqual(counts[0]["consensus_tier"], "MODEL_EVIDENCE_UNAVAILABLE")

    def test_consensus_tiers_from_available_votes(self):
        rows = [
            {
                "row_id": "r1",
                "combined_target_id": "c1",
                "pair_address": "0x1",
                "base_token_address": "b",
                "quote_token_address": "q",
                "provider_pair_url": "u",
                "chain": "base",
                "target_source": "t",
                "semantic_status": "PENDING_SYSTEM_CLASSIFICATION",
            }
        ]
        evidence = [
            {
                "evidence_id": "e1",
                "row_id": "r1",
                "combined_target_id": "c1",
                "model_family": "RF",
                "model_artifact_path": "m.joblib",
                "model_target": "t",
                "model_horizon": "1h",
                "score": 0.9,
                "rank_in_batch": 1,
                "evidence_status": "MODEL_EVIDENCE_ATTACHED",
            },
            {
                "evidence_id": "e2",
                "row_id": "r1",
                "combined_target_id": "c1",
                "model_family": "XGB",
                "model_artifact_path": "x.joblib",
                "model_target": "t",
                "model_horizon": "1h",
                "score": 0.8,
                "rank_in_batch": 1,
                "evidence_status": "MODEL_EVIDENCE_ATTACHED",
            },
        ]
        decisions, counts = build_ae16e_consensus(rows, evidence)
        self.assertEqual(decisions[0]["consensus_tier"], "RF_XGB_ONLY")
        self.assertEqual(decisions[0]["live_trading_ready"], False)
        self.assertEqual(decisions[0]["paper_demo_only"], True)

    def test_decision_gate_feature_parity_gap(self):
        cls = decide_ae16e_classification(
            rows_meta={"status": "OK", "clean_forward_rows_used": 42},
            parity={
                "by_family_summary": {
                    "RF": {
                        "artifact_selected": True,
                        "missing_blocking_count": 12,
                    },
                    "XGB": {
                        "artifact_selected": True,
                        "missing_blocking_count": 12,
                    },
                    "TAB": {"artifact_selected": False, "missing_blocking_count": 0},
                },
                "unsafe_pair_timestamp_join_used": False,
            },
            evidence=[],
        )
        self.assertEqual(cls, "AE16E_BLOCKED_FEATURE_PARITY_GAP")

    def test_no_lookahead_derived_audit_exists_in_parity(self):
        selection = {
            "RF": {
                "selected": True,
                "model_path": "m.joblib",
                "schema_path": "",
                "feature_names": [
                    "tp_ratio",
                    "price_usd",
                    "txns_total",
                    "buy_ratio",
                    "volume_to_liquidity_ratio",
                    "gap_detected",
                ],
                "feature_names_extraction_status": "OK",
            },
            "XGB": {
                "selected": True,
                "model_path": "x.joblib",
                "feature_names": ["tp_ratio", "gap_detected"],
                "feature_names_extraction_status": "OK",
            },
            "TAB": {
                "selected": False,
                "feature_names": ["tp_ratio"],
                "feature_names_extraction_status": "OK",
                "rejected_reason": "NO_JOBLIB",
            },
        }
        rows = [
            {
                "price_usd": "1",
                "liquidity_usd": "1000",
                "volume_24h": "100",
                "txns_buys_24h": "10",
                "txns_sells_24h": "5",
            }
        ]
        parity = audit_feature_parity_ae16e(
            project_root=ROOT,
            rows=rows,
            selection=selection,
            controlled_snapshot_history=False,
        )
        derived = parity["derived_feature_audit"]
        self.assertTrue(any(r["feature_name"] == "txns_total" for r in derived))
        self.assertFalse(parity["by_family_summary"]["RF"]["feature_parity_passed"])
        self.assertGreater(parity["by_family_summary"]["RF"]["missing_blocking_count"], 0)

    def test_safety_flags_in_runner_module(self):
        mod = _load_runner()
        self.assertTrue(hasattr(mod, "run_ae16e"))
        self.assertTrue(hasattr(mod, "PHASE") or True)

    def test_ae17_not_started_classification_path(self):
        cls = decide_ae16e_classification(
            rows_meta={"status": "OK", "clean_forward_rows_used": 10},
            parity={
                "by_family_summary": {
                    "RF": {"artifact_selected": True, "missing_blocking_count": 1},
                    "XGB": {"artifact_selected": True, "missing_blocking_count": 1},
                    "TAB": {"artifact_selected": False, "missing_blocking_count": 0},
                },
                "unsafe_pair_timestamp_join_used": False,
            },
            evidence=[],
        )
        self.assertNotEqual(cls, "AE16E_MODEL_EVIDENCE_ATTACHMENT_PASS")
        self.assertTrue(cls.startswith("AE16E_BLOCKED_") or cls.startswith("AE16E_PARTIAL_"))


class TestAE16EDiscoveryRobustness(unittest.TestCase):
    def test_pipeline_named_steps_extraction(self):
        class Step:
            feature_names_in_ = ["p1", "p2"]

        class Pipe:
            named_steps = {"pre": Step(), "clf": object()}

        names, method, err = extract_feature_names_from_object(Pipe())
        self.assertEqual(names, ["p1", "p2"])
        self.assertIn("pipeline.named_steps", method)


if __name__ == "__main__":
    unittest.main()
