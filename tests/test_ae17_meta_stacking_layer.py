"""Focused unit tests for AE17 Meta-Model / Stacking Layer (deterministic shadow)."""

from __future__ import annotations

import csv
import importlib.util
import inspect
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.meta import (  # noqa: E402
    CLASSIFICATION_MISSING_AE16,
    CLASSIFICATION_PASS,
    CLASSIFICATION_PASS_LIMITATIONS,
)
from app.meta.audits import (  # noqa: E402
    audit_authority,
    audit_feature_parity,
    audit_no_lookahead,
    audit_null_safety,
    audit_pair_concentration,
)
from app.meta.constants import FORBIDDEN_FEATURE_FIELDS, META_FEATURE_FIELDS  # noqa: E402
from app.meta.discovery import discover_ae16_artifacts  # noqa: E402
from app.meta.features import (  # noqa: E402
    build_meta_feature_row_from_ae16,
    feature_matrix_dicts,
    is_forbidden_feature_name,
    parse_optional_float,
)
from app.meta.models import AE17MetaFeatureRow  # noqa: E402
from app.meta.scoring import clamp_score, compute_ae17_meta_shadow_score  # noqa: E402


def _load_runner():
    path = ROOT / "scripts" / "run_ae17_meta_stacking_layer.py"
    spec = importlib.util.spec_from_file_location("run_ae17_meta_stacking_layer", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _sample_ae16_row(**overrides) -> dict:
    base = {
        "price_source_key": "dexscreener|base|0xabc",
        "provider": "dexscreener",
        "chain": "base",
        "pair_address": "0xabc",
        "RF_score": "0.8",
        "RF_vote": "true",
        "RF_status": "MODEL_EVIDENCE_ATTACHED",
        "XGB_score": "0.7",
        "XGB_vote": "true",
        "XGB_status": "MODEL_EVIDENCE_ATTACHED",
        "TAB16_score": "0.9",
        "TAB16_vote": "true",
        "TAB16_status": "MODEL_EVIDENCE_ATTACHED",
        "consensus_preview_tier": "TAB_XGB_RF_ALL3",
        "timestamp": "2026-07-24T19:21:18+00:00",
        "provider_pair_url": "https://dexscreener.com/base/0xabc",
        "provider_payload_hash": "deadbeef",
        "base_token_address": "0xbase",
        "quote_token_address": "0xquote",
        "clean_forward_candidate_id": "cand1",
        "clean_forward_decision_input_id": "dec1",
    }
    base.update(overrides)
    return base


class TestAE17Discovery(unittest.TestCase):
    def test_discovery_success_known_root(self):
        result = discover_ae16_artifacts(
            ROOT,
            ae16_root="data/audits/ae16_tab16_direct_target_serving_safe_20260724T205012Z",
        )
        self.assertEqual(result.status, "AE17_INPUTS_DISCOVERED")
        self.assertIsNotNone(result.selected_consensus_path)
        self.assertIn("consensus", (result.selected_consensus_path or "").lower())

    def test_discovery_controlled_blocker_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "no_ae16_here"
            empty.mkdir()
            # Point project root at empty temp so known roots also miss.
            # Use ae16_root only — still searches known roots under real ROOT when
            # project_root is ROOT. Instead invent a fake project tree.
            fake_root = Path(tmp) / "proj"
            (fake_root / "data" / "audits").mkdir(parents=True)
            result = discover_ae16_artifacts(fake_root, ae16_root=empty)
            self.assertEqual(result.status, CLASSIFICATION_MISSING_AE16)
            self.assertIn("consensus_rows", result.missing_required_artifacts)
            self.assertTrue(result.recommended_next_action)


class TestAE17Features(unittest.TestCase):
    def test_meta_feature_construction(self):
        row = build_meta_feature_row_from_ae16(
            _sample_ae16_row(),
            source_artifact="test.csv",
            source_schema_hash="abc",
        )
        self.assertEqual(row.consensus_tier, "TAB_XGB_RF_ALL3")
        self.assertEqual(row.rf_score, 0.8)
        self.assertTrue(row.rf_vote)
        self.assertIn("rf_score", row.to_dict())

    def test_missing_context_does_not_crash(self):
        row = build_meta_feature_row_from_ae16(
            _sample_ae16_row(),
            source_artifact="test.csv",
            source_schema_hash="abc",
        )
        self.assertFalse(row.context_feature_available)
        self.assertEqual(row.context_score_weight, 0.0)
        self.assertEqual(row.context_status, "AE17_CONTEXT_NOT_AVAILABLE_PENDING_AE18")

    def test_context_score_weight_defaults_to_zero(self):
        row = build_meta_feature_row_from_ae16(
            _sample_ae16_row(context_feature_available="false"),
            source_artifact="t.csv",
            source_schema_hash="h",
        )
        self.assertEqual(row.context_score_weight, 0.0)

    def test_missing_model_score_remains_null_not_zero_filled(self):
        row = build_meta_feature_row_from_ae16(
            _sample_ae16_row(
                RF_score="",
                RF_vote="false",
                RF_status="MODEL_EVIDENCE_UNAVAILABLE",
                consensus_preview_tier="MODEL_EVIDENCE_UNAVAILABLE",
            ),
            source_artifact="t.csv",
            source_schema_hash="h",
        )
        self.assertIsNone(row.rf_score)
        self.assertNotEqual(row.rf_score, 0)
        self.assertNotEqual(row.rf_score, 0.0)

    def test_forbidden_future_outcome_field_exclusion(self):
        for name in ("future_return", "max_upside", "realized_pnl", "hit_tp", "pnl"):
            self.assertTrue(is_forbidden_feature_name(name))
        matrix = feature_matrix_dicts(
            [
                build_meta_feature_row_from_ae16(
                    _sample_ae16_row(future_return="1.2", max_upside="9"),
                    source_artifact="t.csv",
                    source_schema_hash="h",
                )
            ]
        )
        self.assertNotIn("future_return", matrix[0])
        self.assertNotIn("max_upside", matrix[0])
        for col in matrix[0]:
            self.assertFalse(is_forbidden_feature_name(col))
        for forbidden in FORBIDDEN_FEATURE_FIELDS:
            self.assertNotIn(forbidden, META_FEATURE_FIELDS)

    def test_parse_optional_float_null_safety(self):
        self.assertIsNone(parse_optional_float(None))
        self.assertIsNone(parse_optional_float(""))
        self.assertIsNone(parse_optional_float("nan"))
        self.assertEqual(parse_optional_float("0"), 0.0)
        self.assertEqual(parse_optional_float(0.5), 0.5)


class TestAE17Scoring(unittest.TestCase):
    def _row(self, tier: str, **kwargs) -> AE17MetaFeatureRow:
        raw = _sample_ae16_row(consensus_preview_tier=tier, **kwargs)
        return build_meta_feature_row_from_ae16(raw, source_artifact="t.csv", source_schema_hash="h")

    def test_deterministic_rule_based_meta_scoring(self):
        out = compute_ae17_meta_shadow_score(self._row("TAB_XGB_RF_ALL3"))
        self.assertEqual(out.meta_decision, "META_STRONG_WATCH")
        self.assertAlmostEqual(out.meta_score or -1, 0.90)
        self.assertIn("all three model slots agree", out.meta_reason)

        out2 = compute_ae17_meta_shadow_score(self._row("TAB_XGB_ONLY"))
        self.assertEqual(out2.meta_decision, "META_RESEARCH_ONLY")
        self.assertAlmostEqual(out2.meta_score or -1, 0.45)
        self.assertIn("historically weaker/research-only tier", out2.meta_reason)

        out3 = compute_ae17_meta_shadow_score(self._row("REJECT"))
        self.assertEqual(out3.meta_decision, "META_REJECT")
        self.assertEqual(out3.meta_score, 0.0)

    def test_score_clamping(self):
        score, clamped, reason = clamp_score(1.2)
        self.assertEqual(score, 1.0)
        self.assertTrue(clamped)
        score2, clamped2, _ = clamp_score(-0.1)
        self.assertEqual(score2, 0.0)
        self.assertTrue(clamped2)
        score3, clamped3, _ = clamp_score(0.5)
        self.assertEqual(score3, 0.5)
        self.assertFalse(clamped3)
        self.assertIsNone(clamp_score(None)[0])

        row = self._row("TAB_XGB_RF_ALL3")
        row.context_feature_available = True
        row.context_score_weight = 0.20  # excessive boost
        out = compute_ae17_meta_shadow_score(row)
        self.assertEqual(out.meta_score, 1.0)
        self.assertTrue(out.score_clamped)
        self.assertAlmostEqual(out.pre_clamp_meta_score or 0, 1.10, places=5)

    def test_unavailable_consensus_keeps_meta_score_null(self):
        out = compute_ae17_meta_shadow_score(self._row("MODEL_EVIDENCE_UNAVAILABLE"))
        self.assertIsNone(out.meta_score)
        self.assertEqual(out.meta_decision, "META_UNAVAILABLE")
        out2 = compute_ae17_meta_shadow_score(self._row(""))
        # empty tier -> missing
        row = self._row("REJECT")
        row.consensus_tier = None
        out3 = compute_ae17_meta_shadow_score(row)
        self.assertIsNone(out3.meta_score)
        self.assertEqual(out3.meta_decision, "META_UNAVAILABLE")
        self.assertIn("missing consensus tier", out3.meta_reason)

    def test_null_numeric_fields_do_not_cause_typeerror(self):
        row = self._row("REJECT")
        row.rf_score = None
        row.xgb_score = None
        row.tab_score = None
        row.context_score_weight = None  # type: ignore[assignment]
        # scoring must not TypeError
        out = compute_ae17_meta_shadow_score(row)
        self.assertEqual(out.meta_decision, "META_REJECT")
        self.assertEqual(out.context_score_weight, 0.0)


class TestAE17Audits(unittest.TestCase):
    def test_no_lookahead_and_feature_parity(self):
        rows = [
            build_meta_feature_row_from_ae16(
                _sample_ae16_row(),
                source_artifact="t.csv",
                source_schema_hash="h",
            )
        ]
        shadows = [compute_ae17_meta_shadow_score(r) for r in rows]
        la = audit_no_lookahead(rows, source_columns=["future_return", "RF_score"])
        self.assertTrue(la["passed"])
        parity = audit_feature_parity(rows, shadows)
        self.assertTrue(parity["passed"], parity.get("issues"))

    def test_pair_concentration_thresholds(self):
        rows = []
        for i in range(10):
            pair = "0xSAME" if i < 6 else f"0x{i}"
            rows.append(
                build_meta_feature_row_from_ae16(
                    _sample_ae16_row(pair_address=pair, price_source_key=f"dex|base|{pair}"),
                    source_artifact="t.csv",
                    source_schema_hash="h",
                )
            )
        result = audit_pair_concentration(rows)
        self.assertGreater(result.top_pair_share or 0, 0.50)
        self.assertIn("PAIR_CONCENTRATION_HIGH_RISK", result.concentration_status)
        self.assertFalse(result.meta_authority_allowed)
        self.assertIn("SMALL_SAMPLE_WARNING", result.concentration_status)

    def test_authority_remains_shadow_research_only(self):
        rows = [
            build_meta_feature_row_from_ae16(
                _sample_ae16_row(),
                source_artifact="t.csv",
                source_schema_hash="h",
            )
        ]
        shadows = [compute_ae17_meta_shadow_score(r) for r in rows]
        auth = audit_authority(shadows)
        self.assertTrue(auth.passed)
        self.assertFalse(auth.trade_authority)
        self.assertFalse(auth.live_trading_ready)
        self.assertTrue(auth.paper_demo_only)
        self.assertFalse(auth.risk_override_authority)
        self.assertEqual(auth.authority_status, "AE17_RESEARCH_SHADOW_ONLY")
        for s in shadows:
            self.assertFalse(s.trade_authority)
            self.assertEqual(s.meta_mode, "rule_based_meta_shadow")

    def test_null_safety_audit(self):
        rows = [
            build_meta_feature_row_from_ae16(
                _sample_ae16_row(RF_score="", RF_status="MODEL_EVIDENCE_UNAVAILABLE"),
                source_artifact="t.csv",
                source_schema_hash="h",
            )
        ]
        shadows = [compute_ae17_meta_shadow_score(r) for r in rows]
        ns = audit_null_safety(rows, shadows)
        self.assertTrue(ns["passed"], ns.get("issues"))


class TestAE17NoTraining(unittest.TestCase):
    def test_no_sklearn_xgboost_tabicl_training_import_requirement(self):
        import app.meta.scoring as scoring
        import app.meta.features as features
        import app.meta.pipeline as pipeline
        import app.meta.discovery as discovery
        import app.meta.audits as audits

        for mod in (scoring, features, pipeline, discovery, audits):
            src = inspect.getsource(mod)
            self.assertNotIn("import sklearn", src)
            self.assertNotIn("import xgboost", src)
            self.assertNotIn("import tabicl", src)
            self.assertNotIn("from sklearn", src)
            self.assertNotIn("from xgboost", src)
            self.assertNotIn("from tabicl", src)
            self.assertNotIn(".fit(", src)

    def test_no_fit_call_in_scoring(self):
        src = inspect.getsource(compute_ae17_meta_shadow_score)
        self.assertNotIn("fit(", src)


class TestAE17RunnerIntegration(unittest.TestCase):
    def test_runner_on_real_ae16_root(self):
        from app.meta.pipeline import run_ae17_meta_stacking_layer

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ae17_out"
            result = run_ae17_meta_stacking_layer(
                ROOT,
                ae16_root="data/audits/ae16_tab16_direct_target_serving_safe_20260724T205012Z",
                output_root=out,
            )
            self.assertIn(
                result["classification"],
                {CLASSIFICATION_PASS, CLASSIFICATION_PASS_LIMITATIONS},
            )
            self.assertGreater(result["feature_row_count"], 0)
            self.assertTrue((out / "reports" / "ae17_decision_gate.json").is_file())
            self.assertTrue((out / "data" / "ae17_meta_feature_rows.csv").is_file())
            self.assertTrue((out / "audits" / "ae17_pair_concentration_audit.csv").is_file())
            self.assertTrue((out / "audits" / "ae17_authority_audit.json").is_file())
            gate = result["decision_gate"]
            self.assertEqual(gate["ae18_status"], "BLOCKED")
            self.assertEqual(gate["ae19_status"], "BLOCKED")
            self.assertFalse(gate["training_performed"])
            self.assertFalse(gate["trade_authority"])

    def test_runner_module_loads(self):
        mod = _load_runner()
        self.assertTrue(hasattr(mod, "main"))
        self.assertTrue(hasattr(mod, "parse_args"))


if __name__ == "__main__":
    unittest.main()
