"""Focused tests for AE16 model-evidence bridge completion."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.consensus.evidence_bridge import (  # noqa: E402
    audit_exact_id_joins,
    audit_feature_parity,
    build_attachment_v2,
    decide_completion_classification,
    discover_direct_target_artifacts,
)


def _load_runner():
    path = ROOT / "scripts" / "run_ae16_model_evidence_bridge_completion.py"
    spec = importlib.util.spec_from_file_location("run_ae16_bridge", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestAE16EvidenceBridge(unittest.TestCase):
    def test_canonical_roots_existence_recorded(self):
        rows, manifest = discover_direct_target_artifacts(ROOT)
        self.assertTrue(manifest["canonical_expected_roots"])
        for root in manifest["canonical_expected_roots"]:
            self.assertIn("path_exists", root)
            self.assertIn("expected_path", root)
        # At least some artifacts discovered when E4/E5 present
        existing = [r for r in rows if r.get("path_exists")]
        self.assertGreater(len(existing), 0)

    def test_exact_id_join_rejects_pair_only(self):
        candidates = [
            {
                "clean_forward_candidate_id": "cf_abc",
                "pair_address": "0x01DE8B4a7A04012b7BB0a2cD1b5113877F24c3a0",
                "provider_payload_hash": "h",
                "source_clean_forward_row_key": "k",
            }
        ]
        # Synthetic discovery row claiming pair overlap
        discovery = [
            {
                "path": "nonexistent.parquet",
                "path_exists": True,
                "artifact_type": "prediction",
                "artifact_family": "RF",
                "load_success": True,
                "file_size_bytes": "100",
                "columns": "candidate_id|pair_address|predicted_probability",
            }
        ]
        # path doesn't exist -> read errors / not found paths still reject pair join
        rows, summary = audit_exact_id_joins(
            project_root=ROOT, candidates=candidates, discovery_rows=discovery
        )
        pair_rows = [r for r in rows if r["tested_join_key"] == "pair_address"]
        self.assertTrue(pair_rows)
        self.assertTrue(all(r["exact_join_safe"] is False for r in pair_rows))
        rf_pair = [r for r in pair_rows if r["model_family"] == "RF"]
        self.assertTrue(rf_pair)
        self.assertTrue(all(r["rejection_reason"] == "PAIR_TIMESTAMP_JOIN_REJECTED" for r in rf_pair))
        self.assertFalse(summary["any_exact_join_safe"])

    def test_feature_parity_fails_without_full_schema_fields(self):
        candidates = [
            {
                "clean_forward_candidate_id": "c1",
                "pair_address": "p",
                "price_usd": "1.0",
                "liquidity_usd": "1000",
                "volume_24h": "100",
                "txns_buys_24h": "10",
                "txns_sells_24h": "5",
                "price_change_m5": "0.1",
                "price_change_h1": "0.2",
                "price_change_h6": "0.3",
                "price_change_h24": "0.4",
            }
        ]
        rows, _manifest = discover_direct_target_artifacts(ROOT)
        parity_rows, _compat, _matrix, summary = audit_feature_parity(
            project_root=ROOT, candidates=candidates, discovery_rows=rows
        )
        # If E4 schemas exist, RF/XGB parity must fail on missing features
        rf = next((r for r in parity_rows if r["model_family"] == "RF"), None)
        if rf and rf["schema_artifact_exists"]:
            self.assertFalse(rf["feature_parity_passed"])
            self.assertFalse(rf["inference_allowed"])
            self.assertGreater(str(rf["missing_required_features"]).count("|") + 1, 5)
        self.assertFalse(summary["any_inference_allowed"])

    def test_attachment_v2_no_invented_scores(self):
        candidates = [{"clean_forward_candidate_id": "c1", "pair_address": "p", "base_token_address": "b", "quote_token_address": "q", "provider_pair_url": "u", "provider_payload_hash": "h"}]
        decision_by = {"c1": {"clean_forward_decision_input_id": "d1"}}
        join_summary = {
            "any_exact_join_safe": False,
            "by_family": {
                "RF": {"exact_join_safe": False, "rejection_reason": "EXACT_ID_JOIN_NOT_AVAILABLE"},
                "XGB": {"exact_join_safe": False, "rejection_reason": "EXACT_ID_JOIN_NOT_AVAILABLE"},
                "TAB": {"exact_join_safe": False, "rejection_reason": "EXACT_ID_JOIN_NOT_AVAILABLE"},
            },
        }
        parity_summary = {
            "any_feature_parity_passed": False,
            "any_inference_allowed": False,
            "by_family": {
                "RF": {
                    "feature_parity_passed": False,
                    "inference_allowed": False,
                    "model_artifact_exists": True,
                    "missing_required_feature_count": 20,
                    "required_feature_count": 31,
                    "blocker_reason": "FEATURE_PARITY_NOT_APPROVED",
                },
                "XGB": {
                    "feature_parity_passed": False,
                    "inference_allowed": False,
                    "model_artifact_exists": True,
                    "missing_required_feature_count": 20,
                    "required_feature_count": 31,
                    "blocker_reason": "FEATURE_PARITY_NOT_APPROVED",
                },
                "TAB": {
                    "feature_parity_passed": False,
                    "inference_allowed": False,
                    "model_artifact_exists": False,
                    "missing_required_feature_count": 20,
                    "required_feature_count": 31,
                    "blocker_reason": "FEATURE_PARITY_NOT_APPROVED",
                },
            },
        }
        attachments, vote_rows = build_attachment_v2(
            candidates=candidates,
            decision_by_candidate=decision_by,
            join_summary=join_summary,
            parity_summary=parity_summary,
            discovery_rows=[],
        )
        self.assertEqual(len(attachments), 3)
        for a in attachments:
            self.assertFalse(a["evidence_attached"])
            self.assertIsNone(a["score"])
            self.assertIsNone(a["model_vote"])
            self.assertNotEqual(a["score"], 0)
        self.assertTrue(all(v["votes_allowed"] is False for v in vote_rows))

        classification = decide_completion_classification(
            join_summary=join_summary,
            parity_summary=parity_summary,
            attachments=attachments,
            invented_ok=True,
            legacy_ok=True,
            authority_ok=True,
        )
        self.assertEqual(classification, "AE16_BLOCKED_FEATURE_PARITY_GAP")

    def test_runner_smoke_writes_required_outputs(self):
        cleaned = ROOT / "data" / "audits" / "ae15_cleaned_for_ae16_20260722_194200" / "data"
        if not (cleaned / "ae16_clean_forward_candidates.csv").is_file():
            self.skipTest("cleaned AE16 inputs missing")
        mod = _load_runner()
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "out"
            result = mod.run_ae16_bridge_completion(
                mod.parse_args(["--input-root", str(cleaned), "--output-root", str(out)])
            )
            self.assertTrue(str(result["classification"]).startswith("AE16_"))
            required = [
                "reports/ae16_model_evidence_discovery_manifest.json",
                "data/ae16_discovered_model_assets.csv",
                "audits/ae16_direct_target_artifact_discovery_audit.csv",
                "audits/ae16_exact_id_join_audit.csv",
                "reports/ae16_exact_id_join_summary.json",
                "data/ae16_clean_forward_feature_matrix.csv",
                "audits/ae16_feature_parity_audit.csv",
                "audits/ae16_model_artifact_compatibility_audit.csv",
                "data/ae16_model_evidence_attachment_v2.csv",
                "audits/ae16_vote_policy_audit.csv",
                "data/ae16_clean_forward_consensus_decisions_v2.csv",
                "data/ae16_clean_forward_consensus_decisions_v2.jsonl",
                "data/ae16_consensus_tier_summary_v2.csv",
                "audits/ae16_consensus_tier_logic_audit_v2.csv",
                "audits/ae16_no_invented_scores_audit_v2.json",
                "audits/ae16_missing_score_handling_audit_v2.csv",
                "audits/ae16_no_legacy_source_audit_v2.json",
                "audits/ae16_authority_audit_v2.json",
                "reports/ae16_completion_decision_gate.json",
                "reports/ae16_completion_summary_for_upload.txt",
            ]
            for rel in required:
                path = out / rel
                self.assertTrue(path.is_file(), f"missing {rel} under {out}; present={sorted(p.name for p in path.parent.iterdir()) if path.parent.exists() else 'NO_DIR'}")
            gate = json.loads((out / "reports" / "ae16_completion_decision_gate.json").read_text(encoding="utf-8"))
            self.assertFalse(gate["trade_authority"])
            self.assertFalse(gate["live_trading_ready"])
            self.assertFalse(gate["ae16_can_close_as_original_e6_repair"])
            self.assertEqual(gate["model_evidence_attached_counts"]["RF"], 0)
            self.assertEqual(gate["model_evidence_attached_counts"]["XGB"], 0)
            self.assertEqual(gate["model_evidence_attached_counts"]["TAB"], 0)


if __name__ == "__main__":
    unittest.main()
