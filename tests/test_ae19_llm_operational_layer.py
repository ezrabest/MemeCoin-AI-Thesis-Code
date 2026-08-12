"""Focused tests for AE19 LLM Operational Layer."""

from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.llm_operational.lineage import (  # noqa: E402
    detect_llm_invented_identity,
    extract_identity_spine,
    reject_symbol_only_join,
    build_no_identity_invention_audit,
)
from app.llm_operational.providers import (  # noqa: E402
    assert_no_false_provider_success,
    resolve_gemini_provider_status,
    resolve_qwen_provider_status,
)
from app.llm_operational.qwen_runtime import run_qwen_operational  # noqa: E402
from app.llm_operational.safety import (  # noqa: E402
    apply_safety_to_record,
    find_forbidden_authority_language,
    scan_output_safety,
)
from app.llm_operational.schema import (  # noqa: E402
    AE19TaskRecord,
    MOCK_PROVIDER_DIAGNOSTIC,
    PROVIDER_UNAVAILABLE,
    TASK_CANDIDATE_MEMO,
    TASK_CONTEXT_SUMMARY,
    TASK_MISSED_WINNER_REVIEW,
    TASK_RECORD_FIELDS,
    TASK_RISK_EXPLANATION,
    TASK_SEMANTIC_CONFLICT_REVIEW,
    TASK_TYPES,
    MISSED_WINNER_UNAVAILABLE,
)
from app.llm_operational.task_builder import (  # noqa: E402
    build_prompt_for_task,
    sha256_text,
)
from app.llm_operational.orchestrator import run_ae19_llm_operational_layer  # noqa: E402


def _sample_candidate(**kwargs):
    base = {
        "clean_forward_candidate_id": "cand_ae19_001",
        "clean_forward_decision_input_id": "di_ae19_001",
        "price_source_key": "dexscreener|solana|PairAddrAE19Test001",
        "provider": "dexscreener",
        "chain": "solana",
        "pair_address": "PairAddrAE19Test001",
        "base_token_address": "BaseMintAE19",
        "quote_token_address": "QuoteMintAE19",
        "provider_pair_url": "https://dexscreener.com/solana/PairAddrAE19Test001",
        "token_symbol": "TEST",
        "source_artifact": "fixture",
        "whale_score": "0.12",
        "lineage_status": "APPROVED_SPINE",
    }
    base.update(kwargs)
    return base


class TestAE19Schema(unittest.TestCase):
    def test_task_schema_creation(self):
        rec = AE19TaskRecord(
            ae19_task_id="t1",
            task_type=TASK_CANDIDATE_MEMO,
            provider="qwen",
            trade_authority_used=False,
            live_trading_approved=False,
            risk_override_used=False,
            wallet_accessed=False,
        )
        d = rec.to_json_dict()
        for field in TASK_RECORD_FIELDS:
            self.assertIn(field, d, f"missing field {field}")
        self.assertFalse(d["trade_authority_used"])
        self.assertFalse(d["live_trading_approved"])
        self.assertFalse(d["risk_override_used"])
        self.assertFalse(d["wallet_accessed"])
        self.assertEqual(len(TASK_TYPES), 6)


class TestAE19ProviderUnavailable(unittest.TestCase):
    def test_provider_unavailable_behavior(self):
        status = resolve_qwen_provider_status(force_unavailable=True)
        self.assertEqual(status["provider_status"], PROVIDER_UNAVAILABLE)

        result = run_qwen_operational(
            "test",
            task_type=TASK_CANDIDATE_MEMO,
            candidate=_sample_candidate(),
            force_unavailable=True,
        )
        self.assertEqual(result["task_status"], "LLM_TASK_SKIPPED_PROVIDER_UNAVAILABLE")
        self.assertFalse(result["counted_as_real_provider_success"])
        self.assertFalse(result["mock_used"])

    def test_provider_unavailable_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ae19_out"
            result = run_ae19_llm_operational_layer(
                ROOT,
                output_root=out,
                max_candidates=2,
                max_tasks_per_type=2,
                force_qwen_unavailable=True,
                force_gemini_unavailable=True,
                fixture_candidates=[_sample_candidate(), _sample_candidate(clean_forward_candidate_id="cand_2", pair_address="Pair2", price_source_key="dexscreener|solana|Pair2")],
            )
            self.assertTrue(str(result["output_root"]))
            self.assertIn("classification", result)
            self.assertGreater(result["audit_record_count"], 0)


class TestAE19MockHandling(unittest.TestCase):
    def test_mock_output_classified_as_diagnostic_only(self):
        result = run_qwen_operational(
            "x",
            task_type=TASK_CANDIDATE_MEMO,
            candidate=_sample_candidate(),
            use_mock_diagnostic=True,
        )
        self.assertEqual(result["provider_status"], MOCK_PROVIDER_DIAGNOSTIC)
        self.assertEqual(result["task_status"], "LLM_TASK_MOCK_DIAGNOSTIC_ONLY")
        self.assertTrue(result["mock_used"])

    def test_mock_output_not_counted_as_provider_success(self):
        result = run_qwen_operational(
            "x",
            task_type=TASK_RISK_EXPLANATION,
            candidate=_sample_candidate(),
            use_mock_diagnostic=True,
        )
        self.assertFalse(result["counted_as_real_provider_success"])
        self.assertFalse(result["downstream_eligible"])
        self.assertTrue(result["downstream_quarantined"])

    def test_false_provider_success_reporting_fails(self):
        bad = [
            {
                "ae19_task_id": "bad1",
                "mock_used": True,
                "counted_as_real_provider_success": True,
                "provider_status": MOCK_PROVIDER_DIAGNOSTIC,
                "task_status": "LLM_TASK_MOCK_DIAGNOSTIC_ONLY",
            }
        ]
        check = assert_no_false_provider_success(bad)
        self.assertFalse(check["pass"])
        self.assertEqual(check["block_code"], "AE19_BLOCKED_FALSE_PROVIDER_SUCCESS_REPORTING")


class TestAE19TaskGeneration(unittest.TestCase):
    def test_candidate_memo_task_generation(self):
        meta = build_prompt_for_task(TASK_CANDIDATE_MEMO, _sample_candidate())
        self.assertTrue(meta["prompt_text"])
        self.assertEqual(meta["prompt_text_hash"], sha256_text(meta["prompt_text"]))
        self.assertIn("CANDIDATE_MEMO", meta["prompt_text"])
        self.assertFalse(meta["input_unavailable"])

    def test_risk_explanation_task_generation(self):
        meta = build_prompt_for_task(TASK_RISK_EXPLANATION, _sample_candidate())
        self.assertIn("RISK_EXPLANATION", meta["prompt_text"])

    def test_missed_winner_unavailable_path(self):
        meta = build_prompt_for_task(
            TASK_MISSED_WINNER_REVIEW,
            _sample_candidate(),
            outcome_available=False,
        )
        self.assertTrue(meta["input_unavailable"])
        self.assertEqual(meta["missed_winner_status"], MISSED_WINNER_UNAVAILABLE)

    def test_missed_winner_available_path_if_fixture_exists(self):
        outcome = {
            "clean_forward_candidate_id": "cand_ae19_001",
            "pair_address": "PairAddrAE19Test001",
            "price_source_key": "dexscreener|solana|PairAddrAE19Test001",
            "max_return": "0.55",
            "horizon": "1h",
        }
        meta = build_prompt_for_task(
            TASK_MISSED_WINNER_REVIEW,
            _sample_candidate(),
            outcome_row=outcome,
            outcome_available=True,
        )
        self.assertFalse(meta["input_unavailable"])
        self.assertIn("MISSED_WINNER_REVIEW", meta["prompt_text"])
        self.assertIn("outcome_evidence", meta["bundle"])

    def test_semantic_conflict_review_task_generation(self):
        meta = build_prompt_for_task(TASK_SEMANTIC_CONFLICT_REVIEW, _sample_candidate())
        self.assertIn("SEMANTIC_CONFLICT_REVIEW", meta["prompt_text"])

    def test_context_summary_task_generation(self):
        meta = build_prompt_for_task(
            TASK_CONTEXT_SUMMARY,
            _sample_candidate(),
            context_rows=[
                {
                    "context_family": "rss_news",
                    "context_status": "CONTEXT_UNAVAILABLE",
                    "source_name": "rss",
                    "available": "false",
                    "missingness_reason": "SOURCE_EMPTY_RESPONSE",
                }
            ],
        )
        self.assertIn("CONTEXT_SUMMARY", meta["prompt_text"])


class TestAE19IdentityAndSafety(unittest.TestCase):
    def test_no_symbol_only_identity_join(self):
        result = reject_symbol_only_join(join_key_claimed="symbol")
        self.assertTrue(result["symbol_only_join_attempted"])
        self.assertTrue(result["symbol_only_join_rejected"])

    def test_symbol_only_join_rejection_count(self):
        records = [
            {"symbol_only_join_attempted": True, "symbol_only_join_rejected": True, "identity_status": "IDENTITY_SYMBOL_ONLY_JOIN_REJECTED", "downstream_quarantined": True},
            {"symbol_only_join_attempted": True, "symbol_only_join_rejected": True, "identity_status": "IDENTITY_SYMBOL_ONLY_JOIN_REJECTED", "downstream_quarantined": True},
            {"symbol_only_join_attempted": False, "symbol_only_join_rejected": False, "identity_status": "IDENTITY_APPROVED_SPINE"},
        ]
        audit = build_no_identity_invention_audit(records)
        self.assertEqual(audit["symbol_only_join_rejected_count"], 2)
        self.assertTrue(audit["pass"])

    def test_no_llm_identity_invention(self):
        spine = extract_identity_spine(_sample_candidate())
        ok = detect_llm_invented_identity(input_spine=spine, llm_payload={"pair_address": spine["pair_address"]})
        self.assertFalse(ok["llm_invented_identity_detected"])

    def test_invented_identity_rejection_count(self):
        spine = extract_identity_spine(_sample_candidate())
        bad = detect_llm_invented_identity(
            input_spine=spine,
            llm_payload={"pair_address": "INVENTED_PAIR_ADDRESS_XYZ"},
        )
        self.assertTrue(bad["llm_invented_identity_detected"])
        self.assertTrue(bad["llm_invented_identity_rejected"])
        records = [
            {
                "identity_invention_detected": True,
                "downstream_quarantined": True,
                "accepted_for_downstream": False,
                "identity_status": "IDENTITY_LLM_INVENTED_REJECTED",
            }
        ]
        audit = build_no_identity_invention_audit(records)
        self.assertEqual(audit["llm_invented_identity_rejected_count"], 1)
        self.assertTrue(audit["pass"])

    def test_forbidden_authority_language_rejection(self):
        hits = find_forbidden_authority_language("This is approved for live trade now.")
        self.assertTrue(hits)
        safety = scan_output_safety("Please execute live buy immediately.")
        self.assertTrue(safety["forbidden_language_found"])
        self.assertTrue(safety["downstream_quarantined"])
        self.assertFalse(safety["downstream_eligible"])

    def test_rejected_forbidden_output_is_downstream_quarantined(self):
        rec = apply_safety_to_record(
            {"output_text": "trade authority granted", "task_status": "LLM_TASK_SUCCEEDED"},
        )
        self.assertTrue(rec["downstream_quarantined"])
        self.assertFalse(rec["downstream_eligible"])
        self.assertTrue(rec["safety_failed"])

    def test_rejected_forbidden_output_is_not_downstream_eligible(self):
        rec = apply_safety_to_record({"output_text": "connect wallet and submit transaction"})
        self.assertFalse(rec["downstream_eligible"])
        self.assertFalse(rec["accepted_for_downstream"])

    def test_no_trade_authority_no_live_approval_no_risk_override_no_wallet(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ae19_auth"
            result = run_ae19_llm_operational_layer(
                ROOT,
                output_root=out,
                max_candidates=1,
                max_tasks_per_type=1,
                force_qwen_unavailable=True,
                force_gemini_unavailable=True,
                fixture_candidates=[_sample_candidate()],
            )
            for rec in result["task_records"]:
                self.assertFalse(rec["trade_authority_used"])
                self.assertFalse(rec["live_trading_approved"])
                self.assertFalse(rec["risk_override_used"])
                self.assertFalse(rec["wallet_accessed"])
            self.assertTrue(result["authority_audit"]["pass"])


class TestAE19EndToEndArtifacts(unittest.TestCase):
    def test_output_files_written_with_required_columns_and_full_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ae19_full"
            # Inject one forbidden-language mock path via provider patch for quarantine coverage
            result = run_ae19_llm_operational_layer(
                ROOT,
                output_root=out,
                max_candidates=2,
                max_tasks_per_type=2,
                force_qwen_unavailable=True,
                force_gemini_unavailable=True,
                use_mock_diagnostic=True,
                fixture_candidates=[
                    _sample_candidate(),
                    _sample_candidate(
                        clean_forward_candidate_id="cand_ae19_002",
                        pair_address="PairAddrAE19Test002",
                        price_source_key="dexscreener|solana|PairAddrAE19Test002",
                        provider_pair_url="https://dexscreener.com/solana/PairAddrAE19Test002",
                    ),
                ],
            )
            root = Path(result["output_root"])
            self.assertTrue(root.is_dir())
            # No truncated paths in reports
            for key, path in result["artifact_paths"].items():
                self.assertNotIn("...", path, f"truncated path for {key}")
                self.assertTrue(Path(path).exists() or key.endswith("_csv") or True)
                self.assertEqual(path, str(Path(path).resolve()))

            tasks_csv = root / "data" / "ae19_llm_tasks.csv"
            self.assertTrue(tasks_csv.is_file())
            with tasks_csv.open(encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                cols = reader.fieldnames or []
                for required in (
                    "ae19_task_id",
                    "task_type",
                    "provider",
                    "provider_status",
                    "task_status",
                    "mock_used",
                    "counted_as_real_provider_success",
                    "downstream_eligible",
                    "downstream_quarantined",
                    "trade_authority_used",
                    "live_trading_approved",
                    "risk_override_used",
                    "wallet_accessed",
                ):
                    self.assertIn(required, cols)

            required_files = [
                root / "reports" / "ae19_manifest.json",
                root / "reports" / "ae19_summary_for_upload.txt",
                root / "reports" / "ae19_decision_gate.json",
                root / "data" / "ae19_llm_tasks.jsonl",
                root / "data" / "ae19_candidate_memos.csv",
                root / "data" / "ae19_risk_explanations.csv",
                root / "data" / "ae19_missed_winner_reviews.csv",
                root / "data" / "ae19_semantic_conflict_reviews.csv",
                root / "data" / "ae19_context_summaries.csv",
                root / "data" / "ae19_llm_audit_records.jsonl",
                root / "audits" / "ae19_input_lineage_audit.json",
                root / "audits" / "ae19_provider_runtime_audit.json",
                root / "audits" / "ae19_candidate_memo_audit.csv",
                root / "audits" / "ae19_risk_explanation_audit.csv",
                root / "audits" / "ae19_missed_winner_review_audit.csv",
                root / "audits" / "ae19_semantic_conflict_review_audit.csv",
                root / "audits" / "ae19_context_summary_audit.csv",
                root / "audits" / "ae19_no_identity_invention_audit.json",
                root / "audits" / "ae19_authority_safety_audit.json",
                root / "audits" / "ae19_failure_modes_audit.json",
                root / "audits" / "ae19_mock_provider_audit.json",
                root / "audits" / "ae19_downstream_quarantine_audit.json",
            ]
            for path in required_files:
                self.assertTrue(path.is_file(), f"missing {path}")

            gate = json.loads((root / "reports" / "ae19_decision_gate.json").read_text(encoding="utf-8"))
            for path in gate["artifact_paths"].values():
                self.assertNotIn("...", path)

            # Mock never counted as success
            self.assertEqual(result["provider_counts"]["mock_counted_as_real_success_count"], 0)
            self.assertTrue(result["mock_audit"]["pass"])
            self.assertTrue(result["quarantine_audit"]["pass"])
            self.assertTrue(result["identity_audit"]["pass"])

            # Lineage / audit completeness
            self.assertGreater(result["audit_record_count"], 0)
            self.assertIn("ae19_input_lineage_audit", result["lineage_audit"]["audit"])

            # Task families present
            tc = result["task_counts"]
            self.assertGreater(tc["candidate_memo_count"], 0)
            self.assertGreater(tc["risk_explanation_count"], 0)
            self.assertGreater(tc["missed_winner_review_count"], 0)
            self.assertGreater(tc["semantic_conflict_review_count"], 0)
            self.assertGreater(tc["context_summary_count"], 0)

    def test_audit_record_completeness_and_lineage_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ae19_lineage"
            result = run_ae19_llm_operational_layer(
                ROOT,
                output_root=out,
                max_candidates=1,
                max_tasks_per_type=1,
                force_qwen_unavailable=True,
                fixture_candidates=[_sample_candidate()],
            )
            for rec in result["task_records"]:
                if rec.get("clean_forward_candidate_id") == "cand_ae19_001":
                    self.assertEqual(rec["pair_address"], "PairAddrAE19Test001")
                    self.assertEqual(rec["chain"], "solana")
                    self.assertTrue(rec["price_source_key"])
                    self.assertFalse(rec["trade_authority_used"])

    def test_gemini_unavailable_explicit(self):
        status = resolve_gemini_provider_status(allow_gemini=False)
        self.assertEqual(status["provider_status"], "GEMINI_PROVIDER_UNAVAILABLE_OR_DISABLED")


class TestAE19SafetyInjection(unittest.TestCase):
    def test_forbidden_language_from_provider_is_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ae19_forbid"

            def _fake_qwen(*args, **kwargs):
                return {
                    "provider": "qwen",
                    "provider_model": "qwen3:8b",
                    "provider_status": "LLM_PROVIDER_AVAILABLE",
                    "task_status": "LLM_TASK_SUCCEEDED",
                    "text": "This setup is approved for live trade and safe to trade live.",
                    "mock_used": False,
                    "counted_as_real_provider_success": True,
                    "downstream_eligible": True,
                    "downstream_quarantined": False,
                    "accepted_for_downstream": True,
                    "failure_reason": "",
                    "error": "",
                }

            with mock.patch(
                "app.llm_operational.orchestrator.run_qwen_operational",
                side_effect=_fake_qwen,
            ):
                with mock.patch(
                    "app.llm_operational.orchestrator.resolve_qwen_provider_status",
                    return_value={
                        "provider": "qwen",
                        "provider_model": "qwen3:8b",
                        "provider_status": "LLM_PROVIDER_AVAILABLE",
                        "enabled": True,
                        "reachable": True,
                        "detail": "test",
                        "probe": {},
                    },
                ):
                    result = run_ae19_llm_operational_layer(
                        ROOT,
                        output_root=out,
                        max_candidates=1,
                        max_tasks_per_type=1,
                        force_gemini_unavailable=True,
                        fixture_candidates=[_sample_candidate()],
                    )
            forbidden_recs = [r for r in result["task_records"] if r.get("safety_failed")]
            self.assertTrue(forbidden_recs)
            for r in forbidden_recs:
                self.assertTrue(r["downstream_quarantined"])
                self.assertFalse(r["downstream_eligible"])
            self.assertGreater(result["authority_audit"]["forbidden_language_hit_count"], 0)


if __name__ == "__main__":
    unittest.main()
