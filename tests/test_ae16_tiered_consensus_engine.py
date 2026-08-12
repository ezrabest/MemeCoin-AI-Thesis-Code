"""Focused unit tests for AE16 Tiered Consensus Engine."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.consensus.audits import (  # noqa: E402
    audit_authority,
    audit_input_contract,
    audit_no_invented_scores,
    audit_no_legacy_source,
    run_input_path_preflight,
)
from app.consensus.model_evidence import (  # noqa: E402
    AttachmentResult,
    DiscoveredArtifact,
    attach_model_evidence_for_candidate,
    parse_numeric_score,
)
from app.consensus.tiered_engine import (  # noqa: E402
    assign_consensus_tier,
    build_consensus_decision,
    make_synthetic_attachment,
    model_may_vote,
)


def _load_runner():
    path = ROOT / "scripts" / "run_ae16_tiered_consensus_engine.py"
    spec = importlib.util.spec_from_file_location("run_ae16_tiered_consensus_engine", path)
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


def _minimal_package(tmpdir: Path, *, candidates: int = 2) -> Path:
    data = tmpdir / "data"
    data.mkdir(parents=True, exist_ok=True)
    cand_fields = [
        "clean_forward_candidate_id",
        "pair_address",
        "base_token_address",
        "quote_token_address",
        "provider_pair_url",
        "provider_payload_hash",
        "verification_status",
        "freshness_status",
        "identity_status",
        "clean_feed_eligible",
        "paper_demo_only",
        "live_trading_ready",
        "observed_at",
        "fetched_at",
        "ingested_at",
    ]
    di_fields = [
        "clean_forward_decision_input_id",
        "clean_forward_candidate_id",
        "model_scores_available",
        "xgb_score",
        "tab_score",
        "rf_score",
        "model_score_source_status",
        "consensus_tier_shadow",
        "context_status",
        "llm_status",
    ]
    out_fields = [
        "clean_forward_outcome_label_id",
        "clean_forward_candidate_id",
        "clean_forward_decision_input_id",
    ]
    link_fields = [
        "execution_link_id",
        "clean_forward_candidate_id",
        "clean_forward_decision_input_id",
        "paper_order_id",
        "paper_position_id",
        "one_order_to_one_position_passed",
        "pair_address",
        "base_token_address",
        "quote_token_address",
        "provider_payload_hash",
    ]

    cands = []
    dis = []
    outs = []
    for i in range(candidates):
        cid = f"cand_{i}"
        did = f"dec_{i}"
        cands.append(
            {
                "clean_forward_candidate_id": cid,
                "pair_address": f"Pair{i}",
                "base_token_address": f"Base{i}",
                "quote_token_address": f"Quote{i}",
                "provider_pair_url": f"https://example/{i}",
                "provider_payload_hash": f"hash{i}",
                "verification_status": "provider_pair_verified",
                "freshness_status": "fresh",
                "identity_status": "pair_and_tokens_separated",
                "clean_feed_eligible": "true",
                "paper_demo_only": "true",
                "live_trading_ready": "false",
                "observed_at": "2026-07-22T00:00:00+00:00",
                "fetched_at": "2026-07-22T00:00:00+00:00",
                "ingested_at": "2026-07-22T00:00:00+00:00",
            }
        )
        dis.append(
            {
                "clean_forward_decision_input_id": did,
                "clean_forward_candidate_id": cid,
                "model_scores_available": "False",
                "xgb_score": "",
                "tab_score": "",
                "rf_score": "",
                "model_score_source_status": "AE15_SCHEMA_ONLY_NO_MODEL_AUTHORITY",
                "consensus_tier_shadow": "",
                "context_status": "AE15_CONTEXT_NOT_EXECUTED",
                "llm_status": "AE15_LLM_NOT_CALLED",
            }
        )
        outs.append(
            {
                "clean_forward_outcome_label_id": f"out_{i}",
                "clean_forward_candidate_id": cid,
                "clean_forward_decision_input_id": did,
            }
        )

    # Contract expects 961 — for unit tests of contract helper we build exact expected
    # separately. This helper is for runner smoke with overridden expected counts via
    # missing-file / adapter path tests only.
    _write_csv(data / "ae16_clean_forward_candidates.csv", cands, cand_fields)
    _write_csv(data / "ae16_clean_forward_decision_inputs.csv", dis, di_fields)
    _write_csv(data / "ae16_clean_forward_outcome_label_contract.csv", outs, out_fields)
    _write_csv(
        data / "ae16_clean_forward_paper_execution_links.csv",
        [
            {
                "execution_link_id": "link_0",
                "clean_forward_candidate_id": "cand_0",
                "clean_forward_decision_input_id": "dec_0",
                "paper_order_id": "order_1",
                "paper_position_id": "1",
                "one_order_to_one_position_passed": "True",
                "pair_address": "Pair0",
                "base_token_address": "Base0",
                "quote_token_address": "Quote0",
                "provider_payload_hash": "hash0",
            }
        ],
        link_fields,
    )
    return data


class TestAE16Preflight(unittest.TestCase):
    def test_01_required_input_path_preflight(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = _minimal_package(root, candidates=2)
            result = run_input_path_preflight(data)
            self.assertTrue(result["passed"])
            self.assertEqual(len(result["rows"]), 4)
            for row in result["rows"]:
                self.assertTrue(row["exists"])
                self.assertFalse(row["blocking"])
                self.assertIn("file_size_bytes", row)

    def test_02_missing_input_file_blocks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            data = _minimal_package(root, candidates=2)
            (data / "ae16_clean_forward_candidates.csv").unlink()
            result = run_input_path_preflight(data)
            self.assertFalse(result["passed"])
            self.assertIn("ae16_clean_forward_candidates.csv", result["missing_files"])

            mod = _load_runner()
            out = root / "out"
            args = mod.parse_args(["--input-root", str(data), "--output-root", str(out)])
            run_result = mod.run_ae16(args)
            self.assertEqual(run_result["classification"], "AE16_BLOCKED_INPUT_FILES_MISSING")
            self.assertTrue((out / "reports" / "ae16_decision_gate.json").is_file())
            self.assertTrue((out / "reports" / "ae16_manifest.json").is_file())
            self.assertTrue((out / "reports" / "ae16_summary_for_upload.txt").is_file())
            self.assertTrue((out / "audits" / "ae16_input_path_preflight_audit.csv").is_file())


class TestAE16InputContract(unittest.TestCase):
    def test_03_input_package_validation(self):
        # Build a package matching expected counts by expanding minimal rows — too heavy.
        # Instead validate the audit helper logic with synthetic matching counts.
        from app.consensus import EXPECTED_INPUT_COUNTS

        n = EXPECTED_INPUT_COUNTS["candidates"]
        candidates = []
        decision_inputs = []
        outcomes = []
        for i in range(n):
            cid = f"c{i}"
            did = f"d{i}"
            candidates.append(
                {
                    "clean_forward_candidate_id": cid,
                    "pair_address": "P",
                    "base_token_address": "B",
                    "quote_token_address": "Q",
                    "provider_pair_url": "u",
                    "provider_payload_hash": "h",
                    "verification_status": "provider_pair_verified",
                    "freshness_status": "fresh",
                    "identity_status": "pair_and_tokens_separated",
                    "clean_feed_eligible": True,
                    "paper_demo_only": True,
                    "live_trading_ready": False,
                }
            )
            decision_inputs.append(
                {
                    "clean_forward_decision_input_id": did,
                    "clean_forward_candidate_id": cid,
                    "model_scores_available": "False",
                    "xgb_score": "",
                    "tab_score": "",
                    "rf_score": "",
                }
            )
            outcomes.append(
                {
                    "clean_forward_outcome_label_id": f"o{i}",
                    "clean_forward_candidate_id": cid,
                    "clean_forward_decision_input_id": did,
                }
            )
        links = [
            {
                "execution_link_id": "e1",
                "clean_forward_candidate_id": "c0",
                "paper_order_id": "ord",
                "paper_position_id": "1",
                "one_order_to_one_position_passed": True,
            }
        ]
        audit = audit_input_contract(
            candidates=candidates,
            decision_inputs=decision_inputs,
            outcomes=outcomes,
            execution_links=links,
        )
        self.assertTrue(audit["passed"], audit.get("failures"))


class TestAE16Scores(unittest.TestCase):
    def test_04_score_null_handling(self):
        self.assertIsNone(parse_numeric_score(""))
        self.assertIsNone(parse_numeric_score(None))
        self.assertIsNone(parse_numeric_score("null"))
        self.assertIsNone(parse_numeric_score("NaN"))

    def test_05_no_score_defaulting(self):
        # Empty must not become 0
        self.assertIsNone(parse_numeric_score(""))
        self.assertNotEqual(parse_numeric_score(""), 0)
        # Explicit zero remains zero only when provided
        self.assertEqual(parse_numeric_score("0"), 0.0)
        self.assertEqual(parse_numeric_score(0), 0.0)

    def test_06_no_invented_scores(self):
        attachments = [
            make_synthetic_attachment(
                candidate_id="c1", decision_id="d1", family="RF", score=None, attached=False
            ),
            make_synthetic_attachment(
                candidate_id="c1", decision_id="d1", family="XGB", score=None, attached=False
            ),
            make_synthetic_attachment(
                candidate_id="c1", decision_id="d1", family="TAB", score=None, attached=False
            ),
        ]
        decision_inputs = [
            {
                "clean_forward_candidate_id": "c1",
                "xgb_score": "",
                "tab_score": "",
                "rf_score": "",
                "consensus_tier_shadow": "TAB_XGB_RF_ALL3",
                "model_scores_available": "False",
            }
        ]
        decisions = [
            build_consensus_decision(
                candidate={"clean_forward_candidate_id": "c1", "pair_address": "p"},
                decision=decision_inputs[0],
                attachments_by_family={a.model_family: a for a in attachments},
            )
        ]
        audit, _rows = audit_no_invented_scores(
            decision_inputs=decision_inputs, attachments=attachments, decisions=decisions
        )
        self.assertTrue(audit["passed"], audit.get("violations"))
        self.assertEqual(decisions[0]["rf_score"], None)
        self.assertEqual(decisions[0]["model_vote_count"], 0)

    def test_07_missing_score_does_not_count_as_vote(self):
        a = make_synthetic_attachment(
            candidate_id="c1", decision_id="d1", family="RF", score=None, attached=False
        )
        self.assertFalse(model_may_vote(a))
        # Even if someone sets score=0 without attachment flag
        a2 = AttachmentResult(
            clean_forward_candidate_id="c1",
            clean_forward_decision_input_id="d1",
            pair_address="",
            base_token_address="",
            quote_token_address="",
            model_family="RF",
            evidence_attached=False,
            score=0.0,
            rank=None,
            percentile_rank=None,
            source_artifact_path="",
            source_run_id="",
            source_prediction_file="",
            source_model_artifact="",
            candidate_policy_id="",
            target_row_id="",
            target_name="",
            target_version="",
            horizon="",
            filter_name="",
            exit_policy_id="",
            evidence_type="",
            attachment_status="MODEL_EVIDENCE_UNAVAILABLE",
            attachment_failure_reason="missing",
        )
        self.assertFalse(model_may_vote(a2))


class TestAE16Adapter(unittest.TestCase):
    def test_08_adapter_missing_artifact(self):
        result = attach_model_evidence_for_candidate(
            candidate={"clean_forward_candidate_id": "c1", "pair_address": "p"},
            decision={"clean_forward_decision_input_id": "d1", "model_scores_available": "False"},
            model_family="RF",
            discovered={"RF": [], "XGB": [], "TAB": []},
            project_root=ROOT,
        )
        self.assertIn(result.attachment_status, {"ARTIFACT_NOT_FOUND", "MODEL_EVIDENCE_UNAVAILABLE"})
        self.assertFalse(result.evidence_attached)
        self.assertIsNone(result.score)

    def test_09_adapter_exception_captured(self):
        bad = DiscoveredArtifact(
            path=Path("/nonexistent/rf_predictions.csv"),
            model_family="RF",
            role="prediction_table",
            has_exact_id_columns=True,
            has_safe_score_column=True,
            columns=["candidate_policy_id", "predicted_probability"],
        )
        # Force an exception path via a broken discovered object used after pick
        class Boom(list):
            def get(self, *a, **k):  # type: ignore[override]
                raise RuntimeError("boom")

        # Use normal path with unreadable file -> ARTIFACT_READ_ERROR or exception caught
        result = attach_model_evidence_for_candidate(
            candidate={
                "clean_forward_candidate_id": "c1",
                "pair_address": "p",
                "candidate_policy_id": "pol1",
            },
            decision={"clean_forward_decision_input_id": "d1"},
            model_family="RF",
            discovered={"RF": [bad], "XGB": [], "TAB": []},
            project_root=ROOT,
        )
        self.assertIn(
            result.attachment_status,
            {"ARTIFACT_READ_ERROR", "CANDIDATE_ID_NOT_MATCHED", "ATTACHMENT_EXCEPTION_CAUGHT"},
        )
        self.assertFalse(result.evidence_attached)

        # Explicit exception injection
        from app.consensus import model_evidence as me

        original = me._pick_best_prediction

        def _boom(_arts):
            raise RuntimeError("injected")

        me._pick_best_prediction = _boom  # type: ignore[assignment]
        try:
            result2 = attach_model_evidence_for_candidate(
                candidate={"clean_forward_candidate_id": "c1"},
                decision={"clean_forward_decision_input_id": "d1"},
                model_family="XGB",
                discovered={"XGB": [bad], "RF": [], "TAB": []},
                project_root=ROOT,
            )
            self.assertEqual(result2.attachment_status, "ATTACHMENT_EXCEPTION_CAUGHT")
            self.assertFalse(result2.evidence_attached)
        finally:
            me._pick_best_prediction = original  # type: ignore[assignment]


class TestAE16LegacyAndTiers(unittest.TestCase):
    def test_10_no_legacy_source_use(self):
        audit = audit_no_legacy_source(
            used_paths=["data/audits/ae15_cleaned_for_ae16_20260722_194200/data"],
            candidates_from_market_snapshots=False,
            decision_inputs_from_old_feed=False,
        )
        self.assertTrue(audit["passed"])
        blocked = audit_no_legacy_source(
            used_paths=["data/market_snapshots"],
            candidates_from_market_snapshots=True,
        )
        self.assertFalse(blocked["passed"])
        self.assertEqual(blocked["classification_if_failed"], "AE16_BLOCKED_LEGACY_CONTAMINATION")

    def test_11_tier_assignment_only_when_evidence_attached(self):
        tier, _ = assign_consensus_tier(rf_vote=False, xgb_vote=False, tab_vote=False)
        self.assertEqual(tier, "MODEL_EVIDENCE_UNAVAILABLE")

    def test_12_model_evidence_unavailable_when_no_evidence(self):
        attachments = {
            "RF": make_synthetic_attachment(
                candidate_id="c1", decision_id="d1", family="RF", score=None, attached=False
            ),
            "XGB": make_synthetic_attachment(
                candidate_id="c1", decision_id="d1", family="XGB", score=None, attached=False
            ),
            "TAB": make_synthetic_attachment(
                candidate_id="c1", decision_id="d1", family="TAB", score=None, attached=False
            ),
        }
        d = build_consensus_decision(
            candidate={"clean_forward_candidate_id": "c1"},
            decision={"clean_forward_decision_input_id": "d1"},
            attachments_by_family=attachments,
        )
        self.assertEqual(d["consensus_tier"], "MODEL_EVIDENCE_UNAVAILABLE")
        self.assertEqual(d["model_vote_count"], 0)
        self.assertEqual(d["attached_model_count"], 0)

    def test_13_tab_xgb_rf_all3(self):
        attachments = {
            "RF": make_synthetic_attachment(
                candidate_id="c1", decision_id="d1", family="RF", score=0.7
            ),
            "XGB": make_synthetic_attachment(
                candidate_id="c1", decision_id="d1", family="XGB", score=0.8
            ),
            "TAB": make_synthetic_attachment(
                candidate_id="c1", decision_id="d1", family="TAB", score=0.9
            ),
        }
        d = build_consensus_decision(
            candidate={"clean_forward_candidate_id": "c1"},
            decision={"clean_forward_decision_input_id": "d1"},
            attachments_by_family=attachments,
        )
        self.assertEqual(d["consensus_tier"], "TAB_XGB_RF_ALL3")
        self.assertEqual(d["model_vote_count"], 3)

    def test_14_tab_rf_only(self):
        attachments = {
            "RF": make_synthetic_attachment(
                candidate_id="c1", decision_id="d1", family="RF", score=0.7
            ),
            "XGB": make_synthetic_attachment(
                candidate_id="c1", decision_id="d1", family="XGB", score=None, attached=False
            ),
            "TAB": make_synthetic_attachment(
                candidate_id="c1", decision_id="d1", family="TAB", score=0.9
            ),
        }
        d = build_consensus_decision(
            candidate={"clean_forward_candidate_id": "c1"},
            decision={"clean_forward_decision_input_id": "d1"},
            attachments_by_family=attachments,
        )
        self.assertEqual(d["consensus_tier"], "TAB_RF_ONLY")

    def test_15_research_only_tiers(self):
        tab_xgb, reason1 = assign_consensus_tier(rf_vote=False, xgb_vote=True, tab_vote=True)
        xgb_rf, reason2 = assign_consensus_tier(rf_vote=True, xgb_vote=True, tab_vote=False)
        self.assertEqual(tab_xgb, "TAB_XGB_ONLY")
        self.assertEqual(xgb_rf, "XGB_RF_ONLY")
        self.assertIn("Research-only", reason1)
        self.assertIn("Research-only", reason2)


class TestAE16AuthorityAndOutputs(unittest.TestCase):
    def test_16_17_18_authority_flags(self):
        attachments = {
            "RF": make_synthetic_attachment(
                candidate_id="c1", decision_id="d1", family="RF", score=0.7
            ),
            "XGB": make_synthetic_attachment(
                candidate_id="c1", decision_id="d1", family="XGB", score=0.8
            ),
            "TAB": make_synthetic_attachment(
                candidate_id="c1", decision_id="d1", family="TAB", score=0.9
            ),
        }
        d = build_consensus_decision(
            candidate={"clean_forward_candidate_id": "c1", "live_trading_ready": True},
            decision={"clean_forward_decision_input_id": "d1"},
            attachments_by_family=attachments,
        )
        self.assertEqual(d["authority_status"], "RESEARCH_SHADOW_ONLY")
        self.assertFalse(d["live_trading_ready"])
        self.assertFalse(d["trade_authority"])
        self.assertTrue(d["paper_demo_only"])
        auth = audit_authority([d])
        self.assertTrue(auth["passed"])

    def test_19_20_runner_writes_outputs_and_blocked_preflight(self):
        mod = _load_runner()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            # Missing inputs -> blocked but still writes gate/manifest/summary
            empty_in = root / "empty_in"
            empty_in.mkdir()
            out = root / "out_block"
            result = mod.run_ae16(
                mod.parse_args(["--input-root", str(empty_in), "--output-root", str(out)])
            )
            self.assertEqual(result["classification"], "AE16_BLOCKED_INPUT_FILES_MISSING")
            self.assertTrue((out / "reports" / "ae16_manifest.json").is_file())
            self.assertTrue((out / "reports" / "ae16_decision_gate.json").is_file())
            self.assertTrue((out / "reports" / "ae16_summary_for_upload.txt").is_file())

            # Schema-only path with skip discovery against tiny package will fail
            # input contract (counts). Patch EXPECTED via contract test already covered.
            # Full runner against real cleaned package is integration; here verify
            # output writers using a monkeypatched contract by running with skip
            # against real AE16 cleaned inputs if present.
            cleaned = ROOT / "data" / "audits" / "ae15_cleaned_for_ae16_20260722_194200" / "data"
            if cleaned.is_dir() and (cleaned / "ae16_clean_forward_candidates.csv").is_file():
                out2 = root / "out_full"
                result2 = mod.run_ae16(
                    mod.parse_args(
                        [
                            "--input-root",
                            str(cleaned),
                            "--output-root",
                            str(out2),
                            "--skip-artifact-discovery",
                        ]
                    )
                )
                self.assertIn(
                    result2["classification"],
                    {
                        "AE16_TIERED_CONSENSUS_ENGINE_PASS_SCHEMA_ONLY_NO_MODEL_EVIDENCE",
                        "AE16_TIERED_CONSENSUS_ENGINE_PASS_WITH_MODEL_EVIDENCE",
                    },
                )
                required = [
                    "reports/ae16_manifest.json",
                    "reports/ae16_decision_gate.json",
                    "reports/ae16_summary_for_upload.txt",
                    "data/ae16_model_evidence_attachment.csv",
                    "data/ae16_clean_forward_consensus_decisions.csv",
                    "data/ae16_clean_forward_consensus_decisions.jsonl",
                    "data/ae16_consensus_tier_summary.csv",
                    "data/ae16_model_availability_summary.csv",
                    "audits/ae16_input_path_preflight_audit.csv",
                    "audits/ae16_input_contract_audit.json",
                    "audits/ae16_model_evidence_attachment_audit.csv",
                    "audits/ae16_no_invented_scores_audit.json",
                    "audits/ae16_missing_score_handling_audit.csv",
                    "audits/ae16_consensus_tier_logic_audit.csv",
                    "audits/ae16_authority_audit.json",
                    "audits/ae16_no_legacy_source_audit.json",
                ]
                for rel in required:
                    self.assertTrue((out2 / rel).is_file(), f"missing {rel}")
                gate = json.loads((out2 / "reports" / "ae16_decision_gate.json").read_text(encoding="utf-8"))
                self.assertFalse(gate["trade_authority"])
                self.assertFalse(gate["live_trading_ready"])


if __name__ == "__main__":
    unittest.main()
