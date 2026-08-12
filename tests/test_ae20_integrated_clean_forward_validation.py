"""Focused tests for AE20 Integrated Clean Forward Validation."""

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

from app.ae20.clean_forward_inputs import (  # noqa: E402
    classify_candidate_identity,
)
from app.ae20.decisions import derive_strict_exploration  # noqa: E402
from app.ae20.identity_keys import (  # noqa: E402
    INVALID_IDENTITY_LITERALS,
    make_exact_identity_lookup_key,
)
from app.ae20.integrations import (  # noqa: E402
    DEFAULT_AE16_BRIDGE_RELATIVE,
    attach_ae16,
    load_ae16_index,
    resolve_ae16_bridge_source,
    run_ae19_audit_only,
)
from app.ae20.lifecycle import audit_lineage, maybe_create_paper_lifecycle  # noqa: E402
from app.ae20.output_root import allocate_ae20_output_root  # noqa: E402
from app.ae20.pnl import build_pnl_summary  # noqa: E402
from app.ae20.orchestrator import (  # noqa: E402
    compute_unblocked_for_24h,
    decide_classification,
    run_ae20_integrated_clean_forward_validation,
)
from app.consensus.serialization import write_csv  # noqa: E402


def _cf_candidate(**kwargs):
    base = {
        "provider_pair_url_exact": "https://dexscreener.com/solana/PairAddrAE20Test001",
        "canonical_market_identity": "https://dexscreener.com/solana/PairAddrAE20Test001",
        "normalized_provider_pair_url_key": "dexscreener|solana|pairaddrae20test001",
        "price_source_key": "dexscreener|solana|pairaddrae20test001",
        "chain": "solana",
        "pair_address": "PairAddrAE20Test001",
        "candidate_id": "cand_ae20_001",
        "clean_forward_candidate_id": "cand_ae20_001",
        "observed_at": "2026-08-03T12:00:00+00:00",
        "price_usd": "1.25",
        "liquidity_usd": "50000",
        "identity_ok": True,
        "identity_status": "AE20_CLEAN_FORWARD_IDENTITY_OK",
        "legacy_market_snapshots_used": False,
        "symbol_only_join_used": False,
        "market_snapshots_used": False,
    }
    base.update(kwargs)
    return base


class TestOutputRoot(unittest.TestCase):
    def test_collision_safe_microseconds_uuid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out1, audit1 = allocate_ae20_output_root(root)
            out2, audit2 = allocate_ae20_output_root(root)
            self.assertTrue(out1.exists())
            self.assertTrue(out2.exists())
            self.assertNotEqual(out1, out2)
            self.assertTrue(audit1["stamp_has_microseconds"])
            self.assertTrue(audit1["uuid_suffix_present"])
            self.assertFalse(audit1["overwrote_existing"])
            self.assertIn("ae20_integrated_clean_forward_validation_", out1.name)
            # microsecond stamp is 20 chars YYYYMMDDTHHMMSSffffffZ before uuid
            parts = out1.name.replace("ae20_integrated_clean_forward_validation_", "").split("_")
            self.assertGreaterEqual(len(parts[0]), 20)
            self.assertTrue(parts[0].endswith("Z"))

    def test_refuse_overwrite_explicit_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "existing"
            root.mkdir()
            with self.assertRaises(FileExistsError):
                allocate_ae20_output_root(Path(tmp), output_root=root)


class TestCleanForwardIdentity(unittest.TestCase):
    def test_hard_fail_missing_canonical_identity(self):
        out = classify_candidate_identity({"symbol": "DOGE", "chain": "solana"})
        self.assertFalse(out["identity_ok"])
        self.assertIn(out["identity_status"], {
            "AE20_CLEAN_FORWARD_INPUT_FAILURE",
            "AE20_CANDIDATE_IDENTITY_INCOMPLETE",
        })
        self.assertFalse(out["market_snapshots_used"])
        self.assertFalse(out["symbol_only_join_used"])

    def test_ok_with_canonical_fields(self):
        out = classify_candidate_identity(_cf_candidate())
        self.assertTrue(out["identity_ok"])
        self.assertEqual(out["identity_status"], "AE20_CLEAN_FORWARD_IDENTITY_OK")

    def test_no_legacy_contamination_flags(self):
        out = classify_candidate_identity(_cf_candidate())
        self.assertFalse(out["legacy_source_used"])
        self.assertFalse(out["market_snapshots_used"])
        self.assertFalse(out["symbol_only_join_used"])


class TestLLMAuditOnly(unittest.TestCase):
    def test_llm_buy_cannot_authorize_execution(self):
        ae16 = {"ae16_status": "AE16_JOIN_NOT_FOUND", "consensus_tier": ""}
        ae17 = {"ae17_status": "AE17_JOIN_NOT_FOUND", "meta_decision": ""}
        ae18 = {"ae18_status": "AE18_CONTEXT_UNAVAILABLE"}
        ae19 = {
            "ae19_status": "AE19_QWEN_AUDIT_SUCCEEDED",
            "llm_action_label": "BUY",
            "llm_authorizes_execution": True,  # malicious/incorrect flag
            "authority_status": "AUDIT_ONLY_NO_TRADE_AUTHORITY",
        }
        gates = {"gatekeeper_passed": False, "riskguard_passed": False, "gatekeeper_blocker": "x"}
        path = derive_strict_exploration(_cf_candidate(), ae16, ae17, ae18, ae19, gates)
        self.assertEqual(path["final_paper_demo_decision"], "NO_TRADE")
        self.assertFalse(path["trade_authority"])
        self.assertFalse(ae19["llm_authorizes_execution"])

    def test_llm_failure_does_not_raise(self):
        from app.llm_operational.schema import PROVIDER_AVAILABLE

        with mock.patch(
            "app.ae20.integrations.resolve_qwen_provider_status",
            return_value={"provider_status": PROVIDER_AVAILABLE, "provider_model": "qwen3:8b"},
        ), mock.patch(
            "app.ae20.integrations.call_ollama_chat",
            side_effect=RuntimeError("boom"),
        ):
            out = run_ae19_audit_only(
                _cf_candidate(),
                allow_llm=True,
                llm_provider="ollama",
                timeout_seconds=1,
                remaining_budget=2,
            )
        self.assertEqual(out["ae19_status"], "AE19_LLM_AUDIT_FAILED")
        self.assertFalse(out["llm_authorizes_execution"])

    def test_max_calls_budget_skip(self):
        out = run_ae19_audit_only(
            _cf_candidate(),
            allow_llm=True,
            llm_provider="ollama",
            timeout_seconds=1,
            remaining_budget=0,
        )
        self.assertEqual(out["ae19_status"], "AE19_LLM_SKIPPED_BY_CONFIG")


class TestBaselineAndLineage(unittest.TestCase):
    def test_baseline_excluded_from_pnl_and_orphans(self):
        preexisting_positions = [
            {
                "position_id": "pre1",
                "preexisting_baseline": True,
                "created_during_ae20": False,
                "realized_pnl": -10,
                "unrealized_pnl": 0,
                "status": "OPEN",
                "fees_assumption_usd": 0,
                "slippage_assumption_usd": 0,
                "maturity_status": "HISTORICAL",
            }
        ]
        preexisting_trades = [
            {
                "preexisting_baseline": True,
                "created_during_ae20": False,
                "realized_pnl": -5,
                "total_fees": 1,
            }
        ]
        decision = {
            "ae20_run_id": "r1",
            "ae20_cycle_id": "c1",
            "ae20_decision_id": "d1",
            "candidate_id": "cand",
            "provider_pair_url_exact": "https://dexscreener.com/solana/x",
            "canonical_market_identity": "https://dexscreener.com/solana/x",
            "price_source_key": "dexscreener|solana|x",
            "chain": "solana",
            "pair_address": "x",
            "final_paper_demo_decision": "PAPER_DEMO_OPEN",
            "strict_decision": "STRICT_APPROVED_PAPER_DEMO",
            "trade_authority": False,
            "live_trading_enabled": False,
            "created_during_ae20": True,
            "preexisting_baseline": False,
        }
        life = maybe_create_paper_lifecycle(decision, _cf_candidate(price_usd="2.0"))
        self.assertIsNotNone(life)
        assert life is not None
        orders = [life["order"]]
        positions = [life["position"]]
        outcomes = [life["outcome"]]
        lineage = audit_lineage(
            [decision],
            orders,
            positions,
            outcomes,
            preexisting_positions=preexisting_positions,
            preexisting_trades=preexisting_trades,
            preexisting_orders=[],
        )
        self.assertTrue(lineage["lineage_pass"])
        self.assertTrue(lineage["baseline_excluded_from_orphan_checks"])
        self.assertEqual(lineage["orphan_orders_count"], 0)

        pnl_rows, pnl_audit = build_pnl_summary(
            orders=orders,
            positions=positions,
            preexisting_positions=preexisting_positions,
            preexisting_trades=preexisting_trades,
        )
        self.assertTrue(pnl_audit["baseline_excluded_from_ae20_created_pnl"])
        self.assertTrue(pnl_audit["strict_exploration_separated"])
        combined = next(r for r in pnl_rows if r["pnl_scope"] == "AE20_CREATED_ALL")
        self.assertFalse(combined.get("includes_preexisting_baseline"))
        self.assertFalse(combined["profitability_claim"])
        baseline = next(r for r in pnl_rows if r["pnl_scope"] == "PREEXISTING_BASELINE_POSITIONS")
        self.assertEqual(baseline["decision_path"], "PREEXISTING_BASELINE")

    def test_orphan_order_detected(self):
        orders = [
            {
                "order_id": "o1",
                "ae20_decision_id": "missing",
                "position_id": "p1",
                "created_during_ae20": True,
            }
        ]
        lineage = audit_lineage(
            [],
            orders,
            [],
            [],
            preexisting_positions=[],
            preexisting_trades=[],
            preexisting_orders=[],
        )
        self.assertFalse(lineage["lineage_pass"])
        self.assertEqual(lineage["orphan_orders_count"], 1)


class TestClassificationAndE2E(unittest.TestCase):
    def test_status_field_presence_in_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ae20_out"
            # Force LLM unavailable so test is offline-safe and fast.
            result = run_ae20_integrated_clean_forward_validation(
                ROOT,
                smoke_cycles=1,
                output_root=out,
                no_external_llm=False,
                llm_provider="ollama",
                max_llm_calls_per_cycle=2,
                llm_timeout_seconds=5,
                force_llm_unavailable=True,
                max_candidates_per_cycle=3,
            )
            self.assertTrue(Path(result["output_root"]).exists())
            self.assertIn(result["classification"], {
                "AE20_SMOKE_INTEGRATION_PASS_READY_FOR_24H",
                "AE20_SMOKE_PASS_WITH_RUNTIME_LIMITATIONS",
                "AE20_SMOKE_BLOCKED_CLEAN_FORWARD_INPUT_FAILURE",
                "AE20_SMOKE_BLOCKED_INTEGRATION_LAYER_FAILURE",
                "AE20_SMOKE_BLOCKED_LINEAGE_FAILURE",
                "AE20_SMOKE_BLOCKED_LEGACY_CONTAMINATION",
                "AE20_SMOKE_BLOCKED_AUTHORITY_ESCALATION",
            })
            data = Path(result["output_root"]) / "data"
            audits = Path(result["output_root"]) / "audits"
            reports = Path(result["output_root"]) / "reports"
            required = [
                data / "ae20_integrated_decisions.csv",
                data / "ae20_clean_forward_inputs.csv",
                data / "ae20_pnl_summary.csv",
                data / "ae20_strict_vs_exploration.csv",
                data / "ae20_preexisting_positions_baseline.csv",
                data / "ae20_preexisting_trades_baseline.csv",
                data / "ae20_preexisting_orders_baseline.csv",
                audits / "ae20_ae16_consensus_integration_audit.csv",
                audits / "ae20_ae17_meta_integration_audit.csv",
                audits / "ae20_ae18_context_integration_audit.csv",
                audits / "ae20_ae19_llm_integration_audit.csv",
                audits / "ae20_authority_safety_audit.json",
                audits / "ae20_no_legacy_source_audit.json",
                audits / "ae20_output_root_collision_audit.json",
                audits / "ae20_llm_timeout_budget_audit.json",
                audits / "ae20_lineage_integrity_audit.json",
                audits / "ae20_preexisting_position_baseline_audit.json",
                reports / "ae20_manifest.json",
                reports / "ae20_decision_gate.json",
                reports / "ae20_final_closure_audit.json",
                reports / "ae20_summary_for_upload.txt",
            ]
            for path in required:
                self.assertTrue(path.is_file(), f"missing {path}")

            with (data / "ae20_integrated_decisions.csv").open(encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertGreater(len(rows), 0)
            row = rows[0]
            for field in (
                "ae16_status",
                "ae17_status",
                "ae18_status",
                "ae19_status",
                "trade_authority",
                "live_trading_enabled",
                "wallet_connected",
                "profitability_claim",
                "strict_decision",
                "exploration_decision",
            ):
                self.assertIn(field, row)
            self.assertEqual(row["trade_authority"], "false")
            self.assertEqual(row["live_trading_enabled"], "false")
            self.assertEqual(row["profitability_claim"], "false")

            with (data / "ae20_pnl_summary.csv").open(encoding="utf-8") as f:
                pnl = list(csv.DictReader(f))
            paths = {r["decision_path"] for r in pnl}
            self.assertIn("STRICT", paths)
            self.assertIn("EXPLORATION", paths)
            self.assertIn("COMBINED_AE20", paths)
            self.assertIn("PREEXISTING_BASELINE", paths)
            self.assertTrue(all(r.get("profitability_claim") == "false" for r in pnl))

            auth = json.loads((audits / "ae20_authority_safety_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(auth["pass"])
            legacy = json.loads((audits / "ae20_no_legacy_source_audit.json").read_text(encoding="utf-8"))
            self.assertTrue(legacy["pass"])

    def test_decide_classification_authority(self):
        c = decide_classification(
            identity_blocked=False,
            legacy_contaminated=False,
            authority_escalation=True,
            lineage_pass=True,
            integration_ok=True,
            llm_limitations=False,
            identity_failure_ratio=0.0,
            ae16_attached_count=1,
        )
        self.assertEqual(c, "AE20_SMOKE_BLOCKED_AUTHORITY_ESCALATION")

    def test_decision_gate_blockers_and_zero_attached(self):
        self.assertFalse(
            compute_unblocked_for_24h(
                classification="AE20_SMOKE_PASS_WITH_RUNTIME_LIMITATIONS",
                blockers_before_24h=["AE16 exact case-preserved provider_pair_url join attached 0 rows"],
                ae16_attached_count=0,
            )
        )
        self.assertFalse(
            compute_unblocked_for_24h(
                classification="AE20_SMOKE_INTEGRATION_PASS_READY_FOR_24H",
                blockers_before_24h=[],
                ae16_attached_count=0,
            )
        )
        c = decide_classification(
            identity_blocked=False,
            legacy_contaminated=False,
            authority_escalation=False,
            lineage_pass=True,
            integration_ok=True,
            llm_limitations=False,
            identity_failure_ratio=0.0,
            ae16_attached_count=0,
        )
        self.assertNotEqual(c, "AE20_SMOKE_INTEGRATION_PASS_READY_FOR_24H")

class TestExactIdentityLookupKey(unittest.TestCase):
    def test_strips_only_whitespace_preserves_case(self):
        raw = "  https://dexscreener.com/solana/2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd  "
        key = make_exact_identity_lookup_key(raw)
        self.assertEqual(
            key,
            "https://dexscreener.com/solana/2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd",
        )

    def test_helper_source_has_no_lower_or_casefold(self):
        import inspect

        src = inspect.getsource(make_exact_identity_lookup_key)
        self.assertNotIn(".lower()", src)
        self.assertNotIn(".casefold()", src)

    def test_mixed_case_does_not_match_lowercased(self):
        exact = "https://dexscreener.com/solana/2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd"
        # Build keys without using .lower() in test assertions path for identity equality.
        lowered_chars = []
        for ch in exact:
            if "A" <= ch <= "Z":
                lowered_chars.append(chr(ord(ch) + 32))
            else:
                lowered_chars.append(ch)
        lowered = "".join(lowered_chars)
        self.assertNotEqual(
            make_exact_identity_lookup_key(exact),
            make_exact_identity_lookup_key(lowered),
        )

    def test_invalid_literals_filtered_without_casefold(self):
        for lit in sorted(INVALID_IDENTITY_LITERALS):
            self.assertIsNone(make_exact_identity_lookup_key(lit), lit)
        self.assertIsNone(make_exact_identity_lookup_key(None))
        self.assertIsNone(make_exact_identity_lookup_key(float("nan")))
        # Empty keys never match each other (both None)
        self.assertIsNone(make_exact_identity_lookup_key(""))
        self.assertIsNone(make_exact_identity_lookup_key("nan"))
        self.assertIsNone(make_exact_identity_lookup_key("NaN"))
        self.assertIsNone(make_exact_identity_lookup_key("None"))
        self.assertIsNone(make_exact_identity_lookup_key("NULL"))


class TestAE16ExactProviderUrlBridge(unittest.TestCase):
    def _write_bridge(self, path: Path, url: str, tier: str = "TAB_RF_ONLY") -> None:
        write_csv(
            path,
            [
                {
                    "provider_pair_url": url,
                    "rf_evidence_status": "MODEL_EVIDENCE_ATTACHED",
                    "xgb_evidence_status": "MODEL_EVIDENCE_ATTACHED",
                    "tab_evidence_status": "MODEL_EVIDENCE_ATTACHED",
                    "rf_score": "0.55",
                    "xgb_score": "0.40",
                    "tab_score": "0.62",
                    "rf_vote": "true",
                    "xgb_vote": "false",
                    "tab_vote": "true",
                    "model_vote_count": "2",
                    "consensus_tier": tier,
                    "consensus_reason": "fixture_tab_rf",
                    "consensus_engine_version": "ae16_fixture_v1",
                },
                {
                    "provider_pair_url": "",
                    "rf_evidence_status": "X",
                    "xgb_evidence_status": "",
                    "tab_evidence_status": "",
                    "rf_score": "",
                    "xgb_score": "",
                    "tab_score": "",
                    "rf_vote": "",
                    "xgb_vote": "",
                    "tab_vote": "",
                    "model_vote_count": "",
                    "consensus_tier": "",
                    "consensus_reason": "",
                    "consensus_engine_version": "",
                },
                {
                    "provider_pair_url": "nan",
                    "rf_evidence_status": "X",
                    "xgb_evidence_status": "",
                    "tab_evidence_status": "",
                    "rf_score": "",
                    "xgb_score": "",
                    "tab_score": "",
                    "rf_vote": "",
                    "xgb_vote": "",
                    "tab_vote": "",
                    "model_vote_count": "",
                    "consensus_tier": "",
                    "consensus_reason": "",
                    "consensus_engine_version": "",
                },
            ],
        )

    def test_default_relative_path_resolves_from_project_root(self):
        meta = resolve_ae16_bridge_source(ROOT)
        self.assertEqual(meta["ae16_bridge_source_override_type"], "DEFAULT")
        self.assertFalse(meta["ae16_bridge_source_override_used"])
        self.assertTrue(meta["ae16_bridge_source_path_relative"].endswith(
            "ae16_clean_forward_consensus_decisions_v2.csv"
        ))
        expected = (ROOT / DEFAULT_AE16_BRIDGE_RELATIVE).resolve()
        self.assertEqual(Path(meta["ae16_bridge_source_path_resolved"]), expected)
        # No hardcoded absolute Windows path required
        self.assertFalse(DEFAULT_AE16_BRIDGE_RELATIVE.startswith("E:"))
        self.assertFalse(DEFAULT_AE16_BRIDGE_RELATIVE.startswith("E:\\"))

    def test_cli_and_env_overrides(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = Path(tmp) / "bridge.csv"
            self._write_bridge(bridge, "https://dexscreener.com/solana/ExactCaseAddr001")
            meta_cli = resolve_ae16_bridge_source(ROOT, cli_override=bridge)
            self.assertEqual(meta_cli["ae16_bridge_source_override_type"], "CLI")
            self.assertTrue(meta_cli["ae16_bridge_source_exists"])
            meta_env = resolve_ae16_bridge_source(
                ROOT, env_override=str(bridge)
            )
            self.assertEqual(meta_env["ae16_bridge_source_override_type"], "ENV")

    def test_exact_attach_and_no_lowercase_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = Path(tmp) / "bridge.csv"
            exact = "https://dexscreener.com/solana/2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd"
            self._write_bridge(bridge, exact)
            idx = load_ae16_index(ROOT, ae16_bridge_source=bridge)
            self.assertTrue(idx["audit"]["lookup_dictionary_used"])
            self.assertFalse(idx["audit"]["broad_merge_used"])
            self.assertFalse(idx["audit"]["forbidden_pair_chain_join_used"])
            self.assertFalse(idx["audit"]["lowercase_join_used"])
            self.assertGreaterEqual(idx["audit"]["ae16_invalid_provider_pair_url_count"], 2)

            cand = _cf_candidate(provider_pair_url_exact=exact, canonical_market_identity=exact)
            attached = attach_ae16(cand, idx)
            self.assertEqual(attached["ae16_status"], "AE16_EVIDENCE_ATTACHED")
            self.assertEqual(attached["ae16_provider_pair_url_original"], exact)
            self.assertEqual(attached["ae16_consensus_tier"], "TAB_RF_ONLY")
            self.assertFalse(attached["provider_pair_url_exact_mutated"])
            self.assertFalse(attached["forbidden_pair_chain_join_used"])
            # Lowercased AE20 URL must NOT match
            lowered_chars = []
            for ch in exact:
                if "A" <= ch <= "Z":
                    lowered_chars.append(chr(ord(ch) + 32))
                else:
                    lowered_chars.append(ch)
            lowered = "".join(lowered_chars)
            miss = attach_ae16(
                _cf_candidate(provider_pair_url_exact=lowered, canonical_market_identity=lowered),
                idx,
            )
            self.assertEqual(miss["ae16_status"], "AE16_JOIN_NOT_FOUND")

            # Empty AE20 key does not attach
            empty = attach_ae16(_cf_candidate(provider_pair_url_exact=""), idx)
            self.assertEqual(empty["ae16_status"], "AE16_JOIN_NOT_FOUND")

            # Invalid literals never join
            for lit in ("nan", "NaN", "None", "NULL"):
                bad = attach_ae16(_cf_candidate(provider_pair_url_exact=lit), idx)
                self.assertEqual(bad["ae16_status"], "AE16_JOIN_NOT_FOUND", lit)

    def test_attached_skip_reason_not_model_unavailable(self):
        ae16 = {
            "ae16_status": "AE16_EVIDENCE_ATTACHED",
            "ae16_consensus_tier": "MODEL_EVIDENCE_UNAVAILABLE",
            "consensus_tier": "MODEL_EVIDENCE_UNAVAILABLE",
            "exact_identity_join_used": True,
            "forbidden_pair_chain_join_used": False,
        }
        ae17 = {"ae17_status": "AE17_META_COMPUTED", "meta_decision": "META_UNAVAILABLE"}
        ae18 = {"ae18_status": "AE18_CONTEXT_MISSINGNESS_ONLY"}
        ae19 = {
            "ae19_status": "AE19_LLM_SKIPPED_BY_CONFIG",
            "llm_action_label": "",
            "llm_authorizes_execution": False,
            "authority_status": "AUDIT_ONLY_NO_TRADE_AUTHORITY",
        }
        gates = {"gatekeeper_passed": False, "riskguard_passed": False, "gatekeeper_blocker": "stale"}
        path = derive_strict_exploration(_cf_candidate(), ae16, ae17, ae18, ae19, gates)
        self.assertNotEqual(path["skip_reason"], "AE16_MODEL_EVIDENCE_UNAVAILABLE")
        self.assertEqual(path["skip_reason"], "AE17_META_LOW_CONFIDENCE")

    def test_e2e_with_exact_bridge_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            # Build bridge from first identity-index URL so exact join succeeds.
            index_path = ROOT / "data" / "runtime" / "canonical_market_identity_index.jsonl"
            url = None
            with index_path.open(encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    u = rec.get("provider_pair_url_exact")
                    if u:
                        url = u
                        break
            self.assertIsNotNone(url)
            bridge = tmp_path / "ae16_bridge_fixture.csv"
            self._write_bridge(bridge, url, tier="TAB_XGB_RF_ALL3")
            out = tmp_path / "ae20_out"
            result = run_ae20_integrated_clean_forward_validation(
                ROOT,
                smoke_cycles=1,
                output_root=out,
                force_llm_unavailable=True,
                max_candidates_per_cycle=3,
                max_llm_calls_per_cycle=2,
                llm_timeout_seconds=5,
                ae16_bridge_source=bridge,
            )
            self.assertGreater(result["ae16_attached_rows_count"], 0)
            self.assertTrue(result["exact_identity_join_used"])
            self.assertFalse(result["lowercase_join_used"])
            self.assertFalse(result["casefold_join_used"])
            self.assertFalse(result["case_insensitive_join_used"])
            self.assertFalse(result["forbidden_pair_chain_join_used"])
            self.assertFalse(result["broad_merge_used"])
            self.assertTrue(result["lookup_dictionary_used"])
            self.assertEqual(result["provider_pair_url_exact_mutated_count"], 0)
            self.assertEqual(result["ae16_provider_pair_url_mutated_count"], 0)

            decisions = list(
                csv.DictReader((Path(result["output_root"]) / "data" / "ae20_integrated_decisions.csv").open(encoding="utf-8"))
            )
            attached = [d for d in decisions if d.get("ae16_status") == "AE16_EVIDENCE_ATTACHED"]
            self.assertGreater(len(attached), 0)
            for d in attached:
                self.assertNotEqual(d.get("skip_reason"), "AE16_MODEL_EVIDENCE_UNAVAILABLE")
                self.assertEqual(d.get("provider_pair_url_exact"), url)
                self.assertEqual(d.get("ae16_provider_pair_url_original"), url)
                self.assertNotIn("status_x", d)
                self.assertNotIn("provider_pair_url_y", d)
                # AE20 chain not wiped
                self.assertTrue(d.get("chain"))

            audit_rows = list(
                csv.DictReader(
                    (Path(result["output_root"]) / "audits" / "ae20_ae16_consensus_integration_audit.csv").open(
                        encoding="utf-8"
                    )
                )
            )
            self.assertTrue(audit_rows)
            self.assertEqual(audit_rows[0].get("safe_provider_url_join_used"), "true")
            self.assertEqual(audit_rows[0].get("forbidden_pair_chain_join_used"), "false")
            self.assertEqual(audit_rows[0].get("exact_identity_join_used"), "true")
            self.assertEqual(audit_rows[0].get("lowercase_join_used"), "false")
            self.assertEqual(audit_rows[0].get("broad_merge_used"), "false")
            self.assertEqual(audit_rows[0].get("lookup_dictionary_used"), "true")


if __name__ == "__main__":
    unittest.main()
