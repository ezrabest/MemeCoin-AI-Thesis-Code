"""Tests for AE12.7 Intelligent-Agent Operational Demo Layer (unittest + pytest compatible)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from app.intelligent_agents.agent_policy import build_policy_from_args
from app.intelligent_agents.agent_ui_summary import build_ui_summary
from app.intelligent_agents.gemini_selective_audit import run_gemini_selective_audit
from app.intelligent_agents.helius_readonly_enrichment import run_helius_readonly_enrichment
from app.intelligent_agents.qwen_candidate_memo import generate_qwen_candidate_memo
from app.intelligent_agents.run import run_ae12_7_agent_demo
from app.intelligent_agents.safety import reject_authority_language
from app.intelligent_agents.semantic_context import link_semantic_context
from app.intelligent_agents.agent_audit_writer import yyyymmdd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_REL_PATHS = [
    "reports/ae12_7_manifest.json",
    "reports/ae12_7_summary_for_upload.txt",
    "reports/ae12_7_intelligent_agent_decision_gate.json",
    "reports/ae12_7_ui_status_summary.json",
    "data/ae12_7_agent_records.jsonl",
    "data/ae12_7_agent_records.csv",
    "data/ae12_7_qwen_candidate_memos.jsonl",
    "data/ae12_7_gemini_selective_audits.jsonl",
    "data/ae12_7_helius_readonly_enrichment.jsonl",
    "data/ae12_7_rss_context_links.csv",
    "data/ae12_7_semantic_context_links.csv",
    "data/ae12_7_agent_trade_linkage.csv",
    "data/ae12_7_missed_winner_agent_review.csv",
    "audits/ae12_7_agent_authority_audit.json",
    "audits/ae12_7_no_wallet_safety_audit.json",
    "audits/ae12_7_external_api_usage_audit.json",
    "audits/ae12_7_gemini_safety_audit.json",
    "audits/ae12_7_qwen_local_provider_audit.json",
    "audits/ae12_7_helius_readonly_audit.json",
    "audits/ae12_7_semantic_taxonomy_audit.json",
    "audits/ae12_7_linkage_integrity_audit.csv",
]


def _demo_candidate() -> dict:
    return {
        "candidate_id": "cand_test_1",
        "source_decision_id": "dec_test_1",
        "pair_address": "PairTest111",
        "symbol": "TST",
        "chain": "solana",
        "strict_shadow_decision": "NO_TRADE",
        "exploration_decision": "WATCH",
        "reason_not_traded": "price_price_stale",
        "price_freshness_status": "STALE_AT_ENTRY",
        "max_return": 0.7,
        "was_traded": False,
        "is_missed_winner": True,
        "semantic_signal_family": "UNKNOWN_UNRESOLVED",
        "paper_order_id": "paper_1",
        "_source_ref": "test",
    }


class TestAE127IntelligentAgentDemo(unittest.TestCase):
    def setUp(self) -> None:
        self._outs: list[Path] = []

    def _out(self, name: str) -> Path:
        # Prefer tempfile under data/audits for inspectability; use unique name
        from tempfile import mkdtemp

        p = Path(mkdtemp(prefix=f"ae127_{name}_", dir=str(PROJECT_ROOT / "data" / "audits")))
        self._outs.append(p)
        return p

    def test_01_qwen_unavailable_does_not_fail_run(self) -> None:
        policy = build_policy_from_args(
            mode="qwen-local",
            enable_qwen=True,
            provider="ollama",
            force_qwen_unavailable=True,
            no_external_api=True,
        )
        rec = generate_qwen_candidate_memo(_demo_candidate(), policy=policy)
        self.assertIn(rec["agent_status"], {"NOT_CONFIGURED", "SKIPPED"})
        self.assertFalse(rec["trade_authority_used"])

        result = run_ae12_7_agent_demo(
            project_root=PROJECT_ROOT,
            output_root=self._out("qwen"),
            mode="qwen-local",
            limit=5,
            enable_qwen=True,
            provider="ollama",
            force_qwen_unavailable=True,
            no_external_api=True,
            append_daily=False,
        )
        self.assertFalse(str(result["classification"]).startswith("AE12_7_FAIL"))
        self.assertFalse(result["trade_authority_used"])

    def test_02_gemini_disabled_no_external_call(self) -> None:
        out = self._out("gem_off")
        result = run_ae12_7_agent_demo(
            project_root=PROJECT_ROOT,
            output_root=out,
            mode="full-demo",
            limit=5,
            enable_gemini=False,
            no_external_api=True,
            append_daily=False,
        )
        ext = json.loads((out / "audits" / "ae12_7_external_api_usage_audit.json").read_text(encoding="utf-8"))
        self.assertFalse(ext["external_api_used"])
        self.assertEqual(ext["call_count"], 0)
        self.assertFalse(result["gate"]["external_api_used"])

    def test_03_gemini_enabled_selective_audit_only(self) -> None:
        policy = build_policy_from_args(
            mode="gemini-selective",
            enable_gemini=True,
            no_external_api=False,
            inject_gemini_response="Audit-only review: context conflict noted. Do not execute trades.",
            gemini_budget=5,
        )
        rec = run_gemini_selective_audit(_demo_candidate(), policy=policy, index=0)
        self.assertEqual(rec["agent_type"], "GEMINI_SELECTIVE_AUDIT")
        self.assertEqual(rec["agent_status"], "GENERATED")
        self.assertIn(rec["decision_effect"], {"audit_only", "explanation_only", "soft_warning_only"})
        self.assertFalse(rec["trade_authority_used"])
        self.assertTrue(rec.get("gemini_called"))

        out = self._out("gem_on")
        result = run_ae12_7_agent_demo(
            project_root=PROJECT_ROOT,
            output_root=out,
            mode="gemini-selective",
            limit=5,
            enable_gemini=True,
            no_external_api=False,
            inject_gemini_response="Selective audit: reporting only.",
            append_daily=False,
        )
        gem_path = out / "data" / "ae12_7_gemini_selective_audits.jsonl"
        self.assertTrue(gem_path.is_file())
        lines = [ln for ln in gem_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        self.assertTrue(any(json.loads(ln).get("agent_status") == "GENERATED" for ln in lines))
        self.assertFalse(result["trade_authority_used"])

    def test_04_forbidden_trade_language_rejected(self) -> None:
        rejection = reject_authority_language("You should BUY this token now and EXECUTE the trade.")
        self.assertTrue(rejection["forbidden_trade_language_found"])
        self.assertFalse(rejection["output_used_after_rejection"])
        self.assertEqual(rejection["safety_status"], "PASS_REJECTIONS_ENFORCED")

        policy = build_policy_from_args(
            mode="gemini-selective",
            enable_gemini=True,
            no_external_api=False,
            inject_gemini_response="Recommendation: BUY and EXECUTE live trade immediately.",
        )
        rec = run_gemini_selective_audit(_demo_candidate(), policy=policy, index=0)
        self.assertEqual(rec["agent_status"], "REJECTED_SAFETY")
        self.assertFalse(rec["trade_authority_used"])
        self.assertFalse(rec.get("output_used_after_rejection"))

    def test_05_helius_unavailable_not_configured(self) -> None:
        policy = build_policy_from_args(
            mode="helius-readonly",
            enable_helius=True,
            no_external_api=False,
            force_helius_unavailable=True,
        )
        rec = run_helius_readonly_enrichment(_demo_candidate(), policy=policy)
        self.assertEqual(rec["agent_status"], "NOT_CONFIGURED")
        self.assertFalse(rec["trade_authority_used"])

    def test_06_helius_never_accesses_wallet(self) -> None:
        out = self._out("hel")
        result = run_ae12_7_agent_demo(
            project_root=PROJECT_ROOT,
            output_root=out,
            mode="helius-readonly",
            limit=5,
            enable_helius=True,
            no_external_api=False,
            append_daily=False,
        )
        audit = json.loads((out / "audits" / "ae12_7_helius_readonly_audit.json").read_text(encoding="utf-8"))
        self.assertFalse(audit["wallet_accessed"])
        self.assertFalse(audit["private_key_accessed"])
        self.assertTrue(audit["readonly"])
        wallet = json.loads((out / "audits" / "ae12_7_no_wallet_safety_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(wallet["status"], "PASS_NO_WALLET")
        self.assertFalse(result["trade_authority_used"])

    def test_07_agent_output_never_changes_buy_sell_authority(self) -> None:
        out = self._out("auth")
        result = run_ae12_7_agent_demo(
            project_root=PROJECT_ROOT,
            output_root=out,
            mode="full-demo",
            limit=10,
            no_external_api=True,
            append_daily=False,
        )
        auth = json.loads((out / "audits" / "ae12_7_agent_authority_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(auth["status"], "PASS_NO_TRADE_AUTHORITY")
        self.assertFalse(auth["llm_may_authorize_trades"])
        for line in (out / "data" / "ae12_7_agent_records.jsonl").read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            self.assertFalse(r.get("trade_authority_used"))
            self.assertFalse(r.get("live_authority_used"))
        self.assertFalse(result["live_ready"])

    def test_08_linkage_where_ids_exist(self) -> None:
        out = self._out("link")
        result = run_ae12_7_agent_demo(
            project_root=PROJECT_ROOT,
            output_root=out,
            mode="artifact-only",
            limit=20,
            no_external_api=True,
            append_daily=False,
        )
        link_csv = (out / "data" / "ae12_7_agent_trade_linkage.csv").read_text(encoding="utf-8")
        self.assertIn("candidate_id", link_csv)
        self.assertIn("ids_hallucinated", link_csv)
        self.assertGreater(result["record_count"], 0)

    def test_09_missing_ids_recorded_not_hallucinated(self) -> None:
        out = self._out("miss")
        run_ae12_7_agent_demo(
            project_root=PROJECT_ROOT,
            output_root=out,
            mode="artifact-only",
            limit=20,
            no_external_api=True,
            append_daily=False,
        )
        integrity = (out / "audits" / "ae12_7_linkage_integrity_audit.csv").read_text(encoding="utf-8")
        self.assertIn("ids_hallucinated", integrity)
        records = [
            json.loads(ln)
            for ln in (out / "data" / "ae12_7_agent_records.jsonl").read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]
        self.assertTrue(any(r.get("missing_id_flags") is not None or r.get("candidate_id") for r in records))
        self.assertFalse(any(r.get("linkage", {}).get("ids_hallucinated") for r in records if r.get("linkage")))

    def test_10_unknown_unresolved_not_converted(self) -> None:
        rec = link_semantic_context(
            {
                "candidate_id": "c1",
                "semantic_signal_family": "UNKNOWN_UNRESOLVED",
                "legacy_cluster_label": "OPPORTUNISTIC_SPECULATIVE",
            }
        )
        self.assertEqual(rec["agent_status"], "UNKNOWN_UNRESOLVED")
        self.assertEqual(rec["semantic_label"], "UNKNOWN_UNRESOLVED")
        family = rec.get("semantic_signal_family") or rec.get("semantic_label")
        self.assertEqual(family, "UNKNOWN_UNRESOLVED")
        self.assertNotIn("SOCIAL", str(family))
        self.assertFalse(rec.get("legacy_is_final_semantic"))
        self.assertTrue(rec.get("unknown_unresolved_means", {}).get("not_social"))
        self.assertTrue(rec.get("unknown_unresolved_means", {}).get("not_opportunistic"))

    def test_11_ui_summary_trade_authority_false(self) -> None:
        out = self._out("ui")
        result = run_ae12_7_agent_demo(
            project_root=PROJECT_ROOT,
            output_root=out,
            mode="full-demo",
            limit=5,
            no_external_api=True,
            append_daily=False,
        )
        ui = result["ui_summary"]
        self.assertFalse(ui["trade_authority_used"])
        self.assertEqual(ui["wallet_status"], "NOT_CONFIGURED")
        self.assertFalse(ui["live_ready"])
        self.assertFalse(ui["profitability_proven"])
        rebuilt = build_ui_summary(records=[], policy_snapshot={"mode": "x"}, gate={"status": "ok"})
        self.assertFalse(rebuilt["trade_authority_used"])

    def test_12_no_wallet_safety_audit_passes(self) -> None:
        out = self._out("nw")
        run_ae12_7_agent_demo(
            project_root=PROJECT_ROOT,
            output_root=out,
            mode="full-demo",
            limit=5,
            no_external_api=True,
            append_daily=False,
        )
        nw = json.loads((out / "audits" / "ae12_7_no_wallet_safety_audit.json").read_text(encoding="utf-8"))
        self.assertEqual(nw["status"], "PASS_NO_WALLET")

    def test_13_append_only_jsonl_no_overwrite(self) -> None:
        day_dir = PROJECT_ROOT / "data" / "intelligent_agents"
        day_dir.mkdir(parents=True, exist_ok=True)
        day = yyyymmdd()
        rec_path = day_dir / f"ae12_7_agent_records_{day}.jsonl"
        before = rec_path.read_text(encoding="utf-8").count("\n") if rec_path.is_file() else 0

        run_ae12_7_agent_demo(
            project_root=PROJECT_ROOT,
            output_root=self._out("a1"),
            mode="artifact-only",
            limit=2,
            no_external_api=True,
            append_daily=True,
        )
        mid = rec_path.read_text(encoding="utf-8").count("\n")
        self.assertGreater(mid, before)

        run_ae12_7_agent_demo(
            project_root=PROJECT_ROOT,
            output_root=self._out("a2"),
            mode="artifact-only",
            limit=2,
            no_external_api=True,
            append_daily=True,
        )
        after = rec_path.read_text(encoding="utf-8").count("\n")
        self.assertGreater(after, mid)

    def test_14_external_api_usage_audit_records_calls(self) -> None:
        out = self._out("ext")
        run_ae12_7_agent_demo(
            project_root=PROJECT_ROOT,
            output_root=out,
            mode="gemini-selective",
            limit=5,
            enable_gemini=True,
            no_external_api=False,
            inject_gemini_response="Audit only — no trade authority.",
            append_daily=False,
        )
        ext = json.loads((out / "audits" / "ae12_7_external_api_usage_audit.json").read_text(encoding="utf-8"))
        self.assertTrue(ext["every_enabled_external_call_recorded"])
        self.assertGreaterEqual(ext["call_count"], 1)
        self.assertFalse(ext["trade_authority_on_any_call"])

    def test_15_demo_script_disabled_and_local_modes(self) -> None:
        out1 = self._out("dis")
        r1 = run_ae12_7_agent_demo(
            project_root=PROJECT_ROOT,
            output_root=out1,
            mode="artifact-only",
            limit=5,
            no_external_api=True,
            append_daily=False,
        )
        self.assertTrue(r1["classification"].startswith("AE12_7_PASS"))
        for rel in REQUIRED_REL_PATHS:
            self.assertTrue((out1 / rel).is_file(), rel)

        out2 = self._out("local")
        r2 = run_ae12_7_agent_demo(
            project_root=PROJECT_ROOT,
            output_root=out2,
            mode="qwen-local",
            limit=5,
            enable_qwen=True,
            provider="local",
            no_external_api=True,
            append_daily=False,
        )
        self.assertFalse(r2["ae12_closed"])
        self.assertFalse(r2["live_ready"])
        gate = json.loads((out2 / "reports" / "ae12_7_intelligent_agent_decision_gate.json").read_text(encoding="utf-8"))
        self.assertFalse(gate["ae12_closed"])
        self.assertFalse(gate["trade_authority_used"])


if __name__ == "__main__":
    unittest.main()
