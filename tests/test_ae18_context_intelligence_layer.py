"""Focused tests for AE18 Context Intelligence Layer."""

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

from app.ae18.constants import (  # noqa: E402
    CONTEXT_RECORD_FIELDS,
    RESOLVER_SYMBOL_REJECTED,
    SAFETY_BOUNDARY,
    WHALE_SIGNAL_POOL_FLOW_PROXY,
    WHALE_SIGNAL_WALLET_LEVEL,
)
from app.ae18.collectors import (  # noqa: E402
    collect_helius_solana_readonly,
    collect_rss_news_context,
    collect_semantic_context,
    collect_whale_evidence_separated,
)
from app.ae18.models import AE18CandidateTarget, AE18ContextRecord  # noqa: E402
from app.ae18.resolver import resolve_text_to_candidate  # noqa: E402
from app.ae18.audits import (  # noqa: E402
    audit_authority_safety,
    audit_helius_solana_readonly,
    audit_missingness_provenance,
    audit_no_symbol_only_join,
    audit_whale_score_separation,
)
from app.ae18.pipeline import run_ae18_context_intelligence_layer  # noqa: E402


def _sample_candidate(**kwargs) -> AE18CandidateTarget:
    base = dict(
        clean_forward_candidate_id="cand_test_001",
        clean_forward_decision_input_id="di_test_001",
        price_source_key="dexscreener|solana|abc123pair",
        provider="dexscreener",
        chain="solana",
        pair_address="abc123pair",
        base_token_address="tokenMint123",
        combined_target_id="ae16b_test",
        observed_at="2026-07-24T12:00:00+00:00",
    )
    base.update(kwargs)
    return AE18CandidateTarget(**base)


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


class TestAE18ContextRecordSchema(unittest.TestCase):
    def test_context_record_schema_fields(self):
        rec = AE18ContextRecord(
            context_record_id="r1",
            clean_forward_candidate_id="c1",
            context_family="rss_news",
            context_status="CONTEXT_UNAVAILABLE",
            source_name="rss_news",
            source_type="rss_normalized",
            attempted=True,
            available=False,
            missingness_reason="SOURCE_EMPTY_RESPONSE",
        )
        d = rec.to_dict()
        for field in CONTEXT_RECORD_FIELDS:
            self.assertIn(field, d, f"missing field {field}")
        self.assertTrue(d["no_trade_authority"])
        self.assertFalse(d["trade_authority"])
        self.assertFalse(d["wallet_access"])


class TestAE18Resolver(unittest.TestCase):
    def test_resolver_rejects_symbol_only_join(self):
        candidate = _sample_candidate()
        link = resolve_text_to_candidate(
            {"text_item_id": "t1", "symbol": "SOL", "title": "SOL pumps"},
            [candidate],
            context_record_id="ctx1",
        )
        self.assertEqual(link.resolver_status, RESOLVER_SYMBOL_REJECTED)
        self.assertTrue(link.symbol_only_rejected)
        self.assertEqual(link.clean_forward_candidate_id, "")

    def test_resolver_links_via_price_source_key(self):
        candidate = _sample_candidate()
        link = resolve_text_to_candidate(
            {
                "text_item_id": "t2",
                "price_source_key": "dexscreener|solana|abc123pair",
            },
            [candidate],
            context_record_id="ctx2",
        )
        self.assertEqual(link.resolver_status, "RESOLVER_LINKED")
        self.assertEqual(link.join_path, "price_source_key")
        self.assertEqual(link.clean_forward_candidate_id, candidate.clean_forward_candidate_id)

    def test_resolver_links_via_chain_pair(self):
        candidate = _sample_candidate()
        link = resolve_text_to_candidate(
            {"chain": "solana", "pair_address": "abc123pair"},
            [candidate],
            context_record_id="ctx3",
        )
        self.assertEqual(link.resolver_status, "RESOLVER_LINKED")
        self.assertEqual(link.join_path, "chain_pair_address")

    def test_unresolved_remains_flagged(self):
        candidate = _sample_candidate()
        link = resolve_text_to_candidate(
            {"text_item_id": "t4", "chain": "ethereum", "pair_address": "0xdead"},
            [candidate],
            context_record_id="ctx4",
        )
        self.assertEqual(link.resolver_status, "IDENTITY_UNRESOLVED")
        self.assertEqual(link.clean_forward_candidate_id, "")


class TestAE18Missingness(unittest.TestCase):
    def test_missingness_emitted_not_crash(self):
        candidate = _sample_candidate(chain="base", pair_address="0xabc")
        rec, miss, _ = collect_helius_solana_readonly(candidate, raw_payload_row=None)
        self.assertFalse(rec.available)
        self.assertTrue(miss is not None or rec.missingness_reason)
        self.assertIsNotNone(rec.missingness_reason)

    def test_rss_missingness_no_keyerror(self):
        candidate = _sample_candidate()
        rec, miss, items = collect_rss_news_context(candidate, None)
        self.assertFalse(rec.available)
        self.assertEqual(items, [])
        self.assertIn(rec.missingness_reason, ("SOURCE_UNAVAILABLE_PENDING_FETCH", "SOURCE_EMPTY_RESPONSE", "SOURCE_NOT_AVAILABLE_PENDING_FETCH"))

    def test_semantic_missingness_not_silently_dropped(self):
        candidate = _sample_candidate(token_symbol="", token_name="", combined_target_id="")
        rec, miss = collect_semantic_context(candidate)
        self.assertFalse(rec.available)
        self.assertIsNotNone(miss)


class TestAE18HeliusReadonly(unittest.TestCase):
    def test_helius_readonly_safety_fields(self):
        candidate = _sample_candidate()
        rec, _, safety = collect_helius_solana_readonly(candidate)
        self.assertFalse(safety["wallet_access"])
        self.assertFalse(safety["private_key_access"])
        self.assertFalse(safety["signer_available"])
        self.assertFalse(safety["transaction_signing_available"])
        self.assertFalse(safety["transaction_submission_available"])
        self.assertTrue(safety["readonly_rpc_methods_only"])

    def test_readonly_audit_passes(self):
        audit = audit_helius_solana_readonly(ROOT)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["safety_boundary"], SAFETY_BOUNDARY)


class TestAE18WhaleSeparation(unittest.TestCase):
    def test_legacy_whale_score_pool_flow_proxy_only(self):
        candidate = _sample_candidate(whale_score=0.75)
        pool, wallet, _ = collect_whale_evidence_separated(candidate)
        self.assertEqual(pool.whale_signal_type, WHALE_SIGNAL_POOL_FLOW_PROXY)
        self.assertTrue(pool.evidence_payload.get("not_wallet_level_whale_evidence"))
        self.assertFalse(wallet.available)
        self.assertEqual(wallet.whale_signal_type, WHALE_SIGNAL_WALLET_LEVEL)

    def test_wallet_level_requires_provenance(self):
        candidate = _sample_candidate()
        wallet_ev = {
            "available": True,
            "wallet_address": "Wallet111",
            "source_transaction_signature": "sig111",
            "source_rpc_method": "getTransaction",
            "read_only_provider": "helius",
        }
        _, wallet, _ = collect_whale_evidence_separated(candidate, wallet_evidence=wallet_ev)
        self.assertTrue(wallet.available)
        self.assertEqual(wallet.whale_signal_type, WHALE_SIGNAL_WALLET_LEVEL)

    def test_whale_separation_audit(self):
        candidate = _sample_candidate(whale_score=0.5)
        pool, wallet, _ = collect_whale_evidence_separated(candidate)
        audit = audit_whale_score_separation([pool, wallet])
        self.assertTrue(audit["passed"])


class TestAE18Authority(unittest.TestCase):
    def test_no_trade_authority(self):
        candidate = _sample_candidate()
        rec, _, _ = collect_helius_solana_readonly(candidate)
        audit = audit_authority_safety([rec])
        self.assertTrue(audit["passed"])


class TestAE18SymbolOnlyAudit(unittest.TestCase):
    def test_symbol_only_audit_fail_closed(self):
        from app.ae18.models import AE18ResolverLink

        safe = AE18ResolverLink(
            resolver_link_id="l1",
            context_record_id="c1",
            clean_forward_candidate_id="cand1",
            join_path="price_source_key",
            resolver_status="RESOLVER_LINKED",
        )
        rejected = AE18ResolverLink(
            resolver_link_id="l2",
            context_record_id="c2",
            clean_forward_candidate_id="",
            join_path="symbol_only",
            resolver_status=RESOLVER_SYMBOL_REJECTED,
            symbol_only_rejected=True,
        )
        audit = audit_no_symbol_only_join([safe, rejected])
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["symbol_only_rejection_count"], 1)


class TestAE18PipelineIntegration(unittest.TestCase):
    def test_pipeline_produces_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audits = root / "data" / "audits" / "ae17_meta_stacking_layer_test"
            data = audits / "data"
            row = {
                "clean_forward_candidate_id": "cand_pipe_1",
                "clean_forward_decision_input_id": "di_pipe_1",
                "price_source_key": "dexscreener|solana|pairpipe1",
                "provider": "dexscreener",
                "chain": "solana",
                "pair_address": "pairpipe1",
                "base_token_address": "mintpipe1",
                "observed_at": "2026-07-24T12:00:00+00:00",
                "context_status": "AE17_CONTEXT_NOT_AVAILABLE_PENDING_AE18",
            }
            fields = list(row.keys())
            _write_csv(data / "ae17_meta_feature_rows.csv", [row], fields)

            out_root = root / "data" / "audits" / "ae18_test_run"
            result = run_ae18_context_intelligence_layer(
                root,
                ae17_root=audits,
                output_root=out_root,
            )
            self.assertIn("classification", result)
            self.assertGreater(result.get("context_record_count", 0), 0)
            self.assertTrue((out_root / "reports" / "ae18_decision_gate.json").is_file())
            self.assertTrue((out_root / "data" / "ae18_context_records.csv").is_file())
            self.assertTrue((out_root / "audits" / "ae18_no_symbol_only_join_audit.json").is_file())
            gate = json.loads((out_root / "reports" / "ae18_decision_gate.json").read_text())
            self.assertFalse(gate["trade_authority"])
            self.assertEqual(gate["ae19_status"], "BLOCKED")


def _load_runner():
    path = ROOT / "scripts" / "run_ae18_context_intelligence_layer.py"
    spec = importlib.util.spec_from_file_location("run_ae18", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestAE18Runner(unittest.TestCase):
    def test_runner_importable(self):
        mod = _load_runner()
        self.assertTrue(hasattr(mod, "main"))


if __name__ == "__main__":
    unittest.main()
