"""Focused tests for AE18 real Helius/Solana continuation."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ae18.audits import decide_classification, audit_no_symbol_only_join  # noqa: E402
from app.ae18.constants import (  # noqa: E402
    CLASSIFICATION_BLOCKED_HELIUS_NOT_CONFIGURED,
    CLASSIFICATION_INFRA_ONLY,
    CLASSIFICATION_PASS_REAL_FETCH_LIMITATIONS,
    CLASSIFICATION_PASS_REAL_HELIUS,
    RESOLVER_SYMBOL_REJECTED,
)
from app.ae18.models import AE18CandidateTarget, AE18ResolverLink  # noqa: E402
from app.ae18.onchain_extract import (  # noqa: E402
    build_wallet_whale_evidence,
    classify_flow_pressure,
    compute_token_balance_deltas,
    map_account_index_to_pubkey,
)
from app.ae18.preflight import run_solana_preflight_safety  # noqa: E402
from app.ae18.readonly_rpc import AE18ReadOnlyRpcClient, AE18ReadOnlyViolation  # noqa: E402
from app.ae18.selector import select_interesting_solana_candidates  # noqa: E402


def _cand(**kwargs) -> AE18CandidateTarget:
    base = dict(
        clean_forward_candidate_id="c1",
        price_source_key="dexscreener|solana|pair1",
        chain="solana",
        pair_address="pair1",
        base_token_address="baseMint",
        quote_token_address="quoteMint",
    )
    base.update(kwargs)
    c = AE18CandidateTarget(**base)
    for k, v in kwargs.items():
        if k in ("consensus_tier", "meta_score", "meta_decision", "_liquidity_usd"):
            setattr(c, k, v)
    return c


class TestSelector(unittest.TestCase):
    def test_interesting_candidate_selection_priority(self):
        c0 = _cand(clean_forward_candidate_id="open", pair_address="openpair", price_source_key="dexscreener|solana|openpair")
        c1 = _cand(clean_forward_candidate_id="tier", pair_address="tierpair", price_source_key="dexscreener|solana|tierpair")
        setattr(c1, "consensus_tier", "TAB_XGB_ONLY")
        c2 = _cand(clean_forward_candidate_id="meta", pair_address="metapair", price_source_key="dexscreener|solana|metapair")
        setattr(c2, "meta_score", 0.8)
        setattr(c2, "meta_decision", "META_LOW_CONFIDENCE")
        c3 = _cand(clean_forward_candidate_id="liq", pair_address="liqpair", price_source_key="dexscreener|solana|liqpair")
        setattr(c3, "_liquidity_usd", 9_000_000)
        selected, rows = select_interesting_solana_candidates(
            [c3, c2, c1, c0],
            max_candidates=10,
            open_paper_pair_addresses={"openpair"},
        )
        self.assertEqual(selected[0].clean_forward_candidate_id, "open")
        self.assertEqual(rows[0]["selection_priority"], 0)
        self.assertEqual(selected[1].clean_forward_candidate_id, "tier")
        self.assertEqual(rows[1]["selection_priority"], 1)


class TestPreflight(unittest.TestCase):
    def test_preflight_fails_on_private_key_env(self):
        result = run_solana_preflight_safety(env={"PRIVATE_KEY": "secret", "HELIUS_API_KEY": "x"})
        self.assertFalse(result["preflight_passed"])
        self.assertIn("PRIVATE_KEY", result["private_key_env_vars_present"])
        # values must never appear
        blob = json.dumps(result)
        self.assertNotIn("secret", blob)

    def test_preflight_passes_clean_env(self):
        result = run_solana_preflight_safety(env={"HELIUS_API_KEY": "x", "PATH": "/usr/bin"})
        self.assertTrue(result["preflight_passed"])


class TestReadonlyRpc(unittest.TestCase):
    def test_forbids_write_methods(self):
        client = AE18ReadOnlyRpcClient(rpc_url="https://example.invalid", min_delay_ms=0)
        with self.assertRaises(AE18ReadOnlyViolation):
            client.call("sendTransaction", [])
        with self.assertRaises(AE18ReadOnlyViolation):
            client.call("simulateTransaction", [])
        with self.assertRaises(AE18ReadOnlyViolation):
            client.call("getProgramAccounts", [])  # not allowlisted
        # getSignaturesForAddress must remain allowed (contains substring "sign")
        # verified by allowlist membership rather than live call
        from app.ae18.readonly_rpc import ALLOWED_RPC_METHODS, _is_forbidden_method

        self.assertIn("getSignaturesForAddress", ALLOWED_RPC_METHODS)
        self.assertFalse(_is_forbidden_method("getSignaturesForAddress"))
        self.assertTrue(_is_forbidden_method("signTransaction"))

    def test_cache_and_throttle_and_429(self):
        calls = {"n": 0}

        def handler(request: httpx.Request) -> httpx.Response:
            calls["n"] += 1
            if calls["n"] == 1:
                return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "rate"})
            return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1, "result": {"value": None}})

        transport = httpx.MockTransport(handler)
        sleeps: list[float] = []
        client = AE18ReadOnlyRpcClient(
            rpc_url="https://example.invalid",
            min_delay_ms=10,
            max_retries=3,
            transport=transport,
            sleep_fn=lambda s: sleeps.append(s),
        )
        r1 = client.call("getAccountInfo", ["addr"])
        self.assertTrue(r1["success"])
        self.assertGreaterEqual(client.stats.retry_after_used_count, 1)
        r2 = client.call("getAccountInfo", ["addr"])
        self.assertTrue(r2["cache_hit"])
        self.assertEqual(client.stats.rpc_calls_skipped_by_cache, 1)
        self.assertTrue(any(x >= 0 for x in sleeps))
        self.assertTrue(client.raw_call_log)
        self.assertTrue(client.raw_call_log[0]["read_only_enforced"])


class TestOnchainExtract(unittest.TestCase):
    def test_account_index_mapping_and_deltas(self):
        tx = {
            "transaction": {
                "message": {
                    "accountKeys": [
                        {"pubkey": "feePayer", "signer": True},
                        {"pubkey": "tokenAcct", "signer": False},
                    ]
                },
                "signatures": ["sig1"],
            },
            "meta": {
                "err": None,
                "preTokenBalances": [
                    {
                        "accountIndex": 1,
                        "mint": "baseMint",
                        "owner": "owner1",
                        "uiTokenAmount": {"uiAmount": 10.0, "decimals": 6, "amount": "10000000"},
                    }
                ],
                "postTokenBalances": [
                    {
                        "accountIndex": 1,
                        "mint": "baseMint",
                        "owner": "owner1",
                        "uiTokenAmount": {"uiAmount": 5.0, "decimals": 6, "amount": "5000000"},
                    }
                ],
            },
        }
        self.assertEqual(map_account_index_to_pubkey(tx, 1), "tokenAcct")
        deltas = compute_token_balance_deltas(tx)
        self.assertEqual(len(deltas), 1)
        self.assertEqual(deltas[0]["delta"], -5.0)

    def test_ambiguous_flow_returns_unknown(self):
        flow = classify_flow_pressure([], base_token_address="b", quote_token_address="q")
        self.assertEqual(flow["flow_pressure_direction"], "UNKNOWN")
        flow2 = classify_flow_pressure(
            [{"meta": {}}],
            base_token_address="b",
            quote_token_address="q",
        )
        self.assertEqual(flow2["flow_pressure_direction"], "UNKNOWN")

    def test_wallet_evidence_rejects_whale_score(self):
        ev = build_wallet_whale_evidence(
            wallet_behavior={},
            flow={},
            signatures=[],
            rpc_methods=["getTransaction"],
            provenance_hashes=[],
            provider_used="HELIUS_RPC",
            whale_score=0.99,
        )
        self.assertFalse(ev["wallet_evidence_available"])
        self.assertTrue(ev["whale_score_rejected_as_input"])
        self.assertEqual(ev["missingness_reason"], "WALLET_LEVEL_DATA_NOT_AVAILABLE")

    def test_wallet_evidence_from_fee_payers(self):
        ev = build_wallet_whale_evidence(
            wallet_behavior={
                "unique_fee_payers": ["w1", "w1"],
                "unique_signers": ["w1"],
                "repeated_fee_payers": ["w1"],
                "repeated_signers": ["w1"],
                "fee_payer_concentration_share": 1.0,
            },
            flow={"token_owner_wallets_observed": ["w1"]},
            signatures=["sig"],
            rpc_methods=["getTransaction"],
            provenance_hashes=["h"],
            provider_used="HELIUS_RPC",
            whale_score=0.5,
        )
        self.assertTrue(ev["wallet_evidence_available"])
        self.assertTrue(ev["whale_score_rejected_as_input"])
        self.assertEqual(ev["wallet_evidence_source"], "HELIUS_RPC")


class TestClassification(unittest.TestCase):
    def _ok_audits(self):
        return {
            "no_symbol_audit": {"passed": True, "accepted_links_with_symbol_identity_basis": 0},
            "whale_audit": {"passed": True},
            "missingness_audit": {"passed": True},
            "readonly_audit": {"passed": True},
            "authority_audit": {"passed": True},
            "discovery_status": "AE18_INPUTS_DISCOVERED",
            "context_record_count": 10,
        }

    def test_no_fetch_is_infra_only_not_pass(self):
        cls = decide_classification(**self._ok_audits(), external_fetch_enabled=False)
        self.assertEqual(cls, CLASSIFICATION_INFRA_ONLY)

    def test_fetch_not_configured_blocked(self):
        cls = decide_classification(
            **self._ok_audits(),
            external_fetch_enabled=True,
            preflight_passed=True,
            rpc_configured=False,
        )
        self.assertEqual(cls, CLASSIFICATION_BLOCKED_HELIUS_NOT_CONFIGURED)

    def test_real_fetch_limitations(self):
        cls = decide_classification(
            **self._ok_audits(),
            external_fetch_enabled=True,
            preflight_passed=True,
            rpc_configured=True,
            rpc_calls_attempted=12,
            rpc_calls_successful=8,
            context_extracted_count=2,
            raw_payloads_saved=12,
        )
        self.assertEqual(cls, CLASSIFICATION_PASS_REAL_FETCH_LIMITATIONS)

    def test_real_helius_pass(self):
        cls = decide_classification(
            **self._ok_audits(),
            external_fetch_enabled=True,
            preflight_passed=True,
            rpc_configured=True,
            rpc_calls_attempted=40,
            rpc_calls_successful=30,
            context_extracted_count=5,
            raw_payloads_saved=40,
        )
        self.assertEqual(cls, CLASSIFICATION_PASS_REAL_HELIUS)


class TestSymbolOnlyMath(unittest.TestCase):
    def test_no_symbol_only_mathematical(self):
        links = [
            AE18ResolverLink(
                resolver_link_id="1",
                context_record_id="c",
                clean_forward_candidate_id="cand",
                join_path="price_source_key",
                resolver_status="RESOLVER_LINKED",
            ),
            AE18ResolverLink(
                resolver_link_id="2",
                context_record_id="c",
                clean_forward_candidate_id="",
                join_path="symbol_only",
                resolver_status=RESOLVER_SYMBOL_REJECTED,
                symbol_only_rejected=True,
            ),
        ]
        audit = audit_no_symbol_only_join(links)
        self.assertTrue(audit["passed"])
        self.assertEqual(audit["accepted_links_with_symbol_identity_basis"], 0)


if __name__ == "__main__":
    unittest.main()
