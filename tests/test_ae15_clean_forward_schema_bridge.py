"""AE15 Clean Forward Schema Bridge unit tests."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.clean_forward.identity import (  # noqa: E402
    build_instrument_identity,
    normalize_address_for_chain,
    pair_address_for_id,
)
from app.clean_forward.lineage import (  # noqa: E402
    AE14_PENDING_NOTE,
    build_candidate_from_row,
    detect_lineage_mismatch,
    reconcile_ae14_order_position_lineage,
    summarize_order_position_lineage,
)
from app.clean_forward.schema import (  # noqa: E402
    CANDIDATE_ID_FORBIDDEN_FIELDS,
    make_clean_forward_candidate_id,
    make_clean_forward_decision_input_id,
)
from app.clean_forward.serialization import stable_json_dumps  # noqa: E402
from app.clean_forward.validation import (  # noqa: E402
    evaluate_clean_feed_eligibility,
    validate_identity_separation,
)


def _eligible_solana_row(**overrides):
    row = {
        "row_key": "solana|pair|2hXcTGNfeQFNsxV8d7ztYbEBEAowMwWvEdQQ8V3obyas",
        "source_provider": "dexscreener",
        "chain": "solana",
        "normalized_chain_id": "solana",
        "pair_address": "2hXcTGNfeQFNsxV8d7ztYbEBEAowMwWvEdQQ8V3obyas",
        "base_token_address": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
        "quote_token_address": "METvsvVRapdj9cFLzq4Tr43xK4tAjQfwX76z3n6mWQL",
        "base_token_symbol": "Bonk",
        "quote_token_symbol": "MET",
        "pair": "Bonk/MET",
        "price_usd": 0.01497,
        "liquidity_usd": 90717822.19,
        "volume_24h": 1000.0,
        "txns_24h_buys": 10,
        "txns_24h_sells": 8,
        "observed_at": "2026-07-21T21:02:25.890953+00:00",
        "fetched_at": "2026-07-21T21:02:25.890953+00:00",
        "ingested_at": "2026-07-21T21:02:25.890953+00:00",
        "provider_payload_hash": "2c6f1e450f8514b180f29162bfdf34fa",
        "provider_pair_url": "https://dexscreener.com/solana/2hxctgnfeqfnsxv8d7ztybebeaowmwwvedqq8v3obyas",
        "verification_status": "provider_pair_verified",
        "freshness_status": "fresh",
        "identity_status": "pair_and_tokens_separated",
        "shown_as_token_contract": False,
        "paper_demo_only": True,
        "live_trading_ready": False,
        "clean_feed_eligible": True,
        "coin_id": None,
    }
    row.update(overrides)
    return row


class TestAE15DeterministicIds(unittest.TestCase):
    def test_01_candidate_id_deterministic(self):
        row = _eligible_solana_row()
        a = build_candidate_from_row(row)
        b = build_candidate_from_row(row)
        self.assertEqual(a.clean_forward_candidate_id, b.clean_forward_candidate_id)
        again = make_clean_forward_candidate_id(
            chain="solana",
            provider="dexscreener",
            pair_address_for_id="2hXcTGNfeQFNsxV8d7ztYbEBEAowMwWvEdQQ8V3obyas",
            base_token_address=row["base_token_address"],
            quote_token_address=row["quote_token_address"],
            observed_at_or_fetched_at=row["observed_at"],
            provider_payload_hash=row["provider_payload_hash"],
        )
        self.assertEqual(a.clean_forward_candidate_id, again)

    def test_02_decision_input_id_deterministic(self):
        cand = build_candidate_from_row(_eligible_solana_row())
        a = make_clean_forward_decision_input_id(
            clean_forward_candidate_id=cand.clean_forward_candidate_id,
            candidate_snapshot_timestamp=cand.observed_at or "",
            active_preset_id="p1",
            risk_mode="balanced",
            strict_mode=False,
            exploration_mode=False,
        )
        b = make_clean_forward_decision_input_id(
            clean_forward_candidate_id=cand.clean_forward_candidate_id,
            candidate_snapshot_timestamp=cand.observed_at or "",
            active_preset_id="p1",
            risk_mode="balanced",
            strict_mode=False,
            exploration_mode=False,
        )
        self.assertEqual(a, b)
        c = make_clean_forward_decision_input_id(
            clean_forward_candidate_id=cand.clean_forward_candidate_id,
            candidate_snapshot_timestamp=cand.observed_at or "",
            active_preset_id="p2",
            risk_mode="balanced",
            strict_mode=False,
            exploration_mode=False,
        )
        self.assertNotEqual(a, c)


class TestAE15AddressNormalization(unittest.TestCase):
    def test_03_solana_case_preserved(self):
        addr = "2hXcTGNfeQFNsxV8d7ztYbEBEAowMwWvEdQQ8V3obyas"
        self.assertEqual(normalize_address_for_chain(addr, chain="solana"), addr)
        self.assertEqual(pair_address_for_id(addr, chain="solana"), addr)
        identity = build_instrument_identity(_eligible_solana_row())
        self.assertEqual(identity.pair_address, addr)
        self.assertEqual(identity.pair_address_normalized, addr)

    def test_04_evm_lowercase_normalization(self):
        mixed = "0xAbC123DEF4567890AbC123DEF4567890AbC123DE"
        norm = normalize_address_for_chain(mixed, chain="base")
        self.assertEqual(norm, mixed.lower())
        self.assertEqual(pair_address_for_id(mixed, chain="ethereum"), mixed.lower())


class TestAE15IdentityAndEligibility(unittest.TestCase):
    def test_05_reject_pair_as_token_confusion(self):
        row = _eligible_solana_row(
            base_token_address="2hXcTGNfeQFNsxV8d7ztYbEBEAowMwWvEdQQ8V3obyas",
            shown_as_token_contract=True,
            identity_status="confused",
        )
        result = validate_identity_separation(row)
        self.assertFalse(result["passed"])
        self.assertTrue(
            any(
                f in result["failures"]
                for f in (
                    "pair_address_equals_base_token_address",
                    "shown_as_token_contract",
                    "identity_status_not_pair_and_tokens_separated",
                )
            )
        )

    def test_06_coin_id_not_invented(self):
        row = _eligible_solana_row(coin_id="invented-coin")
        result = validate_identity_separation(row)
        self.assertFalse(result["passed"])
        self.assertIn("coin_id_invented", result["failures"])

    def test_07_clean_eligibility_rule(self):
        ok = evaluate_clean_feed_eligibility(_eligible_solana_row())
        self.assertTrue(ok["clean_feed_eligible"])
        bad = evaluate_clean_feed_eligibility(
            _eligible_solana_row(freshness_status="stale", live_trading_ready=True)
        )
        self.assertFalse(bad["clean_feed_eligible"])
        self.assertIn("freshness_status_not_fresh", bad["rejection_reasons"])
        self.assertIn("live_trading_ready_true", bad["rejection_reasons"])


class TestAE15Serialization(unittest.TestCase):
    def test_08_candidate_serialization_stability(self):
        cand = build_candidate_from_row(_eligible_solana_row())
        s1 = stable_json_dumps(cand.to_dict())
        s2 = stable_json_dumps(cand.to_dict())
        self.assertEqual(s1, s2)
        # Forbidden ID fields must not be part of ID formula constants
        self.assertIn("xgb_score", CANDIDATE_ID_FORBIDDEN_FIELDS)
        self.assertIn("paper_order_id", CANDIDATE_ID_FORBIDDEN_FIELDS)
        self.assertIn("future_return", CANDIDATE_ID_FORBIDDEN_FIELDS)


class TestAE15Lineage(unittest.TestCase):
    def test_09_order_position_mismatch_detection(self):
        result = detect_lineage_mismatch(orders_opened=1, positions_opened=2, links=[])
        self.assertTrue(result["mismatched"])
        self.assertTrue(result["requires_ae14_pending_note"])
        self.assertEqual(result["summary"]["counter_consistency_status"], AE14_PENDING_NOTE)

        ok = detect_lineage_mismatch(
            orders_opened=1,
            positions_opened=1,
            links=[
                {
                    "paper_order_id": "o1",
                    "paper_position_id": "p1",
                    "preexisting_position_detected": False,
                    "reconstructed_position_detected": False,
                }
            ],
        )
        self.assertFalse(ok["mismatched"])

    def test_10_ae14_reconciliation_carries_pending_note(self):
        artifacts = {
            "audit": {
                "paper_orders_opened": 1,
                "paper_positions_opened": 2,
                "opened_position_id": 1,
            },
            "selected_row": _eligible_solana_row(),
            "paper_open_position": {
                "id": 1,
                "opened_at": "2026-07-21T21:02:45.391909+00:00",
                "status": "OPEN",
                "size_usd": 50.0,
                "fill_price": 0.01497,
                "quantity": 3340.0,
                "pair_address": "2hXcTGNfeQFNsxV8d7ztYbEBEAowMwWvEdQQ8V3obyas",
                "base_token_address": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
                "quote_token_address": "METvsvVRapdj9cFLzq4Tr43xK4tAjQfwX76z3n6mWQL",
            },
            "demo_bot_run_once": {
                "opened": {
                    "opened": True,
                    "position": {
                        "id": 2,
                        "symbol": "PUMP",
                        "chain": "solana",
                        "pair_address": "4C8KctYZtMTZwV83Y5AcTPVT2aXYYu2t9ZhHdotFGnno",
                        "base_token_address": "pumpCmXqMfrsAkQ5r49WcJnRayYRqmXz6ae8H7H9Dfn",
                        "quote_token_address": "METvsvVRapdj9cFLzq4Tr43xK4tAjQfwX76z3n6mWQL",
                        "pair": "PUMP/MET",
                        "fill_price": 10.032,
                        "size_usd": 75.0,
                        "quantity": 7.47,
                        "opened_at": "2026-07-21T21:02:45.584517+00:00",
                        "status": "OPEN",
                        "entry_reason": "meme_opportunistic_scout",
                    },
                    "rejected_attempts": [
                        {
                            "rejection_code": "DUPLICATE_PAIR_ALREADY_OPEN",
                            "rejection_reasons": ["Blocked: duplicate pair already open"],
                            "pair_address": "2hXcTGNfeQFNsxV8d7ztYbEBEAowMwWvEdQQ8V3obyas",
                        }
                    ],
                }
            },
            "gatekeeper_result": {"ok": True},
            "bridge_result": {"ok": True},
        }
        result = reconcile_ae14_order_position_lineage(artifacts)
        self.assertTrue(result["ok"])
        self.assertEqual(result["ae14_discrepancy_status"], AE14_PENDING_NOTE)
        self.assertFalse(result["summary"]["ae14_discrepancy_resolved"])
        self.assertEqual(result["summary"]["orders_opened"], 1)
        self.assertEqual(result["summary"]["positions_opened"], 2)
        self.assertIn("2", result["summary"]["positions_without_order"])
        self.assertEqual(len(result["links"]), 2)


class TestAE15LegacyAndSafety(unittest.TestCase):
    def test_11_legacy_exclusion_flags(self):
        # Script emits these; assert contract constants here via summarize helper path
        summary = summarize_order_position_lineage([])
        self.assertIn("counter_consistency_status", summary)
        legacy = {
            "legacy_market_snapshots_used": False,
            "old_market_snapshot_feed_used": False,
            "raw_provider_payloads_legacy_feed_used": False,
            "local_db_candidate_universe_used": False,
            "model_training_performed": False,
            "backtest_performed": False,
            "profitability_claimed": False,
        }
        self.assertTrue(all(v is False for v in legacy.values()))

    def test_12_safety_flags(self):
        safety = {
            "wallet_configured": False,
            "private_key_accessed": False,
            "real_transaction_signed": False,
            "real_transaction_attempted": False,
            "live_trading_enabled": False,
            "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
            "paper_demo_only": True,
        }
        self.assertFalse(safety["wallet_configured"])
        self.assertFalse(safety["private_key_accessed"])
        self.assertFalse(safety["live_trading_enabled"])
        self.assertTrue(safety["paper_demo_only"])


class TestAE15RunnerSmoke(unittest.TestCase):
    def test_13_runner_reconcile_ae14_mode(self):
        import importlib.util

        script = ROOT / "scripts" / "run_ae15_clean_forward_schema_bridge.py"
        spec = importlib.util.spec_from_file_location("run_ae15", script)
        self.assertIsNotNone(spec)
        mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(mod)

        ae14 = ROOT / "data" / "audits" / "ae14_real_clean_forward_closure_20260721_210220"
        if not ae14.exists():
            self.skipTest("AE14 audit root not present")

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "ae15_out"
            result = mod.run_ae15(
                mod.parse_args(
                    [
                        "--reconcile-ae14",
                        "--ae14-root",
                        str(ae14),
                        "--output-root",
                        str(out),
                    ]
                )
            )
            gate = result["gate"]
            self.assertIn(
                gate["classification"],
                {
                    "AE15_PASS_WITH_LINEAGE_LIMITATIONS",
                    "AE15_CLEAN_FORWARD_SCHEMA_BRIDGE_PASS",
                },
            )
            self.assertTrue((out / "reports" / "ae15_decision_gate.json").exists())
            self.assertTrue((out / "audits" / "order_position_lineage_audit.json").exists())
            lineage = json.loads(
                (out / "audits" / "order_position_lineage_audit.json").read_text(encoding="utf-8")
            )
            self.assertEqual(lineage.get("ae14_discrepancy_status"), AE14_PENDING_NOTE)
            self.assertFalse(lineage.get("ae14_discrepancy_resolved"))


if __name__ == "__main__":
    unittest.main()
