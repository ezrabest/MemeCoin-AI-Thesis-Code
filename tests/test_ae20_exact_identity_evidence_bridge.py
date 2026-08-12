"""Tests for AE20↔AE16 exact-identity evidence bridge v2."""

from __future__ import annotations

import csv
import inspect
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

from app.ae20.exact_identity_evidence_bridge import (  # noqa: E402
    allocate_bridge_output_root,
    build_bridge_row,
    extract_provider_url_tail_exact,
    has_ascii_upper,
    load_ae16_real_evidence,
    load_canonical_exact_records,
    resolve_ae20_ae16_exact_bridge,
    run_ae20_ae16_exact_identity_evidence_bridge,
)
from app.ae20.identity_keys import make_exact_identity_lookup_key  # noqa: E402
from app.ae20.integrations import (  # noqa: E402
    attach_ae16,
    load_ae16_exact_derived_bridge_index,
    load_ae16_index,
)
from app.ae20.orchestrator import (  # noqa: E402
    compute_unblocked_for_24h,
    decide_classification,
)
from app.ae20.sqlite_readonly import (  # noqa: E402
    ReadOnlySqliteError,
    assert_readonly_sql,
    build_readonly_sqlite_uri,
    open_readonly_sqlite,
    readonly_sqlite,
)
from app.consensus.serialization import write_csv  # noqa: E402


class TestSqliteReadOnly(unittest.TestCase):
    def test_uri_mode_ro_and_query_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.db"
            # Create a tiny db with a writable connection (setup only).
            setup = sqlite3.connect(str(db))
            setup.execute("CREATE TABLE raw_provider_payloads (query TEXT)")
            setup.execute("INSERT INTO raw_provider_payloads(query) VALUES ('https://X/AbC')")
            setup.commit()
            setup.close()

            conn, audit = open_readonly_sqlite(db)
            try:
                self.assertEqual(audit["sqlite_open_mode"], "READ_ONLY_URI_MODE_RO")
                self.assertTrue(audit["sqlite_uri_used"])
                self.assertTrue(audit["sqlite_query_only_pragma_enabled"])
                self.assertIn("mode=ro", audit["sqlite_uri"])
                row = conn.execute(
                    "SELECT COUNT(*) FROM raw_provider_payloads WHERE query = ?",
                    ("https://X/AbC",),
                ).fetchone()
                self.assertEqual(int(row[0]), 1)
            finally:
                conn.close()

            src = inspect.getsource(open_readonly_sqlite)
            self.assertIn("mode=ro", src)
            self.assertIn("uri=True", src)
            self.assertNotIn("sqlite3.connect(db_path)", src)
            self.assertNotIn("sqlite3.connect(str(db_path))", src)

    def test_write_sql_rejected(self):
        assert_readonly_sql("SELECT 1")
        with self.assertRaises(ReadOnlySqliteError):
            assert_readonly_sql("INSERT INTO t VALUES (1)")
        with self.assertRaises(ReadOnlySqliteError):
            assert_readonly_sql("UPDATE t SET a=1")
        with tempfile.TemporaryDirectory() as tmp:
            db = Path(tmp) / "t.db"
            setup = sqlite3.connect(str(db))
            setup.execute("CREATE TABLE raw_provider_payloads (id INTEGER, query TEXT)")
            setup.commit()
            setup.close()
            with readonly_sqlite(db) as conn:
                with self.assertRaises(ReadOnlySqliteError):
                    conn.execute("INSERT INTO raw_provider_payloads(id) VALUES (1)")
                self.assertFalse(conn.audit["db_mutation"])
                self.assertFalse(conn.audit["raw_mutation"])


class TestExactIdentityHelpers(unittest.TestCase):
    def test_tail_and_ascii_upper_preserved(self):
        url = "https://dexscreener.com/solana/2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd"
        self.assertTrue(has_ascii_upper(url))
        self.assertEqual(
            extract_provider_url_tail_exact(url),
            "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd",
        )
        src_tail = inspect.getsource(extract_provider_url_tail_exact)
        self.assertNotIn(".lower()", src_tail)
        self.assertNotIn(".casefold()", src_tail)

    def test_bridge_builder_source_has_no_lower_casefold(self):
        import app.ae20.exact_identity_evidence_bridge as mod

        src = inspect.getsource(mod)
        # Allowlisted only in _bool_arg for flag parsing, not identity joins.
        # Identity path helpers must not call .lower()/.casefold() on identity.
        bridge_src = inspect.getsource(build_bridge_row)
        self.assertNotIn(".lower()", bridge_src)
        self.assertNotIn(".casefold()", bridge_src)
        key_src = inspect.getsource(make_exact_identity_lookup_key)
        self.assertNotIn(".lower()", key_src)
        self.assertNotIn(".casefold()", key_src)


class TestDecisionGateStrict(unittest.TestCase):
    def test_blockers_force_unblocked_false(self):
        self.assertFalse(
            compute_unblocked_for_24h(
                classification="AE20_SMOKE_INTEGRATION_PASS_READY_FOR_24H",
                blockers_before_24h=["AE16 attached 0 rows"],
                ae16_attached_count=5,
            )
        )

    def test_zero_attached_forces_unblocked_false(self):
        self.assertFalse(
            compute_unblocked_for_24h(
                classification="AE20_SMOKE_INTEGRATION_PASS_READY_FOR_24H",
                blockers_before_24h=[],
                ae16_attached_count=0,
            )
        )

    def test_zero_attached_prevents_ready_classification(self):
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
        self.assertEqual(c, "AE20_SMOKE_PASS_WITH_RUNTIME_LIMITATIONS")

    def test_unsafe_flags_force_unblocked_false(self):
        self.assertFalse(
            compute_unblocked_for_24h(
                classification="AE20_SMOKE_INTEGRATION_PASS_READY_FOR_24H",
                blockers_before_24h=[],
                ae16_attached_count=3,
                unsafe_bridge_flags=True,
            )
        )

    def test_ready_when_clear(self):
        self.assertTrue(
            compute_unblocked_for_24h(
                classification="AE20_SMOKE_INTEGRATION_PASS_READY_FOR_24H",
                blockers_before_24h=[],
                ae16_attached_count=3,
                unsafe_bridge_flags=False,
            )
        )


class TestDerivedBridgeLogic(unittest.TestCase):
    def test_pair_chain_only_cannot_close(self):
        ae20 = {
            "ae20_candidate_id_original": "x",
            "ae20_provider_pair_url_exact": "https://dexscreener.com/solana/AbCdEf",
            "ae20_provider_chain_exact": "solana",
            "ae20_provider_pair_tail_exact": "AbCdEf",
            "ae20_provider_pair_url_exact_has_ascii_upper": True,
            "ae20_source_smoke_root": "data/audits/x",
            "ae20_source_file": "data/audits/x/d.csv",
        }
        # No exact source proof, no AE16 evidence — invalid even if pair/chain known.
        row = build_bridge_row(
            ae20,
            canonical_hits=[],
            raw_url_count=0,
            raw_tail_count=0,
            ae16_by_psk={},
            ae16_evidence_path="data/audits/preview.csv",
        )
        self.assertFalse(row["exact_identity_bridge_row_valid"])
        self.assertFalse(row["pair_chain_only_join_used_for_closure"])

    def test_legacy_locator_requires_source_proof(self):
        ae20 = {
            "ae20_candidate_id_original": "x",
            "ae20_provider_pair_url_exact": "https://dexscreener.com/solana/AbCdEf",
            "ae20_provider_chain_exact": "solana",
            "ae20_provider_pair_tail_exact": "AbCdEf",
            "ae20_provider_pair_url_exact_has_ascii_upper": True,
            "ae20_source_smoke_root": "data/audits/x",
            "ae20_source_file": "data/audits/x/d.csv",
        }
        source = {
            "_source_type": "canonical_market_identity_index",
            "_source_path": "data/runtime/canonical_market_identity_index.jsonl",
            "_source_record_ref": "jsonl_line:1",
            "provider_pair_url_exact": "https://dexscreener.com/solana/AbCdEf",
            "provider_pair_url_final_segment_exact": "AbCdEf",
            "price_source_key": "dexscreener|solana|abcdef",
        }
        evidence = {
            "dexscreener|solana|abcdef": {
                "_ae16_evidence_row_ref": "csv_row:2",
                "pair_address": "abcdef",
                "chain": "solana",
                "RF_score": "0.1",
                "RF_vote": "false",
                "RF_status": "MODEL_EVIDENCE_ATTACHED",
                "RF_threshold": "0.5",
                "XGB_score": "0.1",
                "XGB_vote": "false",
                "XGB_status": "MODEL_EVIDENCE_ATTACHED",
                "XGB_threshold": "0.5",
                "TAB16_score": "0.1",
                "TAB16_vote": "false",
                "TAB16_status": "MODEL_EVIDENCE_ATTACHED",
                "TAB16_threshold": "0.5",
                "TAB16_model_variant": "TAB16",
                "TAB16_artifact_path": "models/x.joblib",
                "true_vote_count": "0",
                "TAB_score_for_consensus": "0.1",
                "TAB_vote_for_consensus": "false",
                "consensus_preview_tier": "REJECT",
                "consensus_tab_slot_source": "TAB16",
                "consensus_tab_slot_legacy_tab_used": "false",
                "consensus_tab_slot_status": "MODEL_EVIDENCE_ATTACHED",
            }
        }
        row = build_bridge_row(
            ae20,
            canonical_hits=[source],
            raw_url_count=1,
            raw_tail_count=1,
            ae16_by_psk=evidence,
            ae16_evidence_path="data/audits/preview.csv",
        )
        self.assertTrue(row["exact_identity_bridge_row_valid"])
        self.assertTrue(row["legacy_locator_used"])
        self.assertFalse(row["legacy_locator_was_computed_by_ae20"])
        self.assertFalse(row["legacy_locator_is_canonical_identity"])
        self.assertEqual(row["ae20_provider_pair_url_exact"], ae20["ae20_provider_pair_url_exact"])
        self.assertEqual(row["ae20_provider_pair_tail_exact"], "AbCdEf")
        self.assertEqual(row["ae16_consensus_preview_tier"], "REJECT")
        self.assertFalse(row["lowercase_join_used"])

    def test_collision_safe_root(self):
        root1, a1 = allocate_bridge_output_root(ROOT)
        root2, a2 = allocate_bridge_output_root(ROOT)
        self.assertNotEqual(root1, root2)
        self.assertTrue(a1["stamp_has_microseconds"])
        self.assertTrue(a1["uuid_suffix_present"])
        # cleanup
        for r in (root1, root2):
            for p in sorted(r.rglob("*"), reverse=True):
                if p.is_file():
                    p.unlink()
                elif p.is_dir():
                    p.rmdir()
            if r.exists():
                r.rmdir()

    def test_canonical_discovery_finds_mixed_case(self):
        path = ROOT / "data" / "runtime" / "canonical_market_identity_index.jsonl"
        if not path.is_file():
            self.skipTest("canonical index missing")
        by_url = load_canonical_exact_records(path)
        mixed = None
        for url in by_url:
            if has_ascii_upper(url):
                mixed = url
                break
        self.assertIsNotNone(mixed)
        self.assertIn(mixed, by_url)

    def test_old_legacy_bridge_not_authority_when_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            bridge = Path(tmp) / "legacy.csv"
            write_csv(
                bridge,
                [
                    {
                        "provider_pair_url": "https://dexscreener.com/solana/AbCdEf",
                        "rf_evidence_status": "",
                        "xgb_evidence_status": "",
                        "tab_evidence_status": "",
                        "rf_score": "",
                        "xgb_score": "",
                        "tab_score": "",
                        "rf_vote": "",
                        "xgb_vote": "",
                        "tab_vote": "",
                        "model_vote_count": "",
                        "consensus_tier": "MODEL_EVIDENCE_UNAVAILABLE",
                        "consensus_reason": "legacy",
                        "consensus_engine_version": "v0",
                    }
                ],
            )
            idx = load_ae16_index(ROOT, ae16_bridge_source=bridge)
            self.assertFalse(idx["audit"]["legacy_ae16_bridge_used_as_evidence_authority"])
            attached = attach_ae16(
                {
                    "provider_pair_url_exact": "https://dexscreener.com/solana/AbCdEf",
                    "identity_ok": True,
                },
                idx,
            )
            self.assertEqual(attached["ae16_status"], "AE16_JOIN_NOT_FOUND")

    def test_resolve_exact_bridge_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bridge.csv"
            p.write_text("ae20_provider_pair_url_exact\n", encoding="utf-8")
            meta = resolve_ae20_ae16_exact_bridge(ROOT, cli_override=p)
            self.assertEqual(meta["ae20_ae16_exact_bridge_override_type"], "CLI")
            self.assertTrue(meta["ae20_ae16_exact_bridge_exists"])


class TestBridgeEndToEndFixture(unittest.TestCase):
    def test_runner_attaches_from_exact_bridge(self):
        path = ROOT / "data" / "runtime" / "canonical_market_identity_index.jsonl"
        evidence = (
            ROOT
            / "data/audits/ae16_tab16_direct_target_serving_safe_20260724T205012Z/"
            / "data/rf_xgb_tab16_consensus_preview.csv"
        )
        if not path.is_file() or not evidence.is_file():
            self.skipTest("required sources missing")

        # Use latest smoke roots implicitly; bridge generation is read-only.
        result = run_ae20_ae16_exact_identity_evidence_bridge(
            ROOT,
            paper_demo_only=True,
            clean_forward_only=True,
            no_lowercase_joins=True,
            smoke_roots=[
                ROOT
                / "data/audits/ae20_integrated_clean_forward_validation_20260803T200907844995Z_25a9fd3b",
                ROOT
                / "data/audits/ae20_integrated_clean_forward_validation_20260803T201002162891Z_19c776ad",
            ],
        )
        self.assertIn(result["classification"], {
            "AE20_AE16_EXACT_DERIVED_BRIDGE_PASS",
            "AE20_AE16_EXACT_DERIVED_BRIDGE_PASS_WITH_UNMATCHED_ROWS",
        })
        self.assertGreater(result["ae20_ae16_derived_bridge_matched_rows"], 0)
        self.assertFalse(result["lowercase_join_used"])
        self.assertFalse(result["db_mutation"])
        self.assertFalse(result["raw_mutation"])
        self.assertEqual(result["sqlite_open_mode"], "READ_ONLY_URI_MODE_RO")
        self.assertTrue(result["sqlite_uri_used"])
        self.assertTrue(result["sqlite_query_only_pragma_enabled"])
        self.assertFalse(result["sqlite_write_sql_detected"])
        self.assertFalse(result["legacy_locator_computed_by_ae20"])
        self.assertFalse(result["legacy_locator_is_canonical_identity"])

        bridge_csv = Path(result["bridge_csv"])
        self.assertTrue(bridge_csv.is_file())
        idx = load_ae16_exact_derived_bridge_index(
            ROOT, exact_bridge_path=bridge_csv
        )
        self.assertIsNotNone(idx)
        assert idx is not None
        # Attach first matched URL
        with bridge_csv.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertTrue(rows)
        url = rows[0]["ae20_provider_pair_url_exact"]
        attached = attach_ae16({"provider_pair_url_exact": url}, idx)
        self.assertEqual(
            attached["ae16_status"],
            "AE16_EVIDENCE_ATTACHED_FROM_EXACT_DERIVED_BRIDGE",
        )
        self.assertNotEqual(attached.get("ae16_consensus_tier"), "")
        self.assertFalse(attached["lowercase_join_used"])


if __name__ == "__main__":
    unittest.main()
