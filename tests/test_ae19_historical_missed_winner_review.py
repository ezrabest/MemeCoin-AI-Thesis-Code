"""Tests for AE19 historical missed-winner dual-provider review."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_PATH = ROOT / "scripts" / "run_ae19_historical_missed_winner_review.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_ae19_historical_missed_winner_review", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


def _valid_review(provider: str, model: str = "test-model") -> dict:
    return {
        "review_type": "AE19_HISTORICAL_MISSED_WINNER_REVIEW",
        "provider": provider,
        "model": model,
        "source_file": "fixture.csv",
        "authority_status": "AUDIT_ONLY_NO_TRADE_AUTHORITY",
        "execution_allowed": False,
        "paper_execution_allowed": False,
        "live_execution_allowed": False,
        "risk_override_allowed": False,
        "execution_attempted": False,
        "profitability_claimed": False,
        "summary": f"{provider} audit summary of missed winners",
        "top_findings": ["duplicate-heavy raw tops", "ACTIVE_PAIR_LOCK dominant"],
        "dominant_missed_winner_causes": [
            "ACTIVE_PAIR_LOCK",
            "max_open_positions",
            "price_stale_exploration",
            "WEAK_LINEAGE",
        ],
        "deduplication_warning": "Raw top rows are duplicate-heavy; do not overcount pair 0xe7A381...",
        "no_lookahead_assessment": "no_lookahead_status distribution indicates no lookahead violations in fixture",
        "top_case_review": {"note": "dedup preferred"},
        "blocker_interpretation": {
            "stale_price": "price_stale_exploration present",
            "weak_lineage": "WEAK_LINEAGE / MISSING_CONTEXT_FAMILIES common",
        },
        "model_specific_observations": ["fixture observation"],
        "recommended_research_followups": ["inspect lock contention"],
        "not_allowed_actions": [
            "no trade authority",
            "no execution",
            "no live approval",
            "no risk override",
        ],
    }


def _fixture_rows() -> list[dict[str, str]]:
    """Small fixture: duplicate-heavy top pair, multiple horizons, blockers JSON."""
    base_blockers = '["WEAK_LINEAGE","STALE_CONTEXT","MISSING_CONTEXT_FAMILIES"]'
    rows = []
    # Duplicate-heavy pair A with high max_return
    for i in range(4):
        rows.append(
            {
                "evidence_row_id": f"ev_a_{i}",
                "candidate_id": f"cand_a_{i % 2}",  # 2 candidates for pair A
                "decision_id": f"dec_a_{i}",
                "pair_address": "0xPAIR_DUP_HEAVY",
                "first_seen_timestamp": "2026-07-14T00:00:00Z",
                "horizon": ["5m", "15m", "1h", "6h"][i],
                "max_return": str(10.0 - i * 0.1),
                "threshold": "0.5",
                "was_traded": "false",
                "strict_shadow_decision": "SKIP",
                "exploration_decision": "SKIP",
                "reason_not_traded": "ACTIVE_PAIR_LOCK",
                "rejection_reason": "lock",
                "price_freshness_status": "fresh",
                "context_missingness": "partial",
                "audit_blockers": base_blockers,
                "cooldown_active": "false",
                "max_open_positions_hit": "false",
                "duplicate_active_pair": "true",
                "duplicate_reason": "ACTIVE_PAIR_LOCK",
                "no_lookahead_status": "NO_LOOKAHEAD_OK",
            }
        )
    # Unique pair B lower return
    rows.append(
        {
            "evidence_row_id": "ev_b_0",
            "candidate_id": "cand_b_0",
            "decision_id": "dec_b_0",
            "pair_address": "0xPAIR_OTHER",
            "first_seen_timestamp": "2026-07-14T01:00:00Z",
            "horizon": "24h",
            "max_return": "2.0",
            "threshold": "0.5",
            "was_traded": "false",
            "strict_shadow_decision": "SKIP",
            "exploration_decision": "SKIP",
            "reason_not_traded": "max_open_positions",
            "rejection_reason": "capacity",
            "price_freshness_status": "stale",
            "context_missingness": "high",
            "audit_blockers": '["STALE_CONTEXT"]',
            "cooldown_active": "false",
            "max_open_positions_hit": "true",
            "duplicate_active_pair": "false",
            "duplicate_reason": "",
            "no_lookahead_status": "NO_LOOKAHEAD_OK",
        }
    )
    # price_stale_exploration row
    rows.append(
        {
            "evidence_row_id": "ev_c_0",
            "candidate_id": "cand_c_0",
            "decision_id": "dec_c_0",
            "pair_address": "0xPAIR_STALE",
            "first_seen_timestamp": "2026-07-14T02:00:00Z",
            "horizon": "1h",
            "max_return": "1.5",
            "threshold": "0.5",
            "was_traded": "false",
            "strict_shadow_decision": "SKIP",
            "exploration_decision": "SKIP",
            "reason_not_traded": "price_stale_exploration",
            "rejection_reason": "stale",
            "price_freshness_status": "stale",
            "context_missingness": "high",
            "audit_blockers": '["WEAK_LINEAGE","STALE_CONTEXT"]',
            "cooldown_active": "false",
            "max_open_positions_hit": "false",
            "duplicate_active_pair": "false",
            "duplicate_reason": "",
            "no_lookahead_status": "NO_LOOKAHEAD_OK",
        }
    )
    return rows


def _write_fixture_csv(path: Path, rows: list[dict[str, str]] | None = None) -> Path:
    rows = rows or _fixture_rows()
    fieldnames = list(MOD.REQUIRED_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    return path


def _ok_call(provider: str):
    def _call(prompt: str, model: str | None = None, **kwargs):
        payload = _valid_review(provider, model or f"{provider}-model")
        return {
            "ok": True,
            "text": json.dumps(payload),
            "raw_json": {"ok": True},
            "model": model or f"{provider}-model",
            "error_type": None,
            "error_message": None,
            "timeout_s": 30,
        }

    return _call


class TestRequiredColumns(unittest.TestCase):
    def test_required_columns_validation(self):
        missing = MOD.validate_required_columns(["horizon", "max_return"])
        self.assertIn("evidence_row_id", missing)
        self.assertIn("pair_address", missing)
        ok = MOD.validate_required_columns(list(MOD.REQUIRED_COLUMNS) + ["extra"])
        self.assertEqual(ok, [])


class TestDeterministicSummaries(unittest.TestCase):
    def setUp(self):
        self.rows = _fixture_rows()
        self.summary = MOD.build_deterministic_summary(
            self.rows, source_path=Path("fixture.csv"), top_n=10
        )

    def test_horizon_counts(self):
        hc = self.summary["horizon_counts"]
        self.assertEqual(hc["5m"], 1)
        self.assertEqual(hc["15m"], 1)
        self.assertEqual(hc["1h"], 2)  # one from dup set + stale row
        self.assertEqual(hc["6h"], 1)
        self.assertEqual(hc["24h"], 1)

    def test_raw_top_preserves_duplicates(self):
        top = self.summary["top_raw_rows"]
        pairs = [r["pair_address"] for r in top]
        self.assertGreaterEqual(pairs.count("0xPAIR_DUP_HEAVY"), 2)

    def test_dedup_top_pairs_collapses(self):
        top = self.summary["top_pairs_dedup"]
        pairs = [r["pair_address"] for r in top]
        self.assertEqual(pairs.count("0xPAIR_DUP_HEAVY"), 1)
        heavy = next(r for r in top if r["pair_address"] == "0xPAIR_DUP_HEAVY")
        self.assertEqual(heavy["duplicate_row_count"], 4)

    def test_dedup_top_candidates_collapses(self):
        top = self.summary["top_candidates_dedup"]
        cids = [r["candidate_id"] for r in top]
        self.assertEqual(len(cids), len(set(cids)))

    def test_audit_blockers_counting(self):
        blockers = MOD.parse_audit_blockers('["WEAK_LINEAGE","STALE_CONTEXT"]')
        self.assertEqual(blockers, ["WEAK_LINEAGE", "STALE_CONTEXT"])
        dist = {d["blocker"]: d["count"] for d in self.summary["audit_blockers_distribution"]}
        self.assertGreaterEqual(dist.get("WEAK_LINEAGE", 0), 1)
        self.assertGreaterEqual(dist.get("STALE_CONTEXT", 0), 1)


class TestCompactInputAndTruncation(unittest.TestCase):
    def test_compact_excludes_full_csv(self):
        rows = _fixture_rows()
        # Simulate large total_rows metadata while keeping small lists
        summary = MOD.build_deterministic_summary(rows, source_path=Path("fixture.csv"), top_n=5)
        summary["total_rows"] = 6219
        package = MOD.build_llm_input_package(
            summary, top_n=5, max_llm_rows=60, max_input_chars=40000
        )
        encoded = json.dumps(package)
        self.assertNotIn("all_6219_rows", encoded)
        self.assertTrue(package["_meta"]["full_csv_rows_excluded"])
        self.assertEqual(package["_meta"]["total_source_rows"], 6219)
        # Should not embed thousands of row objects
        self.assertLess(len(package["top_raw_rows_by_max_return"]), 6219)

    def test_max_input_chars_truncation_deterministic(self):
        rows = _fixture_rows()
        big = list(rows)
        for i in range(40):
            r = dict(rows[0])
            r["evidence_row_id"] = f"extra_{i}"
            r["candidate_id"] = f"extra_cand_{i}"
            r["decision_id"] = f"extra_dec_{i}"
            r["pair_address"] = f"0xEXTRA_{i}"
            r["max_return"] = str(5.0 - i * 0.01)
            big.append(r)
        summary = MOD.build_deterministic_summary(big, source_path=Path("fixture.csv"), top_n=30)
        full = MOD.build_llm_input_package(
            summary, top_n=30, max_llm_rows=60, max_input_chars=10**9
        )
        full_chars = full["_meta"]["input_package_chars"]
        # Force truncation while still allowing the min-5 floors to fit
        limit = max(full_chars - 1500, 8000)
        package = MOD.build_llm_input_package(
            summary, top_n=30, max_llm_rows=60, max_input_chars=limit
        )
        meta = package["_meta"]
        self.assertTrue(meta["input_truncated"])
        self.assertLessEqual(meta["input_package_chars"], limit)
        self.assertGreaterEqual(meta["n_raw_included"], 5)
        self.assertGreaterEqual(meta["n_pairs_included"], 5)
        self.assertGreaterEqual(meta["n_candidates_included"], 5)
        self.assertLess(meta["n_raw_included"] + meta["n_pairs_included"] + meta["n_candidates_included"], 90)
        package2 = MOD.build_llm_input_package(
            summary, top_n=30, max_llm_rows=60, max_input_chars=limit
        )
        self.assertEqual(package["_meta"], package2["_meta"])


class TestJsonParsing(unittest.TestCase):
    def test_markdown_fenced_json(self):
        payload = _valid_review("ollama")
        text = "```json\n" + json.dumps(payload) + "\n```"
        parsed, strategy, err = MOD.parse_provider_json(text)
        self.assertIsNone(err)
        self.assertEqual(strategy, "stripped_markdown_fence")
        self.assertEqual(parsed["provider"], "ollama")

    def test_explanatory_text_around_json(self):
        payload = _valid_review("gemini")
        text = "Here is my audit review:\n" + json.dumps(payload) + "\nEnd of review."
        parsed, strategy, err = MOD.parse_provider_json(text)
        self.assertIsNone(err)
        self.assertEqual(strategy, "extracted_balanced_json")
        self.assertEqual(parsed["provider"], "gemini")

    def test_direct_json(self):
        payload = _valid_review("ollama")
        parsed, strategy, err = MOD.parse_provider_json(json.dumps(payload))
        self.assertEqual(strategy, "direct_json")
        self.assertIsNone(err)
        self.assertTrue(parsed)

    def test_malformed_fails(self):
        parsed, strategy, err = MOD.parse_provider_json("not json {{{")
        self.assertIsNone(parsed)
        self.assertEqual(strategy, "failed")
        self.assertIsNotNone(err)


class TestAuthoritySchema(unittest.TestCase):
    def test_authority_fields_enforced(self):
        bad = _valid_review("ollama")
        bad["execution_allowed"] = True
        schema_ok, auth_ok, errors = MOD.validate_review_schema(bad, expected_provider="ollama")
        self.assertTrue(schema_ok)
        self.assertFalse(auth_ok)
        self.assertTrue(any("execution_allowed" in e for e in errors))

    def test_missing_authority_fails(self):
        bad = _valid_review("gemini")
        del bad["live_execution_allowed"]
        schema_ok, auth_ok, errors = MOD.validate_review_schema(bad, expected_provider="gemini")
        self.assertFalse(schema_ok)
        self.assertFalse(auth_ok)


class TestProviderRunArtifacts(unittest.TestCase):
    def test_both_writes_separate_folders_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = _write_fixture_csv(tmp_path / "mw.csv")
            out = tmp_path / "out"
            call_log: list[str] = []
            result = MOD.run_review(
                input_path=csv_path,
                output_root=out,
                provider="both",
                top_n=5,
                max_llm_rows=20,
                max_input_chars=40000,
                call_overrides={
                    "ollama": _ok_call("ollama"),
                    "gemini": _ok_call("gemini"),
                },
                sequential_call_log=call_log,
            )
            ollama_dir = out / "data" / "providers" / "ollama"
            gemini_dir = out / "data" / "providers" / "gemini"
            self.assertTrue((ollama_dir / "llm_review_raw_response.txt").is_file())
            self.assertTrue((gemini_dir / "llm_review_raw_response.txt").is_file())
            self.assertTrue((ollama_dir / "provider_gate.json").is_file())
            self.assertTrue((gemini_dir / "provider_gate.json").is_file())
            # No overwrite collision: both retain distinct content
            o_txt = (ollama_dir / "llm_review_raw_response.txt").read_text(encoding="utf-8")
            g_txt = (gemini_dir / "llm_review_raw_response.txt").read_text(encoding="utf-8")
            self.assertIn("ollama", o_txt)
            self.assertIn("gemini", g_txt)
            self.assertNotEqual(o_txt, g_txt)
            gate = result["gate"]
            self.assertEqual(gate["final_status"], MOD.STATUS_PASS)
            self.assertFalse(gate["mock_used"])
            self.assertTrue(gate["all_required_providers_success"])
            self.assertEqual(call_log, ["ollama", "gemini"])

    def test_mocked_responses_stored_under_provider_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = _write_fixture_csv(tmp_path / "mw.csv")
            out = tmp_path / "out"
            result = MOD.run_review(
                input_path=csv_path,
                output_root=out,
                provider="both",
                top_n=5,
                mock_responses={
                    "ollama": json.dumps(_valid_review("ollama", "qwen3:8b")),
                    "gemini": json.dumps(_valid_review("gemini", "gemini-2.5-flash")),
                },
            )
            self.assertTrue((out / "data" / "providers" / "ollama" / "llm_review_parsed.json").is_file())
            self.assertTrue((out / "data" / "providers" / "gemini" / "llm_review_parsed.json").is_file())
            o_gate = json.loads((out / "data" / "providers" / "ollama" / "provider_gate.json").read_text(encoding="utf-8"))
            g_gate = json.loads((out / "data" / "providers" / "gemini" / "provider_gate.json").read_text(encoding="utf-8"))
            self.assertTrue(o_gate["mock_used"])
            self.assertTrue(g_gate["mock_used"])
            self.assertEqual(o_gate["parse_strategy"], "direct_json")
            self.assertTrue(result["gate"]["mock_used"])
            self.assertNotEqual(result["gate"]["final_status"], MOD.STATUS_PASS)

    def test_malformed_json_provider_fails_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = _write_fixture_csv(tmp_path / "mw.csv")
            out = tmp_path / "out"

            def bad_call(prompt: str, model: str | None = None, **kwargs):
                return {
                    "ok": True,
                    "text": "definitely not json {{{",
                    "raw_json": {},
                    "model": model or "x",
                    "error_type": None,
                    "error_message": None,
                }

            result = MOD.run_review(
                input_path=csv_path,
                output_root=out,
                provider="ollama",
                call_overrides={"ollama": bad_call},
            )
            gate = json.loads((out / "data" / "providers" / "ollama" / "provider_gate.json").read_text(encoding="utf-8"))
            self.assertFalse(gate["success"])
            self.assertEqual(gate["parse_strategy"], "failed")
            self.assertFalse(gate["parse_success"])
            self.assertTrue((out / "data" / "providers" / "ollama" / "llm_review_raw_response.txt").is_file())
            self.assertEqual(result["gate"]["final_status"], MOD.STATUS_PARTIAL_DEBUG)

    def test_provider_gate_records_parse_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = _write_fixture_csv(tmp_path / "mw.csv")
            out = tmp_path / "out"
            fenced = "```json\n" + json.dumps(_valid_review("ollama")) + "\n```"

            def fenced_call(prompt: str, model: str | None = None, **kwargs):
                return {
                    "ok": True,
                    "text": fenced,
                    "raw_json": {},
                    "model": model or "qwen3:8b",
                    "error_type": None,
                    "error_message": None,
                }

            MOD.run_review(
                input_path=csv_path,
                output_root=out,
                provider="ollama",
                call_overrides={"ollama": fenced_call},
            )
            gate = json.loads((out / "data" / "providers" / "ollama" / "provider_gate.json").read_text(encoding="utf-8"))
            self.assertEqual(gate["parse_strategy"], "stripped_markdown_fence")
            self.assertTrue(gate["parse_success"])

    def test_final_gate_requires_both_for_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = _write_fixture_csv(tmp_path / "mw.csv")
            # Only ollama succeeds
            result = MOD.run_review(
                input_path=csv_path,
                output_root=tmp_path / "out1",
                provider="both",
                call_overrides={
                    "ollama": _ok_call("ollama"),
                    "gemini": lambda prompt, model=None, **kw: {
                        "ok": False,
                        "text": "",
                        "raw_json": None,
                        "model": model or "gemini",
                        "error_type": "forced_fail",
                        "error_message": "fail",
                    },
                },
            )
            self.assertEqual(result["gate"]["final_status"], MOD.STATUS_PARTIAL_FAILURE)
            self.assertFalse(result["gate"]["all_required_providers_success"])

    def test_single_provider_debug_status_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = _write_fixture_csv(tmp_path / "mw.csv")
            result = MOD.run_review(
                input_path=csv_path,
                output_root=tmp_path / "out",
                provider="ollama",
                call_overrides={"ollama": _ok_call("ollama")},
            )
            self.assertEqual(result["gate"]["final_status"], MOD.STATUS_PARTIAL_DEBUG)
            self.assertNotEqual(result["gate"]["final_status"], MOD.STATUS_PASS)

    def test_provider_none_deterministic_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = _write_fixture_csv(tmp_path / "mw.csv")
            result = MOD.run_review(
                input_path=csv_path,
                output_root=tmp_path / "out",
                provider="none",
            )
            self.assertEqual(result["gate"]["final_status"], MOD.STATUS_DETERMINISTIC_ONLY)
            self.assertFalse(result["gate"]["llm_called"])

    def test_safety_flags_false(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = _write_fixture_csv(tmp_path / "mw.csv")
            result = MOD.run_review(
                input_path=csv_path,
                output_root=tmp_path / "out",
                provider="both",
                call_overrides={
                    "ollama": _ok_call("ollama"),
                    "gemini": _ok_call("gemini"),
                },
            )
            gate = result["gate"]
            self.assertFalse(gate["db_mutation_attempted"])
            self.assertFalse(gate["execution_attempted"])
            self.assertFalse(gate["paper_execution_attempted"])
            self.assertFalse(gate["live_trading_attempted"])
            self.assertFalse(gate["wallet_access_attempted"])
            self.assertFalse(gate["risk_override_attempted"])
            self.assertEqual(gate["authority_status"], "AUDIT_ONLY_NO_TRADE_AUTHORITY")
            self.assertFalse(gate["mock_used"])
            self.assertTrue(gate["llm_called"])


class TestNoForbiddenImports(unittest.TestCase):
    def test_no_openai_sdk_in_script(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("import openai", source)
        self.assertNotIn("from openai", source)
        self.assertNotIn("OpenAI(", source)

    def test_no_paper_live_execution_imports(self):
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        self.assertNotIn("app.execution.paper", source)
        self.assertNotIn("from app.execution", source)
        self.assertNotIn("open_paper", source)
        self.assertNotIn("close_paper", source)
        self.assertNotIn("connect_wallet", source)
        # Forbid DB mutation APIs / sqlite opens of trader.db (docstring may mention it)
        self.assertNotIn("sqlite3.connect", source)
        self.assertNotIn("Database(", source)
        self.assertNotRegex(source, r"(?m)^\s*(import|from).*trader\.db")
        self.assertNotIn('Path("trader.db")', source)
        self.assertNotIn("Path('trader.db')", source)


class TestSequentialCalls(unittest.TestCase):
    def test_provider_both_sequential(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            csv_path = _write_fixture_csv(tmp_path / "mw.csv")
            order: list[str] = []

            def ollama_call(prompt: str, model: str | None = None, **kwargs):
                order.append("ollama_start")
                out = _ok_call("ollama")(prompt, model=model)
                order.append("ollama_end")
                return out

            def gemini_call(prompt: str, model: str | None = None, **kwargs):
                order.append("gemini_start")
                out = _ok_call("gemini")(prompt, model=model)
                order.append("gemini_end")
                return out

            MOD.run_review(
                input_path=csv_path,
                output_root=tmp_path / "out",
                provider="both",
                call_overrides={"ollama": ollama_call, "gemini": gemini_call},
            )
            self.assertEqual(order, ["ollama_start", "ollama_end", "gemini_start", "gemini_end"])


if __name__ == "__main__":
    unittest.main()
