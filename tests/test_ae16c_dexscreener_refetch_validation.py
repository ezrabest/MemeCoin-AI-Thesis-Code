"""Focused tests for AE16C DexScreener refetch validation (mocked network)."""
from __future__ import annotations

import ast
import csv
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_PATH = ROOT / "scripts" / "run_ae16c_dexscreener_refetch_validation.py"


def _load_mod():
    spec = importlib.util.spec_from_file_location("run_ae16c_dexscreener_refetch_validation", SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


SAMPLE_FIELDS = [
    "combined_target_id",
    "active",
    "chain",
    "provider_pair_url",
    "pair_address",
    "user_supplied_pair_address",
    "resolved_pair_address",
    "target_source",
    "linked_sources",
    "seed_collection",
    "semantic_status",
]


def _base_pair_payload(**overrides: Any) -> dict[str, Any]:
    pair = {
        "chainId": "base",
        "dexId": "uniswap",
        "url": "https://dexscreener.com/base/0xabc",
        "pairAddress": "0xAbC0000000000000000000000000000000000001",
        "baseToken": {"address": "0xbase", "symbol": "BASE", "name": "Base Token"},
        "quoteToken": {"address": "0xquote", "symbol": "USDC", "name": "USD Coin"},
        "priceUsd": "1.23",
        "liquidity": {"usd": 1000},
        "fdv": 5000,
        "marketCap": 4000,
        "volume": {"m5": 1, "h1": 2, "h6": 3, "h24": 4},
        "txns": {
            "m5": {"buys": 1, "sells": 1},
            "h1": {"buys": 2, "sells": 2},
            "h6": {"buys": 3, "sells": 3},
            "h24": {"buys": 4, "sells": 4},
        },
        "priceChange": {"m5": 0.1, "h1": 0.2, "h6": 0.3, "h24": 0.4},
        "pairCreatedAt": 1,
        "info": {"websites": [{"url": "https://example.com"}], "socials": [{"type": "twitter"}]},
    }
    pair.update(overrides)
    return {"pairs": [pair]}


class TestAE16CRefetchValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_mod()

    def test_input_path_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            out = self.mod.run(Path(tmp) / "missing.csv", output_root=Path(tmp) / "out", dry_run=True)
            self.assertFalse(out["manifest"]["input_exists"])
            self.assertEqual(out["gate"]["classification"], "AE16C_REFETCH_VALIDATION_BLOCKED_NO_TARGETS")

    def test_authoritative_id_uses_pair_address_for_clean_forward(self) -> None:
        row = {
            "target_source": "CLEAN_FORWARD_EXISTING",
            "linked_sources": "CLEAN_FORWARD_EXISTING",
            "chain": "solana",
            "pair_address": "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd",
            "user_supplied_pair_address": "",
            "resolved_pair_address": "should_not_win",
            "provider_pair_url": "https://dexscreener.com/solana/2uf4xh61rdwxng9woyxsvqp7zua6klfpb3nvnrqeoisd",
        }
        pair_id, source = self.mod.select_refetch_pair_id(row)
        self.assertEqual(pair_id, "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd")
        self.assertEqual(source, "PAIR_ADDRESS_CLEAN_FORWARD")

    def test_provider_pair_url_not_authoritative_for_clean_solana(self) -> None:
        row = {
            "target_source": "CLEAN_FORWARD_EXISTING",
            "linked_sources": "CLEAN_FORWARD_EXISTING",
            "chain": "solana",
            "pair_address": "AbCMixedCasePair111111111111111111111111",
            "provider_pair_url": "https://dexscreener.com/solana/abcmixedcasepair111111111111111111111111",
        }
        pair_id, source = self.mod.select_refetch_pair_id(row)
        self.assertNotEqual(pair_id, "abcmixedcasepair111111111111111111111111")
        self.assertEqual(source, "PAIR_ADDRESS_CLEAN_FORWARD")
        self.assertEqual(pair_id, "AbCMixedCasePair111111111111111111111111")

    def test_solana_and_xrpl_casing_preserved(self) -> None:
        sol = "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd"
        xrpl = "43554c5400000000000000000000000000000000.rcultakrkbqjk1tpmg5hkw4dpcf9s9kcs_xrp"
        sol_id, _ = self.mod.select_refetch_pair_id(
            {
                "target_source": "CLEAN_FORWARD_EXISTING",
                "linked_sources": "CLEAN_FORWARD_EXISTING",
                "pair_address": sol,
                "provider_pair_url": f"https://dexscreener.com/solana/{sol.lower()}",
            }
        )
        xrpl_id, _ = self.mod.select_refetch_pair_id(
            {
                "target_source": "USER_DEXSCREENER_SEED",
                "linked_sources": "USER_DEXSCREENER_SEED",
                "user_supplied_pair_address": xrpl,
                "provider_pair_url": f"https://dexscreener.com/xrpl/{xrpl}",
            }
        )
        self.assertEqual(sol_id, sol)
        self.assertEqual(xrpl_id, xrpl)

    def test_evm_case_insensitive_match_preserves_original(self) -> None:
        expected = "0xAbC0000000000000000000000000000000000001"
        self.assertTrue(self.mod.addresses_match("base", expected, expected.lower()))
        row = {
            "combined_target_id": "t1",
            "active": "true",
            "chain": "base",
            "target_source": "USER_DEXSCREENER_SEED",
            "linked_sources": "USER_DEXSCREENER_SEED",
            "seed_collection": "USER_SEED_REFI",
            "semantic_status": "PENDING_SYSTEM_CLASSIFICATION",
            "provider_pair_url": f"https://dexscreener.com/base/{expected.lower()}",
            "pair_address": "",
            "user_supplied_pair_address": expected,
            "resolved_pair_address": "",
        }
        payload = _base_pair_payload(pairAddress=expected.lower(), chainId="base")

        def fetch_fn(url: str, timeout: float) -> dict[str, Any]:
            return {
                "ok": True,
                "status_code": 200,
                "raw_text": json.dumps(payload),
                "json": payload,
                "json_parse_error": "",
                "exception_type": "",
                "exception_message": "",
                "headers": {},
            }

        vrow, _, _ = self.mod.validate_target(row, fetch_fn=fetch_fn, sleeper=lambda _s: None)
        self.assertEqual(vrow["refetch_pair_id"], expected)
        self.assertEqual(vrow["acceptance_status"], "PROVIDER_PAIR_RESOLVED")
        self.assertEqual(vrow["clean_forward_candidate_ready"], "true")

    def test_provider_response_parsed_into_base_quote_fields(self) -> None:
        fields = self.mod.parse_pair_fields(_base_pair_payload()["pairs"][0])
        self.assertEqual(fields["provider_base_token_address"], "0xbase")
        self.assertEqual(fields["provider_quote_token_address"], "0xquote")
        self.assertEqual(fields["provider_base_token_symbol"], "BASE")
        self.assertEqual(fields["info_websites_count"], "1")
        self.assertEqual(fields["info_socials_types"], "twitter")

    def test_missing_nested_payload_fields_do_not_raise(self) -> None:
        sparse = {"pairAddress": "0x1", "chainId": "base"}  # no baseToken/quote/liquidity/txns
        fields = self.mod.parse_pair_fields(sparse)
        self.assertEqual(fields["provider_base_token_address"], "")
        self.assertEqual(fields["liquidity_usd"], "")
        self.assertEqual(fields["txns_h24_buys"], "")
        # extract + validate should not KeyError
        pairs = self.mod.extract_pairs({"pairs": [sparse]})
        self.assertEqual(len(pairs), 1)

    def test_empty_provider_response_not_found(self) -> None:
        row = {
            "combined_target_id": "empty1",
            "active": "true",
            "chain": "base",
            "target_source": "USER_DEXSCREENER_SEED",
            "linked_sources": "USER_DEXSCREENER_SEED",
            "seed_collection": "USER_SEED_REFI",
            "semantic_status": "PENDING_SYSTEM_CLASSIFICATION",
            "provider_pair_url": "https://dexscreener.com/base/0x1",
            "pair_address": "",
            "user_supplied_pair_address": "0x1",
            "resolved_pair_address": "",
        }

        def fetch_fn(url: str, timeout: float) -> dict[str, Any]:
            return {
                "ok": True,
                "status_code": 200,
                "raw_text": '{"pairs":null}',
                "json": {"pairs": None},
                "json_parse_error": "",
                "exception_type": "",
                "exception_message": "",
                "headers": {},
            }

        vrow, jrec, _ = self.mod.validate_target(row, fetch_fn=fetch_fn, sleeper=lambda _s: None)
        self.assertEqual(vrow["acceptance_status"], "PROVIDER_PAIR_NOT_FOUND")
        self.assertEqual(jrec["acceptance_status"], "PROVIDER_PAIR_NOT_FOUND")

    def test_http_404_writes_valid_jsonl_record(self) -> None:
        row = {
            "combined_target_id": "nf1",
            "active": "true",
            "chain": "base",
            "target_source": "USER_DEXSCREENER_SEED",
            "linked_sources": "USER_DEXSCREENER_SEED",
            "seed_collection": "USER_SEED_REFI",
            "semantic_status": "PENDING_SYSTEM_CLASSIFICATION",
            "provider_pair_url": "https://dexscreener.com/base/0xdead",
            "pair_address": "",
            "user_supplied_pair_address": "0xdead",
            "resolved_pair_address": "",
        }

        def fetch_fn(url: str, timeout: float) -> dict[str, Any]:
            return {
                "ok": False,
                "status_code": 404,
                "raw_text": "not found",
                "json": {"raw_text_preview": "not found", "json_parse_error": ""},
                "exception_type": "",
                "exception_message": "",
                "headers": {},
            }

        vrow, jrec, _ = self.mod.validate_target(row, fetch_fn=fetch_fn, sleeper=lambda _s: None)
        self.assertEqual(vrow["acceptance_status"], "PROVIDER_PAIR_NOT_FOUND")
        for key in (
            "combined_target_id",
            "chain",
            "refetch_pair_id",
            "refetch_url",
            "fetched_at_utc",
            "http_attempted",
            "http_status_code",
            "http_success",
            "acceptance_status",
            "rejection_reason",
            "exception_type",
            "exception_message",
            "raw_response_sha256",
            "raw_response_json",
        ):
            self.assertIn(key, jrec)

    def test_http_429_triggers_retry_and_jsonl(self) -> None:
        calls = {"n": 0}
        sleeps: list[float] = []

        def fetch_fn(url: str, timeout: float) -> dict[str, Any]:
            calls["n"] += 1
            return {
                "ok": False,
                "status_code": 429,
                "raw_text": "rate limit",
                "json": {"raw_text_preview": "rate limit"},
                "exception_type": "",
                "exception_message": "",
                "headers": {"retry-after": "1"},
            }

        row = {
            "combined_target_id": "rl1",
            "active": "true",
            "chain": "base",
            "target_source": "USER_DEXSCREENER_SEED",
            "linked_sources": "USER_DEXSCREENER_SEED",
            "seed_collection": "USER_SEED_REFI",
            "semantic_status": "PENDING_SYSTEM_CLASSIFICATION",
            "provider_pair_url": "https://dexscreener.com/base/0x1",
            "pair_address": "",
            "user_supplied_pair_address": "0x1",
            "resolved_pair_address": "",
        }
        vrow, jrec, meta = self.mod.validate_target(
            row,
            fetch_fn=fetch_fn,
            sleeper=lambda s: sleeps.append(s),
            max_retries=3,
            backoff_base=0.01,
            backoff_max=1.0,
        )
        self.assertEqual(vrow["acceptance_status"], "PROVIDER_RATE_LIMITED")
        self.assertGreaterEqual(calls["n"], 3)
        self.assertGreaterEqual(int(vrow["retry_count"]), 2)
        self.assertTrue(meta["rate_limited"])
        self.assertEqual(jrec["acceptance_status"], "PROVIDER_RATE_LIMITED")

    def test_http_5xx_triggers_retry(self) -> None:
        calls = {"n": 0}

        def fetch_fn(url: str, timeout: float) -> dict[str, Any]:
            calls["n"] += 1
            return {
                "ok": False,
                "status_code": 503,
                "raw_text": "unavailable",
                "json": {"raw_text_preview": "unavailable"},
                "exception_type": "",
                "exception_message": "",
                "headers": {},
            }

        row = {
            "combined_target_id": "e503",
            "active": "true",
            "chain": "base",
            "target_source": "USER_DEXSCREENER_SEED",
            "linked_sources": "USER_DEXSCREENER_SEED",
            "seed_collection": "USER_SEED_REFI",
            "semantic_status": "PENDING_SYSTEM_CLASSIFICATION",
            "provider_pair_url": "https://dexscreener.com/base/0x1",
            "pair_address": "",
            "user_supplied_pair_address": "0x1",
            "resolved_pair_address": "",
        }
        vrow, _, meta = self.mod.validate_target(
            row,
            fetch_fn=fetch_fn,
            sleeper=lambda _s: None,
            max_retries=3,
            backoff_base=0.01,
            backoff_max=1.0,
        )
        self.assertEqual(vrow["acceptance_status"], "PROVIDER_HTTP_ERROR")
        self.assertEqual(calls["n"], 3)
        self.assertTrue(meta["retryable_failure"])

    def test_timeout_becomes_provider_timeout(self) -> None:
        def fetch_fn(url: str, timeout: float) -> dict[str, Any]:
            return {
                "ok": False,
                "status_code": None,
                "raw_text": "",
                "json": None,
                "exception_type": "ReadTimeout",
                "exception_message": "timed out",
                "is_timeout": True,
                "headers": {},
            }

        row = {
            "combined_target_id": "to1",
            "active": "true",
            "chain": "base",
            "target_source": "USER_DEXSCREENER_SEED",
            "linked_sources": "USER_DEXSCREENER_SEED",
            "seed_collection": "USER_SEED_REFI",
            "semantic_status": "PENDING_SYSTEM_CLASSIFICATION",
            "provider_pair_url": "https://dexscreener.com/base/0x1",
            "pair_address": "",
            "user_supplied_pair_address": "0x1",
            "resolved_pair_address": "",
        }
        vrow, _, _ = self.mod.validate_target(
            row, fetch_fn=fetch_fn, sleeper=lambda _s: None, max_retries=1
        )
        self.assertEqual(vrow["acceptance_status"], "PROVIDER_TIMEOUT")

    def test_request_exception_becomes_provider_exception(self) -> None:
        def fetch_fn(url: str, timeout: float) -> dict[str, Any]:
            return {
                "ok": False,
                "status_code": None,
                "raw_text": "",
                "json": None,
                "exception_type": "ConnectError",
                "exception_message": "connection reset",
                "headers": {},
            }

        row = {
            "combined_target_id": "ex1",
            "active": "true",
            "chain": "base",
            "target_source": "USER_DEXSCREENER_SEED",
            "linked_sources": "USER_DEXSCREENER_SEED",
            "seed_collection": "USER_SEED_REFI",
            "semantic_status": "PENDING_SYSTEM_CLASSIFICATION",
            "provider_pair_url": "https://dexscreener.com/base/0x1",
            "pair_address": "",
            "user_supplied_pair_address": "0x1",
            "resolved_pair_address": "",
        }
        vrow, _, _ = self.mod.validate_target(
            row, fetch_fn=fetch_fn, sleeper=lambda _s: None, max_retries=1
        )
        self.assertEqual(vrow["acceptance_status"], "PROVIDER_EXCEPTION")

    def test_json_parse_error_status(self) -> None:
        def fetch_fn(url: str, timeout: float) -> dict[str, Any]:
            return {
                "ok": True,
                "status_code": 200,
                "raw_text": "<html>nope</html>",
                "json": {"raw_text_preview": "<html>nope</html>", "json_parse_error": "JSONDecodeError"},
                "json_parse_error": "JSONDecodeError: Expecting value",
                "exception_type": "",
                "exception_message": "",
                "headers": {},
            }

        row = {
            "combined_target_id": "jp1",
            "active": "true",
            "chain": "base",
            "target_source": "USER_DEXSCREENER_SEED",
            "linked_sources": "USER_DEXSCREENER_SEED",
            "seed_collection": "USER_SEED_REFI",
            "semantic_status": "PENDING_SYSTEM_CLASSIFICATION",
            "provider_pair_url": "https://dexscreener.com/base/0x1",
            "pair_address": "",
            "user_supplied_pair_address": "0x1",
            "resolved_pair_address": "",
        }
        vrow, _, _ = self.mod.validate_target(row, fetch_fn=fetch_fn, sleeper=lambda _s: None)
        self.assertEqual(vrow["acceptance_status"], "PROVIDER_JSON_PARSE_ERROR")

    def test_dry_run_does_not_call_network(self) -> None:
        called = {"n": 0}

        def fetch_fn(url: str, timeout: float) -> dict[str, Any]:
            called["n"] += 1
            raise AssertionError("network should not be called in dry-run")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inp = tmp_path / "in.csv"
            _write_csv(
                inp,
                [
                    {
                        "combined_target_id": "d1",
                        "active": "true",
                        "chain": "base",
                        "provider_pair_url": "https://dexscreener.com/base/0x1",
                        "pair_address": "",
                        "user_supplied_pair_address": "0x1",
                        "resolved_pair_address": "",
                        "target_source": "USER_DEXSCREENER_SEED",
                        "linked_sources": "USER_DEXSCREENER_SEED",
                        "seed_collection": "USER_SEED_REFI",
                        "semantic_status": "PENDING_SYSTEM_CLASSIFICATION",
                    }
                ],
                SAMPLE_FIELDS,
            )
            out = self.mod.run(
                inp,
                output_root=tmp_path / "out",
                dry_run=True,
                fetch_fn=fetch_fn,
                sleeper=lambda _s: None,
            )
            self.assertEqual(called["n"], 0)
            self.assertEqual(out["rows"][0]["acceptance_status"], "DRY_RUN_NOT_FETCHED")
            self.assertEqual(out["rows"][0]["semantic_status"], "PENDING_SYSTEM_CLASSIFICATION")

    def test_inactive_target_writes_skipped_jsonl_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inp = tmp_path / "in.csv"
            _write_csv(
                inp,
                [
                    {
                        "combined_target_id": "inactive1",
                        "active": "false",
                        "chain": "base",
                        "provider_pair_url": "https://dexscreener.com/base/0x1",
                        "pair_address": "",
                        "user_supplied_pair_address": "0x1",
                        "resolved_pair_address": "",
                        "target_source": "USER_DEXSCREENER_SEED",
                        "linked_sources": "USER_DEXSCREENER_SEED",
                        "seed_collection": "USER_SEED_REFI",
                        "semantic_status": "PENDING_SYSTEM_CLASSIFICATION",
                    }
                ],
                SAMPLE_FIELDS,
            )
            out = self.mod.run(
                inp,
                output_root=tmp_path / "out",
                dry_run=False,
                fetch_fn=lambda u, t: (_ for _ in ()).throw(AssertionError("no fetch")),
                sleeper=lambda _s: None,
            )
            self.assertEqual(out["rows"][0]["acceptance_status"], "TARGET_INACTIVE_SKIPPED")
            jsonl = (tmp_path / "out" / "data" / "ae16c_provider_responses.jsonl").read_text(
                encoding="utf-8"
            ).strip()
            rec = json.loads(jsonl)
            self.assertEqual(rec["acceptance_status"], "TARGET_INACTIVE_SKIPPED")
            self.assertEqual(out["manifest"]["jsonl_lines_written"], 1)
            self.assertEqual(out["manifest"]["jsonl_lines_expected"], 1)

    def test_jsonl_lines_equal_targets_processed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inp = tmp_path / "in.csv"
            rows = []
            for i in range(3):
                rows.append(
                    {
                        "combined_target_id": f"t{i}",
                        "active": "true",
                        "chain": "base",
                        "provider_pair_url": f"https://dexscreener.com/base/0x{i}",
                        "pair_address": "",
                        "user_supplied_pair_address": f"0x{i}",
                        "resolved_pair_address": "",
                        "target_source": "USER_DEXSCREENER_SEED",
                        "linked_sources": "USER_DEXSCREENER_SEED",
                        "seed_collection": "USER_SEED_REFI",
                        "semantic_status": "PENDING_SYSTEM_CLASSIFICATION",
                    }
                )
            _write_csv(inp, rows, SAMPLE_FIELDS)
            out = self.mod.run(
                inp,
                output_root=tmp_path / "out",
                dry_run=True,
                sleeper=lambda _s: None,
            )
            self.assertEqual(out["manifest"]["jsonl_lines_written"], 3)
            self.assertEqual(out["manifest"]["jsonl_lines_expected"], 3)
            self.assertTrue(out["manifest"]["jsonl_error_safety_passed"])
            lines = [
                ln
                for ln in (tmp_path / "out" / "data" / "ae16c_provider_responses.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if ln.strip()
            ]
            self.assertEqual(len(lines), 3)

    def test_rejected_export_preserves_source_fields(self) -> None:
        """Rejected CSV must not blank provenance / identity fields (AE16C-FIX)."""
        sol_url = "https://dexscreener.com/solana/a434xwjq6beuj3ufg1srpr7fm8eixoez1ux1qjzgx19e"
        sol_pair = "a434xwjq6beuj3ufg1srpr7fm8eixoez1ux1qjzgx19e"
        # Provider returns different casing -> CASE_SENSITIVE reject path
        provider_canonical = "A434xWjq6Beuj3uFg1SrpR7fM8EixoeZ1Ux1QjZgX19e"

        def fetch_fn(url: str, timeout: float) -> dict[str, Any]:
            payload = {
                "pairs": [
                    {
                        "chainId": "solana",
                        "dexId": "raydium",
                        "url": f"https://dexscreener.com/solana/{provider_canonical}",
                        "pairAddress": provider_canonical,
                        "baseToken": {"address": "BaseMint111", "symbol": "TOK", "name": "Tok"},
                        "quoteToken": {"address": "QuoteMint111", "symbol": "USDC", "name": "USDC"},
                        "priceUsd": "1.0",
                        "liquidity": {"usd": 100},
                    }
                ]
            }
            return {
                "ok": True,
                "status_code": 200,
                "raw_text": json.dumps(payload),
                "json": payload,
                "json_parse_error": "",
                "exception_type": "",
                "exception_message": "",
                "headers": {},
            }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inp = tmp_path / "in.csv"
            _write_csv(
                inp,
                [
                    {
                        "combined_target_id": "rej_sol_1",
                        "active": "true",
                        "chain": "solana",
                        "provider_pair_url": sol_url,
                        "pair_address": "",
                        "user_supplied_pair_address": sol_pair,
                        "resolved_pair_address": "",
                        "target_source": "USER_DEXSCREENER_SEED",
                        "linked_sources": "USER_DEXSCREENER_SEED",
                        "seed_collection": "USER_SEED_REFI",
                        "semantic_status": "PENDING_SYSTEM_CLASSIFICATION",
                    }
                ],
                SAMPLE_FIELDS,
            )
            out = self.mod.run(
                inp,
                output_root=tmp_path / "out",
                dry_run=False,
                fetch_fn=fetch_fn,
                sleeper=lambda _s: None,
                sleep_seconds=0.0,
            )
            rejected_path = tmp_path / "out" / "data" / "ae16c_rejected_targets.csv"
            self.assertTrue(rejected_path.exists())
            rejected = list(csv.DictReader(rejected_path.open(encoding="utf-8")))
            self.assertEqual(len(rejected), 1)
            row = rejected[0]
            self.assertEqual(row["provider_pair_url"], sol_url)
            self.assertEqual(row["user_supplied_pair_address"], sol_pair)
            self.assertEqual(row["seed_collection"], "USER_SEED_REFI")
            self.assertEqual(row["semantic_status"], "PENDING_SYSTEM_CLASSIFICATION")
            self.assertEqual(row["target_source"], "USER_DEXSCREENER_SEED")
            self.assertEqual(row["linked_sources"], "USER_DEXSCREENER_SEED")
            self.assertEqual(row["refetch_pair_id"], sol_pair)
            # No blanking of required source fields
            for key in (
                "provider_pair_url",
                "user_supplied_pair_address",
                "seed_collection",
                "semantic_status",
                "target_source",
                "linked_sources",
                "refetch_pair_id",
            ):
                self.assertTrue(str(row.get(key) or "").strip(), msg=f"{key} blanked")
            self.assertEqual(out["rows"][0]["acceptance_status"], "CASE_SENSITIVE_PAIR_ID_UNRESOLVED")

    def test_no_collector_or_db_mutation_and_semantic_pending(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("app", imported)
        self.assertNotIn("sqlite3", imported)
        self.assertNotIn("uvicorn", imported)
        self.assertNotIn("fastapi", imported)
        self.assertNotIn('open("trader.db"', source)
        self.assertNotIn("clean_forward_collector", source)
        self.assertIn("PENDING_SYSTEM_CLASSIFICATION", source)

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inp = tmp_path / "in.csv"
            _write_csv(
                inp,
                [
                    {
                        "combined_target_id": "s1",
                        "active": "true",
                        "chain": "solana",
                        "provider_pair_url": "https://dexscreener.com/solana/abc",
                        "pair_address": "AbC",
                        "user_supplied_pair_address": "",
                        "resolved_pair_address": "AbC",
                        "target_source": "CLEAN_FORWARD_EXISTING",
                        "linked_sources": "CLEAN_FORWARD_EXISTING",
                        "seed_collection": "EXISTING_CLEAN_FORWARD",
                        "semantic_status": "SHOULD_BE_OVERWRITTEN",
                    }
                ],
                SAMPLE_FIELDS,
            )
            out = self.mod.run(inp, output_root=tmp_path / "out", dry_run=True)
            self.assertEqual(out["rows"][0]["semantic_status"], "PENDING_SYSTEM_CLASSIFICATION")
            self.assertFalse(out["manifest"]["collector_modified"])
            self.assertFalse(out["manifest"]["trader_db_mutated"])
            self.assertFalse(out["manifest"]["server_required"])
            self.assertFalse(out["manifest"]["internal_api_called"])


if __name__ == "__main__":
    unittest.main()
