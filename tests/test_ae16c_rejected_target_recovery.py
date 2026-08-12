"""Focused mocked tests for AE16C-R rejected target recovery + export fix coverage."""
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
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

SCRIPT_PATH = ROOT / "scripts" / "run_ae16c_rejected_target_recovery.py"


def _load_mod():
    spec = importlib.util.spec_from_file_location("run_ae16c_rejected_target_recovery", SCRIPT_PATH)
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


CANON_FIELDS = [
    "combined_target_id",
    "chain",
    "refetch_pair_id",
    "acceptance_status",
    "target_source",
    "seed_collection",
    "provider_pair_url_fixed",
    "user_supplied_pair_address_fixed",
    "pair_address_fixed",
    "resolved_pair_address_fixed",
    "refetch_pair_id_fixed",
    "seed_collection_fixed",
    "target_source_fixed",
    "linked_sources_fixed",
    "semantic_status_fixed",
    "recovery_input_ready",
]

READY_FIELDS = [
    "combined_target_id",
    "chain",
    "refetch_pair_id",
    "provider_pair_address",
    "provider_chain_id",
    "provider_base_token_address",
    "provider_quote_token_address",
    "price_usd",
    "liquidity_usd",
    "acceptance_status",
    "clean_forward_candidate_ready",
    "target_source",
    "seed_collection",
    "semantic_status",
]


def _pair(
    *,
    chain: str,
    pair_address: str,
    url: str | None = None,
    base: str = "BaseMint1111111111111111111111111111111",
    quote: str = "QuoteMint11111111111111111111111111111",
) -> dict[str, Any]:
    return {
        "chainId": chain,
        "dexId": "raydium",
        "url": url or f"https://dexscreener.com/{chain}/{pair_address}",
        "pairAddress": pair_address,
        "baseToken": {"address": base, "symbol": "TOK", "name": "Token"},
        "quoteToken": {"address": quote, "symbol": "USDC", "name": "USDC"},
        "priceUsd": "1.25",
        "liquidity": {"usd": 999},
        "fdv": 1000,
        "marketCap": 900,
        "volume": {"m5": 1, "h1": 2, "h6": 3, "h24": 4},
        "txns": {
            "m5": {"buys": 1, "sells": 1},
            "h1": {"buys": 2, "sells": 2},
            "h6": {"buys": 3, "sells": 3},
            "h24": {"buys": 4, "sells": 4},
        },
        "priceChange": {"m5": 0.1, "h1": 0.2, "h6": 0.3, "h24": 0.4},
        "pairCreatedAt": 1,
        "info": {"websites": [], "socials": []},
    }


class TestAE16CRejectedTargetRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_mod()

    def test_canonical_recovery_input_loading(self) -> None:
        raw = {
            "combined_target_id": "ae16b_x",
            "chain": "solana",
            "provider_pair_url_fixed": "https://dexscreener.com/solana/abc",
            "user_supplied_pair_address_fixed": "abc",
            "refetch_pair_id_fixed": "abc",
            "seed_collection_fixed": "USER_SEED_REFI",
            "target_source_fixed": "USER_DEXSCREENER_SEED",
            "linked_sources_fixed": "USER_DEXSCREENER_SEED",
            "semantic_status_fixed": "SHOULD_STAY_PENDING",
        }
        norm = self.mod.normalize_recovery_input_row(raw)
        self.assertEqual(norm["provider_pair_url"], "https://dexscreener.com/solana/abc")
        self.assertEqual(norm["user_supplied_pair_address"], "abc")
        self.assertEqual(norm["seed_collection"], "USER_SEED_REFI")
        self.assertEqual(norm["semantic_status"], "PENDING_SYSTEM_CLASSIFICATION")

    def test_search_queries_urlencoded_and_deduped(self) -> None:
        row = {
            "user_supplied_pair_address": "abcXYZ",
            "refetch_pair_id": "abcXYZ",
            "provider_pair_url": "https://dexscreener.com/solana/abcXYZ",
            "pair_address": "",
            "resolved_pair_address": "other",
        }
        all_q, deduped = self.mod.build_search_queries(row)
        self.assertEqual(all_q.count("abcXYZ"), 3)
        self.assertEqual(deduped, ["abcXYZ", "other"])
        encoded = self.mod.url_encode_query("a/b c")
        self.assertEqual(encoded, quote("a/b c", safe=""))
        self.assertIn("%2F", self.mod.build_search_url("a/b"))

    def test_solana_lowercase_recovered_via_search_canonical(self) -> None:
        rejected_lower = "a434xwjq6beuj3ufg1srpr7fm8eixoez1ux1qjzgx19e"
        canonical = "A434xWjq6Beuj3uFg1SrpR7fM8EixoeZ1Ux1QjZgX19e"
        calls: list[str] = []

        def fetch_fn(url: str, timeout: float) -> dict[str, Any]:
            calls.append(url)
            if "/search?" in url:
                payload = {"pairs": [_pair(chain="solana", pair_address=canonical)]}
            else:
                # exact refetch
                self.assertIn(canonical, url)
                payload = {"pairs": [_pair(chain="solana", pair_address=canonical)]}
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
            inp = tmp_path / "rejected.csv"
            ready = tmp_path / "ready.csv"
            _write_csv(
                inp,
                [
                    {
                        "combined_target_id": "rej1",
                        "chain": "solana",
                        "refetch_pair_id": rejected_lower,
                        "acceptance_status": "CASE_SENSITIVE_PAIR_ID_UNRESOLVED",
                        "target_source": "USER_DEXSCREENER_SEED",
                        "seed_collection": "USER_SEED_REFI",
                        "provider_pair_url_fixed": f"https://dexscreener.com/solana/{rejected_lower}",
                        "user_supplied_pair_address_fixed": rejected_lower,
                        "pair_address_fixed": "",
                        "resolved_pair_address_fixed": "",
                        "refetch_pair_id_fixed": rejected_lower,
                        "seed_collection_fixed": "USER_SEED_REFI",
                        "target_source_fixed": "USER_DEXSCREENER_SEED",
                        "linked_sources_fixed": "USER_DEXSCREENER_SEED",
                        "semantic_status_fixed": "PENDING_SYSTEM_CLASSIFICATION",
                        "recovery_input_ready": "true",
                    }
                ],
                CANON_FIELDS,
            )
            _write_csv(
                ready,
                [
                    {
                        "combined_target_id": "ready1",
                        "chain": "base",
                        "refetch_pair_id": "0x1",
                        "provider_pair_address": "0x1",
                        "provider_chain_id": "base",
                        "provider_base_token_address": "0xb",
                        "provider_quote_token_address": "0xq",
                        "price_usd": "1",
                        "liquidity_usd": "2",
                        "acceptance_status": "PROVIDER_PAIR_RESOLVED",
                        "clean_forward_candidate_ready": "true",
                        "target_source": "CLEAN_FORWARD_EXISTING",
                        "seed_collection": "EXISTING_CLEAN_FORWARD",
                        "semantic_status": "PENDING_SYSTEM_CLASSIFICATION",
                    }
                ],
                READY_FIELDS,
            )
            out = self.mod.run(
                inp,
                ready_path=ready,
                output_root=tmp_path / "out",
                fetch_fn=fetch_fn,
                sleeper=lambda _s: None,
                sleep_seconds=0.0,
                max_http_calls=20,
            )
            self.assertEqual(len(out["recovered"]), 1)
            rec = out["recovered"][0]
            self.assertEqual(rec["provider_pair_address"], canonical)
            self.assertEqual(rec["recovery_status"], "RECOVERED_EXACT_REFETCH_CONFIRMED")
            self.assertEqual(rec["semantic_status"], "PENDING_SYSTEM_CLASSIFICATION")
            self.assertEqual(rec["seed_collection"], "USER_SEED_REFI")
            # No local casing guess: query sent as stored lowercase
            self.assertTrue(any(rejected_lower in u for u in calls if "/search?" in u))
            self.assertNotIn(canonical.lower() + "UPPER", "".join(calls))
            # Merged = original ready + recovered
            self.assertEqual(len(out["merged"]), 2)
            ids = {r["combined_target_id"] for r in out["merged"]}
            self.assertEqual(ids, {"ready1", "rej1"})

    def test_no_local_casing_guess_in_query_builder(self) -> None:
        row = {
            "user_supplied_pair_address": "AbC",
            "refetch_pair_id": "AbC",
            "provider_pair_url": "https://dexscreener.com/solana/AbC",
            "pair_address": "",
            "resolved_pair_address": "",
        }
        _, deduped = self.mod.build_search_queries(row)
        self.assertEqual(deduped, ["AbC"])
        self.assertNotIn("abc", deduped)
        self.assertNotIn("ABC", deduped)

    def test_base_token_match_alone_cannot_recover(self) -> None:
        query = "TokenMintOnlyMatch11111111111111111111111"
        other_pair = "DifferentPairAddress1111111111111111111"

        def fetch_fn(url: str, timeout: float) -> dict[str, Any]:
            if "/search?" in url:
                payload = {
                    "pairs": [
                        _pair(
                            chain="solana",
                            pair_address=other_pair,
                            base=query,  # token match only
                        )
                    ]
                }
            else:
                raise AssertionError("exact refetch must not run without strong pair match")
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
            ready = tmp_path / "ready.csv"
            _write_csv(ready, [], READY_FIELDS)
            _write_csv(
                inp,
                [
                    {
                        "combined_target_id": "tok_only",
                        "chain": "solana",
                        "refetch_pair_id": query,
                        "acceptance_status": "CASE_SENSITIVE_PAIR_ID_UNRESOLVED",
                        "target_source": "USER_DEXSCREENER_SEED",
                        "seed_collection": "USER_SEED_OPPORTUNISTIC",
                        "provider_pair_url_fixed": f"https://dexscreener.com/solana/{query}",
                        "user_supplied_pair_address_fixed": query,
                        "pair_address_fixed": "",
                        "resolved_pair_address_fixed": "",
                        "refetch_pair_id_fixed": query,
                        "seed_collection_fixed": "USER_SEED_OPPORTUNISTIC",
                        "target_source_fixed": "USER_DEXSCREENER_SEED",
                        "linked_sources_fixed": "USER_DEXSCREENER_SEED",
                        "semantic_status_fixed": "PENDING_SYSTEM_CLASSIFICATION",
                        "recovery_input_ready": "true",
                    }
                ],
                CANON_FIELDS,
            )
            out = self.mod.run(
                inp,
                ready_path=ready,
                output_root=tmp_path / "out",
                fetch_fn=fetch_fn,
                sleeper=lambda _s: None,
                sleep_seconds=0.0,
            )
            self.assertEqual(len(out["recovered"]), 0)
            self.assertEqual(out["still_rejected"][0]["recovery_status"], "STILL_REJECTED_NO_STRONG_MATCH")

    def test_ambiguous_search_remains_rejected(self) -> None:
        q = "samequery111111111111111111111111111111"

        def fetch_fn(url: str, timeout: float) -> dict[str, Any]:
            payload = {
                "pairs": [
                    _pair(chain="solana", pair_address="PairAAAA111111111111111111111111111111"),
                    _pair(chain="solana", pair_address="PairBBBB111111111111111111111111111111"),
                ]
            }
            # Both URLs/paths won't match q; force strong via putting q into pairAddress differently
            # Instead make both pairAddresses ci-equal to q with different casing? That would be same addr.
            # Use url path match with two different pairs that both have url ending in q - impossible.
            # Score both as strong via user_supplied match on url path:
            payload = {
                "pairs": [
                    _pair(
                        chain="solana",
                        pair_address="CanonAAAA11111111111111111111111111111",
                        url=f"https://dexscreener.com/solana/{q}",
                    ),
                    _pair(
                        chain="solana",
                        pair_address="CanonBBBB11111111111111111111111111111",
                        url=f"https://dexscreener.com/solana/{q.upper()}",
                    ),
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
            ready = tmp_path / "ready.csv"
            _write_csv(ready, [], READY_FIELDS)
            _write_csv(
                inp,
                [
                    {
                        "combined_target_id": "amb1",
                        "chain": "solana",
                        "refetch_pair_id": q,
                        "acceptance_status": "CASE_SENSITIVE_PAIR_ID_UNRESOLVED",
                        "target_source": "USER_DEXSCREENER_SEED",
                        "seed_collection": "USER_SEED_REFI",
                        "provider_pair_url_fixed": f"https://dexscreener.com/solana/{q}",
                        "user_supplied_pair_address_fixed": q,
                        "pair_address_fixed": "",
                        "resolved_pair_address_fixed": "",
                        "refetch_pair_id_fixed": q,
                        "seed_collection_fixed": "USER_SEED_REFI",
                        "target_source_fixed": "USER_DEXSCREENER_SEED",
                        "linked_sources_fixed": "USER_DEXSCREENER_SEED",
                        "semantic_status_fixed": "PENDING_SYSTEM_CLASSIFICATION",
                        "recovery_input_ready": "true",
                    }
                ],
                CANON_FIELDS,
            )
            out = self.mod.run(
                inp,
                ready_path=ready,
                output_root=tmp_path / "out",
                fetch_fn=fetch_fn,
                sleeper=lambda _s: None,
                sleep_seconds=0.0,
            )
            self.assertEqual(out["still_rejected"][0]["recovery_status"], "STILL_REJECTED_SEARCH_AMBIGUOUS")

    def test_exact_refetch_required_and_mismatch_rejected(self) -> None:
        q = "querypair111111111111111111111111111111"
        search_canon = "SearchCanon111111111111111111111111111"
        exact_other = "ExactOther1111111111111111111111111111"

        def fetch_fn(url: str, timeout: float) -> dict[str, Any]:
            if "/search?" in url:
                payload = {
                    "pairs": [
                        _pair(
                            chain="solana",
                            pair_address=search_canon,
                            url=f"https://dexscreener.com/solana/{q}",
                        )
                    ]
                }
            else:
                payload = {"pairs": [_pair(chain="solana", pair_address=exact_other)]}
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
            ready = tmp_path / "ready.csv"
            _write_csv(ready, [], READY_FIELDS)
            _write_csv(
                inp,
                [
                    {
                        "combined_target_id": "mismatch1",
                        "chain": "solana",
                        "refetch_pair_id": q,
                        "acceptance_status": "CASE_SENSITIVE_PAIR_ID_UNRESOLVED",
                        "target_source": "USER_DEXSCREENER_SEED",
                        "seed_collection": "USER_SEED_REFI",
                        "provider_pair_url_fixed": f"https://dexscreener.com/solana/{q}",
                        "user_supplied_pair_address_fixed": q,
                        "pair_address_fixed": "",
                        "resolved_pair_address_fixed": "",
                        "refetch_pair_id_fixed": q,
                        "seed_collection_fixed": "USER_SEED_REFI",
                        "target_source_fixed": "USER_DEXSCREENER_SEED",
                        "linked_sources_fixed": "USER_DEXSCREENER_SEED",
                        "semantic_status_fixed": "PENDING_SYSTEM_CLASSIFICATION",
                        "recovery_input_ready": "true",
                    }
                ],
                CANON_FIELDS,
            )
            out = self.mod.run(
                inp,
                ready_path=ready,
                output_root=tmp_path / "out",
                fetch_fn=fetch_fn,
                sleeper=lambda _s: None,
                sleep_seconds=0.0,
            )
            self.assertEqual(
                out["still_rejected"][0]["recovery_status"],
                "STILL_REJECTED_IDENTITY_CONTRADICTION",
            )

    def test_search_no_result_remains_rejected(self) -> None:
        def fetch_fn(url: str, timeout: float) -> dict[str, Any]:
            return {
                "ok": True,
                "status_code": 200,
                "raw_text": '{"pairs":[]}',
                "json": {"pairs": []},
                "json_parse_error": "",
                "exception_type": "",
                "exception_message": "",
                "headers": {},
            }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inp = tmp_path / "in.csv"
            ready = tmp_path / "ready.csv"
            _write_csv(ready, [], READY_FIELDS)
            _write_csv(
                inp,
                [
                    {
                        "combined_target_id": "none1",
                        "chain": "solana",
                        "refetch_pair_id": "missing111111111111111111111111111111",
                        "acceptance_status": "PROVIDER_PAIR_NOT_FOUND",
                        "target_source": "USER_DEXSCREENER_SEED",
                        "seed_collection": "USER_SEED_OPPORTUNISTIC",
                        "provider_pair_url_fixed": "https://dexscreener.com/solana/missing111111111111111111111111111111",
                        "user_supplied_pair_address_fixed": "missing111111111111111111111111111111",
                        "pair_address_fixed": "",
                        "resolved_pair_address_fixed": "",
                        "refetch_pair_id_fixed": "missing111111111111111111111111111111",
                        "seed_collection_fixed": "USER_SEED_OPPORTUNISTIC",
                        "target_source_fixed": "USER_DEXSCREENER_SEED",
                        "linked_sources_fixed": "USER_DEXSCREENER_SEED",
                        "semantic_status_fixed": "PENDING_SYSTEM_CLASSIFICATION",
                        "recovery_input_ready": "true",
                    }
                ],
                CANON_FIELDS,
            )
            out = self.mod.run(
                inp,
                ready_path=ready,
                output_root=tmp_path / "out",
                fetch_fn=fetch_fn,
                sleeper=lambda _s: None,
                sleep_seconds=0.0,
            )
            self.assertEqual(out["still_rejected"][0]["recovery_status"], "STILL_REJECTED_SEARCH_NO_RESULT")
            jsonl = (tmp_path / "out" / "data" / "ae16c_recovery_provider_responses.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertTrue(jsonl.strip())

    def test_max_http_calls_guardrail(self) -> None:
        calls = {"n": 0}

        def fetch_fn(url: str, timeout: float) -> dict[str, Any]:
            calls["n"] += 1
            return {
                "ok": True,
                "status_code": 200,
                "raw_text": '{"pairs":[]}',
                "json": {"pairs": []},
                "json_parse_error": "",
                "exception_type": "",
                "exception_message": "",
                "headers": {},
            }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inp = tmp_path / "in.csv"
            ready = tmp_path / "ready.csv"
            _write_csv(ready, [], READY_FIELDS)
            rows = []
            for i in range(5):
                q = f"q{i}111111111111111111111111111111111"
                rows.append(
                    {
                        "combined_target_id": f"t{i}",
                        "chain": "solana",
                        "refetch_pair_id": q,
                        "acceptance_status": "CASE_SENSITIVE_PAIR_ID_UNRESOLVED",
                        "target_source": "USER_DEXSCREENER_SEED",
                        "seed_collection": "USER_SEED_REFI",
                        "provider_pair_url_fixed": f"https://dexscreener.com/solana/{q}",
                        "user_supplied_pair_address_fixed": q,
                        "pair_address_fixed": "",
                        "resolved_pair_address_fixed": "",
                        "refetch_pair_id_fixed": q,
                        "seed_collection_fixed": "USER_SEED_REFI",
                        "target_source_fixed": "USER_DEXSCREENER_SEED",
                        "linked_sources_fixed": "USER_DEXSCREENER_SEED",
                        "semantic_status_fixed": "PENDING_SYSTEM_CLASSIFICATION",
                        "recovery_input_ready": "true",
                    }
                )
            _write_csv(inp, rows, CANON_FIELDS)
            out = self.mod.run(
                inp,
                ready_path=ready,
                output_root=tmp_path / "out",
                fetch_fn=fetch_fn,
                sleeper=lambda _s: None,
                sleep_seconds=0.0,
                max_http_calls=2,
            )
            self.assertLessEqual(out["manifest"]["recovery_http_calls_attempted"], 2)
            self.assertTrue(
                any(
                    r["recovery_status"] == "STILL_REJECTED_MAX_HTTP_CALLS_REACHED"
                    for r in out["still_rejected"]
                )
            )

    def test_xrpl_unsupported_or_empty_remains_explicit(self) -> None:
        def fetch_fn(url: str, timeout: float) -> dict[str, Any]:
            return {
                "ok": True,
                "status_code": 200,
                "raw_text": '{"pairs":[]}',
                "json": {"pairs": []},
                "json_parse_error": "",
                "exception_type": "",
                "exception_message": "",
                "headers": {},
            }

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            inp = tmp_path / "in.csv"
            ready = tmp_path / "ready.csv"
            _write_csv(ready, [], READY_FIELDS)
            xrpl = "43554c5400000000000000000000000000000000.rcultakrkbqjk1tpmg5hkw4dpcf9s9kcs_xrp"
            _write_csv(
                inp,
                [
                    {
                        "combined_target_id": "xrpl1",
                        "chain": "xrpl",
                        "refetch_pair_id": xrpl,
                        "acceptance_status": "CASE_SENSITIVE_PAIR_ID_UNRESOLVED",
                        "target_source": "USER_DEXSCREENER_SEED",
                        "seed_collection": "USER_SEED_COMMUNITY_DAO",
                        "provider_pair_url_fixed": f"https://dexscreener.com/xrpl/{xrpl}",
                        "user_supplied_pair_address_fixed": xrpl,
                        "pair_address_fixed": "",
                        "resolved_pair_address_fixed": "",
                        "refetch_pair_id_fixed": xrpl,
                        "seed_collection_fixed": "USER_SEED_COMMUNITY_DAO",
                        "target_source_fixed": "USER_DEXSCREENER_SEED",
                        "linked_sources_fixed": "USER_DEXSCREENER_SEED",
                        "semantic_status_fixed": "PENDING_SYSTEM_CLASSIFICATION",
                        "recovery_input_ready": "true",
                    }
                ],
                CANON_FIELDS,
            )
            out = self.mod.run(
                inp,
                ready_path=ready,
                output_root=tmp_path / "out",
                fetch_fn=fetch_fn,
                sleeper=lambda _s: None,
                sleep_seconds=0.0,
            )
            self.assertEqual(len(out["recovered"]), 0)
            self.assertEqual(out["still_rejected"][0]["chain"], "xrpl")
            self.assertIn("STILL_REJECTED", out["still_rejected"][0]["recovery_status"])

    def test_no_collector_db_mutation(self) -> None:
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
        self.assertNotIn('open("trader.db"', source)
        self.assertNotIn("clean_forward_collector", source)


if __name__ == "__main__":
    unittest.main()
