"""Focused tests for AE16B combined clean universe builder (offline / read-only)."""
from __future__ import annotations

import ast
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

SCRIPT_PATH = ROOT / "scripts" / "build_ae16b_combined_clean_universe.py"


def _load_mod():
    spec = importlib.util.spec_from_file_location("build_ae16b_combined_clean_universe", SCRIPT_PATH)
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


class TestAE16BCombinedCleanUniverse(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.mod = _load_mod()

    def test_input_path_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            missing_seed = tmp_path / "missing_seed.csv"
            missing_clean = tmp_path / "missing_clean.csv"
            seed_ok, clean_ok, errors = self.mod.validate_inputs(missing_seed, missing_clean)
            self.assertFalse(seed_ok)
            self.assertFalse(clean_ok)
            self.assertEqual(len(errors), 2)
            self.assertTrue(any("User seed CSV not found" in e for e in errors))
            self.assertTrue(any("Clean Forward candidates CSV not found" in e for e in errors))

            with self.assertRaises(FileNotFoundError) as ctx:
                self.mod.run(missing_seed, missing_clean, tmp_path / "out")
            self.assertIn("not found", str(ctx.exception).lower())

    def test_loading_user_seed_and_clean_candidate_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            seed = tmp_path / "seed.csv"
            clean = tmp_path / "clean.csv"
            _write_csv(
                seed,
                [
                    {
                        "target_id": "seed_a",
                        "active": "true",
                        "target_source": "USER_DEXSCREENER_SEED",
                        "chain": "base",
                        "user_supplied_pair_address": "0xAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAaAa",
                        "user_supplied_token_address": "",
                        "provider_pair_url": "https://dexscreener.com/base/0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                        "seed_collection": "USER_SEED_REFI",
                        "semantic_status": "PENDING_SYSTEM_CLASSIFICATION",
                        "notes": "provenance only",
                    }
                ],
                [
                    "target_id",
                    "active",
                    "target_source",
                    "chain",
                    "user_supplied_pair_address",
                    "user_supplied_token_address",
                    "provider_pair_url",
                    "seed_collection",
                    "semantic_status",
                    "notes",
                ],
            )
            _write_csv(
                clean,
                [
                    {
                        "clean_forward_candidate_id": "cf1",
                        "source_clean_forward_row_key": "base|pair|0xBbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBb",
                        "provider_payload_hash": "hash1",
                        "provider_pair_url": "https://dexscreener.com/base/0xbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                        "chain": "base",
                        "pair_address": "0xBbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBbBb",
                        "base_token_address": "0xBaseToken",
                        "quote_token_address": "0xQuoteToken",
                        "symbol_pair": "AAA/BBB",
                    }
                ],
                [
                    "clean_forward_candidate_id",
                    "source_clean_forward_row_key",
                    "provider_payload_hash",
                    "provider_pair_url",
                    "chain",
                    "pair_address",
                    "base_token_address",
                    "quote_token_address",
                    "symbol_pair",
                ],
            )
            out = self.mod.run(seed, clean, tmp_path / "out")
            manifest = out["manifest"]
            self.assertTrue(manifest["input_user_seed_exists"])
            self.assertTrue(manifest["input_clean_candidates_exists"])
            self.assertEqual(manifest["clean_targets_loaded"], 1)
            self.assertEqual(manifest["user_seed_targets_loaded"], 1)
            self.assertEqual(manifest["combined_unique_targets"], 2)
            self.assertTrue((tmp_path / "out" / "data" / "ae16b_user_seed_targets.csv").exists())
            self.assertTrue((tmp_path / "out" / "data" / "ae16b_existing_clean_targets.csv").exists())

    def test_seed_collection_provenance_and_semantic_pending(self) -> None:
        clean_rows = [
            {
                "clean_forward_candidate_id": "cf1",
                "source_clean_forward_row_key": "solana|pair|AbC123",
                "provider_payload_hash": "h1",
                "provider_pair_url": "https://dexscreener.com/solana/abc123",
                "chain": "solana",
                "pair_address": "AbC123",
                "base_token_address": "Base1",
                "quote_token_address": "Quote1",
                "symbol_pair": "TOK/USDC",
            }
        ]
        seed_rows = [
            {
                "target_id": "s1",
                "active": "true",
                "chain": "solana",
                "user_supplied_pair_address": "XyZ999",
                "provider_pair_url": "https://dexscreener.com/solana/xyz999",
                "seed_collection": "USER_SEED_SOCIALFI",
                "system_semantic_label": "SHOULD_NOT_BE_COPIED",
            }
        ]
        result = self.mod.build_universe(clean_rows, seed_rows)
        for row in result["combined"]:
            self.assertEqual(row["semantic_status"], "PENDING_SYSTEM_CLASSIFICATION")
            self.assertNotIn("system_semantic_label", row)
        seed_out = result["user_seed_targets"][0]
        self.assertEqual(seed_out["seed_collection"], "USER_SEED_SOCIALFI")
        self.assertEqual(seed_out["semantic_status"], "PENDING_SYSTEM_CLASSIFICATION")
        clean_out = result["existing_targets"][0]
        self.assertEqual(clean_out["seed_collection"], "EXISTING_CLEAN_FORWARD")

    def test_dedupe_by_provider_pair_url(self) -> None:
        clean_rows = [
            {
                "clean_forward_candidate_id": "cf1",
                "source_clean_forward_row_key": "base|pair|0x1",
                "provider_payload_hash": "h1",
                "provider_pair_url": "https://dexscreener.com/base/0xabc",
                "chain": "base",
                "pair_address": "0xABC",
                "base_token_address": "0xb1",
                "quote_token_address": "0xq1",
                "symbol_pair": "A/B",
            }
        ]
        seed_rows = [
            {
                "target_id": "s1",
                "active": "true",
                "chain": "base",
                "user_supplied_pair_address": "0xabc",
                "provider_pair_url": "https://dexscreener.com/base/0xABC",
                "seed_collection": "USER_SEED_REFI",
            }
        ]
        result = self.mod.build_universe(clean_rows, seed_rows)
        self.assertEqual(result["combined_unique_targets"], 1)
        merged = result["combined"][0]
        self.assertEqual(merged["target_source"], "MERGED")
        self.assertEqual(merged["linked_sources"], "CLEAN_FORWARD_EXISTING;USER_DEXSCREENER_SEED")
        self.assertEqual(merged["seed_collection"], "USER_SEED_REFI")
        self.assertEqual(merged["source_clean_forward_candidate_id"], "cf1")
        self.assertTrue(any(d["duplicate_reason"] == "provider_pair_url" for d in result["duplicates"]))

    def test_duplicate_clean_plus_seed_merges_into_one_row(self) -> None:
        clean_rows = [
            {
                "clean_forward_candidate_id": "cf_keep",
                "source_clean_forward_row_key": "ethereum|pair|0xDeadBeef",
                "provider_payload_hash": "payload",
                "provider_pair_url": "https://dexscreener.com/ethereum/0xdeadbeef",
                "chain": "ethereum",
                "pair_address": "0xDeadBeefDeadBeefDeadBeefDeadBeefDeadBeef",
                "base_token_address": "0xBase",
                "quote_token_address": "0xQuote",
                "symbol_pair": "MERGE/ETH",
            }
        ]
        seed_rows = [
            {
                "target_id": "seed_same",
                "active": "true",
                "chain": "ethereum",
                "user_supplied_pair_address": "0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
                "provider_pair_url": "",
                "seed_collection": "USER_SEED_COMMUNITY_DAO",
            }
        ]
        result = self.mod.build_universe(clean_rows, seed_rows)
        self.assertEqual(len(result["combined"]), 1)
        row = result["combined"][0]
        self.assertEqual(row["target_source"], "MERGED")
        self.assertEqual(row["linked_sources"], "CLEAN_FORWARD_EXISTING;USER_DEXSCREENER_SEED")
        self.assertEqual(row["pair_address"], "0xDeadBeefDeadBeefDeadBeefDeadBeefDeadBeef")
        self.assertEqual(row["seed_collection"], "USER_SEED_COMMUNITY_DAO")
        self.assertEqual(row["semantic_status"], "PENDING_SYSTEM_CLASSIFICATION")

    def test_no_dedupe_by_symbol(self) -> None:
        clean_rows = [
            {
                "clean_forward_candidate_id": "cf1",
                "source_clean_forward_row_key": "base|pair|0x1",
                "provider_payload_hash": "h1",
                "provider_pair_url": "https://dexscreener.com/base/0x1111",
                "chain": "base",
                "pair_address": "0x1111",
                "base_token_address": "0xb1",
                "quote_token_address": "0xq1",
                "symbol_pair": "SAME/USDC",
            },
            {
                "clean_forward_candidate_id": "cf2",
                "source_clean_forward_row_key": "base|pair|0x2",
                "provider_payload_hash": "h2",
                "provider_pair_url": "https://dexscreener.com/base/0x2222",
                "chain": "base",
                "pair_address": "0x2222",
                "base_token_address": "0xb2",
                "quote_token_address": "0xq2",
                "symbol_pair": "SAME/USDC",
            },
        ]
        result = self.mod.build_universe(clean_rows, [])
        self.assertEqual(result["combined_unique_targets"], 2)

    def test_solana_xrpl_casing_preserved(self) -> None:
        sol_pair = "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd"
        xrpl_pair = "43554c5400000000000000000000000000000000.rcultakrkbqjk1tpmg5hkw4dpcf9s9kcs_xrp"
        clean_rows = [
            {
                "clean_forward_candidate_id": "cf_sol",
                "source_clean_forward_row_key": f"solana|pair|{sol_pair}",
                "provider_payload_hash": "hs",
                "provider_pair_url": "https://dexscreener.com/solana/2uf4xh61rdwxng9woyxsvqp7zua6klfpb3nvnrqeoisd",
                "chain": "solana",
                "pair_address": sol_pair,
                "base_token_address": "pumpCmXqMfrsAkQ5r49WcJnRayYRqmXz6ae8H7H9Dfn",
                "quote_token_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                "symbol_pair": "PUMP/USDC",
            }
        ]
        seed_rows = [
            {
                "target_id": "xrpl_1",
                "active": "true",
                "chain": "xrpl",
                "user_supplied_pair_address": xrpl_pair,
                "provider_pair_url": f"https://dexscreener.com/xrpl/{xrpl_pair}",
                "seed_collection": "USER_SEED_COMMUNITY_DAO",
            }
        ]
        result = self.mod.build_universe(clean_rows, seed_rows)
        sol = next(r for r in result["combined"] if r["chain"] == "solana")
        xrpl = next(r for r in result["combined"] if r["chain"] == "xrpl")
        self.assertEqual(sol["pair_address"], sol_pair)
        self.assertEqual(xrpl["user_supplied_pair_address"], xrpl_pair)

        # Different Solana casing must NOT match as the same address
        other = {
            "target_id": "sol_diff_case",
            "active": "true",
            "chain": "solana",
            "user_supplied_pair_address": sol_pair.lower(),  # different casing
            "provider_pair_url": "https://dexscreener.com/solana/DIFFERENT_URL_PATH",
            "seed_collection": "USER_SEED_OPPORTUNISTIC",
        }
        result2 = self.mod.build_universe(clean_rows, [other])
        # URL differs and Solana address casing differs => two rows
        self.assertEqual(result2["combined_unique_targets"], 2)

    def test_missing_pair_url_row_goes_to_rejected(self) -> None:
        seed_rows = [
            {
                "target_id": "bad",
                "active": "true",
                "chain": "base",
                "user_supplied_pair_address": "",
                "provider_pair_url": "",
                "seed_collection": "USER_SEED_REFI",
            }
        ]
        clean_rows = [
            {
                "clean_forward_candidate_id": "also_bad",
                "source_clean_forward_row_key": "",
                "provider_payload_hash": "",
                "provider_pair_url": "",
                "chain": "",
                "pair_address": "",
                "base_token_address": "",
                "quote_token_address": "",
                "symbol_pair": "NOPE/USDC",
            }
        ]
        result = self.mod.build_universe(clean_rows, seed_rows)
        self.assertEqual(result["combined_unique_targets"], 0)
        self.assertEqual(result["rejected_or_incomplete_count"], 2)
        reasons = {r["rejection_reason"] for r in result["rejected"]}
        self.assertTrue(any("missing" in r for r in reasons))

    def test_no_dexscreener_calls_or_network_imports(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        forbidden = {
            "requests",
            "httpx",
            "urllib",
            "aiohttp",
            "websocket",
            "app",
            "DexScreener",
            "dexscreener",
        }
        self.assertTrue(imported.isdisjoint(forbidden))
        lowered = source.lower()
        self.assertNotIn("from app.", source)
        self.assertNotIn("import requests", lowered)
        self.assertNotIn("urllib.request", lowered)
        self.assertNotIn("httpx", lowered)

    def test_no_server_requirement_and_no_db_mutation(self) -> None:
        source = SCRIPT_PATH.read_text(encoding="utf-8")
        tree = ast.parse(source)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        self.assertNotIn("uvicorn", imported)
        self.assertNotIn("fastapi", imported)
        self.assertNotIn("sqlite3", imported)
        self.assertNotIn("sqlalchemy", imported)
        self.assertFalse(any(name.startswith("app") for name in imported))
        # No executable writes to trader.db (confirmation text in reports is allowed).
        self.assertNotIn('open("trader.db"', source)
        self.assertNotIn("Path(\"trader.db\")", source)
        self.assertNotIn("connect('trader.db')", source)
        self.assertNotIn("clean_forward_collector", source)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            seed = tmp_path / "seed.csv"
            clean = tmp_path / "clean.csv"
            _write_csv(
                seed,
                [
                    {
                        "target_id": "s1",
                        "active": "true",
                        "chain": "base",
                        "user_supplied_pair_address": "0x1",
                        "provider_pair_url": "https://dexscreener.com/base/0x1",
                        "seed_collection": "USER_SEED_REFI",
                    }
                ],
                [
                    "target_id",
                    "active",
                    "chain",
                    "user_supplied_pair_address",
                    "provider_pair_url",
                    "seed_collection",
                ],
            )
            _write_csv(
                clean,
                [
                    {
                        "clean_forward_candidate_id": "c1",
                        "source_clean_forward_row_key": "base|pair|0x2",
                        "provider_payload_hash": "h",
                        "provider_pair_url": "https://dexscreener.com/base/0x2",
                        "chain": "base",
                        "pair_address": "0x2",
                        "base_token_address": "0xb",
                        "quote_token_address": "0xq",
                        "symbol_pair": "X/Y",
                    }
                ],
                [
                    "clean_forward_candidate_id",
                    "source_clean_forward_row_key",
                    "provider_payload_hash",
                    "provider_pair_url",
                    "chain",
                    "pair_address",
                    "base_token_address",
                    "quote_token_address",
                    "symbol_pair",
                ],
            )
            out = self.mod.run(seed, clean, tmp_path / "out")
            manifest = out["manifest"]
            self.assertTrue(manifest["no_dexscreener_called"])
            self.assertTrue(manifest["no_server_started"])
            self.assertFalse(manifest["collector_modified"])
            self.assertFalse(manifest["trader_db_mutated"])
            summary = (tmp_path / "out" / "reports" / "ae16b_summary_for_upload.txt").read_text(
                encoding="utf-8"
            )
            self.assertIn("DexScreener was not called", summary)
            self.assertIn("server was not started", summary)
            self.assertIn("collector was not modified", summary)
            self.assertIn("trader.db was not mutated", summary)
            self.assertIn("seed_collection is provenance only", summary)
            self.assertIn("PENDING_SYSTEM_CLASSIFICATION", summary)
            manifest_json = json.loads(
                (tmp_path / "out" / "reports" / "ae16b_combined_universe_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest_json["phase"], "AE16B_COMBINED_CLEAN_UNIVERSE")


if __name__ == "__main__":
    unittest.main()
