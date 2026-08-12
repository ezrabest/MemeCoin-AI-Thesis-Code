"""Focused mocked tests for AE16D curated Clean Forward collector overlay."""
from __future__ import annotations

import ast
import csv
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from typing import Any
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.clean_forward import curated_overlay as overlay  # noqa: E402


def _load_script():
    path = ROOT / "scripts" / "run_ae16d_curated_clean_forward_overlay.py"
    spec = importlib.util.spec_from_file_location("run_ae16d_overlay", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


READY_FIELDS = [
    "combined_target_id",
    "chain",
    "target_source",
    "linked_sources",
    "seed_collection",
    "semantic_status",
    "provider_pair_url",
    "provider_pair_address",
    "provider_chain_id",
    "provider_url",
    "provider_base_token_address",
    "provider_quote_token_address",
    "provider_base_token_symbol",
    "provider_quote_token_symbol",
    "clean_forward_candidate_ready",
    "acceptance_status",
    "recovery_status",
]


def _ready_row(**over: str) -> dict[str, str]:
    base = {
        "combined_target_id": "t1",
        "chain": "base",
        "target_source": "USER_DEXSCREENER_SEED",
        "linked_sources": "USER_DEXSCREENER_SEED",
        "seed_collection": "USER_SEED_REFI",
        "semantic_status": "PENDING_SYSTEM_CLASSIFICATION",
        "provider_pair_url": "https://dexscreener.com/base/0xabc",
        "provider_pair_address": "0xAbC0000000000000000000000000000000000001",
        "provider_chain_id": "base",
        "provider_url": "https://dexscreener.com/base/0xabc",
        "provider_base_token_address": "0xbase",
        "provider_quote_token_address": "0xquote",
        "provider_base_token_symbol": "TOK",
        "provider_quote_token_symbol": "USDC",
        "clean_forward_candidate_ready": "true",
        "acceptance_status": "PROVIDER_PAIR_RESOLVED",
        "recovery_status": "ORIGINAL_AE16C_READY",
    }
    base.update(over)
    return base


class TestAE16DCuratedOverlay(unittest.TestCase):
    def test_feature_flag_default_false(self) -> None:
        self.assertFalse(overlay.curated_targets_enabled({}))
        self.assertFalse(overlay.curated_targets_enabled({overlay.FLAG_USE_CURATED: "false"}))
        self.assertTrue(overlay.curated_targets_enabled({overlay.FLAG_USE_CURATED: "true"}))

    def test_load_fixture_46_and_exclude_mew(self) -> None:
        path = overlay.DEFAULT_CURATED_READY_PATH
        if not path.exists():
            self.skipTest("curated ready file not present in workspace")
        loaded = overlay.load_curated_ready_targets(path, explicit_validation=True)
        self.assertEqual(len(loaded["loaded_rows"]), 46)
        self.assertEqual(len(loaded["accepted_rows"]), 46)
        for row in loaded["accepted_rows"]:
            self.assertNotIn(
                "mew1gqwj3nexg2qgeriku7fafj79phvqvrequzscpp5",
                (row.get("provider_pair_address") or "").lower(),
            )
            self.assertEqual(row["semantic_status"], "PENDING_SYSTEM_CLASSIFICATION")
            self.assertEqual(row.get("system_semantic_label"), "")

    def test_still_rejected_not_included(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ready.csv"
            rows = [
                _ready_row(combined_target_id="ok1"),
                _ready_row(
                    combined_target_id="bad1",
                    recovery_status="STILL_REJECTED_NO_STRONG_MATCH",
                    clean_forward_candidate_ready="true",
                    provider_pair_address="0xbad",
                ),
                _ready_row(
                    combined_target_id="mew",
                    chain="solana",
                    provider_chain_id="solana",
                    provider_pair_address="mew1gqwj3nexg2qgeriku7fafj79phvqvrequzscpp5",
                    provider_url="https://dexscreener.com/solana/mew1gqwj3nexg2qgeriku7fafj79phvqvrequzscpp5",
                    provider_pair_url="https://dexscreener.com/solana/mew1gqwj3nexg2qgeriku7fafj79phvqvrequzscpp5",
                    provider_base_token_address="Base",
                    provider_quote_token_address="Quote",
                ),
            ]
            _write_csv(p, rows, READY_FIELDS)
            loaded = overlay.load_curated_ready_targets(p, explicit_validation=True)
            ids = {r["combined_target_id"] for r in loaded["accepted_rows"]}
            self.assertEqual(ids, {"ok1"})
            reasons = {e["exclusion_reason"] for e in loaded["excluded_rows"]}
            self.assertIn("still_rejected_or_unresolved_status", reasons)
            self.assertIn("explicitly_excluded_pair_id", reasons)

    def test_flag_false_curated_loader_not_invoked_via_try(self) -> None:
        with mock.patch.object(overlay, "load_curated_ready_targets") as load_mock:
            result = overlay.try_curated_overlay_or_none(
                environ={overlay.FLAG_USE_CURATED: "false"}
            )
            self.assertIsNone(result)
            load_mock.assert_not_called()

    def test_flag_true_loads_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ready.csv"
            _write_csv(p, [_ready_row()], READY_FIELDS)
            env = {
                overlay.FLAG_USE_CURATED: "true",
                overlay.FLAG_CURATED_PATH: str(p),
            }
            loaded = overlay.load_curated_ready_targets(
                overlay.curated_targets_path(env),
                environ=env,
                explicit_validation=True,
            )
            self.assertEqual(len(loaded["accepted_rows"]), 1)

    def test_custom_path_exists_and_missing_graceful(self) -> None:
        missing = Path("data/audits/__definitely_missing_ae16d__.csv")
        st = overlay.validate_curated_path(missing, explicit_validation=False)
        self.assertFalse(st["path_exists"])
        self.assertFalse(st["ok_for_load"])
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            runtime = overlay.load_curated_ready_targets(
                missing, explicit_validation=False
            )
        self.assertEqual(runtime["accepted_rows"], [])
        self.assertEqual(runtime.get("blocked_classification"), "")

        explicit = overlay.load_curated_ready_targets(missing, explicit_validation=True)
        self.assertEqual(explicit["blocked_classification"], "AE16D_BLOCKED_CURATED_INPUT_MISSING")

    def test_no_fetch_on_module_import(self) -> None:
        src = Path(overlay.__file__).read_text(encoding="utf-8")
        tree = ast.parse(src)
        # Module body should not call verify/http
        for node in tree.body:
            if isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
                text = ast.dump(node)
                self.assertNotIn("verify_provider_pair", text)
                self.assertNotIn("httpx", text)

    def test_no_fetch_on_server_startup_simulation(self) -> None:
        # Importing feed module must not call curated refetch
        import app.ae13b_product.clean_forward_market_feed as feed_mod

        with mock.patch.object(overlay, "run_curated_refetch") as refetch_mock:
            importlib.reload(feed_mod)
            refetch_mock.assert_not_called()

    def test_solana_xrpl_casing_preserved_on_load(self) -> None:
        sol = "2uF4Xh61rDwxnG9woyxsVQP7zuA6kLFpb3NvnRQeoiSd"
        xrpl = "43554c5400000000000000000000000000000000.rcultakrkbqjk1tpmg5hkw4dpcf9s9kcs_xrp"
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ready.csv"
            _write_csv(
                p,
                [
                    _ready_row(
                        combined_target_id="sol1",
                        chain="solana",
                        provider_chain_id="solana",
                        provider_pair_address=sol,
                        provider_url=f"https://dexscreener.com/solana/{sol.lower()}",
                        provider_pair_url=f"https://dexscreener.com/solana/{sol.lower()}",
                        provider_base_token_address="BaseMint",
                        provider_quote_token_address="QuoteMint",
                    ),
                    _ready_row(
                        combined_target_id="xrpl1",
                        chain="xrpl",
                        provider_chain_id="xrpl",
                        provider_pair_address=xrpl,
                        provider_url=f"https://dexscreener.com/xrpl/{xrpl}",
                        provider_pair_url=f"https://dexscreener.com/xrpl/{xrpl}",
                        provider_base_token_address="XBase",
                        provider_quote_token_address="XQuote",
                    ),
                ],
                READY_FIELDS,
            )
            loaded = overlay.load_curated_ready_targets(p, explicit_validation=True)
            by_id = {r["combined_target_id"]: r for r in loaded["accepted_rows"]}
            self.assertEqual(by_id["sol1"]["provider_pair_address"], sol)
            self.assertEqual(by_id["xrpl1"]["provider_pair_address"], xrpl)

    def test_evm_case_insensitive_identity_preserves_output(self) -> None:
        curated = {
            "chain": "base",
            "provider_pair_address": "0xAbC0000000000000000000000000000000000001",
        }
        verified = {
            "pair_address": "0xabc0000000000000000000000000000000000001",
            "normalized_chain_id": "base",
            "base_token_address": "0xb",
            "quote_token_address": "0xq",
        }
        ok, reason = overlay._identity_ok(curated, verified)
        self.assertTrue(ok)
        self.assertEqual(reason, "ok")

    def test_semantic_separation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ready.csv"
            _write_csv(p, [_ready_row(seed_collection="USER_SEED_SOCIALFI")], READY_FIELDS)
            loaded = overlay.load_curated_ready_targets(p, explicit_validation=True)
            row = loaded["accepted_rows"][0]
            self.assertEqual(row["seed_collection"], "USER_SEED_SOCIALFI")
            self.assertEqual(row["semantic_status"], "PENDING_SYSTEM_CLASSIFICATION")
            self.assertEqual(row["system_semantic_label"], "")

    def test_exact_pair_refetch_mocked_and_rejected_not_clean_row(self) -> None:
        curated = [_ready_row(combined_target_id="pass1"), _ready_row(combined_target_id="fail1", provider_pair_address="0xfail")]

        def verify_fn(*, chain_id, pair_address, expected_url=None, use_cache=True):
            if "fail" in pair_address.lower():
                return {
                    "clean_feed_eligible": False,
                    "lookup_ok": False,
                    "verification_status": "provider_pair_not_found",
                    "exclusion_reason": "not_found",
                    "pair_address": pair_address,
                    "normalized_chain_id": chain_id,
                    "base_token_address": "",
                    "quote_token_address": "",
                    "provider_pair_url": "",
                }
            return {
                "clean_feed_eligible": True,
                "lookup_ok": True,
                "verification_status": "provider_pair_verified",
                "pair_address": pair_address,
                "normalized_chain_id": chain_id,
                "chain_id": chain_id,
                "base_token_address": "0xbase",
                "quote_token_address": "0xquote",
                "base_token_symbol": "TOK",
                "quote_token_symbol": "USDC",
                "provider_pair_url": f"https://dexscreener.com/{chain_id}/{pair_address}",
                "provider_pair_url_source": "dexscreener",
                "price_usd": 1.0,
                "liquidity_usd": 100.0,
                "freshness_status": "fresh",
                "identity_status": "pair_and_tokens_separated",
                "tradability_status": "paper_demo_only",
                "dex_id": "uniswap",
                "pair_label": "TOK/USDC",
                "fetched_at": "2026-01-01T00:00:00+00:00",
            }

        def row_builder(v: dict[str, Any]) -> dict[str, Any]:
            return {
                "row_id": f"{v['normalized_chain_id']}|pair|{v['pair_address']}",
                "chain": v["normalized_chain_id"],
                "pair_address": v["pair_address"],
                "provider_pair_url": v["provider_pair_url"],
                "base_token_address": v["base_token_address"],
                "quote_token_address": v["quote_token_address"],
                "clean_feed_eligible": True,
                "paper_demo_only": True,
                "live_trading_ready": False,
                "verification_status": v["verification_status"],
                "freshness_status": v["freshness_status"],
                "identity_status": v["identity_status"],
                "price_usd": v["price_usd"],
                "liquidity_usd": v["liquidity_usd"],
            }

        result = overlay.run_curated_refetch(
            curated,
            dry_run=False,
            sleep_seconds=0.0,
            sleeper=lambda _s: None,
            verify_fn=verify_fn,
            row_builder=row_builder,
        )
        self.assertEqual(len(result["clean_forward_rows"]), 1)
        self.assertEqual(result["clean_forward_rows"][0]["combined_target_id"], "pass1")
        self.assertEqual(len(result["rejected_rows"]), 1)
        self.assertEqual(result["rejected_rows"][0]["combined_target_id"], "fail1")
        self.assertEqual(result["clean_forward_rows"][0]["semantic_status"], "PENDING_SYSTEM_CLASSIFICATION")
        self.assertEqual(result["clean_forward_rows"][0].get("system_semantic_label"), "")

    def test_no_broad_search_trending_in_curated_overlay_source(self) -> None:
        src = Path(overlay.__file__).read_text(encoding="utf-8")
        self.assertNotIn("get_trending_pairs", src)
        self.assertNotIn("/latest/dex/search", src)
        self.assertIn("exact", src.lower())

    def test_dry_run_no_network(self) -> None:
        called = {"n": 0}

        def verify_fn(**kwargs):
            called["n"] += 1
            raise AssertionError("should not verify in dry-run")

        result = overlay.run_curated_refetch(
            [_ready_row()],
            dry_run=True,
            verify_fn=verify_fn,
            sleeper=lambda _s: None,
        )
        self.assertEqual(called["n"], 0)
        self.assertEqual(result["refetch_results"][0]["verification_status"], "DRY_RUN_NOT_FETCHED")

    def test_script_flag_off_and_missing_path_gate(self) -> None:
        mod = _load_script()
        proof = mod.prove_flag_off_unchanged()
        self.assertTrue(proof["existing_collector_behavior_unchanged_when_flag_off"])
        runtime = mod.prove_runtime_blocking_safety()
        self.assertTrue(runtime["runtime_blocking_safety_passed"])

        with tempfile.TemporaryDirectory() as tmp:
            out = mod.run(
                input_path=Path(tmp) / "missing.csv",
                output_root=Path(tmp) / "out",
                dry_run=True,
                explicit_validation=True,
                sleeper=lambda _s: None,
            )
            self.assertEqual(out["gate"]["classification"], "AE16D_BLOCKED_CURATED_INPUT_MISSING")

    def test_no_trader_db_or_wallet_in_overlay_sources(self) -> None:
        for rel in (
            "app/clean_forward/curated_overlay.py",
            "scripts/run_ae16d_curated_clean_forward_overlay.py",
        ):
            src = (ROOT / rel).read_text(encoding="utf-8")
            tree = ast.parse(src)
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        imported.add(a.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertNotIn("sqlite3", imported)
            self.assertNotIn('open("trader.db"', src)
            self.assertNotIn("web3", imported)


if __name__ == "__main__":
    unittest.main()
