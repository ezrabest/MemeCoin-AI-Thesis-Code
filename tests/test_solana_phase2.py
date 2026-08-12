"""Phase 2 tests: Helius validation, budget, and wallet behavior audit."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from app.parsers import solana_pool_activity as pool_parser
from app.parsers.solana_wallet_behavior import (
    IDENTITY_FEE_PAYER_FALLBACK,
    IDENTITY_TOKEN_OWNER_FEE_PAYER,
    IDENTITY_TOKEN_OWNER_SIGNER,
    WARN_LOW_SAMPLE,
    WARN_MULTI_SIGNER,
    WARN_NOT_SIGNAL,
    WARN_RELAYER,
    build_behavior_events,
    resolve_trader_identity,
    summarize_wallet_behavior,
)
from app.providers.helius import (
    HELIUS_CACHE_HIT,
    HELIUS_OK,
    HELIUS_SKIPPED_BUDGET,
    HELIUS_SKIPPED_NOT_ENABLED,
    HELIUS_UNAVAILABLE,
    HELIUS_RATE_LIMITED,
    HeliusClient,
    build_helius_transactions_url,
    compare_raw_with_helius,
    validate_signatures_against_raw,
)
from app.providers.helius_budget import (
    HeliusBudgetManager,
    dynamic_daily_budget,
    estimate_credit_cost,
    load_usage,
    remaining_days_in_month,
    reset_month_if_needed,
    save_usage,
)
from scripts.probe_solana_pool_activity import run_probe


def _mock_response(status_code: int = 200, json_data=None) -> httpx.Response:
    request = httpx.Request("POST", "https://mainnet.helius-rpc.com/v0/transactions")
    content = json.dumps(json_data).encode("utf-8") if json_data is not None else b""
    return httpx.Response(status_code, request=request, content=content)


def _parsed_event(
    *,
    side="BUY_BASE",
    failed=False,
    trader="TraderWallet1111111111111111111111111111111",
    fee_payer="TraderWallet1111111111111111111111111111111",
    usdc="100",
    signature="sig1",
    owner_status=pool_parser.OWNER_MATCHED_FEE_PAYER,
) -> dict:
    return {
        "signature": signature,
        "block_time": 1700000000,
        "failed_transaction": failed,
        "parse_status": "OK" if not failed else pool_parser.PARSE_FAILED_TX,
        "side": pool_parser.SIDE_IGNORED_FAILED if failed else side,
        "trader_wallet": trader,
        "fee_payer": fee_payer,
        "signer_wallets": [fee_payer],
        "quote_token_type": pool_parser.QUOTE_USDC,
        "approx_usd_value": usdc,
        "quote_amount_native": None,
        "base_delta_pool_str": "-10",
        "quote_delta_pool_str": usdc,
        "base_token_mint": "BaseMint1111111111111111111111111111111111",
        "quote_token_mint": pool_parser.USDC_MINT,
        "program_ids": [],
        "token_balances": [
            {
                "token_account": "TokenAcct111111111111111111111111111111111",
                "owner": trader,
                "owner_match_status": owner_status,
                "mint": pool_parser.USDC_MINT,
                "ui_amount_delta_str": usdc,
            }
        ],
    }


class HeliusProviderTests(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("HELIUS_API_KEY", None)

    def test_helius_not_called_unless_enabled(self) -> None:
        client = HeliusClient(enabled=False)
        result = client.fetch_transactions(["sig1"])
        self.assertEqual(result["status"], HELIUS_SKIPPED_NOT_ENABLED)

    def test_helius_api_key_from_environment(self) -> None:
        os.environ["HELIUS_API_KEY"] = "test-key-12345"
        url = build_helius_transactions_url()
        self.assertIn("api-key=test-key", url or "")
        self.assertNotIn("$env:", url or "")

    def test_helius_url_no_literal_env_placeholder(self) -> None:
        os.environ["HELIUS_API_KEY"] = "$env:HELIUS_API_KEY"
        self.assertIsNone(build_helius_transactions_url())

    @patch("app.providers.helius.httpx.Client")
    def test_helius_uses_post_v0_transactions_only(self, mock_client_cls: MagicMock) -> None:
        os.environ["HELIUS_API_KEY"] = "secret-key"
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = _mock_response(json_data=[{"signature": "sig1"}])

        with tempfile.TemporaryDirectory() as tmp:
            client = HeliusClient(enabled=True, cache_dir=Path(tmp))
            client.fetch_transactions(["sig1"])

        called_url = mock_client.post.call_args.args[0]
        self.assertIn("/v0/transactions", called_url)
        self.assertNotIn("/v0/addresses", called_url)
        body = mock_client.post.call_args.kwargs["json"]
        self.assertEqual(body, {"transactions": ["sig1"]})

    def test_api_key_not_in_url_when_missing(self) -> None:
        os.environ.pop("HELIUS_API_KEY", None)
        self.assertIsNone(build_helius_transactions_url())


class HeliusBudgetTests(unittest.TestCase):
    def test_monthly_reset(self) -> None:
        usage = {
            "provider": "helius",
            "monthly_budget": 1000,
            "monthly_used": 500,
            "daily_used": {"2020-01-01": 100},
            "endpoint_used": {"v0/transactions": 500},
            "last_reset_month": "2020-01",
        }
        reset = reset_month_if_needed(usage, date(2020, 2, 1))
        self.assertEqual(reset["monthly_used"], 0)
        self.assertEqual(reset["daily_used"], {})
        self.assertEqual(reset["last_reset_month"], "2020-02")

    def test_dynamic_daily_budget_uses_remaining_month_days(self) -> None:
        day = date(2026, 6, 20)
        days_left = remaining_days_in_month(day)
        self.assertEqual(days_left, 11)
        budget = dynamic_daily_budget(100_000, 0, day)
        self.assertAlmostEqual(budget, 100_000 / 11, places=4)

    def test_budget_skip_when_exceeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            usage_path = Path(tmp) / "helius_usage.json"
            usage = {
                "provider": "helius",
                "monthly_budget": 10,
                "monthly_used": 10,
                "daily_used": {date.today().isoformat(): 10},
                "endpoint_used": {},
                "last_reset_month": date.today().strftime("%Y-%m"),
            }
            save_usage(usage, usage_path=usage_path)
            manager = HeliusBudgetManager(monthly_budget=10, usage_path=usage_path)
            os.environ["HELIUS_API_KEY"] = "key"
            client = HeliusClient(enabled=True, cache_dir=Path(tmp), budget_manager=manager)
            result = client.fetch_transactions(["sig1"])
            self.assertEqual(result["status"], HELIUS_SKIPPED_BUDGET)

    @patch("app.providers.helius.httpx.Client")
    def test_cache_hit_avoids_budget_increment(self, mock_client_cls: MagicMock) -> None:
        os.environ["HELIUS_API_KEY"] = "key"
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            usage_path = cache_dir / "usage.json"
            manager = HeliusBudgetManager(monthly_budget=100_000, usage_path=usage_path)
            client = HeliusClient(enabled=True, cache_dir=cache_dir, budget_manager=manager)

            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = _mock_response(json_data=[{"signature": "sig1"}])

            first = client.fetch_transactions(["sig1"])
            self.assertEqual(first["status"], HELIUS_OK)
            used_after_first = load_usage(usage_path=usage_path)["monthly_used"]

            second = client.fetch_transactions(["sig1"])
            self.assertEqual(second["status"], HELIUS_CACHE_HIT)
            used_after_second = load_usage(usage_path=usage_path)["monthly_used"]
            self.assertEqual(used_after_first, used_after_second)
            self.assertEqual(mock_client.post.call_count, 1)


class HeliusComparisonTests(unittest.TestCase):
    def test_comparison_does_not_overwrite_raw(self) -> None:
        raw = _parsed_event(side="BUY_BASE", usdc="11")
        helius = {
            "signature": "sig1",
            "feePayer": "OtherWallet111111111111111111111111111111111",
            "source": "JUPITER",
            "tokenTransfers": [{"mint": pool_parser.USDC_MINT, "tokenAmount": 99}],
        }
        comparison = compare_raw_with_helius(raw, helius)
        self.assertEqual(raw["side"], "BUY_BASE")
        self.assertEqual(raw["trader_wallet"], "TraderWallet1111111111111111111111111111111")
        self.assertTrue(comparison.get("mismatch"))

    @patch("app.providers.helius.HeliusClient.fetch_transactions")
    def test_helius_unavailable_does_not_crash_probe(self, mock_fetch: MagicMock) -> None:
        mock_fetch.return_value = {"status": HELIUS_UNAVAILABLE, "transactions": []}
        with patch("scripts.probe_solana_pool_activity.SolanaRpcClient") as mock_rpc:
            client = MagicMock()
            mock_rpc.return_value = client
            client.get_rpc_url.return_value = "https://private.example.com"
            client.stats = MagicMock(
                rpc_calls_attempted=0,
                rpc_calls_succeeded=0,
                rpc_rate_limited_count=0,
                rpc_forbidden_count=0,
                rpc_retry_count=0,
                rpc_timeout_count=0,
                rpc_null_result_count=0,
                cache_hits=0,
                cache_misses=0,
            )
            client.get_signatures_for_address.return_value = []
            with tempfile.TemporaryDirectory() as tmp:
                audit = run_probe(
                    pool_address="Pool111",
                    audit_dir=Path(tmp),
                    rpc_url="https://private.example.com",
                    validate_with_helius=True,
                )
        self.assertEqual(audit["helius_validation"]["helius_status"], HELIUS_UNAVAILABLE)


class WalletBehaviorTests(unittest.TestCase):
    def test_excludes_failed_transactions(self) -> None:
        events = build_behavior_events(
            [_parsed_event(failed=True), _parsed_event(side="BUY_BASE")]
        )
        summary = summarize_wallet_behavior(events)
        self.assertEqual(summary["failed_event_count"], 1)
        self.assertEqual(summary["successful_directional_event_count"], 1)

    def test_excludes_unknown_side_from_directional(self) -> None:
        events = build_behavior_events(
            [_parsed_event(side="UNKNOWN"), _parsed_event(side="BUY_BASE")]
        )
        summary = summarize_wallet_behavior(events)
        self.assertEqual(summary["unknown_event_count"], 1)
        self.assertEqual(summary["successful_directional_event_count"], 1)

    def test_low_sample_defaults_unknown(self) -> None:
        events = build_behavior_events([_parsed_event(side="BUY_BASE") for _ in range(3)])
        summary = summarize_wallet_behavior(events)
        self.assertEqual(summary["likely_behavior"], "unknown")
        self.assertEqual(summary["confidence"], "low")
        self.assertIn(WARN_LOW_SAMPLE, summary["warnings"])

    def test_token_owner_prioritized_over_fee_payer(self) -> None:
        parsed = _parsed_event(
            trader="RelayerWallet11111111111111111111111111111111",
            fee_payer="RelayerWallet11111111111111111111111111111111",
            owner_status=pool_parser.OWNER_MATCHED_FEE_PAYER,
        )
        parsed["token_balances"] = [
            {
                "token_account": "TokenAcct111111111111111111111111111111111",
                "owner": "TrueTrader111111111111111111111111111111111",
                "owner_match_status": pool_parser.OWNER_MATCHED_SIGNER,
                "mint": pool_parser.USDC_MINT,
                "ui_amount_delta_str": "10",
            }
        ]
        parsed["signer_wallets"] = [
            "RelayerWallet11111111111111111111111111111111",
            "TrueTrader111111111111111111111111111111111",
        ]
        identity = resolve_trader_identity(parsed)
        self.assertEqual(identity["trader_wallet"], "TrueTrader111111111111111111111111111111111")
        self.assertEqual(identity["trader_identity_source"], IDENTITY_TOKEN_OWNER_SIGNER)

    def test_fee_payer_fallback_only_when_no_token_owner(self) -> None:
        parsed = _parsed_event()
        parsed["token_balances"] = []
        parsed["signer_wallets"] = []
        identity = resolve_trader_identity(parsed)
        self.assertEqual(identity["trader_identity_source"], IDENTITY_FEE_PAYER_FALLBACK)

    def test_multiple_signers_warning(self) -> None:
        parsed = _parsed_event()
        parsed["signer_wallets"] = ["SignerA1111111111111111111111111111111111", "SignerB1111111111111111111111111111111111"]
        parsed["program_ids"] = ["routeUGWgWzqBWFcrCfv8tritsqukccJPu3q5GPP3xS"]
        events = build_behavior_events([parsed])
        summary = summarize_wallet_behavior(events)
        self.assertIn(WARN_MULTI_SIGNER, summary["warnings"])

    def test_relayer_does_not_collapse_unrelated_owners(self) -> None:
        events = []
        for idx, owner in enumerate(
            [
                "OwnerA111111111111111111111111111111111111",
                "OwnerB111111111111111111111111111111111111",
            ]
        ):
            parsed = _parsed_event(
                signature=f"sig{idx}",
                trader="RelayerWallet11111111111111111111111111111111",
                fee_payer="RelayerWallet11111111111111111111111111111111",
            )
            parsed["program_ids"] = ["routeUGWgWzqBWFcrCfv8tritsqukccJPu3q5GPP3xS"]
            parsed["token_balances"] = [
                {
                    "token_account": f"TokenAcct{idx}111111111111111111111111111111111",
                    "owner": owner,
                    "owner_match_status": pool_parser.OWNER_MATCHED_SIGNER,
                    "mint": pool_parser.USDC_MINT,
                    "ui_amount_delta_str": "10",
                }
            ]
            parsed["signer_wallets"] = ["RelayerWallet11111111111111111111111111111111", owner]
            events.extend(build_behavior_events([parsed]))
        summary = summarize_wallet_behavior(events)
        self.assertEqual(summary["unique_trader_wallets"], 2)
        self.assertIn(WARN_RELAYER, summary["warnings"])

    def test_wsol_contributes_to_native_volume(self) -> None:
        parsed = _parsed_event(side="BUY_BASE")
        parsed["quote_token_type"] = pool_parser.QUOTE_WSOL
        parsed["approx_usd_value"] = None
        parsed["quote_amount_native"] = "2.5"
        parsed["quote_delta_pool_str"] = "2.5"
        events = build_behavior_events([parsed])
        summary = summarize_wallet_behavior(events)
        self.assertEqual(summary["wallets"][0]["gross_quote_native_volume"], "2.5")
        self.assertEqual(summary["wallets"][0]["net_wsol"], "-2.5")

    def test_whale_accumulator_requires_net_buy(self) -> None:
        events = build_behavior_events(
            [
                _parsed_event(side="BUY_BASE", usdc="1500", signature=f"sig{i}")
                for i in range(12)
            ]
        )
        summary = summarize_wallet_behavior(events)
        self.assertEqual(summary["likely_behavior"], "possible_whale_accumulator")

    def test_whale_dumper_requires_net_sell(self) -> None:
        events = build_behavior_events(
            [
                _parsed_event(side="SELL_BASE", usdc="1500", signature=f"sig{i}")
                for i in range(12)
            ]
        )
        summary = summarize_wallet_behavior(events)
        self.assertEqual(summary["likely_behavior"], "possible_whale_dumper")

    def test_behavior_not_used_for_trades(self) -> None:
        from app.execution.paper import get_paper_trader

        trader = get_paper_trader()
        before = trader.get_state_snapshot()
        summarize_wallet_behavior(build_behavior_events([_parsed_event()]))
        after = trader.get_state_snapshot()
        self.assertEqual(before, after)

    def test_no_sqlite_writes(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as fh:
            db_path = fh.name
        try:
            before = os.path.getmtime(db_path)
            with patch("scripts.probe_solana_pool_activity.SolanaRpcClient") as mock_rpc:
                client = MagicMock()
                mock_rpc.return_value = client
                client.get_rpc_url.return_value = "https://private.example.com"
                client.stats = MagicMock(
                    rpc_calls_attempted=0,
                    rpc_calls_succeeded=0,
                    rpc_rate_limited_count=0,
                    rpc_forbidden_count=0,
                    rpc_retry_count=0,
                    rpc_timeout_count=0,
                    rpc_null_result_count=0,
                    cache_hits=0,
                    cache_misses=0,
                )
                client.get_signatures_for_address.return_value = []
                with tempfile.TemporaryDirectory() as tmp:
                    run_probe(pool_address="Pool111", audit_dir=Path(tmp), rpc_url="https://private.example.com")
            after = os.path.getmtime(db_path)
            conn = sqlite3.connect(db_path)
            conn.execute("SELECT 1")
            conn.close()
            self.assertEqual(before, after)
        finally:
            os.unlink(db_path)

    def test_no_trading_behavior_changes(self) -> None:
        from app.engine import generate_signal

        pair = {
            "priceUsd": "0.001",
            "liquidity": {"usd": 30_000},
            "volume": {"h24": 100_000, "h1": 5000},
            "priceChange": {"h24": 10, "h1": 5},
            "txns": {"h24": {"buys": 600, "sells": 400}},
        }
        self.assertEqual(generate_signal(pair, 0.55)["action"], "BUY")

    @patch("scripts.probe_solana_pool_activity.validate_signatures_against_raw")
    @patch("scripts.probe_solana_pool_activity.SolanaRpcClient")
    def test_probe_helius_off_by_default(self, mock_rpc: MagicMock, mock_helius: MagicMock) -> None:
        client = MagicMock()
        mock_rpc.return_value = client
        client.get_rpc_url.return_value = "https://private.example.com"
        client.stats = MagicMock(
            rpc_calls_attempted=0,
            rpc_calls_succeeded=0,
            rpc_rate_limited_count=0,
            rpc_forbidden_count=0,
            rpc_retry_count=0,
            rpc_timeout_count=0,
            rpc_null_result_count=0,
            cache_hits=0,
            cache_misses=0,
        )
        client.get_signatures_for_address.return_value = []
        mock_helius.return_value = {"helius_validation_enabled": False}
        with tempfile.TemporaryDirectory() as tmp:
            run_probe(pool_address="Pool111", audit_dir=Path(tmp), rpc_url="https://private.example.com")
        mock_helius.assert_called_once()
        self.assertFalse(mock_helius.call_args.kwargs["enabled"])


class HeliusValidationIntegrationTests(unittest.TestCase):
    def test_validate_signatures_disabled(self) -> None:
        result = validate_signatures_against_raw([], enabled=False)
        self.assertFalse(result["helius_validation_enabled"])
        self.assertEqual(result["helius_status"], HELIUS_SKIPPED_NOT_ENABLED)

    @patch("app.providers.helius.httpx.Client")
    def test_rate_limited_does_not_crash(self, mock_client_cls: MagicMock) -> None:
        os.environ["HELIUS_API_KEY"] = "key"
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = _mock_response(status_code=429)
        with tempfile.TemporaryDirectory() as tmp:
            result = validate_signatures_against_raw(
                [_parsed_event(signature="sig1")],
                enabled=True,
                validation_limit=1,
                cache_dir=Path(tmp),
            )
        self.assertEqual(result["helius_status"], HELIUS_RATE_LIMITED)


if __name__ == "__main__":
    unittest.main()

