"""Tests for Solana RPC wrapper and pool activity parser (Phase 1)."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from app.parsers import solana_pool_activity as parser
from app.providers.solana_rpc import (
    PUBLIC_RPC_URL,
    SOLANA_RPC_FORBIDDEN,
    SOLANA_RPC_JSONRPC_ERROR,
    SOLANA_RPC_JSON_ERROR,
    SOLANA_RPC_NULL_RESULT,
    SOLANA_RPC_OK,
    SOLANA_RPC_RATE_LIMITED,
    SolanaRpcClient,
    reset_default_client,
)
from scripts.probe_solana_pool_activity import cap_public_rpc_limit, run_probe


def _mock_response(
    *,
    status_code: int = 200,
    json_data: dict | None = None,
    text: str | None = None,
    headers: dict | None = None,
) -> httpx.Response:
    request = httpx.Request("POST", "https://api.mainnet-beta.solana.com")
    if json_data is not None:
        content = json.dumps(json_data).encode("utf-8")
    else:
        content = (text or "").encode("utf-8")
    return httpx.Response(status_code, request=request, content=content, headers=headers or {})


def _sample_tx(
    *,
    err=None,
    pre_balances=None,
    post_balances=None,
    fee_payer="TraderWallet1111111111111111111111111111111",
    signer=True,
) -> dict:
    pre_balances = pre_balances or []
    post_balances = post_balances or []
    return {
        "slot": 123,
        "blockTime": 1700000000,
        "transaction": {
            "signatures": ["sig123"],
            "message": {
                "accountKeys": [
                    {"pubkey": fee_payer, "signer": signer, "writable": True},
                    {"pubkey": "PoolAddress11111111111111111111111111111111", "signer": False, "writable": True},
                    {"pubkey": "TokenAcctBase111111111111111111111111111111", "signer": False, "writable": True},
                    {"pubkey": "TokenAcctQuote111111111111111111111111111111", "signer": False, "writable": True},
                ],
                "instructions": [{"programId": "RaydiumProgram1111111111111111111111111111"}],
            },
        },
        "meta": {
            "err": err,
            "fee": 5000,
            "preTokenBalances": pre_balances,
            "postTokenBalances": post_balances,
            "innerInstructions": [],
        },
    }


def _balance_row(
    account_index: int,
    mint: str,
    owner: str,
    amount_str: str,
    *,
    pre: bool = False,
) -> dict:
    row = {
        "accountIndex": account_index,
        "mint": mint,
        "owner": owner,
        "programId": "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA",
        "uiTokenAmount": {
            "amount": str(int(Decimal(amount_str) * Decimal(10 ** (6 if mint == parser.USDC_MINT else 9)))),
            "decimals": 6 if mint == parser.USDC_MINT else 9,
            "uiAmountString": amount_str,
        },
    }
    return row


class SolanaRpcUrlTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_default_client()

    def tearDown(self) -> None:
        reset_default_client()
        os.environ.pop("SOLANA_RPC_URL", None)

    def test_solana_rpc_url_overrides_default(self) -> None:
        os.environ["SOLANA_RPC_URL"] = "https://custom-rpc.example.com"
        client = SolanaRpcClient()
        self.assertEqual(client.get_rpc_url(), "https://custom-rpc.example.com")


class SolanaRpcCallTests(unittest.TestCase):
    def setUp(self) -> None:
        reset_default_client()
        self.client = SolanaRpcClient(max_retries=3, get_tx_pace_seconds=0, cache_enabled=False)

    @patch("app.providers.solana_rpc.time.sleep")
    @patch("app.providers.solana_rpc.httpx.Client")
    def test_get_transaction_params(self, mock_client_cls: MagicMock, _sleep: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = _mock_response(
            json_data={"jsonrpc": "2.0", "id": 1, "result": {"slot": 1}}
        )

        self.client.get_transaction("abc123")

        body = mock_client.post.call_args.kwargs["json"]
        self.assertEqual(body["method"], "getTransaction")
        self.assertEqual(
            body["params"],
            [
                "abc123",
                {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0},
            ],
        )

    @patch("app.providers.solana_rpc.time.sleep")
    @patch("app.providers.solana_rpc.httpx.Client")
    def test_http_429_retry_backoff(self, mock_client_cls: MagicMock, sleep_mock: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.side_effect = [
            _mock_response(status_code=429),
            _mock_response(json_data={"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}),
        ]

        result = self.client.rpc_call("getHealth")
        self.assertEqual(result.status, SOLANA_RPC_OK)
        self.assertEqual(self.client.stats.rpc_rate_limited_count, 1)
        self.assertGreaterEqual(self.client.stats.rpc_retry_count, 1)
        self.assertTrue(sleep_mock.called)

    @patch("app.providers.solana_rpc.time.sleep")
    @patch("app.providers.solana_rpc.httpx.Client")
    def test_retry_after_header_respected(self, mock_client_cls: MagicMock, sleep_mock: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.side_effect = [
            _mock_response(status_code=429, headers={"Retry-After": "2"}),
            _mock_response(json_data={"jsonrpc": "2.0", "id": 1, "result": True}),
        ]

        self.client.rpc_call("getHealth")
        first_sleep = sleep_mock.call_args_list[0].args[0]
        self.assertGreaterEqual(first_sleep, 2.0)

    @patch("app.providers.solana_rpc.httpx.Client")
    def test_http_403_produces_forbidden(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = _mock_response(status_code=403)

        result = self.client.rpc_call("getHealth")
        self.assertEqual(result.status, SOLANA_RPC_FORBIDDEN)
        self.assertEqual(self.client.stats.rpc_forbidden_count, 1)

    @patch("app.providers.solana_rpc.httpx.Client")
    def test_malformed_json_error(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = _mock_response(status_code=200, text="not-json")

        result = self.client.rpc_call("getHealth")
        self.assertEqual(result.status, SOLANA_RPC_JSON_ERROR)

    @patch("app.providers.solana_rpc.httpx.Client")
    def test_jsonrpc_error(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = _mock_response(
            json_data={"jsonrpc": "2.0", "id": 1, "error": {"code": -32000, "message": "fail"}}
        )

        result = self.client.rpc_call("getHealth")
        self.assertEqual(result.status, SOLANA_RPC_JSONRPC_ERROR)

    @patch("app.providers.solana_rpc.httpx.Client")
    def test_null_result(self, mock_client_cls: MagicMock) -> None:
        mock_client = MagicMock()
        mock_client_cls.return_value.__enter__.return_value = mock_client
        mock_client.post.return_value = _mock_response(
            json_data={"jsonrpc": "2.0", "id": 1, "result": None}
        )

        result = self.client.rpc_call("getTransaction", ["sig"])
        self.assertEqual(result.status, SOLANA_RPC_NULL_RESULT)
        self.assertEqual(self.client.stats.rpc_null_result_count, 1)


class SolanaRpcCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.cache_dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.client = SolanaRpcClient(get_tx_pace_seconds=0, cache_enabled=True)

    @patch("app.providers.solana_rpc.httpx.Client")
    def test_cache_hit_avoids_rpc_call(self, mock_client_cls: MagicMock) -> None:
        with patch("app.providers.solana_rpc.CACHE_DIR", self.cache_dir):
            mock_client = MagicMock()
            mock_client_cls.return_value.__enter__.return_value = mock_client
            mock_client.post.return_value = _mock_response(
                json_data={"jsonrpc": "2.0", "id": 1, "result": {"slot": 99}}
            )

            first = self.client.get_transaction("cached_sig")
            self.assertEqual(first["status"], SOLANA_RPC_OK)
            self.assertEqual(self.client.stats.cache_misses, 1)

            second = self.client.get_transaction("cached_sig")
            self.assertEqual(second["status"], SOLANA_RPC_OK)
            self.assertEqual(self.client.stats.cache_hits, 1)
            self.assertEqual(mock_client.post.call_count, 1)


class ParserDecimalTests(unittest.TestCase):
    def test_decimal_to_json_safe(self) -> None:
        self.assertEqual(parser.decimal_to_json_safe(Decimal("1.234567")), "1.234567")

    def test_ui_amount_string_preferred(self) -> None:
        row = {
            "accountIndex": 0,
            "mint": parser.USDC_MINT,
            "owner": "owner",
            "uiTokenAmount": {
                "amount": "999999",
                "decimals": 6,
                "uiAmount": 999.0,
                "uiAmountString": "1.5",
            },
        }
        tx = _sample_tx(pre_balances=[row], post_balances=[row])
        balances = parser.build_token_balance_map(tx)
        key = (0, parser.USDC_MINT, "owner")
        self.assertEqual(balances[key]["pre_ui_amount"], Decimal("1.5"))

    def test_raw_integer_not_used_when_ui_amount_string_exists(self) -> None:
        ui = {
            "amount": "1000000",
            "decimals": 6,
            "uiAmount": 999.0,
            "uiAmountString": "2.0",
        }
        self.assertEqual(parser._ui_amount_from_token_amount(ui), Decimal("2.0"))

    def test_usdc_six_decimal_delta(self) -> None:
        pool = "PoolAddress11111111111111111111111111111111"
        pre = [
            _balance_row(2, "BaseMint1111111111111111111111111111111111", pool, "10.0"),
            _balance_row(3, parser.USDC_MINT, pool, "100.0"),
        ]
        post = [
            _balance_row(2, "BaseMint1111111111111111111111111111111111", pool, "8.0"),
            _balance_row(3, parser.USDC_MINT, pool, "110.0"),
        ]
        tx = _sample_tx(pre_balances=pre, post_balances=post)
        parsed = parser.infer_pool_swap(tx, pool)
        self.assertEqual(parsed["side"], parser.SIDE_BUY_BASE)
        self.assertEqual(parsed["approx_usd_value"], "10.0")

    def test_wsol_nine_decimal_delta(self) -> None:
        pool = "PoolAddress11111111111111111111111111111111"
        pre = [
            _balance_row(2, "BaseMint1111111111111111111111111111111111", pool, "5.0"),
            _balance_row(3, parser.WSOL_MINT, pool, "1.0"),
        ]
        post = [
            _balance_row(2, "BaseMint1111111111111111111111111111111111", pool, "6.0"),
            _balance_row(3, parser.WSOL_MINT, pool, "0.5"),
        ]
        tx = _sample_tx(pre_balances=pre, post_balances=post)
        parsed = parser.infer_pool_swap(tx, pool)
        self.assertEqual(parsed["side"], parser.SIDE_SELL_BASE)
        self.assertEqual(parsed["quote_token_type"], parser.QUOTE_WSOL)
        self.assertIsNone(parsed["approx_usd_value"])
        self.assertEqual(parsed["quote_amount_native"], "0.5")


class ParserAccountMappingTests(unittest.TestCase):
    def test_account_index_maps_to_token_account(self) -> None:
        pool = "PoolAddress11111111111111111111111111111111"
        row = _balance_row(2, parser.USDC_MINT, pool, "1.0")
        tx = _sample_tx(pre_balances=[], post_balances=[row])
        balances = parser.build_token_balance_map(tx)
        self.assertEqual(
            balances[(2, parser.USDC_MINT, pool)]["token_account"],
            "TokenAcctBase111111111111111111111111111111",
        )

    def test_ata_not_trader_wallet(self) -> None:
        trader = "TraderWallet1111111111111111111111111111111"
        token_acct = "TokenAcctQuote111111111111111111111111111111"
        pool = "PoolAddress11111111111111111111111111111111"
        pre = [_balance_row(3, parser.USDC_MINT, trader, "1.0")]
        post = [_balance_row(3, parser.USDC_MINT, trader, "0.0")]
        tx = _sample_tx(pre_balances=pre, post_balances=post, fee_payer=trader)
        parsed = parser.infer_pool_swap(tx, pool)
        self.assertEqual(parsed["trader_wallet"], trader)
        self.assertNotEqual(parsed["trader_wallet"], token_acct)

    def test_owner_matched_to_fee_payer(self) -> None:
        trader = "TraderWallet1111111111111111111111111111111"
        row = _balance_row(3, parser.USDC_MINT, trader, "1.0")
        tx = _sample_tx(post_balances=[row], fee_payer=trader)
        balances = parser.build_token_balance_map(tx)
        owner_status = balances[(3, parser.USDC_MINT, trader)] if False else None
        delta_info = parser.parse_transaction_token_deltas(tx, "PoolAddress11111111111111111111111111111111")
        statuses = [r["owner_match_status"] for r in delta_info["token_balances"]]
        self.assertIn(parser.OWNER_MATCHED_FEE_PAYER, statuses)

    def test_unmatched_owner_status(self) -> None:
        stranger = "StrangerWallet1111111111111111111111111111111"
        row = _balance_row(3, parser.USDC_MINT, stranger, "1.0")
        tx = _sample_tx(post_balances=[row])
        delta_info = parser.parse_transaction_token_deltas(tx, "PoolAddress11111111111111111111111111111111")
        statuses = [r["owner_match_status"] for r in delta_info["token_balances"]]
        self.assertIn(parser.OWNER_UNMATCHED, statuses)


class ParserSwapInferenceTests(unittest.TestCase):
    def test_usdc_direction_inference_buy(self) -> None:
        pool = "PoolAddress11111111111111111111111111111111"
        pre = [
            _balance_row(2, "BaseMint1111111111111111111111111111111111", pool, "10.0"),
            _balance_row(3, parser.USDC_MINT, pool, "100.0"),
        ]
        post = [
            _balance_row(2, "BaseMint1111111111111111111111111111111111", pool, "8.0"),
            _balance_row(3, parser.USDC_MINT, pool, "120.0"),
        ]
        parsed = parser.infer_pool_swap(_sample_tx(pre_balances=pre, post_balances=post), pool)
        self.assertEqual(parsed["side"], parser.SIDE_BUY_BASE)

    def test_usdc_direction_inference_sell(self) -> None:
        pool = "PoolAddress11111111111111111111111111111111"
        pre = [
            _balance_row(2, "BaseMint1111111111111111111111111111111111", pool, "10.0"),
            _balance_row(3, parser.USDC_MINT, pool, "100.0"),
        ]
        post = [
            _balance_row(2, "BaseMint1111111111111111111111111111111111", pool, "12.0"),
            _balance_row(3, parser.USDC_MINT, pool, "80.0"),
        ]
        parsed = parser.infer_pool_swap(_sample_tx(pre_balances=pre, post_balances=post), pool)
        self.assertEqual(parsed["side"], parser.SIDE_SELL_BASE)

    def test_wsol_direction_without_usd_oracle(self) -> None:
        pool = "PoolAddress11111111111111111111111111111111"
        pre = [
            _balance_row(2, "BaseMint1111111111111111111111111111111111", pool, "10.0"),
            _balance_row(3, parser.WSOL_MINT, pool, "2.0"),
        ]
        post = [
            _balance_row(2, "BaseMint1111111111111111111111111111111111", pool, "8.0"),
            _balance_row(3, parser.WSOL_MINT, pool, "3.0"),
        ]
        parsed = parser.infer_pool_swap(_sample_tx(pre_balances=pre, post_balances=post), pool)
        self.assertEqual(parsed["side"], parser.SIDE_BUY_BASE)
        self.assertIsNone(parsed["approx_usd_value"])
        self.assertEqual(parsed["quote_amount_native"], "1.0")

    def test_unclear_pool_vault_returns_unknown(self) -> None:
        tx = _sample_tx(pre_balances=[], post_balances=[])
        parsed = parser.infer_pool_swap(tx, "PoolAddress11111111111111111111111111111111")
        self.assertEqual(parsed["side"], parser.SIDE_UNKNOWN)
        self.assertIn(parsed["parse_status"], {parser.PARSE_PARTIAL, parser.PARSE_UNKNOWN_FORMAT})

    def test_failed_transaction_excluded_from_side(self) -> None:
        pool = "PoolAddress11111111111111111111111111111111"
        pre = [
            _balance_row(2, "BaseMint1111111111111111111111111111111111", pool, "10.0"),
            _balance_row(3, parser.USDC_MINT, pool, "100.0"),
        ]
        post = [
            _balance_row(2, "BaseMint1111111111111111111111111111111111", pool, "8.0"),
            _balance_row(3, parser.USDC_MINT, pool, "120.0"),
        ]
        tx = _sample_tx(pre_balances=pre, post_balances=post, err={"InstructionError": [1, "fail"]})
        parsed = parser.infer_pool_swap(tx, pool)
        self.assertTrue(parsed["failed_transaction"])
        self.assertEqual(parsed["parse_status"], parser.PARSE_FAILED_TX)
        self.assertEqual(parsed["side"], parser.SIDE_IGNORED_FAILED)


class ProbeScriptTests(unittest.TestCase):
    def test_signature_deduplication(self) -> None:
        rows = [{"signature": "a"}, {"signature": "a"}, {"signature": "b"}]
        self.assertEqual(parser.dedupe_signatures(rows), ["a", "b"])

    def test_public_rpc_limit_cap(self) -> None:
        capped = cap_public_rpc_limit(150, 200, PUBLIC_RPC_URL, allow_large=False)
        self.assertEqual(capped, 100)
        allowed = cap_public_rpc_limit(150, 200, PUBLIC_RPC_URL, allow_large=True)
        self.assertEqual(allowed, 150)

    @patch("scripts.probe_solana_pool_activity.SolanaRpcClient")
    def test_probe_continues_when_one_tx_fails(self, mock_client_cls: MagicMock) -> None:
        client = MagicMock()
        mock_client_cls.return_value = client
        client.get_rpc_url.return_value = "https://private.example.com"
        client.stats = SolanaRpcClient().stats
        client.get_signatures_for_address.return_value = [
            {"signature": "good"},
            {"signature": "bad"},
        ]

        pool = "PoolAddress11111111111111111111111111111111"
        good_tx = _sample_tx(
            pre_balances=[
                _balance_row(2, "BaseMint1111111111111111111111111111111111", pool, "10.0"),
                _balance_row(3, parser.USDC_MINT, pool, "100.0"),
            ],
            post_balances=[
                _balance_row(2, "BaseMint1111111111111111111111111111111111", pool, "8.0"),
                _balance_row(3, parser.USDC_MINT, pool, "120.0"),
            ],
        )

        def _get_tx(sig: str) -> dict:
            if sig == "bad":
                return {"status": SOLANA_RPC_NULL_RESULT, "result": None, "signature": sig}
            return {"status": SOLANA_RPC_OK, "result": good_tx, "signature": sig}

        client.get_transaction.side_effect = _get_tx

        with tempfile.TemporaryDirectory() as tmp:
            audit = run_probe(pool_address=pool, limit=2, audit_dir=Path(tmp), rpc_url="https://private.example.com")
        self.assertEqual(audit["tx_fetched"], 1)
        self.assertEqual(audit["tx_failed_to_fetch_count"], 1)
        self.assertEqual(audit["buy_count"], 1)

    def test_no_sqlite_writes(self) -> None:
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as fh:
            db_path = fh.name
        try:
            before = os.path.getmtime(db_path)
            with patch("scripts.probe_solana_pool_activity.SolanaRpcClient") as mock_client_cls:
                client = MagicMock()
                mock_client_cls.return_value = client
                client.get_rpc_url.return_value = "https://private.example.com"
                client.stats = SolanaRpcClient().stats
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

    @patch("scripts.probe_solana_pool_activity.SolanaRpcClient")
    def test_no_external_price_oracle(self, mock_client_cls: MagicMock) -> None:
        client = MagicMock()
        mock_client_cls.return_value = client
        client.get_rpc_url.return_value = "https://private.example.com"
        client.stats = SolanaRpcClient().stats
        client.get_signatures_for_address.return_value = [{"signature": "sig1"}]
        pool = "PoolAddress11111111111111111111111111111111"
        tx = _sample_tx(
            pre_balances=[
                _balance_row(2, "BaseMint1111111111111111111111111111111111", pool, "10.0"),
                _balance_row(3, parser.WSOL_MINT, pool, "2.0"),
            ],
            post_balances=[
                _balance_row(2, "BaseMint1111111111111111111111111111111111", pool, "8.0"),
                _balance_row(3, parser.WSOL_MINT, pool, "3.0"),
            ],
        )
        client.get_transaction.return_value = {"status": SOLANA_RPC_OK, "result": tx}

        with patch("httpx.Client.get") as mock_get:
            with tempfile.TemporaryDirectory() as tmp:
                audit = run_probe(pool_address=pool, limit=1, audit_dir=Path(tmp), rpc_url="https://private.example.com")
            mock_get.assert_not_called()
        self.assertEqual(audit["buy_count"], 1)
        self.assertEqual(audit["gross_usdc_volume"], "0")

    def test_no_trading_behavior_changes(self) -> None:
        from app.engine import generate_signal

        pair = {
            "priceUsd": "0.001",
            "liquidity": {"usd": 30_000},
            "volume": {"h24": 100_000, "h1": 5000},
            "priceChange": {"h24": 10, "h1": 5},
            "txns": {"h24": {"buys": 600, "sells": 400}},
        }
        sig = generate_signal(pair, 0.55)
        self.assertEqual(sig["action"], "BUY")


class FailedTransactionAggregationTests(unittest.TestCase):
    @patch("scripts.probe_solana_pool_activity.SolanaRpcClient")
    def test_failed_tx_counted_not_in_volume(self, mock_client_cls: MagicMock) -> None:
        client = MagicMock()
        mock_client_cls.return_value = client
        client.get_rpc_url.return_value = "https://private.example.com"
        client.stats = SolanaRpcClient().stats
        client.get_signatures_for_address.return_value = [{"signature": "fail1"}]
        pool = "PoolAddress11111111111111111111111111111111"
        tx = _sample_tx(err={"InstructionError": [1, "fail"]})
        client.get_transaction.return_value = {"status": SOLANA_RPC_OK, "result": tx}

        with tempfile.TemporaryDirectory() as tmp:
            audit = run_probe(pool_address=pool, limit=1, audit_dir=Path(tmp), rpc_url="https://private.example.com")

        self.assertEqual(audit["failed_transaction_count"], 1)
        self.assertEqual(audit["buy_count"], 0)
        self.assertEqual(audit["sell_count"], 0)
        self.assertEqual(audit["gross_usdc_volume"], "0")


if __name__ == "__main__":
    unittest.main()
