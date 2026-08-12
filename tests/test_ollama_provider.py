"""Ollama LLM provider selection, parsing, and persistence tests."""
from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx

from app.llm_client import (
    OLLAMA_PROVIDER_TAG,
    _ollama_chat_endpoint,
    extract_json_object,
    fallback_ollama_response,
    generate_assistant_reply,
    generate_decision,
    normalize_ollama_response,
    normalize_strategy_type,
    parse_ollama_response_text,
)


def _mock_httpx_response(
    *,
    status_code: int = 200,
    json_data: dict | None = None,
    text: str | None = None,
    raise_status: bool = True,
) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    if json_data is not None:
        response.json.return_value = json_data
        response.text = json.dumps(json_data)
    elif text is not None:
        response.text = text

        def _bad_json() -> dict:
            raise json.JSONDecodeError("Expecting value", text, 0)

        response.json.side_effect = _bad_json
    else:
        response.json.return_value = {}
        response.text = "{}"

    if raise_status and status_code >= 400:
        request = httpx.Request("POST", "http://127.0.0.1:11434/api/chat")
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"HTTP {status_code}",
            request=request,
            response=httpx.Response(status_code, request=request),
        )
    else:
        response.raise_for_status.return_value = None
    return response


class OllamaParserTests(unittest.TestCase):
    def test_clean_json_parsing(self) -> None:
        raw = {
            "action": "BUY",
            "strategy_type": "MOMENTUM",
            "confidence": 0.82,
            "rationale": "Strong momentum",
            "risk_level": "MEDIUM",
            "expected_direction": "UP",
            "provider": OLLAMA_PROVIDER_TAG,
        }
        parsed = normalize_ollama_response(raw)
        self.assertEqual(parsed["action"], "BUY")
        self.assertEqual(parsed["confidence"], 0.82)
        self.assertEqual(parsed["provider"], OLLAMA_PROVIDER_TAG)

    def test_markdown_json_parsing(self) -> None:
        text = '```json\n{"action":"HOLD","strategy_type":"WHALE_FLOW","confidence":0.5,"rationale":"wait","risk_level":"LOW","expected_direction":"SIDEWAYS","provider":"ollama_qwen3_8b"}\n```'
        parsed = parse_ollama_response_text(text)
        self.assertEqual(parsed["action"], "HOLD")
        self.assertEqual(parsed["strategy_type"], "WHALE_FLOW")

    def test_corrupted_json_fallback(self) -> None:
        parsed = parse_ollama_response_text("not json at all")
        self.assertEqual(parsed["action"], "SKIPPED")
        self.assertIn("invalid response", parsed["rationale"].lower())

    def test_extract_json_from_surrounding_text(self) -> None:
        text = 'Analysis complete.\n{"action":"SELL","strategy_type":"RISK_OFF","confidence":1.5,"rationale":"exit","risk_level":"HIGH","expected_direction":"DOWN","provider":"ollama_qwen3_8b"}\nDone.'
        obj = extract_json_object(text)
        parsed = normalize_ollama_response(obj)
        self.assertEqual(parsed["action"], "SELL")
        self.assertEqual(parsed["confidence"], 1.0)

    def test_fallback_response_shape(self) -> None:
        parsed = fallback_ollama_response()
        self.assertEqual(parsed["action"], "SKIPPED")
        self.assertEqual(parsed["provider"], OLLAMA_PROVIDER_TAG)

    def test_whale_rider_maps_to_whale_flow(self) -> None:
        self.assertEqual(normalize_strategy_type("WHALE_RIDER"), "WHALE_FLOW")
        parsed = normalize_ollama_response({
            "action": "BUY",
            "strategy_type": "WHALE_RIDER",
            "confidence": 0.7,
            "rationale": "whale accumulation",
            "risk_level": "MEDIUM",
            "expected_direction": "UP",
        })
        self.assertEqual(parsed["strategy_type"], "WHALE_FLOW")


class OllamaNativeRestTests(unittest.TestCase):
    """Isolated unit tests for native Ollama REST — no real server, no openai."""

    def setUp(self) -> None:
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["ENABLE_GEMINI"] = "false"
        os.environ["OLLAMA_BASE_URL"] = "http://127.0.0.1:11434"
        os.environ["OLLAMA_MODEL"] = "qwen3:8b"
        # Ensure no OpenAI key is required
        os.environ.pop("OPENAI_API_KEY", None)

    def tearDown(self) -> None:
        for key in (
            "LLM_PROVIDER",
            "ENABLE_GEMINI",
            "OLLAMA_BASE_URL",
            "OLLAMA_MODEL",
        ):
            os.environ.pop(key, None)

    def test_import_does_not_require_openai(self) -> None:
        import app.llm_client as llm_client

        source = Path(llm_client.__file__).read_text(encoding="utf-8")
        self.assertNotIn("import openai", source)
        self.assertNotIn("from openai", source)
        self.assertIn("httpx", source)

        blocked = {name: mod for name, mod in sys.modules.items() if name == "openai" or name.startswith("openai.")}
        for name in list(blocked):
            sys.modules.pop(name, None)

        with patch.dict(sys.modules, {"openai": None}):
            importlib.reload(llm_client)
            self.assertTrue(hasattr(llm_client, "generate_decision"))
            self.assertTrue(hasattr(llm_client, "generate_assistant_reply"))

        importlib.reload(llm_client)

    def test_generate_decision_does_not_import_openai(self) -> None:
        mock_response = _mock_httpx_response(
            json_data={
                "message": {
                    "content": (
                        '{"action":"SKIPPED","confidence":0.0,"rationale":"test",'
                        '"risk_level":"LOW","expected_direction":"SIDEWAYS",'
                        '"provider":"ollama_qwen3_8b"}'
                    )
                },
                "done": True,
            }
        )
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.return_value = mock_response

        with patch("app.llm_client.httpx.Client", return_value=mock_client), patch.dict(
            sys.modules, {"openai": None}
        ):
            parsed = generate_decision("prompt", {"symbol": "X"})

        self.assertEqual(parsed["action"], "SKIPPED")
        mock_client.post.assert_called_once()

    def test_native_payload_includes_required_fields(self) -> None:
        mock_response = _mock_httpx_response(
            json_data={
                "message": {
                    "content": (
                        '{"action":"HOLD","strategy_type":"SENTIMENT","confidence":0.4,'
                        '"rationale":"neutral","risk_level":"LOW","expected_direction":"SIDEWAYS",'
                        '"provider":"ollama_qwen3_8b"}'
                    )
                },
                "done": True,
            }
        )
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.return_value = mock_response

        with patch("app.llm_client.httpx.Client", return_value=mock_client):
            parsed = generate_decision("prompt", {"symbol": "X"})

        self.assertEqual(parsed["action"], "HOLD")
        endpoint, kwargs = mock_client.post.call_args[0][0], mock_client.post.call_args[1]
        self.assertEqual(endpoint, "http://127.0.0.1:11434/api/chat")
        payload = kwargs["json"]
        self.assertIs(payload["stream"], False)
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["model"], "qwen3:8b")
        self.assertEqual(payload["options"]["temperature"], 0)

    def test_base_url_normalization(self) -> None:
        cases = [
            "http://127.0.0.1:11434",
            "http://127.0.0.1:11434/",
            "http://127.0.0.1:11434/v1",
            "http://127.0.0.1:11434/v1/",
        ]
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertEqual(
                    _ollama_chat_endpoint(raw),
                    "http://127.0.0.1:11434/api/chat",
                )

    def test_valid_native_response_parses_content_json(self) -> None:
        mock_response = _mock_httpx_response(
            json_data={
                "message": {
                    "content": (
                        '{"action":"SKIPPED","confidence":0.0,"rationale":"test",'
                        '"risk_level":"LOW","expected_direction":"SIDEWAYS",'
                        '"provider":"ollama_qwen3_8b"}'
                    )
                },
                "done": True,
            }
        )
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.return_value = mock_response

        with patch("app.llm_client.httpx.Client", return_value=mock_client):
            parsed = generate_decision("prompt", {"symbol": "X"})

        self.assertEqual(parsed["action"], "SKIPPED")
        self.assertEqual(parsed["rationale"], "test")
        self.assertEqual(parsed["provider"], OLLAMA_PROVIDER_TAG)

    def test_thinking_only_empty_content_rejected(self) -> None:
        mock_response = _mock_httpx_response(
            json_data={
                "message": {
                    "content": "",
                    "thinking": "reasoning text",
                },
                "done": True,
            }
        )
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.return_value = mock_response

        with patch("app.llm_client.httpx.Client", return_value=mock_client):
            parsed = generate_decision("prompt", {"symbol": "X"})

        self.assertEqual(parsed["action"], "SKIPPED")
        self.assertEqual(parsed["rationale"], "ollama_empty_content_thinking_only")

    def test_top_level_ollama_error_safe_fallback(self) -> None:
        mock_response = _mock_httpx_response(json_data={"error": "model not found"})
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.return_value = mock_response

        with patch("app.llm_client.httpx.Client", return_value=mock_client):
            parsed = generate_decision("prompt", {"symbol": "X"})

        self.assertEqual(parsed["action"], "SKIPPED")
        self.assertIn("ollama_error_response", parsed["rationale"])
        self.assertIn("model not found", parsed["rationale"])

    def test_invalid_json_content_safe_fallback(self) -> None:
        mock_response = _mock_httpx_response(
            json_data={
                "message": {"content": "not-json-at-all"},
                "done": True,
            }
        )
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.return_value = mock_response

        with patch("app.llm_client.httpx.Client", return_value=mock_client):
            parsed = generate_decision("prompt", {"symbol": "X"})

        self.assertEqual(parsed["action"], "SKIPPED")

    def test_missing_message_key_safe_fallback(self) -> None:
        mock_response = _mock_httpx_response(json_data={"done": True})
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.return_value = mock_response

        with patch("app.llm_client.httpx.Client", return_value=mock_client):
            parsed = generate_decision("prompt", {"symbol": "X"})

        self.assertEqual(parsed["action"], "SKIPPED")
        self.assertEqual(parsed["rationale"], "ollama_empty_content")

    def test_timeout_safe_fallback(self) -> None:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.side_effect = httpx.TimeoutException("timed out")

        with patch("app.llm_client.httpx.Client", return_value=mock_client):
            parsed = generate_decision("prompt", {"symbol": "X"})

        self.assertEqual(parsed["action"], "SKIPPED")
        self.assertIn("invalid response", parsed["rationale"].lower())

    def test_connection_error_safe_fallback(self) -> None:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.side_effect = httpx.ConnectError("connection refused")

        with patch("app.llm_client.httpx.Client", return_value=mock_client):
            parsed = generate_decision("prompt", {"symbol": "X"})

        self.assertEqual(parsed["action"], "SKIPPED")
        self.assertIn("invalid response", parsed["rationale"].lower())

    def test_http_404_safe_fallback(self) -> None:
        mock_response = _mock_httpx_response(status_code=404, json_data={"error": "not found"})
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.return_value = mock_response

        with patch("app.llm_client.httpx.Client", return_value=mock_client):
            parsed = generate_decision("prompt", {"symbol": "X"})

        self.assertEqual(parsed["action"], "SKIPPED")

    def test_http_500_safe_fallback(self) -> None:
        mock_response = _mock_httpx_response(status_code=500, json_data={"error": "server"})
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.return_value = mock_response

        with patch("app.llm_client.httpx.Client", return_value=mock_client):
            parsed = generate_decision("prompt", {"symbol": "X"})

        self.assertEqual(parsed["action"], "SKIPPED")

    def test_assistant_reply_native_payload_think_false(self) -> None:
        mock_response = _mock_httpx_response(
            json_data={
                "message": {"content": "Status looks fine."},
                "done": True,
            }
        )
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.return_value = mock_response

        with patch("app.llm_client.httpx.Client", return_value=mock_client):
            reply = generate_assistant_reply(
                user_message="status?",
                context_json_text="{}",
            )

        self.assertEqual(reply, "Status looks fine.")
        payload = mock_client.post.call_args[1]["json"]
        self.assertIs(payload["think"], False)
        self.assertIs(payload["stream"], False)
        self.assertEqual(payload["model"], "qwen3:8b")
        self.assertNotIn("format", payload)
        self.assertEqual(payload["options"]["temperature"], 0.3)

    def test_assistant_reply_does_not_return_thinking(self) -> None:
        mock_response = _mock_httpx_response(
            json_data={
                "message": {
                    "content": "",
                    "thinking": "secret reasoning that must not leak",
                },
                "done": True,
            }
        )
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.return_value = mock_response

        with patch("app.llm_client.httpx.Client", return_value=mock_client):
            with self.assertRaises(RuntimeError) as ctx:
                generate_assistant_reply(
                    user_message="status?",
                    context_json_text="{}",
                )

        self.assertIn("no usable final content", str(ctx.exception).lower())
        self.assertNotIn("secret reasoning", str(ctx.exception))

    def test_enable_gemini_false_does_not_call_gemini(self) -> None:
        self.assertEqual(os.environ.get("ENABLE_GEMINI"), "false")
        with patch.dict(sys.modules, {"app.gemini_context": MagicMock()}):
            # Importing llm_client must not pull Gemini adapter paths
            import app.llm_client as llm_client

            source = Path(llm_client.__file__).read_text(encoding="utf-8")
            self.assertNotIn("gemini", source.lower())

            mock_response = _mock_httpx_response(
                json_data={
                    "message": {
                        "content": (
                            '{"action":"SKIPPED","confidence":0.0,"rationale":"ok",'
                            '"risk_level":"LOW","expected_direction":"SIDEWAYS",'
                            '"provider":"ollama_qwen3_8b"}'
                        )
                    },
                    "done": True,
                }
            )
            mock_client = MagicMock()
            mock_client.__enter__.return_value = mock_client
            mock_client.__exit__.return_value = False
            mock_client.post.return_value = mock_response

            with patch("app.llm_client.httpx.Client", return_value=mock_client):
                parsed = generate_decision("prompt", {"symbol": "X"})

            self.assertEqual(parsed["action"], "SKIPPED")
            # Gemini modules must not be required/imported by this path
            self.assertNotIn("app.llm_audit.gemini_adapter", sys.modules)
            self.assertNotIn("app.intelligent_agents.gemini_selective_audit", sys.modules)

    def test_no_openai_api_key_required(self) -> None:
        self.assertIsNone(os.environ.get("OPENAI_API_KEY"))
        mock_response = _mock_httpx_response(
            json_data={
                "message": {
                    "content": (
                        '{"action":"HOLD","strategy_type":"SENTIMENT","confidence":0.4,'
                        '"rationale":"neutral","risk_level":"LOW","expected_direction":"SIDEWAYS",'
                        '"provider":"ollama_qwen3_8b"}'
                    )
                },
                "done": True,
            }
        )
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.return_value = mock_response

        with patch("app.llm_client.httpx.Client", return_value=mock_client):
            parsed = generate_decision("prompt", {"symbol": "X"})

        self.assertEqual(parsed["action"], "HOLD")


class OllamaProviderTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        os.environ["TRADER_DB_PATH"] = str(Path(self._tmpdir.name) / "test.db")
        os.environ["HEADLESS_DATA_COLLECTION"] = "false"
        os.environ["ENABLE_GEMINI"] = "false"
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["OLLAMA_MAX_CALLS_PER_SCAN"] = "5"

        import app.llm_config as llm_config
        import app.llm_client as llm_client
        import app.database as database
        import app.models.predictor as predictor

        llm_config.reset_llm_counters()
        importlib.reload(llm_config)
        importlib.reload(llm_client)
        importlib.reload(database)
        importlib.reload(predictor)

        llm_config.reset_llm_counters()
        self.llm_config = llm_config
        self.predictor = predictor
        self.db = database
        self.db.init_db()

        self.coin = self.db.upsert_coin({
            "symbol": "QWEN/SOL",
            "pair_address": "0xqwen",
            "chain": "solana",
            "price_usd": 2.0,
        })

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        for key in (
            "TRADER_DB_PATH",
            "HEADLESS_DATA_COLLECTION",
            "ENABLE_GEMINI",
            "LLM_PROVIDER",
            "OLLAMA_MAX_CALLS_PER_SCAN",
        ):
            os.environ.pop(key, None)

    def test_provider_selection(self) -> None:
        self.assertEqual(self.llm_config.get_llm_provider(), "ollama")
        self.assertTrue(self.llm_config.is_ollama_provider_active())
        self.assertFalse(self.llm_config.is_gemini_provider_active())
        self.assertTrue(self.llm_config.is_llm_enabled())

    def test_headless_overrides_ollama(self) -> None:
        os.environ["HEADLESS_DATA_COLLECTION"] = "true"
        import app.llm_config as llm_config

        importlib.reload(llm_config)
        self.assertFalse(llm_config.is_ollama_provider_active())
        self.assertFalse(llm_config.is_llm_enabled())

    def test_gemini_not_called_in_ollama_mode(self) -> None:
        metrics = {
            "symbol": "QWEN/SOL",
            "token_contract_address": "0xqwen",
            "price_usd": 2.0,
            "whale_score": 0.7,
            "price_change_1h": 2.0,
        }
        ollama_response = {
            "action": "BUY",
            "strategy_type": "WHALE_FLOW",
            "confidence": 0.77,
            "rationale": "Local model sees whale flow",
            "risk_level": "MEDIUM",
            "expected_direction": "UP",
            "provider": OLLAMA_PROVIDER_TAG,
        }

        with patch.object(self.predictor, "_gemini_json") as mock_gemini, patch.object(
            self.predictor, "generate_decision", return_value=ollama_response
        ):
            import asyncio

            decision, decision_id = asyncio.run(
                self.predictor.analyze_market_state(
                    metrics,
                    "OPPORTUNISTIC_SPECULATIVE",
                    0.1,
                    coin_id=self.coin["id"],
                    trigger_type="test_ollama",
                )
            )
            mock_gemini.assert_not_called()

        self.assertEqual(decision.decision, "BUY")
        self.assertIsNotNone(decision_id)
        status = self.llm_config.get_llm_runtime_status()
        self.assertEqual(status["gemini_call_count"], 0)
        self.assertGreaterEqual(status["ollama_call_count"], 0)

        stored = self.db.get_gemini_decisions(coin_id=self.coin["id"])[0]
        self.assertEqual(stored["provider"], "ollama")
        self.assertEqual(stored["model_source"], OLLAMA_PROVIDER_TAG)
        self.assertEqual(stored["action"], "BUY")

    def test_ollama_timeout_fallback_persisted(self) -> None:
        metrics = {
            "symbol": "QWEN/SOL",
            "token_contract_address": "0xqwen",
            "price_usd": 2.0,
            "whale_score": 0.7,
        }

        with patch.object(
            self.predictor,
            "generate_decision",
            return_value=fallback_ollama_response("timeout"),
        ):
            import asyncio

            decision, decision_id = asyncio.run(
                self.predictor.analyze_market_state(
                    metrics,
                    "OPPORTUNISTIC_SPECULATIVE",
                    0.0,
                    coin_id=self.coin["id"],
                    trigger_type="test_timeout",
                )
            )

        self.assertEqual(decision.decision, "HOLD")
        self.assertIsNotNone(decision_id)
        stored = self.db.get_gemini_decisions(coin_id=self.coin["id"])[0]
        self.assertEqual(stored["action"], "SKIPPED")
        self.assertIn("timeout", stored["rationale"].lower())
        self.assertEqual(stored["provider"], "ollama")

    def test_budget_skip_persisted(self) -> None:
        os.environ["OLLAMA_MAX_CALLS_PER_SCAN"] = "0"
        import app.llm_config as llm_config

        importlib.reload(llm_config)
        llm_config.reset_ollama_scan_budget()

        metrics = {
            "symbol": "QWEN/SOL",
            "token_contract_address": "0xqwen",
            "price_usd": 2.0,
            "whale_score": 0.7,
        }

        with patch.object(self.predictor, "generate_decision") as mock_gen:
            import asyncio

            _decision, decision_id = asyncio.run(
                self.predictor.analyze_market_state(
                    metrics,
                    "OPPORTUNISTIC_SPECULATIVE",
                    0.0,
                    coin_id=self.coin["id"],
                    trigger_type="test_budget",
                )
            )
            mock_gen.assert_not_called()

        stored = self.db.get_gemini_decisions(coin_id=self.coin["id"])[0]
        self.assertEqual(stored["action"], "SKIPPED")
        self.assertEqual(stored["rationale"], llm_config.SKIP_REASON_BUDGET)
        self.assertIsNotNone(decision_id)

    def test_generate_decision_request_error_fallback(self) -> None:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.side_effect = httpx.ConnectError("connection refused")

        with patch("app.llm_client.httpx.Client", return_value=mock_client):
            parsed = generate_decision("prompt", {"symbol": "X"})
        self.assertEqual(parsed["action"], "SKIPPED")
        self.assertIn("invalid response", parsed["rationale"].lower())

    def test_generate_decision_success_mock(self) -> None:
        mock_response = _mock_httpx_response(
            json_data={
                "message": {
                    "content": (
                        '{"action":"HOLD","strategy_type":"SENTIMENT","confidence":0.4,'
                        '"rationale":"neutral","risk_level":"LOW","expected_direction":"SIDEWAYS",'
                        '"provider":"ollama_qwen3_8b"}'
                    )
                },
                "done": True,
            }
        )
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.__exit__.return_value = False
        mock_client.post.return_value = mock_response

        with patch("app.llm_client.httpx.Client", return_value=mock_client):
            parsed = generate_decision("prompt", {"symbol": "X"})

        self.assertEqual(parsed["action"], "HOLD")
        mock_client.post.assert_called_once()
        payload = mock_client.post.call_args[1]["json"]
        self.assertIs(payload["think"], False)
        self.assertEqual(payload["format"], "json")


if __name__ == "__main__":
    unittest.main()
