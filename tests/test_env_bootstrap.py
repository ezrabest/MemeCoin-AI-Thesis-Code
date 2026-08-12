"""LLM run-mode bootstrap and preset tests."""
from __future__ import annotations

import os
import unittest

from app.env_bootstrap import apply_llm_mode, parse_cli_mode, resolve_active_mode_label


class EnvBootstrapTests(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {key: os.environ.get(key) for key in (
            "HEADLESS_DATA_COLLECTION",
            "LLM_PROVIDER",
            "ENABLE_GEMINI",
            "OLLAMA_MODEL",
        )}

    def tearDown(self) -> None:
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    def test_parse_cli_mode(self) -> None:
        self.assertEqual(parse_cli_mode(["main.py", "--mode", "ollama"]), "ollama")
        self.assertIsNone(parse_cli_mode(["main.py"]))

    def test_headless_preset(self) -> None:
        apply_llm_mode("headless")
        self.assertEqual(os.environ["HEADLESS_DATA_COLLECTION"], "true")
        self.assertEqual(os.environ["LLM_PROVIDER"], "none")
        self.assertEqual(os.environ["ENABLE_GEMINI"], "false")
        self.assertEqual(resolve_active_mode_label(), "headless")

    def test_ollama_preset(self) -> None:
        apply_llm_mode("ollama")
        self.assertEqual(os.environ["LLM_PROVIDER"], "ollama")
        self.assertEqual(os.environ["OLLAMA_MODEL"], "qwen3:8b")
        self.assertEqual(resolve_active_mode_label(), "ollama")

    def test_gemini_preset(self) -> None:
        apply_llm_mode("gemini")
        self.assertEqual(os.environ["LLM_PROVIDER"], "gemini")
        self.assertEqual(os.environ["ENABLE_GEMINI"], "true")
        self.assertEqual(resolve_active_mode_label(), "gemini")


if __name__ == "__main__":
    unittest.main()
