"""
Load .env on startup and apply LLM run-mode presets (headless / ollama / gemini).
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

log = logging.getLogger("env_bootstrap")

LLM_MODE_ENV: dict[str, dict[str, str]] = {
    "headless": {
        "HEADLESS_DATA_COLLECTION": "true",
        "LLM_PROVIDER": "none",
        "ENABLE_GEMINI": "false",
    },
    "ollama": {
        "HEADLESS_DATA_COLLECTION": "false",
        "LLM_PROVIDER": "ollama",
        "ENABLE_GEMINI": "false",
        "OLLAMA_BASE_URL": "http://localhost:11434/v1",
        "OLLAMA_MODEL": "qwen3:8b",
        "OLLAMA_MAX_CALLS_PER_SCAN": "5",
        "OLLAMA_TIMEOUT_SECONDS": "60",
    },
    "gemini": {
        "HEADLESS_DATA_COLLECTION": "false",
        "LLM_PROVIDER": "gemini",
        "ENABLE_GEMINI": "true",
    },
}


def parse_cli_mode(argv: list[str] | None = None) -> str | None:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="MemeCoin AI Trader — paper/demo workstation",
        add_help=True,
    )
    parser.add_argument(
        "--mode",
        choices=tuple(LLM_MODE_ENV),
        default=None,
        help="LLM run mode preset: headless | ollama | gemini (overrides .env for this process)",
    )
    parser.add_argument(
        "--ollama",
        action="store_true",
        help="Alias for --mode ollama",
    )
    parser.add_argument(
        "--gemini",
        action="store_true",
        help="Alias for --mode gemini",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Alias for --mode headless",
    )
    args, remainder = parser.parse_known_args(argv)
    # Reject unknown mode-like flags so users are not silently ignored
    unknown_mode_flags = [
        a for a in remainder
        if a.startswith("--mode") or a in ("--qwen", "--chatgpt", "--openai")
    ]
    if unknown_mode_flags:
        parser.error(
            f"Unsupported flag(s): {', '.join(unknown_mode_flags)}. "
            "Use: python main.py [--mode headless|ollama|gemini] or --ollama / --gemini / --headless"
        )
    aliases = [name for name, on in (("ollama", args.ollama), ("gemini", args.gemini), ("headless", args.headless)) if on]
    if args.mode and aliases:
        parser.error("Use either --mode <name> or a single alias (--ollama / --gemini / --headless), not both.")
    if len(aliases) > 1:
        parser.error("Use only one of --ollama, --gemini, --headless.")
    if aliases:
        return aliases[0]
    return args.mode


def load_environment(project_root: Path) -> None:
    """Load .env without overwriting variables already set in the shell."""
    env_path = project_root / ".env"
    if env_path.is_file():
        load_dotenv(env_path, override=False)
    else:
        load_dotenv(override=False)


def apply_llm_mode(mode: str) -> None:
    """Apply a run-mode preset, overriding .env values for this process."""
    preset = LLM_MODE_ENV.get(mode)
    if not preset:
        raise ValueError(f"Unknown LLM mode: {mode}")
    for key, value in preset.items():
        os.environ[key] = value


def resolve_active_mode_label() -> str:
    from .llm_config import get_llm_provider, is_headless_data_collection

    if is_headless_data_collection():
        return "headless"
    provider = get_llm_provider()
    if provider in LLM_MODE_ENV:
        return provider
    return provider or "custom"


def bootstrap_environment(project_root: Path, argv: list[str] | None = None) -> str | None:
    """Parse CLI mode, load .env, apply mode override. Returns CLI mode if set."""
    mode = parse_cli_mode(argv)
    load_environment(project_root)
    if mode:
        apply_llm_mode(mode)
    return mode


def log_llm_startup_config(cli_mode: str | None = None) -> None:
    """Print active LLM configuration and reset in-process call counters."""
    from .llm_config import (
        get_llm_provider,
        get_llm_runtime_status,
        get_ollama_model,
        is_gemini_env_enabled,
        is_headless_data_collection,
        reset_llm_counters,
        reset_ollama_scan_budget,
    )

    reset_llm_counters()
    reset_ollama_scan_budget()

    active = cli_mode or resolve_active_mode_label()
    runtime = get_llm_runtime_status()
    lines: list[tuple[str, Any]] = [
        ("active_mode", active),
        ("HEADLESS_DATA_COLLECTION", str(is_headless_data_collection()).lower()),
        ("LLM_PROVIDER", get_llm_provider()),
        ("ENABLE_GEMINI", str(is_gemini_env_enabled()).lower()),
    ]
    if get_llm_provider() == "ollama":
        lines.append(("OLLAMA_MODEL", get_ollama_model()))
    lines.extend([
        ("gemini_call_count", runtime["gemini_call_count"]),
        ("ollama_call_count", runtime["ollama_call_count"]),
    ])

    log.info("LLM startup configuration:")
    for key, value in lines:
        log.info("  %s=%s", key, value)
