"""Compatibility wrapper for semantic classifier execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .classification_reports import run_semantic_coin_classifier


def run_llm_semantic_coin_classifier(**kwargs: Any) -> dict[str, Any]:
    return run_semantic_coin_classifier(**kwargs)
