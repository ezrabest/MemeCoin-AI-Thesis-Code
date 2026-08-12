"""Response / artifact schemas for AE12 reporting (typed dicts, no hard-coded results)."""

from __future__ import annotations

from typing import Any, TypedDict


class FileLoadResult(TypedDict, total=False):
    status: str  # OK | MISSING | ERROR
    path: str | None
    missing_file: str | None
    error: str | None
    data: Any


class Ae12Roots(TypedDict, total=False):
    maturation_root: str | None
    census_root: str | None
    quality_root: str | None
    maturity_status: str
    census_status: str
    quality_status: str


SAFETY_DISCLAIMERS: tuple[str, ...] = (
    "forward returns are outcome labels only",
    "paper/demo exploration is not live-trading approval",
    "Qwen/Gemini/Ollama are audit/explanation layers, not trade authority",
    "strict policy approved zero candidates in this AE12 evidence set",
    "future work includes strict policy calibration, runtime UI hardening, "
    "longer forward validation, and optional live-wallet gate only after separate approval",
)

REQUIRED_REPORT_LIMITATION_PHRASES: tuple[str, ...] = (
    "forward returns are outcome labels only",
    "paper/demo exploration is not live-trading approval",
    "Qwen/Gemini/Ollama are audit/explanation layers, not trade authority",
    "strict policy approved zero candidates in this AE12 evidence set",
    "not live-approved",
    "not profitability-proven",
)
