"""Evidence-grounded social / opportunistic semantic classification (audit-only)."""

from __future__ import annotations

from .social_opportunistic_classifier import (
    PROMPT_VERSION,
    classify_token_social_opportunistic,
    get_authoritative_semantic_counts,
)
from .semantic_registry import (
    VERDICTS_PATH,
    load_semantic_verdicts,
    persist_semantic_verdict,
    count_semantic_verdicts,
)
from .curated_hypotheses import count_curated_hypotheses, resolve_project_path

__all__ = [
    "PROMPT_VERSION",
    "VERDICTS_PATH",
    "classify_token_social_opportunistic",
    "get_authoritative_semantic_counts",
    "load_semantic_verdicts",
    "persist_semantic_verdict",
    "count_semantic_verdicts",
    "count_curated_hypotheses",
    "resolve_project_path",
]
