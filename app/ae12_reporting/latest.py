"""Discover latest AE12 audit output roots under data/audits (read-only)."""

from __future__ import annotations

import re
from pathlib import Path


MATURATION_PREFIX = "ae12_forward_evidence_maturation_"
CENSUS_PREFIX = "ae12_runtime_data_census_"
QUALITY_PREFIX = "ae12_forward_evidence_quality_"
TAXONOMY_PREFIX = "ae12_signal_taxonomy_audit_"
SENTIMENTFIX_PREFIX = "ae12_sentimentfix_"
SEMANTIC_CLASSIFIER_PREFIX = "ae12_semantic_coin_classifier_"
GEMINI_ADJUDICATION_PREFIX = "ae12_gemini_semantic_adjudication_"
MANUAL_REVIEW_DRILLDOWN_PREFIX = "ae12_sentimentfix_manual_review_drilldown_"

# Strict timestamped audit roots only (excludes ae12_sentimentfix_state).
SENTIMENTFIX_ROOT_RE = re.compile(r"^ae12_sentimentfix_\d{8}_\d{6}$")
SENTIMENTFIX_SUMMARY = Path("reports") / "ae12_sentimentfix_summary.json"


def audits_dir(project_root: Path) -> Path:
    return Path(project_root) / "data" / "audits"


def list_roots_by_prefix(project_root: Path, prefix: str) -> list[Path]:
    """Return matching directories sorted newest-first by name (timestamp suffix)."""
    base = audits_dir(project_root)
    if not base.is_dir():
        return []
    roots = [p for p in base.iterdir() if p.is_dir() and p.name.startswith(prefix)]
    roots.sort(key=lambda p: p.name, reverse=True)
    return roots


def discover_latest_root(project_root: Path, prefix: str) -> Path | None:
    roots = list_roots_by_prefix(project_root, prefix)
    return roots[0] if roots else None


def discover_latest_maturation_root(project_root: Path) -> Path | None:
    return discover_latest_root(project_root, MATURATION_PREFIX)


def discover_latest_census_root(project_root: Path) -> Path | None:
    return discover_latest_root(project_root, CENSUS_PREFIX)


def discover_latest_quality_root(project_root: Path) -> Path | None:
    return discover_latest_root(project_root, QUALITY_PREFIX)


def discover_latest_taxonomy_root(project_root: Path) -> Path | None:
    return discover_latest_root(project_root, TAXONOMY_PREFIX)


def is_valid_sentimentfix_root(path: Path) -> bool:
    """Strict filter for SentimentFix audit roots (not state/cache/partial)."""
    if not path.is_dir():
        return False
    name = path.name
    if not SENTIMENTFIX_ROOT_RE.match(name):
        return False
    if name.endswith("_state"):
        return False
    if "state" in name.lower():
        return False
    if "cache" in name.lower():
        return False
    if "partial" in name.lower():
        return False
    return (path / SENTIMENTFIX_SUMMARY).is_file()


def discover_latest_sentimentfix_root(project_root: Path) -> Path | None:
    base = audits_dir(project_root)
    if not base.is_dir():
        return None
    roots = [p for p in base.iterdir() if is_valid_sentimentfix_root(p)]
    roots.sort(key=lambda p: p.name, reverse=True)
    return roots[0] if roots else None


def discover_latest_semantic_classifier_root(project_root: Path) -> Path | None:
    return discover_latest_root(project_root, SEMANTIC_CLASSIFIER_PREFIX)


def discover_latest_gemini_adjudication_root(project_root: Path) -> Path | None:
    return discover_latest_root(project_root, GEMINI_ADJUDICATION_PREFIX)


def discover_latest_manual_review_drilldown_root(project_root: Path) -> Path | None:
    return discover_latest_root(project_root, MANUAL_REVIEW_DRILLDOWN_PREFIX)


def discover_all_latest_roots(project_root: Path) -> dict[str, Path | None]:
    return {
        "maturation_root": discover_latest_maturation_root(project_root),
        "census_root": discover_latest_census_root(project_root),
        "quality_root": discover_latest_quality_root(project_root),
        "taxonomy_root": discover_latest_taxonomy_root(project_root),
        "sentimentfix_root": discover_latest_sentimentfix_root(project_root),
        "semantic_classifier_root": discover_latest_semantic_classifier_root(project_root),
        "gemini_adjudication_root": discover_latest_gemini_adjudication_root(project_root),
        "manual_review_drilldown_root": discover_latest_manual_review_drilldown_root(project_root),
    }
