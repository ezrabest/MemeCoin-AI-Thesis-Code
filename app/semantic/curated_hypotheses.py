"""Project-root path helpers and curated hypothesis counters (epistemic level 4).

Curated/user hypotheses never affect system-verified semantic verdict counts.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]

FLAG_USE_CURATED = "CLEAN_FORWARD_USE_CURATED_TARGETS"
FLAG_CURATED_PATH = "CLEAN_FORWARD_CURATED_TARGETS_PATH"
DEFAULT_CURATED_RELATIVE = "data/SeedTargets/clean_forward_curated_ready_targets_active.csv"

HYPOTHESIS_COLUMNS = (
    "user_seed_label",
    "user_hypothesis",
    "seed_label",
    "seed_collection",
    "category",
    "collection",
    "semantic_hint",
    "cluster_label",
)

SOCIAL_HYPOTHESIS_LABELS = frozenset(
    {
        "USER_SEED_REFI",
        "USER_SEED_COMMUNITY_DAO",
        "USER_SEED_SOCIALFI",
        "USER_SEED_FAN_TOKEN",
        "SOCIALLY_MOTIVATED",
        "SOCIAL_CONFIRMED",
        "SOCIAL",
        "SOCIAL?",
    }
)

OPPORTUNISTIC_HYPOTHESIS_LABELS = frozenset(
    {
        "USER_SEED_OPPORTUNISTIC",
        "OPPORTUNISTIC_SPECULATIVE",
        "OPPORTUNISTIC_CONFIRMED",
        "OPPORTUNISTIC",
    }
)


def resolve_project_path(
    path_value: str | None,
    *,
    project_root: Path | None = None,
) -> Path | None:
    """Resolve a path against PROJECT_ROOT when relative. Absolute paths used as-is."""
    raw = str(path_value or "").strip()
    if not raw:
        return None
    root = project_root or PROJECT_ROOT
    # Normalize Windows separators without depending on cwd
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def curated_targets_enabled(environ: dict[str, str] | None = None) -> bool:
    env = environ if environ is not None else os.environ
    raw = str(env.get(FLAG_USE_CURATED, "") or "").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def _configured_curated_path_raw(environ: dict[str, str] | None = None) -> str:
    env = environ if environ is not None else os.environ
    override = str(env.get(FLAG_CURATED_PATH, "") or "").strip()
    if override:
        return override
    return DEFAULT_CURATED_RELATIVE


def _classify_hypothesis_label(raw: str) -> str:
    label = str(raw or "").strip().upper()
    if not label:
        return "unknown"
    if label in SOCIAL_HYPOTHESIS_LABELS:
        return "social"
    if label in OPPORTUNISTIC_HYPOTHESIS_LABELS:
        return "opportunistic"
    # Soft match for SOCIAL? variants already uppercased
    if label.replace(" ", "") == "SOCIAL?":
        return "social"
    return "unknown"


def _row_hypothesis_label(row: dict[str, str], headers: list[str]) -> str:
    header_set = {h.lower(): h for h in headers}
    for col in HYPOTHESIS_COLUMNS:
        key = header_set.get(col.lower())
        if not key:
            continue
        val = str(row.get(key) or "").strip()
        if val:
            return val
    return ""


def count_curated_hypotheses(
    *,
    project_root: Path | None = None,
    environ: dict[str, str] | None = None,
    path_override: str | Path | None = None,
) -> dict[str, Any]:
    """
    Count curated/user hypotheses only.

    Never increments system-verified social/opportunistic confirmed counts.
    Missing file → exists=false, counts=0, no exception.
    """
    root = project_root or PROJECT_ROOT
    env = environ if environ is not None else os.environ
    enabled = curated_targets_enabled(env)
    configured = str(path_override) if path_override is not None else _configured_curated_path_raw(env)
    resolved = resolve_project_path(configured, project_root=root)

    out: dict[str, Any] = {
        "curated_targets_enabled": enabled,
        "curated_targets_path": configured,
        "curated_targets_resolved_path": str(resolved) if resolved is not None else None,
        "curated_targets_file_exists": False,
        "curated_social_hypothesis_count": 0,
        "curated_opportunistic_hypothesis_count": 0,
        "curated_unknown_hypothesis_count": 0,
        "curated_total_hypothesis_count": 0,
        "curated_hypothesis_columns_used": [],
        "note": (
            "Curated/user hypotheses are provenance only and never confirm "
            "system_verified_social_count / system_verified_opportunistic_count."
        ),
    }

    if resolved is None:
        return out

    exists = resolved.is_file()
    out["curated_targets_file_exists"] = exists
    if not exists:
        return out

    # When flag is off, still report path/exists but do not treat as active hypothesis counts
    if not enabled:
        out["note"] = (
            "CLEAN_FORWARD_USE_CURATED_TARGETS is not enabled; "
            "hypothesis counts remain 0 (file may still exist)."
        )
        return out

    social = opp = unknown = 0
    columns_used: list[str] = []
    try:
        with open(resolved, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            headers = list(reader.fieldnames or [])
            columns_used = [c for c in HYPOTHESIS_COLUMNS if any(h.lower() == c.lower() for h in headers)]
            if not columns_used:
                # No recognizable hypothesis column — count rows as unknown total only
                n = sum(1 for _ in reader)
                out["curated_unknown_hypothesis_count"] = n
                out["curated_total_hypothesis_count"] = n
                out["curated_hypothesis_columns_used"] = []
                out["note"] = "No recognizable hypothesis column; rows counted as unknown hypotheses."
                return out
            for row in reader:
                label = _row_hypothesis_label(row, headers)
                kind = _classify_hypothesis_label(label)
                if kind == "social":
                    social += 1
                elif kind == "opportunistic":
                    opp += 1
                else:
                    unknown += 1
    except OSError:
        out["curated_targets_file_exists"] = False
        return out

    out["curated_social_hypothesis_count"] = social
    out["curated_opportunistic_hypothesis_count"] = opp
    out["curated_unknown_hypothesis_count"] = unknown
    out["curated_total_hypothesis_count"] = social + opp + unknown
    out["curated_hypothesis_columns_used"] = columns_used
    return out
