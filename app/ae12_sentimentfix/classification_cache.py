"""JSONL cache for semantic coin classifications (derived output root only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def cache_key(
    *,
    asset_id: str,
    classifier_version: str,
    evidence_hash: str,
    rubric_version: str,
) -> str:
    return "|".join([asset_id, classifier_version, evidence_hash, rubric_version])


def cache_key_fields() -> list[str]:
    return ["asset_id", "classifier_version", "evidence_hash", "rubric_version"]


def cache_uses_evidence_hash() -> bool:
    return True


def cache_path(output_root: Path) -> Path:
    return output_root / "state" / "semantic_coin_classification_cache.jsonl"


def load_cache(output_root: Path) -> dict[str, dict[str, Any]]:
    path = cache_path(output_root)
    if not path.is_file():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = str(row.get("cache_key") or "")
            if key:
                out[key] = row
    return out


def append_cache_rows(output_root: Path, rows: list[dict[str, Any]]) -> None:
    path = cache_path(output_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True))
            fh.write("\n")
