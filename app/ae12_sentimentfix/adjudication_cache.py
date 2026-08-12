"""Freeze-once adjudication cache for AE12-SentimentFix Gemini pass."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .adjudication_schema import ADJUDICATION_RUBRIC_VERSION, ADJUDICATOR_VERSION


def cache_key(*, asset_id: str, rubric_version: str, adjudicator_version: str) -> str:
    return "|".join([asset_id, rubric_version, adjudicator_version])


def cache_key_fields() -> list[str]:
    return ["asset_id", "rubric_version", "adjudicator_version"]


def state_cache_path(project_root: Path) -> Path:
    return project_root / "data" / "audits" / "ae12_sentimentfix_state" / "gemini_semantic_adjudication_cache.jsonl"


def output_cache_path(output_root: Path) -> Path:
    return output_root / "state" / "gemini_semantic_adjudication_cache.jsonl"


def load_cache(paths: list[Path]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in paths:
        if not path.is_file():
            continue
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


def append_cache_rows(output_root: Path, rows: list[dict[str, Any]], *, project_root: Path | None = None) -> None:
    paths = [output_cache_path(output_root)]
    if project_root is not None:
        paths.append(state_cache_path(project_root))
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, sort_keys=True))
                fh.write("\n")


def lookup_frozen_entry(
    *,
    asset_id: str,
    evidence_hash: str,
    cache: dict[str, dict[str, Any]],
    force_refresh: bool = False,
    rubric_version: str = ADJUDICATION_RUBRIC_VERSION,
    adjudicator_version: str = ADJUDICATOR_VERSION,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    key = cache_key(asset_id=asset_id, rubric_version=rubric_version, adjudicator_version=adjudicator_version)
    entry = cache.get(key)
    meta = {
        "cache_hit": False,
        "decision_frozen": False,
        "reclassification_allowed": False,
        "manual_override_allowed": True,
        "force_refresh_required_for_recheck": True,
        "evidence_hash_changed_since_classification": False,
        "stale_evidence_warning": False,
    }
    if not entry:
        return None, meta
    if force_refresh:
        return None, meta
    frozen = bool(entry.get("decision_frozen", True))
    meta["cache_hit"] = True
    meta["decision_frozen"] = frozen
    stored_hash = str(entry.get("evidence_hash_at_classification") or "")
    if stored_hash and evidence_hash and stored_hash != evidence_hash:
        meta["evidence_hash_changed_since_classification"] = True
        meta["stale_evidence_warning"] = True
    if frozen and not force_refresh:
        return entry, meta
    return None, meta


def build_cache_row(
    *,
    asset: dict[str, Any],
    adjudication: dict[str, Any],
    gemini_model: str,
    rubric_version: str = ADJUDICATION_RUBRIC_VERSION,
    adjudicator_version: str = ADJUDICATOR_VERSION,
) -> dict[str, Any]:
    asset_id = str(asset.get("asset_id") or "")
    key = cache_key(asset_id=asset_id, rubric_version=rubric_version, adjudicator_version=adjudicator_version)
    return {
        "cache_key": key,
        "asset_id": asset_id,
        "chain": asset.get("chain"),
        "token_address": asset.get("token_address"),
        "contract_address": asset.get("token_address"),
        "pair_address": asset.get("pair_address"),
        "symbol": asset.get("symbol"),
        "name": asset.get("name"),
        "semantic_coin_class": adjudication.get("semantic_coin_class"),
        "raw_evidence_status": adjudication.get("raw_evidence_status"),
        "classification_confidence": adjudication.get("classification_confidence"),
        "gemini_model": gemini_model,
        "adjudicator_version": adjudicator_version,
        "rubric_version": rubric_version,
        "classified_at_utc": adjudication.get("classified_at_utc"),
        "source_urls": adjudication.get("source_urls") or [],
        "evidence_hash_at_classification": asset.get("evidence_hash"),
        "decision_frozen": True,
        "reclassification_allowed": False,
        "manual_override_allowed": True,
        "force_refresh_required_for_recheck": True,
        "adjudication": adjudication,
    }
