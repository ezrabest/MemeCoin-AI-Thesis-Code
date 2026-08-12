"""Authoritative semantic verdict persistence (JSONL — does not rewrite cluster_registry.json)."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
VERDICTS_PATH = DATA_DIR / "semantic_verdicts.jsonl"
_LOCK = threading.RLock()

VALID_SEMANTIC_STATUS = frozenset(
    {
        "SOCIAL_CONFIRMED",
        "OPPORTUNISTIC_CONFIRMED",
        "INSUFFICIENT_EVIDENCE",
        "CLASSIFICATION_FAILED",
    }
)
VALID_CLUSTER_LABEL = frozenset(
    {
        "SOCIALLY_MOTIVATED",
        "OPPORTUNISTIC_SPECULATIVE",
        "UNKNOWN",
    }
)


def _normalize_verdict(row: dict[str, Any]) -> dict[str, Any] | None:
    if not isinstance(row, dict):
        return None
    status = str(row.get("semantic_status") or "").strip().upper()
    if status and status not in VALID_SEMANTIC_STATUS:
        # Keep readable but do not crash counters on invalid historical rows
        row = dict(row)
        row["semantic_status_raw"] = row.get("semantic_status")
        row["semantic_status"] = "CLASSIFICATION_FAILED"
        status = "CLASSIFICATION_FAILED"
    label = str(row.get("cluster_label") or "UNKNOWN").strip().upper()
    if label not in VALID_CLUSTER_LABEL:
        row = dict(row)
        row["cluster_label_raw"] = row.get("cluster_label")
        row["cluster_label"] = "UNKNOWN"
    row.setdefault("no_trade_authority", True)
    row["no_trade_authority"] = True
    return row


def load_semantic_verdicts(path: Path | None = None) -> list[dict[str, Any]]:
    """Load all verdicts from JSONL (last write wins per identity_key)."""
    target = path or VERDICTS_PATH
    if not target.is_file():
        return []
    by_key: dict[str, dict[str, Any]] = {}
    ordered: list[dict[str, Any]] = []
    with _LOCK:
        try:
            text = target.read_text(encoding="utf-8")
        except OSError:
            return []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError:
            continue
        row = _normalize_verdict(raw)
        if not row:
            continue
        key = str(row.get("identity_key") or "").strip()
        if key:
            by_key[key] = row
        else:
            ordered.append(row)
    # Preserve insertion order of unique keys (last wins already applied)
    result = list(by_key.values()) + ordered
    return result


def persist_semantic_verdict(
    verdict: dict[str, Any],
    *,
    path: Path | None = None,
) -> dict[str, Any]:
    """Append one verdict to JSONL. Does not mutate cluster_registry.json or trade tables."""
    row = _normalize_verdict(dict(verdict)) or {}
    row["no_trade_authority"] = True
    target = path or VERDICTS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False, default=str)
    with _LOCK:
        with open(target, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    return row


def count_semantic_verdicts(path: Path | None = None) -> dict[str, int]:
    rows = load_semantic_verdicts(path=path)
    out = {
        "social_confirmed_count": 0,
        "opportunistic_confirmed_count": 0,
        "insufficient_evidence_count": 0,
        "classification_failed_count": 0,
        "total_semantic_verdicts": len(rows),
    }
    for r in rows:
        st = str(r.get("semantic_status") or "").upper()
        if st == "SOCIAL_CONFIRMED":
            out["social_confirmed_count"] += 1
        elif st == "OPPORTUNISTIC_CONFIRMED":
            out["opportunistic_confirmed_count"] += 1
        elif st == "INSUFFICIENT_EVIDENCE":
            out["insufficient_evidence_count"] += 1
        elif st == "CLASSIFICATION_FAILED":
            out["classification_failed_count"] += 1
        else:
            out["classification_failed_count"] += 1
    return out
