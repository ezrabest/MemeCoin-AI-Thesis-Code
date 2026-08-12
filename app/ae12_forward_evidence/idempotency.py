"""Deterministic IDs and resume/idempotency helpers for AE12."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def stable_hash(parts: Iterable[Any]) -> str:
    payload = "|".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_evidence_row_id(
    *,
    source_file: str,
    source_line_no: int,
    candidate_id: str | None,
    decision_id: str | None,
    pair_address: str | None,
    first_seen_timestamp: str | None,
) -> str:
    return stable_hash(
        [
            source_file,
            source_line_no,
            candidate_id,
            decision_id,
            pair_address,
            first_seen_timestamp,
        ]
    )


def make_horizon_row_id(*, evidence_row_id: str, horizon: str) -> str:
    return stable_hash([evidence_row_id, horizon])


def load_processed_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.is_file():
        return keys
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # tolerate bare key lines
                keys.add(line)
                continue
            if isinstance(obj, dict):
                key = obj.get("key") or obj.get("evidence_row_id") or obj.get("horizon_row_id")
                if key:
                    keys.add(str(key))
            elif isinstance(obj, str):
                keys.add(obj)
    return keys


def append_processed_keys(path: Path, keys: Iterable[str], *, key_field: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for key in keys:
            f.write(json.dumps({key_field: key}, separators=(",", ":")) + "\n")


def write_run_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, default=str) + "\n", encoding="utf-8")


def load_run_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


class IdempotencyGuard:
    """In-run + cross-run dedupe by evidence_row_id / horizon_row_id."""

    def __init__(
        self,
        *,
        processed_evidence: set[str] | None = None,
        processed_horizons: set[str] | None = None,
    ) -> None:
        self.processed_evidence = set(processed_evidence or ())
        self.processed_horizons = set(processed_horizons or ())
        self.seen_evidence_this_run: set[str] = set()
        self.seen_horizons_this_run: set[str] = set()
        self.skipped_duplicate_evidence = 0
        self.skipped_duplicate_horizons = 0
        self.new_evidence_keys: list[str] = []
        self.new_horizon_keys: list[str] = []

    def accept_evidence(self, evidence_row_id: str) -> bool:
        if evidence_row_id in self.processed_evidence or evidence_row_id in self.seen_evidence_this_run:
            self.skipped_duplicate_evidence += 1
            return False
        self.seen_evidence_this_run.add(evidence_row_id)
        self.new_evidence_keys.append(evidence_row_id)
        return True

    def accept_horizon(self, horizon_row_id: str) -> bool:
        if horizon_row_id in self.processed_horizons or horizon_row_id in self.seen_horizons_this_run:
            self.skipped_duplicate_horizons += 1
            return False
        self.seen_horizons_this_run.add(horizon_row_id)
        self.new_horizon_keys.append(horizon_row_id)
        return True

    def to_audit_row(self) -> dict[str, Any]:
        return {
            "skipped_duplicate_evidence": self.skipped_duplicate_evidence,
            "skipped_duplicate_horizons": self.skipped_duplicate_horizons,
            "new_evidence_count": len(self.new_evidence_keys),
            "new_horizon_count": len(self.new_horizon_keys),
            "prior_processed_evidence_count": len(self.processed_evidence),
            "prior_processed_horizon_count": len(self.processed_horizons),
            "idempotency_status": "PASS",
        }
