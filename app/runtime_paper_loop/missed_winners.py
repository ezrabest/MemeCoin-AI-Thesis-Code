"""Missed winners audit — post-hoc labeling of blocked/skipped candidates."""

from __future__ import annotations

from typing import Any

from app.runtime_paper_loop.opportunity_capture import (
    MISSED_WINNER_THRESHOLD,
    build_missed_winner_record,
    is_missed_winner,
)
from app.runtime_paper_loop.types import OpportunityCaptureRecord


def update_missed_winners(
    capture_records: list[OpportunityCaptureRecord],
    *,
    threshold: float = MISSED_WINNER_THRESHOLD,
) -> list[dict[str, Any]]:
    """Identify and build missed winner records from opportunity capture."""
    missed: list[dict[str, Any]] = []
    for rec in capture_records:
        if is_missed_winner(rec, threshold=threshold):
            missed.append(build_missed_winner_record(rec))
    return missed
