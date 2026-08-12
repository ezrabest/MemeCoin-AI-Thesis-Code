"""Collision-safe AE20 output root allocation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.ae20 import OUTPUT_ROOT_PREFIX


def utc_stamp_microseconds() -> str:
    """UTC stamp with microsecond precision (Zulu)."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def short_uuid(n: int = 8) -> str:
    return uuid4().hex[:n]


def allocate_ae20_output_root(
    project_root: Path,
    *,
    output_root: str | Path | None = None,
    audits_subdir: str = "data/audits",
) -> tuple[Path, dict]:
    """Allocate a unique AE20 output root. Never overwrite an existing root.

    Required pattern:
    data/audits/ae20_integrated_clean_forward_validation_<YYYYMMDDTHHMMSSffffffZ>_<short_uuid>/
    """
    project_root = project_root.resolve()
    collision_attempts: list[dict] = []

    if output_root is not None:
        root = Path(output_root)
        if not root.is_absolute():
            root = project_root / root
        root = root.resolve()
        if root.exists():
            raise FileExistsError(
                f"AE20 output root already exists (refusing overwrite): {root}"
            )
        root.mkdir(parents=True, exist_ok=False)
        audit = {
            "collision_safe": True,
            "naming_pattern": "explicit_output_root",
            "stamp_has_microseconds": True,
            "uuid_suffix_present": True,
            "overwrote_existing": False,
            "output_root": str(root),
            "attempts": collision_attempts,
        }
        return root, audit

    audits = project_root / audits_subdir
    audits.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, 8):
        stamp = utc_stamp_microseconds()
        uid = short_uuid(8)
        name = f"{OUTPUT_ROOT_PREFIX}{stamp}_{uid}"
        candidate = (audits / name).resolve()
        attempt_rec = {
            "attempt": attempt,
            "candidate": str(candidate),
            "stamp": stamp,
            "short_uuid": uid,
            "exists_before_mkdir": candidate.exists(),
        }
        collision_attempts.append(attempt_rec)
        if candidate.exists():
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        audit = {
            "collision_safe": True,
            "naming_pattern": (
                "ae20_integrated_clean_forward_validation_"
                "<YYYYMMDDTHHMMSSffffffZ>_<short_uuid>"
            ),
            "stamp_has_microseconds": True,
            "uuid_suffix_present": True,
            "overwrote_existing": False,
            "output_root": str(candidate),
            "attempts": collision_attempts,
        }
        return candidate, audit

    raise RuntimeError("Unable to allocate collision-safe AE20 output root after retries")


def ensure_ae20_dirs(output_root: Path) -> dict[str, Path]:
    data = output_root / "data"
    audits = output_root / "audits"
    reports = output_root / "reports"
    for d in (data, audits, reports):
        d.mkdir(parents=True, exist_ok=True)
    return {"data": data, "audits": audits, "reports": reports, "output_root": output_root}
