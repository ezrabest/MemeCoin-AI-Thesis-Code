"""Discover AE14 audit roots and Clean Forward smoke/poll artifacts (local files only)."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

DEFAULT_AE14_ROOT_NAME = "ae14_real_clean_forward_closure_20260721_210220"
DEFAULT_SMOKE_ROOT_NAME = "clean_forward_smoke_2h_20260721_164202"
AE14_GLOB = "ae14_real_clean_forward_closure_*"
SMOKE_GLOB = "clean_forward_smoke_2h_*"
POLL_NAME_RE = re.compile(r"^poll_(\d+)_", re.IGNORECASE)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def discover_ae14_root(
    root: Path | None = None,
    *,
    explicit: Path | str | None = None,
) -> Path | None:
    """Prefer known AE14 root; else latest ae14_real_clean_forward_closure_*."""
    base = root or project_root()
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = base / p
        return p if p.exists() else None

    preferred = base / "data" / "audits" / DEFAULT_AE14_ROOT_NAME
    if preferred.exists():
        return preferred

    audits = base / "data" / "audits"
    if not audits.exists():
        return None
    candidates = sorted(
        [p for p in audits.glob(AE14_GLOB) if p.is_dir()],
        key=lambda p: p.name,
    )
    return candidates[-1] if candidates else None


def discover_clean_forward_smoke_root(
    root: Path | None = None,
    *,
    explicit: Path | str | None = None,
) -> Path | None:
    base = root or project_root()
    if explicit:
        p = Path(explicit)
        if not p.is_absolute():
            p = base / p
        return p if p.exists() else None

    preferred = base / "data" / DEFAULT_SMOKE_ROOT_NAME
    if preferred.exists():
        return preferred

    data_dir = base / "data"
    if not data_dir.exists():
        return None
    candidates = sorted(
        [p for p in data_dir.glob(SMOKE_GLOB) if p.is_dir()],
        key=lambda p: p.name,
    )
    return candidates[-1] if candidates else None


def load_json(path: Path) -> Any:
    # Smoke polls may be PowerShell-written with UTF-8 BOM.
    return json.loads(path.read_text(encoding="utf-8-sig"))


def iter_poll_files(smoke_root: Path) -> list[Path]:
    polls_dir = smoke_root / "polls"
    if not polls_dir.exists():
        return []
    files = [p for p in polls_dir.glob("poll_*.json") if p.is_file()]

    def sort_key(p: Path) -> tuple[int, str]:
        m = POLL_NAME_RE.match(p.name)
        idx = int(m.group(1)) if m else -1
        return (idx, p.name)

    return sorted(files, key=sort_key)


def poll_index_from_name(path: Path) -> int | None:
    m = POLL_NAME_RE.match(path.name)
    return int(m.group(1)) if m else None


def iter_poll_rows(
    smoke_root: Path,
    *,
    max_polls: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield row dicts with source poll metadata attached."""
    files = iter_poll_files(smoke_root)
    if max_polls is not None:
        files = files[: max(0, max_polls)]
    for poll_path in files:
        try:
            payload = load_json(poll_path)
        except (OSError, json.JSONDecodeError):
            continue
        rows = payload.get("rows") if isinstance(payload, dict) else None
        if not isinstance(rows, list):
            continue
        poll_idx = poll_index_from_name(poll_path)
        for row in rows:
            if not isinstance(row, dict):
                continue
            enriched = dict(row)
            enriched["_source_poll_file"] = str(poll_path).replace("\\", "/")
            enriched["_source_poll_index"] = poll_idx
            yield enriched


def load_ae14_artifacts(ae14_root: Path) -> dict[str, Any]:
    """Load key AE14 closure artifacts used for schema bridge / reconciliation."""
    data = ae14_root / "data"
    reports = ae14_root / "reports"
    out: dict[str, Any] = {
        "ae14_root": str(ae14_root).replace("\\", "/"),
        "audit": None,
        "selected_row": None,
        "paper_open_position": None,
        "demo_bot_run_once": None,
        "bridge_result": None,
        "gatekeeper_result": None,
        "execution_instrument": None,
        "feed_snapshot": None,
    }
    mapping = {
        "audit": reports / "ae14_real_clean_forward_closure_audit.json",
        "selected_row": data / "selected_clean_forward_row.json",
        "paper_open_position": data / "paper_open_position.json",
        "demo_bot_run_once": data / "demo_bot_run_once_response.json",
        "bridge_result": data / "bridge_result.json",
        "gatekeeper_result": data / "gatekeeper_result.json",
        "execution_instrument": data / "execution_instrument.json",
        "feed_snapshot": data / "clean_forward_market_feed_snapshot.json",
    }
    for key, path in mapping.items():
        if path.exists():
            try:
                out[key] = load_json(path)
            except (OSError, json.JSONDecodeError) as exc:
                out[key] = {"_load_error": str(exc), "_path": str(path)}
    return out


def source_artifact_index(
    *,
    ae14_root: Path | None,
    smoke_root: Path | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if ae14_root and ae14_root.exists():
        for path in sorted(ae14_root.rglob("*")):
            if path.is_file():
                rows.append(
                    {
                        "artifact_role": "ae14_closure",
                        "path": str(path).replace("\\", "/"),
                        "relative": str(path.relative_to(ae14_root)).replace("\\", "/"),
                        "size_bytes": path.stat().st_size,
                    }
                )
    if smoke_root and smoke_root.exists():
        # Index summary + a sample of poll files (not all 120 full listings bloating).
        for name in (
            "clean_forward_smoke_2h_summary.csv",
            "clean_forward_smoke_2h_summary_live.csv",
            "ae14_readiness_after_2h_smoke.json",
        ):
            p = smoke_root / name
            if p.exists():
                rows.append(
                    {
                        "artifact_role": "clean_forward_smoke",
                        "path": str(p).replace("\\", "/"),
                        "relative": name,
                        "size_bytes": p.stat().st_size,
                    }
                )
        polls = iter_poll_files(smoke_root)
        rows.append(
            {
                "artifact_role": "clean_forward_smoke_polls",
                "path": str((smoke_root / "polls")).replace("\\", "/"),
                "relative": "polls/",
                "size_bytes": None,
                "poll_file_count": len(polls),
                "first_poll": polls[0].name if polls else None,
                "last_poll": polls[-1].name if polls else None,
            }
        )
    return rows
