"""AE18 last-good display cache — atomic writes + locking.

Display-only continuity when provider symbol resolution fails.
Never fabricates market data or bypasses freshness.
UI GET must only read via ``read_last_good_display_cache`` / ``lookup_last_good_display``.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.clean_forward.provider_url_key import (
    ProviderUrlKeyError,
    normalize_provider_pair_url_key,
    try_normalize_provider_pair_url_key,
)

log = logging.getLogger("ae18.last_good_display_cache")

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / "data" / "runtime"
AUDITS_DIR = ROOT / "data" / "audits"
JSONL_PATH = RUNTIME_DIR / "canonical_market_identity_last_good_display.jsonl"
CSV_PATH = RUNTIME_DIR / "canonical_market_identity_last_good_display.csv"
FAILURE_AUDIT_PATH = AUDITS_DIR / "ae18_last_good_display_cache_write_failure_audit.json"

_CACHE_LOCK = threading.Lock()
_GET_CACHE_WRITE_COUNT = 0  # instrumentation: must stay 0 on GET paths

CACHE_FIELDS = [
    "normalized_provider_pair_url_key",
    "provider_pair_url_exact",
    "symbol_pair_display",
    "provider_base_token_symbol",
    "provider_quote_token_symbol",
    "provider_base_token_address",
    "provider_quote_token_address",
    "provider_dex_id",
    "chain",
    "first_seen_at",
    "last_confirmed_at",
    "source",
    "source_audit",
    "confidence",
    "provenance_status",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def get_ui_get_cache_write_count() -> int:
    return _GET_CACHE_WRITE_COUNT


def reset_ui_get_cache_write_count() -> None:
    global _GET_CACHE_WRITE_COUNT
    _GET_CACHE_WRITE_COUNT = 0


def _note_get_write_attempt() -> None:
    """Instrumentation only — GET paths must never call write functions."""
    global _GET_CACHE_WRITE_COUNT
    _GET_CACHE_WRITE_COUNT += 1


def _guard_cache_write(detail: str) -> None:
    """Fail closed if a cache write is attempted from a UI GET path."""
    try:
        from app.runtime.ui_get_network_guard import assert_not_ui_get
    except Exception:
        return
    if _is_ui_get_active():
        _note_get_write_attempt()
    assert_not_ui_get("cache", detail=detail)


def _is_ui_get_active() -> bool:
    try:
        from app.runtime.ui_get_network_guard import is_ui_get_path_active

        return is_ui_get_path_active()
    except Exception:
        return False


def _read_jsonl_unlocked(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _validate_cache_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    problems: list[str] = []
    keys: set[str] = set()
    for i, row in enumerate(rows):
        key = _cell(row.get("normalized_provider_pair_url_key"))
        exact = _cell(row.get("provider_pair_url_exact"))
        display = _cell(row.get("symbol_pair_display"))
        if not key:
            problems.append(f"row_{i}:missing_normalized_key")
            continue
        if not exact:
            problems.append(f"row_{i}:missing_provider_pair_url_exact")
        if not display or display == "-" or "/" not in display:
            problems.append(f"row_{i}:invalid_symbol_pair_display")
        try:
            expected = normalize_provider_pair_url_key(exact, require_dexscreener=True)
            if expected != key:
                problems.append(f"row_{i}:normalized_key_conflict")
        except ProviderUrlKeyError as exc:
            problems.append(f"row_{i}:malformed_url:{exc.reason}")
        if key in keys:
            problems.append(f"row_{i}:duplicate_normalized_key")
        keys.add(key)
    return {"passed": not problems, "problems": problems, "row_count": len(rows)}


def read_last_good_display_cache(
    *,
    jsonl_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Safe read path — no writes. Suitable for UI GET."""
    path = jsonl_path or JSONL_PATH
    with _CACHE_LOCK:
        return _read_jsonl_unlocked(path)


_SNAPSHOT_LOCK = threading.Lock()
_snapshot_cache: dict[str, Any] = {"mtime_ns": None, "size": None, "by_key": {}}


def last_good_display_snapshot(
    *,
    jsonl_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Read-only mapping normalized_key → row, cached by file mtime/size.

    Avoids re-reading the cache file once per runtime row on GET paths.
    """
    path = jsonl_path or JSONL_PATH
    if jsonl_path is not None and path != JSONL_PATH:
        return {
            _cell(r.get("normalized_provider_pair_url_key")): r
            for r in read_last_good_display_cache(jsonl_path=path)
            if _cell(r.get("normalized_provider_pair_url_key"))
        }

    try:
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        signature = (None, None)

    with _SNAPSHOT_LOCK:
        if (
            _snapshot_cache["mtime_ns"] == signature[0]
            and _snapshot_cache["size"] == signature[1]
        ):
            return _snapshot_cache["by_key"]

    rows = read_last_good_display_cache(jsonl_path=path)
    by_key = {
        _cell(r.get("normalized_provider_pair_url_key")): r
        for r in rows
        if _cell(r.get("normalized_provider_pair_url_key"))
    }
    with _SNAPSHOT_LOCK:
        _snapshot_cache["mtime_ns"] = signature[0]
        _snapshot_cache["size"] = signature[1]
        _snapshot_cache["by_key"] = by_key
    return by_key


def lookup_last_good_display(
    provider_pair_url_exact: str,
    *,
    jsonl_path: Path | None = None,
) -> dict[str, Any] | None:
    """Lookup by normalized URL key. Read-only."""
    key, err = try_normalize_provider_pair_url_key(
        provider_pair_url_exact, require_dexscreener=True
    )
    if not key:
        return None
    row = last_good_display_snapshot(jsonl_path=jsonl_path).get(key)
    return dict(row) if row else None


def _write_failure_audit(payload: dict[str, Any]) -> None:
    if _is_ui_get_active():
        # Never write audits from a UI GET path.
        try:
            from app.runtime.ui_get_network_guard import record_write_attempt

            record_write_attempt("audit", detail="last_good_display_cache_failure")
        except Exception:
            pass
        return
    try:
        AUDITS_DIR.mkdir(parents=True, exist_ok=True)
        with open(FAILURE_AUDIT_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
    except OSError as exc:
        log.warning("failed to write last-good cache failure audit: %s", exc)


def write_last_good_display_cache(
    rows: list[dict[str, Any]],
    *,
    jsonl_path: Path | None = None,
    csv_path: Path | None = None,
) -> dict[str, Any]:
    """Atomic shared cache write: temp → validate → os.replace under lock."""
    _guard_cache_write("write_last_good_display_cache")
    with _CACHE_LOCK:
        return _write_last_good_display_cache_unlocked(
            rows, jsonl_path=jsonl_path, csv_path=csv_path
        )


def _write_last_good_display_cache_unlocked(
    rows: list[dict[str, Any]],
    *,
    jsonl_path: Path | None = None,
    csv_path: Path | None = None,
) -> dict[str, Any]:
    """Caller must hold ``_CACHE_LOCK``."""
    jsonl_path = jsonl_path or JSONL_PATH
    csv_path = csv_path or CSV_PATH
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "temp_write_used": True,
        "temp_validation_used": True,
        "os_replace_used": False,
        "cache_locking_supported": True,
        "temp_validation_passed": False,
        "final_jsonl_replaced": False,
        "final_csv_replaced": False,
        "row_count": 0,
        "problems": [],
        "passed": False,
        "previous_cache_preserved": True,
    }

    by_key: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _cell(row.get("normalized_provider_pair_url_key"))
        if not key:
            exact = _cell(row.get("provider_pair_url_exact"))
            key, _ = try_normalize_provider_pair_url_key(exact, require_dexscreener=True)
        if not key:
            continue
        cleaned = {field: row.get(field, "") for field in CACHE_FIELDS}
        cleaned["normalized_provider_pair_url_key"] = key
        by_key[key] = cleaned
    out_rows = list(by_key.values())

    tmp_dir = jsonl_path.parent
    fd_j, tmp_j = tempfile.mkstemp(prefix=".lgd_", suffix=".jsonl.tmp", dir=str(tmp_dir))
    fd_c, tmp_c = tempfile.mkstemp(prefix=".lgd_", suffix=".csv.tmp", dir=str(tmp_dir))
    os.close(fd_j)
    os.close(fd_c)
    tmp_jsonl = Path(tmp_j)
    tmp_csv = Path(tmp_c)
    try:
        with open(tmp_jsonl, "w", encoding="utf-8") as f:
            for row in out_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(tmp_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CACHE_FIELDS, extrasaction="ignore")
            writer.writeheader()
            for row in out_rows:
                writer.writerow(row)

        written = _read_jsonl_unlocked(tmp_jsonl)
        validation = _validate_cache_rows(written)
        report["problems"] = list(validation["problems"])
        if not validation["passed"]:
            report["temp_validation_passed"] = False
            report["passed"] = False
            _write_failure_audit(
                {
                    "event": "last_good_display_cache_temp_validation_failed",
                    "timestamp": _utc_now(),
                    "problems": report["problems"],
                    "previous_cache_preserved": True,
                }
            )
            return report

        report["temp_validation_passed"] = True
        os.replace(tmp_jsonl, jsonl_path)
        report["final_jsonl_replaced"] = True
        os.replace(tmp_csv, csv_path)
        report["final_csv_replaced"] = True
        report["os_replace_used"] = True
        report["row_count"] = len(written)
        report["passed"] = True
        report["previous_cache_preserved"] = False
        return report
    except Exception as exc:
        report["problems"].append(f"write_exception:{exc}")
        report["passed"] = False
        _write_failure_audit(
            {
                "event": "last_good_display_cache_write_failed",
                "timestamp": _utc_now(),
                "error": str(exc),
                "problems": report["problems"],
                "previous_cache_preserved": True,
            }
        )
        return report
    finally:
        for p in (tmp_jsonl, tmp_csv):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass


def upsert_last_good_display(
    entry: dict[str, Any],
    *,
    jsonl_path: Path | None = None,
    csv_path: Path | None = None,
) -> dict[str, Any]:
    """Merge one successful display resolution into the last-good cache.

    Read-modify-write is performed under a single lock to prevent lost updates.
    """
    _guard_cache_write("upsert_last_good_display")
    exact = _cell(entry.get("provider_pair_url_exact"))
    key, err = try_normalize_provider_pair_url_key(exact, require_dexscreener=True)
    if not key:
        return {"passed": False, "problems": [err or "normalize_failed"]}

    display = _cell(entry.get("symbol_pair_display"))
    if not display or display == "-" or "/" not in display:
        return {"passed": False, "problems": ["invalid_symbol_pair_display"]}

    now = _utc_now()
    jsonl_path = jsonl_path or JSONL_PATH
    csv_path = csv_path or CSV_PATH

    with _CACHE_LOCK:
        existing = _read_jsonl_unlocked(jsonl_path)
        prior = None
        for row in existing:
            if _cell(row.get("normalized_provider_pair_url_key")) == key:
                prior = row
                break

        record = {
            "normalized_provider_pair_url_key": key,
            "provider_pair_url_exact": exact,
            "symbol_pair_display": display,
            "provider_base_token_symbol": _cell(entry.get("provider_base_token_symbol")),
            "provider_quote_token_symbol": _cell(entry.get("provider_quote_token_symbol")),
            "provider_base_token_address": _cell(entry.get("provider_base_token_address")),
            "provider_quote_token_address": _cell(entry.get("provider_quote_token_address")),
            "provider_dex_id": _cell(entry.get("provider_dex_id")),
            "chain": _cell(entry.get("chain")),
            "first_seen_at": _cell(prior.get("first_seen_at") if prior else "") or now,
            "last_confirmed_at": now,
            "source": _cell(entry.get("source")) or "provider_resolution",
            "source_audit": _cell(entry.get("source_audit")) or "ae18_last_good_display",
            "confidence": _cell(entry.get("confidence")) or "HIGH",
            "provenance_status": _cell(entry.get("provenance_status")) or "PROVIDER_CONFIRMED",
        }

        merged = [
            r for r in existing if _cell(r.get("normalized_provider_pair_url_key")) != key
        ]
        merged.append(record)
        return _write_last_good_display_cache_unlocked(
            merged, jsonl_path=jsonl_path, csv_path=csv_path
        )


def ensure_empty_last_good_cache_files() -> None:
    """Create empty cache files if missing (no network)."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if not JSONL_PATH.exists():
        JSONL_PATH.write_text("", encoding="utf-8")
    if not CSV_PATH.exists():
        with open(CSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CACHE_FIELDS)
            writer.writeheader()
