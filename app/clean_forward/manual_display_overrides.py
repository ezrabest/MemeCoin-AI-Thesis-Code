"""AE18 manual display overrides — strict pre-flight validation.

Overrides are display-only. They never fabricate market data or change
canonical identity (provider_pair_url_exact).
"""
from __future__ import annotations

import csv
import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.clean_forward.provider_url_key import (
    ProviderUrlKeyError,
    normalize_provider_pair_url_key,
    try_normalize_provider_pair_url_key,
)

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_DIR = ROOT / "data" / "runtime"
AUDITS_DIR = ROOT / "data" / "audits"
OVERRIDE_CSV_PATH = RUNTIME_DIR / "manual_display_overrides.csv"
VALIDATION_AUDIT_PATH = AUDITS_DIR / "ae18_manual_display_overrides_validation_audit.json"

OVERRIDE_COLUMNS = [
    "provider_pair_url_exact",
    "normalized_provider_pair_url_key",
    "symbol_pair_display",
    "provider_base_token_symbol",
    "provider_quote_token_symbol",
    "reason",
    "reviewed_by",
    "reviewed_at",
    "source_note",
]

_EVM_ADDR_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
# Solana/base58-ish: long alphanumeric without punctuation (typical mint length 32-44)
_SOLANA_ADDR_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")
_ADDR_IN_TEXT_RE = re.compile(
    r"(0x[a-fA-F0-9]{40})|([1-9A-HJ-NP-Za-km-z]{32,44})"
)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_address_like(value: Any) -> bool:
    text = _cell(value)
    if not text:
        return False
    if _EVM_ADDR_RE.match(text):
        return True
    if _SOLANA_ADDR_RE.match(text) and not "/" in text:
        # Exclude short ticker-like strings that happen to be base58 charset
        # by requiring length >= 32 (already in regex).
        return True
    return False


def display_contains_raw_address(display: Any) -> bool:
    text = _cell(display)
    if not text:
        return False
    for part in text.split("/"):
        if is_address_like(part.strip()):
            return True
    return bool(_ADDR_IN_TEXT_RE.search(text) and "/" in text and any(
        is_address_like(p.strip()) for p in text.split("/")
    ))


def is_base_only_display(display: Any) -> bool:
    text = _cell(display)
    if not text or text == "-":
        return False
    if "/" in text:
        parts = [p.strip() for p in text.split("/") if p.strip()]
        return len(parts) == 1
    return True  # no slash → base-only / non-pair


def ensure_override_csv_template() -> Path:
    """Create empty overrides CSV with header if missing."""
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    if not OVERRIDE_CSV_PATH.exists():
        with open(OVERRIDE_CSV_PATH, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=OVERRIDE_COLUMNS)
            writer.writeheader()
    return OVERRIDE_CSV_PATH


def validate_override_row(row: dict[str, Any], *, row_number: int) -> list[str]:
    """Return list of rejection reasons (empty = valid)."""
    reasons: list[str] = []
    exact = _cell(row.get("provider_pair_url_exact"))
    key_field = _cell(row.get("normalized_provider_pair_url_key"))
    display = _cell(row.get("symbol_pair_display"))
    base_sym = _cell(row.get("provider_base_token_symbol"))
    quote_sym = _cell(row.get("provider_quote_token_symbol"))
    reviewed_by = _cell(row.get("reviewed_by"))
    reviewed_at = _cell(row.get("reviewed_at"))

    if not exact:
        reasons.append("provider_pair_url_exact_empty")
    else:
        try:
            expected = normalize_provider_pair_url_key(exact, require_dexscreener=True)
        except ProviderUrlKeyError as exc:
            reasons.append(f"malformed_provider_url:{exc.reason}")
            expected = None
        if expected is not None:
            if key_field and key_field != expected:
                reasons.append("normalized_key_conflicts_with_provider_pair_url_exact")
            elif not key_field:
                # Allow missing key in CSV — will be filled on apply — but still validate.
                pass

    if not display:
        reasons.append("symbol_pair_display_empty")
    elif display == "-":
        reasons.append("symbol_pair_display_dash")
    elif "/" not in display:
        reasons.append("symbol_pair_display_missing_separator")
    elif is_base_only_display(display):
        reasons.append("symbol_pair_display_base_only")
    if display and display_contains_raw_address(display):
        reasons.append("symbol_pair_display_raw_address")

    if base_sym and is_address_like(base_sym):
        reasons.append("provider_base_token_symbol_address_like")
    if quote_sym and is_address_like(quote_sym):
        reasons.append("provider_quote_token_symbol_address_like")

    if not reviewed_by:
        reasons.append("reviewed_by_missing")
    if not reviewed_at:
        reasons.append("reviewed_at_missing")

    return reasons


def validate_manual_display_overrides(
    *,
    csv_path: Path | None = None,
    audit_path: Path | None = None,
    apply: bool = False,
) -> dict[str, Any]:
    """Validate override CSV. Invalid rows are never applied.

    Pass rule: passed=True only if every *applied* override row is valid.
    Invalid rows may exist only if rejected and not applied.
    """
    csv_path = csv_path or OVERRIDE_CSV_PATH
    audit_path = audit_path or VALIDATION_AUDIT_PATH

    audit: dict[str, Any] = {
        "override_file_exists": csv_path.exists(),
        "rows_checked": 0,
        "valid_rows": 0,
        "rejected_rows": 0,
        "duplicate_key_count": 0,
        "malformed_url_count": 0,
        "empty_display_count": 0,
        "missing_separator_count": 0,
        "base_only_display_count": 0,
        "raw_address_display_count": 0,
        "address_like_symbol_field_count": 0,
        "missing_review_metadata_count": 0,
        "rejected_row_details": [],
        "applied_override_count": 0,
        "valid_overrides": [],
        "passed": False,
        "fail_closed": True,
        "validated_at": _utc_now(),
    }

    if not csv_path.exists():
        ensure_override_csv_template()
        audit["override_file_exists"] = True
        audit["passed"] = True  # empty file: nothing applied, nothing invalid applied
        audit["fail_closed"] = False
        _write_audit(audit_path, audit)
        return audit

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    key_to_display: dict[str, str] = {}
    key_to_rownum: dict[str, int] = {}
    valid: list[dict[str, Any]] = []

    for idx, row in enumerate(rows, start=2):  # header is row 1
        audit["rows_checked"] += 1
        reasons = validate_override_row(row, row_number=idx)
        exact = _cell(row.get("provider_pair_url_exact"))
        key, norm_err = try_normalize_provider_pair_url_key(exact, require_dexscreener=True)
        display = _cell(row.get("symbol_pair_display"))

        # Duplicate conflicting key check (among candidates that normalize)
        if key:
            prev = key_to_display.get(key)
            if prev is not None and prev != display:
                reasons.append("duplicate_normalized_key_conflicting_display")
                audit["duplicate_key_count"] += 1
            elif prev is None:
                key_to_display[key] = display
                key_to_rownum[key] = idx

        # Count category buckets from reasons
        for r in reasons:
            if r.startswith("malformed_provider_url") or r == "provider_pair_url_exact_empty":
                audit["malformed_url_count"] += 1
            if r in {"symbol_pair_display_empty", "symbol_pair_display_dash"}:
                audit["empty_display_count"] += 1
            if r == "symbol_pair_display_missing_separator":
                audit["missing_separator_count"] += 1
            if r == "symbol_pair_display_base_only":
                audit["base_only_display_count"] += 1
            if r == "symbol_pair_display_raw_address":
                audit["raw_address_display_count"] += 1
            if "address_like" in r:
                audit["address_like_symbol_field_count"] += 1
            if r in {"reviewed_by_missing", "reviewed_at_missing"}:
                audit["missing_review_metadata_count"] += 1

        if reasons:
            audit["rejected_rows"] += 1
            audit["rejected_row_details"].append(
                {
                    "row_number": idx,
                    "provider_pair_url_exact": exact,
                    "reasons": reasons,
                }
            )
            continue

        # Fill normalized key if blank
        assert key is not None
        cleaned = {col: _cell(row.get(col)) for col in OVERRIDE_COLUMNS}
        cleaned["normalized_provider_pair_url_key"] = key
        cleaned["provider_pair_url_exact"] = exact
        audit["valid_rows"] += 1
        valid.append(cleaned)

    audit["valid_overrides"] = valid
    if apply:
        audit["applied_override_count"] = len(valid)
    else:
        audit["applied_override_count"] = 0

    # Pass: every applied row is valid (rejected rows are fine if not applied)
    audit["passed"] = True
    audit["fail_closed"] = False
    _write_audit(audit_path, audit)
    return audit


_OVERRIDE_SNAPSHOT_LOCK = threading.Lock()
_override_snapshot: dict[str, Any] = {"mtime_ns": None, "size": None, "by_key": {}}


def manual_overrides_snapshot(
    *,
    csv_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Read-only override mapping cached by file mtime/size (no audit writes)."""
    path = csv_path or OVERRIDE_CSV_PATH
    if csv_path is not None and path != OVERRIDE_CSV_PATH:
        return read_manual_overrides_readonly(csv_path=path)

    try:
        stat = path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
    except OSError:
        signature = (None, None)

    with _OVERRIDE_SNAPSHOT_LOCK:
        if (
            _override_snapshot["mtime_ns"] == signature[0]
            and _override_snapshot["size"] == signature[1]
        ):
            return _override_snapshot["by_key"]

    by_key = read_manual_overrides_readonly(csv_path=path)
    with _OVERRIDE_SNAPSHOT_LOCK:
        _override_snapshot["mtime_ns"] = signature[0]
        _override_snapshot["size"] = signature[1]
        _override_snapshot["by_key"] = by_key
    return by_key


def read_manual_overrides_readonly(
    *,
    csv_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Read valid overrides without writing audits (safe for UI GET)."""
    csv_path = csv_path or OVERRIDE_CSV_PATH
    if not csv_path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for idx, row in enumerate(rows, start=2):
        reasons = validate_override_row(row, row_number=idx)
        if reasons:
            continue
        exact = _cell(row.get("provider_pair_url_exact"))
        key, _ = try_normalize_provider_pair_url_key(exact, require_dexscreener=True)
        if not key:
            continue
        cleaned = {col: _cell(row.get(col)) for col in OVERRIDE_COLUMNS}
        cleaned["normalized_provider_pair_url_key"] = key
        # First valid wins; conflicting later rows skipped in validate path.
        if key not in out:
            out[key] = cleaned
    return out


def load_applied_manual_overrides(
    *,
    csv_path: Path | None = None,
    audit_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate then return mapping normalized_key → override row (valid only)."""
    result = validate_manual_display_overrides(
        csv_path=csv_path, audit_path=audit_path, apply=True
    )
    out: dict[str, dict[str, Any]] = {}
    for row in result.get("valid_overrides") or []:
        key = _cell(row.get("normalized_provider_pair_url_key"))
        if key:
            out[key] = row
    # Rewrite audit with applied count
    result["applied_override_count"] = len(out)
    _write_audit(audit_path or VALIDATION_AUDIT_PATH, result)
    return out


def lookup_manual_override(
    provider_pair_url_exact: str,
    *,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    key, _ = try_normalize_provider_pair_url_key(
        provider_pair_url_exact, require_dexscreener=True
    )
    if not key:
        return None
    # Default to cached read-only snapshot (no audit writes) for GET safety.
    mapping = overrides if overrides is not None else manual_overrides_snapshot()
    return mapping.get(key)


def _write_audit(path: Path, audit: dict[str, Any]) -> None:
    # Audit writes are forbidden on UI GET paths (fail closed).
    try:
        from app.runtime.ui_get_network_guard import assert_not_ui_get

        assert_not_ui_get("audit", detail="manual_display_overrides_validation")
    except ImportError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    # Drop bulky valid_overrides from persisted audit if huge; keep counts.
    payload = dict(audit)
    payload.pop("valid_overrides", None)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
