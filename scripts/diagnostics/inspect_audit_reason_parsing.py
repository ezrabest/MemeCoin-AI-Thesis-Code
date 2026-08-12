#!/usr/bin/env python3
"""Diagnostic 11 — audit reason parsing bug detection."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.diagnostics._common import (
    CHUNK_SIZE,
    DB_PATH,
    DiagnosticReport,
    fetch_json_url,
    open_db_readonly,
    parse_audit_reasons_field,
    utc_now,
)


def _broken_api_reason_counts(rows: list[dict]) -> dict[str, int]:
    """Reproduce production bug: iterate JSON string as characters."""
    counts: dict[str, int] = {}
    for r in rows:
        raw = r.get("audit_reasons") or r.get("audit_reasons_json") or []
        for reason in raw if isinstance(raw, list) else raw:
            if isinstance(reason, str) and len(reason) == 1:
                counts[reason] = counts.get(reason, 0) + 1
            elif isinstance(reason, str):
                counts[reason] = counts.get(reason, 0) + 1
    return counts


def run(*, minutes: int, output_dir: Path, api_base: str = "http://127.0.0.1:8000") -> DiagnosticReport:
    report = DiagnosticReport("inspect_audit_reason_parsing", output_dir)
    cutoff = (utc_now() - timedelta(minutes=minutes)).isoformat()
    rows: list[dict] = []

    if DB_PATH.is_file():
        conn = open_db_readonly()
        try:
            offset = 0
            while True:
                batch = conn.execute(
                    """
                    SELECT id, timestamp, audit_reasons_json, details_json, stage, symbol
                    FROM pipeline_audit
                    WHERE timestamp >= ?
                    ORDER BY id DESC
                    LIMIT ? OFFSET ?
                    """,
                    (cutoff, CHUNK_SIZE, offset),
                ).fetchall()
                if not batch:
                    break
                rows.extend(dict(r) for r in batch)
                offset += len(batch)
                if len(batch) < CHUNK_SIZE:
                    break
        finally:
            conn.close()
    else:
        report.add_limitation(f"Database missing: {DB_PATH}")

    api_audit = fetch_json_url(f"{api_base}/api/audit/recent?minutes={minutes}")
    api_reason_counts = (api_audit or {}).get("reason_counts") if api_audit else None

    samples_valid: list[dict] = []
    samples_invalid: list[dict] = []
    corrected = Counter()
    broken = Counter()

    for r in rows:
        raw = r.get("audit_reasons_json")
        reasons, kind = parse_audit_reasons_field(raw)
        for reason in reasons:
            corrected[reason] += 1
        if kind == "json_array_string":
            samples_invalid.append({"id": r.get("id"), "raw": raw, "parsed": reasons})
        elif reasons:
            samples_valid.append({"id": r.get("id"), "kind": kind, "reasons": reasons[:5]})
        if isinstance(raw, str) and raw.startswith("["):
            for ch in raw:
                if len(ch) == 1:
                    broken[ch] += 1

    production_bug = any(len(k) == 1 for k in (api_reason_counts or {}))
    char_keys = [k for k in (api_reason_counts or {}) if len(str(k)) == 1]

    status = "FAIL" if production_bug or char_keys else "PASS"
    if not rows:
        status = "WARN"
        report.add_limitation("No pipeline_audit rows in window")

    report.set_status(status)
    report.data.update({
        "minutes": minutes,
        "sample_raw_rows": rows[:10],
        "valid_reason_examples": samples_valid[:10],
        "invalid_json_string_examples": samples_invalid[:10],
        "api_reason_counts": api_reason_counts,
        "corrected_reason_counts": dict(corrected.most_common(50)),
        "character_iteration_simulation": dict(broken.most_common(20)),
        "likely_production_bug_location": "app/api.py get_recent_audit — iterates audit_reasons_json string without json.loads",
        "recommended_fix": "Use parse_audit_reasons_field() before counting; never iterate a JSON string directly.",
        "production_single_char_keys_in_api": char_keys,
    })
    report.write_json("audit_reason_parsing.json")
    report.write_md([
        f"- API single-char keys: {char_keys or 'none'}",
        f"- Corrected top reasons: {corrected.most_common(10)}",
        f"- Bug location: `{report.data['likely_production_bug_location']}`",
        f"- Fix: {report.data['recommended_fix']}",
    ], "audit_reason_parsing.md")
    report.write_csv(
        [{"reason": k, "count": v} for k, v in corrected.most_common()],
        "corrected_reason_counts.csv",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    report = run(minutes=args.minutes, output_dir=args.output_dir, api_base=args.api_base)
    print(f"Status: {report.data['status']}")
    return 0 if report.data["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
