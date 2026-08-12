"""CSV/JSON/text report writers for AE12 audit outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.ae12_forward_evidence.types import utc_now_iso


def ensure_dirs(output_root: Path) -> dict[str, Path]:
    dirs = {
        "root": output_root,
        "reports": output_root / "reports",
        "data": output_root / "data",
        "audits": output_root / "audits",
        "state": output_root / "state",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        cols = fieldnames or []
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=cols)
            writer.writeheader()
        return
    cols = fieldnames or list(rows[0].keys())
    # union keys to avoid silent drops
    all_keys: list[str] = list(cols)
    seen = set(all_keys)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            flat = {k: _csv_val(r.get(k)) for k in all_keys}
            writer.writerow(flat)


def append_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Append rows to an existing CSV, writing header only if file is new/empty."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file() and path.stat().st_size > 0
    if not exists:
        write_csv(path, rows)
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        existing_fields = list(reader.fieldnames or [])
    all_keys = list(existing_fields)
    seen = set(all_keys)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                all_keys.append(k)
                seen.add(k)
    # If schema grew, rewrite via read+write to keep a single header
    if all_keys != existing_fields:
        prior: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as f:
            prior.extend(csv.DictReader(f))
        write_csv(path, prior + rows, fieldnames=all_keys)
        return
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        for r in rows:
            writer.writerow({k: _csv_val(r.get(k)) for k in all_keys})


def count_csv_rows(path: Path) -> int:
    if not path.is_file() or path.stat().st_size == 0:
        return 0
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        next(reader, None)
        return sum(1 for _ in reader)


def _csv_val(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, (list, dict)):
        return json.dumps(v, default=str, separators=(",", ":"))
    return v


def write_upload_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# AE12.3–AE12.4 Forward Evidence Maturation",
        "",
        f"created_at_utc: {summary.get('created_at_utc') or utc_now_iso()}",
        f"output_root: {summary.get('output_root')}",
        f"phase: {summary.get('phase')}",
        f"schema_version: {summary.get('schema_version')}",
        "",
        "## Counts",
        f"- candidate_evidence_rows: {summary.get('candidate_evidence_row_count')}",
        f"- matured_outcome_rows: {summary.get('matured_outcome_row_count')}",
        f"- missed_winners: {summary.get('missed_winner_count')}",
        f"- missing_reason_rows: {summary.get('missing_reason_count')}",
        f"- missing_data_warnings: {summary.get('missing_data_warning_count')}",
        "",
        "## Reason recovery",
        json.dumps(summary.get("reason_recovery_counts") or {}, indent=2),
        "",
        "## Horizon maturity",
        json.dumps(summary.get("horizon_maturity") or {}, indent=2),
        "",
        "## Missed winners by horizon",
        json.dumps(summary.get("missed_winners_by_horizon") or {}, indent=2),
        "",
        "## Trade vs no-trade (interpretation)",
        json.dumps(summary.get("trade_vs_no_trade_interpretations") or {}, indent=2),
        "",
        "## Strict vs exploration",
        json.dumps(summary.get("strict_vs_exploration") or {}, indent=2),
        "",
        "## Qwen/Ollama linkage",
        json.dumps(summary.get("qwen_linkage_counts") or {}, indent=2),
        "",
        "## Manual linkage sanity sample (up to 10)",
        json.dumps(summary.get("qwen_linkage_sanity_sample") or [], indent=2),
        "",
        "## Wallet safety",
        json.dumps(summary.get("wallet_safety") or {}, indent=2),
        "",
        "## Idempotency",
        json.dumps(summary.get("idempotency") or {}, indent=2),
        "",
        "## Final readiness gate",
        json.dumps(summary.get("readiness_gate") or {}, indent=2),
        "",
        "## Known limitations",
    ]
    for lim in summary.get("known_limitations") or []:
        lines.append(f"- {lim}")
    lines.extend(
        [
            "",
            "## Safety statements",
            "- Does not claim profitability.",
            "- Does not claim live readiness.",
            "- Does not claim Qwen made safe decisions unless row-level linkage proves it.",
            "- Forward returns are outcome labels only; never used as entry features.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
