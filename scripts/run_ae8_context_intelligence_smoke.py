#!/usr/bin/env python3
"""AE8 context intelligence smoke script."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.context_intelligence.context_feature_builder import run_ae8_context_intelligence  # noqa: E402
from scripts.diagnostics._common import open_db_readonly  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="AE8 context intelligence smoke")
    parser.add_argument("--max-records", type=int, default=50)
    parser.add_argument("--lookback-hours", type=float, default=24.0)
    parser.add_argument("--audit-only", action="store_true", default=True)
    parser.add_argument("--no-db-write", action="store_true", default=True)
    parser.add_argument("--allow-external-fetch", action="store_true", default=False)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--freshness-mode",
        choices=["live", "historical-replay"],
        default="live",
    )
    args = parser.parse_args()

    conn = open_db_readonly()
    try:
        summary = run_ae8_context_intelligence(
            project_root=ROOT,
            conn=conn,
            max_records=args.max_records,
            lookback_hours=args.lookback_hours,
            output_root=args.output_root,
            audit_only=args.audit_only,
            no_db_write=args.no_db_write,
            allow_external_fetch=args.allow_external_fetch,
            freshness_mode=args.freshness_mode,
        )
        compact = {
            "phase": summary.get("phase"),
            "final_status": summary.get("final_status"),
            "context_records_created": summary.get("context_records_created"),
            "context_schema_id": summary.get("context_schema_id"),
            "missing_context_families": summary.get("missing_context_families"),
            "memory_safety_status": (summary.get("memory_safety_summary") or {}).get(
                "memory_safety_status"
            ),
            "external_fetch_enabled": (summary.get("external_call_status") or {}).get(
                "external_fetch_enabled"
            ),
            "audit_dir": summary.get("audit_dir"),
            "jsonl_path": summary.get("jsonl_path"),
        }
        print(json.dumps(compact, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
