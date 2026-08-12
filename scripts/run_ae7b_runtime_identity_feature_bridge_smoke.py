#!/usr/bin/env python3
"""AE7B runtime identity + feature matrix bridge smoke script."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.decision.runtime_feature_bridge import run_ae7b_bridge  # noqa: E402
from scripts.diagnostics._common import open_db_readonly  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="AE7B runtime identity feature bridge smoke")
    parser.add_argument("--max-records", type=int, default=50)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--no-db-write",
        action="store_true",
        help="No-op: AE7B never writes runtime feature rows to SQLite",
    )
    parser.add_argument("--lookback-hours", type=float, default=24.0)
    parser.add_argument("--parity-check", action="store_true", default=False)
    args = parser.parse_args()

    del args.no_db_write  # accepted for CLI compatibility; AE7B is JSONL-only

    conn = open_db_readonly()
    try:
        summary = run_ae7b_bridge(
            project_root=ROOT,
            conn=conn,
            max_records=args.max_records,
            lookback_hours=args.lookback_hours,
            output_root=args.output_root,
            audit_only=args.audit_only,
            parity_check=args.parity_check,
        )
        compact = {
            "phase": summary.get("phase"),
            "bridge_readiness_decision": summary.get("bridge_readiness_decision"),
            "bridge_records_created": summary.get("bridge_records_created"),
            "candidate_id_generated": summary.get("candidate_id_generated"),
            "feature_schema_id": summary.get("feature_schema_id"),
            "feature_parity_status": (summary.get("feature_parity") or {}).get(
                "feature_parity_status"
            ),
            "audit_dir": summary.get("audit_dir"),
        }
        print(json.dumps(compact, indent=2))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
