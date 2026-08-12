#!/usr/bin/env python3
"""AE7C-0 scoring policy feature enrichment smoke script."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.decision.ae7c0_feature_enrichment import run_ae7c0_enrichment  # noqa: E402
from scripts.diagnostics._common import open_db_readonly  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="AE7C-0 scoring policy feature enrichment smoke")
    parser.add_argument("--max-records", type=int, default=50)
    parser.add_argument("--lookback-hours", type=float, default=24.0)
    parser.add_argument("--audit-only", action="store_true", default=True)
    parser.add_argument(
        "--no-db-write",
        action="store_true",
        help="No-op: AE7C-0 never writes runtime feature rows to SQLite",
    )
    args = parser.parse_args()
    del args.no_db_write

    conn = open_db_readonly()
    try:
        summary = run_ae7c0_enrichment(
            project_root=ROOT,
            conn=conn,
            max_records=args.max_records,
            lookback_hours=args.lookback_hours,
            audit_only=args.audit_only,
        )
        compact = {
            "phase": summary.get("phase"),
            "feature_count_before": summary.get("feature_count_before"),
            "feature_count_after": summary.get("feature_count_after"),
            "compatible_before": summary.get("compatible_before"),
            "compatible_after": summary.get("compatible_after"),
            "missing_features_resolved_count": summary.get("missing_features_resolved_count"),
            "safe_for_future_inference": summary.get("safe_for_future_inference"),
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
