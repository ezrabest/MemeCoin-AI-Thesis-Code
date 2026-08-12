#!/usr/bin/env python3
"""AE7 model score slot population smoke script — no training, no external APIs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.decision.score_population import run_ae7_score_population  # noqa: E402

DEFAULT_AE6_JSONL = ROOT / "data" / "decision_records" / "ae6_decisions_20260710.jsonl"
DEFAULT_INVENTORY_CSV = (
    ROOT
    / "data"
    / "audits"
    / "ae7_0_model_score_artifact_inventory_20260710_121711"
    / "ae7_0_model_score_artifact_inventory.csv"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AE7 safe model score slot population smoke run"
    )
    parser.add_argument("--ae6-jsonl", type=Path, default=DEFAULT_AE6_JSONL)
    parser.add_argument("--inventory-csv", type=Path, default=DEFAULT_INVENTORY_CSV)
    parser.add_argument("--max-records", type=int, default=50)
    parser.add_argument("--max-artifacts", type=int, default=50)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Build enriched records and audit without appending AE7 JSONL",
    )
    args = parser.parse_args()

    summary = run_ae7_score_population(
        ae6_jsonl=args.ae6_jsonl,
        inventory_csv=args.inventory_csv,
        project_root=ROOT,
        max_records=args.max_records,
        max_artifacts=args.max_artifacts,
        output_root=args.output_root,
        audit_only=args.audit_only,
    )

    compact = {
        "phase": summary.get("phase"),
        "score_population_decision": summary.get("score_population_decision"),
        "ae6_record_count": summary.get("ae6_record_count"),
        "records_enriched": summary.get("records_enriched"),
        "runtime_records_with_any_model_score": summary.get(
            "runtime_records_with_any_model_score"
        ),
        "runtime_records_missing_model_compatible_id": summary.get(
            "runtime_records_missing_model_compatible_id"
        ),
        "safe_prediction_artifact_count": summary.get("safe_prediction_artifact_count"),
        "audit_dir": summary.get("audit_dir"),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
