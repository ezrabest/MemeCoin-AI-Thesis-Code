#!/usr/bin/env python3
"""AE6 consensus decision layer smoke script — read-only DB, JSONL audit output."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.decision.builder import build_decision_record, fetch_recent_signal_candidates  # noqa: E402
from app.decision.persistence import DecisionJsonlWriter, decision_records_path_for_date  # noqa: E402
from app.decision.types import AE6_PHASE  # noqa: E402
from scripts.diagnostics._common import open_db_readonly, timestamp_slug  # noqa: E402


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_audit_summary(records: list[dict]) -> dict:
    status_dist = Counter(r.get("decision_status") for r in records)
    consensus_dist = Counter(
        (r.get("consensus") or {}).get("consensus_family") for r in records
    )
    lineage_strength_dist = Counter(
        (r.get("lineage") or {}).get("lineage_strength") for r in records
    )
    lineage_fallback_count = sum(
        1
        for r in records
        if (r.get("lineage") or {}).get("lineage_mode") == "BEST_EFFORT_IMPLICIT_LINKAGE"
    )

    missingness_counter: Counter[str] = Counter()
    for r in records:
        for m in r.get("missingness") or []:
            missingness_counter[m] += 1

    return {
        "phase": AE6_PHASE,
        "created_at_utc": _utc_now_iso(),
        "decision_record_count": len(records),
        "decision_status_distribution": dict(status_dist),
        "consensus_family_distribution": dict(consensus_dist),
        "lineage_strength_distribution": dict(lineage_strength_dist),
        "lineage_fallback_count": lineage_fallback_count,
        "missingness_summary": dict(missingness_counter.most_common(20)),
        "no_trade_authority": True,
        "safety": {
            "no_model_training": True,
            "no_llm_calls": True,
            "no_external_provider_calls": True,
            "no_real_trading": True,
            "no_paper_trade_execution": True,
        },
    }


def run_smoke(
    *,
    max_records: int = 50,
    output_root: Path | None = None,
    no_db_write: bool = False,
    audit_only: bool = False,
) -> dict:
    del no_db_write  # JSONL-only AE6 path; accepted as no-op

    output_root = output_root or (ROOT / "data" / "audits")
    audit_dir = output_root / f"ae6_consensus_decision_layer_{timestamp_slug()}"
    audit_dir.mkdir(parents=True, exist_ok=True)

    conn = open_db_readonly()
    try:
        candidates = fetch_recent_signal_candidates(conn, limit=max_records)
        jsonl_path = decision_records_path_for_date()

        written_records: list[dict] = []
        if audit_only:
            for bundle in candidates:
                record = build_decision_record(conn=conn, **bundle)
                written_records.append(record.model_dump(mode="json"))
        else:
            with DecisionJsonlWriter(path=jsonl_path) as writer:
                for bundle in candidates:
                    record = build_decision_record(conn=conn, **bundle)
                    writer.append_record(record)
                    written_records.append(record.model_dump(mode="json"))

        summary = _build_audit_summary(written_records)
        summary["jsonl_path"] = str(jsonl_path)
        summary["audit_only"] = audit_only
        summary["candidates_fetched"] = len(candidates)

        summary_path = audit_dir / "ae6_consensus_decision_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        compact = {
            "phase": AE6_PHASE,
            "decision_record_count": summary["decision_record_count"],
            "decision_status_distribution": summary["decision_status_distribution"],
            "consensus_family_distribution": summary["consensus_family_distribution"],
            "lineage_fallback_count": summary["lineage_fallback_count"],
            "audit_dir": str(audit_dir),
            "jsonl_path": str(jsonl_path),
        }
        print(json.dumps(compact, indent=2))
        return summary
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="AE6 consensus decision layer smoke run")
    parser.add_argument("--max-records", type=int, default=50)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--no-db-write",
        action="store_true",
        help="No-op for AE6 JSONL-only persistence (accepted for CLI compatibility)",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Build records and audit summary without appending JSONL",
    )
    args = parser.parse_args()
    run_smoke(
        max_records=args.max_records,
        output_root=args.output_root,
        no_db_write=args.no_db_write,
        audit_only=args.audit_only,
    )


if __name__ == "__main__":
    main()
