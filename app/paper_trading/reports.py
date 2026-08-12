"""AE10 audit report writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_json(path: Path, data: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    return path


def write_jsonl_sample(path: Path, records: list[dict[str, Any]], limit: int = 50) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records[:limit]:
            f.write(json.dumps(rec, default=str, separators=(",", ":")) + "\n")
    return path


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        fields = fieldnames or []
    else:
        fields = fieldnames or list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_ae10_audits(
    audit_dir: Path,
    *,
    summary: dict[str, Any],
    decision_gate: dict[str, Any],
    traceability_records: list[dict[str, Any]],
    paper_orders: list[dict[str, Any]],
    paper_positions: list[dict[str, Any]],
    live_dry_run_orders: list[dict[str, Any]],
    no_wallet_audit: dict[str, Any],
    reset_audit: dict[str, Any] | None,
    traceability_csv_rows: list[dict[str, Any]],
    llm_provider_audit: dict[str, Any],
    state_machine_rows: list[dict[str, Any]],
    price_oracle_rows: list[dict[str, Any]],
    latency_rows: list[dict[str, Any]],
) -> dict[str, str]:
    reports_dir = audit_dir / "reports"
    data_dir = audit_dir / "data"
    audits_dir = audit_dir / "audits"

    paths: dict[str, str] = {}

    paths["ae10_paper_demo_summary"] = str(
        write_json(reports_dir / "ae10_paper_demo_summary.json", summary)
    )
    paths["ae10_decision_gate"] = str(
        write_json(reports_dir / "ae10_decision_gate.json", decision_gate)
    )
    paths["ae10_traceability_records_sample"] = str(
        write_jsonl_sample(data_dir / "ae10_traceability_records_sample.jsonl", traceability_records)
    )
    paths["ae10_paper_orders_sample"] = str(
        write_jsonl_sample(data_dir / "ae10_paper_orders_sample.jsonl", paper_orders)
    )
    paths["ae10_paper_positions_sample"] = str(
        write_jsonl_sample(data_dir / "ae10_paper_positions_sample.jsonl", paper_positions)
    )
    paths["ae10_live_dry_run_orders_sample"] = str(
        write_jsonl_sample(data_dir / "ae10_live_dry_run_orders_sample.jsonl", live_dry_run_orders)
    )
    paths["ae10_no_wallet_live_dry_run_audit"] = str(
        write_json(audits_dir / "ae10_no_wallet_live_dry_run_audit.json", no_wallet_audit)
    )
    if reset_audit:
        paths["ae10_reset_demo_funds_audit"] = str(
            write_json(audits_dir / "ae10_reset_demo_funds_audit.json", reset_audit)
        )
    paths["ae10_traceability_audit"] = str(
        write_csv(audits_dir / "ae10_traceability_audit.csv", traceability_csv_rows)
    )
    paths["ae10_llm_provider_config_audit"] = str(
        write_json(audits_dir / "ae10_llm_provider_config_audit.json", llm_provider_audit)
    )
    paths["ae10_order_state_machine_audit"] = str(
        write_csv(
            audits_dir / "ae10_order_state_machine_audit.csv",
            state_machine_rows,
            fieldnames=[
                "paper_order_id",
                "from_state",
                "to_state",
                "transition_allowed",
                "reason",
                "created_at_utc",
            ],
        )
    )
    paths["ae10_price_oracle_audit"] = str(
        write_csv(
            audits_dir / "ae10_price_oracle_audit.csv",
            price_oracle_rows,
            fieldnames=[
                "price",
                "price_source",
                "price_snapshot_id",
                "price_timestamp",
                "price_timestamp_used",
                "order_timestamp",
                "order_created_at_utc",
                "decision_created_at_utc",
                "system_now_utc",
                "snapshot_provider_timestamp",
                "snapshot_ingested_at_utc",
                "price_age_seconds",
                "max_price_age_seconds",
                "time_skew_seconds",
                "max_provider_time_skew_seconds",
                "lookahead_detected",
                "provider_time_skew_detected",
                "price_status",
                "created_at_utc",
            ],
        )
    )
    paths["ae10_execution_latency_audit"] = str(
        write_csv(
            audits_dir / "ae10_execution_latency_audit.csv",
            latency_rows,
            fieldnames=[
                "paper_order_id",
                "candidate_id",
                "decision_created_at_utc",
                "order_created_at_utc",
                "filled_at_utc",
                "execution_latency_ms",
                "execution_latency_status",
                "created_at_utc",
            ],
        )
    )
    return paths
