#!/usr/bin/env python3
"""Diagnostic 3 — signal-to-trade funnel waterfall."""
from __future__ import annotations

import argparse
import json
import sqlite3
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
    count_reasons,
    open_db_readonly,
    parse_audit_reasons_field,
    utc_now,
)


def _count_since(conn, table: str, cutoff: str) -> int:
    try:
        row = conn.execute(
            f"SELECT COUNT(*) AS cnt FROM {table} WHERE timestamp >= ?",
            (cutoff,),
        ).fetchone()
        return int(row["cnt"]) if row else 0
    except sqlite3.Error:
        return 0


def run(*, minutes: int, output_dir: Path) -> DiagnosticReport:
    from app.observability.audit_reasons import AuditReason
    from app.observability.effective_settings import get_effective_settings

    report = DiagnosticReport("inspect_signal_to_trade_funnel", output_dir)
    if not DB_PATH.is_file():
        report.add_limitation(f"Database missing: {DB_PATH}")
        report.set_status("FAIL")
        return report

    cutoff = (utc_now() - timedelta(minutes=minutes)).isoformat()
    conn = open_db_readonly()
    blocker_counts: Counter[str] = Counter()
    production_parse_bug_chars = 0
    audit_rows: list[dict] = []

    try:
        counts = {
            "raw_provider_payloads": _count_since(conn, "raw_provider_payloads", cutoff),
            "coins": int(conn.execute("SELECT COUNT(*) AS c FROM coins").fetchone()["c"]),
            "market_snapshots": _count_since(conn, "market_snapshots", cutoff),
            "signals": _count_since(conn, "signals", cutoff),
            "whale_alerts": _count_since(conn, "whale_alerts", cutoff),
            "paper_trades": _count_since(conn, "paper_trades", cutoff),
        }

        offset = 0
        while True:
            rows = conn.execute(
                """
                SELECT * FROM pipeline_audit
                WHERE timestamp >= ?
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (cutoff, CHUNK_SIZE, offset),
            ).fetchall()
            if not rows:
                break
            for r in rows:
                d = dict(r)
                audit_rows.append(d)
                reasons, _ = parse_audit_reasons_field(d.get("audit_reasons_json"))
                for reason in reasons:
                    blocker_counts[reason] += 1
                raw = d.get("audit_reasons_json")
                if isinstance(raw, str) and raw.startswith("["):
                    for ch in raw:
                        if len(ch) == 1 and ch in '[",]':
                            production_parse_bug_chars += 1
                            break
            offset += len(rows)
            if len(rows) < CHUNK_SIZE:
                break

        signal_actions = conn.execute(
            """
            SELECT signal_type, COUNT(*) AS cnt FROM signals
            WHERE timestamp >= ?
            GROUP BY signal_type
            """,
            (cutoff,),
        ).fetchall()
        action_map = {str(r["signal_type"]): int(r["cnt"]) for r in signal_actions}

        paper_buy_stages = conn.execute(
            """
            SELECT stage, filter_status, COUNT(*) AS cnt FROM pipeline_audit
            WHERE timestamp >= ?
              AND (stage LIKE '%paper%' OR filter_status LIKE '%BUY%' OR alert_type IS NOT NULL)
            GROUP BY stage, filter_status
            """,
            (cutoff,),
        ).fetchall()
    finally:
        conn.close()

    eff = get_effective_settings()
    settings = eff.canonical
    rf_threshold = float(settings.get("rf_probability_threshold", 0.70))
    economic_on = bool(settings.get("economic_gate_enabled", False))
    paper_on = bool(settings.get("paper_trading_enabled", True))

    actionable_watch = action_map.get("WATCH", 0)
    actionable_buy = 0
    rf_ok = blocker_counts.get(AuditReason.MODEL_RUNTIME_INFERENCE_OK.value, 0)
    rf_fail = sum(
        blocker_counts.get(r, 0)
        for r in (
            AuditReason.MODEL_RUNTIME_INFERENCE_NOT_AVAILABLE.value,
            AuditReason.MODEL_ARTIFACT_LOAD_FAILED.value,
            AuditReason.MODEL_SCHEMA_MISMATCH.value,
            AuditReason.MODEL_FEATURE_MISSING.value,
        )
    )
    rf_above = blocker_counts.get(AuditReason.ECONOMIC_GATE_APPROVED.value, 0) + blocker_counts.get(
        AuditReason.PAPER_BUY_CANDIDATE_CREATED.value, 0
    )
    economic_pass = blocker_counts.get(AuditReason.ECONOMIC_GATE_APPROVED.value, 0)
    paper_candidate = blocker_counts.get(AuditReason.PAPER_BUY_CANDIDATE_CREATED.value, 0)
    paper_executed = blocker_counts.get(AuditReason.PAPER_BUY_EXECUTED.value, 0) + blocker_counts.get(
        AuditReason.PAPER_TRADE_CREATED.value, 0
    )

    if not economic_on:
        report.add_limitation("economic_gate_enabled=false — RF/paper stages may not run in live path")
    if counts["whale_alerts"] == 0:
        report.add_limitation("whale_alerts_delta=0 in window — alert-required gates likely blocking")

    status = "PASS"
    if counts["signals"] > 0 and counts["whale_alerts"] == 0 and counts["paper_trades"] == 0:
        status = "WARN"
    if counts["market_snapshots"] > 100 and counts["paper_trades"] == 0 and not economic_on:
        status = "FAIL" if counts["signals"] > 0 else "WARN"

    report.set_status(status)
    funnel = {
        **counts,
        "pipeline_audit_rows": len(audit_rows),
        "actionable_watch_count": actionable_watch,
        "actionable_buy_candidate_count": paper_candidate,
        "rf_inference_attempted_count": rf_ok + rf_fail,
        "rf_inference_ok_count": rf_ok,
        "rf_inference_failed_count": rf_fail,
        "rf_above_configured_threshold_count": rf_above,
        "configured_rf_threshold": rf_threshold,
        "economic_gate_pass_count": economic_pass,
        "economic_gate_enabled": economic_on,
        "paper_trading_enabled": paper_on,
        "paper_order_attempted_count": paper_candidate + paper_executed,
        "paper_order_inserted_count": counts["paper_trades"],
        "signal_action_breakdown": action_map,
        "paper_related_stages": [dict(r) for r in paper_buy_stages],
        "production_audit_json_string_rows": sum(
            1 for r in audit_rows if isinstance(r.get("audit_reasons_json"), str) and str(r["audit_reasons_json"]).startswith("[")
        ),
    }
    report.data["funnel"] = funnel
    report.data["blocker_counts"] = dict(blocker_counts)
    report.data["minutes"] = minutes
    report.data["cutoff"] = cutoff
    report.data["likely_production_audit_parse_bug"] = production_parse_bug_chars > 0 or funnel["production_audit_json_string_rows"] > 0

    report.write_json("funnel_waterfall.json")
    report.write_md([
        "## Funnel",
        *[f"- {k}: {v}" for k, v in funnel.items() if not isinstance(v, (list, dict))],
        "",
        "## Top blockers",
        *[f"- {k}: {v}" for k, v in blocker_counts.most_common(20)],
    ], "funnel_waterfall.md")
    report.write_csv(
        [{"reason": k, "count": v} for k, v in blocker_counts.most_common()],
        "blocker_counts.csv",
    )

    blocked = [
        {
            "timestamp": r.get("timestamp"),
            "symbol": r.get("symbol"),
            "pair_address": r.get("pair_address"),
            "stage": r.get("stage"),
            "signal_action": r.get("filter_status"),
            "alert_type": r.get("alert_type"),
            "whale_score": r.get("whale_score"),
            "reasons": parse_audit_reasons_field(r.get("audit_reasons_json"))[0],
        }
        for r in audit_rows[:200]
        if r.get("filter_status") in ("WATCH", "NO_TRADE", "HOLD", "audit")
    ]
    report.write_csv(blocked[:100], "top_blocked_candidates.csv")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minutes", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(minutes=args.minutes, output_dir=args.output_dir)
    print(f"Status: {report.data['status']}")
    return 0 if report.data["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
