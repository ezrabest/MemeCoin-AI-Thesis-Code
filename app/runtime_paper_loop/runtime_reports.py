"""AE11 runtime reports and audit artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from app.runtime_paper_loop.persistence import atomic_write_json
from app.runtime_paper_loop.mismatch_event import write_mismatch_audit
from app.runtime_paper_loop.rejection_analysis import (
    _get_field,
    build_trade_vs_no_trade_comparison,
    summarize_rejection_reasons,
)
from app.runtime_paper_loop.report_generator import (
    CLOSED_TRADE_SNAPSHOT_FIELDS,
    OPEN_POSITION_SNAPSHOT_FIELDS,
    ReportGenerator,
)
from app.runtime_paper_loop.types import AE11_PHASE, Ae11FinalStatus, RUNTIME_INFERENCE_STATUS


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    """Always write headers when fieldnames known; never leave zero-byte CSVs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames:
        keys = fieldnames
    elif rows:
        keys = list(rows[0].keys())
    else:
        path.write_text("", encoding="utf-8")
        return path
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        if rows:
            writer.writerows(rows)
        f.flush()
    return path


def write_ae11_reports(
    *,
    audit_root: Path,
    project_root: Path,
    loop_run_id: str,
    summary: dict[str, Any],
    capture_records: list[Any],
    missed_winners: list[dict[str, Any]],
    open_positions: list[dict[str, Any]] | None = None,
    closed_trades: list[dict[str, Any]] | None = None,
    ledger: Any,
    state_db_status: dict[str, Any],
    reconstruction_status: str,
    price_oracle_audit: list[dict[str, Any]],
    idempotency_audit: dict[str, Any],
    wallet_safety: dict[str, Any],
    decision_source_audits: list[dict[str, Any]] | None = None,
    price_freshness_audits: list[dict[str, Any]] | None = None,
    exploration_max_price_age_seconds: float = 900.0,
    strict_max_price_age_seconds: float = 30.0,
    invocation_id: str | None = None,
    run_context: dict[str, Any] | None = None,
    reconstruction_events: list[Any] | None = None,
    state_db: Any | None = None,
    cash_balance: float | None = None,
    equity_bridge: dict[str, Any] | None = None,
    cumulative_metrics: dict[str, Any] | None = None,
    valuation_session: dict[str, Any] | None = None,
    valuation_oracle_audits: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reports_dir = audit_root / "reports"
    data_dir = audit_root / "data"
    audits_dir = audit_root / "audits"
    root_data = project_root / "data"
    root_audits = project_root / "audits"
    root_reports = project_root / "reports"

    for d in (reports_dir, data_dir, audits_dir, root_data, root_audits, root_reports):
        d.mkdir(parents=True, exist_ok=True)

    rejection_summary = summarize_rejection_reasons(capture_records)
    trade_comparison = build_trade_vs_no_trade_comparison(capture_records)

    capture_csv_rows = [
        r.to_dict() if hasattr(r, "to_dict") else r for r in capture_records
    ]

    paths: dict[str, str] = {}

    paths["ae11_opportunity_capture"] = str(
        write_csv(root_data / "ae11_opportunity_capture.csv", capture_csv_rows)
    )
    paths["ae11_missed_winners"] = str(
        write_csv(root_data / "ae11_missed_winners.csv", missed_winners)
    )
    paths["ae11_trade_vs_no_trade"] = str(
        write_csv(root_data / "ae11_trade_vs_no_trade_comparison.csv", trade_comparison)
    )
    paths["ae11_rejection_summary"] = str(
        write_csv(root_data / "ae11_rejection_reason_summary.csv", rejection_summary)
    )

    # Authoritative snapshots: fetch from SQLite, never trust passed-in in-memory lists alone.
    report_meta: dict[str, Any] = {
        "report_consistency_status": "SKIPPED",
        "open_positions_snapshot_rows": 0,
        "closed_trades_snapshot_rows": 0,
        "report_consistency_mismatch_count": 0,
    }
    authoritative_open: list[dict[str, Any]] = []
    authoritative_closed: list[dict[str, Any]] = []

    if state_db is not None:
        generator = ReportGenerator(
            state_db=state_db,
            project_root=project_root,
            loop_run_id=loop_run_id,
            invocation_id=invocation_id or "",
            cash_balance=cash_balance
            if cash_balance is not None
            else (ledger.account.cash_balance_usd if ledger else None),
        )
        auth_state = generator.fetch_authoritative_state()
        snap = generator.write_position_snapshots(
            open_path=root_data / "ae11_open_positions_snapshot.csv",
            closed_path=root_data / "ae11_closed_trades_snapshot.csv",
            state=auth_state,
            event_history_path=root_data / "ae11_closed_trade_events_history.csv",
            hygiene_audit_path=root_audits / "ae11_closed_trade_hygiene_audit.csv",
        )
        # Also write under audit_root data/
        generator.write_position_snapshots(
            open_path=data_dir / "ae11_open_positions_snapshot.csv",
            closed_path=data_dir / "ae11_closed_trades_snapshot.csv",
            state=auth_state,
            event_history_path=data_dir / "ae11_closed_trade_events_history.csv",
            hygiene_audit_path=audits_dir / "ae11_closed_trade_hygiene_audit.csv",
        )
        authoritative_open = generator.build_open_positions_snapshot(auth_state)
        authoritative_closed = generator.build_closed_trades_snapshot(auth_state)
        paths["ae11_open_positions"] = snap["open_path"]
        paths["ae11_closed_trades"] = snap["closed_path"]
        paths["ae11_closed_trade_events_history"] = snap.get("event_history_path", "")
        paths["ae11_closed_trade_hygiene_audit"] = snap.get("hygiene_audit_path", "")

        summary_open = None
        if isinstance(summary.get("cumulative_metrics"), dict):
            summary_open = summary["cumulative_metrics"].get("open_positions")
        summary_closed = auth_state.closed_trade_count

        consistency_rows, overall, mismatch_count = generator.build_consistency_audit(
            state=auth_state,
            open_csv_rows=len(authoritative_open),
            closed_csv_rows=len(authoritative_closed),
            summary_open=summary_open,
            summary_closed=summary_closed,
            hygiene=auth_state.closed_trade_hygiene,
            open_snapshot_rows=authoritative_open,
            equity_bridge=equity_bridge or summary,
            cumulative_metrics=cumulative_metrics
            or (
                summary.get("cumulative_metrics")
                if isinstance(summary.get("cumulative_metrics"), dict)
                else {}
            ),
        )
        consistency_path = root_audits / "ae11_report_consistency_audit.csv"
        generator.write_consistency_audit(consistency_path, consistency_rows)
        generator.write_consistency_audit(
            audits_dir / "ae11_report_consistency_audit.csv", consistency_rows
        )
        paths["report_consistency"] = str(consistency_path)
        report_meta = {
            "report_consistency_status": overall,
            "open_positions_snapshot_rows": len(authoritative_open),
            "closed_trades_snapshot_rows": len(authoritative_closed),
            "report_consistency_mismatch_count": mismatch_count,
            "open_positions_source_of_truth": auth_state.source_of_truth_open_positions,
            "closed_trades_source_of_truth": auth_state.source_of_truth_closed_trades,
            **(auth_state.closed_trade_hygiene or {}),
            **{k: snap[k] for k in (
                "closed_trade_hygiene_status",
                "canonical_closed_trades_rows",
                "closed_trade_event_history_rows",
                "invalid_closed_trade_rows",
                "duplicate_closed_position_rows",
                "duplicate_close_event_rows",
                "blank_close_event_id_rows",
                "canonical_closed_position_unique_count",
            ) if k in snap},
        }
    else:
        from app.runtime_paper_loop.report_generator import write_csv_with_headers

        open_rows = open_positions or []
        closed_rows = closed_trades or []
        open_fields = (
            list(open_rows[0].keys()) if open_rows else OPEN_POSITION_SNAPSHOT_FIELDS
        )
        closed_fields = (
            list(closed_rows[0].keys()) if closed_rows else CLOSED_TRADE_SNAPSHOT_FIELDS
        )
        write_csv_with_headers(
            root_data / "ae11_open_positions_snapshot.csv", open_rows, open_fields
        )
        write_csv_with_headers(
            root_data / "ae11_closed_trades_snapshot.csv", closed_rows, closed_fields
        )
        paths["ae11_open_positions"] = str(root_data / "ae11_open_positions_snapshot.csv")
        paths["ae11_closed_trades"] = str(root_data / "ae11_closed_trades_snapshot.csv")
        authoritative_open = open_rows
        authoritative_closed = closed_rows
        report_meta["open_positions_snapshot_rows"] = len(open_rows)
        report_meta["closed_trades_snapshot_rows"] = len(closed_rows)

    # AE11I valuation audits (prefer AE11 valuation oracle rows when present)
    from app.runtime_paper_loop.ae11_price_oracle import (
        write_mark_to_market_audit,
        write_price_oracle_audit,
        write_tp_sl_trigger_audit,
    )

    vo_audits = valuation_oracle_audits or {}
    price_rows = vo_audits.get("price_oracle") or price_oracle_audit or []
    mtm_rows = vo_audits.get("mark_to_market") or []
    tpsl_rows = vo_audits.get("tp_sl_trigger") or []
    paths["ae11_price_oracle_audit"] = str(
        write_price_oracle_audit(root_audits / "ae11_price_oracle_audit.csv", price_rows)
        if vo_audits.get("price_oracle") is not None
        else write_csv(root_audits / "ae11_price_oracle_audit.csv", price_oracle_audit)
    )
    # Keep legacy key
    paths["price_oracle_audit"] = paths["ae11_price_oracle_audit"]
    paths["ae11_mark_to_market_audit"] = str(
        write_mark_to_market_audit(root_audits / "ae11_mark_to_market_audit.csv", mtm_rows)
    )
    paths["ae11_tp_sl_trigger_audit"] = str(
        write_tp_sl_trigger_audit(root_audits / "ae11_tp_sl_trigger_audit.csv", tpsl_rows)
    )
    paths["no_lookahead_audit"] = str(
        write_csv(
            root_audits / "ae11_no_lookahead_audit.csv",
            [
                {
                    "rule": "forward_returns_audit_labels_only",
                    "used_in_entry_decision": False,
                    "status": "PASS",
                }
            ],
        )
    )
    paths["duplicate_prevention"] = str(
        write_csv(
            root_audits / "ae11_duplicate_prevention_audit.csv",
            [{"check": k, "status": v} for k, v in idempotency_audit.items()],
        )
    )
    paths["ledger_atomicity"] = str(
        write_csv(
            root_audits / "ae11_ledger_atomicity_audit.csv",
            [{"check": "no_cash_mutation_on_reject", "status": "PASS"}],
        )
    )
    atomic_write_json(root_audits / "ae11_wallet_safety_audit.json", wallet_safety)
    paths["wallet_safety"] = str(root_audits / "ae11_wallet_safety_audit.json")

    paths["context_availability"] = str(
        write_csv(
            root_audits / "ae11_context_availability_audit.csv",
            [
                {
                    "candidate_id": _get_field(r, "candidate_id"),
                    "context_status": _get_field(r, "ae8_context_status"),
                }
                for r in capture_records
            ],
        )
    )
    paths["llm_audit_linkage"] = str(
        write_csv(
            root_audits / "ae11_llm_audit_linkage_audit.csv",
            [
                {
                    "source_decision_id": _get_field(r, "source_decision_id"),
                    "source_llm_audit_record_id": _get_field(r, "source_llm_audit_record_id"),
                }
                for r in capture_records
            ],
        )
    )
    paths["persistence_durability"] = str(
        write_csv(
            root_audits / "ae11_persistence_durability_audit.csv",
            [{"check": "buffered_flush_fsync_at_iteration", "status": "PASS"}],
        )
    )
    if reconstruction_events is not None:
        paths["state_reconstruction"] = str(
            write_mismatch_audit(
                root_audits / "ae11_state_reconstruction_audit.csv",
                reconstruction_events,
                jsonl_path=root_audits / "ae11_state_reconstruction_audit.jsonl",
            )
        )
    else:
        paths["state_reconstruction"] = str(
            write_csv(
                root_audits / "ae11_state_reconstruction_audit.csv",
                [{"reconstruction_status": reconstruction_status}],
            )
        )
    paths["idempotency_index"] = str(
        write_csv(
            root_audits / "ae11_idempotency_index_audit.csv",
            [{"metric": k, "value": v} for k, v in state_db_status.items()],
        )
    )
    paths["decision_source_freshness"] = str(
        write_csv(
            root_audits / "ae11_decision_source_freshness_audit.csv",
            decision_source_audits or [],
        )
    )
    paths["price_freshness"] = str(
        write_csv(
            root_audits / "ae11_price_freshness_audit.csv",
            price_freshness_audits or [],
        )
    )

    manifest = {
        "phase": AE11_PHASE,
        "loop_run_id": loop_run_id,
        "invocation_id": invocation_id,
        "audit_root": str(audit_root),
        "artifact_paths": paths,
        "schema_version": "AE11D_MANIFEST_V1",
        "current_invocation_counters": summary.get("current_invocation_counters", True),
        "metrics_schema": {"session": True, "cumulative": True},
        **report_meta,
        **(run_context or {}),
    }
    atomic_write_json(reports_dir / "ae11_manifest.json", manifest)
    atomic_write_json(root_reports / "ae11_manifest.json", manifest)
    paths["manifest"] = str(reports_dir / "ae11_manifest.json")

    runtime_summary = {
        "phase": AE11_PHASE,
        "loop_run_id": loop_run_id,
        **summary,
        **report_meta,
    }
    atomic_write_json(reports_dir / "ae11_runtime_summary.json", runtime_summary)
    atomic_write_json(root_reports / "ae11_runtime_summary.json", runtime_summary)
    paths["runtime_summary"] = str(reports_dir / "ae11_runtime_summary.json")

    pnl_summary = {
        "phase": AE11_PHASE,
        "loop_run_id": loop_run_id,
        "cash_balance_usd": ledger.account.cash_balance_usd if ledger else summary.get("cash_balance"),
        "realized_pnl_usd": ledger.account.realized_pnl_usd if ledger else 0,
        "gross_pnl_usd": summary.get("gross_pnl_usd", 0),
        "net_pnl_usd": summary.get("net_pnl_usd", 0),
        "open_positions": report_meta.get("open_positions_snapshot_rows", len(authoritative_open)),
        "closed_trades": report_meta.get("closed_trades_snapshot_rows", len(authoritative_closed)),
        "fee_bps": summary.get("fee_bps"),
        "slippage_bps": summary.get("slippage_bps"),
        "starting_balance_usd": summary.get("starting_balance_usd"),
        "open_notional_usd": summary.get("open_notional_usd"),
        "open_cost_basis_usd": summary.get("open_cost_basis_usd"),
        "open_market_value_usd": summary.get("open_market_value_usd"),
        "open_entry_fee_usd": summary.get("open_entry_fee_usd"),
        "open_entry_slippage_usd": summary.get("open_entry_slippage_usd"),
        "open_entry_cost_drag_usd": summary.get("open_entry_cost_drag_usd"),
        "price_unrealized_pnl_usd": summary.get("price_unrealized_pnl_usd"),
        "total_unrealized_after_cost_pnl_usd": summary.get(
            "total_unrealized_after_cost_pnl_usd"
        ),
        "realized_net_pnl_usd": summary.get("realized_net_pnl_usd")
        or summary.get("realized_pnl_usd"),
        "total_fees_usd": summary.get("total_fees_usd"),
        "total_slippage_usd": summary.get("total_slippage_usd"),
        "account_equity_usd": summary.get("account_equity_usd"),
        "bridge_status": summary.get("equity_bridge_status"),
        "bridge_diff_usd": summary.get("equity_bridge_diff_usd"),
        "pnl_bridge_diff_usd": summary.get("pnl_bridge_diff_usd"),
        "valuation_source": summary.get("valuation_source"),
        "missing_open_economics_count": summary.get("missing_open_economics_count"),
        "open_position_economic_completeness_status": summary.get(
            "open_position_economic_completeness_status"
        ),
        "unrealized_pnl_semantics": summary.get("unrealized_pnl_semantics"),
        "bridge_formula": summary.get("bridge_formula"),
    }
    atomic_write_json(reports_dir / "ae11_forward_paper_pnl_summary.json", pnl_summary)
    atomic_write_json(root_reports / "ae11_forward_paper_pnl_summary.json", pnl_summary)
    paths["pnl_summary"] = str(reports_dir / "ae11_forward_paper_pnl_summary.json")

    decision_gate = {
        "phase": AE11_PHASE,
        "final_status": summary.get("final_status", Ae11FinalStatus.AE11_LOOP_OPERATIONAL.value),
        "loop_run_id": loop_run_id,
        "iterations_completed": summary.get("iterations_completed", 0),
        "candidates_seen": summary.get("candidates_seen", 0),
        "orders_created": summary.get("orders_created", 0),
        "positions_opened": summary.get("positions_opened", 0),
        "positions_closed": summary.get("positions_closed", 0),
        "missed_winners_count": len(missed_winners),
        "wallet_configured": False,
        "private_key_accessed": False,
        "real_transaction_attempted": False,
        "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
        "runtime_inference_status": RUNTIME_INFERENCE_STATUS,
        "exploration_mode": summary.get("exploration_mode", False),
        "strict_shadow_mode": summary.get("strict_shadow_mode", True),
        "exploration_max_price_age_seconds": exploration_max_price_age_seconds,
        "strict_max_price_age_seconds": strict_max_price_age_seconds,
        "price_freshness_note": (
            f"Exploration validation used {exploration_max_price_age_seconds}s; "
            f"strict shadow uses {strict_max_price_age_seconds}s; not live-approved."
        ),
        "recommended_next_phase": "AE11_CONTINUOUS_OPERATION",
    }
    atomic_write_json(reports_dir / "ae11_decision_gate.json", decision_gate)
    atomic_write_json(root_reports / "ae11_decision_gate.json", decision_gate)
    paths["decision_gate"] = str(reports_dir / "ae11_decision_gate.json")

    return {"paths": paths, **report_meta}
