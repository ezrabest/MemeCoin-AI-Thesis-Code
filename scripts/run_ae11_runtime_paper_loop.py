#!/usr/bin/env python3
"""AE11 runtime paper trading loop — continuous operation with opportunity capture audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.runtime_paper_loop.loop_runner import run_ae11_runtime_paper_loop  # noqa: E402
from app.runtime_paper_loop.types import Ae11LoopConfig  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="AE11 runtime paper trading loop")
    parser.add_argument("--duration-minutes", type=float, default=60.0)
    parser.add_argument("--loop-interval-seconds", type=float, default=30.0)
    parser.add_argument("--enable-paper-demo-orders", action="store_true", default=False)
    parser.add_argument(
        "--allow-paper-trades-with-audit-blockers",
        action="store_true",
        default=False,
    )
    parser.add_argument("--exploration-mode", action="store_true", default=False)
    parser.add_argument("--strict-shadow-mode", action="store_true", default=True)
    parser.add_argument("--enable-live-dry-run", action="store_true", default=False)
    parser.add_argument("--starting-balance-usd", type=float, default=10_000.0)
    parser.add_argument("--notional-usd", type=float, default=100.0)
    parser.add_argument("--max-open-positions", type=int, default=10)
    parser.add_argument("--per-pair-cooldown-minutes", type=float, default=30.0)
    parser.add_argument("--max-price-age-seconds", type=float, default=900.0)
    parser.add_argument("--strict-shadow-max-price-age-seconds", type=float, default=30.0)
    parser.add_argument("--max-scan-records-per-iteration", type=int, default=5000)
    parser.add_argument("--take-profit-pct", type=float, default=20.0)
    parser.add_argument("--stop-loss-pct", type=float, default=10.0)
    parser.add_argument("--time-stop-minutes", type=float, default=240.0)
    parser.add_argument("--fee-bps", type=float, default=30.0)
    parser.add_argument("--entry-fee-bps", type=float, default=None)
    parser.add_argument("--exit-fee-bps", type=float, default=None)
    parser.add_argument("--slippage-bps", type=float, default=50.0)
    parser.add_argument(
        "--provider",
        choices=["mock", "qwen", "ollama", "gemini"],
        default="mock",
    )
    parser.add_argument("--allow-local-qwen", action="store_true", default=False)
    parser.add_argument("--allow-ollama", action="store_true", default=False)
    parser.add_argument("--allow-gemini", action="store_true", default=False)
    parser.add_argument("--max-records-per-iteration", type=int, default=50)
    parser.add_argument("--heartbeat-every-n-iterations", type=int, default=5)
    parser.add_argument("--no-real-wallet", action="store_true", default=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--resume-loop-run-id", type=str, default=None)
    parser.add_argument("--resume-latest-loop", action="store_true", default=False)
    # AE11I valuation / deterministic TP-SL proof (no external APIs)
    parser.add_argument(
        "--valuation-provider",
        choices=["legacy", "deterministic", "local_snapshot"],
        default="legacy",
    )
    parser.add_argument(
        "--deterministic-price-scenario",
        choices=[
            "neutral",
            "tp",
            "sl",
            "mixed",
            "incremental_tp",
            "incremental_sl",
            "incremental_mixed",
        ],
        default="neutral",
    )
    parser.add_argument("--deterministic-price-bump-pct", type=float, default=25.0)
    parser.add_argument("--deterministic-price-drop-pct", type=float, default=15.0)
    parser.add_argument("--deterministic-price-step-pct", type=float, default=5.0)
    parser.add_argument("--price-lifecycle-proof-mode", action="store_true", default=False)
    args = parser.parse_args()

    config = Ae11LoopConfig(
        project_root=ROOT,
        duration_minutes=args.duration_minutes,
        loop_interval_seconds=args.loop_interval_seconds,
        enable_paper_demo_orders=args.enable_paper_demo_orders,
        allow_paper_trades_with_audit_blockers=args.allow_paper_trades_with_audit_blockers,
        exploration_mode=args.exploration_mode,
        strict_shadow_mode=args.strict_shadow_mode,
        enable_live_dry_run=args.enable_live_dry_run,
        starting_balance_usd=args.starting_balance_usd,
        notional_usd=args.notional_usd,
        max_open_positions=args.max_open_positions,
        per_pair_cooldown_minutes=args.per_pair_cooldown_minutes,
        max_price_age_seconds=args.max_price_age_seconds,
        strict_shadow_max_price_age_seconds=args.strict_shadow_max_price_age_seconds,
        max_scan_records_per_iteration=args.max_scan_records_per_iteration,
        take_profit_pct=args.take_profit_pct,
        stop_loss_pct=args.stop_loss_pct,
        time_stop_minutes=args.time_stop_minutes,
        fee_bps=args.fee_bps,
        entry_fee_bps=args.entry_fee_bps,
        exit_fee_bps=args.exit_fee_bps,
        slippage_bps=args.slippage_bps,
        provider=args.provider,
        allow_local_qwen=args.allow_local_qwen,
        allow_ollama=args.allow_ollama,
        allow_gemini=args.allow_gemini,
        max_records_per_iteration=args.max_records_per_iteration,
        heartbeat_every_n_iterations=args.heartbeat_every_n_iterations,
        no_real_wallet=args.no_real_wallet,
        resume_loop_run_id=args.resume_loop_run_id,
        resume_latest_loop=args.resume_latest_loop,
        valuation_provider=args.valuation_provider,
        deterministic_price_scenario=args.deterministic_price_scenario,
        deterministic_price_bump_pct=args.deterministic_price_bump_pct,
        deterministic_price_drop_pct=args.deterministic_price_drop_pct,
        deterministic_price_step_pct=args.deterministic_price_step_pct,
        price_lifecycle_proof_mode=args.price_lifecycle_proof_mode,
    )

    summary = run_ae11_runtime_paper_loop(config)
    compact = {
        "phase": summary.get("phase"),
        "loop_run_id": summary.get("loop_run_id"),
        "invocation_id": summary.get("invocation_id"),
        "run_started_at_utc": summary.get("run_started_at_utc"),
        "final_status": summary.get("final_status"),
        "iterations_completed": summary.get("iterations_completed"),
        "decisions_seen": summary.get("decisions_seen"),
        "new_decisions_processed": summary.get("new_decisions_processed"),
        "duplicates_skipped": summary.get("duplicates_skipped"),
        "orders_created": summary.get("orders_created"),
        "fills_created": summary.get("fills_created"),
        "positions_opened": summary.get("positions_opened"),
        "positions_closed": summary.get("positions_closed"),
        "current_invocation_counters": summary.get("current_invocation_counters"),
        "cumulative_metrics": summary.get("cumulative_metrics"),
        "report_consistency_status": summary.get("report_consistency_status"),
        "open_positions_snapshot_rows": summary.get("open_positions_snapshot_rows"),
        "closed_trades_snapshot_rows": summary.get("closed_trades_snapshot_rows"),
        "report_consistency_mismatch_count": summary.get("report_consistency_mismatch_count"),
        "closed_trade_hygiene_status": summary.get("closed_trade_hygiene_status"),
        "canonical_closed_trades_rows": summary.get("canonical_closed_trades_rows"),
        "closed_trade_event_history_rows": summary.get("closed_trade_event_history_rows"),
        "invalid_closed_trade_rows": summary.get("invalid_closed_trade_rows"),
        "duplicate_closed_position_rows": summary.get("duplicate_closed_position_rows"),
        "duplicate_close_event_rows": summary.get("duplicate_close_event_rows"),
        "duplicate_close_attempt_count": summary.get("duplicate_close_attempt_count"),
        "blank_close_event_id_rows": summary.get("blank_close_event_id_rows"),
        "canonical_closed_position_unique_count": summary.get(
            "canonical_closed_position_unique_count"
        ),
        "cash_balance": summary.get("cash_balance"),
        "open_cost_basis_usd": summary.get("open_cost_basis_usd"),
        "open_market_value_usd": summary.get("open_market_value_usd"),
        "realized_pnl_usd": summary.get("realized_pnl_usd"),
        "unrealized_pnl_usd": summary.get("unrealized_pnl_usd"),
        "account_equity_usd": summary.get("account_equity_usd"),
        "ledger_consistency_status": summary.get("ledger_consistency_status"),
        "ledger_invariant_status": summary.get("ledger_invariant_status"),
        "ledger_invariant_failure_count": summary.get("ledger_invariant_failure_count"),
        "equity_bridge_status": summary.get("equity_bridge_status"),
        "equity_bridge_diff_usd": summary.get("equity_bridge_diff_usd"),
        "pnl_bridge_diff_usd": summary.get("pnl_bridge_diff_usd"),
        "starting_balance_usd": summary.get("starting_balance_usd"),
        "cash_balance_usd": summary.get("cash_balance_usd") or summary.get("cash_balance"),
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
        "account_equity_usd": summary.get("account_equity_usd"),
        "missing_open_economics_count": summary.get("missing_open_economics_count"),
        "blocked_open_economics_count": summary.get("blocked_open_economics_count"),
        "open_position_economic_completeness_status": summary.get(
            "open_position_economic_completeness_status"
        ),
        "current_open_positions_count": summary.get("current_open_positions_count"),
        "canonical_closed_positions_count": summary.get("canonical_closed_positions_count"),
        "position_registry_total_rows": summary.get("position_registry_total_rows"),
        "paper_order_event_count": summary.get("paper_order_event_count"),
        "paper_fill_event_count": summary.get("paper_fill_event_count"),
        "cumulative_metric_semantics_status": summary.get(
            "cumulative_metric_semantics_status"
        ),
        "price_oracle_status": summary.get("price_oracle_status"),
        "mark_to_market_status": summary.get("mark_to_market_status"),
        "valuation_source": summary.get("valuation_source"),
        "price_positions_evaluated": summary.get("price_positions_evaluated"),
        "price_positions_resolved": summary.get("price_positions_resolved"),
        "price_positions_missing": summary.get("price_positions_missing"),
        "price_positions_stale": summary.get("price_positions_stale"),
        "price_positions_pre_entry_stale": summary.get("price_positions_pre_entry_stale"),
        "price_positions_fallback": summary.get("price_positions_fallback"),
        "price_positions_deterministic": summary.get("price_positions_deterministic"),
        "tp_trigger_count": summary.get("tp_trigger_count"),
        "sl_trigger_count": summary.get("sl_trigger_count"),
        "time_stop_trigger_count": summary.get("time_stop_trigger_count"),
        "price_based_positions_closed": summary.get("price_based_positions_closed"),
        "tp_sl_lifecycle_status": summary.get("tp_sl_lifecycle_status"),
        "no_lookahead_status": summary.get("no_lookahead_status"),
        "temporal_validity_status": summary.get("temporal_validity_status"),
        "no_double_count_status": summary.get("no_double_count_status"),
        "price_oracle_audit_rows": summary.get("price_oracle_audit_rows"),
        "mark_to_market_audit_rows": summary.get("mark_to_market_audit_rows"),
        "tp_sl_trigger_audit_rows": summary.get("tp_sl_trigger_audit_rows"),
        "ledger_schema_version": summary.get("ledger_schema_version"),
        "accounting_model_version": summary.get("accounting_model_version"),
        "ledger_cash_tolerance_usd": summary.get("ledger_cash_tolerance_usd"),
        "invariant_check_method": summary.get("invariant_check_method"),
        "account_equity_diff": summary.get("account_equity_diff"),
        "cash_diff": summary.get("cash_diff"),
        "fee_model_status": summary.get("fee_model_status"),
        "entry_fee_bps": summary.get("entry_fee_bps"),
        "exit_fee_bps": summary.get("exit_fee_bps"),
        "slippage_bps": summary.get("slippage_bps"),
        "backfill_success_count": summary.get("backfill_success_count"),
        "backfill_missing_count": summary.get("backfill_missing_count"),
        "lifecycle_audit_status": summary.get("lifecycle_audit_status"),
        "lifecycle_positions_evaluated": summary.get("lifecycle_positions_evaluated"),
        "lifecycle_positions_closed": summary.get("lifecycle_positions_closed"),
        "lifecycle_blocked_count": summary.get("lifecycle_blocked_count"),
        "lifecycle_noop_reason": summary.get("lifecycle_noop_reason"),
        "position_state_semantics_status": summary.get("position_state_semantics_status"),
        "total_position_registry_rows": summary.get("total_position_registry_rows"),
        "open_positions_count": summary.get("open_positions_count"),
        "closed_rows_in_active_positions_count": summary.get(
            "closed_rows_in_active_positions_count"
        ),
        "closed_positions_count": summary.get("closed_positions_count"),
        "active_pair_locks_count": summary.get("active_pair_locks_count"),
        "locks_pointing_to_closed_count": summary.get("locks_pointing_to_closed_count"),
        "locks_without_position_count": summary.get("locks_without_position_count"),
        "ghost_lock_count": summary.get("ghost_lock_count"),
        "ghost_lock_repair_count": summary.get("ghost_lock_repair_count"),
        "cooldowns_count": summary.get("cooldowns_count"),
        "capacity_count_source": summary.get("capacity_count_source"),
        "wallet_configured": summary.get("wallet_configured"),
        "private_key_accessed": summary.get("private_key_accessed"),
        "real_transaction_attempted": summary.get("real_transaction_attempted"),
        "live_submission_status": summary.get("live_submission_status"),
        "audit_root": summary.get("audit_root"),
        "report_paths": summary.get("report_paths"),
        "startup": summary.get("startup"),
    }
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
