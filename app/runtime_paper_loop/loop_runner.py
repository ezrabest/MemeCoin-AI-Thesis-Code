"""AE11 runtime paper trading loop runner."""

from __future__ import annotations

import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.decision.persistence import read_jsonl_records_safe
from app.execution.adapters import PaperExecutionAdapter
from app.execution.execution_orchestrator import (
    _index_by_key,
    _match_ae9,
    build_traceability_record,
)
from app.execution.live_no_wallet_adapter import LiveNoWalletDryRunAdapter
from app.execution.types import OrderIntent
from app.llm_audit.audit_runner import discover_latest_glob
from app.paper_trading.ledger import DemoLedger
from app.paper_trading.order_simulator import PaperOrderSimulator
from app.paper_trading.persistence import (
    paper_orders_path_for_date,
    paper_positions_path_for_date,
    paper_trades_path_for_date,
)
from app.paper_trading.position_manager import close_position_manual
from app.paper_trading.price_oracle import DemoPriceOracle
from app.paper_trading.types import DemoAccount, PaperOrderStatus
from app.runtime_paper_loop.checkpointing import build_checkpoint_payload, load_checkpoint, write_checkpoint
from app.runtime_paper_loop.decision_policy import (
    evaluate_exploration_decision,
    evaluate_strict_shadow_decision,
)
from app.runtime_paper_loop.decision_source import (
    SOURCE_NAME,
    build_source_event_key,
    discover_ae6_path,
    generate_fresh_ae6_decisions,
    load_unprocessed_batch,
)
from app.runtime_paper_loop.price_freshness import evaluate_price_freshness
from app.runtime_paper_loop.trade_decision import build_hierarchical_trade_decision
from app.runtime_paper_loop.idempotency import AE11StateDb
from app.runtime_paper_loop.missed_winners import update_missed_winners
from app.runtime_paper_loop.opportunity_capture import (
    build_opportunity_capture_record,
    compute_forward_returns_no_lookahead,
)
from app.runtime_paper_loop.persistence import IterationWriters, clean_stale_tmp_files
from app.runtime_paper_loop.runtime_reports import write_ae11_reports
from app.runtime_paper_loop.mismatch_event import detect_checkpoint_mismatches, write_mismatch_audit
from app.runtime_paper_loop.run_context import RunContext, RunContextFactory
from app.runtime_paper_loop.run_metrics import RunMetrics
from app.runtime_paper_loop.state_reconstruction import ledger_reconstruction
from app.runtime_paper_loop.report_generator import (
    sqlite_positions_to_paper_positions,
)
from app.runtime_paper_loop.types import (
    AE11_PHASE,
    AE11_SCHEMA_VERSION,
    Ae11FinalStatus,
    Ae11LoopConfig,
    EXPLORATION_OVERRIDE_TYPE,
    utc_now_iso,
)


def _load_candidates_from_jsonl(
    project_root: Path,
    max_records: int,
    *,
    ae6_jsonl: Path | None = None,
    ae8_jsonl: Path | None = None,
    ae9_jsonl: Path | None = None,
) -> list[tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]]:
    ae6_path = ae6_jsonl or discover_latest_glob(
        project_root, "data/decision_records/ae6_decisions_*.jsonl"
    )
    ae8_path = ae8_jsonl or discover_latest_glob(
        project_root, "data/context_intelligence/ae8_context_features_*.jsonl"
    )
    ae9_path = ae9_jsonl or discover_latest_glob(
        project_root, "data/llm_audit/ae9_llm_audit_records_*.jsonl"
    )

    ae6_records: list[dict[str, Any]] = []
    ae8_records: list[dict[str, Any]] = []
    ae9_records: list[dict[str, Any]] = []

    if ae6_path and ae6_path.is_file():
        ae6_records, _ = read_jsonl_records_safe(ae6_path)
    if ae8_path and ae8_path.is_file():
        ae8_records, _ = read_jsonl_records_safe(ae8_path)
    if ae9_path and ae9_path.is_file():
        ae9_records, _ = read_jsonl_records_safe(ae9_path)

    from app.execution.execution_orchestrator import _index_ae9_records

    ae8_index = _index_by_key(ae8_records, "candidate_id", "pair_address", "context_record_id")
    ae9_by_key, ae9_by_decision_id = _index_ae9_records(ae9_records)

    candidates: list[tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]] = []
    if ae6_records:
        for decision in ae6_records[:max_records]:
            identity = decision.get("candidate_identity") or {}
            ctx = None
            for key in (identity.get("candidate_id"), identity.get("pair_address"), decision.get("decision_id")):
                if key and str(key) in ae8_index:
                    ctx = ae8_index[str(key)]
                    break
            aud = _match_ae9(decision, ae9_by_key, ae9_by_decision_id)
            candidates.append((decision, ctx, aud))
    elif ae8_records:
        for ctx in ae8_records[:max_records]:
            aud = ae9_by_key.get(str(ctx.get("candidate_id")))
            candidates.append((None, ctx, aud))

    return candidates


def _run_headless_refresh(project_root: Path, config: Ae11LoopConfig) -> dict[str, Any]:
    """Headless scan / data refresh — read-only DB, optional AE6/AE8/AE9 generation."""
    refresh_status: dict[str, Any] = {"provider": config.provider, "decisions_generated": 0}
    try:
        from scripts.diagnostics._common import open_db_readonly

        conn = open_db_readonly()
        try:
            from app.decision.builder import build_decision_record, fetch_recent_signal_candidates
            from app.decision.persistence import DecisionJsonlWriter, decision_records_path_for_date

            bundles = fetch_recent_signal_candidates(conn, limit=config.max_records_per_iteration)
            if bundles:
                writer = DecisionJsonlWriter(decision_records_path_for_date())
                for bundle in bundles:
                    record = build_decision_record(conn=conn, **bundle)
                    if not config.exploration_mode:
                        pass
                    writer.append_record(record)
                    refresh_status["decisions_generated"] += 1
                writer.close()
        finally:
            conn.close()
    except FileNotFoundError:
        refresh_status["db_unavailable"] = True
    except Exception as exc:
        refresh_status["error"] = str(exc)
    return refresh_status


def _apply_fees(notional: float, fee_bps: float, slippage_bps: float) -> tuple[float, float]:
    from app.runtime_paper_loop.decimal_money import bps_cost

    fee = float(bps_cost(notional, fee_bps))
    slip = float(bps_cost(notional, slippage_bps))
    return fee, slip


def _manage_open_positions(
    ledger: DemoLedger,
    price_oracle: DemoPriceOracle,
    state_db: AE11StateDb,
    writers: IterationWriters,
    *,
    config: Ae11LoopConfig,
    loop_run_id: str,
    iteration: int,
    invocation_id: str = "",
    project_root: Path | None = None,
    valuation_oracle: Any | None = None,
) -> dict[str, Any]:
    """TP / SL / time-stop close lifecycle on authoritative SQLite positions."""
    from app.runtime_paper_loop.position_lifecycle import evaluate_and_close_positions

    root = project_root or Path(__file__).parents[2]
    return evaluate_and_close_positions(
        state_db,
        price_oracle,
        writers,
        config=config,
        loop_run_id=loop_run_id,
        invocation_id=invocation_id,
        iteration=iteration,
        project_root=root,
        ledger=ledger,
        valuation_oracle=valuation_oracle,
    )


class RuntimePaperLoopRunner:
    """Long-running AE11 paper trading loop with opportunity capture audit."""

    def __init__(self, config: Ae11LoopConfig) -> None:
        self.config = config
        self.project_root = Path(config.project_root) if config.project_root else Path(__file__).parents[2]
        self.run_context: RunContext | None = None
        self.session_iteration = 0
        self.state_db = AE11StateDb()
        self.ledger = DemoLedger(
            account=DemoAccount(
                starting_balance_usd=config.starting_balance_usd,
                cash_balance_usd=config.starting_balance_usd,
            )
        )
        self.price_oracle = DemoPriceOracle(max_price_age_seconds=config.max_price_age_seconds)
        from app.runtime_paper_loop.ae11_price_oracle import build_ae11_price_oracle

        self.valuation_oracle = build_ae11_price_oracle(config)
        self.valuation_session: dict[str, Any] = {}
        self.simulator = PaperOrderSimulator(price_oracle=self.price_oracle)
        self.paper_adapter = PaperExecutionAdapter(self.ledger, self.simulator)
        self.live_adapter = LiveNoWalletDryRunAdapter()
        self.capture_records: list[Any] = []
        self.all_missed_winners: list[dict[str, Any]] = []
        self.metrics = RunMetrics()
        self.reconstruction_status = "NOT_RUN"
        self.reconstruction_events: list[Any] = []
        self.latest_decision_cursor: str | None = None
        self.latest_decision_timestamp: str | None = None
        self.decision_source_audits: list[dict[str, Any]] = []
        self.price_freshness_audits: list[dict[str, Any]] = []
        self.last_source_load: Any = None
        self.last_reconstruction_at: str | None = None
        self.ledger_snapshot: Any = None
        self.backfill_summary: dict[str, Any] = {}
        self.migration_summary: dict[str, Any] = {}
        self.lifecycle_session: dict[str, Any] = {
            "positions_evaluated": 0,
            "positions_closed": 0,
            "positions_blocked": 0,
            "duplicate_close_attempt_count": 0,
            "exit_reasons": {},
            "lifecycle_noop_reason": None,
            "lifecycle_audit_status": None,
        }
        self.position_state_semantics: dict[str, Any] = {}
        self.ledger_invariant_status: str = "NOT_CHECKED"
        self.ledger_invariant_failure_count: int = 0
        self.ledger_invariant_last_failure_reason: str | None = None
        self.invariant_check_stage: str | None = None
        self.equity_bridge_result: Any = None
        self.ledger_economic_mutations_blocked: bool = False
        self._iter_stats = {
            "strict_decisions_blocked": 0,
            "exploration_decisions_paper_buy": 0,
            "exploration_decisions_blocked_by_hard_safety": 0,
            "unprocessed_decisions_found": 0,
            "no_new_decisions_available": False,
        }

    def _run_ledger_invariants(self, *, stage: str, raise_on_failure: bool = False) -> dict[str, Any]:
        """AE11H explicit Decimal invariant checks (never Python assert)."""
        from app.runtime_paper_loop.equity_bridge import (
            LedgerInvariantViolation,
            build_equity_bridge,
            check_ledger_invariants,
            write_equity_bridge_audit,
        )
        from app.runtime_paper_loop.ledger_accounting import reconstruct_ledger_from_sqlite

        snap = reconstruct_ledger_from_sqlite(
            self.state_db,
            starting_balance_usd=self.config.starting_balance_usd,
            entry_fee_bps=self.config.resolved_entry_fee_bps(),
            exit_fee_bps=self.config.resolved_exit_fee_bps(),
            slippage_bps=self.config.slippage_bps,
        )
        bridge = build_equity_bridge(
            self.state_db,
            starting_balance_usd=self.config.starting_balance_usd,
            loop_run_id=self.loop_run_id,
            invocation_id=getattr(self.run_context, "invocation_id", "") or "",
            cash_balance_override=snap.cash_balance,
            account_equity_override=snap.account_equity_usd,
        )
        result = check_ledger_invariants(
            cash_balance=snap.cash_balance,
            open_market_value_usd=snap.open_market_value_usd,
            account_equity_usd=snap.account_equity_usd,
            starting_balance_usd=snap.starting_balance_usd,
            realized_net_pnl_usd=snap.realized_pnl_usd,
            total_unrealized_after_cost_pnl_usd=bridge.open_total_unrealized_after_cost_pnl_usd,
            open_entry_slippage_usd=bridge.open_entry_slippage_usd,
            open_cash_debited_usd=bridge.open_cash_debited_usd,
            expected_cash_balance=snap.expected_cash_balance,
            realized_from_cash_flows=(
                bridge.expected_account_equity_usd
                - snap.starting_balance_usd
                - bridge.open_market_value_usd
                + bridge.open_cash_debited_usd
            ),
            stage=stage,
            raise_on_failure=False,
        )
        self.invariant_check_stage = stage
        self.ledger_invariant_status = result["ledger_invariant_status"]
        self.ledger_invariant_failure_count = int(result["ledger_invariant_failure_count"])
        self.ledger_invariant_last_failure_reason = result.get(
            "ledger_invariant_last_failure_reason"
        )
        if result["ledger_invariant_status"] != "PASS":
            self.ledger_economic_mutations_blocked = True
            bridge.bridge_status = "FATAL"
            bridge.notes = (
                f"{bridge.notes}; INVARIANT_FAIL stage={stage}; "
                f"{result.get('ledger_invariant_last_failure_reason')}"
            )
            write_equity_bridge_audit(
                self.project_root / "audits" / "ae11_equity_bridge_audit.csv",
                loop_run_id=self.loop_run_id,
                invocation_id=getattr(self.run_context, "invocation_id", "") or "",
                bridge=bridge,
            )
            if raise_on_failure:
                raise LedgerInvariantViolation(
                    result.get("ledger_invariant_last_failure_reason") or "invariant_failed",
                    stage=stage,
                    details=result,
                )
        return result

    @property
    def loop_run_id(self) -> str:
        return self.run_context.loop_run_id if self.run_context else str(uuid4())

    @property
    def iteration(self) -> int:
        return self.session_iteration

    def startup(self) -> dict[str, Any]:
        """Load checkpoint + migrate/backfill + reconstruct ledger — fresh run identity."""
        from app.runtime_paper_loop.db_migration import (
            migrate_db_schema,
            update_migration_audit_backfill,
        )
        from app.runtime_paper_loop.ledger_accounting import (
            reconstruct_ledger_from_sqlite,
            write_ledger_consistency_audit,
        )
        from app.runtime_paper_loop.mismatch_event import MismatchEvent
        from app.runtime_paper_loop.position_backfill import backfill_position_economics
        from app.runtime_paper_loop.position_state_semantics import (
            audit_position_state_semantics,
        )

        clean_stale_tmp_files()

        # 1) Atomic pre-loop schema migration (never inside iteration loop)
        self.migration_summary = migrate_db_schema(
            db_path=self.state_db.path,
            project_root=self.project_root,
        )

        checkpoint = load_checkpoint()
        explicit_resume = bool(
            self.config.resume_loop_run_id or self.config.resume_latest_loop
        )
        resume_loop_id = self.config.resume_loop_run_id
        if self.config.resume_latest_loop and checkpoint:
            cr = checkpoint.get("current_run") or {}
            resume_loop_id = cr.get("loop_run_id") or checkpoint.get("loop_run_id")
        resume_audit = None
        if explicit_resume and checkpoint:
            cr = checkpoint.get("current_run") or {}
            ar = cr.get("audit_root") or checkpoint.get("audit_root")
            if ar:
                resume_audit = Path(ar)

        self.run_context, _persistent = RunContextFactory.create(
            project_root=self.project_root,
            checkpoint=checkpoint,
            explicit_resume_requested=explicit_resume,
            resume_loop_run_id=resume_loop_id,
            resume_audit_root=resume_audit,
        )

        def _paper_paths():
            paper_dir = self.project_root / "data" / "paper_trading"
            if paper_dir.is_dir():
                orders = list(sorted(paper_dir.glob("paper_orders_*.jsonl")))
                positions = list(sorted(paper_dir.glob("paper_positions_*.jsonl")))
                trades = list(sorted(paper_dir.glob("paper_trades_*.jsonl")))
                # Prefer project-local files only — never fall back to global DATA_DIR
                # when a project paper_trading directory exists (even if empty).
                return orders, positions, trades
            return (
                [paper_orders_path_for_date()],
                [paper_positions_path_for_date()],
                [paper_trades_path_for_date()],
            )

        order_paths, position_paths, trade_paths = _paper_paths()
        reconstructed = ledger_reconstruction(
            paper_orders_paths=order_paths,
            paper_positions_paths=position_paths,
            paper_trades_paths=trade_paths,
            state_db=self.state_db,
            starting_balance_usd=self.config.starting_balance_usd,
            allow_negative_cash=self.config.allow_negative_cash,
            allow_duplicate_pair=self.config.allow_duplicate_pair,
        )

        # Repair SQLite open-state from reconstruction BEFORE economics backfill
        # so INSERT/UPDATE identity repair cannot wipe economics columns.
        if reconstructed.reconstruction_status == "MISMATCH":
            self.state_db.repair_from_reconstruction(reconstructed)

        # 2) Backfill economics from JSONL (pre-loop only, after identity repair)
        self.backfill_summary = backfill_position_economics(
            self.state_db,
            project_root=self.project_root,
            loop_run_id=self.run_context.loop_run_id,
            invocation_id=self.run_context.invocation_id,
            take_profit_pct=self.config.take_profit_pct,
            stop_loss_pct=self.config.stop_loss_pct,
            time_stop_minutes=self.config.time_stop_minutes,
            entry_fee_bps=self.config.resolved_entry_fee_bps(),
            slippage_bps=self.config.slippage_bps,
            historical_cash_debit_notional_only=True,
        )
        update_migration_audit_backfill(
            project_root=self.project_root,
            backfill_started_at_utc=self.backfill_summary["backfill_started_at_utc"],
            backfill_completed_at_utc=self.backfill_summary["backfill_completed_at_utc"],
            backfill_position_count=self.backfill_summary["backfill_position_count"],
            backfill_success_count=self.backfill_summary["backfill_success_count"],
            backfill_partial_count=self.backfill_summary["backfill_partial_count"],
            backfill_missing_count=self.backfill_summary["backfill_missing_count"],
        )

        self.reconstruction_events = detect_checkpoint_mismatches(
            loop_run_id=self.run_context.loop_run_id,
            invocation_id=self.run_context.invocation_id,
            checkpoint=checkpoint,
            reconstructed=reconstructed,
            state_db=self.state_db,
            explicit_resume_requested=explicit_resume,
            run_context=self.run_context,
        )
        # Append position-level economics mismatch events from backfill
        for ev in self.backfill_summary.get("mismatch_events") or []:
            self.reconstruction_events.append(
                MismatchEvent(
                    loop_run_id=self.run_context.loop_run_id,
                    invocation_id=self.run_context.invocation_id,
                    checkpoint_loaded=checkpoint is not None,
                    explicit_resume_requested=explicit_resume,
                    reconstruction_status="MISMATCH_REPAIRED"
                    if ev.get("repair_action") == "BACKFILLED_FROM_JSONL"
                    else "MISMATCH",
                    mismatch_detected=ev.get("sqlite_value") in ("MISSING", "PARTIAL"),
                    mismatch_type=ev.get("mismatch_type", "POSITION_ECONOMICS_MISSING"),
                    component=ev.get("component", "positions"),
                    field_path=ev.get("field_path", ""),
                    field=ev.get("field", "economic_enrichment_status"),
                    sqlite_value=ev.get("sqlite_value"),
                    source_of_truth=ev.get("source_of_truth", "sqlite_jsonl_reconciliation"),
                    repair_action=ev.get("repair_action", "BACKFILL_FAILED_WARNING_ONLY"),
                    repair_applied=ev.get("repair_action") == "BACKFILLED_FROM_JSONL",
                    repair_target="active_positions",
                    severity=ev.get("severity", "WARNING"),
                    notes=ev.get("notes", ""),
                )
            )

        repaired = [e for e in self.reconstruction_events if e.repair_applied and e.mismatch_detected]
        if any(e.mismatch_detected for e in self.reconstruction_events):
            self.reconstruction_status = "MISMATCH_REPAIRED" if repaired else "MISMATCH"
        else:
            self.reconstruction_status = "OK"

        audit_csv = self.project_root / "audits" / "ae11_state_reconstruction_audit.csv"
        audit_jsonl = self.project_root / "audits" / "ae11_state_reconstruction_audit.jsonl"
        write_mismatch_audit(audit_csv, self.reconstruction_events, jsonl_path=audit_jsonl)

        # 3) Authoritative Decimal ledger from SQLite economics (do not reset to starting if opens exist)
        self.ledger_snapshot = reconstruct_ledger_from_sqlite(
            self.state_db,
            starting_balance_usd=self.config.starting_balance_usd,
            entry_fee_bps=self.config.resolved_entry_fee_bps(),
            exit_fee_bps=self.config.resolved_exit_fee_bps(),
            slippage_bps=self.config.slippage_bps,
        )
        write_ledger_consistency_audit(
            self.project_root / "audits" / "ae11_ledger_consistency_audit.csv",
            loop_run_id=self.run_context.loop_run_id,
            invocation_id=self.run_context.invocation_id,
            snapshot=self.ledger_snapshot,
        )
        self.ledger.account.cash_balance_usd = float(self.ledger_snapshot.cash_balance)
        self.ledger.account.realized_pnl_usd = float(self.ledger_snapshot.realized_pnl_usd)
        self._run_ledger_invariants(stage="startup_reconstruction")

        # AE11G: ghost-lock reconcile + position-state semantics (pre-loop)
        startup_semantics = audit_position_state_semantics(
            self.state_db,
            loop_run_id=self.run_context.loop_run_id,
            invocation_id=self.run_context.invocation_id,
            max_open_positions=self.config.max_open_positions,
            open_snapshot_rows=None,
            cumulative_metrics_open_positions=None,
            repair_ghost_locks=True,
            project_root=self.project_root,
        )
        self.position_state_semantics = startup_semantics.to_meta()

        sqlite_positions = (
            self.state_db.get_open_positions()
            if hasattr(self.state_db, "get_open_positions")
            else self.state_db.load_active_positions()
        )
        enrichment: list[dict[str, Any]] = list(sqlite_positions)
        try:
            from app.decision.persistence import read_jsonl_records_safe

            paper_dir = self.project_root / "data" / "paper_trading"
            for path in sorted(paper_dir.glob("paper_orders_*.jsonl")):
                recs, _ = read_jsonl_records_safe(path)
                enrichment.extend([r for r in recs if isinstance(r, dict)])
            for path in sorted(paper_dir.glob("paper_positions_*.jsonl")):
                recs, _ = read_jsonl_records_safe(path)
                enrichment.extend([r for r in recs if isinstance(r, dict)])
        except Exception:
            pass
        self.ledger.positions = sqlite_positions_to_paper_positions(sqlite_positions, enrichment)

        self.session_iteration = 0
        self.metrics = RunMetrics()

        self.last_reconstruction_at = utc_now_iso()
        db_status = self.state_db.status_summary()

        return {
            "state_reconstruction_status": self.reconstruction_status,
            "reconstruction_mismatch_count": sum(1 for e in self.reconstruction_events if e.mismatch_detected),
            "reconstruction_repair_count": len(repaired),
            "reconstruction_warning_count": sum(
                1 for e in self.reconstruction_events if e.severity == "WARNING"
            ),
            "state_source_of_truth": "sqlite",
            "cash_balance": float(self.ledger_snapshot.cash_balance),
            "open_cost_basis_usd": float(self.ledger_snapshot.open_cost_basis_usd),
            "open_market_value_usd": float(self.ledger_snapshot.open_market_value_usd),
            "realized_pnl_usd": float(self.ledger_snapshot.realized_pnl_usd),
            "unrealized_pnl_usd": float(self.ledger_snapshot.unrealized_pnl_usd),
            "account_equity_usd": float(self.ledger_snapshot.account_equity_usd),
            "ledger_consistency_status": self.ledger_snapshot.ledger_consistency_status,
            "fee_model_status": self.ledger_snapshot.fee_model_status,
            "entry_fee_bps": self.config.resolved_entry_fee_bps(),
            "exit_fee_bps": self.config.resolved_exit_fee_bps(),
            "slippage_bps": self.config.slippage_bps,
            "open_positions": db_status.get("open_positions_count")
            or db_status.get("active_open_positions", 0),
            "in_memory_open_positions": len(self.ledger.positions),
            "processed_decisions": db_status.get("processed_decisions_count", 0),
            "checkpoint_loaded": checkpoint is not None,
            "explicit_resume_requested": explicit_resume,
            "migration_status": self.migration_summary.get("status"),
            "backfill_success_count": self.backfill_summary.get("backfill_success_count"),
            "backfill_missing_count": self.backfill_summary.get("backfill_missing_count"),
            "backfill_partial_count": self.backfill_summary.get("backfill_partial_count"),
            **self.position_state_semantics,
            **self.run_context.to_dict(),
        }

    def run_iteration(
        self,
        *,
        candidates_override: list[tuple] | None = None,
        writers: IterationWriters | None = None,
    ) -> dict[str, Any]:
        """Execute one loop iteration."""
        self.session_iteration += 1
        started_at = utc_now_iso()
        start_ts = time.monotonic()
        errors: list[str] = []
        iteration_capture: list[Any] = []
        candidates: list = []
        refresh: dict[str, Any] = {}
        missed: list[dict[str, Any]] = []

        own_writers = writers is None
        if writers is None:
            writers = IterationWriters()

        iter_stats = {
            "strict_decisions_blocked": 0,
            "exploration_decisions_paper_buy": 0,
            "exploration_decisions_blocked_by_hard_safety": 0,
            "unprocessed_decisions_found": 0,
            "no_new_decisions_available": False,
        }
        source_load = None
        price_audit_rows: list[dict[str, Any]] = []

        try:
            ae6_path = discover_ae6_path(self.project_root)
            if ae6_path:
                cursor = self.state_db.get_source_cursor(SOURCE_NAME, str(ae6_path.resolve()))
                if cursor.get("source_file") and cursor.get("source_file") != str(ae6_path.resolve()):
                    self.state_db.reset_source_cursor(SOURCE_NAME, str(ae6_path.resolve()))

            source_load = load_unprocessed_batch(
                project_root=self.project_root,
                state_db=self.state_db,
                batch_size=self.config.max_records_per_iteration,
                max_scan_records=self.config.max_scan_records_per_iteration,
                source_path=ae6_path,
            )

            if (
                not candidates_override
                and source_load.no_new_decisions_available
                and source_load.eof_reached
                and ae6_path
                and self.state_db.processed_count() < max(source_load.records_seen, 1)
            ):
                self.state_db.reset_source_cursor(SOURCE_NAME, str(ae6_path.resolve()))
                source_load = load_unprocessed_batch(
                    project_root=self.project_root,
                    state_db=self.state_db,
                    batch_size=self.config.max_records_per_iteration,
                    max_scan_records=self.config.max_scan_records_per_iteration,
                    source_path=ae6_path,
                )
                source_load.decision_source_status = "BACKLOG_REWIND_RESCAN"

            self.last_source_load = source_load
            candidates = candidates_override or source_load.candidates
            iter_stats["unprocessed_decisions_found"] = source_load.records_selected_for_processing
            iter_stats["no_new_decisions_available"] = source_load.no_new_decisions_available

            if (
                not candidates_override
                and source_load.no_new_decisions_available
                and source_load.eof_reached
            ):
                refresh = generate_fresh_ae6_decisions(
                    self.project_root, limit=self.config.max_records_per_iteration
                )
                source_load.fresh_generation_attempted = True
                source_load.fresh_generation_result = refresh.get("fresh_generation_result")
                if refresh.get("decisions_generated", 0) > 0:
                    ae6_path = discover_ae6_path(self.project_root)
                    source_load = load_unprocessed_batch(
                        project_root=self.project_root,
                        state_db=self.state_db,
                        batch_size=self.config.max_records_per_iteration,
                        max_scan_records=self.config.max_scan_records_per_iteration,
                        source_path=ae6_path,
                    )
                    self.last_source_load = source_load
                    candidates = source_load.candidates
                    iter_stats["unprocessed_decisions_found"] = source_load.records_selected_for_processing
                    iter_stats["no_new_decisions_available"] = source_load.no_new_decisions_available
            else:
                refresh = {"provider": self.config.provider, "decisions_generated": 0}

            conn = None
            try:
                from scripts.diagnostics._common import open_db_readonly

                conn = open_db_readonly()
            except FileNotFoundError:
                pass

            for decision, context, audit in candidates:
                self.metrics.session.decisions_seen += 1
                trace = build_traceability_record(decision, context, audit)
                source_id = trace.source_decision_id
                source_event_key = build_source_event_key(decision) if decision else None

                if source_id and self.state_db.is_decision_processed(
                    str(source_id), source_event_key=source_event_key
                ):
                    self.metrics.session.duplicates_skipped += 1
                    continue

                identity = (decision or {}).get("candidate_identity") or {}
                pair_address = identity.get("pair_address") or (context or {}).get("pair_address") or ""
                coin_id = identity.get("coin_id")
                decision_created_at = (decision or {}).get("created_at_utc")
                order_created_at = utc_now_iso()

                price_result = self.price_oracle.lookup_price(
                    coin_id=coin_id,
                    pair_address=pair_address or trace.candidate_id,
                    order_created_at_utc=order_created_at,
                    decision_created_at_utc=decision_created_at,
                    conn=conn,
                )
                price_dict = price_result.to_dict()
                price_freshness = evaluate_price_freshness(
                    price_dict,
                    decision=decision,
                    exploration_max_price_age_seconds=self.config.max_price_age_seconds,
                    strict_max_price_age_seconds=self.config.strict_shadow_max_price_age_seconds,
                    loop_observed_at_utc=order_created_at,
                )
                price_audit_rows.append(
                    price_freshness.to_audit_dict(
                        loop_run_id=self.loop_run_id,
                        iteration=self.iteration,
                        source_decision_id=str(source_id) if source_id else None,
                        source_event_key=source_event_key,
                        pair_address=pair_address or None,
                    )
                )
                trace_dict = trace.to_dict()

                strict = evaluate_strict_shadow_decision(
                    decision=decision,
                    traceability=trace_dict,
                    price_freshness=price_freshness,
                    open_position_count=self.state_db.count_open_positions()
                    if hasattr(self.state_db, "count_open_positions")
                    else self.state_db.active_open_position_count(),
                    max_open_positions=self.config.max_open_positions,
                    has_active_pair_lock=self.state_db.has_active_pair_lock(pair_address) if pair_address else False,
                    cooldown_active=self.state_db.is_cooldown_active(pair_address) if pair_address else False,
                    already_processed=False,
                    missing_identity=not trace.candidate_id,
                )
                if strict.strict_mode_decision == "BLOCKED":
                    iter_stats["strict_decisions_blocked"] += 1

                policy = evaluate_exploration_decision(
                    strict,
                    decision=decision,
                    exploration_mode=self.config.exploration_mode,
                    enable_paper_demo_orders=self.config.enable_paper_demo_orders,
                    allow_paper_trades_with_audit_blockers=self.config.allow_paper_trades_with_audit_blockers,
                    no_real_wallet=self.config.no_real_wallet,
                    price_freshness=price_freshness,
                    traceability=trace_dict,
                    pair_address=pair_address,
                    has_active_pair_lock=self.state_db.has_active_pair_lock(pair_address)
                    if pair_address
                    else False,
                    cooldown_active=self.state_db.is_cooldown_active(pair_address)
                    if pair_address
                    else False,
                    max_open_positions_hit=(
                        self.state_db.count_open_positions()
                        if hasattr(self.state_db, "count_open_positions")
                        else self.state_db.active_open_position_count()
                    )
                    >= self.config.max_open_positions,
                    missing_identity=not trace.candidate_id,
                    already_processed=False,
                )
                if policy.should_trade_exploration:
                    iter_stats["exploration_decisions_paper_buy"] += 1
                elif policy.exploration_reason and not policy.paper_policy_prevented:
                    iter_stats["exploration_decisions_blocked_by_hard_safety"] += 1

                paper_order_id = None
                position_id = None
                paper_action = "NO_TRADE"

                if policy.should_trade_exploration and self.ledger_economic_mutations_blocked:
                    self.metrics.session.rejected_candidates += 1
                    paper_action = "REJECTED_LEDGER_INVARIANT"
                elif policy.should_trade_exploration:
                    exec_price_dict = dict(price_dict)
                    if price_freshness.exploration_price_fresh:
                        exec_price_dict["price_status"] = "PRICE_OK"

                    intent = OrderIntent(
                        candidate_id=trace.candidate_id,
                        symbol=identity.get("symbol") or (context or {}).get("symbol") or "",
                        pair_address=pair_address,
                        notional_usd=self.config.notional_usd,
                        requested_price_usd=price_result.price,
                        source_decision_id=trace.source_decision_id,
                        source_context_record_id=trace.source_context_record_id,
                        source_llm_audit_record_id=trace.source_llm_audit_record_id,
                        decision_status=trace.decision_status,
                        consensus_family=trace.consensus_family,
                        context_schema_id=trace.context_schema_id,
                        audit_verdict=trace.audit_verdict,
                        audit_blockers=trace.audit_blockers,
                        audit_warnings=trace.audit_warnings,
                        scoring_policy_id=trace.scoring_policy_id,
                        decision_created_at_utc=decision_created_at,
                        order_created_at_utc=order_created_at,
                        coin_id=coin_id,
                    )

                    allow_blockers = (
                        self.config.allow_paper_trades_with_audit_blockers and self.config.exploration_mode
                    )
                    result = self.paper_adapter.execute(
                        intent,
                        traceability=trace_dict,
                        price_result=exec_price_dict,
                        allow_audit_blockers=allow_blockers,
                    )

                    order_record = result.record or {}
                    order_record["override_type"] = policy.override_type or EXPLORATION_OVERRIDE_TYPE
                    order_record["not_model_approved"] = True
                    order_record["not_live_approved"] = True
                    order_record["trade_authority"] = policy.trade_authority
                    order_record["original_no_trade_authority"] = bool(
                        (decision or {}).get("no_trade_authority", True)
                    )

                    order_record.update(
                        {
                            "record_type": "PAPER_ORDER",
                            "schema_version": AE11_SCHEMA_VERSION,
                            "loop_run_id": self.loop_run_id,
                            "loop_iteration": self.iteration,
                            "source_decision_id": source_id,
                            "source_event_key": source_event_key,
                            "candidate_id": trace.candidate_id,
                            "pair_address": pair_address,
                            "strict_shadow_decision": policy.strict_shadow_decision,
                            "exploration_decision": policy.exploration_decision,
                        }
                    )
                    writers.paper_orders.append_dict(order_record)
                    self.metrics.session.orders_created += 1

                    if result.success:
                        self.metrics.session.fills_created += 1
                        self.metrics.session.positions_opened += 1
                        paper_action = "FILLED"
                        paper_order_id = result.order_id
                        pos = self.ledger.positions[-1] if self.ledger.positions else None
                        if pos:
                            position_id = pos.position_id
                            pos_record = {
                                **pos.to_dict(),
                                "record_type": "PAPER_POSITION",
                                "schema_version": AE11_SCHEMA_VERSION,
                                "loop_run_id": self.loop_run_id,
                                "loop_iteration": self.iteration,
                            }
                            writers.paper_positions.append_dict(pos_record)
                            if pair_address:
                                from app.runtime_paper_loop.decimal_money import (
                                    bps_cost,
                                    decimal_to_str,
                                    quantize_price,
                                    quantize_quantity,
                                    quantize_usd,
                                )
                                from datetime import timedelta as _td

                                entry_px = quantize_price(pos.entry_price_usd)
                                qty = quantize_quantity(pos.quantity)
                                notional = quantize_usd(pos.notional_usd)
                                e_fee = bps_cost(notional, self.config.resolved_entry_fee_bps())
                                e_slip = bps_cost(notional, self.config.slippage_bps)
                                cash_debited = quantize_usd(notional + e_fee + e_slip)
                                # Adjust ledger for fees beyond notional already debited by apply_fill
                                self.ledger.account.cash_balance_usd = float(
                                    quantize_usd(
                                        self.ledger.account.cash_balance_usd
                                    )
                                    - e_fee
                                    - e_slip
                                )
                                opened_dt = datetime.now(timezone.utc)
                                try:
                                    opened_dt = datetime.fromisoformat(
                                        pos.opened_at_utc.replace("Z", "+00:00")
                                    )
                                    if opened_dt.tzinfo is None:
                                        opened_dt = opened_dt.replace(tzinfo=timezone.utc)
                                except Exception:
                                    pass
                                time_stop_at = (
                                    opened_dt
                                    + _td(minutes=float(self.config.time_stop_minutes))
                                ).isoformat()
                                tp = quantize_price(
                                    float(entry_px)
                                    * (1 + float(self.config.take_profit_pct) / 100.0)
                                )
                                sl = quantize_price(
                                    float(entry_px)
                                    * (1 - float(self.config.stop_loss_pct) / 100.0)
                                )
                                self.state_db.register_position(
                                    pos.position_id,
                                    pair_address,
                                    paper_order_id=result.order_id or "",
                                    source_decision_id=str(source_id) if source_id else None,
                                    opened_at_utc=pos.opened_at_utc,
                                    economics={
                                        "candidate_id": pos.candidate_id,
                                        "symbol": pos.symbol,
                                        "entry_price": decimal_to_str(entry_px),
                                        "entry_price_timestamp_utc": price_freshness.price_timestamp_utc,
                                        "entry_price_source": "fill",
                                        "entry_snapshot_id": str(price_freshness.source_snapshot_id)
                                        if price_freshness.source_snapshot_id is not None
                                        else None,
                                        "notional_usd": decimal_to_str(notional),
                                        "quantity": decimal_to_str(qty),
                                        "cost_basis_usd": decimal_to_str(
                                            quantize_usd(notional + e_fee)
                                        ),
                                        "entry_fee_usd": decimal_to_str(e_fee),
                                        "entry_slippage_usd": decimal_to_str(e_slip),
                                        "cash_debited_usd": decimal_to_str(cash_debited),
                                        "tp_price": decimal_to_str(tp),
                                        "sl_price": decimal_to_str(sl),
                                        "time_stop_at_utc": time_stop_at,
                                        "take_profit_pct": str(self.config.take_profit_pct),
                                        "stop_loss_pct": str(self.config.stop_loss_pct),
                                        "time_stop_minutes": str(self.config.time_stop_minutes),
                                        "trade_authority": policy.trade_authority,
                                        "not_model_approved": "True",
                                        "not_live_approved": "True",
                                        "override_type": policy.override_type
                                        or EXPLORATION_OVERRIDE_TYPE,
                                        "economic_enrichment_status": "FULL",
                                        "economic_enrichment_missing_fields": "",
                                        "open_market_value_usd": decimal_to_str(notional),
                                        "unrealized_pnl_usd": "0",
                                    },
                                )
                                cooldown_until = (
                                    datetime.now(timezone.utc)
                                    + timedelta(minutes=self.config.per_pair_cooldown_minutes)
                                ).isoformat()
                                self.state_db.set_cooldown(pair_address, cooldown_until)
                                self._run_ledger_invariants(stage="after_paper_buy_fill")
                    else:
                        self.metrics.session.rejected_candidates += 1
                        paper_action = "REJECTED"

                    if self.config.enable_live_dry_run:
                        live_result = self.live_adapter.execute(intent)
                        live_record = live_result.record or {}
                        live_record.update(
                            {
                                "record_type": "LIVE_DRY_RUN_ORDER",
                                "schema_version": AE11_SCHEMA_VERSION,
                                "loop_run_id": self.loop_run_id,
                                "loop_iteration": self.iteration,
                                "wallet_configured": False,
                                "private_key_accessed": False,
                                "real_transaction_attempted": False,
                                "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
                            }
                        )
                        writers.live_dry_run.append_dict(live_record)
                else:
                    self.metrics.session.rejected_candidates += 1

                if source_id:
                    self.state_db.mark_decision_processed(
                        str(source_id),
                        action_taken=paper_action,
                        source_event_key=source_event_key,
                        strict_decision=policy.strict_mode_decision,
                        exploration_decision=policy.exploration_mode_decision,
                        paper_order_id=paper_order_id,
                        candidate_id=trace.candidate_id,
                        pair_address=pair_address or None,
                        event_timestamp=decision_created_at,
                        source_snapshot_id=str(price_freshness.source_snapshot_id)
                        if price_freshness.source_snapshot_id is not None
                        else None,
                        source_snapshot_timestamp=price_freshness.price_timestamp_utc,
                    )
                    self.metrics.session.new_decisions_processed += 1
                    self.latest_decision_cursor = str(source_id)
                    self.latest_decision_timestamp = decision_created_at

                capture = build_opportunity_capture_record(
                    decision=decision,
                    context=context,
                    audit=audit,
                    traceability=trace_dict,
                    policy=policy,
                    price_result=price_dict,
                    loop_run_id=self.loop_run_id,
                    loop_iteration=self.iteration,
                    paper_order_id=paper_order_id,
                    position_id=position_id,
                    paper_action_taken=paper_action,
                )
                capture = compute_forward_returns_no_lookahead(
                    capture,
                    price_oracle=self.price_oracle,
                    coin_id=coin_id,
                    pair_address=pair_address,
                )
                writers.opportunity_capture.append_dict(capture.to_dict())
                iteration_capture.append(capture)
                self.capture_records.append(capture)

                trade_decision = build_hierarchical_trade_decision(
                    source_decision_id=str(source_id) if source_id else None,
                    source_event_key=source_event_key,
                    candidate_id=trace.candidate_id,
                    pair_address=pair_address or None,
                    loop_run_id=self.loop_run_id,
                    loop_iteration=self.iteration,
                    policy=policy,
                    price_freshness=price_freshness,
                    paper_action=paper_action,
                    paper_order_id=paper_order_id,
                    decision=decision,
                )
                writers.trade_decisions.append_dict(trade_decision)

            if conn:
                conn.close()

            lifecycle = _manage_open_positions(
                self.ledger,
                self.price_oracle,
                self.state_db,
                writers,
                config=self.config,
                loop_run_id=self.loop_run_id,
                iteration=self.iteration,
                invocation_id=self.run_context.invocation_id if self.run_context else "",
                project_root=self.project_root,
                valuation_oracle=self.valuation_oracle,
            )
            closed = int(lifecycle.get("positions_closed", 0))
            self.metrics.session.positions_closed += closed
            if closed > 0 and not self.ledger_economic_mutations_blocked:
                self._run_ledger_invariants(stage="after_economic_close")
            self.lifecycle_session["positions_closed"] = (
                self.lifecycle_session.get("positions_closed", 0) + closed
            )
            self.lifecycle_session["positions_evaluated"] = (
                self.lifecycle_session.get("positions_evaluated", 0)
                + int(lifecycle.get("positions_evaluated", 0))
            )
            self.lifecycle_session["positions_blocked"] = (
                self.lifecycle_session.get("positions_blocked", 0)
                + int(lifecycle.get("positions_blocked", 0))
            )
            self.lifecycle_session["duplicate_close_attempt_count"] = (
                self.lifecycle_session.get("duplicate_close_attempt_count", 0)
                + int(lifecycle.get("duplicate_close_attempt_count", 0))
            )
            for reason, cnt in (lifecycle.get("exit_reasons") or {}).items():
                er = self.lifecycle_session.setdefault("exit_reasons", {})
                er[reason] = er.get(reason, 0) + int(cnt)
            self.lifecycle_session["lifecycle_noop_reason"] = lifecycle.get(
                "lifecycle_noop_reason"
            )
            self.lifecycle_session["lifecycle_audit_status"] = lifecycle.get(
                "lifecycle_audit_status"
            )
            for k in (
                "tp_trigger_count",
                "sl_trigger_count",
                "time_stop_trigger_count",
                "price_based_positions_closed",
                "price_oracle_status",
                "mark_to_market_status",
                "tp_sl_lifecycle_status",
                "no_lookahead_status",
                "temporal_validity_status",
                "no_double_count_status",
                "valuation_source",
                "price_positions_evaluated",
                "price_positions_resolved",
                "price_positions_missing",
                "price_positions_stale",
                "price_positions_pre_entry_stale",
                "price_positions_fallback",
                "price_positions_deterministic",
                "price_unrealized_pnl_usd",
                "total_unrealized_after_cost_pnl_usd",
            ):
                if k in lifecycle:
                    if k.endswith("_count") or k.startswith("price_positions_") or k in (
                        "price_based_positions_closed",
                    ):
                        self.valuation_session[k] = int(
                            self.valuation_session.get(k, 0)
                        ) + int(lifecycle.get(k) or 0)
                    elif k in (
                        "price_unrealized_pnl_usd",
                        "total_unrealized_after_cost_pnl_usd",
                    ):
                        # Last lifecycle pass is authoritative open-book totals
                        self.valuation_session[k] = float(lifecycle.get(k) or 0)
                    else:
                        self.valuation_session[k] = lifecycle.get(k)
            self.valuation_session["price_oracle_audit_rows"] = len(
                getattr(self.valuation_oracle, "audit_log", []) or []
            )
            self.valuation_session["mark_to_market_audit_rows"] = len(
                getattr(self.valuation_oracle, "mtm_audit_log", []) or []
            )
            self.valuation_session["tp_sl_trigger_audit_rows"] = len(
                getattr(self.valuation_oracle, "tp_sl_audit_log", []) or []
            )

            missed = update_missed_winners(iteration_capture)
            for mw in missed:
                writers.missed_winners.append_dict(mw)
            self.all_missed_winners.extend(missed)

            if source_load:
                self.decision_source_audits.append(
                    {
                        "loop_run_id": self.loop_run_id,
                        "iteration": self.iteration,
                        "ae6_source_file": source_load.ae6_source_file,
                        "ae6_file_length": source_load.ae6_file_length,
                        "ae6_file_last_write_time": source_load.ae6_file_last_write_time,
                        "records_scanned": source_load.records_scanned,
                        "records_seen": source_load.records_seen,
                        "records_skipped_processed": source_load.records_skipped_processed,
                        "records_skipped_invalid": source_load.records_skipped_invalid,
                        "records_selected_for_processing": source_load.records_selected_for_processing,
                        "eof_reached": source_load.eof_reached,
                        "cursor_before": source_load.cursor_before,
                        "cursor_after": source_load.cursor_after,
                        "latest_record_created_at_utc": source_load.latest_record_created_at_utc,
                        "fresh_generation_attempted": source_load.fresh_generation_attempted,
                        "fresh_generation_result": source_load.fresh_generation_result,
                        "no_new_decisions_available": source_load.no_new_decisions_available,
                        "decision_source_status": source_load.decision_source_status,
                    }
                )
            self.price_freshness_audits.extend(price_audit_rows)

        except Exception as exc:
            errors.append(str(exc))
            errors.append(traceback.format_exc())

        flush_results = writers.flush_and_fsync_all()
        completed_at = utc_now_iso()
        duration = time.monotonic() - start_ts

        event = {
            "record_type": "RUNTIME_EVENT" if not errors else "RUNTIME_ERROR",
            "schema_version": AE11_SCHEMA_VERSION,
            "loop_run_id": self.loop_run_id,
            "loop_iteration": self.iteration,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "duration_seconds": round(duration, 3),
            "decisions_seen": len(candidates),
            **self.metrics.session.to_dict(),
            "missed_winners_updated": len(missed),
            "decision_source_status": source_load.decision_source_status if source_load else None,
            "records_scanned": source_load.records_scanned if source_load else 0,
            "unprocessed_decisions_found": iter_stats["unprocessed_decisions_found"],
            "no_new_decisions_available": iter_stats["no_new_decisions_available"],
            "source_cursor_before": source_load.cursor_before if source_load else None,
            "source_cursor_after": source_load.cursor_after if source_load else None,
            "strict_decisions_blocked": iter_stats["strict_decisions_blocked"],
            "exploration_decisions_paper_buy": iter_stats["exploration_decisions_paper_buy"],
            "exploration_decisions_blocked_by_hard_safety": iter_stats[
                "exploration_decisions_blocked_by_hard_safety"
            ],
            "state_reconstruction_status": self.reconstruction_status if self.session_iteration == 1 else None,
            "writer_flush_status": all(r.get("fsynced") for r in flush_results.values()),
            "writer_fsync_status": flush_results,
            "checkpoint_status": "pending",
            "idempotency_index_status": self.state_db.status_summary(),
            "errors": errors,
            "refresh_status": refresh,
        }

        if self.iteration % self.config.heartbeat_every_n_iterations == 0:
            event["record_type"] = "HEARTBEAT"
            event["cash_balance"] = self.ledger.account.cash_balance_usd
            event["open_position_count"] = self.state_db.active_open_position_count()
            event["active_pair_lock_count"] = self.state_db.status_summary().get("active_pair_locks", 0)

        writers.runtime_events.append_dict(event)
        writers.runtime_events.flush_and_fsync()

        checkpoint_payload = build_checkpoint_payload(
            loop_run_id=self.loop_run_id,
            invocation_id=self.run_context.invocation_id if self.run_context else str(uuid4()),
            audit_root=str(self.run_context.audit_root) if self.run_context else "",
            last_completed_iteration=self.session_iteration,
            session_iterations_completed=self.session_iteration,
            active_position_ids=[p.position_id for p in self.ledger.positions if p.status == "OPEN"],
            active_pair_keys=list(self.state_db.load_cooldowns().keys()),
            cash_balance=self.ledger.account.cash_balance_usd,
            reserved_cash=sum(p.notional_usd for p in self.ledger.positions if p.status == "OPEN"),
            latest_processed_decision_cursor=self.latest_decision_cursor,
            latest_processed_decision_timestamp=self.latest_decision_timestamp,
            idempotency_index_status=self.state_db.status_summary(),
            output_file_paths_for_current_day=writers.output_paths(),
            ae6_source_file=source_load.ae6_source_file if source_load else None,
            ae6_source_cursor_type="byte_offset",
            ae6_source_cursor_value=source_load.cursor_after if source_load else None,
            ae6_last_seen_decision_id=self.latest_decision_cursor,
            ae6_last_seen_created_at_utc=self.latest_decision_timestamp,
            ae6_eof_reached_at_utc=utc_now_iso() if source_load and source_load.eof_reached else None,
            last_state_reconstruction_at_utc=self.last_reconstruction_at,
            explicit_resume_requested=bool(
                self.config.resume_loop_run_id or self.config.resume_latest_loop
            ),
            run_started_at_utc=self.run_context.run_started_at_utc if self.run_context else None,
        )
        write_checkpoint(checkpoint_payload)
        self.state_db.save_loop_checkpoint(
            self.loop_run_id, self.session_iteration, str(checkpoint_payload)
        )
        event["checkpoint_status"] = "written"

        if own_writers:
            writers.close_all()

        return event

    def run_loop(self) -> dict[str, Any]:
        """Run loop until duration expires."""
        startup_info = self.startup()
        assert self.run_context is not None
        self.metrics.session.iterations_completed = 0
        deadline = time.monotonic() + self.config.duration_minutes * 60.0
        iteration_results: list[dict[str, Any]] = []

        while time.monotonic() < deadline:
            result = self.run_iteration()
            iteration_results.append(result)
            self.metrics.session.iterations_completed = self.session_iteration
            if result.get("errors"):
                break
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sleep_for = min(self.config.loop_interval_seconds, remaining)
            time.sleep(sleep_for)

        audit_root = self.run_context.audit_root
        db_status = self.state_db.status_summary()
        source_cursor_value = None
        ae6_path = discover_ae6_path(self.project_root)
        if ae6_path:
            cursor = self.state_db.get_source_cursor(SOURCE_NAME, str(ae6_path.resolve()))
            source_cursor_value = cursor.get("cursor_value")

        from app.runtime_paper_loop.ledger_accounting import (
            reconstruct_ledger_from_sqlite,
            write_ledger_consistency_audit,
        )

        # Ensure latest invocation has lifecycle audit rows even if an iteration
        # errored before _manage_open_positions ran.
        if self.lifecycle_session.get("lifecycle_audit_status") is None:
            from app.runtime_paper_loop.persistence import IterationWriters
            from app.runtime_paper_loop.position_lifecycle import evaluate_and_close_positions

            end_lifecycle = evaluate_and_close_positions(
                self.state_db,
                self.price_oracle,
                IterationWriters(),
                config=self.config,
                loop_run_id=self.loop_run_id,
                invocation_id=self.run_context.invocation_id,
                iteration=self.session_iteration or 0,
                project_root=self.project_root,
                ledger=self.ledger,
                valuation_oracle=self.valuation_oracle,
            )
            self.lifecycle_session["positions_evaluated"] = int(
                end_lifecycle.get("positions_evaluated", 0)
            )
            self.lifecycle_session["positions_closed"] = (
                self.lifecycle_session.get("positions_closed", 0)
                + int(end_lifecycle.get("positions_closed", 0))
            )
            self.lifecycle_session["positions_blocked"] = int(
                end_lifecycle.get("positions_blocked", 0)
            )
            self.lifecycle_session["duplicate_close_attempt_count"] = (
                self.lifecycle_session.get("duplicate_close_attempt_count", 0)
                + int(end_lifecycle.get("duplicate_close_attempt_count", 0))
            )
            self.lifecycle_session["lifecycle_noop_reason"] = end_lifecycle.get(
                "lifecycle_noop_reason"
            )
            self.lifecycle_session["lifecycle_audit_status"] = end_lifecycle.get(
                "lifecycle_audit_status"
            )
            self.metrics.session.positions_closed += int(
                end_lifecycle.get("positions_closed", 0)
            )

        self.ledger_snapshot = reconstruct_ledger_from_sqlite(
            self.state_db,
            starting_balance_usd=self.config.starting_balance_usd,
            entry_fee_bps=self.config.resolved_entry_fee_bps(),
            exit_fee_bps=self.config.resolved_exit_fee_bps(),
            slippage_bps=self.config.slippage_bps,
        )
        write_ledger_consistency_audit(
            self.project_root / "audits" / "ae11_ledger_consistency_audit.csv",
            loop_run_id=self.loop_run_id,
            invocation_id=self.run_context.invocation_id,
            snapshot=self.ledger_snapshot,
            notes="end-of-run ledger reconstruction",
        )
        self.ledger.account.cash_balance_usd = float(self.ledger_snapshot.cash_balance)
        self._run_ledger_invariants(stage="end_of_run_report")

        from app.runtime_paper_loop.equity_bridge import (
            build_equity_bridge,
            write_account_equity_bridge_summary,
            write_equity_bridge_audit,
        )

        self.equity_bridge_result = build_equity_bridge(
            self.state_db,
            starting_balance_usd=self.config.starting_balance_usd,
            loop_run_id=self.loop_run_id,
            invocation_id=self.run_context.invocation_id,
            cash_balance_override=self.ledger_snapshot.cash_balance,
            account_equity_override=self.ledger_snapshot.account_equity_usd,
        )
        write_equity_bridge_audit(
            self.project_root / "audits" / "ae11_equity_bridge_audit.csv",
            loop_run_id=self.loop_run_id,
            invocation_id=self.run_context.invocation_id,
            bridge=self.equity_bridge_result,
        )
        write_account_equity_bridge_summary(
            self.project_root / "data" / "ae11_account_equity_bridge_summary.csv",
            loop_run_id=self.loop_run_id,
            invocation_id=self.run_context.invocation_id,
            bridge=self.equity_bridge_result,
        )

        closed_total = 0
        try:
            row = self.state_db._conn.execute(
                "SELECT COUNT(*) AS cnt FROM closed_positions"
            ).fetchone()
            closed_total = int(row["cnt"]) if row else 0
        except Exception:
            try:
                row = self.state_db._conn.execute(
                    "SELECT COUNT(*) AS cnt FROM active_positions WHERE status = 'CLOSED'"
                ).fetchone()
                closed_total = int(row["cnt"]) if row else 0
            except Exception:
                closed_total = 0

        cumulative_metrics = self.metrics.cumulative_extended(
            self.state_db,
            cash_balance=float(self.ledger_snapshot.cash_balance),
            source_cursor_value=source_cursor_value,
        )
        cumulative_metrics["closed_positions_total"] = closed_total
        cumulative_metrics["open_cost_basis_usd"] = float(self.ledger_snapshot.open_cost_basis_usd)
        cumulative_metrics["open_market_value_usd"] = float(self.ledger_snapshot.open_market_value_usd)
        cumulative_metrics["realized_pnl_usd"] = float(self.ledger_snapshot.realized_pnl_usd)
        cumulative_metrics["unrealized_pnl_usd"] = float(self.ledger_snapshot.unrealized_pnl_usd)
        cumulative_metrics["account_equity_usd"] = float(self.ledger_snapshot.account_equity_usd)

        session = self.metrics.session.to_dict()
        session["session_orders_created"] = session.get("orders_created", 0)
        session["session_fills_created"] = session.get("fills_created", 0)
        session["session_positions_opened"] = session.get("positions_opened", 0)
        session["session_positions_closed"] = session.get("positions_closed", 0)
        bridge_meta = (
            self.equity_bridge_result.to_meta() if self.equity_bridge_result is not None else {}
        )
        ledger_fields = {
            "cash_balance": float(self.ledger_snapshot.cash_balance),
            "starting_balance_usd": float(self.ledger_snapshot.starting_balance_usd),
            "open_cost_basis_usd": float(self.ledger_snapshot.open_cost_basis_usd),
            "open_market_value_usd": float(self.ledger_snapshot.open_market_value_usd),
            "realized_pnl_usd": float(self.ledger_snapshot.realized_pnl_usd),
            "unrealized_pnl_usd": float(self.ledger_snapshot.unrealized_pnl_usd),
            "account_equity_usd": float(self.ledger_snapshot.account_equity_usd),
            "ledger_consistency_status": self.ledger_snapshot.ledger_consistency_status,
            "fee_model_status": self.ledger_snapshot.fee_model_status,
            "entry_fee_bps": self.config.resolved_entry_fee_bps(),
            "exit_fee_bps": self.config.resolved_exit_fee_bps(),
            "slippage_bps": self.config.slippage_bps,
            "backfill_missing_count": self.backfill_summary.get("backfill_missing_count", 0),
            "backfill_success_count": self.backfill_summary.get("backfill_success_count", 0),
            "backfill_partial_count": self.backfill_summary.get("backfill_partial_count", 0),
            "ledger_schema_version": getattr(
                self.ledger_snapshot, "ledger_schema_version", "AE11F_LEDGER_V1"
            ),
            "accounting_model_version": getattr(
                self.ledger_snapshot, "accounting_model_version", "AE11E_DECIMAL_CASH_CREDIT_DEBIT_V1"
            ),
            "ledger_cash_tolerance_usd": float(
                getattr(self.ledger_snapshot, "ledger_cash_tolerance_usd", 0.000001)
            ),
            "invariant_check_method": getattr(
                self.ledger_snapshot,
                "invariant_check_method",
                "quantize_usd_then_abs_diff_le_tolerance",
            ),
            "account_equity_diff": float(
                getattr(self.ledger_snapshot, "account_equity_diff", 0)
            ),
            "cash_diff": float(getattr(self.ledger_snapshot, "cash_diff", 0)),
            "ledger_invariant_status": self.ledger_invariant_status,
            "ledger_invariant_failure_count": self.ledger_invariant_failure_count,
            "ledger_invariant_last_failure_reason": self.ledger_invariant_last_failure_reason,
            "invariant_check_stage": self.invariant_check_stage,
            **bridge_meta,
        }
        lifecycle_fields = {
            "lifecycle_audit_status": self.lifecycle_session.get("lifecycle_audit_status"),
            "lifecycle_positions_evaluated": self.lifecycle_session.get(
                "positions_evaluated", 0
            ),
            "lifecycle_positions_closed": self.lifecycle_session.get("positions_closed", 0),
            "lifecycle_blocked_count": self.lifecycle_session.get("positions_blocked", 0),
            "lifecycle_noop_reason": self.lifecycle_session.get("lifecycle_noop_reason"),
            "duplicate_close_attempt_count": self.lifecycle_session.get(
                "duplicate_close_attempt_count", 0
            ),
        }
        summary = {
            "final_status": (
                Ae11FinalStatus.AE11_LEDGER_INVARIANT_FAILED.value
                if self.ledger_invariant_status == "FAIL"
                else Ae11FinalStatus.AE11_LOOP_OPERATIONAL.value
            ),
            "iterations_completed": self.session_iteration,
            "exploration_mode": self.config.exploration_mode,
            "strict_shadow_mode": self.config.strict_shadow_mode,
            **session,
            "startup": startup_info,
            "current_invocation_counters": True,
            "cumulative_metrics": cumulative_metrics,
            **ledger_fields,
            **lifecycle_fields,
            **self.run_context.to_dict(),
            "current_open_positions_count": cumulative_metrics.get(
                "current_open_positions_count"
            ),
            "canonical_closed_positions_count": cumulative_metrics.get(
                "canonical_closed_positions_count", closed_total
            ),
            "position_registry_total_rows": cumulative_metrics.get(
                "position_registry_total_rows"
            ),
            "paper_order_event_count": cumulative_metrics.get("paper_order_event_count"),
            "paper_fill_event_count": cumulative_metrics.get("paper_fill_event_count"),
            "cumulative_metric_semantics_status": cumulative_metrics.get(
                "cumulative_metric_semantics_status", "PASS"
            ),
            "wallet_configured": False,
            "private_key_accessed": False,
            "real_transaction_attempted": False,
            "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
        }

        # AE11I finalize valuation status for end-of-run reporting
        val_stats = self.valuation_oracle.finalize_session_status()
        self.valuation_session.update(val_stats.to_dict())
        self.valuation_session["price_oracle_audit_rows"] = len(self.valuation_oracle.audit_log)
        self.valuation_session["mark_to_market_audit_rows"] = len(
            self.valuation_oracle.mtm_audit_log
        )
        self.valuation_session["tp_sl_trigger_audit_rows"] = len(
            self.valuation_oracle.tp_sl_audit_log
        )
        summary.update(self.valuation_session)

        from app.runtime_paper_loop.position_state_semantics import (
            audit_position_state_semantics,
        )

        pre_semantics = audit_position_state_semantics(
            self.state_db,
            loop_run_id=self.loop_run_id,
            invocation_id=self.run_context.invocation_id,
            max_open_positions=self.config.max_open_positions,
            open_snapshot_rows=None,
            cumulative_metrics_open_positions=int(
                cumulative_metrics.get("current_open_positions_count")
                or cumulative_metrics.get("open_positions", 0)
            ),
            repair_ghost_locks=True,
            project_root=self.project_root,
        )
        self.position_state_semantics = pre_semantics.to_meta()
        summary.update(self.position_state_semantics)
        summary["capacity_count_source"] = pre_semantics.capacity_count_source

        report_result = write_ae11_reports(
            audit_root=audit_root,
            project_root=self.project_root,
            loop_run_id=self.loop_run_id,
            invocation_id=self.run_context.invocation_id,
            run_context=self.run_context.to_dict(),
            summary=summary,
            capture_records=self.capture_records,
            missed_winners=self.all_missed_winners,
            # Diagnostic only — ReportGenerator fetches authoritative SQLite state independently.
            open_positions=None,
            closed_trades=None,
            ledger=self.ledger,
            state_db=self.state_db,
            cash_balance=self.ledger.account.cash_balance_usd,
            state_db_status=db_status,
            reconstruction_status=self.reconstruction_status,
            reconstruction_events=self.reconstruction_events,
            price_oracle_audit=self.price_oracle.audit_log,
            idempotency_audit={
                "sqlite_index_used": True,
                "historical_jsonl_scan_on_startup": False,
                "duplicate_prevention": "PASS",
                "registry_backend": "sqlite",
            },
            wallet_safety={
                "wallet_configured": False,
                "private_key_accessed": False,
                "real_transaction_attempted": False,
                "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
                "no_real_wallet_path": True,
            },
            decision_source_audits=self.decision_source_audits,
            price_freshness_audits=self.price_freshness_audits,
            exploration_max_price_age_seconds=self.config.max_price_age_seconds,
            strict_max_price_age_seconds=self.config.strict_shadow_max_price_age_seconds,
            equity_bridge=bridge_meta,
            cumulative_metrics=cumulative_metrics,
            valuation_session=self.valuation_session,
            valuation_oracle_audits={
                "price_oracle": list(self.valuation_oracle.audit_log),
                "mark_to_market": list(self.valuation_oracle.mtm_audit_log),
                "tp_sl_trigger": list(self.valuation_oracle.tp_sl_audit_log),
            },
        )
        report_paths = report_result.get("paths", report_result)
        report_consistency_status = report_result.get("report_consistency_status", "SKIPPED")
        open_snapshot_rows = report_result.get("open_positions_snapshot_rows", 0)
        closed_snapshot_rows = report_result.get("closed_trades_snapshot_rows", 0)
        consistency_mismatches = report_result.get("report_consistency_mismatch_count", 0)

        hygiene_fields = {
            "closed_trade_hygiene_status": report_result.get(
                "closed_trade_hygiene_status", "SKIPPED"
            ),
            "canonical_closed_trades_rows": report_result.get(
                "canonical_closed_trades_rows", closed_snapshot_rows
            ),
            "closed_trade_event_history_rows": report_result.get(
                "closed_trade_event_history_rows", 0
            ),
            "invalid_closed_trade_rows": report_result.get("invalid_closed_trade_rows", 0),
            "duplicate_closed_position_rows": report_result.get(
                "duplicate_closed_position_rows", 0
            ),
            "duplicate_close_event_rows": report_result.get("duplicate_close_event_rows", 0),
            "blank_close_event_id_rows": report_result.get("blank_close_event_id_rows", 0),
            "canonical_closed_position_unique_count": report_result.get(
                "canonical_closed_position_unique_count", closed_snapshot_rows
            ),
        }
        # Align closed_positions_total with canonical unique count for report consistency
        canonical_closed = int(
            hygiene_fields.get("canonical_closed_position_unique_count") or closed_total
        )
        cumulative_metrics["closed_positions_total"] = canonical_closed
        cumulative_metrics["canonical_closed_positions_count"] = canonical_closed
        cumulative_metrics["open_positions"] = (
            self.state_db.count_open_positions()
            if hasattr(self.state_db, "count_open_positions")
            else int(cumulative_metrics.get("open_positions", 0))
        )
        cumulative_metrics["current_open_positions_count"] = cumulative_metrics[
            "open_positions"
        ]

        end_semantics = audit_position_state_semantics(
            self.state_db,
            loop_run_id=self.loop_run_id,
            invocation_id=self.run_context.invocation_id,
            max_open_positions=self.config.max_open_positions,
            open_snapshot_rows=open_snapshot_rows,
            cumulative_metrics_open_positions=int(cumulative_metrics.get("open_positions", 0)),
            repair_ghost_locks=True,
            project_root=self.project_root,
        )
        self.position_state_semantics = end_semantics.to_meta()
        semantics_fields = {
            **self.position_state_semantics,
            "capacity_count_source": end_semantics.capacity_count_source,
        }

        self.state_db.close()

        return {
            "phase": AE11_PHASE,
            "loop_run_id": self.loop_run_id,
            "invocation_id": self.run_context.invocation_id,
            "run_started_at_utc": self.run_context.run_started_at_utc,
            "audit_root": str(audit_root),
            "final_status": summary["final_status"],
            "iterations_completed": self.session_iteration,
            "decisions_seen": session["decisions_seen"],
            "new_decisions_processed": session["new_decisions_processed"],
            "duplicates_skipped": session["duplicates_skipped"],
            "orders_created": session["orders_created"],
            "fills_created": session["fills_created"],
            "positions_opened": session["positions_opened"],
            "positions_closed": session["positions_closed"],
            "session_orders_created": session.get("session_orders_created", 0),
            "session_fills_created": session.get("session_fills_created", 0),
            "session_positions_opened": session.get("session_positions_opened", 0),
            "session_positions_closed": session.get("session_positions_closed", 0),
            "current_invocation_counters": True,
            "startup": startup_info,
            "cumulative_metrics": cumulative_metrics,
            "report_paths": report_paths,
            "report_consistency_status": report_consistency_status,
            "open_positions_snapshot_rows": open_snapshot_rows,
            "closed_trades_snapshot_rows": closed_snapshot_rows,
            "report_consistency_mismatch_count": consistency_mismatches,
            "current_open_positions_count": cumulative_metrics.get(
                "current_open_positions_count"
            ),
            "canonical_closed_positions_count": cumulative_metrics.get(
                "canonical_closed_positions_count", canonical_closed
            ),
            "position_registry_total_rows": cumulative_metrics.get(
                "position_registry_total_rows"
            ),
            "paper_order_event_count": cumulative_metrics.get("paper_order_event_count"),
            "paper_fill_event_count": cumulative_metrics.get("paper_fill_event_count"),
            "cumulative_metric_semantics_status": cumulative_metrics.get(
                "cumulative_metric_semantics_status", "PASS"
            ),
            **ledger_fields,
            **lifecycle_fields,
            **hygiene_fields,
            **semantics_fields,
            "wallet_configured": False,
            "private_key_accessed": False,
            "real_transaction_attempted": False,
            "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
            **self.valuation_session,
        }


def run_ae11_runtime_paper_loop(config: Ae11LoopConfig) -> dict[str, Any]:
    """Entry point for AE11 runtime paper loop."""
    if config.no_real_wallet:
        pass
    runner = RuntimePaperLoopRunner(config)
    return runner.run_loop()
