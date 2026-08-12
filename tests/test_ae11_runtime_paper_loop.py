"""Tests for AE11 runtime paper trading loop."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

from app.decision.persistence import read_jsonl_records_safe
from app.paper_trading.price_oracle import DemoPriceOracle
from app.paper_trading.types import PaperOrderStatus, PriceStatus
from app.runtime_paper_loop.checkpointing import load_checkpoint, write_checkpoint
from app.runtime_paper_loop.decision_policy import (
    evaluate_exploration_decision,
    evaluate_strict_shadow_decision,
)
from app.runtime_paper_loop.idempotency import AE11StateDb
from app.runtime_paper_loop.loop_runner import RuntimePaperLoopRunner
from app.runtime_paper_loop.decision_source import build_source_event_key, load_unprocessed_batch
from app.runtime_paper_loop.opportunity_capture import (
    build_opportunity_capture_record,
    compute_forward_returns_no_lookahead,
    is_missed_winner,
)
from app.runtime_paper_loop.price_freshness import evaluate_price_freshness
from app.runtime_paper_loop.trade_decision import build_hierarchical_trade_decision
from app.runtime_paper_loop.persistence import (
    BufferedJsonlWriter,
    atomic_write_json,
    clean_stale_tmp_files,
)
from app.runtime_paper_loop.state_reconstruction import ledger_reconstruction
from app.runtime_paper_loop.types import Ae11LoopConfig, EXPLORATION_OVERRIDE_TYPE, OpportunityCaptureRecord


def _sample_decision(**overrides) -> dict:
    base = {
        "decision_id": "dec-001",
        "created_at_utc": "2026-07-10T09:00:52+00:00",
        "decision_status": "RESEARCH_CANDIDATE",
        "candidate_identity": {
            "candidate_id": "cand-001",
            "pair_address": "0xABC",
            "symbol": "PEPE/WETH",
            "coin_id": 1,
        },
        "market_context": {"price": 0.001, "liquidity": 50000, "whale_score": 0.3},
    }
    base.update(overrides)
    return base


def _sample_context(**overrides) -> dict:
    base = {
        "context_record_id": "ctx-001",
        "candidate_id": "cand-001",
        "pair_address": "0xABC",
        "symbol": "PEPE/WETH",
    }
    base.update(overrides)
    return base


def _sample_audit(**overrides) -> dict:
    base = {
        "audit_record_id": "aud-001",
        "candidate_id": "cand-001",
        "source_decision_id": "dec-001",
        "llm_verdict": "AUDIT_PASS_NO_ACTION",
        "audit_blockers": [],
    }
    base.update(overrides)
    return base


def _fresh_snapshot(order_ts: str, price: float = 0.001) -> dict:
    order_dt = datetime.fromisoformat(order_ts.replace("Z", "+00:00"))
    snap_dt = order_dt - timedelta(seconds=5)
    return {"id": 100, "coin_id": 1, "timestamp": snap_dt.isoformat(), "price": price}


def _price_freshness_ok(**overrides):
    price_dict = {
        "price": 0.001,
        "price_status": "PRICE_OK",
        "snapshot_provider_timestamp": "2026-07-10T09:00:47+00:00",
        "price_timestamp_used": "2026-07-10T09:00:47+00:00",
        "price_snapshot_id": 100,
    }
    price_dict.update(overrides)
    return evaluate_price_freshness(
        price_dict,
        decision=_sample_decision(),
        exploration_max_price_age_seconds=900.0,
        strict_max_price_age_seconds=30.0,
        loop_observed_at_utc="2026-07-10T09:00:52+00:00",
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


@pytest.fixture
def tmp_ae11_env(tmp_path):
    decision_dir = tmp_path / "data" / "decision_records"
    context_dir = tmp_path / "data" / "context_intelligence"
    audit_dir = tmp_path / "data" / "llm_audit"
    runtime_dir = tmp_path / "data" / "runtime_paper_loop" / "state"
    paper_dir = tmp_path / "data" / "paper_trading"
    for d in (decision_dir, context_dir, audit_dir, runtime_dir, paper_dir):
        d.mkdir(parents=True)

    ae6_path = decision_dir / "ae6_decisions_20260711.jsonl"
    ae8_path = context_dir / "ae8_context_features_20260711.jsonl"
    ae9_path = audit_dir / "ae9_llm_audit_records_20260711.jsonl"

    _write_jsonl(ae6_path, [_sample_decision()])
    _write_jsonl(ae8_path, [_sample_context()])
    _write_jsonl(ae9_path, [_sample_audit()])

    return {
        "root": tmp_path,
        "ae6_path": ae6_path,
        "ae8_path": ae8_path,
        "ae9_path": ae9_path,
        "state_db": runtime_dir / "ae11_state.sqlite",
    }


class TestBufferedPersistence:
    def test_buffered_writer_flushes_at_iteration_boundary(self, tmp_path):
        path = tmp_path / "test.jsonl"
        writer = BufferedJsonlWriter(path)
        writer.append_dict({"a": 1})
        writer.append_dict({"a": 2})
        assert not writer.fsync_status
        result = writer.flush_and_fsync()
        assert result["rows_flushed"] == 2
        assert writer.fsync_status
        writer.close()
        records, _ = read_jsonl_records_safe(path)
        assert len(records) == 2

    def test_atomic_checkpoint_write(self, tmp_path):
        ckpt_path = tmp_path / "checkpoint.json"
        payload = {"loop_run_id": "test", "iteration": 1}
        atomic_write_json(ckpt_path, payload)
        assert ckpt_path.is_file()
        loaded = json.loads(ckpt_path.read_text())
        assert loaded["loop_run_id"] == "test"

    def test_recovery_from_stale_tmp_checkpoint(self, tmp_path):
        stale = tmp_path / "ae11_latest_checkpoint.json.tmp"
        stale.write_text('{"partial": true}')
        clean_stale_tmp_files(tmp_path)
        assert not stale.exists()


class TestIdempotency:
    def test_processed_decision_id_sqlite(self, tmp_path):
        db = AE11StateDb(tmp_path / "state.sqlite")
        assert not db.is_decision_processed("dec-001")
        db.mark_decision_processed("dec-001", action_taken="NO_TRADE", candidate_id="cand-001")
        assert db.is_decision_processed("dec-001")
        db.close()

    def test_duplicate_prevention_across_reruns(self, tmp_path):
        db = AE11StateDb(tmp_path / "state.sqlite")
        db.mark_decision_processed("dec-001", action_taken="FILLED", paper_order_id="ord-1")
        db.register_position("pos-1", "0xABC", paper_order_id="ord-1", source_decision_id="dec-001")
        assert db.has_active_pair_lock("0xABC")
        db.close()

    def test_does_not_scan_historical_jsonl_on_startup(self, tmp_ae11_env):
        config = Ae11LoopConfig(project_root=tmp_ae11_env["root"])
        runner = RuntimePaperLoopRunner(config)
        runner.state_db = AE11StateDb(tmp_ae11_env["state_db"])
        with mock.patch("app.runtime_paper_loop.loop_runner.read_jsonl_records_safe") as mock_read:
            runner.startup()
            mock_read.assert_not_called()


class TestDecisionPolicy:
    def test_strict_shadow_blocks_no_trade_authority(self):
        decision = _sample_decision(no_trade_authority=True)
        trace = {
            "candidate_id": "cand-001",
            "source_decision_id": "dec-001",
            "source_context_record_id": "ctx-001",
        }
        pf = _price_freshness_ok()
        strict = evaluate_strict_shadow_decision(
            decision=decision,
            traceability=trace,
            price_freshness=pf,
            open_position_count=0,
            max_open_positions=10,
            has_active_pair_lock=False,
            cooldown_active=False,
            already_processed=False,
            missing_identity=False,
        )
        assert strict.strict_mode_decision == "BLOCKED"
        assert strict.blocked_by_no_trade_authority is True

    def test_exploration_allows_research_candidate_with_no_trade_authority(self):
        decision = _sample_decision(
            no_trade_authority=True,
            consensus={"consensus_family": "NO_MODEL_CONSENSUS_AVAILABLE"},
        )
        trace = {"candidate_id": "cand-001", "source_decision_id": "dec-001"}
        pf = _price_freshness_ok()
        strict = evaluate_strict_shadow_decision(
            decision=decision,
            traceability=trace,
            price_freshness=pf,
            open_position_count=0,
            max_open_positions=10,
            has_active_pair_lock=False,
            cooldown_active=False,
            already_processed=False,
            missing_identity=False,
        )
        exploration = evaluate_exploration_decision(
            strict,
            decision=decision,
            exploration_mode=True,
            enable_paper_demo_orders=True,
            allow_paper_trades_with_audit_blockers=True,
            no_real_wallet=True,
            price_freshness=pf,
            traceability=trace,
            pair_address="0xABC",
            has_active_pair_lock=False,
            cooldown_active=False,
            max_open_positions_hit=False,
            missing_identity=False,
            already_processed=False,
        )
        assert exploration.should_trade_exploration is True
        assert exploration.strict_mode_decision == "BLOCKED"
        assert exploration.exploration_mode_decision == "PAPER_BUY"

    def test_strict_shadow_blocks_audit_blockers(self):
        decision = _sample_decision()
        trace = {
            "candidate_id": "cand-001",
            "source_decision_id": "dec-001",
            "source_context_record_id": "ctx-001",
            "audit_blockers": ["low_confidence"],
        }
        pf = _price_freshness_ok()
        strict = evaluate_strict_shadow_decision(
            decision=decision,
            traceability=trace,
            price_freshness=pf,
            open_position_count=0,
            max_open_positions=10,
            has_active_pair_lock=False,
            cooldown_active=False,
            already_processed=False,
            missing_identity=False,
        )
        assert strict.strict_mode_decision == "BLOCKED"
        assert strict.blocked_by_ae9 is True

    def test_exploration_mode_overrides_audit_blockers(self):
        decision = _sample_decision(no_trade_authority=True)
        trace = {
            "candidate_id": "cand-001",
            "source_decision_id": "dec-001",
            "audit_blockers": ["low_confidence"],
        }
        pf = _price_freshness_ok()
        strict = evaluate_strict_shadow_decision(
            decision=decision,
            traceability=trace,
            price_freshness=pf,
            open_position_count=0,
            max_open_positions=10,
            has_active_pair_lock=False,
            cooldown_active=False,
            already_processed=False,
            missing_identity=False,
        )
        exploration = evaluate_exploration_decision(
            strict,
            decision=decision,
            exploration_mode=True,
            enable_paper_demo_orders=True,
            allow_paper_trades_with_audit_blockers=True,
            no_real_wallet=True,
            price_freshness=pf,
            traceability=trace,
            pair_address="0xABC",
            has_active_pair_lock=False,
            cooldown_active=False,
            max_open_positions_hit=False,
            missing_identity=False,
            already_processed=False,
        )
        assert exploration.should_trade_exploration is True
        assert exploration.override_type == EXPLORATION_OVERRIDE_TYPE

    def test_missing_price_timestamp_blocks(self):
        pf = evaluate_price_freshness(
            {"price": 0.001},
            decision=_sample_decision(),
            exploration_max_price_age_seconds=900.0,
            strict_max_price_age_seconds=30.0,
            loop_observed_at_utc="2026-07-10T09:00:52+00:00",
        )
        assert pf.price_timestamp_missing is True
        assert pf.exploration_price_fresh is False


class TestOpportunityCapture:
    def test_no_lookahead_forward_returns(self):
        first_seen = "2026-07-10T09:00:00+00:00"
        now_early = "2026-07-10T09:02:00+00:00"
        oracle = DemoPriceOracle(max_price_age_seconds=30.0)
        oracle.snapshots = [_fresh_snapshot(first_seen, 0.001)]

        from app.runtime_paper_loop.decision_policy import PolicyDecision

        policy = PolicyDecision()
        capture = build_opportunity_capture_record(
            decision=_sample_decision(),
            context=_sample_context(),
            audit=_sample_audit(),
            traceability={"candidate_id": "cand-001", "source_decision_id": "dec-001"},
            policy=policy,
            price_result={"price": 0.001, "price_status": "PRICE_OK"},
            loop_run_id="run-1",
            loop_iteration=1,
        )
        capture.first_seen_timestamp = first_seen
        capture.price_at_first_seen = 0.001
        updated = compute_forward_returns_no_lookahead(
            capture, price_oracle=oracle, coin_id=1, pair_address="0xABC", now_utc=now_early
        )
        assert updated.horizon_matured_5m is False
        assert updated.max_return_5m is None

    def test_missed_winner_post_hoc_labeling(self):
        rec = OpportunityCaptureRecord(
            loop_run_id="r",
            loop_iteration=1,
            paper_action_taken="NO_TRADE",
            horizon_matured_1h=True,
            max_return_1h=0.25,
        )
        assert is_missed_winner(rec) is True


class TestLoopIntegration:
    def test_one_iteration_run(self, tmp_ae11_env):
        config = Ae11LoopConfig(
            project_root=tmp_ae11_env["root"],
            enable_paper_demo_orders=True,
            exploration_mode=True,
            allow_paper_trades_with_audit_blockers=True,
        )
        runner = RuntimePaperLoopRunner(config)
        runner.state_db = AE11StateDb(tmp_ae11_env["state_db"])
        runner.price_oracle.snapshots = [_fresh_snapshot("2026-07-10T09:00:52+00:00")]
        from app.runtime_paper_loop.run_context import RunContextFactory

        ctx, _ = RunContextFactory.create(project_root=tmp_ae11_env["root"], checkpoint=None)
        runner.run_context = ctx
        candidates = [(_sample_decision(), _sample_context(), _sample_audit())]
        with mock.patch.object(runner, "startup", return_value={"state_reconstruction_status": "OK"}):
            result = runner.run_iteration(candidates_override=candidates)
        assert result["loop_iteration"] == 1
        assert result.get("writer_flush_status") is True

    def test_duplicate_skipped_on_second_run(self, tmp_ae11_env):
        config = Ae11LoopConfig(
            project_root=tmp_ae11_env["root"],
            enable_paper_demo_orders=True,
            exploration_mode=True,
            allow_paper_trades_with_audit_blockers=True,
        )
        runner = RuntimePaperLoopRunner(config)
        runner.state_db = AE11StateDb(tmp_ae11_env["state_db"])
        runner.price_oracle.snapshots = [_fresh_snapshot("2026-07-10T09:00:52+00:00")]
        from app.runtime_paper_loop.run_context import RunContextFactory

        ctx, _ = RunContextFactory.create(project_root=tmp_ae11_env["root"], checkpoint=None)
        runner.run_context = ctx
        candidates = [(_sample_decision(), _sample_context(), _sample_audit())]
        with mock.patch.object(runner, "startup", return_value={}):
            runner.run_iteration(candidates_override=candidates)
            result2 = runner.run_iteration(candidates_override=candidates)
        assert result2["duplicates_skipped"] >= 1

    def test_state_reconstruction_from_logs(self, tmp_path):
        orders_path = tmp_path / "orders.jsonl"
        _write_jsonl(
            orders_path,
            [
                {
                    "paper_order_id": "ord-1",
                    "source_decision_id": "dec-1",
                    "status": "PAPER_FILLED",
                    "notional_usd": 100.0,
                }
            ],
        )
        state = ledger_reconstruction(
            paper_orders_paths=[orders_path],
            paper_positions_paths=[],
            paper_trades_paths=[],
            starting_balance_usd=10_000.0,
        )
        assert state.cash_balance_usd == 9900.0

    def test_no_wallet_path(self, tmp_ae11_env):
        runner = RuntimePaperLoopRunner(Ae11LoopConfig(project_root=tmp_ae11_env["root"]))
        assert runner.live_adapter.is_wallet_configured() is False
        assert runner.live_adapter.private_key_accessed is False


class TestCloseLifecycle:
    def test_tp_close(self):
        from app.paper_trading.position_manager import close_position_manual
        from app.paper_trading.types import PaperOrder, PaperPosition

        order = PaperOrder(
            paper_order_id="ord-1",
            filled_price_usd=0.001,
            quantity=100_000,
            notional_usd=100,
            status=PaperOrderStatus.PAPER_FILLED.value,
        )
        position = PaperPosition(
            paper_order_id="ord-1",
            entry_price_usd=0.001,
            quantity=100_000,
            notional_usd=100,
            status="OPEN",
        )
        trade, _ = close_position_manual(position, order, 0.0012, PaperOrderStatus.PAPER_CLOSED_TP.value)
        assert trade.realized_pnl_usd > 0


class TestCrashSimulation:
    def test_crash_after_decision_before_fill(self, tmp_path):
        db = AE11StateDb(tmp_path / "state.sqlite")
        db.mark_decision_processed("dec-crash", action_taken="PENDING")
        assert db.is_decision_processed("dec-crash")
        db.close()

    def test_crash_after_fill_before_checkpoint(self, tmp_path):
        ckpt = tmp_path / "checkpoint.json"
        write_checkpoint({"loop_run_id": "run-x", "last_completed_iteration": 0}, ckpt)
        assert load_checkpoint(ckpt)["loop_run_id"] == "run-x"


class TestAE11BBacklogDrain:
    def test_drains_unprocessed_when_latest_already_processed(self, tmp_path):
        ae6_path = tmp_path / "ae6_decisions.jsonl"
        records = []
        for i in range(5):
            records.append(_sample_decision(decision_id=f"dec-old-{i}"))
        for i in range(5):
            records.append(_sample_decision(decision_id=f"dec-new-{i}"))
        _write_jsonl(ae6_path, records)

        db = AE11StateDb(tmp_path / "state.sqlite")
        for i in range(5):
            db.mark_decision_processed(f"dec-new-{i}", action_taken="NO_TRADE")

        result = load_unprocessed_batch(
            project_root=tmp_path,
            state_db=db,
            batch_size=10,
            max_scan_records=100,
            source_path=ae6_path,
        )
        assert result.records_selected_for_processing == 5
        assert result.candidates[0][0]["decision_id"] == "dec-old-0"
        db.close()

    def test_cursor_persists_across_loads(self, tmp_path):
        ae6_path = tmp_path / "ae6.jsonl"
        _write_jsonl(ae6_path, [_sample_decision(decision_id=f"d{i}") for i in range(3)])
        db = AE11StateDb(tmp_path / "state.sqlite")
        r1 = load_unprocessed_batch(
            project_root=tmp_path,
            state_db=db,
            batch_size=1,
            max_scan_records=100,
            source_path=ae6_path,
        )
        assert len(r1.candidates) == 1
        db.mark_decision_processed("d0", action_taken="NO_TRADE")
        r2 = load_unprocessed_batch(
            project_root=tmp_path,
            state_db=db,
            batch_size=1,
            max_scan_records=100,
            source_path=ae6_path,
        )
        assert r2.candidates[0][0]["decision_id"] == "d1"
        db.close()

    def test_eof_no_new_decisions_available(self, tmp_path):
        ae6_path = tmp_path / "ae6.jsonl"
        _write_jsonl(ae6_path, [_sample_decision(decision_id="only-one")])
        db = AE11StateDb(tmp_path / "state.sqlite")
        db.mark_decision_processed("only-one", action_taken="NO_TRADE")
        result = load_unprocessed_batch(
            project_root=tmp_path,
            state_db=db,
            batch_size=10,
            max_scan_records=100,
            source_path=ae6_path,
        )
        assert result.no_new_decisions_available is True
        assert result.eof_reached is True
        db.close()

    def test_hierarchical_trade_decision_shape(self):
        from app.runtime_paper_loop.decision_policy import PolicyDecision

        policy = PolicyDecision(
            strict_mode_decision="BLOCKED",
            exploration_mode_decision="PAPER_BUY",
            should_trade_exploration=True,
            hard_safety_gates_passed=True,
            override_type=EXPLORATION_OVERRIDE_TYPE,
        )
        pf = _price_freshness_ok()
        td = build_hierarchical_trade_decision(
            source_decision_id="dec-001",
            source_event_key="evt-001",
            candidate_id="cand-001",
            pair_address="0xABC",
            loop_run_id="run",
            loop_iteration=1,
            policy=policy,
            price_freshness=pf,
            paper_action="FILLED",
            decision=_sample_decision(no_trade_authority=True),
        )
        assert "strict_mode" in td
        assert "exploration_mode" in td
        assert "hard_safety" in td
        assert td["schema_version"] == "AE11B_v1"
        assert td["exploration_mode"]["original_no_trade_authority"] is True

    def test_source_event_key_is_event_level(self):
        d = _sample_decision(
            candidate_identity={
                "candidate_id": "c1",
                "pair_address": "0xPAIR",
                "event_timestamp": "2026-07-10T09:00:00+00:00",
            },
            market_context={"source_snapshot_id": 42},
        )
        key = build_source_event_key(d)
        assert "0xPAIR" in key
        assert "42" in key
        assert "dec-001" in key

    def test_eof_cursor_rewind_finds_backlog(self, tmp_path):
        ae6_path = tmp_path / "ae6.jsonl"
        _write_jsonl(ae6_path, [_sample_decision(decision_id=f"old-{i}") for i in range(3)])
        db = AE11StateDb(tmp_path / "state.sqlite")
        db.update_source_cursor(
            "ae6_decisions_jsonl",
            str(ae6_path.resolve()),
            cursor_type="byte_offset",
            cursor_value=str(ae6_path.stat().st_size),
            eof_reached=True,
        )
        db.mark_decision_processed("old-2", action_taken="NO_TRADE")
        result = load_unprocessed_batch(
            project_root=tmp_path,
            state_db=db,
            batch_size=10,
            max_scan_records=100,
            source_path=ae6_path,
        )
        if result.no_new_decisions_available:
            db.reset_source_cursor("ae6_decisions_jsonl", str(ae6_path.resolve()))
            result = load_unprocessed_batch(
                project_root=tmp_path,
                state_db=db,
                batch_size=10,
                max_scan_records=100,
                source_path=ae6_path,
            )
        assert result.records_selected_for_processing >= 2
        db.close()

    def test_price_freshness_uses_snapshot_timestamp(self):
        pf = _price_freshness_ok()
        assert pf.price_timestamp_source is not None
        assert pf.price_age_seconds is not None
        assert pf.price_age_seconds <= 30
        assert pf.exploration_price_fresh is True


class TestAE11CRunContext:
    def test_new_invocation_generates_fresh_ids_with_checkpoint(self, tmp_path):
        from app.runtime_paper_loop.run_context import RunContextFactory

        ckpt = {
            "loop_run_id": "old-loop-id",
            "audit_root": str(tmp_path / "old_audit"),
            "last_completed_iteration": 98,
            "cash_balance": 7000.0,
            "idempotency_index_status": {"processed_decisions_count": 4700},
        }
        ctx_a, _ = RunContextFactory.create(
            project_root=tmp_path, checkpoint=ckpt, explicit_resume_requested=False
        )
        ctx_b, _ = RunContextFactory.create(
            project_root=tmp_path, checkpoint=ckpt, explicit_resume_requested=False
        )
        assert ctx_a.loop_run_id != "old-loop-id"
        assert ctx_b.loop_run_id != "old-loop-id"
        assert ctx_a.loop_run_id != ctx_b.loop_run_id
        assert ctx_a.invocation_id != ctx_b.invocation_id
        assert ctx_a.audit_root != ctx_b.audit_root
        assert ctx_a.checkpoint_loop_run_id_if_any == "old-loop-id"
        assert ctx_a.loop_run_id_reused_from_checkpoint is False
        assert ctx_a.audit_root_reused_from_checkpoint is False

    def test_explicit_resume_reuses_loop_run_id(self, tmp_path):
        from app.runtime_paper_loop.run_context import RunContextFactory

        old_audit = tmp_path / "old_audit"
        old_audit.mkdir()
        ckpt = {
            "loop_run_id": "resume-me",
            "audit_root": str(old_audit),
            "last_completed_iteration": 10,
        }
        ctx, _ = RunContextFactory.create(
            project_root=tmp_path,
            checkpoint=ckpt,
            explicit_resume_requested=True,
            resume_loop_run_id="resume-me",
            resume_audit_root=old_audit,
        )
        assert ctx.loop_run_id == "resume-me"
        assert ctx.audit_root == old_audit
        assert ctx.explicit_resume_requested is True


class TestAE11CRunMetrics:
    def test_session_counters_reset_per_invocation(self):
        from app.runtime_paper_loop.run_metrics import MetricsCounters, RunMetrics

        m = RunMetrics()
        m.session.iterations_completed = 98
        m.session.decisions_seen = 150
        fresh = RunMetrics()
        assert fresh.session.iterations_completed == 0
        assert fresh.session.decisions_seen == 0

    def test_cumulative_separate_from_session(self, tmp_path):
        from app.runtime_paper_loop.run_metrics import RunMetrics

        db = AE11StateDb(tmp_path / "state.sqlite")
        for i in range(5):
            db.mark_decision_processed(f"d{i}", action_taken="NO_TRADE")
        metrics = RunMetrics()
        metrics.session.iterations_completed = 3
        cumulative = metrics.cumulative_extended(db, cash_balance=9500.0)
        assert metrics.session.iterations_completed == 3
        assert cumulative["processed_decisions"] == 5
        db.close()


class TestAE11CMismatchEvent:
    def test_mismatch_repaired_includes_detail(self, tmp_path):
        from app.runtime_paper_loop.mismatch_event import (
            detect_checkpoint_mismatches,
            write_mismatch_audit,
        )
        from app.runtime_paper_loop.run_context import RunContextFactory
        from app.runtime_paper_loop.types import ReconstructedAccountState

        db = AE11StateDb(tmp_path / "state.sqlite")
        db.mark_decision_processed("d1", action_taken="NO_TRADE")
        ckpt = {
            "cash_balance": 10000.0,
            "processed_registry_count": 0,
            "active_position_ids": [],
            "idempotency_index_status": {"processed_decisions_count": 0},
        }
        reconstructed = ReconstructedAccountState(
            cash_balance_usd=10000.0,
            realized_pnl_usd=0.0,
            open_positions=[],
            reconstruction_status="OK",
            mismatches=[],
        )
        ctx, _ = RunContextFactory.create(project_root=tmp_path, checkpoint=ckpt)
        events = detect_checkpoint_mismatches(
            loop_run_id=ctx.loop_run_id,
            invocation_id=ctx.invocation_id,
            checkpoint=ckpt,
            reconstructed=reconstructed,
            state_db=db,
            explicit_resume_requested=False,
            run_context=ctx,
        )
        repaired = [e for e in events if e.repair_applied and e.mismatch_detected]
        assert len(repaired) >= 1
        assert any(e.field_path for e in repaired)
        csv_path = tmp_path / "audit.csv"
        write_mismatch_audit(csv_path, events)
        text = csv_path.read_text()
        assert "PROCESSED_DECISION_COUNT_MISMATCH" in text or "mismatch_type" in text
        db.close()

    def test_session_counter_not_leaked_from_checkpoint(self, tmp_ae11_env):
        stale_ckpt = {
            "loop_run_id": "stale-run",
            "audit_root": str(tmp_ae11_env["root"] / "old_audit"),
            "last_completed_iteration": 98,
            "cash_balance": 7000.0,
        }
        config = Ae11LoopConfig(
            project_root=tmp_ae11_env["root"],
            duration_minutes=0.001,
            loop_interval_seconds=0.01,
            enable_paper_demo_orders=False,
            exploration_mode=True,
        )
        runner = RuntimePaperLoopRunner(config)
        runner.state_db = AE11StateDb(tmp_ae11_env["state_db"])
        with mock.patch("app.runtime_paper_loop.loop_runner.load_checkpoint", return_value=stale_ckpt):
            with mock.patch("app.runtime_paper_loop.loop_runner.discover_ae6_path", return_value=None):
                with mock.patch("app.runtime_paper_loop.loop_runner.write_checkpoint"):
                    summary = runner.run_loop()
        assert summary["loop_run_id"] != "stale-run"
        assert summary["iterations_completed"] <= 5
        assert summary.get("current_invocation_counters") is True
        assert "cumulative_metrics" in summary


class TestAE11DAuthoritativeSnapshots:
    def _seed_open_positions(self, db: AE11StateDb, n: int = 30) -> None:
        from app.runtime_paper_loop.db_migration import migrate_db_schema

        migrate_db_schema(db_path=db.path, project_root=Path(db.path).parent)
        for i in range(n):
            db.register_position(
                f"pos-{i}",
                f"0xPAIR{i:04d}",
                paper_order_id=f"ord-{i}",
                source_decision_id=f"dec-{i}",
                economics={
                    "entry_price": "1",
                    "quantity": "100",
                    "notional_usd": "100",
                    "cost_basis_usd": "100.3",
                    "entry_fee_usd": "0.3",
                    "entry_slippage_usd": "0.5",
                    "cash_debited_usd": "100.8",
                    "open_market_value_usd": "100",
                    "last_price": "1",
                    "tp_price": "1000000",
                    "sl_price": "0.0000001",
                    "time_stop_at_utc": "2099-01-01T00:00:00+00:00",
                    "opened_at_utc": "2026-01-01T00:00:00+00:00",
                    "economic_enrichment_status": "FULL",
                    "economic_enrichment_missing_fields": "",
                },
            )
            db.set_cooldown(f"0xPAIR{i:04d}", "2099-01-01T00:00:00+00:00")

    def test_fetch_authoritative_state_from_sqlite_not_checkpoint(self, tmp_path):
        from app.runtime_paper_loop.report_generator import ReportGenerator

        db = AE11StateDb(tmp_path / "state.sqlite")
        self._seed_open_positions(db, 30)
        gen = ReportGenerator(
            state_db=db,
            project_root=tmp_path,
            loop_run_id="run-a",
            invocation_id="inv-a",
        )
        state = gen.fetch_authoritative_state()
        assert state.open_position_count == 30
        assert len(state.active_pair_locks) == 30
        # Does not accept/require checkpoint or RunContext position lists
        assert state.source_of_truth_open_positions in ("sqlite", "sqlite_status_OPEN")
        db.close()

    def test_open_snapshot_starts_from_sqlite_left_join_enrichment(self, tmp_path):
        from app.runtime_paper_loop.report_generator import (
            build_open_position_snapshot_rows,
            write_csv_with_headers,
            OPEN_POSITION_SNAPSHOT_FIELDS,
        )

        db = AE11StateDb(tmp_path / "state.sqlite")
        self._seed_open_positions(db, 30)
        sqlite_rows = db.load_active_positions()
        # Enrichment only for first 5 — remaining 25 must still appear
        enrichment = [
            {
                "position_id": f"pos-{i}",
                "symbol": f"SYM{i}",
                "entry_price_usd": 0.001,
                "notional_usd": 100.0,
                "quantity": 100000.0,
                "candidate_id": f"cand-{i}",
            }
            for i in range(5)
        ]
        rows = build_open_position_snapshot_rows(
            sqlite_positions=sqlite_rows,
            cooldowns=db.load_cooldowns(),
            active_pair_locks={r["pair_address"]: r["position_id"] for r in sqlite_rows},
            enrichment_records=enrichment,
            loop_run_id="run",
            invocation_id="inv",
        )
        assert len(rows) == 30
        enriched = [r for r in rows if r["enrichment_available"]]
        missing_enrich = [r for r in rows if not r["enrichment_available"]]
        assert len(enriched) == 5
        assert len(missing_enrich) == 25
        assert all(r["position_id"] for r in missing_enrich)
        assert all(r["enrichment_missing_fields"] for r in missing_enrich)

        out = tmp_path / "open.csv"
        write_csv_with_headers(out, rows, OPEN_POSITION_SNAPSHOT_FIELDS)
        text = out.read_text(encoding="utf-8")
        assert out.stat().st_size > 0
        assert "position_id" in text.splitlines()[0]
        assert len(text.splitlines()) == 31  # header + 30
        db.close()

    def test_closed_snapshot_headers_with_zero_rows(self, tmp_path):
        from app.runtime_paper_loop.report_generator import (
            write_csv_with_headers,
            CLOSED_TRADE_SNAPSHOT_FIELDS,
        )

        out = tmp_path / "closed.csv"
        write_csv_with_headers(out, [], CLOSED_TRADE_SNAPSHOT_FIELDS)
        assert out.stat().st_size > 0
        lines = out.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1
        assert "close_event_id" in lines[0]

    def test_report_consistency_pass_when_counts_match(self, tmp_path):
        from app.runtime_paper_loop.report_generator import ReportGenerator

        db = AE11StateDb(tmp_path / "state.sqlite")
        self._seed_open_positions(db, 30)
        gen = ReportGenerator(
            state_db=db,
            project_root=tmp_path,
            loop_run_id="run",
            invocation_id="inv",
        )
        state = gen.fetch_authoritative_state()
        open_rows = gen.build_open_positions_snapshot(state)
        closed_rows = gen.build_closed_trades_snapshot(state)
        rows, status, mismatches = gen.build_consistency_audit(
            state=state,
            open_csv_rows=len(open_rows),
            closed_csv_rows=len(closed_rows),
            summary_open=30,
            summary_closed=0,
        )
        assert status == "PASS"
        assert mismatches == 0
        assert any(r.report_name == "open_positions_snapshot" and r.status == "PASS" for r in rows)
        db.close()

    def test_report_consistency_fails_empty_snapshot_vs_sqlite(self, tmp_path):
        from app.runtime_paper_loop.report_generator import ReportGenerator

        db = AE11StateDb(tmp_path / "state.sqlite")
        self._seed_open_positions(db, 30)
        gen = ReportGenerator(
            state_db=db,
            project_root=tmp_path,
            loop_run_id="run",
            invocation_id="inv",
        )
        state = gen.fetch_authoritative_state()
        rows, status, mismatches = gen.build_consistency_audit(
            state=state,
            open_csv_rows=0,
            closed_csv_rows=0,
            summary_open=30,
            summary_closed=0,
        )
        assert status == "FAIL"
        assert mismatches >= 1
        assert any(r.mismatch_type == "OPEN_SNAPSHOT_COUNT_MISMATCH" for r in rows)
        db.close()

    def test_startup_syncs_in_memory_from_sqlite_and_reports_30(self, tmp_ae11_env):
        db_path = tmp_ae11_env["state_db"]
        db = AE11StateDb(db_path)
        self._seed_open_positions(db, 30)
        db.close()

        stale_ckpt = {
            "loop_run_id": "old",
            "audit_root": str(tmp_ae11_env["root"] / "old_audit"),
            "last_completed_iteration": 10,
            "cash_balance": 7000.0,
            "active_position_ids": [],
            "idempotency_index_status": {"processed_decisions_count": 0},
        }
        config = Ae11LoopConfig(
            project_root=tmp_ae11_env["root"],
            duration_minutes=0.001,
            loop_interval_seconds=0.01,
            enable_paper_demo_orders=False,
            exploration_mode=True,
            starting_balance_usd=10000.0,
        )
        runner = RuntimePaperLoopRunner(config)
        runner.state_db = AE11StateDb(db_path)
        with mock.patch("app.runtime_paper_loop.loop_runner.load_checkpoint", return_value=stale_ckpt):
            with mock.patch("app.runtime_paper_loop.loop_runner.discover_ae6_path", return_value=None):
                with mock.patch("app.runtime_paper_loop.loop_runner.write_checkpoint"):
                    summary = runner.run_loop()

        assert summary["startup"]["open_positions"] == 30
        assert summary["startup"]["in_memory_open_positions"] == 30
        assert summary.get("open_positions_snapshot_rows") == 30
        assert summary.get("closed_trades_snapshot_rows") == 0
        assert summary.get("report_consistency_status") == "PASS"
        assert summary.get("report_consistency_mismatch_count") == 0

        open_csv = tmp_ae11_env["root"] / "data" / "ae11_open_positions_snapshot.csv"
        closed_csv = tmp_ae11_env["root"] / "data" / "ae11_closed_trades_snapshot.csv"
        assert open_csv.is_file() and open_csv.stat().st_size > 0
        assert closed_csv.is_file() and closed_csv.stat().st_size > 0
        open_lines = open_csv.read_text(encoding="utf-8").strip().splitlines()
        closed_lines = closed_csv.read_text(encoding="utf-8").strip().splitlines()
        assert len(open_lines) == 31
        assert len(closed_lines) == 1
        consistency = tmp_ae11_env["root"] / "audits" / "ae11_report_consistency_audit.csv"
        assert consistency.is_file()
        assert "PASS" in consistency.read_text(encoding="utf-8")


class TestAE11ELedgerLifecycle:
    def test_migrate_db_schema_idempotent(self, tmp_path):
        from app.runtime_paper_loop.db_migration import migrate_db_schema

        db_path = tmp_path / "state.sqlite"
        db = AE11StateDb(db_path)
        db.register_position("p1", "0xA", paper_order_id="o1", source_decision_id="d1")
        db.close()

        r1 = migrate_db_schema(db_path=db_path, project_root=tmp_path)
        assert r1["migration_applied"] is True
        r2 = migrate_db_schema(db_path=db_path, project_root=tmp_path)
        assert r2["migration_applied"] is False
        assert r2["migration_skipped_reason"] == "ALREADY_APPLIED"

        db2 = AE11StateDb(db_path)
        rows = db2.load_active_positions()
        assert len(rows) == 1
        assert rows[0]["position_id"] == "p1"
        db2.close()

    def test_decimal_ledger_30x100_cash_7000(self, tmp_path):
        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.ledger_accounting import reconstruct_ledger_from_sqlite
        from app.runtime_paper_loop.decimal_money import quantize_usd

        db_path = tmp_path / "state.sqlite"
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        db = AE11StateDb(db_path)
        for i in range(30):
            db.register_position(
                f"pos-{i}",
                f"0xP{i}",
                paper_order_id=f"ord-{i}",
                source_decision_id=f"dec-{i}",
                economics={
                    "entry_price": "0.001",
                    "quantity": "100000",
                    "notional_usd": "100",
                    "cost_basis_usd": "100",
                    "cash_debited_usd": "100",
                    "entry_fee_usd": "0",
                    "economic_enrichment_status": "FULL",
                    "open_market_value_usd": "100",
                    "unrealized_pnl_usd": "0",
                },
            )
        snap = reconstruct_ledger_from_sqlite(db, starting_balance_usd=10000.0)
        assert snap.cash_balance == quantize_usd(7000)
        assert snap.open_cost_basis_usd == quantize_usd(3000)
        assert snap.ledger_consistency_status == "PASS"
        db.close()

    def test_backfill_from_jsonl_and_missing(self, tmp_path):
        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.position_backfill import backfill_position_economics
        import json

        db_path = tmp_path / "state.sqlite"
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        paper_dir = tmp_path / "data" / "paper_trading"
        paper_dir.mkdir(parents=True)
        orders = []
        for i in range(25):
            orders.append(
                {
                    "paper_order_id": f"ord-{i}",
                    "filled_price_usd": 0.001,
                    "quantity": 100000.0,
                    "notional_usd": 100.0,
                    "filled_at_utc": "2026-07-11T13:51:33+00:00",
                    "price_timestamp": "2026-07-11T13:51:23+00:00",
                    "price_snapshot_id": 100 + i,
                    "candidate_id": f"c-{i}",
                    "symbol": f"S{i}/WETH",
                    "trade_authority": "PAPER_EXPLORATION_ONLY",
                    "not_model_approved": True,
                    "not_live_approved": True,
                    "override_type": "DEMO",
                }
            )
        (paper_dir / "paper_orders_20260711.jsonl").write_text(
            "\n".join(json.dumps(o) for o in orders) + "\n", encoding="utf-8"
        )
        db = AE11StateDb(db_path)
        for i in range(30):
            db.register_position(
                f"pos-{i}",
                f"0xP{i}",
                paper_order_id=f"ord-{i}",
                source_decision_id=f"dec-{i}",
            )
        result = backfill_position_economics(
            db,
            project_root=tmp_path,
            loop_run_id="run",
            invocation_id="inv",
            historical_cash_debit_notional_only=True,
        )
        assert result["backfill_success_count"] == 25
        assert result["backfill_missing_count"] == 5
        rows = db.load_active_positions()
        assert len(rows) == 30
        full = [r for r in rows if r.get("economic_enrichment_status") == "FULL"]
        missing = [r for r in rows if r.get("economic_enrichment_status") == "MISSING"]
        assert len(full) == 25
        assert len(missing) == 5
        assert (tmp_path / "audits" / "ae11_position_backfill_audit.csv").is_file()
        db.close()

    def test_time_stop_closes_restored_position(self, tmp_path):
        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.position_lifecycle import evaluate_and_close_positions
        from app.runtime_paper_loop.persistence import IterationWriters
        from app.paper_trading.price_oracle import DemoPriceOracle

        db_path = tmp_path / "state.sqlite"
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        db = AE11StateDb(db_path)
        db.register_position(
            "pos-old",
            "0xOLD",
            paper_order_id="ord-old",
            source_decision_id="dec-old",
            opened_at_utc="2026-07-11T13:51:33+00:00",
            economics={
                "entry_price": "0.001",
                "quantity": "100000",
                "notional_usd": "100",
                "cost_basis_usd": "100",
                "cash_debited_usd": "100",
                "entry_fee_usd": "0",
                "tp_price": "0.0012",
                "sl_price": "0.0009",
                "time_stop_at_utc": "2026-07-11T17:51:33+00:00",
                "economic_enrichment_status": "FULL",
            },
        )
        config = Ae11LoopConfig(
            project_root=tmp_path,
            time_stop_minutes=240.0,
            take_profit_pct=20.0,
            stop_loss_pct=10.0,
            fee_bps=0.0,
            slippage_bps=0.0,
            entry_fee_bps=0.0,
            exit_fee_bps=0.0,
        )
        writers = IterationWriters()
        oracle = DemoPriceOracle(max_price_age_seconds=900)
        result = evaluate_and_close_positions(
            db,
            oracle,
            writers,
            config=config,
            loop_run_id="run",
            invocation_id="inv",
            iteration=1,
            project_root=tmp_path,
        )
        assert result["positions_closed"] == 1
        assert result["exit_reasons"].get("TIME_STOP") == 1
        assert db.active_open_position_count() == 0
        writers.close_all()
        db.close()

    def test_missing_economics_blocks_lifecycle(self, tmp_path):
        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.position_lifecycle import evaluate_and_close_positions
        from app.runtime_paper_loop.persistence import IterationWriters
        from app.paper_trading.price_oracle import DemoPriceOracle

        db_path = tmp_path / "state.sqlite"
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        db = AE11StateDb(db_path)
        db.register_position(
            "pos-blind",
            "0xB",
            paper_order_id="ord-missing",
            source_decision_id="dec-b",
            economics={"economic_enrichment_status": "MISSING", "economic_enrichment_missing_fields": "entry_price,quantity"},
        )
        config = Ae11LoopConfig(project_root=tmp_path, time_stop_minutes=1.0)
        result = evaluate_and_close_positions(
            db,
            DemoPriceOracle(),
            IterationWriters(),
            config=config,
            loop_run_id="r",
            invocation_id="i",
            iteration=1,
            project_root=tmp_path,
        )
        assert result["positions_closed"] == 0
        assert result["positions_blocked"] >= 1
        audit = (tmp_path / "audits" / "ae11_position_lifecycle_audit.csv").read_text(encoding="utf-8")
        assert "BLOCKED_MISSING_ECONOMICS" in audit or "BLOCKED_MISSING_ENTRY_ECONOMICS" in audit
        assert "OPEN_POSITION_MISSING_ECONOMICS" in audit
        db.close()


class TestAE11HEquityBridgeAndInvariants:
    def test_equity_bridge_cash_and_pnl_identities(self, tmp_path):
        from decimal import Decimal
        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.equity_bridge import build_equity_bridge
        from app.runtime_paper_loop.decimal_money import decimal_almost_equal

        db_path = tmp_path / "state.sqlite"
        db = AE11StateDb(db_path)
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        # starting 10000; one open: notional 100, fee 0.3, slip 0.5, debit 100.8, mv 100
        db.register_position(
            "p1",
            "0xA",
            paper_order_id="o1",
            source_decision_id="d1",
            economics={
                "entry_price": "1",
                "quantity": "100",
                "notional_usd": "100",
                "cost_basis_usd": "100.3",
                "entry_fee_usd": "0.3",
                "entry_slippage_usd": "0.5",
                "cash_debited_usd": "100.8",
                "open_market_value_usd": "100",
                "opened_at_utc": "2026-01-01T00:00:00+00:00",
                "economic_enrichment_status": "FULL",
            },
        )
        bridge = build_equity_bridge(db, starting_balance_usd=10000)
        assert decimal_almost_equal(
            bridge.cash_balance_usd + bridge.open_market_value_usd,
            bridge.account_equity_usd,
        )
        assert decimal_almost_equal(
            bridge.starting_balance_usd
            + bridge.realized_net_pnl_usd
            + bridge.open_market_value_usd
            - bridge.open_cash_debited_usd,
            bridge.account_equity_usd,
        )
        assert bridge.open_price_unrealized_pnl_usd == Decimal("0")
        assert bridge.open_entry_cost_drag_usd == Decimal("-0.3")
        assert bridge.open_total_unrealized_after_cost_pnl_usd == Decimal("-0.3")
        assert bridge.bridge_status == "PASS"
        assert bridge.missing_open_economics_count == 0
        db.close()

    def test_equity_bridge_fails_when_open_economics_missing(self, tmp_path):
        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.equity_bridge import build_equity_bridge

        db_path = tmp_path / "state.sqlite"
        db = AE11StateDb(db_path)
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        db.register_position("p-miss", "0xB", paper_order_id="o2", source_decision_id="d2")
        # no economics
        bridge = build_equity_bridge(db, starting_balance_usd=10000)
        assert bridge.missing_open_economics_count >= 1
        assert bridge.bridge_status in ("FAIL", "WARNING")
        assert bridge.open_position_economic_completeness_status in ("FAIL", "WARNING")
        db.close()

    def test_ledger_invariant_check_no_assert(self, tmp_path):
        import inspect
        from app.runtime_paper_loop import equity_bridge as eb
        from app.runtime_paper_loop.equity_bridge import check_ledger_invariants

        src = inspect.getsource(eb.check_ledger_invariants)
        assert "assert " not in src
        ok = check_ledger_invariants(
            cash_balance="89.2",
            open_market_value_usd="10",
            account_equity_usd="99.2",
            starting_balance_usd="100",
            realized_net_pnl_usd="0",
            total_unrealized_after_cost_pnl_usd="-0.3",
            open_entry_slippage_usd="0.5",
            open_cash_debited_usd="10.8",
            stage="unit",
        )
        assert ok["ledger_invariant_status"] == "PASS"

        bad = check_ledger_invariants(
            cash_balance="90",
            open_market_value_usd="10",
            account_equity_usd="999",
            starting_balance_usd="100",
            realized_net_pnl_usd="0",
            total_unrealized_after_cost_pnl_usd="0",
            stage="unit",
        )
        assert bad["ledger_invariant_status"] == "FAIL"
        assert bad["ledger_invariant_failure_count"] >= 1

    def test_open_snapshot_includes_economic_fields(self, tmp_path):
        from app.runtime_paper_loop.report_generator import (
            OPEN_POSITION_SNAPSHOT_FIELDS,
            build_open_position_snapshot_rows,
        )

        rows = build_open_position_snapshot_rows(
            sqlite_positions=[
                {
                    "position_id": "p1",
                    "pair_address": "0xA",
                    "status": "OPEN",
                    "opened_at_utc": "2026-01-01T00:00:00+00:00",
                    "entry_price": "1",
                    "quantity": "100",
                    "notional_usd": "100",
                    "cost_basis_usd": "100.3",
                    "cash_debited_usd": "100.8",
                    "entry_fee_usd": "0.3",
                    "entry_slippage_usd": "0.5",
                    "open_market_value_usd": "100",
                    "economic_enrichment_status": "FULL",
                }
            ],
            cooldowns={},
            active_pair_locks={"0xA": "p1"},
            enrichment_records=[],
            loop_run_id="r",
            invocation_id="i",
        )
        assert len(rows) == 1
        for f in (
            "cash_debited_usd",
            "entry_slippage_usd",
            "open_market_value_usd",
            "price_unrealized_pnl_usd",
            "total_unrealized_after_cost_pnl_usd",
            "open_entry_cost_drag_usd",
        ):
            assert f in OPEN_POSITION_SNAPSHOT_FIELDS
            assert rows[0].get(f) not in (None, "")

    def test_metric_semantics_current_vs_legacy(self, tmp_path):
        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.run_metrics import RunMetrics

        db_path = tmp_path / "state.sqlite"
        db = AE11StateDb(db_path)
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        db.register_position(
            "p1",
            "0xA",
            paper_order_id="o1",
            source_decision_id="d1",
            economics={
                "entry_price": "1",
                "quantity": "1",
                "notional_usd": "100",
                "cost_basis_usd": "100",
                "cash_debited_usd": "100",
                "economic_enrichment_status": "FULL",
            },
        )
        metrics = RunMetrics()
        cum = metrics.cumulative_extended(db, cash_balance=9900.0)
        assert cum["current_open_positions_count"] == db.count_open_positions()
        assert cum["orders_total_semantics"] == "legacy_alias_current_open_not_lifetime"
        assert cum["cumulative_metric_semantics_status"] == "PASS"
        db.close()


class TestAE11FClosedTradeIdempotency:
    def test_ae11f_migration_creates_unique_index_and_is_idempotent(self, tmp_path):
        import sqlite3
        from app.runtime_paper_loop.db_migration import (
            MIGRATION_NAME_AE11F,
            migrate_db_schema,
        )

        db_path = tmp_path / "state.sqlite"
        db = AE11StateDb(db_path)
        db.register_position("p-close", "0xC", paper_order_id="o1", source_decision_id="d1")
        db.update_position_economics(
            "p-close",
            {
                "status": "CLOSED",
                "entry_price": "1",
                "exit_price": "1.1",
                "quantity": "10",
                "notional_usd": "100",
                "cash_debited_usd": "100",
                "cash_credited_usd": "110",
                "exit_reason": "TIME_STOP",
                "closed_at_utc": "2026-01-01T00:00:00+00:00",
            },
        )
        db.close_position("p-close", "0xC")
        db.close()

        r1 = migrate_db_schema(db_path=db_path, project_root=tmp_path)
        assert r1.get("ae11f", {}).get("status") in ("APPLIED", "SKIPPED")
        r2 = migrate_db_schema(db_path=db_path, project_root=tmp_path)
        assert r2["ae11f"]["migration_applied"] is False
        assert r2["ae11f"]["migration_skipped_reason"] == "ALREADY_APPLIED"

        conn = sqlite3.connect(str(db_path))
        indexes = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_closed_positions_position_id_unique" in indexes
        applied = conn.execute(
            "SELECT 1 FROM schema_migrations WHERE migration_name = ?",
            (MIGRATION_NAME_AE11F,),
        ).fetchone()
        assert applied is not None
        conn.close()

    def test_duplicate_economic_close_skips_cash_credit(self, tmp_path):
        from uuid import uuid4

        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.decimal_money import quantize_usd
        from app.runtime_paper_loop.ledger_accounting import reconstruct_ledger_from_sqlite

        db_path = tmp_path / "state.sqlite"
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        db = AE11StateDb(db_path)
        pid = str(uuid4())
        payload = {
            "position_id": pid,
            "close_event_id": str(uuid4()),
            "economic_close_key": pid,
            "paper_order_id": "ord-1",
            "source_decision_id": "dec-1",
            "pair_address": "0xPAIR",
            "opened_at_utc": "2026-01-01T00:00:00+00:00",
            "closed_at_utc": "2026-01-01T01:00:00+00:00",
            "close_event_created_at_utc": "2026-01-01T01:00:00+00:00",
            "exit_reason": "TIME_STOP",
            "entry_price": "1",
            "exit_price": "1",
            "quantity": "100",
            "notional_usd": "100",
            "cost_basis_usd": "100",
            "entry_fee_usd": "0",
            "exit_fee_usd": "0",
            "total_fees_usd": "0",
            "gross_pnl_usd": "0",
            "net_pnl_usd": "0",
            "net_return_pct": "0",
            "cash_debited_usd": "100",
            "cash_credited_usd": "100",
            "wallet_configured": False,
            "real_transaction_attempted": False,
            "event_quality": "VALID_CANONICAL_CLOSE",
        }
        r1 = db.record_economic_close(payload)
        assert r1["recorded"] is True
        r2 = db.record_economic_close({**payload, "close_event_id": str(uuid4())})
        assert r2["duplicate"] is True
        assert r2["reason"] == "DUPLICATE_POSITION_CLOSE_SKIPPED"

        db.register_position(
            pid,
            "0xPAIR",
            paper_order_id="ord-1",
            source_decision_id="dec-1",
            economics={
                "entry_price": "1",
                "quantity": "100",
                "notional_usd": "100",
                "cost_basis_usd": "100",
                "cash_debited_usd": "100",
                "cash_credited_usd": "100",
                "net_pnl_usd": "0",
                "economic_enrichment_status": "FULL",
            },
        )
        db.close_position(pid, "0xPAIR")
        snap = reconstruct_ledger_from_sqlite(db, starting_balance_usd=10000.0)
        assert snap.cash_balance == quantize_usd(10000)
        db.close()

    def test_canonical_merge_excludes_pos_old_and_dedupes(self, tmp_path):
        from uuid import uuid4

        from app.runtime_paper_loop.closed_trade_canonical import get_canonical_closed_trades
        from app.runtime_paper_loop.db_migration import migrate_db_schema

        db_path = tmp_path / "state.sqlite"
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        db = AE11StateDb(db_path)
        pid = str(uuid4())
        eid = str(uuid4())
        db.record_economic_close(
            {
                "position_id": pid,
                "close_event_id": eid,
                "economic_close_key": pid,
                "paper_order_id": "ord-x",
                "source_decision_id": "dec-x",
                "pair_address": "0xA",
                "opened_at_utc": "2026-01-01T00:00:00+00:00",
                "closed_at_utc": "2026-01-01T02:00:00+00:00",
                "close_event_created_at_utc": "2026-01-01T02:00:00+00:00",
                "exit_reason": "TIME_STOP",
                "entry_price": "1",
                "exit_price": "1.05",
                "quantity": "100",
                "notional_usd": "100",
                "cost_basis_usd": "100",
                "entry_fee_usd": "0",
                "exit_fee_usd": "0",
                "total_fees_usd": "0",
                "gross_pnl_usd": "5",
                "net_pnl_usd": "5",
                "net_return_pct": "5",
                "cash_debited_usd": "100",
                "cash_credited_usd": "105",
                "event_quality": "VALID_CANONICAL_CLOSE",
            }
        )

        paper_dir = tmp_path / "data" / "paper_trading"
        paper_dir.mkdir(parents=True)
        with open(paper_dir / "paper_trades_2026-01-01.jsonl", "w", encoding="utf-8") as f:
            for _ in range(6):
                f.write(
                    json.dumps(
                        {
                            "position_id": "pos-old",
                            "status": "CLOSED",
                            "record_type": "PAPER_TRADE",
                        }
                    )
                    + "\n"
                )
            f.write(
                json.dumps(
                    {
                        "position_id": pid,
                        "close_event_id": eid,
                        "exit_reason": "TIME_STOP",
                        "entry_price": "1",
                        "exit_price": "1.05",
                        "record_type": "PAPER_TRADE_CLOSE",
                        "status": "CLOSED",
                    }
                )
                + "\n"
            )

        result = get_canonical_closed_trades(
            db, project_root=tmp_path, loop_run_id="r", invocation_id="i"
        )
        assert result.canonical_closed_trades_rows == 1
        assert result.canonical_rows[0]["position_id"] == pid
        assert result.canonical_rows[0]["close_event_id"] == eid
        assert all(r["position_id"] != "pos-old" for r in result.canonical_rows)
        assert result.invalid_closed_trade_rows >= 6
        assert result.closed_trade_event_history_rows >= 7
        assert result.closed_trade_hygiene_status in (
            "PASS",
            "WARNING_WITH_LEGACY_ROWS_EXCLUDED",
        )
        pids = [r["position_id"] for r in result.canonical_rows]
        assert len(pids) == len(set(pids))
        db.close()

    def test_ledger_tolerance_and_impossible_equity_not_pass(self, tmp_path):
        from decimal import Decimal

        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.decimal_money import LEDGER_TOLERANCE, quantize_usd
        from app.runtime_paper_loop.ledger_accounting import (
            LedgerSnapshot,
            write_ledger_consistency_audit,
        )

        migrate_db_schema(db_path=tmp_path / "state.sqlite", project_root=tmp_path)

        snap = LedgerSnapshot(
            starting_balance_usd=quantize_usd(10000),
            cash_balance=quantize_usd(Decimal("10000.0000004")),
            open_position_count=0,
            open_cost_basis_usd=quantize_usd(0),
            open_market_value_usd=quantize_usd(0),
            realized_pnl_usd=quantize_usd(0),
            unrealized_pnl_usd=quantize_usd(0),
            account_equity_usd=quantize_usd(Decimal("10000.0000004")),
            expected_cash_balance=quantize_usd(10000),
            cash_diff=quantize_usd(0),
            ledger_consistency_status="PASS",
            fee_model_status="ZERO_FEES_CONFIGURED",
            entry_fee_bps=0,
            exit_fee_bps=0,
            slippage_bps=0,
            ledger_cash_tolerance_usd=LEDGER_TOLERANCE,
        )
        path = tmp_path / "audits" / "ae11_ledger_consistency_audit.csv"
        write_ledger_consistency_audit(
            path, loop_run_id="r", invocation_id="i", snapshot=snap
        )
        assert snap.ledger_consistency_status == "PASS"

        bad = LedgerSnapshot(
            starting_balance_usd=quantize_usd(10000),
            cash_balance=quantize_usd(15000),
            open_position_count=0,
            open_cost_basis_usd=quantize_usd(0),
            open_market_value_usd=quantize_usd(0),
            realized_pnl_usd=quantize_usd(-24),
            unrealized_pnl_usd=quantize_usd(0),
            account_equity_usd=quantize_usd(15000),
            expected_cash_balance=quantize_usd(15000),
            cash_diff=quantize_usd(0),
            ledger_consistency_status="PASS",
            fee_model_status="ZERO_FEES_CONFIGURED",
            entry_fee_bps=0,
            exit_fee_bps=0,
            slippage_bps=0,
        )
        write_ledger_consistency_audit(
            path, loop_run_id="r2", invocation_id="i2", snapshot=bad
        )
        assert bad.ledger_consistency_status == "FAIL"
        text = path.read_text(encoding="utf-8")
        assert "ledger_schema_version" in text.splitlines()[0]
        assert "IMPOSSIBLE_EQUITY_WITHOUT_DEPOSIT" in text

    def test_lifecycle_noop_row_when_no_open_positions(self, tmp_path):
        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.persistence import IterationWriters
        from app.runtime_paper_loop.position_lifecycle import evaluate_and_close_positions

        db_path = tmp_path / "state.sqlite"
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        db = AE11StateDb(db_path)
        config = Ae11LoopConfig(project_root=tmp_path)
        result = evaluate_and_close_positions(
            db,
            DemoPriceOracle(),
            IterationWriters(),
            config=config,
            loop_run_id="r",
            invocation_id="i",
            iteration=1,
            project_root=tmp_path,
        )
        assert result["lifecycle_noop_reason"] == "NO_OPEN_POSITIONS"
        assert result["lifecycle_audit_rows"] >= 1
        audit = (tmp_path / "audits" / "ae11_position_lifecycle_audit.csv").read_text(
            encoding="utf-8"
        )
        assert "NO_OPEN_POSITIONS" in audit
        db.close()

    def test_migration_not_inside_loop_iteration(self, tmp_path):
        import inspect

        from app.runtime_paper_loop import loop_runner as lr

        src = inspect.getsource(lr.RuntimePaperLoopRunner.run_iteration)
        assert "migrate_db_schema" not in src
        startup_src = inspect.getsource(lr.RuntimePaperLoopRunner.startup)
        assert "migrate_db_schema" in startup_src


class TestAE11GPositionStateSemantics:
    def test_get_open_positions_excludes_closed_registry_rows(self, tmp_path):
        from app.runtime_paper_loop.db_migration import migrate_db_schema

        db_path = tmp_path / "state.sqlite"
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        db = AE11StateDb(db_path)
        db.register_position("open-1", "0xA", paper_order_id="o1", source_decision_id="d1")
        db.register_position("closed-1", "0xB", paper_order_id="o2", source_decision_id="d2")
        db.close_position("closed-1", "0xB")

        opens = db.get_open_positions()
        assert [p["position_id"] for p in opens] == ["open-1"]
        assert db.count_open_positions() == 1
        assert db.count_position_registry_rows() == 2
        assert db.count_closed_rows_in_registry() == 1
        assert db.active_open_position_count() == 1
        db.close()

    def test_capacity_uses_open_count_not_registry_total(self, tmp_path):
        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.decision_policy import evaluate_strict_shadow_decision

        db_path = tmp_path / "state.sqlite"
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        db = AE11StateDb(db_path)
        for i in range(3):
            db.register_position(
                f"o{i}", f"0x{i}", paper_order_id=f"ord{i}", source_decision_id=f"d{i}"
            )
        for i in range(5):
            db.register_position(
                f"c{i}", f"0xC{i}", paper_order_id=f"cord{i}", source_decision_id=f"cd{i}"
            )
            db.close_position(f"c{i}", f"0xC{i}")

        assert db.count_position_registry_rows() == 8
        open_n = db.count_open_positions()
        assert open_n == 3
        # Capacity at max=3 should block using OPEN count (3), not registry total (8)
        result = evaluate_strict_shadow_decision(
            decision=_sample_decision(),
            traceability={
                "candidate_id": "c",
                "source_decision_id": "d",
                "source_context_record_id": "ctx",
            },
            price_freshness=_price_freshness_ok(),
            open_position_count=open_n,
            max_open_positions=3,
            has_active_pair_lock=False,
            cooldown_active=False,
            already_processed=False,
            missing_identity=False,
        )
        assert result.max_open_positions_hit is True
        db.close()

    def test_ae11g_migration_indexes_idempotent(self, tmp_path):
        import sqlite3
        from app.runtime_paper_loop.db_migration import (
            MIGRATION_NAME_AE11G,
            migrate_db_schema,
        )

        db_path = tmp_path / "state.sqlite"
        r1 = migrate_db_schema(db_path=db_path, project_root=tmp_path)
        assert r1.get("ae11g", {}).get("status") in ("APPLIED", "SKIPPED")
        r2 = migrate_db_schema(db_path=db_path, project_root=tmp_path)
        assert r2["ae11g"]["migration_applied"] is False
        assert r2["ae11g"]["migration_skipped_reason"] == "ALREADY_APPLIED"

        conn = sqlite3.connect(str(db_path))
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert "idx_active_positions_status" in names
        assert "idx_active_positions_open" in names
        assert "idx_active_pair_locks_position_id" in names
        assert conn.execute(
            "SELECT 1 FROM schema_migrations WHERE migration_name = ?",
            (MIGRATION_NAME_AE11G,),
        ).fetchone()
        conn.close()

    def test_ghost_lock_fails_then_repair_audited(self, tmp_path):
        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.position_state_semantics import (
            audit_position_state_semantics,
        )

        db_path = tmp_path / "state.sqlite"
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        db = AE11StateDb(db_path)
        db.register_position("p1", "0xA", paper_order_id="o1", source_decision_id="d1")
        db.close_position("p1", "0xA")
        # Re-insert a ghost lock manually
        db._conn.execute(
            "INSERT INTO active_pair_locks (pair_address, position_id, locked_at_utc) VALUES (?,?,?)",
            ("0xA", "p1", "2026-01-01T00:00:00+00:00"),
        )
        db._conn.commit()
        assert db.find_ghost_locks()

        result = audit_position_state_semantics(
            db,
            loop_run_id="r",
            invocation_id="i",
            max_open_positions=10,
            open_snapshot_rows=0,
            cumulative_metrics_open_positions=0,
            repair_ghost_locks=True,
            project_root=tmp_path,
        )
        assert result.ghost_lock_repair_count >= 1
        assert result.locks_pointing_to_closed_count == 0
        assert result.ghost_lock_count == 0
        assert result.position_state_semantics_status in (
            "PASS",
            "PASS_WITH_REGISTRY_SEMANTICS",
        )
        audit = (tmp_path / "audits" / "ae11_position_state_semantics_audit.csv").read_text(
            encoding="utf-8"
        )
        assert "GHOST_LOCKS_DELETED" in audit or "ghost_locks_repaired" in audit
        db.close()

    def test_open_closed_overlap_fails(self, tmp_path):
        from uuid import uuid4
        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.position_state_semantics import (
            audit_position_state_semantics,
        )

        db_path = tmp_path / "state.sqlite"
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        db = AE11StateDb(db_path)
        pid = str(uuid4())
        db.register_position(pid, "0xZ", paper_order_id="o", source_decision_id="d")
        db.record_economic_close(
            {
                "position_id": pid,
                "close_event_id": str(uuid4()),
                "economic_close_key": pid,
                "closed_at_utc": "2026-01-01T00:00:00+00:00",
                "exit_reason": "TIME_STOP",
                "entry_price": "1",
                "exit_price": "1",
                "quantity": "1",
                "notional_usd": "100",
                "cash_debited_usd": "100",
                "cash_credited_usd": "100",
            }
        )
        # Still OPEN in registry but present in closed_positions → FAIL
        result = audit_position_state_semantics(
            db,
            loop_run_id="r",
            invocation_id="i",
            max_open_positions=10,
            open_snapshot_rows=1,
            cumulative_metrics_open_positions=1,
            repair_ghost_locks=False,
            project_root=tmp_path,
        )
        assert result.position_state_semantics_status == "FAIL"
        assert "OPEN_AND_CLOSED_POSITIONS_OVERLAP" in (result.mismatch_type or "")
        db.close()

    def test_closed_registry_plus_closed_positions_accepted(self, tmp_path):
        from uuid import uuid4
        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.position_state_semantics import (
            audit_position_state_semantics,
        )

        db_path = tmp_path / "state.sqlite"
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        db = AE11StateDb(db_path)
        pid = str(uuid4())
        db.register_position(pid, "0xZ", paper_order_id="o", source_decision_id="d")
        db.record_economic_close(
            {
                "position_id": pid,
                "close_event_id": str(uuid4()),
                "economic_close_key": pid,
                "closed_at_utc": "2026-01-01T00:00:00+00:00",
                "exit_reason": "TIME_STOP",
                "entry_price": "1",
                "exit_price": "1",
                "quantity": "1",
                "notional_usd": "100",
                "cash_debited_usd": "100",
                "cash_credited_usd": "100",
            }
        )
        db.close_position(pid, "0xZ")
        result = audit_position_state_semantics(
            db,
            loop_run_id="r",
            invocation_id="i",
            max_open_positions=10,
            open_snapshot_rows=0,
            cumulative_metrics_open_positions=0,
            repair_ghost_locks=True,
            project_root=tmp_path,
        )
        assert result.position_state_semantics_status == "PASS_WITH_REGISTRY_SEMANTICS"
        assert result.open_positions_count == 0
        assert result.closed_rows_in_active_positions_count == 1
        db.close()

    def test_snapshot_mismatch_is_fail(self, tmp_path):
        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.position_state_semantics import (
            audit_position_state_semantics,
        )

        db_path = tmp_path / "state.sqlite"
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        db = AE11StateDb(db_path)
        db.register_position("o1", "0xA", paper_order_id="o", source_decision_id="d")
        result = audit_position_state_semantics(
            db,
            loop_run_id="r",
            invocation_id="i",
            max_open_positions=10,
            open_snapshot_rows=0,  # wrong vs OPEN=1
            cumulative_metrics_open_positions=1,
            repair_ghost_locks=False,
            project_root=tmp_path,
        )
        assert result.position_state_semantics_status == "FAIL"
        assert "OPEN_SNAPSHOT_COUNT_MISMATCH" in (result.mismatch_type or "")
        db.close()

    def test_ae11g_migration_not_in_loop(self):
        import inspect
        from app.runtime_paper_loop import loop_runner as lr

        assert "migrate_db_schema" not in inspect.getsource(
            lr.RuntimePaperLoopRunner.run_iteration
        )
        assert "migrate_db_schema" in inspect.getsource(lr.RuntimePaperLoopRunner.startup)


class TestAE11IPriceOracleTPSL:
    def _seed_pos(self, db, pid, pair, entry="1.0", tp="1.2", sl="0.9", notional="100"):
        from app.runtime_paper_loop.decimal_money import (
            decimal_to_str,
            quantize_usd,
            to_decimal,
        )

        e = to_decimal(entry)
        n = to_decimal(notional)
        fee = quantize_usd(n * to_decimal("0.003"))
        slip = quantize_usd(n * to_decimal("0.005"))
        qty = quantize_usd(n / e)
        db.register_position(
            pid,
            pair,
            paper_order_id=f"ord-{pid}",
            source_decision_id=f"dec-{pid}",
            economics={
                "entry_price": entry,
                "quantity": decimal_to_str(qty),
                "notional_usd": notional,
                "cost_basis_usd": decimal_to_str(n + fee),
                "entry_fee_usd": decimal_to_str(fee),
                "entry_slippage_usd": decimal_to_str(slip),
                "cash_debited_usd": decimal_to_str(n + fee + slip),
                "open_market_value_usd": notional,
                "tp_price": tp,
                "sl_price": sl,
                "time_stop_at_utc": "2099-01-01T00:00:00+00:00",
                "opened_at_utc": "2026-01-01T00:00:00+00:00",
                "economic_enrichment_status": "FULL",
            },
        )

    def test_deterministic_neutral_and_temporal_rules(self):
        from app.runtime_paper_loop.ae11_price_oracle import (
            Ae11PriceOracle,
            validate_temporal_validity,
        )

        oracle = Ae11PriceOracle(
            valuation_provider="deterministic",
            deterministic_price_scenario="neutral",
        )
        pos = {
            "position_id": "p1",
            "pair_address": "0xA",
            "entry_price": "1.0",
            "tp_price": "1.2",
            "sl_price": "0.9",
            "opened_at_utc": "2026-01-01T00:00:00+00:00",
            "economic_enrichment_status": "FULL",
        }
        q = oracle.resolve_current_price(pos, "2026-01-02T00:00:00+00:00")
        assert q.resolution_status == "PRICE_RESOLVED_DETERMINISTIC_TEST"
        assert float(q.current_price) == 1.0
        assert q.is_deterministic_test_quote is True
        assert q.real_market_price is False
        assert q.no_lookahead_status == "NOT_APPLICABLE_DETERMINISTIC_TEST"

        tv, _nl = validate_temporal_validity(
            opened_at_utc="2026-01-02T00:00:00+00:00",
            price_timestamp_utc="2026-01-01T00:00:00+00:00",
            evaluation_at_utc="2026-01-03T00:00:00+00:00",
        )
        assert tv == "BLOCKED_PRE_ENTRY_PRICE"
        tv2, nl2 = validate_temporal_validity(
            opened_at_utc="2026-01-01T00:00:00+00:00",
            price_timestamp_utc="2026-01-05T00:00:00+00:00",
            evaluation_at_utc="2026-01-03T00:00:00+00:00",
        )
        assert tv2 == "FAIL_PRICE_AFTER_EVALUATION_TIME"
        assert nl2 == "FAIL_PRICE_AFTER_EVALUATION_TIME"
        tv3, _ = validate_temporal_validity(
            opened_at_utc="2026-01-01T00:00:00+00:00",
            price_timestamp_utc="2026-01-02T00:00:00+00:00",
            evaluation_at_utc="2026-01-03T00:00:00+00:00",
            price_ingested_at_utc="2026-01-04T00:00:00+00:00",
        )
        assert tv3 == "BLOCKED_UNAVAILABLE_AT_EVALUATION"

    def test_force_trigger_tp_sl_and_bridge(self, tmp_path):
        from app.paper_trading.price_oracle import DemoPriceOracle
        from app.runtime_paper_loop.ae11_price_oracle import Ae11PriceOracle
        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.equity_bridge import build_equity_bridge
        from app.runtime_paper_loop.position_lifecycle import evaluate_and_close_positions
        from app.runtime_paper_loop.persistence import IterationWriters
        from app.runtime_paper_loop.types import Ae11LoopConfig

        db_path = tmp_path / "state.sqlite"
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        db = AE11StateDb(db_path)
        self._seed_pos(db, "ptp", "0xTP", entry="1.0", tp="1.2", sl="0.9")

        # Incremental: steps below TP then force cross
        oracle = Ae11PriceOracle(
            valuation_provider="deterministic",
            deterministic_price_scenario="incremental_tp",
            deterministic_price_step_pct=5.0,
            incremental_step=0,
        )
        config = Ae11LoopConfig(
            project_root=tmp_path,
            take_profit_pct=20.0,
            stop_loss_pct=10.0,
            time_stop_minutes=999999,
            deterministic_price_scenario="incremental_tp",
        )
        closed_any = False
        for _ in range(8):
            # evaluate_and_close advances step for incremental_* once per call
            result = evaluate_and_close_positions(
                db,
                DemoPriceOracle(),
                IterationWriters(),
                config=config,
                loop_run_id="r",
                invocation_id="i",
                iteration=oracle.incremental_step,
                project_root=tmp_path,
                valuation_oracle=oracle,
            )
            if result["positions_closed"] > 0:
                closed_any = True
                assert result["tp_trigger_count"] >= 1
                break
        assert closed_any
        rows = db._conn.execute("SELECT exit_reason FROM closed_positions").fetchall()
        assert any(r["exit_reason"] == "TAKE_PROFIT" for r in rows)
        assert db.count_open_positions() == 0
        locks = db._conn.execute("SELECT COUNT(*) AS c FROM active_pair_locks").fetchone()
        assert int(locks["c"]) == 0
        bridge = build_equity_bridge(db, starting_balance_usd=10000)
        assert bridge.bridge_status == "PASS"
        db.close()

        # SL force trigger
        db2_path = tmp_path / "state_sl.sqlite"
        migrate_db_schema(db_path=db2_path, project_root=tmp_path)
        db2 = AE11StateDb(db2_path)
        self._seed_pos(db2, "psl", "0xSL", entry="1.0", tp="1.2", sl="0.9")
        oracle2 = Ae11PriceOracle(
            valuation_provider="deterministic",
            deterministic_price_scenario="sl",
        )
        config2 = Ae11LoopConfig(project_root=tmp_path, time_stop_minutes=999999)
        r2 = evaluate_and_close_positions(
            db2,
            DemoPriceOracle(),
            IterationWriters(),
            config=config2,
            loop_run_id="r2",
            invocation_id="i2",
            iteration=1,
            project_root=tmp_path,
            valuation_oracle=oracle2,
        )
        assert r2["sl_trigger_count"] >= 1
        assert r2["price_based_positions_closed"] >= 1
        row = db2._conn.execute(
            "SELECT exit_reason, exit_price FROM closed_positions"
        ).fetchone()
        assert row["exit_reason"] == "STOP_LOSS"
        assert float(row["exit_price"]) <= 0.9
        bridge2 = build_equity_bridge(db2, starting_balance_usd=10000)
        assert bridge2.bridge_status == "PASS"
        assert oracle2.session_stats.no_double_count_status == "PASS"
        db2.close()

    def test_mtm_neutral_preserves_equity_fields(self, tmp_path):
        from app.paper_trading.price_oracle import DemoPriceOracle
        from app.runtime_paper_loop.ae11_price_oracle import Ae11PriceOracle
        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.position_lifecycle import evaluate_and_close_positions
        from app.runtime_paper_loop.persistence import IterationWriters
        from app.runtime_paper_loop.types import Ae11LoopConfig

        db_path = tmp_path / "state.sqlite"
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        db = AE11StateDb(db_path)
        self._seed_pos(db, "pmtm", "0xM", entry="1.0", tp="10.0", sl="0.01")
        oracle = Ae11PriceOracle(
            valuation_provider="deterministic",
            deterministic_price_scenario="neutral",
        )
        config = Ae11LoopConfig(project_root=tmp_path, time_stop_minutes=999999)
        evaluate_and_close_positions(
            db,
            DemoPriceOracle(),
            IterationWriters(),
            config=config,
            loop_run_id="r",
            invocation_id="i",
            iteration=1,
            project_root=tmp_path,
            valuation_oracle=oracle,
        )
        open_rows = db.get_open_positions()
        assert len(open_rows) == 1
        assert float(open_rows[0]["open_market_value_usd"]) == 100.0
        assert float(open_rows[0].get("price_unrealized_pnl_usd") or 0) == 0.0
        assert open_rows[0].get("valuation_source") == "DETERMINISTIC_TEST_ORACLE"
        db.close()

    def test_missing_local_snapshot_not_silent_zero(self):
        from app.runtime_paper_loop.ae11_price_oracle import Ae11PriceOracle

        oracle = Ae11PriceOracle(valuation_provider="local_snapshot")
        q = oracle.resolve_current_price(
            {
                "position_id": "px",
                "pair_address": "0x",
                "entry_price": "1",
                "opened_at_utc": "2026-01-01T00:00:00+00:00",
                "economic_enrichment_status": "FULL",
            },
            "2026-01-02T00:00:00+00:00",
        )
        assert q.resolution_status == "PRICE_MISSING"
        assert q.current_price is None
        assert q.missing_reason == "LOCAL_SNAPSHOT_UNAVAILABLE"

    def test_incremental_tp_does_not_close_before_threshold(self, tmp_path):
        from app.paper_trading.price_oracle import DemoPriceOracle
        from app.runtime_paper_loop.ae11_price_oracle import Ae11PriceOracle
        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.position_lifecycle import evaluate_and_close_positions
        from app.runtime_paper_loop.persistence import IterationWriters
        from app.runtime_paper_loop.types import Ae11LoopConfig

        db_path = tmp_path / "state.sqlite"
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        db = AE11StateDb(db_path)
        # TP at +20%; step 5% → need several steps before crossing
        self._seed_pos(db, "pinc", "0xINC", entry="1.0", tp="1.20", sl="0.50")
        oracle = Ae11PriceOracle(
            valuation_provider="deterministic",
            deterministic_price_scenario="incremental_tp",
            deterministic_price_step_pct=5.0,
            incremental_step=0,
        )
        config = Ae11LoopConfig(project_root=tmp_path, time_stop_minutes=999999)
        # First evaluation uses advanced step=1 → price 1.05 < 1.20 → remain open
        r0 = evaluate_and_close_positions(
            db,
            DemoPriceOracle(),
            IterationWriters(),
            config=config,
            loop_run_id="r",
            invocation_id="i",
            iteration=0,
            project_root=tmp_path,
            valuation_oracle=oracle,
        )
        assert r0["positions_closed"] == 0
        assert db.count_open_positions() == 1
        open0 = db.get_open_positions()[0]
        assert float(open0["last_price"]) < 1.20
        assert float(open0.get("price_unrealized_pnl_usd") or 0) != 0.0 or float(
            open0["last_price"]
        ) != 1.0

        closed = False
        for _ in range(10):
            r = evaluate_and_close_positions(
                db,
                DemoPriceOracle(),
                IterationWriters(),
                config=config,
                loop_run_id="r",
                invocation_id="i",
                iteration=oracle.incremental_step,
                project_root=tmp_path,
                valuation_oracle=oracle,
            )
            if r["positions_closed"] > 0:
                closed = True
                assert r["tp_trigger_count"] >= 1
                break
        assert closed
        row = db._conn.execute(
            "SELECT exit_reason, exit_price FROM closed_positions"
        ).fetchone()
        assert row["exit_reason"] == "TAKE_PROFIT"
        assert float(row["exit_price"]) >= 1.20
        db.close()

    def test_mixed_scenario_tp_sl_neutral(self, tmp_path):
        from app.paper_trading.price_oracle import DemoPriceOracle
        from app.runtime_paper_loop.ae11_price_oracle import Ae11PriceOracle
        from app.runtime_paper_loop.db_migration import migrate_db_schema
        from app.runtime_paper_loop.position_lifecycle import evaluate_and_close_positions
        from app.runtime_paper_loop.persistence import IterationWriters
        from app.runtime_paper_loop.types import Ae11LoopConfig

        db_path = tmp_path / "state.sqlite"
        migrate_db_schema(db_path=db_path, project_root=tmp_path)
        db = AE11StateDb(db_path)
        for i in range(6):
            self._seed_pos(db, f"pm{i}", f"0xM{i}", entry="1.0", tp="1.2", sl="0.9")
        oracle = Ae11PriceOracle(
            valuation_provider="deterministic",
            deterministic_price_scenario="mixed",
        )
        # Force deterministic role mix (avoid hash flake)
        oracle._mixed_roles = {
            "pm0": "tp",
            "pm1": "tp",
            "pm2": "sl",
            "pm3": "sl",
            "pm4": "neutral",
            "pm5": "neutral",
        }
        config = Ae11LoopConfig(project_root=tmp_path, time_stop_minutes=999999)
        result = evaluate_and_close_positions(
            db,
            DemoPriceOracle(),
            IterationWriters(),
            config=config,
            loop_run_id="r",
            invocation_id="i",
            iteration=1,
            project_root=tmp_path,
            valuation_oracle=oracle,
        )
        assert result["tp_trigger_count"] >= 1
        assert result["sl_trigger_count"] >= 1
        assert db.count_open_positions() >= 1  # at least one neutral remains
        reasons = {
            r["exit_reason"]
            for r in db._conn.execute("SELECT exit_reason FROM closed_positions")
        }
        assert "TAKE_PROFIT" in reasons
        assert "STOP_LOSS" in reasons
        db.close()

    def test_price_lifecycle_proof_mode_forces_deterministic(self):
        from app.runtime_paper_loop.ae11_price_oracle import build_ae11_price_oracle
        from app.runtime_paper_loop.types import Ae11LoopConfig

        cfg = Ae11LoopConfig(
            valuation_provider="legacy",
            price_lifecycle_proof_mode=True,
            deterministic_price_scenario="tp",
        )
        oracle = build_ae11_price_oracle(cfg)
        assert oracle.valuation_provider == "deterministic"
        assert oracle.price_lifecycle_proof_mode is True
