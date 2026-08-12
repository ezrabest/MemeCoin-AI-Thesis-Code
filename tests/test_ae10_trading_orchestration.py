"""Tests for AE10 trading orchestration layer."""

from __future__ import annotations

import json
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

import pytest

from app.execution.adapters import PaperExecutionAdapter
from app.execution.execution_orchestrator import (
    build_traceability_record,
    run_ae10_trading_orchestration,
)
from app.execution.live_no_wallet_adapter import LiveNoWalletDryRunAdapter
from app.execution.types import OrderIntent
from app.paper_trading.ledger import DemoLedger, reset_demo_account
from app.paper_trading.order_simulator import PaperOrderSimulator, compute_execution_latency
from app.paper_trading.order_state_machine import OrderStateMachine
from app.paper_trading.persistence import JsonlWriter, read_jsonl_safe
from app.paper_trading.price_oracle import DemoPriceOracle
from app.paper_trading.types import DemoAccount, PaperOrder, PaperOrderStatus, PaperPosition, PriceStatus


def _sample_decision(**overrides) -> dict:
    base = {
        "decision_id": "dec-001",
        "created_at_utc": "2026-07-10T09:00:52+00:00",
        "decision_status": "RESEARCH_CANDIDATE",
        "candidate_identity": {
            "candidate_id": "cand-001",
            "pair_address": "0xABC",
            "symbol": "PEPE/WETH",
            "chain": "ethereum",
            "coin_id": 1,
            "event_timestamp": "2026-07-10T09:00:00+00:00",
        },
        "consensus": {"consensus_family": "NO_MODEL_CONSENSUS_AVAILABLE"},
        "market_context": {"price": 0.001},
    }
    base.update(overrides)
    return base


def _sample_context(**overrides) -> dict:
    base = {
        "context_record_id": "ctx-001",
        "candidate_id": "cand-001",
        "context_schema_id": "schema-001",
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
        "audit_warnings": [],
    }
    base.update(overrides)
    return base


def _fresh_snapshot(order_ts: str, price: float = 0.001) -> dict:
    order_dt = datetime.fromisoformat(order_ts.replace("Z", "+00:00"))
    snap_dt = order_dt - timedelta(seconds=5)
    return {
        "id": 100,
        "coin_id": 1,
        "timestamp": snap_dt.isoformat(),
        "price": price,
    }


class TestDemoAccount:
    def test_initializes_deterministically(self):
        a1 = DemoAccount(starting_balance_usd=10_000.0)
        a2 = DemoAccount.from_dict(a1.to_dict())
        assert a2.starting_balance_usd == 10_000.0
        assert a2.cash_balance_usd == 10_000.0
        assert a2.reset_count == 0

    def test_reset_demo_funds(self):
        ledger = DemoLedger(account=DemoAccount(cash_balance_usd=5000.0))
        ledger.account.reset_count = 2
        audit = reset_demo_account(ledger, starting_balance_usd=10_000.0)
        assert ledger.account.cash_balance_usd == 10_000.0
        assert ledger.account.reset_count == 3
        assert audit["reset_executed"] is True
        assert audit["live_wallet_settings_affected"] is False

    def test_reset_preserves_history_by_default(self):
        ledger = DemoLedger()
        from app.paper_trading.types import PaperOrder

        ledger.orders.append(PaperOrder(candidate_id="x"))
        reset_demo_account(ledger)
        assert len(ledger.orders) == 1

    def test_reset_clears_history_when_requested(self):
        ledger = DemoLedger()
        from app.paper_trading.types import PaperOrder

        ledger.orders.append(PaperOrder(candidate_id="x"))
        reset_demo_account(ledger, clear_history=True)
        assert len(ledger.orders) == 0


class TestOrderStateMachine:
    def setup_method(self):
        self.sm = OrderStateMachine()

    def test_pending_to_filled_allowed(self):
        r = self.sm.transition("ord-1", "PAPER_PENDING", "PAPER_FILLED")
        assert r.allowed is True

    def test_pending_to_closed_tp_rejected(self):
        r = self.sm.transition("ord-1", "PAPER_PENDING", "PAPER_CLOSED_TP")
        assert r.allowed is False
        assert r.status == "PAPER_STATE_TRANSITION_REJECTED"

    def test_rejected_to_filled_rejected(self):
        r = self.sm.transition("ord-1", "PAPER_REJECTED", "PAPER_FILLED")
        assert r.allowed is False

    def test_closed_to_filled_rejected(self):
        r = self.sm.transition("ord-1", "PAPER_CLOSED_TP", "PAPER_FILLED")
        assert r.allowed is False


class TestPriceOracle:
    def test_price_ok_within_age(self):
        oracle = DemoPriceOracle(max_price_age_seconds=30.0)
        order_ts = "2026-07-10T09:00:52+00:00"
        oracle.snapshots = [_fresh_snapshot(order_ts)]
        result = oracle.lookup_price(coin_id=1, order_timestamp=order_ts)
        assert result.price_status == PriceStatus.PRICE_OK.value
        assert result.price == 0.001
        assert result.price_age_seconds <= 30.0

    def test_missing_price_rejects(self):
        oracle = DemoPriceOracle(max_price_age_seconds=30.0)
        result = oracle.lookup_price(coin_id=1, order_timestamp="2026-07-10T09:00:52+00:00")
        assert result.price_status == PriceStatus.PRICE_MISSING.value

    def test_stale_price_rejects(self):
        oracle = DemoPriceOracle(max_price_age_seconds=30.0)
        order_ts = "2026-07-10T09:00:52+00:00"
        order_dt = datetime.fromisoformat(order_ts.replace("Z", "+00:00"))
        stale_dt = order_dt - timedelta(seconds=60)
        oracle.snapshots = [
            {"id": 1, "coin_id": 1, "timestamp": stale_dt.isoformat(), "price": 0.001}
        ]
        result = oracle.lookup_price(coin_id=1, order_timestamp=order_ts)
        assert result.price_status == PriceStatus.PRICE_STALE.value

    def test_future_price_lookahead_rejected(self):
        oracle = DemoPriceOracle(max_price_age_seconds=30.0)
        order_ts = "2026-07-10T09:00:52+00:00"
        order_dt = datetime.fromisoformat(order_ts.replace("Z", "+00:00"))
        future_dt = order_dt + timedelta(seconds=10)
        oracle.snapshots = [
            {"id": 1, "coin_id": 1, "timestamp": future_dt.isoformat(), "price": 0.001}
        ]
        result = oracle.lookup_price(coin_id=1, order_timestamp=order_ts)
        assert result.price_status == PriceStatus.PRICE_LOOKAHEAD_REJECTED.value


class TestExecutionLatency:
    def test_computed_when_timestamps_exist(self):
        ms, status = compute_execution_latency(
            "2026-07-10T09:00:00+00:00",
            "2026-07-10T09:00:01+00:00",
        )
        assert ms == 1000.0
        assert status == "OK"

    def test_missing_decision_timestamp(self):
        ms, status = compute_execution_latency(None, "2026-07-10T09:00:01+00:00")
        assert ms is None
        assert status == "MISSING_DECISION_TIMESTAMP"

    def test_not_filled(self):
        ms, status = compute_execution_latency("2026-07-10T09:00:00+00:00", None)
        assert ms is None
        assert status == "NOT_FILLED"


class TestPaperOrderSimulator:
    def _sim_with_price(self, order_ts: str = "2026-07-10T09:00:52+00:00"):
        oracle = DemoPriceOracle(max_price_age_seconds=30.0)
        oracle.snapshots = [_fresh_snapshot(order_ts)]
        return PaperOrderSimulator(price_oracle=oracle)

    def test_paper_order_contains_traceability_fields(self):
        sim = self._sim_with_price()
        trace = build_traceability_record(
            _sample_decision(), _sample_context(), _sample_audit()
        ).to_dict()
        price = sim.price_oracle.lookup_price(coin_id=1, order_timestamp="2026-07-10T09:00:52+00:00")
        order = sim.create_and_fill_order(
            trace,
            price_result=price.to_dict(),
            symbol="PEPE/WETH",
            pair_address="0xABC",
            decision_created_at_utc="2026-07-10T09:00:52+00:00",
        )
        assert order.candidate_id == "cand-001"
        assert order.source_decision_id == "dec-001"
        assert order.source_context_record_id == "ctx-001"
        assert order.source_llm_audit_record_id == "aud-001"
        assert order.no_live_trading is True

    def test_missing_candidate_id_rejects(self):
        sim = self._sim_with_price()
        trace = {"candidate_id": "", "audit_blockers": []}
        price = sim.price_oracle.lookup_price(coin_id=1, order_timestamp="2026-07-10T09:00:52+00:00")
        order = sim.create_and_fill_order(trace, price_result=price.to_dict())
        assert order.status == PaperOrderStatus.PAPER_REJECTED.value

    def test_audit_blockers_require_override(self):
        sim = self._sim_with_price()
        trace = build_traceability_record(
            _sample_decision(),
            _sample_context(),
            _sample_audit(audit_blockers=["weak_lineage"]),
        ).to_dict()
        price = sim.price_oracle.lookup_price(coin_id=1, order_timestamp="2026-07-10T09:00:52+00:00")
        order = sim.create_and_fill_order(trace, price_result=price.to_dict())
        assert order.status == PaperOrderStatus.PAPER_REJECTED.value

    def test_override_marks_exploration(self):
        sim = self._sim_with_price()
        trace = build_traceability_record(
            _sample_decision(),
            _sample_context(),
            _sample_audit(audit_blockers=["weak_lineage"]),
        ).to_dict()
        price = sim.price_oracle.lookup_price(coin_id=1, order_timestamp="2026-07-10T09:00:52+00:00")
        order = sim.create_and_fill_order(
            trace,
            price_result=price.to_dict(),
            allow_audit_blockers=True,
            decision_created_at_utc="2026-07-10T09:00:52+00:00",
        )
        assert order.status == PaperOrderStatus.PAPER_FILLED.value
        assert order.override_type == "DEMO_ONLY_USER_APPROVED_EXPLORATION"
        assert order.not_model_approved is True
        assert order.paper_trade_reason == "DEMO_SANDBOX_EXPLORATION"

    def test_llm_verdict_cannot_approve_alone(self):
        sim = self._sim_with_price()
        trace = build_traceability_record(
            _sample_decision(),
            _sample_context(),
            _sample_audit(audit_verdict="BUY", llm_verdict="BUY"),
        ).to_dict()
        trace["audit_verdict"] = "BUY"
        price = sim.price_oracle.lookup_price(coin_id=1, order_timestamp="2026-07-10T09:00:52+00:00")
        order = sim.create_and_fill_order(trace, price_result=price.to_dict())
        assert order.status == PaperOrderStatus.PAPER_REJECTED.value


class TestLiveNoWalletAdapter:
    def test_never_signs_or_submits(self):
        adapter = LiveNoWalletDryRunAdapter()
        intent = OrderIntent(
            candidate_id="cand-001",
            symbol="PEPE",
            pair_address="0xABC",
            decision_created_at_utc="2026-07-10T09:00:52+00:00",
        )
        result = adapter.execute(intent)
        assert result.real_transaction_attempted is False
        assert result.live_submission_status == "NOT_SUBMITTED_NO_WALLET"
        assert result.execution_mode == "LIVE_NO_WALLET_DRY_RUN"
        assert result.wallet_configured is False
        assert adapter.private_key_accessed is False


class TestJsonlWriter:
    def test_flush_fsync_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.jsonl"
            with JsonlWriter(path) as writer:
                writer.append_dict({"key": "value"})
            records, _ = read_jsonl_safe(path)
            assert len(records) == 1
            assert records[0]["key"] == "value"


class TestOrchestratorIntegration:
    def _write_fixtures(self, tmp: Path) -> tuple[Path, Path, Path]:
        ae6 = tmp / "ae6.jsonl"
        ae8 = tmp / "ae8.jsonl"
        ae9 = tmp / "ae9.jsonl"
        decision = _sample_decision()
        context = _sample_context()
        audit = _sample_audit()
        ae6.write_text(json.dumps(decision) + "\n")
        ae8.write_text(json.dumps(context) + "\n")
        ae9.write_text(json.dumps(audit) + "\n")
        return ae6, ae8, ae9

    def test_no_paper_orders_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ae6, ae8, ae9 = self._write_fixtures(tmp_path)
            summary = run_ae10_trading_orchestration(
                project_root=tmp_path,
                max_records=5,
                audit_only=True,
                no_db_write=True,
                enable_paper_demo_orders=False,
                ae6_jsonl=ae6,
                ae8_context_jsonl=ae8,
                ae9_audit_jsonl=ae9,
            )
            assert summary["paper_orders_created"] == 0
            assert summary["traceability_records_created"] == 1

    def test_paper_orders_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ae6, ae8, ae9 = self._write_fixtures(tmp_path)
            order_ts = decision_ts = "2026-07-10T09:00:52+00:00"
            snap = _fresh_snapshot(order_ts)
            with mock.patch(
                "app.execution.execution_orchestrator.DemoPriceOracle.lookup_price"
            ) as mock_lookup:
                from app.paper_trading.price_oracle import PriceLookupResult

                mock_lookup.return_value = PriceLookupResult(
                    price=0.001,
                    price_snapshot_id=100,
                    price_timestamp=snap["timestamp"],
                    order_timestamp=order_ts,
                    price_age_seconds=5.0,
                    max_price_age_seconds=30.0,
                    price_status=PriceStatus.PRICE_OK.value,
                )
                summary = run_ae10_trading_orchestration(
                    project_root=tmp_path,
                    max_records=5,
                    audit_only=True,
                    no_db_write=True,
                    enable_paper_demo_orders=True,
                    allow_paper_trades_with_audit_blockers=True,
                    ae6_jsonl=ae6,
                    ae8_context_jsonl=ae8,
                    ae9_audit_jsonl=ae9,
                )
            assert summary["paper_orders_created"] >= 1

    def test_live_dry_run_only_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ae6, ae8, ae9 = self._write_fixtures(tmp_path)
            summary = run_ae10_trading_orchestration(
                project_root=tmp_path,
                max_records=5,
                audit_only=True,
                no_db_write=True,
                enable_live_dry_run_orders=True,
                ae6_jsonl=ae6,
                ae8_context_jsonl=ae8,
                ae9_audit_jsonl=ae9,
            )
            assert summary["live_dry_run_orders_created"] == 1
            assert summary["live_submission_status"] == "NOT_SUBMITTED_NO_WALLET"

    def test_output_paths_concrete(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ae6, ae8, ae9 = self._write_fixtures(tmp_path)
            summary = run_ae10_trading_orchestration(
                project_root=tmp_path,
                max_records=5,
                audit_only=True,
                no_db_write=True,
                ae6_jsonl=ae6,
                ae8_context_jsonl=ae8,
                ae9_audit_jsonl=ae9,
            )
            assert summary["output_root"]
            assert summary["output_paths"]
            gate_path = summary["output_paths"].get("ae10_decision_gate")
            assert gate_path
            assert Path(gate_path).is_file()

    def test_no_external_llm_calls_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ae6, ae8, ae9 = self._write_fixtures(tmp_path)
            summary = run_ae10_trading_orchestration(
                project_root=tmp_path,
                max_records=5,
                provider="mock",
                ae6_jsonl=ae6,
                ae8_context_jsonl=ae8,
                ae9_audit_jsonl=ae9,
            )
            assert summary["llm_provider_summary"]["external_calls_made"] == 0
            assert summary["safety_confirmation"]["no_real_llm_calls_by_default"] is True

    def test_gemini_qwen_ollama_optional_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            ae6, ae8, ae9 = self._write_fixtures(tmp_path)
            summary = run_ae10_trading_orchestration(
                project_root=tmp_path,
                max_records=5,
                provider="gemini",
                allow_gemini=False,
                ae6_jsonl=ae6,
                ae8_context_jsonl=ae8,
                ae9_audit_jsonl=ae9,
            )
            assert "CONFIG_MISSING" in summary["llm_provider_summary"]["statuses"]

    def test_traceability_links(self):
        trace = build_traceability_record(
            _sample_decision(), _sample_context(), _sample_audit()
        )
        assert trace.source_decision_id == "dec-001"
        assert trace.source_context_record_id == "ctx-001"
        assert trace.source_llm_audit_record_id == "aud-001"
        assert trace.candidate_id == "cand-001"


class TestGoldenPathFixture:
    def test_price_ok_pending_to_filled_with_ledger(self):
        order_ts = datetime.now(timezone.utc).isoformat()
        snap = _fresh_snapshot(order_ts, price=0.002)
        oracle = DemoPriceOracle(max_price_age_seconds=30.0, snapshots=[snap])
        simulator = PaperOrderSimulator(price_oracle=oracle)
        ledger = DemoLedger(account=DemoAccount(starting_balance_usd=10_000.0, cash_balance_usd=10_000.0))
        adapter = PaperExecutionAdapter(ledger, simulator)

        trace = build_traceability_record(
            _sample_decision(created_at_utc=order_ts),
            _sample_context(),
            _sample_audit(audit_blockers=["weak_lineage"]),
        ).to_dict()
        price = oracle.lookup_price(coin_id=1, order_created_at_utc=order_ts)
        assert price.price_status == PriceStatus.PRICE_OK.value

        intent = OrderIntent(
            candidate_id="cand-001",
            symbol="PEPE/WETH",
            pair_address="0xABC",
            coin_id=1,
            notional_usd=100.0,
            decision_created_at_utc=order_ts,
            order_created_at_utc=order_ts,
        )
        cash_before = ledger.account.cash_balance_usd
        result = adapter.execute(
            intent,
            traceability=trace,
            price_result=price.to_dict(),
            allow_audit_blockers=True,
        )
        assert result.success is True
        assert result.record["status"] == PaperOrderStatus.PAPER_FILLED.value
        assert ledger.account.cash_balance_usd == cash_before - 100.0
        assert ledger.account.open_position_count == 1
        assert len(ledger.positions) == 1
        assert result.record["execution_latency_ms"] is not None
        assert simulator.state_machine.audit_log[-1]["transition_allowed"] is True


class TestLedgerAtomicity:
    def test_rejected_order_does_not_change_cash(self):
        ledger = DemoLedger(account=DemoAccount(cash_balance_usd=10_000.0))
        order = PaperOrder(candidate_id="", status=PaperOrderStatus.PAPER_REJECTED.value)
        ledger.finalize_rejected(order)
        assert ledger.account.cash_balance_usd == 10_000.0

    def test_invalid_transition_does_not_change_cash(self):
        ledger = DemoLedger(account=DemoAccount(cash_balance_usd=10_000.0))
        order = PaperOrder(candidate_id="cand-001", notional_usd=100.0)
        ledger.register_order_intent(order)
        order.status = PaperOrderStatus.PAPER_REJECTED.value
        ledger.finalize_rejected(order)
        assert ledger.account.cash_balance_usd == 10_000.0

    def test_failed_apply_fill_does_not_change_cash(self):
        ledger = DemoLedger(account=DemoAccount(cash_balance_usd=10_000.0))
        order = PaperOrder(
            candidate_id="cand-001",
            status=PaperOrderStatus.PAPER_PENDING.value,
            notional_usd=100.0,
        )
        ledger.register_order_intent(order)
        order.status = PaperOrderStatus.PAPER_FILLED.value
        position = PaperPosition(paper_order_id=order.paper_order_id, candidate_id="cand-001")
        assert ledger.apply_fill(order, position) is True
        assert ledger.account.cash_balance_usd == 9_900.0
        assert ledger.apply_fill(order, position) is False
        assert ledger.account.cash_balance_usd == 9_900.0

    def test_apply_fill_requires_registration(self):
        ledger = DemoLedger(account=DemoAccount(cash_balance_usd=10_000.0))
        order = PaperOrder(
            candidate_id="cand-001",
            status=PaperOrderStatus.PAPER_FILLED.value,
            notional_usd=100.0,
        )
        position = PaperPosition(paper_order_id=order.paper_order_id, candidate_id="cand-001")
        assert ledger.apply_fill(order, position) is False
        assert ledger.account.cash_balance_usd == 10_000.0


class TestPriceOracleSafety:
    def test_provider_time_skew_rejected(self):
        oracle = DemoPriceOracle(max_price_age_seconds=30.0)
        now = datetime.now(timezone.utc)
        future = (now + timedelta(seconds=60)).isoformat()
        order_ts = now.isoformat()
        oracle.snapshots = [{"id": 1, "coin_id": 1, "timestamp": future, "price": 0.001}]
        result = oracle.lookup_price(coin_id=1, order_created_at_utc=order_ts)
        assert result.price_status == PriceStatus.PRICE_PROVIDER_TIME_SKEW_REJECTED.value

    def test_age_against_order_created_at_not_snapshot_only(self):
        oracle = DemoPriceOracle(max_price_age_seconds=30.0)
        order_ts = "2026-07-10T09:00:52+00:00"
        oracle.snapshots = [_fresh_snapshot(order_ts, price=0.001)]
        result = oracle.lookup_price(coin_id=1, order_created_at_utc=order_ts)
        assert result.price_status == PriceStatus.PRICE_OK.value
        assert result.price_age_seconds is not None
        assert result.price_age_seconds <= 30.0
        assert result.system_now_utc
        assert result.order_created_at_utc == order_ts
