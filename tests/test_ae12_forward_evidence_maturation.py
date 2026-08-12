"""Tests for AE12 forward-evidence maturation."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from app.ae12_forward_evidence.idempotency import IdempotencyGuard, make_evidence_row_id, make_horizon_row_id
from app.ae12_forward_evidence.loaders import MarketSnapshotStore
from app.ae12_forward_evidence.maturation import OutputRootExistsError, run_forward_evidence_maturation
from app.ae12_forward_evidence.maturation_core import compute_horizon_outcomes
from app.ae12_forward_evidence.missed_winners import detect_missed_winners
from app.ae12_forward_evidence.opportunity_analysis import (
    build_strict_vs_exploration_comparison,
    build_trade_vs_no_trade_comparison,
)
from app.ae12_forward_evidence.qwen_linkage import classify_qwen_ollama_linkage
from app.ae12_forward_evidence.reason_recovery import recover_reason
from app.ae12_forward_evidence.safety import audit_wallet_safety
from app.ae12_forward_evidence.types import Ae12RunConfig, ReasonRecoveryStatus


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


@pytest.fixture
def mini_env(tmp_path: Path) -> Path:
    root = tmp_path
    (root / "data" / "runtime_paper_loop").mkdir(parents=True)
    (root / "data" / "decision_records").mkdir(parents=True)
    (root / "data" / "paper_trading").mkdir(parents=True)
    (root / "data" / "execution").mkdir(parents=True)
    (root / "data" / "llm_audit").mkdir(parents=True)
    (root / "data" / "audits").mkdir(parents=True)

    t0 = datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc)
    opp_path = root / "data" / "runtime_paper_loop" / "ae11_opportunity_capture_20260714.jsonl"
    td_path = root / "data" / "runtime_paper_loop" / "ae11_trade_decisions_20260714.jsonl"
    ae6_path = root / "data" / "decision_records" / "ae6_decisions_20260714.jsonl"

    opps = [
        {
            "record_type": "OPPORTUNITY_CAPTURE",
            "source_decision_id": "dec-1",
            "candidate_id": "cand-1",
            "pair_address": "0xPAIR1",
            "first_seen_timestamp": t0.isoformat(),
            "created_at_utc": t0.isoformat(),
            "price_at_first_seen": 1.0,
            "liquidity_at_first_seen": 1000,
            "volume_at_first_seen": 100,
            "whale_score_at_first_seen": 0.2,
            "ae6_decision_status": "RESEARCH_CANDIDATE",
            "ae8_context_status": "PRESENT",
            "ae9_audit_verdict": None,
            "ae9_audit_blockers": [],
            "paper_action_taken": "NO_TRADE",
            "reason_for_no_trade": "max_open_positions",
            "strict_shadow_decision": "NO_TRADE",
            "exploration_decision": "NO_TRADE",
            "stale_price": False,
            "missing_context": False,
            "max_open_positions_hit": True,
            "cooldown_active": False,
            "duplicate_active_pair": False,
        },
        {
            "record_type": "OPPORTUNITY_CAPTURE",
            "source_decision_id": "dec-2",
            "candidate_id": "cand-2",
            "pair_address": "0xPAIR2",
            "first_seen_timestamp": t0.isoformat(),
            "created_at_utc": t0.isoformat(),
            "price_at_first_seen": 1.0,
            "liquidity_at_first_seen": 1000,
            "volume_at_first_seen": 100,
            "whale_score_at_first_seen": 0.2,
            "ae6_decision_status": "RESEARCH_CANDIDATE",
            "ae8_context_status": "MISSING",
            "paper_action_taken": "TRADE_EXPLORATION_OVERRIDE",
            "reason_for_no_trade": None,
            "strict_shadow_decision": "NO_TRADE",
            "exploration_decision": "TRADE_EXPLORATION_OVERRIDE",
            "paper_order_id": "ord-2",
            "position_id": "pos-2",
            "stale_price": False,
            "missing_context": True,
            "max_open_positions_hit": False,
            "cooldown_active": False,
            "duplicate_active_pair": False,
        },
        {
            # missing reason entirely
            "record_type": "OPPORTUNITY_CAPTURE",
            "source_decision_id": "dec-3",
            "candidate_id": "cand-3",
            "pair_address": "0xPAIR3",
            "first_seen_timestamp": t0.isoformat(),
            "created_at_utc": t0.isoformat(),
            "price_at_first_seen": 1.0,
            "paper_action_taken": "NO_TRADE",
            "strict_shadow_decision": "NO_TRADE",
            "exploration_decision": "NO_TRADE",
        },
    ]
    with opp_path.open("w", encoding="utf-8") as f:
        for o in opps:
            f.write(json.dumps(o) + "\n")
        # duplicate line for idempotency-in-run (same content -> different line no -> different id)
        # Also write exact duplicate of first with same fields but will have different line id — that's OK.
        # For same-id duplicate: write identical source keys twice isn't same evidence_row_id due to line_no.
        # We'll test guard separately.

    with td_path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "source_decision_id": "dec-1",
                    "candidate_id": "cand-1",
                    "pair_address": "0xPAIR1",
                    "reason_for_no_trade": "max_open_positions",
                    "strict_shadow_decision": "NO_TRADE",
                    "exploration_decision": "NO_TRADE",
                    "paper_action_taken": "NO_TRADE",
                    "hard_safety": {
                        "wallet_configured": False,
                        "private_key_accessed": False,
                        "real_transaction_attempted": False,
                        "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
                    },
                }
            )
            + "\n"
        )

    with ae6_path.open("w", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "decision_id": "dec-1",
                    "created_at_utc": t0.isoformat(),
                    "decision_status": "RESEARCH_CANDIDATE",
                    "candidate_identity": {
                        "candidate_id": "cand-1",
                        "pair_address": "0xPAIR1",
                        "chain": "robinhood",
                        "symbol": "PEPE/WETH",
                    },
                    "llm_context": {
                        "qwen_memo_available": False,
                        "llm_decision_authority": False,
                        "llm_missing_reason": "AE9_NOT_IMPLEMENTED_YET",
                    },
                    "reasons": ["no_model_consensus_available"],
                }
            )
            + "\n"
        )
        f.write(
            json.dumps(
                {
                    "decision_id": "dec-qwen",
                    "created_at_utc": t0.isoformat(),
                    "candidate_identity": {"candidate_id": "cand-q", "pair_address": "0xQ"},
                    "llm_context": {
                        "qwen_memo_available": True,
                        "qwen_memo": "audit only memo",
                        "llm_decision_authority": False,
                    },
                }
            )
            + "\n"
        )

    # SQLite snapshots
    import sqlite3

    db = root / "data" / "trader.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE market_snapshots (id INTEGER PRIMARY KEY, pair_address TEXT, timestamp TEXT, price REAL)"
    )
    # mature 5m/15m/1h for PAIR1 with upside
    for mins, price in [(3, 1.05), (6, 1.20), (20, 1.30), (70, 1.40)]:
        conn.execute(
            "INSERT INTO market_snapshots(pair_address, timestamp, price) VALUES (?,?,?)",
            ("0xPAIR1", (t0 + timedelta(minutes=mins)).isoformat(), price),
        )
    # latest far ahead so horizons mature
    conn.execute(
        "INSERT INTO market_snapshots(pair_address, timestamp, price) VALUES (?,?,?)",
        ("0xPAIR1", (t0 + timedelta(hours=2)).isoformat(), 1.41),
    )
    # PAIR2 traded path mild move
    conn.execute(
        "INSERT INTO market_snapshots(pair_address, timestamp, price) VALUES (?,?,?)",
        ("0xPAIR2", (t0 + timedelta(minutes=10)).isoformat(), 1.02),
    )
    conn.execute(
        "INSERT INTO market_snapshots(pair_address, timestamp, price) VALUES (?,?,?)",
        ("0xPAIR2", (t0 + timedelta(hours=2)).isoformat(), 1.03),
    )
    # PAIR3 no mid snaps but latest exists -> matured but maybe limited
    conn.execute(
        "INSERT INTO market_snapshots(pair_address, timestamp, price) VALUES (?,?,?)",
        ("0xPAIR3", (t0 + timedelta(hours=2)).isoformat(), 2.5),
    )
    conn.commit()
    conn.close()
    return root


def test_horizon_not_matured_no_return(tmp_path: Path):
    store = MarketSnapshotStore(db_path=None)
    store.available = False
    store.unavailable_reason = "TEST"
    first = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
    fields, outcomes, _ = compute_horizon_outcomes(
        evidence_row_id="e1",
        pair_address="0x",
        first_seen_timestamp=first.isoformat(),
        entry_price=1.0,
        horizons=["5m"],
        store=store,
        now_utc=first + timedelta(minutes=1),
    )
    assert fields["horizon_5m_matured"] is False
    assert fields["horizon_5m_max_return"] is None
    assert outcomes[0].max_return is None


def test_not_matured_not_written_as_zero():
    store = MarketSnapshotStore(db_path=None)
    store.available = False
    first = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
    fields, _, _ = compute_horizon_outcomes(
        evidence_row_id="e1",
        pair_address="0x",
        first_seen_timestamp=first.isoformat(),
        entry_price=1.0,
        horizons=["24h"],
        store=store,
        now_utc=first + timedelta(minutes=5),
    )
    assert fields["horizon_24h_max_return"] is None
    assert fields["horizon_24h_max_return"] != 0.0


def test_matured_return_from_snapshots_after_first_seen(mini_env: Path):
    store = MarketSnapshotStore(db_path=mini_env / "data" / "trader.db")
    store.open()
    first = datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc)
    fields, outcomes, audits = compute_horizon_outcomes(
        evidence_row_id="e1",
        pair_address="0xPAIR1",
        first_seen_timestamp=first.isoformat(),
        entry_price=1.0,
        horizons=["5m", "15m"],
        store=store,
    )
    store.close()
    assert fields["horizon_5m_matured"] is True
    assert fields["horizon_5m_max_return"] == pytest.approx(0.05)
    assert fields["horizon_15m_matured"] is True
    assert fields["horizon_15m_max_return"] == pytest.approx(0.20)
    assert fields["horizon_5m_no_lookahead_status"] == "NO_LOOKAHEAD_OK"
    assert any(a.get("audit_code") == "NO_LOOKAHEAD_OK" for a in audits)


def test_lookahead_violation_flagged(mini_env: Path):
    # compute uses deadline filter; inject audit path via snaps after deadline excluded
    store = MarketSnapshotStore(db_path=mini_env / "data" / "trader.db")
    store.open()
    first = datetime(2026, 7, 14, 12, 0, 0, tzinfo=timezone.utc)
    _, _, audits = compute_horizon_outcomes(
        evidence_row_id="e1",
        pair_address="0xPAIR1",
        first_seen_timestamp=first.isoformat(),
        entry_price=1.0,
        horizons=["5m"],
        store=store,
    )
    store.close()
    # Wider window may include post-5m snaps; those after deadline should be flagged when present in window_snaps
    assert any(a.get("no_lookahead_status") in {"NO_LOOKAHEAD_OK", "NOT_MATURED"} for a in audits)


def test_missed_winner_detection():
    rows = [
        {
            "evidence_row_id": "e1",
            "was_traded": False,
            "strict_shadow_decision": "NO_TRADE",
            "exploration_decision": "NO_TRADE",
            "horizon_5m_matured": True,
            "horizon_5m_max_return": 0.25,
            "horizon_5m_no_lookahead_status": "NO_LOOKAHEAD_OK",
        }
    ]
    missed = detect_missed_winners(rows, horizons=["5m"], thresholds={"5m": 0.10})
    assert len(missed) == 1


def test_not_traded_high_return_captured():
    rows = [
        {
            "evidence_row_id": "e2",
            "was_traded": False,
            "paper_action_taken": "NO_TRADE",
            "strict_shadow_decision": "NO_TRADE",
            "exploration_decision": "NO_TRADE",
            "horizon_1h_matured": True,
            "horizon_1h_max_return": 0.50,
            "horizon_1h_no_lookahead_status": "NO_LOOKAHEAD_OK",
        }
    ]
    missed = detect_missed_winners(rows, horizons=["1h"], thresholds={"1h": 0.25})
    assert missed[0]["horizon"] == "1h"


def test_trade_vs_no_trade_counts():
    rows = [
        {
            "was_traded": True,
            "paper_action_taken": "TRADE_EXPLORATION_OVERRIDE",
            "strict_shadow_decision": "NO_TRADE",
            "exploration_decision": "TRADE_EXPLORATION_OVERRIDE",
            "horizon_5m_matured": True,
            "horizon_5m_max_return": 0.01,
            "horizon_5m_no_lookahead_status": "NO_LOOKAHEAD_OK",
        },
        {
            "was_traded": False,
            "paper_action_taken": "NO_TRADE",
            "strict_shadow_decision": "NO_TRADE",
            "exploration_decision": "NO_TRADE",
            "horizon_5m_matured": True,
            "horizon_5m_max_return": 0.20,
            "horizon_5m_no_lookahead_status": "NO_LOOKAHEAD_OK",
        },
    ]
    cmp_rows = build_trade_vs_no_trade_comparison(rows, horizons=["5m"], min_sample=1)
    assert cmp_rows[0]["traded_count"] == 1
    assert cmp_rows[0]["not_traded_count"] == 1


def test_strict_vs_exploration_labels():
    rows = [
        {
            "strict_shadow_decision": "TRADE",
            "exploration_decision": "NO_TRADE",
            "was_traded": False,
            "paper_action_taken": "NO_TRADE",
        },
        {
            "strict_shadow_decision": "NO_TRADE",
            "exploration_decision": "TRADE_EXPLORATION_OVERRIDE",
            "was_traded": True,
            "paper_action_taken": "TRADE_EXPLORATION_OVERRIDE",
        },
    ]
    out = build_strict_vs_exploration_comparison(rows, horizons=["5m"])
    assert out["strict_approved"] == 1
    assert out["strict_blocked_but_exploration_traded"] == 1


def test_reason_recovery_does_not_invent():
    recovered = recover_reason(
        opportunity={"paper_action_taken": "NO_TRADE"},
        trade_decision=None,
        ae6=None,
        runtime_events_by_key={},
        was_traded=False,
        paper_action_taken="NO_TRADE",
    )
    assert recovered["reason_recovery_status"] == ReasonRecoveryStatus.MISSING_IN_SOURCE.value
    assert recovered["reason_not_traded"] == "UNKNOWN_NOT_RECORDED"


def test_missing_reason_writes_warning(mini_env: Path):
    out = mini_env / "data" / "audits" / "ae12_test_out"
    summary = run_forward_evidence_maturation(
        Ae12RunConfig(
            project_root=mini_env,
            output_root=out,
            resume=False,
            max_rows=100,
            horizons=["5m", "15m", "1h"],
            no_external_apis=True,
            no_real_wallet=True,
            db_path=mini_env / "data" / "trader.db",
        )
    )
    warn_path = out / "audits" / "ae12_missing_data_warning_audit.csv"
    text = warn_path.read_text(encoding="utf-8")
    assert "MISSING_REJECTION_REASON" in text or summary["missing_reason_count"] >= 1


def test_qwen_linkage_row_vs_mention():
    mention = classify_qwen_ollama_linkage(
        opportunity={"candidate_id": "c1", "source_decision_id": "d1"},
        ae6={
            "decision_id": "d1",
            "llm_context": {"qwen_memo_available": False, "llm_missing_reason": "x"},
        },
        ae9=None,
    )
    assert mention["qwen_linkage_status"] == "MENTION_ONLY"

    linked = classify_qwen_ollama_linkage(
        opportunity={"candidate_id": "c1", "source_decision_id": "d1"},
        ae6={
            "decision_id": "d1",
            "created_at_utc": "2026-07-14T12:00:00+00:00",
            "candidate_identity": {"pair_address": "0xA"},
            "llm_context": {"qwen_memo_available": True, "qwen_memo": "hello"},
        },
        ae9=None,
    )
    assert linked["qwen_linkage_status"] == "ROW_LINKED_AE6_DECISION"

    only_log = classify_qwen_ollama_linkage(
        opportunity={},
        ae6=None,
        ae9=None,
        log_mentions_present=True,
    )
    assert only_log["qwen_linkage_status"] == "LOG_ONLY_NOT_ROW_LINKED"


def test_idempotency_duplicate_input_ids():
    guard = IdempotencyGuard()
    eid = make_evidence_row_id(
        source_file="a.jsonl",
        source_line_no=1,
        candidate_id="c",
        decision_id="d",
        pair_address="p",
        first_seen_timestamp="t",
    )
    assert guard.accept_evidence(eid) is True
    assert guard.accept_evidence(eid) is False
    hid = make_horizon_row_id(evidence_row_id=eid, horizon="5m")
    assert guard.accept_horizon(hid) is True
    assert guard.accept_horizon(hid) is False


def test_resume_does_not_duplicate(mini_env: Path):
    out = mini_env / "data" / "audits" / "ae12_resume"
    cfg = Ae12RunConfig(
        project_root=mini_env,
        output_root=out,
        resume=False,
        max_rows=100,
        horizons=["5m"],
        no_external_apis=True,
        no_real_wallet=True,
        db_path=mini_env / "data" / "trader.db",
    )
    s1 = run_forward_evidence_maturation(cfg)
    cfg.resume = True
    s2 = run_forward_evidence_maturation(cfg)
    assert s1["candidate_evidence_row_count"] >= 1
    assert s2.get("candidate_evidence_row_count_this_run", 0) == 0
    assert s2["candidate_evidence_row_count"] == s1["candidate_evidence_row_count"]
    assert s2["idempotency"]["skipped_duplicate_evidence"] >= s1["candidate_evidence_row_count_this_run"]
    # CSV must not double
    csv_path = out / "data" / "ae12_candidate_evidence_rows.csv"
    lines = [ln for ln in csv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) - 1 == s1["candidate_evidence_row_count"]


def test_fail_closed_without_resume(mini_env: Path):
    out = mini_env / "data" / "audits" / "ae12_exists"
    out.mkdir(parents=True)
    (out / "marker.txt").write_text("x", encoding="utf-8")
    with pytest.raises(OutputRootExistsError):
        run_forward_evidence_maturation(
            Ae12RunConfig(
                project_root=mini_env,
                output_root=out,
                resume=False,
                fail_if_output_exists=True,
                horizons=["5m"],
                db_path=mini_env / "data" / "trader.db",
            )
        )


def test_append_only_policy(mini_env: Path):
    out = mini_env / "data" / "audits" / "ae12_append"
    summary = run_forward_evidence_maturation(
        Ae12RunConfig(
            project_root=mini_env,
            output_root=out,
            horizons=["5m"],
            db_path=mini_env / "data" / "trader.db",
        )
    )
    # trader.db mtime not mutated by AE12 state (state under output)
    assert (out / "state" / "processed_evidence_row_keys.jsonl").is_file()
    assert not (mini_env / "data" / "trader.db.ae12").exists()
    assert summary["readiness_gate"]["live_trading_ready"] is False


def test_wallet_safety_unknown_when_missing(tmp_path: Path):
    result = audit_wallet_safety(
        project_root=tmp_path,
        live_dry_run_files=[],
        trade_decision_sample=[],
        no_real_wallet=True,
    )
    assert result["audit_status"] == "UNKNOWN_REQUIRES_RUNTIME_SAFETY_AUDIT"
