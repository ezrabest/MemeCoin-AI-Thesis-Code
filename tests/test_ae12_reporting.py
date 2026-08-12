"""Tests for AE12.5 reporting / observability layer."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import pytest

from app.ae12_reporting.doc_templates import DOC_RENDERERS
from app.ae12_reporting.final_docs import build_doc_context, render_all_docs, write_final_docs
from app.ae12_reporting.latest import (
    discover_latest_maturation_root,
    discover_all_latest_roots,
)
from app.ae12_reporting.loaders import load_json_file, load_maturation_summary
from app.ae12_reporting.report_manager import (
    AE12ReportManager,
    reset_ae12_report_manager,
)
from app.ae12_reporting.schemas import REQUIRED_REPORT_LIMITATION_PHRASES


def _write_mini_ae12_root(root: Path) -> Path:
    """Create a minimal AE12 maturation-like tree under tmp."""
    mat = root / "data" / "audits" / "ae12_forward_evidence_maturation_20990101_120000"
    (mat / "reports").mkdir(parents=True)
    (mat / "data").mkdir(parents=True)
    (mat / "audits").mkdir(parents=True)

    summary = {
        "phase": "AE12_FORWARD_EVIDENCE_MATURATION",
        "schema_version": "AE12_V1",
        "created_at_utc": "2099-01-01T12:00:00+00:00",
        "output_root": str(mat),
        "run_id": "ae12_test",
        "candidate_evidence_row_count": 42,
        "matured_outcome_row_count": 100,
        "missed_winner_count": 3,
        "missed_winners_by_horizon": {"5m": 1, "15m": 1, "1h": 1},
        "missing_data_warning_count": 2,
        "reason_recovery_counts": {"RECOVERED_FROM_OPPORTUNITY": 40, "RECOVERED_FROM_TRADE_DECISION": 2},
        "horizon_maturity": {
            "5m": {
                "matured_count": 40,
                "not_matured_count": 2,
                "no_lookahead_ok_count": 40,
                "matured_but_no_snapshots_count": 0,
            }
        },
        "trade_vs_no_trade": [
            {
                "horizon": "5m",
                "traded_count": 2,
                "not_traded_count": 40,
                "median_forward_return_traded": 0.01,
                "median_forward_return_not_traded": 0.0,
                "max_forward_return_traded": 0.2,
                "max_forward_return_not_traded": 0.5,
                "interpretation_status": "MIXED",
            }
        ],
        "trade_vs_no_trade_interpretations": {"5m": "MIXED"},
        "strict_vs_exploration": {
            "total_candidates": 42,
            "strict_approved": 0,
            "strict_blocked": 42,
            "exploration_only_trades": 2,
            "strict_approved_trades": 0,
            "top_strict_blockers": [{"reason": "ACTIVE_PAIR_LOCK", "count": 20}],
            "return_comparison_by_horizon": [],
        },
        "qwen_linkage_counts": {"ROW_LINKED_AE9_RECORD": 30, "MENTION_ONLY": 12},
        "qwen_linkage_sanity_sample": [
            {
                "candidate_id": "c1",
                "qwen_linkage_status": "ROW_LINKED_AE9_RECORD",
                "ollama_linkage_status": "ABSENT",
                "llm_trade_authority_status": "NO_TRADE_AUTHORITY",
            }
        ],
        "wallet_safety": {
            "audit_status": "PASS",
            "wallet_configured": False,
            "private_key_accessed": False,
            "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
            "no_real_wallet_flag": True,
        },
        "readiness_gate": {
            "gate_name": "ae12_final_system_readiness_gate",
            "status": "FORWARD_EVIDENCE_READY_FOR_REPORTING",
            "live_trading_ready": False,
            "profitability_proven": False,
            "qwen_trade_authority": False,
            "wallet_safety_status": "PASS",
            "evidence_row_count": 42,
            "can_proceed_to_ui_final_report": True,
            "needs_persistence_fix": False,
            "notes": ["Safe for Final MSc reporting as forward-evidence audit, not live readiness."],
        },
        "known_limitations": [
            "Forward returns are labels only; not profitability proof.",
        ],
        "source_file_counts": {"opportunity_capture": 1, "paper_trades": 1, "paper_positions": 1},
    }
    gate = dict(summary["readiness_gate"])
    wallet = dict(summary["wallet_safety"])

    (mat / "reports" / "ae12_forward_evidence_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    (mat / "reports" / "ae12_final_system_readiness_gate.json").write_text(
        json.dumps(gate), encoding="utf-8"
    )
    (mat / "audits" / "ae12_wallet_safety_audit.json").write_text(
        json.dumps(wallet), encoding="utf-8"
    )

    with (mat / "data" / "ae12_trade_vs_no_trade_comparison.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "horizon",
                "traded_count",
                "not_traded_count",
                "median_forward_return_traded",
                "median_forward_return_not_traded",
                "max_forward_return_traded",
                "max_forward_return_not_traded",
                "missed_winner_count",
                "sample_size_matured",
                "interpretation_status",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "horizon": "5m",
                "traded_count": 2,
                "not_traded_count": 40,
                "median_forward_return_traded": 0.01,
                "median_forward_return_not_traded": 0.0,
                "max_forward_return_traded": 0.2,
                "max_forward_return_not_traded": 0.5,
                "missed_winner_count": 1,
                "sample_size_matured": 40,
                "interpretation_status": "MIXED",
            }
        )

    with (mat / "data" / "ae12_strict_vs_exploration_comparison.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "total_candidates",
                "strict_approved",
                "strict_blocked",
                "exploration_traded",
                "exploration_only_trades",
                "strict_approved_trades",
                "strict_blocked_but_exploration_traded",
                "horizon",
                "strict_approved_median_return",
                "exploration_only_median_return",
                "strict_approved_n",
                "exploration_only_n",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "total_candidates": 42,
                "strict_approved": 0,
                "strict_blocked": 42,
                "exploration_traded": 2,
                "exploration_only_trades": 2,
                "strict_approved_trades": 0,
                "strict_blocked_but_exploration_traded": 2,
                "horizon": "",
                "strict_approved_median_return": "",
                "exploration_only_median_return": "",
                "strict_approved_n": "",
                "exploration_only_n": "",
            }
        )

    with (mat / "data" / "ae12_missed_winners_full.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "evidence_row_id",
                "candidate_id",
                "decision_id",
                "pair_address",
                "first_seen_timestamp",
                "horizon",
                "max_return",
                "threshold",
                "was_traded",
                "strict_shadow_decision",
                "exploration_decision",
                "reason_not_traded",
                "reason_recovery_status",
                "no_lookahead_status",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "evidence_row_id": "e1",
                "candidate_id": "c1",
                "decision_id": "d1",
                "pair_address": "0xPAIR",
                "first_seen_timestamp": "2099-01-01T00:00:00+00:00",
                "horizon": "5m",
                "max_return": "0.2",
                "threshold": "0.1",
                "was_traded": "False",
                "strict_shadow_decision": "NO_TRADE",
                "exploration_decision": "NO_TRADE",
                "reason_not_traded": "ACTIVE_PAIR_LOCK",
                "reason_recovery_status": "RECOVERED_FROM_OPPORTUNITY",
                "no_lookahead_status": "NO_LOOKAHEAD_OK",
            }
        )

    with (mat / "data" / "ae12_qwen_ollama_linkage.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(
            fh,
            fieldnames=[
                "candidate_id",
                "qwen_linkage_status",
                "ollama_linkage_status",
                "llm_trade_authority_status",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "candidate_id": "c1",
                "qwen_linkage_status": "ROW_LINKED_AE9_RECORD",
                "ollama_linkage_status": "ABSENT",
                "llm_trade_authority_status": "NO_TRADE_AUTHORITY",
            }
        )

    # census root
    census = root / "data" / "audits" / "ae12_runtime_data_census_20990101_110000"
    (census / "reports").mkdir(parents=True)
    (census / "reports" / "ae12_data_census_summary.json").write_text(
        json.dumps(
            {
                "phase": "AE12.1",
                "audit_root": str(census),
                "sqlite_health": {"trader_db_last_write_utc": "2099-01-01T11:30:00+00:00"},
                "top_sqlite_tables_by_rows": [
                    {
                        "table_name": "market_snapshots",
                        "row_count": 1000,
                        "latest_timestamp_value": "2099-01-01T11:29:00+00:00",
                    },
                    {
                        "table_name": "sentiment_records",
                        "row_count": 200,
                        "latest_timestamp_value": "2099-01-01T11:28:00+00:00",
                    },
                ],
                "health_rows": [
                    {
                        "component": "market_snapshots",
                        "rows_or_count": 1000,
                        "latest_timestamp": "2099-01-01T11:29:00+00:00",
                    },
                    {
                        "component": "sentiment_rss",
                        "rows_or_count": 200,
                        "latest_timestamp": "2099-01-01T11:28:00+00:00",
                    },
                ],
                "top_recent_artifacts": [
                    {
                        "relative_path": "data/runtime_paper_loop/state/ae11_latest_checkpoint.json",
                        "last_write_time_utc": "2099-01-01T10:00:00+00:00",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return mat


@pytest.fixture
def mini_ae12(tmp_path: Path) -> Path:
    return _write_mini_ae12_root(tmp_path)


def test_latest_ae12_root_discovery_works(mini_ae12: Path):
    project = mini_ae12.parents[2]  # tmp_path
    # mini root is tmp/data/audits/... so parents[2] is tmp
    # Actually: mat = root/data/audits/... → mat.parents[0]=audits, [1]=data, [2]=root
    root = mini_ae12.parents[2]
    found = discover_latest_maturation_root(root)
    assert found is not None
    assert found == mini_ae12
    all_roots = discover_all_latest_roots(root)
    assert all_roots["maturation_root"] == mini_ae12
    assert all_roots["census_root"] is not None


def test_missing_file_returns_missing_instead_of_crashing(tmp_path: Path):
    missing = tmp_path / "no_such.json"
    result = load_json_file(missing)
    assert result["status"] == "MISSING"
    assert result["missing_file"]
    assert result["data"] is None

    mgr = AE12ReportManager(project_root=tmp_path, ttl_seconds=60)
    fwd = mgr.get_forward_evidence_summary()
    assert fwd["status"] == "MISSING"
    status = mgr.get_status()
    assert status["live_ready"] is False
    assert status["profitability_proven"] is False


def test_report_manager_caches_within_ttl(mini_ae12: Path):
    root = mini_ae12.parents[2]
    mgr = AE12ReportManager(project_root=root, ttl_seconds=300, maturation_root=mini_ae12)
    s1 = mgr.get_forward_evidence_summary()
    counts_after_first = dict(mgr._load_counts)
    s2 = mgr.get_forward_evidence_summary()
    # Summary should not be reloaded within TTL
    assert mgr._load_counts.get("maturation_summary") == counts_after_first.get("maturation_summary")
    assert s1["candidate_evidence_row_count"] == s2["candidate_evidence_row_count"] == 42


def test_report_manager_refresh_reloads_cache(mini_ae12: Path):
    root = mini_ae12.parents[2]
    mgr = AE12ReportManager(project_root=root, ttl_seconds=300, maturation_root=mini_ae12)
    mgr.get_forward_evidence_summary()
    before = mgr._load_counts.get("maturation_summary", 0)
    result = mgr.refresh()
    assert result["refreshed"] is True
    assert result["mutated_source_data"] is False
    after = mgr._load_counts.get("maturation_summary", 0)
    assert after == before + 1


def test_forward_evidence_summary_loads(mini_ae12: Path):
    root = mini_ae12.parents[2]
    mgr = AE12ReportManager(project_root=root, maturation_root=mini_ae12)
    fwd = mgr.get_forward_evidence_summary()
    assert fwd["status"] == "OK"
    assert fwd["candidate_evidence_row_count"] == 42
    assert "5m" in (fwd.get("horizon_maturity") or {})


def test_strict_vs_exploration_summary_loads(mini_ae12: Path):
    root = mini_ae12.parents[2]
    mgr = AE12ReportManager(project_root=root, maturation_root=mini_ae12)
    strict = mgr.get_strict_vs_exploration()
    assert strict["strict_approved"] == 0
    assert strict["strict_blocked"] == 42
    assert strict["exploration_only_trades"] == 2
    assert "zero candidates" in (strict.get("warning") or "").lower() or "zero" in (
        strict.get("explicit_note") or ""
    ).lower()


def test_qwen_linkage_preserves_no_trade_authority(mini_ae12: Path):
    root = mini_ae12.parents[2]
    mgr = AE12ReportManager(project_root=root, maturation_root=mini_ae12)
    qwen = mgr.get_qwen_linkage_summary()
    assert qwen["NO_TRADE_AUTHORITY"] is True
    assert qwen["llm_trade_authority_status"] == "NO_TRADE_AUTHORITY"
    assert qwen["qwen_trade_authority"] is False
    assert qwen["ROW_LINKED_AE9_RECORD"] == 30
    assert qwen["MENTION_ONLY"] == 12


def test_safety_summary_does_not_claim_live_readiness(mini_ae12: Path):
    root = mini_ae12.parents[2]
    mgr = AE12ReportManager(project_root=root, maturation_root=mini_ae12)
    safety = mgr.get_safety_summary()
    assert safety["wallet_configured"] is False
    assert safety["private_key_accessed"] is False
    assert safety["live_submission_status"] == "NOT_SUBMITTED_NO_WALLET"
    assert safety["live_trading_approval"] == "NO"
    assert safety["live_trading_ready"] is False
    assert safety["profitability_proven"] is False
    assert safety["real_wallet_connected"] is False


def test_final_docs_generated_from_source_values_not_hardcoded(mini_ae12: Path, tmp_path: Path):
    root = mini_ae12.parents[2]
    mgr = AE12ReportManager(project_root=root, maturation_root=mini_ae12)
    out = tmp_path / "docs_out"
    manifest = write_final_docs(mgr, out)
    assert manifest["mutated_ae12_source"] is False
    report = (out / "ae12_final_system_report.md").read_text(encoding="utf-8")
    # Injected from mini fixture — if hard-coded real AE12 numbers appeared, this would still pass
    # but must include the fixture values from source JSON:
    assert "42" in report  # candidate_evidence_row_count
    assert "FORWARD_EVIDENCE_READY_FOR_REPORTING" in report
    assert "NOT_SUBMITTED_NO_WALLET" in report
    assert "generated_at" in report
    assert str(mini_ae12) in report or "ae12_forward_evidence_maturation_20990101_120000" in report
    # Must not claim profitability / live readiness
    assert "the system is profitable" not in report.lower()
    assert "the system is live-ready" not in report.lower()


def test_final_report_contains_required_limitation_phrases(mini_ae12: Path):
    root = mini_ae12.parents[2]
    mgr = AE12ReportManager(project_root=root, maturation_root=mini_ae12)
    docs = render_all_docs(mgr)
    blob = "\n".join(docs.values()).lower()
    for phrase in REQUIRED_REPORT_LIMITATION_PHRASES:
        assert phrase.lower() in blob, f"Missing required phrase: {phrase}"
    assert set(DOC_RENDERERS.keys()) == set(docs.keys())


def test_api_endpoints_use_manager_dependency_layer():
    """Endpoints must depend on manager registry, not inline CSV reads."""
    import inspect

    from app import api as api_mod

    src = inspect.getsource(api_mod)
    # Required routes present
    for path in [
        "/api/ae12/status",
        "/api/ae12/forward-evidence-summary",
        "/api/ae12/missed-winners",
        "/api/ae12/trade-vs-no-trade",
        "/api/ae12/strict-vs-exploration",
        "/api/ae12/qwen-linkage",
        "/api/ae12/safety",
        "/api/ae12/final-report-summary",
        "/api/ae12/signal-taxonomy",
        "/api/ae12/sentimentfix",
        "/api/ae12/semantic-coin-classifier",
        "/api/ae12/gemini-semantic-adjudication",
    ]:
        assert path in src
    # No direct CSV open inside the AE12 endpoint section is best-effort;
    # assert Depends + manager methods are used.
    assert "Depends(_get_ae12_manager)" in src
    assert "manager.get_status()" in src
    assert "manager.get_forward_evidence_summary()" in src
    assert "manager.refresh()" in src
    # Endpoints should not call load_csv_rows / open maturation CSVs directly
    ae12_section = src.split("AE12.5")[-1]
    assert "load_csv_rows" not in ae12_section
    assert "ae12_missed_winners_full.csv" not in ae12_section


def test_api_ae12_routes_are_read_only_methods():
    from app.api import app

    ae12_routes = [r for r in app.routes if hasattr(r, "path") and str(r.path).startswith("/api/ae12")]
    assert ae12_routes
    for route in ae12_routes:
        methods = getattr(route, "methods", set()) or set()
        # Only GET and the cache-refresh POST allowed
        assert methods <= {"GET", "POST", "HEAD", "OPTIONS"}
        if "POST" in methods:
            assert route.path == "/api/ae12/refresh-cache"


def test_doc_context_provenance(mini_ae12: Path):
    root = mini_ae12.parents[2]
    mgr = AE12ReportManager(project_root=root, maturation_root=mini_ae12)
    ctx = build_doc_context(mgr)
    assert ctx["source_ae12_output_root"]
    assert ctx["generated_at"]
    assert ctx["source_files_used"]


def test_load_maturation_summary_ok(mini_ae12: Path):
    loaded = load_maturation_summary(mini_ae12)
    assert loaded["status"] == "OK"
    assert loaded["data"]["candidate_evidence_row_count"] == 42


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_ae12_report_manager()
    yield
    reset_ae12_report_manager()
