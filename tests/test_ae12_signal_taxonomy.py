"""Tests for AE12.5 QA polish + signal taxonomy dual-axis audit."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ae12_reporting.ascii_text import contains_mojibake, walk_strings_for_mojibake
from app.ae12_reporting.report_manager import AE12ReportManager, reset_ae12_report_manager
from app.ae12_reporting.summary import (
    build_forward_evidence_payload,
    build_qwen_linkage_payload,
    filter_qwen_sample_rows,
    is_qwen_smoke_test_row,
    normalize_missing_warning_code,
)
from app.ae12_signal_taxonomy.axes import (
    classify_row_axes,
    normalize_semantic_family,
    normalize_trading_state,
    scan_code_for_dangerous_fallbacks,
)
from app.ae12_signal_taxonomy.audit import run_signal_taxonomy_audit


@pytest.fixture(autouse=True)
def _reset_singleton():
    reset_ae12_report_manager()
    yield
    reset_ae12_report_manager()


def test_normalize_missing_warning_code_groups_correctly():
    assert normalize_missing_warning_code({"warning_code": "MISSING_QWEN_ROW_LINKAGE"}) == (
        "MISSING_QWEN_ROW_LINKAGE"
    )
    assert normalize_missing_warning_code({"warning_code": "missing_market_snapshots_for_horizon"}) == (
        "MISSING_MARKET_SNAPSHOTS_FOR_HORIZON"
    )


def test_normalize_missing_warning_code_null_safe():
    assert normalize_missing_warning_code({}) == "UNKNOWN"
    assert normalize_missing_warning_code({"warning_code": None, "missing_field": None}) == "UNKNOWN"
    assert normalize_missing_warning_code({"warning_code": "  "}) == "UNKNOWN"
    assert normalize_missing_warning_code({"missing_field": "qwen_row_linkage"}) == "QWEN_ROW_LINKAGE"
    # Must not KeyError
    assert normalize_missing_warning_code(None) == "UNKNOWN"


def test_missing_warning_breakdown_mixed_rows():
    summary = {
        "candidate_evidence_row_count": 10,
        "horizon_maturity": {},
        "missing_data_warning_count": 3,
        "output_root": "x",
    }
    sample = {
        "rows": [
            {"warning_code": "MISSING_QWEN_ROW_LINKAGE"},
            {"warning_code": "MISSING_QWEN_ROW_LINKAGE"},
            {"warning_code": "MISSING_MARKET_SNAPSHOTS_FOR_HORIZON"},
            {"warning_code": ""},
            {"foo": "bar"},
        ]
    }
    payload = build_forward_evidence_payload(
        summary=summary,
        summary_load={"status": "OK", "path": "p"},
        missing_warnings_sample=sample,
    )
    bd = payload["missing_warning_sample_breakdown"]
    assert bd["MISSING_QWEN_ROW_LINKAGE"] == 2
    assert bd["MISSING_MARKET_SNAPSHOTS_FOR_HORIZON"] == 1
    assert bd["UNKNOWN"] == 2
    hits = walk_strings_for_mojibake(payload)
    assert not any("â" in h for h in hits)


def test_qwen_sample_excludes_smoke_rows():
    rows = [
        {"candidate_id": "cand-001", "decision_id": "dec-001", "pair_address": "0xABC"},
        {
            "candidate_id": "real-cand",
            "decision_id": "real-dec",
            "pair_address": "0x123",
            "ollama_linkage_status": "ABSENT",
            "llm_trade_authority_status": "NO_TRADE_AUTHORITY",
        },
    ]
    assert is_qwen_smoke_test_row(rows[0]) is True
    kept, note = filter_qwen_sample_rows(rows)
    assert len(kept) == 1
    assert kept[0]["candidate_id"] == "real-cand"
    assert note is None

    only_smoke, note2 = filter_qwen_sample_rows([rows[0]])
    assert only_smoke == []
    assert "smoke/test" in (note2 or "").lower()


def test_qwen_payload_preserves_no_trade_authority_and_filters_smoke():
    summary = {
        "output_root": "root",
        "qwen_linkage_counts": {"ROW_LINKED_AE9_RECORD": 10, "MENTION_ONLY": 5},
        "qwen_linkage_sanity_sample": [
            {
                "candidate_id": "cand-001",
                "decision_id": "dec-001",
                "pair_address": "0xABC",
                "llm_trade_authority_status": "NO_TRADE_AUTHORITY",
                "ollama_linkage_status": "ABSENT",
            },
            {
                "candidate_id": "real",
                "decision_id": "d2",
                "pair_address": "0xDEF",
                "llm_trade_authority_status": "NO_TRADE_AUTHORITY",
                "ollama_linkage_status": "ABSENT",
            },
        ],
    }
    payload = build_qwen_linkage_payload(summary=summary, sample_load={"status": "MISSING"})
    assert payload["NO_TRADE_AUTHORITY"] is True
    assert payload["llm_trade_authority_status"] == "NO_TRADE_AUTHORITY"
    assert payload["qwen_trade_authority"] is False
    ids = {r.get("candidate_id") for r in payload["sample_rows"]}
    assert "cand-001" not in ids
    assert "real" in ids


def test_api_report_strings_have_no_mojibake_on_forward_payload():
    payload = build_forward_evidence_payload(
        summary={
            "candidate_evidence_row_count": 1,
            "horizon_maturity": {"5m": {"matured_count": 1, "not_matured_count": 0}},
            "missing_data_warning_count": 0,
            "output_root": "x",
        },
        summary_load={"status": "OK", "path": "p"},
        missing_warnings_sample={"rows": []},
    )
    blob = json.dumps(payload)
    assert "â" not in blob
    assert "\u00e2" not in blob
    assert "\u2260" not in blob  # ≠
    assert "differs from" in payload["price_freshness_vs_horizon_maturity"]["distinction"]


def test_scan_docs_and_reporting_for_mojibake():
    root = Path(__file__).resolve().parents[1]
    hits: list[str] = []
    for path in [
        root / "app" / "ae12_reporting" / "summary.py",
        root / "app" / "ae12_reporting" / "ascii_text.py",
    ]:
        text = path.read_text(encoding="utf-8")
        # ascii_text.py intentionally embeds mojibake/Unicode fragments as detection targets
        if path.name == "ascii_text.py":
            if "\ufffd" in text:
                hits.append(str(path))
            continue
        if contains_mojibake(text) or "\u00e2" in text:
            hits.append(str(path))
        if "\u2260" in text:
            hits.append(f"{path}: contains ≠")
    assert not hits, hits

def test_missing_semantic_does_not_default_to_opportunistic():
    assert normalize_semantic_family(None) == "UNKNOWN"
    assert normalize_semantic_family("") == "UNKNOWN"
    assert normalize_semantic_family("UNKNOWN") == "UNKNOWN"
    axes = classify_row_axes({"pair_address": "0x1"})
    assert axes["semantic_signal_family"] in {"UNKNOWN", "UNCLASSIFIED"}
    assert axes["semantic_signal_family"] != "OPPORTUNISTIC"


def test_explicit_social_and_opportunistic_on_separate_axes():
    row = {
        "cluster_label": "SOCIALLY_MOTIVATED",
        "exploration_decision": "TRADE_EXPLORATION_OVERRIDE",
        "title": "community hype on twitter",
    }
    # Force opportunistic trading via cluster secondary
    row2 = {
        "semantic_signal_family": "SOCIAL",
        "cluster_label": "OPPORTUNISTIC_SPECULATIVE",
        "exploration_decision": "NO_TRADE",
        "paper_action_taken": "NO_TRADE",
    }
    axes = classify_row_axes(row2)
    assert axes["semantic_signal_family"] == "SOCIAL"
    assert axes["trading_opportunity_state"] == "OPPORTUNISTIC"


def test_explicit_opportunistic_is_trading_state_not_semantic():
    axes = classify_row_axes({"cluster_label": "OPPORTUNISTIC_SPECULATIVE"})
    assert axes["trading_opportunity_state"] == "OPPORTUNISTIC"
    assert axes["semantic_signal_family"] in {"UNKNOWN", "UNCLASSIFIED"}


def test_fallback_code_audit_detects_dangerous_default():
    sample = 'cluster = data.get("cluster_label", "OPPORTUNISTIC_SPECULATIVE")\n'
    hits = scan_code_for_dangerous_fallbacks(sample, "sample.py")
    assert hits
    assert hits[0]["severity"] == "HIGH"


def test_signal_taxonomy_endpoint_manager_missing_is_safe(tmp_path: Path):
    mgr = AE12ReportManager(project_root=tmp_path, ttl_seconds=60)
    payload = mgr.get_signal_taxonomy()
    assert payload["status"] == "MISSING"
    assert payload["live_ready"] is False
    assert payload["profitability_proven"] is False


def test_run_taxonomy_audit_mini(tmp_path: Path):
    # Minimal project layout
    (tmp_path / "data" / "decision_records").mkdir(parents=True)
    (tmp_path / "data" / "runtime_paper_loop").mkdir(parents=True)
    (tmp_path / "app" / "analytics").mkdir(parents=True)
    (tmp_path / "static").mkdir(parents=True)
    features = tmp_path / "app" / "analytics" / "features.py"
    features.write_text(
        '''
def resolve_cluster_label():
    """Persistent cluster label — evaluated once per contract_address."""
    existing = get_persisted_cluster(x)
    if existing is not None:
        return existing
    return ClusterLabel.OPPORTUNISTIC_SPECULATIVE
''',
        encoding="utf-8",
    )
    (tmp_path / "static" / "index.html").write_text(
        "function clusterPill(label){ return isSocial ? 'SOCIAL' : 'SPECULATIVE'; }",
        encoding="utf-8",
    )
    ae6 = tmp_path / "data" / "decision_records" / "ae6_decisions_20990101.jsonl"
    ae6.write_text(
        json.dumps(
            {
                "decision_id": "d1",
                "timestamp": "2099-01-01T00:00:00+00:00",
                "cluster_label": "OPPORTUNISTIC_SPECULATIVE",
                "pair_address": "0xPAIR",
            }
        )
        + "\n"
        + json.dumps(
            {
                "decision_id": "d2",
                "timestamp": "2099-01-01T01:00:00+00:00",
                "cluster_label": "SOCIALLY_MOTIVATED",
                "pair_address": "0xPAIR2",
                "title": "community twitter buzz",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    summary = run_signal_taxonomy_audit(
        project_root=tmp_path,
        ae12_root=None,
        max_rows_per_source=100,
        no_external_apis=True,
    )
    assert summary["gate_status"] in {
        "FAIL_CONFLATED_TAXONOMY_AXIS",
        "FAIL_STICKY_OPPORTUNISTIC_FLAG",
        "FAIL_DEFAULT_FALLBACK_BUG",
        "FAIL_UI_MAPPING_BUG",
        "FAIL_SOCIAL_LINKAGE_BUG",
        "NEEDS_MANUAL_REVIEW",
        "PASS_WITH_DATA_LIMITATION",
    }
    out = Path(summary["output_root"])
    assert (out / "reports" / "ae12_signal_taxonomy_audit_summary.json").is_file()
    assert (out / "audits" / "ae12_social_vs_opportunistic_decision_gate.json").is_file()
    gate = json.loads((out / "audits" / "ae12_social_vs_opportunistic_decision_gate.json").read_text(encoding="utf-8"))
    assert gate["live_trading_ready"] is False
    assert gate["profitability_proven"] is False
    assert gate["conflated_axis_found"] is True or gate["default_fallback_bug_found"] is True


def test_safety_labels_unchanged_on_manager(tmp_path: Path):
    mgr = AE12ReportManager(project_root=tmp_path)
    safety = mgr.get_safety_summary()
    assert safety["live_trading_ready"] is False
    assert safety["profitability_proven"] is False
    assert safety["live_trading_approval"] == "NO"
