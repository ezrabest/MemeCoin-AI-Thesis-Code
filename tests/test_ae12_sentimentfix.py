"""Tests for AE12-SentimentFix dual-axis taxonomy repair (not AE12.6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ae12_reporting.report_manager import AE12ReportManager, reset_ae12_report_manager
from app.ae12_sentimentfix.dual_axis_mapper import map_dual_axis
from app.ae12_sentimentfix.run import decide_gate, run_ae12_sentimentfix_audit
from app.ae12_sentimentfix.types import SEMANTIC_UNKNOWN_SHARE_THRESHOLD, null_safe_warning_code
from app.analytics.features import LEGACY_CLUSTER_NOT_SEMANTIC_AUTHORITY


@pytest.fixture(autouse=True)
def _reset():
    reset_ae12_report_manager()
    yield
    reset_ae12_report_manager()


def test_missing_semantic_returns_unknown_not_opportunistic():
    m = map_dual_axis({})
    assert m["semantic_signal_family"] == "UNKNOWN"
    assert "OPPORTUNISTIC" not in m["semantic_signal_family"]


def test_empty_semantic_returns_unknown():
    m = map_dual_axis({"semantic_signal_family": "  ", "cluster_label": ""})
    assert m["semantic_signal_family"] == "UNKNOWN"


def test_warning_code_null_safe():
    assert null_safe_warning_code(None) == "UNKNOWN"
    assert null_safe_warning_code({}) == "UNKNOWN"
    assert null_safe_warning_code({"warning_code": ""}) == "UNKNOWN"
    assert null_safe_warning_code({"warning_code": "MISSING_QWEN_ROW_LINKAGE"}) == "MISSING_QWEN_ROW_LINKAGE"
    assert null_safe_warning_code({"missing_field": "qwen"}) == "QWEN"


def test_opportunistic_maps_to_trading_not_semantic():
    m = map_dual_axis({"cluster_label": "OPPORTUNISTIC_SPECULATIVE"})
    assert m["trading_opportunity_state"] == "OPPORTUNISTIC"
    assert m["semantic_signal_family"] in {"UNKNOWN", "UNCLASSIFIED"}
    assert m["legacy_cluster_label"] == "OPPORTUNISTIC_SPECULATIVE"


def test_legacy_cluster_label_preserved():
    m = map_dual_axis({"cluster_label": "SOCIALLY_MOTIVATED"})
    assert m["legacy_cluster_label"] == "SOCIALLY_MOTIVATED"


def test_social_marker_maps_social():
    m = map_dual_axis({"title": "community twitter buzz hype"})
    assert m["semantic_signal_family"] == "SOCIAL"


def test_news_marker_maps_news():
    m = map_dual_axis({"title": "rss news headline article"})
    assert m["semantic_signal_family"] == "NEWS"


def test_whale_marker_maps_whale_or_onchain():
    m = map_dual_axis({"summary": "whale wallet holder transfer"})
    assert m["semantic_signal_family"] in {"WHALE", "ONCHAIN", "MIXED"}


def test_liquidity_marker():
    m = map_dual_axis({"text": "liquidity volume pool"})
    assert m["semantic_signal_family"] == "LIQUIDITY"


def test_momentum_marker():
    m = map_dual_axis({"text": "price momentum breakout pump"})
    assert m["semantic_signal_family"] == "PRICE_MOMENTUM"


def test_social_and_opportunistic_simultaneous():
    m = map_dual_axis(
        {
            "title": "telegram community buzz",
            "cluster_label": "OPPORTUNISTIC_SPECULATIVE",
        }
    )
    assert m["semantic_signal_family"] == "SOCIAL"
    assert m["trading_opportunity_state"] == "OPPORTUNISTIC"


def test_sticky_legacy_does_not_override_explicit_semantic():
    m = map_dual_axis(
        {
            "semantic_signal_family": "NEWS",
            "cluster_label": "OPPORTUNISTIC_SPECULATIVE",
        }
    )
    assert m["semantic_signal_family"] == "NEWS"
    assert m["trading_opportunity_state"] == "OPPORTUNISTIC"


def test_ui_cluster_pill_does_not_default_missing_to_opportunistic():
    html = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")
    # After AE12.5/SentimentFix, unknown path exists
    assert "UNCLASSIFIED" in html
    assert "UNKNOWN" in html
    # Must not only binary-map non-social -> SPECULATIVE
    assert 'isSocial ? "SOCIAL" : "SPECULATIVE"' not in html


def test_unknown_share_blocks_pass_dual_axis_ready():
    gate = decide_gate(
        prior_gate_status="FAIL_CONFLATED_TAXONOMY_AXIS",
        runtime_info={
            "runtime_future_fields_added": True,
            "sticky_documented_non_authoritative": True,
        },
        semantic_unknown_share=0.99,
        sticky_plan_created=True,
        linkage={"semantic_linkage_gap_found": True, "sentiment_records_count": 100, "sentiment_social_marker_rows": 10},
        default_fallback_still_in_legacy_code=True,
        ui_shows_unknown=True,
    )
    assert gate["status"] != "PASS_DUAL_AXIS_READY"
    assert gate["semantic_unknown_share"] > SEMANTIC_UNKNOWN_SHARE_THRESHOLD
    assert gate["status"] in {
        "PASS_DERIVED_ONLY_RUNTIME_UPDATE_PENDING",
        "HOLD_MANUAL_REVIEW_REQUIRED",
    }


def test_legacy_cluster_not_semantic_authority_flag():
    assert LEGACY_CLUSTER_NOT_SEMANTIC_AUTHORITY is True


def test_opportunity_capture_has_dual_axis_fields():
    from app.runtime_paper_loop.types import OpportunityCaptureRecord

    rec = OpportunityCaptureRecord(loop_run_id="x", loop_iteration=1)
    d = rec.to_dict()
    assert "semantic_signal_family" in d
    assert "trading_opportunity_state" in d
    assert "legacy_cluster_label" in d
    assert d["semantic_signal_family"] == "UNKNOWN"


def test_sentimentfix_manager_missing_safe(tmp_path: Path):
    mgr = AE12ReportManager(project_root=tmp_path)
    payload = mgr.get_sentimentfix()
    assert payload["status"] == "MISSING"
    assert payload["live_ready"] is False
    assert payload["profitability_proven"] is False
    assert payload["qwen_trade_authority"] is False


def test_run_sentimentfix_audit_no_historical_mutation(tmp_path: Path):
    (tmp_path / "data" / "decision_records").mkdir(parents=True)
    (tmp_path / "data" / "runtime_paper_loop").mkdir(parents=True)
    (tmp_path / "app" / "analytics").mkdir(parents=True)
    (tmp_path / "app" / "runtime_paper_loop").mkdir(parents=True)
    (tmp_path / "app" / "observability").mkdir(parents=True)
    (tmp_path / "static").mkdir(parents=True)

    (tmp_path / "app" / "analytics" / "features.py").write_text(
        "LEGACY_CLUSTER_NOT_SEMANTIC_AUTHORITY = True\n# not semantic_signal_family authority\n",
        encoding="utf-8",
    )
    (tmp_path / "app" / "runtime_paper_loop" / "types.py").write_text(
        "semantic_signal_family\ntrading_opportunity_state\nlegacy_cluster_label\n",
        encoding="utf-8",
    )
    (tmp_path / "app" / "runtime_paper_loop" / "opportunity_capture.py").write_text(
        "map_dual_axis\nsemantic_signal_family\n",
        encoding="utf-8",
    )
    (tmp_path / "app" / "observability" / "candidate.py").write_text(
        'semantic_signal_family = "UNKNOWN"\ncluster_label: str = "OPPORTUNISTIC_SPECULATIVE"\n',
        encoding="utf-8",
    )
    (tmp_path / "static" / "index.html").write_text(
        "function clusterPill(){ return 'UNCLASSIFIED'; }\nUNKNOWN\n",
        encoding="utf-8",
    )
    ae6 = tmp_path / "data" / "decision_records" / "ae6_decisions_20990101.jsonl"
    ae6.write_text(
        json.dumps(
            {
                "decision_id": "d1",
                "timestamp": "2099-01-01T00:00:00+00:00",
                "cluster_label": "OPPORTUNISTIC_SPECULATIVE",
                "pair_address": "0x1",
            }
        )
        + "\n"
        + json.dumps(
            {
                "decision_id": "d2",
                "timestamp": "2099-01-01T01:00:00+00:00",
                "title": "community twitter buzz",
                "cluster_label": "OPPORTUNISTIC_SPECULATIVE",
                "pair_address": "0x2",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    # Prior taxonomy audit gate
    tax = tmp_path / "data" / "audits" / "ae12_signal_taxonomy_audit_test"
    (tax / "audits").mkdir(parents=True)
    (tax / "audits" / "ae12_social_vs_opportunistic_decision_gate.json").write_text(
        json.dumps({"status": "FAIL_CONFLATED_TAXONOMY_AXIS"}),
        encoding="utf-8",
    )

    summary = run_ae12_sentimentfix_audit(
        project_root=tmp_path,
        taxonomy_audit_root=tax,
        max_rows_per_source=100,
        no_external_apis=True,
    )
    out = Path(summary["output_root"])
    gate = json.loads((out / "audits" / "ae12_sentimentfix_decision_gate.json").read_text(encoding="utf-8"))
    assert gate["historical_data_mutated"] is False
    assert gate["live_trading_ready"] is False
    assert gate["profitability_proven"] is False
    assert gate["qwen_trade_authority"] is False
    assert gate["legacy_cluster_label_preserved"] is True
    assert gate["dual_axis_mapper_available"] is True
    assert gate["status"] != "PASS_DUAL_AXIS_READY" or summary["semantic_unknown_share"] <= 0.5
    mut = (out / "audits" / "ae12_no_historical_mutation_audit.csv").read_text(encoding="utf-8")
    assert "False" in mut or "false" in mut.lower()
    # AE6 input unchanged
    assert "OPPORTUNISTIC_SPECULATIVE" in ae6.read_text(encoding="utf-8")


def test_phase_name_is_sentimentfix_not_ae126():
    from app.ae12_sentimentfix.types import AE12_SENTIMENTFIX_PHASE

    assert AE12_SENTIMENTFIX_PHASE == "AE12-SentimentFix"
    assert "AE12.6" not in AE12_SENTIMENTFIX_PHASE


def test_sentimentfix_state_excluded_from_latest_root(tmp_path: Path):
    from app.ae12_reporting.latest import discover_latest_sentimentfix_root

    audits = tmp_path / "data" / "audits"
    state = audits / "ae12_sentimentfix_state"
    state.mkdir(parents=True)
    (state / "gemini_semantic_adjudication_cache.jsonl").write_text("{}\n", encoding="utf-8")

    valid = audits / "ae12_sentimentfix_20260715_172645"
    (valid / "reports").mkdir(parents=True)
    (valid / "reports" / "ae12_sentimentfix_summary.json").write_text(
        json.dumps({"gate_status": "PASS_DERIVED_ONLY_RUNTIME_UPDATE_PENDING"}),
        encoding="utf-8",
    )
    # Incomplete timestamp-looking folder without report must be ignored
    partial = audits / "ae12_sentimentfix_20990101_000000"
    partial.mkdir(parents=True)

    root = discover_latest_sentimentfix_root(tmp_path)
    assert root is not None
    assert root.name == "ae12_sentimentfix_20260715_172645"
    assert "state" not in root.name.lower()


def test_sentimentfix_api_marks_dual_axis_not_final(tmp_path: Path):
    audits = tmp_path / "data" / "audits"
    valid = audits / "ae12_sentimentfix_20260715_172645"
    (valid / "reports").mkdir(parents=True)
    (valid / "audits").mkdir(parents=True)
    (valid / "reports" / "ae12_sentimentfix_summary.json").write_text(
        json.dumps(
            {
                "gate_status": "PASS_DERIVED_ONLY_RUNTIME_UPDATE_PENDING",
                "semantic_signal_family_distribution": {"UNKNOWN": 1},
            }
        ),
        encoding="utf-8",
    )
    (valid / "audits" / "ae12_sentimentfix_decision_gate.json").write_text(
        json.dumps({"status": "PASS_DERIVED_ONLY_RUNTIME_UPDATE_PENDING"}),
        encoding="utf-8",
    )
    # Poison: state folder that used to win lexicographic sort
    state = audits / "ae12_sentimentfix_state"
    state.mkdir(parents=True)

    mgr = AE12ReportManager(project_root=tmp_path)
    payload = mgr.get_sentimentfix()
    assert payload["status"] == "OK"
    assert "ae12_sentimentfix_20260715_172645" in str(payload["sentimentfix_root"])
    assert payload["dual_axis_repair"] is True
    assert payload["final_semantic_classification"] is False


def test_taxonomy_and_local_classifier_marked_diagnostic_only(tmp_path: Path):
    mgr = AE12ReportManager(project_root=tmp_path)
    tax = mgr.get_signal_taxonomy()
    assert tax.get("legacy_diagnostic") is True
    assert tax.get("final_semantic_classification") is False
    clf = mgr.get_semantic_coin_classifier()
    assert clf.get("local_classifier") is True
    assert clf.get("final_semantic_classification") is False
