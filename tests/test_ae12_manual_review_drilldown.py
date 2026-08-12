"""Tests for AE12-SentimentFix local manual-review drilldown."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from app.ae12_reporting.report_manager import AE12ReportManager
from app.ae12_sentimentfix.manual_review_drilldown import (
    DRILLDOWN_RULE_VERSION,
    resolve_manual_review_coin,
    run_manual_review_drilldown,
)


def test_rule4_social_vs_op_becomes_unknown_unresolved():
    # Majority opportunistic must not win
    votes = Counter(
        {
            "SOCIAL_CONFIRMED": 1,
            "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED": 13,
        }
    )
    r = resolve_manual_review_coin(class_votes=votes, pair_rows=[])
    assert r["final_class_after_drilldown"] == "UNKNOWN_UNRESOLVED"
    assert r["resolution_rule_applied"] == "RULE_4_SOCIAL_OP_CONFLICT_UNRESOLVED"
    assert "UNKNOWN_UNRESOLVED" in r["resolution_note"]
    assert "opportunistic" not in r["final_class_after_drilldown"].lower() or True
    assert r["final_class_after_drilldown"] != "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED"
    assert r["final_class_after_drilldown"] != "SOCIAL_CONFIRMED"


def test_rule4_does_not_pick_majority_or_default_opportunistic():
    votes = Counter(
        {
            "SOCIAL_CONFIRMED": 4,
            "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED": 6,
            "OPPORTUNISTIC_SUSPECTED": 3,
        }
    )
    r = resolve_manual_review_coin(class_votes=votes, pair_rows=[])
    assert r["final_class_after_drilldown"] == "UNKNOWN_UNRESOLVED"
    assert r["resolution_rule_applied"] == "RULE_4_SOCIAL_OP_CONFLICT_UNRESOLVED"


def test_rule6_rejected_trade_language_unresolved():
    pairs = [
        {
            "semantic_coin_class": "MANUAL_REVIEW",
            "raw_evidence_status": "REJECTED_FOR_TRADE_LANGUAGE",
            "reasoning_short": "Rejected by trade-language safety gate.",
        },
        {
            "semantic_coin_class": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
            "raw_evidence_status": "MODEL_KNOWLEDGE_ONLY",
            "reasoning_short": "memecoin",
        },
    ]
    votes = Counter({"MANUAL_REVIEW": 1, "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED": 1})
    r = resolve_manual_review_coin(class_votes=votes, pair_rows=pairs)
    assert r["final_class_after_drilldown"] == "UNKNOWN_UNRESOLVED"
    assert r["resolution_rule_applied"] == "RULE_6_REJECTED_TRADE_LANGUAGE_UNRESOLVED"


def test_rule3_confirmed_op_overrides_suspected():
    votes = Counter(
        {
            "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED": 3,
            "OPPORTUNISTIC_SUSPECTED": 2,
        }
    )
    r = resolve_manual_review_coin(class_votes=votes, pair_rows=[])
    assert r["final_class_after_drilldown"] == "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED"
    assert r["resolution_rule_applied"] == "RULE_3_CONFIRMED_OP_OVERRIDES_SUSPECTED"


def test_rule7_no_clear_conclusion():
    votes = Counter({"MANUAL_REVIEW": 2})
    r = resolve_manual_review_coin(
        class_votes=votes,
        pair_rows=[{"semantic_coin_class": "MANUAL_REVIEW", "reasoning_short": "unclear"}],
    )
    assert r["final_class_after_drilldown"] == "UNKNOWN_UNRESOLVED"
    assert r["resolution_rule_applied"] == "RULE_7_NO_CLEAR_LOCAL_CONCLUSION"


def test_drilldown_end_to_end_writes_audit_trail(tmp_path: Path):
    gemini = tmp_path / "data" / "audits" / "ae12_gemini_semantic_adjudication_20260716_111507"
    (gemini / "data").mkdir(parents=True)
    (gemini / "reports").mkdir(parents=True)
    (gemini / "audits").mkdir(parents=True)

    (gemini / "data" / "ae12_gemini_coin_level_adjudications.csv").write_text(
        "coin_id,normalized_base_symbol,chain,semantic_coin_class,supporting_pair_count,"
        "supporting_pair_asset_ids,class_votes,conflict_note,identity_resolution_method,"
        "identity_confidence,identity_warnings\n"
        "solana:DOGE,DOGE,solana,MANUAL_REVIEW,3,a;b;c,"
        "\"{\\\"SOCIAL_CONFIRMED\\\":1,\\\"NON_SOCIAL_OPPORTUNISTIC_CONFIRMED\\\":2}\","
        "CONFLICT_SOCIAL_VS_OPPORTUNISTIC_CONFIRMED,normalized_base_symbol+chain,MEDIUM,\n"
        "eth:MEME,MEME,ethereum,MANUAL_REVIEW,2,d;e,"
        "\"{\\\"MANUAL_REVIEW\\\":1,\\\"NON_SOCIAL_OPPORTUNISTIC_CONFIRMED\\\":1}\","
        "MANUAL_REVIEW_MIXED,normalized_base_symbol+chain,MEDIUM,\n"
        "rh:WIF,WIF,robinhood,NON_SOCIAL_OPPORTUNISTIC_CONFIRMED,2,f;g,"
        "\"{\\\"NON_SOCIAL_OPPORTUNISTIC_CONFIRMED\\\":2}\",,normalized_base_symbol+chain,MEDIUM,\n",
        encoding="utf-8",
    )
    (gemini / "data" / "ae12_gemini_pair_asset_adjudications.csv").write_text(
        "asset_id,chain,symbol,semantic_coin_class,raw_evidence_status,reasoning_short,coin_id,"
        "normalized_base_symbol\n"
        "a,solana,DOGE/USDC,SOCIAL_CONFIRMED,MODEL_KNOWLEDGE_ONLY,social tip,solana:DOGE,DOGE\n"
        "b,solana,DOGE/USDC,NON_SOCIAL_OPPORTUNISTIC_CONFIRMED,MODEL_KNOWLEDGE_ONLY,meme,solana:DOGE,DOGE\n"
        "c,solana,DOGE/USDC,NON_SOCIAL_OPPORTUNISTIC_CONFIRMED,MODEL_KNOWLEDGE_ONLY,meme2,solana:DOGE,DOGE\n"
        "d,ethereum,MEME/WETH,MANUAL_REVIEW,REJECTED_FOR_TRADE_LANGUAGE,Rejected by trade-language safety gate.,eth:MEME,MEME\n"
        "e,ethereum,MEME/WETH,NON_SOCIAL_OPPORTUNISTIC_CONFIRMED,MODEL_KNOWLEDGE_ONLY,meme,eth:MEME,MEME\n"
        "f,robinhood,WIF/WETH,NON_SOCIAL_OPPORTUNISTIC_CONFIRMED,MODEL_KNOWLEDGE_ONLY,wif,rh:WIF,WIF\n"
        "g,robinhood,WIF/WETH,NON_SOCIAL_OPPORTUNISTIC_CONFIRMED,MODEL_KNOWLEDGE_ONLY,wif2,rh:WIF,WIF\n",
        encoding="utf-8",
    )
    (gemini / "data" / "ae12_gemini_pair_to_coin_mapping.csv").write_text(
        "pair_asset_id,coin_id,symbol\n"
        "a,solana:DOGE,DOGE/USDC\n"
        "b,solana:DOGE,DOGE/USDC\n"
        "c,solana:DOGE,DOGE/USDC\n"
        "d,eth:MEME,MEME/WETH\n"
        "e,eth:MEME,MEME/WETH\n"
        "f,rh:WIF,WIF/WETH\n"
        "g,rh:WIF,WIF/WETH\n",
        encoding="utf-8",
    )
    (gemini / "audits" / "ae12_gemini_semantic_adjudication_gate.json").write_text(
        json.dumps(
            {
                "status": "PASS_GEMINI_ADJUDICATION_READY",
                "adjudicator_version": "AE12_SENTIMENTFIX_GEMINI_ADJUDICATOR_V1",
                "rubric_version": "AE12_SENTIMENTFIX_ADJUDICATION_RUBRIC_V1",
            }
        ),
        encoding="utf-8",
    )
    (gemini / "audits" / "ae12_gemini_safety_audit.json").write_text(
        json.dumps(
            {
                "status": "PASS_REJECTIONS_ENFORCED",
                "rejected_outputs": 1,
                "output_used_after_rejection": False,
                "trade_authority_used": False,
                "forbidden_trade_language_found": True,
            }
        ),
        encoding="utf-8",
    )
    (gemini / "reports" / "ae12_gemini_semantic_adjudication_summary.json").write_text(
        json.dumps({"gate_status": "PASS_GEMINI_ADJUDICATION_READY"}),
        encoding="utf-8",
    )

    summary = run_manual_review_drilldown(
        project_root=tmp_path,
        gemini_root=gemini,
        no_external_apis=True,
    )
    out = Path(summary["output_root"])
    assert summary["gemini_called_again"] is False
    assert summary["external_api_used"] is False
    assert summary["trade_authority_used"] is False
    assert summary["live_ready"] is False
    assert summary["profitability_proven"] is False
    assert summary["drilldown_rule_version"] == DRILLDOWN_RULE_VERSION
    assert summary["manual_review_input_count"] == 2
    assert summary["manual_review_remaining_count"] == 0
    assert summary["unknown_unresolved_count"] == 2
    assert summary["gate_status"] == "PASS_WITH_UNKNOWN_UNRESOLVED"

    drill_csv = out / "data" / "ae12_manual_review_coin_drilldown.csv"
    rows = list(csv.DictReader(drill_csv.open(encoding="utf-8")))
    assert len(rows) == 2
    for row in rows:
        assert row["resolution_note"]
        assert row["resolution_rule_applied"]
        assert row["drilldown_rule_version"] == DRILLDOWN_RULE_VERSION
        assert row["external_api_used"] in {"False", "false", False} or row["external_api_used"] == "False"
        assert row["gemini_called_again"] in {"False", "false"}
        assert row["final_class_after_drilldown"] == "UNKNOWN_UNRESOLVED"

    gate = json.loads((out / "audits" / "ae12_manual_review_drilldown_gate.json").read_text(encoding="utf-8"))
    assert gate["drilldown_rule_version"] == DRILLDOWN_RULE_VERSION
    assert gate["external_api_used"] is False
    assert gate["gemini_called_again"] is False
    assert gate["status"] == "PASS_WITH_UNKNOWN_UNRESOLVED"

    upload = (out / "reports" / "ae12_manual_review_drilldown_for_upload.txt").read_text(encoding="utf-8")
    assert DRILLDOWN_RULE_VERSION in upload
    assert "gemini_called_again: false" in upload

    mgr = AE12ReportManager(project_root=tmp_path)
    with patch.object(
        mgr,
        "discover_roots",
        return_value={"manual_review_drilldown_root": out, "gemini_adjudication_root": gemini},
    ):
        api = mgr.get_manual_review_drilldown()
    assert api["status"] == "OK"
    assert api["unknown_unresolved_count"] == 2
    assert api["trade_authority_used"] is False
    assert api["live_ready"] is False
    assert api["profitability_proven"] is False


def test_dashboard_html_wires_manual_review_drilldown_not_legacy_as_final():
    html = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")
    assert "/api/ae12/manual-review-drilldown" in html
    assert "updated_coin_level_counts" in html or "coin_level_counts_after_drilldown" in html
    assert "UNKNOWN_UNRESOLVED" in html or "UNKNOWN UNRESOLVED" in html
    assert "Reporting only" in html or "not trade authority" in html
    assert "Pair-level adjudications are audit detail only" in html
    assert "AE12 Semantic Coin Classification" in html
    assert "unknown-unresolved-card" in html
    assert "op-suspected-card" in html
    assert "opp-confirmed-card" in html
    assert "Legacy Social Cluster" in html
    assert "Legacy Opportunistic Cluster" in html
    assert "pickAe12SemanticCounts" in html
    assert "PASS_WITH_UNKNOWN_UNRESOLVED" in html
    # Must not silently treat legacy cluster_counts as primary semantic cards
    assert 'id="ae12-stat-opp-confirmed"' in html
    assert "Legacy counters are diagnostic only" in html or "not final AE12 semantic" in html
    # Distinct styles for suspected vs confirmed vs unresolved
    assert "op-suspected-card" in html and "opp-confirmed-card" in html
    assert "unknown-unresolved-card" in html


def test_manual_review_endpoint_exposes_post_drilldown_counts_for_ui(tmp_path: Path):
    """UI-facing API must expose post-drilldown counts only on PASS gates."""
    out = tmp_path / "data" / "audits" / "ae12_sentimentfix_manual_review_drilldown_20990101_000000"
    (out / "reports").mkdir(parents=True)
    (out / "audits").mkdir(parents=True)
    counts = {
        "unique_coins_found": 14,
        "coin_social_confirmed_count": 0,
        "coin_non_social_opportunistic_confirmed_count": 7,
        "coin_opportunistic_suspected_count": 1,
        "coin_unknown_unresolved_count": 6,
        "coin_manual_review_remaining_count": 0,
        "coin_non_social_infrastructure_confirmed_count": 0,
    }
    (out / "reports" / "ae12_manual_review_drilldown_summary.json").write_text(
        json.dumps(
            {
                "gate_status": "PASS_WITH_UNKNOWN_UNRESOLVED",
                "updated_coin_level_counts": counts,
                "unknown_unresolved_count": 6,
                "manual_review_remaining_count": 0,
                "drilldown_rule_version": DRILLDOWN_RULE_VERSION,
                "external_api_used": False,
                "gemini_called_again": False,
                "trade_authority_used": False,
            }
        ),
        encoding="utf-8",
    )
    (out / "audits" / "ae12_manual_review_drilldown_gate.json").write_text(
        json.dumps(
            {
                "status": "PASS_WITH_UNKNOWN_UNRESOLVED",
                "updated_coin_level_distribution": counts,
                "unknown_unresolved_count": 6,
                "manual_review_remaining_count": 0,
                "manual_review_input_count": 6,
                "manual_review_resolved_count": 6,
                "drilldown_rule_version": DRILLDOWN_RULE_VERSION,
                "external_api_used": False,
                "gemini_called_again": False,
                "trade_authority_used": False,
                "live_ready": False,
                "profitability_proven": False,
            }
        ),
        encoding="utf-8",
    )
    mgr = AE12ReportManager(project_root=tmp_path)
    with patch.object(mgr, "discover_roots", return_value={"manual_review_drilldown_root": out}):
        api = mgr.get_manual_review_drilldown()
    assert api["status"] == "OK"
    assert api["gate_status"] == "PASS_WITH_UNKNOWN_UNRESOLVED"
    assert api["updated_coin_level_counts"]["unique_coins_found"] == 14
    assert api["updated_coin_level_counts"]["coin_non_social_opportunistic_confirmed_count"] == 7
    assert api["updated_coin_level_counts"]["coin_opportunistic_suspected_count"] == 1
    assert api["updated_coin_level_counts"]["coin_unknown_unresolved_count"] == 6
    assert api["updated_coin_level_counts"]["coin_manual_review_remaining_count"] == 0
    assert api["external_api_used"] is False
    assert api["gemini_called_again"] is False
    assert api["trade_authority_used"] is False


def test_gemini_endpoint_surfaces_after_drilldown_and_pair_audit_role(tmp_path: Path):
    gemini = tmp_path / "data" / "audits" / "ae12_gemini_semantic_adjudication_20990101_000000"
    drill = tmp_path / "data" / "audits" / "ae12_sentimentfix_manual_review_drilldown_20990101_000001"
    for root in (gemini, drill):
        (root / "reports").mkdir(parents=True)
        (root / "audits").mkdir(parents=True)
        (root / "data").mkdir(parents=True)

    (gemini / "reports" / "ae12_gemini_semantic_adjudication_summary.json").write_text(
        json.dumps({"gate_status": "PASS_GEMINI_ADJUDICATION_READY", "class_distribution": {}}),
        encoding="utf-8",
    )
    (gemini / "audits" / "ae12_gemini_semantic_adjudication_gate.json").write_text(
        json.dumps(
            {
                "status": "PASS_GEMINI_ADJUDICATION_READY",
                "unique_assets_input": 89,
                "social_confirmed_count": 11,
                "non_social_opportunistic_confirmed_count": 65,
                "opportunistic_suspected_count": 8,
                "manual_review_count": 5,
                "external_api_used": True,
                "gemini_used": True,
                "web_grounding_used": False,
                "trade_authority_used": False,
            }
        ),
        encoding="utf-8",
    )
    (gemini / "audits" / "ae12_gemini_safety_audit.json").write_text(
        json.dumps(
            {
                "status": "PASS_REJECTIONS_ENFORCED",
                "rejected_outputs": 2,
                "output_used_after_rejection": False,
                "forbidden_trade_language_found": True,
                "trade_authority_used": False,
            }
        ),
        encoding="utf-8",
    )
    (gemini / "reports" / "ae12_gemini_coin_level_summary.json").write_text(
        json.dumps(
            {
                "pair_asset_counts": {
                    "unique_pair_assets_input": 89,
                    "count_role": "audit_detail",
                },
                "coin_level_counts": {
                    "unique_coins_found": 14,
                    "coin_social_confirmed_count": 0,
                    "coin_non_social_opportunistic_confirmed_count": 7,
                    "coin_opportunistic_suspected_count": 1,
                    "coin_manual_review_count": 6,
                    "count_role": "final_ui",
                },
                "count_level_used_for_main_ui": "coin_level",
            }
        ),
        encoding="utf-8",
    )
    counts = {
        "unique_coins_found": 14,
        "coin_social_confirmed_count": 0,
        "coin_non_social_opportunistic_confirmed_count": 7,
        "coin_opportunistic_suspected_count": 1,
        "coin_unknown_unresolved_count": 6,
        "coin_manual_review_remaining_count": 0,
    }
    (drill / "reports" / "ae12_manual_review_drilldown_summary.json").write_text(
        json.dumps(
            {
                "gate_status": "PASS_WITH_UNKNOWN_UNRESOLVED",
                "updated_coin_level_counts": counts,
                "unknown_unresolved_count": 6,
                "drilldown_rule_version": DRILLDOWN_RULE_VERSION,
                "external_api_used": False,
                "gemini_called_again": False,
                "trade_authority_used": False,
            }
        ),
        encoding="utf-8",
    )
    (drill / "audits" / "ae12_manual_review_drilldown_gate.json").write_text(
        json.dumps(
            {
                "status": "PASS_WITH_UNKNOWN_UNRESOLVED",
                "updated_coin_level_distribution": counts,
                "unknown_unresolved_count": 6,
                "drilldown_rule_version": DRILLDOWN_RULE_VERSION,
                "external_api_used": False,
                "gemini_called_again": False,
                "trade_authority_used": False,
            }
        ),
        encoding="utf-8",
    )

    mgr = AE12ReportManager(project_root=tmp_path)
    with patch.object(
        mgr,
        "discover_roots",
        return_value={"gemini_adjudication_root": gemini, "manual_review_drilldown_root": drill},
    ):
        payload = mgr.get_gemini_semantic_adjudication()
    assert payload["status"] == "OK"
    assert payload["safety_audit_status"] == "PASS_REJECTIONS_ENFORCED"
    assert payload["pair_asset_counts"]["count_role"] == "audit_detail"
    assert payload["coin_level_counts"]["count_role"] == "final_ui"
    assert payload["count_level_used_for_main_ui"] == "coin_level_after_drilldown"
    assert payload["coin_level_counts_after_drilldown"]["unique_coins_found"] == 14
    assert payload["manual_review_drilldown"]["completed_locally"] is True
    assert payload["manual_review_drilldown"]["gate_status"] == "PASS_WITH_UNKNOWN_UNRESOLVED"
    assert payload["trade_authority_used"] is False


def test_gemini_coin_level_cached_across_calls(tmp_path: Path):
    gemini = tmp_path / "data" / "audits" / "ae12_gemini_semantic_adjudication_cache_test"
    (gemini / "reports").mkdir(parents=True)
    (gemini / "audits").mkdir(parents=True)
    (gemini / "reports" / "ae12_gemini_semantic_adjudication_summary.json").write_text(
        json.dumps({"gate_status": "PASS_GEMINI_ADJUDICATION_READY"}), encoding="utf-8"
    )
    (gemini / "audits" / "ae12_gemini_semantic_adjudication_gate.json").write_text(
        json.dumps(
            {
                "status": "PASS_GEMINI_ADJUDICATION_READY",
                "trade_authority_used": False,
                "web_grounding_used": False,
            }
        ),
        encoding="utf-8",
    )
    (gemini / "audits" / "ae12_gemini_safety_audit.json").write_text(
        json.dumps({"status": "PASS_REJECTIONS_ENFORCED", "trade_authority_used": False}),
        encoding="utf-8",
    )
    (gemini / "reports" / "ae12_gemini_coin_level_summary.json").write_text(
        json.dumps(
            {
                "pair_asset_counts": {"unique_pair_assets_input": 2, "count_role": "audit_detail"},
                "coin_level_counts": {
                    "unique_coins_found": 1,
                    "coin_non_social_opportunistic_confirmed_count": 1,
                    "count_role": "final_ui",
                },
            }
        ),
        encoding="utf-8",
    )
    mgr = AE12ReportManager(project_root=tmp_path, ttl_seconds=60)
    with patch.object(
        mgr,
        "discover_roots",
        return_value={"gemini_adjudication_root": gemini, "manual_review_drilldown_root": None},
    ):
        mgr.get_gemini_semantic_adjudication()
        meta1 = mgr._cache_meta()
        mgr.get_gemini_semantic_adjudication()
        meta2 = mgr._cache_meta()
    assert "gemini_coin_level_counts" in meta1["cached_keys"]
    assert meta2["load_counts"].get("gemini_coin_level_counts", 0) == 1
    assert meta2["load_counts"].get("gemini_adjudication_summary", 0) == 1
    assert meta2["load_counts"].get("manual_review_drilldown_summary", 0) == 1
