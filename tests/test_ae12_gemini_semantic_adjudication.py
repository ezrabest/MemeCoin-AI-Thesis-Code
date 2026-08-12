"""Tests for AE12-SentimentFix Gemini semantic adjudication."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from app.ae12_reporting.report_manager import AE12ReportManager
from app.ae12_sentimentfix.adjudication_cache import cache_key, cache_key_fields, lookup_frozen_entry
from app.ae12_sentimentfix.adjudication_reports import run_gemini_semantic_adjudication
from app.ae12_sentimentfix.adjudication_safety import redact_secrets, sanity_check_adjudication_output
from app.ae12_sentimentfix.adjudication_schema import (
    is_confirmed_opportunistic,
    is_suspected_opportunistic,
    map_local_class_to_adjudication,
    normalize_adjudication_payload,
)
from app.ae12_sentimentfix.evidence_priority import select_priority_snippets
from app.ae12_sentimentfix.gemini_adjudicator import adjudicate_asset


def test_unknown_maps_to_op_suspected():
    assert map_local_class_to_adjudication("UNKNOWN_INSUFFICIENT_EVIDENCE") == "OPPORTUNISTIC_SUSPECTED"


def test_raw_evidence_status_preserved_separately():
    payload = normalize_adjudication_payload(
        {
            "semantic_coin_class": "OPPORTUNISTIC_SUSPECTED",
            "raw_evidence_status": "UNKNOWN_INSUFFICIENT_EVIDENCE",
        }
    )
    assert payload["semantic_coin_class"] == "OPPORTUNISTIC_SUSPECTED"
    assert payload["raw_evidence_status"] == "UNKNOWN_INSUFFICIENT_EVIDENCE"


def test_op_suspected_not_counted_as_confirmed_opportunistic():
    assert is_suspected_opportunistic("OPPORTUNISTIC_SUSPECTED") is True
    assert is_confirmed_opportunistic("OPPORTUNISTIC_SUSPECTED") is False
    assert is_confirmed_opportunistic("NON_SOCIAL_OPPORTUNISTIC_CONFIRMED") is True


def test_charity_returns_social_confirmed():
    r = adjudicate_asset(
        {"asset_id": "x", "evidence_text": "transparent donation mechanism to ngo charity restoration"},
        dry_run=True,
    )
    assert r["semantic_coin_class"] == "SOCIAL_CONFIRMED"


def test_dao_governance_returns_social_confirmed():
    r = adjudicate_asset(
        {"asset_id": "x", "evidence_text": "dao governance voting treasury"},
        dry_run=True,
    )
    assert r["semantic_coin_class"] == "SOCIAL_CONFIRMED"


def test_ordinary_memecoin_confirmed_when_clear():
    r = adjudicate_asset(
        {"asset_id": "x", "evidence_text": "meme token moon 100x roi no utility hype"},
        dry_run=True,
    )
    assert r["semantic_coin_class"] == "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED"


def test_no_evidence_returns_op_suspected():
    r = adjudicate_asset({"asset_id": "x", "evidence_text": ""}, dry_run=True)
    assert r["semantic_coin_class"] == "OPPORTUNISTIC_SUSPECTED"


def test_bitcoin_like_infrastructure_confirmed():
    r = adjudicate_asset(
        {"asset_id": "x", "symbol": "BTC", "name": "Bitcoin", "evidence_text": "layer 1"},
        dry_run=True,
    )
    assert r["semantic_coin_class"] == "NON_SOCIAL_INFRASTRUCTURE_CONFIRMED"


def test_forbidden_trade_words_rejected():
    chk = sanity_check_adjudication_output("BUY NOW", {"reasoning_short": "safe"})
    assert chk["forbidden_trade_language_found"] is True


def test_forbidden_trade_keys_rejected():
    chk = sanity_check_adjudication_output("", {"buy_signal": True, "semantic_coin_class": "OPPORTUNISTIC_SUSPECTED"})
    assert chk["forbidden_trade_key_found"] is True


def test_rejected_gemini_output_not_accepted():
    def bad_call(prompt):
        return (
            '{"semantic_coin_class":"OPPORTUNISTIC_SUSPECTED","raw_evidence_status":"MODEL_KNOWLEDGE_ONLY",'
            '"reasoning_short":"BUY NOW","semantic_social_score":0,"opportunistic_score":0.5,'
            '"infrastructure_score":0,"classification_confidence":0.5,'
            '"positive_criteria_met":[],"negative_triggers_met":[],"evidence_summary":"x",'
            '"evidence_quotes_or_markers":[],"source_urls":[],"requires_manual_review":false}',
            None,
        )

    r = adjudicate_asset(
        {"asset_id": "x", "symbol": "AAA", "evidence_text": "meme"},
        use_gemini=True,
        allow_external_apis=True,
        allow_model_knowledge_fallback=True,
        gemini_call=bad_call,
    )
    assert r.get("accepted") is False or r["semantic_coin_class"] in {"MANUAL_REVIEW", "OPPORTUNISTIC_SUSPECTED"}
    assert r["trade_authority_used"] is False


def test_cache_key_uses_asset_rubric_adjudicator():
    k = cache_key(asset_id="a", rubric_version="r", adjudicator_version="v")
    assert k == "a|r|v"
    assert cache_key_fields() == ["asset_id", "rubric_version", "adjudicator_version"]


def test_evidence_hash_change_does_not_auto_readjudicate():
    cache = {
        cache_key(asset_id="a", rubric_version="r", adjudicator_version="v"): {
            "decision_frozen": True,
            "evidence_hash_at_classification": "old",
            "adjudication": {"semantic_coin_class": "SOCIAL_CONFIRMED"},
        }
    }
    entry, meta = lookup_frozen_entry(
        asset_id="a",
        evidence_hash="new",
        cache=cache,
        force_refresh=False,
        rubric_version="r",
        adjudicator_version="v",
    )
    assert entry is not None
    assert meta["evidence_hash_changed_since_classification"] is True
    assert meta["stale_evidence_warning"] is True


def test_force_refresh_allows_readjudication():
    cache = {
        cache_key(asset_id="a", rubric_version="r", adjudicator_version="v"): {
            "decision_frozen": True,
            "adjudication": {"semantic_coin_class": "SOCIAL_CONFIRMED"},
        }
    }
    entry, _ = lookup_frozen_entry(asset_id="a", evidence_hash="x", cache=cache, force_refresh=True)
    assert entry is None


def test_api_key_redacted_from_artifacts():
    secret = "DUMMY"
    redacted = redact_secrets(f"key={secret}")
    assert secret not in redacted
    assert "[REDACTED" in redacted


def test_runtime_snippets_do_not_crowd_identity(tmp_path: Path):
    snippets = [
        {"text": "max_open_positions | mention_only", "snippet_type": "candidate_reason", "bucket": "runtime_context",
         "linkage_method": "EXACT_PAIR_ADDRESS_MATCH", "semantic_markers_found": [], "negative_markers_found": [],
         "source_table_or_file": "ae12", "source_row_id": "1"},
        {"text": "name:Memecoin", "snippet_type": "identity", "bucket": "identity_metadata",
         "linkage_method": "EXACT_SYMBOL_MATCH", "semantic_markers_found": ["meme"], "negative_markers_found": ["meme"],
         "source_table_or_file": "coins", "source_row_id": "2"},
    ]
    selected, audit = select_priority_snippets(snippets, max_total=2)
    texts = [s["text"] for s in selected]
    assert any("name:Memecoin" in t for t in texts)
    assert audit["truncation_strategy"] == "deterministic_bucket_quota"


def test_dry_run_produces_outputs_without_external_api(tmp_path: Path, monkeypatch):
    classifier_root = tmp_path / "data" / "audits" / "ae12_semantic_coin_classifier_test"
    data_dir = classifier_root / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "ae12_unique_coin_evidence_packages.csv").write_text(
        "asset_id,chain,symbol,name,evidence_text,evidence_hash\n"
        "eth:0x1,eth,AAA,TokenA,charity donation,hash1\n"
        "eth:0x2,eth,BBB,TokenB,,hash2\n",
        encoding="utf-8",
    )
    (data_dir / "ae12_semantic_coin_classifications.csv").write_text(
        "asset_id,semantic_coin_class\neth:0x1,UNKNOWN_INSUFFICIENT_EVIDENCE\neth:0x2,UNKNOWN_INSUFFICIENT_EVIDENCE\n",
        encoding="utf-8",
    )
    summary = run_gemini_semantic_adjudication(
        project_root=tmp_path,
        classifier_root=classifier_root,
        max_assets=10,
        dry_run=True,
        semantic_reporting_only=True,
    )
    assert summary["gate_status"] in {
        "PASS_WITH_OP_SUSPECTED_LIMITATION",
        "PASS_GEMINI_ADJUDICATION_READY",
        "HOLD_GEMINI_API_KEY_MISSING",
    }
    gate = json.loads(
        (Path(summary["output_root"]) / "audits" / "ae12_gemini_semantic_adjudication_gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert gate["external_api_used"] is False
    assert gate["gemini_used"] is False
    assert gate["trade_authority_used"] is False
    assert gate["api_key_logged"] is False
    upload = (Path(summary["output_root"]) / "reports" / "ae12_gemini_semantic_adjudication_for_upload.txt").read_text(
        encoding="utf-8"
    )
    assert "AIza" not in upload


def test_gemini_endpoint_final_flag_only_on_pass(tmp_path: Path):
    root = tmp_path / "data" / "audits" / "ae12_gemini_semantic_adjudication_test"
    (root / "reports").mkdir(parents=True)
    (root / "audits").mkdir(parents=True)
    (root / "reports" / "ae12_gemini_semantic_adjudication_summary.json").write_text(
        json.dumps({"gate_status": "PASS_GEMINI_ADJUDICATION_READY", "class_distribution": {}}),
        encoding="utf-8",
    )
    (root / "audits" / "ae12_gemini_semantic_adjudication_gate.json").write_text(
        json.dumps(
            {
                "status": "PASS_GEMINI_ADJUDICATION_READY",
                "opportunistic_suspected_count": 1,
                "social_confirmed_count": 0,
                "external_api_used": True,
                "gemini_used": True,
                "trade_authority_used": False,
            }
        ),
        encoding="utf-8",
    )
    mgr = AE12ReportManager(project_root=tmp_path)
    with patch.object(mgr, "discover_roots", return_value={"gemini_adjudication_root": root}):
        payload = mgr.get_gemini_semantic_adjudication()
    assert payload["final_semantic_adjudication"] is True
    assert payload["trade_authority_used"] is False


def test_hold_when_api_key_missing(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    classifier_root = tmp_path / "data" / "audits" / "ae12_semantic_coin_classifier_test2"
    data_dir = classifier_root / "data"
    data_dir.mkdir(parents=True)
    (data_dir / "ae12_unique_coin_evidence_packages.csv").write_text(
        "asset_id,chain,symbol,evidence_text,evidence_hash\neth:0x1,eth,AAA,x,h1\n",
        encoding="utf-8",
    )
    summary = run_gemini_semantic_adjudication(
        project_root=tmp_path,
        classifier_root=classifier_root,
        max_assets=5,
        use_gemini=True,
        allow_external_apis=True,
    )
    assert summary["gate_status"] == "HOLD_GEMINI_API_KEY_MISSING"


def test_safety_pass_rejections_enforced_when_forbidden_rejected_unused():
    from app.ae12_sentimentfix.adjudication_safety_status import (
        build_safety_audit,
        resolve_safety_audit_status,
    )

    safety = build_safety_audit(
        total_gemini_outputs=10,
        accepted_outputs=8,
        rejected_outputs=2,
        forbidden_terms={"BUY"},
        forbidden_keys=[],
        output_used_after_rejection=False,
        accepted_classifications_with_forbidden_language=0,
        trade_authority_used=False,
    )
    assert safety["status"] == "PASS_REJECTIONS_ENFORCED"
    assert resolve_safety_audit_status(safety) == "PASS_REJECTIONS_ENFORCED"


def test_safety_fail_when_rejected_output_used():
    from app.ae12_sentimentfix.adjudication_safety_status import (
        gate_allowed_with_safety,
        resolve_safety_audit_status,
    )

    safety = {
        "forbidden_trade_language_found": True,
        "forbidden_trade_key_found": False,
        "rejected_outputs": 1,
        "output_used_after_rejection": True,
        "accepted_classifications_with_forbidden_language": 0,
        "trade_authority_used": False,
    }
    assert resolve_safety_audit_status(safety) == "FAIL"
    assert gate_allowed_with_safety("PASS_GEMINI_ADJUDICATION_READY", safety) is False


def test_pair_to_coin_dedup_doge_usdc_and_wif_weth():
    from app.ae12_sentimentfix.coin_identity import (
        normalize_wrapped_native,
        parse_pair_symbol,
        resolve_coin_identity,
    )
    from app.ae12_sentimentfix.coin_level_aggregation import build_coin_level_payload

    base, quote = parse_pair_symbol("DOGE/USDC")
    assert base == "DOGE"
    assert quote == "USDC"
    assert normalize_wrapped_native("WETH") == "ETH"
    assert normalize_wrapped_native("weth") == "ETH"

    doge_rows = [
        {
            "asset_id": f"solana:PAIR:doge{i}",
            "chain": "solana",
            "symbol": "DOGE/USDC" if i % 2 == 0 else "doge/USDC",
            "name": "",
            "token_address": "",
            "pair_address": f"pair{i}",
            "semantic_coin_class": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
        }
        for i in range(3)
    ]
    wif_rows = [
        {
            "asset_id": f"robinhood:PAIR:wif{i}",
            "chain": "robinhood",
            "symbol": "WIF/WETH" if i % 2 == 0 else "wif/WETH",
            "name": "",
            "token_address": "",
            "pair_address": f"wifpair{i}",
            "semantic_coin_class": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
        }
        for i in range(4)
    ]
    for r in doge_rows + wif_rows:
        ident = resolve_coin_identity(r)
        assert ident["normalized_base_symbol"] in {"DOGE", "WIF"}
        assert ident["normalized_base_symbol"] not in {"USDC", "WETH", "ETH"}
        assert ident["quote_symbol"] in {"USDC", "WETH"}

    payload = build_coin_level_payload(doge_rows + wif_rows)
    assert payload["pair_asset_counts"]["unique_pair_assets_input"] == 7
    assert payload["coin_level_counts"]["unique_coins_found"] == 2
    assert payload["count_level_used_for_main_ui"] == "coin_level"
    coin_ids = {r["coin_id"] for r in payload["coin_rows"]}
    assert len(coin_ids) == 2


def test_gemini_api_exposes_pair_and_coin_counts_separately(tmp_path: Path):
    root = tmp_path / "data" / "audits" / "ae12_gemini_semantic_adjudication_20260716_999999"
    (root / "reports").mkdir(parents=True)
    (root / "audits").mkdir(parents=True)
    (root / "data").mkdir(parents=True)
    (root / "reports" / "ae12_gemini_semantic_adjudication_summary.json").write_text(
        json.dumps({"gate_status": "PASS_GEMINI_ADJUDICATION_READY", "class_distribution": {}}),
        encoding="utf-8",
    )
    (root / "audits" / "ae12_gemini_semantic_adjudication_gate.json").write_text(
        json.dumps(
            {
                "status": "PASS_GEMINI_ADJUDICATION_READY",
                "unique_assets_input": 2,
                "social_confirmed_count": 0,
                "non_social_opportunistic_confirmed_count": 2,
                "opportunistic_suspected_count": 0,
                "manual_review_count": 0,
                "external_api_used": True,
                "gemini_used": True,
                "web_grounding_used": False,
                "trade_authority_used": False,
            }
        ),
        encoding="utf-8",
    )
    (root / "audits" / "ae12_gemini_safety_audit.json").write_text(
        json.dumps(
            {
                "total_gemini_outputs": 2,
                "accepted_outputs": 1,
                "rejected_outputs": 1,
                "forbidden_trade_language_found": True,
                "forbidden_trade_key_found": False,
                "forbidden_terms_found": ["BUY"],
                "forbidden_keys_found": [],
                "trade_authority_used": False,
                "output_used_after_rejection": False,
                "accepted_classifications_with_forbidden_language": 0,
                "status": "FAIL",
            }
        ),
        encoding="utf-8",
    )
    (root / "data" / "ae12_gemini_asset_adjudications.csv").write_text(
        "asset_id,chain,token_address,pair_address,symbol,name,semantic_coin_class,raw_evidence_status\n"
        "robinhood:PAIR:a,robinhood,,0xa,WIF/WETH,,NON_SOCIAL_OPPORTUNISTIC_CONFIRMED,MODEL_KNOWLEDGE_ONLY\n"
        "robinhood:PAIR:b,robinhood,,0xb,wif/WETH,,NON_SOCIAL_OPPORTUNISTIC_CONFIRMED,MODEL_KNOWLEDGE_ONLY\n",
        encoding="utf-8",
    )
    mgr = AE12ReportManager(project_root=tmp_path)
    with patch.object(mgr, "discover_roots", return_value={"gemini_adjudication_root": root}):
        payload = mgr.get_gemini_semantic_adjudication()
    assert payload["safety_audit_status"] == "PASS_REJECTIONS_ENFORCED"
    assert payload["final_semantic_adjudication"] is True
    assert payload["pair_asset_counts"]["unique_pair_assets_input"] == 2
    assert payload["coin_level_counts"]["unique_coins_found"] == 1
    assert payload["count_level_used_for_main_ui"] == "coin_level"
    assert payload["trade_authority_used"] is False
    assert payload["live_ready"] is False
    assert payload["profitability_proven"] is False
    assert payload["web_grounding_used"] is False


def test_main_ui_does_not_dump_raw_json_for_ae12_examples():
    html = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")
    assert "ae12ExampleCards" in html
    assert "JSON.stringify(semanticClassifier.examples_by_class" not in html
    assert "JSON.stringify(geminiAdjudication.raw_evidence_status_distribution" not in html
    assert "deduplicated by coin/token identity" in html
    assert "Legacy Social Cluster" in html or "Legacy cluster counters are diagnostic" in html
    assert "/api/ae12/manual-review-drilldown" in html
    assert "UNKNOWN_UNRESOLVED" in html or "UNKNOWN UNRESOLVED" in html


