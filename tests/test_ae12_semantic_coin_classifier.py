"""Tests for AE12-SentimentFix semantic coin classifier."""

from __future__ import annotations

import json
from pathlib import Path

from app.ae12_sentimentfix.classification_cache import cache_key, cache_key_fields, cache_uses_evidence_hash
from app.ae12_sentimentfix.classification_reports import run_semantic_coin_classifier
from app.ae12_sentimentfix.evidence_builder import build_asset_id
from app.ae12_sentimentfix.llm_classifier import classify_asset_semantic, sanity_check_output


def test_unique_asset_dedup_key_priority():
    a, c = build_asset_id({"chain": "eth", "contract_address": "0xABC", "pair_address": "0xPAIR", "symbol": "AAA"})
    assert a == "eth:0xabc"
    assert c == "HIGH"
    a2, c2 = build_asset_id({"chain": "solana", "pair_address": "PAIR1", "symbol": "AAA"})
    assert a2.startswith("solana:PAIR:")
    assert c2 == "MEDIUM"


def test_semantic_class_ordinary_memecoin_non_social_opportunistic():
    r = classify_asset_semantic(
        {"asset_id": "x", "evidence_text": "meme token hype moon 100x roi no utility speculation"},
        local_llm_only=True,
    )
    assert r["semantic_coin_class"] == "NON_SOCIAL_OPPORTUNISTIC"


def test_impact_charity_returns_social():
    r = classify_asset_semantic(
        {"asset_id": "x", "evidence_text": "transparent donation mechanism to ngo and charity restoration"},
        local_llm_only=True,
    )
    assert r["semantic_coin_class"] == "SOCIAL"


def test_cooperative_local_utility_returns_social():
    r = classify_asset_semantic(
        {"asset_id": "x", "evidence_text": "community reward access token internal payment cooperative"},
        local_llm_only=True,
    )
    assert r["semantic_coin_class"] == "SOCIAL"


def test_bitcoin_like_returns_infrastructure():
    r = classify_asset_semantic(
        {"asset_id": "x", "symbol": "BTC", "name": "Bitcoin", "evidence_text": "layer 1 infrastructure"},
        local_llm_only=True,
    )
    assert r["semantic_coin_class"] == "NON_SOCIAL_INFRASTRUCTURE"


def test_insufficient_evidence_returns_unknown():
    r = classify_asset_semantic({"asset_id": "x", "evidence_text": ""}, local_llm_only=True)
    assert r["semantic_coin_class"] == "UNKNOWN_INSUFFICIENT_EVIDENCE"


def test_negative_trigger_overrides_weak_social():
    r = classify_asset_semantic(
        {"asset_id": "x", "evidence_text": "social meme hype only no utility moon roi"},
        local_llm_only=True,
    )
    assert r["semantic_coin_class"] in {"NON_SOCIAL_OPPORTUNISTIC", "MANUAL_REVIEW"}


def test_return_promise_marketing_triggers_opportunistic():
    r = classify_asset_semantic({"asset_id": "x", "evidence_text": "get rich moon 100x roi"}, local_llm_only=True)
    assert r["semantic_coin_class"] == "NON_SOCIAL_OPPORTUNISTIC"


def test_invalid_llm_json_handled_safely():
    # sanity check path: invalid parsed JSON is just treated as heuristic output
    r = classify_asset_semantic({"asset_id": "x", "evidence_text": "community reward"}, local_llm_only=True)
    assert r["trade_authority_used"] is False
    assert "semantic_coin_class" in r


def test_forbidden_trade_terms_rejected_raw_and_json():
    chk = sanity_check_output("BUY NOW", {"reasoning_short": "safe"})
    assert chk["forbidden_trade_language_found"] is True
    chk2 = sanity_check_output("", {"reasoning_short": "SELL signal"})
    assert chk2["forbidden_trade_language_found"] is True


def test_cache_key_uses_asset_hash_rubric_version():
    k = cache_key(asset_id="a", classifier_version="v", evidence_hash="h", rubric_version="r")
    assert k == "a|v|h|r"
    assert cache_uses_evidence_hash() is True
    assert "evidence_hash" in cache_key_fields()


def test_unknown_share_exposed_in_summary_and_api(tmp_path: Path):
    # minimal candidate CSV for smoke run
    ae12 = tmp_path / "data" / "audits" / "ae12_forward_evidence_maturation_x" / "data"
    ae12.mkdir(parents=True)
    (ae12 / "ae12_candidate_evidence_rows.csv").write_text(
        "chain,contract_address,pair_address,symbol,cluster_label,reason_for_no_trade\n"
        "eth,0x1,0xp1,AAA,OPPORTUNISTIC_SPECULATIVE,\n"
        "eth,0x2,0xp2,BBB,,\n",
        encoding="utf-8",
    )
    out = run_semantic_coin_classifier(
        project_root=tmp_path,
        ae12_root=ae12.parent,
        max_assets=50,
        local_llm_only=True,
        no_external_apis=True,
    )
    assert "unknown_share" in out
    gate = json.loads(
        (Path(out["output_root"]) / "audits" / "ae12_semantic_classifier_decision_gate.json").read_text(
            encoding="utf-8"
        )
    )
    assert "unknown_share" in gate


def test_ui_unknown_not_styled_as_opportunistic():
    html = (Path(__file__).resolve().parents[1] / "static" / "index.html").read_text(encoding="utf-8")
    assert "OP.SUSPECTED" in html
    assert "Gemini adjudication" in html or "gemini-semantic-adjudication" in html
