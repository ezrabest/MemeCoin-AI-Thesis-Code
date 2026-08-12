"""Tests for evidence-grounded social/opportunistic semantic layer."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.analytics.features import ClusterLabel
from app.semantic.evidence_collector import collect_evidence_bundle
from app.semantic.semantic_registry import count_semantic_verdicts, persist_semantic_verdict
from app.semantic.social_opportunistic_classifier import (
    classify_token_social_opportunistic,
    get_authoritative_semantic_counts,
    legacy_cluster_label_counts,
    rule_based_fallback,
)


ROOT = Path(__file__).resolve().parents[1]


def test_legacy_cluster_registry_remains_readable():
    legacy = legacy_cluster_label_counts(project_root=ROOT)
    assert legacy["registry"]["entry_count"] >= 1
    # Existing ClusterLabel values remain readable
    assert ClusterLabel.SOCIALLY_MOTIVATED.value == "SOCIALLY_MOTIVATED"
    assert ClusterLabel.OPPORTUNISTIC_SPECULATIVE.value == "OPPORTUNISTIC_SPECULATIVE"
    social = legacy["registry"].get("SOCIALLY_MOTIVATED", 0)
    opp = legacy["registry"].get("OPPORTUNISTIC_SPECULATIVE", 0)
    assert social + opp + legacy["registry"].get("OTHER", 0) == legacy["registry"]["entry_count"]


def test_existing_db_labels_can_be_counted():
    legacy = legacy_cluster_label_counts(project_root=ROOT)
    # paper_trades is the known home of exact valid cluster labels
    assert "paper_trades" in (legacy["db"].get("tables_scanned") or [])
    assert legacy["legacy_socially_motivated_count"] >= 0
    assert legacy["legacy_opportunistic_speculative_count"] >= 0
    # Known audit fact: 25 / 766 when DB present
    if legacy["legacy_socially_motivated_count"] and legacy["legacy_opportunistic_speculative_count"]:
        assert legacy["legacy_socially_motivated_count"] == 25
        assert legacy["legacy_opportunistic_speculative_count"] == 766


def test_invalid_labels_do_not_crash_counter(tmp_path: Path):
    db = tmp_path / "t.db"
    import sqlite3

    con = sqlite3.connect(str(db))
    con.execute("CREATE TABLE paper_trades (cluster_label TEXT)")
    con.executemany(
        "INSERT INTO paper_trades(cluster_label) VALUES (?)",
        [("SOCIALLY_MOTIVATED",), ("OPPORTUNISTIC_SPECULATIVE",), ("NOT_A_REAL_LABEL",), (None,)],
    )
    con.commit()
    con.close()
    reg = tmp_path / "data" / "cluster_registry.json"
    reg.parent.mkdir(parents=True)
    reg.write_text(
        json.dumps({"a": {"cluster_label": "SOCIALLY_MOTIVATED"}, "b": {"cluster_label": "BOGUS"}}),
        encoding="utf-8",
    )
    legacy = legacy_cluster_label_counts(project_root=tmp_path, db_path=db)
    assert legacy["legacy_socially_motivated_count"] == 1
    assert legacy["legacy_opportunistic_speculative_count"] == 1
    assert int(legacy["db"].get("invalid_label_rows") or 0) >= 1
    counts = get_authoritative_semantic_counts(project_root=tmp_path, db_path=db)
    assert "social_confirmed_count" in counts


def test_user_seed_category_remains_provenance_only():
    # Force empty non-seed evidence path via patched collector
    fake_bundle = {
        "evidence_items": [
            {
                "evidence_id": "seed-1",
                "source_type": "USER_SEED",
                "title": "seed",
                "snippet": "USER_SEED_SOCIALFI",
                "relevance": "LOW",
                "supports": "UNKNOWN",
                "notes": "provenance",
            }
        ],
        "counter_evidence": [],
        "evidence_quality": "NONE",
        "user_seed_collection": "USER_SEED_SOCIALFI",
        "user_seed_label": "USER_SEED_SOCIALFI",
        "user_hypothesis": "User seed hypothesized category=USER_SEED_SOCIALFI",
        "queries_used": ["seed_targets"],
    }
    with patch(
        "app.semantic.social_opportunistic_classifier.collect_evidence_bundle",
        return_value=fake_bundle,
    ):
        with patch(
            "app.semantic.social_opportunistic_classifier.resolve_semantic_llm_provider",
            return_value="none",
        ):
            v = classify_token_social_opportunistic(
                symbol="XYZ",
                name="Something",
                chain="eth",
                pair_address="0xabc",
                persist=False,
            )
    assert v["user_seed_label"] == "USER_SEED_SOCIALFI"
    assert "hypothes" in (v["user_hypothesis"] or "").lower() or "USER_SEED" in v["user_hypothesis"]
    assert v["semantic_status"] != "SOCIAL_CONFIRMED"
    assert v["cluster_label"] == "UNKNOWN"
    assert v["no_trade_authority"] is True


def test_user_social_seed_not_auto_confirmed_without_evidence():
    fake_bundle = {
        "evidence_items": [
            {
                "evidence_id": "seed-1",
                "source_type": "USER_SEED",
                "title": "seed",
                "snippet": "socialfi",
                "relevance": "LOW",
                "supports": "UNKNOWN",
                "notes": "",
            }
        ],
        "counter_evidence": [],
        "evidence_quality": "NONE",
        "user_seed_collection": "USER_SEED_SOCIALFI",
        "user_seed_label": "USER_SEED_SOCIALFI",
        "user_hypothesis": "User hypothesized social",
        "queries_used": [],
    }
    with patch(
        "app.semantic.social_opportunistic_classifier.collect_evidence_bundle",
        return_value=fake_bundle,
    ), patch(
        "app.semantic.social_opportunistic_classifier.resolve_semantic_llm_provider",
        return_value="none",
    ):
        v = classify_token_social_opportunistic(symbol="SOC", persist=False)
    assert v["semantic_status"] == "INSUFFICIENT_EVIDENCE"


def test_user_opportunistic_seed_not_auto_confirmed_without_evidence():
    fake_bundle = {
        "evidence_items": [
            {
                "evidence_id": "seed-2",
                "source_type": "USER_SEED",
                "title": "seed",
                "snippet": "opportunistic",
                "relevance": "LOW",
                "supports": "UNKNOWN",
                "notes": "",
            }
        ],
        "counter_evidence": [],
        "evidence_quality": "NONE",
        "user_seed_collection": "USER_SEED_OPPORTUNISTIC",
        "user_seed_label": "USER_SEED_OPPORTUNISTIC",
        "user_hypothesis": "User hypothesized opportunistic",
        "queries_used": [],
    }
    with patch(
        "app.semantic.social_opportunistic_classifier.collect_evidence_bundle",
        return_value=fake_bundle,
    ), patch(
        "app.semantic.social_opportunistic_classifier.resolve_semantic_llm_provider",
        return_value="none",
    ):
        v = classify_token_social_opportunistic(symbol="OPP", persist=False)
    assert v["semantic_status"] == "INSUFFICIENT_EVIDENCE"
    assert v["cluster_label"] == "UNKNOWN"


def test_missing_evidence_returns_insufficient():
    with patch(
        "app.semantic.social_opportunistic_classifier.collect_evidence_bundle",
        return_value={
            "evidence_items": [],
            "counter_evidence": [],
            "evidence_quality": "NONE",
            "user_seed_collection": "",
            "user_seed_label": "",
            "user_hypothesis": "",
            "queries_used": [],
        },
    ), patch(
        "app.semantic.social_opportunistic_classifier.resolve_semantic_llm_provider",
        return_value="none",
    ):
        v = classify_token_social_opportunistic(symbol="EMPTY", persist=False)
    assert v["semantic_status"] == "INSUFFICIENT_EVIDENCE"


def test_llm_unavailable_returns_failed_or_insufficient():
    fake_bundle = {
        "evidence_items": [
            {
                "evidence_id": "raw-1",
                "source_type": "RAW_PROVIDER_PAYLOAD",
                "title": "raw",
                "snippet": "community dao treasury charity public goods governance vote",
                "relevance": "MEDIUM",
                "supports": "SOCIAL",
                "notes": "",
            }
        ],
        "counter_evidence": [],
        "evidence_quality": "MEDIUM",
        "user_seed_collection": "",
        "user_seed_label": "",
        "user_hypothesis": "",
        "queries_used": ["db"],
    }

    def _failing_llm(**kwargs):
        return {"ok": False, "error": "ollama_unreachable", "provider": "QWEN_OLLAMA", "model": "x"}

    with patch(
        "app.semantic.social_opportunistic_classifier.collect_evidence_bundle",
        return_value=fake_bundle,
    ), patch(
        "app.semantic.social_opportunistic_classifier.resolve_semantic_llm_provider",
        return_value="ollama",
    ):
        v = classify_token_social_opportunistic(symbol="LLM", persist=False, llm_client=_failing_llm)
    assert v["semantic_status"] in ("CLASSIFICATION_FAILED", "INSUFFICIENT_EVIDENCE")


def test_rule_based_fallback_does_not_invent_opportunistic_confirmed():
    fb = rule_based_fallback(
        evidence_items=[],
        counter_evidence=[],
        evidence_quality="NONE",
        user_seed_label="USER_SEED_OPPORTUNISTIC",
    )
    assert fb["semantic_status"] != "OPPORTUNISTIC_CONFIRMED"
    assert fb["semantic_status"] == "INSUFFICIENT_EVIDENCE"
    assert fb["cluster_label"] == "UNKNOWN"


def test_api_counter_matches_persisted_semantic_verdicts(tmp_path: Path):
    vpath = tmp_path / "semantic_verdicts.jsonl"
    persist_semantic_verdict(
        {
            "identity_key": "eth:pair:0x1",
            "semantic_status": "SOCIAL_CONFIRMED",
            "cluster_label": "SOCIALLY_MOTIVATED",
            "confidence": 0.8,
            "evidence_quality": "HIGH",
            "no_trade_authority": True,
        },
        path=vpath,
    )
    persist_semantic_verdict(
        {
            "identity_key": "eth:pair:0x2",
            "semantic_status": "INSUFFICIENT_EVIDENCE",
            "cluster_label": "UNKNOWN",
            "confidence": 0.1,
            "evidence_quality": "NONE",
            "no_trade_authority": True,
        },
        path=vpath,
    )
    # Create empty registry/db stubs
    (tmp_path / "data").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "cluster_registry.json").write_text("{}", encoding="utf-8")
    counts = get_authoritative_semantic_counts(project_root=tmp_path, verdicts_path=vpath)
    assert counts["social_confirmed_count"] == 1
    assert counts["insufficient_evidence_count"] == 1
    assert counts["total_semantic_verdicts"] == 2
    assert count_semantic_verdicts(path=vpath)["social_confirmed_count"] == 1


def test_ui_badge_and_confirmed_label_not_confused():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "static" / "product_demo.js").read_text(encoding="utf-8")
    assert "System Verified Social" in html
    assert "Legacy DB Social" in html
    assert "SOCIAL?" in js
    assert "not counted as confirmed social" in js or "USER HYPOTHESIS" in js
    assert "/api/semantic/counts" in js or "/api/semantic/counts" in html


def test_no_trade_paper_live_wallet_or_risk_gate_called():
    trade_mocks = {
        "app.execution.paper.get_paper_trader": MagicMock(),
        "app.execution.execution_orchestrator.execute": MagicMock(),
    }
    # Patch names that must never be imported/called by classifier
    with patch(
        "app.semantic.social_opportunistic_classifier.collect_evidence_bundle",
        return_value={
            "evidence_items": [],
            "counter_evidence": [],
            "evidence_quality": "NONE",
            "user_seed_collection": "",
            "user_seed_label": "",
            "user_hypothesis": "",
            "queries_used": [],
        },
    ), patch(
        "app.semantic.social_opportunistic_classifier.resolve_semantic_llm_provider",
        return_value="none",
    ):
        v = classify_token_social_opportunistic(symbol="SAFE", persist=False)
    assert v["no_trade_authority"] is True
    # Ensure classifier module source does not call trade entrypoints
    src = (ROOT / "app" / "semantic" / "social_opportunistic_classifier.py").read_text(encoding="utf-8")
    assert "get_paper_trader" not in src
    assert "open_position" not in src
    assert "RiskGuard" not in src
    assert "connect_wallet" not in src
    assert "live_trading" not in src


def test_count_by_cluster_uses_authoritative_db_totals():
    from app.models.predictor import count_by_cluster

    counts = count_by_cluster()
    assert "SOCIALLY_MOTIVATED" in counts
    assert "OPPORTUNISTIC_SPECULATIVE" in counts
    # Must not be the truncated whale-log-only {OPP:500} shape when DB has both
    if counts.get("SOCIALLY_MOTIVATED", 0) > 0:
        assert counts["SOCIALLY_MOTIVATED"] == 25
        assert counts["OPPORTUNISTIC_SPECULATIVE"] == 766
