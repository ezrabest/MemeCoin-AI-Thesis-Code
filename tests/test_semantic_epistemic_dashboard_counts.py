"""Epistemic-level separation for social/opportunistic dashboard counts."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.semantic.curated_hypotheses import count_curated_hypotheses, resolve_project_path
from app.semantic.social_opportunistic_classifier import get_authoritative_semantic_counts

ROOT = Path(__file__).resolve().parents[1]


def test_system_verified_social_zero_when_verdicts_empty(tmp_path: Path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "cluster_registry.json").write_text("{}", encoding="utf-8")
    vpath = tmp_path / "data" / "semantic_verdicts.jsonl"
    # empty / missing verdicts
    counts = get_authoritative_semantic_counts(
        project_root=tmp_path,
        verdicts_path=vpath,
        db_path=tmp_path / "missing.db",
        environ={"CLEAN_FORWARD_USE_CURATED_TARGETS": "false"},
    )
    assert counts["system_verified_social_count"] == 0
    assert counts["social_confirmed_count"] == 0
    assert counts["system_verified_total_count"] == 0


def test_legacy_db_social_and_opportunistic_visible():
    counts = get_authoritative_semantic_counts(
        project_root=ROOT,
        environ={"CLEAN_FORWARD_USE_CURATED_TARGETS": "false"},
    )
    assert counts["legacy_db_social_count"] == 25
    assert counts["legacy_socially_motivated_count"] == 25
    assert counts["legacy_db_opportunistic_count"] == 766
    assert counts["legacy_opportunistic_speculative_count"] == 766


def test_registry_social_and_opportunistic_visible():
    counts = get_authoritative_semantic_counts(
        project_root=ROOT,
        environ={"CLEAN_FORWARD_USE_CURATED_TARGETS": "false"},
    )
    assert counts["legacy_registry_social_count"] == 1
    assert counts["legacy_registry_opportunistic_count"] == 267
    assert counts["legacy_registry_total_count"] == 268


def test_curated_path_resolves_relative_to_project_root_not_cwd(tmp_path: Path, monkeypatch):
    # Create a fake project with relative curated path
    data = tmp_path / "data" / "SeedTargets"
    data.mkdir(parents=True)
    csv_path = data / "clean_forward_curated_ready_targets_active.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed_collection", "symbol"])
        w.writeheader()
        w.writerow({"seed_collection": "USER_SEED_REFI", "symbol": "A"})
        w.writerow({"seed_collection": "USER_SEED_OPPORTUNISTIC", "symbol": "B"})

    other_cwd = tmp_path / "other_cwd"
    other_cwd.mkdir()
    monkeypatch.chdir(other_cwd)
    assert Path.cwd() == other_cwd

    rel = "data/SeedTargets/clean_forward_curated_ready_targets_active.csv"
    resolved = resolve_project_path(rel, project_root=tmp_path)
    assert resolved is not None
    assert resolved.is_file()
    assert resolved == csv_path.resolve()

    env = {
        "CLEAN_FORWARD_USE_CURATED_TARGETS": "true",
        "CLEAN_FORWARD_CURATED_TARGETS_PATH": rel,
    }
    curated = count_curated_hypotheses(project_root=tmp_path, environ=env)
    assert curated["curated_targets_file_exists"] is True
    assert curated["curated_social_hypothesis_count"] == 1
    assert curated["curated_opportunistic_hypothesis_count"] == 1


def test_curated_path_works_when_cwd_changed(tmp_path: Path, monkeypatch):
    csv_file = tmp_path / "hyp.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["user_seed_label"])
        w.writeheader()
        w.writerow({"user_seed_label": "SOCIAL"})
    weird = tmp_path / "somewhere" / "else"
    weird.mkdir(parents=True)
    monkeypatch.chdir(weird)
    resolved = resolve_project_path(str(csv_file), project_root=tmp_path)
    assert resolved == csv_file
    assert resolved.is_file()


def test_missing_curated_file_does_not_crash():
    from app.api import app

    client = TestClient(app)
    with patch.dict(
        os.environ,
        {
            "CLEAN_FORWARD_USE_CURATED_TARGETS": "true",
            "CLEAN_FORWARD_CURATED_TARGETS_PATH": "data/SeedTargets/does_not_exist_xyz.csv",
        },
        clear=False,
    ):
        r = client.get("/api/semantic/counts")
    assert r.status_code == 200
    body = r.json()
    assert body["curated_targets_file_exists"] is False
    assert body["curated_total_hypothesis_count"] == 0


def test_curated_social_hypothesis_only_increments_curated(tmp_path: Path):
    csv_file = tmp_path / "c.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["seed_collection"])
        w.writeheader()
        for lab in (
            "USER_SEED_REFI",
            "USER_SEED_COMMUNITY_DAO",
            "USER_SEED_SOCIALFI",
            "USER_SEED_FAN_TOKEN",
            "SOCIALLY_MOTIVATED",
            "SOCIAL_CONFIRMED",
            "SOCIAL",
            "SOCIAL?",
        ):
            w.writerow({"seed_collection": lab})
    curated = count_curated_hypotheses(
        project_root=tmp_path,
        environ={
            "CLEAN_FORWARD_USE_CURATED_TARGETS": "true",
            "CLEAN_FORWARD_CURATED_TARGETS_PATH": str(csv_file),
        },
    )
    assert curated["curated_social_hypothesis_count"] == 8
    assert curated["curated_opportunistic_hypothesis_count"] == 0

    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "cluster_registry.json").write_text("{}", encoding="utf-8")
    counts = get_authoritative_semantic_counts(
        project_root=tmp_path,
        db_path=tmp_path / "none.db",
        environ={
            "CLEAN_FORWARD_USE_CURATED_TARGETS": "true",
            "CLEAN_FORWARD_CURATED_TARGETS_PATH": str(csv_file),
        },
    )
    assert counts["curated_social_hypothesis_count"] == 8
    assert counts["social_confirmed_count"] == 0
    assert counts["system_verified_social_count"] == 0
    assert counts["opportunistic_confirmed_count"] == 0


def test_curated_opportunistic_hypothesis_only_increments_curated(tmp_path: Path):
    (tmp_path / "data").mkdir(exist_ok=True)
    (tmp_path / "data" / "cluster_registry.json").write_text("{}", encoding="utf-8")
    csv_file = tmp_path / "o.csv"
    with csv_file.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["category"])
        w.writeheader()
        for lab in (
            "USER_SEED_OPPORTUNISTIC",
            "OPPORTUNISTIC_SPECULATIVE",
            "OPPORTUNISTIC_CONFIRMED",
            "OPPORTUNISTIC",
        ):
            w.writerow({"category": lab})
    counts = get_authoritative_semantic_counts(
        project_root=tmp_path,
        db_path=tmp_path / "none.db",
        environ={
            "CLEAN_FORWARD_USE_CURATED_TARGETS": "true",
            "CLEAN_FORWARD_CURATED_TARGETS_PATH": str(csv_file),
        },
    )
    assert counts["curated_opportunistic_hypothesis_count"] == 4
    assert counts["system_verified_opportunistic_count"] == 0
    assert counts["opportunistic_confirmed_count"] == 0


def test_frontend_and_api_avoid_ambiguous_social_label():
    html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")
    js = (ROOT / "static" / "product_demo.js").read_text(encoding="utf-8")
    assert "System Verified Social" in html
    assert "Legacy DB Social" in html
    assert "Registry Social" in html
    assert "Curated Social Hypotheses" in html
    # Avoid ambiguous lone "Social:" dashboard label for system verified
    assert "System Verified Social" in html
    assert "never summed into one Social" in html or "Do not collapse" in html or "do not collapse" in html.lower()
    assert "system_verified_social_count" in js or "System Verified" in html
    assert "no_aggregated_social_total" in (
        get_authoritative_semantic_counts(
            project_root=ROOT,
            environ={"CLEAN_FORWARD_USE_CURATED_TARGETS": "false"},
        )
    )


def test_analytics_summary_cluster_counts_preserved():
    from app.api import app

    client = TestClient(app)
    r = client.get("/api/analytics/summary")
    assert r.status_code == 200
    body = r.json()
    clusters = body.get("cluster_counts") or {}
    assert clusters.get("SOCIALLY_MOTIVATED") == 25
    assert clusters.get("OPPORTUNISTIC_SPECULATIVE") == 766
    sem = body.get("semantic_counts") or {}
    assert "system_verified_social_count" in sem
    assert "legacy_db_social_count" in sem
    assert "legacy_registry_social_count" in sem
    assert "curated_social_hypothesis_count" in sem
    assert sem["legacy_db_social_count"] == 25
    assert sem["legacy_db_opportunistic_count"] == 766


def test_no_trade_llm_or_risk_side_effects_in_counts_path():
    src = (ROOT / "app" / "semantic" / "curated_hypotheses.py").read_text(encoding="utf-8")
    assert "get_paper_trader" not in src
    assert "open_position" not in src
    assert "call_semantic_llm" not in src
    assert "generate_content" not in src
    assert "RiskGuard" not in src
    counts = get_authoritative_semantic_counts(
        project_root=ROOT,
        environ={"CLEAN_FORWARD_USE_CURATED_TARGETS": "false"},
    )
    assert counts["no_trade_authority"] is True
    assert counts["social_confirmed_count"] == counts["system_verified_social_count"]


def test_semantic_counts_endpoint_includes_epistemic_fields():
    from app.api import app

    client = TestClient(app)
    r = client.get("/api/semantic/counts")
    assert r.status_code == 200
    body = r.json()
    for key in (
        "system_verified_social_count",
        "system_verified_opportunistic_count",
        "legacy_db_social_count",
        "legacy_db_opportunistic_count",
        "legacy_registry_social_count",
        "legacy_registry_opportunistic_count",
        "curated_targets_enabled",
        "curated_targets_path",
        "curated_targets_resolved_path",
        "curated_targets_file_exists",
        "curated_social_hypothesis_count",
        "curated_opportunistic_hypothesis_count",
        "curated_unknown_hypothesis_count",
        "curated_total_hypothesis_count",
        "social_confirmed_count",
        "legacy_socially_motivated_count",
    ):
        assert key in body
    assert body["no_aggregated_social_total"] is True
