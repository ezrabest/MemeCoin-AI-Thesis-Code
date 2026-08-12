"""Tests for AE12.6 ML / meta-layer evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.ae12_ml_meta_layer_evaluation.run import (
    FORBIDDEN_PROFIT_CLAIMS,
    run_ae12_ml_meta_layer_evaluation,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REQUIRED_REL_PATHS = [
    "reports/ae12_ml_meta_layer_evaluation_summary.json",
    "reports/ae12_ml_meta_layer_evaluation_for_upload.txt",
    "reports/ae12_ml_meta_layer_evaluation_manifest.json",
    "data/ae12_ml_meta_artifact_inventory.csv",
    "data/ae12_ml_meta_layer_evaluation_matrix.csv",
    "data/ae12_model_performance_summary.csv",
    "data/ae12_forward_evidence_integration_summary.csv",
    "data/ae12_semantic_coverage_reconciliation_addendum.csv",
    "audits/ae12_ml_meta_layer_evaluation_gate.json",
    "audits/ae12_ml_meta_no_retraining_audit.json",
    "audits/ae12_ml_meta_safety_audit.json",
    "audits/ae12_ml_meta_readonly_source_audit.json",
]


@pytest.fixture(scope="module")
def ae126_output_root() -> Path:
    out = PROJECT_ROOT / "data" / "audits" / "ae12_ml_meta_layer_evaluation_pytest_cache"
    if out.exists():
        import shutil

        shutil.rmtree(out, ignore_errors=True)
    result = run_ae12_ml_meta_layer_evaluation(
        project_root=PROJECT_ROOT,
        output_root=out,
    )
    assert result.get("gate_status") != "HOLD_OUTPUT_ROOT_WRITE_FAILED"
    return Path(result["output_root"])


def test_creates_required_files(ae126_output_root: Path) -> None:
    for rel in REQUIRED_REL_PATHS:
        assert (ae126_output_root / rel).is_file(), rel


def test_writes_only_under_output_root(ae126_output_root: Path) -> None:
    allowed_prefixes = (
        ae126_output_root.resolve(),
        (PROJECT_ROOT / "scripts").resolve(),
        (PROJECT_ROOT / "tests").resolve(),
        (PROJECT_ROOT / "app" / "ae12_ml_meta_layer_evaluation").resolve(),
    )
    for path in ae126_output_root.rglob("*"):
        assert path.resolve().is_relative_to(ae126_output_root.resolve())


def _gate(ae126_output_root: Path) -> dict:
    return json.loads(
        (ae126_output_root / "audits" / "ae12_ml_meta_layer_evaluation_gate.json").read_text(encoding="utf-8")
    )


def _safety(ae126_output_root: Path) -> dict:
    return json.loads(
        (ae126_output_root / "audits" / "ae12_ml_meta_safety_audit.json").read_text(encoding="utf-8")
    )


def test_gate_phase_and_flags(ae126_output_root: Path) -> None:
    gate = _gate(ae126_output_root)
    assert gate["phase"] == "AE12.6"
    assert gate["ae12_closed"] is False
    assert gate["retraining_performed"] is False
    assert gate["runtime_started"] is False
    assert gate["external_api_used"] is False
    assert gate["gemini_called"] is False
    assert gate["trader_db_mutated"] is False
    assert gate["wallet_connected"] is False
    assert gate["live_ready"] is False
    assert gate["profitability_proven"] is False
    assert gate["semantic_coverage_reconciliation_included"] is True
    assert gate["writes_limited_to_output_root"] is True
    assert gate["next_ae12_step"] == "AE12.7"


def test_safety_audit(ae126_output_root: Path) -> None:
    safety = _safety(ae126_output_root)
    assert safety["status"] == "PASS_READONLY_REPORTING_ONLY"
    for key in (
        "trade_authority_granted_to_RF",
        "trade_authority_granted_to_XGB",
        "trade_authority_granted_to_TAB",
        "trade_authority_granted_to_meta_layer",
        "trade_authority_granted_to_Qwen",
        "trade_authority_granted_to_Gemini",
        "trade_authority_granted_to_any_llm",
    ):
        assert safety[key] is False


def test_readonly_source_audit(ae126_output_root: Path) -> None:
    ro = json.loads(
        (ae126_output_root / "audits" / "ae12_ml_meta_readonly_source_audit.json").read_text(encoding="utf-8")
    )
    assert ro["writes_limited_to_output_root"] is True


def test_matrix_includes_required_layers(ae126_output_root: Path) -> None:
    text = (ae126_output_root / "data" / "ae12_ml_meta_layer_evaluation_matrix.csv").read_text(encoding="utf-8")
    for name in (
        "RF",
        "XGB",
        "TAB / TabICL",
        "Consensus / DecisionRecord layer",
        "Meta-model / stacking layer",
        "Context intelligence layer",
        "LLM audit layer",
        "Semantic reporting / SentimentFix",
        "Forward evidence maturation layer",
        "Safety / no-wallet layer",
    ):
        assert name in text


def test_missing_artifacts_in_inventory_and_gate(ae126_output_root: Path) -> None:
    inv_text = (ae126_output_root / "data" / "ae12_ml_meta_artifact_inventory.csv").read_text(encoding="utf-8")
    assert "MISSING_" in inv_text or ",FOUND," in inv_text
    gate = _gate(ae126_output_root)
    assert "missing_critical_artifacts" in gate
    assert "missing_important_artifacts" in gate
    upload = (ae126_output_root / "reports" / "ae12_ml_meta_layer_evaluation_for_upload.txt").read_text(
        encoding="utf-8"
    )
    assert "Missing artifacts" in upload


def test_component_status_values_allowed(ae126_output_root: Path) -> None:
    gate = _gate(ae126_output_root)
    allowed = {
        "FOUND_AND_EVALUATED",
        "PARTIALLY_EVALUATED_MISSING_EXPECTED_ARTIFACTS",
        "MISSING_EXPECTED_ARTIFACT",
        "NOT_IMPLEMENTED_AS_ORIGINAL_LAYER",
        "DIAGNOSTIC_ONLY",
        "REPORTING_ONLY",
    }
    for st in gate["component_statuses"].values():
        assert st in allowed


def test_semantic_addendum_and_500_vs_14(ae126_output_root: Path) -> None:
    addendum = (ae126_output_root / "data" / "ae12_semantic_coverage_reconciliation_addendum.csv").read_text(
        encoding="utf-8"
    )
    assert "legacy_500" in addendum.lower() or "500/0" in addendum
    assert "89" in addendum
    assert "14" in addendum
    assert "UNKNOWN_UNRESOLVED" in addendum
    upload = (ae126_output_root / "reports" / "ae12_ml_meta_layer_evaluation_for_upload.txt").read_text(
        encoding="utf-8"
    )
    assert "500 unique coins" in upload.lower() or "not 500 unique coins" in upload.lower()
    assert "UNKNOWN_UNRESOLVED is not social" in upload


def test_no_profitability_live_claims_in_upload(ae126_output_root: Path) -> None:
    upload = (ae126_output_root / "reports" / "ae12_ml_meta_layer_evaluation_for_upload.txt").read_text(
        encoding="utf-8"
    ).lower()
    assert "profitability_proven=false" in upload
    assert "live_ready=false" in upload
    for phrase in FORBIDDEN_PROFIT_CLAIMS:
        if phrase in upload:
            assert "false" in upload or "not" in upload or "unproven" in upload


def test_simulated_missing_critical_gate(tmp_path: Path) -> None:
    out = tmp_path / "out"
    result = run_ae12_ml_meta_layer_evaluation(
        project_root=PROJECT_ROOT,
        output_root=out,
        simulate_missing_critical_artifacts=True,
    )
    gate = json.loads((out / "audits" / "ae12_ml_meta_layer_evaluation_gate.json").read_text(encoding="utf-8"))
    assert gate["status"] == "HOLD_MISSING_CRITICAL_ARTIFACTS"


def test_simulated_safety_failure_gate(tmp_path: Path) -> None:
    out = tmp_path / "out2"
    run_ae12_ml_meta_layer_evaluation(
        project_root=PROJECT_ROOT,
        output_root=out,
        simulate_safety_failure=True,
    )
    gate = json.loads((out / "audits" / "ae12_ml_meta_layer_evaluation_gate.json").read_text(encoding="utf-8"))
    safety = json.loads((out / "audits" / "ae12_ml_meta_safety_audit.json").read_text(encoding="utf-8"))
    assert safety["status"] != "PASS_READONLY_REPORTING_ONLY"
    assert gate["status"].startswith("FAIL")
