"""File loaders for AE12 artifacts — no mutation, graceful MISSING handling."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .schemas import FileLoadResult

# Relative paths under a maturation root
MATURATION_SUMMARY = Path("reports") / "ae12_forward_evidence_summary.json"
MATURATION_GATE = Path("reports") / "ae12_final_system_readiness_gate.json"
TRADE_VS_NO_TRADE = Path("data") / "ae12_trade_vs_no_trade_comparison.csv"
STRICT_VS_EXPLORATION = Path("data") / "ae12_strict_vs_exploration_comparison.csv"
MISSED_WINNERS = Path("data") / "ae12_missed_winners_full.csv"
QWEN_LINKAGE = Path("data") / "ae12_qwen_ollama_linkage.csv"
WALLET_SAFETY = Path("audits") / "ae12_wallet_safety_audit.json"
REASON_COVERAGE = Path("audits") / "ae12_reason_coverage_audit.csv"
HORIZON_MATURITY = Path("audits") / "ae12_horizon_maturity_audit.csv"
MISSING_DATA_WARNINGS = Path("audits") / "ae12_missing_data_warning_audit.csv"

CENSUS_SUMMARY = Path("reports") / "ae12_data_census_summary.json"
QUALITY_SUMMARY = Path("reports") / "ae12_forward_evidence_quality_summary.json"


def _missing(path: Path | None) -> FileLoadResult:
    return {
        "status": "MISSING",
        "path": str(path) if path else None,
        "missing_file": str(path) if path else None,
        "error": None,
        "data": None,
    }


def _error(path: Path, message: str) -> FileLoadResult:
    return {
        "status": "ERROR",
        "path": str(path),
        "missing_file": None,
        "error": message,
        "data": None,
    }


def _ok(path: Path, data: Any) -> FileLoadResult:
    return {
        "status": "OK",
        "path": str(path),
        "missing_file": None,
        "error": None,
        "data": data,
    }


def load_json_file(path: Path) -> FileLoadResult:
    if path is None or not Path(path).is_file():
        return _missing(path)
    try:
        with Path(path).open("r", encoding="utf-8") as fh:
            return _ok(Path(path), json.load(fh))
    except Exception as exc:  # noqa: BLE001 — surface, don't crash API
        return _error(Path(path), f"{type(exc).__name__}: {exc}")


def load_csv_rows(path: Path, *, limit: int | None = None) -> FileLoadResult:
    """Load CSV rows as list[dict]. Optionally stop after `limit` data rows."""
    if path is None or not Path(path).is_file():
        return _missing(path)
    try:
        rows: list[dict[str, str]] = []
        with Path(path).open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                if limit is not None and i >= limit:
                    break
                rows.append(dict(row))
        return _ok(Path(path), rows)
    except Exception as exc:  # noqa: BLE001
        return _error(Path(path), f"{type(exc).__name__}: {exc}")


def load_csv_with_totals(path: Path, *, limit: int | None = None) -> FileLoadResult:
    """Load CSV and also report whether more rows exist beyond limit."""
    result = load_csv_rows(path, limit=limit)
    if result["status"] != "OK":
        return result
    truncated = False
    total_estimate: int | None = None
    if limit is not None and path.is_file():
        # Cheap line count for display (still read-only; avoid holding full CSV)
        try:
            with path.open("r", encoding="utf-8") as fh:
                total_estimate = max(0, sum(1 for _ in fh) - 1)
            truncated = total_estimate > len(result["data"] or [])
        except Exception:  # noqa: BLE001
            truncated = len(result["data"] or []) >= limit
    payload = {
        "rows": result["data"],
        "row_count_loaded": len(result["data"] or []),
        "total_rows_estimate": total_estimate,
        "truncated": truncated,
    }
    return _ok(Path(path), payload)


def maturation_paths(root: Path) -> dict[str, Path]:
    root = Path(root)
    return {
        "summary": root / MATURATION_SUMMARY,
        "gate": root / MATURATION_GATE,
        "trade_vs_no_trade": root / TRADE_VS_NO_TRADE,
        "strict_vs_exploration": root / STRICT_VS_EXPLORATION,
        "missed_winners": root / MISSED_WINNERS,
        "qwen_linkage": root / QWEN_LINKAGE,
        "wallet_safety": root / WALLET_SAFETY,
        "reason_coverage": root / REASON_COVERAGE,
        "horizon_maturity": root / HORIZON_MATURITY,
        "missing_data_warnings": root / MISSING_DATA_WARNINGS,
    }


def load_maturation_summary(root: Path | None) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_json_file(Path(root) / MATURATION_SUMMARY)


def load_readiness_gate(root: Path | None) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_json_file(Path(root) / MATURATION_GATE)


def load_wallet_safety(root: Path | None) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_json_file(Path(root) / WALLET_SAFETY)


def load_trade_vs_no_trade_csv(root: Path | None) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_csv_rows(Path(root) / TRADE_VS_NO_TRADE)


def load_strict_vs_exploration_csv(root: Path | None) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_csv_rows(Path(root) / STRICT_VS_EXPLORATION)


def load_missed_winners_csv(root: Path | None, *, limit: int = 100) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_csv_with_totals(Path(root) / MISSED_WINNERS, limit=limit)


def load_qwen_linkage_sample(root: Path | None, *, limit: int = 50) -> FileLoadResult:
    """Load a small sample only — prefer summary JSON for aggregate counts."""
    if root is None:
        return _missing(None)
    return load_csv_with_totals(Path(root) / QWEN_LINKAGE, limit=limit)


def load_horizon_maturity_audit(root: Path | None, *, limit: int | None = 500) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_csv_rows(Path(root) / HORIZON_MATURITY, limit=limit)


def load_missing_data_warning_sample(root: Path | None, *, limit: int = 100) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_csv_with_totals(Path(root) / MISSING_DATA_WARNINGS, limit=limit)


def load_reason_coverage_audit(root: Path | None, *, limit: int | None = 200) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_csv_rows(Path(root) / REASON_COVERAGE, limit=limit)


def load_census_summary(root: Path | None) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_json_file(Path(root) / CENSUS_SUMMARY)


def load_quality_summary(root: Path | None) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_json_file(Path(root) / QUALITY_SUMMARY)


TAXONOMY_SUMMARY = Path("reports") / "ae12_signal_taxonomy_audit_summary.json"
TAXONOMY_GATE = Path("audits") / "ae12_social_vs_opportunistic_decision_gate.json"
SENTIMENTFIX_SUMMARY = Path("reports") / "ae12_sentimentfix_summary.json"
SENTIMENTFIX_GATE = Path("audits") / "ae12_sentimentfix_decision_gate.json"
SEMANTIC_CLASSIFIER_SUMMARY = Path("reports") / "ae12_semantic_coin_classifier_summary.json"
SEMANTIC_CLASSIFIER_GATE = Path("audits") / "ae12_semantic_classifier_decision_gate.json"
GEMINI_ADJUDICATION_SUMMARY = Path("reports") / "ae12_gemini_semantic_adjudication_summary.json"
GEMINI_ADJUDICATION_GATE = Path("audits") / "ae12_gemini_semantic_adjudication_gate.json"
GEMINI_SAFETY_AUDIT = Path("audits") / "ae12_gemini_safety_audit.json"
MANUAL_REVIEW_DRILLDOWN_SUMMARY = Path("reports") / "ae12_manual_review_drilldown_summary.json"
MANUAL_REVIEW_DRILLDOWN_GATE = Path("audits") / "ae12_manual_review_drilldown_gate.json"


def load_taxonomy_summary(root: Path | None) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_json_file(Path(root) / TAXONOMY_SUMMARY)


def load_taxonomy_gate(root: Path | None) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_json_file(Path(root) / TAXONOMY_GATE)


def load_sentimentfix_summary(root: Path | None) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_json_file(Path(root) / SENTIMENTFIX_SUMMARY)


def load_sentimentfix_gate(root: Path | None) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_json_file(Path(root) / SENTIMENTFIX_GATE)


def load_semantic_classifier_summary(root: Path | None) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_json_file(Path(root) / SEMANTIC_CLASSIFIER_SUMMARY)


def load_semantic_classifier_gate(root: Path | None) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_json_file(Path(root) / SEMANTIC_CLASSIFIER_GATE)


def load_gemini_adjudication_summary(root: Path | None) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_json_file(Path(root) / GEMINI_ADJUDICATION_SUMMARY)


def load_gemini_adjudication_gate(root: Path | None) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_json_file(Path(root) / GEMINI_ADJUDICATION_GATE)


def load_gemini_safety_audit(root: Path | None) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_json_file(Path(root) / GEMINI_SAFETY_AUDIT)


def load_manual_review_drilldown_summary(root: Path | None) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_json_file(Path(root) / MANUAL_REVIEW_DRILLDOWN_SUMMARY)


def load_manual_review_drilldown_gate(root: Path | None) -> FileLoadResult:
    if root is None:
        return _missing(None)
    return load_json_file(Path(root) / MANUAL_REVIEW_DRILLDOWN_GATE)
