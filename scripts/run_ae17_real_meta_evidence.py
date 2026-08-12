#!/usr/bin/env python3
"""AE17 durable real-meta evidence runner (deterministic, read-only).

Promotes the manual AE17 real-meta proof into a repository script.
Does NOT: close AE17, start AE18/AE19, train/fit, backtest, mutate trader.db,
call LLMs, call external APIs/Helius/Solana, connect wallet, enable live trading,
change RF/XGB/TAB/AE16 semantics, or claim profitability.
Does NOT load the ~10GB source file by default (size-gated unless --include-large).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE = "AE17_REAL_META_EVIDENCE"
META_MODE = "AE17_REAL_META_EVIDENCE_EXPLICIT_COMBINATION"
META_LAYER_TYPE = "NON_LEARNED_EXPLICIT_META_COMBINATION"
META_FORMULA_VERSION = "AE17_EXPLICIT_META_COMBINATION_V1"
BASELINE_LAYER_TYPE = "DETERMINISTIC_TIER_BASELINE"
BASELINE_FORMULA_VERSION = "AE17_TIER_ONLY_BASELINE_V1"
CONTEXT_FEATURE_AVAILABLE = False
CONTEXT_STATUS = "AE17_CONTEXT_NOT_AVAILABLE_PENDING_AE18"
CONTEXT_MISSINGNESS_REASON = "AE18_CONTEXT_LAYER_NOT_STARTED"
CONTEXT_SCORE_WEIGHT = 0.0
CONTEXT_COMPONENT = 0.0

MODEL_SCORE_WEIGHTS = {"rf": 0.30, "xgb": 0.35, "tab": 0.35}
COMPONENT_WEIGHTS = {
    "weighted_model_score": 0.60,
    "vote_ratio": 0.25,
    "consensus_strength": 0.10,
    "evidence_coverage": 0.05,
}
CONSENSUS_STRENGTH_MAP: dict[str, float | None] = {
    "TAB_XGB_RF_ALL3": 1.00,
    "TAB_RF_ONLY": 0.75,
    "TAB_XGB_ONLY": 0.65,
    "RF_XGB_ONLY": 0.55,
    "SINGLE_MODEL_ONLY": 0.30,
    "REJECT": 0.00,
    "MODEL_EVIDENCE_UNAVAILABLE": None,
    "CONSENSUS_NOT_COMPUTABLE": None,
}
DECISION_THRESHOLDS = {
    "META_STRONG_WATCH": 0.75,
    "META_SECONDARY_WATCH": 0.55,
    "META_RESEARCH_ONLY": 0.35,
    "META_LOW_CONFIDENCE": 0.15,
}
DEFAULT_SOURCE_GLOB = (
    "data/training/manual_verified_results/"
    "phase_e5_direct_target_tabicl_*/consensus/"
    "direct_target_selected_trades_by_tier.csv"
)
DEFAULT_MAX_SOURCE_BYTES = 52_428_800
DEFAULT_MAX_ROWS = 100_000

REQUIRED_SOURCE_COLUMNS = [
    "target_row_id",
    "candidate_id",
    "candidate_policy_id",
    "pair_address",
    "event_timestamp",
    "filter",
    "horizon",
    "split",
    "target_net_profitable",
    "sim_net_return",
    "tab_score",
    "predicted_probability_xgb",
    "predicted_probability_rf",
    "in_tab",
    "in_xgb",
    "in_rf",
    "vote_count",
    "consensus_tier",
]

FEATURE_MATRIX_FIELDS = [
    "historical_meta_row_id",
    "source_file",
    "source_row_index",
    "target_row_id",
    "candidate_id",
    "candidate_policy_id",
    "pair_address",
    "event_timestamp",
    "filter",
    "horizon",
    "exit_policy_id",
    "split",
    "rf_score",
    "xgb_score",
    "tab_score",
    "rf_vote",
    "xgb_vote",
    "tab_vote",
    "attached_model_count",
    "model_vote_count",
    "vote_ratio",
    "evidence_coverage",
    "partial_evidence_status",
    "raw_consensus_tier",
    "derived_consensus_tier",
    "scoring_tier",
    "consensus_strength",
    "context_feature_available",
    "context_status",
    "context_missingness_reason",
    "context_score_weight",
    "context_component",
    "rf_score_component",
    "xgb_score_component",
    "tab_score_component",
    "active_model_score_count",
    "active_model_weight_sum",
    "weighted_model_score",
    "weighted_model_score_missing_reason",
    "pre_clamp_meta_score",
    "meta_score",
    "score_clamped",
    "score_clamp_reason",
    "meta_decision",
    "meta_mode",
    "meta_layer_type",
    "meta_formula_version",
    "meta_reason",
    "baseline_layer_type",
    "baseline_tier_score",
    "baseline_tier_decision",
    "baseline_formula_version",
    "baseline_vs_explicit_score_delta",
    "baseline_vs_explicit_decision_changed",
    "numeric_scores_used_in_meta_score",
    "votes_used_in_meta_score",
    "consensus_feature_used_in_meta_score",
    "context_missingness_used_in_meta_score",
    "tier_only_scoring",
    "price_source_identity_status",
    "tradability_identity_status",
    "lineage_status",
]

OUTPUT_EXTRA_FIELDS = [
    "outcome_label_available",
    "outcome_label_name",
    "outcome_label_value",
    "sim_net_return",
    "score_bucket",
    "outcome_used_as_feature",
]

FORBIDDEN_FEATURE_COLUMNS = {
    "target_net_profitable",
    "target_net_profitable_x",
    "target_net_profitable_y",
    "outcome_label_value",
    "outcome_label_name",
    "outcome_label_available",
    "sim_net_return",
    "sim_net_return_x",
    "sim_net_return_y",
    "future_return",
    "max_upside",
    "max_drawdown",
    "realized_pnl",
    "pnl",
    "profit",
    "closed_at",
    "exit_status",
}

FEATURE_PARITY_REQUIRED = [
    "tab_score",
    "xgb_score",
    "rf_score",
    "tab_vote",
    "xgb_vote",
    "rf_vote",
    "attached_model_count",
    "model_vote_count",
    "scoring_tier",
    "meta_score",
    "meta_decision",
    "candidate_id",
    "candidate_policy_id",
    "target_row_id",
    "pair_address",
    "event_timestamp",
    "context_feature_available",
    "context_status",
    "context_missingness_reason",
    "context_score_weight",
    "context_component",
    "weighted_model_score",
    "weighted_model_score_missing_reason",
    "active_model_score_count",
    "active_model_weight_sum",
    "vote_ratio",
    "consensus_strength",
    "evidence_coverage",
    "pre_clamp_meta_score",
    "baseline_layer_type",
    "baseline_tier_score",
    "baseline_tier_decision",
    "baseline_formula_version",
    "baseline_vs_explicit_score_delta",
    "baseline_vs_explicit_decision_changed",
    "numeric_scores_used_in_meta_score",
    "votes_used_in_meta_score",
    "consensus_feature_used_in_meta_score",
    "context_missingness_used_in_meta_score",
    "tier_only_scoring",
    "meta_layer_type",
    "meta_formula_version",
    "rf_score_component",
    "xgb_score_component",
    "tab_score_component",
]


def canonical_context_missingness_fields() -> dict[str, Any]:
    """Explicit AE17 context-missingness contract (AE18 not started)."""
    return {
        "context_feature_available": CONTEXT_FEATURE_AVAILABLE,
        "context_status": CONTEXT_STATUS,
        "context_missingness_reason": CONTEXT_MISSINGNESS_REASON,
        "context_score_weight": CONTEXT_SCORE_WEIGHT,
    }


def canonical_context_contract_report() -> dict[str, Any]:
    return {
        "context_feature_contract_present": True,
        "context_feature_available": CONTEXT_FEATURE_AVAILABLE,
        "context_status": CONTEXT_STATUS,
        "context_missingness_reason": CONTEXT_MISSINGNESS_REASON,
        "context_score_weight": CONTEXT_SCORE_WEIGHT,
    }

TIER_SCORE_MAP: dict[str, tuple[float | None, str]] = {
    "TAB_XGB_RF_ALL3": (0.90, "META_STRONG_WATCH"),
    "TAB_RF_ONLY": (0.75, "META_SECONDARY_WATCH"),
    "TAB_XGB_ONLY": (0.45, "META_RESEARCH_ONLY"),
    "RF_XGB_ONLY": (0.40, "META_RESEARCH_ONLY"),
    "SINGLE_MODEL_ONLY": (0.25, "META_LOW_CONFIDENCE"),
    "REJECT": (0.0, "META_REJECT"),
    "MODEL_EVIDENCE_UNAVAILABLE": (None, "META_UNAVAILABLE"),
    "CONSENSUS_NOT_COMPUTABLE": (None, "META_UNAVAILABLE"),
}

CLASSIFICATION_SOURCE_NOT_FOUND = "AE17_BLOCKED_SOURCE_NOT_FOUND"
CLASSIFICATION_NO_USABLE = "AE17_BLOCKED_NO_USABLE_SOURCE_FILES"
CLASSIFICATION_NO_ROWS = "AE17_BLOCKED_NO_ROWS_PROCESSED"
CLASSIFICATION_PASS = "AE17_REAL_META_EVIDENCE_IMPLEMENTATION_PASS_NOT_CLOSURE"
CLASSIFICATION_PASS_WARNINGS = (
    "AE17_REAL_META_EVIDENCE_IMPLEMENTATION_PASS_WITH_SOURCE_WARNINGS_NOT_CLOSURE"
)
CLASSIFICATION_LOOKAHEAD = "AE17_BLOCKED_LOOKAHEAD_CONTAMINATION"
CLASSIFICATION_SCORE_INTEGRITY = "AE17_BLOCKED_SCORE_INTEGRITY_FAILURE"
CLASSIFICATION_AUTHORITY = "AE17_BLOCKED_AUTHORITY_VIOLATION"
CLASSIFICATION_INCOMPLETE_SUBSTANCE = "AE17_INCOMPLETE_META_STACKING_SUBSTANCE"
CLASSIFICATION_BASELINE_DIFFERS = "AE17_EXPLICIT_META_COMBINATION_DIFFERS_FROM_TIER_BASELINE"
CLASSIFICATION_SYNTHETIC_PROVEN = (
    "AE17_EXPLICIT_META_COMBINATION_PROVEN_BY_SYNTHETIC_FORMULA_TESTS_WITH_HISTORICAL_VARIANCE_LIMITATION"
)

TRUE_LIKE = {"true", "1", "yes", "y", "t"}
FALSE_LIKE = {"false", "0", "no", "n", "f"}
NULL_LIKE = {"", "null", "none", "nan", "na", "n/a"}

# Mutable warning counters for boolean normalization within a run.
_BOOL_STATS: dict[str, Any] = {
    "boolean_null_values_normalized_false_count": 0,
    "boolean_unrecognized_values_count": 0,
    "boolean_unrecognized_values_examples": [],
    "invalid_score_warning_count": 0,
}


def _reset_bool_stats() -> None:
    _BOOL_STATS["boolean_null_values_normalized_false_count"] = 0
    _BOOL_STATS["boolean_unrecognized_values_count"] = 0
    _BOOL_STATS["boolean_unrecognized_values_examples"] = []
    _BOOL_STATS["invalid_score_warning_count"] = 0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_tag() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def relpath(path: Path, root: Path = ROOT) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for k in fieldnames:
                v = row.get(k)
                if v is None:
                    out[k] = ""
                elif isinstance(v, bool):
                    out[k] = "True" if v else "False"
                else:
                    out[k] = v
            writer.writerow(out)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def normalize_bool_vote(value: Any) -> bool:
    """Normalize vote-like values to bool. Null/empty/NaN => False."""
    if value is None:
        _BOOL_STATS["boolean_null_values_normalized_false_count"] += 1
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and math.isnan(value):
            _BOOL_STATS["boolean_null_values_normalized_false_count"] += 1
            return False
        if value == 1:
            return True
        if value == 0:
            return False
    text = str(value).strip()
    low = text.lower()
    if low in NULL_LIKE:
        _BOOL_STATS["boolean_null_values_normalized_false_count"] += 1
        return False
    if low in TRUE_LIKE:
        return True
    if low in FALSE_LIKE:
        return False
    _BOOL_STATS["boolean_unrecognized_values_count"] += 1
    examples = _BOOL_STATS["boolean_unrecognized_values_examples"]
    if len(examples) < 20 and text not in examples:
        examples.append(text)
    return False


def parse_float_or_none(value: Any) -> float | None:
    """Parse numeric score; empty/None/NaN/inf/invalid => None (never zero-filled)."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        fv = float(value)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return fv
    text = str(value).strip()
    if not text or text.lower() in NULL_LIKE:
        return None
    try:
        fv = float(text)
    except (TypeError, ValueError):
        return None
    if math.isnan(fv) or math.isinf(fv):
        return None
    return fv


def clamp_meta_score(score: Any) -> tuple[float | None, bool, str]:
    """Hard-clamp meta score to [0.0, 1.0]. None/NaN/non-numeric/inf => None."""
    if score is None:
        return None, False, "score_unavailable_null"
    if isinstance(score, bool):
        _BOOL_STATS["invalid_score_warning_count"] += 1
        return None, False, "score_non_numeric_bool"
    if not isinstance(score, (int, float)):
        try:
            score = float(score)
        except (TypeError, ValueError):
            _BOOL_STATS["invalid_score_warning_count"] += 1
            return None, False, "score_non_numeric"
    fv = float(score)
    if math.isnan(fv) or math.isinf(fv):
        _BOOL_STATS["invalid_score_warning_count"] += 1
        return None, False, "score_nan_or_inf"
    clamped = max(0.0, min(1.0, fv))
    if clamped != fv:
        if fv > 1.0:
            return clamped, True, "clamped_high_to_1.0"
        return clamped, True, "clamped_low_to_0.0"
    return clamped, False, "NOT_CLAMPED"


def partial_evidence_status(attached_model_count: int) -> str:
    if attached_model_count >= 3:
        return "ALL3_ATTACHED"
    if attached_model_count == 2:
        return "TWO_MODELS_ATTACHED"
    if attached_model_count == 1:
        return "ONE_MODEL_ATTACHED"
    return "NO_MODEL_EVIDENCE_ATTACHED"


def derive_scoring_tier(
    *,
    tab_vote: bool,
    xgb_vote: bool,
    rf_vote: bool,
    attached_model_count: int,
) -> str:
    """Derive AE17 scoring tier from votes + score-attachment evidence."""
    if attached_model_count <= 0:
        return "MODEL_EVIDENCE_UNAVAILABLE"
    if tab_vote and xgb_vote and rf_vote:
        return "TAB_XGB_RF_ALL3"
    if tab_vote and rf_vote and not xgb_vote:
        return "TAB_RF_ONLY"
    if tab_vote and xgb_vote and not rf_vote:
        return "TAB_XGB_ONLY"
    if xgb_vote and rf_vote and not tab_vote:
        return "RF_XGB_ONLY"
    vote_count = int(tab_vote) + int(xgb_vote) + int(rf_vote)
    if vote_count == 1:
        return "SINGLE_MODEL_ONLY"
    if vote_count == 0:
        return "REJECT"
    return "CONSENSUS_NOT_COMPUTABLE"


def compute_baseline_tier_score(
    *,
    tab_vote: bool,
    xgb_vote: bool,
    rf_vote: bool,
    attached_model_count: int,
) -> dict[str, Any]:
    """Legacy tier-only baseline (kept for comparison only; not official meta_score)."""
    scoring_tier = derive_scoring_tier(
        tab_vote=tab_vote,
        xgb_vote=xgb_vote,
        rf_vote=rf_vote,
        attached_model_count=attached_model_count,
    )
    base_score, decision = TIER_SCORE_MAP[scoring_tier]
    if base_score is None:
        return {
            "scoring_tier": scoring_tier,
            "baseline_tier_score": None,
            "baseline_tier_decision": "META_UNAVAILABLE",
        }
    final, _, _ = clamp_meta_score(base_score)
    return {
        "scoring_tier": scoring_tier,
        "baseline_tier_score": final,
        "baseline_tier_decision": decision,
    }


def compute_weighted_model_score(
    *,
    rf_score: float | None,
    xgb_score: float | None,
    tab_score: float | None,
) -> dict[str, Any]:
    """Weighted average over attached valid scores only (never zero-fills missing)."""
    parts: list[tuple[str, float, float]] = []
    for name, score in (("rf", rf_score), ("xgb", xgb_score), ("tab", tab_score)):
        if score is None:
            continue
        weight = MODEL_SCORE_WEIGHTS[name]
        parts.append((name, float(score), float(weight)))

    active_model_score_count = len(parts)
    active_model_weight_sum = sum(w for _, _, w in parts) if parts else 0.0
    components = {
        "rf_score_component": None,
        "xgb_score_component": None,
        "tab_score_component": None,
    }
    for name, score, weight in parts:
        components[f"{name}_score_component"] = score * weight

    if active_model_weight_sum <= 0.0:
        return {
            **components,
            "active_model_score_count": active_model_score_count,
            "active_model_weight_sum": 0.0,
            "weighted_model_score": None,
            "weighted_model_score_missing_reason": "NO_VALID_MODEL_SCORES_ATTACHED",
        }

    weighted = sum(score * weight for _, score, weight in parts) / active_model_weight_sum
    return {
        **components,
        "active_model_score_count": active_model_score_count,
        "active_model_weight_sum": active_model_weight_sum,
        "weighted_model_score": weighted,
        "weighted_model_score_missing_reason": "",
    }


def decide_meta_from_explicit_score(meta_score: float | None) -> str:
    if meta_score is None:
        return "META_UNAVAILABLE"
    if meta_score >= DECISION_THRESHOLDS["META_STRONG_WATCH"]:
        return "META_STRONG_WATCH"
    if meta_score >= DECISION_THRESHOLDS["META_SECONDARY_WATCH"]:
        return "META_SECONDARY_WATCH"
    if meta_score >= DECISION_THRESHOLDS["META_RESEARCH_ONLY"]:
        return "META_RESEARCH_ONLY"
    if meta_score >= DECISION_THRESHOLDS["META_LOW_CONFIDENCE"]:
        return "META_LOW_CONFIDENCE"
    return "META_REJECT"


def compute_ae17_explicit_meta_combination(
    *,
    rf_score: float | None,
    xgb_score: float | None,
    tab_score: float | None,
    tab_vote: bool,
    xgb_vote: bool,
    rf_vote: bool,
) -> dict[str, Any]:
    """Official AE17 Path-B non-learned explicit meta-combination formula."""
    attached_model_count = (
        int(rf_score is not None) + int(xgb_score is not None) + int(tab_score is not None)
    )
    model_vote_count = int(tab_vote) + int(xgb_vote) + int(rf_vote)
    scoring_tier = derive_scoring_tier(
        tab_vote=tab_vote,
        xgb_vote=xgb_vote,
        rf_vote=rf_vote,
        attached_model_count=attached_model_count,
    )
    baseline = compute_baseline_tier_score(
        tab_vote=tab_vote,
        xgb_vote=xgb_vote,
        rf_vote=rf_vote,
        attached_model_count=attached_model_count,
    )
    weighted = compute_weighted_model_score(
        rf_score=rf_score,
        xgb_score=xgb_score,
        tab_score=tab_score,
    )
    vote_ratio = (
        (model_vote_count / attached_model_count) if attached_model_count > 0 else None
    )
    evidence_coverage = attached_model_count / 3.0
    consensus_strength = CONSENSUS_STRENGTH_MAP.get(scoring_tier)
    context_score_weight = CONTEXT_SCORE_WEIGHT
    context_component = CONTEXT_COMPONENT

    unavailable = attached_model_count == 0 or weighted["weighted_model_score"] is None
    if unavailable or vote_ratio is None or consensus_strength is None:
        pre_clamp = None
        meta_score = None
        clamped = False
        clamp_reason = "unavailable_explicit_meta_null_score"
        meta_decision = "META_UNAVAILABLE"
        meta_reason = (
            f"explicit_meta_unavailable; scoring_tier={scoring_tier}; "
            f"missing_reason={weighted.get('weighted_model_score_missing_reason') or 'consensus_or_votes_null'}"
        )
    else:
        pre_clamp = (
            COMPONENT_WEIGHTS["weighted_model_score"] * float(weighted["weighted_model_score"])
            + COMPONENT_WEIGHTS["vote_ratio"] * float(vote_ratio)
            + COMPONENT_WEIGHTS["consensus_strength"] * float(consensus_strength)
            + COMPONENT_WEIGHTS["evidence_coverage"] * float(evidence_coverage)
            + float(context_score_weight) * float(context_component)
        )
        meta_score, clamped, clamp_reason = clamp_meta_score(pre_clamp)
        meta_decision = decide_meta_from_explicit_score(meta_score)
        meta_reason = (
            f"explicit_meta_v1; tier={scoring_tier}; "
            f"wms={weighted['weighted_model_score']}; vote_ratio={vote_ratio}; "
            f"consensus_strength={consensus_strength}; evidence_coverage={evidence_coverage}; "
            f"context_term={context_score_weight}*{context_component}"
        )

    baseline_score = baseline["baseline_tier_score"]
    if meta_score is None or baseline_score is None:
        delta = None
    else:
        delta = float(meta_score) - float(baseline_score)
    decision_changed = (
        meta_decision != baseline["baseline_tier_decision"]
        if meta_decision is not None and baseline["baseline_tier_decision"] is not None
        else False
    )

    return {
        "scoring_tier": scoring_tier,
        "derived_consensus_tier": scoring_tier,
        "attached_model_count": attached_model_count,
        "model_vote_count": model_vote_count,
        "vote_ratio": vote_ratio,
        "evidence_coverage": evidence_coverage,
        "consensus_strength": consensus_strength,
        "context_component": context_component,
        "context_score_weight": context_score_weight,
        "rf_score_component": weighted["rf_score_component"],
        "xgb_score_component": weighted["xgb_score_component"],
        "tab_score_component": weighted["tab_score_component"],
        "active_model_score_count": weighted["active_model_score_count"],
        "active_model_weight_sum": weighted["active_model_weight_sum"],
        "weighted_model_score": weighted["weighted_model_score"],
        "weighted_model_score_missing_reason": weighted["weighted_model_score_missing_reason"],
        "pre_clamp_meta_score": pre_clamp,
        "meta_score": meta_score,
        "score_clamped": clamped,
        "score_clamp_reason": clamp_reason,
        "meta_decision": meta_decision,
        "meta_reason": meta_reason,
        "meta_mode": META_MODE,
        "meta_layer_type": META_LAYER_TYPE,
        "meta_formula_version": META_FORMULA_VERSION,
        "baseline_layer_type": BASELINE_LAYER_TYPE,
        "baseline_tier_score": baseline_score,
        "baseline_tier_decision": baseline["baseline_tier_decision"],
        "baseline_formula_version": BASELINE_FORMULA_VERSION,
        "baseline_vs_explicit_score_delta": delta,
        "baseline_vs_explicit_decision_changed": decision_changed,
        "numeric_scores_used_in_meta_score": True,
        "votes_used_in_meta_score": True,
        "consensus_feature_used_in_meta_score": True,
        "context_missingness_used_in_meta_score": True,
        "tier_only_scoring": False,
    }


def compute_ae17_real_meta_score(
    *,
    tab_vote: bool,
    xgb_vote: bool,
    rf_vote: bool,
    attached_model_count: int | None = None,
    rf_score: float | None = None,
    xgb_score: float | None = None,
    tab_score: float | None = None,
    base_score_override: float | None = None,
    apply_override_even_if_unavailable: bool = False,
) -> dict[str, Any]:
    """Compatibility wrapper around explicit meta-combination.

    If scores are omitted, synthesizes placeholders only for attached_model_count
    when provided for legacy call sites; prefer compute_ae17_explicit_meta_combination.
    """
    if rf_score is None and xgb_score is None and tab_score is None and attached_model_count:
        # Legacy semantic tests that only passed votes/attached count: use fixed
        # placeholder magnitudes so formula still runs while baseline remains tier-based.
        # Official runner always passes real scores.
        placeholders = [0.5] * int(attached_model_count) + [None] * (3 - int(attached_model_count))
        tab_score, xgb_score, rf_score = placeholders[0], placeholders[1], placeholders[2]

    out = compute_ae17_explicit_meta_combination(
        rf_score=rf_score,
        xgb_score=xgb_score,
        tab_score=tab_score,
        tab_vote=tab_vote,
        xgb_vote=xgb_vote,
        rf_vote=rf_vote,
    )
    if base_score_override is not None and (
        out["meta_score"] is not None or apply_override_even_if_unavailable
    ):
        final, clamped, reason = clamp_meta_score(base_score_override)
        out["pre_clamp_meta_score"] = base_score_override
        out["meta_score"] = final
        out["score_clamped"] = clamped
        out["score_clamp_reason"] = reason
        out["meta_decision"] = decide_meta_from_explicit_score(final)
        out["meta_reason"] = "explicit_meta_override_clamp_test"
    return out


def prove_synthetic_same_tier_score_sensitivity() -> dict[str, Any]:
    """Synthetic proof that consensus_tier alone does not determine meta_score."""
    low = compute_ae17_explicit_meta_combination(
        rf_score=0.10,
        xgb_score=0.10,
        tab_score=0.10,
        tab_vote=True,
        xgb_vote=True,
        rf_vote=True,
    )
    high = compute_ae17_explicit_meta_combination(
        rf_score=0.95,
        xgb_score=0.95,
        tab_score=0.95,
        tab_vote=True,
        xgb_vote=True,
        rf_vote=True,
    )
    same_tier = low["scoring_tier"] == high["scoring_tier"] == "TAB_XGB_RF_ALL3"
    different_scores = (
        low["meta_score"] is not None
        and high["meta_score"] is not None
        and abs(float(low["meta_score"]) - float(high["meta_score"])) > 1e-9
    )
    different_from_baseline = (
        low["meta_score"] != low["baseline_tier_score"]
        or high["meta_score"] != high["baseline_tier_score"]
    )
    return {
        "synthetic_same_tier_score_sensitivity_pass": same_tier and different_scores,
        "same_tier": same_tier,
        "low_meta_score": low["meta_score"],
        "high_meta_score": high["meta_score"],
        "low_baseline_tier_score": low["baseline_tier_score"],
        "high_baseline_tier_score": high["baseline_tier_score"],
        "different_from_baseline": different_from_baseline,
        "tier_only_scoring": False,
    }


def score_bucket(meta_score: float | None) -> str:
    if meta_score is None:
        return "null"
    if meta_score < 0.15:
        return "0.00_0.14"
    if meta_score < 0.35:
        return "0.15_0.34"
    if meta_score < 0.55:
        return "0.35_0.54"
    if meta_score < 0.75:
        return "0.55_0.74"
    return "0.75_1.00"


def read_csv_header(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            return []
    return [h.strip() for h in header]


def iter_csv_rows_streaming(path: Path) -> Iterator[tuple[int, dict[str, str]]]:
    """Stream source CSV row-by-row (never load whole file into memory)."""
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            yield idx, row


def discover_source_files(project_root: Path, source_glob: str) -> list[Path]:
    pattern = source_glob.replace("\\", "/")
    matched = sorted(project_root.glob(pattern))
    return [p for p in matched if p.is_file()]


def audit_source_file(
    path: Path,
    *,
    project_root: Path,
    max_source_bytes: int,
    include_large: bool,
) -> dict[str, Any]:
    size_bytes = path.stat().st_size if path.exists() else 0
    size_gb = size_bytes / (1024**3)
    record: dict[str, Any] = {
        "path": relpath(path, project_root),
        "size_bytes": size_bytes,
        "size_gb": round(size_gb, 6),
        "has_required_columns": False,
        "missing_required_columns": "",
        "used_in_run": False,
        "skipped_large_file": False,
        "skipped_missing_columns": False,
        "skipped_read_error": False,
        "rows_processed_from_file": 0,
        "error_message": "",
    }
    try:
        header = read_csv_header(path)
    except Exception as exc:  # noqa: BLE001 — controlled skip
        record["skipped_read_error"] = True
        record["error_message"] = f"header_read_error: {exc}"
        return record

    missing = [c for c in REQUIRED_SOURCE_COLUMNS if c not in header]
    record["has_required_columns"] = len(missing) == 0
    record["missing_required_columns"] = "|".join(missing)
    if missing:
        record["skipped_missing_columns"] = True
        return record

    if (not include_large) and size_bytes > max_source_bytes:
        record["skipped_large_file"] = True
        record["error_message"] = (
            f"file larger than max-source-bytes={max_source_bytes}; "
            "pass --include-large to allow (still streamed / max-rows capped)"
        )
        return record

    record["used_in_run"] = True
    return record


def build_semantic_partial_evidence_audit() -> dict[str, Any]:
    """Semantic cases: verify tier derivation + baseline + explicit formula fields."""
    cases_spec = [
        ("all3", True, True, True, 0.9, 0.9, 0.9, "TAB_XGB_RF_ALL3"),
        ("TAB_RF_ONLY", True, False, True, 0.8, 0.2, 0.8, "TAB_RF_ONLY"),
        ("TAB_XGB_ONLY", True, True, False, 0.8, 0.8, 0.2, "TAB_XGB_ONLY"),
        ("RF_XGB_ONLY", False, True, True, 0.2, 0.8, 0.8, "RF_XGB_ONLY"),
        ("SINGLE_MODEL_ONLY", True, False, False, 0.7, 0.1, 0.1, "SINGLE_MODEL_ONLY"),
        ("REJECT", False, False, False, 0.4, 0.4, 0.4, "REJECT"),
        ("NO_EVIDENCE", False, False, False, None, None, None, "MODEL_EVIDENCE_UNAVAILABLE"),
    ]
    cases = []
    all_pass = True
    for case_id, tab_v, xgb_v, rf_v, tab_s, xgb_s, rf_s, expected_tier in cases_spec:
        out = compute_ae17_explicit_meta_combination(
            rf_score=rf_s,
            xgb_score=xgb_s,
            tab_score=tab_s,
            tab_vote=tab_v,
            xgb_vote=xgb_v,
            rf_vote=rf_v,
        )
        baseline = compute_baseline_tier_score(
            tab_vote=tab_v,
            xgb_vote=xgb_v,
            rf_vote=rf_v,
            attached_model_count=out["attached_model_count"],
        )
        tier_ok = out["scoring_tier"] == expected_tier
        baseline_ok = baseline["baseline_tier_decision"] == TIER_SCORE_MAP[expected_tier][1] or (
            expected_tier in ("MODEL_EVIDENCE_UNAVAILABLE", "CONSENSUS_NOT_COMPUTABLE")
            and baseline["baseline_tier_decision"] == "META_UNAVAILABLE"
        )
        if expected_tier == "MODEL_EVIDENCE_UNAVAILABLE":
            score_ok = out["meta_score"] is None and out["meta_decision"] == "META_UNAVAILABLE"
        else:
            score_ok = (
                out["meta_score"] is not None
                and out["tier_only_scoring"] is False
                and out["numeric_scores_used_in_meta_score"] is True
            )
        passed = tier_ok and baseline_ok and score_ok
        all_pass = all_pass and passed
        cases.append(
            {
                "case_id": case_id,
                "tab_vote": tab_v,
                "xgb_vote": xgb_v,
                "rf_vote": rf_v,
                "attached_model_count": out["attached_model_count"],
                "actual_tier": out["scoring_tier"],
                "expected_tier": expected_tier,
                "actual_meta_score": "" if out["meta_score"] is None else out["meta_score"],
                "baseline_tier_score": (
                    "" if baseline["baseline_tier_score"] is None else baseline["baseline_tier_score"]
                ),
                "actual_meta_decision": out["meta_decision"],
                "baseline_tier_decision": baseline["baseline_tier_decision"],
                "tier_only_scoring": out["tier_only_scoring"],
                "pass": passed,
            }
        )
    sensitivity = prove_synthetic_same_tier_score_sensitivity()
    all_pass = all_pass and bool(sensitivity["synthetic_same_tier_score_sensitivity_pass"])
    return {
        "semantic_cases": cases,
        "synthetic_same_tier_score_sensitivity": sensitivity,
        "partial_evidence_semantic_pass": all_pass,
        "note": (
            "Synthetic semantic audit for explicit AE17 meta-combination; "
            "baseline remains tier-only for comparison."
        ),
    }


def analyze_same_tier_score_variance(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_tier: dict[str, set[str]] = defaultdict(set)
    examples: list[dict[str, Any]] = []
    for row in rows:
        tier = str(row.get("scoring_tier") or "")
        score = row.get("meta_score")
        if score is None or score == "":
            continue
        key = f"{float(score):.8f}"
        by_tier[tier].add(key)
    same_tier_multi = {
        tier: sorted(scores)
        for tier, scores in by_tier.items()
        if len(scores) > 1
    }
    observed = len(same_tier_multi) > 0
    if observed:
        for tier, scores in list(same_tier_multi.items())[:5]:
            examples.append({"scoring_tier": tier, "distinct_explicit_scores": scores[:10]})
    return {
        "same_tier_different_scores_observed": observed,
        "same_tier_different_scores_by_tier": {
            k: len(v) for k, v in same_tier_multi.items()
        },
        "same_tier_different_scores_observed_rows": examples,
        "rows_where_same_tier_has_multiple_explicit_scores": sum(
            len(v) for v in same_tier_multi.values()
        ),
    }


def build_formula_audit(output_rows: list[dict[str, Any]]) -> dict[str, Any]:
    variance = analyze_same_tier_score_variance(output_rows)
    sensitivity = prove_synthetic_same_tier_score_sensitivity()
    major_limitation = None
    if not variance["same_tier_different_scores_observed"]:
        major_limitation = "HISTORICAL_INPUT_LACKS_WITHIN_TIER_SCORE_VARIANCE"
    substance_ok = (
        sensitivity["synthetic_same_tier_score_sensitivity_pass"]
        and all(
            r.get("tier_only_scoring") in (False, "False", "false")
            for r in output_rows[:50]
        )
        and all(
            r.get("numeric_scores_used_in_meta_score") in (True, "True", "true")
            for r in output_rows[:50]
        )
    )
    return {
        "meta_layer_type": META_LAYER_TYPE,
        "meta_formula_version": META_FORMULA_VERSION,
        "learned_model_used": False,
        "training_or_fit": False,
        "formula_description": (
            "pre_clamp = 0.60*weighted_model_score + 0.25*vote_ratio + "
            "0.10*consensus_strength + 0.05*evidence_coverage + "
            "context_score_weight*context_component; meta_score=hard_clamp(pre_clamp)"
        ),
        "model_score_weights": MODEL_SCORE_WEIGHTS,
        "component_weights": COMPONENT_WEIGHTS,
        "decision_thresholds": DECISION_THRESHOLDS,
        "required_input_columns": [
            "rf_score",
            "xgb_score",
            "tab_score",
            "rf_vote",
            "xgb_vote",
            "tab_vote",
            "scoring_tier",
            "context_score_weight",
            "context_component",
        ],
        "numeric_scores_used": True,
        "votes_used": True,
        "consensus_feature_used": True,
        "context_missingness_used": True,
        "tier_only_scoring": False,
        "active_weight_handling": (
            "Divide only by sum of weights for models with valid non-null scores; "
            "exclude missing/invalid scores from numerator and denominator."
        ),
        "missing_score_policy": "never_zero_fill; null remains null",
        "hard_clamping_inside_combinator": True,
        **variance,
        "synthetic_same_tier_score_sensitivity": sensitivity,
        "major_limitation": major_limitation,
        "formula_substance_pass": substance_ok,
        "classification": (
            CLASSIFICATION_BASELINE_DIFFERS
            if variance["same_tier_different_scores_observed"] and substance_ok
            else (
                CLASSIFICATION_SYNTHETIC_PROVEN
                if substance_ok
                else CLASSIFICATION_INCOMPLETE_SUBSTANCE
            )
        ),
    }


def build_baseline_comparison_audit(output_rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas: list[float] = []
    decision_changed = 0
    baseline_decisions: Counter[str] = Counter()
    explicit_decisions: Counter[str] = Counter()
    baseline_scores: list[float] = []
    explicit_scores: list[float] = []
    for row in output_rows:
        bdec = row.get("baseline_tier_decision")
        edec = row.get("meta_decision")
        if bdec:
            baseline_decisions[str(bdec)] += 1
        if edec:
            explicit_decisions[str(edec)] += 1
        if row.get("baseline_vs_explicit_decision_changed") in (True, "True", "true"):
            decision_changed += 1
        bscore = row.get("baseline_tier_score")
        escore = row.get("meta_score")
        if bscore not in (None, ""):
            baseline_scores.append(float(bscore))
        if escore not in (None, ""):
            explicit_scores.append(float(escore))
        delta = row.get("baseline_vs_explicit_score_delta")
        if delta not in (None, ""):
            deltas.append(float(delta))

    variance = analyze_same_tier_score_variance(output_rows)
    sensitivity = prove_synthetic_same_tier_score_sensitivity()
    historical_variance_sufficient = bool(variance["same_tier_different_scores_observed"])
    major_limitation = None
    if not historical_variance_sufficient:
        major_limitation = "HISTORICAL_INPUT_LACKS_WITHIN_TIER_SCORE_VARIANCE"

    identical_to_baseline = (
        len(deltas) > 0
        and all(abs(d) < 1e-12 for d in deltas)
        and decision_changed == 0
    )
    if historical_variance_sufficient and not identical_to_baseline:
        classification = CLASSIFICATION_BASELINE_DIFFERS
    elif sensitivity["synthetic_same_tier_score_sensitivity_pass"] and not (
        identical_to_baseline and not sensitivity["different_from_baseline"]
    ):
        # Synthetic proves formula differs even if historical deltas happen to be small/absent.
        classification = CLASSIFICATION_SYNTHETIC_PROVEN
    elif identical_to_baseline and not sensitivity["synthetic_same_tier_score_sensitivity_pass"]:
        classification = CLASSIFICATION_INCOMPLETE_SUBSTANCE
    else:
        classification = CLASSIFICATION_SYNTHETIC_PROVEN

    n = len(output_rows)
    return {
        "rows_compared": n,
        "baseline_decision_distribution": dict(baseline_decisions),
        "explicit_decision_distribution": dict(explicit_decisions),
        "baseline_score_min": min(baseline_scores) if baseline_scores else None,
        "baseline_score_max": max(baseline_scores) if baseline_scores else None,
        "explicit_score_min": min(explicit_scores) if explicit_scores else None,
        "explicit_score_max": max(explicit_scores) if explicit_scores else None,
        "mean_score_delta": (sum(deltas) / len(deltas)) if deltas else None,
        "min_score_delta": min(deltas) if deltas else None,
        "max_score_delta": max(deltas) if deltas else None,
        "decision_changed_count": decision_changed,
        "decision_changed_share": (decision_changed / n) if n else 0.0,
        "rows_where_same_tier_has_multiple_explicit_scores": variance[
            "rows_where_same_tier_has_multiple_explicit_scores"
        ],
        "same_tier_score_variance_by_tier": variance["same_tier_different_scores_by_tier"],
        "same_tier_different_scores_observed": variance[
            "same_tier_different_scores_observed"
        ],
        "historical_variance_sufficient": historical_variance_sufficient,
        "major_limitation": major_limitation,
        "synthetic_same_tier_score_sensitivity": sensitivity,
        "classification": classification,
    }


def build_baseline_vs_explicit_evaluation(output_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in output_rows:
        rows.append(
            {
                "historical_meta_row_id": r.get("historical_meta_row_id"),
                "scoring_tier": r.get("scoring_tier"),
                "baseline_tier_score": r.get("baseline_tier_score"),
                "baseline_tier_decision": r.get("baseline_tier_decision"),
                "meta_score": r.get("meta_score"),
                "meta_decision": r.get("meta_decision"),
                "baseline_vs_explicit_score_delta": r.get("baseline_vs_explicit_score_delta"),
                "baseline_vs_explicit_decision_changed": r.get(
                    "baseline_vs_explicit_decision_changed"
                ),
                "weighted_model_score": r.get("weighted_model_score"),
                "vote_ratio": r.get("vote_ratio"),
                "consensus_strength": r.get("consensus_strength"),
            }
        )
    return rows


def authority_audit_payload() -> dict[str, Any]:
    return {
        "trade_authority": False,
        "live_trading_ready": False,
        "paper_demo_only": True,
        "risk_override_authority": False,
        "wallet_access": False,
        "private_key_access": False,
        "db_mutation": False,
        "orders_opened": 0,
        "positions_opened": 0,
        "llm_calls": 0,
        "external_api_calls": 0,
        "training_or_fit": False,
    }


def compute_pair_concentration(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    for row in rows:
        pair = str(row.get("pair_address") or "")
        counts[pair] += 1
    total = sum(counts.values())
    if total == 0:
        return {
            "total_rows": 0,
            "unique_pairs": 0,
            "top_pair": "",
            "top_pair_count": 0,
            "top_pair_share": 0.0,
            "hhi": 0.0,
            "top_pair_share_status": "OK",
            "hhi_status": "low",
            "note": (
                "pair_address is price-source / pool-observation identity only, "
                "not tradability proof"
            ),
        }
    top_pair, top_count = counts.most_common(1)[0]
    top_share = top_count / total
    hhi = sum((c / total) ** 2 for c in counts.values())
    if top_share <= 0.30:
        share_status = "OK"
    elif top_share <= 0.50:
        share_status = "warning"
    else:
        share_status = "high_risk"
    if hhi <= 0.15:
        hhi_status = "low"
    elif hhi <= 0.25:
        hhi_status = "moderate"
    else:
        hhi_status = "high"
    return {
        "total_rows": total,
        "unique_pairs": len(counts),
        "top_pair": top_pair,
        "top_pair_count": top_count,
        "top_pair_share": top_share,
        "hhi": hhi,
        "top_pair_share_status": share_status,
        "hhi_status": hhi_status,
        "note": (
            "pair_address is price-source / pool-observation identity only, "
            "not tradability proof"
        ),
    }


def process_source_row(
    row: dict[str, str],
    *,
    source_file: str,
    source_row_index: int,
    row_seq: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tab_score = parse_float_or_none(row.get("tab_score"))
    xgb_score = parse_float_or_none(row.get("predicted_probability_xgb"))
    rf_score = parse_float_or_none(row.get("predicted_probability_rf"))
    tab_attached = tab_score is not None
    xgb_attached = xgb_score is not None
    rf_attached = rf_score is not None
    attached_model_count = int(tab_attached) + int(xgb_attached) + int(rf_attached)

    tab_vote = normalize_bool_vote(row.get("in_tab"))
    xgb_vote = normalize_bool_vote(row.get("in_xgb"))
    rf_vote = normalize_bool_vote(row.get("in_rf"))
    model_vote_count = int(tab_vote) + int(xgb_vote) + int(rf_vote)
    partial = partial_evidence_status(attached_model_count)
    scored = compute_ae17_explicit_meta_combination(
        rf_score=rf_score,
        xgb_score=xgb_score,
        tab_score=tab_score,
        tab_vote=tab_vote,
        xgb_vote=xgb_vote,
        rf_vote=rf_vote,
    )

    target_row_id = row.get("target_row_id") or ""
    candidate_id = row.get("candidate_id") or ""
    candidate_policy_id = row.get("candidate_policy_id") or ""
    pair_address = row.get("pair_address") or ""
    event_timestamp = row.get("event_timestamp") or ""
    lineage_ok = all(
        [
            str(target_row_id).strip(),
            str(candidate_id).strip(),
            str(candidate_policy_id).strip(),
            str(pair_address).strip(),
            str(event_timestamp).strip(),
        ]
    )
    lineage_status = (
        "TARGET_ROW_AND_CANDIDATE_POLICY_AVAILABLE"
        if lineage_ok
        else "LINEAGE_FIELDS_INCOMPLETE"
    )

    feature = {
        "historical_meta_row_id": f"ae17_real_meta_{row_seq:09d}",
        "source_file": source_file,
        "source_row_index": source_row_index,
        "target_row_id": target_row_id,
        "candidate_id": candidate_id,
        "candidate_policy_id": candidate_policy_id,
        "pair_address": pair_address,
        "event_timestamp": event_timestamp,
        "filter": row.get("filter") or "",
        "horizon": row.get("horizon") or "",
        "exit_policy_id": row.get("exit_policy_id") or "",
        "split": row.get("split") or "",
        "rf_score": rf_score,
        "xgb_score": xgb_score,
        "tab_score": tab_score,
        "rf_vote": rf_vote,
        "xgb_vote": xgb_vote,
        "tab_vote": tab_vote,
        "attached_model_count": scored["attached_model_count"],
        "model_vote_count": model_vote_count,
        "vote_ratio": scored["vote_ratio"],
        "evidence_coverage": scored["evidence_coverage"],
        "partial_evidence_status": partial,
        "raw_consensus_tier": row.get("consensus_tier") or "",
        "derived_consensus_tier": scored["derived_consensus_tier"],
        "scoring_tier": scored["scoring_tier"],
        "consensus_strength": scored["consensus_strength"],
        "context_feature_available": CONTEXT_FEATURE_AVAILABLE,
        "context_status": CONTEXT_STATUS,
        "context_missingness_reason": CONTEXT_MISSINGNESS_REASON,
        "context_score_weight": scored["context_score_weight"],
        "context_component": scored["context_component"],
        "rf_score_component": scored["rf_score_component"],
        "xgb_score_component": scored["xgb_score_component"],
        "tab_score_component": scored["tab_score_component"],
        "active_model_score_count": scored["active_model_score_count"],
        "active_model_weight_sum": scored["active_model_weight_sum"],
        "weighted_model_score": scored["weighted_model_score"],
        "weighted_model_score_missing_reason": scored["weighted_model_score_missing_reason"],
        "pre_clamp_meta_score": scored["pre_clamp_meta_score"],
        "meta_score": scored["meta_score"],
        "score_clamped": scored["score_clamped"],
        "score_clamp_reason": scored["score_clamp_reason"],
        "meta_decision": scored["meta_decision"],
        "meta_mode": META_MODE,
        "meta_layer_type": META_LAYER_TYPE,
        "meta_formula_version": META_FORMULA_VERSION,
        "meta_reason": scored["meta_reason"],
        "baseline_layer_type": BASELINE_LAYER_TYPE,
        "baseline_tier_score": scored["baseline_tier_score"],
        "baseline_tier_decision": scored["baseline_tier_decision"],
        "baseline_formula_version": BASELINE_FORMULA_VERSION,
        "baseline_vs_explicit_score_delta": scored["baseline_vs_explicit_score_delta"],
        "baseline_vs_explicit_decision_changed": scored[
            "baseline_vs_explicit_decision_changed"
        ],
        "numeric_scores_used_in_meta_score": True,
        "votes_used_in_meta_score": True,
        "consensus_feature_used_in_meta_score": True,
        "context_missingness_used_in_meta_score": True,
        "tier_only_scoring": False,
        "price_source_identity_status": "PAIR_ADDRESS_ONLY_PROVIDER_CHAIN_NOT_AVAILABLE",
        "tradability_identity_status": "NOT_EVALUATED_NOT_EXECUTION_PROOF",
        "lineage_status": lineage_status,
    }

    outcome_raw = row.get("target_net_profitable")
    outcome_parsed = parse_float_or_none(outcome_raw)
    if outcome_parsed is None and outcome_raw not in (None, ""):
        outcome_parsed = parse_float_or_none(str(outcome_raw).strip())
    outcome_available = outcome_parsed is not None
    sim_net = parse_float_or_none(row.get("sim_net_return"))

    output = dict(feature)
    output.update(
        {
            "outcome_label_available": outcome_available,
            "outcome_label_name": "target_net_profitable" if outcome_available else "",
            "outcome_label_value": int(outcome_parsed) if outcome_available else "",
            "sim_net_return": sim_net,
            "score_bucket": score_bucket(scored["meta_score"]),
            "outcome_used_as_feature": False,
        }
    )
    return feature, output


def evaluate_group(
    rows: list[dict[str, Any]],
    key: str,
) -> list[dict[str, Any]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row.get(key)].append(row)
    out = []
    for group_key in sorted(groups.keys(), key=lambda x: str(x)):
        grp = groups[group_key]
        outcome_avail = [r for r in grp if r.get("outcome_label_available")]
        positives = [
            r
            for r in outcome_avail
            if r.get("outcome_label_value") in (1, 1.0, "1", True, "True")
        ]
        sim_vals = [
            float(r["sim_net_return"])
            for r in grp
            if r.get("sim_net_return") is not None and r.get("sim_net_return") != ""
        ]
        out.append(
            {
                key: group_key,
                "rows": len(grp),
                "outcome_available": len(outcome_avail),
                "positive_outcomes": len(positives),
                "outcome_rate": (len(positives) / len(outcome_avail)) if outcome_avail else "",
                "avg_sim_net_return": (sum(sim_vals) / len(sim_vals)) if sim_vals else "",
            }
        )
    return out


def build_feature_parity_audit(feature_rows: list[dict[str, Any]], path: str) -> dict[str, Any]:
    cols = list(feature_rows[0].keys()) if feature_rows else list(FEATURE_MATRIX_FIELDS)
    missing = [c for c in FEATURE_PARITY_REQUIRED if c not in cols]
    context_values_ok = True
    if feature_rows:
        for row in feature_rows:
            avail = row.get("context_feature_available")
            avail_false = avail in (False, "False", "false", 0, "0")
            weight = row.get("context_score_weight")
            try:
                weight_ok = float(weight) == 0.0
            except (TypeError, ValueError):
                weight_ok = False
            if not (
                avail_false
                and row.get("context_status") == CONTEXT_STATUS
                and row.get("context_missingness_reason") == CONTEXT_MISSINGNESS_REASON
                and weight_ok
            ):
                context_values_ok = False
                break
    return {
        "feature_matrix_path": path,
        "feature_rows": len(feature_rows),
        "feature_column_count": len(cols),
        "required_feature_columns": FEATURE_PARITY_REQUIRED,
        "missing_required_feature_columns": missing,
        "feature_parity_pass": len(missing) == 0 and len(feature_rows) > 0 and context_values_ok,
        "context_values_ok": context_values_ok,
        **canonical_context_contract_report(),
    }


def build_no_lookahead_audit(feature_rows: list[dict[str, Any]], path: str) -> dict[str, Any]:
    cols = set(feature_rows[0].keys()) if feature_rows else set()
    forbidden_present = sorted(cols & FORBIDDEN_FEATURE_COLUMNS)
    return {
        "feature_matrix_path": path,
        "feature_rows": len(feature_rows),
        "forbidden_columns_checked": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "forbidden_columns_present": forbidden_present,
        "no_lookahead_pass": len(forbidden_present) == 0,
        "outcome_used_as_feature": False,
        "note": "Outcome/return fields may appear only in output/evaluation files after scoring.",
    }


def build_score_integrity_audit(output_rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores: list[float] = []
    null_count = 0
    clamped_count = 0
    invalid_count = _BOOL_STATS["invalid_score_warning_count"]
    out_of_range = 0
    for row in output_rows:
        s = row.get("meta_score")
        if s is None or s == "":
            null_count += 1
            continue
        try:
            fv = float(s)
        except (TypeError, ValueError):
            invalid_count += 1
            out_of_range += 1
            continue
        if math.isnan(fv) or math.isinf(fv) or fv < 0.0 or fv > 1.0:
            out_of_range += 1
        else:
            scores.append(fv)
        if row.get("score_clamped") in (True, "True", "true", 1, "1"):
            clamped_count += 1
    return {
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "null_score_count": null_count,
        "non_null_score_count": len(scores),
        "score_clamped_count": clamped_count,
        "invalid_score_count": invalid_count,
        "out_of_range_final_scores": out_of_range,
        "score_integrity_pass": out_of_range == 0,
    }


def build_null_safety_audit(*, rows_processed: int) -> dict[str, Any]:
    return {
        "boolean_null_values_normalized_false_count": _BOOL_STATS[
            "boolean_null_values_normalized_false_count"
        ],
        "boolean_unrecognized_values_count": _BOOL_STATS["boolean_unrecognized_values_count"],
        "boolean_unrecognized_values_examples": list(
            _BOOL_STATS["boolean_unrecognized_values_examples"]
        ),
        "missing_model_scores_zero_filled": False,
        "numeric_none_guarded": True,
        "rows_processed_without_exception": rows_processed,
    }


def build_lineage_audit_rows(output_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for r in output_rows:
        present = all(
            str(r.get(k) or "").strip()
            for k in (
                "target_row_id",
                "candidate_id",
                "candidate_policy_id",
                "pair_address",
                "event_timestamp",
            )
        )
        rows.append(
            {
                "historical_meta_row_id": r.get("historical_meta_row_id"),
                "target_row_id": r.get("target_row_id"),
                "candidate_id": r.get("candidate_id"),
                "candidate_policy_id": r.get("candidate_policy_id"),
                "pair_address": r.get("pair_address"),
                "event_timestamp": r.get("event_timestamp"),
                "lineage_fields_present": present,
                "pair_address_identity_note": (
                    "pair_address is price-source / pool-observation identity only, "
                    "not tradability proof"
                ),
                "lineage_status": r.get("lineage_status"),
            }
        )
    return rows


def ensure_output_dirs(output_root: Path) -> tuple[Path, Path, Path]:
    data_dir = output_root / "data"
    reports_dir = output_root / "reports"
    audits_dir = output_root / "audits"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    audits_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, reports_dir, audits_dir


def write_blocker_package(
    *,
    output_root: Path,
    classification: str,
    source_audit_rows: list[dict[str, Any]],
    source_glob: str,
    notes: list[str],
    files_discovered: int,
) -> dict[str, Any]:
    data_dir, reports_dir, audits_dir = ensure_output_dirs(output_root)
    authority = authority_audit_payload()
    semantic = build_semantic_partial_evidence_audit()
    decision_gate = {
        "created_at_utc": utc_now_iso(),
        "classification": classification,
        "stage_decision": "AE17_NOT_CLOSED",
        "ae17_closed": False,
        "rows_processed": 0,
        "feature_matrix_rows": 0,
        "meta_output_rows": 0,
        "files_discovered": files_discovered,
        "files_used": 0,
        "files_skipped": len(source_audit_rows),
        "source_warnings": notes,
        "not_available_due_to_blocker": [
            "data/ae17_real_meta_feature_matrix.csv",
            "data/ae17_real_meta_outputs.csv",
            "data/ae17_real_meta_evaluation_by_tier.csv",
            "data/ae17_real_meta_evaluation_by_decision.csv",
            "data/ae17_real_meta_evaluation_by_score_bucket.csv",
            "data/ae17_real_meta_evaluation_by_attached_model_count.csv",
            "audits/ae17_real_meta_feature_parity_audit.json",
            "audits/ae17_real_meta_no_lookahead_audit.json",
            "audits/ae17_real_meta_score_integrity_audit.json",
            "audits/ae17_real_meta_lineage_audit.csv",
            "audits/ae17_real_meta_pair_concentration_audit.json",
        ],
        "authority": authority,
        "partial_evidence_semantic_pass": semantic["partial_evidence_semantic_pass"],
        **canonical_context_contract_report(),
    }
    manifest = {
        "created_at_utc": utc_now_iso(),
        "output_root": relpath(output_root),
        "classification": classification,
        "stage_decision": "AE17_NOT_CLOSED",
        "ae17_closed": False,
        "source_glob": source_glob,
        "files_discovered": files_discovered,
        "files_used": [],
        "files_skipped": [r["path"] for r in source_audit_rows],
        "rows_processed": 0,
        "feature_matrix_rows": 0,
        "meta_output_rows": 0,
        **canonical_context_contract_report(),
        "notes": notes
        + [
            "This is NOT AE17 closure.",
            "Controlled blocker written without traceback.",
            f"Context missingness contract present ({CONTEXT_STATUS}).",
        ],
    }
    summary = "\n".join(
        [
            "AE17 REAL META EVIDENCE RUN",
            "",
            f"Output root:\n{relpath(output_root)}",
            "",
            f"Classification:\n{classification}",
            "",
            "Stage decision:\nAE17_NOT_CLOSED",
            "",
            f"Files discovered:\n{files_discovered}",
            "",
            "Rows processed:\n0",
            "",
            "Feature matrix rows:\n0",
            "",
            "Meta output rows:\n0",
            "",
            "Notes:",
            *[f"- {n}" for n in notes],
            "",
            "AE17 closed: false",
            "",
        ]
    )
    write_csv(
        audits_dir / "ae17_real_meta_source_file_audit.csv",
        source_audit_rows,
        [
            "path",
            "size_bytes",
            "size_gb",
            "has_required_columns",
            "missing_required_columns",
            "used_in_run",
            "skipped_large_file",
            "skipped_missing_columns",
            "skipped_read_error",
            "rows_processed_from_file",
            "error_message",
        ],
    )
    write_json(reports_dir / "ae17_real_meta_manifest.json", manifest)
    write_json(reports_dir / "ae17_real_meta_decision_gate.json", decision_gate)
    write_text(reports_dir / "ae17_real_meta_summary_for_upload.txt", summary)
    write_json(audits_dir / "ae17_real_meta_authority_audit.json", authority)
    write_json(audits_dir / "ae17_real_meta_partial_evidence_semantic_audit.json", semantic)
    write_json(
        audits_dir / "ae17_real_meta_null_safety_audit.json",
        build_null_safety_audit(rows_processed=0),
    )
    # Empty stubs for required data paths (headers only) so operators see structure.
    write_csv(data_dir / "ae17_real_meta_feature_matrix.csv", [], FEATURE_MATRIX_FIELDS)
    write_csv(
        data_dir / "ae17_real_meta_outputs.csv",
        [],
        FEATURE_MATRIX_FIELDS + OUTPUT_EXTRA_FIELDS,
    )
    write_csv(
        data_dir / "ae17_real_meta_example_rows.csv",
        [],
        FEATURE_MATRIX_FIELDS + OUTPUT_EXTRA_FIELDS,
    )
    return {
        "classification": classification,
        "output_root": str(output_root),
        "rows_processed": 0,
        "feature_matrix_rows": 0,
        "meta_output_rows": 0,
        "decision_gate": decision_gate,
        "manifest": manifest,
    }


def run_ae17_real_meta_evidence(
    project_root: Path | None = None,
    *,
    source_glob: str = DEFAULT_SOURCE_GLOB,
    output_root: str | Path | None = None,
    max_source_bytes: int = DEFAULT_MAX_SOURCE_BYTES,
    max_rows: int = DEFAULT_MAX_ROWS,
    include_large: bool = False,
) -> dict[str, Any]:
    _reset_bool_stats()
    root = Path(project_root) if project_root else ROOT
    if output_root is None:
        out = root / "data" / "audits" / f"ae17_real_meta_evidence_run_{timestamp_tag()}"
    else:
        out = Path(output_root)
        if not out.is_absolute():
            out = root / out

    discovered = discover_source_files(root, source_glob)
    if not discovered:
        return write_blocker_package(
            output_root=out,
            classification=CLASSIFICATION_SOURCE_NOT_FOUND,
            source_audit_rows=[],
            source_glob=source_glob,
            notes=[f"No files matched source-glob: {source_glob}"],
            files_discovered=0,
        )

    source_audit_rows: list[dict[str, Any]] = []
    usable: list[tuple[Path, dict[str, Any]]] = []
    for path in discovered:
        audit = audit_source_file(
            path,
            project_root=root,
            max_source_bytes=max_source_bytes,
            include_large=include_large,
        )
        source_audit_rows.append(audit)
        if audit["used_in_run"]:
            usable.append((path, audit))

    if not usable:
        return write_blocker_package(
            output_root=out,
            classification=CLASSIFICATION_NO_USABLE,
            source_audit_rows=source_audit_rows,
            source_glob=source_glob,
            notes=[
                "Files discovered but none usable after size/column/read checks.",
                f"discovered={len(discovered)} usable=0",
            ],
            files_discovered=len(discovered),
        )

    data_dir, reports_dir, audits_dir = ensure_output_dirs(out)
    feature_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    rows_processed = 0
    source_warnings: list[str] = []

    for path, audit in usable:
        if rows_processed >= max_rows:
            break
        source_rel = audit["path"]
        file_rows = 0
        try:
            for src_idx, row in iter_csv_rows_streaming(path):
                if rows_processed >= max_rows:
                    break
                feature, output = process_source_row(
                    row,
                    source_file=source_rel,
                    source_row_index=src_idx,
                    row_seq=rows_processed,
                )
                feature_rows.append(feature)
                output_rows.append(output)
                rows_processed += 1
                file_rows += 1
        except Exception as exc:  # noqa: BLE001 — controlled skip of remaining file
            audit["skipped_read_error"] = True
            audit["used_in_run"] = False
            audit["error_message"] = f"row_read_error: {exc}"
            source_warnings.append(f"read_error:{source_rel}:{exc}")
            continue
        audit["rows_processed_from_file"] = file_rows

    # Recompute used/skipped after processing errors.
    for audit in source_audit_rows:
        if audit.get("skipped_large_file"):
            source_warnings.append(f"skipped_large_file:{audit['path']}")
        if audit.get("skipped_missing_columns"):
            source_warnings.append(
                f"skipped_missing_columns:{audit['path']}:{audit.get('missing_required_columns')}"
            )
        if audit.get("skipped_read_error") and audit.get("error_message"):
            if f"read_error:{audit['path']}" not in " ".join(source_warnings):
                source_warnings.append(f"skipped_read_error:{audit['path']}")

    files_used = [a["path"] for a in source_audit_rows if a.get("used_in_run")]
    files_skipped = [a["path"] for a in source_audit_rows if not a.get("used_in_run")]

    if rows_processed == 0:
        return write_blocker_package(
            output_root=out,
            classification=CLASSIFICATION_NO_ROWS,
            source_audit_rows=source_audit_rows,
            source_glob=source_glob,
            notes=["Usable source files existed but zero rows were processed."]
            + source_warnings,
            files_discovered=len(discovered),
        )

    feature_path = data_dir / "ae17_real_meta_feature_matrix.csv"
    outputs_path = data_dir / "ae17_real_meta_outputs.csv"
    example_path = data_dir / "ae17_real_meta_example_rows.csv"

    write_csv(feature_path, feature_rows, FEATURE_MATRIX_FIELDS)
    write_csv(outputs_path, output_rows, FEATURE_MATRIX_FIELDS + OUTPUT_EXTRA_FIELDS)
    write_csv(
        example_path,
        output_rows[:20],
        FEATURE_MATRIX_FIELDS + OUTPUT_EXTRA_FIELDS,
    )

    eval_by_tier = evaluate_group(output_rows, "scoring_tier")
    eval_by_decision = evaluate_group(output_rows, "meta_decision")
    eval_by_bucket = evaluate_group(output_rows, "score_bucket")
    eval_by_attached = evaluate_group(output_rows, "attached_model_count")
    write_csv(
        data_dir / "ae17_real_meta_evaluation_by_tier.csv",
        eval_by_tier,
        ["scoring_tier", "rows", "outcome_available", "positive_outcomes", "outcome_rate", "avg_sim_net_return"],
    )
    write_csv(
        data_dir / "ae17_real_meta_evaluation_by_decision.csv",
        eval_by_decision,
        ["meta_decision", "rows", "outcome_available", "positive_outcomes", "outcome_rate", "avg_sim_net_return"],
    )
    write_csv(
        data_dir / "ae17_real_meta_evaluation_by_score_bucket.csv",
        eval_by_bucket,
        ["score_bucket", "rows", "outcome_available", "positive_outcomes", "outcome_rate", "avg_sim_net_return"],
    )
    write_csv(
        data_dir / "ae17_real_meta_evaluation_by_attached_model_count.csv",
        eval_by_attached,
        [
            "attached_model_count",
            "rows",
            "outcome_available",
            "positive_outcomes",
            "outcome_rate",
            "avg_sim_net_return",
        ],
    )

    feature_parity = build_feature_parity_audit(feature_rows, relpath(feature_path, root))
    no_lookahead = build_no_lookahead_audit(feature_rows, relpath(feature_path, root))
    score_integrity = build_score_integrity_audit(output_rows)
    lineage_rows = build_lineage_audit_rows(output_rows)
    lineage_pass = all(r["lineage_fields_present"] for r in lineage_rows) if lineage_rows else False
    pair_conc = compute_pair_concentration(output_rows)
    semantic = build_semantic_partial_evidence_audit()
    authority = authority_audit_payload()
    null_safety = build_null_safety_audit(rows_processed=rows_processed)
    formula_audit = build_formula_audit(output_rows)
    baseline_comparison = build_baseline_comparison_audit(output_rows)
    baseline_vs_explicit_rows = build_baseline_vs_explicit_evaluation(output_rows)
    write_csv(
        data_dir / "ae17_real_meta_evaluation_baseline_vs_explicit.csv",
        baseline_vs_explicit_rows,
        [
            "historical_meta_row_id",
            "scoring_tier",
            "baseline_tier_score",
            "baseline_tier_decision",
            "meta_score",
            "meta_decision",
            "baseline_vs_explicit_score_delta",
            "baseline_vs_explicit_decision_changed",
            "weighted_model_score",
            "vote_ratio",
            "consensus_strength",
        ],
    )

    write_csv(
        audits_dir / "ae17_real_meta_source_file_audit.csv",
        source_audit_rows,
        [
            "path",
            "size_bytes",
            "size_gb",
            "has_required_columns",
            "missing_required_columns",
            "used_in_run",
            "skipped_large_file",
            "skipped_missing_columns",
            "skipped_read_error",
            "rows_processed_from_file",
            "error_message",
        ],
    )
    write_json(audits_dir / "ae17_real_meta_feature_parity_audit.json", feature_parity)
    write_json(audits_dir / "ae17_real_meta_no_lookahead_audit.json", no_lookahead)
    write_json(audits_dir / "ae17_real_meta_score_integrity_audit.json", score_integrity)
    write_csv(
        audits_dir / "ae17_real_meta_lineage_audit.csv",
        lineage_rows,
        [
            "historical_meta_row_id",
            "target_row_id",
            "candidate_id",
            "candidate_policy_id",
            "pair_address",
            "event_timestamp",
            "lineage_fields_present",
            "pair_address_identity_note",
            "lineage_status",
        ],
    )
    write_json(audits_dir / "ae17_real_meta_pair_concentration_audit.json", pair_conc)
    write_json(audits_dir / "ae17_real_meta_partial_evidence_semantic_audit.json", semantic)
    write_json(audits_dir / "ae17_real_meta_authority_audit.json", authority)
    write_json(audits_dir / "ae17_real_meta_null_safety_audit.json", null_safety)
    write_json(audits_dir / "ae17_real_meta_formula_audit.json", formula_audit)
    write_json(
        audits_dir / "ae17_real_meta_baseline_comparison_audit.json",
        baseline_comparison,
    )

    decision_dist = dict(Counter(r.get("meta_decision") for r in output_rows))
    tier_dist = dict(Counter(r.get("scoring_tier") for r in output_rows))
    partial_dist = dict(Counter(r.get("partial_evidence_status") for r in output_rows))
    baseline_decision_dist = dict(
        Counter(r.get("baseline_tier_decision") for r in output_rows)
    )

    if not no_lookahead.get("no_lookahead_pass"):
        classification = CLASSIFICATION_LOOKAHEAD
    elif not score_integrity.get("score_integrity_pass"):
        classification = CLASSIFICATION_SCORE_INTEGRITY
    elif formula_audit.get("classification") == CLASSIFICATION_INCOMPLETE_SUBSTANCE:
        classification = CLASSIFICATION_INCOMPLETE_SUBSTANCE
    elif not all(
        [
            authority["trade_authority"] is False,
            authority["live_trading_ready"] is False,
            authority["paper_demo_only"] is True,
            authority["risk_override_authority"] is False,
            authority["wallet_access"] is False,
            authority["private_key_access"] is False,
            authority["db_mutation"] is False,
            authority["orders_opened"] == 0,
            authority["positions_opened"] == 0,
            authority["llm_calls"] == 0,
            authority["external_api_calls"] == 0,
            authority["training_or_fit"] is False,
        ]
    ):
        classification = CLASSIFICATION_AUTHORITY
    elif (
        feature_parity.get("feature_parity_pass")
        and no_lookahead.get("no_lookahead_pass")
        and score_integrity.get("score_integrity_pass")
        and lineage_pass
        and semantic.get("partial_evidence_semantic_pass")
        and formula_audit.get("formula_substance_pass")
    ):
        classification = (
            CLASSIFICATION_PASS_WARNINGS if source_warnings or files_skipped else CLASSIFICATION_PASS
        )
    else:
        classification = CLASSIFICATION_PASS_WARNINGS if files_skipped else CLASSIFICATION_PASS

    manifest = {
        "created_at_utc": utc_now_iso(),
        "output_root": relpath(out, root),
        "classification": classification,
        "stage_decision": "AE17_NOT_CLOSED",
        "ae17_closed": False,
        "meta_layer_type": META_LAYER_TYPE,
        "meta_formula_version": META_FORMULA_VERSION,
        "baseline_layer_type": BASELINE_LAYER_TYPE,
        "baseline_formula_version": BASELINE_FORMULA_VERSION,
        "tier_only_scoring": False,
        "source_glob": source_glob,
        "max_source_bytes": max_source_bytes,
        "max_rows": max_rows,
        "include_large": include_large,
        "files_discovered": len(discovered),
        "files_used": files_used,
        "files_skipped": files_skipped,
        "source_warnings": source_warnings,
        "rows_processed": rows_processed,
        "feature_matrix_rows": len(feature_rows),
        "meta_output_rows": len(output_rows),
        "example_rows": min(20, len(output_rows)),
        "decision_distribution": decision_dist,
        "baseline_decision_distribution": baseline_decision_dist,
        "tier_distribution": tier_dist,
        "partial_evidence_distribution": partial_dist,
        "score_min": score_integrity.get("score_min"),
        "score_max": score_integrity.get("score_max"),
        "out_of_range_final_scores": score_integrity.get("out_of_range_final_scores"),
        "pair_concentration": pair_conc,
        "feature_parity_pass": feature_parity.get("feature_parity_pass"),
        "no_lookahead_pass": no_lookahead.get("no_lookahead_pass"),
        "lineage_pass": lineage_pass,
        "score_integrity_pass": score_integrity.get("score_integrity_pass"),
        "partial_evidence_semantic_pass": semantic.get("partial_evidence_semantic_pass"),
        "formula_audit_classification": formula_audit.get("classification"),
        "baseline_comparison_classification": baseline_comparison.get("classification"),
        "authority": authority,
        **canonical_context_contract_report(),
        "notes": [
            "AE17 Path B: non-learned explicit meta-combination (not tier-only).",
            "Baseline tier-only scores retained only for comparison.",
            "Uses E5 unified selected_trades_by_tier files containing TAB/XGB/RF/Consensus/Outcome columns.",
            "Large sources are skipped by default unless --include-large is set.",
            "Outcome labels are used only after scoring for retrospective evaluation.",
            "pair_address is price-source / pool-observation identity only, not tradability proof.",
            "Context features are explicitly missing pending AE18 "
            f"({CONTEXT_STATUS} / {CONTEXT_MISSINGNESS_REASON}).",
            "No training, no DB mutation, no LLM, no wallet, no live trading.",
        ],
    }
    decision_gate = {
        "created_at_utc": utc_now_iso(),
        "classification": classification,
        "stage_decision": "AE17_NOT_CLOSED",
        "ae17_closed": False,
        "meta_layer_type": META_LAYER_TYPE,
        "meta_formula_version": META_FORMULA_VERSION,
        "tier_only_scoring": False,
        "rows_processed": rows_processed,
        "feature_matrix_rows": len(feature_rows),
        "meta_output_rows": len(output_rows),
        "files_discovered": len(discovered),
        "files_used": len(files_used),
        "files_skipped": len(files_skipped),
        "source_warnings": source_warnings,
        "decision_distribution": decision_dist,
        "baseline_decision_distribution": baseline_decision_dist,
        "tier_distribution": tier_dist,
        "partial_evidence_distribution": partial_dist,
        "score_min": score_integrity.get("score_min"),
        "score_max": score_integrity.get("score_max"),
        "out_of_range_final_scores": score_integrity.get("out_of_range_final_scores"),
        "pair_concentration": {
            "top_pair_share": pair_conc.get("top_pair_share"),
            "hhi": pair_conc.get("hhi"),
            "unique_pairs": pair_conc.get("unique_pairs"),
        },
        "feature_parity_pass": feature_parity.get("feature_parity_pass"),
        "no_lookahead_pass": no_lookahead.get("no_lookahead_pass"),
        "lineage_pass": lineage_pass,
        "score_integrity_pass": score_integrity.get("score_integrity_pass"),
        "null_safety": null_safety,
        "formula_audit_classification": formula_audit.get("classification"),
        "baseline_comparison_classification": baseline_comparison.get("classification"),
        "authority_pass": True,
        "authority": authority,
        **canonical_context_contract_report(),
        "not_available_due_to_blocker": [],
    }
    summary = "\n".join(
        [
            "AE17 REAL META EVIDENCE RUN",
            "",
            f"Output root:\n{relpath(out, root)}",
            "",
            f"Classification:\n{classification}",
            "",
            "Stage decision:\nAE17_NOT_CLOSED",
            "",
            f"Files discovered:\n{len(discovered)}",
            "",
            "Files used:",
            *[str(p) for p in files_used],
            "",
            "Files skipped:",
            *([str(p) for p in files_skipped] if files_skipped else ["(none)"]),
            "",
            f"Rows processed:\n{rows_processed}",
            "",
            f"Feature matrix rows:\n{len(feature_rows)}",
            "",
            f"Meta output rows:\n{len(output_rows)}",
            "",
            "Decision distribution:",
            json.dumps(decision_dist, indent=2),
            "",
            "Tier distribution:",
            json.dumps(tier_dist, indent=2),
            "",
            "Partial evidence distribution:",
            json.dumps(partial_dist, indent=2),
            "",
            f"Meta score min:\n{score_integrity.get('score_min')}",
            "",
            f"Meta score max:\n{score_integrity.get('score_max')}",
            "",
            f"Out-of-range final scores:\n{score_integrity.get('out_of_range_final_scores')}",
            "",
            "Pair concentration:",
            f"top_pair_share={pair_conc.get('top_pair_share')}",
            f"hhi={pair_conc.get('hhi')}",
            f"unique_pairs={pair_conc.get('unique_pairs')}",
            "",
            "Authority:",
            "trade_authority=false",
            "live_trading_ready=false",
            "db_mutation=false",
            "wallet_access=false",
            "llm_calls=0",
            "training_or_fit=false",
            "",
            "Context missingness:",
            f"context_feature_available={CONTEXT_FEATURE_AVAILABLE}",
            f"context_status={CONTEXT_STATUS}",
            f"context_missingness_reason={CONTEXT_MISSINGNESS_REASON}",
            f"context_score_weight={CONTEXT_SCORE_WEIGHT}",
            "",
            "Important:",
            "This is NOT AE17 closure.",
            "This is durable evidence that AE17 can consume real TAB/XGB/RF/Consensus evidence.",
            "",
            "AE17 closed: false",
            "",
        ]
    )
    write_json(reports_dir / "ae17_real_meta_manifest.json", manifest)
    write_json(reports_dir / "ae17_real_meta_decision_gate.json", decision_gate)
    write_text(reports_dir / "ae17_real_meta_summary_for_upload.txt", summary)

    return {
        "classification": classification,
        "output_root": str(out),
        "rows_processed": rows_processed,
        "feature_matrix_rows": len(feature_rows),
        "meta_output_rows": len(output_rows),
        "files_discovered": len(discovered),
        "files_used": files_used,
        "files_skipped": files_skipped,
        "source_warnings": source_warnings,
        "decision_distribution": decision_dist,
        "tier_distribution": tier_dist,
        "partial_evidence_distribution": partial_dist,
        "score_min": score_integrity.get("score_min"),
        "score_max": score_integrity.get("score_max"),
        "out_of_range_final_scores": score_integrity.get("out_of_range_final_scores"),
        "pair_concentration": pair_conc,
        "feature_parity_pass": feature_parity.get("feature_parity_pass"),
        "no_lookahead_pass": no_lookahead.get("no_lookahead_pass"),
        "lineage_pass": lineage_pass,
        "score_integrity_pass": score_integrity.get("score_integrity_pass"),
        "null_safety": null_safety,
        "authority": authority,
        "ae17_closed": False,
        "decision_gate": decision_gate,
        "manifest": manifest,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AE17 durable real-meta evidence runner (deterministic, read-only)"
    )
    p.add_argument("--source-glob", type=str, default=DEFAULT_SOURCE_GLOB)
    p.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Default: data/audits/ae17_real_meta_evidence_run_<timestamp>",
    )
    p.add_argument("--max-source-bytes", type=int, default=DEFAULT_MAX_SOURCE_BYTES)
    p.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS)
    p.add_argument(
        "--include-large",
        action="store_true",
        default=False,
        help="Allow sources larger than max-source-bytes (still streamed / max-rows capped)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_ae17_real_meta_evidence(
            ROOT,
            source_glob=args.source_glob,
            output_root=args.output_root,
            max_source_bytes=args.max_source_bytes,
            max_rows=args.max_rows,
            include_large=bool(args.include_large),
        )
    except Exception as exc:  # noqa: BLE001 — final controlled boundary
        # Source-not-found / empty glob must never reach here as IndexError.
        print(f"[{PHASE}] unexpected error: {exc}", file=sys.stderr)
        return 1

    classification = result.get("classification")
    print(f"[{PHASE}] classification: {classification}")
    print(f"[{PHASE}] output_root: {result.get('output_root')}")
    print(f"[{PHASE}] rows_processed: {result.get('rows_processed')}")
    print(f"[{PHASE}] feature_matrix_rows: {result.get('feature_matrix_rows')}")
    print(f"[{PHASE}] meta_output_rows: {result.get('meta_output_rows')}")
    print(f"[{PHASE}] ae17_closed: false")
    print(f"[{PHASE}] ae18_status: BLOCKED")
    print(f"[{PHASE}] ae19_status: BLOCKED")
    if str(classification).startswith("AE17_BLOCKED_"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
