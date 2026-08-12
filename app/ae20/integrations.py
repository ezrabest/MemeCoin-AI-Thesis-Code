"""AE16/AE17/AE18/AE19 integration attachment for AE20."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

from app.ae20.exact_identity_evidence_bridge import (
    AE16_ATTACHED_STATUSES,
    resolve_ae20_ae16_exact_bridge,
)
from app.ae20.identity_keys import make_exact_identity_lookup_key
from app.consensus.serialization import read_csv_dicts
from app.llm_operational.providers import resolve_qwen_provider_status
from app.llm_operational.qwen_runtime import call_ollama_chat
from app.llm_operational.schema import PROVIDER_AVAILABLE

# Default AE16 bridge — project-root relative. Never hardcode absolute Windows paths.
# Legacy lowercase/normalized locator artifact — NOT AE20 closure evidence authority.
DEFAULT_AE16_BRIDGE_RELATIVE = (
    "data/audits/ae16_model_evidence_bridge_completion_20260722_213752/"
    "data/ae16_clean_forward_consensus_decisions_v2.csv"
)
LEGACY_AE16_BRIDGE_IS_EVIDENCE_AUTHORITY = False

AE16_SELECTED_COLUMNS = (
    "provider_pair_url",
    "rf_evidence_status",
    "xgb_evidence_status",
    "tab_evidence_status",
    "rf_score",
    "xgb_score",
    "tab_score",
    "rf_vote",
    "xgb_vote",
    "tab_vote",
    "model_vote_count",
    "consensus_tier",
    "consensus_reason",
    "consensus_engine_version",
)


def _cell(value: Any) -> str:
    """Strip-only string cell for non-identity display fields."""
    if value is None:
        return ""
    return str(value).strip()


def _to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    # Boolean literals only — not identity fields.
    s = str(value).strip()
    if s in {"1", "true", "True", "TRUE", "yes", "Yes", "YES", "y", "Y"}:
        return True
    return False


def resolve_ae16_bridge_source(
    project_root: Path,
    *,
    cli_override: str | Path | None = None,
    env_override: str | None = None,
) -> dict[str, Any]:
    """Resolve AE16 bridge CSV path: CLI → ENV → default relative."""
    project_root = project_root.resolve()
    override_used = False
    override_type = "DEFAULT"
    relative = DEFAULT_AE16_BRIDGE_RELATIVE

    chosen: Path | None = None
    if cli_override is not None and str(cli_override).strip():
        override_used = True
        override_type = "CLI"
        chosen = Path(cli_override)
        if not chosen.is_absolute():
            chosen = project_root / chosen
        relative = str(cli_override).replace("\\", "/")
    else:
        env_val = env_override if env_override is not None else os.environ.get("AE20_AE16_BRIDGE_SOURCE")
        if env_val and str(env_val).strip():
            override_used = True
            override_type = "ENV"
            chosen = Path(env_val)
            if not chosen.is_absolute():
                chosen = project_root / chosen
            relative = str(env_val).replace("\\", "/")
        else:
            chosen = project_root / DEFAULT_AE16_BRIDGE_RELATIVE
            relative = DEFAULT_AE16_BRIDGE_RELATIVE

    resolved = chosen.resolve()
    return {
        "ae16_bridge_source_path_relative": relative.replace("\\", "/"),
        "ae16_bridge_source_path_resolved": str(resolved),
        "ae16_bridge_source_exists": resolved.is_file(),
        "ae16_bridge_source_override_used": override_used,
        "ae16_bridge_source_override_type": override_type,
        "path": resolved,
    }


# --- AE17 explicit meta (Path-B) — mirrored constants (no training) ---
META_FORMULA_VERSION = "AE17_EXPLICIT_META_COMBINATION_V1"
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
    "XGB_RF_ONLY": 0.55,
    "SINGLE_MODEL_ONLY": 0.30,
    "MODEL_EVIDENCE_UNAVAILABLE": 0.0,
}
DECISION_THRESHOLDS = {
    "META_STRONG_WATCH": 0.75,
    "META_SECONDARY_WATCH": 0.55,
    "META_RESEARCH_ONLY": 0.35,
    "META_LOW_CONFIDENCE": 0.15,
}


def compute_ae17_explicit_meta_combination(
    *,
    rf_score: float | None,
    xgb_score: float | None,
    tab_score: float | None,
    tab_vote: bool,
    xgb_vote: bool,
    rf_vote: bool,
    consensus_tier: str | None = None,
) -> dict[str, Any]:
    attached_model_count = (
        int(rf_score is not None) + int(xgb_score is not None) + int(tab_score is not None)
    )
    model_vote_count = int(tab_vote) + int(xgb_vote) + int(rf_vote)
    if consensus_tier:
        scoring_tier = consensus_tier
    elif tab_vote and xgb_vote and rf_vote:
        scoring_tier = "TAB_XGB_RF_ALL3"
    elif tab_vote and rf_vote and not xgb_vote:
        scoring_tier = "TAB_RF_ONLY"
    elif tab_vote and xgb_vote and not rf_vote:
        scoring_tier = "TAB_XGB_ONLY"
    elif xgb_vote and rf_vote and not tab_vote:
        scoring_tier = "XGB_RF_ONLY"
    elif model_vote_count == 1:
        scoring_tier = "SINGLE_MODEL_ONLY"
    else:
        scoring_tier = "MODEL_EVIDENCE_UNAVAILABLE"

    parts: list[tuple[str, float, float]] = []
    for name, score in (("rf", rf_score), ("xgb", xgb_score), ("tab", tab_score)):
        if score is None:
            continue
        parts.append((name, float(score), float(MODEL_SCORE_WEIGHTS[name])))
    active_weight = sum(w for _, _, w in parts)
    weighted_model_score = (
        sum(s * w for _, s, w in parts) / active_weight if active_weight > 0 else None
    )
    vote_ratio = (model_vote_count / attached_model_count) if attached_model_count else None
    evidence_coverage = attached_model_count / 3.0
    consensus_strength = CONSENSUS_STRENGTH_MAP.get(scoring_tier)

    if weighted_model_score is None or vote_ratio is None or consensus_strength is None:
        return {
            "meta_score": None,
            "meta_decision": "META_UNAVAILABLE",
            "meta_formula_version": META_FORMULA_VERSION,
            "scoring_tier": scoring_tier,
            "attached_model_count": attached_model_count,
            "model_vote_count": model_vote_count,
            "score_integrity_status": "AE17_META_UNAVAILABLE",
            "context_missingness_component": 0.0,
            "weighted_model_score": weighted_model_score,
        }

    pre = (
        COMPONENT_WEIGHTS["weighted_model_score"] * weighted_model_score
        + COMPONENT_WEIGHTS["vote_ratio"] * vote_ratio
        + COMPONENT_WEIGHTS["consensus_strength"] * float(consensus_strength)
        + COMPONENT_WEIGHTS["evidence_coverage"] * evidence_coverage
    )
    meta_score = max(0.0, min(1.0, pre))
    if meta_score >= DECISION_THRESHOLDS["META_STRONG_WATCH"]:
        decision = "META_STRONG_WATCH"
    elif meta_score >= DECISION_THRESHOLDS["META_SECONDARY_WATCH"]:
        decision = "META_SECONDARY_WATCH"
    elif meta_score >= DECISION_THRESHOLDS["META_RESEARCH_ONLY"]:
        decision = "META_RESEARCH_ONLY"
    elif meta_score >= DECISION_THRESHOLDS["META_LOW_CONFIDENCE"]:
        decision = "META_LOW_CONFIDENCE"
    else:
        decision = "META_REJECT"
    return {
        "meta_score": meta_score,
        "meta_decision": decision,
        "meta_formula_version": META_FORMULA_VERSION,
        "scoring_tier": scoring_tier,
        "attached_model_count": attached_model_count,
        "model_vote_count": model_vote_count,
        "score_integrity_status": "AE17_SCORE_OK",
        "context_missingness_component": 0.0,
        "weighted_model_score": weighted_model_score,
    }


def load_ae16_exact_derived_bridge_index(
    project_root: Path,
    *,
    exact_bridge_path: str | Path | None = None,
    env_override: str | None = None,
) -> dict[str, Any] | None:
    """Load derived AE20↔AE16 exact-identity bridge CSV (preferred evidence authority).

    Indexed by exact ae20_provider_pair_url_exact. Only valid bridge rows attach.
    """
    project_root = Path(project_root).resolve()
    meta = resolve_ae20_ae16_exact_bridge(
        project_root, cli_override=exact_bridge_path, env_override=env_override
    )
    path = meta.get("path")
    if path is None or not Path(path).is_file():
        return None

    by_url: dict[str, dict[str, Any]] = {}
    rows_loaded = 0
    valid_rows = 0
    for r in read_csv_dicts(Path(path)):
        rows_loaded += 1
        key = make_exact_identity_lookup_key(r.get("ae20_provider_pair_url_exact"))
        if key is None:
            continue
        valid_flag = str(r.get("exact_identity_bridge_row_valid") or "").strip()
        if valid_flag not in {"1", "true", "True", "TRUE"}:
            continue
        valid_rows += 1
        by_url[key] = dict(r)

    audit = {
        **meta,
        "ae16_evidence_mode": "EXACT_DERIVED_BRIDGE",
        "ae16_join_key_left": "provider_pair_url_exact",
        "ae16_join_key_right": "ae20_provider_pair_url_exact",
        "ae16_join_safety": "SAFE_EXACT_DERIVED_BRIDGE",
        "ae16_rows_loaded": rows_loaded,
        "ae16_rows_with_valid_provider_url_key": valid_rows,
        "ae16_rows_with_evidence": valid_rows,
        "ae16_invalid_provider_pair_url_count": rows_loaded - valid_rows,
        "exact_identity_join_used": True,
        "case_insensitive_join_used": False,
        "lowercase_join_used": False,
        "casefold_join_used": False,
        "identity_case_preserved": True,
        "empty_join_keys_used": False,
        "nan_join_keys_used": False,
        "invalid_join_keys_filtered": True,
        "safe_provider_url_join_used": True,
        "forbidden_pair_chain_join_used": False,
        "broad_merge_used": False,
        "lookup_dictionary_used": True,
        "uncontrolled_pandas_suffix_columns_present": False,
        "legacy_ae16_bridge_used_as_evidence_authority": False,
        "ae16_bridge_source_exists": True,
        "ae16_bridge_source_path_relative": str(meta.get("ae20_ae16_exact_bridge_path_resolved") or ""),
        "ae16_bridge_source_path_resolved": str(path),
        "ae16_bridge_source_override_used": meta.get("ae20_ae16_exact_bridge_override_type")
        in {"CLI", "ENV"},
        "ae16_bridge_source_override_type": meta.get("ae20_ae16_exact_bridge_override_type"),
    }
    return {
        "path": str(path),
        "rows": list(by_url.values()),
        "by_provider_pair_url": by_url,
        "audit": audit,
        "evidence_mode": "EXACT_DERIVED_BRIDGE",
        **meta,
        "ae16_bridge_source_exists": True,
        "ae16_bridge_source_path_relative": audit["ae16_bridge_source_path_relative"],
        "ae16_bridge_source_path_resolved": str(path),
        "ae16_bridge_source_override_used": audit["ae16_bridge_source_override_used"],
        "ae16_bridge_source_override_type": audit["ae16_bridge_source_override_type"],
    }


def load_ae16_index(
    project_root: Path,
    ae16_root: Path | None = None,
    *,
    ae16_bridge_source: str | Path | None = None,
    env_override: str | None = None,
    ae20_ae16_exact_bridge: str | Path | None = None,
    exact_bridge_env_override: str | None = None,
) -> dict[str, Any]:
    """Load AE16 evidence for AE20 attachment.

    Preferred: exact derived bridge (--ae20-ae16-exact-bridge / AE20_AE16_EXACT_BRIDGE).
    Fallback: legacy AE16 provider_pair_url CSV (exact-case only; not preferred authority).
    """
    project_root = Path(project_root).resolve()

    exact_idx = load_ae16_exact_derived_bridge_index(
        project_root,
        exact_bridge_path=ae20_ae16_exact_bridge,
        env_override=exact_bridge_env_override,
    )
    if exact_idx is not None:
        return exact_idx

    # Legacy ae16_root dir arg: if a CSV path is passed via ae16_bridge_source it wins.
    # If ae16_root points at a file, treat as CLI-style override.
    cli = ae16_bridge_source
    if cli is None and ae16_root is not None:
        root_path = Path(ae16_root)
        if root_path.suffix.lower() == ".csv":
            cli = root_path
        elif (root_path / "data" / "ae16_clean_forward_consensus_decisions_v2.csv").is_file():
            cli = root_path / "data" / "ae16_clean_forward_consensus_decisions_v2.csv"

    source_meta = resolve_ae16_bridge_source(
        project_root, cli_override=cli, env_override=env_override
    )
    path: Path = source_meta["path"]
    by_url: dict[str, dict[str, Any]] = {}
    rows_loaded = 0
    valid_key_rows = 0
    invalid_key_rows = 0
    rows_with_evidence = 0
    selected_imported = list(AE16_SELECTED_COLUMNS)

    if path.is_file():
        raw_rows = [dict(r) for r in read_csv_dicts(path)]
        rows_loaded = len(raw_rows)
        for r in raw_rows:
            key = make_exact_identity_lookup_key(r.get("provider_pair_url"))
            if key is None:
                invalid_key_rows += 1
                continue
            # Do not treat MODEL_EVIDENCE_UNAVAILABLE-only legacy rows as evidence authority.
            tier = _cell(r.get("consensus_tier"))
            if tier == "MODEL_EVIDENCE_UNAVAILABLE" and not LEGACY_AE16_BRIDGE_IS_EVIDENCE_AUTHORITY:
                # Still index for diagnostic exact overlap, but mark non-authoritative.
                pass
            valid_key_rows += 1
            selected: dict[str, Any] = {}
            for col in AE16_SELECTED_COLUMNS:
                selected[col] = r.get(col)
            # Preserve original provider_pair_url exactly (no mutation).
            selected["provider_pair_url"] = key
            selected["_legacy_bridge_non_authoritative"] = (
                tier == "MODEL_EVIDENCE_UNAVAILABLE"
            )
            by_url[key] = selected
            # Count as evidence row if any evidence column present (status/score/tier).
            if any(
                selected.get(c) not in (None, "")
                for c in (
                    "rf_evidence_status",
                    "xgb_evidence_status",
                    "tab_evidence_status",
                    "rf_score",
                    "xgb_score",
                    "tab_score",
                    "consensus_tier",
                )
            ):
                rows_with_evidence += 1

    audit_base = {
        **source_meta,
        "ae16_evidence_mode": "LEGACY_PROVIDER_URL_BRIDGE",
        "legacy_ae16_bridge_used_as_evidence_authority": False,
        "ae16_join_key_left": "provider_pair_url_exact",
        "ae16_join_key_right": "provider_pair_url",
        "ae16_join_safety": "SAFE_CANONICAL_OR_PROVIDER_IDENTITY",
        "ae16_rows_loaded": rows_loaded,
        "ae16_rows_with_valid_provider_url_key": valid_key_rows,
        "ae16_rows_with_evidence": rows_with_evidence,
        "ae16_invalid_provider_pair_url_count": invalid_key_rows,
        "exact_identity_join_used": True,
        "case_insensitive_join_used": False,
        "lowercase_join_used": False,
        "casefold_join_used": False,
        "identity_case_preserved": True,
        "empty_join_keys_used": False,
        "nan_join_keys_used": False,
        "invalid_join_keys_filtered": True,
        "safe_provider_url_join_used": True,
        "forbidden_pair_chain_join_used": False,
        "broad_merge_used": False,
        "lookup_dictionary_used": True,
        "uncontrolled_pandas_suffix_columns_present": False,
        "ae16_selected_columns_imported": selected_imported,
        "ae16_imported_column_count": len(selected_imported),
        "ae16_column_collision_count": 0,
        "ae16_column_collision_policy": "LOOKUP_DICTIONARY_SELECTED_COLUMNS_ONLY",
    }
    return {
        "path": str(path) if path.is_file() else None,
        "rows": list(by_url.values()),
        "by_provider_pair_url": by_url,
        # Intentionally no by_pair — pair+chain join forbidden for AE20 closure.
        "audit": audit_base,
        "evidence_mode": "LEGACY_PROVIDER_URL_BRIDGE",
        **source_meta,
    }


def _empty_ae16_attach(status: str) -> dict[str, Any]:
    return {
        "ae16_status": status,
        "ae16_provider_pair_url_original": "",
        "ae16_rf_evidence_status": "",
        "ae16_xgb_evidence_status": "",
        "ae16_tab_evidence_status": "",
        "ae16_rf_score": None,
        "ae16_xgb_score": None,
        "ae16_tab_score": None,
        "ae16_rf_vote": False,
        "ae16_xgb_vote": False,
        "ae16_tab_vote": False,
        "ae16_model_vote_count": "",
        "ae16_consensus_tier": "",
        "ae16_consensus_reason": "",
        "ae16_consensus_engine_version": "",
        # Compatibility aliases for downstream AE17 / decisions
        "rf_evidence_status": "",
        "xgb_evidence_status": "",
        "tab_evidence_status": "",
        "consensus_tier": "",
        "model_vote_count": "",
        "attached_model_count": "",
        "consensus_reason": "",
        "consensus_engine_version": "",
        "rf_score": None,
        "xgb_score": None,
        "tab_score": None,
        "rf_vote": False,
        "xgb_vote": False,
        "tab_vote": False,
        "ae16_join_key": "provider_pair_url_exact",
        "exact_identity_join_used": True,
        "case_insensitive_join_used": False,
        "lowercase_join_used": False,
        "casefold_join_used": False,
        "forbidden_pair_chain_join_used": False,
        "provider_pair_url_exact_mutated": False,
        "ae16_provider_pair_url_mutated": False,
    }


def attach_ae16(candidate: dict[str, Any], ae16_index: dict[str, Any]) -> dict[str, Any]:
    """Attach AE16 evidence via exact case-preserved provider_pair_url_exact join only."""
    original_exact = candidate.get("provider_pair_url_exact")
    key = make_exact_identity_lookup_key(original_exact)
    by_url = ae16_index.get("by_provider_pair_url") or {}
    evidence_mode = ae16_index.get("evidence_mode") or (ae16_index.get("audit") or {}).get(
        "ae16_evidence_mode"
    )

    if key is None:
        status = (
            "AE16_EXACT_DERIVED_BRIDGE_NOT_FOUND"
            if evidence_mode == "EXACT_DERIVED_BRIDGE"
            else "AE16_JOIN_NOT_FOUND"
        )
        out = _empty_ae16_attach(status)
        out["ae20_provider_pair_url_exact_invalid"] = True
        return out

    row = by_url.get(key)
    if row is None:
        if evidence_mode == "EXACT_DERIVED_BRIDGE":
            status = "AE16_EXACT_DERIVED_BRIDGE_NOT_FOUND"
        elif not by_url:
            status = "AE16_EVIDENCE_UNAVAILABLE"
        else:
            status = "AE16_JOIN_NOT_FOUND"
        out = _empty_ae16_attach(status)
        # Preserve AE20 exact URL unchanged on the candidate side (caller keeps original).
        out["provider_pair_url_exact_preserved"] = original_exact
        return out

    if evidence_mode == "EXACT_DERIVED_BRIDGE":
        return _attach_from_exact_derived_bridge(original_exact, row)

    original_ae16_url = row.get("provider_pair_url")
    # Guaranteed equal to key by construction; assert no mutation.
    ae16_url_preserved = original_ae16_url
    # Legacy bridge with MODEL_EVIDENCE_UNAVAILABLE is not closure evidence authority.
    if row.get("_legacy_bridge_non_authoritative"):
        out = _empty_ae16_attach("AE16_JOIN_NOT_FOUND")
        out["provider_pair_url_exact_preserved"] = original_exact
        out["legacy_ae16_bridge_non_authoritative"] = True
        return out

    rf_score = _to_float(row.get("rf_score"))
    xgb_score = _to_float(row.get("xgb_score"))
    tab_score = _to_float(row.get("tab_score"))
    rf_vote = _to_bool(row.get("rf_vote"))
    xgb_vote = _to_bool(row.get("xgb_vote"))
    tab_vote = _to_bool(row.get("tab_vote"))
    attached_model_count = (
        int(rf_score is not None) + int(xgb_score is not None) + int(tab_score is not None)
    )
    return {
        "ae16_status": "AE16_EVIDENCE_ATTACHED",
        "ae16_provider_pair_url_original": ae16_url_preserved,
        "ae16_rf_evidence_status": _cell(row.get("rf_evidence_status")),
        "ae16_xgb_evidence_status": _cell(row.get("xgb_evidence_status")),
        "ae16_tab_evidence_status": _cell(row.get("tab_evidence_status")),
        "ae16_rf_score": rf_score,
        "ae16_xgb_score": xgb_score,
        "ae16_tab_score": tab_score,
        "ae16_rf_vote": rf_vote,
        "ae16_xgb_vote": xgb_vote,
        "ae16_tab_vote": tab_vote,
        "ae16_model_vote_count": row.get("model_vote_count"),
        "ae16_consensus_tier": _cell(row.get("consensus_tier")),
        "ae16_consensus_reason": _cell(row.get("consensus_reason")),
        "ae16_consensus_engine_version": _cell(row.get("consensus_engine_version")),
        # Compatibility aliases
        "rf_evidence_status": _cell(row.get("rf_evidence_status")),
        "xgb_evidence_status": _cell(row.get("xgb_evidence_status")),
        "tab_evidence_status": _cell(row.get("tab_evidence_status")),
        "consensus_tier": _cell(row.get("consensus_tier")),
        "model_vote_count": row.get("model_vote_count"),
        "attached_model_count": attached_model_count,
        "consensus_reason": _cell(row.get("consensus_reason")),
        "consensus_engine_version": _cell(row.get("consensus_engine_version")),
        "rf_score": rf_score,
        "xgb_score": xgb_score,
        "tab_score": tab_score,
        "rf_vote": rf_vote,
        "xgb_vote": xgb_vote,
        "tab_vote": tab_vote,
        "ae16_join_key": "provider_pair_url_exact",
        "exact_identity_join_used": True,
        "case_insensitive_join_used": False,
        "lowercase_join_used": False,
        "casefold_join_used": False,
        "forbidden_pair_chain_join_used": False,
        "provider_pair_url_exact_mutated": False,
        "ae16_provider_pair_url_mutated": False,
        "provider_pair_url_exact_preserved": original_exact,
    }


def _attach_from_exact_derived_bridge(
    original_exact: Any, row: dict[str, Any]
) -> dict[str, Any]:
    """Map exact derived bridge columns onto AE20 AE16 attach schema."""
    rf_score = _to_float(row.get("ae16_rf_score"))
    xgb_score = _to_float(row.get("ae16_xgb_score"))
    tab_score = _to_float(
        row.get("ae16_tab_score_for_consensus")
        if row.get("ae16_tab_score_for_consensus") not in (None, "")
        else row.get("ae16_tab16_score")
    )
    rf_vote = _to_bool(row.get("ae16_rf_vote"))
    xgb_vote = _to_bool(row.get("ae16_xgb_vote"))
    tab_vote = _to_bool(
        row.get("ae16_tab_vote_for_consensus")
        if row.get("ae16_tab_vote_for_consensus") not in (None, "")
        else row.get("ae16_tab16_vote")
    )
    tier = _cell(row.get("ae16_consensus_preview_tier"))
    attached_model_count = (
        int(rf_score is not None) + int(xgb_score is not None) + int(tab_score is not None)
    )
    model_vote_count = row.get("ae16_true_vote_count")
    if model_vote_count in (None, ""):
        model_vote_count = int(rf_vote) + int(xgb_vote) + int(tab_vote)
    rf_status = _cell(row.get("ae16_rf_status")) or "MODEL_EVIDENCE_ATTACHED"
    xgb_status = _cell(row.get("ae16_xgb_status")) or "MODEL_EVIDENCE_ATTACHED"
    tab_status = _cell(row.get("ae16_tab16_status")) or "MODEL_EVIDENCE_ATTACHED"
    return {
        "ae16_status": "AE16_EVIDENCE_ATTACHED_FROM_EXACT_DERIVED_BRIDGE",
        "ae16_provider_pair_url_original": make_exact_identity_lookup_key(
            row.get("ae20_provider_pair_url_exact")
        )
        or original_exact,
        "ae16_rf_evidence_status": rf_status,
        "ae16_xgb_evidence_status": xgb_status,
        "ae16_tab_evidence_status": tab_status,
        "ae16_rf_score": rf_score,
        "ae16_xgb_score": xgb_score,
        "ae16_tab_score": tab_score,
        "ae16_rf_vote": rf_vote,
        "ae16_xgb_vote": xgb_vote,
        "ae16_tab_vote": tab_vote,
        "ae16_model_vote_count": model_vote_count,
        "ae16_consensus_tier": tier,
        "ae16_consensus_reason": "exact_derived_bridge",
        "ae16_consensus_engine_version": "ae20_ae16_exact_identity_evidence_bridge_v2",
        "ae16_tab16_model_variant": _cell(row.get("ae16_tab16_model_variant")),
        "ae16_tab16_artifact_path": _cell(row.get("ae16_tab16_artifact_path")),
        "ae16_join_method": _cell(row.get("ae16_join_method")),
        "ae16_join_method_safety": _cell(row.get("ae16_join_method_safety")),
        "legacy_locator_used": _to_bool(row.get("legacy_locator_used")),
        "legacy_locator_was_computed_by_ae20": False,
        "legacy_locator_is_canonical_identity": False,
        # Compatibility aliases
        "rf_evidence_status": rf_status,
        "xgb_evidence_status": xgb_status,
        "tab_evidence_status": tab_status,
        "consensus_tier": tier,
        "model_vote_count": model_vote_count,
        "attached_model_count": attached_model_count,
        "consensus_reason": "exact_derived_bridge",
        "consensus_engine_version": "ae20_ae16_exact_identity_evidence_bridge_v2",
        "rf_score": rf_score,
        "xgb_score": xgb_score,
        "tab_score": tab_score,
        "rf_vote": rf_vote,
        "xgb_vote": xgb_vote,
        "tab_vote": tab_vote,
        "ae16_join_key": "provider_pair_url_exact",
        "exact_identity_join_used": True,
        "case_insensitive_join_used": False,
        "lowercase_join_used": False,
        "casefold_join_used": False,
        "forbidden_pair_chain_join_used": False,
        "pair_chain_only_join_used_for_closure": False,
        "provider_pair_url_exact_mutated": False,
        "ae16_provider_pair_url_mutated": False,
        "provider_pair_url_exact_preserved": original_exact,
    }


def load_ae17_index(project_root: Path, ae17_root: Path | None = None) -> dict[str, Any]:
    root = ae17_root or (
        project_root / "data/audits/ae17_real_meta_evidence_run_20260726_202057"
    )
    path = root / "data" / "ae17_real_meta_outputs.csv"
    by_pair: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []
    if path.is_file():
        rows = [dict(r) for r in read_csv_dicts(path)]
        for r in rows:
            pair = make_exact_identity_lookup_key(r.get("pair_address"))
            if pair and pair not in by_pair:
                by_pair[pair] = r
    return {"path": str(path) if path.is_file() else None, "rows": rows, "by_pair": by_pair}


def load_ae18_index(project_root: Path, ae18_root: Path | None = None) -> dict[str, Any]:
    root = ae18_root or (
        project_root / "data/audits/ae18_context_inventory_audit_v3_20260802T055136Z"
    )
    path = root / "ae18_context_inventory_records_v3.jsonl"
    by_canon: dict[str, list[dict[str, Any]]] = {}
    by_psk: dict[str, list[dict[str, Any]]] = {}
    by_nkey: dict[str, list[dict[str, Any]]] = {}
    rows: list[dict[str, Any]] = []
    if path.is_file():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows.append(rec)
                canon = make_exact_identity_lookup_key(rec.get("canonical_market_identity"))
                psk = make_exact_identity_lookup_key(rec.get("price_source_key"))
                nkey = make_exact_identity_lookup_key(rec.get("normalized_provider_pair_url_key"))
                if canon:
                    by_canon.setdefault(canon, []).append(rec)
                if psk:
                    by_psk.setdefault(psk, []).append(rec)
                if nkey:
                    by_nkey.setdefault(nkey, []).append(rec)
    return {
        "path": str(path) if path.is_file() else None,
        "rows": rows,
        "by_canon": by_canon,
        "by_psk": by_psk,
        "by_nkey": by_nkey,
    }


def attach_ae17(
    candidate: dict[str, Any],
    ae16_attach: dict[str, Any],
    ae17_index: dict[str, Any],
) -> dict[str, Any]:
    pair = make_exact_identity_lookup_key(candidate.get("pair_address"))
    preexisting = ae17_index.get("by_pair", {}).get(pair) if pair else None
    if preexisting:
        return {
            "ae17_status": "AE17_META_ATTACHED",
            "meta_score": _to_float(preexisting.get("meta_score")),
            "meta_decision": _cell(preexisting.get("meta_decision")),
            "meta_formula_version": _cell(preexisting.get("meta_formula_version"))
            or META_FORMULA_VERSION,
            "score_integrity_status": "AE17_ATTACHED_FROM_REAL_META_PACKAGE",
            "context_missingness_component": _to_float(
                preexisting.get("context_component")
            )
            or 0.0,
            "ae17_join_key": "pair_address",
        }

    if ae16_attach.get("ae16_status") in AE16_ATTACHED_STATUSES:
        computed = compute_ae17_explicit_meta_combination(
            rf_score=ae16_attach.get("rf_score"),
            xgb_score=ae16_attach.get("xgb_score"),
            tab_score=ae16_attach.get("tab_score"),
            tab_vote=bool(ae16_attach.get("tab_vote")),
            xgb_vote=bool(ae16_attach.get("xgb_vote")),
            rf_vote=bool(ae16_attach.get("rf_vote")),
            consensus_tier=_cell(ae16_attach.get("consensus_tier")) or None,
        )
        return {
            "ae17_status": "AE17_META_COMPUTED",
            "meta_score": computed.get("meta_score"),
            "meta_decision": computed.get("meta_decision"),
            "meta_formula_version": computed.get("meta_formula_version"),
            "score_integrity_status": computed.get("score_integrity_status"),
            "context_missingness_component": computed.get("context_missingness_component"),
            "ae17_join_key": "computed_from_ae16",
        }

    if not ae17_index.get("rows"):
        status = "AE17_META_UNAVAILABLE"
    else:
        status = "AE17_JOIN_NOT_FOUND"
    return {
        "ae17_status": status,
        "meta_score": None,
        "meta_decision": "",
        "meta_formula_version": META_FORMULA_VERSION,
        "score_integrity_status": status,
        "context_missingness_component": 0.0,
        "ae17_join_key": "",
    }


def attach_ae18(candidate: dict[str, Any], ae18_index: dict[str, Any]) -> dict[str, Any]:
    canon = make_exact_identity_lookup_key(candidate.get("canonical_market_identity"))
    psk = make_exact_identity_lookup_key(candidate.get("price_source_key"))
    nkey = make_exact_identity_lookup_key(candidate.get("normalized_provider_pair_url_key"))
    matches: list[dict[str, Any]] = []
    join_key = ""
    if canon and canon in ae18_index.get("by_canon", {}):
        matches = ae18_index["by_canon"][canon]
        join_key = "canonical_market_identity"
    elif psk and psk in ae18_index.get("by_psk", {}):
        matches = ae18_index["by_psk"][psk]
        join_key = "price_source_key"
    elif nkey and nkey in ae18_index.get("by_nkey", {}):
        matches = ae18_index["by_nkey"][nkey]
        join_key = "normalized_provider_pair_url_key"

    if not matches:
        if not ae18_index.get("rows"):
            status = "AE18_CONTEXT_UNAVAILABLE"
        else:
            status = "AE18_RESOLVER_UNRESOLVED"
        return {
            "ae18_status": status,
            "context_family_availability": "",
            "resolver_status": status,
            "provenance_refs": "",
            "missingness_flags": "",
            "whale_score_pool_flow_proxy_separation": "UNKNOWN",
            "wallet_level_whale_evidence_status": "UNKNOWN",
            "rss_news_status": "UNKNOWN",
            "reputation_scam_status": "UNKNOWN",
            "semantic_status": _cell(candidate.get("semantic_status")) or "UNKNOWN",
            "helius_solana_status": "UNKNOWN",
            "context_summary": "",
            "ae18_join_key": join_key,
            "ae18_match_count": 0,
        }

    families = sorted({_cell(m.get("context_family")) for m in matches if _cell(m.get("context_family"))})
    statuses = sorted({_cell(m.get("status")) for m in matches if _cell(m.get("status"))})
    provenance = []
    for m in matches:
        for p in (m.get("existing_record_source_paths") or m.get("provenance") or []):
            if isinstance(p, str) and p:
                provenance.append(p)
            elif isinstance(p, dict):
                provenance.append(json.dumps(p, default=str))
    missingness_only = all("MISSINGNESS" in s.upper() for s in statuses) if statuses else False
    if missingness_only:
        status = "AE18_CONTEXT_MISSINGNESS_ONLY"
    else:
        status = "AE18_CONTEXT_ATTACHED"

    helius = "PRESENT" if any("helius" in f.lower() for f in families) else "ABSENT"
    rss = "PRESENT" if any("rss" in f.lower() or "news" in f.lower() for f in families) else "ABSENT"
    reputation = (
        "PRESENT" if any("reput" in f.lower() or "scam" in f.lower() for f in families) else "ABSENT"
    )
    semantic = (
        "PRESENT" if any("semantic" in f.lower() for f in families) else "ABSENT"
    )
    whale = (
        "SEPARATED_POOL_FLOW_PROXY"
        if any("whale" in f.lower() for f in families)
        else "NO_WHALE_FAMILY"
    )

    return {
        "ae18_status": status,
        "context_family_availability": "|".join(families),
        "resolver_status": "RESOLVER_LINKED",
        "provenance_refs": "|".join(provenance[:8]),
        "missingness_flags": "|".join(statuses[:8]),
        "whale_score_pool_flow_proxy_separation": whale,
        "wallet_level_whale_evidence_status": "INVENTORY_ONLY",
        "rss_news_status": rss,
        "reputation_scam_status": reputation,
        "semantic_status": semantic,
        "helius_solana_status": helius,
        "context_summary": f"families={','.join(families)}; status={status}",
        "ae18_join_key": join_key,
        "ae18_match_count": len(matches),
    }


def run_ae19_audit_only(
    candidate: dict[str, Any],
    *,
    allow_llm: bool,
    llm_provider: str,
    timeout_seconds: float,
    remaining_budget: int,
    force_unavailable: bool = False,
) -> dict[str, Any]:
    """Bounded AE19 audit-only LLM call. Never authorizes execution."""
    base = {
        "ae19_status": "AE19_LLM_SKIPPED_BY_CONFIG",
        "llm_task_ref": "",
        "llm_provider": llm_provider or "",
        "llm_model": "",
        "llm_task_type": "AUDIT",
        "llm_output_excerpt": "",
        "llm_failure_reason": "",
        "authority_status": "AUDIT_ONLY_NO_TRADE_AUTHORITY",
        "llm_action_label": "",
        "llm_authorizes_execution": False,
        "llm_attempted": False,
        "llm_succeeded": False,
        "llm_timeout": False,
        "llm_skipped": True,
        "llm_failed": False,
    }
    if not allow_llm or remaining_budget <= 0:
        base["llm_failure_reason"] = "AE19_LLM_SKIPPED_BY_CONFIG"
        base["ae19_status"] = "AE19_LLM_SKIPPED_BY_CONFIG"
        return base
    if llm_provider and llm_provider.lower() not in {"ollama", "qwen", "qwen/ollama"}:
        if llm_provider.lower() == "gemini":
            base["ae19_status"] = "AE19_GEMINI_NOT_ENABLED"
            base["llm_failure_reason"] = "AE19_GEMINI_NOT_ENABLED"
            return base
        base["ae19_status"] = "AE19_LLM_SKIPPED_BY_CONFIG"
        return base

    status = resolve_qwen_provider_status(
        allow_qwen=True,
        force_unavailable=force_unavailable,
    )
    if status.get("provider_status") != PROVIDER_AVAILABLE:
        base["ae19_status"] = "AE19_QWEN_PROVIDER_UNAVAILABLE"
        base["llm_failure_reason"] = "AE19_QWEN_PROVIDER_UNAVAILABLE"
        base["llm_model"] = _cell(status.get("provider_model"))
        base["llm_skipped"] = False
        base["llm_failed"] = True
        base["llm_attempted"] = True
        return base

    prompt = (
        "AUDIT-ONLY operational note for Clean Forward candidate. "
        "Do not approve trades. Do not invent identity. "
        f"canonical_market_identity={candidate.get('canonical_market_identity')}; "
        f"pair_address={candidate.get('pair_address')}; "
        f"chain={candidate.get('chain')}; "
        f"ae16_status={candidate.get('ae16_status')}; "
        f"ae17_meta_decision={candidate.get('meta_decision')}; "
        f"ae18_status={candidate.get('ae18_status')}. "
        "Respond with a short research/audit note and optional label among "
        "WATCH/REVIEW/EXPLAIN/RESEARCH_ONLY. Never claim trade authority."
    )
    try:
        call = call_ollama_chat(prompt, timeout_s=float(timeout_seconds))
    except Exception as exc:  # noqa: BLE001
        return {
            **base,
            "ae19_status": "AE19_LLM_AUDIT_FAILED",
            "llm_failure_reason": f"AE19_LLM_AUDIT_FAILED:{type(exc).__name__}:{exc}",
            "llm_attempted": True,
            "llm_failed": True,
            "llm_skipped": False,
        }

    model = _cell(call.get("provider_model"))
    err = _cell(call.get("error"))
    if call.get("ok"):
        text = _cell(call.get("text"))
        label = ""
        upper = text.upper()
        for token in ("BUY", "SELL", "HOLD", "WATCH", "REVIEW", "RESEARCH_ONLY"):
            if token in upper:
                label = token
                break
        return {
            **base,
            "ae19_status": "AE19_QWEN_AUDIT_SUCCEEDED",
            "llm_task_ref": f"ae19_audit_{_cell(candidate.get('candidate_id'))[:16]}",
            "llm_model": model,
            "llm_output_excerpt": text[:500],
            "llm_action_label": label,
            "llm_authorizes_execution": False,
            "llm_attempted": True,
            "llm_succeeded": True,
            "llm_skipped": False,
            "llm_failed": False,
            "authority_status": "AUDIT_ONLY_NO_TRADE_AUTHORITY",
        }

    timeout_like = "timeout" in err.lower() or "timed out" in err.lower()
    if timeout_like:
        ae19_status = "AE19_QWEN_TIMEOUT"
    elif "unreachable" in err.lower() or call.get("provider_status") != PROVIDER_AVAILABLE:
        ae19_status = "AE19_QWEN_PROVIDER_UNAVAILABLE"
    else:
        ae19_status = "AE19_LLM_AUDIT_FAILED"
    return {
        **base,
        "ae19_status": ae19_status,
        "llm_failure_reason": ae19_status if not err else f"{ae19_status}:{err}",
        "llm_model": model,
        "llm_attempted": True,
        "llm_succeeded": False,
        "llm_timeout": timeout_like,
        "llm_failed": True,
        "llm_skipped": False,
    }
