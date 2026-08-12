"""AE17 meta feature construction from AE16 consensus/evidence rows."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any

from app.meta.constants import (
    CONTEXT_MISSINGNESS_REASON,
    CONTEXT_STATUS_PENDING,
    FORBIDDEN_FEATURE_FIELDS,
    FORBIDDEN_FEATURE_SUBSTRINGS,
    LINEAGE_COMPLETE,
    LINEAGE_INCOMPLETE,
    LINEAGE_REQUIRED_FIELDS,
    META_FEATURE_FIELDS,
)
from app.meta.models import AE17MetaFeatureRow


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def schema_hash_for_columns(columns: list[str]) -> str:
    material = "|".join(columns)
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _blank(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() in {"nan", "none", "null"}


def _as_str(v: Any, default: str = "") -> str:
    if _blank(v):
        return default
    return str(v).strip()


def _as_bool(v: Any, default: bool = False) -> bool:
    if _blank(v):
        return default
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    return default


def parse_optional_float(v: Any) -> float | None:
    """Parse numeric score; missing/invalid => None (never coerced to 0)."""
    if _blank(v):
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        fv = float(v)
        if math.isnan(fv) or math.isinf(fv):
            return None
        return fv
    try:
        fv = float(str(v).strip())
    except (TypeError, ValueError):
        return None
    if math.isnan(fv) or math.isinf(fv):
        return None
    return fv


def is_forbidden_feature_name(name: str) -> bool:
    n = name.strip().lower()
    if n in {f.lower() for f in FORBIDDEN_FEATURE_FIELDS}:
        return True
    for frag in FORBIDDEN_FEATURE_SUBSTRINGS:
        if frag in n:
            return True
    return False


def filter_forbidden_columns(columns: list[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    removed: list[str] = []
    for c in columns:
        if is_forbidden_feature_name(c):
            removed.append(c)
        else:
            kept.append(c)
    return kept, removed


def _get(row: dict[str, Any], *keys: str, default: str = "") -> str:
    for k in keys:
        if k in row and not _blank(row.get(k)):
            return _as_str(row.get(k))
    return default


def _price_source_key(row: dict[str, Any]) -> str:
    existing = _get(row, "price_source_key")
    if existing:
        return existing
    provider = _get(row, "provider")
    chain = _get(row, "chain")
    pair = _get(row, "pair_address")
    if provider and chain and pair:
        return f"{provider}|{chain}|{pair}"
    return pair or ""


def _normalize_tier(raw: str | None) -> str | None:
    if _blank(raw):
        return None
    tier = str(raw).strip().upper()
    # Alias AE16 tiered_engine naming to AE17 canonical.
    if tier == "XGB_RF_ONLY":
        return "RF_XGB_ONLY"
    return tier


def _status_attached(status: str) -> bool:
    return status == "MODEL_EVIDENCE_ATTACHED"


def _derive_candidate_id(row: dict[str, Any], psk: str) -> str:
    existing = _get(row, "clean_forward_candidate_id")
    if existing:
        return existing
    # Deterministic surrogate — not a full AE15 candidate id; lineage marked incomplete.
    material = f"AE17_SURROGATE_CANDIDATE|{psk}|{_get(row, 'timestamp', 'observed_at', 'fetched_at')}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _derive_decision_input_id(row: dict[str, Any], candidate_id: str) -> str:
    existing = _get(row, "clean_forward_decision_input_id")
    if existing:
        return existing
    material = f"AE17_SURROGATE_DECISION_INPUT|{candidate_id}"
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _lineage_status(row_dict: dict[str, Any]) -> str:
    for field in LINEAGE_REQUIRED_FIELDS:
        if field == "lineage_status":
            continue
        if _blank(row_dict.get(field)):
            return LINEAGE_INCOMPLETE
    return LINEAGE_COMPLETE


def build_meta_feature_row_from_ae16(
    raw: dict[str, Any],
    *,
    source_artifact: str,
    source_schema_hash: str,
    warnings_out: list[str] | None = None,
) -> AE17MetaFeatureRow:
    """Map one AE16 consensus/evidence row into AE17MetaFeatureRow (null-safe)."""
    warnings: list[str] = []
    psk = _price_source_key(raw)

    # Scores — prefer TAB16 columns, fall back to TAB_score_for_consensus / tab_score.
    rf_score = parse_optional_float(raw.get("rf_score", raw.get("RF_score")))
    xgb_score = parse_optional_float(raw.get("xgb_score", raw.get("XGB_score")))
    tab_score = parse_optional_float(
        raw.get(
            "tab_score",
            raw.get(
                "TAB16_score",
                raw.get("TAB_score_for_consensus", raw.get("TAB_score")),
            ),
        )
    )

    # Explicit: never convert missing score to 0.
    for name, val, raw_key_hint in (
        ("rf_score", rf_score, "RF_score"),
        ("xgb_score", xgb_score, "XGB_score"),
        ("tab_score", tab_score, "TAB16_score"),
    ):
        raw_val = raw.get(name, raw.get(raw_key_hint))
        if not _blank(raw_val) and val is None:
            warnings.append(f"non_numeric_score_kept_null:{name}")
        if isinstance(raw_val, str) and raw_val.strip() == "0":
            # Legitimate zero score is allowed when present and parseable.
            pass

    rf_status = _get(raw, "rf_evidence_status", "RF_status", default="MODEL_EVIDENCE_UNAVAILABLE")
    xgb_status = _get(raw, "xgb_evidence_status", "XGB_status", default="MODEL_EVIDENCE_UNAVAILABLE")
    tab_status = _get(
        raw,
        "tab_evidence_status",
        "TAB16_status",
        "TAB_status",
        "consensus_tab_slot_status",
        default="MODEL_EVIDENCE_UNAVAILABLE",
    )

    rf_vote = _as_bool(raw.get("rf_vote", raw.get("RF_vote")), False)
    xgb_vote = _as_bool(raw.get("xgb_vote", raw.get("XGB_vote")), False)
    tab_vote = _as_bool(
        raw.get(
            "tab_vote",
            raw.get("TAB16_vote", raw.get("TAB_vote_for_consensus", raw.get("TAB_vote"))),
        ),
        False,
    )

    # Missing score must not count as a negative vote — if score missing, force vote false.
    if rf_score is None:
        rf_vote = False
    if xgb_score is None:
        xgb_vote = False
    if tab_score is None:
        tab_vote = False

    attached = sum(
        1
        for st, sc in (
            (rf_status, rf_score),
            (xgb_status, xgb_score),
            (tab_status, tab_score),
        )
        if _status_attached(st) and sc is not None
    )
    # Prefer AE16 attached_model_count when present and numeric.
    attached_raw = parse_optional_float(raw.get("attached_model_count"))
    if attached_raw is not None:
        attached = int(attached_raw)

    vote_count = sum(1 for v in (rf_vote, xgb_vote, tab_vote) if v)
    vote_raw = parse_optional_float(raw.get("model_vote_count", raw.get("true_vote_count")))
    if vote_raw is not None:
        vote_count = int(vote_raw)

    tier = _normalize_tier(
        _get(raw, "consensus_tier", "consensus_preview_tier", default="") or None
    )
    reason = _get(raw, "consensus_reason")
    if not reason and tier:
        reason = f"ae16_tier={tier}"

    # Context: AE18 not implemented in this pass — always placeholder unless already present.
    context_available = _as_bool(raw.get("context_feature_available"), False)
    if context_available and not _blank(raw.get("context_score_weight")):
        ctx_w = parse_optional_float(raw.get("context_score_weight"))
        if ctx_w is None:
            ctx_w = 0.0
            warnings.append("invalid_context_score_weight_defaulted_0")
        context_status = _get(raw, "context_status", default="AE17_CONTEXT_ATTACHED")
        context_reason = _get(raw, "context_missingness_reason", default="")
    else:
        context_available = False
        ctx_w = 0.0
        context_status = CONTEXT_STATUS_PENDING
        context_reason = CONTEXT_MISSINGNESS_REASON

    candidate_id = _derive_candidate_id(raw, psk)
    decision_id = _derive_decision_input_id(raw, candidate_id)

    pair = _get(raw, "pair_address")
    provider = _get(raw, "provider")
    chain = _get(raw, "chain")
    if not provider and "|" in psk:
        parts = psk.split("|")
        if len(parts) >= 3:
            provider, chain, pair = parts[0], parts[1], parts[2]

    observed = _get(raw, "observed_at", "timestamp")
    fetched = _get(raw, "fetched_at", "timestamp")
    ingested = _get(raw, "ingested_at", "timestamp")

    provisional = {
        "clean_forward_candidate_id": candidate_id,
        "clean_forward_decision_input_id": decision_id,
        "price_source_key": psk,
        "provider_payload_hash": _get(raw, "provider_payload_hash"),
        "provider_pair_url": _get(
            raw, "provider_pair_url", "selected_provider_pair_url"
        ),
        "pair_address": pair,
        "base_token_address": _get(raw, "base_token_address"),
        "quote_token_address": _get(raw, "quote_token_address"),
        "source_ae16_artifact": source_artifact,
        "source_schema_hash": source_schema_hash,
    }
    # Surrogate IDs without token/hash lineage count as incomplete.
    has_native_cf = not _blank(raw.get("clean_forward_candidate_id"))
    lineage = _lineage_status(provisional)
    if not has_native_cf:
        lineage = LINEAGE_INCOMPLETE
        warnings.append("surrogate_clean_forward_ids_used")

    for col in raw.keys():
        if is_forbidden_feature_name(col):
            warnings.append(f"forbidden_source_column_excluded:{col}")

    if warnings_out is not None:
        warnings_out.extend(warnings)

    return AE17MetaFeatureRow(
        clean_forward_candidate_id=candidate_id,
        clean_forward_decision_input_id=decision_id,
        price_source_key=psk,
        provider=provider,
        chain=chain,
        pair_address=pair,
        base_token_address=_get(raw, "base_token_address"),
        quote_token_address=_get(raw, "quote_token_address"),
        provider_pair_url=_get(raw, "provider_pair_url", "selected_provider_pair_url"),
        provider_payload_hash=_get(raw, "provider_payload_hash"),
        rf_evidence_status=rf_status or "MODEL_EVIDENCE_UNAVAILABLE",
        xgb_evidence_status=xgb_status or "MODEL_EVIDENCE_UNAVAILABLE",
        tab_evidence_status=tab_status or "MODEL_EVIDENCE_UNAVAILABLE",
        rf_score=rf_score,
        xgb_score=xgb_score,
        tab_score=tab_score,
        rf_vote=rf_vote,
        xgb_vote=xgb_vote,
        tab_vote=tab_vote,
        attached_model_count=attached,
        model_vote_count=vote_count,
        consensus_tier=tier,
        consensus_reason=reason,
        context_status=context_status,
        context_feature_available=context_available,
        context_missingness_reason=context_reason,
        context_score_weight=float(ctx_w) if ctx_w is not None else 0.0,
        observed_at=observed,
        fetched_at=fetched,
        ingested_at=ingested,
        source_ae16_artifact=source_artifact,
        source_schema_hash=source_schema_hash,
        lineage_status=lineage,
        warnings=warnings,
    )


def build_meta_feature_rows(
    consensus_rows: list[dict[str, Any]],
    *,
    source_artifact: str,
    source_columns: list[str] | None = None,
) -> tuple[list[AE17MetaFeatureRow], list[dict[str, Any]]]:
    """Build candidate-level meta feature rows; return rows + warning records."""
    cols = source_columns or (list(consensus_rows[0].keys()) if consensus_rows else [])
    schema_hash = schema_hash_for_columns(cols)
    rows: list[AE17MetaFeatureRow] = []
    warn_records: list[dict[str, Any]] = []
    for i, raw in enumerate(consensus_rows):
        local_warnings: list[str] = []
        row = build_meta_feature_row_from_ae16(
            raw,
            source_artifact=source_artifact,
            source_schema_hash=schema_hash,
            warnings_out=local_warnings,
        )
        rows.append(row)
        for w in local_warnings:
            warn_records.append(
                {
                    "row_index": i,
                    "price_source_key": row.price_source_key,
                    "warning": w,
                }
            )
    return rows, warn_records


def feature_matrix_dicts(rows: list[AE17MetaFeatureRow]) -> list[dict[str, Any]]:
    """Emit feature-matrix dicts limited to META_FEATURE_FIELDS (no forbidden cols)."""
    out: list[dict[str, Any]] = []
    for row in rows:
        d = row.to_dict()
        matrix = {k: d.get(k) for k in META_FEATURE_FIELDS}
        # Guard: strip any forbidden keys if somehow present.
        for k in list(matrix.keys()):
            if is_forbidden_feature_name(k):
                del matrix[k]
        out.append(matrix)
    return out


def load_ae16_consensus_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    rows = read_csv_dicts(path)
    columns = list(rows[0].keys()) if rows else []
    return rows, columns
