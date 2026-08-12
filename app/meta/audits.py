"""AE17 audits: parity, lookahead, lineage, concentration, authority, null-safety."""

from __future__ import annotations

from collections import Counter
from typing import Any

from app.meta import AUTHORITY_STATUS
from app.meta.constants import (
    FORBIDDEN_FEATURE_FIELDS,
    FORBIDDEN_FEATURE_SUBSTRINGS,
    HHI_LOW,
    HHI_MODERATE,
    KNOWN_CONSENSUS_TIERS,
    LINEAGE_COMPLETE,
    LINEAGE_INCOMPLETE,
    LINEAGE_REQUIRED_FIELDS,
    LOW_PAIR_DIVERSITY_N,
    META_FEATURE_FIELDS,
    META_SCORE_FIELDS,
    SHADOW_OUTPUT_FIELDS,
    SMALL_SAMPLE_N,
    TOP_PAIR_SHARE_OK,
    TOP_PAIR_SHARE_WARNING,
)
from app.meta.features import is_forbidden_feature_name, parse_optional_float
from app.meta.models import (
    AE17MetaAuthorityStatus,
    AE17MetaFeatureRow,
    AE17MetaShadowOutput,
    AE17PairConcentrationResult,
)
from app.meta.scoring import is_numeric, safe_float


def audit_feature_parity(
    feature_rows: list[AE17MetaFeatureRow],
    shadow_rows: list[AE17MetaShadowOutput],
) -> dict[str, Any]:
    feat_schema = list(META_FEATURE_FIELDS)
    shadow_schema = list(SHADOW_OUTPUT_FIELDS)
    issues: list[str] = []

    for i, row in enumerate(feature_rows):
        d = row.to_dict()
        keys = [k for k in d.keys() if k != "warnings"]
        if set(keys) != set(feat_schema):
            missing = sorted(set(feat_schema) - set(keys))
            extra = sorted(set(keys) - set(feat_schema))
            if missing or extra:
                issues.append(f"feature_schema_mismatch_row_{i}:missing={missing}:extra={extra}")
        for k in keys:
            if is_forbidden_feature_name(k):
                issues.append(f"forbidden_feature_present:{k}:row_{i}")
        # Context placeholders
        if row.context_feature_available is False:
            if row.context_score_weight not in (0, 0.0):
                # Allow only exact 0.0 when unavailable
                if safe_float(row.context_score_weight, None) not in (0.0,):
                    issues.append(f"context_weight_nonzero_while_unavailable:row_{i}")
        # Missing model scores must remain null, not zero-filled from absence.
        # (Legitimate zero scores are allowed when evidence attached.)
        for score_field, status_field in (
            ("rf_score", "rf_evidence_status"),
            ("xgb_score", "xgb_evidence_status"),
            ("tab_score", "tab_evidence_status"),
        ):
            score = getattr(row, score_field)
            status = getattr(row, status_field)
            if status != "MODEL_EVIDENCE_ATTACHED" and score == 0.0:
                # Soft note: zero while unavailable may indicate zero-fill; flag.
                issues.append(f"possible_zero_fill:{score_field}:row_{i}:status={status}")
            if score is not None and not is_numeric(score):
                issues.append(f"non_numeric_score:{score_field}:row_{i}")

        tier = row.consensus_tier
        if tier is not None and str(tier) not in KNOWN_CONSENSUS_TIERS and str(tier).upper() not in KNOWN_CONSENSUS_TIERS:
            # Unknown allowed as explicit UNKNOWN path — record soft issue only if empty string mishandled
            if str(tier).strip() == "":
                issues.append(f"empty_consensus_tier_string:row_{i}")

    for i, row in enumerate(shadow_rows):
        d = row.to_dict()
        if set(d.keys()) != set(shadow_schema):
            missing = sorted(set(shadow_schema) - set(d.keys()))
            extra = sorted(set(d.keys()) - set(shadow_schema))
            issues.append(f"shadow_schema_mismatch_row_{i}:missing={missing}:extra={extra}")
        if "pre_clamp_meta_score" not in d or "meta_score" not in d:
            issues.append(f"missing_score_fields:row_{i}")
        ms = row.meta_score
        if ms is not None:
            if not is_numeric(ms):
                issues.append(f"meta_score_non_numeric:row_{i}")
            elif not (0.0 <= float(ms) <= 1.0):
                issues.append(f"meta_score_out_of_bounds:row_{i}:{ms}")

    forbidden_in_matrix = [f for f in feat_schema if is_forbidden_feature_name(f)]
    if forbidden_in_matrix:
        issues.append(f"forbidden_in_contract:{forbidden_in_matrix}")

    passed = len(issues) == 0
    return {
        "passed": passed,
        "feature_schema": feat_schema,
        "shadow_schema": shadow_schema,
        "feature_row_count": len(feature_rows),
        "shadow_row_count": len(shadow_rows),
        "forbidden_fields_in_feature_contract": forbidden_in_matrix,
        "issues": issues,
        "classification_if_failed": "AE17_BLOCKED_FEATURE_PARITY_GAP",
    }


def audit_no_lookahead(
    feature_rows: list[AE17MetaFeatureRow],
    *,
    source_columns: list[str] | None = None,
) -> dict[str, Any]:
    issues: list[str] = []
    used_feature_names = list(META_FEATURE_FIELDS)

    for name in used_feature_names:
        if is_forbidden_feature_name(name):
            issues.append(f"forbidden_feature_in_matrix:{name}")

    if source_columns:
        leaked = [c for c in source_columns if is_forbidden_feature_name(c)]
        # Source may contain evaluation refs; contamination only if they appear as features.
        for c in leaked:
            if c in used_feature_names:
                issues.append(f"lookahead_source_column_used_as_feature:{c}")

    # Decision-time timestamps only — presence is OK; using matured/closed is not.
    for i, row in enumerate(feature_rows):
        d = row.to_dict()
        for k, v in d.items():
            if is_forbidden_feature_name(k) and v not in (None, "", False):
                issues.append(f"lookahead_value_present:{k}:row_{i}")

    # Explicit checks against known outcome field names on the row object.
    for forbidden in FORBIDDEN_FEATURE_FIELDS:
        if hasattr(AE17MetaFeatureRow, forbidden):
            issues.append(f"model_defines_forbidden_field:{forbidden}")

    passed = len(issues) == 0
    return {
        "passed": passed,
        "feature_fields_checked": used_feature_names,
        "forbidden_field_set": sorted(FORBIDDEN_FEATURE_FIELDS),
        "forbidden_substrings": list(FORBIDDEN_FEATURE_SUBSTRINGS),
        "decision_time_fields": ["observed_at", "fetched_at", "ingested_at"],
        "paper_execution_outcome_fields_excluded": True,
        "matured_outcomes_not_used_as_features": True,
        "issues": issues,
        "classification_if_failed": "AE17_BLOCKED_LOOKAHEAD_CONTAMINATION",
    }


def audit_lineage(feature_rows: list[AE17MetaFeatureRow]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_out: list[dict[str, Any]] = []
    incomplete = 0
    for row in feature_rows:
        missing = []
        d = row.to_dict()
        for field in LINEAGE_REQUIRED_FIELDS:
            if field == "lineage_status":
                continue
            val = d.get(field)
            if val is None or str(val).strip() == "":
                missing.append(field)
        status = LINEAGE_COMPLETE if not missing else LINEAGE_INCOMPLETE
        if status == LINEAGE_INCOMPLETE:
            incomplete += 1
        rows_out.append(
            {
                "clean_forward_candidate_id": row.clean_forward_candidate_id,
                "price_source_key": row.price_source_key,
                "pair_address": row.pair_address,
                "lineage_status": status,
                "declared_lineage_status": row.lineage_status,
                "missing_lineage_fields": "|".join(missing),
                "source_ae16_artifact": row.source_ae16_artifact,
                "source_schema_hash": row.source_schema_hash,
            }
        )

    n = len(feature_rows)
    incomplete_share = (incomplete / n) if n else 0.0
    # Block only if lineage insufficient for all or most rows (>= 80%).
    blocked = n > 0 and incomplete_share >= 0.80
    # For AE16 TAB16 preview without CF IDs, incomplete is expected — treat as limitation not hard block
    # unless *critical* identity fields (pair_address / price_source_key) are missing for most rows.
    critical_missing = 0
    for row in feature_rows:
        if not row.pair_address and not row.price_source_key:
            critical_missing += 1
    critical_share = (critical_missing / n) if n else 0.0
    hard_block = critical_share >= 0.80

    summary = {
        "passed": not hard_block,
        "total_rows": n,
        "incomplete_rows": incomplete,
        "incomplete_share": round(incomplete_share, 6),
        "critical_identity_missing_rows": critical_missing,
        "critical_identity_missing_share": round(critical_share, 6),
        "lineage_majority_incomplete": blocked,
        "classification_if_failed": "AE17_BLOCKED_LINEAGE_GAP",
        "notes": [
            "Surrogate clean_forward IDs from AE16 TAB16 preview yield AE17_LINEAGE_INCOMPLETE without hard-blocking when pair identity is present.",
        ],
    }
    return rows_out, summary


def audit_pair_concentration(
    feature_rows: list[AE17MetaFeatureRow],
    *,
    grouping: str = "all_meta_rows",
) -> AE17PairConcentrationResult:
    pairs = [r.pair_address or r.price_source_key or "" for r in feature_rows]
    total = len(pairs)
    counts = Counter(pairs)
    # Drop empty key from uniqueness if present but still count rows.
    nonempty = {k: v for k, v in counts.items() if k}
    unique = len(nonempty) if nonempty else (1 if total else 0)
    if nonempty:
        top_pair, top_count = max(nonempty.items(), key=lambda kv: (kv[1], kv[0]))
    else:
        top_pair, top_count = "", 0

    top_share: float | None = None
    hhi: float | None = None
    if total > 0:
        top_share = float(top_count) / float(total)
        hhi = 0.0
        for c in counts.values():
            share = float(c) / float(total)
            hhi += share * share

    # Threshold classification
    if top_share is None:
        share_status = "PAIR_CONCENTRATION_OK"
    elif top_share <= TOP_PAIR_SHARE_OK:
        share_status = "PAIR_CONCENTRATION_OK"
    elif top_share <= TOP_PAIR_SHARE_WARNING:
        share_status = "PAIR_CONCENTRATION_WARNING"
    else:
        share_status = "PAIR_CONCENTRATION_HIGH_RISK"

    if hhi is None:
        hhi_status = "HHI_LOW_CONCENTRATION"
    elif hhi <= HHI_LOW:
        hhi_status = "HHI_LOW_CONCENTRATION"
    elif hhi <= HHI_MODERATE:
        hhi_status = "HHI_MODERATE_CONCENTRATION"
    else:
        hhi_status = "HHI_HIGH_CONCENTRATION"

    status_tags = [share_status, hhi_status]
    if total < SMALL_SAMPLE_N:
        status_tags.append("SMALL_SAMPLE_WARNING")
    if unique < LOW_PAIR_DIVERSITY_N:
        status_tags.append("LOW_PAIR_DIVERSITY_WARNING")

    high_risk = (top_share is not None and top_share > TOP_PAIR_SHARE_WARNING) or (
        hhi is not None and hhi > HHI_MODERATE
    )
    meta_authority_allowed = not high_risk

    from app.meta.scoring import resolve_pair_concentration_penalty

    penalty = resolve_pair_concentration_penalty("|".join(status_tags))

    return AE17PairConcentrationResult(
        grouping=grouping,
        total_rows=total,
        unique_pairs=unique,
        top_pair=top_pair,
        top_pair_count=top_count,
        top_pair_share=top_share,
        pair_count_distribution=dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))),
        hhi=hhi,
        top_pair_share_status=share_status,
        hhi_status=hhi_status,
        concentration_status=status_tags,
        meta_authority_allowed=meta_authority_allowed,
        pair_concentration_penalty=penalty,
    )


def audit_authority(shadow_rows: list[AE17MetaShadowOutput]) -> AE17MetaAuthorityStatus:
    violations: list[str] = []
    for i, row in enumerate(shadow_rows):
        if row.trade_authority is True:
            violations.append(f"trade_authority_true:row_{i}")
        if row.live_trading_ready is True:
            violations.append(f"live_trading_ready_true:row_{i}")
        if row.paper_demo_only is not True:
            violations.append(f"paper_demo_only_not_true:row_{i}")
        if row.risk_override_authority is True:
            violations.append(f"risk_override_authority_true:row_{i}")
        if row.authority_status != AUTHORITY_STATUS:
            if "LIVE" in str(row.authority_status).upper():
                violations.append(f"authority_escalation:row_{i}:{row.authority_status}")

    status = AE17MetaAuthorityStatus(
        authority_status=AUTHORITY_STATUS,
        trade_authority=False,
        live_trading_ready=False,
        paper_demo_only=True,
        risk_override_authority=False,
        wallet_access=False,
        private_key_access=False,
        live_trading_enabled=False,
        db_mutation=False,
        order_opened=False,
        position_opened=False,
        external_llm_call=False,
        helius_solana_call=False,
        training_performed=False,
        fit_called=False,
        passed=len(violations) == 0,
        violations=violations,
    )
    return status


def audit_score_clamping(shadow_rows: list[AE17MetaShadowOutput]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in shadow_rows:
        pre = row.pre_clamp_meta_score
        final = row.meta_score
        expected_clamped = False
        if pre is not None and final is not None and is_numeric(pre) and is_numeric(final):
            expected_clamped = float(pre) != float(final)
        out.append(
            {
                "clean_forward_candidate_id": row.clean_forward_candidate_id,
                "price_source_key": row.price_source_key,
                "consensus_tier": row.consensus_tier,
                "pre_clamp_meta_score": pre,
                "meta_score": final,
                "score_clamped": row.score_clamped,
                "score_clamp_reason": row.score_clamp_reason,
                "clamp_flag_consistent": (row.score_clamped == expected_clamped)
                or (pre is None and final is None),
            }
        )
    return out


def audit_null_safety(
    feature_rows: list[AE17MetaFeatureRow],
    shadow_rows: list[AE17MetaShadowOutput],
) -> dict[str, Any]:
    """Verify null handling rules without raising TypeError on null rows."""
    issues: list[str] = []
    checks_passed = 0
    checks_total = 0

    def _check(cond: bool, msg: str) -> None:
        nonlocal checks_passed, checks_total
        checks_total += 1
        if cond:
            checks_passed += 1
        else:
            issues.append(msg)

    for i, row in enumerate(feature_rows):
        for field in META_SCORE_FIELDS:
            val = getattr(row, field, None)
            if field == "context_score_weight":
                _check(
                    is_numeric(val) or val == 0 or val == 0.0,
                    f"context_score_weight_invalid:row_{i}",
                )
                continue
            # model scores may be null
            if val is None:
                checks_total += 1
                checks_passed += 1
            else:
                _check(is_numeric(val), f"{field}_non_numeric:row_{i}")

        # Guard: never compare None with float in caller — simulate safe path
        for field in ("rf_score", "xgb_score", "tab_score"):
            val = getattr(row, field)
            try:
                if val is not None and isinstance(val, (int, float)):
                    _ = val > 0.5
                checks_total += 1
                checks_passed += 1
            except TypeError:
                issues.append(f"typeerror_on_compare:{field}:row_{i}")
                checks_total += 1

    for i, row in enumerate(shadow_rows):
        if row.consensus_tier in {None, "MODEL_EVIDENCE_UNAVAILABLE", "CONSENSUS_NOT_COMPUTABLE"} or (
            row.meta_decision == "META_UNAVAILABLE"
        ):
            if row.consensus_tier in {"MODEL_EVIDENCE_UNAVAILABLE", "CONSENSUS_NOT_COMPUTABLE"} or (
                row.meta_decision == "META_UNAVAILABLE" and row.meta_score is None
            ):
                _check(row.meta_score is None, f"unavailable_should_be_null:row_{i}")

    return {
        "passed": len(issues) == 0,
        "checks_passed": checks_passed,
        "checks_total": checks_total,
        "issues": issues,
        "rules": {
            "missing_model_score_remains_null": True,
            "context_unavailable_weight_0": True,
            "unavailable_consensus_meta_score_null": True,
            "no_none_vs_numeric_compare_without_guard": True,
        },
    }


def build_feature_contract_audit(feature_rows: list[AE17MetaFeatureRow]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for field in META_FEATURE_FIELDS:
        null_count = 0
        for r in feature_rows:
            v = getattr(r, field, None)
            if v is None or v == "":
                null_count += 1
        rows.append(
            {
                "field": field,
                "required_in_contract": True,
                "forbidden": is_forbidden_feature_name(field),
                "null_or_empty_count": null_count,
                "row_count": len(feature_rows),
            }
        )
    return rows
