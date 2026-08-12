"""AE17 deterministic rule-based meta shadow scoring (no ML training)."""

from __future__ import annotations

import math
from typing import Any

from app.meta import AUTHORITY_STATUS, META_MODE
from app.meta.constants import (
    PAIR_CONCENTRATION_PENALTY_HIGH,
    PAIR_CONCENTRATION_PENALTY_WARNING,
    TIER_BASE_SCORES,
)
from app.meta.models import AE17MetaFeatureRow, AE17MetaShadowOutput


def is_numeric(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return False
        return True
    return False


def safe_float(value: Any, default: float | None = None) -> float | None:
    if not is_numeric(value):
        return default
    return float(value)


def clamp_score(value: Any) -> tuple[float | None, bool, str]:
    """Clamp to [0.0, 1.0]. None/NaN/non-numeric => (None, False, reason)."""
    if value is None:
        return None, False, "score_unavailable_null"
    if isinstance(value, bool):
        return None, False, "score_non_numeric_bool"
    if not isinstance(value, (int, float)):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None, False, "score_non_numeric"
    fv = float(value)
    if math.isnan(fv) or math.isinf(fv):
        return None, False, "score_nan_or_inf"
    clamped = min(1.0, max(0.0, fv))
    if clamped != fv:
        if fv > 1.0:
            return clamped, True, "clamped_high_to_1.0"
        return clamped, True, "clamped_low_to_0.0"
    return clamped, False, "within_bounds"


def resolve_pair_concentration_penalty(status: str | None) -> float:
    """Map concentration status string to a small deterministic penalty."""
    if not status:
        return 0.0
    s = str(status).upper()
    if "PAIR_CONCENTRATION_HIGH_RISK" in s or "HHI_HIGH_CONCENTRATION" in s:
        return PAIR_CONCENTRATION_PENALTY_HIGH
    if "PAIR_CONCENTRATION_WARNING" in s or "HHI_MODERATE_CONCENTRATION" in s:
        return PAIR_CONCENTRATION_PENALTY_WARNING
    return 0.0


def compute_ae17_meta_shadow_score(
    row: AE17MetaFeatureRow,
    *,
    pair_concentration_status: str = "PAIR_CONCENTRATION_OK",
    pair_concentration_penalty: float | None = None,
) -> AE17MetaShadowOutput:
    """Static heuristic combinator over AE16 consensus tier (+ bounded adjustments)."""
    tier = row.consensus_tier
    if tier is None or (isinstance(tier, str) and not tier.strip()):
        return AE17MetaShadowOutput(
            clean_forward_candidate_id=row.clean_forward_candidate_id,
            clean_forward_decision_input_id=row.clean_forward_decision_input_id,
            price_source_key=row.price_source_key,
            pair_address=row.pair_address,
            consensus_tier=None,
            pre_clamp_meta_score=None,
            meta_score=None,
            score_clamped=False,
            score_clamp_reason="missing_consensus_tier",
            meta_decision="META_UNAVAILABLE",
            meta_reason="missing consensus tier; meta_score remains null",
            meta_mode=META_MODE,
            context_score_weight=0.0
            if not row.context_feature_available
            else (safe_float(row.context_score_weight, 0.0) or 0.0),
            pair_concentration_status=pair_concentration_status or "PAIR_CONCENTRATION_OK",
            authority_status=AUTHORITY_STATUS,
            trade_authority=False,
            live_trading_ready=False,
            paper_demo_only=True,
            risk_override_authority=False,
        )

    tier_key = str(tier).strip().upper()
    if tier_key == "XGB_RF_ONLY":
        tier_key = "RF_XGB_ONLY"

    base_entry = TIER_BASE_SCORES.get(tier_key)
    if base_entry is None:
        return AE17MetaShadowOutput(
            clean_forward_candidate_id=row.clean_forward_candidate_id,
            clean_forward_decision_input_id=row.clean_forward_decision_input_id,
            price_source_key=row.price_source_key,
            pair_address=row.pair_address,
            consensus_tier=tier_key,
            pre_clamp_meta_score=None,
            meta_score=None,
            score_clamped=False,
            score_clamp_reason="unrecognized_consensus_tier",
            meta_decision="META_UNAVAILABLE",
            meta_reason=f"unrecognized consensus tier {tier_key}; meta_score remains null",
            meta_mode=META_MODE,
            context_score_weight=0.0,
            pair_concentration_status=pair_concentration_status or "PAIR_CONCENTRATION_OK",
            authority_status=AUTHORITY_STATUS,
            trade_authority=False,
            live_trading_ready=False,
            paper_demo_only=True,
            risk_override_authority=False,
        )

    base_score, decision, reason_frag = base_entry

    # Unavailable tiers keep null score (not 0.0).
    if base_score is None:
        return AE17MetaShadowOutput(
            clean_forward_candidate_id=row.clean_forward_candidate_id,
            clean_forward_decision_input_id=row.clean_forward_decision_input_id,
            price_source_key=row.price_source_key,
            pair_address=row.pair_address,
            consensus_tier=tier_key,
            pre_clamp_meta_score=None,
            meta_score=None,
            score_clamped=False,
            score_clamp_reason="unavailable_consensus_null_score",
            meta_decision=decision,
            meta_reason=f"{reason_frag}; consensus_tier={tier_key}",
            meta_mode=META_MODE,
            context_score_weight=0.0,
            pair_concentration_status=pair_concentration_status or "PAIR_CONCENTRATION_OK",
            authority_status=AUTHORITY_STATUS,
            trade_authority=False,
            live_trading_ready=False,
            paper_demo_only=True,
            risk_override_authority=False,
        )

    # Context adjustment only when available.
    ctx_w = 0.0
    if row.context_feature_available is True:
        parsed = safe_float(row.context_score_weight, 0.0)
        ctx_w = parsed if parsed is not None else 0.0
    else:
        ctx_w = 0.0

    # Pair concentration penalty (never grants authority).
    if pair_concentration_penalty is None:
        penalty = resolve_pair_concentration_penalty(pair_concentration_status)
    else:
        penalty_parsed = safe_float(pair_concentration_penalty, 0.0)
        penalty = penalty_parsed if penalty_parsed is not None else 0.0

    # Null-safe arithmetic: base_score is float here.
    pre_clamp = float(base_score) + float(ctx_w) - float(penalty)
    final_score, clamped, clamp_reason = clamp_score(pre_clamp)

    reason_parts = [
        f"tier={tier_key}",
        reason_frag,
        f"base={base_score}",
    ]
    if row.context_feature_available and ctx_w != 0.0:
        reason_parts.append(f"context_boost={ctx_w}")
    else:
        reason_parts.append("context_unavailable_weight=0.0")
    if penalty > 0.0:
        reason_parts.append(f"pair_concentration_penalty={penalty}")
    if row.consensus_reason:
        reason_parts.append(f"ae16_reason={row.consensus_reason}")

    return AE17MetaShadowOutput(
        clean_forward_candidate_id=row.clean_forward_candidate_id,
        clean_forward_decision_input_id=row.clean_forward_decision_input_id,
        price_source_key=row.price_source_key,
        pair_address=row.pair_address,
        consensus_tier=tier_key,
        pre_clamp_meta_score=pre_clamp,
        meta_score=final_score,
        score_clamped=bool(clamped),
        score_clamp_reason=clamp_reason,
        meta_decision=decision,
        meta_reason="; ".join(reason_parts),
        meta_mode=META_MODE,
        context_score_weight=ctx_w,
        pair_concentration_status=pair_concentration_status or "PAIR_CONCENTRATION_OK",
        authority_status=AUTHORITY_STATUS,
        trade_authority=False,
        live_trading_ready=False,
        paper_demo_only=True,
        risk_override_authority=False,
    )


def score_all_rows(
    rows: list[AE17MetaFeatureRow],
    *,
    pair_concentration_status: str = "PAIR_CONCENTRATION_OK",
    pair_concentration_penalty: float = 0.0,
) -> list[AE17MetaShadowOutput]:
    return [
        compute_ae17_meta_shadow_score(
            row,
            pair_concentration_status=pair_concentration_status,
            pair_concentration_penalty=pair_concentration_penalty,
        )
        for row in rows
    ]
