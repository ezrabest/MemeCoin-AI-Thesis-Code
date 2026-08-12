"""Transparent baseline consensus for AE6 decision records.

This module computes consensus families from available RF/XGB/TAB model score slots.
Missing model scores are expected in the current runtime and must never raise exceptions.

`NO_MODEL_CONSENSUS_AVAILABLE` is a valid, non-error state. Decision-record generation
must continue normally when no aligned runtime model scores exist for a candidate.
"""

from __future__ import annotations

from app.decision.types import ConsensusBlock, ConsensusFamily, ModelScoresBlock, ModelScoreSlot

NO_MODEL_CAVEAT = (
    "No aligned runtime RF/XGB/TAB model scores were available for this candidate."
)

_MODEL_KEYS = ("RF", "XGB", "TAB")


def _slot_available(slot: ModelScoreSlot) -> bool:
    return bool(slot.available and slot.score is not None)


def compute_consensus(model_scores: ModelScoresBlock) -> ConsensusBlock:
    """Compute consensus family from RF/XGB/TAB score slots.

    Tolerates all models missing — returns ``NO_MODEL_CONSENSUS_AVAILABLE`` without raising.
    """
    available: dict[str, ModelScoreSlot] = {}
    for key in _MODEL_KEYS:
        slot = getattr(model_scores, key)
        if _slot_available(slot):
            available[key] = slot

    available_count = len(available)
    vote_count = available_count

    if available_count == 0:
        return ConsensusBlock(
            available_model_count=0,
            vote_count=0,
            consensus_family=ConsensusFamily.NO_MODEL_CONSENSUS_AVAILABLE,
            consensus_strength="UNAVAILABLE",
            consensus_caveat=NO_MODEL_CAVEAT,
        )

    has_rf = "RF" in available
    has_xgb = "XGB" in available
    has_tab = "TAB" in available

    if has_tab and has_xgb and has_rf:
        family = ConsensusFamily.TAB_XGB_RF_ALL3
        strength = "STRONG"
    elif has_tab and has_rf and not has_xgb:
        family = ConsensusFamily.TAB_RF_ONLY
        strength = "MODERATE"
    elif has_tab and has_xgb and not has_rf:
        family = ConsensusFamily.TAB_XGB_ONLY
        strength = "MODERATE"
    elif has_xgb and has_rf and not has_tab:
        family = ConsensusFamily.XGB_RF_ONLY
        strength = "MODERATE"
    else:
        family = ConsensusFamily.SINGLE_MODEL_ONLY
        strength = "WEAK"

    caveat = None
    if available_count < 3:
        missing = [k for k in _MODEL_KEYS if k not in available]
        caveat = f"Partial model consensus; missing slots: {', '.join(missing)}."

    return ConsensusBlock(
        available_model_count=available_count,
        vote_count=vote_count,
        consensus_family=family,
        consensus_strength=strength,
        consensus_caveat=caveat,
    )
