"""AE16 tiered consensus engine over attached RF/XGB/TAB evidence only."""
from __future__ import annotations

from typing import Any

from app.consensus import CONSENSUS_ENGINE_VERSION, MODEL_FAMILIES, RESEARCH_ONLY_TIERS
from app.consensus.model_evidence import AttachmentResult, parse_numeric_score


def model_may_vote(attachment: AttachmentResult) -> bool:
    """A model may vote only when evidence is attached with a real numeric score."""
    if attachment.model_family not in MODEL_FAMILIES:
        return False
    if not attachment.evidence_attached:
        return False
    if attachment.attachment_status != "MODEL_EVIDENCE_ATTACHED":
        return False
    if attachment.score is None:
        return False
    # Guard: never treat missing-as-zero
    if isinstance(attachment.score, float) and attachment.score != attachment.score:
        return False
    try:
        float(attachment.score)
    except (TypeError, ValueError):
        return False
    return True


def assign_consensus_tier(*, rf_vote: bool, xgb_vote: bool, tab_vote: bool) -> tuple[str, str]:
    """Return (consensus_tier, consensus_reason)."""
    votes = {"RF": rf_vote, "XGB": xgb_vote, "TAB": tab_vote}
    vote_count = sum(1 for v in votes.values() if v)

    if vote_count == 0:
        return (
            "MODEL_EVIDENCE_UNAVAILABLE",
            "No RF/XGB/TAB evidence attached with numeric scores; consensus not computable from real votes",
        )

    has_rf, has_xgb, has_tab = rf_vote, xgb_vote, tab_vote

    if has_tab and has_xgb and has_rf:
        return ("TAB_XGB_RF_ALL3", "Tier 1: TAB, XGB, and RF all voting from attached evidence")
    if has_tab and has_rf and not has_xgb:
        return ("TAB_RF_ONLY", "Tier 2: TAB and RF voting; XGB unavailable or not voting")
    if has_tab and has_xgb and not has_rf:
        return (
            "TAB_XGB_ONLY",
            "Research-only: TAB and XGB voting; RF unavailable or not voting",
        )
    if has_xgb and has_rf and not has_tab:
        return (
            "XGB_RF_ONLY",
            "Research-only: XGB and RF voting; TAB unavailable or not voting",
        )
    if vote_count == 1:
        voter = next(k for k, v in votes.items() if v)
        return ("SINGLE_MODEL_ONLY", f"Only {voter} voting from attached evidence")

    return ("CONSENSUS_NOT_COMPUTABLE", "Unexpected vote pattern; consensus not computable")


def _bool_false() -> bool:
    return False


def build_consensus_decision(
    *,
    candidate: dict[str, Any],
    decision: dict[str, Any],
    attachments_by_family: dict[str, AttachmentResult],
) -> dict[str, Any]:
    rf = attachments_by_family.get("RF")
    xgb = attachments_by_family.get("XGB")
    tab = attachments_by_family.get("TAB")

    rf_vote = model_may_vote(rf) if rf else False
    xgb_vote = model_may_vote(xgb) if xgb else False
    tab_vote = model_may_vote(tab) if tab else False

    attached_model_count = sum(
        1
        for a in (rf, xgb, tab)
        if a is not None and a.evidence_attached and a.attachment_status == "MODEL_EVIDENCE_ATTACHED"
    )
    model_vote_count = sum(1 for v in (rf_vote, xgb_vote, tab_vote) if v)

    tier, reason = assign_consensus_tier(rf_vote=rf_vote, xgb_vote=xgb_vote, tab_vote=tab_vote)

    # Safety: never promote authority from research-only tiers or missing evidence.
    authority_status = "RESEARCH_SHADOW_ONLY"
    if tier in RESEARCH_ONLY_TIERS:
        authority_status = "RESEARCH_SHADOW_ONLY"
    if tier == "MODEL_EVIDENCE_UNAVAILABLE":
        authority_status = "RESEARCH_SHADOW_ONLY"

    def _score(a: AttachmentResult | None) -> float | None:
        if a is None or not a.evidence_attached:
            return None
        return a.score

    def _status(a: AttachmentResult | None) -> str:
        if a is None:
            return "MODEL_EVIDENCE_UNAVAILABLE"
        return a.attachment_status

    # live_trading_ready / paper_demo_only from candidate, but force safe defaults.
    live_ready = False
    paper_demo = True

    return {
        "clean_forward_candidate_id": candidate.get("clean_forward_candidate_id") or "",
        "clean_forward_decision_input_id": decision.get("clean_forward_decision_input_id") or "",
        "pair_address": candidate.get("pair_address") or "",
        "base_token_address": candidate.get("base_token_address") or "",
        "quote_token_address": candidate.get("quote_token_address") or "",
        "provider_pair_url": candidate.get("provider_pair_url") or "",
        "provider_payload_hash": candidate.get("provider_payload_hash") or "",
        "observed_at": candidate.get("observed_at") or "",
        "fetched_at": candidate.get("fetched_at") or "",
        "ingested_at": candidate.get("ingested_at") or "",
        "verification_status": candidate.get("verification_status") or "",
        "freshness_status": candidate.get("freshness_status") or "",
        "identity_status": candidate.get("identity_status") or "",
        "rf_evidence_status": _status(rf),
        "xgb_evidence_status": _status(xgb),
        "tab_evidence_status": _status(tab),
        "rf_score": _score(rf),
        "xgb_score": _score(xgb),
        "tab_score": _score(tab),
        "rf_vote": rf_vote,
        "xgb_vote": xgb_vote,
        "tab_vote": tab_vote,
        "model_vote_count": model_vote_count,
        "attached_model_count": attached_model_count,
        "consensus_tier": tier,
        "consensus_reason": reason,
        "consensus_engine_version": CONSENSUS_ENGINE_VERSION,
        "authority_status": authority_status,
        "trade_authority": False,
        "live_trading_ready": live_ready,
        "paper_demo_only": paper_demo,
        "wallet_authority": False,
        "risk_gate_override_authority": False,
    }


def build_all_consensus_decisions(
    *,
    candidates: list[dict[str, Any]],
    decision_by_candidate: dict[str, dict[str, Any]],
    attachments: list[AttachmentResult],
) -> list[dict[str, Any]]:
    by_cid: dict[str, dict[str, AttachmentResult]] = {}
    for a in attachments:
        by_cid.setdefault(a.clean_forward_candidate_id, {})[a.model_family] = a

    out: list[dict[str, Any]] = []
    for cand in candidates:
        cid = str(cand.get("clean_forward_candidate_id") or "")
        decision = decision_by_candidate.get(cid) or {}
        out.append(
            build_consensus_decision(
                candidate=cand,
                decision=decision,
                attachments_by_family=by_cid.get(cid, {}),
            )
        )
    return out


def summarize_consensus_tiers(decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for d in decisions:
        tier = str(d.get("consensus_tier") or "UNKNOWN")
        counts[tier] = counts.get(tier, 0) + 1
    rows = [
        {"consensus_tier": tier, "count": count, "share_of_candidates": round(count / max(len(decisions), 1), 6)}
        for tier, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    return rows


def make_synthetic_attachment(
    *,
    candidate_id: str,
    decision_id: str,
    family: str,
    score: float | None,
    attached: bool = True,
    status: str | None = None,
) -> AttachmentResult:
    """Test helper: build an attachment row without inventing production scores."""
    ok = attached and score is not None
    return AttachmentResult(
        clean_forward_candidate_id=candidate_id,
        clean_forward_decision_input_id=decision_id,
        pair_address="",
        base_token_address="",
        quote_token_address="",
        model_family=family,
        evidence_attached=ok,
        score=score if ok else None,
        rank=None,
        percentile_rank=None,
        source_artifact_path="synthetic://test" if ok else "",
        source_run_id="test",
        source_prediction_file="synthetic://test" if ok else "",
        source_model_artifact="",
        candidate_policy_id="test_policy" if ok else "",
        target_row_id="test_target" if ok else "",
        target_name="",
        target_version="",
        horizon="",
        filter_name="",
        exit_policy_id="",
        evidence_type="synthetic_test",
        attachment_status=status or ("MODEL_EVIDENCE_ATTACHED" if ok else "MODEL_EVIDENCE_UNAVAILABLE"),
        attachment_failure_reason="" if ok else "synthetic unavailable",
    )


# Re-export for callers that need score parsing in audits
__all__ = [
    "model_may_vote",
    "assign_consensus_tier",
    "build_consensus_decision",
    "build_all_consensus_decisions",
    "summarize_consensus_tiers",
    "make_synthetic_attachment",
    "parse_numeric_score",
]
