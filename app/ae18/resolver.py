"""AE18 explicit resolver — rejects symbol-only joins."""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from app.ae18.constants import (
    ALLOWED_JOIN_PATHS,
    RESOLVER_AMBIGUOUS,
    RESOLVER_LINKED,
    RESOLVER_SYMBOL_REJECTED,
    RESOLVER_UNRESOLVED,
)
from app.ae18.models import AE18CandidateTarget, AE18ResolverLink


def _link_id(*parts: str) -> str:
    payload = "|".join(parts)
    return hashlib.sha256(f"AE18_RESOLVER|{payload}|{uuid.uuid4()}".encode()).hexdigest()


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def resolve_text_to_candidate(
    text_item: dict[str, Any],
    candidates: list[AE18CandidateTarget],
    *,
    context_record_id: str,
    observed_at: str = "",
) -> AE18ResolverLink:
    """Link a text/news item to a candidate using explicit identity paths only.

    Symbol/name-only hints never produce a link — they emit SYMBOL_ONLY_JOIN_REJECTED.
    """
    text_id = str(text_item.get("text_item_id") or text_item.get("article_id") or uuid.uuid4())
    hints = _extract_identity_hints(text_item)
    symbol_only = _is_symbol_only_hints(hints)

    if symbol_only and not _has_explicit_identity(hints):
        return AE18ResolverLink(
            resolver_link_id=_link_id(text_id, "symbol_rejected"),
            context_record_id=context_record_id,
            clean_forward_candidate_id="",
            join_path="symbol_only",
            resolver_status=RESOLVER_SYMBOL_REJECTED,
            text_item_id=text_id,
            symbol_only_rejected=True,
            provenance_status="SYMBOL_ONLY_JOIN_REJECTED",
            observed_at=observed_at,
        )

    matches = _find_explicit_matches(hints, candidates)
    if not matches:
        return AE18ResolverLink(
            resolver_link_id=_link_id(text_id, "unresolved"),
            context_record_id=context_record_id,
            clean_forward_candidate_id="",
            join_path="",
            resolver_status=RESOLVER_UNRESOLVED,
            text_item_id=text_id,
            provenance_status="IDENTITY_UNRESOLVED",
            observed_at=observed_at,
        )

    if len(matches) > 1:
        best_path, best_candidate, confidence = matches[0]
        return AE18ResolverLink(
            resolver_link_id=_link_id(text_id, "ambiguous"),
            context_record_id=context_record_id,
            clean_forward_candidate_id=best_candidate.clean_forward_candidate_id,
            price_source_key=best_candidate.price_source_key,
            chain=best_candidate.chain,
            pair_address=best_candidate.pair_address,
            token_address=best_candidate.base_token_address,
            join_path=best_path,
            resolver_status=RESOLVER_AMBIGUOUS,
            resolver_confidence=confidence,
            ambiguous=True,
            text_item_id=text_id,
            provenance_status="AMBIGUOUS_MULTIPLE_CANDIDATES",
            observed_at=observed_at,
        )

    join_path, candidate, confidence = matches[0]
    return AE18ResolverLink(
        resolver_link_id=_link_id(text_id, candidate.clean_forward_candidate_id),
        context_record_id=context_record_id,
        clean_forward_candidate_id=candidate.clean_forward_candidate_id,
        price_source_key=candidate.price_source_key,
        chain=candidate.chain,
        pair_address=candidate.pair_address,
        token_address=candidate.base_token_address,
        join_path=join_path,
        resolver_status=RESOLVER_LINKED,
        resolver_confidence=confidence,
        text_item_id=text_id,
        provenance_status="EXPLICIT_IDENTITY_RESOLVED",
        observed_at=observed_at,
    )


def resolve_candidate_identity(
    candidate: AE18CandidateTarget,
    *,
    context_record_id: str,
) -> AE18ResolverLink:
    """Self-resolve a candidate's market identity (always explicit)."""
    join_path = ""
    if candidate.price_source_key:
        join_path = "price_source_key"
    elif candidate.chain and candidate.pair_address:
        join_path = "chain_pair_address"
    elif candidate.chain and candidate.base_token_address:
        join_path = "chain_token_address"
    elif candidate.clean_forward_candidate_id:
        join_path = "clean_forward_candidate_id"
    elif candidate.combined_target_id:
        join_path = "target_lineage_id"

    status = RESOLVER_LINKED if join_path in ALLOWED_JOIN_PATHS else RESOLVER_UNRESOLVED
    return AE18ResolverLink(
        resolver_link_id=_link_id(context_record_id, candidate.clean_forward_candidate_id),
        context_record_id=context_record_id,
        clean_forward_candidate_id=candidate.clean_forward_candidate_id,
        price_source_key=candidate.price_source_key,
        chain=candidate.chain,
        pair_address=candidate.pair_address,
        token_address=candidate.base_token_address,
        join_path=join_path,
        resolver_status=status,
        resolver_confidence=1.0 if status == RESOLVER_LINKED else None,
        provenance_status="CANDIDATE_SELF_IDENTITY",
    )


def _extract_identity_hints(item: dict[str, Any]) -> dict[str, str]:
    return {
        "price_source_key": str(item.get("price_source_key") or "").strip(),
        "chain": str(item.get("chain") or "").strip(),
        "pair_address": str(item.get("pair_address") or "").strip(),
        "token_address": str(item.get("token_address") or item.get("base_token_address") or "").strip(),
        "clean_forward_candidate_id": str(item.get("clean_forward_candidate_id") or "").strip(),
        "combined_target_id": str(item.get("combined_target_id") or item.get("target_id") or "").strip(),
        "symbol": str(item.get("symbol") or item.get("token_symbol") or "").strip(),
        "name": str(item.get("name") or item.get("token_name") or "").strip(),
    }


def _is_symbol_only_hints(hints: dict[str, str]) -> bool:
    return bool(hints.get("symbol") or hints.get("name"))


def _has_explicit_identity(hints: dict[str, str]) -> bool:
    return bool(
        hints.get("price_source_key")
        or (hints.get("chain") and hints.get("pair_address"))
        or (hints.get("chain") and hints.get("token_address"))
        or hints.get("clean_forward_candidate_id")
        or hints.get("combined_target_id")
    )


def _find_explicit_matches(
    hints: dict[str, str],
    candidates: list[AE18CandidateTarget],
) -> list[tuple[str, AE18CandidateTarget, float]]:
    matches: list[tuple[str, AE18CandidateTarget, float]] = []

    psk = hints.get("price_source_key", "")
    if psk:
        for c in candidates:
            if _norm(c.price_source_key) == _norm(psk):
                matches.append(("price_source_key", c, 1.0))

    chain = _norm(hints.get("chain", ""))
    pair = _norm(hints.get("pair_address", ""))
    if chain and pair:
        for c in candidates:
            if _norm(c.chain) == chain and _norm(c.pair_address) == pair:
                matches.append(("chain_pair_address", c, 0.95))

    token = _norm(hints.get("token_address", ""))
    if chain and token:
        for c in candidates:
            if _norm(c.chain) == chain and _norm(c.base_token_address) == token:
                matches.append(("chain_token_address", c, 0.9))

    cf_id = hints.get("clean_forward_candidate_id", "")
    if cf_id:
        for c in candidates:
            if c.clean_forward_candidate_id == cf_id:
                matches.append(("clean_forward_candidate_id", c, 1.0))

    tgt = hints.get("combined_target_id", "")
    if tgt:
        for c in candidates:
            if c.combined_target_id == tgt or c.clean_forward_candidate_id == tgt:
                matches.append(("target_lineage_id", c, 0.85))

    # Deduplicate by candidate id keeping highest confidence
    best: dict[str, tuple[str, AE18CandidateTarget, float]] = {}
    for path, cand, conf in matches:
        key = cand.clean_forward_candidate_id
        if key not in best or conf > best[key][2]:
            best[key] = (path, cand, conf)
    return sorted(best.values(), key=lambda x: -x[2])
