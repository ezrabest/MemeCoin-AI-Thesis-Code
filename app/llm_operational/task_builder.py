"""AE19 task input discovery and prompt construction."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from app.ae18.discovery import discover_candidate_inputs, load_candidate_targets
from app.consensus.serialization import read_csv_dicts
from app.llm_operational.lineage import extract_identity_spine, reject_symbol_only_join
from app.llm_operational.schema import (
    MISSED_WINNER_UNAVAILABLE,
    PROMPT_TEMPLATE_VERSION,
    TASK_CANDIDATE_MEMO,
    TASK_CONTEXT_SUMMARY,
    TASK_MISSED_WINNER_REVIEW,
    TASK_RISK_EXPLANATION,
    TASK_SEMANTIC_CONFLICT_REVIEW,
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def discover_ae18_context_root(project_root: Path, ae18_root: str | Path | None = None) -> dict[str, Any]:
    root = project_root.resolve()
    if ae18_root:
        p = Path(ae18_root)
        if not p.is_absolute():
            p = root / p
        csv_path = p / "data" / "ae18_context_records.csv"
        return {
            "ae18_root": str(p.resolve()),
            "context_csv": str(csv_path.resolve()) if csv_path.is_file() else "",
            "found": csv_path.is_file(),
        }
    audits = root / "data" / "audits"
    hits: list[Path] = []
    if audits.is_dir():
        for child in sorted(audits.iterdir(), reverse=True):
            if child.is_dir() and "ae18_context_intelligence" in child.name.lower():
                csv_path = child / "data" / "ae18_context_records.csv"
                if csv_path.is_file():
                    hits.append(csv_path)
    best = hits[0] if hits else None
    return {
        "ae18_root": str(best.parent.parent.resolve()) if best else "",
        "context_csv": str(best.resolve()) if best else "",
        "found": bool(best),
    }


def discover_outcome_evidence(project_root: Path) -> dict[str, Any]:
    """Locate forward/missed-winner outcome evidence if present."""
    root = project_root.resolve()
    patterns = (
        "**/ae12_missed_winners*.csv",
        "**/ae12_forward*outcomes*.csv",
        "**/missed_winners*.csv",
        "**/forward_outcomes*.csv",
    )
    hits: list[Path] = []
    search_roots = [root / "data" / "audits", root / "data"]
    for sr in search_roots:
        if not sr.is_dir():
            continue
        for pat in patterns:
            hits.extend(p for p in sr.glob(pat) if p.is_file())
    hits = sorted(set(hits), key=lambda p: p.stat().st_mtime, reverse=True)
    best = hits[0] if hits else None
    rows: list[dict[str, str]] = []
    if best:
        try:
            rows = read_csv_dicts(best)
        except OSError:
            rows = []
    return {
        "found": bool(best and rows),
        "path": str(best.resolve()) if best else "",
        "row_count": len(rows),
        "rows": rows[:500],
    }


def load_context_by_candidate(context_csv: str | Path) -> dict[str, list[dict[str, str]]]:
    path = Path(context_csv)
    if not path.is_file():
        return {}
    rows = read_csv_dicts(path)
    by_id: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        cid = (
            row.get("clean_forward_candidate_id")
            or row.get("candidate_id")
            or row.get("combined_target_id")
            or ""
        ).strip()
        if not cid:
            continue
        by_id.setdefault(cid, []).append(row)
    return by_id


def candidate_to_input_dict(candidate: Any, *, source_artifact: str = "") -> dict[str, Any]:
    if hasattr(candidate, "to_dict"):
        d = candidate.to_dict()
    elif isinstance(candidate, dict):
        d = dict(candidate)
    else:
        d = dict(getattr(candidate, "__dict__", {}))
    d["source_artifact"] = source_artifact or d.get("source_artifact") or ""
    return d


def discover_task_candidates(
    project_root: Path,
    *,
    ae17_root: str | Path | None = None,
    ae16_root: str | Path | None = None,
    ae18_root: str | Path | None = None,
    max_candidates: int = 20,
    fixture_candidates: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    root = project_root.resolve()
    if fixture_candidates is not None:
        limited = fixture_candidates[: max(0, max_candidates)]
        return {
            "status": "AE19_INPUTS_FIXTURE",
            "candidates": limited,
            "candidate_count": len(limited),
            "discovery": {"status": "fixture"},
            "ae18_context": discover_ae18_context_root(root, ae18_root),
            "outcome_evidence": discover_outcome_evidence(root),
            "context_by_candidate": {},
        }

    discovery = discover_candidate_inputs(root, ae17_root=ae17_root, ae16_root=ae16_root)
    candidates: list[dict[str, Any]] = []
    if discovery.status == "AE18_INPUTS_DISCOVERED" and discovery.candidate_csv:
        targets = load_candidate_targets(root, discovery.candidate_csv)
        for t in targets[: max(0, max_candidates)]:
            candidates.append(candidate_to_input_dict(t, source_artifact=discovery.candidate_csv))

    ae18_info = discover_ae18_context_root(root, ae18_root)
    context_by = load_context_by_candidate(ae18_info["context_csv"]) if ae18_info.get("found") else {}
    outcome = discover_outcome_evidence(root)

    return {
        "status": "AE19_INPUTS_DISCOVERED" if candidates else "AE19_INPUTS_MISSING",
        "candidates": candidates,
        "candidate_count": len(candidates),
        "discovery": discovery.to_dict() if hasattr(discovery, "to_dict") else {
            "status": discovery.status,
            "candidate_csv": discovery.candidate_csv,
            "candidate_count": discovery.candidate_count,
        },
        "ae18_context": ae18_info,
        "outcome_evidence": {k: v for k, v in outcome.items() if k != "rows"},
        "outcome_rows": outcome.get("rows") or [],
        "context_by_candidate": context_by,
    }


def _safe_bundle(candidate: dict[str, Any], context_rows: list[dict[str, str]], spine: dict[str, Any]) -> dict[str, Any]:
    return {
        "identity": {
            "clean_forward_candidate_id": spine.get("clean_forward_candidate_id"),
            "decision_input_id": spine.get("decision_input_id"),
            "price_source_key": spine.get("price_source_key"),
            "provider_pair_url_exact": spine.get("provider_pair_url_exact"),
            "canonical_market_identity": spine.get("canonical_market_identity"),
            "normalized_provider_pair_url_key": spine.get("normalized_provider_pair_url_key"),
            "pair_address": spine.get("pair_address"),
            "chain": spine.get("chain"),
            "base_token_address": spine.get("base_token_address"),
            "quote_token_address": spine.get("quote_token_address"),
            "symbol_pair_display_only": spine.get("symbol_pair"),
            "identity_status": spine.get("identity_status"),
            "resolver_status": spine.get("resolver_status"),
        },
        "model_evidence": {
            "whale_score_pool_flow_proxy": candidate.get("whale_score"),
            "lineage_status": candidate.get("lineage_status"),
            "source_artifact": candidate.get("source_artifact"),
        },
        "consensus": {
            "note": "Use only provided consensus refs; do not invent tiers.",
            "combined_target_id": candidate.get("combined_target_id"),
        },
        "meta": {
            "note": "Meta decision fields if present on candidate only.",
            "meta_score": candidate.get("meta_score"),
            "meta_decision": candidate.get("meta_decision"),
        },
        "context": [
            {
                "context_family": r.get("context_family"),
                "context_status": r.get("context_status"),
                "source_name": r.get("source_name"),
                "available": r.get("available"),
                "missingness_reason": r.get("missingness_reason"),
                "resolver_status": r.get("resolver_status"),
                "whale_signal_type": r.get("whale_signal_type"),
                "provenance_status": r.get("provenance_status"),
            }
            for r in context_rows[:20]
        ],
        "missingness_provenance": {
            "identity_status": spine.get("identity_status"),
            "context_row_count": len(context_rows),
        },
        "constraints": {
            "no_trade_authority": True,
            "no_live_approval": True,
            "no_risk_override": True,
            "no_wallet_access": True,
            "no_identity_invention": True,
            "symbol_pair_is_display_only": True,
            "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        },
    }


def build_prompt_for_task(
    task_type: str,
    candidate: dict[str, Any],
    *,
    context_rows: list[dict[str, str]] | None = None,
    outcome_row: dict[str, Any] | None = None,
    outcome_available: bool = False,
) -> dict[str, Any]:
    spine = extract_identity_spine(candidate)
    symbol_check = reject_symbol_only_join(join_key_claimed=None, candidate=candidate)
    ctx = context_rows or []
    bundle = _safe_bundle(candidate, ctx, spine)

    if task_type == TASK_CANDIDATE_MEMO:
        instruction = (
            "Write a CANDIDATE_MEMO summarizing market identity, model evidence, consensus/meta if present, "
            "context evidence, and missingness/provenance. Do NOT recommend live trading. "
            "Use tags: WATCH, REVIEW, EXPLAIN, RESEARCH_ONLY, NOT_TRADE_AUTHORITY."
        )
    elif task_type == TASK_RISK_EXPLANATION:
        instruction = (
            "Write a RISK_EXPLANATION covering market risk, liquidity/activity risk, missing context, "
            "unresolved/ambiguous resolver status, model disagreement, consensus/meta limitations, "
            "and scam/reputation/semantic risk if present. Do NOT override any gate. Tag: RISK_NOTE."
        )
    elif task_type == TASK_MISSED_WINNER_REVIEW:
        if not outcome_available or not outcome_row:
            return {
                "prompt_text": "",
                "prompt_text_hash": "",
                "prompt_template_version": PROMPT_TEMPLATE_VERSION,
                "spine": spine,
                "symbol_check": symbol_check,
                "missed_winner_status": MISSED_WINNER_UNAVAILABLE,
                "input_unavailable": True,
                "bundle": bundle,
            }
        bundle["outcome_evidence"] = {
            k: outcome_row.get(k)
            for k in (
                "max_return",
                "forward_return",
                "horizon",
                "was_traded",
                "clean_forward_candidate_id",
                "pair_address",
                "price_source_key",
            )
            if k in outcome_row
        }
        instruction = (
            "Write a MISSED_WINNER_REVIEW using ONLY the provided forward/outcome evidence. "
            "Do not fabricate future performance. RESEARCH_ONLY / PAPER_DEMO_OBSERVATION."
        )
    elif task_type == TASK_SEMANTIC_CONFLICT_REVIEW:
        instruction = (
            "Write a SEMANTIC_CONFLICT_REVIEW identifying conflicts between market evidence, model evidence, "
            "consensus tier, meta decision, AE18 context, RSS/news, reputation/scam, semantic context, "
            "whale context, and missingness/provenance. Unresolved conflicts must remain unresolved. "
            "Tag: REVIEW."
        )
    elif task_type == TASK_CONTEXT_SUMMARY:
        instruction = (
            "Write a CONTEXT_SUMMARY of AE18 context intelligence: Helius/Solana availability, RSS/news, "
            "reputation/scam, semantic context, resolver links, unresolved/ambiguous links, "
            "whale_score as pool-flow proxy, wallet-level whale evidence if available, missingness/provenance. "
            "Tag: CONTEXT_SUMMARY."
        )
    else:
        instruction = "Produce an AUDIT-oriented operational note. RESEARCH_ONLY / NOT_TRADE_AUTHORITY."

    prompt_text = (
        f"Task type: {task_type}\n"
        f"Instruction: {instruction}\n"
        f"Evidence bundle JSON:\n{json.dumps(bundle, default=str, ensure_ascii=True)[:12000]}"
    )
    return {
        "prompt_text": prompt_text,
        "prompt_text_hash": sha256_text(prompt_text),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
        "spine": spine,
        "symbol_check": symbol_check,
        "missed_winner_status": "",
        "input_unavailable": False,
        "bundle": bundle,
        "outcome_row": outcome_row,
    }


def match_outcome_for_candidate(
    candidate: dict[str, Any],
    outcome_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    spine = extract_identity_spine(candidate)
    keys = {
        "psk": (spine.get("price_source_key") or "").lower(),
        "pair": (spine.get("pair_address") or "").lower(),
        "cid": (spine.get("clean_forward_candidate_id") or spine.get("candidate_id") or "").lower(),
    }
    for row in outcome_rows:
        row_psk = (row.get("price_source_key") or "").strip().lower()
        row_pair = (row.get("pair_address") or "").strip().lower()
        row_cid = (
            row.get("clean_forward_candidate_id")
            or row.get("candidate_id")
            or row.get("combined_target_id")
            or ""
        ).strip().lower()
        if keys["psk"] and row_psk and keys["psk"] == row_psk:
            return row
        if keys["cid"] and row_cid and keys["cid"] == row_cid:
            return row
        if keys["pair"] and row_pair and keys["pair"] == row_pair and (row.get("chain") or "").lower() == (
            spine.get("chain") or ""
        ).lower():
            return row
    return None
