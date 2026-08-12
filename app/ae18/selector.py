"""Interesting Clean Forward candidate selector for AE18 real Helius context."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ae18.models import AE18CandidateTarget
from app.clean_forward.price_source_identity import build_price_source_key
from app.consensus.serialization import read_csv_dicts

POSITIVE_TIERS: frozenset[str] = frozenset(
    {
        "TAB_XGB_RF_ALL3",
        "TAB_RF_ONLY",
        "TAB_XGB_ONLY",
        "RF_XGB_ONLY",
        "SINGLE_MODEL_ONLY",
    }
)

DEFAULT_MAX_CANDIDATES = 15
MIN_TARGET_SOLANA = 5


def _norm(s: str) -> str:
    return (s or "").strip().lower()


def _safe_float(val: Any) -> float | None:
    if val is None or val == "":
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def enrich_candidates_with_token_identity(
    candidates: list[AE18CandidateTarget],
    project_root: Path,
) -> list[AE18CandidateTarget]:
    """Enrich missing base/quote token addresses via chain+pair join only (never symbol)."""
    identity_map = _load_pair_token_map(project_root)
    enriched: list[AE18CandidateTarget] = []
    for c in candidates:
        key = (_norm(c.chain), _norm(c.pair_address))
        if key in identity_map:
            base, quote, liq = identity_map[key]
            if not c.base_token_address and base:
                c.base_token_address = base
            if not c.quote_token_address and quote:
                c.quote_token_address = quote
            if not hasattr(c, "_liquidity_usd"):
                setattr(c, "_liquidity_usd", liq)
            elif getattr(c, "_liquidity_usd", None) is None:
                setattr(c, "_liquidity_usd", liq)
        enriched.append(c)
    return enriched


def _load_pair_token_map(project_root: Path) -> dict[tuple[str, str], tuple[str, str, float | None]]:
    mapping: dict[tuple[str, str], tuple[str, str, float | None]] = {}
    paths = [
        project_root / "data" / "SeedTargets" / "clean_forward_curated_ready_targets_active.csv",
        project_root
        / "data"
        / "audits"
        / "ae16f_serving_safe_model_evidence_20260723_170902"
        / "data"
        / "ae16f_tiered_consensus_rows.csv",
    ]
    for path in paths:
        if not path.is_file():
            continue
        for row in read_csv_dicts(path):
            chain = _norm(row.get("chain") or row.get("provider_chain_id") or "")
            pair = _norm(
                row.get("pair_address")
                or row.get("provider_pair_address")
                or row.get("refetch_pair_id")
                or row.get("resolved_pair_address")
                or ""
            )
            if not chain or not pair:
                continue
            base = (row.get("base_token_address") or row.get("provider_base_token_address") or "").strip()
            quote = (row.get("quote_token_address") or row.get("provider_quote_token_address") or "").strip()
            liq = _safe_float(row.get("liquidity_usd") or row.get("liquidity"))
            if base or quote:
                mapping[(_norm(chain), _norm(pair))] = (base, quote, liq)
    return mapping


def attach_selection_fields_from_rows(
    candidates: list[AE18CandidateTarget],
    rows: list[dict[str, str]],
) -> list[AE18CandidateTarget]:
    """Attach consensus_tier / meta_score / liquidity onto candidates via identity join."""
    by_id: dict[str, dict[str, str]] = {}
    by_psk: dict[str, dict[str, str]] = {}
    by_pair: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        cid = (row.get("clean_forward_candidate_id") or row.get("row_id") or "").strip()
        psk = (row.get("price_source_key") or "").strip()
        chain = _norm(row.get("chain") or "")
        pair = _norm(row.get("pair_address") or "")
        if cid:
            by_id[cid] = row
        if psk:
            by_psk[_norm(psk)] = row
        if chain and pair:
            by_pair[(chain, pair)] = row

    for c in candidates:
        row = (
            by_id.get(c.clean_forward_candidate_id)
            or by_psk.get(_norm(c.price_source_key))
            or by_pair.get((_norm(c.chain), _norm(c.pair_address)))
        )
        if not row:
            continue
        setattr(c, "consensus_tier", (row.get("consensus_tier") or row.get("consensus_preview_tier") or "").strip())
        setattr(c, "meta_score", _safe_float(row.get("meta_score")))
        setattr(c, "meta_decision", (row.get("meta_decision") or "").strip())
        liq = _safe_float(row.get("liquidity_usd") or row.get("liquidity"))
        if liq is not None:
            setattr(c, "_liquidity_usd", liq)
        if not c.base_token_address:
            c.base_token_address = (row.get("base_token_address") or "").strip()
        if not c.quote_token_address:
            c.quote_token_address = (row.get("quote_token_address") or "").strip()
    return candidates


def candidate_identity_complete(candidate: AE18CandidateTarget) -> bool:
    if candidate.price_source_key and candidate.chain and candidate.pair_address:
        return True
    if candidate.chain and candidate.pair_address:
        return True
    if candidate.clean_forward_candidate_id and candidate.chain and candidate.pair_address:
        return True
    return False


def select_interesting_solana_candidates(
    candidates: list[AE18CandidateTarget],
    *,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    open_paper_pair_addresses: set[str] | None = None,
) -> tuple[list[AE18CandidateTarget], list[dict[str, Any]]]:
    """Select Solana candidates by priority. Returns (selected, selection_rows)."""
    open_pairs = {_norm(p) for p in (open_paper_pair_addresses or set())}
    scored: list[tuple[int, float, AE18CandidateTarget, str]] = []

    for c in candidates:
        if _norm(c.chain) != "solana":
            continue
        if not candidate_identity_complete(c):
            continue
        if not c.price_source_key and c.chain and c.pair_address:
            c.price_source_key = build_price_source_key(c.provider or "dexscreener", c.chain, c.pair_address)

        tier = str(getattr(c, "consensus_tier", "") or "")
        meta = getattr(c, "meta_score", None)
        meta_f = float(meta) if meta is not None else 0.0
        liq = getattr(c, "_liquidity_usd", None)
        liq_f = float(liq) if liq is not None else 0.0
        meta_decision = str(getattr(c, "meta_decision", "") or "")

        if _norm(c.pair_address) in open_pairs:
            priority = 0
            reason = "OPEN_PAPER_POSITION"
        elif tier in POSITIVE_TIERS:
            priority = 1
            reason = f"POSITIVE_TIER:{tier}"
        elif meta_f > 0 or meta_decision.upper().startswith("META_"):
            priority = 2
            reason = f"META_WATCH:score={meta_f}:decision={meta_decision or 'n/a'}"
        else:
            priority = 3
            reason = f"HIGH_LIQUIDITY_OR_UNIVERSE:liq={liq_f}"

        # Secondary sort: prefer higher meta, then higher liquidity
        secondary = meta_f * 1000.0 + liq_f
        scored.append((priority, -secondary, c, reason))

    scored.sort(key=lambda x: (x[0], x[1]))
    selected = [item[2] for item in scored[: max(0, max_candidates)]]
    rows: list[dict[str, Any]] = []
    for idx, (priority, _neg, c, reason) in enumerate(scored[: max(0, max_candidates)]):
        rows.append(
            {
                "selection_rank": idx + 1,
                "selection_priority": priority,
                "selection_reason": reason,
                "clean_forward_candidate_id": c.clean_forward_candidate_id,
                "price_source_key": c.price_source_key,
                "chain": c.chain,
                "pair_address": c.pair_address,
                "base_token_address": c.base_token_address,
                "quote_token_address": c.quote_token_address,
                "consensus_tier": getattr(c, "consensus_tier", ""),
                "meta_score": getattr(c, "meta_score", None),
                "liquidity_usd": getattr(c, "_liquidity_usd", None),
                "identity_complete": candidate_identity_complete(c),
                "identity_basis": _identity_basis(c),
            }
        )
    return selected, rows


def _identity_basis(c: AE18CandidateTarget) -> str:
    if c.price_source_key:
        return "PRICE_SOURCE_KEY"
    if c.chain and c.pair_address:
        return "CHAIN_PAIR_ADDRESS"
    if c.chain and c.base_token_address:
        return "CHAIN_TOKEN_ADDRESS"
    if c.clean_forward_candidate_id:
        return "CLEAN_FORWARD_CANDIDATE_ID"
    return "IDENTITY_UNRESOLVED"


def load_open_paper_pairs(project_root: Path) -> set[str]:
    """Best-effort open paper position pair addresses from paper_state.json if present."""
    path = project_root / "data" / "paper_state.json"
    if not path.is_file():
        return set()
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    pairs: set[str] = set()
    positions = data.get("positions") or data.get("open_positions") or []
    if isinstance(positions, dict):
        positions = list(positions.values())
    if not isinstance(positions, list):
        return set()
    for pos in positions:
        if not isinstance(pos, dict):
            continue
        status = str(pos.get("status") or pos.get("state") or "open").lower()
        if status not in {"open", "active", ""}:
            continue
        pair = (
            pos.get("pair_address")
            or pos.get("provider_pair_address")
            or pos.get("resolved_pair_address")
            or ""
        )
        if pair:
            pairs.add(str(pair))
    return pairs
