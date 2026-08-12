"""AE18 read-only input discovery for Clean Forward / AE17 / AE16 candidates."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.ae18.constants import (
    AE16_CONSENSUS_PATTERNS,
    AE17_FEATURE_PATTERNS,
    CURATED_TARGETS_PATH,
    KNOWN_AE17_ROOTS,
)
from app.ae18.models import AE18CandidateTarget, AE18InputDiscoveryResult
from app.clean_forward.price_source_identity import (
    build_price_source_key,
    resolve_selected_target_identity,
)
from app.consensus.serialization import read_csv_dicts

DISCOVERY_OK = "AE18_INPUTS_DISCOVERED"
DISCOVERY_MISSING = "AE18_BLOCKED_MISSING_CANDIDATE_INPUTS"


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _glob_under(root: Path, pattern: str) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(p for p in root.glob(pattern) if p.is_file())


def _score_path(path: Path) -> int:
    name = path.name.lower()
    score = 0
    if "ae17_meta_feature_rows" in name:
        score += 100
    if "ae17_real_meta_feature_matrix" in name:
        score += 90
    if "tiered_consensus" in name or "consensus_preview" in name:
        score += 70
    if "curated" in name:
        score += 50
    return score


def discover_candidate_inputs(
    project_root: Path,
    *,
    ae17_root: str | Path | None = None,
    ae16_root: str | Path | None = None,
) -> AE18InputDiscoveryResult:
    root = project_root.resolve()
    searched: list[str] = []
    found: list[dict[str, Any]] = []
    notes: list[str] = []

    search_roots: list[Path] = []
    for arg_root in (ae17_root, ae16_root):
        if arg_root:
            p = Path(arg_root)
            if not p.is_absolute():
                p = root / p
            search_roots.append(p)
            searched.append(_rel(p, root))

    for rel in KNOWN_AE17_ROOTS:
        p = root / rel
        if p not in search_roots:
            search_roots.append(p)
            searched.append(rel)

    audits = root / "data" / "audits"
    if audits.is_dir():
        for child in sorted(audits.iterdir()):
            if child.is_dir() and ("ae17" in child.name.lower() or "ae16" in child.name.lower()):
                if child not in search_roots:
                    search_roots.append(child)
                    searched.append(_rel(child, root))

    hits: list[Path] = []
    for sr in search_roots:
        for pat in AE17_FEATURE_PATTERNS + AE16_CONSENSUS_PATTERNS:
            for hit in _glob_under(sr, pat):
                hits.append(hit)
                found.append({"path": _rel(hit, root), "pattern": pat, "kind": "candidate_csv"})

    curated = root / CURATED_TARGETS_PATH
    if curated.is_file():
        hits.append(curated)
        found.append({"path": CURATED_TARGETS_PATH, "pattern": "curated_targets", "kind": "curated_csv"})
        searched.append(CURATED_TARGETS_PATH)

    if not hits:
        return AE18InputDiscoveryResult(
            status=DISCOVERY_MISSING,
            searched_roots=searched,
            found_artifacts=found,
            missing_artifacts=["ae17_feature_rows_or_ae16_consensus_or_curated_targets"],
            notes=["No AE17/AE16/curated candidate CSV discovered"],
        )

    best = max(hits, key=lambda p: (_score_path(p), p.stat().st_mtime))
    rows = read_csv_dicts(best)
    return AE18InputDiscoveryResult(
        status=DISCOVERY_OK,
        candidate_csv=_rel(best, root),
        candidate_count=len(rows),
        source_kind=_infer_source_kind(best),
        searched_roots=searched,
        found_artifacts=found,
        notes=notes,
    )


def _infer_source_kind(path: Path) -> str:
    name = path.name.lower()
    if "ae17" in name:
        return "ae17_feature_rows"
    if "consensus" in name or "tiered" in name:
        return "ae16_consensus_rows"
    if "curated" in name:
        return "curated_selected_targets"
    return "unknown"


def load_candidate_targets(project_root: Path, csv_path: str | Path) -> list[AE18CandidateTarget]:
    path = Path(csv_path)
    if not path.is_absolute():
        path = project_root / path
    rows = read_csv_dicts(path)
    kind = _infer_source_kind(path)
    targets: list[AE18CandidateTarget] = []
    for row in rows:
        t = _row_to_target(row, kind, str(csv_path))
        if t.clean_forward_candidate_id or t.price_source_key or t.pair_address:
            targets.append(t)
    return targets


def _row_to_target(row: dict[str, str], kind: str, source_artifact: str) -> AE18CandidateTarget:
    if kind == "curated_selected_targets":
        return _curated_row_to_target(row, source_artifact)
    return _consensus_row_to_target(row, source_artifact)


def _consensus_row_to_target(row: dict[str, str], source_artifact: str) -> AE18CandidateTarget:
    chain = (row.get("chain") or "").strip()
    pair = (row.get("pair_address") or "").strip()
    provider = (row.get("provider") or "dexscreener").strip()
    psk = (row.get("price_source_key") or "").strip()
    if not psk and chain and pair:
        psk = build_price_source_key(provider, chain, pair)
    return AE18CandidateTarget(
        clean_forward_candidate_id=(row.get("clean_forward_candidate_id") or row.get("row_id") or "").strip(),
        clean_forward_decision_input_id=(row.get("clean_forward_decision_input_id") or "").strip(),
        price_source_key=psk,
        provider=provider,
        chain=chain,
        pair_address=pair,
        base_token_address=(row.get("base_token_address") or "").strip(),
        quote_token_address=(row.get("quote_token_address") or "").strip(),
        combined_target_id=(row.get("combined_target_id") or "").strip(),
        provider_pair_url=(row.get("provider_pair_url") or "").strip(),
        provider_payload_hash=(row.get("provider_payload_hash") or "").strip(),
        observed_at=(row.get("observed_at") or row.get("timestamp") or row.get("event_timestamp") or "").strip(),
        fetched_at=(row.get("fetched_at") or "").strip(),
        ingested_at=(row.get("ingested_at") or "").strip(),
        whale_score=row.get("whale_score") or row.get("whale_score_asof"),
        source_artifact=source_artifact,
        lineage_status=(row.get("lineage_status") or "").strip(),
    )


def _curated_row_to_target(row: dict[str, str], source_artifact: str) -> AE18CandidateTarget:
    identity = resolve_selected_target_identity(row)
    psk = identity.get("price_source_key") or ""
    chain = identity.get("normalized_chain") or (row.get("chain") or "").strip()
    pair = identity.get("normalized_real_pair_address") or (row.get("provider_pair_address") or row.get("refetch_pair_id") or "").strip()
    return AE18CandidateTarget(
        clean_forward_candidate_id=(row.get("combined_target_id") or "").strip(),
        price_source_key=psk,
        provider="dexscreener",
        chain=chain,
        pair_address=pair,
        base_token_address=(row.get("provider_base_token_address") or "").strip(),
        quote_token_address=(row.get("provider_quote_token_address") or "").strip(),
        combined_target_id=(row.get("combined_target_id") or "").strip(),
        provider_pair_url=(row.get("provider_pair_url") or "").strip(),
        token_symbol=(row.get("provider_base_token_symbol") or "").strip(),
        token_name=(row.get("provider_base_token_name") or "").strip(),
        observed_at="",
        source_artifact=source_artifact,
        lineage_status="CURATED_SELECTED_TARGET",
    )
