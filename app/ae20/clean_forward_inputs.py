"""Clean Forward / Selected / canonical runtime identity inputs for AE20."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.ae20 import REQUIRED_CANONICAL_IDENTITY_FIELDS
from app.clean_forward.canonical_market_identity import resolve_canonical_market_identity
from app.clean_forward.provider_url_key import try_normalize_provider_pair_url_key
from app.consensus.serialization import read_csv_dicts


DEFAULT_IDENTITY_INDEX = Path("data/runtime/canonical_market_identity_index.jsonl")
DEFAULT_CURATED_TARGETS = Path(
    "data/SeedTargets/clean_forward_curated_ready_targets_active.csv"
)

FORBIDDEN_LEGACY_SOURCES = frozenset(
    {
        "market_snapshots",
        "legacy_market_snapshots",
        "Market Snapshot Feed",
        "market_snapshot_feed",
        "symbol_only",
        "symbol-only",
    }
)


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_canonical_identity_index(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_curated_selected_targets(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [dict(r) for r in read_csv_dicts(path)]


def classify_candidate_identity(row: dict[str, Any]) -> dict[str, Any]:
    """Validate required Clean Forward canonical identity fields. No silent fallback."""
    # Prefer already-resolved canonical fields; else resolve from row without
    # reading legacy market_snapshots.
    resolved = resolve_canonical_market_identity(row)
    provider_pair_url_exact = _cell(
        row.get("provider_pair_url_exact") or resolved.get("provider_pair_url_exact")
    )
    canonical_market_identity = _cell(
        row.get("canonical_market_identity") or resolved.get("canonical_market_identity")
    )
    normalized_key = _cell(
        row.get("normalized_provider_pair_url_key")
        or resolved.get("normalized_provider_pair_url_key")
    )
    if not normalized_key and provider_pair_url_exact:
        normalized_key = _cell(try_normalize_provider_pair_url_key(provider_pair_url_exact) or "")

    chain = _cell(row.get("chain") or resolved.get("chain"))
    pair_address = _cell(
        row.get("pair_address")
        or row.get("pair_address_derived")
        or resolved.get("pair_address_derived")
        or resolved.get("pair_address")
    )
    price_source_key = _cell(row.get("price_source_key") or resolved.get("price_source_key"))
    candidate_id = _cell(
        row.get("clean_forward_candidate_id")
        or row.get("candidate_id")
        or row.get("combined_target_id")
        or row.get("price_source_key")
        or canonical_market_identity
    )
    observed_at = _cell(
        row.get("observed_at")
        or row.get("fetched_at")
        or row.get("ingested_at")
        or row.get("provider_fetch_at")
        or row.get("last_market_update_at")
        or row.get("market_data_refreshed_at")
    )

    identity = {
        "provider_pair_url_exact": provider_pair_url_exact,
        "canonical_market_identity": canonical_market_identity,
        "normalized_provider_pair_url_key": normalized_key,
        "price_source_key": price_source_key,
        "chain": chain,
        "pair_address": pair_address,
        "candidate_id": candidate_id,
        "clean_forward_candidate_id": candidate_id,
        "observed_at": observed_at,
        "fetched_at": _cell(row.get("fetched_at") or row.get("provider_fetch_at")),
        "ingested_at": _cell(row.get("ingested_at") or row.get("last_identity_rebuild_at")),
    }

    missing = [f for f in REQUIRED_CANONICAL_IDENTITY_FIELDS if not _cell(identity.get(f))]
    if missing:
        status = (
            "AE20_CANDIDATE_IDENTITY_INCOMPLETE"
            if any(_cell(identity.get(f)) for f in REQUIRED_CANONICAL_IDENTITY_FIELDS)
            else "AE20_CLEAN_FORWARD_INPUT_FAILURE"
        )
        return {
            **identity,
            "identity_status": status,
            "identity_ok": False,
            "missing_identity_fields": missing,
            "legacy_source_used": False,
            "symbol_only_join_used": False,
            "market_snapshots_used": False,
        }

    return {
        **identity,
        "identity_status": "AE20_CLEAN_FORWARD_IDENTITY_OK",
        "identity_ok": True,
        "missing_identity_fields": [],
        "legacy_source_used": False,
        "symbol_only_join_used": False,
        "market_snapshots_used": False,
    }


def build_clean_forward_inputs(
    project_root: Path,
    *,
    max_candidates: int = 12,
    identity_index_path: Path | None = None,
    curated_path: Path | None = None,
    candidate_offset: int = 0,
    previous_identity_keys: set[str] | None = None,
    source_rows_override: list[dict[str, Any]] | None = None,
    source_name_override: str | None = None,
    source_path_override: str | None = None,
    refresh_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load Clean Forward / Selected / canonical runtime identity candidates only."""
    project_root = project_root.resolve()
    index_path = identity_index_path or (project_root / DEFAULT_IDENTITY_INDEX)
    curated = curated_path or (project_root / DEFAULT_CURATED_TARGETS)

    index_rows = load_canonical_identity_index(index_path)
    curated_rows = load_curated_selected_targets(curated)

    # AE20_PROVIDER_REFRESH_SOURCE_V1
    # Provider-refresh rows are allowed only when supplied explicitly by AE20
    # orchestration. They are still classified through the same canonical identity
    # validator. No lowercase/casefold/symbol-only/legacy matching is introduced.
    if source_rows_override is not None:
        selected_source_name = source_name_override or "clean_forward_provider_refresh"
        source_rows: list[tuple[str, dict[str, Any]]] = [
            (selected_source_name, dict(r)) for r in source_rows_override
        ]
    else:
        # AE20_STATIC_REPLAY_FIX_V1
        # Prefer runtime index (canonical identity already present), but never replay
        # the same first N rows on every duration cycle. Candidate diversity is part
        # of AE20 forward validation. This preserves exact identity; no lowercase,
        # casefold, symbol-only, legacy market_snapshots, or pair-chain join is used.
        source_rows = [
            ("canonical_market_identity_index", r) for r in index_rows
        ]
        # Supplement with curated Selected if index empty.
        if not source_rows:
            source_rows = [("clean_forward_curated_ready_targets_active", r) for r in curated_rows]

    source_rows_available = len(source_rows)
    offset_used = 0
    if source_rows_available > 0:
        offset_used = int(candidate_offset or 0) % source_rows_available
        source_rows = source_rows[offset_used:] + source_rows[:offset_used]

    previous_identity_keys = previous_identity_keys or set()
    selected_identity_keys: list[str] = []

    inputs: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for source_name, raw in source_rows:
        raw = dict(raw)
        if source_name == "clean_forward_provider_refresh":
            raw.update(_provider_refresh_exact_identity_overrides_v4(raw))
        classified = classify_candidate_identity(raw)
        row = {
            **classified,
            "input_source": source_name,
            "input_source_path": (
                str(source_path_override)
                if source_path_override
                else str(index_path if source_name.startswith("canonical") else curated)
            ),
            "source_of_truth": "CLEAN_FORWARD_SELECTED_CANONICAL_RUNTIME",
            "legacy_market_snapshots_used": False,
            "symbol": _cell(raw.get("base_token_symbol") or raw.get("symbol") or raw.get("provider_base_token_symbol")),
            "token_symbol": _cell(raw.get("base_token_symbol") or raw.get("provider_base_token_symbol")),
            "quote_symbol": _cell(raw.get("quote_token_symbol") or raw.get("provider_quote_token_symbol")),
            "price_usd": raw.get("price_usd"),
            "liquidity_usd": raw.get("liquidity_usd"),
            "market_activity_status": raw.get("market_activity_status"),
            "tradability_status": raw.get("tradability_status"),
            "whale_score": raw.get("whale_score"),
            "semantic_status": raw.get("semantic_status"),
            "raw_provider_pair_url": _cell(raw.get("provider_pair_url") or raw.get("provider_url")),
        }
        if classified["identity_ok"]:
            identity_key = _cell(
                row.get("provider_pair_url_exact")
                or row.get("canonical_market_identity")
                or row.get("normalized_provider_pair_url_key")
                or row.get("price_source_key")
                or row.get("candidate_id")
            )
            row["candidate_selection_offset_used"] = offset_used
            row["candidate_source_rows_available"] = source_rows_available
            row["candidate_repeated_from_previous_cycle"] = identity_key in previous_identity_keys
            inputs.append(row)
            selected_identity_keys.append(identity_key)
        else:
            row["candidate_selection_offset_used"] = offset_used
            row["candidate_source_rows_available"] = source_rows_available
            row["candidate_repeated_from_previous_cycle"] = False
            failures.append(row)
        if len(inputs) >= max_candidates:
            break

    return {
        "inputs": inputs,
        "failures": failures,
        "index_path": str(index_path.resolve()) if index_path.is_file() else None,
        "curated_path": str(curated.resolve()) if curated.is_file() else None,
        "index_row_count": len(index_rows),
        "curated_row_count": len(curated_rows),
        "selected_ok_count": len(inputs),
        "identity_failure_count": len(failures),
        "source_rows_available": source_rows_available,
        "candidate_selection_offset_used": offset_used,
        "provider_refresh_source_used": source_rows_override is not None,
        "provider_refresh_metadata": refresh_metadata or {},
        "selected_identity_count": len(set(selected_identity_keys)),
        "selected_identity_keys": selected_identity_keys,
        "new_identity_count_vs_previous_cycle": len(
            [k for k in set(selected_identity_keys) if k not in previous_identity_keys]
        ),
        "repeated_identity_count_vs_previous_cycle": len(
            [k for k in selected_identity_keys if k in previous_identity_keys]
        ),
        "candidate_turnover_rate_vs_previous_cycle": (
            len([k for k in set(selected_identity_keys) if k not in previous_identity_keys])
            / len(set(selected_identity_keys))
            if selected_identity_keys
            else 0.0
        ),
        "clean_forward_only": True,
        "legacy_sources_forbidden": sorted(FORBIDDEN_LEGACY_SOURCES),
        "market_snapshots_used_as_source_of_truth": False,
        "symbol_only_joins_used": False,
    }

def _ae20_cell_no_case_mutation_v4(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text in {"", "nan", "NaN", "None", "null"}:
        return ""
    return text


def _provider_refresh_exact_identity_overrides_v4(raw):
    """AE20_EXACT_REFRESH_PAIR_IDENTITY_V4

    Provider refresh rows may expose provider_pair_url / dexscreener_url in
    display-normalized form. Exact identity must preserve the provider-returned
    pair/pool id from pair_address or provider_pair_id. No lowercase/casefold
    matching is introduced.
    """
    chain = _ae20_cell_no_case_mutation_v4(
        raw.get("chain")
        or raw.get("chain_id")
        or raw.get("normalized_chain_id")
    )
    pair = _ae20_cell_no_case_mutation_v4(
        raw.get("pair_address")
        or raw.get("provider_pair_id")
        or raw.get("pairAddress")
    )
    if not chain or not pair:
        return {}

    exact_url = f"https://dexscreener.com/{chain}/{pair}"
    display_url = _ae20_cell_no_case_mutation_v4(
        raw.get("provider_pair_url")
        or raw.get("dexscreener_url")
    )

    return {
        "provider_pair_url_exact": exact_url,
        "raw_provider_pair_url": exact_url,
        "canonical_market_identity": exact_url,
        "candidate_id": exact_url,
        "clean_forward_candidate_id": exact_url,
        "pair_address": pair,
        "provider_pair_id": pair,
        "provider_pair_url_exact_source": "chain_plus_exact_pair_address_from_provider_refresh",
        "provider_pair_url_display": display_url,
    }

