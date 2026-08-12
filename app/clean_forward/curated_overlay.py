"""AE16D — Feature-flagged curated Clean Forward collector overlay.

Default OFF. Does not fetch on import. Does not mutate trader.db.
When disabled, callers must use the existing search-based Clean Forward path.
When enabled, exact DexScreener pair refetch over the approved curated ready list.
"""
from __future__ import annotations

import csv
import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# Feature flags (default OFF) — reading env only; no I/O, no network
# ---------------------------------------------------------------------------

FLAG_USE_CURATED = "CLEAN_FORWARD_USE_CURATED_TARGETS"
FLAG_CURATED_PATH = "CLEAN_FORWARD_CURATED_TARGETS_PATH"

DEFAULT_CURATED_READY_PATH = Path(
    "data/audits/ae16c_rejected_target_recovery_20260723_130202/"
    "data/ae16c_clean_forward_candidate_ready_targets_recovered.csv"
)

# Explicit exclusion from AE16C-R still-rejected set
EXPLICITLY_EXCLUDED_PAIR_IDS = {
    "mew1gqwj3nexg2qgeriku7fafj79phvqvrequzscpp5",
}

SEMANTIC_PENDING = "PENDING_SYSTEM_CLASSIFICATION"

REQUIRED_COLUMNS = (
    "combined_target_id",
    "chain",
    "provider_pair_address",
    "provider_base_token_address",
    "provider_quote_token_address",
)


def curated_targets_enabled(environ: dict[str, str] | None = None) -> bool:
    """Return True only when curated overlay flag is explicitly enabled."""
    env = environ if environ is not None else os.environ
    raw = str(env.get(FLAG_USE_CURATED, "false") or "false").strip().lower()
    return raw in {"1", "true", "yes", "y", "on"}


def curated_targets_path(environ: dict[str, str] | None = None) -> Path:
    """Resolve curated CSV path (override or default). Does not require exists()."""
    env = environ if environ is not None else os.environ
    override = str(env.get(FLAG_CURATED_PATH, "") or "").strip()
    if override:
        return Path(override)
    return DEFAULT_CURATED_READY_PATH


def _cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_bool(value: Any) -> bool:
    return _cell(value).lower() in {"1", "true", "yes", "y"}


def _norm_chain(chain: str) -> str:
    return _cell(chain).lower()


def _is_non_evm(chain: str) -> bool:
    return _norm_chain(chain) in {"solana", "xrpl"}


def _pair_id_excluded(pair_id: str) -> bool:
    pid = _cell(pair_id)
    if not pid:
        return False
    return pid.lower() in {x.lower() for x in EXPLICITLY_EXCLUDED_PAIR_IDS}


def _still_rejected_status(row: dict[str, str]) -> bool:
    for key in ("recovery_status", "acceptance_status"):
        val = _cell(row.get(key)).upper()
        if val.startswith("STILL_REJECTED") or "UNRESOLVED" in val:
            return True
    return False


def validate_curated_path(
    path: Path,
    *,
    explicit_validation: bool = False,
) -> dict[str, Any]:
    """Check curated path without raising on missing file (runtime-safe).

    Returns a status dict. Never raises for missing path.
    """
    result: dict[str, Any] = {
        "path": str(path).replace("\\", "/"),
        "path_exists": False,
        "readable": False,
        "columns_ok": False,
        "missing_columns": [],
        "error": "",
        "classification_hint": "",
        "ok_for_load": False,
    }
    try:
        exists = path.exists()
    except OSError as exc:
        result["error"] = f"exists_check_failed:{type(exc).__name__}:{exc}"
        result["classification_hint"] = "AE16D_BLOCKED_CURATED_INPUT_MISSING"
        return result

    result["path_exists"] = bool(exists)
    if not exists:
        result["error"] = "curated_path_missing"
        result["classification_hint"] = "AE16D_BLOCKED_CURATED_INPUT_MISSING"
        if not explicit_validation:
            warnings.warn(
                f"CLEAN_FORWARD curated targets path missing: {path} "
                f"(falling back to non-curated collector behavior)",
                UserWarning,
                stacklevel=2,
            )
        return result

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = list(reader.fieldnames or [])
            result["readable"] = True
    except OSError as exc:
        result["error"] = f"unreadable:{type(exc).__name__}:{exc}"
        result["classification_hint"] = "AE16D_BLOCKED_CURATED_INPUT_MISSING"
        return result

    missing = [c for c in REQUIRED_COLUMNS if c not in fields]
    # provider_url OR provider_pair_url required conceptually — checked per-row
    result["missing_columns"] = missing
    result["columns_ok"] = len(missing) == 0
    if missing:
        result["error"] = f"missing_columns:{','.join(missing)}"
        result["classification_hint"] = "AE16D_BLOCKED_CURATED_INPUT_MISSING"
        return result

    result["ok_for_load"] = True
    return result


def load_curated_ready_targets(
    path: Path | None = None,
    *,
    environ: dict[str, str] | None = None,
    explicit_validation: bool = False,
) -> dict[str, Any]:
    """Load and filter curated ready targets. No network.

    Returns:
      {
        loaded_rows, accepted_rows, excluded_rows,
        path_status, error, blocked_classification
      }
    """
    target_path = path if path is not None else curated_targets_path(environ)
    path_status = validate_curated_path(target_path, explicit_validation=explicit_validation)
    out: dict[str, Any] = {
        "path": str(target_path).replace("\\", "/"),
        "path_status": path_status,
        "loaded_rows": [],
        "accepted_rows": [],
        "excluded_rows": [],
        "error": path_status.get("error") or "",
        "blocked_classification": "",
    }
    if not path_status.get("ok_for_load"):
        if explicit_validation:
            out["blocked_classification"] = "AE16D_BLOCKED_CURATED_INPUT_MISSING"
        return out

    with target_path.open("r", encoding="utf-8-sig", newline="") as f:
        raw_rows = list(csv.DictReader(f))

    accepted: list[dict[str, str]] = []
    excluded: list[dict[str, Any]] = []

    for raw in raw_rows:
        row = {k: _cell(v) for k, v in raw.items()}
        # Preserve semantic pending always
        row["semantic_status"] = SEMANTIC_PENDING

        reason = ""
        if _still_rejected_status(row):
            reason = "still_rejected_or_unresolved_status"
        elif "clean_forward_candidate_ready" in raw and not _as_bool(
            row.get("clean_forward_candidate_ready")
        ):
            reason = "clean_forward_candidate_ready_false"
        elif not row.get("provider_pair_address"):
            reason = "missing_provider_pair_address"
        elif not (row.get("provider_chain_id") or row.get("chain")):
            reason = "missing_chain"
        elif not (row.get("provider_url") or row.get("provider_pair_url")):
            # Synthesize DexScreener URL from authoritative provider pair id (exact casing).
            # Do not invent pair ids — only format URL when chain + provider_pair_address exist.
            synth_chain = row.get("provider_chain_id") or row.get("chain")
            synth_pair = row.get("provider_pair_address")
            if synth_chain and synth_pair:
                synthesized = f"https://dexscreener.com/{_norm_chain(synth_chain)}/{synth_pair}"
                row["provider_url"] = synthesized
                row["provider_pair_url"] = synthesized
            else:
                reason = "missing_provider_url"
        if reason:
            pass  # fall through
        elif not row.get("provider_base_token_address"):
            reason = "missing_provider_base_token_address"
        elif not row.get("provider_quote_token_address"):
            reason = "missing_provider_quote_token_address"
        elif _pair_id_excluded(row.get("provider_pair_address", "")) or _pair_id_excluded(
            row.get("refetch_pair_id", "")
        ) or _pair_id_excluded(row.get("user_supplied_pair_address", "")):
            reason = "explicitly_excluded_pair_id"

        if reason:
            excluded.append({**row, "exclusion_reason": reason})
            continue

        # Authoritative identity from provider fields
        chain = row.get("provider_chain_id") or row.get("chain")
        pair = row.get("provider_pair_address")  # exact casing preserved
        normalized = {
            **row,
            "chain": chain,
            "provider_chain_id": row.get("provider_chain_id") or chain,
            "pair_address": pair,
            "provider_pair_address": pair,
            "base_token_address": row.get("provider_base_token_address"),
            "quote_token_address": row.get("provider_quote_token_address"),
            "provider_pair_url": row.get("provider_url") or row.get("provider_pair_url"),
            "semantic_status": SEMANTIC_PENDING,
            # Never create system_semantic_label from seed_collection
            "system_semantic_label": "",
        }
        accepted.append(normalized)

    out["loaded_rows"] = [{k: _cell(v) for k, v in r.items()} for r in raw_rows]
    out["accepted_rows"] = accepted
    out["excluded_rows"] = excluded
    return out


def _addresses_match(chain: str, expected: str, actual: str) -> bool:
    exp, act = _cell(expected), _cell(actual)
    if not exp or not act:
        return False
    if _is_non_evm(chain):
        return exp == act
    return exp.lower() == act.lower()


def _identity_ok(curated: dict[str, str], verified: dict[str, Any]) -> tuple[bool, str]:
    chain = curated.get("chain") or curated.get("provider_chain_id") or ""
    expected_pair = curated.get("provider_pair_address") or ""
    got_pair = _cell(verified.get("pair_address") or verified.get("provider_pair_id"))
    got_chain = _cell(verified.get("normalized_chain_id") or verified.get("chain_id"))
    if got_chain and _norm_chain(got_chain) != _norm_chain(chain):
        return False, f"chain_mismatch expected={chain} got={got_chain}"
    if got_pair and not _addresses_match(chain, expected_pair, got_pair):
        return False, f"pair_mismatch expected={expected_pair} got={got_pair}"
    if not verified.get("base_token_address") or not verified.get("quote_token_address"):
        return False, "missing_base_or_quote_after_verify"
    return True, "ok"


def run_curated_refetch(
    accepted_rows: list[dict[str, str]],
    *,
    dry_run: bool = False,
    limit: int | None = None,
    sleep_seconds: float = 1.0,
    use_cache: bool = False,
    sleeper: Callable[[float], None] | None = None,
    verify_fn: Callable[..., dict[str, Any]] | None = None,
    row_builder: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Exact-pair refetch curated targets through Clean Forward verify gates.

    No trending/search. Sleeps between HTTP calls. Does not run on import.
    """
    do_sleep = sleeper or time.sleep
    rows_in = list(accepted_rows)
    if limit is not None and limit >= 0:
        rows_in = rows_in[:limit]

    refetch_results: list[dict[str, Any]] = []
    clean_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    provider_jsonl: list[dict[str, Any]] = []
    http_calls = 0
    rate_limited = 0
    retryable = 0

    # Lazy imports — avoid pulling HTTP stack at module import time for flag-off
    if verify_fn is None and not dry_run:
        from app.ae13b_product.clean_forward_market_feed import verify_provider_pair

        verify_fn = verify_provider_pair
    if row_builder is None and not dry_run:
        from app.ae13b_product.clean_forward_market_feed import _row_from_verification

        row_builder = _row_from_verification

    for idx, curated in enumerate(rows_in):
        chain = curated.get("chain") or curated.get("provider_chain_id") or ""
        pair = curated.get("provider_pair_address") or ""
        base_result: dict[str, Any] = {
            "combined_target_id": curated.get("combined_target_id"),
            "chain": chain,
            "provider_pair_address": pair,
            "target_source": curated.get("target_source"),
            "linked_sources": curated.get("linked_sources"),
            "seed_collection": curated.get("seed_collection"),
            "semantic_status": SEMANTIC_PENDING,
            "system_semantic_label": "",
            "provider_pair_url": curated.get("provider_pair_url"),
            "http_attempted": "false",
            "clean_forward_gate_passed": "false",
            "rejection_reason": "",
            "verification_status": "",
        }

        if dry_run:
            base_result["verification_status"] = "DRY_RUN_NOT_FETCHED"
            base_result["rejection_reason"] = "dry_run"
            refetch_results.append(base_result)
            provider_jsonl.append(
                {
                    "combined_target_id": curated.get("combined_target_id"),
                    "chain": chain,
                    "pair_address": pair,
                    "http_attempted": False,
                    "acceptance_status": "DRY_RUN_NOT_FETCHED",
                    "raw_response_json": None,
                }
            )
            continue

        assert verify_fn is not None and row_builder is not None
        http_calls += 1
        base_result["http_attempted"] = "true"
        verified = verify_fn(
            chain_id=chain,
            pair_address=pair,
            expected_url=curated.get("provider_pair_url") or None,
            use_cache=use_cache,
        )
        status = str(verified.get("verification_status") or verified.get("status") or "")
        base_result["verification_status"] = status
        base_result["verification_http_status"] = verified.get("verification_http_status")
        base_result["clean_feed_eligible"] = bool(verified.get("clean_feed_eligible"))
        base_result["lookup_ok"] = bool(verified.get("lookup_ok"))
        base_result["freshness_status"] = verified.get("freshness_status")
        base_result["identity_status"] = verified.get("identity_status")
        base_result["exclusion_reason"] = verified.get("exclusion_reason") or verified.get(
            "reject_reason"
        )

        if status in {"provider_rate_limited"}:
            rate_limited += 1
            retryable += 1
        if status in {"provider_unavailable", "provider_rate_limited"}:
            retryable += 1

        provider_jsonl.append(
            {
                "combined_target_id": curated.get("combined_target_id"),
                "chain": chain,
                "pair_address": pair,
                "http_attempted": True,
                "verification_status": status,
                "clean_feed_eligible": bool(verified.get("clean_feed_eligible")),
                "provider_pair_url": verified.get("provider_pair_url"),
                "raw_response_json": {
                    k: verified.get(k)
                    for k in (
                        "pair_address",
                        "normalized_chain_id",
                        "base_token_address",
                        "quote_token_address",
                        "price_usd",
                        "liquidity_usd",
                        "provider_pair_url",
                        "verification_status",
                        "exclusion_reason",
                    )
                },
            }
        )

        ok_identity, identity_reason = _identity_ok(curated, verified)
        base_result["identity_match"] = "true" if ok_identity else "false"
        base_result["identity_match_reason"] = identity_reason

        gate_ok = (
            bool(verified.get("clean_feed_eligible"))
            and bool(verified.get("lookup_ok"))
            and bool(verified.get("provider_pair_url"))
            and ok_identity
        )

        if not gate_ok:
            reason = (
                base_result.get("exclusion_reason")
                or identity_reason
                or status
                or "clean_forward_gate_failed"
            )
            base_result["rejection_reason"] = reason
            base_result["clean_forward_gate_passed"] = "false"
            refetch_results.append(base_result)
            rejected.append(
                {
                    **base_result,
                    "seed_collection": curated.get("seed_collection"),
                    "semantic_status": SEMANTIC_PENDING,
                }
            )
        else:
            row = row_builder(verified)
            # Attach curated provenance (semantic separation)
            row["combined_target_id"] = curated.get("combined_target_id")
            row["target_source"] = curated.get("target_source")
            row["linked_sources"] = curated.get("linked_sources")
            row["seed_collection"] = curated.get("seed_collection")
            row["semantic_status"] = SEMANTIC_PENDING
            row["system_semantic_label"] = ""
            row["curated_overlay"] = True
            row["paper_demo_only"] = True
            row["live_trading_ready"] = False
            # Preserve Solana/XRPL provider pair casing from verify (or curated seed)
            if _is_non_evm(chain):
                # Prefer verified pair_address; it should match curated exactly
                row["pair_address"] = verified.get("pair_address") or pair
            base_result["clean_forward_gate_passed"] = "true"
            base_result["rejection_reason"] = ""
            refetch_results.append(base_result)
            clean_rows.append(row)

        if idx < len(rows_in) - 1 and sleep_seconds > 0:
            do_sleep(sleep_seconds)

    return {
        "refetch_results": refetch_results,
        "clean_forward_rows": clean_rows,
        "rejected_rows": rejected,
        "provider_jsonl_records": provider_jsonl,
        "http_calls_attempted": http_calls,
        "rate_limited_count": rate_limited,
        "retryable_failure_count": retryable,
        "dry_run": dry_run,
    }


def build_curated_clean_forward_market_feed(
    *,
    limit: int = 25,
    use_cache: bool = True,
    sleep_seconds: float = 1.0,
    environ: dict[str, str] | None = None,
    path: Path | None = None,
    dry_run: bool = False,
    sleeper: Callable[[float], None] | None = None,
    verify_fn: Callable[..., dict[str, Any]] | None = None,
    **_ignored: Any,
) -> dict[str, Any]:
    """Build a Clean Forward feed from curated ready targets (exact pair only).

    Compatible return shape with build_clean_forward_market_feed for overlay use.
    """
    loaded = load_curated_ready_targets(
        path=path,
        environ=environ,
        explicit_validation=False,
    )
    if loaded.get("blocked_classification") or not loaded.get("accepted_rows"):
        # Graceful fallback signal for runtime — empty curated feed
        return {
            "mode": "curated_overlay",
            "curated_overlay": True,
            "curated_load_error": loaded.get("error") or "no_accepted_curated_targets",
            "fallback_to_non_curated_recommended": True,
            "rows": [],
            "alternative_pools": [],
            "invalid_or_unresolved": loaded.get("excluded_rows") or [],
            "stats": {
                "curated_targets_loaded": len(loaded.get("loaded_rows") or []),
                "curated_targets_accepted": 0,
                "clean_rows_displayed": 0,
            },
            "feed_label": "Clean Forward (curated overlay — unavailable)",
            "paper_demo_only": True,
            "live_trading_ready": False,
        }

    accepted = loaded["accepted_rows"]
    # honor limit like the search feed
    run = run_curated_refetch(
        accepted,
        dry_run=dry_run,
        limit=limit,
        sleep_seconds=sleep_seconds,
        use_cache=use_cache,
        sleeper=sleeper,
        verify_fn=verify_fn,
    )
    rows = run["clean_forward_rows"]
    return {
        "mode": "curated_overlay",
        "curated_overlay": True,
        "rows": rows,
        "alternative_pools": [],
        "invalid_or_unresolved": run["rejected_rows"],
        "stats": {
            "curated_targets_loaded": len(loaded.get("loaded_rows") or []),
            "curated_targets_accepted": len(accepted),
            "curated_targets_excluded": len(loaded.get("excluded_rows") or []),
            "valid_provider_pairs": len(rows),
            "clean_rows_displayed": len(rows),
            "http_calls_attempted": run["http_calls_attempted"],
            "rate_limited_count": run["rate_limited_count"],
        },
        "curated_path": loaded.get("path"),
        "feed_label": "Clean Forward (curated overlay)",
        "paper_demo_only": True,
        "live_trading_ready": False,
        "semantic_status": SEMANTIC_PENDING,
        "system_semantic_label_created": False,
    }


def try_curated_overlay_or_none(
    *,
    limit: int = 25,
    use_cache: bool = True,
    environ: dict[str, str] | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """If flag enabled and path loadable, return curated feed; else None (use legacy)."""
    if not curated_targets_enabled(environ):
        return None
    loaded = load_curated_ready_targets(environ=environ, explicit_validation=False)
    if not loaded.get("accepted_rows"):
        # Graceful fallback — do not crash; signal caller to use existing path
        warnings.warn(
            "Curated overlay enabled but curated targets unavailable; "
            "falling back to existing Clean Forward collector behavior.",
            UserWarning,
            stacklevel=2,
        )
        return None
    return build_curated_clean_forward_market_feed(
        limit=limit,
        use_cache=use_cache,
        environ=environ,
        path=Path(loaded["path"]),
        **kwargs,
    )
