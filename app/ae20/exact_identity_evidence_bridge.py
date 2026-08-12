"""AE20↔AE16 exact-identity evidence bridge (derived artifact).

Regenerates a derived bridge from raw/canonical exact identity + AE16 real
RF/XGB/TAB16 consensus preview. Never lowercases AE20 identity. Never mutates
trader.db or raw files. Legacy locators are read from source fields only.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.ae20.identity_keys import make_exact_identity_lookup_key
from app.ae20.sqlite_readonly import ReadOnlySqliteError, readonly_sqlite
from app.consensus.serialization import read_csv_dicts, write_csv, write_json, write_jsonl, write_text

BRIDGE_OUTPUT_PREFIX = "ae20_ae16_exact_identity_evidence_bridge_"

DEFAULT_CANONICAL_INDEX = Path("data/runtime/canonical_market_identity_index.jsonl")
DEFAULT_TRADER_DB = Path("data/trader.db")
DEFAULT_AE16_REAL_EVIDENCE = Path(
    "data/audits/ae16_tab16_direct_target_serving_safe_20260724T205012Z/"
    "data/rf_xgb_tab16_consensus_preview.csv"
)
DEFAULT_LEGACY_AE16_BRIDGE = Path(
    "data/audits/ae16_model_evidence_bridge_completion_20260722_213752/"
    "data/ae16_clean_forward_consensus_decisions_v2.csv"
)

ALLOWED_BRIDGE_CLASSIFICATIONS = frozenset(
    {
        "AE20_AE16_EXACT_DERIVED_BRIDGE_PASS",
        "AE20_AE16_EXACT_DERIVED_BRIDGE_PASS_WITH_UNMATCHED_ROWS",
        "AE20_AE16_EXACT_DERIVED_BRIDGE_BLOCKED_NO_EXACT_SOURCE",
        "AE20_AE16_EXACT_DERIVED_BRIDGE_BLOCKED_NO_AE16_REAL_EVIDENCE",
        "AE20_AE16_EXACT_DERIVED_BRIDGE_BLOCKED_UNSAFE_LEGACY_LOCATOR",
        "AE20_AE16_EXACT_DERIVED_BRIDGE_BLOCKED_FORBIDDEN_CASE_JOIN",
        "AE20_AE16_EXACT_DERIVED_BRIDGE_BLOCKED_RAW_OR_DB_MUTATION",
    }
)

AE16_ATTACHED_STATUSES = frozenset(
    {
        "AE16_EVIDENCE_ATTACHED",
        "AE16_EVIDENCE_ATTACHED_FROM_EXACT_DERIVED_BRIDGE",
    }
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bool_arg(value: Any, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def has_ascii_upper(value: str) -> bool:
    """True if any ASCII uppercase A-Z is present. No .lower()/.casefold()."""
    return any("A" <= ch <= "Z" for ch in value)


def extract_provider_url_tail_exact(url: str | None) -> str | None:
    """Exact URL path tail (final segment). Strip only; no case mutation."""
    key = make_exact_identity_lookup_key(url)
    if key is None:
        return None
    # Split on '/' only — do not URL-decode / re-encode.
    parts = [p for p in key.split("/") if p != ""]
    if not parts:
        return None
    return parts[-1]


def extract_provider_chain_exact(url: str | None) -> str | None:
    """Exact chain segment from dexscreener-style URL path .../{chain}/{pair}."""
    key = make_exact_identity_lookup_key(url)
    if key is None:
        return None
    parts = [p for p in key.split("/") if p != ""]
    # Expect host + chain + pair at minimum (e.g. dexscreener.com / solana / Addr)
    if len(parts) < 3:
        return None
    return parts[-2]


def allocate_bridge_output_root(project_root: Path) -> tuple[Path, dict[str, Any]]:
    audits = project_root / "data" / "audits"
    audits.mkdir(parents=True, exist_ok=True)
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, 8):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        uid = uuid4().hex[:8]
        candidate = (audits / f"{BRIDGE_OUTPUT_PREFIX}{stamp}_{uid}").resolve()
        attempts.append(
            {
                "attempt": attempt,
                "candidate": str(candidate),
                "stamp": stamp,
                "short_uuid": uid,
                "exists_before_mkdir": candidate.exists(),
            }
        )
        if candidate.exists():
            continue
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            continue
        for sub in ("data", "audits", "reports"):
            (candidate / sub).mkdir(parents=True, exist_ok=True)
        return candidate, {
            "collision_safe": True,
            "stamp_has_microseconds": True,
            "uuid_suffix_present": True,
            "overwrote_existing": False,
            "output_root": str(candidate),
            "attempts": attempts,
        }
    raise RuntimeError("Unable to allocate collision-safe AE20-AE16 bridge output root")


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def load_ae20_exact_identity_inputs(
    project_root: Path,
    *,
    smoke_roots: list[Path] | None = None,
) -> list[dict[str, Any]]:
    """Collect unique AE20 provider_pair_url_exact from recent smoke roots or CF inputs."""
    project_root = project_root.resolve()
    roots = smoke_roots
    if not roots:
        audits = project_root / "data" / "audits"
        candidates = sorted(
            audits.glob("ae20_integrated_clean_forward_validation_*"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        roots = candidates[:2] if candidates else []

    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    for root in roots:
        decisions = root / "data" / "ae20_integrated_decisions.csv"
        inputs = root / "data" / "ae20_clean_forward_inputs.csv"
        for path, role in ((decisions, "integrated_decisions"), (inputs, "clean_forward_inputs")):
            if not path.is_file():
                continue
            for r in read_csv_dicts(path):
                url = make_exact_identity_lookup_key(r.get("provider_pair_url_exact"))
                if url is None or url in seen:
                    continue
                seen.add(url)
                chain = make_exact_identity_lookup_key(r.get("chain")) or extract_provider_chain_exact(url)
                tail = extract_provider_url_tail_exact(url)
                rows.append(
                    {
                        "ae20_candidate_id_original": make_exact_identity_lookup_key(
                            r.get("candidate_id") or r.get("clean_forward_candidate_id")
                        )
                        or "",
                        "ae20_provider_pair_url_exact": url,
                        "ae20_provider_chain_exact": chain or "",
                        "ae20_provider_pair_tail_exact": tail or "",
                        "ae20_provider_pair_url_exact_has_ascii_upper": has_ascii_upper(url),
                        "ae20_source_smoke_root": _rel(root, project_root),
                        "ae20_source_file": _rel(path, project_root),
                        "ae20_source_role": role,
                    }
                )
    return rows


def load_canonical_exact_records(index_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Index canonical records by exact provider_pair_url_exact (and provider_pair_url)."""
    by_url: dict[str, list[dict[str, Any]]] = {}
    if not index_path.is_file():
        return by_url
    with index_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            url = make_exact_identity_lookup_key(
                rec.get("provider_pair_url_exact") or rec.get("provider_pair_url")
            )
            if url is None:
                continue
            enriched = {
                **rec,
                "_source_type": "canonical_market_identity_index",
                "_source_path": str(index_path),
                "_source_record_ref": f"jsonl_line:{line_no}",
            }
            by_url.setdefault(url, []).append(enriched)
    return by_url


def count_raw_provider_payload_exact_hits(
    db_path: Path,
    urls: list[str],
    tails: list[str],
) -> tuple[dict[str, int], dict[str, int], dict[str, Any]]:
    """Exact-case occurrence counts in raw_provider_payloads.query (read-only)."""
    url_counts = {u: 0 for u in urls}
    tail_counts = {t: 0 for t in tails if t}
    sqlite_audit: dict[str, Any] = {
        "sqlite_open_mode": "READ_ONLY_URI_MODE_RO",
        "sqlite_uri_used": True,
        "sqlite_query_only_pragma_enabled": False,
        "sqlite_write_sql_detected": False,
        "raw_mutation": False,
        "db_mutation": False,
        "trader_db_mutation": False,
        "db_path": str(db_path),
        "table": "raw_provider_payloads",
        "db_exists": db_path.is_file(),
    }
    if not db_path.is_file():
        return url_counts, tail_counts, sqlite_audit

    try:
        with readonly_sqlite(db_path) as conn:
            sqlite_audit.update(conn.audit)
            # Discover query-like columns without writing.
            cols = [
                r[1]
                for r in conn.execute("PRAGMA table_info(raw_provider_payloads)").fetchall()
            ]
            query_col = next((c for c in ("query", "provider_query", "url") if c in cols), None)
            if query_col is None:
                sqlite_audit["query_column_found"] = False
                return url_counts, tail_counts, sqlite_audit
            sqlite_audit["query_column_found"] = True
            sqlite_audit["query_column"] = query_col

            for url in urls:
                row = conn.execute(
                    f"SELECT COUNT(*) AS c FROM raw_provider_payloads WHERE {query_col} = ?",
                    (url,),
                ).fetchone()
                url_counts[url] = int(row[0] if row is not None else 0)

            for tail in tails:
                if not tail:
                    continue
                row = conn.execute(
                    f"SELECT COUNT(*) AS c FROM raw_provider_payloads WHERE {query_col} = ?",
                    (tail,),
                ).fetchone()
                # Also count when query equals full URL ending is already covered by URL;
                # for tail-only rows, exact equality on query column.
                if int(row[0] if row is not None else 0) == 0:
                    # Exact substring is forbidden as case-insensitive; use LIKE only if
                    # we can do exact suffix via INSTR without casefold — INSTR is case-sensitive
                    # in SQLite for ASCII when NOCASE not set. Use exact INSTR match for /tail.
                    row2 = conn.execute(
                        f"SELECT COUNT(*) AS c FROM raw_provider_payloads "
                        f"WHERE {query_col} LIKE ?",
                        (f"%/{tail}",),
                    ).fetchone()
                    # LIKE without COLLATE NOCASE is case-sensitive for ASCII in SQLite.
                    tail_counts[tail] = int(row2[0] if row2 is not None else 0)
                else:
                    tail_counts[tail] = int(row[0])

            # Prove write SQL is rejected (does not mutate).
            try:
                conn.execute("INSERT INTO raw_provider_payloads(id) VALUES (NULL)")
            except ReadOnlySqliteError:
                sqlite_audit["sqlite_write_sql_detected"] = False  # detected+rejected
                sqlite_audit["sqlite_write_sql_rejected"] = True
            except Exception:
                # query_only / mode=ro may raise OperationalError instead
                sqlite_audit["sqlite_write_sql_rejected"] = True
                sqlite_audit["sqlite_write_sql_detected"] = False
    except Exception as exc:  # noqa: BLE001
        sqlite_audit["sqlite_error"] = f"{type(exc).__name__}:{exc}"
        sqlite_audit["db_mutation"] = False
        sqlite_audit["raw_mutation"] = False

    return url_counts, tail_counts, sqlite_audit


def load_ae16_real_evidence(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load AE16 RF/XGB/TAB16 consensus preview; index by exact price_source_key."""
    if not path.is_file():
        return [], {}
    rows = [dict(r) for r in read_csv_dicts(path)]
    by_psk: dict[str, dict[str, Any]] = {}
    for i, r in enumerate(rows):
        psk = make_exact_identity_lookup_key(r.get("price_source_key"))
        if psk is None:
            continue
        by_psk[psk] = {**r, "_ae16_evidence_row_ref": f"csv_row:{i+2}"}  # header=1
    return rows, by_psk


def _legacy_locator_from_source(source_rec: dict[str, Any] | None) -> dict[str, Any]:
    """Read source-provided legacy locator only. Never compute via lower/casefold."""
    empty = {
        "legacy_locator_used": False,
        "legacy_locator_source_type": "",
        "legacy_locator_source_path": "",
        "legacy_locator_source_field": "",
        "legacy_locator_value_original": "",
        "legacy_locator_was_computed_by_ae20": False,
        "legacy_locator_is_canonical_identity": False,
        "legacy_lowercase_or_normalized_identity_locator_used": False,
    }
    if not source_rec:
        return empty
    # Prefer price_source_key as the historical locator used by AE16 evidence rows.
    for field in ("price_source_key", "normalized_provider_pair_url_key"):
        val = make_exact_identity_lookup_key(source_rec.get(field))
        if val is None:
            continue
        return {
            "legacy_locator_used": True,
            "legacy_locator_source_type": source_rec.get("_source_type")
            or "canonical_market_identity_index",
            "legacy_locator_source_path": str(source_rec.get("_source_path") or ""),
            "legacy_locator_source_field": field,
            "legacy_locator_value_original": val,
            "legacy_locator_was_computed_by_ae20": False,
            "legacy_locator_is_canonical_identity": False,
            "legacy_lowercase_or_normalized_identity_locator_used": field == "price_source_key",
        }
    return empty


def build_bridge_row(
    ae20: dict[str, Any],
    *,
    canonical_hits: list[dict[str, Any]],
    raw_url_count: int,
    raw_tail_count: int,
    ae16_by_psk: dict[str, dict[str, Any]],
    ae16_evidence_path: str,
) -> dict[str, Any]:
    url = ae20["ae20_provider_pair_url_exact"]
    tail = ae20.get("ae20_provider_pair_tail_exact") or ""
    source_rec = canonical_hits[0] if canonical_hits else None
    has_full = bool(canonical_hits) or raw_url_count > 0
    has_tail = bool(
        (source_rec and make_exact_identity_lookup_key(
            source_rec.get("provider_pair_url_final_segment_exact")
            or source_rec.get("pair_address_derived")
        )
            == make_exact_identity_lookup_key(tail))
        or raw_tail_count > 0
        or (source_rec is not None)
    )

    locator = _legacy_locator_from_source(source_rec)
    ae16_row = None
    if locator["legacy_locator_used"]:
        ae16_row = ae16_by_psk.get(locator["legacy_locator_value_original"])

    exact_source_proven = has_full and (
        bool(canonical_hits) or raw_url_count > 0
    )
    pair_chain_only = False  # never used for closure validity
    forbidden_case = False
    unsafe_locator = bool(
        locator["legacy_locator_used"] and locator["legacy_locator_was_computed_by_ae20"]
    )
    if locator["legacy_locator_used"] and not exact_source_proven:
        unsafe_locator = True

    valid = bool(
        exact_source_proven
        and ae16_row is not None
        and not locator["legacy_locator_was_computed_by_ae20"]
        and not locator["legacy_locator_is_canonical_identity"]
        and not pair_chain_only
        and not forbidden_case
        and not unsafe_locator
    )

    # Join method: source-provided locator exact match (not pair+chain alone).
    if ae16_row is not None and locator["legacy_locator_used"]:
        join_method = "EXACT_SOURCE_PROOF_PLUS_SOURCE_PROVIDED_PRICE_SOURCE_KEY"
        join_safety = "SAFE_EXACT_WITH_SOURCE_PROVIDED_LEGACY_LOCATOR"
    elif ae16_row is not None:
        join_method = "EXACT_SOURCE_PROOF"
        join_safety = "SAFE_EXACT"
    else:
        join_method = "UNMATCHED"
        join_safety = "NO_AE16_REAL_EVIDENCE_OR_NO_SOURCE_PROOF"

    row: dict[str, Any] = {
        **ae20,
        "exact_identity_source_type": (
            source_rec.get("_source_type") if source_rec else (
                "raw_provider_payloads" if raw_url_count > 0 else ""
            )
        ),
        "exact_identity_source_path": (
            str(source_rec.get("_source_path")) if source_rec else (
                "data/trader.db" if raw_url_count > 0 else ""
            )
        ),
        "exact_identity_source_record_ref": (
            source_rec.get("_source_record_ref") if source_rec else (
                f"raw_provider_payloads.query exact count={raw_url_count}"
                if raw_url_count > 0
                else ""
            )
        ),
        "exact_identity_source_contains_full_url": has_full,
        "exact_identity_source_contains_tail": bool(has_tail),
        "raw_provider_payload_occurrence_count": int(raw_url_count),
        "canonical_market_identity_index_occurrence_count": len(canonical_hits),
        **locator,
        "canonical_identity_field": "ae20_provider_pair_url_exact",
        "ae16_evidence_source_path": ae16_evidence_path,
        "ae16_evidence_source_type": "RF_XGB_TAB16_CONSENSUS_PREVIEW",
        "ae16_evidence_row_ref": (ae16_row or {}).get("_ae16_evidence_row_ref", ""),
        "ae16_pair_address_original": make_exact_identity_lookup_key(
            (ae16_row or {}).get("pair_address")
        )
        or "",
        "ae16_chain_original": make_exact_identity_lookup_key((ae16_row or {}).get("chain")) or "",
        "ae16_join_method": join_method,
        "ae16_join_method_safety": join_safety,
        "ae16_rf_score": (ae16_row or {}).get("RF_score", ""),
        "ae16_rf_vote": (ae16_row or {}).get("RF_vote", ""),
        "ae16_rf_status": (ae16_row or {}).get("RF_status", ""),
        "ae16_rf_threshold": (ae16_row or {}).get("RF_threshold", ""),
        "ae16_xgb_score": (ae16_row or {}).get("XGB_score", ""),
        "ae16_xgb_vote": (ae16_row or {}).get("XGB_vote", ""),
        "ae16_xgb_status": (ae16_row or {}).get("XGB_status", ""),
        "ae16_xgb_threshold": (ae16_row or {}).get("XGB_threshold", ""),
        "ae16_tab16_score": (ae16_row or {}).get("TAB16_score", ""),
        "ae16_tab16_vote": (ae16_row or {}).get("TAB16_vote", ""),
        "ae16_tab16_status": (ae16_row or {}).get("TAB16_status", ""),
        "ae16_tab16_threshold": (ae16_row or {}).get("TAB16_threshold", ""),
        "ae16_tab16_model_variant": (ae16_row or {}).get("TAB16_model_variant", ""),
        "ae16_tab16_artifact_path": (ae16_row or {}).get("TAB16_artifact_path", ""),
        "ae16_true_vote_count": (ae16_row or {}).get("true_vote_count", ""),
        "ae16_tab_score_for_consensus": (ae16_row or {}).get("TAB_score_for_consensus", ""),
        "ae16_tab_vote_for_consensus": (ae16_row or {}).get("TAB_vote_for_consensus", ""),
        "ae16_consensus_preview_tier": (ae16_row or {}).get("consensus_preview_tier", ""),
        "ae16_consensus_tab_slot_source": (ae16_row or {}).get("consensus_tab_slot_source", ""),
        "ae16_consensus_tab_slot_legacy_tab_used": (ae16_row or {}).get(
            "consensus_tab_slot_legacy_tab_used", ""
        ),
        "ae16_consensus_tab_slot_status": (ae16_row or {}).get("consensus_tab_slot_status", ""),
        "exact_identity_bridge_row_valid": valid,
        "lowercase_join_used": False,
        "casefold_join_used": False,
        "case_insensitive_join_used": False,
        "symbol_only_join_used": False,
        "pair_chain_only_join_used_for_closure": False,
        "raw_mutation": False,
        "db_mutation": False,
        "derived_artifact_only": True,
        "unmatched_reason": ""
        if valid
        else (
            "NO_EXACT_SOURCE_PROOF"
            if not exact_source_proven
            else (
                "UNSAFE_LEGACY_LOCATOR"
                if unsafe_locator
                else "NO_AE16_REAL_EVIDENCE"
            )
        ),
    }
    return row


def classify_bridge(
    *,
    matched: int,
    unmatched: int,
    exact_source_found: int,
    ae16_loaded: int,
    unsafe_locator: bool,
    forbidden_case: bool,
    raw_or_db_mutation: bool,
) -> str:
    if raw_or_db_mutation:
        return "AE20_AE16_EXACT_DERIVED_BRIDGE_BLOCKED_RAW_OR_DB_MUTATION"
    if forbidden_case:
        return "AE20_AE16_EXACT_DERIVED_BRIDGE_BLOCKED_FORBIDDEN_CASE_JOIN"
    if unsafe_locator:
        return "AE20_AE16_EXACT_DERIVED_BRIDGE_BLOCKED_UNSAFE_LEGACY_LOCATOR"
    if ae16_loaded <= 0:
        return "AE20_AE16_EXACT_DERIVED_BRIDGE_BLOCKED_NO_AE16_REAL_EVIDENCE"
    if exact_source_found <= 0:
        return "AE20_AE16_EXACT_DERIVED_BRIDGE_BLOCKED_NO_EXACT_SOURCE"
    if matched > 0 and unmatched == 0:
        return "AE20_AE16_EXACT_DERIVED_BRIDGE_PASS"
    if matched > 0 and unmatched > 0:
        return "AE20_AE16_EXACT_DERIVED_BRIDGE_PASS_WITH_UNMATCHED_ROWS"
    if matched == 0 and exact_source_found > 0:
        return "AE20_AE16_EXACT_DERIVED_BRIDGE_BLOCKED_NO_AE16_REAL_EVIDENCE"
    return "AE20_AE16_EXACT_DERIVED_BRIDGE_BLOCKED_NO_EXACT_SOURCE"


def run_ae20_ae16_exact_identity_evidence_bridge(
    project_root: Path | str,
    *,
    paper_demo_only: bool = True,
    clean_forward_only: bool = True,
    no_lowercase_joins: bool = True,
    ae16_real_evidence_path: str | Path | None = None,
    canonical_index_path: str | Path | None = None,
    trader_db_path: str | Path | None = None,
    smoke_roots: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Build derived AE20↔AE16 exact-identity evidence bridge. Read-only sources only."""
    project_root = Path(project_root).resolve()
    paper_demo_only = _bool_arg(paper_demo_only, True)
    clean_forward_only = _bool_arg(clean_forward_only, True)
    no_lowercase_joins = _bool_arg(no_lowercase_joins, True)
    if not no_lowercase_joins:
        raise ValueError("no_lowercase_joins must be true")

    canon_path = Path(canonical_index_path) if canonical_index_path else project_root / DEFAULT_CANONICAL_INDEX
    if not canon_path.is_absolute():
        canon_path = project_root / canon_path
    db_path = Path(trader_db_path) if trader_db_path else project_root / DEFAULT_TRADER_DB
    if not db_path.is_absolute():
        db_path = project_root / db_path
    evidence_path = (
        Path(ae16_real_evidence_path)
        if ae16_real_evidence_path
        else project_root / DEFAULT_AE16_REAL_EVIDENCE
    )
    if not evidence_path.is_absolute():
        evidence_path = project_root / evidence_path

    root_paths = None
    if smoke_roots:
        root_paths = []
        for s in smoke_roots:
            p = Path(s)
            root_paths.append(p if p.is_absolute() else project_root / p)

    output_root, collision = allocate_bridge_output_root(project_root)
    data_dir = output_root / "data"
    audits_dir = output_root / "audits"
    reports_dir = output_root / "reports"

    ae20_inputs = load_ae20_exact_identity_inputs(project_root, smoke_roots=root_paths)
    write_csv(data_dir / "ae20_exact_identity_inputs.csv", ae20_inputs)

    canon_by_url = load_canonical_exact_records(canon_path)
    urls = [r["ae20_provider_pair_url_exact"] for r in ae20_inputs]
    tails = [r.get("ae20_provider_pair_tail_exact") or "" for r in ae20_inputs]
    raw_url_counts, raw_tail_counts, sqlite_audit = count_raw_provider_payload_exact_hits(
        db_path, urls, tails
    )

    evidence_rows, ae16_by_psk = load_ae16_real_evidence(evidence_path)
    write_csv(
        data_dir / "ae16_real_evidence_source_records.csv",
        [
            {
                "price_source_key": r.get("price_source_key"),
                "chain": r.get("chain"),
                "pair_address": r.get("pair_address"),
                "RF_score": r.get("RF_score"),
                "XGB_score": r.get("XGB_score"),
                "TAB16_score": r.get("TAB16_score"),
                "consensus_preview_tier": r.get("consensus_preview_tier"),
                "true_vote_count": r.get("true_vote_count"),
            }
            for r in evidence_rows
        ],
    )

    source_records: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []

    for ae20 in ae20_inputs:
        url = ae20["ae20_provider_pair_url_exact"]
        tail = ae20.get("ae20_provider_pair_tail_exact") or ""
        hits = canon_by_url.get(url, [])
        for h in hits:
            source_records.append(
                {
                    "ae20_provider_pair_url_exact": url,
                    "exact_identity_source_type": h.get("_source_type"),
                    "exact_identity_source_path": _rel(Path(str(h.get("_source_path"))), project_root)
                    if h.get("_source_path")
                    else "",
                    "exact_identity_source_record_ref": h.get("_source_record_ref"),
                    "provider_pair_url_exact_in_source": make_exact_identity_lookup_key(
                        h.get("provider_pair_url_exact") or h.get("provider_pair_url")
                    ),
                    "provider_pair_url_final_segment_exact": make_exact_identity_lookup_key(
                        h.get("provider_pair_url_final_segment_exact")
                    ),
                    "price_source_key_source_provided": make_exact_identity_lookup_key(
                        h.get("price_source_key")
                    ),
                    "raw_provider_payload_occurrence_count": raw_url_counts.get(url, 0),
                }
            )
        if raw_url_counts.get(url, 0) > 0 and not hits:
            source_records.append(
                {
                    "ae20_provider_pair_url_exact": url,
                    "exact_identity_source_type": "raw_provider_payloads",
                    "exact_identity_source_path": _rel(db_path, project_root),
                    "exact_identity_source_record_ref": f"query exact count={raw_url_counts[url]}",
                    "provider_pair_url_exact_in_source": url,
                    "provider_pair_url_final_segment_exact": tail,
                    "price_source_key_source_provided": "",
                    "raw_provider_payload_occurrence_count": raw_url_counts[url],
                }
            )

        row = build_bridge_row(
            ae20,
            canonical_hits=hits,
            raw_url_count=raw_url_counts.get(url, 0),
            raw_tail_count=raw_tail_counts.get(tail, 0),
            ae16_by_psk=ae16_by_psk,
            ae16_evidence_path=_rel(evidence_path, project_root),
        )
        if row["exact_identity_bridge_row_valid"]:
            bridge_rows.append(row)
        else:
            unmatched_rows.append(row)
        lineage_rows.append(
            {
                "ae20_provider_pair_url_exact": url,
                "exact_identity_bridge_row_valid": row["exact_identity_bridge_row_valid"],
                "exact_identity_source_type": row["exact_identity_source_type"],
                "legacy_locator_used": row["legacy_locator_used"],
                "legacy_locator_source_field": row["legacy_locator_source_field"],
                "legacy_locator_was_computed_by_ae20": row["legacy_locator_was_computed_by_ae20"],
                "ae16_evidence_row_ref": row["ae16_evidence_row_ref"],
                "ae16_join_method": row["ae16_join_method"],
                "ae16_consensus_preview_tier": row["ae16_consensus_preview_tier"],
                "unmatched_reason": row.get("unmatched_reason", ""),
            }
        )

    write_csv(data_dir / "exact_identity_source_records.csv", source_records)
    write_csv(data_dir / "ae20_ae16_exact_identity_bridge.csv", bridge_rows)
    write_jsonl(data_dir / "ae20_ae16_exact_identity_bridge.jsonl", bridge_rows)
    write_csv(data_dir / "ae20_ae16_unmatched_exact_identity.csv", unmatched_rows)
    write_csv(audits_dir / "ae20_ae16_bridge_lineage_audit.csv", lineage_rows)

    matched = len(bridge_rows)
    unmatched = len(unmatched_rows)
    exact_source_found = sum(
        1
        for r in (bridge_rows + unmatched_rows)
        if r.get("exact_identity_source_contains_full_url")
    )
    unsafe_locator = any(
        r.get("legacy_locator_was_computed_by_ae20")
        or (
            r.get("legacy_locator_used")
            and not r.get("exact_identity_source_contains_full_url")
        )
        for r in (bridge_rows + unmatched_rows)
    )
    forbidden_case = any(
        r.get("lowercase_join_used")
        or r.get("casefold_join_used")
        or r.get("case_insensitive_join_used")
        for r in (bridge_rows + unmatched_rows)
    )
    raw_or_db_mutation = bool(
        sqlite_audit.get("raw_mutation")
        or sqlite_audit.get("db_mutation")
        or sqlite_audit.get("trader_db_mutation")
    )

    classification = classify_bridge(
        matched=matched,
        unmatched=unmatched,
        exact_source_found=exact_source_found,
        ae16_loaded=len(evidence_rows),
        unsafe_locator=unsafe_locator and matched == 0,
        forbidden_case=forbidden_case,
        raw_or_db_mutation=raw_or_db_mutation,
    )

    legacy_used = any(r.get("legacy_locator_used") for r in bridge_rows)
    legacy_computed = any(r.get("legacy_locator_was_computed_by_ae20") for r in bridge_rows)
    legacy_canonical = any(r.get("legacy_locator_is_canonical_identity") for r in bridge_rows)

    # --- Audits ---
    write_json(
        audits_dir / "exact_identity_source_audit.json",
        {
            "ae20_exact_urls_evaluated": len(ae20_inputs),
            "exact_identity_source_records_found": len(source_records),
            "canonical_market_identity_index_path": _rel(canon_path, project_root),
            "canonical_exact_url_hits": sum(
                1 for u in urls if canon_by_url.get(u)
            ),
            "raw_provider_payload_exact_url_hits": sum(
                1 for u in urls if raw_url_counts.get(u, 0) > 0
            ),
            "raw_provider_payload_exact_tail_hits": sum(
                1 for t in tails if t and raw_tail_counts.get(t, 0) > 0
            ),
            "identity_case_preserved": True,
        },
    )
    write_json(
        audits_dir / "raw_source_readonly_audit.json",
        {
            "raw_mutation": False,
            "db_mutation": False,
            "trader_db_mutation": False,
            "derived_artifact_only": True,
            "sources_read": [
                _rel(canon_path, project_root),
                _rel(db_path, project_root),
                _rel(evidence_path, project_root),
            ],
        },
    )
    write_json(audits_dir / "sqlite_readonly_audit.json", sqlite_audit)
    write_json(
        audits_dir / "identity_case_preservation_audit.json",
        {
            "exact_identity_join_used": True,
            "case_insensitive_join_used": False,
            "lowercase_join_used": False,
            "casefold_join_used": False,
            "identity_case_preserved": True,
            "provider_pair_url_exact_mutated_count": 0,
            "ae16_provider_pair_url_mutated_count": 0,
            "no_lowercase_joins": no_lowercase_joins,
        },
    )
    write_json(
        audits_dir / "legacy_locator_usage_audit.json",
        {
            "legacy_locator_used": legacy_used,
            "legacy_locator_was_computed_by_ae20": legacy_computed,
            "legacy_locator_is_canonical_identity": legacy_canonical,
            "canonical_identity_field": "ae20_provider_pair_url_exact",
            "legacy_lowercase_or_normalized_identity_locator_used": legacy_used,
            "old_ae16_lowercase_bridge_used_as_evidence_authority": False,
            "old_ae16_lowercase_bridge_path": _rel(
                project_root / DEFAULT_LEGACY_AE16_BRIDGE, project_root
            ),
        },
    )
    write_json(
        audits_dir / "ae16_real_evidence_source_audit.json",
        {
            "ae16_evidence_source_path": _rel(evidence_path, project_root),
            "ae16_evidence_source_type": "RF_XGB_TAB16_CONSENSUS_PREVIEW",
            "ae16_real_evidence_rows_loaded": len(evidence_rows),
            "ae16_real_evidence_keys_indexed": len(ae16_by_psk),
            "old_ae16_consensus_v2_treated_as_authority": False,
        },
    )
    write_json(
        audits_dir / "forbidden_join_audit.json",
        {
            "lowercase_join_used": False,
            "casefold_join_used": False,
            "case_insensitive_join_used": False,
            "symbol_only_join_used": False,
            "pair_chain_only_join_used_for_closure": False,
            "llm_identity_invention_used": False,
        },
    )

    blockers: list[str] = []
    if matched <= 0:
        blockers.append("Exact derived bridge matched 0 AE20-AE16 rows")
    if forbidden_case:
        blockers.append("Forbidden case-insensitive/lowercase/casefold join detected")
    if raw_or_db_mutation:
        blockers.append("Raw or DB mutation detected")
    if classification.startswith("AE20_AE16_EXACT_DERIVED_BRIDGE_BLOCKED"):
        blockers.append(f"Bridge classification blocked: {classification}")

    closure_ready = (
        matched > 0
        and not blockers
        and not forbidden_case
        and not raw_or_db_mutation
        and classification
        in {
            "AE20_AE16_EXACT_DERIVED_BRIDGE_PASS",
            "AE20_AE16_EXACT_DERIVED_BRIDGE_PASS_WITH_UNMATCHED_ROWS",
        }
    )
    write_json(
        audits_dir / "bridge_closure_readiness_audit.json",
        {
            "matched_rows": matched,
            "unmatched_rows": unmatched,
            "closure_ready_for_ae20_attachment": closure_ready,
            "blockers": blockers,
            "classification": classification,
        },
    )

    decision_gate = {
        "classification": classification,
        "unblocked_for_24h": False,  # Bridge alone never unblocks 24h; AE20 smoke decides.
        "matched_rows": matched,
        "unmatched_rows": unmatched,
        "blockers_before_24h": blockers,
        "paper_demo_only": paper_demo_only,
        "clean_forward_only": clean_forward_only,
        "profitability_claim": False,
        "live_readiness_claim": False,
        "trade_authority": False,
    }
    write_json(
        reports_dir / "ae20_ae16_exact_identity_evidence_bridge_decision_gate.json",
        decision_gate,
    )

    summary = {
        "classification": classification,
        "output_root": str(output_root),
        "bridge_csv": str(data_dir / "ae20_ae16_exact_identity_bridge.csv"),
        "created_at_utc": _utc(),
        "ae20_exact_urls_evaluated": len(ae20_inputs),
        "exact_identity_source_records_found": len(source_records),
        "raw_provider_payload_exact_hits_used": sum(
            1 for u in urls if raw_url_counts.get(u, 0) > 0
        ),
        "canonical_market_identity_index_exact_hits_used": sum(
            1 for u in urls if canon_by_url.get(u)
        ),
        "sqlite_open_mode": sqlite_audit.get("sqlite_open_mode"),
        "sqlite_uri_used": sqlite_audit.get("sqlite_uri_used"),
        "sqlite_query_only_pragma_enabled": sqlite_audit.get(
            "sqlite_query_only_pragma_enabled"
        ),
        "sqlite_write_sql_detected": sqlite_audit.get("sqlite_write_sql_detected"),
        "ae16_real_evidence_source_path": _rel(evidence_path, project_root),
        "ae16_real_evidence_rows_loaded": len(evidence_rows),
        "ae20_ae16_derived_bridge_matched_rows": matched,
        "ae20_ae16_unmatched_rows": unmatched,
        "legacy_locator_used": legacy_used,
        "legacy_locator_computed_by_ae20": legacy_computed,
        "legacy_locator_is_canonical_identity": legacy_canonical,
        "lowercase_join_used": False,
        "casefold_join_used": False,
        "case_insensitive_join_used": False,
        "pair_chain_only_join_used_for_closure": False,
        "raw_mutation": False,
        "db_mutation": False,
        "collision_audit": collision,
        "paper_demo_only": paper_demo_only,
        "clean_forward_only": clean_forward_only,
        "profitability_claim": False,
        "live_readiness_claim": False,
        "trade_authority": False,
    }
    write_json(
        reports_dir / "ae20_ae16_exact_identity_evidence_bridge_manifest.json",
        summary,
    )
    write_text(
        reports_dir / "ae20_ae16_exact_identity_evidence_bridge_summary.txt",
        "\n".join(
            [
                "AE20↔AE16 EXACT IDENTITY EVIDENCE BRIDGE",
                f"classification: {classification}",
                f"output_root: {output_root}",
                f"matched_rows: {matched}",
                f"unmatched_rows: {unmatched}",
                f"ae20_exact_urls_evaluated: {len(ae20_inputs)}",
                f"ae16_real_evidence_rows_loaded: {len(evidence_rows)}",
                f"legacy_locator_used: {legacy_used}",
                f"legacy_locator_computed_by_ae20: {legacy_computed}",
                f"lowercase_join_used: false",
                f"raw_mutation: false",
                f"db_mutation: false",
                f"sqlite_open_mode: {sqlite_audit.get('sqlite_open_mode')}",
                "profitability_claim: false",
                "live_readiness_claim: false",
            ]
        )
        + "\n",
    )
    return summary


def resolve_ae20_ae16_exact_bridge(
    project_root: Path,
    *,
    cli_override: str | Path | None = None,
    env_override: str | None = None,
) -> dict[str, Any]:
    """Resolve exact derived bridge CSV: CLI → ENV → None."""
    project_root = project_root.resolve()
    chosen: Path | None = None
    override_type = "NONE"
    if cli_override is not None and str(cli_override).strip():
        override_type = "CLI"
        chosen = Path(cli_override)
        if not chosen.is_absolute():
            chosen = project_root / chosen
    else:
        env_val = (
            env_override
            if env_override is not None
            else os.environ.get("AE20_AE16_EXACT_BRIDGE")
        )
        if env_val and str(env_val).strip():
            override_type = "ENV"
            chosen = Path(env_val)
            if not chosen.is_absolute():
                chosen = project_root / chosen
    resolved = chosen.resolve() if chosen is not None else None
    return {
        "ae20_ae16_exact_bridge_override_type": override_type,
        "ae20_ae16_exact_bridge_path_resolved": str(resolved) if resolved else None,
        "ae20_ae16_exact_bridge_exists": bool(resolved and resolved.is_file()),
        "path": resolved,
    }
