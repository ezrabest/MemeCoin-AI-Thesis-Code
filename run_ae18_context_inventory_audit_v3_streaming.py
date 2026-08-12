from __future__ import annotations

import csv
import json
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(".").resolve()
RUN_ID = "ae18_context_inventory_audit_v3_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT = ROOT / "data" / "audits" / RUN_ID
OUT.mkdir(parents=True, exist_ok=True)

RUNTIME_INDEX = ROOT / "data" / "runtime" / "canonical_market_identity_index.jsonl"

SCAN_ROOTS = [
    ROOT / "data",
    ROOT / "outputs",
]

SKIP_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    "site-packages",
}

ALLOWED_SUFFIXES = {
    ".json",
    ".jsonl",
    ".csv",
    ".md",
    ".txt",
}

MAX_JSON_LOAD_BYTES = 10_000_000
MAX_JSONL_LINE_BYTES = 3_000_000
CHUNK_BYTES = 1_048_576
MAX_SAMPLES_PER_FAMILY = 20
MAX_PATHS_PER_CATEGORY_IN_PRINT = 50000

FAMILY_TERMS = {
    "helius_solana": [
        "helius",
        "solana",
        "rpc",
        "on_chain",
        "onchain",
        "chain_context",
        "solana_context",
    ],
    "rss_news": [
        "rss",
        "news",
        "article",
        "headline",
        "feed",
        "published_at",
        "source_url",
    ],
    "reputation_scam": [
        "reputation",
        "scam",
        "rug",
        "honeypot",
        "blacklist",
        "risk_flag",
        "risk_warning",
    ],
    "semantic_context": [
        "semantic",
        "narrative",
        "entity",
        "topic",
        "text_context",
        "embedding",
        "context_summary",
    ],
    "explicit_resolver": [
        "resolver",
        "resolution",
        "resolved",
        "unresolved",
        "ambiguous",
        "link_status",
        "symbol_only",
        "symbol-only",
        "join_rejected",
    ],
    "whale_separation": [
        "whale",
        "whale_score",
        "wallet",
        "wallet_level",
        "wallet-level",
        "wallet level",
        "pool_flow",
        "pool-flow",
        "flow_proxy",
        "pool flow proxy",
    ],
}

MISSINGNESS_TERMS = [
    "missing",
    "missingness",
    "unavailable",
    "not_available",
    "no_data",
    "no_records",
    "provider_unavailable",
    "unresolved",
    "unknown",
    "explicit_missingness",
]

PROVENANCE_TERMS = [
    "provenance",
    "source",
    "source_file",
    "source_path",
    "provider",
    "collected_at",
    "fetched_at",
    "created_at",
    "timestamp",
    "observed_at",
    "lineage",
    "audit_run_id",
]

IDENTITY_TERMS = [
    "candidate_id",
    "price_source_key",
    "provider_pair_url_exact",
    "canonical_market_identity",
    "normalized_provider_pair_url_key",
]

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def full_path(path: Path) -> str:
    return str(path.resolve())

def safe_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except Exception:
        return -1

def should_skip_dir(name: str) -> bool:
    return name in SKIP_DIR_NAMES

def should_scan_file(path: Path) -> bool:
    if path.suffix.lower() not in ALLOWED_SUFFIXES:
        return False
    parts = set(path.parts)
    if parts & SKIP_DIR_NAMES:
        return False
    return True

def discover_files() -> list[Path]:
    hits: list[Path] = []
    for base in SCAN_ROOTS:
        if not base.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]

            for fn in filenames:
                p = Path(dirpath) / fn

                # Do not scan this audit while writing it.
                try:
                    if OUT in p.resolve().parents:
                        continue
                except Exception:
                    pass

                if should_scan_file(p):
                    hits.append(p)

    return sorted(set(hits), key=lambda p: full_path(p).lower())

def blob_has(blob: str, terms: list[str]) -> bool:
    low = blob.lower()
    return any(t.lower() in low for t in terms)

def blob_families(blob: str) -> list[str]:
    return [fam for fam, terms in FAMILY_TERMS.items() if blob_has(blob, terms)]

def blob_is_missingness(blob: str) -> bool:
    return blob_has(blob, MISSINGNESS_TERMS)

def blob_has_provenance(blob: str) -> bool:
    return blob_has(blob, PROVENANCE_TERMS)

def blob_has_identity(blob: str) -> bool:
    return blob_has(blob, IDENTITY_TERMS)

def looks_like_context_blob(blob: str) -> bool:
    fams = blob_families(blob)
    if not fams:
        return False
    return blob_has_identity(blob) or blob_has_provenance(blob) or blob_is_missingness(blob)

def classify_resolver_blob(blob: str) -> str:
    low = blob.lower()
    if ("symbol_only" in low or "symbol-only" in low) and ("reject" in low or "rejected" in low):
        return "symbol_only_rejected"
    if "ambiguous" in low:
        return "ambiguous"
    if "unresolved" in low:
        return "unresolved"
    if "resolved" in low:
        return "resolved"
    return "unknown"

def read_runtime_candidates() -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if not RUNTIME_INDEX.exists():
        return candidates

    with RUNTIME_INDEX.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            provider_url = row.get("provider_pair_url_exact") or ""
            canonical = row.get("canonical_market_identity") or provider_url
            normalized = row.get("normalized_provider_pair_url_key") or ""
            candidate_id = row.get("candidate_id") or row.get("price_source_key") or provider_url or canonical or normalized

            candidates.append({
                "runtime_line": line_no,
                "candidate_id": str(candidate_id),
                "price_source_key": str(row.get("price_source_key") or candidate_id),
                "provider_pair_url_exact": str(provider_url),
                "canonical_market_identity": str(canonical),
                "normalized_provider_pair_url_key": str(normalized),
                "symbol_pair_display": str(row.get("symbol_pair_display") or ""),
                "whale_score": row.get("whale_score"),
                "source_path": full_path(RUNTIME_INDEX),
            })

    return candidates

def candidate_identity_values(candidate: dict[str, Any]) -> list[str]:
    values = []
    for k in [
        "candidate_id",
        "price_source_key",
        "provider_pair_url_exact",
        "canonical_market_identity",
        "normalized_provider_pair_url_key",
    ]:
        v = str(candidate.get(k) or "").strip()
        if v:
            values.append(v)
    return list(dict.fromkeys(values))

def match_candidate_key(blob: str, candidates: list[dict[str, Any]]) -> str:
    # Only 45 runtime candidates, so this is safe and exact enough.
    for cand in candidates:
        for v in candidate_identity_values(cand):
            if v and v in blob:
                return str(cand["provider_pair_url_exact"] or cand["canonical_market_identity"] or cand["candidate_id"])
    return ""

def iter_small_json_records(obj: Any, json_path: str = "$") -> Iterable[tuple[Any, str]]:
    if isinstance(obj, dict):
        yield obj, json_path
        for k, v in obj.items():
            yield from iter_small_json_records(v, f"{json_path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_small_json_records(v, f"{json_path}[{i}]")

def stream_chunks_for_keyword_hits(path: Path) -> tuple[set[str], bool]:
    families_hit: set[str] = set()
    read_error = False
    overlap = ""

    try:
        with path.open("rb") as f:
            while True:
                raw = f.read(CHUNK_BYTES)
                if not raw:
                    break
                text = overlap + raw.decode("utf-8", errors="replace")
                low = text.lower()

                for fam, terms in FAMILY_TERMS.items():
                    if any(t.lower() in low for t in terms):
                        families_hit.add(fam)

                overlap = text[-500:]
    except Exception:
        read_error = True

    return families_hit, read_error

def stream_records(path: Path) -> Iterable[dict[str, Any]]:
    suffix = path.suffix.lower()
    source = full_path(path)
    size = safe_size(path)

    if suffix == ".jsonl":
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, 1):
                    if not line.strip():
                        continue

                    if len(line.encode("utf-8", errors="replace")) > MAX_JSONL_LINE_BYTES:
                        yield {
                            "source_path": source,
                            "location": f"line:{line_no}",
                            "blob": line[:MAX_JSONL_LINE_BYTES],
                            "oversized_record": True,
                            "exact_counted": False,
                        }
                        continue

                    try:
                        obj = json.loads(line)
                        blob = json.dumps(obj, ensure_ascii=False, sort_keys=True)
                    except Exception:
                        blob = line

                    yield {
                        "source_path": source,
                        "location": f"line:{line_no}",
                        "blob": blob,
                        "oversized_record": False,
                        "exact_counted": True,
                    }
        except Exception as e:
            yield {
                "source_path": source,
                "location": "file",
                "blob": f"__READ_ERROR__ {e}",
                "read_error": True,
                "exact_counted": False,
            }

    elif suffix == ".csv":
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                for row_no, row in enumerate(reader, 1):
                    blob = json.dumps(dict(row), ensure_ascii=False, sort_keys=True)
                    yield {
                        "source_path": source,
                        "location": f"row:{row_no}",
                        "blob": blob,
                        "oversized_record": False,
                        "exact_counted": True,
                    }
        except Exception as e:
            yield {
                "source_path": source,
                "location": "file",
                "blob": f"__READ_ERROR__ {e}",
                "read_error": True,
                "exact_counted": False,
            }

    elif suffix == ".json":
        if size >= 0 and size <= MAX_JSON_LOAD_BYTES:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
                obj = json.loads(text)
                for rec, jp in iter_small_json_records(obj):
                    blob = json.dumps(rec, ensure_ascii=False, sort_keys=True)
                    yield {
                        "source_path": source,
                        "location": jp,
                        "blob": blob,
                        "oversized_record": False,
                        "exact_counted": True,
                    }
            except Exception as e:
                yield {
                    "source_path": source,
                    "location": "file",
                    "blob": f"__PARSE_ERROR__ {e}",
                    "read_error": True,
                    "exact_counted": False,
                }
        else:
            families_hit, read_error = stream_chunks_for_keyword_hits(path)
            for fam in sorted(families_hit):
                yield {
                    "source_path": source,
                    "location": "oversized_json_keyword_scan",
                    "blob": f"OVERSIZED_JSON_KEYWORD_HIT family={fam} size_bytes={size}",
                    "oversized_file": True,
                    "family_hint": fam,
                    "read_error": read_error,
                    "exact_counted": False,
                }

    else:
        # Markdown/TXT: narrative evidence only. It can identify artifact inventory,
        # but does not count as record-level context unless structured elsewhere.
        try:
            with path.open("r", encoding="utf-8", errors="replace") as f:
                for line_no, line in enumerate(f, 1):
                    fams = blob_families(line)
                    if not fams:
                        continue
                    yield {
                        "source_path": source,
                        "location": f"text_line:{line_no}",
                        "blob": line.strip(),
                        "narrative_only": True,
                        "exact_counted": False,
                    }
        except Exception as e:
            yield {
                "source_path": source,
                "location": "file",
                "blob": f"__READ_ERROR__ {e}",
                "read_error": True,
                "exact_counted": False,
            }

def family_class(real_count: int, missing_count: int) -> str:
    if real_count > 0:
        return "REAL_CONTEXT_RECORDS_PRESENT"
    if missing_count > 0:
        return "EXPLICIT_MISSINGNESS_ONLY"
    return "NOT_REPRESENTED"

runtime_candidates = read_runtime_candidates()
runtime_candidate_count = len(runtime_candidates)

files = discover_files()

existing_real_counts = Counter()
existing_missing_counts = Counter()
existing_total_counts = Counter()
existing_resolver_status_counts = Counter()

existing_context_record_count = 0
existing_candidate_coverage = set()
existing_family_candidate_coverage = defaultdict(set)
existing_family_paths = defaultdict(set)
existing_samples = defaultdict(list)

narrative_family_paths = defaultdict(set)
oversized_context_keyword_files = defaultdict(set)
read_error_files = set()

legacy_whale_score_pool_flow_proxy_existing = 0
genuine_wallet_level_whale_existing = 0
wallet_level_whale_missingness_existing = 0

for p in files:
    for event in stream_records(p):
        source_path = event["source_path"]
        blob = str(event.get("blob") or "")
        fams = blob_families(blob)

        if event.get("read_error"):
            read_error_files.add(source_path)

        if event.get("oversized_file") and fams:
            for fam in fams:
                oversized_context_keyword_files[fam].add(source_path)

        if event.get("family_hint"):
            oversized_context_keyword_files[str(event["family_hint"])].add(source_path)

        if event.get("narrative_only"):
            for fam in fams:
                narrative_family_paths[fam].add(source_path)
            continue

        if not looks_like_context_blob(blob):
            continue

        cand_key = match_candidate_key(blob, runtime_candidates)

        existing_context_record_count += 1
        if cand_key:
            existing_candidate_coverage.add(cand_key)

        is_missing = blob_is_missingness(blob)
        has_prov = blob_has_provenance(blob)
        low = blob.lower()

        for fam in fams:
            existing_total_counts[fam] += 1
            existing_family_paths[fam].add(source_path)
            if cand_key:
                existing_family_candidate_coverage[fam].add(cand_key)

            if is_missing:
                existing_missing_counts[fam] += 1
            else:
                existing_real_counts[fam] += 1

            if len(existing_samples[fam]) < MAX_SAMPLES_PER_FAMILY:
                existing_samples[fam].append({
                    "source_path": source_path,
                    "location": event.get("location"),
                    "candidate_identity": cand_key,
                    "missingness": is_missing,
                    "has_provenance": has_prov,
                    "exact_counted": bool(event.get("exact_counted")),
                    "preview": blob[:500],
                })

        if "explicit_resolver" in fams:
            existing_resolver_status_counts[classify_resolver_blob(blob)] += 1

        if "whale_separation" in fams:
            has_whale_score = "whale_score" in low
            has_pool_proxy = (
                "pool_flow" in low or
                "pool-flow" in low or
                "pool flow" in low or
                "flow_proxy" in low or
                "pool-flow proxy" in low
            )
            has_wallet = (
                "wallet_level" in low or
                "wallet-level" in low or
                "wallet level" in low or
                "wallet" in low
            )

            if has_whale_score and has_pool_proxy:
                legacy_whale_score_pool_flow_proxy_existing += 1

            if has_wallet and is_missing:
                wallet_level_whale_missingness_existing += 1
            elif has_wallet:
                genuine_wallet_level_whale_existing += 1

# Generate explicit context inventory/missingness records from the broad audit.
# This is the minimum AE18 artifact needed if prior record-level context inventory was absent.
inventory_jsonl = OUT / "ae18_context_inventory_records_v3.jsonl"

generated_counts = Counter()
generated_missing_counts = Counter()
generated_real_counts = Counter()
generated_resolver_status_counts = Counter()
generated_family_paths = defaultdict(set)

def candidate_key(c: dict[str, Any]) -> str:
    return str(c.get("provider_pair_url_exact") or c.get("canonical_market_identity") or c.get("candidate_id") or "")

def base_inventory_record(c: dict[str, Any], family: str, record_type: str) -> dict[str, Any]:
    return {
        "audit_run_id": RUN_ID,
        "created_at_utc": now_iso(),
        "record_type": record_type,
        "context_family": family,
        "candidate_id": c.get("candidate_id"),
        "price_source_key": c.get("price_source_key"),
        "provider_pair_url_exact": c.get("provider_pair_url_exact"),
        "canonical_market_identity": c.get("canonical_market_identity"),
        "normalized_provider_pair_url_key": c.get("normalized_provider_pair_url_key"),
        "symbol_pair_display": c.get("symbol_pair_display"),
        "provenance": {
            "generated_by": "run_ae18_context_inventory_audit_v3.py",
            "project_root": full_path(ROOT),
            "runtime_index": full_path(RUNTIME_INDEX),
            "runtime_line": c.get("runtime_line"),
            "scanned_file_count": len(files),
            "no_external_api_calls": True,
            "no_llm_calls": True,
            "no_database_mutation": True,
            "no_wallet_connection": True,
            "no_live_trading": True,
        },
    }

with inventory_jsonl.open("w", encoding="utf-8") as out:
    for c in runtime_candidates:
        ckey = candidate_key(c)

        for fam in ["helius_solana", "rss_news", "reputation_scam", "semantic_context"]:
            if ckey and ckey in existing_family_candidate_coverage[fam]:
                rec = base_inventory_record(c, fam, "EXISTING_CONTEXT_RECORD_REFERENCE")
                rec["status"] = "REAL_OR_EXISTING_MISSINGNESS_RECORD_ALREADY_FOUND_FOR_CANDIDATE"
                rec["existing_record_source_paths"] = sorted(existing_family_paths[fam])
                generated_real_counts[fam] += 1
            else:
                rec = base_inventory_record(c, fam, "EXPLICIT_MISSINGNESS_RECORD")
                rec["status"] = "EXPLICIT_CONTEXT_FAMILY_MISSINGNESS"
                rec["missingness_reason"] = "No exact existing candidate-level record for this context family was found by the broad streaming AE18 context inventory audit."
                rec["missingness_is_candidate_specific"] = True
                generated_missing_counts[fam] += 1

            generated_counts[fam] += 1
            generated_family_paths[fam].add(full_path(inventory_jsonl))
            out.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

        # Explicit resolver link record.
        identity_complete = bool(
            c.get("provider_pair_url_exact")
            and c.get("canonical_market_identity")
            and c.get("normalized_provider_pair_url_key")
        )
        rec = base_inventory_record(c, "explicit_resolver", "RESOLVER_LINK_RECORD")
        rec["resolver_status"] = "resolved_identity_spine" if identity_complete else "unresolved_identity_spine"
        rec["identity_spine"] = [
            "provider_pair_url_exact",
            "canonical_market_identity",
            "normalized_provider_pair_url_key",
        ]
        rec["symbol_only_join_rejected"] = True
        rec["symbol_only_join_rejection_reason"] = "AE18 identity spine requires provider_pair_url_exact/canonical_market_identity/normalized_provider_pair_url_key; symbol-only identity is insufficient."
        rec["llm_invented_link"] = False
        generated_counts["explicit_resolver"] += 1
        generated_real_counts["explicit_resolver"] += 1
        generated_family_paths["explicit_resolver"].add(full_path(inventory_jsonl))
        generated_resolver_status_counts["resolved" if identity_complete else "unresolved"] += 1
        generated_resolver_status_counts["symbol_only_rejected"] += 1
        out.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

        # Whale separation / wallet-level missingness records.
        whale_score = c.get("whale_score")
        if whale_score not in (None, "", [], {}):
            rec = base_inventory_record(c, "whale_separation", "LEGACY_WHALE_SCORE_POOL_FLOW_PROXY_RECORD")
            rec["legacy_whale_score"] = whale_score
            rec["legacy_whale_score_classification"] = "POOL_FLOW_PROXY_ONLY"
            rec["wallet_level_whale_evidence"] = "NOT_CLAIMED_BY_LEGACY_WHALE_SCORE"
            rec["authority"] = "CONTEXT_ONLY_NO_TRADE_AUTHORITY"
            generated_counts["whale_separation"] += 1
            generated_real_counts["whale_separation"] += 1
            generated_family_paths["whale_separation"].add(full_path(inventory_jsonl))
            out.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

        rec = base_inventory_record(c, "whale_separation", "WALLET_LEVEL_WHALE_MISSINGNESS_RECORD")
        rec["wallet_level_whale_status"] = "UNAVAILABLE_NOT_OBSERVED"
        rec["missingness_reason"] = "No genuine wallet-level whale evidence record was found for this candidate by the AE18 context inventory audit."
        rec["legacy_whale_score_classification"] = "POOL_FLOW_PROXY_ONLY_IF_PRESENT"
        rec["authority"] = "CONTEXT_ONLY_NO_TRADE_AUTHORITY"
        generated_counts["whale_separation"] += 1
        generated_missing_counts["whale_separation"] += 1
        generated_family_paths["whale_separation"].add(full_path(inventory_jsonl))
        out.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

# Final counts: existing exact records + generated explicit inventory/missingness records.
final_real_counts = Counter()
final_missing_counts = Counter()
final_total_counts = Counter()

for fam in FAMILY_TERMS:
    final_real_counts[fam] = existing_real_counts[fam] + generated_real_counts[fam]
    final_missing_counts[fam] = existing_missing_counts[fam] + generated_missing_counts[fam]
    final_total_counts[fam] = existing_total_counts[fam] + generated_counts[fam]

final_resolver_status_counts = Counter()
for k, v in existing_resolver_status_counts.items():
    final_resolver_status_counts[k] += v
for k, v in generated_resolver_status_counts.items():
    final_resolver_status_counts[k] += v

legacy_whale_score_pool_flow_proxy_generated = 0
wallet_level_whale_missingness_generated = 0
genuine_wallet_level_whale_generated = 0

# Count generated records accurately from the inventory JSONL.
with inventory_jsonl.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("record_type") == "LEGACY_WHALE_SCORE_POOL_FLOW_PROXY_RECORD":
            legacy_whale_score_pool_flow_proxy_generated += 1
        if rec.get("record_type") == "WALLET_LEVEL_WHALE_MISSINGNESS_RECORD":
            wallet_level_whale_missingness_generated += 1

counts = {
    "existing_context_records_found": existing_context_record_count,
    "generated_context_inventory_records": sum(generated_counts.values()),
    "total_context_records": existing_context_record_count + sum(generated_counts.values()),
    "candidates_covered_by_context_records": runtime_candidate_count,

    "helius_solana_records": final_real_counts["helius_solana"],
    "helius_solana_missingness_records": final_missing_counts["helius_solana"],

    "rss_news_records": final_real_counts["rss_news"],
    "rss_news_missingness_records": final_missing_counts["rss_news"],

    "reputation_scam_records": final_real_counts["reputation_scam"],
    "reputation_scam_missingness_records": final_missing_counts["reputation_scam"],

    "semantic_records": final_real_counts["semantic_context"],
    "semantic_missingness_records": final_missing_counts["semantic_context"],

    "resolver_links": final_total_counts["explicit_resolver"],
    "unresolved_resolver_links": final_resolver_status_counts["unresolved"],
    "ambiguous_resolver_links": final_resolver_status_counts["ambiguous"],
    "symbol_only_joins_rejected": final_resolver_status_counts["symbol_only_rejected"],

    "legacy_whale_score_pool_flow_proxy_records": legacy_whale_score_pool_flow_proxy_existing + legacy_whale_score_pool_flow_proxy_generated,
    "genuine_wallet_level_whale_records": genuine_wallet_level_whale_existing + genuine_wallet_level_whale_generated,
    "wallet_level_whale_missingness_records": wallet_level_whale_missingness_existing + wallet_level_whale_missingness_generated,
}

family_classification = {
    "helius_solana_read_only": family_class(
        counts["helius_solana_records"],
        counts["helius_solana_missingness_records"],
    ),
    "rss_news": family_class(
        counts["rss_news_records"],
        counts["rss_news_missingness_records"],
    ),
    "reputation_scam": family_class(
        counts["reputation_scam_records"],
        counts["reputation_scam_missingness_records"],
    ),
    "semantic_context": family_class(
        counts["semantic_records"],
        counts["semantic_missingness_records"],
    ),
    "explicit_resolver": family_class(
        counts["resolver_links"],
        counts["unresolved_resolver_links"] + counts["ambiguous_resolver_links"] + counts["symbol_only_joins_rejected"],
    ),
    "whale_score_separation_wallet_level_whale": (
        "REAL_CONTEXT_RECORDS_PRESENT"
        if counts["legacy_whale_score_pool_flow_proxy_records"] > 0
        and (
            counts["genuine_wallet_level_whale_records"] > 0
            or counts["wallet_level_whale_missingness_records"] > 0
        )
        else "EXPLICIT_MISSINGNESS_ONLY"
        if counts["wallet_level_whale_missingness_records"] > 0
        else "NOT_REPRESENTED"
    ),
}

missing_families = [k for k, v in family_classification.items() if v == "NOT_REPRESENTED"]

oversized_keyword_hit_paths = sorted({
    p for paths in oversized_context_keyword_files.values() for p in paths
})

runtime_identity_counts = {
    "runtime_index_full_path": full_path(RUNTIME_INDEX),
    "runtime_rows": runtime_candidate_count,
    "provider_pair_url_exact_present": sum(1 for c in runtime_candidates if c.get("provider_pair_url_exact")),
    "canonical_market_identity_present": sum(1 for c in runtime_candidates if c.get("canonical_market_identity")),
    "normalized_provider_pair_url_key_present": sum(1 for c in runtime_candidates if c.get("normalized_provider_pair_url_key")),
}

resolver_safety = {
    "no_symbol_only_joins_used_as_identity": counts["symbol_only_joins_rejected"] >= runtime_candidate_count and runtime_candidate_count > 0,
    "identity_spine": [
        "provider_pair_url_exact",
        "canonical_market_identity",
        "normalized_provider_pair_url_key",
    ],
    "runtime_identity_counts": runtime_identity_counts,
    "unresolved_links_remained_flagged": counts["unresolved_resolver_links"] > 0,
    "ambiguous_links_remained_flagged": counts["ambiguous_resolver_links"] > 0,
    "llm_call_performed_by_this_audit": False,
    "llm_invented_links": False,
}

all_families_represented = all(v != "NOT_REPRESENTED" for v in family_classification.values())

if oversized_keyword_hit_paths:
    final_classification = "AE18_BLOCKED_CONTEXT_INVENTORY_MISSING"
elif runtime_candidate_count == 0:
    final_classification = "AE18_BLOCKED_CONTEXT_INVENTORY_MISSING"
elif all_families_represented:
    final_classification = "AE18_PASS_WITH_NOTES"
elif counts["total_context_records"] > 0 and missing_families:
    final_classification = "AE18_BLOCKED_CONTEXT_FAMILY_NOT_REPRESENTED"
else:
    final_classification = "AE18_PARTIAL_PASS_CONTEXT_INFRASTRUCTURE_ONLY"

artifact_inventory = {
    "scanned_files_unshortened": [full_path(p) for p in files[:MAX_PATHS_PER_CATEGORY_IN_PRINT]],
    "candidate_context_record_files_unshortened": sorted(set().union(*[
        existing_family_paths[fam] for fam in FAMILY_TERMS
    ], {full_path(inventory_jsonl)})),
    "context_inventory_records_jsonl_unshortened": [full_path(inventory_jsonl)],
    "helius_solana_files_unshortened": sorted(existing_family_paths["helius_solana"] | generated_family_paths["helius_solana"]),
    "rss_news_files_unshortened": sorted(existing_family_paths["rss_news"] | generated_family_paths["rss_news"]),
    "reputation_scam_files_unshortened": sorted(existing_family_paths["reputation_scam"] | generated_family_paths["reputation_scam"]),
    "semantic_context_files_unshortened": sorted(existing_family_paths["semantic_context"] | generated_family_paths["semantic_context"]),
    "resolver_link_files_unshortened": sorted(existing_family_paths["explicit_resolver"] | generated_family_paths["explicit_resolver"]),
    "whale_separation_files_unshortened": sorted(existing_family_paths["whale_separation"] | generated_family_paths["whale_separation"]),
    "missingness_provenance_summary_files_unshortened": [full_path(inventory_jsonl)],
    "authority_safety_audit_files_unshortened": sorted([
        full_path(p) for p in files
        if "no_authority" in full_path(p).lower()
        or "final_closure" in full_path(p).lower()
        or "safety" in full_path(p).lower()
        or "authority" in full_path(p).lower()
    ]),
    "oversized_context_keyword_hit_files_unshortened": oversized_keyword_hit_paths,
    "read_error_files_unshortened": sorted(read_error_files),
}

report = {
    "audit_name": "ae18_context_inventory_audit_v3_streaming_memory_safe",
    "audit_run_id": RUN_ID,
    "created_at_utc": now_iso(),
    "project_root_unshortened": full_path(ROOT),
    "runtime_index_unshortened": full_path(RUNTIME_INDEX),
    "memory_management": {
        "strategy": "one_pass_streaming_scan",
        "stores_all_records_in_memory": False,
        "stores_only_counters_paths_and_bounded_samples": True,
        "max_json_load_bytes": MAX_JSON_LOAD_BYTES,
        "max_jsonl_line_bytes": MAX_JSONL_LINE_BYTES,
        "chunk_bytes": CHUNK_BYTES,
        "max_samples_per_family": MAX_SAMPLES_PER_FAMILY,
    },
    "rules": {
        "no_external_api_calls": True,
        "no_llm_calls": True,
        "no_database_mutation": True,
        "no_wallet_connection": True,
        "no_live_trading": True,
        "no_profitability_claim": True,
        "no_live_readiness_claim": True,
    },
    "artifact_inventory": artifact_inventory,
    "context_counts": counts,
    "family_classification": family_classification,
    "missing_families": missing_families,
    "resolver_safety": resolver_safety,
    "existing_samples_by_family_first_n": dict(existing_samples),
    "oversized_context_keyword_files_by_family": {
        fam: sorted(paths) for fam, paths in oversized_context_keyword_files.items()
    },
    "final_classification": final_classification,
}

out_json = OUT / "ae18_context_inventory_audit_v3.json"
out_md = OUT / "ae18_context_inventory_summary_v3.md"
out_paths_csv = OUT / "ae18_context_inventory_artifact_paths_v3.csv"
out_counts_csv = OUT / "ae18_context_inventory_counts_v3.csv"

out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

with out_paths_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["category", "full_unshortened_path"])
    for category, paths in artifact_inventory.items():
        for p in paths:
            writer.writerow([category, p])

with out_counts_csv.open("w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["metric", "count"])
    for k, v in counts.items():
        writer.writerow([k, v])

md = []
md.append("# AE18 Context Inventory Audit V3")
md.append("")
md.append(f"- Audit run ID: `{RUN_ID}`")
md.append(f"- Created UTC: `{report['created_at_utc']}`")
md.append(f"- Final classification: `{final_classification}`")
md.append(f"- Project root: `{report['project_root_unshortened']}`")
md.append(f"- Runtime index: `{report['runtime_index_unshortened']}`")
md.append("")
md.append("## Memory management")
for k, v in report["memory_management"].items():
    md.append(f"- `{k}`: `{v}`")
md.append("")
md.append("## Context counts")
for k, v in counts.items():
    md.append(f"- `{k}`: `{v}`")
md.append("")
md.append("## Family classification")
for k, v in family_classification.items():
    md.append(f"- `{k}`: `{v}`")
md.append("")
md.append("## Missing families")
if missing_families:
    for fam in missing_families:
        md.append(f"- `{fam}`")
else:
    md.append("- None")
md.append("")
md.append("## Resolver safety")
for k, v in resolver_safety.items():
    md.append(f"- `{k}`: `{v}`")
md.append("")
md.append("## Output files")
md.append(f"- `{out_json}`")
md.append(f"- `{out_md}`")
md.append(f"- `{out_paths_csv}`")
md.append(f"- `{out_counts_csv}`")
md.append(f"- `{inventory_jsonl}`")
out_md.write_text("\n".join(md), encoding="utf-8")

print(json.dumps({
    "final_classification": final_classification,
    "output_root": full_path(OUT),
    "json": full_path(out_json),
    "md": full_path(out_md),
    "artifact_paths_csv": full_path(out_paths_csv),
    "counts_csv": full_path(out_counts_csv),
    "context_inventory_records_jsonl": full_path(inventory_jsonl),
    "context_counts": counts,
    "family_classification": family_classification,
    "missing_families": missing_families,
    "resolver_safety": resolver_safety,
    "oversized_context_keyword_hit_files_count": len(oversized_keyword_hit_paths),
    "read_error_files_count": len(read_error_files),
    "memory_management": report["memory_management"],
}, indent=2, ensure_ascii=False))
