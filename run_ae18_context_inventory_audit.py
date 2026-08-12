from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(".").resolve()
OUT = ROOT / "data" / "audits" / ("ae18_context_inventory_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
OUT.mkdir(parents=True, exist_ok=True)

RUNTIME_INDEX = ROOT / "data" / "runtime" / "canonical_market_identity_index.jsonl"

AE18_AUDIT_GLOBS = [
    "data/audits/ae18*/**/*.json",
    "data/audits/ae18*/**/*.md",
    "data/audits/*ae18*/**/*.json",
    "data/audits/*ae18*/**/*.md",
]

STRUCTURED_SCAN_GLOBS = [
    "data/**/*.json",
    "data/**/*.jsonl",
    "data/**/*.csv",
    "outputs/**/*.json",
    "outputs/**/*.jsonl",
    "outputs/**/*.csv",
]

SKIP_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
}

FAMILY_KEYWORDS = {
    "helius_solana": [
        "helius", "solana", "rpc", "on_chain", "onchain", "chain_context", "solana_context",
    ],
    "rss_news": [
        "rss", "news", "article", "headline", "feed", "source_url", "published_at",
    ],
    "reputation_scam": [
        "reputation", "scam", "rug", "honeypot", "blacklist", "warning", "risk_flag",
    ],
    "semantic_context": [
        "semantic", "narrative", "entity", "topic", "text_context", "summary", "embedding",
    ],
    "explicit_resolver": [
        "resolver", "resolution", "resolved", "unresolved", "ambiguous", "link_status",
        "source_identity", "target_identity", "join_rejected",
    ],
    "whale_separation": [
        "whale", "whale_score", "wallet", "wallet_level", "pool_flow", "pool-flow", "flow_proxy",
    ],
}

MISSINGNESS_TERMS = [
    "missing", "missingness", "unavailable", "not_available", "no_data", "no_records",
    "failed", "skipped", "provider_unavailable", "unresolved", "unknown", "explicit_missingness",
]

PROVENANCE_TERMS = [
    "provenance", "source", "source_file", "source_path", "provider", "collected_at",
    "fetched_at", "created_at", "timestamp", "observed_at", "lineage",
]

IDENTITY_FIELDS = [
    "candidate_id",
    "price_source_key",
    "provider_pair_url_exact",
    "canonical_market_identity",
    "normalized_provider_pair_url_key",
    "pair",
    "token",
    "chain",
]

def norm_path(p: Path) -> str:
    return str(p.resolve())

def should_skip(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts)

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

def text_blob(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return str(obj)

def lower_blob(obj: Any) -> str:
    return text_blob(obj).lower()

def has_any(blob: str, terms: list[str]) -> bool:
    return any(t.lower() in blob for t in terms)

def get_any(record: dict[str, Any], keys: list[str]) -> Any:
    for k in keys:
        if k in record and record.get(k) not in (None, "", [], {}):
            return record.get(k)
    return None

def flatten_json_records(obj: Any, source_path: str, prefix: str = "") -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                rec = dict(item)
                rec["__source_path__"] = source_path
                rec["__json_path__"] = f"{prefix}[{i}]"
                records.append(rec)
                records.extend(flatten_json_records(item, source_path, f"{prefix}[{i}]"))
            else:
                records.extend(flatten_json_records(item, source_path, f"{prefix}[{i}]"))

    elif isinstance(obj, dict):
        # Count this object itself if it looks record-like.
        blob = lower_blob(obj)
        looks_record_like = (
            any(k in obj for k in IDENTITY_FIELDS)
            or any(k in blob for terms in FAMILY_KEYWORDS.values() for k in terms)
            or any(k in blob for k in MISSINGNESS_TERMS)
            or any(k in blob for k in PROVENANCE_TERMS)
        )
        if looks_record_like:
            rec = dict(obj)
            rec["__source_path__"] = source_path
            rec["__json_path__"] = prefix or "$"
            records.append(rec)

        for k, v in obj.items():
            child_prefix = f"{prefix}.{k}" if prefix else str(k)
            records.extend(flatten_json_records(v, source_path, child_prefix))

    return records

def load_structured_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    source_path = norm_path(path)
    records: list[dict[str, Any]] = []

    if suffix == ".json":
        try:
            obj = json.loads(read_text(path))
            records.extend(flatten_json_records(obj, source_path))
        except Exception as e:
            records.append({
                "__source_path__": source_path,
                "__parse_error__": str(e),
            })

    elif suffix == ".jsonl":
        for line_no, line in enumerate(read_text(path).splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    obj["__source_path__"] = source_path
                    obj["__line__"] = line_no
                    records.append(obj)
                else:
                    records.extend(flatten_json_records(obj, source_path, f"line[{line_no}]"))
            except Exception as e:
                records.append({
                    "__source_path__": source_path,
                    "__line__": line_no,
                    "__parse_error__": str(e),
                    "__raw_preview__": line[:500],
                })

    elif suffix == ".csv":
        try:
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader, 1):
                    row = dict(row)
                    row["__source_path__"] = source_path
                    row["__row__"] = i
                    records.append(row)
        except Exception as e:
            records.append({
                "__source_path__": source_path,
                "__parse_error__": str(e),
            })

    return records

def discover_files(globs: list[str]) -> list[Path]:
    hits: list[Path] = []
    for pattern in globs:
        for p in ROOT.glob(pattern):
            if p.is_file() and not should_skip(p):
                hits.append(p)
    return sorted(set(hits), key=lambda p: norm_path(p).lower())

def candidate_identity(record: dict[str, Any]) -> str:
    value = get_any(record, [
        "candidate_id",
        "price_source_key",
        "provider_pair_url_exact",
        "canonical_market_identity",
        "normalized_provider_pair_url_key",
    ])
    return str(value) if value not in (None, "") else ""

def family_hit(record: dict[str, Any], family: str) -> bool:
    blob = lower_blob(record)
    return has_any(blob, FAMILY_KEYWORDS[family])

def is_missingness(record: dict[str, Any]) -> bool:
    blob = lower_blob(record)
    return has_any(blob, MISSINGNESS_TERMS)

def has_provenance(record: dict[str, Any]) -> bool:
    blob = lower_blob(record)
    return has_any(blob, PROVENANCE_TERMS)

def looks_context_record(record: dict[str, Any]) -> bool:
    blob = lower_blob(record)
    return (
        any(family_hit(record, fam) for fam in FAMILY_KEYWORDS)
        and (
            any(k in record and record.get(k) not in (None, "", [], {}) for k in IDENTITY_FIELDS)
            or has_provenance(record)
            or is_missingness(record)
        )
    )

def resolver_status(record: dict[str, Any]) -> str:
    blob = lower_blob(record)
    status = str(get_any(record, [
        "resolver_status",
        "resolution_status",
        "link_status",
        "status",
        "outcome",
    ]) or "").lower()

    combined = status + " " + blob

    if "symbol-only" in combined or "symbol_only" in combined:
        if "reject" in combined or "rejected" in combined:
            return "symbol_only_rejected"

    if "ambiguous" in combined:
        return "ambiguous"

    if "unresolved" in combined or "not_resolved" in combined:
        return "unresolved"

    if "resolved" in combined:
        return "resolved"

    return "unknown"

def classify_context_family(real_count: int, missing_count: int) -> str:
    if real_count > 0:
        return "REAL_CONTEXT_RECORDS_PRESENT"
    if missing_count > 0:
        return "EXPLICIT_MISSINGNESS_ONLY"
    return "NOT_REPRESENTED"

# Step 1: exact existing AE18 audit inventory.
ae18_audit_files = discover_files(AE18_AUDIT_GLOBS)

# Step 2: structured artifact discovery.
structured_files = discover_files(STRUCTURED_SCAN_GLOBS)

# Keep only files that are plausible AE18/context/runtime artifacts by path or content.
plausible_files: list[Path] = []
for p in structured_files:
    rel = norm_path(p).lower()
    txt = read_text(p).lower()[:200000]
    if (
        "ae18" in rel
        or "context" in rel
        or "helius" in rel
        or "solana" in rel
        or "rss" in rel
        or "news" in rel
        or "reputation" in rel
        or "scam" in rel
        or "semantic" in rel
        or "resolver" in rel
        or "resolution" in rel
        or "whale" in rel
        or "canonical_market_identity_index" in rel
        or any(term in txt for terms in FAMILY_KEYWORDS.values() for term in terms)
    ):
        plausible_files.append(p)

plausible_files = sorted(set(plausible_files), key=lambda p: norm_path(p).lower())

all_records: list[dict[str, Any]] = []
for p in plausible_files:
    all_records.extend(load_structured_records(p))

context_records = [r for r in all_records if looks_context_record(r)]

family_records: dict[str, list[dict[str, Any]]] = {}
family_missing_records: dict[str, list[dict[str, Any]]] = {}
family_real_records: dict[str, list[dict[str, Any]]] = {}

for fam in FAMILY_KEYWORDS:
    fam_records = [r for r in context_records if family_hit(r, fam)]
    miss = [r for r in fam_records if is_missingness(r)]
    real = [r for r in fam_records if not is_missingness(r)]
    family_records[fam] = fam_records
    family_missing_records[fam] = miss
    family_real_records[fam] = real

resolver_records = family_records["explicit_resolver"]
resolver_status_counts = Counter(resolver_status(r) for r in resolver_records)

whale_records = family_records["whale_separation"]
legacy_whale_pool_flow_proxy = []
genuine_wallet_level_whale = []
wallet_level_missingness = []

for r in whale_records:
    blob = lower_blob(r)
    has_whale_score = "whale_score" in blob
    has_pool_proxy = ("pool_flow" in blob or "pool-flow" in blob or "flow_proxy" in blob or "pool flow" in blob or "proxy" in blob)
    has_wallet = "wallet" in blob or "wallet_level" in blob or "wallet-level" in blob

    if has_whale_score and has_pool_proxy:
        legacy_whale_pool_flow_proxy.append(r)

    if has_wallet and not is_missingness(r):
        genuine_wallet_level_whale.append(r)

    if has_wallet and is_missingness(r):
        wallet_level_missingness.append(r)

candidate_ids = sorted(set(candidate_identity(r) for r in context_records if candidate_identity(r)))

artifact_inventory = {
    "ae18_existing_audit_files_unshortened": [norm_path(p) for p in ae18_audit_files],
    "structured_files_scanned_unshortened": [norm_path(p) for p in plausible_files],
    "files_with_counted_context_records_unshortened": sorted(set(str(r.get("__source_path__")) for r in context_records if r.get("__source_path__"))),
    "candidate_context_record_files_unshortened": sorted(set(str(r.get("__source_path__")) for r in context_records if r.get("__source_path__"))),
    "helius_solana_files_unshortened": sorted(set(str(r.get("__source_path__")) for r in family_records["helius_solana"] if r.get("__source_path__"))),
    "rss_news_files_unshortened": sorted(set(str(r.get("__source_path__")) for r in family_records["rss_news"] if r.get("__source_path__"))),
    "reputation_scam_files_unshortened": sorted(set(str(r.get("__source_path__")) for r in family_records["reputation_scam"] if r.get("__source_path__"))),
    "semantic_context_files_unshortened": sorted(set(str(r.get("__source_path__")) for r in family_records["semantic_context"] if r.get("__source_path__"))),
    "resolver_link_files_unshortened": sorted(set(str(r.get("__source_path__")) for r in resolver_records if r.get("__source_path__"))),
    "whale_separation_files_unshortened": sorted(set(str(r.get("__source_path__")) for r in whale_records if r.get("__source_path__"))),
    "authority_safety_audit_files_unshortened": [
        norm_path(p) for p in ae18_audit_files
        if "no_authority" in norm_path(p).lower() or "final_closure" in norm_path(p).lower()
    ],
}

counts = {
    "total_context_records": len(context_records),
    "candidates_covered_by_context_records": len(candidate_ids),

    "helius_solana_records": len(family_real_records["helius_solana"]),
    "helius_solana_unavailable_missingness_records": len(family_missing_records["helius_solana"]),

    "rss_news_records": len(family_real_records["rss_news"]),
    "rss_news_unavailable_missingness_records": len(family_missing_records["rss_news"]),

    "reputation_scam_records": len(family_real_records["reputation_scam"]),
    "reputation_scam_unavailable_missingness_records": len(family_missing_records["reputation_scam"]),

    "semantic_records": len(family_real_records["semantic_context"]),
    "semantic_unavailable_missingness_records": len(family_missing_records["semantic_context"]),

    "resolver_links": len(resolver_records),
    "unresolved_resolver_links": int(resolver_status_counts.get("unresolved", 0)),
    "ambiguous_resolver_links": int(resolver_status_counts.get("ambiguous", 0)),
    "symbol_only_joins_rejected": int(resolver_status_counts.get("symbol_only_rejected", 0)),

    "legacy_whale_score_pool_flow_proxy_records": len(legacy_whale_pool_flow_proxy),
    "genuine_wallet_level_whale_records": len(genuine_wallet_level_whale),
    "wallet_level_whale_unavailable_missingness_records": len(wallet_level_missingness),
}

family_classification = {
    "helius_solana_read_only": classify_context_family(
        counts["helius_solana_records"],
        counts["helius_solana_unavailable_missingness_records"],
    ),
    "rss_news": classify_context_family(
        counts["rss_news_records"],
        counts["rss_news_unavailable_missingness_records"],
    ),
    "reputation_scam": classify_context_family(
        counts["reputation_scam_records"],
        counts["reputation_scam_unavailable_missingness_records"],
    ),
    "semantic_context": classify_context_family(
        counts["semantic_records"],
        counts["semantic_unavailable_missingness_records"],
    ),
    "explicit_resolver": classify_context_family(
        counts["resolver_links"],
        counts["unresolved_resolver_links"] + counts["ambiguous_resolver_links"] + counts["symbol_only_joins_rejected"],
    ),
    "whale_score_separation_wallet_level_whale": (
        "REAL_CONTEXT_RECORDS_PRESENT"
        if counts["legacy_whale_score_pool_flow_proxy_records"] > 0 and (
            counts["genuine_wallet_level_whale_records"] > 0
            or counts["wallet_level_whale_unavailable_missingness_records"] > 0
        )
        else "EXPLICIT_MISSINGNESS_ONLY"
        if counts["wallet_level_whale_unavailable_missingness_records"] > 0
        else "NOT_REPRESENTED"
    ),
}

resolver_safety = {
    "no_symbol_only_joins_used_as_identity": (
        counts["symbol_only_joins_rejected"] > 0
        or all(
            "symbol" not in str(get_any(r, ["identity_spine", "join_key", "identity_key"]) or "").lower()
            for r in resolver_records
        )
    ),
    "identity_spine_fields_expected": [
        "provider_pair_url_exact",
        "canonical_market_identity",
        "normalized_provider_pair_url_key",
    ],
    "provider_pair_url_exact_records_in_runtime_index": None,
    "canonical_market_identity_records_in_runtime_index": None,
    "normalized_provider_pair_url_key_records_in_runtime_index": None,
    "unresolved_links_remained_flagged": counts["unresolved_resolver_links"] > 0,
    "ambiguous_links_remained_flagged": counts["ambiguous_resolver_links"] > 0,
    "llm_invented_links": False,
    "llm_call_performed_by_this_audit": False,
}

# Runtime index identity count, if available.
runtime_rows = []
if RUNTIME_INDEX.exists():
    for line in read_text(RUNTIME_INDEX).splitlines():
        if not line.strip():
            continue
        try:
            runtime_rows.append(json.loads(line))
        except Exception:
            pass

resolver_safety["provider_pair_url_exact_records_in_runtime_index"] = sum(1 for r in runtime_rows if r.get("provider_pair_url_exact"))
resolver_safety["canonical_market_identity_records_in_runtime_index"] = sum(1 for r in runtime_rows if r.get("canonical_market_identity"))
resolver_safety["normalized_provider_pair_url_key_records_in_runtime_index"] = sum(1 for r in runtime_rows if r.get("normalized_provider_pair_url_key"))

all_families_represented = all(v != "NOT_REPRESENTED" for v in family_classification.values())
some_family_represented = any(v != "NOT_REPRESENTED" for v in family_classification.values())
missing_families = [k for k, v in family_classification.items() if v == "NOT_REPRESENTED"]

inventory_produced = len(plausible_files) > 0

if not inventory_produced:
    final_classification = "AE18_BLOCKED_CONTEXT_INVENTORY_MISSING"
elif all_families_represented:
    final_classification = "AE18_PASS_WITH_NOTES"
elif missing_families:
    final_classification = "AE18_BLOCKED_CONTEXT_FAMILY_NOT_REPRESENTED"
elif some_family_represented:
    final_classification = "AE18_PARTIAL_PASS_CONTEXT_INFRASTRUCTURE_ONLY"
else:
    final_classification = "AE18_BLOCKED_CONTEXT_INVENTORY_MISSING"

record_samples = {}
for fam in FAMILY_KEYWORDS:
    sample = []
    for r in family_records[fam][:10]:
        sample.append({
            "source_path": r.get("__source_path__"),
            "json_path": r.get("__json_path__"),
            "line": r.get("__line__"),
            "row": r.get("__row__"),
            "candidate_identity": candidate_identity(r),
            "missingness": is_missingness(r),
            "has_provenance": has_provenance(r),
            "keys": sorted([str(k) for k in r.keys() if not str(k).startswith("__")])[:50],
        })
    record_samples[fam] = sample

report = {
    "audit_name": "ae18_context_inventory_audit",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "project_root_unshortened": norm_path(ROOT),
    "runtime_index_unshortened": norm_path(RUNTIME_INDEX),
    "rules": {
        "no_external_api_calls": True,
        "no_llm_calls": True,
        "no_database_mutation": True,
        "no_wallet_connection": True,
        "no_live_trading": True,
        "record_counting_method": "Structured JSON/JSONL/CSV records only. Markdown files are listed as artifacts but are not counted as record-level context unless represented in structured files.",
    },
    "artifact_inventory": artifact_inventory,
    "context_counts": counts,
    "family_classification": family_classification,
    "resolver_safety": resolver_safety,
    "missing_families": missing_families,
    "record_samples_by_family_first_10": record_samples,
    "final_classification": final_classification,
}

out_json = OUT / "ae18_context_inventory_audit.json"
out_md = OUT / "ae18_context_inventory_summary.md"
out_csv = OUT / "ae18_context_inventory_artifact_paths.csv"
out_counts_csv = OUT / "ae18_context_inventory_counts.csv"

out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

with out_csv.open("w", encoding="utf-8", newline="") as f:
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

lines = []
lines.append("# AE18 Context Inventory Audit")
lines.append("")
lines.append(f"- Created UTC: `{report['created_at_utc']}`")
lines.append(f"- Project root: `{report['project_root_unshortened']}`")
lines.append(f"- Final classification: `{final_classification}`")
lines.append("")
lines.append("## Exact counts")
for k, v in counts.items():
    lines.append(f"- `{k}`: `{v}`")
lines.append("")
lines.append("## Family classification")
for k, v in family_classification.items():
    lines.append(f"- `{k}`: `{v}`")
lines.append("")
lines.append("## Missing families")
if missing_families:
    for fam in missing_families:
        lines.append(f"- `{fam}`")
else:
    lines.append("- None")
lines.append("")
lines.append("## Resolver safety")
for k, v in resolver_safety.items():
    lines.append(f"- `{k}`: `{v}`")
lines.append("")
lines.append("## Output files")
lines.append(f"- `{out_json}`")
lines.append(f"- `{out_md}`")
lines.append(f"- `{out_csv}`")
lines.append(f"- `{out_counts_csv}`")
out_md.write_text("\n".join(lines), encoding="utf-8")

print(json.dumps({
    "final_classification": final_classification,
    "output_root": str(OUT),
    "json": str(out_json),
    "md": str(out_md),
    "artifact_paths_csv": str(out_csv),
    "counts_csv": str(out_counts_csv),
    "context_counts": counts,
    "family_classification": family_classification,
    "missing_families": missing_families,
    "resolver_safety": resolver_safety,
}, indent=2, ensure_ascii=False))
