#!/usr/bin/env python3
"""AE16B — Combined Clean Universe Builder (offline / read-only).

Merges Clean Forward candidate targets with the user DexScreener seed list.
No network, no DexScreener provider, no server, no trader.db mutation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PHASE = "AE16B_COMBINED_CLEAN_UNIVERSE"
SEMANTIC_PENDING = "PENDING_SYSTEM_CLASSIFICATION"

SOURCE_CLEAN = "CLEAN_FORWARD_EXISTING"
SOURCE_SEED = "USER_DEXSCREENER_SEED"
SOURCE_MERGED = "MERGED"

SEED_COLLECTION_CLEAN = "EXISTING_CLEAN_FORWARD"

NON_EVM_CHAINS = {"solana", "xrpl"}

DEFAULT_USER_SEED = Path("data/SeedTargets/dexscreener_seed_targets_v1.csv")
DEFAULT_CLEAN_CANDIDATES = Path(
    "data/audits/ae15_cleaned_for_ae16_20260722_194200/data/ae16_clean_forward_candidates.csv"
)

COMBINED_FIELDS = [
    "combined_target_id",
    "active",
    "chain",
    "provider_pair_url",
    "pair_address",
    "user_supplied_pair_address",
    "resolved_pair_address",
    "resolved_base_token_address",
    "resolved_quote_token_address",
    "symbol_pair",
    "target_source",
    "linked_sources",
    "seed_collection",
    "semantic_status",
    "source_clean_forward_candidate_id",
    "source_clean_forward_row_key",
    "source_provider_payload_hash",
    "duplicate_group_id",
    "notes",
]

EXISTING_FIELDS = [
    "combined_target_id",
    "active",
    "chain",
    "provider_pair_url",
    "pair_address",
    "user_supplied_pair_address",
    "resolved_pair_address",
    "resolved_base_token_address",
    "resolved_quote_token_address",
    "symbol_pair",
    "target_source",
    "linked_sources",
    "seed_collection",
    "semantic_status",
    "clean_forward_candidate_id",
    "source_clean_forward_row_key",
    "provider_payload_hash",
    "base_token_address",
    "quote_token_address",
    "notes",
]

SEED_OUT_FIELDS = [
    "combined_target_id",
    "active",
    "chain",
    "provider_pair_url",
    "pair_address",
    "user_supplied_pair_address",
    "resolved_pair_address",
    "resolved_base_token_address",
    "resolved_quote_token_address",
    "symbol_pair",
    "target_source",
    "linked_sources",
    "seed_collection",
    "semantic_status",
    "source_seed_target_id",
    "notes",
]

DUPLICATE_FIELDS = [
    "duplicate_group_id",
    "duplicate_reason",
    "kept_combined_target_id",
    "dropped_source",
    "dropped_identity",
    "chain",
    "provider_pair_url",
    "pair_address",
    "user_supplied_pair_address",
    "resolved_pair_address",
    "symbol_pair",
    "seed_collection",
    "linked_sources",
    "notes",
]

REJECTED_FIELDS = [
    "rejection_reason",
    "source",
    "chain",
    "provider_pair_url",
    "pair_address",
    "user_supplied_pair_address",
    "resolved_pair_address",
    "symbol_pair",
    "seed_collection",
    "raw_identity",
    "notes",
]


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_blank(value: Any) -> bool:
    return str(value or "").strip() == ""


def cell(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def chain_norm(chain: str) -> str:
    return cell(chain).lower()


def is_non_evm(chain: str) -> bool:
    return chain_norm(chain) in NON_EVM_CHAINS


def address_match_key(chain: str, address: str) -> str | None:
    addr = cell(address)
    if not addr:
        return None
    ch = chain_norm(chain)
    if not ch:
        return None
    if is_non_evm(ch):
        return f"{ch}|{addr}"
    return f"{ch}|{addr.lower()}"


def url_match_key(chain: str, url: str) -> str | None:
    u = cell(url).rstrip("/")
    if not u:
        return None
    ch = chain_norm(chain)
    if not ch:
        return None
    if is_non_evm(ch):
        return f"{ch}|{u}"
    return f"{ch}|{u.lower()}"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def validate_inputs(user_seed_path: Path, clean_path: Path) -> tuple[bool, bool, list[str]]:
    seed_ok = user_seed_path.exists()
    clean_ok = clean_path.exists()
    errors: list[str] = []
    if not seed_ok:
        errors.append(f"User seed CSV not found: {user_seed_path}")
    if not clean_ok:
        errors.append(f"Clean Forward candidates CSV not found: {clean_path}")
    return seed_ok, clean_ok, errors


def has_usable_identity(chain: str, url: str, *addresses: str) -> bool:
    if is_blank(chain):
        return False
    if not is_blank(url):
        return True
    return any(not is_blank(a) for a in addresses)


def rejection_reason(chain: str, url: str, *addresses: str) -> str:
    reasons: list[str] = []
    if is_blank(chain):
        reasons.append("missing_chain")
    if is_blank(url) and not any(not is_blank(a) for a in addresses):
        reasons.append("missing_provider_pair_url_and_pair_address")
    elif is_blank(url) and is_blank(chain):
        reasons.append("missing_provider_pair_url")
    if not reasons:
        reasons.append("no_usable_token_or_pair_reference")
    return ";".join(reasons)


def stable_id(*parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]
    return f"ae16b_{digest}"


def empty_target() -> dict[str, str]:
    return {k: "" for k in COMBINED_FIELDS}


def from_clean_row(row: dict[str, str], index: int) -> dict[str, str]:
    chain = cell(row.get("chain"))
    pair_address = cell(row.get("pair_address"))
    url = cell(row.get("provider_pair_url"))
    candidate_id = cell(row.get("clean_forward_candidate_id"))
    row_key = cell(row.get("source_clean_forward_row_key"))
    payload_hash = cell(row.get("provider_payload_hash"))
    base = cell(row.get("base_token_address"))
    quote = cell(row.get("quote_token_address"))
    symbol = cell(row.get("symbol_pair"))

    out = empty_target()
    out.update(
        {
            "combined_target_id": stable_id("clean", chain, url or pair_address, candidate_id or str(index)),
            "active": "true",
            "chain": chain,
            "provider_pair_url": url,
            "pair_address": pair_address,
            "user_supplied_pair_address": "",
            "resolved_pair_address": pair_address,
            "resolved_base_token_address": base,
            "resolved_quote_token_address": quote,
            "symbol_pair": symbol,
            "target_source": SOURCE_CLEAN,
            "linked_sources": SOURCE_CLEAN,
            "seed_collection": SEED_COLLECTION_CLEAN,
            "semantic_status": SEMANTIC_PENDING,
            "source_clean_forward_candidate_id": candidate_id,
            "source_clean_forward_row_key": row_key,
            "source_provider_payload_hash": payload_hash,
            "duplicate_group_id": "",
            "notes": "extracted from ae16_clean_forward_candidates; semantic classification pending",
            # retained for existing-targets extract
            "clean_forward_candidate_id": candidate_id,
            "provider_payload_hash": payload_hash,
            "base_token_address": base,
            "quote_token_address": quote,
        }
    )
    return out


def from_seed_row(row: dict[str, str], index: int) -> dict[str, str]:
    chain = cell(row.get("chain"))
    url = cell(row.get("provider_pair_url"))
    user_pair = cell(row.get("user_supplied_pair_address"))
    seed_collection = cell(row.get("seed_collection"))
    seed_id = cell(row.get("target_id")) or f"seed_{index}"
    active = cell(row.get("active")) or "true"
    notes = cell(row.get("notes"))

    out = empty_target()
    out.update(
        {
            "combined_target_id": stable_id("seed", chain, url or user_pair, seed_id),
            "active": active.lower() if active else "true",
            "chain": chain,
            "provider_pair_url": url,
            "pair_address": "",
            "user_supplied_pair_address": user_pair,
            "resolved_pair_address": "",
            "resolved_base_token_address": "",
            "resolved_quote_token_address": "",
            "symbol_pair": "",
            "target_source": SOURCE_SEED,
            "linked_sources": SOURCE_SEED,
            "seed_collection": seed_collection,
            "semantic_status": SEMANTIC_PENDING,
            "source_clean_forward_candidate_id": "",
            "source_clean_forward_row_key": "",
            "source_provider_payload_hash": "",
            "duplicate_group_id": "",
            "notes": notes
            or "user seed category is provenance only; final semantic classification must be system-derived",
            "source_seed_target_id": seed_id,
        }
    )
    return out


def row_addresses(row: dict[str, str]) -> list[str]:
    return [
        cell(row.get("pair_address")),
        cell(row.get("user_supplied_pair_address")),
        cell(row.get("resolved_pair_address")),
    ]


def find_match(
    row: dict[str, str],
    url_index: dict[str, int],
    addr_index: dict[str, int],
) -> tuple[int | None, str | None]:
    uk = url_match_key(row["chain"], row.get("provider_pair_url", ""))
    if uk and uk in url_index:
        return url_index[uk], "provider_pair_url"

    pair_keys: list[tuple[str, str]] = []
    for field in ("pair_address", "user_supplied_pair_address"):
        ak = address_match_key(row["chain"], row.get(field, ""))
        if ak:
            pair_keys.append((ak, "pair_address"))
    for ak, reason in pair_keys:
        if ak in addr_index:
            return addr_index[ak], reason

    rk = address_match_key(row["chain"], row.get("resolved_pair_address", ""))
    if rk and rk in addr_index:
        return addr_index[rk], "resolved_pair_address"

    return None, None


def index_row(row: dict[str, str], idx: int, url_index: dict[str, int], addr_index: dict[str, int]) -> None:
    uk = url_match_key(row["chain"], row.get("provider_pair_url", ""))
    if uk and uk not in url_index:
        url_index[uk] = idx
    for addr in row_addresses(row):
        ak = address_match_key(row["chain"], addr)
        if ak and ak not in addr_index:
            addr_index[ak] = idx


def linked_source_set(value: str) -> list[str]:
    parts = [p for p in cell(value).split(";") if p]
    ordered: list[str] = []
    for preferred in (SOURCE_CLEAN, SOURCE_SEED):
        if preferred in parts and preferred not in ordered:
            ordered.append(preferred)
    for p in parts:
        if p not in ordered:
            ordered.append(p)
    return ordered


def _fill_blank(kept: dict[str, str], incoming: dict[str, str], field: str) -> None:
    if is_blank(kept.get(field)) and not is_blank(incoming.get(field)):
        kept[field] = incoming[field]


def merge_rows(kept: dict[str, str], incoming: dict[str, str], reason: str, group_id: str) -> dict[str, str]:
    sources = linked_source_set(kept.get("linked_sources", ""))
    for s in linked_source_set(incoming.get("linked_sources", "")):
        if s not in sources:
            sources.append(s)

    if SOURCE_CLEAN in sources and SOURCE_SEED in sources:
        kept["target_source"] = SOURCE_MERGED
        kept["linked_sources"] = f"{SOURCE_CLEAN};{SOURCE_SEED}"
    else:
        kept["linked_sources"] = ";".join(sources)
        if len(sources) == 1:
            kept["target_source"] = sources[0]

    # Preserve Clean Forward identity / resolved fields (first non-blank wins).
    for field in (
        "source_clean_forward_candidate_id",
        "source_clean_forward_row_key",
        "source_provider_payload_hash",
        "pair_address",
        "resolved_pair_address",
        "resolved_base_token_address",
        "resolved_quote_token_address",
        "symbol_pair",
        "provider_pair_url",
        "user_supplied_pair_address",
    ):
        _fill_blank(kept, incoming, field)

    # seed_collection is provenance only — prefer user seed provenance on merge.
    incoming_seed = cell(incoming.get("seed_collection"))
    if incoming_seed and incoming_seed != SEED_COLLECTION_CLEAN:
        existing_seed = cell(kept.get("seed_collection"))
        if existing_seed in {"", SEED_COLLECTION_CLEAN}:
            kept["seed_collection"] = incoming_seed
        elif incoming_seed != existing_seed:
            note_extra = f"additional_seed_collection={incoming_seed}"
            kept["notes"] = (
                f"{kept.get('notes', '').rstrip()}; {note_extra}" if kept.get("notes") else note_extra
            )

    kept["semantic_status"] = SEMANTIC_PENDING
    kept["duplicate_group_id"] = group_id or kept.get("duplicate_group_id", "")
    merge_note = f"merged_via={reason}"
    if kept.get("notes"):
        if merge_note not in kept["notes"]:
            kept["notes"] = f"{kept['notes'].rstrip()}; {merge_note}"
    else:
        kept["notes"] = merge_note
    return kept


def duplicate_audit_row(
    kept: dict[str, str],
    incoming: dict[str, str],
    reason: str,
    group_id: str,
) -> dict[str, str]:
    identity = (
        cell(incoming.get("source_seed_target_id"))
        or cell(incoming.get("source_clean_forward_candidate_id"))
        or cell(incoming.get("combined_target_id"))
    )
    return {
        "duplicate_group_id": group_id,
        "duplicate_reason": reason,
        "kept_combined_target_id": kept.get("combined_target_id", ""),
        "dropped_source": incoming.get("target_source", ""),
        "dropped_identity": identity,
        "chain": incoming.get("chain", ""),
        "provider_pair_url": incoming.get("provider_pair_url", ""),
        "pair_address": incoming.get("pair_address", ""),
        "user_supplied_pair_address": incoming.get("user_supplied_pair_address", ""),
        "resolved_pair_address": incoming.get("resolved_pair_address", ""),
        "symbol_pair": incoming.get("symbol_pair", ""),
        "seed_collection": incoming.get("seed_collection", ""),
        "linked_sources": incoming.get("linked_sources", ""),
        "notes": f"duplicate of {kept.get('combined_target_id', '')} via {reason}",
    }


def rejected_row(source: str, row: dict[str, str], reason: str) -> dict[str, str]:
    return {
        "rejection_reason": reason,
        "source": source,
        "chain": cell(row.get("chain")),
        "provider_pair_url": cell(row.get("provider_pair_url")),
        "pair_address": cell(row.get("pair_address")),
        "user_supplied_pair_address": cell(row.get("user_supplied_pair_address")),
        "resolved_pair_address": cell(row.get("resolved_pair_address")),
        "symbol_pair": cell(row.get("symbol_pair")),
        "seed_collection": cell(row.get("seed_collection")),
        "raw_identity": cell(row.get("clean_forward_candidate_id")) or cell(row.get("target_id")),
        "notes": "rejected: no usable target identity",
    }


def build_universe(
    clean_rows: list[dict[str, str]],
    seed_rows: list[dict[str, str]],
) -> dict[str, Any]:
    existing_targets: list[dict[str, str]] = []
    user_seed_targets: list[dict[str, str]] = []
    combined: list[dict[str, str]] = []
    duplicates: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []
    reason_counter: Counter[str] = Counter()

    url_index: dict[str, int] = {}
    addr_index: dict[str, int] = {}
    dup_group_seq = 0

    def next_group() -> str:
        nonlocal dup_group_seq
        dup_group_seq += 1
        return f"dup_{dup_group_seq:04d}"

    def ingest(normalized: dict[str, str], extract_bucket: list[dict[str, str]], source_label: str) -> None:
        chain = normalized["chain"]
        url = normalized.get("provider_pair_url", "")
        addrs = row_addresses(normalized)
        if not has_usable_identity(chain, url, *addrs):
            rejected.append(
                rejected_row(
                    source_label,
                    normalized,
                    rejection_reason(chain, url, *addrs),
                )
            )
            return

        extract_bucket.append(dict(normalized))

        match_idx, reason = find_match(normalized, url_index, addr_index)
        if match_idx is None:
            combined.append(dict(normalized))
            index_row(normalized, len(combined) - 1, url_index, addr_index)
            return

        kept = combined[match_idx]
        group_id = kept.get("duplicate_group_id") or next_group()
        kept["duplicate_group_id"] = group_id
        reason_counter[reason or "unknown"] += 1
        duplicates.append(duplicate_audit_row(kept, normalized, reason or "unknown", group_id))
        merge_rows(kept, normalized, reason or "unknown", group_id)
        # Refresh indexes for any newly filled identity fields
        index_row(kept, match_idx, url_index, addr_index)

    for i, row in enumerate(clean_rows):
        ingest(from_clean_row(row, i), existing_targets, SOURCE_CLEAN)

    # Collapse extract: unique existing clean targets as they appear in combined from clean-only ingest
    # Re-build unique existing list from first-seen clean identities for the extract file.
    unique_existing: list[dict[str, str]] = []
    seen_existing: set[str] = set()
    for row in existing_targets:
        uk = url_match_key(row["chain"], row.get("provider_pair_url", ""))
        ak = address_match_key(row["chain"], row.get("pair_address", ""))
        key = uk or ak or row["combined_target_id"]
        if key in seen_existing:
            continue
        seen_existing.add(key)
        unique_existing.append(row)

    for i, row in enumerate(seed_rows):
        ingest(from_seed_row(row, i), user_seed_targets, SOURCE_SEED)

    # Ensure every combined row has pending semantic status and no system label leak
    for row in combined:
        row["semantic_status"] = SEMANTIC_PENDING
        row.pop("system_semantic_label", None)

    chains = sorted({chain_norm(r["chain"]) for r in combined if not is_blank(r.get("chain"))})
    merged_count = sum(1 for r in combined if r.get("target_source") == SOURCE_MERGED)

    return {
        "existing_targets": unique_existing,
        "existing_targets_loaded_raw": len(existing_targets),
        "user_seed_targets": user_seed_targets,
        "combined": combined,
        "duplicates": duplicates,
        "rejected": rejected,
        "duplicate_count_by_reason": dict(reason_counter),
        "merged_duplicate_count": len(duplicates),
        "merged_cross_source_count": merged_count,
        "chains_in_combined_universe": chains,
        "clean_targets_loaded": len(unique_existing),
        "user_seed_targets_loaded": len(user_seed_targets),
        "combined_unique_targets": len(combined),
        "rejected_or_incomplete_count": len(rejected),
    }


def build_manifest(
    *,
    timestamp: str,
    user_seed_path: Path,
    clean_path: Path,
    seed_exists: bool,
    clean_exists: bool,
    result: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    return {
        "phase": PHASE,
        "timestamp": timestamp,
        "output_root": str(output_root).replace("\\", "/"),
        "input_user_seed_path": str(user_seed_path).replace("\\", "/"),
        "input_user_seed_exists": seed_exists,
        "input_clean_candidates_path": str(clean_path).replace("\\", "/"),
        "input_clean_candidates_exists": clean_exists,
        "clean_targets_loaded": result["clean_targets_loaded"],
        "user_seed_targets_loaded": result["user_seed_targets_loaded"],
        "combined_unique_targets": result["combined_unique_targets"],
        "merged_duplicate_count": result["merged_duplicate_count"],
        "merged_cross_source_count": result["merged_cross_source_count"],
        "rejected_or_incomplete_count": result["rejected_or_incomplete_count"],
        "duplicate_count_by_reason": result["duplicate_count_by_reason"],
        "chains_in_combined_universe": result["chains_in_combined_universe"],
        "no_dexscreener_called": True,
        "no_server_started": True,
        "collector_modified": False,
        "trader_db_mutated": False,
        "seed_collection_is_provenance_only": True,
        "semantic_status_value": SEMANTIC_PENDING,
        "dedupe_logic": [
            "chain + provider_pair_url (strongest)",
            "chain + pair_address / user_supplied_pair_address",
            "chain + resolved_pair_address",
            "no symbol dedupe",
            "no token name dedupe",
            "Solana/XRPL address casing preserved exactly",
            "EVM addresses compared case-insensitively; original casing preserved in output",
        ],
    }


def build_summary(manifest: dict[str, Any], output_root: Path) -> str:
    lines = [
        "AE16B Combined Clean Universe Summary",
        "=====================================",
        f"phase: {manifest['phase']}",
        f"output root: {output_root}",
        f"inputs used:",
        f"  - user seed: {manifest['input_user_seed_path']} (exists={manifest['input_user_seed_exists']})",
        f"  - clean candidates: {manifest['input_clean_candidates_path']} (exists={manifest['input_clean_candidates_exists']})",
        f"clean targets loaded: {manifest['clean_targets_loaded']}",
        f"user seed targets loaded: {manifest['user_seed_targets_loaded']}",
        f"combined unique targets: {manifest['combined_unique_targets']}",
        f"merged duplicate count: {manifest['merged_duplicate_count']}",
        f"rejected/incomplete count: {manifest['rejected_or_incomplete_count']}",
        f"chains represented: {', '.join(manifest['chains_in_combined_universe'])}",
        "duplicate logic used:",
        "  1) chain + provider_pair_url (strongest)",
        "  2) chain + pair_address / user_supplied_pair_address",
        "  3) chain + resolved_pair_address",
        "  - no dedupe by symbol or token name",
        "  - Solana/XRPL casing preserved; EVM match case-insensitive",
        "confirmation: seed_collection is provenance only (not system_semantic_label)",
        f"confirmation: semantic_status remains {SEMANTIC_PENDING}",
        "confirmation: DexScreener was not called",
        "confirmation: server was not started",
        "confirmation: collector was not modified",
        "confirmation: trader.db was not mutated",
        "",
    ]
    return "\n".join(lines)


def run(
    user_seed_path: Path,
    clean_path: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    seed_exists, clean_exists, errors = validate_inputs(user_seed_path, clean_path)
    if errors:
        raise FileNotFoundError("\n".join(errors))

    timestamp = utc_stamp()
    if output_root is None:
        output_root = Path("data/audits") / f"ae16b_combined_clean_universe_{timestamp}"

    data_dir = output_root / "data"
    reports_dir = output_root / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    clean_rows = read_csv(clean_path)
    seed_rows = read_csv(user_seed_path)
    result = build_universe(clean_rows, seed_rows)

    write_csv(data_dir / "ae16b_existing_clean_targets.csv", result["existing_targets"], EXISTING_FIELDS)
    write_csv(data_dir / "ae16b_user_seed_targets.csv", result["user_seed_targets"], SEED_OUT_FIELDS)
    write_csv(data_dir / "ae16b_combined_clean_universe.csv", result["combined"], COMBINED_FIELDS)
    write_csv(data_dir / "ae16b_duplicate_targets.csv", result["duplicates"], DUPLICATE_FIELDS)
    write_csv(data_dir / "ae16b_rejected_or_incomplete_targets.csv", result["rejected"], REJECTED_FIELDS)

    manifest = build_manifest(
        timestamp=timestamp,
        user_seed_path=user_seed_path,
        clean_path=clean_path,
        seed_exists=seed_exists,
        clean_exists=clean_exists,
        result=result,
        output_root=output_root,
    )
    write_json(reports_dir / "ae16b_combined_universe_manifest.json", manifest)
    write_text(reports_dir / "ae16b_summary_for_upload.txt", build_summary(manifest, output_root))

    return {
        "output_root": output_root,
        "manifest": manifest,
        "result": result,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AE16B offline combined clean universe builder")
    parser.add_argument("--user-seed", type=Path, default=DEFAULT_USER_SEED)
    parser.add_argument("--clean-candidates", type=Path, default=DEFAULT_CLEAN_CANDIDATES)
    parser.add_argument("--output-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        out = run(args.user_seed, args.clean_candidates, args.output_root)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    manifest = out["manifest"]
    print(f"phase: {manifest['phase']}")
    print(f"output_root: {out['output_root']}")
    print(f"clean_targets_loaded: {manifest['clean_targets_loaded']}")
    print(f"user_seed_targets_loaded: {manifest['user_seed_targets_loaded']}")
    print(f"combined_unique_targets: {manifest['combined_unique_targets']}")
    print(f"merged_duplicate_count: {manifest['merged_duplicate_count']}")
    print(f"rejected_or_incomplete_count: {manifest['rejected_or_incomplete_count']}")
    print(f"chains: {', '.join(manifest['chains_in_combined_universe'])}")
    print("no_dexscreener_called=true no_server_started=true collector_modified=false trader_db_mutated=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
