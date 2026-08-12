"""Deterministic evidence priority buckets and semantic linkage helpers."""

from __future__ import annotations

import hashlib
import re
from typing import Any

EVIDENCE_PRIORITY_VERSION = "AE12_SENTIMENTFIX_EVIDENCE_PRIORITY_V1"

BUCKET_QUOTAS: dict[str, int] = {
    "identity_metadata": 4,
    "semantic_source": 4,
    "semantic_markers": 2,
    "runtime_context": 2,
}
MAX_SNIPPETS_TOTAL = sum(BUCKET_QUOTAS.values())

SEMANTIC_MARKERS: tuple[str, ...] = (
    "meme",
    "memecoin",
    "no utility",
    "moon",
    "pump",
    "100x",
    "roi",
    "hodl",
    "charity",
    "donation",
    "dao",
    "governance",
    "community",
    "utility",
    "payment",
    "rewards",
    "volunteering",
    "impact",
    "ecosystem",
    "access token",
    "discount",
    "cooperative",
    "local economy",
)

NEGATIVE_MARKERS: tuple[str, ...] = (
    "meme",
    "memecoin",
    "no utility",
    "moon",
    "pump",
    "100x",
    "roi",
    "hodl",
    "speculation",
    "hype",
)

IDENTITY_MARKERS: tuple[str, ...] = (
    "name:",
    "symbol:",
    "description:",
    "contract",
    "token",
    "legacy_cluster",
)

RUNTIME_CONTEXT_MARKERS: tuple[str, ...] = (
    "max_open_positions",
    "cooldown",
    "strict blocker",
    "active pair lock",
    "stale price",
    "trading_opportunity_state",
    "paper trade",
    "no_trade",
    "mention_only",
    "exploration_decision",
    "strict_shadow",
)

LINKAGE_METHOD_RANK: dict[str, int] = {
    "EXACT_TOKEN_ADDRESS_MATCH": 100,
    "EXACT_CONTRACT_ADDRESS_MATCH": 95,
    "EXACT_PAIR_ADDRESS_MATCH": 90,
    "EXACT_SYMBOL_CHAIN_MATCH": 80,
    "EXACT_SYMBOL_MATCH": 70,
    "TIME_WINDOW_MATCH": 60,
    "FUZZY_SYMBOL_OR_NAME_MATCH": 50,
    "GEMINI_WEB_GROUNDED": 45,
    "GEMINI_MODEL_KNOWLEDGE_ONLY": 40,
    "LLM_CONTEXT_INFERRED": 35,
    "LEGACY_CLUSTER_ONLY": 20,
    "UNLINKED_CONTEXT": 10,
    "NO_LINK": 0,
}


def _lower(text: str) -> str:
    return (text or "").lower()


def count_semantic_markers(text: str) -> list[str]:
    blob = _lower(text)
    return [m for m in SEMANTIC_MARKERS if m in blob]


def count_negative_markers(text: str) -> list[str]:
    blob = _lower(text)
    return [m for m in NEGATIVE_MARKERS if m in blob]


def count_identity_markers(text: str) -> list[str]:
    blob = _lower(text)
    return [m for m in IDENTITY_MARKERS if m in blob]


def count_runtime_context_markers(text: str) -> list[str]:
    blob = _lower(text)
    return [m for m in RUNTIME_CONTEXT_MARKERS if m in blob]


def classify_snippet_bucket(text: str, *, snippet_type: str = "") -> str:
    st = _lower(snippet_type)
    blob = _lower(text)
    if st.startswith("identity") or st.startswith("coins.") or any(
        blob.startswith(p) for p in ("name:", "symbol:", "description:", "legacy_cluster:")
    ):
        return "identity_metadata"
    if count_runtime_context_markers(blob):
        return "runtime_context"
    if count_semantic_markers(blob) or st in {"sentiment", "news", "narrative", "llm_context"}:
        if count_semantic_markers(blob):
            return "semantic_markers"
        return "semantic_source"
    if st in {"candidate_reason", "sentiment", "news", "narrative"}:
        return "semantic_source"
    if any(k in blob for k in RUNTIME_CONTEXT_MARKERS):
        return "runtime_context"
    return "semantic_source"


def linkage_confidence(method: str) -> float:
    rank = LINKAGE_METHOD_RANK.get(method, 0)
    return round(rank / 100.0, 4)


def infer_linkage_method(
    *,
    asset: dict[str, Any],
    source_table_or_file: str,
    matched_on: str,
) -> str:
    mo = _lower(matched_on)
    token = _lower(asset.get("token_address") or asset.get("contract_address") or "")
    pair = _lower(asset.get("pair_address") or "")
    sym = _lower(asset.get("symbol") or "")
    chain = _lower(asset.get("chain") or "")
    if token and token in mo:
        return "EXACT_TOKEN_ADDRESS_MATCH" if "token" in mo else "EXACT_CONTRACT_ADDRESS_MATCH"
    if pair and pair in mo:
        return "EXACT_PAIR_ADDRESS_MATCH"
    if sym and chain and sym in mo and chain in mo:
        return "EXACT_SYMBOL_CHAIN_MATCH"
    if sym and sym in mo:
        return "EXACT_SYMBOL_MATCH"
    if "legacy_cluster" in mo or "cluster_label" in source_table_or_file:
        return "LEGACY_CLUSTER_ONLY"
    if sym and (sym in _lower(asset.get("name") or "") or sym[:3] in mo):
        return "FUZZY_SYMBOL_OR_NAME_MATCH"
    if mo:
        return "UNLINKED_CONTEXT"
    return "NO_LINK"


def snippet_sort_key(snippet: dict[str, Any]) -> tuple:
    method = str(snippet.get("linkage_method") or "NO_LINK")
    rank = LINKAGE_METHOD_RANK.get(method, 0)
    ts = str(snippet.get("source_timestamp") or "")
    sem = len(snippet.get("semantic_markers_found") or [])
    neg = len(snippet.get("negative_markers_found") or [])
    src = str(snippet.get("source_table_or_file") or "")
    row_id = str(snippet.get("source_row_id") or "")
    text_hash = hashlib.sha256(str(snippet.get("text") or "").encode("utf-8")).hexdigest()
    return (-rank, -sem, -neg, ts, src, row_id, text_hash)


def select_priority_snippets(snippets: list[dict[str, Any]], *, max_total: int = MAX_SNIPPETS_TOTAL) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select snippets by bucket quotas with deterministic ordering."""
    by_bucket: dict[str, list[dict[str, Any]]] = {k: [] for k in BUCKET_QUOTAS}
    for sn in snippets:
        bucket = str(sn.get("bucket") or classify_snippet_bucket(sn.get("text", ""), snippet_type=str(sn.get("snippet_type") or "")))
        if bucket not in by_bucket:
            bucket = "semantic_source"
        by_bucket[bucket].append(sn)

    selected: list[dict[str, Any]] = []
    bucket_counts_before = {k: len(v) for k, v in by_bucket.items()}
    truncated = 0
    for bucket, quota in BUCKET_QUOTAS.items():
        pool = sorted(by_bucket.get(bucket, []), key=snippet_sort_key)
        take = pool[:quota]
        truncated += max(0, len(pool) - len(take))
        for sn in take:
            sn2 = dict(sn)
            sn2["used_in_classifier_evidence"] = True
            selected.append(sn2)

    # Deterministic final order: bucket priority then sort key
    bucket_order = list(BUCKET_QUOTAS.keys())
    selected.sort(key=lambda s: (bucket_order.index(s.get("bucket", "semantic_source")),) + snippet_sort_key(s))
    if len(selected) > max_total:
        truncated += len(selected) - max_total
        selected = selected[:max_total]

    for sn in snippets:
        if not any(sn.get("text") == x.get("text") and sn.get("source_row_id") == x.get("source_row_id") for x in selected):
            sn = dict(sn)
            sn["used_in_classifier_evidence"] = False
            if classify_snippet_bucket(sn.get("text", "")) == "runtime_context":
                sn["reason_if_not_used"] = "runtime_crowding_quota"
            elif linkage_confidence(str(sn.get("linkage_method") or "")) < 0.5:
                sn["reason_if_not_used"] = "low_linkage_confidence"
            else:
                sn["reason_if_not_used"] = "quota_truncation"

    audit = {
        "evidence_priority_version": EVIDENCE_PRIORITY_VERSION,
        "bucket_quotas": dict(BUCKET_QUOTAS),
        "bucket_total_before_truncation": sum(bucket_counts_before.values()),
        "bucket_total_after_truncation": len(selected),
        "truncated_snippet_count": truncated,
        "truncation_strategy": "deterministic_bucket_quota",
        "evidence_bucket_counts": {k: len([s for s in selected if s.get("bucket") == k]) for k in BUCKET_QUOTAS},
        "linkage_methods_present": sorted({str(s.get("linkage_method") or "NO_LINK") for s in selected}),
        "best_linkage_method": max(
            (str(s.get("linkage_method") or "NO_LINK") for s in selected),
            key=lambda m: LINKAGE_METHOD_RANK.get(m, 0),
            default="NO_LINK",
        ),
        "weakest_linkage_method_used": min(
            (str(s.get("linkage_method") or "NO_LINK") for s in selected),
            key=lambda m: LINKAGE_METHOD_RANK.get(m, 0),
            default="NO_LINK",
        ),
    }
    return selected, audit


def build_linkage_summary(linkage_rows: list[dict[str, Any]]) -> dict[str, Any]:
    used = [r for r in linkage_rows if r.get("used_in_classifier_evidence")]
    not_used = [r for r in linkage_rows if not r.get("used_in_classifier_evidence")]
    dist: dict[str, int] = {}
    for r in linkage_rows:
        m = str(r.get("linkage_method") or "NO_LINK")
        dist[m] = dist.get(m, 0) + 1
    total = len(linkage_rows) or 1
    exact = sum(
        dist.get(k, 0)
        for k in (
            "EXACT_TOKEN_ADDRESS_MATCH",
            "EXACT_CONTRACT_ADDRESS_MATCH",
            "EXACT_PAIR_ADDRESS_MATCH",
            "EXACT_SYMBOL_CHAIN_MATCH",
            "EXACT_SYMBOL_MATCH",
        )
    )
    fuzzy = dist.get("FUZZY_SYMBOL_OR_NAME_MATCH", 0) + dist.get("LLM_CONTEXT_INFERRED", 0)
    web = dist.get("GEMINI_WEB_GROUNDED", 0)
    model_only = dist.get("GEMINI_MODEL_KNOWLEDGE_ONLY", 0)
    no_link = dist.get("NO_LINK", 0) + dist.get("UNLINKED_CONTEXT", 0)
    return {
        "total_linked_snippets": len(used),
        "total_unlinked_snippets": len(not_used),
        "linkage_method_distribution": dist,
        "exact_match_share": round(exact / total, 6),
        "fuzzy_or_inferred_share": round(fuzzy / total, 6),
        "web_grounded_share": round(web / total, 6),
        "model_knowledge_only_share": round(model_only / total, 6),
        "no_link_share": round(no_link / total, 6),
        "used_in_classifier_evidence_count": len(used),
        "not_used_due_to_quota_count": sum(1 for r in not_used if r.get("reason_if_not_used") == "quota_truncation"),
        "not_used_due_to_runtime_crowding_count": sum(
            1 for r in not_used if r.get("reason_if_not_used") == "runtime_crowding_quota"
        ),
        "not_used_due_to_low_linkage_confidence_count": sum(
            1 for r in not_used if r.get("reason_if_not_used") == "low_linkage_confidence"
        ),
    }


def marker_audit_for_text(text: str) -> dict[str, Any]:
    return {
        "semantic_marker_count": len(count_semantic_markers(text)),
        "negative_marker_count": len(count_negative_markers(text)),
        "identity_marker_count": len(count_identity_markers(text)),
        "runtime_context_marker_count": len(count_runtime_context_markers(text)),
        "semantic_markers_found": count_semantic_markers(text),
        "negative_markers_found": count_negative_markers(text),
    }


def is_runtime_only_snippet(text: str) -> bool:
    blob = _lower(text)
    if not blob:
        return False
    runtime_hits = count_runtime_context_markers(blob)
    semantic_hits = count_semantic_markers(blob)
    return bool(runtime_hits) and not semantic_hits and not count_identity_markers(blob)
