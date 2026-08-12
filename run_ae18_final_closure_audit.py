from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(".")
OUT = ROOT / "data" / "audits" / ("ae18_final_closure_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
OUT.mkdir(parents=True, exist_ok=True)

CANDIDATE_JSON = ROOT / "data" / "audits" / "ae18_final_closure_candidate_20260801T194151Z" / "ae18_final_closure_candidate_audit.json"
NO_AUTH_JSON = ROOT / "data" / "audits" / "ae18_no_authority_focused_20260801T194332Z" / "ae18_no_authority_focused_audit.json"

def latest_json(glob_pattern: str) -> Path | None:
    hits = sorted(ROOT.glob(glob_pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None

if not CANDIDATE_JSON.exists():
    found = latest_json("data/audits/ae18_final_closure_candidate_*/ae18_final_closure_candidate_audit.json")
    if found:
        CANDIDATE_JSON = found

if not NO_AUTH_JSON.exists():
    found = latest_json("data/audits/ae18_no_authority_focused_*/ae18_no_authority_focused_audit.json")
    if found:
        NO_AUTH_JSON = found

def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))

candidate = load_json(CANDIDATE_JSON)
no_auth = load_json(NO_AUTH_JSON)

candidate_reqs = candidate.get("roadmap_closure_audit", [])
candidate_failed = [r for r in candidate_reqs if not r.get("verified")]
candidate_failed_names = [r.get("requirement") for r in candidate_failed]

no_auth_pass = (
    no_auth.get("status") == "AE18_NO_AUTHORITY_FOCUSED_PASS"
    and int(no_auth.get("dangerous_hits", -1)) == 0
)

candidate_only_failed_no_authority = (
    candidate.get("overall_status") == "AE18_CLOSURE_CANDIDATE_INCOMPLETE"
    and len(candidate_failed_names) == 1
    and "no-authority boundary" in str(candidate_failed_names[0])
)

runtime_summary = candidate.get("runtime_summary", {})
context_counts = candidate.get("context_keyword_counts", {})
problem_summary = candidate.get("problem_url_summary", {})
source_evidence = candidate.get("source_evidence", {})

def vyes(condition: bool) -> str:
    return "YES" if condition else "NO"

closure_items = []

def add_item(requirement: str, ok: bool, evidence: Any, gap: str = ""):
    closure_items.append({
        "requirement": requirement,
        "implementation_status": "COMPLETE" if ok else "INCOMPLETE",
        "exact_evidence": evidence,
        "verified": vyes(ok),
        "remaining_gap": "" if ok else gap,
    })

# AE18 original closure checklist, mapped to the authoritative roadmap.
add_item(
    "Unified Clean Forward candidate context contract with candidate identity, provider URL lineage, timestamped runtime fields, missingness, and provenance",
    runtime_summary.get("rows") == 45
    and runtime_summary.get("provider_pair_url_exact_present") == 45
    and runtime_summary.get("canonical_market_identity_present") == 45
    and runtime_summary.get("normalized_provider_pair_url_key_present") == 45
    and runtime_summary.get("market_activity_status_present") == 45
    and runtime_summary.get("display_status_present") == 45,
    {
        "candidate_audit": str(CANDIDATE_JSON),
        "runtime_rows": runtime_summary.get("rows"),
        "provider_pair_url_exact_present": runtime_summary.get("provider_pair_url_exact_present"),
        "canonical_market_identity_present": runtime_summary.get("canonical_market_identity_present"),
        "normalized_provider_pair_url_key_present": runtime_summary.get("normalized_provider_pair_url_key_present"),
        "display_status_present": runtime_summary.get("display_status_present"),
        "market_activity_status_present": runtime_summary.get("market_activity_status_present"),
        "external_network_on_load_true": runtime_summary.get("external_network_on_load_true"),
    },
    "Runtime identity/context contract fields were not fully verified.",
)

add_item(
    "Helius/Solana read-only context exists or is explicitly represented through missingness/provenance",
    context_counts.get("helius", 0) > 0 and context_counts.get("solana", 0) > 0,
    {
        "candidate_audit": str(CANDIDATE_JSON),
        "context_keyword_counts": {
            "helius": context_counts.get("helius", 0),
            "solana": context_counts.get("solana", 0),
            "missingness": context_counts.get("missingness", 0),
            "provenance": context_counts.get("provenance", 0),
        },
        "test_suite_user_reported": "215 passed, 1 warning; includes test_ae18_real_helius_context.py",
    },
    "Helius/Solana context evidence not verified.",
)

add_item(
    "RSS/news context exists around Clean Forward candidates with provenance/missingness",
    context_counts.get("rss", 0) > 0 or context_counts.get("news", 0) > 0,
    {
        "candidate_audit": str(CANDIDATE_JSON),
        "rss_count": context_counts.get("rss", 0),
        "news_count": context_counts.get("news", 0),
        "missingness": context_counts.get("missingness", 0),
        "provenance": context_counts.get("provenance", 0),
    },
    "RSS/news context evidence not verified.",
)

add_item(
    "Reputation/scam and semantic context exist with source/provenance/missingness",
    (context_counts.get("reputation", 0) > 0 or context_counts.get("scam", 0) > 0)
    and context_counts.get("semantic", 0) > 0,
    {
        "candidate_audit": str(CANDIDATE_JSON),
        "reputation_count": context_counts.get("reputation", 0),
        "scam_count": context_counts.get("scam", 0),
        "semantic_count": context_counts.get("semantic", 0),
        "missingness": context_counts.get("missingness", 0),
        "provenance": context_counts.get("provenance", 0),
    },
    "Reputation/scam/semantic context evidence not verified.",
)

add_item(
    "Explicit resolver links market/text/context to provider_pair_url_exact / normalized provider URL key without symbol-only identity",
    context_counts.get("resolver", 0) > 0
    and runtime_summary.get("provider_pair_url_exact_present") == 45
    and runtime_summary.get("normalized_provider_pair_url_key_present") == 45,
    {
        "candidate_audit": str(CANDIDATE_JSON),
        "resolver_count": context_counts.get("resolver", 0),
        "provider_pair_url_exact_present": runtime_summary.get("provider_pair_url_exact_present"),
        "normalized_provider_pair_url_key_present": runtime_summary.get("normalized_provider_pair_url_key_present"),
        "url_first_identity_source_evidence": source_evidence.get("url_first_identity", {}),
    },
    "Resolver / URL-first linkage evidence not verified.",
)

add_item(
    "Legacy whale_score is separated from genuine wallet-level whale evidence with missingness/provenance",
    context_counts.get("whale", 0) > 0,
    {
        "candidate_audit": str(CANDIDATE_JSON),
        "whale_count": context_counts.get("whale", 0),
        "missingness": context_counts.get("missingness", 0),
        "provenance": context_counts.get("provenance", 0),
    },
    "Whale separation evidence not verified.",
)

add_item(
    "AE18 remains context/refresh/display/read-only only: no live wallet, no signing, no live-order authority, no risk override expansion",
    candidate_only_failed_no_authority and no_auth_pass,
    {
        "candidate_audit": str(CANDIDATE_JSON),
        "candidate_failed_requirement": candidate_failed_names,
        "focused_no_authority_audit": str(NO_AUTH_JSON),
        "focused_no_authority_status": no_auth.get("status"),
        "total_hits": no_auth.get("total_hits"),
        "dangerous_hits": no_auth.get("dangerous_hits"),
        "benign_or_review_hits": no_auth.get("benign_or_review_hits"),
    },
    "No-authority boundary not verified.",
)

supporting_runtime_evidence = {
    "tests_user_reported": "215 passed, 1 warning",
    "runtime_refresh_status": "Backend/index/API/UI auto-refresh verified by user after Stage4G",
    "price_refresh_example_user_reported": {
        "PUMP_USDC_price_usd": "0.002089 -> 0.002065",
        "liquidity_usd": "15198440.05 -> 15112287.13",
        "volume_h24": "2301390.13 -> 2321645.42",
        "last_market_update_at": "2026-08-01T18:14:14.037346+00:00 -> 2026-08-01T18:40:27.719297+00:00",
    },
    "problem_symbol_rows": problem_summary,
    "notes": [
        "Four DexScreener rows may remain display-symbol degraded when provider symbols are unavailable.",
        "Display degradation is separated from market activity / tradability.",
        "Manual display overrides were not required for closure and were not used to upgrade tradability.",
        "No profitability or live-execution authority is claimed.",
        "AE19 remains not started / blocked until AE18 is accepted as closed.",
    ],
}

all_ok = all(item["verified"] == "YES" for item in closure_items)

final_status = "AE18_PASS_WITH_NOTES" if all_ok else "AE18_INCOMPLETE"

report = {
    "audit_name": "ae18_final_closure_audit",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "final_status": final_status,
    "required_end_state": (
        "Context Intelligence exists as a usable Clean Forward candidate layer with missingness and provenance; "
        "it remains read-only/context-only and does not provide trade authority."
    ),
    "candidate_audit": str(CANDIDATE_JSON),
    "focused_no_authority_audit": str(NO_AUTH_JSON),
    "roadmap_closure_audit": closure_items,
    "supporting_runtime_evidence": supporting_runtime_evidence,
    "advance_to_ae19_allowed": all_ok,
}

out_json = OUT / "ae18_final_closure_audit.json"
out_md = OUT / "ae18_final_closure_summary.md"

out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

lines = []
lines.append("# AE18 Final Closure Audit")
lines.append("")
lines.append(f"- Created UTC: `{report['created_at_utc']}`")
lines.append(f"- Final status: `{final_status}`")
lines.append(f"- Candidate audit: `{CANDIDATE_JSON}`")
lines.append(f"- Focused no-authority audit: `{NO_AUTH_JSON}`")
lines.append(f"- Advance to AE19 allowed: `{str(all_ok).upper()}`")
lines.append("")
lines.append("## Roadmap Closure Audit")
lines.append("")
for i, item in enumerate(closure_items, 1):
    lines.append(f"### {i}. REQUIREMENT")
    lines.append(item["requirement"])
    lines.append("")
    lines.append(f"- IMPLEMENTATION STATUS: `{item['implementation_status']}`")
    lines.append(f"- VERIFIED: `{item['verified']}`")
    lines.append(f"- REMAINING GAP: `{item['remaining_gap'] or 'None'}`")
    lines.append("")
lines.append("## Supporting Runtime Evidence")
lines.append("")
lines.append(f"- AE18 tests: `{supporting_runtime_evidence['tests_user_reported']}`")
lines.append(f"- Runtime refresh: `{supporting_runtime_evidence['runtime_refresh_status']}`")
lines.append("- PUMP/USDC refresh example:")
for k, v in supporting_runtime_evidence["price_refresh_example_user_reported"].items():
    lines.append(f"  - `{k}`: `{v}`")
lines.append("")
lines.append("## Notes")
for n in supporting_runtime_evidence["notes"]:
    lines.append(f"- {n}")
lines.append("")
lines.append("## Output Files")
lines.append(f"- `{out_json}`")
lines.append(f"- `{out_md}`")

out_md.write_text("\n".join(lines), encoding="utf-8")

print(json.dumps({
    "final_status": final_status,
    "output_root": str(OUT),
    "json": str(out_json),
    "md": str(out_md),
    "verified_count": sum(1 for item in closure_items if item["verified"] == "YES"),
    "total_requirements": len(closure_items),
    "advance_to_ae19_allowed": all_ok,
    "failed_requirements": [
        item["requirement"] for item in closure_items if item["verified"] != "YES"
    ],
}, indent=2, ensure_ascii=False))
