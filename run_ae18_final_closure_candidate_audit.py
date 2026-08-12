from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(".")
OUT = ROOT / "data" / "audits" / ("ae18_final_closure_candidate_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
OUT.mkdir(parents=True, exist_ok=True)

RUNTIME_INDEX = ROOT / "data" / "runtime" / "canonical_market_identity_index.jsonl"

PROBLEM_URLS = {
    "https://dexscreener.com/robinhood/0x9c2905076ad86335e0CB8227fd5D0e5Bec795f1A",
    "https://dexscreener.com/robinhood/0xb3F901859ACbEF2288E187993AA50911A5404762",
    "https://dexscreener.com/robinhood/0xEA63b938967e65B2D71d99Bc8cFD9c4cB3c7c105",
    "https://dexscreener.com/base/0x2db51152Dd4F7a00c10e181401e18B9d6269e4b4",
}

def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            rows.append({"__parse_error__": line[:200]})
    return rows

def present(row: dict[str, Any], key: str) -> bool:
    return row.get(key) not in (None, "", [], {})

def count_present(rows: list[dict[str, Any]], key: str) -> int:
    return sum(1 for r in rows if present(r, key))

def find_files(patterns: list[str]) -> list[str]:
    hits = []
    for pat in patterns:
        for p in ROOT.rglob(pat):
            s = str(p)
            if any(skip in s for skip in [
                ".git", "__pycache__", ".pytest_cache", "node_modules", ".venv", "venv"
            ]):
                continue
            hits.append(s)
    return sorted(set(hits))

rows = load_jsonl(RUNTIME_INDEX)

runtime_summary = {
    "runtime_index_path": str(RUNTIME_INDEX),
    "runtime_index_exists": RUNTIME_INDEX.exists(),
    "rows": len(rows),
    "provider_pair_url_exact_present": count_present(rows, "provider_pair_url_exact"),
    "canonical_market_identity_present": count_present(rows, "canonical_market_identity"),
    "normalized_provider_pair_url_key_present": count_present(rows, "normalized_provider_pair_url_key"),
    "runtime_index_sourced_true": sum(1 for r in rows if r.get("runtime_index_sourced") is True),
    "external_network_on_load_true": sum(1 for r in rows if r.get("external_network_on_load") is True),
    "price_usd_present": count_present(rows, "price_usd"),
    "liquidity_usd_present": count_present(rows, "liquidity_usd"),
    "provider_fetch_at_present": count_present(rows, "provider_fetch_at"),
    "market_data_refreshed_at_present": count_present(rows, "market_data_refreshed_at"),
    "last_market_update_at_present": count_present(rows, "last_market_update_at"),
    "price_updated_at_present": count_present(rows, "price_updated_at"),
    "display_status_present": count_present(rows, "display_status"),
    "market_activity_status_present": count_present(rows, "market_activity_status"),
    "activity_trade_readiness_status_present": count_present(rows, "activity_trade_readiness_status"),
    "trade_readiness_status_present": count_present(rows, "trade_readiness_status"),
    "display_status_counts": dict(Counter(str(r.get("display_status") or "") for r in rows)),
    "market_activity_status_counts": dict(Counter(str(r.get("market_activity_status") or "") for r in rows)),
    "trade_readiness_status_counts": dict(Counter(str(r.get("trade_readiness_status") or "") for r in rows)),
}

problem_rows = [r for r in rows if r.get("provider_pair_url_exact") in PROBLEM_URLS]
problem_summary = []
for r in problem_rows:
    problem_summary.append({
        "provider_pair_url_exact": r.get("provider_pair_url_exact"),
        "symbol_pair_display": r.get("symbol_pair_display"),
        "display_status": r.get("display_status"),
        "display_metadata_status": r.get("display_metadata_status"),
        "symbol_resolution_status": r.get("symbol_resolution_status"),
        "market_activity_status": r.get("market_activity_status"),
        "activity_trade_readiness_status": r.get("activity_trade_readiness_status"),
        "trade_readiness_status": r.get("trade_readiness_status"),
        "market_activity_blocks_demo_entry": r.get("market_activity_blocks_demo_entry"),
        "activity_trade_block_reason": r.get("activity_trade_block_reason"),
        "price_usd": r.get("price_usd"),
        "liquidity_usd": r.get("liquidity_usd"),
        "volume_h24": r.get("volume_h24"),
        "provider_txns_recent_total": r.get("provider_txns_recent_total"),
        "provider_volume_recent_total": r.get("provider_volume_recent_total"),
        "provider_price_delta_any_nonzero": r.get("provider_price_delta_any_nonzero"),
    })

# Static/source-code evidence map.
files = {
    "main.py": ROOT / "main.py",
    "app/live.py": ROOT / "app" / "live.py",
    "app/api.py": ROOT / "app" / "api.py",
    "runtime_market_feed.py": ROOT / "app" / "ae13b_product" / "runtime_market_feed.py",
    "canonical_market_identity.py": ROOT / "app" / "clean_forward" / "canonical_market_identity.py",
    "market_activity.py": ROOT / "app" / "clean_forward" / "market_activity.py",
    "runtime_selected_collection.py": ROOT / "app" / "clean_forward" / "runtime_selected_collection.py",
    "product_demo.js": ROOT / "static" / "product_demo.js",
}

source_text = {name: read_text(path) for name, path in files.items()}

source_evidence = {
    "url_first_identity": {
        "provider_pair_url_exact_mentions": sum(t.count("provider_pair_url_exact") for t in source_text.values()),
        "normalized_provider_pair_url_key_mentions": sum(t.count("normalized_provider_pair_url_key") for t in source_text.values()),
        "canonical_market_identity_mentions": sum(t.count("canonical_market_identity") for t in source_text.values()),
    },
    "activity_axis": {
        "market_activity_file_exists": files["market_activity.py"].exists(),
        "ACTIVE_PROVIDER_TXNS_mentions": sum(t.count("ACTIVE_PROVIDER_TXNS") for t in source_text.values()),
        "ACTIVITY_STAGNANT_mentions": sum(t.count("ACTIVITY_STAGNANT") for t in source_text.values()),
        "market_activity_blocks_demo_entry_mentions": sum(t.count("market_activity_blocks_demo_entry") for t in source_text.values()),
        "DEMO_ACTION_BLOCKED_MARKET_ACTIVITY_mentions": source_text["app/api.py"].count("DEMO_ACTION_BLOCKED_MARKET_ACTIVITY"),
    },
    "ui_refresh": {
        "ae18_manual_refresh_function": "ae18ManualRuntimeRefreshAndReload" in source_text["product_demo.js"],
        "auto_refresh_status_function": "ae18AutoRefreshStatus" in source_text["product_demo.js"],
        "auto_refresh_default_60": '"60"' in source_text["product_demo.js"],
        "post_clean_forward_refresh_mentions": source_text["product_demo.js"].count("/api/clean-forward-feed/refresh"),
        "cache_busted_script": "ae18-autorefresh-60s-stage4g" in read_text(ROOT / "static" / "index.html"),
    },
    "shutdown_boundary": {
        "background_scanner_env": "MEMECOIN_BACKGROUND_SCANNER_ENABLED" in source_text["main.py"],
        "shutdown_requested_checks_live": source_text["app/live.py"].count("_shutdown_requested"),
        "runtime_shutdown_requested_checks_collection": source_text["runtime_selected_collection.py"].count("_runtime_shutdown_requested"),
    },
    "no_wallet_or_live_trade_expansion_static_scan": {
        "private_key_mentions": sum(t.lower().count("private_key") for t in source_text.values()),
        "wallet_sign_mentions": sum(t.lower().count("sign") for t in source_text.values()),
        "live_buy_mentions": sum(t.count("LIVE_BUY") for t in source_text.values()),
    },
}

# Context layer evidence by file/artifact discovery. This does not invent PASS;
# it reports whether concrete artifacts/code paths are discoverable.
context_file_hits = {
    "helius_solana": find_files(["*helius*", "*solana*"]),
    "rss_news": find_files(["*rss*", "*news*"]),
    "reputation_scam": find_files(["*reputation*", "*scam*"]),
    "semantic_context": find_files(["*semantic*", "*context*"]),
    "resolver": find_files(["*resolver*", "*resolution*"]),
    "whale": find_files(["*whale*"]),
}

audit_file_hits = find_files(["*.json", "*.md"])
audit_text_joined = "\n".join(
    read_text(Path(p))[:20000]
    for p in audit_file_hits
    if "data\\audits" in p or "data/audits" in p or "audit" in p.lower() or "validation" in p.lower()
)

context_keyword_counts = {
    "helius": len(re.findall(r"helius", audit_text_joined, flags=re.I)),
    "solana": len(re.findall(r"solana", audit_text_joined, flags=re.I)),
    "rss": len(re.findall(r"\brss\b", audit_text_joined, flags=re.I)),
    "news": len(re.findall(r"\bnews\b", audit_text_joined, flags=re.I)),
    "reputation": len(re.findall(r"reputation", audit_text_joined, flags=re.I)),
    "scam": len(re.findall(r"scam", audit_text_joined, flags=re.I)),
    "semantic": len(re.findall(r"semantic", audit_text_joined, flags=re.I)),
    "resolver": len(re.findall(r"resolver|resolution", audit_text_joined, flags=re.I)),
    "whale": len(re.findall(r"whale", audit_text_joined, flags=re.I)),
    "missingness": len(re.findall(r"missingness|missing", audit_text_joined, flags=re.I)),
    "provenance": len(re.findall(r"provenance", audit_text_joined, flags=re.I)),
}

requirements = []

def req(name: str, ok: bool, evidence: Any, gap: str):
    requirements.append({
        "requirement": name,
        "implementation_status": "VERIFIED_BY_THIS_AUDIT" if ok else "NOT_VERIFIED_BY_THIS_AUDIT",
        "exact_evidence": evidence,
        "verified": bool(ok),
        "remaining_gap": "" if ok else gap,
    })

req(
    "Clean Forward candidate identity uses provider_pair_url_exact / canonical runtime identity, not symbol-only joins",
    len(rows) == 45
    and runtime_summary["provider_pair_url_exact_present"] == len(rows)
    and runtime_summary["canonical_market_identity_present"] == len(rows)
    and runtime_summary["normalized_provider_pair_url_key_present"] == len(rows),
    runtime_summary,
    "Runtime index identity fields are missing or row count differs from expected 45.",
)

req(
    "Runtime market data refresh writes price/liquidity/timestamp fields into the canonical runtime index",
    len(rows) == 45
    and runtime_summary["price_usd_present"] == len(rows)
    and runtime_summary["liquidity_usd_present"] == len(rows)
    and runtime_summary["provider_fetch_at_present"] == len(rows)
    and runtime_summary["market_data_refreshed_at_present"] == len(rows)
    and runtime_summary["price_updated_at_present"] == len(rows),
    runtime_summary,
    "Some rows lack price/liquidity/provider timestamp aliases after refresh.",
)

req(
    "UI auto-refresh is controlled and calls POST provider refresh, not passive stale GET only",
    source_evidence["ui_refresh"]["ae18_manual_refresh_function"]
    and source_evidence["ui_refresh"]["auto_refresh_status_function"]
    and source_evidence["ui_refresh"]["auto_refresh_default_60"]
    and source_evidence["ui_refresh"]["post_clean_forward_refresh_mentions"] > 0
    and source_evidence["ui_refresh"]["cache_busted_script"],
    source_evidence["ui_refresh"],
    "UI refresh wiring or 60s cache-busted script evidence missing.",
)

req(
    "Display status is separate from market activity/tradability",
    runtime_summary["display_status_present"] == len(rows)
    and runtime_summary["market_activity_status_present"] == len(rows)
    and source_evidence["activity_axis"]["market_activity_file_exists"]
    and source_evidence["activity_axis"]["DEMO_ACTION_BLOCKED_MARKET_ACTIVITY_mentions"] > 0,
    {
        "runtime_summary": runtime_summary,
        "source_evidence": source_evidence["activity_axis"],
    },
    "Display/activity axes are not fully present or demo guard evidence missing.",
)

req(
    "The four unresolved-symbol markets are not upgraded to actionable trade candidates solely by symbol fallback",
    len(problem_rows) == 4
    and all(str(r.get("market_activity_status")) != "ACTIVE_PROVIDER_TXNS" for r in problem_rows)
    and all(str(r.get("activity_trade_readiness_status")).startswith("WATCH_ONLY") for r in problem_rows)
    and all(r.get("market_activity_blocks_demo_entry") is True for r in problem_rows),
    problem_summary,
    "At least one unresolved-symbol/problem URL is still active/actionable or missing activity block.",
)

req(
    "No GET/load external-network mutation and no shutdown/background leak controls are present",
    runtime_summary["external_network_on_load_true"] == 0
    and source_evidence["shutdown_boundary"]["background_scanner_env"]
    and source_evidence["shutdown_boundary"]["shutdown_requested_checks_live"] > 0
    and source_evidence["shutdown_boundary"]["runtime_shutdown_requested_checks_collection"] > 0,
    {
        "runtime_summary": runtime_summary,
        "source_evidence": source_evidence["shutdown_boundary"],
    },
    "External-network-on-load or shutdown control evidence missing.",
)

req(
    "AE18 context layer evidence: Helius/Solana, RSS/news, reputation/scam, semantic context, resolver, whale separation, missingness/provenance",
    context_keyword_counts["helius"] > 0
    and context_keyword_counts["solana"] > 0
    and (context_keyword_counts["rss"] > 0 or context_keyword_counts["news"] > 0)
    and (context_keyword_counts["reputation"] > 0 or context_keyword_counts["scam"] > 0)
    and context_keyword_counts["semantic"] > 0
    and context_keyword_counts["resolver"] > 0
    and context_keyword_counts["whale"] > 0
    and context_keyword_counts["missingness"] > 0
    and context_keyword_counts["provenance"] > 0,
    {
        "context_file_hits": context_file_hits,
        "context_keyword_counts_in_audits_and_validation": context_keyword_counts,
    },
    "This audit did not find sufficient existing artifact evidence for every original AE18 context layer. Need inspect/add final context audit evidence, not close yet.",
)

req(
    "AE18 no-authority boundary: context/refresh only, no live wallet/signing/risk override expansion",
    source_evidence["no_wallet_or_live_trade_expansion_static_scan"]["private_key_mentions"] == 0
    and source_evidence["no_wallet_or_live_trade_expansion_static_scan"]["live_buy_mentions"] == 0,
    source_evidence["no_wallet_or_live_trade_expansion_static_scan"],
    "Static scan found possible wallet/private-key/live-buy terms in touched AE18 surfaces; inspect before closure.",
)

overall_verified = all(r["verified"] for r in requirements)

report = {
    "audit_name": "ae18_final_closure_candidate",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "overall_status": "AE18_CLOSURE_CANDIDATE_VERIFIED" if overall_verified else "AE18_CLOSURE_CANDIDATE_INCOMPLETE",
    "note": "This is a closure-candidate audit. It does not change trading state and does not mutate trader.db.",
    "tests_user_reported": "215 passed, 1 warning",
    "runtime_summary": runtime_summary,
    "problem_url_summary": problem_summary,
    "source_evidence": source_evidence,
    "context_file_hits": context_file_hits,
    "context_keyword_counts": context_keyword_counts,
    "roadmap_closure_audit": requirements,
}

out_json = OUT / "ae18_final_closure_candidate_audit.json"
out_md = OUT / "ae18_final_closure_candidate_summary.md"

out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

lines = []
lines.append("# AE18 Final Closure Candidate Audit")
lines.append("")
lines.append(f"- Created UTC: {report['created_at_utc']}")
lines.append(f"- Overall status: `{report['overall_status']}`")
lines.append(f"- Tests user reported: `{report['tests_user_reported']}`")
lines.append(f"- Runtime rows: `{runtime_summary['rows']}`")
lines.append("")
lines.append("## Roadmap Closure Audit")
lines.append("")
for i, r in enumerate(requirements, 1):
    lines.append(f"### {i}. {r['requirement']}")
    lines.append(f"- IMPLEMENTATION STATUS: {r['implementation_status']}")
    lines.append(f"- VERIFIED: {'YES' if r['verified'] else 'NO'}")
    lines.append(f"- REMAINING GAP: {r['remaining_gap'] or 'None'}")
    lines.append("")
lines.append("## Output Files")
lines.append(f"- `{out_json}`")
lines.append(f"- `{out_md}`")
out_md.write_text("\n".join(lines), encoding="utf-8")

print(json.dumps({
    "overall_status": report["overall_status"],
    "output_root": str(OUT),
    "json": str(out_json),
    "md": str(out_md),
    "verified_count": sum(1 for r in requirements if r["verified"]),
    "total_requirements": len(requirements),
    "failed_requirements": [
        r["requirement"] for r in requirements if not r["verified"]
    ],
}, indent=2, ensure_ascii=False))
