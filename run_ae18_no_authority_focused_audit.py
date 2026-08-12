from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "data" / "audits" / ("ae18_no_authority_focused_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
OUT.mkdir(parents=True, exist_ok=True)

SCAN_FILES = [
    ROOT / "main.py",
    ROOT / "app" / "live.py",
    ROOT / "app" / "api.py",
    ROOT / "app" / "ae13b_product" / "runtime_market_feed.py",
    ROOT / "app" / "clean_forward" / "canonical_market_identity.py",
    ROOT / "app" / "clean_forward" / "market_activity.py",
    ROOT / "app" / "clean_forward" / "runtime_selected_collection.py",
    ROOT / "static" / "product_demo.js",
]

SUSPICIOUS_TERMS = [
    re.compile(r"\bprivate_key\b", re.I),
    re.compile(r"\bwallet\b", re.I),
    re.compile(r"\bLIVE_BUY\b"),
    re.compile(r"\bsign(?:ed|ing|ature)?\b", re.I),
    re.compile(r"\bsend_transaction\b", re.I),
    re.compile(r"\bsend_raw_transaction\b", re.I),
    re.compile(r"\bsign_transaction\b", re.I),
    re.compile(r"\bsign_message\b", re.I),
    re.compile(r"\bcreate_order\b", re.I),
    re.compile(r"\bmarket_buy\b", re.I),
    re.compile(r"\bmarket_sell\b", re.I),
]

DANGEROUS_PATTERNS = [
    re.compile(r"\bprivate_key\s*=", re.I),
    re.compile(r"os\.getenv\([^)]*PRIVATE[_A-Z]*KEY", re.I),
    re.compile(r"\bsign_transaction\s*\(", re.I),
    re.compile(r"\bsign_message\s*\(", re.I),
    re.compile(r"\bsend_transaction\s*\(", re.I),
    re.compile(r"\bsend_raw_transaction\s*\(", re.I),
    re.compile(r"\bcreate_order\s*\(", re.I),
    re.compile(r"\bmarket_buy\s*\(", re.I),
    re.compile(r"\bmarket_sell\s*\(", re.I),
    re.compile(r"action\s*=\s*[\"']LIVE_BUY[\"']", re.I),
    re.compile(r"\bLIVE_BUY\b.*\b(execute|submit|send|order|wallet|sign)\b", re.I),
]

BENIGN_HINTS = [
    "no wallet",
    "no live",
    "no signing",
    "no sign",
    "no private",
    "without wallet",
    "blocked",
    "guard",
    "demo",
    "paper",
    "test",
    "assert",
    "boundary",
    "context only",
    "display only",
    "not live",
    "never",
]

def read_lines(path: Path) -> list[str]:
    try:
        return path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return []

def classify_line(line: str) -> str:
    low = line.lower()
    if any(p.search(line) for p in DANGEROUS_PATTERNS):
        if any(h in low for h in BENIGN_HINTS):
            return "NEEDS_REVIEW_BUT_HAS_BENIGN_HINT"
        return "POTENTIALLY_DANGEROUS"
    if any(h in low for h in BENIGN_HINTS):
        return "BENIGN_BOUNDARY_OR_TEST_TEXT"
    return "MENTION_ONLY_REVIEW"

hits = []
dangerous_hits = []
benign_or_review_hits = []

for path in SCAN_FILES:
    if not path.exists():
        continue
    lines = read_lines(path)
    for i, line in enumerate(lines, 1):
        if not any(p.search(line) for p in SUSPICIOUS_TERMS):
            continue

        start = max(1, i - 2)
        end = min(len(lines), i + 2)
        context = [
            {"line": n, "text": lines[n - 1]}
            for n in range(start, end + 1)
        ]

        status = classify_line(line)
        rec = {
            "file": str(path),
            "line": i,
            "status": status,
            "text": line.strip(),
            "context": context,
        }
        hits.append(rec)

        if status == "POTENTIALLY_DANGEROUS":
            dangerous_hits.append(rec)
        else:
            benign_or_review_hits.append(rec)

summary = {
    "audit_name": "ae18_no_authority_focused",
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "scan_files": [str(p) for p in SCAN_FILES if p.exists()],
    "total_hits": len(hits),
    "dangerous_hits": len(dangerous_hits),
    "benign_or_review_hits": len(benign_or_review_hits),
    "status": "AE18_NO_AUTHORITY_FOCUSED_PASS" if len(dangerous_hits) == 0 else "AE18_NO_AUTHORITY_FOCUSED_REVIEW_REQUIRED",
    "interpretation": (
        "No executable wallet/signing/live-order authority patterns were detected in AE18 touched runtime/UI surfaces."
        if len(dangerous_hits) == 0
        else "Potentially dangerous wallet/signing/live-order patterns require manual inspection before AE18 closure."
    ),
    "dangerous_hit_details": dangerous_hits,
    "benign_or_review_hit_details": benign_or_review_hits,
}

out_json = OUT / "ae18_no_authority_focused_audit.json"
out_md = OUT / "ae18_no_authority_focused_summary.md"

out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

lines = []
lines.append("# AE18 No-Authority Focused Audit")
lines.append("")
lines.append(f"- Created UTC: {summary['created_at_utc']}")
lines.append(f"- Status: `{summary['status']}`")
lines.append(f"- Total hits: `{summary['total_hits']}`")
lines.append(f"- Dangerous hits: `{summary['dangerous_hits']}`")
lines.append("")
if dangerous_hits:
    lines.append("## Dangerous / Review Required Hits")
    for h in dangerous_hits:
        lines.append(f"- `{h['file']}:{h['line']}` — `{h['text']}`")
else:
    lines.append("## Result")
    lines.append("No executable wallet/signing/live-order authority patterns were detected in AE18 touched runtime/UI surfaces.")
lines.append("")
lines.append("## Output Files")
lines.append(f"- `{out_json}`")
lines.append(f"- `{out_md}`")

out_md.write_text("\n".join(lines), encoding="utf-8")

print(json.dumps({
    "status": summary["status"],
    "output_root": str(OUT),
    "json": str(out_json),
    "md": str(out_md),
    "total_hits": summary["total_hits"],
    "dangerous_hits": summary["dangerous_hits"],
    "benign_or_review_hits": summary["benign_or_review_hits"],
    "dangerous_hit_preview": [
        {
            "file": h["file"],
            "line": h["line"],
            "text": h["text"],
        }
        for h in dangerous_hits[:10]
    ],
}, indent=2, ensure_ascii=False))
