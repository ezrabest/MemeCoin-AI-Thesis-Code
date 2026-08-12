#!/usr/bin/env python3
"""Patch AE13J markdown report identity section after verdict repair."""
from __future__ import annotations

import csv
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "audits" / "ae13j_ground_truth_market_data_audit_20260720_175255"
report_path = OUT / "ae13j_ground_truth_market_data_report.md"
rows = list(csv.DictReader((OUT / "ae13j_token_pair_identity_verdict.csv").open(encoding="utf-8")))

# Highlight key samples
needles = (
    "DDk1QmkbZBtTSpU2oKMmH2jWZFeansd4Z6hku7k1Dfct",
    "9VW8yfZaf2GcEpVb4apuk63oGVnebYZ4pr7ymc8Ftx3i",
    "0xd2391dB4D7B9841b989521088c3Bf8C4cFe404d8",
    "5GZJHDbRVHvHRmah21Tq22V9J1iu3iyvLHwHpJuuiSrm",
    "0x40ec64Cac5F5139605CcFffA9977a28046E2c3e0",
)
lines = []
for n in needles:
    hits = [r for r in rows if r["sampled_address"] == n]
    if not hits:
        continue
    r = hits[0]
    lines.append(
        f"- `{r['sampled_address']}` → **{r['identity_verdict']}** "
        f"({r.get('displayed_symbol')}, {r.get('chain')}, dexId={r.get('dexId')})"
    )

extra = []
for r in rows[:8]:
    if r["sampled_address"] in needles:
        continue
    extra.append(
        f"- `{r['sampled_address']}` → **{r['identity_verdict']}** "
        f"({r.get('displayed_symbol')}, {r.get('chain')})"
    )

block = "\n".join(lines + extra[:6])
text = report_path.read_text(encoding="utf-8")
marker = "Identity sample:"
if marker in text:
    pre, rest = text.split(marker, 1)
    # drop old bullets until next ##
    after = rest.split("\n## ", 1)
    tail = ("\n## " + after[1]) if len(after) > 1 else ""
    text = pre + marker + "\n\n" + block + "\n" + tail
    report_path.write_text(text, encoding="utf-8")
    print("report patched")
else:
    print("marker missing")

# also note UI applied in summary
summary = OUT / "ae13j_summary_for_upload.txt"
s = summary.read_text(encoding="utf-8")
if "ui_label_applied: true" not in s:
    s = s.replace(
        "ui_truth_label: Market Snapshot Feed",
        "ui_truth_label: Market Snapshot Feed\nui_label_applied: true",
    )
    summary.write_text(s, encoding="utf-8")
print("done")
