from __future__ import annotations

import csv
import json
import math
from pathlib import Path

latest = Path(r"E:\Projects\Final Project\memecoin_trader\data\audits\thesis_statistical_robustness_audit_20260810_190513")
inp = latest / "04_canonical_candidate_rows_for_thesis_review.csv"

canonical_source = "phase_b_v5_audited_selected_trades.csv"
baseline_tier = "XGB_RF_ONLY"

wanted_tiers = [
    "TAB_RF_ONLY",
    "TAB_XGB_RF_ALL3",
    "TAB_XGB_ONLY",
    "XGB_RF_ONLY",
]

def ztest_two_prop(pos1, n1, pos0, n0):
    p_pool = (pos1 + pos0) / (n1 + n0)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n0))
    if se == 0:
        return None, None
    z = (pos1/n1 - pos0/n0) / se
    p_two = math.erfc(abs(z) / math.sqrt(2))
    return z, p_two

rows = []
with inp.open("r", encoding="utf-8-sig", newline="") as f:
    for r in csv.DictReader(f):
        if r["source_name"] == canonical_source and r["tier"] in wanted_tiers:
            rows.append(r)

if len(rows) != 4:
    raise SystemExit(f"Expected 4 canonical rows, got {len(rows)}")

by_tier = {r["tier"]: r for r in rows}
base = by_tier[baseline_tier]
base_pos = int(base["positive"])
base_n = int(base["n_event_dedup"])

out_rows = []
for tier in wanted_tiers:
    r = by_tier[tier]
    n = int(r["n_event_dedup"])
    pos = int(r["positive"])
    rate = float(r["positive_rate"])
    ci_low = float(r["positive_rate_ci95_low"])
    ci_high = float(r["positive_rate_ci95_high"])
    avg_return = float(r["avg_return"])
    z, p = (None, None) if tier == baseline_tier else ztest_two_prop(pos, n, base_pos, base_n)

    if tier in ("TAB_RF_ONLY", "TAB_XGB_RF_ALL3"):
        thesis_interpretation = "positive evidence tier"
    elif tier == "TAB_XGB_ONLY":
        thesis_interpretation = "positive-rate misleading; negative average return"
    else:
        thesis_interpretation = "baseline weak/rejected control"

    out_rows.append({
        "canonical_source": canonical_source,
        "tier": tier,
        "n_event_dedup": n,
        "positive": pos,
        "positive_rate": rate,
        "positive_rate_pct": rate * 100,
        "positive_rate_ci95_low": ci_low,
        "positive_rate_ci95_high": ci_high,
        "positive_rate_ci95_low_pct": ci_low * 100,
        "positive_rate_ci95_high_pct": ci_high * 100,
        "avg_net_return": avg_return,
        "avg_net_return_pct": avg_return * 100,
        "comparison_baseline_tier": baseline_tier if tier != baseline_tier else "",
        "z_vs_baseline": z,
        "p_two_sided_vs_baseline": p,
        "thesis_interpretation": thesis_interpretation,
    })

csv_out = latest / "05_thesis_table_8_2_canonical_model_tier_robustness.csv"
json_out = latest / "05_thesis_table_8_2_canonical_model_tier_robustness.json"
md_out = latest / "05_thesis_table_8_2_canonical_model_tier_robustness.md"

with csv_out.open("w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
    w.writeheader()
    w.writerows(out_rows)

with json_out.open("w", encoding="utf-8") as f:
    json.dump({
        "classification": "THESIS_TABLE_8_2_CANONICAL_MODEL_TIER_ROBUSTNESS_READY",
        "canonical_source": canonical_source,
        "duplicate_lineage_sources_not_counted_as_independent": [
            "phase_c_v5_selected_trades_candidate_pool_v5.csv",
            "phase_d_v4_selected_trades_with_direct_target.csv",
        ],
        "baseline_tier": baseline_tier,
        "rows": out_rows,
        "interpretation": (
            "The old small n=4/n=8 integrated subset should be replaced by this broader "
            "retrospective event-level robustness table. This remains retrospective evidence, "
            "not live trading proof."
        )
    }, f, indent=2)

lines = []
lines.append("# Thesis Table 8.2 — Canonical Model-Tier Robustness")
lines.append("")
lines.append(f"Canonical source: `{canonical_source}`")
lines.append("")
lines.append("Duplicate lineage sources not counted as independent:")
lines.append("- `phase_c_v5_selected_trades_candidate_pool_v5.csv`")
lines.append("- `phase_d_v4_selected_trades_with_direct_target.csv`")
lines.append("")
lines.append("| Tier | n | Positive rate | 95% CI | Avg net return | p vs XGB_RF_ONLY | Interpretation |")
lines.append("|---|---:|---:|---:|---:|---:|---|")
for r in out_rows:
    p = r["p_two_sided_vs_baseline"]
    p_s = "" if p is None else f"{p:.3e}"
    lines.append(
        f"| `{r['tier']}` | {r['n_event_dedup']:,} | "
        f"{r['positive_rate_pct']:.2f}% | "
        f"{r['positive_rate_ci95_low_pct']:.2f}–{r['positive_rate_ci95_high_pct']:.2f}% | "
        f"{r['avg_net_return_pct']:+.2f}% | "
        f"{p_s} | "
        f"{r['thesis_interpretation']} |"
    )
lines.append("")
lines.append("Interpretation:")
lines.append("The broader audit resolves the micro-sample criticism of the earlier n=4/n=8 subset.")
lines.append("However, it remains retrospective event-level evidence, not paper-forward or live-trading proof.")
lines.append("Positive rate alone is insufficient: TAB_XGB_ONLY has a moderate positive rate but negative average net return.")

md_out.write_text("\n".join(lines), encoding="utf-8")

print(json.dumps({
    "status": "OK",
    "csv": str(csv_out),
    "json": str(json_out),
    "md": str(md_out),
}, indent=2))

print()
print(md_out.read_text(encoding="utf-8"))
