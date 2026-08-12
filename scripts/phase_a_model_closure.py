from __future__ import annotations

from pathlib import Path
import json
import zipfile
import pandas as pd
import numpy as np


ROOT = Path(".").resolve()
BASE = ROOT / "data" / "training" / "manual_verified_results"
OUT = BASE / "phase_a_model_closure"
OUT.mkdir(parents=True, exist_ok=True)


def find_one(filename: str) -> Path | None:
    matches = []
    for p in BASE.rglob(filename):
        if ".venv" not in p.parts and p.is_file():
            matches.append(p)
    if not matches:
        return None
    matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return matches[0]


def read_csv_required(path: Path | None, label: str) -> pd.DataFrame:
    if path is None or not path.exists():
        raise SystemExit(f"Missing required file for {label}")
    print(f"[OK] {label}: {path}")
    return pd.read_csv(path)


def safe_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def write_csv(df: pd.DataFrame, name: str) -> Path:
    path = OUT / name
    df.to_csv(path, index=False)
    print("WROTE", path)
    return path


def write_json(payload: dict, name: str) -> Path:
    path = OUT / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("WROTE", path)
    return path


def as_str_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.lower().str.strip()


def production_like(df: pd.DataFrame, prefix: str = "_test") -> pd.DataFrame:
    out = df.copy()

    selected = f"selected{prefix}"
    total = f"total_net_return{prefix}"
    unique_pairs = f"unique_pairs{prefix}"
    top_pair_share = f"top_pair_share{prefix}"

    mask = pd.Series(True, index=out.index)

    if selected in out.columns:
        mask &= pd.to_numeric(out[selected], errors="coerce").fillna(0) >= 50
    if total in out.columns:
        mask &= pd.to_numeric(out[total], errors="coerce").fillna(-10**9) > 0
    if unique_pairs in out.columns:
        mask &= pd.to_numeric(out[unique_pairs], errors="coerce").fillna(0) >= 7
    if top_pair_share in out.columns:
        mask &= pd.to_numeric(out[top_pair_share], errors="coerce").fillna(1.0) <= 0.25

    if "pair_cap_str" in out.columns:
        mask &= as_str_series(out["pair_cap_str"]) != "none"
    elif "pair_cap" in out.columns:
        mask &= as_str_series(out["pair_cap"]) != "none"

    return out[mask].copy()


def top_report(df: pd.DataFrame, name: str, sort_col: str, n: int = 30) -> pd.DataFrame:
    if sort_col not in df.columns:
        return pd.DataFrame()
    out = df.sort_values(sort_col, ascending=False).head(n).copy()
    write_csv(out, name)
    return out


def best_row(df: pd.DataFrame, sort_col: str) -> dict:
    if df.empty or sort_col not in df.columns:
        return {}
    row = df.sort_values(sort_col, ascending=False).iloc[0]
    return row.to_dict()


# --------------------------------------------------------------------
# 1. Load standalone model comparison
# --------------------------------------------------------------------

strict_comp_path = find_one("strict_comparison_TAB_RF_XGB_CLEAN_FULL.csv")
xgb_strict_path = find_one("strict_validation_selected_policies_XGB_CLEAN_FULL_applied_to_test.csv")
xgb_manifest_path = find_one("xgb_clean_full_cuda_manifest.json")
xgb_summary_path = find_one("xgb_full_summary_for_upload.txt")

standalone = read_csv_required(strict_comp_path, "TAB/RF/XGB strict comparison")
xgb_strict = read_csv_required(xgb_strict_path, "XGB strict policies")

manifest = {}
if xgb_manifest_path and xgb_manifest_path.exists():
    manifest = json.loads(xgb_manifest_path.read_text(encoding="utf-8"))
    print("[OK] XGB manifest:", xgb_manifest_path)
else:
    print("[WARN] XGB manifest not found")

# --------------------------------------------------------------------
# 2. Standalone summaries
# --------------------------------------------------------------------

main_cols = safe_cols(standalone, [
    "model", "filter", "horizon",
    "top_pct", "pair_cap_str", "tp_ratio", "sl_ratio",
    "selected_test", "total_net_return_test", "avg_net_return_test",
    "target_precision_test", "net_win_rate_test",
    "tp_count_test", "sl_count_test", "time_count_test",
    "unique_pairs_test", "top_pair_share_test",
    "total_net_return_val", "avg_net_return_val",
    "target_precision_val", "net_win_rate_val",
    "robust_score",
])

standalone_sorted = standalone.sort_values("total_net_return_test", ascending=False)
write_csv(standalone_sorted[main_cols], "phase_a_01_standalone_all_models_ranked_by_test_return.csv")

prod_standalone = production_like(standalone, prefix="_test")
prod_standalone_sorted = prod_standalone.sort_values("total_net_return_test", ascending=False)
write_csv(prod_standalone_sorted[main_cols], "phase_a_02_standalone_production_like_ranked.csv")

best_by_model = (
    prod_standalone_sorted
    .sort_values("total_net_return_test", ascending=False)
    .groupby("model", as_index=False)
    .head(1)
)
write_csv(best_by_model[main_cols], "phase_a_03_best_standalone_by_model.csv")

best_by_filter_model = (
    prod_standalone_sorted
    .sort_values("total_net_return_test", ascending=False)
    .groupby(["model", "filter"], as_index=False)
    .head(1)
)
write_csv(best_by_filter_model[main_cols], "phase_a_04_best_standalone_by_model_and_filter.csv")

# --------------------------------------------------------------------
# 3. Whale proxy check: LIQ_5K_HIGH_ACTIVITY vs NO_WHALE_FILTER
# --------------------------------------------------------------------

whale_rows = standalone[standalone["filter"].isin(["LIQ_5K_HIGH_ACTIVITY", "NO_WHALE_FILTER"])].copy()

key_cols = safe_cols(whale_rows, [
    "model", "horizon", "top_pct", "pair_cap_str", "tp_ratio", "sl_ratio"
])

if key_cols:
    liq = whale_rows[whale_rows["filter"] == "LIQ_5K_HIGH_ACTIVITY"].copy()
    no_whale = whale_rows[whale_rows["filter"] == "NO_WHALE_FILTER"].copy()

    compare_cols = [
        "selected_test", "total_net_return_test", "avg_net_return_test",
        "target_precision_test", "net_win_rate_test",
        "unique_pairs_test", "top_pair_share_test",
    ]
    compare_cols = safe_cols(whale_rows, compare_cols)

    liq_small = liq[key_cols + compare_cols].copy()
    no_small = no_whale[key_cols + compare_cols].copy()

    merged = liq_small.merge(
        no_small,
        on=key_cols,
        how="inner",
        suffixes=("_liq", "_no_whale"),
    )

    for c in compare_cols:
        a = f"{c}_liq"
        b = f"{c}_no_whale"
        if a in merged.columns and b in merged.columns:
            merged[f"delta_{c}"] = pd.to_numeric(merged[a], errors="coerce") - pd.to_numeric(merged[b], errors="coerce")

    write_csv(merged, "phase_a_05_liq_vs_no_whale_filter_delta.csv")
else:
    merged = pd.DataFrame()
    print("[WARN] Could not build LIQ vs NO_WHALE delta table")

# --------------------------------------------------------------------
# 4. Consensus files
# --------------------------------------------------------------------

consensus_zip = find_one("consensus_upload_pack*.zip")
strict50 = pd.DataFrame()
highconf20 = pd.DataFrame()
research_only = pd.DataFrame()
consensus_manifest = {}

if consensus_zip and consensus_zip.exists():
    print("[OK] Consensus zip:", consensus_zip)
    with zipfile.ZipFile(consensus_zip) as z:
        names = set(z.namelist())

        if "consensus_manifest.json" in names:
            consensus_manifest = json.loads(z.read("consensus_manifest.json").decode("utf-8"))

        if "strict50_validation_selected_consensus_applied_to_test.csv" in names:
            strict50 = pd.read_csv(z.open("strict50_validation_selected_consensus_applied_to_test.csv"))

        if "highconf20_validation_selected_consensus_applied_to_test.csv" in names:
            highconf20 = pd.read_csv(z.open("highconf20_validation_selected_consensus_applied_to_test.csv"))

        if "research_only_best_test_consensus_NOT_FOR_SELECTION.csv" in names:
            research_only = pd.read_csv(z.open("research_only_best_test_consensus_NOT_FOR_SELECTION.csv"))
else:
    # fallback: maybe files were extracted rather than zipped
    s = find_one("strict50_validation_selected_consensus_applied_to_test.csv")
    h = find_one("highconf20_validation_selected_consensus_applied_to_test.csv")
    r = find_one("research_only_best_test_consensus_NOT_FOR_SELECTION.csv")

    if s:
        strict50 = pd.read_csv(s)
        print("[OK] strict50:", s)
    if h:
        highconf20 = pd.read_csv(h)
        print("[OK] highconf20:", h)
    if r:
        research_only = pd.read_csv(r)
        print("[OK] research_only:", r)

consensus_cols = [
    "strategy", "filter", "horizon",
    "top_pct", "pair_cap", "tp_ratio", "sl_ratio",
    "selected_test", "total_net_return_test", "avg_net_return_test",
    "target_precision_test", "net_win_rate_test",
    "tp_count_test", "sl_count_test", "time_count_test",
    "unique_pairs_test", "top_pair_share_test",
    "total_net_return_val", "avg_net_return_val",
    "target_precision_val", "net_win_rate_val",
    "robust_score",
]

if not strict50.empty:
    cols = safe_cols(strict50, consensus_cols)
    strict50_by_test = strict50.sort_values("total_net_return_test", ascending=False)
    write_csv(strict50_by_test[cols], "phase_a_06_consensus_strict50_ranked_by_test_return.csv")

    strict50_prod = production_like(strict50, prefix="_test")
    strict50_prod_sorted = strict50_prod.sort_values("total_net_return_test", ascending=False)
    write_csv(strict50_prod_sorted[safe_cols(strict50_prod_sorted, consensus_cols)], "phase_a_07_consensus_strict50_production_like.csv")
else:
    print("[WARN] strict50 consensus file not found")

if not highconf20.empty:
    cols = safe_cols(highconf20, consensus_cols)
    highconf20_by_test = highconf20.sort_values("total_net_return_test", ascending=False)
    write_csv(highconf20_by_test[cols], "phase_a_08_consensus_highconf20_ranked_by_test_return.csv")

    highconf20_prod = production_like(highconf20, prefix="_test")
    highconf20_prod_sorted = highconf20_prod.sort_values("total_net_return_test", ascending=False)
    write_csv(highconf20_prod_sorted[safe_cols(highconf20_prod_sorted, consensus_cols)], "phase_a_09_consensus_highconf20_production_like.csv")
else:
    print("[WARN] highconf20 consensus file not found")

if not research_only.empty:
    cols = safe_cols(research_only, [
        "strategy", "split", "filter", "horizon",
        "top_pct", "pair_cap", "tp_ratio", "sl_ratio",
        "selected", "total_net_return", "avg_net_return",
        "target_precision", "net_win_rate",
        "unique_pairs", "top_pair_share",
    ])
    write_csv(
        research_only.sort_values("total_net_return", ascending=False)[cols],
        "phase_a_10_research_only_best_test_NOT_FOR_SELECTION.csv",
    )

# --------------------------------------------------------------------
# 5. Official Phase A recommendation JSON/TXT
# --------------------------------------------------------------------

official = {
    "phase": "A_model_closure",
    "created_by": "phase_a_model_closure script",
    "xgb_manifest": {
        "path": str(xgb_manifest_path) if xgb_manifest_path else None,
        "use_cuda": manifest.get("use_cuda"),
        "dataset_dir": manifest.get("dataset_dir"),
        "xgb_out_dir": manifest.get("xgb_out_dir"),
        "exit_sim_out_dir": manifest.get("exit_sim_out_dir"),
        "filters": manifest.get("filters"),
        "horizons": manifest.get("horizons"),
        "duration_seconds": manifest.get("duration_seconds"),
    },
    "standalone": {
        "best_production_like_by_test_return": best_row(prod_standalone_sorted, "total_net_return_test"),
        "best_by_model": best_by_model.to_dict(orient="records"),
        "recommendation": "XGB should be treated as the leading standalone broad ranking model if it remains the best production-like row.",
    },
    "consensus": {
        "strict50_best_production_like_by_test_return": best_row(
            strict50_prod_sorted if not strict50.empty else pd.DataFrame(),
            "total_net_return_test",
        ),
        "highconf20_best_production_like_by_test_return": best_row(
            highconf20_prod_sorted if not highconf20.empty else pd.DataFrame(),
            "total_net_return_test",
        ),
        "recommendation": "2-of-3 consensus should be treated as a high-confidence overlay. All-three consensus should not be used as the primary rule unless future out-of-time data proves otherwise.",
    },
    "whale_filter": {
        "liq_vs_no_whale_rows_compared": int(len(merged)) if isinstance(merged, pd.DataFrame) else 0,
        "recommendation": "Do not drop whales. Treat the current whale filter as insufficient; move whale analysis to wallet-level Solana/Helius enrichment.",
    },
    "phase_a_decision": {
        "close_heavy_sweeps_on_x2": True,
        "next_phase": "Build direct net_profitable_after_exit_policy target and meta-model/stacking dataset.",
        "do_not_do_in_phase_a": [
            "No Gemini calls",
            "No Qwen calls",
            "No Helius calls",
            "No live/demo behavior changes",
            "No new training unless explicitly starting Phase B/C",
        ],
    },
}

write_json(official, "phase_a_11_official_model_strategy.json")

summary_lines = []
summary_lines.append("PHASE A — MODEL CLOSURE SUMMARY")
summary_lines.append("=" * 80)
summary_lines.append("")
summary_lines.append("Standalone production-like leaderboard:")
summary_lines.append(prod_standalone_sorted[main_cols].head(15).to_string(index=False))
summary_lines.append("")
summary_lines.append("")
summary_lines.append("Best standalone by model:")
summary_lines.append(best_by_model[main_cols].to_string(index=False))
summary_lines.append("")
summary_lines.append("")
if not strict50.empty:
    summary_lines.append("Consensus Strict50 production-like leaderboard:")
    summary_lines.append(strict50_prod_sorted[safe_cols(strict50_prod_sorted, consensus_cols)].head(15).to_string(index=False))
    summary_lines.append("")
if not highconf20.empty:
    summary_lines.append("Consensus HighConf20 production-like leaderboard:")
    summary_lines.append(highconf20_prod_sorted[safe_cols(highconf20_prod_sorted, consensus_cols)].head(15).to_string(index=False))
    summary_lines.append("")
summary_lines.append("")
summary_lines.append("Recommended strategic interpretation:")
summary_lines.append("- XGB = leading standalone broad ranking model.")
summary_lines.append("- TAB = still valuable in focused liquidity/activity regimes.")
summary_lines.append("- RF = weak standalone but useful confirmation/sanity signal.")
summary_lines.append("- 2-of-3 consensus = high-confidence overlay.")
summary_lines.append("- All-three consensus = reject for now; over-filters and transfers poorly.")
summary_lines.append("- Current whale filter did not prove incremental value; do not abandon whales, rebuild them as wallet-level intelligence.")
summary_lines.append("- Qwen/Gemini stay outside Phase A; they enter later as reasoning/reputation/explanation/veto layers, not numeric entry models.")

summary_path = OUT / "phase_a_12_model_closure_summary.txt"
summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
print("WROTE", summary_path)

print("")
print("=" * 80)
print("PHASE A OUTPUT PACK READY")
print("=" * 80)
print("Folder:", OUT)
print("")
print("Send/upload these files:")
print("1. phase_a_12_model_closure_summary.txt")
print("2. phase_a_11_official_model_strategy.json")
print("3. phase_a_02_standalone_production_like_ranked.csv")
print("4. phase_a_07_consensus_strict50_production_like.csv")
print("5. phase_a_05_liq_vs_no_whale_filter_delta.csv")
print("")
