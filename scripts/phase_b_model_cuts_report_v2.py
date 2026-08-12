from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(".").resolve()

SEARCH_ROOTS = [
    ROOT / "data" / "training" / "manual_verified_results",
    ROOT / "data" / "training" / "models",
    ROOT / "data" / "training",
]

OUT_DIR = ROOT / "data" / "training" / "manual_verified_results" / "phase_b_model_cuts_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)


TEXT_COLS = {
    "source_kind",
    "model",
    "strategy",
    "filter",
    "horizon",
    "pair_cap",
    "pair_cap_str",
    "split",
    "split_val",
    "split_test",
    "phase_b_notes",
    "invalid_status_counts_val",
    "invalid_status_counts_test",
}

NUMERIC_EXACT = {
    "top_pct",
    "target_k",
    "target_k_val",
    "target_k_test",
    "pair_cap_val",
    "pair_cap_test",
    "tp_ratio",
    "sl_ratio",
    "round_trip_fee",
    "round_trip_fee_val",
    "round_trip_fee_test",
    "selected",
    "selected_val",
    "selected_test",
    "valid_sims",
    "valid_sims_val",
    "valid_sims_test",
    "valid_sim_rate",
    "valid_sim_rate_val",
    "valid_sim_rate_test",
    "target_precision",
    "target_precision_val",
    "target_precision_test",
    "net_win_rate",
    "net_win_rate_val",
    "net_win_rate_test",
    "avg_net_return",
    "avg_net_return_val",
    "avg_net_return_test",
    "median_net_return",
    "median_net_return_val",
    "median_net_return_test",
    "total_net_return",
    "total_net_return_val",
    "total_net_return_test",
    "tp_count",
    "tp_count_val",
    "tp_count_test",
    "sl_count",
    "sl_count_val",
    "sl_count_test",
    "time_count",
    "time_count_val",
    "time_count_test",
    "unique_pairs",
    "unique_pairs_val",
    "unique_pairs_test",
    "top_pair_share",
    "top_pair_share_val",
    "top_pair_share_test",
    "robust_score",
}


def find_file(name: str) -> Path | None:
    hits = []

    for root in SEARCH_ROOTS:
        if root.exists():
            hits.extend(root.rglob(name))

    if not hits:
        return None

    hits = sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def find_zip() -> Path | None:
    hits = []

    for root in SEARCH_ROOTS:
        if root.exists():
            hits.extend(root.rglob("consensus_upload_pack*.zip"))

    if not hits:
        return None

    hits = sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


def read_csv_required(name: str) -> tuple[pd.DataFrame, str]:
    path = find_file(name)

    if path is None:
        raise SystemExit(f"Missing required file: {name}")

    return pd.read_csv(path), str(path)


def read_csv_from_disk_or_zip(name: str) -> tuple[pd.DataFrame | None, str | None]:
    path = find_file(name)

    if path is not None:
        return pd.read_csv(path), str(path)

    zip_path = find_zip()

    if zip_path is None:
        return None, None

    with zipfile.ZipFile(zip_path) as z:
        if name not in z.namelist():
            return None, str(zip_path)

        with z.open(name) as f:
            return pd.read_csv(f), f"{zip_path}::{name}"


def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    for c in df.columns:
        if c in TEXT_COLS:
            continue

        if c in NUMERIC_EXACT or c.endswith("_val") or c.endswith("_test"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def normalize_policy_table(df: pd.DataFrame, source_kind: str) -> pd.DataFrame:
    df = safe_numeric(df)
    df = df.copy()

    df["source_kind"] = source_kind

    if "model" not in df.columns:
        df["model"] = ""

    if "strategy" not in df.columns:
        df["strategy"] = ""

    # Standalone files may have model but no strategy.
    standalone_mask = df["strategy"].astype(str).str.strip().isin(["", "nan", "NaN", "None"])
    df.loc[standalone_mask & df["model"].astype(str).str.len().gt(0), "strategy"] = df.loc[
        standalone_mask & df["model"].astype(str).str.len().gt(0),
        "model",
    ]

    if "pair_cap_str" not in df.columns:
        if "pair_cap" in df.columns:
            df["pair_cap_str"] = df["pair_cap"].astype(str)
        elif "pair_cap_test" in df.columns:
            df["pair_cap_str"] = df["pair_cap_test"].astype(str)
        elif "pair_cap_val" in df.columns:
            df["pair_cap_str"] = df["pair_cap_val"].astype(str)
        else:
            df["pair_cap_str"] = ""

    if "pair_cap" not in df.columns:
        df["pair_cap"] = df["pair_cap_str"].astype(str)

    return df


def add_phase_b_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = [
        "selected_test",
        "total_net_return_test",
        "avg_net_return_test",
        "unique_pairs_test",
        "top_pair_share_test",
    ]

    for c in required:
        if c not in df.columns:
            df[c] = pd.NA
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["phase_b_positive_test"] = df["total_net_return_test"] > 0
    df["phase_b_avg_positive_ok"] = df["avg_net_return_test"] > 0
    df["phase_b_min_trades_ok"] = df["selected_test"] >= 50
    df["phase_b_unique_pairs_ok"] = df["unique_pairs_test"] >= 7
    df["phase_b_concentration_ok"] = df["top_pair_share_test"] <= 0.25

    df["phase_b_robust_ok"] = (
        df["phase_b_positive_test"]
        & df["phase_b_avg_positive_ok"]
        & df["phase_b_min_trades_ok"]
        & df["phase_b_unique_pairs_ok"]
        & df["phase_b_concentration_ok"]
    )

    notes = []

    for row in df.itertuples(index=False):
        d = row._asdict()
        n = []

        if not d.get("phase_b_positive_test", False):
            n.append("NEGATIVE_TEST_RETURN")
        if not d.get("phase_b_avg_positive_ok", False):
            n.append("NEGATIVE_AVG_RETURN")
        if not d.get("phase_b_min_trades_ok", False):
            n.append("LOW_TRADE_COUNT")
        if not d.get("phase_b_unique_pairs_ok", False):
            n.append("LOW_UNIQUE_PAIRS")
        if not d.get("phase_b_concentration_ok", False):
            n.append("PAIR_CONCENTRATION")

        notes.append("OK" if not n else "|".join(n))

    df["phase_b_notes"] = notes

    return df


def slim(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "source_kind",
        "model",
        "strategy",
        "filter",
        "horizon",
        "top_pct",
        "pair_cap_str",
        "pair_cap",
        "tp_ratio",
        "sl_ratio",
        "selected_test",
        "total_net_return_test",
        "avg_net_return_test",
        "median_net_return_test",
        "target_precision_test",
        "net_win_rate_test",
        "tp_count_test",
        "sl_count_test",
        "time_count_test",
        "unique_pairs_test",
        "top_pair_share_test",
        "selected_val",
        "total_net_return_val",
        "avg_net_return_val",
        "target_precision_val",
        "net_win_rate_val",
        "unique_pairs_val",
        "top_pair_share_val",
        "robust_score",
        "phase_b_robust_ok",
        "phase_b_notes",
    ]

    keep = [c for c in cols if c in df.columns]
    return df[keep].copy()


def rank(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    sort_cols = [
        "phase_b_robust_ok",
        "total_net_return_test",
        "avg_net_return_test",
        "target_precision_test",
        "net_win_rate_test",
    ]

    for c in sort_cols:
        if c not in df.columns:
            df[c] = pd.NA

    return (
        df.sort_values(sort_cols, ascending=[False, False, False, False, False])
        .reset_index(drop=True)
    )


def best_by(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    temp = df.copy()

    for c in cols:
        if c not in temp.columns:
            temp[c] = "UNKNOWN"
        temp[c] = temp[c].fillna("UNKNOWN").astype(str)

    return (
        rank(temp)
        .groupby(cols, dropna=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def write_csv(df: pd.DataFrame, name: str) -> str:
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    return str(path)


def fmt(df: pd.DataFrame, n: int = 20) -> str:
    if df is None or df.empty:
        return "EMPTY"

    cols = [
        "source_kind",
        "model",
        "strategy",
        "filter",
        "horizon",
        "top_pct",
        "pair_cap_str",
        "tp_ratio",
        "sl_ratio",
        "selected_test",
        "total_net_return_test",
        "avg_net_return_test",
        "target_precision_test",
        "net_win_rate_test",
        "unique_pairs_test",
        "top_pair_share_test",
        "phase_b_robust_ok",
        "phase_b_notes",
    ]

    cols = [c for c in cols if c in df.columns]

    return df[cols].head(n).to_string(index=False)


def main() -> None:
    strict_comp, strict_comp_source = read_csv_required(
        "strict_comparison_TAB_RF_XGB_CLEAN_FULL.csv"
    )

    strict50, strict50_source = read_csv_from_disk_or_zip(
        "strict50_validation_selected_consensus_applied_to_test.csv"
    )

    highconf20, highconf20_source = read_csv_from_disk_or_zip(
        "highconf20_validation_selected_consensus_applied_to_test.csv"
    )

    research_only, research_only_source = read_csv_from_disk_or_zip(
        "research_only_best_test_consensus_NOT_FOR_SELECTION.csv"
    )

    standalone = normalize_policy_table(strict_comp, "standalone_validation_selected")
    standalone = add_phase_b_flags(standalone)
    standalone = rank(slim(standalone))

    consensus_frames = []

    if strict50 is not None:
        s50 = normalize_policy_table(strict50, "consensus_strict50")
        s50 = add_phase_b_flags(s50)
        consensus_frames.append(slim(s50))

    if highconf20 is not None:
        hc20 = normalize_policy_table(highconf20, "consensus_highconf20")
        hc20 = add_phase_b_flags(hc20)
        consensus_frames.append(slim(hc20))

    if consensus_frames:
        consensus = pd.concat(consensus_frames, ignore_index=True, sort=False)
        consensus = rank(consensus)
    else:
        consensus = pd.DataFrame()

    all_candidates = rank(pd.concat([standalone, consensus], ignore_index=True, sort=False))

    focused = rank(
        all_candidates[
            all_candidates["filter"].isin(["LIQ_5K_HIGH_ACTIVITY", "NO_WHALE_FILTER"])
        ].copy()
    )

    broad = rank(
        all_candidates[
            all_candidates["filter"].isin(["RAW_ALL_VERIFIED"])
        ].copy()
    )

    low_liq = rank(
        all_candidates[
            all_candidates["filter"].isin(["LOW_LIQ_MOMENTUM"])
        ].copy()
    )

    best_standalone_by_model = best_by(standalone, ["model"])
    best_standalone_by_filter_horizon = best_by(standalone, ["filter", "horizon"])
    best_consensus_by_strategy = best_by(consensus, ["strategy"])
    best_consensus_by_source_strategy = best_by(consensus, ["source_kind", "strategy"])
    best_overall_by_strategy_or_model = best_by(all_candidates, ["source_kind", "strategy"])

    two_of_three = rank(
        consensus[
            consensus["strategy"].astype(str).str.contains("TWO_OF_THREE", case=False, na=False)
        ].copy()
    )

    all3 = rank(
        consensus[
            consensus["strategy"].astype(str).str.contains("ALL3", case=False, na=False)
        ].copy()
    )

    pairwise = rank(
        consensus[
            consensus["strategy"].astype(str).str.contains("INTERSECT", case=False, na=False)
        ].copy()
    )

    strategy_counts = (
        consensus.groupby(["source_kind", "strategy"], dropna=False)
        .size()
        .reset_index(name="rows")
        .sort_values(["source_kind", "strategy"])
        if not consensus.empty
        else pd.DataFrame(columns=["source_kind", "strategy", "rows"])
    )

    outputs = {
        "phase_b_v2_standalone_ranked.csv": standalone,
        "phase_b_v2_consensus_ranked.csv": consensus,
        "phase_b_v2_all_candidates_ranked.csv": all_candidates,
        "phase_b_v2_focused_candidates_ranked.csv": focused,
        "phase_b_v2_broad_candidates_ranked.csv": broad,
        "phase_b_v2_low_liq_candidates_ranked.csv": low_liq,
        "phase_b_v2_best_standalone_by_model.csv": best_standalone_by_model,
        "phase_b_v2_best_standalone_by_filter_horizon.csv": best_standalone_by_filter_horizon,
        "phase_b_v2_best_consensus_by_strategy.csv": best_consensus_by_strategy,
        "phase_b_v2_best_consensus_by_source_strategy.csv": best_consensus_by_source_strategy,
        "phase_b_v2_best_overall_by_strategy_or_model.csv": best_overall_by_strategy_or_model,
        "phase_b_v2_two_of_three_ranked.csv": two_of_three,
        "phase_b_v2_all3_ranked.csv": all3,
        "phase_b_v2_pairwise_intersections_ranked.csv": pairwise,
        "phase_b_v2_strategy_counts.csv": strategy_counts,
    }

    written = {name: write_csv(df, name) for name, df in outputs.items()}

    manifest = {
        "status": "ok",
        "strict_comp_source": strict_comp_source,
        "strict50_source": strict50_source,
        "highconf20_source": highconf20_source,
        "research_only_source": research_only_source,
        "output_dir": str(OUT_DIR),
        "rows": {
            "standalone": int(len(standalone)),
            "consensus": int(len(consensus)),
            "all_candidates": int(len(all_candidates)),
            "focused": int(len(focused)),
            "broad": int(len(broad)),
            "low_liq": int(len(low_liq)),
            "two_of_three": int(len(two_of_three)),
            "all3": int(len(all3)),
            "pairwise": int(len(pairwise)),
        },
        "written": written,
    }

    (OUT_DIR / "phase_b_v2_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = []

    lines.append("PHASE B V2 — STRATEGY-PRESERVED MODEL CUTS")
    lines.append("=" * 120)
    lines.append("")
    lines.append("Inputs")
    lines.append("-" * 120)
    lines.append(f"strict_comp: {strict_comp_source}")
    lines.append(f"strict50:    {strict50_source}")
    lines.append(f"highconf20:  {highconf20_source}")
    lines.append("")
    lines.append("Strategy labels sanity check")
    lines.append("-" * 120)
    lines.append(strategy_counts.to_string(index=False))
    lines.append("")
    lines.append("1) Best standalone by model")
    lines.append("-" * 120)
    lines.append(fmt(best_standalone_by_model, 20))
    lines.append("")
    lines.append("2) Best consensus by strategy")
    lines.append("-" * 120)
    lines.append(fmt(best_consensus_by_strategy, 30))
    lines.append("")
    lines.append("3) Best consensus by source_kind + strategy")
    lines.append("-" * 120)
    lines.append(fmt(best_consensus_by_source_strategy, 40))
    lines.append("")
    lines.append("4) Overall best candidates")
    lines.append("-" * 120)
    lines.append(fmt(all_candidates, 40))
    lines.append("")
    lines.append("5) Focused regime: LIQ_5K_HIGH_ACTIVITY / NO_WHALE_FILTER")
    lines.append("-" * 120)
    lines.append(fmt(focused, 40))
    lines.append("")
    lines.append("6) Broad regime: RAW_ALL_VERIFIED")
    lines.append("-" * 120)
    lines.append(fmt(broad, 40))
    lines.append("")
    lines.append("7) TWO_OF_THREE")
    lines.append("-" * 120)
    lines.append(fmt(two_of_three, 40))
    lines.append("")
    lines.append("8) ALL3")
    lines.append("-" * 120)
    lines.append(fmt(all3, 40))
    lines.append("")
    lines.append("9) Pairwise intersections")
    lines.append("-" * 120)
    lines.append(fmt(pairwise, 40))
    lines.append("")
    lines.append("10) LOW_LIQ_MOMENTUM research-only check")
    lines.append("-" * 120)
    lines.append(fmt(low_liq, 30))
    lines.append("")
    lines.append("Decision criteria")
    lines.append("-" * 120)
    lines.append("Use only validation-selected rows for formal selection.")
    lines.append("Reject or mark research-only if phase_b_robust_ok=False.")
    lines.append("phase_b_robust_ok requires: positive total test return, positive average test return, selected_test>=50, unique_pairs_test>=7, top_pair_share_test<=0.25.")
    lines.append("Research-only best-test rows are not used for formal selection.")
    lines.append("")
    lines.append("Output files")
    lines.append("-" * 120)

    for k, v in written.items():
        lines.append(f"{k}: {v}")

    summary_path = OUT_DIR / "phase_b_v2_summary_for_upload.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("DONE")
    print("Summary:", summary_path)
    print("Manifest:", OUT_DIR / "phase_b_v2_manifest.json")
    print("")
    print("Top strategy counts:")
    print(strategy_counts.to_string(index=False))
    print("")
    print("Best consensus by strategy:")
    print(fmt(best_consensus_by_strategy, 20))


if __name__ == "__main__":
    main()
