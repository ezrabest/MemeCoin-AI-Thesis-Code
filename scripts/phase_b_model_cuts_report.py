from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".").resolve()

SEARCH_ROOTS = [
    ROOT / "data" / "training" / "manual_verified_results",
    ROOT / "data" / "training" / "models",
    ROOT / "data" / "training",
]

OUT_DIR = ROOT / "data" / "training" / "manual_verified_results" / "phase_b_model_cuts"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_file(name: str) -> Path | None:
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        hits = list(root.rglob(name))
        if hits:
            hits = sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)
            return hits[0]
    return None


def read_csv_required(name: str) -> tuple[pd.DataFrame, Path]:
    path = find_file(name)
    if path is None:
        raise SystemExit(f"Missing required file: {name}")
    return pd.read_csv(path), path


def read_csv_from_zip_or_disk(csv_name: str) -> tuple[pd.DataFrame | None, str | None]:
    disk_path = find_file(csv_name)
    if disk_path is not None:
        return pd.read_csv(disk_path), str(disk_path)

    zip_path = find_file("consensus_upload_pack.zip")
    if zip_path is None:
        zip_path = find_file("consensus_upload_pack(1).zip")

    if zip_path is None:
        return None, None

    with zipfile.ZipFile(zip_path) as z:
        if csv_name not in z.namelist():
            return None, str(zip_path)
        with z.open(csv_name) as f:
            return pd.read_csv(f), f"{zip_path}::{csv_name}"


def clean_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if any(k in c for k in ["return", "precision", "rate", "share", "selected", "pairs", "count", "pct", "ratio", "k"]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_policy_flags(df: pd.DataFrame, prefix: str = "test") -> pd.DataFrame:
    df = df.copy()

    selected = f"selected_{prefix}"
    unique = f"unique_pairs_{prefix}"
    share = f"top_pair_share_{prefix}"
    total = f"total_net_return_{prefix}"
    avg = f"avg_net_return_{prefix}"

    for c in [selected, unique, share, total, avg]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["phase_b_positive_test"] = df[total] > 0 if total in df.columns else False
    df["phase_b_min_trades_ok"] = df[selected] >= 50 if selected in df.columns else False
    df["phase_b_unique_pairs_ok"] = df[unique] >= 7 if unique in df.columns else False
    df["phase_b_concentration_ok"] = df[share] <= 0.25 if share in df.columns else False
    df["phase_b_avg_positive_ok"] = df[avg] > 0 if avg in df.columns else False

    df["phase_b_robust_ok"] = (
        df["phase_b_positive_test"]
        & df["phase_b_min_trades_ok"]
        & df["phase_b_unique_pairs_ok"]
        & df["phase_b_concentration_ok"]
        & df["phase_b_avg_positive_ok"]
    )

    notes = []
    for row in df.itertuples(index=False):
        rowd = row._asdict()
        n = []
        if not rowd.get("phase_b_positive_test", False):
            n.append("NEGATIVE_TEST_RETURN")
        if not rowd.get("phase_b_min_trades_ok", False):
            n.append("LOW_TRADE_COUNT")
        if not rowd.get("phase_b_unique_pairs_ok", False):
            n.append("LOW_UNIQUE_PAIRS")
        if not rowd.get("phase_b_concentration_ok", False):
            n.append("PAIR_CONCENTRATION")
        if not rowd.get("phase_b_avg_positive_ok", False):
            n.append("NEGATIVE_AVG_RETURN")
        notes.append("|".join(n) if n else "OK")

    df["phase_b_notes"] = notes
    return df


def slim_policy_cols(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    base_cols = [
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

    out = df.copy()
    out["source_kind"] = kind

    if "model" not in out.columns:
        out["model"] = ""
    if "strategy" not in out.columns:
        out["strategy"] = out["model"]

    if "pair_cap_str" not in out.columns and "pair_cap" in out.columns:
        out["pair_cap_str"] = out["pair_cap"].astype(str)
    if "pair_cap" not in out.columns and "pair_cap_str" in out.columns:
        out["pair_cap"] = out["pair_cap_str"].astype(str)

    cols = [c for c in base_cols if c in out.columns]
    return out[cols].copy()


def write_csv(df: pd.DataFrame, name: str) -> Path:
    path = OUT_DIR / name
    df.to_csv(path, index=False)
    return path


def format_top(df: pd.DataFrame, n: int = 15) -> str:
    if df is None or df.empty:
        return "EMPTY"

    cols = [
        "source_kind", "model", "strategy", "filter", "horizon",
        "top_pct", "pair_cap_str", "pair_cap", "tp_ratio", "sl_ratio",
        "selected_test", "total_net_return_test", "avg_net_return_test",
        "target_precision_test", "net_win_rate_test",
        "unique_pairs_test", "top_pair_share_test",
        "phase_b_robust_ok", "phase_b_notes",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].head(n).to_string(index=False)


def best_by_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    return (
        df.sort_values(
            ["phase_b_robust_ok", "total_net_return_test", "avg_net_return_test", "target_precision_test"],
            ascending=[False, False, False, False],
        )
        .groupby(group_cols, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )


def main() -> None:
    manifest = {}

    strict_comp, strict_comp_path = read_csv_required("strict_comparison_TAB_RF_XGB_CLEAN_FULL.csv")
    xgb_strict, xgb_strict_path = read_csv_required("strict_validation_selected_policies_XGB_CLEAN_FULL_applied_to_test.csv")

    strict_comp = add_policy_flags(clean_numeric(strict_comp), "test")
    xgb_strict = add_policy_flags(clean_numeric(xgb_strict), "test")

    strict50, strict50_source = read_csv_from_zip_or_disk("strict50_validation_selected_consensus_applied_to_test.csv")
    highconf20, highconf20_source = read_csv_from_zip_or_disk("highconf20_validation_selected_consensus_applied_to_test.csv")
    research_only, research_source = read_csv_from_zip_or_disk("research_only_best_test_consensus_NOT_FOR_SELECTION.csv")

    if strict50 is not None:
        strict50 = add_policy_flags(clean_numeric(strict50), "test")
    if highconf20 is not None:
        highconf20 = add_policy_flags(clean_numeric(highconf20), "test")
    if research_only is not None:
        research_only = clean_numeric(research_only)

    single = slim_policy_cols(strict_comp, "standalone_validation_selected")
    single_ranked = single.sort_values(
        ["phase_b_robust_ok", "total_net_return_test", "avg_net_return_test", "target_precision_test"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    best_standalone_by_model = best_by_group(single_ranked, ["model"])
    best_standalone_by_filter_horizon = best_by_group(single_ranked, ["filter", "horizon"])

    consensus_frames = []

    if strict50 is not None:
        sf = slim_policy_cols(strict50, "consensus_strict50")
        consensus_frames.append(sf)

    if highconf20 is not None:
        hf = slim_policy_cols(highconf20, "consensus_highconf20")
        consensus_frames.append(hf)

    consensus = pd.concat(consensus_frames, ignore_index=True, sort=False) if consensus_frames else pd.DataFrame()
    consensus_ranked = consensus.sort_values(
        ["phase_b_robust_ok", "total_net_return_test", "avg_net_return_test", "target_precision_test"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True) if not consensus.empty else consensus

    best_consensus_by_strategy = best_by_group(consensus_ranked, ["strategy"]) if not consensus_ranked.empty else pd.DataFrame()

    all_candidates = pd.concat([single_ranked, consensus_ranked], ignore_index=True, sort=False)
    all_candidates = all_candidates.sort_values(
        ["phase_b_robust_ok", "total_net_return_test", "avg_net_return_test", "target_precision_test"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)

    focused = all_candidates[
        all_candidates["filter"].isin(["LIQ_5K_HIGH_ACTIVITY", "NO_WHALE_FILTER"])
    ].copy()

    broad = all_candidates[
        all_candidates["filter"].isin(["RAW_ALL_VERIFIED"])
    ].copy()

    low_liq = all_candidates[
        all_candidates["filter"].isin(["LOW_LIQ_MOMENTUM"])
    ].copy()

    # Specific diagnostics
    all3_rows = consensus_ranked[consensus_ranked["strategy"].astype(str).str.contains("ALL3", case=False, na=False)].copy() if not consensus_ranked.empty else pd.DataFrame()
    two_rows = consensus_ranked[consensus_ranked["strategy"].astype(str).str.contains("TWO_OF_THREE", case=False, na=False)].copy() if not consensus_ranked.empty else pd.DataFrame()

    outputs = {
        "phase_b_single_model_ranked.csv": single_ranked,
        "phase_b_best_standalone_by_model.csv": best_standalone_by_model,
        "phase_b_best_standalone_by_filter_horizon.csv": best_standalone_by_filter_horizon,
        "phase_b_consensus_ranked.csv": consensus_ranked,
        "phase_b_best_consensus_by_strategy.csv": best_consensus_by_strategy,
        "phase_b_all_candidates_ranked.csv": all_candidates,
        "phase_b_focused_candidates_ranked.csv": focused,
        "phase_b_broad_candidates_ranked.csv": broad,
        "phase_b_low_liq_candidates_ranked.csv": low_liq,
    }

    written = {}
    for name, df in outputs.items():
        written[name] = str(write_csv(df, name))

    manifest = {
        "status": "ok",
        "strict_comparison_source": str(strict_comp_path),
        "xgb_strict_source": str(xgb_strict_path),
        "strict50_source": strict50_source,
        "highconf20_source": highconf20_source,
        "research_only_source": research_source,
        "output_dir": str(OUT_DIR),
        "rows": {
            "single": int(len(single_ranked)),
            "consensus": int(len(consensus_ranked)),
            "all_candidates": int(len(all_candidates)),
            "focused": int(len(focused)),
            "broad": int(len(broad)),
            "low_liq": int(len(low_liq)),
        },
        "written": written,
    }

    (OUT_DIR / "phase_b_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    lines = []
    lines.append("PHASE B — TAB / RF / XGB / CONSENSUS MODEL CUTS")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Inputs")
    lines.append("-" * 100)
    lines.append(f"strict_comparison: {strict_comp_path}")
    lines.append(f"xgb_strict:        {xgb_strict_path}")
    lines.append(f"strict50:          {strict50_source}")
    lines.append(f"highconf20:        {highconf20_source}")
    lines.append("")
    lines.append("1) Best standalone by model")
    lines.append("-" * 100)
    lines.append(format_top(best_standalone_by_model, 20))
    lines.append("")
    lines.append("2) Overall ranked candidates")
    lines.append("-" * 100)
    lines.append(format_top(all_candidates, 30))
    lines.append("")
    lines.append("3) Focused regime only: LIQ_5K_HIGH_ACTIVITY / NO_WHALE_FILTER")
    lines.append("-" * 100)
    lines.append(format_top(focused, 30))
    lines.append("")
    lines.append("4) Broad regime only: RAW_ALL_VERIFIED")
    lines.append("-" * 100)
    lines.append(format_top(broad, 30))
    lines.append("")
    lines.append("5) Low-liq momentum research regime")
    lines.append("-" * 100)
    lines.append(format_top(low_liq, 30))
    lines.append("")
    lines.append("6) Consensus by strategy")
    lines.append("-" * 100)
    lines.append(format_top(best_consensus_by_strategy, 30))
    lines.append("")
    lines.append("7) TWO_OF_THREE rows")
    lines.append("-" * 100)
    lines.append(format_top(two_rows.sort_values(['phase_b_robust_ok','total_net_return_test'], ascending=[False, False]), 30) if len(two_rows) else "EMPTY")
    lines.append("")
    lines.append("8) ALL3 rows")
    lines.append("-" * 100)
    lines.append(format_top(all3_rows.sort_values(['phase_b_robust_ok','total_net_return_test'], ascending=[False, False]), 30) if len(all3_rows) else "EMPTY")
    lines.append("")
    lines.append("Interpretation checklist for user")
    lines.append("-" * 100)
    lines.append("A. Confirm whether XGB is best standalone broad model.")
    lines.append("B. Confirm whether TAB remains valuable in focused LIQ/NO_WHALE regime.")
    lines.append("C. Confirm whether TWO_OF_THREE beats pairwise and ALL3 after validation selection.")
    lines.append("D. Confirm whether ALL3 over-filters and fails to transfer to test.")
    lines.append("E. Confirm whether LOW_LIQ_MOMENTUM is research-only due to low unique-pair coverage.")
    lines.append("F. Confirm whether NO_WHALE_FILTER ~= LIQ_5K_HIGH_ACTIVITY, meaning current whale proxy adds little.")
    lines.append("")
    lines.append("Output files")
    lines.append("-" * 100)
    for k, v in written.items():
        lines.append(f"{k}: {v}")

    summary_path = OUT_DIR / "phase_b_summary_for_upload.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("\n".join(lines[-20:]))
    print("")
    print("DONE")
    print("Summary:", summary_path)
    print("Manifest:", OUT_DIR / "phase_b_manifest.json")


if __name__ == "__main__":
    main()
