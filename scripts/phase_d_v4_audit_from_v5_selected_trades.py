from __future__ import annotations

import json
import os
import traceback
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()

V5_DIR = ROOT / "data" / "training" / "manual_verified_results" / "phase_b_model_cuts_v5"
SELECTED_TRADES = V5_DIR / "phase_b_v5_audited_selected_trades.csv"

OUT_DIR = ROOT / "data" / "training" / "manual_verified_results" / "phase_d_exit_target_audit_v4_from_v5"


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower_to_actual = {str(c).strip().lower(): c for c in df.columns}

    for c in candidates:
        key = str(c).strip().lower()
        if key in lower_to_actual:
            return lower_to_actual[key]

    if required:
        raise RuntimeError(
            "Missing required column. Tried:\n"
            + "\n".join(f"  - {c}" for c in candidates)
            + "\n\nActual columns:\n"
            + "\n".join(f"  - {c}" for c in df.columns)
        )

    return None


def copy_alias(df: pd.DataFrame, canonical: str, candidates: list[str]) -> pd.DataFrame:
    found = pick_col(df, candidates, required=False)
    if found is not None:
        df[canonical] = df[found]
    return df


def as_bool(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(float).ne(0)

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "yes", "y", "t"])
    )


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    alias_map = {
        "strategy": ["strategy", "policy_strategy", "consensus_strategy"],
        "filter": ["filter", "policy_filter", "filter_name", "dataset_filter"],
        "horizon": ["horizon", "policy_horizon", "target_horizon"],
        "top_pct": ["top_pct", "policy_top_pct", "top_percentile"],
        "pair_cap": ["pair_cap", "policy_pair_cap", "pair_cap_str"],
        "tp_ratio": ["tp_ratio", "policy_tp_ratio", "take_profit_ratio"],
        "sl_ratio": ["sl_ratio", "policy_sl_ratio", "stop_loss_ratio"],
        "audit_mode": ["audit_mode", "selection_mode", "mode", "policy_mode"],
    }

    for canonical, candidates in alias_map.items():
        df = copy_alias(df, canonical, candidates)

    return df


def normalize_combo(df: pd.DataFrame) -> pd.DataFrame:
    tab_col = pick_col(df, ["in_TAB", "in_tab"], required=False)
    xgb_col = pick_col(df, ["in_XGB", "in_xgb"], required=False)
    rf_col = pick_col(df, ["in_RF", "in_rf"], required=False)

    if tab_col and xgb_col and rf_col:
        tab = as_bool(df[tab_col])
        xgb = as_bool(df[xgb_col])
        rf = as_bool(df[rf_col])

        combo = pd.Series("OTHER", index=df.index, dtype=object)
        combo.loc[tab & xgb & rf] = "TAB_XGB_RF_ALL3"
        combo.loc[tab & xgb & ~rf] = "TAB_XGB_ONLY"
        combo.loc[tab & ~xgb & rf] = "TAB_RF_ONLY"
        combo.loc[~tab & xgb & rf] = "XGB_RF_ONLY"
    else:
        combo_col = pick_col(df, ["combo", "combo_type", "model_combo", "vote_combo"], required=False)
        if combo_col is None:
            combo = pd.Series("OTHER", index=df.index, dtype=object)
        else:
            combo = df[combo_col].astype(str).str.strip()

    combo = combo.astype(str).str.strip()
    combo = combo.replace({
        "ALL3": "TAB_XGB_RF_ALL3",
        "ALL3_INTERSECT": "TAB_XGB_RF_ALL3",
        "TAB_XGB_RF": "TAB_XGB_RF_ALL3",
        "TAB+XGB+RF": "TAB_XGB_RF_ALL3",
        "TAB_XGB": "TAB_XGB_ONLY",
        "TAB+XGB": "TAB_XGB_ONLY",
        "TAB_RF": "TAB_RF_ONLY",
        "TAB+RF": "TAB_RF_ONLY",
        "XGB_RF": "XGB_RF_ONLY",
        "XGB+RF": "XGB_RF_ONLY",
    })

    df["combo_type_phase_d"] = combo
    return df


def load_v5_selected_trades() -> pd.DataFrame:
    if not SELECTED_TRADES.exists():
        raise RuntimeError(f"Missing selected trades file: {SELECTED_TRADES}")

    df = pd.read_csv(SELECTED_TRADES)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    atomic_write_csv(
        pd.DataFrame({"column": list(df.columns)}),
        OUT_DIR / "phase_d_v4_input_columns.csv",
    )

    df = normalize_columns(df)
    df = normalize_combo(df)

    required_groups = {
        "pair_address": ["pair_address", "pairAddress", "pair"],
        "event_timestamp": ["event_timestamp", "timestamp", "candidate_timestamp", "entry_timestamp"],
        "horizon": ["horizon", "policy_horizon"],
        "net_return": ["net_return"],
        "exit_status": ["exit_status"],
    }

    for canonical, candidates in required_groups.items():
        if canonical not in df.columns:
            found = pick_col(df, candidates, required=True)
            df[canonical] = df[found]

    df["pair_address"] = df["pair_address"].astype(str).str.strip()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True, errors="coerce")
    df["horizon"] = df["horizon"].astype(str).str.strip()

    df["net_return"] = pd.to_numeric(df["net_return"], errors="coerce")
    df["gross_ratio"] = pd.to_numeric(df["gross_ratio"], errors="coerce") if "gross_ratio" in df.columns else np.nan
    df["minutes_to_exit"] = pd.to_numeric(df["minutes_to_exit"], errors="coerce") if "minutes_to_exit" in df.columns else np.nan
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors="coerce") if "entry_price" in df.columns else np.nan
    df["max_ratio"] = pd.to_numeric(df["max_ratio"], errors="coerce") if "max_ratio" in df.columns else np.nan
    df["min_ratio"] = pd.to_numeric(df["min_ratio"], errors="coerce") if "min_ratio" in df.columns else np.nan

    for c in ["top_pct", "tp_ratio", "sl_ratio"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "pair_cap" in df.columns:
        df["pair_cap"] = df["pair_cap"].astype(str).str.strip()

    before = len(df)
    df = df[
        df["pair_address"].str.len().gt(0)
        & df["event_timestamp"].notna()
        & df["horizon"].isin(["30m", "1h", "4h", "8h", "24h"])
        & df["net_return"].notna()
    ].copy()

    if df.empty:
        raise RuntimeError(f"All rows were filtered out. Rows before filtering: {before}")

    df["exit_valid_phase_d"] = True
    df["target_net_profitable_after_exit"] = (df["net_return"] > 0).astype(int)
    df["phase_d_source"] = "phase_b_v5_audited_selected_trades"
    df["direct_target_basis"] = "audited_v5_net_return"

    return df


def summarize_subset(g: pd.DataFrame) -> dict:
    selected = int(len(g))
    valid = g[g["exit_valid_phase_d"] == True].copy()
    valid_n = int(len(valid))

    result = {
        "selected_rows": selected,
        "valid_rows": valid_n,
        "valid_rate": float(valid_n / selected) if selected else 0.0,
    }

    if valid_n == 0:
        result.update({
            "target_precision_direct": np.nan,
            "net_win_rate_direct": np.nan,
            "avg_net_return_direct": np.nan,
            "median_net_return_direct": np.nan,
            "total_net_return_direct": np.nan,
            "tp_count": 0,
            "sl_count": 0,
            "time_count": 0,
            "avg_minutes_to_exit": np.nan,
            "unique_pairs": 0,
            "top_pair_share": np.nan,
        })
        return result

    pair_counts = valid["pair_address"].astype(str).value_counts(normalize=True)
    status_counts = valid["exit_status"].astype(str).str.upper().value_counts().to_dict()

    result.update({
        "target_precision_direct": float(valid["target_net_profitable_after_exit"].mean()),
        "net_win_rate_direct": float((valid["net_return"] > 0).mean()),
        "avg_net_return_direct": float(valid["net_return"].mean()),
        "median_net_return_direct": float(valid["net_return"].median()),
        "total_net_return_direct": float(valid["net_return"].sum()),
        "tp_count": int(status_counts.get("TP", 0)),
        "sl_count": int(status_counts.get("SL", 0)),
        "time_count": int(status_counts.get("TIME", 0)),
        "avg_minutes_to_exit": float(valid["minutes_to_exit"].mean()) if "minutes_to_exit" in valid.columns else np.nan,
        "unique_pairs": int(valid["pair_address"].astype(str).nunique()),
        "top_pair_share": float(pair_counts.iloc[0]) if len(pair_counts) else np.nan,
    })

    return result


def group_summary(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    group_cols = [c for c in group_cols if c in df.columns]
    rows = []

    if df.empty:
        return pd.DataFrame()

    if not group_cols:
        rows.append(summarize_subset(df))
    else:
        for keys, g in df.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_cols, keys))
            row.update(summarize_subset(g))
            rows.append(row)

    out = pd.DataFrame(rows)

    sort_cols = [c for c in ["total_net_return_direct", "avg_net_return_direct", "selected_rows"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols)).reset_index(drop=True)

    return out


def add_tier_decision(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty or "combo_type_phase_d" not in summary.columns:
        return summary

    mapping = {
        "TAB_XGB_RF_ALL3": "TIER_1_PRIMARY",
        "TAB_RF_ONLY": "TIER_2_SECONDARY",
        "TAB_XGB_ONLY": "REJECT_OR_RESEARCH_ONLY",
        "XGB_RF_ONLY": "REJECT_OR_RESEARCH_ONLY",
    }

    summary = summary.copy()
    summary["phase_d_tier_interpretation"] = (
        summary["combo_type_phase_d"].astype(str).map(mapping).fillna("OTHER_REVIEW")
    )
    return summary


def filter_eq(df: pd.DataFrame, col: str, value: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return df[col].astype(str).eq(value)


def filter_num_eq(df: pd.DataFrame, col: str, value: float, decimals: int = 4) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index)
    return pd.to_numeric(df[col], errors="coerce").round(decimals).eq(round(value, decimals))


def build_outputs(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tier_summary = add_tier_decision(group_summary(df, ["combo_type_phase_d"]))

    policy_tier_summary = add_tier_decision(group_summary(
        df,
        [
            "audit_mode",
            "policy_source_kind",
            "strategy",
            "filter",
            "horizon",
            "top_pct",
            "pair_cap",
            "tp_ratio",
            "sl_ratio",
            "combo_type_phase_d",
        ],
    ))

    policy_summary = group_summary(
        df,
        [
            "audit_mode",
            "policy_source_kind",
            "strategy",
            "filter",
            "horizon",
            "top_pct",
            "pair_cap",
            "tp_ratio",
            "sl_ratio",
        ],
    )

    focused_4h = df[
        filter_eq(df, "filter", "LIQ_5K_HIGH_ACTIVITY")
        & filter_eq(df, "horizon", "4h")
        & filter_num_eq(df, "top_pct", 0.05, 4)
        & filter_eq(df, "pair_cap", "50")
        & filter_num_eq(df, "tp_ratio", 2.0308, 4)
        & filter_num_eq(df, "sl_ratio", 0.75, 2)
    ].copy()

    focused_1h = df[
        filter_eq(df, "filter", "LIQ_5K_HIGH_ACTIVITY")
        & filter_eq(df, "horizon", "1h")
        & filter_num_eq(df, "top_pct", 0.05, 4)
        & filter_eq(df, "pair_cap", "50")
        & filter_num_eq(df, "tp_ratio", 2.0308, 4)
        & filter_num_eq(df, "sl_ratio", 0.80, 2)
    ].copy()

    focused_24h = df[
        filter_eq(df, "filter", "LIQ_5K_HIGH_ACTIVITY")
        & filter_eq(df, "horizon", "24h")
        & filter_num_eq(df, "top_pct", 0.05, 4)
        & filter_eq(df, "pair_cap", "50")
        & filter_num_eq(df, "tp_ratio", 2.0308, 4)
    ].copy()

    focused_4h_summary = add_tier_decision(group_summary(focused_4h, ["combo_type_phase_d"]))
    focused_1h_summary = add_tier_decision(group_summary(focused_1h, ["combo_type_phase_d"]))
    focused_24h_summary = add_tier_decision(group_summary(focused_24h, ["combo_type_phase_d"]))

    return {
        "tier_summary": tier_summary,
        "policy_tier_summary": policy_tier_summary,
        "policy_summary": policy_summary,
        "focused_4h_summary": focused_4h_summary,
        "focused_1h_summary": focused_1h_summary,
        "focused_24h_summary": focused_24h_summary,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text("Phase D v4 from V5 selected trades started.\n", OUT_DIR / "phase_d_v4_run_started.txt")

    print("=" * 90)
    print("PHASE D V4 — direct target audit from Phase B V5.1 selected trades")
    print("=" * 90)
    print("selected_trades:", SELECTED_TRADES)
    print("output_dir:", OUT_DIR)

    df = load_v5_selected_trades()
    outputs = build_outputs(df)

    paths = {
        "replayed_trades": OUT_DIR / "phase_d_v4_selected_trades_with_direct_target.csv",
        "tier_summary": OUT_DIR / "phase_d_v4_tier_summary.csv",
        "policy_tier_summary": OUT_DIR / "phase_d_v4_policy_tier_summary.csv",
        "policy_summary": OUT_DIR / "phase_d_v4_policy_summary.csv",
        "focused_4h_summary": OUT_DIR / "phase_d_v4_focused_4h_tier_summary.csv",
        "focused_1h_summary": OUT_DIR / "phase_d_v4_focused_1h_tier_summary.csv",
        "focused_24h_summary": OUT_DIR / "phase_d_v4_focused_24h_tier_summary.csv",
        "summary_txt": OUT_DIR / "phase_d_v4_summary_for_upload.txt",
        "manifest": OUT_DIR / "phase_d_v4_manifest.json",
    }

    atomic_write_csv(df, paths["replayed_trades"])
    atomic_write_csv(outputs["tier_summary"], paths["tier_summary"])
    atomic_write_csv(outputs["policy_tier_summary"], paths["policy_tier_summary"])
    atomic_write_csv(outputs["policy_summary"], paths["policy_summary"])
    atomic_write_csv(outputs["focused_4h_summary"], paths["focused_4h_summary"])
    atomic_write_csv(outputs["focused_1h_summary"], paths["focused_1h_summary"])
    atomic_write_csv(outputs["focused_24h_summary"], paths["focused_24h_summary"])

    lines = []
    lines.append("PHASE D V4 — DIRECT TARGET AUDIT FROM PHASE B V5.1 SELECTED TRADES")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Basis: target_net_profitable_after_exit = audited V5.1 net_return > 0.")
    lines.append("No SQLite replay. No model training. No DB/UI/live/demo changes.")
    lines.append("")
    lines.append("OVERALL TIER SUMMARY")
    lines.append("-" * 100)
    lines.append(outputs["tier_summary"].to_string(index=False))
    lines.append("")
    lines.append("FOCUSED 4H — LIQ_5K_HIGH_ACTIVITY / top 5% / cap 50 / TP 2.0308 / SL 0.75")
    lines.append("-" * 100)
    lines.append(outputs["focused_4h_summary"].to_string(index=False))
    lines.append("")
    lines.append("FOCUSED 1H — LIQ_5K_HIGH_ACTIVITY / top 5% / cap 50 / TP 2.0308 / SL 0.80")
    lines.append("-" * 100)
    lines.append(outputs["focused_1h_summary"].to_string(index=False))
    lines.append("")
    lines.append("FOCUSED 24H — LIQ_5K_HIGH_ACTIVITY / top 5% / cap 50 / TP 2.0308")
    lines.append("-" * 100)
    lines.append(outputs["focused_24h_summary"].to_string(index=False))
    lines.append("")

    atomic_write_text("\n".join(lines), paths["summary_txt"])

    manifest = {
        "status": "ok",
        "phase": "D",
        "version": "v4_from_phase_b_v5_selected_trades",
        "selected_trades": str(SELECTED_TRADES),
        "output_dir": str(OUT_DIR),
        "input_rows": int(len(df)),
        "basis": "Uses audited V5.1 selected-trade net_return as direct net-profitable-after-exit-policy label.",
        "does_not_use_sqlite": True,
        "does_not_modify_db": True,
        "outputs": {k: str(v) for k, v in paths.items()},
    }

    atomic_write_json(manifest, paths["manifest"])
    atomic_write_text("Phase D v4 completed successfully.\n", OUT_DIR / "phase_d_v4_run_completed.txt")

    print("\nWROTE:")
    for k, p in paths.items():
        print(f"  {k}: {p}")

    print("\nOVERALL TIER SUMMARY:")
    print(outputs["tier_summary"].to_string(index=False))

    print("\nFOCUSED 4H SUMMARY:")
    print(outputs["focused_4h_summary"].to_string(index=False))

    print("\nDONE")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        err = traceback.format_exc()
        atomic_write_text(err, OUT_DIR / "phase_d_v4_error.txt")
        print("\nERROR written to:")
        print(OUT_DIR / "phase_d_v4_error.txt")
        print(err)
        raise
