from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()

V5_DIR = ROOT / "data" / "training" / "manual_verified_results" / "phase_b_model_cuts_v5"
SELECTED_TRADES = V5_DIR / "phase_b_v5_audited_selected_trades.csv"

OUT_DIR = ROOT / "data" / "training" / "manual_verified_results" / "phase_d_exit_target_audit_v3_from_v5"

FEE_DEFAULT = 0.0308


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


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def pick_col(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    lower_to_actual = {str(c).strip().lower(): c for c in df.columns}

    for c in candidates:
        key = c.strip().lower()
        if key in lower_to_actual:
            return lower_to_actual[key]

    if required:
        raise SystemExit(
            "Missing required column. Tried:\n"
            + "\n".join(f"  - {c}" for c in candidates)
            + "\n\nActual columns:\n"
            + "\n".join(f"  - {c}" for c in df.columns)
        )

    return None


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


def normalize_combo(df: pd.DataFrame) -> pd.DataFrame:
    combo_col = pick_col(df, ["combo", "combo_type", "model_combo", "vote_combo"], required=False)

    if combo_col:
        raw = df[combo_col].astype(str).str.strip()
    else:
        raw = pd.Series("", index=df.index, dtype=object)

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
        combo = raw.copy()

    # Normalize common variants
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
        raise SystemExit(f"Missing selected trades file: {SELECTED_TRADES}")

    df = pd.read_csv(SELECTED_TRADES)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(pd.DataFrame({"column": list(df.columns)}), OUT_DIR / "phase_d_v3_input_columns.csv")

    required = [
        "pair_address",
        "event_timestamp",
        "horizon",
        "net_return",
        "exit_status",
    ]

    for c in required:
        pick_col(df, [c], required=True)

    df = normalize_combo(df)

    # Normalize known policy columns.
    alias_map = {
        "filter": ["filter", "policy_filter", "filter_name"],
        "strategy": ["strategy", "policy_strategy"],
        "top_pct": ["top_pct", "policy_top_pct"],
        "pair_cap": ["pair_cap", "policy_pair_cap"],
        "tp_ratio": ["tp_ratio", "policy_tp_ratio"],
        "sl_ratio": ["sl_ratio", "policy_sl_ratio"],
        "audit_mode": ["audit_mode", "selection_mode", "mode"],
    }

    for canonical, aliases in alias_map.items():
        found = pick_col(df, aliases, required=False)
        if found and found != canonical:
            df[canonical] = df[found]

    # Direct target from audited V5.1 net_return.
    df["net_return"] = pd.to_numeric(df["net_return"], errors="coerce")
    df["gross_ratio"] = pd.to_numeric(df.get("gross_ratio", np.nan), errors="coerce")
    df["minutes_to_exit"] = pd.to_numeric(df.get("minutes_to_exit", np.nan), errors="coerce")
    df["entry_price"] = pd.to_numeric(df.get("entry_price", np.nan), errors="coerce")
    df["max_ratio"] = pd.to_numeric(df.get("max_ratio", np.nan), errors="coerce")
    df["min_ratio"] = pd.to_numeric(df.get("min_ratio", np.nan), errors="coerce")

    df["exit_valid_phase_d"] = df["net_return"].notna()
    df["target_net_profitable_after_exit"] = (df["net_return"] > 0).astype(int)

    # Useful normalized labels.
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
    if "combo_type_phase_d" not in summary.columns:
        return summary

    decision = []

    for c in summary["combo_type_phase_d"].astype(str):
        if c == "TAB_XGB_RF_ALL3":
            decision.append("TIER_1_PRIMARY")
        elif c == "TAB_RF_ONLY":
            decision.append("TIER_2_SECONDARY")
        elif c in {"TAB_XGB_ONLY", "XGB_RF_ONLY"}:
            decision.append("REJECT_OR_RESEARCH_ONLY")
        else:
            decision.append("OTHER_REVIEW")

    summary = summary.copy()
    summary["phase_d_tier_interpretation"] = decision
    return summary


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text("Phase D v3 from V5 selected trades started.\n", OUT_DIR / "phase_d_v3_run_started.txt")

    print("=" * 80)
    print("PHASE D V3 — direct target audit from Phase B V5.1 audited selected trades")
    print("=" * 80)
    print("selected_trades:", SELECTED_TRADES)
    print("out_dir:", OUT_DIR)

    df = load_v5_selected_trades()

    # Full replayed trade-level file.
    replayed_path = OUT_DIR / "phase_d_v3_selected_trades_with_direct_target.csv"
    atomic_write_csv(df, replayed_path)

    # Main summaries.
    tier_summary = group_summary(df, ["combo_type_phase_d"])
    tier_summary = add_tier_decision(tier_summary)

    policy_tier_summary = group_summary(
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
    )
    policy_tier_summary = add_tier_decision(policy_tier_summary)

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

    # Focused summaries for the policies that Phase B update emphasized.
    focused_4h = df[
        (df.get("filter", "").astype(str) == "LIQ_5K_HIGH_ACTIVITY")
        & (df["horizon"].astype(str) == "4h")
        & (pd.to_numeric(df.get("top_pct", np.nan), errors="coerce") == 0.05)
        & (df.get("pair_cap", "").astype(str) == "50")
        & (pd.to_numeric(df.get("tp_ratio", np.nan), errors="coerce").round(4) == 2.0308)
        & (pd.to_numeric(df.get("sl_ratio", np.nan), errors="coerce").round(2) == 0.75)
    ].copy()

    focused_1h = df[
        (df.get("filter", "").astype(str) == "LIQ_5K_HIGH_ACTIVITY")
        & (df["horizon"].astype(str) == "1h")
        & (pd.to_numeric(df.get("top_pct", np.nan), errors="coerce") == 0.05)
        & (df.get("pair_cap", "").astype(str) == "50")
        & (pd.to_numeric(df.get("tp_ratio", np.nan), errors="coerce").round(4) == 2.0308)
        & (pd.to_numeric(df.get("sl_ratio", np.nan), errors="coerce").round(2) == 0.80)
    ].copy()

    focused_24h = df[
        (df.get("filter", "").astype(str) == "LIQ_5K_HIGH_ACTIVITY")
        & (df["horizon"].astype(str) == "24h")
        & (pd.to_numeric(df.get("top_pct", np.nan), errors="coerce") == 0.05)
        & (df.get("pair_cap", "").astype(str) == "50")
        & (pd.to_numeric(df.get("tp_ratio", np.nan), errors="coerce").round(4) == 2.0308)
    ].copy()

    focused_4h_summary = add_tier_decision(group_summary(focused_4h, ["combo_type_phase_d"]))
    focused_1h_summary = add_tier_decision(group_summary(focused_1h, ["combo_type_phase_d"]))
    focused_24h_summary = add_tier_decision(group_summary(focused_24h, ["combo_type_phase_d"]))

    # Write outputs.
    tier_path = OUT_DIR / "phase_d_v3_tier_summary.csv"
    policy_tier_path = OUT_DIR / "phase_d_v3_policy_tier_summary.csv"
    policy_path = OUT_DIR / "phase_d_v3_policy_summary.csv"
    focused_4h_path = OUT_DIR / "phase_d_v3_focused_4h_tier_summary.csv"
    focused_1h_path = OUT_DIR / "phase_d_v3_focused_1h_tier_summary.csv"
    focused_24h_path = OUT_DIR / "phase_d_v3_focused_24h_tier_summary.csv"
    manifest_path = OUT_DIR / "phase_d_v3_manifest.json"
    summary_txt_path = OUT_DIR / "phase_d_v3_summary_for_upload.txt"

    atomic_write_csv(tier_summary, tier_path)
    atomic_write_csv(policy_tier_summary, policy_tier_path)
    atomic_write_csv(policy_summary, policy_path)
    atomic_write_csv(focused_4h_summary, focused_4h_path)
    atomic_write_csv(focused_1h_summary, focused_1h_path)
    atomic_write_csv(focused_24h_summary, focused_24h_path)

    manifest = {
        "status": "ok",
        "phase": "D",
        "version": "v3_from_phase_b_v5_selected_trades",
        "selected_trades": str(SELECTED_TRADES),
        "output_dir": str(OUT_DIR),
        "input_rows": int(len(df)),
        "basis": "Uses audited V5.1 selected-trade net_return as direct net-profitable-after-exit-policy label.",
        "does_not_use_sqlite": True,
        "does_not_modify_db": True,
        "outputs": {
            "replayed_trades": str(replayed_path),
            "tier_summary": str(tier_path),
            "policy_tier_summary": str(policy_tier_path),
            "policy_summary": str(policy_path),
            "focused_4h_tier_summary": str(focused_4h_path),
            "focused_1h_tier_summary": str(focused_1h_path),
            "focused_24h_tier_summary": str(focused_24h_path),
            "summary_txt": str(summary_txt_path),
            "manifest": str(manifest_path),
        },
    }
    atomic_write_json(manifest, manifest_path)

    lines = []
    lines.append("PHASE D V3 — DIRECT TARGET AUDIT FROM PHASE B V5.1 SELECTED TRADES")
    lines.append("=" * 90)
    lines.append("")
    lines.append("Basis: target_net_profitable_after_exit = net_return > 0 from audited V5.1 selected trades.")
    lines.append("No SQLite replay. No model training. No DB/UI/live/demo changes.")
    lines.append("")
    lines.append("OVERALL TIER SUMMARY")
    lines.append("-" * 90)
    lines.append(tier_summary.to_string(index=False))
    lines.append("")
    lines.append("FOCUSED 4H POLICY — LIQ_5K_HIGH_ACTIVITY / top 5% / cap 50 / TP 2.0308 / SL 0.75")
    lines.append("-" * 90)
    lines.append(focused_4h_summary.to_string(index=False))
    lines.append("")
    lines.append("FOCUSED 1H POLICY — LIQ_5K_HIGH_ACTIVITY / top 5% / cap 50 / TP 2.0308 / SL 0.80")
    lines.append("-" * 90)
    lines.append(focused_1h_summary.to_string(index=False))
    lines.append("")
    lines.append("FOCUSED 24H POLICY — LIQ_5K_HIGH_ACTIVITY / top 5% / cap 50 / TP 2.0308")
    lines.append("-" * 90)
    lines.append(focused_24h_summary.to_string(index=False))
    lines.append("")

    atomic_write_text("\n".join(lines), summary_txt_path)
    atomic_write_text("Phase D v3 completed successfully.\n", OUT_DIR / "phase_d_v3_run_completed.txt")

    print("\nWROTE:")
    for p in [
        replayed_path,
        tier_path,
        policy_tier_path,
        policy_path,
        focused_4h_path,
        focused_1h_path,
        focused_24h_path,
        summary_txt_path,
        manifest_path,
    ]:
        print(" ", p)

    print("\nOVERALL TIER SUMMARY:")
    print(tier_summary.to_string(index=False))

    print("\nFOCUSED 4H SUMMARY:")
    print(focused_4h_summary.to_string(index=False))

    print("\nDONE")


if __name__ == "__main__":
    main()
