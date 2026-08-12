from __future__ import annotations

from pathlib import Path
import json
import sqlite3
import numpy as np
import pandas as pd


ROOT = Path(".").resolve()

OUT_DIR = ROOT / "data" / "training" / "manual_verified_results" / "phase_b_model_cuts_v3"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DB = ROOT / "data" / "trader.db"

XGB_DIR = ROOT / "data" / "training" / "manual_verified_results" / "xgb_clean_full"
PHASE_B_V2_DIR = ROOT / "data" / "training" / "manual_verified_results" / "phase_b_model_cuts_v2"

CANDIDATE_PREDICTION_ROOTS = [
    ROOT / "data" / "training" / "manual_verified_results",
    ROOT / "data" / "training" / "models",
    ROOT / "data" / "training",
]

HORIZON_MINUTES = {
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "8h": 480,
    "24h": 1440,
}

ROUND_TRIP_FEE = 0.0308


def find_file(name: str) -> Path | None:
    hits = []

    for root in CANDIDATE_PREDICTION_ROOTS:
        if root.exists():
            hits.extend(root.rglob(name))

    if not hits:
        return None

    return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def find_prediction_file(model: str, split: str, filter_name: str, horizon: str) -> Path | None:
    model = model.upper()
    split = split.lower()

    exact_candidates = []

    if model == "XGB":
        exact_candidates.extend([
            XGB_DIR / f"xgb_predictions_{split}_CLEAN_{filter_name}_x2_{horizon}_XGB.parquet",
        ])

    # General glob search. Supports prior branch naming differences.
    patterns = [
        f"*{split}*{filter_name}*{horizon}*{model}*.parquet",
        f"*{model}*{split}*{filter_name}*{horizon}*.parquet",
        f"*predictions*{split}*{filter_name}*{horizon}*.parquet",
    ]

    for p in exact_candidates:
        if p.exists():
            return p

    hits = []

    for root in CANDIDATE_PREDICTION_ROOTS:
        if not root.exists():
            continue

        for pat in patterns:
            hits.extend(root.rglob(pat))

    # Filter likely paths.
    filtered = []

    for p in hits:
        name = p.name.lower()
        parent = str(p.parent).lower()

        if split not in name:
            continue

        if filter_name.lower() not in name:
            continue

        if horizon.lower() not in name:
            continue

        if model.lower() == "xgb" and "xgb" not in name:
            continue

        if model.lower() == "rf":
            if not ("rf" in name or "random_forest" in name or "predictions_" in name):
                continue

        if model.lower() == "tab":
            if not ("tab" in name or "tabicl" in name):
                continue

        filtered.append(p)

    if not filtered:
        return None

    filtered = sorted(filtered, key=lambda p: p.stat().st_mtime, reverse=True)
    return filtered[0]


def normalize_prediction_frame(df: pd.DataFrame, model: str, split: str, filter_name: str, horizon: str) -> pd.DataFrame:
    df = df.copy()

    # Normalize timestamp.
    ts_col = None
    for c in ["event_timestamp", "timestamp", "ts", "snapshot_timestamp"]:
        if c in df.columns:
            ts_col = c
            break

    if ts_col is None:
        raise ValueError(f"{model} {split} {filter_name} {horizon}: no timestamp column found. columns={list(df.columns)}")

    pair_col = None
    for c in ["pair_address", "pairAddress", "pair", "contract_address"]:
        if c in df.columns:
            pair_col = c
            break

    if pair_col is None:
        raise ValueError(f"{model} {split} {filter_name} {horizon}: no pair column found. columns={list(df.columns)}")

    score_col = None
    for c in [
        "predicted_probability",
        "probability",
        "score",
        "prediction_score",
        "prob_positive",
        "proba",
        "y_score",
    ]:
        if c in df.columns:
            score_col = c
            break

    # Fallback: find likely probability column.
    if score_col is None:
        for c in df.columns:
            lc = c.lower()
            if ("prob" in lc or "score" in lc) and pd.api.types.is_numeric_dtype(df[c]):
                score_col = c
                break

    if score_col is None:
        raise ValueError(f"{model} {split} {filter_name} {horizon}: no score/probability column found. columns={list(df.columns)}")

    y_col = None
    for c in ["y_true", "target", "label"]:
        if c in df.columns:
            y_col = c
            break

    out = pd.DataFrame({
        "event_timestamp": pd.to_datetime(df[ts_col], utc=True, errors="coerce"),
        "pair_address": df[pair_col].astype(str).str.strip(),
        f"score_{model}": pd.to_numeric(df[score_col], errors="coerce"),
    })

    if y_col is not None:
        out["y_true"] = pd.to_numeric(df[y_col], errors="coerce").fillna(0).astype(int)
    else:
        out["y_true"] = 0

    out = out[
        out["event_timestamp"].notna()
        & out["pair_address"].str.len().gt(0)
        & out[f"score_{model}"].notna()
    ].copy()

    out["event_ns"] = out["event_timestamp"].array.asi8.astype(np.int64)

    # De-dupe exact same event if needed.
    out = (
        out.sort_values(["pair_address", "event_timestamp", f"score_{model}"], ascending=[True, True, False])
        .drop_duplicates(["pair_address", "event_ns"], keep="first")
        .reset_index(drop=True)
    )

    return out


def load_model_predictions(model: str, split: str, filter_name: str, horizon: str) -> tuple[pd.DataFrame, Path]:
    path = find_prediction_file(model, split, filter_name, horizon)

    if path is None:
        raise FileNotFoundError(
            f"Could not find prediction file for model={model}, split={split}, filter={filter_name}, horizon={horizon}"
        )

    df = pd.read_parquet(path)
    return normalize_prediction_frame(df, model, split, filter_name, horizon), path


def select_top_with_pair_cap(df: pd.DataFrame, score_col: str, k: int, pair_cap):
    ranked = df.sort_values(score_col, ascending=False).reset_index(drop=True)

    if str(pair_cap).lower() == "none":
        return ranked.head(k).copy()

    pair_cap = int(float(pair_cap))

    selected_idx = []
    counts = {}

    for idx, row in ranked.iterrows():
        pair = row["pair_address"]
        n = counts.get(pair, 0)

        if n >= pair_cap:
            continue

        selected_idx.append(idx)
        counts[pair] = n + 1

        if len(selected_idx) >= k:
            break

    return ranked.loc[selected_idx].copy()


def load_snapshot_groups():
    if not DB.exists():
        raise FileNotFoundError(f"Missing SQLite DB: {DB}")

    conn = sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)
    try:
        snap = pd.read_sql_query(
            """
            SELECT pair_address, timestamp, price
            FROM market_snapshots
            WHERE pair_address IS NOT NULL
              AND timestamp IS NOT NULL
              AND price IS NOT NULL
              AND price > 0
            """,
            conn,
        )
    finally:
        conn.close()

    snap["pair_address"] = snap["pair_address"].astype(str).str.strip()
    snap["ts"] = pd.to_datetime(snap["timestamp"], utc=True, errors="coerce")
    snap["price"] = pd.to_numeric(snap["price"], errors="coerce")

    snap = snap[
        snap["pair_address"].str.len().gt(0)
        & snap["ts"].notna()
        & snap["price"].gt(0)
    ].copy()

    snap = snap.sort_values(["pair_address", "ts"]).reset_index(drop=True)

    groups = {}

    for pair, g in snap.groupby("pair_address", sort=False):
        ts_ns = g["ts"].array.asi8.astype(np.int64)
        price = g["price"].to_numpy(dtype=float)

        ok = np.isfinite(ts_ns) & np.isfinite(price) & (price > 0)
        ts_ns = ts_ns[ok]
        price = price[ok]

        if len(ts_ns) >= 2:
            order = np.argsort(ts_ns)
            groups[pair] = (ts_ns[order], price[order])

    return groups


SNAP_GROUPS = {}


def simulate_path(pair: str, event_ns: int, horizon_min: int, tp_ratio: float, sl_ratio: float):
    if pair not in SNAP_GROUPS:
        return {"valid": False, "status": "NO_PAIR"}

    ts, price = SNAP_GROUPS[pair]

    entry_idx = np.searchsorted(ts, event_ns, side="right") - 1

    if entry_idx < 0 or entry_idx >= len(ts):
        return {"valid": False, "status": "NO_ENTRY"}

    entry_ts = int(ts[entry_idx])
    entry_price = float(price[entry_idx])

    if not np.isfinite(entry_price) or entry_price <= 0:
        return {"valid": False, "status": "BAD_ENTRY_PRICE"}

    end_ns = entry_ts + int(pd.Timedelta(minutes=horizon_min).value)

    start = entry_idx + 1
    end = np.searchsorted(ts, end_ns, side="right")

    if end <= start:
        return {"valid": False, "status": "NO_FUTURE"}

    future_ts = ts[start:end]
    future_price = price[start:end]

    ok = np.isfinite(future_price) & (future_price > 0)

    if not ok.any():
        return {"valid": False, "status": "NO_GOOD_FUTURE"}

    future_ts = future_ts[ok]
    future_price = future_price[ok]
    ratios = future_price / entry_price

    exit_ratio = float(ratios[-1])
    exit_status = "TIME"
    exit_ts = int(future_ts[-1])

    for ft, ratio in zip(future_ts, ratios):
        ratio = float(ratio)

        if ratio >= tp_ratio:
            exit_ratio = float(tp_ratio)
            exit_status = "TP"
            exit_ts = int(ft)
            break

        if ratio <= sl_ratio:
            exit_ratio = float(sl_ratio)
            exit_status = "SL"
            exit_ts = int(ft)
            break

    return {
        "valid": True,
        "status": exit_status,
        "exit_ratio": exit_ratio,
        "net_return": float((exit_ratio - 1.0) - ROUND_TRIP_FEE),
        "minutes_to_exit": float((exit_ts - entry_ts) / 60_000_000_000),
        "max_ratio": float(np.max(ratios)),
        "min_ratio": float(np.min(ratios)),
    }


def classify_combo(row) -> str:
    tab = bool(row["in_TAB"])
    xgb = bool(row["in_XGB"])
    rf = bool(row["in_RF"])

    if tab and xgb and rf:
        return "TAB_XGB_RF_ALL3"
    if tab and xgb:
        return "TAB_XGB_ONLY"
    if tab and rf:
        return "TAB_RF_ONLY"
    if xgb and rf:
        return "XGB_RF_ONLY"

    return "NOT_TWO_OF_THREE"


def reconstruct_two_of_three_policy(policy: pd.Series, split: str = "test") -> tuple[pd.DataFrame, dict]:
    filter_name = str(policy["filter"])
    horizon = str(policy["horizon"])
    top_pct = float(policy["top_pct"])
    pair_cap = policy["pair_cap_str"]
    tp_ratio = float(policy["tp_ratio"])
    sl_ratio = float(policy["sl_ratio"])

    frames = {}
    paths = {}

    for model in ["TAB", "XGB", "RF"]:
        df, path = load_model_predictions(model, split, filter_name, horizon)
        frames[model] = df
        paths[model] = str(path)

    # Use union of all rows as common universe.
    base = frames["XGB"][["pair_address", "event_timestamp", "event_ns", "y_true", "score_XGB"]].copy()

    for model in ["TAB", "RF"]:
        base = base.merge(
            frames[model][["pair_address", "event_ns", f"score_{model}"]],
            on=["pair_address", "event_ns"],
            how="outer",
        )

    # Fill y_true/event_timestamp from non-XGB frames if outer rows came from TAB/RF.
    # For current clean runs, this should usually not be needed.
    base = base.merge(
        frames["TAB"][["pair_address", "event_ns", "event_timestamp", "y_true"]].rename(
            columns={"event_timestamp": "event_timestamp_TAB", "y_true": "y_true_TAB"}
        ),
        on=["pair_address", "event_ns"],
        how="left",
    )
    base = base.merge(
        frames["RF"][["pair_address", "event_ns", "event_timestamp", "y_true"]].rename(
            columns={"event_timestamp": "event_timestamp_RF", "y_true": "y_true_RF"}
        ),
        on=["pair_address", "event_ns"],
        how="left",
    )

    if "event_timestamp" not in base.columns:
        base["event_timestamp"] = pd.NaT

    base["event_timestamp"] = base["event_timestamp"].fillna(base["event_timestamp_TAB"]).fillna(base["event_timestamp_RF"])

    if "y_true" not in base.columns:
        base["y_true"] = 0

    base["y_true"] = base["y_true"].fillna(base["y_true_TAB"]).fillna(base["y_true_RF"]).fillna(0).astype(int)

    # K is per model top-pct from its universe for the same filter/horizon/split.
    k_by_model = {}
    selected_keys_by_model = {}

    for model in ["TAB", "XGB", "RF"]:
        model_df = frames[model].copy()
        k = max(1, int(len(model_df) * top_pct))
        k_by_model[model] = int(k)

        selected = select_top_with_pair_cap(
            model_df,
            score_col=f"score_{model}",
            k=k,
            pair_cap=pair_cap,
        )

        keys = set(zip(selected["pair_address"], selected["event_ns"]))
        selected_keys_by_model[model] = keys

    base["in_TAB"] = list(zip(base["pair_address"], base["event_ns"]))
    base["in_TAB"] = base["in_TAB"].apply(lambda x: x in selected_keys_by_model["TAB"])

    base["in_XGB"] = list(zip(base["pair_address"], base["event_ns"]))
    base["in_XGB"] = base["in_XGB"].apply(lambda x: x in selected_keys_by_model["XGB"])

    base["in_RF"] = list(zip(base["pair_address"], base["event_ns"]))
    base["in_RF"] = base["in_RF"].apply(lambda x: x in selected_keys_by_model["RF"])

    base["model_vote_count"] = base[["in_TAB", "in_XGB", "in_RF"]].sum(axis=1)

    selected = base[base["model_vote_count"] >= 2].copy()
    selected["combo"] = selected.apply(classify_combo, axis=1)

    # Apply pair cap again to the consensus ranking.
    # Ranking score: vote_count first, then average normalized score rank proxy.
    for m in ["TAB", "XGB", "RF"]:
        c = f"score_{m}"
        selected[c] = pd.to_numeric(selected[c], errors="coerce")

    selected["score_mean"] = selected[["score_TAB", "score_XGB", "score_RF"]].mean(axis=1, skipna=True)
    selected = selected.sort_values(["model_vote_count", "score_mean"], ascending=[False, False]).reset_index(drop=True)

    if str(pair_cap).lower() != "none":
        pair_cap_int = int(float(pair_cap))
        keep_idx = []
        counts = {}

        for idx, row in selected.iterrows():
            pair = row["pair_address"]
            n = counts.get(pair, 0)

            if n >= pair_cap_int:
                continue

            keep_idx.append(idx)
            counts[pair] = n + 1

            # Match the selected_test from the policy if present.
            if "selected_test" in policy.index and len(keep_idx) >= int(policy["selected_test"]):
                break

        selected = selected.loc[keep_idx].copy().reset_index(drop=True)
    else:
        if "selected_test" in policy.index:
            selected = selected.head(int(policy["selected_test"])).copy()

    # Simulate returns for composition breakdown.
    horizon_min = HORIZON_MINUTES[horizon]
    sims = []

    for row in selected.itertuples(index=False):
        sim = simulate_path(
            pair=row.pair_address,
            event_ns=int(row.event_ns),
            horizon_min=horizon_min,
            tp_ratio=tp_ratio,
            sl_ratio=sl_ratio,
        )
        sims.append(sim)

    sim_df = pd.DataFrame(sims)
    selected = pd.concat([selected.reset_index(drop=True), sim_df.reset_index(drop=True)], axis=1)

    metadata = {
        "filter": filter_name,
        "horizon": horizon,
        "top_pct": top_pct,
        "pair_cap": pair_cap,
        "tp_ratio": tp_ratio,
        "sl_ratio": sl_ratio,
        "split": split,
        "paths": paths,
        "k_by_model": k_by_model,
    }

    return selected, metadata


def summarize_composition(selected: pd.DataFrame, policy: pd.Series, metadata: dict) -> list[dict]:
    rows = []

    for combo, g in selected.groupby("combo", dropna=False):
        valid = g[g["valid"] == True].copy()

        if len(g) == 0:
            continue

        if len(valid) > 0:
            total_net = float(valid["net_return"].sum())
            avg_net = float(valid["net_return"].mean())
            precision = float(valid["y_true"].mean())
            net_win = float((valid["net_return"] > 0).mean())
            tp = int((valid["status"] == "TP").sum())
            sl = int((valid["status"] == "SL").sum())
            time_count = int((valid["status"] == "TIME").sum())
            unique_pairs = int(valid["pair_address"].nunique())
            top_pair_share = float(valid["pair_address"].value_counts(normalize=True).iloc[0])
        else:
            total_net = np.nan
            avg_net = np.nan
            precision = np.nan
            net_win = np.nan
            tp = sl = time_count = unique_pairs = 0
            top_pair_share = np.nan

        rows.append({
            "source_kind": policy["source_kind"],
            "strategy": policy["strategy"],
            "filter": policy["filter"],
            "horizon": policy["horizon"],
            "top_pct": policy["top_pct"],
            "pair_cap": policy["pair_cap_str"],
            "tp_ratio": policy["tp_ratio"],
            "sl_ratio": policy["sl_ratio"],
            "policy_selected_test_reported": policy.get("selected_test", np.nan),
            "policy_total_net_reported": policy.get("total_net_return_test", np.nan),
            "policy_avg_net_reported": policy.get("avg_net_return_test", np.nan),
            "combo": combo,
            "combo_selected": int(len(g)),
            "combo_share_of_policy": float(len(g) / max(len(selected), 1)),
            "combo_valid_sims": int(len(valid)),
            "combo_total_net": total_net,
            "combo_avg_net": avg_net,
            "combo_target_precision": precision,
            "combo_net_win_rate": net_win,
            "combo_tp_count": tp,
            "combo_sl_count": sl,
            "combo_time_count": time_count,
            "combo_unique_pairs": unique_pairs,
            "combo_top_pair_share": top_pair_share,
        })

    return rows


def main():
    global SNAP_GROUPS

    two_path = PHASE_B_V2_DIR / "phase_b_v2_two_of_three_ranked.csv"

    if not two_path.exists():
        raise SystemExit(f"Missing {two_path}")

    two = pd.read_csv(two_path)
    two = two[two["phase_b_robust_ok"].astype(str).str.lower().eq("true")].copy()

    if two.empty:
        raise SystemExit("No robust TWO_OF_THREE rows found.")

    SNAP_GROUPS = load_snapshot_groups()

    all_selected = []
    all_composition = []
    all_metadata = []

    for i, policy in two.reset_index(drop=True).iterrows():
        print(f"Reconstructing TWO_OF_THREE policy {i+1}/{len(two)}: "
              f"{policy['source_kind']} {policy['filter']} {policy['horizon']} "
              f"top={policy['top_pct']} cap={policy['pair_cap_str']} TP={policy['tp_ratio']} SL={policy['sl_ratio']}")

        try:
            selected, metadata = reconstruct_two_of_three_policy(policy, split="test")
        except Exception as exc:
            all_metadata.append({
                "policy_index": int(i),
                "status": "failed",
                "error": str(exc),
                "source_kind": policy.get("source_kind"),
                "filter": policy.get("filter"),
                "horizon": policy.get("horizon"),
                "top_pct": policy.get("top_pct"),
                "pair_cap": policy.get("pair_cap_str"),
                "tp_ratio": policy.get("tp_ratio"),
                "sl_ratio": policy.get("sl_ratio"),
            })
            print("FAILED:", exc)
            continue

        selected["policy_index"] = int(i)
        selected["source_kind"] = policy["source_kind"]
        selected["strategy"] = policy["strategy"]
        selected["filter"] = policy["filter"]
        selected["horizon"] = policy["horizon"]
        selected["top_pct"] = policy["top_pct"]
        selected["pair_cap"] = policy["pair_cap_str"]
        selected["tp_ratio"] = policy["tp_ratio"]
        selected["sl_ratio"] = policy["sl_ratio"]

        all_selected.append(selected)
        all_composition.extend(summarize_composition(selected, policy, metadata))

        all_metadata.append({
            "policy_index": int(i),
            "status": "ok",
            "source_kind": policy["source_kind"],
            "strategy": policy["strategy"],
            "filter": policy["filter"],
            "horizon": policy["horizon"],
            "top_pct": float(policy["top_pct"]),
            "pair_cap": str(policy["pair_cap_str"]),
            "tp_ratio": float(policy["tp_ratio"]),
            "sl_ratio": float(policy["sl_ratio"]),
            "selected_reconstructed": int(len(selected)),
            "selected_reported": int(policy["selected_test"]),
            "paths": metadata["paths"],
            "k_by_model": metadata["k_by_model"],
        })

    selected_df = pd.concat(all_selected, ignore_index=True, sort=False) if all_selected else pd.DataFrame()
    comp_df = pd.DataFrame(all_composition)
    meta_df = pd.DataFrame(all_metadata)

    selected_path = OUT_DIR / "phase_b_v3_two_of_three_selected_trade_composition.csv"
    comp_path = OUT_DIR / "phase_b_v3_two_of_three_composition_by_policy.csv"
    meta_path = OUT_DIR / "phase_b_v3_two_of_three_reconstruction_manifest.csv"

    selected_df.to_csv(selected_path, index=False)
    comp_df.to_csv(comp_path, index=False)
    meta_df.to_csv(meta_path, index=False)

    summary_lines = []
    summary_lines.append("PHASE B V3 — TWO_OF_THREE INTERNAL COMPOSITION")
    summary_lines.append("=" * 120)
    summary_lines.append("")
    summary_lines.append("This report decomposes TWO_OF_THREE into:")
    summary_lines.append("- TAB_XGB_ONLY")
    summary_lines.append("- TAB_RF_ONLY")
    summary_lines.append("- XGB_RF_ONLY")
    summary_lines.append("- TAB_XGB_RF_ALL3")
    summary_lines.append("")
    summary_lines.append("Best composition rows by combo_total_net")
    summary_lines.append("-" * 120)

    if not comp_df.empty:
        show_cols = [
            "source_kind",
            "filter",
            "horizon",
            "top_pct",
            "pair_cap",
            "tp_ratio",
            "sl_ratio",
            "policy_selected_test_reported",
            "policy_total_net_reported",
            "combo",
            "combo_selected",
            "combo_share_of_policy",
            "combo_total_net",
            "combo_avg_net",
            "combo_target_precision",
            "combo_net_win_rate",
            "combo_unique_pairs",
            "combo_top_pair_share",
        ]

        show_cols = [c for c in show_cols if c in comp_df.columns]

        summary_lines.append(
            comp_df.sort_values("combo_total_net", ascending=False)[show_cols]
            .head(80)
            .to_string(index=False)
        )
    else:
        summary_lines.append("EMPTY")

    summary_lines.append("")
    summary_lines.append("Outputs")
    summary_lines.append("-" * 120)
    summary_lines.append(str(comp_path))
    summary_lines.append(str(selected_path))
    summary_lines.append(str(meta_path))

    summary_path = OUT_DIR / "phase_b_v3_two_of_three_composition_summary_for_upload.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("DONE")
    print("Summary:", summary_path)
    print("Composition:", comp_path)
    print("Selected trades:", selected_path)
    print("Manifest:", meta_path)


if __name__ == "__main__":
    main()
