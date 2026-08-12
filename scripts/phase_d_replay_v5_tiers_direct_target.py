from __future__ import annotations

import argparse
import json
import os
import sqlite3
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path.cwd()

DEFAULT_V5_DIR = ROOT / "data" / "training" / "manual_verified_results" / "phase_b_model_cuts_v5"
DEFAULT_SELECTED_TRADES = DEFAULT_V5_DIR / "phase_b_v5_audited_selected_trades.csv"
DEFAULT_DB = ROOT / "data" / "trader.db"

OUT_DIR = ROOT / "data" / "training" / "manual_verified_results" / "phase_d_exit_target_audit_v1"

TP_DEFAULT = 2.0308
SL_VALUES_DEFAULT = [0.75, 0.80]
ROUND_TRIP_FEE = 0.0308

HORIZON_MINUTES = {
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "8h": 480,
    "24h": 1440,
}


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
    exact = set(df.columns)
    lower_map = {c.lower(): c for c in df.columns}

    for c in candidates:
        if c in exact:
            return c
        if c.lower() in lower_map:
            return lower_map[c.lower()]

    if required:
        raise SystemExit(
            "Missing required column. Tried: "
            + ", ".join(candidates)
            + "\nActual columns:\n"
            + "\n".join(df.columns)
        )

    return None


def as_bool_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)

    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).astype(float).ne(0)

    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "yes", "y", "t"])
    )


def derive_combo_type(df: pd.DataFrame) -> pd.Series:
    combo_col = pick_col(df, ["combo_type", "combo", "model_combo", "vote_combo"], required=False)
    if combo_col:
        return df[combo_col].astype(str).str.strip()

    tab_col = pick_col(df, ["in_TAB", "in_tab", "tab_in", "selected_by_TAB"], required=False)
    xgb_col = pick_col(df, ["in_XGB", "in_xgb", "xgb_in", "selected_by_XGB"], required=False)
    rf_col = pick_col(df, ["in_RF", "in_rf", "rf_in", "selected_by_RF"], required=False)

    if not (tab_col and xgb_col and rf_col):
        raise SystemExit(
            "Cannot derive combo_type. Need combo_type or in_TAB/in_XGB/in_RF columns.\n"
            "Actual columns:\n" + "\n".join(df.columns)
        )

    tab = as_bool_series(df[tab_col])
    xgb = as_bool_series(df[xgb_col])
    rf = as_bool_series(df[rf_col])

    out = np.full(len(df), "OTHER", dtype=object)
    out[tab & xgb & rf] = "TAB_XGB_RF_ALL3"
    out[tab & xgb & ~rf] = "TAB_XGB_ONLY"
    out[tab & ~xgb & rf] = "TAB_RF_ONLY"
    out[~tab & xgb & rf] = "XGB_RF_ONLY"

    return pd.Series(out, index=df.index)


def load_selected_trades(path: Path) -> tuple[pd.DataFrame, str, str, str]:
    if not path.exists():
        raise SystemExit(f"Missing selected trades file: {path}")

    df = pd.read_csv(path)

    pair_col = pick_col(df, ["pair_address", "pairAddress", "pair", "pair_id"])
    time_col = pick_col(df, ["event_timestamp", "timestamp", "candidate_timestamp", "entry_timestamp", "ts"])
    horizon_col = pick_col(df, ["horizon", "target_horizon", "policy_horizon"])

    df[pair_col] = df[pair_col].astype(str).str.strip()
    df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    df[horizon_col] = df[horizon_col].astype(str).str.strip()

    df = df[
        df[pair_col].str.len().gt(0)
        & df[time_col].notna()
        & df[horizon_col].isin(HORIZON_MINUTES.keys())
    ].copy()

    if df.empty:
        raise SystemExit("Selected trades loaded but no usable rows remained after pair/timestamp/horizon filtering.")

    df["combo_type_direct_audit"] = derive_combo_type(df)

    print("Loaded selected trades:", len(df))
    print("Unique pairs:", df[pair_col].nunique())
    print("Horizons:", sorted(df[horizon_col].unique()))

    return df, pair_col, time_col, horizon_col


def load_snapshot_groups(db_path: Path, pairs: list[str]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not db_path.exists():
        raise SystemExit(f"Missing SQLite DB: {db_path}")

    print("Loading market_snapshots read-only:", db_path)
    print("Pairs requested:", len(pairs))

    chunks = [pairs[i:i + 700] for i in range(0, len(pairs), 700)]
    frames = []

    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        for chunk in chunks:
            placeholders = ",".join(["?"] * len(chunk))
            q = f"""
                SELECT pair_address, timestamp, price
                FROM market_snapshots
                WHERE pair_address IN ({placeholders})
                  AND timestamp IS NOT NULL
                  AND price IS NOT NULL
                  AND price > 0
            """
            frames.append(pd.read_sql_query(q, conn, params=chunk))
    finally:
        conn.close()

    if not frames:
        raise SystemExit("No snapshot query frames created.")

    snap = pd.concat(frames, ignore_index=True)
    if snap.empty:
        raise SystemExit("No market_snapshots found for selected trade pairs.")

    snap["pair_address"] = snap["pair_address"].astype(str).str.strip()
    snap["ts"] = pd.to_datetime(snap["timestamp"], utc=True, errors="coerce")
    snap["price"] = pd.to_numeric(snap["price"], errors="coerce")

    snap = snap[
        snap["pair_address"].str.len().gt(0)
        & snap["ts"].notna()
        & snap["price"].gt(0)
    ].copy()

    snap = snap.sort_values(["pair_address", "ts"]).reset_index(drop=True)

    groups: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for pair, g in snap.groupby("pair_address", sort=False):
        ts_ns = g["ts"].array.asi8.astype(np.int64)
        prices = g["price"].to_numpy(dtype=float)

        ok = np.isfinite(ts_ns) & np.isfinite(prices) & (prices > 0)
        ts_ns = ts_ns[ok]
        prices = prices[ok]

        if len(ts_ns) >= 2:
            order = np.argsort(ts_ns)
            groups[pair] = (ts_ns[order], prices[order])

    print("Snapshot rows loaded:", f"{len(snap):,}")
    print("Snapshot pairs usable:", f"{len(groups):,}")

    return groups


def simulate_exit(
    groups: dict[str, tuple[np.ndarray, np.ndarray]],
    pair: str,
    event_ns: int,
    horizon: str,
    tp_ratio: float,
    sl_ratio: float,
    fee: float,
) -> dict:
    if pair not in groups:
        return {
            "exit_valid": False,
            "exit_invalid_status": "NO_PAIR_SNAPSHOTS",
        }

    ts, prices = groups[pair]
    entry_idx = np.searchsorted(ts, event_ns, side="right") - 1

    if entry_idx < 0 or entry_idx >= len(ts):
        return {
            "exit_valid": False,
            "exit_invalid_status": "NO_ENTRY_SNAPSHOT",
        }

    entry_ts = int(ts[entry_idx])
    entry_price = float(prices[entry_idx])

    if not np.isfinite(entry_price) or entry_price <= 0:
        return {
            "exit_valid": False,
            "exit_invalid_status": "BAD_ENTRY_PRICE",
        }

    horizon_min = HORIZON_MINUTES[horizon]
    end_ns = entry_ts + int(pd.Timedelta(minutes=horizon_min).value)

    start = entry_idx + 1
    end = np.searchsorted(ts, end_ns, side="right")

    if end <= start:
        return {
            "exit_valid": False,
            "exit_invalid_status": "NO_FUTURE_SNAPSHOT",
            "entry_price": entry_price,
        }

    fut_ts = ts[start:end]
    fut_price = prices[start:end]

    ok = np.isfinite(fut_price) & (fut_price > 0)
    if not ok.any():
        return {
            "exit_valid": False,
            "exit_invalid_status": "NO_GOOD_FUTURE_PRICE",
            "entry_price": entry_price,
        }

    fut_ts = fut_ts[ok]
    fut_price = fut_price[ok]
    ratios = fut_price / entry_price

    exit_status = "TIME"
    exit_ratio = float(ratios[-1])
    exit_ts = int(fut_ts[-1])

    for ft, ratio in zip(fut_ts, ratios):
        ratio = float(ratio)

        if ratio >= tp_ratio:
            exit_status = "TP"
            exit_ratio = float(tp_ratio)
            exit_ts = int(ft)
            break

        if ratio <= sl_ratio:
            exit_status = "SL"
            exit_ratio = float(sl_ratio)
            exit_ts = int(ft)
            break

    net_return = float((exit_ratio - 1.0) - fee)

    return {
        "exit_valid": True,
        "exit_invalid_status": "",
        "entry_price": entry_price,
        "exit_status_direct": exit_status,
        "exit_ratio_direct": exit_ratio,
        "exit_net_return_direct": net_return,
        "target_net_profitable_after_exit": int(net_return > 0),
        "exit_minutes_direct": float((exit_ts - entry_ts) / 60_000_000_000),
        "max_ratio_direct": float(np.max(ratios)),
        "min_ratio_direct": float(np.min(ratios)),
    }


def replay_targets(
    trades: pd.DataFrame,
    pair_col: str,
    time_col: str,
    horizon_col: str,
    groups: dict[str, tuple[np.ndarray, np.ndarray]],
    tp_ratio: float,
    sl_values: list[float],
    fee: float,
) -> pd.DataFrame:
    records = []

    for sl_ratio in sl_values:
        label_policy = f"tp{str(tp_ratio).replace('.', '')}_sl{str(sl_ratio).replace('.', '')}"

        print("Replaying direct target:", label_policy)

        for row in trades.itertuples(index=False):
            row_dict = row._asdict()

            pair = str(row_dict[pair_col]).strip()
            ts = pd.Timestamp(row_dict[time_col])
            horizon = str(row_dict[horizon_col]).strip()

            sim = simulate_exit(
                groups=groups,
                pair=pair,
                event_ns=int(ts.value),
                horizon=horizon,
                tp_ratio=tp_ratio,
                sl_ratio=sl_ratio,
                fee=fee,
            )

            rec = dict(row_dict)
            rec["direct_label_policy"] = label_policy
            rec["direct_tp_ratio"] = tp_ratio
            rec["direct_sl_ratio"] = sl_ratio
            rec["direct_fee"] = fee
            rec.update(sim)
            records.append(rec)

    out = pd.DataFrame(records)
    return out


def summarize_group(g: pd.DataFrame, pair_col: str) -> pd.Series:
    selected = len(g)
    valid = g[g["exit_valid"] == True].copy()
    valid_n = len(valid)

    result = {
        "selected_rows": int(selected),
        "valid_sims": int(valid_n),
        "valid_sim_rate": float(valid_n / selected) if selected else 0.0,
    }

    if valid_n == 0:
        result.update({
            "target_precision_direct": np.nan,
            "net_win_rate_direct": np.nan,
            "avg_net_return_direct": np.nan,
            "median_net_return_direct": np.nan,
            "total_net_return_direct": np.nan,
            "tp_count_direct": 0,
            "sl_count_direct": 0,
            "time_count_direct": 0,
            "avg_exit_minutes_direct": np.nan,
            "unique_pairs": 0,
            "top_pair_share": np.nan,
        })
        return pd.Series(result)

    statuses = valid["exit_status_direct"].value_counts().to_dict()
    pairs = valid[pair_col].value_counts(normalize=True)

    result.update({
        "target_precision_direct": float(valid["target_net_profitable_after_exit"].mean()),
        "net_win_rate_direct": float((valid["exit_net_return_direct"] > 0).mean()),
        "avg_net_return_direct": float(valid["exit_net_return_direct"].mean()),
        "median_net_return_direct": float(valid["exit_net_return_direct"].median()),
        "total_net_return_direct": float(valid["exit_net_return_direct"].sum()),
        "tp_count_direct": int(statuses.get("TP", 0)),
        "sl_count_direct": int(statuses.get("SL", 0)),
        "time_count_direct": int(statuses.get("TIME", 0)),
        "avg_exit_minutes_direct": float(valid["exit_minutes_direct"].mean()),
        "unique_pairs": int(valid[pair_col].nunique()),
        "top_pair_share": float(pairs.iloc[0]) if len(pairs) else np.nan,
    })

    return pd.Series(result)


def existing_cols(df: pd.DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def build_summaries(replayed: pd.DataFrame, pair_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_cols = existing_cols(
        replayed,
        [
            "direct_label_policy",
            "strategy",
            "selection_mode",
            "mode",
            "policy_mode",
            "filter",
            "horizon",
            "top_pct",
            "pair_cap",
            "tp_ratio",
            "sl_ratio",
        ],
    )

    combo_cols = base_cols + ["combo_type_direct_audit"]

    combo_summary = (
        replayed
        .groupby(combo_cols, dropna=False)
        .apply(lambda g: summarize_group(g, pair_col), include_groups=False)
        .reset_index()
        .sort_values(
            ["direct_label_policy", "total_net_return_direct", "avg_net_return_direct"],
            ascending=[True, False, False],
        )
    )

    policy_mode_summary = (
        replayed
        .groupby(base_cols, dropna=False)
        .apply(lambda g: summarize_group(g, pair_col), include_groups=False)
        .reset_index()
        .sort_values(
            ["direct_label_policy", "total_net_return_direct", "avg_net_return_direct"],
            ascending=[True, False, False],
        )
    )

    tier_summary = (
        replayed
        .groupby(["direct_label_policy", "combo_type_direct_audit"], dropna=False)
        .apply(lambda g: summarize_group(g, pair_col), include_groups=False)
        .reset_index()
        .sort_values(
            ["direct_label_policy", "total_net_return_direct", "avg_net_return_direct"],
            ascending=[True, False, False],
        )
    )

    return combo_summary, policy_mode_summary, tier_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selected-trades", type=Path, default=DEFAULT_SELECTED_TRADES)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--tp", type=float, default=TP_DEFAULT)
    parser.add_argument("--sl-values", default=",".join(str(x) for x in SL_VALUES_DEFAULT))
    parser.add_argument("--fee", type=float, default=ROUND_TRIP_FEE)
    args = parser.parse_args()

    sl_values = [float(x.strip()) for x in str(args.sl_values).split(",") if x.strip()]

    print("=" * 80)
    print("PHASE D1 — replay Phase B V5.1 selected trades against direct exit target")
    print("=" * 80)
    print("selected_trades:", args.selected_trades)
    print("db:", args.db)
    print("tp:", args.tp)
    print("sl_values:", sl_values)
    print("fee:", args.fee)

    trades, pair_col, time_col, horizon_col = load_selected_trades(args.selected_trades)

    pairs = sorted(trades[pair_col].astype(str).str.strip().unique())
    groups = load_snapshot_groups(args.db, pairs)

    replayed = replay_targets(
        trades=trades,
        pair_col=pair_col,
        time_col=time_col,
        horizon_col=horizon_col,
        groups=groups,
        tp_ratio=args.tp,
        sl_values=sl_values,
        fee=args.fee,
    )

    combo_summary, policy_mode_summary, tier_summary = build_summaries(replayed, pair_col)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    replayed_path = OUT_DIR / "phase_d_v5_selected_trades_replayed_direct_targets.csv"
    combo_path = OUT_DIR / "phase_d_v5_combo_direct_target_summary.csv"
    policy_path = OUT_DIR / "phase_d_v5_policy_mode_direct_target_summary.csv"
    tier_path = OUT_DIR / "phase_d_v5_tier_direct_target_summary.csv"
    manifest_path = OUT_DIR / "phase_d_v5_direct_target_manifest.json"

    atomic_write_csv(replayed, replayed_path)
    atomic_write_csv(combo_summary, combo_path)
    atomic_write_csv(policy_mode_summary, policy_path)
    atomic_write_csv(tier_summary, tier_path)

    manifest = {
        "status": "ok",
        "phase": "D1",
        "purpose": "Replay Phase B V5.1 selected trades against direct net-profitable-after-exit-policy targets.",
        "selected_trades": str(args.selected_trades),
        "db": str(args.db),
        "output_dir": str(OUT_DIR),
        "tp": args.tp,
        "sl_values": sl_values,
        "fee": args.fee,
        "input_rows": int(len(trades)),
        "replayed_rows": int(len(replayed)),
        "pair_col": pair_col,
        "time_col": time_col,
        "horizon_col": horizon_col,
        "snapshot_pairs_loaded": int(len(groups)),
        "outputs": {
            "replayed_selected_trades": str(replayed_path),
            "combo_summary": str(combo_path),
            "policy_mode_summary": str(policy_path),
            "tier_summary": str(tier_path),
            "manifest": str(manifest_path),
        },
    }

    atomic_write_json(manifest, manifest_path)

    print("\nWROTE:")
    print(" ", replayed_path)
    print(" ", combo_path)
    print(" ", policy_path)
    print(" ", tier_path)
    print(" ", manifest_path)

    print("\nTOP TIER SUMMARY:")
    show_cols = [
        "direct_label_policy",
        "combo_type_direct_audit",
        "selected_rows",
        "valid_sims",
        "valid_sim_rate",
        "target_precision_direct",
        "net_win_rate_direct",
        "avg_net_return_direct",
        "total_net_return_direct",
        "tp_count_direct",
        "sl_count_direct",
        "time_count_direct",
        "unique_pairs",
        "top_pair_share",
    ]

    print(tier_summary[show_cols].head(40).to_string(index=False))

    print("\nDONE")


if __name__ == "__main__":
    main()
