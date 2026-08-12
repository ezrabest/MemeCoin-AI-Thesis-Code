from __future__ import annotations

from pathlib import Path
import json
import os
import shutil
import sqlite3
import time
import uuid

import numpy as np
import pandas as pd

from xgboost import XGBClassifier


# ============================================================
# FULL XGBOOST CLEAN EVALUATION — CUDA VERSION
# Same scope as TAB/RF:
# 4 filters x 5 horizons x validation/test x same exit sweep
# ============================================================

DATASET_DIR = Path("data/training/manual_verified_datasets_clean_for_model")

XGB_OUT_DIR = Path("data/training/manual_verified_results/xgb_clean_full")
OUT_DIR = Path("data/training/manual_verified_results/exit_sim_xgb_full")

DB = Path("data/trader.db").resolve()

FILTERS = [
    "RAW_ALL_VERIFIED",
    "LIQ_5K_HIGH_ACTIVITY",
    "LOW_LIQ_MOMENTUM",
    "NO_WHALE_FILTER",
]

HORIZONS = ["30m", "1h", "4h", "8h", "24h"]

HORIZON_MINUTES = {
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "8h": 480,
    "24h": 1440,
}

ROUND_TRIP_FEE = 0.0308

TOP_PCTS = [0.005, 0.01, 0.02, 0.05]
PAIR_CAPS = [1, 2, 3, 5, 10, 20, 50, None]

TP_RATIOS = [1.20, 1.30, 1.50, 1.70, 1.85, 2.00, 2.0308]
SL_RATIOS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

USE_CUDA = True

STRICT_MIN_SELECTED = 50
STRICT_MIN_VALID_SIM_RATE = 0.95
STRICT_MIN_UNIQUE_PAIRS = 7
STRICT_MAX_TOP_PAIR_SHARE = 0.25

SNAP_GROUPS = {}


def atomic_write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        df.to_parquet(tmp, index=False)
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


def atomic_write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_json(payload, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def validate_required_inputs() -> None:
    if not DB.exists():
        raise SystemExit(f"Missing SQLite DB: {DB}")

    if not DATASET_DIR.exists():
        raise SystemExit(f"Missing clean dataset directory: {DATASET_DIR}")

    missing = []
    found = []

    for f in FILTERS:
        for h in HORIZONS:
            dataset = DATASET_DIR / f"{f}_x2_{h}_CLEAN_MODEL_INPUT.parquet"
            if dataset.exists():
                found.append(dataset)
            else:
                missing.append(dataset)

    if missing:
        print("\nMissing required CLEAN_MODEL_INPUT files:")
        for p in missing:
            print("  -", p)
        raise SystemExit("\nStop: not deleting XGB outputs because required inputs are missing.")

    print("Validated clean model input files:", len(found))


def safe_clean_xgb_outputs() -> None:
    allowed_root = Path("data/training/manual_verified_results").resolve()
    allowed_names = {"xgb_clean_full", "exit_sim_xgb_full"}

    for p in [XGB_OUT_DIR, OUT_DIR]:
        resolved = p.resolve()

        if resolved == allowed_root:
            raise SystemExit(f"Refusing to delete root results directory: {resolved}")

        if allowed_root not in resolved.parents:
            raise SystemExit(f"Refusing to delete outside manual_verified_results: {resolved}")

        if p.name not in allowed_names:
            raise SystemExit(f"Refusing to delete unexpected directory: {resolved}")

        if p.exists():
            print("Deleting XGB-only output directory:", p)
            shutil.rmtree(p)

        p.mkdir(parents=True, exist_ok=True)

    print("Clean XGB output directories recreated.")
    print("XGB_OUT_DIR:", XGB_OUT_DIR)
    print("OUT_DIR:", OUT_DIR)


def load_snapshot_groups():
    print("Loading raw snapshots read-only:", DB)

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

    print("snapshot rows:", f"{len(snap):,}")
    print("pairs:", f"{len(groups):,}")
    return groups


def select_feature_cols(df: pd.DataFrame):
    excluded = {"event_timestamp", "pair_address", "split", "target"}

    features = [
        c for c in df.columns
        if c not in excluded
        and (pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]))
    ]

    bad_tokens = [
        "target",
        "future",
        "minutes_to",
        "usable_x2",
        "split_x2",
        "max_future",
        "min_future",
        "return_",
        "optimal_trade",
        "filter_status",
        "drop_reason",
        "legacy",
    ]

    bad = [c for c in features if any(tok in c.lower() for tok in bad_tokens)]
    if bad:
        raise SystemExit(f"XGB leakage features found: {bad}")

    return features


def make_xgb_classifier(scale_pos_weight: float) -> XGBClassifier:
    model_kwargs = dict(
        n_estimators=700,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=42,
        scale_pos_weight=scale_pos_weight,
    )

    if USE_CUDA:
        model_kwargs["device"] = "cuda"

    return XGBClassifier(**model_kwargs)


def fit_xgb_or_fail_cuda(xgb: XGBClassifier, X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    try:
        xgb.fit(X_train, y_train)
        return xgb
    except Exception as exc:
        if not USE_CUDA:
            raise

        print("\nCUDA fit failed with modern XGBoost params.")
        print("Trying legacy GPU params: tree_method='gpu_hist', predictor='gpu_predictor'")

        params = xgb.get_params()
        params.pop("device", None)
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"

        xgb_legacy = XGBClassifier(**params)
        try:
            xgb_legacy.fit(X_train, y_train)
            return xgb_legacy
        except Exception as exc2:
            raise SystemExit(
                "\nXGB GPU training failed in both modern and legacy modes.\n"
                "This usually means the installed xgboost package has no CUDA support, "
                "or CUDA is unavailable to this venv.\n"
                f"First error: {exc}\n"
                f"Second error: {exc2}\n"
            ) from exc2


def positive_proba(xgb: XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    proba = xgb.predict_proba(X)
    classes = list(getattr(xgb, "classes_", []))

    if 1 in classes:
        idx = classes.index(1)
    elif True in classes:
        idx = classes.index(True)
    else:
        idx = 1

    return np.asarray(proba[:, idx], dtype=float)


def train_xgb_predictions(filter_name: str, horizon: str):
    dataset = DATASET_DIR / f"{filter_name}_x2_{horizon}_CLEAN_MODEL_INPUT.parquet"

    if not dataset.exists():
        print("MISSING DATASET:", dataset)
        return {}

    out = {
        "validation": XGB_OUT_DIR / f"xgb_predictions_validation_CLEAN_{filter_name}_x2_{horizon}_XGB.parquet",
        "test": XGB_OUT_DIR / f"xgb_predictions_test_CLEAN_{filter_name}_x2_{horizon}_XGB.parquet",
    }

    feature_path = XGB_OUT_DIR / f"xgb_features_CLEAN_{filter_name}_x2_{horizon}_XGB.json"

    for p in list(out.values()) + [feature_path]:
        p.unlink(missing_ok=True)

    print("\nTraining XGB:", filter_name, horizon)
    print("dataset:", dataset)
    print("CUDA requested:", USE_CUDA)

    df = pd.read_parquet(dataset)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True, errors="coerce")
    df["pair_address"] = df["pair_address"].astype(str).str.strip()
    df = df.sort_values("event_timestamp").reset_index(drop=True)

    train = df[df["split"] == "train"].copy()
    val = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    features = select_feature_cols(df)

    y_train = train["target"].astype(int)
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)

    if pos == 0 or neg == 0:
        print("SKIP one-class train:", filter_name, horizon, "pos:", pos, "neg:", neg)
        return {}

    scale_pos_weight = neg / max(pos, 1)

    print(
        "train rows:", f"{len(train):,}",
        "pos:", f"{pos:,}",
        "neg:", f"{neg:,}",
        "scale_pos_weight:", round(scale_pos_weight, 3),
    )
    print("validation rows:", f"{len(val):,}", "test rows:", f"{len(test):,}", "features:", len(features))

    X_train = train[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)

    xgb = make_xgb_classifier(scale_pos_weight=scale_pos_weight)
    xgb = fit_xgb_or_fail_cuda(xgb, X_train, y_train)

    atomic_write_json(features, feature_path)

    for split_name, part in [("validation", val), ("test", test)]:
        X = part[features].replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
        prob = positive_proba(xgb, X)

        pred = pd.DataFrame({
            "event_timestamp": part["event_timestamp"].values,
            "pair_address": part["pair_address"].astype(str).str.strip().values,
            "split": split_name,
            "y_true": part["target"].astype(int).values,
            "predicted_probability": prob,
            "model_name": "XGB_CLEAN_FULL_CUDA",
            "filter": filter_name,
            "horizon": horizon,
        })

        pred_path = XGB_OUT_DIR / f"xgb_predictions_{split_name}_CLEAN_{filter_name}_x2_{horizon}_XGB.parquet"
        atomic_write_parquet(pred, pred_path)
        out[split_name] = pred_path
        print("wrote:", pred_path)

    return out


def pred_path(split: str, f: str, h: str):
    return XGB_OUT_DIR / f"xgb_predictions_{split}_CLEAN_{f}_x2_{h}_XGB.parquet"


def load_predictions(path: Path, split: str, f: str, h: str):
    df = pd.read_parquet(path)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"], utc=True, errors="coerce")
    df["event_ns"] = df["event_timestamp"].array.asi8.astype(np.int64)
    df["pair_address"] = df["pair_address"].astype(str).str.strip()
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").fillna(0).astype(int)
    df["predicted_probability"] = pd.to_numeric(df["predicted_probability"], errors="coerce").fillna(0.0)

    df = df[
        df["event_timestamp"].notna()
        & df["pair_address"].str.len().gt(0)
    ].copy()

    df["model"] = "XGB"
    df["split"] = split
    df["filter"] = f
    df["horizon"] = h

    return df


def select_top_with_pair_cap(df: pd.DataFrame, k: int, pair_cap):
    ranked = df.sort_values("predicted_probability", ascending=False).reset_index(drop=True)

    if pair_cap is None:
        return ranked.head(k).copy()

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


def summarize_sim(candidates: pd.DataFrame, h: str, tp_ratio: float, sl_ratio: float):
    horizon_min = HORIZON_MINUTES[h]
    records = []

    for row in candidates.itertuples(index=False):
        sim = simulate_path(
            pair=row.pair_address,
            event_ns=int(row.event_ns),
            horizon_min=horizon_min,
            tp_ratio=tp_ratio,
            sl_ratio=sl_ratio,
        )

        sim["y_true"] = int(row.y_true)
        sim["pair_address"] = row.pair_address
        sim["score"] = float(row.predicted_probability)
        records.append(sim)

    sim_df = pd.DataFrame(records)
    selected = len(sim_df)

    if selected == 0:
        return {
            "selected": 0,
            "valid_sims": 0,
            "valid_sim_rate": 0.0,
            "target_precision": np.nan,
            "net_win_rate": np.nan,
            "avg_net_return": np.nan,
            "median_net_return": np.nan,
            "total_net_return": np.nan,
            "tp_count": 0,
            "sl_count": 0,
            "time_count": 0,
            "avg_minutes_to_exit": np.nan,
            "max_ratio_p50": np.nan,
            "max_ratio_p90": np.nan,
            "unique_pairs": 0,
            "top_pair_share": np.nan,
        }

    valid_df = sim_df[sim_df["valid"] == True].copy()
    valid = len(valid_df)
    valid_rate = valid / selected

    if valid == 0:
        return {
            "selected": int(selected),
            "valid_sims": 0,
            "valid_sim_rate": float(valid_rate),
            "target_precision": np.nan,
            "net_win_rate": np.nan,
            "avg_net_return": np.nan,
            "median_net_return": np.nan,
            "total_net_return": np.nan,
            "tp_count": 0,
            "sl_count": 0,
            "time_count": 0,
            "avg_minutes_to_exit": np.nan,
            "max_ratio_p50": np.nan,
            "max_ratio_p90": np.nan,
            "unique_pairs": 0,
            "top_pair_share": np.nan,
        }

    statuses = valid_df["status"].value_counts().to_dict()
    pairs = valid_df["pair_address"].value_counts(normalize=True)

    return {
        "selected": int(selected),
        "valid_sims": int(valid),
        "valid_sim_rate": float(valid_rate),
        "target_precision": float(valid_df["y_true"].mean()),
        "net_win_rate": float((valid_df["net_return"] > 0).mean()),
        "avg_net_return": float(valid_df["net_return"].mean()),
        "median_net_return": float(valid_df["net_return"].median()),
        "total_net_return": float(valid_df["net_return"].sum()),
        "tp_count": int(statuses.get("TP", 0)),
        "sl_count": int(statuses.get("SL", 0)),
        "time_count": int(statuses.get("TIME", 0)),
        "avg_minutes_to_exit": float(valid_df["minutes_to_exit"].mean()),
        "max_ratio_p50": float(valid_df["max_ratio"].quantile(0.50)),
        "max_ratio_p90": float(valid_df["max_ratio"].quantile(0.90)),
        "unique_pairs": int(valid_df["pair_address"].nunique()),
        "top_pair_share": float(pairs.iloc[0]) if len(pairs) else np.nan,
    }


def run_xgb_exit_sweep():
    rows = []

    for f in FILTERS:
        for h in HORIZONS:
            for split in ["validation", "test"]:
                path = pred_path(split, f, h)

                if not path.exists():
                    print("MISSING prediction:", path)
                    continue

                print(f"\nSIM XGB {split} {f} {h}")
                df = load_predictions(path, split, f, h)

                for top_pct in TOP_PCTS:
                    k = max(1, int(len(df) * top_pct))

                    for pair_cap in PAIR_CAPS:
                        selected = select_top_with_pair_cap(df, k, pair_cap)

                        for tp_ratio in TP_RATIOS:
                            for sl_ratio in SL_RATIOS:
                                m = summarize_sim(selected, h, tp_ratio, sl_ratio)

                                rows.append({
                                    "model": "XGB",
                                    "split": split,
                                    "filter": f,
                                    "horizon": h,
                                    "top_pct": top_pct,
                                    "target_k": k,
                                    "pair_cap": "none" if pair_cap is None else pair_cap,
                                    "tp_ratio": tp_ratio,
                                    "sl_ratio": sl_ratio,
                                    "round_trip_fee": ROUND_TRIP_FEE,
                                    **m,
                                })

                print("done policies:", len(TOP_PCTS) * len(PAIR_CAPS) * len(TP_RATIOS) * len(SL_RATIOS))

    sweep = pd.DataFrame(rows)
    sweep_path = OUT_DIR / "exit_policy_sweep_XGB_CLEAN_FULL.csv"
    atomic_write_csv(sweep, sweep_path)

    print("\nWROTE XGB FULL SWEEP:", sweep_path)
    print("sweep rows:", len(sweep))

    return sweep


def strict_selector(sweep: pd.DataFrame):
    sweep = sweep.copy()
    sweep["pair_cap_str"] = sweep["pair_cap"].astype(str)

    policy_cols = ["model", "filter", "horizon", "top_pct", "pair_cap_str", "tp_ratio", "sl_ratio"]

    val = sweep[sweep["split"] == "validation"].copy()
    test = sweep[sweep["split"] == "test"].copy()

    eligible = val[
        (val["valid_sim_rate"] >= STRICT_MIN_VALID_SIM_RATE)
        & (val["selected"] >= STRICT_MIN_SELECTED)
        & (val["total_net_return"].notna())
        & (val["total_net_return"] > 0)
        & (val["avg_net_return"] > 0)
        & (val["pair_cap_str"] != "none")
        & (val["unique_pairs"] >= STRICT_MIN_UNIQUE_PAIRS)
        & (val["top_pair_share"] <= STRICT_MAX_TOP_PAIR_SHARE)
    ].copy()

    print("\nSTRICT eligible XGB validation policies:", len(eligible))

    if eligible.empty:
        out_path = OUT_DIR / "strict_validation_selected_policies_XGB_CLEAN_FULL_applied_to_test.csv"
        atomic_write_csv(pd.DataFrame(), out_path)
        print("No strict XGB policies.")
        print("WROTE EMPTY XGB FULL STRICT:", out_path)
        return pd.DataFrame()

    eligible["robust_score"] = (
        eligible["avg_net_return"] * 100
        + eligible["target_precision"] * 10
        + eligible["net_win_rate"] * 5
        + np.log1p(eligible["unique_pairs"]) * 2
        - eligible["top_pair_share"] * 10
    )

    best_val = (
        eligible.sort_values(
            ["robust_score", "avg_net_return", "total_net_return", "target_precision"],
            ascending=False,
        )
        .groupby(["model", "filter", "horizon"], as_index=False)
        .head(1)
    )

    merged = best_val.merge(
        test,
        on=policy_cols,
        suffixes=("_val", "_test"),
        how="left",
    )

    out_path = OUT_DIR / "strict_validation_selected_policies_XGB_CLEAN_FULL_applied_to_test.csv"
    atomic_write_csv(merged, out_path)
    print("WROTE XGB FULL STRICT:", out_path)

    return merged


def build_comparison_files(xgb_strict: pd.DataFrame):
    existing_path = Path("data/training/manual_verified_results/exit_sim_fixed/strict_validation_selected_policies_applied_to_test.csv")

    if existing_path.exists() and len(xgb_strict):
        existing = pd.read_csv(existing_path)
        existing["pair_cap_str"] = existing["pair_cap_str"].astype(str)
        combined = pd.concat([existing, xgb_strict], ignore_index=True, sort=False)
    else:
        combined = xgb_strict.copy()

    combined_path = OUT_DIR / "strict_comparison_TAB_RF_XGB_CLEAN_FULL.csv"
    atomic_write_csv(combined, combined_path)

    cols = [
        "model", "filter", "horizon",
        "top_pct", "pair_cap_str", "tp_ratio", "sl_ratio",
        "selected_test", "total_net_return_test", "avg_net_return_test",
        "target_precision_test", "net_win_rate_test",
        "tp_count_test", "sl_count_test", "time_count_test",
        "unique_pairs_test", "top_pair_share_test",
    ]

    existing_cols = [c for c in cols if c in combined.columns]

    if len(combined) and existing_cols:
        comparison_sorted = combined.sort_values(
            ["total_net_return_test", "avg_net_return_test", "target_precision_test"],
            ascending=False,
        )[existing_cols].copy()
    else:
        comparison_sorted = pd.DataFrame(columns=existing_cols)

    summary_txt = OUT_DIR / "xgb_full_summary_for_upload.txt"

    lines = []
    lines.append("STRICT COMPARISON — TAB / RF / XGB — FULL XGB SCOPE")
    lines.append("=" * 80)
    lines.append("")
    lines.append(comparison_sorted.head(80).to_string(index=False))
    lines.append("")
    lines.append("")
    lines.append("BEST XGB ONLY")
    lines.append("=" * 80)
    lines.append("")

    if len(xgb_strict) and existing_cols:
        xgb_sorted = xgb_strict.sort_values(
            ["total_net_return_test", "avg_net_return_test", "target_precision_test"],
            ascending=False,
        )[existing_cols].head(40)
        lines.append(xgb_sorted.to_string(index=False))
    else:
        lines.append("No strict XGB policies.")

    lines.append("")

    atomic_write_text("\n".join(lines), summary_txt)

    print("\nWROTE COMBINED:", combined_path)
    print("WROTE SUMMARY TXT:", summary_txt)

    print("\nSTRICT COMPARISON — TAB/RF/XGB")
    if len(comparison_sorted):
        print(comparison_sorted.head(50).to_string(index=False))
    else:
        print("No comparison rows.")


def write_run_manifest(started_at: float, status: str):
    payload = {
        "status": status,
        "created_at_utc": pd.Timestamp.utcnow().isoformat(),
        "duration_seconds": round(time.time() - started_at, 3),
        "use_cuda": USE_CUDA,
        "dataset_dir": str(DATASET_DIR),
        "xgb_out_dir": str(XGB_OUT_DIR),
        "exit_sim_out_dir": str(OUT_DIR),
        "filters": FILTERS,
        "horizons": HORIZONS,
        "round_trip_fee": ROUND_TRIP_FEE,
        "top_pcts": TOP_PCTS,
        "pair_caps": ["none" if p is None else p for p in PAIR_CAPS],
        "tp_ratios": TP_RATIOS,
        "sl_ratios": SL_RATIOS,
        "strict": {
            "min_selected": STRICT_MIN_SELECTED,
            "min_valid_sim_rate": STRICT_MIN_VALID_SIM_RATE,
            "min_unique_pairs": STRICT_MIN_UNIQUE_PAIRS,
            "max_top_pair_share": STRICT_MAX_TOP_PAIR_SHARE,
        },
    }

    atomic_write_json(payload, OUT_DIR / "xgb_clean_full_cuda_manifest.json")


def main():
    global SNAP_GROUPS

    started_at = time.time()

    print("=" * 80)
    print("FULL XGB CLEAN CUDA RUN")
    print("=" * 80)

    validate_required_inputs()
    safe_clean_xgb_outputs()

    SNAP_GROUPS = load_snapshot_groups()

    for f in FILTERS:
        for h in HORIZONS:
            train_xgb_predictions(f, h)

    sweep = run_xgb_exit_sweep()
    strict = strict_selector(sweep)
    build_comparison_files(strict)

    write_run_manifest(started_at, "ok")

    print("\nDONE")
    print("XGB predictions:", XGB_OUT_DIR)
    print("XGB exit simulation:", OUT_DIR)
    print("Duration seconds:", round(time.time() - started_at, 3))


if __name__ == "__main__":
    main()
