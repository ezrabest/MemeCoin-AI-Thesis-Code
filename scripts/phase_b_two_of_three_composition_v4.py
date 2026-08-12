from __future__ import annotations

import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(".").resolve()

OUT_DIR = ROOT / "data" / "training" / "manual_verified_results" / "phase_b_model_cuts_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = ROOT / "data" / "trader.db"

PHASE_B_V2_DIR = ROOT / "data" / "training" / "manual_verified_results" / "phase_b_model_cuts_v2"

SEARCH_ROOTS = [
    ROOT / "data" / "training" / "manual_verified_results",
    ROOT / "data" / "training" / "models",
    ROOT / "data" / "training",
]

MODELS = ["TAB", "XGB", "RF"]

HORIZON_MINUTES = {
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "8h": 480,
    "24h": 1440,
}

ROUND_TRIP_FEE_DEFAULT = 0.0308

MODEL_TOKENS = {
    "TAB": ["tab", "tabicl"],
    "XGB": ["xgb", "xgboost"],
    "RF": ["rf", "random_forest", "randomforest"],
}

BAD_MODEL_TOKENS = {
    "TAB": ["xgb", "xgboost", "rf", "random_forest", "randomforest"],
    "XGB": ["tab", "tabicl", "rf", "random_forest", "randomforest"],
    "RF": ["tab", "tabicl", "xgb", "xgboost"],
}

TEXT_COLUMNS = {
    "model",
    "strategy",
    "filter",
    "filter_name",
    "dataset_filter",
    "horizon",
    "split",
    "pair_address",
    "pairAddress",
    "contract_address",
    "token_address",
    "timestamp",
    "event_timestamp",
    "snapshot_timestamp",
    "created_at",
    "phase_b_notes",
    "source_kind",
    "pair_cap",
    "pair_cap_str",
}

SCORE_CANDIDATES = [
    "score",
    "prediction_score",
    "predicted_probability",
    "probability",
    "prob_positive",
    "proba",
    "y_score",
    "pred_proba",
    "positive_probability",
    "p_positive",
    "prob",
    "tab_score",
    "rf_score",
    "xgb_score",
]

TIMESTAMP_CANDIDATES = [
    "event_timestamp",
    "timestamp",
    "snapshot_timestamp",
    "ts",
    "created_at",
    "datetime",
]

PAIR_CANDIDATES = [
    "pair_address",
    "pairAddress",
    "pair",
    "contract_address",
    "token_address",
]

TARGET_CANDIDATES = [
    "y_true",
    "target",
    "label",
    "target_x2",
    "target_profitable",
    "target_profitable_30m",
    "target_profitable_1h",
    "target_profitable_4h",
    "target_profitable_8h",
    "target_profitable_24h",
]

ROW_ID_CANDIDATES = [
    "row_id",
    "event_id",
    "candidate_id",
    "snapshot_id",
    "source_row_id",
    "market_snapshot_id",
    "id",
]


@dataclass
class PredictionLoadResult:
    model: str
    split: str
    filter_name: str
    horizon: str
    path: str
    df: pd.DataFrame
    schema: dict[str, Any]
    selection_reason: str


def normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def exact_horizon_in_name(name: str, horizon: str) -> bool:
    n = normalize_name(name)
    h = normalize_name(horizon)
    parts = n.split("_")
    return h in parts or f"x2_{h}" in n or f"_{h}_" in f"_{n}_"


def exact_filter_in_name(name: str, filter_name: str) -> bool:
    return normalize_name(filter_name) in normalize_name(name)


def split_in_name(name: str, split: str) -> bool:
    n = normalize_name(name)
    s = normalize_name(split)
    return s in n.split("_") or f"_{s}_" in f"_{n}_"


def model_token_in_name(name: str, model: str) -> bool:
    n = normalize_name(name)
    return any(tok in n for tok in MODEL_TOKENS[model])


def bad_model_token_in_name(name: str, model: str) -> bool:
    n = normalize_name(name)
    return any(tok in n for tok in BAD_MODEL_TOKENS[model])


def list_parquet_files() -> list[Path]:
    files = []

    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        files.extend(root.rglob("*.parquet"))

    files = sorted(set(files), key=lambda p: str(p).lower())
    return files


def safe_read_parquet(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    exact = {c.lower(): c for c in df.columns}

    for wanted in candidates:
        if wanted.lower() in exact:
            return exact[wanted.lower()]

    normalized = {normalize_name(c): c for c in df.columns}

    for wanted in candidates:
        nw = normalize_name(wanted)
        if nw in normalized:
            return normalized[nw]

    return None


def find_score_col(df: pd.DataFrame, model: str) -> str:
    cols = list(df.columns)

    model_specific = {
        "TAB": ["score_TAB", "score_tab", "tab_score", "tabicl_score", "prob_TAB", "prob_tab"],
        "XGB": ["score_XGB", "score_xgb", "xgb_score", "xgboost_score", "prob_XGB", "prob_xgb"],
        "RF": ["score_RF", "score_rf", "rf_score", "random_forest_score", "prob_RF", "prob_rf"],
    }

    for c in model_specific[model]:
        if c in cols:
            return c

    for c in SCORE_CANDIDATES:
        if c in cols:
            return c

    numeric_cols = []

    for c in cols:
        lc = normalize_name(c)
        if any(tok in lc for tok in ["prob", "score", "pred"]):
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_cols.append(c)

    if len(numeric_cols) == 1:
        return numeric_cols[0]

    raise ValueError(
        f"Could not identify unique score column for model={model}. "
        f"Candidates={numeric_cols}. Columns={cols}"
    )


def filter_if_column_exists(df: pd.DataFrame, column_names: list[str], expected: str) -> tuple[pd.DataFrame, str | None, bool]:
    col = find_col(df, column_names)

    if col is None:
        return df, None, False

    before = len(df)
    expected_norm = normalize_name(expected)

    mask = df[col].astype(str).map(normalize_name).eq(expected_norm)
    out = df[mask].copy()

    return out, col, before != len(out)


def read_and_normalize_prediction(path: Path, model: str, split: str, filter_name: str, horizon: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = safe_read_parquet(path)

    if raw is None:
        raise ValueError(f"Could not read parquet: {path}")

    df = raw.copy()
    original_rows = len(df)

    df, split_col, split_filtered = filter_if_column_exists(df, ["split", "dataset_split"], split)
    df, filter_col, filter_filtered = filter_if_column_exists(df, ["filter", "filter_name", "dataset_filter"], filter_name)
    df, horizon_col, horizon_filtered = filter_if_column_exists(df, ["horizon", "target_horizon"], horizon)

    model_col = find_col(df, ["model", "model_name", "model_type", "provider"])

    if model_col is not None:
        mnorm = normalize_name(model)
        model_mask = df[model_col].astype(str).map(normalize_name).str.contains(mnorm, regex=False)
        if model_mask.any():
            df = df[model_mask].copy()

    if df.empty:
        raise ValueError(
            f"After filtering, no rows left. path={path}, model={model}, split={split}, filter={filter_name}, horizon={horizon}"
        )

    ts_col = find_col(df, TIMESTAMP_CANDIDATES)

    if ts_col is None:
        raise ValueError(f"No timestamp column in {path}. Columns={list(df.columns)}")

    pair_col = find_col(df, PAIR_CANDIDATES)

    if pair_col is None:
        raise ValueError(f"No pair address column in {path}. Columns={list(df.columns)}")

    score_col = find_score_col(df, model)

    target_col = find_col(df, TARGET_CANDIDATES)
    row_id_col = find_col(df, ROW_ID_CANDIDATES)

    out = pd.DataFrame()
    out["pair_address"] = df[pair_col].astype(str).str.strip().str.lower()
    out["event_timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    out[f"score_{model}"] = pd.to_numeric(df[score_col], errors="coerce")

    if target_col is not None:
        out["target"] = pd.to_numeric(df[target_col], errors="coerce")
    else:
        out["target"] = np.nan

    if row_id_col is not None:
        out["row_id"] = df[row_id_col].astype(str)
    else:
        out["row_id"] = ""

    out = out[
        out["pair_address"].str.len().gt(0)
        & out["event_timestamp"].notna()
        & out[f"score_{model}"].notna()
    ].copy()

    out["event_ns"] = out["event_timestamp"].array.asi8.astype(np.int64)

    # Canonical key: pair + exact event timestamp. This is the safest common key across model outputs.
    out["event_key"] = out["pair_address"] + "|" + out["event_ns"].astype(str)

    dup = out["event_key"].duplicated(keep=False)

    if dup.any():
        dup_count = int(dup.sum())
        examples = out.loc[dup, ["event_key", "pair_address", "event_timestamp"]].head(10).to_dict("records")
        raise ValueError(
            f"Duplicate event_key within prediction file. path={path}, model={model}, duplicates={dup_count}, examples={examples}"
        )

    out = out.sort_values("event_key").reset_index(drop=True)

    schema = {
        "path": str(path),
        "model": model,
        "split": split,
        "filter": filter_name,
        "horizon": horizon,
        "original_rows": int(original_rows),
        "filtered_rows": int(len(out)),
        "columns": list(raw.columns),
        "pair_col": pair_col,
        "timestamp_col": ts_col,
        "score_col": score_col,
        "target_col": target_col,
        "row_id_col": row_id_col,
        "split_col": split_col,
        "filter_col": filter_col,
        "horizon_col": horizon_col,
        "split_filtered": split_filtered,
        "filter_filtered": filter_filtered,
        "horizon_filtered": horizon_filtered,
        "min_timestamp": str(out["event_timestamp"].min()),
        "max_timestamp": str(out["event_timestamp"].max()),
        "unique_pairs": int(out["pair_address"].nunique()),
    }

    return out, schema


def score_prediction_candidate(path: Path, model: str, split: str, filter_name: str, horizon: str) -> tuple[int, list[str]]:
    name = path.name
    full = str(path)

    score = 0
    reasons = []

    if bad_model_token_in_name(full, model):
        return -10_000, [f"reject_bad_model_token_for_{model}"]

    if model_token_in_name(full, model):
        score += 100
        reasons.append("model_token_in_path")

    if split_in_name(name, split):
        score += 30
        reasons.append("split_in_filename")

    if exact_filter_in_name(name, filter_name):
        score += 30
        reasons.append("filter_in_filename")

    if exact_horizon_in_name(name, horizon):
        score += 30
        reasons.append("horizon_in_filename")

    lower_full = normalize_name(full)

    if "manual_verified" in lower_full:
        score += 20
        reasons.append("manual_verified_path")

    if "clean" in lower_full:
        score += 15
        reasons.append("clean_path")

    if model == "XGB" and "xgb_clean_full" in lower_full:
        score += 40
        reasons.append("xgb_clean_full_path")

    if model == "TAB" and ("tab" in lower_full or "tabicl" in lower_full):
        score += 40
        reasons.append("tab_path")

    if model == "RF" and ("rf" in lower_full or "random_forest" in lower_full):
        score += 40
        reasons.append("rf_path")

    return score, reasons


def discover_prediction_file(model: str, split: str, filter_name: str, horizon: str) -> PredictionLoadResult:
    files = list_parquet_files()
    candidates = []

    for p in files:
        score, reasons = score_prediction_candidate(p, model, split, filter_name, horizon)

        if score < 30:
            continue

        # Hard safety: if filename has explicit wrong horizon/filter, reject.
        name_norm = normalize_name(p.name)

        explicit_horizons = ["30m", "1h", "4h", "8h", "24h"]
        horizons_in_name = [h for h in explicit_horizons if exact_horizon_in_name(p.name, h)]

        if horizons_in_name and horizon not in horizons_in_name:
            continue

        explicit_filters = [
            "RAW_ALL_VERIFIED",
            "LIQ_5K_HIGH_ACTIVITY",
            "LOW_LIQ_MOMENTUM",
            "NO_WHALE_FILTER",
        ]
        filters_in_name = [f for f in explicit_filters if exact_filter_in_name(p.name, f)]

        if filters_in_name and filter_name not in filters_in_name:
            continue

        candidates.append((score, reasons, p))

    if not candidates:
        raise FileNotFoundError(
            f"No candidate parquet found for model={model}, split={split}, filter={filter_name}, horizon={horizon}"
        )

    # Try reading high-score candidates and validate schema/content.
    candidates = sorted(candidates, key=lambda x: (x[0], str(x[2])), reverse=True)

    successful = []

    for score, reasons, path in candidates[:30]:
        try:
            df, schema = read_and_normalize_prediction(path, model, split, filter_name, horizon)
        except Exception as exc:
            continue

        # If the file did not encode filter/horizon in name, it must have filtered by columns.
        name_has_filter = exact_filter_in_name(path.name, filter_name)
        name_has_horizon = exact_horizon_in_name(path.name, horizon)
        name_has_split = split_in_name(path.name, split)

        if not name_has_filter and not schema["filter_filtered"] and schema["filter_col"] is None:
            continue

        if not name_has_horizon and not schema["horizon_filtered"] and schema["horizon_col"] is None:
            continue

        if not name_has_split and not schema["split_filtered"] and schema["split_col"] is None:
            continue

        successful.append((score, reasons, path, df, schema))

    if not successful:
        raise FileNotFoundError(
            f"Candidates existed but none passed schema/content validation for model={model}, split={split}, filter={filter_name}, horizon={horizon}"
        )

    successful = sorted(successful, key=lambda x: (x[0], x[4]["filtered_rows"], str(x[2])), reverse=True)
    top = successful[0]

    # Ambiguity guard: if multiple valid files tie closely and differ in path, fail.
    close = [x for x in successful if x[0] == top[0] and x[4]["filtered_rows"] == top[4]["filtered_rows"]]

    if len(close) > 1:
        paths = [str(x[2]) for x in close[:10]]
        raise RuntimeError(
            f"Ambiguous prediction files for {model}/{split}/{filter_name}/{horizon}. "
            f"Top candidates tie: {paths}"
        )

    score, reasons, path, df, schema = top

    return PredictionLoadResult(
        model=model,
        split=split,
        filter_name=filter_name,
        horizon=horizon,
        path=str(path),
        df=df,
        schema=schema,
        selection_reason=";".join(reasons),
    )


def validate_common_universe(loads: dict[str, PredictionLoadResult]) -> dict[str, Any]:
    key_sets = {m: set(loads[m].df["event_key"]) for m in MODELS}
    counts = {m: len(key_sets[m]) for m in MODELS}

    base = key_sets["XGB"]
    mismatches = {}

    for m in ["TAB", "RF"]:
        missing_from_m = base - key_sets[m]
        extra_in_m = key_sets[m] - base

        mismatches[m] = {
            "missing_from_model_vs_xgb": int(len(missing_from_m)),
            "extra_in_model_vs_xgb": int(len(extra_in_m)),
            "missing_examples": list(sorted(missing_from_m))[:5],
            "extra_examples": list(sorted(extra_in_m))[:5],
        }

    ok = all(v["missing_from_model_vs_xgb"] == 0 and v["extra_in_model_vs_xgb"] == 0 for v in mismatches.values())

    return {
        "ok": bool(ok),
        "counts": counts,
        "mismatches": mismatches,
    }


def build_common_frame(loads: dict[str, PredictionLoadResult]) -> pd.DataFrame:
    frames = []

    for model in MODELS:
        df = loads[model].df[["event_key", "pair_address", "event_timestamp", "event_ns", f"score_{model}", "target"]].copy()
        df = df.rename(columns={"target": f"target_{model}"})
        frames.append(df)

    common = frames[0]

    for f in frames[1:]:
        common = common.merge(
            f,
            on=["event_key", "pair_address", "event_timestamp", "event_ns"],
            how="inner",
            validate="one_to_one",
        )

    # Check target consistency if available.
    target_cols = [f"target_{m}" for m in MODELS if f"target_{m}" in common.columns]
    numeric_targets = []

    for c in target_cols:
        if common[c].notna().any():
            numeric_targets.append(c)

    if numeric_targets:
        common["target"] = common[numeric_targets].bfill(axis=1).iloc[:, 0]
    else:
        common["target"] = np.nan

    for m in MODELS:
        common[f"score_{m}"] = pd.to_numeric(common[f"score_{m}"], errors="coerce")

    common = common.dropna(subset=[f"score_{m}" for m in MODELS]).copy()
    common = common.sort_values("event_key").reset_index(drop=True)

    return common


def top_k_count(n: int, top_pct: float) -> int:
    return max(1, int(n * top_pct))


def select_model_top_set(common: pd.DataFrame, model: str, top_pct: float, pair_cap: str, ranking_mode: str) -> set[str]:
    k = top_k_count(len(common), top_pct)
    score_col = f"score_{model}"

    if ranking_mode == "score_desc":
        ranked = common.sort_values(score_col, ascending=False).copy()
    elif ranking_mode == "score_desc_time_asc":
        ranked = common.sort_values([score_col, "event_timestamp"], ascending=[False, True]).copy()
    elif ranking_mode == "score_desc_pair_asc":
        ranked = common.sort_values([score_col, "pair_address", "event_timestamp"], ascending=[False, True, True]).copy()
    else:
        raise ValueError(f"Unknown ranking_mode={ranking_mode}")

    selected = []

    if str(pair_cap).lower() == "none":
        selected = ranked.head(k)["event_key"].tolist()
        return set(selected)

    cap = int(float(pair_cap))
    pair_counts = {}

    for row in ranked.itertuples(index=False):
        pair = row.pair_address
        n = pair_counts.get(pair, 0)

        if n >= cap:
            continue

        selected.append(row.event_key)
        pair_counts[pair] = n + 1

        if len(selected) >= k:
            break

    return set(selected)


def apply_consensus_final_pair_cap(df: pd.DataFrame, pair_cap: str, expected_selected: int | None, final_ranking_mode: str) -> pd.DataFrame:
    out = df.copy()

    if final_ranking_mode == "vote_score_mean":
        out["final_rank_score"] = out[["score_TAB", "score_XGB", "score_RF"]].mean(axis=1)
        out = out.sort_values(["vote_count", "final_rank_score"], ascending=[False, False]).copy()
    elif final_ranking_mode == "score_mean":
        out["final_rank_score"] = out[["score_TAB", "score_XGB", "score_RF"]].mean(axis=1)
        out = out.sort_values(["final_rank_score"], ascending=False).copy()
    elif final_ranking_mode == "min_score":
        out["final_rank_score"] = out[["score_TAB", "score_XGB", "score_RF"]].min(axis=1)
        out = out.sort_values(["vote_count", "final_rank_score"], ascending=[False, False]).copy()
    elif final_ranking_mode == "event_time":
        out = out.sort_values(["event_timestamp"], ascending=True).copy()
    elif final_ranking_mode == "no_final_rank":
        out = out.sort_values(["event_key"], ascending=True).copy()
    else:
        raise ValueError(f"Unknown final_ranking_mode={final_ranking_mode}")

    if str(pair_cap).lower() == "none":
        if expected_selected is not None:
            return out.head(expected_selected).copy().reset_index(drop=True)
        return out.reset_index(drop=True)

    cap = int(float(pair_cap))
    keep = []
    pair_counts = {}

    for idx, row in out.iterrows():
        pair = row["pair_address"]
        n = pair_counts.get(pair, 0)

        if n >= cap:
            continue

        keep.append(idx)
        pair_counts[pair] = n + 1

        if expected_selected is not None and len(keep) >= expected_selected:
            break

    return out.loc[keep].copy().reset_index(drop=True)


def load_snapshot_groups() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing SQLite DB: {DB_PATH}")

    conn = sqlite3.connect(f"file:{DB_PATH.as_posix()}?mode=ro", uri=True)

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

    snap["pair_address"] = snap["pair_address"].astype(str).str.strip().str.lower()
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

    return groups


SNAPSHOT_GROUPS: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def simulate_trade(pair: str, event_ns: int, horizon: str, tp_ratio: float, sl_ratio: float, fee: float) -> dict[str, Any]:
    if pair not in SNAPSHOT_GROUPS:
        return {
            "valid": False,
            "exit_status": "NO_PAIR",
            "net_return": np.nan,
            "gross_ratio": np.nan,
            "minutes_to_exit": np.nan,
        }

    ts, prices = SNAPSHOT_GROUPS[pair]

    entry_idx = np.searchsorted(ts, event_ns, side="right") - 1

    if entry_idx < 0 or entry_idx >= len(ts):
        return {
            "valid": False,
            "exit_status": "NO_ENTRY",
            "net_return": np.nan,
            "gross_ratio": np.nan,
            "minutes_to_exit": np.nan,
        }

    entry_ts = int(ts[entry_idx])
    entry_price = float(prices[entry_idx])

    if not np.isfinite(entry_price) or entry_price <= 0:
        return {
            "valid": False,
            "exit_status": "BAD_ENTRY_PRICE",
            "net_return": np.nan,
            "gross_ratio": np.nan,
            "minutes_to_exit": np.nan,
        }

    horizon_min = HORIZON_MINUTES[horizon]
    end_ns = entry_ts + int(pd.Timedelta(minutes=horizon_min).value)

    start = entry_idx + 1
    end = np.searchsorted(ts, end_ns, side="right")

    if end <= start:
        return {
            "valid": False,
            "exit_status": "NO_FUTURE",
            "net_return": np.nan,
            "gross_ratio": np.nan,
            "minutes_to_exit": np.nan,
        }

    future_ts = ts[start:end]
    future_prices = prices[start:end]
    ok = np.isfinite(future_prices) & (future_prices > 0)

    if not ok.any():
        return {
            "valid": False,
            "exit_status": "NO_VALID_FUTURE",
            "net_return": np.nan,
            "gross_ratio": np.nan,
            "minutes_to_exit": np.nan,
        }

    future_ts = future_ts[ok]
    future_prices = future_prices[ok]
    ratios = future_prices / entry_price

    exit_status = "TIME"
    gross_ratio = float(ratios[-1])
    exit_ts = int(future_ts[-1])

    for t, ratio in zip(future_ts, ratios):
        ratio = float(ratio)

        if ratio >= tp_ratio:
            exit_status = "TP"
            gross_ratio = float(tp_ratio)
            exit_ts = int(t)
            break

        if ratio <= sl_ratio:
            exit_status = "SL"
            gross_ratio = float(sl_ratio)
            exit_ts = int(t)
            break

    net_return = float((gross_ratio - 1.0) - fee)

    return {
        "valid": True,
        "exit_status": exit_status,
        "net_return": net_return,
        "gross_ratio": gross_ratio,
        "minutes_to_exit": float((exit_ts - entry_ts) / 60_000_000_000),
        "entry_price": entry_price,
        "max_ratio": float(np.max(ratios)),
        "min_ratio": float(np.min(ratios)),
    }


def classify_combo(in_tab: bool, in_xgb: bool, in_rf: bool) -> str:
    if in_tab and in_xgb and in_rf:
        return "TAB_XGB_RF_ALL3"
    if in_tab and in_xgb:
        return "TAB_XGB_ONLY"
    if in_tab and in_rf:
        return "TAB_RF_ONLY"
    if in_xgb and in_rf:
        return "XGB_RF_ONLY"
    return "NOT_TWO_OF_THREE"


def reconstruct_policy(policy: pd.Series, loads_cache: dict[tuple[str, str, str, str], PredictionLoadResult]) -> tuple[pd.DataFrame, dict[str, Any]]:
    split = "test"
    filter_name = str(policy["filter"])
    horizon = str(policy["horizon"])
    top_pct = float(policy["top_pct"])
    pair_cap = str(policy["pair_cap_str"])
    tp_ratio = float(policy["tp_ratio"])
    sl_ratio = float(policy["sl_ratio"])
    fee = float(policy.get("round_trip_fee_test", ROUND_TRIP_FEE_DEFAULT)) if "round_trip_fee_test" in policy.index else ROUND_TRIP_FEE_DEFAULT

    loads: dict[str, PredictionLoadResult] = {}

    for model in MODELS:
        key = (model, split, filter_name, horizon)

        if key not in loads_cache:
            loads_cache[key] = discover_prediction_file(model, split, filter_name, horizon)

        loads[model] = loads_cache[key]

    universe_check = validate_common_universe(loads)

    if not universe_check["ok"]:
        raise RuntimeError(
            f"Universe mismatch for {filter_name}/{horizon}. Details={json.dumps(universe_check, indent=2)}"
        )

    common = build_common_frame(loads)

    if common.empty:
        raise RuntimeError(f"Common frame empty for {filter_name}/{horizon}")

    expected_selected = int(policy["selected_test"])
    expected_total = float(policy["total_net_return_test"])

    ranking_modes = [
        "score_desc",
        "score_desc_time_asc",
        "score_desc_pair_asc",
    ]

    final_modes = [
        "no_final_rank",
        "vote_score_mean",
        "score_mean",
        "min_score",
        "event_time",
    ]

    attempts = []

    for ranking_mode in ranking_modes:
        top_sets = {}

        for model in MODELS:
            top_sets[model] = select_model_top_set(
                common=common,
                model=model,
                top_pct=top_pct,
                pair_cap=pair_cap,
                ranking_mode=ranking_mode,
            )

        temp = common.copy()
        temp["in_TAB"] = temp["event_key"].isin(top_sets["TAB"])
        temp["in_XGB"] = temp["event_key"].isin(top_sets["XGB"])
        temp["in_RF"] = temp["event_key"].isin(top_sets["RF"])
        temp["vote_count"] = temp[["in_TAB", "in_XGB", "in_RF"]].sum(axis=1)
        temp = temp[temp["vote_count"] >= 2].copy()

        if temp.empty:
            attempts.append({
                "ranking_mode": ranking_mode,
                "final_mode": "none",
                "selected": 0,
                "total": np.nan,
                "selected_diff": expected_selected,
                "total_diff": np.nan,
                "status": "EMPTY_TWO_OF_THREE",
            })
            continue

        temp["combo"] = [
            classify_combo(a, b, c)
            for a, b, c in zip(temp["in_TAB"], temp["in_XGB"], temp["in_RF"])
        ]

        for final_mode in final_modes:
            selected = apply_consensus_final_pair_cap(
                temp,
                pair_cap=pair_cap,
                expected_selected=expected_selected,
                final_ranking_mode=final_mode,
            )

            sims = [
                simulate_trade(
                    pair=row.pair_address,
                    event_ns=int(row.event_ns),
                    horizon=horizon,
                    tp_ratio=tp_ratio,
                    sl_ratio=sl_ratio,
                    fee=fee,
                )
                for row in selected.itertuples(index=False)
            ]

            sim_df = pd.DataFrame(sims)
            selected_eval = pd.concat([selected.reset_index(drop=True), sim_df.reset_index(drop=True)], axis=1)

            valid = selected_eval[selected_eval["valid"] == True].copy()
            reconstructed_total = float(valid["net_return"].sum()) if len(valid) else np.nan

            attempts.append({
                "ranking_mode": ranking_mode,
                "final_mode": final_mode,
                "selected": int(len(selected_eval)),
                "valid_sims": int(len(valid)),
                "total": reconstructed_total,
                "selected_diff": int(len(selected_eval) - expected_selected),
                "total_diff": float(reconstructed_total - expected_total) if np.isfinite(reconstructed_total) else np.nan,
                "abs_total_diff": abs(float(reconstructed_total - expected_total)) if np.isfinite(reconstructed_total) else np.nan,
                "status": "OK_ATTEMPT",
                "selected_eval": selected_eval,
                "paths": {m: loads[m].path for m in MODELS},
                "schemas": {m: loads[m].schema for m in MODELS},
                "universe_check": universe_check,
            })

    valid_attempts = [
        a for a in attempts
        if a["selected"] == expected_selected
        and a["valid_sims"] == expected_selected
        and np.isfinite(a["total"])
    ]

    if not valid_attempts:
        best = sorted(
            [a for a in attempts if isinstance(a.get("abs_total_diff"), float) and np.isfinite(a.get("abs_total_diff"))],
            key=lambda a: (abs(a["selected_diff"]), a["abs_total_diff"]),
        )

        diagnostics = []
        for a in best[:10]:
            diagnostics.append({
                "ranking_mode": a["ranking_mode"],
                "final_mode": a["final_mode"],
                "selected": a["selected"],
                "valid_sims": a["valid_sims"],
                "total": a["total"],
                "selected_diff": a["selected_diff"],
                "total_diff": a["total_diff"],
            })

        raise RuntimeError(
            "No reconstruction attempt matched selected count + valid sims. "
            f"Policy expected_selected={expected_selected}, expected_total={expected_total}. "
            f"Best attempts={json.dumps(diagnostics, indent=2)}"
        )

    # Choose the attempt closest to the official total.
    valid_attempts = sorted(valid_attempts, key=lambda a: a["abs_total_diff"])
    chosen = valid_attempts[0]

    # Very strict guard: if not close to official aggregate, do not trust composition.
    # A tolerance of 1e-6 should pass if simulator/selection are exact.
    if chosen["abs_total_diff"] > 1e-6:
        diagnostics = [
            {
                "ranking_mode": a["ranking_mode"],
                "final_mode": a["final_mode"],
                "selected": a["selected"],
                "valid_sims": a["valid_sims"],
                "total": a["total"],
                "selected_diff": a["selected_diff"],
                "total_diff": a["total_diff"],
                "abs_total_diff": a["abs_total_diff"],
            }
            for a in valid_attempts[:10]
        ]

        raise RuntimeError(
            "Reconstructed selected count matched, but total return did not reconcile exactly. "
            f"Expected total={expected_total}, best total={chosen['total']}, diff={chosen['total_diff']}. "
            f"Best attempts={json.dumps(diagnostics, indent=2)}"
        )

    selected_eval = chosen["selected_eval"].copy()

    selected_eval["policy_source_kind"] = policy["source_kind"]
    selected_eval["policy_strategy"] = policy["strategy"]
    selected_eval["policy_filter"] = filter_name
    selected_eval["policy_horizon"] = horizon
    selected_eval["policy_top_pct"] = top_pct
    selected_eval["policy_pair_cap"] = pair_cap
    selected_eval["policy_tp_ratio"] = tp_ratio
    selected_eval["policy_sl_ratio"] = sl_ratio
    selected_eval["policy_selected_reported"] = expected_selected
    selected_eval["policy_total_reported"] = expected_total
    selected_eval["ranking_mode_chosen"] = chosen["ranking_mode"]
    selected_eval["final_mode_chosen"] = chosen["final_mode"]

    metadata = {
        "status": "RECONCILED",
        "filter": filter_name,
        "horizon": horizon,
        "top_pct": top_pct,
        "pair_cap": pair_cap,
        "tp_ratio": tp_ratio,
        "sl_ratio": sl_ratio,
        "fee": fee,
        "expected_selected": expected_selected,
        "expected_total": expected_total,
        "reconstructed_selected": int(len(selected_eval)),
        "reconstructed_total": float(selected_eval["net_return"].sum()),
        "total_diff": float(selected_eval["net_return"].sum() - expected_total),
        "ranking_mode_chosen": chosen["ranking_mode"],
        "final_mode_chosen": chosen["final_mode"],
        "paths": chosen["paths"],
        "schemas": chosen["schemas"],
        "universe_check": chosen["universe_check"],
    }

    return selected_eval, metadata


def summarize_selected(selected: pd.DataFrame, metadata: dict[str, Any]) -> pd.DataFrame:
    rows = []

    for combo, g in selected.groupby("combo", dropna=False):
        valid = g[g["valid"] == True].copy()

        row = {
            "source_kind": metadata["source_kind"] if "source_kind" in metadata else selected["policy_source_kind"].iloc[0],
            "strategy": selected["policy_strategy"].iloc[0],
            "filter": metadata["filter"],
            "horizon": metadata["horizon"],
            "top_pct": metadata["top_pct"],
            "pair_cap": metadata["pair_cap"],
            "tp_ratio": metadata["tp_ratio"],
            "sl_ratio": metadata["sl_ratio"],
            "policy_selected_reported": metadata["expected_selected"],
            "policy_total_reported": metadata["expected_total"],
            "reconstructed_total": metadata["reconstructed_total"],
            "reconstruction_total_diff": metadata["total_diff"],
            "ranking_mode_chosen": metadata["ranking_mode_chosen"],
            "final_mode_chosen": metadata["final_mode_chosen"],
            "combo": combo,
            "combo_selected": int(len(g)),
            "combo_share_of_policy": float(len(g) / max(len(selected), 1)),
            "combo_valid_sims": int(len(valid)),
            "combo_total_net": float(valid["net_return"].sum()) if len(valid) else np.nan,
            "combo_avg_net": float(valid["net_return"].mean()) if len(valid) else np.nan,
            "combo_target_precision": float(valid["target"].mean()) if "target" in valid.columns and valid["target"].notna().any() else np.nan,
            "combo_net_win_rate": float((valid["net_return"] > 0).mean()) if len(valid) else np.nan,
            "combo_tp_count": int((valid["exit_status"] == "TP").sum()) if len(valid) else 0,
            "combo_sl_count": int((valid["exit_status"] == "SL").sum()) if len(valid) else 0,
            "combo_time_count": int((valid["exit_status"] == "TIME").sum()) if len(valid) else 0,
            "combo_unique_pairs": int(valid["pair_address"].nunique()) if len(valid) else 0,
            "combo_top_pair_share": float(valid["pair_address"].value_counts(normalize=True).iloc[0]) if len(valid) else np.nan,
        }

        rows.append(row)

    return pd.DataFrame(rows)


def load_two_of_three_policies() -> pd.DataFrame:
    path = PHASE_B_V2_DIR / "phase_b_v2_two_of_three_ranked.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing TWO_OF_THREE policies: {path}")

    df = pd.read_csv(path)

    # Only robust policies are eligible for final decomposition.
    df = df[df["phase_b_robust_ok"].astype(str).str.lower().eq("true")].copy()

    if df.empty:
        raise RuntimeError("No robust TWO_OF_THREE policies found.")

    # Focus on formal policies, not research-only.
    df = df[~df["filter"].astype(str).eq("LOW_LIQ_MOMENTUM")].copy()

    # Sort exactly as V2.
    df["total_net_return_test"] = pd.to_numeric(df["total_net_return_test"], errors="coerce")
    df["avg_net_return_test"] = pd.to_numeric(df["avg_net_return_test"], errors="coerce")

    df = df.sort_values(
        ["total_net_return_test", "avg_net_return_test"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return df


def main() -> None:
    global SNAPSHOT_GROUPS

    print("Loading raw SQLite market snapshots...")
    SNAPSHOT_GROUPS = load_snapshot_groups()
    print(f"Loaded snapshot groups: {len(SNAPSHOT_GROUPS)} pairs")

    policies = load_two_of_three_policies()
    loads_cache: dict[tuple[str, str, str, str], PredictionLoadResult] = {}

    selected_frames = []
    composition_frames = []
    policy_results = []
    failures = []

    for idx, policy in policies.iterrows():
        label = (
            f"{policy['source_kind']} / {policy['strategy']} / {policy['filter']} / {policy['horizon']} / "
            f"top={policy['top_pct']} / cap={policy['pair_cap_str']} / TP={policy['tp_ratio']} / SL={policy['sl_ratio']}"
        )

        print(f"[{idx + 1}/{len(policies)}] Reconstructing {label}")

        try:
            selected, metadata = reconstruct_policy(policy, loads_cache)
            metadata["policy_index"] = int(idx)
            metadata["source_kind"] = policy["source_kind"]
            metadata["strategy"] = policy["strategy"]

            comp = summarize_selected(selected, metadata)
            comp["policy_index"] = int(idx)

            selected["policy_index"] = int(idx)

            selected_frames.append(selected)
            composition_frames.append(comp)

            policy_results.append({
                "policy_index": int(idx),
                "status": "RECONCILED",
                "source_kind": policy["source_kind"],
                "strategy": policy["strategy"],
                "filter": policy["filter"],
                "horizon": policy["horizon"],
                "top_pct": policy["top_pct"],
                "pair_cap": policy["pair_cap_str"],
                "tp_ratio": policy["tp_ratio"],
                "sl_ratio": policy["sl_ratio"],
                "selected_reported": int(policy["selected_test"]),
                "total_reported": float(policy["total_net_return_test"]),
                "selected_reconstructed": metadata["reconstructed_selected"],
                "total_reconstructed": metadata["reconstructed_total"],
                "total_diff": metadata["total_diff"],
                "ranking_mode_chosen": metadata["ranking_mode_chosen"],
                "final_mode_chosen": metadata["final_mode_chosen"],
                "tab_path": metadata["paths"]["TAB"],
                "xgb_path": metadata["paths"]["XGB"],
                "rf_path": metadata["paths"]["RF"],
            })

            print("  OK — fully reconciled")

        except Exception as exc:
            msg = str(exc)

            failures.append({
                "policy_index": int(idx),
                "status": "FAILED_RECONCILIATION",
                "source_kind": policy.get("source_kind"),
                "strategy": policy.get("strategy"),
                "filter": policy.get("filter"),
                "horizon": policy.get("horizon"),
                "top_pct": policy.get("top_pct"),
                "pair_cap": policy.get("pair_cap_str"),
                "tp_ratio": policy.get("tp_ratio"),
                "sl_ratio": policy.get("sl_ratio"),
                "selected_reported": policy.get("selected_test"),
                "total_reported": policy.get("total_net_return_test"),
                "error": msg[:4000],
            })

            print("  FAILED — not trusted")
            print("  ", msg[:500])

    selected_all = pd.concat(selected_frames, ignore_index=True, sort=False) if selected_frames else pd.DataFrame()
    composition_all = pd.concat(composition_frames, ignore_index=True, sort=False) if composition_frames else pd.DataFrame()
    policy_results_df = pd.DataFrame(policy_results)
    failures_df = pd.DataFrame(failures)

    selected_path = OUT_DIR / "phase_b_v4_selected_trades_reconciled.csv"
    composition_path = OUT_DIR / "phase_b_v4_composition_by_policy_reconciled.csv"
    policy_results_path = OUT_DIR / "phase_b_v4_policy_reconciliation.csv"
    failures_path = OUT_DIR / "phase_b_v4_failed_reconstructions.csv"

    selected_all.to_csv(selected_path, index=False)
    composition_all.to_csv(composition_path, index=False)
    policy_results_df.to_csv(policy_results_path, index=False)
    failures_df.to_csv(failures_path, index=False)

    manifest = {
        "status": "ok",
        "method": "strict fail-closed reconstruction from raw prediction parquet + raw SQLite market_snapshots",
        "db_path": str(DB_PATH),
        "phase_b_v2_dir": str(PHASE_B_V2_DIR),
        "output_dir": str(OUT_DIR),
        "policies_attempted": int(len(policies)),
        "policies_reconciled": int(len(policy_results_df)),
        "policies_failed": int(len(failures_df)),
        "selected_trades_rows": int(len(selected_all)),
        "composition_rows": int(len(composition_all)),
        "guardrails": [
            "exact model/file/schema validation",
            "no wrong model token in selected prediction path",
            "exact filter/horizon matching by filename or data columns",
            "common event universe required across TAB/XGB/RF",
            "duplicate event_key rejected",
            "policy selected count must match",
            "valid sims count must match selected count",
            "sum of reconstructed net return must equal reported total_net_return_test within 1e-6",
            "unreconciled policies are excluded from final composition table",
        ],
        "outputs": {
            "selected_trades": str(selected_path),
            "composition": str(composition_path),
            "policy_reconciliation": str(policy_results_path),
            "failures": str(failures_path),
        },
    }

    manifest_path = OUT_DIR / "phase_b_v4_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    summary_lines = []
    summary_lines.append("PHASE B V4 — FAIL-CLOSED TWO_OF_THREE COMPOSITION")
    summary_lines.append("=" * 140)
    summary_lines.append("")
    summary_lines.append("Purpose")
    summary_lines.append("-" * 140)
    summary_lines.append("V4 decomposes TWO_OF_THREE only when the reconstruction fully reconciles to the official policy aggregate.")
    summary_lines.append("If selected count, valid simulations, or total_net_return do not match exactly, the policy is rejected from final composition.")
    summary_lines.append("")
    summary_lines.append("Reconciliation status")
    summary_lines.append("-" * 140)
    summary_lines.append(f"Policies attempted:   {len(policies)}")
    summary_lines.append(f"Policies reconciled:  {len(policy_results_df)}")
    summary_lines.append(f"Policies failed:      {len(failures_df)}")
    summary_lines.append("")
    summary_lines.append("Policy reconciliation table")
    summary_lines.append("-" * 140)

    if not policy_results_df.empty:
        show_cols = [
            "source_kind",
            "strategy",
            "filter",
            "horizon",
            "top_pct",
            "pair_cap",
            "tp_ratio",
            "sl_ratio",
            "selected_reported",
            "selected_reconstructed",
            "total_reported",
            "total_reconstructed",
            "total_diff",
            "ranking_mode_chosen",
            "final_mode_chosen",
        ]
        summary_lines.append(policy_results_df[show_cols].to_string(index=False))
    else:
        summary_lines.append("NO RECONCILED POLICIES")

    summary_lines.append("")
    summary_lines.append("Composition table — reconciled only")
    summary_lines.append("-" * 140)

    if not composition_all.empty:
        show_cols = [
            "source_kind",
            "strategy",
            "filter",
            "horizon",
            "top_pct",
            "pair_cap",
            "tp_ratio",
            "sl_ratio",
            "policy_selected_reported",
            "policy_total_reported",
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

        composition_sorted = composition_all.sort_values(
            ["policy_total_reported", "combo_total_net"],
            ascending=[False, False],
        )

        summary_lines.append(composition_sorted[show_cols].to_string(index=False))
    else:
        summary_lines.append("NO TRUSTED COMPOSITION ROWS")

    summary_lines.append("")
    summary_lines.append("Failures")
    summary_lines.append("-" * 140)

    if not failures_df.empty:
        show_cols = [
            "source_kind",
            "strategy",
            "filter",
            "horizon",
            "top_pct",
            "pair_cap",
            "tp_ratio",
            "sl_ratio",
            "selected_reported",
            "total_reported",
            "error",
        ]
        summary_lines.append(failures_df[show_cols].to_string(index=False))
    else:
        summary_lines.append("NONE")

    summary_lines.append("")
    summary_lines.append("Outputs")
    summary_lines.append("-" * 140)
    summary_lines.append(str(composition_path))
    summary_lines.append(str(selected_path))
    summary_lines.append(str(policy_results_path))
    summary_lines.append(str(failures_path))
    summary_lines.append(str(manifest_path))

    summary_path = OUT_DIR / "phase_b_v4_summary_for_upload.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("")
    print("DONE")
    print("Summary:", summary_path)
    print("Composition:", composition_path)
    print("Policy reconciliation:", policy_results_path)
    print("Failures:", failures_path)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
