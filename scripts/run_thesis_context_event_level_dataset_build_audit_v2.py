from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("THESIS_ROOT", r"E:\Projects\Final Project\memecoin_trader"))
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = ROOT / "data" / "audits" / f"thesis_context_event_level_dataset_build_audit_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

DB = ROOT / "data" / "trader.db"

EVENT_BUCKET_MINUTES = int(os.environ.get("THESIS_EVENT_BUCKET_MINUTES", "120"))
HORIZON_MINUTES = int(os.environ.get("THESIS_HORIZON_MINUTES", "240"))

TP_RATIO = float(os.environ.get("THESIS_TP_RATIO", "2.0308"))
SL_RATIO = float(os.environ.get("THESIS_SL_RATIO", "0.80"))
TP_RETURN = TP_RATIO - 1.0
SL_RETURN = SL_RATIO - 1.0

CONTEXT_WINDOWS_HOURS = [24, 168]


REQUESTED_COLS = {
    "coins": [
        "id", "symbol", "name", "chain", "pair_address"
    ],
    "signals": [
        "id", "coin_id", "symbol", "timestamp",
        "score", "confidence", "signal_score", "signal_confidence",
        "signal_type", "model_source", "signal_model_source",
        "reason", "signal_reason", "action"
    ],
    "market_snapshots": [
        "id", "coin_id", "symbol", "chain", "pair_address", "timestamp",
        "price", "price_usd", "liquidity", "liquidity_usd",
        "volume", "volume_24h", "latest_volume_24h",
        "fdv", "txns_buys", "txns_sells", "txns_total",
        "buy_ratio",
        "price_change_m5", "price_change_h1", "price_change_h6", "price_change_h24",
        "whale_score", "market_whale_score_pool_proxy",
        "filter_status", "drop_reason", "source_query"
    ],
    "raw_provider_payloads": [
        "id", "timestamp", "provider", "source", "symbol", "chain",
        "pair_address", "url"
    ],
    "whale_alerts": [
        "id", "coin_id", "symbol", "chain", "pair_address", "timestamp",
        "alert_type", "whale_score", "volume", "volume_usd",
        "price_impact_pct", "tx_count", "description"
    ],
    "gemini_decisions": [
        "id", "coin_id", "symbol", "timestamp",
        "decision", "action", "risk_score", "confidence",
        "prompt_summary", "strategy_type"
    ],
    "paper_trades": [
        "id", "coin_id", "symbol", "chain", "timestamp",
        "side", "price", "amount", "value", "fee", "slippage",
        "pnl", "gross_pnl", "realized_pnl", "net_roi_pct",
        "status", "reason", "source", "decision_ref_id", "position_id"
    ],
    "sentiment_records": [
        "id", "timestamp", "url", "source", "provider", "title",
        "sentiment_score", "relevance", "symbol", "coin_id"
    ],
}


def norm(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "na"}:
        return ""
    return s.upper()


def norm_symbol_full(x: Any) -> str:
    return norm(x)


def norm_symbol_base(x: Any) -> str:
    s = norm(x)
    if "/" in s:
        return s.split("/", 1)[0].strip()
    return s


def connect_ro() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def table_cols(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]


def table_count(con: sqlite3.Connection, table: str) -> int:
    return int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def read_selected(con: sqlite3.Connection, table: str, requested: list[str]) -> pd.DataFrame:
    if not table_exists(con, table):
        return pd.DataFrame()

    available = table_cols(con, table)
    use_cols = [c for c in requested if c in available]
    if not use_cols:
        return pd.DataFrame()

    sql_cols = ", ".join([f'"{c}"' for c in use_cols])
    return pd.read_sql_query(f'SELECT {sql_cols} FROM "{table}"', con)


def first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_coin_bridge(coins: pd.DataFrame) -> dict[str, Any]:
    c = coins.copy()
    c["coin_id_bridge"] = pd.to_numeric(c["id"], errors="coerce").astype("Int64")
    c["symbol_full_norm"] = c["symbol"].map(norm_symbol_full) if "symbol" in c.columns else ""
    c["symbol_base_norm"] = c["symbol"].map(norm_symbol_base) if "symbol" in c.columns else ""
    c["pair_norm"] = c["pair_address"].map(norm) if "pair_address" in c.columns else ""

    id_map = {
        int(r["coin_id_bridge"]): int(r["coin_id_bridge"])
        for _, r in c.dropna(subset=["coin_id_bridge"]).iterrows()
    }

    pair_map = {}
    if "pair_norm" in c.columns:
        for _, r in c.iterrows():
            p = r.get("pair_norm", "")
            cid = r.get("coin_id_bridge")
            if p and pd.notna(cid):
                pair_map[p] = int(cid)

    def unique_symbol_map(col: str) -> dict[str, int]:
        d: dict[str, set[int]] = {}
        for _, r in c.iterrows():
            key = r.get(col, "")
            cid = r.get("coin_id_bridge")
            if key and pd.notna(cid):
                d.setdefault(key, set()).add(int(cid))
        return {k: next(iter(v)) for k, v in d.items() if len(v) == 1}

    return {
        "coins": c,
        "id_map": id_map,
        "pair_map": pair_map,
        "symbol_full_map": unique_symbol_map("symbol_full_norm"),
        "symbol_base_map": unique_symbol_map("symbol_base_norm"),
    }


def resolve_table(df: pd.DataFrame, bridge: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["canonical_coin_id"] = pd.NA
    out["canonical_bridge_method"] = "UNRESOLVED"

    def apply_mapping(mask: pd.Series, mapped: pd.Series, method: str) -> None:
        idx = out.index[mask & out["canonical_coin_id"].isna()]
        if len(idx):
            out.loc[idx, "canonical_coin_id"] = mapped.loc[idx]
            out.loc[idx, "canonical_bridge_method"] = method

    if "coin_id" in out.columns:
        coin_num = pd.to_numeric(out["coin_id"], errors="coerce")
        mapped = coin_num.map(lambda x: bridge["id_map"].get(int(x), pd.NA) if pd.notna(x) else pd.NA)
        apply_mapping(mapped.notna(), mapped, "coin_id_to_coins_id")

    if "pair_address" in out.columns:
        pair_norm = out["pair_address"].map(norm)
        mapped = pair_norm.map(lambda x: bridge["pair_map"].get(x, pd.NA))
        apply_mapping(mapped.notna(), mapped, "pair_address_to_coins_pair_address")

    if "symbol" in out.columns:
        full = out["symbol"].map(norm_symbol_full)
        mapped_full = full.map(lambda x: bridge["symbol_full_map"].get(x, pd.NA))
        apply_mapping(mapped_full.notna(), mapped_full, "symbol_full_unique_to_coins_symbol")

        base = out["symbol"].map(norm_symbol_base)
        mapped_base = base.map(lambda x: bridge["symbol_base_map"].get(x, pd.NA))
        apply_mapping(mapped_base.notna(), mapped_base, "symbol_base_unique_to_coins_symbol_base")

    out["canonical_coin_id"] = pd.to_numeric(out["canonical_coin_id"], errors="coerce").astype("Int64")

    if "timestamp" in out.columns:
        out["event_time_utc"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    else:
        out["event_time_utc"] = pd.NaT

    return out


def build_signal_events(signals: pd.DataFrame) -> pd.DataFrame:
    s = signals.copy()
    s = s[s["canonical_coin_id"].notna() & s["event_time_utc"].notna()].copy()
    s["event_bucket"] = s["event_time_utc"].dt.floor(f"{EVENT_BUCKET_MINUTES}min")

    key_cols = ["canonical_coin_id", "event_bucket"]
    g = s.groupby(key_cols, dropna=False)

    events = g.agg(
        candidate_event_start_utc=("event_time_utc", "min"),
        candidate_event_time_utc=("event_time_utc", "min"),
        candidate_event_end_utc=("event_time_utc", "max"),
        signal_rows=("event_time_utc", "size"),
    ).reset_index()

    events["candidate_event_id"] = (
        "SIGEVT|"
        + events["canonical_coin_id"].astype(str)
        + "|"
        + events["event_bucket"].astype(str)
    )

    for col in ["score", "signal_score", "confidence", "signal_confidence"]:
        if col in s.columns:
            s[col] = pd.to_numeric(s[col], errors="coerce")
            agg = g[col].agg(["mean", "max", "min"]).reset_index()
            for stat in ["mean", "max", "min"]:
                events[f"{col}_{stat}"] = agg[stat]

    def top_values(x: pd.Series) -> str:
        vals = x.dropna().astype(str)
        if vals.empty:
            return ""
        return "|".join(vals.value_counts().head(5).index.tolist())

    for col in ["signal_type", "model_source", "signal_model_source", "reason", "signal_reason", "action"]:
        if col in s.columns:
            agg = g[col].apply(top_values).reset_index(name=f"{col}_top_values")
            events = events.merge(agg, on=key_cols, how="left")

    events = events.sort_values("candidate_event_time_utc").reset_index(drop=True)
    n = len(events)

    if n:
        train_end = int(n * 0.70)
        valid_end = int(n * 0.85)
        events["chronological_split"] = "test"
        events.loc[:train_end - 1, "chronological_split"] = "train"
        events.loc[train_end:valid_end - 1, "chronological_split"] = "validation"

    return events


def prepare_market(market: pd.DataFrame) -> pd.DataFrame:
    m = market.copy()
    m = m[m["canonical_coin_id"].notna() & m["event_time_utc"].notna()].copy()

    price_col = first_existing(m, ["price", "price_usd"])
    if price_col is None:
        raise SystemExit("No usable market price column found.")

    m["market_price"] = pd.to_numeric(m[price_col], errors="coerce")
    m = m[m["market_price"].notna() & (m["market_price"] > 0)].copy()
    m = m.sort_values(["canonical_coin_id", "event_time_utc"]).reset_index(drop=True)

    return m



def datetime_ns_array(series: pd.Series) -> np.ndarray:
    """
    Robust UTC nanosecond conversion.

    This avoids pandas/backend-dependent timestamp unit issues that can turn
    minute-scale differences into multi-decade artifacts.
    """
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    dt_naive_utc = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    return dt_naive_utc.astype("datetime64[ns]").astype("int64").to_numpy()


def timestamp_ns(value: Any) -> int:
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.value)

def attach_market_and_labels(events: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()

    feature_cols = [
        "market_price",
        "liquidity", "liquidity_usd",
        "volume", "volume_24h", "latest_volume_24h",
        "fdv", "txns_buys", "txns_sells", "txns_total",
        "buy_ratio",
        "price_change_m5", "price_change_h1", "price_change_h6", "price_change_h24",
        "whale_score", "market_whale_score_pool_proxy",
    ]
    feature_cols = [c for c in feature_cols if c in market.columns]

    for c in feature_cols:
        out[f"asof_{c}"] = np.nan

    label_cols = [
        "label_available",
        "market_asof_age_minutes",
        "future_rows_4h",
        "future_return_end_4h",
        "future_return_max_4h",
        "future_return_min_4h",
        "tp_hit_4h",
        "sl_hit_4h",
        "first_tp_minutes",
        "first_sl_minutes",
        "label_x2_sl_4h",
    ]
    for c in label_cols:
        out[c] = np.nan

    out["label_available"] = False
    out["tp_hit_4h"] = False
    out["sl_hit_4h"] = False
    out["label_x2_sl_4h"] = "NO_LABEL"

    market_by_coin = {
        int(cid): grp.sort_values("event_time_utc").reset_index(drop=True)
        for cid, grp in market.groupby("canonical_coin_id")
    }

    for idx, r in out.iterrows():
        cid = int(r["canonical_coin_id"])
        ev_time = r["candidate_event_time_utc"]

        if cid not in market_by_coin or pd.isna(ev_time):
            continue

        m = market_by_coin[cid]
        times = datetime_ns_array(m["event_time_utc"])
        prices = m["market_price"].to_numpy(dtype=float)

        ev_ns = timestamp_ns(ev_time)
        horizon_ns = (pd.Timestamp(ev_time) + pd.Timedelta(minutes=HORIZON_MINUTES)).value

        asof_idx = np.searchsorted(times, ev_ns, side="right") - 1
        if asof_idx < 0:
            continue

        base_price = prices[asof_idx]
        if not np.isfinite(base_price) or base_price <= 0:
            continue

        asof_time = m.iloc[asof_idx]["event_time_utc"]
        out.at[idx, "market_asof_age_minutes"] = (
            pd.Timestamp(ev_time) - pd.Timestamp(asof_time)
        ).total_seconds() / 60.0

        for c in feature_cols:
            if c in m.columns:
                out.at[idx, f"asof_{c}"] = pd.to_numeric(pd.Series([m.iloc[asof_idx][c]]), errors="coerce").iloc[0]

        start = np.searchsorted(times, ev_ns, side="right")
        end = np.searchsorted(times, horizon_ns, side="right")

        if end <= start:
            continue

        future_prices = prices[start:end]
        future_times = times[start:end]
        valid_future_mask = np.isfinite(future_prices) & (future_prices > 0)
        future_prices = future_prices[valid_future_mask]
        future_times = future_times[valid_future_mask]

        if len(future_prices) == 0:
            continue

        returns = future_prices / base_price - 1.0

        out.at[idx, "label_available"] = True
        out.at[idx, "future_rows_4h"] = int(len(returns))
        out.at[idx, "future_return_end_4h"] = float(returns[-1])
        out.at[idx, "future_return_max_4h"] = float(np.max(returns))
        out.at[idx, "future_return_min_4h"] = float(np.min(returns))

        tp_positions = np.where(returns >= TP_RETURN)[0]
        sl_positions = np.where(returns <= SL_RETURN)[0]

        tp_hit = len(tp_positions) > 0
        sl_hit = len(sl_positions) > 0

        out.at[idx, "tp_hit_4h"] = bool(tp_hit)
        out.at[idx, "sl_hit_4h"] = bool(sl_hit)

        first_tp = int(tp_positions[0]) if tp_hit else None
        first_sl = int(sl_positions[0]) if sl_hit else None

        if first_tp is not None:
            out.at[idx, "first_tp_minutes"] = (
                pd.Timestamp(future_times[first_tp]) - pd.Timestamp(ev_ns)
            ).total_seconds() / 60.0

        if first_sl is not None:
            out.at[idx, "first_sl_minutes"] = (
                pd.Timestamp(future_times[first_sl]) - pd.Timestamp(ev_ns)
            ).total_seconds() / 60.0

        if tp_hit and (not sl_hit or first_tp <= first_sl):
            out.at[idx, "label_x2_sl_4h"] = "WINNER"
        elif sl_hit and (not tp_hit or first_sl < first_tp):
            out.at[idx, "label_x2_sl_4h"] = "LOSER"
        else:
            out.at[idx, "label_x2_sl_4h"] = "FLAT"

    return out


def attach_context_counts(events: pd.DataFrame, context: pd.DataFrame, prefix: str, numeric_cols: list[str]) -> pd.DataFrame:
    out = events.copy()

    context = context[context["canonical_coin_id"].notna() & context["event_time_utc"].notna()].copy()
    context = context.sort_values(["canonical_coin_id", "event_time_utc"]).reset_index(drop=True)

    numeric_cols = [c for c in numeric_cols if c in context.columns]
    for c in numeric_cols:
        context[c] = pd.to_numeric(context[c], errors="coerce")

    for h in CONTEXT_WINDOWS_HOURS:
        out[f"{prefix}_count_past_{h}h"] = 0
        for c in numeric_cols:
            out[f"{prefix}_{c}_mean_past_{h}h"] = np.nan
            out[f"{prefix}_{c}_sum_past_{h}h"] = np.nan

    ctx_by_coin = {
        int(cid): grp.sort_values("event_time_utc").reset_index(drop=True)
        for cid, grp in context.groupby("canonical_coin_id")
    }

    for idx, r in out.iterrows():
        cid = int(r["canonical_coin_id"])
        ev_time = r["candidate_event_time_utc"]

        if cid not in ctx_by_coin or pd.isna(ev_time):
            continue

        cdf = ctx_by_coin[cid]
        times = datetime_ns_array(cdf["event_time_utc"])
        ev_ns = timestamp_ns(ev_time)

        for h in CONTEXT_WINDOWS_HOURS:
            start_ns = timestamp_ns(pd.Timestamp(ev_time) - pd.Timedelta(hours=h))
            lo = np.searchsorted(times, start_ns, side="left")
            hi = np.searchsorted(times, ev_ns, side="right")

            count = int(max(0, hi - lo))
            out.at[idx, f"{prefix}_count_past_{h}h"] = count

            if count > 0:
                window = cdf.iloc[lo:hi]
                for c in numeric_cols:
                    vals = pd.to_numeric(window[c], errors="coerce")
                    if vals.notna().sum() > 0:
                        out.at[idx, f"{prefix}_{c}_mean_past_{h}h"] = float(vals.mean())
                        out.at[idx, f"{prefix}_{c}_sum_past_{h}h"] = float(vals.sum())

    return out


def summarize_dataset(df: pd.DataFrame) -> dict[str, Any]:
    label_df = df[df["label_available"] == True].copy()

    summary = {
        "candidate_event_rows": int(len(df)),
        "label_available_rows": int(len(label_df)),
        "label_available_rate": float(len(label_df) / len(df)) if len(df) else None,
        "unique_canonical_coins": int(df["canonical_coin_id"].nunique(dropna=True)),
        "chronological_split_counts": df["chronological_split"].value_counts(dropna=False).to_dict()
        if "chronological_split" in df.columns else {},
        "label_x2_sl_4h_counts": label_df["label_x2_sl_4h"].value_counts(dropna=False).to_dict()
        if not label_df.empty else {},
    }

    for col in [
        "raw_payload_count_past_24h",
        "raw_payload_count_past_168h",
        "pool_flow_count_past_24h",
        "pool_flow_count_past_168h",
        "gemini_count_past_24h",
        "gemini_count_past_168h",
    ]:
        if col in df.columns:
            summary[f"{col}_coverage_rows"] = int((df[col] > 0).sum())
            summary[f"{col}_coverage_rate"] = float((df[col] > 0).mean()) if len(df) else None

    return summary


def main() -> None:
    summary = {
        "classification": "THESIS_CONTEXT_EVENT_LEVEL_DATASET_BUILD_AUDIT_COMPLETED",
        "root": str(ROOT),
        "output_root": str(OUT),
        "safety": {
            "read_only_audit": True,
            "training_run": False,
            "backtest_run": False,
            "trader_db_mutated": False,
            "wallet_connected": False,
            "live_trading_enabled": False,
            "new_llm_calls": False,
        },
        "event_bucket_minutes": EVENT_BUCKET_MINUTES,
        "horizon_minutes": HORIZON_MINUTES,
        "tp_ratio": TP_RATIO,
        "sl_ratio": SL_RATIO,
        "notes": [
            "candidate event timestamp is the first signal timestamp inside the event bucket",
            "future labels start strictly after candidate_event_time_utc",
            "context windows include only records at or before candidate_event_time_utc",
            "whale_alerts are treated only as pool-flow/activity proxy, not wallet-level whale evidence",
            "sentiment_records are excluded from token-level context because canonical bridge resolution is 0%",
        ],
    }

    con = connect_ro()

    loaded = {}
    counts = {}
    for table, requested in REQUESTED_COLS.items():
        if table_exists(con, table):
            counts[table] = table_count(con, table)
            loaded[table] = read_selected(con, table, requested)

    con.close()

    if "coins" not in loaded or loaded["coins"].empty:
        raise SystemExit("coins table required")
    if "signals" not in loaded or loaded["signals"].empty:
        raise SystemExit("signals table required")
    if "market_snapshots" not in loaded or loaded["market_snapshots"].empty:
        raise SystemExit("market_snapshots table required")

    bridge = build_coin_bridge(loaded["coins"])

    resolved = {}
    for table, df in loaded.items():
        if table == "coins":
            continue
        resolved[table] = resolve_table(df, bridge)

    signal_events = build_signal_events(resolved["signals"])
    market = prepare_market(resolved["market_snapshots"])

    dataset = attach_market_and_labels(signal_events, market)

    if "raw_provider_payloads" in resolved:
        dataset = attach_context_counts(
            dataset,
            resolved["raw_provider_payloads"],
            "raw_payload",
            numeric_cols=[],
        )

    if "whale_alerts" in resolved:
        dataset = attach_context_counts(
            dataset,
            resolved["whale_alerts"],
            "pool_flow",
            numeric_cols=["whale_score", "volume", "volume_usd", "price_impact_pct", "tx_count"],
        )

    if "gemini_decisions" in resolved:
        dataset = attach_context_counts(
            dataset,
            resolved["gemini_decisions"],
            "gemini",
            numeric_cols=["risk_score", "confidence"],
        )

    dataset_summary = summarize_dataset(dataset)

    # Quality tables.
    resolution_rows = []
    for table, df in resolved.items():
        resolution_rows.append({
            "table": table,
            "rows_loaded": int(len(df)),
            "resolved_rows": int(df["canonical_coin_id"].notna().sum()),
            "resolved_rate": float(df["canonical_coin_id"].notna().mean()) if len(df) else None,
            "unique_canonical_coins": int(df["canonical_coin_id"].nunique(dropna=True)),
        })

    resolution_df = pd.DataFrame(resolution_rows).sort_values("table")
    resolution_df.to_csv(OUT / "00_bridge_resolution_used_for_dataset.csv", index=False, encoding="utf-8-sig")

    label_df = dataset.groupby(["chronological_split", "label_available", "label_x2_sl_4h"], dropna=False).size().reset_index(name="rows")
    label_df.to_csv(OUT / "01_label_distribution_by_split.csv", index=False, encoding="utf-8-sig")

    context_cov_rows = []
    for col in dataset.columns:
        if col.endswith("_count_past_24h") or col.endswith("_count_past_168h"):
            context_cov_rows.append({
                "feature": col,
                "rows_with_feature_positive": int((dataset[col] > 0).sum()),
                "coverage_rate": float((dataset[col] > 0).mean()) if len(dataset) else None,
                "mean_count": float(dataset[col].mean()) if len(dataset) else None,
                "max_count": int(dataset[col].max()) if len(dataset) else None,
            })
    context_cov_df = pd.DataFrame(context_cov_rows)
    context_cov_df.to_csv(OUT / "02_context_feature_coverage.csv", index=False, encoding="utf-8-sig")

    # Save dataset.
    dataset_path_csv = OUT / "03_event_level_context_rebuild_dataset.csv"
    dataset.to_csv(dataset_path_csv, index=False, encoding="utf-8-sig")

    dataset_path_parquet = OUT / "03_event_level_context_rebuild_dataset.parquet"
    parquet_status = "not_written"
    try:
        dataset.to_parquet(dataset_path_parquet, index=False)
        parquet_status = "written"
    except Exception as exc:
        parquet_status = f"not_written: {exc!r}"

    summary.update({
        "db_table_counts": counts,
        "dataset_summary": dataset_summary,
        "parquet_status": parquet_status,
        "outputs": {
            "bridge_resolution_used_for_dataset": str(OUT / "00_bridge_resolution_used_for_dataset.csv"),
            "label_distribution_by_split": str(OUT / "01_label_distribution_by_split.csv"),
            "context_feature_coverage": str(OUT / "02_context_feature_coverage.csv"),
            "event_level_dataset_csv": str(dataset_path_csv),
            "event_level_dataset_parquet": str(dataset_path_parquet) if parquet_status == "written" else parquet_status,
        },
    })

    with open(OUT / "thesis_context_event_level_dataset_build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Thesis Context Event-Level Dataset Build Audit")
    lines.append("")
    lines.append(f"Output root: `{OUT}`")
    lines.append("")
    lines.append("## Safety")
    lines.append("- Read-only audit")
    lines.append("- No training")
    lines.append("- No backtest")
    lines.append("- No trader.db mutation")
    lines.append("- No wallet")
    lines.append("- No live trading")
    lines.append("- No new LLM calls")
    lines.append("")
    lines.append("## Dataset design")
    lines.append("- Candidate source: `signals`")
    lines.append(f"- Event unit: `canonical_coin_id + {EVENT_BUCKET_MINUTES}-minute bucket`")
    lines.append(f"- Candidate timestamp: first signal timestamp inside event bucket")
    lines.append(f"- Label source: future `market_snapshots` over {HORIZON_MINUTES} minutes")
    lines.append(f"- TP ratio: {TP_RATIO}")
    lines.append(f"- SL ratio: {SL_RATIO}")
    lines.append("- Context windows: 24h and 168h before candidate timestamp")
    lines.append("- `whale_alerts` used only as pool-flow/activity proxy")
    lines.append("- `sentiment_records` excluded because canonical token bridge resolution is 0%")
    lines.append("")
    lines.append("## Dataset summary")
    for k, v in dataset_summary.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Bridge resolution used")
    for _, r in resolution_df.iterrows():
        lines.append(
            f"- `{r['table']}`: rows={int(r['rows_loaded']):,}, "
            f"resolved={int(r['resolved_rows']):,}, "
            f"resolved_rate={float(r['resolved_rate'])*100:.2f}%, "
            f"unique_coins={int(r['unique_canonical_coins'])}"
        )
    lines.append("")
    lines.append("## Context coverage")
    if context_cov_df.empty:
        lines.append("No context coverage features produced.")
    else:
        for _, r in context_cov_df.iterrows():
            lines.append(
                f"- `{r['feature']}`: coverage_rows={int(r['rows_with_feature_positive']):,}, "
                f"coverage_rate={float(r['coverage_rate'])*100:.2f}%, "
                f"mean_count={float(r['mean_count']):.3f}, "
                f"max_count={int(r['max_count'])}"
            )
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "This file is a clean event-level rebuild dataset candidate. It is not a model result and not a backtest. "
        "It is suitable for the next audit only if label availability, class balance, chronological split coverage, "
        "and context coverage are acceptable."
    )
    lines.append("")
    lines.append("## Output files")
    for _, path in summary["outputs"].items():
        lines.append(f"- `{Path(str(path)).name if isinstance(path, str) else path}`")

    md = OUT / "thesis_context_event_level_dataset_build_summary.md"
    md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(OUT / "thesis_context_event_level_dataset_build_summary.json"),
        "summary_md": str(md),
        "dataset_csv": str(dataset_path_csv),
        "parquet_status": parquet_status,
    }, indent=2, ensure_ascii=False))
    print()
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
