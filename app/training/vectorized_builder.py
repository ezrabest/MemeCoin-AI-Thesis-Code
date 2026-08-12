"""
Vectorized training dataset pipeline (pandas groupby / merge_asof).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import pandas as pd

from ..database import DB_PATH, get_db
from .config import (
    get_round_trip_fee_pct,
    get_whale_wave_aggressive_threshold,
    get_whale_wave_normal_threshold,
)
from .snapshot_features import HISTORICAL_FEATURE_COLUMNS, compute_snapshot_historical_features
from .wave_engine import (
    add_position_sizing_labels,
    add_whale_wave_score,
    attach_historical_features,
    compute_future_labels,
)

log = logging.getLogger("training.vectorized_builder")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TRAINING_DIR = DATA_DIR / "training"
SUMMARY_FILENAME = "training_dataset_summary.json"


def _load_snapshots_df(conn) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT id, coin_id, timestamp, chain, pair_address, price, liquidity,
               volume_24h, fdv, txns_buys, txns_sells, buy_ratio, whale_score,
               price_change_h1, price_change_h24
        FROM market_snapshots
        ORDER BY coin_id, timestamp
        """,
        conn,
    )


def _load_signals_df(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM signals ORDER BY id", conn)


def _load_llm_df(conn) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM gemini_decisions ORDER BY id", conn)


def _load_coins_df(conn) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT id, symbol, chain, pair_address FROM coins",
        conn,
    )


def _parse_features_json(series: pd.Series) -> pd.DataFrame:
    def _one(val: Any) -> dict:
        if isinstance(val, dict):
            return val
        if not val or (isinstance(val, float) and np.isnan(val)):
            return {}
        try:
            parsed = json.loads(val)
            return parsed if isinstance(parsed, dict) else {}
        except (TypeError, json.JSONDecodeError):
            return {}

    parsed = series.map(_one)
    return pd.DataFrame(parsed.tolist(), index=series.index)


def _build_signal_events(signals: pd.DataFrame, coins: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return pd.DataFrame()
    df = signals.copy()
    coin_lookup = coins.rename(columns={
        "id": "coin_id",
        "symbol": "coin_symbol",
        "chain": "coin_chain",
        "pair_address": "coin_pair_address",
    })
    df = df.merge(coin_lookup, on="coin_id", how="left")
    feats = _parse_features_json(df.get("features_json", pd.Series(index=df.index)))
    for col in ("sentiment_score", "volume_24h", "liquidity_usd", "cluster_label"):
        if col in feats.columns:
            df[col] = feats[col]
    df["event_timestamp"] = df["timestamp"]
    df["signal_id"] = df["id"]
    df["signal_action"] = df["signal_type"]
    df["signal_confidence"] = pd.to_numeric(df.get("confidence", df.get("score")), errors="coerce")
    df["symbol"] = df.get("symbol", pd.Series(dtype=object)).fillna(df.get("coin_symbol"))
    df["pair_address"] = df.get("pair_address", df.get("coin_pair_address"))
    df["chain"] = df.get("chain", pd.Series(dtype=object)).fillna(df.get("coin_chain"))
    df["source_kind"] = "signal"
    return df


def _build_llm_events(llm: pd.DataFrame, coins: pd.DataFrame) -> pd.DataFrame:
    if llm.empty:
        return pd.DataFrame()
    df = llm.copy()
    coin_lookup = coins.rename(columns={
        "id": "coin_id",
        "symbol": "coin_symbol",
        "chain": "coin_chain",
        "pair_address": "coin_pair_address",
    })
    df = df.merge(coin_lookup, on="coin_id", how="left")
    df["event_timestamp"] = df["timestamp"]
    df["decision_id"] = df["id"]
    df["llm_action"] = df["action"]
    df["llm_confidence"] = pd.to_numeric(df.get("confidence"), errors="coerce")
    df["reasoning"] = df.get("rationale")
    df["reasoning_summary"] = df["reasoning"].astype(str).str.slice(0, 240)
    df["response_json"] = df.get("gemini_response_json")
    df["provider"] = df.get("provider", "unknown").fillna("unknown")
    df["model_source"] = df.get("model_source", "unknown").fillna("unknown")
    df["trigger_type"] = df.get("trigger_type", "unknown").fillna("unknown")
    df["strategy_type"] = df.get("strategy_type", "unknown").fillna("unknown")
    df["symbol"] = df.get("symbol", pd.Series(dtype=object)).fillna(df.get("coin_symbol"))
    df["pair_address"] = df.get("pair_address", df.get("coin_pair_address"))
    df["chain"] = df.get("chain", pd.Series(dtype=object)).fillna(df.get("coin_chain"))
    df["source_kind"] = "llm_decision"
    return df


def _enrich_events(
    events: pd.DataFrame,
    featured_snaps: pd.DataFrame,
    snaps: pd.DataFrame,
    fee_pct: float,
    warnings: list[str],
) -> pd.DataFrame:
    if events.empty:
        return events
    merged = attach_historical_features(events, featured_snaps, warnings)
    merged = compute_future_labels(merged, snaps, fee_pct)
    merged = add_whale_wave_score(merged)
    merged = add_position_sizing_labels(merged)
    return merged


def _df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out = df.replace({np.nan: None})
    return out.to_dict(orient="records")


def _bool_rate(series: pd.Series) -> float | None:
    valid = series.dropna()
    if valid.empty:
        return None
    return round(float(valid.astype(bool).mean()), 6)


def _count_true(series: pd.Series) -> int:
    return int(series.fillna(False).astype(bool).sum())


def _class_distribution(df: pd.DataFrame, col: str) -> dict[str, int]:
    if col not in df.columns or df.empty:
        return {}
    return {str(k): int(v) for k, v in df[col].value_counts().items()}


def build_summary_from_frames(
    signal_df: pd.DataFrame,
    llm_df: pd.DataFrame,
    model_df: pd.DataFrame,
    *,
    warnings: list[str],
    output_files: list[str],
    fee_pct: float,
    timings: dict[str, float],
) -> dict[str, Any]:
    all_df = pd.concat([signal_df, llm_df], ignore_index=True) if not signal_df.empty or not llm_df.empty else pd.DataFrame()

    def ready_count(frame: pd.DataFrame, h: str) -> int:
        col = f"future_return_{h}"
        return int(frame[col].notna().sum()) if col in frame.columns else 0

    summary: dict[str, Any] = {
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "database_path": str(DB_PATH),
        "round_trip_fee_pct": fee_pct,
        "rows_signal_total": len(signal_df),
        "rows_signal_ready_15m": ready_count(signal_df, "15m"),
        "rows_signal_ready_1h": ready_count(signal_df, "1h"),
        "rows_signal_ready_4h": ready_count(signal_df, "4h"),
        "rows_llm_total": len(llm_df),
        "rows_llm_ready_15m": ready_count(llm_df, "15m"),
        "rows_llm_ready_1h": ready_count(llm_df, "1h"),
        "rows_llm_ready_4h": ready_count(llm_df, "4h"),
        "rows_model_ready": len(model_df),
        "pending_outcome_rows": int(all_df["pending_outcome"].sum()) if "pending_outcome" in all_df.columns else 0,
        "warnings": warnings,
        "output_files": output_files,
        "build_stage_timings": {k: round(v, 3) for k, v in timings.items()},
    }

    if not all_df.empty:
        for h in ("15m", "1h", "4h"):
            ret_col = f"future_return_{h}"
            if ret_col in all_df.columns:
                vals = all_df[ret_col].dropna()
                summary[f"average_return_{h}"] = round(float(vals.mean()), 6) if not vals.empty else None
            summary[f"positive_rate_{h}"] = _bool_rate(all_df.get(f"label_up_{h}"))
            summary[f"profitable_after_fees_rate_{h}"] = _bool_rate(all_df.get(f"label_profitable_after_fees_{h}"))
            summary[f"big_pump_rate_{h}"] = _bool_rate(all_df.get(f"big_pump_{h}"))
            summary[f"big_dump_rate_{h}"] = _bool_rate(all_df.get(f"big_dump_{h}"))
            summary[f"big_pump_count_{h}"] = _count_true(all_df.get(f"big_pump_{h}", pd.Series()))
            summary[f"big_dump_count_{h}"] = _count_true(all_df.get(f"big_dump_{h}", pd.Series()))

        summary["pump_then_dump_rate_1h"] = _bool_rate(all_df.get("pump_then_dump_1h"))
        summary["pump_then_dump_rate_4h"] = _bool_rate(all_df.get("pump_then_dump_4h"))
        summary["pump_then_dump_count_1h"] = _count_true(all_df.get("pump_then_dump_1h", pd.Series()))
        summary["pump_then_dump_count_4h"] = _count_true(all_df.get("pump_then_dump_4h", pd.Series()))

        summary["whale_wave_aggressive_threshold_used"] = get_whale_wave_aggressive_threshold()
        summary["whale_wave_normal_threshold_used"] = get_whale_wave_normal_threshold()

        if "whale_wave_score" in all_df.columns:
            wws = all_df["whale_wave_score"].dropna()
            summary["average_whale_wave_score"] = round(float(wws.mean()), 6) if not wws.empty else None
            summary["top_whale_wave_score"] = round(float(wws.max()), 6) if not wws.empty else None
            for pct in (50, 75, 90, 95, 99):
                summary[f"whale_wave_score_p{pct}"] = (
                    round(float(wws.quantile(pct / 100.0)), 6) if not wws.empty else None
                )
            for horizon in ("1h", "4h"):
                class_col = f"optimal_trade_class_{horizon}"
                if class_col in all_df.columns:
                    grouped = (
                        all_df.dropna(subset=["whale_wave_score"])
                        .groupby(class_col)["whale_wave_score"]
                        .mean()
                    )
                    summary[f"whale_wave_score_by_optimal_trade_class_{horizon}"] = {
                        str(k): round(float(v), 6) for k, v in grouped.items()
                    }
        summary["rows_with_whale_wave_features"] = int(all_df.get("has_whale_wave_history", pd.Series()).fillna(False).astype(bool).sum())
        summary["rows_missing_required_history"] = int((~all_df.get("has_whale_wave_history", pd.Series()).fillna(False).astype(bool)).sum())

        summary["optimal_trade_class_1h_distribution"] = _class_distribution(all_df, "optimal_trade_class_1h")
        summary["optimal_trade_class_4h_distribution"] = _class_distribution(all_df, "optimal_trade_class_4h")
        summary["aggressive_whale_trade_count_1h"] = int((all_df.get("optimal_trade_class_1h") == "AGGRESSIVE_WHALE_TRADE").sum())
        summary["aggressive_whale_trade_count_4h"] = int((all_df.get("optimal_trade_class_4h") == "AGGRESSIVE_WHALE_TRADE").sum())
        summary["small_probe_count_1h"] = int((all_df.get("optimal_trade_class_1h") == "SMALL_PROBE").sum())
        summary["small_probe_count_4h"] = int((all_df.get("optimal_trade_class_4h") == "SMALL_PROBE").sum())
        summary["avoid_dump_count_1h"] = int((all_df.get("optimal_trade_class_1h") == "AVOID_DUMP").sum())
        summary["avoid_dump_count_4h"] = int((all_df.get("optimal_trade_class_4h") == "AVOID_DUMP").sum())

        if "event_timestamp" in all_df.columns:
            stamps = all_df["event_timestamp"].dropna()
            summary["oldest_event_timestamp"] = str(stamps.min()) if not stamps.empty else None
            summary["latest_event_timestamp"] = str(stamps.max()) if not stamps.empty else None

    summary["total_runtime_seconds"] = round(timings.get("total", 0.0), 3)
    return summary


def build_model_ready_frame(signal_df: pd.DataFrame, llm_df: pd.DataFrame, *, include_pending: bool = False) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    if not signal_df.empty:
        sdf = signal_df.copy()
        if not include_pending and "pending_outcome" in sdf.columns:
            sdf = sdf[~sdf["pending_outcome"].fillna(True)]
        sdf["source_type"] = "signal"
        sdf["source_id"] = sdf.get("signal_id")
        frames.append(sdf)

    if not llm_df.empty:
        ldf = llm_df.copy()
        if not include_pending and "pending_outcome" in ldf.columns:
            ldf = ldf[~ldf["pending_outcome"].fillna(True)]
        ldf["source_type"] = "llm_decision"
        ldf["source_id"] = ldf.get("decision_id")
        frames.append(ldf)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    rename = {
        "future_return_15m": "target_return_15m",
        "future_return_1h": "target_return_1h",
        "future_return_4h": "target_return_4h",
        "label_profitable_after_fees_15m": "target_profitable_15m",
        "label_profitable_after_fees_1h": "target_profitable_1h",
        "label_profitable_after_fees_4h": "target_profitable_4h",
    }
    combined = combined.rename(columns={k: v for k, v in rename.items() if k in combined.columns})
    return combined


def build_training_datasets_vectorized(*, include_pending: bool = False) -> dict[str, Any]:
    warnings: list[str] = []
    fee_pct = get_round_trip_fee_pct()
    timings: dict[str, float] = {}
    t_total = time.perf_counter()

    t0 = time.perf_counter()
    with get_db() as conn:
        snap_df = _load_snapshots_df(conn)
        signals_df = _load_signals_df(conn)
        llm_df = _load_llm_df(conn)
        coins_df = _load_coins_df(conn)
    timings["snapshot_load_seconds"] = time.perf_counter() - t0
    timings["event_load_seconds"] = timings["snapshot_load_seconds"]

    t0 = time.perf_counter()
    snap_df["ts"] = pd.to_datetime(snap_df["timestamp"], utc=True, errors="coerce")
    snap_df = snap_df.dropna(subset=["ts", "coin_id"])
    featured_snaps = compute_snapshot_historical_features(snap_df, warnings)
    timings["historical_feature_calculation_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    signal_events = _build_signal_events(signals_df, coins_df)
    llm_events = _build_llm_events(llm_df, coins_df)
    timings["event_prepare_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    signal_out = _enrich_events(signal_events, featured_snaps, snap_df, fee_pct, warnings)
    timings["signal_future_and_wave_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    llm_out = _enrich_events(llm_events, featured_snaps, snap_df, fee_pct, warnings)
    timings["llm_future_and_wave_seconds"] = time.perf_counter() - t0
    timings["future_label_calculation_seconds"] = timings["signal_future_and_wave_seconds"] + timings["llm_future_and_wave_seconds"]
    timings["whale_wave_scoring_seconds"] = timings["future_label_calculation_seconds"]
    timings["position_label_calculation_seconds"] = timings["future_label_calculation_seconds"]

    model_df = build_model_ready_frame(signal_out, llm_out, include_pending=include_pending)

    t0 = time.perf_counter()
    from .dataset_builder import TRAINING_DIR as OUT_DIR, write_dataset_file

    output_files: list[str] = []
    for stem, frame in (
        ("signal_outcomes", signal_out),
        ("llm_decision_outcomes", llm_out),
        ("model_ready_dataset", model_df),
    ):
        rows = _df_to_records(frame)
        path, file_warnings = write_dataset_file(rows, stem, training_dir=OUT_DIR)
        output_files.append(path)
        warnings.extend(file_warnings)
    timings["file_write_seconds"] = time.perf_counter() - t0

    timings["total"] = time.perf_counter() - t_total
    summary = build_summary_from_frames(
        signal_out, llm_out, model_df,
        warnings=warnings,
        output_files=output_files,
        fee_pct=fee_pct,
        timings=timings,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / SUMMARY_FILENAME
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    output_files.append(str(summary_path))

    _log_timings(timings, summary)

    return {
        "database_path": str(DB_PATH),
        "signal_rows_processed": len(signal_out),
        "llm_rows_processed": len(llm_out),
        "ready_rows": len(model_df),
        "pending_rows": summary.get("pending_outcome_rows", 0),
        "output_files": output_files,
        "warnings": warnings,
        "summary": summary,
        "timings": timings,
    }


def _log_timings(timings: dict[str, float], summary: dict[str, Any]) -> None:
    log.info("Training dataset stage timings (seconds):")
    for key, val in timings.items():
        log.info("  %s: %.3f", key, val)
    log.info(
        "Training dataset build complete — model-ready=%s pending=%s summary=%s",
        summary.get("rows_model_ready"),
        summary.get("pending_outcome_rows"),
        TRAINING_DIR / SUMMARY_FILENAME,
    )
