"""
Build training-ready outcome datasets from SQLite (offline batch).
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from ..database import DB_PATH, get_db
from .outcome_labeler import (
    get_round_trip_fee_pct,
    label_outcomes,
    parse_timestamp,
)

log = logging.getLogger("training.dataset_builder")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
TRAINING_DIR = DATA_DIR / "training"
SUMMARY_FILENAME = "training_dataset_summary.json"


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_json_field(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def _summarize_text(text: str | None, limit: int = 240) -> str | None:
    if not text:
        return None
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3] + "..."


def _load_coins(conn) -> dict[int, dict[str, Any]]:
    rows = conn.execute("SELECT * FROM coins").fetchall()
    return {int(r["id"]): dict(r) for r in rows}


def _load_snapshots_by_coin(
    conn,
) -> tuple[dict[int, list[dict[str, Any]]], dict[int, list]]:
    rows = conn.execute(
        "SELECT * FROM market_snapshots ORDER BY coin_id ASC, timestamp ASC"
    ).fetchall()
    by_coin: dict[int, list[dict[str, Any]]] = {}
    times_by_coin: dict[int, list] = {}
    for row in rows:
        coin_id = row["coin_id"]
        if coin_id is None:
            continue
        cid = int(coin_id)
        snap = dict(row)
        by_coin.setdefault(cid, []).append(snap)
        times_by_coin.setdefault(cid, []).append(parse_timestamp(snap.get("timestamp")))
    return by_coin, times_by_coin


def _load_all_signals(conn) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM signals ORDER BY id ASC").fetchall()
    return [dict(r) for r in rows]


def _load_all_llm_decisions(conn) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM gemini_decisions ORDER BY id ASC").fetchall()
    return [dict(r) for r in rows]


def build_signal_outcomes(
    *,
    snapshots_by_coin: dict[int, list[dict[str, Any]]],
    snapshot_times_by_coin: dict[int, list],
    coins_by_id: dict[int, dict[str, Any]],
    fee_pct: float,
) -> list[dict[str, Any]]:
    with get_db() as conn:
        signals = _load_all_signals(conn)

    rows: list[dict[str, Any]] = []
    for sig in signals:
        coin_id = sig.get("coin_id")
        event_ts = parse_timestamp(sig.get("timestamp"))
        if coin_id is None or event_ts is None:
            continue

        coin = coins_by_id.get(int(coin_id), {})
        features = _parse_json_field(sig.get("features_json"))
        cid = int(coin_id)
        snapshots = snapshots_by_coin.get(cid, [])
        times = snapshot_times_by_coin.get(cid, [])
        outcomes = label_outcomes(snapshots, event_ts, fee_pct=fee_pct, times=times)

        row = {
            "signal_id": sig.get("id"),
            "coin_id": int(coin_id),
            "pair_address": coin.get("pair_address") or outcomes.get("pair_address"),
            "symbol": sig.get("symbol") or coin.get("symbol"),
            "chain": coin.get("chain") or outcomes.get("chain"),
            "event_timestamp": sig.get("timestamp"),
            "signal_type": sig.get("signal_type"),
            "signal_action": sig.get("signal_type"),
            "signal_confidence": _safe_float(sig.get("confidence") or sig.get("score")),
            "score": _safe_float(sig.get("score")),
            "reason": sig.get("reason"),
            "model_source": sig.get("model_source"),
            "sentiment_score": _safe_float(features.get("sentiment_score")),
            "volume_24h": _safe_float(features.get("volume_24h") or outcomes.get("volume_24h")),
            "liquidity_usd": _safe_float(features.get("liquidity_usd") or outcomes.get("liquidity_usd")),
            "txns_buys_24h": outcomes.get("txns_buys"),
            "txns_sells_24h": outcomes.get("txns_sells"),
            "buy_ratio": _safe_float(outcomes.get("buy_ratio")),
            "price_change_1h": _safe_float(outcomes.get("price_change_1h")),
            "price_change_24h": _safe_float(outcomes.get("price_change_24h")),
            "whale_score": _safe_float(outcomes.get("whale_score")),
        }
        row.update(outcomes)
        rows.append(row)
    return rows


def build_llm_decision_outcomes(
    *,
    snapshots_by_coin: dict[int, list[dict[str, Any]]],
    snapshot_times_by_coin: dict[int, list],
    coins_by_id: dict[int, dict[str, Any]],
    fee_pct: float,
) -> list[dict[str, Any]]:
    with get_db() as conn:
        decisions = _load_all_llm_decisions(conn)

    rows: list[dict[str, Any]] = []
    for dec in decisions:
        coin_id = dec.get("coin_id")
        event_ts = parse_timestamp(dec.get("timestamp"))
        if coin_id is None or event_ts is None:
            continue

        coin = coins_by_id.get(int(coin_id), {})
        cid = int(coin_id)
        snapshots = snapshots_by_coin.get(cid, [])
        times = snapshot_times_by_coin.get(cid, [])
        outcomes = label_outcomes(snapshots, event_ts, fee_pct=fee_pct, times=times)
        response = _parse_json_field(dec.get("gemini_response_json"))

        row = {
            "decision_id": dec.get("id"),
            "provider": dec.get("provider") or "unknown",
            "model_source": dec.get("model_source") or "unknown",
            "trigger_type": dec.get("trigger_type") or "unknown",
            "strategy_type": dec.get("strategy_type") or "unknown",
            "coin_id": int(coin_id),
            "pair_address": coin.get("pair_address") or outcomes.get("pair_address"),
            "symbol": dec.get("symbol") or coin.get("symbol"),
            "chain": coin.get("chain") or outcomes.get("chain"),
            "event_timestamp": dec.get("timestamp"),
            "llm_action": dec.get("action"),
            "llm_confidence": _safe_float(dec.get("confidence")),
            "risk_score": dec.get("risk_score"),
            "reasoning": dec.get("rationale"),
            "reasoning_summary": _summarize_text(dec.get("rationale")),
            "input_context_json": dec.get("input_context_json"),
            "response_json": dec.get("gemini_response_json"),
            "prompt_summary": dec.get("prompt_summary"),
            "price_usd": outcomes.get("price_usd"),
            "liquidity_usd": _safe_float(outcomes.get("liquidity_usd")),
            "volume_24h": _safe_float(outcomes.get("volume_24h")),
            "txns_buys_24h": outcomes.get("txns_buys"),
            "txns_sells_24h": outcomes.get("txns_sells"),
            "buy_ratio": _safe_float(outcomes.get("buy_ratio")),
            "price_change_1h": _safe_float(outcomes.get("price_change_1h")),
            "price_change_24h": _safe_float(outcomes.get("price_change_24h")),
            "whale_score": _safe_float(outcomes.get("whale_score")),
            "response_action": response.get("action"),
            "response_strategy_type": response.get("strategy_type"),
        }
        row.update(outcomes)
        rows.append(row)
    return rows


def build_model_ready_dataset(
    signal_rows: list[dict[str, Any]],
    llm_rows: list[dict[str, Any]],
    *,
    include_pending: bool = False,
) -> list[dict[str, Any]]:
    combined: list[dict[str, Any]] = []

    for row in signal_rows:
        if not include_pending and row.get("pending_outcome"):
            continue
        combined.append({
            "source_type": "signal",
            "source_id": row.get("signal_id"),
            "coin_id": row.get("coin_id"),
            "symbol": row.get("symbol"),
            "pair_address": row.get("pair_address"),
            "chain": row.get("chain"),
            "event_timestamp": row.get("event_timestamp"),
            "price_usd": row.get("price_usd"),
            "liquidity_usd": row.get("liquidity_usd"),
            "volume_24h": row.get("volume_24h"),
            "buy_ratio": row.get("buy_ratio"),
            "whale_score": row.get("whale_score"),
            "price_change_1h": row.get("price_change_1h"),
            "price_change_24h": row.get("price_change_24h"),
            "sentiment_score": row.get("sentiment_score"),
            "signal_action": row.get("signal_action"),
            "signal_confidence": row.get("signal_confidence"),
            "llm_action": None,
            "llm_confidence": None,
            "strategy_type": None,
            "provider": None,
            "pending_outcome": row.get("pending_outcome"),
            "target_return_15m": row.get("future_return_15m"),
            "target_return_1h": row.get("future_return_1h"),
            "target_return_4h": row.get("future_return_4h"),
            "target_profitable_15m": row.get("label_profitable_after_fees_15m"),
            "target_profitable_1h": row.get("label_profitable_after_fees_1h"),
            "target_profitable_4h": row.get("label_profitable_after_fees_4h"),
        })

    for row in llm_rows:
        if not include_pending and row.get("pending_outcome"):
            continue
        combined.append({
            "source_type": "llm_decision",
            "source_id": row.get("decision_id"),
            "coin_id": row.get("coin_id"),
            "symbol": row.get("symbol"),
            "pair_address": row.get("pair_address"),
            "chain": row.get("chain"),
            "event_timestamp": row.get("event_timestamp"),
            "price_usd": row.get("price_usd"),
            "liquidity_usd": row.get("liquidity_usd"),
            "volume_24h": row.get("volume_24h"),
            "buy_ratio": row.get("buy_ratio"),
            "whale_score": row.get("whale_score"),
            "price_change_1h": row.get("price_change_1h"),
            "price_change_24h": row.get("price_change_24h"),
            "sentiment_score": None,
            "signal_action": None,
            "signal_confidence": None,
            "llm_action": row.get("llm_action"),
            "llm_confidence": row.get("llm_confidence"),
            "strategy_type": row.get("strategy_type"),
            "provider": row.get("provider"),
            "pending_outcome": row.get("pending_outcome"),
            "target_return_15m": row.get("future_return_15m"),
            "target_return_1h": row.get("future_return_1h"),
            "target_return_4h": row.get("future_return_4h"),
            "target_profitable_15m": row.get("label_profitable_after_fees_15m"),
            "target_profitable_1h": row.get("label_profitable_after_fees_1h"),
            "target_profitable_4h": row.get("label_profitable_after_fees_4h"),
        })

    return combined


def _ready_count(rows: list[dict[str, Any]], horizon: str) -> int:
    key = f"future_return_{horizon}"
    return sum(1 for r in rows if r.get(key) is not None)


def _avg_return(rows: list[dict[str, Any]], horizon: str) -> float | None:
    key = f"future_return_{horizon}"
    values = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(mean(values), 6) if values else None


def _positive_rate(rows: list[dict[str, Any]], horizon: str) -> float | None:
    key = f"label_up_{horizon}"
    labeled = [r for r in rows if r.get(key) is not None]
    if not labeled:
        return None
    positives = sum(1 for r in labeled if r.get(key))
    return round(positives / len(labeled), 6)


def _profitable_rate(rows: list[dict[str, Any]], horizon: str) -> float | None:
    key = f"label_profitable_after_fees_{horizon}"
    labeled = [r for r in rows if r.get(key) is not None]
    if not labeled:
        return None
    positives = sum(1 for r in labeled if r.get(key))
    return round(positives / len(labeled), 6)


def _event_timestamp_bounds(
    signal_rows: list[dict[str, Any]],
    llm_rows: list[dict[str, Any]],
) -> tuple[str | None, str | None]:
    stamps = [
        r.get("event_timestamp")
        for r in (*signal_rows, *llm_rows)
        if r.get("event_timestamp")
    ]
    if not stamps:
        return None, None
    return min(stamps), max(stamps)


def _latest_labeled_timestamp(rows: list[dict[str, Any]]) -> str | None:
    stamps = [r.get("latest_labeled_timestamp") for r in rows if r.get("latest_labeled_timestamp")]
    return max(stamps) if stamps else None


def build_summary(
    *,
    signal_rows: list[dict[str, Any]],
    llm_rows: list[dict[str, Any]],
    model_ready_rows: list[dict[str, Any]],
    warnings: list[str],
    output_files: list[str],
    fee_pct: float,
) -> dict[str, Any]:
    all_rows = [*signal_rows, *llm_rows]
    oldest, latest_event = _event_timestamp_bounds(signal_rows, llm_rows)
    return {
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "database_path": str(DB_PATH),
        "round_trip_fee_pct": fee_pct,
        "rows_signal_total": len(signal_rows),
        "rows_signal_ready_15m": _ready_count(signal_rows, "15m"),
        "rows_signal_ready_1h": _ready_count(signal_rows, "1h"),
        "rows_signal_ready_4h": _ready_count(signal_rows, "4h"),
        "rows_llm_total": len(llm_rows),
        "rows_llm_ready_15m": _ready_count(llm_rows, "15m"),
        "rows_llm_ready_1h": _ready_count(llm_rows, "1h"),
        "rows_llm_ready_4h": _ready_count(llm_rows, "4h"),
        "rows_model_ready": len(model_ready_rows),
        "pending_outcome_rows": sum(1 for r in all_rows if r.get("pending_outcome")),
        "average_return_15m": _avg_return(all_rows, "15m"),
        "average_return_1h": _avg_return(all_rows, "1h"),
        "average_return_4h": _avg_return(all_rows, "4h"),
        "positive_rate_15m": _positive_rate(all_rows, "15m"),
        "positive_rate_1h": _positive_rate(all_rows, "1h"),
        "positive_rate_4h": _positive_rate(all_rows, "4h"),
        "profitable_after_fees_rate_15m": _profitable_rate(all_rows, "15m"),
        "profitable_after_fees_rate_1h": _profitable_rate(all_rows, "1h"),
        "profitable_after_fees_rate_4h": _profitable_rate(all_rows, "4h"),
        "latest_event_timestamp": latest_event,
        "latest_labeled_timestamp": _latest_labeled_timestamp(all_rows),
        "oldest_event_timestamp": oldest,
        "output_files": output_files,
        "warnings": warnings,
    }


def write_dataset_file(
    rows: list[dict[str, Any]],
    stem: str,
    *,
    training_dir: Path | None = None,
) -> tuple[str, list[str]]:
    """Write parquet (preferred) or CSV. Returns (path_written, warnings)."""
    out_dir = training_dir or TRAINING_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    parquet_path = out_dir / f"{stem}.parquet"
    csv_path = out_dir / f"{stem}.csv"

    try:
        import pandas as pd

        frame = pd.DataFrame(rows)
        try:
            frame.to_parquet(parquet_path, index=False)
            return str(parquet_path), warnings
        except Exception as exc:
            warnings.append(
                f"Parquet write failed for {stem} ({exc}); falling back to CSV."
            )
            frame.to_csv(csv_path, index=False)
            return str(csv_path), warnings
    except ImportError:
        warnings.append(
            "pandas/pyarrow not installed — writing CSV instead of parquet. "
            "Install with: pip install pandas pyarrow"
        )
        if rows:
            import csv

            fieldnames = list(rows[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
        else:
            csv_path.write_text("", encoding="utf-8")
        return str(csv_path), warnings


def load_training_summary(path: Path | None = None) -> dict[str, Any] | None:
    summary_path = path or (TRAINING_DIR / SUMMARY_FILENAME)
    if not summary_path.is_file():
        return None
    try:
        with open(summary_path, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def build_training_datasets(*, include_pending: bool = False) -> dict[str, Any]:
    """Main entry: vectorized read-only pipeline, writes under data/training/."""
    from .vectorized_builder import build_training_datasets_vectorized

    return build_training_datasets_vectorized(include_pending=include_pending)


def print_run_report(report: dict[str, Any]) -> None:
    print("Training dataset build complete")
    print(f"  database: {report['database_path']}")
    print(f"  signal rows processed: {report['signal_rows_processed']}")
    print(f"  LLM decision rows processed: {report['llm_rows_processed']}")
    print(f"  model-ready rows: {report['ready_rows']}")
    print(f"  pending outcome rows: {report['pending_rows']}")
    timings = report.get("timings") or report.get("summary", {}).get("build_stage_timings", {})
    if timings:
        print("  stage timings (seconds):")
        for key, val in timings.items():
            print(f"    {key}: {val:.3f}" if isinstance(val, (int, float)) else f"    {key}: {val}")
    print("  output files:")
    for path in report["output_files"]:
        print(f"    - {path}")
    if report["warnings"]:
        print("  warnings:")
        for warning in report["warnings"]:
            print(f"    - {warning}")
