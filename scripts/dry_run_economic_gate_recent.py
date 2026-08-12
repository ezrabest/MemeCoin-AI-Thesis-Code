#!/usr/bin/env python3
"""Dry-run economic gate on recent SQLite signals — no trades, no LLM."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import database as db
from app.observability.audit_io import (
    AUDITS_DIR,
    get_decision_trace_writer,
    utc_date_slug,
)
from app.observability.audit_reasons import AuditReason
from app.observability.candidate import TradeCandidate
from app.observability.decision_trace import safe_persist_decision_trace
from app.observability.economic_gate import (
    evaluate_economic_trade_candidate,
    evaluate_tab_confidence_boost,
)
from app.observability.effective_settings import get_effective_settings
from app.observability.llm_gate import BEARISH_ALERT_TYPES

SCHEMA_MISMATCH = {
    AuditReason.MODEL_SCHEMA_MISMATCH.value,
    AuditReason.MODEL_FEATURE_MISSING.value,
    AuditReason.MODEL_FEATURE_EXTRA.value,
    AuditReason.MODEL_SCHEMA_METADATA_MISSING.value,
    AuditReason.MODEL_PREPROCESSOR_MISSING.value,
}
MODEL_UNAVAILABLE = {
    AuditReason.MODEL_RUNTIME_INFERENCE_NOT_AVAILABLE.value,
    AuditReason.MODEL_ARTIFACT_LOAD_FAILED.value,
    AuditReason.MODEL_UNAVAILABLE_OR_STALE.value,
}


def _parse_ts(ts: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _load_recent_rows(minutes: int, limit: int) -> list[dict[str, Any]]:
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
    with db.get_db() as conn:
        rows = conn.execute(
            """
            SELECT s.id AS signal_id, s.timestamp, s.coin_id, s.symbol, s.signal_type,
                   s.score, s.confidence, s.features_json,
                   c.pair_address, c.chain,
                   (
                     SELECT ms.price FROM market_snapshots ms
                     WHERE ms.coin_id = s.coin_id
                     ORDER BY ms.timestamp DESC LIMIT 1
                   ) AS snap_price,
                   (
                     SELECT ms.liquidity FROM market_snapshots ms
                     WHERE ms.coin_id = s.coin_id
                     ORDER BY ms.timestamp DESC LIMIT 1
                   ) AS snap_liquidity,
                   (
                     SELECT ms.volume_24h FROM market_snapshots ms
                     WHERE ms.coin_id = s.coin_id
                     ORDER BY ms.timestamp DESC LIMIT 1
                   ) AS snap_volume_24h,
                   (
                     SELECT ms.txns_buys FROM market_snapshots ms
                     WHERE ms.coin_id = s.coin_id
                     ORDER BY ms.timestamp DESC LIMIT 1
                   ) AS snap_txns_buys,
                   (
                     SELECT ms.txns_sells FROM market_snapshots ms
                     WHERE ms.coin_id = s.coin_id
                     ORDER BY ms.timestamp DESC LIMIT 1
                   ) AS snap_txns_sells,
                   (
                     SELECT ms.buy_ratio FROM market_snapshots ms
                     WHERE ms.coin_id = s.coin_id
                     ORDER BY ms.timestamp DESC LIMIT 1
                   ) AS snap_buy_ratio,
                   (
                     SELECT ms.whale_score FROM market_snapshots ms
                     WHERE ms.coin_id = s.coin_id
                     ORDER BY ms.timestamp DESC LIMIT 1
                   ) AS snap_whale_score,
                   (
                     SELECT wa.alert_type FROM whale_alerts wa
                     WHERE wa.coin_id = s.coin_id
                     ORDER BY wa.timestamp DESC LIMIT 1
                   ) AS latest_alert_type
            FROM signals s
            LEFT JOIN coins c ON c.id = s.coin_id
            WHERE s.timestamp >= ?
            ORDER BY s.timestamp DESC
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def _features_dict(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _build_pair(row: dict[str, Any], feats: dict[str, Any]) -> dict[str, Any]:
    symbol = str(row.get("symbol") or "?")
    base_sym = symbol.split("/")[0] if "/" in symbol else symbol
    buys = int(row.get("snap_txns_buys") or feats.get("txns_buys_24h") or 0)
    sells = int(row.get("snap_txns_sells") or feats.get("txns_sells_24h") or 0)
    return {
        "pairAddress": row.get("pair_address") or "",
        "chainId": (row.get("chain") or "unknown").lower(),
        "baseToken": {"symbol": base_sym},
        "priceUsd": str(row.get("snap_price") or feats.get("price_usd") or 0),
        "liquidity": {"usd": float(row.get("snap_liquidity") or feats.get("liquidity_usd") or 0)},
        "volume": {"h24": float(row.get("snap_volume_24h") or feats.get("volume_24h") or 0)},
        "txns": {"h24": {"buys": buys, "sells": sells}},
        "priceChange": {
            "h1": float(feats.get("price_change_1h") or 0),
            "h24": float(feats.get("price_change_24h") or 0),
        },
    }


def _build_candidate(row: dict[str, Any], pair: dict[str, Any], feats: dict[str, Any]) -> TradeCandidate:
    buys = int(pair["txns"]["h24"]["buys"])
    sells = int(pair["txns"]["h24"]["sells"])
    br = float(row.get("snap_buy_ratio") or feats.get("buy_ratio") or (buys / max(buys + sells, 1)))
    alert = row.get("latest_alert_type")
    price = float(row.get("snap_price") or feats.get("price_usd") or 0)
    return TradeCandidate(
        pair_address=str(row.get("pair_address") or "").strip(),
        chain=str(row.get("chain") or "unknown"),
        symbol=str(row.get("symbol") or "?"),
        price=price,
        liquidity_usd=float(row.get("snap_liquidity") or feats.get("liquidity_usd") or 0),
        whale_score=float(row.get("snap_whale_score") or row.get("score") or 0),
        signal_score=float(row.get("score") or row.get("confidence") or 0),
        signal_type=str(row.get("signal_type") or "WATCH"),
        coin_id=int(row["coin_id"]) if row.get("coin_id") is not None else None,
        volume_24h=float(row.get("snap_volume_24h") or feats.get("volume_24h") or 0) or None,
        buy_count=buys,
        sell_count=sells,
        buy_ratio=round(br, 4),
        alert_type=alert,
        bearish_alert_active=alert in BEARISH_ALERT_TYPES,
        current_execution_price=price,
        event_timestamp=str(row.get("timestamp") or datetime.now(timezone.utc).isoformat()),
        scan_id=f"dry_run_{row.get('signal_id')}",
    )


def _count_reason(reasons: list[str], code: str | set[str]) -> bool:
    if isinstance(code, str):
        return code in reasons
    return any(r in reasons for r in code)


def run_dry_run(*, minutes: int, limit: int) -> dict[str, Any]:
    db.init_pool()
    eff = get_effective_settings()
    settings = dict(eff.canonical)
    settings["economic_gate_enabled"] = True
    settings["paper_trading_enabled"] = True
    settings["trading_mode"] = "DEMO"

    rows = _load_recent_rows(minutes, limit)
    stats: Counter[str] = Counter()
    top_candidates: list[dict[str, Any]] = []

    for row in rows:
        stats["total_candidates"] += 1
        feats = _features_dict(row.get("features_json"))
        pair = _build_pair(row, feats)
        candidate = _build_candidate(row, pair, feats)
        candidate.settings_hash = eff.settings_hash

        result = evaluate_economic_trade_candidate(candidate, settings, pair=pair)
        reasons = list(result.reasons)

        if AuditReason.MODEL_RUNTIME_INFERENCE_OK.value in candidate.audit_reasons:
            stats["rf_inference_ok"] += 1
        if _count_reason(reasons, SCHEMA_MISMATCH) or _count_reason(candidate.audit_reasons, SCHEMA_MISMATCH):
            stats["schema_mismatch"] += 1
        if _count_reason(reasons, MODEL_UNAVAILABLE) or _count_reason(candidate.audit_reasons, MODEL_UNAVAILABLE):
            stats["model_unavailable_stale"] += 1
        if _count_reason(reasons, AuditReason.MODEL_TRAINED_WITH_TARGET_LEAKAGE.value):
            stats["target_leakage_blocked"] += 1
        if _count_reason(reasons, AuditReason.PROBABILITY_BELOW_THRESHOLD.value):
            stats["probability_below_threshold"] += 1
        if _count_reason(reasons, AuditReason.BLOCKED_BY_SLIPPAGE_LIMIT.value):
            stats["slippage_blocked"] += 1
        if _count_reason(reasons, AuditReason.BLOCKED_BY_PRICE_DRIFT.value):
            stats["price_drift_blocked"] += 1
        if _count_reason(reasons, AuditReason.BLOCKED_BY_BEARISH_ALERT.value):
            stats["bearish_alert_blocked"] += 1
        if _count_reason(reasons, AuditReason.MISSING_PRICE_OR_PAIR.value):
            stats["missing_required_fields"] += 1
        if _count_reason(reasons, AuditReason.MARKET_SNAPSHOT_TOO_OLD.value):
            stats["market_snapshot_too_old"] += 1
        if result.action == "PAPER_BUY_CANDIDATE":
            stats["paper_buy_candidate"] += 1

        if candidate.tab_prediction:
            stats["tab_available"] += 1

        rf_prob = float((candidate.rf_prediction or {}).get("predicted_probability") or 0)
        tab_mult, tab_reasons = evaluate_tab_confidence_boost(
            candidate,
            settings,
            rf_probability=rf_prob,
            trading_mode=str(settings.get("trading_mode", "DEMO")),
        )
        if tab_mult > 1.0:
            stats["tab_boost_eligible"] += 1
        if AuditReason.TAB_CONFIDENCE_BOOST_APPLIED.value in tab_reasons:
            stats["tab_boost_applied"] += 1

        safe_persist_decision_trace(
            candidate=candidate,
            result=result,
            stage="dry_run_economic_gate",
            extra={"dry_run": True, "signal_id": row.get("signal_id")},
        )

        top_candidates.append({
            "symbol": candidate.symbol,
            "pair_address": candidate.pair_address,
            "rf_probability": rf_prob,
            "action": result.action,
            "signal_type": candidate.signal_type,
            "whale_score": candidate.whale_score,
        })

    top_candidates.sort(key=lambda x: x["rf_probability"], reverse=True)
    top10 = top_candidates[:10]

    day = utc_date_slug()
    trace_path = AUDITS_DIR / f"decision_trace_{day}.jsonl"
    pipeline_path = AUDITS_DIR / f"pipeline_reasons_{day}.jsonl"
    report_path = AUDITS_DIR / f"dry_run_economic_gate_{day}.json"

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "minutes": minutes,
        "limit": limit,
        "total_candidates_evaluated": stats["total_candidates"],
        "rf_inference_ok": stats["rf_inference_ok"],
        "schema_mismatch": stats["schema_mismatch"],
        "model_unavailable_stale": stats["model_unavailable_stale"],
        "target_leakage_blocked": stats["target_leakage_blocked"],
        "probability_below_threshold": stats["probability_below_threshold"],
        "slippage_blocked": stats["slippage_blocked"],
        "price_drift_blocked": stats["price_drift_blocked"],
        "bearish_alert_blocked": stats["bearish_alert_blocked"],
        "missing_price_liquidity_pair_coin_id": stats["missing_required_fields"],
        "market_snapshot_too_old": stats["market_snapshot_too_old"],
        "tab_available": stats["tab_available"],
        "tab_boost_eligible": stats["tab_boost_eligible"],
        "tab_boost_applied": stats["tab_boost_applied"],
        "paper_buy_candidate": stats["paper_buy_candidate"],
        "top_10_by_rf_probability": top10,
        "output_paths": {
            "decision_trace_jsonl": str(trace_path),
            "pipeline_reasons_jsonl": str(pipeline_path),
            "summary_json": str(report_path),
        },
    }

    from app.observability.audit_io import write_json_report_atomic
    write_json_report_atomic(report_path.name, summary)

    # Close writers so paths are flushed
    from app.observability.audit_io import get_pipeline_reasons_writer, reset_audit_writers_for_tests
    get_decision_trace_writer().close()
    get_pipeline_reasons_writer().close()
    reset_audit_writers_for_tests()

    return summary


def _print_summary(summary: dict[str, Any]) -> None:
    print("=== Economic Gate Dry Run ===")
    for key in (
        "total_candidates_evaluated",
        "rf_inference_ok",
        "schema_mismatch",
        "model_unavailable_stale",
        "target_leakage_blocked",
        "probability_below_threshold",
        "slippage_blocked",
        "price_drift_blocked",
        "bearish_alert_blocked",
        "missing_price_liquidity_pair_coin_id",
        "market_snapshot_too_old",
        "tab_available",
        "tab_boost_eligible",
        "tab_boost_applied",
        "paper_buy_candidate",
    ):
        print(f"  {key}: {summary.get(key, 0)}")
    print("\nTop 10 by RF probability:")
    for i, row in enumerate(summary.get("top_10_by_rf_probability") or [], start=1):
        print(
            f"  {i}. {row.get('symbol')} rf={row.get('rf_probability', 0):.4f} "
            f"action={row.get('action')} whale={row.get('whale_score')}"
        )
    print("\nOutput paths:")
    for label, path in (summary.get("output_paths") or {}).items():
        print(f"  {label}: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run economic gate on recent SQLite signals")
    parser.add_argument("--minutes", type=int, default=30, help="Lookback window in minutes")
    parser.add_argument("--limit", type=int, default=50, help="Max candidates to evaluate")
    args = parser.parse_args()

    summary = run_dry_run(minutes=args.minutes, limit=args.limit)
    _print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
