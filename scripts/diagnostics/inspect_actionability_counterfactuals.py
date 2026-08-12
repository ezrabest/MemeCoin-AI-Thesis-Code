#!/usr/bin/env python3
"""Diagnostic 10 — actionability counterfactual scenarios (no settings mutation)."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.diagnostics._common import (
    CHUNK_SIZE,
    DB_PATH,
    DiagnosticReport,
    build_pair_from_signal_row,
    features_dict,
    iter_candidate_signal_chunks,
    open_db_readonly,
    safe_float,
)


SCENARIOS: list[dict] = [
    {"name": "current_settings", "overrides": {}},
    {"name": "allow_watch_to_buy_promotion_true", "overrides": {"allow_watch_to_buy_promotion": True}},
    {"name": "alert_required_false_diagnostic", "overrides": {"_diag_ignore_missing_alert": True}},
    {"name": "min_whale_score_0.15", "overrides": {"min_whale_score": 0.15}},
    {"name": "min_whale_score_0.20", "overrides": {"min_whale_score": 0.20}},
    {"name": "min_whale_score_0.25", "overrides": {"min_whale_score": 0.25}},
    {"name": "rf_threshold_0.01", "overrides": {"rf_probability_threshold": 0.01}},
    {"name": "rf_threshold_0.03", "overrides": {"rf_probability_threshold": 0.03}},
    {"name": "rf_threshold_0.05", "overrides": {"rf_probability_threshold": 0.05}},
    {"name": "rf_threshold_0.10", "overrides": {"rf_probability_threshold": 0.10}},
    {"name": "max_snapshot_age_900s", "overrides": {"max_market_snapshot_age_seconds": 900}},
    {"name": "model_artifact_age_ignored", "overrides": {"max_model_artifact_age_hours": 999999}},
]


def _build_candidate(row: dict, *, ignore_alert: bool):
    from app.observability.candidate import TradeCandidate
    from app.observability.llm_gate import BEARISH_ALERT_TYPES

    feats = features_dict(row.get("features_json"))
    pair = build_pair_from_signal_row(row, feats)
    buys = int(pair["txns"]["h24"]["buys"])
    sells = int(pair["txns"]["h24"]["sells"])
    br = safe_float(row.get("snap_buy_ratio") or (buys / max(buys + sells, 1)))
    alert = row.get("latest_alert_type")
    if ignore_alert:
        alert = alert or "LARGE_BUY"
    candidate = TradeCandidate(
        pair_address=str(row.get("pair_address") or "").strip(),
        chain=str(row.get("chain") or "unknown"),
        symbol=str(row.get("symbol") or "?"),
        price=safe_float(row.get("snap_price")),
        liquidity_usd=safe_float(row.get("snap_liquidity")),
        whale_score=safe_float(row.get("snap_whale_score") or row.get("score")),
        signal_score=safe_float(row.get("score") or row.get("confidence")),
        signal_type=str(row.get("signal_type") or "WATCH"),
        coin_id=int(row["coin_id"]) if row.get("coin_id") is not None else None,
        volume_24h=safe_float(row.get("snap_volume_24h")) or None,
        buy_count=buys,
        sell_count=sells,
        buy_ratio=round(br, 4),
        alert_type=alert,
        bearish_alert_active=alert in BEARISH_ALERT_TYPES,
        event_timestamp=str(row.get("timestamp") or ""),
    )
    return candidate, pair


def _eval_scenario(
    prepared: list[tuple],
    base_settings: dict,
    overrides: dict,
) -> dict:
    from app.observability.audit_reasons import AuditReason
    from app.observability.economic_gate import evaluate_economic_trade_candidate

    settings = deepcopy(base_settings)
    settings["economic_gate_enabled"] = True
    settings["paper_trading_enabled"] = True
    settings["trading_mode"] = "DEMO"
    settings.update({k: v for k, v in overrides.items() if not k.startswith("_diag")})
    ignore_alert = bool(overrides.get("_diag_ignore_missing_alert"))

    stats: Counter[str] = Counter()
    blockers: Counter[str] = Counter()
    examples: list[dict] = []

    for base_candidate, pair in prepared:
        stats["candidates_evaluated"] += 1
        candidate = deepcopy(base_candidate)
        candidate.audit_reasons = list(getattr(base_candidate, "audit_reasons", []) or [])
        if ignore_alert and not candidate.alert_type:
            candidate.alert_type = "LARGE_BUY"
            candidate.bearish_alert_active = False
        result = evaluate_economic_trade_candidate(candidate, settings, pair=pair)
        for r in result.reasons:
            blockers[r] += 1
        if result.action == "WATCH":
            stats["actionable_watch_count"] += 1
        if result.action == "PAPER_BUY_CANDIDATE":
            stats["actionable_buy_like_count"] += 1
            stats["paper_buy_like_count"] += 1
        if AuditReason.MODEL_RUNTIME_INFERENCE_OK.value in result.reasons:
            stats["rf_ok_count"] += 1
        if result.audit_payload.get("rf_probability") is not None:
            thr = float(settings.get("rf_probability_threshold", 0.70))
            if float(result.audit_payload["rf_probability"]) >= thr:
                stats["rf_above_threshold_count"] += 1
        if AuditReason.ECONOMIC_GATE_APPROVED.value in result.reasons:
            stats["economic_pass_count"] += 1
        if len(examples) < 5 and result.action in ("PAPER_BUY_CANDIDATE", "HOLD", "BLOCKED"):
            examples.append({
                "symbol": candidate.symbol,
                "action": result.action,
                "whale_score": candidate.whale_score,
                "reasons": result.reasons[:5],
            })

    return {
        **dict(stats),
        "top_blockers": blockers.most_common(10),
        "top_examples": examples,
    }


def run(*, latest_candidates: int, output_dir: Path) -> DiagnosticReport:
    from app.observability import economic_gate as economic_gate_mod
    from app.observability.effective_settings import get_effective_settings

    report = DiagnosticReport("inspect_actionability_counterfactuals", output_dir)
    if not DB_PATH.is_file():
        report.add_limitation(f"Database missing: {DB_PATH}")
        report.set_status("FAIL")
        return report

    eval_cap = min(latest_candidates, 200)
    if latest_candidates > eval_cap:
        report.add_limitation(
            f"Counterfactual evaluation capped at {eval_cap} candidates (requested {latest_candidates})"
        )

    conn = open_db_readonly()
    rows: list[dict] = []
    try:
        for chunk in iter_candidate_signal_chunks(conn, limit=eval_cap, chunk_size=CHUNK_SIZE):
            rows.extend(chunk)
    finally:
        conn.close()

    base = dict(get_effective_settings().canonical)
    base["economic_gate_enabled"] = True
    base["paper_trading_enabled"] = True
    base["trading_mode"] = "DEMO"

    _orig_enrich = economic_gate_mod.enrich_candidate_with_model

    def _enrich_once(candidate, settings, **kwargs):
        if candidate.rf_prediction is not None:
            return candidate
        return _orig_enrich(candidate, settings, **kwargs)

    economic_gate_mod.enrich_candidate_with_model = _enrich_once
    prepared: list[tuple] = []
    scenario_results: list[dict] = []
    try:
        for row in rows:
            candidate, pair = _build_candidate(row, ignore_alert=False)
            _orig_enrich(candidate, base, pair=pair)
            prepared.append((candidate, pair))
        for sc in SCENARIOS:
            scenario_results.append({"scenario": sc["name"], **_eval_scenario(prepared, base, sc["overrides"])})
    finally:
        economic_gate_mod.enrich_candidate_with_model = _orig_enrich

    current = next(s for s in scenario_results if s["scenario"] == "current_settings")
    best_buy = max(s.get("actionable_buy_like_count", 0) for s in scenario_results)
    status = "WARN" if current.get("actionable_buy_like_count", 0) == 0 else "PASS"
    if best_buy > 0 and current.get("actionable_buy_like_count", 0) == 0:
        report.add_limitation("Counterfactual scenarios produce buy-like counts but current settings do not")

    report.set_status(status)
    report.data["scenarios"] = scenario_results
    report.data["candidates_loaded"] = len(rows)
    report.data["rf_enrichment_runs"] = len(prepared)
    report.write_json("actionability_counterfactuals.json")
    report.write_md([
        "## Scenario summary",
        *[f"- **{s['scenario']}**: buy-like={s.get('actionable_buy_like_count', 0)} rf_ok={s.get('rf_ok_count', 0)}" for s in scenario_results],
    ], "actionability_counterfactuals.md")
    flat = [{k: v for k, v in s.items() if k not in ("top_blockers", "top_examples")} for s in scenario_results]
    report.write_csv(flat, "counterfactual_summary.csv")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest-candidates", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(latest_candidates=args.latest_candidates, output_dir=args.output_dir)
    print(f"Status: {report.data['status']}")
    return 0 if report.data["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
