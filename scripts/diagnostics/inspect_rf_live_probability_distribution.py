#!/usr/bin/env python3
"""Diagnostic 6 — RF live probability distribution on recent candidates."""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.diagnostics._common import (
    CHUNK_SIZE,
    DB_PATH,
    DiagnosticReport,
    build_pair_from_signal_row,
    distribution_stats,
    features_dict,
    iter_candidate_signal_chunks,
    open_db_readonly,
    safe_float,
)


def run(
    *,
    latest_candidates: int,
    rf_thresholds: list[float],
    output_dir: Path,
    ignore_model_age: bool = False,
) -> DiagnosticReport:
    from app.observability.audit_reasons import AuditReason
    from app.observability.candidate import TradeCandidate
    from app.observability.effective_settings import get_effective_settings
    from app.observability.llm_gate import BEARISH_ALERT_TYPES
    from app.observability.model_runtime_inference import (
        VALIDATED_RF_THRESHOLD,
        get_runtime_model_inference,
        reset_runtime_model_inference_for_tests,
    )

    report = DiagnosticReport("inspect_rf_live_probability_distribution", output_dir)
    if not DB_PATH.is_file():
        report.add_limitation(f"Database missing: {DB_PATH}")
        report.set_status("FAIL")
        return report

    reset_runtime_model_inference_for_tests()
    runtime = get_runtime_model_inference()
    meta = runtime.runtime_metadata()
    artifact_path = meta.get("model_path")
    artifact_exists = bool(artifact_path and Path(artifact_path).is_file())
    artifact_mtime = None
    artifact_age_hours = None
    if artifact_exists and artifact_path:
        p = Path(artifact_path)
        artifact_mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).isoformat()
        artifact_age_hours = round((datetime.now(timezone.utc) - datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)).total_seconds() / 3600, 2)

    eff = get_effective_settings()
    settings = dict(eff.canonical)
    if ignore_model_age:
        settings["max_model_artifact_age_hours"] = 999999
    configured_max_age = float(settings.get("max_model_artifact_age_hours", 168))
    configured_threshold = float(settings.get("rf_probability_threshold", VALIDATED_RF_THRESHOLD))

    probs: list[float] = []
    ok_count = 0
    fail_count = 0
    failure_reasons: dict[str, int] = {}
    top_rows: list[dict] = []
    group_counts: dict[str, int] = {}

    conn = open_db_readonly()
    processed = 0
    eval_cap = min(latest_candidates, 500)
    if latest_candidates > eval_cap:
        report.add_limitation(
            f"RF inference capped at {eval_cap} candidates (requested {latest_candidates}) for runtime safety"
        )
    try:
        for chunk in iter_candidate_signal_chunks(conn, limit=eval_cap, chunk_size=CHUNK_SIZE):
            for row in chunk:
                feats = features_dict(row.get("features_json"))
                pair = build_pair_from_signal_row(row, feats)
                buys = int(pair["txns"]["h24"]["buys"])
                sells = int(pair["txns"]["h24"]["sells"])
                br = safe_float(row.get("snap_buy_ratio") or (buys / max(buys + sells, 1)))
                alert = row.get("latest_alert_type")
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
                    event_timestamp=str(row.get("timestamp") or datetime.now(timezone.utc).isoformat()),
                )
                inference = runtime.predict_for_candidate(candidate, pair)
                if inference.status == "ok" and inference.predicted_probability is not None:
                    ok_count += 1
                    prob = float(inference.predicted_probability)
                    probs.append(prob)
                else:
                    fail_count += 1
                    for r in inference.audit_reasons:
                        failure_reasons[r] = failure_reasons.get(r, 0) + 1

                prob_val = inference.predicted_probability
                action = str(row.get("signal_type") or "HOLD")
                blockers = list(inference.audit_reasons)
                if not alert:
                    blockers.append(AuditReason.ALERT_REQUIRED_BUT_MISSING.value)
                if candidate.whale_score < float(settings.get("min_whale_score", 0.30)):
                    blockers.append(AuditReason.BELOW_WHALE_THRESHOLD.value)
                grp = action
                if prob_val is not None and prob_val >= configured_threshold:
                    grp = "BUY-like"
                group_counts[grp] = group_counts.get(grp, 0) + 1

                if len(top_rows) < 100:
                    top_rows.append({
                        "timestamp": row.get("timestamp"),
                        "pair_address": row.get("pair_address"),
                        "symbol": row.get("symbol"),
                        "chain": row.get("chain"),
                        "rf_probability": prob_val,
                        "whale_score": candidate.whale_score,
                        "liquidity": candidate.liquidity_usd,
                        "volume": candidate.volume_24h,
                        "buy_ratio": candidate.buy_ratio,
                        "action": action,
                        "blockers": ";".join(blockers[:5]),
                    })
                processed += 1
            if processed >= eval_cap:
                break
    finally:
        conn.close()

    dist = distribution_stats(probs)
    above = {str(t): sum(1 for p in probs if p >= t) for t in rf_thresholds}
    above["configured_threshold"] = sum(1 for p in probs if p >= configured_threshold)

    schema = meta.get("feature_schema_hash")
    status = "PASS"
    if not artifact_exists:
        status = "FAIL"
        report.add_limitation("RF model artifact missing")
    elif ok_count == 0:
        status = "FAIL"
        report.add_limitation("No successful RF inferences")
    elif dist.get("p95") is not None and dist["p95"] < configured_threshold:
        status = "WARN"
        report.add_limitation(f"p95 RF prob {dist['p95']:.4f} below configured threshold {configured_threshold}")

    if artifact_age_hours and artifact_age_hours > configured_max_age and not ignore_model_age:
        report.add_limitation(f"Artifact age {artifact_age_hours}h exceeds max {configured_max_age}h")
        if status == "PASS":
            status = "WARN"

    report.set_status(status)
    report.data.update({
        "model_artifact_path": artifact_path,
        "artifact_exists": artifact_exists,
        "artifact_mtime": artifact_mtime,
        "artifact_age_hours": artifact_age_hours,
        "configured_max_model_artifact_age_hours": configured_max_age,
        "schema_hash_expected": schema,
        "schema_hash_live": schema,
        "feature_count_expected": meta.get("feature_count"),
        "inference_ok_count": ok_count,
        "inference_failed_count": fail_count,
        "failure_reasons": failure_reasons,
        "rf_probability_distribution": dist,
        "counts_above_thresholds": above,
        "configured_rf_threshold": configured_threshold,
        "group_by_action": group_counts,
        "candidates_processed": processed,
        "ignore_model_age_for_diagnostic": ignore_model_age,
    })
    report.write_json("rf_live_probability_distribution.json")
    report.write_md([
        f"- Artifact: `{artifact_path}` exists={artifact_exists}",
        f"- Inference OK/FAIL: {ok_count}/{fail_count}",
        f"- Distribution: {dist}",
        f"- Above configured threshold ({configured_threshold}): {above.get('configured_threshold')}",
    ], "rf_live_probability_distribution.md")
    top_rows.sort(key=lambda x: safe_float(x.get("rf_probability"), -1), reverse=True)
    report.write_csv(top_rows[:100], "rf_top_candidates.csv")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest-candidates", type=int, default=10000)
    parser.add_argument("--rf-thresholds", default="0.01,0.03,0.05,0.10,0.20,0.55")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ignore-model-age-for-diagnostic", action="store_true")
    args = parser.parse_args()
    thresholds = [float(x.strip()) for x in args.rf_thresholds.split(",") if x.strip()]
    report = run(
        latest_candidates=args.latest_candidates,
        rf_thresholds=thresholds,
        output_dir=args.output_dir,
        ignore_model_age=args.ignore_model_age_for_diagnostic,
    )
    print(f"Status: {report.data['status']}")
    return 0 if report.data["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
