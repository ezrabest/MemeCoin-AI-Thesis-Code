#!/usr/bin/env python3
"""Diagnostic 7 — training vs live feature drift."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.diagnostics._common import (
    CHUNK_SIZE,
    DB_PATH,
    PROJECT_ROOT,
    DiagnosticReport,
    build_pair_from_signal_row,
    features_dict,
    iter_candidate_signal_chunks,
    open_db_readonly,
    percentile,
    safe_float,
)

TRAIN_PATH = PROJECT_ROOT / "data" / "training" / "model_ready_dataset.parquet"
METRICS_PATH = PROJECT_ROOT / "data" / "training" / "models" / "baseline_metrics.json"


def _feature_stats(series) -> dict:
    import pandas as pd

    s = pd.to_numeric(series, errors="coerce")
    s = s.dropna()
    if s.empty:
        return {"count": 0, "missing_rate": 1.0}
    vals = s.astype(float)
    return {
        "count": int(len(vals)),
        "missing_rate": float(series.isna().mean()) if hasattr(series, "isna") else 0.0,
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "std": float(vals.std()) if len(vals) > 1 else 0.0,
        "p05": percentile(list(vals), 0.05),
        "p25": percentile(list(vals), 0.25),
        "p50": percentile(list(vals), 0.50),
        "p75": percentile(list(vals), 0.75),
        "p95": percentile(list(vals), 0.95),
        "p99": percentile(list(vals), 0.99),
    }


def run(*, latest_candidates: int, output_dir: Path) -> DiagnosticReport:
    import json

    import pandas as pd

    from app.observability.candidate import TradeCandidate
    from app.observability.model_runtime_inference import get_runtime_model_inference, reset_runtime_model_inference_for_tests

    report = DiagnosticReport("inspect_training_vs_live_feature_drift", output_dir)
    if not TRAIN_PATH.is_file():
        report.add_limitation(f"Training dataset missing: {TRAIN_PATH}")
        report.set_status("WARN")
        report.data["summary"] = {"error": "training_dataset_missing"}
        report.write_json("training_vs_live_feature_drift.json")
        report.write_md(["Training dataset not available."], "training_vs_live_feature_drift.md")
        return report

    feature_cols: list[str] = []
    if METRICS_PATH.is_file():
        try:
            with open(METRICS_PATH, encoding="utf-8") as f:
                metrics = json.load(f)
            feature_cols = list(metrics.get("numeric_features") or []) + list(metrics.get("categorical_features") or [])
        except (OSError, json.JSONDecodeError):
            pass
    if not feature_cols:
        cols_path = PROJECT_ROOT / "data" / "training" / "model_ready_dataset_columns.txt"
        if cols_path.is_file():
            feature_cols = [ln.strip() for ln in cols_path.read_text(encoding="utf-8").splitlines() if ln.strip()]

    sample_n = min(50_000, latest_candidates * 5)
    try:
        train_df = pd.read_parquet(TRAIN_PATH, columns=[c for c in feature_cols if c] or None)
        if len(train_df) > sample_n:
            train_df = train_df.sample(n=sample_n, random_state=42)
            report.add_limitation(f"Training data sampled to {sample_n} rows")
    except Exception as exc:
        report.add_limitation(f"Could not load training parquet: {exc}")
        report.set_status("WARN")
        report.write_json("training_vs_live_feature_drift.json")
        return report

    if not feature_cols:
        feature_cols = [c for c in train_df.columns if c not in ("target", "label", "event_timestamp", "pair_address")][:80]
        report.add_limitation("Used training column list fallback (first 80 non-id columns)")

    reset_runtime_model_inference_for_tests()
    runtime = get_runtime_model_inference()
    live_frames: list[pd.DataFrame] = []

    if DB_PATH.is_file():
        conn = open_db_readonly()
        processed = 0
        try:
            for chunk in iter_candidate_signal_chunks(conn, limit=latest_candidates, chunk_size=CHUNK_SIZE):
                for row in chunk:
                    feats = features_dict(row.get("features_json"))
                    pair = build_pair_from_signal_row(row, feats)
                    candidate = TradeCandidate(
                        pair_address=str(row.get("pair_address") or ""),
                        chain=str(row.get("chain") or "unknown"),
                        symbol=str(row.get("symbol") or "?"),
                        price=safe_float(row.get("snap_price")),
                        liquidity_usd=safe_float(row.get("snap_liquidity")),
                        whale_score=safe_float(row.get("snap_whale_score")),
                        signal_score=safe_float(row.get("score")),
                        signal_type=str(row.get("signal_type") or "WATCH"),
                        coin_id=int(row["coin_id"]) if row.get("coin_id") is not None else None,
                        event_timestamp=str(row.get("timestamp") or ""),
                    )
                    try:
                        frame = runtime._build_live_feature_row(candidate, pair)
                        live_frames.append(frame)
                    except Exception:
                        pass
                    processed += 1
                if processed >= latest_candidates:
                    break
        finally:
            conn.close()
    else:
        report.add_limitation("Live DB missing — train-only stats reported")

    live_df = pd.concat(live_frames, ignore_index=True) if live_frames else pd.DataFrame()
    drift_rows: list[dict] = []
    high_drift: list[str] = []

    for col in feature_cols:
        if col not in train_df.columns:
            drift_rows.append({"feature": col, "note": "missing_in_train"})
            continue
        train_stats = _feature_stats(train_df[col])
        if live_df.empty or col not in live_df.columns:
            drift_rows.append({
                "feature": col,
                "train_missing_rate": train_stats.get("missing_rate"),
                "live_missing_rate": 1.0,
                "live_out_of_train_range_pct": None,
                "note": "missing_in_live",
            })
            continue
        live_stats = _feature_stats(live_df[col])
        out_range = 0
        s = pd.to_numeric(live_df[col], errors="coerce").dropna()
        if len(s) and train_stats.get("p05") is not None and train_stats.get("p95") is not None:
            out_range = float(((s < train_stats["p05"]) | (s > train_stats["p95"])).mean())
        psi = None
        ks = None
        try:
            from scipy import stats as scipy_stats

            t = pd.to_numeric(train_df[col], errors="coerce").dropna()
            l = pd.to_numeric(live_df[col], errors="coerce").dropna()
            if len(t) > 30 and len(l) > 30:
                ks = float(scipy_stats.ks_2samp(t, l).statistic)
        except Exception:
            pass

        row = {
            "feature": col,
            "train_missing_rate": round(train_stats.get("missing_rate", 0), 4),
            "live_missing_rate": round(live_stats.get("missing_rate", 0), 4),
            "train_mean": train_stats.get("mean"),
            "live_mean": live_stats.get("mean"),
            "train_p50": train_stats.get("p50"),
            "live_p50": live_stats.get("p50"),
            "live_out_of_train_range_pct": round(out_range, 4),
            "psi": psi,
            "ks_statistic": ks,
            "feature_constant_in_live": live_stats.get("std") == 0,
            "feature_constant_in_train": train_stats.get("std") == 0,
        }
        drift_rows.append(row)
        if out_range > 0.25 or (ks is not None and ks > 0.2):
            high_drift.append(col)

    severity = "low"
    if len(high_drift) >= 10:
        severity = "high"
    elif len(high_drift) >= 3:
        severity = "moderate"

    status = "PASS" if severity == "low" else "WARN"
    if live_df.empty:
        status = "WARN"

    report.set_status(status)
    report.data.update({
        "train_rows_sampled": len(train_df),
        "live_rows_built": len(live_df),
        "features_compared": len(drift_rows),
        "top_drifted_features": high_drift[:20],
        "likely_feature_drift_severity": severity,
        "feature_drift_table": drift_rows,
        "limitations": report.data.get("limitations", []),
    })
    report.write_json("training_vs_live_feature_drift.json")
    report.write_md([
        f"- Train rows: {len(train_df)} | Live feature rows: {len(live_df)}",
        f"- Drift severity: **{severity}**",
        f"- Top drifted: {', '.join(high_drift[:10]) or 'none'}",
    ], "training_vs_live_feature_drift.md")
    report.write_csv(drift_rows, "feature_drift_table.csv")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latest-candidates", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(latest_candidates=args.latest_candidates, output_dir=args.output_dir)
    print(f"Status: {report.data['status']}")
    return 0 if report.data["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
