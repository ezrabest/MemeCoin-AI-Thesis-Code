from __future__ import annotations

import json
import math
import os
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(os.environ.get("THESIS_ROOT", r"E:\Projects\Final Project\memecoin_trader"))
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = ROOT / "data" / "audits" / f"thesis_statistical_robustness_audit_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

EVENT_BUCKET_MINUTES = int(os.environ.get("THESIS_EVENT_BUCKET_MINUTES", "120"))
CSV_CHUNKSIZE = int(os.environ.get("THESIS_CSV_CHUNKSIZE", "200000"))

SEARCH_ROOTS = [
    ROOT / "data" / "training" / "manual_verified_results",
    ROOT / "data" / "audits",
]

EXCLUDE_PATH_PARTS = [
    "\\backup\\", "\\backups\\", "\\archive\\", "\\archives\\",
    "_backup", "backup_", "corrupted", "__pycache__",
]

AGGREGATE_NAME_PARTS = [
    "summary", "manifest", "metrics", "audit", "grid",
    "inventory", "schema", "feature_columns", "dataset_summary",
    "split_summary", "baseline_vs", "evaluation_by_tier",
]

GOOD_NAME_HINTS = [
    "selected_trades",
    "trades_by_tier",
    "model_evidence",
    "meta_outputs",
    "predictions_test",
    "predictions_validation",
    "consensus",
]

TIER_COLS = [
    "consensus_tier", "tier", "model_evidence_tier", "combo_type",
    "combo", "agreement_tier", "evidence_tier", "consensus_label",
]

RETURN_COLS = [
    "simulated_net_return", "net_return", "net_return_test",
    "actual_net_return", "avg_net_return", "net_roi_pct",
    "actual_net_roi_pct", "pnl", "actual_pnl", "outcome_pnl",
    "realized_pnl", "gross_pnl", "total_net_return",
]

BOOL_OUTCOME_COLS = [
    "positive_outcome", "is_positive", "target", "label",
    "target_net_profitable", "target_net_profitable_after_exit",
    "profitable", "winner", "is_winner",
]

CLASS_OUTCOME_COLS = [
    "actual_class", "outcome_class", "ground_truth", "class",
]

PAIR_COLS = [
    "provider_pair_url_exact", "provider_pair_url", "pair_address",
    "pair", "token_mint", "token_address", "contract_address",
    "candidate_id", "coin_id", "symbol",
]

TIME_COLS = [
    "timestamp", "event_timestamp", "entry_timestamp", "decision_timestamp",
    "created_at", "observed_at", "time", "datetime", "entry_time",
]

MODEL_FLAG_COLS = {
    "TAB": ["in_TAB", "in_tab", "tab_vote", "tab_true", "TAB_vote", "tab_support"],
    "XGB": ["in_XGB", "in_xgb", "xgb_vote", "xgb_true", "XGB_vote", "xgb_support"],
    "RF":  ["in_RF",  "in_rf",  "rf_vote",  "rf_true",  "RF_vote",  "rf_support"],
}


def path_is_excluded(p: Path) -> bool:
    s = str(p).lower()
    return any(x.lower() in s for x in EXCLUDE_PATH_PARTS)


def is_aggregate_file(p: Path) -> bool:
    name = p.name.lower()
    if "selected_trades" in name or "trades_by_tier" in name:
        return False
    return any(x in name for x in AGGREGATE_NAME_PARTS)


def has_good_hint(p: Path) -> bool:
    name = p.name.lower()
    return any(x in name for x in GOOD_NAME_HINTS)


def list_candidate_files() -> list[Path]:
    files: list[Path] = []
    for root in SEARCH_ROOTS:
        if not root.exists():
            continue
        for suffix in ("*.csv", "*.parquet"):
            for p in root.rglob(suffix):
                if path_is_excluded(p):
                    continue
                if is_aggregate_file(p):
                    continue
                if has_good_hint(p):
                    files.append(p)
    return sorted(set(files), key=lambda x: str(x).lower())


def read_columns_sample(p: Path) -> tuple[list[str], str | None]:
    try:
        if p.suffix.lower() == ".csv":
            df = pd.read_csv(p, nrows=25, low_memory=False)
        else:
            df = pd.read_parquet(p)
            df = df.head(25)
        return list(df.columns), None
    except Exception as exc:
        return [], repr(exc)


def pick_col(cols: list[str], candidates: list[str]) -> str | None:
    lower_map = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower_map:
            return lower_map[c.lower()]
    return None


def truthy_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0) > 0
    return (
        s.astype(str)
        .str.strip()
        .str.lower()
        .isin(["1", "true", "yes", "y", "winner", "win", "positive", "profitable"])
    )


def derive_tier(df: pd.DataFrame) -> pd.Series | None:
    cols = list(df.columns)
    tier_col = pick_col(cols, TIER_COLS)
    if tier_col:
        return df[tier_col].astype(str).str.strip().replace({"": "UNKNOWN"})

    flags: dict[str, pd.Series] = {}
    for model, candidates in MODEL_FLAG_COLS.items():
        col = pick_col(cols, candidates)
        if col:
            flags[model] = truthy_series(df[col])

    if not flags:
        return None

    tab = flags.get("TAB", pd.Series(False, index=df.index))
    xgb = flags.get("XGB", pd.Series(False, index=df.index))
    rf = flags.get("RF", pd.Series(False, index=df.index))

    out = pd.Series("REJECT_OR_NO_MODEL_SUPPORT", index=df.index, dtype="object")
    out[tab & xgb & rf] = "TAB_XGB_RF_ALL3"
    out[tab & rf & ~xgb] = "TAB_RF_ONLY"
    out[tab & xgb & ~rf] = "TAB_XGB_ONLY"
    out[xgb & rf & ~tab] = "RF_XGB_ONLY"
    single = (tab.astype(int) + xgb.astype(int) + rf.astype(int)) == 1
    out[single] = "SINGLE_MODEL_ONLY"
    return out


def derive_outcome_and_return(df: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None, str | None]:
    cols = list(df.columns)

    ret_col = pick_col(cols, RETURN_COLS)
    if ret_col:
        r = pd.to_numeric(df[ret_col], errors="coerce")
        return r > 0, r, ret_col

    bool_col = pick_col(cols, BOOL_OUTCOME_COLS)
    if bool_col:
        y = truthy_series(df[bool_col])
        return y, None, bool_col

    class_col = pick_col(cols, CLASS_OUTCOME_COLS)
    if class_col:
        raw = df[class_col].astype(str).str.strip().str.upper()
        y = raw.isin(["WINNER", "WIN", "POSITIVE", "PROFITABLE", "TP"])
        known = raw.isin(["WINNER", "WIN", "POSITIVE", "PROFITABLE", "TP", "LOSER", "LOSS", "NEGATIVE", "SL", "FLAT"])
        y = y.where(known, pd.NA)
        return y, None, class_col

    return None, None, None


def derive_event_hash(df: pd.DataFrame, tier: pd.Series) -> pd.Series:
    cols = list(df.columns)
    pair_col = pick_col(cols, PAIR_COLS)
    time_col = pick_col(cols, TIME_COLS)

    if pair_col:
        pair = df[pair_col].astype(str).fillna("NO_PAIR")
    else:
        pair = pd.Series([f"ROW_{i}" for i in range(len(df))], index=df.index)

    if time_col:
        ts = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        try:
            bucket = ts.dt.floor(f"{EVENT_BUCKET_MINUTES}min").astype(str)
        except Exception:
            bucket = ts.astype(str)
    else:
        bucket = pd.Series([f"NO_TIME_{i}" for i in range(len(df))], index=df.index)

    key_df = pd.DataFrame({
        "pair": pair.astype(str),
        "bucket": bucket.astype(str),
        "tier": tier.astype(str),
    })
    return pd.util.hash_pandas_object(key_df, index=False).astype("uint64")


def init_stats() -> dict[str, Any]:
    return {"n": 0, "positive": 0, "return_n": 0, "return_sum": 0.0, "return_min": None, "return_max": None}


def update_stats(stats: dict[str, Any], y: pd.Series, r: pd.Series | None) -> None:
    yy = y.dropna().astype(bool)
    stats["n"] += int(len(yy))
    stats["positive"] += int(yy.sum())

    if r is not None:
        rr = pd.to_numeric(r.loc[yy.index], errors="coerce").dropna()
        if len(rr):
            stats["return_n"] += int(len(rr))
            stats["return_sum"] += float(rr.sum())
            mn = float(rr.min())
            mx = float(rr.max())
            stats["return_min"] = mn if stats["return_min"] is None else min(stats["return_min"], mn)
            stats["return_max"] = mx if stats["return_max"] is None else max(stats["return_max"], mx)


def wilson_ci(pos: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    phat = pos / n
    denom = 1 + z*z/n
    center = (phat + z*z/(2*n)) / denom
    half = (z * math.sqrt((phat*(1-phat)/n) + (z*z/(4*n*n)))) / denom
    return max(0.0, center - half), min(1.0, center + half)


def ztest_two_prop(pos1: int, n1: int, pos0: int, n0: int) -> tuple[float | None, float | None]:
    if n1 <= 0 or n0 <= 0:
        return None, None
    p_pool = (pos1 + pos0) / (n1 + n0)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n0))
    if se == 0:
        return None, None
    z = (pos1/n1 - pos0/n0) / se
    p_two = math.erfc(abs(z) / math.sqrt(2))
    return z, p_two


def analyze_file(p: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    file_summary = {
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "status": "STARTED",
        "error": None,
        "eligible_rows_after_event_dedup": 0,
        "tier_count": 0,
        "outcome_source": None,
        "event_bucket_minutes": EVENT_BUCKET_MINUTES,
    }

    by_tier: dict[str, dict[str, Any]] = defaultdict(init_stats)
    seen_hashes: set[int] = set()

    def process_chunk(df: pd.DataFrame) -> None:
        nonlocal file_summary, seen_hashes

        tier = derive_tier(df)
        if tier is None:
            return

        y, r, outcome_source = derive_outcome_and_return(df)
        if y is None:
            return

        if file_summary["outcome_source"] is None:
            file_summary["outcome_source"] = outcome_source

        h = derive_event_hash(df, tier)
        keep = ~h.isin(seen_hashes)
        new_hashes = h.loc[keep].tolist()
        seen_hashes.update(int(x) for x in new_hashes)

        work = pd.DataFrame({
            "tier": tier.loc[keep].astype(str),
            "y": y.loc[keep],
        })
        if r is not None:
            work["return"] = r.loc[keep]

        work = work.dropna(subset=["tier", "y"])
        if work.empty:
            return

        file_summary["eligible_rows_after_event_dedup"] += int(len(work))

        for t, sub in work.groupby("tier", dropna=False):
            update_stats(by_tier[str(t)], sub["y"], sub["return"] if "return" in sub.columns else None)

    try:
        if p.suffix.lower() == ".csv":
            for chunk in pd.read_csv(p, chunksize=CSV_CHUNKSIZE, low_memory=False):
                process_chunk(chunk)
        else:
            df = pd.read_parquet(p)
            process_chunk(df)

        file_summary["status"] = "OK"
        file_summary["tier_count"] = len(by_tier)
    except Exception as exc:
        file_summary["status"] = "ERROR"
        file_summary["error"] = repr(exc)

    rows: list[dict[str, Any]] = []
    for tier, s in by_tier.items():
        n = s["n"]
        pos = s["positive"]
        lo, hi = wilson_ci(pos, n)
        rows.append({
            "source_path": str(p),
            "source_name": p.name,
            "tier": tier,
            "n_event_dedup": n,
            "positive": pos,
            "positive_rate": (pos / n) if n else None,
            "positive_rate_ci95_low": lo,
            "positive_rate_ci95_high": hi,
            "return_n": s["return_n"],
            "avg_return": (s["return_sum"] / s["return_n"]) if s["return_n"] else None,
            "return_min": s["return_min"],
            "return_max": s["return_max"],
            "outcome_source": file_summary["outcome_source"],
            "event_bucket_minutes": EVENT_BUCKET_MINUTES,
        })
    return rows, file_summary


def add_baseline_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    out_rows = []
    for source, sub in df.groupby("source_path"):
        baseline_candidates = sub[sub["tier"].astype(str).str.upper().isin([
            "REJECT", "REJECT_OR_NO_MODEL_SUPPORT", "MODEL_EVIDENCE_UNAVAILABLE"
        ])]
        if baseline_candidates.empty:
            baseline = None
        else:
            baseline = baseline_candidates.sort_values("n_event_dedup", ascending=False).iloc[0]

        for _, row in sub.iterrows():
            d = row.to_dict()
            d["comparison_baseline_tier"] = None
            d["z_vs_baseline"] = None
            d["p_two_sided_vs_baseline"] = None
            d["evidence_grade"] = "INSUFFICIENT_N"

            n = int(row["n_event_dedup"])
            if n >= 30:
                d["evidence_grade"] = "N_GE_30_EXPLORATORY"
            if n >= 50:
                d["evidence_grade"] = "N_GE_50_MORE_STABLE"

            if baseline is not None and str(row["tier"]) != str(baseline["tier"]):
                z, p = ztest_two_prop(
                    int(row["positive"]),
                    int(row["n_event_dedup"]),
                    int(baseline["positive"]),
                    int(baseline["n_event_dedup"]),
                )
                d["comparison_baseline_tier"] = baseline["tier"]
                d["z_vs_baseline"] = z
                d["p_two_sided_vs_baseline"] = p
                if n >= 30 and p is not None and p <= 0.05:
                    d["evidence_grade"] = "N_GE_30_AND_P_LE_0_05_VS_BASELINE"
                if n >= 50 and p is not None and p <= 0.05:
                    d["evidence_grade"] = "N_GE_50_AND_P_LE_0_05_VS_BASELINE"

            out_rows.append(d)

    return pd.DataFrame(out_rows)


def db_readonly_snapshot() -> dict[str, Any]:
    db_path = ROOT / "data" / "trader.db"
    if not db_path.exists():
        return {"db_exists": False}
    out = {"db_exists": True, "db_path": str(db_path), "tables": {}}
    uri = f"file:{db_path.as_posix()}?mode=ro"
    try:
        con = sqlite3.connect(uri, uri=True)
        tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name", con)["name"].tolist()
        for t in tables:
            try:
                n = pd.read_sql_query(f'SELECT COUNT(*) AS n FROM "{t}"', con)["n"].iloc[0]
                out["tables"][t] = int(n)
            except Exception as exc:
                out["tables"][t] = f"ERROR: {exc!r}"
        con.close()
    except Exception as exc:
        out["error"] = repr(exc)
    return out


def main() -> None:
    candidate_files = list_candidate_files()

    inventory_rows = []
    all_rows = []
    file_summaries = []

    for p in candidate_files:
        cols, err = read_columns_sample(p)
        inventory_rows.append({
            "path": str(p),
            "name": p.name,
            "suffix": p.suffix,
            "size_bytes": p.stat().st_size,
            "columns_seen": len(cols),
            "sample_error": err,
            "has_tier_col": any(c.lower() in [x.lower() for x in TIER_COLS] for c in cols),
            "has_model_flag_cols": any(any(x.lower() == c.lower() for x in sum(MODEL_FLAG_COLS.values(), [])) for c in cols),
            "has_return_col": any(c.lower() in [x.lower() for x in RETURN_COLS] for c in cols),
            "has_outcome_col": any(c.lower() in [x.lower() for x in BOOL_OUTCOME_COLS + CLASS_OUTCOME_COLS] for c in cols),
            "columns": "|".join(cols[:120]),
        })

    inv = pd.DataFrame(inventory_rows)
    inv.to_csv(OUT / "00_candidate_file_inventory.csv", index=False, encoding="utf-8-sig")

    eligible_paths = []
    for r in inventory_rows:
        if r["sample_error"]:
            continue
        if (r["has_tier_col"] or r["has_model_flag_cols"]) and (r["has_return_col"] or r["has_outcome_col"]):
            eligible_paths.append(Path(r["path"]))

    for p in eligible_paths:
        print(f"[ANALYZE] {p}")
        rows, summary = analyze_file(p)
        all_rows.extend(rows)
        file_summaries.append(summary)

    pd.DataFrame(file_summaries).to_csv(OUT / "01_file_processing_summary.csv", index=False, encoding="utf-8-sig")

    raw_results = pd.DataFrame(all_rows)
    if raw_results.empty:
        final_results = raw_results
    else:
        final_results = add_baseline_comparisons(raw_results)
        final_results = final_results.sort_values(["source_name", "n_event_dedup"], ascending=[True, False])

    final_results.to_csv(OUT / "02_model_tier_event_level_results.csv", index=False, encoding="utf-8-sig")

    if not raw_results.empty:
        pooled = raw_results.groupby("tier", dropna=False).agg(
            source_count=("source_path", "nunique"),
            n_event_dedup=("n_event_dedup", "sum"),
            positive=("positive", "sum"),
            return_n=("return_n", "sum"),
        ).reset_index()
        pooled["positive_rate"] = pooled["positive"] / pooled["n_event_dedup"]
        pooled.to_csv(OUT / "03_cautious_pooled_by_tier_do_not_use_without_source_review.csv", index=False, encoding="utf-8-sig")
    else:
        pooled = pd.DataFrame()

    db_snapshot = db_readonly_snapshot()

    summary = {
        "classification": "THESIS_STATISTICAL_ROBUSTNESS_AUDIT_COMPLETED",
        "root": str(ROOT),
        "output_root": str(OUT),
        "safety": {
            "read_only_audit": True,
            "training_run": False,
            "backtest_run": False,
            "trader_db_mutated": False,
            "wallet_connected": False,
            "live_trading_enabled": False,
        },
        "event_bucket_minutes": EVENT_BUCKET_MINUTES,
        "candidate_files_found": len(candidate_files),
        "eligible_model_tier_files": len(eligible_paths),
        "result_rows": int(len(final_results)) if not final_results.empty else 0,
        "db_readonly_snapshot": db_snapshot,
    }

    with open(OUT / "thesis_statistical_robustness_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Thesis Statistical Robustness Audit")
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
    lines.append("")
    lines.append("## File discovery")
    lines.append(f"- Candidate files found: {len(candidate_files)}")
    lines.append(f"- Eligible model-tier files: {len(eligible_paths)}")
    lines.append(f"- Event bucket minutes: {EVENT_BUCKET_MINUTES}")
    lines.append("")

    if final_results.empty:
        lines.append("## Result")
        lines.append("No eligible raw/event-level model-tier source was found.")
        lines.append("This means the existing Table 8.2 should remain exploratory unless a raw source is supplied.")
    else:
        lines.append("## Largest eligible source/tier rows")
        top = final_results.sort_values("n_event_dedup", ascending=False).head(30)
        for _, r in top.iterrows():
            lines.append(
                f"- `{r['source_name']}` | tier `{r['tier']}` | "
                f"n={int(r['n_event_dedup'])} | positive_rate={r['positive_rate']:.4f} | "
                f"grade={r['evidence_grade']}"
            )

    lines.append("")
    lines.append("## Key output files")
    lines.append("- `00_candidate_file_inventory.csv`")
    lines.append("- `01_file_processing_summary.csv`")
    lines.append("- `02_model_tier_event_level_results.csv`")
    lines.append("- `03_cautious_pooled_by_tier_do_not_use_without_source_review.csv`")
    lines.append("- `thesis_statistical_robustness_summary.json`")

    with open(OUT / "thesis_statistical_robustness_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nDONE. Summary MD:\n{OUT / 'thesis_statistical_robustness_summary.md'}")


if __name__ == "__main__":
    main()
