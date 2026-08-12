from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT = Path(os.environ.get("THESIS_ROOT", r"E:\Projects\Final Project\memecoin_trader"))
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = ROOT / "data" / "audits" / f"thesis_context_rebuild_feasibility_matrix_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

DB = ROOT / "data" / "trader.db"

SAMPLE_LIMIT = int(os.environ.get("THESIS_SAMPLE_LIMIT", "500000"))
EVENT_BUCKET_MINUTES = int(os.environ.get("THESIS_EVENT_BUCKET_MINUTES", "120"))

REQUIRED_TABLES = [
    "signals",
    "market_snapshots",
    "sentiment_records",
    "gemini_decisions",
    "whale_alerts",
    "paper_trades",
]


def connect_ro() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def cols(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]


def count(con: sqlite3.Connection, table: str) -> int:
    return int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def read_sample(con: sqlite3.Connection, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT {SAMPLE_LIMIT}', con)


def pick(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def safe_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)


def build_event_key(df: pd.DataFrame, table: str) -> pd.DataFrame:
    ts_col = pick(df, ["timestamp", "created_at", "observed_at", "fetched_at", "time"])
    pair_col = pick(df, ["pair_address", "market_identity", "coin_id", "symbol"])

    out = df.copy()
    out["_timestamp_col_used"] = ts_col or ""
    out["_identity_col_used"] = pair_col or ""

    if ts_col is None:
        out["_event_time"] = pd.NaT
        out["_event_bucket"] = ""
    else:
        out["_event_time"] = safe_dt(out[ts_col])
        out["_event_bucket"] = out["_event_time"].dt.floor(f"{EVENT_BUCKET_MINUTES}min").astype(str)

    if pair_col is None:
        out["_event_identity"] = ""
    else:
        out["_event_identity"] = out[pair_col].astype(str).fillna("").str.upper().str.strip()

    out["_event_key"] = out["_event_identity"] + "|" + out["_event_bucket"]
    return out


def summarize_events(table: str, df: pd.DataFrame) -> dict:
    e = build_event_key(df, table)
    valid = e["_event_time"].notna() & e["_event_identity"].ne("") & e["_event_identity"].ne("NAN")
    ev = e.loc[valid].copy()

    if ev.empty:
        return {
            "table": table,
            "sample_rows": len(df),
            "valid_event_rows": 0,
            "event_unique_keys": 0,
            "event_duplicate_rows": None,
            "event_duplicate_rate": None,
            "timestamp_col_used": "",
            "identity_col_used": "",
            "min_time": None,
            "max_time": None,
        }

    counts = ev["_event_key"].value_counts()
    duplicate_rows = int((counts - 1).clip(lower=0).sum())

    return {
        "table": table,
        "sample_rows": int(len(df)),
        "valid_event_rows": int(len(ev)),
        "event_unique_keys": int(len(counts)),
        "event_duplicate_rows": duplicate_rows,
        "event_duplicate_rate": duplicate_rows / int(len(ev)) if len(ev) else None,
        "timestamp_col_used": str(ev["_timestamp_col_used"].iloc[0]),
        "identity_col_used": str(ev["_identity_col_used"].iloc[0]),
        "min_time": str(ev["_event_time"].min()),
        "max_time": str(ev["_event_time"].max()),
    }


def aggregate_context_counts(base_events: pd.DataFrame, context_df: pd.DataFrame, table: str) -> dict:
    """
    Feasibility only:
    counts how many signal event identities appear in each context table sample,
    without using future information or building a model.
    """
    b = build_event_key(base_events, "signals")
    c = build_event_key(context_df, table)

    b_valid = b["_event_identity"].ne("") & b["_event_identity"].ne("NAN") & b["_event_time"].notna()
    c_valid = c["_event_identity"].ne("") & c["_event_identity"].ne("NAN") & c["_event_time"].notna()

    b_ids = set(b.loc[b_valid, "_event_identity"].unique())
    c_ids = set(c.loc[c_valid, "_event_identity"].unique())

    overlap_ids = b_ids & c_ids

    return {
        "context_table": table,
        "signal_unique_identities": len(b_ids),
        "context_unique_identities": len(c_ids),
        "overlap_unique_identities": len(overlap_ids),
        "identity_overlap_rate_vs_signals": len(overlap_ids) / len(b_ids) if b_ids else None,
        "context_valid_event_rows": int(c_valid.sum()),
    }


def paper_label_feasibility(signals: pd.DataFrame, paper: pd.DataFrame) -> dict:
    s = build_event_key(signals, "signals")
    p = build_event_key(paper, "paper_trades")

    s_valid = s["_event_identity"].ne("") & s["_event_identity"].ne("NAN") & s["_event_time"].notna()
    p_valid = p["_event_identity"].ne("") & p["_event_identity"].ne("NAN") & p["_event_time"].notna()

    s_ids = set(s.loc[s_valid, "_event_identity"].unique())
    p_ids = set(p.loc[p_valid, "_event_identity"].unique())
    overlap = s_ids & p_ids

    return {
        "signal_unique_identities": len(s_ids),
        "paper_unique_identities": len(p_ids),
        "overlap_unique_identities": len(overlap),
        "identity_overlap_rate_vs_signals": len(overlap) / len(s_ids) if s_ids else None,
        "signal_valid_event_rows": int(s_valid.sum()),
        "paper_valid_event_rows": int(p_valid.sum()),
        "label_feasibility_verdict": (
            "PAPER_JOIN_FEASIBLE" if len(overlap) >= 10 else
            "PAPER_JOIN_WEAK_USE_FUTURE_MARKET_RETURNS_INSTEAD"
        ),
    }


def main() -> None:
    summary = {
        "classification": "THESIS_CONTEXT_REBUILD_FEASIBILITY_MATRIX_COMPLETED",
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
        "sample_limit": SAMPLE_LIMIT,
        "event_bucket_minutes": EVENT_BUCKET_MINUTES,
    }

    con = connect_ro()

    table_rows = []
    samples = {}

    for table in REQUIRED_TABLES:
        exists = table_exists(con, table)
        c = cols(con, table) if exists else []
        n = count(con, table) if exists else 0

        table_rows.append({
            "table": table,
            "exists": exists,
            "row_count": n,
            "columns_count": len(c),
            "columns": "|".join(c),
        })

        if exists and n > 0:
            samples[table] = read_sample(con, table)

    con.close()

    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(OUT / "00_table_schema_inventory.csv", index=False, encoding="utf-8-sig")

    event_rows = []
    for table, df in samples.items():
        event_rows.append(summarize_events(table, df))
    event_df = pd.DataFrame(event_rows)
    event_df.to_csv(OUT / "01_event_level_feasibility.csv", index=False, encoding="utf-8-sig")

    context_rows = []
    if "signals" in samples:
        for t in ["market_snapshots", "sentiment_records", "gemini_decisions", "whale_alerts"]:
            if t in samples:
                context_rows.append(aggregate_context_counts(samples["signals"], samples[t], t))
    context_df = pd.DataFrame(context_rows)
    context_df.to_csv(OUT / "02_context_identity_overlap_vs_signals.csv", index=False, encoding="utf-8-sig")

    label_result = {}
    if "signals" in samples and "paper_trades" in samples:
        label_result = paper_label_feasibility(samples["signals"], samples["paper_trades"])
    pd.DataFrame([label_result]).to_csv(OUT / "03_label_source_feasibility.csv", index=False, encoding="utf-8-sig")

    design = {
        "recommended_candidate_event_source": "signals",
        "recommended_event_unit": f"identity + {EVENT_BUCKET_MINUTES}-minute time bucket",
        "mandatory_dedup": True,
        "recommended_label_strategy": (
            "Prefer future market-return labels from market_snapshots when paper_trades overlap is weak; "
            "use paper_trades only as secondary validation or if join feasibility is good."
        ),
        "internal_only_block": [
            "signal score/confidence/model_source/reason-derived fields",
            "market state at or before event timestamp",
            "liquidity/volume/price-change/buy-sell activity",
        ],
        "context_block": [
            "past-window sentiment counts and sentiment aggregates",
            "past-window GDELT/RSS/context file features where available",
            "Gemini/LLM decision records only if timestamped and no-lookahead",
            "whale_alerts only as pool-flow/activity proxy, not real wallet-level whale evidence",
        ],
        "excluded_or_demoted": [
            "raw snapshot-level rows as independent observations",
            "future context after event timestamp",
            "whale_alerts as wallet-level whale evidence unless separately linked",
            "duplicate lineage files counted as independent experiments",
        ],
        "next_step": "build read-only event-level dataset candidate table and report post-dedup n, label availability, and context coverage before modeling",
    }

    with open(OUT / "04_recommended_context_rebuild_design.json", "w", encoding="utf-8") as f:
        json.dump(design, f, indent=2, ensure_ascii=False)

    summary.update({
        "table_counts": {
            r["table"]: int(r["row_count"])
            for r in table_rows
            if r["exists"]
        },
        "outputs": {
            "table_schema_inventory": str(OUT / "00_table_schema_inventory.csv"),
            "event_level_feasibility": str(OUT / "01_event_level_feasibility.csv"),
            "context_identity_overlap_vs_signals": str(OUT / "02_context_identity_overlap_vs_signals.csv"),
            "label_source_feasibility": str(OUT / "03_label_source_feasibility.csv"),
            "recommended_context_rebuild_design": str(OUT / "04_recommended_context_rebuild_design.json"),
        },
        "label_result": label_result,
    })

    with open(OUT / "thesis_context_rebuild_feasibility_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Thesis Context Rebuild Feasibility Matrix")
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
    lines.append("## Table counts")
    for k, v in summary["table_counts"].items():
        lines.append(f"- `{k}`: {v:,}")
    lines.append("")
    lines.append("## Event-level feasibility")
    if not event_df.empty:
        for _, r in event_df.iterrows():
            dup = r["event_duplicate_rate"]
            dup_s = "NA" if pd.isna(dup) else f"{float(dup)*100:.2f}%"
            lines.append(
                f"- `{r['table']}`: sample_rows={int(r['sample_rows']):,}, "
                f"valid_event_rows={int(r['valid_event_rows']):,}, "
                f"unique_event_keys={int(r['event_unique_keys']):,}, "
                f"duplicate_rate={dup_s}, "
                f"identity_col=`{r['identity_col_used']}`, "
                f"timestamp_col=`{r['timestamp_col_used']}`"
            )
    lines.append("")
    lines.append("## Context identity overlap vs signals")
    if not context_df.empty:
        for _, r in context_df.iterrows():
            rate = r["identity_overlap_rate_vs_signals"]
            rate_s = "NA" if pd.isna(rate) else f"{float(rate)*100:.2f}%"
            lines.append(
                f"- `{r['context_table']}`: signal identities={int(r['signal_unique_identities']):,}, "
                f"context identities={int(r['context_unique_identities']):,}, "
                f"overlap={int(r['overlap_unique_identities']):,}, "
                f"overlap rate={rate_s}"
            )
    lines.append("")
    lines.append("## Label feasibility")
    if label_result:
        for k, v in label_result.items():
            lines.append(f"- `{k}`: {v}")
    else:
        lines.append("No label feasibility result produced.")
    lines.append("")
    lines.append("## Recommended design")
    lines.append("- Candidate event source: `signals`")
    lines.append(f"- Event unit: identity + {EVENT_BUCKET_MINUTES}-minute time bucket")
    lines.append("- Mandatory event-level dedup before statistical testing")
    lines.append("- Main labels should be built from future market returns if paper-trade overlap is weak")
    lines.append("- Context must be aggregated only from timestamps at or before the candidate event")
    lines.append("- `whale_alerts` may be used only as pool-flow/activity proxy unless real wallet-level linkage is separately proven")
    lines.append("")
    lines.append("## Output files")
    for _, path in summary["outputs"].items():
        lines.append(f"- `{Path(path).name}`")

    md = OUT / "thesis_context_rebuild_feasibility_summary.md"
    md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(OUT / "thesis_context_rebuild_feasibility_summary.json"),
        "summary_md": str(md),
    }, indent=2, ensure_ascii=False))
    print()
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
