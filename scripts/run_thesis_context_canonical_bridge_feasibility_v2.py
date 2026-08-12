from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(os.environ.get("THESIS_ROOT", r"E:\Projects\Final Project\memecoin_trader"))
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = ROOT / "data" / "audits" / f"thesis_context_canonical_bridge_feasibility_v2_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

DB = ROOT / "data" / "trader.db"
ROW_LIMIT = int(os.environ.get("THESIS_ROW_LIMIT", "0"))  # 0 = full selected columns
EVENT_BUCKET_MINUTES = int(os.environ.get("THESIS_EVENT_BUCKET_MINUTES", "120"))

TABLE_COLS = {
    "coins": [
        "id", "symbol", "name", "chain", "pair_address"
    ],
    "signals": [
        "id", "coin_id", "symbol", "timestamp", "action",
        "score", "confidence", "signal_type", "model_source", "reason"
    ],
    "market_snapshots": [
        "id", "coin_id", "symbol", "chain", "pair_address", "timestamp",
        "price", "price_usd", "liquidity", "liquidity_usd",
        "volume", "volume_24h", "txns_buys", "txns_sells",
        "whale_score", "source_query"
    ],
    "whale_alerts": [
        "id", "coin_id", "symbol", "chain", "pair_address", "timestamp",
        "alert_type", "whale_score", "volume", "volume_usd",
        "price_impact_pct", "tx_count", "description"
    ],
    "gemini_decisions": [
        "id", "coin_id", "symbol", "timestamp", "decision",
        "action", "risk_score", "confidence", "prompt_summary",
        "strategy_type"
    ],
    "paper_trades": [
        "id", "coin_id", "symbol", "chain", "timestamp", "side",
        "price", "amount", "value", "fee", "slippage",
        "pnl", "gross_pnl", "realized_pnl", "net_roi_pct",
        "status", "reason", "source", "decision_ref_id", "position_id"
    ],
    "sentiment_records": [
        "id", "timestamp", "url", "source", "provider", "title",
        "sentiment_score", "relevance", "symbol", "coin_id"
    ],
    "raw_provider_payloads": [
        "id", "timestamp", "provider", "source", "symbol", "chain",
        "pair_address", "url"
    ],
}


def norm(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "na"}:
        return ""
    return s.upper()


def norm_pair(x: Any) -> str:
    return norm(x)


def norm_symbol_full(x: Any) -> str:
    return norm(x)


def norm_symbol_base(x: Any) -> str:
    s = norm(x)
    if "/" in s:
        return s.split("/", 1)[0].strip()
    return s


def connect_ro() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)


def exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone() is not None


def cols(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]


def count(con: sqlite3.Connection, table: str) -> int:
    return int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def read_selected(con: sqlite3.Connection, table: str, requested: list[str]) -> pd.DataFrame:
    available = cols(con, table)
    use_cols = [c for c in requested if c in available]
    if not use_cols:
        return pd.DataFrame()
    col_sql = ", ".join([f'"{c}"' for c in use_cols])
    limit_sql = "" if ROW_LIMIT <= 0 else f" LIMIT {ROW_LIMIT}"
    return pd.read_sql_query(f'SELECT {col_sql} FROM "{table}"{limit_sql}', con)


def build_coin_bridge(coins: pd.DataFrame) -> dict[str, Any]:
    c = coins.copy()
    c["coin_id_bridge"] = pd.to_numeric(c["id"], errors="coerce").astype("Int64")
    c["symbol_full_norm"] = c["symbol"].map(norm_symbol_full) if "symbol" in c.columns else ""
    c["symbol_base_norm"] = c["symbol"].map(norm_symbol_base) if "symbol" in c.columns else ""
    c["pair_norm"] = c["pair_address"].map(norm_pair) if "pair_address" in c.columns else ""
    c["chain_norm"] = c["chain"].map(norm) if "chain" in c.columns else ""

    id_map = {}
    for _, r in c.dropna(subset=["coin_id_bridge"]).iterrows():
        id_map[int(r["coin_id_bridge"])] = int(r["coin_id_bridge"])

    pair_map = {}
    for _, r in c.iterrows():
        p = r.get("pair_norm", "")
        cid = r.get("coin_id_bridge")
        if p and pd.notna(cid):
            pair_map[p] = int(cid)

    def unique_map(col: str) -> dict[str, int]:
        m: dict[str, set[int]] = {}
        for _, r in c.iterrows():
            key = r.get(col, "")
            cid = r.get("coin_id_bridge")
            if key and pd.notna(cid):
                m.setdefault(key, set()).add(int(cid))
        return {k: next(iter(v)) for k, v in m.items() if len(v) == 1}

    def ambiguous_set(col: str) -> set[str]:
        m: dict[str, set[int]] = {}
        for _, r in c.iterrows():
            key = r.get(col, "")
            cid = r.get("coin_id_bridge")
            if key and pd.notna(cid):
                m.setdefault(key, set()).add(int(cid))
        return {k for k, v in m.items() if len(v) > 1}

    return {
        "coins_prepared": c,
        "id_map": id_map,
        "pair_map": pair_map,
        "symbol_full_unique_map": unique_map("symbol_full_norm"),
        "symbol_base_unique_map": unique_map("symbol_base_norm"),
        "symbol_full_ambiguous": ambiguous_set("symbol_full_norm"),
        "symbol_base_ambiguous": ambiguous_set("symbol_base_norm"),
    }


def resolve_table(df: pd.DataFrame, table: str, bridge: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    out["canonical_coin_id"] = pd.NA
    out["canonical_bridge_method"] = "UNRESOLVED"
    out["canonical_bridge_confidence"] = "NONE"
    out["canonical_bridge_warning"] = ""

    id_map = bridge["id_map"]
    pair_map = bridge["pair_map"]
    full_map = bridge["symbol_full_unique_map"]
    base_map = bridge["symbol_base_unique_map"]
    full_amb = bridge["symbol_full_ambiguous"]
    base_amb = bridge["symbol_base_ambiguous"]

    def set_if_unresolved(mask, values, method, confidence, warning=""):
        idx = out.index[mask & out["canonical_coin_id"].isna()]
        if len(idx) == 0:
            return
        out.loc[idx, "canonical_coin_id"] = values.loc[idx]
        out.loc[idx, "canonical_bridge_method"] = method
        out.loc[idx, "canonical_bridge_confidence"] = confidence
        if warning:
            out.loc[idx, "canonical_bridge_warning"] = warning

    if "coin_id" in out.columns:
        coin_num = pd.to_numeric(out["coin_id"], errors="coerce")
        mapped = coin_num.map(lambda x: id_map.get(int(x)) if pd.notna(x) and int(x) in id_map else pd.NA)
        set_if_unresolved(mapped.notna(), mapped, "coin_id_to_coins_id", "EXACT_ID")

    if "pair_address" in out.columns:
        pair_norm = out["pair_address"].map(norm_pair)
        mapped = pair_norm.map(lambda x: pair_map.get(x, pd.NA))
        set_if_unresolved(mapped.notna(), mapped, "pair_address_to_coins_pair_address", "EXACT_PAIR")

    if "symbol" in out.columns:
        full = out["symbol"].map(norm_symbol_full)
        mapped_full = full.map(lambda x: full_map.get(x, pd.NA))
        ambiguous_full = full.isin(full_amb)
        set_if_unresolved(mapped_full.notna(), mapped_full, "symbol_full_unique_to_coins_symbol", "SYMBOL_UNIQUE")
        out.loc[out["canonical_coin_id"].isna() & ambiguous_full, "canonical_bridge_warning"] = "symbol_full_ambiguous_not_used"

        base = out["symbol"].map(norm_symbol_base)
        mapped_base = base.map(lambda x: base_map.get(x, pd.NA))
        ambiguous_base = base.isin(base_amb)
        set_if_unresolved(mapped_base.notna(), mapped_base, "symbol_base_unique_to_coins_symbol_base", "SYMBOL_BASE_UNIQUE")
        out.loc[out["canonical_coin_id"].isna() & ambiguous_base, "canonical_bridge_warning"] = "symbol_base_ambiguous_not_used"

    out["canonical_coin_id"] = pd.to_numeric(out["canonical_coin_id"], errors="coerce").astype("Int64")

    if "timestamp" in out.columns:
        out["event_time_utc"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
        out["event_bucket"] = out["event_time_utc"].dt.floor(f"{EVENT_BUCKET_MINUTES}min").astype(str)
    else:
        out["event_time_utc"] = pd.NaT
        out["event_bucket"] = ""

    out["canonical_event_key"] = out["canonical_coin_id"].astype(str) + "|" + out["event_bucket"].astype(str)

    return out


def summarize_resolution(table: str, df: pd.DataFrame) -> dict[str, Any]:
    n = len(df)
    resolved = int(df["canonical_coin_id"].notna().sum()) if "canonical_coin_id" in df.columns else 0
    methods = df["canonical_bridge_method"].value_counts(dropna=False).to_dict() if "canonical_bridge_method" in df.columns else {}
    warnings = df["canonical_bridge_warning"].value_counts(dropna=False).to_dict() if "canonical_bridge_warning" in df.columns else {}

    valid_events = df["canonical_coin_id"].notna() & df["event_time_utc"].notna()
    if int(valid_events.sum()) > 0:
        counts = df.loc[valid_events, "canonical_event_key"].value_counts()
        dup_rows = int((counts - 1).clip(lower=0).sum())
        unique_event_keys = int(len(counts))
        dup_rate = dup_rows / int(valid_events.sum())
    else:
        unique_event_keys = 0
        dup_rows = 0
        dup_rate = None

    return {
        "table": table,
        "rows_loaded": n,
        "resolved_rows": resolved,
        "resolved_rate": resolved / n if n else None,
        "unique_canonical_coins": int(df["canonical_coin_id"].nunique(dropna=True)) if "canonical_coin_id" in df.columns else 0,
        "valid_event_rows_after_bridge": int(valid_events.sum()),
        "unique_event_keys_after_bridge": unique_event_keys,
        "event_duplicate_rows_after_bridge": dup_rows,
        "event_duplicate_rate_after_bridge": dup_rate,
        "bridge_methods_json": json.dumps(methods, ensure_ascii=False),
        "bridge_warnings_json": json.dumps(warnings, ensure_ascii=False),
    }


def overlap_summary(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    if "signals" not in samples:
        return pd.DataFrame(rows)

    sig = samples["signals"]
    sig_ids = set(sig["canonical_coin_id"].dropna().astype(int).unique().tolist())
    sig_events = set(sig.loc[sig["canonical_coin_id"].notna() & sig["event_time_utc"].notna(), "canonical_event_key"].unique().tolist())

    for table, df in samples.items():
        if table == "signals":
            continue

        ids = set(df["canonical_coin_id"].dropna().astype(int).unique().tolist())
        events = set(df.loc[df["canonical_coin_id"].notna() & df["event_time_utc"].notna(), "canonical_event_key"].unique().tolist())

        rows.append({
            "context_table": table,
            "signals_unique_canonical_coins": len(sig_ids),
            "context_unique_canonical_coins": len(ids),
            "identity_overlap_unique": len(sig_ids & ids),
            "identity_overlap_rate_vs_signals": len(sig_ids & ids) / len(sig_ids) if sig_ids else None,
            "signals_unique_event_keys": len(sig_events),
            "context_unique_event_keys": len(events),
            "same_bucket_event_overlap": len(sig_events & events),
            "same_bucket_event_overlap_rate_vs_signals": len(sig_events & events) / len(sig_events) if sig_events else None,
        })

    return pd.DataFrame(rows).sort_values(
        ["identity_overlap_unique", "same_bucket_event_overlap"],
        ascending=[False, False],
        na_position="last",
    )


def make_design(res_df: pd.DataFrame, ov_df: pd.DataFrame) -> dict[str, Any]:
    def get_rate(table: str) -> float:
        r = res_df.loc[res_df["table"] == table]
        if r.empty:
            return 0.0
        v = r.iloc[0]["resolved_rate"]
        return 0.0 if pd.isna(v) else float(v)

    def get_overlap(table: str) -> float:
        r = ov_df.loc[ov_df["context_table"] == table]
        if r.empty:
            return 0.0
        v = r.iloc[0]["identity_overlap_rate_vs_signals"]
        return 0.0 if pd.isna(v) else float(v)

    return {
        "classification": "CONTEXT_REBUILD_DESIGN_AFTER_CANONICAL_BRIDGE_V2",
        "candidate_source": "signals",
        "candidate_source_resolution_rate": get_rate("signals"),
        "market_snapshots_resolution_rate": get_rate("market_snapshots"),
        "market_identity_overlap_vs_signals": get_overlap("market_snapshots"),
        "whale_identity_overlap_vs_signals": get_overlap("whale_alerts"),
        "gemini_identity_overlap_vs_signals": get_overlap("gemini_decisions"),
        "paper_identity_overlap_vs_signals": get_overlap("paper_trades"),
        "sentiment_identity_overlap_vs_signals": get_overlap("sentiment_records"),
        "recommended_label_source": (
            "future_market_returns_from_market_snapshots"
            if get_overlap("market_snapshots") >= 0.5
            else "blocked_or_requires_alternative_label_source"
        ),
        "recommended_context_sources": {
            "market_snapshots": "baseline/internal market feature source and future-return label source if no-lookahead and horizon rules are enforced",
            "whale_alerts": "pool-flow/activity proxy only, not real wallet-level whale evidence",
            "gemini_decisions": "optional audit/context feature only if timestamped before event and overlap sufficient",
            "sentiment_records": "use only if identity can be resolved; current DB schema may be weak if only URL is available",
            "raw_provider_payloads": "provenance/identity support, not direct predictive feature unless parsed safely",
            "paper_trades": "secondary validation only unless bridge/time overlap is strong",
        },
        "mandatory_controls": [
            "exclude cross-table id primary-key overlaps",
            "use canonical_coin_id from coins bridge",
            "deduplicate to canonical_coin_id + time bucket before statistics",
            "aggregate context only at or before candidate timestamp",
            "derive labels only from future windows after candidate timestamp",
            "use chronological split for any modeling",
            "report both row-level and event-level n",
        ],
    }


def main() -> None:
    summary = {
        "classification": "THESIS_CONTEXT_CANONICAL_BRIDGE_FEASIBILITY_V2_COMPLETED",
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
        "row_limit": ROW_LIMIT,
        "event_bucket_minutes": EVENT_BUCKET_MINUTES,
        "bridge_policy": {
            "allowed": [
                "table.coin_id -> coins.id",
                "table.pair_address -> coins.pair_address",
                "table.symbol -> coins.symbol only when unique",
                "table.base_symbol -> unique coins base symbol only when unique",
            ],
            "explicitly_excluded": [
                "cross-table id overlaps",
                "ambiguous symbol matches",
                "post-hoc ground truth joins",
            ],
        },
    }

    con = connect_ro()

    available_tables = {}
    table_counts = {}
    for table in TABLE_COLS:
        if exists(con, table):
            table_counts[table] = count(con, table)
            available_tables[table] = read_selected(con, table, TABLE_COLS[table])

    con.close()

    if "coins" not in available_tables or available_tables["coins"].empty:
        raise SystemExit("coins table is required for canonical bridge v2")

    bridge = build_coin_bridge(available_tables["coins"])

    resolved = {}
    for table, df in available_tables.items():
        if table == "coins":
            continue
        resolved[table] = resolve_table(df, table, bridge)

    # Also export coin bridge core.
    bridge["coins_prepared"].to_csv(OUT / "00_canonical_coins_bridge.csv", index=False, encoding="utf-8-sig")

    res_rows = []
    for table, df in resolved.items():
        res_rows.append(summarize_resolution(table, df))
    res_df = pd.DataFrame(res_rows).sort_values("table")
    res_df.to_csv(OUT / "01_bridge_resolution_by_table.csv", index=False, encoding="utf-8-sig")

    ov_df = overlap_summary(resolved)
    ov_df.to_csv(OUT / "02_signal_context_overlap_after_bridge.csv", index=False, encoding="utf-8-sig")

    # Export small post-bridge samples for inspection, not full data.
    for table, df in resolved.items():
        keep = [c for c in [
            "id", "coin_id", "symbol", "chain", "pair_address", "timestamp",
            "canonical_coin_id", "canonical_bridge_method", "canonical_bridge_confidence",
            "canonical_bridge_warning", "event_time_utc", "event_bucket", "canonical_event_key"
        ] if c in df.columns]
        df[keep].head(5000).to_csv(OUT / f"sample_resolved_{table}.csv", index=False, encoding="utf-8-sig")

    design = make_design(res_df, ov_df)
    with open(OUT / "03_context_rebuild_design_after_bridge_v2.json", "w", encoding="utf-8") as f:
        json.dump(design, f, indent=2, ensure_ascii=False)

    summary.update({
        "table_counts": table_counts,
        "resolution": res_df.to_dict(orient="records"),
        "overlap": ov_df.to_dict(orient="records"),
        "design": design,
        "outputs": {
            "canonical_coins_bridge": str(OUT / "00_canonical_coins_bridge.csv"),
            "bridge_resolution_by_table": str(OUT / "01_bridge_resolution_by_table.csv"),
            "signal_context_overlap_after_bridge": str(OUT / "02_signal_context_overlap_after_bridge.csv"),
            "context_rebuild_design_after_bridge_v2": str(OUT / "03_context_rebuild_design_after_bridge_v2.json"),
        },
    })

    with open(OUT / "thesis_context_canonical_bridge_feasibility_v2_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Thesis Context Canonical Bridge Feasibility v2")
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
    lines.append("## Bridge policy")
    lines.append("- Allowed: `coin_id -> coins.id`, `pair_address -> coins.pair_address`, unique non-ambiguous symbol bridge")
    lines.append("- Excluded: cross-table `id` primary-key overlaps")
    lines.append("- Excluded: ambiguous symbol matches")
    lines.append("")
    lines.append("## Table counts")
    for k, v in table_counts.items():
        lines.append(f"- `{k}`: {v:,}")
    lines.append("")
    lines.append("## Bridge resolution by table")
    for _, r in res_df.iterrows():
        rr = r["resolved_rate"]
        dr = r["event_duplicate_rate_after_bridge"]
        rr_s = "NA" if pd.isna(rr) else f"{float(rr)*100:.2f}%"
        dr_s = "NA" if pd.isna(dr) else f"{float(dr)*100:.2f}%"
        lines.append(
            f"- `{r['table']}`: rows={int(r['rows_loaded']):,}, "
            f"resolved={int(r['resolved_rows']):,} ({rr_s}), "
            f"unique_coins={int(r['unique_canonical_coins'])}, "
            f"valid_event_rows={int(r['valid_event_rows_after_bridge']):,}, "
            f"unique_event_keys={int(r['unique_event_keys_after_bridge']):,}, "
            f"duplicate_rate={dr_s}, "
            f"methods={r['bridge_methods_json']}"
        )
    lines.append("")
    lines.append("## Signal-context overlap after canonical bridge")
    if ov_df.empty:
        lines.append("No overlap rows produced.")
    else:
        for _, r in ov_df.iterrows():
            id_rate = r["identity_overlap_rate_vs_signals"]
            ev_rate = r["same_bucket_event_overlap_rate_vs_signals"]
            id_s = "NA" if pd.isna(id_rate) else f"{float(id_rate)*100:.2f}%"
            ev_s = "NA" if pd.isna(ev_rate) else f"{float(ev_rate)*100:.2f}%"
            lines.append(
                f"- `signals` ↔ `{r['context_table']}`: "
                f"identity_overlap={int(r['identity_overlap_unique'])}/"
                f"{int(r['signals_unique_canonical_coins'])} ({id_s}), "
                f"same_bucket_event_overlap={int(r['same_bucket_event_overlap'])}/"
                f"{int(r['signals_unique_event_keys'])} ({ev_s})"
            )
    lines.append("")
    lines.append("## Recommended next step")
    lines.append(f"- Label source: `{design['recommended_label_source']}`")
    lines.append("- Build a read-only event-level candidate dataset only after this bridge is accepted.")
    lines.append("- Use `whale_alerts` only as pool-flow/activity proxy.")
    lines.append("- Do not use unresolved `sentiment_records` as token-level context unless identity is separately resolved.")
    lines.append("")
    lines.append("## Output files")
    for _, path in summary["outputs"].items():
        lines.append(f"- `{Path(path).name}`")

    md = OUT / "thesis_context_canonical_bridge_feasibility_v2_summary.md"
    md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(OUT / "thesis_context_canonical_bridge_feasibility_v2_summary.json"),
        "summary_md": str(md),
    }, indent=2, ensure_ascii=False))
    print()
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
