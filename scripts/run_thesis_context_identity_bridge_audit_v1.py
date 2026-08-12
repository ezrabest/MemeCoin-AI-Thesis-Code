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
OUT = ROOT / "data" / "audits" / f"thesis_context_identity_bridge_audit_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

DB = ROOT / "data" / "trader.db"
SAMPLE_LIMIT = int(os.environ.get("THESIS_SAMPLE_LIMIT", "300000"))

TABLES = [
    "coins",
    "market_snapshots",
    "signals",
    "sentiment_records",
    "gemini_decisions",
    "whale_alerts",
    "paper_trades",
    "raw_provider_payloads",
]

IDENTITY_CANDIDATES = [
    "id",
    "coin_id",
    "symbol",
    "coin_symbol",
    "token_symbol",
    "base_symbol",
    "name",
    "coin_name",
    "token_name",
    "chain",
    "network",
    "pair_address",
    "base_token_address",
    "quote_token_address",
    "market_identity",
    "market_identity_raw",
    "source_query",
    "url",
    "pair_url",
    "provider_pair_url",
]

TIMESTAMP_CANDIDATES = [
    "timestamp",
    "created_at",
    "updated_at",
    "observed_at",
    "fetched_at",
    "time",
    "datetime",
    "date",
]


def norm_value(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "na"}:
        return ""
    return s.upper()


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


def read_sample(con: sqlite3.Connection, table: str) -> pd.DataFrame:
    return pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT {SAMPLE_LIMIT}', con)


def existing(cols: list[str], candidates: list[str]) -> list[str]:
    lower = {c.lower(): c for c in cols}
    return [lower[c.lower()] for c in candidates if c.lower() in lower]


def col_profile(df: pd.DataFrame, table: str, col: str) -> dict[str, Any]:
    s = df[col]
    n = len(df)
    non_null = int(s.notna().sum())
    unique = int(s.dropna().astype(str).map(norm_value).replace("", pd.NA).dropna().nunique())
    examples = (
        s.dropna()
        .astype(str)
        .map(lambda x: x[:160])
        .drop_duplicates()
        .head(10)
        .tolist()
    )
    return {
        "table": table,
        "column": col,
        "sample_rows": n,
        "non_null": non_null,
        "non_null_rate": non_null / n if n else None,
        "unique_normalized_non_empty": unique,
        "examples": "|".join(examples),
    }


def value_set(df: pd.DataFrame, col: str) -> set[str]:
    if col not in df.columns:
        return set()
    return set(
        df[col]
        .dropna()
        .map(norm_value)
        .loc[lambda x: x.ne("")]
        .unique()
        .tolist()
    )


def direct_overlap(samples: dict[str, pd.DataFrame], profiles: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for left_table, left_cols in profiles.items():
        for right_table, right_cols in profiles.items():
            if left_table >= right_table:
                continue
            left_df = samples[left_table]
            right_df = samples[right_table]

            for lc in left_cols:
                lv = value_set(left_df, lc)
                if not lv:
                    continue
                for rc in right_cols:
                    rv = value_set(right_df, rc)
                    if not rv:
                        continue
                    inter = lv & rv
                    rows.append({
                        "left_table": left_table,
                        "left_col": lc,
                        "right_table": right_table,
                        "right_col": rc,
                        "left_unique": len(lv),
                        "right_unique": len(rv),
                        "overlap_unique": len(inter),
                        "overlap_rate_vs_left": len(inter) / len(lv) if lv else None,
                        "overlap_rate_vs_right": len(inter) / len(rv) if rv else None,
                        "examples_overlap": "|".join(list(sorted(inter))[:20]),
                    })
    return pd.DataFrame(rows).sort_values(
        ["overlap_unique", "overlap_rate_vs_left"],
        ascending=[False, False],
        na_position="last",
    )


def timestamp_profile(df: pd.DataFrame, table: str, col: str) -> dict[str, Any]:
    parsed = pd.to_datetime(df[col], errors="coerce", utc=True)
    non_null = int(df[col].notna().sum())
    ok = int(parsed.notna().sum())
    return {
        "table": table,
        "timestamp_col": col,
        "non_null": non_null,
        "parsed_ok": ok,
        "parsed_ok_rate": ok / non_null if non_null else None,
        "min_timestamp": str(parsed.min()) if ok else None,
        "max_timestamp": str(parsed.max()) if ok else None,
    }


def bridge_candidates(samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    # Try coins as bridge if available.
    if "coins" not in samples:
        return pd.DataFrame([{
            "bridge": "coins",
            "status": "MISSING",
            "details": "coins table not available",
        }])

    coins = samples["coins"]
    coin_cols = list(coins.columns)

    possible_coin_id_cols = existing(coin_cols, ["id", "coin_id"])
    possible_symbol_cols = existing(coin_cols, ["symbol", "coin_symbol", "token_symbol"])
    possible_pair_cols = existing(coin_cols, ["pair_address", "address", "base_token_address"])
    possible_chain_cols = existing(coin_cols, ["chain", "network"])

    rows.append({
        "bridge": "coins",
        "status": "AVAILABLE",
        "coin_id_cols": "|".join(possible_coin_id_cols),
        "symbol_cols": "|".join(possible_symbol_cols),
        "pair_address_cols": "|".join(possible_pair_cols),
        "chain_cols": "|".join(possible_chain_cols),
        "coins_rows_sampled": len(coins),
    })

    # Estimate whether each table can bridge through coins by coin_id/symbol/pair_address.
    for table, df in samples.items():
        if table == "coins":
            continue

        table_cols = list(df.columns)
        for table_col in existing(table_cols, ["coin_id", "symbol", "coin_symbol", "token_symbol", "pair_address", "base_token_address"]):
            tv = value_set(df, table_col)

            best = []
            for cc in possible_coin_id_cols + possible_symbol_cols + possible_pair_cols:
                cv = value_set(coins, cc)
                inter = tv & cv
                if inter:
                    best.append((cc, len(inter), len(inter) / len(tv) if tv else None))

            best = sorted(best, key=lambda x: x[1], reverse=True)
            rows.append({
                "bridge": "coins",
                "status": "TABLE_BRIDGE_TEST",
                "table": table,
                "table_col": table_col,
                "table_unique": len(tv),
                "best_coins_col": best[0][0] if best else "",
                "best_overlap_unique": best[0][1] if best else 0,
                "best_overlap_rate_vs_table": best[0][2] if best else 0,
                "all_matches": "|".join([f"{cc}:{n}" for cc, n, _ in best[:10]]),
            })

    return pd.DataFrame(rows)


def main() -> None:
    summary = {
        "classification": "THESIS_CONTEXT_IDENTITY_BRIDGE_AUDIT_COMPLETED",
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
    }

    con = connect_ro()

    table_inventory = []
    samples: dict[str, pd.DataFrame] = {}
    identity_profiles: dict[str, list[str]] = {}
    col_profiles = []
    ts_profiles = []

    for table in TABLES:
        exists_flag = table_exists(con, table)
        cols = table_cols(con, table) if exists_flag else []
        row_count = table_count(con, table) if exists_flag else 0

        id_cols = existing(cols, IDENTITY_CANDIDATES)
        ts_cols = existing(cols, TIMESTAMP_CANDIDATES)

        table_inventory.append({
            "table": table,
            "exists": exists_flag,
            "row_count": row_count,
            "columns_count": len(cols),
            "identity_candidate_columns": "|".join(id_cols),
            "timestamp_candidate_columns": "|".join(ts_cols),
            "all_columns": "|".join(cols),
        })

        if exists_flag and row_count > 0:
            df = read_sample(con, table)
            samples[table] = df
            identity_profiles[table] = id_cols

            for c in id_cols:
                col_profiles.append(col_profile(df, table, c))

            for c in ts_cols:
                ts_profiles.append(timestamp_profile(df, table, c))

    con.close()

    inv_df = pd.DataFrame(table_inventory)
    prof_df = pd.DataFrame(col_profiles)
    ts_df = pd.DataFrame(ts_profiles)
    overlap_df = direct_overlap(samples, identity_profiles)
    bridge_df = bridge_candidates(samples)

    inv_df.to_csv(OUT / "00_table_identity_schema_inventory.csv", index=False, encoding="utf-8-sig")
    prof_df.to_csv(OUT / "01_identity_column_profiles.csv", index=False, encoding="utf-8-sig")
    ts_df.to_csv(OUT / "02_timestamp_column_profiles.csv", index=False, encoding="utf-8-sig")
    overlap_df.to_csv(OUT / "03_direct_identity_overlaps.csv", index=False, encoding="utf-8-sig")
    bridge_df.to_csv(OUT / "04_coins_bridge_candidates.csv", index=False, encoding="utf-8-sig")

    # Build concise verdict.
    top_direct = []
    if not overlap_df.empty:
        top_direct = overlap_df.head(20).to_dict(orient="records")

    top_bridge = []
    if not bridge_df.empty and "best_overlap_unique" in bridge_df.columns:
        tmp = bridge_df.copy()
        tmp["best_overlap_unique_num"] = pd.to_numeric(tmp["best_overlap_unique"], errors="coerce").fillna(0)
        top_bridge = tmp.sort_values("best_overlap_unique_num", ascending=False).head(20).drop(columns=["best_overlap_unique_num"]).to_dict(orient="records")

    summary.update({
        "tables_scanned": len(TABLES),
        "tables_available": int(inv_df["exists"].sum()) if not inv_df.empty else 0,
        "top_direct_overlaps": top_direct,
        "top_coins_bridge_candidates": top_bridge,
        "outputs": {
            "table_identity_schema_inventory": str(OUT / "00_table_identity_schema_inventory.csv"),
            "identity_column_profiles": str(OUT / "01_identity_column_profiles.csv"),
            "timestamp_column_profiles": str(OUT / "02_timestamp_column_profiles.csv"),
            "direct_identity_overlaps": str(OUT / "03_direct_identity_overlaps.csv"),
            "coins_bridge_candidates": str(OUT / "04_coins_bridge_candidates.csv"),
        },
    })

    with open(OUT / "thesis_context_identity_bridge_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Thesis Context Identity Bridge Audit")
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
    lines.append("## Tables")
    for _, r in inv_df.iterrows():
        lines.append(
            f"- `{r['table']}`: exists={r['exists']}, rows={int(r['row_count']):,}, "
            f"identity_cols=`{r['identity_candidate_columns']}`, "
            f"timestamp_cols=`{r['timestamp_candidate_columns']}`"
        )
    lines.append("")
    lines.append("## Top direct identity overlaps")
    if overlap_df.empty:
        lines.append("No direct identity overlaps found.")
    else:
        for _, r in overlap_df.head(15).iterrows():
            lines.append(
                f"- `{r['left_table']}.{r['left_col']}` ↔ `{r['right_table']}.{r['right_col']}`: "
                f"overlap={int(r['overlap_unique'])}, "
                f"left_unique={int(r['left_unique'])}, "
                f"right_unique={int(r['right_unique'])}, "
                f"examples=`{r['examples_overlap']}`"
            )
    lines.append("")
    lines.append("## Top coins-bridge candidates")
    if bridge_df.empty:
        lines.append("No coins bridge result.")
    else:
        show = bridge_df.copy()
        if "best_overlap_unique" in show.columns:
            show["_sort"] = pd.to_numeric(show["best_overlap_unique"], errors="coerce").fillna(0)
            show = show.sort_values("_sort", ascending=False)
        for _, r in show.head(20).iterrows():
            if r.get("status") == "TABLE_BRIDGE_TEST":
                lines.append(
                    f"- `{r.get('table')}.{r.get('table_col')}` via `coins.{r.get('best_coins_col')}`: "
                    f"overlap={r.get('best_overlap_unique')}, "
                    f"rate_vs_table={r.get('best_overlap_rate_vs_table')}, "
                    f"matches=`{r.get('all_matches')}`"
                )
            else:
                lines.append(f"- {dict(r.drop(labels=['_sort'], errors='ignore'))}")
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "If direct overlap is weak but coins-bridge overlap is strong, the previous feasibility matrix should be treated "
        "as a resolver failure rather than data absence. The context rebuild should then use a canonical identity bridge "
        "before constructing event-level datasets."
    )
    lines.append("")
    lines.append("## Output files")
    for _, path in summary["outputs"].items():
        lines.append(f"- `{Path(path).name}`")

    md = OUT / "thesis_context_identity_bridge_summary.md"
    md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(OUT / "thesis_context_identity_bridge_summary.json"),
        "summary_md": str(md),
    }, indent=2, ensure_ascii=False))
    print()
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
