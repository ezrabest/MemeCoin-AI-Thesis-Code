from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(os.environ.get("THESIS_ROOT", r"E:\Projects\Final Project\memecoin_trader"))
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = ROOT / "data" / "audits" / f"thesis_context_data_availability_quality_audit_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

DB_PATH = ROOT / "data" / "trader.db"
AUDITS_ROOT = ROOT / "data" / "audits"

SAMPLE_LIMIT = int(os.environ.get("THESIS_SAMPLE_LIMIT", "200000"))
EVENT_BUCKET_MINUTES = int(os.environ.get("THESIS_EVENT_BUCKET_MINUTES", "120"))

EXPECTED_TABLES = {
    "market_snapshots": {
        "role": "core_market_state",
        "required_for_context_rebuild": True,
        "important_columns": [
            "id", "coin_id", "timestamp", "symbol", "chain", "pair_address",
            "price", "price_usd", "liquidity", "liquidity_usd",
            "volume", "volume_24h", "txns_buys", "txns_sells",
            "whale_score", "source_query"
        ],
        "timestamp_columns": ["timestamp"],
        "identity_columns": ["pair_address", "coin_id", "symbol"],
        "numeric_quality_columns": [
            "price", "price_usd", "liquidity", "liquidity_usd",
            "volume", "volume_24h", "txns_buys", "txns_sells", "whale_score"
        ],
    },
    "signals": {
        "role": "internal_logic_candidate_signal",
        "required_for_context_rebuild": True,
        "important_columns": [
            "id", "coin_id", "timestamp", "symbol", "chain", "pair_address",
            "signal_type", "score", "confidence", "reason", "features_json",
            "model_source"
        ],
        "timestamp_columns": ["timestamp"],
        "identity_columns": ["pair_address", "coin_id", "symbol"],
        "numeric_quality_columns": ["score", "confidence"],
    },
    "sentiment_records": {
        "role": "rss_sentiment_context",
        "required_for_context_rebuild": True,
        "important_columns": [
            "id", "timestamp", "source", "provider", "symbol", "coin_id",
            "sentiment_score", "relevance", "title", "url"
        ],
        "timestamp_columns": ["timestamp"],
        "identity_columns": ["symbol", "coin_id", "url"],
        "numeric_quality_columns": ["sentiment_score", "relevance"],
    },
    "gemini_decisions": {
        "role": "llm_decision_or_context_review_records",
        "required_for_context_rebuild": False,
        "important_columns": [
            "id", "timestamp", "symbol", "coin_id", "pair_address",
            "decision", "action", "risk_score", "confidence",
            "input_context_json", "gemini_response_json",
            "prompt_summary", "strategy_type"
        ],
        "timestamp_columns": ["timestamp"],
        "identity_columns": ["pair_address", "coin_id", "symbol"],
        "numeric_quality_columns": ["risk_score", "confidence"],
    },
    "whale_alerts": {
        "role": "pool_flow_proxy_or_wallet_whale_candidate",
        "required_for_context_rebuild": True,
        "important_columns": [
            "id", "timestamp", "coin_id", "symbol", "chain", "pair_address",
            "alert_type", "whale_score", "liquidity", "volume",
            "tx_summary_json", "is_real_wallet_level", "provider"
        ],
        "timestamp_columns": ["timestamp"],
        "identity_columns": ["pair_address", "coin_id", "symbol"],
        "numeric_quality_columns": ["whale_score", "liquidity", "volume", "is_real_wallet_level"],
    },
    "paper_trades": {
        "role": "paper_execution_outcome_or_label_source",
        "required_for_context_rebuild": True,
        "important_columns": [
            "id", "timestamp", "position_id", "coin_id", "symbol", "chain",
            "side", "price", "fill_price", "quantity", "amount",
            "notional_usd", "value", "fee", "slippage",
            "gross_pnl", "realized_pnl", "net_roi_pct",
            "reason", "reason_code", "source", "decision_ref_id"
        ],
        "timestamp_columns": ["timestamp"],
        "identity_columns": ["position_id", "coin_id", "symbol"],
        "numeric_quality_columns": [
            "price", "fill_price", "quantity", "amount", "notional_usd",
            "value", "fee", "slippage", "gross_pnl", "realized_pnl", "net_roi_pct"
        ],
    },
    "raw_provider_payloads": {
        "role": "raw_source_provenance",
        "required_for_context_rebuild": False,
        "important_columns": [
            "id", "timestamp", "provider", "source", "payload_json", "raw_json",
            "url", "symbol", "pair_address"
        ],
        "timestamp_columns": ["timestamp"],
        "identity_columns": ["provider", "source", "symbol", "pair_address"],
        "numeric_quality_columns": [],
    },
}


def safe_float(x: Any) -> float | None:
    try:
        if x is None:
            return None
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None


def connect_ro() -> sqlite3.Connection:
    if not DB_PATH.exists():
        raise FileNotFoundError(f"trader.db not found: {DB_PATH}")
    uri = f"file:{DB_PATH.as_posix()}?mode=ro"
    return sqlite3.connect(uri, uri=True)


def table_exists(con: sqlite3.Connection, table: str) -> bool:
    q = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return con.execute(q, (table,)).fetchone() is not None


def get_columns(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]


def count_rows(con: sqlite3.Connection, table: str) -> int:
    return int(con.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])


def get_sample(con: sqlite3.Connection, table: str, columns: list[str]) -> pd.DataFrame:
    cols_sql = ", ".join([f'"{c}"' for c in columns])
    return pd.read_sql_query(f'SELECT {cols_sql} FROM "{table}" LIMIT {SAMPLE_LIMIT}', con)


def timestamp_quality(df: pd.DataFrame, col: str) -> dict[str, Any]:
    s = df[col]
    parsed = pd.to_datetime(s, errors="coerce", utc=True)
    non_missing = int(s.notna().sum())
    parsed_ok = int(parsed.notna().sum())
    return {
        "timestamp_col": col,
        "non_missing": non_missing,
        "parsed_ok": parsed_ok,
        "parsed_fail": int(non_missing - parsed_ok),
        "parsed_ok_rate": parsed_ok / non_missing if non_missing else None,
        "min_timestamp": str(parsed.min()) if parsed_ok else None,
        "max_timestamp": str(parsed.max()) if parsed_ok else None,
    }


def column_quality(df: pd.DataFrame, table: str, col: str, role: str) -> dict[str, Any]:
    s = df[col]
    n = len(df)
    non_null = int(s.notna().sum())
    unique = int(s.nunique(dropna=True))

    row = {
        "table": table,
        "role": role,
        "column": col,
        "sample_rows": n,
        "non_null": non_null,
        "missing": int(n - non_null),
        "non_null_rate": non_null / n if n else None,
        "unique_non_null": unique,
        "example_non_null_values": "|".join([str(x)[:120] for x in s.dropna().astype(str).head(5).tolist()]),
        "numeric_valid": None,
        "numeric_positive": None,
        "numeric_negative": None,
        "numeric_zero": None,
        "numeric_min": None,
        "numeric_median": None,
        "numeric_max": None,
    }

    num = pd.to_numeric(s, errors="coerce")
    valid = int(num.notna().sum())
    if valid:
        row["numeric_valid"] = valid
        row["numeric_positive"] = int((num > 0).sum())
        row["numeric_negative"] = int((num < 0).sum())
        row["numeric_zero"] = int((num == 0).sum())
        row["numeric_min"] = float(num.min())
        row["numeric_median"] = float(num.median())
        row["numeric_max"] = float(num.max())

    return row


def event_duplication_quality(df: pd.DataFrame, table: str, identity_cols: list[str], timestamp_cols: list[str]) -> dict[str, Any]:
    available_id_cols = [c for c in identity_cols if c in df.columns]
    available_ts_cols = [c for c in timestamp_cols if c in df.columns]

    out = {
        "table": table,
        "sample_rows": len(df),
        "event_bucket_minutes": EVENT_BUCKET_MINUTES,
        "identity_cols_used": "|".join(available_id_cols),
        "timestamp_col_used": available_ts_cols[0] if available_ts_cols else None,
        "event_key_available": False,
        "event_unique_keys": None,
        "event_duplicate_rows": None,
        "event_duplicate_rate": None,
        "max_rows_per_event_key": None,
    }

    if not available_id_cols or not available_ts_cols:
        return out

    ts_col = available_ts_cols[0]
    ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    if ts.notna().sum() == 0:
        return out

    bucket = ts.dt.floor(f"{EVENT_BUCKET_MINUTES}min").astype(str)
    key_df = pd.DataFrame({"bucket": bucket})
    for c in available_id_cols:
        key_df[c] = df[c].astype(str).fillna("")

    valid_key = ts.notna()
    key_df = key_df.loc[valid_key]
    if key_df.empty:
        return out

    counts = key_df.value_counts(dropna=False)
    unique_keys = int(len(counts))
    duplicate_rows = int((counts - 1).clip(lower=0).sum())
    out.update({
        "event_key_available": True,
        "event_unique_keys": unique_keys,
        "event_duplicate_rows": duplicate_rows,
        "event_duplicate_rate": duplicate_rows / int(valid_key.sum()) if int(valid_key.sum()) else None,
        "max_rows_per_event_key": int(counts.max()) if len(counts) else None,
    })
    return out


def source_inventory_files() -> pd.DataFrame:
    patterns = [
        "*context*.csv",
        "*sentiment*.csv",
        "*blind*.csv",
        "*prediction*.csv",
        "*ground_truth*.csv",
        "*whale*.csv",
        "*semantic*.csv",
    ]

    rows = []
    if not AUDITS_ROOT.exists():
        return pd.DataFrame(rows)

    seen = set()
    for pat in patterns:
        for p in AUDITS_ROOT.rglob(pat):
            if p in seen:
                continue
            seen.add(p)
            if p.is_dir():
                continue
            try:
                n_rows = sum(1 for _ in open(p, "r", encoding="utf-8", errors="ignore")) - 1
                header = open(p, "r", encoding="utf-8", errors="ignore").readline().strip()
                cols = header.split(",") if header else []
                rows.append({
                    "path": str(p),
                    "name": p.name,
                    "size_bytes": p.stat().st_size,
                    "rows_estimated": n_rows,
                    "columns_count": len(cols),
                    "columns_preview": "|".join(cols[:80]),
                    "likely_context": "context" in p.name.lower(),
                    "likely_sentiment": "sentiment" in p.name.lower(),
                    "likely_whale": "whale" in p.name.lower(),
                    "likely_blind": "blind" in p.name.lower(),
                    "likely_ground_truth": "ground_truth" in p.name.lower(),
                    "likely_prediction": "prediction" in p.name.lower(),
                })
            except Exception as exc:
                rows.append({
                    "path": str(p),
                    "name": p.name,
                    "size_bytes": p.stat().st_size if p.exists() else None,
                    "rows_estimated": None,
                    "columns_count": None,
                    "columns_preview": "",
                    "error": repr(exc),
                })

    return pd.DataFrame(rows).sort_values(["rows_estimated", "size_bytes"], ascending=[False, False], na_position="last")


def assess_joinability(table_samples: dict[str, pd.DataFrame]) -> pd.DataFrame:
    checks = []

    def available(table: str, col: str) -> bool:
        return table in table_samples and col in table_samples[table].columns

    def coverage(table: str, col: str) -> float | None:
        if not available(table, col):
            return None
        df = table_samples[table]
        if len(df) == 0:
            return None
        return float(df[col].notna().mean())

    joins = [
        ("signals_to_market_by_coin_time", "signals", "market_snapshots", ["coin_id", "timestamp"], "candidate internal signal can join market snapshots"),
        ("signals_to_market_by_pair_time", "signals", "market_snapshots", ["pair_address", "timestamp"], "candidate internal signal can join market snapshots by pair"),
        ("sentiment_to_market_by_symbol_time", "sentiment_records", "market_snapshots", ["symbol", "timestamp"], "RSS/sentiment can be aligned to market events"),
        ("gemini_to_market_by_symbol_time", "gemini_decisions", "market_snapshots", ["symbol", "timestamp"], "LLM records can be aligned to market events"),
        ("whale_to_market_by_pair_time", "whale_alerts", "market_snapshots", ["pair_address", "timestamp"], "pool-flow/whale proxy can be aligned to market events"),
        ("paper_to_market_by_coin_time", "paper_trades", "market_snapshots", ["coin_id", "timestamp"], "paper outcomes can be linked back to market events"),
        ("paper_to_signals_by_coin_time", "paper_trades", "signals", ["coin_id", "timestamp"], "paper executions can be linked to candidate signals"),
    ]

    for name, left, right, cols, purpose in joins:
        row = {
            "join_check": name,
            "left_table": left,
            "right_table": right,
            "join_cols": "|".join(cols),
            "purpose": purpose,
            "left_exists": left in table_samples,
            "right_exists": right in table_samples,
            "all_cols_available": True,
            "left_min_col_coverage": None,
            "right_min_col_coverage": None,
            "verdict": "UNKNOWN",
            "notes": "",
        }

        left_covs = []
        right_covs = []
        for c in cols:
            if not available(left, c):
                row["all_cols_available"] = False
                row["notes"] += f"left missing {c}; "
            else:
                left_covs.append(coverage(left, c))

            if not available(right, c):
                row["all_cols_available"] = False
                row["notes"] += f"right missing {c}; "
            else:
                right_covs.append(coverage(right, c))

        if left_covs:
            row["left_min_col_coverage"] = min(x for x in left_covs if x is not None)
        if right_covs:
            row["right_min_col_coverage"] = min(x for x in right_covs if x is not None)

        if not row["left_exists"] or not row["right_exists"]:
            row["verdict"] = "BLOCKED_TABLE_MISSING"
        elif not row["all_cols_available"]:
            row["verdict"] = "BLOCKED_COLUMNS_MISSING"
        elif (row["left_min_col_coverage"] or 0) >= 0.8 and (row["right_min_col_coverage"] or 0) >= 0.8:
            row["verdict"] = "JOIN_KEYS_AVAILABLE_GOOD"
        elif (row["left_min_col_coverage"] or 0) >= 0.3 and (row["right_min_col_coverage"] or 0) >= 0.3:
            row["verdict"] = "JOIN_KEYS_AVAILABLE_PARTIAL"
        else:
            row["verdict"] = "JOIN_KEYS_WEAK"

        checks.append(row)

    return pd.DataFrame(checks)


def main() -> None:
    summary: dict[str, Any] = {
        "classification": "THESIS_CONTEXT_DATA_AVAILABILITY_QUALITY_AUDIT_COMPLETED",
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
        "sample_limit_per_table": SAMPLE_LIMIT,
        "event_bucket_minutes": EVENT_BUCKET_MINUTES,
        "db_path": str(DB_PATH),
        "db_exists": DB_PATH.exists(),
    }

    db_rows = []
    col_quality_rows = []
    timestamp_rows = []
    dup_rows = []
    red_flags = []
    table_samples: dict[str, pd.DataFrame] = {}

    if DB_PATH.exists():
        con = connect_ro()

        for table, spec in EXPECTED_TABLES.items():
            exists = table_exists(con, table)
            cols = get_columns(con, table) if exists else []
            row_count = count_rows(con, table) if exists else 0

            present_important = [c for c in spec["important_columns"] if c in cols]
            missing_important = [c for c in spec["important_columns"] if c not in cols]

            db_rows.append({
                "table": table,
                "role": spec["role"],
                "required_for_context_rebuild": spec["required_for_context_rebuild"],
                "exists": exists,
                "row_count": row_count,
                "columns_count": len(cols),
                "present_important_columns": "|".join(present_important),
                "missing_important_columns": "|".join(missing_important),
                "important_column_coverage": len(present_important) / len(spec["important_columns"]) if spec["important_columns"] else None,
            })

            if not exists:
                red_flags.append({
                    "severity": "HIGH" if spec["required_for_context_rebuild"] else "MEDIUM",
                    "area": "table_availability",
                    "table": table,
                    "issue": "expected table missing",
                    "implication": "cannot use this data source in context predictive rebuild",
                })
                continue

            sample_cols = sorted(set(cols))
            try:
                df = get_sample(con, table, sample_cols)
                table_samples[table] = df
            except Exception as exc:
                red_flags.append({
                    "severity": "HIGH",
                    "area": "table_readability",
                    "table": table,
                    "issue": f"failed to read sample: {exc!r}",
                    "implication": "table cannot be quality-audited",
                })
                continue

            for c in present_important:
                if c in df.columns:
                    col_quality_rows.append(column_quality(df, table, c, spec["role"]))

            for ts_col in spec["timestamp_columns"]:
                if ts_col in df.columns:
                    tq = timestamp_quality(df, ts_col)
                    tq["table"] = table
                    timestamp_rows.append(tq)

                    if tq["parsed_ok_rate"] is not None and tq["parsed_ok_rate"] < 0.95:
                        red_flags.append({
                            "severity": "HIGH",
                            "area": "timestamp_quality",
                            "table": table,
                            "issue": f"timestamp parse ok rate below 95% for {ts_col}: {tq['parsed_ok_rate']}",
                            "implication": "no-lookahead context alignment may be unreliable",
                        })
                else:
                    red_flags.append({
                        "severity": "MEDIUM",
                        "area": "timestamp_quality",
                        "table": table,
                        "issue": f"timestamp column missing: {ts_col}",
                        "implication": "cannot enforce no-lookahead alignment for this table",
                    })

            dq = event_duplication_quality(
                df,
                table,
                spec["identity_columns"],
                spec["timestamp_columns"],
            )
            dup_rows.append(dq)

            if dq["event_duplicate_rate"] is not None and dq["event_duplicate_rate"] > 0.5:
                red_flags.append({
                    "severity": "MEDIUM",
                    "area": "event_inflation",
                    "table": table,
                    "issue": f"high event duplicate rate under {EVENT_BUCKET_MINUTES}m bucketing: {dq['event_duplicate_rate']}",
                    "implication": "must use event-level dedup before statistical testing",
                })

            # Specific semantic warning: whale_alerts may be pool-flow proxy, not wallet-level.
            if table == "whale_alerts" and "is_real_wallet_level" in df.columns:
                real_rate = pd.to_numeric(df["is_real_wallet_level"], errors="coerce").fillna(0).gt(0).mean()
                if real_rate == 0:
                    red_flags.append({
                        "severity": "HIGH",
                        "area": "wallet_flow_semantics",
                        "table": table,
                        "issue": "sample has 0 real wallet-level whale rows",
                        "implication": "whale_alerts should be treated as pool-flow proxy unless linked wallet-level evidence is separately available",
                    })

        con.close()

    db_df = pd.DataFrame(db_rows)
    col_df = pd.DataFrame(col_quality_rows)
    ts_df = pd.DataFrame(timestamp_rows)
    dup_df = pd.DataFrame(dup_rows)
    join_df = assess_joinability(table_samples)
    file_inv_df = source_inventory_files()
    red_df = pd.DataFrame(red_flags)

    db_df.to_csv(OUT / "00_db_table_availability.csv", index=False, encoding="utf-8-sig")
    col_df.to_csv(OUT / "01_db_column_quality_sample.csv", index=False, encoding="utf-8-sig")
    ts_df.to_csv(OUT / "02_timestamp_quality.csv", index=False, encoding="utf-8-sig")
    dup_df.to_csv(OUT / "03_event_duplication_risk.csv", index=False, encoding="utf-8-sig")
    join_df.to_csv(OUT / "04_join_key_buildability.csv", index=False, encoding="utf-8-sig")
    file_inv_df.to_csv(OUT / "05_context_file_inventory.csv", index=False, encoding="utf-8-sig")
    red_df.to_csv(OUT / "06_data_quality_red_flags.csv", index=False, encoding="utf-8-sig")

    table_counts = {}
    if not db_df.empty:
        table_counts = {r["table"]: int(r["row_count"]) for _, r in db_df.iterrows() if bool(r["exists"])}

    verdict = "READY_FOR_CONTEXT_REBUILD_DESIGN"
    high_flags = 0 if red_df.empty else int((red_df["severity"] == "HIGH").sum())
    if high_flags >= 3:
        verdict = "READY_WITH_MAJOR_DATA_QUALITY_CONSTRAINTS"
    if not DB_PATH.exists():
        verdict = "BLOCKED_NO_TRADER_DB"

    summary.update({
        "verdict": verdict,
        "table_counts": table_counts,
        "red_flags_total": int(len(red_df)),
        "red_flags_high": high_flags,
        "join_verdict_counts": join_df["verdict"].value_counts().to_dict() if not join_df.empty else {},
        "context_files_found": int(len(file_inv_df)),
        "outputs": {
            "db_table_availability": str(OUT / "00_db_table_availability.csv"),
            "db_column_quality_sample": str(OUT / "01_db_column_quality_sample.csv"),
            "timestamp_quality": str(OUT / "02_timestamp_quality.csv"),
            "event_duplication_risk": str(OUT / "03_event_duplication_risk.csv"),
            "join_key_buildability": str(OUT / "04_join_key_buildability.csv"),
            "context_file_inventory": str(OUT / "05_context_file_inventory.csv"),
            "data_quality_red_flags": str(OUT / "06_data_quality_red_flags.csv"),
        }
    })

    with open(OUT / "thesis_context_data_availability_quality_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Thesis Context Data Availability + Quality Audit")
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
    lines.append("## Verdict")
    lines.append(f"`{verdict}`")
    lines.append("")
    lines.append("## DB table counts")
    for k, v in table_counts.items():
        lines.append(f"- `{k}`: {v:,}")
    lines.append("")
    lines.append("## Join buildability")
    if not join_df.empty:
        for verdict_name, count in join_df["verdict"].value_counts().items():
            lines.append(f"- `{verdict_name}`: {count}")
    lines.append("")
    lines.append("## Data quality red flags")
    if red_df.empty:
        lines.append("No red flags detected by this audit.")
    else:
        for _, r in red_df.head(30).iterrows():
            lines.append(f"- {r['severity']} | {r['area']} | `{r['table']}` | {r['issue']} | {r['implication']}")
    lines.append("")
    lines.append("## Output files")
    for _, path in summary["outputs"].items():
        lines.append(f"- `{Path(path).name}`")

    md_path = OUT / "thesis_context_data_availability_quality_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(OUT / "thesis_context_data_availability_quality_summary.json"),
        "summary_md": str(md_path),
        "verdict": verdict,
        "red_flags_total": int(len(red_df)),
        "red_flags_high": high_flags,
        "context_files_found": int(len(file_inv_df)),
    }, indent=2, ensure_ascii=False))
    print()
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
