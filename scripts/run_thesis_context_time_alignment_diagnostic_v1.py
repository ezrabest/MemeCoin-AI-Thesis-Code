from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("THESIS_ROOT", r"E:\Projects\Final Project\memecoin_trader"))
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = ROOT / "data" / "audits" / f"thesis_context_time_alignment_diagnostic_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

DB = ROOT / "data" / "trader.db"
EVENT_BUCKET_MINUTES = int(os.environ.get("THESIS_EVENT_BUCKET_MINUTES", "120"))
HORIZON_MINUTES = int(os.environ.get("THESIS_HORIZON_MINUTES", "240"))


def norm(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "na"}:
        return ""
    return s.upper()


def connect_ro() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)


def cols(con: sqlite3.Connection, table: str) -> list[str]:
    return [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]


def read_cols(con: sqlite3.Connection, table: str, wanted: list[str]) -> pd.DataFrame:
    available = cols(con, table)
    use = [c for c in wanted if c in available]
    sql_cols = ", ".join([f'"{c}"' for c in use])
    return pd.read_sql_query(f'SELECT {sql_cols} FROM "{table}"', con)


def build_bridge(coins: pd.DataFrame) -> dict[str, Any]:
    c = coins.copy()
    c["cid"] = pd.to_numeric(c["id"], errors="coerce").astype("Int64")
    c["pair_norm"] = c["pair_address"].map(norm) if "pair_address" in c.columns else ""
    c["symbol_norm"] = c["symbol"].map(norm) if "symbol" in c.columns else ""

    id_map = {int(x): int(x) for x in c["cid"].dropna().tolist()}

    pair_map = {}
    for _, r in c.iterrows():
        if r.get("pair_norm") and pd.notna(r.get("cid")):
            pair_map[r["pair_norm"]] = int(r["cid"])

    symbol_groups = {}
    for _, r in c.iterrows():
        s = r.get("symbol_norm", "")
        if s and pd.notna(r.get("cid")):
            symbol_groups.setdefault(s, set()).add(int(r["cid"]))
    symbol_map = {k: next(iter(v)) for k, v in symbol_groups.items() if len(v) == 1}

    return {"id_map": id_map, "pair_map": pair_map, "symbol_map": symbol_map}


def resolve(df: pd.DataFrame, bridge: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()

    # Use pandas nullable Int64 from the start.
    # This avoids pandas/Python 3.13 dtype errors when some bridge methods
    # resolve rows and later methods have empty/NA assignments.
    out["canonical_coin_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")

    def apply_mapped(mapped: pd.Series, method: str) -> None:
        mapped_num = pd.to_numeric(mapped, errors="coerce").astype("Int64")
        mask = out["canonical_coin_id"].isna() & mapped_num.notna()
        if bool(mask.any()):
            out["canonical_coin_id"] = out["canonical_coin_id"].where(~mask, mapped_num)

    if "coin_id" in out.columns:
        cid = pd.to_numeric(out["coin_id"], errors="coerce")
        mapped = cid.map(
            lambda x: bridge["id_map"].get(int(x), pd.NA)
            if pd.notna(x) and int(x) in bridge["id_map"]
            else pd.NA
        )
        apply_mapped(mapped, "coin_id_to_coins_id")

    if "pair_address" in out.columns:
        pair = out["pair_address"].map(norm)
        mapped = pair.map(lambda x: bridge["pair_map"].get(x, pd.NA))
        apply_mapped(mapped, "pair_address_to_coins_pair_address")

    if "symbol" in out.columns:
        sym = out["symbol"].map(norm)
        mapped = sym.map(lambda x: bridge["symbol_map"].get(x, pd.NA))
        apply_mapped(mapped, "symbol_unique_to_coins_symbol")

    if "timestamp" in out.columns:
        out["event_time_utc"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    else:
        out["event_time_utc"] = pd.NaT

    return out

def time_range(df: pd.DataFrame, table: str) -> dict[str, Any]:
    x = df[df["canonical_coin_id"].notna() & df["event_time_utc"].notna()].copy()
    return {
        "table": table,
        "rows": int(len(df)),
        "resolved_time_rows": int(len(x)),
        "unique_coins": int(x["canonical_coin_id"].nunique(dropna=True)) if len(x) else 0,
        "min_time": str(x["event_time_utc"].min()) if len(x) else None,
        "max_time": str(x["event_time_utc"].max()) if len(x) else None,
    }


def build_signal_event_strategies(signals: pd.DataFrame) -> pd.DataFrame:
    s = signals[signals["canonical_coin_id"].notna() & signals["event_time_utc"].notna()].copy()
    s["bucket"] = s["event_time_utc"].dt.floor(f"{EVENT_BUCKET_MINUTES}min")
    s = s.sort_values(["canonical_coin_id", "bucket", "event_time_utc"])

    g = s.groupby(["canonical_coin_id", "bucket"], dropna=False)

    rows = []
    first = g.first().reset_index()
    last = g.last().reset_index()
    size = g.size().reset_index(name="signal_rows")

    merged = first[["canonical_coin_id", "bucket", "event_time_utc"]].rename(columns={"event_time_utc": "first_signal_time"})
    merged = merged.merge(
        last[["canonical_coin_id", "bucket", "event_time_utc"]].rename(columns={"event_time_utc": "last_signal_time"}),
        on=["canonical_coin_id", "bucket"],
        how="left",
    )
    merged = merged.merge(size, on=["canonical_coin_id", "bucket"], how="left")
    merged["bucket_start_time"] = merged["bucket"]
    merged["bucket_end_minus_1s"] = merged["bucket"] + pd.Timedelta(minutes=EVENT_BUCKET_MINUTES) - pd.Timedelta(seconds=1)

    for strategy, col in [
        ("first_signal_time", "first_signal_time"),
        ("last_signal_time", "last_signal_time"),
        ("bucket_start_time", "bucket_start_time"),
        ("bucket_end_minus_1s", "bucket_end_minus_1s"),
    ]:
        tmp = merged[["canonical_coin_id", "bucket", "signal_rows", col]].copy()
        tmp = tmp.rename(columns={col: "candidate_time"})
        tmp["strategy"] = strategy
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True)



def datetime_ns_array(series: pd.Series) -> np.ndarray:
    """
    Robust UTC nanosecond conversion.

    Avoids pandas/backend-dependent behavior where astype("int64") may return
    a unit that is not nanoseconds, causing ~56-year lag artifacts.
    """
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    dt_naive_utc = dt.dt.tz_convert("UTC").dt.tz_localize(None)
    return dt_naive_utc.astype("datetime64[ns]").astype("int64").to_numpy()


def timestamp_ns(value: Any) -> int:
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.value)

def market_future_diagnostics(events: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    m = market[market["canonical_coin_id"].notna() & market["event_time_utc"].notna()].copy()
    m = m.sort_values(["canonical_coin_id", "event_time_utc"])

    m_by_coin = {
        int(cid): datetime_ns_array(grp["event_time_utc"])
        for cid, grp in m.groupby("canonical_coin_id")
    }

    out_rows = []

    for strategy, df in events.groupby("strategy"):
        n = len(df)
        prev_exists = 0
        next_exists = 0
        future_4h_exists = 0
        past_24h_exists = 0

        prev_lags = []
        next_lags = []
        future_counts = []

        for _, r in df.iterrows():
            cid = int(r["canonical_coin_id"])
            t = pd.Timestamp(r["candidate_time"])
            if cid not in m_by_coin or pd.isna(t):
                continue

            arr = m_by_coin[cid]
            t_ns = timestamp_ns(t)
            h_ns = (t + pd.Timedelta(minutes=HORIZON_MINUTES)).value
            p24_ns = (t - pd.Timedelta(hours=24)).value

            prev_idx = np.searchsorted(arr, t_ns, side="right") - 1
            next_idx = np.searchsorted(arr, t_ns, side="right")

            if prev_idx >= 0:
                prev_exists += 1
                prev_lags.append((t_ns - arr[prev_idx]) / 1e9 / 60.0)

            if next_idx < len(arr):
                next_exists += 1
                next_lags.append((arr[next_idx] - t_ns) / 1e9 / 60.0)

            start = np.searchsorted(arr, t_ns, side="right")
            end = np.searchsorted(arr, h_ns, side="right")
            fcnt = max(0, end - start)
            future_counts.append(fcnt)
            if fcnt > 0:
                future_4h_exists += 1

            lo = np.searchsorted(arr, p24_ns, side="left")
            hi = np.searchsorted(arr, t_ns, side="right")
            if max(0, hi - lo) > 0:
                past_24h_exists += 1

        def med(vals):
            return None if not vals else float(np.median(vals))

        out_rows.append({
            "strategy": strategy,
            "candidate_events": n,
            "has_previous_market_rows": prev_exists,
            "previous_market_rate": prev_exists / n if n else None,
            "median_previous_market_lag_minutes": med(prev_lags),
            "has_next_market_rows": next_exists,
            "next_market_rate": next_exists / n if n else None,
            "median_next_market_lag_minutes": med(next_lags),
            "has_future_market_rows_4h": future_4h_exists,
            "future_market_4h_rate": future_4h_exists / n if n else None,
            "median_future_rows_4h": med(future_counts),
            "has_past_market_rows_24h": past_24h_exists,
            "past_market_24h_rate": past_24h_exists / n if n else None,
        })

    return pd.DataFrame(out_rows).sort_values("future_market_4h_rate", ascending=False)


def context_before_after(events: pd.DataFrame, context: pd.DataFrame, context_name: str) -> pd.DataFrame:
    c = context[context["canonical_coin_id"].notna() & context["event_time_utc"].notna()].copy()
    c = c.sort_values(["canonical_coin_id", "event_time_utc"])

    c_by_coin = {
        int(cid): datetime_ns_array(grp["event_time_utc"])
        for cid, grp in c.groupby("canonical_coin_id")
    }

    rows = []

    for strategy, df in events.groupby("strategy"):
        n = len(df)
        past24 = 0
        past168 = 0
        next24 = 0

        for _, r in df.iterrows():
            cid = int(r["canonical_coin_id"])
            t = pd.Timestamp(r["candidate_time"])
            if cid not in c_by_coin or pd.isna(t):
                continue

            arr = c_by_coin[cid]
            t_ns = timestamp_ns(t)
            p24 = (t - pd.Timedelta(hours=24)).value
            p168 = (t - pd.Timedelta(hours=168)).value
            n24 = (t + pd.Timedelta(hours=24)).value

            if np.searchsorted(arr, t_ns, side="right") - np.searchsorted(arr, p24, side="left") > 0:
                past24 += 1
            if np.searchsorted(arr, t_ns, side="right") - np.searchsorted(arr, p168, side="left") > 0:
                past168 += 1
            if np.searchsorted(arr, n24, side="right") - np.searchsorted(arr, t_ns, side="right") > 0:
                next24 += 1

        rows.append({
            "context_source": context_name,
            "strategy": strategy,
            "candidate_events": n,
            "past_24h_context_events": past24,
            "past_24h_coverage_rate": past24 / n if n else None,
            "past_168h_context_events": past168,
            "past_168h_coverage_rate": past168 / n if n else None,
            "next_24h_context_events": next24,
            "next_24h_coverage_rate": next24 / n if n else None,
        })

    return pd.DataFrame(rows)


def main() -> None:
    summary = {
        "classification": "THESIS_CONTEXT_TIME_ALIGNMENT_DIAGNOSTIC_COMPLETED",
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
        "event_bucket_minutes": EVENT_BUCKET_MINUTES,
        "horizon_minutes": HORIZON_MINUTES,
    }

    con = connect_ro()

    coins = read_cols(con, "coins", ["id", "symbol", "name", "chain", "pair_address"])
    bridge = build_bridge(coins)

    raw = {
        "signals": read_cols(con, "signals", ["id", "coin_id", "symbol", "timestamp", "score", "confidence", "signal_type", "model_source", "reason"]),
        "market_snapshots": read_cols(con, "market_snapshots", ["id", "coin_id", "symbol", "chain", "pair_address", "timestamp", "price", "price_usd"]),
        "raw_provider_payloads": read_cols(con, "raw_provider_payloads", ["id", "timestamp", "provider", "source", "symbol", "chain", "pair_address", "url"]),
        "whale_alerts": read_cols(con, "whale_alerts", ["id", "coin_id", "symbol", "chain", "pair_address", "timestamp", "whale_score", "alert_type"]),
        "gemini_decisions": read_cols(con, "gemini_decisions", ["id", "coin_id", "symbol", "timestamp", "decision", "action", "risk_score", "confidence"]),
    }

    con.close()

    resolved = {k: resolve(v, bridge) for k, v in raw.items()}

    ranges = pd.DataFrame([time_range(df, k) for k, df in resolved.items()])
    ranges.to_csv(OUT / "00_time_ranges_by_source.csv", index=False, encoding="utf-8-sig")

    events = build_signal_event_strategies(resolved["signals"])
    events.to_csv(OUT / "01_signal_event_time_strategies.csv", index=False, encoding="utf-8-sig")

    market_diag = market_future_diagnostics(events, resolved["market_snapshots"])
    market_diag.to_csv(OUT / "02_market_future_availability_by_event_strategy.csv", index=False, encoding="utf-8-sig")

    ctx_parts = []
    for name in ["raw_provider_payloads", "whale_alerts", "gemini_decisions"]:
        ctx_parts.append(context_before_after(events, resolved[name], name))
    ctx_diag = pd.concat(ctx_parts, ignore_index=True)
    ctx_diag.to_csv(OUT / "03_context_before_after_by_event_strategy.csv", index=False, encoding="utf-8-sig")

    best_strategy = None
    if not market_diag.empty:
        best_strategy = market_diag.sort_values(
            ["future_market_4h_rate", "previous_market_rate"],
            ascending=False,
        ).iloc[0].to_dict()

    summary.update({
        "best_market_label_strategy": best_strategy,
        "outputs": {
            "time_ranges_by_source": str(OUT / "00_time_ranges_by_source.csv"),
            "signal_event_time_strategies": str(OUT / "01_signal_event_time_strategies.csv"),
            "market_future_availability_by_event_strategy": str(OUT / "02_market_future_availability_by_event_strategy.csv"),
            "context_before_after_by_event_strategy": str(OUT / "03_context_before_after_by_event_strategy.csv"),
        },
    })

    with open(OUT / "thesis_context_time_alignment_diagnostic_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Thesis Context Time Alignment Diagnostic")
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
    lines.append("## Source time ranges")
    for _, r in ranges.iterrows():
        lines.append(
            f"- `{r['table']}`: rows={int(r['rows']):,}, "
            f"resolved_time_rows={int(r['resolved_time_rows']):,}, "
            f"unique_coins={int(r['unique_coins'])}, "
            f"min={r['min_time']}, max={r['max_time']}"
        )
    lines.append("")
    lines.append("## Market label availability by event-time strategy")
    for _, r in market_diag.iterrows():
        lines.append(
            f"- `{r['strategy']}`: events={int(r['candidate_events']):,}, "
            f"prev_market={float(r['previous_market_rate'])*100:.2f}%, "
            f"median_prev_lag_min={r['median_previous_market_lag_minutes']}, "
            f"next_market={float(r['next_market_rate'])*100:.2f}%, "
            f"median_next_lag_min={r['median_next_market_lag_minutes']}, "
            f"future_4h={float(r['future_market_4h_rate'])*100:.2f}%, "
            f"median_future_rows_4h={r['median_future_rows_4h']}, "
            f"past_market_24h={float(r['past_market_24h_rate'])*100:.2f}%"
        )
    lines.append("")
    lines.append("## Context before/after availability by event-time strategy")
    for _, r in ctx_diag.iterrows():
        lines.append(
            f"- `{r['context_source']}` / `{r['strategy']}`: "
            f"past24={float(r['past_24h_coverage_rate'])*100:.2f}%, "
            f"past168={float(r['past_168h_coverage_rate'])*100:.2f}%, "
            f"next24={float(r['next_24h_coverage_rate'])*100:.2f}%"
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "If first-signal or bucket-start has future labels but last-signal does not, "
        "the dataset builder should be changed to a no-lookahead first-event representative rather than last signal in bucket."
    )
    lines.append("")
    lines.append("## Output files")
    for _, path in summary["outputs"].items():
        lines.append(f"- `{Path(path).name}`")

    md = OUT / "thesis_context_time_alignment_diagnostic_summary.md"
    md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(OUT / "thesis_context_time_alignment_diagnostic_summary.json"),
        "summary_md": str(md),
    }, indent=2, ensure_ascii=False))
    print()
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
