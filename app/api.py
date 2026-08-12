"""
FastAPI application — all routes.
Serves the dashboard HTML at / and the API at /api/*.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import logging

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import database as db
from .dexscreener import get_trending_pairs, search_pairs
from .engine import compute_whale_score, detect_whale_alert, generate_signal
from .execution.paper import get_paper_trader
from .analytics.sentiment import fetch_rss_sentiment_matrix
from .models import LOG_PATH, WhaleActivity
from .models.predictor import (
    MODEL_NAME,
    TradeDecision,
    analyze_market_state,
    avg_whale_score,
    count_by_cluster,
    execute_trade_decision,
    get_chat_service,
    read_whale_log_rows,
)
from .analytics.watchlist import (
    add_to_watchlist,
    disable_watchlist_item,
    enable_watchlist_item,
    list_watchlist,
    mark_analyzed,
    remove_watchlist_item,
    upsert_watchlist_item,
)
from .live import analyze_contract_address, get_token_transparency_logs

STATIC_DIR = Path(__file__).parent.parent / "static"
log = logging.getLogger("api")

app = FastAPI(title="MemeCoin Whale Trader", version="2.0.0")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_ok(data: Any) -> Any:
    """AE13I Smoke Addendum (Part D): sanitize a response payload to ASCII-safe text.

    Recursively repairs mojibake and normalizes em-dash/en-dash/ellipsis/middle-dot
    punctuation to ASCII before the payload is returned to the client. Never
    raises — falls back to the original payload if sanitization fails for any
    reason, so this can never turn a working endpoint into a 500.
    """
    try:
        from app.ae13b_product.text_sanitizer import sanitize_payload

        return sanitize_payload(data)
    except Exception as exc:
        log.warning("response sanitizer failed; returning unsanitized payload: %s", exc, exc_info=True)
        return data


# ── Dashboard HTML ────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def dashboard():
    return FileResponse(
        str(STATIC_DIR / "index.html"),
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/healthz")
async def health():
    return {"status": "ok", "service": "memecoin-whale-trader-python"}


# ── Coins ─────────────────────────────────────────────────────────────────────

@app.get("/api/coins")
def list_coins(
    limit: int = Query(50, ge=1, le=200),
    sort_by: str = Query("whale_score"),
    chain: str | None = Query(None),
):
    return db.get_coins(limit=limit, sort_by=sort_by, chain=chain)


@app.get("/api/coins/refresh")
async def refresh_hint():
    return {"message": "Use POST /api/coins/refresh to trigger a scan"}


@app.post("/api/coins/refresh")
async def refresh_coins(body: dict[str, Any] = {}):
    from .analytics.scan_persist import archive_dexscreener_search, new_scan_id, persist_pair_pipeline

    MIN_LIQ = 5_000.0
    scan_id = new_scan_id()

    symbols = (body or {}).get("symbols")
    if symbols:
        results = await asyncio.gather(*[search_pairs(s) for s in symbols], return_exceptions=True)
        seen: set[str] = set()
        pairs: list[dict] = []
        for r in results:
            if isinstance(r, Exception):
                continue
            for p in r:
                if p.get("pairAddress") not in seen and p.get("priceUsd"):
                    seen.add(p["pairAddress"])
                    pairs.append(p)
    else:
        pairs = await get_trending_pairs()

    archive_dexscreener_search(
        "refresh_batch",
        {"pair_count": len(pairs), "scan_id": scan_id},
        source_type="refresh_summary",
    )

    count = 0
    for pair in pairs:
        if not pair.get("priceUsd"):
            continue
        liq = float((pair.get("liquidity") or {}).get("usd") or 0)
        filter_status = "passed" if liq >= MIN_LIQ else "dropped"
        drop_reason = None if filter_status == "passed" else f"Liquidity under ${MIN_LIQ:,.0f}"

        try:
            persisted = await asyncio.to_thread(
                persist_pair_pipeline,
                pair,
                scan_id=scan_id,
                source_query="refresh",
                filter_status=filter_status,
                drop_reason=drop_reason or "",
            )
            if persisted and filter_status == "passed":
                count += 1
        except Exception:
            continue

    return {
        "count": count,
        "scan_id": scan_id,
        "message": f"Refreshed {count} coins from DexScreener with full persistence",
        "storage": db.get_storage_stats(),
    }


@app.get("/api/coins/{coin_id}")
def get_coin(coin_id: int):
    coin = db.get_coin_by_id(coin_id)
    if not coin:
        raise HTTPException(status_code=404, detail="Coin not found")
    return coin


@app.get("/api/coins/{coin_id}/detail")
def get_coin_detail(coin_id: int):
    detail = db.get_coin_detail(coin_id)
    if not detail:
        raise HTTPException(status_code=404, detail="Coin not found")
    return detail


@app.get("/api/coins/{coin_id}/snapshots")
def coin_snapshots(coin_id: int, limit: int = Query(200, ge=1, le=2000)):
    if not db.get_coin_by_id(coin_id):
        raise HTTPException(status_code=404, detail="Coin not found")
    return {"coin_id": coin_id, "snapshots": db.get_market_snapshots(coin_id, limit=limit)}


@app.get("/api/coins/{coin_id}/chart")
def coin_chart(coin_id: int, interval: str = Query("1m")):
    if not db.get_coin_by_id(coin_id):
        raise HTTPException(status_code=404, detail="Coin not found")
    candles = db.derive_chart_candles(coin_id, interval=interval)
    return {
        "coin_id": coin_id,
        "interval": interval,
        "derived": True,
        "derivation_note": "snapshot-derived OHLC from stored market_snapshots",
        "candles": candles,
    }


@app.get("/api/coins/{coin_id}/signals")
def coin_signals(coin_id: int, limit: int = Query(50, ge=1, le=200)):
    if not db.get_coin_by_id(coin_id):
        raise HTTPException(status_code=404, detail="Coin not found")
    return {"coin_id": coin_id, "signals": db.get_signals(limit=limit, coin_id=coin_id)}


@app.get("/api/coins/{coin_id}/whale-alerts")
def coin_whale_alerts(coin_id: int, limit: int = Query(50, ge=1, le=200)):
    if not db.get_coin_by_id(coin_id):
        raise HTTPException(status_code=404, detail="Coin not found")
    return {
        "coin_id": coin_id,
        "alerts": db.get_whale_alerts(limit=limit, coin_id=coin_id),
        "note": "aggregate whale-like market flow — not wallet-level unless is_real_wallet_level=true",
    }


@app.get("/api/coins/{coin_id}/gemini-decisions")
def coin_gemini_decisions(coin_id: int, limit: int = Query(50, ge=1, le=200)):
    if not db.get_coin_by_id(coin_id):
        raise HTTPException(status_code=404, detail="Coin not found")
    return {"coin_id": coin_id, "decisions": db.get_gemini_decisions(limit=limit, coin_id=coin_id)}


@app.get("/api/raw/recent")
def raw_recent(limit: int = Query(50, ge=1, le=200), provider: str | None = None):
    return {"payloads": db.get_recent_raw_payloads(limit=limit, provider=provider)}


@app.get("/api/debug/storage")
def debug_storage():
    return db.get_storage_stats()


@app.get("/api/debug/collection")
def debug_collection():
    """Verify headless collection: row counts, newest timestamps, LLM skip/call counters."""
    return db.get_collection_debug_status()


@app.get("/api/debug/training-dataset")
def debug_training_dataset():
    """Latest training_dataset_summary.json from offline dataset builder."""
    from .training.dataset_builder import load_training_summary

    summary = load_training_summary()
    if summary is None:
        return {
            "status": "not_built_yet",
            "message": "Run python scripts/build_training_dataset.py first.",
        }
    return {"status": "ok", **summary}


@app.get("/api/debug/training-dataset/build-status")
def debug_training_dataset_build_status():
    """Scheduler status for automatic and manual training dataset builds."""
    from .training.scheduler import public_build_status

    return public_build_status()


@app.post("/api/debug/training-dataset/build-now")
def debug_training_dataset_build_now():
    """Trigger a background training dataset build if none is running."""
    from .training.scheduler import get_training_scheduler

    result = get_training_scheduler().request_build("api")
    return {"status": result}


@app.get("/api/debug/training-dataset/baseline-metrics")
def debug_training_baseline_metrics():
    """Latest baseline_metrics.json from offline sklearn training."""
    from .training.baseline_model import load_baseline_metrics

    metrics = load_baseline_metrics()
    if metrics is None:
        return {
            "status": "not_trained_yet",
            "message": "Baseline models have not been trained yet.",
            "manual_command": "python scripts/train_baseline_model.py",
        }
    return {"status": "ok", **metrics}


@app.get("/api/pipeline/audit")
def pipeline_audit(
    scan_id: str | None = None,
    filter_status: str | None = None,
    limit: int = Query(200, ge=1, le=1000),
):
    """Before/after filter transparency from SQLite pipeline_audit table."""
    rows = db.get_pipeline_audit(scan_id=scan_id, limit=limit, filter_status=filter_status)
    return {"rows": rows, "count": len(rows)}


@app.get("/api/pipeline/audit/recent")
def pipeline_audit_recent(
    minutes: int = Query(30, ge=1, le=180),
    limit: int = Query(500, ge=1, le=2000),
):
    """Pipeline audit rows and JSONL decision traces from the last N minutes."""
    from datetime import datetime, timedelta, timezone
    from pathlib import Path

    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()
    rows = db.get_pipeline_audit(limit=limit)
    recent_rows = [r for r in rows if (r.get("timestamp") or "") >= cutoff]

    audits_dir = Path(__file__).parent.parent / "data" / "audits"
    jsonl_events: list[dict] = []
    if audits_dir.is_dir():
        import json
        for path in sorted(audits_dir.glob("decision_trace_*.jsonl"), reverse=True)[:3]:
            try:
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        if (obj.get("timestamp") or "") >= cutoff:
                            jsonl_events.append(obj)
            except (OSError, json.JSONDecodeError):
                continue

    reason_counts: dict[str, int] = {}
    for r in recent_rows + jsonl_events:
        for reason in (r.get("audit_reasons") or r.get("audit_reasons_json") or []):
            if isinstance(reason, str):
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    return {
        "minutes": minutes,
        "cutoff": cutoff,
        "pipeline_audit_count": len(recent_rows),
        "jsonl_trace_count": len(jsonl_events),
        "reason_counts": reason_counts,
        "pipeline_audit_sample": recent_rows[:100],
        "jsonl_trace_sample": jsonl_events[:50],
    }


# ── Signals ───────────────────────────────────────────────────────────────────

@app.get("/api/signals")
def list_signals(
    limit: int = Query(50, ge=1, le=200),
    action: str | None = Query(None),
):
    return db.get_signals(limit=limit, action=action)


# ── Positions ─────────────────────────────────────────────────────────────────

class OpenPositionBody(BaseModel):
    coin_id: int
    size_usd: float | None = None
    cluster_label: str | None = None




# PORTFOLIO_LATEST_DB_MARK_PRICE_FIXED_V1
def _portfolio_latest_db_mark_price_v1(position: dict[str, Any]) -> dict[str, Any] | None:
    """Return the newest exact-pair mark price known to SQLite for an open portfolio position.

    This is display/portfolio mark-to-market only. It does not create orders, does not
    approve trades, and does not touch wallet/live execution.
    """
    try:
        import sqlite3
        from datetime import datetime, timezone
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        db_path = root / "data" / "trader.db"
        if not db_path.exists():
            return None

        pair_address = str(position.get("pair_address") or position.get("pair_address_derived") or "").strip()
        coin_id = position.get("coin_id")
        if not pair_address and coin_id is None:
            return None

        def _parse_ts(v: Any):
            if not v:
                return None
            try:
                ts = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return ts
            except Exception:
                return None

        def _float_pos(v: Any):
            try:
                x = float(v)
                return x if x > 0 else None
            except Exception:
                return None

        candidates: list[dict[str, Any]] = []

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            if pair_address:
                rows = conn.execute(
                    """
                    SELECT price, timestamp, source_query, pair_address, coin_id, chain
                    FROM market_snapshots
                    WHERE pair_address = ?
                      AND price IS NOT NULL
                      AND CAST(price AS REAL) > 0
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (pair_address,),
                ).fetchall()
                for r in rows:
                    candidates.append({
                        "price": r["price"],
                        "timestamp": r["timestamp"],
                        "source": "market_snapshots:exact_pair",
                        "source_query": r["source_query"],
                        "pair_address": r["pair_address"],
                        "coin_id": r["coin_id"],
                        "chain": r["chain"],
                    })

            if coin_id is not None:
                rows = conn.execute(
                    """
                    SELECT price, timestamp, source_query, pair_address, coin_id, chain
                    FROM market_snapshots
                    WHERE coin_id = ?
                      AND price IS NOT NULL
                      AND CAST(price AS REAL) > 0
                    ORDER BY timestamp DESC
                    LIMIT 1
                    """,
                    (coin_id,),
                ).fetchall()
                for r in rows:
                    candidates.append({
                        "price": r["price"],
                        "timestamp": r["timestamp"],
                        "source": "market_snapshots:coin_id",
                        "source_query": r["source_query"],
                        "pair_address": r["pair_address"],
                        "coin_id": r["coin_id"],
                        "chain": r["chain"],
                    })

            if pair_address:
                rows = conn.execute(
                    """
                    SELECT latest_price, last_seen_at, pair_address, id AS coin_id, chain
                    FROM coins
                    WHERE pair_address = ?
                      AND latest_price IS NOT NULL
                      AND CAST(latest_price AS REAL) > 0
                    ORDER BY last_seen_at DESC
                    LIMIT 1
                    """,
                    (pair_address,),
                ).fetchall()
                for r in rows:
                    candidates.append({
                        "price": r["latest_price"],
                        "timestamp": r["last_seen_at"],
                        "source": "coins:exact_pair",
                        "source_query": "coins.latest_price",
                        "pair_address": r["pair_address"],
                        "coin_id": r["coin_id"],
                        "chain": r["chain"],
                    })

        finally:
            conn.close()

        usable = []
        for c in candidates:
            px = _float_pos(c.get("price"))
            ts = _parse_ts(c.get("timestamp"))
            if px is not None and ts is not None:
                usable.append({**c, "price": px, "_ts": ts})

        if not usable:
            return None

        usable.sort(key=lambda x: x["_ts"], reverse=True)
        best = usable[0]

        existing_ts = (
            _parse_ts(position.get("price_updated_at"))
            or _parse_ts(position.get("current_price_timestamp"))
        )

        # Only overwrite display marks when DB has a newer mark or the position lacks a timestamp.
        if existing_ts is not None and best["_ts"] <= existing_ts:
            return None

        return {
            "price": float(best["price"]),
            "timestamp": best["_ts"].isoformat(),
            "source": best["source"],
            "source_query": best.get("source_query"),
            "pair_address": best.get("pair_address"),
            "coin_id": best.get("coin_id"),
            "chain": best.get("chain"),
        }
    except Exception:
        return None


def _apply_portfolio_latest_db_mark_price_v1(position: dict[str, Any]) -> None:
    latest = _portfolio_latest_db_mark_price_v1(position)
    if not latest:
        return

    px = float(latest["price"])
    ts = latest["timestamp"]

    position["current_price"] = px
    position["current_price_numeric"] = px
    position["current_price_usd"] = px
    position["mark_price_usd"] = px
    position["latest_price"] = px

    position["current_price_source"] = latest["source"]
    position["current_price_status"] = "PRICE_OK_LATEST_DB_MARK"
    position["mark_price_status"] = "PRICE_OK_LATEST_DB_MARK"
    position["mark_price_lookup_status"] = "PRICE_OK_LATEST_DB_MARK"
    position["price_updated_at"] = ts
    position["current_price_timestamp"] = ts
    position["price_status_detail"] = (
        "portfolio mark refreshed from latest DB exact-pair/coin mark before display"
    )
    position["mark_price_unavailable_reason"] = ""
    position["price_resolution_failure_reason"] = ""
    position["position_market_data_state"] = "MARKET_DATA_READY"
    position["financial_data_status"] = "READY"
    position["pnl_display_status"] = "ready"
    position["pnl_display_message"] = ""

    try:
        qty = float(position.get("quantity") or 0)
        entry = float(position.get("entry_price") or position.get("fill_price") or 0)
        if qty > 0 and entry > 0:
            pnl_usd = (px - entry) * qty
            pnl_pct = ((px / entry) - 1.0) * 100.0

            position["unrealized_pnl_usd"] = pnl_usd
            position["unrealized_pnl_numeric"] = pnl_usd
            position["unrealized_pnl_display"] = f"${pnl_usd:.4f}"

            position["unrealized_pnl_pct"] = pnl_pct
            position["unrealized_pnl_pct_numeric"] = pnl_pct
            position["unrealized_pnl_pct_display"] = f"{pnl_pct:.1f}%"

            position["position_value_usd"] = qty * px
            position["position_value_display"] = f"${(qty * px):.4f}"
    except Exception:
        pass




# UI_FINANCIAL_FINALIZER_FIXED_V1
def _ui_financial_finalizer_float_v1(v):
    try:
        if v is None or isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            x = float(v)
            return x if x == x and x not in (float("inf"), float("-inf")) else None
        s = str(v).strip().replace("$", "").replace(",", "").replace("%", "")
        if not s or s.upper() in {"N/A", "NA", "NONE", "NULL", "UNAVAILABLE"}:
            return None
        x = float(s)
        return x if x == x and x not in (float("inf"), float("-inf")) else None
    except Exception:
        return None


def _ui_financial_finalizer_parse_ts_v1(v):
    try:
        if not v:
            return None
        from datetime import datetime, timezone
        ts = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return ts
    except Exception:
        return None


def _ui_financial_finalizer_price_display_v1(px):
    try:
        x = float(px)
        if x == 0:
            return "0"
        if abs(x) < 0.0001:
            return f"{x:.12f}".rstrip("0").rstrip(".")
        if abs(x) < 1:
            return f"{x:.8f}".rstrip("0").rstrip(".")
        return f"{x:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return str(px)


def _ui_financial_finalizer_is_position_like_v1(d):
    if not isinstance(d, dict):
        return False
    if not ("id" in d or "position_id" in d):
        return False
    if not ("quantity" in d and ("entry_price" in d or "fill_price" in d)):
        return False

    # Do not rewrite realized historical/closed positions.
    status = str(d.get("status") or "").upper()
    if status in {"CLOSED", "SOLD", "EXITED"}:
        return False
    if d.get("closed_at") or d.get("close_price"):
        return False

    return True


def _ui_financial_finalizer_latest_db_mark_v1(position, cache):
    try:
        import sqlite3
        from pathlib import Path

        pair_address = str(
            position.get("pair_address")
            or position.get("pair_address_derived")
            or ""
        ).strip()

        if not pair_address:
            snap = position.get("entry_continuity_snapshot")
            if isinstance(snap, dict):
                pair_address = str(
                    snap.get("pair_address")
                    or snap.get("pair_address_derived")
                    or snap.get("provider_pair_url_final_segment_exact")
                    or ""
                ).strip()

        coin_id = position.get("coin_id")
        cache_key = (pair_address, str(coin_id) if coin_id is not None else "")
        if cache_key in cache:
            return cache[cache_key]

        root = Path(__file__).resolve().parents[1]
        db_path = root / "data" / "trader.db"
        if not db_path.exists():
            cache[cache_key] = None
            return None

        candidates = []

        def add_candidate(source, price, ts, row_extra):
            px = _ui_financial_finalizer_float_v1(price)
            parsed_ts = _ui_financial_finalizer_parse_ts_v1(ts)
            if px is None or px <= 0 or parsed_ts is None:
                return
            candidates.append({
                "price": float(px),
                "timestamp": parsed_ts,
                "source": source,
                "row": row_extra or {},
            })

        def table_exists(conn, table):
            return conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone() is not None

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            if table_exists(conn, "market_snapshots"):
                if pair_address:
                    rows = conn.execute(
                        """
                        SELECT price, timestamp, source_query, pair_address, coin_id, chain
                        FROM market_snapshots
                        WHERE pair_address = ?
                          AND price IS NOT NULL
                          AND CAST(price AS REAL) > 0
                        ORDER BY timestamp DESC
                        LIMIT 5
                        """,
                        (pair_address,),
                    ).fetchall()
                    for r in rows:
                        add_candidate("market_snapshots:exact_pair", r["price"], r["timestamp"], dict(r))

                if coin_id is not None:
                    rows = conn.execute(
                        """
                        SELECT price, timestamp, source_query, pair_address, coin_id, chain
                        FROM market_snapshots
                        WHERE coin_id = ?
                          AND price IS NOT NULL
                          AND CAST(price AS REAL) > 0
                        ORDER BY timestamp DESC
                        LIMIT 5
                        """,
                        (coin_id,),
                    ).fetchall()
                    for r in rows:
                        if pair_address and str(r["pair_address"] or "").strip() != pair_address:
                            continue
                        add_candidate("market_snapshots:coin_id_same_pair", r["price"], r["timestamp"], dict(r))

            if table_exists(conn, "coins") and pair_address:
                rows = conn.execute(
                    """
                    SELECT latest_price, last_seen_at, pair_address, id AS coin_id, chain, symbol
                    FROM coins
                    WHERE pair_address = ?
                      AND latest_price IS NOT NULL
                      AND CAST(latest_price AS REAL) > 0
                    ORDER BY last_seen_at DESC
                    LIMIT 5
                    """,
                    (pair_address,),
                ).fetchall()
                for r in rows:
                    add_candidate("coins:exact_pair", r["latest_price"], r["last_seen_at"], dict(r))

            if table_exists(conn, "pipeline_audit") and pair_address:
                rows = conn.execute(
                    """
                    SELECT current_execution_price, timestamp, pair_address, coin_id, chain, symbol
                    FROM pipeline_audit
                    WHERE pair_address = ?
                      AND current_execution_price IS NOT NULL
                      AND CAST(current_execution_price AS REAL) > 0
                    ORDER BY timestamp DESC
                    LIMIT 5
                    """,
                    (pair_address,),
                ).fetchall()
                for r in rows:
                    add_candidate("pipeline_audit:exact_pair_execution_price", r["current_execution_price"], r["timestamp"], dict(r))
        finally:
            conn.close()

        if not candidates:
            cache[cache_key] = None
            return None

        candidates.sort(key=lambda x: x["timestamp"], reverse=True)
        best = candidates[0]
        out = {
            "price": float(best["price"]),
            "timestamp": best["timestamp"].isoformat(),
            "source": best["source"],
        }
        cache[cache_key] = out
        return out
    except Exception:
        return None


def _ui_financial_finalizer_apply_position_v1(d, cache):
    if not _ui_financial_finalizer_is_position_like_v1(d):
        return

    latest = _ui_financial_finalizer_latest_db_mark_v1(d, cache)

    px = None
    ts = None
    source = None

    if latest:
        px = _ui_financial_finalizer_float_v1(latest.get("price"))
        ts = latest.get("timestamp")
        source = latest.get("source")

    if px is None:
        px = (
            _ui_financial_finalizer_float_v1(d.get("current_price"))
            or _ui_financial_finalizer_float_v1(d.get("current_price_usd"))
            or _ui_financial_finalizer_float_v1(d.get("mark_price_usd"))
            or _ui_financial_finalizer_float_v1(d.get("latest_price"))
        )
        ts = d.get("price_updated_at") or d.get("current_price_timestamp")
        source = d.get("current_price_source") or "existing_payload_mark"

    qty = _ui_financial_finalizer_float_v1(d.get("quantity"))
    entry = _ui_financial_finalizer_float_v1(d.get("entry_price") or d.get("fill_price"))

    if px is None or px <= 0:
        return

    d["current_price"] = float(px)
    d["current_price_numeric"] = float(px)
    d["current_price_usd"] = float(px)
    d["mark_price_usd"] = float(px)
    d["latest_price"] = float(px)
    d["current_price_display"] = _ui_financial_finalizer_price_display_v1(px)

    d["current_price_source"] = source
    d["current_price_status"] = "PRICE_OK_UI_FINANCIAL_FINALIZER"
    d["mark_price_status"] = "PRICE_OK_UI_FINANCIAL_FINALIZER"
    d["mark_price_lookup_status"] = "PRICE_OK_UI_FINANCIAL_FINALIZER"

    if ts:
        d["price_updated_at"] = ts
        d["current_price_timestamp"] = ts

    d["financial_data_status"] = "READY"
    d["position_market_data_state"] = "MARKET_DATA_READY"
    d["pnl_display_status"] = "ready"
    d["pnl_display_message"] = ""
    d["mark_price_unavailable_reason"] = ""
    d["price_resolution_failure_reason"] = ""
    d["price_status_detail"] = "financial UI fields finalized from one selected mark price"

    if qty is not None and qty > 0:
        value = float(px) * float(qty)
        d["position_value_usd"] = value
        d["position_value_display"] = f"${value:.4f}"

    if qty is not None and qty > 0 and entry is not None and entry > 0:
        pnl = (float(px) - float(entry)) * float(qty)
        pct = ((float(px) / float(entry)) - 1.0) * 100.0

        d["unrealized_pnl_usd"] = pnl
        d["unrealized_pnl_numeric"] = pnl
        d["unrealized_pnl_display"] = f"${pnl:.4f}"

        # API convention: percent-points. -69.3 means -69.3%, not -0.693.
        d["unrealized_pnl_pct"] = pct
        d["unrealized_pnl_pct_numeric"] = pct
        d["unrealized_pnl_pct_display"] = f"{pct:.1f}%"

        d["ui_financial_fields_finalized"] = True
        d["ui_financial_finalizer_version"] = "v1"


def _finalize_ui_financial_payload_v1(obj, cache=None, depth=0):
    if cache is None:
        cache = {}

    if depth > 12:
        return obj

    try:
        if isinstance(obj, dict):
            _ui_financial_finalizer_apply_position_v1(obj, cache)
            for v in list(obj.values()):
                if isinstance(v, (dict, list)):
                    _finalize_ui_financial_payload_v1(v, cache, depth + 1)
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, (dict, list)):
                    _finalize_ui_financial_payload_v1(item, cache, depth + 1)
    except Exception:
        return obj

    return obj




# TARGETED_UI_FINANCIAL_FINALIZER_V2
def _targeted_ui_financial_float_v2(v):
    try:
        if v is None or isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            x = float(v)
            return x if x == x and x not in (float("inf"), float("-inf")) else None
        text = str(v).strip().replace("$", "").replace(",", "").replace("%", "")
        if not text or text.upper() in {"N/A", "NA", "NONE", "NULL", "UNAVAILABLE"}:
            return None
        x = float(text)
        return x if x == x and x not in (float("inf"), float("-inf")) else None
    except Exception:
        return None


def _targeted_ui_price_display_v2(px):
    try:
        x = float(px)
        if abs(x) < 0.0001:
            return f"{x:.12f}".rstrip("0").rstrip(".")
        if abs(x) < 1:
            return f"{x:.8f}".rstrip("0").rstrip(".")
        return f"{x:.6f}".rstrip("0").rstrip(".")
    except Exception:
        return str(px)


def _targeted_ui_is_open_position_v2(d):
    if not isinstance(d, dict):
        return False
    if not ("id" in d or "position_id" in d):
        return False
    if not ("quantity" in d and ("entry_price" in d or "fill_price" in d)):
        return False
    status = str(d.get("status") or "").upper()
    if status in {"CLOSED", "SOLD", "EXITED"}:
        return False
    if d.get("closed_at") or d.get("close_price"):
        return False
    return True


def _targeted_ui_latest_mark_v2(position, cache):
    try:
        import sqlite3
        from pathlib import Path
        from datetime import datetime, timezone

        def parse_ts(v):
            try:
                if not v:
                    return None
                ts = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                return ts
            except Exception:
                return None

        pair = str(position.get("pair_address") or position.get("pair_address_derived") or "").strip()
        coin_id = position.get("coin_id")
        key = (pair, str(coin_id) if coin_id is not None else "")

        if key in cache:
            return cache[key]

        root = Path(__file__).resolve().parents[1]
        db_path = root / "data" / "trader.db"
        if not db_path.exists():
            cache[key] = None
            return None

        candidates = []

        def add(source, price, ts):
            px = _targeted_ui_financial_float_v2(price)
            parsed = parse_ts(ts)
            if px is not None and px > 0 and parsed is not None:
                candidates.append({
                    "price": float(px),
                    "timestamp": parsed,
                    "source": source,
                })

        def table_exists(conn, table):
            return conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (table,),
            ).fetchone() is not None

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            if table_exists(conn, "market_snapshots"):
                if pair:
                    for r in conn.execute(
                        """
                        SELECT price, timestamp
                        FROM market_snapshots
                        WHERE pair_address = ?
                          AND price IS NOT NULL
                          AND CAST(price AS REAL) > 0
                        ORDER BY timestamp DESC
                        LIMIT 3
                        """,
                        (pair,),
                    ).fetchall():
                        add("market_snapshots:exact_pair", r["price"], r["timestamp"])

                if coin_id is not None:
                    for r in conn.execute(
                        """
                        SELECT price, timestamp, pair_address
                        FROM market_snapshots
                        WHERE coin_id = ?
                          AND price IS NOT NULL
                          AND CAST(price AS REAL) > 0
                        ORDER BY timestamp DESC
                        LIMIT 3
                        """,
                        (coin_id,),
                    ).fetchall():
                        if pair and str(r["pair_address"] or "").strip() != pair:
                            continue
                        add("market_snapshots:coin_id_same_pair", r["price"], r["timestamp"])

            if table_exists(conn, "coins") and pair:
                for r in conn.execute(
                    """
                    SELECT latest_price, last_seen_at
                    FROM coins
                    WHERE pair_address = ?
                      AND latest_price IS NOT NULL
                      AND CAST(latest_price AS REAL) > 0
                    ORDER BY last_seen_at DESC
                    LIMIT 3
                    """,
                    (pair,),
                ).fetchall():
                    add("coins:exact_pair", r["latest_price"], r["last_seen_at"])
        finally:
            conn.close()

        if not candidates:
            cache[key] = None
            return None

        candidates.sort(key=lambda x: x["timestamp"], reverse=True)
        best = candidates[0]
        out = {
            "price": best["price"],
            "timestamp": best["timestamp"].isoformat(),
            "source": best["source"],
        }
        cache[key] = out
        return out
    except Exception:
        return None


def _targeted_ui_finalize_position_v2(d, cache):
    if not _targeted_ui_is_open_position_v2(d):
        return d

    mark = _targeted_ui_latest_mark_v2(d, cache)

    if mark:
        px = _targeted_ui_financial_float_v2(mark.get("price"))
        source = mark.get("source") or "latest_db_mark"
        ts = mark.get("timestamp")
    else:
        px = (
            _targeted_ui_financial_float_v2(d.get("current_price"))
            or _targeted_ui_financial_float_v2(d.get("current_price_usd"))
            or _targeted_ui_financial_float_v2(d.get("mark_price_usd"))
            or _targeted_ui_financial_float_v2(d.get("latest_price"))
        )
        source = d.get("current_price_source") or "existing_payload_mark"
        ts = d.get("price_updated_at") or d.get("current_price_timestamp")

    qty = _targeted_ui_financial_float_v2(d.get("quantity"))
    entry = _targeted_ui_financial_float_v2(d.get("entry_price") or d.get("fill_price"))

    if px is None or px <= 0:
        return d

    d["current_price"] = float(px)
    d["current_price_numeric"] = float(px)
    d["current_price_usd"] = float(px)
    d["mark_price_usd"] = float(px)
    d["latest_price"] = float(px)
    d["current_price_display"] = _targeted_ui_price_display_v2(px)

    d["current_price_source"] = source
    d["current_price_status"] = "PRICE_OK_TARGETED_UI_FINALIZER"
    d["mark_price_status"] = "PRICE_OK_TARGETED_UI_FINALIZER"
    d["mark_price_lookup_status"] = "PRICE_OK_TARGETED_UI_FINALIZER"

    if ts:
        d["price_updated_at"] = ts
        d["current_price_timestamp"] = ts

    d["financial_data_status"] = "READY"
    d["position_market_data_state"] = "MARKET_DATA_READY"
    d["pnl_display_status"] = "ready"
    d["pnl_display_message"] = ""
    d["mark_price_unavailable_reason"] = ""
    d["price_resolution_failure_reason"] = ""

    if qty is not None and qty > 0:
        value = float(px) * float(qty)
        d["position_value_usd"] = value
        d["position_value_numeric"] = value
        d["position_value_display"] = f"${value:.4f}"

    if qty is not None and qty > 0 and entry is not None and entry > 0:
        pnl = (float(px) - float(entry)) * float(qty)
        pct = ((float(px) / float(entry)) - 1.0) * 100.0

        d["unrealized_pnl_usd"] = pnl
        d["unrealized_pnl_numeric"] = pnl
        d["unrealized_pnl_display"] = f"${pnl:.4f}"

        d["unrealized_pnl_pct"] = pct
        d["unrealized_pnl_pct_numeric"] = pct
        d["unrealized_pnl_pct_display"] = f"{pct:.1f}%"

    d["ui_financial_fields_finalized"] = True
    d["ui_financial_finalizer_version"] = "targeted_v2"
    return d


def _targeted_ui_finalize_payload_v2(obj):
    cache = {}

    try:
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    _targeted_ui_finalize_position_v2(item, cache)
            return obj

        if isinstance(obj, dict):
            # Only live/current open positions. Do not touch archive display-only rows.
            rows = obj.get("current_open_positions")
            if isinstance(rows, list):
                for item in rows:
                    if isinstance(item, dict):
                        _targeted_ui_finalize_position_v2(item, cache)

            # If a response contains a single fresh position object.
            if _targeted_ui_is_open_position_v2(obj):
                _targeted_ui_finalize_position_v2(obj, cache)

            return obj
    except Exception:
        return obj

    return obj


class ManualSellBody(BaseModel):
    position_id: int | None = None
    close_price: float | None = None
    close_reason: str | None = None
    close_note: str | None = None
    manual_close_warning_shown: bool | None = None


class ClosePositionBody(BaseModel):
    close_price: float | None = None
    close_reason: str | None = None
    close_note: str | None = None
    manual_close_warning_shown: bool | None = None


_ALLOWED_MANUAL_CLOSE_REASONS = {
    "manual_take_profit",
    "manual_stop_loss",
    "user_exit",
    "risk_reduction",
    "testing",
    "custom",
}


def _normalize_manual_close_reason(
    close_reason: str | None,
    close_note: str | None = None,
) -> tuple[str, str]:
    reason = str(close_reason or "user_exit").strip().lower().replace(" ", "_")
    note = str(close_note or "").strip()
    if reason not in _ALLOWED_MANUAL_CLOSE_REASONS:
        # Treat unknown labels as custom with the raw text preserved in the note.
        if reason and reason != "custom":
            note = (note + " | " if note else "") + f"raw_reason={reason}"
        reason = "custom" if note else "user_exit"
    if reason == "custom" and not note:
        note = "custom manual close"
    return reason, note


def _resolve_manual_close_price_source(
    pos: dict[str, Any],
    trader: Any,
    requested_close_price: float | None,
) -> dict[str, Any]:
    """Explicitly resolve the manual-close price and its provenance (AE13I Smoke Addendum Part A).

    Returns close_price / close_price_source / price_timestamp / close_price_age_seconds.
    close_price_source is always one of: proposed_price, db, provider, mark, entry_price.
    A client-supplied close_price is never independently verified as fresh, so it is
    always labeled "proposed_price" with no server timestamp.
    """
    now = datetime.now(timezone.utc)

    def _age_from_ts(ts_value: Any) -> float | None:
        if not ts_value:
            return None
        try:
            ts = datetime.fromisoformat(str(ts_value).replace("Z", "+00:00"))
        except (ValueError, TypeError):
            return None
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return max(0.0, (now - ts).total_seconds())

    if requested_close_price is not None:
        return {
            "close_price": float(requested_close_price),
            "close_price_source": "proposed_price",
            "price_timestamp": None,
            "close_price_age_seconds": None,
        }

    pair_address = str(pos.get("pair_address") or "")
    coin_rec = db.get_coin_by_pair_address(pair_address) if pair_address else None
    if coin_rec and coin_rec.get("price_usd"):
        ts = coin_rec.get("last_seen_at")
        return {
            "close_price": float(coin_rec["price_usd"]),
            "close_price_source": "db",
            "price_timestamp": ts,
            "close_price_age_seconds": _age_from_ts(ts),
        }

    marked = trader.mark_positions_to_market([pos])
    mark = marked[0] if marked else {}
    if mark.get("current_price") is not None:
        ts = mark.get("current_price_timestamp")
        return {
            "close_price": float(mark["current_price"]),
            "close_price_source": "provider" if mark.get("mark_fresh") else "mark",
            "price_timestamp": ts,
            "close_price_age_seconds": mark.get("price_age_seconds"),
        }

    return {
        "close_price": float(pos.get("entry_price") or 0),
        "close_price_source": "entry_price",
        "price_timestamp": None,
        "close_price_age_seconds": None,
    }


def _manual_close_response(closed: dict[str, Any], wallet: dict[str, Any] | None = None) -> dict[str, Any]:
    from app.ae13b_product.close_freshness import MANUAL_CLOSE_FALLBACK_WARNING

    is_manual = closed.get("closed_by", "user_manual") == "user_manual"
    used_fallback_or_stale = bool(
        closed.get("close_used_fallback_price")
        or closed.get("close_freshness_status") in ("stale", "unknown_or_fallback")
    )
    warning = None
    if is_manual and used_fallback_or_stale:
        warning = MANUAL_CLOSE_FALLBACK_WARNING
    message = f"Closed demo position #{closed.get('id')}."
    if is_manual:
        message = (
            f"Closed demo position #{closed.get('id')}. "
            "Re-entry for this pair is blocked for 1 hour."
        )

    payload = {
        "closed": closed,
        "id": closed.get("id"),
        "position_id": closed.get("id"),
        "symbol": closed.get("symbol"),
        "chain": closed.get("chain"),
        "pair_address": closed.get("pair_address"),
        "closed_by": closed.get("closed_by", "user_manual"),
        "close_reason": closed.get("close_reason"),
        "close_note": closed.get("close_note"),
        "close_price": closed.get("close_price"),
        "close_price_source": closed.get("close_price_source") or closed.get("fill_price_source"),
        "realized_pnl_usd": closed.get("realized_pnl_usd", closed.get("realized_pnl")),
        "realized_pnl_pct": closed.get("realized_pnl_pct"),
        "net_roi_pct": closed.get("net_roi_pct"),
        "fees": closed.get("fees", closed.get("exit_fees")),
        "exit_fees": closed.get("exit_fees"),
        "paper_demo_only": True,
        "not_live_approved": True,
        "not_profitability_evidence": True,
        "trade_authority": "PAPER_DEMO_ONLY",
        "message": message,
        # AE13I manual-close / reentry / freshness disclosure fields
        "manual_close": is_manual,
        "manual_cooldown_expiry": None,
        "reentry_block_seconds": 3600 if is_manual else 300,
        "close_price_age_seconds": closed.get("close_price_age_seconds"),
        "close_freshness_status": closed.get("close_freshness_status"),
        "close_used_fallback_price": closed.get("close_used_fallback_price"),
        "manual_close_warning_shown": closed.get("manual_close_warning_shown"),
        "warning": warning,
    }
    if is_manual:
        try:
            from app.ae13b_product.reentry_blocks import get_manual_cooldown_fields

            cooldown = get_manual_cooldown_fields(
                pair_address=closed.get("pair_address"),
                chain=closed.get("chain"),
                symbol=closed.get("symbol"),
            )
            payload["manual_cooldown_expiry"] = cooldown.get("manual_cooldown_expiry")
        except Exception:
            pass
    if wallet is not None:
        payload["wallet"] = wallet
    return payload


async def _optional_close_body(request: Request) -> ClosePositionBody:
    """Parse JSON body when present; tolerate empty PUT from legacy UI."""
    try:
        raw = await request.body()
    except Exception:
        return ClosePositionBody()
    if not raw or not raw.strip():
        return ClosePositionBody()
    try:
        data = json.loads(raw.decode("utf-8"))
    except Exception:
        return ClosePositionBody()
    if not isinstance(data, dict):
        return ClosePositionBody()
    return ClosePositionBody(
        close_price=data.get("close_price"),
        close_reason=data.get("close_reason"),
        close_note=data.get("close_note"),
        manual_close_warning_shown=data.get("manual_close_warning_shown"),
    )


def _require_demo_mode() -> None:
    trader = get_paper_trader()
    if trader.get_wallet_summary().get("trading_mode", "DEMO") != "DEMO":
        raise HTTPException(
            status_code=400,
            detail="Paper trading requires DEMO mode. Click DEMO in Position Manager.",
        )


def _enforce_paper_demo_execution_guard(*, acceptance: bool = False) -> dict[str, Any]:
    """Server-side fail-closed guard for any paper/demo order path."""
    from app.ae13b_product.execution_guard import (
        DemoExecutionGuardError,
        assert_paper_demo_allowed,
        resolve_runtime_guard_context,
    )

    ctx = resolve_runtime_guard_context()
    flags: dict[str, Any] = {
        "paper_demo_only": True,
        "not_live_approved": True,
        "not_profitability_evidence": True,
    }
    if acceptance:
        flags["demo_acceptance_only"] = True
        flags["not_strategy_evidence"] = True
    try:
        return assert_paper_demo_allowed(
            trading_mode=ctx["trading_mode"],
            live_trading_enabled=ctx["live_trading_enabled"],
            wallet_configured=False,
            private_key_accessed=False,
            real_signing_enabled=False,
            real_submission_enabled=False,
            order_flags=flags,
            demo_acceptance_mode_enabled=ctx.get("demo_acceptance_mode") if acceptance else None,
        )
    except DemoExecutionGuardError as exc:
        raise HTTPException(
            status_code=403,
            detail={
                "error": "demo_execution_guard_rejected",
                "message": str(exc),
                "reasons": exc.reasons,
                "guard": exc.detail,
            },
        ) from exc


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str
    tools_used: list[str]


class AnalyzeRequest(BaseModel):
    metrics: dict[str, Any]
    cluster_label: str
    sentiment_score: float = 0.0
    execute: bool = False


class WatchlistAddBody(BaseModel):
    contract_address: str | None = None
    symbol: str | None = None
    pair: str | None = None
    chain: str | None = None
    note: str | None = None
    expected_category: str | None = None
    name: str | None = None
    display_label: str | None = None
    analyze_now: bool = False


class WatchlistRemoveBody(BaseModel):
    id: str | None = None
    contract_address: str | None = None


class WatchlistActionBody(BaseModel):
    id: str | None = None
    watchlist_id: str | None = None
    pinned: bool | None = None
    user_evidence_url: str | None = None
    user_evidence_note: str | None = None
    user_claimed_social_mission: str | None = None
    user_expected_category: str | None = None
    max_notional: float | None = None
    risk_mode: str | None = None
    # Identity edit fields
    name: str | None = None
    symbol: str | None = None
    pair: str | None = None
    chain: str | None = None
    contract_or_pair_address: str | None = None
    display_label: str | None = None
    tracking_enabled: bool | None = None
    allow_external: bool | None = None
    user_confirmed_external: bool | None = False


class WatchlistFilterQuery(BaseModel):
    filter: str | None = "all"
    filter_mode: str | None = "hide"


class DemoQueueAddBody(BaseModel):
    watchlist_id: str | None = None
    symbol: str | None = None
    pair: str | None = None
    chain: str | None = None
    contract_or_pair_address: str | None = None
    source: str = "watchlist_manual"
    semantic_label: str | None = None
    market_match_status: str | None = None
    max_notional: float | None = None
    # None = inherit the active demo bot preset (AE13G); explicit value always wins.
    risk_mode: str | None = None
    user_hypothesis: str | None = None
    user_evidence_note: str | None = None
    user_claimed_social_mission: str | None = None


class DemoQueueIdBody(BaseModel):
    queue_id: str


class LiveMarketEvalBody(BaseModel):
    symbol: str | None = None
    pair: str | None = None
    pair_address: str | None = None
    contract_address: str | None = None
    chain: str | None = None
    row_key: str | None = None


@app.get("/api/positions")
def list_positions(status: str | None = Query(None)):
    positions = get_paper_trader().get_positions(status)
    if positions is None:
        positions = []
    return _targeted_ui_finalize_payload_v2(positions)


@app.get("/api/paper/mode")
def get_paper_mode():
    trader = get_paper_trader()
    mode = trader.get_wallet_summary().get("trading_mode", "DEMO")
    return {"trading_mode": mode, "wallet": trader.get_wallet_summary()}


@app.post("/api/positions")
def open_position(body: OpenPositionBody):
    return demo_manual_buy(body)


@app.post("/api/demo/buy")
def demo_manual_buy(body: OpenPositionBody):
    """Open a paper position (DEMO mode only) with full fee deduction."""
    _require_demo_mode()
    _enforce_paper_demo_execution_guard(acceptance=False)
    coin = db.get_coin_by_id(body.coin_id)
    if not coin:
        raise HTTPException(status_code=404, detail="Coin not found")

    settings = db.get_settings()
    trader = get_paper_trader()
    coin_normalized = db.normalize_coin_for_trade(coin)
    trader.set_market_prices([
        {
            "pair_address": coin_normalized.get("pair_address"),
            "coin_id": coin_normalized.get("coin_id") or coin_normalized.get("id"),
            "price_usd": coin_normalized.get("price_usd"),
        }
    ])
    pos = trader.open_position(
        coin_normalized,
        size_usd=body.size_usd,
        cluster_label=body.cluster_label or "SOCIALLY_MOTIVATED",
        settings=settings,
        reason_code="MANUAL_BUY",
        allow_coin_price_fallback=True,
    )
    if not pos:
        raise HTTPException(status_code=400, detail="Insufficient demo wallet cash or invalid order")
    return {"position": pos, "wallet": trader.get_wallet_summary(), "paper_demo_only": True, "not_live_approved": True}


class DemoBuyCandidateBody(BaseModel):
    canonical_market_identity: str | None = None
    provider_pair_url_exact: str | None = None
    size_usd: float | None = None
    cluster_label: str | None = None


def _demo_action_result(
    code: str,
    message: str,
    *,
    canonical: str = "",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": False,
        "status": "blocked" if code != "DEMO_ACTION_FAILED_INTERNAL_ERROR" else "error",
        "demo_action_status": code,
        "demo_action_blocked_reason": message,
        "blocked_reason_explicit": True,
        "user_message": message,
        "canonical_market_identity": canonical,
        "canonical_market_identity_type": "PROVIDER_URL",
        "pair_address_required_as_canonical": False,
        "paper_demo_only": True,
        "live_trading_enabled": False,
        "wallet_configured": False,
        "real_signing_enabled": False,
    }
    if extra:
        payload.update(extra)
    return payload


@app.post("/api/ae13b/demo/buy-candidate")
def ae13b_demo_buy_candidate(body: DemoBuyCandidateBody):
    """AE18 demo-only buy for a Market Opportunities candidate (URL-first identity).

    Paper/demo execution only — no wallet, signer, or live order path.
    """
    from app.ae13b_product.runtime_market_feed import find_index_row_by_canonical
    from app.clean_forward.runtime_identity_index import load_runtime_identity_index
    from app.runtime.ui_get_network_guard import ui_get_network_guard

    canonical = str(body.canonical_market_identity or body.provider_pair_url_exact or "").strip()
    if not canonical:
        return _json_ok(
            _demo_action_result(
                "DEMO_ACTION_BLOCKED_IDENTITY_UNRESOLVED",
                "No canonical market URL supplied for this candidate.",
            )
        )
    if not canonical.lower().startswith("http"):
        return _json_ok(
            _demo_action_result(
                "DEMO_ACTION_BLOCKED_IDENTITY_UNRESOLVED",
                "Canonical identity must be a provider market URL (pair_address is not accepted).",
                canonical=canonical,
            )
        )

    try:
        trader = get_paper_trader()
        if trader.get_wallet_summary().get("trading_mode", "DEMO") != "DEMO":
            return _json_ok(
                _demo_action_result(
                    "DEMO_ACTION_BLOCKED_MODE_DISABLED",
                    "Demo mode is not active. Switch to DEMO in Position Manager.",
                    canonical=canonical,
                )
            )
        try:
            guard = _enforce_paper_demo_execution_guard(acceptance=False)
        except HTTPException as exc:
            return _json_ok(
                _demo_action_result(
                    "DEMO_ACTION_BLOCKED_MODE_DISABLED",
                    "Paper/demo execution guard rejected this action.",
                    canonical=canonical,
                    extra={"execution_guard": exc.detail},
                )
            )

        # Candidate lookup is local-index only (no provider calls on this action).
        with ui_get_network_guard("/api/ae13b/demo/buy-candidate"):
            loaded = load_runtime_identity_index()
            index_rows = loaded.get("rows") or []
            row = find_index_row_by_canonical(canonical, index_rows)
        if not row:
            return _json_ok(
                _demo_action_result(
                    "DEMO_ACTION_BLOCKED_CANDIDATE_NOT_FOUND",
                    "Candidate is not present in the runtime canonical identity index. "
                    "Run a manual refresh or rebuild the index.",
                    canonical=canonical,
                )
            )

        activity_status = str(row.get("market_activity_status") or "ACTIVITY_UNKNOWN").strip() or "ACTIVITY_UNKNOWN"
        if activity_status != "ACTIVE_PROVIDER_TXNS":
            return _json_ok(
                _demo_action_result(
                    "DEMO_ACTION_BLOCKED_MARKET_ACTIVITY",
                    row.get("activity_trade_block_reason")
                    or f"{activity_status} — market is not classified as active provider flow.",
                    canonical=canonical,
                    extra={
                        "market_activity_status": activity_status,
                        "activity_trade_readiness_status": row.get("activity_trade_readiness_status"),
                        "activity_trade_block_reason": row.get("activity_trade_block_reason"),
                        "provider_txns_recent_total": row.get("provider_txns_recent_total"),
                        "provider_volume_recent_total": row.get("provider_volume_recent_total"),
                        "provider_price_delta_any_nonzero": row.get("provider_price_delta_any_nonzero"),
                        "display_status": row.get("display_status"),
                        "symbol_pair_display": row.get("symbol_pair_display"),
                    },
                )
            )

        price = row.get("price_usd")
        try:
            price_f = float(price) if price not in (None, "") else 0.0
        except (TypeError, ValueError):
            price_f = 0.0
        if price_f <= 0:
            return _json_ok(
                _demo_action_result(
                    "DEMO_ACTION_BLOCKED_PRICE_UNAVAILABLE",
                    "No cached mark price for this market URL. Refresh the Clean Forward feed first.",
                    canonical=canonical,
                    extra={"mark_price_lookup_status": row.get("mark_price_lookup_status")},
                )
            )

        from app.ae13b_product.runtime_market_feed import apply_index_mark_prices_to_trader

        apply_index_mark_prices_to_trader(trader)

        display = str(row.get("symbol_pair_display") or "")
        coin = {
            "coin_id": None,
            "symbol": display,
            "symbol_pair_display": display,
            "chain": row.get("chain"),
            "dex_id": row.get("dex_id") or row.get("provider_dex_id"),
            "canonical_market_identity": canonical,
            "canonical_market_identity_type": "PROVIDER_URL",
            "provider_pair_url_exact": row.get("provider_pair_url_exact") or canonical,
            "normalized_provider_pair_url_key": row.get("normalized_provider_pair_url_key"),
            "provider_pair_url_final_segment_exact": row.get(
                "provider_pair_url_final_segment_exact"
            ),
            "open_chart_url": row.get("open_chart_url") or canonical,

            # URL-first paper/demo execution identity.
            # pair_address remains a derived/helper field only, never canonical.
            "instrument_id": f"clean_forward:{row.get('chain') or ''}:{canonical}",
            "execution_instrument_id": f"clean_forward:{row.get('chain') or ''}:{canonical}",
            "instrument_source": "clean_forward_market_feed",
            "candidate_source": "clean_forward_market_feed",
            "mark_price_lookup_key": canonical,

            # derived helper only — never the canonical key
            "pair_address": row.get("pair_address_derived"),
            "pair_address_derived": row.get("pair_address_derived"),
            "token_contract_address": row.get("provider_base_token_address")
            or row.get("base_token_address_derived"),
            "base_token_address": row.get("provider_base_token_address"),
            "quote_token_address": row.get("provider_quote_token_address"),
            "provider_base_token_address": row.get("provider_base_token_address"),
            "provider_quote_token_address": row.get("provider_quote_token_address"),

            # Price aliases required by legacy paper/demo risk guards.
            # These all represent the same cached current mark price from the runtime index.
            "price_usd": price_f,
            "latest_price": price_f,
            "price": price_f,
            "market_price_usd": price_f,
            "price_source": "market_canonical_url",
            "entry_price_source": "market_canonical_url",
            "price_timestamp": loaded.get("loaded_at"),
            "last_seen_at": loaded.get("loaded_at"),
            "price_updated_at": loaded.get("loaded_at"),

            # Liquidity/volume aliases required by legacy risk guards.
            "liquidity_usd": row.get("liquidity_usd"),
            "latest_liquidity": row.get("liquidity_usd"),
            "liquidity_at_entry": row.get("liquidity_usd"),
            "volume_24h": row.get("volume_h24"),
            "latest_volume_24h": row.get("volume_h24"),
            "price_change_m5": row.get("price_change_m5"),
            "price_change_h1": row.get("price_change_h1"),
            "price_change_h24": row.get("price_change_h24"),
            "last_updated": loaded.get("loaded_at"),
        }

        settings = db.get_settings()
        pos = trader.open_position(
            coin,
            size_usd=body.size_usd,
            cluster_label=body.cluster_label or "SOCIALLY_MOTIVATED",
            settings=settings,
            reason_code="MANUAL_DEMO_CANDIDATE_BUY",
            allow_coin_price_fallback=True,
        )
        last = trader._state.get("last_open_result") if hasattr(trader, "_state") else None
        if not pos:
            reason = ""
            if isinstance(last, dict):
                reason = str(last.get("rejection_reason") or last.get("rejection_code") or "")
            return _json_ok(
                _demo_action_result(
                    "DEMO_ACTION_BLOCKED_RISK_GATE",
                    reason or "Demo risk gate rejected this order (no position opened).",
                    canonical=canonical,
                    extra={
                        "risk_gate_result": last if isinstance(last, dict) else {},
                        "risk_gate_evaluated": True,
                    },
                )
            )

        return _json_ok(
            {
                "ok": True,
                "status": "opened",
                "demo_action_status": "DEMO_ACTION_OPENED",
                "user_message": f"Demo position opened for {display or canonical}.",
                "position": pos,
                "wallet": trader.get_wallet_summary(),
                "canonical_market_identity": canonical,
                "canonical_market_identity_type": "PROVIDER_URL",
                "pair_address_required_as_canonical": False,
                "risk_gate_evaluated": True,
                "risk_gate_result": last if isinstance(last, dict) else {},
                "execution_guard": guard,
                "paper_demo_only": True,
                "live_trading_enabled": False,
                "wallet_configured": False,
                "real_signing_enabled": False,
                "blocked_reason_explicit": True,
            }
        )
    except Exception as exc:  # noqa: BLE001 - always structured
        return _json_ok(
            _demo_action_result(
                "DEMO_ACTION_FAILED_INTERNAL_ERROR",
                f"Demo action failed: {type(exc).__name__}: {str(exc)[:200]}",
                canonical=canonical,
            )
        )


@app.post("/api/demo/sell")
def demo_manual_sell(body: ManualSellBody):
    """Close a paper position (DEMO mode only) with exit fees and net ROI."""
    _require_demo_mode()
    _enforce_paper_demo_execution_guard(acceptance=False)
    trader = get_paper_trader()
    positions = trader.get_positions(status="OPEN")
    if not positions:
        raise HTTPException(status_code=404, detail="No open positions to sell")

    pos_id = body.position_id
    if pos_id is None:
        pos_id = int(positions[-1]["id"])

    pos = next((p for p in positions if p["id"] == pos_id), None)
    if not pos:
        raise HTTPException(status_code=404, detail="Position not found")

    close_reason, close_note = _normalize_manual_close_reason(body.close_reason, body.close_note)

    from app.ae13b_product.close_freshness import classify_manual_close_freshness

    price_info = _resolve_manual_close_price_source(pos, trader, body.close_price)
    freshness = classify_manual_close_freshness(
        close_price=price_info["close_price"],
        price_timestamp=price_info["price_timestamp"],
        close_price_source=price_info["close_price_source"],
        close_price_age_seconds=price_info["close_price_age_seconds"],
        warning_shown=body.manual_close_warning_shown,
    )
    trader.set_market_prices(
        [
            {
                "pair_address": pos.get("pair_address"),
                "coin_id": pos.get("coin_id"),
                "price_usd": price_info["close_price"],
            }
        ],
        price_timestamp=price_info["price_timestamp"],
    )
    closed = trader.close_position(
        pos_id,
        price_info["close_price"],
        reason_code="MANUAL_SELL",
        proposed_pair_address=pos.get("pair_address"),
        proposed_coin_id=pos.get("coin_id"),
        close_reason=close_reason,
        close_note=close_note,
        closed_by="user_manual",
        close_price_source=price_info["close_price_source"],
        close_price_age_seconds=price_info["close_price_age_seconds"],
        close_freshness_status=freshness["close_freshness_status"],
        close_used_fallback_price=freshness["close_used_fallback_price"],
        manual_close_warning_shown=freshness["manual_close_warning_shown"],
    )
    if not closed:
        raise HTTPException(status_code=500, detail="Failed to close position")
    return _manual_close_response(closed, trader.get_wallet_summary())


@app.put("/api/positions/{pos_id}/close")
async def close_position(
    pos_id: int,
    request: Request,
    close_price: float | None = Query(None),
    close_reason: str | None = Query(None),
    close_note: str | None = Query(None),
):
    """Close a single paper/demo position by ID. Never touches wallet or live trading."""
    _require_demo_mode()
    _enforce_paper_demo_execution_guard(acceptance=False)
    trader = get_paper_trader()
    positions = trader.get_positions(status="OPEN")
    pos = next((p for p in positions if p["id"] == pos_id), None)
    if not pos:
        raise HTTPException(status_code=404, detail="Open position not found")

    body = await _optional_close_body(request)
    requested_close_price = close_price if close_price is not None else body.close_price

    reason_raw = body.close_reason if body.close_reason is not None else close_reason
    note_raw = body.close_note if body.close_note is not None else close_note
    resolved_reason, resolved_note = _normalize_manual_close_reason(reason_raw, note_raw)

    from app.ae13b_product.close_freshness import classify_manual_close_freshness

    price_info = _resolve_manual_close_price_source(pos, trader, requested_close_price)
    freshness = classify_manual_close_freshness(
        close_price=price_info["close_price"],
        price_timestamp=price_info["price_timestamp"],
        close_price_source=price_info["close_price_source"],
        close_price_age_seconds=price_info["close_price_age_seconds"],
        warning_shown=body.manual_close_warning_shown,
    )

    trader.set_market_prices(
        [
            {
                "pair_address": pos.get("pair_address"),
                "coin_id": pos.get("coin_id"),
                "price_usd": price_info["close_price"],
            }
        ],
        price_timestamp=price_info["price_timestamp"],
    )
    closed = trader.close_position(
        pos_id,
        price_info["close_price"],
        reason_code="MANUAL_SELL",
        proposed_pair_address=pos.get("pair_address"),
        proposed_coin_id=pos.get("coin_id"),
        close_reason=resolved_reason,
        close_note=resolved_note,
        closed_by="user_manual",
        close_price_source=price_info["close_price_source"],
        close_price_age_seconds=price_info["close_price_age_seconds"],
        close_freshness_status=freshness["close_freshness_status"],
        close_used_fallback_price=freshness["close_used_fallback_price"],
        manual_close_warning_shown=freshness["manual_close_warning_shown"],
    )
    if not closed:
        raise HTTPException(status_code=500, detail="Failed to close position")
    return _manual_close_response(closed, trader.get_wallet_summary())


@app.post("/api/analyze")
async def analyze_market(body: AnalyzeRequest) -> dict[str, Any]:
    trader = get_paper_trader()
    decision, decision_id = await analyze_market_state(
        body.metrics,
        body.cluster_label,
        body.sentiment_score,
        open_positions=trader.get_positions("OPEN"),
        coin_id=body.metrics.get("coin_id"),
        trigger_type="api_manual_analyze",
    )
    execution = None
    if body.execute and decision.decision in ("BUY", "SELL"):
        settings = db.get_settings()
        coin = {
            "symbol": body.metrics.get("symbol", decision.symbol or "?"),
            "chain": body.metrics.get("network", "solana"),
            "price_usd": float(body.metrics.get("price_usd") or 0),
            "pair_address": body.metrics.get("token_contract_address", ""),
            "coin_id": body.metrics.get("coin_id"),
            "decision_ref_id": decision_id,
        }
        execution = execute_trade_decision(
            decision,
            coin,
            body.cluster_label,
            settings,
            cur_price=float(body.metrics.get("price_usd") or 0) or None,
            decision_ref_id=decision_id,
            coin_id=body.metrics.get("coin_id"),
        )
    return {
        "decision": decision,
        "decision_id": decision_id,
        "execution": execution,
        "wallet": trader.get_wallet_summary(),
    }


@app.get("/api/tokens/raw")
def tokens_raw():
    """Academic transparency: passed vs dropped from latest scan + SQLite pipeline audit."""
    data = get_token_transparency_logs()
    latest_audit = db.get_pipeline_audit(limit=100)
    return {
        "scan_at": data.get("scan_at"),
        "passed": data.get("passed", []),
        "dropped": data.get("dropped", []),
        "passed_count": data.get("passed_count", 0),
        "dropped_count": data.get("dropped_count", 0),
        "pipeline_audit_sample": latest_audit[:50],
        "storage": db.get_storage_stats(),
    }


@app.get("/api/watchlist")
def get_watchlist():
    items = list_watchlist()
    return _json_ok({
        "items": items,
        "count": len(items),
        "paper_demo_only": True,
        "not_live_approved": True,
        "live_trading_implied": False,
        "note": "Watchlist is a research/tracking aid - paper/demo only, not live trading.",
    })


@app.post("/api/watchlist/add")
async def watchlist_add(body: WatchlistAddBody):
    """Add symbol/pair/contract — persisted; optional analysis bypasses liquidity gates."""
    try:
        if body.symbol or body.pair or (body.contract_address and not body.analyze_now):
            entry = upsert_watchlist_item(
                symbol=body.symbol,
                pair=body.pair,
                contract_address=body.contract_address,
                chain=body.chain,
                note=body.note,
                expected_category=body.expected_category,
                name=body.name,
                display_label=body.display_label,
            )
        else:
            addr = (body.contract_address or "").strip()
            if not addr:
                raise ValueError("symbol, pair, or contract_address required")
            entry = add_to_watchlist(addr, body.chain)
            if body.note or body.expected_category or body.symbol:
                entry = upsert_watchlist_item(
                    symbol=body.symbol or entry.get("symbol"),
                    pair=body.pair or entry.get("pair"),
                    contract_address=addr,
                    chain=body.chain or entry.get("chain"),
                    note=body.note,
                    expected_category=body.expected_category,
                )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result: dict[str, Any] = {
        "watchlist_entry": entry,
        "items": list_watchlist(),
        "saved": True,
        "paper_demo_only": True,
        "not_live_approved": True,
        "live_trading_implied": False,
    }

    addr = (body.contract_address or entry.get("contract_address") or "").strip()
    if body.analyze_now and addr:
        analysis = await analyze_contract_address(
            addr,
            body.chain,
            bypass_filters=True,
        )
        result["analysis"] = analysis
        if analysis.get("ok"):
            dec = (analysis.get("decision") or {}).get("decision", "—")
            mark_analyzed(
                addr,
                symbol=analysis.get("symbol"),
                cluster_label=analysis.get("cluster_label"),
                last_decision=dec,
                whale_score=analysis.get("whale_score"),
                filter_note=analysis.get("filter_note", "bypassed_standard_filters"),
                execution_ok=(analysis.get("execution") or {}).get("ok"),
            )
        else:
            mark_analyzed(
                addr,
                last_decision="ERROR",
                analysis_error=analysis.get("error"),
                filter_note="bypassed_standard_filters",
            )
        result["items"] = list_watchlist()

    return result


@app.post("/api/watchlist/remove")
def watchlist_remove(body: WatchlistRemoveBody):
    ok = remove_watchlist_item(body.id, contract_address=body.contract_address)
    if not ok:
        raise HTTPException(status_code=404, detail="Watchlist item not found")
    return {"removed": True, "ok": True, "items": list_watchlist(), "paper_demo_only": True}


@app.post("/api/watchlist/disable")
def watchlist_disable(body: WatchlistRemoveBody):
    if not body.id and not body.contract_address:
        raise HTTPException(status_code=400, detail="id or contract_address required")
    item = None
    if body.id:
        item = disable_watchlist_item(body.id)
    if item is None and body.contract_address:
        for row in list_watchlist(include_disabled=True):
            addr = str(row.get("user_contract_address") or row.get("contract_address") or "").lower()
            pair = str(row.get("user_pair") or row.get("pair") or "").lower()
            target = str(body.contract_address).lower()
            if addr == target or pair == target:
                item = disable_watchlist_item(str(row.get("id")))
                break
    if not item:
        raise HTTPException(status_code=404, detail="Watchlist item not found")
    return {
        "disabled": True,
        "ok": True,
        "item": item,
        "items": list_watchlist(),
        "paper_demo_only": True,
    }


@app.post("/api/watchlist/enable")
def watchlist_enable(body: WatchlistRemoveBody):
    if not body.id:
        raise HTTPException(status_code=400, detail="id required")
    item = enable_watchlist_item(body.id)
    if not item:
        raise HTTPException(status_code=404, detail="Watchlist item not found")
    return {
        "enabled": True,
        "ok": True,
        "item": item,
        "items": list_watchlist(),
        "paper_demo_only": True,
    }


def _wl_id(body: WatchlistActionBody) -> str:
    wid = body.id or body.watchlist_id
    if not wid:
        raise HTTPException(status_code=400, detail="id required")
    return str(wid)


@app.post("/api/watchlist/pin")
def watchlist_pin(body: WatchlistActionBody):
    from app.analytics.watchlist import pin_watchlist_item

    item = pin_watchlist_item(_wl_id(body), pinned=True if body.pinned is None else bool(body.pinned))
    if not item:
        raise HTTPException(status_code=404, detail="Watchlist item not found")
    return {"ok": True, "item": item, "items": list_watchlist(), "paper_demo_only": True}


@app.post("/api/watchlist/resolve")
def watchlist_resolve(body: WatchlistActionBody):
    from app.analytics.watchlist import resolve_watchlist_identity

    result = resolve_watchlist_identity(
        _wl_id(body),
        allow_external=bool(body.allow_external),
        user_confirmed_external=bool(body.user_confirmed_external),
    )
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error") or "not found")
    result["items"] = list_watchlist()
    return result


@app.post("/api/watchlist/identity")
def watchlist_edit_identity(body: WatchlistActionBody):
    """Add / Edit Identity — only path that mutates UserEnteredIdentity."""
    from app.analytics.watchlist import update_watchlist_identity

    item = update_watchlist_identity(
        _wl_id(body),
        name=body.name,
        symbol=body.symbol,
        pair=body.pair,
        chain=body.chain,
        contract_or_pair_address=body.contract_or_pair_address,
        display_label=body.display_label,
    )
    if not item:
        raise HTTPException(status_code=404, detail="Watchlist item not found")
    return {
        "ok": True,
        "item": item,
        "items": list_watchlist(),
        "user_entered_identity": item.get("user_entered_identity"),
        "note": "User-entered identity updated. Resolver enrichment is separate.",
        "paper_demo_only": True,
    }


@app.post("/api/watchlist/track")
def watchlist_track(body: WatchlistActionBody):
    """Track Continuously / Stop Tracking."""
    from app.analytics.watchlist import set_tracking_enabled

    enabled = True if body.tracking_enabled is None else bool(body.tracking_enabled)
    item = set_tracking_enabled(_wl_id(body), enabled=enabled)
    if not item:
        raise HTTPException(status_code=404, detail="Watchlist item not found")
    return {
        "ok": True,
        "item": item,
        "items": list_watchlist(),
        "tracking_enabled": item.get("tracking_enabled"),
        "collection_status": item.get("collection_status"),
        "note": (
            "Tracking enabled — local collection attempts on resolve/refresh."
            if enabled
            else "Tracking stopped — item retained; not removed."
        ),
        "paper_demo_only": True,
        "live_trading_implied": False,
    }


@app.post("/api/watchlist/collect")
def watchlist_collect(body: WatchlistActionBody):
    from app.analytics.watchlist import run_watchlist_collection_attempt

    result = run_watchlist_collection_attempt(_wl_id(body))
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error") or "not found")
    result["items"] = list_watchlist()
    return result


@app.post("/api/watchlist/enable-external-lookup")
def watchlist_enable_external(body: WatchlistActionBody):
    from app.analytics.watchlist import enable_external_lookup_for_watchlist_item

    result = enable_external_lookup_for_watchlist_item(_wl_id(body))
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error") or "not found")
    result["items"] = list_watchlist()
    return result


@app.get("/api/external-resolver/status")
def external_resolver_status():
    from app.ae13b_product.external_resolver import get_external_resolver_status

    return get_external_resolver_status()


@app.post("/api/external-resolver/mode")
def external_resolver_set_mode(body: dict[str, Any]):
    from app.ae13b_product.external_resolver import set_external_resolver_mode

    mode = str((body or {}).get("mode") or "local_only")
    provider = (body or {}).get("provider_name")
    try:
        return set_external_resolver_mode(mode, provider_name=provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/watchlist/semantic-check")
def watchlist_semantic_check(body: WatchlistActionBody):
    from app.analytics.watchlist import run_watchlist_semantic_check

    result = run_watchlist_semantic_check(_wl_id(body))
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error") or "not found")
    result["items"] = list_watchlist()
    return result


@app.post("/api/watchlist/evaluate")
def watchlist_evaluate(body: WatchlistActionBody):
    from app.analytics.watchlist import evaluate_watchlist_item

    result = evaluate_watchlist_item(_wl_id(body))
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error") or "not found")
    result["items"] = list_watchlist()
    return result


@app.post("/api/watchlist/evidence")
def watchlist_evidence(body: WatchlistActionBody):
    from app.analytics.watchlist import set_watchlist_evidence

    item = set_watchlist_evidence(
        _wl_id(body),
        user_evidence_url=body.user_evidence_url,
        user_evidence_note=body.user_evidence_note,
        user_claimed_social_mission=body.user_claimed_social_mission,
        user_expected_category=body.user_expected_category,
    )
    if not item:
        raise HTTPException(status_code=404, detail="Watchlist item not found")
    return {
        "ok": True,
        "item": item,
        "items": list_watchlist(),
        "note": "Evidence saved - SOCIAL_CONFIRMED requires validated supporting evidence.",
        "paper_demo_only": True,
    }


@app.post("/api/watchlist/demo-queue")
def watchlist_add_demo_queue(body: WatchlistActionBody):
    """Add watchlist item to Demo Trade Queue (paper only). Does not place a trade."""
    from app.ae13b_product.demo_queue import add_to_demo_queue, list_demo_queue
    from app.analytics.watchlist import get_watchlist_item, mark_active_demo_candidate

    wid = _wl_id(body)
    item = get_watchlist_item(wid)
    if not item:
        raise HTTPException(status_code=404, detail="Watchlist item not found")
    entry = add_to_demo_queue(
        watchlist_id=str(item.get("watchlist_id") or item.get("id")),
        symbol=item.get("display_symbol") or item.get("user_entered_symbol") or item.get("symbol"),
        pair=item.get("user_entered_pair") or item.get("pair"),
        chain=item.get("display_chain") or item.get("chain"),
        contract_or_pair_address=item.get("display_id")
        or item.get("user_entered_contract_or_pair_address"),
        source="watchlist_manual",
        semantic_label=item.get("semantic_signal_family") or item.get("semantic_classification"),
        market_match_status=item.get("market_match_status"),
        max_notional=body.max_notional,
        # None (default) inherits the active demo bot preset (AE13G preset propagation).
        risk_mode=body.risk_mode,
        user_hypothesis=item.get("user_expected_category") or item.get("expected_category"),
        user_evidence_note=item.get("user_evidence_note"),
        user_claimed_social_mission=item.get("user_claimed_social_mission"),
    )
    mark_active_demo_candidate(wid, active=True, demo_queue_status="queued_for_evaluation")
    return _json_ok({
        "ok": True,
        "queue_entry": entry,
        "queue_items": list_demo_queue(),
        "items": list_watchlist(),
        "note": "Added to Demo Trade Queue - paper only. Risk guard still applies.",
        "paper_demo_only": True,
        "not_live_approved": True,
        "live_trading_implied": False,
    })


@app.get("/api/demo-queue")
def get_demo_queue():
    from app.ae13b_product.demo_queue import list_demo_queue

    items = list_demo_queue()
    return _json_ok({
        "ok": True,
        "items": items,
        "count": len(items),
        "label": "Demo Trade Queue - paper only",
        "strategy_lane": "Manual Watchlist Scout",
        "paper_demo_only": True,
        "not_live_approved": True,
        "live_trading_implied": False,
    })


@app.post("/api/demo-queue/add")
def demo_queue_add(body: DemoQueueAddBody):
    from app.ae13b_product.demo_queue import add_to_demo_queue

    try:
        entry = add_to_demo_queue(
            watchlist_id=body.watchlist_id,
            symbol=body.symbol,
            pair=body.pair,
            chain=body.chain,
            contract_or_pair_address=body.contract_or_pair_address or body.pair,
            source=body.source or "watchlist_manual",
            semantic_label=body.semantic_label,
            market_match_status=body.market_match_status,
            max_notional=body.max_notional,
            # None (default) inherits the active demo bot preset (AE13G preset propagation).
            risk_mode=body.risk_mode,
            user_hypothesis=body.user_hypothesis,
            user_evidence_note=body.user_evidence_note,
            user_claimed_social_mission=body.user_claimed_social_mission,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _json_ok({
        "ok": True,
        "queue_entry": entry,
        "items": __import__("app.ae13b_product.demo_queue", fromlist=["list_demo_queue"]).list_demo_queue(),
        "paper_demo_only": True,
        "not_live_approved": True,
        "live_trading_implied": False,
    })


@app.post("/api/demo-queue/evaluate")
def demo_queue_evaluate(body: DemoQueueIdBody):
    from app.ae13b_product.demo_queue import evaluate_queue_item

    result = evaluate_queue_item(body.queue_id)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error") or "not found")
    return _json_ok(result)


@app.post("/api/demo-queue/remove")
def demo_queue_remove(body: DemoQueueIdBody):
    from app.ae13b_product.demo_queue import list_demo_queue, remove_from_demo_queue

    ok = remove_from_demo_queue(body.queue_id)
    if not ok:
        raise HTTPException(status_code=404, detail="queue item not found")
    return _json_ok({"ok": True, "removed": True, "items": list_demo_queue(), "paper_demo_only": True})


@app.post("/api/ae13b/live-market/evaluate")
def live_market_evaluate(body: LiveMarketEvalBody):
    """Safe paper/demo evaluation for a live market row — does not force a trade."""
    from app.ae13b_product.contract_resolver import resolve_identity
    from app.ae13b_product.demo_risk_guard import evaluate_demo_risk_guard

    pair = body.pair_address or body.pair or body.contract_address
    resolution = resolve_identity(
        chain=body.chain,
        contract_or_pair_address=pair,
        symbol=body.symbol,
        allow_external=False,
    )
    fam = None
    try:
        from app.ae13_semantic.runtime_registry import get_semantic_registry

        rec = get_semantic_registry().observe_candidate(
            {
                "symbol": body.symbol or resolution.get("matched_symbol"),
                "chain": body.chain or resolution.get("matched_chain"),
                "pair_address": pair or resolution.get("matched_pair_address"),
                "force_reclassify": False,
            }
        )
        fam = rec.get("semantic_signal_family")
        opp = rec.get("trading_opportunity_state")
        evidence = rec.get("evidence_summary")
    except Exception:
        opp = None
        evidence = None

    risk = evaluate_demo_risk_guard(
        requested_notional=75.0,
        pair_address=pair or resolution.get("matched_pair_address"),
        symbol=body.symbol or resolution.get("matched_symbol"),
        chain=body.chain,
        price=resolution.get("matched_price"),
        price_timestamp=resolution.get("matched_price_ts"),
        liquidity=resolution.get("matched_liquidity"),
        strategy_lane="manual_watchlist_scout",
    )
    blocked = not risk["risk_guard_passed"]
    selected = bool(resolution.get("matched_price")) and not blocked
    return {
        "ok": True,
        "selected": selected,
        "blocked": blocked,
        "reason": risk["risk_guard_reason"] if blocked else (resolution.get("reason") or "ok"),
        "data_status": resolution.get("resolution_status"),
        "semantic_label": fam,
        "opportunity_state": opp,
        "evidence_summary": evidence,
        "price_freshness": "ok" if resolution.get("matched_price") else "missing",
        "risk_status": "passed" if risk["risk_guard_passed"] else "blocked",
        "risk_guard": risk,
        "resolution": resolution,
        "next_possible_action": (
            "Add to Demo Queue or Watchlist (paper/demo only)"
            if selected
            else "Blocked or insufficient data — see reason"
        ),
        "paper_demo_only": True,
        "not_live_approved": True,
        "live_trading_implied": False,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(body: ChatRequest) -> ChatResponse:
    service = get_chat_service()
    reply, sid, tools = await service.chat(body.message, body.session_id)
    return ChatResponse(reply=reply, session_id=sid, tools_used=tools)

@app.get("/api/whale-log")
def whale_transaction_log(
    limit: int = Query(100, ge=1, le=500),
    cluster: str | None = Query(None, description="Filter by cluster_label"),
):
    """15-column whale_trades_log.csv for the live market table."""
    rows = read_whale_log_rows(limit=500)
    if cluster:
        cluster_upper = cluster.upper()
        rows = [r for r in rows if (r.get("cluster_label") or "").upper() == cluster_upper]
    rows = rows[-limit:]
    rows.reverse()
    return {
        "columns": WhaleActivity.CSV_FIELDS,
        "count": len(rows),
        "rows": rows,
    }


@app.get("/api/sentiment/matrix")
async def sentiment_matrix(limit: int = Query(15, ge=1, le=30)):
    """Cointelegraph RSS headlines with per-line sentiment scores."""
    return await fetch_rss_sentiment_matrix(limit=limit)


@app.get("/api/analytics/summary")
def analytics_summary():
    """Aggregated stats for dashboard header cards."""
    trader = get_paper_trader()
    # Authoritative legacy cluster counts (paper_trades + registry) — not whale-log truncation.
    try:
        from app.semantic.social_opportunistic_classifier import get_authoritative_semantic_counts

        auth = get_authoritative_semantic_counts(project_root=Path(__file__).resolve().parents[1])
        clusters = {
            "SOCIALLY_MOTIVATED": int(auth.get("legacy_socially_motivated_count") or 0),
            "OPPORTUNISTIC_SPECULATIVE": int(auth.get("legacy_opportunistic_speculative_count") or 0),
        }
        semantic_counts = auth
    except Exception as exc:  # noqa: BLE001 — dashboard must degrade safely
        log.warning("authoritative semantic counts failed; falling back to whale log: %s", exc)
        clusters = count_by_cluster()
        semantic_counts = None
    wallet = trader.get_wallet_summary()
    roi = trader.net_roi_summary()
    audit = trader._trade_audit_summary()
    return {
        "avg_whale_score": avg_whale_score(),
        "cluster_counts": clusters,
        "semantic_counts": semantic_counts,
        "paper_net_roi": roi,
        "portfolio_roi_pct": roi.get("portfolio_roi_pct"),
        "avg_closed_trade_roi_pct": roi.get("avg_net_roi_pct"),
        "invalid_trade_rows_excluded": audit.get("invalid_rows", 0),
        "paper_state_contaminated": audit.get("paper_state_contaminated", False),
        "wallet": wallet,
        "open_positions_count": wallet.get("open_positions_count", 0),
        "whale_log_path": str(LOG_PATH),
        "model": MODEL_NAME,
        "transparency": get_token_transparency_logs(),
    }


@app.get("/api/semantic/counts")
def semantic_counts():
    """Authoritative social/opportunistic semantic counters (audit-only, no trade authority)."""
    from app.semantic.social_opportunistic_classifier import get_authoritative_semantic_counts

    return get_authoritative_semantic_counts(project_root=Path(__file__).resolve().parents[1])


@app.get("/api/analytics/charts")
def analytics_charts():
    """Chart.js series: clusters, net ROI, sentiment vs price change."""
    trader = get_paper_trader()
    clusters = count_by_cluster()
    rows = read_whale_log_rows(limit=200)
    scatter = [
        {
            "x": float(r.get("price_change_24h") or 0),
            "y": float(r.get("buy_ratio") or 0.5),
            "label": r.get("symbol", ""),
            "cluster": r.get("cluster_label", ""),
        }
        for r in rows[-80:]
        if r.get("whale_score")
    ]
    return {
        "cluster_labels": list(clusters.keys()),
        "cluster_values": list(clusters.values()),
        "paper_charts": trader.chart_series(),
        "sentiment_price_scatter": scatter,
    }


@app.post("/api/paper/reset")
def paper_reset():
    return get_paper_trader().reset_demo_wallet()


@app.post("/api/demo/reset")
def demo_reset():
    """Reset demo wallet to clean $10,000 USD baseline for benchmarking."""
    wallet = get_paper_trader().reset_demo_wallet()
    return {
        "ok": True,
        "message": "Demo wallet reset to $10,000 USD",
        "wallet": wallet,
    }


@app.put("/api/paper/mode")
def paper_mode(body: dict[str, Any]):
    mode = str(body.get("mode", "DEMO")).upper()
    if mode not in ("DEMO", "LIVE"):
        raise HTTPException(status_code=400, detail="mode must be DEMO or LIVE")
    trader = get_paper_trader()
    applied = trader.set_trading_mode(mode)
    db.upsert_setting("trading_mode", applied)
    wallet = trader.get_wallet_summary()
    return {
        "trading_mode": applied,
        "message": f"System is now in {applied} mode",
        "wallet": wallet,
    }


@app.put("/api/agent/auto-execution")
def toggle_auto_execution(body: dict[str, Any]):
    enabled = bool(body.get("enabled", True))
    db.upsert_setting("auto_execution_enabled", enabled)
    return {"auto_execution_enabled": enabled}


# ── Trades ────────────────────────────────────────────────────────────────────

def _is_legacy_malformed_trade_row(row: dict[str, Any]) -> bool:
    """AE13I: flag pre-migration rows that are structurally incomplete.

    A row is "legacy_malformed" if it is an old RISK_GUARD_BLOCK rejection
    row written before the current CSV header (and rejection_code /
    blocking_guards fields) existed. Those rows have an empty rejection_code
    despite being a block event, which is the signature of the older, less
    structured rejection log format (fields effectively shifted/missing
    relative to today's schema).
    """
    if bool(row.get("legacy_malformed")):
        return True
    event_type = str(row.get("event_type") or "").upper()
    reason_code = str(row.get("reason_code") or row.get("reason") or "").upper()
    is_block_event = event_type == "RISK_GUARD_BLOCK" or reason_code == "RISK_GUARD_BLOCK"
    rejection_code = row.get("rejection_code") or row.get("code")
    return bool(is_block_event and not rejection_code)


def _alias_trade_row_for_ui(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize SQLite/CSV trade rows to UI field names (notional_usd, total_fees, reason_code)."""
    out = dict(row)
    if out.get("notional_usd") is None and out.get("value") is not None:
        out["notional_usd"] = out.get("value")
    if out.get("total_fees") is None and out.get("fee") is not None:
        out["total_fees"] = out.get("fee")
    if out.get("reason_code") is None and out.get("reason") is not None:
        out["reason_code"] = out.get("reason")
    if out.get("fill_price") is None and out.get("price") is not None:
        out["fill_price"] = out.get("price")
    if out.get("quantity") is None and out.get("amount") is not None:
        out["quantity"] = out.get("amount")
    out.setdefault("paper_demo_only", True)
    out.setdefault("not_live_approved", True)
    out["legacy_malformed"] = _is_legacy_malformed_trade_row(out)
    return out


@app.get("/api/trades")
def list_trades(
    limit: int = Query(100, ge=1, le=500),
    include_legacy_risk_blocks: bool = Query(
        False,
        description="Include legacy/malformed RISK_GUARD_BLOCK rows written before the current schema.",
    ),
):
    """Paper trades for UI. Prefers CSV write-path rows; falls back to SQLite with UI field aliases."""
    trader = get_paper_trader()
    csv_rows = trader.get_trades_from_log(limit=limit)
    rows = csv_rows if csv_rows else db.get_trades(limit=limit)
    aliased = [_alias_trade_row_for_ui(r) for r in rows]
    if include_legacy_risk_blocks:
        return _json_ok(aliased)
    return _json_ok([r for r in aliased if not r.get("legacy_malformed")])


# ── AE13 Virtual Ledger / Semantic Coverage ───────────────────────────────────

@app.get("/api/ae13/virtual-ledger")
def ae13_virtual_ledger(
    limit_orders: int = Query(100, ge=1, le=500),
    limit_positions: int = Query(100, ge=1, le=500),
    limit_trades: int = Query(100, ge=1, le=500),
):
    """Non-destructive Virtual Ledger View (merged paper/demo read model)."""
    from app.ae13_reconciliation.bridge import build_virtual_ledger_view

    view = build_virtual_ledger_view(Path(__file__).resolve().parents[1])
    return view.to_api_payload(
        limit_orders=limit_orders,
        limit_positions=limit_positions,
        limit_trades=limit_trades,
    )


@app.get("/api/ae13/demo-ledger")
def ae13_demo_ledger(
    limit_orders: int = Query(50, ge=1, le=500),
    limit_positions: int = Query(50, ge=1, le=500),
    limit_trades: int = Query(50, ge=1, le=500),
):
    """UI-oriented demo ledger payload from Virtual Ledger View + write SoT wallet."""
    from app.ae13_reconciliation.bridge import build_virtual_ledger_view

    trader = get_paper_trader()
    view = build_virtual_ledger_view(Path(__file__).resolve().parents[1])
    payload = view.to_api_payload(
        limit_orders=limit_orders,
        limit_positions=limit_positions,
        limit_trades=limit_trades,
    )
    wallet = trader.get_wallet_summary()
    return {
        **payload,
        "wallet_write_sot": wallet,
        "source_of_truth_label": (
            f"Write SoT: paper_state.json · Read model: Virtual Ledger View "
            f"(built {view.built_at_utc})"
        ),
        "no_wallet": True,
        "no_live": True,
        "paper_demo_only": True,
    }


@app.get("/api/ae13/semantic-coverage")
def ae13_semantic_coverage():
    """Static AE12 vs runtime semantic/sentiment coverage (no external LLM calls)."""
    from app.ae13_reconciliation.semantic_coverage import build_semantic_coverage

    return build_semantic_coverage(Path(__file__).resolve().parents[1])


@app.get("/api/ae13/safety")
def ae13_safety():
    from app.ae13_reconciliation.safety import build_no_wallet_safety_audit

    trader = get_paper_trader()
    settings = db.get_settings()
    return build_no_wallet_safety_audit(
        trading_mode=str(trader.get_wallet_summary().get("trading_mode") or "DEMO"),
        live_trading_enabled=bool(settings.get("live_trading_enabled", False)),
        wallet_configured=False,
        demo_acceptance_used=False,
    )


class DemoAcceptanceBody(BaseModel):
    enabled: bool = True
    notional_usd: float = 25.0
    close_after: bool = False


@app.post("/api/ae13/demo-acceptance")
def ae13_demo_acceptance(body: DemoAcceptanceBody):
    """Create a bounded DEMO_ACCEPTANCE_MODE paper order (fail-closed; off unless enabled)."""
    from app.ae13_reconciliation.demo_acceptance import (
        create_demo_acceptance_order,
        maybe_close_demo_acceptance_position,
    )

    _require_demo_mode()
    settings = db.get_settings()
    mode_enabled = bool(body.enabled) and bool(settings.get("demo_acceptance_mode", False))
    # Allow explicit body.enabled only when settings flag OR AE13 env is set
    import os

    env_on = os.getenv("AE13_DEMO_ACCEPTANCE_MODE", "").strip() in ("1", "true", "TRUE", "yes")
    mode_enabled = bool(body.enabled) and (
        bool(settings.get("demo_acceptance_mode", False)) or env_on
    )

    result = create_demo_acceptance_order(
        trading_mode=str(get_paper_trader().get_wallet_summary().get("trading_mode") or "DEMO"),
        live_trading_enabled=bool(settings.get("live_trading_enabled", False)),
        wallet_configured=False,
        demo_acceptance_mode_enabled=mode_enabled,
        settings=settings,
        notional_usd=float(body.notional_usd),
        execute=True,
    )
    if body.close_after and result.get("status") == "CREATED":
        pos = result.get("position") or {}
        result["close_result"] = maybe_close_demo_acceptance_position(
            trading_mode="DEMO",
            live_trading_enabled=False,
            wallet_configured=False,
            demo_acceptance_mode_enabled=mode_enabled,
            position_id=int(pos["id"]) if pos.get("id") is not None else None,
        )
    return result


@app.get("/api/paper/wallet")
def paper_wallet():
    """Demo wallet from write SoT, annotated with Virtual Ledger View freshness."""
    trader = get_paper_trader()
    wallet = trader.get_wallet_summary()
    try:
        from app.ae13_reconciliation.bridge import build_virtual_ledger_view

        view = build_virtual_ledger_view(Path(__file__).resolve().parents[1])
        wallet = {
            **wallet,
            "read_model": "virtual_ledger_view",
            "ui_write_source_of_truth": "legacy_paper_state",
            "merged_open_positions_count": len(view.open_positions),
            "merged_orders_count": len(view.orders),
            "merged_closed_trades_count": len(view.closed_trades),
            "ledger_warnings": view.warnings[:5],
            "paper_demo_only": True,
            "not_live_approved": True,
            "wallet_configured": False,
        }
    except Exception:
        wallet = {**wallet, "paper_demo_only": True, "wallet_configured": False}
    return wallet


# ── Whale alerts ──────────────────────────────────────────────────────────────

@app.get("/api/whale-alerts")
def list_whale_alerts(limit: int = Query(50, ge=1, le=200)):
    return db.get_whale_alerts(limit=limit)


# ── Dashboard summary ─────────────────────────────────────────────────────────

@app.get("/api/dashboard/summary")
def dashboard_summary():
    return db.get_dashboard_summary()


# ── Market scan ───────────────────────────────────────────────────────────────

@app.get("/api/market/scan")
def market_scan(
    limit: int = Query(50, ge=1, le=200),
    sort_by: str = Query("whale_score"),
    chain: str | None = Query(None),
):
    return db.get_coins(limit=limit, sort_by=sort_by, chain=chain)


# ── Settings ──────────────────────────────────────────────────────────────────

@app.get("/api/settings")
def get_settings():
    return db.get_settings()


@app.get("/api/settings/effective")
def get_effective_settings_endpoint():
    """Canonical runtime settings, hidden thresholds, aliases, and settings_hash."""
    from .observability.effective_settings import get_effective_settings

    eff = get_effective_settings()
    report_path = eff.write_audit_report()
    payload = eff.to_api_response()
    payload["audit_report_path"] = str(report_path)
    return payload


@app.put("/api/settings")
def update_settings(body: dict[str, Any]):
    for key, value in body.items():
        db.upsert_setting(key, value)
    return db.get_settings()


@app.patch("/api/settings")
def patch_settings_endpoint(body: dict[str, Any]):
    """PATCH dirty canonical settings; returns GET /api/settings/effective-compatible payload."""
    from fastapi import HTTPException

    from .observability.settings_patch import SettingsPatchError, patch_settings

    try:
        return patch_settings(body)
    except SettingsPatchError as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": str(exc), "field_errors": exc.field_errors},
        ) from exc


# ── AE12.5 Runtime Observability / Final Reporting (read-only) ────────────────

def _get_ae12_manager():
    """App-level AE12ReportManager registry — do not construct heavy loaders per request."""
    from .ae12_reporting import get_ae12_report_manager

    return get_ae12_report_manager()


def _ae12_maybe_refresh(manager, refresh: bool) -> None:
    if refresh:
        manager.refresh()


@app.get("/api/ae12/status")
def ae12_status(
    refresh: bool = Query(False, description="Reload memory cache only; does not mutate AE12 files"),
    manager=Depends(_get_ae12_manager),
):
    _ae12_maybe_refresh(manager, refresh)
    return manager.get_status()


@app.get("/api/ae12/forward-evidence-summary")
def ae12_forward_evidence_summary(
    refresh: bool = Query(False),
    manager=Depends(_get_ae12_manager),
):
    _ae12_maybe_refresh(manager, refresh)
    return manager.get_forward_evidence_summary()


@app.get("/api/ae12/missed-winners")
def ae12_missed_winners(
    limit: int = Query(100, ge=1, le=500),
    refresh: bool = Query(False),
    manager=Depends(_get_ae12_manager),
):
    _ae12_maybe_refresh(manager, refresh)
    return manager.get_missed_winners(limit=limit)


@app.get("/api/ae12/trade-vs-no-trade")
def ae12_trade_vs_no_trade(
    refresh: bool = Query(False),
    manager=Depends(_get_ae12_manager),
):
    _ae12_maybe_refresh(manager, refresh)
    return manager.get_trade_vs_no_trade()


@app.get("/api/ae12/strict-vs-exploration")
def ae12_strict_vs_exploration(
    refresh: bool = Query(False),
    manager=Depends(_get_ae12_manager),
):
    _ae12_maybe_refresh(manager, refresh)
    return manager.get_strict_vs_exploration()


@app.get("/api/ae12/qwen-linkage")
def ae12_qwen_linkage(
    refresh: bool = Query(False),
    manager=Depends(_get_ae12_manager),
):
    _ae12_maybe_refresh(manager, refresh)
    return manager.get_qwen_linkage_summary()


@app.get("/api/ae12/safety")
def ae12_safety(
    refresh: bool = Query(False),
    manager=Depends(_get_ae12_manager),
):
    _ae12_maybe_refresh(manager, refresh)
    return manager.get_safety_summary()


@app.get("/api/ae12/final-report-summary")
def ae12_final_report_summary(
    refresh: bool = Query(False),
    manager=Depends(_get_ae12_manager),
):
    _ae12_maybe_refresh(manager, refresh)
    return manager.get_final_report_summary()


@app.get("/api/ae12/runtime-collection")
def ae12_runtime_collection(
    refresh: bool = Query(False),
    manager=Depends(_get_ae12_manager),
):
    """Runtime/data-collection panel source (census + maturation summary)."""
    _ae12_maybe_refresh(manager, refresh)
    return manager.get_runtime_collection_status()


@app.get("/api/ae12/signal-taxonomy")
def ae12_signal_taxonomy(
    refresh: bool = Query(False),
    manager=Depends(_get_ae12_manager),
):
    """Signal taxonomy QA panel - dual-axis semantic vs trading state (read-only)."""
    _ae12_maybe_refresh(manager, refresh)
    return manager.get_signal_taxonomy()


@app.get("/api/ae12/sentimentfix")
def ae12_sentimentfix(
    refresh: bool = Query(False),
    manager=Depends(_get_ae12_manager),
):
    """AE12-SentimentFix dual-axis repair status (read-only)."""
    _ae12_maybe_refresh(manager, refresh)
    return manager.get_sentimentfix()


@app.get("/api/ae12/semantic-coin-classifier")
def ae12_semantic_coin_classifier(
    refresh: bool = Query(False),
    manager=Depends(_get_ae12_manager),
):
    """AE12-SentimentFix semantic coin classifier status (read-only)."""
    _ae12_maybe_refresh(manager, refresh)
    return manager.get_semantic_coin_classifier()


@app.get("/api/ae12/gemini-semantic-adjudication")
def ae12_gemini_semantic_adjudication(
    refresh: bool = Query(False),
    manager=Depends(_get_ae12_manager),
):
    """AE12-SentimentFix Gemini semantic adjudication (read-only, not trade authority)."""
    _ae12_maybe_refresh(manager, refresh)
    return manager.get_gemini_semantic_adjudication()


@app.get("/api/ae12/manual-review-drilldown")
def ae12_manual_review_drilldown(
    refresh: bool = Query(False),
    manager=Depends(_get_ae12_manager),
):
    """AE12-SentimentFix local manual-review drilldown (no Gemini / no external APIs)."""
    _ae12_maybe_refresh(manager, refresh)
    return manager.get_manual_review_drilldown()


@app.post("/api/ae12/refresh-cache")
def ae12_refresh_cache(manager=Depends(_get_ae12_manager)):
    """Operational memory-cache reload only - does not mutate AE12 source artifacts."""
    return manager.refresh()


# ── AE12.7 Intelligent Agent Layer (read-only; no trade authority) ────────────

def _get_ae127_store():
    from .intelligent_agents.api_store import get_ae127_agent_store

    return get_ae127_agent_store()


@app.get("/api/ae12/agents/status")
def ae12_agents_status(
    refresh: bool = Query(False),
    store=Depends(_get_ae127_store),
):
    if refresh:
        store.refresh()
    return store.get_status()


@app.get("/api/ae12/agents/recent")
def ae12_agents_recent(
    limit: int = Query(50, ge=1, le=500),
    refresh: bool = Query(False),
    store=Depends(_get_ae127_store),
):
    if refresh:
        store.refresh()
    return store.get_recent(limit=limit)


@app.get("/api/ae12/agents/by-candidate/{candidate_id}")
def ae12_agents_by_candidate(
    candidate_id: str,
    refresh: bool = Query(False),
    store=Depends(_get_ae127_store),
):
    if refresh:
        store.refresh()
    return store.get_by_candidate(candidate_id)


@app.get("/api/ae12/agents/by-pair/{pair_address}")
def ae12_agents_by_pair(
    pair_address: str,
    refresh: bool = Query(False),
    store=Depends(_get_ae127_store),
):
    if refresh:
        store.refresh()
    return store.get_by_pair(pair_address)


@app.get("/api/ae12/agents/authority-audit")
def ae12_agents_authority_audit(
    refresh: bool = Query(False),
    store=Depends(_get_ae127_store),
):
    if refresh:
        store.refresh()
    return store.get_authority_audit()


@app.get("/api/ae12/agents/ui-summary")
def ae12_agents_ui_summary(
    refresh: bool = Query(False),
    store=Depends(_get_ae127_store),
):
    if refresh:
        store.refresh()
    return store.get_ui_summary()


# ── AE13B Product Demo APIs ───────────────────────────────────────────────────

@app.get("/api/ae13b/demo-bot/status")
def ae13b_demo_bot_status():
    from app.ae13b_product.demo_bot import get_demo_bot

    try:
        data = get_demo_bot().status()
        try:
            from app.ae13b_product.ae14_readiness import compute_ae14_readiness
            from app.ae13b_product.live_market import build_live_market

            market_rows = build_live_market(limit=200).get("rows")
            data["ae14_readiness"] = compute_ae14_readiness(market_rows=market_rows)
        except Exception:
            pass
        return _json_ok({
            "ok": True,
            "status": "ready",
            "user_message": "",
            **data,
        })
    except Exception as exc:  # noqa: BLE001 - fail-soft for UI shell
        return _json_ok({
            "ok": False,
            "status": "unavailable",
            "user_message": "Demo bot status is unavailable. Demo controls are still available.",
            "details": str(exc)[:300],
            "bot_status": "Unavailable",
            "demo_mode_active": True,
            "live_trading_disabled": True,
            "wallet_not_connected": True,
            "strategy_lanes": [],
            "activity": [],
            "open_positions": [],
            "open_positions_count": 0,
            "wallet": {},
            "what_bot_is_doing": {
                "what_is_happening": "Status unavailable",
                "why_traded_or_not": str(exc)[:200],
                "next_action": "Retry Refresh or use demo controls.",
            },
        })


@app.post("/api/ae13b/demo-bot/start")
def ae13b_demo_bot_start():
    from app.ae13b_product.demo_bot import get_demo_bot

    _enforce_paper_demo_execution_guard(acceptance=False)
    return get_demo_bot().start()


@app.post("/api/ae13b/demo-bot/stop")
def ae13b_demo_bot_stop():
    from app.ae13b_product.demo_bot import get_demo_bot

    return get_demo_bot().stop()


@app.post("/api/ae13b/demo-bot/pause")
def ae13b_demo_bot_pause():
    from app.ae13b_product.demo_bot import get_demo_bot

    return get_demo_bot().pause()


@app.post("/api/ae13b/demo-bot/run-once")
def ae13b_demo_bot_run_once():
    from app.ae13b_product.demo_bot import get_demo_bot

    _enforce_paper_demo_execution_guard(acceptance=False)
    return get_demo_bot().run_once()


@app.post("/api/ae13b/demo-bot/cycle")
def ae13b_demo_bot_cycle(force: bool = Query(True)):
    """Legacy alias: force=True runs one cycle without starting continuous loop."""
    from app.ae13b_product.demo_bot import get_demo_bot

    _enforce_paper_demo_execution_guard(acceptance=False)
    if force:
        return get_demo_bot().run_once()
    return get_demo_bot().run_cycle(force=False, from_loop=False)


@app.get("/api/ae13b/demo-bot/events")
def ae13b_demo_bot_events(limit: int = Query(50, ge=1, le=200)):
    from app.ae13b_product.demo_bot import get_demo_bot

    return get_demo_bot().events(limit=limit)


@app.post("/api/ae13b/demo-bot/close-all")
def ae13b_demo_bot_close_all():
    from app.ae13b_product.demo_bot import get_demo_bot

    _enforce_paper_demo_execution_guard(acceptance=False)
    return get_demo_bot().close_all()


@app.post("/api/ae13b/demo-bot/reset-wallet")
def ae13b_demo_bot_reset_wallet():
    from app.ae13b_product.demo_bot import get_demo_bot

    _enforce_paper_demo_execution_guard(acceptance=False)
    return get_demo_bot().reset_wallet()


@app.post("/api/ae13b/demo-bot/preset")
def ae13b_demo_bot_preset(body: dict[str, Any]):
    from app.ae13b_product.demo_bot import get_demo_bot

    return get_demo_bot().apply_preset(str(body.get("preset_id") or "balanced"))


@app.post("/api/ae13b/demo-bot/max-trades-per-hour")
def ae13b_demo_bot_max_trades(body: dict[str, Any]):
    from app.ae13b_product.demo_bot import get_demo_bot

    return get_demo_bot().set_max_trades_per_hour(int(body.get("max_trades_per_hour") or 12))


@app.get("/api/ae13b/presets")
def ae13b_presets():
    from app.ae13b_product.presets import list_presets, list_strategy_lanes

    return {"presets": list_presets(), "strategy_lanes": list_strategy_lanes()}


@app.get("/api/ae13b/portfolio")
def ae13b_portfolio():
    """User-facing portfolio: current write-SoT positions + archived VLV display-only.

    Hot path: runtime index only — no DexScreener/Helius/RSS.
    """
    from app.runtime.ui_get_network_guard import ui_get_network_guard

    with ui_get_network_guard("/api/ae13b/portfolio"):
        try:
            from app.ae13_reconciliation.bridge import build_virtual_ledger_view
            from app.ae13b_product.demo_bot import get_demo_bot
            from app.ae13b_product.runtime_market_feed import (
                apply_index_mark_prices_to_trader,
                repair_legacy_position_identity,
            )

            trader = get_paper_trader()
            index_meta = apply_index_mark_prices_to_trader(trader)
            wallet = trader.get_wallet_summary()
            index_rows = index_meta.get("rows") or []
            raw_open = (
                trader.get_positions(status="OPEN")
                if hasattr(trader, "get_positions")
                else []
            )
            repaired = [repair_legacy_position_identity(p, index_rows) for p in raw_open]
            # Re-apply mark prices after identity repair so URL keys resolve
            if index_meta.get("applied") and repaired:
                from app.clean_forward.runtime_identity_index import index_rows_to_market_price_entries

                trader.set_market_prices(
                    index_rows_to_market_price_entries(index_rows),
                    price_timestamp=index_meta.get("loaded_at"),
                )
            current_open = (
                trader.mark_positions_to_market(repaired)
                if hasattr(trader, "mark_positions_to_market")
                else repaired
            )
            # Preserve repair status messages on marked rows
            for i, p in enumerate(current_open):
                if i < len(repaired):
                    if repaired[i].get("mark_price_lookup_status") == "LEGACY_POSITION_IDENTITY_REPAIR_NEEDED":
                        p["mark_price_lookup_status"] = "LEGACY_POSITION_IDENTITY_REPAIR_NEEDED"
                        p["mark_price_unavailable_reason"] = "LEGACY_POSITION_IDENTITY_REPAIR_NEEDED"
                        p["price_resolution_failure_reason"] = repaired[i].get(
                            "price_resolution_failure_reason"
                        )
                    for k in (
                        "canonical_market_identity",
                        "provider_pair_url_exact",
                        "provider_pair_url_final_segment_exact",
                        "symbol_pair_display",
                        "open_chart_url",
                        "pair_address_derived",
                    ):
                        if repaired[i].get(k) and not p.get(k):
                            p[k] = repaired[i][k]

            current_trades = trader.get_trades_from_log(limit=50)
            view = build_virtual_ledger_view(Path(__file__).resolve().parents[1])
            archive_open = [
                {**p, "display_only": True, "tradable": False, "archive": True}
                for p in view.open_positions
                if str(p.get("source_layer")) != "legacy_paper_state"
            ][:30]
            bot = get_demo_bot().status()
            return _json_ok({
                "ok": True,
                "status": "ready",
                "user_message": "",
                "wallet": wallet,
                "current_open_positions": current_open,
                "current_trades": [_alias_trade_row_for_ui(t) for t in current_trades],
                "archive_open_positions_display_only": archive_open,
                "source_status": {
                    "current_demo_wallet_source": "paper_state.json (PaperTrader write SoT)",
                    "historical_archive_source": "Virtual Ledger View (read-only)",
                    "archived_positions": "display-only - not tradable",
                    "current_tradable_positions": "current demo state only",
                    "vlv_read_only": True,
                },
                "bot": {
                    "bot_status": bot.get("bot_status"),
                    "what_bot_is_doing": bot.get("what_bot_is_doing"),
                },
                "runtime_identity_index": {
                    "applied": index_meta.get("applied", False),
                    "measured_load_time_ms": index_meta.get("measured_load_time_ms"),
                    "external_network_calls_on_load": False,
                    "error_code": index_meta.get("error_code"),
                    "rebuild_instruction": index_meta.get("rebuild_instruction"),
                },
                "paper_demo_only": True,
                "not_live_approved": True,
                "wallet_configured": False,
                "live_trading_ready": False,
            })
        except Exception as exc:  # noqa: BLE001
            return _json_ok({
                "ok": False,
                "status": "unavailable",
                "user_message": "Portfolio data is unavailable. Demo controls are still available.",
                "details": str(exc)[:300],
                "wallet": {},
                "current_open_positions": [],
                "current_trades": [],
                "archive_open_positions_display_only": [],
                "paper_demo_only": True,
                "wallet_configured": False,
                "live_trading_ready": False,
            })


@app.get("/api/ae13b/opportunities")
def ae13b_opportunities(limit: int = Query(40, ge=1, le=100)):
    from app.runtime.ui_get_network_guard import ui_get_network_guard

    with ui_get_network_guard("/api/ae13b/opportunities"):
        return _build_opportunities(limit)


def _build_opportunities(limit: int):
    try:
        from app.ae13_semantic.runtime_registry import get_semantic_registry
        from app.ae13b_product.runtime_market_feed import (
            build_opportunities_from_index,
            enrich_opportunity_rows,
        )

        registry = get_semantic_registry()
        coins = db.get_coins(limit=limit, sort_by="whale_score")
        # coins table has no cluster_label — join sticky cluster_registry when present
        try:
            from app.analytics.features import get_persisted_cluster, list_cluster_registry

            _reg = list_cluster_registry()
        except Exception:
            _reg = {}
            get_persisted_cluster = None  # type: ignore
        open_pairs = {
            str(p.get("pair_address") or "")
            for p in get_paper_trader().get_positions(status="OPEN")
        }
        rows = []
        for c in coins:
            token_addr = str(c.get("token_address") or "")
            pair_addr = str(c.get("pair_address") or "")
            legacy_cluster = c.get("cluster_label")
            if not legacy_cluster and get_persisted_cluster is not None:
                for key in (token_addr, pair_addr):
                    if not key:
                        continue
                    try:
                        persisted = get_persisted_cluster(key)
                        if persisted is not None:
                            legacy_cluster = persisted.value
                            break
                    except Exception:
                        pass
                    entry = _reg.get(key) or _reg.get(key.lower()) if isinstance(_reg, dict) else None
                    if isinstance(entry, dict) and entry.get("cluster_label"):
                        legacy_cluster = entry.get("cluster_label")
                        break
            rec = registry.observe_candidate(
                {
                    "id": c.get("id"),
                    "coin_id": c.get("id"),
                    "symbol": c.get("symbol"),
                    "name": c.get("name"),
                    "chain": c.get("chain"),
                    "pair_address": c.get("pair_address"),
                    "price_usd": c.get("latest_price"),
                    "liquidity_usd": c.get("latest_liquidity"),
                    "volume_24h": c.get("latest_volume_24h"),
                    "whale_score": c.get("latest_whale_score"),
                    "cluster_label": legacy_cluster,
                }
            )
            pair = str(c.get("pair_address") or "")
            in_pos = pair in open_pairs
            family = str(rec.get("semantic_signal_family") or "UNKNOWN_UNRESOLVED")
            if in_pos:
                action = "Already in position"
                reason = "You already hold a demo position on this pair."
            elif family == "UNKNOWN_UNRESOLVED":
                action = "Watch"
                reason = rec.get("unresolved_reason") or "Unknown — not enough evidence yet."
            elif not c.get("latest_price"):
                action = "Blocked"
                reason = "Price data too old or missing for a confident trade."
            else:
                action = "Demo Buy candidate"
                reason = (
                    f"{rec.get('semantic_label_human')} — eligible for bounded paper/demo exploration."
                )
            rows.append(
                {
                    "symbol": c.get("symbol"),
                    "chain": c.get("chain"),
                    "pair_address": pair,
                    "price_usd": c.get("latest_price"),
                    "liquidity_usd": c.get("latest_liquidity"),
                    "volume_24h": c.get("latest_volume_24h"),
                    "semantic_label": rec.get("semantic_label_human"),
                    "semantic_family": family,
                    "opportunity_state": rec.get("trading_opportunity_state"),
                    "action": action,
                    "reason": reason,
                    "risk_flag": "needs_review" if rec.get("needs_review") else "ok",
                    "freshness": rec.get("last_seen_at"),
                    "seen_count": rec.get("seen_count"),
                    "classification_source": rec.get("classification_source"),
                    "paper_demo_only": True,
                    "demo_action_endpoint": "/api/ae13b/demo/buy-candidate",
                }
            )
        # AE18 cross-surface parity: Market Opportunities is sourced from the
        # runtime canonical identity index (same source as Clean Forward and
        # Market Snapshot). Legacy DB candidates are supplemental metadata only
        # and are joined strictly by normalized_provider_pair_url_key.
        built = build_opportunities_from_index(limit=limit, supplemental_rows=rows)
        if not built.get("ok"):
            return {
                "ok": False,
                "status": built.get("status", "index_missing"),
                "user_message": built.get("user_message", "Runtime index unavailable."),
                "error_code": built.get("error_code"),
                "count": 0,
                "opportunities": [],
            }
        rows = built["rows"]
        enriched = enrich_opportunity_rows(rows)
        rows = enriched["rows"]
        for r in rows:
            r.setdefault("paper_demo_only", True)
            r.setdefault("demo_action_endpoint", "/api/ae13b/demo/buy-candidate")
            trade_status = str(r.get("trade_readiness_status") or "")
            block_reason = str(r.get("trade_block_reason") or "")
            activity_status = str(r.get("market_activity_status") or "ACTIVITY_UNKNOWN")
            if not r.get("canonical_market_identity"):
                r["action"] = "Blocked"
                r["reason"] = (
                    "DEMO_ACTION_BLOCKED_IDENTITY_UNRESOLVED — no canonical market URL in runtime index."
                )
            elif activity_status != "ACTIVE_PROVIDER_TXNS":
                r["action"] = "Watch"
                r["reason"] = (
                    r.get("activity_trade_block_reason")
                    or f"{activity_status} — market is not classified as active provider flow."
                )
            elif trade_status == "PAPER_ELIGIBLE":
                r["action"] = "Demo Buy candidate"
                r["reason"] = "Eligible for bounded paper/demo exploration."
            else:
                # Blocking is driven by market data / identity / continuity only,
                # never by a missing symbol_pair_display.
                r["action"] = "Blocked" if trade_status.startswith("ENTRY_BLOCKED") else "Watch"
                r["reason"] = f"{trade_status} — {block_reason}" if block_reason else trade_status
        return {
            "ok": True,
            "status": "ready" if rows else "empty",
            "user_message": "",
            "count": len(rows),
            "opportunities": rows,
            "canonical_identity_type": "PROVIDER_URL",
            "pair_address_is_canonical": False,
            "base_only_display_count": enriched["base_only_display_count"],
            "opportunities_joined_runtime_index_count": enriched[
                "opportunities_joined_runtime_index_count"
            ],
            "opportunities_missing_runtime_index_join_count": enriched[
                "opportunities_missing_runtime_index_join_count"
            ],
            "opportunities_join_failure_reasons": enriched[
                "opportunities_join_failure_reasons"
            ],
            "opportunities_rows_where_valid_symbol_was_overwritten": enriched[
                "opportunities_rows_where_valid_symbol_was_overwritten"
            ],
            "joined_by": "normalized_provider_pair_url_key",
            "supplemental_candidates_joined": built.get("supplemental_joined_count", 0),
            "runtime_index_sourced": True,
            "external_network_on_load": False,
            "demo_action_endpoint": "/api/ae13b/demo/buy-candidate",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": "unavailable",
            "user_message": "Market opportunities are unavailable.",
            "details": str(exc)[:300],
            "count": 0,
            "opportunities": [],
        }


@app.get("/api/ae13b/semantic-registry")
def ae13b_semantic_registry():
    try:
        from app.ae13_reconciliation.semantic_coverage import build_semantic_coverage
        from app.ae13_semantic.runtime_registry import get_semantic_registry

        snap = get_semantic_registry().snapshot()
        static = build_semantic_coverage(Path(__file__).resolve().parents[1])
        return {
            "ok": True,
            "status": "ready",
            "user_message": "",
            **snap,
            "static_ae12_context": {
                "unique_coins_static": (static.get("ui_counters") or {}).get("unique_coins_static"),
                "semantic_source_label_static": static.get("semantic_source_label"),
                "note": snap.get("static_ae12_note"),
                "is_live_universe": False,
            },
            "combined_semantic_source_label": (
                "Semantic Source: Runtime Registry + Static AE12 Snapshot"
                if (static.get("ui_counters") or {}).get("unique_coins_static")
                else snap.get("semantic_source_label")
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": "unavailable",
            "user_message": "Semantic registry is unavailable.",
            "details": str(exc)[:300],
            "counters": {},
            "records": [],
        }


@app.get("/api/ae13b/navigation")
def ae13b_navigation():
    return {
        "default_tab": "demo",
        "primary_tabs": [
            {"id": "demo", "label": "Demo Trading"},
            {"id": "clean-forward", "label": "Clean Forward Market Feed"},
            {"id": "live-market", "label": "Market Snapshot Feed"},
            {"id": "portfolio", "label": "Portfolio"},
            {"id": "market", "label": "Market Opportunities"},
            {"id": "insights", "label": "AI Insights"},
            {"id": "settings", "label": "Settings"},
            {"id": "vault", "label": "Research Evidence / Audit Vault"},
        ],
        "vault_sections": [
            "Safety",
            "Forward Evidence",
            "Model Evidence",
            "Agent Evidence",
            "Artifact Roots",
            "Raw Audit Details",
            "Former panel mapping",
        ],
        "view_switcher": "ViewSwitcher",
        "data_loader": "DataLoader",
        "view_switcher_decoupled_from_data_loader": True,
        "ae12_primary_tab": False,
        "internal_phase_labels_in_main_ui": False,
    }


@app.get("/api/ae13b/clean-forward-market-feed")
def ae13b_clean_forward_market_feed(
    limit: int = Query(25, ge=1, le=100),
    max_rows_per_base_token: int = Query(1, ge=1, le=10),
    max_rows_per_symbol: int = Query(1, ge=1, le=10),
    max_verify: int = Query(40, ge=5, le=80),
    use_cache: bool = Query(True, description="Use pair-verify TTL cache (Ctrl+F5 bootstrap path)"),
):
    """AE13K Clean Forward Market Feed — hot path reads runtime identity index only.

    Not Live Market. Does not mutate historical/training data. Paper/demo/research only.
    Use POST refresh or rebuild script for cold-path provider refetch.
    """
    from app.runtime.ui_get_network_guard import ui_get_network_guard

    with ui_get_network_guard("/api/ae13b/clean-forward-market-feed"):
        try:
            from app.ae13b_product.runtime_market_feed import build_clean_forward_from_index

            data = build_clean_forward_from_index(limit=limit)
            if isinstance(data, dict):
                data.setdefault("ok", True)
                data.setdefault("user_message", "")
                data["endpoint_path"] = "/api/ae13b/clean-forward-market-feed"
                data["bootstrap_path"] = "ctrl_f5_full_reload_or_tab_open"
                data["ae14_candidate_source_policy"] = "clean_forward_market_feed_only"
                data["candidate_source"] = "clean_forward_market_feed"
                data["legacy_market_snapshots_used"] = False
                data["live_execution_enabled"] = False
            return _json_ok(data)
        except Exception as exc:  # noqa: BLE001
            return _json_ok(
                {
                    "ok": False,
                    "status": "unavailable",
                    "panel_title": "Clean Forward Market Feed",
                    "not_live_market": True,
                    "user_message": (
                        "Clean Forward Market Feed is unavailable. "
                        "Market Snapshot Feed and demo controls are still available."
                    ),
                    "details": str(exc)[:300],
                    "rows": [],
                    "alternative_pools": [],
                    "stats": {},
                    "demo_mode_badge": "LIVE DISABLED / DEMO ONLY",
                    "paper_demo_only": True,
                    "wallet_configured": False,
                    "live_trading_ready": False,
                }
            )


class CleanForwardRefreshBody(BaseModel):
    force: bool = False
    clear_cache: bool = False
    limit: int = 25
    max_rows_per_base_token: int = 1
    max_rows_per_symbol: int = 1
    max_verify: int = 40
    previous_rows: list[dict[str, Any]] | None = None


def _clean_forward_refresh_handler(body: CleanForwardRefreshBody) -> dict[str, Any]:
    """Explicit POST-only manual refresh — URL-first, updates runtime index atomically."""
    from app.ae13b_product.manual_refresh_runtime_index import manual_refresh_runtime_index
    from app.ae13b_product.runtime_market_feed import build_clean_forward_from_index
    from app.runtime.shutdown import is_shutting_down

    from app.ae13b_product.provider_refresh_errors import build_refresh_failure

    if is_shutting_down():
        failure = build_refresh_failure(
            error_code="CONTROLLED_SHUTDOWN_SKIP", shutdown_event_set=True
        )
        return {
            "ok": False,
            "status": "shutdown",
            "user_message": failure["user_message"],
            "rows": [],
            "refresh_failure": failure,
            "refresh": {
                "refresh_mode": "cancelled_shutdown",
                "provider_refetch_attempted": False,
                "provider_refresh_cancelled": True,
                "provider_refresh_cancel_reason": "CONTROLLED_SHUTDOWN_SKIP",
                **failure,
            },
        }

    refresh_meta = manual_refresh_runtime_index(
        force=body.force,
        clear_cache=body.clear_cache,
        max_rows=body.limit * 4 if body.limit else None,
        allow_dexscreener=True,
        refresh_requested_by="ui_manual_refresh",
    )
    data = build_clean_forward_from_index(limit=body.limit or 25)
    if isinstance(data, dict):
        data.setdefault("ok", True)
        data["endpoint_path"] = "/api/clean-forward-feed/refresh"
        data["old_data_touched"] = False
        data["training_run"] = False
        data["backtest_run"] = False
        data["ae14_run"] = False
        data["paper_positions_opened_from_clean_feed"] = 0
        data["live_trading_enabled"] = False
        data["live_execution_enabled"] = False
        data["ae14_candidate_source_policy"] = "clean_forward_market_feed_only"
        data["candidate_source"] = "clean_forward_market_feed"
        data["legacy_market_snapshots_used"] = False
        failures = refresh_meta.get("refresh_failures") or []
        first_failure = failures[0] if failures else None
        if refresh_meta.get("runtime_index_updated"):
            ui_message = "Provider refresh completed; runtime index updated."
            if failures:
                ui_message += f" {len(failures)} row(s) could not be refreshed."
        elif refresh_meta.get("provider_refresh_cancelled"):
            ui_message = "Provider refresh cancelled due to shutdown."
        elif first_failure:
            ui_message = (
                f"{first_failure['user_message']} [{first_failure['refresh_error_code']}] "
                f"{first_failure['recovery_instruction']}"
            )
        else:
            ui_message = "Provider refresh finished."
        data["refresh"] = {
            **refresh_meta,
            "provider_refetch_attempted": True,
            "ui_message": ui_message,
        }
        if first_failure:
            data["refresh_failure"] = first_failure
            data["refresh_error_code"] = first_failure["refresh_error_code"]
            data["refresh_error_reason"] = first_failure["refresh_error_reason"]
            data["recovery_instruction"] = first_failure["recovery_instruction"]
            data["retryable"] = first_failure["retryable"]
            if not refresh_meta.get("runtime_index_updated"):
                data["user_message"] = ui_message
        data["refresh_metadata"] = data["refresh"]
    return data


@app.post("/api/clean-forward-feed/refresh")
def clean_forward_feed_refresh(body: CleanForwardRefreshBody):
    """AE13K UI refresh — explicit provider refetch with refresh metadata.

    force=True bypasses pair-verify TTL cache (still rate-limited).
    clear_cache=True clears in-process verify cache (Force Provider Refresh).
    """
    try:
        return _json_ok(_clean_forward_refresh_handler(body))
    except Exception as exc:  # noqa: BLE001
        return _json_ok(_structured_refresh_failure_response(exc, "Clean Forward Market Feed"))


def _structured_refresh_failure_response(exc: Exception, panel_title: str) -> dict[str, Any]:
    from app.ae13b_product.provider_refresh_errors import (
        build_refresh_failure,
        classify_refresh_exception,
    )
    from app.runtime.shutdown import is_shutting_down

    failure = build_refresh_failure(
        error_code=classify_refresh_exception(exc),
        exception=exc,
        reason=str(exc)[:300],
        shutdown_event_set=is_shutting_down(),
    )
    return {
        "ok": False,
        "status": "unavailable",
        "panel_title": panel_title,
        "user_message": (
            f"{failure['user_message']} [{failure['refresh_error_code']}] "
            f"{failure['recovery_instruction']}"
        ),
        "details": failure["refresh_error_reason"],
        "rows": [],
        "refresh_failure": failure,
        "refresh_error_code": failure["refresh_error_code"],
        "refresh_error_reason": failure["refresh_error_reason"],
        "recovery_instruction": failure["recovery_instruction"],
        "retryable": failure["retryable"],
        "refresh": {
            "refresh_mode": "verification_deferred",
            "provider_refetch_attempted": False,
            **failure,
        },
    }


@app.post("/api/ae13b/clean-forward-market-feed/refresh")
def ae13b_clean_forward_feed_refresh(body: CleanForwardRefreshBody):
    """Alias for Clean Forward refresh endpoint."""
    try:
        data = _clean_forward_refresh_handler(body)
        if isinstance(data, dict):
            data["endpoint_path"] = "/api/ae13b/clean-forward-market-feed/refresh"
        return _json_ok(data)
    except Exception as exc:  # noqa: BLE001
        return _json_ok(_structured_refresh_failure_response(exc, "Clean Forward Market Feed"))


@app.get("/api/ae13b/live-market")
def ae13b_live_market(
    limit: int = Query(50, ge=1, le=200),
    status: str | None = Query(
        None,
        description=(
            "API-level filter: all|passed|blocked|watch|demo|social|opportunistic|"
            "unknown|unresolved|registered"
        ),
    ),
    filter_mode: str | None = Query(
        "hide",
        description="hide (default) or highlight — hide removes non-matching rows",
    ),
):
    from app.runtime.ui_get_network_guard import ui_get_network_guard

    with ui_get_network_guard("/api/ae13b/live-market"):
        try:
            from app.ae13b_product.runtime_market_feed import build_live_market_from_index

            data = build_live_market_from_index(
                limit=limit,
                status_filter=status,
                filter_mode=filter_mode,
            )
            if isinstance(data, dict):
                data.setdefault("ok", True)
                data.setdefault("status", "ready")
                data.setdefault("user_message", "")
            return _json_ok(data)
        except Exception as exc:  # noqa: BLE001
            return _json_ok({
                "ok": False,
                "status": "unavailable",
                "user_message": "Live market data is unavailable. Demo controls are still available.",
                "details": str(exc)[:300],
                "rows": [],
                "live_pairs_count": 0,
                "passed_filter": 0,
                "dropped_blocked": 0,
                "demo_mode_badge": "LIVE DISABLED / DEMO ONLY",
                "filter_mode": filter_mode or "hide",
                "filter_hides_non_matching": True,
                "paper_demo_only": True,
            })


@app.get("/api/ae13b/news-sentiment-cache")
def ae13b_news_sentiment_cache(limit: int = Query(15, ge=1, le=50)):
    """AE18 cached-only RSS / News Sentiment panel — no live RSS fetch on GET."""
    from app.runtime.ui_get_network_guard import ui_get_network_guard

    with ui_get_network_guard("/api/ae13b/news-sentiment-cache"):
        try:
            from app.ae13b_product.news_sentiment_cache import build_cached_news_sentiment

            return _json_ok(build_cached_news_sentiment(limit=limit))
        except Exception as exc:  # noqa: BLE001
            from app.ae13b_product.news_sentiment_cache import (
                NEWS_SENTIMENT_CACHE_UNAVAILABLE,
            )

            return _json_ok(
                {
                    "ok": True,
                    "status": "cached_status",
                    "rss_news_sentiment_status": NEWS_SENTIMENT_CACHE_UNAVAILABLE,
                    "sentiment_cache_missing_reason": f"{type(exc).__name__}: {str(exc)[:200]}",
                    "user_message": f"{NEWS_SENTIMENT_CACHE_UNAVAILABLE}: local sentiment cache could not be read.",
                    "rss_cached_items_count": 0,
                    "cached_sentiment_records_count": 0,
                    "rss_last_fetch_at": None,
                    "aggregate_sentiment_score": None,
                    "items": [],
                    "count": 0,
                    "panel_blank": False,
                    "get_path_fetches_rss_live": False,
                }
            )


@app.post("/api/ae13b/news-sentiment-refresh")
async def ae13b_news_sentiment_refresh(limit: int = Query(15, ge=1, le=50)):
    """Explicit POST-only sentiment refresh — the only path allowed to fetch RSS."""
    from app.ae13b_product.news_sentiment_cache import build_cached_news_sentiment
    from app.runtime.shutdown import CONTROLLED_SHUTDOWN_SKIP, is_shutting_down

    if is_shutting_down():
        data = build_cached_news_sentiment(limit=limit)
        data["refresh_status"] = "SKIPPED"
        data["refresh_error_code"] = CONTROLLED_SHUTDOWN_SKIP
        data["user_message"] = "Sentiment refresh skipped — application is shutting down."
        return _json_ok(data)

    try:
        from app.analytics.sentiment import archive_rss_sentiment

        await archive_rss_sentiment()
        data = build_cached_news_sentiment(limit=limit)
        data["refresh_status"] = "OK"
        return _json_ok(data)
    except Exception as exc:  # noqa: BLE001
        from app.ae13b_product.provider_refresh_errors import (
            build_refresh_failure,
            classify_refresh_exception,
        )

        data = build_cached_news_sentiment(limit=limit)
        failure = build_refresh_failure(
            error_code=classify_refresh_exception(exc),
            provider="rss",
            exception=exc,
            reason=f"RSS refresh failed: {exc}",
        )
        data.update(failure)
        data["ok"] = True
        return _json_ok(data)


@app.get("/api/ae13b/rss-sentiment")
async def ae13b_rss_sentiment(limit: int = Query(15, ge=1, le=30)):
    from app.analytics.sentiment import fetch_rss_sentiment_matrix

    try:
        data = await fetch_rss_sentiment_matrix(limit=limit)
        items = []
        for it in data.get("items") or []:
            score = float(it.get("score") or 0)
            if score > 0.05:
                label = "Positive"
            elif score < -0.05:
                label = "Negative"
            else:
                label = "Neutral"
            items.append(
                {
                    "headline": it.get("headline"),
                    "source": it.get("source") or data.get("source"),
                    "timestamp": it.get("timestamp") or data.get("checked_at_utc") or _utc_now_iso(),
                    "sentiment_score": score,
                    "sentiment_label": label,
                    "related_coin_pair": None,
                    "relevance_note": "Headline lexicon score — not SOCIAL_CONFIRMED",
                }
            )
        return {
            "ok": True,
            "status": "ready" if items else "empty",
            "user_message": "",
            "available": bool(items),
            "unavailable_reason": None if items else "RSS feed returned no headlines",
            "aggregate_sentiment_score": data.get("aggregate_score"),
            "latest_rss_update": _utc_now_iso() if items else None,
            "feed_url": data.get("feed_url"),
            "source": data.get("source"),
            "count": len(items),
            "items": items,
            "rss_status": "available" if items else "empty",
        }
    except Exception as exc:
        return {
            "ok": False,
            "status": "unavailable",
            "user_message": "RSS / news sentiment is unavailable.",
            "available": False,
            "unavailable_reason": f"RSS unavailable: {exc}",
            "aggregate_sentiment_score": None,
            "latest_rss_update": None,
            "items": [],
            "count": 0,
            "rss_status": "failed",
            "details": str(exc)[:300],
        }


@app.get("/api/ae13b/provider-status")
def ae13b_provider_status():
    try:
        from app.ae13b_product.provider_status import build_provider_status

        data = build_provider_status()
        if isinstance(data, dict):
            data.setdefault("ok", True)
            data.setdefault("status", "ready")
        return _json_ok(data)
    except Exception as exc:  # noqa: BLE001
        return _json_ok({
            "ok": False,
            "status": "unavailable",
            "user_message": "Provider status is unavailable.",
            "details": str(exc)[:300],
            "provider_selected": "none",
            "provider_reachable": False,
            "model_name": None,
            "model_available": False,
            "assistant_mode": "inactive",
            "local_rules_active": True,
            "rss_active": False,
            "demo_trading_blocked_by_provider": False,
            "provider_health": "inactive",
            "provider_health_label": "Inactive",
            "provider_status_explanation": (
                "Provider status unavailable. Local rules still active. "
                "Demo trading not blocked by LLM."
            ),
            "trade_authority": "none",
            "runtime_mode": "DEMO",
            "llm_provider_selected": "none",
            "llm_provider_actually_active": False,
            "fail_soft": True,
            "last_health_check_at": _utc_now_iso(),
        })


@app.get("/api/ae14/readiness")
def ae14_readiness(min_tradable_rows: int = Query(10, ge=1, le=1000)):
    """AE13I Smoke Addendum (Part E): AE14 negative-control / trading-validation readiness."""
    try:
        from app.ae13b_product.ae14_readiness import compute_ae14_readiness
        from app.ae13b_product.live_market import build_live_market

        market = build_live_market(limit=200)
        readiness = compute_ae14_readiness(
            market_rows=market.get("rows"),
            min_tradable_rows_for_ae14=min_tradable_rows,
        )
        return _json_ok({
            "ok": True,
            "status": "ready",
            **readiness,
        })
    except Exception as exc:  # noqa: BLE001
        from app.ae13b_product.ae14_readiness import compute_ae14_readiness

        fallback = compute_ae14_readiness(market_rows=[], min_tradable_rows_for_ae14=min_tradable_rows)
        return _json_ok({
            "ok": False,
            "status": "unavailable",
            "user_message": "AE14 readiness could not be fully computed; treated as no tradable rows.",
            "details": str(exc)[:300],
            **fallback,
        })


@app.get("/api/ae13b/ai-assistant-status")
def ae13b_ai_assistant_status():
    try:
        from app.ae13b_product.provider_status import build_ai_assistant_status

        data = build_ai_assistant_status()
        if isinstance(data, dict):
            data.setdefault("ok", True)
            data.setdefault("status", "ready")
        return data
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": "unavailable",
            "user_message": "AI assistant status is unavailable.",
            "details": str(exc)[:300],
            "label": "AI Assistant unavailable",
            "capability": "Explanation only when available — cannot place trades.",
        }


# BEGIN MANUAL_PORTFOLIO_PRICE_ALIAS_SAFE_JSON_OK_WRAPPER_V2
def _manual_portfolio_alias_num_v2(value):
    try:
        if value is None:
            return None
        s = str(value).strip()
        if not s or s.upper() in {"N/A", "NA", "NONE", "NULL", "UNAVAILABLE"}:
            return None
        x = float(s)
        return x if x > 0 else None
    except Exception:
        return None


def _manual_portfolio_alias_price_display_v2(value):
    x = _manual_portfolio_alias_num_v2(value)
    if x is None:
        return "N/A (UNAVAILABLE)"
    return f"{x:.12g}"


def _manual_portfolio_alias_money_display_v2(value):
    try:
        return f"${float(value):.4f}"
    except Exception:
        return "N/A (UNAVAILABLE)"


def _manual_portfolio_alias_pct_display_v2(value):
    try:
        return f"{float(value):.1f}%"
    except Exception:
        return "N/A (UNAVAILABLE)"


def _manual_portfolio_alias_position_like_v2(d):
    if not isinstance(d, dict):
        return False
    return (
        ("current_price_display" in d or "mark_price_lookup_status" in d or "unrealized_pnl_display" in d)
        and ("symbol" in d or "pair_address" in d or "canonical_market_identity" in d or "mark_price_lookup_key" in d)
    )


def _manual_portfolio_alias_fix_position_v2(d):
    if not _manual_portfolio_alias_position_like_v2(d):
        return d

    # Important: do NOT use entry_price or fill_price as current price.
    # Use only real current/mark aliases already present in the API payload.
    price = None
    source = None
    for key in (
        "current_price_usd",
        "mark_price_usd",
        "latest_price",
        "latest_price_usd",
        "market_price_usd",
        "tradable_price_usd",
    ):
        x = _manual_portfolio_alias_num_v2(d.get(key))
        if x is not None:
            price = x
            source = key
            break

    if price is None:
        return d

    display_is_bad = str(d.get("current_price_display") or "").upper().startswith("N/A")
    status_is_bad = str(d.get("mark_price_lookup_status") or "").upper() in {
        "PRICE_NOT_AVAILABLE",
        "LEGACY_POSITION_IDENTITY_REPAIR_NEEDED",
    }
    failure_is_bad = str(d.get("price_resolution_failure_reason") or "").strip() not in {"", "None", "null"}

    if d.get("current_price") is None or display_is_bad or status_is_bad or failure_is_bad:
        d["current_price"] = price
        d["current_price_numeric"] = price
        d["current_price_usd"] = price
        d["mark_price_usd"] = price
        d["latest_price"] = price

        d["current_price_display"] = _manual_portfolio_alias_price_display_v2(price)
        d["current_price_status"] = "PRICE_OK_FROM_NUMERIC_ALIAS"
        d["mark_price_status"] = "PRICE_OK_FROM_NUMERIC_ALIAS"
        d["mark_price_lookup_status"] = "PRICE_OK_FROM_NUMERIC_ALIAS"
        d["current_price_source"] = f"numeric_alias:{source}"
        d["mark_price_unavailable_reason"] = ""
        d["price_resolution_failure_reason"] = ""
        # PORTFOLIO_LATEST_DB_MARK_PRICE_APPLY_V1
        # Refresh stale paper_state price aliases from the newest DB exact-pair/coin mark.
        _apply_portfolio_latest_db_mark_price_v1(d)

        d["price_status_detail"] = "current price resolved from numeric current/mark alias already present in portfolio payload"

        # API_PORTFOLIO_EXIT_STATE_CONSISTENCY_FIXED_V1
        # If numeric current/mark price exists, all derived display/exit fields
        # must be consistent. Do not leave stale PRICE_NOT_AVAILABLE blockers.
        d["position_market_data_state"] = "MARKET_DATA_READY"
        d["financial_data_status"] = "READY"
        d["frontend_must_not_compute_pnl"] = False
        d["mark_fresh"] = True
        d["matched_market_pair_status"] = "MATCHED_FROM_NUMERIC_ALIAS"
        d["position_value_display"] = d.get("position_value_display")
        d["pnl_display_status"] = "ready"
        d["pnl_display_message"] = ""
        d["reason"] = "Current mark price available."
        d["traffic_light_reason"] = "Current mark price available."
        d["traffic_light_status"] = "green"

        if str(d.get("exit_blocker") or "").upper() == "PRICE_NOT_AVAILABLE":
            d["exit_blocker"] = ""

        d["bot_exit_reason"] = (
            "No TP/SL/time-stop exit rule triggered."
            if str(d.get("bot_exit_reason") or "").lower() in {"no current price", "no current mark price"}
            else d.get("bot_exit_reason")
        )

        d["close_freshness_status"] = d.get("close_freshness_status") or "PRICE_OK_FROM_NUMERIC_ALIAS"
        d["close_price_source"] = d.get("close_price_source") or d.get("current_price_source") or "numeric_alias_current_price"
        d["close_used_fallback_price"] = False

        d["exit_status"] = (
            "OPEN_MONITORING"
            if str(d.get("exit_status") or "").upper() in {"", "PRICE_NOT_AVAILABLE"}
            else d.get("exit_status")
        )
        d["exit_status_display"] = (
            "Price has not reached TP/SL"
            if not d.get("exit_status_display") or "PRICE_NOT_AVAILABLE" in str(d.get("exit_status_display")).upper()
            else d.get("exit_status_display")
        )
        d["exit_status_label"] = (
            "Price has not reached TP/SL"
            if not d.get("exit_status_label") or "PRICE_NOT_AVAILABLE" in str(d.get("exit_status_label")).upper()
            else d.get("exit_status_label")
        )

        # Fill position_value_display and recompute PnL from the final selected mark price.
        # PORTFOLIO_PNL_RECOMPUTE_AFTER_LATEST_MARK_FIXED_V1
        try:
            _qty = float(d.get("quantity") or 0)
            _px = float(d.get("current_price") or d.get("current_price_usd") or d.get("mark_price_usd") or 0)
            _entry = float(d.get("entry_price") or d.get("fill_price") or 0)

            if _qty > 0 and _px > 0:
                d["position_value_usd"] = _qty * _px
                d["position_value_display"] = f"${(_qty * _px):.4f}"

            if _qty > 0 and _px > 0 and _entry > 0:
                _pnl_usd = (_px - _entry) * _qty
                _pnl_pct = ((_px / _entry) - 1.0) * 100.0

                d["unrealized_pnl_usd"] = _pnl_usd
                d["unrealized_pnl_numeric"] = _pnl_usd
                d["unrealized_pnl_display"] = f"${_pnl_usd:.4f}"

                # Store as percent-points, consistent with existing API payload style:
                # -66.7 means -66.7%, not -0.667.
                d["unrealized_pnl_pct"] = _pnl_pct
                d["unrealized_pnl_pct_numeric"] = _pnl_pct
                d["unrealized_pnl_pct_display"] = f"{_pnl_pct:.1f}%"

                d["pnl_display_status"] = "ready"
                d["pnl_display_message"] = ""
        except Exception:
            pass


        ts = d.get("current_price_timestamp") or d.get("price_updated_at") or d.get("last_seen_at")
        if ts:
            d["current_price_timestamp"] = ts

        qty = _manual_portfolio_alias_num_v2(d.get("quantity"))
        entry = _manual_portfolio_alias_num_v2(d.get("entry_price") or d.get("fill_price"))
        if qty is not None and entry is not None and entry > 0:
            pnl = (price - entry) * qty
            pct = ((price / entry) - 1.0) * 100.0
            d["unrealized_pnl_usd"] = pnl
            d["unrealized_pnl_pct"] = pct
            d["unrealized_pnl_numeric"] = pnl
            d["unrealized_pnl_pct_numeric"] = pct
            d["unrealized_pnl_display"] = _manual_portfolio_alias_money_display_v2(pnl)
            d["unrealized_pnl_pct_display"] = _manual_portfolio_alias_pct_display_v2(pct)

    return d


def _manual_portfolio_alias_walk_v2(obj):
    if isinstance(obj, dict):
        if _manual_portfolio_alias_position_like_v2(obj):
            _manual_portfolio_alias_fix_position_v2(obj)
        for k, v in list(obj.items()):
            obj[k] = _manual_portfolio_alias_walk_v2(v)
        return obj
    if isinstance(obj, list):
        return [_manual_portfolio_alias_walk_v2(x) for x in obj]
    return obj


if "_json_ok" in globals() and not getattr(_json_ok, "_manual_portfolio_alias_wrapped_v2", False):
    _manual_original_json_ok_v2 = _json_ok

    def _json_ok(*args, **kwargs):
        args = list(args)
        if args:
            try:
                args[0] = _manual_portfolio_alias_walk_v2(args[0])
            except Exception:
                pass
            try:
                args[0] = _targeted_ui_finalize_payload_v2(args[0])
            except Exception:
                pass
        else:
            # Common names, depending on the local _json_ok signature.
            for key in ("data", "payload", "content", "body"):
                if key in kwargs:
                    try:
                        kwargs[key] = _manual_portfolio_alias_walk_v2(kwargs[key])
                    except Exception:
                        pass
                    try:
                        kwargs[key] = _targeted_ui_finalize_payload_v2(kwargs[key])
                    except Exception:
                        pass
                    break
        return _manual_original_json_ok_v2(*args, **kwargs)

    _json_ok._manual_portfolio_alias_wrapped_v2 = True
# END MANUAL_PORTFOLIO_PRICE_ALIAS_SAFE_JSON_OK_WRAPPER_V2

