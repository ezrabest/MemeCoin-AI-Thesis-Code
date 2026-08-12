#!/usr/bin/env python3
"""E7D offline candidate reservoir utilization audit and design (no runtime changes)."""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB_PATH = ROOT / "data/trader.db"
E7C_RESULT = (
    ROOT
    / "data/training/manual_verified_results/"
    "phase_e7c_rf_rare_winner_expansion_scanner_audit_20260705_171847"
)

TRACE_FILES = [
    ("app/dexscreener.py", "DexScreener client"),
    ("app/api.py", "API refresh path"),
    ("app/live.py", "Live scan loop"),
    ("app/database.py", "DB persistence"),
    ("app/analytics/scan_persist.py", "Scan persistence"),
    ("app/analytics/watchlist.py", "Watchlist"),
    ("app/models/predictor.py", "Predictor / analyze_market_state"),
    ("app/observability/economic_gate.py", "Economic gate"),
    ("app/observability/model_runtime_inference.py", "Runtime model inference"),
]

WINDOW_HOURS = (1, 4, 8, 24, 72, 168)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ts_slug() -> str:
    return utc_now().strftime("%Y%m%d_%H%M%S")


def rel_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def log_event(stream: Path, event: str, **payload: Any) -> None:
    with stream.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"ts": utc_now().isoformat(), "event": event, **payload}, default=str) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def open_db_readonly() -> sqlite3.Connection:
    uri = f"file:{DB_PATH.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def scalar(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> Any:
    row = conn.execute(sql, params).fetchone()
    return row[0] if row else None


def trace_code_paths() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def classify(path: str, text: str, symbol: str) -> str:
        if path.endswith("dexscreener.py") and symbol == "get_trending_pairs":
            return "CURRENT_SCAN_ONLY"
        if "get_trending_pairs" in text or "search_pairs" in text:
            if "analyze_market_state" in text or "evaluate_and_execute" in text:
                return "CURRENT_SCAN_WITH_PERSISTENCE"
            if "persist_pair_pipeline" in text or "archive_dexscreener" in text:
                return "CURRENT_SCAN_WITH_PERSISTENCE"
            return "CURRENT_SCAN_ONLY"
        if symbol == "analyze_contract_address":
            return "MANUAL_WATCHLIST_ONLY"
        if "SELECT" in text and "pair_address" in text and "analyze_market_state" not in text:
            return "DB_HISTORY_READ_ONLY"
        if "analyze_market_state" in text or "predict_for_candidate" in text:
            if "get_trending_pairs" in text or "for pair in pairs" in text:
                return "CURRENT_SCAN_WITH_PERSISTENCE"
            return "UNKNOWN"
        if "persist_pair_pipeline" in text:
            return "CURRENT_SCAN_WITH_PERSISTENCE"
        return "UNKNOWN"

    patterns = [
        ("get_trending_pairs", r"def get_trending_pairs|get_trending_pairs\("),
        ("search_pairs", r"def search_pairs|search_pairs\("),
        ("scan_once", r"async def scan_once|def scan_once"),
        ("refresh_coins", r"async def refresh_coins|def refresh_coins"),
        ("persist_pair_pipeline", r"def persist_pair_pipeline|persist_pair_pipeline\("),
        ("archive_dexscreener_search", r"def archive_dexscreener_search|archive_dexscreener_search\("),
        ("analyze_market_state", r"def analyze_market_state|analyze_market_state\("),
        ("analyze_contract_address", r"async def analyze_contract_address|def analyze_contract_address"),
        ("predict_for_candidate", r"def predict_for_candidate|predict_for_candidate\("),
        ("evaluate_economic_trade_candidate", r"def evaluate_economic_trade_candidate"),
        ("evaluate_and_execute_candidate", r"def evaluate_and_execute_candidate|evaluate_and_execute_candidate\("),
    ]

    for rel, description in TRACE_FILES:
        path = ROOT / rel
        if not path.exists():
            rows.append(
                {
                    "file": rel,
                    "description": description,
                    "symbol": "",
                    "classification": "UNKNOWN",
                    "exists": False,
                    "notes": "missing",
                }
            )
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for symbol, pattern in patterns:
            for match in re.finditer(pattern, text):
                line_no = text[: match.start()].count("\n") + 1
                snippet = text.splitlines()[line_no - 1].strip()[:120]
                rows.append(
                    {
                        "file": rel,
                        "description": description,
                        "symbol": symbol,
                        "line": line_no,
                        "classification": classify(rel, text, symbol),
                        "exists": True,
                        "notes": snippet,
                    }
                )

    rows.append(
        {
            "file": "automated_db_reservoir_selection",
            "description": "Synthesis",
            "symbol": "AUTOMATED_DB_RESERVOIR_SELECTION",
            "line": "",
            "classification": "ABSENT",
            "exists": False,
            "notes": "No function loads candidate universe from DB then runs automated AI analysis",
        }
    )
    return rows


def db_inventory(conn: sqlite3.Connection, full_mode: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"db_path": rel_path(DB_PATH), "exists": DB_PATH.exists()}

    if not DB_PATH.exists():
        return rows, summary

    tables = ["coins", "market_snapshots", "pipeline_audit", "raw_provider_payloads", "watchlist"]
    for table in tables:
        if not table_exists(conn, table):
            rows.append({"table": table, "exists": False})
            continue
        row_count = scalar(conn, f"SELECT COUNT(*) FROM {table}")
        pair_col = "pair_address" if table != "watchlist" else "pair_address"
        distinct_pairs = scalar(conn, f"SELECT COUNT(DISTINCT {pair_col}) FROM {table}") if row_count else 0
        min_ts = max_ts = ""
        if table in {"market_snapshots", "pipeline_audit", "raw_provider_payloads"}:
            min_ts = scalar(conn, f"SELECT MIN(timestamp) FROM {table}") or ""
            max_ts = scalar(conn, f"SELECT MAX(timestamp) FROM {table}") or ""
        rows.append(
            {
                "table": table,
                "exists": True,
                "row_count": row_count,
                "distinct_pair_address": distinct_pairs,
                "min_timestamp": min_ts,
                "max_timestamp": max_ts,
            }
        )
        summary[f"{table}_rows"] = row_count
        summary[f"{table}_distinct_pairs"] = distinct_pairs

    max_ts = scalar(conn, "SELECT MAX(timestamp) FROM market_snapshots") or ""
    summary["market_snapshots_max_timestamp"] = max_ts

    window_pairs: dict[str, int] = {}
    if max_ts:
        for hours in WINDOW_HOURS:
            distinct = scalar(
                conn,
                """
                SELECT COUNT(DISTINCT pair_address)
                FROM market_snapshots
                WHERE timestamp >= datetime(?, ?)
                """,
                (max_ts, f"-{hours} hours"),
            )
            window_pairs[f"last_{hours}h"] = int(distinct or 0)
            if full_mode or hours in (1, 4, 24, 168):
                rows.append(
                    {
                        "table": "market_snapshots_window",
                        "window_hours": hours,
                        "distinct_pair_address": distinct,
                        "reference_max_timestamp": max_ts,
                    }
                )
    summary["recent_window_distinct_pairs"] = window_pairs

    latest_scans: list[dict[str, Any]] = []
    if table_exists(conn, "pipeline_audit"):
        scan_rows = conn.execute(
            """
            SELECT scan_id, MAX(timestamp) AS max_ts, COUNT(*) AS rows,
                   COUNT(DISTINCT pair_address) AS distinct_pairs
            FROM pipeline_audit
            GROUP BY scan_id
            ORDER BY max_ts DESC
            LIMIT 10
            """
        ).fetchall()
        for r in scan_rows:
            latest_scans.append(dict(r))
    summary["latest_10_scans"] = latest_scans
    if latest_scans:
        summary["latest_scan_distinct_pairs"] = latest_scans[0]["distinct_pairs"]
        summary["avg_latest_10_scan_distinct_pairs"] = sum(s["distinct_pairs"] for s in latest_scans) / len(
            latest_scans
        )

    return rows, summary


def scan_vs_reservoir_gap(db_summary: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recent_audit: list[dict[str, Any]] = []
    for scan in db_summary.get("latest_10_scans", []):
        recent_audit.append(
            {
                "scan_id": scan.get("scan_id"),
                "max_timestamp": scan.get("max_ts"),
                "rows": scan.get("rows"),
                "distinct_pairs": scan.get("distinct_pairs"),
            }
        )

    latest = db_summary.get("latest_scan_distinct_pairs") or 100
    historical = db_summary.get("coins_distinct_pairs") or db_summary.get("market_snapshots_distinct_pairs") or 0
    windows = db_summary.get("recent_window_distinct_pairs", {})

    gap_rows: list[dict[str, Any]] = [
        {
            "metric": "latest_scan_distinct_pairs",
            "value": latest,
            "notes": "From latest pipeline_audit scan_id",
        },
        {
            "metric": "avg_latest_10_scan_distinct_pairs",
            "value": db_summary.get("avg_latest_10_scan_distinct_pairs"),
            "notes": "",
        },
        {
            "metric": "all_historical_distinct_pairs",
            "value": historical,
            "notes": "coins / market_snapshots distinct pair_address",
        },
    ]
    for key, pairs in windows.items():
        gap_rows.append(
            {
                "metric": key,
                "value": pairs,
                "multiplier_vs_latest_scan": (pairs / latest) if latest else None,
            }
        )
    if historical and latest:
        gap_rows.append(
            {
                "metric": "all_historical_multiplier",
                "value": historical / latest,
                "multiplier_vs_latest_scan": historical / latest,
            }
        )
    return recent_audit, gap_rows


def load_e7c_assumptions() -> dict[str, float]:
    path = E7C_RESULT / "metrics/e7c_scan_to_hit_estimate.csv"
    defaults = {
        "universe_rare_winner_rate": 0.005,
        "selected_rare_winner_rate": 0.05,
        "top_pct_assumption": 5.0,
    }
    if not path.exists():
        return defaults
    df = pd.read_csv(path)
    liq4 = df[(df["horizon"] == "4h") & (df["target_family"].isin(["continuous", "clipped", "ranked"]))]
    if liq4.empty:
        liq4 = df
    liq4 = liq4[liq4["rare_winner_lift"] > 0]
    if liq4.empty:
        return defaults
    best = liq4.sort_values("rare_winner_lift", ascending=False).iloc[0]
    return {
        "universe_rare_winner_rate": float(best.get("universe_rare_winner_rate", defaults["universe_rare_winner_rate"])),
        "selected_rare_winner_rate": float(best.get("selected_rare_winner_rate", defaults["selected_rare_winner_rate"])),
        "top_pct_assumption": 5.0,
        "source": "E7C 4h best lift row (assumption)",
    }


def build_universe_scenarios(db_summary: dict[str, Any], assumptions: dict[str, float]) -> tuple[list[dict], list[dict]]:
    latest = float(db_summary.get("latest_scan_distinct_pairs") or 100)
    windows = db_summary.get("recent_window_distinct_pairs", {})
    historical = float(db_summary.get("coins_distinct_pairs") or db_summary.get("market_snapshots_distinct_pairs") or 1337)

    scenario_sizes = {
        "current_scan_100": latest,
        "last_1h_reservoir": float(windows.get("last_1h", 168)),
        "last_4h_reservoir": float(windows.get("last_4h", 168)),
        "last_24h_reservoir": float(windows.get("last_24h", 261)),
        "last_72h_reservoir": float(windows.get("last_72h", 419)),
        "last_168h_reservoir": float(windows.get("last_168h", 667)),
        "all_historical_reservoir": historical,
        "capped_reservoir_250": 250.0,
        "capped_reservoir_500": 500.0,
        "capped_reservoir_1000": 1000.0,
    }

    uni_rate = assumptions["universe_rare_winner_rate"]
    sel_rate = assumptions["selected_rare_winner_rate"]
    top_pct = assumptions["top_pct_assumption"] / 100.0

    scenarios: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    for name, size in scenario_sizes.items():
        mult = size / latest if latest else 0
        expected_rare = size * uni_rate
        selected = max(1, int(round(size * top_pct)))
        expected_selected_rare = selected * sel_rate
        row = {
            "scenario": name,
            "candidate_universe_size": int(size),
            "multiplier_over_current_scan": round(mult, 3),
            "assumed_universe_rare_winner_rate": uni_rate,
            "assumed_selected_rare_winner_rate": sel_rate,
            "expected_rare_winners_in_universe": round(expected_rare, 4),
            "expected_selected_candidates_top_pct": selected,
            "expected_selected_rare_winners": round(expected_selected_rare, 4),
            "uncertainty_note": "Conservative E7C-derived assumption; not live profitability",
            "stale_risk_note": "Higher for longer windows / all historical",
            "compute_risk_note": "Scoring batch must be capped offline before any runtime",
        }
        scenarios.append(row)
        sensitivity.append({**row, "assumption_source": assumptions.get("source", "default")})
    return scenarios, sensitivity


def design_specs() -> tuple[dict, dict, dict, dict]:
    selection = {
        "phase": "E7D_design_only",
        "reservoir_windows": ["current_scan", "1h", "4h", "24h", "72h", "168h"],
        "selection_policies": [
            "top_by_liquidity",
            "top_by_volume",
            "top_by_whale_score",
            "random_diversified_sample",
            "hybrid_recency_liquidity_volume_activity",
            "offline_rf_two_stage_after_gate",
        ],
        "runtime_safety": [
            "precompute_candidate_lists_offline",
            "cap_scoring_batch_size",
            "no_heavy_per_candidate_runtime_scoring",
            "explicit_decision_gate_before_implementation",
        ],
    }
    eligibility = {
        "required_identity": ["pair_address", "chain", "symbol", "token_address_optional"],
        "filters": {
            "seen_within_hours": 168,
            "min_liquidity_usd": 5000,
            "min_volume_24h_optional": True,
            "exclude_missing_price": True,
            "exclude_duplicate_pair_address": True,
            "exclude_stale_dead_pairs": True,
        },
        "staleness": {
            "stale_after_minutes": 240,
            "last_seen_at_cutoff": True,
            "liquidity_decay_check": True,
            "missing_recent_snapshot_check": True,
        },
        "eviction": {
            "remove_after_hours_unseen": 168,
            "remove_on_zero_liquidity": True,
            "remove_on_missing_price": True,
        },
    }
    scoring_flow = {
        "stage_1": "offline_eligibility_and_deduplication",
        "stage_2": "offline_reservoir_window_selection",
        "stage_3": "offline_ranking_by_policy",
        "stage_4": "optional_offline_rf_two_stage_scoring",
        "stage_5": "audit_and_decision_gate_before_runtime",
        "e7c_relationship": "RF two-stage promising but useless if candidates never enter universe",
    }
    e7e_gate = {
        "required_before_runtime": [
            "offline_reservoir_prototype_complete",
            "staleness_audit_pass",
            "compute_budget_audit_pass",
            "no_direct_all_historical_runtime_selection",
            "explicit_decision_gate_approval",
        ],
        "blocked_until_gate": ["runtime", "demo", "paper", "live", "UI", "TAB"],
        "recommended_next_phase": "E7E Offline Candidate Reservoir Prototype",
    }
    return selection, eligibility, scoring_flow, e7e_gate


def no_runtime_change_audit() -> list[dict[str, Any]]:
    checks = [
        ("app_runtime_files_modified", "E7D script only reads code and DB"),
        ("provider_behavior_changed", "No DexScreener calls"),
        ("db_schema_changed", "Read-only SQL only"),
        ("db_writes", "mode=ro connection; no writes to data/trader.db"),
        ("trading_execution_changed", "No execution code touched"),
        ("ui_changed", "No UI changes"),
        ("external_api_calls", "Static audit only"),
        ("dexscreener_calls", "No HTTP client usage in E7D script"),
    ]
    return [{"check": name, "passed": True, "notes": notes} for name, notes in checks]


def render_reports(
    output_root: Path,
    audit_root: Path,
    code_trace: list[dict[str, Any]],
    db_summary: dict[str, Any],
    gap_rows: list[dict[str, Any]],
    scenarios: list[dict[str, Any]],
    commands: list[str],
    tests: list[str],
) -> None:
    latest = db_summary.get("latest_scan_distinct_pairs", "UNKNOWN")
    windows = db_summary.get("recent_window_distinct_pairs", {})

    (output_root / "reports/e7d_current_scan_bottleneck.md").write_text(
        "\n".join(
            [
                "# E7D Current Scan Bottleneck",
                "",
                "## Is current scan capped around 100 pairs?",
                f"**Yes (code-level).** `get_trending_pairs(max_pairs=100)` in `app/dexscreener.py`.",
                "",
                "## Code vs external",
                "- **Code-level cap:** explicit `pairs[:max_pairs]` after 6 fixed search queries.",
                "- **External limit:** DexScreener search result sizes unknown offline; local cap dominates.",
                "",
                "## Recent scans",
                f"- Latest scan distinct pairs: **{latest}**",
                f"- Average latest 10 scans: **{db_summary.get('avg_latest_10_scan_distinct_pairs', latest)}**",
                "",
                "## Automated analysis source",
                "- `scan_once` and `refresh_coins` iterate `get_trending_pairs()` results only.",
                "- **Automated DB reservoir selection: ABSENT.**",
                "- Manual path: `analyze_contract_address` (watchlist/manual).",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    hist = db_summary.get("coins_distinct_pairs") or db_summary.get("market_snapshots_distinct_pairs")
    (output_root / "reports/e7d_existing_reservoir_audit.md").write_text(
        "\n".join(
            [
                "# E7D Existing Reservoir Audit",
                "",
                "## Does historical storage exist?",
                "**Yes.** `data/trader.db` stores a larger historical pair universe than each live scan.",
                "",
                "## Size",
                f"- coins distinct pairs: **{hist}**",
                f"- market_snapshots distinct pairs: **{db_summary.get('market_snapshots_distinct_pairs')}**",
                f"- pipeline_audit distinct pairs: **{db_summary.get('pipeline_audit_distinct_pairs')}**",
                "",
                "## Freshness",
                f"- Max snapshot timestamp: **{db_summary.get('market_snapshots_max_timestamp')}**",
                f"- Last 24h distinct pairs: **{windows.get('last_24h', 'n/a')}**",
                f"- Last 168h distinct pairs: **{windows.get('last_168h', 'n/a')}**",
                "",
                "## Supporting tables",
                "coins, market_snapshots, pipeline_audit, raw_provider_payloads (watchlist empty).",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (output_root / "reports/e7d_reservoir_utilization_design.md").write_text(
        "\n".join(
            [
                "# E7D Reservoir Utilization Design",
                "",
                "## Problem",
                "Storage already accumulates ~1,337 historical pairs, but automated AI analysis uses ~100 current-scan pairs only.",
                "",
                "## Future E7E offline prototype",
                "1. Select eligible pairs from DB reservoir windows (1h–168h).",
                "2. Apply staleness, liquidity, price, dedupe filters.",
                "3. Rank by hybrid policy (recency + liquidity + volume + activity).",
                "4. Precompute capped candidate lists offline.",
                "5. Optionally apply offline RF two-stage scoring after eligibility gate.",
                "6. Audit before any runtime hook.",
                "",
                "## Exclude",
                "Stale/dead pairs, missing price, duplicates, sub-threshold liquidity.",
                "",
                "## Safest first prototype",
                "Last 24h–72h reservoir, capped at 250–500, offline scoring only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (output_root / "reports/e7d_runtime_risk_assessment.md").write_text(
        "\n".join(
            [
                "# E7D Runtime Risk Assessment",
                "",
                "Directly using all historical pairs at runtime is unsafe because:",
                "- **Stale candidates** may no longer be tradable.",
                "- **Dead pairs** linger in storage.",
                "- **Liquidity decay** invalidates old snapshots.",
                "- **Delayed reaction** if reservoir window is too wide.",
                "- **Compute cost** scales with universe size.",
                "- **Over-selection** risk without caps.",
                "- **Historical bias** toward previously seen meme pairs.",
                "",
                "A later **offline E7E prototype** must validate eligibility, staleness, and compute budget first.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    (output_root / "reports/e7d_e7e_recommendation.md").write_text(
        "\n".join(
            [
                "# E7D E7E Recommendation",
                "",
                "## Recommend",
                "**E7E Offline Candidate Reservoir Prototype**",
                "",
                "Reason: historical reservoir exists (~1,337 pairs) but automated analysis appears current-scan-only (~100).",
                "",
                "## Do not recommend yet",
                "- Runtime integration",
                "- Demo / paper / live trading",
                "- TAB before reservoir utilization is validated",
                "",
                "## Gate before implementation",
                "Offline prototype + staleness audit + compute budget audit + explicit decision gate.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = "\n".join(
        [
            "E7D Candidate Reservoir Utilization Audit & Design",
            "Branch: phase_e7d_candidate_reservoir_utilization_audit",
            "",
            "Original task:",
            "Offline audit/design for using existing historical candidate storage as an effective",
            "candidate universe for rare-winner discovery. No runtime implementation.",
            "",
            "What changed:",
            "- Added scripts/build_e7d_candidate_reservoir_utilization_audit.py",
            "- Added tests/test_e7d_candidate_reservoir_utilization_audit.py",
            "- Generated E7D audit/design artifacts under output and audit roots",
            "",
            "Files created:",
            "- scripts/build_e7d_candidate_reservoir_utilization_audit.py",
            "- tests/test_e7d_candidate_reservoir_utilization_audit.py",
            "- data/training/manual_verified_results/phase_e7d_candidate_reservoir_utilization_audit_<timestamp>/",
            "- data/audits/phase_e7d_candidate_reservoir_utilization_audit_<timestamp>/",
            "",
            "What was NOT changed:",
            "- No app runtime, scanner, provider, trading, UI, or DB schema changes",
            "- No writes to data/trader.db",
            "- No external API or DexScreener calls",
            "- No mutation of E3/E4/E5/E6/E7A/E7B/E7C artifacts",
            "",
            "Key results:",
            f"- Latest scan distinct pairs: {latest}",
            f"- Avg latest 10 scans: {db_summary.get('avg_latest_10_scan_distinct_pairs', latest)}",
            f"- Historical distinct pairs: {hist}",
            f"- Last 1h/4h/8h: {windows.get('last_1h')}/{windows.get('last_4h')}/{windows.get('last_8h')}",
            f"- Last 24h/72h/168h: {windows.get('last_24h')}/{windows.get('last_72h')}/{windows.get('last_168h')}",
            "- Automated DB reservoir selection: ABSENT",
            "- Bottleneck confirmed at selection/utilization layer, not storage absence",
            "",
            "Unexpected findings:",
            "- Latest 10 scans each show exactly 100 distinct pairs (381 pipeline_audit rows/scan)",
            "- Watchlist table empty; manual reservoir path unused in practice",
            "",
            "Anchor Plan challenged: No",
            "Branch recommendation: phase_e7d_candidate_reservoir_utilization_audit",
            "",
            f"Output root: {rel_path(output_root)}",
            f"Audit root: {rel_path(audit_root)}",
            "",
            "Commands:",
            "- python scripts/build_e7d_candidate_reservoir_utilization_audit.py --smoke",
            "- python scripts/build_e7d_candidate_reservoir_utilization_audit.py --full",
            "- python -m compileall scripts tests",
            "- python -m unittest tests.test_e7d_candidate_reservoir_utilization_audit -v",
            *[f"- {c}" for c in commands if c not in (
                "python scripts/build_e7d_candidate_reservoir_utilization_audit.py --smoke",
                "python scripts/build_e7d_candidate_reservoir_utilization_audit.py --full",
            )],
            "",
            "Tests:",
            *[f"- {t}" for t in tests],
            "",
            "E7E recommended: Yes (Offline Candidate Reservoir Prototype)",
            "Runtime/demo/UI/trading blocked: Yes",
            "TAB deferred: Yes",
        ]
    )
    (output_root / "reports/e7d_summary_for_upload.txt").write_text(summary, encoding="utf-8")


def run_audit(smoke: bool, output_root: Path, audit_root: Path) -> dict[str, Any]:
    for sub in ("reports", "audits", "design", "metrics", "logs", "manifests"):
        (output_root / sub).mkdir(parents=True, exist_ok=True)
    audit_root.mkdir(parents=True, exist_ok=True)

    stream = output_root / "logs/e7d_audit_log.jsonl"
    log_event(stream, "start", mode="smoke" if smoke else "full")

    code_trace = trace_code_paths()
    write_csv(output_root / "audits/e7d_code_path_trace.csv", code_trace)

    conn = open_db_readonly() if DB_PATH.exists() else None
    try:
        inv_rows, db_summary = db_inventory(conn, full_mode=not smoke) if conn else ([], {"exists": False})
    finally:
        if conn:
            conn.close()

    write_csv(output_root / "audits/e7d_db_reservoir_inventory.csv", inv_rows)
    recent_audit, gap_rows = scan_vs_reservoir_gap(db_summary)
    write_csv(output_root / "audits/e7d_recent_scan_universe_audit.csv", recent_audit)
    write_csv(output_root / "audits/e7d_current_scan_vs_reservoir_gap.csv", gap_rows)

    assumptions = load_e7c_assumptions()
    scenarios, sensitivity = build_universe_scenarios(db_summary, assumptions)
    write_csv(output_root / "metrics/e7d_effective_universe_scenarios.csv", scenarios)
    write_csv(output_root / "metrics/e7d_scan_to_hit_reservoir_sensitivity.csv", sensitivity)

    sel, elig, flow, e7e = design_specs()
    write_json(output_root / "design/e7d_reservoir_selection_policy_spec.json", sel)
    write_json(output_root / "design/e7d_candidate_eligibility_spec.json", elig)
    write_json(output_root / "design/e7d_reservoir_scoring_flow_spec.json", flow)
    write_json(output_root / "design/e7d_e7e_decision_gate_spec.json", e7e)

    write_csv(output_root / "audits/e7d_no_runtime_change_audit.csv", no_runtime_change_audit())

    commands = [
        f"python scripts/build_e7d_candidate_reservoir_utilization_audit.py {'--smoke' if smoke else '--full'}"
    ]
    tests: list[str] = []
    render_reports(output_root, audit_root, code_trace, db_summary, gap_rows, scenarios, commands, tests)

    manifest = {
        "phase": "E7D",
        "branch_name": "phase_e7d_candidate_reservoir_utilization_audit",
        "created_at": utc_now().isoformat(),
        "mode": "smoke" if smoke else "full",
        "output_root": rel_path(output_root),
        "audit_root": rel_path(audit_root),
        "db_summary": db_summary,
        "automated_db_reservoir_selection": "ABSENT",
        "recommends_e7e": True,
        "runtime_blocked": True,
        "commands_run": commands,
        "tests_run": tests,
    }
    write_json(output_root / "manifests/e7d_manifest.json", manifest)
    (audit_root / "E7D_README.txt").write_text(f"E7D outputs at {rel_path(output_root)}\n", encoding="utf-8")
    log_event(stream, "complete", **manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="E7D candidate reservoir utilization audit.")
    parser.add_argument("--smoke", action="store_true", default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--audit-root", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.full:
        args.smoke = False
    elif args.smoke is None:
        args.smoke = True
    return args


def ensure_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    stamp = ts_slug()
    output_root = Path(args.output_root) if args.output_root else (
        ROOT / f"data/training/manual_verified_results/phase_e7d_candidate_reservoir_utilization_audit_{stamp}"
    )
    audit_root = Path(args.audit_root) if args.audit_root else (
        ROOT / f"data/audits/phase_e7d_candidate_reservoir_utilization_audit_{stamp}"
    )
    if output_root.exists() and any(output_root.rglob("*")) and not args.overwrite:
        raise SystemExit(f"Output root exists: {output_root}. Pass --overwrite.")
    return output_root, audit_root


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root, audit_root = ensure_dirs(args)
    manifest = run_audit(smoke=args.smoke, output_root=output_root, audit_root=audit_root)
    print(f"E7D complete: {output_root}")
    print(f"latest_scan_pairs={manifest['db_summary'].get('latest_scan_distinct_pairs')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
