"""Build API-facing summary payloads from loaded AE12 artifacts (no file I/O)."""

from __future__ import annotations

from typing import Any

from .schemas import SAFETY_DISCLAIMERS

# Smoke/test markers excluded from Qwen API/UI samples (source artifacts untouched).
_QWEN_SMOKE_CANDIDATE_IDS = frozenset({"cand-001", "test-candidate", "dummy"})
_QWEN_SMOKE_DECISION_IDS = frozenset({"dec-001", "test-decision", "dummy"})
_QWEN_SMOKE_PAIR_ADDRESSES = frozenset({"0xabc"})


def normalize_missing_warning_code(row: dict[str, Any] | None) -> str:
    """Null-safe warning code from a missing-data warning row. Never KeyError."""
    if not row:
        return "UNKNOWN"
    raw = (
        row.get("warning_code")
        or row.get("missing_field")
        or row.get("warning")
        or row.get("warning_type")
        or row.get("status")
    )
    if raw is None:
        return "UNKNOWN"
    text = str(raw).strip()
    if not text:
        return "UNKNOWN"
    return text.upper()


def is_qwen_smoke_test_row(row: dict[str, Any] | None) -> bool:
    """True for obvious smoke/test Qwen linkage sample rows."""
    if not row:
        return False
    cand = str(row.get("candidate_id") or "").strip().lower()
    dec = str(row.get("decision_id") or "").strip().lower()
    pair = str(row.get("pair_address") or "").strip().lower()
    if cand in _QWEN_SMOKE_CANDIDATE_IDS:
        return True
    if dec in _QWEN_SMOKE_DECISION_IDS:
        return True
    if pair in _QWEN_SMOKE_PAIR_ADDRESSES:
        return True
    return False


def filter_qwen_sample_rows(rows: list[dict[str, Any]] | None) -> tuple[list[dict[str, Any]], str | None]:
    """Exclude smoke/test rows from API/UI samples. Source files unchanged."""
    src = list(rows or [])
    kept = [r for r in src if not is_qwen_smoke_test_row(r)]
    note = None
    if src and not kept:
        note = "Only smoke/test Qwen rows were available in sample window."
    return kept, note


def _safe_int(value: Any, default: int | None = None) -> int | None:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_status_payload(
    *,
    maturation_root: str | None,
    census_root: str | None,
    quality_root: str | None,
    summary: dict[str, Any] | None,
    gate: dict[str, Any] | None,
    summary_load: dict[str, Any],
    gate_load: dict[str, Any],
    cache_meta: dict[str, Any],
) -> dict[str, Any]:
    gate = gate or {}
    summary = summary or {}
    missing: list[str] = []
    if summary_load.get("status") != "OK":
        missing.append(summary_load.get("missing_file") or summary_load.get("path") or "summary")
    if gate_load.get("status") != "OK":
        missing.append(gate_load.get("missing_file") or gate_load.get("path") or "gate")

    return {
        "phase": "AE12.5_RUNTIME_OBSERVABILITY_UI_FINAL_REPORTING",
        "read_only": True,
        "latest_ae12_output_root": maturation_root,
        "census_root": census_root,
        "quality_root": quality_root,
        "forward_evidence_gate": gate.get("status") or summary.get("readiness_gate", {}).get("status"),
        "needs_persistence_fix": bool(
            gate.get("needs_persistence_fix", summary.get("readiness_gate", {}).get("needs_persistence_fix", False))
        ),
        "live_ready": False,
        "live_trading_ready": False,
        "profitability_proven": False,
        "real_wallet_connected": False,
        "can_proceed_to_ui_final_report": bool(gate.get("can_proceed_to_ui_final_report", False)),
        "wallet_safety_status": gate.get("wallet_safety_status"),
        "evidence_row_count": gate.get("evidence_row_count") or summary.get("candidate_evidence_row_count"),
        "notes": list(gate.get("notes") or []),
        "missing_files": missing,
        "file_status": {
            "summary": summary_load.get("status"),
            "gate": gate_load.get("status"),
        },
        "cache": cache_meta,
        "disclaimers": list(SAFETY_DISCLAIMERS),
        "labels": {
            "mode": "paper/demo/exploration",
            "research_only": True,
            "not_live_approved": True,
            "not_profitability_proven": True,
        },
    }


def build_runtime_collection_payload(
    *,
    census: dict[str, Any] | None,
    census_load: dict[str, Any],
    summary: dict[str, Any] | None,
) -> dict[str, Any]:
    census = census or {}
    summary = summary or {}
    sqlite_health = census.get("sqlite_health") or {}
    health_rows = census.get("health_rows") or []
    tables = {t.get("table_name"): t for t in (census.get("top_sqlite_tables_by_rows") or []) if t.get("table_name")}

    def _health(component: str) -> dict[str, Any] | None:
        for row in health_rows:
            if row.get("component") == component:
                return row
        return None

    snaps = _health("market_snapshots") or tables.get("market_snapshots") or {}
    sentiment = _health("sentiment_rss") or tables.get("sentiment_records") or {}
    paper = tables.get("paper_trades") or {}

    db_collection_ts = sqlite_health.get("trader_db_last_write_utc")
    # AE11 loop latest from census recent artifacts / health if present
    ae11_ts = None
    for art in census.get("top_recent_artifacts") or []:
        rel = str(art.get("relative_path") or "")
        if "ae11" in rel.lower() and "checkpoint" in rel.lower():
            ae11_ts = art.get("last_write_time_utc")
            break
    if ae11_ts is None:
        for art in census.get("top_recent_artifacts") or []:
            rel = str(art.get("relative_path") or "")
            if "ae11_opportunity" in rel.lower() or "runtime_paper_loop" in rel.lower():
                ae11_ts = art.get("last_write_time_utc")
                break

    ae11_older_warning = False
    if ae11_ts and db_collection_ts:
        ae11_older_warning = str(ae11_ts) < str(db_collection_ts)

    paper_demo_count = None
    src = summary.get("source_file_counts") or {}
    if src:
        paper_demo_count = {
            "paper_trades_files": src.get("paper_trades"),
            "paper_positions_files": src.get("paper_positions"),
            "opportunity_capture_files": src.get("opportunity_capture"),
            "trade_decision_files": src.get("trade_decisions"),
        }

    return {
        "status": census_load.get("status", "MISSING"),
        "census_root": census.get("audit_root"),
        "missing_file": census_load.get("missing_file"),
        "latest_runtime_collection_timestamp": db_collection_ts,
        "market_snapshot_count": snaps.get("row_count") or snaps.get("rows_or_count"),
        "market_snapshot_latest_timestamp": snaps.get("latest_timestamp_value") or snaps.get("latest_timestamp"),
        "sentiment_rss_count": sentiment.get("row_count") or sentiment.get("rows_or_count"),
        "sentiment_rss_latest_timestamp": sentiment.get("latest_timestamp_value") or sentiment.get("latest_timestamp"),
        "paper_demo_evidence": paper_demo_count
        or {
            "sqlite_paper_trades_rows": paper.get("row_count"),
            "sqlite_paper_trades_latest": paper.get("latest_timestamp_value"),
        },
        "ae11_loop_timestamp": ae11_ts,
        "ae11_loop_older_than_db_collection": ae11_older_warning,
        "warning": (
            "AE11 loop timestamp is older than DB collection timestamp - "
            "price freshness and horizon maturity must not be confused."
            if ae11_older_warning
            else None
        ),
        "read_only": True,
        "labels": {"mode": "paper/demo/exploration", "not_live_approved": True},
    }


def build_forward_evidence_payload(
    *,
    summary: dict[str, Any] | None,
    summary_load: dict[str, Any],
    missing_warnings_sample: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if summary_load.get("status") != "OK" or not summary:
        return {
            "status": summary_load.get("status", "MISSING"),
            "missing_file": summary_load.get("missing_file") or summary_load.get("path"),
            "error": summary_load.get("error"),
            "read_only": True,
        }

    missing_snap_warnings = 0
    warning_breakdown: dict[str, int] = {}
    if missing_warnings_sample and missing_warnings_sample.get("rows"):
        for row in missing_warnings_sample["rows"]:
            kind = normalize_missing_warning_code(row)
            warning_breakdown[kind] = warning_breakdown.get(kind, 0) + 1
            if "SNAPSHOT" in kind or "MARKET" in kind:
                missing_snap_warnings += 1

    # Prefer summary totals when available
    total_missing = summary.get("missing_data_warning_count")
    hm = summary.get("horizon_maturity") or {}

    return {
        "status": "OK",
        "source_file": summary_load.get("path"),
        "output_root": summary.get("output_root"),
        "run_id": summary.get("run_id"),
        "created_at_utc": summary.get("created_at_utc"),
        "candidate_evidence_row_count": summary.get("candidate_evidence_row_count"),
        "matured_outcome_row_count": summary.get("matured_outcome_row_count"),
        "horizon_maturity": hm,
        "matured_by_horizon": {h: (v or {}).get("matured_count") for h, v in hm.items()},
        "not_matured_by_horizon": {h: (v or {}).get("not_matured_count") for h, v in hm.items()},
        "no_lookahead_by_horizon": {h: (v or {}).get("no_lookahead_ok_count") for h, v in hm.items()},
        "missing_data_warning_count": total_missing,
        "missing_snapshot_warnings_note": (
            "Missing market-snapshot warnings are reported in AE12 missing-data audits; "
            "price freshness differs from horizon maturity."
        ),
        "missing_warning_sample_breakdown": warning_breakdown or None,
        "reason_recovery_counts": summary.get("reason_recovery_counts"),
        "missed_winner_count": summary.get("missed_winner_count"),
        "missed_winners_by_horizon": summary.get("missed_winners_by_horizon"),
        "market_snapshots_available": summary.get("market_snapshots_available"),
        "price_freshness_vs_horizon_maturity": {
            "price_freshness": "Whether the entry price was fresh/stale at decision time (runtime/gate concern).",
            "horizon_maturity": "Whether enough time has elapsed after first_seen for a horizon outcome to be computed.",
            "distinction": "A matured horizon can still have stale entry price; price freshness differs from horizon maturity.",
        },
        "no_lookahead_status": "NO_LOOKAHEAD_OK counts are per-horizon in horizon_maturity; zero-fill is not used for missing snapshots.",
        "known_limitations": summary.get("known_limitations") or [],
        "warning": "forward returns are outcome labels only",
        "read_only": True,
        "labels": {
            "research_only": True,
            "not_live_approved": True,
            "not_profitability_proven": True,
        },
    }


def build_missed_winners_payload(
    *,
    summary: dict[str, Any] | None,
    missed_load: dict[str, Any],
    limit: int,
) -> dict[str, Any]:
    summary = summary or {}
    if missed_load.get("status") != "OK":
        return {
            "status": missed_load.get("status", "MISSING"),
            "missing_file": missed_load.get("missing_file") or missed_load.get("path"),
            "error": missed_load.get("error"),
            "total_missed_winners": summary.get("missed_winner_count"),
            "missed_winners_by_horizon": summary.get("missed_winners_by_horizon"),
            "rows": [],
            "warning": "missed winners are outcome labels only",
            "read_only": True,
        }

    data = missed_load.get("data") or {}
    rows_raw = data.get("rows") if isinstance(data, dict) else data
    rows: list[dict[str, Any]] = []
    for r in rows_raw or []:
        reason = r.get("reason_not_traded") or r.get("rejection_reason") or r.get("reason_for_no_trade") or ""
        rows.append(
            {
                "evidence_row_id": r.get("evidence_row_id"),
                "candidate_id": r.get("candidate_id"),
                "decision_id": r.get("decision_id"),
                "pair_address": r.get("pair_address"),
                "first_seen_timestamp": r.get("first_seen_timestamp"),
                "horizon": r.get("horizon"),
                "max_return": _safe_float(r.get("max_return")),
                "threshold": _safe_float(r.get("threshold")),
                "was_traded": str(r.get("was_traded", "")).lower() in {"true", "1", "yes"},
                "strict_shadow_decision": r.get("strict_shadow_decision"),
                "exploration_decision": r.get("exploration_decision"),
                "reason_for_no_trade": reason,
                "reason_not_traded": reason,
                "reason_recovery_status": r.get("reason_recovery_status"),
                "price_freshness_status": r.get("price_freshness_status"),
                "no_lookahead_status": r.get("no_lookahead_status"),
            }
        )

    return {
        "status": "OK",
        "source_file": missed_load.get("path"),
        "total_missed_winners": summary.get("missed_winner_count"),
        "missed_winners_by_horizon": summary.get("missed_winners_by_horizon"),
        "limit": limit,
        "rows": rows[:limit],
        "truncated": bool(isinstance(data, dict) and data.get("truncated")),
        "total_rows_estimate": data.get("total_rows_estimate") if isinstance(data, dict) else None,
        "warning": "missed winners are outcome labels only - they do not prove the strategy would have profited",
        "read_only": True,
        "labels": {
            "research_only": True,
            "not_profitability_proven": True,
            "not_live_approved": True,
        },
    }


def build_trade_vs_no_trade_payload(
    *,
    summary: dict[str, Any] | None,
    csv_load: dict[str, Any],
) -> dict[str, Any]:
    summary = summary or {}
    rows_from_summary = summary.get("trade_vs_no_trade")
    interpretations = summary.get("trade_vs_no_trade_interpretations") or {}

    rows: list[dict[str, Any]] = []
    if csv_load.get("status") == "OK" and csv_load.get("data"):
        for r in csv_load["data"]:
            rows.append(
                {
                    "horizon": r.get("horizon"),
                    "traded_count": _safe_int(r.get("traded_count")),
                    "not_traded_count": _safe_int(r.get("not_traded_count")),
                    "median_forward_return_traded": _safe_float(r.get("median_forward_return_traded")),
                    "median_forward_return_not_traded": _safe_float(r.get("median_forward_return_not_traded")),
                    "max_forward_return_traded": _safe_float(r.get("max_forward_return_traded")),
                    "max_forward_return_not_traded": _safe_float(r.get("max_forward_return_not_traded")),
                    "missed_winner_count": _safe_int(r.get("missed_winner_count")),
                    "sample_size_matured": _safe_int(r.get("sample_size_matured")),
                    "interpretation_status": r.get("interpretation_status")
                    or interpretations.get(r.get("horizon", "")),
                }
            )
    elif rows_from_summary:
        rows = list(rows_from_summary)

    traded = rows[0].get("traded_count") if rows else None
    not_traded = rows[0].get("not_traded_count") if rows else None

    status = "OK" if rows else csv_load.get("status", "MISSING")
    return {
        "status": status,
        "source_file": csv_load.get("path") if csv_load.get("status") == "OK" else None,
        "missing_file": csv_load.get("missing_file") if status == "MISSING" else None,
        "traded_count": traded,
        "not_traded_count": not_traded,
        "by_horizon": rows,
        "interpretations": interpretations,
        "warning": "outcome labels only, not profitability proof",
        "explicit_warning": "forward returns are outcome labels only - not a profitability claim",
        "read_only": True,
        "labels": {"not_profitability_proven": True, "research_only": True},
    }


def build_strict_vs_exploration_payload(
    *,
    summary: dict[str, Any] | None,
    csv_load: dict[str, Any],
) -> dict[str, Any]:
    summary = summary or {}
    sve = summary.get("strict_vs_exploration") or {}
    top_blockers = list(sve.get("top_strict_blockers") or [])

    horizon_rows: list[dict[str, Any]] = []
    if csv_load.get("status") == "OK" and csv_load.get("data"):
        for r in csv_load["data"]:
            if r.get("horizon"):
                horizon_rows.append(
                    {
                        "horizon": r.get("horizon"),
                        "strict_approved_median_return": _safe_float(r.get("strict_approved_median_return")),
                        "exploration_only_median_return": _safe_float(r.get("exploration_only_median_return")),
                        "strict_approved_n": _safe_int(r.get("strict_approved_n")),
                        "exploration_only_n": _safe_int(r.get("exploration_only_n")),
                    }
                )
    elif sve.get("return_comparison_by_horizon"):
        horizon_rows = list(sve["return_comparison_by_horizon"])

    strict_approved = sve.get("strict_approved")
    if strict_approved is None and csv_load.get("data"):
        for r in csv_load["data"]:
            if r.get("strict_approved") not in (None, ""):
                strict_approved = _safe_int(r.get("strict_approved"))
                break

    return {
        "status": "OK" if sve or csv_load.get("status") == "OK" else csv_load.get("status", "MISSING"),
        "source_file": csv_load.get("path") if csv_load.get("status") == "OK" else None,
        "missing_file": csv_load.get("missing_file"),
        "total_candidates": sve.get("total_candidates"),
        "strict_approved": strict_approved if strict_approved is not None else 0,
        "strict_blocked": sve.get("strict_blocked"),
        "exploration_only_trades": sve.get("exploration_only_trades") or sve.get("exploration_traded"),
        "strict_approved_trades": sve.get("strict_approved_trades", 0),
        "top_blockers": top_blockers,
        "return_comparison_by_horizon": horizon_rows,
        "warning": "strict policy currently approves zero candidates",
        "explicit_note": "strict policy approved zero candidates in this AE12 evidence set",
        "read_only": True,
        "labels": {
            "paper_demo_exploration": True,
            "not_live_approved": True,
            "not_profitability_proven": True,
        },
    }


def build_qwen_linkage_payload(
    *,
    summary: dict[str, Any] | None,
    sample_load: dict[str, Any],
) -> dict[str, Any]:
    summary = summary or {}
    counts = summary.get("qwen_linkage_counts") or {}
    sample = summary.get("qwen_linkage_sanity_sample") or []
    if not sample and sample_load.get("status") == "OK":
        data = sample_load.get("data") or {}
        raw = data.get("rows") if isinstance(data, dict) else data
        sample = list(raw or [])[:50]

    sample, sample_filter_note = filter_qwen_sample_rows(sample)

    ollama_statuses: dict[str, int] = {}
    authority_statuses: dict[str, int] = {}
    for row in sample:
        o = row.get("ollama_linkage_status") or "UNKNOWN"
        a = row.get("llm_trade_authority_status") or "UNKNOWN"
        ollama_statuses[o] = ollama_statuses.get(o, 0) + 1
        authority_statuses[a] = authority_statuses.get(a, 0) + 1

    return {
        "status": "OK" if counts else sample_load.get("status", "MISSING"),
        "source_summary_file": summary.get("output_root"),
        "sample_source_file": sample_load.get("path") if sample_load.get("status") == "OK" else None,
        "missing_file": sample_load.get("missing_file") if not counts else None,
        "ROW_LINKED_AE9_RECORD": counts.get("ROW_LINKED_AE9_RECORD"),
        "MENTION_ONLY": counts.get("MENTION_ONLY"),
        "linkage_counts": counts,
        "ollama_status": "ABSENT" if (not ollama_statuses or set(ollama_statuses) <= {"ABSENT"}) else ollama_statuses,
        "ollama_sample_breakdown": ollama_statuses,
        "NO_TRADE_AUTHORITY": True,
        "llm_trade_authority_status": "NO_TRADE_AUTHORITY",
        "authority_sample_breakdown": authority_statuses,
        "qwen_trade_authority": False,
        "sample_rows": sample[:10],
        "sample_filter_note": sample_filter_note,
        "warning": "Qwen/Gemini/Ollama do not create entries and are not trade authority",
        "explicit_note": "Qwen/Gemini/Ollama are audit/explanation layers, not trade authority",
        "read_only": True,
        "labels": {"research_only": True, "not_live_approved": True},
    }


def build_safety_payload(
    *,
    wallet: dict[str, Any] | None,
    wallet_load: dict[str, Any],
    gate: dict[str, Any] | None,
) -> dict[str, Any]:
    wallet = wallet or {}
    gate = gate or {}
    status = wallet_load.get("status", "MISSING")
    if status != "OK" and not wallet:
        # Fall back to nested summary wallet_safety if present via gate path
        return {
            "status": status,
            "missing_file": wallet_load.get("missing_file") or wallet_load.get("path"),
            "wallet_configured": False,
            "private_key_accessed": False,
            "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
            "live_trading_approval": "NO",
            "live_trading_ready": False,
            "profitability_proven": False,
            "real_wallet_connected": False,
            "read_only": True,
        }

    return {
        "status": "OK" if wallet else status,
        "source_file": wallet_load.get("path"),
        "audit_status": wallet.get("audit_status") or gate.get("wallet_safety_status"),
        "wallet_configured": bool(wallet.get("wallet_configured", False)),
        "private_key_accessed": bool(wallet.get("private_key_accessed", False)),
        "real_transaction_signed": bool(wallet.get("real_transaction_signed", False)),
        "live_submission_status": wallet.get("live_submission_status") or "NOT_SUBMITTED_NO_WALLET",
        "live_trading_approval": "NO",
        "live_trading_ready": False,
        "profitability_proven": False,
        "real_wallet_connected": False,
        "qwen_trade_authority": False,
        "no_real_wallet_flag": bool(wallet.get("no_real_wallet_flag", True)),
        "note": wallet.get("note"),
        "gate_status": gate.get("status"),
        "read_only": True,
        "labels": {
            "paper_demo_exploration": True,
            "not_live_approved": True,
            "not_profitability_proven": True,
        },
    }


def build_final_report_summary(
    *,
    status: dict[str, Any],
    forward: dict[str, Any],
    trade_vs: dict[str, Any],
    strict: dict[str, Any],
    qwen: dict[str, Any],
    safety: dict[str, Any],
    missed: dict[str, Any],
    runtime: dict[str, Any],
) -> dict[str, Any]:
    return {
        "phase": "AE12.5_FINAL_REPORT_SUMMARY",
        "read_only": True,
        "status": status,
        "runtime_collection": runtime,
        "forward_evidence": {
            "candidate_evidence_row_count": forward.get("candidate_evidence_row_count"),
            "horizon_maturity": forward.get("horizon_maturity"),
            "missed_winner_count": forward.get("missed_winner_count"),
            "missing_data_warning_count": forward.get("missing_data_warning_count"),
        },
        "missed_winners": {
            "total": missed.get("total_missed_winners"),
            "by_horizon": missed.get("missed_winners_by_horizon"),
        },
        "trade_vs_no_trade": {
            "traded_count": trade_vs.get("traded_count"),
            "not_traded_count": trade_vs.get("not_traded_count"),
            "interpretations": trade_vs.get("interpretations"),
        },
        "strict_vs_exploration": {
            "strict_approved": strict.get("strict_approved"),
            "strict_blocked": strict.get("strict_blocked"),
            "exploration_only_trades": strict.get("exploration_only_trades"),
            "top_blockers": strict.get("top_blockers"),
        },
        "qwen_linkage": {
            "ROW_LINKED_AE9_RECORD": qwen.get("ROW_LINKED_AE9_RECORD"),
            "MENTION_ONLY": qwen.get("MENTION_ONLY"),
            "NO_TRADE_AUTHORITY": qwen.get("NO_TRADE_AUTHORITY"),
            "ollama_status": qwen.get("ollama_status"),
        },
        "safety": {
            "wallet_configured": safety.get("wallet_configured"),
            "private_key_accessed": safety.get("private_key_accessed"),
            "live_submission_status": safety.get("live_submission_status"),
            "live_trading_approval": safety.get("live_trading_approval"),
            "live_ready": False,
            "profitability_proven": False,
        },
        "disclaimers": list(SAFETY_DISCLAIMERS),
        "labels": {
            "research_only": True,
            "not_live_approved": True,
            "not_profitability_proven": True,
            "mode": "paper/demo/exploration",
        },
    }
