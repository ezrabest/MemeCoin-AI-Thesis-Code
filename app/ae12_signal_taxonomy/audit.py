"""Run AE12 signal taxonomy audit (read-only vs trader.db / AE12 artifacts)."""

from __future__ import annotations

import csv
import json
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .axes import (
    classify_row_axes,
    empty_semantic_counts,
    empty_trading_counts,
    scan_code_for_dangerous_fallbacks,
)

CODE_PATHS_TO_SCAN = (
    "app/analytics/features.py",
    "app/models/predictor.py",
    "app/observability/sentiment_cluster_audit.py",
    "app/observability/candidate.py",
    "app/observability/actionability.py",
    "app/live.py",
    "app/database.py",
    "static/index.html",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _iter_jsonl(path: Path, *, max_rows: int) -> Iterable[dict[str, Any]]:
    if not path.is_file():
        return
    count = 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if count >= max_rows:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj
                count += 1


def _latest_glob(directory: Path, pattern: str) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(directory.glob(pattern), key=lambda p: p.name, reverse=True)


def _day_key(ts: Any) -> str:
    if not ts:
        return "UNKNOWN_DAY"
    s = str(ts)
    return s[:10] if len(s) >= 10 else s


def _sqlite_readonly_counts(db_path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {"status": "MISSING", "path": str(db_path)}
    if not db_path.is_file():
        return out
    try:
        uri = f"file:{db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        try:
            def count(table: str) -> int | None:
                try:
                    return int(conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"])
                except sqlite3.Error:
                    return None

            def latest(table: str, col: str) -> str | None:
                try:
                    row = conn.execute(f"SELECT {col} AS t FROM {table} ORDER BY {col} DESC LIMIT 1").fetchone()
                    return str(row["t"]) if row and row["t"] is not None else None
                except sqlite3.Error:
                    return None

            # Social marker presence in sentiment text if columns exist
            social_marker_rows = None
            try:
                cols = {r["name"] for r in conn.execute("PRAGMA table_info(sentiment_records)").fetchall()}
                text_col = None
                for c in ("title", "text", "body", "content", "summary", "source"):
                    if c in cols:
                        text_col = c
                        break
                if text_col:
                    social_marker_rows = int(
                        conn.execute(
                            f"""
                            SELECT COUNT(*) AS c FROM sentiment_records
                            WHERE lower(COALESCE({text_col}, '')) LIKE '%social%'
                               OR lower(COALESCE({text_col}, '')) LIKE '%twitter%'
                               OR lower(COALESCE({text_col}, '')) LIKE '%telegram%'
                               OR lower(COALESCE({text_col}, '')) LIKE '%reddit%'
                               OR lower(COALESCE({text_col}, '')) LIKE '%community%'
                            """
                        ).fetchone()["c"]
                    )
            except sqlite3.Error:
                social_marker_rows = None

            out.update(
                {
                    "status": "OK",
                    "sentiment_records_count": count("sentiment_records"),
                    "sentiment_records_latest": latest("sentiment_records", "timestamp"),
                    "signals_count": count("signals"),
                    "market_snapshots_count": count("market_snapshots"),
                    "whale_alerts_count": count("whale_alerts"),
                    "coins_count": count("coins"),
                    "pipeline_audit_count": count("pipeline_audit"),
                    "sentiment_social_marker_rows": social_marker_rows,
                }
            )
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        out["status"] = "ERROR"
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _accumulate_source(
    source: str,
    rows: Iterable[dict[str, Any]],
    *,
    by_source: dict[str, dict[str, Any]],
    by_time: dict[str, Counter],
    sample_rows: list[dict[str, Any]],
    marker_rows: list[dict[str, Any]],
    sticky_pairs: dict[str, list[str]],
) -> int:
    n = 0
    if source not in by_source:
        by_source[source] = {
            "source": source,
            "rows_sampled": 0,
            **{f"semantic_{k}": 0 for k in empty_semantic_counts()},
            **{f"trading_{k}": 0 for k in empty_trading_counts()},
        }
    bucket = by_source[source]
    for row in rows:
        axes = classify_row_axes(row)
        sem = axes["semantic_signal_family"]
        trad = axes["trading_opportunity_state"]
        bucket["rows_sampled"] += 1
        bucket[f"semantic_{sem}"] = int(bucket.get(f"semantic_{sem}", 0)) + 1
        bucket[f"trading_{trad}"] = int(bucket.get(f"trading_{trad}", 0)) + 1

        day = _day_key(
            row.get("first_seen_timestamp")
            or row.get("timestamp")
            or row.get("created_at_utc")
            or row.get("decision_timestamp")
        )
        by_time[day][f"semantic_{sem}"] += 1
        by_time[day][f"trading_{trad}"] += 1
        by_time[day]["total"] += 1

        pair = str(row.get("pair_address") or row.get("token_contract_address") or "").strip()
        if pair:
            sticky_pairs.setdefault(pair, []).append(f"{sem}|{trad}|{row.get('cluster_label')}")

        if len(sample_rows) < 80:
            sample_rows.append(
                {
                    "source": source,
                    "pair_address": pair,
                    "candidate_id": row.get("candidate_id") or row.get("decision_id"),
                    "cluster_label": row.get("cluster_label"),
                    "semantic_signal_family": sem,
                    "trading_opportunity_state": trad,
                    "marker_hits": axes.get("marker_hits"),
                    "timestamp": row.get("first_seen_timestamp")
                    or row.get("timestamp")
                    or row.get("created_at_utc"),
                }
            )

        if axes.get("marker_hits") and len(marker_rows) < 200:
            # Marker evidence exists but semantic became UNCLASSIFIED/UNKNOWN while trading opportunistic
            marker_rows.append(
                {
                    "source": source,
                    "pair_address": pair,
                    "marker_hits": axes.get("marker_hits"),
                    "marker_primary_semantic": axes.get("marker_primary_semantic"),
                    "semantic_signal_family": sem,
                    "trading_opportunity_state": trad,
                    "cluster_label": row.get("cluster_label"),
                    "mismatch": (
                        "YES"
                        if axes.get("marker_primary_semantic")
                        and sem in {"UNKNOWN", "UNCLASSIFIED"}
                        and trad == "OPPORTUNISTIC"
                        else "NO"
                    ),
                }
            )
        n += 1
    return n


def run_signal_taxonomy_audit(
    *,
    project_root: Path,
    ae12_root: Path | None = None,
    max_rows_per_source: int = 5000,
    no_external_apis: bool = True,
) -> dict[str, Any]:
    """
    Read-only taxonomy audit. Does not mutate trader.db or AE12 maturation artifacts.
    Writes a new audit root under data/audits/ae12_signal_taxonomy_audit_<ts>/.
    """
    project_root = Path(project_root).resolve()
    if ae12_root is None:
        audits = project_root / "data" / "audits"
        mats = sorted(
            audits.glob("ae12_forward_evidence_maturation_*"),
            key=lambda p: p.name,
            reverse=True,
        )
        ae12_root = mats[0] if mats else None
    else:
        ae12_root = Path(ae12_root)
        if not ae12_root.is_absolute():
            ae12_root = (project_root / ae12_root).resolve()
        else:
            ae12_root = ae12_root.resolve()

    out_root = project_root / "data" / "audits" / f"ae12_signal_taxonomy_audit_{_ts_slug()}"
    reports = out_root / "reports"
    data_dir = out_root / "data"
    audits_dir = out_root / "audits"
    for d in (reports, data_dir, audits_dir):
        d.mkdir(parents=True, exist_ok=True)

    by_source: dict[str, dict[str, Any]] = {}
    by_time: dict[str, Counter] = defaultdict(Counter)
    sample_rows: list[dict[str, Any]] = []
    marker_rows: list[dict[str, Any]] = []
    sticky_pairs: dict[str, list[str]] = {}
    evidence_sources: list[str] = []

    # AE6
    ae6_files = _latest_glob(project_root / "data" / "decision_records", "ae6_decisions_*.jsonl")[:3]
    for path in ae6_files:
        evidence_sources.append(str(path.relative_to(project_root)))
        _accumulate_source(
            "ae6_decisions",
            _iter_jsonl(path, max_rows=max_rows_per_source),
            by_source=by_source,
            by_time=by_time,
            sample_rows=sample_rows,
            marker_rows=marker_rows,
            sticky_pairs=sticky_pairs,
        )

    # AE11 opportunity + trade decisions
    for label, pattern in (
        ("ae11_opportunity_capture", "ae11_opportunity_capture_*.jsonl"),
        ("ae11_trade_decisions", "ae11_trade_decisions_*.jsonl"),
    ):
        files = _latest_glob(project_root / "data" / "runtime_paper_loop", pattern)[:3]
        for path in files:
            evidence_sources.append(str(path.relative_to(project_root)))
            _accumulate_source(
                label,
                _iter_jsonl(path, max_rows=max_rows_per_source),
                by_source=by_source,
                by_time=by_time,
                sample_rows=sample_rows,
                marker_rows=marker_rows,
                sticky_pairs=sticky_pairs,
            )

    # AE12 candidate evidence CSV (sample)
    if ae12_root and (ae12_root / "data" / "ae12_candidate_evidence_rows.csv").is_file():
        path = ae12_root / "data" / "ae12_candidate_evidence_rows.csv"
        evidence_sources.append(str(path.relative_to(project_root)))
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for i, row in enumerate(reader):
                if i >= max_rows_per_source:
                    break
                rows.append(dict(row))
        _accumulate_source(
            "ae12_candidate_evidence",
            rows,
            by_source=by_source,
            by_time=by_time,
            sample_rows=sample_rows,
            marker_rows=marker_rows,
            sticky_pairs=sticky_pairs,
        )

    # Cluster registry (sticky pair/token labels)
    registry_path = project_root / "data" / "cluster_registry.json"
    registry_dist: dict[str, int] = {}
    sticky_registry_note = None
    if registry_path.is_file():
        evidence_sources.append(str(registry_path.relative_to(project_root)))
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            registry = {}
        labels = [str((v or {}).get("cluster_label") or "UNKNOWN") for v in registry.values()]
        registry_dist = dict(Counter(labels))
        sticky_registry_note = (
            "cluster_registry.json persists a single cluster_label per contract_address "
            "(assigned once; subsequent resolve_cluster_label returns cached value without re-evaluation)."
        )
        # Represent registry on both axes for audit visibility
        reg_rows = [
            {
                "pair_address": k,
                "cluster_label": (v or {}).get("cluster_label"),
                "timestamp": (v or {}).get("assigned_at"),
            }
            for k, v in list(registry.items())[:max_rows_per_source]
        ]
        _accumulate_source(
            "cluster_registry",
            reg_rows,
            by_source=by_source,
            by_time=by_time,
            sample_rows=sample_rows,
            marker_rows=marker_rows,
            sticky_pairs=sticky_pairs,
        )

    sqlite_info = _sqlite_readonly_counts(project_root / "data" / "trader.db")
    if sqlite_info.get("status") == "OK":
        evidence_sources.append("data/trader.db (read-only counts)")

    # Code fallback audit
    fallback_rows: list[dict[str, str]] = []
    code_paths_checked: list[str] = []
    for rel in CODE_PATHS_TO_SCAN:
        path = project_root / rel
        if not path.is_file():
            continue
        code_paths_checked.append(rel)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        fallback_rows.extend(scan_code_for_dangerous_fallbacks(text, rel))

    # Sticky opportunistic audit from repeated pair labels
    sticky_rows: list[dict[str, Any]] = []
    sticky_flag_found = False
    for pair, seq in list(sticky_pairs.items())[:5000]:
        if len(seq) < 2:
            continue
        uniq = sorted(set(seq))
        first = seq[0]
        later_same = all(x.split("|")[2] == first.split("|")[2] for x in seq if len(x.split("|")) > 2)
        # Sticky if cluster_label component never changes across repeats
        clusters = [x.split("|")[2] if len(x.split("|")) > 2 else "" for x in seq]
        non_empty = [c for c in clusters if c]
        if non_empty and len(set(non_empty)) == 1 and "OPPORTUNISTIC" in non_empty[0].upper():
            sticky_flag_found = True
            sticky_rows.append(
                {
                    "pair_address": pair,
                    "observations": len(seq),
                    "unique_axis_combos": len(uniq),
                    "sticky_cluster_label": non_empty[0],
                    "status": "STICKY_OPPORTUNISTIC_OBSERVED",
                }
            )
        elif later_same and non_empty and "OPPORTUNISTIC" in non_empty[0].upper() and len(seq) >= 3:
            sticky_flag_found = True
            sticky_rows.append(
                {
                    "pair_address": pair,
                    "observations": len(seq),
                    "unique_axis_combos": len(uniq),
                    "sticky_cluster_label": non_empty[0],
                    "status": "STICKY_OPPORTUNISTIC_OBSERVED",
                }
            )

    # Design-gap sticky from code (persist once)
    sticky_code_found = any(
        "persist_cluster" in r.get("line_snippet", "") or "get_persisted_cluster" in r.get("line_snippet", "")
        or "Persistent cluster" in r.get("line_snippet", "")
        for r in fallback_rows
    )
    # Explicit sticky design in features.py
    features_path = project_root / "app" / "analytics" / "features.py"
    if features_path.is_file():
        feat = features_path.read_text(encoding="utf-8", errors="replace")
        if "Persistent cluster label" in feat or "get_persisted_cluster" in feat:
            sticky_flag_found = True
            sticky_code_found = True
            sticky_rows.insert(
                0,
                {
                    "pair_address": "(code_path)",
                    "observations": "",
                    "unique_axis_combos": "",
                    "sticky_cluster_label": "OPPORTUNISTIC_SPECULATIVE_OR_CACHED",
                    "status": "STICKY_BY_DESIGN_IN_resolve_cluster_label",
                    "note": sticky_registry_note or "persist once per contract_address",
                },
            )

    # UI mapping bug: historical non-SOCIAL -> SPECULATIVE default (fixed in this pass if still present)
    ui_path = project_root / "static" / "index.html"
    ui_mapping_bug_found = False
    if ui_path.is_file():
        ui_text = ui_path.read_text(encoding="utf-8", errors="replace")
        # Old buggy pattern: anything not SOCIAL rendered as SPECULATIVE
        if 'isSocial ? "SOCIAL" : "SPECULATIVE"' in ui_text or "isSocial ? 'SOCIAL' : 'SPECULATIVE'" in ui_text:
            ui_mapping_bug_found = True
        # After fix, UNKNOWN/UNCLASSIFIED path should exist
        if "UNCLASSIFIED" not in ui_text and "UNKNOWN" not in ui_text.split("clusterPill")[1][:800]:
            # Only flag if clusterPill still binary-maps
            if "SPECULATIVE" in ui_text and "clusterPill" in ui_text:
                ui_mapping_bug_found = True

    conflated_axis_found = True  # historical design uses single cluster_label for social vs opportunistic
    default_fallback_bug_found = len(fallback_rows) > 0

    # Social linkage: sentiment exists but AE12/ae6 semantic SOCIAL share near zero
    total_sem = Counter()
    total_trad = Counter()
    for bucket in by_source.values():
        for k, v in bucket.items():
            if k.startswith("semantic_"):
                total_sem[k.replace("semantic_", "")] += int(v or 0)
            if k.startswith("trading_"):
                total_trad[k.replace("trading_", "")] += int(v or 0)

    social_rows = int(total_sem.get("SOCIAL", 0))
    opportunistic_rows = int(total_trad.get("OPPORTUNISTIC", 0))
    unknown_rows = int(total_sem.get("UNKNOWN", 0)) + int(total_sem.get("UNCLASSIFIED", 0))
    total_rows = sum(int(b.get("rows_sampled") or 0) for b in by_source.values()) or 1
    social_share = round(social_rows / total_rows, 6)
    opportunistic_share = round(opportunistic_rows / total_rows, 6)
    unknown_share = round(unknown_rows / total_rows, 6)

    sentiment_count = sqlite_info.get("sentiment_records_count") or 0
    social_linkage_bug_found = bool(
        sentiment_count and sentiment_count > 1000 and social_share < 0.01 and opportunistic_share > 0.2
    )

    # Decision gate status priority
    fix_applied: list[str] = []
    if ui_path.is_file():
        ui_text = ui_path.read_text(encoding="utf-8", errors="replace")
        if "UNCLASSIFIED" in ui_text and "clusterPill" in ui_text:
            fix_applied.append(
                "static/index.html clusterPill: missing/non-social labels map to UNKNOWN/UNCLASSIFIED "
                "instead of always SPECULATIVE/OPPORTUNISTIC"
            )
            ui_mapping_bug_found = False  # fixed in this pass

    statuses: list[str] = []
    if conflated_axis_found:
        statuses.append("FAIL_CONFLATED_TAXONOMY_AXIS")
    if sticky_flag_found:
        statuses.append("FAIL_STICKY_OPPORTUNISTIC_FLAG")
    if default_fallback_bug_found:
        statuses.append("FAIL_DEFAULT_FALLBACK_BUG")
    if social_linkage_bug_found:
        statuses.append("FAIL_SOCIAL_LINKAGE_BUG")
    if ui_mapping_bug_found:
        statuses.append("FAIL_UI_MAPPING_BUG")

    if statuses:
        # Primary gate: prefer conflated axis as design gap when present
        if "FAIL_CONFLATED_TAXONOMY_AXIS" in statuses:
            gate_status = "FAIL_CONFLATED_TAXONOMY_AXIS"
        else:
            gate_status = statuses[0]
    elif total_rows < 50:
        gate_status = "PASS_WITH_DATA_LIMITATION"
    else:
        gate_status = "NEEDS_MANUAL_REVIEW"

    limitations = [
        "semantic_signal_family and trading_opportunity_state were not historically separate schema fields.",
        "cluster_label conflates social vs opportunistic into one sticky binary axis.",
        "Audit classifies axes retrospectively from available fields/markers; not a retrain or LLM re-label.",
        "JSONL/CSV sampling may be capped by max_rows_per_source.",
        "No external APIs and no Qwen/Gemini/Ollama calls in this audit.",
        "Distribution must not be treated as profitability evidence.",
        "Do not claim live readiness.",
    ]

    recommendation = (
        "Do not treat opportunistic/social shares as reliable semantic taxonomy until dual-axis fields exist. "
        "Future work: introduce semantic_signal_family + trading_opportunity_state; stop defaulting missing "
        "classification to OPPORTUNISTIC_SPECULATIVE; re-evaluate or soft-expire sticky cluster_registry entries; "
        "link sentiment_records into decision/opportunity rows without granting LLM trade authority."
    )

    gate = {
        "gate_name": "ae12_social_vs_opportunistic_decision_gate",
        "status": gate_status,
        "all_status_flags": statuses,
        "social_rows_found": social_rows,
        "opportunistic_rows_found": opportunistic_rows,
        "unknown_rows_found": unknown_rows,
        "social_share": social_share,
        "opportunistic_share": opportunistic_share,
        "unknown_share": unknown_share,
        "semantic_signal_family_distribution": dict(total_sem),
        "trading_opportunity_state_distribution": dict(total_trad),
        "evidence_sources_checked": evidence_sources,
        "code_paths_checked": code_paths_checked,
        "sticky_flag_found": sticky_flag_found,
        "sticky_code_found": sticky_code_found,
        "conflated_axis_found": conflated_axis_found,
        "default_fallback_bug_found": default_fallback_bug_found,
        "social_linkage_bug_found": social_linkage_bug_found,
        "ui_mapping_bug_found": ui_mapping_bug_found,
        "fix_applied": fix_applied,
        "cluster_registry_distribution": registry_dist,
        "sqlite_readonly": sqlite_info,
        "ae12_root": str(ae12_root) if ae12_root else None,
        "limitations": limitations,
        "recommendation": recommendation,
        "live_trading_ready": False,
        "profitability_proven": False,
        "qwen_trade_authority": False,
        "created_at_utc": _utc_now(),
        "no_external_apis": no_external_apis,
    }

    # Time distribution CSV
    time_rows: list[dict[str, Any]] = []
    for day, ctr in sorted(by_time.items()):
        total = int(ctr.get("total") or 0) or 1
        s = int(ctr.get("semantic_SOCIAL", 0))
        o = int(ctr.get("trading_OPPORTUNISTIC", 0))
        u = int(ctr.get("semantic_UNKNOWN", 0)) + int(ctr.get("semantic_UNCLASSIFIED", 0))
        time_rows.append(
            {
                "day": day,
                "total": ctr.get("total", 0),
                "social_count": s,
                "opportunistic_count": o,
                "unknown_count": u,
                "social_share": round(s / total, 6),
                "opportunistic_share": round(o / total, 6),
                "unknown_share": round(u / total, 6),
            }
        )

    social_linkage = [
        {
            "metric": "sentiment_records_count",
            "value": sqlite_info.get("sentiment_records_count"),
            "note": "SQLite read-only",
        },
        {
            "metric": "sentiment_records_latest",
            "value": sqlite_info.get("sentiment_records_latest"),
            "note": "",
        },
        {
            "metric": "sentiment_social_marker_rows",
            "value": sqlite_info.get("sentiment_social_marker_rows"),
            "note": "Conservative LIKE markers on one text column if present",
        },
        {
            "metric": "semantic_SOCIAL_rows_across_sampled_sources",
            "value": social_rows,
            "note": "",
        },
        {
            "metric": "trading_OPPORTUNISTIC_rows_across_sampled_sources",
            "value": opportunistic_rows,
            "note": "",
        },
        {
            "metric": "social_share",
            "value": social_share,
            "note": "Not a profitability claim",
        },
        {
            "metric": "social_linkage_bug_found",
            "value": social_linkage_bug_found,
            "note": "Sentiment volume high but semantic SOCIAL share near zero",
        },
    ]

    summary = {
        "phase": "AE12.5_SIGNAL_TAXONOMY_AUDIT",
        "created_at_utc": _utc_now(),
        "output_root": str(out_root),
        "ae12_root": str(ae12_root) if ae12_root else None,
        "gate_status": gate_status,
        "gate": gate,
        "rows_sampled_total": total_rows,
        "by_source_keys": list(by_source.keys()),
        "fallback_pattern_hits": len(fallback_rows),
        "sticky_pairs_flagged": len([r for r in sticky_rows if r.get("pair_address") != "(code_path)"]),
        "fix_applied": fix_applied,
        "live_trading_ready": False,
        "profitability_proven": False,
        "known_limitations": limitations,
        "recommendation": recommendation,
    }

    # Write artifacts
    _write_json(reports / "ae12_signal_taxonomy_audit_summary.json", summary)
    _write_json(audits_dir / "ae12_social_vs_opportunistic_decision_gate.json", gate)
    _write_csv(data_dir / "ae12_signal_taxonomy_distribution_by_source.csv", list(by_source.values()))
    _write_csv(data_dir / "ae12_signal_taxonomy_distribution_by_time.csv", time_rows)
    _write_csv(data_dir / "ae12_social_linkage_audit.csv", social_linkage)
    _write_csv(data_dir / "ae12_category_fallback_code_audit.csv", fallback_rows)
    _write_csv(data_dir / "ae12_category_sample_rows.csv", sample_rows)
    _write_csv(data_dir / "ae12_sticky_opportunistic_audit.csv", sticky_rows)
    _write_csv(data_dir / "ae12_semantic_text_marker_audit.csv", marker_rows)

    upload = _render_upload_txt(summary, gate)
    (reports / "ae12_signal_taxonomy_audit_for_upload.txt").write_text(upload, encoding="utf-8")

    return summary


def _render_upload_txt(summary: dict[str, Any], gate: dict[str, Any]) -> str:
    lines = [
        "AE12.5 Signal Taxonomy Audit (Social vs Opportunistic / Dual-Axis)",
        f"created_at_utc: {summary.get('created_at_utc')}",
        f"output_root: {summary.get('output_root')}",
        f"gate_status: {gate.get('status')}",
        "",
        "IMPORTANT:",
        "- forward returns are outcome labels only",
        "- paper/demo exploration is not live-trading approval",
        "- Qwen/Gemini/Ollama are audit/explanation layers, not trade authority",
        "- live_trading_ready=false",
        "- profitability_proven=false",
        "",
        "Dual-axis model:",
        "- semantic_signal_family: SOCIAL/NEWS/ONCHAIN/PRICE_MOMENTUM/LIQUIDITY/WHALE/LLM_CONTEXT/UNKNOWN/UNCLASSIFIED",
        "- trading_opportunity_state: OPPORTUNISTIC/EXPLORATION/STRICT_BLOCKED/NO_TRADE/PAPER_TRADED/UNKNOWN",
        "- A row may be SOCIAL and OPPORTUNISTIC at the same time.",
        "",
        f"social_rows_found: {gate.get('social_rows_found')}",
        f"opportunistic_rows_found: {gate.get('opportunistic_rows_found')}",
        f"unknown_rows_found: {gate.get('unknown_rows_found')}",
        f"social_share: {gate.get('social_share')}",
        f"opportunistic_share: {gate.get('opportunistic_share')}",
        f"unknown_share: {gate.get('unknown_share')}",
        "",
        f"sticky_flag_found: {gate.get('sticky_flag_found')}",
        f"conflated_axis_found: {gate.get('conflated_axis_found')}",
        f"default_fallback_bug_found: {gate.get('default_fallback_bug_found')}",
        f"social_linkage_bug_found: {gate.get('social_linkage_bug_found')}",
        f"ui_mapping_bug_found: {gate.get('ui_mapping_bug_found')}",
        f"fix_applied: {gate.get('fix_applied')}",
        "",
        "semantic_signal_family_distribution:",
        json.dumps(gate.get("semantic_signal_family_distribution"), indent=2),
        "",
        "trading_opportunity_state_distribution:",
        json.dumps(gate.get("trading_opportunity_state_distribution"), indent=2),
        "",
        "recommendation:",
        str(gate.get("recommendation")),
        "",
        "limitations:",
    ]
    for lim in gate.get("limitations") or []:
        lines.append(f"- {lim}")
    return "\n".join(lines) + "\n"
