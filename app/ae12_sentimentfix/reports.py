"""Report writers for AE12-SentimentFix audit outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def ensure_dirs(root: Path) -> dict[str, Path]:
    paths = {
        "root": root,
        "reports": root / "reports",
        "data": root / "data",
        "audits": root / "audits",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys: list[str] = []
        seen: set[str] = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys or ["empty"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def render_upload_txt(summary: dict[str, Any], gate: dict[str, Any]) -> str:
    lines = [
        "AE12-SentimentFix Audit (Dual-Axis Taxonomy Repair)",
        "NOTE: This is AE12-SentimentFix, not AE12.6.",
        f"created_at_utc: {summary.get('created_at_utc')}",
        f"output_root: {summary.get('output_root')}",
        f"gate_status: {gate.get('status')}",
        f"prior_gate_status: {gate.get('prior_gate_status')}",
        "",
        "Safety:",
        "- live_trading_ready=false",
        "- profitability_proven=false",
        "- qwen_trade_authority=false",
        "- historical_data_mutated=false",
        "",
        "Dual-axis:",
        "- semantic_signal_family is independent of trading_opportunity_state",
        "- legacy_cluster_label preserved for audit only",
        "- missing semantic -> UNKNOWN (never OPPORTUNISTIC)",
        "",
        f"semantic_unknown_share: {gate.get('semantic_unknown_share')}",
        f"semantic_unknown_threshold: {gate.get('semantic_unknown_threshold')}",
        f"dual_axis_mapper_available: {gate.get('dual_axis_mapper_available')}",
        f"runtime_future_fields_added: {gate.get('runtime_future_fields_added')}",
        f"sticky_cluster_still_authoritative: {gate.get('sticky_cluster_still_authoritative')}",
        f"sticky_cluster_soft_expiry_plan_created: {gate.get('sticky_cluster_soft_expiry_plan_created')}",
        f"default_fallback_fixed: {gate.get('default_fallback_fixed')}",
        f"semantic_linkage_gap_found: {gate.get('semantic_linkage_gap_found')}",
        f"legacy_cluster_label_preserved: {gate.get('legacy_cluster_label_preserved')}",
        "",
        "recommendation:",
        str(gate.get("recommendation") or ""),
        "",
        "limitations:",
    ]
    for lim in gate.get("limitations") or []:
        lines.append(f"- {lim}")
    return "\n".join(lines) + "\n"
