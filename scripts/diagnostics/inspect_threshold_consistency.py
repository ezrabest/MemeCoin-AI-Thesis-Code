#!/usr/bin/env python3
"""Diagnostic 5 — threshold consistency across settings, code, and UI."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.diagnostics._common import (
    DiagnosticReport,
    PROJECT_ROOT,
    fetch_json_url,
    load_settings_file,
)


def _classify_mismatch(name: str, values: dict[str, float | None]) -> str:
    nums = [v for v in values.values() if v is not None]
    if len(set(round(v, 6) for v in nums)) <= 1:
        return "aligned"
    whale_keys = {"canonical_min_whale_score", "detect_whale_alert.min_whale_score", "live_min_whale_score_effective"}
    if whale_keys.intersection(values.keys()) and max(nums) - min(nums) >= 0.15:
        return "dangerous_mismatch"
    if "ui_" in name or "display" in name:
        return "display_only_mismatch"
    return "suspicious_mismatch"


def _parse_system_config_js(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    text = path.read_text(encoding="utf-8")
    out: dict[str, Any] = {}
    for key in ("minWhaleScore", "minLiquidity", "rfProbabilityThreshold", "economicGateEnabled"):
        m = re.search(rf"{key}\s*[:=]\s*([0-9.]+|true|false)", text, re.I)
        if m:
            val = m.group(1)
            out[key] = float(val) if re.match(r"^[0-9.]+$", val) else val.lower() == "true"
    return out


def run(*, output_dir: Path, api_base: str = "http://127.0.0.1:8000") -> DiagnosticReport:
    from app.engine import (
        SIGNAL_BUY_LIQUIDITY_USD,
        SIGNAL_BUY_PROB_THRESHOLD,
        SIGNAL_BUY_WHALE_THRESHOLD,
        SIGNAL_WATCH_PROB_THRESHOLD,
        SIGNAL_WATCH_WHALE_THRESHOLD,
        WHALE_ALERT_MIN_VOLUME_24H,
        WHALE_ALERT_MIN_WHALE_SCORE,
        get_alert_thresholds,
        get_signal_thresholds,
    )
    from app.observability.effective_settings import get_effective_settings

    report = DiagnosticReport("inspect_threshold_consistency", output_dir)
    settings_file = load_settings_file()
    eff = get_effective_settings(settings_file if settings_file else None)
    api_payload = fetch_json_url(f"{api_base}/api/settings/effective")

    hidden = eff.hidden_thresholds
    canonical = eff.canonical

    thresholds = {
        "canonical_min_whale_score": float(canonical.get("min_whale_score", 0.30)),
        "detect_whale_alert.min_whale_score": float(get_alert_thresholds()["min_whale_score"]),
        "detect_whale_alert.min_volume_24h": float(get_alert_thresholds()["min_volume_24h"]),
        "generate_signal.watch_whale_score_threshold": float(get_signal_thresholds()["watch_whale_score_threshold"]),
        "generate_signal.buy_whale_score_threshold": float(get_signal_thresholds()["buy_whale_score_threshold"]),
        "generate_signal.buy_prob_threshold": float(get_signal_thresholds()["buy_prob_threshold"]),
        "generate_signal.buy_liquidity_usd": float(get_signal_thresholds()["buy_liquidity_usd"]),
        "live_min_whale_score_effective": float(hidden["live_scan_gates"]["min_whale_score_effective"]),
        "live_min_liquidity_usd": float(hidden["live_scan_gates"]["min_liquidity_usd_effective"]),
        "rf_probability_threshold": float(canonical.get("rf_probability_threshold", 0.70)),
        "max_model_artifact_age_hours": float(canonical.get("max_model_artifact_age_hours", 168)),
        "max_slippage_pct": float(canonical.get("max_slippage_pct", 0.015)),
        "max_price_drift_from_model_pct": float(canonical.get("max_price_drift_from_model_pct", 0.01)),
        "max_market_snapshot_age_seconds": float(canonical.get("max_market_snapshot_age_seconds", 300)),
        "economic_gate_enabled": bool(canonical.get("economic_gate_enabled", False)),
        "paper_trading_enabled": bool(canonical.get("paper_trading_enabled", True)),
    }

    ui = _parse_system_config_js(PROJECT_ROOT / "static" / "system_config.js")
    if ui:
        thresholds["ui_minWhaleScore"] = ui.get("minWhaleScore")
        thresholds["ui_rfProbabilityThreshold"] = ui.get("rfProbabilityThreshold")

    mismatches = []
    groups = {
        "whale_score_chain": {
            "canonical_min_whale_score": thresholds["canonical_min_whale_score"],
            "detect_whale_alert.min_whale_score": thresholds["detect_whale_alert.min_whale_score"],
            "generate_signal.watch_whale_score_threshold": thresholds["generate_signal.watch_whale_score_threshold"],
            "generate_signal.buy_whale_score_threshold": thresholds["generate_signal.buy_whale_score_threshold"],
            "live_min_whale_score_effective": thresholds["live_min_whale_score_effective"],
        },
    }
    for name, vals in groups.items():
        kind = _classify_mismatch(name, vals)
        if kind != "aligned":
            mismatches.append({"group": name, "classification": kind, "values": vals})

    api_canonical = (api_payload or {}).get("canonical") if api_payload else None
    if api_payload is None:
        report.add_limitation("Backend API unavailable — used direct effective settings builder")
    elif api_canonical and float(api_canonical.get("min_whale_score", 0)) != thresholds["canonical_min_whale_score"]:
        mismatches.append({
            "group": "api_vs_local",
            "classification": "suspicious_mismatch",
            "values": {
                "api": api_canonical.get("min_whale_score"),
                "local": thresholds["canonical_min_whale_score"],
            },
        })

    status = "PASS"
    if any(m["classification"] == "dangerous_mismatch" for m in mismatches):
        status = "WARN"
    if len(mismatches) >= 2:
        status = "WARN"

    report.set_status(status)
    report.data.update({
        "thresholds": thresholds,
        "hidden_thresholds_from_effective_settings": hidden,
        "settings_sources": eff.sources,
        "mismatches": mismatches,
        "api_available": api_payload is not None,
        "code_constants": {
            "WHALE_ALERT_MIN_WHALE_SCORE": WHALE_ALERT_MIN_WHALE_SCORE,
            "WHALE_ALERT_MIN_VOLUME_24H": WHALE_ALERT_MIN_VOLUME_24H,
            "SIGNAL_BUY_WHALE_THRESHOLD": SIGNAL_BUY_WHALE_THRESHOLD,
            "SIGNAL_WATCH_WHALE_THRESHOLD": SIGNAL_WATCH_WHALE_THRESHOLD,
        },
        "ui_detected": ui,
    })
    report.write_json("threshold_consistency.json")
    mismatch_lines = (
        [f"- **{m['group']}** ({m['classification']}): {m['values']}" for m in mismatches]
        if mismatches
        else ["- None"]
    )
    report.write_md([
        "## Thresholds",
        *[f"- `{k}` = {v}" for k, v in thresholds.items()],
        "",
        "## Mismatches",
        *mismatch_lines,
    ], "threshold_consistency.md")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    report = run(output_dir=args.output_dir, api_base=args.api_base)
    print(f"Status: {report.data['status']}")
    return 0 if report.data["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
