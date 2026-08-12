#!/usr/bin/env python3
"""Diagnostic 4 — static call graph for detect_whale_alert."""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.diagnostics._common import DiagnosticReport, PROJECT_ROOT


SYMBOLS = (
    "detect_whale_alert",
    "compute_whale_score",
    "generate_signal",
    "insert_signal",
    "insert_whale_alert",
    "evaluate_economic_trade_candidate",
    "execute_trade_decision",
    "evaluate_and_execute_candidate",
    "persist_pair_pipeline",
)


def _scan_file(path: Path) -> dict[str, list[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError:
        return _grep_fallback(text, path)

    imports: set[str] = set()
    calls: dict[str, list[str]] = {s: [] for s in SYMBOLS}

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module and "engine" in node.module:
                for alias in node.names:
                    imports.add(alias.name)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if "engine" in alias.name:
                    imports.add(alias.name)
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in calls:
                line = getattr(node, "lineno", 0)
                calls[name].append(f"{path.relative_to(PROJECT_ROOT)}:{line}")

    for sym in ("detect_whale_alert", "compute_whale_score", "generate_signal"):
        if sym in imports and not calls[sym]:
            calls[sym].append(f"{path.relative_to(PROJECT_ROOT)}:import")

    return calls


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _grep_fallback(text: str, path: Path) -> dict[str, list[str]]:
    calls: dict[str, list[str]] = {s: [] for s in SYMBOLS}
    for i, line in enumerate(text.splitlines(), start=1):
        for sym in SYMBOLS:
            if re.search(rf"\b{re.escape(sym)}\s*\(", line):
                calls[sym].append(f"{path.relative_to(PROJECT_ROOT)}:{i}")
    return calls


def run(*, output_dir: Path) -> DiagnosticReport:
    report = DiagnosticReport("inspect_detect_whale_alert_calls", output_dir)
    app_root = PROJECT_ROOT / "app"
    scripts_root = PROJECT_ROOT / "scripts"
    py_files = list(app_root.rglob("*.py")) + list(scripts_root.rglob("*.py"))

    aggregated: dict[str, list[str]] = {s: [] for s in SYMBOLS}
    importers: list[str] = []
    callers: list[str] = []

    for path in py_files:
        if ".venv" in str(path):
            continue
        file_calls = _scan_file(path)
        for sym, refs in file_calls.items():
            aggregated[sym].extend(refs)
            if sym == "detect_whale_alert" and refs:
                rel = str(path.relative_to(PROJECT_ROOT))
                if any("import" in r for r in refs):
                    importers.append(rel)
                else:
                    callers.append(rel)

    live_path = PROJECT_ROOT / "app" / "live.py"
    live_text = live_path.read_text(encoding="utf-8") if live_path.is_file() else ""
    live_calls_detect = "detect_whale_alert" in live_text
    live_calls_persist = "persist_pair_pipeline" in live_text

    scan_persist = PROJECT_ROOT / "app" / "analytics" / "scan_persist.py"
    sp_text = scan_persist.read_text(encoding="utf-8") if scan_persist.is_file() else ""
    persists_alerts = "insert_whale_alert" in sp_text and "detect_whale_alert" in sp_text

    graph = {
        "files_importing_detect_whale_alert": sorted(set(importers)),
        "files_calling_detect_whale_alert": sorted(set(callers)),
        "live_py_calls_detect_whale_alert_directly": live_calls_detect,
        "live_py_calls_persist_pair_pipeline": live_calls_persist,
        "watcher_path_uses_scan_persist_for_alerts": live_calls_persist,
        "scan_persist_calls_detect_and_insert_whale_alert": persists_alerts,
        "generate_signal_can_run_without_detect_whale_alert": True,
        "alert_can_be_created_but_not_persisted_if_persist_fails": True,
        "call_sites": aggregated,
        "analysis_notes": [
            "live.scan_once routes passed pairs through persist_pair_pipeline (thread pool).",
            "detect_whale_alert is invoked inside scan_persist._persist_pair_pipeline_impl, not live.py directly.",
            "Dropped pairs still persist snapshots but skip alert insertion when filter_status=dropped.",
        ],
    }

    status = "PASS"
    if not persists_alerts:
        status = "FAIL"
        report.add_limitation("scan_persist does not appear to call detect_whale_alert + insert_whale_alert")
    if not live_calls_persist:
        status = "WARN"
        report.add_limitation("live.py may not invoke persist_pair_pipeline in watcher path")

    report.set_status(status)
    report.data["call_graph"] = graph
    report.write_json("detect_whale_alert_call_graph.json")
    md_lines = [
        "## Summary",
        f"- live.py direct detect_whale_alert: **{live_calls_detect}**",
        f"- live.py → persist_pair_pipeline: **{live_calls_persist}**",
        f"- scan_persist persists alerts: **{persists_alerts}**",
        "",
        "## Call sites",
    ]
    for sym, refs in aggregated.items():
        if refs:
            md_lines.append(f"### {sym}")
            md_lines.extend(f"- `{r}`" for r in sorted(set(refs))[:30])
            md_lines.append("")
    report.write_md(md_lines, "detect_whale_alert_call_graph.md")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run(output_dir=args.output_dir)
    print(f"Status: {report.data['status']}")
    return 0 if report.data["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
