#!/usr/bin/env python3
"""Phase 4 diagnostic suite wrapper — runs all diagnostics, writes summary + ZIP."""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.diagnostics._common import PROJECT_ROOT, timestamp_slug  # noqa: E402

DIAGNOSTIC_SCRIPT_MARKERS = (
    "run_phase4_diagnostics",
    "scripts/diagnostics/",
    "scripts\\diagnostics\\",
)

PROCESS_IMAGE_NAMES = ("python.exe", "powershell.exe", "pwsh.exe")

DIAGNOSTICS: list[dict] = [
    {
        "id": 1,
        "module": "scripts.diagnostics.replay_whale_alert_detection",
        "script": "replay_whale_alert_detection.py",
        "args": lambda a: [
            "--auto-window" if a.auto_whale_window else None,
            "--start", "2026-06-10T15:00:00Z",
            "--end", "2026-06-10T20:00:00Z",
            "--limit", str(a.latest_snapshots),
        ],
        "skip_flag": None,
    },
    {
        "id": 2,
        "module": "scripts.diagnostics.sweep_whale_alert_thresholds",
        "script": "sweep_whale_alert_thresholds.py",
        "args": lambda a: ["--latest-snapshots", str(a.latest_snapshots)],
        "skip_flag": None,
    },
    {
        "id": 3,
        "module": "scripts.diagnostics.inspect_signal_to_trade_funnel",
        "script": "inspect_signal_to_trade_funnel.py",
        "args": lambda a: ["--minutes", str(a.minutes)],
        "skip_flag": None,
    },
    {
        "id": 4,
        "module": "scripts.diagnostics.inspect_detect_whale_alert_calls",
        "script": "inspect_detect_whale_alert_calls.py",
        "args": lambda _a: [],
        "skip_flag": None,
    },
    {
        "id": 5,
        "module": "scripts.diagnostics.inspect_threshold_consistency",
        "script": "inspect_threshold_consistency.py",
        "args": lambda _a: [],
        "skip_flag": None,
    },
    {
        "id": 6,
        "module": "scripts.diagnostics.inspect_rf_live_probability_distribution",
        "script": "inspect_rf_live_probability_distribution.py",
        "args": lambda a: ["--latest-candidates", str(a.latest_candidates)],
        "skip_flag": None,
    },
    {
        "id": 7,
        "module": "scripts.diagnostics.inspect_training_vs_live_feature_drift",
        "script": "inspect_training_vs_live_feature_drift.py",
        "args": lambda a: ["--latest-candidates", str(a.latest_candidates)],
        "skip_flag": "skip_feature_drift",
    },
    {
        "id": 8,
        "module": "scripts.diagnostics.inspect_duplicate_candidate_events",
        "script": "inspect_duplicate_candidate_events.py",
        "args": lambda a: ["--latest-candidates", str(a.latest_candidates)],
        "skip_flag": None,
    },
    {
        "id": 9,
        "module": "scripts.diagnostics.inspect_paper_trade_source_of_truth",
        "script": "inspect_paper_trade_source_of_truth.py",
        "args": lambda _a: [],
        "skip_flag": None,
    },
    {
        "id": 10,
        "module": "scripts.diagnostics.inspect_actionability_counterfactuals",
        "script": "inspect_actionability_counterfactuals.py",
        "args": lambda a: ["--latest-candidates", str(min(a.latest_candidates, 5000))],
        "skip_flag": "skip_counterfactuals",
    },
    {
        "id": 11,
        "module": "scripts.diagnostics.inspect_audit_reason_parsing",
        "script": "inspect_audit_reason_parsing.py",
        "args": lambda a: ["--minutes", str(a.minutes)],
        "skip_flag": None,
    },
    {
        "id": 12,
        "module": "scripts.diagnostics.test_isolation_audit",
        "script": "test_isolation_audit.py",
        "args": lambda _a: [],
        "skip_flag": "skip_test_isolation",
    },
]


def _bounded_diagnostic_process_check(
    *,
    stop_stale: bool = False,
    max_seconds: float = 30.0,
) -> dict:
    """
    Bounded process scan: only python.exe / powershell.exe / pwsh.exe.
    Never enumerates all processes. Hard cap max_seconds (default 30s).
    """
    started = time.monotonic()
    result: dict = {
        "checked_image_names": list(PROCESS_IMAGE_NAMES),
        "matched_pids": [],
        "stopped_pids": [],
        "elapsed_seconds": 0.0,
        "timed_out": False,
    }
    my_pid = os.getpid()
    markers = DIAGNOSTIC_SCRIPT_MARKERS + (str(PROJECT_ROOT).lower(),)
    pid_re = re.compile(r",(\d+)\s*$")

    def _remaining() -> float:
        return max(0.0, max_seconds - (time.monotonic() - started))

    if sys.platform != "win32":
        result["note"] = "Non-Windows: bounded check skipped (no stale stop)"
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        return result

    for image in PROCESS_IMAGE_NAMES:
        if _remaining() <= 0:
            result["timed_out"] = True
            break
        try:
            proc = subprocess.run(
                [
                    "wmic", "process", "where", f"name='{image}'",
                    "get", "ProcessId,CommandLine", "/format:csv",
                ],
                capture_output=True,
                text=True,
                timeout=min(10.0, _remaining()),
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            if isinstance(exc, subprocess.TimeoutExpired):
                result["timed_out"] = True
            else:
                result.setdefault("warnings", []).append(f"wmic unavailable for {image}: {exc}")
            break

        for line in (proc.stdout or "").splitlines():
            if _remaining() <= 0:
                result["timed_out"] = True
                break
            line_lower = line.lower()
            if not any(m.lower() in line_lower for m in markers):
                continue
            m = pid_re.search(line)
            if not m:
                continue
            pid = int(m.group(1))
            if pid == my_pid:
                continue
            result["matched_pids"].append({"pid": pid, "image": image})
            if stop_stale and _remaining() > 0:
                try:
                    subprocess.run(
                        ["taskkill", "/PID", str(pid), "/F"],
                        capture_output=True,
                        timeout=min(5.0, _remaining()),
                    )
                    result["stopped_pids"].append(pid)
                except (subprocess.TimeoutExpired, OSError):
                    pass

    if not result["matched_pids"] and _remaining() > 1 and sys.platform == "win32":
        try:
            ps_cmd = (
                "Get-CimInstance Win32_Process -Filter "
                "\"Name='python.exe' OR Name='powershell.exe' OR Name='pwsh.exe'\" "
                "| Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
            )
            proc = subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps_cmd],
                capture_output=True,
                text=True,
                timeout=min(15.0, _remaining()),
            )
            if proc.stdout.strip():
                payload = json.loads(proc.stdout)
                items = payload if isinstance(payload, list) else [payload]
                for item in items:
                    cmdline = str(item.get("CommandLine") or "").lower()
                    if not any(m.lower() in cmdline for m in markers):
                        continue
                    pid = int(item.get("ProcessId") or 0)
                    if pid and pid != my_pid:
                        result["matched_pids"].append({
                            "pid": pid,
                            "image": item.get("Name"),
                        })
                        if stop_stale and _remaining() > 0:
                            try:
                                subprocess.run(
                                    ["taskkill", "/PID", str(pid), "/F"],
                                    capture_output=True,
                                    timeout=min(5.0, _remaining()),
                                )
                                result["stopped_pids"].append(pid)
                            except (subprocess.TimeoutExpired, OSError):
                                pass
        except (subprocess.TimeoutExpired, OSError, json.JSONDecodeError) as exc:
            result.setdefault("warnings", []).append(f"powershell fallback failed: {exc}")

    result["elapsed_seconds"] = round(time.monotonic() - started, 3)
    return result


def _diagnostic_subdir(diag_dir: Path, spec: dict) -> Path:
    return diag_dir / f"d{spec['id']:02d}_{spec['script'].replace('.py', '')}"


def _diagnostic_report_status(sub_out: Path) -> str | None:
    for jp in sub_out.glob("*.json"):
        if jp.name == "run_meta.json":
            continue
        try:
            with open(jp, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and data.get("diagnostic") and data.get("status"):
                return str(data["status"])
        except (OSError, json.JSONDecodeError):
            continue
    return None


def _load_existing_result(sub_out: Path, spec: dict) -> dict | None:
    report_status = _diagnostic_report_status(sub_out)
    if report_status:
        meta_path = sub_out / "run_meta.json"
        meta = {}
        if meta_path.is_file():
            try:
                with open(meta_path, encoding="utf-8") as f:
                    meta = json.load(f)
            except (OSError, json.JSONDecodeError):
                pass
        return {
            "id": spec["id"],
            "name": spec["script"],
            "status": report_status,
            "exit_code": meta.get("exit_code"),
            "output_subdir": str(sub_out.relative_to(PROJECT_ROOT)),
            "meta": {k: v for k, v in meta.items() if k not in ("stdout", "stderr")},
        }
    return None


def _should_skip_diagnostic(sub_out: Path, spec: dict, *, resume: bool) -> tuple[bool, dict | None]:
    if not resume:
        return False, None
    existing = _load_existing_result(sub_out, spec)
    if existing is None:
        return False, None
    meta_path = sub_out / "run_meta.json"
    if meta_path.is_file():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("stderr") == "Diagnostic timed out after 900s" and not meta.get("report_status"):
                return False, None
        except (OSError, json.JSONDecodeError):
            pass
    return True, existing


def _run_diagnostic(script_path: Path, extra_args: list[str], output_dir: Path) -> dict:
    cmd = [
        sys.executable,
        str(script_path),
        "--output-dir",
        str(output_dir),
        *[x for x in extra_args if x is not None],
    ]
    meta = {
        "command": cmd,
        "script": script_path.name,
        "status": "FAIL",
        "exit_code": None,
        "stdout": "",
        "stderr": "",
        "report_status": None,
    }
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), capture_output=True, text=True, timeout=900, env=env,
        )
        meta["exit_code"] = proc.returncode
        meta["stdout"] = proc.stdout[-8000:]
        meta["stderr"] = proc.stderr[-8000:]
        meta["status"] = "PASS" if proc.returncode == 0 else "FAIL"
    except subprocess.TimeoutExpired:
        meta["stderr"] = "Diagnostic timed out after 900s"
        meta["status"] = "FAIL"
    except Exception as exc:
        meta["stderr"] = traceback.format_exc()
        meta["status"] = "FAIL"
        meta["error"] = str(exc)

    json_candidates = [p for p in output_dir.glob("*.json") if p.name != "run_meta.json"]
    for jp in json_candidates:
        try:
            with open(jp, encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "status" in data and data.get("diagnostic"):
                meta["report_status"] = data["status"]
                meta["status"] = data["status"]
                meta["report_file"] = jp.name
                break
        except (OSError, json.JSONDecodeError):
            continue
    return meta


def _build_findings(results: list[dict], diag_dir: Path) -> tuple[dict, str, str]:
    findings: list[dict] = []

    def _load(name: str) -> dict:
        for p in diag_dir.glob(f"**/*{name}*.json"):
            if p.name == "run_meta.json":
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get("diagnostic"):
                    return data
            except (OSError, json.JSONDecodeError):
                pass
        return {}

    replay = _load("replay_whale")
    sweep = _load("sweep_whale")
    funnel = _load("funnel_waterfall")
    rf = _load("rf_live")
    threshold = _load("threshold_consistency")
    counter = _load("actionability_counterfactuals")
    audit = _load("audit_reason_parsing")
    paper = _load("paper_trade")

    if funnel.get("funnel", {}).get("whale_alerts", 0) == 0 and funnel.get("funnel", {}).get("signals", 0) > 0:
        findings.append({
            "severity": "critical",
            "title": "Zero whale alerts despite active signal generation",
            "evidence": funnel.get("funnel"),
        })
    if funnel.get("likely_production_audit_parse_bug"):
        findings.append({
            "severity": "high",
            "title": "Audit reason JSON stored as string — API may count characters",
            "evidence": audit.get("production_single_char_keys_in_api"),
        })
    if threshold.get("mismatches"):
        findings.append({
            "severity": "high",
            "title": "Threshold chain mismatches across settings/code/UI",
            "evidence": threshold.get("mismatches"),
        })
    if replay.get("would_have_alert_count", 0) == 0 and replay.get("historical_persisted_whale_alerts_count", 0) > 0:
        findings.append({
            "severity": "high",
            "title": "Whale alert detector fails to recreate historical alerts",
            "evidence": {
                "historical": replay.get("historical_persisted_whale_alerts_count"),
                "replay": replay.get("would_have_alert_count"),
            },
        })
    dist = sweep.get("whale_score_distribution") or {}
    if dist.get("p95") is not None and dist["p95"] < 0.30:
        findings.append({
            "severity": "high",
            "title": "Live whale_score p95 below alert gate (0.30)",
            "evidence": dist,
        })
    rf_dist = rf.get("rf_probability_distribution") or {}
    if rf_dist.get("p95") is not None and rf_dist["p95"] < float(rf.get("configured_rf_threshold", 0.70)):
        findings.append({
            "severity": "high",
            "title": "RF live probabilities below configured threshold scale",
            "evidence": {"distribution": rf_dist, "threshold": rf.get("configured_rf_threshold")},
        })
    if not funnel.get("funnel", {}).get("economic_gate_enabled", True):
        findings.append({
            "severity": "critical",
            "title": "economic_gate_enabled=false blocks paper buy path",
            "evidence": funnel.get("funnel"),
        })
    scenarios = counter.get("scenarios") or []
    current = next((s for s in scenarios if s.get("scenario") == "current_settings"), {})
    best = max((s.get("actionable_buy_like_count", 0) for s in scenarios), default=0)
    if best > 0 and current.get("actionable_buy_like_count", 0) == 0:
        findings.append({
            "severity": "medium",
            "title": "Counterfactual settings would produce buy-like candidates",
            "evidence": {"current": current, "best_buy_like": best},
        })
    if paper.get("sources"):
        findings.append({
            "severity": "medium",
            "title": "Paper trade source comparison",
            "evidence": paper.get("comparisons"),
        })

    findings.sort(key=lambda x: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(x["severity"], 9))

    retrain_now = any(f["title"].startswith("RF live") for f in findings) and not any(
        "detector fails" in f["title"] for f in findings
    )
    tab_now = False
    retrain_rec = "after fixes to whale alerts, thresholds, and economic_gate_enabled" if not retrain_now else "not yet — fix data/threshold blockers first"
    tab_rec = "after RF/actionability fixes — not now"

    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "diagnostics": results,
        "findings_ranked": findings,
        "recommendations": {
            "retrain_rf_now": retrain_now,
            "retrain_rf_note": retrain_rec,
            "tab_re_evaluation_now": tab_now,
            "tab_re_evaluation_note": tab_rec,
            "next_engineering_phase": "Phase 5 — fix whale alert threshold alignment, enable economic gate for DEMO, fix audit JSON parsing, then reassess RF calibration",
        },
    }

    md_lines = [
        "# Phase 4 Findings (Ranked)",
        "",
        "## Top critical blockers",
    ]
    for i, f in enumerate(findings[:10], start=1):
        md_lines.append(f"{i}. **{f['title']}** ({f['severity']})")
        md_lines.append(f"   - Evidence: `{json.dumps(f.get('evidence'), default=str)[:200]}`")
    md_lines.extend([
        "",
        "## Recommendations",
        f"- Retrain RF now: **{retrain_now}** — {retrain_rec}",
        f"- TAB re-evaluation now: **{tab_now}** — {tab_rec}",
        f"- Next phase: {summary['recommendations']['next_engineering_phase']}",
    ])
    report_md = [
        "# Phase 4 Diagnostic Report",
        "",
        f"Generated: {summary['timestamp']}",
        "",
        "## Diagnostic status",
        *[f"- D{r['id']} {r['name']}: **{r['status']}**" for r in results],
        "",
        f"See `phase4_findings_ranked.md` for ranked root causes.",
    ]
    return summary, "\n".join(report_md), "\n".join(md_lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 4 diagnostic suite")
    parser.add_argument("--minutes", type=int, default=120)
    parser.add_argument("--latest-candidates", type=int, default=10000)
    parser.add_argument("--latest-snapshots", type=int, default=10000)
    parser.add_argument("--auto-whale-window", action="store_true")
    parser.add_argument("--skip-feature-drift", action="store_true")
    parser.add_argument("--skip-counterfactuals", action="store_true")
    parser.add_argument("--skip-test-isolation", action="store_true")
    parser.add_argument("--use-helius", action="store_true", help="Ignored — Helius disabled by default")
    parser.add_argument(
        "--resume-from",
        type=Path,
        default=None,
        help="Existing phase4_diagnostics_* output directory; skip completed diagnostics",
    )
    parser.add_argument(
        "--stop-stale-processes",
        action="store_true",
        help="Bounded stop of stale python/powershell phase4 child processes (max 30s)",
    )
    args = parser.parse_args()

    proc_check = _bounded_diagnostic_process_check(
        stop_stale=args.stop_stale_processes,
        max_seconds=30.0,
    )
    if proc_check.get("matched_pids"):
        print(f"Bounded process check: matched {len(proc_check['matched_pids'])} diagnostic-related process(es)")
    if proc_check.get("stopped_pids"):
        print(f"Stopped stale PIDs: {proc_check['stopped_pids']}")

    resume = args.resume_from is not None
    if resume:
        output_dir = Path(args.resume_from).resolve()
        if not output_dir.is_dir():
            print(f"ERROR: resume directory not found: {output_dir}", file=sys.stderr)
            return 1
    else:
        ts = timestamp_slug()
        output_dir = PROJECT_ROOT / "tests" / "Results" / f"phase4_diagnostics_{ts}"
        output_dir.mkdir(parents=True, exist_ok=True)

    diag_dir = output_dir / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    scripts_root = PROJECT_ROOT / "scripts" / "diagnostics"

    for spec in DIAGNOSTICS:
        skip_flag = spec.get("skip_flag")
        if skip_flag and getattr(args, skip_flag, False):
            existing = _load_existing_result(_diagnostic_subdir(diag_dir, spec), spec) if resume else None
            if existing and existing.get("status") == "SKIPPED":
                results.append(existing)
                continue
            results.append({
                "id": spec["id"],
                "name": spec["script"],
                "status": "SKIPPED",
                "skipped": True,
            })
            continue

        script_path = scripts_root / spec["script"]
        sub_out = _diagnostic_subdir(diag_dir, spec)
        sub_out.mkdir(parents=True, exist_ok=True)

        should_skip, existing = _should_skip_diagnostic(sub_out, spec, resume=resume)
        if should_skip and existing:
            print(f"Skipping diagnostic {spec['id']} (already complete): {existing.get('status')}")
            results.append(existing)
            continue

        extra = spec["args"](args)
        print(f"Running diagnostic {spec['id']}: {spec['script']} ...")
        meta = _run_diagnostic(script_path, extra, sub_out)
        results.append({
            "id": spec["id"],
            "name": spec["script"],
            "status": meta.get("report_status") or meta["status"],
            "exit_code": meta.get("exit_code"),
            "output_subdir": str(sub_out.relative_to(PROJECT_ROOT)),
            "meta": {k: v for k, v in meta.items() if k not in ("stdout", "stderr")},
        })
        log_path = sub_out / "run_meta.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)

    summary, report_md, findings_md = _build_findings(results, diag_dir)
    summary["output_dir"] = str(output_dir)
    summary["resume_from"] = str(args.resume_from) if resume else None
    summary["process_check"] = proc_check
    with open(output_dir / "phase4_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    with open(output_dir / "phase4_report.md", "w", encoding="utf-8") as f:
        f.write(report_md)
    with open(output_dir / "phase4_findings_ranked.md", "w", encoding="utf-8") as f:
        f.write(findings_md)

    zip_base = output_dir.parent / output_dir.name
    zip_path = shutil.make_archive(str(zip_base), "zip", root_dir=str(output_dir))
    print(f"Phase 4 complete — output: {output_dir}")
    print(f"ZIP: {zip_path}")
    for r in results:
        print(f"  D{r['id']:02d} {r['name']}: {r['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
