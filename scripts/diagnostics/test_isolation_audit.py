#!/usr/bin/env python3
"""Diagnostic 12 — unit test isolation audit."""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.diagnostics._common import DiagnosticReport, PROJECT_ROOT


def _grep_tests(pattern: str) -> list[str]:
    hits: list[str] = []
    tests_dir = PROJECT_ROOT / "tests"
    for path in tests_dir.rglob("test_*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            continue
        if re.search(pattern, text):
            hits.append(str(path.relative_to(PROJECT_ROOT)))
    return sorted(hits)


def _run_unittest_quick() -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-q"],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        return {
            "exit_code": proc.returncode,
            "stdout_tail": (proc.stdout or "")[-4000:],
            "stderr_tail": (proc.stderr or "")[-4000:],
            "passed": proc.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        return {"exit_code": -1, "error": "unittest timeout after 300s"}
    except Exception as exc:
        return {"exit_code": -1, "error": str(exc)}


def run(*, output_dir: Path, run_tests: bool = True) -> DiagnosticReport:
    report = DiagnosticReport("test_isolation_audit", output_dir)
    db_hits = _grep_tests(r"trader\.db|TRADER_DB|get_db\(\)|DB_PATH")
    settings_hits = _grep_tests(r"settings\.json|get_settings|upsert_setting|SETTINGS_PATH")
    mtime_hits = _grep_tests(r"st_mtime|touch\(|utime")
    economic_hits = _grep_tests(r"economic_gate_enabled\s*=\s*False|economic_gate_enabled.*false")

    test_run = _run_unittest_quick() if run_tests else {"skipped": True}

    status = "PASS" if test_run.get("passed") else "WARN"
    if test_run.get("exit_code", 0) not in (0, None) and not test_run.get("skipped"):
        status = "WARN" if test_run.get("exit_code") == 1 else status

    report.set_status(status)
    report.data.update({
        "unittest_result": test_run,
        "tests_accessing_trader_db": db_hits,
        "tests_accessing_settings_json": settings_hits,
        "tests_touching_sqlite_mtime": mtime_hits,
        "tests_assuming_economic_gate_disabled": economic_hits,
        "isolation_recommendations": [
            "Use tempfile NamedTemporaryFile SQLite URIs with mode=ro for read fixtures.",
            "Patch db.DB_PATH / TRADER_DB_PATH to tmp_path in setUp.",
            "Never call init_db() against production data/trader.db in unit tests.",
            "Load settings from deepcopy of defaults; avoid reading data/settings.json.",
            "Set economic_gate_enabled explicitly in each Phase 2 gate test fixture.",
        ],
    })
    report.write_json("test_isolation_audit.json")
    report.write_md([
        f"- Unittest passed: {test_run.get('passed', 'skipped')}",
        f"- Tests touching trader.db ({len(db_hits)}):",
        *[f"  - `{p}`" for p in db_hits[:20]],
        f"- Tests touching settings.json ({len(settings_hits)}):",
        *[f"  - `{p}`" for p in settings_hits[:20]],
    ], "test_isolation_audit.md")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-test-run", action="store_true")
    args = parser.parse_args()
    report = run(output_dir=args.output_dir, run_tests=not args.skip_test_run)
    print(f"Status: {report.data['status']}")
    return 0 if report.data["status"] != "FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
