"""Tests for E7D candidate reservoir utilization audit."""
from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import scripts.build_e7d_candidate_reservoir_utilization_audit as e7d

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_e7d_candidate_reservoir_utilization_audit.py"

REQUIRED_REPORTS = {
    "e7d_summary_for_upload.txt",
    "e7d_current_scan_bottleneck.md",
    "e7d_existing_reservoir_audit.md",
    "e7d_reservoir_utilization_design.md",
    "e7d_runtime_risk_assessment.md",
    "e7d_e7e_recommendation.md",
}

REQUIRED_AUDITS = {
    "e7d_code_path_trace.csv",
    "e7d_db_reservoir_inventory.csv",
    "e7d_recent_scan_universe_audit.csv",
    "e7d_current_scan_vs_reservoir_gap.csv",
    "e7d_no_runtime_change_audit.csv",
}

REQUIRED_DESIGN = {
    "e7d_reservoir_selection_policy_spec.json",
    "e7d_candidate_eligibility_spec.json",
    "e7d_reservoir_scoring_flow_spec.json",
    "e7d_e7e_decision_gate_spec.json",
}


class E7DCandidateReservoirAuditTests(unittest.TestCase):
    def test_script_source_has_no_http_calls(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("httpx", text)
        self.assertNotIn("requests.", text)
        self.assertNotIn("urllib.request", text)

    def test_script_does_not_import_dexscreener_fetch(self):
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
        imports = [
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module and "dexscreener" in node.module
        ]
        self.assertEqual(imports, [])

    def test_open_db_readonly_uri(self):
        text = SCRIPT.read_text(encoding="utf-8")
        self.assertIn("mode=ro", text)

    def test_reservoir_design_specs_contain_required_fields(self):
        sel, elig, flow, gate = e7d.design_specs()
        self.assertIn("selection_policies", sel)
        self.assertIn("filters", elig)
        self.assertIn("staleness", elig)
        self.assertIn("eviction", elig)
        self.assertIn("stage_2", flow)
        self.assertIn("required_before_runtime", gate)

    def test_e7e_recommendation_keeps_runtime_blocked(self):
        _, _, _, gate = e7d.design_specs()
        self.assertIn("runtime", gate["blocked_until_gate"])
        self.assertIn("TAB", gate["blocked_until_gate"])
        self.assertEqual(gate["recommended_next_phase"], "E7E Offline Candidate Reservoir Prototype")

    def test_gap_computation_structure(self):
        summary = {
            "latest_scan_distinct_pairs": 100,
            "coins_distinct_pairs": 1337,
            "recent_window_distinct_pairs": {"last_24h": 261, "last_168h": 667},
            "latest_10_scans": [{"scan_id": "x", "distinct_pairs": 100}],
        }
        _, gap = e7d.scan_vs_reservoir_gap(summary)
        mult = next(r for r in gap if r["metric"] == "last_24h")
        self.assertEqual(mult["multiplier_vs_latest_scan"], 2.61)

    def test_no_runtime_change_audit_all_pass(self):
        rows = e7d.no_runtime_change_audit()
        self.assertTrue(all(r["passed"] for r in rows))

    def test_output_root_fails_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "phase_e7d"
            out.mkdir()
            (out / "x.txt").write_text("x", encoding="utf-8")
            with self.assertRaises(SystemExit):
                e7d.ensure_dirs(
                    argparse.Namespace(
                        output_root=str(out),
                        audit_root=str(Path(tmp) / "audit"),
                        overwrite=False,
                    )
                )

    def test_db_inventory_when_db_exists(self):
        if not e7d.DB_PATH.exists():
            self.skipTest("trader.db missing")
        conn = e7d.open_db_readonly()
        try:
            _, summary = e7d.db_inventory(conn, full_mode=False)
        finally:
            conn.close()
        self.assertTrue(summary.get("exists", True))
        if summary.get("latest_scan_distinct_pairs") is not None:
            self.assertGreater(summary["latest_scan_distinct_pairs"], 0)

    def test_smoke_writes_required_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            audit = Path(tmp) / "audit"
            proc = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--smoke",
                    "--output-root",
                    str(out),
                    "--audit-root",
                    str(audit),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
            for name in REQUIRED_REPORTS:
                self.assertTrue((out / "reports" / name).is_file(), name)
            for name in REQUIRED_AUDITS:
                self.assertTrue((out / "audits" / name).is_file(), name)
            for name in REQUIRED_DESIGN:
                self.assertTrue((out / "design" / name).is_file(), name)
            manifest = out / "manifests" / "e7d_manifest.json"
            self.assertTrue(manifest.is_file())
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertTrue(payload["runtime_blocked"])
            self.assertEqual(payload["automated_db_reservoir_selection"], "ABSENT")

    def test_manifest_exists_after_full_run(self):
        full_dirs = sorted(
            (ROOT / "data/training/manual_verified_results").glob(
                "phase_e7d_candidate_reservoir_utilization_audit_*"
            )
        )
        if not full_dirs:
            self.skipTest("no E7D full output yet")
        manifest = full_dirs[-1] / "manifests" / "e7d_manifest.json"
        self.assertTrue(manifest.is_file())
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        self.assertEqual(payload["phase"], "E7D")
        self.assertTrue(payload["recommends_e7e"])


if __name__ == "__main__":
    unittest.main()
