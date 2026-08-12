"""Unit tests for System Configuration web UI helpers and PATCH endpoint."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.observability.settings_patch import SettingsPatchError, patch_settings
from app.web.dirty_payload import build_dirty_payload
from app.web.percent_conversion import (
    display_to_internal_number,
    format_display_value,
    internal_to_display_number,
)
from app.web.settings_field_metadata import EDITABLE_KEYS
from app.web.settings_inspector import build_canonical_inspector_rows, build_inspector_rows, dependency_notes
from app.web.validation import validate_field


class PercentConversionTests(unittest.TestCase):
    def test_max_slippage_display(self) -> None:
        self.assertAlmostEqual(internal_to_display_number("max_slippage_pct", 0.015), 1.5, places=6)
        self.assertEqual(format_display_value("max_slippage_pct", 0.015), "1.5%")

    def test_max_slippage_save(self) -> None:
        self.assertAlmostEqual(display_to_internal_number("max_slippage_pct", 1.5), 0.015, places=6)

    def test_rf_probability_display(self) -> None:
        self.assertAlmostEqual(internal_to_display_number("rf_probability_threshold", 0.70), 70.0, places=4)
        self.assertAlmostEqual(display_to_internal_number("rf_probability_threshold", 70), 0.70, places=4)


class DirtyPayloadTests(unittest.TestCase):
    def test_only_modified_canonical_keys(self) -> None:
        loaded = {"max_slippage_pct": 0.015, "economic_gate_enabled": False}
        form = {"max_slippage_pct": 0.02, "economic_gate_enabled": False}
        dirty = build_dirty_payload(form_values=form, loaded_canonical=loaded, editable_keys=EDITABLE_KEYS)
        self.assertEqual(list(dirty.keys()), ["max_slippage_pct"])
        self.assertAlmostEqual(dirty["max_slippage_pct"], 0.02, places=6)

    def test_no_legacy_alias_keys(self) -> None:
        loaded = {"min_liquidity_usd": 5000.0}
        form = {"min_liquidity_usd": 6000.0}
        dirty = build_dirty_payload(form_values=form, loaded_canonical=loaded, editable_keys=EDITABLE_KEYS)
        self.assertNotIn("minLiquidity", dirty)
        self.assertIn("min_liquidity_usd", dirty)

    def test_read_only_not_in_editable(self) -> None:
        self.assertNotIn("live_trading_enabled", EDITABLE_KEYS)
        self.assertNotIn("tab_standalone_trading_enabled", EDITABLE_KEYS)


class InspectorTests(unittest.TestCase):
    def test_inspector_from_effective_payload(self) -> None:
        audit_path = Path(__file__).resolve().parent.parent / "data" / "audits"
        samples = sorted(audit_path.glob("settings_effective_*.json"))
        if not samples:
            self.skipTest("No settings_effective audit sample on disk")
        payload = json.loads(samples[-1].read_text(encoding="utf-8"))
        rows = build_inspector_rows(payload)
        self.assertGreater(len(rows), 0)
        canonical_rows = build_canonical_inspector_rows(payload)
        slippage = next(r for r in canonical_rows if r["canonical_key"] == "max_slippage_pct")
        self.assertEqual(slippage["displayed_value"], "1.5%")
        self.assertIn("0.015", slippage["internal_value"])
        self.assertIn("source", slippage)
        self.assertIn("default_value", slippage)

    def test_inspector_not_static_fake(self) -> None:
        payload_a = {
            "canonical": {"economic_gate_enabled": True, "max_slippage_pct": 0.02},
            "sources": {"economic_gate_enabled": "settings.json", "max_slippage_pct": "default"},
            "defaults": {"economic_gate_enabled": False, "max_slippage_pct": 0.015},
            "aliases_resolved": {},
        }
        payload_b = {
            "canonical": {"economic_gate_enabled": False, "max_slippage_pct": 0.015},
            "sources": {"economic_gate_enabled": "default", "max_slippage_pct": "default"},
            "defaults": {"economic_gate_enabled": False, "max_slippage_pct": 0.015},
            "aliases_resolved": {},
        }
        row_a = build_canonical_inspector_rows(payload_a)[0]
        row_b = build_canonical_inspector_rows(payload_b)[0]
        self.assertNotEqual(row_a["internal_value"], row_b["internal_value"])

    def test_economic_gate_and_tab_demo_in_metadata(self) -> None:
        from app.web.settings_field_metadata import FIELD_SPEC_BY_KEY

        self.assertIn("economic_gate_enabled", FIELD_SPEC_BY_KEY)
        self.assertIn("tab_confidence_boost_enabled_demo", FIELD_SPEC_BY_KEY)


class ValidationTests(unittest.TestCase):
    def test_highlights_invalid_numeric(self) -> None:
        self.assertIsNotNone(validate_field("max_slippage_pct", "abc"))
        self.assertIsNone(validate_field("max_slippage_pct", 1.5))

    def test_min_liquidity_non_negative(self) -> None:
        self.assertIsNotNone(validate_field("min_liquidity_usd", -1))
        self.assertIsNone(validate_field("min_liquidity_usd", 5000))


class SettingsPatchApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._root = Path(self._tmpdir.name)
        self._settings_path = self._root / "settings.json"
        self._settings_path.write_text("{}", encoding="utf-8")
        os.environ["TRADER_DB_PATH"] = str(self._root / "test.db")

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        os.environ.pop("TRADER_DB_PATH", None)

    def _patch_settings_path(self):
        import app.database as database

        return patch.object(database, "SETTINGS_PATH", self._settings_path)

    def test_patch_returns_effective_shape(self) -> None:
        with self._patch_settings_path():
            result = patch_settings({"max_slippage_pct": 1.5, "economic_gate_enabled": True})
        self.assertIn("canonical", result)
        self.assertIn("settings_hash", result)
        self.assertAlmostEqual(result["canonical"]["max_slippage_pct"], 0.015, places=6)
        self.assertTrue(result["canonical"]["economic_gate_enabled"])

    def test_rejects_legacy_alias(self) -> None:
        with self._patch_settings_path():
            with self.assertRaises(SettingsPatchError) as ctx:
                patch_settings({"minLiquidity": 5000})
            self.assertIn("minLiquidity", ctx.exception.field_errors)

    def test_rejects_live_trading_enable(self) -> None:
        with self._patch_settings_path():
            with self.assertRaises(SettingsPatchError) as ctx:
                patch_settings({"live_trading_enabled": True})
            self.assertIn("live_trading_enabled", ctx.exception.field_errors)

    def test_patch_http_endpoint(self) -> None:
        with self._patch_settings_path():
            import importlib
            import app.database as database

            database.init_db()
            import main

            importlib.reload(main)
            client = TestClient(main.app)
            r = client.patch("/api/settings", json={"max_slippage_pct": 1.5})
            self.assertEqual(r.status_code, 200)
            data = r.json()
            self.assertAlmostEqual(data["canonical"]["max_slippage_pct"], 0.015, places=6)
            self.assertIn("tab_confidence_boost_enabled_demo", data["canonical"])


class WebUiArtifactTests(unittest.TestCase):
    def test_no_pyside6_desktop_stack(self) -> None:
        root = Path(__file__).resolve().parent.parent
        self.assertFalse((root / "desktop.py").exists())
        self.assertFalse((root / "app" / "ui").exists())
        req = (root / "requirements.txt").read_text(encoding="utf-8")
        self.assertNotIn("PySide6", req)

    def test_web_dashboard_has_system_config(self) -> None:
        html = (Path(__file__).resolve().parent.parent / "static" / "index.html").read_text(encoding="utf-8")
        js = (Path(__file__).resolve().parent.parent / "static" / "system_config.js").read_text(encoding="utf-8")
        self.assertIn("System Configuration", html)
        self.assertIn("/api/settings/effective", js)
        self.assertIn('method: "PATCH"', js)
        self.assertIn("economic_gate_enabled", js)
        self.assertIn("tab_confidence_boost_enabled_demo", js)
        self.assertIn("Discard Changes / Refresh from Server", html)


class DependencyWarningLogicTests(unittest.TestCase):
    def test_tab_inactive_when_economic_gate_off(self) -> None:
        notes = dependency_notes(
            "tab_confidence_boost_enabled",
            {"tab_confidence_boost_enabled": True, "economic_gate_enabled": False},
        )
        self.assertIn("economic gate OFF", notes)


if __name__ == "__main__":
    unittest.main()
