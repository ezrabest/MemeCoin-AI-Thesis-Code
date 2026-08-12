"""Training dataset auto-build scheduler tests."""
from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from app.training.scheduler import (
    TrainingDatasetScheduler,
    compute_next_build_at,
    is_auto_build_enabled,
    public_build_status,
    reset_training_scheduler_for_tests,
)


class SchedulerConfigTests(unittest.TestCase):
    def tearDown(self) -> None:
        for key in (
            "AUTO_BUILD_TRAINING_DATASET",
            "TRAINING_DATASET_BUILD_ON_STARTUP",
            "TRAINING_DATASET_BUILD_HOUR",
            "TRAINING_DATASET_BUILD_MINUTE",
            "TRAINING_DATASET_BUILD_INTERVAL_HOURS",
        ):
            os.environ.pop(key, None)

    def test_auto_build_disabled_by_default(self) -> None:
        os.environ.pop("AUTO_BUILD_TRAINING_DATASET", None)
        self.assertFalse(is_auto_build_enabled())


class TrainingSchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._status_path = Path(self._tmpdir.name) / "last_build_status.json"
        os.environ["AUTO_BUILD_TRAINING_DATASET"] = "false"
        os.environ["TRAINING_DATASET_BUILD_ON_STARTUP"] = "false"
        reset_training_scheduler_for_tests()

    def tearDown(self) -> None:
        reset_training_scheduler_for_tests()
        self._tmpdir.cleanup()
        for key in (
            "AUTO_BUILD_TRAINING_DATASET",
            "TRAINING_DATASET_BUILD_ON_STARTUP",
            "TRAINER_DB_PATH",
        ):
            os.environ.pop(key, None)

    def test_auto_build_false_does_not_start_scheduler_thread(self) -> None:
        scheduler = TrainingDatasetScheduler(status_path=self._status_path)
        scheduler.start()
        self.assertIsNone(scheduler._scheduler_thread)
        scheduler.stop()

    def test_prevents_overlapping_builds(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def slow_build() -> dict:
            started.set()
            release.wait(timeout=5)
            return {"summary": {"rows_model_ready": 1, "pending_outcome_rows": 0}}

        scheduler = TrainingDatasetScheduler(
            status_path=self._status_path,
            build_fn=slow_build,
        )
        self.assertEqual(scheduler.request_build("test"), "started")
        self.assertTrue(started.wait(timeout=2))
        self.assertEqual(scheduler.request_build("test"), "already_running")
        release.set()
        time.sleep(0.2)
        scheduler.stop()

    def test_failed_build_captured_in_status(self) -> None:
        def failing_build() -> dict:
            raise RuntimeError("synthetic failure")

        scheduler = TrainingDatasetScheduler(
            status_path=self._status_path,
            build_fn=failing_build,
        )
        scheduler.request_build("test")
        time.sleep(0.3)
        status = scheduler.get_status()
        self.assertFalse(status["last_success"])
        self.assertIn("synthetic failure", status["last_error"])
        self.assertFalse(status["is_running"])

    def test_build_status_public_fields(self) -> None:
        scheduler = TrainingDatasetScheduler(status_path=self._status_path)
        scheduler.start()
        payload = public_build_status(scheduler.get_status())
        scheduler.stop()
        for key in (
            "auto_build_enabled",
            "is_running",
            "last_started_at",
            "last_finished_at",
            "last_success",
            "last_error",
            "last_duration_seconds",
            "last_summary_path",
            "next_scheduled_build_at",
            "manual_command",
        ):
            self.assertIn(key, payload)

    def test_automatic_build_does_not_call_llm_providers(self) -> None:
        mock_report = {
            "summary": {"rows_model_ready": 5, "pending_outcome_rows": 1},
        }
        with patch("app.models.predictor.generate_decision") as mock_ollama, patch(
            "app.models.predictor._gemini_json"
        ) as mock_gemini, patch(
            "app.training.scheduler.build_training_datasets",
            return_value=mock_report,
        ) as mock_build:
            scheduler = TrainingDatasetScheduler(
                status_path=self._status_path,
                build_fn=mock_build,
            )
            scheduler.request_build("test")
            time.sleep(0.2)
        mock_build.assert_called_once()
        mock_ollama.assert_not_called()
        mock_gemini.assert_not_called()


class TrainingSchedulerApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self._status_path = Path(self._tmpdir.name) / "last_build_status.json"
        os.environ["AUTO_BUILD_TRAINING_DATASET"] = "false"
        reset_training_scheduler_for_tests()

        import importlib
        import app.training.scheduler as scheduler_mod

        self.scheduler_mod = scheduler_mod
        self.scheduler = TrainingDatasetScheduler(
            status_path=self._status_path,
            build_fn=lambda: {"summary": {"rows_model_ready": 2, "pending_outcome_rows": 1}},
        )
        scheduler_mod._scheduler = self.scheduler

        import main
        from fastapi.testclient import TestClient

        importlib.reload(main)
        self.client = TestClient(main.app)

    def tearDown(self) -> None:
        reset_training_scheduler_for_tests()
        self._tmpdir.cleanup()
        os.environ.pop("AUTO_BUILD_TRAINING_DATASET", None)

    def test_build_now_started(self) -> None:
        response = self.client.post("/api/debug/training-dataset/build-now")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "started")

    def test_build_now_already_running(self) -> None:
        self.scheduler._status["is_running"] = True
        response = self.client.post("/api/debug/training-dataset/build-now")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "already_running")

    def test_build_status_endpoint(self) -> None:
        response = self.client.get("/api/debug/training-dataset/build-status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("manual_command", data)
        self.assertEqual(data["manual_command"], "python scripts/build_training_dataset.py")


if __name__ == "__main__":
    unittest.main()
