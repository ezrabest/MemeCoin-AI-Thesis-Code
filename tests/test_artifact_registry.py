"""Tests for Phase E1 artifact registry foundation."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.artifacts.hash_utils import (  # noqa: E402
    build_artifact_id,
    build_logical_artifact_key,
    build_path_id,
    compute_content_hash,
    compute_schema_hash,
    normalize_dtype,
    normalize_dtypes_map,
    normalize_project_relative_path,
    to_posix_relative_path,
)
from app.artifacts.manifest_schema import ArtifactRecord  # noqa: E402
from app.artifacts.registry import (  # noqa: E402
    get_git_commit_hash,
    load_registry,
    scan_artifacts,
    validate_registry,
    write_registry_jsonl,
    write_validation_report,
)

PARQUET_AVAILABLE = True
try:
    import pyarrow  # noqa: F401
except ImportError:
    PARQUET_AVAILABLE = False


class HashUtilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _write(self, rel: str, content: bytes) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_content_hash_deterministic(self) -> None:
        path = self._write("a/file.txt", b"hello")
        h1 = compute_content_hash(path)
        h2 = compute_content_hash(path)
        self.assertEqual(h1, h2)

    def test_content_hash_changes_with_content(self) -> None:
        p1 = self._write("a.txt", b"one")
        p2 = self._write("b.txt", b"two")
        self.assertNotEqual(compute_content_hash(p1), compute_content_hash(p2))

    def test_content_hash_streaming_large_file(self) -> None:
        path = self._write("big.bin", b"x" * (3 * 1024 * 1024))
        digest = compute_content_hash(path, chunk_size=1024)
        self.assertEqual(len(digest), 64)

    def test_posix_relative_path(self) -> None:
        file_path = self._write("data/training/x.csv", b"a")
        rel = normalize_project_relative_path(file_path, self.root)
        self.assertEqual(rel, "data/training/x.csv")
        self.assertNotIn("\\", rel)
        self.assertNotIn(":", rel)

    def test_artifact_id_stable_across_roots(self) -> None:
        file_path = self._write("data/x.csv", b"same")
        content_hash = compute_content_hash(file_path)
        key = build_logical_artifact_key(
            phase="PHASE_B",
            artifact_type="model_prediction",
            logical_name="x",
            model="XGB",
        )
        root_a = self.root / "project_a"
        root_b = self.root / "project_b"
        copied_a = root_a / "data" / "x.csv"
        copied_b = root_b / "data" / "x.csv"
        copied_a.parent.mkdir(parents=True, exist_ok=True)
        copied_b.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(file_path, copied_a)
        shutil.copyfile(file_path, copied_b)
        hash_a = compute_content_hash(copied_a)
        hash_b = compute_content_hash(copied_b)
        self.assertEqual(hash_a, hash_b)
        self.assertEqual(build_artifact_id(key, hash_a), build_artifact_id(key, hash_b))

    def test_path_id_changes_when_relative_path_changes(self) -> None:
        self.assertNotEqual(build_path_id("a.csv"), build_path_id("b.csv"))

    def test_content_hash_same_for_copied_file(self) -> None:
        original = self._write("orig.csv", b"payload")
        copy_path = self._write("copy.csv", b"payload")
        self.assertEqual(compute_content_hash(original), compute_content_hash(copy_path))


class SchemaHashTests(unittest.TestCase):
    def test_schema_hash_stable(self) -> None:
        cols = ["a", "b"]
        dtypes = {"a": "int64", "b": "float64"}
        h1 = compute_schema_hash(cols, dtypes)
        h2 = compute_schema_hash(cols, dtypes)
        self.assertEqual(h1, h2)

    def test_schema_hash_changes_on_column_order(self) -> None:
        dtypes_a = {"a": "int64", "b": "float64"}
        dtypes_b = {"b": "float64", "a": "int64"}
        h1 = compute_schema_hash(["a", "b"], dtypes_a)
        h2 = compute_schema_hash(["b", "a"], dtypes_b)
        self.assertNotEqual(h1, h2)

    def test_schema_hash_changes_on_columns(self) -> None:
        dtypes = {"a": "int64"}
        h1 = compute_schema_hash(["a"], dtypes)
        h2 = compute_schema_hash(["a", "b"], {"a": "int64", "b": "int64"})
        self.assertNotEqual(h1, h2)

    def test_int32_int64_normalize_same_schema_hash(self) -> None:
        raw_a = {"x": "int32"}
        raw_b = {"x": "int64"}
        norm_a = normalize_dtypes_map(raw_a)
        norm_b = normalize_dtypes_map(raw_b)
        self.assertEqual(norm_a["x"], "int64")
        self.assertEqual(
            compute_schema_hash(["x"], norm_a),
            compute_schema_hash(["x"], norm_b),
        )

    def test_float32_float64_normalize_same_schema_hash(self) -> None:
        norm_a = normalize_dtypes_map({"x": "float32"})
        norm_b = normalize_dtypes_map({"x": "float64"})
        self.assertEqual(
            compute_schema_hash(["x"], norm_a),
            compute_schema_hash(["x"], norm_b),
        )

    def test_raw_dtypes_preserved_in_normalization(self) -> None:
        raw = {"x": "int32"}
        norm = normalize_dtypes_map(raw)
        self.assertEqual(raw["x"], "int32")
        self.assertEqual(norm["x"], "int64")
        self.assertEqual(normalize_dtype("Int64"), "int64")


class RegistryBehaviorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "app").mkdir()
        (self.root / "data").mkdir()
        self.scan_root = "data/training/sample"
        target = self.root / self.scan_root.replace("/", os.sep)
        target.mkdir(parents=True, exist_ok=True)
        (target / "xgb_pred_test.csv").write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
        (target / "notes.txt").write_text("plain", encoding="utf-8")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _scan(self, **kwargs):
        return scan_artifacts(
            project_root=self.root,
            scan_roots=[self.scan_root],
            branch_name="test_branch",
            generated_by_script="tests",
            **kwargs,
        )

    def test_artifact_id_deterministic(self) -> None:
        records1, _ = self._scan()
        records2, _ = self._scan(force_rehash=True)
        self.assertEqual(len(records1), len(records2))
        self.assertEqual(records1[0].artifact_id, records2[0].artifact_id)

    def test_csv_row_count_detection(self) -> None:
        records, _ = self._scan(force_rehash=True)
        csv_records = [r for r in records if r.extension == ".csv"]
        self.assertTrue(csv_records)
        self.assertEqual(csv_records[0].row_count, 2)

    def test_non_tabular_without_schema(self) -> None:
        records, _ = self._scan(force_rehash=True)
        txt = next(r for r in records if r.extension == ".txt")
        self.assertIsNone(txt.row_count)
        self.assertIsNone(txt.schema_hash)

    @unittest.skipUnless(PARQUET_AVAILABLE, "pyarrow not available")
    def test_parquet_row_count_detection(self) -> None:
        pq_path = self.root / self.scan_root.replace("/", os.sep) / "sample.parquet"
        pd.DataFrame({"x": [1, 2, 3]}).to_parquet(pq_path, index=False)
        records, _ = self._scan(force_rehash=True)
        pq = next(r for r in records if r.extension == ".parquet")
        self.assertEqual(pq.row_count, 3)
        self.assertIsNotNone(pq.schema_hash)

    def test_hash_status_computed_then_cached(self) -> None:
        records1, _ = self._scan(force_rehash=True)
        self.assertTrue(all(r.hash_status == "computed" for r in records1))
        cache = {r.project_relative_path: r for r in records1}
        records2, _ = self._scan(previous_registry=cache)
        self.assertTrue(all(r.hash_status == "reused_from_cache" for r in records2))
        self.assertTrue(all(r.metadata.get("cache_reused") for r in records2))

    def test_force_rehash_recomputes(self) -> None:
        records1, _ = self._scan(force_rehash=True)
        cache = {r.project_relative_path: r for r in records1}
        records2, _ = self._scan(previous_registry=cache, force_rehash=True)
        self.assertTrue(all(r.hash_status == "computed" for r in records2))

    def test_missing_git_warning(self) -> None:
        commit, warnings = get_git_commit_hash(self.root)
        self.assertIsNone(commit)
        self.assertIn("GIT_METADATA_UNAVAILABLE", warnings)

    def test_registry_writer_idempotent(self) -> None:
        out = self.root / "registry.jsonl"
        records, _ = self._scan(force_rehash=True)
        write_registry_jsonl(records, out)
        first = out.read_text(encoding="utf-8")
        write_registry_jsonl(records, out)
        second = out.read_text(encoding="utf-8")
        self.assertEqual(first, second)

    def test_validation_separates_warnings_and_errors(self) -> None:
        records, _ = self._scan(force_rehash=True)
        registry_path = self.root / "registry.jsonl"
        write_registry_jsonl(records, registry_path)
        report = validate_registry(registry_path, project_root=self.root)
        self.assertIn("errors", report)
        self.assertIn("warnings", report)
        self.assertIn(report["status"], {"ok", "warning", "error"})

    def test_validation_report_writes_json(self) -> None:
        records, _ = self._scan(force_rehash=True)
        registry_path = self.root / "registry.jsonl"
        report_path = self.root / "report.json"
        write_registry_jsonl(records, registry_path)
        report = validate_registry(registry_path, project_root=self.root)
        write_validation_report(report, report_path)
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], report["status"])

    def test_duplicate_path_detection_in_validation(self) -> None:
        records, _ = self._scan(force_rehash=True)
        dup = ArtifactRecord.from_dict(records[0].to_dict())
        write_registry_jsonl([records[0], dup], self.root / "registry.jsonl")
        report = validate_registry(self.root / "registry.jsonl", project_root=self.root)
        self.assertTrue(any("DUPLICATE" in err for err in report["errors"]))

    def test_dry_run_does_not_write_files(self) -> None:
        import importlib.util

        script_path = ROOT / "scripts" / "register_existing_artifacts.py"
        spec = importlib.util.spec_from_file_location("register_existing_artifacts", script_path)
        assert spec and spec.loader
        reg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reg)
        out = self.root / "data" / "artifact_registry" / "artifact_registry.jsonl"
        argv = [
            "register_existing_artifacts.py",
            "--dry-run",
            "--output-dir",
            "data/artifact_registry",
            "--include-root",
            self.scan_root,
        ]
        with mock.patch.object(reg, "detect_project_root", return_value=self.root), mock.patch(
            "sys.argv", argv
        ):
            code = reg.main()
        self.assertEqual(code, 0)
        self.assertFalse(out.exists())

    def test_no_sqlite_writes(self) -> None:
        db_path = self.root / "data" / "trader.db"
        db_path.write_bytes(b"sqlite")
        mtime_before = db_path.stat().st_mtime_ns
        self._scan(force_rehash=True)
        self.assertEqual(db_path.stat().st_mtime_ns, mtime_before)


class WindowsPathStabilityTests(unittest.TestCase):
    def test_to_posix_relative_path_windows_style_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            file_path = root / "data" / "x.csv"
            file_path.parent.mkdir(parents=True)
            file_path.write_text("a\n", encoding="utf-8")
            rel = to_posix_relative_path(file_path, root)
            self.assertEqual(rel, "data/x.csv")


if __name__ == "__main__":
    unittest.main()
