"""Offline TabICL prediction lookup and calibration metadata — RF uses runtime inference."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("model_lookup")

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "data" / "training"
MODELS_DIR = TRAINING_DIR / "models"

PRIMARY_RF_TARGET = "label_profitable_after_fees_4h"
PRIMARY_RF_MODEL = "random_forest"
VALIDATED_RF_THRESHOLD = 0.70
DEFAULT_TAB_SUFFIX = "nearest_neighbors_context_4096"
EXPECTED_RETURN_CALIBRATION_PATH = MODELS_DIR / "expected_return_calibration.json"


class ModelPredictionLookup:
    """Offline Tab lookup and calibration metadata. RF gate uses runtime inference."""

    def __init__(self) -> None:
        self._tab_index: dict[str, dict[str, Any]] | None = None
        self._tab_p98_threshold: float | None = None
        self._calibration: dict[str, Any] | None = None
        self._offline_file_mtime: float | None = None
        self._loaded = False

    def _load(self) -> None:
        if self._loaded:
            return
        self._loaded = True
        self._calibration = self._load_calibration()
        self._tab_index = self._build_tab_index()
        self._offline_file_mtime = self._resolve_offline_file_mtime()

    def _resolve_offline_file_mtime(self) -> float | None:
        mtimes: list[float] = []
        for name in ("predictions_validation.parquet", "predictions_test.parquet"):
            p = MODELS_DIR / name
            if p.is_file():
                try:
                    mtimes.append(p.stat().st_mtime)
                except OSError:
                    pass
        return max(mtimes) if mtimes else None

    def offline_prediction_file_age_seconds(self) -> float | None:
        self._load()
        if self._offline_file_mtime is None:
            return None
        now = datetime.now(timezone.utc)
        return (now - datetime.fromtimestamp(self._offline_file_mtime, tz=timezone.utc)).total_seconds()

    def expected_return_calibration_path(self) -> Path:
        return EXPECTED_RETURN_CALIBRATION_PATH

    def _load_calibration(self) -> dict[str, Any]:
        path = MODELS_DIR / "baseline_metrics.json"
        if not path.is_file():
            return {}
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            best = (data.get("best_model_by_target") or {}).get(PRIMARY_RF_TARGET) or {}
            return {
                "target_name": PRIMARY_RF_TARGET,
                "target_horizon": "4h",
                "model_name": best.get("model_name", PRIMARY_RF_MODEL),
                "validated_probability_threshold": VALIDATED_RF_THRESHOLD,
                "calibration_threshold": best.get("best_validation_threshold"),
                "metrics_available": bool(best),
            }
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Calibration metadata load failed: %s", exc)
            return {}

    def _read_parquet_frames(self, *paths: Path) -> Any:
        try:
            import pandas as pd
        except ImportError:
            return None
        frames = []
        for p in paths:
            if p.is_file():
                frames.append(pd.read_parquet(p))
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True)

    def _build_tab_index(self) -> dict[str, dict[str, Any]]:
        suffix = DEFAULT_TAB_SUFFIX
        val = MODELS_DIR / f"tabicl_v2_predictions_validation_{suffix}.parquet"
        test = MODELS_DIR / f"tabicl_v2_predictions_test_{suffix}.parquet"
        if not val.is_file():
            val = MODELS_DIR / "tabicl_v2_predictions_validation.parquet"
            test = MODELS_DIR / "tabicl_v2_predictions_test.parquet"
        frame = self._read_parquet_frames(val, test)
        if frame is None:
            return {}
        subset = frame[frame["target_name"] == PRIMARY_RF_TARGET] if "target_name" in frame.columns else frame
        index: dict[str, dict[str, Any]] = {}
        scores: list[float] = []
        for _, row in subset.iterrows():
            pair = str(row.get("pair_address") or "").strip()
            if not pair:
                continue
            score = float(row.get("predicted_probability") or row.get("tab_score") or 0)
            scores.append(score)
            ts_str = str(row.get("event_timestamp", ""))
            key = f"{pair}|{ts_str}"
            index[key] = {
                "pair_address": pair,
                "event_timestamp": ts_str,
                "tab_score": score,
                "tab_suffix": suffix,
                "target_name": PRIMARY_RF_TARGET,
                "model_name": "tabicl_v2",
            }
        if scores:
            try:
                import numpy as np
                self._tab_p98_threshold = float(np.percentile(scores, 98))
            except Exception:
                self._tab_p98_threshold = 0.98
        return index

    def lookup_tab_exact(self, pair_address: str, event_timestamp: str) -> dict[str, Any] | None:
        """Tab lookup only when exact pair_address + event_timestamp match exists."""
        self._load()
        if not self._tab_index:
            return None
        key = f"{pair_address.strip()}|{str(event_timestamp)}"
        entry = self._tab_index.get(key)
        if entry and self._tab_p98_threshold is not None:
            entry = dict(entry)
            entry["percentile_threshold"] = self._tab_p98_threshold
            entry["meets_percentile"] = float(entry.get("tab_score", 0)) >= self._tab_p98_threshold
        return entry

    def get_calibration(self) -> dict[str, Any]:
        self._load()
        return dict(self._calibration or {})


_lookup_singleton: ModelPredictionLookup | None = None


def get_model_lookup() -> ModelPredictionLookup:
    global _lookup_singleton
    if _lookup_singleton is None:
        _lookup_singleton = ModelPredictionLookup()
    return _lookup_singleton


def reset_model_lookup_for_tests() -> None:
    global _lookup_singleton
    _lookup_singleton = None
