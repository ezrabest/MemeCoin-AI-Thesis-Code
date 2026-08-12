"""Runtime RF inference — load trained artifact, validate schema, predict_proba."""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("model_runtime_inference")

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "data" / "training"
MODELS_DIR = TRAINING_DIR / "models"
METRICS_PATH = MODELS_DIR / "baseline_metrics.json"

PRIMARY_RF_TARGET = "label_profitable_after_fees_4h"
PRIMARY_RF_MODEL = "random_forest"
PRIMARY_RF_HORIZON = "4h"
VALIDATED_RF_THRESHOLD = 0.70

# Future outcome columns — preserved in storage, excluded from live inference.
OUTCOME_COLUMN_EXACT = frozenset({
    "target_return_4h",
    "target_return_1h",
    "target_return_15m",
    "future_return",
    "future_return_4h",
    "future_return_1h",
    "future_return_15m",
    "realized_return",
    "target",
    "label",
    "label_profitable_after_fees_4h",
    "label_profitable_after_fees_1h",
    "label_profitable_after_fees_15m",
    "label_up_4h",
    "label_up_1h",
    "label_up_15m",
    "target_profitable_4h",
    "target_profitable_1h",
    "profitable_after_fees",
    "y_true",
    "predicted_probability",
    "predicted_class",
})

OUTCOME_COLUMN_SUBSTRINGS = (
    "future_return",
    "future_price",
    "max_future_return",
    "min_future_return",
    "label_",
    "profitable_after_fees",
    "realized",
    "outcome",
    "target_return",
    "target_profitable",
)


def _is_outcome_column(name: str) -> bool:
    lower = name.lower()
    if name in OUTCOME_COLUMN_EXACT:
        return True
    return any(sub in lower for sub in OUTCOME_COLUMN_SUBSTRINGS)


def _feature_schema_hash(numeric: list[str], categorical: list[str]) -> str:
    payload = json.dumps({"numeric": numeric, "categorical": categorical}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class RuntimeInferenceResult:
    status: str  # ok / not_available / load_failed / schema_error / inference_error
    predicted_probability: float | None = None
    prediction_generated_at: str | None = None
    model_snapshot_price: float | None = None
    audit_reasons: list[str] = field(default_factory=list)
    runtime_metadata: dict[str, Any] = field(default_factory=dict)
    rf_prediction: dict[str, Any] | None = None


class RuntimeModelInference:
    """Lazy-loaded runtime RF pipeline with strict schema alignment."""

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._model_path: Path | None = None
        self._schema: dict[str, Any] | None = None
        self._loaded_at: str | None = None
        self._is_full_pipeline: bool = False
        self._load_error: str | None = None
        self._artifact_mtime: float | None = None

    def _resolve_artifact_path(self) -> Path | None:
        candidates: list[Path] = []
        if METRICS_PATH.is_file():
            try:
                with open(METRICS_PATH, encoding="utf-8") as f:
                    metrics = json.load(f)
                best = (metrics.get("best_model_by_target") or {}).get(PRIMARY_RF_TARGET) or {}
                model_name = best.get("model_name", PRIMARY_RF_MODEL)
                models_block = (metrics.get("models_by_target") or {}).get(PRIMARY_RF_TARGET) or {}
                model_entry = (models_block.get("models") or {}).get(model_name) or {}
                path_str = model_entry.get("model_path")
                if path_str:
                    candidates.append(Path(path_str))
                candidates.append(MODELS_DIR / f"{PRIMARY_RF_TARGET}__{model_name}.joblib")
            except (OSError, json.JSONDecodeError) as exc:
                log.warning("Metrics-based artifact resolution failed: %s", exc)
        candidates.extend([
            MODELS_DIR / f"{PRIMARY_RF_TARGET}_best.joblib",
            MODELS_DIR / f"{PRIMARY_RF_TARGET}__{PRIMARY_RF_MODEL}.joblib",
        ])
        seen: set[str] = set()
        for path in candidates:
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            if path.is_file():
                return path
        return None

    def _load_schema_metadata(self) -> dict[str, Any] | None:
        if not METRICS_PATH.is_file():
            return None
        try:
            with open(METRICS_PATH, encoding="utf-8") as f:
                metrics = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("Schema metadata load failed: %s", exc)
            return None

        numeric = list(metrics.get("numeric_features") or [])
        categorical = list(metrics.get("categorical_features") or [])
        if not numeric and not categorical:
            return None

        best = (metrics.get("best_model_by_target") or {}).get(PRIMARY_RF_TARGET) or {}
        for col in numeric + categorical:
            if _is_outcome_column(col):
                return {
                    "invalid_for_runtime": True,
                    "target_leakage_columns": [col],
                    "numeric_features": numeric,
                    "categorical_features": categorical,
                }

        return {
            "numeric_features": numeric,
            "categorical_features": categorical,
            "feature_count": len(numeric) + len(categorical),
            "feature_schema_hash": _feature_schema_hash(numeric, categorical),
            "target_name": PRIMARY_RF_TARGET,
            "horizon": PRIMARY_RF_HORIZON,
            "model_name": best.get("model_name", PRIMARY_RF_MODEL),
            "allow_drop_extra_features": True,  # sklearn Pipeline ColumnTransformer remainder='drop'
            "schema_source": str(METRICS_PATH),
        }

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None or self._load_error == "not_found":
            return

        self._schema = self._load_schema_metadata()
        if self._schema is None:
            self._load_error = "schema_missing"
            return
        if self._schema.get("invalid_for_runtime"):
            self._load_error = "target_leakage_in_schema"
            return

        path = self._resolve_artifact_path()
        if path is None:
            self._load_error = "not_found"
            return

        try:
            import joblib
            obj = joblib.load(path)
        except Exception as exc:
            log.warning("Model artifact load failed (%s): %s", path, exc)
            self._load_error = "load_failed"
            self._model_path = path
            return

        self._pipeline = obj
        self._model_path = path
        self._is_full_pipeline = hasattr(obj, "predict_proba") and hasattr(obj, "named_steps")
        self._loaded_at = datetime.now(timezone.utc).isoformat()
        try:
            self._artifact_mtime = path.stat().st_mtime
        except OSError:
            self._artifact_mtime = None

    @property
    def artifact_available(self) -> bool:
        self._ensure_loaded()
        return self._pipeline is not None and self._schema is not None

    def runtime_metadata(self) -> dict[str, Any]:
        self._ensure_loaded()
        meta: dict[str, Any] = {
            "model_path": str(self._model_path) if self._model_path else None,
            "model_name": (self._schema or {}).get("model_name", PRIMARY_RF_MODEL),
            "target_name": PRIMARY_RF_TARGET,
            "horizon": PRIMARY_RF_HORIZON,
            "loaded_at": self._loaded_at,
            "is_full_pipeline": self._is_full_pipeline,
            "load_error": self._load_error,
        }
        if self._schema:
            meta.update({
                "feature_count": self._schema.get("feature_count"),
                "feature_schema_hash": self._schema.get("feature_schema_hash"),
                "feature_schema_source": self._schema.get("schema_source"),
            })
        if self._model_path and self._model_path.is_file():
            meta["model_version_hash"] = hashlib.sha256(
                self._model_path.read_bytes()
            ).hexdigest()[:16]
            meta["model_artifact_mtime"] = datetime.fromtimestamp(
                self._model_path.stat().st_mtime, tz=timezone.utc,
            ).isoformat()
        return meta

    def artifact_age_seconds(self, now: datetime | None = None) -> float | None:
        if self._artifact_mtime is None:
            self._ensure_loaded()
        if self._artifact_mtime is None:
            return None
        now = now or datetime.now(timezone.utc)
        return (now - datetime.fromtimestamp(self._artifact_mtime, tz=timezone.utc)).total_seconds()

    def _build_live_feature_row(
        self,
        candidate: Any,
        pair: dict[str, Any] | None,
    ) -> Any:
        import pandas as pd

        from ..training.snapshot_features import compute_snapshot_historical_features
        from ..training.wave_engine import add_whale_wave_score
        from .whale_wave_features import load_snapshots_for_pair

        self._ensure_loaded()
        schema = self._schema or {}
        pair = pair or {}
        txns = (pair.get("txns") or {}).get("h24") or {}
        pc = pair.get("priceChange") or {}

        row: dict[str, Any] = {
            "price_usd": candidate.price,
            "price": candidate.price,
            "liquidity_usd": candidate.liquidity_usd,
            "liquidity": candidate.liquidity_usd,
            "volume_24h": candidate.volume_24h or 0,
            "volume_24h_snap": candidate.volume_24h or 0,
            "whale_score": candidate.whale_score,
            "buy_ratio": candidate.buy_ratio,
            "txns_buys": candidate.buy_count or int(txns.get("buys") or 0),
            "txns_sells": candidate.sell_count or int(txns.get("sells") or 0),
            "price_change_h1": float(pc.get("h1") or 0),
            "price_change_h24": float(pc.get("h24") or 0),
            "price_change_1h": float(pc.get("h1") or 0),
            "price_change_24h": float(pc.get("h24") or 0),
            "sentiment_score": candidate.sentiment_score,
            "signal_confidence": candidate.signal_score,
            "score": candidate.signal_score,
            "confidence": candidate.signal_score,
        }

        for col in list(schema.get("numeric_features") or []) + list(schema.get("categorical_features") or []):
            row.setdefault(col, None)

        def _coerce_row_frame(frame: Any) -> Any:
            for col in schema.get("numeric_features") or []:
                if col in frame.columns:
                    frame[col] = pd.to_numeric(frame[col], errors="coerce")
            return frame

        snaps = load_snapshots_for_pair(candidate.pair_address, limit=300)
        if snaps:
            snap_df = pd.DataFrame(snaps)
            snap_df["coin_id"] = candidate.coin_id or 0
            snap_df["timestamp"] = snap_df.get("timestamp", pd.Series(dtype=object))
            if "liquidity" not in snap_df.columns and "liquidity_usd" in snap_df.columns:
                snap_df["liquidity"] = snap_df["liquidity_usd"]
            warnings: list[str] = []
            featured = compute_snapshot_historical_features(snap_df, warnings)
            if not featured.empty:
                last = featured.iloc[-1]
                for col in featured.columns:
                    if col not in row and col not in OUTCOME_COLUMN_EXACT:
                        val = last.get(col)
                        if val is not None and not (isinstance(val, float) and pd.isna(val)):
                            row[col] = val
            wave_df = add_whale_wave_score(_coerce_row_frame(pd.DataFrame([row])))
            if not wave_df.empty:
                for col in ("whale_wave_score", "whale_wave_direction", "has_whale_wave_history"):
                    if col in wave_df.columns:
                        row[col] = wave_df.iloc[0][col]

        return _coerce_row_frame(pd.DataFrame([row]))

    def _validate_and_align(
        self,
        frame: Any,
        schema: dict[str, Any],
    ) -> tuple[Any | None, list[str]]:
        import pandas as pd

        from .audit_reasons import AuditReason

        reasons: list[str] = []
        numeric = list(schema.get("numeric_features") or [])
        categorical = list(schema.get("categorical_features") or [])
        expected = numeric + categorical

        if not expected:
            reasons.append(AuditReason.MODEL_SCHEMA_METADATA_MISSING.value)
            return None, reasons

        present = set(frame.columns)
        leakage_in_frame = [c for c in present if _is_outcome_column(c)]
        if leakage_in_frame:
            reasons.append(AuditReason.MODEL_TRAINED_WITH_TARGET_LEAKAGE.value)
            return None, reasons

        missing = [c for c in expected if c not in present]
        if missing:
            reasons.append(AuditReason.MODEL_FEATURE_MISSING.value)
            return None, reasons

        extra = [c for c in present if c not in expected and not _is_outcome_column(c)]
        if extra and not schema.get("allow_drop_extra_features"):
            reasons.append(AuditReason.MODEL_FEATURE_EXTRA.value)
            return None, reasons

        aligned = frame[expected].copy()
        for col in numeric:
            if col in aligned.columns:
                aligned[col] = pd.to_numeric(aligned[col], errors="coerce")
        return aligned, reasons

    def predict_for_candidate(
        self,
        candidate: Any,
        pair: dict[str, Any] | None = None,
        *,
        now: datetime | None = None,
    ) -> RuntimeInferenceResult:
        from .audit_reasons import AuditReason

        now = now or datetime.now(timezone.utc)
        self._ensure_loaded()
        base_meta = self.runtime_metadata()

        if self._load_error == "target_leakage_in_schema":
            return RuntimeInferenceResult(
                status="schema_error",
                audit_reasons=[AuditReason.MODEL_TRAINED_WITH_TARGET_LEAKAGE.value],
                runtime_metadata=base_meta,
            )

        if self._schema is None:
            return RuntimeInferenceResult(
                status="schema_error",
                audit_reasons=[AuditReason.MODEL_SCHEMA_METADATA_MISSING.value],
                runtime_metadata=base_meta,
            )

        if self._pipeline is None:
            reason = AuditReason.MODEL_ARTIFACT_LOAD_FAILED.value if self._load_error == "load_failed" else AuditReason.MODEL_RUNTIME_INFERENCE_NOT_AVAILABLE.value
            return RuntimeInferenceResult(
                status="not_available" if self._load_error == "not_found" else "load_failed",
                audit_reasons=[reason],
                runtime_metadata=base_meta,
            )

        try:
            raw_frame = self._build_live_feature_row(candidate, pair)
        except Exception as exc:
            log.warning("Live feature row build failed: %s", exc)
            return RuntimeInferenceResult(
                status="inference_error",
                audit_reasons=[AuditReason.MODEL_SCHEMA_MISMATCH.value],
                runtime_metadata=base_meta,
            )

        aligned, val_reasons = self._validate_and_align(raw_frame, self._schema)
        if aligned is None:
            return RuntimeInferenceResult(
                status="schema_error",
                audit_reasons=val_reasons,
                runtime_metadata=base_meta,
            )

        assert not any(_is_outcome_column(c) for c in aligned.columns)

        try:
            proba = float(self._pipeline.predict_proba(aligned)[0, 1])
        except Exception as exc:
            log.warning("Runtime predict_proba failed: %s", exc)
            return RuntimeInferenceResult(
                status="inference_error",
                audit_reasons=[AuditReason.MODEL_SCHEMA_MISMATCH.value],
                runtime_metadata=base_meta,
            )

        pred_at = now.isoformat()
        rf_prediction = {
            "pair_address": candidate.pair_address,
            "event_timestamp": candidate.event_timestamp,
            "predicted_probability": proba,
            "target_name": PRIMARY_RF_TARGET,
            "target_horizon": PRIMARY_RF_HORIZON,
            "model_name": base_meta.get("model_name", PRIMARY_RF_MODEL),
            "prediction_source": "runtime_inference",
            "prediction_generated_at": pred_at,
        }

        return RuntimeInferenceResult(
            status="ok",
            predicted_probability=proba,
            prediction_generated_at=pred_at,
            model_snapshot_price=candidate.price,
            audit_reasons=[AuditReason.MODEL_RUNTIME_INFERENCE_OK.value],
            runtime_metadata={
                **base_meta,
                "feature_schema_source": self._schema.get("schema_source"),
                "inference_feature_columns": list(aligned.columns),
            },
            rf_prediction=rf_prediction,
        )


_inference_singleton: RuntimeModelInference | None = None


def get_runtime_model_inference() -> RuntimeModelInference:
    global _inference_singleton
    if _inference_singleton is None:
        _inference_singleton = RuntimeModelInference()
    return _inference_singleton


def reset_runtime_model_inference_for_tests() -> None:
    global _inference_singleton
    _inference_singleton = None
