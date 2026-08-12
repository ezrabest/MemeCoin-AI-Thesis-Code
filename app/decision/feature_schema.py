"""AE7B runtime feature schema definition and enforcement."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

AE7B_FEATURE_SCHEMA_VERSION = "AE7B_RUNTIME_V1"
AE7C0_FEATURE_SCHEMA_VERSION = "AE7C0_RUNTIME_V1"

# Explicit allowlist — policy/scoring parameters, not outcome features.
POLICY_PARAMETER_ALLOWLIST: frozenset[str] = frozenset(
    {
        "tp_ratio",
        "sl_ratio",
        "time_stop_minutes",
        "round_trip_fee_pct",
    }
)

FORBIDDEN_FEATURE_PATTERNS: tuple[str, ...] = (
    "target",
    "label",
    "future",
    "net_return",
    "realized_return",
    "future_return",
    "gross_return",
    "exit_result",
    "exit_outcome",
    "exit_simulation",
    "simulation_outcome",
    "profit",
    "profitable",
    "realized",
    "outcome",
    "simulation",
    "_test",
    "_train",
    "_validation",
)

FORBIDDEN_EXACT_FEATURES: frozenset[str] = frozenset(
    {
        "test",
        "train",
        "validation",
    }
)

# Runtime feature definitions: name -> (source_table, source_column, required, dtype)
RUNTIME_FEATURE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "price_usd": {"source_table": "market_snapshots", "source_column": "price", "required": True, "dtype": "float"},
    "liquidity_usd": {"source_table": "market_snapshots", "source_column": "liquidity", "required": True, "dtype": "float"},
    "volume_h24": {"source_table": "market_snapshots", "source_column": "volume_24h", "required": False, "dtype": "float"},
    "txns_h24_buys": {"source_table": "market_snapshots", "source_column": "txns_buys", "required": False, "dtype": "float"},
    "txns_h24_sells": {"source_table": "market_snapshots", "source_column": "txns_sells", "required": False, "dtype": "float"},
    "txns_h24_total": {"source_table": "market_snapshots", "source_column": "txns_total", "required": False, "dtype": "float"},
    "buy_sell_ratio_h24": {"source_table": "market_snapshots", "source_column": "buy_ratio", "required": False, "dtype": "float"},
    "price_change_m5": {"source_table": "market_snapshots", "source_column": "price_change_m5", "required": False, "dtype": "float"},
    "price_change_h1": {"source_table": "market_snapshots", "source_column": "price_change_h1", "required": False, "dtype": "float"},
    "price_change_h6": {"source_table": "market_snapshots", "source_column": "price_change_h6", "required": False, "dtype": "float"},
    "price_change_h24": {"source_table": "market_snapshots", "source_column": "price_change_h24", "required": False, "dtype": "float"},
    "fdv": {"source_table": "market_snapshots", "source_column": "fdv", "required": False, "dtype": "float"},
    "whale_score_asof": {"source_table": "market_snapshots", "source_column": "whale_score", "required": False, "dtype": "float", "research_only": True},
    "signal_score": {"source_table": "signals", "source_column": "score", "required": False, "dtype": "float"},
    "sentiment_score": {"source_table": "sentiment_records", "source_column": "sentiment_score", "required": False, "dtype": "float"},
    "sentiment_source_count": {"source_table": "sentiment_records", "source_column": "_sentiment_source_count", "required": False, "dtype": "int"},
}

AE7C0_ADDITIONAL_FEATURE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "tp_ratio": {
        "source_table": "scoring_policy",
        "source_column": "tp_ratio",
        "required": False,
        "dtype": "float",
        "policy_parameter": True,
    },
    "sl_ratio": {
        "source_table": "scoring_policy",
        "source_column": "sl_ratio",
        "required": False,
        "dtype": "float",
        "policy_parameter": True,
    },
    "time_stop_minutes": {
        "source_table": "scoring_policy",
        "source_column": "time_stop_minutes",
        "required": False,
        "dtype": "int",
        "policy_parameter": True,
    },
    "round_trip_fee_pct": {
        "source_table": "scoring_policy",
        "source_column": "round_trip_fee_pct",
        "required": False,
        "dtype": "float",
        "policy_parameter": True,
    },
    "volume_to_liquidity_ratio": {
        "source_table": "derived",
        "source_column": "volume_h24/liquidity_usd",
        "required": False,
        "dtype": "float",
        "derived": True,
    },
}

AE7C0_DERIVED_OVERRIDES: dict[str, dict[str, Any]] = {
    "txns_h24_total": {
        "source_table": "derived",
        "source_column": "txns_h24_buys+txns_h24_sells",
        "required": False,
        "dtype": "float",
        "derived": True,
    },
    "buy_sell_ratio_h24": {
        "source_table": "derived",
        "source_column": "txns_h24_buys/txns_h24_sells",
        "required": False,
        "dtype": "float",
        "derived": True,
    },
}


class FeatureRowStatus(StrEnum):
    OK = "OK"
    MISSING_REQUIRED_FEATURE = "MISSING_REQUIRED_FEATURE"
    PARTIAL_OPTIONAL_MISSING = "PARTIAL_OPTIONAL_MISSING"


class BridgeReadinessDecision(StrEnum):
    RUNTIME_FEATURE_BRIDGE_CREATED = "RUNTIME_FEATURE_BRIDGE_CREATED"
    RUNTIME_FEATURE_BRIDGE_PARTIAL = "RUNTIME_FEATURE_BRIDGE_PARTIAL"
    RUNTIME_FEATURE_BRIDGE_BLOCKED_SCHEMA_GAP = "RUNTIME_FEATURE_BRIDGE_BLOCKED_SCHEMA_GAP"
    RUNTIME_FEATURE_BRIDGE_BLOCKED_LINEAGE_GAP = "RUNTIME_FEATURE_BRIDGE_BLOCKED_LINEAGE_GAP"
    RUNTIME_FEATURE_BRIDGE_BLOCKED_PARITY_GAP = "RUNTIME_FEATURE_BRIDGE_BLOCKED_PARITY_GAP"


def is_forbidden_feature_name(name: str) -> bool:
    if name in POLICY_PARAMETER_ALLOWLIST:
        return False
    lower = name.lower()
    if lower in FORBIDDEN_EXACT_FEATURES:
        return True
    return any(pat in lower for pat in FORBIDDEN_FEATURE_PATTERNS)


def get_runtime_feature_definitions(*, enriched: bool = False) -> dict[str, dict[str, Any]]:
    if not enriched:
        return dict(RUNTIME_FEATURE_DEFINITIONS)
    merged = dict(RUNTIME_FEATURE_DEFINITIONS)
    merged.update(AE7C0_DERIVED_OVERRIDES)
    merged.update(AE7C0_ADDITIONAL_FEATURE_DEFINITIONS)
    return merged


def filter_feature_definitions(*, enriched: bool = False) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """Return safe definitions and rejected feature names."""
    definitions = get_runtime_feature_definitions(enriched=enriched)
    safe: dict[str, dict[str, Any]] = {}
    rejected: list[str] = []
    for name, meta in definitions.items():
        if is_forbidden_feature_name(name):
            rejected.append(name)
        else:
            safe[name] = meta
    return safe, rejected


@dataclass
class RuntimeFeatureSchema:
    feature_schema_id: str
    feature_schema_version: str
    feature_names: list[str]
    feature_dtypes: dict[str, str]
    feature_source_map: dict[str, dict[str, str]]
    required_features: list[str]
    optional_features: list[str]
    forbidden_feature_patterns: list[str]
    schema_hash: str
    rejected_features: list[str] = field(default_factory=list)
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_schema_id": self.feature_schema_id,
            "feature_schema_version": self.feature_schema_version,
            "feature_names": self.feature_names,
            "feature_dtypes": self.feature_dtypes,
            "feature_source_map": self.feature_source_map,
            "required_features": self.required_features,
            "optional_features": self.optional_features,
            "forbidden_feature_patterns": self.forbidden_feature_patterns,
            "schema_hash": self.schema_hash,
            "rejected_features": self.rejected_features,
            "created_at_utc": self.created_at_utc,
        }


def compute_schema_hash(content: dict[str, Any]) -> str:
    serialized = json.dumps(content, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def build_runtime_feature_schema(*, enriched: bool = False) -> RuntimeFeatureSchema:
    """Build runtime feature schema from definitions (AE7B or AE7C0 enriched)."""
    version = AE7C0_FEATURE_SCHEMA_VERSION if enriched else AE7B_FEATURE_SCHEMA_VERSION
    safe_defs, rejected = filter_feature_definitions(enriched=enriched)
    feature_names = sorted(safe_defs.keys())
    required = [n for n in feature_names if safe_defs[n].get("required")]
    optional = [n for n in feature_names if not safe_defs[n].get("required")]
    dtypes = {n: safe_defs[n]["dtype"] for n in feature_names}
    source_map = {
        n: {
            "source_table": safe_defs[n]["source_table"],
            "source_column": safe_defs[n]["source_column"],
        }
        for n in feature_names
    }
    core = {
        "version": version,
        "feature_names": feature_names,
        "required_features": required,
        "optional_features": optional,
        "feature_dtypes": dtypes,
    }
    schema_hash = compute_schema_hash(core)
    schema_id = hashlib.sha256(f"{version}|{schema_hash}".encode("utf-8")).hexdigest()
    return RuntimeFeatureSchema(
        feature_schema_id=schema_id,
        feature_schema_version=version,
        feature_names=feature_names,
        feature_dtypes=dtypes,
        feature_source_map=source_map,
        required_features=required,
        optional_features=optional,
        forbidden_feature_patterns=list(FORBIDDEN_FEATURE_PATTERNS),
        schema_hash=schema_hash,
        rejected_features=rejected,
    )


def build_enriched_runtime_feature_schema() -> RuntimeFeatureSchema:
    """AE7C-0 enriched schema with policy parameters and safe derived features."""
    return build_runtime_feature_schema(enriched=True)


@dataclass
class FeatureRowBuildResult:
    feature_values: dict[str, float | int | None]
    feature_missingness: list[str]
    feature_source_columns: dict[str, str]
    feature_source_tables: dict[str, str]
    feature_status: str
    missing_required: list[str]
    rejected_features: list[str]
    whale_score_metadata: dict[str, Any] | None = None
    policy_feature_metadata: dict[str, dict[str, Any]] | None = None
    feature_missingness_reasons: dict[str, str] = field(default_factory=dict)

    @property
    def has_schema_gap(self) -> bool:
        return bool(self.missing_required)


def _is_missing_value(val: Any) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and str(val) == "nan":
        return True
    return False


def _record_missing(
    fname: str,
    *,
    missingness: list[str],
    missing_required: list[str],
    missing_reasons: dict[str, str],
    values: dict[str, float | int | None],
    schema: RuntimeFeatureSchema,
    reason: str = "MISSING_SOURCE_FEATURE",
) -> None:
    missingness.append(fname)
    missing_reasons[fname] = reason
    if fname in schema.required_features:
        missing_required.append(fname)
    values[fname] = None


def _apply_derived_features(
    *,
    values: dict[str, float | int | None],
    missingness: list[str],
    missing_required: list[str],
    missing_reasons: dict[str, str],
    schema: RuntimeFeatureSchema,
    snapshot_row: dict[str, Any] | None,
) -> None:
    volume = values.get("volume_h24")
    if volume is None and snapshot_row:
        volume = snapshot_row.get("volume_24h")
    liquidity = values.get("liquidity_usd")
    if liquidity is None and snapshot_row:
        liquidity = snapshot_row.get("liquidity")

    if "volume_to_liquidity_ratio" in schema.feature_names:
        if _is_missing_value(volume) or _is_missing_value(liquidity):
            _record_missing(
                "volume_to_liquidity_ratio",
                missingness=missingness,
                missing_required=missing_required,
                missing_reasons=missing_reasons,
                values=values,
                schema=schema,
            )
        elif float(liquidity) <= 0:
            _record_missing(
                "volume_to_liquidity_ratio",
                missingness=missingness,
                missing_required=missing_required,
                missing_reasons=missing_reasons,
                values=values,
                schema=schema,
                reason="INVALID_LIQUIDITY_USD_LE_ZERO",
            )
        else:
            values["volume_to_liquidity_ratio"] = float(volume) / float(liquidity)

    buys = values.get("txns_h24_buys")
    sells = values.get("txns_h24_sells")
    if buys is None and snapshot_row:
        buys = snapshot_row.get("txns_buys")
    if sells is None and snapshot_row:
        sells = snapshot_row.get("txns_sells")

    if "txns_h24_total" in schema.feature_names:
        if _is_missing_value(buys) or _is_missing_value(sells):
            if values.get("txns_h24_total") is None and snapshot_row:
                col_val = snapshot_row.get("txns_total")
                if not _is_missing_value(col_val):
                    values["txns_h24_total"] = float(col_val)
                else:
                    _record_missing(
                        "txns_h24_total",
                        missingness=missingness,
                        missing_required=missing_required,
                        missing_reasons=missing_reasons,
                        values=values,
                        schema=schema,
                    )
            else:
                _record_missing(
                    "txns_h24_total",
                    missingness=missingness,
                    missing_required=missing_required,
                    missing_reasons=missing_reasons,
                    values=values,
                    schema=schema,
                )
        else:
            values["txns_h24_total"] = float(buys) + float(sells)

    if "buy_sell_ratio_h24" in schema.feature_names:
        if _is_missing_value(buys) or _is_missing_value(sells):
            if values.get("buy_sell_ratio_h24") is None and snapshot_row:
                col_val = snapshot_row.get("buy_ratio")
                if not _is_missing_value(col_val):
                    values["buy_sell_ratio_h24"] = float(col_val)
                else:
                    _record_missing(
                        "buy_sell_ratio_h24",
                        missingness=missingness,
                        missing_required=missing_required,
                        missing_reasons=missing_reasons,
                        values=values,
                        schema=schema,
                    )
            else:
                _record_missing(
                    "buy_sell_ratio_h24",
                    missingness=missingness,
                    missing_required=missing_required,
                    missing_reasons=missing_reasons,
                    values=values,
                    schema=schema,
                )
        elif float(sells) <= 0:
            _record_missing(
                "buy_sell_ratio_h24",
                missingness=missingness,
                missing_required=missing_required,
                missing_reasons=missing_reasons,
                values=values,
                schema=schema,
                reason="INVALID_TXNS_H24_SELLS_LE_ZERO",
            )
        else:
            values["buy_sell_ratio_h24"] = float(buys) / float(sells)


def build_feature_values(
    *,
    snapshot_row: dict[str, Any] | None,
    signal_row: dict[str, Any] | None,
    sentiment_agg: dict[str, Any] | None,
    schema: RuntimeFeatureSchema,
    policy_context: dict[str, Any] | None = None,
) -> FeatureRowBuildResult:
    """Construct as-of feature values from local runtime rows."""
    values: dict[str, float | int | None] = {}
    missingness: list[str] = []
    source_cols: dict[str, str] = {}
    source_tables: dict[str, str] = {}
    missing_required: list[str] = []
    missing_reasons: dict[str, str] = {}
    whale_meta: dict[str, Any] | None = None
    policy_meta: dict[str, dict[str, Any]] | None = None

    enriched = schema.feature_schema_version == AE7C0_FEATURE_SCHEMA_VERSION
    safe_defs, _ = filter_feature_definitions(enriched=enriched)

    for fname in schema.feature_names:
        meta = safe_defs.get(fname, {})
        src_table = meta.get("source_table", "")
        src_col = meta.get("source_column", "")
        source_tables[fname] = src_table
        source_cols[fname] = src_col

        if meta.get("policy_parameter"):
            continue
        if meta.get("derived"):
            continue

        val = None
        if src_table == "market_snapshots" and snapshot_row:
            val = snapshot_row.get(src_col)
        elif src_table == "signals" and signal_row:
            val = signal_row.get(src_col)
        elif src_table == "sentiment_records" and sentiment_agg:
            if src_col == "_sentiment_source_count":
                val = sentiment_agg.get("source_count")
            else:
                val = sentiment_agg.get(src_col)

        if _is_missing_value(val):
            _record_missing(
                fname,
                missingness=missingness,
                missing_required=missing_required,
                missing_reasons=missing_reasons,
                values=values,
                schema=schema,
            )
        else:
            try:
                if meta.get("dtype") == "int":
                    values[fname] = int(val)
                else:
                    values[fname] = float(val)
            except (TypeError, ValueError):
                _record_missing(
                    fname,
                    missingness=missingness,
                    missing_required=missing_required,
                    missing_reasons=missing_reasons,
                    values=values,
                    schema=schema,
                    reason="INVALID_SOURCE_VALUE",
                )

        if fname == "whale_score_asof" and values.get(fname) is not None:
            whale_meta = {
                "whale_score_status": "RESEARCH_ONLY_PLAUSIBLE_FEATURE_CANDIDATE",
                "not_rule": True,
                "not_runtime_approved_as_standalone_signal": True,
            }

    if enriched:
        _apply_derived_features(
            values=values,
            missingness=missingness,
            missing_required=missing_required,
            missing_reasons=missing_reasons,
            schema=schema,
            snapshot_row=snapshot_row,
        )

    if policy_context and enriched:
        policy_values = policy_context.get("values") or {}
        policy_meta = policy_context.get("metadata") or {}
        for fname in POLICY_PARAMETER_ALLOWLIST:
            if fname not in schema.feature_names:
                continue
            val = policy_values.get(fname)
            if val is None:
                _record_missing(
                    fname,
                    missingness=missingness,
                    missing_required=missing_required,
                    missing_reasons=missing_reasons,
                    values=values,
                    schema=schema,
                    reason="MISSING_SCORING_POLICY_CONTEXT",
                )
            else:
                meta = safe_defs.get(fname, {})
                if meta.get("dtype") == "int":
                    values[fname] = int(val)
                else:
                    values[fname] = float(val)

    if missing_required:
        status = FeatureRowStatus.MISSING_REQUIRED_FEATURE.value
    elif missingness:
        status = FeatureRowStatus.PARTIAL_OPTIONAL_MISSING.value
    else:
        status = FeatureRowStatus.OK.value

    return FeatureRowBuildResult(
        feature_values=values,
        feature_missingness=missingness,
        feature_source_columns=source_cols,
        feature_source_tables=source_tables,
        feature_status=status,
        missing_required=missing_required,
        rejected_features=[],
        whale_score_metadata=whale_meta,
        policy_feature_metadata=policy_meta,
        feature_missingness_reasons=missing_reasons,
    )


def infer_model_family_from_schema_path(path: str) -> str:
    lower = path.lower().replace("\\", "/")
    if "_rf_" in lower or "clean_rf" in lower or "randomforest" in lower:
        return "RF"
    if "_xgb" in lower or "xgboost" in lower:
        return "XGB"
    if "tabicl" in lower or "_tab_" in lower:
        return "TAB"
    if "rf" in lower:
        return "RF"
    return "UNKNOWN"
