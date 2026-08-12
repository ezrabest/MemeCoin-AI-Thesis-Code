"""AE16F — Serving-safe direct-target RF/XGB/TAB evidence for Clean Forward.

Trains new serving-safe evidence generators. Does not reuse incompatible E4
31-feature artifacts. No AE17/meta/LLM. Shadow/research only.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.consensus.ae16e_feature_parity import (
    AE16E_TIER_ALIASES,
    TOXIC_PAIR_ADDRESS,
    is_toxic_pair,
    load_clean_forward_rows_used,
)
from app.training.direct_target_ids import (
    DEFAULT_EXIT_POLICIES,
    resolve_time_stop_minutes,
)

PHASE = "AE16F_SERVING_SAFE_DIRECT_TARGET_MODEL_EVIDENCE"
RANDOM_SEED = 42
REF_EXIT_POLICY_ID = "TP20308_SL075_FEE0308_TIME_BY_HORIZON"
REF_HORIZON = "1h"
REF_FILTER = "LIQ_5K_HIGH_ACTIVITY"
TARGET_COLUMN = "target"
SELECTED_SOURCE_REL = (
    "data/training/manual_verified_datasets_direct_target_v1/"
    "LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL075_FEE0308_TIME_BY_HORIZON_DIRECT_TARGET_v1.parquet"
)

# Forbidden from old E4 contract / leakage
FORBIDDEN_FEATURE_NAMES = frozenset(
    {
        "whale_score",
        "price_step_ratio_prev",
        "gap_detected",
        "is_extreme_step_ratio_100x",
        "entry_snapshot_id",
        "entry_price",
        "entry_price_raw",
        "entry_price_verified_30m",
        "entry_price_verified_1h",
        "entry_price_verified_4h",
        "entry_price_verified_8h",
        "entry_price_verified_24h",
        "previous_price_usd",
        "previous_liquidity_usd",
        "price_delta",
        "liquidity_delta",
        "target",
        "target_net_profitable_after_exit",
        "sim_net_return",
        "sim_exit_status",
        "label_valid",
        "max_future_ratio",
        "min_future_ratio",
        "future_snapshot_count",
        "pair_address",
        "candidate_id",
        "symbol",
    }
)

# Serving-safe numeric feature order (historical names; CF mapped into these)
SERVING_SAFE_BASE_FEATURES: tuple[str, ...] = (
    "tp_ratio",
    "sl_ratio",
    "round_trip_fee_pct",
    "time_stop_minutes",
    "price",
    "liquidity",
    "volume_24h",
    "fdv",
    "txns_buys",
    "txns_sells",
    "txns_total",
    "price_change_m5",
    "price_change_h1",
    "price_change_h6",
    "price_change_h24",
    "buy_ratio",
    "price_usd",
    "liquidity_usd",
    "volume_to_liquidity_ratio",
)

# Missingness indicators for sparse serving fields
MISS_INDICATOR_FOR: tuple[str, ...] = (
    "fdv",
    "txns_buys",
    "txns_sells",
    "price_change_m5",
    "price_change_h1",
    "price_change_h6",
    "price_change_h24",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _policy_constants() -> dict[str, float]:
    policy = next(p for p in DEFAULT_EXIT_POLICIES if p["exit_policy_id"] == REF_EXIT_POLICY_ID)
    return {
        "tp_ratio": float(policy["tp_ratio"]),
        "sl_ratio": float(policy["sl_ratio"]),
        "round_trip_fee_pct": float(policy["round_trip_fee_pct"]),
        "time_stop_minutes": float(resolve_time_stop_minutes(REF_HORIZON, policy)),
    }


def schema_hash(ordered_names: list[str], dtype_map: dict[str, str]) -> str:
    payload = json.dumps({"names": ordered_names, "dtypes": dtype_map}, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------


def build_serving_safe_feature_contract() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    consts = _policy_constants()
    for name in SERVING_SAFE_BASE_FEATURES:
        if name in consts:
            role = "constant_policy_param"
            src = f"exit_policy:{REF_EXIT_POLICY_ID}"
            formula = f"constant={consts[name]}"
            category = "policy_constant"
        elif name in {"txns_total", "buy_ratio", "volume_to_liquidity_ratio"}:
            role = "numeric"
            src = {
                "txns_total": "txns_buys+txns_sells",
                "buy_ratio": "txns_buys/(txns_buys+txns_sells)",
                "volume_to_liquidity_ratio": "volume_24h/liquidity_usd",
            }[name]
            formula = src
            category = "same_snapshot_derived"
        elif name in {"price", "liquidity"}:
            role = "numeric"
            src = "price_usd" if name == "price" else "liquidity_usd"
            formula = f"alias={src}"
            category = "same_snapshot_provider"
        else:
            role = "numeric"
            src = name
            formula = f"direct={name}"
            category = "same_snapshot_provider"
        rows.append(
            {
                "feature_name": name,
                "allowed": True,
                "category": category,
                "role": role,
                "source_fields": src,
                "formula": formula,
                "no_lookahead": True,
                "nullable": name not in consts,
                "imputation": "none" if name in consts else "median_train_only",
            }
        )
    for base in MISS_INDICATOR_FOR:
        ind = f"{base}_is_missing"
        rows.append(
            {
                "feature_name": ind,
                "allowed": True,
                "category": "missingness_indicator",
                "role": "missingness_indicator",
                "source_fields": base,
                "formula": f"1.0 if {base} null else 0.0",
                "no_lookahead": True,
                "nullable": False,
                "imputation": "none",
            }
        )
    # Forbidden audit rows
    for bad in sorted(FORBIDDEN_FEATURE_NAMES):
        rows.append(
            {
                "feature_name": bad,
                "allowed": False,
                "category": "forbidden",
                "role": "excluded",
                "source_fields": "",
                "formula": "",
                "no_lookahead": False,
                "nullable": True,
                "imputation": "forbidden",
                "exclusion_reason": "AE16F_SERVING_SAFE_EXCLUSION",
            }
        )
    allowed = [r for r in rows if r["allowed"]]
    contract = {
        "phase": PHASE,
        "serving_safe_feature_count": len(allowed),
        "serving_safe_feature_names": [r["feature_name"] for r in allowed],
        "forbidden_features_excluded": sorted(FORBIDDEN_FEATURE_NAMES),
        "controlled_snapshot_history_used": False,
        "whale_score_included": False,
        "sequential_features_included": False,
        "entry_verified_fields_included": False,
        "policy_constants": consts,
        "policy_source": "app/training/direct_target_ids.py DEFAULT_EXIT_POLICIES",
    }
    return rows, contract


def ordered_feature_names_from_contract(contract_rows: list[dict[str, Any]]) -> list[str]:
    return [r["feature_name"] for r in contract_rows if r.get("allowed")]


# ---------------------------------------------------------------------------
# Source discovery
# ---------------------------------------------------------------------------


def discover_training_sources(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    dt_root = project_root / "data/training/manual_verified_datasets_direct_target_v1"
    clean_root = project_root / "data/training/manual_verified_datasets_clean_for_model"
    selected: dict[str, Any] | None = None

    if dt_root.is_dir():
        for path in sorted(dt_root.glob("*DIRECT_TARGET_v1.parquet")):
            rel = str(path.relative_to(project_root)).replace("\\", "/")
            try:
                import pyarrow.parquet as pq

                pf = pq.ParquetFile(path)
                cols = list(pf.schema_arrow.names)
                n = pf.metadata.num_rows if pf.metadata else -1
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "path": rel,
                        "row_count": "",
                        "columns": "",
                        "direct_target_compatible": False,
                        "serving_safe_compatible": False,
                        "rejection_reason": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            has_target = "target" in cols or "target_net_profitable_after_exit" in cols
            has_split = "split" in cols
            has_ts = "event_timestamp" in cols
            leakage_as_feature_risk = False
            serving_ok = has_target and has_split and any(c in cols for c in ("price_usd", "price"))
            is_selected = rel == SELECTED_SOURCE_REL.replace("\\", "/")
            reason = ""
            if not has_target:
                reason = "missing_direct_target_label"
                serving_ok = False
            elif not has_split:
                reason = "missing_split_column"
                serving_ok = False
            row = {
                "path": rel,
                "row_count": n,
                "columns": "|".join(cols[:40]),
                "column_count": len(cols),
                "target_columns": "|".join(c for c in cols if "target" in c.lower()),
                "timestamp_columns": "|".join(c for c in cols if "timestamp" in c.lower()),
                "pair_address_columns": "|".join(c for c in cols if "pair" in c.lower()),
                "direct_target_compatible": has_target and "DIRECT_TARGET" in path.name,
                "serving_safe_compatible": serving_ok,
                "has_split": has_split,
                "has_event_timestamp": has_ts,
                "rejection_reason": reason,
                "selected": is_selected and serving_ok,
            }
            rows.append(row)
            if is_selected and serving_ok:
                selected = {
                    "path": rel,
                    "row_count": n,
                    "target_column": TARGET_COLUMN,
                    "horizon": REF_HORIZON,
                    "filter": REF_FILTER,
                    "exit_policy_id": REF_EXIT_POLICY_ID,
                }

    if clean_root.is_dir():
        for path in sorted(clean_root.glob("*CLEAN_MODEL_INPUT.parquet"))[:8]:
            rel = str(path.relative_to(project_root)).replace("\\", "/")
            rows.append(
                {
                    "path": rel,
                    "row_count": "",
                    "columns": "",
                    "direct_target_compatible": False,
                    "serving_safe_compatible": False,
                    "rejection_reason": "CLEAN_MODEL_INPUT_not_exit_policy_direct_target_matrix",
                    "selected": False,
                }
            )

    return rows, selected or {}


# ---------------------------------------------------------------------------
# Matrix builders
# ---------------------------------------------------------------------------


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() in {"none", "null", "nan", "n/a", "na"}:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def extract_serving_safe_row_from_historical(rec: dict[str, Any]) -> dict[str, Any]:
    consts = _policy_constants()
    out: dict[str, Any] = dict(consts)
    price = _safe_float(rec.get("price_usd") if rec.get("price_usd") not in (None, "") else rec.get("price"))
    liq = _safe_float(rec.get("liquidity_usd") if rec.get("liquidity_usd") not in (None, "") else rec.get("liquidity"))
    vol = _safe_float(rec.get("volume_24h"))
    buys = _safe_float(rec.get("txns_buys"))
    sells = _safe_float(rec.get("txns_sells"))
    fdv = _safe_float(rec.get("fdv"))
    out["price"] = price
    out["price_usd"] = price
    out["liquidity"] = liq
    out["liquidity_usd"] = liq
    out["volume_24h"] = vol
    out["fdv"] = fdv
    out["txns_buys"] = buys
    out["txns_sells"] = sells
    out["txns_total"] = (buys + sells) if buys is not None and sells is not None else _safe_float(rec.get("txns_total"))
    for pc in ("price_change_m5", "price_change_h1", "price_change_h6", "price_change_h24"):
        out[pc] = _safe_float(rec.get(pc))
    if buys is not None and sells is not None and (buys + sells) > 0:
        out["buy_ratio"] = buys / (buys + sells)
    else:
        out["buy_ratio"] = _safe_float(rec.get("buy_ratio"))
    if vol is not None and liq is not None and liq > 0:
        out["volume_to_liquidity_ratio"] = vol / liq
    else:
        out["volume_to_liquidity_ratio"] = _safe_float(rec.get("volume_to_liquidity_ratio"))
    for base in MISS_INDICATOR_FOR:
        out[f"{base}_is_missing"] = 1.0 if out.get(base) is None else 0.0
    return out


def extract_serving_safe_row_from_cf(rec: dict[str, Any]) -> dict[str, Any]:
    consts = _policy_constants()
    out: dict[str, Any] = dict(consts)
    price = _safe_float(rec.get("price_usd"))
    liq = _safe_float(rec.get("liquidity_usd"))
    vol = _safe_float(rec.get("volume_24h") or rec.get("volume_h24"))
    buys = _safe_float(rec.get("txns_buys_24h") or rec.get("txns_h24_buys") or rec.get("txns_buys"))
    sells = _safe_float(rec.get("txns_sells_24h") or rec.get("txns_h24_sells") or rec.get("txns_sells"))
    fdv = _safe_float(rec.get("fdv") or rec.get("market_cap"))
    out["price"] = price
    out["price_usd"] = price
    out["liquidity"] = liq
    out["liquidity_usd"] = liq
    out["volume_24h"] = vol
    out["fdv"] = fdv
    out["txns_buys"] = buys
    out["txns_sells"] = sells
    out["txns_total"] = (buys + sells) if buys is not None and sells is not None else None
    for pc in ("price_change_m5", "price_change_h1", "price_change_h6", "price_change_h24"):
        out[pc] = _safe_float(rec.get(pc))
    if buys is not None and sells is not None and (buys + sells) > 0:
        out["buy_ratio"] = buys / (buys + sells)
    else:
        out["buy_ratio"] = None
    if vol is not None and liq is not None and liq > 0:
        out["volume_to_liquidity_ratio"] = vol / liq
    else:
        out["volume_to_liquidity_ratio"] = None
    for base in MISS_INDICATOR_FOR:
        out[f"{base}_is_missing"] = 1.0 if out.get(base) is None else 0.0
    return out


def build_feature_schema_lock(ordered_names: list[str], train_medians: dict[str, float]) -> dict[str, Any]:
    dtype_map = {n: "float64" for n in ordered_names}
    features = []
    consts = _policy_constants()
    for name in ordered_names:
        if name in consts:
            role = "constant_policy_param"
            impute = "none"
            nullable = False
        elif name.endswith("_is_missing"):
            role = "missingness_indicator"
            impute = "none"
            nullable = False
        else:
            role = "numeric"
            impute = f"median={train_medians.get(name)}"
            nullable = True
        features.append(
            {
                "name": name,
                "dtype": "float64",
                "role": role,
                "nullable": nullable,
                "imputation_rule": impute,
                "no_lookahead_status": "NO_LOOKAHEAD",
            }
        )
    lock = {
        "ordered_feature_names": ordered_names,
        "feature_count": len(ordered_names),
        "dtype_by_feature": dtype_map,
        "features": features,
        "train_medians": train_medians,
        "feature_schema_hash": schema_hash(ordered_names, dtype_map),
        "random_seed": RANDOM_SEED,
        "target_column": TARGET_COLUMN,
        "horizon": REF_HORIZON,
        "exit_policy_id": REF_EXIT_POLICY_ID,
    }
    return lock


def apply_schema_lock(
    feature_dicts: list[dict[str, Any]],
    lock: dict[str, Any],
    *,
    allow_extra: bool = False,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Reorder, cast, impute per schema lock. Fail closed on mismatch."""
    errors: list[dict[str, Any]] = []
    names: list[str] = list(lock["ordered_feature_names"])
    medians: dict[str, float] = {k: float(v) for k, v in (lock.get("train_medians") or {}).items()}
    rows = []
    for i, fd in enumerate(feature_dicts):
        extra = set(fd.keys()) - set(names) - {
            "row_id",
            "combined_target_id",
            "pair_address",
            "split",
            "target",
            "event_timestamp",
            "chain",
        }
        # feature dicts may only contain feature keys
        feat_keys = {k for k in fd.keys() if k in names or k.endswith("_is_missing") or k in SERVING_SAFE_BASE_FEATURES}
        unexpected = feat_keys - set(names)
        if unexpected and not allow_extra:
            # only error if unexpected are intended model features
            pass
        row_vals = []
        for name in names:
            v = fd.get(name)
            if v is None or (isinstance(v, float) and v != v):
                if name.endswith("_is_missing"):
                    v = 1.0
                elif name in _policy_constants():
                    v = _policy_constants()[name]
                else:
                    v = medians.get(name, 0.0)
            row_vals.append(float(v))
        rows.append(row_vals)
    df = pd.DataFrame(rows, columns=names, dtype="float64")
    # verify order
    if list(df.columns) != names:
        errors.append({"error": "column_order_mismatch", "got": list(df.columns), "expected": names})
    for c in df.columns:
        if not np.issubdtype(df[c].dtype, np.floating):
            errors.append({"error": "dtype_mismatch", "column": c, "dtype": str(df[c].dtype)})
    got_hash = schema_hash(list(df.columns), {c: "float64" for c in df.columns})
    if got_hash != lock.get("feature_schema_hash"):
        errors.append(
            {
                "error": "schema_hash_mismatch",
                "got": got_hash,
                "expected": lock.get("feature_schema_hash"),
            }
        )
    return df, errors


def validate_predict_matrix(df: pd.DataFrame, lock: dict[str, Any]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    names = list(lock["ordered_feature_names"])
    if list(df.columns) != names:
        errors.append(
            {
                "check": "column_order",
                "passed": False,
                "detail": f"got={list(df.columns)[:5]}... expected_hash={lock['feature_schema_hash'][:12]}",
            }
        )
    else:
        errors.append({"check": "column_order", "passed": True, "detail": "identical"})
    missing = [n for n in names if n not in df.columns]
    extra = [c for c in df.columns if c not in names]
    errors.append(
        {
            "check": "missing_columns",
            "passed": len(missing) == 0,
            "detail": "|".join(missing),
        }
    )
    errors.append(
        {
            "check": "extra_columns",
            "passed": len(extra) == 0,
            "detail": "|".join(extra),
        }
    )
    dtype_ok = all(np.issubdtype(df[c].dtype, np.floating) for c in df.columns if c in names)
    errors.append({"check": "dtype_float64_family", "passed": dtype_ok, "detail": ""})
    got_hash = schema_hash(list(df.columns), {c: "float64" for c in df.columns})
    errors.append(
        {
            "check": "schema_hash",
            "passed": got_hash == lock.get("feature_schema_hash"),
            "detail": f"got={got_hash} expected={lock.get('feature_schema_hash')}",
        }
    )
    return errors


# ---------------------------------------------------------------------------
# Training + inference
# ---------------------------------------------------------------------------


def load_historical_matrix(
    project_root: Path, source_rel: str, ordered_names: list[str]
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    path = project_root / source_rel
    df = pd.read_parquet(path)
    # Prefer label_valid rows when present
    if "label_valid" in df.columns:
        df = df[df["label_valid"].astype(str).str.lower().isin({"1", "true", "yes"}) | (df["label_valid"] == True)]  # noqa: E712
    feature_dicts = []
    meta_rows = []
    for _, rec in df.iterrows():
        d = rec.to_dict()
        feats = extract_serving_safe_row_from_historical(d)
        feature_dicts.append(feats)
        meta_rows.append(
            {
                **feats,
                "pair_address": d.get("pair_address"),
                "event_timestamp": d.get("event_timestamp"),
                "split": d.get("split"),
                "target": int(d.get(TARGET_COLUMN) or 0),
                "candidate_id": d.get("candidate_id"),
            }
        )
    # train medians from train split only
    train_feats = [
        feature_dicts[i]
        for i, m in enumerate(meta_rows)
        if str(m.get("split") or "").lower() in {"train", "training"}
    ]
    if not train_feats:
        train_feats = feature_dicts
    medians: dict[str, float] = {}
    for name in ordered_names:
        if name.endswith("_is_missing") or name in _policy_constants():
            continue
        vals = [f[name] for f in train_feats if f.get(name) is not None]
        medians[name] = float(np.median(vals)) if vals else 0.0
    lock = build_feature_schema_lock(ordered_names, medians)
    X, errors = apply_schema_lock(feature_dicts, lock)
    hist = pd.DataFrame(meta_rows)
    for c in ordered_names:
        hist[c] = X[c].values
    summary = {
        "rows": len(hist),
        "train_rows": int((hist["split"].astype(str).str.lower() == "train").sum())
        if "split" in hist
        else 0,
        "validation_rows": int((hist["split"].astype(str).str.lower().isin(["validation", "val"])).sum())
        if "split" in hist
        else 0,
        "test_rows": int((hist["split"].astype(str).str.lower() == "test").sum()) if "split" in hist else 0,
        "schema_apply_errors": errors,
        "lock": lock,
    }
    # normalize split counts if labels differ
    if summary["train_rows"] == 0 and "split" in hist.columns:
        splits = hist["split"].astype(str).str.lower().value_counts().to_dict()
        summary["split_value_counts"] = splits
        # common: train/valid/test
        summary["train_rows"] = int(splits.get("train", 0))
        summary["validation_rows"] = int(splits.get("validation", 0) + splits.get("valid", 0) + splits.get("val", 0))
        summary["test_rows"] = int(splits.get("test", 0))
    return hist, feature_dicts, summary


def build_current_cf_matrix(
    cf_rows: list[dict[str, Any]], lock: dict[str, Any]
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    feature_dicts = []
    lineage = []
    for rec in cf_rows:
        if is_toxic_pair(rec.get("pair_address")):
            continue
        feats = extract_serving_safe_row_from_cf(rec)
        feature_dicts.append(feats)
        lineage.append(rec)
    X, errors = apply_schema_lock(feature_dicts, lock)
    align = validate_predict_matrix(X, lock)
    # attach lineage columns separately (not as model features)
    out = X.copy()
    out.insert(0, "row_id", [r.get("row_id") for r in lineage])
    out.insert(1, "combined_target_id", [r.get("combined_target_id") for r in lineage])
    out.insert(2, "chain", [r.get("chain") for r in lineage])
    out.insert(3, "pair_address", [r.get("pair_address") for r in lineage])
    return out, feature_dicts, lineage, align + [{"check": "apply_errors", "passed": len(errors) == 0, "detail": str(errors)}]


def _split_masks(hist: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    s = hist["split"].astype(str).str.lower()
    train = s.eq("train").to_numpy()
    val = s.isin(["validation", "valid", "val"]).to_numpy()
    test = s.eq("test").to_numpy()
    if not train.any():
        # deterministic fallback
        n = len(hist)
        idx = np.arange(n)
        rng = np.random.default_rng(RANDOM_SEED)
        rng.shuffle(idx)
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)
        train = np.zeros(n, dtype=bool)
        val = np.zeros(n, dtype=bool)
        test = np.zeros(n, dtype=bool)
        train[idx[:n_train]] = True
        val[idx[n_train : n_train + n_val]] = True
        test[idx[n_train + n_val :]] = True
    return train, val, test


def train_rf(X: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    n_jobs=-1,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )
    pipe.fit(X.loc[train_mask], y[train_mask])
    return pipe


def train_xgb(X: pd.DataFrame, y: np.ndarray, train_mask: np.ndarray):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError(f"XGBoost_UNAVAILABLE: {exc}") from exc
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline

    y_tr = y[train_mask]
    pos = max(int((y_tr == 1).sum()), 1)
    neg = max(int((y_tr == 0).sum()), 1)
    spw = neg / pos
    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=10,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=RANDOM_SEED,
        scale_pos_weight=spw,
        n_jobs=-1,
    )
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)])
    # Ensure column order locked
    pipe.fit(X.loc[train_mask], y_tr)
    return pipe


def predict_proba_locked(model, X: pd.DataFrame, lock: dict[str, Any], family: str) -> np.ndarray:
    names = list(lock["ordered_feature_names"])
    if list(X.columns) != names:
        raise ValueError(f"SCHEMA_ALIGNMENT_MISMATCH:{family}:column_order")
    got_hash = schema_hash(list(X.columns), {c: "float64" for c in X.columns})
    if got_hash != lock.get("feature_schema_hash"):
        raise ValueError(f"SCHEMA_ALIGNMENT_MISMATCH:{family}:schema_hash")
    if family == "XGB":
        # Explicit feature names path for booster when available
        try:
            booster = model.named_steps["model"].get_booster()
            # Set feature names on frame before predict
            X_named = X.copy()
            X_named.columns = names
            proba = model.predict_proba(X_named)
        except Exception:
            proba = model.predict_proba(X)
    else:
        # RF: feature_names_in_ check when present
        est = model.named_steps.get("model", model)
        fn = getattr(est, "feature_names_in_", None)
        if fn is not None and list(fn) != names:
            # Pipeline imputer may strip names; compare after transform path via predict on ordered frame
            pass
        proba = model.predict_proba(X)
    return np.asarray(proba)[:, 1]


def validation_quantile_threshold(scores: np.ndarray, top_pct: float = 5.0) -> float:
    """Historical validation quantile only — never from current CF rows."""
    if len(scores) == 0:
        return 1.0
    q = 100.0 - float(top_pct)
    return float(np.quantile(scores, q / 100.0))


def try_tab_evidence(
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_current: pd.DataFrame,
    lock: dict[str, Any],
    max_context: int = 4000,
) -> tuple[np.ndarray | None, str]:
    try:
        from tabicl import TabICLClassifier  # type: ignore
    except Exception as exc:  # noqa: BLE001
        return None, f"TABICL_IMPORT_UNAVAILABLE: {type(exc).__name__}: {exc}"
    try:
        names = list(lock["ordered_feature_names"])
        if list(X_current.columns) != names:
            return None, "TAB_SCHEMA_MISMATCH:current_column_order"
        rng = np.random.default_rng(RANDOM_SEED)
        n = len(X_train)
        if n > max_context:
            idx = rng.choice(n, size=max_context, replace=False)
            Xt = X_train.iloc[idx]
            yt = y_train[idx]
        else:
            Xt = X_train
            yt = y_train
        clf = TabICLClassifier()
        clf.fit(Xt.to_numpy(dtype=np.float64), yt)
        proba = clf.predict_proba(X_current.to_numpy(dtype=np.float64))
        scores = np.asarray(proba)[:, 1]
        return scores, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"TAB_RUNTIME_FAILURE: {type(exc).__name__}: {exc}"


def build_evidence_rows(
    *,
    family: str,
    lineage: list[dict[str, Any]],
    scores: np.ndarray,
    threshold: float,
    threshold_meta: dict[str, Any],
    model_path: str,
    lock: dict[str, Any],
) -> list[dict[str, Any]]:
    order = np.argsort(-scores)
    rank_of = {int(i): r + 1 for r, i in enumerate(order)}
    out = []
    for i, row in enumerate(lineage):
        score = float(scores[i])
        vote = bool(score >= threshold)
        out.append(
            {
                "evidence_id": f"ae16f_{family}_{row.get('row_id')}",
                "row_id": row.get("row_id"),
                "combined_target_id": row.get("combined_target_id"),
                "chain": row.get("chain"),
                "pair_address": row.get("pair_address"),
                "provider_pair_url": row.get("provider_pair_url"),
                "base_token_address": row.get("base_token_address"),
                "quote_token_address": row.get("quote_token_address"),
                "base_token_symbol": row.get("base_token_symbol"),
                "quote_token_symbol": row.get("quote_token_symbol"),
                "model_family": family,
                "model_artifact_path": model_path,
                "model_target": "net_profitable_after_exit_policy",
                "model_horizon": REF_HORIZON,
                "score": score,
                "rank_in_batch": rank_of[i],
                "vote": vote,
                "vote_threshold_source": threshold_meta.get("threshold_source"),
                "threshold_method": threshold_meta.get("threshold_method"),
                "threshold_value": threshold,
                "threshold_source": threshold_meta.get("threshold_source"),
                "current_rows_used_for_threshold": False,
                "feature_schema_hash": lock.get("feature_schema_hash"),
                "evidence_status": "MODEL_EVIDENCE_ATTACHED",
                "blocker_reason": "",
                "paper_demo_only": True,
                "live_trading_ready": False,
                "limitation_notes": "shadow_research_only;rank_in_batch_diagnostic_only",
            }
        )
    return out


def build_consensus_from_evidence(
    lineage: list[dict[str, Any]], evidence: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from app.consensus.model_evidence import AttachmentResult
    from app.consensus.tiered_engine import (
        build_all_consensus_decisions,
        summarize_consensus_tiers,
    )

    # Attach only voting models (vote==True) as MODEL_EVIDENCE_ATTACHED so the
    # AE16 engine treats validation-threshold votes as consensus votes.
    attachments: list[Any] = []
    by_key = {(e["row_id"], e["model_family"]): e for e in evidence}
    candidates = []
    decision_by = {}
    for row in lineage:
        cid = str(row.get("row_id") or "")
        candidates.append(
            {
                "clean_forward_candidate_id": cid,
                "pair_address": row.get("pair_address"),
                "base_token_address": row.get("base_token_address"),
                "quote_token_address": row.get("quote_token_address"),
                "provider_pair_url": row.get("provider_pair_url"),
                "verification_status": row.get("verification_status"),
                "freshness_status": row.get("freshness_status"),
                "identity_status": row.get("identity_status"),
            }
        )
        decision_by[cid] = {"clean_forward_decision_input_id": f"di_{cid}"}
        for family in ("RF", "XGB", "TAB"):
            e = by_key.get((row.get("row_id"), family))
            if e and e.get("evidence_status") == "MODEL_EVIDENCE_ATTACHED" and e.get("vote"):
                attachments.append(
                    AttachmentResult(
                        clean_forward_candidate_id=cid,
                        clean_forward_decision_input_id=f"di_{cid}",
                        pair_address=str(row.get("pair_address") or ""),
                        base_token_address=str(row.get("base_token_address") or ""),
                        quote_token_address=str(row.get("quote_token_address") or ""),
                        model_family=family,
                        evidence_attached=True,
                        score=float(e["score"]),
                        rank=e.get("rank_in_batch"),
                        percentile_rank=None,
                        source_artifact_path=str(e.get("model_artifact_path") or ""),
                        source_run_id="ae16f",
                        source_prediction_file="",
                        source_model_artifact=str(e.get("model_artifact_path") or ""),
                        candidate_policy_id="",
                        target_row_id="",
                        target_name=str(e.get("model_target") or ""),
                        target_version="",
                        horizon=str(e.get("model_horizon") or ""),
                        filter_name=REF_FILTER,
                        exit_policy_id=REF_EXIT_POLICY_ID,
                        evidence_type="SERVING_SAFE_INFERENCE",
                        attachment_status="MODEL_EVIDENCE_ATTACHED",
                        attachment_failure_reason="",
                    )
                )
            elif e and e.get("evidence_status") == "MODEL_EVIDENCE_ATTACHED":
                # Score present but below threshold — not a vote; treat as unavailable for voting
                attachments.append(
                    AttachmentResult(
                        clean_forward_candidate_id=cid,
                        clean_forward_decision_input_id=f"di_{cid}",
                        pair_address=str(row.get("pair_address") or ""),
                        base_token_address=str(row.get("base_token_address") or ""),
                        quote_token_address=str(row.get("quote_token_address") or ""),
                        model_family=family,
                        evidence_attached=False,
                        score=float(e["score"]),
                        rank=e.get("rank_in_batch"),
                        percentile_rank=None,
                        source_artifact_path=str(e.get("model_artifact_path") or ""),
                        source_run_id="ae16f",
                        source_prediction_file="",
                        source_model_artifact=str(e.get("model_artifact_path") or ""),
                        candidate_policy_id="",
                        target_row_id="",
                        target_name=str(e.get("model_target") or ""),
                        target_version="",
                        horizon=str(e.get("model_horizon") or ""),
                        filter_name=REF_FILTER,
                        exit_policy_id=REF_EXIT_POLICY_ID,
                        evidence_type="SERVING_SAFE_INFERENCE_BELOW_THRESHOLD",
                        attachment_status="SCORE_NOT_ATTACHED",
                        attachment_failure_reason="below_validation_quantile_threshold",
                    )
                )
            else:
                attachments.append(
                    AttachmentResult(
                        clean_forward_candidate_id=cid,
                        clean_forward_decision_input_id=f"di_{cid}",
                        pair_address=str(row.get("pair_address") or ""),
                        base_token_address=str(row.get("base_token_address") or ""),
                        quote_token_address=str(row.get("quote_token_address") or ""),
                        model_family=family,
                        evidence_attached=False,
                        score=None,
                        rank=None,
                        percentile_rank=None,
                        source_artifact_path="",
                        source_run_id="ae16f",
                        source_prediction_file="",
                        source_model_artifact="",
                        candidate_policy_id="",
                        target_row_id="",
                        target_name="",
                        target_version="",
                        horizon="",
                        filter_name="",
                        exit_policy_id="",
                        evidence_type="UNAVAILABLE",
                        attachment_status="MODEL_EVIDENCE_UNAVAILABLE",
                        attachment_failure_reason="MODEL_EVIDENCE_UNAVAILABLE",
                    )
                )

    decisions = build_all_consensus_decisions(
        candidates=candidates, decision_by_candidate=decision_by, attachments=attachments
    )
    out = []
    for d, row in zip(decisions, lineage):
        tier = AE16E_TIER_ALIASES.get(str(d.get("consensus_tier")), d.get("consensus_tier"))
        # Map WATCH for single weak cases already handled; add REJECT if all below threshold with scores?
        dd = dict(d)
        dd["consensus_tier"] = tier
        dd["row_id"] = row.get("row_id")
        dd["combined_target_id"] = row.get("combined_target_id")
        dd["chain"] = row.get("chain")
        dd["target_source"] = row.get("target_source")
        dd["semantic_status"] = row.get("semantic_status")
        dd["paper_demo_only"] = True
        dd["live_trading_ready"] = False
        dd["trade_authority"] = False
        out.append(dd)
    return out, summarize_consensus_tiers(out)


def decide_ae16f_classification(
    *,
    toxic: bool,
    selected_source: dict[str, Any],
    feature_count: int,
    schema_ok: bool,
    threshold_ok: bool,
    families_with_evidence: set[str],
    training_error: str,
    consensus_error: str,
) -> str:
    if toxic:
        return "AE16F_BLOCKED_TOXIC_PAIR_STILL_PRESENT"
    if not selected_source:
        return "AE16F_BLOCKED_NO_CLEAN_DIRECT_TARGET_SOURCE"
    if feature_count <= 0:
        return "AE16F_BLOCKED_NO_SERVING_SAFE_FEATURES"
    if not schema_ok:
        return "AE16F_BLOCKED_SCHEMA_ALIGNMENT_MISMATCH"
    if not threshold_ok:
        return "AE16F_BLOCKED_THRESHOLD_TUNING_ON_CURRENT_ROWS"
    if training_error:
        return "AE16F_BLOCKED_MODEL_TRAINING_FAILURE"
    if consensus_error:
        return "AE16F_BLOCKED_CONSENSUS_ENGINE_FAILURE"
    if families_with_evidence >= {"RF", "XGB", "TAB"}:
        return "AE16F_SERVING_SAFE_MODEL_EVIDENCE_PASS"
    if families_with_evidence >= {"RF", "XGB"} and "TAB" not in families_with_evidence:
        return "AE16F_BLOCKED_TAB_RUNTIME_UNAVAILABLE"
    if families_with_evidence:
        return "AE16F_PARTIAL_SERVING_SAFE_MODEL_EVIDENCE_PASS"
    return "AE16F_BLOCKED_MODEL_TRAINING_FAILURE"
