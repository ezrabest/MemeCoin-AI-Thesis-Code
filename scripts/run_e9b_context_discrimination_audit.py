#!/usr/bin/env python
"""
E9B — Context Feature Discrimination Audit

Purpose:
  Offline, read-only diagnostic audit to test whether context features distinguish
  rare-winner positives from matched controls, using the E9A row-level matched-control contract.

Strict boundaries:
  - No model training
  - No RF/TAB/XGB retraining
  - No runtime changes
  - No UI changes
  - No trading/demo/live changes
  - No SQLite writes
  - No external API calls
  - No Qwen/Gemini/Ollama calls
  - No use of labels, targets, returns, future columns, or simulation columns as features

Outputs:
  reports/e9b_manifest.json
  reports/e9b_summary_for_upload.txt
  data/e9b_feature_scope.csv
  data/e9b_feature_discrimination_matrix.csv
  data/e9b_pair_balanced_feature_summary.csv
  data/e9b_matched_pair_feature_deltas.csv
  data/e9b_feature_missingness_summary.csv
  audits/e9b_label_leakage_audit.csv
  audits/e9b_top_pair_removal_audit.csv
  audits/e9b_leave_one_pair_out_audit.csv
  audits/e9b_matching_strictness_sensitivity.csv
  audits/e9b_timestamp_sensitivity_audit.csv
  audits/e9b_decision_gate.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_E9A_ROOT = ROOT / "data" / "training" / "manual_verified_results" / "phase_e9a_matched_control_contract_20260708_202222"
DEFAULT_E8_ROOT = ROOT / "data" / "training" / "manual_verified_results" / "phase_e8e_rare_winner_context_forensics_20260707_195349"


FORBIDDEN_FEATURE_PATTERNS = [
    r"target",
    r"label",
    r"future",
    r"outcome",
    r"net_return",
    r"return_after",
    r"realized",
    r"pnl",
    r"profit",
    r"profitable",
    r"exit",
    r"simulation",
    r"sim_",
    r"tp_hit",
    r"sl_hit",
    r"time_stop",
    r"max_return",
    r"min_return",
    r"drawdown_after",
    r"price_after",
    r"hit_",
    r"winner",
    r"positive_label",
]

META_OR_ID_PATTERNS = [
    r"^id$",
    r"_id$",
    r"candidate_id",
    r"candidate_policy_id",
    r"target_row_id",
    r"row_id",
    r"pair_address",
    r"pairaddress",
    r"token_address",
    r"contract_address",
    r"address",
    r"timestamp",
    r"time$",
    r"datetime",
    r"date",
    r"source",
    r"split",
    r"filter",
    r"horizon",
    r"exit_policy",
    r"policy",
    r"strictness",
    r"matching",
    r"control",
    r"positive",
    r"group",
]

SAFE_CONTEXT_HINT_PATTERNS = [
    r"liquidity",
    r"volume",
    r"txns",
    r"buys",
    r"sells",
    r"buyer",
    r"seller",
    r"whale",
    r"wallet",
    r"holder",
    r"age",
    r"created",
    r"activity",
    r"sentiment",
    r"rss",
    r"news",
    r"reputation",
    r"risk",
    r"social",
    r"momentum",
    r"volatility",
    r"price_change",
    r"price_step",
    r"ratio",
    r"market_cap",
    r"fdv",
]


@dataclass
class FeatureScope:
    feature_name: str
    status: str
    source: str
    reason: str
    used_for_e9b: bool


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def norm_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv_required(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required {name} file does not exist: {path}")
    return pd.read_csv(path, low_memory=False)


def read_json_optional(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"_json_read_error": str(path)}


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8")


def pattern_match(name: str, patterns: list[str]) -> bool:
    n = norm_col(name)
    return any(re.search(p, n, flags=re.IGNORECASE) for p in patterns)


def is_forbidden_feature_name(name: str) -> bool:
    return pattern_match(name, FORBIDDEN_FEATURE_PATTERNS)


def is_meta_or_id_name(name: str) -> bool:
    return pattern_match(name, META_OR_ID_PATTERNS)


def is_context_hint(name: str) -> bool:
    return pattern_match(name, SAFE_CONTEXT_HINT_PATTERNS)


def coerce_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def numeric_ratio(s: pd.Series) -> float:
    if len(s) == 0:
        return 0.0
    converted = coerce_numeric(s)
    non_missing_orig = s.notna().sum()
    if non_missing_orig == 0:
        return 0.0
    return float(converted.notna().sum() / non_missing_orig)


def classify_column_type(pos: pd.Series, ctrl: pd.Series) -> str:
    both = pd.concat([pos, ctrl], ignore_index=True)
    ratio = numeric_ratio(both)
    if ratio >= 0.85:
        return "numeric"
    return "categorical_or_text"


def cliffs_delta(x: pd.Series, y: pd.Series) -> float | None:
    xv = coerce_numeric(x).dropna().to_numpy(dtype=float)
    yv = coerce_numeric(y).dropna().to_numpy(dtype=float)
    if len(xv) == 0 or len(yv) == 0:
        return None

    y_sorted = np.sort(yv)
    greater = 0
    less = 0
    for value in xv:
        greater += int(np.searchsorted(y_sorted, value, side="left"))
        less += int(len(y_sorted) - np.searchsorted(y_sorted, value, side="right"))

    denom = len(xv) * len(yv)
    if denom == 0:
        return None
    return float((greater - less) / denom)


def safe_median(s: pd.Series) -> float | None:
    vals = coerce_numeric(s).dropna()
    if len(vals) == 0:
        return None
    return float(vals.median())


def safe_mean(s: pd.Series) -> float | None:
    vals = coerce_numeric(s).dropna()
    if len(vals) == 0:
        return None
    return float(vals.mean())


def safe_iqr(s: pd.Series) -> float | None:
    vals = coerce_numeric(s).dropna()
    if len(vals) == 0:
        return None
    q75 = vals.quantile(0.75)
    q25 = vals.quantile(0.25)
    return float(q75 - q25)


def signed_label(delta: float | None) -> str:
    if delta is None or math.isnan(delta):
        return "NA"
    if delta > 0:
        return "HIGHER_IN_POSITIVE"
    if delta < 0:
        return "LOWER_IN_POSITIVE"
    return "NO_DIRECTION"


def find_first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    norm_to_orig = {norm_col(c): c for c in df.columns}
    for c in candidates:
        nc = norm_col(c)
        if nc in norm_to_orig:
            return norm_to_orig[nc]
    return None


def find_col_by_contains(df: pd.DataFrame, must: list[str], must_any: list[str] | None = None) -> str | None:
    for col in df.columns:
        n = norm_col(col)
        if all(m in n for m in must):
            if must_any is None or any(a in n for a in must_any):
                return col
    return None


def detect_pair_col(df: pd.DataFrame) -> str | None:
    return find_first_existing_col(df, [
        "pair_address",
        "pairAddress",
        "pair",
        "token_pair_address",
        "contract_address",
    ])


def detect_target_row_col(df: pd.DataFrame) -> str | None:
    return find_first_existing_col(df, [
        "target_row_id",
        "targetRowId",
        "row_target_id",
    ])


def detect_timestamp_col(df: pd.DataFrame) -> str | None:
    return find_first_existing_col(df, [
        "event_timestamp",
        "timestamp",
        "candidate_timestamp",
        "snapshot_timestamp",
        "created_at",
    ])


def discover_e8_feature_scope(e8_root: Path) -> pd.DataFrame:
    """
    Best-effort discovery of feature-level whitelist/blacklist tables from E8.
    If unavailable, the script falls back to safe common columns and marks that explicitly.
    """
    rows: list[dict[str, Any]] = []

    if not e8_root.exists():
        return pd.DataFrame(rows)

    candidate_files: list[Path] = []
    for p in e8_root.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if p.suffix.lower() not in {".csv", ".json"}:
            continue
        if any(k in name for k in ["feature", "whitelist", "blacklist", "gate", "context"]):
            candidate_files.append(p)

    for path in candidate_files:
        try:
            if path.suffix.lower() == ".csv":
                df = pd.read_csv(path, nrows=5000, low_memory=False)
                cols_norm = {norm_col(c): c for c in df.columns}
                feature_col = None
                for key in ["feature_name", "feature", "column", "column_name", "name"]:
                    if key in cols_norm:
                        feature_col = cols_norm[key]
                        break
                if feature_col is None:
                    continue

                status_col = None
                for key in ["e9_eligibility", "eligibility", "status", "classification", "leakage_status", "decision"]:
                    if key in cols_norm:
                        status_col = cols_norm[key]
                        break

                reason_col = None
                for key in ["reason", "classification_reason", "notes", "comment"]:
                    if key in cols_norm:
                        reason_col = cols_norm[key]
                        break

                for _, r in df.iterrows():
                    feature = str(r.get(feature_col, "")).strip()
                    if not feature or feature.lower() == "nan":
                        continue
                    status = str(r.get(status_col, "unknown")).strip() if status_col else "unknown"
                    reason = str(r.get(reason_col, "")).strip() if reason_col else ""
                    rows.append({
                        "feature_name": feature,
                        "status": status,
                        "reason": reason,
                        "source_file": str(path),
                    })

            elif path.suffix.lower() == ".json":
                obj = json.loads(path.read_text(encoding="utf-8"))
                extracted = []

                if isinstance(obj, list):
                    extracted = obj
                elif isinstance(obj, dict):
                    for key, value in obj.items():
                        if isinstance(value, list) and any(k in key.lower() for k in ["feature", "whitelist", "blacklist", "allowed", "forbidden"]):
                            for item in value:
                                if isinstance(item, dict):
                                    extracted.append(item)
                                else:
                                    extracted.append({"feature_name": item, "status": key})
                        elif isinstance(value, dict) and any(k in key.lower() for k in ["feature", "whitelist", "blacklist"]):
                            for fk, fv in value.items():
                                extracted.append({"feature_name": fk, "status": str(fv)})

                for item in extracted:
                    if not isinstance(item, dict):
                        continue
                    feature = item.get("feature_name") or item.get("feature") or item.get("column") or item.get("name")
                    if feature is None:
                        continue
                    rows.append({
                        "feature_name": str(feature).strip(),
                        "status": str(item.get("e9_eligibility") or item.get("eligibility") or item.get("status") or item.get("classification") or "unknown"),
                        "reason": str(item.get("reason") or item.get("notes") or ""),
                        "source_file": str(path),
                    })
        except Exception:
            continue

    if not rows:
        return pd.DataFrame(rows)

    out = pd.DataFrame(rows).drop_duplicates(subset=["feature_name", "status", "source_file"])
    return out


def build_feature_scope(
    positives: pd.DataFrame,
    controls: pd.DataFrame,
    e8_root: Path,
    explicit_feature_file: Path | None = None,
) -> tuple[list[FeatureScope], bool]:
    common_cols = sorted(set(positives.columns).intersection(set(controls.columns)))
    pos_cols_norm = {norm_col(c): c for c in positives.columns}
    ctrl_cols_norm = {norm_col(c): c for c in controls.columns}

    scope_rows: list[FeatureScope] = []
    fallback_used = False

    discovered = pd.DataFrame()
    if explicit_feature_file is not None:
        if not explicit_feature_file.exists():
            raise FileNotFoundError(f"Explicit feature file not found: {explicit_feature_file}")
        if explicit_feature_file.suffix.lower() == ".csv":
            discovered = pd.read_csv(explicit_feature_file, low_memory=False)
        elif explicit_feature_file.suffix.lower() == ".json":
            obj = json.loads(explicit_feature_file.read_text(encoding="utf-8"))
            discovered = pd.DataFrame(obj if isinstance(obj, list) else obj.get("features", []))
        else:
            raise ValueError("Explicit feature file must be .csv or .json")
    else:
        discovered = discover_e8_feature_scope(e8_root)

    if not discovered.empty:
        cols_norm = {norm_col(c): c for c in discovered.columns}
        feature_col = None
        for key in ["feature_name", "feature", "column", "column_name", "name"]:
            if key in cols_norm:
                feature_col = cols_norm[key]
                break

        status_col = None
        for key in ["e9_eligibility", "eligibility", "status", "classification", "leakage_status", "decision"]:
            if key in cols_norm:
                status_col = cols_norm[key]
                break

        reason_col = None
        for key in ["reason", "classification_reason", "notes", "comment"]:
            if key in cols_norm:
                reason_col = cols_norm[key]
                break

        if feature_col is not None:
            for _, row in discovered.iterrows():
                feature_raw = str(row.get(feature_col, "")).strip()
                if not feature_raw or feature_raw.lower() == "nan":
                    continue

                nf = norm_col(feature_raw)
                actual = pos_cols_norm.get(nf) or ctrl_cols_norm.get(nf)
                status = str(row.get(status_col, "unknown")).strip() if status_col else "unknown"
                reason = str(row.get(reason_col, "")).strip() if reason_col else ""

                allowed_status = any(k in status.lower() for k in [
                    "allowed",
                    "available",
                    "unproven",
                    "research",
                    "e9",
                    "pre_entry",
                    "pre-entry",
                ])
                forbidden_status = any(k in status.lower() for k in [
                    "forbidden",
                    "leakage",
                    "unavailable",
                    "post_entry",
                    "post-entry",
                    "rejected",
                ])

                if actual is None:
                    scope_rows.append(FeatureScope(
                        feature_name=feature_raw,
                        status="NOT_PRESENT_IN_E9A_ROWS",
                        source="E8_DISCOVERED",
                        reason=reason or "Feature listed in E8 scope but missing from E9A positive/control rows.",
                        used_for_e9b=False,
                    ))
                    continue

                if is_forbidden_feature_name(actual) or is_meta_or_id_name(actual) or forbidden_status:
                    scope_rows.append(FeatureScope(
                        feature_name=actual,
                        status="FORBIDDEN_OR_META_EXCLUDED",
                        source="E8_DISCOVERED",
                        reason=reason or f"Excluded by safety pattern or E8 status: {status}",
                        used_for_e9b=False,
                    ))
                    continue

                scope_rows.append(FeatureScope(
                    feature_name=actual,
                    status="USED_E8_SCOPE_RESEARCH_FEATURE",
                    source="E8_DISCOVERED",
                    reason=reason or f"E8 status: {status}",
                    used_for_e9b=bool(allowed_status or not forbidden_status),
                ))

    used = [r for r in scope_rows if r.used_for_e9b]

    if not used:
        fallback_used = True
        for col in common_cols:
            if is_forbidden_feature_name(col):
                scope_rows.append(FeatureScope(
                    feature_name=col,
                    status="FORBIDDEN_LEAKAGE_PATTERN_EXCLUDED",
                    source="SAFE_COMMON_FALLBACK",
                    reason="Column name matched forbidden leakage/outcome/target pattern.",
                    used_for_e9b=False,
                ))
                continue

            if is_meta_or_id_name(col):
                scope_rows.append(FeatureScope(
                    feature_name=col,
                    status="META_OR_ID_EXCLUDED",
                    source="SAFE_COMMON_FALLBACK",
                    reason="Column is ID, timestamp, pair, policy, grouping, or matching metadata.",
                    used_for_e9b=False,
                ))
                continue

            if not is_context_hint(col):
                scope_rows.append(FeatureScope(
                    feature_name=col,
                    status="NON_CONTEXT_COMMON_COLUMN_EXCLUDED",
                    source="SAFE_COMMON_FALLBACK",
                    reason="Column is common but does not match context-feature hint patterns.",
                    used_for_e9b=False,
                ))
                continue

            scope_rows.append(FeatureScope(
                feature_name=col,
                status="USED_SAFE_COMMON_CONTEXT_FALLBACK",
                source="SAFE_COMMON_FALLBACK",
                reason="No usable E8 feature whitelist was discovered; using safe common context-like column.",
                used_for_e9b=True,
            ))

    # Deduplicate while preserving any used row.
    by_name: dict[str, FeatureScope] = {}
    for r in scope_rows:
        key = r.feature_name
        if key not in by_name or (r.used_for_e9b and not by_name[key].used_for_e9b):
            by_name[key] = r

    return list(by_name.values()), fallback_used


def feature_scope_to_df(scope: list[FeatureScope]) -> pd.DataFrame:
    return pd.DataFrame([{
        "feature_name": s.feature_name,
        "status": s.status,
        "source": s.source,
        "reason": s.reason,
        "used_for_e9b": s.used_for_e9b,
    } for s in scope])


def build_leakage_audit(positives: pd.DataFrame, controls: pd.DataFrame, scope: list[FeatureScope]) -> pd.DataFrame:
    common_cols = sorted(set(positives.columns).intersection(set(controls.columns)))
    used_names = {s.feature_name for s in scope if s.used_for_e9b}

    rows = []
    for col in common_cols:
        forbidden = is_forbidden_feature_name(col)
        meta = is_meta_or_id_name(col)
        used = col in used_names
        status = "USED" if used else "EXCLUDED"
        if used and (forbidden or meta):
            status = "CRITICAL_ERROR_USED_FORBIDDEN"
        elif forbidden:
            status = "EXCLUDED_FORBIDDEN_LEAKAGE_PATTERN"
        elif meta:
            status = "EXCLUDED_META_OR_ID"
        elif not used:
            status = "EXCLUDED_NOT_IN_FEATURE_SCOPE"

        rows.append({
            "column": col,
            "used_for_e9b": used,
            "forbidden_pattern": forbidden,
            "meta_or_id_pattern": meta,
            "audit_status": status,
        })
    return pd.DataFrame(rows)


def build_missingness_summary(positives: pd.DataFrame, controls: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        rows.append({
            "feature_name": f,
            "positive_rows": int(len(positives)),
            "control_rows": int(len(controls)),
            "positive_missing": int(positives[f].isna().sum()),
            "control_missing": int(controls[f].isna().sum()),
            "positive_missing_rate": float(positives[f].isna().mean()) if len(positives) else None,
            "control_missing_rate": float(controls[f].isna().mean()) if len(controls) else None,
            "missing_rate_diff_abs": float(abs(positives[f].isna().mean() - controls[f].isna().mean())) if len(positives) and len(controls) else None,
        })
    return pd.DataFrame(rows)


def build_discrimination_matrix(positives: pd.DataFrame, controls: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for f in features:
        ftype = classify_column_type(positives[f], controls[f])
        pos_missing_rate = float(positives[f].isna().mean()) if len(positives) else None
        ctrl_missing_rate = float(controls[f].isna().mean()) if len(controls) else None
        missing_diff_abs = abs(pos_missing_rate - ctrl_missing_rate) if pos_missing_rate is not None and ctrl_missing_rate is not None else None

        if ftype == "numeric":
            delta = cliffs_delta(positives[f], controls[f])
            rows.append({
                "feature_name": f,
                "feature_type": ftype,
                "positive_non_missing": int(coerce_numeric(positives[f]).notna().sum()),
                "control_non_missing": int(coerce_numeric(controls[f]).notna().sum()),
                "positive_missing_rate": pos_missing_rate,
                "control_missing_rate": ctrl_missing_rate,
                "missing_rate_diff_abs": missing_diff_abs,
                "positive_median": safe_median(positives[f]),
                "control_median": safe_median(controls[f]),
                "median_diff": None if safe_median(positives[f]) is None or safe_median(controls[f]) is None else safe_median(positives[f]) - safe_median(controls[f]),
                "positive_mean": safe_mean(positives[f]),
                "control_mean": safe_mean(controls[f]),
                "positive_iqr": safe_iqr(positives[f]),
                "control_iqr": safe_iqr(controls[f]),
                "cliffs_delta": delta,
                "abs_cliffs_delta": None if delta is None else abs(delta),
                "auc_like_from_delta": None if delta is None else (delta + 1.0) / 2.0,
                "direction": signed_label(delta),
            })
        else:
            pos_values = positives[f].astype(str).replace("nan", np.nan)
            ctrl_values = controls[f].astype(str).replace("nan", np.nan)
            pos_top = pos_values.value_counts(dropna=True).head(3).to_dict()
            ctrl_top = ctrl_values.value_counts(dropna=True).head(3).to_dict()
            rows.append({
                "feature_name": f,
                "feature_type": ftype,
                "positive_non_missing": int(pos_values.notna().sum()),
                "control_non_missing": int(ctrl_values.notna().sum()),
                "positive_missing_rate": pos_missing_rate,
                "control_missing_rate": ctrl_missing_rate,
                "missing_rate_diff_abs": missing_diff_abs,
                "positive_median": None,
                "control_median": None,
                "median_diff": None,
                "positive_mean": None,
                "control_mean": None,
                "positive_iqr": None,
                "control_iqr": None,
                "cliffs_delta": None,
                "abs_cliffs_delta": None,
                "auc_like_from_delta": None,
                "direction": "CATEGORICAL_NOT_SCORED",
                "positive_top_values": json.dumps(pos_top, ensure_ascii=False),
                "control_top_values": json.dumps(ctrl_top, ensure_ascii=False),
            })

    return pd.DataFrame(rows)


def build_pair_balanced_summary(
    positives: pd.DataFrame,
    controls: pd.DataFrame,
    features: list[str],
    pair_col: str | None,
) -> pd.DataFrame:
    rows = []
    if pair_col is None or pair_col not in positives.columns:
        return pd.DataFrame([{
            "audit_status": "SKIPPED_NO_POSITIVE_PAIR_COLUMN",
            "feature_name": None,
        }])

    pairs = sorted(positives[pair_col].dropna().astype(str).unique())
    for f in features:
        if classify_column_type(positives[f], controls[f]) != "numeric":
            continue

        pair_effects = []
        for pair in pairs:
            sub = positives[positives[pair_col].astype(str) == pair]
            if len(sub) == 0:
                continue
            d = cliffs_delta(sub[f], controls[f])
            if d is not None:
                pair_effects.append(d)

        if not pair_effects:
            rows.append({
                "feature_name": f,
                "positive_pair_count": len(pairs),
                "pair_effect_count": 0,
                "mean_pair_cliffs_delta": None,
                "min_abs_pair_cliffs_delta": None,
                "sign_consistency_share": None,
                "audit_status": "NO_PAIR_EFFECTS",
            })
            continue

        signs = [1 if x > 0 else -1 if x < 0 else 0 for x in pair_effects]
        dominant_sign = 1 if sum(1 for s in signs if s > 0) >= sum(1 for s in signs if s < 0) else -1
        sign_consistency = sum(1 for s in signs if s == dominant_sign) / len(signs)

        rows.append({
            "feature_name": f,
            "positive_pair_count": len(pairs),
            "pair_effect_count": len(pair_effects),
            "mean_pair_cliffs_delta": float(np.mean(pair_effects)),
            "median_pair_cliffs_delta": float(np.median(pair_effects)),
            "min_abs_pair_cliffs_delta": float(np.min(np.abs(pair_effects))),
            "max_abs_pair_cliffs_delta": float(np.max(np.abs(pair_effects))),
            "sign_consistency_share": float(sign_consistency),
            "audit_status": "OK",
        })

    return pd.DataFrame(rows)


def build_top_pair_removal_audit(
    positives: pd.DataFrame,
    controls: pd.DataFrame,
    features: list[str],
    pair_col: str | None,
) -> pd.DataFrame:
    rows = []
    if pair_col is None or pair_col not in positives.columns:
        return pd.DataFrame([{
            "audit_status": "SKIPPED_NO_POSITIVE_PAIR_COLUMN",
            "feature_name": None,
        }])

    counts = positives[pair_col].astype(str).value_counts()
    if counts.empty:
        return pd.DataFrame([{
            "audit_status": "SKIPPED_NO_POSITIVE_PAIR_VALUES",
            "feature_name": None,
        }])

    top_pair = str(counts.index[0])
    top_pair_rows = int(counts.iloc[0])
    top_pair_share = float(top_pair_rows / len(positives)) if len(positives) else None
    reduced = positives[positives[pair_col].astype(str) != top_pair].copy()

    for f in features:
        if classify_column_type(positives[f], controls[f]) != "numeric":
            continue

        original = cliffs_delta(positives[f], controls[f])
        after = cliffs_delta(reduced[f], controls[f]) if len(reduced) else None
        rows.append({
            "feature_name": f,
            "top_pair": top_pair,
            "top_pair_rows": top_pair_rows,
            "top_pair_share": top_pair_share,
            "positive_rows_after_top_pair_removal": int(len(reduced)),
            "original_cliffs_delta": original,
            "after_top_pair_removal_cliffs_delta": after,
            "original_abs_cliffs_delta": None if original is None else abs(original),
            "after_abs_cliffs_delta": None if after is None else abs(after),
            "same_direction_after_removal": bool(original is not None and after is not None and np.sign(original) == np.sign(after) and np.sign(original) != 0),
            "audit_status": "OK",
        })

    return pd.DataFrame(rows)


def build_leave_one_pair_out_audit(
    positives: pd.DataFrame,
    controls: pd.DataFrame,
    features: list[str],
    pair_col: str | None,
) -> pd.DataFrame:
    rows = []
    if pair_col is None or pair_col not in positives.columns:
        return pd.DataFrame([{
            "audit_status": "SKIPPED_NO_POSITIVE_PAIR_COLUMN",
            "feature_name": None,
        }])

    pairs = sorted(positives[pair_col].dropna().astype(str).unique())
    if not pairs:
        return pd.DataFrame([{
            "audit_status": "SKIPPED_NO_POSITIVE_PAIR_VALUES",
            "feature_name": None,
        }])

    for f in features:
        if classify_column_type(positives[f], controls[f]) != "numeric":
            continue

        original = cliffs_delta(positives[f], controls[f])
        effects = []
        for pair in pairs:
            reduced = positives[positives[pair_col].astype(str) != pair]
            after = cliffs_delta(reduced[f], controls[f]) if len(reduced) else None
            effects.append(after)
            rows.append({
                "feature_name": f,
                "removed_pair": pair,
                "original_cliffs_delta": original,
                "after_pair_removal_cliffs_delta": after,
                "original_abs_cliffs_delta": None if original is None else abs(original),
                "after_abs_cliffs_delta": None if after is None else abs(after),
                "same_direction_after_removal": bool(original is not None and after is not None and np.sign(original) == np.sign(after) and np.sign(original) != 0),
                "audit_status": "OK",
            })

    return pd.DataFrame(rows)


def detect_pair_ref_cols(pairs: pd.DataFrame) -> tuple[str | None, str | None]:
    pos_candidates = [
        "positive_target_row_id",
        "pos_target_row_id",
        "target_row_id_positive",
        "positive_row_target_id",
        "positive_candidate_policy_id",
        "pos_candidate_policy_id",
        "candidate_policy_id_positive",
    ]
    ctrl_candidates = [
        "control_target_row_id",
        "ctrl_target_row_id",
        "target_row_id_control",
        "control_row_target_id",
        "control_candidate_policy_id",
        "ctrl_candidate_policy_id",
        "candidate_policy_id_control",
    ]

    pos_ref = find_first_existing_col(pairs, pos_candidates)
    ctrl_ref = find_first_existing_col(pairs, ctrl_candidates)

    if pos_ref is None:
        pos_ref = find_col_by_contains(pairs, ["positive"], ["target_row_id", "candidate_policy_id", "candidate_id"])
    if ctrl_ref is None:
        ctrl_ref = find_col_by_contains(pairs, ["control"], ["target_row_id", "candidate_policy_id", "candidate_id"])

    return pos_ref, ctrl_ref


def detect_join_key_for_refs(ref_col: str | None, positives: pd.DataFrame, controls: pd.DataFrame) -> str | None:
    if ref_col is None:
        return None
    n = norm_col(ref_col)
    if "target_row_id" in n:
        return detect_target_row_col(positives) if detect_target_row_col(positives) in controls.columns else "target_row_id"
    if "candidate_policy_id" in n:
        c = find_first_existing_col(positives, ["candidate_policy_id"])
        return c if c and c in controls.columns else "candidate_policy_id"
    if "candidate_id" in n:
        c = find_first_existing_col(positives, ["candidate_id"])
        return c if c and c in controls.columns else "candidate_id"
    return None


def build_paired_merged(
    pairs: pd.DataFrame,
    positives: pd.DataFrame,
    controls: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, str]:
    pos_ref, ctrl_ref = detect_pair_ref_cols(pairs)
    if pos_ref is None or ctrl_ref is None:
        return pd.DataFrame(), "SKIPPED_COULD_NOT_DETECT_POSITIVE_CONTROL_REFERENCE_COLUMNS"

    pos_join = detect_join_key_for_refs(pos_ref, positives, controls)
    ctrl_join = detect_join_key_for_refs(ctrl_ref, positives, controls)
    if pos_join is None or ctrl_join is None:
        return pd.DataFrame(), "SKIPPED_COULD_NOT_DETECT_JOIN_KEYS"

    if pos_join not in positives.columns or ctrl_join not in controls.columns:
        return pd.DataFrame(), f"SKIPPED_JOIN_KEYS_NOT_FOUND pos_join={pos_join} ctrl_join={ctrl_join}"

    meta_cols = [c for c in pairs.columns if c not in features]
    pos_cols = [pos_join] + [f for f in features if f in positives.columns]
    ctrl_cols = [ctrl_join] + [f for f in features if f in controls.columns]

    pos_suffix = positives[pos_cols].copy()
    ctrl_suffix = controls[ctrl_cols].copy()

    pos_suffix = pos_suffix.rename(columns={c: f"pos__{c}" for c in pos_suffix.columns if c != pos_join})
    ctrl_suffix = ctrl_suffix.rename(columns={c: f"ctrl__{c}" for c in ctrl_suffix.columns if c != ctrl_join})

    merged = pairs[meta_cols].copy()
    merged["_pos_ref_value"] = pairs[pos_ref].astype(str)
    merged["_ctrl_ref_value"] = pairs[ctrl_ref].astype(str)

    pos_suffix["_pos_ref_value"] = pos_suffix[pos_join].astype(str)
    ctrl_suffix["_ctrl_ref_value"] = ctrl_suffix[ctrl_join].astype(str)

    merged = merged.merge(pos_suffix.drop(columns=[pos_join]), on="_pos_ref_value", how="left")
    merged = merged.merge(ctrl_suffix.drop(columns=[ctrl_join]), on="_ctrl_ref_value", how="left")

    status = f"OK pos_ref={pos_ref} ctrl_ref={ctrl_ref} pos_join={pos_join} ctrl_join={ctrl_join}"
    return merged, status


def build_matched_pair_deltas(
    pairs: pd.DataFrame,
    positives: pd.DataFrame,
    controls: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    merged, status = build_paired_merged(pairs, positives, controls, features)
    if merged.empty:
        empty = pd.DataFrame([{"audit_status": status}])
        return empty, empty.copy(), empty.copy(), status

    rows = []
    strict_rows = []
    time_rows = []

    strict_col = None
    for c in merged.columns:
        n = norm_col(c)
        if "strict" in n or "level" in n or "match_quality" in n:
            strict_col = c
            break

    pos_time_col = None
    ctrl_time_col = None
    for c in merged.columns:
        n = norm_col(c)
        if ("positive" in n or "pos" in n) and ("time" in n or "timestamp" in n):
            pos_time_col = c
        if ("control" in n or "ctrl" in n) and ("time" in n or "timestamp" in n):
            ctrl_time_col = c

    # Try more specific timestamp columns if features did not include them.
    for f in features:
        pass

    later_control = None
    if pos_time_col and ctrl_time_col:
        try:
            pt = pd.to_datetime(merged[pos_time_col], errors="coerce", utc=True)
            ct = pd.to_datetime(merged[ctrl_time_col], errors="coerce", utc=True)
            later_control = ct > pt
        except Exception:
            later_control = None

    for f in features:
        pos_f = f"pos__{f}"
        ctrl_f = f"ctrl__{f}"
        if pos_f not in merged.columns or ctrl_f not in merged.columns:
            continue

        pvals = coerce_numeric(merged[pos_f])
        cvals = coerce_numeric(merged[ctrl_f])
        valid = pvals.notna() & cvals.notna()
        if valid.sum() == 0:
            continue

        delta = pvals[valid] - cvals[valid]
        rows.append({
            "feature_name": f,
            "matched_pair_rows": int(len(merged)),
            "valid_numeric_pairs": int(valid.sum()),
            "median_pair_delta": float(delta.median()),
            "mean_pair_delta": float(delta.mean()),
            "positive_gt_control_share": float((delta > 0).mean()),
            "positive_lt_control_share": float((delta < 0).mean()),
            "pair_delta_abs_median": float(delta.abs().median()),
            "audit_status": status,
        })

        if strict_col is not None:
            tmp = pd.DataFrame({
                "strictness": merged.loc[valid, strict_col].astype(str),
                "delta": delta,
            })
            for strictness, g in tmp.groupby("strictness"):
                strict_rows.append({
                    "feature_name": f,
                    "strictness": strictness,
                    "valid_numeric_pairs": int(len(g)),
                    "median_pair_delta": float(g["delta"].median()),
                    "mean_pair_delta": float(g["delta"].mean()),
                    "positive_gt_control_share": float((g["delta"] > 0).mean()),
                    "audit_status": status,
                })

        if later_control is not None:
            tmp = pd.DataFrame({
                "later_control": later_control[valid].astype(str),
                "delta": delta,
            })
            for later_value, g in tmp.groupby("later_control"):
                time_rows.append({
                    "feature_name": f,
                    "later_control": later_value,
                    "valid_numeric_pairs": int(len(g)),
                    "median_pair_delta": float(g["delta"].median()),
                    "mean_pair_delta": float(g["delta"].mean()),
                    "positive_gt_control_share": float((g["delta"] > 0).mean()),
                    "audit_status": status,
                })

    out = pd.DataFrame(rows) if rows else pd.DataFrame([{"audit_status": "NO_NUMERIC_MATCHED_PAIR_DELTAS", "merge_status": status}])
    strict_out = pd.DataFrame(strict_rows) if strict_rows else pd.DataFrame([{"audit_status": "NO_STRICTNESS_SENSITIVITY_AVAILABLE", "merge_status": status}])
    time_out = pd.DataFrame(time_rows) if time_rows else pd.DataFrame([{"audit_status": "NO_TIMESTAMP_SENSITIVITY_AVAILABLE", "merge_status": status}])
    return out, strict_out, time_out, status


def infer_weak_contract(e9a_root: Path) -> tuple[bool, dict[str, Any]]:
    decision_path = e9a_root / "audits" / "e9a_decision_gate.json"
    strictness_path = e9a_root / "audits" / "e9a_matching_strictness_audit.csv"

    details: dict[str, Any] = {
        "e9a_decision_gate_path": str(decision_path),
        "e9a_matching_strictness_path": str(strictness_path),
        "weak_contract_inferred": True,
        "reason": "Default weak unless proven strong.",
    }

    decision = read_json_optional(decision_path)
    details["e9a_decision_gate"] = decision

    text = json.dumps(decision, ensure_ascii=False).lower()
    if "weak" in text or "research_only" in text:
        details["weak_contract_inferred"] = True
        details["reason"] = "E9A decision gate contains weak/research-only language."
        return True, details

    if strictness_path.exists():
        try:
            df = pd.read_csv(strictness_path, low_memory=False)
            cols = {norm_col(c): c for c in df.columns}
            level_col = None
            count_col = None
            for k in ["strictness", "matching_strictness", "level", "match_level"]:
                if k in cols:
                    level_col = cols[k]
                    break
            for k in ["count", "rows", "matched_pairs"]:
                if k in cols:
                    count_col = cols[k]
                    break

            if level_col and count_col:
                total = float(pd.to_numeric(df[count_col], errors="coerce").fillna(0).sum())
                strong = float(pd.to_numeric(df[df[level_col].astype(str).str.contains("EXACT|STRONG|LEVEL_0", case=False, na=False)][count_col], errors="coerce").fillna(0).sum())
                share = strong / total if total else 0.0
                details["strict_matching_share"] = share
                if share >= 0.70:
                    details["weak_contract_inferred"] = False
                    details["reason"] = "Strict matching share >= 70%."
                    return False, details
                details["weak_contract_inferred"] = True
                details["reason"] = "Strict matching share < 70%."
                return True, details
        except Exception as exc:
            details["strictness_read_error"] = str(exc)

    return bool(details["weak_contract_inferred"]), details


def decide_e9b(
    matrix: pd.DataFrame,
    top_pair: pd.DataFrame,
    leave_one: pd.DataFrame,
    missingness: pd.DataFrame,
    feature_scope_fallback_used: bool,
    weak_contract: bool,
    min_abs_delta: float,
    min_after_abs_delta: float,
    max_missing_diff: float,
) -> dict[str, Any]:
    blockers = []
    warnings = []
    candidates = []

    if feature_scope_fallback_used:
        warnings.append("FEATURE_SCOPE_FALLBACK_USED_NO_EXPLICIT_E8_WHITELIST_DISCOVERED")

    if weak_contract:
        warnings.append("E9A_WEAK_CONTROL_CONTRACT_CAPS_E9B_AT_RESEARCH_ONLY")

    if matrix.empty or "feature_name" not in matrix.columns:
        return {
            "decision": "E9B_FAIL_NO_FEATURES",
            "candidate_features": [],
            "blockers": ["NO_FEATURE_MATRIX"],
            "warnings": warnings,
            "approved_for_modeling": False,
            "approved_for_runtime": False,
        }

    top_map = {}
    if not top_pair.empty and "feature_name" in top_pair.columns:
        for _, r in top_pair.iterrows():
            f = r.get("feature_name")
            if pd.notna(f):
                top_map[str(f)] = r.to_dict()

    loo_group = {}
    if not leave_one.empty and "feature_name" in leave_one.columns:
        for f, g in leave_one.groupby("feature_name"):
            loo_group[str(f)] = g

    miss_map = {}
    if not missingness.empty and "feature_name" in missingness.columns:
        for _, r in missingness.iterrows():
            f = r.get("feature_name")
            if pd.notna(f):
                miss_map[str(f)] = r.to_dict()

    raw_discriminative = []

    for _, r in matrix.iterrows():
        f = str(r.get("feature_name"))
        ftype = str(r.get("feature_type"))
        if ftype != "numeric":
            continue

        delta = r.get("cliffs_delta")
        abs_delta = r.get("abs_cliffs_delta")
        if pd.isna(abs_delta):
            continue

        miss = miss_map.get(f, {})
        missing_diff = miss.get("missing_rate_diff_abs", r.get("missing_rate_diff_abs"))
        missing_ok = missing_diff is not None and not pd.isna(missing_diff) and float(missing_diff) <= max_missing_diff

        if float(abs_delta) >= min_abs_delta:
            raw_discriminative.append(f)

        top = top_map.get(f, {})
        after_abs = top.get("after_abs_cliffs_delta")
        same_after_top = bool(top.get("same_direction_after_removal", False))
        top_survives = (
            after_abs is not None
            and not pd.isna(after_abs)
            and float(after_abs) >= min_after_abs_delta
            and same_after_top
        )

        loo = loo_group.get(f)
        loo_survives = False
        loo_min_abs = None
        loo_same_share = None
        if loo is not None and not loo.empty and "after_abs_cliffs_delta" in loo.columns:
            vals = pd.to_numeric(loo["after_abs_cliffs_delta"], errors="coerce").dropna()
            if len(vals):
                loo_min_abs = float(vals.min())
                same = loo["same_direction_after_removal"].fillna(False).astype(bool)
                loo_same_share = float(same.mean()) if len(same) else 0.0
                loo_survives = loo_min_abs >= min_after_abs_delta and loo_same_share >= 0.67

        if (
            float(abs_delta) >= min_abs_delta
            and missing_ok
            and top_survives
            and loo_survives
        ):
            candidates.append({
                "feature_name": f,
                "cliffs_delta": None if pd.isna(delta) else float(delta),
                "abs_cliffs_delta": float(abs_delta),
                "missing_rate_diff_abs": None if missing_diff is None or pd.isna(missing_diff) else float(missing_diff),
                "after_top_pair_removal_abs_delta": None if after_abs is None or pd.isna(after_abs) else float(after_abs),
                "leave_one_pair_min_abs_delta": loo_min_abs,
                "leave_one_pair_same_direction_share": loo_same_share,
            })

    if not raw_discriminative:
        decision = "E9B_FAIL_NO_DISCRIMINATIVE_CONTEXT_FEATURES"
        blockers.append("NO_FEATURE_REACHED_RAW_DISCRIMINATION_THRESHOLD")
    elif not candidates:
        decision = "E9B_RESEARCH_ONLY_PAIR_OR_MISSINGNESS_DRIVEN"
        blockers.append("RAW_DISCRIMINATION_EXISTS_BUT_DID_NOT_SURVIVE_ROBUSTNESS")
    else:
        if weak_contract or feature_scope_fallback_used:
            decision = "E9B_RESEARCH_ONLY_FEATURE_CANDIDATES"
        else:
            decision = "E9B_CONDITIONAL_PASS_FEATURE_CANDIDATES"

    return {
        "decision": decision,
        "candidate_features": candidates,
        "raw_discriminative_feature_count": len(raw_discriminative),
        "robust_candidate_feature_count": len(candidates),
        "blockers": blockers,
        "warnings": warnings,
        "approved_for_modeling": bool(decision == "E9B_CONDITIONAL_PASS_FEATURE_CANDIDATES"),
        "approved_for_runtime": False,
        "approved_for_training": False,
        "thresholds": {
            "min_abs_cliffs_delta": min_abs_delta,
            "min_after_removal_abs_cliffs_delta": min_after_abs_delta,
            "max_missing_rate_diff_abs": max_missing_diff,
        },
    }


def write_summary(
    path: Path,
    manifest: dict[str, Any],
    decision: dict[str, Any],
    matrix: pd.DataFrame,
    scope_df: pd.DataFrame,
    weak_contract: bool,
) -> None:
    lines = []
    lines.append("Phase / branch name")
    lines.append("")
    lines.append("E9B — Context Feature Discrimination Audit")
    lines.append("")
    lines.append("Run status")
    lines.append("")
    lines.append("COMPLETED")
    lines.append("")
    lines.append("Decision")
    lines.append("")
    lines.append(str(decision.get("decision")))
    lines.append("")
    lines.append("Scope")
    lines.append("")
    lines.append("Offline read-only diagnostic audit.")
    lines.append("No model training, no runtime, no UI, no trading, no SQLite writes, no external APIs, no LLM calls.")
    lines.append("")
    lines.append("Inputs")
    lines.append("")
    for k, v in manifest.get("inputs", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("Feature counts")
    lines.append("")
    lines.append(f"- total feature-scope rows: {len(scope_df)}")
    if "used_for_e9b" in scope_df.columns:
        lines.append(f"- used for E9B: {int(scope_df['used_for_e9b'].sum())}")
    lines.append(f"- discrimination matrix rows: {len(matrix)}")
    lines.append("")
    lines.append("E9A control contract")
    lines.append("")
    lines.append(f"- weak_contract_inferred: {weak_contract}")
    lines.append("")
    lines.append("Candidate features")
    lines.append("")
    candidates = decision.get("candidate_features") or []
    if not candidates:
        lines.append("None.")
    else:
        for c in candidates:
            lines.append(f"- {c.get('feature_name')}: abs_delta={c.get('abs_cliffs_delta')}, after_top_pair_abs={c.get('after_top_pair_removal_abs_delta')}")
    lines.append("")
    lines.append("Warnings")
    lines.append("")
    warnings = decision.get("warnings") or []
    if not warnings:
        lines.append("None.")
    else:
        for w in warnings:
            lines.append(f"- {w}")
    lines.append("")
    lines.append("Blockers")
    lines.append("")
    blockers = decision.get("blockers") or []
    if not blockers:
        lines.append("None.")
    else:
        for b in blockers:
            lines.append(f"- {b}")
    lines.append("")
    lines.append("Final interpretation")
    lines.append("")
    if decision.get("approved_for_modeling"):
        lines.append("E9B conditionally supports moving to E9C/E9D design, but still does not approve runtime.")
    elif candidates:
        lines.append("E9B found research-only feature candidates. Because of weak control contract or fallback scope, this does not approve modeling/runtime.")
    else:
        lines.append("E9B did not find context features robust enough to justify context-enhanced modeling.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="E9B Context Feature Discrimination Audit")
    parser.add_argument("--e9a-root", type=Path, default=DEFAULT_E9A_ROOT)
    parser.add_argument("--e8-root", type=Path, default=DEFAULT_E8_ROOT)
    parser.add_argument("--feature-scope", type=Path, default=None, help="Optional explicit E8 feature whitelist/blacklist file.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--min-abs-cliffs-delta", type=float, default=0.20)
    parser.add_argument("--min-after-removal-abs-cliffs-delta", type=float, default=0.10)
    parser.add_argument("--max-missing-rate-diff-abs", type=float, default=0.25)
    args = parser.parse_args()

    e9a_root = args.e9a_root
    e8_root = args.e8_root
    output_root = args.output_root or (
        ROOT / "data" / "training" / "manual_verified_results" / f"phase_e9b_context_discrimination_{timestamp_slug()}"
    )

    reports_dir = output_root / "reports"
    data_dir = output_root / "data"
    audits_dir = output_root / "audits"
    ensure_dir(reports_dir)
    ensure_dir(data_dir)
    ensure_dir(audits_dir)

    positive_path = e9a_root / "data" / "e9a_positive_rows.csv"
    control_path = e9a_root / "data" / "e9a_control_rows.csv"
    matched_pairs_path = e9a_root / "data" / "e9a_matched_pairs.csv"
    match_group_summary_path = e9a_root / "data" / "e9a_match_group_summary.csv"

    positives = read_csv_required(positive_path, "E9A positive rows")
    controls = read_csv_required(control_path, "E9A control rows")
    pairs = read_csv_required(matched_pairs_path, "E9A matched pairs")
    match_groups = read_csv_required(match_group_summary_path, "E9A match group summary")

    scope, fallback_used = build_feature_scope(
        positives=positives,
        controls=controls,
        e8_root=e8_root,
        explicit_feature_file=args.feature_scope,
    )
    scope_df = feature_scope_to_df(scope)
    write_csv(data_dir / "e9b_feature_scope.csv", scope_df)

    features = [s.feature_name for s in scope if s.used_for_e9b and s.feature_name in positives.columns and s.feature_name in controls.columns]

    leakage_audit = build_leakage_audit(positives, controls, scope)
    write_csv(audits_dir / "e9b_label_leakage_audit.csv", leakage_audit)

    critical_leakage = leakage_audit[leakage_audit["audit_status"] == "CRITICAL_ERROR_USED_FORBIDDEN"]
    if not critical_leakage.empty:
        decision = {
            "decision": "E9B_ABORTED_CRITICAL_LEAKAGE",
            "approved_for_modeling": False,
            "approved_for_runtime": False,
            "approved_for_training": False,
            "critical_columns": critical_leakage["column"].tolist(),
            "created_at_utc": utc_now_iso(),
        }
        write_json(audits_dir / "e9b_decision_gate.json", decision)
        raise RuntimeError(f"Critical leakage: forbidden columns selected as features: {decision['critical_columns']}")

    missingness = build_missingness_summary(positives, controls, features)
    matrix = build_discrimination_matrix(positives, controls, features)

    pair_col = detect_pair_col(positives)
    pair_balanced = build_pair_balanced_summary(positives, controls, features, pair_col)
    top_pair = build_top_pair_removal_audit(positives, controls, features, pair_col)
    leave_one = build_leave_one_pair_out_audit(positives, controls, features, pair_col)
    matched_deltas, strictness_sensitivity, timestamp_sensitivity, matched_merge_status = build_matched_pair_deltas(
        pairs=pairs,
        positives=positives,
        controls=controls,
        features=features,
    )

    write_csv(data_dir / "e9b_feature_discrimination_matrix.csv", matrix)
    write_csv(data_dir / "e9b_pair_balanced_feature_summary.csv", pair_balanced)
    write_csv(data_dir / "e9b_matched_pair_feature_deltas.csv", matched_deltas)
    write_csv(data_dir / "e9b_feature_missingness_summary.csv", missingness)
    write_csv(audits_dir / "e9b_top_pair_removal_audit.csv", top_pair)
    write_csv(audits_dir / "e9b_leave_one_pair_out_audit.csv", leave_one)
    write_csv(audits_dir / "e9b_matching_strictness_sensitivity.csv", strictness_sensitivity)
    write_csv(audits_dir / "e9b_timestamp_sensitivity_audit.csv", timestamp_sensitivity)

    weak_contract, weak_details = infer_weak_contract(e9a_root)

    decision = decide_e9b(
        matrix=matrix,
        top_pair=top_pair,
        leave_one=leave_one,
        missingness=missingness,
        feature_scope_fallback_used=fallback_used,
        weak_contract=weak_contract,
        min_abs_delta=args.min_abs_cliffs_delta,
        min_after_abs_delta=args.min_after_removal_abs_cliffs_delta,
        max_missing_diff=args.max_missing_rate_diff_abs,
    )
    decision["created_at_utc"] = utc_now_iso()
    decision["e9a_weak_contract_details"] = weak_details
    decision["matched_pair_merge_status"] = matched_merge_status
    decision["positive_rows"] = int(len(positives))
    decision["control_rows"] = int(len(controls))
    decision["matched_pair_rows"] = int(len(pairs))
    decision["feature_scope_fallback_used"] = bool(fallback_used)
    decision["used_feature_count"] = int(len(features))
    decision["positive_pair_col"] = pair_col
    if pair_col and pair_col in positives.columns:
        decision["unique_positive_pairs"] = int(positives[pair_col].astype(str).nunique())
        decision["top_positive_pair_share"] = float(positives[pair_col].astype(str).value_counts().iloc[0] / len(positives)) if len(positives) else None

    write_json(audits_dir / "e9b_decision_gate.json", decision)

    manifest = {
        "phase": "E9B",
        "branch_name": "phase_e9b_context_feature_discrimination",
        "created_at_utc": utc_now_iso(),
        "status": "completed",
        "boundaries": {
            "model_training": False,
            "runtime_changes": False,
            "ui_changes": False,
            "trading_changes": False,
            "sqlite_writes": False,
            "external_api_calls": False,
            "llm_calls": False,
            "reservoir_scoring_deployment": False,
        },
        "inputs": {
            "e9a_root": str(e9a_root),
            "e8_root": str(e8_root),
            "positive_rows": str(positive_path),
            "control_rows": str(control_path),
            "matched_pairs": str(matched_pairs_path),
            "match_group_summary": str(match_group_summary_path),
            "feature_scope": str(args.feature_scope) if args.feature_scope else None,
        },
        "outputs": {
            "output_root": str(output_root),
            "manifest": str(reports_dir / "e9b_manifest.json"),
            "summary": str(reports_dir / "e9b_summary_for_upload.txt"),
            "feature_scope": str(data_dir / "e9b_feature_scope.csv"),
            "feature_discrimination_matrix": str(data_dir / "e9b_feature_discrimination_matrix.csv"),
            "pair_balanced_feature_summary": str(data_dir / "e9b_pair_balanced_feature_summary.csv"),
            "matched_pair_feature_deltas": str(data_dir / "e9b_matched_pair_feature_deltas.csv"),
            "feature_missingness_summary": str(data_dir / "e9b_feature_missingness_summary.csv"),
            "label_leakage_audit": str(audits_dir / "e9b_label_leakage_audit.csv"),
            "top_pair_removal_audit": str(audits_dir / "e9b_top_pair_removal_audit.csv"),
            "leave_one_pair_out_audit": str(audits_dir / "e9b_leave_one_pair_out_audit.csv"),
            "matching_strictness_sensitivity": str(audits_dir / "e9b_matching_strictness_sensitivity.csv"),
            "timestamp_sensitivity_audit": str(audits_dir / "e9b_timestamp_sensitivity_audit.csv"),
            "decision_gate": str(audits_dir / "e9b_decision_gate.json"),
        },
        "row_counts": {
            "positive_rows": int(len(positives)),
            "control_rows": int(len(controls)),
            "matched_pairs": int(len(pairs)),
            "match_group_summary_rows": int(len(match_groups)),
        },
        "feature_counts": {
            "feature_scope_rows": int(len(scope_df)),
            "used_features": int(len(features)),
            "feature_scope_fallback_used": bool(fallback_used),
        },
        "decision": decision.get("decision"),
    }

    write_json(reports_dir / "e9b_manifest.json", manifest)
    write_summary(
        path=reports_dir / "e9b_summary_for_upload.txt",
        manifest=manifest,
        decision=decision,
        matrix=matrix,
        scope_df=scope_df,
        weak_contract=weak_contract,
    )

    print(json.dumps({
        "status": "completed",
        "phase": "E9B",
        "output_root": str(output_root),
        "decision": decision.get("decision"),
        "used_feature_count": len(features),
        "approved_for_modeling": decision.get("approved_for_modeling"),
        "approved_for_runtime": decision.get("approved_for_runtime"),
        "summary": str(reports_dir / "e9b_summary_for_upload.txt"),
        "decision_gate": str(audits_dir / "e9b_decision_gate.json"),
    }, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
