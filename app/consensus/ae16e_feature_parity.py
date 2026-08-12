"""AE16E — Direct-target model evidence attachment + strict feature parity.

Fail-closed. No training, backtest, trader.db mutation, wallet, or live trading.
Does not fill missing features with train medians merely to force inference.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.consensus import MODEL_FAMILIES
from app.consensus.evidence_bridge import (
    discover_direct_target_artifacts,
    load_required_features_from_schema,
)
from app.training.direct_target_ids import (
    DEFAULT_EXIT_POLICIES,
    HORIZON_MINUTES,
    resolve_time_stop_minutes,
)

PHASE = "AE16E_DIRECT_TARGET_MODEL_EVIDENCE_ATTACHMENT"
TOXIC_PAIR_ADDRESS = "0x708aEf3736C15FCc1c0b3606C5b9a33fe8656784"

# Reference selected artifacts (direct-target E4 / E5).
REF_RF_STEM = (
    "direct_target_RF_LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL075_FEE0308_TIME_BY_HORIZON"
)
REF_XGB_STEM = (
    "direct_target_XGB_LIQ_5K_HIGH_ACTIVITY_1h_TP20308_SL075_FEE0308_TIME_BY_HORIZON"
)
REF_EXIT_POLICY_ID = "TP20308_SL075_FEE0308_TIME_BY_HORIZON"
REF_HORIZON = "1h"
REF_FILTER = "LIQ_5K_HIGH_ACTIVITY"

SEQUENTIAL_FEATURES = frozenset(
    {
        "gap_detected",
        "price_step_ratio_prev",
        "is_extreme_step_ratio_100x",
        "previous_price_usd",
        "previous_liquidity_usd",
        "price_delta",
        "liquidity_delta",
    }
)

POLICY_CONSTANT_FEATURES = frozenset(
    {"tp_ratio", "sl_ratio", "round_trip_fee_pct", "time_stop_minutes"}
)

PARITY_CLASSES = (
    "AVAILABLE_DIRECT",
    "AVAILABLE_DERIVED_NO_LOOKAHEAD",
    "MISSING_BLOCKING",
    "UNSAFE_LOOKAHEAD",
    "LEGACY_ONLY_REJECTED",
    "CONSTANT_POLICY_PARAM_SAFE",
    "NOT_REQUIRED_BY_SELECTED_ARTIFACT",
)

AE16E_TIER_ALIASES = {
    "XGB_RF_ONLY": "RF_XGB_ONLY",
    "MODEL_DISAGREEMENT": "DISAGREEMENT",
    "RESEARCH_ONLY_WATCH": "WATCH",
    "REJECT_OR_SKIP": "REJECT",
    "CONSENSUS_NOT_COMPUTABLE": "DISAGREEMENT",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _norm_addr(value: Any) -> str:
    return str(value or "").strip().lower()


def is_toxic_pair(value: Any) -> bool:
    return _norm_addr(value) == _norm_addr(TOXIC_PAIR_ADDRESS)


def file_sha256(path: Path, *, max_bytes: int = 64 * 1024 * 1024) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            remaining = max_bytes
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                h.update(chunk)
                remaining -= len(chunk)
        return h.hexdigest()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Feature-name extraction (robust; never aborts discovery)
# ---------------------------------------------------------------------------


def extract_feature_names_from_object(obj: Any) -> tuple[list[str], str, str]:
    """Return (names, method, error). Never raises."""
    try:
        names = getattr(obj, "feature_names_in_", None)
        if names is not None:
            return [str(x) for x in list(names)], "feature_names_in_", ""
    except Exception as exc:  # noqa: BLE001
        pass

    # sklearn Pipeline
    try:
        named = getattr(obj, "named_steps", None)
        if named:
            for step_name, step in named.items():
                try:
                    n = getattr(step, "feature_names_in_", None)
                    if n is not None:
                        return (
                            [str(x) for x in list(n)],
                            f"pipeline.named_steps[{step_name}].feature_names_in_",
                            "",
                        )
                except Exception:  # noqa: BLE001
                    continue
                # nested estimator
                try:
                    est = getattr(step, "estimator_", None) or getattr(step, "estimator", None)
                    if est is not None:
                        n = getattr(est, "feature_names_in_", None)
                        if n is not None:
                            return (
                                [str(x) for x in list(n)],
                                f"pipeline.named_steps[{step_name}].estimator.feature_names_in_",
                                "",
                            )
                except Exception:  # noqa: BLE001
                    continue
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
    else:
        err = ""

    # XGBoost
    try:
        booster = None
        if hasattr(obj, "get_booster"):
            booster = obj.get_booster()
        elif hasattr(obj, "booster"):
            booster = obj.booster
        if booster is not None:
            fn = getattr(booster, "feature_names", None)
            if fn:
                return [str(x) for x in list(fn)], "booster.feature_names", ""
            return [], "booster.feature_names", "feature_names_missing_or_none"
    except AttributeError as exc:
        err = f"AttributeError: {exc}"
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"

    # LightGBM
    try:
        booster = getattr(obj, "booster_", None)
        if booster is not None and hasattr(booster, "feature_name"):
            fn = booster.feature_name()
            if fn:
                return [str(x) for x in list(fn)], "booster_.feature_name()", ""
        if hasattr(obj, "feature_name"):
            fn = obj.feature_name()
            if fn:
                return [str(x) for x in list(fn)], "feature_name()", ""
    except AttributeError as exc:
        err = f"AttributeError: {exc}"
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"

    return [], "unsupported_or_missing", err or "no_feature_names_found"


def extract_feature_names_from_joblib(path: Path) -> dict[str, Any]:
    """Controlled joblib load + feature-name extraction."""
    out: dict[str, Any] = {
        "path": str(path),
        "load_success": False,
        "feature_names": [],
        "feature_count": 0,
        "extraction_method": "",
        "extraction_status": "FAILED",
        "extraction_error": "",
    }
    try:
        import joblib

        obj = joblib.load(path)
        out["load_success"] = True
        names, method, err = extract_feature_names_from_object(obj)
        out["feature_names"] = names
        out["feature_count"] = len(names)
        out["extraction_method"] = method
        if names:
            out["extraction_status"] = "OK"
            out["extraction_error"] = ""
        else:
            out["extraction_status"] = "MISSING_FEATURE_NAMES"
            out["extraction_error"] = err or "feature_names_empty"
    except FileNotFoundError as exc:
        out["extraction_error"] = f"FileNotFoundError: {exc}"
        out["extraction_status"] = "FILE_NOT_FOUND"
    except Exception as exc:  # noqa: BLE001
        out["extraction_error"] = f"{type(exc).__name__}: {exc}"
        out["extraction_status"] = "LOAD_OR_EXTRACT_ERROR"
    return out


def extract_feature_names_from_schema_json(path: Path) -> dict[str, Any]:
    out: dict[str, Any] = {
        "path": str(path),
        "load_success": False,
        "feature_names": [],
        "feature_count": 0,
        "extraction_method": "schema_json",
        "extraction_status": "FAILED",
        "extraction_error": "",
    }
    try:
        names = load_required_features_from_schema(path)
        out["load_success"] = True
        out["feature_names"] = names
        out["feature_count"] = len(names)
        out["extraction_method"] = "feature_columns_in_order|feature_columns|feature_names_in_"
        out["extraction_status"] = "OK" if names else "EMPTY_SCHEMA"
        if not names:
            out["extraction_error"] = "no_feature_list_in_schema"
    except FileNotFoundError as exc:
        out["extraction_error"] = f"FileNotFoundError: {exc}"
        out["extraction_status"] = "FILE_NOT_FOUND"
    except Exception as exc:  # noqa: BLE001
        out["extraction_error"] = f"{type(exc).__name__}: {exc}"
        out["extraction_status"] = "SCHEMA_PARSE_ERROR"
    return out


# ---------------------------------------------------------------------------
# Artifact discovery
# ---------------------------------------------------------------------------


def _policy_constants_for_selected() -> dict[str, float]:
    policy = next(p for p in DEFAULT_EXIT_POLICIES if p["exit_policy_id"] == REF_EXIT_POLICY_ID)
    return {
        "tp_ratio": float(policy["tp_ratio"]),
        "sl_ratio": float(policy["sl_ratio"]),
        "round_trip_fee_pct": float(policy["round_trip_fee_pct"]),
        "time_stop_minutes": float(resolve_time_stop_minutes(REF_HORIZON, policy)),
    }


def discover_ae16e_artifacts(
    project_root: Path,
    *,
    exclude_roots: tuple[str, ...] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Discover RF/XGB/TAB artifacts with robust feature-name extraction audit."""
    base_rows, base_manifest = discover_direct_target_artifacts(
        project_root, exclude_roots=exclude_roots
    )
    extraction_audit: list[dict[str, Any]] = []
    enriched: list[dict[str, Any]] = []

    for row in base_rows:
        erow = dict(row)
        family = str(row.get("artifact_family") or "UNKNOWN")
        atype = str(row.get("artifact_type") or "unknown")
        rel = str(row.get("path") or "")
        abs_path = project_root / rel if rel and not Path(rel).is_absolute() else Path(rel)

        feat_names: list[str] = []
        method = ""
        status = "NOT_ATTEMPTED"
        error = ""
        inference_possible = False
        direct_target_compatible = False
        legacy_only = False

        try:
            if atype == "schema" and abs_path.suffix.lower() == ".json" and abs_path.is_file():
                extracted = extract_feature_names_from_schema_json(abs_path)
                feat_names = extracted["feature_names"]
                method = extracted["extraction_method"]
                status = extracted["extraction_status"]
                error = extracted["extraction_error"]
                direct_target_compatible = "direct_target" in rel.replace("\\", "/").lower()
            elif atype == "model" and abs_path.suffix.lower() in {".joblib", ".pkl"} and abs_path.is_file():
                # Prefer sibling preprocessing schema over deserializing every model.
                schema_sib = abs_path.with_name(abs_path.stem + "_preprocessing.json")
                if schema_sib.is_file():
                    extracted = extract_feature_names_from_schema_json(schema_sib)
                    feat_names = extracted["feature_names"]
                    method = "sibling_preprocessing_json:" + extracted["extraction_method"]
                    status = extracted["extraction_status"]
                    error = extracted["extraction_error"]
                    direct_target_compatible = "direct_target" in abs_path.name.lower()
                    inference_possible = bool(feat_names) and family in ("RF", "XGB")
                else:
                    extracted = extract_feature_names_from_joblib(abs_path)
                    feat_names = extracted["feature_names"]
                    method = extracted["extraction_method"]
                    status = extracted["extraction_status"]
                    error = extracted["extraction_error"]
                    direct_target_compatible = "direct_target" in abs_path.name.lower()
                    inference_possible = bool(feat_names) and family in ("RF", "XGB")
            elif atype == "prediction":
                # Historical predictions — not direct CF ID compatible by default
                legacy_only = True
                direct_target_compatible = False
                status = "PREDICTION_TABLE_LEGACY"
                method = "header_probe_only"
            elif family == "TAB" and "preprocessing" in rel.lower():
                extracted = extract_feature_names_from_schema_json(abs_path)
                feat_names = extracted["feature_names"]
                method = extracted["extraction_method"]
                status = extracted["extraction_status"]
                error = extracted["extraction_error"]
                direct_target_compatible = bool(feat_names)
                inference_possible = False  # no CF-compatible TabICL joblib path
            else:
                status = "SKIPPED_NON_MODEL_SCHEMA"
                method = "n/a"
        except Exception as exc:  # noqa: BLE001 — discovery must not crash
            status = "DISCOVERY_EXCEPTION"
            error = f"{type(exc).__name__}: {exc}"
            method = "exception_handler"

        target_name = ""
        horizon = ""
        name_l = abs_path.name.lower()
        if "direct_target" in name_l:
            target_name = "net_profitable_after_exit_policy"
        for h in HORIZON_MINUTES:
            if f"_{h}_" in name_l or name_l.endswith(f"_{h}") or f"_{h}." in name_l:
                horizon = h
                break
        if "1h" in name_l and not horizon:
            horizon = "1h"

        erow.update(
            {
                "model_family": family if family in MODEL_FAMILIES else family,
                "feature_names": "|".join(feat_names),
                "feature_count": len(feat_names),
                "feature_name_extraction_method": method,
                "feature_name_extraction_status": status,
                "feature_name_extraction_error": error,
                "inference_possible": inference_possible,
                "direct_target_compatible": direct_target_compatible,
                "legacy_only_rejected": legacy_only,
                "target_name": target_name,
                "horizon": horizon,
                "modified_time_utc": row.get("modified_time_utc") or "",
                "exists": bool(row.get("path_exists")),
            }
        )
        enriched.append(erow)
        extraction_audit.append(
            {
                "path": rel,
                "model_family": family,
                "artifact_type": atype,
                "feature_name_extraction_method": method,
                "feature_name_extraction_status": status,
                "feature_name_extraction_error": error,
                "feature_count": len(feat_names),
                "feature_names_preview": "|".join(feat_names[:8]),
            }
        )

    selection = select_reference_artifacts(project_root, enriched)
    manifest = {
        **base_manifest,
        "ae16e_phase": PHASE,
        "selection": selection,
        "extraction_error_count": sum(
            1
            for a in extraction_audit
            if a["feature_name_extraction_status"]
            in {"FAILED", "LOAD_OR_EXTRACT_ERROR", "DISCOVERY_EXCEPTION", "FILE_NOT_FOUND", "SCHEMA_PARSE_ERROR"}
        ),
    }
    return enriched, extraction_audit, manifest


def select_reference_artifacts(
    project_root: Path, discovery_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    """Pick canonical RF/XGB/TAB artifacts; reject unsafe ones explicitly."""
    e4 = (
        project_root
        / "data/training/manual_verified_results"
        / "phase_e4_direct_target_xgb_rf_full_20260630_195312"
        / "models"
    )
    e5a = (
        project_root
        / "data/training/manual_verified_results"
        / "phase_e5_direct_target_tabicl_20260703_203824"
    )

    def _pick_model(stem: str, family: str) -> dict[str, Any]:
        model = e4 / f"{stem}.joblib"
        schema = e4 / f"{stem}_preprocessing.json"
        ok = model.is_file() and schema.is_file()
        reason = ""
        if not model.is_file():
            reason = "MODEL_ARTIFACT_NOT_FOUND"
        elif not schema.is_file():
            reason = "SCHEMA_ARTIFACT_NOT_FOUND"
        else:
            reason = "SELECTED_DIRECT_TARGET_COMPATIBLE"
        return {
            "model_family": family,
            "selected": ok,
            "model_path": str(model.relative_to(project_root)).replace("\\", "/") if model.is_file() else "",
            "schema_path": str(schema.relative_to(project_root)).replace("\\", "/") if schema.is_file() else "",
            "rejected_reason": "" if ok else reason,
            "direct_target_compatible": ok,
            "target_name": "net_profitable_after_exit_policy",
            "horizon": REF_HORIZON,
            "filter": REF_FILTER,
            "exit_policy_id": REF_EXIT_POLICY_ID,
        }

    rf = _pick_model(REF_RF_STEM, "RF")
    xgb = _pick_model(REF_XGB_STEM, "XGB")

    # TAB: schema only; no CF-compatible joblib
    tab_schema = None
    if e5a.is_dir():
        schemas = sorted(e5a.glob("metrics/direct_target_tabicl_preprocessing_*.json"))
        if schemas:
            tab_schema = schemas[0]
    tab = {
        "model_family": "TAB",
        "selected": False,
        "model_path": "",
        "schema_path": str(tab_schema.relative_to(project_root)).replace("\\", "/") if tab_schema else "",
        "rejected_reason": (
            "NO_SAFE_CLEAN_FORWARD_TAB_JOBLIB: TabICL has preprocessing schema but no "
            "local CF-compatible joblib/pt inference artifact"
        ),
        "direct_target_compatible": False,
        "target_name": "net_profitable_after_exit_policy",
        "horizon": "",
        "filter": "",
        "exit_policy_id": "",
        "schema_available": bool(tab_schema),
    }
    if tab_schema:
        feats = load_required_features_from_schema(tab_schema)
        tab["feature_names"] = feats
        tab["required_feature_count"] = len(feats)
        tab["feature_names_extraction_status"] = "OK" if feats else "EMPTY_SCHEMA"
    else:
        tab["feature_names"] = []
        tab["required_feature_count"] = 0
        tab["feature_names_extraction_status"] = "SCHEMA_NOT_FOUND"
        tab["rejected_reason"] = "TAB_SCHEMA_NOT_FOUND"

    # Annotate extraction status for RF/XGB from schema
    for sel in (rf, xgb):
        if sel["schema_path"]:
            feats = load_required_features_from_schema(project_root / sel["schema_path"])
            sel["required_feature_count"] = len(feats)
            sel["feature_names_extraction_status"] = "OK" if feats else "EMPTY_SCHEMA"
            sel["feature_names"] = feats
        else:
            sel["required_feature_count"] = 0
            sel["feature_names_extraction_status"] = "SCHEMA_NOT_FOUND"
            sel["feature_names"] = []

    return {"RF": rf, "XGB": xgb, "TAB": tab, "discovery_row_count": len(discovery_rows)}


# ---------------------------------------------------------------------------
# Clean Forward row loading
# ---------------------------------------------------------------------------


def find_latest_ae16d_clean_forward_rows(project_root: Path) -> Path | None:
    audits = project_root / "data" / "audits"
    if not audits.is_dir():
        return None
    candidates = sorted(
        audits.glob("ae16d_curated_clean_forward_overlay_*/data/ae16d_curated_clean_forward_rows.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        try:
            # Prefer non-empty
            text = path.read_text(encoding="utf-8")
            if text.count("\n") > 1:
                return path
        except OSError:
            continue
    return candidates[0] if candidates else None


def load_clean_forward_rows_used(
    project_root: Path,
    *,
    active_curated_path: Path | None = None,
    ae16d_rows_path: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Persist exact CF rows: AE16D runtime ∩ active curated, toxic excluded, enriched."""
    from app.consensus.serialization import read_csv_dicts

    active_path = active_curated_path or (
        project_root / "data/SeedTargets/clean_forward_curated_ready_targets_active.csv"
    )
    if not active_path.is_absolute():
        active_path = project_root / active_path
    if not active_path.is_file():
        return [], {
            "status": "BLOCKED_RUNTIME_INPUT_MISSING",
            "reason": f"active curated missing: {active_path}",
            "curated_active_targets_loaded": 0,
            "clean_forward_rows_used": 0,
            "toxic_pair_present_anywhere": False,
        }

    active = read_csv_dicts(active_path)
    toxic_in_active = any(
        is_toxic_pair(r.get("pair_address"))
        or is_toxic_pair(r.get("provider_pair_address"))
        or is_toxic_pair(r.get("resolved_pair_address"))
        or is_toxic_pair(r.get("user_supplied_pair_address"))
        for r in active
    )
    active_by_id = {str(r.get("combined_target_id") or ""): r for r in active if r.get("combined_target_id")}
    active_pairs = {
        _norm_addr(r.get("pair_address") or r.get("provider_pair_address") or r.get("resolved_pair_address"))
        for r in active
    }
    active_pairs.discard("")

    cf_path = ae16d_rows_path or find_latest_ae16d_clean_forward_rows(project_root)
    if cf_path is None or not Path(cf_path).is_file():
        return [], {
            "status": "BLOCKED_RUNTIME_INPUT_MISSING",
            "reason": "AE16D clean forward rows not found",
            "curated_active_targets_loaded": len(active),
            "clean_forward_rows_used": 0,
            "toxic_pair_present_anywhere": toxic_in_active,
            "active_curated_path": str(active_path),
        }
    cf_path = Path(cf_path)
    if not cf_path.is_absolute():
        cf_path = project_root / cf_path

    ae16d_rows = read_csv_dicts(cf_path)
    used: list[dict[str, Any]] = []
    toxic_seen = toxic_in_active
    excluded_toxic = 0
    excluded_not_in_active = 0

    for row in ae16d_rows:
        if is_toxic_pair(row.get("pair_address")):
            toxic_seen = True
            excluded_toxic += 1
            continue
        cid = str(row.get("combined_target_id") or "")
        pair = _norm_addr(row.get("pair_address"))
        if cid not in active_by_id and pair not in active_pairs:
            excluded_not_in_active += 1
            continue
        curated = active_by_id.get(cid) or {}
        # Prefer curated match by pair if id miss
        if not curated and pair:
            curated = next(
                (
                    a
                    for a in active
                    if _norm_addr(a.get("pair_address") or a.get("provider_pair_address")) == pair
                ),
                {},
            )
        enriched = dict(row)
        # Enrich no-lookahead provider fields from active curated (exact ID/pair)
        for col in (
            "fdv",
            "market_cap",
            "volume_m5",
            "volume_h1",
            "volume_h6",
            "volume_h24",
            "txns_m5_buys",
            "txns_m5_sells",
            "txns_h1_buys",
            "txns_h1_sells",
            "txns_h6_buys",
            "txns_h6_sells",
            "txns_h24_buys",
            "txns_h24_sells",
            "price_change_m5",
            "price_change_h1",
            "price_change_h6",
            "price_change_h24",
        ):
            if curated.get(col) not in (None, ""):
                enriched[col] = curated.get(col)
        if curated.get("volume_h24") not in (None, "") and not enriched.get("volume_24h"):
            enriched["volume_24h"] = curated.get("volume_h24")
        if curated.get("txns_h24_buys") not in (None, ""):
            enriched["txns_buys_24h"] = curated.get("txns_h24_buys")
        if curated.get("txns_h24_sells") not in (None, ""):
            enriched["txns_sells_24h"] = curated.get("txns_h24_sells")
        # Lineage
        enriched["clean_forward_candidate_id"] = enriched.get("row_id") or cid
        enriched["source_ae16d_path"] = str(cf_path.relative_to(project_root)).replace("\\", "/")
        enriched["source_active_curated_path"] = str(active_path.relative_to(project_root)).replace(
            "\\", "/"
        )
        enriched["paper_demo_only"] = True
        enriched["live_trading_ready"] = False
        used.append(enriched)

    meta = {
        "status": "OK" if used else "BLOCKED_RUNTIME_INPUT_MISSING",
        "reason": "" if used else "zero Clean Forward rows after toxic exclusion / active filter",
        "curated_active_targets_loaded": len(active),
        "ae16d_rows_loaded": len(ae16d_rows),
        "clean_forward_rows_used": len(used),
        "excluded_toxic_count": excluded_toxic,
        "excluded_not_in_active_count": excluded_not_in_active,
        "toxic_pair_present_anywhere": toxic_seen and excluded_toxic == 0 and toxic_in_active,
        "toxic_pair_excluded": excluded_toxic > 0 or not any(is_toxic_pair(r.get("pair_address")) for r in used),
        "active_curated_path": str(active_path.relative_to(project_root)).replace("\\", "/"),
        "ae16d_rows_path": str(cf_path.relative_to(project_root)).replace("\\", "/"),
        "valid_provider_pairs": len(used),
    }
    # toxic_pair_present_anywhere means it survived into used outputs — check used
    meta["toxic_pair_present_anywhere"] = any(is_toxic_pair(r.get("pair_address")) for r in used) or toxic_in_active
    if any(is_toxic_pair(r.get("pair_address")) for r in used):
        meta["status"] = "AE16E_BLOCKED_TOXIC_PAIR_STILL_PRESENT"
    elif toxic_in_active:
        meta["toxic_pair_present_anywhere"] = True
        meta["status"] = "AE16E_BLOCKED_TOXIC_PAIR_STILL_PRESENT"
    else:
        # Toxic may appear in AE16D source but was excluded from used — that is OK
        meta["toxic_pair_present_anywhere"] = False
        meta["toxic_excluded_from_ae16d_source"] = excluded_toxic > 0

    return used, meta


# ---------------------------------------------------------------------------
# Feature parity
# ---------------------------------------------------------------------------


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    s = str(value).strip()
    if s == "" or s.lower() in {"none", "null", "nan", "n/a", "na"}:
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def build_available_field_map(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Union of fields present with at least one non-null across CF rows."""
    present: dict[str, int] = {}
    for r in rows:
        for k, v in r.items():
            if v is None or str(v).strip() == "":
                continue
            present[k] = present.get(k, 0) + 1
    return present


def classify_feature(
    feature: str,
    *,
    available_fields: dict[str, int],
    controlled_snapshot_history: bool,
) -> dict[str, Any]:
    """Classify one required feature exactly once."""
    f = feature
    seq_tokens = ("prev", "prior", "lag", "rolling", "window", "gap", "step", "delta")
    is_sequential = f in SEQUENTIAL_FEATURES or any(t in f.lower() for t in seq_tokens)

    # Policy constants
    if f in POLICY_CONSTANT_FEATURES:
        consts = _policy_constants_for_selected()
        return {
            "feature_name": f,
            "classification": "CONSTANT_POLICY_PARAM_SAFE",
            "source_fields": "selected_exit_policy:" + REF_EXIT_POLICY_ID,
            "formula": f"constant={consts[f]}",
            "no_lookahead_justification": "Exit-policy constant from selected E4 artifact; not market outcome",
            "null_handling": "always_set_from_policy",
            "candidate_specific_or_constant": "constant",
            "requires_sequential_observations": False,
            "controlled_snapshot_history_exists": controlled_snapshot_history,
            "blocker_detail": "",
        }

    # Direct mappings
    direct_map = {
        "price_usd": "price_usd",
        "price": "price_usd",
        "liquidity_usd": "liquidity_usd",
        "liquidity": "liquidity_usd",
        "volume_24h": "volume_24h",
        "fdv": "fdv",
        "price_change_m5": "price_change_m5",
        "price_change_h1": "price_change_h1",
        "price_change_h6": "price_change_h6",
        "price_change_h24": "price_change_h24",
        "txns_buys": "txns_buys_24h",
        "txns_sells": "txns_sells_24h",
    }
    if f in direct_map:
        src = direct_map[f]
        # volume_24h may also come from volume_h24
        alt = "volume_h24" if f == "volume_24h" else None
        ok = available_fields.get(src, 0) > 0 or (alt and available_fields.get(alt, 0) > 0)
        # txns may come from txns_h24_*
        if f == "txns_buys" and available_fields.get("txns_h24_buys", 0) > 0:
            ok = True
            src = "txns_buys_24h|txns_h24_buys"
        if f == "txns_sells" and available_fields.get("txns_h24_sells", 0) > 0:
            ok = True
            src = "txns_sells_24h|txns_h24_sells"
        return {
            "feature_name": f,
            "classification": "AVAILABLE_DIRECT" if ok else "MISSING_BLOCKING",
            "source_fields": src,
            "formula": f"direct={src}",
            "no_lookahead_justification": "Current provider snapshot field" if ok else "",
            "null_handling": "row_null_blocks_inference" if ok else "missing",
            "candidate_specific_or_constant": "candidate_specific",
            "requires_sequential_observations": False,
            "controlled_snapshot_history_exists": controlled_snapshot_history,
            "blocker_detail": "" if ok else f"field {src} absent from Clean Forward rows",
        }

    # Derived no-lookahead from same snapshot
    if f == "txns_total":
        ok = (
            available_fields.get("txns_buys_24h", 0) > 0
            or available_fields.get("txns_h24_buys", 0) > 0
        ) and (
            available_fields.get("txns_sells_24h", 0) > 0
            or available_fields.get("txns_h24_sells", 0) > 0
        )
        return {
            "feature_name": f,
            "classification": "AVAILABLE_DERIVED_NO_LOOKAHEAD" if ok else "MISSING_BLOCKING",
            "source_fields": "txns_buys_24h+txns_sells_24h",
            "formula": "txns_buys + txns_sells",
            "no_lookahead_justification": "Same-snapshot arithmetic; no future data",
            "null_handling": "null_if_either_missing",
            "candidate_specific_or_constant": "candidate_specific",
            "requires_sequential_observations": False,
            "controlled_snapshot_history_exists": controlled_snapshot_history,
            "blocker_detail": "" if ok else "txns buys/sells missing",
        }
    if f == "buy_ratio":
        ok = (
            available_fields.get("txns_buys_24h", 0) > 0
            or available_fields.get("txns_h24_buys", 0) > 0
        ) and (
            available_fields.get("txns_sells_24h", 0) > 0
            or available_fields.get("txns_h24_sells", 0) > 0
        )
        return {
            "feature_name": f,
            "classification": "AVAILABLE_DERIVED_NO_LOOKAHEAD" if ok else "MISSING_BLOCKING",
            "source_fields": "txns_buys_24h,txns_sells_24h",
            "formula": "buys / (buys + sells) if total>0 else null",
            "no_lookahead_justification": "Same-snapshot ratio",
            "null_handling": "null_if_total_zero_or_missing",
            "candidate_specific_or_constant": "candidate_specific",
            "requires_sequential_observations": False,
            "controlled_snapshot_history_exists": controlled_snapshot_history,
            "blocker_detail": "" if ok else "txns buys/sells missing",
        }
    if f == "volume_to_liquidity_ratio":
        ok = (
            available_fields.get("volume_24h", 0) > 0 or available_fields.get("volume_h24", 0) > 0
        ) and available_fields.get("liquidity_usd", 0) > 0
        return {
            "feature_name": f,
            "classification": "AVAILABLE_DERIVED_NO_LOOKAHEAD" if ok else "MISSING_BLOCKING",
            "source_fields": "volume_24h,liquidity_usd",
            "formula": "volume_24h / liquidity_usd if liquidity>0 else null",
            "no_lookahead_justification": "Same-snapshot ratio",
            "null_handling": "null_if_liquidity_zero_or_missing",
            "candidate_specific_or_constant": "candidate_specific",
            "requires_sequential_observations": False,
            "controlled_snapshot_history_exists": controlled_snapshot_history,
            "blocker_detail": "" if ok else "volume or liquidity missing",
        }

    # whale_score: do not pretend wallet-level; training lineage is CLEAN_MODEL_INPUT
    # without proven formula equality to compute_whale_score → fail closed
    if f == "whale_score":
        return {
            "feature_name": f,
            "classification": "MISSING_BLOCKING",
            "source_fields": "",
            "formula": "",
            "no_lookahead_justification": "",
            "null_handling": "not_filled",
            "candidate_specific_or_constant": "candidate_specific",
            "requires_sequential_observations": False,
            "controlled_snapshot_history_exists": controlled_snapshot_history,
            "blocker_detail": (
                "Training whale_score comes from CLEAN_MODEL_INPUT parquet; "
                "cannot prove CF reconstruction equals training feature without unsafe approximation"
            ),
        }

    # entry / snapshot identity — do not invent
    if f in {
        "entry_snapshot_id",
        "entry_price",
        "entry_price_raw",
        "entry_price_verified_30m",
        "entry_price_verified_1h",
        "entry_price_verified_4h",
        "entry_price_verified_8h",
        "entry_price_verified_24h",
    }:
        return {
            "feature_name": f,
            "classification": "MISSING_BLOCKING",
            "source_fields": "",
            "formula": "",
            "no_lookahead_justification": "",
            "null_handling": "not_filled_no_median_imputation",
            "candidate_specific_or_constant": "candidate_specific",
            "requires_sequential_observations": False,
            "controlled_snapshot_history_exists": controlled_snapshot_history,
            "blocker_detail": (
                "Historical entry/snapshot semantics not proven equivalent to current CF provider price; "
                "train-median fill forbidden"
            ),
        }

    # Sequential features
    if is_sequential:
        return {
            "feature_name": f,
            "classification": "MISSING_BLOCKING",
            "source_fields": "",
            "formula": "",
            "no_lookahead_justification": "",
            "null_handling": "not_approximated_from_single_snapshot",
            "candidate_specific_or_constant": "candidate_specific",
            "requires_sequential_observations": True,
            "controlled_snapshot_history_exists": controlled_snapshot_history,
            "blocker_detail": (
                "Requires controlled Clean Forward snapshot history; AE16E has single-snapshot only"
                if not controlled_snapshot_history
                else "Sequential history insufficient or unproven"
            ),
        }

    return {
        "feature_name": f,
        "classification": "MISSING_BLOCKING",
        "source_fields": "",
        "formula": "",
        "no_lookahead_justification": "",
        "null_handling": "not_filled",
        "candidate_specific_or_constant": "unknown",
        "requires_sequential_observations": False,
        "controlled_snapshot_history_exists": controlled_snapshot_history,
        "blocker_detail": "unmapped required feature",
    }


def sequential_feature_questionnaire(
    feature: str, *, controlled_snapshot_history: bool
) -> dict[str, Any]:
    requires_prior = feature in SEQUENTIAL_FEATURES or any(
        t in feature.lower() for t in ("prev", "prior", "lag", "rolling", "window", "gap", "step", "delta")
    )
    if not requires_prior:
        return {
            "feature_name": feature,
            "requires_prior_observations": "NO",
            "has_controlled_cf_snapshot_history": "YES" if controlled_snapshot_history else "NO",
            "history_from_cf_ae16d_path_not_legacy": "N/A",
            "history_timestamped": "N/A",
            "history_strictly_prior_or_current": "N/A",
            "join_key_exact_and_safe": "N/A",
            "minimum_observations_available": "N/A",
            "any_future_leakage": "NO",
            "formula_identical_to_training": "N/A",
            "final_classification": "NOT_SEQUENTIAL",
        }
    # Single snapshot → all critical answers NO/UNKNOWN → MISSING_BLOCKING
    return {
        "feature_name": feature,
        "requires_prior_observations": "YES",
        "has_controlled_cf_snapshot_history": "YES" if controlled_snapshot_history else "NO",
        "history_from_cf_ae16d_path_not_legacy": "NO",
        "history_timestamped": "UNKNOWN",
        "history_strictly_prior_or_current": "UNKNOWN",
        "join_key_exact_and_safe": "UNKNOWN",
        "minimum_observations_available": "NO",
        "any_future_leakage": "UNKNOWN",
        "formula_identical_to_training": "UNKNOWN",
        "final_classification": "MISSING_BLOCKING",
    }


def audit_feature_parity_ae16e(
    *,
    project_root: Path,
    rows: list[dict[str, Any]],
    selection: dict[str, Any],
    controlled_snapshot_history: bool = False,
) -> dict[str, Any]:
    available = build_available_field_map(rows)
    policy_source = (
        "app/training/direct_target_ids.py:DEFAULT_EXIT_POLICIES + "
        f"E4 preprocessing {REF_RF_STEM}_preprocessing.json"
    )
    family_parity: dict[str, list[dict[str, Any]]] = {}
    family_summary: dict[str, Any] = {}
    seq_rows: list[dict[str, Any]] = []
    derived_audit: list[dict[str, Any]] = []
    seen_seq: set[str] = set()

    for family in MODEL_FAMILIES:
        sel = selection.get(family) or {}
        required: list[str] = list(sel.get("feature_names") or [])
        if not required and sel.get("schema_path"):
            required = load_required_features_from_schema(project_root / sel["schema_path"])
            sel["feature_names"] = required

        parity_rows: list[dict[str, Any]] = []
        for feat in required:
            classified = classify_feature(
                feat,
                available_fields=available,
                controlled_snapshot_history=controlled_snapshot_history,
            )
            if classified["classification"] == "CONSTANT_POLICY_PARAM_SAFE":
                classified["source_fields"] = policy_source
            parity_rows.append(classified)
            if classified.get("requires_sequential_observations") or feat in SEQUENTIAL_FEATURES:
                if feat not in seen_seq:
                    seen_seq.add(feat)
                    seq_rows.append(
                        sequential_feature_questionnaire(
                            feat, controlled_snapshot_history=controlled_snapshot_history
                        )
                    )
            if classified["classification"] == "AVAILABLE_DERIVED_NO_LOOKAHEAD":
                derived_audit.append(classified)

        missing_blocking = [
            r for r in parity_rows if r["classification"] == "MISSING_BLOCKING"
        ]
        unsafe = [r for r in parity_rows if r["classification"] == "UNSAFE_LOOKAHEAD"]
        legacy = [r for r in parity_rows if r["classification"] == "LEGACY_ONLY_REJECTED"]
        available_ok = [
            r
            for r in parity_rows
            if r["classification"]
            in {
                "AVAILABLE_DIRECT",
                "AVAILABLE_DERIVED_NO_LOOKAHEAD",
                "CONSTANT_POLICY_PARAM_SAFE",
            }
        ]
        parity_passed = bool(required) and not missing_blocking and not unsafe and not legacy
        # TAB never inference-allowed without CF joblib
        inference_allowed = (
            parity_passed
            and bool(sel.get("selected"))
            and bool(sel.get("model_path"))
            and family in ("RF", "XGB")
        )
        blocker = ""
        if family == "TAB":
            blocker = sel.get("rejected_reason") or "TAB_INFERENCE_UNAVAILABLE"
            inference_allowed = False
            if not required:
                blocker = "TAB_SCHEMA_MISSING"
        elif not sel.get("selected"):
            blocker = sel.get("rejected_reason") or "NO_SAFE_MODEL_ARTIFACT"
        elif missing_blocking:
            blocker = (
                "FEATURE_PARITY_NOT_APPROVED:"
                + "|".join(r["feature_name"] for r in missing_blocking)
            )
        elif not parity_passed:
            blocker = "FEATURE_PARITY_NOT_APPROVED"

        family_parity[family] = parity_rows
        family_summary[family] = {
            "required_features_count": len(required),
            "available_count": len(available_ok),
            "missing_blocking_count": len(missing_blocking),
            "unsafe_lookahead_count": len(unsafe),
            "legacy_rejected_count": len(legacy),
            "feature_parity_passed": parity_passed,
            "inference_allowed": inference_allowed,
            "blocker_reason": blocker,
            "artifact_selected": bool(sel.get("selected")),
            "model_path": sel.get("model_path") or "",
            "schema_path": sel.get("schema_path") or "",
            "feature_names_extraction_status": sel.get("feature_names_extraction_status") or "",
            "missing_blocking_features": [r["feature_name"] for r in missing_blocking],
        }

    return {
        "by_family_rows": family_parity,
        "by_family_summary": family_summary,
        "sequential_questionnaire": seq_rows,
        "derived_feature_audit": derived_audit,
        "available_fields": sorted(available.keys()),
        "controlled_snapshot_history_used": controlled_snapshot_history,
        "legacy_market_snapshots_used": False,
        "unsafe_pair_timestamp_join_used": False,
        "any_inference_allowed": any(v["inference_allowed"] for v in family_summary.values()),
        "any_parity_passed": any(v["feature_parity_passed"] for v in family_summary.values()),
    }


def build_feature_matrix_for_family(
    *,
    rows: list[dict[str, Any]],
    family: str,
    required: list[str],
    parity_rows: list[dict[str, Any]],
    inference_allowed: bool,
) -> list[dict[str, Any]]:
    """Only non-empty when inference_allowed (parity pass). Never fills blocking gaps."""
    if not inference_allowed:
        return []
    class_by = {r["feature_name"]: r for r in parity_rows}
    consts = _policy_constants_for_selected()
    out: list[dict[str, Any]] = []
    for row in rows:
        if is_toxic_pair(row.get("pair_address")):
            continue
        m: dict[str, Any] = {
            "row_id": row.get("row_id") or "",
            "combined_target_id": row.get("combined_target_id") or "",
            "chain": row.get("chain") or "",
            "pair_address": row.get("pair_address") or "",
            "provider_pair_url": row.get("provider_pair_url") or "",
            "base_token_address": row.get("base_token_address") or "",
            "quote_token_address": row.get("quote_token_address") or "",
            "base_token_symbol": row.get("base_token_symbol") or "",
            "quote_token_symbol": row.get("quote_token_symbol") or "",
            "feature_parity_status": "PASSED",
            "no_lookahead_status": "NO_LOOKAHEAD_APPROVED",
            "model_family": family,
        }
        buys = _safe_float(row.get("txns_buys_24h") or row.get("txns_h24_buys"))
        sells = _safe_float(row.get("txns_sells_24h") or row.get("txns_h24_sells"))
        vol = _safe_float(row.get("volume_24h") or row.get("volume_h24"))
        liq = _safe_float(row.get("liquidity_usd"))
        price = _safe_float(row.get("price_usd"))
        for feat in required:
            cls = class_by.get(feat, {}).get("classification")
            if cls == "CONSTANT_POLICY_PARAM_SAFE":
                m[feat] = consts[feat]
            elif feat in {"price", "price_usd"}:
                m[feat] = price
            elif feat in {"liquidity", "liquidity_usd"}:
                m[feat] = liq
            elif feat == "volume_24h":
                m[feat] = vol
            elif feat == "fdv":
                m[feat] = _safe_float(row.get("fdv"))
            elif feat == "txns_buys":
                m[feat] = buys
            elif feat == "txns_sells":
                m[feat] = sells
            elif feat == "txns_total":
                m[feat] = (buys + sells) if buys is not None and sells is not None else None
            elif feat == "buy_ratio":
                if buys is not None and sells is not None and (buys + sells) > 0:
                    m[feat] = buys / (buys + sells)
                else:
                    m[feat] = None
            elif feat == "volume_to_liquidity_ratio":
                m[feat] = (vol / liq) if vol is not None and liq and liq > 0 else None
            elif feat.startswith("price_change_"):
                m[feat] = _safe_float(row.get(feat))
            else:
                # Should not reach for parity-passed set
                m[feat] = None
        out.append(m)
    return out


# ---------------------------------------------------------------------------
# Evidence + consensus
# ---------------------------------------------------------------------------


def run_inference_if_allowed(
    *,
    project_root: Path,
    family: str,
    model_path: str,
    matrix: list[dict[str, Any]],
    required: list[str],
) -> tuple[list[dict[str, Any]], str]:
    """Run existing artifact inference only. Returns (score rows, error)."""
    if not matrix or not model_path:
        return [], "inference_not_allowed_or_empty_matrix"
    abs_model = project_root / model_path
    if not abs_model.is_file():
        return [], "MODEL_ARTIFACT_NOT_FOUND"
    try:
        import joblib
        import numpy as np

        model = joblib.load(abs_model)
        X = []
        meta = []
        for row in matrix:
            vals = []
            ok = True
            for f in required:
                v = row.get(f)
                if v is None or (isinstance(v, float) and v != v):
                    ok = False
                    break
                vals.append(float(v))
            if not ok:
                continue
            X.append(vals)
            meta.append(row)
        if not X:
            return [], "all_rows_had_null_required_features"
        arr = np.asarray(X, dtype=float)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(arr)
            scores = [float(p[1] if len(p) > 1 else p[0]) for p in proba]
        elif hasattr(model, "predict"):
            pred = model.predict(arr)
            scores = [float(x) for x in pred]
        else:
            return [], "model_has_no_predict"
        # rank in batch
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        rank_of = {i: rank + 1 for rank, i in enumerate(order)}
        out = []
        for i, row in enumerate(meta):
            out.append(
                {
                    "row_id": row.get("row_id"),
                    "combined_target_id": row.get("combined_target_id"),
                    "score": scores[i],
                    "rank_in_batch": rank_of[i],
                    "model_family": family,
                    "model_artifact_path": model_path,
                    "model_artifact_hash": file_sha256(abs_model),
                }
            )
        return out, ""
    except Exception as exc:  # noqa: BLE001
        return [], f"{type(exc).__name__}: {exc}"


def build_model_evidence(
    *,
    rows: list[dict[str, Any]],
    parity_summary: dict[str, Any],
    selection: dict[str, Any],
    inference_by_family: dict[str, list[dict[str, Any]]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    evidence: list[dict[str, Any]] = []
    unavailable: list[dict[str, Any]] = []
    fam_summary = parity_summary.get("by_family_summary") or {}

    for family in MODEL_FAMILIES:
        info = fam_summary.get(family) or {}
        sel = selection.get(family) or {}
        scores = {r["row_id"]: r for r in inference_by_family.get(family) or []}
        if info.get("inference_allowed") and scores:
            for row in rows:
                if is_toxic_pair(row.get("pair_address")):
                    continue
                rid = row.get("row_id")
                sc = scores.get(rid)
                if not sc:
                    unavailable.append(
                        {
                            "model_family": family,
                            "row_id": rid,
                            "combined_target_id": row.get("combined_target_id"),
                            "evidence_status": "MODEL_EVIDENCE_UNAVAILABLE",
                            "blocker_reason": "row_missing_from_inference_batch",
                        }
                    )
                    continue
                evidence.append(
                    {
                        "evidence_id": f"ae16e_{family}_{rid}",
                        "row_id": rid,
                        "combined_target_id": row.get("combined_target_id"),
                        "chain": row.get("chain"),
                        "pair_address": row.get("pair_address"),
                        "provider_pair_url": row.get("provider_pair_url"),
                        "base_token_address": row.get("base_token_address"),
                        "quote_token_address": row.get("quote_token_address"),
                        "target_source": row.get("target_source"),
                        "semantic_status": row.get("semantic_status"),
                        "model_family": family,
                        "model_artifact_path": sel.get("model_path") or "",
                        "model_artifact_hash": sc.get("model_artifact_hash") or "",
                        "model_target": sel.get("target_name") or "net_profitable_after_exit_policy",
                        "model_horizon": sel.get("horizon") or REF_HORIZON,
                        "score": sc["score"],
                        "rank_in_batch": sc["rank_in_batch"],
                        "vote": None,  # vote applied at consensus with threshold policy
                        "vote_threshold_source": "POLICY_UNAVAILABLE_RANK_PROXY_NOT_APPLIED",
                        "feature_parity_status": "PASSED",
                        "no_lookahead_status": "NO_LOOKAHEAD_APPROVED",
                        "evidence_status": "MODEL_EVIDENCE_ATTACHED",
                        "blocker_reason": "",
                    }
                )
        else:
            unavailable.append(
                {
                    "model_family": family,
                    "row_id": "",
                    "combined_target_id": "",
                    "evidence_status": "MODEL_EVIDENCE_UNAVAILABLE",
                    "blocker_reason": info.get("blocker_reason")
                    or sel.get("rejected_reason")
                    or "MODEL_EVIDENCE_UNAVAILABLE",
                    "required_features_count": info.get("required_features_count"),
                    "missing_blocking_count": info.get("missing_blocking_count"),
                    "artifact_selected": info.get("artifact_selected"),
                    "model_path": info.get("model_path") or "",
                    "schema_path": info.get("schema_path") or "",
                    "candidates_affected": len(rows),
                }
            )
    return evidence, unavailable


def evidence_to_attachments(
    rows: list[dict[str, Any]], evidence: list[dict[str, Any]]
) -> list[Any]:
    from app.consensus.model_evidence import AttachmentResult

    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for e in evidence:
        if e.get("evidence_status") != "MODEL_EVIDENCE_ATTACHED":
            continue
        rid = str(e.get("row_id") or "")
        by_key[(rid, str(e.get("model_family")))] = e

    attachments: list[Any] = []
    for row in rows:
        rid = str(row.get("row_id") or "")
        cid = rid  # use row_id as clean_forward_candidate_id
        for family in MODEL_FAMILIES:
            e = by_key.get((rid, family))
            if e:
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
                        source_run_id="ae16e",
                        source_prediction_file="",
                        source_model_artifact=str(e.get("model_artifact_path") or ""),
                        candidate_policy_id="",
                        target_row_id="",
                        target_name=str(e.get("model_target") or ""),
                        target_version="",
                        horizon=str(e.get("model_horizon") or ""),
                        filter_name=REF_FILTER,
                        exit_policy_id=REF_EXIT_POLICY_ID,
                        evidence_type="EXISTING_MODEL_INFERENCE",
                        attachment_status="MODEL_EVIDENCE_ATTACHED",
                        attachment_failure_reason="",
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
                        source_run_id="ae16e",
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
    return attachments


def build_ae16e_consensus(
    rows: list[dict[str, Any]], evidence: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    from app.consensus.tiered_engine import (
        build_all_consensus_decisions,
        summarize_consensus_tiers,
    )

    attachments = evidence_to_attachments(rows, evidence)
    candidates = []
    decision_by = {}
    for row in rows:
        cid = str(row.get("row_id") or "")
        candidates.append(
            {
                "clean_forward_candidate_id": cid,
                "pair_address": row.get("pair_address"),
                "base_token_address": row.get("base_token_address"),
                "quote_token_address": row.get("quote_token_address"),
                "provider_pair_url": row.get("provider_pair_url"),
                "provider_payload_hash": "",
                "verification_status": row.get("verification_status"),
                "freshness_status": row.get("freshness_status"),
                "identity_status": row.get("identity_status"),
                "combined_target_id": row.get("combined_target_id"),
                "chain": row.get("chain"),
                "target_source": row.get("target_source"),
                "semantic_status": row.get("semantic_status"),
                "paper_demo_only": True,
                "live_trading_ready": False,
            }
        )
        decision_by[cid] = {"clean_forward_decision_input_id": f"di_{cid}"}

    decisions = build_all_consensus_decisions(
        candidates=candidates,
        decision_by_candidate=decision_by,
        attachments=attachments,
    )
    # Map tier names + preserve lineage
    out = []
    for d, row in zip(decisions, rows):
        tier = str(d.get("consensus_tier") or "")
        tier = AE16E_TIER_ALIASES.get(tier, tier)
        d = dict(d)
        d["consensus_tier"] = tier
        d["row_id"] = row.get("row_id")
        d["combined_target_id"] = row.get("combined_target_id")
        d["chain"] = row.get("chain")
        d["target_source"] = row.get("target_source")
        d["semantic_status"] = row.get("semantic_status")
        d["paper_demo_only"] = True
        d["live_trading_ready"] = False
        d["trade_authority"] = False
        d["wallet_authority"] = False
        out.append(d)
    counts = summarize_consensus_tiers(out)
    # normalize count tier names already applied
    return out, counts


def decide_ae16e_classification(
    *,
    rows_meta: dict[str, Any],
    parity: dict[str, Any],
    evidence: list[dict[str, Any]],
    discovery_crashed: bool = False,
    toxic_in_outputs: bool = False,
) -> str:
    if discovery_crashed:
        return "AE16E_BLOCKED_ARTIFACT_DISCOVERY_CRASH"
    if toxic_in_outputs or rows_meta.get("status") == "AE16E_BLOCKED_TOXIC_PAIR_STILL_PRESENT":
        return "AE16E_BLOCKED_TOXIC_PAIR_STILL_PRESENT"
    if rows_meta.get("status") == "BLOCKED_RUNTIME_INPUT_MISSING" or not rows_meta.get(
        "clean_forward_rows_used"
    ):
        return "AE16E_BLOCKED_RUNTIME_INPUT_MISSING"

    fams_attached = {
        e["model_family"]
        for e in evidence
        if e.get("evidence_status") == "MODEL_EVIDENCE_ATTACHED"
    }
    if fams_attached >= {"RF", "XGB", "TAB"}:
        return "AE16E_MODEL_EVIDENCE_ATTACHMENT_PASS"
    if fams_attached:
        return "AE16E_PARTIAL_MODEL_EVIDENCE_ATTACHMENT_PASS"

    summary = parity.get("by_family_summary") or {}
    any_selected = any((summary.get(f) or {}).get("artifact_selected") for f in ("RF", "XGB"))
    any_parity_gap = any(
        (summary.get(f) or {}).get("missing_blocking_count", 0) > 0
        and (summary.get(f) or {}).get("artifact_selected")
        for f in MODEL_FAMILIES
    )
    if parity.get("unsafe_pair_timestamp_join_used"):
        return "AE16E_BLOCKED_UNSAFE_JOIN_ONLY"
    if any_parity_gap or any_selected:
        return "AE16E_BLOCKED_FEATURE_PARITY_GAP"
    return "AE16E_BLOCKED_NO_SAFE_MODEL_ARTIFACT"


def assert_no_toxic_in_outputs(rows_lists: list[list[dict[str, Any]]]) -> bool:
    for rows in rows_lists:
        for r in rows:
            for key in ("pair_address", "provider_pair_address", "resolved_pair_address"):
                if is_toxic_pair(r.get(key)):
                    return True
    return False
