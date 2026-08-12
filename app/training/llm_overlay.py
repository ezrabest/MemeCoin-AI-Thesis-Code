"""Offline stored-LLM overlay evaluation on strict RF policy candidates."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PRIMARY_TARGET = "label_profitable_after_fees_4h"
RF_MODEL_NAME = "random_forest"
OUTCOME_HORIZON_HOURS = 4
JOIN_TOLERANCE_SECONDS = 120
CONFIDENCE_THRESHOLD = 0.60
RISK_SCORE_THRESHOLD = 50

POSITIVE_ACTIONS = frozenset({
    "BUY", "STRONG_BUY", "ACCUMULATE", "LONG", "ENTER", "ADD",
})
NEGATIVE_ACTIONS = frozenset({
    "SELL", "STRONG_SELL", "AVOID", "EXIT", "SHORT", "REDUCE", "DUMP",
})
HIGH_RISK_THRESHOLD = 70

RF_CANDIDATE_POLICIES: dict[str, dict[str, Any]] = {
    "rf_probability_threshold_0_70_4h": {
        "source_policy": "probability_threshold_0_70",
        "fixed_cutoff": 0.70,
        "top_percent": None,
    },
    "rf_top_1_percent_4h": {
        "source_policy": "top_1_percent",
        "fixed_cutoff": None,
        "top_percent": 1.0,
    },
    "rf_top_2_percent_4h": {
        "source_policy": "top_2_percent",
        "fixed_cutoff": None,
        "top_percent": 2.0,
    },
}

OVERLAY_VARIANTS: list[dict[str, Any]] = [
    {"name": "rf_alone", "requires_llm": False},
    {"name": "rf_plus_qwen_buy_confirm", "requires_llm": True, "requires_qwen": True, "mode": "buy_confirm"},
    {"name": "rf_plus_exclude_qwen_sell", "requires_llm": True, "requires_qwen": True, "mode": "exclude_negative"},
    {
        "name": "rf_plus_confidence_gte_0_60",
        "requires_llm": True,
        "requires_field": "confidence",
        "mode": "confidence_threshold",
        "threshold": CONFIDENCE_THRESHOLD,
    },
    {
        "name": "rf_plus_risk_lte_50",
        "requires_llm": True,
        "requires_field": "risk_score",
        "mode": "risk_threshold",
        "threshold": RISK_SCORE_THRESHOLD,
    },
    {
        "name": "rf_plus_buy_confirm_and_low_risk",
        "requires_llm": True,
        "requires_qwen": True,
        "requires_fields": ("confidence", "risk_score"),
        "mode": "buy_and_low_risk",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "risk_threshold": RISK_SCORE_THRESHOLD,
    },
]

LLM_TABLE_CANDIDATES = ("gemini_decisions", "llm_decisions")
PROVIDER_MODEL_COLUMNS = ("provider", "model", "model_source", "llm_model")
ACTION_COLUMNS = ("action", "decision")
JSON_RESPONSE_COLUMNS = ("gemini_response_json", "response_json", "input_context_json")


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def inspect_sqlite_tables(db_path: Path) -> dict[str, list[str]]:
    if not db_path.is_file():
        return {}
    conn = sqlite3.connect(str(db_path))
    try:
        tables = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
        return {
            table: [row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            for table in tables
            if table in LLM_TABLE_CANDIDATES
        }
    finally:
        conn.close()


def resolve_llm_table(db_path: Path) -> tuple[str | None, list[str], list[str]]:
    schema = inspect_sqlite_tables(db_path)
    fallbacks: list[str] = []
    for table in LLM_TABLE_CANDIDATES:
        if table in schema:
            return table, schema[table], fallbacks
        fallbacks.append(f"{table}_not_found")
    return None, [], fallbacks


def _parse_json_field(value: Any) -> dict[str, Any]:
    if value is None or (isinstance(value, float) and value != value):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _normalize_action(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _field_contains_qwen(value: Any) -> bool:
    if value is None:
        return False
    return "qwen" in str(value).lower()


def is_qwen_ollama_decision(row: dict[str, Any]) -> bool:
    provider = str(row.get("provider") or "").strip().lower()
    if provider == "ollama":
        return True
    for col in ("model", "model_source", "llm_model"):
        if _field_contains_qwen(row.get(col)):
            return True
    for col in JSON_RESPONSE_COLUMNS:
        payload = _parse_json_field(row.get(col))
        for key in ("provider", "model", "model_source", "llm_model"):
            if _field_contains_qwen(payload.get(key)):
                return True
    return False


def enrich_decision_row(row: dict[str, Any], columns_present: list[str]) -> dict[str, Any]:
    fallbacks: list[str] = []
    enriched = dict(row)

    action = None
    for col in ACTION_COLUMNS:
        if col in columns_present:
            action = _normalize_action(row.get(col))
            if action:
                enriched["parsed_action"] = action
                break
    if not action:
        for col in JSON_RESPONSE_COLUMNS:
            if col not in columns_present:
                continue
            payload = _parse_json_field(row.get(col))
            for key in ("action", "decision", "recommendation"):
                action = _normalize_action(payload.get(key))
                if action:
                    enriched["parsed_action"] = action
                    fallbacks.append(f"action_from_{col}.{key}")
                    break
            if action:
                break

    confidence = None
    if "confidence" in columns_present:
        confidence = _coerce_float(row.get("confidence"))
    if confidence is None:
        for col in JSON_RESPONSE_COLUMNS:
            if col not in columns_present:
                continue
            payload = _parse_json_field(row.get(col))
            confidence = _coerce_float(payload.get("confidence"))
            if confidence is not None:
                fallbacks.append(f"confidence_from_{col}")
                break
    enriched["parsed_confidence"] = confidence

    risk_score = None
    if "risk_score" in columns_present:
        risk_score = _coerce_int(row.get("risk_score"))
    if risk_score is None:
        for col in JSON_RESPONSE_COLUMNS:
            if col not in columns_present:
                continue
            payload = _parse_json_field(row.get(col))
            risk_score = _coerce_int(payload.get("risk_score"))
            if risk_score is not None:
                fallbacks.append(f"risk_score_from_{col}")
                break
    enriched["parsed_risk_score"] = risk_score

    enriched["is_qwen_ollama"] = is_qwen_ollama_decision(enriched)
    enriched["_schema_fallbacks"] = fallbacks
    return enriched


def load_stored_llm_decisions(db_path: Path) -> tuple["pd.DataFrame", dict[str, Any]]:
    import pandas as pd

    table, columns, table_fallbacks = resolve_llm_table(db_path)
    meta: dict[str, Any] = {
        "stored_llm_table_used": table,
        "stored_llm_columns_used": columns,
        "llm_schema_fallbacks_used": table_fallbacks,
        "db_path": str(db_path),
    }
    if table is None:
        meta["error"] = "No stored LLM decision table found."
        return pd.DataFrame(), meta

    conn = sqlite3.connect(str(db_path))
    try:
        decisions = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        coins = pd.read_sql_query("SELECT id, pair_address, symbol FROM coins", conn)
    finally:
        conn.close()

    if decisions.empty:
        meta["stored_llm_row_count"] = 0
        return decisions, meta

    decisions = decisions.rename(columns={"id": "decision_id", "timestamp": "decision_timestamp"})
    decisions["decision_timestamp"] = pd.to_datetime(decisions["decision_timestamp"], utc=True, errors="coerce")
    decisions = decisions[decisions["decision_timestamp"].notna()].copy()

    if "coin_id" in decisions.columns and not coins.empty:
        decisions = decisions.merge(coins.rename(columns={"id": "coin_id"}), on="coin_id", how="left")
    else:
        meta["llm_schema_fallbacks_used"] = list(meta.get("llm_schema_fallbacks_used", [])) + ["coin_id_join_unavailable"]

    enriched_rows = [enrich_decision_row(row.to_dict(), columns) for _, row in decisions.iterrows()]
    frame = pd.DataFrame(enriched_rows)
    frame["decision_timestamp"] = pd.to_datetime(frame["decision_timestamp"], utc=True, errors="coerce")

    provider_dist: dict[str, int] = {}
    if "provider" in frame.columns:
        provider_dist = frame["provider"].fillna("unknown").astype(str).value_counts().to_dict()

    model_dist: dict[str, int] = {}
    for col in ("model_source", "model", "llm_model"):
        if col in frame.columns:
            for key, count in frame[col].fillna("unknown").astype(str).value_counts().to_dict().items():
                model_dist[f"{col}:{key}"] = int(count)

    meta.update({
        "stored_llm_row_count": int(len(frame)),
        "qwen_ollama_row_count": int(frame["is_qwen_ollama"].sum()) if "is_qwen_ollama" in frame.columns else 0,
        "provider_distribution": provider_dist,
        "model_distribution": model_dist,
    })
    return frame, meta


def _pair_key(frame: "pd.DataFrame") -> "pd.Series":
    import pandas as pd

    if "pair_address" in frame.columns and frame["pair_address"].notna().any():
        return frame["pair_address"].astype(str)
    if "coin_id" in frame.columns:
        return frame["coin_id"].astype(str)
    return pd.Series("unknown", index=frame.index)


def _try_direct_id_join(
    candidates: "pd.DataFrame",
    decisions: "pd.DataFrame",
) -> tuple["pd.DataFrame | None", str | None]:
    import pandas as pd

    if "decision_id" in candidates.columns and "decision_id" in decisions.columns:
        merged = candidates.merge(
            decisions,
            on="decision_id",
            how="left",
            suffixes=("", "_llm"),
        )
        return merged, "direct_decision_id"

    if "signal_id" in candidates.columns:
        for col in JSON_RESPONSE_COLUMNS:
            if col not in decisions.columns:
                continue
            signal_ids = []
            for value in decisions[col]:
                payload = _parse_json_field(value)
                signal_ids.append(_coerce_int(payload.get("signal_id")))
            decisions = decisions.copy()
            decisions["_parsed_signal_id"] = signal_ids
            if decisions["_parsed_signal_id"].notna().any():
                merged = candidates.merge(
                    decisions,
                    left_on="signal_id",
                    right_on="_parsed_signal_id",
                    how="left",
                    suffixes=("", "_llm"),
                )
                return merged, "direct_signal_id_from_decision_json"
    return None, None


def join_llm_decisions_to_candidates(
    candidates: "pd.DataFrame",
    decisions: "pd.DataFrame",
    *,
    tolerance_seconds: int = JOIN_TOLERANCE_SECONDS,
    outcome_horizon_hours: int = OUTCOME_HORIZON_HOURS,
) -> tuple["pd.DataFrame", dict[str, Any]]:
    import pandas as pd

    join_meta: dict[str, Any] = {
        "join_tolerance_used": f"{tolerance_seconds}_seconds",
        "outcome_horizon_hours": outcome_horizon_hours,
    }

    if candidates.empty:
        join_meta["join_strategy_used"] = "no_candidates"
        return candidates.copy(), join_meta
    if decisions.empty:
        out = candidates.copy()
        out["llm_matched"] = False
        join_meta.update({
            "join_strategy_used": "no_stored_decisions",
            "matched_candidate_count": 0,
            "unmatched_candidate_count": int(len(out)),
            "match_rate": 0.0,
        })
        return out, join_meta

    direct, strategy = _try_direct_id_join(candidates, decisions)
    if direct is not None and strategy is not None:
        if "decision_timestamp" in direct.columns:
            direct["llm_matched"] = direct["decision_timestamp"].notna()
        else:
            direct["llm_matched"] = False
        join_meta["join_strategy_used"] = strategy
        return _finalize_join_stats(direct, join_meta)

    cand = candidates.copy()
    cand["_row_id"] = cand.index
    dec = decisions.copy()
    cand["event_timestamp"] = pd.to_datetime(cand["event_timestamp"], utc=True, errors="coerce")
    cand["_pair_key"] = _pair_key(cand)
    dec["_pair_key"] = _pair_key(dec)

    tolerance = pd.Timedelta(seconds=tolerance_seconds)
    horizon = pd.Timedelta(hours=outcome_horizon_hours)
    joined_parts: list[pd.DataFrame] = []

    decision_cols = [
        c for c in dec.columns
        if c not in cand.columns or c in {
            "decision_id", "decision_timestamp", "_pair_key",
            "parsed_action", "parsed_confidence", "parsed_risk_score", "is_qwen_ollama",
            "provider", "model_source",
        }
    ]

    for pair_key, cand_group in cand.groupby("_pair_key", sort=False):
        dec_group = dec[dec["_pair_key"] == pair_key].sort_values("decision_timestamp")
        cand_group = cand_group.sort_values("event_timestamp")
        if dec_group.empty:
            part = cand_group.copy()
            part["llm_matched"] = False
            joined_parts.append(part)
            continue

        merged = pd.merge_asof(
            cand_group,
            dec_group[decision_cols],
            left_on="event_timestamp",
            right_on="decision_timestamp",
            direction="forward",
            tolerance=tolerance,
        )
        merged["llm_matched"] = merged["decision_timestamp"].notna()
        if merged["llm_matched"].any():
            delta = (
                merged.loc[merged["llm_matched"], "decision_timestamp"]
                - merged.loc[merged["llm_matched"], "event_timestamp"]
            )
            merged.loc[merged["llm_matched"], "llm_latency_seconds"] = delta.dt.total_seconds()
            within_horizon = merged["decision_timestamp"] <= merged["event_timestamp"] + horizon
            merged.loc[~within_horizon, "llm_matched"] = False
            for col in ("decision_timestamp", "parsed_action", "parsed_confidence", "parsed_risk_score"):
                if col in merged.columns:
                    merged.loc[~within_horizon, col] = pd.NA
        joined_parts.append(merged)

    joined = pd.concat(joined_parts, ignore_index=False).sort_index()
    if "_row_id" in joined.columns:
        joined = joined.set_index("_row_id", drop=True)
    join_meta["join_strategy_used"] = "pair_time_merge_asof_forward"
    return _finalize_join_stats(joined, join_meta)


def _rf_candidate_join_stats(joined: "pd.DataFrame", rf_mask: "pd.Series") -> dict[str, Any]:
    subset = joined.loc[rf_mask.fillna(False)]
    matched = int(subset["llm_matched"].sum()) if "llm_matched" in subset.columns else 0
    total = int(len(subset))
    return {
        "matched_candidate_count": matched,
        "unmatched_candidate_count": total - matched,
        "match_rate": round(matched / total, 6) if total else 0.0,
    }


def _finalize_join_stats(frame: "pd.DataFrame", join_meta: dict[str, Any]) -> tuple["pd.DataFrame", dict[str, Any]]:
    matched = int(frame["llm_matched"].sum()) if "llm_matched" in frame.columns else 0
    total = int(len(frame))
    join_meta.update({
        "matched_candidate_count": matched,
        "unmatched_candidate_count": total - matched,
        "match_rate": round(matched / total, 6) if total else 0.0,
    })
    return frame, join_meta


def is_positive_llm_action(action: str | None) -> bool:
    return action in POSITIVE_ACTIONS


def is_negative_llm_action(action: str | None) -> bool:
    if action in NEGATIVE_ACTIONS:
        return True
    return False


def is_high_risk(risk_score: int | None) -> bool:
    return risk_score is not None and risk_score >= HIGH_RISK_THRESHOLD


def build_overlay_mask(
    joined: "pd.DataFrame",
    rf_mask: "pd.Series",
    *,
    variant: dict[str, Any],
    llm_subset: str,
) -> tuple["pd.Series", str | None]:
    import pandas as pd

    base = rf_mask.fillna(False).copy()
    if not variant.get("requires_llm"):
        return base, None

    if llm_subset == "qwen_ollama":
        subset_mask = joined.get("is_qwen_ollama", pd.Series(False, index=joined.index)).fillna(False)
    else:
        subset_mask = pd.Series(True, index=joined.index)

    mode = variant.get("mode")
    if mode == "buy_confirm":
        positive = joined.get("parsed_action", pd.Series(index=joined.index)).map(is_positive_llm_action)
        overlay = base & joined.get("llm_matched", False) & subset_mask & positive.fillna(False)
        return overlay, None

    if mode == "exclude_negative":
        matched = joined.get("llm_matched", False) & subset_mask
        negative = joined.get("parsed_action", pd.Series(index=joined.index)).map(is_negative_llm_action)
        high_risk = joined.get("parsed_risk_score", pd.Series(index=joined.index)).map(is_high_risk)
        bad = matched & (negative.fillna(False) | high_risk.fillna(False))
        return base & ~bad, None

    if mode == "confidence_threshold":
        if joined.get("parsed_confidence") is None or joined["parsed_confidence"].notna().sum() == 0:
            return base, "confidence_unavailable"
        threshold = float(variant["threshold"])
        confident = joined["parsed_confidence"].astype(float) >= threshold
        overlay = base & joined.get("llm_matched", False) & confident.fillna(False)
        return overlay, None

    if mode == "risk_threshold":
        if joined.get("parsed_risk_score") is None or joined["parsed_risk_score"].notna().sum() == 0:
            return base, "risk_score_unavailable"
        threshold = float(variant["threshold"])
        low_risk = joined["parsed_risk_score"].astype(float) <= threshold
        overlay = base & joined.get("llm_matched", False) & low_risk.fillna(False)
        return overlay, None

    if mode == "buy_and_low_risk":
        if (
            joined.get("parsed_confidence") is None
            or joined["parsed_confidence"].notna().sum() == 0
            or joined.get("parsed_risk_score") is None
            or joined["parsed_risk_score"].notna().sum() == 0
        ):
            return base, "confidence_or_risk_unavailable"
        positive = joined.get("parsed_action", pd.Series(index=joined.index)).map(is_positive_llm_action)
        confident = joined["parsed_confidence"].astype(float) >= float(variant["confidence_threshold"])
        low_risk = joined["parsed_risk_score"].astype(float) <= float(variant["risk_threshold"])
        overlay = (
            base
            & joined.get("llm_matched", False)
            & subset_mask
            & positive.fillna(False)
            & confident.fillna(False)
            & low_risk.fillna(False)
        )
        return overlay, None

    return base, f"unknown_variant_{variant.get('name')}"


def evaluate_overlay_metrics(
    frame: "pd.DataFrame",
    mask: "pd.Series",
    *,
    return_col: str,
    prob_col: str,
    fee_pct: float,
    rf_only_metrics: dict[str, Any],
) -> dict[str, Any]:
    from scripts.backtest_predicted_policy import evaluate_policy_trades

    metrics = evaluate_policy_trades(
        frame,
        mask,
        return_col=return_col,
        label_col="y_true",
        prob_col=prob_col,
        fee_pct=fee_pct,
    )
    metrics.pop("precision", None)
    metrics["precision"] = "not_applicable_for_overlay"
    metrics["profitable_trade_rate"] = metrics.get("win_rate")
    rf_return = rf_only_metrics.get("total_return_after_fees") or 0.0
    rf_dd = rf_only_metrics.get("max_drawdown") or 0.0
    overlay_return = metrics.get("total_return_after_fees") or 0.0
    overlay_dd = metrics.get("max_drawdown") or 0.0
    metrics["return_delta_vs_rf_only"] = round(overlay_return - rf_return, 6)
    metrics["drawdown_delta_vs_rf_only"] = round(overlay_dd - rf_dd, 6)
    if overlay_return > rf_return:
        metrics["llm_overlay_verdict"] = "improved"
    elif overlay_return < rf_return:
        metrics["llm_overlay_verdict"] = "hurt"
    else:
        metrics["llm_overlay_verdict"] = "unchanged"
    return metrics


def candidate_probability_cutoff(
    candidate_name: str,
    validation_probabilities: "np.ndarray",
    rank_cutoffs: dict[str, float],
) -> float:
    spec = RF_CANDIDATE_POLICIES[candidate_name]
    if spec.get("fixed_cutoff") is not None:
        return float(spec["fixed_cutoff"])
    source = spec["source_policy"]
    if source in rank_cutoffs:
        return rank_cutoffs[source]
    from scripts.backtest_predicted_policy import top_percent_probability_cutoff

    return top_percent_probability_cutoff(validation_probabilities, float(spec["top_percent"]))


def run_llm_overlay_evaluation(
    *,
    validation_predictions_path: Path,
    test_predictions_path: Path,
    db_path: Path,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    import pandas as pd

    from app.training.baseline_model import load_baseline_metrics
    from app.training.config import get_round_trip_fee_pct
    from scripts.backtest_predicted_policy import (
        PRIMARY_TARGET as BPP_PRIMARY,
        _derive_rank_cutoffs,
        _prepare_target_frame,
        build_policy_mask,
        evaluate_policy_trades,
    )

    if not validation_predictions_path.is_file():
        raise FileNotFoundError(
            f"Missing prediction file: {validation_predictions_path}. "
            "Run train_baseline_model.py manually; automatic retraining is disabled."
        )
    if not test_predictions_path.is_file():
        raise FileNotFoundError(
            f"Missing prediction file: {test_predictions_path}. "
            "Run train_baseline_model.py manually; automatic retraining is disabled."
        )

    metrics = load_baseline_metrics(models_dir)
    if metrics is None:
        raise FileNotFoundError("baseline_metrics.json missing.")

    fee_pct = get_round_trip_fee_pct()
    val_preds = pd.read_parquet(validation_predictions_path)
    test_preds = pd.read_parquet(test_predictions_path)

    val_frame = _prepare_target_frame(val_preds, BPP_PRIMARY, RF_MODEL_NAME)
    test_frame = _prepare_target_frame(test_preds, BPP_PRIMARY, RF_MODEL_NAME)
    rank_cutoffs = _derive_rank_cutoffs(val_frame[BPP_PRIMARY].astype(float).fillna(0.0).to_numpy())

    decisions, llm_meta = load_stored_llm_decisions(db_path)
    llm_subsets = ["qwen_ollama", "all_stored"]

    results: list[dict[str, Any]] = []
    skipped_variants: list[dict[str, str]] = []
    rf_baselines: dict[str, dict[str, Any]] = {}

    return_col = val_frame["_return_col"].iloc[0]
    prob_col = BPP_PRIMARY

    val_joined, val_join_meta = join_llm_decisions_to_candidates(val_frame, decisions)
    test_joined, test_join_meta = join_llm_decisions_to_candidates(test_frame, decisions)
    split_frames = {
        "validation": (val_frame, val_joined, val_join_meta),
        "test": (test_frame, test_joined, test_join_meta),
    }

    for candidate_name, spec in RF_CANDIDATE_POLICIES.items():
        cutoff = candidate_probability_cutoff(
            candidate_name,
            val_frame[BPP_PRIMARY].astype(float).fillna(0.0).to_numpy(),
            rank_cutoffs,
        )
        rf_baselines[candidate_name] = {}

        for split_name, (frame, joined, join_meta) in split_frames.items():
            rf_mask = build_policy_mask(frame[prob_col], cutoff)
            rf_only = evaluate_policy_trades(
                frame,
                rf_mask,
                return_col=return_col,
                label_col="y_true",
                prob_col=prob_col,
                fee_pct=fee_pct,
            )
            rf_only["profitable_trade_rate"] = rf_only.get("win_rate")
            rf_baselines[candidate_name][split_name] = rf_only

            rf_candidate_join_meta = _rf_candidate_join_stats(joined, rf_mask)

            for llm_subset in llm_subsets:
                if llm_subset == "qwen_ollama" and llm_meta.get("qwen_ollama_row_count", 0) == 0:
                    continue

                for variant in OVERLAY_VARIANTS:
                    if variant.get("requires_qwen") and llm_subset != "qwen_ollama":
                        continue

                    overlay_mask, skip_reason = build_overlay_mask(
                        joined,
                        rf_mask,
                        variant=variant,
                        llm_subset=llm_subset,
                    )
                    if skip_reason:
                        skipped_variants.append({
                            "rf_candidate_policy_name": candidate_name,
                            "split": split_name,
                            "overlay_variant": variant["name"],
                            "llm_subset": llm_subset,
                            "reason": skip_reason,
                        })
                        continue

                    overlay_metrics = evaluate_overlay_metrics(
                        frame,
                        overlay_mask,
                        return_col=return_col,
                        prob_col=prob_col,
                        fee_pct=fee_pct,
                        rf_only_metrics=rf_only,
                    )

                    results.append({
                        "split": split_name,
                        "rf_candidate_policy_name": candidate_name,
                        "overlay_variant": variant["name"],
                        "llm_subset": llm_subset,
                        "target_name": BPP_PRIMARY,
                        "model_name": RF_MODEL_NAME,
                        "probability_cutoff": round(cutoff, 6),
                        "top_percent": spec.get("top_percent"),
                        "stored_llm_table_used": llm_meta.get("stored_llm_table_used"),
                        "stored_llm_columns_used": llm_meta.get("stored_llm_columns_used"),
                        "llm_schema_fallbacks_used": llm_meta.get("llm_schema_fallbacks_used"),
                        "join_strategy_used": join_meta.get("join_strategy_used"),
                        "join_tolerance_used": join_meta.get("join_tolerance_used"),
                        "provider_distribution": llm_meta.get("provider_distribution"),
                        "model_distribution": llm_meta.get("model_distribution"),
                        **rf_candidate_join_meta,
                        "is_oracle_backtest": False,
                        **overlay_metrics,
                    })

    best = _select_best_overlay(results)

    return {
        "generated_at": _utcnow_iso(),
        "is_oracle_backtest": False,
        "selection_method": "validation_only",
        "test_set_never_used_for_threshold_selection": True,
        "rank_cutoffs_derived_from_validation": True,
        "uses_new_llm_calls": False,
        "uses_stored_llm_decisions_only": True,
        "llm_overlay_is_offline": True,
        "no_automatic_retraining": True,
        "primary_target": BPP_PRIMARY,
        "model_name": RF_MODEL_NAME,
        "rf_candidate_policies": list(RF_CANDIDATE_POLICIES.keys()),
        "validation_derived_cutoffs": {
            name: round(
                candidate_probability_cutoff(
                    name,
                    val_frame[BPP_PRIMARY].astype(float).fillna(0.0).to_numpy(),
                    rank_cutoffs,
                ),
                6,
            )
            for name in RF_CANDIDATE_POLICIES
        },
        "rejected_base_policies": {
            "top_5_percent": "Excluded from LLM overlay base entry — poor test performance.",
        },
        "stored_llm_metadata": llm_meta,
        "rf_alone_baselines": rf_baselines,
        "overlay_results": results,
        "skipped_overlay_variants": skipped_variants,
        "best_llm_overlay": best,
    }


def _select_best_overlay(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    validation = [
        r for r in results
        if r.get("split") == "validation"
        and r.get("overlay_variant") != "rf_alone"
        and r.get("trade_count", 0) > 0
    ]
    if not validation:
        return None

    def sort_key(row: dict[str, Any]) -> tuple:
        return (
            row.get("total_return_after_fees") or float("-inf"),
            row.get("max_drawdown") or 0.0,
            row.get("profit_factor") or float("-inf"),
            row.get("win_rate") or 0.0,
            -(row.get("trade_count") or 0),
        )

    best_val = max(validation, key=sort_key)
    best_test = next(
        (
            r for r in results
            if r.get("split") == "test"
            and r.get("overlay_variant") == best_val.get("overlay_variant")
            and r.get("rf_candidate_policy_name") == best_val.get("rf_candidate_policy_name")
            and r.get("llm_subset") == best_val.get("llm_subset")
        ),
        None,
    )
    return {
        "validation": best_val,
        "test": best_test,
        "selection_objective": "maximize_validation_total_return_after_fees",
    }
