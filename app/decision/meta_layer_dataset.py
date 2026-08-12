"""AE7 FINAL offline meta-dataset builder."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_CONSENSUS_ARTIFACT = (
    "data/training/manual_verified_results/phase_e6r_tabicl_smoke_20260704_124645"
    "/consensus/direct_target_selected_trades_by_tier.csv"
)
DEFAULT_RF_PREDICTION_GLOB = (
    "data/training/manual_verified_results/phase_e8b_clean_historical_rf_20260706_180738"
    "/predictions/*validation_predictions.csv"
)

FORBIDDEN_FEATURE_PATTERNS = (
    "target",
    "label",
    "future",
    "realized",
    "outcome",
    "profit",
    "profitable",
    "net_return",
    "realized_return",
    "future_return",
    "exit_result",
    "exit_outcome",
    "simulation_result",
)

POLICY_CONTEXT_FEATURES = (
    "tp_ratio",
    "sl_ratio",
    "round_trip_fee_pct",
    "time_stop_minutes",
    "horizon",
    "filter_name",
    "exit_policy",
)

AUDIT_ONLY_COLUMNS = (
    "candidate_id",
    "candidate_policy_id",
    "target_row_id",
    "pair_address",
    "scoring_policy_id",
    "policy_content_hash",
    "policy_binding_status",
    "policy_source",
    "policy_version",
    "event_timestamp",
    "split",
)

TARGET_COLUMN_CANDIDATES = (
    "target_net_profitable",
    "target_net_profitable_after_exit",
    "target_net_profitable_x",
    "target_net_profitable_y",
)


@dataclass
class MetaDatasetBuildResult:
    frame: pd.DataFrame
    target_column: str
    feature_columns: list[str]
    audit_columns: list[str]
    source_artifacts: list[str] = field(default_factory=list)
    rows_before_limit: int = 0
    signal_families_present: dict[str, list[str]] = field(default_factory=dict)
    signal_families_missing: dict[str, str] = field(default_factory=dict)
    policy_audit: dict[str, Any] = field(default_factory=dict)

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "rows": len(self.frame),
            "features": len(self.feature_columns),
            "target_column": self.target_column,
            "source_artifacts": self.source_artifacts,
            "signal_families_present": self.signal_families_present,
            "signal_families_missing": self.signal_families_missing,
            "policy_audit": self.policy_audit,
        }


def is_forbidden_meta_feature(name: str) -> bool:
    if name in POLICY_CONTEXT_FEATURES:
        return False
    lower = name.lower()
    if lower == "validation" or lower.endswith("_validation"):
        return False
    return any(pat in lower for pat in FORBIDDEN_FEATURE_PATTERNS)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "predicted_probability_rf": "rf_score",
        "predicted_probability_xgb": "xgb_score",
        "tab_rank_pct": "tab_rank",
        "filter": "filter_name",
        "exit_policy_id": "exit_policy",
        "sim_net_return": "sim_net_return_audit",
    }
    out = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    if "tab_score" not in out.columns and "tab_score" in df.columns:
        out["tab_score"] = df["tab_score"]
    return out


def _pick_target_column(df: pd.DataFrame) -> str:
    for col in TARGET_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError("no_safe_target_column_found_in_meta_artifacts")


def _is_target_variant_column(name: str) -> bool:
    lower = name.lower()
    return any(
        lower == cand.lower() or lower.startswith(cand.lower() + "_")
        for cand in TARGET_COLUMN_CANDIDATES
    )


def _derive_rank_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for score_col, rank_col in (
        ("rf_score", "rf_rank"),
        ("xgb_score", "xgb_rank"),
        ("tab_score", "tab_rank"),
    ):
        if score_col in out.columns and rank_col not in out.columns:
            out[rank_col] = out.groupby(["split", "filter_name", "horizon", "exit_policy"])[
                score_col
            ].rank(pct=True, ascending=False)
    return out


def _add_missing_signal_families(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, str]]:
    present: dict[str, list[str]] = {}
    missing: dict[str, str] = {}

    families = {
        "model_score_family": ["rf_score", "xgb_score", "tab_score", "rf_rank", "xgb_rank", "tab_rank"],
        "consensus_family": ["vote_count", "consensus_tier", "in_tab", "in_xgb", "in_rf"],
        "policy_family": list(POLICY_CONTEXT_FEATURES),
        "liquidity_activity_family": [
            "liquidity_usd",
            "volume_h24",
            "volume_to_liquidity_ratio",
            "txns_h24_total",
            "buy_sell_ratio_h24",
        ],
        "concentration_robustness_family": [
            "pair_count_in_policy",
            "top_pair_share_bucket",
            "robustness_pass_flag",
            "outlier_dependency_flag",
        ],
        "whale_family": ["whale_score_asof", "whale_score_missingness"],
        "context_family": [
            "rss_sentiment_score",
            "helius_indicator",
            "solana_reputation_score",
            "scam_indicator",
        ],
    }

    out = df.copy()
    for family, cols in families.items():
        avail = [c for c in cols if c in out.columns]
        if avail:
            present[family] = avail
        else:
            missing[family] = "NOT_AVAILABLE_IN_OFFLINE_ARTIFACTS"
            for col in cols:
                miss_col = f"{col}_missingness"
                if miss_col not in out.columns and col not in out.columns:
                    out[miss_col] = 1

    if "whale_score_asof" not in out.columns:
        out["whale_score_missingness"] = 1
    return out, present, missing


def build_meta_dataset(
    *,
    project_root: Path,
    max_rows: int = 50_000,
    consensus_artifact: Path | None = None,
    policy_audit: dict[str, Any] | None = None,
) -> MetaDatasetBuildResult:
    artifact = consensus_artifact or (project_root / DEFAULT_CONSENSUS_ARTIFACT)
    if not artifact.is_file():
        raise FileNotFoundError(f"consensus meta artifact not found: {artifact}")

    df = pd.read_csv(artifact, nrows=max_rows)
    rows_before = len(df)
    df = _normalize_columns(df)
    df = _derive_rank_features(df)
    df, families_present, families_missing = _add_missing_signal_families(df)

    target_col = _pick_target_column(df)
    y = df[target_col]

    audit_cols = [c for c in AUDIT_ONLY_COLUMNS if c in df.columns]
    policy_audit = policy_audit or {}
    for col, key in (
        ("scoring_policy_id", "scoring_policy_id"),
        ("policy_content_hash", "policy_content_hash"),
        ("policy_binding_status", "policy_binding_status"),
        ("policy_source", "policy_source"),
        ("policy_version", "policy_version"),
    ):
        if col not in df.columns and policy_audit.get(key):
            df[col] = policy_audit[key]
            audit_cols.append(col)

    candidate_features = []
    for col in df.columns:
        if col == target_col or _is_target_variant_column(col) or col in audit_cols:
            continue
        if col.endswith("_missingness"):
            candidate_features.append(col)
            continue
        if is_forbidden_meta_feature(col):
            continue
        if col == "pair_address":
            continue
        candidate_features.append(col)

    feature_cols = sorted(set(candidate_features))
    meta = df[audit_cols + feature_cols + [target_col]].copy()
    meta["meta_target_y"] = y

    return MetaDatasetBuildResult(
        frame=meta,
        target_column="meta_target_y",
        feature_columns=feature_cols,
        audit_columns=audit_cols,
        source_artifacts=[str(artifact)],
        rows_before_limit=rows_before,
        signal_families_present=families_present,
        signal_families_missing=families_missing,
        policy_audit=policy_audit,
    )


def save_meta_dataset(result: MetaDatasetBuildResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        result.frame.to_parquet(path, index=False)
    else:
        result.frame.to_csv(path, index=False)
