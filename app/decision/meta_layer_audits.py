"""AE7 FINAL meta-layer audits."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from app.decision.meta_layer_dataset import is_forbidden_meta_feature


@dataclass
class MetaAuditBundle:
    leakage_status: str
    forbidden_feature_count: int
    pair_address_predictive_use: bool
    split_integrity_status: str
    policy_hash_status: str
    target_availability_status: str
    missingness_status: str
    pair_concentration_status: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "leakage_status": self.leakage_status,
            "forbidden_feature_count": self.forbidden_feature_count,
            "pair_address_predictive_use": self.pair_address_predictive_use,
            "split_integrity_status": self.split_integrity_status,
            "policy_hash_status": self.policy_hash_status,
            "target_availability_status": self.target_availability_status,
            "missingness_status": self.missingness_status,
            "pair_concentration_status": self.pair_concentration_status,
            "details": self.details,
        }


def run_meta_audits(
    *,
    frame: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    policy_audit: dict[str, Any],
) -> MetaAuditBundle:
    forbidden = [c for c in feature_columns if is_forbidden_meta_feature(c)]
    pair_in_x = "pair_address" in feature_columns
    leakage_status = "PASS" if not forbidden and not pair_in_x else "FAIL"

    split_status = "PASS"
    if "split" in frame.columns:
        counts = frame["split"].value_counts(dropna=False).to_dict()
        if len(counts) < 2:
            split_status = "WARN_SINGLE_SPLIT_ONLY"
    else:
        split_status = "WARN_NO_SPLIT_COLUMN"

    policy_hash_status = "PASS" if policy_audit.get("policy_config_status") != "POLICY_CONFIG_VALIDATION_FAILED" else "FAIL"
    if policy_audit.get("policy_config_status") == "NOT_PROVIDED_ARTIFACT_EMBEDDED":
        policy_hash_status = "ARTIFACT_EMBEDDED"

    target_status = "PASS" if target_column in frame.columns and frame[target_column].notna().any() else "FAIL"

    missingness = {}
    for col in feature_columns:
        if col in frame.columns:
            missingness[col] = int(frame[col].isna().sum())
    missingness_status = "PASS" if missingness else "WARN_NO_FEATURES"

    pair_concentration = {}
    if "pair_address" in frame.columns:
        counts = frame["pair_address"].value_counts(normalize=True)
        pair_concentration = {
            "top_pair_share": float(counts.iloc[0]) if len(counts) else 0.0,
            "unique_pairs": int(counts.shape[0]),
        }
    concentration_status = (
        "WARN_HIGH_CONCENTRATION"
        if pair_concentration.get("top_pair_share", 0) > 0.35
        else "PASS"
    )

    return MetaAuditBundle(
        leakage_status=leakage_status,
        forbidden_feature_count=len(forbidden),
        pair_address_predictive_use=pair_in_x,
        split_integrity_status=split_status,
        policy_hash_status=policy_hash_status,
        target_availability_status=target_status,
        missingness_status=missingness_status,
        pair_concentration_status=concentration_status,
        details={
            "forbidden_features": forbidden,
            "missingness": missingness,
            "pair_concentration": pair_concentration,
            "policy_audit": policy_audit,
        },
    )


def write_audit_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        rows = [{"status": "EMPTY"}]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_audit_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
