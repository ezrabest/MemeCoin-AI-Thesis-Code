"""AE7C-0 scoring policy parameter resolution for runtime feature enrichment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.training.direct_target_ids import (
    DEFAULT_EXIT_POLICIES,
    HORIZON_MINUTES,
    resolve_time_stop_minutes,
)

AE7C0_PLACEHOLDER_POLICY_ID = "TP20308_SL080_FEE0308_TIME_BY_HORIZON"
AE7C0_DEFAULT_HORIZON = "1h"
POLICY_FEATURE_SOURCE_PLACEHOLDER = "AE7C0_PLACEHOLDER_NON_TRADING_SCORING_POLICY"
POLICY_FEATURE_STATUS_PLACEHOLDER = "PLACEHOLDER_NO_MODEL_INFERENCE"

POLICY_FEATURE_NAMES = (
    "tp_ratio",
    "sl_ratio",
    "time_stop_minutes",
    "round_trip_fee_pct",
)


def _placeholder_policy(horizon: str = AE7C0_DEFAULT_HORIZON) -> dict[str, Any]:
    policy = dict(DEFAULT_EXIT_POLICIES[0])
    policy["time_stop_minutes"] = resolve_time_stop_minutes(horizon, policy)
    policy["horizon"] = horizon
    return policy


def _policy_from_settings(settings: dict[str, Any], horizon: str) -> dict[str, Any] | None:
    """Return policy dict when explicit scoring-policy keys exist in settings."""
    keys = ("tp_ratio", "sl_ratio", "round_trip_fee_pct", "time_stop_minutes")
    if not any(k in settings for k in keys):
        return None
    policy: dict[str, Any] = {
        "exit_policy_id": settings.get("exit_policy_id", AE7C0_PLACEHOLDER_POLICY_ID),
        "horizon": settings.get("horizon", horizon),
        "tp_ratio": settings.get("tp_ratio"),
        "sl_ratio": settings.get("sl_ratio"),
        "round_trip_fee_pct": settings.get("round_trip_fee_pct"),
        "time_stop_minutes": settings.get("time_stop_minutes"),
    }
    if policy["time_stop_minutes"] is None and policy["horizon"] in HORIZON_MINUTES:
        policy["time_stop_minutes"] = HORIZON_MINUTES[str(policy["horizon"])]
    if all(policy.get(k) is not None for k in keys):
        return policy
    return None


def resolve_scoring_policy_context(
    *,
    settings_path: Path | None = None,
    horizon: str = AE7C0_DEFAULT_HORIZON,
) -> dict[str, Any]:
    """Resolve as-of scoring policy parameters from config or documented placeholder."""
    source = POLICY_FEATURE_SOURCE_PLACEHOLDER
    status = POLICY_FEATURE_STATUS_PLACEHOLDER
    policy = _placeholder_policy(horizon)

    if settings_path and settings_path.is_file():
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            from_settings = _policy_from_settings(settings, horizon)
            if from_settings:
                policy = from_settings
                source = "project_settings"
                status = "CONFIGURED_NON_TRADING_SCORING_POLICY"
        except (json.JSONDecodeError, OSError):
            pass

    values = {
        "tp_ratio": float(policy["tp_ratio"]),
        "sl_ratio": float(policy["sl_ratio"]),
        "time_stop_minutes": int(policy["time_stop_minutes"]),
        "round_trip_fee_pct": float(policy["round_trip_fee_pct"]),
    }
    metadata = {
        name: {
            "feature_source": "scoring_policy",
            "as_of_safe": True,
            "not_label": True,
            "not_future_outcome": True,
            "used_for_inference": False,
            "policy_feature_source": source,
            "policy_feature_status": status,
            "exit_policy_id": policy.get("exit_policy_id", AE7C0_PLACEHOLDER_POLICY_ID),
            "horizon": policy.get("horizon", horizon),
        }
        for name in POLICY_FEATURE_NAMES
    }
    return {
        "values": values,
        "metadata": metadata,
        "policy_feature_source": source,
        "policy_feature_status": status,
        "exit_policy_id": policy.get("exit_policy_id", AE7C0_PLACEHOLDER_POLICY_ID),
        "horizon": policy.get("horizon", horizon),
    }
