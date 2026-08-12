"""AE7C-1 deterministic runtime scoring-policy binding."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.training.direct_target_ids import (
    DEFAULT_EXIT_POLICIES,
    HORIZON_MINUTES,
    resolve_time_stop_minutes,
)

SCORING_POLICY_ID_NAMESPACE = "AE7C1_SCORING_POLICY_V1"
SCORING_POLICY_VERSION = "AE7C1_V1"
MODEL_FAMILY_TARGETS = ["RF", "XGB", "TAB"]

HORIZON_PATTERN = re.compile(r"\b(30m|1h|4h|8h|24h)\b", re.IGNORECASE)
EXIT_POLICY_PATTERN = re.compile(r"TP\d+_SL\d+_FEE\d+_TIME_BY_HORIZON", re.IGNORECASE)


class ScoringPolicyBindingStatus(StrEnum):
    PASS_CONFIG_BOUND = "PASS_CONFIG_BOUND"
    PASS_SIGNAL_CONTEXT_BOUND = "PASS_SIGNAL_CONTEXT_BOUND"
    PLACEHOLDER_BOUND = "PLACEHOLDER_BOUND"
    BLOCKED_NO_POLICY_CONTEXT = "BLOCKED_NO_POLICY_CONTEXT"
    BLOCKED_INCONSISTENT_POLICY_CONTEXT = "BLOCKED_INCONSISTENT_POLICY_CONTEXT"


@dataclass
class ScoringPolicyBinding:
    scoring_policy_id: str
    scoring_policy_version: str
    scoring_policy_source: str
    scoring_policy_binding_status: str
    policy_features: dict[str, float | int]
    horizon: str
    exit_policy: str
    model_family_targets: list[str] = field(default_factory=lambda: list(MODEL_FAMILY_TARGETS))
    binding_caveats: list[str] = field(default_factory=list)
    policy_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scoring_policy_id": self.scoring_policy_id,
            "scoring_policy_version": self.scoring_policy_version,
            "scoring_policy_source": self.scoring_policy_source,
            "scoring_policy_binding_status": self.scoring_policy_binding_status,
            "policy_features": self.policy_features,
            "horizon": self.horizon,
            "exit_policy": self.exit_policy,
            "model_family_targets": self.model_family_targets,
            "binding_caveats": self.binding_caveats,
        }

    def to_audit_row(self, *, candidate_id: str | None = None, signal_id: Any = None) -> dict[str, Any]:
        row = self.to_dict()
        row["candidate_id"] = candidate_id
        row["signal_id"] = signal_id
        return row


def _sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_scoring_policy_id_from_content(
    *,
    exit_policy: str,
    horizon: str,
    policy_features: dict[str, float | int],
) -> str:
    core = {
        "exit_policy": exit_policy,
        "horizon": horizon,
        "policy_features": {k: policy_features[k] for k in sorted(policy_features)},
    }
    serialized = json.dumps(core, sort_keys=True, separators=(",", ":"))
    return _sha256_hex(f"{SCORING_POLICY_ID_NAMESPACE}|{serialized}")


def _parse_features_json(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            return dict(parsed) if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _lookup_exit_policy(exit_policy_id: str) -> dict[str, Any] | None:
    for policy in DEFAULT_EXIT_POLICIES:
        if str(policy.get("exit_policy_id", "")).upper() == exit_policy_id.upper():
            out = dict(policy)
            return out
    return None


def _policy_features_from_dict(policy: dict[str, Any], horizon: str) -> dict[str, float | int]:
    time_stop = policy.get("time_stop_minutes")
    if time_stop is None:
        time_stop = resolve_time_stop_minutes(horizon, policy)
    return {
        "tp_ratio": float(policy["tp_ratio"]),
        "sl_ratio": float(policy["sl_ratio"]),
        "time_stop_minutes": int(time_stop),
        "round_trip_fee_pct": float(policy["round_trip_fee_pct"]),
    }


def _extract_config_policy(settings: dict[str, Any]) -> dict[str, Any] | None:
    explicit_keys = ("tp_ratio", "sl_ratio", "round_trip_fee_pct", "time_stop_minutes")
    if all(settings.get(k) is not None for k in explicit_keys):
        horizon = str(settings.get("horizon", "1h"))
        return {
            "source": "explicit_settings_keys",
            "exit_policy": str(settings.get("exit_policy_id", "EXPLICIT_CONFIG_POLICY")),
            "horizon": horizon,
            "policy_features": {
                "tp_ratio": float(settings["tp_ratio"]),
                "sl_ratio": float(settings["sl_ratio"]),
                "time_stop_minutes": int(settings["time_stop_minutes"]),
                "round_trip_fee_pct": float(settings["round_trip_fee_pct"]),
            },
        }

    exit_policy_id = settings.get("exit_policy_id")
    horizon = settings.get("horizon")
    if exit_policy_id and horizon:
        matched = _lookup_exit_policy(str(exit_policy_id))
        if matched and str(horizon) in HORIZON_MINUTES:
            features = _policy_features_from_dict(matched, str(horizon))
            return {
                "source": "settings_exit_policy_and_horizon",
                "exit_policy": str(exit_policy_id),
                "horizon": str(horizon),
                "policy_features": features,
            }
    return None


def _extract_signal_policy(signal_row: dict[str, Any] | None) -> dict[str, Any] | None:
    if not signal_row:
        return None
    feats = _parse_features_json(signal_row.get("features_json"))
    horizon = feats.get("horizon") or signal_row.get("horizon")
    exit_policy = (
        feats.get("exit_policy_id")
        or feats.get("exit_policy")
        or signal_row.get("exit_policy_id")
    )

    explicit = {
        k: feats.get(k) or signal_row.get(k)
        for k in ("tp_ratio", "sl_ratio", "round_trip_fee_pct", "time_stop_minutes")
    }
    if all(explicit[k] is not None for k in explicit):
        hz = str(horizon or feats.get("horizon") or "1h")
        return {
            "source": "signal_explicit_policy_fields",
            "exit_policy": str(exit_policy or "SIGNAL_EXPLICIT_POLICY"),
            "horizon": hz,
            "policy_features": {
                "tp_ratio": float(explicit["tp_ratio"]),
                "sl_ratio": float(explicit["sl_ratio"]),
                "time_stop_minutes": int(explicit["time_stop_minutes"]),
                "round_trip_fee_pct": float(explicit["round_trip_fee_pct"]),
            },
        }

    if not horizon and signal_row.get("reason"):
        m = HORIZON_PATTERN.search(str(signal_row["reason"]))
        if m:
            horizon = m.group(1).lower()
    if not exit_policy:
        for text in (signal_row.get("model_source"), signal_row.get("reason")):
            if not text:
                continue
            m = EXIT_POLICY_PATTERN.search(str(text))
            if m:
                exit_policy = m.group(0).upper()
                break

    if exit_policy and horizon:
        matched = _lookup_exit_policy(str(exit_policy))
        if matched and str(horizon) in HORIZON_MINUTES:
            features = _policy_features_from_dict(matched, str(horizon))
            return {
                "source": "signal_exit_policy_and_horizon",
                "exit_policy": str(exit_policy),
                "horizon": str(horizon),
                "policy_features": features,
            }
    return None


def _placeholder_binding(horizon: str = "1h") -> ScoringPolicyBinding:
    policy = dict(DEFAULT_EXIT_POLICIES[0])
    features = _policy_features_from_dict(policy, horizon)
    exit_policy = str(policy["exit_policy_id"])
    return ScoringPolicyBinding(
        scoring_policy_id=generate_scoring_policy_id_from_content(
            exit_policy=exit_policy,
            horizon=horizon,
            policy_features=features,
        ),
        scoring_policy_version=SCORING_POLICY_VERSION,
        scoring_policy_source="AE7C0_PLACEHOLDER_NON_TRADING_SCORING_POLICY",
        scoring_policy_binding_status=ScoringPolicyBindingStatus.PLACEHOLDER_BOUND.value,
        policy_features=features,
        horizon=horizon,
        exit_policy=exit_policy,
        binding_caveats=["no_config_or_signal_policy_context_using_documented_placeholder"],
        policy_context={
            "values": features,
            "policy_feature_source": "AE7C0_PLACEHOLDER_NON_TRADING_SCORING_POLICY",
            "policy_feature_status": "PLACEHOLDER_NO_MODEL_INFERENCE",
            "exit_policy_id": exit_policy,
            "horizon": horizon,
        },
    )


def _policies_consistent(a: dict[str, Any], b: dict[str, Any], tol: float = 1e-9) -> bool:
    fa = a.get("policy_features") or {}
    fb = b.get("policy_features") or {}
    for key in ("tp_ratio", "sl_ratio", "round_trip_fee_pct", "time_stop_minutes"):
        if key not in fa or key not in fb:
            return False
        if key == "time_stop_minutes":
            if int(fa[key]) != int(fb[key]):
                return False
        elif abs(float(fa[key]) - float(fb[key])) > tol:
            return False
    return True


def bind_scoring_policy(
    *,
    settings: dict[str, Any] | None = None,
    signal_row: dict[str, Any] | None = None,
    default_horizon: str = "1h",
) -> ScoringPolicyBinding:
    """Bind scoring policy from config, signal context, or placeholder."""
    settings = settings or {}
    config_policy = _extract_config_policy(settings)
    signal_policy = _extract_signal_policy(signal_row)

    if config_policy and signal_policy and not _policies_consistent(config_policy, signal_policy):
        return ScoringPolicyBinding(
            scoring_policy_id="",
            scoring_policy_version=SCORING_POLICY_VERSION,
            scoring_policy_source="inconsistent_config_and_signal",
            scoring_policy_binding_status=ScoringPolicyBindingStatus.BLOCKED_INCONSISTENT_POLICY_CONTEXT.value,
            policy_features={},
            horizon=str(signal_policy.get("horizon") or default_horizon),
            exit_policy=str(signal_policy.get("exit_policy") or ""),
            binding_caveats=[
                "config_policy_and_signal_policy_disagree",
                f"config_source={config_policy.get('source')}",
                f"signal_source={signal_policy.get('source')}",
            ],
        )

    if config_policy:
        features = config_policy["policy_features"]
        exit_policy = str(config_policy["exit_policy"])
        horizon = str(config_policy["horizon"])
        return ScoringPolicyBinding(
            scoring_policy_id=generate_scoring_policy_id_from_content(
                exit_policy=exit_policy,
                horizon=horizon,
                policy_features=features,
            ),
            scoring_policy_version=SCORING_POLICY_VERSION,
            scoring_policy_source=str(config_policy["source"]),
            scoring_policy_binding_status=ScoringPolicyBindingStatus.PASS_CONFIG_BOUND.value,
            policy_features=features,
            horizon=horizon,
            exit_policy=exit_policy,
            policy_context={
                "values": features,
                "policy_feature_source": "project_config",
                "policy_feature_status": "CONFIG_BOUND_NON_TRADING",
                "exit_policy_id": exit_policy,
                "horizon": horizon,
            },
        )

    if signal_policy:
        features = signal_policy["policy_features"]
        exit_policy = str(signal_policy["exit_policy"])
        horizon = str(signal_policy["horizon"])
        return ScoringPolicyBinding(
            scoring_policy_id=generate_scoring_policy_id_from_content(
                exit_policy=exit_policy,
                horizon=horizon,
                policy_features=features,
            ),
            scoring_policy_version=SCORING_POLICY_VERSION,
            scoring_policy_source=str(signal_policy["source"]),
            scoring_policy_binding_status=ScoringPolicyBindingStatus.PASS_SIGNAL_CONTEXT_BOUND.value,
            policy_features=features,
            horizon=horizon,
            exit_policy=exit_policy,
            policy_context={
                "values": features,
                "policy_feature_source": "signal_context",
                "policy_feature_status": "SIGNAL_BOUND_NON_TRADING",
                "exit_policy_id": exit_policy,
                "horizon": horizon,
            },
        )

    return _placeholder_binding(default_horizon)


def load_settings_dict(settings_path: Path | None) -> dict[str, Any]:
    if not settings_path or not settings_path.is_file():
        return {}
    try:
        return json.loads(settings_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
