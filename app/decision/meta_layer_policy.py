"""AE7 FINAL scoring policy config loading with fail-hard semantics."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Any

from app.decision.scoring_policy_binding import generate_scoring_policy_id_from_content

POLICY_CONFIG_VERSION = "AE7_FINAL_V1"
MATERIAL_POLICY_FIELDS = (
    "tp_ratio",
    "sl_ratio",
    "round_trip_fee_pct",
    "time_stop_minutes",
    "horizon",
    "exit_policy",
    "model_family_targets",
    "policy_version",
    "policy_status",
    "allow_trading",
    "allow_paper_trading",
    "allow_model_inference",
)


class PolicyConfigStatus(StrEnum):
    NOT_PROVIDED_ARTIFACT_EMBEDDED = "NOT_PROVIDED_ARTIFACT_EMBEDDED"
    PASS_LOADED = "PASS_LOADED"
    POLICY_CONFIG_MISSING = "POLICY_CONFIG_MISSING"
    POLICY_CONFIG_INVALID = "POLICY_CONFIG_INVALID"
    POLICY_CONFIG_VALIDATION_FAILED = "POLICY_CONFIG_VALIDATION_FAILED"
    PLACEHOLDER_BOUND_OFFLINE_SMOKE = "PLACEHOLDER_BOUND_OFFLINE_SMOKE"


class PolicyConfigError(Exception):
    def __init__(self, status: str, reason: str) -> None:
        super().__init__(reason)
        self.status = status
        self.reason = reason


def canonical_policy_json(obj: dict[str, Any]) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def policy_content_hash(policy_content: dict[str, Any]) -> str:
    material = {k: policy_content[k] for k in MATERIAL_POLICY_FIELDS if k in policy_content}
    serialized = canonical_policy_json(material if material else policy_content)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def scoring_policy_id_from_policy_content(policy_content: dict[str, Any]) -> str:
    features = {
        "tp_ratio": float(policy_content.get("tp_ratio", 0)),
        "sl_ratio": float(policy_content.get("sl_ratio", 0)),
        "time_stop_minutes": int(policy_content.get("time_stop_minutes", 0)),
        "round_trip_fee_pct": float(policy_content.get("round_trip_fee_pct", 0)),
    }
    return generate_scoring_policy_id_from_content(
        exit_policy=str(policy_content.get("exit_policy", policy_content.get("exit_policy_id", ""))),
        horizon=str(policy_content.get("horizon", "")),
        policy_features=features,
    )


def validate_policy_content(policy: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    required_numeric = ("tp_ratio", "sl_ratio", "round_trip_fee_pct", "time_stop_minutes")
    for key in required_numeric:
        if key not in policy or policy[key] is None:
            errors.append(f"missing_{key}")
    if not policy.get("horizon") and not policy.get("exit_policy") and not policy.get("exit_policy_id"):
        errors.append("missing_horizon_or_exit_policy")
    return errors


def load_scoring_policy_config_strict(
    path: Path | None,
    *,
    offline_smoke_mode: bool = False,
) -> dict[str, Any]:
    """Load policy config. Fail hard when path is explicitly provided."""
    if path is None:
        return {
            "policy_config_status": PolicyConfigStatus.NOT_PROVIDED_ARTIFACT_EMBEDDED.value,
            "policy_content": {},
            "policy_content_hash": "",
            "scoring_policy_id": "",
            "policy_source": "artifact_embedded",
            "policy_binding_status": "ARTIFACT_EMBEDDED",
        }

    if not path.is_file():
        raise PolicyConfigError(
            PolicyConfigStatus.POLICY_CONFIG_MISSING.value,
            f"scoring policy config not found: {path}",
        )

    try:
        raw = path.read_text(encoding="utf-8")
        policy = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PolicyConfigError(
            PolicyConfigStatus.POLICY_CONFIG_INVALID.value,
            f"malformed scoring policy config JSON: {exc}",
        ) from exc
    except OSError as exc:
        raise PolicyConfigError(
            PolicyConfigStatus.POLICY_CONFIG_MISSING.value,
            f"unable to read scoring policy config: {exc}",
        ) from exc

    if not isinstance(policy, dict):
        raise PolicyConfigError(
            PolicyConfigStatus.POLICY_CONFIG_INVALID.value,
            "scoring policy config must be a JSON object",
        )

    errors = validate_policy_content(policy)
    if errors:
        raise PolicyConfigError(
            PolicyConfigStatus.POLICY_CONFIG_VALIDATION_FAILED.value,
            f"policy validation failed: {','.join(errors)}",
        )

    content_hash = policy_content_hash(policy)
    policy_id = scoring_policy_id_from_policy_content(policy)
    return {
        "policy_config_status": PolicyConfigStatus.PASS_LOADED.value,
        "policy_content": policy,
        "policy_content_hash": content_hash,
        "scoring_policy_id": policy_id,
        "policy_source": str(path),
        "policy_binding_status": "PASS_CONFIG_BOUND",
        "policy_version": policy.get("policy_version", POLICY_CONFIG_VERSION),
    }
