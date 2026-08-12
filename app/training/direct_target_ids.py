"""Deterministic identifiers for Phase E3 direct exit-policy target rows."""

from __future__ import annotations

from app.artifacts.hash_utils import sha256_hex

NOT_APPLICABLE = "not_applicable"
TARGET_NAME = "net_profitable_after_exit_policy"

DEFAULT_EXIT_POLICIES: tuple[dict[str, float | str | int], ...] = (
    {
        "exit_policy_id": "TP20308_SL080_FEE0308_TIME_BY_HORIZON",
        "tp_ratio": 2.0308,
        "sl_ratio": 0.80,
        "round_trip_fee_pct": 0.0308,
        "time_stop_minutes": None,  # resolved from horizon
    },
    {
        "exit_policy_id": "TP20308_SL075_FEE0308_TIME_BY_HORIZON",
        "tp_ratio": 2.0308,
        "sl_ratio": 0.75,
        "round_trip_fee_pct": 0.0308,
        "time_stop_minutes": None,
    },
)

HORIZON_MINUTES: dict[str, int] = {
    "30m": 30,
    "1h": 60,
    "4h": 240,
    "8h": 480,
    "24h": 1440,
}

DEFAULT_FILTERS: tuple[str, ...] = (
    "RAW_ALL_VERIFIED",
    "LIQ_5K_HIGH_ACTIVITY",
    "LOW_LIQ_MOMENTUM",
    "NO_WHALE_FILTER",
)

DEFAULT_HORIZONS: tuple[str, ...] = ("30m", "1h", "4h", "8h", "24h")


def resolve_time_stop_minutes(horizon: str, policy: dict) -> int:
    """Return time_stop_minutes from policy or horizon mapping."""
    explicit = policy.get("time_stop_minutes")
    if explicit is not None:
        return int(explicit)
    if horizon not in HORIZON_MINUTES:
        raise ValueError(f"Unknown horizon: {horizon!r}")
    return HORIZON_MINUTES[horizon]


def compute_candidate_policy_id(
    *,
    candidate_id: str,
    filter_name: str,
    horizon: str,
    exit_policy_id: str,
    tp_ratio: float,
    sl_ratio: float,
    time_stop_minutes: int,
    round_trip_fee_pct: float,
    top_pct: str = NOT_APPLICABLE,
    pair_cap: str = NOT_APPLICABLE,
) -> str:
    """SHA-256 id for candidate under a specific exit-policy / horizon / filter context."""
    parts = [
        candidate_id,
        filter_name,
        horizon,
        top_pct,
        pair_cap,
        exit_policy_id,
        str(tp_ratio),
        str(sl_ratio),
        str(time_stop_minutes),
        str(round_trip_fee_pct),
    ]
    payload = "|".join(parts)
    return sha256_hex(f"candidate_policy:v1|{payload}")


def compute_target_row_id(
    *,
    candidate_policy_id: str,
    target_name: str = TARGET_NAME,
    target_version: str,
    label_source_artifact_id: str,
) -> str:
    """SHA-256 id for an exact target label row."""
    parts = [
        candidate_policy_id,
        target_name,
        target_version,
        label_source_artifact_id,
    ]
    payload = "|".join(parts)
    return sha256_hex(f"target_row:v1|{payload}")


def input_dataset_filename(filter_name: str, horizon: str) -> str:
    return f"{filter_name}_x2_{horizon}_CLEAN_MODEL_INPUT.parquet"


def output_dataset_basename(filter_name: str, horizon: str, exit_policy_id: str, target_version: str) -> str:
    return f"{filter_name}_{horizon}_{exit_policy_id}_DIRECT_TARGET_{target_version}"


def label_source_artifact_id_for_input(input_path: str) -> str:
    """Deterministic label-source id from input dataset path."""
    return sha256_hex(f"label_source:v1|{input_path.replace(chr(92), '/')}")
