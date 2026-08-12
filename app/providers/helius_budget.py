"""
Helius API credit budget manager — monthly/daily pacing for validation calls.
"""
from __future__ import annotations

import calendar
import json
import logging
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("helius_budget")

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
USAGE_PATH = DATA_DIR / "api_usage" / "helius_usage.json"

DEFAULT_MONTHLY_BUDGET = 100_000
CREDITS_PER_SIGNATURE = 1


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _month_key(day: date | None = None) -> str:
    day = day or _utc_today()
    return day.strftime("%Y-%m")


def _day_key(day: date | None = None) -> str:
    day = day or _utc_today()
    return day.isoformat()


def remaining_days_in_month(day: date | None = None) -> int:
    """Calendar days remaining in the month, including today."""
    day = day or _utc_today()
    last_day = calendar.monthrange(day.year, day.month)[1]
    return max(last_day - day.day + 1, 1)


def dynamic_daily_budget(
    monthly_budget: int,
    monthly_used: int,
    day: date | None = None,
) -> float:
    remaining = max(monthly_budget - monthly_used, 0)
    days_left = remaining_days_in_month(day)
    return remaining / days_left


def default_usage(monthly_budget: int = DEFAULT_MONTHLY_BUDGET) -> dict[str, Any]:
    return {
        "provider": "helius",
        "monthly_budget": monthly_budget,
        "monthly_used": 0,
        "daily_used": {},
        "endpoint_used": {},
        "last_reset_month": _month_key(),
    }


def load_usage(
    *,
    usage_path: Path | None = None,
    monthly_budget: int = DEFAULT_MONTHLY_BUDGET,
) -> dict[str, Any]:
    path = usage_path or USAGE_PATH
    try:
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                data.setdefault("monthly_budget", monthly_budget)
                return data
    except Exception as exc:
        log.debug("Helius usage load failed: %s", exc)
    return default_usage(monthly_budget)


def save_usage(usage: dict[str, Any], *, usage_path: Path | None = None) -> None:
    path = usage_path or USAGE_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(usage, indent=2), encoding="utf-8")
    except Exception as exc:
        log.debug("Helius usage save failed: %s", exc)


def reset_month_if_needed(usage: dict[str, Any], day: date | None = None) -> dict[str, Any]:
    current = _month_key(day)
    if usage.get("last_reset_month") != current:
        usage["monthly_used"] = 0
        usage["daily_used"] = {}
        usage["endpoint_used"] = {}
        usage["last_reset_month"] = current
    return usage


def estimate_credit_cost(signature_count: int) -> int:
    return max(signature_count, 0) * CREDITS_PER_SIGNATURE


def can_spend(cost: int, usage: dict[str, Any], day: date | None = None) -> bool:
    usage = reset_month_if_needed(usage, day)
    daily_budget = dynamic_daily_budget(
        int(usage.get("monthly_budget", DEFAULT_MONTHLY_BUDGET)),
        int(usage.get("monthly_used", 0)),
        day,
    )
    today_key = _day_key(day)
    daily_used = float((usage.get("daily_used") or {}).get(today_key, 0))
    return daily_used + cost <= daily_budget


def record_spend(
    cost: int,
    usage: dict[str, Any],
    *,
    endpoint: str = "v0/transactions",
    day: date | None = None,
) -> dict[str, Any]:
    usage = reset_month_if_needed(usage, day)
    today_key = _day_key(day)
    usage["monthly_used"] = int(usage.get("monthly_used", 0)) + cost
    daily_used = dict(usage.get("daily_used") or {})
    daily_used[today_key] = float(daily_used.get(today_key, 0)) + cost
    usage["daily_used"] = daily_used
    endpoint_used = dict(usage.get("endpoint_used") or {})
    endpoint_used[endpoint] = int(endpoint_used.get(endpoint, 0)) + cost
    usage["endpoint_used"] = endpoint_used
    return usage


class HeliusBudgetManager:
    def __init__(
        self,
        *,
        monthly_budget: int = DEFAULT_MONTHLY_BUDGET,
        usage_path: Path | None = None,
    ) -> None:
        self.monthly_budget = monthly_budget
        self.usage_path = usage_path or USAGE_PATH
        self.usage = reset_month_if_needed(
            load_usage(usage_path=self.usage_path, monthly_budget=monthly_budget)
        )

    def refresh(self) -> None:
        self.usage = reset_month_if_needed(
            load_usage(usage_path=self.usage_path, monthly_budget=self.monthly_budget)
        )

    def dynamic_daily_budget(self, day: date | None = None) -> float:
        return dynamic_daily_budget(
            int(self.usage.get("monthly_budget", self.monthly_budget)),
            int(self.usage.get("monthly_used", 0)),
            day,
        )

    def remaining_monthly_budget(self) -> int:
        return max(
            int(self.usage.get("monthly_budget", self.monthly_budget))
            - int(self.usage.get("monthly_used", 0)),
            0,
        )

    def can_spend(self, cost: int, day: date | None = None) -> bool:
        return can_spend(cost, self.usage, day)

    def record_spend(self, cost: int, *, endpoint: str = "v0/transactions") -> None:
        self.usage = record_spend(cost, self.usage, endpoint=endpoint)
        save_usage(self.usage, usage_path=self.usage_path)

    def budget_snapshot(self) -> dict[str, Any]:
        return {
            "monthly_budget": int(self.usage.get("monthly_budget", self.monthly_budget)),
            "monthly_used": int(self.usage.get("monthly_used", 0)),
            "remaining_monthly_budget": self.remaining_monthly_budget(),
            "dynamic_daily_budget": round(self.dynamic_daily_budget(), 4),
            "daily_used_today": float((self.usage.get("daily_used") or {}).get(_day_key(), 0)),
            "last_reset_month": self.usage.get("last_reset_month"),
        }
