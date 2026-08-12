"""AE13B fail-closed paper/demo execution guard.

Server-side only. Never relies on frontend controls.
Rejects any paper/demo order path that is not DEMO/PAPER with live disabled and no wallet.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


class DemoExecutionGuardError(Exception):
    """Raised when a paper/demo order is rejected by fail-closed safety checks."""

    def __init__(self, reasons: list[str], *, detail: dict[str, Any] | None = None) -> None:
        self.reasons = list(reasons)
        self.detail = detail or {}
        msg = "Demo/paper order rejected: " + "; ".join(self.reasons)
        super().__init__(msg)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def evaluate_paper_demo_execution_guard(
    *,
    trading_mode: str,
    live_trading_enabled: bool,
    wallet_configured: bool,
    private_key_accessed: bool = False,
    real_signing_enabled: bool = False,
    real_submission_enabled: bool = False,
    order_flags: dict[str, Any] | None = None,
    demo_acceptance_mode_enabled: bool | None = None,
) -> dict[str, Any]:
    """Evaluate fail-closed conditions for PAPER/DEMO order creation."""
    flags = dict(order_flags or {})
    mode = str(trading_mode or "").upper()
    reasons: list[str] = []

    if mode not in ("DEMO", "PAPER"):
        reasons.append(f"application_mode_not_demo_or_paper:{mode or 'MISSING'}")
    if live_trading_enabled:
        reasons.append("live_trading_enabled_true")
    if wallet_configured:
        reasons.append("wallet_configured_true")
    if private_key_accessed:
        reasons.append("private_key_accessed_true")
    if real_signing_enabled:
        reasons.append("real_signing_enabled_true")
    if real_submission_enabled:
        reasons.append("real_submission_enabled_true")

    if flags.get("paper_demo_only") is not True:
        reasons.append("order_flag_missing_or_false:paper_demo_only")
    if flags.get("not_live_approved") is not True:
        reasons.append("order_flag_missing_or_false:not_live_approved")
    if flags.get("not_profitability_evidence") is not True:
        reasons.append("order_flag_missing_or_false:not_profitability_evidence")

    if flags.get("demo_acceptance_only") is True:
        if demo_acceptance_mode_enabled is not True:
            reasons.append("demo_acceptance_mode_not_enabled")
        if flags.get("not_strategy_evidence") is not True:
            reasons.append("order_flag_missing_or_false:not_strategy_evidence")
        if mode == "LIVE":
            reasons.append("demo_acceptance_forbidden_in_live_mode")

    allowed = len(reasons) == 0
    return {
        "allowed": allowed,
        "rejected": not allowed,
        "reasons": reasons,
        "checked_at_utc": _utc_now(),
        "trading_mode": mode,
        "live_trading_enabled": bool(live_trading_enabled),
        "wallet_configured": bool(wallet_configured),
        "private_key_accessed": bool(private_key_accessed),
        "real_signing_enabled": bool(real_signing_enabled),
        "real_submission_enabled": bool(real_submission_enabled),
        "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
        "live_trading_ready": False,
        "live_trading_approval": "NO",
        "profitability_proven": False,
    }


def assert_paper_demo_allowed(
    *,
    trading_mode: str,
    live_trading_enabled: bool,
    wallet_configured: bool = False,
    private_key_accessed: bool = False,
    real_signing_enabled: bool = False,
    real_submission_enabled: bool = False,
    order_flags: dict[str, Any] | None = None,
    demo_acceptance_mode_enabled: bool | None = None,
) -> dict[str, Any]:
    """Raise DemoExecutionGuardError unless paper/demo order is allowed."""
    result = evaluate_paper_demo_execution_guard(
        trading_mode=trading_mode,
        live_trading_enabled=live_trading_enabled,
        wallet_configured=wallet_configured,
        private_key_accessed=private_key_accessed,
        real_signing_enabled=real_signing_enabled,
        real_submission_enabled=real_submission_enabled,
        order_flags=order_flags,
        demo_acceptance_mode_enabled=demo_acceptance_mode_enabled,
    )
    if result["rejected"]:
        raise DemoExecutionGuardError(result["reasons"], detail=result)
    return result


def resolve_runtime_guard_context() -> dict[str, Any]:
    """Load current mode/settings for guard checks (no wallet/private key access)."""
    trading_mode = "DEMO"
    live_trading_enabled = False
    demo_acceptance_mode = False
    settings: dict[str, Any] = {}
    try:
        from app.execution.paper import get_paper_trader

        trading_mode = str(
            get_paper_trader().get_wallet_summary().get("trading_mode") or "DEMO"
        ).upper()
    except Exception:
        trading_mode = "DEMO"
    try:
        from app import database as db

        settings = db.get_settings()
        live_trading_enabled = bool(settings.get("live_trading_enabled", False))
        demo_acceptance_mode = bool(settings.get("demo_acceptance_mode", False))
    except Exception:
        settings = {}
    return {
        "trading_mode": trading_mode,
        "live_trading_enabled": live_trading_enabled,
        "wallet_configured": False,
        "private_key_accessed": False,
        "real_signing_enabled": False,
        "real_submission_enabled": False,
        "demo_acceptance_mode": demo_acceptance_mode,
        "settings": settings if isinstance(settings, dict) else {},
    }
