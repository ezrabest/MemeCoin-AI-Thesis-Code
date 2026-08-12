"""AE13 no-wallet / no-live safety audit helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def build_no_wallet_safety_audit(
    *,
    trading_mode: str = "DEMO",
    live_trading_enabled: bool = False,
    wallet_configured: bool = False,
    demo_acceptance_used: bool = False,
) -> dict[str, Any]:
    return {
        "phase": "AE13",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "wallet_configured": bool(wallet_configured),
        "private_key_accessed": False,
        "real_transaction_signed": False,
        "real_transaction_attempted": False,
        "live_submission_status": "NOT_SUBMITTED_NO_WALLET",
        "live_trading_ready": False,
        "live_trading_approval": "NO",
        "live_trading_enabled": bool(live_trading_enabled),
        "trading_mode": str(trading_mode).upper(),
        "profitability_proven": False,
        "demo_acceptance_used": bool(demo_acceptance_used),
        "demo_acceptance_is_live_authority": False,
        "audit_status": "PASS" if not wallet_configured and not live_trading_enabled else "FAIL",
    }
