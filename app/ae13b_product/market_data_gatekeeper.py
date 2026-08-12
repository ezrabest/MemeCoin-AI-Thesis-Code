"""Reusable market-data gate middleware (AE13I) — not coupled to PaperTrader."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.ae13b_product.address_role import enrich_row_with_address_role
from app.ae13b_product.provenance_enricher import enrich_market_provenance
from app.ae13b_product.reentry_blocks import check_reentry_block
from app.ae13b_product.stagnant_price_guard import evaluate_stagnant_price
from app.ae13b_product.system_reentry_signal import (
    REENTRY_BLOCK_NO_NEW_SIGNAL,
    check_system_reentry_signal,
)

DEFAULT_MAX_PRICE_AGE_SECONDS = 900
DEFAULT_MAX_LIQUIDITY_AGE_SECONDS = 1800
DEFAULT_MAX_PROVIDER_SEEN_AGE_SECONDS = 1800


#: Fixed AE13I tradability_status enum — never emit a value outside this set.
ALLOWED_TRADABILITY_STATUSES = frozenset(
    {
        "tradable_now",
        "stale_market_data",
        "historical_only",
        "watchlist_only",
        "resolver_only",
        "explorer_only",
        "missing_price",
        "missing_liquidity",
        "missing_price_timestamp",
        "missing_liquidity_timestamp",
        "provider_unresolved",
        "unsupported_chain",
        "inactive_pool",
        "ambiguous_address_role",
        "pair_token_identity_conflict",
        "not_tradable_without_market_price",
        "unknown",
    }
)

MANUAL_REENTRY_BLOCK_ACTIVE = "MANUAL_REENTRY_BLOCK_ACTIVE"


@dataclass
class MarketDataGateResult:
    passed: bool
    tradability_status: str
    freshness_gate_status: str
    primary_blocker: str | None
    rejection_code: str | None
    rejection_reasons: list[str] = field(default_factory=list)
    blocking_guards: list[str] = field(default_factory=list)
    provenance_status: str | None = None
    address_role_status: str | None = None
    market_data_status: str | None = None
    semantic_status: str | None = None
    decision: str | None = None
    candidate_context: dict[str, Any] = field(default_factory=dict)
    checked_at_utc: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IdentityNormalizer:
    """Light identity normalization — does not overwrite user-entered fields."""

    @staticmethod
    def normalize(row: dict[str, Any]) -> dict[str, Any]:
        out = dict(row or {})
        chain = (
            out.get("chain")
            or out.get("network")
            or out.get("market_chain")
            or out.get("resolved_chain")
        )
        symbol = (
            out.get("symbol")
            or out.get("market_symbol")
            or out.get("resolved_symbol")
        )
        pair_address = (
            out.get("pair_address")
            or out.get("matched_pair_address")
            or out.get("resolved_pair_address")
        )
        token_mint = out.get("token_mint_address") or out.get("token_mint")
        token_contract = (
            out.get("token_contract_address")
            or out.get("token_contract")
            or out.get("base_token_address")
        )
        out["chain"] = str(chain).strip().lower() if chain else None
        out["symbol"] = str(symbol).strip().upper() if symbol else None
        out["pair_address"] = str(pair_address).strip() if pair_address else None
        out["token_mint_address"] = str(token_mint).strip() if token_mint else None
        out["token_contract_address"] = str(token_contract).strip() if token_contract else None
        price = out.get("latest_price") if out.get("latest_price") is not None else out.get("price")
        if price is not None:
            try:
                out["latest_price"] = float(price)
            except (TypeError, ValueError):
                pass
        liq = out.get("latest_liquidity") if out.get("latest_liquidity") is not None else out.get("liquidity")
        if liq is not None:
            try:
                out["latest_liquidity"] = float(liq)
            except (TypeError, ValueError):
                pass
        return out


class AddressRoleClassifier:
    @staticmethod
    def classify(row: dict[str, Any]) -> dict[str, Any]:
        return enrich_row_with_address_role(row)


class ProvenanceEnricher:
    @staticmethod
    def enrich(row: dict[str, Any]) -> dict[str, Any]:
        merged = dict(row)
        merged.update(enrich_market_provenance(row))
        return merged


class FreshnessValidator:
    def __init__(
        self,
        *,
        max_price_age_seconds: float = DEFAULT_MAX_PRICE_AGE_SECONDS,
        max_liquidity_age_seconds: float = DEFAULT_MAX_LIQUIDITY_AGE_SECONDS,
        max_provider_seen_age_seconds: float = DEFAULT_MAX_PROVIDER_SEEN_AGE_SECONDS,
    ) -> None:
        self.max_price_age_seconds = float(max_price_age_seconds)
        self.max_liquidity_age_seconds = float(max_liquidity_age_seconds)
        self.max_provider_seen_age_seconds = float(max_provider_seen_age_seconds)

    def validate(self, row: dict[str, Any]) -> dict[str, Any]:
        reasons: list[str] = []
        guards: list[str] = []
        code: str | None = None

        price = row.get("latest_price")
        if price is None or float(price or 0) <= 0:
            code = code or "NOT_OPENED_MISSING_PRICE"
            reasons.append("Missing or zero latest price.")
            guards.append("freshness_missing_price")

        if row.get("price_updated_at") is None:
            code = code or "NOT_OPENED_MISSING_PRICE_TIMESTAMP"
            reasons.append("Price timestamp missing.")
            guards.append("freshness_missing_price_timestamp")

        liq = row.get("latest_liquidity")
        if liq is None:
            code = code or "NOT_OPENED_MISSING_LIQUIDITY"
            reasons.append("Liquidity missing.")
            guards.append("freshness_missing_liquidity")

        if row.get("liquidity_updated_at") is None:
            code = code or "NOT_OPENED_MISSING_LIQUIDITY_TIMESTAMP"
            reasons.append("Liquidity timestamp missing.")
            guards.append("freshness_missing_liquidity_timestamp")

        if not row.get("source_provider"):
            code = code or "NOT_OPENED_SOURCE_PROVIDER_MISSING"
            reasons.append("Source provider missing.")
            guards.append("freshness_missing_source_provider")

        price_age = row.get("price_age_seconds")
        if price_age is not None and float(price_age) > self.max_price_age_seconds:
            code = code or "NOT_OPENED_STALE_MARKET_DATA"
            reasons.append(
                f"Price age {float(price_age):.0f}s exceeds {self.max_price_age_seconds:.0f}s."
            )
            guards.append("freshness_stale_price")

        liq_age = row.get("liquidity_age_seconds")
        if liq_age is not None and float(liq_age) > self.max_liquidity_age_seconds:
            code = code or "NOT_OPENED_STALE_MARKET_DATA"
            reasons.append(
                f"Liquidity age {float(liq_age):.0f}s exceeds {self.max_liquidity_age_seconds:.0f}s."
            )
            guards.append("freshness_stale_liquidity")

        seen_age = row.get("provider_seen_age_seconds")
        if seen_age is not None and float(seen_age) > self.max_provider_seen_age_seconds:
            code = code or "NOT_OPENED_PROVIDER_LAST_SEEN_TOO_OLD"
            reasons.append(
                f"Provider last seen age {float(seen_age):.0f}s exceeds "
                f"{self.max_provider_seen_age_seconds:.0f}s."
            )
            guards.append("freshness_provider_last_seen")

        passed = code is None
        return {
            "passed": passed,
            "rejection_code": code,
            "rejection_reasons": reasons,
            "blocking_guards": guards,
            "freshness_gate_status": "pass" if passed else "fail",
        }


def compute_tradability_status(
    *,
    passed: bool,
    for_open: bool,
    row: dict[str, Any],
    freshness_status: str,
    blocking_guards: list[str] | None = None,
    rejection_code: str | None = None,
) -> str:
    """Map the gate outcome onto the fixed AE13I tradability_status enum.

    Only values from ALLOWED_TRADABILITY_STATUSES may be returned.
    """
    guards = set(blocking_guards or [])

    if row.get("pair_token_identity_conflict"):
        status = "pair_token_identity_conflict"
    elif row.get("address_role_status") == "ambiguous" or row.get("is_ambiguous"):
        status = "ambiguous_address_role"
    elif row.get("unsupported_chain") or rejection_code == "UNSUPPORTED_CHAIN":
        status = "unsupported_chain"
    elif row.get("historical_only") or row.get("is_historical_only") or row.get("data_mode") == "historical":
        status = "historical_only"
    elif row.get("inactive_pool"):
        status = "inactive_pool"
    elif row.get("provider_unresolved") or row.get("provider_status") == "unresolved":
        status = "provider_unresolved"
    elif row.get("watchlist_only"):
        status = "watchlist_only"
    elif row.get("resolver_only"):
        status = "resolver_only"
    elif row.get("explorer_only"):
        status = "explorer_only"
    elif "freshness_missing_price" in guards:
        status = "missing_price"
    elif "freshness_missing_liquidity" in guards:
        status = "missing_liquidity"
    elif "freshness_missing_price_timestamp" in guards:
        status = "missing_price_timestamp"
    elif "freshness_missing_liquidity_timestamp" in guards:
        status = "missing_liquidity_timestamp"
    elif not passed and freshness_status == "fail":
        status = "stale_market_data"
    elif passed:
        # Passing always means the market row is currently tradable — for_open
        # and watch-only evaluations both resolve to tradable_now.
        status = "tradable_now"
    elif for_open:
        status = "not_tradable_without_market_price"
    else:
        status = "unknown"

    if status not in ALLOWED_TRADABILITY_STATUSES:
        status = "unknown"
    return status


def validate_market_data_gate(
    row: dict[str, Any],
    *,
    for_open: bool = True,
    skip_reentry: bool = False,
    skip_stagnant: bool = False,
    max_price_age_seconds: float = DEFAULT_MAX_PRICE_AGE_SECONDS,
    max_liquidity_age_seconds: float = DEFAULT_MAX_LIQUIDITY_AGE_SECONDS,
    max_provider_seen_age_seconds: float = DEFAULT_MAX_PROVIDER_SEEN_AGE_SECONDS,
) -> dict[str, Any]:
    checked_at = datetime.now(timezone.utc).isoformat()
    rejection_reasons: list[str] = []
    blocking_guards: list[str] = []
    primary_blocker: str | None = None
    rejection_code: str | None = None

    ctx = IdentityNormalizer.normalize(row)
    ctx = AddressRoleClassifier.classify(ctx)
    ctx = ProvenanceEnricher.enrich(ctx)

    address_role_status = str(ctx.get("address_role_status") or "unknown")

    if ctx.get("pair_token_identity_conflict"):
        rejection_code = "NOT_OPENED_PAIR_TOKEN_IDENTITY_CONFLICT"
        rejection_reasons.append("Pair address conflicts with token mint/contract identity.")
        blocking_guards.append("identity_pair_token_conflict")
        primary_blocker = primary_blocker or rejection_code

    if ctx.get("address_role_status") == "ambiguous" or ctx.get("is_ambiguous"):
        rejection_code = rejection_code or "NOT_OPENED_AMBIGUOUS_ADDRESS_ROLE"
        rejection_reasons.append(ctx.get("address_role_note") or "Ambiguous address role.")
        blocking_guards.append("address_role_ambiguous")
        primary_blocker = primary_blocker or rejection_code

    if ctx.get("historical_only") or ctx.get("is_historical_only") or ctx.get("data_mode") == "historical":
        rejection_code = rejection_code or "NOT_OPENED_HISTORICAL_ONLY"
        rejection_reasons.append("Historical-only market data cannot open positions.")
        blocking_guards.append("historical_only")
        primary_blocker = primary_blocker or rejection_code

    freshness = FreshnessValidator(
        max_price_age_seconds=max_price_age_seconds,
        max_liquidity_age_seconds=max_liquidity_age_seconds,
        max_provider_seen_age_seconds=max_provider_seen_age_seconds,
    ).validate(ctx)
    freshness_gate_status = str(freshness.get("freshness_gate_status") or "unknown")
    if not freshness.get("passed"):
        rejection_code = rejection_code or freshness.get("rejection_code")
        rejection_reasons.extend(freshness.get("rejection_reasons") or [])
        blocking_guards.extend(freshness.get("blocking_guards") or [])
        primary_blocker = primary_blocker or str(freshness.get("rejection_code"))

    if for_open and not skip_stagnant:
        stagnant = evaluate_stagnant_price(ctx)
        if not stagnant.get("passed"):
            code = stagnant.get("rejection_code")
            rejection_code = rejection_code or code
            if stagnant.get("rejection_reason"):
                rejection_reasons.append(str(stagnant.get("rejection_reason")))
            blocking_guards.extend(stagnant.get("blocking_guards") or [])
            primary_blocker = primary_blocker or str(code)

    if for_open and not skip_reentry:
        block = check_reentry_block(
            ctx.get("pair_address"),
            ctx.get("chain"),
            ctx.get("token_contract_address"),
            ctx.get("token_mint_address"),
            ctx.get("symbol"),
        )
        if block:
            if block.get("block_kind") == "system_close":
                blocking_guards.append("reentry_block_active")
                signal = check_system_reentry_signal(ctx, block.get("close_snapshot"))
                if not signal.get("passed"):
                    rejection_code = rejection_code or REENTRY_BLOCK_NO_NEW_SIGNAL
                    rejection_reasons.append(
                        "System reentry cooldown active without meaningful new signal."
                    )
                    blocking_guards.append("system_reentry_no_new_signal")
                    primary_blocker = primary_blocker or REENTRY_BLOCK_NO_NEW_SIGNAL
            else:
                # AE13I: manual-close reentry cooldown — exact rejection code/guard
                # required so UI/API can render the dedicated blocked-decision state.
                blocking_guards.append("manual_reentry_block")
                rejection_code = rejection_code or MANUAL_REENTRY_BLOCK_ACTIVE
                rejection_reasons.append(
                    f"Manual close cooldown active until {block.get('expires_at_utc')}."
                )
                primary_blocker = primary_blocker or MANUAL_REENTRY_BLOCK_ACTIVE
            ctx["active_reentry_block"] = block

    passed = rejection_code is None and not rejection_reasons
    provenance_status = str(ctx.get("provenance_status") or "unknown")
    if provenance_status != "ok" and passed:
        market_data_status = "incomplete_provenance"
    elif freshness_gate_status == "fail":
        market_data_status = "stale"
    elif passed:
        market_data_status = "ok"
    else:
        market_data_status = "blocked"

    tradability_status = compute_tradability_status(
        passed=passed,
        for_open=for_open,
        row=ctx,
        freshness_status=freshness_gate_status,
        blocking_guards=blocking_guards,
        rejection_code=rejection_code,
    )

    # Semantic labels are informational only — captured for display, but they
    # must never flip `passed` to True. Freshness/identity/reentry gates above
    # are the sole authority over `passed`.
    semantic_status = ctx.get("semantic_status") or ctx.get("semantic_signal_family")

    decision: str | None = None
    if rejection_code == MANUAL_REENTRY_BLOCK_ACTIVE:
        decision = "BLOCKED_MANUAL_REENTRY_COOLDOWN"
    elif passed and for_open:
        decision = "TRADABLE_NOW"
    elif not passed:
        decision = "BLOCKED"

    result = MarketDataGateResult(
        passed=passed,
        tradability_status=tradability_status,
        freshness_gate_status=freshness_gate_status,
        primary_blocker=primary_blocker,
        rejection_code=rejection_code,
        rejection_reasons=rejection_reasons,
        blocking_guards=blocking_guards,
        provenance_status=provenance_status,
        address_role_status=address_role_status,
        market_data_status=market_data_status,
        semantic_status=semantic_status,
        decision=decision,
        candidate_context=ctx,
        checked_at_utc=checked_at,
    )
    return result.to_dict()