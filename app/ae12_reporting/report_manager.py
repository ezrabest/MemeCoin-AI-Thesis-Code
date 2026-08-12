"""AE12ReportManager — cached, read-only access to latest AE12 audit artifacts."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

from . import loaders
from . import summary as summary_builders
from .latest import discover_all_latest_roots

DEFAULT_CACHE_TTL_SECONDS = 300
MISSED_WINNERS_CACHE_MAX = 500
QWEN_SAMPLE_LIMIT = 50


class AE12ReportManager:
    """
    Discovers latest AE12 output roots, loads summary JSON/CSV via loaders,
    caches parsed results in memory, and exposes read-only methods for API/docs.

    Does not mutate source files. Missing files -> status MISSING (no crash).
    """

    def __init__(
        self,
        project_root: Path | str | None = None,
        *,
        ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
        maturation_root: Path | str | None = None,
        census_root: Path | str | None = None,
        quality_root: Path | str | None = None,
        taxonomy_root: Path | str | None = None,
    ) -> None:
        if project_root is None:
            project_root = Path(__file__).resolve().parents[2]
        self.project_root = Path(project_root)
        self.ttl_seconds = int(ttl_seconds)
        self._forced_maturation = Path(maturation_root) if maturation_root else None
        self._forced_census = Path(census_root) if census_root else None
        self._forced_quality = Path(quality_root) if quality_root else None
        self._forced_taxonomy = Path(taxonomy_root) if taxonomy_root else None
        self._lock = threading.RLock()
        self._cache: dict[str, tuple[float, Any]] = {}
        self._load_counts: dict[str, int] = {}
        self._last_refresh_monotonic: float | None = None

    # -- roots ---------------------------------------------------------------

    def discover_roots(self, *, force: bool = False) -> dict[str, Path | None]:
        def _load() -> dict[str, Path | None]:
            discovered = discover_all_latest_roots(self.project_root)
            return {
                "maturation_root": self._forced_maturation or discovered["maturation_root"],
                "census_root": self._forced_census or discovered["census_root"],
                "quality_root": self._forced_quality or discovered["quality_root"],
                "taxonomy_root": self._forced_taxonomy or discovered.get("taxonomy_root"),
                "sentimentfix_root": discovered.get("sentimentfix_root"),
                "semantic_classifier_root": discovered.get("semantic_classifier_root"),
                "gemini_adjudication_root": discovered.get("gemini_adjudication_root"),
                "manual_review_drilldown_root": discovered.get("manual_review_drilldown_root"),
            }

        return self._cached("roots", _load, force=force)

    def maturation_root(self) -> Path | None:
        return self.discover_roots().get("maturation_root")

    # ── cache internals ────────────────────────────────────────────────────

    def _cache_meta(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ttl_seconds": self.ttl_seconds,
                "cached_keys": sorted(self._cache.keys()),
                "load_counts": dict(self._load_counts),
                "last_refresh_monotonic": self._last_refresh_monotonic,
                "entries": {
                    k: {"age_seconds": round(time.monotonic() - ts, 3)}
                    for k, (ts, _) in self._cache.items()
                },
            }

    def _cached(self, key: str, loader_fn, *, force: bool = False) -> Any:
        with self._lock:
            now = time.monotonic()
            if not force and key in self._cache:
                ts, value = self._cache[key]
                if (now - ts) < self.ttl_seconds:
                    return value
            value = loader_fn()
            self._cache[key] = (now, value)
            self._load_counts[key] = self._load_counts.get(key, 0) + 1
            return value

    def refresh(self) -> dict[str, Any]:
        """Clear memory cache and reload root discovery. Does not mutate source data."""
        with self._lock:
            self._cache.clear()
            self._last_refresh_monotonic = time.monotonic()
        roots = self.discover_roots(force=True)
        # Warm critical summary/gate caches
        self._load_summary(force=True)
        self._load_gate(force=True)
        return {
            "refreshed": True,
            "read_only": True,
            "mutated_source_data": False,
            "roots": {k: str(v) if v else None for k, v in roots.items()},
            "cache": self._cache_meta(),
        }

    # ── underlying loads ───────────────────────────────────────────────────

    def _load_summary(self, *, force: bool = False) -> dict[str, Any]:
        def _load() -> dict[str, Any]:
            return loaders.load_maturation_summary(self.maturation_root())

        return self._cached("maturation_summary", _load, force=force)

    def _load_gate(self, *, force: bool = False) -> dict[str, Any]:
        def _load() -> dict[str, Any]:
            return loaders.load_readiness_gate(self.maturation_root())

        return self._cached("readiness_gate", _load, force=force)

    def _load_wallet(self, *, force: bool = False) -> dict[str, Any]:
        def _load() -> dict[str, Any]:
            return loaders.load_wallet_safety(self.maturation_root())

        return self._cached("wallet_safety", _load, force=force)

    def _load_census(self, *, force: bool = False) -> dict[str, Any]:
        def _load() -> dict[str, Any]:
            roots = self.discover_roots()
            return loaders.load_census_summary(roots.get("census_root"))

        return self._cached("census_summary", _load, force=force)

    def _load_trade_csv(self, *, force: bool = False) -> dict[str, Any]:
        def _load() -> dict[str, Any]:
            return loaders.load_trade_vs_no_trade_csv(self.maturation_root())

        return self._cached("trade_vs_no_trade_csv", _load, force=force)

    def _load_strict_csv(self, *, force: bool = False) -> dict[str, Any]:
        def _load() -> dict[str, Any]:
            return loaders.load_strict_vs_exploration_csv(self.maturation_root())

        return self._cached("strict_vs_exploration_csv", _load, force=force)

    def _load_missed(self, *, force: bool = False) -> dict[str, Any]:
        def _load() -> dict[str, Any]:
            return loaders.load_missed_winners_csv(
                self.maturation_root(), limit=MISSED_WINNERS_CACHE_MAX
            )

        return self._cached("missed_winners", _load, force=force)

    def _load_qwen_sample(self, *, force: bool = False) -> dict[str, Any]:
        def _load() -> dict[str, Any]:
            return loaders.load_qwen_linkage_sample(
                self.maturation_root(), limit=QWEN_SAMPLE_LIMIT
            )

        return self._cached("qwen_sample", _load, force=force)

    def _load_missing_warnings_sample(self, *, force: bool = False) -> dict[str, Any]:
        def _load() -> dict[str, Any]:
            return loaders.load_missing_data_warning_sample(self.maturation_root(), limit=100)

        return self._cached("missing_warnings_sample", _load, force=force)

    def _summary_data(self) -> dict[str, Any] | None:
        loaded = self._load_summary()
        return loaded.get("data") if loaded.get("status") == "OK" else None

    def _gate_data(self) -> dict[str, Any] | None:
        loaded = self._load_gate()
        if loaded.get("status") == "OK":
            return loaded.get("data")
        # Fall back to nested gate in summary
        summary = self._summary_data() or {}
        return summary.get("readiness_gate")

    def _wallet_data(self) -> dict[str, Any] | None:
        loaded = self._load_wallet()
        if loaded.get("status") == "OK":
            return loaded.get("data")
        summary = self._summary_data() or {}
        return summary.get("wallet_safety")

    # ── public API ─────────────────────────────────────────────────────────

    def get_status(self) -> dict[str, Any]:
        roots = self.discover_roots()
        summary_load = self._load_summary()
        gate_load = self._load_gate()
        return summary_builders.build_status_payload(
            maturation_root=str(roots["maturation_root"]) if roots.get("maturation_root") else None,
            census_root=str(roots["census_root"]) if roots.get("census_root") else None,
            quality_root=str(roots["quality_root"]) if roots.get("quality_root") else None,
            summary=self._summary_data(),
            gate=self._gate_data(),
            summary_load=summary_load,
            gate_load=gate_load,
            cache_meta=self._cache_meta(),
        )

    def get_runtime_collection_status(self) -> dict[str, Any]:
        census_load = self._load_census()
        census = census_load.get("data") if census_load.get("status") == "OK" else None
        return summary_builders.build_runtime_collection_payload(
            census=census,
            census_load=census_load,
            summary=self._summary_data(),
        )

    def get_forward_evidence_summary(self) -> dict[str, Any]:
        summary_load = self._load_summary()
        warn_load = self._load_missing_warnings_sample()
        warn_data = warn_load.get("data") if warn_load.get("status") == "OK" else None
        return summary_builders.build_forward_evidence_payload(
            summary=self._summary_data(),
            summary_load=summary_load,
            missing_warnings_sample=warn_data,
        )

    def get_missed_winners(self, limit: int = 100) -> dict[str, Any]:
        limit = max(1, min(int(limit), MISSED_WINNERS_CACHE_MAX))
        missed_load = self._load_missed()
        return summary_builders.build_missed_winners_payload(
            summary=self._summary_data(),
            missed_load=missed_load,
            limit=limit,
        )

    def get_trade_vs_no_trade(self) -> dict[str, Any]:
        return summary_builders.build_trade_vs_no_trade_payload(
            summary=self._summary_data(),
            csv_load=self._load_trade_csv(),
        )

    def get_strict_vs_exploration(self) -> dict[str, Any]:
        return summary_builders.build_strict_vs_exploration_payload(
            summary=self._summary_data(),
            csv_load=self._load_strict_csv(),
        )

    def get_qwen_linkage_summary(self) -> dict[str, Any]:
        return summary_builders.build_qwen_linkage_payload(
            summary=self._summary_data(),
            sample_load=self._load_qwen_sample(),
        )

    def get_safety_summary(self) -> dict[str, Any]:
        return summary_builders.build_safety_payload(
            wallet=self._wallet_data(),
            wallet_load=self._load_wallet(),
            gate=self._gate_data(),
        )

    def get_final_report_summary(self) -> dict[str, Any]:
        return summary_builders.build_final_report_summary(
            status=self.get_status(),
            forward=self.get_forward_evidence_summary(),
            trade_vs=self.get_trade_vs_no_trade(),
            strict=self.get_strict_vs_exploration(),
            qwen=self.get_qwen_linkage_summary(),
            safety=self.get_safety_summary(),
            missed=self.get_missed_winners(limit=20),
            runtime=self.get_runtime_collection_status(),
        )

    def get_signal_taxonomy(self) -> dict[str, Any]:
        """Read-only cached view of latest signal taxonomy QA artifacts."""

        def _load_summary() -> dict[str, Any]:
            roots = self.discover_roots()
            return loaders.load_taxonomy_summary(roots.get("taxonomy_root"))

        def _load_gate() -> dict[str, Any]:
            roots = self.discover_roots()
            return loaders.load_taxonomy_gate(roots.get("taxonomy_root"))

        summary_load = self._cached("taxonomy_summary", _load_summary)
        gate_load = self._cached("taxonomy_gate", _load_gate)
        roots = self.discover_roots()
        if summary_load.get("status") != "OK" and gate_load.get("status") != "OK":
            return {
                "status": "MISSING",
                "missing_file": summary_load.get("missing_file")
                or gate_load.get("missing_file")
                or "ae12_signal_taxonomy_audit_*",
                "taxonomy_root": str(roots.get("taxonomy_root")) if roots.get("taxonomy_root") else None,
                "warning": (
                    "No reliable dual-axis semantic taxonomy audit found yet. "
                    "Run scripts/run_ae12_signal_taxonomy_audit.py. "
                    "Do not interpret opportunistic/social shares as proven semantic truth."
                ),
                "live_ready": False,
                "profitability_proven": False,
                "read_only": True,
                "legacy_diagnostic": True,
                "final_semantic_classification": False,
            }

        summary = summary_load.get("data") or {}
        gate = gate_load.get("data") or summary.get("gate") or {}
        return {
            "status": "OK",
            "taxonomy_root": str(roots.get("taxonomy_root")) if roots.get("taxonomy_root") else None,
            "gate_status": gate.get("status") or summary.get("gate_status"),
            "social_rows_found": gate.get("social_rows_found"),
            "opportunistic_rows_found": gate.get("opportunistic_rows_found"),
            "unknown_rows_found": gate.get("unknown_rows_found"),
            "social_share": gate.get("social_share"),
            "opportunistic_share": gate.get("opportunistic_share"),
            "unknown_share": gate.get("unknown_share"),
            "semantic_signal_family_distribution": gate.get("semantic_signal_family_distribution"),
            "trading_opportunity_state_distribution": gate.get("trading_opportunity_state_distribution"),
            "sticky_flag_found": gate.get("sticky_flag_found"),
            "conflated_axis_found": gate.get("conflated_axis_found"),
            "default_fallback_bug_found": gate.get("default_fallback_bug_found"),
            "social_linkage_bug_found": gate.get("social_linkage_bug_found"),
            "ui_mapping_bug_found": gate.get("ui_mapping_bug_found"),
            "fix_applied": gate.get("fix_applied"),
            "recommendation": gate.get("recommendation"),
            "limitations": gate.get("limitations"),
            "warning": (
                "No reliable dual-axis semantic category unless gate says otherwise. "
                "Opportunistic is a trading_opportunity_state, not a semantic_signal_family fallback."
            ),
            "live_ready": False,
            "profitability_proven": False,
            "qwen_trade_authority": False,
            "read_only": True,
            "legacy_diagnostic": True,
            "final_semantic_classification": False,
            "labels": {
                "research_only": True,
                "not_live_approved": True,
                "not_profitability_proven": True,
            },
        }

    def get_sentimentfix(self) -> dict[str, Any]:
        """Read-only AE12-SentimentFix gate + dual-axis summary."""

        def _load_summary() -> dict[str, Any]:
            roots = self.discover_roots()
            return loaders.load_sentimentfix_summary(roots.get("sentimentfix_root"))

        def _load_gate() -> dict[str, Any]:
            roots = self.discover_roots()
            return loaders.load_sentimentfix_gate(roots.get("sentimentfix_root"))

        summary_load = self._cached("sentimentfix_summary", _load_summary)
        gate_load = self._cached("sentimentfix_gate", _load_gate)
        roots = self.discover_roots()
        if summary_load.get("status") != "OK" and gate_load.get("status") != "OK":
            return {
                "status": "MISSING",
                "phase": "AE12-SentimentFix",
                "missing_file": summary_load.get("missing_file")
                or gate_load.get("missing_file")
                or "ae12_sentimentfix_*",
                "sentimentfix_root": str(roots.get("sentimentfix_root"))
                if roots.get("sentimentfix_root")
                else None,
                "warning": (
                    "AE12-SentimentFix audit not found. Run scripts/run_ae12_sentimentfix_audit.py. "
                    "Legacy opportunistic/social shares are unreliable until dual-axis outputs exist."
                ),
                "live_ready": False,
                "profitability_proven": False,
            "qwen_trade_authority": False,
            "read_only": True,
            "dual_axis_repair": True,
            "final_semantic_classification": False,
        }

        summary = summary_load.get("data") or {}
        gate = gate_load.get("data") or summary.get("gate") or {}
        return {
            "status": "OK",
            "phase": "AE12-SentimentFix",
            "sentimentfix_root": str(roots.get("sentimentfix_root"))
            if roots.get("sentimentfix_root")
            else None,
            "gate_status": gate.get("status") or summary.get("gate_status"),
            "prior_gate_status": gate.get("prior_gate_status"),
            "semantic_signal_family_distribution": summary.get("semantic_signal_family_distribution"),
            "trading_opportunity_state_distribution": summary.get(
                "trading_opportunity_state_distribution"
            ),
            "legacy_cluster_label_distribution": summary.get("legacy_cluster_label_distribution"),
            "semantic_unknown_share": gate.get("semantic_unknown_share")
            or summary.get("semantic_unknown_share"),
            "sentiment_records_count": gate.get("sentiment_records_count"),
            "sentiment_social_marker_rows": gate.get("sentiment_social_marker_rows"),
            "sticky_cluster_still_authoritative": gate.get("sticky_cluster_still_authoritative"),
            "sticky_cluster_soft_expiry_plan_created": gate.get(
                "sticky_cluster_soft_expiry_plan_created"
            ),
            "default_fallback_fixed": gate.get("default_fallback_fixed"),
            "semantic_linkage_gap_found": gate.get("semantic_linkage_gap_found"),
            "dual_axis_mapper_available": gate.get("dual_axis_mapper_available"),
            "runtime_future_fields_added": gate.get("runtime_future_fields_added"),
            "legacy_cluster_label_preserved": gate.get("legacy_cluster_label_preserved"),
            "historical_data_mutated": gate.get("historical_data_mutated"),
            "recommendation": gate.get("recommendation"),
            "limitations": gate.get("limitations"),
            "warning": (
                "Legacy opportunistic/social shares are unreliable until dual-axis fields exist. "
                "Missing semantic category displays as UNKNOWN/UNCLASSIFIED, not opportunistic."
            ),
            "live_ready": False,
            "profitability_proven": False,
            "qwen_trade_authority": False,
            "read_only": True,
            "dual_axis_repair": True,
            "final_semantic_classification": False,
        }

    def get_semantic_coin_classifier(self) -> dict[str, Any]:
        """Read-only summary for unique-asset semantic coin classifier."""

        def _load_summary() -> dict[str, Any]:
            roots = self.discover_roots()
            return loaders.load_semantic_classifier_summary(roots.get("semantic_classifier_root"))

        def _load_gate() -> dict[str, Any]:
            roots = self.discover_roots()
            return loaders.load_semantic_classifier_gate(roots.get("semantic_classifier_root"))

        summary_load = self._cached("semantic_classifier_summary", _load_summary)
        gate_load = self._cached("semantic_classifier_gate", _load_gate)
        roots = self.discover_roots()
        if summary_load.get("status") != "OK" and gate_load.get("status") != "OK":
            return {
                "status": "MISSING",
                "phase": "AE12-SentimentFix",
                "missing_file": summary_load.get("missing_file")
                or gate_load.get("missing_file")
                or "ae12_semantic_coin_classifier_*",
                "semantic_classifier_root": str(roots.get("semantic_classifier_root"))
                if roots.get("semantic_classifier_root")
                else None,
                "warning": (
                    "Semantic coin classifier outputs are missing. "
                    "Run scripts/run_ae12_semantic_coin_classifier.py."
                ),
                "trade_authority_used": False,
                "external_api_used": False,
                "local_classifier": True,
                "final_semantic_classification": False,
                "live_ready": False,
                "profitability_proven": False,
            }

        summary = summary_load.get("data") or {}
        gate = gate_load.get("data") or {}
        return {
            "status": "OK",
            "phase": "AE12-SentimentFix",
            "semantic_classifier_root": str(roots.get("semantic_classifier_root"))
            if roots.get("semantic_classifier_root")
            else None,
            "decision_gate": gate,
            "unique_assets_found": gate.get("unique_assets_found") or summary.get("unique_assets_found"),
            "unique_assets_classified": gate.get("unique_assets_classified")
            or summary.get("unique_assets_classified"),
            "social_count": gate.get("social_count"),
            "social_share": gate.get("social_share"),
            "non_social_opportunistic_count": gate.get("non_social_opportunistic_count"),
            "non_social_opportunistic_share": gate.get("opportunistic_share"),
            "non_social_infrastructure_count": gate.get("non_social_infrastructure_count"),
            "non_social_infrastructure_share": (
                gate.get("non_social_infrastructure_count") / max(int(gate.get("unique_assets_classified") or 1), 1)
            ),
            "unknown_count": gate.get("unknown_count"),
            "unknown_share": gate.get("unknown_share"),
            "manual_review_count": gate.get("manual_review_count"),
            "manual_review_share": gate.get("manual_review_share"),
            "unknown_reason_breakdown": gate.get("unknown_reason_breakdown") or summary.get("unknown_reason_breakdown"),
            "examples_by_class": summary.get("examples_by_class") or {},
            "classifier_status": gate.get("classifier_status"),
            "classifier_model": gate.get("classifier_model"),
            "classifier_version": gate.get("classifier_version"),
            "rubric_version": gate.get("rubric_version"),
            "classification_cache_used": gate.get("classification_cache_used"),
            "cache_key_fields": gate.get("cache_key_fields"),
            "cache_key_uses_evidence_hash": gate.get("cache_key_uses_evidence_hash"),
            "classifier_safety_status": (
                "FORBIDDEN_OUTPUTS_REJECTED"
                if gate.get("forbidden_trade_language_found")
                else "PASS_NO_FORBIDDEN_TERMS_FOUND"
            ),
            "trade_authority_used": False,
            "external_api_used": False,
            "warning": (
                "Semantic classifier is reporting only, not trade authority. "
                "UNKNOWN means insufficient evidence or not evaluated."
            ),
            "local_classifier": True,
            "final_semantic_classification": False,
            "live_ready": False,
            "profitability_proven": False,
            "qwen_trade_authority": False,
        }

    def get_gemini_semantic_adjudication(self) -> dict[str, Any]:
        """Read-only summary for Gemini semantic adjudication (final SOCIAL vs OP.SUSPECTED)."""
        from app.ae12_sentimentfix.adjudication_safety_status import (
            gate_allowed_with_safety,
            resolve_safety_audit_status,
        )
        from app.ae12_sentimentfix.coin_level_aggregation import load_or_derive_coin_level

        def _load_summary() -> dict[str, Any]:
            roots = self.discover_roots()
            return loaders.load_gemini_adjudication_summary(roots.get("gemini_adjudication_root"))

        def _load_gate() -> dict[str, Any]:
            roots = self.discover_roots()
            return loaders.load_gemini_adjudication_gate(roots.get("gemini_adjudication_root"))

        def _load_safety() -> dict[str, Any]:
            roots = self.discover_roots()
            return loaders.load_gemini_safety_audit(roots.get("gemini_adjudication_root"))

        summary_load = self._cached("gemini_adjudication_summary", _load_summary)
        gate_load = self._cached("gemini_adjudication_gate", _load_gate)
        safety_load = self._cached("gemini_safety_audit", _load_safety)
        roots = self.discover_roots()
        if summary_load.get("status") != "OK" and gate_load.get("status") != "OK":
            return {
                "status": "MISSING",
                "phase": "AE12-SentimentFix",
                "missing_file": summary_load.get("missing_file")
                or gate_load.get("missing_file")
                or "ae12_gemini_semantic_adjudication_*",
                "gemini_adjudication_root": str(roots.get("gemini_adjudication_root"))
                if roots.get("gemini_adjudication_root")
                else None,
                "warning": (
                    "Gemini semantic adjudication outputs are missing. "
                    "Run scripts/run_ae12_gemini_semantic_adjudication.py."
                ),
                "final_semantic_adjudication": False,
                "trade_authority_used": False,
                "external_api_used": False,
                "live_ready": False,
                "profitability_proven": False,
            }

        summary = summary_load.get("data") or {}
        gate = gate_load.get("data") or {}
        safety = dict(safety_load.get("data") or summary.get("safety_audit") or {})
        # Correct inconsistent historical FAIL when rejections were enforced and unused.
        safety_status = resolve_safety_audit_status(safety)
        safety["status"] = safety_status
        gate_status = gate.get("status") or summary.get("gate_status")
        if safety_status == "FAIL" and gate_status in {
            "PASS_GEMINI_ADJUDICATION_READY",
            "PASS_WITH_OP_SUSPECTED_LIMITATION",
        }:
            gate_status = "FAIL_INVALID_GEMINI_OUTPUT"

        def _load_coin_level() -> dict[str, Any]:
            return load_or_derive_coin_level(roots.get("gemini_adjudication_root")) or {}

        coin_level = self._cached("gemini_coin_level_counts", _load_coin_level)
        pair_counts = coin_level.get("pair_asset_counts") or summary.get("pair_asset_counts") or {}
        coin_counts = coin_level.get("coin_level_counts") or summary.get("coin_level_counts") or {}
        use_coin = bool(coin_counts)
        main_counts = coin_counts if use_coin else {}

        payload = {
            "status": "OK",
            "phase": "AE12-SentimentFix",
            "gemini_adjudication_root": str(roots.get("gemini_adjudication_root"))
            if roots.get("gemini_adjudication_root")
            else None,
            "decision_gate": gate,
            "gate_status": gate_status,
            "unique_assets_input": gate.get("unique_assets_input") or summary.get("unique_assets_input"),
            "unique_assets_adjudicated": gate.get("unique_assets_adjudicated")
            or summary.get("unique_assets_adjudicated"),
            # Pair-level retained as audit detail only
            "pair_asset_counts": pair_counts,
            "pair_level_counts_role": "audit_detail",
            # Coin-level is final UI answer when available
            "coin_level_counts": coin_counts,
            "count_level_used_for_main_ui": "coin_level" if use_coin else "pair_level_fallback",
            "identity_resolution_method_distribution": coin_level.get(
                "identity_resolution_method_distribution"
            )
            or summary.get("identity_resolution_method_distribution")
            or {},
            "identity_warning_count": coin_level.get("identity_warning_count")
            or summary.get("identity_warning_count")
            or 0,
            "conflict_count": coin_level.get("conflict_count") or summary.get("conflict_count") or 0,
            # Main UI fields prefer coin-level
            "social_confirmed_count": main_counts.get("coin_social_confirmed_count")
            if use_coin
            else gate.get("social_confirmed_count"),
            "social_confirmed_share": main_counts.get("coin_social_confirmed_share")
            if use_coin
            else gate.get("social_confirmed_share"),
            "non_social_opportunistic_confirmed_count": main_counts.get(
                "coin_non_social_opportunistic_confirmed_count"
            )
            if use_coin
            else gate.get("non_social_opportunistic_confirmed_count"),
            "opportunistic_confirmed_share": main_counts.get("coin_opportunistic_confirmed_share")
            if use_coin
            else gate.get("opportunistic_confirmed_share"),
            "opportunistic_suspected_count": main_counts.get("coin_opportunistic_suspected_count")
            if use_coin
            else gate.get("opportunistic_suspected_count"),
            "opportunistic_suspected_share": main_counts.get("coin_opportunistic_suspected_share")
            if use_coin
            else gate.get("opportunistic_suspected_share"),
            "non_social_infrastructure_confirmed_count": main_counts.get(
                "coin_non_social_infrastructure_confirmed_count"
            )
            if use_coin
            else gate.get("non_social_infrastructure_confirmed_count"),
            "infrastructure_share": (
                (main_counts.get("coin_non_social_infrastructure_confirmed_count") or 0)
                / max(int(main_counts.get("unique_coins_found") or 1), 1)
            )
            if use_coin
            else gate.get("infrastructure_share"),
            "manual_review_count": main_counts.get("coin_manual_review_count")
            if use_coin
            else gate.get("manual_review_count"),
            "manual_review_share": main_counts.get("coin_manual_review_share")
            if use_coin
            else gate.get("manual_review_share"),
            "unique_coins_found": main_counts.get("unique_coins_found") if use_coin else None,
            "raw_evidence_status_distribution": gate.get("raw_evidence_status_distribution")
            or summary.get("raw_evidence_status_distribution"),
            "examples_by_class": coin_level.get("examples_by_class")
            or summary.get("examples_by_class")
            or {},
            "gemini_model": gate.get("gemini_model"),
            "adjudicator_version": gate.get("adjudicator_version"),
            "rubric_version": gate.get("rubric_version"),
            "external_api_used": gate.get("external_api_used"),
            "gemini_used": gate.get("gemini_used"),
            "web_grounding_used": gate.get("web_grounding_used"),
            "model_knowledge_only_count": gate.get("model_knowledge_only_count"),
            "source_url_count": gate.get("source_url_count"),
            "freeze_once_policy_enabled": gate.get("freeze_once_policy_enabled"),
            "cache_key_fields": gate.get("cache_key_fields"),
            "safety_audit": safety,
            "safety_audit_status": safety_status,
            "rejected_outputs": safety.get("rejected_outputs"),
            "accepted_outputs": safety.get("accepted_outputs"),
            "output_used_after_rejection": safety.get("output_used_after_rejection", False),
            "accepted_classifications_with_forbidden_language": safety.get(
                "accepted_classifications_with_forbidden_language", 0
            ),
            "forbidden_trade_language_found": safety.get("forbidden_trade_language_found"),
            "forbidden_trade_key_found": safety.get("forbidden_trade_key_found"),
            "class_distribution": summary.get("class_distribution") or {},
            "ui_labels": {
                "OPPORTUNISTIC_SUSPECTED": "OP.SUSPECTED",
            },
            "final_semantic_adjudication": gate_allowed_with_safety(str(gate_status), safety),
            "semantic_reporting_only": True,
            "trade_authority_used": False,
            "ui_notes": [
                "Main counts are deduplicated by coin/token identity.",
                "Pair-level adjudications are retained for audit but are not final coin counts.",
                "OP.SUSPECTED means suspected opportunistic, not confirmed opportunistic.",
                "Gemini adjudication is semantic reporting only and is not trade authority.",
                "Web grounding was unavailable in this run; most labels are based on Gemini model knowledge only.",
            ],
            "warning": (
                "Gemini adjudication is semantic reporting only, not trade authority. "
                "OP.SUSPECTED means suspected opportunistic due to insufficient social/infrastructure evidence; "
                "it is not confirmed opportunistic. "
                "Main counts are deduplicated by coin/token identity. "
                "Pair-level adjudications are audit detail only. "
                "Web grounding was unavailable in this run; most labels are based on Gemini model knowledge only."
            ),
            "live_ready": False,
            "profitability_proven": False,
            "qwen_trade_authority": False,
            "read_only": True,
        }
        drill = self.get_manual_review_drilldown()
        if drill.get("status") == "OK":
            payload["manual_review_drilldown"] = {
                "completed_locally": True,
                "gate_status": drill.get("gate_status"),
                "drilldown_rule_version": drill.get("drilldown_rule_version"),
                "unknown_unresolved_count": drill.get("unknown_unresolved_count"),
                "manual_review_remaining_count": drill.get("manual_review_remaining_count"),
                "updated_coin_level_counts": drill.get("updated_coin_level_counts"),
                "resolution_rule_distribution": drill.get("resolution_rule_distribution"),
            }
            # Prefer post-drilldown coin counts for main UI when available
            uc = drill.get("updated_coin_level_counts") or {}
            if uc:
                payload["coin_level_counts_after_drilldown"] = uc
                payload["count_level_used_for_main_ui"] = "coin_level_after_drilldown"
                payload["social_confirmed_count"] = uc.get("coin_social_confirmed_count")
                payload["social_confirmed_share"] = uc.get("coin_social_confirmed_share")
                payload["non_social_opportunistic_confirmed_count"] = uc.get(
                    "coin_non_social_opportunistic_confirmed_count"
                )
                payload["opportunistic_confirmed_share"] = uc.get(
                    "coin_opportunistic_confirmed_share"
                )
                payload["opportunistic_suspected_count"] = uc.get(
                    "coin_opportunistic_suspected_count"
                )
                payload["opportunistic_suspected_share"] = uc.get(
                    "coin_opportunistic_suspected_share"
                )
                payload["manual_review_count"] = uc.get("coin_manual_review_remaining_count")
                payload["manual_review_share"] = uc.get("coin_manual_review_remaining_share")
                payload["unknown_unresolved_count"] = uc.get("coin_unknown_unresolved_count")
                payload["unknown_unresolved_share"] = uc.get("coin_unknown_unresolved_share")
                payload["ui_notes"] = list(payload.get("ui_notes") or []) + [
                    "Manual review drilldown: completed locally.",
                    "UNKNOWN_UNRESOLVED means unresolved without external evidence.",
                ]
        else:
            payload["manual_review_drilldown"] = {"completed_locally": False}
        return payload

    def get_manual_review_drilldown(self) -> dict[str, Any]:
        """Read-only local manual-review drilldown summary (no Gemini)."""

        def _load_summary() -> dict[str, Any]:
            roots = self.discover_roots()
            return loaders.load_manual_review_drilldown_summary(
                roots.get("manual_review_drilldown_root")
            )

        def _load_gate() -> dict[str, Any]:
            roots = self.discover_roots()
            return loaders.load_manual_review_drilldown_gate(
                roots.get("manual_review_drilldown_root")
            )

        summary_load = self._cached("manual_review_drilldown_summary", _load_summary)
        gate_load = self._cached("manual_review_drilldown_gate", _load_gate)
        roots = self.discover_roots()
        if summary_load.get("status") != "OK" and gate_load.get("status") != "OK":
            return {
                "status": "MISSING",
                "phase": "AE12-SentimentFix",
                "missing_file": summary_load.get("missing_file")
                or gate_load.get("missing_file")
                or "ae12_sentimentfix_manual_review_drilldown_*",
                "manual_review_drilldown_root": str(roots.get("manual_review_drilldown_root"))
                if roots.get("manual_review_drilldown_root")
                else None,
                "warning": (
                    "Manual-review drilldown not found. "
                    "Run scripts/run_ae12_manual_review_drilldown.py --no-external-apis."
                ),
                "external_api_used": False,
                "gemini_called_again": False,
                "trade_authority_used": False,
                "live_ready": False,
                "profitability_proven": False,
            }

        summary = summary_load.get("data") or {}
        gate = gate_load.get("data") or {}
        return {
            "status": "OK",
            "phase": "AE12-SentimentFix",
            "not_ae12_6": True,
            "manual_review_drilldown_root": str(roots.get("manual_review_drilldown_root"))
            if roots.get("manual_review_drilldown_root")
            else None,
            "gate_status": gate.get("status") or summary.get("gate_status"),
            "decision_gate": gate,
            "source_gemini_root": gate.get("source_gemini_root") or summary.get("source_gemini_root"),
            "drilldown_rule_version": gate.get("drilldown_rule_version")
            or summary.get("drilldown_rule_version"),
            "coin_aggregation_rule_version": gate.get("coin_aggregation_rule_version")
            or summary.get("coin_aggregation_rule_version"),
            "rubric_version": gate.get("rubric_version") or summary.get("rubric_version"),
            "manual_review_input_count": gate.get("manual_review_input_count")
            or summary.get("manual_review_input_count"),
            "manual_review_resolved_count": gate.get("manual_review_resolved_count")
            or summary.get("manual_review_resolved_count"),
            "unknown_unresolved_count": gate.get("unknown_unresolved_count")
            or summary.get("unknown_unresolved_count"),
            "manual_review_remaining_count": gate.get("manual_review_remaining_count")
            or summary.get("manual_review_remaining_count"),
            "updated_coin_level_counts": gate.get("updated_coin_level_distribution")
            or summary.get("updated_coin_level_counts")
            or {},
            "resolution_rule_distribution": gate.get("resolution_rule_distribution")
            or summary.get("resolution_rule_distribution")
            or {},
            "unresolved_examples": summary.get("unresolved_examples") or [],
            "external_api_used": False,
            "gemini_called_again": False,
            "trade_authority_used": False,
            "warning": (
                "UNKNOWN_UNRESOLVED means unresolved without external evidence; "
                "it is not opportunistic and not social."
            ),
            "live_ready": False,
            "profitability_proven": False,
            "read_only": True,
        }

    def get_source_bundle_for_docs(self) -> dict[str, Any]:
        """Bundle of parsed AE12 values for template-based doc generation."""
        roots = self.discover_roots()
        summary = self._summary_data() or {}
        gate = self._gate_data() or {}
        wallet = self._wallet_data() or {}
        return {
            "maturation_root": str(roots["maturation_root"]) if roots.get("maturation_root") else None,
            "census_root": str(roots["census_root"]) if roots.get("census_root") else None,
            "quality_root": str(roots["quality_root"]) if roots.get("quality_root") else None,
            "summary": summary,
            "gate": gate,
            "wallet": wallet,
            "forward": self.get_forward_evidence_summary(),
            "trade_vs_no_trade": self.get_trade_vs_no_trade(),
            "strict_vs_exploration": self.get_strict_vs_exploration(),
            "qwen": self.get_qwen_linkage_summary(),
            "safety": self.get_safety_summary(),
            "missed_winners": self.get_missed_winners(limit=25),
            "runtime": self.get_runtime_collection_status(),
            "final": self.get_final_report_summary(),
            "source_files": {
                "summary_json": loaders.maturation_paths(roots["maturation_root"])["summary"]
                if roots.get("maturation_root")
                else None,
                "gate_json": loaders.maturation_paths(roots["maturation_root"])["gate"]
                if roots.get("maturation_root")
                else None,
                "trade_vs_no_trade_csv": loaders.maturation_paths(roots["maturation_root"])[
                    "trade_vs_no_trade"
                ]
                if roots.get("maturation_root")
                else None,
                "strict_vs_exploration_csv": loaders.maturation_paths(roots["maturation_root"])[
                    "strict_vs_exploration"
                ]
                if roots.get("maturation_root")
                else None,
                "wallet_safety_json": loaders.maturation_paths(roots["maturation_root"])[
                    "wallet_safety"
                ]
                if roots.get("maturation_root")
                else None,
            },
        }


# App-level registry (singleton) — endpoints must not construct heavy loaders per call.
_MANAGER: AE12ReportManager | None = None
_MANAGER_LOCK = threading.Lock()


def get_ae12_report_manager(
    project_root: Path | str | None = None,
    *,
    ttl_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    maturation_root: Path | str | None = None,
) -> AE12ReportManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = AE12ReportManager(
                project_root=project_root,
                ttl_seconds=ttl_seconds,
                maturation_root=maturation_root,
            )
        return _MANAGER


def reset_ae12_report_manager() -> None:
    """Test helper — clears the process-level singleton."""
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = None
