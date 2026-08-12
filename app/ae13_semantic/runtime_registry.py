"""AE13 Runtime Semantic Registry — dynamic observe/classify/cache read-model.

UI must read from this registry for live semantic state, not from static AE12 audit files.
Does not call Gemini/Helius/Qwen/Ollama by default.

AE13C taxonomy:
- SOCIAL_CONFIRMED requires explicit social/public-good evidence (never invented locally).
- Obvious meme/hype coins → NON_SOCIAL_OPPORTUNISTIC_CONFIRMED or OPPORTUNISTIC_SUSPECTED.
- Unknown reserved for genuine insufficient evidence.
"""

from __future__ import annotations

import json
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.ae13b_product.copy import semantic_label_human

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
REGISTRY_PATH = DATA_DIR / "ae13b_runtime_semantic_registry.json"
TAXONOMY_VERSION = "AE13C_V1"
_LOCK = threading.RLock()
_INSTANCE: "SemanticRegistry | None" = None

# Bound growth for long-running demos — prevent unbounded memory / JSON growth
DEFAULT_MAX_ENTRIES = 2_000
_SAVE_EVERY_N_OBS = 25  # debounce persistence under high observation volume


def _env_max_entries() -> int:
    import os

    raw = os.getenv("AE13_SEMANTIC_REGISTRY_MAX_ENTRIES", "").strip()
    if not raw:
        return DEFAULT_MAX_ENTRIES
    try:
        return max(50, int(raw))
    except ValueError:
        return DEFAULT_MAX_ENTRIES


# Seed: obvious meme / opportunistic tokens (not social merely because they have communities)
MEME_OPPORTUNISTIC_SEEDS: dict[str, str] = {
    "WIF": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
    "DOGWIFHAT": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
    "PEPE": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
    "BONK": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
    "DOGE": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
    "SHIB": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
    "FLOKI": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
    "MEME": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
    "WOJAK": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
    "MYRO": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
    "POPCAT": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
    "MEW": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
    "BOME": "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED",
    "TRUMP": "OPPORTUNISTIC_SUSPECTED",
    "MAGA": "OPPORTUNISTIC_SUSPECTED",
}

MEME_PATTERN = re.compile(
    r"\b(DOG|CAT|FROG|PEPE|WIF|BONK|SHIB|DOGE|FLOKI|MEME|WOJAK|INU|MOON|CHAD|ELON)\b",
    re.I,
)

INFRA_HINTS = (
    "SOL",
    "ETH",
    "BTC",
    "USDC",
    "USDT",
    "JUP",
    "RAY",
    "ORCA",
    "MSOL",
    "JITOSOL",
    "PYTH",
    "RENDER",
    "LINK",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _base_symbol(symbol: str | None) -> str:
    raw = str(symbol or "").strip().upper()
    if not raw:
        return ""
    # WIF/SOL, WIF/WETH, WIF-SOL → WIF
    for sep in ("/", "-", "_", " "):
        if sep in raw:
            raw = raw.split(sep)[0].strip()
            break
    return raw


def _identity_key(*, pair_address: str | None, coin_id: Any, symbol: str | None) -> str:
    if pair_address:
        return f"pair:{str(pair_address).strip().lower()}"
    if coin_id is not None:
        return f"coin:{coin_id}"
    return f"symbol:{(symbol or 'unknown').strip().upper()}"


def _local_classify(candidate: dict[str, Any]) -> dict[str, Any]:
    """Rule-based local classification — honest confidence, no external APIs."""
    symbol_raw = str(candidate.get("symbol") or "")
    base = _base_symbol(symbol_raw)
    name = str(candidate.get("name") or "").upper()
    liq = float(candidate.get("liquidity_usd") or candidate.get("latest_liquidity") or 0)
    vol = float(candidate.get("volume_24h") or candidate.get("latest_volume_24h") or 0)
    legacy = str(candidate.get("cluster_label") or candidate.get("legacy_cluster") or "").upper()
    user_hint = str(candidate.get("user_expected_category") or "").lower()

    social_source_available = bool(candidate.get("social_source_available"))
    social_mission_evidence = bool(candidate.get("social_mission_evidence"))
    evidence: list[str] = ["local_rules_only", f"taxonomy:{TAXONOMY_VERSION}"]

    family = "UNKNOWN_INSUFFICIENT_EVIDENCE"
    confidence = "low"
    needs_review = True
    unresolved_reason = "Insufficient evidence for confirmation."
    opportunity = "UNKNOWN"

    # Explicit social/public-good evidence only → SOCIAL_CONFIRMED
    if social_mission_evidence and social_source_available:
        family = "SOCIAL_CONFIRMED"
        confidence = "high"
        needs_review = False
        unresolved_reason = ""
        opportunity = "WATCH"
        evidence.append("explicit_social_mission_evidence")
    elif base in MEME_OPPORTUNISTIC_SEEDS:
        family = MEME_OPPORTUNISTIC_SEEDS[base]
        confidence = "high" if family.endswith("CONFIRMED") else "medium"
        needs_review = family == "OPPORTUNISTIC_SUSPECTED"
        unresolved_reason = ""
        opportunity = "DEMO_CANDIDATE"
        evidence.append(f"seed_meme_opportunistic:{base}")
        evidence.append("meme_hype_is_not_social_confirmed")
    elif MEME_PATTERN.search(base) or MEME_PATTERN.search(name) or any(
        h in base or h in name for h in ("PEPE", "WIF", "BONK", "SHIB", "DOGE", "FLOKI", "MEME")
    ):
        family = "OPPORTUNISTIC_SUSPECTED"
        confidence = "medium"
        needs_review = True
        unresolved_reason = ""
        opportunity = "EXPLORATION_CANDIDATE"
        evidence.append("meme_pattern_heuristic")
        evidence.append("no_utility_social_mission_assumed")
    elif legacy in ("OPPORTUNISTIC_SPECULATIVE", "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED"):
        family = (
            "NON_SOCIAL_OPPORTUNISTIC_CONFIRMED"
            if "CONFIRMED" in legacy
            else "OPPORTUNISTIC_SUSPECTED"
        )
        confidence = "medium"
        needs_review = family == "OPPORTUNISTIC_SUSPECTED"
        unresolved_reason = ""
        opportunity = "DEMO_CANDIDATE"
        evidence.append("legacy_opportunistic_cluster")
    elif legacy in ("SOCIALLY_MOTIVATED",):
        # Legacy social cluster is diagnostic only — do NOT invent SOCIAL_CONFIRMED
        family = "NEEDS_REVIEW"
        confidence = "low"
        needs_review = True
        unresolved_reason = (
            "Legacy social cluster present, but SOCIAL_CONFIRMED requires explicit "
            "social/public-good mission evidence (not mere community hype)."
        )
        opportunity = "WATCH"
        evidence.append("legacy_cluster_diagnostic_only")
    elif base in INFRA_HINTS or any(h == base for h in INFRA_HINTS):
        family = "NON_SOCIAL_INFRASTRUCTURE_CONFIRMED"
        confidence = "medium"
        needs_review = False
        unresolved_reason = ""
        opportunity = "NOT_ACTIONABLE"
        evidence.append("infrastructure_symbol_seed")
    elif liq >= 50_000 and vol >= 20_000:
        family = "OPPORTUNISTIC_SUSPECTED"
        confidence = "medium"
        needs_review = True
        unresolved_reason = ""
        opportunity = "EXPLORATION_CANDIDATE"
        evidence.append("liquidity_volume_heuristic")
    else:
        family = "UNKNOWN_INSUFFICIENT_EVIDENCE"
        confidence = "low"
        needs_review = True
        unresolved_reason = "Not enough local evidence to resolve semantic family."
        opportunity = "UNKNOWN"
        evidence.append("insufficient_local_evidence")

    # User watchlist hints (never invent SOCIAL_CONFIRMED from user opinion alone)
    if user_hint in ("opportunistic", "user thinks opportunistic") and family.startswith("UNKNOWN"):
        family = "OPPORTUNISTIC_SUSPECTED"
        opportunity = "WATCH"
        evidence.append("user_watchlist_hint_opportunistic")
        unresolved_reason = ""
    elif user_hint in ("social", "user thinks social") and family.startswith("UNKNOWN"):
        family = "NEEDS_REVIEW"
        opportunity = "WATCH"
        evidence.append("user_watchlist_hint_social_needs_evidence")
        unresolved_reason = "User suspects social mission - awaiting explicit evidence."
    elif user_hint in ("investigation", "user wants investigation"):
        family = "NEEDS_REVIEW" if family.startswith("UNKNOWN") else family
        opportunity = "WATCH"
        evidence.append("user_watchlist_investigation")

    if family in ("UNKNOWN_INSUFFICIENT_EVIDENCE", "UNKNOWN_UNRESOLVED"):
        semantic_status = "Unresolved"
    elif family == "NEEDS_REVIEW" or needs_review:
        semantic_status = "Needs Review"
    else:
        semantic_status = "Classified"

    return {
        "semantic_signal_family": family,
        "semantic_label_human": semantic_label_human(family),
        "semantic_status": semantic_status,
        "trading_opportunity_state": opportunity,
        "classification_source": "runtime_local_rules",
        "taxonomy_version": TAXONOMY_VERSION,
        "confidence": confidence,
        "evidence_summary": "; ".join(evidence),
        "is_static_snapshot": False,
        "is_runtime_classified": family
        not in ("UNKNOWN_INSUFFICIENT_EVIDENCE", "UNKNOWN_UNRESOLVED"),
        "needs_review": needs_review,
        "unresolved_reason": unresolved_reason,
        "social_source_available": social_source_available,
        "gemini_used": False,
        "ollama_used": False,
        "qwen_used": False,
        "rss_linked": False,
        "parsed_base_symbol": base,
    }


class SemanticRegistry:
    """In-memory runtime semantic registry with optional JSON persistence.

    Growth is bounded by ``max_entries`` with LRU eviction (least recently seen).
    Pinned records (e.g. active watchlist identities) are preserved when possible.
    """

    def __init__(self, path: Path | None = None, max_entries: int | None = None) -> None:
        self.path = path or REGISTRY_PATH
        self.max_entries = int(max_entries) if max_entries is not None else _env_max_entries()
        self._records: dict[str, dict[str, Any]] = {}
        self._observation_count = 0
        self._eviction_count = 0
        self._obs_since_save = 0
        self._last_classification_update: str | None = None
        self._last_observation: str | None = None
        self._load()
        self._evict_if_needed(force_save=False)

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            records = data.get("records") if isinstance(data, dict) else None
            if isinstance(records, dict):
                self._records = {str(k): dict(v) for k, v in records.items() if isinstance(v, dict)}
            self._observation_count = int(data.get("observation_count") or 0)
            self._eviction_count = int(data.get("eviction_count") or 0)
            self._last_classification_update = data.get("last_classification_update")
            self._last_observation = data.get("last_observation")
            if data.get("max_entries") is not None:
                try:
                    loaded_max = int(data["max_entries"])
                    if loaded_max > 0:
                        self.max_entries = loaded_max
                except (TypeError, ValueError):
                    pass
        except (OSError, json.JSONDecodeError):
            self._records = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": "AE13C_RUNTIME_SEMANTIC_REGISTRY_V1",
            "taxonomy_version": TAXONOMY_VERSION,
            "updated_at_utc": _utc_now(),
            "record_count": len(self._records),
            "observation_count": self._observation_count,
            "eviction_count": self._eviction_count,
            "max_entries": self.max_entries,
            "last_classification_update": self._last_classification_update,
            "last_observation": self._last_observation,
            "records": self._records,
            "notes": [
                "Runtime registry is the live semantic read-model.",
                "Static AE12 snapshot is separate context only.",
                "SOCIAL_CONFIRMED requires explicit social/public-good evidence; local rules never invent it.",
                "Meme/hype community activity is opportunistic, not social.",
                f"Max entries={self.max_entries}; LRU eviction by last_seen_at (pinned preserved).",
            ],
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        tmp.replace(self.path)
        self._obs_since_save = 0

    def _maybe_save(self, *, force: bool = False) -> None:
        self._obs_since_save += 1
        if force or self._obs_since_save >= _SAVE_EVERY_N_OBS:
            self._save()

    def _evict_if_needed(self, *, force_save: bool = True) -> int:
        """Evict least-recently-seen non-pinned records when over max_entries."""
        overflow = len(self._records) - self.max_entries
        if overflow <= 0:
            return 0
        # Prefer evicting non-pinned; if still over, evict oldest pinned too
        ranked = sorted(
            self._records.items(),
            key=lambda kv: (
                1 if kv[1].get("pinned") else 0,
                str(kv[1].get("last_seen_at") or kv[1].get("first_seen_at") or ""),
            ),
        )
        removed = 0
        for key, _rec in ranked:
            if len(self._records) <= self.max_entries:
                break
            del self._records[key]
            removed += 1
        self._eviction_count += removed
        if removed and force_save:
            self._save()
        return removed

    def _should_reclassify(self, existing: dict[str, Any], candidate: dict[str, Any]) -> bool:
        if existing.get("taxonomy_version") != TAXONOMY_VERSION:
            return True
        fam = str(existing.get("semantic_signal_family") or "")
        if fam in ("UNKNOWN_UNRESOLVED", "UNKNOWN_INSUFFICIENT_EVIDENCE", "NEEDS_REVIEW"):
            # Re-run rules so seeds like WIF get upgraded
            return True
        if candidate.get("force_reclassify"):
            return True
        return False

    def observe_candidate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Observe a runtime candidate: classify new identities, refresh/reclassify as needed."""
        with _LOCK:
            key = _identity_key(
                pair_address=candidate.get("pair_address"),
                coin_id=candidate.get("coin_id") or candidate.get("id"),
                symbol=candidate.get("symbol"),
            )
            now = _utc_now()
            self._observation_count += 1
            self._last_observation = now
            existing = self._records.get(key)
            if existing and not self._should_reclassify(existing, candidate):
                existing["seen_count"] = int(existing.get("seen_count") or 1) + 1
                existing["last_seen_at"] = now
                if candidate.get("pinned"):
                    existing["pinned"] = True
                for fld in (
                    "liquidity_usd",
                    "volume_24h",
                    "price_usd",
                    "whale_score",
                    "symbol",
                    "name",
                ):
                    if candidate.get(fld) is not None:
                        existing[fld] = candidate.get(fld)
                    elif fld == "symbol" and candidate.get("symbol"):
                        existing["symbol"] = candidate.get("symbol")
                self._records[key] = existing
                self._evict_if_needed(force_save=False)
                self._maybe_save()
                return dict(existing)

            classified = _local_classify(candidate)
            self._last_classification_update = now
            if existing:
                rec = {
                    **existing,
                    **classified,
                    "seen_count": int(existing.get("seen_count") or 1) + 1,
                    "last_seen_at": now,
                    "reclassified_at": now,
                    "symbol": candidate.get("symbol") or existing.get("symbol"),
                    "price_usd": candidate.get("price_usd")
                    or candidate.get("latest_price")
                    or existing.get("price_usd"),
                    "liquidity_usd": candidate.get("liquidity_usd")
                    or candidate.get("latest_liquidity")
                    or existing.get("liquidity_usd"),
                    "volume_24h": candidate.get("volume_24h")
                    or candidate.get("latest_volume_24h")
                    or existing.get("volume_24h"),
                    "whale_score": candidate.get("whale_score")
                    or candidate.get("latest_whale_score")
                    or existing.get("whale_score"),
                }
            else:
                rec = {
                    "registry_key": key,
                    "coin_identity": candidate.get("coin_id") or candidate.get("id"),
                    "pair_identity": candidate.get("pair_address"),
                    "symbol": candidate.get("symbol"),
                    "name": candidate.get("name"),
                    "chain": candidate.get("chain"),
                    "pair_address": candidate.get("pair_address"),
                    "price_usd": candidate.get("price_usd") or candidate.get("latest_price"),
                    "liquidity_usd": candidate.get("liquidity_usd")
                    or candidate.get("latest_liquidity"),
                    "volume_24h": candidate.get("volume_24h") or candidate.get("latest_volume_24h"),
                    "whale_score": candidate.get("whale_score")
                    or candidate.get("latest_whale_score"),
                    "classification_timestamp": now,
                    "first_seen_at": now,
                    "last_seen_at": now,
                    "seen_count": 1,
                    **classified,
                }
            if candidate.get("pinned") or (existing and existing.get("pinned")):
                rec["pinned"] = True
            self._records[key] = rec
            self._evict_if_needed(force_save=False)
            self._maybe_save(force=True)
            return dict(rec)

    def observe_many(self, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [self.observe_candidate(c) for c in candidates]

    def snapshot(self) -> dict[str, Any]:
        with _LOCK:
            rows = list(self._records.values())
            obs_count = self._observation_count
            last_cls = self._last_classification_update
            last_obs = self._last_observation
            max_entries = self.max_entries
            eviction_count = self._eviction_count

        families: dict[str, int] = {}
        for r in rows:
            fam = str(r.get("semantic_signal_family") or "UNKNOWN_INSUFFICIENT_EVIDENCE")
            # Normalize legacy unknown key for counters
            if fam == "UNKNOWN_UNRESOLVED":
                fam = "UNKNOWN_INSUFFICIENT_EVIDENCE"
            families[fam] = families.get(fam, 0) + 1

        unique_pairs = len(
            {str(r.get("pair_address") or r.get("pair_identity") or "").lower() for r in rows if r.get("pair_address") or r.get("pair_identity")}
        )
        unique_coins = len(
            {
                str(r.get("parsed_base_symbol") or _base_symbol(r.get("symbol")) or r.get("coin_identity") or r.get("registry_key"))
                for r in rows
            }
        )
        classified = sum(
            1
            for r in rows
            if str(r.get("semantic_signal_family"))
            not in ("UNKNOWN_INSUFFICIENT_EVIDENCE", "UNKNOWN_UNRESOLVED", "")
        )
        unknown = int(families.get("UNKNOWN_INSUFFICIENT_EVIDENCE", 0)) + int(
            families.get("UNKNOWN_UNRESOLVED", 0)
        )
        social = int(families.get("SOCIAL_CONFIRMED", 0))
        opp_conf = int(families.get("NON_SOCIAL_OPPORTUNISTIC_CONFIRMED", 0)) + int(
            families.get("OPPORTUNISTIC_CONFIRMED", 0)
        )
        opp_sus = int(families.get("OPPORTUNISTIC_SUSPECTED", 0))
        infra = int(families.get("NON_SOCIAL_INFRASTRUCTURE_CONFIRMED", 0)) + int(
            families.get("INFRASTRUCTURE_CONFIRMED", 0)
        )

        warning = None
        if rows and classified == 0:
            warning = (
                "Runtime semantic rules are not classifying current assets. "
                "Check taxonomy rules/provider."
            )

        return {
            "semantic_source_label": "Semantic Source: Runtime Registry",
            "semantic_source": "Runtime Registry",
            "taxonomy_version": TAXONOMY_VERSION,
            "runtime_rows_observed": obs_count,
            "runtime_candidates_seen": obs_count,
            "unique_runtime_pairs": unique_pairs,
            "unique_runtime_coins": unique_coins,
            "runtime_unique_identities": len(rows),
            "runtime_classified_coins": classified,
            "runtime_candidates_classified": classified,
            "runtime_unknown": unknown,
            "runtime_candidates_unresolved": unknown,
            "runtime_opportunistic_confirmed": opp_conf,
            "runtime_social_confirmed": social,
            "runtime_suspected_opportunistic": opp_sus,
            "runtime_infrastructure_confirmed": infra,
            "static_research_snapshot_coins": 14,  # AE12 research reference only
            "last_classification_update": last_cls,
            "last_registry_observation": last_obs,
            "max_entries": max_entries,
            "eviction_count": eviction_count,
            "family_counts": families,
            "coin_social_confirmed_count": social,
            "coin_opportunistic_confirmed_count": opp_conf,
            "coin_opportunistic_suspected_count": opp_sus,
            "coin_unknown_unresolved_count": unknown,
            "counters": {
                "runtime_rows_observed": obs_count,
                "unique_runtime_pairs": unique_pairs,
                "unique_runtime_coins": unique_coins,
                "runtime_classified_coins": classified,
                "runtime_opportunistic_confirmed": opp_conf,
                "runtime_social_confirmed": social,
                "runtime_suspected_opportunistic": opp_sus,
                "runtime_infrastructure_confirmed": infra,
                "runtime_unknown": unknown,
                "static_research_snapshot_coins": 14,
                "last_classification_update": last_cls,
                "last_registry_observation": last_obs,
                "max_entries": max_entries,
                "eviction_count": eviction_count,
            },
            "wording": {
                "classified": "Classified: labels assigned",
                "unknown": "Unknown: not enough evidence",
                "registered": "Registered: seen and tracked",
                "static_snapshot": "Static Snapshot: historical research reference",
            },
            "classification_warning": warning,
            "social_confirmed_explanation": (
                "SOCIAL_CONFIRMED requires explicit social/charitable/public-good evidence. "
                "Meme hype / internet coordination / pump communities are opportunistic, not social."
                if social == 0
                else None
            ),
            "social_sources_available": False,
            "gemini_status": "Gemini not needed for local classification - audit/reporting only",
            "qwen_status": "Explanation/memo only - no trade authority",
            "rss_status": "Headline sentiment matrix only - not linked to SOCIAL_CONFIRMED",
            "ollama_status": "Ollama not needed for local classification",
            "static_ae12_note": (
                "Static AE12 snapshot (e.g. 14 coins) is research context only and is not "
                "the live semantic universe."
            ),
            "records": sorted(rows, key=lambda r: str(r.get("last_seen_at") or ""), reverse=True),
            "updated_at_utc": _utc_now(),
            "persistence_path": str(self.path),
        }

    def records(self) -> list[dict[str, Any]]:
        return list(self.snapshot().get("records") or [])


def get_semantic_registry() -> SemanticRegistry:
    global _INSTANCE
    with _LOCK:
        if _INSTANCE is None:
            _INSTANCE = SemanticRegistry()
        return _INSTANCE


def reset_semantic_registry_for_tests() -> None:
    global _INSTANCE
    with _LOCK:
        _INSTANCE = None
