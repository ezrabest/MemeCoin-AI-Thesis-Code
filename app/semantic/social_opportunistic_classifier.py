"""Evidence-grounded social / opportunistic classifier (audit-only, no trade authority)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.analytics.features import ClusterLabel, CLUSTER_REGISTRY_PATH
from app.semantic.curated_hypotheses import count_curated_hypotheses
from app.semantic.evidence_collector import collect_evidence_bundle, has_sufficient_evidence_for_llm
from app.semantic.llm_semantic_client import PROMPT_VERSION, call_semantic_llm, resolve_semantic_llm_provider
from app.semantic.semantic_registry import (
    VERDICTS_PATH,
    count_semantic_verdicts,
    persist_semantic_verdict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = Path(__import__("os").getenv("TRADER_DB_PATH", str(DATA_DIR / "trader.db")))

# Re-export for callers / tests
__all__ = [
    "PROMPT_VERSION",
    "classify_token_social_opportunistic",
    "get_authoritative_semantic_counts",
    "legacy_cluster_label_counts",
    "rule_based_fallback",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _identity_key(
    *,
    chain: str,
    pair_address: str,
    token_address: str,
    symbol: str,
) -> str:
    if pair_address:
        return f"{(chain or 'unknown').lower()}:pair:{pair_address.strip().lower()}"
    if token_address:
        return f"{(chain or 'unknown').lower()}:token:{token_address.strip().lower()}"
    return f"{(chain or 'unknown').lower()}:symbol:{(symbol or 'unknown').strip().upper()}"


def _empty_verdict_base(identity: dict[str, Any], seed_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "identity_key": identity["identity_key"],
        "chain": identity.get("chain") or "",
        "pair_address": identity.get("pair_address") or "",
        "token_address": identity.get("token_address") or "",
        "symbol": identity.get("symbol") or "",
        "name": identity.get("name") or "",
        "provider_url": identity.get("provider_url") or "",
        "user_seed_collection": seed_meta.get("user_seed_collection") or "",
        "user_seed_label": seed_meta.get("user_seed_label") or "",
        "user_hypothesis": seed_meta.get("user_hypothesis") or "",
        "semantic_status": "INSUFFICIENT_EVIDENCE",
        "cluster_label": "UNKNOWN",
        "confidence": 0.0,
        "evidence_quality": "NONE",
        "evidence_items": [],
        "counter_evidence": [],
        "reasoning": "",
        "provider": "NONE",
        "model": "",
        "prompt_version": PROMPT_VERSION,
        "classified_at_utc": _utc_now(),
        "no_trade_authority": True,
    }


def rule_based_fallback(
    *,
    evidence_items: list[dict[str, Any]],
    counter_evidence: list[dict[str, Any]],
    evidence_quality: str,
    user_seed_label: str = "",
) -> dict[str, Any]:
    """
    Conservative fallback.

    Must NOT invent SOCIAL_CONFIRMED / OPPORTUNISTIC_CONFIRMED unless strong
    explicit local evidence exists. User seeds alone never confirm.
    """
    non_seed = [i for i in evidence_items if str(i.get("source_type")) != "USER_SEED"]
    # Strong explicit: HIGH relevance + supporting SOCIAL/OPPORTUNISTIC from non-seed sources
    strong_social = [
        i
        for i in non_seed
        if i.get("supports") == "SOCIAL"
        and i.get("relevance") == "HIGH"
        and str(i.get("source_type"))
        in ("RAW_PROVIDER_PAYLOAD", "RSS", "PUBLIC_WEB", "OFFICIAL_WEBSITE", "LOCAL_CACHE")
        and len(str(i.get("snippet") or "")) > 80
    ]
    strong_opp = [
        i
        for i in non_seed
        if i.get("supports") == "OPPORTUNISTIC"
        and i.get("relevance") == "HIGH"
        and str(i.get("source_type"))
        in ("RAW_PROVIDER_PAYLOAD", "RSS", "PUBLIC_WEB", "OFFICIAL_WEBSITE", "LOCAL_CACHE")
        and len(str(i.get("snippet") or "")) > 80
    ]

    # Explicit: do not confirm from user seed
    _ = user_seed_label

    if strong_social and not strong_opp and not counter_evidence:
        return {
            "semantic_status": "SOCIAL_CONFIRMED",
            "cluster_label": ClusterLabel.SOCIALLY_MOTIVATED.value,
            "confidence": 0.55,
            "evidence_quality": evidence_quality if evidence_quality != "NONE" else "MEDIUM",
            "reasoning": "Rule-based: strong explicit local social/public-good evidence present.",
            "provider": "RULE_BASED_FALLBACK",
        }
    if strong_opp and not strong_social:
        return {
            "semantic_status": "OPPORTUNISTIC_CONFIRMED",
            "cluster_label": ClusterLabel.OPPORTUNISTIC_SPECULATIVE.value,
            "confidence": 0.55,
            "evidence_quality": evidence_quality if evidence_quality != "NONE" else "MEDIUM",
            "reasoning": "Rule-based: strong explicit local opportunistic/meme evidence present.",
            "provider": "RULE_BASED_FALLBACK",
        }

    # Default safe outcomes only
    if not non_seed or evidence_quality in ("NONE", "LOW"):
        return {
            "semantic_status": "INSUFFICIENT_EVIDENCE",
            "cluster_label": "UNKNOWN",
            "confidence": 0.15,
            "evidence_quality": evidence_quality or "NONE",
            "reasoning": (
                "Rule-based fallback: insufficient evidence; "
                "user seed (if any) remains provenance only."
            ),
            "provider": "RULE_BASED_FALLBACK",
        }
    return {
        "semantic_status": "INSUFFICIENT_EVIDENCE",
        "cluster_label": "UNKNOWN",
        "confidence": 0.25,
        "evidence_quality": evidence_quality or "LOW",
        "reasoning": (
            "Rule-based fallback: evidence present but not strong enough for confirmation; "
            "cluster_label left UNKNOWN (low confidence)."
        ),
        "provider": "RULE_BASED_FALLBACK",
    }


def classify_token_social_opportunistic(
    *,
    chain: str = "",
    pair_address: str = "",
    token_address: str = "",
    symbol: str = "",
    name: str = "",
    provider_url: str = "",
    persist: bool = False,
    force_llm: bool = False,
    allow_web_search: bool = False,
    llm_client: Any | None = None,
) -> dict[str, Any]:
    """
    Classify one token. Audit-only. Never opens trades or overrides risk gates.

    llm_client: optional injectable callable for tests (returns call_semantic_llm-shaped dict).
    """
    identity_key = _identity_key(
        chain=chain,
        pair_address=pair_address,
        token_address=token_address,
        symbol=symbol,
    )
    identity = {
        "identity_key": identity_key,
        "chain": chain,
        "pair_address": pair_address,
        "token_address": token_address,
        "symbol": symbol,
        "name": name,
        "provider_url": provider_url,
    }

    bundle = collect_evidence_bundle(
        chain=chain,
        pair_address=pair_address,
        token_address=token_address,
        symbol=symbol,
        name=name,
        provider_url=provider_url,
        allow_web_search=allow_web_search,
    )
    seed_meta = {
        "user_seed_collection": bundle.get("user_seed_collection") or "",
        "user_seed_label": bundle.get("user_seed_label") or "",
        "user_hypothesis": bundle.get("user_hypothesis") or "",
    }
    verdict = _empty_verdict_base(identity, seed_meta)
    verdict["evidence_items"] = list(bundle.get("evidence_items") or [])
    verdict["counter_evidence"] = list(bundle.get("counter_evidence") or [])
    verdict["evidence_quality"] = str(bundle.get("evidence_quality") or "NONE")

    sufficient = has_sufficient_evidence_for_llm(bundle)
    provider_key = resolve_semantic_llm_provider()

    # Missing / weak evidence → INSUFFICIENT_EVIDENCE (do not force classification)
    if not sufficient and not force_llm:
        fb = rule_based_fallback(
            evidence_items=verdict["evidence_items"],
            counter_evidence=verdict["counter_evidence"],
            evidence_quality=verdict["evidence_quality"],
            user_seed_label=seed_meta["user_seed_label"],
        )
        # Even with rule-based strong path, respect "do not force when weak"
        if fb["semantic_status"] in ("SOCIAL_CONFIRMED", "OPPORTUNISTIC_CONFIRMED"):
            verdict.update(fb)
        else:
            verdict.update(
                {
                    "semantic_status": "INSUFFICIENT_EVIDENCE",
                    "cluster_label": "UNKNOWN",
                    "confidence": float(fb.get("confidence") or 0.15),
                    "reasoning": fb.get("reasoning")
                    or "Insufficient evidence for confirmation.",
                    "provider": "RULE_BASED_FALLBACK",
                    "model": "rule_based",
                }
            )
        if persist:
            persist_semantic_verdict(verdict)
        return verdict

    # LLM path
    if provider_key == "none":
        fb = rule_based_fallback(
            evidence_items=verdict["evidence_items"],
            counter_evidence=verdict["counter_evidence"],
            evidence_quality=verdict["evidence_quality"],
            user_seed_label=seed_meta["user_seed_label"],
        )
        # When LLM disabled: never invent confirmed unless strong evidence path fired
        verdict.update(fb)
        verdict["model"] = "rule_based"
        if persist:
            persist_semantic_verdict(verdict)
        return verdict

    caller = llm_client or call_semantic_llm
    try:
        llm_result = caller(
            identity=identity,
            user_hypothesis=seed_meta["user_hypothesis"],
            evidence_items=verdict["evidence_items"],
            counter_evidence=verdict["counter_evidence"],
        )
    except Exception as exc:  # noqa: BLE001
        verdict.update(
            {
                "semantic_status": "CLASSIFICATION_FAILED",
                "cluster_label": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": f"LLM call raised: {type(exc).__name__}: {exc}",
                "provider": "NONE",
                "model": "",
            }
        )
        if persist:
            persist_semantic_verdict(verdict)
        return verdict

    if not llm_result or not llm_result.get("ok"):
        # LLM unavailable / parse failure
        err = str((llm_result or {}).get("error") or "llm_unavailable")
        if err in ("llm_disabled",):
            status = "INSUFFICIENT_EVIDENCE"
        else:
            status = "CLASSIFICATION_FAILED"
        verdict.update(
            {
                "semantic_status": status,
                "cluster_label": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": f"LLM unavailable or failed ({err}); no invented fallback label.",
                "provider": str((llm_result or {}).get("provider") or "NONE"),
                "model": str((llm_result or {}).get("model") or ""),
            }
        )
        if persist:
            persist_semantic_verdict(verdict)
        return verdict

    parsed = llm_result.get("parsed") or {}
    try:
        # Schema validation already done in client; re-check critical fields
        status = str(parsed.get("semantic_status") or "")
        if status not in (
            "SOCIAL_CONFIRMED",
            "OPPORTUNISTIC_CONFIRMED",
            "INSUFFICIENT_EVIDENCE",
        ):
            raise ValueError("schema_validation_failed")
        verdict.update(
            {
                "semantic_status": status,
                "cluster_label": str(parsed.get("cluster_label") or "UNKNOWN"),
                "confidence": float(parsed.get("confidence") or 0.0),
                "evidence_quality": str(
                    parsed.get("evidence_quality") or verdict["evidence_quality"]
                ),
                "reasoning": str(parsed.get("reasoning") or ""),
                "provider": str(llm_result.get("provider") or "NONE"),
                "model": str(llm_result.get("model") or ""),
                "supporting_evidence_ids": list(parsed.get("supporting_evidence_ids") or []),
                "counter_evidence_ids": list(parsed.get("counter_evidence_ids") or []),
                "risk_notes": list(parsed.get("risk_notes") or []),
            }
        )
    except Exception as exc:  # noqa: BLE001
        verdict.update(
            {
                "semantic_status": "CLASSIFICATION_FAILED",
                "cluster_label": "UNKNOWN",
                "confidence": 0.0,
                "reasoning": f"Parser/schema validation failed: {exc}",
                "provider": str(llm_result.get("provider") or "NONE"),
                "model": str(llm_result.get("model") or ""),
            }
        )

    verdict["no_trade_authority"] = True
    if persist:
        persist_semantic_verdict(verdict)
    return verdict


def _count_labels(counter: dict[str, int], label: str) -> int:
    return int(counter.get(label) or 0)


def _safe_db_cluster_counts(db_path: Path | None = None) -> dict[str, Any]:
    """Count valid ClusterLabel values from tables that actually have cluster_label."""
    path = db_path or DB_PATH
    out: dict[str, Any] = {
        "tables_scanned": [],
        "SOCIALLY_MOTIVATED": 0,
        "OPPORTUNISTIC_SPECULATIVE": 0,
        "invalid_label_rows": 0,
        "by_table": {},
    }
    if not path.is_file():
        out["error"] = "db_missing"
        return out
    try:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        out["error"] = str(exc)
        return out
    try:
        tables = [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY 1"
            ).fetchall()
        ]
        valid = {ClusterLabel.SOCIALLY_MOTIVATED.value, ClusterLabel.OPPORTUNISTIC_SPECULATIVE.value}
        for table in tables:
            cols = {c[1] for c in con.execute(f"PRAGMA table_info({table})").fetchall()}
            if "cluster_label" not in cols:
                continue
            out["tables_scanned"].append(table)
            try:
                rows = con.execute(
                    f"SELECT cluster_label, COUNT(*) AS cnt FROM {table} GROUP BY cluster_label"
                ).fetchall()
            except sqlite3.Error:
                continue
            table_counts: dict[str, int] = {}
            for row in rows:
                raw = str(row["cluster_label"] or "")
                cnt = int(row["cnt"] or 0)
                if raw in valid:
                    out[raw] = int(out.get(raw) or 0) + cnt
                    table_counts[raw] = cnt
                elif raw:
                    out["invalid_label_rows"] = int(out["invalid_label_rows"]) + cnt
                    table_counts[f"INVALID:{raw}"] = cnt
            out["by_table"][table] = table_counts
    finally:
        con.close()
    return out


def legacy_cluster_label_counts(
    *,
    project_root: Path | None = None,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Read legacy ClusterLabel-compatible counts from registry + DB (no whale-log truncation)."""
    root = project_root or PROJECT_ROOT
    registry_path = root / "data" / "cluster_registry.json"
    if not registry_path.exists():
        registry_path = CLUSTER_REGISTRY_PATH

    registry_counts = {
        ClusterLabel.SOCIALLY_MOTIVATED.value: 0,
        ClusterLabel.OPPORTUNISTIC_SPECULATIVE.value: 0,
        "OTHER": 0,
        "entry_count": 0,
    }
    if registry_path.is_file():
        try:
            import json

            data = json.loads(registry_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                registry_counts["entry_count"] = len(data)
                for v in data.values():
                    if isinstance(v, str):
                        lab = v
                    elif isinstance(v, dict):
                        lab = str(v.get("cluster_label") or "")
                    else:
                        lab = ""
                    if lab == ClusterLabel.SOCIALLY_MOTIVATED.value:
                        registry_counts[ClusterLabel.SOCIALLY_MOTIVATED.value] += 1
                    elif lab == ClusterLabel.OPPORTUNISTIC_SPECULATIVE.value:
                        registry_counts[ClusterLabel.OPPORTUNISTIC_SPECULATIVE.value] += 1
                    else:
                        registry_counts["OTHER"] += 1
        except (OSError, json.JSONDecodeError, ValueError):
            pass

    db_counts = _safe_db_cluster_counts(db_path=db_path or (root / "data" / "trader.db"))

    # Prefer paper_trades exact valid counts for legacy_* when available
    paper = (db_counts.get("by_table") or {}).get("paper_trades") or {}
    legacy_social = int(
        paper.get(ClusterLabel.SOCIALLY_MOTIVATED.value)
        or db_counts.get(ClusterLabel.SOCIALLY_MOTIVATED.value)
        or 0
    )
    legacy_opp = int(
        paper.get(ClusterLabel.OPPORTUNISTIC_SPECULATIVE.value)
        or db_counts.get(ClusterLabel.OPPORTUNISTIC_SPECULATIVE.value)
        or 0
    )

    return {
        "registry": registry_counts,
        "db": db_counts,
        "legacy_socially_motivated_count": legacy_social,
        "legacy_opportunistic_speculative_count": legacy_opp,
        "coins_table_has_cluster_label": "coins" in (db_counts.get("tables_scanned") or []),
        "note": (
            "Legacy counts come from paper_trades.cluster_label and cluster_registry.json. "
            "coins table has no cluster_label column in current schema."
        ),
    }


def get_authoritative_semantic_counts(
    *,
    project_root: Path | None = None,
    verdicts_path: Path | None = None,
    db_path: Path | None = None,
    environ: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Authoritative counter for UI/API with strict epistemic separation.

    Inspects:
      1) semantic_verdicts.jsonl (system-verified)
      2) cluster_registry.json (legacy registry)
      3) DB tables that actually contain cluster_label (legacy DB)
      4) curated target CSV hypotheses (user/curated only)
      5) does NOT assume a candidates table
    """
    root = project_root or PROJECT_ROOT
    vpath = verdicts_path or (root / "data" / "semantic_verdicts.jsonl")
    if not vpath.exists() and VERDICTS_PATH.exists():
        vpath = VERDICTS_PATH

    verdict_counts = count_semantic_verdicts(path=vpath)
    legacy = legacy_cluster_label_counts(project_root=root, db_path=db_path)
    curated = count_curated_hypotheses(project_root=root, environ=environ)

    social_confirmed = int(verdict_counts["social_confirmed_count"])
    opp_confirmed = int(verdict_counts["opportunistic_confirmed_count"])
    insuff = int(verdict_counts["insufficient_evidence_count"])
    failed = int(verdict_counts["classification_failed_count"])
    total_verdicts = int(verdict_counts["total_semantic_verdicts"])

    legacy_db_social = int(legacy["legacy_socially_motivated_count"])
    legacy_db_opp = int(legacy["legacy_opportunistic_speculative_count"])
    registry_social = int(legacy["registry"].get(ClusterLabel.SOCIALLY_MOTIVATED.value, 0))
    registry_opp = int(
        legacy["registry"].get(ClusterLabel.OPPORTUNISTIC_SPECULATIVE.value, 0)
    )
    registry_total = int(legacy["registry"].get("entry_count", 0))

    return {
        # Backward-compatible system-verified fields
        "social_confirmed_count": social_confirmed,
        "opportunistic_confirmed_count": opp_confirmed,
        "insufficient_evidence_count": insuff,
        "classification_failed_count": failed,
        "total_semantic_verdicts": total_verdicts,
        # Explicit epistemic aliases — System Verified
        "system_verified_social_count": social_confirmed,
        "system_verified_opportunistic_count": opp_confirmed,
        "system_verified_insufficient_evidence_count": insuff,
        "system_verified_classification_failed_count": failed,
        "system_verified_total_count": total_verdicts,
        # Backward-compatible legacy DB fields
        "legacy_socially_motivated_count": legacy_db_social,
        "legacy_opportunistic_speculative_count": legacy_db_opp,
        # Explicit epistemic aliases — Legacy DB
        "legacy_db_social_count": legacy_db_social,
        "legacy_db_opportunistic_count": legacy_db_opp,
        # Legacy registry
        "legacy_registry_socially_motivated_count": registry_social,
        "legacy_registry_opportunistic_speculative_count": registry_opp,
        "legacy_registry_entry_count": registry_total,
        "legacy_registry_social_count": registry_social,
        "legacy_registry_opportunistic_count": registry_opp,
        "legacy_registry_total_count": registry_total,
        # Curated / user hypotheses (never leak into system verified)
        **curated,
        "db_tables_with_cluster_label": legacy["db"].get("tables_scanned") or [],
        "db_cluster_by_table": legacy["db"].get("by_table") or {},
        "coins_table_has_cluster_label": legacy.get("coins_table_has_cluster_label", False),
        "verdicts_path": str(vpath),
        "sources": {
            "semantic_verdicts": str(vpath),
            "cluster_registry": str(root / "data" / "cluster_registry.json"),
            "trader_db": str(db_path or root / "data" / "trader.db"),
            "curated_targets": curated.get("curated_targets_resolved_path"),
        },
        "epistemic_levels": [
            "system_verified",
            "legacy_db",
            "legacy_registry",
            "curated_user_hypotheses",
        ],
        "no_aggregated_social_total": True,
        "no_trade_authority": True,
        "ui_display": {
            "SYSTEM_VERIFIED": "system_verified_* from semantic_verdicts.jsonl only",
            "LEGACY_DB": "legacy_db_* from trader.db cluster_label columns (paper_trades)",
            "LEGACY_REGISTRY": "legacy_registry_* from cluster_registry.json",
            "CURATED_HYPOTHESES": "curated_*_hypothesis_count — provenance only",
            "DO_NOT_COLLAPSE": "Do not recombine epistemic levels into one Social number",
            "SOCIAL_BADGE_NOT_CONFIRMED": "Visual SOCIAL? badges are not counted as confirmed social",
        },
        "built_at_utc": _utc_now(),
    }


def cluster_label_for_compatibility(verdict: dict[str, Any]) -> str | None:
    """Map confirmed verdicts to existing ClusterLabel values; UNKNOWN stays None for sticky registry."""
    label = str(verdict.get("cluster_label") or "").upper()
    if label == ClusterLabel.SOCIALLY_MOTIVATED.value:
        return ClusterLabel.SOCIALLY_MOTIVATED.value
    if label == ClusterLabel.OPPORTUNISTIC_SPECULATIVE.value:
        return ClusterLabel.OPPORTUNISTIC_SPECULATIVE.value
    return None
