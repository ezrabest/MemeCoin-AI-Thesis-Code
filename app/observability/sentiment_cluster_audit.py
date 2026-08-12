"""Sentiment and cluster assignment audit report."""
from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .. import database as db
from ..analytics.features import list_cluster_registry
from .audit_io import utc_timestamp_slug, write_json_report_atomic
from .effective_settings import get_effective_settings

log = logging.getLogger("sentiment_cluster_audit")

DEFAULT_CLUSTER = "OPPORTUNISTIC_SPECULATIVE"


def run_sentiment_cluster_audit() -> dict[str, Any]:
    eff = get_effective_settings()
    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "settings_hash": eff.settings_hash,
        "phase": 1,
    }

    with db.get_db() as conn:
        sentiment_count = conn.execute("SELECT COUNT(*) AS c FROM sentiment_records").fetchone()["c"]
        recent_sentiment = conn.execute(
            "SELECT COUNT(*) AS c FROM sentiment_records WHERE timestamp > datetime('now', '-7 days')"
        ).fetchone()["c"]
        signal_rows = conn.execute(
            "SELECT features_json FROM signals ORDER BY id DESC LIMIT 500"
        ).fetchall()

    registry = list_cluster_registry()
    cluster_labels = [e.get("cluster_label", DEFAULT_CLUSTER) for e in registry.values()]
    label_dist = dict(Counter(cluster_labels))

    missing_sentiment = 0
    defaulted_cluster = 0
    non_default_examples: list[dict[str, Any]] = []

    for row in signal_rows:
        fj = row["features_json"]
        if not fj:
            missing_sentiment += 1
            continue
        try:
            data = json.loads(fj) if isinstance(fj, str) else fj
        except (json.JSONDecodeError, TypeError):
            missing_sentiment += 1
            continue
        sent = data.get("sentiment_score")
        if sent is None:
            missing_sentiment += 1
        cluster = data.get("cluster_label", DEFAULT_CLUSTER)
        if cluster == DEFAULT_CLUSTER:
            defaulted_cluster += 1
        elif len(non_default_examples) < 5:
            non_default_examples.append({"cluster_label": cluster, "sentiment_score": sent})

    total_signals = len(signal_rows) or 1
    opportunistic_count = label_dist.get(DEFAULT_CLUSTER, 0)
    total_registry = len(registry) or 1

    report["sentiment_records"] = {
        "total_count": sentiment_count,
        "recent_7d_growth": recent_sentiment,
    }
    report["cluster_registry"] = {
        "total_tokens": len(registry),
        "label_distribution": label_dist,
        "opportunistic_speculative_count": opportunistic_count,
        "opportunistic_is_fallback": opportunistic_count > total_registry * 0.8,
        "non_default_examples": non_default_examples,
    }
    report["signal_feature_coverage"] = {
        "signals_sampled": len(signal_rows),
        "pct_missing_sentiment": round(100 * missing_sentiment / total_signals, 2),
        "pct_default_cluster": round(100 * defaulted_cluster / total_signals, 2),
        "missing_sentiment_explicit": missing_sentiment > 0,
        "default_cluster_explicit": defaulted_cluster > 0,
        "default_cluster_low_confidence": True,
    }
    report["social_news_used"] = {
        "sentiment_in_signals": missing_sentiment < total_signals,
        "cluster_in_signals": defaulted_cluster < total_signals,
        "cluster_classifier_path": "app.analytics.features.resolve_cluster_label",
    }

    ts = utc_timestamp_slug()
    path = write_json_report_atomic(f"sentiment_cluster_audit_{ts}.json", report)
    report["output_path"] = str(path)
    log.info("Sentiment/cluster audit written: %s", path)
    return report
