from __future__ import annotations

import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("THESIS_ROOT", r"E:\Projects\Final Project\memecoin_trader"))
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = ROOT / "data" / "audits" / f"thesis_context_incremental_predictive_value_audit_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

DATASET_CSV = os.environ.get("THESIS_CONTEXT_DATASET_CSV", "").strip()
BOOTSTRAP_N = int(os.environ.get("THESIS_BOOTSTRAP_N", "3000"))
RANDOM_SEED = int(os.environ.get("THESIS_RANDOM_SEED", "42"))

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def find_latest_dataset() -> Path:
    if DATASET_CSV:
        p = Path(DATASET_CSV)
        if not p.exists():
            raise FileNotFoundError(f"THESIS_CONTEXT_DATASET_CSV not found: {p}")
        return p

    candidates = list(
        (ROOT / "data" / "audits").glob(
            "thesis_context_event_level_dataset_build_audit_*/03_event_level_context_rebuild_dataset.csv"
        )
    )
    if not candidates:
        raise FileNotFoundError("No event-level context rebuild dataset CSV found.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def import_sklearn():
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            average_precision_score,
            brier_score_loss,
            log_loss,
            roc_auc_score,
        )
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        return {
            "ColumnTransformer": ColumnTransformer,
            "SimpleImputer": SimpleImputer,
            "LogisticRegression": LogisticRegression,
            "Pipeline": Pipeline,
            "OneHotEncoder": OneHotEncoder,
            "StandardScaler": StandardScaler,
            "average_precision_score": average_precision_score,
            "brier_score_loss": brier_score_loss,
            "log_loss": log_loss,
            "roc_auc_score": roc_auc_score,
            "ohe_instance": ohe,
        }
    except Exception as exc:
        raise SystemExit(f"scikit-learn import failed: {exc!r}")


SK = import_sklearn()


LEAKAGE_PATTERNS = [
    "label",
    "future_",
    "tp_hit",
    "sl_hit",
    "first_tp",
    "first_sl",
    "return_end",
    "return_max",
    "return_min",
    "pnl",
    "outcome",
]

IDENTIFIER_OR_TIME_COLS = {
    "candidate_event_id",
    "canonical_coin_id",
    "event_bucket",
    "candidate_event_start_utc",
    "candidate_event_time_utc",
    "candidate_event_end_utc",
    "chronological_split",
}


def is_leakage_col(col: str) -> bool:
    c = col.lower()
    if col in IDENTIFIER_OR_TIME_COLS:
        return True
    return any(p in c for p in LEAKAGE_PATTERNS)


def pick_feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    numeric_cols = []
    categorical_cols = []

    for c in df.columns:
        if is_leakage_col(c):
            continue

        if c.startswith("raw_payload_") or c.startswith("pool_flow_") or c.startswith("gemini_"):
            continue

        if c.startswith("asof_") or c.startswith("score_") or c.startswith("signal_score_") or c.startswith("confidence_") or c.startswith("signal_confidence_") or c == "signal_rows":
            if pd.api.types.is_numeric_dtype(pd.to_numeric(df[c], errors="coerce")):
                numeric_cols.append(c)
            continue

        if c.endswith("_top_values"):
            nunique = df[c].astype(str).nunique(dropna=True)
            if 1 < nunique <= 100:
                categorical_cols.append(c)

    context_numeric = []
    for c in df.columns:
        if is_leakage_col(c):
            continue
        if c.startswith("raw_payload_") or c.startswith("pool_flow_") or c.startswith("gemini_"):
            val = pd.to_numeric(df[c], errors="coerce")
            if val.notna().sum() > 0:
                context_numeric.append(c)

    return {
        "internal_numeric": sorted(set(numeric_cols)),
        "internal_categorical": sorted(set(categorical_cols)),
        "context_numeric": sorted(set(context_numeric)),
    }


def make_model(numeric_features: list[str], categorical_features: list[str]):
    ColumnTransformer = SK["ColumnTransformer"]
    SimpleImputer = SK["SimpleImputer"]
    LogisticRegression = SK["LogisticRegression"]
    Pipeline = SK["Pipeline"]
    StandardScaler = SK["StandardScaler"]

    transformers = []

    if numeric_features:
        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", num_pipe, numeric_features))

    if categorical_features:
        try:
            ohe = SK["OneHotEncoder"](handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = SK["OneHotEncoder"](handle_unknown="ignore", sparse=False)

        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", ohe),
        ])
        transformers.append(("cat", cat_pipe, categorical_features))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )

    return Pipeline([
        ("preprocess", pre),
        ("model", clf),
    ])


def metric_safe(y_true: np.ndarray, y_score: np.ndarray, metric: str) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None

    if metric == "roc_auc":
        return float(SK["roc_auc_score"](y_true, y_score))

    if metric == "average_precision":
        return float(SK["average_precision_score"](y_true, y_score))

    if metric == "brier":
        return float(SK["brier_score_loss"](y_true, y_score))

    if metric == "log_loss":
        eps = 1e-12
        y_score_clip = np.clip(y_score, eps, 1 - eps)
        return float(SK["log_loss"](y_true, y_score_clip, labels=[0, 1]))

    raise ValueError(metric)


def topk_metrics(y_true: np.ndarray, y_score: np.ndarray, split: str, model_name: str) -> list[dict[str, Any]]:
    rows = []
    n = len(y_true)
    total_pos = int(y_true.sum())
    base_rate = total_pos / n if n else 0.0

    for frac in [0.01, 0.02, 0.05, 0.10]:
        k = max(1, int(round(n * frac)))
        order = np.argsort(-y_score)
        idx = order[:k]
        pos = int(y_true[idx].sum())
        rate = pos / k if k else None
        lift = (rate / base_rate) if base_rate > 0 and rate is not None else None

        rows.append({
            "split": split,
            "model": model_name,
            "top_fraction": frac,
            "k": k,
            "winners_in_top_k": pos,
            "total_winners_in_split": total_pos,
            "winner_rate_top_k": rate,
            "base_winner_rate": base_rate,
            "lift_vs_base_rate": lift,
            "winner_capture_rate": pos / total_pos if total_pos else None,
        })

    return rows


def bootstrap_delta(y: np.ndarray, s_internal: np.ndarray, s_context: np.ndarray, metric: str) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y)

    observed_internal = metric_safe(y, s_internal, metric)
    observed_context = metric_safe(y, s_context, metric)

    if observed_internal is None or observed_context is None:
        return {
            "metric": metric,
            "observed_internal": observed_internal,
            "observed_context": observed_context,
            "observed_delta_context_minus_internal": None,
            "bootstrap_n_valid": 0,
            "ci95_low": None,
            "ci95_high": None,
            "p_two_sided_delta_zero": None,
            "conclusion": "NOT_COMPUTABLE_SINGLE_CLASS",
        }

    deltas = []
    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue

        a = metric_safe(yy, s_internal[idx], metric)
        b = metric_safe(yy, s_context[idx], metric)
        if a is None or b is None:
            continue
        deltas.append(b - a)

    if not deltas:
        return {
            "metric": metric,
            "observed_internal": observed_internal,
            "observed_context": observed_context,
            "observed_delta_context_minus_internal": observed_context - observed_internal,
            "bootstrap_n_valid": 0,
            "ci95_low": None,
            "ci95_high": None,
            "p_two_sided_delta_zero": None,
            "conclusion": "BOOTSTRAP_FAILED",
        }

    arr = np.array(deltas)
    lo, hi = np.quantile(arr, [0.025, 0.975])
    p_low = np.mean(arr <= 0)
    p_high = np.mean(arr >= 0)
    p_two = float(min(1.0, 2 * min(p_low, p_high)))

    obs_delta = observed_context - observed_internal

    if lo > 0:
        conclusion = "CONTEXT_SIGNIFICANTLY_IMPROVES_METRIC"
    elif hi < 0:
        conclusion = "CONTEXT_SIGNIFICANTLY_DEGRADES_METRIC"
    else:
        conclusion = "NO_SIGNIFICANT_CONTEXT_DELTA"

    return {
        "metric": metric,
        "observed_internal": observed_internal,
        "observed_context": observed_context,
        "observed_delta_context_minus_internal": obs_delta,
        "bootstrap_n_valid": int(len(arr)),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_two_sided_delta_zero": p_two,
        "conclusion": conclusion,
    }


def label_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, g in df.groupby("chronological_split"):
        rows.append({
            "split": split,
            "rows": len(g),
            "winner": int((g["target_winner"] == 1).sum()),
            "non_winner": int((g["target_winner"] == 0).sum()),
            "winner_rate": float((g["target_winner"] == 1).mean()) if len(g) else None,
            "flat": int((g["label_x2_sl_4h"] == "FLAT").sum()),
            "loser": int((g["label_x2_sl_4h"] == "LOSER").sum()),
        })
    return pd.DataFrame(rows)


def main() -> None:
    dataset_path = find_latest_dataset()
    df = pd.read_csv(dataset_path, low_memory=False)

    required = ["label_available", "label_x2_sl_4h", "chronological_split"]
    for c in required:
        if c not in df.columns:
            raise SystemExit(f"Required column missing from dataset: {c}")

    df = df[df["label_available"].astype(str).str.upper().isin(["TRUE", "1"])].copy()
    df = df[df["label_x2_sl_4h"].isin(["WINNER", "LOSER", "FLAT"])].copy()
    df["target_winner"] = (df["label_x2_sl_4h"] == "WINNER").astype(int)

    split_counts = df["chronological_split"].value_counts().to_dict()
    needed_splits = {"train", "validation", "test"}
    if not needed_splits.issubset(set(split_counts)):
        raise SystemExit(f"Missing chronological splits. Found: {split_counts}")

    feature_groups = pick_feature_groups(df)

    train = df[df["chronological_split"] == "train"].copy()
    val = df[df["chronological_split"] == "validation"].copy()
    test = df[df["chronological_split"] == "test"].copy()

    # Drop features that are entirely null in train.
    def keep_non_empty(cols: list[str]) -> list[str]:
        out = []
        for c in cols:
            if c not in train.columns:
                continue
            if pd.to_numeric(train[c], errors="coerce").notna().sum() > 0 or train[c].astype(str).replace("nan", "").str.len().gt(0).sum() > 0:
                out.append(c)
        return out

    internal_numeric = keep_non_empty(feature_groups["internal_numeric"])
    internal_categorical = feature_groups["internal_categorical"]
    context_numeric = keep_non_empty(feature_groups["context_numeric"])

    internal_features = internal_numeric + internal_categorical
    augmented_numeric = internal_numeric + context_numeric
    augmented_categorical = internal_categorical

    if not internal_features:
        raise SystemExit("No internal features selected.")
    if not context_numeric:
        raise SystemExit("No context features selected.")

    X_train_int = train[internal_features]
    X_val_int = val[internal_features]
    X_test_int = test[internal_features]

    X_train_aug = train[augmented_numeric + augmented_categorical]
    X_val_aug = val[augmented_numeric + augmented_categorical]
    X_test_aug = test[augmented_numeric + augmented_categorical]

    y_train = train["target_winner"].to_numpy()
    y_val = val["target_winner"].to_numpy()
    y_test = test["target_winner"].to_numpy()

    model_internal = make_model(internal_numeric, internal_categorical)
    model_context = make_model(augmented_numeric, augmented_categorical)

    model_internal.fit(X_train_int, y_train)
    model_context.fit(X_train_aug, y_train)

    pred = {
        "validation": {
            "y": y_val,
            "internal": model_internal.predict_proba(X_val_int)[:, 1],
            "context": model_context.predict_proba(X_val_aug)[:, 1],
        },
        "test": {
            "y": y_test,
            "internal": model_internal.predict_proba(X_test_int)[:, 1],
            "context": model_context.predict_proba(X_test_aug)[:, 1],
        },
    }

    metrics_rows = []
    topk_rows = []
    boot_rows = []

    for split, obj in pred.items():
        y = obj["y"]
        for model_name in ["internal", "context"]:
            s = obj[model_name]
            metrics_rows.append({
                "split": split,
                "model": model_name,
                "rows": int(len(y)),
                "winners": int(y.sum()),
                "winner_rate": float(y.mean()) if len(y) else None,
                "roc_auc": metric_safe(y, s, "roc_auc"),
                "average_precision": metric_safe(y, s, "average_precision"),
                "brier": metric_safe(y, s, "brier"),
                "log_loss": metric_safe(y, s, "log_loss"),
            })
            topk_rows.extend(topk_metrics(y, s, split, model_name))

        for metric in ["roc_auc", "average_precision"]:
            b = bootstrap_delta(y, obj["internal"], obj["context"], metric)
            b["split"] = split
            boot_rows.append(b)

    labels_df = label_summary(df)
    metrics_df = pd.DataFrame(metrics_rows)
    topk_df = pd.DataFrame(topk_rows)
    boot_df = pd.DataFrame(boot_rows)

    feature_inventory = []
    for c in internal_numeric:
        feature_inventory.append({"feature": c, "group": "internal_numeric"})
    for c in internal_categorical:
        feature_inventory.append({"feature": c, "group": "internal_categorical"})
    for c in context_numeric:
        feature_inventory.append({"feature": c, "group": "context_numeric"})

    feature_df = pd.DataFrame(feature_inventory)

    labels_df.to_csv(OUT / "00_label_distribution_by_chronological_split.csv", index=False, encoding="utf-8-sig")
    feature_df.to_csv(OUT / "01_feature_inventory_internal_vs_context.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(OUT / "02_incremental_predictive_metrics.csv", index=False, encoding="utf-8-sig")
    topk_df.to_csv(OUT / "03_topk_winner_capture.csv", index=False, encoding="utf-8-sig")
    boot_df.to_csv(OUT / "04_bootstrap_metric_delta_context_minus_internal.csv", index=False, encoding="utf-8-sig")

    # Make concise conclusion based on test average precision and ROC-AUC.
    test_boot = boot_df[boot_df["split"] == "test"].copy()

    ap_row = test_boot[test_boot["metric"] == "average_precision"]
    roc_row = test_boot[test_boot["metric"] == "roc_auc"]

    ap_conclusion = ap_row.iloc[0]["conclusion"] if not ap_row.empty else "MISSING"
    roc_conclusion = roc_row.iloc[0]["conclusion"] if not roc_row.empty else "MISSING"

    if ap_conclusion == "CONTEXT_SIGNIFICANTLY_IMPROVES_METRIC" or roc_conclusion == "CONTEXT_SIGNIFICANTLY_IMPROVES_METRIC":
        thesis_conclusion = "CONTEXT_HAS_STATISTICALLY_DETECTABLE_INCREMENTAL_SIGNAL_IN_EVENT_LEVEL_REBUILD"
    elif ap_conclusion == "CONTEXT_SIGNIFICANTLY_DEGRADES_METRIC" and roc_conclusion == "CONTEXT_SIGNIFICANTLY_DEGRADES_METRIC":
        thesis_conclusion = "CONTEXT_DEGRADES_EVENT_LEVEL_REBUILD_METRICS"
    else:
        thesis_conclusion = "NO_STATISTICALLY_STABLE_INCREMENTAL_CONTEXT_GAIN_IN_EVENT_LEVEL_REBUILD"

    summary = {
        "classification": "THESIS_CONTEXT_INCREMENTAL_PREDICTIVE_VALUE_AUDIT_COMPLETED",
        "root": str(ROOT),
        "output_root": str(OUT),
        "dataset_csv": str(dataset_path),
        "safety": {
            "read_only_audit": True,
            "offline_research_training": True,
            "backtest_run": False,
            "trader_db_mutated": False,
            "wallet_connected": False,
            "live_trading_enabled": False,
            "new_llm_calls": False,
            "trade_authority": False,
        },
        "target": "WINNER vs NON_WINNER",
        "model": "class_weight_balanced_logistic_regression",
        "chronological_split": split_counts,
        "feature_counts": {
            "internal_numeric": len(internal_numeric),
            "internal_categorical": len(internal_categorical),
            "context_numeric": len(context_numeric),
        },
        "thesis_conclusion": thesis_conclusion,
        "test_bootstrap_conclusions": test_boot.to_dict(orient="records"),
        "outputs": {
            "label_distribution": str(OUT / "00_label_distribution_by_chronological_split.csv"),
            "feature_inventory": str(OUT / "01_feature_inventory_internal_vs_context.csv"),
            "incremental_predictive_metrics": str(OUT / "02_incremental_predictive_metrics.csv"),
            "topk_winner_capture": str(OUT / "03_topk_winner_capture.csv"),
            "bootstrap_metric_delta": str(OUT / "04_bootstrap_metric_delta_context_minus_internal.csv"),
        },
    }

    with open(OUT / "thesis_context_incremental_predictive_value_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Thesis Context Incremental Predictive Value Audit")
    lines.append("")
    lines.append(f"Output root: `{OUT}`")
    lines.append(f"Dataset: `{dataset_path}`")
    lines.append("")
    lines.append("## Safety")
    lines.append("- Read-only audit")
    lines.append("- Offline research training only")
    lines.append("- No backtest")
    lines.append("- No trader.db mutation")
    lines.append("- No wallet")
    lines.append("- No live trading")
    lines.append("- No new LLM calls")
    lines.append("- No trade authority")
    lines.append("")
    lines.append("## Target")
    lines.append("`WINNER` vs `NON_WINNER`")
    lines.append("")
    lines.append("## Label distribution")
    for _, r in labels_df.iterrows():
        lines.append(
            f"- `{r['split']}`: rows={int(r['rows']):,}, "
            f"winner={int(r['winner'])}, non_winner={int(r['non_winner'])}, "
            f"winner_rate={float(r['winner_rate'])*100:.3f}%, "
            f"flat={int(r['flat'])}, loser={int(r['loser'])}"
        )
    lines.append("")
    lines.append("## Feature groups")
    lines.append(f"- Internal numeric features: {len(internal_numeric)}")
    lines.append(f"- Internal categorical features: {len(internal_categorical)}")
    lines.append(f"- Context numeric features: {len(context_numeric)}")
    lines.append("")
    lines.append("## Validation/Test metrics")
    for _, r in metrics_df.iterrows():
        lines.append(
            f"- `{r['split']}` / `{r['model']}`: "
            f"rows={int(r['rows'])}, winners={int(r['winners'])}, "
            f"ROC-AUC={r['roc_auc']}, AP={r['average_precision']}, "
            f"Brier={r['brier']}, LogLoss={r['log_loss']}"
        )
    lines.append("")
    lines.append("## Bootstrap delta: context minus internal")
    for _, r in boot_df.iterrows():
        lines.append(
            f"- `{r['split']}` / `{r['metric']}`: "
            f"internal={r['observed_internal']}, context={r['observed_context']}, "
            f"delta={r['observed_delta_context_minus_internal']}, "
            f"CI95=[{r['ci95_low']}, {r['ci95_high']}], "
            f"p≈{r['p_two_sided_delta_zero']}, "
            f"conclusion=`{r['conclusion']}`"
        )
    lines.append("")
    lines.append("## Top-K winner capture")
    for _, r in topk_df.iterrows():
        lines.append(
            f"- `{r['split']}` / `{r['model']}` / top {float(r['top_fraction'])*100:.0f}%: "
            f"k={int(r['k'])}, winners={int(r['winners_in_top_k'])}/"
            f"{int(r['total_winners_in_split'])}, "
            f"winner_rate={float(r['winner_rate_top_k'])*100:.3f}%, "
            f"lift={r['lift_vs_base_rate']}, "
            f"capture={r['winner_capture_rate']}"
        )
    lines.append("")
    lines.append("## Thesis conclusion")
    lines.append(f"`{thesis_conclusion}`")
    lines.append("")
    if thesis_conclusion == "CONTEXT_HAS_STATISTICALLY_DETECTABLE_INCREMENTAL_SIGNAL_IN_EVENT_LEVEL_REBUILD":
        lines.append(
            "The event-level rebuild provides evidence that timestamped context features add measurable incremental signal "
            "over the internal-only feature block under chronological validation. This remains offline retrospective research evidence, "
            "not paper-forward or live-trading proof."
        )
    elif thesis_conclusion == "NO_STATISTICALLY_STABLE_INCREMENTAL_CONTEXT_GAIN_IN_EVENT_LEVEL_REBUILD":
        lines.append(
            "The event-level rebuild did not establish a statistically stable incremental gain from the context feature block. "
            "The context layer remains supported for auditability, explanation, and provenance, but not as independently validated predictive lift in this audit."
        )
    else:
        lines.append(
            "The event-level rebuild suggests that the context feature block degraded model metrics in this configuration. "
            "This should be reported conservatively and investigated before any predictive claim."
        )
    lines.append("")
    lines.append("## Output files")
    for _, path in summary["outputs"].items():
        lines.append(f"- `{Path(path).name}`")

    md = OUT / "thesis_context_incremental_predictive_value_summary.md"
    md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(OUT / "thesis_context_incremental_predictive_value_summary.json"),
        "summary_md": str(md),
        "thesis_conclusion": thesis_conclusion,
    }, indent=2, ensure_ascii=False))

    print()
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
