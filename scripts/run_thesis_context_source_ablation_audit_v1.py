from __future__ import annotations

import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("THESIS_ROOT", r"E:\Projects\Final Project\memecoin_trader"))
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = ROOT / "data" / "audits" / f"thesis_context_source_ablation_audit_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

DATASET_CSV = os.environ.get("THESIS_CONTEXT_DATASET_CSV", "").strip()
BOOTSTRAP_N = int(os.environ.get("THESIS_BOOTSTRAP_N", "3000"))
RANDOM_SEED = int(os.environ.get("THESIS_RANDOM_SEED", "42"))

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def find_dataset() -> Path:
    if DATASET_CSV:
        p = Path(DATASET_CSV)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    candidates = list(
        (ROOT / "data" / "audits").glob(
            "thesis_context_event_level_dataset_build_audit_*/03_event_level_context_rebuild_dataset.csv"
        )
    )
    if not candidates:
        raise FileNotFoundError("No dataset CSV found.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_sklearn():
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    }


SK = load_sklearn()

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

BLOCKED_COLS = {
    "candidate_event_id",
    "canonical_coin_id",
    "event_bucket",
    "candidate_event_start_utc",
    "candidate_event_time_utc",
    "candidate_event_end_utc",
    "chronological_split",
}


def is_blocked(col: str) -> bool:
    c = col.lower()
    if col in BLOCKED_COLS:
        return True
    return any(p in c for p in LEAKAGE_PATTERNS)


def numeric_nonempty(df: pd.DataFrame, col: str) -> bool:
    return pd.to_numeric(df[col], errors="coerce").notna().sum() > 0


def pick_features(df: pd.DataFrame, train: pd.DataFrame) -> dict[str, list[str]]:
    internal_numeric = []
    internal_categorical = []
    raw_payload = []
    pool_flow = []
    gemini = []

    for c in df.columns:
        if is_blocked(c):
            continue

        if c.startswith("raw_payload_"):
            if numeric_nonempty(train, c):
                raw_payload.append(c)
            continue

        if c.startswith("pool_flow_"):
            if numeric_nonempty(train, c):
                pool_flow.append(c)
            continue

        if c.startswith("gemini_"):
            if numeric_nonempty(train, c):
                gemini.append(c)
            continue

        if (
            c == "signal_rows"
            or c.startswith("asof_")
            or c.startswith("score_")
            or c.startswith("signal_score_")
            or c.startswith("confidence_")
            or c.startswith("signal_confidence_")
        ):
            if numeric_nonempty(train, c):
                internal_numeric.append(c)
            continue

        if c.endswith("_top_values"):
            nunique = train[c].astype(str).nunique(dropna=True)
            if 1 < nunique <= 100:
                internal_categorical.append(c)

    return {
        "internal_numeric": sorted(set(internal_numeric)),
        "internal_categorical": sorted(set(internal_categorical)),
        "raw_payload": sorted(set(raw_payload)),
        "pool_flow": sorted(set(pool_flow)),
        "gemini": sorted(set(gemini)),
    }


def make_model(numeric_features: list[str], categorical_features: list[str]):
    Pipeline = SK["Pipeline"]
    ColumnTransformer = SK["ColumnTransformer"]
    SimpleImputer = SK["SimpleImputer"]
    StandardScaler = SK["StandardScaler"]
    LogisticRegression = SK["LogisticRegression"]

    transformers = []

    if numeric_features:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_features,
        ))

    if categorical_features:
        try:
            ohe = SK["OneHotEncoder"](handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = SK["OneHotEncoder"](handle_unknown="ignore", sparse=False)

        transformers.append((
            "cat",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", ohe),
            ]),
            categorical_features,
        ))

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


def safe_metric(y: np.ndarray, s: np.ndarray, metric: str) -> float | None:
    if len(np.unique(y)) < 2:
        return None

    if metric == "roc_auc":
        return float(SK["roc_auc_score"](y, s))
    if metric == "average_precision":
        return float(SK["average_precision_score"](y, s))
    if metric == "brier":
        return float(SK["brier_score_loss"](y, s))
    if metric == "log_loss":
        ss = np.clip(s, 1e-12, 1 - 1e-12)
        return float(SK["log_loss"](y, ss, labels=[0, 1]))

    raise ValueError(metric)


def bootstrap_delta(y: np.ndarray, base: np.ndarray, challenger: np.ndarray, metric: str) -> dict[str, Any]:
    base_obs = safe_metric(y, base, metric)
    ch_obs = safe_metric(y, challenger, metric)

    if base_obs is None or ch_obs is None:
        return {
            "metric": metric,
            "internal": base_obs,
            "challenger": ch_obs,
            "delta": None,
            "ci95_low": None,
            "ci95_high": None,
            "p_two_sided": None,
            "conclusion": "NOT_COMPUTABLE",
        }

    rng = np.random.default_rng(RANDOM_SEED)
    deltas = []
    n = len(y)

    for _ in range(BOOTSTRAP_N):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        a = safe_metric(yy, base[idx], metric)
        b = safe_metric(yy, challenger[idx], metric)
        if a is not None and b is not None:
            deltas.append(b - a)

    arr = np.array(deltas)
    lo, hi = np.quantile(arr, [0.025, 0.975])
    p_two = float(min(1.0, 2 * min(np.mean(arr <= 0), np.mean(arr >= 0))))

    if lo > 0:
        conclusion = "SIGNIFICANT_IMPROVEMENT"
    elif hi < 0:
        conclusion = "SIGNIFICANT_DEGRADATION"
    else:
        conclusion = "NO_SIGNIFICANT_DELTA"

    return {
        "metric": metric,
        "internal": base_obs,
        "challenger": ch_obs,
        "delta": ch_obs - base_obs,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_two_sided": p_two,
        "bootstrap_n_valid": int(len(arr)),
        "conclusion": conclusion,
    }


def topk(y: np.ndarray, s: np.ndarray, split: str, model_name: str) -> list[dict[str, Any]]:
    rows = []
    n = len(y)
    total_winners = int(y.sum())
    base_rate = total_winners / n if n else 0

    for frac in [0.01, 0.02, 0.05, 0.10]:
        k = max(1, int(round(n * frac)))
        idx = np.argsort(-s)[:k]
        winners = int(y[idx].sum())
        rate = winners / k if k else None
        rows.append({
            "split": split,
            "model": model_name,
            "top_fraction": frac,
            "k": k,
            "winners_in_top_k": winners,
            "total_winners": total_winners,
            "winner_rate_top_k": rate,
            "base_winner_rate": base_rate,
            "lift": rate / base_rate if base_rate and rate is not None else None,
            "capture_rate": winners / total_winners if total_winners else None,
        })

    return rows


def main() -> None:
    dataset_path = find_dataset()
    df = pd.read_csv(dataset_path, low_memory=False)

    df = df[df["label_available"].astype(str).str.upper().isin(["TRUE", "1"])].copy()
    df = df[df["label_x2_sl_4h"].isin(["WINNER", "LOSER", "FLAT"])].copy()
    df["target_winner"] = (df["label_x2_sl_4h"] == "WINNER").astype(int)

    train = df[df["chronological_split"] == "train"].copy()
    validation = df[df["chronological_split"] == "validation"].copy()
    test = df[df["chronological_split"] == "test"].copy()

    features = pick_features(df, train)

    internal_num = features["internal_numeric"]
    internal_cat = features["internal_categorical"]

    feature_sets = {
        "internal": {
            "numeric": internal_num,
            "categorical": internal_cat,
            "description": "internal-only baseline",
        },
        "internal_plus_raw_payload": {
            "numeric": internal_num + features["raw_payload"],
            "categorical": internal_cat,
            "description": "internal + raw provider activity/provenance counts",
        },
        "internal_plus_pool_flow": {
            "numeric": internal_num + features["pool_flow"],
            "categorical": internal_cat,
            "description": "internal + pool-flow proxy features",
        },
        "internal_plus_gemini": {
            "numeric": internal_num + features["gemini"],
            "categorical": internal_cat,
            "description": "internal + prior Gemini/LLM record features",
        },
        "internal_plus_raw_pool": {
            "numeric": internal_num + features["raw_payload"] + features["pool_flow"],
            "categorical": internal_cat,
            "description": "internal + raw provider + pool-flow proxy",
        },
        "internal_plus_pool_gemini": {
            "numeric": internal_num + features["pool_flow"] + features["gemini"],
            "categorical": internal_cat,
            "description": "internal + pool-flow proxy + Gemini",
        },
        "internal_plus_all_context": {
            "numeric": internal_num + features["raw_payload"] + features["pool_flow"] + features["gemini"],
            "categorical": internal_cat,
            "description": "internal + all available context blocks",
        },
    }

    y = {
        "train": train["target_winner"].to_numpy(),
        "validation": validation["target_winner"].to_numpy(),
        "test": test["target_winner"].to_numpy(),
    }

    preds: dict[str, dict[str, np.ndarray]] = {"validation": {}, "test": {}}
    metric_rows = []
    topk_rows = []

    for name, spec in feature_sets.items():
        numeric = list(dict.fromkeys(spec["numeric"]))
        categorical = list(dict.fromkeys(spec["categorical"]))

        if not numeric and not categorical:
            continue

        train_cols = numeric + categorical
        model = make_model(numeric, categorical)
        model.fit(train[train_cols], y["train"])

        for split_name, split_df in [("validation", validation), ("test", test)]:
            yy = y[split_name]
            score = model.predict_proba(split_df[train_cols])[:, 1]
            preds[split_name][name] = score

            metric_rows.append({
                "split": split_name,
                "model": name,
                "description": spec["description"],
                "rows": int(len(yy)),
                "winners": int(yy.sum()),
                "winner_rate": float(yy.mean()) if len(yy) else None,
                "roc_auc": safe_metric(yy, score, "roc_auc"),
                "average_precision": safe_metric(yy, score, "average_precision"),
                "brier": safe_metric(yy, score, "brier"),
                "log_loss": safe_metric(yy, score, "log_loss"),
                "numeric_feature_count": len(numeric),
                "categorical_feature_count": len(categorical),
            })

            topk_rows.extend(topk(yy, score, split_name, name))

    bootstrap_rows = []
    for split_name in ["validation", "test"]:
        if "internal" not in preds[split_name]:
            continue
        base = preds[split_name]["internal"]
        yy = y[split_name]

        for model_name, score in preds[split_name].items():
            if model_name == "internal":
                continue
            for metric in ["roc_auc", "average_precision"]:
                row = bootstrap_delta(yy, base, score, metric)
                row["split"] = split_name
                row["challenger_model"] = model_name
                bootstrap_rows.append(row)

    label_rows = []
    for split_name, split_df in [("train", train), ("validation", validation), ("test", test)]:
        label_rows.append({
            "split": split_name,
            "rows": int(len(split_df)),
            "winner": int((split_df["target_winner"] == 1).sum()),
            "non_winner": int((split_df["target_winner"] == 0).sum()),
            "winner_rate": float(split_df["target_winner"].mean()) if len(split_df) else None,
            "flat": int((split_df["label_x2_sl_4h"] == "FLAT").sum()),
            "loser": int((split_df["label_x2_sl_4h"] == "LOSER").sum()),
        })

    feature_inventory = []
    for group, cols in features.items():
        for c in cols:
            feature_inventory.append({"feature_group": group, "feature": c})

    labels_df = pd.DataFrame(label_rows)
    features_df = pd.DataFrame(feature_inventory)
    metrics_df = pd.DataFrame(metric_rows)
    topk_df = pd.DataFrame(topk_rows)
    boot_df = pd.DataFrame(bootstrap_rows)

    labels_df.to_csv(OUT / "00_label_distribution.csv", index=False, encoding="utf-8-sig")
    features_df.to_csv(OUT / "01_feature_inventory.csv", index=False, encoding="utf-8-sig")
    metrics_df.to_csv(OUT / "02_ablation_metrics.csv", index=False, encoding="utf-8-sig")
    topk_df.to_csv(OUT / "03_ablation_topk_winner_capture.csv", index=False, encoding="utf-8-sig")
    boot_df.to_csv(OUT / "04_ablation_bootstrap_deltas_vs_internal.csv", index=False, encoding="utf-8-sig")

    test_boot = boot_df[boot_df["split"] == "test"].copy()
    significant_improvements = test_boot[test_boot["conclusion"] == "SIGNIFICANT_IMPROVEMENT"]
    significant_degradations = test_boot[test_boot["conclusion"] == "SIGNIFICANT_DEGRADATION"]

    if not significant_improvements.empty:
        thesis_conclusion = "AT_LEAST_ONE_CONTEXT_SOURCE_SHOWS_SIGNIFICANT_TEST_IMPROVEMENT_EXPLORATORY_ABLATION"
    elif not significant_degradations.empty:
        thesis_conclusion = "CONTEXT_SOURCES_DO_NOT_IMPROVE_AND_SOME_SIGNIFICANTLY_DEGRADE_TEST_METRICS"
    else:
        thesis_conclusion = "NO_CONTEXT_SOURCE_SHOWS_SIGNIFICANT_TEST_IMPROVEMENT_IN_ABLATION"

    summary = {
        "classification": "THESIS_CONTEXT_SOURCE_ABLATION_AUDIT_COMPLETED",
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
        "bootstrap_n": BOOTSTRAP_N,
        "feature_counts": {k: len(v) for k, v in features.items()},
        "thesis_conclusion": thesis_conclusion,
        "outputs": {
            "label_distribution": str(OUT / "00_label_distribution.csv"),
            "feature_inventory": str(OUT / "01_feature_inventory.csv"),
            "ablation_metrics": str(OUT / "02_ablation_metrics.csv"),
            "ablation_topk_winner_capture": str(OUT / "03_ablation_topk_winner_capture.csv"),
            "ablation_bootstrap_deltas": str(OUT / "04_ablation_bootstrap_deltas_vs_internal.csv"),
        },
    }

    with open(OUT / "thesis_context_source_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Thesis Context Source Ablation Audit")
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
    lines.append("## Label distribution")
    for _, r in labels_df.iterrows():
        lines.append(
            f"- `{r['split']}`: rows={int(r['rows']):,}, "
            f"winner={int(r['winner'])}, non_winner={int(r['non_winner'])}, "
            f"winner_rate={float(r['winner_rate'])*100:.3f}%, "
            f"flat={int(r['flat'])}, loser={int(r['loser'])}"
        )
    lines.append("")
    lines.append("## Feature counts")
    for k, v in summary["feature_counts"].items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Metrics")
    for _, r in metrics_df.iterrows():
        lines.append(
            f"- `{r['split']}` / `{r['model']}`: "
            f"ROC-AUC={r['roc_auc']}, AP={r['average_precision']}, "
            f"Brier={r['brier']}, LogLoss={r['log_loss']}, "
            f"features={int(r['numeric_feature_count'])}+{int(r['categorical_feature_count'])}"
        )
    lines.append("")
    lines.append("## Bootstrap deltas vs internal")
    for _, r in boot_df.iterrows():
        lines.append(
            f"- `{r['split']}` / `{r['challenger_model']}` / `{r['metric']}`: "
            f"internal={r['internal']}, challenger={r['challenger']}, "
            f"delta={r['delta']}, CI95=[{r['ci95_low']}, {r['ci95_high']}], "
            f"p≈{r['p_two_sided']}, conclusion=`{r['conclusion']}`"
        )
    lines.append("")
    lines.append("## Top-K winner capture")
    for _, r in topk_df.iterrows():
        lines.append(
            f"- `{r['split']}` / `{r['model']}` / top {float(r['top_fraction'])*100:.0f}%: "
            f"k={int(r['k'])}, winners={int(r['winners_in_top_k'])}/"
            f"{int(r['total_winners'])}, "
            f"winner_rate={float(r['winner_rate_top_k'])*100:.3f}%, "
            f"lift={r['lift']}, capture={r['capture_rate']}"
        )
    lines.append("")
    lines.append("## Thesis conclusion")
    lines.append(f"`{thesis_conclusion}`")
    lines.append("")
    lines.append("## Output files")
    for _, path in summary["outputs"].items():
        lines.append(f"- `{Path(path).name}`")

    md = OUT / "thesis_context_source_ablation_summary.md"
    md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(OUT / "thesis_context_source_ablation_summary.json"),
        "summary_md": str(md),
        "thesis_conclusion": thesis_conclusion,
    }, indent=2, ensure_ascii=False))

    print()
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
