from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("THESIS_ROOT", r"E:\Projects\Final Project\memecoin_trader"))
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = ROOT / "data" / "audits" / f"thesis_context_multiplicity_reporting_audit_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

ABLATION_ROOT_ENV = os.environ.get("THESIS_CONTEXT_ABLATION_ROOT", "").strip()
ALPHA = float(os.environ.get("THESIS_ALPHA", "0.05"))


def find_latest_ablation_root() -> Path:
    if ABLATION_ROOT_ENV:
        p = Path(ABLATION_ROOT_ENV)
        if not p.exists():
            raise FileNotFoundError(f"THESIS_CONTEXT_ABLATION_ROOT not found: {p}")
        return p

    candidates = list((ROOT / "data" / "audits").glob("thesis_context_source_ablation_audit_*"))
    if not candidates:
        raise FileNotFoundError("No thesis_context_source_ablation_audit_* directory found.")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def bh_fdr(p_values: pd.Series) -> pd.Series:
    """
    Benjamini-Hochberg adjusted q-values.
    """
    p = pd.to_numeric(p_values, errors="coerce")
    out = pd.Series(np.nan, index=p.index, dtype=float)

    valid = p.dropna()
    if valid.empty:
        return out

    order = valid.sort_values().index
    ranked = valid.loc[order].to_numpy(dtype=float)
    m = len(ranked)

    raw_q = ranked * m / np.arange(1, m + 1)
    # enforce monotonicity from largest rank backward
    q = np.minimum.accumulate(raw_q[::-1])[::-1]
    q = np.clip(q, 0, 1)

    out.loc[order] = q
    return out


def direction(delta: Any) -> str:
    try:
        d = float(delta)
    except Exception:
        return "UNKNOWN"
    if d > 0:
        return "IMPROVEMENT"
    if d < 0:
        return "DEGRADATION"
    return "NO_CHANGE"


def interpret_row(row: pd.Series, family: str) -> str:
    d = direction(row.get("delta"))
    p = row.get("p_two_sided")
    q = row.get("bh_fdr_q")
    bonf = row.get("bonferroni_p")

    try:
        p = float(p)
    except Exception:
        p = np.nan
    try:
        q = float(q)
    except Exception:
        q = np.nan
    try:
        bonf = float(bonf)
    except Exception:
        bonf = np.nan

    if d == "IMPROVEMENT":
        if not np.isnan(bonf) and bonf < ALPHA:
            return "CORRECTION_ROBUST_IMPROVEMENT_BONFERRONI"
        if not np.isnan(q) and q < ALPHA:
            return "FDR05_IMPROVEMENT"
        if not np.isnan(q) and q < 0.10:
            return "FDR10_EXPLORATORY_IMPROVEMENT"
        if not np.isnan(p) and p < ALPHA:
            return "NOMINAL_EXPLORATORY_IMPROVEMENT_ONLY"
        return "NO_SIGNIFICANT_IMPROVEMENT"

    if d == "DEGRADATION":
        if not np.isnan(bonf) and bonf < ALPHA:
            return "CORRECTION_ROBUST_DEGRADATION_BONFERRONI"
        if not np.isnan(q) and q < ALPHA:
            return "FDR05_DEGRADATION"
        if not np.isnan(q) and q < 0.10:
            return "FDR10_EXPLORATORY_DEGRADATION"
        if not np.isnan(p) and p < ALPHA:
            return "NOMINAL_EXPLORATORY_DEGRADATION_ONLY"
        return "NO_SIGNIFICANT_DEGRADATION"

    return "NO_DIRECTIONAL_DELTA"


def load_required(ablation_root: Path) -> dict[str, pd.DataFrame]:
    paths = {
        "labels": ablation_root / "00_label_distribution.csv",
        "features": ablation_root / "01_feature_inventory.csv",
        "metrics": ablation_root / "02_ablation_metrics.csv",
        "topk": ablation_root / "03_ablation_topk_winner_capture.csv",
        "bootstrap": ablation_root / "04_ablation_bootstrap_deltas_vs_internal.csv",
    }

    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing ablation output files:\n" + "\n".join(missing))

    return {k: pd.read_csv(v) for k, v in paths.items()}


def add_corrections(df: pd.DataFrame, family_name: str) -> pd.DataFrame:
    out = df.copy()
    out["p_two_sided"] = pd.to_numeric(out["p_two_sided"], errors="coerce")
    m = int(out["p_two_sided"].notna().sum())
    out["test_family"] = family_name
    out["family_n_tests"] = m
    out["bonferroni_p"] = np.minimum(out["p_two_sided"] * m, 1.0)
    out["bh_fdr_q"] = bh_fdr(out["p_two_sided"])
    out["delta_direction"] = out["delta"].map(direction)
    out["reporting_interpretation"] = out.apply(lambda r: interpret_row(r, family_name), axis=1)
    return out


def summarize_topk(topk: pd.DataFrame) -> pd.DataFrame:
    test = topk[topk["split"] == "test"].copy()
    if test.empty:
        return pd.DataFrame()

    internal = test[test["model"] == "internal"].copy()
    rows = []

    for _, r in test.iterrows():
        frac = r["top_fraction"]
        base = internal[internal["top_fraction"] == frac]
        if base.empty:
            continue
        b = base.iloc[0]

        rows.append({
            "model": r["model"],
            "top_fraction": frac,
            "k": int(r["k"]),
            "winners_in_top_k": int(r["winners_in_top_k"]),
            "total_winners": int(r["total_winners"]),
            "capture_rate": float(r["capture_rate"]),
            "winner_rate_top_k": float(r["winner_rate_top_k"]),
            "lift": float(r["lift"]) if pd.notna(r["lift"]) else np.nan,
            "internal_winners_in_top_k": int(b["winners_in_top_k"]),
            "internal_capture_rate": float(b["capture_rate"]),
            "winner_capture_delta_vs_internal": float(r["capture_rate"]) - float(b["capture_rate"]),
            "winner_count_delta_vs_internal": int(r["winners_in_top_k"]) - int(b["winners_in_top_k"]),
        })

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["top_fraction", "winner_count_delta_vs_internal", "capture_rate"],
        ascending=[True, False, False],
    )


def build_decision_table(primary: pd.DataFrame, secondary_all: pd.DataFrame, metrics: pd.DataFrame, topk_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def get_primary(model: str) -> pd.Series | None:
        x = primary[primary["challenger_model"] == model]
        if x.empty:
            return None
        return x.iloc[0]

    def get_metric(split: str, model: str) -> pd.Series | None:
        x = metrics[(metrics["split"] == split) & (metrics["model"] == model)]
        if x.empty:
            return None
        return x.iloc[0]

    model_labels = {
        "internal_plus_raw_payload": "raw_provider_activity_provenance",
        "internal_plus_pool_flow": "pool_flow_proxy",
        "internal_plus_gemini": "gemini_llm_records",
        "internal_plus_raw_pool": "raw_provider_plus_pool_flow",
        "internal_plus_pool_gemini": "pool_flow_plus_gemini",
        "internal_plus_all_context": "all_context_combined",
    }

    for model, source_label in model_labels.items():
        p = get_primary(model)
        m = get_metric("test", model)

        row = {
            "context_source": source_label,
            "model": model,
            "test_ap": None,
            "test_roc_auc": None,
            "primary_test_ap_delta_vs_internal": None,
            "primary_test_ap_p": None,
            "primary_test_ap_bh_fdr_q": None,
            "primary_test_ap_bonferroni_p": None,
            "primary_test_ap_interpretation": "MISSING",
            "thesis_reporting_level": "DO_NOT_REPORT_AS_PREDICTIVE",
        }

        if m is not None:
            row["test_ap"] = m.get("average_precision")
            row["test_roc_auc"] = m.get("roc_auc")

        if p is not None:
            row["primary_test_ap_delta_vs_internal"] = p.get("delta")
            row["primary_test_ap_p"] = p.get("p_two_sided")
            row["primary_test_ap_bh_fdr_q"] = p.get("bh_fdr_q")
            row["primary_test_ap_bonferroni_p"] = p.get("bonferroni_p")
            row["primary_test_ap_interpretation"] = p.get("reporting_interpretation")

            interp = str(p.get("reporting_interpretation"))
            if "BONFERRONI" in interp or "FDR05_IMPROVEMENT" in interp:
                row["thesis_reporting_level"] = "CORRECTION_ROBUST_INCREMENTAL_SIGNAL"
            elif "FDR10_EXPLORATORY_IMPROVEMENT" in interp:
                row["thesis_reporting_level"] = "EXPLORATORY_FDR10_SIGNAL"
            elif "NOMINAL_EXPLORATORY_IMPROVEMENT_ONLY" in interp:
                row["thesis_reporting_level"] = "NOMINAL_EXPLORATORY_SIGNAL_ONLY"
            elif "DEGRADATION" in interp:
                row["thesis_reporting_level"] = "NEGATIVE_OR_NOISY_CONTEXT_EVIDENCE"
            else:
                row["thesis_reporting_level"] = "NO_STABLE_INCREMENTAL_SIGNAL"

        # Add top-K exploratory notes.
        tk = topk_summary[
            (topk_summary["model"] == model)
            & (topk_summary["top_fraction"].isin([0.01, 0.02, 0.05, 0.10]))
        ].copy()

        if not tk.empty:
            best = tk.sort_values(
                ["winner_count_delta_vs_internal", "top_fraction"],
                ascending=[False, True],
            ).iloc[0]
            row["best_topk_fraction"] = best["top_fraction"]
            row["best_topk_winner_delta_vs_internal"] = best["winner_count_delta_vs_internal"]
            row["best_topk_capture_rate"] = best["capture_rate"]
            row["best_topk_internal_capture_rate"] = best["internal_capture_rate"]
        else:
            row["best_topk_fraction"] = None
            row["best_topk_winner_delta_vs_internal"] = None
            row["best_topk_capture_rate"] = None
            row["best_topk_internal_capture_rate"] = None

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    ablation_root = find_latest_ablation_root()
    data = load_required(ablation_root)

    boot = data["bootstrap"].copy()
    metrics = data["metrics"].copy()
    topk = data["topk"].copy()
    labels = data["labels"].copy()
    features = data["features"].copy()

    # Primary reporting family: test split, Average Precision only, one comparison per context source.
    # This is the most thesis-relevant family for rare-winner detection.
    primary = boot[
        (boot["split"] == "test")
        & (boot["metric"] == "average_precision")
    ].copy()

    primary_corr = add_corrections(primary, "PRIMARY_TEST_AVERAGE_PRECISION_CONTEXT_SOURCE_FAMILY")

    # Secondary broader families for caution.
    test_all = boot[boot["split"] == "test"].copy()
    test_all_corr = add_corrections(test_all, "SECONDARY_TEST_ALL_METRICS_CONTEXT_SOURCE_FAMILY")

    all_rows_corr = add_corrections(boot.copy(), "EXPLORATORY_ALL_SPLITS_ALL_METRICS_FAMILY")

    topk_summary = summarize_topk(topk)
    decision = build_decision_table(primary_corr, all_rows_corr, metrics, topk_summary)

    primary_corr.to_csv(OUT / "00_primary_test_average_precision_multiplicity.csv", index=False, encoding="utf-8-sig")
    test_all_corr.to_csv(OUT / "01_secondary_test_all_metrics_multiplicity.csv", index=False, encoding="utf-8-sig")
    all_rows_corr.to_csv(OUT / "02_exploratory_all_bootstrap_multiplicity.csv", index=False, encoding="utf-8-sig")
    topk_summary.to_csv(OUT / "03_test_topk_delta_vs_internal.csv", index=False, encoding="utf-8-sig")
    decision.to_csv(OUT / "04_context_source_reporting_decision_table.csv", index=False, encoding="utf-8-sig")

    # Extract key findings.
    raw_row = decision[decision["model"] == "internal_plus_raw_payload"]
    all_row = decision[decision["model"] == "internal_plus_all_context"]
    gemini_row = decision[decision["model"] == "internal_plus_gemini"]
    pool_row = decision[decision["model"] == "internal_plus_pool_flow"]

    def one(df: pd.DataFrame) -> dict[str, Any]:
        if df.empty:
            return {}
        return df.iloc[0].to_dict()

    raw = one(raw_row)
    allc = one(all_row)
    gem = one(gemini_row)
    pool = one(pool_row)

    # Determine overall reporting conclusion.
    robust_improvement = decision[
        decision["thesis_reporting_level"].isin([
            "CORRECTION_ROBUST_INCREMENTAL_SIGNAL",
            "EXPLORATORY_FDR10_SIGNAL",
        ])
    ]
    nominal_improvement = decision[
        decision["thesis_reporting_level"] == "NOMINAL_EXPLORATORY_SIGNAL_ONLY"
    ]
    negative = decision[
        decision["thesis_reporting_level"] == "NEGATIVE_OR_NOISY_CONTEXT_EVIDENCE"
    ]

    if not robust_improvement.empty:
        reporting_conclusion = "SELECTED_CONTEXT_SOURCE_SHOWS_MULTIPLICITY_AWARE_EXPLORATORY_OR_ROBUST_SIGNAL"
    elif not nominal_improvement.empty:
        reporting_conclusion = "SELECTED_CONTEXT_SOURCE_SHOWS_NOMINAL_ONLY_EXPLORATORY_SIGNAL"
    elif not negative.empty:
        reporting_conclusion = "CONTEXT_NOISY_OR_DEGRADING_WITH_NO_CORRECTION_SAFE_IMPROVEMENT"
    else:
        reporting_conclusion = "NO_CONTEXT_SOURCE_SUPPORTS_INCREMENTAL_PREDICTIVE_CLAIM"

    summary = {
        "classification": "THESIS_CONTEXT_MULTIPLICITY_REPORTING_AUDIT_COMPLETED",
        "root": str(ROOT),
        "output_root": str(OUT),
        "ablation_root": str(ablation_root),
        "alpha": ALPHA,
        "safety": {
            "read_only_post_processing": True,
            "offline_research_training": False,
            "new_model_training": False,
            "backtest_run": False,
            "trader_db_mutated": False,
            "wallet_connected": False,
            "live_trading_enabled": False,
            "new_llm_calls": False,
            "trade_authority": False,
        },
        "primary_family": "test split Average Precision deltas vs internal across context-source ablations",
        "primary_family_n_tests": int(len(primary_corr)),
        "reporting_conclusion": reporting_conclusion,
        "raw_payload_decision": raw,
        "all_context_decision": allc,
        "gemini_decision": gem,
        "pool_flow_decision": pool,
        "outputs": {
            "primary_test_average_precision_multiplicity": str(OUT / "00_primary_test_average_precision_multiplicity.csv"),
            "secondary_test_all_metrics_multiplicity": str(OUT / "01_secondary_test_all_metrics_multiplicity.csv"),
            "exploratory_all_bootstrap_multiplicity": str(OUT / "02_exploratory_all_bootstrap_multiplicity.csv"),
            "test_topk_delta_vs_internal": str(OUT / "03_test_topk_delta_vs_internal.csv"),
            "context_source_reporting_decision_table": str(OUT / "04_context_source_reporting_decision_table.csv"),
            "thesis_safe_context_wording": str(OUT / "05_thesis_safe_context_wording.md"),
        },
    }

    safe_wording = []
    safe_wording.append("# Thesis-safe RQ5 Context Reporting Wording")
    safe_wording.append("")
    safe_wording.append("## Core conclusion")
    safe_wording.append("")
    safe_wording.append(
        "The event-level rebuild did not validate context as a uniformly beneficial predictive layer. "
        "When all available context sources were added together, performance degraded, indicating that sparse or noisy context can harm rare-winner detection. "
        "However, source-level ablation identified a narrow raw-provider activity/provenance feature block that improved test-set Average Precision over the internal-only baseline before multiplicity correction. "
        "After correcting for the multiple source-level comparisons, this effect should be reported as exploratory rather than as correction-robust proof."
    )
    safe_wording.append("")
    safe_wording.append("## Frequency-mismatch limitation")
    safe_wording.append("")
    safe_wording.append(
        "A central limitation is the frequency mismatch between market activity and external context. "
        "Meme-coin market events and internal signals can update many times per minute, whereas several external or semantic context sources are slower, coarser, or event-based, sometimes closer to daily cadence. "
        "This makes it difficult to attribute short-horizon four-hour market outcomes to external context observations. "
        "Accordingly, the context layer should be interpreted primarily as provenance, regime, audit, and risk-explanation infrastructure, with only limited exploratory evidence for incremental prediction from high-coverage provider-activity features."
    )
    safe_wording.append("")
    safe_wording.append("## Recommended thesis sentence")
    safe_wording.append("")
    safe_wording.append(
        "In a clean event-level rebuild, the combined context block did not improve rare-winner prediction and in some configurations degraded Average Precision. "
        "A source-level ablation found that raw-provider activity/provenance features produced a small nominal improvement in test-set Average Precision over the internal-only baseline, while Gemini/LLM-derived and all-context combinations did not provide stable predictive lift. "
        "Because multiple context-source comparisons were evaluated and the test set contained only a very small number of winners, this result is treated as exploratory. "
        "The stronger validated role of context in this thesis is therefore auditability, provenance, and risk interpretation, rather than general short-horizon predictive authority."
    )
    safe_wording.append("")
    safe_wording.append("## Do not claim")
    safe_wording.append("")
    safe_wording.append("- Do not claim that context generally improves meme-coin prediction.")
    safe_wording.append("- Do not claim semantic/LLM context has validated predictive lift.")
    safe_wording.append("- Do not claim causal effect from external context to short-horizon price outcome.")
    safe_wording.append("- Do not claim live or paper-forward profitability.")
    safe_wording.append("")
    safe_wording.append("## Acceptable claim")
    safe_wording.append("")
    safe_wording.append(
        "Selected high-coverage provider-activity context showed exploratory incremental signal in one source-level ablation, while broader context remained noisy and primarily useful for provenance, auditability, and explanation."
    )

    wording_path = OUT / "05_thesis_safe_context_wording.md"
    wording_path.write_text("\n".join(safe_wording), encoding="utf-8")

    with open(OUT / "thesis_context_multiplicity_reporting_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Thesis Context Multiplicity + Reporting Audit")
    lines.append("")
    lines.append(f"Output root: `{OUT}`")
    lines.append(f"Ablation root: `{ablation_root}`")
    lines.append("")
    lines.append("## Safety")
    lines.append("- Read-only post-processing audit")
    lines.append("- No new model training")
    lines.append("- No backtest")
    lines.append("- No trader.db mutation")
    lines.append("- No wallet")
    lines.append("- No live trading")
    lines.append("- No new LLM calls")
    lines.append("- No trade authority")
    lines.append("")
    lines.append("## Primary multiplicity family")
    lines.append("`test split Average Precision deltas vs internal across context-source ablations`")
    lines.append(f"- n_tests: {len(primary_corr)}")
    lines.append("")
    lines.append("## Primary test AP corrections")
    for _, r in primary_corr.sort_values("p_two_sided").iterrows():
        lines.append(
            f"- `{r['challenger_model']}`: "
            f"delta={r['delta']}, p={r['p_two_sided']}, "
            f"BH-FDR q={r['bh_fdr_q']}, Bonferroni p={r['bonferroni_p']}, "
            f"direction={r['delta_direction']}, interpretation=`{r['reporting_interpretation']}`"
        )
    lines.append("")
    lines.append("## Reporting decision table")
    for _, r in decision.iterrows():
        lines.append(
            f"- `{r['context_source']}` / `{r['model']}`: "
            f"test_AP={r['test_ap']}, "
            f"AP_delta={r['primary_test_ap_delta_vs_internal']}, "
            f"p={r['primary_test_ap_p']}, "
            f"q={r['primary_test_ap_bh_fdr_q']}, "
            f"Bonferroni={r['primary_test_ap_bonferroni_p']}, "
            f"level=`{r['thesis_reporting_level']}`, "
            f"best_topK_delta={r['best_topk_winner_delta_vs_internal']}"
        )
    lines.append("")
    lines.append("## Top-K exploratory notes")
    if topk_summary.empty:
        lines.append("No top-K summary available.")
    else:
        notable = topk_summary[
            (topk_summary["winner_count_delta_vs_internal"] > 0)
            & (topk_summary["model"] != "internal")
        ].copy()
        if notable.empty:
            lines.append("No source improved top-K winner count versus internal in the tested fractions.")
        else:
            for _, r in notable.iterrows():
                lines.append(
                    f"- `{r['model']}` top {float(r['top_fraction'])*100:.0f}%: "
                    f"winners={int(r['winners_in_top_k'])}/{int(r['total_winners'])}, "
                    f"internal={int(r['internal_winners_in_top_k'])}/{int(r['total_winners'])}, "
                    f"delta={int(r['winner_count_delta_vs_internal'])}, "
                    f"capture={r['capture_rate']}"
                )
    lines.append("")
    lines.append("## Reporting conclusion")
    lines.append(f"`{reporting_conclusion}`")
    lines.append("")
    lines.append("## Thesis-safe wording")
    lines.append("See `05_thesis_safe_context_wording.md`.")
    lines.append("")
    lines.append("## Output files")
    for _, path in summary["outputs"].items():
        lines.append(f"- `{Path(path).name}`")

    md = OUT / "thesis_context_multiplicity_reporting_summary.md"
    md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(OUT / "thesis_context_multiplicity_reporting_summary.json"),
        "summary_md": str(md),
        "reporting_conclusion": reporting_conclusion,
    }, indent=2, ensure_ascii=False))
    print()
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
