from __future__ import annotations

import json
import math
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(r"E:\Projects\Final Project\memecoin_trader")
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

OUT = ROOT / "data" / "audits" / f"thesis_context_rq5_handoff_package_{STAMP}"
TABLES = OUT / "tables"
SOURCE_SUMMARIES = OUT / "source_summaries"

OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)
SOURCE_SUMMARIES.mkdir(parents=True, exist_ok=True)

PATHS = {
    "data_quality_root": ROOT / "data" / "audits" / "thesis_context_data_availability_quality_audit_20260810_201451",
    "identity_bridge_root": ROOT / "data" / "audits" / "thesis_context_identity_bridge_audit_20260810_202816",
    "canonical_bridge_root": ROOT / "data" / "audits" / "thesis_context_canonical_bridge_feasibility_v2_20260810_203604",
    "time_alignment_root": ROOT / "data" / "audits" / "thesis_context_time_alignment_diagnostic_20260810_212216",
    "dataset_root": ROOT / "data" / "audits" / "thesis_context_event_level_dataset_build_audit_20260810_212600",
    "incremental_root": ROOT / "data" / "audits" / "thesis_context_incremental_predictive_value_audit_20260810_213311",
    "ablation_root": ROOT / "data" / "audits" / "thesis_context_source_ablation_audit_20260810_213753",
    "multiplicity_root": ROOT / "data" / "audits" / "thesis_context_multiplicity_reporting_audit_20260810_215209",
}

FILES = {
    "dataset_summary_json": PATHS["dataset_root"] / "thesis_context_event_level_dataset_build_summary.json",
    "dataset_summary_md": PATHS["dataset_root"] / "thesis_context_event_level_dataset_build_summary.md",
    "dataset_csv": PATHS["dataset_root"] / "03_event_level_context_rebuild_dataset.csv",

    "incremental_summary_json": PATHS["incremental_root"] / "thesis_context_incremental_predictive_value_summary.json",
    "incremental_summary_md": PATHS["incremental_root"] / "thesis_context_incremental_predictive_value_summary.md",

    "ablation_summary_json": PATHS["ablation_root"] / "thesis_context_source_ablation_summary.json",
    "ablation_summary_md": PATHS["ablation_root"] / "thesis_context_source_ablation_summary.md",
    "ablation_labels_csv": PATHS["ablation_root"] / "00_label_distribution.csv",
    "ablation_features_csv": PATHS["ablation_root"] / "01_feature_inventory.csv",
    "ablation_metrics_csv": PATHS["ablation_root"] / "02_ablation_metrics.csv",
    "ablation_topk_csv": PATHS["ablation_root"] / "03_ablation_topk_winner_capture.csv",
    "ablation_bootstrap_csv": PATHS["ablation_root"] / "04_ablation_bootstrap_deltas_vs_internal.csv",

    "multiplicity_summary_json": PATHS["multiplicity_root"] / "thesis_context_multiplicity_reporting_summary.json",
    "multiplicity_summary_md": PATHS["multiplicity_root"] / "thesis_context_multiplicity_reporting_summary.md",
    "primary_ap_corrections_csv": PATHS["multiplicity_root"] / "00_primary_test_average_precision_multiplicity.csv",
    "test_all_metrics_corrections_csv": PATHS["multiplicity_root"] / "01_secondary_test_all_metrics_multiplicity.csv",
    "topk_delta_csv": PATHS["multiplicity_root"] / "03_test_topk_delta_vs_internal.csv",
    "decision_table_csv": PATHS["multiplicity_root"] / "04_context_source_reporting_decision_table.csv",
    "safe_wording_md": PATHS["multiplicity_root"] / "05_thesis_safe_context_wording.md",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def clean_value(x: Any) -> Any:
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if pd.isna(x):
        return None
    return x


def one(df: pd.DataFrame, **conds) -> dict[str, Any]:
    x = df.copy()
    for k, v in conds.items():
        x = x[x[k] == v]
    if x.empty:
        return {}
    return {k: clean_value(v) for k, v in x.iloc[0].to_dict().items()}


def copy_if_exists(src: Path, dst_dir: Path) -> str:
    if not src.exists():
        return "MISSING"
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return str(dst)


def df_to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, r in df.iterrows():
        rows.append({k: clean_value(v) for k, v in r.to_dict().items()})
    return rows


# Load core files.
dataset_summary = read_json(FILES["dataset_summary_json"])
incremental_summary = read_json(FILES["incremental_summary_json"])
ablation_summary = read_json(FILES["ablation_summary_json"])
multiplicity_summary = read_json(FILES["multiplicity_summary_json"])

labels = pd.read_csv(FILES["ablation_labels_csv"])
metrics = pd.read_csv(FILES["ablation_metrics_csv"])
topk = pd.read_csv(FILES["ablation_topk_csv"])
boot = pd.read_csv(FILES["ablation_bootstrap_csv"])
primary = pd.read_csv(FILES["primary_ap_corrections_csv"])
decision = pd.read_csv(FILES["decision_table_csv"])
topk_delta = pd.read_csv(FILES["topk_delta_csv"])

# Extract key rows.
test_internal = one(metrics, split="test", model="internal")
test_raw = one(metrics, split="test", model="internal_plus_raw_payload")
test_pool = one(metrics, split="test", model="internal_plus_pool_flow")
test_gemini = one(metrics, split="test", model="internal_plus_gemini")
test_raw_pool = one(metrics, split="test", model="internal_plus_raw_pool")
test_pool_gemini = one(metrics, split="test", model="internal_plus_pool_gemini")
test_all = one(metrics, split="test", model="internal_plus_all_context")

primary_raw = one(primary, challenger_model="internal_plus_raw_payload")
primary_pool = one(primary, challenger_model="internal_plus_pool_flow")
primary_gemini = one(primary, challenger_model="internal_plus_gemini")
primary_raw_pool = one(primary, challenger_model="internal_plus_raw_pool")
primary_pool_gemini = one(primary, challenger_model="internal_plus_pool_gemini")
primary_all = one(primary, challenger_model="internal_plus_all_context")

decision_raw = one(decision, model="internal_plus_raw_payload")
decision_pool = one(decision, model="internal_plus_pool_flow")
decision_gemini = one(decision, model="internal_plus_gemini")
decision_raw_pool = one(decision, model="internal_plus_raw_pool")
decision_pool_gemini = one(decision, model="internal_plus_pool_gemini")
decision_all = one(decision, model="internal_plus_all_context")

dataset_info = dataset_summary.get("dataset_summary", {})

key_numbers = {
    "package_created_at": STAMP,
    "purpose": "Thesis Context / RQ5 robustness handoff package",
    "safety": {
        "new_training": False,
        "new_backtest": False,
        "trader_db_mutation": False,
        "wallet_connected": False,
        "live_trading": False,
        "new_llm_calls": False,
        "trade_authority": False,
        "package_only": True,
    },
    "source_roots": {k: str(v) for k, v in PATHS.items()},
    "dataset": {
        "candidate_event_rows": dataset_info.get("candidate_event_rows"),
        "label_available_rows": dataset_info.get("label_available_rows"),
        "label_available_rate": dataset_info.get("label_available_rate"),
        "unique_canonical_coins": dataset_info.get("unique_canonical_coins"),
        "chronological_split_counts": dataset_info.get("chronological_split_counts"),
        "label_x2_sl_4h_counts": dataset_info.get("label_x2_sl_4h_counts"),
        "context_coverage": {
            k: v for k, v in dataset_info.items()
            if "coverage" in str(k)
        },
    },
    "label_distribution": df_to_records(labels),
    "test_metrics": {
        "internal": test_internal,
        "internal_plus_raw_payload": test_raw,
        "internal_plus_pool_flow": test_pool,
        "internal_plus_gemini": test_gemini,
        "internal_plus_raw_pool": test_raw_pool,
        "internal_plus_pool_gemini": test_pool_gemini,
        "internal_plus_all_context": test_all,
    },
    "primary_test_average_precision_multiplicity": {
        "family": "test split Average Precision deltas vs internal across 6 context-source ablations",
        "internal_plus_raw_payload": primary_raw,
        "internal_plus_pool_flow": primary_pool,
        "internal_plus_gemini": primary_gemini,
        "internal_plus_raw_pool": primary_raw_pool,
        "internal_plus_pool_gemini": primary_pool_gemini,
        "internal_plus_all_context": primary_all,
    },
    "reporting_decisions": {
        "raw_provider_activity_provenance": decision_raw,
        "pool_flow_proxy": decision_pool,
        "gemini_llm_records": decision_gemini,
        "raw_provider_plus_pool_flow": decision_raw_pool,
        "pool_flow_plus_gemini": decision_pool_gemini,
        "all_context_combined": decision_all,
    },
    "topk_exploratory_positive_notes": df_to_records(
        topk_delta[
            (topk_delta["model"] != "internal")
            & (topk_delta["winner_count_delta_vs_internal"] > 0)
        ]
    ),
    "final_reporting_conclusion": multiplicity_summary.get("reporting_conclusion"),
    "thesis_safe_claim": (
        "Context was not validated as a uniformly beneficial predictive layer. "
        "Raw-provider activity/provenance features showed FDR10-level exploratory improvement "
        "in test-set Average Precision, while all-context and Gemini-containing combinations were noisy or degrading."
    ),
    "critical_limitation": (
        "There is a temporal-frequency mismatch between fast meme-coin trading signals, which can update many times per minute, "
        "and slower external/semantic context, which may update daily, irregularly, or only around discrete events."
    ),
}

# Write JSON.
key_json_path = OUT / "01_context_rq5_key_numbers.json"
key_json_path.write_text(json.dumps(key_numbers, indent=2, ensure_ascii=False), encoding="utf-8")

# Write flat CSV.
flat_rows = []

def add(section: str, metric: str, value: Any, note: str = "") -> None:
    flat_rows.append({
        "section": section,
        "metric": metric,
        "value": clean_value(value),
        "note": note,
    })

add("dataset", "candidate_event_rows", dataset_info.get("candidate_event_rows"))
add("dataset", "label_available_rows", dataset_info.get("label_available_rows"))
add("dataset", "label_available_rate", dataset_info.get("label_available_rate"))
add("dataset", "unique_canonical_coins", dataset_info.get("unique_canonical_coins"))
add("dataset", "FLAT_count", dataset_info.get("label_x2_sl_4h_counts", {}).get("FLAT"))
add("dataset", "LOSER_count", dataset_info.get("label_x2_sl_4h_counts", {}).get("LOSER"))
add("dataset", "WINNER_count", dataset_info.get("label_x2_sl_4h_counts", {}).get("WINNER"))

for _, r in labels.iterrows():
    split = r["split"]
    add(f"labels_{split}", "rows", r["rows"])
    add(f"labels_{split}", "winner", r["winner"])
    add(f"labels_{split}", "winner_rate", r["winner_rate"])
    add(f"labels_{split}", "flat", r["flat"])
    add(f"labels_{split}", "loser", r["loser"])

for model_name, row in key_numbers["test_metrics"].items():
    add(f"test_metrics_{model_name}", "average_precision", row.get("average_precision"))
    add(f"test_metrics_{model_name}", "roc_auc", row.get("roc_auc"))
    add(f"test_metrics_{model_name}", "brier", row.get("brier"))
    add(f"test_metrics_{model_name}", "log_loss", row.get("log_loss"))

for model_name, row in key_numbers["primary_test_average_precision_multiplicity"].items():
    if model_name == "family":
        continue
    add(f"primary_AP_{model_name}", "delta", row.get("delta"))
    add(f"primary_AP_{model_name}", "p_two_sided", row.get("p_two_sided"))
    add(f"primary_AP_{model_name}", "BH_FDR_q", row.get("bh_fdr_q"))
    add(f"primary_AP_{model_name}", "Bonferroni_p", row.get("bonferroni_p"))
    add(f"primary_AP_{model_name}", "interpretation", row.get("reporting_interpretation"))

for source_name, row in key_numbers["reporting_decisions"].items():
    add(f"decision_{source_name}", "thesis_reporting_level", row.get("thesis_reporting_level"))
    add(f"decision_{source_name}", "test_AP", row.get("test_ap"))
    add(f"decision_{source_name}", "AP_delta_vs_internal", row.get("primary_test_ap_delta_vs_internal"))

flat_csv_path = OUT / "02_context_rq5_key_numbers_flat.csv"
pd.DataFrame(flat_rows).to_csv(flat_csv_path, index=False, encoding="utf-8-sig")

# Copy selected tables/summaries.
manifest_rows = []
for name, src in FILES.items():
    if name == "dataset_csv":
        # Do not copy the large dataset into this summary package. Reference it only.
        manifest_rows.append({
            "item": name,
            "source_path": str(src),
            "copied_to_package": "",
            "status": "REFERENCED_ONLY_LARGE_DATASET",
            "role": "Large event-level dataset; referenced but not copied into summary package",
        })
        continue

    target_dir = TABLES if src.suffix.lower() == ".csv" else SOURCE_SUMMARIES
    copied = copy_if_exists(src, target_dir)
    manifest_rows.append({
        "item": name,
        "source_path": str(src),
        "copied_to_package": copied if copied != "MISSING" else "",
        "status": "OK" if copied != "MISSING" else "MISSING",
        "role": "Source audit output copied for handoff",
    })

manifest_path = OUT / "05_manifest.csv"
pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False, encoding="utf-8-sig")

# Write concise README and main-thread update.
readme = f"""# Thesis Context / RQ5 Handoff Package

Created: `{STAMP}`

## Purpose

This package summarizes the final Context / RQ5 robustness results for thesis reporting and main-thread handoff.

It contains only reporting artifacts and copied audit summaries. It does not run new training, backtesting, paper/demo positions, wallet operations, live trading, LLM calls, or database mutation.

## Core result

Context was **not** validated as a uniformly beneficial predictive layer.

The combined context block degraded test Average Precision, while a narrow raw-provider activity/provenance feature block showed a multiplicity-aware exploratory signal.

## Event-level dataset

- Candidate event rows: `{dataset_info.get("candidate_event_rows")}`
- Label-available rows: `{dataset_info.get("label_available_rows")}`
- Label-available rate: `{dataset_info.get("label_available_rate")}`
- Unique canonical coins: `{dataset_info.get("unique_canonical_coins")}`
- Label counts: `{dataset_info.get("label_x2_sl_4h_counts")}`

## Rare-event warning

WINNER is extremely sparse. Accuracy is not an appropriate primary metric.

The thesis should emphasize Average Precision / PR-AUC, top-K winner capture, and lift.

## Main test-set AP results

- Internal-only AP: `{test_internal.get("average_precision")}`
- Internal + raw_payload AP: `{test_raw.get("average_precision")}`
- Internal + all_context AP: `{test_all.get("average_precision")}`
- Internal + gemini AP: `{test_gemini.get("average_precision")}`
- Internal + pool_gemini AP: `{test_pool_gemini.get("average_precision")}`

## Multiplicity-aware raw_payload result

- AP delta vs internal: `{primary_raw.get("delta")}`
- p: `{primary_raw.get("p_two_sided")}`
- BH-FDR q: `{primary_raw.get("bh_fdr_q")}`
- Bonferroni p: `{primary_raw.get("bonferroni_p")}`
- Interpretation: `{primary_raw.get("reporting_interpretation")}`
- Reporting level: `{decision_raw.get("thesis_reporting_level")}`

## Negative/noisy context findings

- all_context AP delta: `{primary_all.get("delta")}`, q: `{primary_all.get("bh_fdr_q")}`, interpretation: `{primary_all.get("reporting_interpretation")}`
- pool_gemini AP delta: `{primary_pool_gemini.get("delta")}`, q: `{primary_pool_gemini.get("bh_fdr_q")}`, interpretation: `{primary_pool_gemini.get("reporting_interpretation")}`
- gemini alone did not provide stable predictive lift.

## Top-K exploratory note

Pool-flow did not improve AP significantly, but in the test top 5% ranking it improved winner capture from 5/7 to 6/7. This should be reported only as exploratory ranking behavior.

## Required limitation

External/context data has a temporal-frequency mismatch with meme-coin trading.

Market and signal events can update many times per minute, while semantic/external context may update daily, irregularly, or only around discrete events. This weakens short-horizon causal attribution and supports treating context primarily as provenance, auditability, risk explanation, and regime evidence.

## Files in this package

- `01_context_rq5_key_numbers.json`
- `02_context_rq5_key_numbers_flat.csv`
- `03_main_thread_update_context_rq5.txt`
- `04_thesis_safe_context_wording.md`
- `05_manifest.csv`
- `tables/`
- `source_summaries/`
"""

(OUT / "00_README_CONTEXT_RQ5_HANDOFF.md").write_text(readme, encoding="utf-8")

main_thread_update = f"""THESIS CONTEXT / RQ5 ROBUSTNESS — FILE HANDOFF PACKAGE

This package summarizes the completed Context/RQ5 robustness work.

Package root:
{OUT}

Key source audit roots:
- Dataset build:
  {PATHS["dataset_root"]}
- Incremental context audit:
  {PATHS["incremental_root"]}
- Source ablation:
  {PATHS["ablation_root"]}
- Multiplicity/reporting audit:
  {PATHS["multiplicity_root"]}

Core dataset:
- candidate_event_rows: {dataset_info.get("candidate_event_rows")}
- label_available_rows: {dataset_info.get("label_available_rows")}
- label_available_rate: {dataset_info.get("label_available_rate")}
- unique_canonical_coins: {dataset_info.get("unique_canonical_coins")}
- label counts: {dataset_info.get("label_x2_sl_4h_counts")}

Rare-event warning:
WINNER is extremely sparse. The full dataset had 94 winners, and the test split had 7 winners. Accuracy must not be used as the primary metric.

Main test AP:
- internal-only: {test_internal.get("average_precision")}
- internal + raw_payload: {test_raw.get("average_precision")}
- internal + all_context: {test_all.get("average_precision")}
- internal + gemini: {test_gemini.get("average_precision")}
- internal + pool_gemini: {test_pool_gemini.get("average_precision")}

Raw-provider activity/provenance result:
- AP delta vs internal: {primary_raw.get("delta")}
- p: {primary_raw.get("p_two_sided")}
- BH-FDR q: {primary_raw.get("bh_fdr_q")}
- Bonferroni p: {primary_raw.get("bonferroni_p")}
- interpretation: {primary_raw.get("reporting_interpretation")}
- reporting level: {decision_raw.get("thesis_reporting_level")}

Negative/noisy context:
- all_context degraded AP:
  delta={primary_all.get("delta")}
  q={primary_all.get("bh_fdr_q")}
  interpretation={primary_all.get("reporting_interpretation")}
- pool_gemini degraded AP:
  delta={primary_pool_gemini.get("delta")}
  q={primary_pool_gemini.get("bh_fdr_q")}
  interpretation={primary_pool_gemini.get("reporting_interpretation")}

Thesis-safe conclusion:
Context was not validated as a uniformly beneficial predictive layer. Selected high-coverage raw-provider activity/provenance context showed FDR10-level exploratory improvement in test-set Average Precision, while broader context combinations, especially Gemini/LLM-derived or all-context combinations, were noisy or degrading.

Required limitation:
There is a temporal-frequency mismatch between fast meme-coin trading signals and slower external/semantic context. Context should therefore be framed primarily as provenance, auditability, risk explanation, and regime evidence, with only limited exploratory evidence for incremental prediction from selected provider-activity features.

No AE stages should be reopened. This is thesis reporting / robustness work only.
"""

(OUT / "03_main_thread_update_context_rq5.txt").write_text(main_thread_update, encoding="utf-8")

safe_wording_src = FILES["safe_wording_md"]
safe_wording_dst = OUT / "04_thesis_safe_context_wording.md"
if safe_wording_src.exists():
    shutil.copy2(safe_wording_src, safe_wording_dst)
else:
    safe_wording_dst.write_text(
        "Safe wording source file was missing. See README and main-thread update.",
        encoding="utf-8",
    )

# Create zip.
zip_path = OUT.with_suffix(".zip")
with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
    for p in OUT.rglob("*"):
        z.write(p, p.relative_to(OUT.parent))

print(json.dumps({
    "status": "OK",
    "package_root": str(OUT),
    "zip": str(zip_path),
    "readme": str(OUT / "00_README_CONTEXT_RQ5_HANDOFF.md"),
    "key_numbers_json": str(key_json_path),
    "key_numbers_csv": str(flat_csv_path),
    "main_thread_update": str(OUT / "03_main_thread_update_context_rq5.txt"),
    "safe_wording": str(safe_wording_dst),
    "manifest": str(manifest_path),
}, indent=2, ensure_ascii=False))

print()
print("Created files:")
for p in [
    OUT / "00_README_CONTEXT_RQ5_HANDOFF.md",
    key_json_path,
    flat_csv_path,
    OUT / "03_main_thread_update_context_rq5.txt",
    safe_wording_dst,
    manifest_path,
    zip_path,
]:
    print("-", p)
