from __future__ import annotations

import csv
import json
import math
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(os.environ.get("THESIS_ROOT", r"E:\Projects\Final Project\memecoin_trader"))
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUT = ROOT / "data" / "audits" / f"thesis_blind_context_significance_audit_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

AUDITS_ROOT = ROOT / "data" / "audits"

BOOTSTRAP_N = int(os.environ.get("THESIS_BOOTSTRAP_N", "20000"))
RANDOM_SEED = int(os.environ.get("THESIS_RANDOM_SEED", "42"))
MAX_SCAN_FILES = int(os.environ.get("THESIS_MAX_SCAN_FILES", "5000"))

PREFERRED_ROW_COUNTS = {120, 90, 36, 45}

CASE_ID_CANDIDATES = [
    "case_id", "blind_case_id", "id", "row_id", "candidate_id", "source_case_id"
]

ACTUAL_CANDIDATES = [
    "actual_class", "ground_truth", "ground_truth_class", "true_class",
    "outcome_class", "actual_outcome", "label", "target_class", "actual_label"
]

INTERNAL_PRED_CANDIDATES = [
    "internal_primary_prediction", "internal_prediction", "internal_pred",
    "baseline_prediction", "baseline_pred", "model_only_prediction",
    "model_prediction", "internal_decision", "baseline_decision",
    "internal_primary_decision", "prediction_internal"
]

CONTEXT_PRED_CANDIDATES = [
    "context_primary_prediction", "context_prediction", "context_pred",
    "timestamped_context_prediction", "with_context_prediction",
    "external_context_prediction", "context_decision",
    "timestamped_context_decision", "prediction_context",
    "llm_context_prediction"
]

INTERNAL_CORRECT_CANDIDATES = [
    "internal_primary_correct", "internal_correct", "baseline_correct",
    "model_only_correct", "model_correct", "internal_is_correct"
]

CONTEXT_CORRECT_CANDIDATES = [
    "context_primary_correct", "context_correct", "timestamped_context_correct",
    "with_context_correct", "external_context_correct", "context_is_correct"
]


def norm_col(s: str) -> str:
    return s.strip().lower().replace(" ", "_").replace("-", "_")


def pick_col(cols: list[str], candidates: list[str]) -> str | None:
    m = {norm_col(c): c for c in cols}
    for cand in candidates:
        if norm_col(cand) in m:
            return m[norm_col(cand)]
    return None


def truthy(v: Any) -> bool | None:
    if pd.isna(v):
        return None
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, float)):
        if math.isnan(float(v)):
            return None
        return float(v) != 0.0
    x = str(v).strip().upper()
    if x in {"TRUE", "T", "YES", "Y", "1", "CORRECT"}:
        return True
    if x in {"FALSE", "F", "NO", "N", "0", "INCORRECT"}:
        return False
    return None


def normalize_label(v: Any) -> str | None:
    if pd.isna(v):
        return None
    x = str(v).strip().upper()
    if not x or x in {"NAN", "NONE", "NULL", "NA"}:
        return None

    # keep common thesis outcome labels
    aliases = {
        "WIN": "WINNER",
        "PROFIT": "WINNER",
        "PROFITABLE": "WINNER",
        "POSITIVE": "WINNER",
        "TP": "WINNER",

        "LOSS": "LOSER",
        "NEGATIVE": "LOSER",
        "UNPROFITABLE": "LOSER",
        "SL": "LOSER",

        "WATCH": "FLAT",
        "NO_TRADE": "FLAT",
        "HOLD": "FLAT",
        "NEUTRAL": "FLAT",
    }
    return aliases.get(x, x)


def mcnemar_exact_p(b: int, c: int) -> float | None:
    # b = internal correct, context wrong
    # c = internal wrong, context correct
    n = b + c
    if n == 0:
        return None
    k = min(b, c)
    lower = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2.0 * lower)


def bootstrap_ci(diffs: list[float], n_boot: int = BOOTSTRAP_N, seed: int = RANDOM_SEED) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    vals = []
    for _ in range(n_boot):
        s = 0.0
        for _j in range(n):
            s += diffs[rng.randrange(n)]
        vals.append(s / n)
    vals.sort()
    lo = vals[int(0.025 * n_boot)]
    hi = vals[int(0.975 * n_boot)]
    return lo, hi


def score_candidate_file(path: Path, rows: int, cols: list[str]) -> int:
    cols_norm = {norm_col(c) for c in cols}
    name = path.name.lower()
    score = 0

    if rows in PREFERRED_ROW_COUNTS:
        score += 50
    if rows == 120:
        score += 100

    if any(x in name for x in ["result", "prediction", "unblind", "accuracy", "comparison", "blind"]):
        score += 20
    if "context" in name:
        score += 20
    if "ground_truth" in name:
        score += 5

    if pick_col(cols, INTERNAL_CORRECT_CANDIDATES) and pick_col(cols, CONTEXT_CORRECT_CANDIDATES):
        score += 100
    if pick_col(cols, ACTUAL_CANDIDATES) and pick_col(cols, INTERNAL_PRED_CANDIDATES) and pick_col(cols, CONTEXT_PRED_CANDIDATES):
        score += 100

    if "do_not_upload" in name.lower():
        score -= 30

    return score


def scan_csv_files() -> list[dict[str, Any]]:
    rows = []
    if not AUDITS_ROOT.exists():
        return rows

    files = list(AUDITS_ROOT.rglob("*.csv"))
    files = [p for p in files if "blind" in str(p).lower() or "context" in str(p).lower() or "prediction" in str(p).lower()]
    files = files[:MAX_SCAN_FILES]

    for p in files:
        try:
            df = pd.read_csv(p, nrows=2000, low_memory=False)
            full_rows = None
            try:
                full_rows = sum(1 for _ in open(p, "r", encoding="utf-8", errors="ignore")) - 1
            except Exception:
                full_rows = len(df)

            cols = list(df.columns)
            rows.append({
                "path": str(p),
                "name": p.name,
                "rows": int(full_rows),
                "sample_rows_read": int(len(df)),
                "columns_count": len(cols),
                "columns": "|".join(cols),
                "case_id_col": pick_col(cols, CASE_ID_CANDIDATES),
                "actual_col": pick_col(cols, ACTUAL_CANDIDATES),
                "internal_pred_col": pick_col(cols, INTERNAL_PRED_CANDIDATES),
                "context_pred_col": pick_col(cols, CONTEXT_PRED_CANDIDATES),
                "internal_correct_col": pick_col(cols, INTERNAL_CORRECT_CANDIDATES),
                "context_correct_col": pick_col(cols, CONTEXT_CORRECT_CANDIDATES),
                "candidate_score": score_candidate_file(p, int(full_rows), cols),
                "error": "",
            })
        except Exception as exc:
            rows.append({
                "path": str(p),
                "name": p.name,
                "rows": None,
                "sample_rows_read": 0,
                "columns_count": 0,
                "columns": "",
                "case_id_col": None,
                "actual_col": None,
                "internal_pred_col": None,
                "context_pred_col": None,
                "internal_correct_col": None,
                "context_correct_col": None,
                "candidate_score": -999,
                "error": repr(exc),
            })

    return sorted(rows, key=lambda r: r["candidate_score"], reverse=True)


def compute_from_file(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    df = pd.read_csv(path, low_memory=False)

    internal_correct_col = meta.get("internal_correct_col")
    context_correct_col = meta.get("context_correct_col")

    if internal_correct_col and context_correct_col:
        internal_correct = df[internal_correct_col].map(truthy)
        context_correct = df[context_correct_col].map(truthy)
        method = "correctness_columns"
    else:
        actual_col = meta.get("actual_col")
        internal_pred_col = meta.get("internal_pred_col")
        context_pred_col = meta.get("context_pred_col")

        if not (actual_col and internal_pred_col and context_pred_col):
            raise ValueError("Selected file does not contain paired correctness columns or actual/internal/context prediction columns.")

        actual = df[actual_col].map(normalize_label)
        internal_pred = df[internal_pred_col].map(normalize_label)
        context_pred = df[context_pred_col].map(normalize_label)

        internal_correct = actual.eq(internal_pred)
        context_correct = actual.eq(context_pred)
        internal_correct = internal_correct.where(actual.notna() & internal_pred.notna(), None)
        context_correct = context_correct.where(actual.notna() & context_pred.notna(), None)
        method = "actual_vs_prediction_columns"

    work = pd.DataFrame({
        "internal_correct": internal_correct,
        "context_correct": context_correct,
    }).dropna()

    work["internal_correct"] = work["internal_correct"].astype(bool)
    work["context_correct"] = work["context_correct"].astype(bool)

    n = int(len(work))
    if n == 0:
        raise ValueError("No valid paired rows after dropping missing correctness values.")

    internal_correct_n = int(work["internal_correct"].sum())
    context_correct_n = int(work["context_correct"].sum())

    both_correct = int((work["internal_correct"] & work["context_correct"]).sum())
    both_wrong = int((~work["internal_correct"] & ~work["context_correct"]).sum())
    internal_only = int((work["internal_correct"] & ~work["context_correct"]).sum())
    context_only = int((~work["internal_correct"] & work["context_correct"]).sum())

    internal_acc = internal_correct_n / n
    context_acc = context_correct_n / n
    diff = context_acc - internal_acc

    diffs = (work["context_correct"].astype(int) - work["internal_correct"].astype(int)).astype(float).tolist()
    ci_lo, ci_hi = bootstrap_ci(diffs)

    p = mcnemar_exact_p(internal_only, context_only)

    if p is None:
        conclusion = "NO_DISCORDANT_PAIRS_NO_SIGNIFICANCE_TEST"
    elif p <= 0.05 and diff > 0:
        conclusion = "CONTEXT_IMPROVEMENT_STATISTICALLY_SIGNIFICANT"
    elif p <= 0.05 and diff < 0:
        conclusion = "CONTEXT_DEGRADATION_STATISTICALLY_SIGNIFICANT"
    else:
        conclusion = "NO_STATISTICALLY_SIGNIFICANT_CONTEXT_GAIN"

    return {
        "method": method,
        "source_path": str(path),
        "rows_used": n,
        "internal_correct": internal_correct_n,
        "context_correct": context_correct_n,
        "internal_accuracy": internal_acc,
        "context_accuracy": context_acc,
        "accuracy_delta_context_minus_internal": diff,
        "accuracy_delta_percentage_points": diff * 100,
        "bootstrap_ci95_delta_low": ci_lo,
        "bootstrap_ci95_delta_high": ci_hi,
        "bootstrap_ci95_delta_low_percentage_points": ci_lo * 100,
        "bootstrap_ci95_delta_high_percentage_points": ci_hi * 100,
        "paired_counts": {
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "internal_correct_context_wrong": internal_only,
            "internal_wrong_context_correct": context_only,
        },
        "mcnemar_exact_p_two_sided": p,
        "conclusion": conclusion,
    }


def main() -> None:
    inventory = scan_csv_files()
    inv_df = pd.DataFrame(inventory)
    inv_path = OUT / "00_blind_context_candidate_file_inventory.csv"
    inv_df.to_csv(inv_path, index=False, encoding="utf-8-sig")

    selected = None
    for row in inventory:
        if row["candidate_score"] >= 100:
            selected = row
            break

    result = None
    error = None

    if selected:
        try:
            result = compute_from_file(Path(selected["path"]), selected)
        except Exception as exc:
            error = repr(exc)

    summary = {
        "classification": "THESIS_BLIND_CONTEXT_SIGNIFICANCE_AUDIT_COMPLETED",
        "root": str(ROOT),
        "output_root": str(OUT),
        "safety": {
            "read_only_audit": True,
            "training_run": False,
            "backtest_run": False,
            "trader_db_mutated": False,
            "wallet_connected": False,
            "live_trading_enabled": False,
            "new_llm_calls": False,
            "ground_truth_used_only_for_posthoc_significance": True,
        },
        "candidate_files_scanned": len(inventory),
        "selected_file": selected,
        "result": result,
        "error": error,
    }

    with open(OUT / "thesis_blind_context_significance_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    lines = []
    lines.append("# Thesis Blind Context Significance Audit")
    lines.append("")
    lines.append(f"Output root: `{OUT}`")
    lines.append("")
    lines.append("## Safety")
    lines.append("- Read-only audit")
    lines.append("- No training")
    lines.append("- No backtest")
    lines.append("- No trader.db mutation")
    lines.append("- No wallet")
    lines.append("- No live trading")
    lines.append("- No new LLM calls")
    lines.append("")

    lines.append("## Candidate discovery")
    lines.append(f"- Candidate CSV files scanned: {len(inventory)}")
    lines.append(f"- Candidate inventory: `{inv_path.name}`")
    lines.append("")

    if selected:
        lines.append("## Selected paired file")
        lines.append(f"- `{selected['path']}`")
        lines.append(f"- Rows: {selected['rows']}")
        lines.append(f"- Candidate score: {selected['candidate_score']}")
        lines.append("")
    else:
        lines.append("## Selected paired file")
        lines.append("No valid paired internal/context prediction file was automatically identified.")
        lines.append("Use `00_blind_context_candidate_file_inventory.csv` to select the correct file manually.")
        lines.append("")

    if result:
        lines.append("## Result")
        lines.append(f"- Rows used: {result['rows_used']}")
        lines.append(f"- Internal-only accuracy: {result['internal_accuracy']*100:.2f}%")
        lines.append(f"- Context-aware accuracy: {result['context_accuracy']*100:.2f}%")
        lines.append(f"- Delta: {result['accuracy_delta_percentage_points']:+.2f} percentage points")
        lines.append(
            f"- Bootstrap 95% CI for delta: "
            f"{result['bootstrap_ci95_delta_low_percentage_points']:+.2f} to "
            f"{result['bootstrap_ci95_delta_high_percentage_points']:+.2f} percentage points"
        )
        lines.append(f"- McNemar exact p-value: {result['mcnemar_exact_p_two_sided']}")
        lines.append(f"- Conclusion: `{result['conclusion']}`")
        lines.append("")
        pc = result["paired_counts"]
        lines.append("## Paired counts")
        lines.append(f"- Both correct: {pc['both_correct']}")
        lines.append(f"- Both wrong: {pc['both_wrong']}")
        lines.append(f"- Internal correct / context wrong: {pc['internal_correct_context_wrong']}")
        lines.append(f"- Internal wrong / context correct: {pc['internal_wrong_context_correct']}")
        lines.append("")

        if result["conclusion"] == "NO_STATISTICALLY_SIGNIFICANT_CONTEXT_GAIN":
            lines.append("## Thesis wording")
            lines.append(
                "The measured context-aware change was not statistically significant. "
                "The result supports the feasibility of evaluating timestamped context under a locked blind protocol, "
                "but it does not establish that external context improved predictive performance beyond the internal logic."
            )
        elif result["conclusion"] == "CONTEXT_IMPROVEMENT_STATISTICALLY_SIGNIFICANT":
            lines.append("## Thesis wording")
            lines.append(
                "The context-aware layer produced a statistically significant paired improvement in this blind audit. "
                "This should still be interpreted as decision-support evidence rather than trade authority or live-profitability proof."
            )
        elif result["conclusion"] == "CONTEXT_DEGRADATION_STATISTICALLY_SIGNIFICANT":
            lines.append("## Thesis wording")
            lines.append(
                "The context-aware layer significantly degraded paired accuracy in this blind audit. "
                "The thesis should describe context as auditable but not beneficial for prediction in this configuration."
            )
    else:
        lines.append("## Result")
        lines.append("No final significance result was computed automatically.")
        if error:
            lines.append(f"Error from selected file: `{error}`")
        lines.append("")
        lines.append("Next step: inspect the top rows in `00_blind_context_candidate_file_inventory.csv` and provide the correct paired result file.")

    md_path = OUT / "thesis_blind_context_significance_summary.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(OUT / "thesis_blind_context_significance_summary.json"),
        "summary_md": str(md_path),
        "inventory_csv": str(inv_path),
        "selected_file": selected["path"] if selected else None,
        "result_conclusion": result["conclusion"] if result else None,
    }, indent=2, ensure_ascii=False))

    print()
    print(md_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
