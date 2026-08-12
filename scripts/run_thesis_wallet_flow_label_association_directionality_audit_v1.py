from __future__ import annotations

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("THESIS_ROOT", r"E:\Projects\Final Project\memecoin_trader"))
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

WALLET_AUDIT_ROOT = Path(os.environ.get(
    "THESIS_WALLET_FLOW_AUDIT_ROOT",
    r"E:\Projects\Final Project\memecoin_trader\data\audits\thesis_wallet_flow_coverage_expansion_audit_20260810_222809"
))

OUT = ROOT / "data" / "audits" / f"thesis_wallet_flow_label_association_directionality_audit_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

ALPHA = float(os.environ.get("THESIS_ALPHA", "0.05"))


FILES = {
    "summary_json": WALLET_AUDIT_ROOT / "thesis_wallet_flow_coverage_expansion_summary.json",
    "summary_md": WALLET_AUDIT_ROOT / "thesis_wallet_flow_coverage_expansion_summary.md",
    "candidate_cases": WALLET_AUDIT_ROOT / "00_wallet_flow_candidate_cases.csv",
    "merged_windows": WALLET_AUDIT_ROOT / "01_helius_merged_query_windows.csv",
    "fetch_summary": WALLET_AUDIT_ROOT / "02_helius_fetch_summary.csv",
    "case_features": WALLET_AUDIT_ROOT / "05_wallet_flow_case_features.csv",
    "coverage_by_label": WALLET_AUDIT_ROOT / "05_wallet_flow_coverage_by_label.csv",
    "coverage_by_split": WALLET_AUDIT_ROOT / "06_wallet_flow_coverage_by_split.csv",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(x: Any) -> Any:
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        if math.isnan(float(x)) or math.isinf(float(x)):
            return None
        return float(x)
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if pd.isna(x):
        return None
    return x


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def require_files() -> None:
    missing = [str(p) for p in FILES.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required input files:\n" + "\n".join(missing))


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def log_hypergeom_prob(a: int, row1: int, col1: int, n: int) -> float:
    # 2x2 table probability for fixed margins.
    # a = top-left, row1 = a+b, col1 = a+c, n = total
    return (
        math.lgamma(col1 + 1)
        - math.lgamma(a + 1)
        - math.lgamma(col1 - a + 1)
        + math.lgamma(n - col1 + 1)
        - math.lgamma(row1 - a + 1)
        - math.lgamma((n - col1) - (row1 - a) + 1)
        - (
            math.lgamma(n + 1)
            - math.lgamma(row1 + 1)
            - math.lgamma(n - row1 + 1)
        )
    )


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """
    Two-sided Fisher exact test using fixed margins and probability <= observed probability.
    Table:
      [[a, b],
       [c, d]]
    """
    row1 = a + b
    row2 = c + d
    col1 = a + c
    n = row1 + row2

    if n == 0:
        return float("nan")

    lo = max(0, col1 - row2)
    hi = min(row1, col1)

    obs_logp = log_hypergeom_prob(a, row1, col1, n)
    probs = []
    for x in range(lo, hi + 1):
        lp = log_hypergeom_prob(x, row1, col1, n)
        if lp <= obs_logp + 1e-12:
            probs.append(math.exp(lp))

    return min(1.0, float(sum(probs)))


def odds_ratio_haldane(a: int, b: int, c: int, d: int) -> float:
    return ((a + 0.5) * (d + 0.5)) / ((b + 0.5) * (c + 0.5))


def risk_ratio_haldane(a: int, b: int, c: int, d: int) -> float:
    r1 = (a + 0.5) / (a + b + 1.0)
    r2 = (c + 0.5) / (c + d + 1.0)
    return r1 / r2 if r2 != 0 else float("nan")


def binary_test(df: pd.DataFrame, target_col: str, feature_col: str, target_label: str, feature_label: str) -> dict[str, Any]:
    pos = df[target_col].astype(bool)
    feat = df[feature_col].astype(bool)

    a = int((pos & feat).sum())
    b = int((pos & ~feat).sum())
    c = int((~pos & feat).sum())
    d = int((~pos & ~feat).sum())

    p = fisher_exact_two_sided(a, b, c, d)
    orr = odds_ratio_haldane(a, b, c, d)
    rr = risk_ratio_haldane(a, b, c, d)

    pos_rate = a / (a + b) if (a + b) else None
    other_rate = c / (c + d) if (c + d) else None

    if pos_rate is None or other_rate is None:
        direction = "NOT_COMPUTABLE"
    elif pos_rate > other_rate:
        direction = f"{feature_label}_ENRICHED_IN_{target_label}"
    elif pos_rate < other_rate:
        direction = f"{feature_label}_DEPLETED_IN_{target_label}"
    else:
        direction = "NO_RATE_DIFFERENCE"

    if p < ALPHA and pos_rate is not None and other_rate is not None:
        if pos_rate > other_rate:
            result = "SIGNIFICANT_POSITIVE_ASSOCIATION"
        elif pos_rate < other_rate:
            result = "SIGNIFICANT_NEGATIVE_ASSOCIATION"
        else:
            result = "SIGNIFICANT_BUT_NO_RATE_DIRECTION"
    else:
        result = "NO_SIGNIFICANT_ASSOCIATION"

    return {
        "target": target_label,
        "feature": feature_label,
        "a_target_and_feature": a,
        "b_target_no_feature": b,
        "c_non_target_and_feature": c,
        "d_non_target_no_feature": d,
        "target_feature_rate": pos_rate,
        "non_target_feature_rate": other_rate,
        "odds_ratio_haldane": orr,
        "risk_ratio_haldane": rr,
        "fisher_exact_two_sided_p": p,
        "direction": direction,
        "result": result,
    }


def summarize_numeric(df: pd.DataFrame, group_col: str, numeric_cols: list[str]) -> pd.DataFrame:
    rows = []
    for group, g in df.groupby(group_col, dropna=False):
        for col in numeric_cols:
            x = pd.to_numeric(g[col], errors="coerce").dropna()
            if x.empty:
                rows.append({
                    group_col: group,
                    "metric": col,
                    "n": 0,
                    "mean": None,
                    "median": None,
                    "p75": None,
                    "p90": None,
                    "p95": None,
                    "max": None,
                })
                continue

            rows.append({
                group_col: group,
                "metric": col,
                "n": int(len(x)),
                "mean": float(x.mean()),
                "median": float(x.median()),
                "p75": float(x.quantile(0.75)),
                "p90": float(x.quantile(0.90)),
                "p95": float(x.quantile(0.95)),
                "max": float(x.max()),
            })
    return pd.DataFrame(rows)


def main() -> None:
    require_files()

    source_summary = read_json(FILES["summary_json"])

    features = pd.read_csv(FILES["case_features"], low_memory=False)
    candidate_cases = pd.read_csv(FILES["candidate_cases"], low_memory=False)
    fetch_summary = pd.read_csv(FILES["fetch_summary"], low_memory=False)
    coverage_by_label = pd.read_csv(FILES["coverage_by_label"], low_memory=False)
    coverage_by_split = pd.read_csv(FILES["coverage_by_split"], low_memory=False)

    required_cols = [
        "candidate_event_id",
        "label_x2_sl_4h",
        "chronological_split",
        "pre_tx_count",
        "pre_transfer_count",
        "pre_external_wallet_count",
        "large_pre_wallet_link_count",
        "crude_query_address_direction",
        "real_wallet_level_evidence",
        "large_wallet_level_evidence",
        "evidence_class",
    ]
    missing_cols = [c for c in required_cols if c not in features.columns]
    if missing_cols:
        raise ValueError(f"case_features missing columns: {missing_cols}")

    df = features.copy()

    # Normalize booleans safely.
    for col in ["real_wallet_level_evidence", "large_wallet_level_evidence"]:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
        else:
            df[col] = df[col].astype(bool)

    df["pre_tx_count"] = pd.to_numeric(df["pre_tx_count"], errors="coerce").fillna(0)
    df["pre_transfer_count"] = pd.to_numeric(df["pre_transfer_count"], errors="coerce").fillna(0)
    df["pre_external_wallet_count"] = pd.to_numeric(df["pre_external_wallet_count"], errors="coerce").fillna(0)
    df["large_pre_wallet_link_count"] = pd.to_numeric(df["large_pre_wallet_link_count"], errors="coerce").fillna(0)
    df["pre_token_transfer_count"] = pd.to_numeric(df.get("pre_token_transfer_count", 0), errors="coerce").fillna(0)
    df["pre_native_transfer_count"] = pd.to_numeric(df.get("pre_native_transfer_count", 0), errors="coerce").fillna(0)

    df["is_winner"] = df["label_x2_sl_4h"].eq("WINNER")
    df["is_loser"] = df["label_x2_sl_4h"].eq("LOSER")
    df["is_flat"] = df["label_x2_sl_4h"].eq("FLAT")
    df["is_nonflat"] = ~df["is_flat"]

    df["has_pre_tx"] = df["pre_tx_count"] > 0
    df["has_pre_transfer"] = df["pre_transfer_count"] > 0
    df["has_external_wallet"] = df["pre_external_wallet_count"] > 0
    df["has_large_wallet"] = df["large_wallet_level_evidence"]
    df["has_real_wallet"] = df["real_wallet_level_evidence"]
    df["has_no_helius_pre_event_activity"] = df["evidence_class"].eq("NO_HELIUS_PRE_EVENT_ACTIVITY")
    df["direction_net_out"] = df["crude_query_address_direction"].eq("NET_OUT_FROM_QUERY_ADDRESS")
    df["direction_balanced"] = df["crude_query_address_direction"].eq("BALANCED_OR_MIXED")
    df["direction_no_direct"] = df["crude_query_address_direction"].eq("NO_DIRECT_QUERY_ADDRESS_DIRECTION")

    # Inventory.
    inventory_rows = []
    for name, path in FILES.items():
        inventory_rows.append({
            "input_name": name,
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        })
    pd.DataFrame(inventory_rows).to_csv(OUT / "00_input_file_inventory.csv", index=False, encoding="utf-8-sig")

    # Contingency tables.
    label_evidence = (
        df.groupby(["label_x2_sl_4h", "evidence_class"], dropna=False)
        .size()
        .reset_index(name="cases")
        .sort_values(["label_x2_sl_4h", "evidence_class"])
    )
    label_evidence.to_csv(OUT / "01_label_evidence_contingency.csv", index=False, encoding="utf-8-sig")

    label_split = (
        df.groupby(["label_x2_sl_4h", "chronological_split"], dropna=False)
        .size()
        .reset_index(name="cases")
        .sort_values(["label_x2_sl_4h", "chronological_split"])
    )
    label_split.to_csv(OUT / "02_label_split_counts.csv", index=False, encoding="utf-8-sig")

    direction_by_label = (
        df.groupby(["label_x2_sl_4h", "crude_query_address_direction"], dropna=False)
        .size()
        .reset_index(name="cases")
        .sort_values(["label_x2_sl_4h", "crude_query_address_direction"])
    )
    direction_by_label.to_csv(OUT / "03_direction_by_label.csv", index=False, encoding="utf-8-sig")

    direction_by_evidence = (
        df.groupby(["evidence_class", "crude_query_address_direction"], dropna=False)
        .size()
        .reset_index(name="cases")
        .sort_values(["evidence_class", "crude_query_address_direction"])
    )
    direction_by_evidence.to_csv(OUT / "04_direction_by_evidence_class.csv", index=False, encoding="utf-8-sig")

    # Rate summary by label.
    rate_rows = []
    for label, g in df.groupby("label_x2_sl_4h", dropna=False):
        n = len(g)
        for feature_col, feature_label in [
            ("has_large_wallet", "large_wallet_level_evidence"),
            ("has_real_wallet", "real_wallet_level_evidence"),
            ("has_pre_tx", "pre_tx_activity"),
            ("has_pre_transfer", "pre_transfer_activity"),
            ("has_external_wallet", "external_wallet_actor"),
            ("has_no_helius_pre_event_activity", "no_helius_pre_event_activity"),
            ("direction_net_out", "net_out_direction_proxy"),
            ("direction_balanced", "balanced_direction_proxy"),
            ("direction_no_direct", "no_direct_direction_proxy"),
        ]:
            k = int(g[feature_col].sum())
            lo, hi = wilson_ci(k, n)
            rate_rows.append({
                "label_x2_sl_4h": label,
                "feature": feature_label,
                "n": n,
                "count": k,
                "rate": k / n if n else None,
                "wilson95_low": lo,
                "wilson95_high": hi,
            })

    rate_summary = pd.DataFrame(rate_rows)
    rate_summary.to_csv(OUT / "05_rate_summary_by_label.csv", index=False, encoding="utf-8-sig")

    # Binary Fisher tests.
    tests = []
    for target_col, target_label in [
        ("is_winner", "WINNER"),
        ("is_loser", "LOSER"),
        ("is_flat", "FLAT"),
        ("is_nonflat", "NONFLAT"),
    ]:
        for feature_col, feature_label in [
            ("has_large_wallet", "LARGE_WALLET_LEVEL_EVIDENCE"),
            ("has_real_wallet", "REAL_WALLET_LEVEL_EVIDENCE"),
            ("has_pre_tx", "PRE_EVENT_TX_ACTIVITY"),
            ("has_external_wallet", "EXTERNAL_WALLET_ACTOR"),
            ("has_no_helius_pre_event_activity", "NO_HELIUS_PRE_EVENT_ACTIVITY"),
            ("direction_net_out", "NET_OUT_DIRECTION_PROXY"),
            ("direction_balanced", "BALANCED_OR_MIXED_DIRECTION_PROXY"),
        ]:
            tests.append(binary_test(df, target_col, feature_col, target_label, feature_label))

    tests_df = pd.DataFrame(tests)
    tests_df.to_csv(OUT / "06_binary_association_tests.csv", index=False, encoding="utf-8-sig")

    # Numeric summaries.
    numeric_cols = [
        "pre_tx_count",
        "post_tx_count",
        "pre_transfer_count",
        "pre_token_transfer_count",
        "pre_native_transfer_count",
        "pre_external_wallet_count",
        "large_pre_token_transfer_count",
        "large_pre_wallet_link_count",
        "pre_in_to_query_address_transfer_count",
        "pre_out_from_query_address_transfer_count",
    ]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    numeric_by_label = summarize_numeric(df, "label_x2_sl_4h", numeric_cols)
    numeric_by_label.to_csv(OUT / "07_numeric_feature_summary_by_label.csv", index=False, encoding="utf-8-sig")

    numeric_by_evidence = summarize_numeric(df, "evidence_class", numeric_cols)
    numeric_by_evidence.to_csv(OUT / "08_numeric_feature_summary_by_evidence_class.csv", index=False, encoding="utf-8-sig")

    # Large threshold sanity.
    total = len(df)
    real_n = int(df["has_real_wallet"].sum())
    large_n = int(df["has_large_wallet"].sum())
    real_and_large = int((df["has_real_wallet"] & df["has_large_wallet"]).sum())
    real_not_large = int((df["has_real_wallet"] & ~df["has_large_wallet"]).sum())
    large_not_real = int((df["has_large_wallet"] & ~df["has_real_wallet"]).sum())

    large_threshold_rows = [{
        "metric": "total_cases",
        "value": total,
        "interpretation": "",
    }, {
        "metric": "real_wallet_level_cases",
        "value": real_n,
        "interpretation": "",
    }, {
        "metric": "large_wallet_level_cases",
        "value": large_n,
        "interpretation": "",
    }, {
        "metric": "real_and_large_cases",
        "value": real_and_large,
        "interpretation": "",
    }, {
        "metric": "real_not_large_cases",
        "value": real_not_large,
        "interpretation": "",
    }, {
        "metric": "large_not_real_cases",
        "value": large_not_real,
        "interpretation": "",
    }, {
        "metric": "large_among_real_rate",
        "value": real_and_large / real_n if real_n else None,
        "interpretation": "If this is ~1.0, the large threshold does not distinguish among real wallet-level cases at case level.",
    }, {
        "metric": "large_equals_real_all_rows",
        "value": bool((df["has_real_wallet"] == df["has_large_wallet"]).all()),
        "interpretation": "True means large_wallet_level_evidence is not adding case-level discrimination beyond real_wallet_level_evidence.",
    }, {
        "metric": "large_pre_wallet_link_count_median_all",
        "value": float(df["large_pre_wallet_link_count"].median()),
        "interpretation": "",
    }, {
        "metric": "large_pre_wallet_link_count_median_real_wallet_cases",
        "value": float(df.loc[df["has_real_wallet"], "large_pre_wallet_link_count"].median()) if real_n else None,
        "interpretation": "",
    }, {
        "metric": "large_pre_wallet_link_count_max",
        "value": float(df["large_pre_wallet_link_count"].max()),
        "interpretation": "",
    }]
    large_threshold_df = pd.DataFrame(large_threshold_rows)
    large_threshold_df.to_csv(OUT / "09_large_threshold_sanity.csv", index=False, encoding="utf-8-sig")

    # Split limitation.
    split_label_pivot = pd.crosstab(df["chronological_split"], df["label_x2_sl_4h"])
    split_label_pivot.to_csv(OUT / "10_split_label_pivot.csv", encoding="utf-8-sig")

    train_winners = int(((df["chronological_split"] == "train") & df["is_winner"]).sum())
    validation_winners = int(((df["chronological_split"] == "validation") & df["is_winner"]).sum())
    test_winners = int(((df["chronological_split"] == "test") & df["is_winner"]).sum())

    # Key extracted rates.
    def rate(label: str, feature_col: str) -> tuple[int, int, float | None]:
        g = df[df["label_x2_sl_4h"] == label]
        n = len(g)
        k = int(g[feature_col].sum())
        return k, n, k / n if n else None

    winner_large_k, winner_n, winner_large_rate = rate("WINNER", "has_large_wallet")
    loser_large_k, loser_n, loser_large_rate = rate("LOSER", "has_large_wallet")
    flat_large_k, flat_n, flat_large_rate = rate("FLAT", "has_large_wallet")

    # Decide claim status.
    winner_large_test = tests_df[
        (tests_df["target"] == "WINNER")
        & (tests_df["feature"] == "LARGE_WALLET_LEVEL_EVIDENCE")
    ].iloc[0].to_dict()

    flat_large_test = tests_df[
        (tests_df["target"] == "FLAT")
        & (tests_df["feature"] == "LARGE_WALLET_LEVEL_EVIDENCE")
    ].iloc[0].to_dict()

    large_threshold_degenerate = bool((df["has_real_wallet"] == df["has_large_wallet"]).all())

    if winner_large_k == 0 and flat_large_rate is not None and flat_large_rate > 0.50:
        wallet_signal_conclusion = "NO_WINNER_WHALE_ENTRY_SIGNAL_OBSERVED_FLAT_CONCENTRATED"
    elif winner_large_rate is not None and loser_large_rate is not None and winner_large_rate > loser_large_rate:
        wallet_signal_conclusion = "EXPLORATORY_WINNER_ENRICHMENT_REQUIRES_VALIDATION"
    else:
        wallet_signal_conclusion = "NO_POSITIVE_WINNER_WALLET_SIGNAL_SUPPORTED"

    if validation_winners == 0 and test_winners == 0:
        chronological_conclusion = "NO_VALIDATION_OR_TEST_WINNER_GENERALIZATION_POSSIBLE_IN_SELECTED_SOLANA_WALLET_SAMPLE"
    elif test_winners == 0:
        chronological_conclusion = "NO_TEST_WINNER_GENERALIZATION_POSSIBLE_IN_SELECTED_SOLANA_WALLET_SAMPLE"
    else:
        chronological_conclusion = "TEST_WINNERS_AVAILABLE"

    if large_threshold_degenerate:
        large_threshold_conclusion = "CASE_LEVEL_LARGE_FLAG_DEGENERATES_TO_REAL_WALLET_FLAG"
    else:
        large_threshold_conclusion = "LARGE_FLAG_ADDS_CASE_LEVEL_DISCRIMINATION"

    decision_rows = [{
        "question": "Did wallet-flow expansion produce enough case-level coverage to analyze?",
        "answer": "YES",
        "evidence": f"{real_n}/{total} real-wallet-level cases; {large_n}/{total} large-wallet-level cases.",
        "classification": "TESTABLE_COVERAGE",
    }, {
        "question": "Does pre-event wallet evidence support a WINNER/whale-entry signal in this selected sample?",
        "answer": "NO",
        "evidence": f"WINNER large-wallet cases: {winner_large_k}/{winner_n}; FLAT large-wallet cases: {flat_large_k}/{flat_n}; LOSER large-wallet cases: {loser_large_k}/{loser_n}.",
        "classification": wallet_signal_conclusion,
    }, {
        "question": "Is the large-wallet flag discriminative beyond real-wallet evidence at case level?",
        "answer": "NO" if large_threshold_degenerate else "YES",
        "evidence": f"real_not_large={real_not_large}; large_not_real={large_not_real}; large_among_real={real_and_large}/{real_n}.",
        "classification": large_threshold_conclusion,
    }, {
        "question": "Can chronological validation/test winner generalization be assessed from this selected Solana wallet sample?",
        "answer": "NO" if (validation_winners == 0 or test_winners == 0) else "YES",
        "evidence": f"train_winners={train_winners}; validation_winners={validation_winners}; test_winners={test_winners}.",
        "classification": chronological_conclusion,
    }, {
        "question": "Should pool/pair address direction be treated as whale-entry/exit direction?",
        "answer": "NO",
        "evidence": "Direction is computed relative to query/pair address and remains a crude proxy unless transfer-level token/account semantics are validated.",
        "classification": "DIRECTIONALITY_PROXY_ONLY",
    }]
    decision_df = pd.DataFrame(decision_rows)
    decision_df.to_csv(OUT / "11_claim_decision_table.csv", index=False, encoding="utf-8-sig")

    # Summary JSON.
    summary = {
        "classification": "THESIS_WALLET_FLOW_LABEL_ASSOCIATION_DIRECTIONALITY_AUDIT_COMPLETED",
        "root": str(ROOT),
        "wallet_flow_audit_root": str(WALLET_AUDIT_ROOT),
        "output_root": str(OUT),
        "created_at": now_iso(),
        "safety": {
            "read_only_post_processing": True,
            "helius_queries": False,
            "new_model_training": False,
            "backtest_run": False,
            "trader_db_mutated": False,
            "wallet_connected": False,
            "live_trading_enabled": False,
            "new_llm_calls": False,
            "trade_authority": False,
            "raw_transaction_files_required": False,
        },
        "input_counts": {
            "case_feature_rows": int(len(df)),
            "candidate_case_rows": int(len(candidate_cases)),
            "fetch_windows": int(len(fetch_summary)),
            "source_real_wallet_level_cases": source_summary.get("coverage", {}).get("real_wallet_level_cases"),
            "source_large_wallet_level_cases": source_summary.get("coverage", {}).get("large_wallet_level_cases"),
        },
        "label_counts": {str(k): int(v) for k, v in df["label_x2_sl_4h"].value_counts(dropna=False).to_dict().items()},
        "split_winner_counts": {
            "train_winners": train_winners,
            "validation_winners": validation_winners,
            "test_winners": test_winners,
        },
        "key_rates": {
            "winner_large_wallet": {"count": winner_large_k, "n": winner_n, "rate": winner_large_rate},
            "loser_large_wallet": {"count": loser_large_k, "n": loser_n, "rate": loser_large_rate},
            "flat_large_wallet": {"count": flat_large_k, "n": flat_n, "rate": flat_large_rate},
        },
        "key_tests": {
            "winner_vs_nonwinner_large_wallet": {k: clean(v) for k, v in winner_large_test.items()},
            "flat_vs_nonflat_large_wallet": {k: clean(v) for k, v in flat_large_test.items()},
        },
        "large_threshold_conclusion": large_threshold_conclusion,
        "wallet_signal_conclusion": wallet_signal_conclusion,
        "chronological_conclusion": chronological_conclusion,
        "final_scientific_conclusion": (
            "WALLET_FLOW_COVERAGE_TESTABLE_BUT_NO_WINNER_WHALE_SIGNAL_AND_NO_CHRONOLOGICAL_WINNER_GENERALIZATION"
            if "NO" in chronological_conclusion and "NO_WINNER" in wallet_signal_conclusion
            else wallet_signal_conclusion
        ),
        "outputs": {
            "input_inventory": str(OUT / "00_input_file_inventory.csv"),
            "label_evidence_contingency": str(OUT / "01_label_evidence_contingency.csv"),
            "label_split_counts": str(OUT / "02_label_split_counts.csv"),
            "direction_by_label": str(OUT / "03_direction_by_label.csv"),
            "direction_by_evidence_class": str(OUT / "04_direction_by_evidence_class.csv"),
            "rate_summary_by_label": str(OUT / "05_rate_summary_by_label.csv"),
            "binary_association_tests": str(OUT / "06_binary_association_tests.csv"),
            "numeric_feature_summary_by_label": str(OUT / "07_numeric_feature_summary_by_label.csv"),
            "numeric_feature_summary_by_evidence_class": str(OUT / "08_numeric_feature_summary_by_evidence_class.csv"),
            "large_threshold_sanity": str(OUT / "09_large_threshold_sanity.csv"),
            "split_label_pivot": str(OUT / "10_split_label_pivot.csv"),
            "claim_decision_table": str(OUT / "11_claim_decision_table.csv"),
            "summary_json": str(OUT / "thesis_wallet_flow_label_association_directionality_summary.json"),
            "summary_md": str(OUT / "thesis_wallet_flow_label_association_directionality_summary.md"),
        },
    }

    summary_json = OUT / "thesis_wallet_flow_label_association_directionality_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Markdown summary.
    lines = []
    lines.append("# Audit 3B — Wallet-flow Label Association and Directionality")
    lines.append("")
    lines.append(f"Output root: `{OUT}`")
    lines.append(f"Input wallet-flow audit root: `{WALLET_AUDIT_ROOT}`")
    lines.append("")
    lines.append("## Safety")
    lines.append("- Read-only post-processing audit")
    lines.append("- No Helius queries")
    lines.append("- No raw transaction/transfer files required")
    lines.append("- No model training")
    lines.append("- No backtest")
    lines.append("- No trader.db mutation")
    lines.append("- No wallet connection")
    lines.append("- No live trading")
    lines.append("- No new LLM calls")
    lines.append("- No trade authority")
    lines.append("")
    lines.append("## Input counts")
    lines.append(f"- case feature rows: {len(df):,}")
    lines.append(f"- candidate case rows: {len(candidate_cases):,}")
    lines.append(f"- fetch windows: {len(fetch_summary):,}")
    lines.append("")
    lines.append("## Label distribution")
    for label, count in df["label_x2_sl_4h"].value_counts(dropna=False).to_dict().items():
        lines.append(f"- `{label}`: {count}")
    lines.append("")
    lines.append("## Large-wallet evidence by label")
    lines.append(f"- WINNER: {winner_large_k}/{winner_n} ({winner_large_rate})")
    lines.append(f"- LOSER: {loser_large_k}/{loser_n} ({loser_large_rate})")
    lines.append(f"- FLAT: {flat_large_k}/{flat_n} ({flat_large_rate})")
    lines.append("")
    lines.append("## Key Fisher exact tests")
    lines.append(
        "- WINNER vs non-WINNER, large-wallet evidence: "
        f"p={winner_large_test.get('fisher_exact_two_sided_p')}, "
        f"OR={winner_large_test.get('odds_ratio_haldane')}, "
        f"direction=`{winner_large_test.get('direction')}`, "
        f"result=`{winner_large_test.get('result')}`"
    )
    lines.append(
        "- FLAT vs non-FLAT, large-wallet evidence: "
        f"p={flat_large_test.get('fisher_exact_two_sided_p')}, "
        f"OR={flat_large_test.get('odds_ratio_haldane')}, "
        f"direction=`{flat_large_test.get('direction')}`, "
        f"result=`{flat_large_test.get('result')}`"
    )
    lines.append("")
    lines.append("## Directionality proxy by label")
    for _, r in direction_by_label.iterrows():
        lines.append(f"- `{r['label_x2_sl_4h']}` / `{r['crude_query_address_direction']}`: {int(r['cases'])}")
    lines.append("")
    lines.append("## Large-threshold sanity")
    lines.append(f"- real wallet-level cases: {real_n}/{total}")
    lines.append(f"- large wallet-level cases: {large_n}/{total}")
    lines.append(f"- real-and-large cases: {real_and_large}/{real_n}")
    lines.append(f"- real-not-large cases: {real_not_large}")
    lines.append(f"- large-not-real cases: {large_not_real}")
    lines.append(f"- conclusion: `{large_threshold_conclusion}`")
    lines.append("")
    lines.append("## Chronological split limitation")
    lines.append(f"- train winners: {train_winners}")
    lines.append(f"- validation winners: {validation_winners}")
    lines.append(f"- test winners: {test_winners}")
    lines.append(f"- conclusion: `{chronological_conclusion}`")
    lines.append("")
    lines.append("## Claim decision table")
    for _, r in decision_df.iterrows():
        lines.append(f"- {r['question']} `{r['classification']}` — {r['evidence']}")
    lines.append("")
    lines.append("## Final scientific conclusion")
    lines.append(f"`{summary['final_scientific_conclusion']}`")
    lines.append("")
    lines.append("## Output files")
    for _, path in summary["outputs"].items():
        lines.append(f"- `{Path(path).name}`")

    summary_md = OUT / "thesis_wallet_flow_label_association_directionality_summary.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "final_scientific_conclusion": summary["final_scientific_conclusion"],
        "wallet_signal_conclusion": wallet_signal_conclusion,
        "large_threshold_conclusion": large_threshold_conclusion,
        "chronological_conclusion": chronological_conclusion,
    }, indent=2, ensure_ascii=False))

    print()
    print(summary_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
