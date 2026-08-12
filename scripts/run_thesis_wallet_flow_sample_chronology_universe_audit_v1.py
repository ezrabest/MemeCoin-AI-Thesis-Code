from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("THESIS_ROOT", r"E:\Projects\Final Project\memecoin_trader"))
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

DATASET_CSV = Path(os.environ.get(
    "THESIS_CONTEXT_DATASET_CSV",
    r"E:\Projects\Final Project\memecoin_trader\data\audits\thesis_context_event_level_dataset_build_audit_20260810_212600\03_event_level_context_rebuild_dataset.csv"
))

WALLET_AUDIT_ROOT = Path(os.environ.get(
    "THESIS_WALLET_FLOW_AUDIT_ROOT",
    r"E:\Projects\Final Project\memecoin_trader\data\audits\thesis_wallet_flow_coverage_expansion_audit_20260810_222809"
))

WALLET_3B_ROOT = Path(os.environ.get(
    "THESIS_WALLET_FLOW_3B_ROOT",
    r"E:\Projects\Final Project\memecoin_trader\data\audits\thesis_wallet_flow_label_association_directionality_audit_20260810_225031"
))

OUT = ROOT / "data" / "audits" / f"thesis_wallet_flow_sample_chronology_universe_audit_{STAMP}"
OUT.mkdir(parents=True, exist_ok=True)

DB = ROOT / "data" / "trader.db"

FILES = {
    "event_level_dataset": DATASET_CSV,
    "wallet_candidate_cases": WALLET_AUDIT_ROOT / "00_wallet_flow_candidate_cases.csv",
    "wallet_case_features": WALLET_AUDIT_ROOT / "05_wallet_flow_case_features.csv",
    "wallet_coverage_by_label": WALLET_AUDIT_ROOT / "05_wallet_flow_coverage_by_label.csv",
    "wallet_coverage_by_split": WALLET_AUDIT_ROOT / "06_wallet_flow_coverage_by_split.csv",
    "wallet_3b_summary": WALLET_3B_ROOT / "thesis_wallet_flow_label_association_directionality_summary.json",
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


def norm(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"nan", "none", "null", "na"}:
        return ""
    return s


def looks_like_solana_address(x: Any) -> bool:
    s = norm(x)
    if not (32 <= len(s) <= 60):
        return False
    bad = set("0OIl+/=")
    if any(ch in bad for ch in s):
        return False
    return True


def require(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def connect_ro() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{DB.as_posix()}?mode=ro", uri=True)


def read_table_cols(con: sqlite3.Connection, table: str, wanted: list[str]) -> pd.DataFrame:
    cols = [r[1] for r in con.execute(f'PRAGMA table_info("{table}")').fetchall()]
    use = [c for c in wanted if c in cols]
    if not use:
        return pd.DataFrame()
    sql = ", ".join(f'"{c}"' for c in use)
    return pd.read_sql_query(f'SELECT {sql} FROM "{table}"', con)


def add_event_key(df: pd.DataFrame, time_col: str = "candidate_event_time_utc") -> pd.DataFrame:
    out = df.copy()

    if "canonical_coin_id" not in out.columns:
        raise ValueError("Missing canonical_coin_id")
    if time_col not in out.columns:
        raise ValueError(f"Missing {time_col}")

    out["canonical_coin_id_num"] = pd.to_numeric(out["canonical_coin_id"], errors="coerce").astype("Int64")
    out["_event_time_dt"] = pd.to_datetime(out[time_col], errors="coerce", utc=True)

    # Use nanosecond timestamp to avoid formatting mismatch.
    valid_time = out["_event_time_dt"].notna()
    out["_event_time_ns"] = pd.Series([pd.NA] * len(out), dtype="Int64")
    out.loc[valid_time, "_event_time_ns"] = out.loc[valid_time, "_event_time_dt"].astype("int64").astype("Int64")

    out["event_key"] = (
        out["canonical_coin_id_num"].astype(str)
        + "|"
        + out["_event_time_ns"].astype(str)
    )

    out.loc[
        out["canonical_coin_id_num"].isna() | out["_event_time_dt"].isna(),
        "event_key"
    ] = ""

    return out


def count_by_label_split(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["scope", "chronological_split", "label_x2_sl_4h", "rows"])

    g = (
        df.groupby(["chronological_split", "label_x2_sl_4h"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    g.insert(0, "scope", scope)
    return g.sort_values(["scope", "chronological_split", "label_x2_sl_4h"])


def count_by_label(df: pd.DataFrame, scope: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["scope", "label_x2_sl_4h", "rows"])

    g = (
        df.groupby(["label_x2_sl_4h"], dropna=False)
        .size()
        .reset_index(name="rows")
    )
    g.insert(0, "scope", scope)
    return g.sort_values(["scope", "label_x2_sl_4h"])


def winner_split_counts(df: pd.DataFrame) -> dict[str, int]:
    if df.empty:
        return {}
    w = df[df["label_x2_sl_4h"].eq("WINNER")]
    return {str(k): int(v) for k, v in w["chronological_split"].value_counts(dropna=False).to_dict().items()}


def scope_summary(df: pd.DataFrame, scope: str) -> dict[str, Any]:
    return {
        "scope": scope,
        "rows": int(len(df)),
        "unique_event_keys": int(df["event_key"].nunique()) if "event_key" in df.columns else None,
        "unique_canonical_coins": int(df["canonical_coin_id_num"].nunique()) if "canonical_coin_id_num" in df.columns else None,
        "unique_pair_addresses": int(df["pair_address"].nunique()) if "pair_address" in df.columns else None,
        "label_counts": {str(k): int(v) for k, v in df["label_x2_sl_4h"].value_counts(dropna=False).to_dict().items()} if "label_x2_sl_4h" in df.columns else {},
        "split_counts": {str(k): int(v) for k, v in df["chronological_split"].value_counts(dropna=False).to_dict().items()} if "chronological_split" in df.columns else {},
        "winner_split_counts": winner_split_counts(df) if "label_x2_sl_4h" in df.columns else {},
    }


def main() -> None:
    for name, path in FILES.items():
        if name == "wallet_3b_summary":
            continue
        require(path, name)

    events = pd.read_csv(FILES["event_level_dataset"], low_memory=False)
    selected = pd.read_csv(FILES["wallet_candidate_cases"], low_memory=False)
    features = pd.read_csv(FILES["wallet_case_features"], low_memory=False)
    coverage_by_label = pd.read_csv(FILES["wallet_coverage_by_label"], low_memory=False)
    coverage_by_split = pd.read_csv(FILES["wallet_coverage_by_split"], low_memory=False)
    summary_3b = read_json(FILES["wallet_3b_summary"])

    required_event_cols = ["canonical_coin_id", "candidate_event_time_utc", "label_x2_sl_4h", "chronological_split"]
    missing = [c for c in required_event_cols if c not in events.columns]
    if missing:
        raise ValueError(f"Event dataset missing required columns: {missing}")

    events = add_event_key(events)
    selected = add_event_key(selected)
    features = add_event_key(features)

    con = connect_ro()
    coins = read_table_cols(con, "coins", ["id", "symbol", "name", "chain", "pair_address"])
    con.close()

    if coins.empty:
        raise ValueError("Could not read coins table with required identity columns.")

    coins["canonical_coin_id_num"] = pd.to_numeric(coins["id"], errors="coerce").astype("Int64")
    coins["coin_chain_norm"] = coins.get("chain", "").astype(str).str.upper().str.strip()
    coins["pair_address"] = coins.get("pair_address", "").astype(str).str.strip()
    coins["symbol_coin"] = coins.get("symbol", "").astype(str)

    # Full event-level dataset merged to coins.
    events_m = events.merge(
        coins[["canonical_coin_id_num", "symbol_coin", "coin_chain_norm", "pair_address"]],
        on="canonical_coin_id_num",
        how="left",
    )

    events_m["is_solana_resolvable"] = (
        events_m["coin_chain_norm"].str.contains("SOL", na=False)
        & events_m["pair_address"].map(looks_like_solana_address)
    )

    solana_universe = events_m[events_m["is_solana_resolvable"]].copy()

    selected_keys = set(selected["event_key"].dropna().astype(str))
    feature_keys = set(features["event_key"].dropna().astype(str))

    events_m["selected_for_wallet_audit"] = events_m["event_key"].isin(selected_keys)
    events_m["has_wallet_case_features"] = events_m["event_key"].isin(feature_keys)

    solana_universe["selected_for_wallet_audit"] = solana_universe["event_key"].isin(selected_keys)
    solana_universe["has_wallet_case_features"] = solana_universe["event_key"].isin(feature_keys)

    selected_m = selected.copy()
    if "pair_address" not in selected_m.columns:
        selected_m = selected_m.merge(
            coins[["canonical_coin_id_num", "coin_chain_norm", "pair_address"]],
            on="canonical_coin_id_num",
            how="left",
        )

    feature_m = features.copy()
    if "pair_address" not in feature_m.columns:
        feature_m = feature_m.merge(
            coins[["canonical_coin_id_num", "coin_chain_norm", "pair_address"]],
            on="canonical_coin_id_num",
            how="left",
        )

    # Write inventories.
    file_inventory = []
    for name, path in FILES.items():
        file_inventory.append({
            "input_name": name,
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else None,
        })
    pd.DataFrame(file_inventory).to_csv(OUT / "00_input_file_inventory.csv", index=False, encoding="utf-8-sig")

    scope_summaries = [
        scope_summary(events_m, "full_event_level_dataset"),
        scope_summary(solana_universe, "full_solana_resolvable_universe"),
        scope_summary(selected_m, "selected_wallet_audit_cases"),
        scope_summary(feature_m, "wallet_case_feature_rows"),
    ]
    pd.DataFrame(scope_summaries).to_json(
        OUT / "01_scope_summaries.json",
        orient="records",
        indent=2,
        force_ascii=False,
    )

    # Counts by label/split.
    label_split_all = pd.concat([
        count_by_label_split(events_m, "full_event_level_dataset"),
        count_by_label_split(solana_universe, "full_solana_resolvable_universe"),
        count_by_label_split(selected_m, "selected_wallet_audit_cases"),
        count_by_label_split(feature_m, "wallet_case_feature_rows"),
    ], ignore_index=True)
    label_split_all.to_csv(OUT / "02_counts_by_scope_label_split.csv", index=False, encoding="utf-8-sig")

    label_all = pd.concat([
        count_by_label(events_m, "full_event_level_dataset"),
        count_by_label(solana_universe, "full_solana_resolvable_universe"),
        count_by_label(selected_m, "selected_wallet_audit_cases"),
        count_by_label(feature_m, "wallet_case_feature_rows"),
    ], ignore_index=True)
    label_all.to_csv(OUT / "03_counts_by_scope_label.csv", index=False, encoding="utf-8-sig")

    # Selection coverage by label and split within Solana universe.
    coverage_rows = []
    for (split, label), g in solana_universe.groupby(["chronological_split", "label_x2_sl_4h"], dropna=False):
        n = len(g)
        selected_n = int(g["selected_for_wallet_audit"].sum())
        feature_n = int(g["has_wallet_case_features"].sum())
        coverage_rows.append({
            "chronological_split": split,
            "label_x2_sl_4h": label,
            "solana_universe_rows": n,
            "selected_for_wallet_audit": selected_n,
            "has_wallet_case_features": feature_n,
            "selected_rate": selected_n / n if n else None,
            "feature_rate": feature_n / n if n else None,
        })

    selection_coverage = pd.DataFrame(coverage_rows).sort_values(["chronological_split", "label_x2_sl_4h"])
    selection_coverage.to_csv(OUT / "04_solana_universe_selection_coverage_by_label_split.csv", index=False, encoding="utf-8-sig")

    # Missing winners analysis.
    sol_winners = solana_universe[solana_universe["label_x2_sl_4h"].eq("WINNER")].copy()
    missing_winners = sol_winners[~sol_winners["selected_for_wallet_audit"]].copy()
    selected_winners = sol_winners[sol_winners["selected_for_wallet_audit"]].copy()

    missing_winner_cols = [
        c for c in [
            "event_key",
            "canonical_coin_id",
            "canonical_coin_id_num",
            "candidate_event_time_utc",
            "chronological_split",
            "label_x2_sl_4h",
            "symbol_coin",
            "coin_chain_norm",
            "pair_address",
        ]
        if c in missing_winners.columns
    ]
    missing_winners[missing_winner_cols].head(500).to_csv(
        OUT / "05_missing_solana_winners_not_selected_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected_winners[missing_winner_cols].head(500).to_csv(
        OUT / "06_selected_solana_winners_sample.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Key counts.
    full_winner_split = winner_split_counts(events_m)
    sol_winner_split = winner_split_counts(solana_universe)
    selected_winner_split = winner_split_counts(selected_m)
    feature_winner_split = winner_split_counts(feature_m)

    sol_val_test_winners = int(
        sol_winners["chronological_split"].isin(["validation", "test"]).sum()
    )
    selected_val_test_winners = int(
        selected_winners["chronological_split"].isin(["validation", "test"]).sum()
    )
    missing_val_test_winners = int(
        missing_winners["chronological_split"].isin(["validation", "test"]).sum()
    )

    if len(solana_universe) == 0:
        chronology_classification = "NO_SOLANA_RESOLVABLE_UNIVERSE"
    elif sol_val_test_winners == 0:
        chronology_classification = "NO_VALIDATION_TEST_WINNERS_IN_SOLANA_RESOLVABLE_UNIVERSE"
    elif selected_val_test_winners == 0 and missing_val_test_winners > 0:
        chronology_classification = "SELECTION_CAP_OR_SELECTION_POLICY_EXCLUDED_VALIDATION_TEST_WINNERS"
    elif selected_val_test_winners > 0:
        chronology_classification = "VALIDATION_TEST_WINNERS_INCLUDED_IN_WALLET_SAMPLE"
    else:
        chronology_classification = "CHRONOLOGY_STATUS_AMBIGUOUS"

    selected_all_solana_winners = int(len(missing_winners) == 0)

    if chronology_classification == "NO_VALIDATION_TEST_WINNERS_IN_SOLANA_RESOLVABLE_UNIVERSE":
        next_step_classification = "NO_MORE_HELIUS_CAN_FIX_CHRONOLOGICAL_WINNER_GENERALIZATION_FOR_THIS_DATASET"
    elif chronology_classification == "SELECTION_CAP_OR_SELECTION_POLICY_EXCLUDED_VALIDATION_TEST_WINNERS":
        next_step_classification = "TARGETED_HELIUS_ON_MISSING_VALIDATION_TEST_WINNERS_REQUIRED_BEFORE_FINAL_CLAIM"
    elif chronology_classification == "VALIDATION_TEST_WINNERS_INCLUDED_IN_WALLET_SAMPLE":
        next_step_classification = "CHRONOLOGICAL_GENERALIZATION_CAN_BE_EVALUATED_FROM_EXISTING_SAMPLE"
    else:
        next_step_classification = "INSPECT_UNIVERSE_AND_SELECTION"

    # Bring in 3B conclusions if present.
    prior_3b = {
        "final_scientific_conclusion": summary_3b.get("final_scientific_conclusion"),
        "wallet_signal_conclusion": summary_3b.get("wallet_signal_conclusion"),
        "large_threshold_conclusion": summary_3b.get("large_threshold_conclusion"),
        "chronological_conclusion": summary_3b.get("chronological_conclusion"),
    }

    decision_rows = [{
        "question": "Are validation/test WINNER cases available in the full Solana-resolvable wallet universe?",
        "answer": "YES" if sol_val_test_winners > 0 else "NO",
        "evidence": f"Solana-resolvable WINNER split counts: {sol_winner_split}",
        "classification": chronology_classification,
    }, {
        "question": "Did the selected Helius sample include all Solana-resolvable WINNER cases?",
        "answer": "YES" if selected_all_solana_winners else "NO",
        "evidence": f"selected_solana_winners={len(selected_winners)}; missing_solana_winners_not_selected={len(missing_winners)}",
        "classification": "ALL_SOLANA_WINNERS_SELECTED" if selected_all_solana_winners else "SOME_SOLANA_WINNERS_NOT_SELECTED",
    }, {
        "question": "Can additional Helius calls fix the lack of validation/test winner generalization?",
        "answer": "NO" if next_step_classification == "NO_MORE_HELIUS_CAN_FIX_CHRONOLOGICAL_WINNER_GENERALIZATION_FOR_THIS_DATASET" else "POSSIBLY",
        "evidence": f"solana_val_test_winners={sol_val_test_winners}; selected_val_test_winners={selected_val_test_winners}; missing_val_test_winners={missing_val_test_winners}",
        "classification": next_step_classification,
    }, {
        "question": "Should the negative wallet-flow winner result be reported as chronologically validated?",
        "answer": "NO",
        "evidence": f"feature winner split counts: {feature_winner_split}",
        "classification": "RETROSPECTIVE_SELECTED_SAMPLE_ASSOCIATION_ONLY",
    }]
    decision_df = pd.DataFrame(decision_rows)
    decision_df.to_csv(OUT / "07_chronology_selection_decision_table.csv", index=False, encoding="utf-8-sig")

    # Summary.
    summary = {
        "classification": "THESIS_WALLET_FLOW_SAMPLE_CHRONOLOGY_UNIVERSE_AUDIT_COMPLETED",
        "root": str(ROOT),
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
        },
        "inputs": {k: str(v) for k, v in FILES.items()},
        "scope_summaries": scope_summaries,
        "winner_split_counts": {
            "full_event_level_dataset": full_winner_split,
            "full_solana_resolvable_universe": sol_winner_split,
            "selected_wallet_audit_cases": selected_winner_split,
            "wallet_case_feature_rows": feature_winner_split,
        },
        "selection_checks": {
            "solana_winner_rows": int(len(sol_winners)),
            "selected_solana_winner_rows": int(len(selected_winners)),
            "missing_solana_winner_rows_not_selected": int(len(missing_winners)),
            "solana_validation_test_winner_rows": sol_val_test_winners,
            "selected_validation_test_winner_rows": selected_val_test_winners,
            "missing_validation_test_winner_rows": missing_val_test_winners,
        },
        "prior_3b_conclusions": prior_3b,
        "chronology_classification": chronology_classification,
        "next_step_classification": next_step_classification,
        "final_scientific_conclusion": (
            "WALLET_FLOW_NEGATIVE_WINNER_ASSOCIATION_IS_RETROSPECTIVE_SELECTED_SAMPLE_ONLY_DUE_TO_NO_SOLANA_VALIDATION_TEST_WINNERS"
            if chronology_classification == "NO_VALIDATION_TEST_WINNERS_IN_SOLANA_RESOLVABLE_UNIVERSE"
            else
            "WALLET_FLOW_SAMPLE_SELECTION_MAY_BE_INCOMPLETE_FOR_CHRONOLOGICAL_WINNER_CLAIMS"
            if chronology_classification == "SELECTION_CAP_OR_SELECTION_POLICY_EXCLUDED_VALIDATION_TEST_WINNERS"
            else
            "WALLET_FLOW_CHRONOLOGY_STATUS_REQUIRES_REVIEW"
        ),
        "outputs": {
            "input_inventory": str(OUT / "00_input_file_inventory.csv"),
            "scope_summaries": str(OUT / "01_scope_summaries.json"),
            "counts_by_scope_label_split": str(OUT / "02_counts_by_scope_label_split.csv"),
            "counts_by_scope_label": str(OUT / "03_counts_by_scope_label.csv"),
            "selection_coverage_by_label_split": str(OUT / "04_solana_universe_selection_coverage_by_label_split.csv"),
            "missing_solana_winners_not_selected": str(OUT / "05_missing_solana_winners_not_selected_sample.csv"),
            "selected_solana_winners": str(OUT / "06_selected_solana_winners_sample.csv"),
            "decision_table": str(OUT / "07_chronology_selection_decision_table.csv"),
            "summary_json": str(OUT / "thesis_wallet_flow_sample_chronology_universe_summary.json"),
            "summary_md": str(OUT / "thesis_wallet_flow_sample_chronology_universe_summary.md"),
        },
    }

    summary_json = OUT / "thesis_wallet_flow_sample_chronology_universe_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("# Audit 3C — Wallet-flow Sample / Chronology / Universe")
    lines.append("")
    lines.append(f"Output root: `{OUT}`")
    lines.append("")
    lines.append("## Safety")
    lines.append("- Read-only post-processing audit")
    lines.append("- No Helius queries")
    lines.append("- No model training")
    lines.append("- No backtest")
    lines.append("- No trader.db mutation")
    lines.append("- No wallet connection")
    lines.append("- No live trading")
    lines.append("- No new LLM calls")
    lines.append("- No trade authority")
    lines.append("")
    lines.append("## Scope summaries")
    for s in scope_summaries:
        lines.append(f"### {s['scope']}")
        lines.append(f"- rows: {s['rows']}")
        lines.append(f"- unique event keys: {s['unique_event_keys']}")
        lines.append(f"- unique canonical coins: {s['unique_canonical_coins']}")
        lines.append(f"- unique pair addresses: {s['unique_pair_addresses']}")
        lines.append(f"- label counts: {s['label_counts']}")
        lines.append(f"- split counts: {s['split_counts']}")
        lines.append(f"- winner split counts: {s['winner_split_counts']}")
        lines.append("")
    lines.append("## Winner split counts")
    lines.append(f"- full event-level dataset: {full_winner_split}")
    lines.append(f"- full Solana-resolvable universe: {sol_winner_split}")
    lines.append(f"- selected wallet audit cases: {selected_winner_split}")
    lines.append(f"- wallet case feature rows: {feature_winner_split}")
    lines.append("")
    lines.append("## Selection checks")
    lines.append(f"- Solana WINNER rows: {len(sol_winners)}")
    lines.append(f"- selected Solana WINNER rows: {len(selected_winners)}")
    lines.append(f"- missing Solana WINNER rows not selected: {len(missing_winners)}")
    lines.append(f"- Solana validation/test WINNER rows: {sol_val_test_winners}")
    lines.append(f"- selected validation/test WINNER rows: {selected_val_test_winners}")
    lines.append(f"- missing validation/test WINNER rows: {missing_val_test_winners}")
    lines.append("")
    lines.append("## Prior 3B conclusions")
    for k, v in prior_3b.items():
        lines.append(f"- `{k}`: `{v}`")
    lines.append("")
    lines.append("## Decision table")
    for _, r in decision_df.iterrows():
        lines.append(f"- {r['question']} `{r['classification']}` — {r['evidence']}")
    lines.append("")
    lines.append("## Chronology classification")
    lines.append(f"`{chronology_classification}`")
    lines.append("")
    lines.append("## Next-step classification")
    lines.append(f"`{next_step_classification}`")
    lines.append("")
    lines.append("## Final scientific conclusion")
    lines.append(f"`{summary['final_scientific_conclusion']}`")
    lines.append("")
    lines.append("## Output files")
    for _, path in summary["outputs"].items():
        lines.append(f"- `{Path(path).name}`")

    summary_md = OUT / "thesis_wallet_flow_sample_chronology_universe_summary.md"
    summary_md.write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({
        "status": "OK",
        "output_root": str(OUT),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "chronology_classification": chronology_classification,
        "next_step_classification": next_step_classification,
        "final_scientific_conclusion": summary["final_scientific_conclusion"],
    }, indent=2, ensure_ascii=False))

    print()
    print(summary_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
