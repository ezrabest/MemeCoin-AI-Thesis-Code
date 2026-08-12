from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(".").resolve()
OUT_DIR = ROOT / "data" / "training" / "manual_verified_results" / "phase_b_model_cuts_v5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))

import phase_b_two_of_three_composition_v4 as v4  # noqa: E402


MODES = [
    {
        "mode": "PER_MODEL_CAP_ONLY",
        "model_pair_cap": True,
        "final_pair_cap": False,
        "truncate_to_reported": False,
        "description": "Apply pair_cap inside each model top-k selection; then take vote_count>=2; no final cap.",
    },
    {
        "mode": "NO_MODEL_CAP_POST_CONSENSUS_CAP",
        "model_pair_cap": False,
        "final_pair_cap": True,
        "truncate_to_reported": False,
        "description": "Take each model top-k without cap; then vote_count>=2; then apply pair_cap after consensus.",
    },
    {
        "mode": "NO_MODEL_CAP_POST_CONSENSUS_CAP_TRUNC_TO_REPORTED",
        "model_pair_cap": False,
        "final_pair_cap": True,
        "truncate_to_reported": True,
        "description": "Take each model top-k without cap; then vote_count>=2; then apply pair_cap and truncate to V2 selected count.",
    },
    {
        "mode": "BOTH_MODEL_AND_POST_CONSENSUS_CAP",
        "model_pair_cap": True,
        "final_pair_cap": True,
        "truncate_to_reported": False,
        "description": "Apply pair_cap inside each model and again after consensus.",
    },
    {
        "mode": "NO_CAPS_RAW_VOTE_OVERLAP",
        "model_pair_cap": False,
        "final_pair_cap": False,
        "truncate_to_reported": False,
        "description": "No pair caps; pure overlap of model top-k sets.",
    },
]


def top_k_count(n: int, top_pct: float) -> int:
    return max(1, int(n * top_pct))


def select_top_set(common: pd.DataFrame, model: str, top_pct: float, pair_cap: str | None) -> set[str]:
    k = top_k_count(len(common), top_pct)
    ranked = common.sort_values([f"score_{model}", "event_timestamp", "pair_address"], ascending=[False, True, True]).copy()

    if pair_cap is None or str(pair_cap).lower() == "none":
        return set(ranked.head(k)["event_key"])

    cap = int(float(pair_cap))
    selected = []
    counts = {}

    for row in ranked.itertuples(index=False):
        pair = row.pair_address
        n = counts.get(pair, 0)

        if n >= cap:
            continue

        selected.append(row.event_key)
        counts[pair] = n + 1

        if len(selected) >= k:
            break

    return set(selected)


def build_consensus_selection(common: pd.DataFrame, policy: pd.Series, mode_cfg: dict) -> pd.DataFrame:
    top_pct = float(policy["top_pct"])
    pair_cap = str(policy["pair_cap_str"])
    expected_selected = int(policy["selected_test"])

    model_cap = pair_cap if mode_cfg["model_pair_cap"] else None

    top_sets = {
        model: select_top_set(common, model, top_pct, model_cap)
        for model in v4.MODELS
    }

    df = common.copy()
    df["in_TAB"] = df["event_key"].isin(top_sets["TAB"])
    df["in_XGB"] = df["event_key"].isin(top_sets["XGB"])
    df["in_RF"] = df["event_key"].isin(top_sets["RF"])
    df["vote_count"] = df[["in_TAB", "in_XGB", "in_RF"]].sum(axis=1)

    df = df[df["vote_count"] >= 2].copy()

    df["combo"] = [
        v4.classify_combo(a, b, c)
        for a, b, c in zip(df["in_TAB"], df["in_XGB"], df["in_RF"])
    ]

    df["score_mean"] = df[["score_TAB", "score_XGB", "score_RF"]].mean(axis=1)
    df["score_min"] = df[["score_TAB", "score_XGB", "score_RF"]].min(axis=1)

    df = df.sort_values(
        ["vote_count", "score_mean", "score_min", "event_timestamp", "pair_address"],
        ascending=[False, False, False, True, True],
    ).copy()

    if mode_cfg["final_pair_cap"]:
        selected_idx = []
        counts = {}
        cap = int(float(pair_cap)) if str(pair_cap).lower() != "none" else None

        for idx, row in df.iterrows():
            pair = row["pair_address"]

            if cap is not None:
                n = counts.get(pair, 0)
                if n >= cap:
                    continue
                counts[pair] = n + 1

            selected_idx.append(idx)

            if mode_cfg["truncate_to_reported"] and len(selected_idx) >= expected_selected:
                break

        df = df.loc[selected_idx].copy()

    elif mode_cfg["truncate_to_reported"]:
        df = df.head(expected_selected).copy()

    return df.reset_index(drop=True)


def simulate_selection(selected: pd.DataFrame, policy: pd.Series, mode_name: str) -> pd.DataFrame:
    horizon = str(policy["horizon"])
    tp_ratio = float(policy["tp_ratio"])
    sl_ratio = float(policy["sl_ratio"])

    fee = float(policy["round_trip_fee_test"]) if "round_trip_fee_test" in policy.index else v4.ROUND_TRIP_FEE_DEFAULT

    sims = []

    for row in selected.itertuples(index=False):
        sims.append(
            v4.simulate_trade(
                pair=row.pair_address,
                event_ns=int(row.event_ns),
                horizon=horizon,
                tp_ratio=tp_ratio,
                sl_ratio=sl_ratio,
                fee=fee,
            )
        )

    sim_df = pd.DataFrame(sims)

    out = pd.concat([selected.reset_index(drop=True), sim_df.reset_index(drop=True)], axis=1)

    out["audit_mode"] = mode_name
    out["policy_source_kind"] = policy["source_kind"]
    out["policy_strategy"] = policy["strategy"]
    out["policy_filter"] = policy["filter"]
    out["policy_horizon"] = policy["horizon"]
    out["policy_top_pct"] = policy["top_pct"]
    out["policy_pair_cap"] = policy["pair_cap_str"]
    out["policy_tp_ratio"] = policy["tp_ratio"]
    out["policy_sl_ratio"] = policy["sl_ratio"]
    out["policy_selected_reported"] = int(policy["selected_test"])
    out["policy_total_reported"] = float(policy["total_net_return_test"])
    out["policy_avg_reported"] = float(policy["avg_net_return_test"])

    return out


def summarize_policy_mode(selected: pd.DataFrame, policy: pd.Series, mode_cfg: dict) -> dict:
    valid = selected[selected["valid"] == True].copy()

    selected_count = int(len(selected))
    valid_count = int(len(valid))

    total = float(valid["net_return"].sum()) if valid_count else np.nan
    avg = float(valid["net_return"].mean()) if valid_count else np.nan

    reported_selected = int(policy["selected_test"])
    reported_total = float(policy["total_net_return_test"])

    return {
        "audit_mode": mode_cfg["mode"],
        "mode_description": mode_cfg["description"],
        "source_kind": policy["source_kind"],
        "strategy": policy["strategy"],
        "filter": policy["filter"],
        "horizon": policy["horizon"],
        "top_pct": float(policy["top_pct"]),
        "pair_cap": str(policy["pair_cap_str"]),
        "tp_ratio": float(policy["tp_ratio"]),
        "sl_ratio": float(policy["sl_ratio"]),
        "reported_selected": reported_selected,
        "reported_total": reported_total,
        "audited_selected": selected_count,
        "audited_valid_sims": valid_count,
        "audited_total": total,
        "audited_avg": avg,
        "selected_diff_vs_reported": selected_count - reported_selected,
        "total_diff_vs_reported": total - reported_total if np.isfinite(total) else np.nan,
        "unique_pairs": int(valid["pair_address"].nunique()) if valid_count else 0,
        "top_pair_share": float(valid["pair_address"].value_counts(normalize=True).iloc[0]) if valid_count else np.nan,
        "target_precision": float(valid["target"].mean()) if "target" in valid.columns and valid["target"].notna().any() else np.nan,
        "net_win_rate": float((valid["net_return"] > 0).mean()) if valid_count else np.nan,
        "tp_count": int((valid["exit_status"] == "TP").sum()) if valid_count else 0,
        "sl_count": int((valid["exit_status"] == "SL").sum()) if valid_count else 0,
        "time_count": int((valid["exit_status"] == "TIME").sum()) if valid_count else 0,
    }


def summarize_combo(selected: pd.DataFrame, policy: pd.Series, mode_cfg: dict) -> list[dict]:
    rows = []

    for combo, g in selected.groupby("combo", dropna=False):
        valid = g[g["valid"] == True].copy()

        rows.append({
            "audit_mode": mode_cfg["mode"],
            "source_kind": policy["source_kind"],
            "strategy": policy["strategy"],
            "filter": policy["filter"],
            "horizon": policy["horizon"],
            "top_pct": float(policy["top_pct"]),
            "pair_cap": str(policy["pair_cap_str"]),
            "tp_ratio": float(policy["tp_ratio"]),
            "sl_ratio": float(policy["sl_ratio"]),
            "reported_selected": int(policy["selected_test"]),
            "reported_total": float(policy["total_net_return_test"]),
            "combo": combo,
            "combo_selected": int(len(g)),
            "combo_share_of_policy": float(len(g) / max(len(selected), 1)),
            "combo_valid_sims": int(len(valid)),
            "combo_total_net": float(valid["net_return"].sum()) if len(valid) else np.nan,
            "combo_avg_net": float(valid["net_return"].mean()) if len(valid) else np.nan,
            "combo_target_precision": float(valid["target"].mean()) if "target" in valid.columns and valid["target"].notna().any() else np.nan,
            "combo_net_win_rate": float((valid["net_return"] > 0).mean()) if len(valid) else np.nan,
            "combo_tp_count": int((valid["exit_status"] == "TP").sum()) if len(valid) else 0,
            "combo_sl_count": int((valid["exit_status"] == "SL").sum()) if len(valid) else 0,
            "combo_time_count": int((valid["exit_status"] == "TIME").sum()) if len(valid) else 0,
            "combo_unique_pairs": int(valid["pair_address"].nunique()) if len(valid) else 0,
            "combo_top_pair_share": float(valid["pair_address"].value_counts(normalize=True).iloc[0]) if len(valid) else np.nan,
        })

    return rows


def load_policies() -> pd.DataFrame:
    path = v4.PHASE_B_V2_DIR / "phase_b_v2_two_of_three_ranked.csv"

    if not path.exists():
        raise FileNotFoundError(f"Missing policy file: {path}")

    df = pd.read_csv(path)

    df = df[df["phase_b_robust_ok"].astype(str).str.lower().eq("true")].copy()
    df = df[~df["filter"].astype(str).eq("LOW_LIQ_MOMENTUM")].copy()

    for c in ["total_net_return_test", "avg_net_return_test", "selected_test"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(
        ["total_net_return_test", "avg_net_return_test", "selected_test"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return df


def main() -> None:
    print("Loading raw market snapshots from SQLite...")
    v4.SNAPSHOT_GROUPS = v4.load_snapshot_groups()
    print(f"Snapshot pair groups: {len(v4.SNAPSHOT_GROUPS)}")

    policies = load_policies()

    loads_cache = {}
    all_selected = []
    all_policy_rows = []
    all_combo_rows = []
    load_records = []
    failures = []

    for pi, policy in policies.iterrows():
        label = (
            f"{policy['source_kind']} / {policy['filter']} / {policy['horizon']} / "
            f"top={policy['top_pct']} / cap={policy['pair_cap_str']} / "
            f"TP={policy['tp_ratio']} / SL={policy['sl_ratio']}"
        )

        print(f"\nPolicy {pi+1}/{len(policies)}: {label}")

        try:
            split = "test"
            filter_name = str(policy["filter"])
            horizon = str(policy["horizon"])

            loads = {}

            for model in v4.MODELS:
                key = (model, split, filter_name, horizon)

                if key not in loads_cache:
                    loads_cache[key] = v4.discover_prediction_file(model, split, filter_name, horizon)

                loads[model] = loads_cache[key]

            universe_check = v4.validate_common_universe(loads)

            if not universe_check["ok"]:
                raise RuntimeError(f"Universe mismatch: {json.dumps(universe_check, indent=2)}")

            common = v4.build_common_frame(loads)

            if common.empty:
                raise RuntimeError("Common frame is empty after model merge.")

            for model in v4.MODELS:
                load_records.append({
                    "policy_index": int(pi),
                    "model": model,
                    "split": split,
                    "filter": filter_name,
                    "horizon": horizon,
                    "path": loads[model].path,
                    "rows": int(len(loads[model].df)),
                    "score_col": loads[model].schema.get("score_col"),
                    "timestamp_col": loads[model].schema.get("timestamp_col"),
                    "pair_col": loads[model].schema.get("pair_col"),
                    "target_col": loads[model].schema.get("target_col"),
                    "selection_reason": loads[model].selection_reason,
                })

            for mode_cfg in MODES:
                selected = build_consensus_selection(common, policy, mode_cfg)
                selected = simulate_selection(selected, policy, mode_cfg["mode"])

                selected["policy_index"] = int(pi)

                all_selected.append(selected)
                all_policy_rows.append(summarize_policy_mode(selected, policy, mode_cfg))
                all_combo_rows.extend(summarize_combo(selected, policy, mode_cfg))

                print(
                    f"  {mode_cfg['mode']}: "
                    f"selected={len(selected)} "
                    f"total={selected[selected['valid'] == True]['net_return'].sum():.6f} "
                    f"reported={float(policy['total_net_return_test']):.6f}"
                )

        except Exception as exc:
            failures.append({
                "policy_index": int(pi),
                "source_kind": policy.get("source_kind"),
                "strategy": policy.get("strategy"),
                "filter": policy.get("filter"),
                "horizon": policy.get("horizon"),
                "top_pct": policy.get("top_pct"),
                "pair_cap": policy.get("pair_cap_str"),
                "tp_ratio": policy.get("tp_ratio"),
                "sl_ratio": policy.get("sl_ratio"),
                "error": str(exc),
            })
            print("  FAILED:", str(exc)[:500])

    selected_df = pd.concat(all_selected, ignore_index=True, sort=False) if all_selected else pd.DataFrame()
    policy_df = pd.DataFrame(all_policy_rows)
    combo_df = pd.DataFrame(all_combo_rows)
    loads_df = pd.DataFrame(load_records)
    failures_df = pd.DataFrame(failures)

    selected_path = OUT_DIR / "phase_b_v5_audited_selected_trades.csv"
    policy_path = OUT_DIR / "phase_b_v5_audited_policy_mode_comparison.csv"
    combo_path = OUT_DIR / "phase_b_v5_audited_combo_composition.csv"
    loads_path = OUT_DIR / "phase_b_v5_prediction_file_audit.csv"
    failures_path = OUT_DIR / "phase_b_v5_failures.csv"

    selected_df.to_csv(selected_path, index=False)
    policy_df.to_csv(policy_path, index=False)
    combo_df.to_csv(combo_path, index=False)
    loads_df.to_csv(loads_path, index=False)
    failures_df.to_csv(failures_path, index=False)

    # Best interpretation per policy: closest to reported selected, then closest total.
    if not policy_df.empty:
        policy_df["abs_selected_diff"] = policy_df["selected_diff_vs_reported"].abs()
        policy_df["abs_total_diff"] = policy_df["total_diff_vs_reported"].abs()
        best_modes = (
            policy_df.sort_values(["source_kind", "filter", "horizon", "reported_total", "abs_selected_diff", "abs_total_diff"],
                                  ascending=[True, True, True, False, True, True])
            .groupby(["source_kind", "filter", "horizon", "top_pct", "pair_cap", "tp_ratio", "sl_ratio"], dropna=False)
            .head(1)
            .reset_index(drop=True)
        )
    else:
        best_modes = pd.DataFrame()

    best_modes_path = OUT_DIR / "phase_b_v5_best_audited_mode_per_policy.csv"
    best_modes.to_csv(best_modes_path, index=False)

    manifest = {
        "status": "ok",
        "method": "audited deterministic rerun; not retroactive reconstruction",
        "important_note": "Composition conclusions should be based on audited modes, not on V3 reconstruction.",
        "policies_attempted": int(len(policies)),
        "modes_per_policy": len(MODES),
        "policy_mode_rows": int(len(policy_df)),
        "combo_rows": int(len(combo_df)),
        "selected_trade_rows": int(len(selected_df)),
        "failures": int(len(failures_df)),
        "outputs": {
            "selected_trades": str(selected_path),
            "policy_mode_comparison": str(policy_path),
            "combo_composition": str(combo_path),
            "best_mode_per_policy": str(best_modes_path),
            "prediction_file_audit": str(loads_path),
            "failures": str(failures_path),
        },
        "modes": MODES,
    }

    manifest_path = OUT_DIR / "phase_b_v5_audited_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = []
    lines.append("PHASE B V5.1 — AUDITED TWO_OF_THREE RERUN")
    lines.append("=" * 140)
    lines.append("")
    lines.append("Purpose")
    lines.append("-" * 140)
    lines.append("V5.1 does not try to infer hidden composition from old aggregate files.")
    lines.append("It reruns transparent TWO_OF_THREE selection modes and exports every selected trade with model votes and combo type.")
    lines.append("")
    lines.append("Why this exists")
    lines.append("-" * 140)
    lines.append("The original consensus generator was not located in the repository inventory.")
    lines.append("V4 proved that retroactive reconstruction from aggregate files is not reliable enough for reporting.")
    lines.append("")
    lines.append("Policy-mode comparison")
    lines.append("-" * 140)

    if not policy_df.empty:
        show_cols = [
            "audit_mode",
            "source_kind",
            "filter",
            "horizon",
            "top_pct",
            "pair_cap",
            "tp_ratio",
            "sl_ratio",
            "reported_selected",
            "audited_selected",
            "selected_diff_vs_reported",
            "reported_total",
            "audited_total",
            "total_diff_vs_reported",
            "unique_pairs",
            "top_pair_share",
            "target_precision",
            "net_win_rate",
        ]
        lines.append(policy_df[show_cols].sort_values(
            ["reported_total", "filter", "horizon", "audit_mode"],
            ascending=[False, True, True, True],
        ).to_string(index=False))
    else:
        lines.append("EMPTY")

    lines.append("")
    lines.append("Best audited mode per policy")
    lines.append("-" * 140)

    if not best_modes.empty:
        show_cols = [
            "audit_mode",
            "source_kind",
            "filter",
            "horizon",
            "top_pct",
            "pair_cap",
            "reported_selected",
            "audited_selected",
            "selected_diff_vs_reported",
            "reported_total",
            "audited_total",
            "total_diff_vs_reported",
        ]
        lines.append(best_modes[show_cols].to_string(index=False))
    else:
        lines.append("EMPTY")

    lines.append("")
    lines.append("Combo composition")
    lines.append("-" * 140)

    if not combo_df.empty:
        show_cols = [
            "audit_mode",
            "source_kind",
            "filter",
            "horizon",
            "top_pct",
            "pair_cap",
            "reported_total",
            "combo",
            "combo_selected",
            "combo_share_of_policy",
            "combo_total_net",
            "combo_avg_net",
            "combo_target_precision",
            "combo_net_win_rate",
            "combo_unique_pairs",
            "combo_top_pair_share",
        ]
        lines.append(combo_df[show_cols].sort_values(
            ["reported_total", "filter", "horizon", "audit_mode", "combo_total_net"],
            ascending=[False, True, True, True, False],
        ).to_string(index=False))
    else:
        lines.append("EMPTY")

    lines.append("")
    lines.append("Outputs")
    lines.append("-" * 140)
    for k, v in manifest["outputs"].items():
        lines.append(f"{k}: {v}")

    summary_path = OUT_DIR / "phase_b_v5_audited_summary_for_upload.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")

    print("")
    print("DONE")
    print("Summary:", summary_path)
    print("Policy modes:", policy_path)
    print("Combo composition:", combo_path)
    print("Best modes:", best_modes_path)
    print("Selected trades:", selected_path)
    print("Prediction file audit:", loads_path)
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
