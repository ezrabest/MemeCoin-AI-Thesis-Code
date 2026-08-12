import csv
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict, Counter

PROJECT_ROOT = Path.cwd()
E8E_DIR = PROJECT_ROOT / "data/training/manual_verified_results/phase_e8e_rare_winner_context_forensics_20260707_195349"
REPORTS = E8E_DIR / "reports"

if not REPORTS.exists():
    raise SystemExit(f"Reports folder not found: {REPORTS}")

OUT_DIR = E8E_DIR / ("e8e_manual_closure_qa_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
OUT_REPORTS = OUT_DIR / "reports"
OUT_REPORTS.mkdir(parents=True, exist_ok=True)

def read_csv_required(name):
    path = REPORTS / name
    if not path.exists():
        raise SystemExit(f"Required CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def read_csv_optional(name):
    path = REPORTS / name
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def to_bool(v):
    if v is None:
        return False
    return str(v).strip().lower() in {"true", "1", "yes", "y"}

def to_float(v, default=math.nan):
    if v is None:
        return default
    s = str(v).strip()
    if s == "" or s.lower() == "nan":
        return default
    try:
        return float(s)
    except Exception:
        return default

def is_blank(v):
    if v is None:
        return True
    s = str(v).strip()
    return s == "" or s.lower() == "nan"

def norm_pair(v):
    if v is None:
        return ""
    return str(v).strip().lower()

feature_map = read_csv_required("e8e_context_feature_candidate_map.csv")
patterns = read_csv_required("e8e_pattern_candidates.csv")
matched_controls = read_csv_required("e8e_matched_control_comparison.csv")
identity_map = read_csv_required("e8e_candidate_identity_map.csv")
reservoir_overlap = read_csv_required("e8e_reservoir_overlap.csv")
final_classification = read_csv_optional("e8e_final_classification.csv")

# -----------------------------
# 1. Feature gate
# -----------------------------

feature_gate = []
for f in feature_map:
    feature_name = f.get("feature_name", "")
    orig = f.get("recommended_status", "")
    pre_entry_legal = to_bool(f.get("pre_entry_legal"))
    is_leaky = to_bool(f.get("is_leaky"))
    missing = to_float(f.get("missingness_rate"), 1.0)
    pair_risk = str(f.get("pair_identity_risk", "")).strip().lower()
    leakage_risk = str(f.get("leakage_risk", "")).strip().lower()
    dist = f.get("distinguishes_winners_from_controls", "")

    if is_leaky:
        status = "REJECT_LEAKAGE"
        reason = "is_leaky=true"
    elif not pre_entry_legal:
        status = "REJECT_LEAKAGE"
        reason = "pre_entry_legal is not true"
    elif missing > 0.50:
        status = "REJECT_NOT_AVAILABLE"
        reason = "missingness_rate > 0.50"
    elif pair_risk == "high":
        status = "REJECT_PAIR_IDENTITY_RISK"
        reason = "pair_identity_risk=high"
    elif orig == "KEEP_FOR_E9" and is_blank(dist):
        status = "AVAILABLE_BUT_UNPROVEN"
        reason = "legal and available, but no winner-vs-control distinction evidence in feature map"
    elif to_bool(dist) and orig == "KEEP_FOR_E9":
        status = "KEEP_AS_E8_CONTEXT_CANDIDATE"
        reason = "legal, available, and distinction evidence present"
    else:
        status = "REJECT_NO_DISTINCTION_EVIDENCE"
        reason = "no sufficient distinction evidence"

    feature_gate.append({
        "feature_name": feature_name,
        "original_recommended_status": orig,
        "pre_entry_legal": pre_entry_legal,
        "is_leaky": is_leaky,
        "missingness_rate": missing,
        "available_for_runtime_estimate": f.get("available_for_runtime_estimate", ""),
        "distinguishes_winners_from_controls": dist,
        "pair_identity_risk": pair_risk,
        "leakage_risk": leakage_risk,
        "closure_status": status,
        "closure_reason": reason,
    })

write_csv(OUT_REPORTS / "e8e_manual_closure_feature_gate.csv", feature_gate)

# -----------------------------
# 2. Pattern gate
# -----------------------------

pattern_gate = []
for p in patterns:
    winner = to_float(p.get("winner_value"))
    loser = to_float(p.get("loser_value"))
    control = to_float(p.get("control_value"))
    direction = str(p.get("effect_direction", "")).strip()
    legal = to_bool(p.get("pre_entry_legal"))
    risk = str(p.get("pair_identity_risk", "")).strip().lower()

    passes_loser = False
    passes_control = False

    if direction == "higher_in_winners":
        passes_loser = winner > loser
        passes_control = winner > control
    elif direction == "lower_in_winners":
        passes_loser = winner < loser
        passes_control = winner < control

    if not legal:
        status = "REJECT_LEAKAGE"
        reason = "pattern is not pre-entry legal"
    elif risk == "high":
        status = "REJECT_PAIR_IDENTITY_RISK"
        reason = "high pair identity risk"
    elif passes_loser and passes_control:
        if risk == "medium":
            status = "CONTEXT_CANDIDATE_WEAK"
            reason = "passes loser and control direction, but pair_identity_risk=medium"
        else:
            status = "PROVEN_CONTEXT_PATTERN"
            reason = "passes loser and control direction"
    elif passes_loser and not passes_control:
        status = "FAILS_CONTROL_COMPARISON"
        reason = "passes selected-losers comparison but fails matched-controls comparison"
    elif not passes_loser and passes_control:
        status = "FAILS_LOSER_COMPARISON"
        reason = "passes controls comparison but fails selected-losers comparison"
    else:
        status = "FAILS_BOTH_COMPARISONS"
        reason = "does not distinguish winners from losers or controls"

    pattern_gate.append({
        "pattern_name": p.get("pattern_name", ""),
        "winner_value": winner,
        "loser_value": loser,
        "control_value": control,
        "effect_direction": direction,
        "pre_entry_legal": legal,
        "pair_identity_risk": risk,
        "passes_loser_comparison": passes_loser,
        "passes_control_comparison": passes_control,
        "pattern_status": status,
        "pattern_reason": reason,
    })

write_csv(OUT_REPORTS / "e8e_manual_closure_pattern_gate.csv", pattern_gate)

# -----------------------------
# 3. Concentration gate
# -----------------------------

by_group = defaultdict(list)
for r in identity_map:
    by_group[r.get("group", "")].append({
        "pair": norm_pair(r.get("pair_address")),
        "identity_key": r.get("identity_key", ""),
    })

concentration_gate = []
for group, rows in by_group.items():
    counts = Counter(r["pair"] for r in rows)
    top_pair, top_count = ("", 0)
    if counts:
        top_pair, top_count = counts.most_common(1)[0]
    row_count = len(rows)
    unique_pairs = len(counts)
    top_share = (top_count / row_count) if row_count else 0.0

    concentration_gate.append({
        "group": group,
        "row_count": row_count,
        "unique_pairs": unique_pairs,
        "top_pair": top_pair,
        "top_pair_count": top_count,
        "top_pair_share": top_share,
    })

write_csv(OUT_REPORTS / "e8e_manual_closure_concentration_gate.csv", concentration_gate)

# -----------------------------
# 4. Reservoir gate
# -----------------------------

res_by_group = defaultdict(list)
for r in reservoir_overlap:
    res_by_group[r.get("group", "")].append({
        "pair": norm_pair(r.get("pair_address")),
        "hit": to_bool(r.get("in_e7_reservoir")),
    })

reservoir_gate = []
for group, rows in res_by_group.items():
    hits = [r for r in rows if r["hit"]]
    unique_pairs = len(set(r["pair"] for r in rows))
    hit_pairs = len(set(r["pair"] for r in hits))
    hit_rate = (len(hits) / len(rows)) if rows else 0.0

    reservoir_gate.append({
        "group": group,
        "row_count": len(rows),
        "unique_pairs": unique_pairs,
        "reservoir_hit_rows": len(hits),
        "reservoir_hit_pairs": hit_pairs,
        "reservoir_hit_rate": hit_rate,
    })

write_csv(OUT_REPORTS / "e8e_manual_closure_reservoir_gate.csv", reservoir_gate)

# -----------------------------
# 5. Final strict classification
# -----------------------------

def count_status(rows, key, value):
    return sum(1 for r in rows if r.get(key) == value)

kept_features = count_status(feature_gate, "closure_status", "KEEP_AS_E8_CONTEXT_CANDIDATE")
available_but_unproven = count_status(feature_gate, "closure_status", "AVAILABLE_BUT_UNPROVEN")
rejected_leakage = count_status(feature_gate, "closure_status", "REJECT_LEAKAGE")
rejected_not_available = count_status(feature_gate, "closure_status", "REJECT_NOT_AVAILABLE")

proven_patterns = count_status(pattern_gate, "pattern_status", "PROVEN_CONTEXT_PATTERN")
weak_patterns = count_status(pattern_gate, "pattern_status", "CONTEXT_CANDIDATE_WEAK")
failed_control_patterns = count_status(pattern_gate, "pattern_status", "FAILS_CONTROL_COMPARISON")

conc_by_group = {r["group"]: r for r in concentration_gate}
pos_conc = conc_by_group.get("rare_winner_selected_positives", {})
positive_unique_pairs = int(pos_conc.get("unique_pairs", 0) or 0)
positive_top_pair_share = float(pos_conc.get("top_pair_share", 1.0) or 1.0)

res_by_group2 = {r["group"]: r for r in reservoir_gate}
pos_res = res_by_group2.get("rare_winner_selected_positives", {})
ctrl_res = res_by_group2.get("matched_random_controls", {})
positive_res_hit_rate = float(pos_res.get("reservoir_hit_rate", 0.0) or 0.0)
control_res_hit_rate = float(ctrl_res.get("reservoir_hit_rate", 0.0) or 0.0)
reservoir_differentiates = abs(positive_res_hit_rate - control_res_hit_rate) >= 0.25

prior_classification = ""
if final_classification and "final_classification" in final_classification[0]:
    prior_classification = final_classification[0].get("final_classification", "")

blockers = []
if positive_unique_pairs < 5:
    blockers.append("rare_winner_selected_positives unique_pairs < 5")
if positive_top_pair_share > 0.50:
    blockers.append("rare_winner_selected_positives top_pair_share > 0.50")
if kept_features == 0:
    blockers.append("no features passed strict KEEP_AS_E8_CONTEXT_CANDIDATE gate")
if proven_patterns == 0:
    blockers.append("no PROVEN_CONTEXT_PATTERN after loser/control comparison")
if not reservoir_differentiates:
    blockers.append("reservoir overlap does not sufficiently distinguish positives from controls")

if positive_unique_pairs < 5:
    strict_classification = "E8E_RARE_WINNER_RESEARCH_ONLY"
    reason = "Rare-winner signal remains concentrated in too few positive pairs; E8 cannot validate a general context pattern."
elif proven_patterns > 0 and kept_features > 0 and reservoir_differentiates:
    strict_classification = "E8E_CONTEXT_PATTERN_VALIDATED"
    reason = "Strict gates passed: legal features, proven patterns, sufficient pair breadth, and differentiating reservoir support."
elif available_but_unproven > 0 or weak_patterns > 0:
    strict_classification = "E8E_CONTEXT_CANDIDATES_FOUND_BUT_UNPROVEN"
    reason = "Legal context candidates exist, but distinction evidence is incomplete or weak."
elif positive_unique_pairs <= 3 or positive_top_pair_share > 0.50:
    strict_classification = "E8E_PAIR_IDENTITY_ARTIFACT"
    reason = "Signal is mainly explained by a small number of pairs."
else:
    strict_classification = "E8E_INSUFFICIENT_CONTEXT_DATA"
    reason = "Context data did not support a strict E8 closure decision."

reclassification = [{
    "prior_cursor_classification": prior_classification,
    "strict_closure_classification": strict_classification,
    "reason": reason,
    "kept_features_strict": kept_features,
    "available_but_unproven_features": available_but_unproven,
    "rejected_leakage_features": rejected_leakage,
    "rejected_not_available_features": rejected_not_available,
    "proven_patterns": proven_patterns,
    "weak_patterns": weak_patterns,
    "failed_control_patterns": failed_control_patterns,
    "positive_unique_pairs": positive_unique_pairs,
    "positive_top_pair_share": positive_top_pair_share,
    "positive_reservoir_hit_rate": positive_res_hit_rate,
    "control_reservoir_hit_rate": control_res_hit_rate,
    "reservoir_differentiates": reservoir_differentiates,
    "blockers": " | ".join(blockers),
    "no_training_performed": True,
    "no_runtime_changes": True,
    "no_db_writes": True,
    "reservoir_scoring_performed": False,
    "recommended_next_step_within_E8": "Prepare main-thread E8 closure report; do not open E9 from this branch.",
}]

write_csv(OUT_REPORTS / "e8e_manual_closure_reclassification.csv", reclassification)

manifest = {
    "phase": "E8E Manual Closure QA",
    "created_at": datetime.now(timezone.utc).isoformat(),
    "input_e8e_output_dir": str(E8E_DIR.resolve()),
    "output_dir": str(OUT_DIR.resolve()),
    "prior_cursor_classification": prior_classification,
    "strict_closure_classification": strict_classification,
    "reason": reason,
    "blockers": blockers,
    "feature_counts": {
        "keep_as_e8_context_candidate": kept_features,
        "available_but_unproven": available_but_unproven,
        "reject_leakage": rejected_leakage,
        "reject_not_available": rejected_not_available,
    },
    "pattern_counts": {
        "proven_context_pattern": proven_patterns,
        "weak_context_candidate": weak_patterns,
        "fails_control_comparison": failed_control_patterns,
    },
    "concentration": {
        "positive_unique_pairs": positive_unique_pairs,
        "positive_top_pair_share": positive_top_pair_share,
    },
    "reservoir": {
        "positive_hit_rate": positive_res_hit_rate,
        "control_hit_rate": control_res_hit_rate,
        "differentiates": reservoir_differentiates,
    },
    "no_training_performed": True,
    "no_runtime_changes": True,
    "no_db_writes": True,
    "reservoir_scoring_performed": False,
}

(OUT_REPORTS / "e8e_manual_closure_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

summary = f"""Phase E8E Manual Closure QA

input_e8e_output_dir: {E8E_DIR}
output_dir: {OUT_DIR}

Prior Cursor classification:
{prior_classification}

Strict E8 closure classification:
{strict_classification}

Reason:
{reason}

Key counts:
- strict kept context features: {kept_features}
- available but unproven features: {available_but_unproven}
- rejected leakage features: {rejected_leakage}
- rejected not available features: {rejected_not_available}
- proven context patterns: {proven_patterns}
- weak context patterns: {weak_patterns}
- failed control-comparison patterns: {failed_control_patterns}
- rare winner positive unique pairs: {positive_unique_pairs}
- rare winner positive top pair share: {positive_top_pair_share}
- positive reservoir hit rate: {positive_res_hit_rate}
- matched-control reservoir hit rate: {control_res_hit_rate}
- reservoir differentiates positives from controls: {reservoir_differentiates}

Blockers:
{chr(10).join(blockers) if blockers else "None"}

Safety:
- no_training_performed = true
- no_runtime_changes = true
- no_db_writes = true
- reservoir_scoring_performed = false

E8 recommendation:
Prepare main-thread E8 closure report. Do not open E9 from this branch.
"""

(OUT_REPORTS / "e8e_manual_closure_decision_summary.txt").write_text(summary, encoding="utf-8")

print("\nDONE — E8E Manual Closure QA")
print("Output dir:")
print(OUT_DIR)
print("\nStrict classification:")
print(strict_classification)
print("\nSummary:")
print(summary)
print("\nFiles written:")
for p in sorted(OUT_REPORTS.glob("*")):
    print(f"- {p.name} ({p.stat().st_size} bytes)")
