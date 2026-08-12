from __future__ import annotations

from pathlib import Path
import re
import json


ROOT = Path(".").resolve()

OUT_DIR = ROOT / "data" / "training" / "manual_verified_results" / "phase_b_model_cuts_v5"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEARCH_DIRS = [
    ROOT / "scripts",
    ROOT / "app",
    ROOT / "tests",
]

KEYWORDS = [
    "TWO_OF_THREE",
    "ALL3_INTERSECT",
    "TAB_XGB_INTERSECT",
    "TAB_RF_INTERSECT",
    "XGB_RF_INTERSECT",
    "strict50_validation_selected_consensus_applied_to_test",
    "highconf20_validation_selected_consensus_applied_to_test",
    "research_only_best_test_consensus_NOT_FOR_SELECTION",
    "consensus_intersections",
    "consensus_upload_pack",
]

OUTPUT_FILENAMES = [
    "strict50_validation_selected_consensus_applied_to_test.csv",
    "highconf20_validation_selected_consensus_applied_to_test.csv",
    "research_only_best_test_consensus_NOT_FOR_SELECTION.csv",
    "consensus_summary_for_upload.txt",
    "consensus_manifest.json",
    "consensus_merge_manifest.csv",
]

EXCLUDE_DIR_TOKENS = {
    ".venv",
    "__pycache__",
    ".git",
    ".pytest_cache",
    "node_modules",
}


def should_skip(path: Path) -> bool:
    parts = {p.lower() for p in path.parts}
    return any(tok.lower() in parts for tok in EXCLUDE_DIR_TOKENS)


def read_text_safe(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return path.read_text(encoding="utf-8-sig")
        except Exception:
            return None
    except Exception:
        return None


def collect_python_files() -> list[Path]:
    files = []

    for d in SEARCH_DIRS:
        if not d.exists():
            continue

        for p in d.rglob("*.py"):
            if should_skip(p):
                continue
            files.append(p)

    return sorted(files)


def context_lines(text: str, keyword: str, radius: int = 8) -> list[dict]:
    lines = text.splitlines()
    hits = []

    for i, line in enumerate(lines, start=1):
        if keyword in line:
            start = max(1, i - radius)
            end = min(len(lines), i + radius)

            snippet = []
            for j in range(start, end + 1):
                snippet.append(f"{j:05d}: {lines[j-1]}")

            hits.append({
                "keyword": keyword,
                "line": i,
                "snippet": "\n".join(snippet),
            })

    return hits


def score_file(path: Path, text: str) -> tuple[int, list[str]]:
    score = 0
    reasons = []

    lower_path = str(path).lower()

    if "consensus" in lower_path:
        score += 30
        reasons.append("path_contains_consensus")

    if "intersection" in lower_path:
        score += 20
        reasons.append("path_contains_intersection")

    if "phase_b" in lower_path:
        score -= 20
        reasons.append("path_contains_phase_b_generated")

    for kw in KEYWORDS:
        count = text.count(kw)
        if count:
            score += count * 10
            reasons.append(f"{kw} x{count}")

    for name in OUTPUT_FILENAMES:
        if name in text:
            score += 50
            reasons.append(f"writes_or_mentions_{name}")

    if "to_csv" in text:
        score += 10
        reasons.append("contains_to_csv")

    if "read_parquet" in text:
        score += 10
        reasons.append("contains_read_parquet")

    if "selected_test" in text:
        score += 10
        reasons.append("contains_selected_test")

    if "total_net_return_test" in text:
        score += 10
        reasons.append("contains_total_net_return_test")

    return score, reasons


def main() -> None:
    files = collect_python_files()
    candidates = []

    for path in files:
        text = read_text_safe(path)
        if text is None:
            continue

        score, reasons = score_file(path, text)

        if score <= 0:
            continue

        hits = []
        for kw in KEYWORDS + OUTPUT_FILENAMES:
            hits.extend(context_lines(text, kw, radius=6))

        candidates.append({
            "path": str(path),
            "score": score,
            "reasons": reasons,
            "hits": hits[:30],
            "hit_count": len(hits),
            "line_count": len(text.splitlines()),
        })

    candidates = sorted(candidates, key=lambda x: (x["score"], x["hit_count"]), reverse=True)

    manifest = {
        "status": "ok",
        "repo": str(ROOT),
        "candidate_count": len(candidates),
        "candidates": [
            {
                "rank": i + 1,
                "path": c["path"],
                "score": c["score"],
                "hit_count": c["hit_count"],
                "line_count": c["line_count"],
                "reasons": c["reasons"],
            }
            for i, c in enumerate(candidates)
        ],
    }

    manifest_path = OUT_DIR / "phase_b_v5_consensus_source_inventory.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    report = []
    report.append("PHASE B V5.0 — CONSENSUS SOURCE INVENTORY")
    report.append("=" * 120)
    report.append("")
    report.append(f"Candidate files found: {len(candidates)}")
    report.append("")

    for i, c in enumerate(candidates[:12], start=1):
        report.append("-" * 120)
        report.append(f"RANK {i}")
        report.append(f"PATH: {c['path']}")
        report.append(f"SCORE: {c['score']}")
        report.append(f"HITS: {c['hit_count']}")
        report.append("REASONS:")
        for r in c["reasons"]:
            report.append(f"  - {r}")
        report.append("")
        report.append("SNIPPETS:")
        for h in c["hits"][:10]:
            report.append("")
            report.append(f"### keyword={h['keyword']} line={h['line']}")
            report.append(h["snippet"])
        report.append("")

    report_path = OUT_DIR / "phase_b_v5_consensus_source_inventory.txt"
    report_path.write_text("\n".join(report), encoding="utf-8")

    print("DONE")
    print("Inventory report:", report_path)
    print("Inventory JSON:", manifest_path)
    print("")
    print("Top candidates:")
    for i, c in enumerate(candidates[:10], start=1):
        print(f"{i}. score={c['score']} hits={c['hit_count']} path={c['path']}")


if __name__ == "__main__":
    main()
