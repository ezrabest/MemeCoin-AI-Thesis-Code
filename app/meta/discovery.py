"""AE17 read-only AE16 artifact discovery with controlled failure."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from app.meta.constants import (
    AE16_CONSENSUS_PATTERNS,
    AE16_EVIDENCE_PATTERNS,
    KNOWN_AE16_ROOTS,
)
from app.meta.models import AE17InputDiscoveryResult

CLASSIFICATION_MISSING = "AE17_BLOCKED_MISSING_AE16_INPUTS"
CLASSIFICATION_OK = "AE17_INPUTS_DISCOVERED"


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except ValueError:
        return str(path.resolve()).replace("\\", "/")


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _glob_under(root: Path, pattern: str) -> list[Path]:
    if not root.is_dir():
        return []
    # Patterns are **/name.csv — Path.glob supports **.
    return sorted(p for p in root.glob(pattern) if p.is_file())


def _score_consensus_candidate(path: Path) -> int:
    """Prefer TAB16 consensus preview, then AE16 consensus decisions."""
    name = path.name.lower()
    score = 0
    if "rf_xgb_tab16_consensus_preview" in name:
        score += 100
    if "ae16_clean_forward_consensus_decisions" in name:
        score += 80
    if "tiered_consensus_rows" in name:
        score += 60
    if "consensus" in name:
        score += 10
    # Prefer newer mtime as weak tie-break encoded as 0..9 via caller sort.
    return score


def _score_evidence_candidate(path: Path) -> int:
    name = path.name.lower()
    score = 0
    if "rf_xgb_tab16_current_model_evidence" in name:
        score += 100
    if "ae16_model_evidence_attachment" in name:
        score += 80
    if "model_evidence" in name:
        score += 40
    return score


def discover_ae16_artifacts(
    project_root: Path,
    *,
    ae16_root: str | Path | None = None,
) -> AE17InputDiscoveryResult:
    """Discover AE16 consensus/evidence CSVs with controlled missing-input status.

    Never raises FileNotFoundError for missing AE16 paths.
    """
    root = project_root.resolve()
    audits = root / "data" / "audits"
    searched: list[str] = []
    expected = list(AE16_CONSENSUS_PATTERNS) + list(AE16_EVIDENCE_PATTERNS)
    found: list[dict[str, Any]] = []
    notes: list[str] = []

    search_roots: list[Path] = []
    if ae16_root:
        p = Path(ae16_root)
        if not p.is_absolute():
            p = root / p
        search_roots.append(p)
        searched.append(_rel(p, root) if p.exists() or True else str(ae16_root))
        if not p.is_dir():
            notes.append(f"provided --ae16-root is missing or not a directory: {_rel(p, root)}")

    for rel in KNOWN_AE16_ROOTS:
        p = root / rel
        if p not in search_roots:
            search_roots.append(p)
            searched.append(rel)

    # Broader scan of data/audits for ae16* / *tab16* roots.
    if audits.is_dir():
        for child in sorted(audits.iterdir()):
            if not child.is_dir():
                continue
            name = child.name.lower()
            if "ae16" in name or "tab16" in name:
                if child not in search_roots:
                    search_roots.append(child)
                    searched.append(_rel(child, root))

    consensus_hits: list[Path] = []
    evidence_hits: list[Path] = []

    for sr in search_roots:
        for pat in AE16_CONSENSUS_PATTERNS:
            for hit in _glob_under(sr, pat):
                consensus_hits.append(hit)
                found.append(
                    {
                        "kind": "consensus_rows",
                        "path": _rel(hit, root),
                        "root": _rel(sr, root) if sr.exists() else str(sr),
                        "score": _score_consensus_candidate(hit),
                    }
                )
        for pat in AE16_EVIDENCE_PATTERNS:
            for hit in _glob_under(sr, pat):
                evidence_hits.append(hit)
                found.append(
                    {
                        "kind": "model_evidence",
                        "path": _rel(hit, root),
                        "root": _rel(sr, root) if sr.exists() else str(sr),
                        "score": _score_evidence_candidate(hit),
                    }
                )

    # Deduplicate by resolved path.
    def _dedupe(paths: list[Path]) -> list[Path]:
        seen: set[str] = set()
        out: list[Path] = []
        for p in paths:
            key = str(p.resolve())
            if key in seen:
                continue
            seen.add(key)
            out.append(p)
        return out

    consensus_hits = _dedupe(consensus_hits)
    evidence_hits = _dedupe(evidence_hits)

    # Prefer hits under explicit --ae16-root when provided.
    preferred_root: Path | None = None
    if ae16_root:
        preferred_root = Path(ae16_root)
        if not preferred_root.is_absolute():
            preferred_root = root / preferred_root

    def _rank_consensus(p: Path) -> tuple[int, int, float]:
        prefer = 1 if preferred_root and _is_under(p, preferred_root) else 0
        return (prefer, _score_consensus_candidate(p), p.stat().st_mtime if p.is_file() else 0.0)

    def _rank_evidence(p: Path) -> tuple[int, int, float]:
        prefer = 1 if preferred_root and _is_under(p, preferred_root) else 0
        return (prefer, _score_evidence_candidate(p), p.stat().st_mtime if p.is_file() else 0.0)

    selected_consensus: Path | None = None
    selected_evidence: Path | None = None
    if consensus_hits:
        selected_consensus = sorted(consensus_hits, key=_rank_consensus, reverse=True)[0]
    if evidence_hits:
        selected_evidence = sorted(evidence_hits, key=_rank_evidence, reverse=True)[0]

    missing: list[str] = []
    if selected_consensus is None:
        missing.append("consensus_rows")

    if missing:
        return AE17InputDiscoveryResult(
            status=CLASSIFICATION_MISSING,
            searched_roots=searched,
            expected_patterns=expected,
            missing_required_artifacts=missing,
            found_candidate_artifacts=found,
            selected_consensus_path=None,
            selected_evidence_path=_rel(selected_evidence, root) if selected_evidence else None,
            selected_root=None,
            recommended_next_action=(
                "Re-run AE16 TAB16 / tiered consensus to produce "
                "rf_xgb_tab16_consensus_preview.csv or ae16_clean_forward_consensus_decisions.csv, "
                "then re-run AE17 with --ae16-root pointing at that audit package."
            ),
            notes=notes,
        )

    # Infer selected root as the audit package containing the consensus file.
    sel_root = selected_consensus.parent
    if sel_root.name.lower() in {"data", "reports", "audits"}:
        sel_root = sel_root.parent

    return AE17InputDiscoveryResult(
        status=CLASSIFICATION_OK,
        searched_roots=searched,
        expected_patterns=expected,
        missing_required_artifacts=[],
        found_candidate_artifacts=found,
        selected_consensus_path=_rel(selected_consensus, root),
        selected_evidence_path=_rel(selected_evidence, root) if selected_evidence else None,
        selected_root=_rel(sel_root, root),
        recommended_next_action="Proceed with AE17 meta feature construction from selected AE16 consensus artifact.",
        notes=notes
        + (
            []
            if selected_evidence
            else ["No separate model-evidence CSV found; consensus artifact alone will be used."]
        ),
    )
