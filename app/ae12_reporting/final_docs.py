"""Generate Final MSc markdown docs from AE12ReportManager source data."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .doc_templates import DOC_RENDERERS
from .report_manager import AE12ReportManager


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_doc_context(manager: AE12ReportManager) -> dict[str, Any]:
    bundle = manager.get_source_bundle_for_docs()
    source_files: list[str] = []
    for path in (bundle.get("source_files") or {}).values():
        if path is not None:
            source_files.append(str(path))
    # Also note summary nested paths if present
    summary = bundle.get("summary") or {}
    if summary.get("output_root"):
        source_files.append(str(Path(summary["output_root"]) / "reports" / "ae12_forward_evidence_summary.json"))

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique_sources: list[str] = []
    for s in source_files:
        if s not in seen:
            seen.add(s)
            unique_sources.append(s)

    return {
        "generated_at": _utc_now(),
        "source_ae12_output_root": bundle.get("maturation_root"),
        "source_files_used": unique_sources,
        "summary": summary,
        "gate": bundle.get("gate") or {},
        "wallet": bundle.get("wallet") or {},
        "forward": bundle.get("forward") or {},
        "trade_vs_no_trade": bundle.get("trade_vs_no_trade") or {},
        "strict_vs_exploration": bundle.get("strict_vs_exploration") or {},
        "qwen": bundle.get("qwen") or {},
        "safety": bundle.get("safety") or {},
        "missed_winners": bundle.get("missed_winners") or {},
        "runtime": bundle.get("runtime") or {},
        "final": bundle.get("final") or {},
    }


def render_all_docs(manager: AE12ReportManager) -> dict[str, str]:
    ctx = build_doc_context(manager)
    return {name: renderer(ctx) for name, renderer in DOC_RENDERERS.items()}


def write_final_docs(
    manager: AE12ReportManager,
    output_root: Path | str,
) -> dict[str, Any]:
    """Render and write markdown docs. Creates output_root if needed. Read-only vs AE12 sources."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    rendered = render_all_docs(manager)
    written: list[str] = []
    for name, body in rendered.items():
        path = output_root / name
        path.write_text(body, encoding="utf-8")
        written.append(str(path))
    ctx = build_doc_context(manager)
    manifest = {
        "generated_at": ctx["generated_at"],
        "source_ae12_output_root": ctx["source_ae12_output_root"],
        "source_files_used": ctx["source_files_used"],
        "output_root": str(output_root),
        "files_written": written,
        "note": "Values generated from AE12 artifacts; not hard-coded constants.",
        "mutated_ae12_source": False,
        "live_trading_ready": False,
        "profitability_proven": False,
    }
    manifest_path = output_root / "ae12_final_docs_manifest.json"
    import json

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written.append(str(manifest_path))
    return manifest
