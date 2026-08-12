"""CLI: Generate Final MSc reporting markdown from AE12 audit artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ae12_reporting.final_docs import write_final_docs
from app.ae12_reporting.report_manager import AE12ReportManager


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AE12.5 — generate Final MSc docs from AE12 JSON/CSV source data"
    )
    p.add_argument(
        "--ae12-root",
        type=str,
        default=None,
        help="Explicit AE12 maturation output root (default: latest under data/audits)",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=str(PROJECT_ROOT / "docs" / "msc_final"),
        help="Directory for generated markdown docs",
    )
    p.add_argument(
        "--project-root",
        type=str,
        default=str(PROJECT_ROOT),
    )
    p.add_argument(
        "--ttl-seconds",
        type=int,
        default=300,
        help="Manager cache TTL (docs generation is typically a single load)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manager = AE12ReportManager(
        project_root=Path(args.project_root),
        ttl_seconds=int(args.ttl_seconds),
        maturation_root=Path(args.ae12_root) if args.ae12_root else None,
    )
    status = manager.get_status()
    if not status.get("latest_ae12_output_root"):
        print("ERROR: No AE12 maturation root found.", file=sys.stderr)
        return 2
    manifest = write_final_docs(manager, Path(args.output_root))
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
