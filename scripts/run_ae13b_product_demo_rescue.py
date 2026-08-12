"""CLI: AE13B Real Demo Product Rescue package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ae13b_product.run import run_ae13b_product_demo_rescue


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="AE13B Real Demo Product Rescue")
    p.add_argument("--project-root", default=str(PROJECT_ROOT))
    p.add_argument("--demo-cycles", type=int, default=3)
    p.add_argument("--output-root", default=None)
    args = p.parse_args(argv)
    summary = run_ae13b_product_demo_rescue(
        project_root=Path(args.project_root),
        demo_cycles=int(args.demo_cycles),
        output_root=Path(args.output_root) if args.output_root else None,
        runtime_stopped_before_editing=True,
    )
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
