#!/usr/bin/env python3
"""Validate data/runtime/manual_display_overrides.csv before any overrides are applied.

Fail-closed: rejected rows are never applied and are recorded in the audit.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.clean_forward.manual_display_overrides import (  # noqa: E402
    OVERRIDE_CSV_PATH,
    VALIDATION_AUDIT_PATH,
    ensure_override_csv_template,
    validate_manual_display_overrides,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=OVERRIDE_CSV_PATH,
        help="Path to manual_display_overrides.csv",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=VALIDATION_AUDIT_PATH,
        help="Path to write validation audit JSON",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Mark valid rows as applied in the audit (still never applies rejected rows)",
    )
    args = parser.parse_args()

    ensure_override_csv_template()
    result = validate_manual_display_overrides(
        csv_path=args.csv, audit_path=args.audit, apply=args.apply
    )
    print(json.dumps({k: result[k] for k in result if k != "valid_overrides"}, indent=2))
    # Exit 0 when validation contract holds (rejected rows OK if not applied).
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
