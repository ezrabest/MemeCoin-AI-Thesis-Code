#!/usr/bin/env python3
"""Print row counts and latest timestamp per SQLite table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.database import get_collection_debug_status, init_pool


def main() -> None:
    init_pool()
    print(json.dumps(get_collection_debug_status(), indent=2))


if __name__ == "__main__":
    main()
