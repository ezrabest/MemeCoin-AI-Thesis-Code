"""Re-apply AE18 resilience statuses to the runtime canonical identity index.

Offline maintenance only — never invoked from a UI GET path. Performs no
provider network calls: statuses are derived from already-indexed fields plus
the read-only last-good / manual-override display caches.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.clean_forward.display_resilience import apply_display_resilience  # noqa: E402
from app.clean_forward.runtime_identity_index import (  # noqa: E402
    load_runtime_identity_index,
    write_runtime_index,
)

#: The 4 provider URLs confirmed unresolvable via the official DexScreener API.
PROBED_UNRESOLVED = {
    "https://dexscreener.com/robinhood/0x9c2905076ad86335e0CB8227fd5D0e5Bec795f1A",
    "https://dexscreener.com/robinhood/0xb3F901859ACbEF2288E187993AA50911A5404762",
    "https://dexscreener.com/base/0x2db51152Dd4F7a00c10e181401e18B9d6269e4b4",
    "https://dexscreener.com/robinhood/0xEA63b938967e65B2D71d99Bc8cFD9c4cB3c7c105",
}


def main() -> int:
    loaded = load_runtime_identity_index()
    rows = loaded.get("rows") or []
    if not rows:
        print("runtime index empty; nothing to refresh")
        return 1

    for row in rows:
        exact = str(row.get("provider_pair_url_exact") or "").strip()
        if exact in PROBED_UNRESOLVED:
            row["provider_probe_attempted"] = True
        apply_display_resilience(
            row,
            allow_cache_lookup=True,
            provider_probe_attempted=exact in PROBED_UNRESOLVED,
        )

    write_runtime_index(rows)
    print("rows:", len(rows), "written: True")
    for field in (
        "display_metadata_status",
        "provider_resolution_status",
        "symbol_resolution_status",
        "market_data_status",
        "identity_readiness_status",
        "trade_readiness_status",
    ):
        print(f"  {field}: {dict(Counter(str(r.get(field) or '') for r in rows))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
