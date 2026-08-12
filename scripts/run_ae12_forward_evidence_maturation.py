"""CLI: AE12 forward-evidence maturation (derived audit outputs only)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ae12_forward_evidence.maturation import OutputRootExistsError, run_forward_evidence_maturation
from app.ae12_forward_evidence.types import (
    DEFAULT_MISSED_WINNER_THRESHOLDS,
    HORIZON_SECONDS,
    Ae12RunConfig,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="AE12.3–AE12.4 Forward Evidence Maturation (read-only vs runtime writers)"
    )
    p.add_argument("--output-root", type=str, default=None, help="Explicit audit output root")
    p.add_argument("--resume", action="store_true", help="Resume into existing output root without duplication")
    p.add_argument(
        "--fail-if-output-exists",
        action="store_true",
        default=True,
        help="Fail closed if output root exists and --resume not set (default)",
    )
    p.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="Do not fail if output exists (still prefer --resume for idempotency)",
    )
    p.add_argument("--max-rows", type=int, default=None, help="Max opportunity capture rows to process")
    p.add_argument(
        "--horizons",
        nargs="+",
        default=None,
        choices=list(HORIZON_SECONDS.keys()),
        help="Horizons to mature",
    )
    p.add_argument(
        "--missed-winner-thresholds-default",
        action="store_true",
        help="Use default missed-winner thresholds",
    )
    p.add_argument("--no-external-apis", action="store_true", default=True)
    p.add_argument("--no-real-wallet", action="store_true", default=True)
    p.add_argument("--db-path", type=str, default=None, help="Optional trader.db path (read-only)")
    p.add_argument("--project-root", type=str, default=str(PROJECT_ROOT))
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    horizons = args.horizons or list(HORIZON_SECONDS.keys())
    thresholds = dict(DEFAULT_MISSED_WINNER_THRESHOLDS)
    fail_if = True
    if args.allow_existing_output:
        fail_if = False

    config = Ae12RunConfig(
        project_root=Path(args.project_root),
        output_root=Path(args.output_root) if args.output_root else None,
        resume=bool(args.resume),
        fail_if_output_exists=fail_if,
        max_rows=args.max_rows,
        horizons=list(horizons),
        missed_winner_thresholds=thresholds,
        no_external_apis=bool(args.no_external_apis),
        no_real_wallet=bool(args.no_real_wallet),
        db_path=Path(args.db_path) if args.db_path else None,
    )
    try:
        summary = run_forward_evidence_maturation(config)
    except OutputRootExistsError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise

    print("AE12 forward evidence maturation complete")
    print(f"output_root={summary.get('output_root')}")
    print(f"candidate_evidence_row_count={summary.get('candidate_evidence_row_count')}")
    print(f"missed_winner_count={summary.get('missed_winner_count')}")
    print(f"missing_reason_count={summary.get('missing_reason_count')}")
    print(f"wallet_safety={((summary.get('wallet_safety') or {}).get('audit_status'))}")
    print(f"readiness={((summary.get('readiness_gate') or {}).get('status'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
