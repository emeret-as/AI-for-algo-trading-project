#!/usr/bin/env python3
"""
CLI entry point for the volatility infrastructure pipeline.

Usage:
  python scripts/run_pipeline.py --date 2026-04-06 --underlying SPY --spot 450
  python scripts/run_pipeline.py --date 2026-04-06 --dry-run
  python scripts/run_pipeline.py --replay --start 2026-04-01 --end 2026-04-06
"""

import argparse
import json
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.orchestration.jobs import run_full_pipeline
from src.orchestration.replay import replay_date, replay_date_range


def parse_args():
    parser = argparse.ArgumentParser(
        description="Volatility Infrastructure Pipeline CLI"
    )
    parser.add_argument(
        "--date", type=str, default="2026-04-06",
        help="Trade date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--underlying", type=str, default="SPY",
        help="Underlying symbol"
    )
    parser.add_argument(
        "--spot", type=float, default=450.0,
        help="Underlying spot price"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Root data directory"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Do not write outputs"
    )
    parser.add_argument(
        "--regime", type=str, default="normal",
        choices=["calm", "normal", "stressed", "sparse"],
        help="Synthetic market regime"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--replay", action="store_true",
        help="Run in replay mode"
    )
    parser.add_argument(
        "--start", type=str,
        help="Replay start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end", type=str,
        help="Replay end date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--output-json", type=str,
        help="Write results to JSON file"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    config = {
        "regime": args.regime,
        "seed": args.seed,
    }

    if args.replay:
        if args.start and args.end:
            results = replay_date_range(
                args.start, args.end,
                args.underlying, data_dir=args.data_dir, config=config,
            )
        else:
            results = [replay_date(
                args.date, args.underlying, args.spot,
                args.data_dir, config,
            )]
    else:
        results = run_full_pipeline(
            trade_date_str=args.date,
            underlying_symbol=args.underlying,
            spot=args.spot,
            data_dir=args.data_dir,
            config=config,
            dry_run=args.dry_run,
        )

    # Print summary
    if isinstance(results, dict):
        _print_summary(results)
    elif isinstance(results, list):
        for r in results:
            _print_summary(r)

    # Optionally write JSON
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults written to {args.output_json}")


def _print_summary(results: dict):
    """Print a human-readable pipeline summary."""
    print("\n" + "=" * 60)
    print(f"Pipeline Summary: {results.get('trade_date', 'N/A')}")
    print("=" * 60)

    for key in ["underlying", "universe_size", "raw_events",
                 "snapshot_options", "qc_accepted", "qc_rejected",
                 "forward_count", "iv_total", "iv_converged",
                 "surface_slices", "svi_fits", "priced_contracts",
                 "risk_positions", "scenario_count"]:
        if key in results:
            print(f"  {key:25s}: {results[key]}")

    if "worst_scenario" in results:
        print(f"  {'worst_scenario':25s}: {results['worst_scenario']} "
              f"(PnL: {results.get('worst_pnl', 0):+,.2f})")

    qc = results.get("qc_report", {}).get("summary", {})
    if qc:
        print(f"\n  QC Summary: {qc.get('pass', 0)} pass | "
              f"{qc.get('warn', 0)} warn | {qc.get('fail', 0)} fail")

    failures = results.get("qc_report", {}).get("failures", [])
    if failures:
        print("\n  QC Failures:")
        for f in failures[:5]:
            print(f"    - {f['check']}: {f['target']} ({f['reason']})")

    print("=" * 60)


if __name__ == "__main__":
    main()
