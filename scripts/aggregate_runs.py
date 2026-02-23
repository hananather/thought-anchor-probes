#!/usr/bin/env python
"""Aggregate multi-seed metrics for one run root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.aggregate import aggregate_lopo_metrics, aggregate_run_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, help="Directory with per-seed metrics files")
    parser.add_argument(
        "--output-json",
        default=None,
        help=(
            "Optional path for aggregated JSON output. "
            "Defaults to <run-root>/aggregate_metrics.json."
        ),
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help=(
            "Optional path for aggregated markdown output. "
            "Defaults to <run-root>/aggregate_metrics.md."
        ),
    )
    parser.add_argument(
        "--best-of-k",
        type=int,
        default=1,
        help="Select top-k seeds by validation primary metric for best-of-k reporting.",
    )
    parser.add_argument(
        "--lopo",
        action="store_true",
        help="Aggregate LOPO folds instead of single-split seed metrics.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Bootstrap iterations for LOPO fold-level deltas.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Bootstrap seed for LOPO fold-level deltas.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.lopo:
        summary = aggregate_lopo_metrics(
            run_root=args.run_root,
            output_json_path=args.output_json,
            output_md_path=args.output_md,
            best_of_k=args.best_of_k,
            bootstrap_iterations=args.bootstrap_iterations,
            bootstrap_seed=args.bootstrap_seed,
        )
    else:
        summary = aggregate_run_metrics(
            run_root=args.run_root,
            output_json_path=args.output_json,
            output_md_path=args.output_md,
            best_of_k=args.best_of_k,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
