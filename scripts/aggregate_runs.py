#!/usr/bin/env python
"""Aggregate multi-seed metrics for one run root."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.aggregate import aggregate_run_metrics


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = aggregate_run_metrics(
        run_root=args.run_root,
        output_json_path=args.output_json,
        output_md_path=args.output_md,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
