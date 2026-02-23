#!/usr/bin/env python
"""Evaluate a saved prediction score column."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.eval import evaluate_prediction_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Path to predictions parquet")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    parser.add_argument("--score-column", required=True, help="Score column name")
    parser.add_argument("--k-values", nargs="+", type=int, default=[5, 10], help="Top-k values")
    parser.add_argument(
        "--split",
        default="test",
        help=(
            "Optional split filter when predictions include multiple splits. "
            "Use 'none' to disable."
        ),
    )
    parser.add_argument(
        "--split-column",
        default="split",
        help="Split column name used for filtering.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_prediction_file(
        predictions_path=args.predictions,
        output_path=args.output,
        score_column=args.score_column,
        k_values=args.k_values,
        split=None if str(args.split).lower() == "none" else args.split,
        split_column=args.split_column,
    )
    print(metrics)


if __name__ == "__main__":
    main()
