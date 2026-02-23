#!/usr/bin/env python
"""Plan two-threshold deferral policies from validation/test prediction scores."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.deferrals import plan_deferrals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions",
        nargs="+",
        default=None,
        help=(
            "One or more prediction parquet files containing both validation and test rows "
            "with a split column."
        ),
    )
    parser.add_argument(
        "--validation",
        nargs="+",
        default=None,
        help="Optional validation parquet files (used instead of --predictions split filtering).",
    )
    parser.add_argument(
        "--test",
        nargs="+",
        default=None,
        help="Optional test parquet files (used instead of --predictions split filtering).",
    )
    parser.add_argument("--score-column", required=True, help="Score column to threshold.")
    parser.add_argument("--label-column", default="anchor", help="Binary label column.")
    parser.add_argument("--split-column", default="split", help="Split column name.")
    parser.add_argument("--val-split", default="val", help="Validation split name.")
    parser.add_argument("--test-split", default="test", help="Test split name.")
    parser.add_argument(
        "--group-column",
        default=None,
        help="Optional fold/group column for per-fold threshold fitting and projection.",
    )
    parser.add_argument(
        "--objective",
        choices=["budget", "error"],
        default="budget",
        help="Optimization objective for threshold selection.",
    )
    parser.add_argument(
        "--deferral-budget",
        type=float,
        default=0.2,
        help="Maximum allowed deferral rate on validation when objective=budget.",
    )
    parser.add_argument(
        "--target-accepted-error",
        type=float,
        default=None,
        help="Target accepted-set error when objective=error.",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=101,
        help="Number of quantile points used to build threshold candidates.",
    )
    parser.add_argument("--output", required=True, help="Output JSON path.")
    return parser.parse_args()


def _load_and_concat(paths: list[str]) -> pd.DataFrame:
    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _resolve_frames(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    has_split_inputs = bool(args.predictions)
    has_explicit_inputs = bool(args.validation) or bool(args.test)
    if has_split_inputs and has_explicit_inputs:
        msg = "Use either --predictions OR (--validation and --test), not both."
        raise ValueError(msg)

    if has_split_inputs:
        frame = _load_and_concat(args.predictions)
        if frame.empty:
            msg = "No rows loaded from --predictions."
            raise ValueError(msg)
        if args.split_column not in frame.columns:
            msg = f"Missing split column '{args.split_column}' in --predictions input."
            raise ValueError(msg)
        validation = frame[frame[args.split_column] == args.val_split].copy()
        test = frame[frame[args.split_column] == args.test_split].copy()
        if validation.empty or test.empty:
            msg = (
                "Predictions input did not contain both requested splits: "
                f"val={args.val_split}, test={args.test_split}."
            )
            raise ValueError(msg)
        return validation, test

    if not args.validation or not args.test:
        msg = "Provide --predictions or both --validation and --test."
        raise ValueError(msg)

    validation = _load_and_concat(args.validation)
    test = _load_and_concat(args.test)
    if validation.empty or test.empty:
        msg = "Validation/test input files must contain at least one row."
        raise ValueError(msg)
    return validation, test


def main() -> None:
    args = parse_args()
    validation_frame, test_frame = _resolve_frames(args)

    output_payload = plan_deferrals(
        validation_frame=validation_frame,
        test_frame=test_frame,
        score_column=args.score_column,
        label_column=args.label_column,
        objective=args.objective,
        deferral_budget=args.deferral_budget if args.objective == "budget" else None,
        target_accepted_error=(
            args.target_accepted_error if args.objective == "error" else None
        ),
        group_column=args.group_column,
        grid_size=args.grid_size,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2, sort_keys=True)

    print(json.dumps(output_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
