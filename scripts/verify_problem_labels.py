#!/usr/bin/env python
"""Recompute resampling and optional counterfactual labels for one problem."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.config import load_config
from ta_probe.data_loading import load_problem_full
from ta_probe.verification import verify_problem_importance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/experiment.yaml", help="Path to experiment config"
    )
    parser.add_argument("--problem-id", type=int, required=True, help="Problem ID to verify")
    parser.add_argument(
        "--counterfactual",
        action="store_true",
        help="Also recompute counterfactual importance (requires sentence-transformers)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    problem_data = load_problem_full(
        problem_id=args.problem_id,
        repo_id=config.dataset.repo_id,
        model_dir=config.dataset.model_dir,
        temp_dir=config.dataset.temp_dir,
        split_dir=config.dataset.split_dir,
        forced_split_dir=config.dataset.forced_split_dir,
        load_forced=False,
        verbose=True,
    )

    result = verify_problem_importance(
        problem_data=problem_data,
        compute_counterfactual=args.counterfactual,
    )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
