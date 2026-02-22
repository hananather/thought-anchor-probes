#!/usr/bin/env python
"""Run span integrity checks for one problem."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from transformers import AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.config import load_config
from ta_probe.data_loading import load_problem_metadata
from ta_probe.spans import get_whitebox_example_data, span_integrity_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/experiment.yaml", help="Path to experiment config"
    )
    parser.add_argument("--problem-id", type=int, required=True, help="Problem ID to inspect")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of random sentences")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    problem_data = load_problem_metadata(
        problem_id=args.problem_id,
        repo_id=config.dataset.repo_id,
        model_dir=config.dataset.model_dir,
        temp_dir=config.dataset.temp_dir,
        split_dir=config.dataset.split_dir,
    )

    text, sentences, _ = get_whitebox_example_data(problem_data)
    tokenizer = AutoTokenizer.from_pretrained(config.activations.model_name_or_path, use_fast=True)

    report = span_integrity_report(
        text=text,
        sentences=sentences,
        tokenizer=tokenizer,
        sample_size=args.sample_size,
        seed=config.training.random_seed,
    )

    pass_rate = float(report["pass"].mean()) if len(report) else 0.0
    print(f"Pass rate: {pass_rate:.3f}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
