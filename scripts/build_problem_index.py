#!/usr/bin/env python
"""List dataset problem IDs, sample N, and write train/val/test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.config import ensure_parent_dirs, load_config
from ta_probe.data_loading import create_splits, list_problem_ids, sample_problem_ids, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/experiment.yaml", help="Path to experiment config"
    )
    parser.add_argument("--refresh", action="store_true", help="Refresh cached problem IDs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_parent_dirs(config)

    problem_ids = list_problem_ids(
        repo_id=config.dataset.repo_id,
        model_dir=config.dataset.model_dir,
        temp_dir=config.dataset.temp_dir,
        split_dir=config.dataset.split_dir,
        cache_path=config.paths.problem_ids_json,
        force_refresh=args.refresh,
    )

    sampled_ids = sample_problem_ids(problem_ids, config.dataset.num_problems, config.dataset.seed)
    splits = create_splits(
        sampled_ids,
        train_fraction=config.training.train_fraction,
        val_fraction=config.training.val_fraction,
        test_fraction=config.training.test_fraction,
        seed=config.training.random_seed,
    )

    write_json(config.paths.problem_ids_json, sampled_ids)
    write_json(config.paths.splits_json, splits)

    print(f"Total available problems: {len(problem_ids)}")
    print(f"Sampled problems: {len(sampled_ids)}")
    print(f"Train/Val/Test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")


if __name__ == "__main__":
    main()
