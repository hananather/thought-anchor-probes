#!/usr/bin/env python
"""Train and evaluate sentence-level Thought Anchor probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.config import ensure_parent_dirs, load_config
from ta_probe.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/experiment.yaml", help="Path to experiment config"
    )
    parser.add_argument(
        "--no-tripwires",
        action="store_true",
        help="Skip random-label and overfit-one-problem checks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_parent_dirs(config)

    metrics = run_training(
        metadata_path=config.paths.metadata_parquet,
        embeddings_memmap_path=config.paths.embeddings_memmap,
        embeddings_shape_path=config.paths.embeddings_shape_json,
        splits_path=config.paths.splits_json,
        metrics_output_path=config.paths.metrics_json,
        predictions_output_path=config.paths.predictions_parquet,
        random_seed=config.training.random_seed,
        k_values=config.training.k_values,
        mlp_hidden_dim=config.training.mlp_hidden_dim,
        mlp_max_iter=config.training.mlp_max_iter,
        run_tripwires=not args.no_tripwires,
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
