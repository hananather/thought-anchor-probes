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


def with_run_suffix(path: str, run_name: str | None) -> str:
    """Add a run suffix to an output file path."""
    if not run_name:
        return path
    output_path = Path(path)
    suffix = output_path.suffix
    stem = output_path.stem
    return str(output_path.with_name(f"{stem}_{run_name}{suffix}"))


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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed override for probe training.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run label used to suffix output file names.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_parent_dirs(config)

    random_seed = config.training.random_seed if args.seed is None else int(args.seed)
    metrics_output_path = with_run_suffix(config.paths.metrics_json, args.run_name)
    predictions_output_path = with_run_suffix(config.paths.predictions_parquet, args.run_name)

    metrics = run_training(
        metadata_path=config.paths.metadata_parquet,
        embeddings_memmap_path=config.paths.embeddings_memmap,
        embeddings_shape_path=config.paths.embeddings_shape_json,
        splits_path=config.paths.splits_json,
        metrics_output_path=metrics_output_path,
        predictions_output_path=predictions_output_path,
        random_seed=random_seed,
        k_values=config.training.k_values,
        mlp_hidden_dim=config.training.mlp_hidden_dim,
        mlp_max_iter=config.training.mlp_max_iter,
        bootstrap_iterations=config.training.bootstrap_iterations,
        bootstrap_seed=config.training.bootstrap_seed,
        position_bins=config.training.position_bins,
        expected_counterfactual_field=config.labels.counterfactual_field,
        expected_anchor_percentile=config.labels.anchor_percentile,
        expected_drop_last_chunk=config.labels.drop_last_chunk,
        expected_model_name_or_path=config.activations.model_name_or_path,
        expected_pooling=config.activations.pooling,
        expected_layer_mode=config.activations.layer_mode,
        expected_requested_layer_index=config.activations.layer_index,
        expected_repo_id=config.dataset.repo_id,
        expected_model_dir=config.dataset.model_dir,
        expected_temp_dir=config.dataset.temp_dir,
        expected_split_dir=config.dataset.split_dir,
        expected_compute_dtype=config.activations.dtype,
        run_tripwires=not args.no_tripwires,
        run_name=args.run_name,
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
