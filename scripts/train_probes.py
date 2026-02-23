#!/usr/bin/env python
"""Train and evaluate sentence-level Thought Anchor probes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.config import ensure_parent_dirs, load_config, resolve_embeddings_memmap_path
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

    if config.split.strategy != "single_split":
        raise ValueError(
            "train_probes.py only supports split.strategy=single_split. "
            "Use scripts/run_lopo_cv.py for LOPO evaluation."
        )

    random_seed = config.training.random_seed if args.seed is None else int(args.seed)
    embeddings_memmap_path = resolve_embeddings_memmap_path(config)
    metrics_output_path = with_run_suffix(config.paths.metrics_json, args.run_name)
    predictions_output_path = with_run_suffix(config.paths.predictions_parquet, args.run_name)

    metrics = run_training(
        metadata_path=config.paths.metadata_parquet,
        embeddings_memmap_path=embeddings_memmap_path,
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
        expected_vertical_attention_mode=config.activations.vertical_attention.mode,
        expected_vertical_attention_depth_control=(
            config.activations.vertical_attention.depth_control
        ),
        expected_vertical_attention_light_last_n_tokens=(
            config.activations.vertical_attention.light_last_n_tokens
        ),
        expected_vertical_attention_full_max_seq_len=(
            config.activations.vertical_attention.full_max_seq_len
        ),
        token_probe_heads=config.training.token_probe_heads,
        token_probe_mlp_width=config.training.token_probe_mlp_width,
        token_probe_mlp_depth=config.training.token_probe_mlp_depth,
        token_probe_batch_size=config.training.token_probe_batch_size,
        token_probe_max_epochs=config.training.token_probe_max_epochs,
        token_probe_patience=config.training.token_probe_patience,
        token_probe_learning_rate=config.training.token_probe_learning_rate,
        token_probe_weight_decay=config.training.token_probe_weight_decay,
        token_probe_continuous_loss=config.training.token_probe_continuous_loss,
        token_probe_device=config.training.token_probe_device,
        run_tripwires=not args.no_tripwires,
        run_name=args.run_name,
        target_mode=config.labels.target_mode,
        residualize_against=config.training.residualize_against,
    )

    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
