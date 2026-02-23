#!/usr/bin/env python
"""Run Leave-One-Problem-Out cross-validation for probe evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.activations import extract_and_cache_embeddings
from ta_probe.aggregate import aggregate_lopo_metrics
from ta_probe.config import ensure_parent_dirs, load_config
from ta_probe.data_loading import create_lopo_folds, list_problem_ids, sample_problem_ids, write_json
from ta_probe.train import run_training


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/experiment.yaml", help="Path to experiment config"
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Run name used under artifacts/runs/<run-name>.",
    )
    parser.add_argument(
        "--run-root",
        default=None,
        help="Optional explicit run root. Overrides --run-name.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Training seeds to run per fold.",
    )
    parser.add_argument(
        "--problem-ids",
        default=None,
        help="Optional JSON path with explicit problem IDs.",
    )
    parser.add_argument(
        "--max-problems",
        type=int,
        default=None,
        help="Optional cap on number of problems (useful for CI pilots).",
    )
    parser.add_argument(
        "--refresh-problem-list",
        action="store_true",
        help="Refresh cached problem listing from Hugging Face.",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Skip problems that fail extraction and log them instead of failing.",
    )
    parser.add_argument(
        "--failure-log",
        default=None,
        help="Optional path to write extraction failure details as JSON.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse existing extraction artifacts when cache provenance matches config.",
    )
    parser.add_argument(
        "--no-tripwires",
        action="store_true",
        help="Skip random-label and overfit-one-problem checks.",
    )
    return parser.parse_args()


def _load_problem_ids(path: str | Path) -> list[int]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return sorted({int(problem_id) for problem_id in json.load(handle)})


def _resolve_run_root(args: argparse.Namespace, config: Any) -> Path:
    if args.run_root is not None:
        return Path(args.run_root)
    run_name = args.run_name
    if not run_name:
        run_name = f"{config.dataset.model_dir}__{config.dataset.split_dir}__lopo"
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "artifacts" / "runs" / run_name


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_parent_dirs(config)

    if config.split.strategy != "lopo_cv":
        raise ValueError(
            "run_lopo_cv.py requires split.strategy=lopo_cv in the config. "
            "Update the config or use scripts/train_probes.py for single splits."
        )

    run_root = _resolve_run_root(args, config)
    run_root.mkdir(parents=True, exist_ok=True)

    if args.problem_ids is not None:
        available_ids = _load_problem_ids(args.problem_ids)
    else:
        cache_path = run_root / "problem_ids_cache.json"
        available_ids = list_problem_ids(
            repo_id=config.dataset.repo_id,
            model_dir=config.dataset.model_dir,
            temp_dir=config.dataset.temp_dir,
            split_dir=config.dataset.split_dir,
            cache_path=cache_path,
            force_refresh=args.refresh_problem_list,
        )

    max_allowed = config.dataset.num_problems
    if args.max_problems is not None:
        max_allowed = min(max_allowed, int(args.max_problems))

    sampled_ids = sample_problem_ids(available_ids, num_problems=max_allowed, seed=config.dataset.seed)
    if not sampled_ids:
        raise RuntimeError("No problem IDs available for LOPO run")

    folds = create_lopo_folds(sampled_ids, val_fraction=config.training.val_fraction)

    write_json(run_root / "problem_ids.json", sampled_ids)
    write_json(run_root / "folds.json", folds)

    failure_log = args.failure_log
    if failure_log is None:
        failure_log = str(run_root / "extraction_failures.json")

    extract_and_cache_embeddings(
        problem_ids=sampled_ids,
        repo_id=config.dataset.repo_id,
        model_dir=config.dataset.model_dir,
        temp_dir=config.dataset.temp_dir,
        split_dir=config.dataset.split_dir,
        model_name_or_path=config.activations.model_name_or_path,
        layer_mode=config.activations.layer_mode,
        layer_index=config.activations.layer_index,
        counterfactual_field=config.labels.counterfactual_field,
        anchor_percentile=config.labels.anchor_percentile,
        drop_last_chunk=config.labels.drop_last_chunk,
        device=config.activations.device,
        dtype_name=config.activations.dtype,
        pooling=config.activations.pooling,
        storage_dtype_name=config.activations.storage_dtype,
        embeddings_memmap_path=config.paths.embeddings_memmap,
        embeddings_shape_path=config.paths.embeddings_shape_json,
        metadata_path=config.paths.metadata_parquet,
        skip_failed_problems=args.skip_failed,
        failure_log_path=failure_log,
        reuse_cache_if_valid=args.reuse_cache,
    )

    for fold_id, split in folds.items():
        fold_dir = run_root / f"fold_{int(fold_id)}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        splits_path = fold_dir / "splits.json"
        write_json(splits_path, split)

        for seed in args.seeds:
            seed_dir = fold_dir / str(int(seed))
            seed_dir.mkdir(parents=True, exist_ok=True)

            metrics_path = seed_dir / "metrics.json"
            predictions_path = seed_dir / "predictions.parquet"

            run_training(
                metadata_path=config.paths.metadata_parquet,
                embeddings_memmap_path=config.paths.embeddings_memmap,
                embeddings_shape_path=config.paths.embeddings_shape_json,
                splits_path=splits_path,
                metrics_output_path=metrics_path,
                predictions_output_path=predictions_path,
                random_seed=int(seed),
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
                run_name=f"fold_{int(fold_id)}_seed_{int(seed)}",
            )

    bootstrap_seed = (
        config.training.random_seed
        if config.training.bootstrap_seed is None
        else int(config.training.bootstrap_seed)
    )

    aggregate_lopo_metrics(
        run_root=run_root,
        best_of_k=config.training.best_of_k,
        bootstrap_iterations=config.training.bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )

    summary = {
        "run_root": str(run_root),
        "num_folds": len(folds),
        "num_seeds": len(args.seeds),
        "aggregate_json": str(run_root / "aggregate_lopo_metrics.json"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
