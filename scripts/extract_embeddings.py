#!/usr/bin/env python
"""Extract one-layer sentence embeddings and cache them to disk."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.activations import extract_and_cache_embeddings
from ta_probe.config import ensure_parent_dirs, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/experiment.yaml", help="Path to experiment config"
    )
    parser.add_argument(
        "--problem-ids",
        default=None,
        help="Optional JSON path with explicit problem IDs. Default uses splits union.",
    )
    return parser.parse_args()


def load_problem_ids(config_path: str, explicit_path: str | None) -> list[int]:
    if explicit_path is not None:
        with Path(explicit_path).open("r", encoding="utf-8") as handle:
            ids = list(json.load(handle))
        return sorted({int(problem_id) for problem_id in ids})

    with Path(config_path).open("r", encoding="utf-8") as handle:
        splits = json.load(handle)
    all_ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
    return sorted(int(problem_id) for problem_id in all_ids)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    ensure_parent_dirs(config)

    problem_ids = load_problem_ids(config.paths.splits_json, args.problem_ids)

    result = extract_and_cache_embeddings(
        problem_ids=problem_ids,
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
        embeddings_memmap_path=config.paths.embeddings_memmap,
        embeddings_shape_path=config.paths.embeddings_shape_json,
        metadata_path=config.paths.metadata_parquet,
    )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
