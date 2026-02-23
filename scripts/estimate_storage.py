#!/usr/bin/env python
"""Estimate activation-cache disk usage from extracted metadata and shape payload."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.storage import estimate_embedding_storage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape-json",
        required=True,
        help="Path to sentence_embeddings_shape.json",
    )
    parser.add_argument(
        "--metadata-parquet",
        required=True,
        help="Path to sentence_metadata.parquet with token_length or token_count",
    )
    parser.add_argument(
        "--metadata-row-overhead-bytes",
        type=int,
        default=192,
        help="Estimated parquet metadata bytes per sentence row.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with Path(args.shape_json).open("r", encoding="utf-8") as handle:
        shape_payload = json.load(handle)
    metadata = pd.read_parquet(args.metadata_parquet)

    raw_layout = str(shape_payload.get("embedding_layout", "")).strip().lower()
    if raw_layout not in {"sentence_mean", "token_ragged"}:
        pooling_mode = str(shape_payload.get("pooling", "")).strip().lower()
        if pooling_mode == "tokens":
            raw_layout = "token_ragged"
        elif pooling_mode == "mean":
            raw_layout = "sentence_mean"
        elif "token_length" in metadata.columns:
            raw_layout = "token_ragged"
        else:
            raw_layout = "sentence_mean"

    token_column = ""
    if "token_length" in metadata.columns:
        token_column = "token_length"
    elif "token_count" in metadata.columns:
        token_column = "token_count"

    total_tokens: int | None = None
    if token_column:
        total_tokens = int(metadata[token_column].to_numpy(dtype=np.int64).sum())
    elif raw_layout == "token_ragged":
        total_tokens = int(metadata.shape[0])

    estimate = estimate_embedding_storage(
        embedding_layout=raw_layout,  # type: ignore[arg-type]
        hidden_dim=int(shape_payload["hidden_dim"]),
        storage_dtype_name=str(shape_payload.get("dtype", "float32")),
        num_sentences=int(metadata.shape[0]),
        total_tokens=total_tokens,
        metadata_row_overhead_bytes=int(args.metadata_row_overhead_bytes),
    )
    estimate["shape_json"] = str(Path(args.shape_json))
    estimate["metadata_parquet"] = str(Path(args.metadata_parquet))
    estimate["token_count_column"] = token_column or (
        "row_count" if raw_layout == "token_ragged" else "none"
    )
    estimate["embedding_rows_basis"] = (
        "num_sentences" if raw_layout == "sentence_mean" else (token_column or "row_count")
    )
    print(json.dumps(estimate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
