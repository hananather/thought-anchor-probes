"""Label utilities for counterfactual sentence importance."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def build_problem_label_frame(
    *,
    problem_id: int,
    chunks_labeled: list[dict[str, Any]],
    counterfactual_field: str = "counterfactual_importance_accuracy",
    percentile: float = 90.0,
    drop_last_chunk: bool = True,
) -> pd.DataFrame:
    """Build sentence-level labels for one problem."""
    usable_chunks = (
        chunks_labeled[:-1] if drop_last_chunk and len(chunks_labeled) > 1 else chunks_labeled
    )

    raw_scores = np.array([float(chunk.get(counterfactual_field, 0.0)) for chunk in usable_chunks])
    scores = np.abs(raw_scores)

    threshold = float(np.percentile(scores, percentile)) if len(scores) else 0.0
    anchors = (scores >= threshold).astype(int)

    rows: list[dict[str, Any]] = []
    n_chunks = len(usable_chunks)
    for idx, (chunk, raw_score, score, anchor) in enumerate(
        zip(usable_chunks, raw_scores, scores, anchors, strict=True)
    ):
        tags = chunk.get("function_tags") or []
        function_tag = tags[0] if tags else "unknown"

        rows.append(
            {
                "problem_id": problem_id,
                "chunk_idx": idx,
                "num_chunks": n_chunks,
                "chunk_text": chunk.get("chunk", ""),
                "function_tag": function_tag,
                "counterfactual_raw": float(raw_score),
                "importance_score": float(score),
                "anchor": int(anchor),
            }
        )

    return pd.DataFrame(rows)


def combine_problem_label_frames(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Merge per-problem frames into one stable frame."""
    merged = pd.concat(list(frames), ignore_index=True)
    return merged.sort_values(["problem_id", "chunk_idx"], ignore_index=True)
