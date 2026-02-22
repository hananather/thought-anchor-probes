"""One-problem label verification utilities."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def normalize_answer(answer: str) -> str:
    """Normalize numeric answers to a compact comparable form."""
    allowed = set("0123456789.-")
    cleaned = "".join(char for char in str(answer) if char in allowed)
    return cleaned.strip()


def extract_answer_from_cot(cot: str) -> str:
    """Extract a boxed numeric answer from a full CoT string."""
    match = re.search(r"\\boxed\{([^}]*)\}", cot)
    if match:
        return normalize_answer(match.group(1))

    numbers = re.findall(r"-?\d+(?:\.\d+)?", cot)
    if not numbers:
        return ""
    return normalize_answer(numbers[-1])


def calculate_answer_importance(full_cot_list: list[list[str]], answer: str) -> list[float]:
    """Compute chunk-wise accuracy deltas from rollout lists."""
    answer_norm = normalize_answer(answer)

    probabilities: list[float] = []
    for rollout_group in full_cot_list:
        if len(rollout_group) == 0:
            probabilities.append(0.0)
            continue
        correct = sum(extract_answer_from_cot(cot) == answer_norm for cot in rollout_group)
        probabilities.append(correct / len(rollout_group))

    if len(probabilities) < 2:
        return []
    return np.diff(probabilities).astype(np.float32).tolist()


def calculate_importance_from_correctness(correctness_by_chunk: list[list[bool]]) -> list[float]:
    """Compute chunk-wise accuracy deltas from boolean correctness lists."""
    probabilities: list[float] = []
    for correctness in correctness_by_chunk:
        if len(correctness) == 0:
            probabilities.append(0.0)
            continue
        probabilities.append(float(np.mean(np.array(correctness, dtype=np.float32))))

    if len(probabilities) < 2:
        return []
    return np.diff(probabilities).astype(np.float32).tolist()


def calculate_counterfactual_importance(
    *,
    chunks_removed: list[str],
    chunks_resampled: list[list[str]],
    correctness_by_chunk: list[list[bool]],
    threshold: float = 0.8,
    min_samples: int = 5,
    embedding_model: Any,
) -> list[float]:
    """Compute counterfactual importance using semantic filtering."""
    if embedding_model is None:
        msg = "embedding_model is required"
        raise ValueError(msg)

    filtered_probabilities: list[float | None] = []
    for original_chunk, resampled_chunks, correctness_group in zip(
        chunks_removed,
        chunks_resampled,
        correctness_by_chunk,
        strict=True,
    ):
        emb_original = embedding_model.encode(
            [original_chunk],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )[0]
        emb_resampled = embedding_model.encode(
            resampled_chunks,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        cosine_similarities = emb_resampled @ emb_original
        indices = np.where(cosine_similarities < threshold)[0]

        if len(indices) < min_samples:
            filtered_probabilities.append(None)
            continue

        correct = 0
        for idx in indices:
            if idx >= len(correctness_group):
                continue
            if correctness_group[idx]:
                correct += 1
        filtered_probabilities.append(correct / len(indices))

    smooth_probs = (
        pd.Series(filtered_probabilities, dtype=float).ffill().bfill().fillna(0.0).to_list()
    )
    if len(smooth_probs) < 2:
        return []
    return np.diff(smooth_probs).astype(np.float32).tolist()


def verify_problem_importance(
    *,
    problem_data: dict[str, Any],
    counterfactual_threshold: float = 0.8,
    counterfactual_min_samples: int = 5,
    compute_counterfactual: bool = False,
    embedding_model_name: str = "all-MiniLM-L6-v2",
) -> dict[str, float | bool]:
    """Recompute metrics for one problem and compare with precomputed labels."""
    chunks_labeled = problem_data["chunks_labeled"]
    correctness_resampled = [
        [bool(rollout.get("is_correct", False)) for rollout in chunk_rollouts]
        for chunk_rollouts in problem_data["chunk_solutions"]
    ]

    resampling = calculate_importance_from_correctness(correctness_resampled)
    precomputed_resampling = [
        float(chunk.get("resampling_importance_accuracy", 0.0)) for chunk in chunks_labeled[:-1]
    ]
    resampling_avg_diff = float(np.abs(np.subtract(resampling, precomputed_resampling)).mean())

    result: dict[str, float | bool] = {
        "resampling_avg_diff": resampling_avg_diff,
        "resampling_pass": bool(resampling_avg_diff < 0.01),
    }

    if compute_counterfactual:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            msg = "Install optional verify deps: pip install -e '.[verify]'"
            raise ImportError(msg) from exc

        model = SentenceTransformer(embedding_model_name)
        chunks_removed = [chunk["chunk"] for chunk in chunks_labeled]
        chunks_resampled = [
            [rollout["chunk_resampled"] for rollout in chunk_rollouts]
            for chunk_rollouts in problem_data["chunk_solutions"]
        ]

        counterfactual = calculate_counterfactual_importance(
            chunks_removed=chunks_removed,
            chunks_resampled=chunks_resampled,
            correctness_by_chunk=correctness_resampled,
            threshold=counterfactual_threshold,
            min_samples=counterfactual_min_samples,
            embedding_model=model,
        )
        precomputed_counterfactual = [
            -float(chunk.get("counterfactual_importance_accuracy", 0.0))
            for chunk in chunks_labeled[:-1]
        ]
        counterfactual_avg_diff = float(
            np.abs(np.subtract(counterfactual, precomputed_counterfactual)).mean()
        )
        result["counterfactual_avg_diff"] = counterfactual_avg_diff
        result["counterfactual_pass"] = bool(counterfactual_avg_diff < 0.025)

    return result
