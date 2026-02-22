"""Evaluation metrics for sentence-level probe outputs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import auc, precision_recall_curve


def precision_recall_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute area under the precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    return float(auc(recall, precision))


def top_k_recall_per_problem(
    frame: pd.DataFrame,
    *,
    score_col: str,
    true_importance_col: str,
    problem_col: str = "problem_id",
    k: int,
) -> float:
    """Compute average top-k overlap recall across problems."""
    recalls: list[float] = []
    grouped = frame.groupby(problem_col, sort=False)
    for _, group in grouped:
        if len(group) == 0:
            continue
        k_eff = min(k, len(group))
        true_top = set(group.nlargest(k_eff, true_importance_col).index.tolist())
        pred_top = set(group.nlargest(k_eff, score_col).index.tolist())
        overlap = len(true_top.intersection(pred_top))
        recalls.append(overlap / float(k_eff))

    if not recalls:
        return float("nan")
    return float(np.mean(recalls))


def mean_problem_spearman(
    frame: pd.DataFrame,
    *,
    score_col: str,
    true_importance_col: str,
    problem_col: str = "problem_id",
) -> float:
    """Compute mean Spearman correlation across problems."""
    corrs: list[float] = []
    grouped = frame.groupby(problem_col, sort=False)
    for _, group in grouped:
        if len(group) < 2:
            continue
        corr, _ = spearmanr(group[score_col].to_numpy(), group[true_importance_col].to_numpy())
        if not np.isnan(corr):
            corrs.append(float(corr))
    if not corrs:
        return float("nan")
    return float(np.mean(corrs))


def evaluate_frame(
    frame: pd.DataFrame,
    *,
    score_col: str,
    label_col: str = "anchor",
    true_importance_col: str = "importance_score",
    problem_col: str = "problem_id",
    k_values: list[int] | tuple[int, ...] = (5, 10),
) -> dict[str, Any]:
    """Compute all key evaluation metrics for one prediction frame."""
    y_true = frame[label_col].to_numpy(dtype=np.int64)
    y_score = frame[score_col].to_numpy(dtype=np.float32)

    metrics: dict[str, Any] = {
        "pr_auc": precision_recall_auc(y_true, y_score),
        "spearman_mean": mean_problem_spearman(
            frame,
            score_col=score_col,
            true_importance_col=true_importance_col,
            problem_col=problem_col,
        ),
    }

    for k in k_values:
        metrics[f"top_{k}_recall"] = top_k_recall_per_problem(
            frame,
            score_col=score_col,
            true_importance_col=true_importance_col,
            problem_col=problem_col,
            k=k,
        )

    return metrics
