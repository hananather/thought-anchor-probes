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


def pr_auc_by_position_bins(
    frame: pd.DataFrame,
    *,
    score_col: str,
    label_col: str = "anchor",
    position_col: str = "relative_position",
    n_bins: int = 5,
) -> list[dict[str, Any]]:
    """Compute PR AUC in uniform position bins over relative position [0, 1]."""
    if n_bins <= 0:
        msg = "n_bins must be positive"
        raise ValueError(msg)
    if position_col not in frame.columns:
        msg = f"Missing required position column: {position_col}"
        raise ValueError(msg)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    position_values = frame[position_col].to_numpy(dtype=np.float32)
    scores = frame[score_col].to_numpy(dtype=np.float32)
    labels = frame[label_col].to_numpy(dtype=np.int64)

    results: list[dict[str, Any]] = []
    for idx in range(n_bins):
        left = float(edges[idx])
        right = float(edges[idx + 1])
        if idx == n_bins - 1:
            mask = (position_values >= left) & (position_values <= right)
        else:
            mask = (position_values >= left) & (position_values < right)
        count = int(mask.sum())
        if count == 0:
            auc_value = float("nan")
            prevalence = float("nan")
        else:
            bin_labels = labels[mask]
            bin_scores = scores[mask]
            prevalence = float(bin_labels.mean())
            if np.unique(bin_labels).size < 2:
                auc_value = float("nan")
            else:
                auc_value = precision_recall_auc(bin_labels, bin_scores)
        results.append(
            {
                "bin_index": idx,
                "left_inclusive": left,
                "right_inclusive": right,
                "count": count,
                "prevalence": prevalence,
                "pr_auc": float(auc_value),
            }
        )
    return results


def bootstrap_pr_auc_delta_by_group(
    frame: pd.DataFrame,
    *,
    score_col_a: str,
    score_col_b: str,
    label_col: str = "anchor",
    group_col: str = "problem_id",
    n_bootstrap: int = 1000,
    random_seed: int = 0,
) -> dict[str, Any]:
    """Bootstrap PR AUC deltas by resampling groups with replacement."""
    if n_bootstrap <= 0:
        msg = "n_bootstrap must be positive"
        raise ValueError(msg)
    if group_col not in frame.columns:
        msg = f"Missing required group column: {group_col}"
        raise ValueError(msg)

    grouped_frames = [group.copy() for _, group in frame.groupby(group_col, sort=False)]
    n_groups = len(grouped_frames)
    if n_groups == 0:
        msg = "Cannot bootstrap with no groups"
        raise ValueError(msg)

    rng = np.random.default_rng(random_seed)
    deltas: list[float] = []

    for _ in range(n_bootstrap):
        sampled_indices = rng.integers(0, n_groups, size=n_groups)
        sampled = pd.concat([grouped_frames[idx] for idx in sampled_indices], ignore_index=True)
        y_true = sampled[label_col].to_numpy(dtype=np.int64)
        if np.unique(y_true).size < 2:
            continue

        scores_a = sampled[score_col_a].to_numpy(dtype=np.float32)
        scores_b = sampled[score_col_b].to_numpy(dtype=np.float32)
        auc_a = precision_recall_auc(y_true, scores_a)
        auc_b = precision_recall_auc(y_true, scores_b)
        deltas.append(float(auc_a - auc_b))

    if not deltas:
        return {
            "score_a": score_col_a,
            "score_b": score_col_b,
            "delta_mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_bootstrap": int(n_bootstrap),
            "n_valid_bootstrap": 0,
            "group_col": group_col,
        }

    delta_arr = np.asarray(deltas, dtype=np.float64)
    return {
        "score_a": score_col_a,
        "score_b": score_col_b,
        "delta_mean": float(delta_arr.mean()),
        "ci_low": float(np.quantile(delta_arr, 0.025)),
        "ci_high": float(np.quantile(delta_arr, 0.975)),
        "n_bootstrap": int(n_bootstrap),
        "n_valid_bootstrap": int(len(delta_arr)),
        "group_col": group_col,
    }
