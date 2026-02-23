from __future__ import annotations

import pandas as pd

from ta_probe.metrics import (
    bootstrap_pr_auc_delta_by_group,
    evaluate_frame,
    pr_auc_by_position_bins,
    top_k_recall_per_problem,
)


def test_top_k_recall_per_problem() -> None:
    frame = pd.DataFrame(
        {
            "problem_id": [1, 1, 1, 2, 2, 2],
            "score": [0.9, 0.8, 0.1, 0.7, 0.2, 0.1],
            "importance_score": [0.95, 0.7, 0.1, 0.6, 0.5, 0.1],
        }
    )

    value = top_k_recall_per_problem(
        frame,
        score_col="score",
        true_importance_col="importance_score",
        k=1,
    )
    assert 0.99 <= value <= 1.0


def test_evaluate_frame_keys() -> None:
    frame = pd.DataFrame(
        {
            "problem_id": [1, 1, 2, 2],
            "pred_score": [0.9, 0.1, 0.8, 0.2],
            "anchor": [1, 0, 1, 0],
            "importance_score": [0.8, 0.1, 0.7, 0.2],
        }
    )

    metrics = evaluate_frame(frame, score_col="pred_score", k_values=[1, 2])
    assert "pr_auc" in metrics
    assert "top_1_recall" in metrics
    assert "top_2_recall" in metrics


def test_pr_auc_by_position_bins_returns_fixed_number_of_bins() -> None:
    frame = pd.DataFrame(
        {
            "relative_position": [0.0, 0.2, 0.49, 0.5, 0.8, 1.0],
            "anchor": [0, 1, 0, 1, 0, 1],
            "pred_score": [0.1, 0.9, 0.3, 0.8, 0.2, 0.7],
        }
    )
    bins = pr_auc_by_position_bins(frame, score_col="pred_score", n_bins=5)
    assert len(bins) == 5
    assert sum(int(item["count"]) for item in bins) == len(frame)


def test_bootstrap_pr_auc_delta_by_group_returns_ci_payload() -> None:
    frame = pd.DataFrame(
        {
            "problem_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "anchor": [1, 0, 0, 1, 0, 0, 1, 0, 0],
            "score_a": [0.9, 0.2, 0.1, 0.85, 0.15, 0.2, 0.88, 0.2, 0.1],
            "score_b": [0.8, 0.3, 0.2, 0.7, 0.2, 0.3, 0.72, 0.3, 0.2],
        }
    )
    result = bootstrap_pr_auc_delta_by_group(
        frame,
        score_col_a="score_a",
        score_col_b="score_b",
        n_bootstrap=200,
        random_seed=0,
    )
    assert result["n_bootstrap"] == 200
    assert result["n_valid_bootstrap"] > 0
    assert "ci_low" in result and "ci_high" in result
