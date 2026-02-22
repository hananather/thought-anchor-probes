from __future__ import annotations

import pandas as pd

from ta_probe.metrics import evaluate_frame, top_k_recall_per_problem


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
