from __future__ import annotations

import numpy as np
import pandas as pd

from ta_probe.deferrals import evaluate_deferral_policy, plan_deferrals, tune_deferral_thresholds


def test_evaluate_deferral_policy_counts_and_rates() -> None:
    scores = np.array([0.1, 0.4, 0.7, 0.9], dtype=np.float32)
    labels = np.array([0, 0, 1, 1], dtype=np.int64)
    metrics = evaluate_deferral_policy(
        scores=scores,
        labels=labels,
        negative_threshold=0.2,
        positive_threshold=0.8,
    )

    assert metrics["accepted_count"] == 2
    assert metrics["deferred_count"] == 2
    assert np.isclose(metrics["deferral_rate"], 0.5)
    assert np.isclose(metrics["accepted_error"], 0.0)


def test_tune_deferral_thresholds_budget_respects_budget() -> None:
    val_scores = np.array([0.05, 0.2, 0.45, 0.55, 0.8, 0.95], dtype=np.float32)
    val_labels = np.array([0, 0, 1, 0, 1, 1], dtype=np.int64)
    tuned = tune_deferral_thresholds(
        validation_scores=val_scores,
        validation_labels=val_labels,
        objective="budget",
        deferral_budget=0.35,
        target_accepted_error=None,
        grid_size=31,
    )

    assert tuned["meets_objective"] is True
    assert tuned["selection"]["deferral_rate"] <= 0.35 + 1e-9
    assert tuned["selection"]["negative_threshold"] <= tuned["selection"]["positive_threshold"]


def test_plan_deferrals_grouped_projection() -> None:
    validation = pd.DataFrame(
        {
            "fold_id": [0, 0, 0, 1, 1, 1],
            "score_model": [0.05, 0.25, 0.9, 0.1, 0.45, 0.85],
            "anchor": [0, 0, 1, 0, 1, 1],
        }
    )
    test = pd.DataFrame(
        {
            "fold_id": [0, 0, 0, 1, 1, 1],
            "score_model": [0.1, 0.5, 0.92, 0.12, 0.52, 0.9],
            "anchor": [0, 1, 1, 0, 0, 1],
        }
    )

    planned = plan_deferrals(
        validation_frame=validation,
        test_frame=test,
        score_column="score_model",
        label_column="anchor",
        objective="error",
        deferral_budget=None,
        target_accepted_error=0.0,
        group_column="fold_id",
        grid_size=31,
    )

    assert planned["num_groups"] == 2
    assert planned["group_column"] == "fold_id"
    assert 0.0 <= planned["test_projection_summary"]["deferral_rate"] <= 1.0
    assert planned["groups"][0]["thresholds"]["negative"] <= planned["groups"][0]["thresholds"][
        "positive"
    ]
