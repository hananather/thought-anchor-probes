from __future__ import annotations

from ta_probe.labels import build_problem_label_frame


def test_build_problem_label_frame_uses_absolute_scores() -> None:
    chunks = [
        {"chunk": "a", "function_tags": ["x"], "counterfactual_importance_accuracy": -0.9},
        {"chunk": "b", "function_tags": ["y"], "counterfactual_importance_accuracy": 0.1},
        {"chunk": "c", "function_tags": ["z"], "counterfactual_importance_accuracy": -0.2},
    ]

    frame = build_problem_label_frame(
        problem_id=1,
        chunks_labeled=chunks,
        percentile=90,
        drop_last_chunk=False,
    )

    assert frame.loc[0, "importance_score"] == 0.9
    assert frame.loc[0, "anchor"] == 1
    assert frame.loc[1, "anchor"] == 0


def test_build_problem_label_frame_drops_last_chunk() -> None:
    chunks = [
        {"chunk": "a", "function_tags": ["x"], "counterfactual_importance_accuracy": 0.9},
        {"chunk": "b", "function_tags": ["x"], "counterfactual_importance_accuracy": 0.7},
        {"chunk": "c", "function_tags": ["x"], "counterfactual_importance_accuracy": 0.1},
    ]

    frame = build_problem_label_frame(
        problem_id=1,
        chunks_labeled=chunks,
        percentile=90,
        drop_last_chunk=True,
    )

    assert len(frame) == 2
    assert frame["chunk_idx"].tolist() == [0, 1]
