from __future__ import annotations

import numpy as np
import pandas as pd

from ta_probe.train import _build_residual_anchors, _fit_baseline_and_residuals


def _make_frame(problem_id: int, chunk_idx: list[int], target: list[float]) -> pd.DataFrame:
    num_chunks = [max(chunk_idx) + 1] * len(chunk_idx)
    relative_position = [idx / (num_chunks[0] - 1) if num_chunks[0] > 1 else 0.0 for idx in chunk_idx]
    return pd.DataFrame(
        {
            "problem_id": [problem_id] * len(chunk_idx),
            "chunk_idx": chunk_idx,
            "num_chunks": num_chunks,
            "relative_position": relative_position,
            "token_count": [10] * len(chunk_idx),
            "chunk_text": ["x"] * len(chunk_idx),
            "importance_score": target,
        }
    )


def test_residuals_fit_on_train_only() -> None:
    train = _make_frame(1, [0, 1], [0.0, 1.0])
    val = _make_frame(2, [0, 1], [100.0, 100.0])
    test = _make_frame(3, [0, 1], [100.0, 100.0])

    residuals = _fit_baseline_and_residuals(
        train_frame=train,
        val_frame=val,
        test_frame=test,
        target_col="importance_score",
        residualize_against="position",
        random_seed=0,
    )

    train_hat = residuals["train_hat"]
    assert np.max(np.abs(train_hat - np.array([0.0, 1.0], dtype=np.float32))) < 5.0


def test_residual_baseline_is_invariant_to_val_test_shift() -> None:
    train = _make_frame(1, [0, 1, 2], [0.0, 0.5, 1.0])
    val_a = _make_frame(2, [0, 1, 2], [10.0, 11.0, 12.0])
    test_a = _make_frame(3, [0, 1, 2], [20.0, 21.0, 22.0])
    val_b = _make_frame(2, [0, 1, 2], [1000.0, 1000.0, 1000.0])
    test_b = _make_frame(3, [0, 1, 2], [-1000.0, -1000.0, -1000.0])

    residuals_a = _fit_baseline_and_residuals(
        train_frame=train,
        val_frame=val_a,
        test_frame=test_a,
        target_col="importance_score",
        residualize_against="position",
        random_seed=0,
    )
    residuals_b = _fit_baseline_and_residuals(
        train_frame=train,
        val_frame=val_b,
        test_frame=test_b,
        target_col="importance_score",
        residualize_against="position",
        random_seed=0,
    )

    assert np.allclose(residuals_a["train_hat"], residuals_b["train_hat"])
    assert np.allclose(residuals_a["train_residual"], residuals_b["train_residual"])


def test_residual_anchor_thresholds_by_problem() -> None:
    frame = pd.concat(
        [
            _make_frame(1, [0, 1, 2, 3], [0.0, 1.0, 2.0, 3.0]),
            _make_frame(2, [0, 1, 2, 3], [10.0, 11.0, 12.0, 13.0]),
        ],
        ignore_index=True,
    )
    frame["residual_target"] = frame["importance_score"].to_numpy(dtype=np.float32)

    anchors = _build_residual_anchors(frame, residual_col="residual_target", percentile=50.0)
    frame["residual_anchor"] = anchors

    counts = frame.groupby("problem_id")["residual_anchor"].sum().to_dict()
    assert counts[1] == 2
    assert counts[2] == 2
