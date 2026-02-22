from __future__ import annotations

from ta_probe.data_loading import create_splits


def test_create_splits_handles_small_problem_count() -> None:
    problem_ids = [1, 2, 3, 4, 5]
    splits = create_splits(
        problem_ids,
        train_fraction=0.7,
        val_fraction=0.15,
        test_fraction=0.15,
        seed=0,
    )

    assert len(splits["train"]) == 3
    assert len(splits["val"]) == 1
    assert len(splits["test"]) == 1
