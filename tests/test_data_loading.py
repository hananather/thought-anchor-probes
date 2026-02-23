from __future__ import annotations

from ta_probe.data_loading import create_lopo_folds, create_splits


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


def test_create_lopo_folds_deterministic() -> None:
    problem_ids = [5, 3, 1, 2, 4]
    folds_a = create_lopo_folds(problem_ids, val_fraction=0.25)
    folds_b = create_lopo_folds(problem_ids, val_fraction=0.25)
    assert folds_a == folds_b


def test_create_lopo_folds_counts() -> None:
    problem_ids = [1, 2, 3, 4]
    folds = create_lopo_folds(problem_ids, val_fraction=0.25)
    assert len(folds) == 4
    for fold_id, split in folds.items():
        assert split["test"] == [fold_id]
        assert len(split["val"]) == 1
        assert len(split["train"]) == 2
