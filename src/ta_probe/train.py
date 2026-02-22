"""Training and evaluation pipeline for sentence-level thought-anchor probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ta_probe.metrics import evaluate_frame, precision_recall_auc
from ta_probe.models import (
    build_position_features,
    make_linear_probe,
    make_mlp_probe,
    make_position_baseline,
    predict_scores,
)


def _load_splits(path: str | Path) -> dict[str, list[int]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_shape(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_embeddings(memmap_path: str | Path, shape_path: str | Path) -> np.memmap:
    shape = _load_shape(shape_path)
    return np.memmap(
        memmap_path,
        dtype=np.float32,
        mode="r",
        shape=(shape["rows"], shape["hidden_dim"]),
    )


def _subset_by_problem_ids(frame: pd.DataFrame, problem_ids: list[int]) -> pd.DataFrame:
    return frame[frame["problem_id"].isin(problem_ids)].copy().reset_index(drop=True)


def _embedding_features(frame: pd.DataFrame, embeddings: np.memmap) -> np.ndarray:
    rows = frame["embedding_row"].to_numpy(dtype=np.int64)
    return np.asarray(embeddings[rows], dtype=np.float32)


def _fit_and_score(
    *,
    model,
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    test_x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model.fit(train_x, train_y)
    val_scores = predict_scores(model, val_x)
    test_scores = predict_scores(model, test_x)
    return val_scores, test_scores


def run_training(
    *,
    metadata_path: str | Path,
    embeddings_memmap_path: str | Path,
    embeddings_shape_path: str | Path,
    splits_path: str | Path,
    metrics_output_path: str | Path,
    predictions_output_path: str | Path,
    random_seed: int,
    k_values: list[int],
    mlp_hidden_dim: int,
    mlp_max_iter: int,
    run_tripwires: bool = True,
) -> dict[str, Any]:
    """Train baseline and probe models, then write metrics and predictions."""
    metadata = pd.read_parquet(metadata_path)
    embeddings = _load_embeddings(embeddings_memmap_path, embeddings_shape_path)
    splits = _load_splits(splits_path)

    train_frame = _subset_by_problem_ids(metadata, splits["train"])
    val_frame = _subset_by_problem_ids(metadata, splits["val"])
    test_frame = _subset_by_problem_ids(metadata, splits["test"])

    if train_frame.empty or val_frame.empty or test_frame.empty:
        msg = "Split contains empty partition. Rebuild splits with more problems."
        raise ValueError(msg)

    train_y = train_frame["anchor"].to_numpy(dtype=np.int64)

    train_pos_x = build_position_features(train_frame)
    val_pos_x = build_position_features(val_frame)
    test_pos_x = build_position_features(test_frame)

    train_emb_x = _embedding_features(train_frame, embeddings)
    val_emb_x = _embedding_features(val_frame, embeddings)
    test_emb_x = _embedding_features(test_frame, embeddings)

    models = {
        "position_baseline": make_position_baseline(random_seed),
        "linear_probe": make_linear_probe(random_seed),
        "mlp_probe": make_mlp_probe(mlp_hidden_dim, mlp_max_iter, random_seed),
    }

    model_inputs = {
        "position_baseline": (train_pos_x, val_pos_x, test_pos_x),
        "linear_probe": (train_emb_x, val_emb_x, test_emb_x),
        "mlp_probe": (train_emb_x, val_emb_x, test_emb_x),
    }

    val_scores: dict[str, np.ndarray] = {}
    test_scores: dict[str, np.ndarray] = {}

    for name, model in models.items():
        train_x, val_x, test_x = model_inputs[name]
        cur_val_scores, cur_test_scores = _fit_and_score(
            model=model,
            train_x=train_x,
            train_y=train_y,
            val_x=val_x,
            test_x=test_x,
        )
        val_scores[name] = cur_val_scores
        test_scores[name] = cur_test_scores

    val_metrics: dict[str, Any] = {}
    test_metrics: dict[str, Any] = {}

    for name in models:
        val_eval = val_frame.copy()
        val_eval["pred_score"] = val_scores[name]
        test_eval = test_frame.copy()
        test_eval["pred_score"] = test_scores[name]

        val_metrics[name] = evaluate_frame(
            val_eval,
            score_col="pred_score",
            k_values=k_values,
        )
        test_metrics[name] = evaluate_frame(
            test_eval,
            score_col="pred_score",
            k_values=k_values,
        )

    predictions = test_frame.copy()
    for name in models:
        predictions[f"score_{name}"] = test_scores[name]

    tripwires: dict[str, Any] = {}
    if run_tripwires:
        tripwires = run_tripwire_checks(
            train_frame=train_frame,
            test_frame=test_frame,
            train_emb_x=train_emb_x,
            test_emb_x=test_emb_x,
            random_seed=random_seed,
        )

    metrics_payload = {
        "val": val_metrics,
        "test": test_metrics,
        "tripwires": tripwires,
        "splits": {key: len(value) for key, value in splits.items()},
        "rows": {
            "train": len(train_frame),
            "val": len(val_frame),
            "test": len(test_frame),
        },
    }

    metrics_path = Path(metrics_output_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, sort_keys=True)

    predictions_path = Path(predictions_output_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(predictions_path, index=False)

    return metrics_payload


def run_tripwire_checks(
    *,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    train_emb_x: np.ndarray,
    test_emb_x: np.ndarray,
    random_seed: int,
) -> dict[str, Any]:
    """Run quick tests that catch silent data/label bugs."""
    result: dict[str, Any] = {}

    rng = np.random.default_rng(random_seed)

    shuffled_labels = train_frame[["problem_id", "anchor"]].copy()
    shuffled_labels["anchor"] = (
        shuffled_labels.groupby("problem_id", group_keys=False)["anchor"]
        .apply(lambda series: pd.Series(rng.permutation(series.to_numpy()), index=series.index))
        .to_numpy(dtype=np.int64)
    )

    random_model = make_linear_probe(random_seed)
    random_model.fit(train_emb_x, shuffled_labels["anchor"].to_numpy(dtype=np.int64))
    shuffled_scores = predict_scores(random_model, test_emb_x)
    chance_pr_auc = precision_recall_auc(
        test_frame["anchor"].to_numpy(dtype=np.int64), shuffled_scores
    )

    prevalence = float(test_frame["anchor"].mean())
    result["random_label_test"] = {
        "test_pr_auc": chance_pr_auc,
        "test_prevalence": prevalence,
        "near_chance": bool(chance_pr_auc <= prevalence + 0.1),
    }

    one_problem = int(train_frame["problem_id"].iloc[0])
    subset = train_frame[train_frame["problem_id"] == one_problem].copy()
    subset_rows = subset["embedding_row"].to_numpy(dtype=np.int64)
    subset_x = train_emb_x[
        np.isin(train_frame["embedding_row"].to_numpy(dtype=np.int64), subset_rows)
    ]
    subset_y = subset["anchor"].to_numpy(dtype=np.int64)

    if len(subset) >= 5 and len(np.unique(subset_y)) > 1:
        overfit_model = make_mlp_probe(hidden_dim=50, max_iter=500, random_seed=random_seed)
        overfit_model.fit(subset_x, subset_y)
        train_scores = predict_scores(overfit_model, subset_x)
        train_pr_auc = precision_recall_auc(subset_y, train_scores)
        result["overfit_one_problem_test"] = {
            "problem_id": one_problem,
            "train_pr_auc": train_pr_auc,
            "can_memorize": bool(train_pr_auc >= 0.9),
        }
    else:
        result["overfit_one_problem_test"] = {
            "problem_id": one_problem,
            "skipped": True,
            "reason": "Not enough samples or only one class in selected problem",
        }

    return result
