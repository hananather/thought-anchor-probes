"""Training and evaluation pipeline for sentence-level thought-anchor probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ta_probe.metrics import (
    bootstrap_pr_auc_delta_by_group,
    evaluate_frame,
    pr_auc_by_position_bins,
    precision_recall_auc,
)
from ta_probe.models import (
    build_position_features,
    build_text_features,
    make_linear_probe,
    make_mlp_probe,
    make_position_baseline,
    make_text_baseline,
    predict_scores,
)


def _load_splits(path: str | Path) -> dict[str, list[int]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_shape(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_embedding_numpy_dtype(dtype_name: str) -> np.dtype:
    mapping = {
        "float16": np.float16,
        "float32": np.float32,
    }
    if dtype_name not in mapping:
        msg = f"Unsupported embedding dtype in shape payload: {dtype_name}"
        raise ValueError(msg)
    return np.dtype(mapping[dtype_name])


def _load_embeddings(memmap_path: str | Path, shape: dict[str, Any]) -> np.memmap:
    dtype_name = str(shape.get("dtype", "float32"))
    memmap_dtype = _resolve_embedding_numpy_dtype(dtype_name)
    return np.memmap(
        memmap_path,
        dtype=memmap_dtype,
        mode="r",
        shape=(shape["rows"], shape["hidden_dim"]),
    )


def _validate_embedding_provenance(
    shape_payload: dict[str, Any],
    *,
    expected_counterfactual_field: str | None,
    expected_anchor_percentile: float | None,
    expected_drop_last_chunk: bool | None,
    expected_model_name_or_path: str | None,
    expected_pooling: str | None,
    expected_layer_mode: str | None,
    expected_requested_layer_index: int | None,
    expected_repo_id: str | None,
    expected_model_dir: str | None,
    expected_temp_dir: str | None,
    expected_split_dir: str | None,
    expected_compute_dtype: str | None,
) -> None:
    """Fail fast when training config does not match extraction-time cache metadata.

    This prevents stale-cache bugs where users change label/model config but reuse
    old embeddings and labels.
    """
    expected_values = {
        "counterfactual_field": expected_counterfactual_field,
        "anchor_percentile": expected_anchor_percentile,
        "drop_last_chunk": expected_drop_last_chunk,
        "model_name_or_path": expected_model_name_or_path,
        "pooling": expected_pooling,
        "layer_mode": expected_layer_mode,
        "requested_layer_index": expected_requested_layer_index,
        "repo_id": expected_repo_id,
        "model_dir": expected_model_dir,
        "temp_dir": expected_temp_dir,
        "split_dir": expected_split_dir,
        "compute_dtype": expected_compute_dtype,
    }

    for key, expected in expected_values.items():
        if expected is None:
            continue

        if key not in shape_payload:
            msg = (
                f"Embedding cache missing provenance key '{key}'. "
                "Re-run scripts/extract_embeddings.py with the current code."
            )
            raise ValueError(msg)

        cached = shape_payload[key]
        if isinstance(expected, float):
            try:
                matches = bool(np.isclose(float(cached), expected, rtol=0.0, atol=1e-9))
            except (TypeError, ValueError):
                matches = False
        else:
            matches = cached == expected

        if not matches:
            msg = (
                f"Embedding cache provenance mismatch for '{key}': "
                f"expected={expected!r}, cached={cached!r}. "
                "Re-run scripts/extract_embeddings.py to rebuild artifacts."
            )
            raise ValueError(msg)


def _validate_split_coverage(metadata: pd.DataFrame, splits: dict[str, list[int]]) -> None:
    """Ensure every split problem ID has extracted rows in metadata."""
    available_problem_ids = set(metadata["problem_id"].to_numpy(dtype=np.int64).tolist())
    missing: dict[str, list[int]] = {}
    for split_name, problem_ids in splits.items():
        missing_ids = sorted(
            int(problem_id)
            for problem_id in problem_ids
            if problem_id not in available_problem_ids
        )
        if missing_ids:
            missing[split_name] = missing_ids

    if missing:
        summary = "; ".join(f"{name}={ids}" for name, ids in sorted(missing.items()))
        msg = (
            "Metadata is missing split problem IDs: "
            f"{summary}. "
            "Re-run scripts/extract_embeddings.py without skipping those problems."
        )
        raise ValueError(msg)


def _subset_by_problem_ids(frame: pd.DataFrame, problem_ids: list[int]) -> pd.DataFrame:
    return frame[frame["problem_id"].isin(problem_ids)].copy().reset_index(drop=True)


def _embedding_features(frame: pd.DataFrame, embeddings: np.memmap) -> np.ndarray:
    rows = frame["embedding_row"].to_numpy(dtype=np.int64)
    return np.asarray(embeddings[rows], dtype=np.float32)


def _concat_features(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.concatenate([left, right], axis=1).astype(np.float32, copy=False)


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
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int | None = None,
    position_bins: int = 5,
    expected_counterfactual_field: str | None = None,
    expected_anchor_percentile: float | None = None,
    expected_drop_last_chunk: bool | None = None,
    expected_model_name_or_path: str | None = None,
    expected_pooling: str | None = None,
    expected_layer_mode: str | None = None,
    expected_requested_layer_index: int | None = None,
    expected_repo_id: str | None = None,
    expected_model_dir: str | None = None,
    expected_temp_dir: str | None = None,
    expected_split_dir: str | None = None,
    expected_compute_dtype: str | None = None,
    run_tripwires: bool = True,
    run_name: str | None = None,
) -> dict[str, Any]:
    """Train baseline and probe models, then write metrics and predictions."""
    metadata = pd.read_parquet(metadata_path)
    shape_payload = _load_shape(embeddings_shape_path)
    _validate_embedding_provenance(
        shape_payload,
        expected_counterfactual_field=expected_counterfactual_field,
        expected_anchor_percentile=expected_anchor_percentile,
        expected_drop_last_chunk=expected_drop_last_chunk,
        expected_model_name_or_path=expected_model_name_or_path,
        expected_pooling=expected_pooling,
        expected_layer_mode=expected_layer_mode,
        expected_requested_layer_index=expected_requested_layer_index,
        expected_repo_id=expected_repo_id,
        expected_model_dir=expected_model_dir,
        expected_temp_dir=expected_temp_dir,
        expected_split_dir=expected_split_dir,
        expected_compute_dtype=expected_compute_dtype,
    )
    embeddings = _load_embeddings(embeddings_memmap_path, shape_payload)
    splits = _load_splits(splits_path)
    _validate_split_coverage(metadata, splits)

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

    train_text_x = build_text_features(train_frame)
    val_text_x = build_text_features(val_frame)
    test_text_x = build_text_features(test_frame)

    train_emb_x = _embedding_features(train_frame, embeddings)
    val_emb_x = _embedding_features(val_frame, embeddings)
    test_emb_x = _embedding_features(test_frame, embeddings)

    train_emb_pos_x = _concat_features(train_emb_x, train_pos_x)
    val_emb_pos_x = _concat_features(val_emb_x, val_pos_x)
    test_emb_pos_x = _concat_features(test_emb_x, test_pos_x)

    models = {
        "position_baseline": make_position_baseline(random_seed),
        "text_only_baseline": make_text_baseline(random_seed),
        "linear_probe": make_linear_probe(random_seed),
        "mlp_probe": make_mlp_probe(mlp_hidden_dim, mlp_max_iter, random_seed),
        "activations_plus_position": make_linear_probe(random_seed),
    }

    model_inputs = {
        "position_baseline": (train_pos_x, val_pos_x, test_pos_x),
        "text_only_baseline": (train_text_x, val_text_x, test_text_x),
        "linear_probe": (train_emb_x, val_emb_x, test_emb_x),
        "mlp_probe": (train_emb_x, val_emb_x, test_emb_x),
        "activations_plus_position": (train_emb_pos_x, val_emb_pos_x, test_emb_pos_x),
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
    val_position_bin_metrics: dict[str, Any] = {}
    test_position_bin_metrics: dict[str, Any] = {}

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
        if position_bins > 0:
            val_position_bin_metrics[name] = pr_auc_by_position_bins(
                val_eval,
                score_col="pred_score",
                n_bins=position_bins,
            )

        test_metrics[name] = evaluate_frame(
            test_eval,
            score_col="pred_score",
            k_values=k_values,
        )
        if position_bins > 0:
            test_position_bin_metrics[name] = pr_auc_by_position_bins(
                test_eval,
                score_col="pred_score",
                n_bins=position_bins,
            )

    predictions = test_frame.copy()
    for name in models:
        predictions[f"score_{name}"] = test_scores[name]

    effective_bootstrap_seed = random_seed if bootstrap_seed is None else int(bootstrap_seed)
    confidence_intervals: dict[str, Any] = {}

    ci_comparisons = [
        ("score_activations_plus_position", "score_position_baseline"),
        ("score_activations_plus_position", "score_text_only_baseline"),
    ]
    for score_a, score_b in ci_comparisons:
        if score_a not in predictions.columns or score_b not in predictions.columns:
            continue
        ci_key = f"{score_a}_minus_{score_b}"
        ci_payload = bootstrap_pr_auc_delta_by_group(
            predictions,
            score_col_a=score_a,
            score_col_b=score_b,
            n_bootstrap=bootstrap_iterations,
            random_seed=effective_bootstrap_seed,
            group_col="problem_id",
        )
        point_a = float(test_metrics["activations_plus_position"]["pr_auc"])
        if score_b == "score_position_baseline":
            point_b = float(test_metrics["position_baseline"]["pr_auc"])
        elif score_b == "score_text_only_baseline":
            point_b = float(test_metrics["text_only_baseline"]["pr_auc"])
        else:
            point_b = float("nan")
        ci_payload["point_delta"] = float(point_a - point_b)
        confidence_intervals[ci_key] = ci_payload

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
        "run_name": run_name,
        "seed": int(random_seed),
        "val": val_metrics,
        "test": test_metrics,
        "position_bin_metrics": {
            "val": val_position_bin_metrics,
            "test": test_position_bin_metrics,
            "num_bins": int(position_bins),
        },
        "confidence_intervals": confidence_intervals,
        "tripwires": tripwires,
        "splits": {key: len(value) for key, value in splits.items()},
        "rows": {
            "train": len(train_frame),
            "val": len(val_frame),
            "test": len(test_frame),
        },
        # Persist extraction-time metadata in metrics for lightweight auditability.
        "cache_provenance": {
            key: shape_payload.get(key)
            for key in [
                "counterfactual_field",
                "anchor_percentile",
                "drop_last_chunk",
                "model_name_or_path",
                "pooling",
                "layer_mode",
                "requested_layer_index",
                "repo_id",
                "model_dir",
                "temp_dir",
                "split_dir",
                "compute_dtype",
                "layer_index",
                "dtype",
            ]
            if key in shape_payload
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
