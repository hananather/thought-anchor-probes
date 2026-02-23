"""Threshold planning for cascade/deferral policies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


def evaluate_deferral_policy(
    *,
    scores: np.ndarray,
    labels: np.ndarray,
    negative_threshold: float,
    positive_threshold: float,
) -> dict[str, Any]:
    """Evaluate a three-way policy: negative / positive / defer."""
    if negative_threshold > positive_threshold:
        msg = "negative_threshold must be <= positive_threshold"
        raise ValueError(msg)

    score_values = np.asarray(scores, dtype=np.float32)
    label_values = np.asarray(labels, dtype=np.int64)
    if score_values.shape[0] != label_values.shape[0]:
        msg = "scores and labels must have the same length"
        raise ValueError(msg)
    if score_values.size == 0:
        msg = "Cannot evaluate deferral policy on empty arrays"
        raise ValueError(msg)

    confident_negative = score_values <= float(negative_threshold)
    confident_positive = score_values >= float(positive_threshold)
    accepted = confident_negative | confident_positive
    deferred = ~accepted

    pred = np.zeros_like(label_values)
    pred[confident_positive] = 1

    accepted_count = int(accepted.sum())
    deferred_count = int(deferred.sum())
    total_count = int(score_values.shape[0])
    deferral_rate = float(deferred_count / total_count)
    accepted_coverage = float(accepted_count / total_count)

    if accepted_count == 0:
        accepted_error = float("nan")
        accepted_accuracy = float("nan")
    else:
        accepted_error = float(np.mean(pred[accepted] != label_values[accepted]))
        accepted_accuracy = float(1.0 - accepted_error)

    return {
        "negative_threshold": float(negative_threshold),
        "positive_threshold": float(positive_threshold),
        "deferral_rate": deferral_rate,
        "accepted_coverage": accepted_coverage,
        "accepted_error": accepted_error,
        "accepted_accuracy": accepted_accuracy,
        "accepted_count": accepted_count,
        "deferred_count": deferred_count,
        "total_count": total_count,
    }


def _candidate_thresholds(scores: np.ndarray, grid_size: int) -> np.ndarray:
    if grid_size <= 1:
        msg = "grid_size must be > 1"
        raise ValueError(msg)
    values = np.asarray(scores, dtype=np.float32)
    if values.size == 0:
        msg = "Cannot build threshold candidates for empty scores"
        raise ValueError(msg)

    lo = float(values.min()) - 1e-6
    hi = float(values.max()) + 1e-6
    quantiles = np.quantile(values, np.linspace(0.0, 1.0, int(grid_size)))
    return np.unique(np.concatenate(([lo], quantiles, [hi])).astype(np.float64))


def tune_deferral_thresholds(
    *,
    validation_scores: np.ndarray,
    validation_labels: np.ndarray,
    objective: str,
    deferral_budget: float | None,
    target_accepted_error: float | None,
    grid_size: int = 101,
) -> dict[str, Any]:
    """Search threshold pairs using validation data."""
    if objective not in {"budget", "error"}:
        msg = f"Unsupported objective: {objective}"
        raise ValueError(msg)
    if objective == "budget":
        if deferral_budget is None:
            msg = "deferral_budget is required for objective='budget'"
            raise ValueError(msg)
        if not 0.0 <= float(deferral_budget) <= 1.0:
            msg = "deferral_budget must be in [0, 1]"
            raise ValueError(msg)
    if objective == "error":
        if target_accepted_error is None:
            msg = "target_accepted_error is required for objective='error'"
            raise ValueError(msg)
        if not 0.0 <= float(target_accepted_error) <= 1.0:
            msg = "target_accepted_error must be in [0, 1]"
            raise ValueError(msg)

    scores = np.asarray(validation_scores, dtype=np.float32)
    labels = np.asarray(validation_labels, dtype=np.int64)
    candidates = _candidate_thresholds(scores, grid_size=grid_size)

    best_payload: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    fallback_payload: dict[str, Any] | None = None
    fallback_key: tuple[float, ...] | None = None
    feasible_count = 0

    for neg_idx, neg_threshold in enumerate(candidates):
        for pos_threshold in candidates[neg_idx:]:
            metrics = evaluate_deferral_policy(
                scores=scores,
                labels=labels,
                negative_threshold=float(neg_threshold),
                positive_threshold=float(pos_threshold),
            )
            if np.isnan(metrics["accepted_error"]):
                continue

            fallback_candidate_key = (
                float(metrics["accepted_error"]),
                float(metrics["deferral_rate"]),
                float(metrics["negative_threshold"]),
                float(metrics["positive_threshold"]),
            )
            if fallback_key is None or fallback_candidate_key < fallback_key:
                fallback_key = fallback_candidate_key
                fallback_payload = metrics

            if objective == "budget":
                if float(metrics["deferral_rate"]) > float(deferral_budget) + 1e-12:
                    continue
                candidate_key = (
                    float(metrics["accepted_error"]),
                    float(metrics["deferral_rate"]),
                    float(metrics["negative_threshold"]),
                    float(metrics["positive_threshold"]),
                )
            else:
                if float(metrics["accepted_error"]) > float(target_accepted_error) + 1e-12:
                    continue
                candidate_key = (
                    float(metrics["deferral_rate"]),
                    float(metrics["accepted_error"]),
                    float(metrics["negative_threshold"]),
                    float(metrics["positive_threshold"]),
                )

            feasible_count += 1
            if best_key is None or candidate_key < best_key:
                best_key = candidate_key
                best_payload = metrics

    if best_payload is None:
        if fallback_payload is None:
            msg = "Could not find any valid threshold pairs."
            raise RuntimeError(msg)
        best_payload = fallback_payload

    return {
        "objective": objective,
        "deferral_budget": float(deferral_budget) if deferral_budget is not None else None,
        "target_accepted_error": (
            float(target_accepted_error) if target_accepted_error is not None else None
        ),
        "grid_size": int(grid_size),
        "feasible_count": int(feasible_count),
        "meets_objective": bool(feasible_count > 0),
        "selection": best_payload,
    }


def _weighted_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}
    total = float(sum(int(record["total_count"]) for record in records))
    if total <= 0:
        return {}

    def weighted_mean(metric_key: str) -> float:
        numer = 0.0
        denom = 0.0
        for record in records:
            value = float(record[metric_key])
            if np.isnan(value):
                continue
            weight = float(record["total_count"])
            numer += value * weight
            denom += weight
        return float(numer / denom) if denom > 0 else float("nan")

    return {
        "deferral_rate": weighted_mean("deferral_rate"),
        "accepted_coverage": weighted_mean("accepted_coverage"),
        "accepted_error": weighted_mean("accepted_error"),
        "accepted_accuracy": weighted_mean("accepted_accuracy"),
        "accepted_count": int(sum(int(record["accepted_count"]) for record in records)),
        "deferred_count": int(sum(int(record["deferred_count"]) for record in records)),
        "total_count": int(sum(int(record["total_count"]) for record in records)),
    }


def plan_deferrals(
    *,
    validation_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    score_column: str,
    label_column: str = "anchor",
    objective: str = "budget",
    deferral_budget: float | None = None,
    target_accepted_error: float | None = None,
    group_column: str | None = None,
    grid_size: int = 101,
) -> dict[str, Any]:
    """Fit thresholds on validation data and project deferral policy to test."""
    required_columns = {score_column, label_column}
    missing_val = sorted(required_columns - set(validation_frame.columns))
    missing_test = sorted(required_columns - set(test_frame.columns))
    if missing_val:
        msg = f"Validation frame missing required columns: {missing_val}"
        raise ValueError(msg)
    if missing_test:
        msg = f"Test frame missing required columns: {missing_test}"
        raise ValueError(msg)

    if group_column is not None:
        if group_column not in validation_frame.columns or group_column not in test_frame.columns:
            msg = f"group_column '{group_column}' must exist in both validation and test frames"
            raise ValueError(msg)
        group_values = sorted(
            set(validation_frame[group_column].tolist()).intersection(
                set(test_frame[group_column].tolist())
            )
        )
    else:
        group_values = ["__all__"]
        validation_frame = validation_frame.copy()
        test_frame = test_frame.copy()
        validation_frame["__all__"] = "__all__"
        test_frame["__all__"] = "__all__"
        group_column = "__all__"

    group_records: list[dict[str, Any]] = []
    for group_value in group_values:
        val_group = validation_frame[validation_frame[group_column] == group_value].copy()
        test_group = test_frame[test_frame[group_column] == group_value].copy()
        if val_group.empty or test_group.empty:
            continue

        tuning = tune_deferral_thresholds(
            validation_scores=val_group[score_column].to_numpy(dtype=np.float32),
            validation_labels=val_group[label_column].to_numpy(dtype=np.int64),
            objective=objective,
            deferral_budget=deferral_budget,
            target_accepted_error=target_accepted_error,
            grid_size=grid_size,
        )
        selection = tuning["selection"]
        val_metrics = evaluate_deferral_policy(
            scores=val_group[score_column].to_numpy(dtype=np.float32),
            labels=val_group[label_column].to_numpy(dtype=np.int64),
            negative_threshold=float(selection["negative_threshold"]),
            positive_threshold=float(selection["positive_threshold"]),
        )
        test_metrics = evaluate_deferral_policy(
            scores=test_group[score_column].to_numpy(dtype=np.float32),
            labels=test_group[label_column].to_numpy(dtype=np.int64),
            negative_threshold=float(selection["negative_threshold"]),
            positive_threshold=float(selection["positive_threshold"]),
        )
        group_records.append(
            {
                "group": group_value,
                "n_validation": int(len(val_group)),
                "n_test": int(len(test_group)),
                "thresholds": {
                    "negative": float(selection["negative_threshold"]),
                    "positive": float(selection["positive_threshold"]),
                },
                "tuning": {
                    "objective": tuning["objective"],
                    "feasible_count": tuning["feasible_count"],
                    "meets_objective": tuning["meets_objective"],
                },
                "validation": val_metrics,
                "test_projection": test_metrics,
            }
        )

    if not group_records:
        msg = "No groups with both validation and test rows were available."
        raise ValueError(msg)

    validation_summary = _weighted_summary([record["validation"] for record in group_records])
    test_summary = _weighted_summary([record["test_projection"] for record in group_records])

    return {
        "score_column": score_column,
        "label_column": label_column,
        "objective": objective,
        "deferral_budget": float(deferral_budget) if deferral_budget is not None else None,
        "target_accepted_error": (
            float(target_accepted_error) if target_accepted_error is not None else None
        ),
        "group_column": None if group_column == "__all__" else group_column,
        "grid_size": int(grid_size),
        "num_groups": int(len(group_records)),
        "groups": group_records,
        "validation_summary": validation_summary,
        "test_projection_summary": test_summary,
    }
