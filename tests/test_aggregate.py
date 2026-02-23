from __future__ import annotations

import json
from pathlib import Path

import pytest

from ta_probe.aggregate import aggregate_lopo_metrics, aggregate_run_metrics


def write_metrics_file(path: Path, seed: int, linear_pr_auc: float, mlp_pr_auc: float) -> None:
    payload = {
        "seed": seed,
        "run_name": f"seed_{seed}",
        "val": {
            "linear_probe": {
                "pr_auc": linear_pr_auc - 0.05,
                "spearman_mean": 0.2,
                "top_5_recall": 0.3,
                "top_10_recall": 0.4,
            },
            "mlp_probe": {
                "pr_auc": mlp_pr_auc - 0.05,
                "spearman_mean": 0.1,
                "top_5_recall": 0.2,
                "top_10_recall": 0.3,
            },
        },
        "test": {
            "linear_probe": {
                "pr_auc": linear_pr_auc,
                "spearman_mean": 0.3,
                "top_5_recall": 0.4,
                "top_10_recall": 0.5,
            },
            "mlp_probe": {
                "pr_auc": mlp_pr_auc,
                "spearman_mean": 0.2,
                "top_5_recall": 0.3,
                "top_10_recall": 0.4,
            },
        },
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_aggregate_run_metrics_computes_summary(tmp_path: Path) -> None:
    write_metrics_file(
        tmp_path / "metrics_seed_0.json", seed=0, linear_pr_auc=0.70, mlp_pr_auc=0.60
    )
    write_metrics_file(
        tmp_path / "metrics_seed_1.json", seed=1, linear_pr_auc=0.75, mlp_pr_auc=0.62
    )
    write_metrics_file(
        tmp_path / "metrics_seed_2.json", seed=2, linear_pr_auc=0.80, mlp_pr_auc=0.61
    )

    summary = aggregate_run_metrics(tmp_path)

    assert summary["num_runs"] == 3
    assert summary["primary_metric_name"] == "pr_auc"
    assert summary["best_model_by_mean_pr_auc"] == "linear_probe"
    best_seed = summary["best_seed_by_primary_metric"]
    assert best_seed is not None
    assert best_seed["model"] == "linear_probe"
    assert best_seed["seed"] == 2
    assert best_seed["val_primary_metric"] == pytest.approx(0.75)
    assert best_seed["test_primary_metric"] == pytest.approx(0.8)
    assert len(summary["summary_by_model"]) == 2
    assert Path(summary["output_json"]).exists()
    assert Path(summary["output_md"]).exists()


def test_aggregate_prefers_seed_metrics_and_ignores_backups(tmp_path: Path) -> None:
    write_metrics_file(
        tmp_path / "metrics_seed_0.json", seed=0, linear_pr_auc=0.70, mlp_pr_auc=0.60
    )
    write_metrics_file(
        tmp_path / "metrics_before_altlabels.json", seed=0, linear_pr_auc=0.10, mlp_pr_auc=0.10
    )
    summary = aggregate_run_metrics(tmp_path)
    assert summary["num_runs"] == 1
    assert summary["metric_files"] == [str(tmp_path / "metrics_seed_0.json")]


def test_aggregate_includes_ci_summary(tmp_path: Path) -> None:
    payload = {
        "seed": 0,
        "run_name": "seed_0",
        "val": {
            "linear_probe": {
                "pr_auc": 0.1,
                "spearman_mean": 0.1,
                "top_5_recall": 0.1,
                "top_10_recall": 0.1,
            }
        },
        "test": {
            "linear_probe": {
                "pr_auc": 0.1,
                "spearman_mean": 0.1,
                "top_5_recall": 0.1,
                "top_10_recall": 0.1,
            }
        },
        "confidence_intervals": {
            "score_activations_plus_position_minus_score_position_baseline": {
                "point_delta": 0.02,
                "delta_mean": 0.018,
                "ci_low": 0.005,
                "ci_high": 0.03,
                "n_bootstrap": 100,
                "n_valid_bootstrap": 100,
            }
        },
    }
    with (tmp_path / "metrics_seed_0.json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    summary = aggregate_run_metrics(tmp_path)
    assert len(summary["ci_records"]) == 1
    assert len(summary["ci_summary"]) == 1


def _write_lopo_metrics(
    root: Path,
    *,
    fold_id: int,
    seed: int,
    act_pr_auc: float,
    pos_pr_auc: float,
) -> None:
    payload = {
        "seed": seed,
        "run_name": f"fold_{fold_id}_seed_{seed}",
        "val": {
            "activations_plus_position": {
                "pr_auc": act_pr_auc - 0.1,
                "spearman_mean": 0.2,
                "top_5_recall": 0.3,
                "top_10_recall": 0.4,
            },
            "position_baseline": {
                "pr_auc": pos_pr_auc - 0.1,
                "spearman_mean": 0.1,
                "top_5_recall": 0.2,
                "top_10_recall": 0.3,
            },
        },
        "test": {
            "activations_plus_position": {
                "pr_auc": act_pr_auc,
                "spearman_mean": 0.2,
                "top_5_recall": 0.3,
                "top_10_recall": 0.4,
            },
            "position_baseline": {
                "pr_auc": pos_pr_auc,
                "spearman_mean": 0.1,
                "top_5_recall": 0.2,
                "top_10_recall": 0.3,
            },
        },
    }
    metrics_path = root / f"fold_{fold_id}" / str(seed) / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def test_aggregate_lopo_uses_fold_level_deltas(tmp_path: Path) -> None:
    _write_lopo_metrics(tmp_path, fold_id=1, seed=0, act_pr_auc=0.8, pos_pr_auc=0.5)
    _write_lopo_metrics(tmp_path, fold_id=1, seed=1, act_pr_auc=0.6, pos_pr_auc=0.5)
    _write_lopo_metrics(tmp_path, fold_id=2, seed=0, act_pr_auc=0.3, pos_pr_auc=0.6)
    _write_lopo_metrics(tmp_path, fold_id=2, seed=1, act_pr_auc=0.5, pos_pr_auc=0.4)

    summary = aggregate_lopo_metrics(
        tmp_path, best_of_k=1, bootstrap_iterations=10, bootstrap_seed=0
    )
    deltas = [
        record
        for record in summary["paired_delta_records"]
        if record["comparison"] == "activations_plus_position_minus_position_baseline"
        and record["agg_type"] == "mean_seeds"
    ]
    delta_values = sorted(round(float(record["delta"]), 4) for record in deltas)
    assert delta_values == [-0.1, 0.2]

    summary_rows = [
        row
        for row in summary["paired_delta_summary"]
        if row["comparison"] == "activations_plus_position_minus_position_baseline"
        and row["agg_type"] == "mean_seeds"
    ]
    assert len(summary_rows) == 1
    assert summary_rows[0]["n_folds"] == 2
    assert round(float(summary_rows[0]["mean"]), 4) == 0.05


def test_aggregate_run_metrics_uses_spearman_for_continuous_targets(tmp_path: Path) -> None:
    payload_seed_0 = {
        "seed": 0,
        "target_mode": "importance_signed",
        "val": {
            "linear_probe": {
                "pr_auc": 0.95,
                "spearman_mean": 0.2,
                "top_5_recall": float("nan"),
                "top_10_recall": float("nan"),
            },
            "mlp_probe": {
                "pr_auc": 0.10,
                "spearman_mean": 0.9,
                "top_5_recall": float("nan"),
                "top_10_recall": float("nan"),
            },
        },
        "test": {
            "linear_probe": {
                "pr_auc": 0.99,
                "spearman_mean": 0.3,
                "top_5_recall": float("nan"),
                "top_10_recall": float("nan"),
            },
            "mlp_probe": {
                "pr_auc": 0.05,
                "spearman_mean": 0.8,
                "top_5_recall": float("nan"),
                "top_10_recall": float("nan"),
            },
        },
    }
    payload_seed_1 = {
        "seed": 1,
        "target_mode": "importance_signed",
        "val": {
            "linear_probe": {
                "pr_auc": 0.90,
                "spearman_mean": 0.1,
                "top_5_recall": float("nan"),
                "top_10_recall": float("nan"),
            },
            "mlp_probe": {
                "pr_auc": 0.80,
                "spearman_mean": 0.4,
                "top_5_recall": float("nan"),
                "top_10_recall": float("nan"),
            },
        },
        "test": {
            "linear_probe": {
                "pr_auc": 0.98,
                "spearman_mean": 0.2,
                "top_5_recall": float("nan"),
                "top_10_recall": float("nan"),
            },
            "mlp_probe": {
                "pr_auc": 0.70,
                "spearman_mean": 0.6,
                "top_5_recall": float("nan"),
                "top_10_recall": float("nan"),
            },
        },
    }
    (tmp_path / "metrics_seed_0.json").write_text(json.dumps(payload_seed_0), encoding="utf-8")
    (tmp_path / "metrics_seed_1.json").write_text(json.dumps(payload_seed_1), encoding="utf-8")

    summary = aggregate_run_metrics(tmp_path, best_of_k=1)
    assert summary["primary_metric_name"] == "spearman"
    assert summary["primary_metric"] == "spearman_mean"
    assert summary["best_model_by_primary_metric"] == "mlp_probe"
    assert summary["best_model_by_mean_pr_auc"] == "mlp_probe"
    best_seed = summary["best_seed_by_primary_metric"]
    assert best_seed is not None
    assert best_seed["model"] == "mlp_probe"
    assert best_seed["seed"] == 0
    assert best_seed["val_primary_metric"] == pytest.approx(0.9)
    assert best_seed["test_primary_metric"] == pytest.approx(0.8)

    best_of_k_rows = {
        row["model"]: row for row in summary["best_of_k_summary_by_model"]
    }
    assert best_of_k_rows["mlp_probe"]["selected_seeds"] == [0]
    assert best_of_k_rows["linear_probe"]["selected_seeds"] == [0]


def test_aggregate_lopo_uses_primary_metric_for_continuous_targets(tmp_path: Path) -> None:
    payload_fold_1 = {
        "seed": 0,
        "target_mode": "importance_abs",
        "val": {
            "activations_plus_position": {
                "pr_auc": 0.2,
                "spearman_mean": 0.7,
                "top_5_recall": float("nan"),
                "top_10_recall": float("nan"),
            },
            "position_baseline": {
                "pr_auc": 0.8,
                "spearman_mean": 0.2,
                "top_5_recall": float("nan"),
                "top_10_recall": float("nan"),
            },
        },
        "test": {
            "activations_plus_position": {
                "pr_auc": 0.1,
                "spearman_mean": 0.8,
                "top_5_recall": float("nan"),
                "top_10_recall": float("nan"),
            },
            "position_baseline": {
                "pr_auc": 0.9,
                "spearman_mean": 0.2,
                "top_5_recall": float("nan"),
                "top_10_recall": float("nan"),
            },
        },
    }
    payload_fold_2 = {
        "seed": 0,
        "target_mode": "importance_abs",
        "val": {
            "activations_plus_position": {
                "pr_auc": 0.8,
                "spearman_mean": 0.2,
                "top_5_recall": float("nan"),
                "top_10_recall": float("nan"),
            },
            "position_baseline": {
                "pr_auc": 0.2,
                "spearman_mean": 0.4,
                "top_5_recall": float("nan"),
                "top_10_recall": float("nan"),
            },
        },
        "test": {
            "activations_plus_position": {
                "pr_auc": 0.9,
                "spearman_mean": 0.1,
                "top_5_recall": float("nan"),
                "top_10_recall": float("nan"),
            },
            "position_baseline": {
                "pr_auc": 0.1,
                "spearman_mean": 0.4,
                "top_5_recall": float("nan"),
                "top_10_recall": float("nan"),
            },
        },
    }
    fold_1_path = tmp_path / "fold_1" / "0" / "metrics.json"
    fold_1_path.parent.mkdir(parents=True, exist_ok=True)
    fold_1_path.write_text(json.dumps(payload_fold_1), encoding="utf-8")
    fold_2_path = tmp_path / "fold_2" / "0" / "metrics.json"
    fold_2_path.parent.mkdir(parents=True, exist_ok=True)
    fold_2_path.write_text(json.dumps(payload_fold_2), encoding="utf-8")

    summary = aggregate_lopo_metrics(
        tmp_path,
        best_of_k=1,
        bootstrap_iterations=10,
        bootstrap_seed=0,
    )
    assert summary["primary_metric_name"] == "spearman"
    assert summary["primary_metric"] == "spearman_mean"

    delta_rows = [
        row
        for row in summary["paired_delta_summary"]
        if row["comparison"] == "activations_plus_position_minus_position_baseline"
        and row["agg_type"] == "mean_seeds"
    ]
    assert len(delta_rows) == 1
    assert delta_rows[0]["metric"] == "spearman_mean"
    assert round(float(delta_rows[0]["mean"]), 4) == 0.15


def test_aggregate_lopo_bootstrap_uses_folds_not_rows(tmp_path: Path) -> None:
    # Fold 1 has many seeds with a positive delta; fold 2 has one seed with
    # an equal-magnitude negative delta. Fold-level bootstrap should treat the
    # two folds equally and include zero in the CI.
    for seed in range(40):
        _write_lopo_metrics(tmp_path, fold_id=1, seed=seed, act_pr_auc=1.0, pos_pr_auc=0.0)
    _write_lopo_metrics(tmp_path, fold_id=2, seed=0, act_pr_auc=0.0, pos_pr_auc=1.0)

    summary = aggregate_lopo_metrics(
        tmp_path,
        best_of_k=1,
        bootstrap_iterations=500,
        bootstrap_seed=7,
    )

    rows = [
        row
        for row in summary["paired_delta_summary"]
        if row["comparison"] == "activations_plus_position_minus_position_baseline"
        and row["agg_type"] == "mean_seeds"
    ]
    assert len(rows) == 1
    row = rows[0]

    assert row["n_folds"] == 2
    assert row["mean"] == pytest.approx(0.0, abs=1e-12)
    assert float(row["bootstrap_ci_low"]) < 0.0
    assert float(row["bootstrap_ci_high"]) > 0.0
