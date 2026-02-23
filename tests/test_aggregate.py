from __future__ import annotations

import json
from pathlib import Path

from ta_probe.aggregate import aggregate_run_metrics


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
    assert summary["best_model_by_mean_pr_auc"] == "linear_probe"
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
