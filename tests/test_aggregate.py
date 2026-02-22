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
