from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
        check=False,
    )


def test_generate_sweep_configs_stage1_smoke(tmp_path: Path) -> None:
    sweep_root = tmp_path / "sweep"
    result = _run_script(
        [
            "scripts/generate_sweep_configs.py",
            "--base-config",
            "configs/scaling_qwen_correct.yaml",
            "--sweep-root",
            str(sweep_root),
            "--stage",
            "stage1",
            "--num-layers",
            "40",
            "--limit-configs",
            "3",
            "--seeds",
            "0",
            "1",
            "--reuse-cache",
            "--skip-failed",
        ]
    )
    assert result.returncode == 0, result.stderr

    stage_manifest = sweep_root / "manifest_stage1.jsonl"
    global_manifest = sweep_root / "manifest.jsonl"
    assert stage_manifest.exists()
    assert global_manifest.exists()

    rows = [
        json.loads(line) for line in stage_manifest.read_text(encoding="utf-8").splitlines() if line
    ]
    assert len(rows) == 3
    assert all("config_id" in row for row in rows)
    assert all("commands" in row for row in rows)
    assert all(Path(row["config_path"]).exists() for row in rows)


def test_run_systematic_sweep_executes_manifest(tmp_path: Path) -> None:
    sweep_root = tmp_path / "sweep"
    sweep_root.mkdir(parents=True, exist_ok=True)
    run_root = sweep_root / "runs" / "stage1" / "cfg_1"

    manifest_row = {
        "sweep_id": "test",
        "stage": "stage1",
        "config_id": "cfg_1",
        "config_path": str(sweep_root / "configs" / "stage1" / "cfg_1.yaml"),
        "run_root": str(run_root),
        "factor_values": {"target_mode": "anchor_binary"},
        "commands": [
            {"name": "step_a", "args": ["python", "-c", "print('a')"]},
            {"name": "step_b", "args": ["python", "-c", "print('b')"]},
        ],
    }
    manifest_path = sweep_root / "manifest.jsonl"
    manifest_path.write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")

    result = _run_script(
        [
            "scripts/run_systematic_sweep.py",
            "--sweep-root",
            str(sweep_root),
            "--retries",
            "0",
        ]
    )
    assert result.returncode == 0, result.stderr

    registry_path = sweep_root / "run_registry.jsonl"
    assert registry_path.exists()
    records = [
        json.loads(line)
        for line in registry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any(record.get("status") == "success" for record in records)

    step_log = sweep_root / "logs" / "cfg_1" / "attempt_1_step_a.log"
    assert step_log.exists()
    first_line = step_log.read_text(encoding="utf-8").splitlines()[0]
    assert first_line.startswith("command: ")
    assert not first_line.startswith("command: python ")


def test_summarize_sweep_writes_outputs(tmp_path: Path) -> None:
    sweep_root = tmp_path / "sweep"
    run_root = sweep_root / "runs" / "stage1" / "cfg_a"
    run_root.mkdir(parents=True, exist_ok=True)

    manifest_row = {
        "sweep_id": "test",
        "stage": "stage1",
        "config_id": "cfg_a",
        "config_path": str(sweep_root / "configs" / "stage1" / "cfg_a.yaml"),
        "run_root": str(run_root),
        "factor_values": {
            "target_mode": "anchor_binary",
            "split_dir": "correct_base_solution",
            "pooling": "mean",
            "layer_mode": "mid",
            "layer_index": None,
            "residualize_against": "none",
            "vertical_attention_mode": "off",
        },
        "commands": [],
    }
    sweep_root.mkdir(parents=True, exist_ok=True)
    (sweep_root / "manifest.jsonl").write_text(json.dumps(manifest_row) + "\n", encoding="utf-8")
    (sweep_root / "run_registry.jsonl").write_text(
        json.dumps(
            {
                "config_id": "cfg_a",
                "stage": "stage1",
                "status": "success",
                "attempt": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    aggregate_payload = {
        "best_model_by_primary_metric": "linear_probe",
        "primary_metric_name": "pr_auc",
        "primary_metric": "pr_auc",
        "num_folds": 5,
        "fold_aggregate_by_model": [
            {
                "agg_type": "mean_seeds",
                "model": "linear_probe",
                "pr_auc_mean": 0.2,
                "pr_auc_std": 0.01,
                "spearman_mean_mean": 0.3,
                "spearman_mean_std": 0.02,
                "top_5_recall_std": 0.0,
                "top_10_recall_std": 0.0,
            }
        ],
        "paired_delta_summary": [
            {
                "comparison": "score_a_minus_score_b",
                "agg_type": "mean_seeds",
                "mean": 0.1,
                "std": 0.02,
                "bootstrap_ci_low": 0.01,
                "bootstrap_ci_high": 0.2,
                "n_folds": 5,
            }
        ],
    }
    (run_root / "aggregate_lopo_metrics.json").write_text(
        json.dumps(aggregate_payload, indent=2),
        encoding="utf-8",
    )

    result = _run_script(
        [
            "scripts/summarize_sweep.py",
            "--sweep-root",
            str(sweep_root),
            "--stage2-top-n-per-target-mode",
            "1",
            "--stage3-top-n",
            "1",
        ]
    )
    assert result.returncode == 0, result.stderr

    summary_dir = sweep_root / "summary"
    assert (summary_dir / "leaderboard.csv").exists()
    assert (summary_dir / "paired_deltas.csv").exists()
    assert (summary_dir / "stability_report.csv").exists()
    assert (summary_dir / "summary.md").exists()

    shortlist = json.loads((summary_dir / "shortlist_stage2.json").read_text(encoding="utf-8"))
    assert shortlist
    assert shortlist[0]["config_id"] == "cfg_a"
