#!/usr/bin/env python
"""Run pilot and full probe experiments and update README results."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.readme_update import update_readme_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment.yaml", help="Base config path")
    parser.add_argument(
        "--pilot-config",
        default="configs/experiment_pilot.yaml",
        help="Pilot config path",
    )
    parser.add_argument(
        "--full-config",
        default="configs/experiment_full.yaml",
        help="Full config path",
    )
    parser.add_argument(
        "--problem-id",
        type=int,
        default=330,
        help="Problem ID used for required verification checks.",
    )
    parser.add_argument(
        "--readme-path",
        default="README.md",
        help="README path to update with final experiment results.",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> dict[str, Any]:
    """Run one command and return command metadata and captured output."""
    start = time.time()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    duration = time.time() - start

    record = {
        "command": " ".join(command),
        "duration_sec": round(duration, 3),
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }

    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout)[-3000:]
        msg = f"Command failed: {' '.join(command)}\n{tail}"
        raise RuntimeError(msg)

    return record


def parse_last_json(text: str) -> dict[str, Any]:
    """Parse the last JSON object found in command output."""
    cursor = text.rfind("{")
    while cursor != -1:
        candidate = text[cursor:]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            cursor = text.rfind("{", 0, cursor)
    msg = "Could not parse JSON from command output"
    raise ValueError(msg)


def parse_span_pass_rate(text: str) -> float:
    """Extract span integrity pass rate from command output."""
    match = re.search(r"Pass rate:\s*([0-9.]+)", text)
    if match is None:
        msg = "Could not parse span pass rate from output"
        raise ValueError(msg)
    return float(match.group(1))


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_seed_metrics(run_root: Path) -> dict[str, dict[str, Any]]:
    """Load per-seed metrics payloads keyed by run label."""
    payloads: dict[str, dict[str, Any]] = {}
    for metrics_file in sorted(run_root.glob("metrics_seed_*.json")):
        run_label = metrics_file.stem.replace("metrics_", "")
        payloads[run_label] = load_json(metrics_file)
    return payloads


def format_seed_table(seed_records: list[dict[str, Any]]) -> str:
    """Build a markdown table from test split seed records."""
    rows = [record for record in seed_records if record.get("split") == "test"]
    rows = sorted(rows, key=lambda row: (row["run_label"], row["model"]))

    lines = [
        "| Run | Seed | Model | PR AUC | Spearman | Top-5 | Top-10 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run_label']} | {int(row['seed'])} | {row['model']} "
            f"| {row['pr_auc']:.4f} | {row['spearman_mean']:.4f} "
            f"| {row['top_5_recall']:.4f} | {row['top_10_recall']:.4f} |"
        )
    return "\n".join(lines)


def format_summary_table(summary_records: list[dict[str, Any]]) -> str:
    """Build a markdown table for mean and std metrics across seeds."""
    lines = [
        (
            "| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std "
            "| Top-5 mean | Top-5 std | Top-10 mean | Top-10 std |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_records:
        lines.append(
            f"| {row['model']} "
            f"| {row['pr_auc_mean']:.4f} | {row['pr_auc_std']:.4f} "
            f"| {row['spearman_mean_mean']:.4f} | {row['spearman_mean_std']:.4f} "
            f"| {row['top_5_recall_mean']:.4f} | {row['top_5_recall_std']:.4f} "
            f"| {row['top_10_recall_mean']:.4f} | {row['top_10_recall_std']:.4f} |"
        )
    return "\n".join(lines)


def format_tripwire_table(seed_metrics: dict[str, dict[str, Any]]) -> str:
    """Build a markdown table for tripwire outcomes per seed."""
    lines = [
        (
            "| Run | Random-label near chance | Random-label PR AUC | Prevalence "
            "| Overfit can memorize | Overfit PR AUC |"
        ),
        "|---|---|---:|---:|---|---:|",
    ]

    for run_label, payload in sorted(seed_metrics.items()):
        tripwires = payload.get("tripwires", {})
        random_test = tripwires.get("random_label_test", {})
        overfit_test = tripwires.get("overfit_one_problem_test", {})

        near_chance = random_test.get("near_chance", False)
        random_auc = float(random_test.get("test_pr_auc", float("nan")))
        prevalence = float(random_test.get("test_prevalence", float("nan")))
        can_mem = overfit_test.get("can_memorize")
        overfit_auc = overfit_test.get("train_pr_auc", float("nan"))

        lines.append(
            f"| {run_label} | {near_chance} | {random_auc:.4f} | {prevalence:.4f} "
            f"| {can_mem} | {float(overfit_auc):.4f} |"
        )

    return "\n".join(lines)


def format_failure_summary(failure_payload: dict[str, Any]) -> str:
    """Format extraction failures as markdown bullets."""
    skipped = failure_payload.get("skipped_problem_ids", [])
    reasons = failure_payload.get("skip_reasons", {})
    if not skipped:
        return "- No extraction failures were logged."

    lines = [f"- Skipped problems: {len(skipped)}"]
    for problem_id in skipped:
        reason = reasons.get(str(problem_id), "No reason recorded")
        lines.append(f"- Problem {problem_id}: {reason}")
    return "\n".join(lines)


def format_command_log(commands: list[dict[str, Any]]) -> str:
    """Format executed commands with duration and status."""
    lines = []
    for record in commands:
        lines.append(
            f"- `{record['command']}` | exit={record['returncode']} | {record['duration_sec']:.2f}s"
        )
    return "\n".join(lines)


def build_results_block(
    *,
    commands: list[dict[str, Any]],
    verification: dict[str, Any],
    span_pass_rate: float,
    pilot_aggregate: dict[str, Any],
    full_aggregate: dict[str, Any],
    pilot_failures: dict[str, Any],
    full_failures: dict[str, Any],
    pilot_splits: dict[str, list[int]],
    full_splits: dict[str, list[int]],
    pilot_seed_metrics: dict[str, dict[str, Any]],
    full_seed_metrics: dict[str, dict[str, Any]],
) -> str:
    """Build the final README experiment results markdown block."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        f"Last updated: {now}",
        "",
        "### Objective",
        "Run the full Thought Anchor probe plan with pilot and full stages.",
        "",
        "### Environment",
        f"- Host: {platform.node()}",
        f"- Platform: {platform.platform()}",
        f"- Python: {platform.python_version()}",
        "- Dataset listing date context: February 22, 2026",
        "",
        "### Commands Executed",
        format_command_log(commands),
        "",
        "### Verification Gates",
        f"- Resampling avg diff: {verification.get('resampling_avg_diff', float('nan')):.6f}",
        f"- Resampling pass: {verification.get('resampling_pass', False)}",
        f"- Span pass rate: {span_pass_rate:.3f}",
        "",
        "### Dataset and Split Summary",
        (
            f"- Pilot split sizes: train={len(pilot_splits['train'])}, "
            f"val={len(pilot_splits['val'])}, test={len(pilot_splits['test'])}"
        ),
        (
            f"- Full split sizes: train={len(full_splits['train'])}, "
            f"val={len(full_splits['val'])}, test={len(full_splits['test'])}"
        ),
        "",
        "### Pilot Per-Seed Test Metrics",
        format_seed_table(pilot_aggregate["seed_records"]),
        "",
        "### Pilot Mean and Std Across Seeds",
        format_summary_table(pilot_aggregate["summary_by_model"]),
        "",
        f"- Pilot best model by mean PR AUC: `{pilot_aggregate.get('best_model_by_mean_pr_auc')}`",
        "",
        "### Pilot Tripwire Outcomes",
        format_tripwire_table(pilot_seed_metrics),
        "",
        "### Pilot Extraction Failures",
        format_failure_summary(pilot_failures),
        "",
        "### Full Per-Seed Test Metrics",
        format_seed_table(full_aggregate["seed_records"]),
        "",
        "### Full Mean and Std Across Seeds",
        format_summary_table(full_aggregate["summary_by_model"]),
        "",
        f"- Full best model by mean PR AUC: `{full_aggregate.get('best_model_by_mean_pr_auc')}`",
        "",
        "### Full Tripwire Outcomes",
        format_tripwire_table(full_seed_metrics),
        "",
        "### Full Extraction Failures",
        format_failure_summary(full_failures),
        "",
        "### Methodology Fidelity",
        (
            "- Planned and executed: resampling verification, span checks, "
            "pilot gate, full run, and three seeds."
        ),
        "- Planned and executed: position baseline, linear probe, and MLP probe.",
        "- Planned and executed: problem-level train, validation, and test splits.",
        "",
        "### Deviations",
        "#### Minor Changes",
        "- Verification uses fixed problem ID `330` for determinism.",
        "- Full config uses `num_problems: 9999` to include all available IDs automatically.",
        "",
        "#### Major Methodology Changes",
        "1. Three-seed training and aggregation added.",
        "2. Skip-and-log extraction policy added.",
        "3. Pilot gate before full run added.",
    ]

    return "\n".join(lines)


def stage_run(
    *,
    config_path: Path,
    run_root: Path,
    repo_root: Path,
    commands: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Run extraction, seed training, and aggregation for one stage."""
    failure_log = run_root / "extraction_failures.json"

    extraction = run_command(
        [
            "python",
            "scripts/extract_embeddings.py",
            "--config",
            str(config_path),
            "--skip-failed",
            "--failure-log",
            str(failure_log),
        ],
        cwd=repo_root,
    )
    commands.append(extraction)

    for seed in [0, 1, 2]:
        train_run = run_command(
            [
                "python",
                "scripts/train_probes.py",
                "--config",
                str(config_path),
                "--seed",
                str(seed),
                "--run-name",
                f"seed_{seed}",
            ],
            cwd=repo_root,
        )
        commands.append(train_run)

    aggregate = run_command(
        [
            "python",
            "scripts/aggregate_runs.py",
            "--run-root",
            str(run_root),
        ],
        cwd=repo_root,
    )
    commands.append(aggregate)

    aggregate_payload = load_json(run_root / "aggregate_metrics.json")
    failure_payload = load_json(failure_log)
    seed_payloads = load_seed_metrics(run_root)

    return (
        parse_last_json(extraction["stdout"]),
        aggregate_payload,
        failure_payload,
        seed_payloads,
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    commands: list[dict[str, Any]] = []

    # Phase 1: preflight and required checks.
    commands.append(run_command(["python", "-m", "ruff", "check", "."], cwd=repo_root))
    commands.append(run_command(["python", "-m", "pytest", "-q"], cwd=repo_root))
    commands.append(
        run_command(
            [
                "python",
                "scripts/build_problem_index.py",
                "--config",
                args.config,
                "--refresh",
            ],
            cwd=repo_root,
        )
    )

    verification_run = run_command(
        [
            "python",
            "scripts/verify_problem_labels.py",
            "--config",
            args.config,
            "--problem-id",
            str(args.problem_id),
        ],
        cwd=repo_root,
    )
    commands.append(verification_run)
    verification = parse_last_json(verification_run["stdout"])

    if not bool(verification.get("resampling_pass", False)):
        msg = f"Resampling verification failed: {verification}"
        raise RuntimeError(msg)

    span_run = run_command(
        [
            "python",
            "scripts/check_spans.py",
            "--config",
            args.config,
            "--problem-id",
            str(args.problem_id),
            "--sample-size",
            "20",
        ],
        cwd=repo_root,
    )
    commands.append(span_run)
    span_pass_rate = parse_span_pass_rate(span_run["stdout"])

    if span_pass_rate < 0.90:
        msg = f"Span integrity gate failed with pass rate {span_pass_rate:.3f}"
        raise RuntimeError(msg)

    # Build pilot and full split files.
    commands.append(
        run_command(
            [
                "python",
                "scripts/build_problem_index.py",
                "--config",
                args.pilot_config,
                "--refresh",
            ],
            cwd=repo_root,
        )
    )
    commands.append(
        run_command(
            [
                "python",
                "scripts/build_problem_index.py",
                "--config",
                args.full_config,
                "--refresh",
            ],
            cwd=repo_root,
        )
    )

    pilot_run_root = repo_root / "artifacts" / "runs" / "pilot"
    full_run_root = repo_root / "artifacts" / "runs" / "full"

    _, pilot_aggregate, pilot_failures, pilot_seed_metrics = stage_run(
        config_path=Path(args.pilot_config),
        run_root=pilot_run_root,
        repo_root=repo_root,
        commands=commands,
    )

    _, full_aggregate, full_failures, full_seed_metrics = stage_run(
        config_path=Path(args.full_config),
        run_root=full_run_root,
        repo_root=repo_root,
        commands=commands,
    )

    pilot_splits = load_json(repo_root / "data" / "splits_pilot.json")
    full_splits = load_json(repo_root / "data" / "splits_full.json")

    block = build_results_block(
        commands=commands,
        verification=verification,
        span_pass_rate=span_pass_rate,
        pilot_aggregate=pilot_aggregate,
        full_aggregate=full_aggregate,
        pilot_failures=pilot_failures,
        full_failures=full_failures,
        pilot_splits=pilot_splits,
        full_splits=full_splits,
        pilot_seed_metrics=pilot_seed_metrics,
        full_seed_metrics=full_seed_metrics,
    )

    readme_path = repo_root / args.readme_path
    update_readme_results(readme_path, block)

    summary = {
        "verification": verification,
        "span_pass_rate": span_pass_rate,
        "pilot_best_model": pilot_aggregate.get("best_model_by_mean_pr_auc"),
        "full_best_model": full_aggregate.get("best_model_by_mean_pr_auc"),
        "readme_path": str(readme_path),
        "commands_executed": len(commands),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
