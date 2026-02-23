#!/usr/bin/env python
"""Release harness for lint/tests/experiments/scaling with strict validations."""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.config import load_config, resolve_embeddings_memmap_path
from ta_probe.readme_update import (
    RESULTS_END,
    RESULTS_START,
    SCALING_RESULTS_END,
    SCALING_RESULTS_START,
)

DEFAULT_SCALING_CONFIGS = [
    "configs/scaling_llama_correct.yaml",
    "configs/scaling_llama_incorrect.yaml",
    "configs/scaling_qwen_correct.yaml",
    "configs/scaling_qwen_incorrect.yaml",
]


class ReleaseError(RuntimeError):
    """Raised when the release harness should fail immediately."""


@dataclass
class CommandRecord:
    stage: str
    command: list[str]
    cwd: Path
    log_path: Path
    returncode: int
    duration_sec: float


@dataclass
class RunSpec:
    label: str
    config_path: Path
    split_strategy: str
    target_mode: str
    run_root: Path
    embeddings_path: Path
    embeddings_shape_path: Path
    metadata_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiment.yaml",
        help="Base experiment config path passed to scripts/run_experiments.py.",
    )
    parser.add_argument(
        "--pilot-config",
        default="configs/experiment_pilot.yaml",
        help="Pilot config path passed to scripts/run_experiments.py.",
    )
    parser.add_argument(
        "--full-config",
        default="configs/experiment_full.yaml",
        help="Full config path passed to scripts/run_experiments.py.",
    )
    parser.add_argument(
        "--problem-id",
        type=int,
        default=330,
        help="Problem ID used by scripts/run_experiments.py verification gates.",
    )
    parser.add_argument(
        "--readme-path",
        default="README.md",
        help="README file updated by run_experiments and run_scaling_grid.",
    )
    parser.add_argument(
        "--scaling-configs",
        nargs="+",
        default=DEFAULT_SCALING_CONFIGS,
        help="Config paths passed to scripts/run_scaling_grid.py --configs.",
    )
    parser.add_argument(
        "--scaling-seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Seed list passed to scripts/run_scaling_grid.py --seeds.",
    )
    parser.add_argument(
        "--no-reuse-cache",
        action="store_true",
        help="Pass --no-reuse-cache to scaling and LOPO runs.",
    )
    parser.add_argument(
        "--lopo-configs",
        nargs="*",
        default=[],
        help="Optional LOPO configs. If omitted, LOPO stage is skipped.",
    )
    parser.add_argument(
        "--lopo-seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Seeds passed to scripts/run_lopo_cv.py when --lopo-configs is set.",
    )
    parser.add_argument(
        "--python-executable",
        default=sys.executable,
        help="Python executable used for all invoked commands.",
    )
    parser.add_argument(
        "--logs-root",
        default="artifacts/logs",
        help="Root folder where timestamped release logs are written.",
    )
    parser.add_argument(
        "--timestamp",
        default=None,
        help="Optional timestamp override for deterministic log folder naming.",
    )
    return parser.parse_args()


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_") or "stage"


def _resolve_repo_relative_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _parse_last_json(stdout_text: str) -> dict[str, Any]:
    cursor = stdout_text.rfind("{")
    while cursor != -1:
        candidate = stdout_text[cursor:]
        try:
            payload = json.loads(candidate)
        except json.JSONDecodeError:
            cursor = stdout_text.rfind("{", 0, cursor)
            continue
        if isinstance(payload, dict):
            return payload
        cursor = stdout_text.rfind("{", 0, cursor)
    raise ValueError("Could not parse JSON payload from command stdout.")


def _run_command(
    *,
    stage: str,
    command: list[str],
    cwd: Path,
    logs_dir: Path,
    records: list[CommandRecord],
) -> subprocess.CompletedProcess[str]:
    log_path = logs_dir / f"{len(records) + 1:02d}_{_safe_slug(stage)}.log"
    started_at = datetime.now(timezone.utc).isoformat()
    start_time = time.time()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    duration_sec = time.time() - start_time
    log_text = [
        f"stage: {stage}",
        f"started_at_utc: {started_at}",
        f"cwd: {cwd}",
        f"command: {' '.join(command)}",
        f"exit_code: {completed.returncode}",
        f"duration_sec: {duration_sec:.3f}",
        "",
        "=== STDOUT ===",
        completed.stdout,
        "",
        "=== STDERR ===",
        completed.stderr,
    ]
    log_path.write_text("\n".join(log_text), encoding="utf-8")
    records.append(
        CommandRecord(
            stage=stage,
            command=command,
            cwd=cwd,
            log_path=log_path,
            returncode=int(completed.returncode),
            duration_sec=duration_sec,
        )
    )
    if completed.returncode != 0:
        raise ReleaseError(
            "Command failed "
            f"(stage='{stage}', exit={completed.returncode}): {' '.join(command)}. "
            f"See log: {log_path}"
        )
    return completed


def _run_optional_command(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": int(completed.returncode),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _write_environment_snapshot(
    *,
    repo_root: Path,
    logs_dir: Path,
    python_executable: str,
) -> dict[str, str]:
    git_info = _run_optional_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    git_commit = git_info["stdout"].strip() if git_info["returncode"] == 0 else "unavailable"
    (logs_dir / "git_commit.txt").write_text(
        (git_info["stdout"] + git_info["stderr"]).strip() + "\n",
        encoding="utf-8",
    )

    python_info = _run_optional_command([python_executable, "--version"], cwd=repo_root)
    python_version_text = (python_info["stdout"] or python_info["stderr"]).strip()
    (logs_dir / "python_version.txt").write_text(python_version_text + "\n", encoding="utf-8")

    pip_freeze = _run_optional_command([python_executable, "-m", "pip", "freeze"], cwd=repo_root)
    (logs_dir / "pip_freeze.txt").write_text(
        (pip_freeze["stdout"] + pip_freeze["stderr"]).strip() + "\n",
        encoding="utf-8",
    )

    nvidia_info = _run_optional_command(["nvidia-smi"], cwd=repo_root)
    nvidia_text = (nvidia_info["stdout"] + nvidia_info["stderr"]).strip()
    if not nvidia_text:
        nvidia_text = "nvidia-smi output unavailable."
    (logs_dir / "nvidia_smi.txt").write_text(nvidia_text + "\n", encoding="utf-8")

    versions_cmd = [
        python_executable,
        "-c",
        (
            "import torch, transformers; "
            "print(f'torch={torch.__version__}\\ntransformers={transformers.__version__}')"
        ),
    ]
    package_versions = _run_optional_command(versions_cmd, cwd=repo_root)
    package_text = (package_versions["stdout"] + package_versions["stderr"]).strip()
    if not package_text:
        package_text = "torch/transformers version lookup unavailable."
    (logs_dir / "torch_transformers_versions.txt").write_text(package_text + "\n", encoding="utf-8")

    return {
        "git_commit": git_commit,
        "python_version": python_version_text or "unavailable",
        "torch_transformers": package_text,
    }


def _build_run_spec(label: str, config_path: str, repo_root: Path) -> RunSpec:
    resolved_config_path = _resolve_repo_relative_path(config_path, repo_root)
    config = load_config(resolved_config_path)
    metrics_json_path = _resolve_repo_relative_path(config.paths.metrics_json, repo_root)
    run_root = metrics_json_path.parent
    embeddings_path = _resolve_repo_relative_path(
        resolve_embeddings_memmap_path(config), repo_root
    )
    embeddings_shape_path = _resolve_repo_relative_path(
        config.paths.embeddings_shape_json, repo_root
    )
    metadata_path = _resolve_repo_relative_path(config.paths.metadata_parquet, repo_root)
    return RunSpec(
        label=label,
        config_path=resolved_config_path,
        split_strategy=config.split.strategy,
        target_mode=config.labels.target_mode,
        run_root=run_root,
        embeddings_path=embeddings_path,
        embeddings_shape_path=embeddings_shape_path,
        metadata_path=metadata_path,
    )


def _normalize_primary_metric_name(value: Any) -> str:
    text = str(value) if value is not None else "pr_auc"
    if text in {"spearman", "spearman_mean"}:
        return "spearman"
    if text == "pr_auc":
        return "pr_auc"
    return "pr_auc"


def _validate_single_split_artifacts(specs: list[RunSpec]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in specs:
        required_paths = [
            ("embeddings_memmap", spec.embeddings_path),
            ("embeddings_shape_json", spec.embeddings_shape_path),
            ("metadata_parquet", spec.metadata_path),
            ("aggregate_metrics_json", spec.run_root / "aggregate_metrics.json"),
        ]
        metrics_files = sorted(spec.run_root.glob("metrics_seed_*.json"))
        checks.append(
            {
                "setting": spec.label,
                "name": "metrics_seed_glob",
                "path": str(spec.run_root / "metrics_seed_*.json"),
                "exists": bool(metrics_files),
            }
        )
        if not metrics_files:
            missing.append(f"{spec.label}: missing metrics_seed_*.json under {spec.run_root}")

        for name, path in required_paths:
            exists = path.exists()
            checks.append(
                {
                    "setting": spec.label,
                    "name": name,
                    "path": str(path),
                    "exists": exists,
                }
            )
            if not exists:
                missing.append(f"{spec.label}: missing {name} at {path}")

    if missing:
        raise ReleaseError("Artifact invariant failed:\n- " + "\n- ".join(missing))
    return checks


def _validate_tripwires(specs: list[RunSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    failures: list[str] = []
    for spec in specs:
        if spec.target_mode != "anchor_binary":
            continue
        metrics_files = sorted(spec.run_root.glob("metrics_seed_*.json"))
        for metrics_path in metrics_files:
            payload = _load_json(metrics_path)
            tripwires = payload.get("tripwires", {})
            random_test = tripwires.get("random_label_test", {})
            overfit_test = tripwires.get("overfit_one_problem_test", {})
            near_chance = bool(random_test.get("near_chance", False))
            random_pr_auc = float(random_test.get("test_pr_auc", float("nan")))
            prevalence = float(random_test.get("test_prevalence", float("nan")))
            can_memorize = bool(overfit_test.get("can_memorize", False))
            skipped = bool(overfit_test.get("skipped", False))
            rows.append(
                {
                    "setting": spec.label,
                    "metrics_file": str(metrics_path),
                    "near_chance": near_chance,
                    "random_pr_auc": random_pr_auc,
                    "prevalence": prevalence,
                    "can_memorize": can_memorize,
                    "overfit_skipped": skipped,
                }
            )
            if not near_chance:
                failures.append(
                    f"{spec.label} {metrics_path.name}: random-label test not near chance "
                    f"(pr_auc={random_pr_auc:.4f}, prevalence={prevalence:.4f})."
                )
            if random_pr_auc > prevalence + 0.1:
                failures.append(
                    f"{spec.label} {metrics_path.name}: random-label pr_auc exceeds "
                    "prevalence + 0.1 guardrail."
                )
            if skipped:
                failures.append(
                    f"{spec.label} {metrics_path.name}: overfit one-problem test was skipped."
                )
            if not can_memorize:
                failures.append(
                    f"{spec.label} {metrics_path.name}: "
                    "overfit one-problem test failed to memorize."
                )
    if failures:
        raise ReleaseError("Tripwire invariant failed:\n- " + "\n- ".join(failures))
    return rows


def _resolve_best_metric(
    aggregate_payload: dict[str, Any],
    summary_rows: list[dict[str, Any]],
) -> tuple[str, str, float]:
    primary_metric_name = _normalize_primary_metric_name(
        aggregate_payload.get(
            "primary_metric_name",
            aggregate_payload.get("primary_metric", "pr_auc"),
        )
    )
    best_model = aggregate_payload.get("best_model_by_primary_metric")
    if best_model is None:
        best_model = aggregate_payload.get("best_model_by_mean_pr_auc")
    if best_model is None:
        return primary_metric_name, "n/a", float("nan")
    metric_key = "pr_auc_mean" if primary_metric_name == "pr_auc" else "spearman_mean_mean"
    best_value = float("nan")
    for row in summary_rows:
        if str(row.get("model")) == str(best_model):
            best_value = float(row.get(metric_key, float("nan")))
            break
    return primary_metric_name, str(best_model), best_value


def _validate_primary_metric_not_nan(
    single_split_specs: list[RunSpec],
    lopo_run_roots: list[tuple[str, Path]],
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    failures: list[str] = []

    for spec in single_split_specs:
        aggregate_path = spec.run_root / "aggregate_metrics.json"
        payload = _load_json(aggregate_path)
        summary_rows = payload.get("summary_by_model", [])
        primary_metric_name = _normalize_primary_metric_name(
            payload.get("primary_metric_name", payload.get("primary_metric", "pr_auc"))
        )
        metric_key = "pr_auc_mean" if primary_metric_name == "pr_auc" else "spearman_mean_mean"
        for row in summary_rows:
            value = float(row.get(metric_key, float("nan")))
            if not math.isfinite(value):
                failures.append(
                    f"{spec.label}: non-finite {metric_key} for model {row.get('model')} "
                    f"in {aggregate_path}"
                )
        primary_metric, best_model, best_value = _resolve_best_metric(payload, summary_rows)
        metrics.append(
            {
                "setting": spec.label,
                "best_model": best_model,
                "primary_metric": primary_metric,
                "best_value": best_value,
            }
        )

    for label, run_root in lopo_run_roots:
        aggregate_path = run_root / "aggregate_lopo_metrics.json"
        payload = _load_json(aggregate_path)
        primary_metric_name = _normalize_primary_metric_name(
            payload.get("primary_metric_name", payload.get("primary_metric", "pr_auc"))
        )
        metric_key = "pr_auc_mean" if primary_metric_name == "pr_auc" else "spearman_mean_mean"
        rows = [
            row
            for row in payload.get("fold_aggregate_by_model", [])
            if row.get("agg_type") == "mean_seeds"
        ]
        for row in rows:
            value = float(row.get(metric_key, float("nan")))
            if not math.isfinite(value):
                failures.append(
                    f"{label}: non-finite {metric_key} for model {row.get('model')} "
                    f"in {aggregate_path}"
                )
        best_model = payload.get("best_model_by_primary_metric")
        best_value = float("nan")
        for row in rows:
            if str(row.get("model")) == str(best_model):
                best_value = float(row.get(metric_key, float("nan")))
                break
        metrics.append(
            {
                "setting": label,
                "best_model": str(best_model) if best_model is not None else "n/a",
                "primary_metric": primary_metric_name,
                "best_value": best_value,
            }
        )

    if failures:
        raise ReleaseError("Primary-metric invariant failed:\n- " + "\n- ".join(failures))
    return metrics


def _extract_marked_block(text: str, start_marker: str, end_marker: str) -> str:
    if start_marker not in text or end_marker not in text:
        raise ReleaseError(f"README markers missing: {start_marker} / {end_marker}")
    _, remainder = text.split(start_marker, maxsplit=1)
    block, _ = remainder.split(end_marker, maxsplit=1)
    return block.strip()


def _validate_readme_markers(readme_path: Path) -> dict[str, str]:
    text = readme_path.read_text(encoding="utf-8")
    experiment_block = _extract_marked_block(text, RESULTS_START, RESULTS_END)
    scaling_block = _extract_marked_block(text, SCALING_RESULTS_START, SCALING_RESULTS_END)
    if not experiment_block:
        raise ReleaseError("README experiment results block is empty after run.")
    if not scaling_block:
        raise ReleaseError("README scaling results block is empty after run.")
    if "Last updated:" not in experiment_block:
        raise ReleaseError(
            "README experiment results block was not refreshed "
            "(missing 'Last updated:')."
        )
    if "# Scaling Matrix Summary" not in scaling_block:
        raise ReleaseError(
            "README scaling results block was not refreshed (missing '# Scaling Matrix Summary')."
        )
    return {
        "experiment_block": experiment_block,
        "scaling_block": scaling_block,
    }


def _collect_extraction_failures(specs: list[RunSpec]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in specs:
        failure_log = spec.run_root / "extraction_failures.json"
        if not failure_log.exists():
            rows.append(
                {
                    "setting": spec.label,
                    "path": str(failure_log),
                    "missing": True,
                    "skipped_count": 0,
                    "sample_problem_ids": [],
                }
            )
            continue
        payload = _load_json(failure_log)
        skipped = [int(problem_id) for problem_id in payload.get("skipped_problem_ids", [])]
        rows.append(
            {
                "setting": spec.label,
                "path": str(failure_log),
                "missing": False,
                "skipped_count": len(skipped),
                "sample_problem_ids": skipped[:10],
            }
        )
    return rows


def _render_command_table(records: list[CommandRecord]) -> list[str]:
    lines = [
        "| Stage | Exit | Duration (s) | Log File | Command |",
        "|---|---:|---:|---|---|",
    ]
    for record in records:
        command_text = " ".join(record.command)
        lines.append(
            f"| {record.stage} | {record.returncode} | {record.duration_sec:.2f} "
            f"| `{record.log_path.name}` | `{command_text}` |"
        )
    return lines


def _render_key_metrics_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Setting | Best Model | Primary Metric | Best Primary Metric Value |",
        "|---|---|---|---:|",
    ]
    for row in rows:
        best_value = float(row.get("best_value", float("nan")))
        rendered_value = f"{best_value:.4f}" if math.isfinite(best_value) else "nan"
        lines.append(
            f"| {row['setting']} | {row['best_model']} | "
            f"{row['primary_metric']} | {rendered_value} |"
        )
    return lines


def _render_artifact_checks_table(checks: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Setting | Check | Exists | Path |",
        "|---|---|---|---|",
    ]
    for row in checks:
        lines.append(
            f"| {row['setting']} | {row['name']} | {bool(row['exists'])} | `{row['path']}` |"
        )
    return lines


def _render_tripwire_table(rows: list[dict[str, Any]]) -> list[str]:
    if not rows:
        return ["- No binary-mode tripwire rows were checked."]
    lines = [
        "| Setting | Metrics File | Near Chance | Random PR-AUC | Prevalence | Can Memorize |",
        "|---|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['setting']} | `{Path(row['metrics_file']).name}` | {row['near_chance']} "
            f"| {row['random_pr_auc']:.4f} | {row['prevalence']:.4f} | {row['can_memorize']} |"
        )
    return lines


def _render_extraction_failures(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| Setting | Failure Log | Missing | Skipped Count | Sample Problem IDs |",
        "|---|---|---|---:|---|",
    ]
    for row in rows:
        sample = ", ".join(str(pid) for pid in row.get("sample_problem_ids", []))
        lines.append(
            f"| {row['setting']} | `{row['path']}` | {row['missing']} | "
            f"{int(row['skipped_count'])} | {sample} |"
        )
    return lines


def _write_summary(
    *,
    summary_path: Path,
    timestamp: str,
    logs_dir: Path,
    environment: dict[str, str],
    command_records: list[CommandRecord],
    key_metrics: list[dict[str, Any]],
    artifact_checks: list[dict[str, Any]],
    tripwire_rows: list[dict[str, Any]],
    extraction_failures: list[dict[str, Any]],
    readme_path: Path,
    readme_blocks: dict[str, str] | None,
    disk_usage: str,
    verdict: str,
    failure_reason: str | None,
) -> None:
    lines: list[str] = [
        "# Release Run Summary",
        "",
        f"- Timestamp (UTC): {timestamp}",
        f"- Log bundle: `{logs_dir}`",
        f"- README: `{readme_path}`",
        "",
        f"## Verdict: {verdict}",
    ]
    if failure_reason:
        lines.extend(["", "### Failure", failure_reason])

    lines.extend(
        [
            "",
            "## Environment Snapshot",
            f"- Git commit: `{environment.get('git_commit', 'unavailable')}`",
            f"- Python: `{environment.get('python_version', 'unavailable')}`",
            f"- Torch/Transformers: `{environment.get('torch_transformers', 'unavailable')}`",
            "- Files: `git_commit.txt`, `python_version.txt`, `pip_freeze.txt`, "
            "`nvidia_smi.txt`, `torch_transformers_versions.txt`",
            "",
            "## Commands Executed",
        ]
    )
    lines.extend(_render_command_table(command_records))
    lines.extend(["", "## Key Metrics (Best by Primary Metric per Setting)"])
    lines.extend(_render_key_metrics_table(key_metrics))
    lines.extend(["", "## Artifact Existence Checks"])
    lines.extend(_render_artifact_checks_table(artifact_checks))
    lines.extend(["", "## Tripwire Sanity Checks"])
    lines.extend(_render_tripwire_table(tripwire_rows))
    lines.extend(["", "## Extraction Failures"])
    lines.extend(_render_extraction_failures(extraction_failures))
    lines.extend(["", "## Disk Usage", f"- `du -sh artifacts/`: {disk_usage}"])

    if readme_blocks is not None:
        lines.extend(
            [
                "",
                "## README Marker Checks",
                f"- Experiment block lines: {len(readme_blocks['experiment_block'].splitlines())}",
                f"- Scaling block lines: {len(readme_blocks['scaling_block'].splitlines())}",
            ]
        )

    summary_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    timestamp = args.timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    logs_dir = _resolve_repo_relative_path(args.logs_root, repo_root) / timestamp
    logs_dir.mkdir(parents=True, exist_ok=True)
    readme_path = _resolve_repo_relative_path(args.readme_path, repo_root)

    command_records: list[CommandRecord] = []
    key_metrics: list[dict[str, Any]] = []
    artifact_checks: list[dict[str, Any]] = []
    tripwire_rows: list[dict[str, Any]] = []
    extraction_failures: list[dict[str, Any]] = []
    readme_blocks: dict[str, str] | None = None
    failure_reason: str | None = None
    verdict = "NOT OK"

    environment = _write_environment_snapshot(
        repo_root=repo_root,
        logs_dir=logs_dir,
        python_executable=args.python_executable,
    )

    pilot_spec = _build_run_spec("pilot", args.pilot_config, repo_root)
    full_spec = _build_run_spec("full", args.full_config, repo_root)
    scaling_specs = [
        _build_run_spec(f"scaling:{Path(config_path).stem}", config_path, repo_root)
        for config_path in args.scaling_configs
    ]
    all_specs = [pilot_spec, full_spec, *scaling_specs]
    single_split_specs = [spec for spec in all_specs if spec.split_strategy == "single_split"]

    lopo_run_roots: list[tuple[str, Path]] = []
    for config_path in args.lopo_configs:
        config_name = Path(config_path).stem
        run_root = repo_root / "artifacts" / "runs" / "release_lopo" / timestamp / config_name
        lopo_run_roots.append((f"lopo:{config_name}", run_root))

    try:
        _run_command(
            stage="ruff_check",
            command=[args.python_executable, "-m", "ruff", "check", "."],
            cwd=repo_root,
            logs_dir=logs_dir,
            records=command_records,
        )
        _run_command(
            stage="pytest",
            command=[args.python_executable, "-m", "pytest", "-q"],
            cwd=repo_root,
            logs_dir=logs_dir,
            records=command_records,
        )

        experiments_cmd = [
            args.python_executable,
            "scripts/run_experiments.py",
            "--config",
            args.config,
            "--pilot-config",
            args.pilot_config,
            "--full-config",
            args.full_config,
            "--problem-id",
            str(args.problem_id),
            "--readme-path",
            args.readme_path,
        ]
        _run_command(
            stage="run_experiments",
            command=experiments_cmd,
            cwd=repo_root,
            logs_dir=logs_dir,
            records=command_records,
        )

        scaling_cmd = [
            args.python_executable,
            "scripts/run_scaling_grid.py",
            "--configs",
            *args.scaling_configs,
            "--seeds",
            *[str(seed) for seed in args.scaling_seeds],
            "--readme-path",
            args.readme_path,
        ]
        if args.no_reuse_cache:
            scaling_cmd.append("--no-reuse-cache")
        _run_command(
            stage="run_scaling_grid",
            command=scaling_cmd,
            cwd=repo_root,
            logs_dir=logs_dir,
            records=command_records,
        )

        for (_lopo_label, run_root), config_path in zip(
            lopo_run_roots,
            args.lopo_configs,
            strict=True,
        ):
            lopo_cmd = [
                args.python_executable,
                "scripts/run_lopo_cv.py",
                "--config",
                config_path,
                "--run-root",
                str(run_root),
                "--seeds",
                *[str(seed) for seed in args.lopo_seeds],
                "--skip-failed",
            ]
            if not args.no_reuse_cache:
                lopo_cmd.append("--reuse-cache")
            _run_command(
                stage=f"run_lopo_cv:{Path(config_path).stem}",
                command=lopo_cmd,
                cwd=repo_root,
                logs_dir=logs_dir,
                records=command_records,
            )

        artifact_checks = _validate_single_split_artifacts(single_split_specs)
        tripwire_rows = _validate_tripwires(single_split_specs)
        key_metrics = _validate_primary_metric_not_nan(single_split_specs, lopo_run_roots)
        readme_blocks = _validate_readme_markers(readme_path)
        extraction_failures = _collect_extraction_failures(single_split_specs)
        verdict = "OK"
    except ReleaseError as exc:
        failure_reason = str(exc)
        verdict = "NOT OK"
    except Exception as exc:  # pragma: no cover - defensive catch for unexpected failures.
        failure_reason = f"Unhandled failure: {exc}"
        verdict = "NOT OK"

    du_result = _run_optional_command(["du", "-sh", "artifacts"], cwd=repo_root)
    disk_usage_output = (du_result["stdout"] or du_result["stderr"]).strip() or "unavailable"

    summary_path = logs_dir / "RUN_SUMMARY.md"
    _write_summary(
        summary_path=summary_path,
        timestamp=timestamp,
        logs_dir=logs_dir,
        environment=environment,
        command_records=command_records,
        key_metrics=key_metrics,
        artifact_checks=artifact_checks,
        tripwire_rows=tripwire_rows,
        extraction_failures=extraction_failures,
        readme_path=readme_path,
        readme_blocks=readme_blocks,
        disk_usage=disk_usage_output,
        verdict=verdict,
        failure_reason=failure_reason,
    )

    payload = {
        "timestamp": timestamp,
        "verdict": verdict,
        "logs_dir": str(logs_dir),
        "summary_path": str(summary_path),
        "commands_executed": len(command_records),
        "failure": failure_reason,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if verdict != "OK":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
