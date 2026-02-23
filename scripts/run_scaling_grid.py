#!/usr/bin/env python
"""Run the four-setting high-confidence scaling matrix."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.config import ensure_parent_dirs, load_config
from ta_probe.data_loading import create_splits, list_problem_ids, sample_problem_ids, write_json

DEFAULT_CONFIGS = [
    "configs/scaling_qwen_correct.yaml",
    "configs/scaling_qwen_incorrect.yaml",
]
PARITY_FIELDS = (
    ("training", "train_fraction"),
    ("training", "val_fraction"),
    ("training", "test_fraction"),
    ("training", "random_seed"),
    ("dataset", "seed"),
    ("training", "k_values"),
)


def _sanitize_cache_component(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return sanitized.strip("_") or "unknown"


def discovery_cache_path_for_config(config, cache_root: Path) -> Path:
    model = _sanitize_cache_component(config.dataset.model_dir)
    split = _sanitize_cache_component(config.dataset.split_dir)
    return cache_root / f"{model}__{split}.json"


def build_discovery_cache_paths(configs: list[Any], cache_root: Path) -> list[Path]:
    paths = [discovery_cache_path_for_config(config, cache_root) for config in configs]
    unique = {str(path) for path in paths}
    if len(unique) != len(paths):
        rendered = ", ".join(str(path) for path in paths)
        msg = (
            "Discovery cache paths are not unique across configs. "
            f"Resolved paths: {rendered}"
        )
        raise ValueError(msg)
    return paths


def _resolve_config_field(config: Any, field_path: tuple[str, str]) -> Any:
    value = config
    for segment in field_path:
        value = getattr(value, segment)
    if isinstance(value, list):
        return list(value)
    return value


def validate_scaling_config_parity(
    configs: list[Any], config_paths: list[Path] | None = None
) -> None:
    if len(configs) <= 1:
        return

    labels = (
        [str(path) for path in config_paths]
        if config_paths is not None
        else [f"config[{idx}]" for idx in range(len(configs))]
    )
    mismatches: list[str] = []

    for field_path in PARITY_FIELDS:
        field_name = ".".join(field_path)
        expected = _resolve_config_field(configs[0], field_path)
        for idx, config in enumerate(configs[1:], start=1):
            value = _resolve_config_field(config, field_path)
            if value != expected:
                mismatches.append(
                    f"{field_name}: {labels[0]}={expected!r}, "
                    f"{labels[idx]}={value!r}"
                )

    if mismatches:
        joined = "; ".join(mismatches)
        msg = f"Scaling config parity mismatch across selected configs: {joined}"
        raise ValueError(msg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="+",
        default=DEFAULT_CONFIGS,
        help="Scaling config files to run.",
    )
    parser.add_argument(
        "--shared-problem-ids",
        default="data/problem_ids_scaling_shared.json",
        help="Path to write shared problem IDs used by all settings.",
    )
    parser.add_argument(
        "--shared-splits",
        default="data/splits_scaling_shared.json",
        help="Path to write shared train/val/test split IDs.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="Training seeds to run per setting.",
    )
    parser.add_argument(
        "--refresh-problem-list",
        action="store_true",
        help="Refresh dataset problem listing from Hugging Face.",
    )
    parser.add_argument(
        "--no-reuse-cache",
        action="store_true",
        help="Disable extraction cache reuse checks.",
    )
    return parser.parse_args()


def _run(command: list[str], cwd: Path) -> dict[str, Any]:
    start = time.time()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        check=False,
    )
    duration = round(time.time() - start, 3)
    record = {
        "command": " ".join(command),
        "returncode": int(completed.returncode),
        "duration_sec": duration,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout)[-3000:]
        msg = f"Command failed: {' '.join(command)}\n{tail}"
        raise RuntimeError(msg)
    return record


def _load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_markdown_summary(
    *,
    output_path: Path,
    shared_ids: list[int],
    shared_splits: dict[str, list[int]],
    per_setting: list[dict[str, Any]],
) -> None:
    lines = [
        "# Scaling Matrix Summary",
        "",
        f"- Shared problems: {len(shared_ids)}",
        (
            f"- Shared splits: train={len(shared_splits['train'])}, "
            f"val={len(shared_splits['val'])}, test={len(shared_splits['test'])}"
        ),
        "",
        "| Setting | Best model by mean PR AUC | Mean PR AUC (best) | CI rows |",
        "|---|---|---:|---:|",
    ]
    for item in per_setting:
        best_model = item.get("best_model")
        best_pr_auc = float(item.get("best_pr_auc_mean", float("nan")))
        ci_rows = int(item.get("num_ci_records", 0))
        lines.append(f"| {item['setting']} | {best_model} | {best_pr_auc:.4f} | {ci_rows} |")
    lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    config_paths = [Path(path) for path in args.configs]
    configs = [load_config(path) for path in config_paths]
    validate_scaling_config_parity(configs, config_paths=config_paths)
    for config in configs:
        ensure_parent_dirs(config)

    discovery_cache_root = repo_root / "artifacts" / "scaling" / "id_cache"
    discovery_cache_paths = build_discovery_cache_paths(configs, discovery_cache_root)

    id_sets: list[set[int]] = []
    for config, discovery_cache_path in zip(configs, discovery_cache_paths, strict=True):
        ids = list_problem_ids(
            repo_id=config.dataset.repo_id,
            model_dir=config.dataset.model_dir,
            temp_dir=config.dataset.temp_dir,
            split_dir=config.dataset.split_dir,
            cache_path=discovery_cache_path,
            force_refresh=args.refresh_problem_list,
        )
        id_sets.append({int(problem_id) for problem_id in ids})

    shared_ids = sorted(set.intersection(*id_sets))
    if not shared_ids:
        raise RuntimeError("No shared problem IDs across the selected configs")

    num_problems = min(config.dataset.num_problems for config in configs)
    split_seed = configs[0].training.random_seed
    sampling_seed = configs[0].dataset.seed
    sampled_ids = sample_problem_ids(shared_ids, num_problems=num_problems, seed=sampling_seed)
    shared_splits = create_splits(
        sampled_ids,
        train_fraction=configs[0].training.train_fraction,
        val_fraction=configs[0].training.val_fraction,
        test_fraction=configs[0].training.test_fraction,
        seed=split_seed,
    )

    shared_ids_path = Path(args.shared_problem_ids)
    shared_splits_path = Path(args.shared_splits)
    write_json(shared_ids_path, sampled_ids)
    write_json(shared_splits_path, shared_splits)
    for config in configs:
        write_json(config.paths.problem_ids_json, sampled_ids)
        write_json(config.paths.splits_json, shared_splits)

    command_log: list[dict[str, Any]] = []
    per_setting_summary: list[dict[str, Any]] = []

    for config_path, config in zip(config_paths, configs, strict=True):
        setting_name = f"{config.dataset.model_dir}__{config.dataset.split_dir}"
        run_root = Path(config.paths.metrics_json).parent
        failure_log = run_root / "extraction_failures.json"

        extract_command = [
            "python",
            "scripts/extract_embeddings.py",
            "--config",
            str(config_path),
            "--problem-ids",
            str(shared_ids_path),
            "--skip-failed",
            "--failure-log",
            str(failure_log),
        ]
        if not args.no_reuse_cache:
            extract_command.append("--reuse-cache")
        command_log.append(_run(extract_command, cwd=repo_root))

        for seed in args.seeds:
            command_log.append(
                _run(
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
            )

        command_log.append(
            _run(
                [
                    "python",
                    "scripts/aggregate_runs.py",
                    "--run-root",
                    str(run_root),
                ],
                cwd=repo_root,
            )
        )

        aggregate_payload = _load_json(run_root / "aggregate_metrics.json")
        summary_by_model = aggregate_payload.get("summary_by_model", [])
        best_model = aggregate_payload.get("best_model_by_mean_pr_auc")
        best_pr_auc = float("nan")
        for row in summary_by_model:
            if row.get("model") == best_model:
                best_pr_auc = float(row.get("pr_auc_mean", float("nan")))
                break
        per_setting_summary.append(
            {
                "setting": setting_name,
                "run_root": str(run_root),
                "best_model": best_model,
                "best_pr_auc_mean": best_pr_auc,
                "num_ci_records": len(aggregate_payload.get("ci_records", [])),
                "num_runs": int(aggregate_payload.get("num_runs", 0)),
            }
        )

    markdown_path = repo_root / "artifacts" / "scaling" / "scaling_summary.md"
    _write_markdown_summary(
        output_path=markdown_path,
        shared_ids=sampled_ids,
        shared_splits=shared_splits,
        per_setting=per_setting_summary,
    )

    summary = {
        "shared_problem_ids_path": str(shared_ids_path),
        "shared_splits_path": str(shared_splits_path),
        "num_shared_problems": len(sampled_ids),
        "num_commands": len(command_log),
        "settings": per_setting_summary,
        "summary_markdown": str(markdown_path),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
