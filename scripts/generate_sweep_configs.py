#!/usr/bin/env python
"""Generate config manifests for large systematic sweep stages."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.config import load_config
from ta_probe.sweep import (
    apply_factor_values,
    build_lopo_commands,
    build_stage1_factor_values,
    build_stage2_factor_expansions,
    build_stage3_factor_expansions,
    factor_values_from_config,
    infer_num_layers,
    load_jsonl,
    make_config_id,
    utc_now_iso,
    write_config_yaml,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-config",
        default="configs/scaling_qwen_correct.yaml",
        help="Base config used to generate derived sweep configs.",
    )
    parser.add_argument(
        "--sweep-root",
        required=True,
        help="Sweep root directory (for configs, manifests, and run roots).",
    )
    parser.add_argument(
        "--sweep-id",
        default=None,
        help="Optional explicit sweep ID. Defaults to sweep-root directory name.",
    )
    parser.add_argument(
        "--stage",
        choices=["stage1", "stage2", "stage3"],
        required=True,
        help="Sweep stage to generate.",
    )
    parser.add_argument(
        "--seed-configs",
        default=None,
        help=(
            "JSON list of seed configs for stage2/stage3. Each row can be a config path string "
            "or an object containing config_path."
        ),
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=None,
        help="Optional transformer layer count override (used for stage1 quartile layer indices).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(20)),
        help="Training seeds passed to run_lopo_cv for each config.",
    )
    parser.add_argument(
        "--limit-configs",
        type=int,
        default=None,
        help="Optional cap on generated configs for smoke/debug runs.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help="Bootstrap iterations passed to aggregate_runs for generated commands.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=0,
        help="Bootstrap seed passed to aggregate_runs for generated commands.",
    )
    parser.add_argument(
        "--best-of-k",
        type=int,
        default=1,
        help="best-of-k seed selection passed to aggregate_runs for generated commands.",
    )
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Include --reuse-cache in generated run_lopo_cv commands.",
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        help="Include --skip-failed in generated run_lopo_cv commands.",
    )
    parser.add_argument(
        "--run-tripwires",
        action="store_true",
        help="Run tripwire checks in generated run_lopo_cv commands (default skips tripwires).",
    )
    return parser.parse_args()


def _normalize_seed_configs(path: Path) -> list[Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("seed-configs payload must be a JSON list")

    result: list[Path] = []
    for row in payload:
        if isinstance(row, str):
            result.append(Path(row))
            continue
        if isinstance(row, dict) and "config_path" in row:
            result.append(Path(str(row["config_path"])))
            continue
        raise ValueError("Each seed-configs row must be a string path or object with config_path")
    return result


def _manifest_stage_sort_key(row: dict[str, Any]) -> tuple[str, str]:
    return (str(row.get("stage", "")), str(row.get("config_id", "")))


def _merge_manifests(
    existing: list[dict[str, Any]], new_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in existing:
        config_id = str(row.get("config_id", ""))
        if not config_id:
            continue
        merged[config_id] = row
    for row in new_rows:
        merged[str(row["config_id"])] = row
    return sorted(merged.values(), key=_manifest_stage_sort_key)


def _build_stage1_specs(
    *,
    base_config_path: Path,
    num_layers: int,
) -> list[tuple[dict[str, Any], Any]]:
    base_config = load_config(base_config_path)
    rows = build_stage1_factor_values(num_layers=num_layers)
    return [(row, base_config) for row in rows]


def _build_stage2_specs(
    *,
    seed_config_paths: list[Path],
) -> list[tuple[dict[str, Any], Any]]:
    expansions = build_stage2_factor_expansions()
    output: list[tuple[dict[str, Any], Any]] = []
    for seed_config_path in seed_config_paths:
        seed_config = load_config(seed_config_path)
        parent_factors = factor_values_from_config(seed_config)
        parent_config_id = seed_config_path.stem
        for expansion in expansions:
            factors = dict(parent_factors)
            factors.update(expansion)
            factors["parent_config_id"] = parent_config_id
            output.append((factors, seed_config))
    return output


def _build_stage3_specs(
    *,
    seed_config_paths: list[Path],
) -> list[tuple[dict[str, Any], Any]]:
    output: list[tuple[dict[str, Any], Any]] = []
    for seed_config_path in seed_config_paths:
        seed_config = load_config(seed_config_path)
        parent_factors = factor_values_from_config(seed_config)
        parent_config_id = seed_config_path.stem
        expansions = build_stage3_factor_expansions(seed_config.labels.target_mode)
        for expansion in expansions:
            factors = dict(parent_factors)
            factors.update(expansion)
            factors["parent_config_id"] = parent_config_id
            output.append((factors, seed_config))
    return output


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    base_config_path = Path(args.base_config)
    if not base_config_path.is_absolute():
        base_config_path = repo_root / base_config_path

    sweep_root = Path(args.sweep_root)
    if not sweep_root.is_absolute():
        sweep_root = repo_root / sweep_root
    sweep_root.mkdir(parents=True, exist_ok=True)

    sweep_id = str(args.sweep_id) if args.sweep_id else sweep_root.name

    if args.stage == "stage1":
        base_config = load_config(base_config_path)
        num_layers = infer_num_layers(
            base_config.activations.model_name_or_path,
            explicit_num_layers=args.num_layers,
        )
        stage_specs = _build_stage1_specs(base_config_path=base_config_path, num_layers=num_layers)
    else:
        if not args.seed_configs:
            raise ValueError("--seed-configs is required for stage2/stage3")
        seed_configs_path = Path(args.seed_configs)
        if not seed_configs_path.is_absolute():
            seed_configs_path = repo_root / seed_configs_path
        seed_config_paths = _normalize_seed_configs(seed_configs_path)
        if args.stage == "stage2":
            stage_specs = _build_stage2_specs(seed_config_paths=seed_config_paths)
        else:
            stage_specs = _build_stage3_specs(seed_config_paths=seed_config_paths)

    if args.limit_configs is not None:
        stage_specs = stage_specs[: int(args.limit_configs)]

    stage_rows: list[dict[str, Any]] = []
    for factor_values, source_config in stage_specs:
        config_id = make_config_id(args.stage, factor_values)
        run_root = sweep_root / "runs" / args.stage / config_id
        config_path = sweep_root / "configs" / args.stage / f"{config_id}.yaml"

        config = apply_factor_values(
            base_config=source_config,
            factor_values=factor_values,
            run_root=run_root,
        )
        write_config_yaml(config, config_path)

        commands = build_lopo_commands(
            config_path=config_path,
            run_root=run_root,
            seeds=[int(seed) for seed in args.seeds],
            reuse_cache=bool(args.reuse_cache),
            skip_failed=bool(args.skip_failed),
            no_tripwires=not bool(args.run_tripwires),
            bootstrap_iterations=int(args.bootstrap_iterations),
            bootstrap_seed=int(args.bootstrap_seed),
            best_of_k=int(args.best_of_k),
        )

        stage_rows.append(
            {
                "sweep_id": sweep_id,
                "stage": args.stage,
                "config_id": config_id,
                "parent_config_id": factor_values.get("parent_config_id"),
                "config_path": str(config_path),
                "run_root": str(run_root),
                "factor_values": factor_values,
                "commands": commands,
                "created_at": utc_now_iso(),
            }
        )

    stage_rows = sorted(stage_rows, key=_manifest_stage_sort_key)

    stage_manifest_path = sweep_root / f"manifest_{args.stage}.jsonl"
    write_jsonl(stage_manifest_path, stage_rows)

    global_manifest_path = sweep_root / "manifest.jsonl"
    existing_manifest = load_jsonl(global_manifest_path)
    merged_manifest = _merge_manifests(existing_manifest, stage_rows)
    write_jsonl(global_manifest_path, merged_manifest)

    summary = {
        "sweep_id": sweep_id,
        "stage": args.stage,
        "num_stage_configs": len(stage_rows),
        "num_total_configs": len(merged_manifest),
        "stage_manifest_path": str(stage_manifest_path),
        "manifest_path": str(global_manifest_path),
        "sweep_root": str(sweep_root),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
