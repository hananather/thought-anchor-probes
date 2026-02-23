#!/usr/bin/env python
"""Aggregate results across a systematic sweep campaign."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from ta_probe.sweep import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-root",
        required=True,
        help="Sweep root containing manifest, registry, and run outputs.",
    )
    parser.add_argument(
        "--manifest",
        default=None,
        help="Optional manifest path override. Defaults to <sweep-root>/manifest.jsonl.",
    )
    parser.add_argument(
        "--registry",
        default=None,
        help="Optional registry path override. Defaults to <sweep-root>/run_registry.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional output directory override. Defaults to <sweep-root>/summary.",
    )
    parser.add_argument(
        "--stage2-source-stage",
        default="stage1",
        help="Source stage used to generate shortlist_stage2.json.",
    )
    parser.add_argument(
        "--stage3-source-stage",
        default="stage2",
        help="Source stage used to generate shortlist_stage3.json.",
    )
    parser.add_argument(
        "--stage2-top-n-per-target-mode",
        type=int,
        default=24,
        help="Top-N configs per target mode for stage-2 shortlist.",
    )
    parser.add_argument(
        "--stage3-top-n",
        type=int,
        default=12,
        help="Top-N configs overall for stage-3 shortlist.",
    )
    return parser.parse_args()


def _latest_status_by_config(registry_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for row in registry_rows:
        config_id = str(row.get("config_id", ""))
        if not config_id:
            continue
        latest[config_id] = row
    return latest


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {}


def _best_row_from_payload(payload: dict[str, Any]) -> tuple[str, dict[str, Any] | None, str, str]:
    rows = payload.get("fold_aggregate_by_model", [])
    if not isinstance(rows, list):
        rows = []
    mean_rows = [row for row in rows if row.get("agg_type") == "mean_seeds"]

    primary_metric_name = str(payload.get("primary_metric_name", "pr_auc"))
    primary_metric = str(payload.get("primary_metric", "pr_auc"))
    metric_mean_col = (
        primary_metric if primary_metric.endswith("_mean") else f"{primary_metric}_mean"
    )

    best_model = payload.get("best_model_by_primary_metric")
    if best_model is None and mean_rows:
        ranked = sorted(
            mean_rows,
            key=lambda row: float(row.get(metric_mean_col, float("-inf"))),
            reverse=True,
        )
        best_model = ranked[0].get("model")

    if best_model is None:
        return "n/a", None, primary_metric_name, primary_metric

    for row in mean_rows:
        if str(row.get("model")) == str(best_model):
            return str(best_model), row, primary_metric_name, primary_metric
    return str(best_model), None, primary_metric_name, primary_metric


def _sorted_leaderboard(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("target_mode", "")),
            -float(row.get("primary_metric_mean", float("-inf"))),
            str(row.get("config_id", "")),
        ),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _select_stage2_shortlist(
    rows: list[dict[str, Any]],
    *,
    source_stage: str,
    top_n_per_target_mode: int,
) -> list[dict[str, Any]]:
    filtered = [
        row for row in rows if row.get("stage") == source_stage and row.get("status") == "success"
    ]
    by_target: dict[str, list[dict[str, Any]]] = {}
    for row in filtered:
        target_mode = str(row.get("target_mode", "unknown"))
        by_target.setdefault(target_mode, []).append(row)

    selected: list[dict[str, Any]] = []
    for _target_mode, group in sorted(by_target.items()):
        ranked = sorted(
            group,
            key=lambda row: (
                -float(row.get("primary_metric_mean", float("-inf"))),
                str(row.get("config_id", "")),
            ),
        )
        selected.extend(ranked[:top_n_per_target_mode])
    return selected


def _select_stage3_shortlist(
    rows: list[dict[str, Any]],
    *,
    source_stage: str,
    top_n: int,
) -> list[dict[str, Any]]:
    filtered = [
        row for row in rows if row.get("stage") == source_stage and row.get("status") == "success"
    ]
    ranked = sorted(
        filtered,
        key=lambda row: (
            -float(row.get("primary_metric_mean", float("-inf"))),
            str(row.get("config_id", "")),
        ),
    )
    return ranked[:top_n]


def _write_shortlist(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = [
        {
            "config_id": row.get("config_id"),
            "config_path": row.get("config_path"),
            "run_root": row.get("run_root"),
            "target_mode": row.get("target_mode"),
            "primary_metric_name": row.get("primary_metric_name"),
            "primary_metric_mean": row.get("primary_metric_mean"),
            "stage": row.get("stage"),
        }
        for row in rows
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    sweep_root = Path(args.sweep_root)
    if not sweep_root.is_absolute():
        sweep_root = repo_root / sweep_root

    manifest_path = Path(args.manifest) if args.manifest else sweep_root / "manifest.jsonl"
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path

    registry_path = Path(args.registry) if args.registry else sweep_root / "run_registry.jsonl"
    if not registry_path.is_absolute():
        registry_path = repo_root / registry_path

    output_dir = Path(args.output_dir) if args.output_dir else sweep_root / "summary"
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = load_jsonl(manifest_path)
    registry_rows = load_jsonl(registry_path)
    latest_status = _latest_status_by_config(registry_rows)

    leaderboard_rows: list[dict[str, Any]] = []
    paired_delta_rows: list[dict[str, Any]] = []
    stability_rows: list[dict[str, Any]] = []

    status_counts: dict[str, int] = {}

    for row in manifest_rows:
        config_id = str(row.get("config_id", ""))
        if not config_id:
            continue
        stage = str(row.get("stage", ""))
        factor_values = row.get("factor_values", {})
        if not isinstance(factor_values, dict):
            factor_values = {}

        status_payload = latest_status.get(config_id, {})
        status = str(status_payload.get("status", "pending"))
        status_counts[status] = status_counts.get(status, 0) + 1

        run_root = Path(str(row.get("run_root", "")))
        if not run_root.is_absolute():
            run_root = repo_root / run_root

        aggregate_payload = (
            _load_json(run_root / "aggregate_lopo_metrics.json") if status == "success" else {}
        )
        best_model, best_row, primary_metric_name, primary_metric = _best_row_from_payload(
            aggregate_payload
        )
        metric_mean_col = (
            primary_metric if primary_metric.endswith("_mean") else f"{primary_metric}_mean"
        )
        metric_std_col = (
            primary_metric if primary_metric.endswith("_std") else f"{primary_metric}_std"
        )

        leaderboard_entry = {
            "config_id": config_id,
            "stage": stage,
            "status": status,
            "config_path": row.get("config_path"),
            "run_root": str(run_root),
            "target_mode": factor_values.get("target_mode"),
            "split_dir": factor_values.get("split_dir"),
            "pooling": factor_values.get("pooling"),
            "layer_mode": factor_values.get("layer_mode"),
            "layer_index": factor_values.get("layer_index"),
            "residualize_against": factor_values.get("residualize_against"),
            "vertical_attention_mode": factor_values.get("vertical_attention_mode"),
            "best_model": best_model,
            "primary_metric_name": primary_metric_name,
            "primary_metric": primary_metric,
            "primary_metric_mean": (
                float(best_row.get(metric_mean_col, float("nan")))
                if isinstance(best_row, dict)
                else float("nan")
            ),
            "primary_metric_std": (
                float(best_row.get(metric_std_col, float("nan")))
                if isinstance(best_row, dict)
                else float("nan")
            ),
            "pr_auc_mean": (
                float(best_row.get("pr_auc_mean", float("nan")))
                if isinstance(best_row, dict)
                else float("nan")
            ),
            "spearman_mean_mean": (
                float(best_row.get("spearman_mean_mean", float("nan")))
                if isinstance(best_row, dict)
                else float("nan")
            ),
            "num_folds": int(aggregate_payload.get("num_folds", 0)) if aggregate_payload else 0,
        }
        leaderboard_rows.append(leaderboard_entry)

        if aggregate_payload:
            for delta in aggregate_payload.get("paired_delta_summary", []):
                if not isinstance(delta, dict):
                    continue
                paired_delta_rows.append(
                    {
                        "config_id": config_id,
                        "stage": stage,
                        "target_mode": factor_values.get("target_mode"),
                        "comparison": delta.get("comparison"),
                        "agg_type": delta.get("agg_type"),
                        "mean": delta.get("mean"),
                        "std": delta.get("std"),
                        "bootstrap_ci_low": delta.get("bootstrap_ci_low"),
                        "bootstrap_ci_high": delta.get("bootstrap_ci_high"),
                        "n_folds": delta.get("n_folds"),
                    }
                )

            if isinstance(best_row, dict):
                stability_rows.append(
                    {
                        "config_id": config_id,
                        "stage": stage,
                        "target_mode": factor_values.get("target_mode"),
                        "best_model": best_model,
                        "primary_metric_name": primary_metric_name,
                        "primary_metric_std": leaderboard_entry["primary_metric_std"],
                        "pr_auc_std": best_row.get("pr_auc_std"),
                        "spearman_mean_std": best_row.get("spearman_mean_std"),
                        "top_5_recall_std": best_row.get("top_5_recall_std"),
                        "top_10_recall_std": best_row.get("top_10_recall_std"),
                        "num_folds": int(aggregate_payload.get("num_folds", 0)),
                    }
                )

    leaderboard_sorted = _sorted_leaderboard(leaderboard_rows)

    leaderboard_path = output_dir / "leaderboard.csv"
    deltas_path = output_dir / "paired_deltas.csv"
    stability_path = output_dir / "stability_report.csv"

    _write_csv(leaderboard_path, leaderboard_sorted)
    _write_csv(deltas_path, paired_delta_rows)
    _write_csv(stability_path, stability_rows)

    shortlist_stage2 = _select_stage2_shortlist(
        leaderboard_sorted,
        source_stage=str(args.stage2_source_stage),
        top_n_per_target_mode=int(args.stage2_top_n_per_target_mode),
    )
    shortlist_stage3 = _select_stage3_shortlist(
        leaderboard_sorted,
        source_stage=str(args.stage3_source_stage),
        top_n=int(args.stage3_top_n),
    )

    shortlist_stage2_path = output_dir / "shortlist_stage2.json"
    shortlist_stage3_path = output_dir / "shortlist_stage3.json"
    _write_shortlist(shortlist_stage2_path, shortlist_stage2)
    _write_shortlist(shortlist_stage3_path, shortlist_stage3)

    top_success = [row for row in leaderboard_sorted if row.get("status") == "success"][:10]

    summary_lines = [
        "# Sweep Summary",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Registry: `{registry_path}`",
        f"- Total configs: {len(leaderboard_rows)}",
        f"- Status counts: {status_counts}",
        "",
        "## Top Successful Configs",
        "",
        "| Rank | Config | Stage | Target | Primary metric | Primary mean | Best model |",
        "|---:|---|---|---|---|---:|---|",
    ]
    for rank, row in enumerate(top_success, start=1):
        primary_metric_mean = float(row.get("primary_metric_mean", float("nan")))
        summary_lines.append(
            "| "
            f"{rank} | {row['config_id']} | {row['stage']} | {row.get('target_mode')} | "
            f"{row.get('primary_metric_name')} | {primary_metric_mean:.4f} | "
            f"{row.get('best_model')} |"
        )
    summary_lines.extend(
        [
            "",
            "## Shortlist Files",
            "",
            f"- Stage-2 shortlist: `{shortlist_stage2_path}`",
            f"- Stage-3 shortlist: `{shortlist_stage3_path}`",
            "",
        ]
    )

    summary_md_path = output_dir / "summary.md"
    summary_md_path.write_text("\n".join(summary_lines), encoding="utf-8")

    payload = {
        "manifest_path": str(manifest_path),
        "registry_path": str(registry_path),
        "leaderboard_csv": str(leaderboard_path),
        "paired_deltas_csv": str(deltas_path),
        "stability_report_csv": str(stability_path),
        "summary_md": str(summary_md_path),
        "shortlist_stage2": str(shortlist_stage2_path),
        "shortlist_stage3": str(shortlist_stage3_path),
        "status_counts": status_counts,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
