"""Aggregate multi-seed probe metrics into summary tables."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

SEED_METRIC_COLUMNS = [
    "pr_auc",
    "spearman_mean",
    "top_5_recall",
    "top_10_recall",
]


def discover_metric_files(run_root: str | Path) -> list[Path]:
    """Find per-run metrics files under a run root directory.

    Prefer `metrics_seed_*.json` and ignore ad-hoc backups like
    `metrics_before_*.json`.
    """
    root = Path(run_root)
    seed_files = sorted(root.glob("metrics_seed_*.json"))
    if seed_files:
        return seed_files
    default_file = root / "metrics.json"
    if default_file.exists():
        return [default_file]
    return []


def _run_label_from_path(path: Path) -> str:
    """Extract run label from a metrics file name."""
    stem = path.stem
    if stem == "metrics":
        return "default"
    if stem.startswith("metrics_"):
        return stem[len("metrics_") :]
    return stem


def _load_metrics_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _flatten_seed_records(payload: dict[str, Any], run_label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seed = int(payload.get("seed", -1))

    for split_name in ["val", "test"]:
        split_metrics = payload.get(split_name, {})
        for model_name, model_metrics in split_metrics.items():
            record = {
                "run_label": run_label,
                "seed": seed,
                "split": split_name,
                "model": model_name,
            }
            for column in SEED_METRIC_COLUMNS:
                record[column] = float(model_metrics.get(column, float("nan")))
            records.append(record)

    return records


def _aggregate_summary(records: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean and std metrics per model using test split records."""
    test_records = records[records["split"] == "test"].copy()
    grouped = test_records.groupby("model", as_index=False)[SEED_METRIC_COLUMNS]
    mean_df = grouped.mean().rename(
        columns={column: f"{column}_mean" for column in SEED_METRIC_COLUMNS}
    )
    std_df = grouped.std(ddof=0).rename(
        columns={column: f"{column}_std" for column in SEED_METRIC_COLUMNS}
    )

    summary = mean_df.merge(std_df, on="model", how="left")
    return summary.sort_values("pr_auc_mean", ascending=False, ignore_index=True)


def _flatten_ci_records(payload: dict[str, Any], run_label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seed = int(payload.get("seed", -1))
    ci_payload = payload.get("confidence_intervals", {})
    for comparison, metrics in ci_payload.items():
        ci_low = float(metrics.get("ci_low", float("nan")))
        ci_high = float(metrics.get("ci_high", float("nan")))
        excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)
        records.append(
            {
                "run_label": run_label,
                "seed": seed,
                "comparison": comparison,
                "point_delta": float(metrics.get("point_delta", float("nan"))),
                "delta_mean": float(metrics.get("delta_mean", float("nan"))),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_valid_bootstrap": int(metrics.get("n_valid_bootstrap", 0)),
                "n_bootstrap": int(metrics.get("n_bootstrap", 0)),
                "ci_excludes_zero": excludes_zero,
            }
        )
    return records


def _aggregate_ci_summary(ci_records: pd.DataFrame) -> pd.DataFrame:
    if ci_records.empty:
        return pd.DataFrame(
            columns=[
                "comparison",
                "point_delta_mean",
                "point_delta_std",
                "delta_mean_mean",
                "delta_mean_std",
                "ci_excludes_zero_count",
                "seed_count",
            ]
        )
    grouped = ci_records.groupby("comparison", as_index=False)
    summary = grouped.agg(
        point_delta_mean=("point_delta", "mean"),
        point_delta_std=("point_delta", lambda series: float(series.std(ddof=0))),
        delta_mean_mean=("delta_mean", "mean"),
        delta_mean_std=("delta_mean", lambda series: float(series.std(ddof=0))),
        ci_excludes_zero_count=("ci_excludes_zero", "sum"),
        seed_count=("ci_excludes_zero", "count"),
    )
    return summary.sort_values("comparison", ignore_index=True)


def _summary_markdown(seed_records: pd.DataFrame, summary_records: pd.DataFrame) -> str:
    """Build a markdown summary table for README-friendly reporting."""
    lines: list[str] = [
        "# Aggregated Probe Metrics",
        "",
        "## Per-Seed Test Metrics",
        "",
        "| Run | Seed | Model | PR AUC | Spearman | Top-5 | Top-10 |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]

    test_rows = seed_records[seed_records["split"] == "test"].sort_values(
        ["run_label", "model"], ignore_index=True
    )
    for _, row in test_rows.iterrows():
        lines.append(
            f"| {row['run_label']} | {int(row['seed'])} | {row['model']} "
            f"| {row['pr_auc']:.4f} | {row['spearman_mean']:.4f} "
            f"| {row['top_5_recall']:.4f} | {row['top_10_recall']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Mean and Std Across Seeds (Test)",
            "",
            (
                "| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std "
                "| Top-5 mean | Top-5 std | Top-10 mean | Top-10 std |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )

    for _, row in summary_records.iterrows():
        lines.append(
            f"| {row['model']} "
            f"| {row['pr_auc_mean']:.4f} | {row['pr_auc_std']:.4f} "
            f"| {row['spearman_mean_mean']:.4f} | {row['spearman_mean_std']:.4f} "
            f"| {row['top_5_recall_mean']:.4f} | {row['top_5_recall_std']:.4f} "
            f"| {row['top_10_recall_mean']:.4f} | {row['top_10_recall_std']:.4f} |"
        )

    lines.append("")
    return "\n".join(lines)


def _ci_markdown(ci_records: pd.DataFrame, ci_summary: pd.DataFrame) -> str:
    if ci_records.empty:
        return ""

    lines = [
        "## Bootstrap CI Summary (Test Deltas)",
        "",
        (
            "| Comparison | Point delta mean | Point delta std | Bootstrap delta mean "
            "| Bootstrap delta std | Seeds CI excludes 0 | Seeds |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in ci_summary.iterrows():
        lines.append(
            f"| {row['comparison']} | {row['point_delta_mean']:.4f} | {row['point_delta_std']:.4f} "
            f"| {row['delta_mean_mean']:.4f} | {row['delta_mean_std']:.4f} "
            f"| {int(row['ci_excludes_zero_count'])} | {int(row['seed_count'])} |"
        )

    lines.extend(
        [
            "",
            "### Per-Seed CIs",
            "",
            "| Run | Seed | Comparison | CI low | CI high | Excludes 0 |",
            "|---|---:|---|---:|---:|---|",
        ]
    )
    for _, row in ci_records.sort_values(["run_label", "comparison"], ignore_index=True).iterrows():
        lines.append(
            f"| {row['run_label']} | {int(row['seed'])} | {row['comparison']} "
            f"| {row['ci_low']:.4f} | {row['ci_high']:.4f} | {bool(row['ci_excludes_zero'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def aggregate_run_metrics(
    run_root: str | Path,
    output_json_path: str | Path | None = None,
    output_md_path: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate per-seed metrics from one run root."""
    root = Path(run_root)
    metric_files = discover_metric_files(root)
    if not metric_files:
        msg = f"No metrics files found under {root}"
        raise FileNotFoundError(msg)

    records: list[dict[str, Any]] = []
    ci_records: list[dict[str, Any]] = []
    for metric_file in metric_files:
        payload = _load_metrics_file(metric_file)
        run_label = _run_label_from_path(metric_file)
        records.extend(_flatten_seed_records(payload, run_label))
        ci_records.extend(_flatten_ci_records(payload, run_label))

    seed_df = pd.DataFrame(records)
    summary_df = _aggregate_summary(seed_df)
    ci_df = pd.DataFrame(ci_records)
    ci_summary_df = _aggregate_ci_summary(ci_df)
    best_model = summary_df.iloc[0]["model"] if not summary_df.empty else None

    summary_payload = {
        "run_root": str(root),
        "metric_files": [str(path) for path in metric_files],
        "num_runs": int(seed_df["run_label"].nunique()),
        "num_seed_records": int(len(seed_df)),
        "best_model_by_mean_pr_auc": best_model,
        "seed_records": seed_df.to_dict(orient="records"),
        "summary_by_model": summary_df.to_dict(orient="records"),
        "ci_records": ci_df.to_dict(orient="records"),
        "ci_summary": ci_summary_df.to_dict(orient="records"),
    }

    if output_json_path is None:
        output_json_path = root / "aggregate_metrics.json"
    if output_md_path is None:
        output_md_path = root / "aggregate_metrics.md"

    json_path = Path(output_json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)

    md_path = Path(output_md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    markdown = _summary_markdown(seed_df, summary_df)
    ci_markdown = _ci_markdown(ci_df, ci_summary_df)
    if ci_markdown:
        markdown = f"{markdown}\n{ci_markdown}".strip() + "\n"
    md_path.write_text(markdown, encoding="utf-8")

    summary_payload["output_json"] = str(json_path)
    summary_payload["output_md"] = str(md_path)
    return summary_payload
