"""Markdown report writer for probe experiment outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _format_metric(value: float | int | bool) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}"


def _model_metric_table(metrics: dict[str, Any], split_name: str) -> str:
    lines = [
        f"### {split_name.title()} Metrics",
        "",
        "| Model | PR AUC | Spearman | Top-5 Recall | Top-10 Recall |",
        "|---|---:|---:|---:|---:|",
    ]
    for model_name, model_metrics in metrics.items():
        top5 = model_metrics.get("top_5_recall", float("nan"))
        top10 = model_metrics.get("top_10_recall", float("nan"))
        line = (
            f"| {model_name} | {_format_metric(model_metrics['pr_auc'])} "
            f"| {_format_metric(model_metrics['spearman_mean'])} "
            f"| {_format_metric(top5)} | {_format_metric(top10)} |"
        )
        lines.append(line)
    lines.append("")
    return "\n".join(lines)


def _qualitative_examples(predictions: pd.DataFrame, model_col: str, top_n: int = 5) -> str:
    top = predictions.nlargest(top_n, model_col)
    lines = [f"### Top {top_n} Predictions ({model_col})", ""]
    for _, row in top.iterrows():
        lines.append(
            "- "
            f"Problem {int(row['problem_id'])}, chunk {int(row['chunk_idx'])}, "
            f"pred={row[model_col]:.4f}, true_importance={row['importance_score']:.4f}, "
            f"anchor={int(row['anchor'])}: {str(row['chunk_text'])[:160]}"
        )
    lines.append("")
    return "\n".join(lines)


def _confidence_interval_section(metrics: dict[str, Any]) -> str:
    ci_payload = metrics.get("confidence_intervals", {})
    if not ci_payload:
        return ""

    lines = [
        "## Bootstrap Confidence Intervals",
        "",
        "| Comparison | Point delta | Delta mean | 95% CI low | 95% CI high | Excludes 0 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for comparison, payload in sorted(ci_payload.items()):
        ci_low = float(payload.get("ci_low", float("nan")))
        ci_high = float(payload.get("ci_high", float("nan")))
        excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)
        lines.append(
            f"| {comparison} | {float(payload.get('point_delta', float('nan'))):.4f} "
            f"| {float(payload.get('delta_mean', float('nan'))):.4f} "
            f"| {ci_low:.4f} | {ci_high:.4f} | {excludes_zero} |"
        )
    lines.append("")
    return "\n".join(lines)


def _position_bin_section(metrics: dict[str, Any]) -> str:
    payload = metrics.get("position_bin_metrics", {})
    test_bins = payload.get("test", {})
    num_bins = int(payload.get("num_bins", 0))
    if not test_bins or num_bins <= 0:
        return ""

    lines = ["## Position-Bin Diagnostics (Test PR AUC)", ""]
    for model_name, bins in sorted(test_bins.items()):
        lines.extend(
            [
                f"### {model_name}",
                "",
                "| Bin | Count | Prevalence | PR AUC |",
                "|---:|---:|---:|---:|",
            ]
        )
        for item in bins:
            lines.append(
                f"| {int(item['bin_index'])} | {int(item['count'])} "
                f"| {_format_metric(float(item['prevalence']))} "
                f"| {_format_metric(float(item['pr_auc']))} |"
            )
        lines.append("")
    return "\n".join(lines)


def build_report(
    *,
    metrics_path: str | Path,
    predictions_path: str | Path,
    output_path: str | Path,
    top_n_examples: int = 5,
) -> str:
    """Build and save a markdown report from saved artifacts."""
    with Path(metrics_path).open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    predictions = pd.read_parquet(predictions_path)

    lines: list[str] = [
        "# Thought Anchor Probe Report",
        "",
        "This report summarizes sentence-level probe performance.",
        "",
        "## Dataset Summary",
        "",
        f"- Train rows: {metrics['rows']['train']}",
        f"- Validation rows: {metrics['rows']['val']}",
        f"- Test rows: {metrics['rows']['test']}",
        f"- Train problems: {metrics['splits']['train']}",
        f"- Validation problems: {metrics['splits']['val']}",
        f"- Test problems: {metrics['splits']['test']}",
        "",
    ]

    lines.append(_model_metric_table(metrics["val"], "validation"))
    lines.append(_model_metric_table(metrics["test"], "test"))

    if metrics.get("tripwires"):
        lines.extend(["## Tripwire Checks", ""])
        for name, payload in metrics["tripwires"].items():
            lines.append(f"### {name}")
            lines.append("")
            for key, value in payload.items():
                lines.append(f"- {key}: {_format_metric(value)}")
            lines.append("")

    ci_section = _confidence_interval_section(metrics)
    if ci_section:
        lines.append(ci_section)

    position_section = _position_bin_section(metrics)
    if position_section:
        lines.append(position_section)

    score_cols = [column for column in predictions.columns if column.startswith("score_")]
    for score_col in score_cols:
        lines.append(_qualitative_examples(predictions, score_col, top_n=top_n_examples))

    output_text = "\n".join(lines).strip() + "\n"

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(output_text, encoding="utf-8")
    return output_text
