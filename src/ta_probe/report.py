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

    score_cols = [column for column in predictions.columns if column.startswith("score_")]
    for score_col in score_cols:
        lines.append(_qualitative_examples(predictions, score_col, top_n=top_n_examples))

    output_text = "\n".join(lines).strip() + "\n"

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(output_text, encoding="utf-8")
    return output_text
