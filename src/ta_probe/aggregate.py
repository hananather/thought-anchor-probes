"""Aggregate multi-seed probe metrics into summary tables."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

SEED_METRIC_COLUMNS = [
    "pr_auc",
    "spearman_mean",
    "top_5_recall",
    "top_10_recall",
]
RESIDUAL_METRIC_COLUMNS = [
    "residual_spearman",
    "residual_pr_auc",
]

LOPO_COMPARISONS = [
    ("activations_plus_position", "position_baseline"),
    ("linear_probe", "position_baseline"),
    ("mlp_probe", "position_baseline"),
]

FOLD_PATTERN = re.compile(r"fold_(\d+)")


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


def discover_lopo_metric_files(run_root: str | Path) -> list[Path]:
    """Find LOPO metrics files under fold directories."""
    root = Path(run_root)
    metric_files = sorted(root.glob("fold_*/*/metrics.json"))
    if metric_files:
        return metric_files
    return sorted(root.glob("fold_*/*/metrics_seed_*.json"))


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


def _flatten_residual_records(payload: dict[str, Any], run_label: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seed = int(payload.get("seed", -1))
    residual_payload = payload.get("residual_metrics", {})
    for split_name in ["val", "test"]:
        split_metrics = residual_payload.get(split_name, {})
        for model_name, model_metrics in split_metrics.items():
            record = {
                "run_label": run_label,
                "seed": seed,
                "split": split_name,
                "model": model_name,
            }
            for column in RESIDUAL_METRIC_COLUMNS:
                record[column] = float(model_metrics.get(column, float("nan")))
            records.append(record)
    return records


def _fold_id_from_path(path: Path) -> int:
    for parent in path.parents:
        match = FOLD_PATTERN.fullmatch(parent.name)
        if match:
            return int(match.group(1))
    msg = f"Could not infer fold id from path: {path}"
    raise ValueError(msg)


def _flatten_seed_records_lopo(
    payload: dict[str, Any], run_label: str, fold_id: int
) -> list[dict[str, Any]]:
    records = _flatten_seed_records(payload, run_label)
    for record in records:
        record["fold_id"] = int(fold_id)
    return records


def _flatten_residual_records_lopo(
    payload: dict[str, Any], run_label: str, fold_id: int
) -> list[dict[str, Any]]:
    records = _flatten_residual_records(payload, run_label)
    for record in records:
        record["fold_id"] = int(fold_id)
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


def _aggregate_residual_summary(records: pd.DataFrame) -> pd.DataFrame:
    if records.empty:
        return pd.DataFrame(
            columns=[
                "model",
                "residual_spearman_mean",
                "residual_spearman_std",
                "residual_pr_auc_mean",
                "residual_pr_auc_std",
            ]
        )
    test_records = records[records["split"] == "test"].copy()
    grouped = test_records.groupby("model", as_index=False)[RESIDUAL_METRIC_COLUMNS]
    mean_df = grouped.mean().rename(
        columns={column: f"{column}_mean" for column in RESIDUAL_METRIC_COLUMNS}
    )
    std_df = grouped.std(ddof=0).rename(
        columns={column: f"{column}_std" for column in RESIDUAL_METRIC_COLUMNS}
    )
    summary = mean_df.merge(std_df, on="model", how="left")
    return summary.sort_values("residual_spearman_mean", ascending=False, ignore_index=True)


def _best_of_k_summary_by_model(seed_records: pd.DataFrame, best_of_k: int) -> pd.DataFrame:
    if best_of_k <= 0:
        return pd.DataFrame()
    val_records = seed_records[seed_records["split"] == "val"][
        ["model", "seed", "pr_auc"]
    ].copy()
    test_records = seed_records[seed_records["split"] == "test"].copy()
    rows: list[dict[str, Any]] = []
    for model, group in val_records.groupby("model", sort=False):
        ordered = (
            group.sort_values(["pr_auc", "seed"], ascending=[False, True])
            .head(best_of_k)["seed"]
            .to_list()
        )
        selected = [int(seed) for seed in ordered]
        filtered = test_records[
            (test_records["model"] == model) & (test_records["seed"].isin(selected))
        ]
        if filtered.empty:
            continue
        row: dict[str, Any] = {"model": str(model), "k": len(selected), "selected_seeds": selected}
        for column in SEED_METRIC_COLUMNS:
            row[f"{column}_mean"] = float(filtered[column].mean())
            row[f"{column}_std"] = float(filtered[column].std(ddof=0))
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("pr_auc_mean", ascending=False, ignore_index=True)


def _mean_std_by_model(records: pd.DataFrame) -> pd.DataFrame:
    grouped = records.groupby("model", as_index=False)[SEED_METRIC_COLUMNS]
    mean_df = grouped.mean().rename(
        columns={column: f"{column}_mean" for column in SEED_METRIC_COLUMNS}
    )
    std_df = grouped.std(ddof=0).rename(
        columns={column: f"{column}_std" for column in SEED_METRIC_COLUMNS}
    )
    return mean_df.merge(std_df, on="model", how="left")


def _select_best_of_k_seeds(
    seed_records: pd.DataFrame, best_of_k: int
) -> dict[tuple[int, str], list[int]]:
    if best_of_k <= 0:
        return {}
    val_records = seed_records[seed_records["split"] == "val"][
        ["fold_id", "model", "seed", "pr_auc"]
    ].copy()
    selections: dict[tuple[int, str], list[int]] = {}
    for (fold_id, model), group in val_records.groupby(["fold_id", "model"], sort=False):
        ordered = (
            group.sort_values(["pr_auc", "seed"], ascending=[False, True])
            .head(best_of_k)["seed"]
            .to_list()
        )
        selections[(int(fold_id), str(model))] = [int(seed) for seed in ordered]
    return selections


def _summarize_test_by_selected_seeds(
    seed_records: pd.DataFrame,
    selections: dict[tuple[int, str], list[int]],
) -> pd.DataFrame:
    test_records = seed_records[seed_records["split"] == "test"].copy()
    mask = []
    for _, row in test_records.iterrows():
        key = (int(row["fold_id"]), str(row["model"]))
        selected = selections.get(key)
        mask.append(selected is not None and int(row["seed"]) in selected)
    filtered = test_records[pd.Series(mask, index=test_records.index)]
    if filtered.empty:
        return pd.DataFrame(columns=["fold_id", "model", *SEED_METRIC_COLUMNS])
    grouped = filtered.groupby(["fold_id", "model"], as_index=False)[SEED_METRIC_COLUMNS]
    return grouped.mean()


def _bootstrap_mean_ci(
    values: list[float],
    *,
    n_bootstrap: int,
    random_seed: int,
) -> dict[str, Any]:
    if n_bootstrap <= 0:
        msg = "n_bootstrap must be positive"
        raise ValueError(msg)
    clean = [float(value) for value in values if not math.isnan(float(value))]
    n = len(clean)
    if n == 0:
        return {
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n_bootstrap": int(n_bootstrap),
            "n_valid_bootstrap": 0,
        }
    rng = pd.Series(clean).sample
    means: list[float] = []
    for idx in range(n_bootstrap):
        sampled = rng(n=n, replace=True, random_state=random_seed + idx).to_numpy()
        means.append(float(sampled.mean()))
    series = pd.Series(means)
    return {
        "mean": float(series.mean()),
        "ci_low": float(series.quantile(0.025)),
        "ci_high": float(series.quantile(0.975)),
        "n_bootstrap": int(n_bootstrap),
        "n_valid_bootstrap": int(series.shape[0]),
    }


def _paired_deltas_by_fold(
    fold_summary: pd.DataFrame,
    *,
    metric: str,
    comparisons: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    for model_a, model_b in comparisons:
        for fold_id, group in fold_summary.groupby("fold_id", sort=False):
            row_a = group[group["model"] == model_a]
            row_b = group[group["model"] == model_b]
            if row_a.empty or row_b.empty:
                continue
            value_a = float(row_a.iloc[0][metric])
            value_b = float(row_b.iloc[0][metric])
            deltas.append(
                {
                    "fold_id": int(fold_id),
                    "comparison": f"{model_a}_minus_{model_b}",
                    "metric": metric,
                    "delta": value_a - value_b,
                }
            )
    return deltas


def _summarize_deltas(
    delta_records: list[dict[str, Any]],
    *,
    n_bootstrap: int,
    random_seed: int,
) -> list[dict[str, Any]]:
    if not delta_records:
        return []
    df = pd.DataFrame(delta_records)
    summaries: list[dict[str, Any]] = []
    for comparison, group in df.groupby("comparison", sort=False):
        values = group["delta"].astype(float).to_list()
        mean = float(pd.Series(values).mean())
        std = float(pd.Series(values).std(ddof=0))
        positive = sum(value > 0 for value in values)
        bootstrap = _bootstrap_mean_ci(values, n_bootstrap=n_bootstrap, random_seed=random_seed)
        summaries.append(
            {
                "comparison": comparison,
                "metric": str(group.iloc[0]["metric"]),
                "mean": mean,
                "std": std,
                "n_folds": len(values),
                "positive_count": int(positive),
                "positive_fraction": float(positive / len(values)),
                "bootstrap_mean": bootstrap["mean"],
                "bootstrap_ci_low": bootstrap["ci_low"],
                "bootstrap_ci_high": bootstrap["ci_high"],
                "n_bootstrap": bootstrap["n_bootstrap"],
                "n_valid_bootstrap": bootstrap["n_valid_bootstrap"],
            }
        )
    return summaries


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


def _summary_markdown(
    seed_records: pd.DataFrame,
    summary_records: pd.DataFrame,
    best_of_k_records: pd.DataFrame,
    residual_summary_records: pd.DataFrame,
) -> str:
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

    if not best_of_k_records.empty:
        lines.extend(
            [
                "",
                f"## Best-of-K by Validation PR AUC (k={int(best_of_k_records.iloc[0]['k'])})",
                "",
                (
                    "| Model | PR AUC mean | PR AUC std | Spearman mean | Spearman std "
                    "| Top-5 mean | Top-5 std | Top-10 mean | Top-10 std | Seeds |"
                ),
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for _, row in best_of_k_records.iterrows():
            lines.append(
                f"| {row['model']} "
                f"| {row['pr_auc_mean']:.4f} | {row['pr_auc_std']:.4f} "
                f"| {row['spearman_mean_mean']:.4f} | {row['spearman_mean_std']:.4f} "
                f"| {row['top_5_recall_mean']:.4f} | {row['top_5_recall_std']:.4f} "
                f"| {row['top_10_recall_mean']:.4f} | {row['top_10_recall_std']:.4f} "
                f"| {row['selected_seeds']} |"
            )

    if not residual_summary_records.empty:
        lines.extend(
            [
                "",
                "## Beyond-Position Residual Metrics (Test)",
                "",
                "| Model | Residual Spearman mean | Residual Spearman std | Residual PR AUC mean | Residual PR AUC std |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in residual_summary_records.iterrows():
            lines.append(
                f"| {row['model']} "
                f"| {row['residual_spearman_mean']:.4f} | {row['residual_spearman_std']:.4f} "
                f"| {row['residual_pr_auc_mean']:.4f} | {row['residual_pr_auc_std']:.4f} |"
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
    best_of_k: int = 1,
) -> dict[str, Any]:
    """Aggregate per-seed metrics from one run root."""
    root = Path(run_root)
    metric_files = discover_metric_files(root)
    if not metric_files:
        msg = f"No metrics files found under {root}"
        raise FileNotFoundError(msg)

    records: list[dict[str, Any]] = []
    ci_records: list[dict[str, Any]] = []
    residual_records: list[dict[str, Any]] = []
    for metric_file in metric_files:
        payload = _load_metrics_file(metric_file)
        run_label = _run_label_from_path(metric_file)
        records.extend(_flatten_seed_records(payload, run_label))
        ci_records.extend(_flatten_ci_records(payload, run_label))
        residual_records.extend(_flatten_residual_records(payload, run_label))

    seed_df = pd.DataFrame(records)
    summary_df = _aggregate_summary(seed_df)
    best_of_k_df = _best_of_k_summary_by_model(seed_df, best_of_k)
    ci_df = pd.DataFrame(ci_records)
    ci_summary_df = _aggregate_ci_summary(ci_df)
    residual_df = pd.DataFrame(residual_records)
    residual_summary_df = _aggregate_residual_summary(residual_df)
    best_model = summary_df.iloc[0]["model"] if not summary_df.empty else None

    summary_payload = {
        "run_root": str(root),
        "metric_files": [str(path) for path in metric_files],
        "num_runs": int(seed_df["run_label"].nunique()),
        "num_seed_records": int(len(seed_df)),
        "best_model_by_mean_pr_auc": best_model,
        "seed_records": seed_df.to_dict(orient="records"),
        "summary_by_model": summary_df.to_dict(orient="records"),
        "best_of_k_summary_by_model": best_of_k_df.to_dict(orient="records"),
        "best_of_k": int(best_of_k),
        "residual_records": residual_df.to_dict(orient="records"),
        "residual_summary_by_model": residual_summary_df.to_dict(orient="records"),
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
    markdown = _summary_markdown(seed_df, summary_df, best_of_k_df, residual_summary_df)
    ci_markdown = _ci_markdown(ci_df, ci_summary_df)
    if ci_markdown:
        markdown = f"{markdown}\n{ci_markdown}".strip() + "\n"
    md_path.write_text(markdown, encoding="utf-8")

    summary_payload["output_json"] = str(json_path)
    summary_payload["output_md"] = str(md_path)
    return summary_payload


def aggregate_lopo_metrics(
    run_root: str | Path,
    *,
    output_json_path: str | Path | None = None,
    output_md_path: str | Path | None = None,
    best_of_k: int = 1,
    bootstrap_iterations: int = 1000,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    """Aggregate LOPO metrics across folds and seeds."""
    root = Path(run_root)
    metric_files = discover_lopo_metric_files(root)
    if not metric_files:
        msg = f"No LOPO metrics files found under {root}"
        raise FileNotFoundError(msg)

    records: list[dict[str, Any]] = []
    residual_records: list[dict[str, Any]] = []
    for metric_file in metric_files:
        payload = _load_metrics_file(metric_file)
        fold_id = _fold_id_from_path(metric_file)
        run_label = metric_file.parent.name
        records.extend(_flatten_seed_records_lopo(payload, run_label, fold_id))
        residual_records.extend(_flatten_residual_records_lopo(payload, run_label, fold_id))

    seed_df = pd.DataFrame(records)
    residual_df = pd.DataFrame(residual_records)
    test_df = seed_df[seed_df["split"] == "test"].copy()

    fold_mean = (
        test_df.groupby(["fold_id", "model"], as_index=False)[SEED_METRIC_COLUMNS]
        .mean()
        .assign(agg_type="mean_seeds", k=int(best_of_k))
    )
    seed_counts = (
        test_df.groupby(["fold_id", "model"], as_index=False)["seed"]
        .nunique()
        .rename(columns={"seed": "seed_count"})
    )
    fold_mean = fold_mean.merge(seed_counts, on=["fold_id", "model"], how="left")

    selections = _select_best_of_k_seeds(seed_df, best_of_k)
    selection_payload: dict[str, dict[str, list[int]]] = {}
    for (fold_id, model), seeds in selections.items():
        selection_payload.setdefault(str(fold_id), {})[str(model)] = list(seeds)
    fold_best = _summarize_test_by_selected_seeds(seed_df, selections)
    if not fold_best.empty:
        fold_best = fold_best.assign(agg_type="best_of_k", k=int(best_of_k))
        fold_best = fold_best.merge(seed_counts, on=["fold_id", "model"], how="left")

    fold_summary = pd.concat([fold_mean, fold_best], ignore_index=True)

    residual_fold_summary = pd.DataFrame()
    residual_fold_agg = pd.DataFrame()
    if not residual_df.empty:
        residual_test_df = residual_df[residual_df["split"] == "test"].copy()
        residual_mean = (
            residual_test_df.groupby(["fold_id", "model"], as_index=False)[RESIDUAL_METRIC_COLUMNS]
            .mean()
            .assign(agg_type="mean_seeds", k=int(best_of_k))
        )
        residual_seed_counts = (
            residual_test_df.groupby(["fold_id", "model"], as_index=False)["seed"]
            .nunique()
            .rename(columns={"seed": "seed_count"})
        )
        residual_mean = residual_mean.merge(
            residual_seed_counts, on=["fold_id", "model"], how="left"
        )

        residual_best = pd.DataFrame()
        if selections:
            mask = []
            for _, row in residual_test_df.iterrows():
                key = (int(row["fold_id"]), str(row["model"]))
                selected = selections.get(key)
                mask.append(selected is not None and int(row["seed"]) in selected)
            residual_filtered = residual_test_df[pd.Series(mask, index=residual_test_df.index)]
            if not residual_filtered.empty:
                residual_best = (
                    residual_filtered.groupby(["fold_id", "model"], as_index=False)[
                        RESIDUAL_METRIC_COLUMNS
                    ]
                    .mean()
                    .assign(agg_type="best_of_k", k=int(best_of_k))
                )
                residual_best = residual_best.merge(
                    residual_seed_counts, on=["fold_id", "model"], how="left"
                )

        residual_fold_summary = pd.concat([residual_mean, residual_best], ignore_index=True)

        residual_grouped = residual_fold_summary.groupby(
            ["agg_type", "model"], as_index=False
        )[RESIDUAL_METRIC_COLUMNS]
        residual_mean_df = residual_grouped.mean().rename(
            columns={column: f"{column}_mean" for column in RESIDUAL_METRIC_COLUMNS}
        )
        residual_std_df = residual_grouped.std(ddof=0).rename(
            columns={column: f"{column}_std" for column in RESIDUAL_METRIC_COLUMNS}
        )
        residual_fold_agg = residual_mean_df.merge(
            residual_std_df, on=["agg_type", "model"], how="left"
        )

    fold_grouped = fold_summary.groupby(["agg_type", "model"], as_index=False)[SEED_METRIC_COLUMNS]
    fold_mean_df = fold_grouped.mean().rename(
        columns={column: f"{column}_mean" for column in SEED_METRIC_COLUMNS}
    )
    fold_std_df = fold_grouped.std(ddof=0).rename(
        columns={column: f"{column}_std" for column in SEED_METRIC_COLUMNS}
    )
    fold_agg = fold_mean_df.merge(fold_std_df, on=["agg_type", "model"], how="left")

    delta_records: list[dict[str, Any]] = []
    delta_summaries: list[dict[str, Any]] = []
    for agg_type in fold_summary["agg_type"].unique():
        subset = fold_summary[fold_summary["agg_type"] == agg_type].copy()
        deltas = _paired_deltas_by_fold(
            subset,
            metric="pr_auc",
            comparisons=LOPO_COMPARISONS,
        )
        for record in deltas:
            record["agg_type"] = agg_type
            record["k"] = int(best_of_k)
        delta_records.extend(deltas)
        summaries = _summarize_deltas(
            deltas,
            n_bootstrap=bootstrap_iterations,
            random_seed=bootstrap_seed,
        )
        for summary in summaries:
            summary["agg_type"] = agg_type
            summary["k"] = int(best_of_k)
        delta_summaries.extend(summaries)

    summary_payload = {
        "run_root": str(root),
        "metric_files": [str(path) for path in metric_files],
        "num_folds": int(seed_df["fold_id"].nunique()),
        "num_seed_records": int(len(seed_df)),
        "best_of_k": int(best_of_k),
        "seed_records": seed_df.to_dict(orient="records"),
        "fold_summary_by_model": fold_summary.to_dict(orient="records"),
        "fold_aggregate_by_model": fold_agg.to_dict(orient="records"),
        "residual_fold_summary_by_model": residual_fold_summary.to_dict(orient="records"),
        "residual_fold_aggregate_by_model": residual_fold_agg.to_dict(orient="records"),
        "paired_delta_records": delta_records,
        "paired_delta_summary": delta_summaries,
        "best_of_k_selections": selection_payload,
    }

    if output_json_path is None:
        output_json_path = root / "aggregate_lopo_metrics.json"
    if output_md_path is None:
        output_md_path = root / "aggregate_lopo_metrics.md"

    json_path = Path(output_json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2, sort_keys=True)

    md_path = Path(output_md_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_lines = [
        "# LOPO Aggregated Probe Metrics",
        "",
        f"- Folds: {summary_payload['num_folds']}",
        f"- Seed records: {summary_payload['num_seed_records']}",
        "",
        "## Paired Delta Summary (PR AUC, folds as unit)",
        "",
        "| Agg | Comparison | Mean | Std | Positive | Folds | CI low | CI high |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in delta_summaries:
        md_lines.append(
            f"| {row['agg_type']} | {row['comparison']} "
            f"| {row['mean']:.4f} | {row['std']:.4f} "
            f"| {row['positive_count']} | {row['n_folds']} "
            f"| {row['bootstrap_ci_low']:.4f} | {row['bootstrap_ci_high']:.4f} |"
        )
    md_lines.append("")

    if not residual_fold_agg.empty:
        md_lines.extend(
            [
                "## Beyond-Position Residual Metrics (Test)",
                "",
                "| Agg | Model | Residual Spearman mean | Residual Spearman std | Residual PR AUC mean | Residual PR AUC std |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for _, row in residual_fold_agg.iterrows():
            md_lines.append(
                f"| {row['agg_type']} | {row['model']} "
                f"| {row['residual_spearman_mean']:.4f} | {row['residual_spearman_std']:.4f} "
                f"| {row['residual_pr_auc_mean']:.4f} | {row['residual_pr_auc_std']:.4f} |"
            )
        md_lines.append("")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    summary_payload["output_json"] = str(json_path)
    summary_payload["output_md"] = str(md_path)
    return summary_payload
