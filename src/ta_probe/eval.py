"""Standalone evaluation helpers for saved prediction files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ta_probe.metrics import evaluate_frame


def evaluate_prediction_file(
    *,
    predictions_path: str | Path,
    output_path: str | Path,
    score_column: str,
    k_values: list[int],
) -> dict[str, Any]:
    """Evaluate one score column from a saved predictions parquet."""
    frame = pd.read_parquet(predictions_path)
    metrics = evaluate_frame(
        frame,
        score_col=score_column,
        k_values=k_values,
    )

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    return metrics
