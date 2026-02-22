"""Configuration models and loaders for thought-anchor probe experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator


class DatasetConfig(BaseModel):
    """Dataset paths and sampling controls."""

    repo_id: str = "uzaymacar/math-rollouts"
    model_dir: str = "deepseek-r1-distill-llama-8b"
    temp_dir: str = "temperature_0.6_top_p_0.95"
    split_dir: str = "correct_base_solution"
    forced_split_dir: str = "correct_base_solution_forced_answer"
    num_problems: int = 50
    seed: int = 0


class LabelConfig(BaseModel):
    """Label construction options."""

    counterfactual_field: str = "counterfactual_importance_accuracy"
    anchor_percentile: float = 90.0
    drop_last_chunk: bool = True

    @field_validator("anchor_percentile")
    @classmethod
    def validate_percentile(cls, value: float) -> float:
        if not 0.0 < value < 100.0:
            msg = "anchor_percentile must be between 0 and 100"
            raise ValueError(msg)
        return value


class ActivationConfig(BaseModel):
    """Activation extraction settings."""

    model_name_or_path: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    layer_mode: Literal["mid", "index"] = "mid"
    layer_index: int | None = None
    pooling: Literal["mean"] = "mean"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    device: Literal["cpu", "mps", "cuda", "auto"] = "auto"
    batch_size: int = 1


class TrainingConfig(BaseModel):
    """Probe training options."""

    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    random_seed: int = 0
    k_values: list[int] = Field(default_factory=lambda: [5, 10])
    mlp_hidden_dim: int = 100
    mlp_max_iter: int = 300

    @field_validator("train_fraction", "val_fraction", "test_fraction")
    @classmethod
    def validate_fraction(cls, value: float) -> float:
        if value <= 0 or value >= 1:
            msg = "split fractions must be between 0 and 1"
            raise ValueError(msg)
        return value


class PathsConfig(BaseModel):
    """Output and cache paths."""

    problem_ids_json: str = "data/problem_ids.json"
    splits_json: str = "data/splits.json"
    embeddings_memmap: str = "artifacts/sentence_embeddings.dat"
    embeddings_shape_json: str = "artifacts/sentence_embeddings_shape.json"
    metadata_parquet: str = "artifacts/sentence_metadata.parquet"
    metrics_json: str = "artifacts/metrics.json"
    predictions_parquet: str = "artifacts/predictions.parquet"
    report_md: str = "artifacts/report.md"


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    activations: ActivationConfig = Field(default_factory=ActivationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)


def load_config(path: str | Path) -> ExperimentConfig:
    """Load experiment config from YAML."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return ExperimentConfig.model_validate(payload)


def ensure_parent_dirs(config: ExperimentConfig) -> None:
    """Create parent directories for configured files."""
    outputs = [
        config.paths.problem_ids_json,
        config.paths.splits_json,
        config.paths.embeddings_memmap,
        config.paths.embeddings_shape_json,
        config.paths.metadata_parquet,
        config.paths.metrics_json,
        config.paths.predictions_parquet,
        config.paths.report_md,
    ]
    for output in outputs:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
