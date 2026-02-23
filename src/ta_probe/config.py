"""Configuration models and loaders for thought-anchor probe experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

DEFAULT_TOKEN_EMBEDDINGS_MEMMAP = "artifacts/token_embeddings.dat"


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
    target_mode: Literal["anchor_binary", "importance_abs", "importance_signed"] = (
        "anchor_binary"
    )

    @field_validator("anchor_percentile")
    @classmethod
    def validate_percentile(cls, value: float) -> float:
        if not 0.0 < value < 100.0:
            msg = "anchor_percentile must be between 0 and 100"
            raise ValueError(msg)
        return value


class VerticalAttentionConfig(BaseModel):
    """Sentence-level vertical-attention score extraction settings."""

    mode: Literal["off", "light", "full"] = "off"
    depth_control: bool = True
    light_last_n_tokens: int = 1
    full_max_seq_len: int = 1024

    @field_validator("light_last_n_tokens")
    @classmethod
    def validate_light_last_n_tokens(cls, value: int) -> int:
        if value <= 0:
            msg = "light_last_n_tokens must be positive"
            raise ValueError(msg)
        return value

    @field_validator("full_max_seq_len")
    @classmethod
    def validate_full_max_seq_len(cls, value: int) -> int:
        if value <= 0:
            msg = "full_max_seq_len must be positive"
            raise ValueError(msg)
        return value


class ActivationConfig(BaseModel):
    """Activation extraction settings."""

    model_name_or_path: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    layer_mode: Literal["mid", "index"] = "mid"
    layer_index: int | None = None
    pooling: Literal["mean", "tokens"] = "mean"
    dtype: Literal["float16", "bfloat16", "float32"] = "float16"
    storage_dtype: Literal["float16", "float32"] = "float32"
    device: Literal["cpu", "mps", "cuda", "auto"] = "auto"
    batch_size: int = 1
    vertical_attention: VerticalAttentionConfig = Field(default_factory=VerticalAttentionConfig)


class TrainingConfig(BaseModel):
    """Probe training options."""

    train_fraction: float = 0.7
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    random_seed: int = 0
    k_values: list[int] = Field(default_factory=lambda: [5, 10])
    mlp_hidden_dim: int = 100
    mlp_max_iter: int = 300
    bootstrap_iterations: int = 1000
    bootstrap_seed: int | None = None
    position_bins: int = 5
    best_of_k: int = 1
    residualize_against: Literal["none", "position", "position_plus_text"] = "none"
    token_probe_heads: int = 4
    token_probe_mlp_width: int = 128
    token_probe_mlp_depth: int = 1
    token_probe_batch_size: int = 32
    token_probe_max_epochs: int = 50
    token_probe_patience: int = 5
    token_probe_learning_rate: float = 1e-3
    token_probe_weight_decay: float = 0.0
    token_probe_continuous_loss: Literal["mse", "huber"] = "huber"
    token_probe_device: Literal["cpu", "mps", "cuda", "auto"] = "auto"

    @field_validator("train_fraction", "val_fraction", "test_fraction")
    @classmethod
    def validate_fraction(cls, value: float) -> float:
        if value <= 0 or value >= 1:
            msg = "split fractions must be between 0 and 1"
            raise ValueError(msg)
        return value

    @field_validator("bootstrap_iterations")
    @classmethod
    def validate_bootstrap_iterations(cls, value: int) -> int:
        if value <= 0:
            msg = "bootstrap_iterations must be positive"
            raise ValueError(msg)
        return value

    @field_validator("position_bins")
    @classmethod
    def validate_position_bins(cls, value: int) -> int:
        if value < 0:
            msg = "position_bins must be zero or positive"
            raise ValueError(msg)
        return value

    @field_validator("best_of_k")
    @classmethod
    def validate_best_of_k(cls, value: int) -> int:
        if value <= 0:
            msg = "best_of_k must be positive"
            raise ValueError(msg)
        return value

    @field_validator(
        "token_probe_heads",
        "token_probe_mlp_width",
        "token_probe_mlp_depth",
        "token_probe_batch_size",
        "token_probe_max_epochs",
        "token_probe_patience",
    )
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value <= 0:
            msg = "token probe integer hyperparameters must be positive"
            raise ValueError(msg)
        return value

    @field_validator("token_probe_learning_rate")
    @classmethod
    def validate_learning_rate(cls, value: float) -> float:
        if value <= 0.0:
            msg = "token_probe_learning_rate must be positive"
            raise ValueError(msg)
        return value

    @field_validator("token_probe_weight_decay")
    @classmethod
    def validate_weight_decay(cls, value: float) -> float:
        if value < 0.0:
            msg = "token_probe_weight_decay must be zero or positive"
            raise ValueError(msg)
        return value


class SplitConfig(BaseModel):
    """Split strategy configuration."""

    strategy: Literal["single_split", "lopo_cv"] = "single_split"


class PathsConfig(BaseModel):
    """Output and cache paths."""

    problem_ids_json: str = "data/problem_ids.json"
    splits_json: str = "data/splits.json"
    embeddings_memmap: str = "artifacts/sentence_embeddings.dat"
    token_embeddings_memmap: str = DEFAULT_TOKEN_EMBEDDINGS_MEMMAP
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
    split: SplitConfig = Field(default_factory=SplitConfig)
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
        config.paths.token_embeddings_memmap,
        config.paths.embeddings_shape_json,
        config.paths.metadata_parquet,
        config.paths.metrics_json,
        config.paths.predictions_parquet,
        config.paths.report_md,
    ]
    for output in outputs:
        Path(output).parent.mkdir(parents=True, exist_ok=True)


def resolve_embeddings_memmap_path(config: ExperimentConfig) -> str:
    """Select the active embeddings memmap path for the configured pooling mode."""
    if config.activations.pooling == "tokens":
        configured = Path(config.paths.token_embeddings_memmap)
        if str(configured) == DEFAULT_TOKEN_EMBEDDINGS_MEMMAP:
            return str(Path(config.paths.embeddings_memmap).with_name("token_embeddings.dat"))
        return config.paths.token_embeddings_memmap
    return config.paths.embeddings_memmap
